"""ProcessingManager - Lógica de procesamiento DICOM→métricas."""
from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import numpy as np

from core import dicom_loader
from core.aha_segments import map_to_17_segments, phase_by_segment, territory_analysis
from core.metrics import calculate_phase_metrics
from core.phase_analysis import phase_analysis
from core.robustness import (
    bootstrap_phase_metrics,
    calculate_segmental_metrics,
    roi_sensitivity_analysis,
)
from core.segmentation import segment_myocardium


class ProcessingManager:
    """Gestiona el pipeline completo de procesamiento."""

    def __init__(self):
        self.study = None
        self.seg = None
        self.phase_result = None
        self.phase_result_raw = None
        self.metrics = None
        self.metrics_raw = None
        self.phase_qc = None
        self.aha = None
        self.phase_by_seg = None
        self.territory = None
        self._cache_study_sig = ""
        self._cache_seg_sig = ""
        self._cache_phase_sig = ""

    def load_study(self, path: str) -> dict[str, Any]:
        """Carga un estudio DICOM y retorna metadatos."""
        self.study = dicom_loader.load(path, verbose=False)
        self._cache_study_sig = self._build_study_signature(path)
        self._cache_seg_sig = ""
        self._cache_phase_sig = ""
        return {
            "patient_name": getattr(self.study, "patient_name", ""),
            "patient_id": getattr(self.study, "patient_id", ""),
            "study_date": getattr(self.study, "study_date", ""),
            "dimensions": self.study.cube.shape,
        }

    def _build_study_signature(self, path: str) -> str:
        """Construye una firma única del estudio para cache."""
        import hashlib
        stat = os.stat(path)
        payload = f"{path}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.md5(payload.encode()).hexdigest()

    def segment(
        self,
        method: str = "auto",
        threshold_frac: float = 0.35,
        smooth_sigma: float = 1.0,
        manual_rois: dict | None = None,
        cube_override: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Ejecuta segmentación del miocardio."""
        if self.study is None:
            raise ValueError("No hay estudio cargado")

        cube = cube_override if cube_override is not None else self.study.cube
        seg_sig = self._build_seg_signature(method, threshold_frac, smooth_sigma, manual_rois)

        if seg_sig != self._cache_seg_sig or self.seg is None:
            self.seg = segment_myocardium(
                cube,
                method=method,
                threshold_frac=threshold_frac,
                smooth_sigma=smooth_sigma,
                manual_rois=manual_rois,
            )
            self._cache_seg_sig = seg_sig
            self._cache_phase_sig = ""

        return {
            "method": self.seg.method,
            "n_voxels": self.seg.n_voxels,
            "n_slices": self.seg.mask.shape[0],
        }

    def _build_seg_signature(self, method, threshold, sigma, manual_rois) -> str:
        """Firma para cache de segmentación."""
        import hashlib
        rois_txt = str(sorted(manual_rois.items())) if manual_rois else ""
        payload = f"{method}|{threshold}|{sigma}|{rois_txt}"
        return hashlib.md5(payload.encode()).hexdigest()

    def analyze_phase(
        self,
        harmonics: int = 1,
        clinical_amp_filter: float = 0.40,
        raw_amp_filter: float = 0.10,
        normalize_reference: bool = False,
    ) -> dict[str, Any]:
        """Ejecuta análisis de fase crudo y clínico."""
        if self.study is None or self.seg is None:
            raise ValueError("No hay estudio/segmentación disponible")

        phase_sig = self._build_phase_signature(harmonics, clinical_amp_filter, raw_amp_filter, normalize_reference)

        if phase_sig != self._cache_phase_sig or self.phase_result is None:
            cube = np.asarray(self.study.cube, dtype=np.float64)

            # Fase cruda (QC)
            self.phase_result_raw = phase_analysis(
                cube,
                self.seg.mask,
                harmonics=harmonics,
                amplitude_threshold_frac=raw_amp_filter,
                normalize_reference=normalize_reference,
            )

            # Fase clínica
            self.phase_result = phase_analysis(
                cube,
                self.seg.mask,
                harmonics=harmonics,
                amplitude_threshold_frac=clinical_amp_filter,
                normalize_reference=normalize_reference,
            )

            # Métricas
            self.metrics_raw = self._annotate_phase_metrics(
                calculate_phase_metrics(self.phase_result_raw.phases_deg),
                self.phase_result_raw,
                raw_amp_filter,
                "crudo ROI",
            )
            self.metrics = self._annotate_phase_metrics(
                calculate_phase_metrics(self.phase_result.phases_deg),
                self.phase_result,
                clinical_amp_filter,
                "clínico robusto",
            )

            # QC
            self.phase_qc = self._build_phase_qc(
                self.phase_result_raw,
                self.phase_result,
                self.metrics_raw,
                self.metrics,
            )

            # AHA segments
            self.aha = map_to_17_segments(self.seg)
            self.phase_by_seg = phase_by_segment(self.phase_result.phase_map, self.aha)
            self.territory = territory_analysis(self.phase_by_seg)

            # Robustez
            self._attach_robustness_metrics()

            self._cache_phase_sig = phase_sig

        return {
            "phase_sd": self.metrics.get("phase_sd"),
            "bandwidth": self.metrics.get("bandwidth"),
            "technical_classification": self.metrics.get("technical_classification"),
            "n_voxels_kept": self.phase_result.n_voxels_kept,
        }

    def _build_phase_signature(self, harmonics, clinical_amp, raw_amp, normalize) -> str:
        """Firma para cache de fase."""
        import hashlib
        payload = f"{harmonics}|{clinical_amp}|{raw_amp}|{normalize}|{self._cache_seg_sig}"
        return hashlib.md5(payload.encode()).hexdigest()

    def _annotate_phase_metrics(self, metrics: dict, phase_result, amp_filter: float, label: str) -> dict:
        """Agrega metadatos a las métricas."""
        out = dict(metrics or {})
        out["amp_filter"] = round(float(amp_filter), 2)
        out["amp_label"] = str(label)
        out["n_voxels_kept"] = int(getattr(phase_result, "n_voxels_kept", out.get("n_voxels", 0)))
        out["n_voxels_total"] = int(getattr(phase_result, "n_voxels_total", out.get("n_voxels", 0)))
        return out

    def _build_phase_qc(self, raw_result, clinical_result, raw_metrics: dict, clinical_metrics: dict) -> dict:
        """Construye QC de fase."""
        if raw_result is None or clinical_result is None:
            return {}
        raw_phases = np.asarray(getattr(raw_result, "phases_deg", []), dtype=np.float64)
        raw_amps = np.asarray(getattr(raw_result, "amplitudes", []), dtype=np.float64)
        if raw_phases.size == 0 or raw_amps.size != raw_phases.size:
            return {}

        clinical_filter = float(getattr(clinical_result, "amplitude_threshold_frac", 0.40))
        amp_max = float(np.nanmax(raw_amps)) if raw_amps.size else 0.0
        low_amp = raw_amps < (clinical_filter * amp_max) if amp_max > 0.0 else np.zeros(raw_amps.shape, dtype=bool)
        clinical_mean = float(clinical_metrics.get("mean_phase", raw_metrics.get("mean_phase", 0.0)))
        centered = (raw_phases - clinical_mean + 180.0) % 360.0 - 180.0
        late_tail = np.abs(centered) > 120.0
        low_tail = low_amp & late_tail
        low_tail_pct = float(np.mean(low_tail) * 100.0) if raw_phases.size else 0.0
        low_tail_n = int(np.count_nonzero(low_tail))
        class_changed = str(raw_metrics.get("classification", "")) != str(clinical_metrics.get("classification", ""))
        warn = bool(low_tail_pct >= 5.0 or class_changed)

        return {
            "raw_filter": float(0.10),
            "clinical_filter": round(clinical_filter, 2),
            "raw_classification": str(raw_metrics.get("classification", "N/D")),
            "clinical_classification": str(clinical_metrics.get("classification", "N/D")),
            "class_changed": class_changed,
            "low_confidence_tail_pct": round(low_tail_pct, 1),
            "low_confidence_tail_n": low_tail_n,
            "raw_voxels": int(getattr(raw_result, "n_voxels_kept", raw_phases.size)),
            "clinical_voxels": int(getattr(clinical_result, "n_voxels_kept", 0)),
            "total_voxels": int(getattr(raw_result, "n_voxels_total", raw_phases.size)),
            "warn": warn,
        }

    def _attach_robustness_metrics(self):
        """Agrega métricas de robustez."""
        if self.metrics is None or self.phase_result is None or self.phase_by_seg is None or self.study is None or self.seg is None:
            return

        self.metrics["mode"] = "voxel"
        self.metrics["segmental_aha"] = calculate_segmental_metrics(self.phase_by_seg)
        self.metrics["bootstrap"] = bootstrap_phase_metrics(
            self.phase_result.phases_deg,
            n_iter=500,
            sample_frac=0.80,
            seed=20260720,
        )
        self.metrics["roi_sensitivity"] = roi_sensitivity_analysis(
            self.study.cube,
            self.seg,
            harmonics=int(self.phase_result.harmonics),
            amplitude_threshold_frac=float(self.phase_result.amplitude_threshold_frac),
            normalize_reference=bool(self.phase_result.normalize_reference) if hasattr(self.phase_result, "normalize_reference") else False,
            delta_px=1.0,
        )

    def get_metrics_summary(self) -> dict[str, Any]:
        """Retorna resumen de métricas para UI."""
        if self.metrics is None:
            return {}
        return {
            "phase_sd": self.metrics.get("phase_sd"),
            "bandwidth": self.metrics.get("bandwidth"),
            "entropy_normalized_pct": self.metrics.get("entropy_normalized_pct"),
            "technical_classification": self.metrics.get("technical_classification"),
            "n_voxels_kept": self.metrics.get("n_voxels_kept"),
            "n_voxels_total": self.metrics.get("n_voxels_total"),
        }

    def get_robustness_summary(self) -> dict[str, Any]:
        """Retorna resumen de robustez para UI."""
        if self.metrics is None:
            return {}
        return {
            "segmental_aha": self.metrics.get("segmental_aha"),
            "bootstrap": self.metrics.get("bootstrap"),
            "roi_sensitivity": self.metrics.get("roi_sensitivity"),
        }

    def get_phase_qc_summary(self) -> dict[str, Any]:
        """Retorna resumen de QC para UI."""
        return self.phase_qc or {}

    def get_segmentation_summary(self) -> dict[str, Any]:
        """Retorna resumen de segmentación para UI."""
        if self.seg is None:
            return {}
        return {
            "method": self.seg.method,
            "n_voxels": self.seg.n_voxels,
            "n_slices": self.seg.mask.shape[0],
        }

    def clear_cache(self):
        """Limpia cache de procesamiento."""
        self._cache_seg_sig = ""
        self._cache_phase_sig = ""
