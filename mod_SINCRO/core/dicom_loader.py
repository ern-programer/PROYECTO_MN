"""
SINCRO - core.dicom_loader
===========================

Loader inteligente y auto-descriptivo de estudios Gated SPECT cardíacos.

Absorbe todo el conocimiento de formatos descubierto durante la Fase 0:
- Detección de gated cardíaco (vía NumberOfTimeSlots / vectores DICOM / RR).
- Desempaquetado de MONTAGE (cortes concatenados horizontalmente: Cols = N × Rows),
  tal como exporta Xeleris/MyoVation el Short Axis gated (ej: 418×22 = 19 cortes de 22×22).
- Separación del frame SUMADO (perfusión) de los gates reales.
- Reshape 4D vía vectores (SliceVector/TimeSlotVector) — AGNÓSTICO al orden.
- Auto-QC: FFT de la curva global → verifica que "el corazón late" (1er armónico dominante).

El resultado siempre es un cubo 4D normalizado: ``(n_gates, n_slices, H, W)``.

Uso CLI:
    python -m core.dicom_loader "ruta/al/REST_IRNCG_SA001_DS.dcm"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


# --- Tags DICOM relevantes (grupo 0054 = Nuclear Medicine) ---
TAG_NUMBER_OF_FRAMES = (0x0028, 0x0008)
TAG_NUMBER_OF_TIME_SLICES = (0x0054, 0x0101)
TAG_NUMBER_OF_TIME_SLOTS = (0x0054, 0x0071)
TAG_NUMBER_OF_SLICES = (0x0054, 0x0081)
TAG_TIME_SLOT_VECTOR = (0x0054, 0x0070)
TAG_SLICE_VECTOR = (0x0054, 0x0080)
TAG_ANGULAR_VIEW_VECTOR = (0x0054, 0x0090)
TAG_RR_INTERVAL_VECTOR = (0x0054, 0x0060)
TAG_PHASE_INFO_SEQUENCE = (0x0054, 0x0032)

# Umbral de QC: fracción mínima de energía en el 1er armónico para considerar
# que la curva de actividad es un latido cardíaco coherente.
QC_FIRST_HARMONIC_MIN = 0.40


@dataclass
class GatedStudy:
    """Estudio gated cardíaco normalizado y listo para el análisis de fase."""

    cube: np.ndarray                      # (n_gates, n_slices, H, W) float64
    n_gates: int
    n_slices: int
    rows: int
    cols: int
    pixel_spacing: tuple[float, float] | None
    source_path: str
    z_spacing_mm: float | None = None
    slice_thickness_mm: float | None = None
    spacing_between_slices_mm: float | None = None
    image_type: list[str] = field(default_factory=list)
    series_description: str = ""
    study_description: str = ""
    patient_name: str = ""
    patient_id: str = ""
    patient_sex: str = ""
    patient_birth_date: str = ""
    study_date: str = ""
    study_time: str = ""
    accession_number: str = ""
    study_instance_uid: str = ""
    was_montage: bool = False
    had_summed_frame: bool = False
    reconstructed: bool = True            # False si venía crudo (proyecciones)
    qc_first_harmonic: float = 0.0
    qc_passed: bool = False
    gating_info: dict = field(default_factory=dict)  # Datos ECG adquisición (3 derivaciones / gating)
    # --- Geometría de adquisición (solo modo crudo/proyecciones) ---
    patient_position: str = ""       # HFS/FFS/HFP/FFP
    start_angle: float | None = None
    angular_step: float | None = None
    rotation_direction: str = ""     # CW / CC
    scan_arc: float | None = None
    angles_deg: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        patient_txt = self.patient_name.strip() or "N/D"
        patient_id_txt = self.patient_id.strip() or "N/D"
        study_date_txt = self.study_date.strip() or "N/D"
        lines = [
            f"Paciente       : {patient_txt}  (ID: {patient_id_txt})",
            f"Fecha estudio  : {study_date_txt}",
            f"Estudio        : {self.study_description} | {self.series_description}",
            f"Cubo 4D        : {self.cube.shape}  (gates × slices × H × W)",
            f"Gates          : {self.n_gates}",
            f"Slices         : {self.n_slices}",
            f"Matriz corte   : {self.rows}×{self.cols // self.n_slices if self.was_montage else self.cols}",
            f"Montage        : {'sí (desempaquetado)' if self.was_montage else 'no'}",
            f"Frame sumado   : {'sí (descartado)' if self.had_summed_frame else 'no'}",
            f"Reconstruido   : {'sí' if self.reconstructed else 'NO (crudo, requiere recon)'}",
            f"Auto-QC latido : 1er armónico={self.qc_first_harmonic:.3f}  "
            f"→ {'OK (late)' if self.qc_passed else 'REVISAR (posible gating error / reshape)'}",
        ]
        if self.pixel_spacing is not None:
            lines.append(f"Pixel spacing  : {self.pixel_spacing[0]:.3f} x {self.pixel_spacing[1]:.3f} mm")
        if self.z_spacing_mm is not None:
            lines.append(f"Espesor Z      : {self.z_spacing_mm:.3f} mm")
        if self.notes:
            lines.append("Notas          :")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


class LoaderError(Exception):
    pass


def _get(ds, tag, default=None):
    return ds[tag].value if tag in ds else default


def _extract_gating_info(ds) -> dict:
    """
    Extrae datos de gating / ECG de adquisición (monitor de 3 derivaciones)
    embebidos en el DICOM SPECT. Devuelve dict con lo que encuentre.
    """
    import numpy as _np
    info: dict = {}

    # HeartRate (0018,1088) — FC en lpm
    hr = _get(ds, (0x0018, 0x1088), None)
    if hr is not None:
        try:
            info["heart_rate"] = int(hr)
        except Exception:
            pass

    # RRIntervalVector (0054,0060) — intervalos RR en ms por gate
    rr_vec = _get(ds, TAG_RR_INTERVAL_VECTOR, None)
    if rr_vec is not None:
        try:
            rr = [float(v) for v in rr_vec]
            # Filtrar placeholders inválidos: RR fisiológico está ~250-2000 ms.
            # GE/Xeleris a veces exporta 1.0 ms como placeholder → descartar.
            rr = [v for v in rr if 250.0 <= v <= 2000.0]
            if rr:
                info["rr_intervals_ms"] = rr
                info["rr_mean_ms"] = float(_np.mean(rr))
                info["rr_cv_pct"] = float(100.0 * _np.std(rr) / _np.mean(rr)) if _np.mean(rr) > 0 else 0.0
                # FC estimada desde RR medio si no vino HeartRate
                if "heart_rate" not in info and info["rr_mean_ms"] > 0:
                    info["heart_rate_est"] = int(round(60000.0 / info["rr_mean_ms"]))
            else:
                # RR vector inválido/placeholder → no reportar FC estimada ni RR
                info["rr_placeholder"] = True
        except Exception:
            pass

    # Trigger source/type
    trig = _get(ds, (0x0018, 0x1061), None)
    if trig:
        info["trigger_source"] = str(trig)

    # Trigger window (%) (0018,1094)
    tw = _get(ds, (0x0018, 0x1094), None)
    if tw is not None:
        try:
            info["trigger_window_pct"] = float(tw)
        except Exception:
            pass

    # Heart rate limits (ventana de aceptación)
    hr_lo = _get(ds, (0x0018, 0x1081), None)
    hr_hi = _get(ds, (0x0018, 0x1082), None)
    if hr_lo is not None:
        info["hr_low_limit"] = hr_lo
    if hr_hi is not None:
        info["hr_high_limit"] = hr_hi

    # Beat rejection / PVC rejection
    beat_rej = _get(ds, (0x0018, 0x1080), None)
    if beat_rej is not None:
        info["beat_rejection"] = str(beat_rej)
    pvc = _get(ds, (0x0018, 0x1085), None)
    if pvc is not None:
        info["pvc_rejection"] = str(pvc)
    skip = _get(ds, (0x0018, 0x1086), None)
    if skip is not None:
        info["skip_beats"] = skip

    # Flags privados GE comunes (número de latidos, ventanas RR)
    n_beats = _get(ds, (0x0011, 0x100C), None)
    if n_beats is not None:
        try:
            info["n_beats"] = int(n_beats)
        except Exception:
            pass

    # Frame time / duración por gate (ms) — útil para ponderar desgatillado si gates no uniformes
    ft = _get(ds, (0x0018, 0x1063), None)  # FrameTime (a veces por frame en NM)
    if ft is not None:
        try:
            info["frame_time_ms"] = float(ft)
        except Exception:
            pass
    # TimeSlotTime (0054,0030) — duración del time slot en segundos
    tst = _get(ds, (0x0054, 0x0030), None)
    if tst is not None:
        try:
            info["time_slot_time_s"] = float(tst)
        except Exception:
            pass
    # TriggerTime (0018,1060) — tiempo desde trigger nominal
    tt = _get(ds, (0x0018, 0x1060), None)
    if tt is not None:
        try:
            info["trigger_time_ms"] = float(tt)
        except Exception:
            pass
    # NumberOfTimeSlots / TimeSlices (gates)
    nslots = _get(ds, (0x0054, 0x0071), None) or _get(ds, (0x0054, 0x0101), None)
    if nslots is not None:
        try:
            info["n_time_slots"] = int(nslots)
        except Exception:
            pass
    # Framing type / cardiac synchronization technique
    framing = _get(ds, (0x0018, 0x1064), None)
    if framing:
        info["cardiac_framing_type"] = str(framing)
    sync_tech = _get(ds, (0x0018, 0x9037), None)
    if sync_tech:
        info["cardiac_sync_technique"] = str(sync_tech)

    # QC simple: variabilidad RR alta sugiere FA/extrasístoles
    if info.get("rr_cv_pct") is not None:
        info["rr_variability_flag"] = "alta" if info["rr_cv_pct"] > 15.0 else "normal"

    return info


def _is_raw_projections(ds) -> bool:
    """Proyecciones angulares crudas = tiene AngularViewVector y NO tiene SliceVector."""
    has_angular = TAG_ANGULAR_VIEW_VECTOR in ds
    has_slices = TAG_SLICE_VECTOR in ds
    itype = list(_get(ds, (0x0008, 0x0008), []))
    is_gated_tomo = "GATED TOMO" in " ".join(str(x) for x in itype)
    return has_angular and not has_slices and is_gated_tomo


def _detect_montage(rows: int, cols: int) -> Optional[int]:
    """Si cols es múltiplo entero de rows → montage de (cols/rows) cortes. Devuelve n_cortes o None."""
    if rows > 0 and cols % rows == 0 and cols // rows > 1:
        return cols // rows
    return None


def _first_harmonic_fraction(cube: np.ndarray) -> float:
    """Fracción de energía en el 1er armónico de la curva de actividad global por gate."""
    t = cube.shape[0]
    if t < 3:
        return 0.0
    curve = cube.reshape(t, -1).sum(axis=1).astype(np.float64)
    curve -= curve.mean()
    power = np.abs(np.fft.fft(curve)) ** 2
    denom = power[1: t // 2 + 1].sum()
    return float(power[1] / denom) if denom > 0 else 0.0


def _to_float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)) and value:
            return float(value[0])
        return float(value)
    except Exception:
        return None


def _to_clean_str(value) -> str:
    if value is None:
        return ""
    try:
        text = str(value)
    except Exception:
        return ""
    return text.strip()


def _unpack_montage(frames: np.ndarray, n_slices: int) -> np.ndarray:
    """(F, rows, cols) con cols = n_slices*rows  →  (F, n_slices, rows, cell)."""
    F, rows, cols = frames.shape
    cell = cols // n_slices
    # (F, rows, n_slices, cell) → (F, n_slices, rows, cell)
    return frames.reshape(F, rows, n_slices, cell).transpose(0, 2, 1, 3)


def _separate_summed_frame(frames_by_time: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Detecta si el primer frame es el SUMADO (perfusión) en vez de un gate.
    Heurística: el sumado tiene ~(n_gates) veces la suma de un gate individual.
    Devuelve (frames_sin_sumado, had_summed).
    """
    n = frames_by_time.shape[0]
    if n < 3:
        return frames_by_time, False
    sums = frames_by_time.reshape(n, -1).sum(axis=1)
    rest_mean = sums[1:].mean()
    if rest_mean > 0 and sums[0] > 2.5 * rest_mean:
        return frames_by_time[1:], True
    return frames_by_time, False


def load(path: str, verbose: bool = False) -> GatedStudy:
    """Carga un DICOM gated cardíaco y devuelve un GatedStudy normalizado (cubo 4D)."""
    if pydicom is None:
        raise LoaderError("pydicom no está instalado (pip install pydicom).")

    ds = pydicom.dcmread(path, force=True)
    notes: list[str] = []

    modality = str(_get(ds, (0x0008, 0x0060), "?"))
    if modality not in ("NM", "PT", "ST"):
        notes.append(f"Modality={modality} (esperado NM/PT). Continuando igual.")

    itype = [str(x) for x in _get(ds, (0x0008, 0x0008), [])]
    series_desc = str(_get(ds, (0x0008, 0x103E), "") or "")
    study_desc = str(_get(ds, (0x0008, 0x1030), "") or "")
    patient_name = _to_clean_str(_get(ds, (0x0010, 0x0010), ""))
    patient_id = _to_clean_str(_get(ds, (0x0010, 0x0020), ""))
    patient_sex = _to_clean_str(_get(ds, (0x0010, 0x0040), ""))
    patient_birth_date = _to_clean_str(_get(ds, (0x0010, 0x0030), ""))
    study_date = _to_clean_str(_get(ds, (0x0008, 0x0020), ""))
    study_time = _to_clean_str(_get(ds, (0x0008, 0x0030), ""))
    accession_number = _to_clean_str(_get(ds, (0x0008, 0x0050), ""))
    study_instance_uid = _to_clean_str(_get(ds, (0x0020, 0x000D), ""))
    rows = int(_get(ds, (0x0028, 0x0010), 0) or 0)
    cols = int(_get(ds, (0x0028, 0x0011), 0) or 0)
    px = _get(ds, (0x0028, 0x0030), None)
    pixel_spacing = (float(px[0]), float(px[1])) if px else None
    slice_thickness_mm = _to_float_or_none(_get(ds, (0x0018, 0x0050), None))
    spacing_between_slices_mm = _to_float_or_none(_get(ds, (0x0018, 0x0088), None))
    z_spacing_mm = spacing_between_slices_mm if spacing_between_slices_mm else slice_thickness_mm

    # --- Caso 1: proyecciones crudas → cargar como RawGatedProjections (cine/QC/gating/motion) ---
    if _is_raw_projections(ds):
        from core.raw_projections import load_raw_projections
        raw = load_raw_projections(path)
        # Devolver un GatedStudy en modo "raw": cube = proyecciones (gates, angles, H, W),
        # reconstructed=False para que la UI lo trate como crudo, no como SA reconstruido.
        notes.append(
            f"CRUDO gated: {raw.n_gates} gates × {raw.n_angles} ángulos × {raw.rows}×{raw.cols}. "
            "Modo proyecciones: cine QC, gating y motion correction disponibles; "
            "análisis de fase requiere reconstrucción."
        )
        study = GatedStudy(
            cube=raw.projections,
            n_gates=raw.n_gates,
            n_slices=raw.n_angles,   # en modo raw, "slices" = ángulos de proyección
            rows=raw.rows,
            cols=raw.cols,
            pixel_spacing=pixel_spacing,
            z_spacing_mm=z_spacing_mm,
            slice_thickness_mm=slice_thickness_mm,
            spacing_between_slices_mm=spacing_between_slices_mm,
            source_path=path,
            image_type=itype,
            series_description=raw.series_description,
            study_description=raw.study_description,
            patient_name=raw.patient_name,
            patient_id=raw.patient_id,
            patient_sex=patient_sex,
            patient_birth_date=patient_birth_date,
            study_date=study_date,
            study_time=study_time,
            accession_number=accession_number,
            study_instance_uid=study_instance_uid,
            was_montage=False,
            had_summed_frame=False,
            reconstructed=False,
            qc_first_harmonic=0.0,
            qc_passed=False,
            gating_info=raw.gating_info,
            patient_position=getattr(raw, "patient_position", ""),
            start_angle=getattr(raw, "start_angle", None),
            angular_step=getattr(raw, "angular_step", None),
            rotation_direction=getattr(raw, "rotation_direction", ""),
            scan_arc=getattr(raw, "scan_arc", None),
            angles_deg=getattr(raw, "angles_deg", None),
            notes=notes + raw.notes,
        )
        return study

    arr = ds.pixel_array.astype(np.float64)
    if arr.ndim == 2:
        raise LoaderError("Imagen 2D única: no es un estudio gated multiframe.")

    n_frames = arr.shape[0]

    # --- Detectar dimensión temporal (gates) ---
    n_time = _get(ds, TAG_NUMBER_OF_TIME_SLOTS, None) or _get(ds, TAG_NUMBER_OF_TIME_SLICES, None)
    n_slices_tag = _get(ds, TAG_NUMBER_OF_SLICES, None)
    slice_vec = _get(ds, TAG_SLICE_VECTOR, None)
    time_vec = _get(ds, TAG_TIME_SLOT_VECTOR, None)

    # --- Caso 2: montage (Short Axis gated de Xeleris/MyoVation) ---
    # frames = gates(+sumado); cada frame es N cortes concatenados (cols = N*rows).
    montage_slices = _detect_montage(rows, cols)
    if montage_slices and (n_time is None or int(n_time or 0) <= 1):
        # Los "frames" son time-bins (+ posible sumado); cada uno es un montage de cortes.
        frames_time, had_summed = _separate_summed_frame(arr)
        cube = _unpack_montage(frames_time, montage_slices)  # (gates, slices, rows, cell)
        was_montage = True
        n_gates = cube.shape[0]
        n_slices = cube.shape[1]
        notes.append(
            f"Montage detectado: cols={cols} = {montage_slices}×{rows} → {montage_slices} cortes/frame."
        )
        if had_summed:
            notes.append("Frame 0 detectado como SUMADO (perfusión) → descartado del análisis de fase.")

    # --- Caso 3: reshape 4D vía vectores DICOM (agnóstico al orden) ---
    elif slice_vec is not None and time_vec is not None \
            and len(slice_vec) == n_frames and len(time_vec) == n_frames:
        sv = list(slice_vec)
        tv = list(time_vec)
        n_slices = len(set(sv))
        n_gates = len(set(tv))
        H, W = arr.shape[1], arr.shape[2]
        cube = np.zeros((n_gates, n_slices, H, W), dtype=np.float64)
        for f in range(n_frames):
            cube[tv[f] - 1, sv[f] - 1] = arr[f]
        was_montage = False
        had_summed = False
        notes.append(f"Reshape 4D vía vectores DICOM: {n_gates} gates × {n_slices} slices.")

    # --- Caso 4: cubo ya 4D o inferible por producto ---
    elif n_time and n_slices_tag and int(n_time) * int(n_slices_tag) == n_frames:
        n_gates, n_slices = int(n_time), int(n_slices_tag)
        H, W = arr.shape[1], arr.shape[2]
        cube = arr.reshape(n_gates, n_slices, H, W)
        was_montage = False
        had_summed = False
        notes.append(f"Reshape por producto: {n_gates} gates × {n_slices} slices.")

    else:
        raise LoaderError(
            f"No pude determinar la estructura gated. frames={n_frames}, time={n_time}, "
            f"slices={n_slices_tag}, montage={montage_slices}. "
            "Revisar el estudio con las herramientas de reconocimiento."
        )

    # --- Auto-QC: ¿el corazón late? ---
    frac = _first_harmonic_fraction(cube)
    qc_passed = frac >= QC_FIRST_HARMONIC_MIN
    if not qc_passed:
        notes.append(
            f"QC latido bajo (1er armónico={frac:.3f} < {QC_FIRST_HARMONIC_MIN}). "
            "Posible gating error, reshape incorrecto o estudio no cardíaco."
        )

    gating_info = _extract_gating_info(ds)

    study = GatedStudy(
        cube=cube,
        n_gates=n_gates,
        n_slices=n_slices,
        rows=rows,
        cols=cols,
        pixel_spacing=pixel_spacing,
        z_spacing_mm=z_spacing_mm,
        slice_thickness_mm=slice_thickness_mm,
        spacing_between_slices_mm=spacing_between_slices_mm,
        source_path=path,
        image_type=itype,
        series_description=series_desc,
        study_description=study_desc,
        patient_name=patient_name,
        patient_id=patient_id,
        patient_sex=patient_sex,
        patient_birth_date=patient_birth_date,
        study_date=study_date,
        study_time=study_time,
        accession_number=accession_number,
        study_instance_uid=study_instance_uid,
        was_montage=was_montage,
        had_summed_frame=locals().get("had_summed", False),
        reconstructed=True,
        qc_first_harmonic=frac,
        qc_passed=qc_passed,
        gating_info=gating_info,
        notes=notes,
    )

    if verbose:
        print(study.summary())
    return study


def _cli() -> int:
    from core.console_utf8 import enable_utf8
    enable_utf8()
    if len(sys.argv) < 2:
        print("Uso: python -m core.dicom_loader <archivo.dcm>")
        return 1
    try:
        study = load(sys.argv[1], verbose=True)
    except LoaderError as e:
        print(f"[LoaderError] {e}")
        return 2
    return 0 if study.qc_passed else 3


if __name__ == "__main__":
    raise SystemExit(_cli())
