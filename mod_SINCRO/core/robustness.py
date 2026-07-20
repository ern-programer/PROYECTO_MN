"""Herramientas de robustez para métricas de fase GammaSync."""
from __future__ import annotations

import numpy as np

from core.metrics import calculate_phase_metrics
from core.phase_analysis import phase_analysis


def calculate_segmental_metrics(phase_by_seg: dict[int, float]) -> dict:
    """Calcula métricas sobre las medias circulares de los segmentos AHA disponibles."""
    values = np.array([float(phase_by_seg[k]) for k in sorted(phase_by_seg) if np.isfinite(float(phase_by_seg[k]))], dtype=np.float64)
    if values.size < 3:
        return {"available": False, "n_segments": int(values.size)}
    out = calculate_phase_metrics(values, hist_bins=72)
    out["available"] = True
    out["n_segments"] = int(values.size)
    out["mode"] = "segmental_aha"
    return out


def bootstrap_phase_metrics(
    phases_deg: np.ndarray,
    *,
    n_iter: int = 500,
    sample_frac: float = 0.80,
    seed: int = 20260720,
) -> dict:
    """Estima estabilidad de PSD/BW por bootstrap de voxels válidos."""
    phases = np.asarray(phases_deg, dtype=np.float64)
    phases = phases[np.isfinite(phases)]
    if phases.size < 20 or n_iter <= 0:
        return {"available": False, "n_voxels": int(phases.size), "n_iter": int(max(0, n_iter))}

    rng = np.random.default_rng(int(seed))
    sample_size = max(8, int(round(float(sample_frac) * phases.size)))
    sample_size = min(sample_size, int(phases.size))
    psd = np.empty((int(n_iter),), dtype=np.float64)
    bw = np.empty((int(n_iter),), dtype=np.float64)
    entropy_pct = np.empty((int(n_iter),), dtype=np.float64)
    for idx in range(int(n_iter)):
        sample = phases[rng.integers(0, phases.size, size=sample_size)]
        metrics = calculate_phase_metrics(sample)
        psd[idx] = float(metrics.get("phase_sd", np.nan))
        bw[idx] = float(metrics.get("bandwidth", np.nan))
        entropy_pct[idx] = float(metrics.get("entropy_normalized_pct", np.nan))

    def stats(values: np.ndarray) -> dict:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {"mean": None, "ci95_low": None, "ci95_high": None, "sd": None}
        return {
            "mean": round(float(np.mean(values)), 2),
            "ci95_low": round(float(np.percentile(values, 2.5)), 2),
            "ci95_high": round(float(np.percentile(values, 97.5)), 2),
            "sd": round(float(np.std(values, ddof=1)), 2) if values.size > 1 else 0.0,
        }

    return {
        "available": True,
        "n_voxels": int(phases.size),
        "sample_size": int(sample_size),
        "n_iter": int(n_iter),
        "sample_frac": round(float(sample_frac), 3),
        "phase_sd": stats(psd),
        "bandwidth": stats(bw),
        "entropy_normalized_pct": stats(entropy_pct),
    }


def _mask_from_roi_arrays(
    centers: np.ndarray,
    inner: np.ndarray,
    outer: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    n_slices, height, width = shape
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[:height, :width]
    for slice_index in range(n_slices):
        cy = float(centers[slice_index, 0]) if centers.shape[0] > slice_index else np.nan
        cx = float(centers[slice_index, 1]) if centers.shape[0] > slice_index else np.nan
        r_inner = float(inner[slice_index]) if inner.shape[0] > slice_index else np.nan
        r_outer = float(outer[slice_index]) if outer.shape[0] > slice_index else np.nan
        if not np.isfinite(cy) or not np.isfinite(cx) or not np.isfinite(r_outer) or r_outer <= 0.0:
            continue
        has_inner = np.isfinite(r_inner) and r_inner > 0.0 and r_inner < r_outer
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask[slice_index] = ((dist >= r_inner) & (dist <= r_outer)) if has_inner else (dist <= r_outer)
    return mask


def _variant_arrays(seg, variant: str, delta_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(getattr(seg, "center_per_slice", np.empty((0, 2))), dtype=np.float64).copy()
    inner = np.asarray(getattr(seg, "inner_radius", np.empty((0,))), dtype=np.float64).copy()
    outer = np.asarray(getattr(seg, "outer_radius", np.empty((0,))), dtype=np.float64).copy()
    if centers.ndim != 2 or centers.shape[1] != 2:
        return centers, inner, outer
    if variant == "inner_roi":
        inner = np.where(np.isfinite(inner), inner + delta_px, inner)
        outer = np.where(np.isfinite(outer), np.maximum(outer - delta_px, 0.5), outer)
    elif variant == "outer_roi":
        inner = np.where(np.isfinite(inner), np.maximum(inner - delta_px, 0.0), inner)
        outer = np.where(np.isfinite(outer), outer + delta_px, outer)
    elif variant == "shift_y_minus":
        centers[:, 0] -= delta_px
    elif variant == "shift_y_plus":
        centers[:, 0] += delta_px
    elif variant == "shift_x_minus":
        centers[:, 1] -= delta_px
    elif variant == "shift_x_plus":
        centers[:, 1] += delta_px
    valid = np.isfinite(inner) & np.isfinite(outer)
    inner[valid] = np.minimum(inner[valid], np.maximum(outer[valid] - 0.5, 0.0))
    return centers, inner, outer


def roi_sensitivity_analysis(
    cube: np.ndarray,
    seg,
    *,
    harmonics: int = 1,
    amplitude_threshold_frac: float = 0.40,
    normalize_reference: bool = False,
    delta_px: float = 1.0,
) -> dict:
    """Recalcula PSD/BW con pequeñas perturbaciones de ROI para cuantificar dependencia de máscara."""
    arr = np.asarray(cube, dtype=np.float64)
    if arr.ndim != 4 or seg is None:
        return {"available": False, "reason": "input inválido"}
    mask_shape = tuple(np.asarray(seg.mask).shape)
    variants = [
        ("current", "ROI actual"),
        ("inner_roi", "ROI más interna"),
        ("outer_roi", "ROI más externa"),
        ("shift_y_minus", "ROI desplazada -1px Y"),
        ("shift_y_plus", "ROI desplazada +1px Y"),
        ("shift_x_minus", "ROI desplazada -1px X"),
        ("shift_x_plus", "ROI desplazada +1px X"),
    ]
    rows: list[dict] = []
    for variant, label in variants:
        try:
            if variant == "current":
                mask = np.asarray(seg.mask, dtype=bool)
            else:
                centers, inner, outer = _variant_arrays(seg, variant, float(delta_px))
                mask = _mask_from_roi_arrays(centers, inner, outer, mask_shape)
            if int(np.count_nonzero(mask)) < 10:
                continue
            result = phase_analysis(
                arr,
                mask,
                harmonics=int(harmonics),
                amplitude_threshold_frac=float(amplitude_threshold_frac),
                normalize_reference=bool(normalize_reference),
            )
            metrics = calculate_phase_metrics(result.phases_deg)
            rows.append({
                "variant": variant,
                "label": label,
                "mask_voxels": int(np.count_nonzero(mask)),
                "phase_voxels": int(result.n_voxels_kept),
                "phase_sd": float(metrics.get("phase_sd", np.nan)),
                "bandwidth": float(metrics.get("bandwidth", np.nan)),
                "entropy_normalized_pct": float(metrics.get("entropy_normalized_pct", np.nan)),
                "technical_classification": str(metrics.get("technical_classification", metrics.get("classification", "N/D"))),
            })
        except Exception as exc:
            rows.append({"variant": variant, "label": label, "error": str(exc)})

    valid = [row for row in rows if "error" not in row and np.isfinite(float(row.get("phase_sd", np.nan)))]
    if len(valid) < 2:
        return {"available": False, "delta_px": float(delta_px), "variants": rows, "reason": "variantes insuficientes"}
    psd_values = np.array([float(row["phase_sd"]) for row in valid], dtype=np.float64)
    bw_values = np.array([float(row["bandwidth"]) for row in valid], dtype=np.float64)
    current = next((row for row in valid if row.get("variant") == "current"), valid[0])
    max_psd_delta = float(np.max(np.abs(psd_values - float(current["phase_sd"]))))
    max_bw_delta = float(np.max(np.abs(bw_values - float(current["bandwidth"]))))
    warn = bool(max_psd_delta >= 10.0 or max_bw_delta >= 30.0)
    return {
        "available": True,
        "delta_px": float(delta_px),
        "variants": rows,
        "phase_sd_min": round(float(np.min(psd_values)), 2),
        "phase_sd_max": round(float(np.max(psd_values)), 2),
        "bandwidth_min": round(float(np.min(bw_values)), 2),
        "bandwidth_max": round(float(np.max(bw_values)), 2),
        "max_phase_sd_delta": round(max_psd_delta, 2),
        "max_bandwidth_delta": round(max_bw_delta, 2),
        "warn": warn,
    }
