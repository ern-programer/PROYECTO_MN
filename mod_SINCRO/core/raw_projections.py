"""Carga y QC de proyecciones SPECT crudas (raw gated).

Carga proyecciones crudas GATED TOMO (AngularViewVector + TimeSlotVector),
las organiza como (n_gates, n_angles, H, W), genera sinogramas H/V para QC
visual estilo Odyssey, extrae datos de gating completos y calcula el tracking
del centro de masa vs ángulo (base de la motion correction).

NO reconstruye: eso queda para el pipeline OPEN (FBP/Butterworth).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RawGatedProjections:
    """Proyecciones crudas gated organizadas."""
    projections: np.ndarray          # (n_gates, n_angles, H, W)
    n_gates: int
    n_angles: int
    rows: int
    cols: int
    angles_deg: np.ndarray           # ángulo de cada proyección (si se puede inferir)
    gating_info: dict = field(default_factory=dict)
    source_path: str = ""
    patient_name: str = ""
    patient_id: str = ""
    study_description: str = ""
    series_description: str = ""
    notes: list[str] = field(default_factory=list)


def _get(ds, tag, default=None):
    return ds[tag].value if tag in ds else default


def load_raw_projections(path: str) -> RawGatedProjections:
    """
    Carga proyecciones crudas gated desde DICOM.

    Acepta archivos GATED TOMO con AngularViewVector (proyecciones angulares).
    Organiza los frames como (n_gates, n_angles, H, W).
    """
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom requerido") from exc

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Se esperaba multiframe 3D (frames,H,W); recibió {arr.shape}")

    n_frames, H, W = arr.shape

    ang_vec = _get(ds, (0x0054, 0x0090), None)   # AngularViewVector
    time_vec = _get(ds, (0x0054, 0x0070), None)  # TimeSlotVector
    n_time = _get(ds, (0x0054, 0x0071), None) or _get(ds, (0x0054, 0x0101), None)

    itype = " ".join(str(x) for x in _get(ds, (0x0008, 0x0008), []))
    if "GATED TOMO" not in itype and ang_vec is None:
        raise ValueError("No parecen proyecciones crudas gated (sin GATED TOMO ni AngularViewVector).")

    notes: list[str] = []

    # Organizar por gates × ángulos
    if ang_vec is not None and time_vec is not None and len(ang_vec) == n_frames and len(time_vec) == n_frames:
        av = [int(v) for v in ang_vec]
        tv = [int(v) for v in time_vec]
        n_angles = len(set(av))
        n_gates = len(set(tv))
        projections = np.zeros((n_gates, n_angles, H, W), dtype=np.float64)
        for f in range(n_frames):
            projections[tv[f] - 1, av[f] - 1] = arr[f]
        notes.append(f"Organizado por vectores DICOM: {n_gates} gates × {n_angles} ángulos.")
    elif n_time and int(n_time) > 0 and n_frames % int(n_time) == 0:
        n_gates = int(n_time)
        n_angles = n_frames // n_gates
        projections = arr.reshape(n_gates, n_angles, H, W)
        notes.append(f"Reshape por producto: {n_gates} gates × {n_angles} ángulos (orden asumido gate-major).")
    else:
        raise ValueError(
            f"No se pudo organizar el crudo: frames={n_frames}, "
            f"AngularViewVector={'sí' if ang_vec is not None else 'no'}, n_time={n_time}."
        )

    # Ángulos en grados (inferir distribución uniforme 0-360 si no hay metadata)
    start_angle = _get(ds, (0x0054, 0x0020), None)  # StartAngle (a veces)
    angular_step = _get(ds, (0x0018, 0x1140), None)  # RotationDirection/AngularStep varía
    if start_angle is not None and angular_step is not None:
        angles_deg = (float(start_angle) + np.arange(n_angles) * float(angular_step)) % 360.0
    else:
        angles_deg = np.linspace(0.0, 360.0, n_angles, endpoint=False)
        notes.append("Ángulos inferidos como distribución uniforme 0-360° (sin metadata angular explícita).")

    # Gating completo del crudo (los crudos suelen traer más que los reconstruidos)
    from core.dicom_loader import _extract_gating_info
    gating_info = _extract_gating_info(ds)

    return RawGatedProjections(
        projections=projections,
        n_gates=int(n_gates),
        n_angles=int(n_angles),
        rows=int(H),
        cols=int(W),
        angles_deg=angles_deg,
        gating_info=gating_info,
        source_path=path,
        patient_name=str(_get(ds, (0x0010, 0x0010), "") or ""),
        patient_id=str(_get(ds, (0x0010, 0x0020), "") or ""),
        study_description=str(_get(ds, (0x0008, 0x1030), "") or ""),
        series_description=str(_get(ds, (0x0008, 0x103E), "") or ""),
        notes=notes,
    )


def ungate_projections(projections: np.ndarray) -> np.ndarray:
    """
    Desgatilla el crudo gated: suma todos los gates por proyección → UngGat.

    El UngGat tiene ~n_gates× más cuentas que cada gate individual, por lo que
    es la base de trabajo para motion correction, reconstrucción y cortes
    (flujo Odyssey: desgatillar primero, trabajar con alta estadística, luego
    aplicar los mismos parámetros geométricos al gated).

    Parameters
    ----------
    projections : ndarray (n_gates, n_angles, H, W)

    Returns
    -------
    ndarray (n_angles, H, W)
        Proyecciones desgatilladas (UngGat / suma de gates).
    """
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibió {proj.shape}")
    return proj.sum(axis=0)


def apply_shifts_to_projections(projections: np.ndarray, shifts_y: np.ndarray, shifts_x: np.ndarray | None = None) -> np.ndarray:
    """
    Aplica shifts de motion correction a las proyecciones (Y-only por defecto, como Odyssey).

    El mismo shift se aplica a la proyección completa (todos los gates comparten
    la misma posición angular → el paciente se mueve igual en esa proyección).

    Parameters
    ----------
    projections : ndarray (n_gates, n_angles, H, W) o (n_angles, H, W)
    shifts_y : ndarray (n_angles,) — shift vertical en px por ángulo.
    shifts_x : ndarray (n_angles,), optional — shift horizontal en px por ángulo.

    Returns
    -------
    ndarray — proyecciones corregidas con la misma forma que la entrada.
    """
    from scipy.ndimage import shift as _ndi_shift

    proj = np.asarray(projections, dtype=np.float64)
    shifts_y = np.asarray(shifts_y, dtype=np.float64)
    if shifts_x is None:
        shifts_x = np.zeros_like(shifts_y)
    else:
        shifts_x = np.asarray(shifts_x, dtype=np.float64)

    if proj.ndim == 3:  # (angles, H, W) — UngGat
        out = np.empty_like(proj)
        for a in range(proj.shape[0]):
            out[a] = _ndi_shift(proj[a], shift=(shifts_y[a], shifts_x[a]), order=1, mode="nearest")
        return out
    if proj.ndim == 4:  # (gates, angles, H, W) — gated
        out = np.empty_like(proj)
        for a in range(proj.shape[1]):
            out[:, a] = _ndi_shift(proj[:, a], shift=(0.0, shifts_y[a], shifts_x[a]), order=1, mode="nearest")
        return out
    raise ValueError(f"projections debe ser 3D o 4D; recibió {proj.shape}")


def build_sinograms(projections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera sinogramas horizontal y vertical para QC visual (estilo Odyssey).

    Parameters
    ----------
    projections : ndarray (n_gates, n_angles, H, W)

    Returns
    -------
    (sino_h, sino_v) : tuple de ndarray
        sino_h: (n_angles, H) — suma sobre gates y columnas (perfil vertical vs ángulo).
        sino_v: (n_angles, W) — suma sobre gates y filas (perfil horizontal vs ángulo).
        El movimiento del paciente aparece como discontinuidades/ondulaciones.
    """
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibió {proj.shape}")
    # Sumar sobre gates para máxima estadística
    summed = proj.sum(axis=0)          # (n_angles, H, W)
    sino_h = summed.sum(axis=2)        # (n_angles, H) — perfil vertical
    sino_v = summed.sum(axis=1)        # (n_angles, W) — perfil horizontal
    return sino_h, sino_v


def _tracking_from_com(projections: np.ndarray, axis: str, threshold_frac: float) -> dict:
    """Tracking simple por centro de masa sobre máscara por threshold."""
    from scipy.ndimage import center_of_mass as _com

    proj = np.asarray(projections, dtype=np.float64)
    summed = proj.sum(axis=0)
    n_angles = summed.shape[0]
    com_series = np.full((n_angles,), np.nan, dtype=np.float64)
    threshold_frac = float(threshold_frac)
    threshold_frac = min(max(threshold_frac, 0.01), 0.90)

    for a in range(n_angles):
        img = summed[a]
        if img.max() <= 0:
            continue
        mask = img > (threshold_frac * img.max())
        if mask.sum() < 4:
            continue
        cy, cx = _com(mask)
        com_series[a] = cy if axis == "y" else cx

    return {"axis": axis, "com_series": com_series, "method": "com", "threshold_frac": threshold_frac}


def _tracking_from_threshold(projections: np.ndarray, axis: str, threshold_frac: float) -> dict:
    """Tracking por bounding box de la máscara (más parecido a flujo Odyssey threshold + centro del objeto)."""
    proj = np.asarray(projections, dtype=np.float64)
    summed = proj.sum(axis=0)
    n_angles = summed.shape[0]
    com_series = np.full((n_angles,), np.nan, dtype=np.float64)
    threshold_frac = float(threshold_frac)
    threshold_frac = min(max(threshold_frac, 0.01), 0.90)

    for a in range(n_angles):
        img = summed[a]
        if img.max() <= 0:
            continue
        mask = img > (threshold_frac * img.max())
        ys, xs = np.where(mask)
        if ys.size < 4:
            continue
        cy = 0.5 * (float(ys.min()) + float(ys.max()))
        cx = 0.5 * (float(xs.min()) + float(xs.max()))
        com_series[a] = cy if axis == "y" else cx

    return {"axis": axis, "com_series": com_series, "method": "threshold", "threshold_frac": threshold_frac}


def _finalize_tracking(tracking: dict) -> dict:
    """Calcula outliers, shifts y flag de movimiento para un tracking dado."""
    com_series = np.asarray(tracking.get("com_series", []), dtype=np.float64)
    n_angles = com_series.shape[0]
    outliers = np.zeros((n_angles,), dtype=bool)
    shifts = np.zeros((n_angles,), dtype=np.float64)
    valid = np.isfinite(com_series)
    if valid.sum() >= 3:
        med = float(np.median(com_series[valid]))
        mad = float(np.median(np.abs(com_series[valid] - med)))
        sigma = 1.4826 * mad if mad > 0 else 1.0
        outliers = valid & (np.abs(com_series - med) > 2.5 * sigma)
        shifts = np.where(valid, med - com_series, 0.0)

    max_shift = float(np.nanmax(np.abs(shifts))) if valid.any() else 0.0
    tracking.update({
        "outliers": outliers,
        "suggested_shifts_px": shifts,
        "n_outliers": int(outliers.sum()),
        "max_shift_px": round(max_shift, 2),
        "motion_suspected": bool(max_shift > 1.5 or outliers.sum() >= 2),
    })
    return tracking


def center_of_mass_tracking(projections: np.ndarray, axis: str = "y", threshold_frac: float = 0.20) -> dict:
    """
    Tracking del centro de masa del corazón vs ángulo (base de motion correction).
    """
    return _finalize_tracking(_tracking_from_com(projections, axis=axis, threshold_frac=threshold_frac))


def motion_correct_projections(
    projections: np.ndarray,
    axis: str = "y",
    threshold_frac: float = 0.20,
    method: str = "com",
    manual_shifts_y: np.ndarray | None = None,
    manual_shifts_x: np.ndarray | None = None,
) -> dict:
    """
    Motion correction de proyecciones SPECT gated.

    Methods:
      - com: centro de masa sobre máscara por threshold.
      - threshold: centro del bounding box de la máscara por threshold (más robusto a hígado/ruido en algunos casos).
    """
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibió {proj.shape}")

    method = str(method or "com").strip().lower()
    if method not in ("com", "threshold"):
        raise ValueError("method debe ser 'com' o 'threshold'")

    axes_to_correct = ["y", "x"] if axis == "xy" else [axis]
    tracking = {}
    shifts_y = np.zeros((proj.shape[1],), dtype=np.float64)
    shifts_x = np.zeros((proj.shape[1],), dtype=np.float64)

    for ax in axes_to_correct:
        if method == "threshold":
            trk = _finalize_tracking(_tracking_from_threshold(proj, axis=ax, threshold_frac=threshold_frac))
        else:
            trk = _finalize_tracking(_tracking_from_com(proj, axis=ax, threshold_frac=threshold_frac))
        tracking[ax] = trk
        if ax == "y":
            shifts_y = np.asarray(trk["suggested_shifts_px"], dtype=np.float64)
        else:
            shifts_x = np.asarray(trk["suggested_shifts_px"], dtype=np.float64)

    if manual_shifts_y is not None:
        shifts_y = np.asarray(manual_shifts_y, dtype=np.float64)
    if manual_shifts_x is not None:
        shifts_x = np.asarray(manual_shifts_x, dtype=np.float64)

    corrected = apply_shifts_to_projections(proj, shifts_y, shifts_x)
    max_shift = float(max(
        float(np.abs(shifts_y).max()) if shifts_y.size else 0.0,
        float(np.abs(shifts_x).max()) if shifts_x.size else 0.0,
    ))
    motion_detected = any(tracking[ax]["motion_suspected"] for ax in axes_to_correct)

    return {
        "corrected": corrected,
        "tracking_y": tracking.get("y"),
        "tracking_x": tracking.get("x"),
        "applied_shifts_y": shifts_y,
        "applied_shifts_x": shifts_x,
        "motion_detected": bool(motion_detected),
        "max_shift_px": round(max_shift, 2),
        "axis_corrected": axis,
        "method": method,
        "threshold_frac": float(threshold_frac),
        "manual_override": bool(manual_shifts_y is not None or manual_shifts_x is not None),
    }
