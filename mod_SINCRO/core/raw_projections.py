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


def center_of_mass_tracking(projections: np.ndarray, axis: str = "y") -> dict:
    """
    Tracking del centro de masa del corazón vs ángulo (base de motion correction).

    Parameters
    ----------
    projections : ndarray (n_gates, n_angles, H, W)
    axis : 'y' (vertical, default, la más común) o 'x' (horizontal)

    Returns
    -------
    dict con serie de COM vs ángulo, outliers y sugerencia de shifts.
    """
    from scipy.ndimage import center_of_mass as _com

    proj = np.asarray(projections, dtype=np.float64)
    summed = proj.sum(axis=0)  # (n_angles, H, W)
    n_angles = summed.shape[0]
    com_series = np.full((n_angles,), np.nan, dtype=np.float64)

    for a in range(n_angles):
        img = summed[a]
        if img.max() <= 0:
            continue
        # Threshold al 20% del máximo para aislar el corazón del fondo
        mask = img > (0.20 * img.max())
        if mask.sum() < 4:
            continue
        cy, cx = _com(mask)
        com_series[a] = cy if axis == "y" else cx

    # Detectar outliers por desviación robusta (mediana ± k*MAD)
    valid = np.isfinite(com_series)
    outliers = np.zeros((n_angles,), dtype=bool)
    shifts = np.zeros((n_angles,), dtype=np.float64)
    if valid.sum() >= 3:
        med = float(np.median(com_series[valid]))
        mad = float(np.median(np.abs(com_series[valid] - med)))
        sigma = 1.4826 * mad if mad > 0 else 1.0
        outliers = valid & (np.abs(com_series - med) > 2.5 * sigma)
        # Shift sugerido = mediana - valor (para alinear todo a la mediana)
        shifts = np.where(valid, med - com_series, 0.0)

    max_shift = float(np.nanmax(np.abs(shifts))) if valid.any() else 0.0
    return {
        "axis": axis,
        "com_series": com_series,
        "outliers": outliers,
        "suggested_shifts_px": shifts,
        "n_outliers": int(outliers.sum()),
        "max_shift_px": round(max_shift, 2),
        "motion_suspected": bool(max_shift > 1.5 or outliers.sum() >= 2),
    }
