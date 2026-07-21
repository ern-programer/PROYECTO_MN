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


def reconstruct_transaxial_slices(
    projections: np.ndarray,
    angles_deg: np.ndarray | None = None,
    filter_name: str = "ramp",
) -> np.ndarray:
    """
    Reconstrucción transaxial rápida (FBP) del bruto para generar cortes
    transaxiales anatómicos (como los "cortes rápidos" de Odyssey para motion
    correction / pick de órgano).

    Las proyecciones gated son (n_gates, n_angles, H, W). Se suman los gates
    (UngGat) y se reconstruye cada slice transaxial (cada fila H) por FBP con
    iradon. El resultado es un volumen (H_slices, W, W) donde cada slice es una
    vista transaxial anatómica (corazón separable del hígado).

    Parameters
    ----------
    projections : ndarray (n_gates, n_angles, H, W)
        Proyecciones crudas gated.
    angles_deg : ndarray (n_angles,), optional
        Ángulos en grados de cada proyección. Si es None, distribución uniforme 0-180.
    filter_name : str
        Filtro FBP de iradon ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann').

    Returns
    -------
    ndarray (H_slices, W, W)
        Volumen transaxial reconstruido (cortes anatómicos).
    """
    from skimage.transform import iradon

    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibió {proj.shape}")
    ung = proj.sum(axis=0)  # (n_angles, H, W) — UngGat
    n_angles, H, W = ung.shape

    if angles_deg is None:
        # SPECT típico: 180° o 360°. iradon espera los ángulos de las proyecciones.
        angles_deg = np.linspace(0.0, 360.0, n_angles, endpoint=False)
    angles_deg = np.asarray(angles_deg, dtype=np.float64)

    # Reconstruir cada slice transaxial (fila H) con FBP.
    # iradon espera sinograma como (detector, ángulos) → transposeamos (H, n_angles).
    volume = np.zeros((H, W, W), dtype=np.float64)
    for s in range(H):
        sino = ung[:, s, :].T  # (W, n_angles) — sinograma del slice s
        rec = iradon(sino, theta=angles_deg, filter_name=filter_name, output_size=W)
        volume[s] = rec
    return volume


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


def _tracking_from_com(
    projections: np.ndarray,
    axis: str,
    threshold_frac: float,
    seed: tuple[float, float] | None = None,
) -> dict:
    """Tracking simple por centro de masa sobre máscara por threshold.

    Si hay seed (pick del usuario en el corazón), usa SOLO la componente de ese
    órgano (evita hígado); si no, usa la máscara completa por threshold.
    """
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
        # Máscara del órgano (selección de componente disponible en todos los métodos).
        mask = _organ_mask(img, threshold_frac, seed=seed, use_organ_selection=True)
        if mask.sum() < 4:
            continue
        cy, cx = _com(mask)
        com_series[a] = cy if axis == "y" else cx

    return {"axis": axis, "com_series": com_series, "method": "com", "threshold_frac": threshold_frac, "seed": seed}


def _select_organ_component(
    mask: np.ndarray,
    seed: tuple[float, float] | None = None,
    auto: bool = True,
) -> np.ndarray:
    """
    Selecciona la componente conexa del órgano deseado dentro de la máscara por threshold.

    Flujo (como Odyssey "Select Object" / GammaSync):
      1. La máscara por threshold suele contener corazón + hígado + ruido.
      2. Se etiquetan las componentes conexas.
      3. Si el usuario dio un seed (click en el corazón), se elige la componente
         que contiene ese punto (o la más cercana a él).
      4. Si es automático, se elige la componente más compatible con corazón:
         central, grande, no pegada al borde inferior/lateral (típico hígado).

    Returns
    -------
    ndarray bool — máscara solo del órgano seleccionado (vacía si no hay).
    """
    from scipy.ndimage import center_of_mass as _com
    from scipy.ndimage import label as _label

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    lbl, n = _label(mask)
    if n <= 0:
        return np.zeros_like(mask, dtype=bool)

    h, w = mask.shape
    cy0, cx0 = (h - 1) * 0.5, (w - 1) * 0.5

    # Selección por seed (click del usuario)
    if seed is not None and np.isfinite(seed[0]) and np.isfinite(seed[1]):
        sy, sx = float(seed[0]), float(seed[1])
        iy, ix = int(round(sy)), int(round(sx))
        if 0 <= iy < h and 0 <= ix < w and lbl[iy, ix] > 0:
            return lbl == lbl[iy, ix]
        # Si el click no cae en una componente, elegir la más cercana.
        best, best_d = None, 1e18
        for cid in range(1, n + 1):
            comp = lbl == cid
            if comp.sum() < 4:
                continue
            cy, cx = _com(comp)
            d = (cy - sy) ** 2 + (cx - sx) ** 2
            if d < best_d:
                best_d, best = d, comp
        if best is not None:
            return best
        return np.zeros_like(mask, dtype=bool)

    if not auto:
        return mask

    # Selección automática GammaSync: componente central y grande (corazón),
    # evitando la más periférica/inferior (típico hígado).
    best, best_score = None, -1e18
    for cid in range(1, n + 1):
        comp = lbl == cid
        area = int(comp.sum())
        if area < 6:
            continue
        cy, cx = _com(comp)
        dist_c = float(np.hypot(cy - cy0, cx - cx0)) / max(1.0, 0.5 * min(h, w))
        area_frac = area / float(h * w)
        # Score: favorece central y con área razonable, penaliza muy periférico.
        score = 2.0 * (1.0 - dist_c) + 1.0 * min(1.0, area_frac / 0.05)
        if dist_c > 0.95:
            score -= 2.0
        if score > best_score:
            best_score, best = score, comp
    return best if best is not None else mask


def _organ_mask(
    img: np.ndarray,
    threshold_frac: float,
    seed: tuple[float, float] | None = None,
    use_organ_selection: bool = True,
) -> np.ndarray:
    """
    Máscara del órgano para tracking, disponible como ajuste en TODOS los métodos.

    - Si use_organ_selection es False: máscara simple por threshold (máscara completa).
    - Si hay seed (pick del usuario): componente del órgano bajo el pick.
    - Si no hay seed (automático): componente central/grande (corazón, evita hígado).

    Esto permite que cualquier método (COM, Stasis, Hopkins, Odyssey, Threshold)
    use la selección de órgano como ajuste para evitar que el hígado influya.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.size == 0 or img.max() <= 0:
        return np.zeros_like(img, dtype=bool)
    mask = img > (threshold_frac * img.max())
    if not use_organ_selection:
        return mask
    return _select_organ_component(mask, seed=seed, auto=(seed is None))


def _tracking_gammasync(
    projections: np.ndarray,
    axis: str,
    threshold_frac: float,
    seed: tuple[float, float] | None = None,
) -> dict:
    """
    Tracking GammaSync: threshold + selección de componente del órgano (corazón).

    Flujo (el que pediste):
      1. Threshold enmascara corazón + hígado + ruido.
      2. Selección de la componente del órgano:
         - si el usuario dio seed (click en el corazón), sigue esa componente;
         - si es automático, elige la componente central/grande (corazón), evitando hígado.
      3. El tracking usa SOLO esa componente → el hígado no influye en la corrección.
    """
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
        organ = _select_organ_component(mask, seed=seed, auto=(seed is None))
        if organ.sum() < 4:
            continue
        cy, cx = _com(organ)
        com_series[a] = cy if axis == "y" else cx

    return {
        "axis": axis,
        "com_series": com_series,
        "method": "gammasync",
        "threshold_frac": threshold_frac,
        "seed": seed,
    }


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


def _tracking_odyssey(
    projections: np.ndarray,
    axis: str,
    threshold_frac: float,
    n_iterations: int = 3,
) -> dict:
    """
    Tracking estilo Odyssey (manual LX tux079 pág 79-87).

    Algoritmo replicado del manual:
      "Motion Correction is accomplished by comparing each acquired projection
       to a re-projected image (similar to maximum pixel raytrace) computed at
       the same acquisition angle."

    Flujo:
      1. Máscara del órgano por threshold interactivo (Select Object).
      2. Estimar volumen actual (suma de proyecciones = proxy de la estimación).
      3. Re-proyectar esa estimación a cada ángulo (max pixel raytrace = máximo
         a lo largo del eje perpendicular a la proyección).
      4. Comparar cada proyección adquirida contra su re-proyección (centro del
         órgano) y calcular el shift.
      5. Iterar hasta 3 veces (una por eje, como indica el manual), refinando
         la estimación con las proyecciones corregidas.

    La corrección final se aplica a todas las proyecciones (todos los gates).
    """
    from scipy.ndimage import shift as _ndi_shift

    proj = np.asarray(projections, dtype=np.float64)
    n_gates, n_angles, H, W = proj.shape
    threshold_frac = float(threshold_frac)
    threshold_frac = min(max(threshold_frac, 0.01), 0.90)
    n_iterations = int(max(1, min(3, n_iterations)))

    from scipy.ndimage import center_of_mass as _com

    current = proj.copy()
    com_series = np.full((n_angles,), np.nan, dtype=np.float64)
    total_shifts = np.zeros((n_angles,), dtype=np.float64)

    for _ in range(n_iterations):
        summed = current.sum(axis=0)  # (n_angles, H, W) — UngGat de la estimación actual
        # Re-proyección esperada: el centro de masa de la proyección sumada de
        # TODOS los ángulos colapsado sobre el eje perpendicular. Es el análogo
        # del "max pixel raytrace" del manual: la posición del objeto esperada
        # si estuviera alineado. Comparamos el centro de cada proyección contra
        # esa referencia global y corregimos la desviación.
        # Referencia global: centro de masa del objeto sobre el volumen completo
        # (todos los ángulos sumados), que es la re-proyección "ideal" alineada.
        total_img = summed.sum(axis=0)  # (H, W) suma de todos los ángulos
        if total_img.max() <= 0:
            break
        mask_total = total_img > (threshold_frac * total_img.max())
        if mask_total.sum() < 4:
            break
        cy_ref, cx_ref = _com(mask_total)
        ref = cy_ref if axis == "y" else cx_ref

        centers = np.full((n_angles,), np.nan, dtype=np.float64)
        for a in range(n_angles):
            img = summed[a]
            if img.max() <= 0:
                continue
            mask = img > (threshold_frac * img.max())
            if mask.sum() < 4:
                continue
            cy, cx = _com(mask)
            centers[a] = cy if axis == "y" else cx

        valid = np.isfinite(centers)
        if valid.sum() < 3:
            break
        # Shift de esta iteración: llevar cada ángulo a la referencia global.
        shifts_iter = np.where(valid, ref - centers, 0.0)
        # Suavizar para evitar sobre-corrección en una sola iteración.
        shifts_iter = shifts_iter * 0.8
        total_shifts += shifts_iter
        if axis == "y":
            current = apply_shifts_to_projections(current, shifts_iter, np.zeros_like(shifts_iter))
        else:
            current = apply_shifts_to_projections(current, np.zeros_like(shifts_iter), shifts_iter)
        com_series = centers

    # Devolver el tracking final: com_series del último estado + shifts totales.
    return {
        "axis": axis,
        "com_series": com_series,
        "method": "odyssey",
        "threshold_frac": threshold_frac,
        "n_iterations": n_iterations,
        "_odyssey_total_shifts": total_shifts,
    }


def _tracking_stasis(
    projections: np.ndarray,
    axis: str,
    threshold_frac: float,
    seed: tuple[float, float] | None = None,
) -> dict:
    """
    Tracking estilo Stasis (método Xeleris/Myovation).

    Concepto: el corazón vuelve a una posición "estática" (baseline) entre
    movimientos respiratorios. Stasis detecta esa posición de referencia como
    la MODA (valor más frecuente) del centro del órgano a lo largo de los frames,
    y corrige cada frame hacia esa referencia estática (no hacia la mediana).

    Ventaja vs mediana: la mediana se sesga si el paciente pasa más tiempo en una
    posición no-baseline; la moda (stasis) captura la posición de reposo real.
    Hopkins es una variante que pondera por estabilidad temporal.
    """
    from scipy.ndimage import center_of_mass as _com

    proj = np.asarray(projections, dtype=np.float64)
    summed = proj.sum(axis=0)
    n_angles = summed.shape[0]
    threshold_frac = float(threshold_frac)
    threshold_frac = min(max(threshold_frac, 0.01), 0.90)

    # Centro del órgano por frame (con selección de componente si hay seed o auto).
    centers = np.full((n_angles,), np.nan, dtype=np.float64)
    for a in range(n_angles):
        img = summed[a]
        if img.max() <= 0:
            continue
        organ = _organ_mask(img, threshold_frac, seed=seed, use_organ_selection=True)
        if organ.sum() < 4:
            continue
        cy, cx = _com(organ)
        centers[a] = cy if axis == "y" else cx

    valid = np.isfinite(centers)
    if valid.sum() < 3:
        return {"axis": axis, "com_series": centers, "method": "stasis", "threshold_frac": threshold_frac, "_stasis_shifts": np.zeros((n_angles,))}

    # Referencia estática (stasis): moda del centro redondeado a 0.5px.
    # Agrupa frames por posición similar y toma la posición del grupo más grande.
    rounded = np.round(centers[valid] * 2.0) / 2.0  # resolución 0.5px
    # Hopkins: pondera por estabilidad (frames consecutivos en la misma posición).
    best_ref, best_count = None, -1
    for val in np.unique(rounded):
        count = int(np.sum(rounded == val))
        if count > best_count:
            best_count, best_ref = count, float(val)
    if best_ref is None:
        best_ref = float(np.median(centers[valid]))

    # Shifts: llevar cada frame a la referencia estática.
    shifts = np.where(valid, best_ref - centers, 0.0)
    return {
        "axis": axis,
        "com_series": centers,
        "method": "stasis",
        "threshold_frac": threshold_frac,
        "stasis_reference": best_ref,
        "stasis_baseline_frames": best_count,
        "_stasis_shifts": shifts,
    }


def _tracking_hopkins(
    projections: np.ndarray,
    axis: str,
    threshold_frac: float,
    seed: tuple[float, float] | None = None,
) -> dict:
    """
    Tracking estilo Hopkins (variante de Stasis con estabilidad temporal).

    Similar a Stasis pero la referencia se calcula como el centro del frame más
    estable (el que minimiza la varianza con sus vecinos temporales), no la moda.
    Útil cuando el movimiento es gradual (respiración) más que por saltos.
    """
    from scipy.ndimage import center_of_mass as _com

    proj = np.asarray(projections, dtype=np.float64)
    summed = proj.sum(axis=0)
    n_angles = summed.shape[0]
    threshold_frac = float(threshold_frac)
    threshold_frac = min(max(threshold_frac, 0.01), 0.90)

    centers = np.full((n_angles,), np.nan, dtype=np.float64)
    for a in range(n_angles):
        img = summed[a]
        if img.max() <= 0:
            continue
        organ = _organ_mask(img, threshold_frac, seed=seed, use_organ_selection=True)
        if organ.sum() < 4:
            continue
        cy, cx = _com(organ)
        centers[a] = cy if axis == "y" else cx

    valid = np.isfinite(centers)
    if valid.sum() < 3:
        return {"axis": axis, "com_series": centers, "method": "hopkins", "threshold_frac": threshold_frac, "_hopkins_shifts": np.zeros((n_angles,))}

    # Referencia Hopkins: centro del frame con menor varianza local (más estable).
    valid_idx = np.where(valid)[0]
    best_ref, best_var = None, 1e18
    for idx in valid_idx:
        lo = max(0, idx - 1)
        hi = min(n_angles, idx + 2)
        neighbors = centers[lo:hi][np.isfinite(centers[lo:hi])]
        if neighbors.size < 2:
            continue
        var = float(np.var(neighbors))
        if var < best_var:
            best_var, best_ref = var, float(centers[idx])
    if best_ref is None:
        best_ref = float(np.median(centers[valid]))

    shifts = np.where(valid, best_ref - centers, 0.0)
    return {
        "axis": axis,
        "com_series": centers,
        "method": "hopkins",
        "threshold_frac": threshold_frac,
        "hopkins_reference": best_ref,
        "_hopkins_shifts": shifts,
    }


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
    seed: tuple[float, float] | None = None,
    manual_shifts_y: np.ndarray | None = None,
    manual_shifts_x: np.ndarray | None = None,
) -> dict:
    """
    Motion correction de proyecciones SPECT gated.

    Methods:
      - gammasync: threshold + selección de componente del órgano (corazón), automática o por seed (click). Recomendado.
      - stasis: referencia estática (moda) del centro del órgano, como Xeleris Stasis.
      - hopkins: referencia del frame más estable temporalmente, como Xeleris Hopkins.
      - odyssey: re-proyección iterativa (manual LX).
      - com: centro de masa sobre máscara por threshold.
      - threshold: centro del bounding box de la máscara por threshold.
    """
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibió {proj.shape}")

    method = str(method or "gammasync").strip().lower()
    if method not in ("gammasync", "stasis", "hopkins", "com", "threshold", "odyssey"):
        raise ValueError("method debe ser 'gammasync', 'stasis', 'hopkins', 'com', 'threshold' u 'odyssey'")

    axes_to_correct = ["y", "x"] if axis == "xy" else [axis]
    tracking = {}
    shifts_y = np.zeros((proj.shape[1],), dtype=np.float64)
    shifts_x = np.zeros((proj.shape[1],), dtype=np.float64)

    for ax in axes_to_correct:
        if method == "gammasync":
            trk = _finalize_tracking(_tracking_gammasync(proj, axis=ax, threshold_frac=threshold_frac, seed=seed))
        elif method == "stasis":
            trk = _tracking_stasis(proj, axis=ax, threshold_frac=threshold_frac, seed=seed)
            stasis_shifts = np.asarray(trk.pop("_stasis_shifts", np.zeros((proj.shape[1],))), dtype=np.float64)
            trk = _finalize_tracking(trk)
            trk["suggested_shifts_px"] = stasis_shifts
            trk["max_shift_px"] = round(float(np.abs(stasis_shifts).max()) if stasis_shifts.size else 0.0, 2)
            trk["motion_suspected"] = bool(trk["max_shift_px"] > 1.5 or trk.get("n_outliers", 0) >= 2)
        elif method == "hopkins":
            trk = _tracking_hopkins(proj, axis=ax, threshold_frac=threshold_frac, seed=seed)
            hopkins_shifts = np.asarray(trk.pop("_hopkins_shifts", np.zeros((proj.shape[1],))), dtype=np.float64)
            trk = _finalize_tracking(trk)
            trk["suggested_shifts_px"] = hopkins_shifts
            trk["max_shift_px"] = round(float(np.abs(hopkins_shifts).max()) if hopkins_shifts.size else 0.0, 2)
            trk["motion_suspected"] = bool(trk["max_shift_px"] > 1.5 or trk.get("n_outliers", 0) >= 2)
        elif method == "odyssey":
            trk = _tracking_odyssey(proj, axis=ax, threshold_frac=threshold_frac)
            # Odyssey: usar los shifts acumulados de las iteraciones, no la mediana.
            odyssey_shifts = np.asarray(trk.pop("_odyssey_total_shifts", np.zeros((proj.shape[1],))), dtype=np.float64)
            trk = _finalize_tracking(trk)
            trk["suggested_shifts_px"] = odyssey_shifts
            trk["max_shift_px"] = round(float(np.abs(odyssey_shifts).max()) if odyssey_shifts.size else 0.0, 2)
            trk["motion_suspected"] = bool(trk["max_shift_px"] > 1.5 or trk.get("n_outliers", 0) >= 2)
        elif method == "threshold":
            trk = _finalize_tracking(_tracking_from_threshold(proj, axis=ax, threshold_frac=threshold_frac))
        else:
            trk = _finalize_tracking(_tracking_from_com(proj, axis=ax, threshold_frac=threshold_frac, seed=seed))
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
