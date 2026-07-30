"""
SINCRO - core.perfusion_texture
================================

Textura GLCM de la perfusión miocárdica, combinada con la fase por segmento AHA.

Fundamento: la fase (cuándo se contrae cada región) y la perfusión (cuánto capta)
viven separadas en todos los paquetes clínicos. Jiang et al. 2025 (PMID 40391672)
mostró que combinar la TEXTURA de perfusión (features de matriz de co-ocurrencia
de niveles de gris, GLCM) con la fase mejora la predicción de respuesta a CRT.

Este módulo:
1. Cuantiza la imagen de perfusión a ``levels`` niveles (0..levels-1).
2. Calcula la GLCM (Haralick) y de ahí las features estándar:
   contrast, dissimilarity, homogeneity, energy (ASM), correlation, más la
   entropía de la GLCM (no la da skimage, se calcula a mano).
3. Opcionalmente calcula esas features POR SEGMENTO AHA (recorte del bounding box
   del segmento con máscara), y las une a la media de fase del segmento para
   producir una tabla perfusión+fase lista para exportar/analizar.

La imagen de perfusión típica es el gate de fin de diástole (ED) o la suma/mean de
gates; se pasa como array 2D (un corte) o 3D (volumen short-axis, se agregan las
features por segmento sobre todos los cortes donde aparece el segmento).

Requiere scikit-image (``graycomatrix``/``graycoprops``). Si no está instalado,
las funciones devuelven un dict con ``available=False`` en vez de romper.
"""
from __future__ import annotations

import numpy as np

try:  # skimage es dependencia del módulo, pero degradamos con elegancia.
    from skimage.feature import graycomatrix, graycoprops

    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover - entorno sin skimage
    _HAVE_SKIMAGE = False

#: Ángulos estándar de Haralick (0/45/90/135°) promediados para invariancia
#: rotacional aproximada.
_DEFAULT_ANGLES = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)


def _quantize(image: np.ndarray, mask: np.ndarray | None, levels: int) -> np.ndarray:
    """Escala la imagen a enteros 0..levels-1 usando el rango dentro de la máscara.

    Se normaliza por el percentil 1-99 de los píxeles válidos para que un píxel
    caliente aislado no colapse toda la escala (mismo criterio que el resto del
    módulo). Los píxeles fuera de máscara quedan en 0.
    """
    img = np.asarray(image, dtype=np.float64)
    if mask is None:
        valid = img[np.isfinite(img)]
    else:
        mask = np.asarray(mask, dtype=bool)
        valid = img[mask & np.isfinite(img)]
    if valid.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    lo = float(np.percentile(valid, 1.0))
    hi = float(np.percentile(valid, 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    q = np.rint(scaled * (levels - 1)).astype(np.uint8)
    if mask is not None:
        q = np.where(mask, q, 0).astype(np.uint8)
    return q


def glcm_features(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    levels: int = 16,
    distances: tuple[int, ...] = (1,),
    angles: tuple[float, ...] = _DEFAULT_ANGLES,
) -> dict:
    """Features GLCM (Haralick) de una imagen 2D.

    Returns
    -------
    dict
        ``available`` (bool) y, si lo está, ``contrast``, ``dissimilarity``,
        ``homogeneity``, ``energy``, ``correlation``, ``glcm_entropy`` (promedio
        sobre distancias y ángulos) y ``levels``.
    """
    if not _HAVE_SKIMAGE:
        return {"available": False, "reason": "scikit-image no instalado"}

    q = _quantize(image, mask, levels)
    if q.ndim != 2:
        return {"available": False, "reason": f"se esperaba 2D; llegó {q.ndim}D"}
    if int(q.max()) == 0 and int(q.min()) == 0:
        return {"available": False, "reason": "imagen vacía tras cuantizar"}

    glcm = graycomatrix(
        q,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )

    feats = {
        "contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        "dissimilarity": float(np.mean(graycoprops(glcm, "dissimilarity"))),
        "homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        "energy": float(np.mean(graycoprops(glcm, "energy"))),
        "correlation": float(np.mean(graycoprops(glcm, "correlation"))),
    }

    # Entropía de la GLCM (skimage no la expone). Se promedia sobre d y ángulo.
    with np.errstate(divide="ignore", invalid="ignore"):
        p = glcm.astype(np.float64)
        ent = -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=(0, 1))  # (d, ang)
    feats["glcm_entropy"] = float(np.mean(ent))
    feats["available"] = True
    feats["levels"] = int(levels)
    return feats


def perfusion_texture_by_segment(
    perfusion: np.ndarray,
    segment_map: np.ndarray,
    levels: int = 16,
    distances: tuple[int, ...] = (1,),
    angles: tuple[float, ...] = _DEFAULT_ANGLES,
    min_pixels: int = 12,
) -> dict[int, dict]:
    """Features GLCM de perfusión por segmento AHA (1..17).

    Parameters
    ----------
    perfusion : ndarray (n_slices, H, W) o (H, W)
        Mapa de perfusión (gate ED o media de gates), short-axis.
    segment_map : ndarray igual shape que ``perfusion``
        Mapa de segmentos AHA (0 = fuera, 1..17 = segmento), de
        ``core.aha_segments.map_to_17_segments``.
    min_pixels : int
        Mínimo de píxeles del segmento para que la GLCM sea confiable.

    Returns
    -------
    dict[int, dict]
        segmento → features (o ``available=False`` si el segmento es chico/ausente).
    """
    perfusion = np.asarray(perfusion, dtype=np.float64)
    segment_map = np.asarray(segment_map)
    if perfusion.shape != segment_map.shape:
        raise ValueError(
            f"perfusion {perfusion.shape} y segment_map {segment_map.shape} deben coincidir"
        )

    # Cuantización global (mismo rango para todos los segmentos → comparables).
    myo_mask = segment_map > 0
    q = _quantize(perfusion, myo_mask, levels)

    out: dict[int, dict] = {}
    for seg_id in range(1, 18):
        seg_mask = segment_map == seg_id
        n_px = int(seg_mask.sum())
        if n_px < min_pixels:
            out[seg_id] = {"available": False, "reason": f"pocos píxeles ({n_px})", "n_pixels": n_px}
            continue

        # Recorte al bounding box del segmento sobre el eje espacial. Para 3D se
        # toma el corte con más píxeles del segmento (el más representativo).
        if q.ndim == 3:
            per_slice = seg_mask.reshape(seg_mask.shape[0], -1).sum(axis=1)
            s = int(np.argmax(per_slice))
            q2d = q[s]
            m2d = seg_mask[s]
        else:
            q2d = q
            m2d = seg_mask

        ys, xs = np.nonzero(m2d)
        if ys.size < min_pixels:
            out[seg_id] = {"available": False, "reason": "recorte vacío", "n_pixels": int(ys.size)}
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        patch = q2d[y0:y1, x0:x1].copy()
        patch_mask = m2d[y0:y1, x0:x1]
        # Fuera del segmento → 0 (el nivel 0 domina el fondo, se acepta como en
        # cualquier ROI recortado; la señal útil es la textura intra-segmento).
        patch = np.where(patch_mask, patch, 0).astype(np.uint8)

        feats = glcm_features(patch, mask=None, levels=levels, distances=distances, angles=angles)
        feats["n_pixels"] = n_px
        out[seg_id] = feats
    return out


def combine_perfusion_phase(
    texture_by_seg: dict[int, dict],
    phase_by_seg: dict[int, float],
) -> list[dict]:
    """Une textura de perfusión y fase por segmento en una tabla plana.

    Cada fila es un segmento AHA con su fase media y sus features GLCM. Es la
    representación que necesita un análisis perfusión+fase (Jiang 2025,
    PMID 40391672) para predecir respuesta a CRT.

    Returns
    -------
    list[dict]
        Una fila por segmento 1..17 (ordenadas), con claves ``segment``,
        ``phase_deg`` y las features GLCM (NaN si no disponibles).
    """
    rows: list[dict] = []
    feat_keys = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "glcm_entropy")
    for seg_id in range(1, 18):
        tex = texture_by_seg.get(seg_id, {}) if texture_by_seg else {}
        row: dict = {
            "segment": seg_id,
            "phase_deg": float(phase_by_seg.get(seg_id, np.nan)) if phase_by_seg else float("nan"),
            "n_pixels": int(tex.get("n_pixels", 0)),
        }
        available = bool(tex.get("available", False))
        for k in feat_keys:
            row[k] = float(tex.get(k, np.nan)) if available else float("nan")
        rows.append(row)
    return rows
