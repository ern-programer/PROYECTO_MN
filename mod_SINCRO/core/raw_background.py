"""SINCRO - core.raw_background

Sustracción de fondo sobre la imagen CRUDA (proyecciones SPECT), como realce
visual / preprocesado para orientar y reconstruir con menos dudas.

QUÉ ES (Y QUÉ NO ES)
--------------------
Es la resta de un **piso de cuentas de fondo** medido en una región **sin
corazón** (dispersión, actividad de tejido blando, ruido de fondo). Ese piso es
**aditivo**: se suma pixel a pixel a la señal del miocardio. Restarlo levanta el
contraste relativo del corazón y "despeja" la imagen en estudios de bajas cuentas
o alto fondo.

**No es corrección de atenuación.** La atenuación es multiplicativa y depende de
la profundidad del tejido; acá no se corrige eso. Esta herramienta es puramente
un realce de visualización / preprocesado para la etapa de orientación.

DOS MODOS
---------
- ``constant``: se mide un nivel en la ROI de fondo y se resta parejo a toda la
  imagen (clip a 0). Simple y predecible.
- ``localized``: se mide el nivel en la VOI de fondo pero solo se resta dentro de
  la ROI del corazón (con borde suavizado), para no tocar el resto de la imagen.

Todas las funciones son numpy puro (sin PyQt) para poder testearse aparte.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BackgroundResult:
    """Resultado de una sustracción de fondo sobre la imagen cruda."""

    image: np.ndarray
    level: float
    method: str
    clipped_fraction: float = 0.0
    notes: list[str] = field(default_factory=list)


def polygon_mask(shape: tuple[int, int], polygon) -> np.ndarray:
    """Rasteriza un polígono (lista de (x, y) en píxeles) a una máscara booleana.

    Regla par-impar (even-odd) vectorizada; sin dependencias externas.
    ``shape`` es (H, W). Puntos con x=columna, y=fila.
    """
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=bool)
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return mask

    xs = pts[:, 0]
    ys = pts[:, 1]
    # Grilla de centros de píxel.
    grid_y = np.arange(h, dtype=np.float64)[:, None]  # (H, 1)
    grid_x = np.arange(w, dtype=np.float64)[None, :]  # (1, W)

    n = pts.shape[0]
    inside = np.zeros((h, w), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]
        # ¿El lado (j->i) cruza la horizontal que pasa por cada fila?
        cond = ((yi > grid_y) != (yj > grid_y))
        # Coordenada x de la intersección para cada fila (evitar div por cero).
        denom = (yj - yi)
        denom = np.where(denom == 0.0, np.nan, denom)
        x_cross = xi + (grid_y - yi) * (xj - xi) / denom  # (H, 1)
        crosses = cond & (grid_x < x_cross)
        inside ^= crosses
        j = i
    mask[:] = inside
    return mask


def measure_background_level(image: np.ndarray, mask: np.ndarray, *, stat: str = "median") -> float:
    """Nivel de fondo (mediana o media) de los píxeles de ``image`` dentro de ``mask``."""
    arr = np.asarray(image, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if arr.shape != m.shape or not m.any():
        return 0.0
    vals = arr[m]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    if stat == "mean":
        return float(np.mean(vals))
    return float(np.median(vals))


def _feathered_weight(mask: np.ndarray, feather_px: float) -> np.ndarray:
    """Peso [0..1] a partir de una máscara con borde suavizado ``feather_px`` píxeles."""
    weight = np.asarray(mask, dtype=np.float64)
    if feather_px <= 0:
        return weight
    try:
        from scipy import ndimage as ndi

        dist = ndi.distance_transform_edt(mask.astype(bool))
        soft = np.clip(dist / float(feather_px), 0.0, 1.0)
        return soft
    except Exception:
        return weight


def subtract_constant(projections: np.ndarray, level: float) -> BackgroundResult:
    """Resta un nivel constante a toda la imagen (o stack) y hace clip a 0.

    ``projections`` puede ser 2D (H, W) o N-D (…, H, W). No modifica la entrada.
    """
    arr = np.asarray(projections, dtype=np.float64)
    lvl = float(max(0.0, level))
    out = arr - lvl
    negatives = out < 0.0
    clipped_fraction = float(np.count_nonzero(negatives) / out.size) if out.size else 0.0
    np.clip(out, 0.0, None, out=out)
    notes: list[str] = []
    if clipped_fraction > 0.35:
        notes.append(
            "Posible sobre-sustracción: más del 35% de los píxeles quedaron en cero. "
            "Bajá el nivel o achicá la ROI de fondo."
        )
    return BackgroundResult(
        image=out,
        level=lvl,
        method="constant",
        clipped_fraction=clipped_fraction,
        notes=notes,
    )


def subtract_localized(
    projections: np.ndarray,
    level: float,
    heart_mask: np.ndarray,
    *,
    feather_px: float = 2.0,
) -> BackgroundResult:
    """Resta el nivel de fondo solo dentro de ``heart_mask`` (borde suavizado).

    ``projections`` es 2D (H, W) o N-D (…, H, W); ``heart_mask`` es (H, W).
    """
    arr = np.asarray(projections, dtype=np.float64)
    m = np.asarray(heart_mask, dtype=bool)
    if arr.shape[-2:] != m.shape:
        return BackgroundResult(image=arr, level=float(level), method="localized",
                                notes=["La máscara del corazón no coincide con la imagen."])
    lvl = float(max(0.0, level))
    weight = _feathered_weight(m, feather_px)  # (H, W) en [0..1]
    out = arr - lvl * weight
    negatives = out < 0.0
    clipped_fraction = float(np.count_nonzero(negatives) / out.size) if out.size else 0.0
    np.clip(out, 0.0, None, out=out)
    notes: list[str] = []
    if clipped_fraction > 0.35:
        notes.append(
            "Posible sobre-sustracción dentro de la ROI del corazón. Revisá el nivel de la VOI de fondo."
        )
    return BackgroundResult(
        image=out,
        level=lvl,
        method="localized",
        clipped_fraction=clipped_fraction,
        notes=notes,
    )
