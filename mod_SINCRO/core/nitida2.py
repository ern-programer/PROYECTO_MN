"""SINCRO - core.nitida2  (NITIDA II, rama NITIDA_II)

Denoiser de gated SPECT de bajo conteo **guiado por la imagen ungated** de alto
conteo (guided filter, He et al. 2013), como primer ingrediente de NITIDA II.

IDEA
----
Un gated de 5s tiene ~1/8 de las cuentas por gate -> cada frame sale ruidoso.
El **ungated** (suma de los 8 gates) tiene la estadística completa y define bien
la anatomía (dónde está la pared, los bordes endo/epi), pero **no tiene
movimiento** (está promediado en el tiempo).

El guided filter usa el ungated como **imagen-guía**: reescribe cada gate como
una transformación lineal LOCAL de la guía (q = a·I + b por ventana), ajustada
para parecerse al gate ruidoso. Con eso:

- **Presta la anatomía/bordes** de la guía (alto conteo) -> baja el ruido y
  preserva los bordes del miocardio.
- **NO presta el movimiento**: los coeficientes a,b se ajustan al nivel local de
  CADA gate, así la contracción real (diferencias entre gates) se conserva. La
  misma guía aplicada a los 8 gates no introduce dispersión de fase artificial.

Esto es distinto de un gaussiano (que borronea uniforme y come bordes) y de
"dividir el ungated en 8" (que borra el movimiento). Acá el movimiento sigue
saliendo de las cuentas de cada gate; solo se limpia ruido con anatomía común.

USO
---
    from core.nitida2 import denoise_gates_with_guide
    clean = denoise_gates_with_guide(gated_cube, radius=2, eps=0.01)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def _box_mean(vol: np.ndarray, size) -> np.ndarray:
    return uniform_filter(vol, size=size, mode="nearest")


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Guided filter ND (He et al. 2013).

    Parameters
    ----------
    guide : imagen-guía (alto conteo), normalizada preferentemente a ~[0,1].
    src   : imagen a filtrar (gate ruidoso), en las MISMAS unidades que se quiere
            devolver (el filtro es afín, conserva la escala de ``src``).
    radius: radio de la ventana en píxeles (ventana = 2·radius+1 por eje).
    eps   : regularización. Más grande = más suavizado (menos fiel a bordes de la
            guía). En unidades de la guía normalizada (p.ej. 0.001-0.05).

    Returns
    -------
    Imagen filtrada, misma forma que ``src``.
    """
    guide = np.asarray(guide, dtype=np.float64)
    src = np.asarray(src, dtype=np.float64)
    size = int(2 * int(radius) + 1)

    mean_I = _box_mean(guide, size)
    mean_p = _box_mean(src, size)
    corr_I = _box_mean(guide * guide, size)
    corr_Ip = _box_mean(guide * src, size)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I

    mean_a = _box_mean(a, size)
    mean_b = _box_mean(b, size)
    return mean_a * guide + mean_b


def _normalize_guide(guide: np.ndarray) -> np.ndarray:
    g = np.asarray(guide, dtype=np.float64)
    lo = float(np.min(g))
    hi = float(np.max(g))
    if hi - lo < 1e-12:
        return np.zeros_like(g)
    return (g - lo) / (hi - lo)


def denoise_gates_with_guide(
    gated_cube: np.ndarray,
    *,
    guide_volume: np.ndarray | None = None,
    radius: int = 2,
    eps: float = 0.01,
) -> np.ndarray:
    """Denoisa cada gate usando el ungated de alto conteo como guía.

    Parameters
    ----------
    gated_cube : (n_gates, n_slices, H, W) gated SPECT reconstruido.
    guide_volume : (n_slices, H, W) imagen-guía de alto conteo (ungated /
        Summed Tomo). Si es None se usa la SUMA de los gates (misma estadística
        que el ungated real).
    radius, eps : parámetros del guided filter.

    Returns
    -------
    Cubo denoisado, misma forma. Conserva la escala de cada gate (el total de
    cuentas por gate se mantiene ~igual; solo se redistribuye el ruido).
    """
    cube = np.asarray(gated_cube, dtype=np.float64)
    if cube.ndim != 4:
        raise ValueError(f"gated_cube debe ser 4D (gates,slices,H,W); recibió {cube.shape}")

    guide = cube.sum(axis=0) if guide_volume is None else np.asarray(guide_volume, dtype=np.float64)
    if guide.shape != cube.shape[1:]:
        raise ValueError(
            f"guide_volume {guide.shape} no coincide con el volumen del gated {cube.shape[1:]}."
        )

    guide_n = _normalize_guide(guide)
    out = np.empty_like(cube)
    for g in range(cube.shape[0]):
        out[g] = guided_filter(guide_n, cube[g], radius=radius, eps=eps)
    return np.clip(out, 0.0, None)


def temporal_harmonic_filter(
    gated_cube: np.ndarray,
    *,
    n_harmonics: int = 2,
    axis: int = 0,
) -> np.ndarray:
    """Filtrado temporal de Fourier a lo largo de los gates (estilo QGS/Emory).

    El movimiento cardíaco con 8 gates es de BAJA frecuencia (1ª armónica domina,
    algo de 2ª por el engrosamiento); el ruido Poisson es de banda ancha en las 8
    muestras temporales. Conservando solo los armónicos 0..``n_harmonics`` de la
    FFT a lo largo del eje de gates se elimina la mayor parte del ruido SIN tocar
    el movimiento (se conserva por construcción).

    Es LINEAL a lo largo de los gates, por lo que conmuta con la retroproyección:
    aplicarlo en proyecciones equivale a aplicarlo en el volumen reconstruido.

    Parameters
    ----------
    gated_cube : array con los gates en ``axis`` (proyecciones o reconstruido).
    n_harmonics : número de armónicos a conservar además del DC (0=solo media,
        1=media+fundamental, 2=+2ª armónica). Típico clínico: 2.
    axis : eje temporal (gates). Por defecto 0.

    Returns
    -------
    Cubo filtrado, misma forma, no negativo.
    """
    cube = np.asarray(gated_cube, dtype=np.float64)
    n_gates = cube.shape[axis]
    if n_gates < 3:
        return cube.copy()
    spec = np.fft.rfft(cube, axis=axis)
    keep = int(max(0, n_harmonics)) + 1  # +1 por el DC (bin 0)
    if keep < spec.shape[axis]:
        idx = [slice(None)] * spec.ndim
        idx[axis] = slice(keep, None)
        spec[tuple(idx)] = 0.0
    out = np.fft.irfft(spec, n=n_gates, axis=axis)
    return np.clip(out, 0.0, None)
