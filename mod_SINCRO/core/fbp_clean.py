"""SINCRO - core.fbp_clean  (FBP_CLEAN, rama FBP_CLEAN / FBP_POCO_ORTODOXO)

Denoise de SPECT miocárdico de mitad de tiempo/dosis EN EL SINOGRAMA (pre-FBP).

Lecciones del banco de pruebas (harness 022/023/024/025):
  - El ruido del 5s FBP son ESTRÍAS RADIALES (artefacto de la retroproyección en
    bajo conteo), no moteado puntual. Un denoiser espacial POST-recon no las
    quita sin difuminar (022: descartado).
  - El ruido hay que atacarlo en las PROYECCIONES, donde es Poisson puro, ANTES
    de que el FBP lo convierta en streaks (023: funciona).
  - Filtro ganador: BILATERAL por proyección (preserva bordes del sinograma),
    sigma_color=0.04, sigma_spatial=1.5. Más de 0.08 difumina (025).
  - El nivel de ruido se estima con la resta 10s−5s (idea del usuario, validada
    en 024): std de la resta / std de la señal ≈ 0.48 en el estudio de prueba.

API:
  - denoise_projections_bilateral(proj, sigma_color=0.04, ...)  -> proyecciones limpias
  - fbp_clean(volume, ...)        -> (legado) denoise post-recon edge-preserving
  - cnr_cavity(slice2d, ...)      -> métrica CNR cavidad/pared
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


def denoise_projections_bilateral(
    projections: np.ndarray,
    *,
    sigma_color: float = 0.04,
    sigma_spatial: float = 1.5,
) -> np.ndarray:
    """Denoise Poisson de las proyecciones crudas, ANTES del FBP.

    Bilateral por proyección (2D, ejes H y W del detector): suaviza el ruido
    granular preservando los bordes del sinograma (el contorno del corazón).
    sigma_color=0.04 es el default calibrado (banco 025): limpia el fondo sin
    difuminar la cavidad. >0.08 difumina de más.

    Entrada: (n_angles, H, W) o (gates, n_angles, H, W). Devuelve igual shape.
    """
    from skimage.restoration import denoise_bilateral
    proj = np.asarray(projections, dtype=np.float64)
    single = proj.ndim == 3
    if single:
        proj = proj[None, ...]
    out = np.empty_like(proj)
    for g in range(proj.shape[0]):
        for a in range(proj.shape[1]):
            p = proj[g, a]
            pmax = float(p.max()) or 1.0
            out[g, a] = denoise_bilateral(
                p / pmax, sigma_color=float(sigma_color), sigma_spatial=float(sigma_spatial)
            ) * pmax
    return out[0] if single else out


def estimate_noise_ratio(projections_low: np.ndarray, projections_high: np.ndarray) -> float:
    """Ratio ruido/señal estimado por la resta alto−bajo conteo (idea del usuario).

    Escala el bajo conteo a la misma suma que el alto y mide std(resta)/std(bajo).
    Sirve para calibrar la fuerza del filtro según el ruido real del estudio.
    """
    lo = np.asarray(projections_low, dtype=np.float64)
    hi = np.asarray(projections_high, dtype=np.float64)
    s_lo, s_hi = float(lo.sum()), float(hi.sum())
    if s_lo <= 0 or s_hi <= 0:
        return 0.0
    lo_s = lo * (s_hi / s_lo)
    noise = float((hi - lo_s).std())
    sig = float(lo_s.std()) or 1.0
    return noise / sig


def sharpen_by_subtraction(volume_sharp: np.ndarray, volume_blur: np.ndarray, k: float) -> np.ndarray:
    """Realce de bordes/cavidad por resta (idea del usuario, banco 026/027).

    out = clip(nítido − k × difuso, 0). Restar una fracción de la versión muy
    suavizada realza los bordes y ABRE la cavidad (unsharp mask). k típico 0.5;
    más k = más realce pero más ruido de fondo. Como realza la cavidad, también
    realza los defectos de perfusión (diagnósticamente deseable).
    """
    return np.clip(np.asarray(volume_sharp, dtype=np.float64) - float(k) * np.asarray(volume_blur, dtype=np.float64), 0.0, None)


def _local_stats(v: np.ndarray, win: int = 5):
    mean = uniform_filter(v, size=win)
    mean_sq = uniform_filter(v * v, size=win)
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return mean, np.sqrt(var)


def fbp_clean(
    volume: np.ndarray,
    *,
    strength: float = 0.6,
    sigma_spatial: float = 1.0,
    edge_threshold: float | None = None,
    win: int = 5,
) -> np.ndarray:
    """(LEGADO) Denoise edge-preserving post-recon. El banco 022 lo descartó:
    no quita las estrías del FBP. Usar `denoise_projections_bilateral` (pre-FBP).
    Se conserva por compatibilidad / comparación.
    """
    v = np.asarray(volume, dtype=np.float64)
    smooth = gaussian_filter(v, sigma=sigma_spatial, mode="nearest")
    gz, gy, gx = np.gradient(v)
    grad = np.sqrt(gz * gz + gy * gy + gx * gx)
    thr = float(edge_threshold) if edge_threshold is not None else float(np.median(grad) + 1e-6)
    w = 1.0 / (1.0 + (grad / max(thr, 1e-6)) ** 2)
    w = float(np.clip(strength, 0.0, 1.0)) * w
    return w * smooth + (1.0 - w) * v


def cnr_cavity(slice2d: np.ndarray, cy: float, cx: float, r_cav=3.0, r_in=4.0, r_out=9.0) -> float:
    """CNR cavidad/pared en un corte centrado en (cy,cx)."""
    yy, xx = np.mgrid[0:slice2d.shape[0], 0:slice2d.shape[1]]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cav = slice2d[d < r_cav]
    ring = slice2d[(d >= r_in) & (d <= r_out)]
    if cav.size == 0 or ring.size == 0 or cav.std() <= 0:
        return float("nan")
    return float((ring.mean() - cav.mean()) / cav.std())
