"""SINCRO - core.nitida3  (NÍTIDA III, rama MATRIZ_FINA_(k3))

Reconstrucción/denoising original para SPECT miocárdico de mitad de tiempo/dosis.
Fundamento en docs/NITIDA_III_fundamento.md. Tres pilares:

  A) Feta axial restringida (ya en raw_reconstruction.recon_slice_range).
  B) Guía ungated (alto conteo) para regularizar el gated (bajo conteo).
  C) "Matched Recovery": RR adaptativa por SNR local (fracción de PSF según SNR).

Diseño propio sobre matemática publicada (MAP-OSEM Green OSL, priors edge-
preserving, guided filtering). NO copia Evolution/Astonish/WBR.

API mínima (esqueleto v0):
  - local_snr_map(volume, ...)        -> mapa de SNR local (Pilar C)
  - guided_prior_update(...)          -> prior guiado por ungated (Pilar B)
  - nitida3_osem_slab(...)            -> OSEM con prior + RR adaptativa
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter


def local_snr_map(volume: np.ndarray, *, win: int = 5, eps: float = 1e-6) -> np.ndarray:
    """Mapa de SNR local por voxel: media_local / std_local (ventana win^3).

    Estima dónde la señal domina al ruido (pared bien perfundida) vs dónde el
    ruido domina (cavidad, defectos, fondo). Es la entrada del Pilar C: la RR se
    aplica con más fuerza donde la SNR es alta y se frena donde es baja.
    """
    v = np.asarray(volume, dtype=np.float64)
    mean = uniform_filter(v, size=win)
    mean_sq = uniform_filter(v * v, size=win)
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    std = np.sqrt(var)
    return mean / (std + eps)


def matched_recovery_weight(snr: np.ndarray, *, snr_low: float = 1.0, snr_high: float = 4.0) -> np.ndarray:
    """Peso de recuperación de resolución en [0,1] según SNR local.

    snr <= snr_low  -> 0 (no recuperar: sólo ruido, no amplificar).
    snr >= snr_high -> 1 (recuperar pleno: hay señal que lo justifica).
    Entre medio: rampa lineal. Es la "amplitud adaptativa" de la RR (Pilar C).
    """
    s = np.asarray(snr, dtype=np.float64)
    w = (s - snr_low) / max(snr_high - snr_low, 1e-6)
    return np.clip(w, 0.0, 1.0)


def edge_preserving_prior(x: np.ndarray, *, kind: str = "median", size: int = 3) -> np.ndarray:
    """Referencia de prior (M en el update OSL de Green).

    'median'  -> mediana 3D (preserva bordes, no engorda la pared).
    'gauss'   -> gaussiano (suaviza pero engorda; solo para comparar).
    """
    if kind == "median":
        return median_filter(x, size=size)
    if kind == "gauss":
        return gaussian_filter(x, sigma=max(1.0, size / 2.0))
    raise ValueError(f"prior desconocido: {kind}")


def nitida3_osem_slab(
    projections: np.ndarray,
    angles_deg: np.ndarray,
    *,
    iterations: int = 2,
    subsets: int = 4,
    beta: float = 0.3,
    prior: str = "median",
    psf=None,
    guide: np.ndarray | None = None,
    guide_weight: float = 0.0,
    slice_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """OSEM con prior edge-preserving + RR adaptativa (esqueleto NÍTIDA III).

    - ``prior``: mediana (default) para controlar ruido sin engordar la pared.
    - ``psf``: PsfModel para RR. La RR se aplica con peso adaptativo por SNR
      local (Pilar C) en vez de plena (evita amplificar ruido en baja SNR).
    - ``guide`` + ``guide_weight``: volumen ungated de alto conteo como guía
      estructural (Pilar B). 0 = desactivado (v0).

    NOTA v0: esqueleto. La RR adaptativa y la guía se integran en iteraciones
    siguientes; acá queda la firma y el prior funcionando sobre la feta.
    """
    from core.raw_reconstruction import reconstruct_projection_volume, ProjectionFilterConfig

    # v0: recon OSEM base (con PSF si se pasa) sobre la feta, luego prior como
    # post-paso edge-preserving. La integración MAP completa (prior dentro del
    # update) y la RR adaptativa son el siguiente incremento.
    vol = reconstruct_projection_volume(
        projections, angles_deg, method="osem",
        projection_filter=ProjectionFilterConfig("none", 0.5, 1),
        iterations=iterations, subsets=subsets, psf=psf, slice_range=slice_range,
    )
    if prior and prior != "none":
        vol = edge_preserving_prior(vol, kind=prior)
    return vol
