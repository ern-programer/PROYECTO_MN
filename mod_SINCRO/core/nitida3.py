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


def huber_prior_grad(x: np.ndarray, *, delta: float = 0.1) -> np.ndarray:
    """Gradiente del prior de SUAVIDAD (Huber) sobre las 6 caras vecinas (2D).

    A diferencia de ``edge_preserving_prior`` (mediana, cuyo gradiente depende
    de la MAGNITUD de x), el prior de suavidad penaliza el CONTRASTE ESPACIAL
    (la diferencia voxel-vecino). Consecuencia clave para gated: un valor alto
    que OSCILA en el tiempo (la pared, lo que mide la fase) tiene vecinos que
    también oscilan igual, así que su contraste espacial es bajo y el prior NO
    lo frena. Solo frena el ruido (alta frecuencia espacial).

    Huber: cuadrático para |Δ|<=delta (suave, no escala bordes débiles) y
    lineal para |Δ|>delta (no penaliza de más los bordes reales pared/cavidad).
    Devuelve dU/dx normalizado a [~0..1] (listo para el denominador OSL).
    """
    v = np.asarray(x, dtype=np.float64)
    # Gradiente espacial por diferencias finitas (6 vecinos en 3D -> usamos 4 en 2D
    # por slice, suficiente para el ruido granular del detector).
    acc = np.zeros_like(v)
    for axis in (-2, -1):
        diff_p = np.diff(v, axis=axis)
        diff_m = -diff_p
        # Huber: psi(d) = d si |d|<=delta, sign(d)*delta si no. dU/dx ~ sum psi.
        def _psi(d):
            return np.where(np.abs(d) <= delta, d, np.sign(d) * delta)
        # Acumular psi en ambos extremos de cada arista.
        sl_p = [slice(None)] * v.ndim
        sl_m = [slice(None)] * v.ndim
        sl_p[axis] = slice(1, None)
        sl_m[axis] = slice(None, -1)
        acc[tuple(sl_p)] += _psi(diff_p)
        acc[tuple(sl_m)] += _psi(diff_m)
    # Normalizar: Huber acotado a delta por arista, 4 aristas -> max ~4*delta.
    return acc / max(4.0 * delta, 1e-9)


def prior_grad_adaptive(
    x: np.ndarray,
    *,
    snr: np.ndarray,
    beta0: float = 0.4,
    snr_low: float = 1.0,
    snr_high: float = 4.0,
    delta: float = 0.1,
) -> np.ndarray:
    """Gradiente del prior con beta ESPACIAL adaptativo por SNR local (Pilar C).

    beta_voxel = beta0 * matched_recovery_weight(snr_voxel). Donde la SNR es alta
    (pared bien perfundida) el freno es fuerte (limpieza); donde es baja
    (cavidad, fondo, y sobre todo las zonas donde la señal ES el movimiento) el
    freno es débil (no aplasta H1). Combinado con el prior de suavidad Huber,
    ataca el ruido sin tocar la oscilación cardíaca.

    Devuelve el campo ya multiplicado por beta espacial (para el denominador OSL:
    ``denom * (1 + prior_grad_adaptive(...))``).
    """
    v = np.asarray(x, dtype=np.float64)
    if snr.shape != v.shape:
        snr = np.broadcast_to(snr, v.shape).astype(np.float64)
    w = matched_recovery_weight(snr, snr_low=snr_low, snr_high=snr_high)
    beta_map = float(beta0) * w
    return beta_map * huber_prior_grad(v, delta=delta)


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


def nitida3_map_osem_gated(
    projections: np.ndarray,
    angles_deg: np.ndarray,
    *,
    iterations: int = 2,
    subsets: int = 4,
    beta: float = 0.3,
    prior: str = "median",
    prior_size: int = 3,
    psf=None,
    slice_range: tuple[int, int] | None = None,
    rescale: bool = True,
) -> np.ndarray:
    """MAP-OSEM gated (NÍTIDA III, Pilar MAP): prior edge-preserving DENTRO del
    update OSEM (Green OSL), aplicado gate por gate sobre las proyecciones 4D.

    A diferencia del v0 (prior como post-paso) y de la guía ungated por-gate
    (que aplasta el movimiento, −93% H1-2 medido en NITIDA II), el prior es
    LOCAL: suaviza cada voxel hacia la mediana de SU vecindario, sin imponer un
    valor temporal fijo. Así controla el ruido de la recon sin tocar la
    oscilación cardíaca (H1-2) que mide la fase y la FEVI.

    ``rescale`` (default True): tras cada gate, reescala el volumen para que su
    suma total de cuentas coincida con la del OSEM sin prior. El prior mediana
    baja levemente las cuentas totales; sin reescalar se introduciría una
    distorsión en la curva de volumen (FEVI). La fase (FFT sin DC) es
    insensible a un factor global por gate, pero la FEVI no.

    Devuelve (n_gates, H, W, W), mismo shape que reconstruct_gated_projection_volume.
    """
    from core.raw_reconstruction import reconstruct_gated_projection_volume

    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibio {proj.shape}")

    vol_map = reconstruct_gated_projection_volume(
        proj, angles_deg, method="osem",
        projection_filter=None, iterations=iterations, subsets=subsets,
        psf=psf, slice_range=slice_range,
        map_beta=beta, map_prior=prior, map_prior_size=prior_size,
    )
    if not rescale:
        return vol_map

    # Reescalado por gate: conserva la suma de cuentas de cada gate (FEVI).
    # Referencia = OSEM sin prior sobre las mismas proyecciones.
    vol_ref = reconstruct_gated_projection_volume(
        proj, angles_deg, method="osem",
        projection_filter=None, iterations=iterations, subsets=subsets,
        psf=psf, slice_range=slice_range,
    )
    n_gates = int(proj.shape[0])
    out = np.empty_like(vol_map)
    for g in range(n_gates):
        s_map = float(vol_map[g].sum())
        s_ref = float(vol_ref[g].sum())
        out[g] = vol_map[g] * (s_ref / s_map) if s_map > 0.0 else vol_map[g]
    return out


def nitida3_map_osem_gated_adaptive(
    projections: np.ndarray,
    angles_deg: np.ndarray,
    *,
    iterations: int = 2,
    subsets: int = 4,
    beta0: float = 0.4,
    psf=None,
    slice_range: tuple[int, int] | None = None,
    rescale: bool = True,
    ref_volume: np.ndarray | None = None,
    progress_callback=None,
) -> np.ndarray:
    """MAP-OSEM gated NÍTIDA III con Pilar C (Matched Recovery / SNR adaptativa).

    A diferencia de ``nitida3_map_osem_gated`` (prior mediana global, que a β
    alto baja el movimiento H1 ~19%), acá el prior es de SUAVIDAD (Huber) con un
    beta ESPACIAL adaptativo por SNR local: fuerte en la pared bien perfundida
    (alta SNR -> limpia), flojo donde la señal ES el movimiento o el ruido. El
    prior de suavidad penaliza el contraste espacial, no el brillo, así que la
    pared que oscila (vecinos que oscilan igual, contraste bajo) no se frena.

    ``rescale`` conserva la suma de cuentas por gate (FEVI). Si se pasa
    ``ref_volume`` (p.ej. el gated ya reconstruido por el pipeline), se usa como
    referencia de cuentas y se EVITA una segunda reconstrucción OSEM completa
    (casi duplica la velocidad). Si no, se reconstruye la referencia (lento).
    ``progress_callback`` (opcional): callable(fracción 0..1, mensaje).
    Devuelve (n_gates, H, W, W), mismo shape que reconstruct_gated_projection_volume.
    """
    from core.raw_reconstruction import reconstruct_gated_projection_volume

    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibio {proj.shape}")

    vol_map = reconstruct_gated_projection_volume(
        proj, angles_deg, method="osem",
        projection_filter=None, iterations=iterations, subsets=subsets,
        psf=psf, slice_range=slice_range,
        map_adaptive=True, map_beta0=beta0,
        progress=progress_callback,
    )
    if not rescale:
        return vol_map

    # Reescalado por gate: conserva la suma de cuentas de cada gate (FEVI).
    # Referencia: la provista (rápido) o una reconstrucción OSEM sin prior (lento).
    if ref_volume is not None:
        vol_ref = np.asarray(ref_volume, dtype=np.float64)
    else:
        vol_ref = reconstruct_gated_projection_volume(
            proj, angles_deg, method="osem",
            projection_filter=None, iterations=iterations, subsets=subsets,
            psf=psf, slice_range=slice_range,
        )
    n_gates = int(proj.shape[0])
    out = np.empty_like(vol_map)
    for g in range(n_gates):
        s_map = float(vol_map[g].sum())
        s_ref = float(vol_ref[g].sum())
        out[g] = vol_map[g] * (s_ref / s_map) if s_map > 0.0 else vol_map[g]
    return out
