"""Pipeline crudo gated -> motion correction -> reconstruccion.

Este modulo cierra el contrato geometrico del flujo Odyssey/Xeleris:
corregir movimiento sobre proyecciones, reconstruir UngGat con alta estadistica y
aplicar los mismos shifts geometricos al gated, usando filtros separados para cada rama.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import rotate

from core.raw_projections import apply_shifts_to_projections, motion_correct_projections, ungate_projections


@dataclass(frozen=True)
class ProjectionFilterConfig:
    """Filtro aplicado a proyecciones antes de FBP.

    kind acepta: none, lowpass, butterworth, wiener. Se aplica como filtro 2D
    radial sobre cada proyeccion (ejes detector H y W), replicando el pre-filtro
    de GE Xeleris ECToolbox, antes de reconstruir con FBP. ``cutoff`` en fraccion
    de Nyquist (0-1); ``order`` = exponente Butterworth (misma convencion que Xeleris).
    """
    kind: str = "butterworth"
    cutoff: float = 0.52
    order: int = 5
    noise_power: float = 0.02


@dataclass(frozen=True)
class RawReconConfig:
    """Configuracion del pipeline raw gated.

    reconstruction_method soporta 'fbp', 'osem' y 'mlem' (motor iterativo CPU
    implementado en este módulo). OSEM admite recuperación de resolución (RR)
    dependiente de profundidad pasando un PsfModel (ver core.resolution_recovery).
    """
    reconstruction_method: str = "fbp"
    fbp_filter_name: str = "ramp"
    # Método independiente para la rama gated. Si es None, la rama gated usa
    # reconstruction_method (que es el método de la rama ungated/perfusión).
    gated_method: str | None = None
    # Calcado de Xeleris ECToolbox (FBP) para este protocolo cardiaco:
    # ungated Butterworth cutoff 0.52 / orden 5; gated cutoff 0.40 / orden 10
    # (cutoff en fraccion de Nyquist).
    # NOTA: con metodo iterativo (OSEM/MLEM) el Butterworth se aplica
    # POST-reconstruccion como filtro 3D radial (pre-filtrar el sinograma
    # rompe el modelo Poisson y genera ruido correlacionado).
    ungated_filter: ProjectionFilterConfig = field(default_factory=lambda: ProjectionFilterConfig("butterworth", 0.52, 5))
    gated_filter: ProjectionFilterConfig = field(default_factory=lambda: ProjectionFilterConfig("butterworth", 0.40, 10))
    iterative_iterations: int = 4
    osem_subsets: int = 4
    fevi_slice_step_px: int = 1
    display_slice_step_px: int = 2
    # --- Recuperación de resolución NÍTIDA (OmniRes) ---
    # Si resolution_recovery=True y el método es iterativo (osem/mlem), la
    # reconstrucción modela la PSF dependiente de profundidad del colimador
    # (psf_model, un core.resolution_recovery.PsfModel). No aplica a FBP.
    # Por rama: rr_ungated / rr_gated permiten NÍTIDA solo en una rama (p.ej.
    # ungated nítido para perfusión, gated FBP/NITIDA III para movimiento).
    # `resolution_recovery` global se mantiene por compatibilidad: si es True y
    # las por-rama no se setearon, aplica a ambas.
    resolution_recovery: bool = False
    rr_ungated: bool | None = None   # None = hereda resolution_recovery
    rr_gated: bool | None = None     # None = hereda resolution_recovery
    psf_model: object | None = None
    # Post-filtro gaussiano 3D opcional (control de ruido post-recon). 0 = off.
    # Sigma en píxeles; es la contraparte de ruido de la recuperación de resolución.
    # Por rama: si las por-rama son None, se usa post_filter_sigma_px (global).
    post_filter_sigma_px: float = 0.0
    post_filter_sigma_ungated_px: float | None = None
    post_filter_sigma_gated_px: float | None = None
    # Post-filtro alternativo Butterworth 3D radial (protocolo Xeleris óseo:
    # OSEM 8×4 + Butter 0.35/5 post-recon). kind: "gaussian" | "butterworth".
    # Con "butterworth" se ignoran las sigmas y se usa cutoff/orden (frac. Nyquist).
    post_filter_kind: str = "gaussian"
    post_filter_cutoff: float = 0.35
    post_filter_order: int = 5
    # --- Feta axial (cilindro del corazón) ---
    # Rango de cortes axiales (z = fila del detector) a reconstruir, inclusive:
    # (z0, z1). None = volumen completo. En SPECT paralelo cada corte z se
    # reconstruye independiente de su fila del sinograma, así que limitar la
    # banda es EXACTO (no aproxima) y más rápido: las lonjas fuera de la banda
    # quedan en cero (feta). Los cortes dentro de la banda son idénticos a los
    # del volumen completo.
    recon_slice_range: tuple[int, int] | None = None
    # --- NITIDA II: denoiser temporal/espaciotemporal por armónicos ---
    # Denoiser gated de bajo conteo aplicado post-recon SOLO al volumen gated
    # (core.nitida2). Preserva el movimiento cardíaco (bandas de baja frecuencia)
    # y elimina el ruido de banda alta. Modos:
    #   "none"          -> desactivado.
    #   "temporal"      -> filtro temporal por armónicos (conserva DC..H_n).
    #   "spatiotemporal"-> además suaviza espacialmente DC (fuerte) y H1-2 (suave).
    nitida2_mode: str = "none"
    nitida2_harmonics: int = 2
    nitida2_band_sigma: float = 0.7
    nitida2_dc_radius: int = 2
    nitida2_dc_eps: float = 0.01
    # --- NITIDA III: MAP-OSEM gated con Pilar C (SNR adaptativa) ---
    # Reconstruye el GATED con un prior de suavidad Huber cuyo beta es ESPACIAL
    # (fuerte donde la SNR es alta = pared, flojo donde la señal ES el
    # movimiento/ruido). Limpia la pared sin aplastar la oscilación cardíaca
    # (H1). Medido en el estudio 5s real: SNR x1.49 conservando H1 x0.88.
    # OFF por defecto. Es una RECONSTRUCCIÓN del gated (no un post-filtro):
    # reemplaza al OSEM/FBP del gated cuando está activa.
    nitida3_enabled: bool = False
    nitida3_beta0: float = 0.6     # tope de beta espacial (pared)
    nitida3_iterations: int = 2
    nitida3_subsets: int = 4
    # --- NITIDA 4D (4D-OSEM): prior TEMPORAL entre gates ---
    # Reconstruye los gates JUNTOS con un prior de suavidad temporal Huber
    # DENTRO del update OSEM (Green OSL). A diferencia de promediar gates
    # (motion-frozen, que congela el latido), el prior temporal solo frena el
    # ruido incorrelado; la contracción (borde temporal real) se conserva.
    # Reescalado por gate para conservar cuentas (FEVI). OFF por defecto.
    nitida4d_enabled: bool = False
    nitida4d_beta_temporal: float = 0.3   # fuerza del acoplamiento temporal
    nitida4d_delta_temporal: float = 0.05  # umbral Huber (fracción del rango)
    nitida4d_iterations: int = 4
    nitida4d_subsets: int = 4
    # --- FBP_CLEAN: denoise Poisson en sinograma + realce por resta ---
    # Denoise bilateral de las proyecciones ANTES del FBP (ataca las estrías en
    # la raíz, banco 023/025) + realce de cavidad/bordes restando una fracción
    # de la versión muy suavizada (unsharp mask, idea del usuario, banco 026/027).
    # Aplica solo a la rama FBP. 0 = desactivado.
    fbp_clean_sigma_color: float = 0.0   # 0=off; 0.04 calibrado (banco 025)
    fbp_clean_sharpen_k: float = 0.5     # factor de realce (0.3–0.7, default 0.5)
    fbp_clean_blur_sigma_color: float = 0.24  # versión muy suavizada para la resta
    # --- Denoise+ UNGATED: denoise de sinograma + realce por resta ---
    # El ungated (alto conteo) también sufre scatter/fondo que RELLENA la cavidad
    # (medido: contraste cav/pared 0.68 vs 0.79 con Denoise+, harness 037). Este
    # paso aplica al ungated el MISMO tratamiento que FBP_CLEAN al gated: denoise
    # bilateral del sinograma + realce por resta (k=0.5). Abre la cavidad y afina
    # la pared. OFF por defecto. Independiente de FBP_CLEAN (que es gated).
    ungated_denoise_plus: bool = False
    ungated_denoise_plus_k: float = 0.20  # óptimo medido 0.20 (abre cavidad sin comer pared)
    # --- Denoise+ GATED: mismo tratamiento pero para la rama gated ---
    # FBP_CLEAN hace esto pero SOLO si la rama gated es FBP (fue diseñado para
    # atacar las estrías de FBP). Como el gated ahora defaultea a OSEM (que no
    # tiene estrías pero igual pierde la cavidad por bajo conteo), este paso
    # aplica el MISMO realce por resta al gated CON CUALQUIER MÉTODO: denoise
    # bilateral del sinograma gated + doble recon (nítida + difusa) + resta.
    # k=0.5 (calibrado para bajo conteo, más agresivo que el 0.20 del ungated).
    # OFF por defecto. Independiente de FBP_CLEAN.
    gated_denoise_plus: bool = False
    gated_denoise_plus_k: float = 0.50  # default 0.50 (bajo conteo tolera más realce)
    # --- Motion-frozen 3D (post-recon, pre-reorientación) ---
    # Alinea cada gate del volumen 4D al end-diastole y promedia. Recupera la
    # nitidez de un gate con todas las cuentas (el "cardiac morphing" físico).
    # OFF por defecto. Reemplaza al ungated para display/montaje; el gated
    # original se conserva para FEVI/volúmenes/asincronía.
    # Método default "stable": alinea por el contexto (tórax/hígado/fondo)
    # excluyendo el corazón, así corrige el desplazamiento del paciente SIN
    # arrastrar el latido (el centroide global congela la contracción).
    motion_frozen: bool = False
    motion_frozen_method: str = "rigid"   # "rigid" | "displacement" | "stable"
    motion_frozen_ref_gate: int | None = None  # None = auto (end-diastole)
    # Motion-frozen POR GATE: NO VIABLE FÍSICAMENTE (promediar gates alineados
    # congela el latido; verificado en fantoma 2026-08-13). Código conservado
    # en core/motion_frozen.py pero sin UI. Para mejorar gates sin promediar:
    # NITIDA II/III.
    motion_frozen_per_gate: bool = False
    # --- Descuento de fondo automático (pre-recon, sobre el sinograma) ---
    # Resta un piso de cuentas medido automáticamente en la zona de bajo fondo
    # (pulmón/tejido blando; excluye el aire por umbral de cuerpo y las
    # vísceras calientes por percentil bajo, ver core.raw_background). Se resta
    # de TODAS las proyecciones, incluido el corazón: el fondo (scatter +
    # actividad difusa) es aditivo y también está debajo del miocardio.
    # Aumenta el contraste cavidad/pared. OFF por defecto.
    background_subtract: bool = False
    # --- Corrección de SCATTER por ventana energética (pre-recon) ---
    # Si la adquisición es dual EM/SC (p.ej. Infinia exporta *_EM_* y *_SC_*),
    # el loader adjunta scatter_projections al estudio. Si el usuario lo
    # habilita, se resta k*SC de CADA proyección ANTES de reconstruir:
    #   P_primaria = P_EM - k * P_SC   (dual-window; TEW cuando haya anchos
    #   de ventana en el DICOM). k=1.0 por defecto (placeholder; calibrar con
    #   fantoma/estudio real). Se aplica a gated y ungated por igual.
    scatter_subtract: bool = False
    scatter_k: float = 1.0
    # --- Corrección de atenuación (AC) en reconstrucción iterativa ---
    # Requiere proveer μ-map en grilla de reconstrucción al pipeline.
    # Aplica en OSEM/MLEM (forward/backprojection). FBP no usa AC.
    attenuation_correction: bool = False
    attenuation_mu_scale: float = 1.0


@dataclass
class RawReconResult:
    original_projections: np.ndarray
    corrected_projections: np.ndarray
    ungated_projections: np.ndarray
    ungated_volume: np.ndarray
    gated_volume: np.ndarray
    phase_cube: np.ndarray
    display_cube: np.ndarray
    shifts_y: np.ndarray
    shifts_x: np.ndarray
    config: RawReconConfig
    motion_result: dict
    notes: list[str] = field(default_factory=list)
    # True si se aplicó espejo L/R (flip en eje x del volumen reconstruido)
    # para converger orientación CW/CCW a una convención canónica.
    flip_x_applied: bool = False
    # Motion-frozen: volumen 4D alineado y promediado (None si no se pidió).
    ungated_volume_mf: np.ndarray | None = None
    # Motion-frozen por gate: 4D de "cine nítido" (None si no se pidió).
    gated_volume_mf_per_gate: np.ndarray | None = None
    # Volumen ungated ANTES del post-filtro gaussiano (para toggle con/sin filtro).
    ungated_volume_unfiltered: np.ndarray | None = None


def _normalize_filter_kind(kind: str) -> str:
    key = str(kind or "none").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "no": "none",
        "off": "none",
        "sin": "none",
        "sin_filtro": "none",
        "low_pass": "lowpass",
        "lowpass": "lowpass",
        "butter": "butterworth",
        "bw": "butterworth",
        "butterworth": "butterworth",
        "wiener": "wiener",
    }
    return aliases.get(key, key)


def _validate_config(config: RawReconConfig) -> RawReconConfig:
    method = str(config.reconstruction_method or "fbp").strip().lower()
    if method not in {"fbp", "osem", "mlem"}:
        raise ValueError("reconstruction_method debe ser 'fbp', 'osem' o 'mlem'")
    if config.gated_method is not None and str(config.gated_method).strip().lower() not in {"fbp", "osem", "mlem"}:
        raise ValueError("gated_method debe ser 'fbp', 'osem', 'mlem' o None")
    if int(config.iterative_iterations) < 1:
        raise ValueError("iterative_iterations debe ser >= 1")
    if int(config.osem_subsets) < 1:
        raise ValueError("osem_subsets debe ser >= 1")
    if int(config.fevi_slice_step_px) != 1:
        raise ValueError("fevi_slice_step_px debe ser 1 para calcular FEVI con todos los cortes reconstruidos")
    if int(config.display_slice_step_px) < 1:
        raise ValueError("display_slice_step_px debe ser >= 1")
    nitida2_mode = str(getattr(config, "nitida2_mode", "none") or "none").strip().lower()
    if nitida2_mode not in {"none", "temporal", "spatiotemporal"}:
        raise ValueError("nitida2_mode debe ser 'none', 'temporal' o 'spatiotemporal'")
    return config


def _butterworth_3d(volume: np.ndarray, cutoff: float, order: int) -> np.ndarray:
    """Butterworth 3D radial post-recon (cutoff en fracción de Nyquist).

    Misma convención de respuesta que filter_projections: corta el ruido de alta
    frecuencia preservando frecuencias medias (bordes óseos), a diferencia del
    gaussiano que atenúa todo el espectro.
    """
    vol = np.asarray(volume, dtype=np.float64)
    cutoff = float(np.clip(float(cutoff), 1e-4, 1.0))
    order = max(1, int(order))
    fz = np.abs(np.fft.fftfreq(vol.shape[0])) / 0.5
    fy = np.abs(np.fft.fftfreq(vol.shape[1])) / 0.5
    fx = np.abs(np.fft.fftfreq(vol.shape[2])) / 0.5
    freq = np.sqrt(fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2)
    resp = 1.0 / np.sqrt(1.0 + (freq / cutoff) ** (2 * order))
    return np.fft.ifftn(np.fft.fftn(vol) * resp).real


def filter_projections(projections: np.ndarray, config: ProjectionFilterConfig) -> np.ndarray:
    """Aplica filtro frecuencial/espacial simple sobre el eje detector.

    Parameters
    ----------
    projections : ndarray (angles,H,W) o (gates,angles,H,W)
    config : ProjectionFilterConfig

    Returns
    -------
    ndarray con la misma forma.
    """
    arr = np.asarray(projections, dtype=np.float64)
    kind = _normalize_filter_kind(config.kind)
    if kind == "none":
        return arr.copy()
    if arr.ndim not in (3, 4):
        raise ValueError(f"projections debe ser 3D o 4D; recibio {arr.shape}")

    cutoff = float(np.clip(float(config.cutoff), 1e-4, 1.0))
    order = max(1, int(config.order))

    if kind in {"lowpass", "butterworth", "wiener"}:
        # GE Xeleris ECToolbox aplica el pre-filtro como Butterworth 2D radial sobre
        # cada proyeccion (ejes detector H y W). Replicamos esa geometria en vez de
        # filtrar solo el eje columna.
        h = int(arr.shape[-2])
        w = int(arr.shape[-1])
        fy = np.abs(np.fft.fftfreq(h)) / 0.5
        fx = np.abs(np.fft.fftfreq(w)) / 0.5
        freq = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        if kind == "lowpass":
            response = (freq <= cutoff).astype(np.float64)
        elif kind == "butterworth":
            response = 1.0 / np.sqrt(1.0 + (freq / max(cutoff, 1e-6)) ** (2 * order))
        else:
            # Wiener multiplicativo simple: conserva bajas frecuencias y reduce ruido fino.
            base = 1.0 / np.sqrt(1.0 + (freq / max(cutoff, 1e-6)) ** (2 * order))
            noise = max(1e-6, float(config.noise_power))
            response = (base * base) / (base * base + noise)
        resp = response.reshape((1,) * (arr.ndim - 2) + (h, w))
        fft = np.fft.fft2(arr, axes=(-2, -1))
        return np.fft.ifft2(fft * resp, axes=(-2, -1)).real

    raise ValueError("Filtro no soportado: use none, lowpass, butterworth o wiener")


def _sub_progress(progress, index: int, count: int):
    """Deriva un callback de progreso para el tramo [index/count, (index+1)/count].

    ``progress`` es un callable(global_fraction) o None. El resultado es un
    callable(local_fraction 0..1) que mapea el avance local del sub-bloque
    (p.ej. un gate) al tramo global correspondiente. Devuelve None si no hay
    progress, para que el bucle interno no haga trabajo extra.
    """
    if progress is None or count <= 0:
        return None
    start = index / count
    span = 1.0 / count

    def _inner(local_fraction: float) -> None:
        progress(start + span * max(0.0, min(1.0, float(local_fraction))))

    return _inner


def _resolve_slice_range(slice_range: tuple[int, int] | None, height: int) -> tuple[int, int]:
    """Normaliza (z0, z1) inclusive a [0, height-1]. None => banda completa."""
    if slice_range is None:
        return 0, height - 1
    z0, z1 = int(slice_range[0]), int(slice_range[1])
    if z0 > z1:
        z0, z1 = z1, z0
    z0 = max(0, min(z0, height - 1))
    z1 = max(0, min(z1, height - 1))
    return z0, z1


def reconstruct_fbp_volume(
    projections: np.ndarray,
    angles_deg: np.ndarray | None = None,
    *,
    projection_filter: ProjectionFilterConfig | None = None,
    fbp_filter_name: str = "ramp",
    slice_range: tuple[int, int] | None = None,
    output_size: int | None = None,
    progress=None,
) -> np.ndarray:
    """Reconstruye un volumen transaxial por FBP desde proyecciones 3D.

    Entrada: (n_angles, H, W). Salida: (H, out, out), un corte por fila axial del detector.
    ``slice_range`` (opcional): (z0, z1) inclusive; solo reconstruye esa banda
    axial (feta), el resto queda en cero. ``output_size`` (opcional): tamaño de
    la matriz de salida en el plano (default = ancho del detector W). >W =
    matriz fina (voxel más chico) para resolver la cavidad. ``progress``: callable.
    """
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 3:
        raise ValueError(f"projections debe ser 3D (angles,H,W); recibio {proj.shape}")
    if projection_filter is not None:
        proj = filter_projections(proj, projection_filter)
    n_angles, height, width = proj.shape
    out_size = int(output_size) if output_size else width
    if angles_deg is None:
        theta = np.linspace(0.0, 360.0, n_angles, endpoint=False)
    else:
        theta = np.asarray(angles_deg, dtype=np.float64)
        if theta.size != n_angles:
            raise ValueError(f"angles_deg debe tener {n_angles} valores; recibio {theta.size}")

    volume = np.zeros((height, out_size, out_size), dtype=np.float64)
    z0, z1 = _resolve_slice_range(slice_range, height)
    n_band = z1 - z0 + 1
    for done, slice_idx in enumerate(range(z0, z1 + 1), start=1):
        sinogram = proj[:, slice_idx, :].T
        volume[slice_idx] = _iradon_or_fallback(sinogram, theta, filter_name=str(fbp_filter_name), output_size=out_size)
        if progress is not None and n_band:
            progress(done / n_band)
    return volume


def _rotate_volume_z(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rota todos los cortes axiales a la vez (una llamada C, ejes y-x)."""
    return rotate(volume, angle=float(angle_deg), axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0, prefilter=False)


def _iterative_reconstruct_volume_fast(
    proj: np.ndarray,
    theta: np.ndarray,
    *,
    iterations: int,
    subsets: int,
    slice_range: tuple[int, int] | None = None,
    progress=None,
    mu_map_3d: np.ndarray | None = None,
    mu_scale: float = 1.0,
    px_cm: float = 1.0,
) -> np.ndarray:
    """MLEM/OSEM volumétrico en float32: geometría idéntica al path por corte.

    En SPECT paralelo cada corte axial es independiente, así que rotar el
    volumen (z,y,x) alrededor de z equivale a rotar cada corte 2D por separado;
    hacerlo en una sola llamada elimina el overhead Python de
    cortes × iteraciones × ángulos rotaciones individuales (~18k para 128³).
    Con ``mu_map_3d`` modela atenuación en el update (AC iterativa): los rayos
    integran sobre axis=1 y tau acumula µ hacia el detector (lado y=N-1).
    """
    measured = np.clip(np.asarray(proj, dtype=np.float32), 0.0, None)
    n_angles, height, width = measured.shape
    theta = np.asarray(theta, dtype=np.float64)
    z0, z1 = _resolve_slice_range(slice_range, height)
    # Volumen de trabajo (banda de cortes): (nz, width, width)
    band = measured[:, z0:z1 + 1, :]
    nz = band.shape[1]
    if not np.any(band > 0):
        out = np.zeros((height, width, width), dtype=np.float64)
        return out

    mu_band = None
    if mu_map_3d is not None:
        mu_band = np.clip(np.asarray(mu_map_3d, dtype=np.float32)[z0:z1 + 1], 0.0, None)
        mu_band = mu_band * np.float32(max(0.0, float(mu_scale)))
        _px = np.float32(max(1e-6, float(px_cm)))

    def _transmission(angle: float) -> np.ndarray | None:
        if mu_band is None:
            return None
        mu_rot = _rotate_volume_z(mu_band, -float(angle))
        tau = np.cumsum(mu_rot[:, ::-1, :], axis=1)[:, ::-1, :] * _px
        return np.exp(-tau, dtype=np.float32)

    image = np.full((nz, width, width), max(float(band.mean()), 1.0), dtype=np.float32)
    subset_count = max(1, min(int(subsets), int(theta.size)))
    angle_indices = np.arange(theta.size)
    eps = np.float32(1e-6)

    # Sensibilidad por subset: retroproyección volumétrica de A^T 1 (con AC =
    # retroproyección de la transmisión). ndimage libera el GIL: hilos.
    from concurrent.futures import ThreadPoolExecutor
    import os as _os
    n_workers = max(2, min(8, int(_os.cpu_count() or 4)))
    ones_slab = np.ones((nz, width, width), dtype=np.float32)
    sens: dict[int, np.ndarray] = {}

    def _sensitivity_part(angle: float) -> np.ndarray:
        trans = _transmission(angle)
        slab = ones_slab if trans is None else trans
        return _rotate_volume_z(slab, float(angle))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for subset_id in range(subset_count):
            idx = angle_indices[subset_id::subset_count]
            if idx.size == 0:
                continue
            parts = list(pool.map(_sensitivity_part, theta[idx]))
            acc = np.sum(parts, axis=0, dtype=np.float32)
            acc *= np.float32(np.pi / (2.0 * float(idx.size)))
            sens[subset_id] = np.maximum(acc, eps)

        total_updates = max(1, int(iterations)) * subset_count
        done_updates = 0
        for _iter in range(max(1, int(iterations))):
            for subset_id in range(subset_count):
                idx = angle_indices[subset_id::subset_count]
                if idx.size == 0:
                    continue

                def _angle_contribution(args):
                    k, a = args
                    rot = _rotate_volume_z(image, -float(a))
                    trans = _transmission(a)
                    if trans is not None:
                        prof = (rot * trans).sum(axis=1)
                    else:
                        prof = rot.sum(axis=1)
                    ratio_prof = band[k] / np.maximum(prof, eps)
                    slab = np.repeat(ratio_prof[:, np.newaxis, :], width, axis=1)
                    if trans is not None:
                        slab = slab * trans
                    return _rotate_volume_z(slab, float(a))

                parts = list(pool.map(_angle_contribution, zip(idx.tolist(), theta[idx])))
                correction = np.sum(parts, axis=0, dtype=np.float32)
                correction *= np.float32(np.pi / (2.0 * float(idx.size)))
                image *= correction / sens[subset_id]
                np.clip(image, 0.0, None, out=image)
                done_updates += 1
                if progress is not None:
                    progress(done_updates / total_updates)

    out = np.zeros((height, width, width), dtype=np.float64)
    out[z0:z1 + 1] = image.astype(np.float64)
    return out


def reconstruct_projection_volume(
    projections: np.ndarray,
    angles_deg: np.ndarray | None = None,
    *,
    method: str = "fbp",
    projection_filter: ProjectionFilterConfig | None = None,
    fbp_filter_name: str = "ramp",
    iterations: int = 4,
    subsets: int = 4,
    sensitivity_cache: dict[int, np.ndarray] | None = None,
    psf=None,
    slice_range: tuple[int, int] | None = None,
    progress=None,
    map_beta: float = 0.0,
    map_prior: str = "none",
    map_prior_size: int = 3,
    map_adaptive: bool = False,
    map_beta0: float = 0.4,
    attenuation_mu_map: np.ndarray | None = None,
    attenuation_mu_scale: float = 1.0,
    attenuation_pixel_size_cm: float = 1.0,
) -> np.ndarray:
    """Reconstruye volumen desde proyecciones 3D con FBP/MLEM/OSEM.

    ``progress`` (opcional): callable(local_fraction 0..1) para reportar avance.

    ``sensitivity_cache`` (opcional): si se pasa (ver `_build_sensitivity_cache`),
    se reutiliza tal cual en vez de recalcularla. Sirve para compartir las
    imágenes de sensibilidad entre gates cuando se reconstruye un estudio
    gated completo (mismos ángulos/geometría en todos los gates).

    ``psf`` (opcional, PsfModel de core.resolution_recovery): activa la
    recuperación de resolución (RR) dependiente de profundidad en el path
    iterativo (OSEM/MLEM). Sin psf, el comportamiento es idéntico al previo.

    ``map_beta``/``map_prior``/``map_prior_size`` (opcionales): MAP-OSEM (Green
    OSL) con prior edge-preserving dentro del update (NÍTIDA III). 0/"none" =
    OSEM puro (comportamiento previo).
    """
    method_key = str(method or "fbp").strip().lower()
    if method_key == "fbp":
        return reconstruct_fbp_volume(
            projections, angles_deg, projection_filter=projection_filter, fbp_filter_name=fbp_filter_name,
            slice_range=slice_range, progress=progress,
        )
    if method_key not in {"mlem", "osem"}:
        raise ValueError("method debe ser 'fbp', 'mlem' u 'osem'")
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 3:
        raise ValueError(f"projections debe ser 3D (angles,H,W); recibio {proj.shape}")
    # En iterativo el Butterworth NO se pre-aplica al sinograma (rompe el modelo
    # Poisson de OSEM/MLEM y genera ruido correlacionado): se aplica POST-recon
    # como filtro 3D radial. Los demás kinds (lowpass/wiener) conservan el
    # comportamiento pre-filtro histórico.
    post_butterworth: tuple[float, int] | None = None
    if projection_filter is not None:
        _pf_kind = _normalize_filter_kind(projection_filter.kind)
        if _pf_kind == "butterworth":
            post_butterworth = (float(projection_filter.cutoff), int(projection_filter.order))
        elif _pf_kind != "none":
            proj = filter_projections(proj, projection_filter)
    proj = np.clip(proj, 0.0, None)
    n_angles, height, width = proj.shape
    theta = np.linspace(0.0, 360.0, n_angles, endpoint=False) if angles_deg is None else np.asarray(angles_deg, dtype=np.float64)
    if theta.size != n_angles:
        raise ValueError(f"angles_deg debe tener {n_angles} valores; recibio {theta.size}")
    mu_map_3d = None
    if attenuation_mu_map is not None:
        mu_map_3d = np.asarray(attenuation_mu_map, dtype=np.float64)
        if mu_map_3d.shape != (height, width, width):
            raise ValueError(
                "attenuation_mu_map debe tener shape (H,W,W) de la grilla de reconstrucción; "
                f"esperado {(height, width, width)}, recibido {mu_map_3d.shape}"
            )
    effective_subsets = int(subsets) if method_key == "osem" else 1
    use_map = (float(map_beta) > 0.0 and str(map_prior).lower() not in ("none", "")) or bool(map_adaptive)
    # Camino rápido volumétrico (float32, rotación 3D por ángulo): mismo
    # resultado geométrico que el bucle por corte, sin overhead Python por slice.
    # Soporta AC (mu_map); quedan fuera PSF (RR) y priors MAP.
    if psf is None and not use_map:
        out_fast = _iterative_reconstruct_volume_fast(
            proj,
            theta,
            iterations=int(iterations),
            subsets=effective_subsets,
            slice_range=slice_range,
            progress=progress,
            mu_map_3d=mu_map_3d,
            mu_scale=float(attenuation_mu_scale),
            px_cm=float(attenuation_pixel_size_cm),
        )
        return _apply_post_butterworth(out_fast, post_butterworth, slice_range)
    # La imagen de "sensibilidad" (retroproyección de un sinograma de unos) NO
    # depende de los datos medidos ni de la iteración actual: solo depende de
    # la geometría (ángulos del subset + tamaño de detector/salida), que es
    # IDÉNTICA para todos los cortes axiales y todas las iteraciones. Antes se
    # recalculaba `height * iterations * subsets` veces (p.ej. 64*4*4=1024
    # retroproyecciones redundantes); acá se calcula una sola vez por subset
    # (a lo sumo `subsets` veces) y se reutiliza para todos los cortes e
    # iteraciones. Es una optimización exacta (no una aproximación): el
    # resultado numérico es idéntico, solo se evita repetir el mismo cálculo.
    if sensitivity_cache is None:
        sensitivity_cache = _build_sensitivity_cache(
            theta, subsets=effective_subsets, detector_size=width, output_size=width, psf=psf
        )
    out = np.zeros((height, width, width), dtype=np.float64)
    z0, z1 = _resolve_slice_range(slice_range, height)
    n_band = z1 - z0 + 1
    for done, slice_idx in enumerate(range(z0, z1 + 1), start=1):
        sinogram = proj[:, slice_idx, :].T
        mu_slice = None if mu_map_3d is None else mu_map_3d[slice_idx]
        out[slice_idx] = _iterative_reconstruct_slice(
            sinogram,
            theta,
            output_size=width,
            iterations=int(iterations),
            subsets=effective_subsets,
            sensitivity_cache=sensitivity_cache if mu_slice is None else None,
            psf=psf,
            map_beta=map_beta,
            map_prior=map_prior,
            map_prior_size=map_prior_size,
            map_adaptive=map_adaptive,
            map_beta0=map_beta0,
            attenuation_mu_map_2d=mu_slice,
            attenuation_mu_scale=float(attenuation_mu_scale),
            attenuation_pixel_size_cm=float(attenuation_pixel_size_cm),
        )
        if progress is not None and n_band:
            progress(done / n_band)
    return _apply_post_butterworth(out, post_butterworth, slice_range)


def _apply_post_butterworth(
    volume: np.ndarray,
    post_butterworth: tuple[float, int] | None,
    slice_range: tuple[int, int] | None,
) -> np.ndarray:
    """Post-filtro Butterworth 3D para OSEM/MLEM, respetando la feta axial."""
    if post_butterworth is None:
        return volume
    cutoff, order = post_butterworth
    if slice_range is None:
        return _butterworth_3d(volume, cutoff, order)
    # Filtrar solo la banda reconstruida: fuera de ella el volumen es cero y
    # el FFT 3D sobre el volumen completo metería ringing en los bordes.
    z0, z1 = _resolve_slice_range(slice_range, int(volume.shape[0]))
    out = np.asarray(volume, dtype=np.float64).copy()
    out[z0:z1 + 1] = _butterworth_3d(out[z0:z1 + 1], cutoff, order)
    return out


def _build_sensitivity_cache(
    theta: np.ndarray, *, subsets: int, detector_size: int, output_size: int, psf=None
) -> dict[int, np.ndarray]:
    """Precalcula, una sola vez, la imagen de sensibilidad de cada subset de
    ángulos (retroproyección de un sinograma de unos). Se reutiliza para
    todos los cortes axiales y todas las iteraciones de MLEM/OSEM. Si ``psf``
    se pasa, la sensibilidad incorpora la PSF (para OSEM con RR)."""
    subset_count = max(1, min(int(subsets), int(theta.size)))
    angle_indices = np.arange(theta.size)
    cache: dict[int, np.ndarray] = {}
    for subset_id in range(subset_count):
        idx = angle_indices[subset_id::subset_count]
        if idx.size == 0:
            continue
        theta_sub = theta[idx]
        ones_sub = np.ones((int(detector_size), theta_sub.size), dtype=np.float64)
        cache[subset_id] = _backproject_slice(ones_sub, theta_sub, output_size=output_size, psf=psf)
    return cache



def _iradon_or_fallback(sinogram: np.ndarray, theta: np.ndarray, *, filter_name: str, output_size: int) -> np.ndarray:
    """Usa scikit-image si está instalado; si no, FBP con rampa Ram-Lak propia."""
    try:
        from skimage.transform import iradon
        return np.asarray(iradon(sinogram, theta=theta, filter_name=filter_name, output_size=output_size), dtype=np.float64)
    except ModuleNotFoundError:
        # Fallback: aplicar el filtro rampa (imprescindible en FBP) antes de
        # retroproyectar. Sin rampa la retroproyección rellena la cavidad y el
        # miocardio sale como blob sólido.
        use_ramp = str(filter_name).lower() not in ("none", "")
        sino_f = _ramp_filter_sinogram(sinogram) if use_ramp else sinogram
        return _simple_backprojection(sino_f, theta, output_size=output_size)


def _ramp_filter_sinogram(sinogram: np.ndarray) -> np.ndarray:
    """Filtra cada proyección con la rampa Ram-Lak (|f|) en frecuencia."""
    sino = np.asarray(sinogram, dtype=np.float64)
    n_det = int(sino.shape[0])
    pad = max(64, int(2 ** np.ceil(np.log2(max(2 * n_det, 2)))))
    ramp = (2.0 * np.abs(np.fft.fftfreq(pad))).reshape(-1, 1)
    proj = np.zeros((pad, sino.shape[1]), dtype=np.float64)
    proj[:n_det] = sino
    ft = np.fft.fft(proj, axis=0) * ramp
    return np.real(np.fft.ifft(ft, axis=0))[:n_det]


def _simple_backprojection(
    sinogram: np.ndarray,
    theta: np.ndarray,
    *,
    output_size: int,
    psf=None,
    attenuation_mu_map_2d: np.ndarray | None = None,
    attenuation_mu_scale: float = 1.0,
    attenuation_pixel_size_cm: float = 1.0,
) -> np.ndarray:
    """Retroproyección paralela simple (adjunto). Sin filtro rampa.

    Usada por el path iterativo (MLEM/OSEM, donde el adjunto NO debe filtrarse)
    y por el fallback de FBP tras aplicar la rampa aparte. Si ``psf`` (PsfModel)
    se pasa, aplica el difuminado dependiente de profundidad (adjunto de la PSF,
    que es simétrica) para la recuperación de resolución.
    """
    sino = np.asarray(sinogram, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if sino.ndim != 2:
        raise ValueError(f"sinogram debe ser 2D (detector,angles); recibio {sino.shape}")
    if sino.shape[1] != theta.size:
        raise ValueError(f"theta debe tener {sino.shape[1]} valores; recibio {theta.size}")
    n_det = int(sino.shape[0])
    out_size = int(output_size)
    x_old = np.linspace(-1.0, 1.0, n_det)
    x_new = np.linspace(-1.0, 1.0, out_size)
    volume = np.zeros((out_size, out_size), dtype=np.float64)
    mu2 = None if attenuation_mu_map_2d is None else np.asarray(attenuation_mu_map_2d, dtype=np.float64)
    for idx, angle in enumerate(theta):
        profile = np.interp(x_new, x_old, sino[:, idx], left=0.0, right=0.0)
        slab = np.tile(profile.reshape(1, out_size), (out_size, 1))
        if mu2 is not None:
            mu_rot = rotate(mu2, angle=-float(angle), reshape=False, order=1, mode="constant", cval=0.0)
            # Mismo eje y lado que el forward: rayos sobre axis=0, detector en y=N-1.
            tau = np.cumsum(np.clip(mu_rot, 0.0, None)[::-1, :], axis=0)[::-1, :] * max(1e-6, float(attenuation_pixel_size_cm))
            slab = slab * np.exp(-max(0.0, float(attenuation_mu_scale)) * tau)
        if psf is not None:
            from core.resolution_recovery import variable_depth_gaussian
            slab = variable_depth_gaussian(slab, psf)
        volume += rotate(slab, angle=float(angle), reshape=False, order=1, mode="constant", cval=0.0)
    if theta.size:
        volume *= np.pi / (2.0 * float(theta.size))
    return volume


def _forward_project_slice(
    image: np.ndarray,
    theta: np.ndarray,
    *,
    detector_size: int,
    psf=None,
    attenuation_mu_map_2d: np.ndarray | None = None,
    attenuation_mu_scale: float = 1.0,
    attenuation_pixel_size_cm: float = 1.0,
) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    sino = np.zeros((int(detector_size), int(theta.size)), dtype=np.float64)
    x_old = np.linspace(-1.0, 1.0, img.shape[1])
    x_new = np.linspace(-1.0, 1.0, int(detector_size))
    mu2 = None if attenuation_mu_map_2d is None else np.asarray(attenuation_mu_map_2d, dtype=np.float64)
    for idx, angle in enumerate(theta):
        rot = rotate(img, angle=-float(angle), reshape=False, order=1, mode="constant", cval=0.0)
        if psf is not None:
            from core.resolution_recovery import variable_depth_gaussian
            rot = variable_depth_gaussian(rot, psf)
        if mu2 is not None:
            mu_rot = rotate(mu2, angle=-float(angle), reshape=False, order=1, mode="constant", cval=0.0)
            # El rayo integra sobre axis=0 (prof = sum(axis=0)); tau acumula µ
            # desde cada voxel hacia el detector (lado y=N-1) sobre ese MISMO eje.
            tau = np.cumsum(np.clip(mu_rot, 0.0, None)[::-1, :], axis=0)[::-1, :] * max(1e-6, float(attenuation_pixel_size_cm))
            trans = np.exp(-max(0.0, float(attenuation_mu_scale)) * tau)
            prof = (rot * trans).sum(axis=0)
        else:
            prof = rot.sum(axis=0)
        sino[:, idx] = np.interp(x_new, x_old, prof, left=0.0, right=0.0)
    return sino


def _backproject_slice(
    sinogram: np.ndarray,
    theta: np.ndarray,
    *,
    output_size: int,
    psf=None,
    attenuation_mu_map_2d: np.ndarray | None = None,
    attenuation_mu_scale: float = 1.0,
    attenuation_pixel_size_cm: float = 1.0,
) -> np.ndarray:
    return _simple_backprojection(
        sinogram,
        theta,
        output_size=output_size,
        psf=psf,
        attenuation_mu_map_2d=attenuation_mu_map_2d,
        attenuation_mu_scale=attenuation_mu_scale,
        attenuation_pixel_size_cm=attenuation_pixel_size_cm,
    )


def _iterative_reconstruct_slice(
    sinogram: np.ndarray,
    theta: np.ndarray,
    *,
    output_size: int,
    iterations: int,
    subsets: int,
    sensitivity_cache: dict[int, np.ndarray] | None = None,
    psf=None,
    map_beta: float = 0.0,
    map_prior: str = "none",
    map_prior_size: int = 3,
    map_adaptive: bool = False,
    map_beta0: float = 0.4,
    attenuation_mu_map_2d: np.ndarray | None = None,
    attenuation_mu_scale: float = 1.0,
    attenuation_pixel_size_cm: float = 1.0,
) -> np.ndarray:
    """MLEM/OSEM paralela simple por slice.

    subsets=1 equivale a MLEM; subsets>1 a OSEM. Implementacion CPU de referencia,
    pensada para validar el flujo y ser reemplazable por ASTRA para produccion.

    ``sensitivity_cache`` (opcional): imágenes de sensibilidad precalculadas por
    subset (ver `_build_sensitivity_cache`). Si no se pasa, se calculan igual
    que antes (por compatibilidad con llamadas directas a esta función, p.ej.
    en tests), pero el caller de producción (`reconstruct_projection_volume`)
    siempre las precalcula una vez para evitar recomputarlas por cada corte e
    iteración.

    MAP-OSEM (Green OSL, opcional): si ``map_beta`` > 0 y ``map_prior`` != "none",
    divide el update por ``(1 + beta * dU)`` con ``dU`` el gradiente del prior
    edge-preserving (mediana: x − mediana(x)). Es local (no impone valor temporal
    fijo), así que controla el ruido de la recon sin aplastar el movimiento
    cardíaco en gated (a diferencia de una guía ungated por-gate). beta típico
    0.2-0.5; 0 desactiva (OSEM puro, comportamiento idéntico al previo).
    Pilar C (SNR adaptativa): si ``map_beta_map`` (array (H,W)) se pasa, se usa
    como campo beta ESPACIAL con el prior de suavidad Huber (nitida3): el freno
    es fuerte donde la SNR es alta (pared) y débil donde domina el movimiento /
    ruido. Tiene prioridad sobre map_beta/map_prior escalares.    """
    measured = np.clip(np.asarray(sinogram, dtype=np.float64), 0.0, None)
    theta = np.asarray(theta, dtype=np.float64)
    detector_size = int(measured.shape[0])
    out_size = int(output_size)
    if measured.ndim != 2 or measured.shape[1] != theta.size:
        raise ValueError("sinogram/theta incompatibles para reconstruccion iterativa")
    if not np.any(measured > 0):
        return np.zeros((out_size, out_size), dtype=np.float64)

    image = np.full((out_size, out_size), max(float(measured.mean()), 1.0), dtype=np.float64)
    subset_count = max(1, min(int(subsets), int(theta.size)))
    angle_indices = np.arange(theta.size)
    eps = 1e-6
    use_map = float(map_beta) > 0.0 and str(map_prior).lower() not in ("none", "")
    if use_map:
        from core.nitida3 import edge_preserving_prior
    if map_adaptive:
        from core.nitida3 import local_snr_map, matched_recovery_weight, huber_prior_grad

    def _prior_grad(img: np.ndarray) -> np.ndarray:
        ref = edge_preserving_prior(img, kind=str(map_prior), size=int(map_prior_size))
        # Gradiente del prior (local): cuánto se desvía cada voxel de la referencia
        # suavizada. Positivo donde x > referencia -> baja; negativo donde x < ref.
        return img - ref

    for _iter in range(max(1, int(iterations))):
        for subset_id in range(subset_count):
            idx = angle_indices[subset_id::subset_count]
            if idx.size == 0:
                continue
            theta_sub = theta[idx]
            measured_sub = measured[:, idx]
            estimated_sub = _forward_project_slice(
                image,
                theta_sub,
                detector_size=detector_size,
                psf=psf,
                attenuation_mu_map_2d=attenuation_mu_map_2d,
                attenuation_mu_scale=attenuation_mu_scale,
                attenuation_pixel_size_cm=attenuation_pixel_size_cm,
            )
            ratio = measured_sub / np.maximum(estimated_sub, eps)
            correction = _backproject_slice(
                ratio,
                theta_sub,
                output_size=out_size,
                psf=psf,
                attenuation_mu_map_2d=attenuation_mu_map_2d,
                attenuation_mu_scale=attenuation_mu_scale,
                attenuation_pixel_size_cm=attenuation_pixel_size_cm,
            )
            if sensitivity_cache is not None and subset_id in sensitivity_cache:
                sensitivity = sensitivity_cache[subset_id]
            else:
                sensitivity = _backproject_slice(
                    np.ones_like(measured_sub),
                    theta_sub,
                    output_size=out_size,
                    psf=psf,
                    attenuation_mu_map_2d=attenuation_mu_map_2d,
                    attenuation_mu_scale=attenuation_mu_scale,
                    attenuation_pixel_size_cm=attenuation_pixel_size_cm,
                )
            denom = np.maximum(sensitivity, eps)
            if map_adaptive:
                # Pilar C: beta espacial (SNR local) x gradiente Huber (suavidad,
                # no mata el movimiento). denom * (1 + beta0 * w(snr) * dU_huber).
                snr = local_snr_map(image)
                w = matched_recovery_weight(snr)
                denom = np.maximum(denom * (1.0 + float(map_beta0) * w * huber_prior_grad(image)), 1e-3)
            elif use_map:
                # Green OSL: la sensibilidad queda atenuada por (1 + beta * dU).
                # Clamp a un piso positivo para evitar división por ~0 / negativo.
                denom = np.maximum(denom * (1.0 + float(map_beta) * _prior_grad(image)), 1e-3)
            image *= correction / denom
            image = np.clip(image, 0.0, None)
    return image


def reconstruct_gated_fbp_volume(
    projections: np.ndarray,
    angles_deg: np.ndarray | None = None,
    *,
    projection_filter: ProjectionFilterConfig | None = None,
    fbp_filter_name: str = "ramp",
    slice_range: tuple[int, int] | None = None,
    progress=None,
) -> np.ndarray:
    """Reconstruye cada gate por separado. Entrada (gates,angles,H,W)."""
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibio {proj.shape}")
    n_gates = int(proj.shape[0])
    volumes = []
    for gate in range(n_gates):
        gate_progress = _sub_progress(progress, gate, n_gates)
        volumes.append(
            reconstruct_fbp_volume(
                proj[gate], angles_deg, projection_filter=projection_filter,
                fbp_filter_name=fbp_filter_name, slice_range=slice_range, progress=gate_progress,
            )
        )
    return np.stack(volumes, axis=0)


def reconstruct_gated_projection_volume(
    projections: np.ndarray,
    angles_deg: np.ndarray | None = None,
    *,
    method: str = "fbp",
    projection_filter: ProjectionFilterConfig | None = None,
    fbp_filter_name: str = "ramp",
    iterations: int = 4,
    subsets: int = 4,
    psf=None,
    slice_range: tuple[int, int] | None = None,
    progress=None,
    map_beta: float = 0.0,
    map_prior: str = "none",
    map_prior_size: int = 3,
    map_adaptive: bool = False,
    map_beta0: float = 0.4,
    attenuation_mu_map: np.ndarray | None = None,
    attenuation_mu_scale: float = 1.0,
    attenuation_pixel_size_cm: float = 1.0,
) -> np.ndarray:
    """Reconstruye cada gate por separado con FBP/MLEM/OSEM.

    ``psf`` (opcional): activa la recuperación de resolución (RR) en el path
    iterativo, compartida entre todos los gates (misma geometría/colimador).
    ``progress`` (opcional): callable(local_fraction 0..1) repartido entre gates.
    ``map_beta``/``map_prior``/``map_prior_size`` (opcionales): MAP-OSEM (Green
    OSL) con prior edge-preserving dentro del update (NÍTIDA III). 0/"none" =
    OSEM puro (comportamiento previo).
    """
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibio {proj.shape}")
    method_key = str(method or "fbp").strip().lower()
    sensitivity_cache: dict[int, np.ndarray] | None = None
    if method_key in {"mlem", "osem"} and proj.shape[0] > 0:
        # Todos los gates comparten la misma geometría (ángulos + tamaño de
        # detector/salida), así que la sensibilidad se calcula UNA sola vez
        # para el estudio completo y se reutiliza en todos los gates (antes
        # se recalculaba, de forma redundante, gate por gate).
        n_angles = proj.shape[1]
        width = proj.shape[3]
        theta = np.linspace(0.0, 360.0, n_angles, endpoint=False) if angles_deg is None else np.asarray(angles_deg, dtype=np.float64)
        effective_subsets = int(subsets) if method_key == "osem" else 1
        sensitivity_cache = _build_sensitivity_cache(
            theta, subsets=effective_subsets, detector_size=width, output_size=width, psf=psf
        )
    volumes = [
        reconstruct_projection_volume(
            proj[gate],
            angles_deg,
            method=method,
            projection_filter=projection_filter,
            fbp_filter_name=fbp_filter_name,
            iterations=iterations,
            subsets=subsets,
            sensitivity_cache=sensitivity_cache,
            psf=psf,
            slice_range=slice_range,
            progress=_sub_progress(progress, gate, int(proj.shape[0])),
            map_beta=map_beta,
            map_prior=map_prior,
            map_prior_size=map_prior_size,
            map_adaptive=map_adaptive,
            map_beta0=map_beta0,
            attenuation_mu_map=attenuation_mu_map,
            attenuation_mu_scale=float(attenuation_mu_scale),
            attenuation_pixel_size_cm=float(attenuation_pixel_size_cm),
        )
        for gate in range(proj.shape[0])
    ]
    return np.stack(volumes, axis=0)


def make_display_cube(phase_cube: np.ndarray, step_px: int = 2) -> np.ndarray:
    """Submuestrea cortes solo para presentacion; no altera el cubo de calculo."""
    cube = np.asarray(phase_cube, dtype=np.float64)
    step = max(1, int(step_px))
    return cube[:, ::step].copy()


# Sentido de giro que recibe el flip L/R para converger a orientación anatómica
# canónica (VI a la izquierda del paciente). Es un único interruptor global: si
# la validación clínica muestra TODOS los estudios espejados, invertir a False.
_FLIP_X_ON_CCW = False


def _detect_rotation_ccw(angles_deg: np.ndarray | None) -> bool | None:
    """Detecta el sentido de giro desde los ángulos de proyección.

    Devuelve True si el ángulo crece con el índice (CC/CCW), False si decrece
    (CW), o None si no hay metadata suficiente para decidir. Usa unwrap para
    tolerar el cruce por 0/360°.
    """
    if angles_deg is None:
        return None
    a = np.asarray(angles_deg, dtype=np.float64)
    if a.size < 2:
        return None
    diffs = np.diff(np.unwrap(np.deg2rad(a)))
    mean_step = float(np.mean(diffs))
    if abs(mean_step) < 1e-6:
        return None
    return mean_step > 0.0


def reconstruct_raw_gated_pipeline(
    projections: np.ndarray,
    angles_deg: np.ndarray | None = None,
    *,
    motion_result: dict | None = None,
    config: RawReconConfig | None = None,
    motion_kwargs: dict | None = None,
    progress_callback=None,
    scatter_projections: np.ndarray | None = None,
    attenuation_mu_map: np.ndarray | None = None,
    attenuation_pixel_size_cm: float | None = None,
) -> RawReconResult:
    """Ejecuta el pipeline raw gated central.

    Si motion_result se provee, usa sus shifts. Si no, calcula motion correction
    con motion_correct_projections(**motion_kwargs). Los mismos shifts se aplican
    al UngGat y al gated; los filtros se aplican despues y son independientes.

    ``progress_callback`` (opcional): callable(fraction 0..1, message) invocado
    durante la reconstruccion (UngGat + gates) para alimentar una barra de
    progreso. La UI puede usarlo para mostrar avance en OSEM/MLEM y NITIDA.
    """
    cfg = _validate_config(config or RawReconConfig())
    raw = np.asarray(projections, dtype=np.float64)
    if raw.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibio {raw.shape}")

    notes: list[str] = []
    if motion_result is None:
        kwargs = dict(motion_kwargs or {})
        kwargs.setdefault("method", "sinusoid")
        kwargs.setdefault("axis", "y")
        kwargs.setdefault("angles_deg", angles_deg)
        motion_result = motion_correct_projections(raw, **kwargs)
        notes.append("Motion correction calculada dentro del pipeline.")
    else:
        notes.append("Motion correction provista externamente; se reutilizan shifts geometricos.")

    shifts_y = np.asarray(motion_result.get("applied_shifts_y", np.zeros((raw.shape[1],))), dtype=np.float64)
    shifts_x = np.asarray(motion_result.get("applied_shifts_x", np.zeros((raw.shape[1],))), dtype=np.float64)
    if shifts_y.size != raw.shape[1] or shifts_x.size != raw.shape[1]:
        raise ValueError("Los shifts de motion correction deben tener longitud n_angles")

    corrected = apply_shifts_to_projections(raw, shifts_y, shifts_x)
    ungated_corrected = apply_shifts_to_projections(ungate_projections(raw), shifts_y, shifts_x)

    # Corrección de SCATTER por ventana energética (dual EM/SC). El SC recibe
    # los MISMOS shifts geométricos de motion correction (misma adquisición,
    # misma geometría) y se resta con factor k de cada proyección, ANTES de
    # cualquier otro procesamiento. Clip a 0: las cuentas no pueden ser
    # negativas (el exceso de resta es ruido/fondo sobre-estimado).
    if bool(getattr(cfg, "scatter_subtract", False)) and scatter_projections is not None:
        sc = np.asarray(scatter_projections, dtype=np.float64)
        k_sc = float(getattr(cfg, "scatter_k", 1.0) or 1.0)
        n_gates_em = int(raw.shape[0])
        n_gates_sc = int(sc.shape[0]) if sc.ndim >= 4 else 1
        # SC no gatillado (1 gate) + EM gated (N gates): el SC tiene TODAS las
        # cuentas de scatter (no se divide por gates), cada gate EM tiene 1/N.
        # Para restar por gate hay que repartir el SC: SC_gate = SC_ungated / N.
        # Física: el scatter es de baja frecuencia temporal (el corazón no se
        # mueve mucho en la ventana de scatter) -> se considera estacionario.
        if n_gates_sc == 1 and n_gates_em >= 2:
            sc = np.repeat(sc / float(n_gates_em), n_gates_em, axis=0)
            notes.append(
                f"SC no gatillado detectado (1 gate): repartido como SC/N_gates "
                f"({n_gates_em}) para la resta por gate."
            )
        elif n_gates_sc != n_gates_em:
            notes.append(
                f"[WARN] Scatter: gates SC={n_gates_sc} != EM={n_gates_em}: no se aplica."
            )
            sc = None
        if sc is not None and sc.shape == raw.shape:
            sc_corr = apply_shifts_to_projections(sc, shifts_y, shifts_x)
            corrected = np.clip(corrected - k_sc * sc_corr, 0.0, None)
            sc_ung = apply_shifts_to_projections(ungate_projections(sc), shifts_y, shifts_x)
            ungated_corrected = np.clip(ungated_corrected - k_sc * sc_ung, 0.0, None)
            notes.append(
                f"Corrección de scatter EM/SC aplicada: P = EM - {k_sc:.2f}×SC "
                f"(pre-recon, proyección por proyección, gated y ungated)."
            )
        elif sc is not None:
            notes.append(
                f"[WARN] Scatter pedido pero shape SC {sc.shape} != EM {raw.shape}: no se aplica."
            )

    method = str(cfg.reconstruction_method).strip().lower()
    subsets = int(cfg.osem_subsets) if method == "osem" else 1
    # RR NÍTIDA (OmniRes) por rama: rr_ungated/rr_gated (None = hereda el global
    # resolution_recovery). Solo aplica al path iterativo (osem/mlem).
    _rr_global = bool(getattr(cfg, "resolution_recovery", False))
    rr_ung = _rr_global if getattr(cfg, "rr_ungated", None) is None else bool(cfg.rr_ungated)
    rr_gat = _rr_global if getattr(cfg, "rr_gated", None) is None else bool(cfg.rr_gated)
    rr_psf = cfg.psf_model if (rr_ung and method in {"osem", "mlem"}) else None
    if rr_ung and rr_psf is None:
        notes.append("NÍTIDA (OmniRes) ungated pedido pero inactivo: requiere método OSEM/MLEM y un PsfModel.")

    # Método independiente de la rama gated (None => hereda el de la rama ungated).
    gated_method = str(cfg.gated_method or method).strip().lower()
    gated_subsets = int(cfg.osem_subsets) if gated_method == "osem" else 1
    gated_rr_psf = cfg.psf_model if (rr_gat and gated_method in {"osem", "mlem"}) else None

    if method in {"osem", "mlem"} and _normalize_filter_kind(cfg.ungated_filter.kind) == "butterworth":
        notes.append(
            f"UngGat {method.upper()}: Butterworth {cfg.ungated_filter.cutoff:.2f}/{cfg.ungated_filter.order} "
            "aplicado POST-reconstrucción (3D radial), no como pre-filtro del sinograma."
        )
    if gated_method in {"osem", "mlem"} and _normalize_filter_kind(cfg.gated_filter.kind) == "butterworth":
        notes.append(
            f"Gated {gated_method.upper()}: Butterworth {cfg.gated_filter.cutoff:.2f}/{cfg.gated_filter.order} "
            "aplicado POST-reconstrucción (3D radial), no como pre-filtro del sinograma."
        )

    ac_requested = bool(getattr(cfg, "attenuation_correction", False))
    ac_mu_map = None
    ac_mu_scale = float(getattr(cfg, "attenuation_mu_scale", 1.0) or 1.0)
    if ac_requested:
        if attenuation_mu_map is None:
            notes.append("[WARN] AC iterativa pedida pero no se proveyó ATT MAP en grilla de reconstrucción; se desactiva AC.")
        else:
            ac_mu_map = np.asarray(attenuation_mu_map, dtype=np.float64)
            expected_shape = (int(raw.shape[2]), int(raw.shape[3]), int(raw.shape[3]))
            if ac_mu_map.shape != expected_shape:
                notes.append(
                    "[WARN] AC iterativa pedida pero ATT MAP con shape incompatible "
                    f"{ac_mu_map.shape} (esperado {expected_shape}); se desactiva AC."
                )
                ac_mu_map = None
    # Fallback estable para reconstrucción iterativa cuando no se conoce spacing:
    # 6.8 mm ~= 0.68 cm (matriz típica NM 64x64 con FOV cardíaco estándar).
    ac_px_cm = max(1e-4, float(attenuation_pixel_size_cm) if attenuation_pixel_size_cm is not None else 0.68)

    ac_ung_enabled = ac_mu_map is not None and method in {"osem", "mlem"}
    ac_gat_enabled = ac_mu_map is not None and gated_method in {"osem", "mlem"}
    if ac_requested and ac_mu_map is not None and not (ac_ung_enabled or ac_gat_enabled):
        notes.append("AC iterativa disponible pero no aplicada: método(s) FBP no usan modelo de atenuación en el update.")
    if ac_ung_enabled or ac_gat_enabled:
        notes.append(
            "AC iterativa habilitada en reconstrucción "
            f"(μ-scale={ac_mu_scale:.3f}, px={ac_px_cm*10.0:.3f} mm, "
            f"ungated={'ON' if ac_ung_enabled else 'OFF'}, gated={'ON' if ac_gat_enabled else 'OFF'})."
        )

    # Reparto del presupuesto de progreso: UngGat ~25%, gates ~70%, post ~5%.
    n_gates = int(raw.shape[0])
    method_label = ("NÍTIDA/" + method.upper()) if rr_psf is not None else method.upper()
    gated_method_label = ("NÍTIDA/" + gated_method.upper()) if gated_rr_psf is not None else gated_method.upper()

    def _ung_progress(frac: float) -> None:
        if progress_callback is not None:
            progress_callback(0.25 * max(0.0, min(1.0, frac)), f"Reconstruyendo UngGat ({method_label})...")

    def _gated_progress(frac: float) -> None:
        if progress_callback is not None:
            gate_1based = min(n_gates, int(frac * n_gates) + 1)
            progress_callback(0.25 + 0.70 * max(0.0, min(1.0, frac)), f"Reconstruyendo gate {gate_1based}/{n_gates} ({gated_method_label})...")

    ung_progress = _ung_progress if progress_callback is not None else None
    gated_progress = _gated_progress if progress_callback is not None else None

    # Descuento de fondo automático: se mide el piso en la imagen ungated media
    # y se resta de TODAS las proyecciones (incluido el VI: el fondo es aditivo
    # y también está debajo del miocardio). Al gated se le resta nivel/n_gates
    # por gate, así la suma de gates coincide con el ungated. Va ANTES del
    # denoise FBP_CLEAN: el fondo es señal aditiva del sinograma, no ruido.
    if getattr(cfg, "background_subtract", False):
        from core.raw_background import auto_background_level, subtract_constant
        bg_level = auto_background_level(ungated_corrected.mean(axis=0))
        if bg_level > 0.0:
            ungated_corrected = subtract_constant(ungated_corrected, bg_level).image
            corrected = subtract_constant(corrected, bg_level / max(n_gates, 1)).image
            notes.append(
                f"Descuento de fondo automático: nivel {bg_level:.1f} restado de todas "
                f"las proyecciones (gated: {bg_level / max(n_gates, 1):.2f}/gate)."
            )
        else:
            notes.append("Descuento de fondo automático: nivel medido 0 (sin efecto).")

    # FBP_CLEAN: denoise Poisson del sinograma ANTES del FBP (ataca las estrías
    # en la raíz). Aplica solo a la rama FBP (ungated y/o gated según método).
    fbc_sigma = float(getattr(cfg, "fbp_clean_sigma_color", 0.0) or 0.0)
    ungated_src, gated_src = ungated_corrected, corrected
    if fbc_sigma > 0.0 and method == "fbp":
        from core.fbp_clean import denoise_projections_bilateral
        ungated_src = denoise_projections_bilateral(ungated_corrected, sigma_color=fbc_sigma)
        notes.append(f"FBP_CLEAN: denoise bilateral de proyecciones UngGat (σc={fbc_sigma:.3f}).")
    gated_fbc = fbc_sigma > 0.0 and str(cfg.gated_method or method).strip().lower() == "fbp"
    if gated_fbc:
        from core.fbp_clean import denoise_projections_bilateral
        gated_src = denoise_projections_bilateral(corrected, sigma_color=fbc_sigma)
        notes.append(f"FBP_CLEAN: denoise bilateral de proyecciones gated (σc={fbc_sigma:.3f}).")

    ungated_volume = reconstruct_projection_volume(
        ungated_src,
        angles_deg,
        method=method,
        projection_filter=cfg.ungated_filter,
        fbp_filter_name=cfg.fbp_filter_name,
        iterations=int(cfg.iterative_iterations),
        subsets=subsets,
        psf=rr_psf,
        slice_range=cfg.recon_slice_range,
        progress=ung_progress,
        attenuation_mu_map=ac_mu_map if ac_ung_enabled else None,
        attenuation_mu_scale=ac_mu_scale,
        attenuation_pixel_size_cm=ac_px_cm,
    )
    gated_volume = reconstruct_gated_projection_volume(
        gated_src,
        angles_deg,
        method=gated_method,
        projection_filter=cfg.gated_filter,
        fbp_filter_name=cfg.fbp_filter_name,
        iterations=int(cfg.iterative_iterations),
        subsets=gated_subsets,
        psf=gated_rr_psf,
        slice_range=cfg.recon_slice_range,
        progress=gated_progress,
        attenuation_mu_map=ac_mu_map if ac_gat_enabled else None,
        attenuation_mu_scale=ac_mu_scale,
        attenuation_pixel_size_cm=ac_px_cm,
    )
    if rr_psf is not None or gated_rr_psf is not None:
        notes.append("NÍTIDA (OmniRes) activo: recuperación de resolución dependiente de profundidad.")
    if gated_method != method:
        notes.append(f"Método por rama: UngGat={method.upper()}, gated={gated_method.upper()}.")

    # Denoise+ UNGATED: denoise bilateral del sinograma ungated + realce por resta.
    # Abre la cavidad y afina la pared (el ungated también sufre scatter/fondo).
    # Se aplica sobre el volumen ungated ya reconstruido, restando una fracción
    # de la versión muy suavizada (misma idea que el realce de FBP_CLEAN).
    if getattr(cfg, "ungated_denoise_plus", False):
        from core.fbp_clean import denoise_projections_bilateral, sharpen_by_subtraction
        k_u = float(getattr(cfg, "ungated_denoise_plus_k", 0.20))
        blur_sc = float(getattr(cfg, "fbp_clean_blur_sigma_color", 0.24))
        ung_blur = reconstruct_projection_volume(
            denoise_projections_bilateral(ungated_corrected, sigma_color=blur_sc),
            angles_deg, method=method, projection_filter=cfg.ungated_filter,
            fbp_filter_name=cfg.fbp_filter_name, iterations=int(cfg.iterative_iterations),
            subsets=subsets, psf=rr_psf, slice_range=cfg.recon_slice_range,
            attenuation_mu_map=ac_mu_map if ac_ung_enabled else None,
            attenuation_mu_scale=ac_mu_scale,
            attenuation_pixel_size_cm=ac_px_cm)
        ungated_volume = sharpen_by_subtraction(ungated_volume, ung_blur, k_u)
        notes.append(f"Denoise+ ungated: realce por resta (k={k_u:.2f}, difuso σc={blur_sc:.2f}).")

    # NITIDA III: reconstruye el GATED con MAP-OSEM Pilar C (SNR adaptativa).
    # Es una reconstrucción (reemplaza al gated FBP/OSEM de arriba): prior de
    # suavidad Huber con beta espacial por SNR local -> limpia la pared sin
    # aplastar el movimiento. Reescalado por gate para conservar cuentas (FEVI).
    if getattr(cfg, "nitida3_enabled", False) and gated_volume.shape[0] >= 3:
        from core.nitida3 import nitida3_map_osem_gated_adaptive
        _n3_gates = int(gated_volume.shape[0])

        def _n3_progress(frac: float, msg: str = "") -> None:
            if progress_callback is not None:
                # Barra propia de NITIDA III: mapea su avance interno (0..1) al
                # tramo 0.90-0.97 de la barra global. El progress interno solo
                # pasa la fracción (no mensaje), así que el gate se deriva de ella.
                f = max(0.0, min(1.0, float(frac)))
                gate_1b = min(_n3_gates, int(f * _n3_gates) + 1)
                text = msg or f"NITIDA III (MAP-OSEM Pilar C) gate {gate_1b}/{_n3_gates}..."
                progress_callback(0.90 + 0.07 * f, text)

        # Referencia de cuentas para el reescalado = el gated YA reconstruido
        # arriba (evita una 2da reconstrucción OSEM completa -> ~2x más rápido).
        gated_volume = nitida3_map_osem_gated_adaptive(
            gated_src,
            angles_deg,
            iterations=int(getattr(cfg, "nitida3_iterations", 2)),
            subsets=int(getattr(cfg, "nitida3_subsets", 4)),
            beta0=float(getattr(cfg, "nitida3_beta0", 0.6)),
            psf=gated_rr_psf,
            slice_range=cfg.recon_slice_range,
            rescale=True,
            ref_volume=gated_volume,
            progress_callback=_n3_progress,
        )
        notes.append(
            f"NITIDA III: gated reconstruido con MAP-OSEM Pilar C "
            f"(Huber por SNR, beta0={float(getattr(cfg, 'nitida3_beta0', 0.6)):.2f})."
        )

    # NITIDA 4D (4D-OSEM): prior TEMPORAL entre gates. Reconstruye los gates
    # JUNTOS con suavidad temporal Huber dentro del update OSEM. A diferencia
    # de promediar gates (motion-frozen, que congela el latido), el prior
    # temporal solo frena el ruido incorrelado; la contracción se conserva.
    # Reescalado por gate para conservar cuentas (FEVI). Reemplaza al gated.
    if getattr(cfg, "nitida4d_enabled", False) and gated_volume.shape[0] >= 3:
        from core.nitida4d import nitida4d_osem_gated
        _n4_gates = int(gated_volume.shape[0])

        def _n4_progress(frac: float, msg: str = "") -> None:
            if progress_callback is not None:
                f = max(0.0, min(1.0, float(frac)))
                text = msg or f"NITIDA 4D (4D-OSEM, prior temporal)..."
                progress_callback(0.90 + 0.07 * f, text)

        gated_volume = nitida4d_osem_gated(
            gated_src,
            angles_deg,
            iterations=int(getattr(cfg, "nitida4d_iterations", 4)),
            subsets=int(getattr(cfg, "nitida4d_subsets", 4)),
            beta_temporal=float(getattr(cfg, "nitida4d_beta_temporal", 0.3)),
            delta_temporal=float(getattr(cfg, "nitida4d_delta_temporal", 0.05)),
            psf=gated_rr_psf,
            slice_range=cfg.recon_slice_range,
            rescale=True,
            ref_volume=gated_volume,
            progress_callback=_n4_progress,
        )
        notes.append(
            f"NITIDA 4D: gated reconstruido con 4D-OSEM (prior temporal Huber, "
            f"beta_temp={float(getattr(cfg, 'nitida4d_beta_temporal', 0.3)):.2f}, "
            f"delta_temp={float(getattr(cfg, 'nitida4d_delta_temporal', 0.05)):.3f})."
        )

    # FBP_CLEAN paso 2: realce de cavidad/bordes por resta de una fracción de la
    # versión muy suavizada (unsharp mask, idea del usuario). out = nítido − k×difuso.
    # Se aplica a los volúmenes ya reconstruidos (ungated y gated) si FBP_CLEAN
    # está activo. La versión difusa se obtiene con un bilateral fuerte (σc_blur)
    # sobre las mismas proyecciones ya denoised, reconstruida igual.
    if fbc_sigma > 0.0:
        from core.fbp_clean import denoise_projections_bilateral, sharpen_by_subtraction
        k = float(getattr(cfg, "fbp_clean_sharpen_k", 0.5))
        blur_sc = float(getattr(cfg, "fbp_clean_blur_sigma_color", 0.24))
        if method == "fbp":
            ung_blur = reconstruct_projection_volume(
                denoise_projections_bilateral(ungated_corrected, sigma_color=blur_sc),
                angles_deg, method="fbp", projection_filter=cfg.ungated_filter,
                fbp_filter_name=cfg.fbp_filter_name, slice_range=cfg.recon_slice_range)
            ungated_volume = sharpen_by_subtraction(ungated_volume, ung_blur, k)
        if gated_fbc:
            gated_blur = reconstruct_gated_projection_volume(
                denoise_projections_bilateral(corrected, sigma_color=blur_sc),
                angles_deg, method="fbp", projection_filter=cfg.gated_filter,
                fbp_filter_name=cfg.fbp_filter_name, slice_range=cfg.recon_slice_range)
            gated_volume = sharpen_by_subtraction(gated_volume, gated_blur, k)
        notes.append(f"FBP_CLEAN: realce por resta (k={k:.2f}, difuso σc={blur_sc:.2f}).")

    # Denoise+ GATED (method-agnostic): el MISMO realce por resta que abre la
    # cavidad del ungated, pero para la rama gated y con CUALQUIER método (FBP u
    # OSEM). Motivo: FBP_CLEAN solo corre si gated es FBP, y el gated OSEM (default
    # actual) quedaba sin tratamiento de cavidad -> se veía difuminado y se
    # perdía la cavidad (reportado por el usuario 2026-08-14). Denoise bilateral
    # del sinograma gated (σc=0.04 nítido / 0.24 difuso) + doble recon con el
    # MISMO método + resta con k. Aplica sobre el gated YA reconstruido.
    if getattr(cfg, "gated_denoise_plus", False) and gated_volume is not None and gated_volume.ndim == 4:
        from core.fbp_clean import denoise_projections_bilateral, sharpen_by_subtraction
        k_g = float(getattr(cfg, "gated_denoise_plus_k", 0.50))
        blur_sc = float(getattr(cfg, "fbp_clean_blur_sigma_color", 0.24))
        try:
            gated_blur = reconstruct_gated_projection_volume(
                denoise_projections_bilateral(corrected, sigma_color=blur_sc),
                angles_deg, method=gated_method, projection_filter=cfg.gated_filter,
                fbp_filter_name=cfg.fbp_filter_name, iterations=int(cfg.iterative_iterations),
                subsets=gated_subsets, psf=gated_rr_psf, slice_range=cfg.recon_slice_range,
                attenuation_mu_map=ac_mu_map if ac_gat_enabled else None,
                attenuation_mu_scale=ac_mu_scale,
                attenuation_pixel_size_cm=ac_px_cm)
            gated_volume = sharpen_by_subtraction(gated_volume, gated_blur, k_g)
            notes.append(
                f"Denoise+ GATED: realce por resta (k={k_g:.2f}, difuso σc={blur_sc:.2f}, "
                f"método {gated_method.upper()}). Abre la cavidad del gated."
            )
        except Exception as exc:
            notes.append(f"[WARN] Denoise+ GATED falló ({exc}); se omite.")

    # Orientación radiológica L/R: la retroproyección produce el volumen con el
    # eje izquierda/derecha del paciente (columnas, axis -1) espejado respecto de
    # la convención de despliegue estándar (Xeleris/Odyssey/atlas): en la vista
    # ANTERIOR el corazón (VI) debe quedar a la IZQUIERDA del paciente = DERECHA
    # de la pantalla, y la LATERAL izquierda debe mirar hacia la izquierda de la
    # pantalla. Se espeja el eje x una sola vez aquí, de modo que las vistas de
    # referencia y todos los cortes SA/HLA/VLA hereden la orientación correcta.
    # CLAVE: iradon produce imágenes ESPEJADAS entre CW y CCW. Aplicar el flip de
    # forma incondicional dejaba correctos los estudios de un sentido de giro y
    # espejados los del otro (causa del "posterior/lateral derecha" en la mayoría
    # salvo el de calibración). Se condiciona el flip al sentido detectado para
    # que AMBOS sentidos converjan a la misma orientación anatómica. Si no hay
    # metadata angular, se conserva el flip previo (comportamiento heredado).
    ccw = _detect_rotation_ccw(angles_deg)
    flip_x = True if ccw is None else (bool(ccw) == _FLIP_X_ON_CCW)
    if flip_x:
        ungated_volume = np.ascontiguousarray(np.flip(ungated_volume, axis=-1))
        gated_volume = np.ascontiguousarray(np.flip(gated_volume, axis=-1))
    notes.append(
        f"Orientacion L/R: sentido={'CCW' if ccw else ('CW' if ccw is False else '?')}, "
        f"flip_x={flip_x} (converge CW/CCW a orientacion canonica)."
    )

    # Post-filtro gaussiano 3D (control de ruido) POR RAMA. Contraparte de la RR:
    # la recon iterativa con PSF amplifica el ruido; este suavizado lo regula.
    # Si las sigmas por rama son None, se usa la global post_filter_sigma_px.
    from scipy.ndimage import gaussian_filter as _gf
    _post_global = float(getattr(cfg, "post_filter_sigma_px", 0.0) or 0.0)
    post_ung = _post_global if getattr(cfg, "post_filter_sigma_ungated_px", None) is None else float(cfg.post_filter_sigma_ungated_px)
    post_gat = _post_global if getattr(cfg, "post_filter_sigma_gated_px", None) is None else float(cfg.post_filter_sigma_gated_px)

    # Guardar copia SIN filtro para toggle en UI (se asigna al result al final)
    _ungated_unfiltered = np.ascontiguousarray(ungated_volume.copy())

    _post_kind = str(getattr(cfg, "post_filter_kind", "gaussian") or "gaussian").strip().lower()
    if _post_kind == "butterworth":
        if progress_callback is not None:
            progress_callback(0.96, "Aplicando post-filtro Butterworth 3D...")
        _bc = float(getattr(cfg, "post_filter_cutoff", 0.35))
        _bo = int(getattr(cfg, "post_filter_order", 5))
        ungated_volume = np.clip(_butterworth_3d(ungated_volume, _bc, _bo), 0.0, None)
        for g in range(gated_volume.shape[0]):
            gated_volume[g] = np.clip(_butterworth_3d(gated_volume[g], _bc, _bo), 0.0, None)
        notes.append(f"Post-filtro Butterworth 3D: cutoff={_bc:.2f} Nyquist, orden={_bo} (ungated y gated).")
    elif post_ung > 0.05 or post_gat > 0.05:
        if progress_callback is not None:
            progress_callback(0.96, "Aplicando post-filtro (suavizado)...")
        if post_ung > 0.05:
            ungated_volume = _gf(ungated_volume, sigma=post_ung, mode="constant")
        if post_gat > 0.05:
            for g in range(gated_volume.shape[0]):
                gated_volume[g] = _gf(gated_volume[g], sigma=post_gat, mode="constant")
        notes.append(
            f"Post-filtro gaussiano 3D por rama: ungated sigma={post_ung:.2f}px, "
            f"gated sigma={post_gat:.2f}px."
        )

    # NITIDA II: denoiser gated temporal/espaciotemporal por armónicos. Solo toca
    # el volumen gated (el ungated ya es de alto conteo). Preserva el movimiento
    # (bandas de baja frecuencia) y elimina el ruido de banda alta.
    nitida2_mode = str(getattr(cfg, "nitida2_mode", "none") or "none").strip().lower()
    if nitida2_mode in {"temporal", "spatiotemporal"} and gated_volume.shape[0] >= 3:
        if progress_callback is not None:
            progress_callback(0.98, f"Aplicando NITIDA II ({nitida2_mode})...")
        from core.nitida2 import denoise_spatiotemporal, temporal_harmonic_filter
        n_harm = int(getattr(cfg, "nitida2_harmonics", 2))
        if nitida2_mode == "temporal":
            gated_volume = temporal_harmonic_filter(gated_volume, n_harmonics=n_harm)
            notes.append(f"NITIDA II temporal (armónicos 0..{n_harm}) aplicado al gated.")
        else:
            gated_volume = denoise_spatiotemporal(
                gated_volume,
                n_harmonics=n_harm,
                dc_radius=int(getattr(cfg, "nitida2_dc_radius", 2)),
                dc_eps=float(getattr(cfg, "nitida2_dc_eps", 0.01)),
                band_sigma=float(getattr(cfg, "nitida2_band_sigma", 0.7)),
                guide_volume=ungated_volume / gated_volume.shape[0],
            )
            notes.append(
                f"NITIDA II espaciotemporal (armónicos 0..{n_harm}, "
                f"band_sigma={float(getattr(cfg, 'nitida2_band_sigma', 0.7)):.2f}) aplicado al gated."
            )

    # --- Motion-frozen 3D (post-recon, pre-reorientación) ---
    # Alinea cada gate del volumen 4D al end-diastole y promedia. El resultado
    # es una imagen de perfusión con la nitidez de un gate y todas las cuentas.
    # Se aplica DESPUÉS de todos los filtros de recon (Denoise+, FBP_CLEAN,
    # NITIDA, post-filtro) y ANTES de reorientar, porque el movimiento 3D real
    # solo existe en el volumen transaxial, no en los cortes reorientados.
    ungated_volume_mf = None
    if bool(getattr(cfg, "motion_frozen", False)) and gated_volume.ndim == 4 and gated_volume.shape[0] >= 2:
        if progress_callback is not None:
            progress_callback(0.97, "Motion-frozen: alineando gates 4D...")
        from core.motion_frozen import rigid_register_gates, displacement_field_register_gates
        mf_method = str(getattr(cfg, "motion_frozen_method", "rigid") or "rigid").strip().lower()
        mf_ref = getattr(cfg, "motion_frozen_ref_gate", None)
        try:
            if mf_method == "displacement":
                aligned, _u = displacement_field_register_gates(
                    gated_volume, ref_gate=mf_ref, smooth_sigma=1.5, reg_lambda=0.5, n_iter=30
                )
                notes.append("Motion-frozen: alineación por campo de desplazamiento 3D (demons-like).")
            else:
                aligned = rigid_register_gates(gated_volume, ref_gate=mf_ref, smooth_sigma=1.0)
                notes.append("Motion-frozen: alineación rígida 3D (traslación por centroide).")
            ungated_volume_mf = aligned.mean(axis=0)
            # PRIMERA IMPLEMENTACIÓN (la que funcionaba): solo alinear gates y
            # promediar. SIN Denoise+ al MF y SIN re-centrado — esos dos pasos
            # (agregados después) restaban volúmenes en distinta geometría y
            # generaban fantasmas en "C" / cortes corridos (reportado 2026-08-13).
            # Si se quiere mejorar la imagen del MF, el usuario aplica los
            # filtros al UNGATED por separado y compara.
            notes.append(
                f"Motion-frozen: {gated_volume.shape[0]} gates alineados y promediados "
                f"(método={mf_method}, ref_gate={mf_ref if mf_ref is not None else 'auto'})."
            )
        except Exception as exc:
            notes.append(f"[WARN] Motion-frozen falló ({exc}); se omite.")
            ungated_volume_mf = None

    # Motion-frozen POR GATE (cine nítido): alinea todos los gates a cada gate
    # sucesivamente y promedia. Cada "gate" resultante tiene la nitidez del MF
    # pero en su propia fase del ciclo. Útil para motilidad, NO para FEVI.
    gated_volume_mf_per_gate = None
    if bool(getattr(cfg, "motion_frozen_per_gate", False)) and gated_volume.ndim == 4 and gated_volume.shape[0] >= 2:
        if progress_callback is not None:
            progress_callback(0.97, "Motion-frozen por gate (cine nítido)...")
        from core.motion_frozen import motion_frozen_per_gate
        # Para el CINE el método es 'stable': alinea por el contexto (tórax,
        # hígado, fondo) EXCLUYENDO el corazón. Así se corrige el desplazamiento
        # global del paciente SIN congelar la contracción (el centroide global
        # está dominado por el corazón y arrastraba el latido -> mini-
        # desplazamientos raros reportados por el usuario).
        _mfpg_method = str(getattr(cfg, "motion_frozen_method", "rigid") or "rigid").strip().lower()
        if _mfpg_method == "rigid":
            _mfpg_method = "stable"
        try:
            gated_volume_mf_per_gate = motion_frozen_per_gate(
                gated_volume, method=_mfpg_method, smooth_sigma=1.0
            )
            notes.append(
                f"Motion-frozen por gate: {gated_volume.shape[0]} gates alineados "
                f"por contexto estable (conserva el latido) a cada fase y promediados "
                f"(método={_mfpg_method})."
            )
        except Exception as exc:
            notes.append(f"[WARN] Motion-frozen por gate falló ({exc}); se omite.")
            gated_volume_mf_per_gate = None

    if progress_callback is not None:
        progress_callback(1.0, "Reconstrucción completa")

    phase_cube = gated_volume[:, :: int(cfg.fevi_slice_step_px)].copy()
    display_cube = make_display_cube(phase_cube, step_px=int(cfg.display_slice_step_px))

    notes.append(
        f"Reconstruccion {method.upper()} aplicada. Filtros separados: "
        f"UngGat={cfg.ungated_filter.kind}(cutoff={cfg.ungated_filter.cutoff}, order={cfg.ungated_filter.order}); "
        f"gated={cfg.gated_filter.kind}(cutoff={cfg.gated_filter.cutoff}, order={cfg.gated_filter.order})."
    )
    notes.append("Cubo FEVI/fase conserva cortes cada 1 px; display_cube submuestrea solo para visualizacion.")

    return RawReconResult(
        original_projections=raw,
        corrected_projections=corrected,
        ungated_projections=ungated_corrected,
        ungated_volume=ungated_volume,
        gated_volume=gated_volume,
        phase_cube=phase_cube,
        display_cube=display_cube,
        shifts_y=shifts_y,
        shifts_x=shifts_x,
        config=cfg,
        motion_result=dict(motion_result),
        flip_x_applied=bool(flip_x),
        notes=notes,
        ungated_volume_mf=ungated_volume_mf,
        gated_volume_mf_per_gate=gated_volume_mf_per_gate,
        ungated_volume_unfiltered=_ungated_unfiltered,
    )
