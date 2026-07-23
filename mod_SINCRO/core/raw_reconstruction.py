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

    reconstruction_method hoy soporta fbp. OSEM/MLEM quedan declarados para UI y
    configuracion, pero levantan NotImplementedError hasta portar un motor iterativo.
    """
    reconstruction_method: str = "fbp"
    fbp_filter_name: str = "ramp"
    # Calcado de Xeleris ECToolbox (FBP) para este protocolo cardiaco:
    # ungated Butterworth cutoff 0.52 / orden 5; gated cutoff 0.40 / orden 10
    # (cutoff en fraccion de Nyquist).
    ungated_filter: ProjectionFilterConfig = field(default_factory=lambda: ProjectionFilterConfig("butterworth", 0.52, 5))
    gated_filter: ProjectionFilterConfig = field(default_factory=lambda: ProjectionFilterConfig("butterworth", 0.40, 10))
    iterative_iterations: int = 4
    osem_subsets: int = 4
    fevi_slice_step_px: int = 1
    display_slice_step_px: int = 2


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
    if int(config.iterative_iterations) < 1:
        raise ValueError("iterative_iterations debe ser >= 1")
    if int(config.osem_subsets) < 1:
        raise ValueError("osem_subsets debe ser >= 1")
    if int(config.fevi_slice_step_px) != 1:
        raise ValueError("fevi_slice_step_px debe ser 1 para calcular FEVI con todos los cortes reconstruidos")
    if int(config.display_slice_step_px) < 1:
        raise ValueError("display_slice_step_px debe ser >= 1")
    return config


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


def reconstruct_fbp_volume(
    projections: np.ndarray,
    angles_deg: np.ndarray | None = None,
    *,
    projection_filter: ProjectionFilterConfig | None = None,
    fbp_filter_name: str = "ramp",
) -> np.ndarray:
    """Reconstruye un volumen transaxial por FBP desde proyecciones 3D.

    Entrada: (n_angles, H, W). Salida: (H, W, W), un corte por fila axial del detector.
    """
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 3:
        raise ValueError(f"projections debe ser 3D (angles,H,W); recibio {proj.shape}")
    if projection_filter is not None:
        proj = filter_projections(proj, projection_filter)
    n_angles, height, width = proj.shape
    if angles_deg is None:
        theta = np.linspace(0.0, 360.0, n_angles, endpoint=False)
    else:
        theta = np.asarray(angles_deg, dtype=np.float64)
        if theta.size != n_angles:
            raise ValueError(f"angles_deg debe tener {n_angles} valores; recibio {theta.size}")

    volume = np.zeros((height, width, width), dtype=np.float64)
    for slice_idx in range(height):
        sinogram = proj[:, slice_idx, :].T
        volume[slice_idx] = _iradon_or_fallback(sinogram, theta, filter_name=str(fbp_filter_name), output_size=width)
    return volume


def reconstruct_projection_volume(
    projections: np.ndarray,
    angles_deg: np.ndarray | None = None,
    *,
    method: str = "fbp",
    projection_filter: ProjectionFilterConfig | None = None,
    fbp_filter_name: str = "ramp",
    iterations: int = 4,
    subsets: int = 4,
) -> np.ndarray:
    """Reconstruye volumen desde proyecciones 3D con FBP/MLEM/OSEM."""
    method_key = str(method or "fbp").strip().lower()
    if method_key == "fbp":
        return reconstruct_fbp_volume(
            projections, angles_deg, projection_filter=projection_filter, fbp_filter_name=fbp_filter_name
        )
    if method_key not in {"mlem", "osem"}:
        raise ValueError("method debe ser 'fbp', 'mlem' u 'osem'")
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 3:
        raise ValueError(f"projections debe ser 3D (angles,H,W); recibio {proj.shape}")
    if projection_filter is not None:
        proj = filter_projections(proj, projection_filter)
    proj = np.clip(proj, 0.0, None)
    n_angles, height, width = proj.shape
    theta = np.linspace(0.0, 360.0, n_angles, endpoint=False) if angles_deg is None else np.asarray(angles_deg, dtype=np.float64)
    if theta.size != n_angles:
        raise ValueError(f"angles_deg debe tener {n_angles} valores; recibio {theta.size}")
    out = np.zeros((height, width, width), dtype=np.float64)
    for slice_idx in range(height):
        sinogram = proj[:, slice_idx, :].T
        out[slice_idx] = _iterative_reconstruct_slice(
            sinogram,
            theta,
            output_size=width,
            iterations=int(iterations),
            subsets=(int(subsets) if method_key == "osem" else 1),
        )
    return out


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


def _simple_backprojection(sinogram: np.ndarray, theta: np.ndarray, *, output_size: int) -> np.ndarray:
    """Retroproyección paralela simple (adjunto). Sin filtro rampa.

    Usada por el path iterativo (MLEM/OSEM, donde el adjunto NO debe filtrarse)
    y por el fallback de FBP tras aplicar la rampa aparte.
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
    for idx, angle in enumerate(theta):
        profile = np.interp(x_new, x_old, sino[:, idx], left=0.0, right=0.0)
        slab = np.tile(profile.reshape(1, out_size), (out_size, 1))
        volume += rotate(slab, angle=float(angle), reshape=False, order=1, mode="constant", cval=0.0)
    if theta.size:
        volume *= np.pi / (2.0 * float(theta.size))
    return volume


def _forward_project_slice(image: np.ndarray, theta: np.ndarray, *, detector_size: int) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    sino = np.zeros((int(detector_size), int(theta.size)), dtype=np.float64)
    x_old = np.linspace(-1.0, 1.0, img.shape[1])
    x_new = np.linspace(-1.0, 1.0, int(detector_size))
    for idx, angle in enumerate(theta):
        rot = rotate(img, angle=-float(angle), reshape=False, order=1, mode="constant", cval=0.0)
        prof = rot.sum(axis=0)
        sino[:, idx] = np.interp(x_new, x_old, prof, left=0.0, right=0.0)
    return sino


def _backproject_slice(sinogram: np.ndarray, theta: np.ndarray, *, output_size: int) -> np.ndarray:
    return _simple_backprojection(sinogram, theta, output_size=output_size)


def _iterative_reconstruct_slice(
    sinogram: np.ndarray,
    theta: np.ndarray,
    *,
    output_size: int,
    iterations: int,
    subsets: int,
) -> np.ndarray:
    """MLEM/OSEM paralela simple por slice.

    subsets=1 equivale a MLEM; subsets>1 a OSEM. Implementacion CPU de referencia,
    pensada para validar el flujo y ser reemplazable por ASTRA para produccion.
    """
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
    for _iter in range(max(1, int(iterations))):
        for subset_id in range(subset_count):
            idx = angle_indices[subset_id::subset_count]
            if idx.size == 0:
                continue
            theta_sub = theta[idx]
            measured_sub = measured[:, idx]
            estimated_sub = _forward_project_slice(image, theta_sub, detector_size=detector_size)
            ratio = measured_sub / np.maximum(estimated_sub, eps)
            correction = _backproject_slice(ratio, theta_sub, output_size=out_size)
            sensitivity = _backproject_slice(np.ones_like(measured_sub), theta_sub, output_size=out_size)
            image *= correction / np.maximum(sensitivity, eps)
            image = np.clip(image, 0.0, None)
    return image


def reconstruct_gated_fbp_volume(
    projections: np.ndarray,
    angles_deg: np.ndarray | None = None,
    *,
    projection_filter: ProjectionFilterConfig | None = None,
    fbp_filter_name: str = "ramp",
) -> np.ndarray:
    """Reconstruye cada gate por separado. Entrada (gates,angles,H,W)."""
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibio {proj.shape}")
    volumes = [
        reconstruct_fbp_volume(proj[gate], angles_deg, projection_filter=projection_filter, fbp_filter_name=fbp_filter_name)
        for gate in range(proj.shape[0])
    ]
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
) -> np.ndarray:
    """Reconstruye cada gate por separado con FBP/MLEM/OSEM."""
    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibio {proj.shape}")
    volumes = [
        reconstruct_projection_volume(
            proj[gate],
            angles_deg,
            method=method,
            projection_filter=projection_filter,
            fbp_filter_name=fbp_filter_name,
            iterations=iterations,
            subsets=subsets,
        )
        for gate in range(proj.shape[0])
    ]
    return np.stack(volumes, axis=0)


def make_display_cube(phase_cube: np.ndarray, step_px: int = 2) -> np.ndarray:
    """Submuestrea cortes solo para presentacion; no altera el cubo de calculo."""
    cube = np.asarray(phase_cube, dtype=np.float64)
    step = max(1, int(step_px))
    return cube[:, ::step].copy()


def reconstruct_raw_gated_pipeline(
    projections: np.ndarray,
    angles_deg: np.ndarray | None = None,
    *,
    motion_result: dict | None = None,
    config: RawReconConfig | None = None,
    motion_kwargs: dict | None = None,
) -> RawReconResult:
    """Ejecuta el pipeline raw gated central.

    Si motion_result se provee, usa sus shifts. Si no, calcula motion correction
    con motion_correct_projections(**motion_kwargs). Los mismos shifts se aplican
    al UngGat y al gated; los filtros se aplican despues y son independientes.
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

    method = str(cfg.reconstruction_method).strip().lower()
    subsets = int(cfg.osem_subsets) if method == "osem" else 1
    ungated_volume = reconstruct_projection_volume(
        ungated_corrected,
        angles_deg,
        method=method,
        projection_filter=cfg.ungated_filter,
        fbp_filter_name=cfg.fbp_filter_name,
        iterations=int(cfg.iterative_iterations),
        subsets=subsets,
    )
    gated_volume = reconstruct_gated_projection_volume(
        corrected,
        angles_deg,
        method=method,
        projection_filter=cfg.gated_filter,
        fbp_filter_name=cfg.fbp_filter_name,
        iterations=int(cfg.iterative_iterations),
        subsets=subsets,
    )

    # Orientación radiológica L/R: la retroproyección produce el volumen con el
    # eje izquierda/derecha del paciente (columnas, axis -1) espejado respecto de
    # la convención de despliegue estándar (Xeleris/Odyssey/atlas): en la vista
    # ANTERIOR el corazón (VI) debe quedar a la IZQUIERDA del paciente = DERECHA
    # de la pantalla, y la LATERAL izquierda debe mirar hacia la izquierda de la
    # pantalla. Se espeja el eje x una sola vez aquí, de modo que las vistas de
    # referencia y todos los cortes SA/HLA/VLA hereden la orientación correcta.
    ungated_volume = np.ascontiguousarray(np.flip(ungated_volume, axis=-1))
    gated_volume = np.ascontiguousarray(np.flip(gated_volume, axis=-1))

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
        notes=notes,
    )
