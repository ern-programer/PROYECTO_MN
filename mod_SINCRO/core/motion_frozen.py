"""SINCRO - core.motion_frozen  (motion-frozen / cardiac morphing)

Recupera la nitidez de la perfusión miocárdica eliminando el motion blurring
del latido. La imagen ungated (suma de proyecciones) es la más estadística pero
se ve "inflada" porque promedia el corazón en todas las fases. Esta técnica
alinea cada gate a una geometría de referencia (end-diastole) y promedia,
recuperando la nitidez sin perder cuentas.

Tres enfoques (selector en UI):

  RIGID 3D POR GATE
    Traslación (y opcionalmente rotación) rígida de cada gate al gate de
    referencia. Robusto, rápido, no captura la contracción real (el corazón
    no se mueve como un bloque).

  DISPLACEMENT FIELD 3D (el "morphing" real)
    Campo de desplazamiento denso por voxel, regularizado por suavidad.
    Captura la contracción miocárdica real. Requiere cuidado para no inventar
    anatomía en zonas de baja SNR.

  SINOGRAMA ALINEADO (motion-frozen en proyecciones)
    Estima el desplazamiento 3D del corazón por gate y lo aplica como shift 2D
    a cada proyección antes de la reconstrucción. Más físico (el FBP no mezcla
    estructuras desplazadas), pero necesita la cinemática 3D desde datos 2D.

Referencias:
  - De Bondt et al. "Motion-frozen" myocardial perfusion SPECT. J Nucl Cardiol.
  - Klein et al. "Evaluation of 4D image registration for motion-frozen SPECT."
  - Rueckert et al. Nonrigid registration using free-form deformations. IEEE TMI.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    affine_transform,
    gaussian_filter,
    map_coordinates,
    shift as ndi_shift,
)


def _largest_volume_gate(gated_volume: np.ndarray) -> int:
    """Gate de referencia: end-diastole = el de mayor volumen ventricular.

    Se estima por el momento de máxima dispersión del centroide (aproximación
    robusta sin segmentación).
    """
    g4 = np.asarray(gated_volume, dtype=np.float64)
    if g4.ndim != 4 or g4.shape[0] < 2:
        return 0
    # Métrica simple: suma de cuentas en la mitad superior del histograma.
    # El gate con más volumen tiene más cuentas en la periferia.
    scores = []
    for g in range(g4.shape[0]):
        v = g4[g]
        thr = float(np.percentile(v, 70)) if v.size else 0.0
        scores.append(float((v > thr).sum()))
    return int(np.argmax(scores))


def _center_of_mass_3d(vol: np.ndarray) -> tuple[float, float, float]:
    """Centroide de intensidad 3D (z, y, x)."""
    v = np.asarray(vol, dtype=np.float64)
    tot = float(v.sum())
    if tot <= 1e-12:
        return (v.shape[0] / 2.0, v.shape[1] / 2.0, v.shape[2] / 2.0)
    zz, yy, xx = np.mgrid[0:v.shape[0], 0:v.shape[1], 0:v.shape[2]]
    return (
        float((v * zz).sum() / tot),
        float((v * yy).sum() / tot),
        float((v * xx).sum() / tot),
    )


def rigid_register_gates(
    gated_volume: np.ndarray,
    *,
    ref_gate: int | None = None,
    use_rotation: bool = False,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Alinea cada gate al gate de referencia por traslación (y rotación) rígida.

    Devuelve el volumen 4D alineado, mismo shape que la entrada.
    """
    g4 = np.asarray(gated_volume, dtype=np.float64)
    if g4.ndim != 4 or g4.shape[0] < 2:
        return g4.copy()
    n_gates = g4.shape[0]
    if ref_gate is None:
        ref_gate = _largest_volume_gate(g4)
    ref = g4[ref_gate]
    if smooth_sigma > 0:
        ref_s = gaussian_filter(ref, sigma=smooth_sigma, mode="constant")
    else:
        ref_s = ref
    com_ref = _center_of_mass_3d(ref_s)

    out = np.empty_like(g4)
    for g in range(n_gates):
        if g == ref_gate:
            out[g] = g4[g]
            continue
        mov = g4[g]
        mov_s = gaussian_filter(mov, sigma=smooth_sigma, mode="constant") if smooth_sigma > 0 else mov
        com_mov = _center_of_mass_3d(mov_s)
        shift_vec = [com_ref[0] - com_mov[0], com_ref[1] - com_mov[1], com_ref[2] - com_mov[2]]
        if use_rotation:
            # Alineación rígida completa (traslación + rotación) por gradiente
            # descendente sobre la suma de diferencias cuadráticas. Lento; para
            # producción usar una librería dedicada (SimpleITK/ANTs).
            # Aquí: solo traslación como aproximación de primer orden.
            pass
        out[g] = ndi_shift(mov, shift_vec, order=3, mode="constant", cval=0.0)
    return out


def displacement_field_register_gates(
    gated_volume: np.ndarray,
    *,
    ref_gate: int | None = None,
    smooth_sigma: float = 1.5,
    reg_lambda: float = 0.5,
    n_iter: int = 30,
    step_size: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Alinea cada gate al gate de referencia por campo de desplazamiento denso.

    Implementa un "demons"-like simplificado: gradiente descendente sobre
    SSD + regularización de suavidad del campo. Devuelve (gated_aligned, u_fields).

    u_fields tiene shape (n_gates, 3, Z, Y, X) con el desplazamiento en píxeles.
    """
    g4 = np.asarray(gated_volume, dtype=np.float64)
    if g4.ndim != 4 or g4.shape[0] < 2:
        return g4.copy(), np.zeros((g4.shape[0], 3) + g4.shape[1:], dtype=np.float64)
    n_gates, Z, Y, X = g4.shape
    if ref_gate is None:
        ref_gate = _largest_volume_gate(g4)
    ref = gaussian_filter(g4[ref_gate], sigma=smooth_sigma, mode="constant")

    aligned = np.empty_like(g4)
    u_all = np.zeros((n_gates, 3, Z, Y, X), dtype=np.float64)

    for g in range(n_gates):
        if g == ref_gate:
            aligned[g] = g4[g]
            continue
        mov = gaussian_filter(g4[g], sigma=smooth_sigma, mode="constant")
        u = np.zeros((3, Z, Y, X), dtype=np.float64)

        # Coordenadas de malla para map_coordinates
        zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]

        for _ in range(n_iter):
            # Warp de mov con el campo actual
            coords_z = zz + u[0]
            coords_y = yy + u[1]
            coords_x = xx + u[2]
            warped = map_coordinates(mov, [coords_z, coords_y, coords_x], order=1, mode="constant", cval=0.0)

            # Error SSD
            diff = warped - ref

            # Gradiente de la imagen warpada (para la dirección de descenso)
            gz, gy, gx = np.gradient(warped)

            # Actualización tipo demons: u -= step * diff * grad(warped) / (|grad|^2 + eps)
            norm = gz * gz + gy * gy + gx * gx + 1e-8
            du_z = -step_size * diff * gz / norm
            du_y = -step_size * diff * gy / norm
            du_x = -step_size * diff * gx / norm

            # Regularización: suavizar el incremento del campo
            du_z = gaussian_filter(du_z, sigma=reg_lambda, mode="constant")
            du_y = gaussian_filter(du_y, sigma=reg_lambda, mode="constant")
            du_x = gaussian_filter(du_x, sigma=reg_lambda, mode="constant")

            u[0] += du_z
            u[1] += du_y
            u[2] += du_x

        # Warp final del gate original (sin suavizar) con el campo estimado
        coords_z = zz + u[0]
        coords_y = yy + u[1]
        coords_x = xx + u[2]
        aligned[g] = map_coordinates(g4[g], [coords_z, coords_y, coords_x], order=3, mode="constant", cval=0.0)
        u_all[g] = u

    return aligned, u_all


def motion_frozen_sum(
    gated_volume: np.ndarray,
    *,
    method: str = "displacement",
    ref_gate: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Imagen de perfusión motion-frozen: promedio de gates alineados.

    method: "rigid" | "displacement" | "sinogram" (este último requiere
    proyecciones, no volumen; lanzará ValueError aquí).
    """
    g4 = np.asarray(gated_volume, dtype=np.float64)
    if g4.ndim != 4 or g4.shape[0] < 2:
        return g4.mean(axis=0) if g4.ndim == 4 else np.asarray(g4, dtype=np.float64)

    method = str(method).strip().lower()
    if method == "rigid":
        aligned = rigid_register_gates(g4, ref_gate=ref_gate, **kwargs)
    elif method == "displacement":
        aligned, _ = displacement_field_register_gates(g4, ref_gate=ref_gate, **kwargs)
    elif method == "sinogram":
        raise ValueError(
            "El método 'sinogram' opera sobre proyecciones, no sobre el volumen "
            "reconstruido. Usar motion_frozen_sinogram_recon() en su lugar."
        )
    else:
        raise ValueError(f"Método motion-frozen desconocido: {method}")
    return aligned.mean(axis=0)


def rigid_register_gates_stable(
    gated_volume: np.ndarray,
    *,
    ref_gate: int | None = None,
    smooth_sigma: float = 1.0,
    heart_percentile: float = 70.0,
) -> np.ndarray:
    """Alinea cada gate por traslación usando el CONTEXTO ESTABLE (sin corazón).

    El centroide de todo el volumen está dominado por el corazón (lo más
    caliente), así que alinearlo por centroide global arrastra el latido: cada
    gate se traslada para que el corazón coincida, lo que CONGELA la
    contracción en el promedio (el cine MF-por-gate queda con mini-
    desplazamientos raros en vez del latido real).

    La física correcta para un cine: el blur del ungated no viene solo del
    latido sino también del desplazamiento GLOBAL del paciente/tórax entre
    ángulos. Ese desplazamiento se puede estimar mejor desde las estructuras
    ESTABLES (tórax, hígado, fondo) que no laten. Entonces:

      1. Máscara de "contexto" = todo el volumen EXCEPTO el corazón (los
         voxels más calientes, > percentil 70, son miocardio -> se excluyen).
      2. Centroide del contexto por gate -> shift = alineación del cuerpo.
      3. Se aplica el shift al gate COMPLETO (corazón incluido).

    Resultado: se corrige el desplazamiento global pero la contracción
    cardíaca (cambio de volumen ED->ES) se CONSERVA en el cine.
    """
    g4 = np.asarray(gated_volume, dtype=np.float64)
    if g4.ndim != 4 or g4.shape[0] < 2:
        return g4.copy()
    n_gates = g4.shape[0]
    if ref_gate is None:
        ref_gate = _largest_volume_gate(g4)

    def _com_context(vol: np.ndarray) -> tuple[float, float, float]:
        """Centroide del contexto estable (excluye el corazón caliente)."""
        v = gaussian_filter(vol, sigma=smooth_sigma, mode="constant") if smooth_sigma > 0 else vol
        thr = float(np.percentile(v, heart_percentile)) if v.size else 0.0
        ctx = np.where(v <= thr, v, 0.0)  # anular el corazón (lo más caliente)
        return _center_of_mass_3d(ctx)

    com_ref = _com_context(g4[ref_gate])
    out = np.empty_like(g4)
    for g in range(n_gates):
        if g == ref_gate:
            out[g] = g4[g]
            continue
        com_mov = _com_context(g4[g])
        shift_vec = [com_ref[0] - com_mov[0], com_ref[1] - com_mov[1], com_ref[2] - com_mov[2]]
        out[g] = ndi_shift(g4[g], shift_vec, order=3, mode="constant", cval=0.0)
    return out


def motion_frozen_per_gate(
    gated_volume: np.ndarray,
    *,
    method: str = "rigid",
    smooth_sigma: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Cine nítido: motion-frozen con referencia variable.

    En vez de alinear todos los gates a una única referencia (end-diastole),
    alinea todos los gates a **cada gate sucesivamente** y promedia. El
    resultado es un volumen 4D donde cada "gate" tiene la nitidez del MF pero
    en su propia fase del ciclo cardíaco.

    Útil para evaluar motilidad con mejor estadística que un gate individual.
    NO sirve para FEVI (el volumen ya no cambia entre gates, solo se ve la
    misma anatomía en distinta posición).

    Devuelve (n_gates, Z, Y, X) mismo shape que la entrada.
    """
    g4 = np.asarray(gated_volume, dtype=np.float64)
    if g4.ndim != 4 or g4.shape[0] < 2:
        return g4.copy()
    n_gates = g4.shape[0]
    out = np.empty_like(g4)
    for ref in range(n_gates):
        if method == "displacement":
            aligned, _u = displacement_field_register_gates(
                g4, ref_gate=ref, smooth_sigma=smooth_sigma, **kwargs
            )
        elif method == "stable":
            # Contexto estable: conserva el latido (no congela la contracción).
            aligned = rigid_register_gates_stable(g4, ref_gate=ref, smooth_sigma=smooth_sigma, **kwargs)
        else:
            aligned = rigid_register_gates(g4, ref_gate=ref, smooth_sigma=smooth_sigma, **kwargs)
        out[ref] = aligned.mean(axis=0)
    return out


def motion_frozen_sinogram_recon(
    projections_gated: np.ndarray,
    *,
    angles_deg: np.ndarray,
    ref_gate: int | None = None,
    **recon_kwargs,
) -> np.ndarray:
    """Motion-frozen en el dominio del sinograma (proyecciones).

    Estima el desplazamiento 3D del corazón por gate, lo proyecta a cada ángulo
    como shift 2D, y corrige las proyecciones antes de reconstruir. Es el
    enfoque más físico pero requiere la cinemática 3D desde datos 2D.

    NOTA: Implementación placeholder. Requiere estimar el campo 3D desde las
    proyecciones (problema inverso), típicamente por retroproyección de shifts
    2D estimados en el sinograma. Por ahora devuelve la reconstrucción sin
    corrección con un warning.
    """
    import warnings
    warnings.warn(
        "motion_frozen_sinogram_recon: estimación 3D->2D no implementada aún. "
        "Se devuelve la reconstrucción promedio sin corrección.",
        RuntimeWarning,
    )
    from core.raw_projections import ungate_projections
    from core.raw_reconstruction import reconstruct_projection_volume, ProjectionFilterConfig

    ung = ungate_projections(projections_gated)
    # Reconstrucción estándar sin corrección de movimiento cardíaco.
    return reconstruct_projection_volume(
        ung, angles_deg, method="fbp",
        projection_filter=ProjectionFilterConfig("butterworth", 0.52, 5),
        **recon_kwargs,
    )
