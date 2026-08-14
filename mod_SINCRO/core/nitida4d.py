"""SINCRO - core.nitida4d  (4D-OSEM: regularización temporal entre gates)

Reconstrucción 4D-OSEM para SPECT miocárdico gated de bajo conteo.

A diferencia de NITIDA III (MAP-OSEM con prior ESPACIAL por gate, sin
acoplamiento entre gates), el 4D-OSEM agrega un **prior TEMPORAL** que acopla
gates vecinos DENTRO del update OSEM:

    V_g^{n+1} = V_g^n / [ denom_g * (1 + beta_esp*dU_esp + beta_temp*dU_temp) ]
                       * backproj( p_g / A V_g^n )

con dU_temp = psi(V_g - V_{g-1}) + psi(V_g - V_{g+1}), psi = influencia Huber.

Por qué NO aplasta el latido (la diferencia con el motion-frozen, que falló):
  - El ruido de Poisson es INCORRELADO entre gates; la señal cardíaca es SUAVE
    en el tiempo. El prior temporal transfiere estadística entre vecinos sin
    imponer un valor fijo.
  - La transición ED->ES es un "borde temporal" REAL. La función de Huber es
    cuadrática para |dV|<=delta (ruido: se suaviza) y LINEAL para |dV|>delta
    (borde real: NO se penaliza de más). Así se conserva el cambio de volumen.
  - NO hay promedio entre gates: cada gate conserva sus cuentas (reescalado
    por gate, como NITIDA III). FEVI y fase quedan intactos en principio.

Diseño propio sobre matemática publicada (Green OSL-MAP, priors de suavidad
temporal 4D en EM reconstruction). No copia ningún software comercial.

API:
  - temporal_huber_grad(cube, delta)     -> gradiente temporal por gate
  - nitida4d_osem_gated(...)             -> 4D-OSEM completo (gates, Z, Y, X)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _huber_psi(d: np.ndarray, delta: float) -> np.ndarray:
    """Función de influencia de Huber psi(d) = dU/dd.

    Cuadrática para |d|<=delta (ruido: se suaviza fuerte), lineal para
    |d|>delta (borde real: no se penaliza de más, preserva el latido).
    NO se normaliza por el máximo: hacerlo achata el gradiente en zonas de
    bajo contraste (donde vive el ruido) y anula el efecto del prior.
    """
    d = np.asarray(d, dtype=np.float64)
    absd = np.abs(d)
    return np.where(absd <= delta, d, delta * np.sign(d))


def temporal_huber_grad(cube: np.ndarray, *, delta: float = 0.1) -> np.ndarray:
    """Gradiente del prior de suavidad TEMPORAL (entre gates vecinos).

    Entrada: array (n_gates, ...) donde el eje 0 es el temporal y el resto son
    ejes espaciales (2D por slice (n_gates,H,W) o 3D (n_gates,Z,Y,X)). Devuelve
    dU_temp/dV con mismo shape: para cada gate g, psi(V_g - V_{g-1}) +
    psi(V_g - V_{g+1}). Los gates de los extremos tienen un solo vecino.
    Frontera CERRADA (el ciclo cardíaco NO se enrola).

    ``delta`` se interpreta como FRACCIÓN del rango de datos (si delta <= 1):
    el umbral absoluto es delta * (max-min). Así el mismo default 0.1 funciona
    para cualquier escala de cuentas.
    """
    c = np.asarray(cube, dtype=np.float64)
    if c.ndim < 2 or c.shape[0] < 2:
        return np.zeros_like(c)
    d_abs = float(delta)
    if d_abs <= 1.0:
        rng = float(c.max() - c.min())
        d_abs = max(1e-6, d_abs * rng)
    n = int(c.shape[0])
    out = np.zeros_like(c)
    for g in range(n):
        acc = np.zeros_like(c[g])
        if g > 0:
            acc = acc + _huber_psi(c[g] - c[g - 1], d_abs)
        if g < n - 1:
            acc = acc + _huber_psi(c[g] - c[g + 1], d_abs)
        out[g] = acc
    return out


def nitida4d_osem_gated(
    projections: np.ndarray,
    angles_deg: np.ndarray,
    *,
    iterations: int = 2,
    subsets: int = 4,
    beta_spatial: float = 0.0,
    beta_temporal: float = 0.3,
    delta_temporal: float = 0.1,
    psf=None,
    slice_range: tuple[int, int] | None = None,
    rescale: bool = True,
    ref_volume: np.ndarray | None = None,
    progress_callback=None,
) -> np.ndarray:
    """4D-OSEM gated: OSEM con prior temporal Huber DENTRO del update (OSL).

    Reconstruye los n_gates gates JUNTOS, compartiendo información entre
    vecinos temporales en cada iteración. A diferencia de promediar gates
    (motion-frozen, que congela el latido), el prior temporal solo frena el
    ruido incorrelado; la señal cardíaca (suave en el tiempo) y la contracción
    (borde temporal real, protegido por Huber) se conservan.

    ``beta_temporal``: fuerza del acoplamiento temporal (0 = OSEM por gate
    independiente = comportamiento previo). Típico 0.2-0.5.
    ``delta_temporal``: umbral de Huber para el borde temporal. Mayor = más
    suavizado temporal (riesgo de comer el latido); menor = más preservación
    del movimiento pero menos limpieza. Default 0.1 (fracción del rango).
    ``beta_spatial``: opcional, prior espacial Huber (Pilar C de NITIDA III)
    combinado con el temporal. 0 = solo temporal.
    ``rescale``: conserva la suma de cuentas por gate (FEVI). ``ref_volume``:
    referencia de cuentas (p.ej. el gated OSEM del pipeline) para evitar una
    segunda reconstrucción.

    Devuelve (n_gates, Z, Y, X), mismo shape que reconstruct_gated_projection_volume.
    """
    from core.raw_reconstruction import (
        _backproject_slice,
        _build_sensitivity_cache,
        _forward_project_slice,
        reconstruct_gated_projection_volume,
    )

    proj = np.asarray(projections, dtype=np.float64)
    if proj.ndim != 4:
        raise ValueError(f"projections debe ser 4D (gates,angles,H,W); recibió {proj.shape}")
    n_gates, n_angles, H, W = proj.shape
    theta = np.asarray(angles_deg, dtype=np.float64)
    detector_size = int(W)
    out_size = int(W)

    # Rango axial (feta) a reconstruir.
    if slice_range is not None:
        z0, z1 = int(slice_range[0]), int(slice_range[1])
    else:
        z0, z1 = 0, int(H) - 1
    z_indices = list(range(z0, z1 + 1))
    n_slices = len(z_indices)

    # Sensibilidad por subset: se calcula UNA vez (todos los gates/slices
    # comparten geometría) y se reutiliza.
    effective_subsets = max(1, min(int(subsets), int(theta.size)))
    sensitivity_cache = _build_sensitivity_cache(
        theta, subsets=effective_subsets, detector_size=detector_size, output_size=out_size, psf=psf
    )
    angle_indices = np.arange(theta.size)
    eps = 1e-6

    use_spatial = float(beta_spatial) > 0.0
    use_temporal = float(beta_temporal) > 0.0 and n_gates >= 3
    if use_spatial:
        from core.nitida3 import huber_prior_grad

    # Estado 4D: (n_gates, n_slices, W, W). Inicialización constante por gate.
    init_val = max(float(proj.mean()), 1.0)
    images = np.full((n_gates, n_slices, out_size, out_size), init_val, dtype=np.float64)

    total_steps = max(1, int(iterations)) * effective_subsets * n_slices
    step = 0

    for _it in range(max(1, int(iterations))):
        for subset_id in range(effective_subsets):
            idx = angle_indices[subset_id::effective_subsets]
            if idx.size == 0:
                continue
            theta_sub = theta[idx]
            sensitivity = sensitivity_cache.get(subset_id)
            for si, z in enumerate(z_indices):
                # Prior temporal: se calcula sobre el estado 4D ACTUAL del slice,
                # UNA vez por (subset, slice), antes del update de todos los gates.
                temp_grad = None
                if use_temporal:
                    temp_grad = temporal_huber_grad(images[:, si, :, :], delta=float(delta_temporal))
                for g in range(n_gates):
                    # Sinograma del gate g, slice z, ángulos del subset.
                    sino_sub = proj[g, idx, z, :].T  # (detector, n_angles_sub)
                    measured = np.clip(sino_sub, 0.0, None)
                    image = images[g, si]

                    estimated = _forward_project_slice(image, theta_sub, detector_size=detector_size, psf=psf)
                    ratio = measured / np.maximum(estimated, eps)
                    correction = _backproject_slice(ratio, theta_sub, output_size=out_size, psf=psf)

                    if sensitivity is not None:
                        denom = np.maximum(sensitivity, eps)
                    else:
                        denom = np.maximum(
                            _backproject_slice(np.ones_like(measured), theta_sub, output_size=out_size, psf=psf), eps
                        )

                    # Denominador OSL: espacial (opcional) + temporal (este).
                    factor = np.ones_like(image)
                    if use_spatial:
                        factor = factor + float(beta_spatial) * huber_prior_grad(image)
                    if use_temporal:
                        factor = factor + float(beta_temporal) * temp_grad[g]
                    denom = np.maximum(denom * factor, 1e-3)

                    image = image * (correction / denom)
                    images[g, si] = np.clip(image, 0.0, None)

                step += 1
                if progress_callback is not None:
                    progress_callback(min(1.0, step / total_steps), f"4D-OSEM slice {si + 1}/{n_slices}")

    # Reescalado por gate: conserva la suma de cuentas de cada gate (FEVI).
    out = images
    if rescale:
        if ref_volume is not None:
            vol_ref = np.asarray(ref_volume, dtype=np.float64)
            # ref puede ser el volumen completo (n_gates, H, W, W): recortar a la feta.
            ref_slices = vol_ref[:, z0:z1 + 1] if vol_ref.ndim == 4 and vol_ref.shape[1] == H else vol_ref
        else:
            ref_full = reconstruct_gated_projection_volume(
                proj, angles_deg, method="osem",
                projection_filter=None, iterations=iterations, subsets=subsets,
                psf=psf, slice_range=slice_range,
            )
            ref_slices = ref_full
        for g in range(n_gates):
            s_out = float(out[g].sum())
            s_ref = float(ref_slices[g].sum())
            if s_out > 0.0 and s_ref > 0.0:
                out[g] = out[g] * (s_ref / s_out)

    # Reensamblar al volumen completo (n_gates, H, W, W) si se reconstruyó feta.
    if slice_range is not None and n_slices != H:
        full = np.zeros((n_gates, H, out_size, out_size), dtype=np.float64)
        for si, z in enumerate(z_indices):
            full[:, z] = out[:, si]
        return full
    return out
