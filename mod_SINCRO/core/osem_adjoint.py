# -*- coding: utf-8 -*-
"""OSEM con proyector ray-driven adyunto — alternativa al OSEM de raw_reconstruction.py.

Diferencias clave con el OSEM actual (_iterative_reconstruct_slice):
1. Proyector forward y backward son adyuntos exactos (mismo ray-tracing).
2. Interpolación lineal a lo largo del rayo (no rotación bilineal acumulada).
3. Sin rotación de imagen completa → menos blur acumulado.
4. Centro de rotación explícito y consistente.

Uso:
    from core.osem_adjoint import osem_adjoint_reconstruct_slice
    recon = osem_adjoint_reconstruct_slice(sinogram, theta, output_size=64, iterations=4, subsets=8)

NO toca el OSEM existente. Es un módulo independiente para comparar y validar.
"""
from __future__ import annotations

import numpy as np


def _build_ray_coords(H: int, W: int, angle_rad: float, det_size: int):
    """Precomputa coordenadas de rayos para forward y backward (compartidas)."""
    cx, cy = W / 2.0, H / 2.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    max_dim = max(H, W)
    max_len = max_dim * np.sqrt(2.0)
    n_steps = int(np.ceil(max_len)) + 1
    t = np.linspace(-max_len / 2.0, max_len / 2.0, n_steps)
    step_len = float(t[1] - t[0]) if n_steps > 1 else 1.0
    det_positions = np.linspace(-1.0, 1.0, det_size)

    # Coordenadas de todos los puntos de todos los rayos: (det_size, n_steps)
    d_grid = det_positions[:, None]  # (D, 1)
    t_grid = t[None, :]             # (1, S)
    xs = cx + d_grid * (-sin_a) * max_dim / 2.0 - cos_a * max_len / 2.0 + cos_a * t_grid
    ys = cy + d_grid * cos_a * max_dim / 2.0 - sin_a * max_len / 2.0 + sin_a * t_grid

    xi = np.clip(xs, 0, W - 1.001)
    yi = np.clip(ys, 0, H - 1.001)
    x0i = np.floor(xi).astype(np.int32)
    y0i = np.floor(yi).astype(np.int32)
    x1i = np.minimum(x0i + 1, W - 1)
    y1i = np.minimum(y0i + 1, H - 1)
    fx = xi - x0i
    fy = yi - y0i
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    in_bounds = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return (y0i, x0i, y1i, x1i, w00, w10, w01, w11, in_bounds, step_len, n_steps)


def _ray_forward_project(image: np.ndarray, angle_rad: float, det_size: int) -> np.ndarray:
    """Proyección forward ray-driven vectorizada para UN ángulo."""
    H, W = image.shape
    y0i, x0i, y1i, x1i, w00, w10, w01, w11, in_bounds, _, _ = _build_ray_coords(H, W, angle_rad, det_size)
    vals = (
        image[y0i, x0i] * w00 +
        image[y0i, x1i] * w10 +
        image[y1i, x0i] * w01 +
        image[y1i, x1i] * w11
    )
    vals[~in_bounds] = 0.0
    return vals.sum(axis=1)


def _ray_backproject(profile: np.ndarray, angle_rad: float, out_size: int) -> np.ndarray:
    """Retroproyección ray-driven vectorizada para UN ángulo (adyunto del forward)."""
    H = W = out_size
    y0i, x0i, y1i, x1i, w00, w10, w01, w11, in_bounds, step_len, n_steps = _build_ray_coords(H, W, angle_rad, len(profile))
    # profile: (D,) → (D, 1) multiplicado por step_len, broadcast a (D, S).
    weight = np.broadcast_to(profile[:, None] * step_len, in_bounds.shape).copy()
    weight[~in_bounds] = 0.0
    volume = np.zeros((H, W), dtype=np.float64)
    # Acumular usando np.add.at (correcto para índices repetidos).
    np.add.at(volume, (y0i.ravel(), x0i.ravel()), (weight * w00).ravel())
    np.add.at(volume, (y0i.ravel(), x1i.ravel()), (weight * w10).ravel())
    np.add.at(volume, (y1i.ravel(), x0i.ravel()), (weight * w01).ravel())
    np.add.at(volume, (y1i.ravel(), x1i.ravel()), (weight * w11).ravel())
    return volume


def osem_adjoint_reconstruct_slice(
    sinogram: np.ndarray,
    theta: np.ndarray,
    *,
    output_size: int = 64,
    iterations: int = 4,
    subsets: int = 8,
    eps: float = 1e-6,
) -> np.ndarray:
    """OSEM con proyector ray-driven adyunto.

    Parameters
    ----------
    sinogram : (det_size, n_angles) proyecciones medidas.
    theta : (n_angles,) ángulos en grados.
    output_size : tamaño de la imagen de salida.
    iterations : número de iteraciones.
    subsets : número de subsets.
    eps : piso numérico para evitar división por cero.

    Returns
    -------
    (output_size, output_size) imagen reconstruida.
    """
    measured = np.clip(np.asarray(sinogram, dtype=np.float64), 0.0, None)
    theta_rad = np.deg2rad(np.asarray(theta, dtype=np.float64))
    det_size = measured.shape[0]
    n_angles = measured.shape[1]
    out_size = int(output_size)

    if measured.ndim != 2 or measured.shape[1] != n_angles:
        raise ValueError("sinogram/theta incompatibles")

    subset_count = max(1, min(int(subsets), n_angles))
    angle_indices = np.arange(n_angles)

    # Imagen inicial: media de las proyecciones.
    image = np.full((out_size, out_size), max(float(measured.mean()), 1.0), dtype=np.float64)

    for _iter in range(max(1, int(iterations))):
        for subset_id in range(subset_count):
            idx = angle_indices[subset_id::subset_count]
            if idx.size == 0:
                continue
            theta_sub = theta_rad[idx]
            measured_sub = measured[:, idx]

            # Forward project current estimate.
            estimated = np.zeros((det_size, len(idx)), dtype=np.float64)
            for i, angle in enumerate(theta_sub):
                estimated[:, i] = _ray_forward_project(image, angle, det_size)

            # Ratio sinograma.
            ratio = measured_sub / np.maximum(estimated, eps)

            # Backproject correction.
            correction = np.zeros((out_size, out_size), dtype=np.float64)
            for i, angle in enumerate(theta_sub):
                correction += _ray_backproject(ratio[:, i], angle, out_size)

            # Sensitivity (backproject ones).
            sensitivity = np.zeros((out_size, out_size), dtype=np.float64)
            ones = np.ones((det_size, len(idx)), dtype=np.float64)
            for i, angle in enumerate(theta_sub):
                sensitivity += _ray_backproject(ones[:, i], angle, out_size)

            # OSEM update.
            image *= correction / np.maximum(sensitivity, eps)
            image = np.clip(image, 0.0, None)

    return image
