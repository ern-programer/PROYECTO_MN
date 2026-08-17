# -*- coding: utf-8 -*-
"""Prisma anular miocárdico: una feta SA con forma de anillo, extruida en Z.

Genera la superficie 3D de una feta de eje corto como un prisma anular:
- Cara externa: epicardio (radio externo).
- Cara interna: endocardio (radio interno).
- Tapas anulares en las dos caras (espesor visible).

A diferencia del cubo rectangular, esto mantiene la forma real del miocardio
en cada corte, sin fondo ni hígado.
"""
from __future__ import annotations

import numpy as np


def ring_prism_mesh(
    endo_radii: np.ndarray,
    epi_radii: np.ndarray,
    z_center_mm: float,
    thickness_mm: float,
    *,
    centers_mm: np.ndarray | None = None,
):
    """Prisma anular de una feta SA extruida al espesor de corte.

    Parameters
    ----------
    endo_radii : (n_angles,) radios endocárdicos en mm para UN corte.
    epi_radii : (n_angles,) radios epicárdicos del mismo corte.
    z_center_mm : posición axial del centro de la feta en mm.
    thickness_mm : espesor de corte en mm.
    centers_mm : (2,) offset (x,y) del centro del VI en mm, o None.

    Returns
    -------
    pv.PolyData — superficie cerrada del anillo extruido.
    """
    import pyvista as pv

    endo = np.asarray(endo_radii, dtype=np.float64)
    epi = np.asarray(epi_radii, dtype=np.float64)
    n_angles = len(endo)
    z0 = float(z_center_mm) - 0.5 * float(thickness_mm)
    z1 = float(z_center_mm) + 0.5 * float(thickness_mm)

    cx, cy = 0.0, 0.0
    if centers_mm is not None:
        c = np.asarray(centers_mm, dtype=np.float64)
        if c.shape == (2,):
            cx, cy = float(c[0]), float(c[1])

    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # 4 anillos: epi_inf, epi_sup, endo_inf, endo_sup.
    pts = []
    for z in (z0, z1):
        pts.append(np.column_stack([cx + epi * cos_a, cy + epi * sin_a, np.full(n_angles, z)]))
    for z in (z0, z1):
        pts.append(np.column_stack([cx + endo * cos_a, cy + endo * sin_a, np.full(n_angles, z)]))
    points = np.vstack(pts)

    # Espesor de pared por punto: distancia epi↔endo.
    thickness = np.zeros(4 * n_angles, dtype=np.float64)
    for a in range(n_angles):
        t = float(np.linalg.norm(points[a] - points[2 * n_angles + a]))
        thickness[a] = t  # epi_inf
        thickness[a + n_angles] = t  # epi_sup
        thickness[a + 2 * n_angles] = t  # endo_inf
        thickness[a + 3 * n_angles] = t  # endo_sup

    faces = []
    # Caras laterales externas (epi): quads entre epi_inf y epi_sup.
    for a in range(n_angles):
        a2 = (a + 1) % n_angles
        p0 = a
        p1 = a2
        p2 = a2 + n_angles
        p3 = a + n_angles
        faces.extend([(3, p0, p1, p2), (3, p0, p2, p3)])

    # Caras laterales internas (endo): quads invertidos (normales hacia adentro).
    for a in range(n_angles):
        a2 = (a + 1) % n_angles
        p0 = a + 2 * n_angles
        p1 = a2 + 2 * n_angles
        p2 = a2 + 3 * n_angles
        p3 = a + 3 * n_angles
        faces.extend([(3, p0, p2, p1), (3, p0, p3, p2)])

    # Tapas anulares: conectan epi con endo en cada cara.
    for ring in (0, 1):  # 0=inf, 1=sup
        epi_base = ring * n_angles
        endo_base = (2 + ring) * n_angles
        flip = (ring == 1)
        for a in range(n_angles):
            a2 = (a + 1) % n_angles
            q = [epi_base + a, epi_base + a2, endo_base + a2, endo_base + a]
            if flip:
                q = [q[0], q[3], q[2], q[1]]
            faces.extend([(3, q[0], q[1], q[2]), (3, q[0], q[2], q[3])])

    face_arr = np.hstack([np.asarray(f).ravel() for f in faces])
    mesh = pv.PolyData(points, face_arr)
    mesh.point_data["thickness"] = thickness
    return mesh
