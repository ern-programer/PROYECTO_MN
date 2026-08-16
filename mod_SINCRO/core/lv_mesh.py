# -*- coding: utf-8 -*-
"""Conversión de los radios ECTb (cavidad/pared del VI) a meshes 3D.

El método ECTb (`core.ectb_lv.analyze_lv_ectb`) devuelve, para cada gate y cada
corte válido de eje corto, los radios endocárdico y epicárdico en mm a lo largo
de N ángulos (coordenadas cilíndricas alrededor del eje largo del VI). Este
módulo convierte esas superficies cilíndricas en meshes trianguladas 3D para
renderizarlas con PyVista: el "fantasma de alambre" del VI (ED fija + gate
actual animado) y la superficie del miocardio.

Convención de ejes del volumen SA reorientado en SINCRO:
  - eje 0 (k): ápex → base (eje largo del VI)
  - ejes 1/2: plano del corte de eje corto
El eje largo del VI es el eje Z del mesh 3D (para que el VI "apunte" hacia
arriba en la vista inicial). Los radios ECTb ya vienen en mm, así que el mesh
está en mm reales (escala física, no píxeles).
"""
from __future__ import annotations

import numpy as np


def radii_to_points(
    radii_mm: np.ndarray,
    z_positions_mm: np.ndarray,
    n_angles: int,
    centers_mm: np.ndarray | None = None,
) -> np.ndarray:
    """Convierte radios cilíndricos por corte a una nube de puntos 3D.

    Parameters
    ----------
    radii_mm : (n_slices, n_angles) radios en mm para UN gate.
    z_positions_mm : (n_slices,) posición axial de cada corte en mm
        (ápex=0 hacia base, o la coordenada que corresponda).
    n_angles : cantidad de ángulos (para regenerar el vector angular).

    Returns
    -------
    pts : (n_slices * n_angles, 3) coordenadas (x, y, z) en mm.
    """
    radii_mm = np.asarray(radii_mm, dtype=np.float64)
    z_positions_mm = np.asarray(z_positions_mm, dtype=np.float64)
    n_slices = radii_mm.shape[0]
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    centers = np.zeros((n_slices, 2), dtype=np.float64)
    if centers_mm is not None:
        c = np.asarray(centers_mm, dtype=np.float64)
        if c.shape == (n_slices, 2):
            centers = c
    pts = np.empty((n_slices * int(n_angles), 3), dtype=np.float64)
    idx = 0
    for s in range(n_slices):
        r = radii_mm[s]
        pts[idx:idx + n_angles, 0] = centers[s, 0] + r * cos_a
        pts[idx:idx + n_angles, 1] = centers[s, 1] + r * sin_a
        pts[idx:idx + n_angles, 2] = z_positions_mm[s]
        idx += n_angles
    return pts


def _lathe_faces(n_slices: int, n_angles: int) -> np.ndarray:
    """Caras trianguladas de una superficie de revolución (cilindro abierto).

    Conecta anillos consecutivos de puntos con quads (dos triángulos).
    """
    faces = []
    for s in range(n_slices - 1):
        for a in range(n_angles):
            a2 = (a + 1) % n_angles
            p0 = s * n_angles + a
            p1 = s * n_angles + a2
            p2 = (s + 1) * n_angles + a2
            p3 = (s + 1) * n_angles + a
            faces.extend([(3, p0, p1, p2), (3, p0, p2, p3)])
    return np.asarray(faces, dtype=np.int64)


def _cap_fan_faces(center_idx: int, ring_start: int, n_angles: int, flip: bool) -> list:
    """Abanico de triángulos para tapar un anillo (ápex o base) con un centro."""
    faces = []
    for a in range(n_angles):
        a2 = (a + 1) % n_angles
        p0 = ring_start + a
        p1 = ring_start + a2
        if flip:
            faces.append((3, center_idx, p1, p0))
        else:
            faces.append((3, center_idx, p0, p1))
    return faces


def radii_to_mesh(
    radii_mm: np.ndarray,
    z_positions_mm: np.ndarray,
    *,
    cap_apex: bool = True,
    cap_base: bool = True,
):
    """Construye un mesh PyVista (superficie cerrada) de un gate del VI.

    La superficie es un "lathe" (revolución) de los radios alrededor del eje Z,
    con tapas opcionales en ápex (z min) y base (z max).

    Parameters
    ----------
    radii_mm : (n_slices, n_angles) radios en mm para UN gate.
    z_positions_mm : (n_slices,) posición axial de cada corte en mm.
    cap_apex / cap_base : si True, cierran los extremos con un punto central.

    Returns
    -------
    pv.PolyData
    """
    import pyvista as pv

    radii_mm = np.asarray(radii_mm, dtype=np.float64)
    n_slices, n_angles = radii_mm.shape

    pts = [radii_to_points(radii_mm, z_positions_mm, n_angles)]
    faces = [_lathe_faces(n_slices, n_angles)]

    # Tapa del ápex: punto central en z_min con radio 0 (o el radio medio del
    # primer anillo si no cierra a 0, p.ej. ápex con cavidad residual).
    if cap_apex:
        apex_center = np.array([[0.0, 0.0, float(z_positions_mm[0])]])
        base_idx = pts[0].shape[0]
        pts.append(apex_center)
        faces.append(np.asarray(
            _cap_fan_faces(base_idx, 0, n_angles, flip=False), dtype=np.int64))

    # Tapa de la base: punto central en z_max.
    if cap_base:
        base_center = np.array([[0.0, 0.0, float(z_positions_mm[-1])]])
        c_idx = sum(p.shape[0] for p in pts)
        ring_start = (n_slices - 1) * n_angles
        pts.append(base_center)
        faces.append(np.asarray(
            _cap_fan_faces(c_idx, ring_start, n_angles, flip=True), dtype=np.int64))

    points = np.vstack(pts)
    face_arr = np.hstack([f.ravel() for f in faces])
    mesh = pv.PolyData(points, face_arr)
    return mesh


def lv_meshes_from_ectb(
    result,
    slice_mm: float,
    *,
    surface: str = "endo",
    seg=None,
    pixel_mm: tuple[float, float] | None = None,
    volume_shape: tuple[int, int, int] | None = None,
    smooth_temporal: float = 0.0,
) -> dict:
    """Genera los meshes de todos los gates a partir de un ECTbLVResult.

    Parameters
    ----------
    result : ECTbLVResult con endo_radii_mm/epi_radii_mm (n_gates, n_valid, n_ang).
    slice_mm : separación entre cortes en mm.
    surface : "endo" (cavidad) o "epi" (superficie externa del miocardio).
    seg : segmentación (para calcular el origen físico del volumen). Opcional.
    pixel_mm : (dy, dx) tamaño de píxel en mm. Opcional (para el origen).
    volume_shape : (K, H, W) shape del volumen SA reorientado. Opcional (para el origen).
    smooth_temporal : suavizado temporal de los radios entre gates (0=off). Opcional.

    Returns
    -------
    dict con:
        meshes: lista de pv.PolyData por gate (índice = gate)
        ed_gate / es_gate: índices 0-based de telediástole y telesístole
        z_positions_mm: (n_valid,) posiciones axiales usadas
        origin_mm: (ox, oy, oz) origen físico del volumen para alinear la cáscara
    """
    import pyvista as pv  # noqa: F401  (importación perezosa, por si no está)

    if not getattr(result, "available", False):
        raise ValueError("ECTbLVResult no disponible: " + str(getattr(result, "reason", "?")))

    radii_all = result.endo_radii_mm if surface == "endo" else result.epi_radii_mm
    radii_all = np.asarray(radii_all, dtype=np.float64)
    if radii_all.ndim != 3 or radii_all.shape[0] < 2:
        raise ValueError(f"radii con shape inesperado: {radii_all.shape}")

    n_gates, n_valid, n_ang = radii_all.shape
    dz = abs(float(slice_mm))
    if dz <= 0:
        raise ValueError("slice_mm inválido")

    # Posición axial de cada corte válido: ápex=0, base=(n_valid-1)*dz.
    # Los cortes válidos vienen ordenados ápex→base en apex_to_base_slices; los
    # radios están indexados por la posición en `valid` (que es el mismo orden
    # en que se muestrearon). Usamos el orden de muestreo: índice 0 = primer
    # corte válido. Si el ápex está primero (base_last=False), z crece hacia la
    # base; si la base está primero, invertimos para que z=0 sea siempre ápex.
    base_last = bool(getattr(result, "_base_last_cache", True))
    valid = list(getattr(result, "valid_slices", range(n_valid)))
    if len(valid) != n_valid:
        valid = list(range(n_valid))
    # Heurística: si el primer corte válido tiene índice mayor que el último,
    # el orden es base→ápex y hay que invertir el eje z.
    if len(valid) >= 2 and valid[0] > valid[-1]:
        base_last = False
    z = np.arange(n_valid, dtype=np.float64) * dz
    if not base_last:
        z = z[::-1].copy()

    meshes = []
    for g in range(n_gates):
        meshes.append(radii_to_mesh(radii_all[g], z, cap_apex=True, cap_base=True))

    # Suavizado temporal de los radios (entre gates): el latido es suave, no
    # hay saltos bruscos entre gates vecinos. Se aplica DESPUÉS de construir
    # las mallas para no alterar la geometría original del ECTb.
    if smooth_temporal > 0 and n_gates >= 3:
        endo_smooth = _smooth_radii_temporal(result.endo_radii_mm, sigma=smooth_temporal)
        epi_smooth = _smooth_radii_temporal(result.epi_radii_mm, sigma=smooth_temporal)
        # Reconstruir las mallas con los radios suavizados.
        meshes = []
        for g in range(n_gates):
            meshes.append(radii_to_mesh(endo_smooth[g], z, cap_apex=True, cap_base=True))
        # Actualizar los radios crudos para la cáscara (para que el espesor
        # refleje el suavizado temporal).
        result.endo_radii_mm = endo_smooth
        result.epi_radii_mm = epi_smooth

    # Calcular el origen físico del volumen SA para alinear la cáscara.
    # La cáscara está centrada en (0,0,0) con z=0 en el ápex. El volumen tiene
    # su origen en la esquina del array. Para que la cáscara (0,0,0) caiga en el
    # centro del VI, el origen debe ser el NEGATIVO del centro del VI:
    # pts - origin = 0 - (-c) = c.
    origin_mm = (0.0, 0.0, 0.0)
    if seg is not None and pixel_mm is not None and volume_shape is not None:
        try:
            centers = np.asarray(getattr(seg, "center_per_slice", None), dtype=np.float64)
            valid_slices = list(getattr(result, "valid_slices", range(n_valid)))
            if centers is not None and len(centers) > 0 and len(valid_slices) > 0:
                # Centro medio del VI en los cortes válidos (en píxeles).
                center_px = np.mean(centers[valid_slices], axis=0)  # (y, x)
                dz_mm = abs(float(slice_mm))
                dy_mm = abs(float(pixel_mm[0]))
                dx_mm = abs(float(pixel_mm[1]))
                # El origen físico para que la cáscara (0,0,0) caiga en el
                # centro del VI: (centro_x*dx, centro_y*dy, z_apex*dz).
                # Pero la cáscara está centrada en (0,0), así que el origen es
                # el NEGATIVO del centro del VI en mm.
                origin_mm = (
                    -float(center_px[1]) * dx_mm,  # ox = -centro_x * dx
                    -float(center_px[0]) * dy_mm,  # oy = -centro_y * dy
                    -float(valid_slices[0]) * dz_mm,  # oz = -primer corte válido * dz
                )
        except Exception:
            pass  # si falla, se usa (0,0,0) y el sampleo no funciona (fallback)

    return {
        "meshes": meshes,
        "ed_gate": int(getattr(result, "ed_gate", 1)) - 1,
        "es_gate": int(getattr(result, "es_gate", 1)) - 1,
        "z_positions_mm": z,
        "n_gates": n_gates,
        # Radios crudos para la cáscara del miocardio (endo/epi por gate).
        "endo_radii_mm": np.asarray(result.endo_radii_mm, dtype=np.float64),
        "epi_radii_mm": np.asarray(result.epi_radii_mm, dtype=np.float64),
        # Trayectoria real del centro del VI por corte válido, expresada como
        # offsets (x,y) en mm respecto del centro medio. El loft usa estos
        # offsets para conservar la curvatura del ventrículo.
        "center_offsets_mm": (
            np.column_stack([
                np.asarray(seg.center_per_slice, dtype=np.float64)[valid, 1] * float(pixel_mm[1]),
                np.asarray(seg.center_per_slice, dtype=np.float64)[valid, 0] * float(pixel_mm[0]),
            ])
            - np.nanmean(
                np.column_stack([
                    np.asarray(seg.center_per_slice, dtype=np.float64)[valid, 1] * float(pixel_mm[1]),
                    np.asarray(seg.center_per_slice, dtype=np.float64)[valid, 0] * float(pixel_mm[0]),
                ]), axis=0, keepdims=True,
            )
            if seg is not None and pixel_mm is not None else np.zeros((n_valid, 2), dtype=np.float64)
        ),
        # Origen físico del volumen para samplear la actividad.
        "origin_mm": origin_mm,
    }


def _smooth_mesh_laplacian(mesh, n_iter: int = 10, relaxation: float = 0.1):
    """Suavizado Laplaciano de una malla PyVista (VTK).

    Cada vértice se mueve hacia el promedio de sus vecinos, `n_iter` veces,
    con paso `relaxation` (0-1). Más iteraciones = más suave pero más lento
    y más contracción del volumen. relaxation chico = más estable.

    Returns
    -------
    pv.PolyData (nueva malla suavizada; la original no se toca).
    """
    import pyvista as pv
    if n_iter <= 0:
        return mesh
    try:
        smooth = mesh.smooth(n_iter=int(n_iter), relaxation_factor=float(relaxation))
        return smooth
    except Exception:
        return mesh


def _interpolate_radii_z(
    radii_mm: np.ndarray,
    z_positions_mm: np.ndarray,
    z_factor: float = 2.0,
):
    """Interpolación spline cúbica de los radios en el eje Z (cortes).

    Genera `z_factor` veces más cortes intermedios, suavizando la geometría
    axial (los "escalones" entre cortes discretos). No toca los ángulos.

    Parameters
    ----------
    radii_mm : (n_slices, n_angles) radios de UN gate.
    z_positions_mm : (n_slices,) posiciones axiales.
    z_factor : multiplicador de cortes (2.0 = el doble).

    Returns
    -------
    (radii_interp, z_interp) con shape (n_slices*z_factor, n_angles) y (n_slices*z_factor,).
    """
    from scipy.ndimage import zoom
    radii_mm = np.asarray(radii_mm, dtype=np.float64)
    z_positions_mm = np.asarray(z_positions_mm, dtype=np.float64)
    if z_factor <= 1.0:
        return radii_mm, z_positions_mm
    # Interpolar radios en Z (eje 0) con spline cúbico.
    radii_interp = zoom(radii_mm, (float(z_factor), 1.0), order=3, mode="nearest")
    # Interpolar posiciones Z linealmente.
    z_interp = np.linspace(z_positions_mm[0], z_positions_mm[-1], radii_interp.shape[0])
    return radii_interp, z_interp


def _smooth_radii_angular(radii_mm: np.ndarray, sigma: float = 1.0):
    """Suavizado angular de los radios (en el plano del corte).

    Cada corte de eje corto tiene radios por ángulo (0..2π). Si hay ruido o
    irregularidades (p.ej. un ángulo con radio muy distinto al vecino), se
    suaviza con un filtro gaussiano circular (el ángulo 0 y 2π son el mismo).
    """
    from scipy.ndimage import gaussian_filter1d
    radii_mm = np.asarray(radii_mm, dtype=np.float64)
    if sigma <= 0:
        return radii_mm
    # Suavizar cada corte por separado (eje 1 = ángulos), con wrap-around.
    return gaussian_filter1d(radii_mm, sigma=float(sigma), axis=1, mode="wrap")


def _smooth_radii_temporal(radii_all: np.ndarray, sigma: float = 1.0):
    """Suavizado temporal de los radios (entre gates).

    El latido es suave: el radio de un gate no debería saltar bruscamente al
    siguiente. Se suaviza con un filtro gaussiano en el eje 0 (gates).
    """
    from scipy.ndimage import gaussian_filter1d
    radii_all = np.asarray(radii_all, dtype=np.float64)
    if sigma <= 0 or radii_all.shape[0] < 3:
        return radii_all
    return gaussian_filter1d(radii_all, sigma=float(sigma), axis=0, mode="nearest")


def myocardium_shell_mesh(
    endo_radii_mm: np.ndarray,
    epi_radii_mm: np.ndarray,
    z_positions_mm: np.ndarray,
    *,
    smooth_iter: int = 0,
    smooth_relax: float = 0.1,
    interp_z: float = 1.0,
    apex_round: bool = False,
    smooth_angular: float = 0.0,
    smooth_temporal: float = 0.0,
    shape_index: float = 0.0,
    centers_mm: np.ndarray | None = None,
    apex_virtual_rings: int = 3,
    apex_taper: float = 0.0,
):
    """Cáscara del miocardio: superficie cerrada entre epicardio y endocardio.

    A diferencia de la isosuperficie del volumen crudo (que incluye fondo,
    hígado e intestino si superan el umbral), la cáscara ECTb es EXACTAMENTE
    la pared del VI: el volumen entre la superficie epicárdica (externa) y la
    endocárdica (interna). Sale una dona limpia sin importar el fondo.

    Parameters
    ----------
    endo_radii_mm : (n_slices, n_angles) radios endocárdicos de UN gate.
    epi_radii_mm : (n_slices, n_angles) radios epicárdicos del mismo gate.
    z_positions_mm : (n_slices,) posiciones axiales en mm.

    Returns
    -------
    pv.PolyData — superficie cerrada (epi por fuera, endo por dentro, tapas
    anulares en ápex y base conectando ambas).
    """
    import pyvista as pv

    endo = np.asarray(endo_radii_mm, dtype=np.float64)
    epi = np.asarray(epi_radii_mm, dtype=np.float64)
    z = np.asarray(z_positions_mm, dtype=np.float64)
    centers = np.zeros((endo.shape[0], 2), dtype=np.float64)
    if centers_mm is not None:
        c = np.asarray(centers_mm, dtype=np.float64)
        if c.shape == centers.shape:
            centers = c - np.nanmean(c, axis=0, keepdims=True)

    # Suavizado angular (opcional): elimina irregularidades entre ángulos de un
    # mismo corte (el ángulo 0 y 2π son el mismo, por eso mode="wrap").
    if smooth_angular > 0:
        endo = _smooth_radii_angular(endo, sigma=smooth_angular)
        epi = _smooth_radii_angular(epi, sigma=smooth_angular)

    # Seguir la línea central real del VI entre cortes, en vez de apilar todos
    # los anillos alrededor de un mismo eje artificial.
    if centers.shape[0] >= 3:
        try:
            from scipy.ndimage import gaussian_filter1d
            centers = gaussian_filter1d(centers, sigma=0.8, axis=0, mode="nearest")
        except Exception:
            pass

    # Modelado elipsoidal (opcional): el VI real no es un cilindro (barril),
    # es un elipsoide alargado (bala). Se modela como una función de z que
    # achata la base y afina el ápex, usando el índice de esfericidad del ECTb.
    if shape_index > 0.0:
        # shape_index = diámetro_corto / longitud_larga. 0.0 = cilindro puro,
        # 1.0 = esfera. El VI normal es ~0.5-0.7 (elipsoide alargado).
        # Se aplica una deformación que achata la base (z alto) y afina el ápex
        # (z bajo), manteniendo el volumen aproximadamente constante.
        z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)  # 0=ápex, 1=base
        # Factor de achatamiento: más achatado en la base, más cilíndrico en el ápex.
        # Usamos una función suave (coseno) que no cambia el volumen bruscamente.
        flatten = 1.0 - 0.3 * shape_index * np.cos(np.pi * z_norm)
        endo = endo * flatten[:, None]
        epi = epi * flatten[:, None]

    # Espesor muscular apical de referencia ANTES del afinado geométrico. El
    # taper modifica la silueta externa, pero no debe adelgazar el músculo.
    apex_wall_reference_mm = float(np.nanmedian(np.maximum(epi[0] - endo[0], 0.0)))

    # Afinado apical conservador. Los primeros contornos ECTb todavía
    # pertenecen a cortes con pared visible y suelen tener un radio grande;
    # extrapolarlos sin transición produce el aspecto de "barril". Este
    # taper afecta solo el tercio apical y converge suavemente a 1.0.
    taper = max(0.0, min(0.6, float(apex_taper)))
    if taper > 0.0 and len(z) >= 3:
        z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)
        apical_u = np.clip(z_norm / 0.35, 0.0, 1.0)
        factor = 1.0 - taper * (1.0 - apical_u) ** 2
        endo = endo * factor[:, None]
        epi = epi * factor[:, None]

    # Interpolación de radios en Z (opcional): suaviza los escalones axiales.
    if interp_z > 1.0:
        endo, z = _interpolate_radii_z(endo, z, z_factor=interp_z)
        epi, _ = _interpolate_radii_z(epi, z_positions_mm, z_factor=interp_z)
        try:
            from scipy.ndimage import zoom
            centers = zoom(centers, (float(interp_z), 1.0), order=3, mode="nearest")
        except Exception:
            centers = np.zeros((len(z), 2), dtype=np.float64)

    # Endocardio y epicardio comparten Z en los cortes medidos, pero tendrán
    # cierres apicales distintos: el epicardio llega a la punta externa y la
    # cavidad endocárdica termina más proximal, dejando músculo entre ambos.
    z_endo = z.copy()

    # Cierre apical progresivo con anillos virtuales. Así la punta converge
    # suavemente y no aparece la tapa plana con triángulos radiales.
    n_apex = max(0, int(apex_virtual_rings))
    apex_epi_target = None
    apex_endo_target = None
    if n_apex and endo.shape[0] >= 2:
        dz_apex = abs(float(z[1] - z[0])) if len(z) > 1 else 1.0
        wall_apex_mm = float(np.clip(apex_wall_reference_mm, 4.0, 12.0))
        endo_extra = []
        epi_extra = []
        z_epi_extra = []
        z_endo_extra = []
        c_extra = []
        # Orden distal→proximal: radios y Z crecen monótonamente hacia el
        # primer corte real. La versión anterior insertaba estos anillos en
        # orden Z inverso, cruzando caras y achatando la punta.
        extension = max(2.2 * dz_apex, 1.35 * wall_apex_mm)
        z_epi_tip = float(z[0] - extension)
        z_endo_tip = float(z_epi_tip + wall_apex_mm)
        for j in range(n_apex):
            # Casquete circular muestreado por ÁNGULO, no por Z. Así se
            # concentran anillos cerca del polo y los primeros triángulos no
            # forman un pincho largo. r=R·sin(phi), z=R·(1-cos(phi)).
            frac = j / float(n_apex)
            phi = 0.5 * np.pi * frac
            scale = float(np.sin(phi))
            axial_frac = float(1.0 - np.cos(phi))
            if j == 0:
                # Dos puntos sobre el eje largo: punta epicárdica externa y
                # cierre endocárdico más proximal. La separación axial es el
                # espesor apical medido, no una tapa plana concéntrica.
                endo_extra.append(np.zeros_like(endo[0]))
                epi_extra.append(np.zeros_like(epi[0]))
            else:
                endo_extra.append(endo[0] * scale)
                epi_extra.append(epi[0] * scale)
            if j == 0:
                z_epi_extra.append(z_epi_tip)
                z_endo_extra.append(z_endo_tip)
            else:
                z_epi_extra.append(z_epi_tip + axial_frac * (z[0] - z_epi_tip))
                z_endo_extra.append(z_endo_tip + axial_frac * (z[0] - z_endo_tip))
            c_extra.append(centers[0] + (centers[0] - centers[1]) * (1.0 - frac) * 0.35)
        tip_center = np.asarray(c_extra[0], dtype=np.float64)
        apex_epi_target = np.array([tip_center[0], tip_center[1], z_epi_tip], dtype=np.float64)
        apex_endo_target = np.array([tip_center[0], tip_center[1], z_endo_tip], dtype=np.float64)
        endo = np.vstack([np.asarray(endo_extra), endo])
        epi = np.vstack([np.asarray(epi_extra), epi])
        z = np.concatenate([np.asarray(z_epi_extra), z])
        z_endo = np.concatenate([np.asarray(z_endo_extra), z_endo])
        centers = np.vstack([np.asarray(c_extra), centers])

    n_slices, n_angles = endo.shape

    pts_endo = radii_to_points(endo, z_endo, n_angles, centers_mm=centers)
    pts_epi = radii_to_points(epi, z, n_angles, centers_mm=centers)
    n_ring = n_slices * n_angles
    apex_epi_rings_target = pts_epi[:n_apex * n_angles].copy() if n_apex > 0 else None
    apex_endo_rings_target = pts_endo[:n_apex * n_angles].copy() if n_apex > 0 else None

    # Puntos: primero todos los epicárdicos, después todos los endocárdicos.
    points = np.vstack([pts_epi, pts_endo])

    # Espesor de pared por punto: distancia euclidiana entre el punto del
    # epicardio y su pareja del endocardio (mismo ángulo, mismo corte).
    # Los puntos del epicardio son 0..n_ring-1; los del endocardio son
    # n_ring..2*n_ring-1. La pareja del punto i es i + n_ring.
    thickness = np.zeros(points.shape[0], dtype=np.float64)
    for i in range(n_ring):
        p_epi = points[i]
        p_endo = points[i + n_ring]
        thickness[i] = float(np.linalg.norm(p_epi - p_endo))
        thickness[i + n_ring] = thickness[i]  # mismo valor en la pareja

    faces = []
    # Cara EXTERNA (epicardio): normales hacia afuera.
    faces.append(_lathe_faces(n_slices, n_angles))
    # Cara INTERNA (endocardio): mismas conexiones pero con winding invertido
    # (normales hacia adentro), desplazada por n_ring.
    lathe = _lathe_faces(n_slices, n_angles).reshape(-1, 4)
    lathe_flip = lathe.copy()
    lathe_flip[:, [2, 3]] = lathe_flip[:, [3, 2]]  # invertir winding
    faces.append((lathe_flip + np.array([0, n_ring, n_ring, n_ring])).ravel())

    # Tapas ANULARES en ápex y base: conectan el anillo epi con el endo.
    def _annular_cap(s: int, flip: bool):
        cap = []
        e0 = s * n_angles
        d0 = n_ring + s * n_angles
        for a in range(n_angles):
            a2 = (a + 1) % n_angles
            # quad epi[a] -> epi[a+1] -> endo[a+1] -> endo[a]
            q = [e0 + a, e0 + a2, d0 + a2, d0 + a]
            if flip:
                q = [q[0], q[3], q[2], q[1]]
            cap.extend([(3, q[0], q[1], q[2]), (3, q[0], q[2], q[3])])
        return np.asarray(cap, dtype=np.int64)

    faces.append(_annular_cap(0, flip=False))            # ápex
    faces.append(_annular_cap(n_slices - 1, flip=True))  # base

    face_arr = np.hstack([np.asarray(f).ravel() for f in faces])
    mesh = pv.PolyData(points, face_arr)
    mesh.point_data["thickness"] = thickness
    mesh.field_data["shell_n_ring"] = np.asarray([n_ring], dtype=np.int64)
    mesh.field_data["shell_surface_cells"] = np.asarray(
        [2 * (n_slices - 1) * n_angles], dtype=np.int64,
    )

    # Ápex redondeado (opcional): reemplaza la tapa plana por una semiesfera
    # que cierra la punta suavemente. Se aplica DESPUÉS de construir la cáscara
    # para no romper la topología anular.
    if apex_round:
        mesh = _round_apex(mesh, z, endo, epi, n_angles)

    # Suavizado Laplaciano (opcional): suaviza la geometría completa.
    if smooth_iter > 0:
        mesh = _smooth_mesh_laplacian(mesh, n_iter=smooth_iter, relaxation=smooth_relax)

    # El suavizado puede abrir cada punto apical degenerado. Volver a colapsar
    # cada anillo sobre su propio centro conserva DOS cierres: epi distal y
    # endo proximal, manteniendo el espesor muscular longitudinal entre ambos.
    if (n_apex > 0 and apex_epi_target is not None and apex_endo_target is not None
            and apex_epi_rings_target is not None and apex_endo_rings_target is not None
            and mesh.n_points >= 2 * n_ring):
        pts = np.asarray(mesh.points, dtype=np.float64).copy()
        apex_count = n_apex * n_angles
        # Preservar el casquete completo. Si solo se restaura el polo después
        # del Laplaciano, los anillos vecinos suavizados quedan retraídos y se
        # forma visualmente un pincho entre ambos.
        pts[:apex_count] = apex_epi_rings_target
        pts[n_ring:n_ring + apex_count] = apex_endo_rings_target
        tip_epi = np.arange(n_angles, dtype=np.int64)
        tip_endo = n_ring + tip_epi
        pts[tip_epi] = apex_epi_target
        pts[tip_endo] = apex_endo_target
        mesh.points = pts
        tip_thickness = float(np.linalg.norm(apex_endo_target - apex_epi_target))
        if "thickness" in mesh.point_data:
            thickness_out = np.asarray(mesh.point_data["thickness"], dtype=np.float64).copy()
            thickness_out[tip_epi] = tip_thickness
            thickness_out[tip_endo] = tip_thickness
            mesh.point_data["thickness"] = thickness_out

    return mesh


def split_myocardium_shell(mesh):
    """Separa una cáscara miocárdica en superficies epi y endocárdica.

    Las tapas anulares (base y ápex) se agrupan con el epicardio: son la
    superficie que muestra el grosor muscular cuando se mira el corazón desde
    abajo. El slider de transparencia controla epi+tapas juntos, mientras que
    el endocardio (cavidad) permanece siempre opaco.

    Returns
    -------
    (epi_with_caps, endo) — dos pv.PolyData.
    """
    try:
        n_cells = int(np.asarray(mesh.field_data["shell_surface_cells"]).ravel()[0])
    except Exception as exc:
        raise ValueError("La malla no contiene metadatos de cáscara") from exc
    # epi + tapas anulares (base y ápex): lo que el slider controla.
    epi_idx = np.concatenate([
        np.arange(0, n_cells, dtype=np.int64),
        np.arange(2 * n_cells, mesh.n_cells, dtype=np.int64),
    ])
    epi = mesh.extract_cells(epi_idx).extract_surface(
        algorithm="dataset_surface"
    )
    # endocardio (cavidad): siempre opaco.
    endo = mesh.extract_cells(np.arange(n_cells, 2 * n_cells, dtype=np.int64)).extract_surface(
        algorithm="dataset_surface"
    )
    return epi, endo


def _round_apex(mesh, z, endo, epi, n_angles):
    """Reemplaza la tapa plana del ápex por una semiesfera suave.

    La tapa plana conecta endo y epi en el primer corte (z_min). El ápex
    redondeado agrega una semiesfera de radio = radio endocárdico del primer
    anillo, centrada en z_min, que cierra la punta suavemente sin agregar
    volumen extra (la semiesfera reemplaza la tapa plana, no se suma).
    """
    import pyvista as pv
    # Radio endocárdico del primer anillo (ápex): la semiesfera cierra la
    # cavidad, no la pared completa.
    r_apex = float(np.mean(endo[0]))
    if r_apex <= 0:
        return mesh
    z_apex = float(z[0])
    # Semiesfera que cierra la punta: centro en z_apex, radio = r_apex.
    # Solo la mitad inferior (hacia z-) es visible; la superior queda dentro.
    sphere = pv.Sphere(radius=r_apex, center=(0, 0, z_apex),
                       theta_resolution=n_angles, phi_resolution=max(8, n_angles // 2))
    try:
        merged = mesh.merge(sphere)
        return merged
    except Exception:
        return mesh
    except Exception:
        return mesh


def volume_to_isosurface(volume: np.ndarray, spacing_mm: tuple, level_frac: float = 0.5):
    """Isosuperficie de un volumen 3D (p.ej. miocardio reorientado SA).

    Parameters
    ----------
    volume : (K, H, W) volumen de actividad.
    spacing_mm : (dz, dy, dx) en mm.
    level_frac : umbral como fracción del máximo (0.5 = 50%, estilo QGS).

    Returns
    -------
    pv.PolyData (superficie del miocardio), o None si el volumen está vacío.
    """
    import pyvista as pv

    vol = np.asarray(volume, dtype=np.float64)
    if vol.size == 0 or vol.max() <= 0:
        return None
    level = float(vol.max()) * float(level_frac)

    grid = pv.ImageData()
    grid.dimensions = vol.shape[::-1]  # numpy (k,j,i) -> vtk (i,j,k)
    grid.spacing = (float(spacing_mm[2]), float(spacing_mm[1]), float(spacing_mm[0]))
    grid.point_data["values"] = vol.ravel(order="F")
    surf = grid.contour([level])
    if surf.n_points == 0:
        return None
    return surf


def sample_volume_on_mesh(mesh, volume: np.ndarray, spacing_mm: tuple, origin_mm=(0, 0, 0)):
    """Samplea la actividad de un volumen 3D sobre los vértices de una malla.

    Para cada vértice (x,y,z) de la malla, busca el valor del volumen en esa
    posición física (interpolación trilineal) y lo asigna como scalar. Sirve
    para "pintar" la cáscara del VI con la actividad real: los defectos
    hipocaptantes aparecen como zonas oscuras/frías.

    Parameters
    ----------
    mesh : pv.PolyData con points en mm (físicos).
    volume : (K, H, W) volumen de actividad (SA reorientado).
    spacing_mm : (dz, dy, dx) en mm.
    origin_mm : (ox, oy, oz) origen físico del volumen (default 0,0,0).

    Returns
    -------
    pv.PolyData — copia de la malla con `point_data["activity"]` asignado.
    """
    import pyvista as pv
    from scipy.ndimage import map_coordinates

    vol = np.asarray(volume, dtype=np.float64)
    if vol.size == 0:
        return mesh

    # Coordenadas físicas de los vértices (mm) → índices de voxel.
    pts = np.asarray(mesh.points, dtype=np.float64)
    dz, dy, dx = float(spacing_mm[0]), float(spacing_mm[1]), float(spacing_mm[2])
    ox, oy, oz = float(origin_mm[0]), float(origin_mm[1]), float(origin_mm[2])

    # Voxel = (fisico - origen) / spacing. El volumen es (K, H, W) = (z, y, x).
    iz = (pts[:, 2] - oz) / max(dz, 1e-6)
    iy = (pts[:, 1] - oy) / max(dy, 1e-6)
    ix = (pts[:, 0] - ox) / max(dx, 1e-6)

    # Interpolación trilineal (order=1). mode='constant' fuera del volumen → 0.
    coords = np.vstack([iz, iy, ix])
    vals = map_coordinates(vol, coords, order=1, mode="constant", cval=0.0)

    out = mesh.copy()
    out.point_data["activity"] = vals
    return out


def gradient_edge_mesh(volume: np.ndarray, spacing_mm: tuple, *,
                       smooth_sigma: float = 1.0, grad_percentile: float = 85.0,
                       level_frac: float = 0.5):
    """Malla del borde del miocardio por gradiente (el "borde blanco" de french).

    En vez de threshold por actividad (umbral fijo), detecta el borde como la
    superficie de máximo gradiente (la transición miocardio↔fondo). Es más
    robusto a ruido y a variaciones de captación porque no depende de un valor
    absoluto, sino del cambio espacial.

    Parameters
    ----------
    volume : (K, H, W) volumen de actividad (SA reorientado).
    spacing_mm : (dz, dy, dx) en mm.
    smooth_sigma : suavizado gaussiano previo (reduce ruido antes del gradiente).
    grad_percentile : percentil del gradiente para definir el borde (85 = top 15%).
    level_frac : umbral de la isosuperficie sobre el gradiente normalizado.

    Returns
    -------
    pv.PolyData — superficie del borde del miocardio, o None si falla.
    """
    import pyvista as pv
    from scipy.ndimage import gaussian_filter

    vol = np.asarray(volume, dtype=np.float64)
    if vol.size == 0 or vol.max() <= 0:
        return None

    # Suavizado previo: reduce ruido antes de calcular el gradiente.
    if smooth_sigma > 0:
        vol_s = gaussian_filter(vol, sigma=float(smooth_sigma))
    else:
        vol_s = vol

    # Gradiente 3D (magnitud). El "borde blanco" de french es la zona de alto
    # gradiente (transición miocardio↔fondo).
    gz, gy, gx = np.gradient(vol_s)
    grad_mag = np.sqrt(gz**2 + gy**2 + gx**2)

    # Umbral del gradiente: solo la zona de alto gradiente (el borde).
    # El percentil alto (85) captura el borde sin ruido de fondo.
    gmax = float(np.percentile(grad_mag, grad_percentile))
    if gmax <= 0:
        return None
    level = gmax * float(level_frac)

    # Isosuperficie sobre el gradiente (no sobre la actividad).
    grid = pv.ImageData()
    grid.dimensions = vol.shape[::-1]
    grid.spacing = (float(spacing_mm[2]), float(spacing_mm[1]), float(spacing_mm[0]))
    grid.point_data["values"] = grad_mag.ravel(order="F")
    surf = grid.contour([level])
    if surf.n_points == 0:
        return None
    return surf


def wall_volume_from_ectb(result, seg, pixel_mm, *, gate_index: int):
    """Reconstruye el volumen 3D del miocardio desde la segmentación ECTb.

    En vez de usar la actividad cruda (que tiene fondo/hígado), usa los radios
    ECTb (endo/epi por ángulo) para marcar los píxeles que son pared del VI.
    El resultado es un volumen binario (o de actividad) con solo el miocardio,
    sin fondo.

    Parameters
    ----------
    result : ECTbLVResult con endo_radii_mm/epi_radii_mm.
    seg : segmentación semilla (aporta forma de la máscara y centros por corte).
    pixel_mm : (dy, dx) del píxel, para pasar los radios de mm a píxeles.
    gate_index : gate a usar (no promedia: cada gate tiene su propia geometría).

    Returns
    -------
    (K, H, W) ndarray float64 — volumen binario del miocardio (1=pared, 0=fondo).
    """
    from core.ectb_lv import wall_segmentation_from_ectb
    import numpy as np

    # Generar la máscara del gate específico (no promediada).
    wall_seg = wall_segmentation_from_ectb(
        result, seg, pixel_mm, gate_index=gate_index
    )
    if wall_seg is None:
        return None

    # La máscara es (n_slices, H, W) con 1=pared, 0=fondo. Pero tiene solo los
    # cortes válidos (valid_slices), no todos los cortes del volumen. Para que
    # la isosuperficie la tome bien, expandirla a todos los cortes del volumen
    # (poner 0 en los no válidos).
    mask_full = np.zeros_like(seg.mask, dtype=np.float64)
    valid = list(getattr(result, "valid_slices", range(mask_full.shape[0])))
    if len(valid) == wall_seg.mask.shape[0]:
        for i, s in enumerate(valid):
            if 0 <= s < mask_full.shape[0]:
                mask_full[s] = wall_seg.mask[i]
    else:
        # Si no coinciden los shapes, usar la máscara tal cual (fallback).
        mask_full = wall_seg.mask
    return mask_full


def dynamic_volume_mesh(result, seg, pixel_mm, slice_mm, *, gate_index: int,
                        level: float = 0.5):
    """Malla 3D del miocardio reconstruido desde la segmentación ECTb.

    Reconstruye el volumen del miocardio (sin fondo) desde los radios ECTb del
    gate actual, apila los cortes en Z, y genera la isosuperficie. Es la
    "Reconstrucción Dinámica 3D": la pared exacta del VI latiendo, sin fondo.

    Parameters
    ----------
    result : ECTbLVResult con endo_radii_mm/epi_radii_mm.
    seg : segmentación semilla.
    pixel_mm : (dy, dx) del píxel.
    slice_mm : separación entre cortes.
    gate_index : gate a usar.
    level : umbral de la isosuperficie (0.5 = 50% de la máscara binaria).

    Returns
    -------
    pv.PolyData — superficie del miocardio del gate actual.
    """
    import pyvista as pv

    vol = wall_volume_from_ectb(result, seg, pixel_mm, gate_index=gate_index)
    if vol is None or vol.size == 0 or vol.max() <= 0:
        return None

    # Interpolar la máscara en Z (eje 0) para conectar los cortes: sin esto,
    # la isosuperficie sale en tiras porque los cortes están muy separados.
    try:
        from scipy.ndimage import zoom as _zoom
        z_factor = 3.0
        vol = _zoom(vol, (z_factor, 1.0, 1.0), order=1, mode="nearest")
        slice_mm = float(slice_mm) / z_factor
    except Exception:
        pass  # si scipy falla, se usa la máscara cruda (tiras, pero no crashea)

    # Isosuperficie de la máscara binaria (o de actividad si se samplea).
    grid = pv.ImageData()
    grid.dimensions = vol.shape[::-1]
    grid.spacing = (float(pixel_mm[1]), float(pixel_mm[0]), float(slice_mm))
    grid.point_data["values"] = vol.ravel(order="F")
    surf = grid.contour([float(level)])
    if surf.n_points == 0:
        return None
    return surf


def masked_activity_mesh(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing_mm: tuple,
    *,
    level_frac: float = 0.5,
):
    """Isosuperficie del volumen de actividad ENMASCARADO por la pared ECTb.

    En vez de samplear la actividad sobre la superficie de la cáscara (que puede
    quedar negra si el origen no está alineado), multiplicamos el volumen de
    actividad por la máscara de la pared (1=pared, 0=fondo) y hacemos la
    isosuperficie de ESO. El resultado es la pared del VI con la actividad
    real, sin fondo ni hígado.

    Es la opción "Máscara de actividad en cáscara": la geometría viene del ECTb
    (la máscara), pero el color/actividad viene del volumen real.

    Parameters
    ----------
    volume : (K, H, W) volumen de actividad (SA reorientado).
    mask : (K, H, W) máscara binaria de la pared (1=pared, 0=fondo).
    spacing_mm : (dz, dy, dx) en mm.
    level_frac : umbral de la isosuperficie sobre el volumen enmascarado.

    Returns
    -------
    pv.PolyData — superficie del miocardio con actividad real, o None si falla.
    """
    import pyvista as pv

    vol = np.asarray(volume, dtype=np.float64)
    msk = np.asarray(mask, dtype=np.float64)
    if vol.size == 0 or vol.max() <= 0 or msk.size == 0 or msk.max() <= 0:
        return None

    # Enmascarar: actividad solo dentro de la pared.
    vol_masked = vol * msk
    if vol_masked.max() <= 0:
        return None

    level = float(vol_masked.max()) * float(level_frac)

    grid = pv.ImageData()
    grid.dimensions = vol.shape[::-1]
    grid.spacing = (float(spacing_mm[2]), float(spacing_mm[1]), float(spacing_mm[0]))
    grid.point_data["values"] = vol_masked.ravel(order="F")
    surf = grid.contour([level])
    if surf.n_points == 0:
        return None
    return surf


def polar_texture_on_shell(mesh, polar_map: np.ndarray, z_positions_mm: np.ndarray):
    """Proyecta el mapa polar de perfusión sobre la cáscara del VI.

    En vez de samplear el volumen de actividad (que tiene problemas de
    alineación física), proyectamos el mapa polar 2D (apex-centro, base-borde)
    sobre la superficie 3D. Cada punto de la cáscara tiene coordenadas
    cilíndricas (ángulo, corte) que coinciden con el mapa polar (ángulo, radio).

    Es la opción "Mapa polar como textura": la geometría viene del ECTb (la
    cáscara), pero el color/actividad viene del mapa polar de perfusión (que ya
    está suavizado y normalizado).

    Parameters
    ----------
    mesh : pv.PolyData con points en mm (físicos).
    polar_map : (n_rings, n_angles) mapa polar de perfusión (0..1).
    z_positions_mm : (n_slices,) posiciones axiales de los cortes en mm.

    Returns
    -------
    pv.PolyData — copia de la malla con `point_data["perfusion"]` asignado.
    """
    import pyvista as pv

    pm = np.asarray(polar_map, dtype=np.float64)
    if pm.size == 0 or pm.max() <= 0:
        return mesh

    # Coordenadas cilíndricas de los vértices de la cáscara.
    pts = np.asarray(mesh.points, dtype=np.float64)
    # Ángulo: atan2(y, x) en grados 0..360.
    theta = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
    # Radio: distancia al eje z (en mm, normalizada a 0..1 por el rango de z).
    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    z = pts[:, 2]
    # Normalizar z a 0..1 (ápex=0, base=1).
    z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)

    # Índices en el mapa polar.
    n_rings, n_angles = pm.shape
    # El mapa polar tiene n_rings anillos (ápex=centro, base=borde) y n_angles
    # sectores (0..360°).
    ri = np.clip((z_norm * (n_rings - 1)).astype(np.int32), 0, n_rings - 1)
    ti = np.clip(np.floor(theta).astype(np.int32), 0, n_angles - 1)

    # Asignar el valor del mapa polar a cada vértice.
    vals = pm[ri, ti]

    out = mesh.copy()
    out.point_data["perfusion"] = vals
    return out
