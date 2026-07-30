"""Tests del centrado en cavidad (core.lv_center).

La hipótesis a verificar es la que originó el cambio: con captación desigual, el
centroide de la máscara de miocardio se corre hacia el sector caliente, mientras
que el centro de la cavidad se queda donde tiene que estar.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass

from core.lv_center import (
    ANGULAR_COVERAGE_MIN,
    _wall_angular_coverage,
    cavity_center_from_image,
    cavity_center_from_mask,
    refine_center_to_cavity,
)
from core.segmentation import segment_myocardium


def _annulus(size: int, cy: float, cx: float, r_in: float, r_out: float) -> np.ndarray:
    ys, xs = np.ogrid[:size, :size]
    d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    return (d >= r_in) & (d <= r_out)


def _arc(size: int, cy: float, cx: float, r_in: float, r_out: float, half_deg: float) -> np.ndarray:
    """Sector de anillo abierto (herradura) centrado hacia arriba (ang = -90°).

    ``half_deg`` es el semiángulo: 90 => media pared (180°), 45 => arco de 90°.
    """
    ys, xs = np.mgrid[:size, :size]
    d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    ang = np.arctan2(ys - cy, xs - cx)
    delta = np.angle(np.exp(1j * (ang - (-np.pi / 2))))
    return (d >= r_in) & (d <= r_out) & (np.abs(delta) < np.radians(half_deg))


def test_cavity_center_from_mask_recupera_el_hueco():
    mask = _annulus(40, 20.0, 20.0, 5.0, 9.0)
    center = cavity_center_from_mask(mask)
    assert center is not None
    cy, cx = center
    assert abs(cy - 20.0) < 0.5
    assert abs(cx - 20.0) < 0.5


def test_cavity_center_from_mask_sin_hueco_devuelve_none():
    # Disco lleno: no hay cavidad que encontrar.
    ys, xs = np.ogrid[:30, :30]
    disk = np.sqrt((ys - 15.0) ** 2 + (xs - 15.0) ** 2) <= 8.0
    assert cavity_center_from_mask(disk) is None


def test_centroide_de_masa_se_sesga_y_la_cavidad_no():
    """Núcleo del problema: pared lateral caliente arrastra el centroide."""
    size = 40
    ring = _annulus(size, 20.0, 20.0, 5.0, 9.0)
    img = np.zeros((size, size), dtype=np.float64)
    img[ring] = 100.0
    # Sector lateral (columnas altas) con el doble de captación.
    ys, xs = np.mgrid[:size, :size]
    img[ring & (xs > 20)] = 220.0

    # El centroide ponderado por intensidad se corre hacia el sector caliente.
    biased_cy, biased_cx = center_of_mass(img)
    assert biased_cx - 20.0 > 1.5

    # El refinamiento por cavidad lo devuelve al centro real.
    fixed_cy, fixed_cx = refine_center_to_cavity(
        biased_cy, biased_cx, 9.0, mask=ring, img=img
    )
    assert abs(fixed_cx - 20.0) < abs(biased_cx - 20.0)
    assert abs(fixed_cx - 20.0) < 0.5
    assert abs(fixed_cy - 20.0) < 0.5


def test_cavity_center_from_image_sin_mascara():
    """Camino de baja resolución: sin hueco de píxeles, se usa hipocaptación."""
    size = 24
    ring = _annulus(size, 12.0, 12.0, 2.5, 5.0)
    img = np.zeros((size, size), dtype=np.float64)
    img[ring] = 150.0
    center = cavity_center_from_image(img, 14.0, 14.0, 5.0, low_res=True)
    assert center is not None
    cy, cx = center
    assert abs(cy - 12.0) < 1.5
    assert abs(cx - 12.0) < 1.5


def test_refine_center_rechaza_saltos_implausibles():
    """Un candidato fuera del anillo no debe reemplazar al centro de entrada."""
    mask = _annulus(40, 20.0, 20.0, 5.0, 9.0)
    img = np.zeros((40, 40), dtype=np.float64)
    img[mask] = 100.0
    cy, cx = refine_center_to_cavity(
        20.0, 20.0, 9.0, mask=mask, img=img, max_shift_px=0.0001
    )
    assert (cy, cx) == (20.0, 20.0)


def test_refine_center_nunca_devuelve_nan_ni_rompe():
    vacio = np.zeros((20, 20), dtype=bool)
    cy, cx = refine_center_to_cavity(10.0, 10.0, 5.0, mask=vacio, img=None)
    assert (cy, cx) == (10.0, 10.0)


def test_segment_myocardium_acepta_el_flag_y_no_degrada():
    """El flag no debe romper la segmentación ni perder cortes."""
    n_gates, n_slices, size = 8, 6, 32
    cube = np.zeros((n_gates, n_slices, size, size), dtype=np.float64)
    ring = _annulus(size, 16.0, 16.0, 4.0, 8.0)
    for g in range(n_gates):
        for s in range(n_slices):
            cube[g, s][ring] = 100.0

    base = segment_myocardium(cube, method="auto")
    refined = segment_myocardium(cube, method="auto", refine_cavity_center=True)

    assert refined.mask.shape == base.mask.shape
    assert refined.n_voxels > 0
    # Con un anillo simétrico ambos deben coincidir dentro de un píxel.
    valid = np.isfinite(base.center_per_slice[:, 0]) & np.isfinite(refined.center_per_slice[:, 0])
    assert valid.any()
    delta = np.abs(refined.center_per_slice[valid] - base.center_per_slice[valid])
    assert float(np.max(delta)) < 1.0


def test_el_flag_mueve_el_centro_cuando_el_anillo_queda_abierto():
    """Regresión: el flag no producía ningún cambio observable.

    El caso donde importa es el que se da en la práctica: la pared fría cae por
    debajo del umbral y el anillo queda ABIERTO. Sin anillo cerrado no hay hueco
    que rellenar, así que el pipeline caía al centroide del músculo, que con solo
    media pared visible se para directamente encima del sector caliente.
    """
    n_gates, n_slices, size = 8, 6, 32
    cube = np.zeros((n_gates, n_slices, size, size), dtype=np.float64)
    ring = _annulus(size, 16.0, 16.0, 4.0, 8.0)
    _, xs = np.mgrid[:size, :size]
    hot = ring & (xs > 16)
    cold = ring & (xs <= 16)
    for g in range(n_gates):
        for s in range(n_slices):
            # 70 queda por debajo del umbral (0.35 * ~260) y desaparece.
            cube[g, s][cold] = 70.0
            cube[g, s][hot] = 260.0

    base = segment_myocardium(cube, method="auto")
    refined = segment_myocardium(cube, method="auto", refine_cavity_center=True)

    valid = np.isfinite(base.center_per_slice[:, 0]) & np.isfinite(refined.center_per_slice[:, 0])
    assert valid.any()
    delta = np.abs(refined.center_per_slice[valid] - base.center_per_slice[valid])
    assert float(np.max(delta)) > 0.05, "el centrado en cavidad tiene que mover algo"

    # Y tiene que acercarlo al centro real, no solo moverlo.
    real = np.array([16.0, 16.0])
    d_base = np.mean(np.hypot(*(base.center_per_slice[valid] - real).T))
    d_ref = np.mean(np.hypot(*(refined.center_per_slice[valid] - real).T))
    assert d_ref < d_base

    # Y el desplazamiento tiene que quedar informado, para poder verificarlo.
    assert refined.center_shift_px is not None
    shifts = np.asarray(refined.center_shift_px, dtype=np.float64)
    assert np.isfinite(shifts).any()
    assert base.center_shift_px is None or not np.isfinite(
        np.asarray(base.center_shift_px, dtype=np.float64)
    ).any()


def test_cobertura_angular_distingue_cavidad_de_arco():
    """El discriminador del guard: cavidad cerrada ~1.0; arco angosto, bajo."""
    size = 48
    ring = _annulus(size, 24.0, 24.0, 7.0, 11.0)
    img_ring = np.full((size, size), 6.0)
    img_ring[ring] = 200.0
    cov_ring = _wall_angular_coverage(img_ring, 24.0, 24.0, 11.0, wall_level=100.0)
    assert cov_ring > 0.9

    arc = _arc(size, 24.0, 24.0, 7.0, 11.0, half_deg=45.0)
    img_arc = np.full((size, size), 6.0)
    img_arc[arc] = 200.0
    # Visto desde el centro real, el arco de 90° cubre poco: cae bajo el umbral.
    cov_arc = _wall_angular_coverage(img_arc, 24.0, 24.0, 11.0, wall_level=100.0)
    assert cov_arc < ANGULAR_COVERAGE_MIN


def test_guard_rechaza_candidato_en_fondo_de_arco_angosto():
    """Regresión del ROI descolgado: un arco angosto (base/ápex) no debe dar cavidad.

    Con solo un sector chico de pared, la mayor masa 'por debajo del nivel de
    pared' es el fondo. Sin el guard, el centro se iría ahí y deformaría el ROI;
    con el guard, ``cavity_center_from_image`` devuelve None y el refinamiento
    conserva el centro de entrada.
    """
    size = 48
    arc = _arc(size, 24.0, 24.0, 7.0, 11.0, half_deg=45.0)
    img = np.full((size, size), 6.0)
    img[arc] = 200.0
    ys, xs = np.where(arc)
    in_cy, in_cx = float(ys.mean()), float(xs.mean())

    assert cavity_center_from_image(img, in_cy, in_cx, 11.0, low_res=True) is None

    out_cy, out_cx = refine_center_to_cavity(in_cy, in_cx, 11.0, img=img, low_res=True)
    assert (out_cy, out_cx) == (in_cy, in_cx)


def test_guard_acepta_herradura_amplia():
    """Contraparte: una herradura amplia (~270°) SÍ es una cavidad y se acepta."""
    size = 48
    arc = _arc(size, 24.0, 24.0, 7.0, 11.0, half_deg=135.0)
    img = np.full((size, size), 6.0)
    img[arc] = 200.0
    ys, xs = np.where(arc)
    in_cy, in_cx = float(ys.mean()), float(xs.mean())

    center = cavity_center_from_image(img, in_cy, in_cx, 11.0, low_res=True)
    assert center is not None
    cy, cx = center
    # El refinamiento acerca el centro al de la cavidad real (24, 24).
    assert abs(cy - 24.0) < abs(in_cy - 24.0)
    assert abs(cx - 24.0) < 1.5
