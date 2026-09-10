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
    heart_axial_bounds_from_spect,
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


def _blob_slice(size: int, amp: float, cy: float | None = None, cx: float | None = None) -> np.ndarray:
    """Corte 2D con una mancha gaussiana caliente (corazón/hígado).

    Por defecto la mancha va centrada; ``cy``/``cx`` permiten desplazarla para
    reproducir la anatomía real (el hígado queda lateral/inferior al corazón).
    """
    ys, xs = np.ogrid[:size, :size]
    c = size / 2.0
    cy = c if cy is None else cy
    cx = c if cx is None else cx
    d2 = (ys - cy) ** 2 + (xs - cx) ** 2
    return amp * np.exp(-d2 / (2.0 * (size * 0.15) ** 2))


def test_axial_bounds_banda_unica():
    """Corazón como banda caliente contigua: los límites la cubren + margen."""
    n, size = 20, 24
    vol = np.zeros((n, size, size), dtype=np.float64)
    # Joroba miocárdica en z=[6,13], fondo tenue en el resto.
    for z in range(n):
        vol[z] += 2.0  # fondo
    for z in range(6, 14):
        vol[z] += _blob_slice(size, amp=100.0)

    bounds = heart_axial_bounds_from_spect(vol, margin=1)
    assert bounds is not None
    lo, hi = bounds
    # Cubre la banda real con el margen aplicado.
    assert lo <= 6 and hi >= 13
    # No se traga todo el volumen.
    assert lo >= 4 and hi <= 15


def test_axial_bounds_descarta_higado_caudal():
    """Corazón (caliente) + hígado (algo más frío) separados por un valle:
    la banda elegida es la del corazón, no la del hígado."""
    n, size = 24, 24
    vol = np.full((n, size, size), 1.5, dtype=np.float64)
    # Corazón z=[4,10] (pico alto).
    for z in range(4, 11):
        vol[z] += _blob_slice(size, amp=120.0)
    # Hígado z=[16,22] (más frío) tras un valle profundo (z=[11,15] casi fondo).
    for z in range(16, 23):
        vol[z] += _blob_slice(size, amp=45.0)

    bounds = heart_axial_bounds_from_spect(vol, margin=1)
    assert bounds is not None
    lo, hi = bounds
    # Debe quedarse con el corazón y NO extenderse al hígado.
    assert lo <= 4 and hi <= 12
    assert hi < 16


def test_axial_bounds_4d_suma_gates():
    """Acepta volumen 4D (g,K,H,W) sumando gates."""
    n, size = 16, 20
    vol4 = np.zeros((8, n, size, size), dtype=np.float64)
    for g in range(8):
        for z in range(5, 11):
            vol4[g, z] += _blob_slice(size, amp=30.0)
        vol4[g] += 1.0
    bounds = heart_axial_bounds_from_spect(vol4, margin=1)
    assert bounds is not None
    lo, hi = bounds
    assert lo <= 5 and hi >= 10


def test_axial_bounds_volumen_vacio_devuelve_none():
    """Sin señal (todo cero) no se puede estimar: None (el llamador no empeora)."""
    assert heart_axial_bounds_from_spect(np.zeros((10, 16, 16))) is None
    # Muy pocos cortes tampoco.
    assert heart_axial_bounds_from_spect(np.ones((2, 16, 16))) is None


def test_axial_bounds_gated_latido_recorta_higado_estatico():
    """Método por latido: el corazón (que late) marca la banda; el hígado caudal
    caliente pero ESTÁTICO no aparece en el perfil de amplitud → no se anexa.

    Es el caso que el perfil de intensidad no resolvía (el hígado estiraba la
    banda hasta casi el final del FOV). Con gating, la banda queda ajustada al
    corazón.
    """
    g, n, size = 8, 24, 20
    vol4 = np.full((g, n, size, size), 2.0, dtype=np.float64)
    # Corazón z=[5,11]: LATE (amplitud sinusoidal entre gates).
    for gi in range(g):
        beat = 1.0 + 0.6 * np.sin(2.0 * np.pi * gi / g)
        for z in range(5, 12):
            vol4[gi, z] += _blob_slice(size, amp=90.0 * beat)
    # Hígado z=[17,23]: caliente pero ESTÁTICO (igual en todos los gates).
    for gi in range(g):
        for z in range(17, 24):
            vol4[gi, z] += _blob_slice(size, amp=110.0)

    bounds = heart_axial_bounds_from_spect(vol4, margin=1)
    assert bounds is not None
    lo, hi = bounds
    # La banda es la del corazón; el hígado (z>=17) queda fuera.
    assert lo <= 5 and hi >= 11
    assert hi < 16


def test_axial_bounds_un_gate_cae_a_intensidad():
    """1 gate (no gatillado): no hay latido → usa el perfil de intensidad."""
    n, size = 20, 20
    vol4 = np.full((1, n, size, size), 1.5, dtype=np.float64)
    for z in range(6, 13):
        vol4[0, z] += _blob_slice(size, amp=80.0)
    bounds = heart_axial_bounds_from_spect(vol4, margin=1)
    assert bounds is not None
    lo, hi = bounds
    assert lo <= 6 and hi >= 12


def test_axial_bounds_gated_ignora_arrastre_global():
    """Caso real GE con movimiento: todo el volumen se TRASLADA entre gates
    (arrastre global) pero solo el corazón se CONTRAE. La traslación de una
    estructura rígida hace latir sus bordes opuestos en ANTIFASE, así que su F1
    complejo se cancela al sumarlo; la contracción cardíaca es coherente (en
    fase) y sobrevive. El perfil por suma coherente |Σ F1| no debe inflarse al
    hígado ni a full-FOV.

    Antes (|F1| absoluto) esto daba z≈[2,63]/64 porque el borde del hígado
    desplazado y el arrastre global 'latían'. Con la suma coherente, el hígado
    (antifase) se cancela y solo el miocardio pulsátil en fase cuenta.
    """
    from scipy.ndimage import shift as _ndi_shift

    g, n, size = 8, 32, 24
    base = np.full((n, size, size), 2.0, dtype=np.float64)
    # Hígado caudal brillante y ESTÁTICO en forma (solo se traslada con el resto).
    # Va DESPLAZADO (lateral/inferior) respecto del corazón, como en la anatomía real.
    liver_cy, liver_cx = size * 0.72, size * 0.70
    for z in range(22, 30):
        base[z] += _blob_slice(size, amp=130.0, cy=liver_cy, cx=liver_cx)
    vol4 = np.empty((g, n, size, size), dtype=np.float64)
    for gi in range(g):
        # Arrastre global: traslada TODO el volumen (movimiento del paciente).
        dy = 4.0 * np.sin(2.0 * np.pi * gi / g)
        dx = 3.0 * np.cos(2.0 * np.pi * gi / g)
        frame = _ndi_shift(base, (0.0, dy, dx), order=1, mode="nearest")
        # Corazón z=[6,13] centrado: además del arrastre, SE CONTRAE (modula amplitud).
        beat = 1.0 + 0.5 * np.sin(2.0 * np.pi * gi / g)
        for z in range(6, 14):
            frame[z] += _blob_slice(size, amp=70.0 * beat)
        vol4[gi] = frame

    bounds = heart_axial_bounds_from_spect(vol4, margin=1)
    assert bounds is not None
    lo, hi = bounds
    # La banda es la del corazón; no llega al hígado ni ocupa todo el FOV.
    assert lo <= 7
    assert hi < 20
