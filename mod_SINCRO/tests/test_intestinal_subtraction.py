"""Tests de la sustracción de fondo intestinal (core.intestinal_subtraction).

La hipótesis central a verificar es la que justifica todo el módulo: como el
intestino no late, aporta una componente DC. Restarla **no mueve la fase** del
primer armónico pero **sí mejora la amplitud relativa**, que es lo que mira el
filtro de amplitud del análisis de fase. Atenuar multiplicando, en cambio, deja
la amplitud relativa exactamente igual.
"""
from __future__ import annotations

import numpy as np

from core.intestinal_subtraction import (
    MIN_REFERENCE_PIXELS,
    apply_intestinal_subtraction,
    estimate_background_map,
    reference_levels,
    relative_first_harmonic_amplitude,
    subtract_background_from_slice,
)


def _box(shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _phase_of(gate_stack: np.ndarray, mask: np.ndarray) -> float:
    series = gate_stack[:, mask].mean(axis=1)
    return float(np.angle(np.fft.rfft(series)[1]))


def test_reference_levels_usa_mediana_y_descarta_referencias_chicas():
    img = np.zeros((20, 20), dtype=np.float64)
    big = _box(img.shape, 2, 6, 2, 6)  # 16 px
    img[big] = 100.0
    img[2, 2] = 5000.0  # outlier: la media se dispararía, la mediana no
    tiny = _box(img.shape, 15, 16, 15, 16)  # 1 px

    levels = reference_levels(img, [big, tiny])
    assert levels[0] == 100.0
    assert np.isnan(levels[1])
    assert MIN_REFERENCE_PIXELS > 1


def test_estimate_background_sin_referencias_no_resta_nada():
    img = np.full((16, 16), 50.0)
    bg, info = estimate_background_map(img, [])
    assert not info["applicable"]
    assert np.all(bg == 0.0)


def test_estimate_background_una_referencia_da_nivel_constante():
    img = np.zeros((20, 20), dtype=np.float64)
    ref = _box(img.shape, 2, 8, 2, 8)
    img[ref] = 80.0
    bg, info = estimate_background_map(img, [ref])
    assert info["applicable"]
    assert info["n_references"] == 1
    assert np.allclose(bg, 80.0)


def test_estimate_background_interpola_entre_entrada_y_salida():
    """Dos referencias con niveles distintos: en el medio queda el promedio."""
    img = np.zeros((21, 41), dtype=np.float64)
    entrada = _box(img.shape, 8, 13, 1, 6)
    salida = _box(img.shape, 8, 13, 35, 40)
    img[entrada] = 100.0
    img[salida] = 200.0

    bg, info = estimate_background_map(img, [entrada, salida])
    assert info["n_references"] == 2

    # Sobre cada referencia domina su propio nivel (no es exacto: el epsilon del
    # peso IDW deja una contribución mínima de la otra referencia).
    assert abs(bg[10, 3] - 100.0) < 0.05
    assert abs(bg[10, 37] - 200.0) < 0.05
    # ...y a mitad de camino el valor queda entre ambos, cerca del promedio.
    medio = bg[10, 20]
    assert 100.0 < medio < 200.0
    assert abs(medio - 150.0) < 15.0


def test_media_simple_aplica_un_unico_nivel_constante():
    """Modo 'mean': el fondo es un solo número, igual en toda la imagen."""
    img = np.zeros((21, 41), dtype=np.float64)
    entrada = _box(img.shape, 8, 13, 1, 6)
    salida = _box(img.shape, 8, 13, 35, 40)
    img[entrada] = 100.0
    img[salida] = 200.0

    bg, info = estimate_background_map(img, [entrada, salida], method="mean")
    assert info["method"] == "mean"
    assert info["n_references"] == 2
    assert np.allclose(bg, 150.0)
    assert info["background_level"] == 150.0


def test_media_simple_con_tres_referencias_pesa_igual_a_cada_roi():
    """Cada ROI cuenta lo mismo aunque tenga distinto tamaño."""
    img = np.zeros((30, 30), dtype=np.float64)
    r1 = _box(img.shape, 1, 4, 1, 4)  # 9 px
    r2 = _box(img.shape, 10, 20, 10, 20)  # 100 px, mucho más grande
    r3 = _box(img.shape, 24, 28, 24, 28)  # 16 px
    img[r1] = 60.0
    img[r2] = 90.0
    img[r3] = 120.0

    bg, info = estimate_background_map(img, [r1, r2, r3], method="mean")
    assert info["n_references"] == 3
    # Media de los niveles (60+90+120)/3 = 90, no una media ponderada por píxeles
    # (que el ROI grande arrastraría hacia 90 por casualidad no: sería 88.6).
    assert np.allclose(bg, 90.0)
    assert sorted(info["levels"]) == [60.0, 90.0, 120.0]


def test_media_simple_ignora_referencias_demasiado_chicas():
    img = np.zeros((20, 20), dtype=np.float64)
    ok1 = _box(img.shape, 2, 6, 2, 6)
    ok2 = _box(img.shape, 12, 16, 12, 16)
    tiny = _box(img.shape, 0, 1, 0, 1)
    img[ok1] = 40.0
    img[ok2] = 80.0
    img[tiny] = 9000.0

    bg, info = estimate_background_map(img, [ok1, tiny, ok2], method="mean")
    assert info["n_references"] == 2
    assert info["n_references_discarded"] == 1
    assert np.allclose(bg, 60.0)


def test_apply_propaga_el_metodo_de_fondo():
    cube = np.full((8, 1, 24, 24), 20.0)
    cube[:, 0, 4:8, 4:8] = 100.0
    cube[:, 0, 16:20, 16:20] = 300.0
    refs = [_box((24, 24), 4, 8, 4, 8), _box((24, 24), 16, 20, 16, 20)]
    weight = np.zeros((24, 24), dtype=np.float64)
    weight[10:14, 10:14] = 1.0

    _, info = apply_intestinal_subtraction(cube, {0: weight}, {0: refs}, method="mean")
    assert info["method"] == "mean"
    assert info["per_slice"][0]["background_level"] == 200.0


def test_subtraccion_recorta_en_cero_y_avisa_de_sobre_sustraccion():
    gates = np.full((8, 12, 12), 30.0)
    background = np.full((12, 12), 500.0)  # muy por encima de las cuentas reales
    weight = np.zeros((12, 12), dtype=np.float64)
    weight[4:8, 4:8] = 1.0

    corrected, info = subtract_background_from_slice(gates, background, weight)
    assert np.all(corrected >= 0.0)
    assert np.all(corrected[:, 4:8, 4:8] == 0.0)
    assert corrected[0, 0, 0] == 30.0  # fuera del ROI no se toca
    assert info["oversubtracted"]
    assert info["clipped_fraction"] > 0.9


def test_restar_dc_intestinal_mejora_amplitud_sin_mover_la_fase():
    """El test que justifica el módulo entero.

    Miocardio con amplitud relativa 0.30 contaminado por un intestino constante
    que la baja a 0.10. Restar el DC la devuelve a 0.30 y deja la fase intacta.
    """
    n_gates = 16
    shape = (12, 12)
    region = _box(shape, 4, 8, 4, 8)
    t = np.arange(n_gates) * 2.0 * np.pi / n_gates
    m0, amp, fase = 100.0, 30.0, 0.7
    intestino = 200.0

    miocardio = np.zeros((n_gates, *shape), dtype=np.float64)
    for g in range(n_gates):
        miocardio[g][region] = m0 + amp * np.cos(t[g] + fase)

    contaminado = miocardio.copy()
    contaminado[:, region] += intestino

    rel_limpio = relative_first_harmonic_amplitude(miocardio, region)
    rel_contaminado = relative_first_harmonic_amplitude(contaminado, region)
    assert abs(rel_limpio - 0.30) < 0.01
    assert abs(rel_contaminado - 0.10) < 0.01

    # Atenuar multiplicando NO cambia la amplitud relativa: el factor se cancela.
    atenuado = contaminado * 0.4
    assert abs(relative_first_harmonic_amplitude(atenuado, region) - rel_contaminado) < 1e-9

    # Restar el DC sí la recupera.
    background = np.zeros(shape, dtype=np.float64)
    background[region] = intestino
    corregido, info = subtract_background_from_slice(
        contaminado, background, region.astype(np.float64)
    )
    assert abs(info["rel_amp_after"] - rel_limpio) < 0.01
    assert info["rel_amp_after"] > info["rel_amp_before"]
    assert not info["oversubtracted"]

    # Y la fase del primer armónico queda donde estaba.
    assert abs(_phase_of(corregido, region) - _phase_of(contaminado, region)) < 1e-6
    assert abs(_phase_of(corregido, region) - _phase_of(miocardio, region)) < 1e-6


def test_apply_no_modifica_el_cubo_original_ni_los_cortes_sin_roi():
    cube = np.full((8, 4, 16, 16), 120.0)
    original = cube.copy()

    weight = np.zeros((16, 16), dtype=np.float64)
    weight[6:10, 6:10] = 1.0
    ref = _box((16, 16), 1, 6, 1, 6)

    out, info = apply_intestinal_subtraction(cube, {2: weight}, {2: [ref]})

    assert np.array_equal(cube, original)  # no se toca la entrada
    assert info["applied"]
    assert info["slices_corrected"] == [2]
    for s in (0, 1, 3):
        assert np.array_equal(out[:, s], original[:, s])
    assert np.all(out[:, 2, 6:10, 6:10] < original[:, 2, 6:10, 6:10])


def test_apply_sin_referencia_no_corrige_y_lo_reporta():
    cube = np.full((8, 2, 12, 12), 90.0)
    weight = np.zeros((12, 12), dtype=np.float64)
    weight[4:8, 4:8] = 1.0

    out, info = apply_intestinal_subtraction(cube, {1: weight}, {})

    assert not info["applied"]
    assert info["slices_without_reference"] == [1]
    assert np.array_equal(out, cube)
    assert "sin ROI de referencia" in info["message"]
