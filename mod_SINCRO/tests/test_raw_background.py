"""Tests de la sustracción de fondo sobre imagen cruda (core.raw_background).

Verifica: rasterización de polígono, medición del nivel de fondo, resta
constante con clip a 0 y aviso de sobre-sustracción, y resta localizada que solo
toca la región del corazón.
"""
from __future__ import annotations

import numpy as np

from core.raw_background import (
    measure_background_level,
    polygon_mask,
    subtract_constant,
    subtract_localized,
)


def _square_polygon(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_polygon_mask_square():
    mask = polygon_mask((10, 10), _square_polygon(2, 2, 6, 6))
    assert mask.shape == (10, 10)
    # El interior del cuadrado está marcado; las esquinas fuera no.
    assert mask[3, 3]
    assert mask[4, 4]
    assert not mask[0, 0]
    assert not mask[9, 9]
    # Área razonable (par-impar sobre centros de píxel).
    assert 10 <= int(mask.sum()) <= 20


def test_polygon_mask_degenerate_returns_empty():
    assert not polygon_mask((8, 8), [(1, 1), (2, 2)]).any()
    assert not polygon_mask((8, 8), []).any()


def test_measure_background_level_median_and_mean():
    img = np.zeros((10, 10), dtype=np.float64)
    img[2:6, 2:6] = 40.0
    mask = polygon_mask((10, 10), _square_polygon(2, 2, 6, 6))
    assert measure_background_level(img, mask, stat="median") == 40.0
    assert measure_background_level(img, mask, stat="mean") == 40.0
    # Máscara vacía -> 0.
    assert measure_background_level(img, np.zeros((10, 10), bool)) == 0.0


def test_subtract_constant_clips_at_zero():
    img = np.array([[10.0, 30.0], [5.0, 100.0]])
    res = subtract_constant(img, 20.0)
    assert res.method == "constant"
    assert res.level == 20.0
    # 10-20 y 5-20 -> 0 (clip); 30-20=10; 100-20=80.
    np.testing.assert_allclose(res.image, [[0.0, 10.0], [0.0, 80.0]])
    assert res.clipped_fraction == 0.5
    # No modifica la entrada.
    assert img[0, 0] == 10.0


def test_subtract_constant_negative_level_is_clamped():
    img = np.array([[10.0, 20.0]])
    res = subtract_constant(img, -5.0)
    np.testing.assert_allclose(res.image, img)
    assert res.level == 0.0


def test_subtract_constant_oversubtraction_note():
    img = np.full((10, 10), 5.0)
    res = subtract_constant(img, 100.0)
    assert res.clipped_fraction == 1.0
    assert any("sobre-sustracción" in n.lower() for n in res.notes)


def test_subtract_localized_only_touches_heart():
    img = np.full((10, 10), 50.0)
    heart = np.zeros((10, 10), dtype=bool)
    heart[4:6, 4:6] = True
    res = subtract_localized(img, 20.0, heart, feather_px=0.0)
    # Dentro del corazón: 50-20=30; fuera: intacto 50.
    assert res.image[4, 4] == 30.0
    assert res.image[0, 0] == 50.0
    assert res.method == "localized"


def test_subtract_localized_broadcasts_over_stack():
    stack = np.full((3, 8, 8), 60.0)  # (A, H, W)
    heart = np.zeros((8, 8), dtype=bool)
    heart[3:5, 3:5] = True
    res = subtract_localized(stack, 25.0, heart, feather_px=0.0)
    assert res.image.shape == (3, 8, 8)
    assert np.all(res.image[:, 3, 3] == 35.0)
    assert np.all(res.image[:, 0, 0] == 60.0)


def test_subtract_constant_over_stack():
    stack = np.stack([np.full((6, 6), 30.0), np.full((6, 6), 10.0)])
    res = subtract_constant(stack, 15.0)
    assert res.image[0, 0, 0] == 15.0  # 30-15
    assert res.image[1, 0, 0] == 0.0   # 10-15 -> clip 0
