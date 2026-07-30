"""Tests para core.perfusion_texture (GLCM de perfusión + fase por segmento)."""
from __future__ import annotations

import numpy as np
import pytest

from core.perfusion_texture import (
    combine_perfusion_phase,
    glcm_features,
    perfusion_texture_by_segment,
)


def _textured_image(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((h, w)) * 50.0
    # estructura de bloques para que la GLCM tenga textura no trivial
    base[8:24, 8:24] += 150.0
    base[12:16, 12:20] += 80.0
    return base


def test_glcm_features_basicas():
    img = _textured_image()
    feats = glcm_features(img, levels=16)
    assert feats["available"] is True
    for k in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "glcm_entropy"):
        assert k in feats
        assert np.isfinite(feats[k])
    # homogeneidad y energía acotadas en [0,1]
    assert 0.0 <= feats["homogeneity"] <= 1.0
    assert 0.0 <= feats["energy"] <= 1.0


def test_glcm_imagen_uniforme_max_homogeneidad():
    uniforme = np.full((24, 24), 100.0)
    feats = glcm_features(uniforme, levels=16)
    # Imagen constante → tras cuantizar todo es un nivel → GLCM vacía/uniforme
    assert feats["available"] is False or feats["homogeneity"] >= 0.99


def test_glcm_contraste_mayor_en_imagen_ruidosa():
    suave = np.tile(np.linspace(0, 100, 32), (32, 1))
    ruidosa = _textured_image(seed=1)
    f_suave = glcm_features(suave, levels=16)
    f_ruido = glcm_features(ruidosa, levels=16)
    assert f_ruido["contrast"] > f_suave["contrast"]


def test_textura_por_segmento():
    # dos segmentos con texturas distintas
    perfusion = np.zeros((2, 32, 32))
    seg_map = np.zeros((2, 32, 32), dtype=np.int32)
    perfusion[0] = _textured_image(seed=2)
    perfusion[1] = np.tile(np.linspace(0, 100, 32), (32, 1))
    seg_map[0, 8:24, 8:24] = 1
    seg_map[1, 8:24, 8:24] = 5
    out = perfusion_texture_by_segment(perfusion, seg_map, levels=16, min_pixels=12)
    assert out[1]["available"] is True
    assert out[5]["available"] is True
    # segmento inexistente
    assert out[10]["available"] is False


def test_textura_segmento_chico_descarta():
    perfusion = _textured_image()[None, :, :]
    seg_map = np.zeros((1, 32, 32), dtype=np.int32)
    seg_map[0, 0:2, 0:2] = 3  # 4 píxeles < min_pixels
    out = perfusion_texture_by_segment(perfusion, seg_map, min_pixels=12)
    assert out[3]["available"] is False
    assert out[3]["n_pixels"] == 4


def test_shapes_incompatibles_lanza():
    with pytest.raises(ValueError):
        perfusion_texture_by_segment(np.zeros((32, 32)), np.zeros((2, 32, 32), dtype=np.int32))


def test_combine_perfusion_phase_17_filas():
    perfusion = _textured_image()[None, :, :]
    seg_map = np.zeros((1, 32, 32), dtype=np.int32)
    seg_map[0, 8:24, 8:24] = 1
    tex = perfusion_texture_by_segment(perfusion, seg_map, min_pixels=12)
    phase = {1: 45.0, 2: 90.0}
    rows = combine_perfusion_phase(tex, phase)
    assert len(rows) == 17
    assert rows[0]["segment"] == 1
    assert rows[0]["phase_deg"] == 45.0
    assert np.isfinite(rows[0]["contrast"])
    # segmento sin textura → NaN en features pero fila presente
    assert np.isnan(rows[1]["contrast"])
