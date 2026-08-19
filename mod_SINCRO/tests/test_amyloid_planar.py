# -*- coding: utf-8 -*-
"""Tests para core.amyloid_planar (HMR planar para amiloidosis)."""
import numpy as np

from core.amyloid_planar import ROICircle, compute_hmr, PERUGINI_SCORES


def test_hmr_positive():
    """HMR ≥1.5 → clasificación POSITIVO (ATTR)."""
    img = np.zeros((64, 64), dtype=np.float64)
    # Corazón: ROI 10x1000 (área 100)
    for y in range(20, 40):
        for x in range(20, 40):
            if (y - 30) ** 2 + (x - 30) ** 2 <= 100:
                img[y, x] = 1000
    # Mediastino: ROI contralateral 10x500 (área 100)
    for y in range(20, 40):
        for x in range(40, 60):
            if (y - 30) ** 2 + (x - 50) ** 2 <= 100:
                img[y, x] = 600
    roi_h = ROICircle(30, 30, 10)
    roi_m = ROICircle(30, 50, 10)
    result = compute_hmr(img, roi_h, roi_m)
    assert result.hmr >= 1.5
    assert result.classification == "POSITIVO (sugiere ATTR)"


def test_hmr_negative():
    """HMR <1.0 → clasificación NEGATIVO."""
    img = np.zeros((64, 64), dtype=np.float64)
    # Corazón: ROI 10x500 (área 100)
    for y in range(20, 40):
        for x in range(20, 40):
            if (y - 30) ** 2 + (x - 30) ** 2 <= 100:
                img[y, x] = 500
    # Mediastino: ROI contralateral 10x1000 (área 100)
    for y in range(20, 40):
        for x in range(40, 60):
            if (y - 30) ** 2 + (x - 50) ** 2 <= 100:
                img[y, x] = 1000
    roi_h = ROICircle(30, 30, 10)
    roi_m = ROICircle(30, 50, 10)
    result = compute_hmr(img, roi_h, roi_m)
    assert result.hmr < 1.0
    assert result.classification == "NEGATIVO"


def test_hmr_equivocal():
    """1.0 ≤ HMR < 1.5 → clasificación EQUÍVOCO."""
    img = np.zeros((64, 64), dtype=np.float64)
    # Corazón: ROI 10x700 (área 100)
    for y in range(20, 40):
        for x in range(20, 40):
            if (y - 30) ** 2 + (x - 30) ** 2 <= 100:
                img[y, x] = 700
    # Mediastino: ROI contralateral 10x500 (área 100)
    for y in range(20, 40):
        for x in range(40, 60):
            if (y - 30) ** 2 + (x - 50) ** 2 <= 100:
                img[y, x] = 500
    roi_h = ROICircle(30, 30, 10)
    roi_m = ROICircle(30, 50, 10)
    result = compute_hmr(img, roi_h, roi_m)
    assert 1.0 <= result.hmr < 1.5
    assert result.classification == "EQUÍVOCO (complementar con SPECT o repeat a 3h)"


def test_perugini_scores():
    """Perugini score entries existen para todos los grados."""
    for score in (0, 1, 2, 3):
        assert score in PERUGINI_SCORES
        assert "captación" in PERUGINI_SCORES[score].lower() or "hueso" in PERUGINI_SCORES[score].lower()


def test_roi_mask():
    """La máscara del ROI cubre el área esperada."""
    img_shape = (64, 64)
    roi = ROICircle(30, 30, 10)
    mask = roi.mask(img_shape)
    area = np.count_nonzero(mask)
    expected = np.pi * 10 ** 2
    assert abs(area - expected) / expected < 0.05
