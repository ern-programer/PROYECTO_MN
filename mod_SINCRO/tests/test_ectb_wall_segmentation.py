"""Tests de la máscara de miocardio derivada de los contornos irregulares ECTb."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ectb_lv import wall_segmentation_from_ectb


def _make_result(n_angles: int = 64, *, irregular: bool) -> SimpleNamespace:
    theta = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    endo = np.full(n_angles, 4.0)
    epi = np.full(n_angles, 8.0)
    if irregular:
        # Pared más gruesa de un lado: es justo lo que el anillo circular no puede.
        epi = epi + 2.0 * np.cos(theta)
    return SimpleNamespace(
        available=True,
        valid_slices=(1,),
        n_slices_total=3,
        endo_radii_mm=np.stack([endo, endo])[:, None, :],
        epi_radii_mm=np.stack([epi, epi])[:, None, :],
    )


def _make_seg() -> SimpleNamespace:
    return SimpleNamespace(
        mask=np.zeros((3, 32, 32), dtype=bool),
        center_per_slice=np.array([[16.0, 16.0], [16.0, 16.0], [16.0, 16.0]]),
    )


def test_wall_segmentation_fills_only_valid_slices():
    seg = wall_segmentation_from_ectb(_make_result(irregular=False), _make_seg(), (1.0, 1.0))
    assert seg is not None
    assert seg.method == "ectb_wall"
    assert seg.mask[1].any()
    assert not seg.mask[0].any()
    assert not seg.mask[2].any()
    assert seg.n_voxels == int(seg.mask.sum())


def test_wall_segmentation_respects_radii():
    seg = wall_segmentation_from_ectb(_make_result(irregular=False), _make_seg(), (1.0, 1.0))
    assert seg is not None
    ys, xs = np.nonzero(seg.mask[1])
    dist = np.sqrt((ys - 16.0) ** 2 + (xs - 16.0) ** 2)
    assert dist.min() >= 3.5
    assert dist.max() <= 8.5


def test_wall_segmentation_is_not_circular_when_contours_are_irregular():
    seg = wall_segmentation_from_ectb(_make_result(irregular=True), _make_seg(), (1.0, 1.0))
    assert seg is not None
    ys, xs = np.nonzero(seg.mask[1])
    right = np.sqrt((ys[xs > 16] - 16.0) ** 2 + (xs[xs > 16] - 16.0) ** 2).max()
    left = np.sqrt((ys[xs < 16] - 16.0) ** 2 + (xs[xs < 16] - 16.0) ** 2).max()
    # El lóbulo con epicardio más lejano tiene que llegar más lejos del centro.
    assert right > left + 1.0


def test_wall_segmentation_returns_none_when_unavailable():
    result = _make_result(irregular=False)
    result.available = False
    assert wall_segmentation_from_ectb(result, _make_seg(), (1.0, 1.0)) is None
