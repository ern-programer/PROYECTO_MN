import numpy as np
import pytest

from core.amyloid_spect import (
    HmrSpectMethod,
    VOISphere,
    compute_hmr_spect,
    compute_spect_ratio,
)


SPACING = (2.0, 2.0, 2.0)


def test_hmr_uses_voxel_means_for_different_voi_sizes():
    volume = np.full((41, 41, 41), 5.0)
    heart = VOISphere(20, 20, 20, 12.0)
    mediastinum = VOISphere(20, 20, 30, 6.0)

    result = compute_hmr_spect(volume, SPACING, heart, mediastinum)

    assert result.hmr == pytest.approx(1.0)
    assert result.heart_counts != pytest.approx(result.mediastinum_counts)


def test_hmr_slice_rejects_voi_that_does_not_intersect_slice():
    volume = np.full((41, 41, 41), 5.0)
    heart = VOISphere(20, 20, 20, 12.0)
    mediastinum = VOISphere(35, 20, 20, 4.0)

    with pytest.raises(ValueError, match="mismo nivel axial"):
        compute_hmr_spect(
            volume,
            SPACING,
            heart,
            mediastinum,
            method=HmrSpectMethod.SLICE_CENTRAL,
        )


def test_svd_uses_voxel_means_for_different_voi_sizes():
    volume = np.full((41, 41, 41), 5.0)
    heart = VOISphere(20, 20, 20, 12.0)
    vertebra = VOISphere(20, 10, 20, 5.0)
    aorta = VOISphere(20, 30, 20, 4.0)

    result = compute_spect_ratio(volume, SPACING, heart, vertebra, aorta)

    assert result.s_vd == pytest.approx(1.0)
    assert result.s_v == pytest.approx(1.0)
    assert result.s_d == pytest.approx(1.0)
    assert result.v_d == pytest.approx(1.0)


def test_svd_rejects_reference_voi_without_signal():
    volume = np.full((41, 41, 41), 5.0)
    heart = VOISphere(20, 20, 20, 12.0)
    vertebra = VOISphere(20, 10, 20, 5.0)
    aorta = VOISphere(20, 30, 20, 4.0)
    volume[vertebra.mask_3d(volume.shape, SPACING)] = 0.0

    with pytest.raises(ValueError, match=r"V \("):
        compute_spect_ratio(volume, SPACING, heart, vertebra, aorta)