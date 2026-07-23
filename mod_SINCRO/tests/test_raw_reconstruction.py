"""Tests del pipeline crudo gated -> motion correction -> reconstruccion."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.raw_projections import make_synthetic_raw_motion_projections, motion_correct_projections
from core.raw_reconstruction import (
    ProjectionFilterConfig,
    RawReconConfig,
    filter_projections,
    reconstruct_raw_gated_pipeline,
)


class RawReconstructionTests(unittest.TestCase):
    def test_raw_recon_pipeline_uses_separate_filters_and_display_step(self):
        raw = make_synthetic_raw_motion_projections(n_gates=2, n_angles=12, rows=16, cols=16).projections
        angles = np.linspace(0.0, 360.0, raw.shape[1], endpoint=False)
        motion = motion_correct_projections(raw, method="sinusoid", axis="y", angles_deg=angles, max_abs_shift_px=2.0)
        cfg = RawReconConfig(
            ungated_filter=ProjectionFilterConfig("lowpass", cutoff=0.45, order=2),
            gated_filter=ProjectionFilterConfig("wiener", cutoff=0.30, order=4),
            display_slice_step_px=2,
        )

        result = reconstruct_raw_gated_pipeline(raw, angles, motion_result=motion, config=cfg)

        self.assertEqual(result.ungated_volume.shape, (16, 16, 16))
        self.assertEqual(result.gated_volume.shape, (2, 16, 16, 16))
        self.assertEqual(result.phase_cube.shape, (2, 16, 16, 16))
        self.assertEqual(result.display_cube.shape, (2, 8, 16, 16))
        self.assertEqual(result.config.ungated_filter.kind, "lowpass")
        self.assertEqual(result.config.gated_filter.kind, "wiener")
        self.assertTrue(np.isfinite(result.gated_volume).all())
        self.assertTrue(np.allclose(result.shifts_y, motion["applied_shifts_y"]))
        self.assertTrue(any("Filtros separados" in note for note in result.notes))

    def test_projection_filters_are_not_aliases(self):
        raw = make_synthetic_raw_motion_projections(n_gates=1, n_angles=8, rows=8, cols=8).projections[0]
        lowpass = filter_projections(raw, ProjectionFilterConfig("lowpass", cutoff=0.20, order=2))
        wiener = filter_projections(raw, ProjectionFilterConfig("wiener", cutoff=0.60, order=6, noise_power=0.20))

        self.assertEqual(lowpass.shape, raw.shape)
        self.assertEqual(wiener.shape, raw.shape)
        self.assertFalse(np.allclose(lowpass, wiener))

    def test_iterative_methods_reconstruct_small_synthetic(self):
        raw = make_synthetic_raw_motion_projections(n_gates=1, n_angles=8, rows=8, cols=8).projections

        for method in ("mlem", "osem"):
            result = reconstruct_raw_gated_pipeline(
                raw,
                config=RawReconConfig(reconstruction_method=method, iterative_iterations=1, osem_subsets=2),
            )
            self.assertEqual(result.ungated_volume.shape, (8, 8, 8))
            self.assertEqual(result.gated_volume.shape, (1, 8, 8, 8))
            self.assertTrue(np.isfinite(result.gated_volume).all())
            self.assertTrue(any(method.upper() in note for note in result.notes))

    def test_fevi_slice_step_must_stay_one_pixel(self):
        raw = make_synthetic_raw_motion_projections(n_gates=1, n_angles=8, rows=8, cols=8).projections

        with self.assertRaisesRegex(ValueError, "FEVI"):
            reconstruct_raw_gated_pipeline(raw, config=RawReconConfig(fevi_slice_step_px=2))


if __name__ == "__main__":
    unittest.main()
