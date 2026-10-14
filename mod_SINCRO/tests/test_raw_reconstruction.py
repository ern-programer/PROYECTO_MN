"""Tests del pipeline crudo gated -> motion correction -> reconstruccion."""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.raw_projections import make_synthetic_raw_motion_projections, motion_correct_projections
from core.raw_reconstruction import (
    ProjectionFilterConfig,
    RawReconConfig,
    _detect_rotation_ccw,
    _apply_post_butterworth,
    _butterworth_3d,
    filter_projections,
    reconstruct_gated_projection_volume,
    reconstruct_raw_gated_pipeline,
)


class RawReconstructionTests(unittest.TestCase):
    def test_fine_iterative_operator_is_adjoint_and_preserves_physical_density(self):
        from core.raw_reconstruction import _fine_system_matrix
        rng = np.random.default_rng(123)
        theta = np.array([0.0, 37.0, 90.0])
        operator = _fine_system_matrix(theta, 16, 32)
        image = rng.random(32 * 32)
        detector = rng.random(16 * len(theta))
        self.assertAlmostEqual(float(np.dot(operator @ image, detector)),
                               float(np.dot(image, operator.T @ detector)), places=10)
        for size in (16, 32):
            operator = _fine_system_matrix(np.array([0.0, 90.0]), 16, size)
            profile = (operator @ np.ones(size * size)).reshape(2, 16)
            np.testing.assert_allclose(profile[:, 2:-2], 16.0, atol=1e-10)

    def test_fine_pipeline_is_isotropic_keeps_projections_and_reports_voxel_scale(self):
        raw = make_synthetic_raw_motion_projections(n_gates=2, n_angles=12, rows=8, cols=16).projections
        original = raw.copy()
        result = reconstruct_raw_gated_pipeline(raw, config=RawReconConfig(reconstruction_matrix=32))
        self.assertEqual(result.gated_volume.shape, (2, 16, 32, 32))
        self.assertEqual(result.ungated_volume.shape, (16, 32, 32))
        self.assertEqual(result.voxel_scale, 0.5)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(result.original_projections, original)

    def test_fine_matrix_supports_iterative_methods(self):
        raw = make_synthetic_raw_motion_projections(n_gates=1, n_angles=8, rows=8, cols=8).projections
        for method in ("osem", "mlem"):
            result = reconstruct_raw_gated_pipeline(raw, config=RawReconConfig(
                reconstruction_method=method, reconstruction_matrix=16,
                iterative_iterations=2, osem_subsets=2, recon_slice_range=(2, 5)))
            self.assertEqual(result.gated_volume.shape, (1, 16, 16, 16))
            self.assertEqual(result.voxel_scale, 0.5)
            self.assertTrue(np.isfinite(result.gated_volume).all())
            self.assertGreaterEqual(float(result.gated_volume.min()), 0.0)
            np.testing.assert_array_equal(result.gated_volume[:, :4], 0.0)
            np.testing.assert_array_equal(result.gated_volume[:, 12:], 0.0)

    def test_fine_iterative_update_reduces_projection_error(self):
        from core.raw_reconstruction import _fine_system_matrix, _iterative_reconstruct_slice
        theta = np.linspace(0, 180, 16, endpoint=False)
        operator = _fine_system_matrix(theta, 16, 32)
        truth = np.zeros((32, 32))
        truth[10:22, 12:20] = 3.0
        measured = (operator @ truth.ravel()).reshape(len(theta), 16).T
        errors = []
        for iterations in (1, 6):
            recon = _iterative_reconstruct_slice(measured, theta, output_size=32,
                                                 iterations=iterations, subsets=1)
            errors.append(np.linalg.norm(operator @ recon.ravel() - measured.T.ravel()))
        self.assertLess(errors[1], errors[0] * 0.6)

    def test_fine_rr_operator_is_adjoint_and_keeps_psf_in_millimeters(self):
        from core.raw_reconstruction import _fine_system_matrix
        from core.resolution_recovery import PsfModel
        psf = PsfModel(3.8, 1.4, 30.0, 200.0, 6.8)
        theta = np.array([0.0, 37.0, 90.0])
        native = _fine_system_matrix(theta, 16, 16, psf=psf)
        fine = _fine_system_matrix(theta, 16, 32, psf=psf)
        np.testing.assert_allclose(native[:, 8 * 16 + 10].toarray(),
                                   fine[:, 16 * 32 + 20].toarray() / 0.25, atol=1e-12)
        rng = np.random.default_rng(17)
        image, measured = rng.random(32 * 32), rng.random(16 * len(theta))
        self.assertAlmostEqual(float(np.dot(fine @ image, measured)),
                               float(np.dot(image, fine.T @ measured)), places=10)
        raw = make_synthetic_raw_motion_projections(n_gates=3, n_angles=8, rows=8, cols=8).projections
        result = reconstruct_raw_gated_pipeline(raw, config=RawReconConfig(
            reconstruction_method="osem", reconstruction_matrix=16, iterative_iterations=1,
            resolution_recovery=True, psf_model=psf, nitida3_enabled=True,
            nitida3_iterations=1, recon_slice_range=(2, 5)))
        self.assertEqual(result.gated_volume.shape, (3, 16, 16, 16))
        self.assertTrue(np.isfinite(result.gated_volume).all())
        self.assertGreaterEqual(float(result.gated_volume.min()), 0.0)

    def test_fine_nitida4d_keeps_geometry_feta_and_gate_reference_sums(self):
        from core.nitida4d import nitida4d_osem_gated
        raw = make_synthetic_raw_motion_projections(n_gates=4, n_angles=8, rows=8, cols=8).projections
        kwargs = dict(output_size=16, axial_resample=True, iterations=1, subsets=2,
                      slice_range=(2, 5))
        reference = reconstruct_gated_projection_volume(raw, method="osem", **kwargs)
        result = nitida4d_osem_gated(raw, np.linspace(0, 360, 8, endpoint=False),
                                    ref_volume=reference, **kwargs)
        self.assertEqual(result.shape, reference.shape)
        np.testing.assert_allclose(result.sum(axis=(1, 2, 3)), reference.sum(axis=(1, 2, 3)), rtol=1e-10)
        np.testing.assert_array_equal(result[:, :4], 0.0)
        np.testing.assert_array_equal(result[:, 12:], 0.0)
        self.assertTrue(np.isfinite(result).all())
        config = RawReconConfig(reconstruction_matrix=16, reconstruction_method="osem",
                                iterative_iterations=1, osem_subsets=2, nitida4d_enabled=True,
                                nitida4d_iterations=1, nitida4d_subsets=2, nitida2_mode="spatiotemporal",
                                recon_slice_range=(2, 5))
        pipeline = reconstruct_raw_gated_pipeline(raw, config=config)
        self.assertEqual(pipeline.gated_volume.shape, reference.shape)
        self.assertTrue(np.isfinite(pipeline.gated_volume).all())

    def test_fine_nitida4d_zero_prior_matches_independent_osem(self):
        from core.nitida4d import nitida4d_osem_gated
        raw = make_synthetic_raw_motion_projections(n_gates=3, n_angles=8, rows=8, cols=8).projections
        theta = np.linspace(0, 180, 8, endpoint=False)
        kwargs = dict(output_size=16, axial_resample=True, iterations=2, subsets=2, slice_range=(2, 5))
        expected = reconstruct_gated_projection_volume(raw, theta, method="osem", **kwargs)
        actual = nitida4d_osem_gated(raw, theta, beta_temporal=0, rescale=False, **kwargs)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_explicit_acquired_matrix_keeps_native_results_unchanged(self):
        from dataclasses import replace
        raw = make_synthetic_raw_motion_projections(n_gates=4, n_angles=8, rows=8, cols=8).projections
        for options in ({}, {"nitida3_enabled": True}, {"nitida4d_enabled": True},
                        {"nitida2_mode": "spatiotemporal"}):
            config = RawReconConfig(reconstruction_method="osem", iterative_iterations=1,
                                    osem_subsets=2, nitida3_iterations=1, nitida4d_iterations=1,
                                    recon_slice_range=(2, 5), **options)
            native = reconstruct_raw_gated_pipeline(raw, config=config)
            with patch("core.raw_reconstruction._fine_system_matrix", side_effect=AssertionError("native entered fine operator")):
                explicit = reconstruct_raw_gated_pipeline(raw, config=replace(config, reconstruction_matrix=8))
            np.testing.assert_array_equal(explicit.gated_volume, native.gated_volume)
            np.testing.assert_array_equal(explicit.ungated_volume, native.ungated_volume)

    def test_fine_gates_share_projection_operators(self):
        from core.raw_reconstruction import _fine_system_matrix
        raw = make_synthetic_raw_motion_projections(n_gates=3, n_angles=8, rows=8, cols=8).projections
        with patch("core.raw_reconstruction._fine_system_matrix", wraps=_fine_system_matrix) as builder:
            reconstruct_gated_projection_volume(raw, method="osem", output_size=16,
                                                 axial_resample=True, iterations=1, subsets=2)
        self.assertEqual(builder.call_count, 2)

    def test_fine_ac_is_rejected_instead_of_using_native_geometry(self):
        raw = make_synthetic_raw_motion_projections(n_gates=1, n_angles=8, rows=8, cols=8).projections
        with self.assertRaisesRegex(ValueError, "AC/registro CT"):
            reconstruct_raw_gated_pipeline(raw, config=RawReconConfig(
                reconstruction_matrix=16, reconstruction_method="osem", attenuation_correction=True))

    def test_fine_feta_uses_original_detector_rows_and_keeps_native_samples(self):
        from core.raw_reconstruction import reconstruct_fbp_volume
        raw = make_synthetic_raw_motion_projections(n_gates=1, n_angles=12, rows=8, cols=16).projections[0]
        native = reconstruct_fbp_volume(raw, slice_range=(2, 5))
        fine = reconstruct_fbp_volume(raw, output_size=32, axial_resample=True, slice_range=(2, 5))
        self.assertEqual(fine.shape, (16, 32, 32))
        np.testing.assert_allclose(fine[::2, ::2, ::2], native, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(fine[:4], 0.0)
        np.testing.assert_array_equal(fine[12:], 0.0)
        np.testing.assert_allclose(fine[5], 0.5 * (fine[4] + fine[6]), rtol=1e-10, atol=1e-10)

    def test_fine_grid_composite_branches_keep_matching_geometry(self):
        raw = make_synthetic_raw_motion_projections(n_gates=4, n_angles=12, rows=8, cols=16).projections
        cfg = RawReconConfig(reconstruction_matrix=32, ungated_denoise_plus=True,
                             gated_denoise_plus=True, fbp_clean_sigma_color=0.04,
                             nitida2_mode="temporal", background_subtract=True)
        result = reconstruct_raw_gated_pipeline(raw, config=cfg)
        self.assertEqual(result.gated_volume.shape, (4, 16, 32, 32))
        self.assertEqual(result.ungated_volume.shape, (16, 32, 32))
        self.assertTrue(np.isfinite(result.gated_volume).all())
        self.assertGreaterEqual(float(result.gated_volume.min()), 0.0)
        self.assertFalse(any("fall" in note.lower() for note in result.notes), result.notes)

    def test_fine_grid_preserves_physical_post_filter_width(self):
        import scipy.ndimage
        raw = make_synthetic_raw_motion_projections(n_gates=2, n_angles=8, rows=8, cols=16).projections
        cfg = RawReconConfig(reconstruction_matrix=32, post_filter_sigma_px=0.75)
        with patch("scipy.ndimage.gaussian_filter", wraps=scipy.ndimage.gaussian_filter) as smoothing:
            reconstruct_raw_gated_pipeline(raw, config=cfg)
        self.assertEqual(smoothing.call_count, 3)
        for call in smoothing.call_args_list:
            self.assertEqual(call.kwargs["sigma"], 1.5)

    def test_fbp_fine_matrix_preserves_field_and_native_sample_values(self):
        from core.raw_reconstruction import reconstruct_fbp_volume
        raw = make_synthetic_raw_motion_projections(n_gates=1, n_angles=12, rows=8, cols=16).projections[0]
        for filter_name in ("ramp", "hann", "cosine"):
            native = reconstruct_fbp_volume(raw, fbp_filter_name=filter_name)
            fine = reconstruct_fbp_volume(raw, fbp_filter_name=filter_name, output_size=32)
            self.assertEqual(fine.shape, (8, 32, 32))
            np.testing.assert_allclose(fine[:, ::2, ::2], native, rtol=1e-10, atol=1e-10)

    def test_snapshot_reslice_keeps_physical_center_and_grid_between_matrices(self):
        from _xel3_filter_cmp import reslice_snapshot_grid
        native_axes = np.indices((16, 16, 16), dtype=float)
        native = (native_axes[0] + 2 * native_axes[1] + 3 * native_axes[2])[None]
        fine_axes = np.indices((32, 32, 32), dtype=float) * 0.5
        fine_axes[2] -= 0.5
        fine = (fine_axes[0] + 2 * fine_axes[1] + 3 * fine_axes[2])[None]
        orientation = {"center": [8, 8, 8], "long_axis": [1, 0, 0],
                       "sample_scale": 1.0, "out_size": 8}
        expected = reslice_snapshot_grid(native, orientation, 1.0)
        actual = reslice_snapshot_grid(fine, orientation, 0.5, source_origin=(0, 0, -0.5))
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertEqual(reslice_snapshot_grid(fine, orientation, 0.5, output_scale=0.5).shape,
                         (1, 16, 16, 16))

    def test_matrix_comparison_rejects_stale_snapshot_before_creating_output(self):
        import json
        import tempfile
        from pathlib import Path
        from _xel3_filter_cmp import compare_snapshot_matrices
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dcm"
            source.touch()
            metadata = {"schema_version": 2, "stage": "stress", "source_path": str(source),
                        "reconstruction_voxel_scale": 1.0, "reconstruction_sha256": "stale"}
            (root / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "reorient.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.save(root / "gated_volume.npy", np.ones((2, 4, 4, 4)))
            with self.assertRaisesRegex(ValueError, "changed after orientation"):
                compare_snapshot_matrices(root, root / "output", root)
            self.assertFalse((root / "output").exists())

    def test_manual_audit_rejects_changed_reconstruction(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from _xel3_filter_cmp import audit_manual_snapshot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "gated_volume.npy", np.ones((8, 8, 8, 8)))
            with (root / "gated_volume.npy").open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            metadata = {"schema_version": 2, "stage": "rest", "reconstruction_sha256": digest}
            (root / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "reorient.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.save(root / "gated_volume.npy", np.zeros((8, 8, 8, 8)))
            with self.assertRaisesRegex(ValueError, "changed after orientation"):
                audit_manual_snapshot(root, root / "audit")
            self.assertFalse((root / "audit").exists())

    def test_manual_audit_is_repeatable_without_claiming_clinical_validation(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from _xel3_filter_cmp import audit_manual_snapshot
        from tests.test_ectb_lv import _make_phantom, PIXEL_MM
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cube, segmentation = _make_phantom()
            np.save(root / "gated_volume.npy", cube)
            np.save(root / "reoriented_gated.npy", cube)
            with (root / "gated_volume.npy").open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            metadata = {"schema_version": 2, "stage": "rest", "reconstruction_sha256": digest,
                        "pixel_spacing": [PIXEL_MM, PIXEL_MM]}
            orientation = {**metadata, "base_k": 0, "apex_k": cube.shape[1] - 1, "voxel_mm": PIXEL_MM}
            (root / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "reorient.json").write_text(json.dumps(orientation), encoding="utf-8")
            with patch("core.segmentation.segment_myocardium", return_value=segmentation):
                report = audit_manual_snapshot(root, root / "audit")
            self.assertTrue(report["available"])
            self.assertEqual(len(report["gate_volumes_ml"]), cube.shape[0])
            self.assertTrue(report["bitwise_repeatable"])
            self.assertFalse(report["clinical_validation"])
            self.assertTrue((root / "audit" / "gate_contours.png").exists())

    def test_iterative_butterworth_clips_ringing_without_modifying_input(self):
        volume = np.zeros((16, 16, 16), dtype=np.float64)
        volume[8, 8, 8] = 1.0
        original = volume.copy()
        filtered = _butterworth_3d(volume, 0.40, 10)
        self.assertLess(float(filtered.min()), -1e-6)
        actual = _apply_post_butterworth(volume, (0.40, 10), None)
        np.testing.assert_allclose(actual, np.clip(filtered, 0.0, None))
        np.testing.assert_array_equal(volume, original)

    def test_iterative_butterworth_clips_ringing_only_in_reconstructed_band(self):
        volume = np.zeros((16, 16, 16), dtype=np.float64)
        volume[8, 8, 8] = 1.0
        original = volume.copy()
        actual = _apply_post_butterworth(volume, (0.40, 10), (4, 11))
        self.assertGreaterEqual(float(actual.min()), 0.0)
        expected = np.clip(_butterworth_3d(volume[4:12], 0.40, 10), 0.0, None)
        np.testing.assert_allclose(actual[4:12], expected)
        np.testing.assert_array_equal(actual[:4], volume[:4])
        np.testing.assert_array_equal(actual[12:], volume[12:])
        np.testing.assert_array_equal(volume, original)

    def test_osem_butterworth_is_post_reconstruction_and_changes_output(self):
        raw = make_synthetic_raw_motion_projections(n_gates=2, n_angles=8, rows=8, cols=8).projections
        original = raw.copy()
        baseline = reconstruct_gated_projection_volume(raw, method="osem", iterations=2, subsets=2)
        outputs = []
        for cutoff in (0.20, 0.60):
            actual = reconstruct_gated_projection_volume(
                raw, method="osem", iterations=2, subsets=2,
                projection_filter=ProjectionFilterConfig("butterworth", cutoff, 5),
            )
            expected = np.stack([_apply_post_butterworth(gate, (cutoff, 5), None) for gate in baseline])
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
            self.assertTrue(np.isfinite(actual).all())
            self.assertGreaterEqual(float(actual.min()), 0.0)
            self.assertFalse(np.allclose(actual, baseline))
            outputs.append(actual)
        self.assertFalse(np.allclose(outputs[0], outputs[1]))
        np.testing.assert_array_equal(raw, original)

    def test_gated_denoise_plus_preserves_branch_iterations_and_subsets(self):
        raw = make_synthetic_raw_motion_projections(n_gates=2, n_angles=8, rows=8, cols=8).projections
        cfg = RawReconConfig(
            reconstruction_method="osem", iterative_iterations=1, osem_subsets=2,
            gated_method="osem", gated_iterations=3, gated_osem_subsets=4,
            gated_denoise_plus=True,
        )
        with patch("core.raw_reconstruction.reconstruct_gated_projection_volume",
                   wraps=reconstruct_gated_projection_volume) as reconstruct:
            result = reconstruct_raw_gated_pipeline(raw, config=cfg)
        self.assertEqual(reconstruct.call_count, 2)
        for call in reconstruct.call_args_list:
            self.assertEqual(call.kwargs["iterations"], 3)
            self.assertEqual(call.kwargs["subsets"], 4)
        self.assertTrue(np.isfinite(result.gated_volume).all())
        self.assertTrue(any("Denoise+ GATED:" in note for note in result.notes))

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

    def test_detect_rotation_ccw_from_angles(self):
        self.assertTrue(_detect_rotation_ccw(np.array([0.0, 3.0, 6.0, 9.0])))       # crece = CCW
        self.assertFalse(_detect_rotation_ccw(np.array([9.0, 6.0, 3.0, 0.0])))      # decrece = CW
        self.assertTrue(_detect_rotation_ccw(np.array([358.0, 1.0, 4.0])))          # cruce 360 crece
        self.assertIsNone(_detect_rotation_ccw(None))
        self.assertIsNone(_detect_rotation_ccw(np.array([5.0])))


if __name__ == "__main__":
    unittest.main()
