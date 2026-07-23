"""Tests de exportación DICOM de ejes cardíacos SA/HLA/VLA."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dicom_export import save_cardiac_axes_dicoms
from core.dicom_loader import load


class CardiacAxesDicomExportTests(unittest.TestCase):
    def test_save_cardiac_axes_dicoms_roundtrip_sa(self):
        axes = {
            "SA": np.random.default_rng(1).integers(0, 1000, (2, 3, 8, 8)).astype(np.float64),
            "HLA": np.random.default_rng(2).integers(0, 1000, (2, 4, 3, 8)).astype(np.float64),
            "VLA": np.random.default_rng(3).integers(0, 1000, (2, 4, 3, 8)).astype(np.float64),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_cardiac_axes_dicoms(axes, tmpdir, base_name="axes_test", slice_thickness_mm=2.0)

            self.assertEqual(set(paths), {"SA", "HLA", "VLA"})
            for path in paths.values():
                self.assertTrue(os.path.exists(path))

            sa = load(paths["SA"])
            self.assertTrue(sa.reconstructed)
            self.assertEqual(sa.cube.shape, axes["SA"].shape)
            self.assertIn("SA", sa.image_type)
            self.assertIn("GammaSync SA", sa.series_description)


if __name__ == "__main__":
    unittest.main()
