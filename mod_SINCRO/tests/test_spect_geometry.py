"""Tests de core/spect_geometry: extracción y vistas anatómicas de referencia."""
import unittest

import numpy as np

from core.spect_geometry import SpectGeometry, reference_views, reproject_view


class TestSpectGeometryAngles(unittest.TestCase):
    def test_anterior_lateral_cw_180(self):
        # Órbita cardíaca 180° típica: start 225°, CW.
        geo = SpectGeometry(patient_position="FFS", start_angle=225.0, angular_step=3.0,
                            rotation_direction="CW", scan_arc=180.0, n_angles=60)
        # CW resta: anterior = 225-45 = 180; lateral = 225-135 = 90.
        self.assertAlmostEqual(geo.anterior_angle(), 180.0, places=3)
        self.assertAlmostEqual(geo.left_lateral_angle(), 90.0, places=3)
        # Separación anterior<->lateral siempre 90°.
        diff = abs(geo.anterior_angle() - geo.left_lateral_angle()) % 360.0
        self.assertAlmostEqual(min(diff, 360.0 - diff), 90.0, places=3)

    def test_anterior_lateral_cc(self):
        geo = SpectGeometry(start_angle=45.0, angular_step=3.0, rotation_direction="CC",
                            scan_arc=180.0, n_angles=60)
        # CC suma: anterior = 45+45 = 90; lateral = 45+135 = 180.
        self.assertAlmostEqual(geo.anterior_angle(), 90.0, places=3)
        self.assertAlmostEqual(geo.left_lateral_angle(), 180.0, places=3)

    def test_no_angle_metadata(self):
        geo = SpectGeometry(patient_position="HFS")
        self.assertIsNone(geo.anterior_angle())
        self.assertIsNone(geo.left_lateral_angle())

    def test_patient_position_flags(self):
        self.assertTrue(SpectGeometry(patient_position="HFS").head_first)
        self.assertFalse(SpectGeometry(patient_position="FFS").head_first)
        self.assertTrue(SpectGeometry(patient_position="FFS").supine)
        self.assertFalse(SpectGeometry(patient_position="FFP").supine)

    def test_from_raw_projections(self):
        class _Raw:
            patient_position = "FFS"
            start_angle = 225.0
            angular_step = 3.0
            rotation_direction = "CW"
            scan_arc = 180.0
            n_angles = 60
        geo = SpectGeometry.from_raw_projections(_Raw())
        self.assertEqual(geo.patient_position, "FFS")
        self.assertAlmostEqual(geo.start_angle, 225.0)
        self.assertEqual(geo.n_angles, 60)


class TestReferenceViews(unittest.TestCase):
    def _phantom(self, nz=16, w=24):
        vol = np.zeros((nz, w, w), dtype=np.float64)
        zz, yy, xx = np.mgrid[0:nz, 0:w, 0:w]
        vol += np.exp(-(((zz - nz / 2) ** 2) / 8.0 + ((yy - w / 2) ** 2) / 6.0 + ((xx - w / 2) ** 2) / 6.0))
        return vol

    def test_reproject_view_shape(self):
        vol = self._phantom()
        view = reproject_view(vol, 90.0)
        self.assertEqual(view.shape, (vol.shape[0], vol.shape[2]))
        self.assertTrue(np.all(np.isfinite(view)))
        self.assertGreater(view.sum(), 0.0)

    def test_reference_views_with_geometry(self):
        vol = self._phantom()
        geo = SpectGeometry(patient_position="FFS", start_angle=225.0, angular_step=3.0,
                            rotation_direction="CW", scan_arc=180.0, n_angles=60)
        out = reference_views(vol, geo)
        self.assertEqual(out["anterior"].ndim, 2)
        self.assertEqual(out["left_lateral"].ndim, 2)
        self.assertFalse(out["synthesized_lateral"])
        self.assertAlmostEqual(out["anterior_angle"], 180.0, places=3)
        self.assertAlmostEqual(out["left_lateral_angle"], 90.0, places=3)

    def test_reference_views_synthesized_lateral(self):
        vol = self._phantom()
        geo = SpectGeometry(patient_position="FFS", start_angle=123.7, angular_step=2.8125,
                            rotation_direction="CW", scan_arc=101.25, n_angles=36)
        out = reference_views(vol, geo)
        self.assertTrue(out["synthesized_lateral"])

    def test_reference_views_fallback_no_geometry(self):
        vol = self._phantom()
        geo = SpectGeometry(patient_position="HFS")
        out = reference_views(vol, geo)
        self.assertIsNone(out["anterior_angle"])
        self.assertEqual(out["anterior"].ndim, 2)
        self.assertEqual(out["left_lateral"].ndim, 2)


if __name__ == "__main__":
    unittest.main()
