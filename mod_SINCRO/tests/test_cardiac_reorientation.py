import unittest

import numpy as np

from core.cardiac_reorientation import (
    ReorientationParams,
    default_center,
    long_axis_vector,
    reslice_from_vector,
    reslice_from_vector_gated,
    reslice_oblique,
    reslice_oblique_gated,
)


def _shell_volume(n=32):
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    shell = ((r > n * 0.12) & (r < n * 0.22) & (zz > n * 0.25) & (zz < n * 0.75)).astype(float)
    return shell


class CardiacReorientationTests(unittest.TestCase):
    def test_long_axis_vector_is_unit(self):
        u = long_axis_vector(0.4, 0.3)
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=6)

    def test_default_center_is_volume_center(self):
        v = _shell_volume(32)
        cz, cy, cx = default_center(v)
        self.assertAlmostEqual(cy, 16.0, delta=1.0)
        self.assertAlmostEqual(cx, 16.0, delta=1.0)

    def test_reslice_oblique_shape_and_energy(self):
        v = _shell_volume(32)
        p = ReorientationParams(center=default_center(v), theta=0.0, phi=0.0, out_size=32)
        out = reslice_oblique(v, p)
        self.assertEqual(out.shape, (32, 32, 32))
        self.assertGreater(float(out.max()), 0.5)

    def test_reslice_gated_preserves_gate_axis(self):
        v = _shell_volume(24)
        cube = np.stack([v, v * 0.8, v * 0.6])
        p = ReorientationParams(center=default_center(v), theta=0.1, phi=0.05, out_size=24)
        out = reslice_oblique_gated(cube, p)
        self.assertEqual(out.shape, (3, 24, 24, 24))

    def test_reslice_from_vector_axial_axis_matches_oblique(self):
        # Con el MISMO eje largo, reslice_from_vector == reslice_oblique.
        # long_axis_vector(0,0) = (0,0,1) (eje x), no el eje z.
        v = _shell_volume(32)
        c = default_center(v)
        u = long_axis_vector(0.0, 0.0)
        out_vec = reslice_from_vector(v, c, u, 32)
        out_obl = reslice_oblique(v, ReorientationParams(center=c, theta=0.0, phi=0.0, out_size=32))
        self.assertEqual(out_vec.shape, (32, 32, 32))
        self.assertGreater(float(out_vec.max()), 0.5)
        self.assertLess(float(np.abs(out_vec - out_obl).mean()), 1e-6)

    def test_reslice_from_vector_tilted_shape_and_energy(self):
        v = _shell_volume(32)
        u = np.array([0.9, 0.2, 0.3])
        out = reslice_from_vector(v, default_center(v), u, 32)
        self.assertEqual(out.shape, (32, 32, 32))
        self.assertGreater(float(out.max()), 0.4)

    def test_reslice_from_vector_gated_preserves_gate_axis(self):
        v = _shell_volume(24)
        cube = np.stack([v, v * 0.8, v * 0.6])
        out = reslice_from_vector_gated(cube, default_center(v), np.array([1.0, 0.1, 0.0]), 24)
        self.assertEqual(out.shape, (3, 24, 24, 24))

    def test_reslice_from_vector_zero_vector_falls_back(self):
        v = _shell_volume(16)
        out = reslice_from_vector(v, default_center(v), np.array([0.0, 0.0, 0.0]), 16)
        self.assertEqual(out.shape, (16, 16, 16))


if __name__ == "__main__":
    unittest.main()
