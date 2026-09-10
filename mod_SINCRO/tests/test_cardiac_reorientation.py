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

    def test_reslice_from_vector_supports_finer_sampling(self):
        v = _shell_volume(32)
        c = default_center(v)
        u = np.array([0.0, 0.2, 1.0])
        out = reslice_from_vector(v, c, u, 32, sample_scale=0.5)
        self.assertEqual(out.shape, (32, 32, 32))
        self.assertGreater(float(out.max()), 0.4)

    def test_reslice_from_vector_zero_vector_falls_back(self):
        v = _shell_volume(16)
        out = reslice_from_vector(v, default_center(v), np.array([0.0, 0.0, 0.0]), 16)
        self.assertEqual(out.shape, (16, 16, 16))


class LongAxisRefinementTests(unittest.TestCase):
    @staticmethod
    def _oblique_tube_points(u_true, length=18.0, radius=6.0, n_ax=40, n_ang=36):
        """Nube (N,3) de una pared cilíndrica hueca a lo largo de ``u_true``.

        Cortada ⟂ a ``u_true`` da anillos circulares: es el caso ideal para el
        refinamiento por circularidad.
        """
        u = np.asarray(u_true, dtype=np.float64)
        u = u / np.linalg.norm(u)
        # base perpendicular a u
        a = np.array([0.0, 1.0, 0.0]) if abs(u[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = a - np.dot(a, u) * u
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        pts = []
        for t in np.linspace(-length, length, n_ax):
            for ang in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
                pts.append(t * u + radius * (np.cos(ang) * e1 + np.sin(ang) * e2))
        return np.asarray(pts, dtype=np.float64)

    def test_refine_improves_from_nearby_seed(self):
        from core.cardiac_reorientation import _refine_long_axis_circularity

        u_true = np.array([0.7, 0.0, 0.714])
        u_true = u_true / np.linalg.norm(u_true)
        pts = self._oblique_tube_points(u_true)
        w = np.ones(len(pts))
        # Seed ~14° del eje real (rango realista de un PCA): el default debe afinar.
        u0 = np.array([0.85, 0.0, 0.527])
        u0 = u0 / np.linalg.norm(u0)
        u_ref = _refine_long_axis_circularity(pts, w, u0)
        cos_true = abs(float(np.dot(u_ref, u_true)))
        cos_start = abs(float(np.dot(u0, u_true)))
        self.assertGreater(cos_true, cos_start)
        self.assertGreater(cos_true, 0.98)

    def test_refine_default_cone_is_conservative(self):
        from core.cardiac_reorientation import _refine_long_axis_circularity

        # Desde un seed lejano (vertical, ~45° del real) el DEFAULT no debe saltar
        # de hemisferio: se queda anclado al seed (cono estrecho = estabilidad).
        u_true = np.array([0.7, 0.0, 0.714])
        u_true = u_true / np.linalg.norm(u_true)
        pts = self._oblique_tube_points(u_true)
        w = np.ones(len(pts))
        u0 = np.array([1.0, 0.0, 0.0])
        u_ref = _refine_long_axis_circularity(pts, w, u0)
        self.assertLessEqual(
            np.rad2deg(np.arccos(abs(float(np.dot(u_ref, u0))))), 20.5
        )

    def test_refine_keeps_axis_within_cone(self):
        from core.cardiac_reorientation import _refine_long_axis_circularity

        u_true = np.array([0.7, 0.0, 0.714])
        u_true = u_true / np.linalg.norm(u_true)
        pts = self._oblique_tube_points(u_true)
        w = np.ones(len(pts))
        u0 = np.array([1.0, 0.0, 0.0])
        u_ref = _refine_long_axis_circularity(pts, w, u0, max_deg=10.0)
        # con un cono de 10° no puede alcanzar los ~45° hacia u_true
        self.assertLessEqual(
            np.rad2deg(np.arccos(abs(float(np.dot(u_ref, u0))))), 10.5
        )


class ApexSignResolverTests(unittest.TestCase):
    @staticmethod
    def _tapered_shell(u_true, apex_at_plus=True, length=18.0, n_ax=40, n_ang=36):
        """Nube cónica hueca: ancha en un extremo, angosta en el otro (ápex)."""
        u = np.asarray(u_true, float); u /= np.linalg.norm(u)
        a = np.array([0.0, 1.0, 0.0]) if abs(u[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = a - np.dot(a, u) * u; e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        pts = []
        for t in np.linspace(-length, length, n_ax):
            frac = (t + length) / (2 * length)  # 0 en -u .. 1 en +u
            # radio grande en la base, chico en el ápex
            base_side = (1.0 - frac) if apex_at_plus else frac
            radius = 2.0 + 6.0 * base_side
            for ang in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
                pts.append(t * u + radius * (np.cos(ang) * e1 + np.sin(ang) * e2))
        return np.asarray(pts, float)

    def test_apex_sign_points_to_narrow_end(self):
        from core.cardiac_reorientation import _resolve_apex_sign

        u = np.array([0.3, 0.8, 0.5]); u /= np.linalg.norm(u)
        pts = self._tapered_shell(u, apex_at_plus=True)
        w = np.ones(len(pts))
        self.assertEqual(_resolve_apex_sign(pts, w, u), 1)
        pts2 = self._tapered_shell(u, apex_at_plus=False)
        self.assertEqual(_resolve_apex_sign(pts2, np.ones(len(pts2)), u), -1)

    def test_apex_sign_ambiguous_on_symmetric_tube(self):
        from core.cardiac_reorientation import _resolve_apex_sign

        u = np.array([0.3, 0.8, 0.5]); u /= np.linalg.norm(u)
        # tubo recto (sin taper) -> empate -> 0 (desempate externo)
        pts = LongAxisRefinementTests._oblique_tube_points(u)
        self.assertEqual(_resolve_apex_sign(pts, np.ones(len(pts)), u), 0)


if __name__ == "__main__":
    unittest.main()

