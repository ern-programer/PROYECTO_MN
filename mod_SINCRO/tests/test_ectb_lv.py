"""
Test del método ECTb de cuantificación de VI (core/ectb_lv.py).

Usa un fantoma sintético de geometría CONOCIDA: un cilindro miocárdico que se
contrae y engrosa entre telediástole y telesístole. Como el volumen verdadero
es calculable analíticamente, se puede verificar que el método recupera la FEVI.

Correr:  python -m pytest tests/test_ectb_lv.py -v
     o:  python tests/test_ectb_lv.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.console_utf8 import enable_utf8  # noqa: E402
from core.ectb_lv import (  # noqa: E402
    SEPTAL_ANGLE_DEG,
    ECTbLVConfig,
    _base_is_last,
    _valve_plane_weights,
    analyze_lv_ectb,
    apply_regression,
    convert_ef_pct,
    regression_equation_text,
)

enable_utf8()

PIXEL_MM = 6.4
SLICE_MM = 6.4


@pytest.mark.parametrize("minimum_mm", [0.0, 3.0, 8.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_lv_axial_interpolation_stays_within_neighboring_contours(minimum_mm, reverse):
    from core.lv_mesh import _interpolate_radii_z
    z = np.arange(7, dtype=float) * 6.0
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    sector = np.exp(-(np.angle(np.exp(1j * angles)) / 0.5) ** 2)
    radii = np.full((7, 64), 26.0)
    radii[2:4] -= (26.0 - minimum_mm) * sector[None, :]
    original = radii.copy()
    order = slice(None, None, -1) if reverse else slice(None)
    interpolated, positions = _interpolate_radii_z(radii[order], z[order], 3.0)
    left = np.clip(np.searchsorted(z, positions, side="right") - 1, 0, len(z) - 2)
    lower = np.minimum(radii[left], radii[left + 1])
    upper = np.maximum(radii[left], radii[left + 1])
    assert np.all(interpolated >= lower - 1e-10)
    assert np.all(interpolated <= upper + 1e-10)
    assert np.array_equal(radii, original)
    assert np.allclose(interpolated[[0, -1]], radii[order][[0, -1]])


@pytest.mark.parametrize("reverse", [False, True])
def test_lv_model_low_local_radii_do_not_invert_vertices(reverse):
    from core.lv_mesh import myocardium_shell_mesh
    z = np.arange(7, dtype=float) * 6.0
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    sector = np.exp(-(np.angle(np.exp(1j * angles)) / 0.5) ** 2)
    endo = np.full((7, 64), 26.0)
    epi = np.full((7, 64), 33.0)
    endo[2:4] -= 25.0 * sector[None, :]
    epi[2:4] -= 30.0 * sector[None, :]
    original_endo, original_epi = endo.copy(), epi.copy()
    order = slice(None, None, -1) if reverse else slice(None)
    shell = myocardium_shell_mesh(endo[order], epi[order], z[order], interp_z=3.0,
                                  smooth_angular=1.0, ventricular_shape=True)
    n_ring = int(shell.field_data["shell_n_ring"][0])
    for offset in (0, n_ring):
        rings = np.asarray(shell.points)[offset:offset + n_ring].reshape(-1, 64, 3)
        signed_radii = np.einsum("raj,aj->ra", rings[:, :, :2], directions)
        assert np.all(signed_radii >= -1e-10)
    assert np.all(shell.compute_cell_sizes().cell_data["Area"] > 1e-8)
    assert np.array_equal(endo, original_endo)
    assert np.array_equal(epi, original_epi)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("interp_z", [1.0, 3.0])
@pytest.mark.parametrize("virtual_rings", [24, 80])
def test_lv_shell_apex_joins_measured_contours_smoothly(reverse, interp_z, virtual_rings):
    from core.lv_mesh import myocardium_shell_mesh
    angles = 64
    z = np.arange(6, dtype=float) * 6.0
    endo = np.broadcast_to((10.0 + 0.5 * z)[:, None], (6, angles)).copy()
    epi = endo + 7.0
    order = slice(None, None, -1) if reverse else slice(None)
    mesh = myocardium_shell_mesh(endo[order], epi[order], z[order], interp_z=interp_z,
                                 apex_virtual_rings=virtual_rings, smooth_iter=6, smooth_relax=0.04)
    points = np.asarray(mesh.points)
    n_ring = int(mesh.field_data["shell_n_ring"][0])
    for offset, radii in ((0, epi), (n_ring, endo)):
        rings = points[offset:offset + n_ring].reshape(-1, angles, 3)
        radial = np.linalg.norm(rings[:, :, :2], axis=2).mean(axis=1)
        axial = rings[:, :, 2].mean(axis=1)
        assert np.all(np.diff(axial) > 0)
        assert np.isclose(radial[virtual_rings], radii[0, 0])
        if interp_z == 1:
            assert np.allclose(radial[virtual_rings:virtual_rings + 2], radii[:2, 0])
        measured_slope = (radial[virtual_rings + 1] - radial[virtual_rings]) / (axial[virtual_rings + 1] - axial[virtual_rings])
        join_slope = (radial[virtual_rings] - radial[virtual_rings - 1]) / (axial[virtual_rings] - axial[virtual_rings - 1])
        assert abs(join_slope - measured_slope) < 0.2
    assert np.isclose(points[n_ring, 2] - points[0, 2], 7.0)
    assert np.isfinite(points).all()


@pytest.mark.parametrize("wall_mm", [4.0, 7.0, 12.0])
@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("cylindrical", [False, True])
def test_ventricular_profiles_close_independently_without_flat_apex(cylindrical, asymmetric, wall_mm):
    from core.lv_mesh import _ventricular_shell_profiles
    z = np.linspace(0, 36, 7)
    endo_radius = np.full(7, 18.0) if cylindrical else 22.0 * np.sqrt(1 - ((z - 24) / 42) ** 2)
    endo = np.broadcast_to(endo_radius[:, None], (7, 64)).copy()
    if asymmetric:
        endo *= 1.0 + 0.12 * np.cos(np.linspace(0, 2 * np.pi, 64, endpoint=False))[None, :]
    epi = endo + wall_mm
    original_endo, original_epi = endo.copy(), epi.copy()
    inner, outer, inner_z, outer_z, inner_centers, outer_centers = _ventricular_shell_profiles(
        endo, epi, z, np.zeros((7, 2)), 96,
    )
    assert np.all(inner[0] == 0) and np.all(outer[0] == 0)
    assert np.isclose(inner_z[0] - outer_z[0], wall_mm)
    assert -outer_z[0] >= 0.85 * epi[0].mean()
    assert outer_z[-1] == inner_z[-1] == z[-1]
    assert np.allclose(inner[-1], endo[-1]) and np.allclose(outer[-1], epi[-1])
    for radii, positions in ((inner, inner_z), (outer, outer_z)):
        assert np.all(np.diff(positions) > 0)
        assert np.isfinite(radii).all()
        assert radii[5].mean() < radii[48].mean() * 0.25
    assert np.array_equal(endo, original_endo) and np.array_equal(epi, original_epi)
    for angle in range(64):
        inner_at_outer = np.interp(outer_z, inner_z, inner[:, angle], left=0.0)
        assert np.all(outer[1:, angle] > inner_at_outer[1:])


def test_ventricular_mesh_has_two_separate_poles_and_open_basal_cavity():
    from core.lv_mesh import myocardium_shell_mesh, split_myocardium_shell
    z = np.linspace(0, 36, 7)
    endo = np.full((7, 64), 18.0)
    epi = endo + 7.0
    shell = myocardium_shell_mesh(endo, epi, z, ventricular_shape=True, smooth_iter=6)
    outer, inner = split_myocardium_shell(shell)
    assert shell.n_open_edges == 0
    assert inner.n_open_edges == 64
    assert np.isclose(inner.bounds[4] - outer.bounds[4], 7.0)
    assert inner.bounds[5] == outer.bounds[5] == z[-1]
    assert np.all(shell.compute_cell_sizes().cell_data["Area"] > 1e-8)
    for mesh in (outer, inner):
        assert np.isfinite(mesh.compute_normals().point_data["Normals"]).all()
        pole = mesh.points[np.isclose(mesh.points[:, 2], mesh.bounds[4])]
        assert len(pole) == 1
        basal = mesh.points[np.isclose(mesh.points[:, 2], z[-1])]
        assert np.min(np.linalg.norm(basal[:, :2], axis=1)) > 10


class _FakeSeg:
    """Segmentación mínima con lo que consume `analyze_lv_ectb`."""

    def __init__(self, mask, centers, outer, inner=None):
        self.mask = mask
        self.center_per_slice = centers
        self.outer_radius = outer
        self.inner_radius = (
            inner if inner is not None else np.full((mask.shape[0],), np.nan)
        )


def _make_phantom(n_gates=8, n_slices=12, size=32, ef_target=0.55, wall_ed_mm=10.0, seed=0):
    """Cilindro miocárdico gatillado con FEVI conocida.

    El radio endocárdico varía cosenoidalmente entre ED y ES de modo que la
    relación de áreas dé la FEVI pedida. La pared engrosa en sístole
    conservando el área miocárdica (músculo incompresible), y las cuentas se
    hacen proporcionales al espesor para reproducir el volumen parcial.
    """
    rng = np.random.default_rng(seed)
    cy = cx = size / 2.0
    wall_ed_px = wall_ed_mm / PIXEL_MM
    r_endo_ed_px = 7.0

    # Area_ES = Area_ED * (1 - EF)  ->  r_ES = r_ED * sqrt(1 - EF)
    r_endo_es_px = r_endo_ed_px * np.sqrt(1.0 - ef_target)
    area_myo = np.pi * ((r_endo_ed_px + wall_ed_px) ** 2 - r_endo_ed_px ** 2)

    cube = np.zeros((n_gates, n_slices, size, size), dtype=np.float64)
    ys, xs = np.ogrid[:size, :size]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    for g in range(n_gates):
        # gate 0 = ED, mitad del ciclo = ES
        u = 0.5 * (1.0 + np.cos(2.0 * np.pi * g / n_gates))  # 1 en ED, 0 en ES
        r_endo = r_endo_es_px + (r_endo_ed_px - r_endo_es_px) * u
        r_epi = np.sqrt(r_endo ** 2 + area_myo / np.pi)
        thickness = r_epi - r_endo
        # Volumen parcial: cuentas proporcionales al espesor.
        counts = 1000.0 * (thickness / wall_ed_px)
        ring = (dist >= r_endo) & (dist <= r_epi)
        for s in range(n_slices):
            cube[g, s][ring] = counts
        cube[g] += rng.normal(0.0, 2.0, size=(n_slices, size, size))

    cube = np.clip(cube, 0.0, None)
    mask = cube[0] > (0.35 * cube[0].max())
    centers = np.tile([cy, cx], (n_slices, 1))
    outer = np.full((n_slices,), r_endo_ed_px + wall_ed_px + 2.0)
    return cube, _FakeSeg(mask, centers, outer)


@pytest.mark.parametrize("n_gates", [8, 16])
@pytest.mark.parametrize("amplitude", [0.10, 0.25])
def test_gate_count_compensation_does_not_invent_contraction(n_gates, amplitude):
    cube, seg = _make_phantom(n_gates=n_gates, ef_target=0.0)
    stationary = np.broadcast_to(cube[0], cube.shape).copy()
    gains = 1.0 + amplitude * (1.0 - np.cos(2 * np.pi * np.arange(n_gates) / n_gates))
    observed = stationary * gains[:, None, None, None]
    original = observed.copy()
    baseline = analyze_lv_ectb(observed, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)
    result = analyze_lv_ectb(
        observed, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
        ECTbLVConfig(normalize_gate_counts=True),
    )
    assert baseline.available and result.available
    assert baseline.ef_pct > 4.0
    assert result.ef_pct < 1e-8
    assert np.allclose(result.mean_wall_thickness_mm, 10.0)
    assert np.allclose(result.gate_count_scale_factors, 1.0 / gains)
    assert np.array_equal(observed, original)
    assert any("experimental" in note.lower() for note in result.notes)


@pytest.mark.parametrize("ef_target", [0.25, 0.55])
@pytest.mark.parametrize("n_gates", [8, 16])
def test_gate_count_compensation_is_gain_invariant_with_moving_wall(ef_target, n_gates):
    cube, seg = _make_phantom(n_gates=n_gates, ef_target=ef_target)
    gains = 1.0 + 0.2 * np.sin(2 * np.pi * np.arange(n_gates) / n_gates)
    config = ECTbLVConfig(normalize_gate_counts=True)
    reference = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM, config)
    observed = analyze_lv_ectb(
        cube * gains[:, None, None, None], seg, (PIXEL_MM, PIXEL_MM), SLICE_MM, config,
    )
    assert reference.available and observed.available
    assert reference.ef_pct > 10.0
    assert np.allclose(observed.gate_volumes_ml, reference.gate_volumes_ml, atol=1e-8)
    assert np.allclose(observed.mean_wall_thickness_mm, reference.mean_wall_thickness_mm)


@pytest.mark.parametrize("invalid_value", [0.0, np.nan, np.inf])
def test_ectb_rejects_empty_or_nonfinite_gate(invalid_value):
    cube, seg = _make_phantom()
    cube[3] = invalid_value
    result = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)
    assert not result.available
    assert "gate" in result.reason.lower()


def test_gate_count_compensation_rejects_excessive_rescaling():
    cube, seg = _make_phantom(ef_target=0.0)
    cube[3] *= 0.1
    result = analyze_lv_ectb(
        cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
        ECTbLVConfig(normalize_gate_counts=True),
    )
    assert not result.available
    assert "cuentas" in result.reason.lower()


@pytest.mark.parametrize("n_gates", [8, 16])
@pytest.mark.parametrize("ed_radius_mm", [12.0, 24.0, 36.0])
def test_count_conserving_blurred_phantom_recovers_known_fevi(n_gates, ed_radius_mm):
    from scipy.ndimage import gaussian_filter
    pixel_mm, size, slices, subpixels = 2.0, 64, 6, 4
    coordinates = (np.arange(size * subpixels) + 0.5) / subpixels - size / 2
    radius = np.hypot(coordinates[:, None], coordinates[None, :]) * pixel_mm
    myocardial_area = (ed_radius_mm + 10.0) ** 2 - ed_radius_mm ** 2
    es_radius_mm = ed_radius_mm * np.sqrt(1.0 - 0.55)
    images = []
    for gate in range(n_gates):
        inner = es_radius_mm + (ed_radius_mm - es_radius_mm) * 0.5 * (1 + np.cos(2 * np.pi * gate / n_gates))
        outer = np.sqrt(inner ** 2 + myocardial_area)
        tissue = ((radius >= inner) & (radius <= outer)).astype(float)
        image = tissue.reshape(size, subpixels, size, subpixels).mean(axis=(1, 3)) * 1000.0
        images.append(gaussian_filter(image, 10.0 / 2.35482 / pixel_mm, mode="constant"))
    cube = np.repeat(np.array(images)[:, None], slices, axis=1)
    mask = np.repeat((images[0] > 0.35 * np.max(images[0]))[None], slices, axis=0)
    seg = _FakeSeg(mask, np.tile([size / 2 - 0.5, size / 2 - 0.5], (slices, 1)),
                   np.full(slices, (ed_radius_mm + 10.0) / pixel_mm + 2.0))
    reference = analyze_lv_ectb(cube, seg, (pixel_mm, pixel_mm), 6.0,
                                 ECTbLVConfig(use_valve_plane=False))
    compensated = analyze_lv_ectb(cube, seg, (pixel_mm, pixel_mm), 6.0,
                                   ECTbLVConfig(use_valve_plane=False, normalize_gate_counts=True))
    assert reference.available and compensated.available
    assert abs(reference.ef_pct - 55.0) < 3.0
    assert abs(compensated.ef_pct - 55.0) < 3.0
    assert abs(compensated.ef_pct - reference.ef_pct) < 0.2


def test_ectb_rejects_fully_collapsed_cavity_in_a_gate():
    cube, seg = _make_phantom()
    result = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                              ECTbLVConfig(ed_wall_thickness_mm=65.0))
    assert not result.available and result.qc_blocked
    assert "cavidad" in result.reason.lower()
    assert result.endo_zero_fraction.max() == 1.0


def test_recupera_fevi_conocida():
    """Sobre un fantoma con FEVI 55% el método debe caer cerca de ese valor."""
    cube, seg = _make_phantom(ef_target=0.55)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    assert res.available, res.reason
    assert 40.0 <= res.ef_pct <= 70.0, f"FEVI fuera de rango: {res.ef_pct:.1f}%"
    assert res.edv_ml > res.esv_ml > 0.0
    assert np.isclose(res.sv_ml, res.edv_ml - res.esv_ml)
    print(f"[OK] FEVI recuperada: {res.ef_pct:.1f}% (objetivo 55%) | "
          f"EDV={res.edv_ml:.1f} ESV={res.esv_ml:.1f} mL")


def test_detecta_ventriculo_deprimido():
    """Un fantoma con FEVI baja tiene que dar claramente menos que uno normal."""
    cube_norm, seg_norm = _make_phantom(ef_target=0.60, seed=1)
    cube_low, seg_low = _make_phantom(ef_target=0.25, seed=1)

    ef_norm = analyze_lv_ectb(cube_norm, seg_norm, (PIXEL_MM, PIXEL_MM), SLICE_MM).ef_pct
    ef_low = analyze_lv_ectb(cube_low, seg_low, (PIXEL_MM, PIXEL_MM), SLICE_MM).ef_pct

    assert ef_norm - ef_low > 15.0, f"no discrimina: normal={ef_norm:.1f}% bajo={ef_low:.1f}%"
    print(f"[OK] discrimina función: normal={ef_norm:.1f}% vs deprimido={ef_low:.1f}%")


def test_ed_gate_y_es_gate_coherentes():
    """El gate de volumen máximo debe ser ED y el de mínimo, ES."""
    cube, seg = _make_phantom(ef_target=0.55, seed=2)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    vols = res.gate_volumes_ml
    assert res.ed_gate == int(np.argmax(vols)) + 1
    assert res.es_gate == int(np.argmin(vols)) + 1
    assert res.ed_gate == 1, f"el fantoma tiene ED en el gate 1, dio {res.ed_gate}"
    print(f"[OK] ED gate={res.ed_gate}, ES gate={res.es_gate}")


def test_espesor_ed_escala_los_volumenes():
    """Asumir una pared más gruesa desplaza el endocardio y achica la cavidad."""
    cube, seg = _make_phantom(seed=3)
    fino = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                           ECTbLVConfig(ed_wall_thickness_mm=8.0))
    grueso = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                             ECTbLVConfig(ed_wall_thickness_mm=14.0))

    assert grueso.edv_ml < fino.edv_ml, "más espesor debe achicar la cavidad"
    assert grueso.myocardial_mass_g > fino.myocardial_mass_g
    print(f"[OK] espesor ED escala volúmenes: 8mm->EDV {fino.edv_ml:.1f} mL | "
          f"14mm->EDV {grueso.edv_ml:.1f} mL")


def test_masa_miocardica_usa_densidad():
    """La masa es el volumen de pared por 1.05 g/mL."""
    cube, seg = _make_phantom(seed=4)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    assert res.myocardial_volume_ml > 0.0
    assert np.isclose(res.myocardial_mass_g, res.myocardial_volume_ml * 1.05)
    print(f"[OK] masa miocárdica: {res.myocardial_mass_g:.1f} g "
          f"(volumen {res.myocardial_volume_ml:.1f} mL x 1.05)")


def test_engrosamiento_positivo():
    """En un ventrículo que se contrae la pared engrosa en sístole."""
    cube, seg = _make_phantom(ef_target=0.55, seed=5)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    assert res.thickening_pct > 0.0, f"engrosamiento no positivo: {res.thickening_pct:.1f}%"
    print(f"[OK] engrosamiento sistólico: {res.thickening_pct:.1f}%")


def test_espesor_ed_es_mm_consistente_con_engrosamiento():
    """wall_thickness_ed_mm/es_mm deben ser positivos y derivar el mismo % informado."""
    cube, seg = _make_phantom(ef_target=0.55, seed=5)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    assert res.wall_thickness_ed_mm > 0.0
    assert res.wall_thickness_es_mm > 0.0
    esperado = (res.wall_thickness_es_mm - res.wall_thickness_ed_mm) / res.wall_thickness_ed_mm * 100.0
    assert np.isclose(res.thickening_pct, esperado, atol=0.5), (
        f"engrosamiento {res.thickening_pct:.1f}% no coincide con mm "
        f"({res.wall_thickness_ed_mm:.1f}->{res.wall_thickness_es_mm:.1f})"
    )
    print(
        f"[OK] espesor ED={res.wall_thickness_ed_mm:.1f} mm -> ES={res.wall_thickness_es_mm:.1f} mm "
        f"consistente con engrosamiento {res.thickening_pct:.1f}%"
    )


def test_rechaza_entradas_invalidas():
    """Cubos no gatillados o segmentación incoherente devuelven available=False."""
    cube, seg = _make_phantom(seed=6)

    res = analyze_lv_ectb(cube[:2], seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)
    assert not res.available and res.reason

    res = analyze_lv_ectb(cube, seg, (0.0, 0.0), SLICE_MM)
    assert not res.available and "spacing" in res.reason.lower()
    print("[OK] entradas inválidas rechazadas con motivo explícito")


def test_regresion_qgs():
    """La regresión R1 publicada convierte FEVI de ECTb a escala QGS."""
    # y = 0.96x - 0.053 con x como fracción 0-1
    assert np.isclose(apply_regression(0.60, 0.96, -0.053), 0.5230)
    # ECTb corre ~7-8% por encima de QGS-8: la conversión debe bajar el valor.
    assert apply_regression(0.60, 0.96, -0.053) < 0.60
    print("[OK] regresión ECTb->QGS aplicada correctamente")


def test_convert_ef_pct_maneja_unidades():
    """convert_ef_pct recibe y devuelve %, sin importar la unidad de la ecuación."""
    # QGS-8 está publicada en fracción: 60% -> 0.96*0.60 - 0.053 = 0.523 -> 52.3%
    assert np.isclose(convert_ef_pct(60.0, "qgs8"), 52.3)
    # QGS-16 está publicada en porcentaje: 0.855*60 + 1.73 = 53.03%
    assert np.isclose(convert_ef_pct(60.0, "qgs16"), 53.03)
    # MUGA, en fracción: 1.22*0.60 - 0.072 = 0.660 -> 66.0%
    assert np.isclose(convert_ef_pct(60.0, "muga"), 66.0)
    print("[OK] conversión de unidades correcta en las tres regresiones")


def test_conversion_baja_respecto_de_qgs():
    """El sesgo conocido: ECTb informa más alto que QGS en el rango clínico."""
    for ef in (35.0, 45.0, 55.0, 65.0):
        assert convert_ef_pct(ef, "qgs8") < ef, f"a {ef}% QGS-8 no dio menor"
        assert convert_ef_pct(ef, "qgs16") < ef, f"a {ef}% QGS-16 no dio menor"
    print("[OK] la conversión a escala QGS baja el valor en todo el rango clínico")


def test_regresion_desconocida_falla_explicito():
    """Pedir una regresión inexistente tiene que romper con KeyError, no en silencio."""
    try:
        convert_ef_pct(55.0, "inexistente")
    except KeyError:
        print("[OK] regresión inexistente rechazada con KeyError")
        return
    raise AssertionError("debería haber fallado con KeyError")


def test_texto_de_ecuacion_declara_unidad():
    """El texto de la ecuación tiene que dejar clara la unidad de x."""
    assert "fracción" in regression_equation_text("qgs8")
    assert "porcentaje" in regression_equation_text("qgs16")
    # El signo del término independiente debe leerse correcto.
    assert "−" in regression_equation_text("qgs8")
    assert "+" in regression_equation_text("qgs16")
    print("[OK] las ecuaciones declaran su unidad y su signo")


def test_radios_permiten_reconstruir_la_cota():
    """La cota manual se apoya en endo/epi: su separación es el espesor usado."""
    cube, seg = _make_phantom(seed=7)
    cfg = ECTbLVConfig(ed_wall_thickness_mm=12.0, median_kernel_large=0, median_kernel_small=0)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM, cfg)

    assert res.available, res.reason
    gate = res.ed_gate - 1
    separacion = res.epi_radii_mm[gate] - res.endo_radii_mm[gate]
    # En el gate de referencia el espesor debe ser el configurado.
    assert np.allclose(separacion, 12.0, atol=0.6), (
        f"separación endo-epi {separacion.mean():.2f} mm, esperada 12.0 mm"
    )
    print(f"[OK] cota reconstruible: separación endo-epi = {separacion.mean():.2f} mm")


# ---------------------------------------------------------------------------
# Plano valvular de dos piezas
# ---------------------------------------------------------------------------

def test_plano_valvular_no_toca_la_mitad_lateral():
    """"Dos piezas" significa: perpendicular en TODO el lado lateral."""
    n_ang = 64
    angles = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    w, u_cut = _valve_plane_weights(
        n_valid=10, angles=angles, dz_mm=SLICE_MM,
        septal_angle_deg=SEPTAL_ANGLE_DEG, offset_mm=10.0, base_is_last=True,
    )
    # Lateral = donde cos(theta - 180) <= 0, o sea la mitad opuesta al septum.
    lateral = np.cos(angles - np.deg2rad(SEPTAL_ANGLE_DEG)) <= 0.0
    assert np.allclose(w[:, lateral], 1.0), "el plano recortó del lado lateral"
    assert np.isclose(u_cut[lateral].min(), u_cut[lateral].max()), "el lado lateral no quedó plano"
    # En el medio del septum tiene que recortar el offset pedido.
    i_sept = int(np.argmin(np.abs(angles - np.deg2rad(SEPTAL_ANGLE_DEG))))
    recorte_cortes = u_cut[lateral][0] - u_cut[i_sept]
    assert np.isclose(recorte_cortes, 10.0 / SLICE_MM, atol=1e-6)
    print(f"[OK] plano de 2 piezas: lateral intacto, septum recortado {recorte_cortes:.2f} cortes")


def test_plano_valvular_con_offset_cero_no_cambia_nada():
    """Con offset 0 los pesos deben dar 1: idem plano perpendicular de siempre."""
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    w, _ = _valve_plane_weights(8, angles, SLICE_MM, SEPTAL_ANGLE_DEG, 0.0, True)
    assert np.allclose(w, 1.0)

    cube, seg = _make_phantom(seed=8)
    sin_plano = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                                ECTbLVConfig(use_valve_plane=False))
    offset_cero = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                                  ECTbLVConfig(valve_septal_offset_mm=0.0))
    assert np.isclose(sin_plano.edv_ml, offset_cero.edv_ml)
    assert np.isclose(sin_plano.ef_pct, offset_cero.ef_pct)
    print("[OK] desactivar el plano equivale a offset 0 mm")


def test_plano_valvular_descuenta_volumen_basal():
    """El plano tiene que sacar cavidad de la base, no agregarla."""
    cube, seg = _make_phantom(seed=9)
    plano = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                            ECTbLVConfig(use_valve_plane=False))
    ectb = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    assert ectb.edv_ml < plano.edv_ml, "el plano valvular no redujo el EDV"
    assert ectb.esv_ml < plano.esv_ml
    assert ectb.valve_removed_ml > 0.0
    assert np.isclose(ectb.valve_removed_ml, plano.edv_ml - ectb.edv_ml, atol=0.5)
    assert any("valvular" in n.lower() for n in ectb.notes), "falta la nota de auditoría"
    print(f"[OK] plano valvular descontó {ectb.valve_removed_ml:.1f} mL del EDV "
          f"({plano.edv_ml:.1f} -> {ectb.edv_ml:.1f} mL)")


def test_plano_valvular_es_continuo():
    """Mover el offset no debe producir saltos de un corte entero.

    Es el requisito para que el control se pueda mover en vivo sin que el
    número pegue escalones.
    """
    cube, seg = _make_phantom(seed=10)
    offsets = np.arange(0.0, 13.0, 0.5)
    edvs = [
        analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                        ECTbLVConfig(valve_septal_offset_mm=float(o))).edv_ml
        for o in offsets
    ]
    edvs = np.asarray(edvs)
    saltos = np.abs(np.diff(edvs))
    assert np.all(np.diff(edvs) <= 1e-9), "el EDV no baja de forma monótona"
    assert saltos.max() < 0.03 * edvs[0], (
        f"salto de {saltos.max():.1f} mL entre pasos de 0.5 mm: no es continuo"
    )
    print(f"[OK] respuesta continua: salto máximo {saltos.max():.2f} mL por 0.5 mm")


def test_detecta_de_que_lado_esta_la_base():
    """El plano necesita saber hacia dónde queda la base, venga como venga la pila."""
    n_slices, size = 8, 24
    mask = np.zeros((n_slices, size, size), dtype=bool)
    ys, xs = np.ogrid[:size, :size]
    dist = np.sqrt((ys - size / 2) ** 2 + (xs - size / 2) ** 2)
    # Pila ápex(chico) -> base(grande).
    for s in range(n_slices):
        mask[s] = dist <= (3.0 + s)
    inner = np.linspace(0.5, 6.0, n_slices)
    outer = np.linspace(4.0, 10.0, n_slices)
    centers = np.tile([size / 2, size / 2], (n_slices, 1))

    seg_ab = _FakeSeg(mask, centers, outer, inner)
    assert _base_is_last(seg_ab, list(range(n_slices))) is True

    seg_ba = _FakeSeg(mask[::-1].copy(), centers, outer[::-1].copy(), inner[::-1].copy())
    assert _base_is_last(seg_ba, list(range(n_slices))) is False
    print("[OK] la base se detecta por geometría en ambos sentidos de la pila")


def test_indice_esfericidad_entre_0_y_1_y_menor_en_sistole():
    """El índice tiene que ser positivo y bajar en sístole.

    El eje largo lo fija la extensión de la pila de cortes, que no cambia entre
    gates, mientras que el eje corto se achica en sístole. Por lo tanto el
    índice en ES tiene que ser menor que en ED.
    """
    cube, seg = _make_phantom(ef_target=0.55, seed=5)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    assert res.available, res.reason
    assert res.long_axis_mm > 0.0
    assert 0.0 < res.shape_index_ed < 3.0, res.shape_index_ed
    assert res.shape_index_es < res.shape_index_ed, (res.shape_index_ed, res.shape_index_es)
    assert res.short_axis_es_mm < res.short_axis_ed_mm
    print(f"[OK] esfericidad ED={res.shape_index_ed:.2f} > ES={res.shape_index_es:.2f} "
          f"(eje largo {res.long_axis_mm:.0f} mm)")


def test_ventriculo_mas_corto_es_mas_esferico():
    """Con menos cortes (ventrículo más corto) el índice tiene que subir.

    Es la propiedad que le da sentido clínico: a igual diámetro de eje corto,
    un ventrículo más corto en el eje largo es geométricamente más esférico.
    """
    cube_largo, seg_largo = _make_phantom(n_slices=20, seed=6)
    cube_corto, seg_corto = _make_phantom(n_slices=8, seed=6)

    si_largo = analyze_lv_ectb(cube_largo, seg_largo, (PIXEL_MM, PIXEL_MM), SLICE_MM).shape_index_ed
    si_corto = analyze_lv_ectb(cube_corto, seg_corto, (PIXEL_MM, PIXEL_MM), SLICE_MM).shape_index_ed

    assert si_corto > si_largo, (si_largo, si_corto)
    print(f"[OK] ventriculo corto mas esferico: {si_corto:.2f} vs largo {si_largo:.2f}")


if __name__ == "__main__":
    test_recupera_fevi_conocida()
    test_detecta_ventriculo_deprimido()
    test_ed_gate_y_es_gate_coherentes()
    test_espesor_ed_escala_los_volumenes()
    test_masa_miocardica_usa_densidad()
    test_engrosamiento_positivo()
    test_espesor_ed_es_mm_consistente_con_engrosamiento()
    test_rechaza_entradas_invalidas()
    test_regresion_qgs()
    test_convert_ef_pct_maneja_unidades()
    test_conversion_baja_respecto_de_qgs()
    test_regresion_desconocida_falla_explicito()
    test_texto_de_ecuacion_declara_unidad()
    test_radios_permiten_reconstruir_la_cota()
    test_plano_valvular_no_toca_la_mitad_lateral()
    test_plano_valvular_con_offset_cero_no_cambia_nada()
    test_plano_valvular_descuenta_volumen_basal()
    test_plano_valvular_es_continuo()
    test_detecta_de_que_lado_esta_la_base()
    test_indice_esfericidad_entre_0_y_1_y_menor_en_sistole()
    test_ventriculo_mas_corto_es_mas_esferico()
    print("\n[TODOS LOS TESTS DE ECTb LV PASARON]")
