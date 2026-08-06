"""Tests del resolvedor de orientacion (core/orientation_resolver.py).

Valores de referencia tomados de la auditoria real de 877 DICOMs (43 grupos):
  - GE cardiaco crudo (start 225, CW): IOP detector [0.707,-0.707,0, 0,0,-1].
  - Millennium dual-head crudo: IOP [1,0,0, 0,0,-1].
  - Recon transaxial cardiaco: IOP [1,0,0, 0,1,0] (col horizontal -> flip por IOP indefinido).
"""
import numpy as np
import pytest

from core.orientation_resolver import (
    parse_iop,
    read_detector_iop,
    resolve_orientation,
    select_profile,
)

SQRT2 = float(np.sqrt(0.5))


def test_parse_iop_normaliza_y_calcula_normal():
    parsed = parse_iop([2.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    assert parsed is not None
    row, col, normal = parsed
    assert np.allclose(row, [1, 0, 0])
    assert np.allclose(col, [0, 1, 0])
    assert np.allclose(normal, [0, 0, 1])


def test_parse_iop_invalido_devuelve_none():
    assert parse_iop(None) is None
    assert parse_iop([1, 0, 0]) is None
    assert parse_iop([0, 0, 0, 0, 0, 0]) is None


def test_flip_z_ge_cardiaco_crudo():
    # col=(0,0,-1): z crece hacia pies -> z=0 ya es cabeza -> NO voltear.
    res = resolve_orientation(
        manufacturer="GE MEDICAL SYSTEMS", start_angle=225.0, rotation_direction="CW",
        detector_iop=[SQRT2, -SQRT2, 0.0, 0.0, 0.0, -1.0],
    )
    assert res.flip_z is False
    assert res.source == "iop"
    assert res.profile_key == "ge_cardiac"


def test_flip_z_millennium_crudo():
    res = resolve_orientation(
        manufacturer="GE MEDICAL SYSTEMS", start_angle=123.7, rotation_direction="CW",
        detector_iop=[1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    )
    assert res.flip_z is False
    assert res.source == "iop"


def test_flip_z_col_horizontal_cae_a_patient_position():
    # col=(0,1,0): componente Z ~0 -> IOP indefinido -> usa PatientPosition.
    res = resolve_orientation(
        manufacturer="GE MEDICAL SYSTEMS", patient_position="FFS",
        start_angle=225.0, rotation_direction="CW",
        detector_iop=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    assert res.flip_z is True
    assert res.source == "patient_position"


def test_flip_z_patient_position_sin_iop():
    hfs = resolve_orientation(manufacturer="GE", patient_position="HFS", start_angle=0.0)
    ffs = resolve_orientation(manufacturer="GE", patient_position="FFS", start_angle=0.0)
    assert hfs.flip_z is False and hfs.source == "patient_position"
    assert ffs.flip_z is True and ffs.source == "patient_position"


def test_flip_z_sin_metadatos_asume_false():
    res = resolve_orientation(manufacturer="Desconocido", start_angle=0.0)
    assert res.flip_z is False
    assert res.source == "profile"


def test_angulos_ge_cardiaco_cw():
    # start=225, CW -> sign=-1 -> anterior=225-45=180, lateral=225-135=90.
    res = resolve_orientation(
        manufacturer="GE MEDICAL SYSTEMS", start_angle=225.0, rotation_direction="CW",
    )
    assert res.anterior_angle_deg == pytest.approx(180.0)
    assert res.left_lateral_angle_deg == pytest.approx(90.0)


def test_angulos_cc_suma_offset():
    res = resolve_orientation(manufacturer="Picker", start_angle=0.0, rotation_direction="CC")
    assert res.anterior_angle_deg == pytest.approx(45.0)
    assert res.left_lateral_angle_deg == pytest.approx(135.0)


def test_sin_start_angle_sin_angulos():
    res = resolve_orientation(manufacturer="GE", patient_position="HFS")
    assert res.anterior_angle_deg is None
    assert res.left_lateral_angle_deg is None


def test_perfil_gvi_no_calibrado():
    prof = select_profile("GVI", "OnePass")
    assert prof.key == "gvi_onepass"
    assert prof.calibrated is False
    res = resolve_orientation(manufacturer="GVI")
    assert res.profile_key == "gvi_onepass"
    assert res.calibrated is False


def test_perfil_fallback_generico():
    prof = select_profile("FabricanteRaro", "ModeloX")
    assert prof.key == "generic"


def test_marconi_ap_espejado_calibrado():
    # Marconi start=-132 CW -> anterior=-132-45=-177%360=183, lateral=-132-135=93.
    res = resolve_orientation(
        manufacturer="Marconi", start_angle=-132.0, rotation_direction="CW",
        detector_iop=[1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    )
    assert res.profile_key == "marconi_picker"
    assert res.anterior_angle_deg == pytest.approx(183.0)
    assert res.left_lateral_angle_deg == pytest.approx(93.0)
    assert res.mirror_ap_lr is True
    assert res.mirror_ll_lr is False
    assert res.calibrated is True


class _FakeElem:
    def __init__(self, value):
        self.value = value


class _FakeDataset:
    def __init__(self, mapping):
        self._m = mapping

    def __contains__(self, key):
        return key in self._m

    def __getitem__(self, key):
        return _FakeElem(self._m[key])


def test_read_detector_iop_desde_secuencia():
    iop = [0.707, -0.707, 0.0, 0.0, 0.0, -1.0]
    inner = _FakeDataset({(0x0020, 0x0037): iop})
    ds = _FakeDataset({(0x0054, 0x0022): [inner]})
    assert read_detector_iop(ds) == iop


def test_read_detector_iop_ausente():
    ds = _FakeDataset({})
    assert read_detector_iop(ds) is None
