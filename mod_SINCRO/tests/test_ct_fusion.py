"""Tests de core.ct_fusion (fachada neutra CT/fusión + conversión HU→μ)."""

import numpy as np
import pytest

from core.ct_fusion import (
    MU_WATER_140KEV_CM,
    mu_map_from_ct_hu,
    validate_mu_map,
)


def test_mu_map_aire_es_cero():
    mu, _ = mu_map_from_ct_hu(np.full((2, 4, 4), -1000.0))
    assert np.allclose(mu, 0.0)


def test_mu_map_agua_es_mu_water():
    mu, _ = mu_map_from_ct_hu(np.zeros((2, 4, 4)))
    assert np.allclose(mu, MU_WATER_140KEV_CM)


def test_mu_map_hueso_supera_agua_y_es_lineal():
    hu = np.array([[[500.0, 1000.0]]])
    mu, _ = mu_map_from_ct_hu(hu)
    assert mu[0, 0, 1] > mu[0, 0, 0] > MU_WATER_140KEV_CM
    # a 1000 HU debe dar exactamente mu_bone default
    assert mu[0, 0, 1] == pytest.approx(0.250, abs=1e-9)


def test_mu_map_clip_negativo():
    mu, _ = mu_map_from_ct_hu(np.full((1, 2, 2), -2000.0))
    assert np.allclose(mu, 0.0)


def test_mu_map_pulmon_intermedio():
    # Pulmón ~ -700 HU: μ entre aire y agua.
    mu, _ = mu_map_from_ct_hu(np.full((1, 2, 2), -700.0))
    assert 0.0 < float(mu[0, 0, 0]) < MU_WATER_140KEV_CM


def test_validate_mu_map_detecta_vacio():
    ok, notes = validate_mu_map(np.zeros((4, 8, 8)))
    assert not ok
    assert any("0" in n for n in notes)


def test_validate_mu_map_detecta_2d():
    ok, _ = validate_mu_map(np.zeros((8, 8)))
    assert not ok


def test_validate_mu_map_acepta_valido():
    mu, _ = mu_map_from_ct_hu(np.zeros((4, 8, 8)))
    ok, notes = validate_mu_map(mu)
    assert ok
    assert notes and "OK" in notes[0]


def test_facade_reexporta_nucleo_amylo():
    # La fachada debe exponer el núcleo maduro sin nombres 'amyloid'.
    from core import ct_fusion

    for name in (
        "load_ct_volume_from_path",
        "load_attenuation_map_from_path",
        "list_ct_series_in_path",
        "resample_volume_to_spect_grid",
        "register_ct_to_spect_rigid",
        "align_ct_orientation_to_spect",
        "refine_ct_to_spect_translation",
        "refine_ct_to_spect_rotation",
        "apply_attenuation_correction_chang",
        "apply_attenuation_correction_prototype",
        "remove_ct_table",
        "central_slices_preview",
        "CTVolumeResult",
        "AttenuationMapResult",
    ):
        assert hasattr(ct_fusion, name), f"falta {name} en ct_fusion"


def test_resample_passthrough_mismo_shape():
    from core.ct_fusion import resample_volume_to_spect_grid

    vol = np.random.default_rng(1).normal(size=(4, 8, 8))
    out, notes = resample_volume_to_spect_grid(vol, np.zeros((4, 8, 8)))
    assert out.shape == (4, 8, 8)
    assert np.allclose(out, vol)
    assert notes


def _thorax_phantom():
    ct = np.full((32, 40, 40), -1000.0)
    ct[2:30, 4:36, 4:36] = 40.0        # cuerpo tejido blando
    ct[6:15, 10:28, 6:16] = -800.0     # pulmón derecho
    ct[6:15, 10:28, 24:34] = -800.0    # pulmón izquierdo
    ct[8:13, 14:24, 16:24] = 50.0      # corazón entre pulmones
    return ct


def test_lung_mask_excluye_aire_exterior():
    from core.ct_fusion import lung_mask_from_ct_hu

    lungs = lung_mask_from_ct_hu(_thorax_phantom())
    assert lungs.any()
    assert not lungs[0].any()


def test_subdiaphragmatic_enmascara_higado_no_corazon():
    from core.ct_fusion import subdiaphragmatic_mask_from_ct

    ct = _thorax_phantom()
    mask, notes = subdiaphragmatic_mask_from_ct(ct)
    liver = np.zeros_like(mask)
    liver[16:30, 4:36, 4:36] = True
    liver_soft = liver & (ct >= -100) & (ct <= 200)
    heart = np.zeros_like(mask)
    heart[8:13, 14:24, 16:24] = True
    assert mask[liver_soft].mean() > 0.8
    assert mask[heart].mean() < 0.05
    assert notes


def test_subdiaphragmatic_sin_pulmones_devuelve_vacio():
    from core.ct_fusion import subdiaphragmatic_mask_from_ct

    ct = np.full((8, 16, 16), 40.0)
    mask, notes = subdiaphragmatic_mask_from_ct(ct)
    assert not mask.any()
    assert any("Sin pulmones" in n for n in notes)
