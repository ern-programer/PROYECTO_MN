"""Tests del RR multi-fabricante: tabla de colimadores, PSF y OSEM+RR."""
from __future__ import annotations

import numpy as np

from core.collimator_specs import CollimatorSpec, lookup_collimator
from core.resolution_recovery import PsfModel, correct_axial_magnification
from core.raw_reconstruction import reconstruct_projection_volume


def test_lookup_collimator_by_manufacturer_and_name():
    spec = lookup_collimator("GE MEDICAL SYSTEMS", "LEHR", "PARA")
    assert spec.manufacturer == "GE"
    assert spec.geometry == "parallel"


def test_lookup_collimator_alias_starcam():
    spec = lookup_collimator("GE MEDICAL SYSTEMS, NUCLEAR", "99", "PARA")
    assert spec.name == "STARCAM-GP"


def test_lookup_collimator_fanbeam_gvi():
    spec = lookup_collimator("GVI", "NGSPECT", "FANB")
    assert spec.geometry == "fanbeam"
    assert spec.axial_magnification is None  # pendiente de datasheet


def test_lookup_collimator_fallback():
    spec = lookup_collimator("MARCA_DESCONOCIDA", "XYZ", "")
    assert isinstance(spec, CollimatorSpec)
    assert spec.geometry == "parallel"


def test_psf_sigma_increases_with_distance():
    spec = lookup_collimator("GE", "LEHR")
    psf = PsfModel.from_collimator(spec, radius_mm=250.0, pixel_mm=6.4)
    sig = psf.sigma_px_for_rows(64)
    # La fila del lado detector (última) debe ser MENOS borrosa que el fondo (primera).
    assert sig[-1] < sig[0]
    assert np.all(sig >= 0)


def test_correct_axial_magnification_identity():
    proj = np.random.default_rng(0).random((8, 64, 64))
    out = correct_axial_magnification(proj, 1.0)
    assert out.shape == proj.shape
    assert np.allclose(out, proj)


def test_osem_rr_runs_and_matches_shape():
    rng = np.random.default_rng(1)
    proj = rng.poisson(20.0, size=(24, 8, 32)).astype(np.float64)  # (ang,H,W)
    angles = np.linspace(0.0, 180.0, 24, endpoint=False)
    spec = lookup_collimator("GE", "LEHR")
    psf = PsfModel.from_collimator(spec, radius_mm=200.0, pixel_mm=6.4)
    vol_plain = reconstruct_projection_volume(proj, angles, method="osem", iterations=2, subsets=3)
    vol_rr = reconstruct_projection_volume(proj, angles, method="osem", iterations=2, subsets=3, psf=psf)
    assert vol_plain.shape == vol_rr.shape == (8, 32, 32)
    assert np.all(np.isfinite(vol_rr))
    # El RR cambia el resultado respecto del OSEM plano.
    assert not np.allclose(vol_plain, vol_rr)
