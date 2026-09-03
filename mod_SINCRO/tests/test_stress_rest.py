"""Tests para core.stress_rest (comparación de fase esfuerzo vs reposo)."""
from __future__ import annotations

import math

import numpy as np

from core.stress_rest import (
    circular_delta_deg,
    compare_stress_rest,
    compare_territories,
    transient_ischemic_dilation,
    TID_GATED_SOFT_CUTOFF,
)


def _metrics(phase_sd, bandwidth, entropy_bits, entropy_pct, ai, mean_phase):
    return {
        "phase_sd": phase_sd,
        "bandwidth": bandwidth,
        "entropy_shannon_bits": entropy_bits,
        "entropy_normalized_pct": entropy_pct,
        "asynchrony_index": ai,
        "skewness": 0.1,
        "kurtosis": 0.2,
        "peak_width": 10.0,
        "mean_phase": mean_phase,
        "peak_phase": mean_phase,
        "latest_activation_phase": (mean_phase + 30.0) % 360.0,
    }


def test_circular_delta_wraps():
    assert circular_delta_deg(10.0, 350.0) == 20.0
    assert circular_delta_deg(350.0, 10.0) == -20.0
    # 180 y -180 son el mismo ángulo (borde del wrap): aceptar cualquiera.
    assert abs(circular_delta_deg(0.0, 180.0)) == 180.0
    assert math.isnan(circular_delta_deg(float("nan"), 10.0))


def test_scalar_deltas_signo_esfuerzo_menos_reposo():
    stress = _metrics(45.0, 80.0, 4.5, 70.0, 12.0, 100.0)
    rest = _metrics(35.0, 60.0, 4.0, 62.0, 8.0, 100.0)
    out = compare_stress_rest(stress, rest)
    assert out["available"] is True
    d = out["deltas"]
    assert d["phase_sd"] == 10.0
    assert d["bandwidth"] == 20.0
    assert abs(d["entropy_normalized_pct"] - 8.0) < 1e-9
    assert d["asynchrony_index"] == 4.0


def test_angular_delta_es_circular():
    stress = _metrics(30.0, 50.0, 4.0, 60.0, 5.0, 10.0)
    rest = _metrics(30.0, 50.0, 4.0, 60.0, 5.0, 350.0)
    out = compare_stress_rest(stress, rest)
    assert out["deltas"]["mean_phase"] == 20.0


def test_faltan_metricas_devuelve_no_disponible():
    assert compare_stress_rest({}, {"phase_sd": 30})["available"] is False
    assert compare_stress_rest({"phase_sd": 30}, {})["available"] is False


def test_notas_citan_papers_cuando_hay_inducible():
    stress = _metrics(45.0, 90.0, 4.6, 72.0, 14.0, 100.0)
    rest = _metrics(30.0, 60.0, 4.0, 60.0, 8.0, 100.0)
    out = compare_stress_rest(stress, rest)
    joined = " ".join(out["notes"])
    assert "40021521" in joined  # Fukumoto entropy
    assert "39948439" in joined  # Tanaka bandwidth
    assert "INDUCIBLE" in joined


def test_territorio_delta_circular_y_lineal():
    stress_t = {
        "LAD": {"mean": 10.0, "std": 15.0, "min": 0, "max": 20, "n": 7},
        "LCx": {"mean": 100.0, "std": 20.0, "min": 90, "max": 110, "n": 5},
        "RCA": {"mean": 200.0, "std": 25.0, "min": 190, "max": 210, "n": 5},
    }
    rest_t = {
        "LAD": {"mean": 350.0, "std": 10.0, "min": 340, "max": 360, "n": 7},
        "LCx": {"mean": 100.0, "std": 18.0, "min": 90, "max": 110, "n": 5},
        "RCA": {"mean": 200.0, "std": 20.0, "min": 190, "max": 210, "n": 5},
    }
    out = compare_territories(stress_t, rest_t)
    assert out["LAD"]["delta_mean_circular"] == 20.0  # 10 - 350 circular
    assert out["LAD"]["delta_std"] == 5.0
    assert out["LCx"]["delta_mean_circular"] == 0.0


def test_territorio_ausente_da_nan():
    out = compare_territories(None, None)
    for terr in ("LAD", "LCx", "RCA"):
        assert math.isnan(out[terr]["delta_mean_circular"])


def test_tid_ratio_esfuerzo_sobre_reposo():
    out = transient_ischemic_dilation(130.0, 100.0)
    assert out["available"] is True
    assert abs(out["ratio"] - 1.30) < 1e-9
    assert out["stress_edv_ml"] == 130.0
    assert out["rest_edv_ml"] == 100.0
    assert out["elevated"] is True  # 1.30 >= cutoff


def test_tid_no_elevado_bajo_cutoff():
    out = transient_ischemic_dilation(100.0, 100.0)
    assert out["available"] is True
    assert out["elevated"] is False
    assert out["soft_cutoff"] == TID_GATED_SOFT_CUTOFF


def test_tid_edv_invalido_no_disponible():
    assert transient_ischemic_dilation(None, 100.0)["available"] is False
    assert transient_ischemic_dilation(100.0, 0.0)["available"] is False
    assert transient_ischemic_dilation(float("nan"), 100.0)["available"] is False


def test_tid_incluido_en_compare_stress_rest():
    stress = _metrics(45.0, 80.0, 4.5, 70.0, 12.0, 100.0)
    rest = _metrics(35.0, 60.0, 4.0, 62.0, 8.0, 100.0)
    out = compare_stress_rest(
        stress, rest,
        stress_ef={"edv_ml": 132.0}, rest_ef={"edv_ml": 110.0},
    )
    assert out["tid"]["available"] is True
    assert abs(out["tid"]["ratio"] - 1.2) < 1e-9
