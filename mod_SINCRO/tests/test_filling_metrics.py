"""Tests de core.filling_metrics (PFR / TVmáx)."""
import numpy as np

from core.filling_metrics import (
    compute_filling_metrics,
    format_pfr,
    format_tvmax,
)


def _synthetic_volume_curve(n=16):
    """Curva de volumen sintética: sístole rápida, llenado rápido temprano + diástasis.

    ED (máx) en gate 1, ES (mín) alrededor de 1/3 del ciclo, luego llenado.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Volumen tipo: cae a ES y vuelve a subir; asimétrico para que el pico de
    # llenado sea nítido.
    v = 50.0 + 30.0 * np.cos(t) ** 3
    return v


def test_pfr_tvmax_disponibles_con_rr():
    v = _synthetic_volume_curve(16)
    edv = float(v.max())
    es_gate = int(np.argmin(v)) + 1
    fm = compute_filling_metrics(v, edv, es_gate, rr_ms=800.0)
    assert fm["available"] is True
    assert fm["pfr_edv_per_rr"] is not None and fm["pfr_edv_per_rr"] > 0
    assert fm["pfr_edv_per_s"] is not None and fm["pfr_edv_per_s"] > 0
    assert fm["tpfr_pct_rr"] is not None and 0.0 <= fm["tpfr_pct_rr"] < 100.0
    assert fm["tpfr_ms"] is not None and fm["tpfr_ms"] >= 0.0
    assert fm["rr_ms_used"] == 800.0


def test_sin_rr_solo_normalizacion_geometrica():
    v = _synthetic_volume_curve(16)
    edv = float(v.max())
    es_gate = int(np.argmin(v)) + 1
    fm = compute_filling_metrics(v, edv, es_gate, rr_ms=None)
    assert fm["available"] is True
    # VTD/RR y %RR NO dependen del RR real
    assert fm["pfr_edv_per_rr"] is not None
    assert fm["tpfr_pct_rr"] is not None
    # Unidades absolutas quedan sin definir
    assert fm["pfr_edv_per_s"] is None
    assert fm["tpfr_ms"] is None
    assert fm["rr_ms_used"] is None


def test_relacion_por_segundo_vs_por_rr():
    v = _synthetic_volume_curve(20)
    edv = float(v.max())
    es_gate = int(np.argmin(v)) + 1
    rr = 1000.0  # 1 s → per_s == per_rr
    fm = compute_filling_metrics(v, edv, es_gate, rr_ms=rr)
    assert np.isclose(fm["pfr_edv_per_s"], fm["pfr_edv_per_rr"], rtol=1e-6)
    # con RR de 500 ms, per_s duplica per_rr
    fm2 = compute_filling_metrics(v, edv, es_gate, rr_ms=500.0)
    assert np.isclose(fm2["pfr_edv_per_s"], 2.0 * fm2["pfr_edv_per_rr"], rtol=1e-6)


def test_rr_no_fisiologico_se_ignora():
    v = _synthetic_volume_curve(16)
    edv = float(v.max())
    es_gate = int(np.argmin(v)) + 1
    fm = compute_filling_metrics(v, edv, es_gate, rr_ms=1.0)  # placeholder GE
    assert fm["available"] is True
    assert fm["pfr_edv_per_s"] is None
    assert fm["rr_ms_used"] is None


def test_curva_invalida():
    assert compute_filling_metrics([1.0, 2.0], 2.0, 1, 800.0)["available"] is False
    assert compute_filling_metrics([1.0, 2.0, 3.0], 0.0, 1, 800.0)["available"] is False
    assert compute_filling_metrics([np.nan, 1.0, 2.0], 2.0, 1, 800.0)["available"] is False


def test_formato_ectb():
    fm = compute_filling_metrics(_synthetic_volume_curve(16), 80.0, 6, rr_ms=800.0)
    pfr_txt = format_pfr(fm)
    tv_txt = format_tvmax(fm)
    assert "VTD/s" in pfr_txt and "VTD/RR" in pfr_txt
    assert "ms" in tv_txt and "%RR" in tv_txt
    # sin RR: solo la normalización geométrica
    fm2 = compute_filling_metrics(_synthetic_volume_curve(16), 80.0, 6, rr_ms=None)
    assert format_pfr(fm2).endswith("VTD/RR")
    assert format_tvmax(fm2).endswith("%RR")
