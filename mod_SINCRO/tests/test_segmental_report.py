"""Tests de core.segmental_report (tabla fase + perfusión + viabilidad)."""
import numpy as np

from core.segmental_report import (
    build_segmental_report,
    build_segmental_rows,
    latest_activation_segment,
    perfusion_percent_by_segment,
    territory_summary,
)


def _phase_all_segments(base=30.0, late_seg=5, late_val=120.0):
    p = {s: base for s in range(1, 18)}
    p[late_seg] = late_val
    return p


def _segmap_and_perfusion():
    """Mapa de segmentos 1..17 en una grilla + perfusión decreciente por segmento."""
    seg = np.zeros((17, 4), dtype=np.int32)
    perf = np.zeros((17, 4), dtype=np.float64)
    for i in range(17):
        seg[i, :] = i + 1
        perf[i, :] = 100.0 - i * 4.0  # seg 1 el más captante, cae hasta ~36% en seg 17
    return perf, seg


def test_perfusion_percent_normaliza_al_maximo():
    perf, seg = _segmap_and_perfusion()
    pct = perfusion_percent_by_segment(perf, seg)
    assert np.isclose(pct[1], 100.0)  # segmento más captante = 100%
    assert pct[17] < pct[1]
    assert all(0.0 < v <= 100.0 for v in pct.values())


def test_build_rows_17_filas_con_viabilidad():
    perf, seg = _segmap_and_perfusion()
    phase = _phase_all_segments()
    rows = build_segmental_rows(phase, perfusion=perf, segment_map=seg)
    assert len(rows) == 17
    for r in rows:
        assert set(r.keys()) >= {"segment", "name", "territory", "phase_deg", "perfusion_pct", "viability"}
        assert r["territory"] in ("LAD", "LCx", "RCA")
    # seg 1 (100%) es viable; algún segmento bajo cae a dudosa/no viable
    seg1 = next(r for r in rows if r["segment"] == 1)
    assert seg1["viability"] == "viable"
    clases = {r["viability"] for r in rows}
    assert "dudosa" in clases or "no viable" in clases


def test_umbrales_viabilidad_parametrizables():
    perf, seg = _segmap_and_perfusion()
    phase = _phase_all_segments()
    # Con umbral viable muy alto, hasta el segmento tope deja de ser 'viable'
    rows = build_segmental_rows(phase, perfusion=perf, segment_map=seg, viable_pct=101.0, dudosa_pct=90.0)
    seg1 = next(r for r in rows if r["segment"] == 1)
    assert seg1["viability"] in ("dudosa", "no viable")


def test_sin_perfusion_solo_fase():
    phase = _phase_all_segments()
    rows = build_segmental_rows(phase)
    assert len(rows) == 17
    assert all(not np.isfinite(r["perfusion_pct"]) for r in rows)
    assert all(r["viability"] == "N/D" for r in rows)


def test_latest_activation_segment():
    phase = _phase_all_segments(base=30.0, late_seg=8, late_val=200.0)
    assert latest_activation_segment(phase) == 8
    assert latest_activation_segment({}) is None
    assert latest_activation_segment({1: np.nan, 2: np.nan}) is None


def test_territory_summary_incluye_perfusion():
    perf, seg = _segmap_and_perfusion()
    phase = _phase_all_segments()
    pct = perfusion_percent_by_segment(perf, seg)
    summ = territory_summary(phase, pct)
    for terr in ("LAD", "LCx", "RCA"):
        assert terr in summ
        assert "mean" in summ[terr] and "perfusion_pct" in summ[terr]
        assert np.isfinite(summ[terr]["perfusion_pct"])


def test_reporte_completo():
    perf, seg = _segmap_and_perfusion()
    phase = _phase_all_segments(late_seg=12, late_val=150.0)
    rep = build_segmental_report(phase, perf, seg)
    assert rep["has_perfusion"] is True
    assert len(rep["rows"]) == 17
    assert rep["latest_segment"] == 12
    assert set(rep["territories"].keys()) == {"LAD", "LCx", "RCA"}


def test_shape_mismatch_levanta():
    perf = np.zeros((17, 4))
    seg = np.zeros((17, 5), dtype=np.int32)
    try:
        perfusion_percent_by_segment(perf, seg)
    except ValueError:
        return
    raise AssertionError("esperaba ValueError por shapes distintas")
