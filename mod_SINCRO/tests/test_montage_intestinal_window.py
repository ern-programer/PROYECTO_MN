"""Ventana LCARS de atenuación intestinal del montaje: modelo de ROI, ventaneo y equivalencia con el motor."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from ui.montage_intestinal_window import IntestinalRoiModel, montage_display_norm

ROOT = Path(__file__).resolve().parents[1]
_CINE = next(node for node in ast.parse((ROOT / "ui" / "cine_widget.py").read_text(encoding="utf-8")).body
             if isinstance(node, ast.ClassDef) and node.name == "CineWidget")
_METHODS = {node.name: node for node in _CINE.body if isinstance(node, ast.FunctionDef)}

TRI_A = [[1.0, 1.0], [1.0, 5.0], [5.0, 3.0]]
TRI_B = [[2.0, 2.0], [2.0, 7.0], [7.0, 4.0]]
TRI_C = [[3.0, 3.0], [3.0, 9.0], [9.0, 6.0]]


def _engine(state, scope):
    """Motor real del visor (CineWidget) sin Qt: carga estado y resuelve polígonos por gate/corte."""
    engine = SimpleNamespace(
        _intestinal_scope_mode=scope, current_gate_index=lambda: 0, current_slice_index=lambda: 0,
        preview=SimpleNamespace(set_exclusion_polygon=lambda *_: None, set_reference_polygons=lambda *_: None),
        _refresh_intestinal_ref_label=lambda: None, _update_view=lambda: None,
        set_intestinal_mode=lambda *_: None, set_intestinal_background_method=lambda *_: None,
    )
    for name in ("set_intestinal_roi_state", "_intestinal_polygon_for_slice", "_intestinal_ref_polygons_for_slice"):
        namespace = {}
        exec(compile(ast.Module(body=[_METHODS[name]], type_ignores=[]), "cine_widget", "exec"), namespace)
        setattr(engine, name, MethodType(namespace[name], engine))
    engine.set_intestinal_roi_state(state)
    return engine


def _resolved(engine, n_gates, n_slices):
    return {(g, s): ([tuple(p) for p in engine._intestinal_polygon_for_slice(s, gate_index=g)],
                     [[tuple(p) for p in poly] for poly in engine._intestinal_ref_polygons_for_slice(s, gate_index=g)])
            for g in range(n_gates) for s in range(n_slices)}


MIXED_STATE = {
    "slice_polygons": [{"slice": 1, "points": TRI_A}, {"slice": 3, "points": TRI_B}],
    "gate_polygons": [{"gate": 0, "slice": 2, "points": TRI_C}, {"gate": 2, "slice": 1, "points": TRI_B}],
    "reference_slice_polygons": [{"slice": 1, "points": TRI_B}],
    "reference_gate_polygons": [{"gate": 1, "slice": 3, "points": TRI_A}],
    "mode": "subtract", "background_method": "mean",
}


@pytest.mark.parametrize("scope", ["slice", "all_slices", "gate_slices", "gate_override"])
def test_model_reproduces_engine_effect_for_every_scope(scope):
    n_gates, n_slices = 3, 5
    before = _resolved(_engine(MIXED_STATE, scope), n_gates, n_slices)
    model = IntestinalRoiModel.from_state(MIXED_STATE, scope, n_gates, n_slices)
    after = _resolved(_engine(model.to_state(), model.scope()), n_gates, n_slices)
    assert after == before
    assert (model.mode, model.bg_method) == ("subtract", "mean")


def test_gate_override_priority_matches_model_effective():
    model = IntestinalRoiModel(4, 6)
    model.set_slice(2, TRI_A)
    model.set_gate(3, 2, TRI_B)
    model.set_gate(1, 4, TRI_C)
    engine = _engine(model.to_state(), model.scope())
    assert model.scope() == "gate_override"
    for g in range(4):
        for s in range(6):
            poly, source = model.effective(g, s)
            assert [tuple(p) for p in engine._intestinal_polygon_for_slice(s, gate_index=g)] == [tuple(p) for p in poly]
            assert source == ("gate" if (g, s) in ((3, 2), (1, 4)) else "slice" if s == 2 else None)


def test_model_editing_operations():
    model = IntestinalRoiModel(4, 5)
    model.set_slice(1, TRI_A, all_slices=True)
    assert model.counts() == (5, 0) and model.scope() == "slice"
    model.set_gate(2, 1, TRI_B)
    model.copy_gate_to_all(2, 1)
    assert all(model.effective(g, 1) == ([tuple(p) for p in TRI_B], "gate") for g in range(4))
    assert model.effective(0, 0)[1] == "slice"
    model.clear_gate(0, 1)
    assert model.effective(0, 1) == ([tuple(p) for p in TRI_A], "slice")
    model.reset_gates(1)
    assert model.counts() == (5, 0) and model.scope() == "slice"
    model.clear_slice(3)
    assert 3 not in model.slice_polys and model.effective(0, 3) == ([], None)
    model.clear_slice(0, all_slices=True)
    assert model.counts() == (0, 0)


@pytest.mark.parametrize("all_slices", [False, True])
def test_slice_roi_replaces_gate_overrides_and_attenuates_selected_slice(all_slices):
    """Caso reportado: ajustes por gate en el corte elegido dejaban ese corte sin el ROI nuevo."""
    model = IntestinalRoiModel(8, 18)
    for g in range(8):
        model.set_gate(g, 9, [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)])
    model.set_gate(4, 2, TRI_C)
    big = [(1.0, 1.0), (1.0, 62.0), (62.0, 32.0)]
    model.set_slice(9, big, all_slices=all_slices)
    assert model.gate_overrides(9) == {}
    assert model.gate_overrides(2) == ({} if all_slices else {4: [tuple(p) for p in TRI_C]})
    engine = _engine(model.to_state(), model.scope())
    for g in range(8):
        assert engine._intestinal_polygon_for_slice(9, gate_index=g) == big
    model.set_gate(3, 9, TRI_A)
    model.clear_slice(9)
    assert model.effective(3, 9) == ([], None) and model.gate_overrides(9) == {}


def test_qc_gate_roi_can_span_all_slices():
    model = IntestinalRoiModel(4, 5)
    model.set_slice(0, TRI_A, all_slices=True)
    model.set_gate(2, 1, TRI_B, all_slices=True)
    assert all(model.effective(2, s) == ([tuple(p) for p in TRI_B], "gate") for s in range(5))
    assert all(model.effective(1, s)[1] == "slice" for s in range(5))
    model.clear_gate(2, 0, all_slices=True)
    assert model.counts() == (5, 0)


def test_polygon_area_rejects_degenerate_clicks():
    from ui.montage_intestinal_window import polygon_area
    assert polygon_area([(5.0, 5.0), (5.0, 5.2), (5.1, 5.0)]) < 1.0
    assert polygon_area([(0.0, 0.0), (0.0, 4.0), (4.0, 0.0)]) == pytest.approx(8.0)
    assert polygon_area([(1.0, 1.0), (2.0, 2.0)]) == 0.0


def test_model_ignores_invalid_polygons_and_bad_state():
    state = {"slice_polygons": [{"slice": 0, "points": [[1, 1], [2, 2]]}, {"slice": 1, "points": TRI_A}]}
    model = IntestinalRoiModel.from_state(state, "slice", 2, 3)
    assert list(model.slice_polys) == [1]
    assert IntestinalRoiModel.from_state(None, "slice", 2, 3).counts() == (0, 0)


def test_percentile_display_matches_montage_formula():
    rng = np.random.default_rng(3)
    volume = rng.gamma(2.0, 30.0, size=(6, 16, 16))
    out = montage_display_norm(volume, "percentil", 2.0, 99.5, 0.0, 1.0)
    base, top = np.percentile(volume, 2.0), np.percentile(volume, 99.5)
    np.testing.assert_allclose(out, np.clip((volume - base) / (top - base), 0.0, 1.0))
    np.testing.assert_allclose(montage_display_norm(volume, "percentil", 50.0, 10.0, 0, 1),
                               montage_display_norm(volume, "percentil", 50.0, 51.0, 0, 1))


@pytest.mark.parametrize("low, high", [(0.0, 1.0), (0.2, 1.5), (0.5, 0.4)])
def test_linear_display_matches_montage_formula(low, high):
    volume = np.linspace(10.0, 110.0, 2 * 4 * 4).reshape(2, 4, 4)
    norm = (volume - 10.0) / 100.0
    w1 = high if high > low else low + 0.01
    expected = np.clip((norm - low) / (w1 - low), 0.0, 1.0)
    np.testing.assert_allclose(montage_display_norm(volume, "lineal", 2.0, 99.5, low, high), expected)
    assert not montage_display_norm(np.full((2, 3, 3), 5.0), "lineal", 0, 100, 0.0, 1.0).any()
