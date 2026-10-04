"""Tests de la sustracción de fondo sobre imagen cruda (core.raw_background).

Verifica: rasterización de polígono, medición del nivel de fondo, resta
constante con clip a 0 y aviso de sobre-sustracción, y resta localizada que solo
toca la región del corazón.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from core.raw_background import (
    auto_background_level,
    measure_background_level,
    polygon_mask,
    subtract_constant,
    subtract_localized,
)


def _square_polygon(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_polygon_mask_square():
    mask = polygon_mask((10, 10), _square_polygon(2, 2, 6, 6))
    assert mask.shape == (10, 10)
    # El interior del cuadrado está marcado; las esquinas fuera no.
    assert mask[3, 3]
    assert mask[4, 4]
    assert not mask[0, 0]
    assert not mask[9, 9]
    # Área razonable (par-impar sobre centros de píxel).
    assert 10 <= int(mask.sum()) <= 20


def test_polygon_mask_degenerate_returns_empty():
    assert not polygon_mask((8, 8), [(1, 1), (2, 2)]).any()
    assert not polygon_mask((8, 8), []).any()


def test_measure_background_level_median_and_mean():
    img = np.zeros((10, 10), dtype=np.float64)
    img[2:6, 2:6] = 40.0
    mask = polygon_mask((10, 10), _square_polygon(2, 2, 6, 6))
    assert measure_background_level(img, mask, stat="median") == 40.0
    assert measure_background_level(img, mask, stat="mean") == 40.0
    # Máscara vacía -> 0.
    assert measure_background_level(img, np.zeros((10, 10), bool)) == 0.0


def test_subtract_constant_clips_at_zero():
    img = np.array([[10.0, 30.0], [5.0, 100.0]])
    res = subtract_constant(img, 20.0)
    assert res.method == "constant"
    assert res.level == 20.0
    # 10-20 y 5-20 -> 0 (clip); 30-20=10; 100-20=80.
    np.testing.assert_allclose(res.image, [[0.0, 10.0], [0.0, 80.0]])
    assert res.clipped_fraction == 0.5
    # No modifica la entrada.
    assert img[0, 0] == 10.0


def test_auto_background_level_ignores_air_and_hot_viscera():
    # Proyección sintética: 45% aire (0), 35% pulmón/tejido (10),
    # 10% corazón (100), 10% hígado/intestino (200).
    img = np.zeros((20, 20), dtype=np.float64)
    img[:9, :] = 0.0      # aire (45%)
    img[9:16, :] = 10.0   # pulmón / tejido blando (35%)
    img[16:18, :] = 100.0  # corazón (10%)
    img[18:, :] = 200.0   # hígado / intestino / vesícula (10%)
    level = auto_background_level(img)
    # El nivel debe medir el fondo pulmonar (~10): ni el aire (0)
    # ni las vísceras calientes (>=100).
    assert level == 10.0


def test_auto_background_level_zero_image():
    assert auto_background_level(np.zeros((10, 10))) == 0.0


def test_subtract_constant_negative_level_is_clamped():
    img = np.array([[10.0, 20.0]])
    res = subtract_constant(img, -5.0)
    np.testing.assert_allclose(res.image, img)
    assert res.level == 0.0


def test_subtract_constant_oversubtraction_note():
    img = np.full((10, 10), 5.0)
    res = subtract_constant(img, 100.0)
    assert res.clipped_fraction == 1.0
    assert any("sobre-sustracción" in n.lower() for n in res.notes)


def test_subtract_localized_only_touches_heart():
    img = np.full((10, 10), 50.0)
    heart = np.zeros((10, 10), dtype=bool)
    heart[4:6, 4:6] = True
    res = subtract_localized(img, 20.0, heart, feather_px=0.0)
    # Dentro del corazón: 50-20=30; fuera: intacto 50.
    assert res.image[4, 4] == 30.0
    assert res.image[0, 0] == 50.0
    assert res.method == "localized"


def test_subtract_localized_broadcasts_over_stack():
    stack = np.full((3, 8, 8), 60.0)  # (A, H, W)
    heart = np.zeros((8, 8), dtype=bool)
    heart[3:5, 3:5] = True
    res = subtract_localized(stack, 25.0, heart, feather_px=0.0)
    assert res.image.shape == (3, 8, 8)
    assert np.all(res.image[:, 3, 3] == 35.0)
    assert np.all(res.image[:, 0, 0] == 60.0)


def test_subtract_constant_over_stack():
    stack = np.stack([np.full((6, 6), 30.0), np.full((6, 6), 10.0)])
    res = subtract_constant(stack, 15.0)
    assert res.image[0, 0, 0] == 15.0  # 30-15
    assert res.image[1, 0, 0] == 0.0   # 10-15 -> clip 0


def _bind_background_method(window, name, class_name="BackgroundSubtractionPanel", file_name="background_subtraction_window.py"):
    source = ast.parse((Path(__file__).resolve().parents[1] / "ui" / file_name).read_text(encoding="utf-8"))
    owner = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<background-roi>", "exec"), namespace)
    setattr(window, name, MethodType(namespace[name], window))


def test_background_stage_reload_restores_roi_settings_and_subtraction():
    source = np.full((3, 8, 8), 60.0)
    spec = {"method": "constant", "impact": "chain", "bg_polygon": _square_polygon(0, 0, 2, 2), "heart_polygon": []}
    restored = []
    def combo(values):
        return SimpleNamespace(blockSignals=lambda value: None, findData=lambda value: values.index(value),
                               setCurrentIndex=lambda value: restored.append(value))
    window = SimpleNamespace(
        _current_stage=lambda: "stress", _main=SimpleNamespace(_prep_mip_source_for_stage=lambda stage: ("proj", source, "original")),
        _bg_base={"stress": source.copy()}, _bg_spec={"stress": spec}, _proj_status={},
        mip=SimpleNamespace(set_source=lambda *args: restored.append(args),
                            set_polygons=lambda *args: restored.append(args)),
        _set_controls_enabled=lambda enabled: None, _refresh_bg_widgets=lambda: None,
        bg_method_combo=combo(["constant", "localized"]),
        bg_impact_switch=SimpleNamespace(blockSignals=lambda value: None, setChecked=lambda value: restored.append(value)),
        bg_amount_spin=SimpleNamespace(blockSignals=lambda value: None, setValue=lambda value: None),
        _bg_preview=lambda: restored.append("reapply-from-base"), previewChanged=SimpleNamespace(emit=lambda: None),
        set_bg_status=lambda text: None,
    )
    _bind_background_method(window, "_load_stage_mip")
    window._load_stage_mip()
    assert restored[-1] == "reapply-from-base"
    assert restored[-2] == (spec["bg_polygon"], [])
    assert restored[1:3] == [0, 1]
    np.testing.assert_array_equal(restored[0][1], source)


@pytest.mark.parametrize("checked, impact", [(False, "visual"), (True, "chain")])
def test_background_impact_switch_selects_visual_or_chain(checked, impact):
    window = SimpleNamespace(bg_impact_switch=SimpleNamespace(isChecked=lambda: checked))
    _bind_background_method(window, "_bg_impact")
    assert window._bg_impact() == impact


@pytest.mark.parametrize("stage", ["stress", "rest"])
@pytest.mark.parametrize("kind, enabled", [("proj", True), ("", False), ("vol", False)])
def test_roi_buttons_are_independent_of_method_and_stage(stage, kind, enabled):
    buttons = {}
    panel = SimpleNamespace(
        _stage=stage, mip=SimpleNamespace(kind=lambda: kind),
        bg_roi_btn=SimpleNamespace(setEnabled=lambda value: buttons.update(background=value)),
        heart_roi_btn=SimpleNamespace(setEnabled=lambda value: buttons.update(heart=value)),
    )
    _bind_background_method(panel, "_refresh_bg_widgets")
    panel._refresh_bg_widgets()
    assert buttons == {"background": enabled, "heart": enabled}


@pytest.mark.parametrize("stage", ["stress", "rest"])
@pytest.mark.parametrize("roi, method", [("background", "constant"), ("heart", "localized")])
def test_roi_choice_selects_its_method_without_requiring_other_roi(stage, roi, method):
    state = SimpleNamespace(method=None, draw=None)
    panel = SimpleNamespace(
        _stage=stage, mip=SimpleNamespace(kind=lambda: "proj", set_draw_mode=lambda value: setattr(state, "draw", value)),
        bg_method_combo=SimpleNamespace(findData=lambda value: ["constant", "localized"].index(value),
                                       setCurrentIndex=lambda value: setattr(state, "method", ["constant", "localized"][value])),
        _refresh_bg_widgets=lambda: None, set_bg_status=lambda text: None,
    )
    _bind_background_method(panel, "_bg_draw")
    panel._bg_draw(roi)
    assert state.method == method and state.draw == roi


@pytest.mark.parametrize("stage", ["stress", "rest"])
@pytest.mark.parametrize("impact", ["visual", "chain"])
def test_background_alone_applies_without_heart_and_refreshes_modern(stage, impact):
    refreshed = []
    main = SimpleNamespace(_raw_bg_spec={}, _log=lambda *args: None,
                           _refresh_cine_crudo_view=lambda: refreshed.append("main"),
                           _refresh_modern_raw_preview=lambda: refreshed.append("modern"))
    _bind_background_method(main, "set_raw_background_subtraction", "MainWindow", "main_window.py")
    panel, state = _roi_panel(stage, main, np.full((3, 8, 8), 40.0), method="localized", impact=impact)
    state.heart = []
    panel.mip.set_draw_mode = lambda mode: None
    _bind_background_method(panel, "_bg_draw")
    panel._bg_draw("background")
    assert panel._bg_apply()
    assert main._raw_bg_spec[stage]["method"] == "constant"
    assert main._raw_bg_spec[stage]["heart_polygon"] == []
    np.testing.assert_allclose(state.image, 30.0)
    assert refreshed == ["main", "modern"]


def test_background_dialog_reuses_frameless_epar_header():
    source = ast.parse((Path(__file__).resolve().parents[1] / "ui" / "background_subtraction_window.py").read_text(encoding="utf-8"))
    dialog = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "BackgroundSubtractionWindow")
    assert ast.unparse(dialog.bases[0]) == "EParDialog"
    constructor = next(node for node in dialog.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assert "self.content_layout" in ast.unparse(constructor)
    assert "self.apply_compact_style()" in ast.unparse(constructor)


@pytest.mark.parametrize("impact", ["visual", "chain"])
def test_background_commit_persists_independent_stages_and_refreshes_cines(impact):
    calls = []
    window = SimpleNamespace(_raw_bg_spec={}, _log=lambda *args: None,
                             _refresh_cine_crudo_view=lambda: calls.append("main"),
                             _refresh_modern_raw_preview=lambda: calls.append("modern"))
    _bind_background_method(window, "set_raw_background_subtraction", "MainWindow", "main_window.py")
    _bind_background_method(window, "clear_raw_background_subtraction", "MainWindow", "main_window.py")
    _bind_background_method(window, "_apply_raw_bg_to_recon_cube", "MainWindow", "main_window.py")
    cube = np.full((4, 3, 8, 8), 60.0)
    window.set_raw_background_subtraction("stress", None, {"impact": impact, "method": "constant", "level": 80.0})
    window.set_raw_background_subtraction("rest", None, {"impact": "chain", "method": "constant", "level": 40.0})
    np.testing.assert_allclose(window._apply_raw_bg_to_recon_cube(cube, "stress", for_preview=True), 40.0)
    np.testing.assert_allclose(window._apply_raw_bg_to_recon_cube(cube, "stress"), 40.0 if impact == "chain" else 60.0)
    np.testing.assert_allclose(window._apply_raw_bg_to_recon_cube(cube, "rest"), 50.0)
    assert calls == ["main", "modern", "main", "modern"]
    window.clear_raw_background_subtraction("rest")
    assert "stress" in window._raw_bg_spec and "rest" not in window._raw_bg_spec
    np.testing.assert_allclose(cube, 60.0)


def _roi_panel(stage, main, source, method="constant", impact="visual"):
    state = SimpleNamespace(image=source.copy(), amount=25.0, impact=impact, method=method, status="",
                            background=_square_polygon(0, 0, 2, 2), heart=_square_polygon(2, 2, 7, 7))
    panel = SimpleNamespace(
        _stage=stage, _main=main, _bg_base={stage: source.copy()}, _bg_spec={}, _proj_status={stage: "raw"},
        bg_amount_spin=SimpleNamespace(value=lambda: state.amount, blockSignals=lambda value: None,
                           setValue=lambda value: setattr(state, "amount", value)),
        bg_impact_switch=SimpleNamespace(isChecked=lambda: state.impact == "chain", blockSignals=lambda value: None,
                         setChecked=lambda value: setattr(state, "impact", "chain" if value else "visual")),
        bg_method_combo=SimpleNamespace(blockSignals=lambda value: None,
                           findData=lambda value: ["constant", "localized"].index(value),
                           setCurrentIndex=lambda value: setattr(state, "method", ["constant", "localized"][value])),
        _bg_method=lambda: state.method, set_bg_status=lambda text: setattr(state, "status", text),
        _set_controls_enabled=lambda enabled: None, _refresh_bg_widgets=lambda: None,
        previewChanged=SimpleNamespace(emit=lambda: None),
        mip=SimpleNamespace(kind=lambda: "proj", frames_stack=lambda: state.image.copy(),
                            background_polygon=lambda: state.background, heart_polygon=lambda: state.heart,
                            set_source=lambda kind, image, status: setattr(state, "image", image.copy()),
                            clear_polygons=lambda: (setattr(state, "background", []), setattr(state, "heart", [])),
                            set_polygons=lambda background, heart: (setattr(state, "background", list(background)),
                                                                    setattr(state, "heart", list(heart)))),
    )
    for name in ("_current_stage", "_bg_impact", "_bg_preview", "_bg_apply", "refresh", "_load_stage_mip"):
        _bind_background_method(panel, name)
    return panel, state


@pytest.mark.parametrize("method", ["constant", "localized"])
@pytest.mark.parametrize("impact", ["visual", "chain"])
def test_roi_preview_does_not_commit_or_accumulate_and_apply_preserves_each_stage(method, impact):
    main = SimpleNamespace(_raw_bg_spec={}, _log=lambda *args: None,
                           _refresh_cine_crudo_view=lambda: None, _refresh_modern_raw_preview=lambda: None)
    _bind_background_method(main, "set_raw_background_subtraction", "MainWindow", "main_window.py")
    source = np.full((3, 8, 8), 40.0)
    source[:, 3:6, 3:6] = 200.0
    effort, effort_state = _roi_panel("stress", main, source, method, impact)
    rest, rest_state = _roi_panel("rest", main, source * 0.5, method, impact)
    assert effort._bg_preview()
    assert main._raw_bg_spec == {}
    effort_state.amount = 50.0
    assert effort._bg_preview()
    expected = subtract_constant(source, 20.0) if method == "constant" else subtract_localized(
        source, 20.0, polygon_mask((8, 8), effort_state.heart), feather_px=2.0)
    np.testing.assert_allclose(effort_state.image, expected.image)
    assert effort._bg_apply()
    assert rest._bg_apply()
    assert main._raw_bg_spec["stress"]["level"] == 20.0
    assert main._raw_bg_spec["rest"]["level"] == 5.0
    assert main._raw_bg_spec["stress"]["impact"] == impact
    assert main._raw_bg_spec["rest"]["impact"] == impact
    effort_state.amount = 150.0
    assert effort._bg_preview()
    assert main._raw_bg_spec["stress"]["level"] == 20.0
    np.testing.assert_allclose(effort._bg_base["stress"], source)
    assert "sin aplicar" in effort_state.status


@pytest.mark.parametrize("impact", ["visual", "chain"])
def test_reopening_restores_applied_settings_for_both_stages(impact):
    source = np.full((3, 8, 8), 40.0)
    main = SimpleNamespace(_raw_bg_spec={}, _log=lambda *args: None,
                           _prep_mip_source_for_stage=lambda stage: ("proj", source if stage == "stress" else source * 0.5, "raw"),
                           _refresh_cine_crudo_view=lambda: None, _refresh_modern_raw_preview=lambda: None)
    _bind_background_method(main, "set_raw_background_subtraction", "MainWindow", "main_window.py")
    effort, effort_state = _roi_panel("stress", main, source, impact=impact)
    rest, rest_state = _roi_panel("rest", main, source * 0.5, impact=impact)
    effort_state.amount = 50.0
    assert effort._bg_apply() and rest._bg_apply()
    reopened_effort, effort_ui = _roi_panel("stress", main, source)
    reopened_rest, rest_ui = _roi_panel("rest", main, source * 0.5)
    reopened_effort.refresh()
    reopened_rest.refresh()
    assert effort_ui.amount == 50.0 and rest_ui.amount == 25.0
    assert effort_ui.impact == rest_ui.impact == impact
    assert effort_ui.background == effort_state.background
    np.testing.assert_allclose(effort_ui.image, 20.0)
    np.testing.assert_allclose(rest_ui.image, 15.0)
    assert effort_ui.status == rest_ui.status == "Fondo aplicado restaurado."


@pytest.mark.parametrize("valid_roi", [False, True])
def test_apply_and_close_keeps_dialog_open_for_invalid_roi(valid_roi):
    main = SimpleNamespace(set_raw_background_subtraction=lambda *args: None)
    panel, state = _roi_panel("stress", main, np.full((3, 8, 8), 40.0))
    state.background = _square_polygon(0, 0, 2, 2) if valid_roi else [(0, 0), (1, 1)]
    closed = []
    dialog = SimpleNamespace(_panels={"stress": panel}, close=lambda: closed.append(True))
    _bind_background_method(dialog, "_apply_and_close", "BackgroundSubtractionWindow")
    dialog._apply_and_close()
    assert bool(closed) == valid_roi


@pytest.mark.parametrize("common", [False, True])
def test_background_comparison_scale_is_fixed_to_original_counts(common):
    scales = {}
    panels = {stage: SimpleNamespace(_bg_base={stage: np.full((3, 8, 8), maximum)},
              mip=SimpleNamespace(set_display_max=lambda value, key=stage: scales.update({key: value})))
              for stage, maximum in (("stress", 100.0), ("rest", 50.0))}
    dialog = SimpleNamespace(_panels=panels, common_scale_check=SimpleNamespace(isChecked=lambda: common))
    _bind_background_method(dialog, "_sync_display_scale", "BackgroundSubtractionWindow")
    dialog._sync_display_scale()
    assert scales == {"stress": 100.0, "rest": 100.0 if common else 50.0}


def test_background_cine_renderer_consumes_applied_visual_spec():
    import matplotlib
    cube = np.full((4, 3, 8, 8), 60.0)
    cube[:, :, 3:5, 3:5] = 200.0
    study = SimpleNamespace(cube=cube, stage="Esfuerzo")
    main = SimpleNamespace(
        _raw_bg_spec={"stress": {"impact": "visual", "method": "constant", "level": 80.0}},
        _cine_crudo_stage_display=lambda value: value.stage, _secondary_cine_crudo_study=lambda: None,
        cine_crudo_compare_check=None, cine_crudo_mask_check=None,
        _cine_crudo_threshold_value=lambda: 0.35, _scale_seed_for_study=lambda value: None,
        _append_cine_crudo_sinogram_panel=lambda rgb, *args: rgb,
        _draw_cine_crudo_reference_line=lambda rgb, *args: rgb, _rgb_frame_to_qpixmap_raw=lambda rgb: rgb,
        _log=lambda *args: None,
    )
    _bind_background_method(main, "_apply_raw_bg_to_recon_cube", "MainWindow", "main_window.py")
    _bind_background_method(main, "_build_cine_crudo_frames_for_study", "MainWindow", "main_window.py")
    frames, _, _ = main._build_cine_crudo_frames_for_study(study, None, "UngGat", "Esfuerzo")
    images = np.clip(cube.sum(axis=0) - 80.0, 0.0, None)
    expected = (matplotlib.colormaps["gray"](images / np.percentile(images, 99.0))[..., :3] * 255).astype(np.uint8)
    np.testing.assert_array_equal(np.stack(frames), expected)


@pytest.mark.parametrize("primary_label", ["Esfuerzo", "Reposo"])
def test_background_stage_sources_ignore_active_recon_selector(primary_label):
    primary = SimpleNamespace(reconstructed=False, cube=np.full((1, 3, 8, 8), 40.0), stage=primary_label)
    secondary_label = "Reposo" if primary_label == "Esfuerzo" else "Esfuerzo"
    secondary = SimpleNamespace(reconstructed=False, cube=np.full((1, 3, 8, 8), 20.0), stage=secondary_label)
    main = SimpleNamespace(study=primary, cine_crudo_raw_study_for_recon=secondary,
                           _cine_crudo_stage_display=lambda study: study.stage,
                           _secondary_cine_crudo_study=lambda: secondary)
    _bind_background_method(main, "_prep_mip_source_for_stage", "MainWindow", "main_window.py")
    for stage, label in (("stress", "Esfuerzo"), ("rest", "Reposo")):
        kind, image, _ = main._prep_mip_source_for_stage(stage)
        assert kind == "proj"
        np.testing.assert_allclose(image, 40.0 if label == primary_label else 20.0)
