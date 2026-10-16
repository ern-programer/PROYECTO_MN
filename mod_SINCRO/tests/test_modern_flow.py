"""Modern load/preview wiring without creating QApplication or real output files."""
import ast
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace, MethodType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ast.parse((ROOT / "ui" / "main_window.py").read_text(encoding="utf-8"))
MAIN_CLASS = next(node for node in SOURCE.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
METHODS = {
    node.name: node for node in MAIN_CLASS.body if isinstance(node, ast.FunctionDef)
}


def _bind(window, name, **globals_dict):
    namespace = {"os": os, "perf_counter": perf_counter, **globals_dict}
    exec(compile(ast.Module(body=[METHODS[name]], type_ignores=[]), str(ROOT / "ui" / "main_window.py"), "exec"), namespace)
    setattr(window, name, MethodType(namespace[name], window))


@pytest.mark.parametrize("size", [64, 128])
@pytest.mark.parametrize("global_enabled,local_enabled", [(False, False), (False, True), (True, False)])
def test_raw_montage_disabled_intestinal_preserves_original_cuts(size, global_enabled, local_enabled):
    import numpy as np
    from core.cardiac_reorientation import anatomical_cuts_gated

    source = np.random.default_rng(42).random((1, 8, size, size))
    widget = SimpleNamespace(intestinal_apply_enabled=lambda: local_enabled)
    window = SimpleNamespace(global_intestinal_render_check=SimpleNamespace(isChecked=lambda: global_enabled))
    _bind(window, "_apply_intestinal_mask_to_cube", np=np, CineWidget=object)
    _bind(window, "_recut_crudo_stage_with_intestinal", np=np)
    original = source.copy()
    window._recut_crudo_stage_with_intestinal("stress", {"gated": source, "ungated": source}, widget)
    expected = anatomical_cuts_gated(original)
    for axis in ("SA", "HLA", "VLA"):
        np.testing.assert_array_equal(window.cine_crudo_axes_for_export_stress[axis], expected[axis.lower()])
        np.testing.assert_array_equal(window.cine_crudo_axes_for_export_ungated_stress[axis], expected[axis.lower()])
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("primary_stage", ["stress", "rest"])
@pytest.mark.parametrize("stage", ["stress", "rest"])
def test_reconstruction_invalidates_only_stage_intestinal_rois_and_montage(primary_stage, stage):
    from unittest.mock import MagicMock

    primary, compare = MagicMock(), MagicMock()
    sources = {"stress": object(), "rest": object()}
    window = SimpleNamespace(study=object(), cine=primary, cine_compare=compare,
        _cine_crudo_stage_display=lambda study: "Reposo" if primary_stage == "rest" else "Esfuerzo",
        _crudo_montage_reo_by_stage=sources.copy(), _stop_montage_cine=MagicMock(), _log=MagicMock(),
        _montage_gray_cache=object(), _montage_volume_cache={1: object()},
        _montage_cut_cache={1: object()}, _montage_tile_cache={1: object()})
    other = "rest" if stage == "stress" else "stress"
    for prefix in ("cine_crudo_axes_for_export", "cine_crudo_axes_for_export_ungated", "cine_crudo_axes_for_export_mf"):
        setattr(window, f"{prefix}_{stage}", {"SA": object()})
        setattr(window, f"{prefix}_{other}", sources[other])
    _bind(window, "_invalidate_reconstructed_intestinal_stage")
    window._invalidate_reconstructed_intestinal_stage(stage)
    target, untouched = (primary, compare) if stage == primary_stage else (compare, primary)
    target.set_intestinal_roi_state.assert_called_once_with(None)
    untouched.set_intestinal_roi_state.assert_not_called()
    target.set_intestinal_params.assert_not_called()
    assert window._crudo_montage_reo_by_stage == {other: sources[other]}
    for prefix in ("cine_crudo_axes_for_export", "cine_crudo_axes_for_export_ungated", "cine_crudo_axes_for_export_mf"):
        assert getattr(window, f"{prefix}_{stage}") == {}
        assert getattr(window, f"{prefix}_{other}") is sources[other]
    assert window._montage_gray_cache is None
    assert not window._montage_volume_cache and not window._montage_cut_cache and not window._montage_tile_cache
    window._stop_montage_cine.assert_called_once_with()


def test_reconstruction_invalidates_intestinal_only_after_pipeline_succeeds():
    method = METHODS["_reconstruct_cine_crudo_raw"]
    body = next(node for node in method.body if isinstance(node, ast.Try)).body
    invalidation = next(index for index, node in enumerate(body)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "_invalidate_reconstructed_intestinal_stage")
    pipeline = next(index for index, node in enumerate(body)
        if isinstance(node, ast.Try) and any(isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name) and child.func.id == "reconstruct_raw_gated_pipeline" for child in ast.walk(node)))
    assert pipeline < invalidation
    assert "self.cine_crudo_recon_result = result" == ast.unparse(body[invalidation + 1])


@pytest.mark.parametrize("primary_stage", ["stress", "rest"])
def test_montage_intestinal_editor_sources_raw_use_unmasked_sa_before_processing(primary_stage):
    import numpy as np
    stress, rest = np.ones((8, 20, 64, 64)), np.ones((1, 18, 64, 64))
    primary, compare = object(), object()
    window = SimpleNamespace(study=SimpleNamespace(reconstructed=False), cine=primary, cine_compare=compare,
        _cine_crudo_recon_stage=primary_stage,
        _crudo_montage_reo_by_stage={"stress": {"gated": stress}, "rest": {"ungated": rest}})
    _bind(window, "_montage_intestinal_sources", np=np)
    sources = window._montage_intestinal_sources()
    assert sources[0][0] == "stress" and sources[0][1] is stress
    assert sources[1][0] == "rest" and sources[1][1] is rest
    assert sources[0][2] is (primary if primary_stage == "stress" else compare)
    assert sources[1][2] is (primary if primary_stage == "rest" else compare)


@pytest.mark.parametrize("secondary_loaded", [False, True])
def test_montage_intestinal_editor_sources_direct_sa_support_pending_second_stage(secondary_loaded):
    import numpy as np
    first, second = np.ones((8, 20, 64, 64)), np.ones((1, 16, 64, 64))
    window = SimpleNamespace(study=SimpleNamespace(cube=first), cine=object(), cine_compare=object(),
        _sa_direct_montage_stages=("rest", "stress" if secondary_loaded else None),
        _second_stage_study=lambda: SimpleNamespace(cube=second))
    _bind(window, "_montage_intestinal_sources", np=np)
    sources = window._montage_intestinal_sources()
    assert sources[0][0] == "rest" and sources[0][1] is first and sources[0][2] is window.cine
    assert len(sources) == 1 + int(secondary_loaded)
    if secondary_loaded:
        assert sources[1][0] == "stress" and sources[1][1] is second and sources[1][2] is window.cine_compare


def test_montage_intestinal_editor_sources_empty_without_montage():
    import numpy as np
    window = SimpleNamespace(study=None)
    _bind(window, "_montage_intestinal_sources", np=np)
    assert window._montage_intestinal_sources() == []
    window.study = object()
    assert window._montage_intestinal_sources() == []


@pytest.mark.parametrize("action", ["apply", "close", "stale", "global_off"])
def test_montage_intestinal_window_applies_explicitly_and_cleans_up(monkeypatch, action):
    from unittest.mock import MagicMock
    cube = object()
    targets = [MagicMock(), MagicMock()]
    sources = [("stress", cube, targets[0]), ("rest", cube, targets[1])]
    lcars, engine, results = MagicMock(), object(), []
    window_cls = MagicMock(return_value=lcars)
    monkeypatch.setitem(sys.modules, "ui.montage_intestinal_window",
                        SimpleNamespace(MontageIntestinalWindow=window_cls))
    window = SimpleNamespace(
        _montage_intestinal_sources=MagicMock(return_value=sources),
        _stop_montage_cine=MagicMock(), _dialog_parent=lambda: None,
        _apply_montage_intestinal_editor=MagicMock(),
        global_intestinal_render_check=SimpleNamespace(isChecked=lambda: action != "global_off"),
        cine_crudo_montage_cmap="hot", cine_crudo_montage_win_mode="lineal", _all_cmaps=["gray", "hot"],
        cine_crudo_montage_win_low=5.0, cine_crudo_montage_win_high=98.0,
        cine_crudo_montage_lin_low=0.1, cine_crudo_montage_lin_high=1.4,
    )

    def exercise():
        if action == "close":
            return
        if action == "stale":
            window._montage_intestinal_sources.return_value = []
        apply_stage = window_cls.call_args.kwargs["apply_stage"]
        try:
            results.append(apply_stage(1, engine))
        except RuntimeError as exc:
            results.append(exc)

    lcars.exec.side_effect = exercise
    _bind(window, "_open_montage_intestinal_editor", QMessageBox=MagicMock())
    window._open_montage_intestinal_editor()
    kwargs = window_cls.call_args.kwargs
    assert kwargs["sources"] is sources
    assert kwargs["display"] == {"cmap": "hot", "cmaps": ["gray", "hot"], "mode": "lineal", "win_low": 5.0,
                                 "win_high": 98.0, "lin_low": 0.1, "lin_high": 1.4}
    if action in ("apply", "global_off"):
        window._apply_montage_intestinal_editor.assert_called_once_with(engine, targets[1])
        assert ("desactivada" in results[0]) == (action == "global_off")
    else:
        window._apply_montage_intestinal_editor.assert_not_called()
    if action == "stale":
        assert isinstance(results[0], RuntimeError)
    window._stop_montage_cine.assert_called_once()
    lcars.shutdown.assert_called_once()
    lcars.deleteLater.assert_called_once()


def test_montage_intestinal_apply_updates_only_selected_stage_and_refreshes():
    calls = []
    state = {"slice_polygons": [{"slice": 3, "points": [[1, 1], [3, 1], [2, 3]]}],
             "mode": "attenuate", "background_method": "idw"}
    editor = SimpleNamespace(intestinal_params=lambda: (70, 2), intestinal_scope=lambda: "all_slices",
                             intestinal_roi_state=lambda: state, intestinal_apply_enabled=lambda: True)
    target = SimpleNamespace(
        set_intestinal_params=lambda *values: calls.append(("params", values)),
        set_intestinal_scope=lambda value: calls.append(("scope", value)),
        set_intestinal_roi_state=lambda value: calls.append(("roi", value)),
        set_intestinal_apply_enabled=lambda value: calls.append(("enabled", value)),
    )
    window = SimpleNamespace(
        _stop_montage_cine=lambda: calls.append("stop"),
        _maybe_rebuild_sa_direct_montage=lambda: calls.append("sa"),
        _maybe_rebuild_crudo_montage=lambda: calls.append("raw"),
        _show_cine_crudo_sa_montage=lambda **kwargs: calls.append(kwargs),
    )
    _bind(window, "_apply_montage_intestinal_editor")
    window._apply_montage_intestinal_editor(editor, target)
    assert calls == [("params", (70, 2)), ("scope", "all_slices"), ("roi", state), ("enabled", True),
                     "stop", "sa", "raw", {"navigate": False, "persist": False}]


@pytest.mark.parametrize("zoom", [1.0, 5.0, 8.0])
def test_motion_grid_lines_are_one_screen_pixel_on_pixel_borders(zoom):
    window = SimpleNamespace()
    _bind(window, "_cine_crudo_motion_grid_rects")
    _bind(window, "_cine_crudo_motion_grid_lines")
    rects = window._cine_crudo_motion_grid_rects((64, 64, 2))
    assert rects == [(0, 0.0, 64, 64), (64, 0.0, 64, 64)]
    lines = window._cine_crudo_motion_grid_lines(rects, zoom, zoom)
    assert len(lines) == 2 * 2 * (8 + 7)
    vertical = {(x1, alpha) for alpha, x1, y1, x2, y2 in lines if x1 == x2 and y1 != y2}
    assert (round(4 * zoom), 70) in vertical and (round(8 * zoom), 140) in vertical
    assert (round(72 * zoom), 140) in vertical and (round(68 * zoom), 70) in vertical
    assert not any(x in (round(2 * zoom), round(64 * zoom)) for x, _ in vertical if zoom > 1)
    assert all(x1 == x2 or y1 == y2 for _, x1, y1, x2, y2 in lines)


def test_motion_grid_rects_follow_dual_supersampled_layout():
    window = SimpleNamespace()
    _bind(window, "_cine_crudo_motion_grid_rects")
    meta = {"enabled": True, "ss": 3, "canvas_w": 300, "bar_h": 30, "split_y": 250,
            "top_w": 225, "bottom_w": 225}
    rects = window._cine_crudo_motion_grid_rects((64, 64, 1), (64, 64, 1), meta)
    assert rects == [(37 / 3, 10.0, 64, 64), (37 / 3, 280 / 3, 64, 64)]
    assert window._cine_crudo_motion_grid_rects(None) == []


@pytest.mark.parametrize("enabled", [False, None])
def test_motion_grid_disabled_leaves_display_untouched(enabled):
    window = SimpleNamespace(_cine_crudo_grid_rects=[(0, 0, 64, 64)])
    if enabled is not None:
        window.cine_crudo_motion_grid_btn = SimpleNamespace(isChecked=lambda: enabled)
    _bind(window, "_paint_cine_crudo_motion_grid", QPixmap=object, QSize=object)
    pix = object()
    assert window._paint_cine_crudo_motion_grid(pix, SimpleNamespace(isEmpty=lambda: False)) is pix


@pytest.mark.parametrize("source", ["UngGat", "Gated"])
@pytest.mark.parametrize("mode", ["original", "compare", "difference"])
def test_motion_grid_preview_keeps_data_and_sinogram_unchanged(source, mode):
    import numpy as np
    original = np.zeros((2, 3, 64, 64), dtype=np.float64)
    corrected = np.zeros_like(original)
    study = SimpleNamespace(cube=original)
    checked = lambda value: SimpleNamespace(isChecked=lambda: value, isEnabled=lambda: True)
    window = SimpleNamespace(
        cine_crudo_motion_grid_btn=checked(True),
        cine_crudo_compare_check=checked(mode == "compare"),
        cine_crudo_diff_check=checked(mode == "difference"),
        cine_crudo_mask_check=checked(False),
        _cine_crudo_stage_display=lambda study: "Esfuerzo",
        _cine_crudo_threshold_value=lambda: 0.2,
        _scale_seed_for_study=lambda study: None,
        _cine_crudo_marker_color=lambda: np.array([200, 200, 200]),
        _append_cine_crudo_sinogram_panel=lambda rgb, *args: np.concatenate(
            [rgb, np.zeros((64, 11, 3), dtype=np.uint8)], axis=1),
        _draw_cine_crudo_reference_line=lambda rgb, height: rgb,
        _rgb_frame_to_qpixmap_raw=lambda rgb: rgb,
        cine_crudo_roi_spin=SimpleNamespace(value=lambda: 6),
        cine_crudo_roi_mode_combo=SimpleNamespace(currentText=lambda: "Caja"),
    )
    _bind(window, "_cine_crudo_seed_segments", np=np)
    _bind(window, "_build_cine_crudo_frames_for_study", np=np, QPixmap=object)
    frames, counts, _ = window._build_cine_crudo_frames_for_study(
        study, corrected, source, "Esfuerzo", seed_stage=(30, 30))
    assert len(window._cine_crudo_seed_last) == 6
    assert len(frames) == 3
    assert not counts.any()
    assert not original.any() and not corrected.any()
    assert window._cine_crudo_grid_last == (64, 64, 1 if mode == "original" else 2)
    window.cine_crudo_motion_grid_btn = checked(False)
    plain, _, _ = window._build_cine_crudo_frames_for_study(
        study, corrected, source, "Esfuerzo")
    assert window._cine_crudo_seed_last == []
    assert all(np.array_equal(frame, baseline) for frame, baseline in zip(frames, plain))


def test_seed_overlay_segments_box_band_and_cross():
    import numpy as np
    window = SimpleNamespace(
        cine_crudo_roi_spin=SimpleNamespace(value=lambda: 6),
        cine_crudo_roi_mode_combo=SimpleNamespace(currentText=lambda: "Caja"),
        _cine_crudo_band_bounds=lambda H: (20.0, 41.7),
    )
    _bind(window, "_cine_crudo_seed_segments", np=np)
    assert window._cine_crudo_seed_segments(None, 64, 64) == []
    box = window._cine_crudo_seed_segments((30, 30), 64, 64)
    assert box[:4] == [(24.5, 24.5, 36.5, 24.5), (24.5, 36.5, 36.5, 36.5),
                       (24.5, 24.5, 24.5, 36.5), (36.5, 24.5, 36.5, 36.5)]
    assert box[4:] == [(27.5, 30.5, 33.5, 30.5), (30.5, 27.5, 30.5, 33.5)]
    edge = window._cine_crudo_seed_segments((2, 62), 64, 64)
    assert all(0.5 <= v <= 63.5 for segment in edge for v in segment)
    window.cine_crudo_roi_mode_combo = SimpleNamespace(currentText=lambda: "Banda Y")
    band = window._cine_crudo_seed_segments((30, 30), 64, 64)
    assert band[:2] == [(0.5, 20.5, 63.5, 20.5), (0.5, 41.5, 63.5, 41.5)]
    window.cine_crudo_roi_spin = SimpleNamespace(value=lambda: 0)
    assert len(window._cine_crudo_seed_segments((30, 30), 64, 64)) == 2


def test_seed_overlay_base_segments_follow_original_panel_of_each_stage():
    window = SimpleNamespace()
    _bind(window, "_cine_crudo_seed_base_segments")
    rects = [(10.0, 5.0, 64, 64), (74.0, 5.0, 64, 64), (12.0, 90.0, 64, 64), (76.0, 90.0, 64, 64)]
    seg = [(1.5, 2.5, 3.5, 2.5)]
    assert window._cine_crudo_seed_base_segments(rects, (64, 64, 2), seg, seg) == [
        (11.5, 7.5, 13.5, 7.5), (13.5, 92.5, 15.5, 92.5)]
    assert window._cine_crudo_seed_base_segments(rects[:2], (64, 64, 2), seg, []) == [(11.5, 7.5, 13.5, 7.5)]
    assert window._cine_crudo_seed_base_segments([], None, seg, seg) == []


def test_motion_grid_is_available_before_manual_and_painted_after_zoom():
    source = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert 'toolbar2.addWidget(self.cine_crudo_motion_grid_btn)' in source
    assert 'self.cine_crudo_motion_grid_btn.toggled.connect(lambda _=False: self._apply_preview_zoom("cine_crudo"))' in source
    zoom = ast.unparse(METHODS["_apply_preview_zoom"])
    assert zoom.index("pix.scaled(") < zoom.index("self._paint_cine_crudo_screen_overlays(scaled, base_size)")
    loader = ast.unparse(METHODS["_load_cine_crudo_frames"])
    assert "self._cine_crudo_motion_grid_rects(primary_grid, compare_grid, self._cine_crudo_dual_render_meta)" in loader


@pytest.mark.parametrize("voxel_scale", [1.0, 0.5])
def test_diagnostic_snapshots_keep_stages_separate_and_invalidate_old_geometry(tmp_path, voxel_scale):
    import json
    import numpy as np
    from core.raw_reconstruction import RawReconConfig
    window = SimpleNamespace(output_dir=str(tmp_path), _log=lambda *_: None)
    _bind(window, "_dump_feta_for_harness", np=np)
    _bind(window, "_dump_reorient_for_harness", np=np)
    config = RawReconConfig(gated_method="osem", gated_iterations=5, gated_osem_subsets=4)
    study = SimpleNamespace(pixel_spacing=(6.8, 6.8), source_path="test.dcm")
    result = SimpleNamespace(voxel_scale=voxel_scale, ungated_projections=np.ones((4, 6, 6)),
                             corrected_projections=np.ones((8, 4, 6, 6)),
                             ungated_volume=np.ones((6, 6, 6)), gated_volume=np.ones((8, 6, 6, 6)))
    dialog = SimpleNamespace(result_long_axis=(1, 0, 0), result_center=(3, 3, 3), _reformat_sample_scale=0.5)
    for stage, value in (("stress", 1.0), ("rest", 2.0)):
        window._cine_crudo_recon_stage = stage
        window._dump_feta_for_harness(result, study, np.arange(4), config, 1, 4, stage)
        window.cine_crudo_reoriented_gated = np.full((8, 4, 6, 6), value)
        window._dump_reorient_for_harness(dialog)
    root = tmp_path / "_feta_harness"
    for stage, value in (("stress", 1.0), ("rest", 2.0)):
        folder = root / stage
        metadata = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        assert metadata["pixel_mm"] == 6.8 * voxel_scale
        assert metadata["config"]["gated_iterations"] == 5
        assert metadata["config"]["gated_osem_subsets"] == 4
        orientation = json.loads((folder / "reorient.json").read_text())
        assert orientation["stage"] == stage
        assert orientation["voxel_mm"] == 3.4 * voxel_scale
        assert orientation["reconstruction_sha256"] == metadata["reconstruction_sha256"]
        np.testing.assert_array_equal(np.load(folder / "reoriented_gated.npy"), value)
    window._dump_feta_for_harness(result, study, np.arange(4), config, 2, 4, "rest")
    assert not (root / "rest" / "reorient.json").exists()
    assert not (root / "rest" / "reoriented_gated.npy").exists()
    assert (root / "stress" / "reorient.json").exists()


@pytest.mark.parametrize("matrix", [None, 128])
def test_reconstruction_config_keeps_fine_grid_opt_in(matrix):
    from core.raw_reconstruction import ProjectionFilterConfig
    window = SimpleNamespace(
        cine_crudo_matrix_combo=SimpleNamespace(currentData=lambda: matrix),
        _research_flag=lambda key, default: True,
        _cine_crudo_recon_filter_config=lambda branch: ProjectionFilterConfig(),
        _cine_crudo_post_filter_sigma_px=lambda study: 0.0,
        _cine_crudo_post_filter_gated_sigma_px=lambda study: 0.0,
    )
    _bind(window, "_cine_crudo_recon_config")
    config = window._cine_crudo_recon_config()
    assert config.reconstruction_matrix == matrix
    assert config.reconstruction_method == config.gated_method == "fbp"
    assert config.ungated_filter == config.gated_filter == ProjectionFilterConfig()


def test_matrix_change_resets_pixel_based_orientation_locks():
    window = SimpleNamespace(_reorient_locked_voi={"rx": 10}, _reorient_seed={"half_length": 12},
                             _log=lambda *_: None)
    _bind(window, "_on_reconstruction_matrix_changed")
    window._on_reconstruction_matrix_changed(1)
    assert window._reorient_locked_voi is None
    assert window._reorient_seed is None


def test_feta_matrix_selector_lives_in_visible_pipeline_not_sidebar():
    def assignments(method):
        return [node for node in ast.walk(METHODS[method]) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Attribute) and target.attr == "cine_crudo_matrix_combo"
                        for target in node.targets)]
    assert len(assignments("_build_pipeline_step_bar")) == 1
    assert not assignments("_build_sidebar")
    source = ast.unparse(METHODS["_build_pipeline_step_bar"])
    assert "Matriz de feta" in source
    assert "128 x 128 (experimental)" in source
    assert "self.cine_crudo_matrix_combo.currentIndexChanged.connect(self._on_reconstruction_matrix_changed)" in source
    assert "matrix_row.addWidget(self._effective_matrix_label, 1)" in source


@pytest.mark.parametrize("size,band,spacing,expected", [
    (64, None, 6.8, "recon 64 x 64 (volumen) · 6.800 mm/px"),
    (128, (12, 31), 3.4, "recon 128 x 128 (feta) · experimental · 3.400 mm/px"),
])
def test_effective_matrix_readout_uses_volume_not_selected_target(size, band, spacing, expected):
    import numpy as np
    result = SimpleNamespace(gated_volume=np.zeros((2, 8, size, size)),
                             voxel_scale=64 / size, config=SimpleNamespace(recon_slice_range=band))
    stress = SimpleNamespace(recon_result=result, raw_study=SimpleNamespace(pixel_spacing=(6.8, 6.8)))
    states = {"stress": stress, "rest": SimpleNamespace()}
    label = SimpleNamespace(setText=lambda text: setattr(label, "text", text))
    window = SimpleNamespace(_effective_matrix_label=label,
                             cine_crudo_matrix_combo=SimpleNamespace(currentData=lambda: 128),
                             _dual_session=lambda: SimpleNamespace(stage=lambda stage: states[stage]))
    _bind(window, "_refresh_effective_matrix_readout")
    window._refresh_effective_matrix_readout()
    assert label.text == "Esfuerzo: " + expected
    stress.cut_study = SimpleNamespace(cube=np.zeros((2, 10, 32, 32)), pixel_spacing=(spacing / 2, spacing / 2))
    window._refresh_effective_matrix_readout()
    assert f"SA 32 x 32 · {spacing / 2:.3f} mm/px" in label.text
    states["rest"] = SimpleNamespace(recon_result=SimpleNamespace(
        gated_volume=np.zeros((2, 8, 64, 64)), config=SimpleNamespace(recon_slice_range=None)))
    window._refresh_effective_matrix_readout()
    assert "Reposo: recon 64 x 64 (volumen)" in label.text
    states["stress"] = states["rest"] = SimpleNamespace()
    window._refresh_effective_matrix_readout()
    assert label.text == "Matriz efectiva: sin reconstrucción"


@pytest.mark.parametrize("matrix,feta_only,force,expected_matrix,expected_method", [
    (128, False, False, None, "fbp"),
    (128, True, False, 128, "osem"),
    (None, False, False, None, "osem"),
    (128, False, True, None, "fbp"),
    (None, False, True, None, "fbp"),
])
def test_fine_matrix_applies_only_after_native_fbp_localizer(matrix, feta_only, force,
                                                          expected_matrix, expected_method):
    from core.raw_reconstruction import RawReconConfig
    method = METHODS["_reconstruct_cine_crudo_raw"]
    body = next(node for node in method.body if isinstance(node, ast.Try)).body
    localizer = next(node for node in body if isinstance(node, ast.If)
                     and "force_fbp_localizer" in ast.unparse(node.test))
    config = RawReconConfig(reconstruction_matrix=matrix, reconstruction_method="osem",
                            gated_method="osem", nitida3_enabled=True, nitida4d_enabled=True,
                            resolution_recovery=True, ungated_denoise_plus=True,
                            gated_denoise_plus=True, fbp_clean_sigma_color=0.04,
                            attenuation_correction=True)
    window = SimpleNamespace(_use_adjoint_osem=True, _log=lambda *_: None)
    namespace = {"self": window, "cfg": config, "feta_only": feta_only, "force_fbp_localizer": force}
    exec(compile(ast.Module(body=[localizer], type_ignores=[]), "localizer_route", "exec"), namespace)
    actual = namespace["cfg"]
    assert actual.reconstruction_matrix == expected_matrix
    assert actual.reconstruction_method == actual.gated_method == expected_method
    if expected_method == "fbp":
        assert not actual.resolution_recovery and not actual.nitida3_enabled and not actual.nitida4d_enabled
        assert not actual.ungated_denoise_plus and not actual.gated_denoise_plus
        assert actual.fbp_clean_sigma_color == 0.0 and not actual.attenuation_correction
        assert not window._use_adjoint_osem
    else:
        assert actual == config


@pytest.mark.parametrize("method,expected", [("FBP", "fbp"), ("MLEM", "mlem"), ("OSEM", "osem"), ("OSEM-Adj", "osem")])
def test_fine_matrix_config_accepts_all_visible_reconstruction_methods(method, expected):
    from core.raw_reconstruction import ProjectionFilterConfig
    window = SimpleNamespace(
        cine_crudo_matrix_combo=SimpleNamespace(currentData=lambda: 128),
        _research_flag=lambda key, default: True,
        cine_crudo_recon_method_combo=SimpleNamespace(currentText=lambda: method),
        cine_crudo_gated_method_combo=SimpleNamespace(currentText=lambda: method),
        _cine_crudo_recon_filter_config=lambda branch: ProjectionFilterConfig(),
        _cine_crudo_post_filter_sigma_px=lambda study: 0.0,
        _cine_crudo_post_filter_gated_sigma_px=lambda study: 0.0,
    )
    _bind(window, "_cine_crudo_recon_config")
    config = window._cine_crudo_recon_config()
    assert config.reconstruction_matrix == 128
    assert config.reconstruction_method == config.gated_method == expected


@pytest.mark.parametrize("gates", [1, 2, 8, 16])
@pytest.mark.parametrize("voxel_scale", [1.0, 0.5])
@pytest.mark.parametrize("stage", ["stress", "rest"])
def test_raw_recon_applies_automatic_feta_for_gated_and_ungated_volumes(
        monkeypatch, gates, voxel_scale, stage):
    import numpy as np
    from core import lv_center
    block = next(node for node in ast.walk(METHODS["_reconstruct_cine_crudo_raw"])
                 if isinstance(node, ast.If) and "hasattr(self, 'cine_crudo_cut_base_spin')" in ast.unparse(node.test))
    result = SimpleNamespace(gated_volume=np.ones((gates, 32, 4, 4)),
                             ungated_volume=np.ones((32, 4, 4)), voxel_scale=voxel_scale)
    calls = []
    values = {}
    def detect(volume, margin):
        calls.append((volume, margin))
        return (8, 22)
    monkeypatch.setattr(lv_center, "heart_axial_bounds_from_spect", detect)
    def spin(key):
        return SimpleNamespace(setEnabled=lambda enabled: None, setRange=lambda lower, upper: None,
                               setValue=lambda value: values.update({key: value}))
    window = SimpleNamespace(
        cine_crudo_cut_base_spin=spin("base"), cine_crudo_cut_apex_spin=spin("apex"),
        _suggest_feta_limits_from_fusion=lambda stage: False,
        _cine_crudo_stage_limits_set=lambda stage, base, apex, size: values.update(
            saved=(stage, base, apex, size)),
        _log=lambda message: None,
    )
    namespace = {"self": window, "result": result, "n_slices": 32,
                 "feta_only": False, "stage": stage, "np": np}
    exec(compile(ast.Module(body=[block], type_ignores=[]), "automatic_feta", "exec"), namespace)
    assert len(calls) == 1
    assert calls[0][0] is (result.gated_volume if gates >= 3 else result.ungated_volume)
    assert calls[0][1] == int(np.ceil(3 / voxel_scale))
    assert values == {"base": 10, "apex": 22, "saved": (stage, 10, 22, 32)}


@pytest.mark.parametrize("previous_scale,new_scale,base,apex,expected", [
    (1.0, 0.5, 12, 24, (23, 47)),
    (0.5, 0.5, 23, 47, (23, 47)),
    (0.5, 1.0, 23, 47, (12, 24)),
])
def test_feta_matrix_change_preserves_markers_without_accumulating_safety_margin(
        previous_scale, new_scale, base, apex, expected):
    local = next(node for node in ast.walk(METHODS["_reconstruct_cine_crudo_raw"])
                 if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                 and isinstance(node.test.operand, ast.Name) and node.test.operand.id == "feta_only")
    values = {}
    window = SimpleNamespace(
        cine_crudo_cut_base_spin=SimpleNamespace(setValue=lambda value: values.update(base=value)),
        cine_crudo_cut_apex_spin=SimpleNamespace(setValue=lambda value: values.update(apex=value)),
        _cine_crudo_capture_limits_from_spins=lambda stage: None,
    )
    namespace = {"self": window, "result": SimpleNamespace(voxel_scale=new_scale),
                 "previous_scale": previous_scale, "selected_base": base, "selected_apex": apex,
                 "n_slices": int(64 / new_scale), "stage": "stress"}
    exec(compile(ast.Module(body=local.orelse, type_ignores=[]), "feta_markers", "exec"), namespace)
    assert (values["base"], values["apex"]) == expected


def test_research_preference_controls_fine_selector_and_preserves_existing_volume():
    class Combo:
        def __init__(self):
            self.options = [("Nativa", None), ("128 x 128 (experimental)", 128)]
            self.index = 1
        def findData(self, value):
            return next((index for index, option in enumerate(self.options) if option[1] == value), -1)
        def addItem(self, text, value):
            self.options.append((text, value))
        def setCurrentIndex(self, index):
            self.index = index
        def removeItem(self, index):
            self.options.pop(index)
        def setEnabled(self, enabled):
            self.enabled = enabled
    values = {}
    volume = object()
    window = SimpleNamespace(cine_crudo_matrix_combo=Combo(), cine_crudo_recon_result=volume,
                             _ui_settings=SimpleNamespace(value=lambda key, default, type: values.get(key, default)))
    _bind(window, "_research_flag")
    _bind(window, "_apply_fine_matrix_research_preference")
    window._apply_fine_matrix_research_preference()
    combo = window.cine_crudo_matrix_combo
    assert combo.options == [("Nativa", None)] and combo.index == 0
    assert combo.enabled is False and window.cine_crudo_recon_result is volume
    values["research/fine_reconstruction_enabled"] = True
    window._apply_fine_matrix_research_preference()
    window._apply_fine_matrix_research_preference()
    assert len(combo.options) == 2 and combo.findData(128) == 1 and combo.enabled
    fresh_window = SimpleNamespace(_ui_settings=window._ui_settings)
    _bind(fresh_window, "_research_flag")
    assert fresh_window._research_flag("fine_reconstruction_enabled", False) is True
    source = ast.unparse(METHODS["open_ui_preferences_dialog"])
    assert "research/fine_reconstruction_enabled" in source
    assert "fine_matrix_enabled.isChecked()" in source


@pytest.mark.parametrize("enabled", [False, True])
def test_research_preference_gates_fine_config_even_with_stale_selector(enabled):
    from core.raw_reconstruction import ProjectionFilterConfig
    window = SimpleNamespace(
        cine_crudo_matrix_combo=SimpleNamespace(currentData=lambda: 128),
        _research_flag=lambda key, default: enabled,
        _cine_crudo_recon_filter_config=lambda branch: ProjectionFilterConfig(),
        _cine_crudo_post_filter_sigma_px=lambda study: 0.0,
        _cine_crudo_post_filter_gated_sigma_px=lambda study: 0.0,
    )
    _bind(window, "_cine_crudo_recon_config")
    assert window._cine_crudo_recon_config().reconstruction_matrix == (128 if enabled else None)


def test_perfusion_quant_attaches_reviewed_manual_scores_and_resets():
    import numpy as np
    from core.perfusion_quant import perfusion_by_segment, perfusion_quant_summary, perfusion_scores
    segment_map = (np.arange(192) % 17 + 1).reshape(3, 8, 8)
    cube = np.ones((2, 3, 8, 8))
    cube[:, segment_map == 1] = 0.3
    window = SimpleNamespace(study=SimpleNamespace(cube=cube), aha=SimpleNamespace(segment_map=segment_map),
                             seg=None, compare_bundle=None, _rest_stage_state=lambda: None)
    names = dict(np=np, perfusion_by_segment=perfusion_by_segment,
                 perfusion_quant_summary=perfusion_quant_summary, perfusion_scores=perfusion_scores)
    _bind(window, "_perfusion_stage_inputs", **names)
    _bind(window, "_compute_perfusion_quant", **names)
    _bind(window, "_reset_perfusion_scores")
    scores = window._compute_perfusion_quant()["scores"]
    assert scores["rows"][0]["stress_score"] == 4 and scores["sss"] == 4 and not scores["reviewed"]
    window._perfusion_score_overrides = {"stress": {1: 2}, "rest": {}}
    window._perfusion_score_reviewed = True
    scores = window._compute_perfusion_quant()["scores"]
    assert scores["rows"][0]["stress_edited"] and scores["sss"] == 2 and scores["reviewed"]
    window._reset_perfusion_scores()
    assert window._perfusion_score_overrides == {"stress": {}, "rest": {}}
    assert window._compute_perfusion_quant()["scores"]["sss"] == 4
    assert "self._reset_perfusion_scores()" in ast.unparse(METHODS["_reset_session_data"])
    assert "self._reset_perfusion_scores()" in ast.unparse(METHODS["process_current"])
    exporter = ast.unparse(METHODS["_export_structured_results"])
    assert "Exporta resultados" in exporter and "Exporta resultados" not in ast.unparse(METHODS["open_perfusion_score_dialog"])


def test_fine_reconstruction_does_not_reuse_native_ct_mask_or_registration():
    stage = SimpleNamespace(recon_result=SimpleNamespace(voxel_scale=0.5))
    window = SimpleNamespace(_dual_session=lambda: SimpleNamespace(stage=lambda name: stage))
    _bind(window, "_suggest_feta_limits_from_fusion")
    assert window._suggest_feta_limits_from_fusion("stress") is False
    _bind(window, "_stage_ct_on_recon_grid")
    with pytest.raises(ValueError, match="matriz fina"):
        window._stage_ct_on_recon_grid("stress", (128, 128, 128))


@pytest.mark.parametrize("scale,base,apex,size,expected", [
    (1.0, 10, 20, 64, (7, 21)), (0.5, 19, 39, 128, (14, 42)),
])
def test_feta_safety_margin_keeps_physical_width(scale, base, apex, size, expected):
    import numpy as np
    window = SimpleNamespace(_cine_crudo_stage_limits_get=lambda stage, count: (base, apex),
                             cine_crudo_recon_result=SimpleNamespace(voxel_scale=scale))
    _bind(window, "_cine_crudo_cut_bounds", np=np)
    assert window._cine_crudo_cut_bounds(size) == expected


@pytest.mark.parametrize("voxel_scale,sample_scale", [(1.0, 1.0), (0.5, 1.0), (0.5, 0.5)])
def test_generated_sa_metadata_uses_effective_voxel_not_detector_pixel(voxel_scale, sample_scale):
    import numpy as np
    method = METHODS["_generate_cine_crudo_cardiac_cuts"]
    body = next(node for node in method.body if isinstance(node, ast.Try)).body
    start = next(index for index, node in enumerate(body)
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "raw_px")
    end = next(index for index, node in enumerate(body)
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)
               and node.targets[0].attr == "cine_crudo_cut_study")
    study = SimpleNamespace(pixel_spacing=(6.8, 6.8), source_path="test.dcm")
    window = SimpleNamespace(cine_crudo_reoriented_gated=np.ones((2, 8, 16, 16)),
                             cine_crudo_reoriented_voi={"sample_scale": sample_scale},
                             cine_crudo_cut_source_label="test")
    namespace = {"self": window, "raw_study": study, "result": SimpleNamespace(voxel_scale=voxel_scale, notes=[]),
                 "sa_cube": window.cine_crudo_reoriented_gated, "sa_cube_phase": None, "thickness": 2,
                 "z0": 0, "z1": 7,
                 "dicom_loader": SimpleNamespace(GatedStudy=lambda **kwargs: SimpleNamespace(**kwargs))}
    exec(compile(ast.Module(body=body[start:end + 1], type_ignores=[]), "sa_metadata", "exec"), namespace)
    expected = 6.8 * voxel_scale * sample_scale
    cuts = window.cine_crudo_cut_study
    assert cuts.pixel_spacing == (expected, expected)
    assert cuts.z_spacing_mm == cuts.spacing_between_slices_mm == expected
    assert cuts.slice_thickness_mm == 2 * expected
    assert study.pixel_spacing == (6.8, 6.8)


@pytest.mark.parametrize("enabled", [False, True])
def test_ectb_count_compensation_config_and_reset(enabled):
    from core.ectb_lv import ECTbLVConfig
    source = ast.parse((ROOT / "ui" / "ectb_window.py").read_text(encoding="utf-8"))
    window_class = next(node for node in source.body if isinstance(node, ast.ClassDef)
                        and any(isinstance(method, ast.FunctionDef) and method.name == "current_config"
                                for method in node.body))
    methods = [node for node in window_class.body if isinstance(node, ast.FunctionDef)
               and node.name in ("current_config", "reset_to_defaults")]
    namespace = {"ECTbLVConfig": ECTbLVConfig}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "ectb_window.py", "exec"), namespace)
    class Control:
        def __init__(self, value):
            self.current = value
            self.blocked = False
        def value(self):
            return self.current
        currentData = value
        isChecked = value
        def setValue(self, value):
            self.current = value
        setChecked = setValue
        setCurrentText = setValue
        def blockSignals(self, blocked):
            self.blocked = blocked
        def setEnabled(self, enabled):
            self.enabled = enabled
    values = {
        "thickness_spin": 10, "oversample_spin": 4, "angles_combo": 64,
        "median_large_combo": 7, "median_small_combo": 3, "thickening_check": True,
        "dropout_check": True, "normalize_counts_check": enabled, "valve_check": True,
        "valve_offset_spin": 10, "valve_angle_spin": 180, "manual_thickness_check": False,
    }
    calls = []
    window = SimpleNamespace(**{name: Control(value) for name, value in values.items()},
                             recompute=lambda: calls.append("recompute"))
    config = namespace["current_config"](window)
    assert config.normalize_gate_counts is enabled
    assert config.ed_wall_thickness_mm == 10 and config.use_thickening
    namespace["reset_to_defaults"](window)
    assert window.normalize_counts_check.isChecked() is False
    assert window.normalize_counts_check.enabled is True
    assert not window.normalize_counts_check.blocked
    assert calls == ["recompute"]


@pytest.mark.parametrize("enabled", [False, True])
def test_ectb_report_preserves_count_compensation_audit(enabled):
    import numpy as np
    from core.ectb_lv import ECTbLVConfig, ECTbLVResult
    result = ECTbLVResult(
        available=True, config=ECTbLVConfig(normalize_gate_counts=enabled),
        gate_counts=np.array([100., 80., 90.]),
        gate_count_scale_factors=np.array([1., 1.25, 100 / 90]) if enabled else np.ones(3),
        notes=["experimental"] if enabled else [],
    )
    window = SimpleNamespace(
        study=SimpleNamespace(cube=np.ones((3, 3, 5, 5)), pixel_spacing=(6.4, 6.4), z_spacing_mm=6.4),
        seg=object(), ectb_config=lambda: result.config, fevi_regression=lambda: None,
        _apply_gate_dropout_correction=lambda cube, **kwargs: (cube, {}),
    )
    _bind(window, "_estimate_lv_ef_ectb", analyze_lv_ectb=lambda *args: result)
    payload = window._estimate_lv_ef_ectb()
    assert payload["normalize_gate_counts"] is enabled
    assert payload["gate_counts"] == result.gate_counts.tolist()
    assert payload["gate_count_scale_factors"] == result.gate_count_scale_factors.tolist()
    assert payload["notes"] == result.notes


@pytest.mark.parametrize("qc_blocked", [False, True])
def test_ectb_qc_rejection_cannot_be_replaced_by_threshold_fevi(qc_blocked):
    result = {"available": False, "reason": "gate invalid", "qc_blocked": qc_blocked}
    calls = []
    window = SimpleNamespace(
        FEVI_METHOD_THRESHOLD="threshold", fevi_method=lambda: "ectb",
        _estimate_lv_ef_ectb=lambda: result,
        _estimate_lv_ef_preliminary=lambda: calls.append("fallback") or {"available": True},
        _augment_with_filling_metrics=lambda payload: payload, _log=lambda message: None,
    )
    _bind(window, "_estimate_lv_ef")
    payload = window._estimate_lv_ef()
    assert calls == ([] if qc_blocked else ["fallback"])
    if qc_blocked:
        assert payload is result and not payload["available"]
    else:
        assert payload["available"] and payload["fallback_from"] == "ectb_max_counts"


def test_ectb_pdf_shows_experimental_and_qc_warnings():
    from xml.sax.saxutils import escape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, Spacer
    source = ast.parse((ROOT / "report" / "report_generator.py").read_text(encoding="utf-8"))
    generate = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "generate_report")
    warning_loop = next(node for node in generate.body if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Tuple)
                        and any(isinstance(target, ast.Name) and target.id == "stage_label"
                                for target in node.target.elts))
    story = []
    namespace = {
        "ef": {"normalize_gate_counts": True, "notes": ["QC: cuenta < 5 & revisión"]},
        "rest_func_main": {"qc_blocked": True, "reason": "gate vacío"},
        "story": story, "Paragraph": Paragraph, "Spacer": Spacer, "escape": escape,
        "body_style": getSampleStyleSheet()["BodyText"], "mm": 1,
    }
    exec(compile(ast.Module(body=[warning_loop], type_ignores=[]), "report_generator.py", "exec"), namespace)
    text = " ".join(paragraph.getPlainText() for paragraph in story if isinstance(paragraph, Paragraph))
    assert "experimental" in text and "no validada clínicamente" in text
    assert "cuenta < 5 & revisión" in text
    assert "FEVI Reposo: gate vacío" in text


@pytest.mark.parametrize("filename, class_name", [
    ("amyloid_window.py", "AmyloidWindow"),
    ("amyloid_spect_panel.py", "AmyloidSpectPanel"),
])
def test_amyloid_window_controls_minimize_and_close(filename, class_name):
    source = ast.parse((ROOT / "ui" / filename).read_text(encoding="utf-8"))
    window_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    initializer = next(node for node in window_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    start = next(index for index, node in enumerate(initializer.body)
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "window_controls" for target in node.targets))
    end = next(index for index, node in enumerate(initializer.body[start:], start)
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "addLayout"
               and any(isinstance(argument, ast.Name) and argument.id == "window_controls" for argument in node.value.args))
    calls, widgets, layouts = [], [], []
    class Button:
        def __init__(self, parent):
            self.parent = parent
            self.clicked = SimpleNamespace(connect=lambda callback: setattr(self, "callback", callback))
        def setIcon(self, icon):
            self.icon = icon
        def setToolTip(self, tooltip):
            self.tooltip = tooltip
        def setAccessibleName(self, name):
            self.accessible_name = name
        def setFixedSize(self, width, height):
            self.size = (width, height)
    controls = SimpleNamespace(addWidget=lambda widget, *args: widgets.append(widget))
    window = SimpleNamespace(
        _info_lbl=object(), showMinimized=lambda: calls.append("minimize"), close=lambda: calls.append("close"),
        style=lambda: SimpleNamespace(standardIcon=lambda icon: icon),
    )
    namespace = {
        "self": window, "root": SimpleNamespace(addLayout=layouts.append),
        "QHBoxLayout": lambda: controls, "QLabel": lambda title: object(), "QToolButton": Button,
        "QStyle": SimpleNamespace(StandardPixmap=SimpleNamespace(SP_TitleBarMinButton=1, SP_TitleBarCloseButton=2)),
    }
    exec(compile(ast.Module(body=initializer.body[start:end + 1], type_ignores=[]), filename, "exec"), namespace)
    assert layouts == [controls]
    assert len(widgets) == 3
    for button, tooltip, icon in zip(widgets[-2:], ("Minimizar", "Cerrar"), (1, 2)):
        assert button.parent is window
        assert button.tooltip == button.accessible_name == tooltip
        assert button.icon == icon and button.size == (28, 28)
        button.callback()
    assert calls == ["minimize", "close"]


@pytest.mark.parametrize("counts", [(8,), (8, 8), (8, 16), (16, 8)])
def test_lv_3d_stage_context_preserves_data_and_syncs_cycle(counts):
    source = ast.parse((ROOT / "ui" / "lv_3d_panel.py").read_text(encoding="utf-8"))
    dialog = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "LV3DDialog")
    method = next(node for node in dialog.body if isinstance(node, ast.FunctionDef) and node.name == "_for_each_stage")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "lv_3d_panel.py", "exec"), namespace)
    states = [{"_n_gates": count, "_gate": 0, "_meshes": object(), "_actor_gate": None} for count in counts]
    window = SimpleNamespace(_STAGE_FIELDS=tuple(states[0]), _stage_states=states, _gate=counts[0] - 1)
    window._for_each_stage = MethodType(namespace["_for_each_stage"], window)
    observed = []
    def render():
        observed.append((window._stage_index, window._gate, window._meshes))
        window._actor_gate = f"actor-{window._stage_index}"
    window._for_each_stage(render, sync_gate=True)
    assert observed == [(index, (counts[0] - 1) * count // counts[0], states[index]["_meshes"]) for index, count in enumerate(counts)]
    assert [state["_actor_gate"] for state in states] == [f"actor-{index}" for index in range(len(counts))]
    assert window._stage_index == 0 and window._meshes is states[0]["_meshes"]
    assert window._gate == counts[0] - 1


@pytest.mark.parametrize("reversed_stages", [False, True])
def test_lv_3d_open_uses_both_processed_stages(monkeypatch, reversed_stages):
    calls = []
    primary_study, other_study = object(), object()
    primary_seg, other_seg = object(), object()
    dialog = SimpleNamespace(show=lambda: None, raise_=lambda: None, activateWindow=lambda: None)
    module = SimpleNamespace(LV3DDialog=lambda parent, **kwargs: calls.append(kwargs) or dialog)
    monkeypatch.setitem(sys.modules, "ui.lv_3d_panel", module)
    def stage_data(study, seg):
        assert (study, seg) in ((primary_study, primary_seg), (other_study, other_seg))
        return {"lv_meshes": study, "seg": seg}
    window = SimpleNamespace(
        study=primary_study, seg=primary_seg, compare_bundle={"study": other_study, "seg": other_seg},
        _lv_3d_stage_data=stage_data, _dual_compare_labels=lambda: ("Reposo", "Esfuerzo") if reversed_stages else ("Esfuerzo", "Reposo"),
        _log=lambda message: pytest.fail(message),
    )
    _bind(window, "open_lv_3d_window")
    assert window.open_lv_3d_window() is dialog
    assert calls[0]["stage_label"] == "Esfuerzo"
    assert calls[0]["comparison"]["stage_label"] == "Reposo"
    assert calls[0]["lv_meshes"] is (other_study if reversed_stages else primary_study)
    assert calls[0]["comparison"]["lv_meshes"] is (primary_study if reversed_stages else other_study)


def test_lv_3d_offscreen_comparison_renders_both_stages(tmp_path):
    import numpy as np
    import pyvista as pv
    from core.lv_mesh import lv_meshes_from_ectb
    source = ast.parse((ROOT / "ui" / "lv_3d_panel.py").read_text(encoding="utf-8"))
    dialog = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "LV3DDialog")
    namespace = {"QDialog": object, "np": np}
    exec(compile(ast.Module(body=[dialog], type_ignores=[]), "lv_3d_panel.py", "exec"), namespace)
    cls = namespace["LV3DDialog"]
    window = cls.__new__(cls)
    window._stage_states = []
    window._stage_labels = ["Esfuerzo", "Reposo"]
    window._stage_index = 0
    window._cmap = "turbo"
    window._color_low, window._color_high = 0.0, 1.0
    window.combo_surface = SimpleNamespace(currentData=lambda: "shell")
    window.epi_opacity_slider = SimpleNamespace(value=lambda: 100)
    window.lbl_info = SimpleNamespace(setText=lambda value: None)
    for index, gates in enumerate((8, 16)):
        radii = np.broadcast_to(np.linspace(10, 24, 6)[None, :, None], (gates, 6, 64)).copy()
        radii *= (1 - 0.12 * (1 - np.cos(2 * np.pi * np.arange(gates) / gates)))[:, None, None]
        result = SimpleNamespace(available=True, endo_radii_mm=radii, epi_radii_mm=radii + 7,
                                 valid_slices=list(range(6)), ed_gate=1, es_gate=gates // 2 + 1)
        lv = lv_meshes_from_ectb(result, 6.0)
        state = {name: None for name in cls._STAGE_FIELDS}
        state.update(_lv=lv, _n_gates=gates, _ed_gate=0, _gate=0, _meshes=lv["meshes"],
                     _endo_radii=lv["endo_radii_mm"], _epi_radii=lv["epi_radii_mm"],
                     _z_positions=lv["z_positions_mm"], _center_offsets_mm=lv["center_offsets_mm"],
                     _polar_map=np.tile(np.linspace(0.2, 1, 360)[::1 if index == 0 else -1], (220, 1)),
                     _slab_actors=[], _surface_status="", _shell_cache={}, _wireframe_cache={})
        window._stage_states.append(state)
    for name, value in window._stage_states[0].items():
        setattr(window, name, value)
    window._plotter = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1200, 800))
    try:
        window._for_each_stage(window._init_stage_scene, sync_gate=True)
        window._plotter.link_views()
        window._plotter.reset_camera()
        window._set_view("Anterior")
        for index, renderer in enumerate(window._plotter.renderers):
            stage = index % 2
            side = "left" if index < 2 else "right"
            assert any(f"anat_{side}_{stage}" in name for name in renderer.actors)
        image = window._plotter.screenshot(str(tmp_path / "vi_3d_comparison.png"), return_img=True)
        assert image.shape[:2] == (800, 1200)
        for top, left in ((0, 0), (0, 600), (400, 0), (400, 600)):
            region = image[top + 60:top + 370, left + 80:left + 520, :3]
            assert np.count_nonzero(region.max(axis=2) - region.min(axis=2) > 40) > 500
        first_surfaces = [state["_shell_cache"][("shell", 0)][0] for state in window._stage_states]
        window._gate = 4
        window._update_meshes()
        assert [state["_gate"] for state in window._stage_states] == [4, 8]
        systolic_image = window._plotter.screenshot(str(tmp_path / "vi_3d_comparison_systole.png"), return_img=True)
        assert np.count_nonzero(np.any(systolic_image != image, axis=2)) > 1000
        window._gate = 0
        window._update_meshes()
        assert all(state["_shell_cache"][("shell", 0)][0] is first_surfaces[index] for index, state in enumerate(window._stage_states))
        assert window._stage_states[0]["_actor_epi"] is not window._stage_states[1]["_actor_epi"]
    finally:
        window._plotter.close()


def test_lv_3d_polar_maps_are_independent_of_primary_cache():
    import numpy as np
    rows, columns = np.mgrid[-4:5, -4:5]
    angle = np.arctan2(rows, columns)
    mask = np.broadcast_to((np.hypot(rows, columns) >= 2) & (np.hypot(rows, columns) <= 4), (3, 9, 9))
    seg = SimpleNamespace(mask=mask, center_per_slice=np.full((3, 2), 4.0))
    primary = SimpleNamespace(cube=np.broadcast_to(2 + np.cos(angle), (8, 3, 9, 9)))
    secondary = SimpleNamespace(cube=np.broadcast_to(2 + np.sin(angle), (16, 3, 9, 9)))
    cached = np.full((220, 360), 0.99)
    window = SimpleNamespace(study=primary, seg=seg, _polar_perf_cart_cache={"polar_map": cached})
    _bind(window, "_polar_perfusion_map_for_3d", np=np)
    assert window._polar_perfusion_map_for_3d() is cached
    first = window._polar_perfusion_map_for_3d(study=primary, seg=seg)
    second = window._polar_perfusion_map_for_3d(study=secondary, seg=seg)
    assert first.shape == second.shape == cached.shape
    assert not np.allclose(first, second)
    assert window._polar_perf_cart_cache["polar_map"] is cached
    assert window.study is primary and window.seg is seg


def test_lv_3d_stage_data_uses_each_study_spacing_and_volume(monkeypatch):
    import numpy as np
    import core.lv_mesh as mesh_module
    mesh_calls, analysis_calls = [], []
    def analyze(cube, seg, pixel_mm, slice_mm, config):
        analysis_calls.append((cube, seg, pixel_mm, slice_mm))
        return SimpleNamespace(available=True)
    def build(result, slice_mm, **kwargs):
        mesh_calls.append((slice_mm, kwargs))
        return object()
    monkeypatch.setattr(mesh_module, "lv_meshes_from_ectb", build)
    polar = object()
    window = SimpleNamespace(
        _apply_gate_dropout_correction=lambda cube, **kwargs: (cube, {}), ectb_config=lambda: None,
        _polar_perfusion_map_for_3d=lambda **kwargs: polar,
    )
    _bind(window, "_lv_3d_stage_data", np=np, analyze_lv_ectb=analyze)
    for gates, spacing in ((8, (6.4, 4.8, 4.8)), (16, (5.0, 3.2, 3.2))):
        study = SimpleNamespace(cube=np.ones((gates, 3, 5, 7)), z_spacing_mm=spacing[0], pixel_spacing=spacing[1:])
        seg = object()
        data = window._lv_3d_stage_data(study, seg)
        assert data["seg"] is seg and data["polar_map"] is polar
        assert data["spacing_mm"] == spacing and data["pixel_mm"] == spacing[1:]
        assert np.array_equal(data["myo_volume"], study.cube.sum(axis=0))
        assert analysis_calls[-1][1:] == (seg, spacing[1:], spacing[0])
        assert mesh_calls[-1] == (spacing[0], {"surface": "endo", "seg": seg, "pixel_mm": spacing[1:], "volume_shape": (3, 5, 7)})


def test_montage_selection_does_not_write_png_or_gif():
    calls = []
    pix = SimpleNamespace(save=lambda *args: calls.append("save"))
    window = SimpleNamespace(
        _montage_gray_cache={"rows_meta": [{"selection_key": "REPOSO:SA", "selected": False}]},
        cine_crudo_selected_stripes={"REPOSO:SA"}, _montage_panel_px=512,
        _montage_pix_from_gray=lambda cmap: pix, preview_labels={},
        output_dir="unused", _montage_cine_frames=[pix, pix],
    )
    _bind(window, "_recolor_montage_from_cache")
    _bind(window, "_refresh_montage_selection_overlay")
    window._refresh_montage_selection_overlay()
    assert window._montage_gray_cache["rows_meta"][0]["selected"]
    assert calls == []


@pytest.mark.parametrize("zoom, ratio, expected", [(0.2, 1, 160), (0.2, 2, 320), (0.5, 1, 384), (1, 1, 512), (2, 2, 512)])
def test_montage_render_resolution_tracks_visible_pixels(zoom, ratio, expected):
    import numpy as np
    window = SimpleNamespace(preview_zoom={"comparacion_ejes": zoom}, preview_labels={"comparacion_ejes": SimpleNamespace(devicePixelRatioF=lambda: ratio)})
    _bind(window, "_montage_display_panel_px", np=np)
    assert window._montage_display_panel_px() == expected


def test_montage_settle_reuses_render_at_visible_resolution():
    calls = []
    window = SimpleNamespace(
        cine_crudo_preview_mode="sa_montage", _montage_gray_cache={"geom": (160,)},
        _montage_last_signature="current", _montage_signature=lambda: "current",
        _montage_display_panel_px=lambda: 160, preview_labels={"comparacion_ejes": object()},
        _apply_preview_zoom=lambda name: calls.append("smooth-display"),
        _show_cine_crudo_sa_montage=lambda **kwargs: calls.append("rebuild"),
    )
    _bind(window, "_render_montage_hq")
    window._render_montage_hq()
    assert calls == ["smooth-display"] and window._montage_panel_px == 160


def test_montage_overlay_reuses_color_pixmap_without_lut_or_files():
    calls = []
    base = SimpleNamespace(copy=lambda: calls.append("copy") or object())
    window = SimpleNamespace(
        _montage_gray_cache={"base_pixmap": base}, preview_labels={}, _montage_panel_px=512,
        _paint_montage_overlays=lambda *args: calls.append("overlay"),
        _montage_pix_from_gray=lambda *args: pytest.fail("selection recomputed LUT"),
    )
    _bind(window, "_recolor_montage_from_cache")
    window._recolor_montage_from_cache("cool", persist=False, overlays_only=True)
    assert calls == ["copy", "overlay"]


@pytest.mark.parametrize("move_plane", [False, True])
def test_montage_triangulation_updates_only_changed_planes(move_plane):
    import numpy as np
    from core.cardiac_reorientation import anatomical_cuts_gated, triangulation_views
    calls = []
    class Pixmap:
        def __init__(self, pixels):
            self.pixels = pixels
        def copy(self):
            return Pixmap(self.pixels.copy())
    class Image:
        Format = SimpleNamespace(Format_RGB888=1)
        def __init__(self, buffer, width, height, stride, format):
            self.pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3).copy()
    class Painter:
        def __init__(self, pixmap):
            self.pixmap = pixmap
        def drawImage(self, left, top, image):
            calls.append((left, top))
            height, width = image.pixels.shape[:2]
            self.pixmap.pixels[top:top + height, left:left + width] = image.pixels
        def end(self):
            pass
    volumes = {key.upper(): value[0] / 315 for key, value in anatomical_cuts_gated(np.arange(315).reshape(1, 5, 7, 9)).items()}
    stages = {stage: volumes for stage in ("ESFUERZO", "REPOSO")}
    metadata = []
    for column, (stage, stage_volumes) in enumerate(stages.items()):
        for row, (axis, view) in enumerate(triangulation_views(stage_volumes, [0.5] * 3).items()):
            metadata.append({"stage": stage, "axis": axis, "index": view["index"], "cross": view["cross"],
                             "image_geom": (120 + column * 60, row * 60, 50, 50)})
    gray = np.full((180, 240), 12, dtype=np.uint8)
    original = gray.copy()
    pixels = np.repeat(gray[..., None], 3, axis=2)
    point = [0.25, 0.5, 0.5] if move_plane else [0.5, 0.52, 0.5]
    lut = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    window = SimpleNamespace(
        _montage_gray_cache={"geom": (160,), "image_signature": "current", "gate": None,
                            "tri_views": metadata, "gray": gray, "base_pixmap": Pixmap(pixels)},
        _montage_panel_px=160, _montage_signature=lambda **kwargs: "current",
        _montage_tri_volumes=stages, _montage_tri_points={stage: point for stage in stages},
        _montage_tri_stage="REPOSO", _montage_cmap_lut=lambda *args: calls.append("LUT") or lut,
        _refresh_montage_selection_overlay=lambda: calls.append("overlay"),
    )
    _bind(window, "_update_montage_triangulation_from_cache", np=np, QImage=Image, QPainter=Painter)
    assert window._update_montage_triangulation_from_cache()
    np.testing.assert_array_equal(gray[:, :120], original[:, :120])
    np.testing.assert_array_equal(window._montage_gray_cache["base_pixmap"].pixels[:, :120], pixels[:, :120])
    assert calls == (["LUT", (120, 0), (180, 0), "overlay"] if move_plane else ["overlay"])
    for view in metadata:
        expected = triangulation_views(volumes, point)[view["axis"]]
        assert view["index"] == expected["index"] and view["cross"] == expected["cross"]
    if move_plane:
        assert np.any(gray[:50, 120:170] != original[:50, 120:170])
        np.testing.assert_array_equal(gray[60:, :], original[60:, :])
        np.testing.assert_array_equal(window._montage_gray_cache["base_pixmap"].pixels[:50, 120:170, 0], gray[:50, 120:170])


@pytest.mark.parametrize("reason", ["window", "resolution", "gate"])
def test_montage_partial_update_invalidates_changed_image_state(reason):
    window = SimpleNamespace(
        _montage_gray_cache={"tri_views": [object()], "base_pixmap": object(), "geom": (160,), "image_signature": "current", "gate": None},
        _montage_panel_px=128 if reason == "resolution" else 160,
        _montage_signature=lambda **kwargs: "new" if reason == "window" else "current",
        cine_crudo_montage_source="gated" if reason == "gate" else "ungated",
        cine_crudo_montage_cine_playing=reason == "gate",
    )
    _bind(window, "_update_montage_triangulation_from_cache")
    assert not window._update_montage_triangulation_from_cache()


def test_montage_return_to_tab_prefers_live_image_to_stale_png():
    calls = []
    window = SimpleNamespace(
        cine_crudo_preview_mode="sa_montage", preview_labels={"comparacion_ejes": object()},
        preview_pixmaps={"comparacion_ejes": object()}, _apply_preview_zoom=lambda name: calls.append(name),
    )
    _bind(window, "_load_compare_axes_preview")
    window._load_compare_axes_preview()
    assert calls == ["comparacion_ejes"]


def test_montage_png_export_renders_latest_state_in_hq(tmp_path):
    calls = []
    source = tmp_path / "sa_montage.png"
    target = tmp_path / "export.png"
    source.write_bytes(b"previous-state")
    window = SimpleNamespace(
        cine_crudo_axes_for_export=True, cine_crudo_preview_mode="sa_montage", output_dir=str(tmp_path),
        _montage_panel_px=160, _dialog_parent=lambda: None, _log=lambda message: None,
    )
    def render(**kwargs):
        assert window._montage_panel_px == 512 and window._montage_cine_frames_sig is None
        assert kwargs == {"navigate": False, "persist": True}
        source.write_bytes(b"current-state-hq")
        calls.append("render")
    window._show_cine_crudo_sa_montage = render
    _bind(window, "_export_cine_crudo_montage_png",
          QFileDialog=SimpleNamespace(getSaveFileName=lambda *args: (str(target), "")),
          QMessageBox=SimpleNamespace(information=lambda *args: None, warning=lambda *args: pytest.fail(str(args))))
    window._export_cine_crudo_montage_png()
    assert target.read_bytes() == b"current-state-hq" and calls == ["render"]


def test_montage_pdf_report_materializes_current_hq_image():
    method = METHODS["_generate_pdf_report"]
    stop = next(index for index, statement in enumerate(method.body) if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "pdf_path" for target in statement.targets))
    selected = ast.parse("def prepare_report(self):\n    pass").body[0]
    assert isinstance(selected, ast.FunctionDef)
    selected.body = method.body[:stop]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])), "<report-preparation>", "exec"), namespace)
    calls = []
    window = SimpleNamespace(study=object(), seg=object(), cine_crudo_preview_mode="sa_montage", cine_crudo_axes_for_export=True,
                             _show_cine_crudo_sa_montage=lambda **kwargs: calls.append(kwargs))
    namespace["prepare_report"](window)
    assert window._montage_panel_px == 512
    assert calls == [{"navigate": False, "persist": True}]


def test_montage_signature_separates_cursor_from_actual_image_changes():
    window = SimpleNamespace(cine_crudo_axes_for_export={}, cine_crudo_axes_for_export_ungated={"SA": object()})
    _bind(window, "_montage_signature")
    initial = window._montage_signature(include_navigation=False)
    window._montage_tri_points = {"ESFUERZO": [0.2, 0.3, 0.4]}
    window._montage_tri_focus = "HLA"
    window.cine_crudo_selected_stripes = {"REPOSO:SA"}
    assert window._montage_signature(include_navigation=False) == initial
    window.cine_crudo_axes_for_export_ungated = {"SA": object()}
    assert window._montage_signature(include_navigation=False) != initial
    initial = window._montage_signature(include_navigation=False)
    window.cine_crudo_montage_lin_high = 0.5
    assert window._montage_signature(include_navigation=False) != initial


def test_montage_direct_lut_keeps_masked_background_and_clean_overlay_base():
    import numpy as np
    class Image:
        Format = SimpleNamespace(Format_RGB888=1)
        def __init__(self, buffer, width, height, stride, format):
            self.pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3).copy()
    class Pixmap:
        def __init__(self, pixels):
            self.pixels = pixels
        @staticmethod
        def fromImage(image):
            return Pixmap(image.pixels)
        def copy(self):
            return Pixmap(self.pixels.copy())
    gray = np.array([[0, 12, 40], [80, 120, 255]], dtype=np.uint8)
    mask = np.array([[False, True, True], [True, False, True]])
    lut = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    lut[0] = [12, 24, 36]
    expected = np.zeros((2, 3, 3), dtype=np.uint8)
    expected[mask] = lut[gray[mask]]
    window = SimpleNamespace(_montage_gray_cache={"gray": gray, "mask": mask}, _montage_cmap_lut=lambda name: lut,
                             _paint_montage_overlays=lambda pix, cache: pix.pixels.__setitem__((0, 0), [255, 0, 0]))
    _bind(window, "_montage_pix_from_gray", np=np, QImage=Image, QPixmap=Pixmap)
    pix = window._montage_pix_from_gray("cool")
    np.testing.assert_array_equal(window._montage_gray_cache["base_pixmap"].pixels, expected)
    assert list(pix.pixels[0, 0]) == [255, 0, 0]
    assert list(window._montage_gray_cache["base_pixmap"].pixels[0, 0]) == [0, 0, 0]


@pytest.mark.parametrize("panel", [128, 160, 256, 512])
@pytest.mark.parametrize("zoom", [0.2, 0.5, 1.0])
def test_montage_cached_hq_cine_preserves_zoom_after_interactive_render(panel, zoom):
    class Size:
        def __init__(self, width, height):
            self._width, self._height = width, height
        def width(self):
            return self._width
        def height(self):
            return self._height
    frames = [SimpleNamespace(size=lambda: Size(7200, 3600), width=lambda: 7200, height=lambda: 3600) for _ in range(8)]
    displayed = []
    window = SimpleNamespace(
        _montage_panel_px=panel, _montage_cine_frames=frames,
        preview_labels={"comparacion_ejes": object()}, preview_pixmaps={}, preview_base_sizes={},
        preview_zoom={"comparacion_ejes": zoom}, _select_tab_by_title=lambda name: None,
    )
    def display(name, fast=False):
        base = window.preview_base_sizes[name]
        displayed.append((round(base.width() * window.preview_zoom[name]), round(base.height() * window.preview_zoom[name])))
    window._apply_preview_zoom = display
    _bind(window, "_montage_display_base_size", QSize=Size)
    _bind(window, "_blit_montage_cine_frame")
    for index in range(8):
        window.cine_crudo_montage_cine_frame = index
        window._blit_montage_cine_frame()
        assert window.preview_pixmaps["comparacion_ejes"] is frames[index]
    assert displayed == [(round(7200 * zoom), round(3600 * zoom))] * 8
    assert window.preview_zoom["comparacion_ejes"] == zoom
    assert window._montage_panel_px == panel


@pytest.mark.parametrize("source_stage", ["ESFUERZO", "REPOSO"])
@pytest.mark.parametrize("offset", [0, 1])
def test_linked_triangulation_points_preserve_corresponding_planes(source_stage, offset):
    import numpy as np
    from core.cardiac_reorientation import linked_triangulation_points, triangulation_views
    volumes = {
        "ESFUERZO": {axis: np.zeros((5, 5, 5)) for axis in ("SA", "VLA", "HLA")},
        "REPOSO": {axis: np.zeros((9, 9, 9)) for axis in ("SA", "VLA", "HLA")},
    }
    point = [0.5 + offset / 8] * 3 if source_stage == "REPOSO" else [0.5] * 3
    linked = linked_triangulation_points(volumes, source_stage, point, {axis: offset for axis in ("SA", "VLA", "HLA")})
    assert linked["ESFUERZO"] == [0.5] * 3
    assert linked["REPOSO"] == [0.5 + offset / 8] * 3
    for view in triangulation_views(volumes["ESFUERZO"], linked["ESFUERZO"]).values():
        assert view["index"] == 2
    for view in triangulation_views(volumes["REPOSO"], linked["REPOSO"]).values():
        assert view["index"] == 4 + offset


def test_triangulation_points_match_anatomical_slices():
    import numpy as np
    from core.cardiac_reorientation import anatomical_cuts_gated, triangulation_views, triangulation_point
    cube = np.zeros((1, 5, 7, 9))
    cube[0, 1, 4, 6] = 99.0
    axes = {key.upper(): value[0] for key, value in anatomical_cuts_gated(cube).items()}
    point = [1 / 4, 4 / 6, 6 / 8]
    views = triangulation_views(axes, point)
    for axis, view in views.items():
        image = view["image"]
        horizontal, vertical = view["cross"]
        assert image[round(vertical * (image.shape[0] - 1)), round(horizontal * (image.shape[1] - 1))] == 99.0
        np.testing.assert_allclose(triangulation_point(axis, horizontal, vertical, point), point)


@pytest.mark.parametrize("axis, dimension", [("SA", 0), ("VLA", 2), ("HLA", 1)])
@pytest.mark.parametrize("step", [-1, 1, -100, 100])
def test_triangulation_wheel_clamps_without_moving_strips(axis, dimension, step):
    import numpy as np
    calls = []
    window = SimpleNamespace(
        _montage_tri_focus=axis, _montage_tri_stage="REPOSO",
        _montage_tri_volumes={"REPOSO": {axis: np.zeros((5, 7, 9))}},
        _montage_tri_points={"REPOSO": [0.5, 0.5, 0.5], "ESFUERZO": [0.2, 0.3, 0.4]},
        cine_crudo_stripe_start_by_stage={"REPOSO": {"SA": 3}},
        _schedule_montage_refresh=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    _bind(window, "_set_montage_triangulation_point")
    _bind(window, "_navigate_montage_triangulation", np=np)
    assert window._navigate_montage_triangulation(step)
    expected = [0.5, 0.5, 0.5]
    expected[dimension] = float(np.clip(2 - step, 0, 4)) / 4
    assert window._montage_tri_points["REPOSO"] == expected
    assert window._montage_tri_points["ESFUERZO"] == [0.2, 0.3, 0.4]
    assert window.cine_crudo_stripe_start_by_stage == {"REPOSO": {"SA": 3}}
    assert len(calls) == 1


@pytest.mark.parametrize("zoom", [0.25, 1.0, 2.0])
def test_triangulation_hit_test_uses_scaled_letterbox_geometry(zoom):
    view = {"axis": "HLA", "box": (800, 300, 200, 200), "image_geom": (800, 350, 200, 100)}
    label = SimpleNamespace(width=lambda: 1100 * zoom, height=lambda: 700 * zoom)
    window = SimpleNamespace(_montage_gray_cache={"geom": (150, 4, 16, 100, 40, 154, 170, 1, 1100, 700), "tri_views": [view]}, preview_labels={})
    _bind(window, "_montage_triangulation_at_event")
    event = SimpleNamespace(pos=lambda: SimpleNamespace(x=lambda: 900 * zoom, y=lambda: 400 * zoom))
    assert window._montage_triangulation_at_event(event, label) == (view, 900.0, 400.0)
    event.pos = lambda: SimpleNamespace(x=lambda: 400 * zoom, y=lambda: 400 * zoom)
    assert window._montage_triangulation_at_event(event, label) is None


def test_montage_fast_timer_coalesces_without_starving_updates():
    calls = []
    active = [False]
    def start(interval):
        calls.append(("render", interval))
        active[0] = True
    window = SimpleNamespace(
        cine_crudo_preview_mode="sa_montage",
        _montage_display_panel_px=lambda: 512,
        _montage_refresh_timer=SimpleNamespace(start=start, isActive=lambda: active[0]),
        _montage_hq_timer=SimpleNamespace(start=lambda interval: calls.append(("hq", interval)), stop=lambda: calls.append("stop")),
    )
    _bind(window, "_schedule_montage_refresh")
    for _ in range(20):
        window._schedule_montage_refresh(8, fast=True)
    assert calls.count(("render", 16)) == 1
    assert calls.count(("hq", 400)) == 20
    window._schedule_montage_refresh(0)
    assert window._montage_panel_px == 512
    assert calls[-2:] == ["stop", ("render", 0)]


@pytest.mark.parametrize("dual", [False, True])
def test_montage_compositor_reuses_resized_tiles_and_fits_triangulation(monkeypatch, dual):
    import numpy as np
    from PIL import Image
    calls = []
    original = Image.Image.resize
    def resize(image, *args, **kwargs):
        calls.append(image.size)
        return original(image, *args, **kwargs)
    monkeypatch.setattr(Image.Image, "resize", resize)
    image = np.ones((7, 9)) * 0.7
    views = {axis: {"image": image, "index": 2, "cross": (0.25, 0.75), "cache_key": axis} for axis in ("SA", "VLA", "HLA")}
    rows = [{"tag": "", "prefix": "SA", "selected": True, "used_cols": 1,
             "panels": [{"col": 0, "img": image, "cache_key": "cut"}]}] * 3
    window = SimpleNamespace(_montage_panel_px=192, _montage_tri_views=views)
    if dual:
        rows *= 2
        rest = {axis: {**view, "image": image + 0.1, "cache_key": ("REPOSO", axis)} for axis, view in views.items()}
        window._montage_tri_views_by_stage = {"ESFUERZO": views, "REPOSO": rest}
    window._montage_pix_from_gray = lambda cmap: window._montage_gray_cache["gray"]
    _bind(window, "_composite_montage_pixmap", np=np)
    first = window._composite_montage_pixmap(rows, 1, "cool", "Patient").copy()
    assert len(calls) == (7 if dual else 4)
    second = window._composite_montage_pixmap(rows, 1, "cool", "Patient")
    assert len(calls) == (7 if dual else 4)
    np.testing.assert_array_equal(first, second)
    metadata = window._montage_gray_cache["tri_views"]
    assert len(metadata) == (6 if dual else 3)
    if dual:
        for stress_view, rest_view in zip(metadata[:3], metadata[3:]):
            assert stress_view["stage"] == "ESFUERZO" and rest_view["stage"] == "REPOSO"
            assert stress_view["axis"] == rest_view["axis"]
            assert stress_view["box"][0] + stress_view["box"][2] < rest_view["box"][0]
            assert stress_view["box"][1:] == rest_view["box"][1:]
    for view in window._montage_gray_cache["tri_views"]:
        left, top, width, height = view["image_geom"]
        assert left + width <= second.shape[1] and top + height <= second.shape[0]
        assert second[top:top + height, left:left + width].min() > 0
    window._montage_panel_px = 512
    window._composite_montage_pixmap(rows, 1, "cool", "Patient")
    assert len(calls) == (19 if dual else 10)


@pytest.mark.parametrize("stage", ["ESFUERZO", "REPOSO"])
@pytest.mark.parametrize("axis, dimension", [("SA", 0), ("HLA", 1), ("VLA", 2)])
def test_dual_triangulation_wheel_updates_both_stages(stage, axis, dimension):
    import numpy as np
    calls = []
    window = SimpleNamespace(
        _montage_tri_focus=axis, _montage_tri_stage=stage,
        _montage_tri_volumes={phase: {name: np.zeros((9, 9, 9)) for name in ("SA", "HLA", "VLA")} for phase in ("ESFUERZO", "REPOSO")},
        _montage_tri_points={"ESFUERZO": [0.5] * 3, "REPOSO": [0.625] * 3},
        cine_crudo_rest_offset={"SA": 1, "HLA": 1, "VLA": 1},
        cine_crudo_stripe_start_by_stage={"ESFUERZO": {"SA": 2}, "REPOSO": {"SA": 3}},
        _schedule_montage_refresh=lambda *args, **kwargs: calls.append("render"),
    )
    _bind(window, "_set_montage_triangulation_point")
    _bind(window, "_navigate_montage_triangulation", np=np)
    assert window._navigate_montage_triangulation(1)
    expected = [0.5] * 3
    expected[dimension] = 0.375
    assert window._montage_tri_points["ESFUERZO"] == expected
    np.testing.assert_allclose(window._montage_tri_points["REPOSO"], np.asarray(expected) + 0.125)
    assert window.cine_crudo_stripe_start_by_stage == {"ESFUERZO": {"SA": 2}, "REPOSO": {"SA": 3}}
    assert calls == ["render"]


@pytest.mark.parametrize("stage", ["ESFUERZO", "REPOSO"])
@pytest.mark.parametrize("axis", ["SA", "VLA", "HLA"])
def test_dual_triangulation_click_selects_phase_and_links_point(stage, axis):
    import numpy as np
    from core.cardiac_reorientation import triangulation_point
    calls = []
    view = {"axis": axis, "stage": stage, "image_geom": (400, 200, 100, 100)}
    qt = SimpleNamespace(MouseButton=SimpleNamespace(MiddleButton=2, LeftButton=1), KeyboardModifier=SimpleNamespace(AltModifier=4, ControlModifier=1))
    window = SimpleNamespace(
        cine_crudo_preview_mode="sa_montage", _montage_tri_stage="ESFUERZO",
        _montage_tri_points={"ESFUERZO": [0.5] * 3, "REPOSO": [0.5] * 3},
        _montage_tri_volumes={phase: {name: np.zeros((5, 7, 9)) for name in ("SA", "HLA", "VLA")} for phase in ("ESFUERZO", "REPOSO")},
        _montage_triangulation_at_event=lambda *args, **kwargs: (view, 425, 260),
        _schedule_montage_refresh=lambda *args, **kwargs: calls.append("render"),
        _log=lambda message: pytest.fail(message),
    )
    event = SimpleNamespace(button=lambda: 1, modifiers=lambda: 0, accept=lambda: calls.append("accepted"))
    _bind(window, "_set_montage_triangulation_point")
    _bind(window, "_on_cine_crudo_mouse_press_safe", np=np, Qt=qt)
    window._on_cine_crudo_mouse_press_safe(event)
    expected = triangulation_point(axis, 25 / 99, 60 / 99, [0.5] * 3)
    assert window._montage_tri_stage == stage and window._montage_tri_focus == axis
    assert window._montage_tri_points == {"ESFUERZO": expected, "REPOSO": expected}
    assert calls == ["render", "accepted"]


def test_triangulation_handles_isotropic_long_axis_stacks():
    import numpy as np
    from scipy.ndimage import zoom
    from core.cardiac_reorientation import anatomical_cuts_gated, triangulation_views
    cube = np.zeros((1, 5, 7, 9))
    cube[0, 1, 4, 6] = 1.0
    original = anatomical_cuts_gated(cube)
    isotropic = anatomical_cuts_gated(zoom(cube, (1, 13 / 5, 1, 1), order=1))
    axes = {"SA": original["sa"][0], "VLA": isotropic["vla"][0], "HLA": isotropic["hla"][0]}
    for view in triangulation_views(axes, [1 / 4, 4 / 6, 6 / 8]).values():
        image = view["image"]
        horizontal, vertical = view["cross"]
        assert image[round(vertical * (image.shape[0] - 1)), round(horizontal * (image.shape[1] - 1))] == 1.0


@pytest.mark.parametrize("selection", [None, set(), {"ESFUERZO:VLA"}])
def test_montage_normalization_reused_for_strips_but_invalidated_by_window(monkeypatch, selection):
    import numpy as np
    from core.cardiac_reorientation import anatomical_cuts_gated
    axes = {key.upper(): value for key, value in anatomical_cuts_gated(np.arange(315).reshape(1, 5, 7, 9)).items()}
    calls = []
    percentiles = []
    original = np.percentile
    def percentile(*args, **kwargs):
        percentiles.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(np, "percentile", percentile)
    window = SimpleNamespace(
        cine_crudo_montage_source="ungated", cine_crudo_montage_crop_mode="limits",
        cine_crudo_rest_offset={"SA": 0, "VLA": 0, "HLA": 0},
        cine_crudo_stripe_start_by_stage={"ESFUERZO": {"SA": 1, "VLA": 1, "HLA": 1}}, MONTAGE_LAYOUTS={"denso": {"per_strip": 3}},
        _montage_axes_for_source=lambda: (axes, {}), _cine_crudo_cut_thickness_px=lambda: 1,
        _patient_banner_text=lambda **kwargs: "Patient", _montage_signature=lambda: (),
        _composite_montage_pixmap=lambda rows, *args, **kwargs: calls.append(rows) or object(),
        _last_cine_crudo_preview_mode="sa_montage", _montage_panel_px=256,
        preview_labels={}, _log=lambda message: None,
    )
    if selection is not None:
        window.cine_crudo_selected_stripes = selection
    def fail(*args):
        pytest.fail(str(args))
    _bind(window, "_set_montage_triangulation_point")
    _bind(window, "_show_cine_crudo_sa_montage", np=np, QMessageBox=SimpleNamespace(information=fail, warning=fail))
    window._show_cine_crudo_sa_montage(navigate=False)
    assert len(percentiles) == 6
    volumes = dict(window._montage_tri_volumes["ESFUERZO"])
    window.cine_crudo_stripe_start_by_stage["ESFUERZO"]["SA"] = 2
    window._show_cine_crudo_sa_montage(navigate=False)
    assert len(percentiles) == 6
    expected_selected = {"ESFUERZO:SA"} if selection is None else selection
    assert {row["selection_key"] for row in calls[0] if row["selected"]} == expected_selected
    assert {row["selection_key"] for row in calls[1] if row["selected"]} == expected_selected
    assert all(window._montage_tri_volumes["ESFUERZO"][axis] is volume for axis, volume in volumes.items())
    assert len(calls) == 2 and calls[0][0]["idxs"] != calls[1][0]["idxs"]
    window.cine_crudo_montage_win_high = 95.0
    window._show_cine_crudo_sa_montage(navigate=False)
    assert len(percentiles) == 12
    assert all(window._montage_tri_volumes["ESFUERZO"][axis] is not volume for axis, volume in volumes.items())


@pytest.mark.parametrize("selected, ctrl, expected", [
    ({"REPOSO:SA"}, True, set()),
    ({"REPOSO:SA", "ESFUERZO:SA"}, True, {"ESFUERZO:SA"}),
    (set(), True, {"REPOSO:SA"}),
    ({"ESFUERZO:SA"}, True, {"ESFUERZO:SA", "REPOSO:SA"}),
    ({"ESFUERZO:SA"}, False, {"REPOSO:SA"}),
])
def test_montage_strip_click_can_deselect_last_strip(selected, ctrl, expected):
    rows = [
        {"selection_key": "REPOSO:SA", "prefix": "SA", "tag": "REPOSO", "selected": "REPOSO:SA" in selected},
        {"selection_key": "ESFUERZO:SA", "prefix": "SA", "tag": "ESFUERZO", "selected": "ESFUERZO:SA" in selected},
    ]
    refreshes = []
    messages = []
    qt = SimpleNamespace(MouseButton=SimpleNamespace(MiddleButton=2, LeftButton=1), KeyboardModifier=SimpleNamespace(AltModifier=4, ControlModifier=1))
    window = SimpleNamespace(
        cine_crudo_preview_mode="sa_montage", cine_crudo_selected_stripes=set(selected),
        _montage_gray_cache={"rows_meta": rows},
        _montage_triangulation_at_event=lambda *args, **kwargs: None,
        _montage_selection_key_at_event=lambda *args, **kwargs: "REPOSO:SA",
        _recolor_montage_from_cache=lambda *args, **kwargs: refreshes.append(kwargs),
        _schedule_montage_refresh=lambda *args, **kwargs: None,
        cine_crudo_axes_for_export_rest={}, _log=messages.append,
        statusBar=lambda: SimpleNamespace(showMessage=lambda *args: None),
    )
    event = SimpleNamespace(button=lambda: 1, modifiers=lambda: int(ctrl),
                            pos=lambda: SimpleNamespace(x=lambda: 100), accept=lambda: None)
    _bind(window, "_refresh_montage_selection_overlay")
    _bind(window, "_on_cine_crudo_mouse_press_safe", Qt=qt)
    window._on_cine_crudo_mouse_press_safe(event)
    assert window.cine_crudo_selected_stripes == expected
    assert {row["selection_key"] for row in rows if row["selected"]} == expected
    assert refreshes == [{"persist": False, "overlays_only": True}]
    assert not any("[WARN]" in message for message in messages)


@pytest.mark.parametrize("focused", [True, False])
def test_montage_wheel_routes_to_selected_view_or_strips(focused):
    import numpy as np
    calls = []
    qt = SimpleNamespace(KeyboardModifier=SimpleNamespace(ControlModifier=1, NoModifier=0))
    window = SimpleNamespace(
        cine_crudo_preview_mode="sa_montage", _montage_tri_focus="SA" if focused else None,
        _montage_tri_stage="REPOSO", _montage_tri_volumes={"REPOSO": {"SA": np.zeros((5, 7, 9))}},
        _montage_tri_points={"REPOSO": [0.5, 0.5, 0.5]},
        cine_crudo_selected_stripes={"REPOSO:SA"}, cine_crudo_stripe_start_by_stage={"REPOSO": {"SA": 5}},
        _schedule_montage_refresh=lambda *args, **kwargs: calls.append("render"),
        _log=lambda message: pytest.fail(message),
    )
    event = SimpleNamespace(modifiers=lambda: 0, angleDelta=lambda: SimpleNamespace(y=lambda: 120), accept=lambda: calls.append("accepted"))
    _bind(window, "_set_montage_triangulation_point")
    _bind(window, "_navigate_montage_triangulation", np=np)
    _bind(window, "_on_cine_crudo_mouse_wheel_safe", np=np, Qt=qt)
    window._on_cine_crudo_mouse_wheel_safe(event)
    assert window._montage_tri_points["REPOSO"][0] == (0.25 if focused else 0.5)
    assert window.cine_crudo_stripe_start_by_stage["REPOSO"]["SA"] == (5 if focused else 4)
    assert calls == ["render", "accepted"]


@pytest.mark.parametrize("axis", ["SA", "VLA", "HLA"])
def test_triangulation_click_routes_before_strip_selection(axis):
    import numpy as np
    from core.cardiac_reorientation import triangulation_point
    calls = []
    view = {"axis": axis, "image_geom": (400, 200, 100, 100)}
    qt = SimpleNamespace(MouseButton=SimpleNamespace(MiddleButton=2, LeftButton=1), KeyboardModifier=SimpleNamespace(AltModifier=4, ControlModifier=1))
    window = SimpleNamespace(
        cine_crudo_preview_mode="sa_montage", _montage_tri_stage="REPOSO",
        _montage_tri_points={"REPOSO": [0.5, 0.5, 0.5]},
        _montage_triangulation_at_event=lambda *args, **kwargs: (view, 425, 260),
        _schedule_montage_refresh=lambda *args, **kwargs: calls.append("render"),
        _montage_drag_axis="SA", _montage_drag_mode=None, _montage_drag_start_x=0,
        _log=lambda message: pytest.fail(message),
    )
    event = SimpleNamespace(button=lambda: 1, modifiers=lambda: 0, accept=lambda: calls.append("accepted"))
    _bind(window, "_set_montage_triangulation_point")
    _bind(window, "_on_cine_crudo_mouse_press_safe", np=np, Qt=qt)
    window._on_cine_crudo_mouse_press_safe(event)
    assert window._montage_tri_focus == axis and window._montage_drag_axis is None
    assert window._montage_tri_points["REPOSO"] == triangulation_point(axis, 25 / 99, 60 / 99, [0.5, 0.5, 0.5])
    assert calls == ["render", "accepted"]


@pytest.mark.parametrize("cached", [True, False])
def test_montage_cine_uses_updated_geometry_instead_of_old_frames(cached):
    calls = []
    window = SimpleNamespace(
        cine_crudo_montage_cine_playing=True, cine_crudo_montage_cine_frame=0,
        _montage_cine_frames=[object(), object()], _montage_cine_frames_sig="new" if cached else "old",
        _montage_cine_signature=lambda: "new", _blit_montage_cine_frame=lambda: calls.append("blit"),
        _show_cine_crudo_sa_montage=lambda **kwargs: calls.append("render"),
        _montage_display_panel_px=lambda: 160,
    )
    _bind(window, "_advance_montage_cine_frame")
    window._advance_montage_cine_frame()
    assert calls == ["blit" if cached else "render"]
    assert window.cine_crudo_montage_cine_frame == 1


def test_montage_cine_timer_does_not_reenter_during_preload():
    window = SimpleNamespace(_montage_cine_preloading=True)
    _bind(window, "_advance_montage_cine_frame")
    window._advance_montage_cine_frame()


def test_triangulation_double_click_recenters_without_resetting_strips():
    calls = []
    window = SimpleNamespace(
        cine_crudo_preview_mode="sa_montage", _montage_tri_stage="REPOSO",
        _montage_tri_points={"REPOSO": [0.1, 0.2, 0.3]},
        _montage_triangulation_at_event=lambda *args, **kwargs: ({"axis": "VLA"}, 0, 0),
        _schedule_montage_refresh=lambda *args, **kwargs: calls.append("render"),
        cine_crudo_stripe_start_by_stage={"REPOSO": {"VLA": 4}},
    )
    _bind(window, "_set_montage_triangulation_point")
    _bind(window, "_on_cine_crudo_mouse_double_click_safe")
    window._on_cine_crudo_mouse_double_click_safe(SimpleNamespace(accept=lambda: calls.append("accepted")))
    assert window._montage_tri_focus == "VLA"
    assert window._montage_tri_points["REPOSO"] == [0.5, 0.5, 0.5]
    assert window.cine_crudo_stripe_start_by_stage == {"REPOSO": {"VLA": 4}}
    assert calls == ["render", "accepted"]


@pytest.mark.parametrize("mode, minimized", [("modern", False), ("plus", False), ("", True), ("", False)])
def test_second_launch_recovers_active_ui_without_creating_or_docking(mode, minimized):
    source = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    method = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_activate_existing_window")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<activate-existing>", "exec"), namespace)
    calls = []
    window = SimpleNamespace(
        _active_detached_console=mode,
        bring_epar_plus_console_to_front=lambda: calls.append("console"),
        isMinimized=lambda: minimized, showNormal=lambda: calls.append("restore"),
        show=lambda: calls.append("show"), raise_=lambda: calls.append("raise"),
        activateWindow=lambda: calls.append("activate"),
    )
    namespace["_activate_existing_window"](window)
    assert calls == (["console"] if mode else ["restore" if minimized else "show", "raise", "activate"])


@pytest.mark.parametrize("primary, connected, listen_ok", [(True, False, True), (False, True, True), (False, False, True), (True, False, False)])
def test_instance_server_claim_or_notify_without_starting_duplicate(monkeypatch, primary, connected, listen_ok):
    import hashlib
    source = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    method = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_claim_application_instance")
    calls = []
    handler = SimpleNamespace(callback=None)
    server = SimpleNamespace(
        setSocketOptions=lambda options: None,
        newConnection=SimpleNamespace(connect=lambda callback: setattr(handler, "callback", callback)),
        listen=lambda name: calls.append(("listen", name)) or listen_ok,
        close=lambda: calls.append("close"),
        hasPendingConnections=lambda: False,
    )
    server_factory = lambda app: server
    server_factory.SocketOption = SimpleNamespace(UserAccessOption=1)
    lock = SimpleNamespace(
        setStaleLockTime=lambda timeout: None, tryLock=lambda timeout: primary,
        unlock=lambda: calls.append("unlock"),
    )
    paths = SimpleNamespace(StandardLocation=SimpleNamespace(TempLocation=1), writableLocation=lambda location: str(ROOT))
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", SimpleNamespace(QLockFile=lambda path: lock, QStandardPaths=paths))
    socket = SimpleNamespace(
        connectToServer=lambda name: calls.append(("connect", name)),
        waitForConnected=lambda timeout: connected, waitForReadyRead=lambda timeout: False,
        write=lambda data: calls.append(("write", data)) or len(data), bytesToWrite=lambda: 0,
        disconnectFromServer=lambda: calls.append("disconnect"),
    )
    monkeypatch.setitem(sys.modules, "PyQt6.QtNetwork", SimpleNamespace(QLocalServer=server_factory, QLocalSocket=lambda: socket))
    namespace = dict(os=os, hashlib=hashlib)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<single-instance>", "exec"), namespace)
    app = SimpleNamespace(aboutToQuit=SimpleNamespace(connect=lambda callback: calls.append("cleanup")))
    if primary and not listen_ok:
        with pytest.raises(RuntimeError, match="canal"):
            namespace["_claim_application_instance"](app, lambda: None)
        assert calls[-1] == "unlock"
        return
    if not primary and not connected:
        with pytest.raises(RuntimeError, match="contactar"):
            namespace["_claim_application_instance"](app, lambda: None)
        return
    result = namespace["_claim_application_instance"](app, lambda: None)
    assert (result is server) == primary
    assert ("cleanup" in calls) == primary
    assert (("write", b"activate\n") in calls) == (not primary)
    if not primary:
        assert not any(isinstance(call, tuple) and call[0] == "listen" for call in calls)
        assert result is None and calls[-1] == "disconnect"


def test_instance_server_real_qt_core_ipc_between_processes():
    code = '''
import os
import sys
import subprocess
import main as entry
from PyQt6.QtCore import QCoreApplication, QTimer
app = QCoreApplication([])
key = 'GammaSync-IPC-test-' + str(os.getpid())
entry.os.path.expanduser = lambda path: key
activated = []
server = entry._claim_application_instance(app, lambda: (activated.append(True), app.quit()))
assert server is not None
child_code = '\\n'.join([
    'import main as entry',
    'from PyQt6.QtCore import QCoreApplication',
    'app = QCoreApplication([])',
    'entry.os.path.expanduser = lambda path: ' + repr(key),
    'assert entry._claim_application_instance(app, lambda: None) is None',
])
child = subprocess.Popen([sys.executable, '-c', child_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
QTimer.singleShot(6000, app.quit)
app.exec()
output, errors = child.communicate(timeout=8)
assert child.returncode == 0 and activated, (child.returncode, activated, output, errors)
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def _window():
    state = SimpleNamespace(frames=[], stopped=False, status="", results="", visible=True, calls=[])
    console = SimpleNamespace(
        set_raw_preview=lambda frames: setattr(state, "frames", list(frames)),
        stop_raw_preview=lambda: (setattr(state, "frames", []), setattr(state, "stopped", True)),
        set_status_text=lambda text: setattr(state, "status", text),
        set_results_html=lambda text: setattr(state, "results", text),
        collapse_for_processing=lambda: state.calls.append("collapse-modern"),
    )
    primary = SimpleNamespace(reconstructed=False, stage="Esfuerzo")
    window = SimpleNamespace(
        _epar_modern_console=console, _active_detached_console="modern",
        study=primary, metrics=None,
        hide=lambda: setattr(state, "visible", False),
        showMaximized=lambda: setattr(state, "visible", True),
        raise_=lambda: None, activateWindow=lambda: None,
        _cine_crudo_stage_display=lambda study: study.stage,
        _secondary_cine_crudo_study=lambda: None,
        _second_stage_study=lambda: None,
        _build_modern_sa_preview_frames=lambda study: ["sa-preview"],
        _build_cine_crudo_frames_for_study=lambda *args: ([0, 1, 2], None, None),
        _refresh_readonly_results_panel=lambda: state.calls.append("patient-data"),
        _sync_epar_modern_clinical_panels=lambda: state.calls.append("modern-data"),
        _log=lambda *args: None,
    )
    for name in ("load_modern_studies", "_refresh_modern_raw_preview", "open_modern_processing", "_begin_modern_raw_processing"):
        _bind(window, name)
    return window, state


def test_load_hides_main_and_prepares_preview_after_loading():
    window, state = _window()

    def load():
        assert window._modern_preview_loading
        window._modern_raw_preview_pending = True

    window.load_one_or_two_studies = load
    window.load_modern_studies()
    assert not state.visible
    assert state.frames == [0, 1, 2]
    assert state.calls == ["patient-data", "modern-data"]
    assert not window._modern_preview_loading
    assert not window._modern_raw_preview_pending


@pytest.mark.parametrize("mode", ["modern", "plus", ""])
def test_asynchrony_layout_is_exclusive_to_modern(mode):
    calls = []
    window = SimpleNamespace(
        _active_detached_console=mode,
        _epar_modern_console=SimpleNamespace(toggle_asynchrony=lambda: calls.append("modern")),
        _toggle_lower_cine_band=lambda: calls.append("legacy"),
    )
    _bind(window, "toggle_modern_asynchrony")
    window.toggle_modern_asynchrony()
    assert calls == ["modern" if mode == "modern" else "legacy"]


@pytest.mark.parametrize("top_mode, frames", [(True, [0, 1]), (False, [0, 1]), (True, [])])
def test_raw_preview_pins_top_console_until_user_unpins(top_mode, frames):
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    method = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == "set_raw_preview")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / "ui" / "epar_modern_console.py"), "exec"), namespace)
    calls = []
    pin = SimpleNamespace(checked=False)
    console = SimpleNamespace(
        _top_mode=top_mode,
        _top_pin_btn=SimpleNamespace(setChecked=lambda checked: setattr(pin, "checked", checked)),
        _expand_top_console=lambda: calls.append("expand"),
        _raw_preview_timer=SimpleNamespace(stop=lambda: None, start=lambda: calls.append("start")),
        _results_stack=SimpleNamespace(setCurrentWidget=lambda widget: calls.append("preview")),
        _raw_preview_label=object(),
        _scale_raw_preview=lambda: None,
        isVisible=lambda: True,
        stop_raw_preview=lambda: calls.append("clear"),
    )
    namespace["set_raw_preview"](console, frames)
    assert pin.checked == bool(top_mode and frames)
    assert calls == (["expand", "preview", "start"] if top_mode and frames else ["preview", "start"] if frames else ["clear"])
    console._top_pin_btn.setChecked(False)
    assert not pin.checked


def test_main_close_in_modern_only_hides_images():
    calls = []
    window = SimpleNamespace(
        _active_detached_console="modern", _application_close_requested=False,
        hide=lambda: calls.append("hide-images"),
        _epar_modern_console=SimpleNamespace(bring_to_front=lambda: calls.append("modern")),
    )
    _bind(window, "closeEvent")
    window.closeEvent(SimpleNamespace(ignore=lambda: calls.append("ignore")))
    assert calls == ["ignore", "hide-images", "modern"]


@pytest.mark.parametrize("accepted", [True, False])
def test_modern_application_exit_respects_main_close_confirmation(accepted):
    calls = []
    window = SimpleNamespace(_application_close_requested=False)

    def close():
        assert window._application_close_requested
        calls.append("close")
        return accepted

    window.close = close
    application = SimpleNamespace(instance=lambda: SimpleNamespace(quit=lambda: calls.append("quit")))
    _bind(window, "request_application_close", QApplication=application)
    window.request_application_close()
    assert calls == (["close", "quit"] if accepted else ["close"])
    assert not window._application_close_requested


@pytest.mark.parametrize("cancel", [True, False])
def test_application_close_preserves_cleanup_and_cancel(cancel):
    calls = []
    window = SimpleNamespace(
        _active_detached_console="modern", _application_close_requested=True,
        _last_browse_dir="", _check_unsaved_study=lambda: cancel,
        _epar_plus_console=None,
        _epar_modern_console=SimpleNamespace(saveGeometry=lambda: "geometry", close_for_app=lambda: calls.append("close-modern")),
        _ui_settings=SimpleNamespace(setValue=lambda *args: calls.append("settings")),
        _save_window_layout=lambda: calls.append("layout"),
    )
    _bind(window, "closeEvent", super=lambda: SimpleNamespace(closeEvent=lambda event: calls.append("accept")))
    window.closeEvent(SimpleNamespace(ignore=lambda: calls.append("ignore")))
    assert calls == (["ignore"] if cancel else ["settings", "layout", "close-modern", "accept"])


@pytest.mark.parametrize("closing_for_app", [True, False])
def test_modern_close_routes_to_application_exit_not_docking(closing_for_app):
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    method = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == "closeEvent")
    namespace = {"QCloseEvent": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / "ui" / "epar_modern_console.py"), "exec"), namespace)
    calls = []
    console = SimpleNamespace(
        _closing_for_app=closing_for_app,
        _owner=SimpleNamespace(request_application_close=lambda: calls.append("exit")),
    )
    event = SimpleNamespace(accept=lambda: calls.append("accept"), ignore=lambda: calls.append("ignore"))
    namespace["closeEvent"](console, event)
    assert calls == (["accept"] if closing_for_app else ["ignore", "exit"])


def test_modern_mockup_actions_positions_and_existing_callbacks():
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    init = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    calls = []
    owner = SimpleNamespace(
        open_modern_processing=lambda: calls.append("process"), open_amyloid_window=lambda: calls.append("planar"),
        open_amyloid_spect_window=lambda: calls.append("spect"), open_ui_preferences_dialog=lambda: calls.append("config"),
    )
    console = SimpleNamespace(
        dockRequested=SimpleNamespace(emit=lambda: calls.append("dock")),
        _button=lambda text, callback, role: SimpleNamespace(text=text, callback=callback, role=role),
    )
    expected = {
        "_process_btn": ("PROCESAR", "green", (2, 0)),
        "_amyloid_planar_btn": ("AMYLOIDOSIS\nPLANAR", "amyloidPlanar", None),
        "_amyloid_spect_btn": ("AMYLOIDOSIS\nSPECT / CT", "amyloidSpect", None),
        "_config_btn": ("CONFIG.", "blue", (2, 9, 1, 2)),
        "_dock_btn": ("ACOPLAR", "mint", (2, 11)),
    }
    namespace = dict(self=console, owner=owner)
    for name, (text, role, position) in expected.items():
        assignment = next(node for node in init.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Attribute) and target.attr == name for target in node.targets))
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "<modern-mockup>", "exec"), namespace)
        button = getattr(console, name)
        assert (button.text, button.role) == (text, role)
        button.callback()
        if position is not None:
            placement = next(node.value for node in init.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                             and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "addWidget"
                             and isinstance(node.value.args[0], ast.Attribute) and node.value.args[0].attr == name)
            assert tuple(ast.literal_eval(arg) for arg in placement.args[1:]) == position
    assert calls == ["process", "planar", "spect", "config", "dock"]
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "setColumnMinimumWidth"
               and [ast.literal_eval(arg) for arg in node.value.args] == [1, 75] for node in init.body)
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name)
               and node.value.func.value.id == "amyloid_layout" and node.value.func.attr == "setSpacing"
               and ast.literal_eval(node.value.args[0]) == 3 for node in init.body)


@pytest.mark.parametrize("initially_hidden", [False, True])
def test_modern_moves_amyloid_actions_out_of_sidebar_and_restores_visibility(initially_hidden):
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    namespace = {"QWidget": object}
    calls = []
    buttons = [SimpleNamespace(isHidden=lambda: initially_hidden, hide=lambda: calls.append("hide"),
                               setVisible=lambda visible: calls.append(("visible", visible))) for index in range(2)]
    sidebar = SimpleNamespace(setParent=lambda parent: None, show=lambda: None)
    console = SimpleNamespace(
        _owner=SimpleNamespace(amyloid_btn=buttons[0], amyloid_spect_btn=buttons[1]),
        _sidebar_host=SimpleNamespace(hide=lambda: None),
        _sidebar_layout=SimpleNamespace(addWidget=lambda widget: None, takeAt=lambda index: SimpleNamespace(widget=lambda: sidebar)),
        set_top_mode=lambda enabled: None, release_asynchrony=lambda: None,
    )
    for name in ("take_sidebar", "release_sidebar"):
        method = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "<amyloid-sidebar>", "exec"), namespace)
    namespace["take_sidebar"](console, sidebar)
    assert calls == ["hide", "hide"]
    assert namespace["release_sidebar"](console) is sidebar
    assert calls[-2:] == [("visible", not initially_hidden)] * 2
    assert console._amyloid_sidebar_visibility == []


def test_modern_exit_marks_closing_before_teardown_and_disposes_independent_tab():
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    method = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == "close_for_app")
    calls = []
    console = SimpleNamespace(
        _closing_for_app=False,
        _top_tab=SimpleNamespace(hide=lambda: calls.append("hide-tab"), close=lambda: calls.append("close-tab"),
                                 deleteLater=lambda: calls.append("delete-tab")),
        set_top_mode=lambda enabled: calls.append(("top-mode", enabled, console._closing_for_app)),
        release_asynchrony=lambda: calls.append("release"), stop_raw_preview=lambda: calls.append("stop-preview"),
        _clinical_resize_timer=SimpleNamespace(stop=lambda: calls.append("stop-resize")),
        close=lambda: calls.append("close-modern"),
    )
    namespace = {"QApplication": SimpleNamespace(instance=lambda: SimpleNamespace(removeEventFilter=lambda obj: calls.append("remove-filter")))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<modern-exit>", "exec"), namespace)
    namespace["close_for_app"](console)
    assert calls == ["hide-tab", ("top-mode", False, True), "release", "stop-preview", "stop-resize",
                     "remove-filter", "close-tab", "delete-tab", "close-modern"]
    namespace["close_for_app"](console)
    assert calls.count("close-tab") == calls.count("delete-tab") == calls.count("close-modern") == 1


@pytest.mark.parametrize("method_name", ["_watch_top_console", "_expand_top_console", "_keep_top_tab_above", "_collapse_top_console", "bring_to_front", "eventFilter"])
def test_pending_modern_callbacks_cannot_resurrect_ui_during_exit(method_name):
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    method = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<pending-exit>", "exec"), namespace)
    console = SimpleNamespace(_closing_for_app=True)
    if method_name == "eventFilter":
        assert namespace[method_name](console, None, None) is False
    else:
        namespace[method_name](console)


@pytest.mark.parametrize("pinned, visible", [(True, True), (True, False), (False, True), (False, False)])
def test_pin_controls_topmost_and_preserves_geometry(pinned, visible):
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    method = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == "_set_top_pinned")
    flag = object()
    namespace = {"Qt": SimpleNamespace(WindowType=SimpleNamespace(WindowStaysOnTopHint=flag))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<pin>", "exec"), namespace)
    calls = []
    geometry = object()
    console = SimpleNamespace(
        _top_pin_btn=SimpleNamespace(setToolTip=lambda text: None),
        isVisible=lambda: visible, geometry=lambda: geometry,
        setWindowFlag=lambda value, enabled: calls.append((value, enabled)),
        setGeometry=lambda value: calls.append(("geometry", value)),
        show=lambda: calls.append("show"),
        _top_hide_timer=SimpleNamespace(stop=lambda: calls.append("stop-hide")),
        _expand_top_console=lambda: calls.append("expand"),
    )
    namespace["_set_top_pinned"](console, pinned)
    assert calls[:2] == [(flag, pinned), ("geometry", geometry)]
    assert ("show" in calls) == visible
    assert ("expand" in calls) == pinned
    assert ("stop-hide" in calls) == pinned


def test_top_tab_is_independent_of_console_native_visibility_and_cleaned_up():
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    init = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    tab_index = next(index for index, node in enumerate(init.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Attribute) and target.attr == "_top_tab" for target in node.targets))
    calls = []
    tab = SimpleNamespace(deleteLater=lambda: calls.append("delete-tab"))

    def widget_factory(parent, flags):
        assert parent is None
        assert flags == 15
        return tab

    console = SimpleNamespace(destroyed=SimpleNamespace(connect=lambda callback: calls.append(callback)))
    namespace = dict(self=console, QWidget=widget_factory, Qt=SimpleNamespace(WindowType=SimpleNamespace(
        Tool=1, FramelessWindowHint=2, WindowStaysOnTopHint=4, WindowDoesNotAcceptFocus=8)))
    exec(compile(ast.Module(body=init.body[tab_index:tab_index + 2], type_ignores=[]), "<top-tab>", "exec"), namespace)
    assert console._top_tab is tab
    assert calls == [tab.deleteLater]
    calls[0]()
    assert calls[-1] == "delete-tab"


@pytest.mark.parametrize("pinned, owned_window", [(True, False), (True, True), (False, False)])
def test_pinned_watch_reasserts_order_without_stealing_focus(pinned, owned_window):
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    method = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == "_watch_top_console")
    namespace = {"QAbstractAnimation": SimpleNamespace(State=SimpleNamespace(Running=1))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<watch>", "exec"), namespace)
    calls = []
    console = SimpleNamespace(
        _top_mode=True, _top_collapsed=False,
        _top_slide=SimpleNamespace(state=lambda: 0),
        _top_tab=SimpleNamespace(hide=lambda: calls.append("tab-hide")),
        _top_open_timer=SimpleNamespace(stop=lambda: calls.append("stop-open")),
        _keep_top_tab_above=lambda: calls.append("tab-show"),
        _position_top_console=lambda: None, isVisible=lambda: True, isMinimized=lambda: False,
        _top_pin_btn=SimpleNamespace(isChecked=lambda: pinned),
        _top_interaction_active=lambda: False, _top_owned_window_active=lambda: owned_window,
        _top_hide_timer=SimpleNamespace(stop=lambda: calls.append("stop-hide"),
                                       isActive=lambda: False, start=lambda: calls.append("start-hide")),
        raise_=lambda: calls.append("raise"),
    )
    for cycle in range(10):
        namespace["_watch_top_console"](console)
    assert "tab-show" not in calls
    assert ("raise" in calls) == (pinned and not owned_window)
    assert ("start-hide" in calls) == (not pinned)


@pytest.mark.parametrize("top_mode, collapsed, visible, minimized", [
    (False, True, False, False), (False, False, True, False),
    (True, True, True, False), (True, False, True, False),
    (True, True, False, True), (True, True, False, False),
])
def test_top_tab_is_shown_only_for_hidden_collapsed_top_console(top_mode, collapsed, visible, minimized):
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    method = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == "_keep_top_tab_above")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<top-tab-visibility>", "exec"), namespace)
    calls = []
    console = SimpleNamespace(
        _top_mode=top_mode, _top_collapsed=collapsed,
        isVisible=lambda: visible, isMinimized=lambda: minimized,
        _top_tab=SimpleNamespace(show=lambda: calls.append("show"), raise_=lambda: calls.append("raise"),
                                 hide=lambda: calls.append("hide")),
    )
    namespace["_keep_top_tab_above"](console)
    assert calls == (["show", "raise"] if top_mode and collapsed and not visible and not minimized else ["hide"])


@pytest.mark.parametrize("top_mode", [False, True])
def test_top_watch_clears_stale_collapse_and_hover_when_console_visible(top_mode):
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    method = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == "_watch_top_console")
    namespace = {"QAbstractAnimation": SimpleNamespace(State=SimpleNamespace(Running=1))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<stale-top-watch>", "exec"), namespace)
    calls = []
    console = SimpleNamespace(
        _top_mode=top_mode, _top_collapsed=True,
        isVisible=lambda: True, isMinimized=lambda: False,
        _top_tab=SimpleNamespace(hide=lambda: calls.append("hide")),
        _top_open_timer=SimpleNamespace(stop=lambda: calls.append("stop-open")),
        _top_slide=SimpleNamespace(state=lambda: 1),
    )
    namespace["_watch_top_console"](console)
    assert calls == ["hide", "stop-open"]
    if top_mode:
        assert not console._top_collapsed


@pytest.mark.parametrize("stages", [("Reposo",), ("Esfuerzo",), ("Esfuerzo", "Reposo")])
def test_preview_patient_fields_are_available_before_processing(stages):
    import copy
    method = copy.deepcopy(METHODS["_refresh_readonly_results_panel"])
    end = next(index for index, node in enumerate(method.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "ef" for target in node.targets))
    method.body = method.body[:end]
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<patient-data>", "exec"), namespace)
    studies = [SimpleNamespace(patient_name="APELLIDO^NOMBRE", patient_id="ID-123", study_date="20261003",
                               accession_number=f"ACC-{stage}", stage=stage) for stage in stages]
    texts = []
    window = SimpleNamespace(
        study=studies[0], metrics=None, phase_result=None,
        file_edit=SimpleNamespace(text=lambda: ""),
        patient_data_label=SimpleNamespace(setText=lambda text: texts.append(text)),
        _second_stage_study=lambda: studies[1] if len(studies) == 2 else None,
        _cine_crudo_stage_display=lambda study: study.stage,
        _phase_label_from_path=lambda *args: "Estudio",
        _patient_biometrics_line=lambda study: "",
    )
    _bind(window, "_format_dicom_date")
    _bind(window, "_study_context")
    namespace["_refresh_readonly_results_panel"](window)
    text = texts[-1]
    for value in ("APELLIDO NOMBRE", "ID-123", "03/10/2026", "Tipo: " + " + ".join(stages)):
        assert value in text
    for stage in stages:
        assert f"ACC-{stage}" in text


def test_modern_asynchrony_real_widgets_restore_legacy_layout(tmp_path):
    script = r'''
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PyQt6.QtCore import QPoint, QRect, QSettings, QEvent, Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QGridLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget
import ui.floating_toolbar as toolbar_module
from ui.cine_widget import CineWidget
from ui.epar_modern_asynchrony import EParModernAsynchrony
from ui.epar_modern_console import EParModernConsole

app = QApplication([])
folder = Path(sys.argv[1])
settings = QSettings(str(folder / 'isolated.ini'), QSettings.Format.IniFormat)
toolbar_module.QSettings = lambda *args: settings
owner = QWidget()
owner.winId()
owner._ui_settings = settings
owner.cine = CineWidget(compact_viewer=True)
owner.cine_compare = CineWidget(compact_viewer=True, is_compare=True)
for cine in (owner.cine, owner.cine_compare):
    assert cine.intestinal_scope() == 'all_slices'
    assert cine.intestinal_scope_combo.currentData() == 'all_slices'
    assert cine.intestinal_scope_combo.currentText() == 'Todos los slices'
    cine.set_intestinal_scope('slice')
    assert cine.intestinal_scope_combo.currentData() == 'slice'
    cine.set_intestinal_scope('all_slices')
owner.cine.set_compare_viewer(owner.cine_compare)
owner.cine.gate_slider.valueChanged.connect(owner.cine_compare.gate_slider.setValue)
owner.cine.slice_slider.valueChanged.connect(owner.cine_compare.slice_slider.setValue)
owner.cine.cmap_combo.currentTextChanged.connect(owner.cine_compare.cmap_combo.setCurrentText)
owner.main_metrics_readout = QLabel('PSD: 12.5 / BW: 34.0')
owner.asynchrony_review_btn = QPushButton('Vista asincronia')
owner._roi_manual_btn = QToolButton()
calls = []
owner._on_cine_panel_activated = lambda stage: calls.append(stage)
owner._roi_manual_btn.clicked.connect(lambda: calls.append('manual'))
owner._toolbar_group_menus = {'roi_manual_por_slice': (
    SimpleNamespace(toggle_near=lambda anchor: calls.append('manual')),
    owner._roi_manual_btn,
)}
owner.asynchrony_review_btn.clicked.connect(lambda: calls.append('review'))
legacy = QVBoxLayout(owner)
owner._lower_cine_panel = QWidget()
legacy.addWidget(owner._lower_cine_panel)
owner._lower_cine_panel.show()
for widget in (owner.cine, owner.main_metrics_readout, owner.asynchrony_review_btn, owner._roi_manual_btn):
    legacy.addWidget(widget)
axis_y, axis_x = np.mgrid[-1:1:40j, -1:1:40j]
ring = np.exp(-((np.hypot(axis_x, axis_y) - .55) / .09) ** 2)
cube = np.stack([np.stack([ring * (1 + gate / 8)] * 5) for gate in range(8)])
owner.cine.set_cube(cube)
owner.cine_compare.set_cube(cube.copy())
owner.cine.set_phase_title('Esfuerzo')
owner.cine_compare.set_phase_title('Reposo')
for name in ('restart_workspace_state', 'load_modern_studies', 'open_modern_processing',
             'open_ui_preferences_dialog', 'open_pdf', 'open_html_report',
             'open_amyloid_window', 'open_amyloid_spect_window'):
    setattr(owner, name, lambda: None)
owner.toggle_modern_asynchrony = lambda: console.toggle_asynchrony()
console = EParModernConsole(owner)
console.show()
for size in ((960, 300), (1500, 700)):
    console.resize(*size)
    app.processEvents()
    slot_position = console._controls_slot.mapTo(console._header, QPoint())
    async_slot_position = console._asynchrony_slot.mapTo(console._header, QPoint())
    controls_rect = console._controls_btn.geometry()
    reset_rect = console._restart_btn.geometry()
    async_rect = console._asynchrony_btn.geometry()
    load_rect = console._load_btn.geometry()
    assert controls_rect.bottomRight() == slot_position + QPoint(console._controls_slot.width() - 1, 41)
    assert controls_rect.size() == console._controls_slot.size()
    assert reset_rect.bottomRight() == slot_position + QPoint(39, 37)
    assert reset_rect.width() == reset_rect.height() == 42
    assert async_rect.y() == async_slot_position.y() + 3
    assert load_rect.y() - async_rect.bottom() - 1 == 3
original_layouts = {}
normal_geometry = console.geometry()
console._top_mode_btn.click()
assert console._top_mode and console._top_pin_btn.isEnabled()
area = console.screen().availableGeometry()
assert console.y() == area.top()
assert abs(console.geometry().center().x() - area.center().x()) <= 1
QCursor.setPos(area.bottomRight() + QPoint(100, 100))
console._collapse_top_console()
assert console._top_collapsed and console._top_tab.isVisible() and not console.isVisible()
assert console._top_tab.height() == 6
console._top_tab.grab().save(str(folder / 'modern_top_tab.png'))
app.sendEvent(console._top_tab, QEvent(QEvent.Type.Enter))
app.sendEvent(console._top_tab, QEvent(QEvent.Type.Leave))
assert not console._top_open_timer.isActive()
QTest.qWait(300)
assert console._top_collapsed
QCursor.setPos(console._top_tab.geometry().center())
app.sendEvent(console._top_tab, QEvent(QEvent.Type.Enter))
assert console._top_open_timer.isActive()
QTest.qWait(500)
assert console.isVisible() and not console._top_collapsed and not console._top_tab.isVisible()
assert console.y() == area.top()
assert not console._top_mode_btn.icon().isNull() and not console._top_pin_btn.icon().isNull()
console._top_pin_btn.click()
console.grab().save(str(folder / 'modern_top_pinned.png'))
QCursor.setPos(console.geometry().center())
console.collapse_for_processing()
assert console._top_collapsed and not console.isVisible()
assert console._top_pin_btn.isChecked() and console._top_tab.isVisible()
assert console._top_tab.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
assert console._top_tab.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
processing_window = QWidget()
processing_window.showMaximized()
processing_window.raise_()
processing_window.activateWindow()
app.processEvents()
assert console._top_tab.isVisible()
from PyQt6.QtGui import QGuiApplication
if QGuiApplication.platformName() == 'windows':
    import ctypes
    from ctypes import wintypes
    get_style = ctypes.windll.user32.GetWindowLongW
    get_style.argtypes = [wintypes.HWND, ctypes.c_int]
    get_style.restype = ctypes.c_long
    assert get_style(int(console._top_tab.winId()), -20) & 0x8
processing_window.hide()
console.bring_to_front()
QTest.qWait(220)
assert console._top_pin_btn.isChecked() and not console._top_collapsed
QCursor.setPos(area.bottomRight() + QPoint(100, 100))
console._collapse_top_console()
assert console.isVisible() and not console._top_collapsed
console._top_pin_btn.click()
dialog = QWidget(console, Qt.WindowType.Dialog)
dialog.show()
app.processEvents()
console._collapse_top_console()
assert not console._top_collapsed
dialog.hide()
console._watch_top_console()
assert console._top_hide_timer.isActive()
QTest.qWait(750)
assert console._top_collapsed
console.bring_to_front()
QTest.qWait(220)
assert console.isVisible() and not console._top_collapsed
console.showMinimized()
app.processEvents()
assert console.isMinimized() and not console._top_tab.isVisible()
console.showNormal()
console._top_mode_btn.click()
assert not console._top_mode and not console._top_watch_timer.isActive()
assert console.geometry() == normal_geometry
assert not console._top_tab.isVisible()
def layout_state(layout):
    if isinstance(layout, QGridLayout):
        return {layout.getItemPosition(index): (layout.itemAt(index).widget(), layout.itemAt(index).layout())
                for index in range(layout.count())}
    return [(layout.itemAt(index).widget(), layout.itemAt(index).layout())
            for index in range(layout.count())]
def remember_layout(layout):
    if layout is None or layout in original_layouts:
        return
    original_layouts[layout] = layout_state(layout)
    for index in range(layout.count()):
        item = layout.itemAt(index)
        remember_layout(item.layout())
        if item.widget() is not None:
            remember_layout(item.widget().layout())
remember_layout(owner.layout())
console._asynchrony_btn.click()
panel = console._asynchrony_panel
saved = list(panel._borrowed)
assert len(saved) >= 25
panel.resize(1120, 590)
panel.show()
app.processEvents()
assert owner.cine.preview.window() is panel
assert owner.cine_compare.preview.window() is panel
assert owner._lower_cine_panel.isHidden()
assert owner.cine.preview.width() >= 140
assert owner.cine_compare.preview.width() >= 140
initial = owner.cine.gate_slider.value()
owner.cine.gate_next_btn.click()
assert owner.cine.gate_slider.value() == initial + 1
assert owner.cine_compare.gate_slider.value() == initial + 1
owner.cine.range_slider.set_values(10, 120)
assert owner.cine._window_low == .1
assert owner.cine_compare._window_low == 0
panel._manual_buttons[1].click()
assert calls[-2:] == ['secondary', 'manual']
owner.asynchrony_review_btn.click()
assert calls[-1] == 'review'
panel._stage_buttons[0].click()
assert panel._stage_panels[1].isHidden()
panel._stage_buttons[2].click()
assert not panel._stage_panels[1].isHidden()
for size in ((1120, 590), (920, 540)):
    panel.resize(*size)
    app.processEvents()
    for cine in (owner.cine, owner.cine_compare):
        assert cine.preview.width() >= 140 and cine.preview.height() >= 140
        assert cine.range_slider.width() == 28 and cine.range_slider.height() >= 140
        preview_rect = QRect(cine.preview.mapToGlobal(QPoint()), cine.preview.size())
        slider_rect = QRect(cine.range_slider.mapToGlobal(QPoint()), cine.range_slider.size())
        assert not preview_rect.intersects(slider_rect)
    assert abs(owner.cine.preview.height() - owner.cine_compare.preview.height()) <= 1
    assert str(owner.cine.current_gate_index() + 1) in panel._primary_position.text()
    panel.grab().save(str(folder / ('modern_asynchrony_%sx%s.png' % size)))
owner.cine_compare.set_cube(None)
panel.refresh_state()
assert not panel._stage_buttons[1].isEnabled()
assert not panel._manual_buttons[1].isEnabled()
assert panel._stage_summary.text() == 'Esfuerzo'
panel._select_stage('secondary')
assert panel._stage_filter == 'both'
owner.cine.set_cube(None)
owner.main_metrics_readout.setText('Sin resultados: procesa el estudio.')
console.set_patient_html('Sin estudio cargado.')
console.set_results_html(owner.main_metrics_readout.text())
assert console._study_label.text() == 'Sin estudio cargado.'
assert console._status_label.text() == 'Sin resultados: procesa el estudio.'
assert panel._stage_summary.text() == 'SIN ESTUDIO'
assert all(not button.isEnabled() for button in panel._manual_buttons)
assert all(not button.isEnabled() for button in panel._stage_buttons[:2])
owner.cine.stop_playback()
console.release_asynchrony()
assert console._asynchrony_panel is None
assert not owner._lower_cine_panel.isHidden()
assert not panel._borrowed
for saved_widget in saved:
    widget, parent, layout, index, grid, alignment, stretch, minimum, maximum, policy, stylesheet, hidden, text, role = saved_widget
    assert widget.parentWidget() is parent
    assert widget.minimumSize() == minimum
    assert widget.maximumSize() == maximum
    assert widget.sizePolicy() == policy
    assert widget.styleSheet() == stylesheet
    found = panel._locate(parent.layout(), widget)
    assert found is not None
    assert found[0] is layout
    assert layout.itemAt(found[1]).alignment() == alignment
    if grid is not None:
        assert layout.getItemPosition(found[1]) == grid
for layout, entries in original_layouts.items():
    assert layout_state(layout) == entries, type(layout).__name__
assert owner.cine.preview.size().width() == 160
assert owner.asynchrony_review_btn.text() == 'Vista asincronia'
assert owner.cine.range_slider.values() == (10, 120)
console._asynchrony_btn.click()
assert console._asynchrony_panel is not None
console.release_asynchrony()
console.set_top_mode(True)
QCursor.setPos(area.bottomRight() + QPoint(100, 100))
console._collapse_top_console()
assert console._top_tab.isVisible()
console.release_sidebar()
assert not console._top_mode and not console._top_tab.isVisible()
assert console.geometry() == normal_geometry
console.set_top_mode(True)
console._collapse_top_console()
console.close_for_app()
assert not console._top_tab.isVisible()
assert not console._top_watch_timer.isActive()
print('Real Qt controls: navigation, ROI routing, stage filtering, per-stage window and legacy restore OK')
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], cwd=ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_modern_preview_timer_uses_70_ms_interval():
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    intervals = [
        call.args[0].value for call in ast.walk(source)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "self._raw_preview_timer.setInterval"
    ]
    assert intervals == [70]


def test_cancel_preserves_existing_preview():
    window, state = _window()
    state.frames = ["existing"]
    window.load_one_or_two_studies = lambda: None
    window.load_modern_studies()
    assert state.frames == ["existing"]
    assert not window._modern_preview_loading


def test_failed_load_resets_loading_flag():
    window, state = _window()

    def fail():
        raise ValueError("synthetic load failure")

    window.load_one_or_two_studies = fail
    with pytest.raises(ValueError):
        window.load_modern_studies()
    assert not window._modern_preview_loading


def test_reconstructed_load_refreshes_patient_and_replaces_previous_raw_preview():
    window, state = _window()
    state.frames = ["previous-raw"]

    def load():
        window.study = SimpleNamespace(reconstructed=True, stage="Reposo")

    window.load_one_or_two_studies = load
    window.load_modern_studies()
    assert state.frames == ["sa-preview"]
    assert "Cine SA" in state.status
    assert state.calls == ["patient-data", "modern-data"]


@pytest.mark.parametrize("stage", ["Reposo", "Esfuerzo"])
@pytest.mark.parametrize("dual", [False, True])
def test_modern_sa_preview_supports_single_or_both_stages(stage, dual):
    window, state = _window()
    window.study = SimpleNamespace(reconstructed=True, stage=stage)
    secondary = SimpleNamespace(reconstructed=True, stage="Reposo" if stage == "Esfuerzo" else "Esfuerzo")
    window._second_stage_study = lambda: secondary if dual else None
    window._build_modern_sa_preview_frames = lambda study: [0, 1, 2] if study is window.study else [10, 11, 12, 13, 14]
    window._build_cine_crudo_frames_for_study = lambda *args: pytest.fail("SA must not use projection rendering")
    window._epar_modern_console.compose_raw_preview_pair = lambda *args: args
    window._refresh_modern_raw_preview()
    assert state.calls == ["patient-data", "modern-data"]
    assert "Cine SA" in state.status
    assert state.status.endswith("70 ms")
    if dual:
        assert state.frames[0] == (0, 10, stage, secondary.stage)
        assert state.frames[-1] == (2, 14, stage, secondary.stage)
    else:
        assert state.frames == [0, 1, 2]


@pytest.mark.parametrize("gates, slices", [(1, 5), (4, 5), (4, 4), (4, 1)])
def test_modern_sa_frames_use_gates_or_slices_without_mutating_cube(gates, slices):
    import numpy as np
    from viz.colormaps import get_phase_cmap
    cube = np.arange(gates * slices * 16, dtype=float).reshape(gates, slices, 4, 4)
    original = cube.copy()
    window = SimpleNamespace(_rgb_frame_to_qpixmap_raw=lambda rgb: rgb.copy())
    _bind(window, "_build_modern_sa_preview_frames", np=np)
    frames = window._build_modern_sa_preview_frames(SimpleNamespace(cube=cube))
    assert len(frames) == (gates if gates >= 3 else slices)
    assert all(frame.dtype == np.uint8 and frame.shape[-1] == 3 for frame in frames)
    assert frames[0].shape == (4, 4, 3)
    images = cube[:, slices // 2] if gates >= 3 else cube.sum(axis=0)
    maximum = max(float(images.max()), 1e-9)
    expected = (get_phase_cmap("french")(images / maximum)[..., :3] * 255).astype(np.uint8)
    assert all(np.array_equal(frame, image) for frame, image in zip(frames, expected))
    assert np.array_equal(cube, original)


def test_dual_preview_preserves_both_endpoints_with_unequal_counts():
    window, state = _window()
    secondary = SimpleNamespace(reconstructed=False, stage="Reposo")
    window._secondary_cine_crudo_study = lambda: secondary
    window._build_cine_crudo_frames_for_study = lambda study, *args: (
        [0, 1, 2] if study is window.study else [10, 11, 12, 13, 14], None, None
    )
    window._epar_modern_console.compose_raw_preview_pair = lambda top, bottom, top_label, bottom_label: (top, bottom, top_label, bottom_label)
    window._refresh_modern_raw_preview()
    assert len(state.frames) == 5
    assert state.frames[0] == (0, 10, "Esfuerzo", "Reposo")
    assert state.frames[-1] == (2, 14, "Esfuerzo", "Reposo")


def test_preview_preserves_main_viewer_coordinate_metadata():
    window, state = _window()
    original = {"ss": 3, "split_y": 300}
    window._cine_crudo_dual_render_meta = original
    window._secondary_cine_crudo_study = lambda: SimpleNamespace(stage="Reposo")

    def stack(*args):
        raise AssertionError("Modern must not use the main viewer compositor")

    window._stack_cine_crudo_dual_pixmaps = stack
    window._epar_modern_console.compose_raw_preview_pair = lambda *args: "horizontal-frame"
    window._refresh_modern_raw_preview()
    assert state.frames
    assert window._cine_crudo_dual_render_meta is original


def test_open_processing_keeps_preview_until_reconstruction():
    window, state = _window()
    state.frames = ["preview"]
    window.process_current = lambda: state.calls.append("process")
    window.open_modern_processing()
    assert state.visible and state.calls == ["collapse-modern", "process"]
    assert state.frames == ["preview"]
    assert not window._pending_dual_raw_load
    window._begin_modern_raw_processing()
    assert state.stopped and not state.frames
    assert not window._modern_raw_preview_enabled
    assert "procesamiento en curso" in state.results
    window._refresh_modern_raw_preview()
    assert not state.frames


@pytest.mark.parametrize("preview_only, reconstructed", [(False, False), (True, False), (True, True)])
def test_process_current_preview_fast_path_is_exclusive_to_modern(preview_only, reconstructed):
    window, state = _window()
    window.study.reconstructed = reconstructed
    path = str(ROOT / "version.py")
    window.file_edit = SimpleNamespace(text=lambda: path)
    window._async_skip_compare_reprocess = False
    window.compare_bundle = None
    window._last_primary_path = os.path.abspath(path)
    window._cache_study_sig = "same-study"
    window._build_study_signature = lambda path: "same-study"
    window._set_progress = lambda *args: None
    window._refresh_tab_enabled_states = lambda: None
    window._apply_gated_controls_state = lambda: None
    window._clear_compare_state = lambda: state.calls.append("clear-compare")
    window._refresh_readonly_results_panel = lambda: state.calls.append("readouts")
    window._handle_raw_projections_loaded = lambda *args: state.calls.append("qc")
    window.cine_crudo_timer = SimpleNamespace(stop=lambda: state.calls.append("stop-cine"))
    window._modern_preview_loading = preview_only
    window.metrics = {"old": 1}
    window.phase_result = "old"
    _bind(window, "process_current")
    window.process_current()
    if preview_only:
        assert "qc" not in state.calls
        assert "clear-compare" in state.calls
        assert window.metrics is None and window.phase_result is None
        assert window.seg is None
        assert window._modern_raw_preview_pending and window._modern_raw_preview_enabled
    else:
        assert state.calls == ["qc"]


@pytest.mark.parametrize("patient_ok, zoom_ok", [(True, True), (False, True), (True, False)])
def test_compare_sa_preview_keeps_patient_guards_and_defers_processing(patient_ok, zoom_ok):
    calls = []
    study = SimpleNamespace(reconstructed=True)
    window = SimpleNamespace(
        _modern_preview_loading=True,
        _set_progress=lambda *args: None,
        _check_second_stage_patient=lambda value: patient_ok,
        _check_stage_zoom_consistency=lambda value: zoom_ok,
        _process_secondary_bundle=lambda *args, **kwargs: pytest.fail("Preview must not process the secondary SA"),
        _refresh_readonly_results_panel=lambda: calls.append("patient"),
        _log=lambda *args: None,
        compare_bundle=None,
    )
    _bind(window, "_load_compare_study_from_path", dicom_loader=SimpleNamespace(load=lambda *args, **kwargs: study))
    window._load_compare_study_from_path("rest_sa.dcm")
    if patient_ok and zoom_ok:
        assert window.compare_bundle is None
        assert window.compare_raw_study is study
        assert window.compare_raw_path == "rest_sa.dcm"
        assert window.compare_metrics is None and window.compare_ef is None
        assert window._modern_raw_preview_pending and window.dual_mode_active
        assert calls == ["patient"]
    else:
        assert window.compare_bundle is None and not calls


def test_compare_raw_preview_does_not_generate_qc_or_start_main_cine():
    window, state = _window()
    secondary = SimpleNamespace(reconstructed=False, stage="Reposo")
    loader = SimpleNamespace(load=lambda *args, **kwargs: secondary)
    window._modern_preview_loading = True
    window._check_second_stage_patient = lambda study: True
    window._check_stage_zoom_consistency = lambda study: True
    window._refresh_cine_source_selector = lambda: None
    window._apply_cine_source = lambda *args, **kwargs: None
    window._set_active_cine_crudo_stage = lambda *args, **kwargs: None
    window._refresh_readonly_results_panel = lambda: state.calls.append("readouts")
    window._set_progress = lambda *args: None
    _bind(window, "_load_compare_raw_study_from_path", dicom_loader=loader)
    window._load_compare_raw_study_from_path("synthetic-rest.dcm")
    assert window.compare_raw_study is secondary
    assert window.dual_mode_active
    assert state.calls == ["readouts"]

@pytest.mark.parametrize("tab_name", ["polar_perfusion_directa", "panel_funcional_gated", "guia_fase_vi", "comparacion_ejes", "curva_fevi"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_requested_lazy_tabs_restore_rockford_after_loading(tab_name, cached, fails):
    calls = []

    def load(*args, **kwargs):
        calls.append("work")
        if fails:
            raise RuntimeError("synthetic render failure")

    window = SimpleNamespace(
        study=object(), seg=object(), phase_result=object(), compare_bundle=None,
        _lazy_render_pending_tabs=set(), _deferred_hq_running=False,
        _is_tab_render_ready=lambda name: cached,
        _load_preview=load, _write_outputs=load,
        _load_previews_selected=lambda pending: calls.append("loaded"),
        _set_progress=lambda *args: calls.append("progress"),
        _log=lambda message: calls.append("error"),
        _begin_background_busy=lambda: calls.append("busy"),
        _end_background_busy=lambda: calls.append("restore"),
    )
    _bind(window, "_request_lazy_tab_render")
    if cached and fails:
        with pytest.raises(RuntimeError, match="synthetic"):
            window._request_lazy_tab_render(tab_name)
    else:
        window._request_lazy_tab_render(tab_name)
    if tab_name == "curva_fevi":
        assert "busy" not in calls and "restore" not in calls
    else:
        assert calls[0] == "busy" and calls[-1] == "restore"
        assert calls.count("busy") == calls.count("restore") == 1
    assert "work" in calls


@pytest.mark.parametrize("secondary_state", ["none", "pending", "processed"])
def test_sa_montage_builds_cuts_from_pending_or_processed_secondary(secondary_state):
    import numpy as np
    primary = SimpleNamespace(cube=np.full((4, 6, 8, 8), 2.0))
    secondary = SimpleNamespace(cube=np.full((4, 6, 8, 8), 7.0))
    window = SimpleNamespace(
        study=primary, cine=None, cine_compare=None,
        compare_bundle={"study": secondary} if secondary_state == "processed" else None,
        compare_raw_study=secondary if secondary_state == "pending" else None,
        cine_crudo_stripe_start_by_stage={},
        _apply_intestinal_mask_to_cube=lambda cube, widget: cube,
        _montage_available_gates=lambda: 4,
        _refresh_montage_gated_source_enabled=lambda: None,
        _sa_direct_intestinal_sig=lambda: "current",
        _log=lambda *args: None,
    )
    _bind(window, "_second_stage_study")
    _bind(window, "_build_sa_direct_clinical_montage", np=np)
    compare_stage = None if secondary_state == "none" else "rest"
    window._build_sa_direct_clinical_montage("stress", compare_stage)
    assert set(window.cine_crudo_axes_for_export_stress) == {"SA", "HLA", "VLA"}
    assert window._sa_direct_montage_stages == ("stress", compare_stage)
    assert window._cine_crudo_recon_stage == "stress"
    if compare_stage:
        assert set(window.cine_crudo_axes_for_export_rest) == {"SA", "HLA", "VLA"}
        assert window.cine_crudo_axes_for_export_rest["SA"].max() == 7.0
        assert window.cine_crudo_axes_for_export_ungated_rest["SA"].max() == 28.0


@pytest.mark.parametrize("dual", [False, True])
def test_sa_selected_load_builds_montage_during_modern_preview(dual):
    method = METHODS["_load_sa_recon_direct"]
    start = next(index for index, statement in enumerate(method.body)
                 if ast.unparse(statement) == "self.file_edit.setText(primary)")
    selected_load = ast.parse("def load_selected_sa(self):\n    pass").body[0]
    selected_load.body = method.body[start:]
    namespace = {"os": os, "primary": "stress.dcm", "sel": ["stress.dcm", "rest.dcm"] if dual else ["stress.dcm"],
                 "candidates": [("stress", "stress.dcm", "stress", True), ("rest", "rest.dcm", "rest", True)]}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selected_load], type_ignores=[])), "<selected-sa-load>", "exec"), namespace)
    calls = []
    window = SimpleNamespace(
        _modern_preview_loading=True, study=object(), metrics=None,
        file_edit=SimpleNamespace(setText=lambda path: None),
        process_current=lambda: calls.append("preview-load"),
        _load_compare_study_from_path=lambda path: calls.append("secondary-preview"),
        _second_stage_study=lambda: object() if dual else None,
        _build_sa_direct_clinical_montage=lambda *stages: calls.append(stages),
        _log=lambda *args: None,
    )
    namespace["load_selected_sa"](window)
    assert calls == (["preview-load", "secondary-preview", ("stress", "rest")] if dual else ["preview-load", ("stress", None)])


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("failure_stage", [None, "rebuild", "display"])
def test_montage_tab_restores_rockford_after_rebuild_or_display(cached, failure_stage):
    calls = []

    def action(name, stage):
        calls.append(name)
        if failure_stage == stage:
            raise RuntimeError("synthetic montage failure")

    window = SimpleNamespace(
        tabs=SimpleNamespace(tabText=lambda index: "Montaje clínico"),
        _tab_name_from_title=lambda title: "comparacion_ejes",
        cine_crudo_axes_for_export=True, cine_crudo_preview_mode="sa_montage",
        preview_pixmaps={"comparacion_ejes": object()},
        _montage_last_signature="current" if cached else "old",
        _montage_signature=lambda: "current",
        _maybe_rebuild_sa_direct_montage=lambda: action("rebuild-sa", "rebuild"),
        _maybe_rebuild_crudo_montage=lambda: calls.append("rebuild-crudo"),
        _apply_preview_zoom=lambda name: action("zoom", "display"),
        _show_cine_crudo_sa_montage=lambda: action("render", "display"),
        _begin_background_busy=lambda: calls.append("busy"),
        _end_background_busy=lambda: calls.append("restore"),
    )
    _bind(window, "_on_preview_tab_changed")
    if failure_stage is not None:
        with pytest.raises(RuntimeError, match="synthetic"):
            window._on_preview_tab_changed(0)
    else:
        window._on_preview_tab_changed(0)
    assert calls[0] == "busy" and calls[-1] == "restore"
    assert calls.count("busy") == calls.count("restore") == 1
    if failure_stage != "rebuild":
        assert ("zoom" if cached else "render") in calls


@pytest.mark.parametrize("fail_raw, fail_feta, auto_ok, fail_process", [
    (False, False, True, False),
    (True, False, True, False),
    (False, True, True, False),
    (False, False, False, False),
    (False, False, True, True),
])
def test_trust_pipeline_wraps_whole_flow_with_rockford(fail_raw, fail_feta, auto_ok, fail_process):
    calls = []

    def recon(*, force_fbp_localizer=False, feta_only=False):
        calls.append(("recon", force_fbp_localizer, feta_only))
        if not feta_only and fail_raw:
            return False
        if feta_only and fail_feta:
            return False
        return True

    def process():
        calls.append("process")
        if fail_process:
            raise RuntimeError("synthetic trust failure")

    window = SimpleNamespace(
        _cine_crudo_recon_target=lambda: (SimpleNamespace(reconstructed=False), None, None, None),
        _begin_background_busy=lambda: calls.append("busy"),
        _end_background_busy=lambda: calls.append("restore"),
        _cine_crudo_target_stages=lambda: ["reposo", "esfuerzo"],
        _log=lambda message: calls.append(("log", message)),
        statusBar=lambda: SimpleNamespace(showMessage=lambda text, timeout=0: calls.append(("status", text))),
        _reconstruct_cine_crudo_raw=recon,
        _trust_localizer_fast_fbp=True,
        _auto_reorient_single_stage=lambda stage: auto_ok,
        _process_cine_crudo_reconstruction=process,
        _show_fading_notice=lambda *args, **kwargs: calls.append("notice"),
    )
    _bind(window, "_run_trust_pipeline", QMessageBox=SimpleNamespace(
        information=lambda *args: calls.append("info"),
        warning=lambda *args: calls.append("warn"),
    ))
    window._run_trust_pipeline()

    assert calls[0] == "busy" and calls[-1] == "restore"
    assert calls.count("busy") == calls.count("restore") == 1
    if fail_raw:
        assert "process" not in calls
    if fail_feta:
        assert "process" not in calls
    if not auto_ok:
        assert "warn" in calls and "process" not in calls
    if fail_process:
        assert "notice" not in calls
    if not (fail_raw or fail_feta or (not auto_ok) or fail_process):
        assert "notice" in calls and "process" in calls


@pytest.mark.parametrize("active, reuse", [("", False), ("", True), ("plus", False), ("plus", True), ("modern", True)])
def test_modern_opens_anchored_at_top_without_overriding_existing_choice(monkeypatch, active, reuse):
    calls = []
    control = SimpleNamespace(setText=lambda text: None, setVisible=lambda visible: None)
    console = SimpleNamespace(
        dockRequested=SimpleNamespace(connect=lambda callback: None),
        take_sidebar=lambda sidebar: None,
        restore_safe_geometry=lambda geometry: calls.append("geometry"),
        set_study_text=lambda text: None, set_status_text=lambda text: None,
        set_top_mode=lambda enabled: calls.append(("top", enabled)),
        _top_pin_btn=SimpleNamespace(setChecked=lambda checked: calls.append(("pin", checked))),
        bring_to_front=lambda: calls.append("front"),
    )
    monkeypatch.setitem(sys.modules, "ui.epar_modern_console", SimpleNamespace(EParModernConsole=lambda owner: console))
    window = SimpleNamespace(
        _active_detached_console=active, _epar_modern_console=console if reuse else None,
        dock_epar_plus_console=lambda: calls.append("dock-plus"),
        dock_epar_modern_console=lambda: None,
        main_splitter=SimpleNamespace(sizes=lambda: [300, 1000], setSizes=lambda sizes: None),
        _sidebar_widget=SimpleNamespace(setMaximumWidth=lambda width: None),
        _ui_settings=SimpleNamespace(value=lambda *args: None, setValue=lambda *args: None, sync=lambda: None),
        file_edit=SimpleNamespace(text=lambda: ""),
        _progress_bar=SimpleNamespace(format=lambda: "SISTEMA LISTO"),
        _sync_epar_modern_clinical_panels=lambda: None,
        _side_cards_host=control, _set_lower_cine_band_visible=lambda visible: None,
        width=lambda: 1300, epar_modern_btn=control, _epar_plus_front_btn=control,
        _refresh_modern_raw_preview=lambda: None, hide=lambda: calls.append("hide-main"),
    )
    _bind(window, "detach_epar_modern_console")
    window.detach_epar_modern_console()
    if active == "modern":
        assert calls == ["front"]
    else:
        assert calls == (["dock-plus"] if active == "plus" else []) + ["geometry", ("top", True), ("pin", True), "front", "hide-main"]
        assert window._active_detached_console == "modern"


def test_rockford_gif_has_transparent_background_in_every_frame():
    from PIL import Image

    with Image.open(ROOT / "assets" / "rockford-boulder-dash.gif") as animation:
        assert animation.n_frames == 35
        assert animation.info["loop"] == 0
        for frame_index in range(animation.n_frames):
            animation.seek(frame_index)
            alpha = animation.convert("RGBA").getchannel("A")
            width, height = alpha.size
            for border in ((0, 0, width, 1), (0, height - 1, width, height),
                           (0, 0, 1, height), (width - 1, 0, width, height)):
                assert alpha.crop(border).getextrema() == (0, 0), frame_index
            assert alpha.getextrema() == (0, 255), frame_index
            assert animation.info["duration"] > 0


def test_rockford_black_eye_pixels_are_transparent():
    import numpy as np
    from PIL import Image

    with Image.open(ROOT / "assets" / "rockford-boulder-dash.gif") as animation:
        for frame_index in range(animation.n_frames):
            animation.seek(frame_index)
            frame = animation.convert("RGBA")
            for eye_box in ((190, 148, 246, 220), (366, 148, 422, 220)):
                pixels = np.asarray(frame.crop(eye_box))
                opaque_black = np.all(pixels[:, :, :3] == 0, axis=2) & (pixels[:, :, 3] > 0)
                assert not opaque_black.any(), frame_index


@pytest.mark.parametrize("valid_gif", [True, False])
def test_rockford_busy_cursor_animates_and_restores_nested_calls(valid_gif):
    calls = []
    signal = SimpleNamespace(callback=None)
    pixmap = SimpleNamespace(isNull=lambda: False, width=lambda: 40, height=lambda: 40)
    movie = SimpleNamespace(
        setCacheMode=lambda value: None, setScaledSize=lambda size: calls.append(("size", size)),
        frameChanged=SimpleNamespace(connect=lambda callback: setattr(signal, "callback", callback)),
        isValid=lambda: valid_gif, jumpToFrame=lambda frame: calls.append(("frame", frame)),
        start=lambda: calls.append("start"), stop=lambda: calls.append("stop"),
        currentPixmap=lambda: pixmap,
    )

    def movie_factory(path, parent):
        assert Path(path).name == "rockford-boulder-dash.gif" and Path(path).is_file()
        calls.append("load-gif")
        return movie

    movie_factory.CacheMode = SimpleNamespace(CacheAll=1)
    window = SimpleNamespace(_busy_cursor_depth=0, _busy_cursor_movie=None)
    application = SimpleNamespace(
        setOverrideCursor=lambda cursor: calls.append("push"),
        changeOverrideCursor=lambda cursor: calls.append(("cursor", cursor)),
        restoreOverrideCursor=lambda: calls.append("restore"),
        processEvents=lambda flags: calls.append("paint"),
    )
    globals_dict = dict(
        QApplication=application, QMovie=movie_factory,
        QSize=lambda width, height: (width, height), QCursor=lambda pix, x, y: (pix, x, y),
        Qt=SimpleNamespace(CursorShape=SimpleNamespace(WaitCursor=1)),
        QEventLoop=SimpleNamespace(ProcessEventsFlag=SimpleNamespace(ExcludeUserInputEvents=1)),
        __file__=str(ROOT / "ui" / "main_window.py"),
    )
    for name in ("_begin_background_busy", "_update_background_busy_cursor", "_end_background_busy"):
        _bind(window, name, **globals_dict)
    window._begin_background_busy()
    window._begin_background_busy()
    assert calls.count("push") == calls.count("load-gif") == 1
    assert ("size", (32, 32)) in calls
    assert "paint" in calls
    assert ("start" in calls) == valid_gif
    if valid_gif:
        assert ("cursor", (pixmap, 20, 20)) in calls
        signal.callback(1)
    window._end_background_busy()
    assert "restore" not in calls and "stop" not in calls
    window._end_background_busy()
    assert calls[-2:] == ["stop", "restore"]
    final_calls = list(calls)
    signal.callback(2)
    window._end_background_busy()
    assert calls == final_calls


@pytest.mark.parametrize("mode", ["perfusion", "cine"])
def test_apply_polar_math_keeps_rendered_result_without_changing_scale(mode):
    calls = []
    window = SimpleNamespace(
        _polar_cine_cart_cache={"frames": ["polar-maps"], "disk_cmap": "odyssey_cool"},
        polar_perf_screen_cmap="odyssey_cool", polar_view_mode=mode,
        _begin_background_busy=lambda: calls.append("busy"),
        _end_background_busy=lambda: calls.append("restore"),
        _set_polar_view_mode=lambda value: calls.append(("mode", value)),
        _rebuild_polar_cine_frames_screen=lambda: calls.append("render-result") or True,
        _load_preview=lambda name: calls.append("reload-old-gif"),
    )
    _bind(window, "_apply_polar_math")
    window._apply_polar_math()
    assert calls == (["busy", ("mode", "cine"), "render-result", "restore"]
                     if mode == "perfusion" else ["busy", "render-result", "restore"])


@pytest.mark.parametrize("raises", [False, True])
def test_apply_polar_math_restores_cursor_on_render_failure(raises):
    calls = []

    def render():
        if raises:
            raise RuntimeError("render failed")
        return False

    window = SimpleNamespace(
        _polar_cine_cart_cache={"frames": ["polar-maps"]}, polar_view_mode="cine",
        _begin_background_busy=lambda: calls.append("busy"),
        _end_background_busy=lambda: calls.append("restore"),
        _rebuild_polar_cine_frames_screen=render,
        statusBar=lambda: SimpleNamespace(showMessage=lambda *args: calls.append("failure")),
    )
    _bind(window, "_apply_polar_math")
    if raises:
        with pytest.raises(RuntimeError, match="render failed"):
            window._apply_polar_math()
    else:
        window._apply_polar_math()
    assert calls[0] == "busy" and calls[-1] == "restore"
    assert ("failure" in calls) == (not raises)


def test_polar_screen_scale_keeps_view_mode_and_uses_compare_rerender():
    calls = []
    window = SimpleNamespace(
        polar_view_mode="perfusion",
        compare_bundle=object(),
        polar_perf_screen_cmap="odyssey_cool",
        polar_screen_color_strip=SimpleNamespace(set_cmap=lambda name: calls.append(("strip", name))),
        polar_perf_view_perf_btn=None,
        polar_perf_view_cine_btn=None,
        _set_preview_background=lambda name, cmap: calls.append(("bg", name, cmap)),
        _rebuild_polar_cine_frames_screen=lambda: calls.append("cine") or True,
        _rerender_polar_perfusion_screen=lambda: calls.append("single") or True,
        _rerender_polar_perfusion_compare_screen=lambda: calls.append("compare") or True,
    )
    _bind(window, "_on_polar_screen_cmap_changed")
    window._on_polar_screen_cmap_changed("odyssey_SPECT")
    assert window.polar_view_mode == "perfusion"
    assert ("bg", "polar_perfusion_directa", "odyssey_SPECT") in calls
    assert "compare" in calls and "single" not in calls and "cine" not in calls

def test_polar_screen_scale_in_cine_does_not_flip_mode():
    calls = []
    window = SimpleNamespace(
        polar_view_mode="cine",
        compare_bundle=object(),
        polar_perf_screen_cmap="odyssey_cool",
        polar_screen_color_strip=None,
        polar_perf_view_perf_btn=None,
        polar_perf_view_cine_btn=None,
        _set_preview_background=lambda name, cmap: calls.append(("bg", name, cmap)),
        _rebuild_polar_cine_frames_screen=lambda: calls.append("cine") or True,
        _rerender_polar_perfusion_screen=lambda: calls.append("single") or True,
        _rerender_polar_perfusion_compare_screen=lambda: calls.append("compare") or True,
    )
    _bind(window, "_on_polar_screen_cmap_changed")
    window._on_polar_screen_cmap_changed("hot")
    assert window.polar_view_mode == "cine"
    assert "cine" in calls and "compare" not in calls and "single" not in calls

def test_readouts_switch_to_results_when_metrics_arrive():
    window, state = _window()
    state.frames = ["preview"]
    window.metrics = {"PSD": 10}
    window.patient_data_label = SimpleNamespace(text=lambda: "patient")
    window.main_metrics_readout = SimpleNamespace(text=lambda: "clinical-results")
    window._epar_modern_console.set_patient_html = lambda text: None
    _bind(window, "_sync_epar_modern_clinical_panels")
    window._sync_epar_modern_clinical_panels()
    assert state.stopped and state.results == "clinical-results"


def test_preview_bounces_without_repeating_endpoints_or_rescaling():
    source = ast.parse((ROOT / "ui" / "epar_modern_console.py").read_text(encoding="utf-8"))
    console_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "EParModernConsole")
    advance = next(node for node in console_class.body if isinstance(node, ast.FunctionDef) and node.name == "_advance_raw_preview")
    namespace = {}
    exec(compile(ast.Module(body=[advance], type_ignores=[]), "<cine bounce>", "exec"), namespace)
    displayed = []
    console = SimpleNamespace(
        _raw_preview_scaled_frames=[0, 1, 2], _raw_preview_index=0,
        _raw_preview_direction=1,
        _raw_preview_label=SimpleNamespace(setPixmap=displayed.append),
    )
    for count in range(6):
        namespace["_advance_raw_preview"](console)
    assert displayed == [1, 2, 1, 0, 1, 2]


@pytest.mark.parametrize("with_cards", [False, True])
def test_empty_session_refresh_clears_modern_clinical_readouts(with_cards):
    window, state = _window()
    window.study = None
    window.metrics = None
    state.results = "PSD: 99 / BW: 180 - previous patient"
    state.patient = "Previous patient"
    window._epar_modern_console.set_patient_html = lambda text: setattr(state, "patient", text)
    window._sidebar_widget = object()
    window.patient_data_label = SimpleNamespace(text=lambda: "Sin estudio cargado.")
    window.main_metrics_readout = SimpleNamespace(text=lambda: "Sin resultados: procesá el estudio.")
    deleted = []
    for attr in ("_patient_card", "_results_card"):
        setattr(window, attr, SimpleNamespace(deleteLater=lambda name=attr: deleted.append(name)) if with_cards else None)
    window._side_cards_host = SimpleNamespace(setVisible=lambda visible: setattr(state, "cards_visible", visible))
    window._reposition_fading_notices = lambda: None
    _bind(window, "_sync_epar_modern_clinical_panels")
    _bind(window, "_refresh_persistent_patient_card")
    window._refresh_persistent_patient_card()
    assert state.patient == "Sin estudio cargado."
    assert state.results == "Sin resultados: procesá el estudio."
    assert not state.cards_visible
    assert window._patient_card is None and window._results_card is None
    assert len(deleted) == (2 if with_cards else 0)


@pytest.mark.parametrize("mode", ["modern", "plus", ""])
@pytest.mark.parametrize("main_visible", [False, True])
def test_restart_clears_floating_cine_and_reenables_next_load(mode, main_visible):
    window, state = _window()
    window._active_detached_console = mode
    state.visible = main_visible
    def bring_modern_front():
        assert not state.visible
        assert window.study is None
        state.calls.append("modern-front")
    window._epar_modern_console.bring_to_front = bring_modern_front
    state.frames = ["previous-preview"]
    state.results = "PSD: 99 / BW: 180 - previous patient"
    state.cubes = ["primary", "secondary"]
    window._modern_raw_preview_enabled = False
    window._modern_raw_preview_pending = True
    cleared = SimpleNamespace(clear=lambda: None)
    window._reset_session_data = lambda: setattr(window, "study", None)
    window.file_edit = cleared
    window.manual_rois = cleared
    window._sync_manual_rois = lambda *args: None
    window._push_manual_centers_to_cine = lambda: None
    window.seg_method = SimpleNamespace(currentText=lambda: "auto")
    window.summary_clinical = cleared
    window.summary_technical = cleared
    window.summary_executive = cleared
    window._refresh_readonly_results_panel = lambda: setattr(state, "source_results", "Sin resultados: procesá el estudio.")
    window._sidebar_widget = object()
    window.patient_data_label = SimpleNamespace(text=lambda: "Sin estudio cargado.")
    window.main_metrics_readout = SimpleNamespace(text=lambda: state.source_results)
    window._patient_card = None
    window._results_card = None
    window._reposition_fading_notices = lambda: None
    window._epar_modern_console.set_patient_html = lambda text: setattr(state, "patient", text)
    def results_updated(text):
        assert state.cubes == [None, None]
        state.results = text
    window._epar_modern_console.set_results_html = results_updated
    _bind(window, "_sync_epar_modern_clinical_panels")
    _bind(window, "_refresh_persistent_patient_card")
    window.preview_movies = {}
    window.preview_pixmaps = {"previous": 1}
    window.preview_labels = {}
    window.gate_dropout_status = SimpleNamespace(setText=lambda text: None)
    window.cine = SimpleNamespace(set_cube=lambda cube: state.cubes.__setitem__(0, cube))
    window.cine_compare = SimpleNamespace(set_cube=lambda cube: state.cubes.__setitem__(1, cube))
    window._refresh_cine_source_selector = lambda: None
    window._progress_bar = SimpleNamespace(setValue=lambda value: None, setFormat=lambda text: None)
    window.log_box = cleared
    window.statusBar = lambda: SimpleNamespace(showMessage=lambda text: None)
    _bind(window, "restart_workspace_state")
    window.restart_workspace_state()
    assert state.stopped and not state.frames
    assert state.status == "Listo"
    assert window._modern_raw_preview_enabled
    assert not window._modern_raw_preview_pending
    assert window.study is None and not window.preview_pixmaps
    assert state.cubes == [None, None]
    assert state.patient == "Sin estudio cargado."
    assert state.results == "Sin resultados: procesá el estudio."
    assert state.visible == (False if mode == "modern" else main_visible)
    assert ("modern-front" in state.calls) == (mode == "modern")