"""Modern load/preview wiring without creating QApplication or real output files."""
import ast
import os
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


def _window():
    state = SimpleNamespace(frames=[], stopped=False, status="", results="", visible=True, calls=[])
    console = SimpleNamespace(
        set_raw_preview=lambda frames: setattr(state, "frames", list(frames)),
        stop_raw_preview=lambda: (setattr(state, "frames", []), setattr(state, "stopped", True)),
        set_status_text=lambda text: setattr(state, "status", text),
        set_results_html=lambda text: setattr(state, "results", text),
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
        _build_cine_crudo_frames_for_study=lambda *args: ([0, 1, 2], None, None),
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
    assert not window._modern_preview_loading
    assert not window._modern_raw_preview_pending


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


def test_reconstructed_load_clears_previous_raw_preview():
    window, state = _window()
    state.frames = ["previous-raw"]

    def load():
        window.study = SimpleNamespace(reconstructed=True)

    window.load_one_or_two_studies = load
    window.load_modern_studies()
    assert state.stopped and not state.frames


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
    assert state.visible and state.calls == ["process"]
    assert state.frames == ["preview"]
    assert not window._pending_dual_raw_load
    window._begin_modern_raw_processing()
    assert state.stopped and not state.frames
    assert not window._modern_raw_preview_enabled
    assert "procesamiento en curso" in state.results
    window._refresh_modern_raw_preview()
    assert not state.frames


@pytest.mark.parametrize("preview_only", [False, True])
def test_process_current_raw_fast_path_is_exclusive_to_modern(preview_only):
    window, state = _window()
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
        assert window._modern_raw_preview_pending and window._modern_raw_preview_enabled
    else:
        assert state.calls == ["qc"]


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


def test_restart_clears_floating_cine_and_reenables_next_load():
    window, state = _window()
    state.frames = ["previous-preview"]
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
    window._refresh_readonly_results_panel = lambda: None
    window._refresh_persistent_patient_card = lambda: None
    window.preview_movies = {}
    window.preview_pixmaps = {"previous": 1}
    window.preview_labels = {}
    window.gate_dropout_status = SimpleNamespace(setText=lambda text: None)
    window.cine = SimpleNamespace(set_cube=lambda cube: None)
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