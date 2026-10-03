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
             'open_ui_preferences_dialog', 'open_pdf', 'open_html_report'):
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
    assert state.visible and state.calls == ["collapse-modern", "process"]
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


def test_restart_clears_floating_cine_and_reenables_next_load():
    window, state = _window()
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