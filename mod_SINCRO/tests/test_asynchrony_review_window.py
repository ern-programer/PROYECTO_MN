import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.asynchrony_review_window import AsynchronyReviewWindow, STAGES, STYLE, _StageView
from ui.epar_dialog import EParDialog
from ui.main_window import MainWindow
from viz.perfusion_score_figure import AMBER, BG, BLUE, TEXT


def test_asynchrony_review_reuses_lcars_dialog_and_palette():
    assert issubclass(AsynchronyReviewWindow, EParDialog)
    for color in (AMBER, BG, BLUE, TEXT):
        assert color in STYLE
    for selector in ('QFrame#asyncReviewRail', 'QFrame#asyncStageRow', 'QScrollArea#asyncReviewScroll'):
        assert selector in STYLE
    assert STAGES == ("stress", "rest")


def test_shared_navigation_clamps_each_stage_like_main_window():
    assert AsynchronyReviewWindow._clamped_position((8, 33, 64, 64), 5, 30) == (5, 30)
    assert AsynchronyReviewWindow._clamped_position((8, 28, 64, 64), 9, 30) == (7, 27)
    assert AsynchronyReviewWindow._clamped_position((1, 28, 64, 64), 4, -2) == (0, 0)


def _routing_view(edit_stage="stress"):
    calls = []
    view = AsynchronyReviewWindow.__new__(AsynchronyReviewWindow)
    view._edit_stage = edit_stage
    view._views = {stage: _StageView(stage) for stage in STAGES}
    view._push_manual_centers = lambda: calls.append("push")
    view._set_status = lambda *args, **kwargs: calls.append(("status", args))
    view._main = SimpleNamespace(
        compare_bundle={"study": object()},
        study=object(),
        _on_center_picked=lambda sl, c: calls.append(("stress", sl, c)),
        _on_compare_center_picked=lambda sl, c: calls.append(("rest", sl, c)),
        _clear_manual_centers=lambda: calls.append("clear_stress"),
        _clear_compare_manual_centers=lambda: calls.append("clear_rest"),
        reprocess_async_stage=lambda stage: calls.append(("reprocess", stage)),
    )
    return view, calls


def test_center_click_goes_to_stage_of_clicked_viewer():
    view, calls = _routing_view(edit_stage="stress")
    view._on_center_picked(3, (10.0, 12.0), stage="rest")
    view._on_center_picked(4, None, stage="stress")
    assert calls == [("rest", 3, (10.0, 12.0)), "push", ("stress", 4, None), "push"]


def test_clear_centers_follows_selected_processing_stage():
    for stage, expected in (("stress", ["clear_stress"]), ("rest", ["clear_rest"]), ("both", ["clear_stress", "clear_rest"])):
        view, calls = _routing_view(edit_stage=stage)
        view._on_manual_center_clear()
        assert [call for call in calls if isinstance(call, str) and call.startswith("clear")] == expected


def test_metrics_summary_keeps_psd_first_and_never_diagnoses():
    big, detail = AsynchronyReviewWindow._metrics_summary(
        {"phase_sd": 18.24, "bandwidth": 52, "entropy_normalized_pct": 41, "technical_classification": "normal"},
        {"available": True, "ef_pct": 63.0},
    )
    assert big == "PSD 18.2°"
    assert "BW 52.0°" in detail and "FEVI 63.0%" in detail and "(no dx)" in detail
    assert AsynchronyReviewWindow._metrics_summary(None, None)[0] == "—"


def test_review_status_is_legible_in_lcars_palette():
    label = SimpleNamespace()
    label.setText = lambda text: setattr(label, "text", text)
    label.setStyleSheet = lambda style: setattr(label, "style", style)
    view = SimpleNamespace(status_label=label)
    AsynchronyReviewWindow._set_status(view, "Sin contornos")
    assert label.text == "Sin contornos"
    assert AMBER in label.style
    AsynchronyReviewWindow._set_status(view, "Contornos disponibles", ok=True)
    assert label.text == "Contornos disponibles"
    assert "#90e277" in label.style


class DummyReviewWindow:
    def __init__(self, main_window):
        self.main_window = main_window
        self.visible = False

    def show(self):
        self.visible = True

    def raise_(self):
        return None

    def activateWindow(self):
        return None

    def sync_from_main(self):
        return None

    def isVisible(self):
        return self.visible


def test_open_asynchrony_review_window_creates_window(monkeypatch):
    monkeypatch.setattr("ui.asynchrony_review_window.AsynchronyReviewWindow", DummyReviewWindow)

    window = MainWindow.__new__(MainWindow)
    window._asynchrony_review_window = None
    window._ui_show_helpers = True
    window._ui_enable_tooltips = True
    window._ui_compact_controls = False
    window.study = None

    created = window.open_asynchrony_review_window()

    assert created is not None
    assert created is window._asynchrony_review_window
    assert created.isVisible() is True


def test_contours_for_gate_slice_change_with_gate():
    view = AsynchronyReviewWindow.__new__(AsynchronyReviewWindow)
    view._wall_px_mm = 1.0
    view._show_center_contour = True
    view._show_endo_contour = True
    view._show_epi_contour = True

    seg = SimpleNamespace(center_per_slice=np.array([[10.0, 10.0]], dtype=np.float64))
    result = SimpleNamespace(
        valid_slices=(0,),
        center_radii_mm=np.array(
            [
                [[5.0, 5.0, 5.0, 5.0]],
                [[6.0, 6.0, 6.0, 6.0]],
            ],
            dtype=np.float64,
        ),
        endo_radii_mm=np.array(
            [
                [[3.0, 3.0, 3.0, 3.0]],
                [[4.0, 4.0, 4.0, 4.0]],
            ],
            dtype=np.float64,
        ),
        epi_radii_mm=np.array(
            [
                [[7.0, 7.0, 7.0, 7.0]],
                [[8.0, 8.0, 8.0, 8.0]],
            ],
            dtype=np.float64,
        ),
    )

    c0 = view._contours_for_gate_slice(result, seg, 0, 0)
    c1 = view._contours_for_gate_slice(result, seg, 0, 1)
    assert len(view._contours_for_gate_slice(result, seg, 0, 0, px_mm=2.0)) == 3

    assert len(c0) == 3
    assert len(c1) == 3
    yx0 = np.asarray(c0[1][1], dtype=np.float64)
    yx1 = np.asarray(c1[1][1], dtype=np.float64)
    d0 = np.mean(np.sqrt((yx0[:, 0] - 10.0) ** 2 + (yx0[:, 1] - 10.0) ** 2))
    d1 = np.mean(np.sqrt((yx1[:, 0] - 10.0) ** 2 + (yx1[:, 1] - 10.0) ** 2))
    assert d1 > d0


def test_contour_visibility_toggles_filter_output():
    view = AsynchronyReviewWindow.__new__(AsynchronyReviewWindow)
    view._wall_px_mm = 1.0
    view._show_center_contour = False
    view._show_endo_contour = True
    view._show_epi_contour = False

    seg = SimpleNamespace(center_per_slice=np.array([[10.0, 10.0]], dtype=np.float64))
    result = SimpleNamespace(
        valid_slices=(0,),
        center_radii_mm=np.array([[[5.0, 5.0, 5.0, 5.0]]], dtype=np.float64),
        endo_radii_mm=np.array([[[3.0, 3.0, 3.0, 3.0]]], dtype=np.float64),
        epi_radii_mm=np.array([[[7.0, 7.0, 7.0, 7.0]]], dtype=np.float64),
    )

    contours = view._contours_for_gate_slice(result, seg, 0, 0)
    assert len(contours) == 1


def _seg(method: str):
    return SimpleNamespace(
        method=method,
        center_per_slice=np.zeros((1, 2), dtype=np.float64),
    )


def test_ectb_seed_prefiere_el_anillo_cuando_se_aplico_la_pared():
    """Regresión: aplicar la pared ECTb dejaba al ECTb sin semilla anular.

    `main.seg` pasa a ser la pared irregular, y alimentar al ECTb con su propia
    salida lo deja sin cavidad desde donde tirar rayos: dejaba de dibujar.
    """
    view = AsynchronyReviewWindow.__new__(AsynchronyReviewWindow)
    ring = _seg("auto")
    view._main = SimpleNamespace(seg=_seg("ectb_wall"), seg_ring_base=ring)

    assert view._ectb_seed_segmentation() is ring


def test_ectb_seed_nunca_devuelve_una_pared_como_semilla():
    view = AsynchronyReviewWindow.__new__(AsynchronyReviewWindow)
    wall = _seg("ectb_wall")
    # Sin copia anular disponible, es preferible no calcular a calcular mal.
    view._main = SimpleNamespace(seg=wall, seg_ring_base=wall)

    assert view._ectb_seed_segmentation() is None


def test_ectb_seed_usa_la_segmentacion_normal_sin_pared_aplicada():
    view = AsynchronyReviewWindow.__new__(AsynchronyReviewWindow)
    ring = _seg("auto")
    view._main = SimpleNamespace(seg=ring, seg_ring_base=None)

    assert view._ectb_seed_segmentation() is ring
