import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.asynchrony_review_window import AsynchronyReviewWindow
from ui.main_window import MainWindow


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
