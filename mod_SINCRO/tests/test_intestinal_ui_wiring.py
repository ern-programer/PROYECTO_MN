"""Tests de la capa de UI de la sustracción intestinal, sin QApplication.

Un QApplication real dentro de pytest crashea el proceso (0xc000001d), así que
se instancian las clases con `__new__` y se les pone solo el estado que la
lógica bajo prueba necesita.

Lo que se verifica acá es el cableado que el motor de `core` no puede cubrir: el
alcance (slice / todos / gate), la conversión de polígonos a máscaras, y —lo más
importante— que la sustracción alimente el cubo de análisis y no solo el de
segmentación.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.cine_widget import CineWidget
from ui.main_window import MainWindow


def _square(cy: float, cx: float, half: float) -> list[tuple[float, float]]:
    return [
        (cy - half, cx - half),
        (cy - half, cx + half),
        (cy + half, cx + half),
        (cy + half, cx - half),
    ]


def _widget(scope: str = "slice", n_slices: int = 4) -> CineWidget:
    w = CineWidget.__new__(CineWidget)
    w._intestinal_roi_polygons = {}
    w._intestinal_roi_polygons_by_gate = {}
    w._intestinal_ref_polygons = {}
    w._intestinal_ref_polygons_by_gate = {}
    w._intestinal_scope_mode = scope
    w._intestinal_mode = "subtract"
    w._intestinal_bg_method = "idw"
    w._intestinal_apply_enabled = True
    w._intestinal_feather_px = 0
    w._intestinal_attenuation_pct = 60
    w._cube = np.zeros((6, n_slices, 20, 20), dtype=np.float64)
    w.current_gate_index = lambda: 0
    return w


def test_referencias_por_slice_no_se_filtran_a_otros_cortes():
    w = _widget(scope="slice")
    w._store_reference_polygon(2, 0, _square(5, 5, 2))

    assert len(w._intestinal_ref_polygons_for_slice(2)) == 1
    assert w._intestinal_ref_polygons_for_slice(1) == []
    assert w._intestinal_ref_polygons_for_slice(3) == []


def test_referencias_alcance_todos_los_slices_se_propagan():
    w = _widget(scope="all_slices", n_slices=4)
    w._store_reference_polygon(0, 0, _square(5, 5, 2))

    for s in range(4):
        assert len(w._intestinal_ref_polygons_for_slice(s)) == 1


def test_se_acumulan_entrada_y_salida_del_asa():
    w = _widget(scope="slice")
    w._store_reference_polygon(1, 0, _square(4, 4, 2))
    w._store_reference_polygon(1, 0, _square(15, 15, 2))

    assert len(w._intestinal_ref_polygons_for_slice(1)) == 2


def test_se_admiten_mas_de_dos_referencias():
    """No hay tope: entrada, salida y las que hagan falta."""
    w = _widget(scope="slice")
    w._store_reference_polygon(1, 0, _square(4, 4, 2))
    w._store_reference_polygon(1, 0, _square(10, 4, 2))
    w._store_reference_polygon(1, 0, _square(15, 15, 2))

    assert len(w._intestinal_ref_polygons_for_slice(1)) == 3
    refs = w.intestinal_reference_masks((20, 20), 4)
    assert len(refs[1]) == 3


def test_main_window_usa_el_metodo_media_simple():
    w = _widget(scope="slice")
    w._intestinal_bg_method = "mean"
    w._intestinal_roi_polygons[1] = _square(10, 10, 3)
    w._store_reference_polygon(1, 0, _square(4, 4, 2))
    w._store_reference_polygon(1, 0, _square(16, 4, 2))
    w._store_reference_polygon(1, 0, _square(16, 16, 2))
    w.intestinal_apply_enabled = lambda: True
    w.intestinal_mode = lambda: w._intestinal_mode

    cube = np.full((6, 4, 20, 20), 100.0)
    cube[:, 1, 2:7, 2:7] = 120.0
    cube[:, 1, 14:19, 2:7] = 150.0
    cube[:, 1, 14:19, 14:19] = 180.0

    mw = _main_window_with(w)
    _out, info = mw._apply_intestinal_subtraction_to_cube(cube, w)

    assert info is not None
    assert info["method"] == "mean"
    detail = info["per_slice"][1]
    assert detail["n_references"] == 3
    # (120 + 150 + 180) / 3
    assert abs(detail["background_level"] - 150.0) < 1e-9


def test_mascaras_y_pesos_salen_con_la_forma_del_frame():
    w = _widget(scope="slice")
    w._intestinal_roi_polygons[1] = _square(10, 10, 3)
    w._store_reference_polygon(1, 0, _square(4, 4, 2))

    weights = w.intestinal_target_weights((20, 20), 4)
    refs = w.intestinal_reference_masks((20, 20), 4)

    assert set(weights) == {1}
    assert weights[1].shape == (20, 20)
    assert weights[1].max() > 0
    assert set(refs) == {1}
    assert refs[1][0].dtype == bool
    assert refs[1][0].sum() > 0


def test_sin_roi_a_corregir_no_se_devuelve_nada():
    w = _widget(scope="slice")
    w._store_reference_polygon(1, 0, _square(4, 4, 2))
    assert w.intestinal_target_weights((20, 20), 4) == {}


def _main_window_with(widget) -> MainWindow:
    mw = MainWindow.__new__(MainWindow)
    mw.intestinal_subtraction_info = None
    mw._log = lambda *_a, **_k: None
    return mw


def test_main_window_no_resta_en_modo_atenuar():
    """La atenuación porcentual no debe tocar el cubo de análisis."""
    w = _widget(scope="slice")
    w._intestinal_mode = "attenuate"
    w._intestinal_roi_polygons[1] = _square(10, 10, 3)
    w._store_reference_polygon(1, 0, _square(4, 4, 2))
    w.intestinal_apply_enabled = lambda: True
    w.intestinal_mode = lambda: w._intestinal_mode

    mw = _main_window_with(w)
    cube = np.full((6, 4, 20, 20), 100.0)
    out, info = mw._apply_intestinal_subtraction_to_cube(cube, w)

    assert info is None
    assert np.array_equal(out, cube)


def test_main_window_resta_y_reporta_en_modo_subtract():
    w = _widget(scope="slice")
    w._intestinal_roi_polygons[1] = _square(10, 10, 3)
    w._store_reference_polygon(1, 0, _square(4, 4, 2))
    w.intestinal_apply_enabled = lambda: True
    w.intestinal_mode = lambda: w._intestinal_mode

    cube = np.full((6, 4, 20, 20), 100.0)
    cube[:, 1, 8:13, 8:13] += 250.0  # asa caliente sobre la zona a corregir

    mw = _main_window_with(w)
    out, info = mw._apply_intestinal_subtraction_to_cube(cube, w)

    assert info is not None
    assert info["applied"]
    assert info["slices_corrected"] == [1]
    assert info["counts_subtracted"] > 0
    # Los cortes sin ROI quedan intactos.
    for s in (0, 2, 3):
        assert np.array_equal(out[:, s], cube[:, s])
    # Y en el corregido bajaron las cuentas.
    assert out[:, 1, 10, 10].max() < cube[:, 1, 10, 10].max()


def test_main_window_sin_referencia_no_corrige():
    w = _widget(scope="slice")
    w._intestinal_roi_polygons[1] = _square(10, 10, 3)
    w.intestinal_apply_enabled = lambda: True
    w.intestinal_mode = lambda: w._intestinal_mode

    mw = _main_window_with(w)
    cube = np.full((6, 4, 20, 20), 100.0)
    out, info = mw._apply_intestinal_subtraction_to_cube(cube, w)

    assert info is not None
    assert not info["applied"]
    assert info["slices_without_reference"] == [1]
    assert np.array_equal(out, cube)
