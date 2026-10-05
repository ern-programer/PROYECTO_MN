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