"""Ventana LCARS de atenuación intestinal del montaje clínico: ROI manual por corte y ajuste por gate (QC)."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from ui.cine_widget import CineWidget, RangeSlider, VerticalColorStrip, _resolve_cmap
from ui.epar_dialog import EParDialog
from viz.perfusion_score_figure import AMBER, BG, BLUE, PANEL, TEXT

LILAC = "#c79bc4"
CYAN = "#81c7ce"

STYLE = f"""
QWidget#intestinalLcars {{ background:{BG}; color:{TEXT}; font-family:'Arial Narrow','Bahnschrift Condensed','Segoe UI'; }}
QFrame#lcarsBar {{ background:{AMBER}; border-top-left-radius:16px; border-bottom-left-radius:4px; }}
QFrame#lcarsRail {{ background:{PANEL}; border-left:10px solid {BLUE}; border-top-left-radius:18px; border-bottom-left-radius:18px; }}
QFrame#readout {{ background:{PANEL}; border:1px solid #343945; border-radius:10px; }}
QPushButton[lcars="true"] {{ border:none; border-radius:13px; padding:7px 12px; color:#172028;
  font-weight:800; background:{BLUE}; text-align:right; }}
QPushButton[lcars="true"]:checked {{ background:{AMBER}; }}
QPushButton[lcars="true"]:hover {{ background:#9bb6bd; }}
QPushButton[lcars="true"]:disabled {{ background:#2a2e36; color:#5b6670; }}
QPushButton#accept {{ background:#90e277; }}
QPushButton#reject {{ background:#a7615e; color:white; }}
QPushButton#mini {{ border:none; border-radius:8px; padding:2px 8px; color:#172028; font-weight:800; background:{BLUE}; }}
QPushButton#mini:hover {{ background:{AMBER}; }}
QLabel#caption {{ color:{BLUE}; font-size:10px; font-weight:800; background:transparent; }}
QLabel#head {{ color:{AMBER}; font-weight:800; background:transparent; }}
QLabel#value {{ color:{TEXT}; background:transparent; }}
QLabel#bigValue {{ color:{AMBER}; font-size:22px; font-weight:800; background:transparent; }}
QLabel#status {{ color:{CYAN}; font-weight:700; background:transparent; }}
QLabel#small {{ color:#8a979c; font-size:9px; background:transparent; }}
QComboBox {{ background:#1b2230; color:{TEXT}; border:1px solid {BLUE}; border-radius:8px; padding:2px 8px; }}
QComboBox QAbstractItemView {{ background:#1b2230; color:{TEXT}; selection-background-color:{AMBER};
  selection-color:#172028; }}
QSlider::groove:horizontal {{ height:6px; background:#2a2e36; border-radius:3px; }}
QSlider::sub-page:horizontal {{ background:{BLUE}; border-radius:3px; }}
QSlider::handle:horizontal {{ background:{AMBER}; width:14px; margin:-5px 0; border-radius:7px; }}
"""


def montage_display_norm(volume, mode: str, win_low: float, win_high: float,
                         lin_low: float, lin_high: float) -> np.ndarray:
    """Misma normalización por volumen que el montaje clínico (percentil o lineal 0..200%)."""
    v = np.asarray(volume, dtype=np.float64)
    if v.size == 0:
        return v
    if mode == "lineal":
        vmin, vmax = float(v.min()), float(v.max())
        norm = (v - vmin) / max(vmax - vmin, 1e-8) if vmax > vmin else np.zeros_like(v)
        w0 = max(0.0, float(lin_low))
        w1 = max(0.0, min(2.0, float(lin_high)))
        if w1 <= w0:
            w1 = min(2.0, w0 + 0.01)
        return np.clip((norm - w0) / max(1e-8, w1 - w0), 0.0, 1.0)
    lo = float(win_low or 0.0)
    hi = float(win_high or 100.0)
    if hi <= lo:
        hi = min(100.0, lo + 1.0)
    top, base = float(np.percentile(v, hi)), float(np.percentile(v, lo))
    return np.clip((v - base) / max(top - base, 1e-8), 0.0, 1.0)


def _points(item) -> list[tuple[float, float]]:
    try:
        return [(float(p[0]), float(p[1])) for p in (item.get("points") or [])]
    except Exception:
        return []


class IntestinalRoiModel:
    """ROI intestinal por corte (común a todos los gates) con ajuste opcional por gate (QC)."""

    def __init__(self, n_gates: int, n_slices: int):
        self.n_gates, self.n_slices = int(n_gates), int(n_slices)
        self.slice_polys: dict[int, list] = {}
        self.gate_polys: dict[tuple[int, int], list] = {}
        self.ref_slice: dict[int, list] = {}
        self.ref_gate: dict[tuple[int, int], list] = {}
        self.mode = "attenuate"
        self.bg_method = "idw"

    @classmethod
    def from_state(cls, state, scope: str, n_gates: int, n_slices: int) -> "IntestinalRoiModel":
        """Normaliza un estado del visor a polígonos explícitos con el mismo efecto que su alcance."""
        model = cls(n_gates, n_slices)
        state = state if isinstance(state, dict) else {}
        for item in state.get("slice_polygons") or []:
            pts = _points(item)
            if len(pts) >= 3:
                model.slice_polys[int(item["slice"])] = pts
        for item in state.get("gate_polygons") or []:
            pts = _points(item)
            if len(pts) >= 3:
                model.gate_polys[(int(item["gate"]), int(item["slice"]))] = pts
        for item in state.get("reference_slice_polygons") or []:
            pts = _points(item)
            if len(pts) >= 3:
                model.ref_slice.setdefault(int(item["slice"]), []).append(pts)
        for item in state.get("reference_gate_polygons") or []:
            pts = _points(item)
            if len(pts) >= 3:
                model.ref_gate.setdefault((int(item["gate"]), int(item["slice"])), []).append(pts)
        model.mode = str(state.get("mode") or "attenuate")
        model.bg_method = str(state.get("background_method") or "idw")
        scope = str(scope or "slice")
        if scope == "all_slices" and model.slice_polys:
            first = next(iter(model.slice_polys.values()))
            model.slice_polys = {s: model.slice_polys.get(s, first) for s in range(model.n_slices)}
        if scope == "gate_slices":
            model.gate_polys = cls._expand_gate_fallback(model.gate_polys, model.n_gates, model.n_slices)
            model.ref_gate = cls._expand_gate_fallback(model.ref_gate, model.n_gates, model.n_slices)
            model.slice_polys, model.ref_slice = {}, {}
        elif scope != "gate_override":
            model.gate_polys, model.ref_gate = {}, {}
        return model

    @staticmethod
    def _expand_gate_fallback(polys: dict, n_gates: int, n_slices: int) -> dict:
        out = {}
        for g in range(n_gates):
            fallback = next((p for (gg, _s), p in polys.items() if int(gg) == g and p), None)
            for s in range(n_slices):
                value = polys.get((g, s)) or fallback
                if value:
                    out[(g, s)] = value
        return out

    def effective(self, gate: int, sl: int) -> tuple[list, str | None]:
        poly = self.gate_polys.get((int(gate), int(sl)))
        if poly:
            return poly, "gate"
        poly = self.slice_polys.get(int(sl))
        return (poly, "slice") if poly else ([], None)

    def set_slice(self, sl: int, pts, *, all_slices: bool = False):
        """ROI común del corte; descarta los ajustes por gate de esos cortes (lo último dibujado manda)."""
        pts = [tuple(map(float, p)) for p in pts]
        slices = set(range(self.n_slices)) if all_slices else {int(sl)}
        for s in slices:
            self.slice_polys[s] = list(pts)
        self.gate_polys = {k: v for k, v in self.gate_polys.items() if k[1] not in slices}

    def clear_slice(self, sl: int, *, all_slices: bool = False):
        slices = set(range(self.n_slices)) if all_slices else {int(sl)}
        self.slice_polys = {s: p for s, p in self.slice_polys.items() if s not in slices}
        self.gate_polys = {k: v for k, v in self.gate_polys.items() if k[1] not in slices}

    def set_gate(self, gate: int, sl: int, pts, *, all_slices: bool = False):
        pts = [tuple(map(float, p)) for p in pts]
        for s in (range(self.n_slices) if all_slices else [int(sl)]):
            self.gate_polys[(int(gate), s)] = list(pts)

    def clear_gate(self, gate: int, sl: int, *, all_slices: bool = False):
        for s in (range(self.n_slices) if all_slices else [int(sl)]):
            self.gate_polys.pop((int(gate), s), None)

    def gate_overrides(self, sl: int) -> dict[int, list]:
        return {g: p for (g, s), p in sorted(self.gate_polys.items()) if s == int(sl)}

    def copy_gate_to_all(self, gate: int, sl: int):
        poly, _source = self.effective(gate, sl)
        if poly:
            for g in range(self.n_gates):
                self.gate_polys[(g, int(sl))] = list(poly)

    def reset_gates(self, sl: int):
        self.gate_polys = {k: v for k, v in self.gate_polys.items() if k[1] != int(sl)}

    def scope(self) -> str:
        return "gate_override" if (self.gate_polys or self.ref_gate) else "slice"

    def counts(self) -> tuple[int, int]:
        return len(self.slice_polys), len(self.gate_polys)

    def to_state(self) -> dict:
        def pts(poly):
            return [[float(a), float(b)] for a, b in poly]
        return {
            "slice_polygons": [{"slice": s, "points": pts(p)} for s, p in sorted(self.slice_polys.items())],
            "gate_polygons": [{"gate": g, "slice": s, "points": pts(p)} for (g, s), p in sorted(self.gate_polys.items())],
            "reference_slice_polygons": [{"slice": s, "points": pts(p)}
                                         for s, polys in sorted(self.ref_slice.items()) for p in polys],
            "reference_gate_polygons": [{"gate": g, "slice": s, "points": pts(p)}
                                        for (g, s), polys in sorted(self.ref_gate.items()) for p in polys],
            "mode": self.mode,
            "background_method": self.bg_method,
        }


def polygon_area(poly) -> float:
    if not poly or len(poly) < 3:
        return 0.0
    ys, xs = np.asarray(poly, dtype=np.float64).T
    return 0.5 * abs(float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))


def _rgb_image(norm: np.ndarray, cmap_name: str) -> QImage:
    rgb = np.ascontiguousarray((np.asarray(_resolve_cmap(cmap_name)(norm))[..., :3] * 255).astype(np.uint8))
    height, width, _ = rgb.shape
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


def _fit_rect(area: QRectF, image: QImage) -> QRectF:
    scale = min(area.width() / max(1, image.width()), area.height() / max(1, image.height()))
    w, h = image.width() * scale, image.height() * scale
    return QRectF(area.center().x() - w / 2.0, area.center().y() - h / 2.0, w, h)


def _draw_polygon(painter: QPainter, rect: QRectF, shape: tuple[int, int], poly, color: str, *,
                  dashed: bool = False, closed: bool = True, width: float = 2.0):
    if not poly or len(poly) < 2:
        return
    sx, sy = rect.width() / max(1, shape[1]), rect.height() / max(1, shape[0])
    pts = [QPointF(rect.x() + cx * sx, rect.y() + cy * sy) for cy, cx in poly]
    pen = QPen(QColor(color), width, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
    painter.setPen(pen)
    fill = QColor(color)
    fill.setAlpha(0 if dashed or not closed else 40)
    painter.setBrush(fill)
    if closed and len(pts) >= 3:
        painter.drawPolygon(QPolygonF(pts))
    else:
        painter.drawPolyline(QPolygonF(pts))


class IntestinalCanvas(QWidget):
    """Imagen del corte con polígonos; en modo dibujo: clic agrega, doble clic cierra, clic der. borra."""

    polygonFinished = pyqtSignal(object)
    polygonCleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image: QImage | None = None
        self.shape = (1, 1)
        self.overlays: list[tuple[list, str, bool]] = []
        self.title = ""
        self.draw_mode = False
        self.draft: list[tuple[float, float]] = []
        self.setMinimumSize(360, 360)
        self.setMouseTracking(True)

    def set_content(self, image: QImage | None, shape, overlays, title: str):
        self.image, self.shape, self.overlays, self.title = image, tuple(shape), list(overlays), title
        self.update()

    def set_draw_mode(self, enabled: bool):
        self.draw_mode = bool(enabled)
        self.draft = []
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def _rect(self) -> QRectF | None:
        if self.image is None:
            return None
        return _fit_rect(QRectF(4, 26, self.width() - 8, self.height() - 30), self.image)

    def _to_image(self, pos) -> tuple[float, float] | None:
        rect = self._rect()
        if rect is None or not rect.contains(pos):
            return None
        return ((pos.y() - rect.y()) / rect.height() * self.shape[0],
                (pos.x() - rect.x()) / rect.width() * self.shape[1])

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG))
        painter.setFont(QFont("Arial Narrow", 10, QFont.Weight.Bold))
        painter.setPen(QColor(AMBER))
        painter.drawText(QRectF(8, 2, self.width() - 16, 22), Qt.AlignmentFlag.AlignVCenter, self.title.upper())
        rect = self._rect()
        if rect is None:
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(rect, self.image)
        painter.setPen(QPen(QColor(BLUE), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        for poly, color, dashed in self.overlays:
            _draw_polygon(painter, rect, self.shape, poly, color, dashed=dashed)
        if self.draft:
            _draw_polygon(painter, rect, self.shape, self.draft, CYAN, closed=False)

    def mousePressEvent(self, event):
        if not self.draw_mode:
            return
        mapped = self._to_image(event.position())
        if event.button() == Qt.MouseButton.RightButton:
            self.draft = []
            self.polygonCleared.emit()
        elif event.button() == Qt.MouseButton.LeftButton and mapped is not None:
            self.draft.append(mapped)
        self.update()

    def mouseDoubleClickEvent(self, event):
        if not self.draw_mode or event.button() != Qt.MouseButton.LeftButton:
            return
        mapped = self._to_image(event.position())
        if mapped is not None:
            self.draft.append(mapped)
        if polygon_area(self.draft) >= 1.0:
            self.polygonFinished.emit(list(self.draft))
        self.draft = []
        self.update()


class GateMosaic(QWidget):
    """QC gates: todos los gates del corte con su ROI efectiva; clic selecciona el gate."""

    gateSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tiles: list[tuple[QImage, list]] = []
        self.shape = (1, 1)
        self.selected = 0
        self.setFixedWidth(230)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_tiles(self, tiles, shape, selected: int):
        self.tiles, self.shape, self.selected = list(tiles), tuple(shape), int(selected)
        self.update()

    def _cells(self) -> list[QRectF]:
        n = len(self.tiles)
        cols = 2 if n > 1 else 1
        rows = max(1, -(-n // cols))
        cw, ch = (self.width() - 6) / cols, (self.height() - 6) / rows
        side = min(cw, ch)
        return [QRectF(3 + (i % cols) * cw + (cw - side) / 2, 3 + (i // cols) * ch + (ch - side) / 2, side, side)
                for i in range(n)]

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG))
        painter.setFont(QFont("Arial Narrow", 9, QFont.Weight.Bold))
        for gate, (cell, (image, overlays)) in enumerate(zip(self._cells(), self.tiles)):
            inner = cell.adjusted(3, 3, -3, -3)
            rect = _fit_rect(inner, image)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(rect, image)
            for poly, color, dashed in overlays:
                _draw_polygon(painter, rect, self.shape, poly, color, dashed=dashed, width=1.4)
            selected = gate == self.selected
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(AMBER if selected else "#343945"), 3.0 if selected else 1.0))
            painter.drawRoundedRect(cell.adjusted(1, 1, -1, -1), 6, 6)
            painter.setPen(QColor(AMBER if selected else BLUE))
            painter.drawText(cell.adjusted(6, 2, -4, -2), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                             f"G{gate + 1}")

    def mousePressEvent(self, event):
        for gate, cell in enumerate(self._cells()):
            if cell.contains(event.position()):
                self.gateSelected.emit(gate)
                return


class LcarsRangeSlider(RangeSlider):
    """Mismo control Base/Top 0..200% del montaje clínico, con paleta LCARS."""

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx = self.width() / 2.0
        y0, y1 = self._val_to_y(self.RANGE), self._val_to_y(0)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#2a2e36"))
        p.drawRoundedRect(QRectF(cx - 3, y0, 6, y1 - y0), 3, 3)
        ya, yb = self._val_to_y(self._high), self._val_to_y(self._low)
        p.setBrush(QColor(BLUE))
        p.drawRoundedRect(QRectF(cx - 3, ya, 6, yb - ya), 3, 3)
        p.setPen(QPen(QColor("#5b6670"), 1))
        for mark in self.STOP_MARKS:
            ym = self._val_to_y(mark)
            p.drawLine(QPointF(cx - 7, ym), QPointF(cx + 7, ym))
        for val, color in ((self._low, CYAN), (self._high, AMBER)):
            y = self._val_to_y(val)
            p.setBrush(QColor(color))
            p.setPen(QPen(QColor("#172028"), 1))
            p.drawRoundedRect(QRectF(cx - 10, y - 6, 20, 12), 6, 6)


class _Stage:
    def __init__(self, stage: str, cube, target, engine: CineWidget):
        self.stage = stage
        self.cube = np.asarray(cube, dtype=np.float64)
        self.target = target
        self.engine = engine
        n_gates, n_slices = int(self.cube.shape[0]), int(self.cube.shape[1])
        self.model = IntestinalRoiModel.from_state(target.intestinal_roi_state(), target.intestinal_scope(),
                                                   n_gates, n_slices)
        self.params = tuple(int(v) for v in target.intestinal_params())
        self.enabled = bool(target.intestinal_apply_enabled())
        self.slice = n_slices // 2
        self.gate = 0
        self.dirty = False
        self.masked: list[np.ndarray] | None = None

    @property
    def label(self) -> str:
        return "REPOSO" if self.stage == "rest" else "ESFUERZO"


class MontageIntestinalWindow(EParDialog):
    """Atenuación intestinal del montaje clínico: dibujo por corte, QC por gate y ventana Base/Top del montaje."""

    def __init__(self, parent=None, *, sources, apply_stage, display: dict | None = None, settings=None):
        super().__init__(parent, title="ATENUACIÓN INTESTINAL · MONTAJE CLÍNICO", accent=AMBER,
                         settings=settings, geometry_key="dialogs/montage_intestinal_lcars/geometry")
        self._apply_stage = apply_stage
        display = dict(display or {})
        self.cmap = str(display.get("cmap") or "gray")
        self.cmaps = list(display.get("cmaps") or _cmap_names())
        self.win_mode = str(display.get("mode") or "percentil")
        self.win = {"win_low": float(display.get("win_low", 2.0)), "win_high": float(display.get("win_high", 99.5)),
                    "lin_low": float(display.get("lin_low", 0.0)), "lin_high": float(display.get("lin_high", 1.0))}
        self.stages: list[_Stage] = []
        for stage, cube, target in sources:
            engine = CineWidget(self)
            engine.hide()
            self.stages.append(_Stage(stage, cube, target, engine))
        self.current = 0
        self._build()
        self._load_stage()
        self.resize(1320, 820)

    # ---------- construcción ----------
    def _pill(self, text: str, *, checkable: bool = False, checked: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("lcars", True)
        button.setCheckable(checkable)
        button.setChecked(checked)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _caption(self, text: str, name: str = "caption") -> QLabel:
        label = QLabel(text)
        label.setObjectName(name)
        return label

    def _slider(self, maximum: int, tip: str) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, maximum)
        slider.setToolTip(tip)
        return slider

    def _build(self):
        body = QWidget()
        body.setObjectName("intestinalLcars")
        body.setStyleSheet(STYLE)
        self.content_layout.addWidget(body, 1)
        outer = QVBoxLayout(body)
        bar = QFrame()
        bar.setObjectName("lcarsBar")
        bar.setFixedHeight(12)
        outer.addWidget(bar)
        main = QHBoxLayout()
        outer.addLayout(main, 1)
        main.addWidget(self._build_rail())

        center = QVBoxLayout()
        views = QHBoxLayout()
        self.mosaic = GateMosaic()
        self.mosaic.gateSelected.connect(self._on_gate_selected)
        self.mosaic.hide()
        views.addWidget(self.mosaic)
        self.canvas = IntestinalCanvas()
        self.canvas.polygonFinished.connect(self._on_polygon_finished)
        self.canvas.polygonCleared.connect(self._on_polygon_cleared)
        views.addWidget(self.canvas, 1)
        center.addLayout(views, 1)
        nav = QHBoxLayout()
        nav.addWidget(self._caption("CORTE"))
        self.slice_slider = self._slider(0, "Corte SA (base → ápex).")
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        nav.addWidget(self.slice_slider, 1)
        self.slice_label = self._caption("", "value")
        self.slice_label.setMinimumWidth(56)
        nav.addWidget(self.slice_label)
        center.addLayout(nav)
        main.addLayout(center, 1)

        main.addWidget(self._build_window_column())
        main.addWidget(self._build_readout())

    def _build_rail(self) -> QFrame:
        rail = QFrame()
        rail.setObjectName("lcarsRail")
        rail.setFixedWidth(200)
        lay = QVBoxLayout(rail)
        lay.setContentsMargins(8, 12, 8, 12)
        lay.addWidget(self._caption("ETAPA"))
        self.stage_group = QButtonGroup(self)
        self.stage_buttons = []
        for index, stage in enumerate(self.stages):
            button = self._pill(stage.label, checkable=True, checked=index == 0)
            self.stage_group.addButton(button, index)
            self.stage_buttons.append(button)
            lay.addWidget(button)
        self.stage_group.idClicked.connect(self._on_stage_clicked)
        lay.addSpacing(8)
        lay.addWidget(self._caption("ROI INTESTINAL"))
        self.btn_draw = self._pill("DIBUJAR ROI", checkable=True)
        self.btn_draw.setToolTip("Clic izq. agrega vértice · doble clic cierra · clic der. borra.")
        self.btn_draw.toggled.connect(self.canvas_draw_mode)
        self.btn_all = self._pill("TODOS LOS CORTES", checkable=True, checked=True)
        self.btn_all.setToolTip("Vista por corte: el ROI dibujado o borrado vale para todos los cortes.\n"
                                "QC: el ROI va al gate elegido en todos los cortes.\n"
                                "Dibujar el ROI del corte reemplaza los ajustes por gate de esos cortes.")
        self.btn_clear = self._pill("BORRAR ROI")
        self.btn_clear.setToolTip("Vista por corte: borra el ROI del corte. QC: quita el ajuste del gate.")
        self.btn_clear.clicked.connect(self._on_polygon_cleared)
        for button in (self.btn_draw, self.btn_all, self.btn_clear):
            lay.addWidget(button)
        lay.addSpacing(8)
        lay.addWidget(self._caption("QC GATES"))
        self.btn_qc = self._pill("QC GATES", checkable=True)
        self.btn_qc.setToolTip("Mosaico de todos los gates del corte: el ROI dibujado afecta SOLO al gate elegido.")
        self.btn_qc.toggled.connect(self._on_qc_toggled)
        self.btn_copy = self._pill("COPIAR A TODOS")
        self.btn_copy.setToolTip("Copia el ROI del gate elegido a todos los gates de este corte.")
        self.btn_copy.clicked.connect(self._on_copy_to_all)
        self.btn_reset = self._pill("RESTABLECER GATES")
        self.btn_reset.setToolTip("Quita los ajustes por gate de este corte; vuelven a usar el ROI del corte.")
        self.btn_reset.clicked.connect(self._on_reset_gates)
        for button in (self.btn_qc, self.btn_copy, self.btn_reset):
            lay.addWidget(button)
        lay.addSpacing(8)
        lay.addWidget(self._caption("ATENUACIÓN"))
        self.btn_enabled = self._pill("ATENUACIÓN ON", checkable=True)
        self.btn_enabled.toggled.connect(self._on_enabled_toggled)
        lay.addWidget(self.btn_enabled)
        row = QHBoxLayout()
        self.atten_slider = self._slider(100, "Porcentaje de reducción de cuentas dentro del ROI.")
        self.atten_slider.setTracking(False)
        self.atten_slider.valueChanged.connect(self._on_params_changed)
        self.atten_label = self._caption("", "value")
        self.atten_slider.sliderMoved.connect(lambda v: self.atten_label.setText(f"{v}%"))
        row.addWidget(self.atten_slider, 1)
        row.addWidget(self.atten_label)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.feather_slider = self._slider(12, "Borde blando alrededor del ROI (px).")
        self.feather_slider.setTracking(False)
        self.feather_slider.valueChanged.connect(self._on_params_changed)
        self.feather_label = self._caption("", "value")
        self.feather_slider.sliderMoved.connect(lambda v: self.feather_label.setText(f"{v} px"))
        row.addWidget(self.feather_slider, 1)
        row.addWidget(self.feather_label)
        lay.addLayout(row)
        self.btn_original = self._pill("VER ORIGINAL", checkable=True)
        self.btn_original.setToolTip("Muestra el corte sin atenuar para comparar.")
        self.btn_original.toggled.connect(self._refresh)
        lay.addWidget(self.btn_original)
        lay.addStretch(1)
        hint = self._caption("Lila: ROI del corte\nÁmbar: ajuste del gate (QC)\n"
                             "Clic izq. vértice · doble clic cierra\nClic der. borra", "small")
        lay.addWidget(hint)
        return rail

    def canvas_draw_mode(self, enabled: bool):
        self.canvas.set_draw_mode(enabled)

    def _build_window_column(self) -> QFrame:
        col = QFrame()
        col.setObjectName("readout")
        col.setFixedWidth(128)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(6, 8, 6, 8)
        lay.addWidget(self._caption("ESCALA"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(self.cmaps if self.cmap in self.cmaps else self.cmaps + [self.cmap])
        self.cmap_combo.setCurrentText(self.cmap)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        lay.addWidget(self.cmap_combo)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Percentil", "percentil")
        self.mode_combo.addItem("Lineal", "lineal")
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(self.win_mode)))
        self.mode_combo.setToolTip("Percentil: ventana por percentiles del volumen.\n"
                                   "Lineal: min/máx del volumen y ventana 0–200%.")
        self.mode_combo.currentIndexChanged.connect(self._on_win_mode_changed)
        lay.addWidget(self.mode_combo)
        top = QHBoxLayout()
        self.btn_top = QPushButton("TOP")
        self.btn_top.setObjectName("mini")
        self.btn_top.setToolTip("Volver Top a 100%")
        self.btn_top.clicked.connect(self._reset_top)
        self.top_label = self._caption("", "value")
        top.addWidget(self.btn_top)
        top.addWidget(self.top_label)
        lay.addLayout(top)
        sliders = QHBoxLayout()
        self.range_slider = LcarsRangeSlider()
        self.range_slider.valuesChanged.connect(self._on_window_changed)
        sliders.addWidget(self.range_slider, 1)
        self.color_strip = VerticalColorStrip(self.cmap)
        sliders.addWidget(self.color_strip)
        lay.addLayout(sliders, 1)
        base = QHBoxLayout()
        self.btn_base = QPushButton("BASE")
        self.btn_base.setObjectName("mini")
        self.btn_base.setToolTip("Volver Base a 0%")
        self.btn_base.clicked.connect(self._reset_base)
        self.base_label = self._caption("", "value")
        base.addWidget(self.btn_base)
        base.addWidget(self.base_label)
        lay.addLayout(base)
        self._slider_from_window()
        return col

    def _build_readout(self) -> QFrame:
        readout = QFrame()
        readout.setObjectName("readout")
        readout.setFixedWidth(240)
        lay = QVBoxLayout(readout)
        lay.addWidget(self._caption("ESTADO", "head"))
        self.stage_value = self._caption("", "bigValue")
        lay.addWidget(self.stage_value)
        self.position_value = self._caption("", "value")
        lay.addWidget(self.position_value)
        lay.addSpacing(6)
        lay.addWidget(self._caption("ROI"))
        self.count_value = self._caption("", "value")
        self.count_value.setWordWrap(True)
        lay.addWidget(self.count_value)
        lay.addSpacing(6)
        lay.addWidget(self._caption("MODO"))
        self.mode_value = self._caption("", "value")
        self.mode_value.setWordWrap(True)
        lay.addWidget(self.mode_value)
        lay.addSpacing(8)
        self.status = self._caption("", "status")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        lay.addStretch(1)
        note = self._caption("Solo visual: no cambia fase, métricas ni FEVI.\n"
                             "Cerrar descarta lo no aplicado.", "small")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.btn_apply = self._pill("APLICAR ETAPA")
        self.btn_apply.setObjectName("accept")
        self.btn_apply.clicked.connect(self._on_apply)
        lay.addWidget(self.btn_apply)
        close = self._pill("CERRAR")
        close.setObjectName("reject")
        close.clicked.connect(self.reject)
        lay.addWidget(close)
        return readout

    # ---------- estado ----------
    @property
    def stage(self) -> _Stage:
        return self.stages[self.current]

    def _load_stage(self):
        st = self.stage
        for widget, value in ((self.atten_slider, st.params[0]), (self.feather_slider, st.params[1])):
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)
        self.btn_enabled.blockSignals(True)
        self.btn_enabled.setChecked(st.enabled)
        self.btn_enabled.blockSignals(False)
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, max(0, st.cube.shape[1] - 1))
        self.slice_slider.setValue(st.slice)
        self.slice_slider.blockSignals(False)
        gated = st.cube.shape[0] > 1
        self.btn_qc.setEnabled(gated)
        self.btn_qc.setToolTip("Mosaico de todos los gates del corte: el ROI dibujado afecta SOLO al gate elegido."
                               if gated else "La fuente del montaje no tiene gates.")
        if not gated and self.btn_qc.isChecked():
            self.btn_qc.setChecked(False)
        self._refresh()

    def _mark_dirty(self):
        st = self.stage
        st.dirty = True
        st.masked = None
        self._refresh()

    def _sync_engine(self, st: _Stage):
        st.engine.set_intestinal_params(*st.params)
        st.engine.set_intestinal_scope(st.model.scope())
        st.engine.set_intestinal_roi_state(st.model.to_state())
        st.engine.set_intestinal_apply_enabled(st.enabled)

    def _masked(self, st: _Stage) -> list[np.ndarray]:
        if st.masked is None:
            self._sync_engine(st)
            st.masked = [np.asarray(st.engine.apply_intestinal_mask_to_gate_volume(st.cube[g], gate_index=g))
                         for g in range(st.cube.shape[0])]
        return st.masked

    def _norm(self, volume) -> np.ndarray:
        return montage_display_norm(volume, self.win_mode, **self.win)

    def _overlays(self, st: _Stage, gate: int | None) -> list:
        if gate is None:
            poly = st.model.slice_polys.get(st.slice)
            overlays = [(poly, LILAC, False)] if poly else []
            seen = []
            for override in st.model.gate_overrides(st.slice).values():
                if override not in seen:
                    seen.append(override)
                    overlays.append((override, AMBER, True))
            return overlays
        poly, source = st.model.effective(gate, st.slice)
        if source == "gate":
            return [(poly, AMBER, False)]
        return [(poly, LILAC, True)] if poly else []

    # ---------- render ----------
    def _refresh(self, *_args):
        st = self.stage
        volumes = [st.cube[g] for g in range(st.cube.shape[0])] if self.btn_original.isChecked() else self._masked(st)
        shape = st.cube.shape[2:]
        qc = self.btn_qc.isChecked()
        if qc:
            tiles = [(_rgb_image(self._norm(v)[st.slice], self.cmap), self._overlays(st, g)) for g, v in enumerate(volumes)]
            self.mosaic.set_tiles(tiles, shape, st.gate)
            image, overlays = tiles[st.gate]
            title = f"{st.label} · corte {st.slice + 1} · gate {st.gate + 1}"
        else:
            image = _rgb_image(self._norm(np.sum(volumes, axis=0))[st.slice], self.cmap)
            overlays = self._overlays(st, None)
            title = f"{st.label} · corte {st.slice + 1} · todos los gates"
        if self.btn_original.isChecked():
            title += " · original"
        self.mosaic.setVisible(qc)
        self.canvas.set_content(image, shape, overlays, title)
        self._refresh_readout()

    def _refresh_readout(self):
        st = self.stage
        n_slices = st.cube.shape[1]
        self.slice_label.setText(f"{st.slice + 1} / {n_slices}")
        self.atten_label.setText(f"{self.atten_slider.value()}%")
        self.feather_label.setText(f"{self.feather_slider.value()} px")
        self.stage_value.setText(st.label)
        gate_txt = f" · gate {st.gate + 1}/{st.cube.shape[0]}" if self.btn_qc.isChecked() else ""
        self.position_value.setText(f"Corte {st.slice + 1}/{n_slices}{gate_txt}")
        n_slice, n_gate = st.model.counts()
        here = len(st.model.gate_overrides(st.slice))
        here_txt = (f"\nEste corte: {here} gate(s) con ROI propio (ámbar); reemplaza al ROI del corte en esos gates."
                    if here else "")
        self.count_value.setText(f"Cortes con ROI: {n_slice}/{n_slices}\nAjustes por gate: {n_gate}{here_txt}")
        self.mode_value.setText("Restar fondo (definido en el visor)" if st.model.mode == "subtract"
                                else "Atenuar %")
        for button, item in zip(self.stage_buttons, self.stages):
            button.setText(item.label + (" •" if item.dirty else ""))
        self.btn_copy.setEnabled(self.btn_qc.isChecked())
        self.btn_reset.setEnabled(self.btn_qc.isChecked())
        if st.dirty:
            self.status.setText("Cambios sin aplicar en esta etapa.")

    # ---------- eventos ----------
    def _on_stage_clicked(self, index: int):
        self.current = int(index)
        self.status.setText("")
        self._load_stage()

    def _on_slice_changed(self, value: int):
        self.stage.slice = int(value)
        self._refresh()

    def _on_gate_selected(self, gate: int):
        self.stage.gate = int(gate)
        self._refresh()

    def _on_qc_toggled(self, _checked: bool):
        self.canvas.set_draw_mode(self.btn_draw.isChecked())
        self._refresh()

    def _on_polygon_finished(self, pts):
        st = self.stage
        if self.btn_qc.isChecked():
            st.model.set_gate(st.gate, st.slice, pts, all_slices=self.btn_all.isChecked())
        else:
            st.model.set_slice(st.slice, pts, all_slices=self.btn_all.isChecked())
        self._mark_dirty()

    def _on_polygon_cleared(self, *_args):
        st = self.stage
        if self.btn_qc.isChecked():
            st.model.clear_gate(st.gate, st.slice, all_slices=self.btn_all.isChecked())
        else:
            st.model.clear_slice(st.slice, all_slices=self.btn_all.isChecked())
        self._mark_dirty()

    def _on_copy_to_all(self):
        self.stage.model.copy_gate_to_all(self.stage.gate, self.stage.slice)
        self._mark_dirty()

    def _on_reset_gates(self):
        self.stage.model.reset_gates(self.stage.slice)
        self._mark_dirty()

    def _on_enabled_toggled(self, checked: bool):
        self.stage.enabled = bool(checked)
        self._mark_dirty()

    def _on_params_changed(self, _value: int):
        self.stage.params = (int(self.atten_slider.value()), int(self.feather_slider.value()))
        self._mark_dirty()

    def _on_cmap_changed(self, name: str):
        self.cmap = str(name)
        self.color_strip.set_cmap(self.cmap)
        self._refresh()

    def _on_win_mode_changed(self, _index: int):
        self.win_mode = str(self.mode_combo.currentData() or "percentil")
        self._slider_from_window()
        self._refresh()

    def _slider_from_window(self):
        if self.win_mode == "lineal":
            low, high = round(self.win["lin_low"] * 100), round(self.win["lin_high"] * 100)
        else:
            low, high = round(self.win["win_low"] * 2), round(self.win["win_high"] * 2)
        self.range_slider.blockSignals(True)
        self.range_slider.set_values(int(low), int(high))
        self.range_slider.blockSignals(False)
        self._update_window_labels(*self.range_slider.values())

    def _update_window_labels(self, low: int, high: int):
        if self.win_mode == "lineal":
            self.top_label.setText(f"{int(high)}%")
            self.base_label.setText(f"{int(low)}%")
        else:
            self.top_label.setText(f"{high / 2:.1f}%")
            self.base_label.setText(f"{low / 2:.1f}%")

    def _on_window_changed(self, low: int, high: int):
        if self.win_mode == "lineal":
            self.win["lin_low"], self.win["lin_high"] = low / 100.0, high / 100.0
        else:
            self.win["win_low"], self.win["win_high"] = low / 2.0, high / 2.0
        self._update_window_labels(low, high)
        self._refresh()

    def _reset_top(self):
        low, _high = self.range_slider.values()
        self.range_slider.set_values(low, 100 if self.win_mode == "lineal" else 200)

    def _reset_base(self):
        _low, high = self.range_slider.values()
        self.range_slider.set_values(0, high)

    def _on_apply(self):
        st = self.stage
        try:
            self._sync_engine(st)
            message = self._apply_stage(self.current, st.engine)
        except Exception as exc:
            QMessageBox.warning(self, "Atenuar intestino", f"No se pudo actualizar el montaje:\n{exc}")
            return
        st.dirty = False
        self._refresh_readout()
        self.status.setText(str(message or "Aplicado al montaje clínico."))

    def reject(self):
        pending = [st.label for st in self.stages if st.dirty]
        if pending:
            answer = QMessageBox.question(
                self, "Atenuar intestino",
                f"Hay cambios sin aplicar en {', '.join(pending)}.\n¿Cerrar y descartarlos?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        super().reject()

    def shutdown(self):
        for st in self.stages:
            st.engine._timer.stop()


def _cmap_names() -> list[str]:
    try:
        from core.col_registry import available_colormaps
        return list(available_colormaps())
    except Exception:
        return ["gray", "hot", "jet", "viridis", "french"]
