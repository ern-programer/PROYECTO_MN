"""SINCRO - ui.cine_widget.

Visor interactivo para navegar gates y slices, con edición visual básica de ROI.
"""
from __future__ import annotations

import math
from typing import Optional

import matplotlib
import numpy as np
from scipy.ndimage import (
	binary_closing,
	binary_dilation,
	binary_erosion,
	binary_fill_holes,
	binary_opening,
	center_of_mass,
	gaussian_filter,
	label,
)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from PyQt6.QtCore import QTimer, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDialog, QGridLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox

from core.col_registry import register_all_colormaps, available_colormaps


_FRENCH_CMAP = LinearSegmentedColormap.from_list(
	"french",
	[
		(0.0, "#0b3fa5"),
		(0.5, "#ffffff"),
		(1.0, "#d62828"),
	],
)


def _resolve_cmap(name: str):
	"""Compatibilidad matplotlib vieja/nueva para obtener colormaps por nombre."""
	invert = False
	base_name = str(name)
	if base_name.endswith("_r"):
		invert = True
		base_name = base_name[:-2]

	if base_name == "french":
		if invert:
			try:
				return _FRENCH_CMAP.reversed()
			except Exception:
				return LinearSegmentedColormap.from_list(
					"french_r",
					[(0.0, "#d62828"), (0.5, "#ffffff"), (1.0, "#0b3fa5")],
				)
		return _FRENCH_CMAP
	# Matplotlib moderno (>= 3.6 aprox): matplotlib.colormaps
	colormaps = getattr(matplotlib, "colormaps", None)
	if colormaps is not None:
		try:
			cmap = colormaps.get_cmap(base_name)
			return cmap.reversed() if invert else cmap
		except Exception:
			return colormaps.get_cmap("gray")
	# Fallback para versiones viejas
	try:
		from matplotlib import cm
		cmap = cm.get_cmap(base_name)
		return cmap.reversed() if invert else cmap
	except Exception:
		from matplotlib import cm
		return cm.get_cmap("gray")


def _array_to_pixmap(
	frame: np.ndarray,
	cmap_name: str = "gray",
	smooth_sigma: float = 0.0,
	invert_cmap: bool = False,
	window_low: float = 0.0,
	window_high: float = 1.0,
) -> QPixmap:
	data = np.asarray(frame, dtype=np.float64)
	if smooth_sigma and smooth_sigma > 0:
		data = gaussian_filter(data, sigma=float(smooth_sigma))
	finite = np.isfinite(data)
	if not finite.any():
		data = np.zeros_like(data, dtype=np.float64)
	else:
		valid = data[finite]
		lo = float(valid.min())
		hi = float(valid.max())
		if hi > lo:
			data = (data - lo) / (hi - lo)
		else:
			data = np.zeros_like(data, dtype=np.float64)

	w0 = max(0.0, min(1.0, float(window_low)))
	w1 = max(0.0, min(1.0, float(window_high)))
	if w1 <= w0:
		w1 = min(1.0, w0 + 0.01)
	data = np.clip((data - w0) / max(1e-8, (w1 - w0)), 0.0, 1.0)

	name = f"{cmap_name}_r" if invert_cmap else str(cmap_name)
	cmap = _resolve_cmap(name)
	rgb = np.asarray(cmap(np.clip(data, 0.0, 1.0))[..., :3], dtype=np.float32)

	rgb8 = (rgb * 255.0).astype(np.uint8)
	h, w, _ = rgb8.shape
	qimg = QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
	return QPixmap.fromImage(qimg.copy())


class RoiImageLabel(QLabel):
	roiChanged = pyqtSignal(int, object)
	zoomChanged = pyqtSignal(float)
	middleClicked = pyqtSignal()
	exclusionPolygonEdited = pyqtSignal(int, object)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setMinimumSize(360, 360)
		self.setMouseTracking(True)
		self.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.setStyleSheet("background:#111; color:#ddd; border:1px solid #444;")
		self.setCursor(Qt.CursorShape.CrossCursor)
		self._base_pixmap: Optional[QPixmap] = None
		self._frame_shape: tuple[int, int] | None = None
		self._slice_index = 0
		self._roi: tuple[float, float, float, float] | None = None
		self._exclusion_polygon: list[tuple[float, float]] = []
		self._draft_exclusion_polygon: list[tuple[float, float]] = []
		self._draw_exclusion_mode = False
		self._message = "Cargá un estudio para ver el cine"
		self._zoom = 1.0

	def set_message(self, message: str):
		self._message = message
		self.update()

	def set_slice_index(self, slice_index: int):
		self._slice_index = int(slice_index)

	def set_frame(
		self,
		frame: np.ndarray | None,
		cmap_name: str = "gray",
		smooth_sigma: float = 0.0,
		invert_cmap: bool = False,
		window_low: float = 0.0,
		window_high: float = 1.0,
	):
		if frame is None:
			self._base_pixmap = None
			self._frame_shape = None
			self.update()
			return
		self._base_pixmap = _array_to_pixmap(
			frame,
			cmap_name=cmap_name,
			smooth_sigma=smooth_sigma,
			invert_cmap=invert_cmap,
			window_low=window_low,
			window_high=window_high,
		)
		self._frame_shape = tuple(frame.shape[:2])
		self.update()

	def set_roi(self, roi: tuple[float, float, float, float] | None):
		self._roi = roi
		self.update()

	def set_exclusion_polygon(self, polygon: list[tuple[float, float]] | None):
		self._exclusion_polygon = [tuple(map(float, p)) for p in (polygon or [])]
		self._draft_exclusion_polygon = []
		self.update()

	def set_exclusion_draw_mode(self, enabled: bool):
		self._draw_exclusion_mode = bool(enabled)
		if not self._draw_exclusion_mode:
			self._draft_exclusion_polygon = []
		self.update()

	def roi(self):
		return self._roi

	def zoom(self) -> float:
		return float(self._zoom)

	def set_zoom(self, value: float):
		self._zoom = max(0.40, min(5.00, float(value)))
		self.zoomChanged.emit(self._zoom)
		self.update()

	def reset_zoom(self):
		self.set_zoom(1.0)

	def _image_rect(self) -> QRectF | None:
		if self._base_pixmap is None:
			return None
		scaled = self._base_pixmap.scaled(
			self.size(),
			Qt.AspectRatioMode.KeepAspectRatio,
			Qt.TransformationMode.SmoothTransformation,
		)
		zw = scaled.width() * self._zoom
		zh = scaled.height() * self._zoom
		x = (self.width() - zw) / 2.0
		y = (self.height() - zh) / 2.0
		return QRectF(x, y, zw, zh)

	def _roi_to_widget(self):
		rect = self._image_rect()
		if rect is None or self._roi is None or self._frame_shape is None:
			return None
		cy, cx, r_inner, r_outer = self._roi
		if not all(np.isfinite(v) for v in (cy, cx, r_inner, r_outer)):
			return None
		h, w = self._frame_shape
		scale_x = rect.width() / max(1, w)
		scale_y = rect.height() / max(1, h)
		scale = min(scale_x, scale_y)
		center = QPointF(rect.x() + cx * scale_x, rect.y() + cy * scale_y)
		return center, float(r_inner) * scale, float(r_outer) * scale

	def _widget_to_image(self, pos) -> tuple[float, float] | None:
		rect = self._image_rect()
		if rect is None or self._frame_shape is None:
			return None
		if not rect.contains(pos):
			return None
		h, w = self._frame_shape
		rel_x = (pos.x() - rect.x()) / max(1.0, rect.width())
		rel_y = (pos.y() - rect.y()) / max(1.0, rect.height())
		cx = rel_x * w
		cy = rel_y * h
		return float(cy), float(cx)

	def _polygon_to_widget(self, polygon: list[tuple[float, float]]) -> list[QPointF]:
		rect = self._image_rect()
		if rect is None or self._frame_shape is None:
			return []
		h, w = self._frame_shape
		sx = rect.width() / max(1.0, float(w))
		sy = rect.height() / max(1.0, float(h))
		pts: list[QPointF] = []
		for cy, cx in polygon:
			pts.append(QPointF(rect.x() + float(cx) * sx, rect.y() + float(cy) * sy))
		return pts

	def paintEvent(self, event):
		painter = QPainter(self)
		painter.fillRect(self.rect(), QColor("#111111"))

		if self._base_pixmap is None:
			painter.setPen(QColor("#dddddd"))
			painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
			return

		rect = self._image_rect()
		if rect is None:
			return
		painter.drawPixmap(rect.toRect(), self._base_pixmap)

		# ROI intestinal irregular (overlay de referencia para atenuación local).
		poly_draw = self._exclusion_polygon
		if self._draw_exclusion_mode and self._draft_exclusion_polygon:
			poly_draw = self._draft_exclusion_polygon
		wpts = self._polygon_to_widget(poly_draw)
		if len(wpts) >= 2:
			painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
			painter.setPen(QPen(QColor("#ff4dd2"), 2, Qt.PenStyle.DashLine))
			for i in range(1, len(wpts)):
				painter.drawLine(wpts[i - 1], wpts[i])
			if len(wpts) >= 3 and (not self._draw_exclusion_mode or len(poly_draw) == len(self._exclusion_polygon)):
				painter.drawLine(wpts[-1], wpts[0])
				painter.setPen(QPen(QColor("#ff4dd2"), 1))
				painter.setBrush(QColor(255, 77, 210, 35))
				painter.drawPolygon(QPolygonF(wpts))

		roi_data = self._roi_to_widget()
		if roi_data is not None:
			center, r_inner, r_outer = roi_data
			painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
			painter.setPen(QPen(QColor("#00d1ff"), 2))
			painter.drawEllipse(center, 4, 4)
			if r_outer > 0:
				painter.setPen(QPen(QColor("#ffcc00"), 2, Qt.PenStyle.DashLine))
				painter.drawEllipse(center, r_outer, r_outer)
			if r_inner > 0:
				painter.setPen(QPen(QColor("#ff6666"), 2, Qt.PenStyle.DotLine))
				painter.drawEllipse(center, r_inner, r_inner)

			painter.setPen(QColor("#ffffff"))
			label = f"Slice {self._slice_index + 1} | clic = centro | Shift = radio externo | Ctrl = radio interno | botón derecho = borrar"
			if self._draw_exclusion_mode:
				label += " | ROI intestino: clic agrega punto, doble clic cierra, clic derecho borra"
			painter.drawText(12, 22, label)
		else:
			painter.setPen(QColor("#ffffff"))
			painter.drawText(12, 22, f"Slice {self._slice_index + 1}")

	def mousePressEvent(self, event):
		if self._base_pixmap is None or self._frame_shape is None:
			return
		if event.button() == Qt.MouseButton.MiddleButton:
			self.middleClicked.emit()
			return
		if self._draw_exclusion_mode:
			mapped = self._widget_to_image(event.position())
			if mapped is None:
				return
			if event.button() == Qt.MouseButton.RightButton:
				self._exclusion_polygon = []
				self._draft_exclusion_polygon = []
				self.exclusionPolygonEdited.emit(self._slice_index, None)
				self.update()
				return
			if event.button() == Qt.MouseButton.LeftButton:
				self._draft_exclusion_polygon.append((float(mapped[0]), float(mapped[1])))
				self.update()
				return
		mapped = self._widget_to_image(event.position())
		if mapped is None:
			return
		cy, cx = mapped
		if event.button() == Qt.MouseButton.RightButton:
			self._roi = None
			self.roiChanged.emit(self._slice_index, None)
			self.update()
			return

		modifier = event.modifiers()
		if self._roi is None:
			self._roi = (cy, cx, 0.0, 0.0)
		else:
			_, _, r_inner, r_outer = self._roi
			if modifier & Qt.KeyboardModifier.ShiftModifier:
				center_cy, center_cx = self._roi[0], self._roi[1]
				r_outer = math.hypot(cy - center_cy, cx - center_cx)
			elif modifier & Qt.KeyboardModifier.ControlModifier:
				center_cy, center_cx = self._roi[0], self._roi[1]
				r_inner = math.hypot(cy - center_cy, cx - center_cx)
				if r_outer and r_inner > r_outer:
					r_outer = r_inner + 1.0
			else:
				self._roi = (cy, cx, r_inner, r_outer)
				self.roiChanged.emit(self._slice_index, self._roi)
				self.update()
				return
			self._roi = (self._roi[0], self._roi[1], r_inner, r_outer)

		self.roiChanged.emit(self._slice_index, self._roi)
		self.update()

	def mouseDoubleClickEvent(self, event):
		if not self._draw_exclusion_mode:
			super().mouseDoubleClickEvent(event)
			return
		if event.button() != Qt.MouseButton.LeftButton:
			return
		mapped = self._widget_to_image(event.position())
		if mapped is not None:
			self._draft_exclusion_polygon.append((float(mapped[0]), float(mapped[1])))
		if len(self._draft_exclusion_polygon) >= 3:
			self._exclusion_polygon = list(self._draft_exclusion_polygon)
			self.exclusionPolygonEdited.emit(self._slice_index, list(self._exclusion_polygon))
		self._draft_exclusion_polygon = []
		self.update()

	def wheelEvent(self, event):
		delta = event.angleDelta().y()
		if delta > 0:
			self.set_zoom(self._zoom + 0.10)
		elif delta < 0:
			self.set_zoom(self._zoom - 0.10)
		event.accept()


class ClickableLabel(QLabel):
	clicked = pyqtSignal()

	def mousePressEvent(self, event):
		if event.button() == Qt.MouseButton.LeftButton:
			self.clicked.emit()
		super().mousePressEvent(event)


class GateCurveWidget(QWidget):
	"""Curvas simples en tiempo real (intensidad por gate)."""

	def __init__(self, parent=None):
		super().__init__(parent)
		self._slice_curve: np.ndarray | None = None
		self._roi_curve: np.ndarray | None = None
		self._current_gate = 0
		self.setMinimumHeight(88)
		self.setMaximumHeight(120)

	def set_data(self, slice_curve: np.ndarray | None, roi_curve: np.ndarray | None, current_gate: int):
		self._slice_curve = None if slice_curve is None else np.asarray(slice_curve, dtype=np.float64)
		self._roi_curve = None if roi_curve is None else np.asarray(roi_curve, dtype=np.float64)
		self._current_gate = int(current_gate)
		self.update()

	def _normalize(self, y: np.ndarray | None) -> np.ndarray | None:
		if y is None or y.size < 2:
			return None
		finite = y[np.isfinite(y)]
		if finite.size < 2:
			return None
		lo = float(np.min(finite))
		hi = float(np.max(finite))
		if hi <= lo:
			return np.full_like(y, 0.5, dtype=np.float64)
		return (y - lo) / (hi - lo)

	def paintEvent(self, event):
		p = QPainter(self)
		p.fillRect(self.rect(), QColor("#0f1218"))

		r = self.rect().adjusted(10, 8, -10, -16)
		p.setPen(QPen(QColor("#2b3240"), 1))
		p.drawRect(r)

		y_slice = self._normalize(self._slice_curve)
		y_roi = self._normalize(self._roi_curve)
		n = 0
		if y_slice is not None:
			n = max(n, y_slice.size)
		if y_roi is not None:
			n = max(n, y_roi.size)
		if n < 2:
			p.setPen(QColor("#808a9a"))
			p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Curvas en vivo: cargá estudio / ROI")
			return

		def draw_curve(y: np.ndarray, color: str):
			p.setPen(QPen(QColor(color), 1.8))
			pts = []
			for i in range(y.size):
				if not np.isfinite(y[i]):
					continue
				x = r.left() + (i / max(1, y.size - 1)) * r.width()
				yp = r.bottom() - float(y[i]) * r.height()
				pts.append(QPointF(x, yp))
			for i in range(1, len(pts)):
				p.drawLine(pts[i - 1], pts[i])

		if y_slice is not None:
			draw_curve(y_slice, "#48c0ff")
		if y_roi is not None:
			draw_curve(y_roi, "#ffd54a")

		gate = max(0, min(n - 1, int(self._current_gate)))
		xg = r.left() + (gate / max(1, n - 1)) * r.width()
		p.setPen(QPen(QColor("#e74c3c"), 1.2, Qt.PenStyle.DashLine))
		p.drawLine(QPointF(xg, r.top()), QPointF(xg, r.bottom()))

		p.setPen(QColor("#d0d7e2"))
		p.drawText(12, self.height() - 2, "Azul: slice | Amarillo: ROI | Línea roja: gate actual")


class CineWidget(QWidget):
	roiEdited = pyqtSignal(int, object)
	playStateChanged = pyqtSignal(bool)
	playbackSpeedChanged = pyqtSignal(int)
	activated = pyqtSignal()

	def __init__(self, parent=None):
		super().__init__(parent)
		self._cube = None
		self._rois: dict[int, tuple[float, float, float, float]] = {}
		self._roi_source: dict[int, str] = {}
		self._current_slice = 0
		self._playing = False
		self._smooth_sigma = 0.0
		self._window_low = 0.0
		self._window_high = 1.0
		self._auto_roi_method = "robusto"
		self._intestinal_roi_polygons: dict[int, list[tuple[float, float]]] = {}
		self._intestinal_roi_polygons_by_gate: dict[tuple[int, int], list[tuple[float, float]]] = {}
		self._intestinal_attenuation_pct = 60
		self._intestinal_feather_px = 2
		self._intestinal_scope_mode = "slice"
		self._intestinal_apply_enabled = False
		self._tooltips_cache: dict[QWidget, str] = {}
		self._helpers_visible = True
		self._compact_controls = False
		self._controls_visible = True
		self._timer = QTimer(self)
		self._timer.setInterval(250)
		self._timer.timeout.connect(self._advance_gate)

		self.preview = RoiImageLabel()
		self.preview.setMinimumSize(220, 220)

		self.gate_slider = QSlider(Qt.Orientation.Horizontal)
		self.slice_slider = QSlider(Qt.Orientation.Horizontal)
		self.gate_slider.valueChanged.connect(self._update_view)
		self.slice_slider.valueChanged.connect(self._update_view)
		self.gate_prev_btn = QPushButton("<")
		self.gate_next_btn = QPushButton(">")
		self.slice_prev_btn = QPushButton("<")
		self.slice_next_btn = QPushButton(">")
		for btn in (self.gate_prev_btn, self.gate_next_btn, self.slice_prev_btn, self.slice_next_btn):
			btn.setFixedWidth(24)
			btn.setMaximumHeight(20)
			btn.setAutoRepeat(True)
			btn.setAutoRepeatDelay(260)
			btn.setAutoRepeatInterval(70)
		self.gate_prev_btn.setToolTip("Gate anterior")
		self.gate_next_btn.setToolTip("Gate siguiente")
		self.slice_prev_btn.setToolTip("Slice anterior")
		self.slice_next_btn.setToolTip("Slice siguiente")
		self.gate_prev_btn.clicked.connect(lambda: self._step_slider(self.gate_slider, -1))
		self.gate_next_btn.clicked.connect(lambda: self._step_slider(self.gate_slider, 1))
		self.slice_prev_btn.clicked.connect(lambda: self._step_slider(self.slice_slider, -1))
		self.slice_next_btn.clicked.connect(lambda: self._step_slider(self.slice_slider, 1))

		self.cmap_combo = QComboBox()
		register_all_colormaps()
		self.cmap_combo.addItems(available_colormaps())
		self.cmap_combo.setCurrentText("hot")
		self.cmap_combo.currentIndexChanged.connect(self._update_view)
		self.invert_cmap_check = QCheckBox("Invertir")
		self.invert_cmap_check.toggled.connect(self._update_view)

		self.gate_label = QLabel("Gate: -")
		self.slice_label = QLabel("Slice: -")
		self.matrix_label = QLabel("Matriz: -")
		self.help_label = QLabel(
			"Mouse: clic izq = centro | Shift+clic = radio externo | Ctrl+clic = radio interno | clic der = borrar ROI | "
			"apex/base sin cavidad: usar 'Borrar internos'"
		)
		self.help_label.setWordWrap(True)
		self.help_label.setStyleSheet("color:#666;")
		self.help_label.setMaximumHeight(52)

		self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
		self.zoom_slider.setRange(40, 500)
		self.zoom_slider.setValue(100)
		self.zoom_slider.setMaximumHeight(20)
		self.zoom_slider.setMaximumWidth(280)
		self.zoom_slider.valueChanged.connect(self._on_zoom_slider)
		self.zoom_prev_btn = QPushButton("<")
		self.zoom_next_btn = QPushButton(">")
		for btn in (self.zoom_prev_btn, self.zoom_next_btn):
			btn.setFixedWidth(24)
			btn.setMaximumHeight(20)
			btn.setAutoRepeat(True)
			btn.setAutoRepeatDelay(260)
			btn.setAutoRepeatInterval(70)
		self.zoom_prev_btn.setToolTip("Menos zoom")
		self.zoom_next_btn.setToolTip("Más zoom")
		self.zoom_prev_btn.clicked.connect(lambda: self._step_slider(self.zoom_slider, -1))
		self.zoom_next_btn.clicked.connect(lambda: self._step_slider(self.zoom_slider, 1))
		self.zoom_label = QLabel("100%")
		self.zoom_reset = QPushButton("Reset zoom")
		self.zoom_reset.clicked.connect(self._on_zoom_reset)
		self.auto_window_btn = QPushButton("Auto ventana")
		self.auto_window_btn.clicked.connect(self._auto_window)
		self.auto_window_btn.setToolTip("Ajusta Base/Top automáticamente usando percentiles del slice actual.")
		self.auto_roi_btn = QPushButton("Auto ROI")
		self.auto_roi_btn.clicked.connect(self._auto_roi_current_slice)
		self.auto_roi_btn.setToolTip("Dibuja ROI automático para el slice actual (visual).")
		self.auto_roi_all_btn = QPushButton("Auto ROI todos")
		self.auto_roi_all_btn.clicked.connect(self._auto_roi_all_slices)
		self.auto_roi_all_btn.setToolTip("Detecta ROIs automáticamente en todos los slices del volumen.")
		self.auto_roi_config_btn = QPushButton("Config ROI")
		self.auto_roi_config_btn.clicked.connect(self._open_auto_roi_config)
		self.auto_roi_help_btn = QPushButton("Help ROI")
		self.auto_roi_help_btn.clicked.connect(self._show_auto_roi_help)
		self.intestinal_roi_toggle_btn = QPushButton("ROI intestino")
		self.intestinal_roi_toggle_btn.setCheckable(True)
		self.intestinal_roi_toggle_btn.toggled.connect(self._on_intestinal_draw_toggled)
		self.intestinal_apply_btn = QPushButton("Aplicar ROI intestino")
		self.intestinal_apply_btn.setCheckable(True)
		self.intestinal_apply_btn.toggled.connect(self._on_intestinal_apply_toggled)
		self.intestinal_apply_btn.setStyleSheet("font-weight:600;")
		self.intestinal_roi_clear_btn = QPushButton("Borrar intestino")
		self.intestinal_roi_clear_btn.clicked.connect(self._clear_intestinal_roi_current_slice)
		self.intestinal_scope_combo = QComboBox()
		self.intestinal_scope_combo.addItem("Slice actual", "slice")
		self.intestinal_scope_combo.addItem("Todos los slices", "all_slices")
		self.intestinal_scope_combo.addItem("Gate actual + todos slices", "gate_slices")
		self.intestinal_scope_combo.currentIndexChanged.connect(self._on_intestinal_scope_changed)
		self.intestinal_atten_slider = QSlider(Qt.Orientation.Horizontal)
		self.intestinal_atten_slider.setRange(0, 100)
		self.intestinal_atten_slider.setValue(int(self._intestinal_attenuation_pct))
		self.intestinal_atten_slider.valueChanged.connect(self._on_intestinal_attenuation_changed)
		self.intestinal_atten_label = QLabel(f"{int(self._intestinal_attenuation_pct)}%")
		self.intestinal_feather_slider = QSlider(Qt.Orientation.Horizontal)
		self.intestinal_feather_slider.setRange(0, 12)
		self.intestinal_feather_slider.setValue(int(self._intestinal_feather_px))
		self.intestinal_feather_slider.valueChanged.connect(self._on_intestinal_feather_changed)
		self.intestinal_feather_label = QLabel(f"{int(self._intestinal_feather_px)} px")
		self.auto_roi_method_label = QLabel("Robusto")
		self.auto_roi_method_label.setStyleSheet("color:#4b5563;")
		self.auto_roi_config_btn.setToolTip("Abre la comparación visual de métodos Auto ROI y deja seleccionado el método aplicado.")
		self.auto_roi_help_btn.setToolTip("Ayuda rápida de controles y métodos Auto ROI.")
		self.auto_roi_method_label.setToolTip("Método Auto ROI activo en este visor. También se guarda en presets.")
		self.intestinal_roi_toggle_btn.setToolTip("Activa dibujo irregular del ROI intestinal: clic agrega puntos, doble clic cierra, clic derecho borra.")
		self.intestinal_apply_btn.setToolTip("Activa o desactiva la atenuación intestinal. Si está activo, se ve en tiempo real y afecta Auto ROI.")
		self.intestinal_roi_clear_btn.setToolTip("Borra el ROI intestinal según el alcance seleccionado.")
		self.intestinal_atten_slider.setToolTip("Porcentaje de reducción de cuentas dentro del ROI intestinal (solo para Auto ROI).")
		self.intestinal_feather_slider.setToolTip("Suavizado/borde blando alrededor del ROI intestinal para evitar cortes bruscos.")
		self.intestinal_scope_combo.setToolTip(
			"Elegí alcance del ROI intestinal: solo slice actual, todos los slices, o gate actual + todos los slices."
		)
		self.auto_roi_empty_only_check = QCheckBox("solo vacíos")
		self.auto_roi_empty_only_check.setChecked(True)
		self.auto_roi_empty_only_check.setToolTip("Si está activo, Auto ROI todos no sobrescribe slices que ya tienen ROI.")
		self.show_auto_roi_check = QCheckBox("Ver auto ROI")
		self.show_auto_roi_check.setChecked(True)
		self.show_auto_roi_check.setToolTip("Muestra u oculta los ROIs que fueron generados automáticamente.")
		self.show_auto_roi_check.toggled.connect(self._update_view)

		self.play_button = QPushButton("▶ Reproducir")
		self.play_button.clicked.connect(self.toggle_playback)
		self.play_button.setToolTip("Reproduce los gates en tiempo real.")
		self.speed_slider = QSlider(Qt.Orientation.Horizontal)
		self.speed_slider.setRange(50, 600)
		self.speed_slider.setValue(250)
		self.speed_slider.setMaximumHeight(20)
		self.speed_slider.setMaximumWidth(280)
		self.speed_slider.valueChanged.connect(self._on_speed_change)
		self.speed_label = QLabel("250 ms")
		self.speed_slider.setToolTip("Tiempo por frame: más bajo = más rápido.")

		self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
		self.smooth_slider.setRange(0, 30)
		self.smooth_slider.setValue(0)
		self.smooth_slider.setMaximumHeight(20)
		self.smooth_slider.setMaximumWidth(280)
		self.smooth_slider.valueChanged.connect(self._on_smooth_change)
		self.smooth_prev_btn = QPushButton("<")
		self.smooth_next_btn = QPushButton(">")
		for btn in (self.smooth_prev_btn, self.smooth_next_btn):
			btn.setFixedWidth(24)
			btn.setMaximumHeight(20)
			btn.setAutoRepeat(True)
			btn.setAutoRepeatDelay(260)
			btn.setAutoRepeatInterval(70)
		self.smooth_prev_btn.setToolTip("Menos smooth")
		self.smooth_next_btn.setToolTip("Más smooth")
		self.smooth_prev_btn.clicked.connect(lambda: self._step_slider(self.smooth_slider, -1))
		self.smooth_next_btn.clicked.connect(lambda: self._step_slider(self.smooth_slider, 1))
		self.smooth_label = QLabel("0.0")
		self.smooth_slider.setToolTip("Smooth visual de la imagen en la preview (no altera el motor).")

		self.window_low_slider = QSlider(Qt.Orientation.Horizontal)
		self.window_low_slider.setRange(0, 99)
		self.window_low_slider.setValue(0)
		self.window_low_slider.setMaximumHeight(20)
		self.window_low_slider.setMaximumWidth(220)
		self.window_low_slider.valueChanged.connect(self._on_window_low_change)
		self.window_low_label = QLabel("0%")

		self.window_high_slider = QSlider(Qt.Orientation.Horizontal)
		self.window_high_slider.setRange(1, 100)
		self.window_high_slider.setValue(100)
		self.window_high_slider.setMaximumHeight(20)
		self.window_high_slider.setMaximumWidth(220)
		self.window_high_slider.valueChanged.connect(self._on_window_high_change)
		self.window_high_label = QLabel("100%")

		controls = QGridLayout()
		controls.setHorizontalSpacing(8)
		controls.setVerticalSpacing(2)
		controls.addWidget(QLabel("Colormap"), 0, 0)
		controls.addWidget(self.cmap_combo, 0, 1)
		controls.addWidget(self.invert_cmap_check, 0, 2)
		controls.addWidget(self.play_button, 0, 3)
		controls.addWidget(self.zoom_reset, 0, 4)
		controls.addWidget(self.auto_window_btn, 0, 5)
		controls.addWidget(self.auto_roi_btn, 0, 6)
		controls.addWidget(self.auto_roi_all_btn, 0, 7)
		controls.addWidget(self.auto_roi_config_btn, 0, 8)
		controls.addWidget(self.auto_roi_help_btn, 0, 9)
		controls.addWidget(self.auto_roi_method_label, 0, 10)
		controls.addWidget(self.auto_roi_empty_only_check, 0, 11)
		controls.addWidget(self.show_auto_roi_check, 0, 12)
		controls.addWidget(self.intestinal_roi_toggle_btn, 1, 9)
		controls.addWidget(self.intestinal_apply_btn, 1, 10)
		controls.addWidget(self.intestinal_roi_clear_btn, 1, 11)
		controls.addWidget(QLabel("Atenuar int."), 1, 12)
		controls.addWidget(self.intestinal_atten_slider, 1, 13)
		controls.addWidget(QLabel("Feather"), 2, 8)
		controls.addWidget(self.intestinal_feather_slider, 2, 9)
		controls.addWidget(self.intestinal_feather_label, 2, 10)
		controls.addWidget(self.intestinal_atten_label, 2, 11)
		controls.addWidget(QLabel("Alcance int."), 2, 12)
		controls.addWidget(self.intestinal_scope_combo, 2, 13)

		controls.addWidget(self.gate_label, 1, 0)
		controls.addWidget(self.gate_prev_btn, 1, 1)
		controls.addWidget(self.gate_slider, 1, 2)
		controls.addWidget(self.gate_next_btn, 1, 3)
		controls.addWidget(self.slice_label, 1, 4)
		controls.addWidget(self.slice_prev_btn, 1, 5)
		controls.addWidget(self.slice_slider, 1, 6)
		controls.addWidget(self.slice_next_btn, 1, 7)
		controls.addWidget(self.matrix_label, 1, 8)

		controls.addWidget(QLabel("Zoom"), 2, 0)
		controls.addWidget(self.zoom_prev_btn, 2, 1)
		controls.addWidget(self.zoom_slider, 2, 2)
		controls.addWidget(self.zoom_next_btn, 2, 3)
		controls.addWidget(self.zoom_label, 2, 4)
		controls.addWidget(QLabel("Speed"), 2, 5)
		controls.addWidget(self.speed_slider, 2, 6)
		controls.addWidget(self.speed_label, 2, 7)

		controls.addWidget(QLabel("Smooth"), 3, 0)
		controls.addWidget(self.smooth_prev_btn, 3, 1)
		controls.addWidget(self.smooth_slider, 3, 2)
		controls.addWidget(self.smooth_next_btn, 3, 3)
		controls.addWidget(self.smooth_label, 3, 4)
		self.window_low_slider.setOrientation(Qt.Orientation.Vertical)
		self.window_high_slider.setOrientation(Qt.Orientation.Vertical)
		self.window_low_slider.setMaximumHeight(130)
		self.window_high_slider.setMaximumHeight(130)
		self.window_low_slider.setMaximumWidth(18)
		self.window_high_slider.setMaximumWidth(18)

		window_panel = QVBoxLayout()
		window_panel.setSpacing(2)
		window_panel.addWidget(QLabel("Top"), 0, Qt.AlignmentFlag.AlignHCenter)
		window_panel.addWidget(self.window_high_slider, 0, Qt.AlignmentFlag.AlignHCenter)
		window_panel.addWidget(self.window_high_label, 0, Qt.AlignmentFlag.AlignHCenter)
		window_panel.addSpacing(4)
		window_panel.addWidget(QLabel("Base"), 0, Qt.AlignmentFlag.AlignHCenter)
		window_panel.addWidget(self.window_low_slider, 0, Qt.AlignmentFlag.AlignHCenter)
		window_panel.addWidget(self.window_low_label, 0, Qt.AlignmentFlag.AlignHCenter)
		self.window_panel_widget = QWidget()
		self.window_panel_widget.setLayout(window_panel)

		preview_row = QHBoxLayout()
		preview_row.addWidget(self.preview, 1)
		preview_row.addWidget(self.window_panel_widget)

		self.controls_panel = QWidget()
		self.controls_panel.setLayout(controls)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(4, 4, 4, 4)
		layout.setSpacing(2)
		layout.addLayout(preview_row)
		layout.addWidget(self.help_label)
		layout.addWidget(self.controls_panel)

		self.preview.roiChanged.connect(self._on_roi_changed)
		self.preview.zoomChanged.connect(self._on_preview_zoom_changed)
		self.preview.middleClicked.connect(self.activated.emit)
		self.preview.exclusionPolygonEdited.connect(self._on_exclusion_polygon_edited)
		self.setMinimumHeight(260)
		self.setSizePolicy(self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())
		self.set_active_highlight(False)
		self._refresh_intestinal_apply_button_text()
		self._capture_tooltips()

	def set_controls_visible(self, visible: bool):
		self._controls_visible = bool(visible)
		self._refresh_ui_visibility()

	def _refresh_ui_visibility(self):
		show = bool(self._controls_visible)
		self.help_label.setVisible(show and bool(self._helpers_visible))
		self.controls_panel.setVisible(show)
		self.window_panel_widget.setVisible(show)

	def _capture_tooltips(self):
		for w in self.findChildren(QWidget):
			tip = w.toolTip()
			if tip:
				self._tooltips_cache[w] = tip

	def _apply_tooltips_enabled(self, enabled: bool):
		for w, tip in list(self._tooltips_cache.items()):
			if w is None:
				continue
			w.setToolTip(tip if enabled else "")

	def _apply_compact_controls(self):
		compact = bool(self._compact_controls)
		hide_when_compact = [
			self.auto_window_btn,
			self.auto_roi_help_btn,
			self.gate_prev_btn,
			self.gate_next_btn,
			self.slice_prev_btn,
			self.slice_next_btn,
			self.zoom_prev_btn,
			self.zoom_next_btn,
			self.smooth_prev_btn,
			self.smooth_next_btn,
			self.auto_roi_empty_only_check,
			self.show_auto_roi_check,
		]
		for w in hide_when_compact:
			w.setVisible(not compact)

	def set_ui_preferences(self, *, show_helpers: bool, enable_tooltips: bool, compact_controls: bool):
		self._helpers_visible = bool(show_helpers)
		self._compact_controls = bool(compact_controls)
		self._apply_tooltips_enabled(bool(enable_tooltips))
		self._apply_compact_controls()
		self._refresh_ui_visibility()

	def set_active_highlight(self, active: bool):
		if active:
			self.preview.setStyleSheet("background:#111; color:#ddd; border:2px solid #d61f1f;")
		else:
			self.preview.setStyleSheet("background:#111; color:#ddd; border:1px solid #444;")

	def set_manual_rois(self, rois: dict[int, tuple[float, float, float, float]] | None):
		old_sources = dict(self._roi_source)
		self._rois = dict(rois or {})
		self._roi_source = {int(sl): old_sources.get(int(sl), "manual") for sl in self._rois.keys()}
		self._update_view()

	def set_intestinal_params(self, attenuation_pct: int | float, feather_px: int | float):
		self._intestinal_attenuation_pct = max(0, min(100, int(round(float(attenuation_pct)))))
		self._intestinal_feather_px = max(0, min(16, int(round(float(feather_px)))))
		self.intestinal_atten_slider.blockSignals(True)
		self.intestinal_feather_slider.blockSignals(True)
		self.intestinal_atten_slider.setValue(int(self._intestinal_attenuation_pct))
		self.intestinal_feather_slider.setValue(int(self._intestinal_feather_px))
		self.intestinal_atten_slider.blockSignals(False)
		self.intestinal_feather_slider.blockSignals(False)
		self.intestinal_atten_label.setText(f"{int(self._intestinal_attenuation_pct)}%")
		self.intestinal_feather_label.setText(f"{int(self._intestinal_feather_px)} px")

	def intestinal_params(self) -> tuple[int, int]:
		return int(self._intestinal_attenuation_pct), int(self._intestinal_feather_px)

	def set_intestinal_apply_enabled(self, enabled: bool):
		self._intestinal_apply_enabled = bool(enabled)
		self.intestinal_apply_btn.blockSignals(True)
		self.intestinal_apply_btn.setChecked(self._intestinal_apply_enabled)
		self.intestinal_apply_btn.blockSignals(False)
		self._refresh_intestinal_apply_button_text()
		self._update_view()

	def intestinal_apply_enabled(self) -> bool:
		return bool(self._intestinal_apply_enabled)

	def set_intestinal_scope(self, scope: str):
		mode = str(scope or "").strip().lower()
		if mode not in ("slice", "all_slices", "gate_slices"):
			mode = "slice"
		self._intestinal_scope_mode = mode
		idx = self.intestinal_scope_combo.findData(mode)
		if idx >= 0 and self.intestinal_scope_combo.currentIndex() != idx:
			self.intestinal_scope_combo.blockSignals(True)
			self.intestinal_scope_combo.setCurrentIndex(idx)
			self.intestinal_scope_combo.blockSignals(False)

	def intestinal_scope(self) -> str:
		return str(self._intestinal_scope_mode)

	def intestinal_roi_state(self) -> dict[str, object]:
		"""Estado serializable del ROI intestinal dibujado."""
		slice_polygons = []
		for slice_index, polygon in sorted((self._intestinal_roi_polygons or {}).items()):
			pts = [[float(cy), float(cx)] for cy, cx in (polygon or [])]
			if len(pts) >= 3:
				slice_polygons.append({"slice": int(slice_index), "points": pts})

		gate_polygons = []
		for (gate_index, slice_index), polygon in sorted((self._intestinal_roi_polygons_by_gate or {}).items()):
			pts = [[float(cy), float(cx)] for cy, cx in (polygon or [])]
			if len(pts) >= 3:
				gate_polygons.append({"gate": int(gate_index), "slice": int(slice_index), "points": pts})

		return {
			"slice_polygons": slice_polygons,
			"gate_polygons": gate_polygons,
		}

	def set_intestinal_roi_state(self, state: dict | None):
		"""Restaura polígonos de ROI intestinal desde un preset."""
		self._intestinal_roi_polygons = {}
		self._intestinal_roi_polygons_by_gate = {}
		if isinstance(state, dict):
			for item in state.get("slice_polygons", []) or []:
				try:
					slice_index = int(item.get("slice"))
					points = [tuple(float(v) for v in pt[:2]) for pt in (item.get("points") or [])]
				except Exception:
					continue
				if len(points) >= 3:
					self._intestinal_roi_polygons[slice_index] = points
			for item in state.get("gate_polygons", []) or []:
				try:
					gate_index = int(item.get("gate"))
					slice_index = int(item.get("slice"))
					points = [tuple(float(v) for v in pt[:2]) for pt in (item.get("points") or [])]
				except Exception:
					continue
				if len(points) >= 3:
					self._intestinal_roi_polygons_by_gate[(gate_index, slice_index)] = points
		self.preview.set_exclusion_polygon(self._intestinal_polygon_for_slice(self.current_slice_index(), gate_index=self.current_gate_index()))
		self._update_view()

	def _intestinal_polygon_for_slice(self, slice_index: int, gate_index: int | None = None) -> list[tuple[float, float]]:
		sl = int(slice_index)
		if self._intestinal_scope_mode == "gate_slices":
			g = int(self.current_gate_index() if gate_index is None else gate_index)
			poly_g = self._intestinal_roi_polygons_by_gate.get((g, sl))
			if poly_g:
				return poly_g
			for (gg, ss), poly_any in self._intestinal_roi_polygons_by_gate.items():
				if int(gg) == g and poly_any:
					return poly_any
			return []
		poly = self._intestinal_roi_polygons.get(sl)
		if poly:
			return poly
		if self._intestinal_scope_mode == "all_slices" and self._intestinal_roi_polygons:
			return next(iter(self._intestinal_roi_polygons.values()))
		return []

	def roi_for_slice(self, slice_index: int):
		return self._rois.get(int(slice_index))

	def estimate_auto_roi_for_slice(self, slice_index: int):
		if self._cube is None:
			return None
		sl = int(slice_index)
		if sl < 0 or sl >= int(self._cube.shape[1]):
			return None
		img = np.asarray(self._cube[:, sl].mean(axis=0), dtype=np.float64)
		img = self._attenuate_image_with_intestinal_roi(img, sl)
		return self._auto_roi_from_image(img)

	def _polygon_to_mask(self, shape: tuple[int, int], polygon: list[tuple[float, float]] | None) -> np.ndarray:
		if not polygon or len(polygon) < 3:
			return np.zeros(shape, dtype=bool)
		h, w = int(shape[0]), int(shape[1])
		verts = np.asarray([(float(cx), float(cy)) for cy, cx in polygon], dtype=np.float64)
		path = MplPath(verts)
		xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
		pts = np.column_stack((xs.ravel(), ys.ravel()))
		inside = path.contains_points(pts)
		return inside.reshape(h, w)

	def _soft_mask_from_polygon(self, shape: tuple[int, int], polygon: list[tuple[float, float]] | None) -> np.ndarray:
		base = self._polygon_to_mask(shape, polygon)
		if not np.any(base):
			return np.zeros(shape, dtype=np.float64)
		feather = max(0, int(self._intestinal_feather_px))
		if feather > 0:
			dil = binary_dilation(base, iterations=max(1, feather // 2 + 1))
			soft = gaussian_filter(dil.astype(np.float64), sigma=max(0.8, float(feather) * 0.6))
			soft = soft / max(1e-8, float(np.max(soft)))
			return np.clip(soft, 0.0, 1.0)
		return base.astype(np.float64)

	def _attenuate_image_with_intestinal_roi(self, img: np.ndarray, slice_index: int, gate_index: int | None = None) -> np.ndarray:
		img = np.asarray(img, dtype=np.float64)
		if not self._intestinal_apply_enabled:
			return img
		g = self.current_gate_index() if gate_index is None else int(gate_index)
		poly = self._intestinal_polygon_for_slice(int(slice_index), gate_index=g)
		if not poly:
			return img
		atten = max(0.0, min(1.0, float(self._intestinal_attenuation_pct) / 100.0))
		if atten <= 1e-6:
			return img
		mask = self._soft_mask_from_polygon(img.shape, poly)
		if not np.isfinite(mask).any() or float(np.max(mask)) <= 0.0:
			return img
		factor = 1.0 - atten * np.clip(mask, 0.0, 1.0)
		return np.asarray(img * factor, dtype=np.float64)

	def apply_intestinal_mask_to_gate_volume(self, gate_volume: np.ndarray, gate_index: int | None = None) -> np.ndarray:
		"""Aplica atenuación intestinal slice a slice sobre un volumen de gate.

		Se usa para mejorar vistas comparativas (SA/HLA/VLA) sin alterar el cubo base.
		"""
		vol = np.asarray(gate_volume, dtype=np.float64)
		if vol.ndim != 3:
			return vol
		if not self._intestinal_apply_enabled:
			return vol
		g = self.current_gate_index() if gate_index is None else int(gate_index)
		out = np.array(vol, dtype=np.float64, copy=True)
		for s in range(int(out.shape[0])):
			out[s] = self._attenuate_image_with_intestinal_roi(out[s], s, gate_index=g)
		return out

	def build_adjusted_auto_rois(
		self,
		reference_slice: int,
		reference_roi: tuple[float, float, float, float],
		*,
		adjust_center: bool,
		adjust_inner: bool,
		adjust_outer: bool,
		center_gain: float = 1.0,
		inner_extra: float = 0.0,
		outer_extra: float = 0.0,
		max_distance: int = -1,
	) -> dict[int, tuple[float, float, float, float]]:
		if self._cube is None:
			return {}
		if reference_roi is None or not all(np.isfinite(v) for v in reference_roi):
			return {}

		ref_auto = self.estimate_auto_roi_for_slice(reference_slice)
		if ref_auto is None:
			return {}

		ref_cy, ref_cx, ref_ri, ref_ro = (float(v) for v in reference_roi)
		auto_cy, auto_cx, auto_ri, auto_ro = (float(v) for v in ref_auto)
		delta_cy = (ref_cy - auto_cy) * float(center_gain)
		delta_cx = (ref_cx - auto_cx) * float(center_gain)
		delta_ri = 0.0 if not np.isfinite(ref_ri) else (ref_ri - auto_ri)
		delta_ro = ref_ro - auto_ro

		adjusted: dict[int, tuple[float, float, float, float]] = {}
		for sl in range(int(self._cube.shape[1])):
			if int(max_distance) >= 0 and abs(int(sl) - int(reference_slice)) > int(max_distance):
				continue
			roi_auto = self.estimate_auto_roi_for_slice(sl)
			if roi_auto is None:
				continue
			cy, cx, r_inner, r_outer = (float(v) for v in roi_auto)
			if adjust_center:
				cy += delta_cy
				cx += delta_cx
			if adjust_inner:
				r_inner += delta_ri + float(inner_extra)
			if adjust_outer:
				r_outer += delta_ro + float(outer_extra)

			r_inner = max(0.0, float(r_inner))
			r_outer = max(r_inner + 1.0, float(r_outer))
			adjusted[int(sl)] = (float(cy), float(cx), float(r_inner), float(r_outer))
		return adjusted

	def current_slice_index(self) -> int:
		return int(self._current_slice)

	def current_gate_index(self) -> int:
		return int(self.gate_slider.value())

	def set_cube(self, cube: np.ndarray | None):
		self._cube = cube
		if cube is None:
			self._intestinal_roi_polygons = {}
			self._intestinal_roi_polygons_by_gate = {}
			self.preview.set_exclusion_polygon([])
			self.preview.set_message("Cargá un estudio para ver el cine")
			self.gate_slider.setRange(0, 0)
			self.slice_slider.setRange(0, 0)
			self.gate_label.setText("Gate: -")
			self.slice_label.setText("Slice: -")
			self.matrix_label.setText("Matriz: -")
			self.preview.set_frame(None)
			return

		n_gates, n_slices = cube.shape[:2]
		self.gate_slider.blockSignals(True)
		self.slice_slider.blockSignals(True)
		self.gate_slider.setRange(0, max(0, n_gates - 1))
		self.slice_slider.setRange(0, max(0, n_slices - 1))
		self.gate_slider.setValue(n_gates // 2)
		self.slice_slider.setValue(n_slices // 2)
		self.gate_slider.blockSignals(False)
		self.slice_slider.blockSignals(False)
		self._update_view()

	def set_smooth_sigma(self, value: float):
		self._smooth_sigma = max(0.0, float(value))
		self.smooth_slider.blockSignals(True)
		self.smooth_slider.setValue(int(round(self._smooth_sigma * 10.0)))
		self.smooth_slider.blockSignals(False)
		self.smooth_label.setText(f"{self._smooth_sigma:.1f}")
		self._update_view()

	def toggle_playback(self):
		self._playing = not self._playing
		if self._playing:
			self._timer.start()
			self.play_button.setText("⏸ Pausar")
		else:
			self._timer.stop()
			self.play_button.setText("▶ Reproducir")
		self.playStateChanged.emit(self._playing)

	def stop_playback(self):
		if self._playing:
			self.toggle_playback()

	def _advance_gate(self):
		if self._cube is None:
			return
		n_gates = self._cube.shape[0]
		self.gate_slider.setValue((self.gate_slider.value() + 1) % n_gates)

	def _step_slider(self, slider: QSlider, delta: int):
		value = int(slider.value()) + int(delta)
		value = max(int(slider.minimum()), min(int(slider.maximum()), value))
		slider.setValue(value)

	def _update_view(self, *args):
		if self._cube is None:
			return
		gate = int(self.gate_slider.value())
		sl = int(self.slice_slider.value())
		self._current_slice = sl
		frame = np.asarray(self._cube[gate, sl], dtype=np.float64)
		if self._intestinal_apply_enabled:
			frame = self._attenuate_image_with_intestinal_roi(frame, sl)
		self.preview.set_slice_index(sl)
		self.preview.set_frame(
			frame,
			cmap_name=str(self.cmap_combo.currentText()),
			smooth_sigma=self._smooth_sigma,
			invert_cmap=self.invert_cmap_check.isChecked(),
			window_low=self._window_low,
			window_high=self._window_high,
		)
		roi = self._rois.get(sl)
		if roi is not None and not self.show_auto_roi_check.isChecked() and self._roi_source.get(sl) == "auto":
			roi = None
		self.preview.set_roi(roi)
		self.preview.set_exclusion_polygon(self._intestinal_polygon_for_slice(sl, gate_index=gate))
		self.gate_label.setText(f"Gate: {gate + 1}/{self._cube.shape[0]}")
		self.slice_label.setText(f"Slice: {sl + 1}/{self._cube.shape[1]}")
		self.matrix_label.setText(f"Matriz: {self._cube.shape[2]}x{self._cube.shape[3]}")
		self.smooth_label.setText(f"{self._smooth_sigma:.1f}")
		self.window_low_label.setText(f"{int(round(self._window_low * 100))}%")
		self.window_high_label.setText(f"{int(round(self._window_high * 100))}%")

		# La curva temporal bajo la imagen se retiró del flujo visual por pedido de uso clínico.

	def _on_speed_change(self, value: int):
		self._timer.setInterval(int(value))
		self.speed_label.setText(f"{int(value)} ms")
		self.playbackSpeedChanged.emit(int(value))

	def _on_smooth_change(self, value: int):
		self._smooth_sigma = float(value) / 10.0
		self.smooth_label.setText(f"{self._smooth_sigma:.1f}")
		self._update_view()

	def _on_window_low_change(self, value: int):
		if value >= self.window_high_slider.value():
			self.window_high_slider.blockSignals(True)
			self.window_high_slider.setValue(min(100, value + 1))
			self.window_high_slider.blockSignals(False)
		self._window_low = float(value) / 100.0
		self._window_high = float(self.window_high_slider.value()) / 100.0
		self._update_view()

	def _on_window_high_change(self, value: int):
		if value <= self.window_low_slider.value():
			self.window_low_slider.blockSignals(True)
			self.window_low_slider.setValue(max(0, value - 1))
			self.window_low_slider.blockSignals(False)
		self._window_low = float(self.window_low_slider.value()) / 100.0
		self._window_high = float(value) / 100.0
		self._update_view()

	def _auto_window(self):
		if self._cube is None:
			return
		gate = int(self.gate_slider.value())
		sl = int(self.slice_slider.value())
		frame = np.asarray(self._cube[gate, sl], dtype=np.float64)
		finite = frame[np.isfinite(frame)]
		if finite.size < 8:
			return

		lo = float(np.percentile(finite, 12))
		hi = float(np.percentile(finite, 98))
		fmin = float(np.min(finite))
		fmax = float(np.max(finite))
		if fmax <= fmin:
			return

		base = int(round(100.0 * (lo - fmin) / (fmax - fmin + 1e-8)))
		top = int(round(100.0 * (hi - fmin) / (fmax - fmin + 1e-8)))
		base = max(0, min(98, base))
		top = max(base + 1, min(100, top))

		self.window_low_slider.blockSignals(True)
		self.window_high_slider.blockSignals(True)
		self.window_low_slider.setValue(base)
		self.window_high_slider.setValue(top)
		self.window_low_slider.blockSignals(False)
		self.window_high_slider.blockSignals(False)
		self._window_low = float(base) / 100.0
		self._window_high = float(top) / 100.0
		self._update_view()

	def _auto_roi_current_slice(self):
		if self._cube is None:
			return
		sl = int(self.slice_slider.value())
		roi = self.estimate_auto_roi_for_slice(sl)
		if roi is None:
			return
		self._rois[sl] = roi
		self._roi_source[sl] = "auto"
		self.roiEdited.emit(sl, roi)
		self._update_view()

	def _auto_roi_all_slices(self):
		if self._cube is None:
			return
		empty_only = self.auto_roi_empty_only_check.isChecked()
		n_slices = int(self._cube.shape[1])
		for sl in range(n_slices):
			existing_roi = self._rois.get(sl)
			source = self._roi_source.get(sl)
			# "solo vacíos" protege ROIs manuales, pero permite regenerar los automáticos
			# para que un segundo click sobre "Auto ROI todos" siga funcionando.
			if empty_only and existing_roi is not None and source != "auto":
				continue
			roi = self.estimate_auto_roi_for_slice(sl)
			if roi is None:
				continue
			self._rois[sl] = roi
			self._roi_source[sl] = "auto"
			self.roiEdited.emit(sl, roi)
		self._update_view()

	def _open_auto_roi_config(self):
		# Config ROI ahora abre la comparativa visual para elegir método+ROI en un paso.
		self._compare_auto_roi_methods_current_slice()

	def _show_auto_roi_help(self):
		msg = (
			"Auto ROI - guía rápida\n\n"
			"Controles:\n"
			"• Auto ROI: aplica en slice actual.\n"
			"• Auto ROI todos: recorre todo el volumen.\n"
			"• solo vacíos: no pisa ROIs manuales existentes.\n"
			"• Config ROI: abre la comparativa visual y aplica método/ROI en un clic.\n"
			"  Tip: podés hacer clic directamente sobre la imagen del método para seleccionarlo.\n\n"
			"Métodos:\n"
			"1) Robusto central: prior espacial del VI + umbral robusto (recomendado).\n"
			"2) Clásico: umbral + componente mayor.\n"
			"3) Gradiente: bordes por gradiente + morfología.\n"
			"4) Hot bowel: variante robusta con penalización inferior para focos intestinales intensos.\n"
			"5) Percentil central: umbral adaptativo por percentiles + prior central (útil en matrices bajas).\n"
			"6) Consenso: combina varios métodos y sugiere el más estable.\n\n"
			"7) Inferior superpuesto: suprime focos calientes periféricos inferiores (hígado/intestino) y luego detecta VI.\n\n"
			"ROI intestino irregular:\n"
			"• Activá 'ROI intestino' y dibujá polígono (doble clic para cerrar).\n"
			"• Ajustá Atenuar % y Feather para bajar cuentas con borde suave.\n\n"
			"Tip clínico: en 22x22, usar primero Robusto u Hot bowel y validar con Comparar ROI."
		)
		QMessageBox.information(self, "SINCRO - Help Auto ROI", msg)

	def set_auto_roi_method(self, method: str):
		key = str(method or "").strip().lower()
		if key not in ("robusto", "clasico", "gradiente", "hotbowel", "percentil_central", "consenso", "inferior_overlap"):
			key = "robusto"
		self._auto_roi_method = key
		if key == "clasico":
			self.auto_roi_method_label.setText("Clásico")
		elif key == "gradiente":
			self.auto_roi_method_label.setText("Gradiente")
		elif key == "hotbowel":
			self.auto_roi_method_label.setText("Hot bowel")
		elif key == "percentil_central":
			self.auto_roi_method_label.setText("Percentil central")
		elif key == "consenso":
			self.auto_roi_method_label.setText("Consenso")
		elif key == "inferior_overlap":
			self.auto_roi_method_label.setText("Inferior superpuesto")
		else:
			self.auto_roi_method_label.setText("Robusto")

	def auto_roi_method(self) -> str:
		return str(self._auto_roi_method)

	def _method_label(self, method: str) -> str:
		m = str(method).lower()
		if m == "clasico":
			return "Clásico"
		if m == "gradiente":
			return "Gradiente"
		if m == "hotbowel":
			return "Hot bowel"
		if m == "percentil_central":
			return "Percentil central"
		if m == "consenso":
			return "Consenso"
		if m == "inferior_overlap":
			return "Inferior superpuesto"
		return "Robusto"

	def _build_inferior_hot_suppression_map(self, img: np.ndarray, low_res: bool) -> np.ndarray:
		img = np.asarray(img, dtype=np.float64)
		h, w = img.shape
		finite = img[np.isfinite(img)]
		if finite.size < 12:
			return np.ones_like(img, dtype=np.float64)
		p_hot = float(np.percentile(finite, 88.0 if low_res else 91.5))
		ys, xs = np.ogrid[:h, :w]
		cy0 = 0.5 * (h - 1)
		cx0 = 0.5 * (w - 1)
		rr = np.sqrt((ys - cy0) ** 2 + (xs - cx0) ** 2)
		rmin = float(min(h, w))
		inferior = ys > (cy0 + 0.08 * h)
		peripheral = (rr >= 0.45 * rmin) & (rr <= 0.98 * rmin)
		hot = img >= p_hot
		seeds = hot & inferior & peripheral
		if int(np.count_nonzero(seeds)) < 4:
			return np.ones_like(img, dtype=np.float64)
		spread = gaussian_filter(seeds.astype(np.float64), sigma=1.2 if low_res else 1.8)
		mx = float(np.max(spread))
		if mx <= 1e-8:
			return np.ones_like(img, dtype=np.float64)
		spread /= mx
		strength = 0.55 if low_res else 0.48
		penalty = 1.0 - strength * np.clip(spread, 0.0, 1.0)
		# No penalizar el anillo central donde normalmente vive el VI.
		penalty = np.where(rr <= 0.34 * rmin, 1.0, penalty)
		return np.asarray(np.clip(penalty, 0.25, 1.0), dtype=np.float64)

	def _auto_roi_from_image_inferior_overlap(self, img: np.ndarray, low_res: bool):
		img = np.asarray(img, dtype=np.float64)
		supp = self._build_inferior_hot_suppression_map(img, low_res)
		img_w = img * supp
		finite = img_w[np.isfinite(img_w)]
		if finite.size < 8:
			return self._auto_roi_from_image_hotbowel(img, low_res)
		p99 = float(np.percentile(finite, 99.0))
		thr_floor = float(np.percentile(finite, 70.0 if low_res else 67.0))
		thr = max(0.54 * p99, thr_floor)
		bin_mask = img_w > thr
		mask = self._component_with_center_prior(bin_mask, penalize_inferior=True)
		if mask is None:
			return self._auto_roi_from_image_hotbowel(img, low_res)
		roi = self._roi_from_binary_mask(mask, low_res)
		if roi is not None:
			return roi
		return self._auto_roi_from_image_hotbowel(img, low_res)

	def _auto_roi_from_image_percentil_central(self, img: np.ndarray, low_res: bool):
		finite = img[np.isfinite(img)]
		if finite.size < 8:
			return None
		p88 = float(np.percentile(finite, 88.0 if low_res else 84.0))
		p70 = float(np.percentile(finite, 70.0 if low_res else 66.0))
		thr = max(p70, 0.72 * p88)
		bin_mask = img > thr
		mask = self._component_with_center_prior(bin_mask, penalize_inferior=True)
		if mask is None:
			return self._auto_roi_from_image_robusto(img, low_res)
		return self._roi_from_binary_mask(mask, low_res)

	def _auto_roi_from_image_consenso(self, img: np.ndarray, low_res: bool):
		candidates: list[tuple[str, tuple[float, float, float, float]]] = []
		for method in ("robusto", "hotbowel", "percentil_central", "gradiente"):
			roi = self._auto_roi_from_image_with_method(img, low_res=low_res, method=method)
			if roi is not None:
				candidates.append((method, roi))
		if not candidates:
			return None
		if len(candidates) == 1:
			return candidates[0][1]
		ys = np.asarray([float(r[0]) for _, r in candidates], dtype=np.float64)
		xs = np.asarray([float(r[1]) for _, r in candidates], dtype=np.float64)
		ym = float(np.median(ys))
		xm = float(np.median(xs))
		filtered: list[tuple[float, float, float, float]] = []
		for _m, roi in candidates:
			cy, cx, ri, ro = (float(v) for v in roi)
			if math.hypot(cy - ym, cx - xm) <= (3.2 if low_res else 4.5):
				filtered.append((cy, cx, ri, ro))
		if not filtered:
			filtered = [tuple(float(v) for v in r) for _, r in candidates]
		arr = np.asarray(filtered, dtype=np.float64)
		cy = float(np.median(arr[:, 0]))
		cx = float(np.median(arr[:, 1]))
		ri = float(np.median(arr[:, 2]))
		ro = float(np.median(arr[:, 3]))
		if ro <= ri:
			ro = ri + 1.0
		return (cy, cx, max(0.0, ri), ro)

	def _component_with_center_prior(self, bin_mask: np.ndarray, *, penalize_inferior: bool = False):
		bin_mask = np.asarray(bin_mask, dtype=bool)
		lbl, n = label(bin_mask)
		if n <= 0:
			return None
		h, w = bin_mask.shape
		cy0 = (h - 1) * 0.5
		cx0 = (w - 1) * 0.5
		ys_grid, xs_grid = np.ogrid[:h, :w]
		rr = np.sqrt((ys_grid - cy0) ** 2 + (xs_grid - cx0) ** 2)
		prior = (rr >= 0.10 * min(h, w)) & (rr <= 0.50 * min(h, w))

		best_score = -1e9
		best = None
		for comp_id in range(1, n + 1):
			comp = lbl == comp_id
			area = int(np.count_nonzero(comp))
			if area < 5:
				continue
			cy_c, cx_c = center_of_mass(comp)
			if not (np.isfinite(cy_c) and np.isfinite(cx_c)):
				continue
			dist = float(np.sqrt((cy_c - cy0) ** 2 + (cx_c - cx0) ** 2))
			dist_norm = dist / max(1e-6, 0.5 * min(h, w))
			overlap = float(np.count_nonzero(comp & prior)) / float(area)
			filled = binary_fill_holes(comp)
			cavity = filled & (~comp)
			hole_frac = float(np.count_nonzero(cavity)) / max(1.0, float(np.count_nonzero(filled)))
			score = 2.6 * overlap + 1.8 * max(0.0, 1.0 - dist_norm) + 0.6 * min(1.0, hole_frac / 0.18)
			if dist_norm > 0.95:
				score -= 2.5
			if penalize_inferior and float(cy_c) > 0.62 * float(h):
				score -= 1.25
			if score > best_score:
				best_score = score
				best = comp
		return best

	def _roi_from_binary_mask(self, mask: np.ndarray, low_res: bool):
		mask = np.asarray(mask, dtype=bool)
		if int(mask.sum()) < 8:
			return None
		cy, cx = center_of_mass(mask)
		ys, xs = np.nonzero(mask)
		d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
		if d.size < 4:
			return None
		h, w = mask.shape
		r_inner = float(np.percentile(d, 24 if low_res else 22))
		r_outer = float(np.percentile(d, 76 if low_res else 82))
		max_outer = (0.42 if low_res else 0.48) * float(min(h, w))
		r_outer = min(float(r_outer), float(max_outer))
		if r_outer <= r_inner:
			r_outer = r_inner + 1.0
		r_inner = min(float(r_inner), 0.84 * float(r_outer))
		return (float(cy), float(cx), r_inner, r_outer)

	def _auto_roi_from_image_clasico(self, img: np.ndarray, low_res: bool):
		thr = float(np.percentile(img[np.isfinite(img)], 70.0))
		bin_mask = img > thr
		lbl, n = label(bin_mask)
		if n <= 0:
			return None
		counts = np.bincount(lbl.ravel())
		counts[0] = 0
		largest = int(np.argmax(counts))
		mask = lbl == largest
		return self._roi_from_binary_mask(mask, low_res)

	def _auto_roi_from_image_gradiente(self, img: np.ndarray, low_res: bool):
		gy, gx = np.gradient(img)
		grad = np.hypot(gx, gy)
		finite = grad[np.isfinite(grad)]
		if finite.size < 8:
			return None
		thr = float(np.percentile(finite, 74.0 if low_res else 80.0))
		edges = grad > thr
		k = 2 if low_res else 3
		st = np.ones((k, k), dtype=bool)
		edges = binary_opening(edges, structure=st)
		edges = binary_closing(edges, structure=st)
		filled = binary_fill_holes(edges)
		mask = self._component_with_center_prior(filled)
		if mask is None:
			return None
		boundary = mask & (~binary_erosion(mask, structure=st))
		if int(np.count_nonzero(boundary)) >= 8:
			cy, cx = center_of_mass(mask)
			ys, xs = np.nonzero(boundary)
			d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
			r_outer = float(np.percentile(d, 68 if low_res else 74))
			max_outer = (0.42 if low_res else 0.48) * float(min(mask.shape))
			r_outer = min(float(r_outer), float(max_outer))
			r_inner = max(0.0, 0.46 * float(r_outer))
			return (float(cy), float(cx), float(r_inner), float(r_outer))
		return self._roi_from_binary_mask(mask, low_res)

	def _auto_roi_from_image_robusto(self, img: np.ndarray, low_res: bool):
		finite = img[np.isfinite(img)]
		p99 = float(np.percentile(finite, 99.0)) if finite.size else 0.0
		thr_floor = float(np.percentile(finite, 72.0 if low_res else 70.0))
		thr = max(0.52 * p99, thr_floor)
		bin_mask = img > thr
		mask = self._component_with_center_prior(bin_mask)
		if mask is None:
			lbl, n = label(bin_mask)
			if n <= 0:
				return None
			counts = np.bincount(lbl.ravel())
			counts[0] = 0
			largest = int(np.argmax(counts))
			mask = lbl == largest
		return self._roi_from_binary_mask(mask, low_res)

	def _auto_roi_from_image_hotbowel(self, img: np.ndarray, low_res: bool):
		finite = img[np.isfinite(img)]
		p99 = float(np.percentile(finite, 99.0)) if finite.size else 0.0
		thr_floor = float(np.percentile(finite, 73.0 if low_res else 71.0))
		thr = max(0.55 * p99, thr_floor)
		bin_mask = img > thr
		mask = self._component_with_center_prior(bin_mask, penalize_inferior=True)
		if mask is None:
			lbl, n = label(bin_mask)
			if n <= 0:
				return None
			counts = np.bincount(lbl.ravel())
			counts[0] = 0
			largest = int(np.argmax(counts))
			mask = lbl == largest
		return self._roi_from_binary_mask(mask, low_res)

	def _auto_roi_from_image(self, img: np.ndarray):
		img = np.asarray(img, dtype=np.float64)
		if img.ndim != 2:
			return None
		h, w = img.shape
		low_res = min(h, w) <= 28
		img = gaussian_filter(img, sigma=1.2 if low_res else 1.0)
		if not np.isfinite(img).any() or float(np.max(img)) <= 0.0:
			return None
		return self._auto_roi_from_image_with_method(img, low_res=low_res, method=self._auto_roi_method)

	def _auto_roi_from_image_with_method(self, img: np.ndarray, *, low_res: bool, method: str):
		method_key = str(method or "").strip().lower()
		if method_key == "clasico":
			return self._auto_roi_from_image_clasico(img, low_res)
		if method_key == "gradiente":
			return self._auto_roi_from_image_gradiente(img, low_res)
		if method_key == "hotbowel":
			return self._auto_roi_from_image_hotbowel(img, low_res)
		if method_key == "percentil_central":
			return self._auto_roi_from_image_percentil_central(img, low_res)
		if method_key == "consenso":
			return self._auto_roi_from_image_consenso(img, low_res)
		if method_key == "inferior_overlap":
			return self._auto_roi_from_image_inferior_overlap(img, low_res)
		return self._auto_roi_from_image_robusto(img, low_res)

	def _score_auto_roi_candidate(self, img: np.ndarray, roi: tuple[float, float, float, float], slice_index: int) -> float:
		cy, cx, ri, ro = (float(v) for v in roi)
		h, w = img.shape
		ys, xs = np.ogrid[:h, :w]
		d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
		ring = (d >= max(0.0, ri - 0.75)) & (d <= max(ri + 1.0, ro))
		inside = d <= ro
		outside = (d > ro) & (d <= ro + max(2.0, 0.35 * ro))
		if int(np.count_nonzero(ring)) < 8 or int(np.count_nonzero(outside)) < 8:
			return -1e6
		ring_mean = float(np.mean(img[ring]))
		outside_mean = float(np.mean(img[outside]))
		contrast = ring_mean - outside_mean
		inferior_out = (d > ro) & (d <= ro + max(2.0, 0.42 * ro)) & (ys > cy + 0.12 * ro)
		if int(np.count_nonzero(inferior_out)) >= 8:
			inferior_hot_pen = max(0.0, float(np.mean(img[inferior_out])) - ring_mean)
		else:
			inferior_hot_pen = 0.0
		center_dist = float(math.hypot(cy - 0.5 * (h - 1), cx - 0.5 * (w - 1)))
		center_pen = center_dist / max(1e-6, 0.55 * min(h, w))
		area_pen = abs(float(np.count_nonzero(inside)) / max(1.0, float(h * w)) - 0.24)
		score = 2.2 * contrast - 0.95 * center_pen - 0.55 * area_pen - 1.25 * inferior_hot_pen
		poly = self._intestinal_roi_polygons.get(int(slice_index))
		if poly:
			mask_int = self._polygon_to_mask(img.shape, poly)
			if np.any(mask_int):
				overlap = float(np.count_nonzero(inside & mask_int)) / max(1.0, float(np.count_nonzero(inside)))
				score -= 0.8 * overlap
		return float(score)

	def _compare_auto_roi_methods_current_slice(self):
		if self._cube is None:
			return
		sl = int(self.slice_slider.value())
		img = np.asarray(self._cube[:, sl].mean(axis=0), dtype=np.float64)
		img = self._attenuate_image_with_intestinal_roi(img, sl)
		if img.ndim != 2:
			return
		low_res = min(img.shape) <= 28
		img_s = gaussian_filter(img, sigma=1.2 if low_res else 1.0)
		if not np.isfinite(img_s).any() or float(np.max(img_s)) <= 0.0:
			return

		method_order = ["robusto", "clasico", "gradiente", "hotbowel", "percentil_central", "consenso", "inferior_overlap"]
		options = []
		by_label = {}
		best_label = ""
		best_score = -1e9
		for method in method_order:
			roi = self._auto_roi_from_image_with_method(img_s, low_res=low_res, method=method)
			if roi is None:
				continue
			cy, cx, ri, ro = (float(v) for v in roi)
			score = self._score_auto_roi_candidate(img_s, roi, sl)
			label = f"{self._method_label(method)} | cy={cy:.1f} cx={cx:.1f} ri={ri:.1f} ro={ro:.1f} | score={score:.2f}"
			if score > best_score:
				best_score = score
				best_label = label
			options.append(label)
			by_label[label] = (method, roi, score)

		if not options:
			return

		dialog = QDialog(self)
		dialog.setWindowTitle(f"Comparar Auto ROI - Slice {sl + 1}")
		dialog.setModal(True)
		dialog.resize(1220, 760)
		root = QVBoxLayout(dialog)
		root.addWidget(QLabel("Vista previa de métodos Auto ROI. Elegí uno para aplicar en este slice."))

		grid = QGridLayout()
		grid.setHorizontalSpacing(8)
		grid.setVerticalSpacing(8)
		selected: dict[str, tuple[str, tuple[float, float, float, float]] | None] = {"value": None}

		for idx, option in enumerate(options):
			entry = by_label.get(option)
			if entry is None:
				continue
			method, roi, score = entry
			card = QWidget()
			card_layout = QVBoxLayout(card)
			card_layout.setContentsMargins(6, 6, 6, 6)
			card_layout.setSpacing(4)
			card.setStyleSheet("background:#0f172a; border:1px solid #334155; border-radius:6px;")

			title_text = self._method_label(method)
			if option == best_label:
				title_text += "  |  SUGERIDO"
			title = QLabel(title_text)
			title.setStyleSheet("color:#e2e8f0; font-weight:600;")
			title.setToolTip(
				"Robusto: prior central/anular.\n"
				"Clásico: umbral + componente mayor.\n"
				"Gradiente: bordes por gradiente.\n"
				"Hot bowel: robusto + penalización inferior.\n"
				"Percentil central: prior central + percentiles adaptativos.\n"
				"Consenso: mediana entre métodos robustos.\n"
				"Inferior superpuesto: reduce impacto de focos inferiores extracardíacos."
			)
			card_layout.addWidget(title)

			img_label = ClickableLabel()
			img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
			img_label.setCursor(Qt.CursorShape.PointingHandCursor)
			img_label.setPixmap(self._build_roi_preview_pixmap(img_s, roi, size=220))
			img_label.setToolTip(f"Clic para aplicar método {self._method_label(method)}")
			def _on_pick(m=method, r=roi):
				selected["value"] = (m, r)
				dialog.accept()
			img_label.clicked.connect(_on_pick)
			card_layout.addWidget(img_label)

			cy, cx, ri, ro = (float(v) for v in roi)
			metrics = QLabel(f"cy={cy:.1f}  cx={cx:.1f}\nri={ri:.1f}  ro={ro:.1f}\nscore={score:.2f}")
			metrics.setStyleSheet("color:#cbd5e1;")
			metrics.setToolTip("Centro (cy/cx) y radios interno/externo calculados para este método.")
			card_layout.addWidget(metrics)

			apply_btn = QPushButton("Aplicar")
			apply_btn.setToolTip(f"Aplicar método {self._method_label(method)} en este slice y dejarlo activo.")
			def _on_apply(_checked=False, m=method, r=roi):
				selected["value"] = (m, r)
				dialog.accept()
			apply_btn.clicked.connect(_on_apply)
			card_layout.addWidget(apply_btn)

			row = idx // 4
			col = idx % 4
			grid.addWidget(card, row, col)

		root.addLayout(grid)
		if best_label:
			best_entry = by_label.get(best_label)
			if best_entry is not None:
				best_method, best_roi, _best_score = best_entry
				apply_best_btn = QPushButton(f"Aplicar sugerido ({self._method_label(best_method)})")
				apply_best_btn.setToolTip("Aplica directamente el método sugerido por score en este slice.")
				def _on_apply_best(_checked=False, m=best_method, r=best_roi):
					selected["value"] = (m, r)
					dialog.accept()
				apply_best_btn.clicked.connect(_on_apply_best)
				root.addWidget(apply_best_btn)
		cancel_btn = QPushButton("Cancelar")
		cancel_btn.clicked.connect(dialog.reject)
		root.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)

		if dialog.exec() != int(QDialog.DialogCode.Accepted):
			return

		picked = selected.get("value")
		if picked is None:
			return
		method, roi = picked
		self.set_auto_roi_method(method)
		self._rois[sl] = roi
		self._roi_source[sl] = "auto"
		self.roiEdited.emit(sl, roi)
		self._update_view()

	def _build_roi_preview_pixmap(self, img: np.ndarray, roi: tuple[float, float, float, float], size: int = 220) -> QPixmap:
		base = _array_to_pixmap(
			img,
			cmap_name=str(self.cmap_combo.currentText()),
			smooth_sigma=0.0,
			invert_cmap=self.invert_cmap_check.isChecked(),
			window_low=self._window_low,
			window_high=self._window_high,
		)
		scaled = base.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
		canvas = QPixmap(size, size)
		canvas.fill(QColor("#020617"))
		painter = QPainter(canvas)
		x0 = int((size - scaled.width()) / 2)
		y0 = int((size - scaled.height()) / 2)
		painter.drawPixmap(x0, y0, scaled)

		try:
			cy, cx, r_inner, r_outer = (float(v) for v in roi)
			h, w = int(img.shape[0]), int(img.shape[1])
			sx = float(scaled.width()) / max(1.0, float(w))
			sy = float(scaled.height()) / max(1.0, float(h))
			s = min(sx, sy)
			ccx = x0 + cx * sx
			ccy = y0 + cy * sy
			painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
			painter.setPen(QPen(QColor("#22d3ee"), 2))
			painter.drawEllipse(QPointF(ccx, ccy), 2.5, 2.5)
			if r_outer > 0:
				painter.setPen(QPen(QColor("#facc15"), 2, Qt.PenStyle.DashLine))
				painter.drawEllipse(QPointF(ccx, ccy), r_outer * s, r_outer * s)
			if r_inner > 0:
				painter.setPen(QPen(QColor("#f87171"), 2, Qt.PenStyle.DotLine))
				painter.drawEllipse(QPointF(ccx, ccy), r_inner * s, r_inner * s)
		except Exception:
			pass

		painter.end()
		return canvas

	def _on_roi_changed(self, slice_index: int, roi):
		if roi is None:
			self._rois.pop(int(slice_index), None)
			self._roi_source.pop(int(slice_index), None)
		else:
			self._rois[int(slice_index)] = tuple(float(v) for v in roi)
			self._roi_source[int(slice_index)] = "manual"
		self.roiEdited.emit(int(slice_index), roi)

	def _on_zoom_slider(self, value: int):
		zoom = max(0.40, min(5.00, float(value) / 100.0))
		if abs(self.preview.zoom() - zoom) > 1e-6:
			self.preview.set_zoom(zoom)

	def _on_preview_zoom_changed(self, zoom: float):
		self.zoom_slider.blockSignals(True)
		self.zoom_slider.setValue(int(round(zoom * 100)))
		self.zoom_slider.blockSignals(False)
		self.zoom_label.setText(f"{int(round(zoom * 100))}%")

	def _on_zoom_reset(self):
		self.preview.reset_zoom()

	def _on_intestinal_draw_toggled(self, checked: bool):
		enabled = bool(checked)
		self.preview.set_exclusion_draw_mode(enabled)
		if enabled:
			self.help_label.setText(
				"Modo ROI intestino activo: clic izquierdo agrega vértices, doble clic cierra polígono, clic derecho borra."
			)
		else:
			self.help_label.setText(
				"Mouse: clic izq = centro | Shift+clic = radio externo | Ctrl+clic = radio interno | clic der = borrar ROI | "
				"apex/base sin cavidad: usar 'Borrar internos'"
			)

	def _on_exclusion_polygon_edited(self, slice_index: int, polygon):
		sl = int(slice_index)
		gate = int(self.current_gate_index())
		if polygon is None:
			if self._intestinal_scope_mode == "all_slices":
				self._intestinal_roi_polygons = {}
			elif self._intestinal_scope_mode == "gate_slices":
				self._intestinal_roi_polygons_by_gate = {
					(k, s): p for (k, s), p in self._intestinal_roi_polygons_by_gate.items() if int(k) != gate
				}
			else:
				self._intestinal_roi_polygons.pop(sl, None)
		else:
			pts = [tuple(map(float, p)) for p in (polygon or [])]
			if len(pts) >= 3:
				if self._intestinal_scope_mode == "all_slices" and self._cube is not None:
					n_slices = int(self._cube.shape[1])
					self._intestinal_roi_polygons = {int(i): list(pts) for i in range(n_slices)}
				elif self._intestinal_scope_mode == "gate_slices":
					if self._cube is not None:
						n_slices = int(self._cube.shape[1])
						for i in range(n_slices):
							self._intestinal_roi_polygons_by_gate[(gate, int(i))] = list(pts)
					else:
						self._intestinal_roi_polygons_by_gate[(gate, sl)] = list(pts)
				else:
					self._intestinal_roi_polygons[sl] = pts
			else:
				if self._intestinal_scope_mode == "gate_slices":
					self._intestinal_roi_polygons_by_gate.pop((gate, sl), None)
				else:
					self._intestinal_roi_polygons.pop(sl, None)
		self._update_view()

	def _clear_intestinal_roi_current_slice(self):
		sl = int(self.slice_slider.value())
		if self._intestinal_scope_mode == "all_slices":
			self._intestinal_roi_polygons = {}
		elif self._intestinal_scope_mode == "gate_slices":
			gate = int(self.current_gate_index())
			self._intestinal_roi_polygons_by_gate = {
				(k, s): p for (k, s), p in self._intestinal_roi_polygons_by_gate.items() if int(k) != gate
			}
		else:
			self._intestinal_roi_polygons.pop(sl, None)
		self.preview.set_exclusion_polygon([])
		self._update_view()

	def _on_intestinal_scope_changed(self, _index: int):
		scope = self.intestinal_scope_combo.currentData()
		self._intestinal_scope_mode = str(scope or "slice")

	def _on_intestinal_apply_toggled(self, checked: bool):
		self._intestinal_apply_enabled = bool(checked)
		self._refresh_intestinal_apply_button_text()
		self._update_view()

	def _refresh_intestinal_apply_button_text(self):
		if self._intestinal_apply_enabled:
			self.intestinal_apply_btn.setText("ROI intestinal ON")
		else:
			self.intestinal_apply_btn.setText("Aplicar ROI intestino")

	def _on_intestinal_attenuation_changed(self, value: int):
		self._intestinal_attenuation_pct = max(0, min(100, int(value)))
		self.intestinal_atten_label.setText(f"{int(self._intestinal_attenuation_pct)}%")

	def _on_intestinal_feather_changed(self, value: int):
		self._intestinal_feather_px = max(0, min(16, int(value)))
		self.intestinal_feather_label.setText(f"{int(self._intestinal_feather_px)} px")

	def resizeEvent(self, event):
		super().resizeEvent(event)
		self._update_view()
