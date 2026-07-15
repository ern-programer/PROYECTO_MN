"""SINCRO - ui.cine_widget.

Visor interactivo para navegar gates y slices, con edición visual básica de ROI.
"""
from __future__ import annotations

import math
from typing import Optional

import matplotlib
import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, label
from matplotlib.colors import LinearSegmentedColormap
from PyQt6.QtCore import QTimer, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget, QHBoxLayout


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
			painter.drawText(12, 22, label)
		else:
			painter.setPen(QColor("#ffffff"))
			painter.drawText(12, 22, f"Slice {self._slice_index + 1}")

	def mousePressEvent(self, event):
		if self._base_pixmap is None or self._frame_shape is None:
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

	def wheelEvent(self, event):
		delta = event.angleDelta().y()
		if delta > 0:
			self.set_zoom(self._zoom + 0.10)
		elif delta < 0:
			self.set_zoom(self._zoom - 0.10)
		event.accept()


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
		self.cmap_combo.addItems(["gray", "hot", "cool", "prism", "french"])
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
		controls.addWidget(self.auto_roi_empty_only_check, 0, 8)
		controls.addWidget(self.show_auto_roi_check, 0, 9)

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

		preview_row = QHBoxLayout()
		preview_row.addWidget(self.preview, 1)
		preview_row.addLayout(window_panel)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(4, 4, 4, 4)
		layout.setSpacing(2)
		layout.addLayout(preview_row)
		layout.addWidget(self.help_label)
		layout.addLayout(controls)

		self.preview.roiChanged.connect(self._on_roi_changed)
		self.preview.zoomChanged.connect(self._on_preview_zoom_changed)
		self.setMinimumHeight(260)
		self.setSizePolicy(self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())

	def set_manual_rois(self, rois: dict[int, tuple[float, float, float, float]] | None):
		old_sources = dict(self._roi_source)
		self._rois = dict(rois or {})
		self._roi_source = {int(sl): old_sources.get(int(sl), "manual") for sl in self._rois.keys()}
		self._update_view()

	def roi_for_slice(self, slice_index: int):
		return self._rois.get(int(slice_index))

	def estimate_auto_roi_for_slice(self, slice_index: int):
		if self._cube is None:
			return None
		sl = int(slice_index)
		if sl < 0 or sl >= int(self._cube.shape[1]):
			return None
		img = np.asarray(self._cube[:, sl].mean(axis=0), dtype=np.float64)
		return self._auto_roi_from_image(img)

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
		frame = self._cube[gate, sl]
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
			if empty_only and self._rois.get(sl) is not None:
				continue
			roi = self.estimate_auto_roi_for_slice(sl)
			if roi is None:
				continue
			self._rois[sl] = roi
			self._roi_source[sl] = "auto"
			self.roiEdited.emit(sl, roi)
		self._update_view()

	def _auto_roi_from_image(self, img: np.ndarray):
		img = gaussian_filter(np.asarray(img, dtype=np.float64), sigma=1.0)
		if not np.isfinite(img).any() or float(np.max(img)) <= 0.0:
			return None

		thr = float(np.percentile(img[np.isfinite(img)], 70.0))
		bin_mask = img > thr
		lbl, n = label(bin_mask)
		if n <= 0:
			return None
		counts = np.bincount(lbl.ravel())
		counts[0] = 0
		largest = int(np.argmax(counts))
		mask = lbl == largest
		if int(mask.sum()) < 8:
			return None

		cy, cx = center_of_mass(mask)
		ys, xs = np.nonzero(mask)
		d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
		if d.size < 4:
			return None
		r_inner = float(np.percentile(d, 22))
		r_outer = float(np.percentile(d, 82))
		if r_outer <= r_inner:
			r_outer = r_inner + 1.0
		return (float(cy), float(cx), r_inner, r_outer)

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

	def resizeEvent(self, event):
		super().resizeEvent(event)
		self._update_view()
