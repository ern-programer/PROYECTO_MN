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
from PyQt6.QtWidgets import (
	QCheckBox, QComboBox, QDialog, QGridLayout, QLabel, QPushButton, QSlider,
	QToolButton, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox, QSizePolicy,
)

from core.col_registry import register_all_colormaps, available_colormaps
from core.intestinal_subtraction import (
	BACKGROUND_METHODS,
	apply_intestinal_subtraction,
	estimate_background_map,
)
from core.lv_center import refine_center_to_cavity
from ui.floating_toolbar import FloatingToolbar


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

	w0 = max(0.0, float(window_low))
	# window_high puede superar 1.0 (hasta 2.0 = 200%) para 'desquemar' la imagen:
	# el rango (w1 - w0) se amplía y los valores quedan comprimidos (menos saturados).
	w1 = max(0.0, min(2.0, float(window_high)))
	if w1 <= w0:
		w1 = min(2.0, w0 + 0.01)
	data = np.clip((data - w0) / max(1e-8, (w1 - w0)), 0.0, 1.0)

	name = f"{cmap_name}_r" if invert_cmap else str(cmap_name)
	cmap = _resolve_cmap(name)
	rgb = np.asarray(cmap(np.clip(data, 0.0, 1.0))[..., :3], dtype=np.float32)

	rgb8 = (rgb * 255.0).astype(np.uint8)
	h, w, _ = rgb8.shape
	qimg = QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
	return QPixmap.fromImage(qimg.copy())


class RangeSlider(QWidget):
	"""Slider vertical de DOS handles (base y top) en un solo control, rango 0-200%.

	Handle inferior = base de la ventana (0-200), handle superior = top (0-200).
	El top puede pasar de 100% para 'desquemar' (desaturar) la imagen. Marcas
	cada 50% con un stop visual. Emite valuesChanged(low, high) al arrastrar.
	"""

	valuesChanged = pyqtSignal(int, int)
	RANGE = 200
	STOP_MARKS = (0, 50, 100, 150, 200)

	def __init__(self, parent=None):
		super().__init__(parent)
		self._low = 0
		self._high = 100
		self._drag: str | None = None  # "low" | "high" | None
		self.setMinimumWidth(28)
		self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
		self.setToolTip("Ventana: handle inferior = Base, superior = Top (hasta 200%).")

	def set_values(self, low: int, high: int) -> None:
		low = max(0, min(self.RANGE, int(low)))
		high = max(0, min(self.RANGE, int(high)))
		if high <= low:
			high = min(self.RANGE, low + 1)
		if low != self._low or high != self._high:
			self._low, self._high = low, high
			self.valuesChanged.emit(self._low, self._high)
			self.update()

	def values(self) -> tuple[int, int]:
		return self._low, self._high

	def _val_to_y(self, val: int) -> float:
		h = self.height()
		margin = 10.0
		usable = h - 2 * margin
		return margin + usable * (1.0 - (val / self.RANGE))

	def _y_to_val(self, y: float) -> int:
		h = self.height()
		margin = 10.0
		usable = h - 2 * margin
		if usable <= 0:
			return 0
		frac = 1.0 - ((y - margin) / usable)
		return max(0, min(self.RANGE, round(frac * self.RANGE)))

	def _handle_at(self, y: float) -> str | None:
		y_low = self._val_to_y(self._low)
		y_high = self._val_to_y(self._high)
		# El más cercano al click, con tolerancia de 12px.
		if abs(y - y_high) <= 12 and abs(y - y_high) <= abs(y - y_low):
			return "high"
		if abs(y - y_low) <= 12:
			return "low"
		return None

	def mousePressEvent(self, event):
		self._drag = self._handle_at(event.position().y())
		if self._drag:
			self._apply_y(event.position().y())

	def mouseMoveEvent(self, event):
		if self._drag:
			self._apply_y(event.position().y())

	def mouseReleaseEvent(self, event):
		self._drag = None

	def _apply_y(self, y: float) -> None:
		val = self._y_to_val(y)
		if self._drag == "low":
			self.set_values(min(val, self._high - 1), self._high)
		elif self._drag == "high":
			self.set_values(self._low, max(val, self._low + 1))

	def paintEvent(self, event):
		p = QPainter(self)
		p.setRenderHint(QPainter.RenderHint.Antialiasing)
		cx = self.width() / 2.0
		# Groove (riel) vertical.
		y0 = self._val_to_y(self.RANGE)
		y1 = self._val_to_y(0)
		p.setPen(Qt.PenStyle.NoPen)
		p.setBrush(QColor("#d1d5db"))
		p.drawRoundedRect(QRectF(cx - 3, y0, 6, y1 - y0), 3, 3)
		# Zona activa (entre base y top).
		ya = self._val_to_y(self._high)
		yb = self._val_to_y(self._low)
		p.setBrush(QColor("#93c5fd"))
		p.drawRoundedRect(QRectF(cx - 3, ya, 6, yb - ya), 3, 3)
		# Marcas cada 50%.
		p.setPen(QPen(QColor("#64748b"), 1))
		for m in self.STOP_MARKS:
			ym = self._val_to_y(m)
			p.drawLine(QPointF(cx - 7, ym), QPointF(cx + 7, ym))
		# Handles (base abajo, top arriba).
		for val, color in ((self._low, "#2563eb"), (self._high, "#1e40af")):
			y = self._val_to_y(val)
			p.setBrush(QColor(color))
			p.setPen(QPen(QColor("#0f172a"), 1))
			p.drawRoundedRect(QRectF(cx - 9, y - 6, 18, 12), 3, 3)


class GateMontageLabel(QLabel):
	"""Mosaico de todos los gates de un slice con ROIs superpuestos.

	Muestra los N gates del slice actual en una grilla (p.ej. 2x4 para 8 gates),
	con el ROI de cada gate dibujado. Click en una celda selecciona ese gate.
	"""
	gateSelected = pyqtSignal(int)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setMinimumSize(360, 220)
		self.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.setStyleSheet("background:#111; color:#ddd; border:1px solid #444;")
		self.setCursor(Qt.CursorShape.PointingHandCursor)
		self._cube = None
		self._slice_index = 0
		self._current_gate = 0
		self._rois_by_gate: dict[tuple[int, int], tuple[float, float, float, float]] = {}
		self._rois_common: dict[int, tuple[float, float, float, float]] = {}
		self._per_gate_mode = False
		self._cmap_name = "hot"
		self._smooth_sigma = 0.0
		self._invert_cmap = False
		self._window_low = 0.0
		self._window_high = 1.0
		self._cell_rects: list[tuple[int, QRectF]] = []

	def set_cube(self, cube: np.ndarray | None):
		self._cube = cube
		self.update()

	def set_slice_index(self, slice_index: int):
		self._slice_index = int(slice_index)
		self.update()

	def set_current_gate(self, gate_index: int):
		self._current_gate = int(gate_index)
		self.update()

	def set_rois(self, rois_common: dict, rois_by_gate: dict, per_gate_mode: bool):
		self._rois_common = {int(k): tuple(v) for k, v in (rois_common or {}).items()}
		self._rois_by_gate = {tuple(k): tuple(v) for k, v in (rois_by_gate or {}).items()}
		self._per_gate_mode = bool(per_gate_mode)
		self.update()

	def set_display_params(self, cmap_name: str, smooth_sigma: float, invert_cmap: bool, window_low: float, window_high: float):
		self._cmap_name = str(cmap_name)
		self._smooth_sigma = float(smooth_sigma)
		self._invert_cmap = bool(invert_cmap)
		self._window_low = float(window_low)
		self._window_high = float(window_high)
		self.update()

	def _roi_for_gate(self, gate_index: int) -> tuple[float, float, float, float] | None:
		key = (int(gate_index), self._slice_index)
		if key in self._rois_by_gate:
			return self._rois_by_gate[key]
		return self._rois_common.get(self._slice_index)

	def paintEvent(self, event):
		painter = QPainter(self)
		painter.fillRect(self.rect(), QColor("#111111"))
		self._cell_rects = []
		if self._cube is None or self._cube.ndim != 4:
			painter.setPen(QColor("#dddddd"))
			painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sin datos gated")
			return
		n_gates = int(self._cube.shape[0])
		n_slices = int(self._cube.shape[1])
		if self._slice_index < 0 or self._slice_index >= n_slices:
			painter.setPen(QColor("#dddddd"))
			painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Slice fuera de rango")
			return

		cols = int(np.ceil(np.sqrt(n_gates)))
		rows = int(np.ceil(n_gates / cols))
		w = self.width()
		h = self.height()
		cell_w = w / cols
		cell_h = h / rows

		for g in range(n_gates):
			row = g // cols
			col = g % cols
			x0 = col * cell_w
			y0 = row * cell_h
			cell_rect = QRectF(x0, y0, cell_w, cell_h)
			self._cell_rects.append((g, cell_rect))

			frame = np.asarray(self._cube[g, self._slice_index], dtype=np.float64)
			pix = _array_to_pixmap(
				frame,
				cmap_name=self._cmap_name,
				smooth_sigma=self._smooth_sigma,
				invert_cmap=self._invert_cmap,
				window_low=self._window_low,
				window_high=self._window_high,
			)
			scaled = pix.scaled(
				int(cell_w), int(cell_h),
				Qt.AspectRatioMode.KeepAspectRatio,
				Qt.TransformationMode.SmoothTransformation,
			)
			px = x0 + (cell_w - scaled.width()) / 2.0
			py = y0 + (cell_h - scaled.height()) / 2.0
			painter.drawPixmap(int(px), int(py), scaled)

			# ROI del gate
			roi = self._roi_for_gate(g)
			if roi is not None:
				cy, cx, r_inner, r_outer = roi
				hf, wf = frame.shape[:2]
				scale = min(scaled.width() / max(1, wf), scaled.height() / max(1, hf))
				cx_p = px + cx * scale
				cy_p = py + cy * scale
				r_in_p = r_inner * scale
				r_out_p = r_outer * scale
				painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
				if r_out_p > 0:
					painter.setPen(QPen(QColor("#ffcc00"), 1.5, Qt.PenStyle.DashLine))
					painter.drawEllipse(QPointF(cx_p, cy_p), r_out_p, r_out_p)
				if r_in_p > 0:
					painter.setPen(QPen(QColor("#ff6666"), 1.5, Qt.PenStyle.DotLine))
					painter.drawEllipse(QPointF(cx_p, cy_p), r_in_p, r_in_p)
				painter.setPen(QPen(QColor("#00d1ff"), 1.5))
				painter.drawEllipse(QPointF(cx_p, cy_p), 3, 3)

			# Highlight del gate actual
			if g == self._current_gate:
				painter.setPen(QPen(QColor("#d61f1f"), 2.5))
				painter.drawRect(cell_rect.adjusted(1, 1, -1, -1))

			# Etiqueta
			painter.setPen(QColor("#ffffff"))
			painter.drawText(int(x0) + 6, int(y0) + 18, f"G{g + 1}")

		painter.end()

	def mousePressEvent(self, event):
		if event.button() == Qt.MouseButton.LeftButton:
			pos = event.position()
			for g, rect in self._cell_rects:
				if rect.contains(pos):
					self.gateSelected.emit(int(g))
					return
		super().mousePressEvent(event)


class RoiImageLabel(QLabel):
	roiChanged = pyqtSignal(int, object)
	zoomChanged = pyqtSignal(float)
	middleClicked = pyqtSignal()
	exclusionPolygonEdited = pyqtSignal(int, object)
	centerPicked = pyqtSignal(int, object)

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
		self._overlay_contours: list[tuple[QColor, np.ndarray]] = []
		self._exclusion_polygon: list[tuple[float, float]] = []
		self._draft_exclusion_polygon: list[tuple[float, float]] = []
		self._reference_polygons: list[list[tuple[float, float]]] = []
		self._draw_exclusion_mode = False
		self._center_pick_mode = False
		self._manual_centers: dict[int, tuple[float, float]] = {}
		self._message = "Cargá un estudio para ver el cine"
		self._zoom = 1.0

	def set_center_pick_mode(self, enabled: bool):
		"""Modo 'fijar centro de cavidad': el clic izquierdo define solo el
		centro del corte (no un ROI completo) y el derecho lo borra."""
		self._center_pick_mode = bool(enabled)
		self.update()

	def set_manual_centers(self, centers: dict[int, tuple[float, float]] | None):
		"""Recibe el dict slice->(cy,cx) de centros manuales para dibujarlos."""
		self._manual_centers = dict(centers or {})
		self.update()

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

	def set_overlay_contours(self, contours: list[tuple[QColor, np.ndarray]] | None):
		"""Define contornos arbitrarios (coords de imagen) para superponer."""
		normalized: list[tuple[QColor, np.ndarray]] = []
		for item in contours or []:
			if not isinstance(item, tuple) or len(item) != 2:
				continue
			color, points = item
			arr = np.asarray(points, dtype=np.float64)
			if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
				continue
			normalized.append((QColor(color), arr))
		self._overlay_contours = normalized
		self.update()

	def set_reference_polygons(self, polygons: list[list[tuple[float, float]]] | None):
		self._reference_polygons = [
			[tuple(map(float, p)) for p in (poly or [])]
			for poly in (polygons or [])
			if poly and len(poly) >= 3
		]
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

	def _points_to_widget(self, points_yx: np.ndarray) -> list[QPointF]:
		rect = self._image_rect()
		if rect is None or self._frame_shape is None:
			return []
		h, w = self._frame_shape
		sx = rect.width() / max(1.0, float(w))
		sy = rect.height() / max(1.0, float(h))
		out: list[QPointF] = []
		for p in np.asarray(points_yx, dtype=np.float64):
			if p.shape[0] < 2:
				continue
			y = float(p[0])
			x = float(p[1])
			out.append(QPointF(rect.x() + x * sx, rect.y() + y * sy))
		return out

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

		# ROI de referencia del asa intestinal limpia (de donde sale el nivel de
		# fondo a restar). Se dibujan en cian para no confundirlas con la zona a
		# corregir, que va en magenta.
		for ref_poly in self._reference_polygons:
			rpts = self._polygon_to_widget(ref_poly)
			if len(rpts) < 3:
				continue
			painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
			painter.setPen(QPen(QColor("#22d3ee"), 2, Qt.PenStyle.DotLine))
			painter.setBrush(QColor(34, 211, 238, 45))
			painter.drawPolygon(QPolygonF(rpts))

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

		for color, pts_img in self._overlay_contours:
			wpoly = self._points_to_widget(pts_img)
			if len(wpoly) < 3:
				continue
			painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
			painter.setPen(QPen(color, 1.8, Qt.PenStyle.SolidLine))
			painter.setBrush(Qt.BrushStyle.NoBrush)
			poly = QPolygonF(wpoly)
			painter.drawPolyline(poly)
			painter.drawLine(wpoly[-1], wpoly[0])

		# Marcador del centro manual del operador (cruz verde), independiente del
		# ROI: se dibuja aunque no haya ROI en el corte para dar feedback del clic.
		man_c = self._manual_centers.get(self._slice_index)
		if man_c is not None and self._frame_shape is not None:
			rect_img = self._image_rect()
			if rect_img is not None:
				h, w = self._frame_shape
				sx = rect_img.width() / max(1.0, float(w))
				sy = rect_img.height() / max(1.0, float(h))
				px = rect_img.x() + float(man_c[1]) * sx
				py = rect_img.y() + float(man_c[0]) * sy
				painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
				painter.setPen(QPen(QColor("#39ff14"), 2))
				painter.drawLine(QPointF(px - 8, py), QPointF(px + 8, py))
				painter.drawLine(QPointF(px, py - 8), QPointF(px, py + 8))
				painter.drawEllipse(QPointF(px, py), 5, 5)

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
			# Sin texto sobre la imagen en el visor compacto (el slice/gate ya se
			# muestran debajo). Solo se avisa en modos especiales de edición.
			label = ""
			if self._center_pick_mode:
				label = "CENTRO MANUAL: clic izq = fijar · clic der = borrar"
			if self._draw_exclusion_mode:
				label = (label + " | " if label else "") + "ROI intestino: clic agrega, doble clic cierra, clic der borra"
			if label:
				painter.drawText(12, 22, label)

	def mousePressEvent(self, event):
		if self._base_pixmap is None or self._frame_shape is None:
			return
		if event.button() == Qt.MouseButton.MiddleButton:
			self.middleClicked.emit()
			return
		if self._center_pick_mode:
			mapped = self._widget_to_image(event.position())
			if mapped is None:
				return
			if event.button() == Qt.MouseButton.RightButton:
				self._manual_centers.pop(self._slice_index, None)
				self.centerPicked.emit(self._slice_index, None)
				self.update()
				return
			if event.button() == Qt.MouseButton.LeftButton:
				cy, cx = mapped
				self._manual_centers[self._slice_index] = (float(cy), float(cx))
				self.centerPicked.emit(self._slice_index, (float(cy), float(cx)))
				self.update()
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
	roiEditedGate = pyqtSignal(int, int, object)  # (gate, slice, roi)
	playStateChanged = pyqtSignal(bool)
	playbackSpeedChanged = pyqtSignal(int)
	activated = pyqtSignal()
	centerPicked = pyqtSignal(int, object)  # (slice, (cy,cx)) o (slice, None) para borrar

	def __init__(self, parent=None, *, compact_viewer=False, is_compare=False):
		super().__init__(parent)
		# Visor reducido: solo controles para MIRAR el cine (ventana principal).
		self._compact_viewer = bool(compact_viewer)
		# Visor de COMPARACIÓN (2da etapa): no lleva título propio ni sliders
		# Base/Top — el título lo pone el contenedor y los sliders son solo de la
		# 1ra etapa. Así las dos imágenes quedan alineadas a la misma altura.
		self._is_compare = bool(is_compare)
		self._cube = None
		self._rois: dict[int, tuple[float, float, float, float]] = {}
		self._roi_source: dict[int, str] = {}
		# QC por gate: ROIs manuales por (gate, slice). Cuando el modo "ROI por
		# gate" está activo, la edición afecta solo al gate actual, no a todos.
		self._rois_by_gate: dict[tuple[int, int], tuple[float, float, float, float]] = {}
		self._roi_by_gate_source: dict[tuple[int, int], str] = {}
		self._per_gate_roi_mode = False
		self._current_slice = 0
		self._playing = False
		self._smooth_sigma = 0.0
		self._window_low = 0.0
		self._window_high = 1.0
		self._auto_roi_method = "robusto"
		self._refine_cavity_center = False
		self._intestinal_roi_polygons: dict[int, list[tuple[float, float]]] = {}
		self._intestinal_roi_polygons_by_gate: dict[tuple[int, int], list[tuple[float, float]]] = {}
		# ROI de referencia sobre el asa intestinal limpia: de ahí sale el nivel de
		# fondo que después se resta en la zona de solapamiento. Son varias por
		# corte (típicamente la "entrada" y la "salida" del asa).
		self._intestinal_ref_polygons: dict[int, list[list[tuple[float, float]]]] = {}
		self._intestinal_ref_polygons_by_gate: dict[tuple[int, int], list[list[tuple[float, float]]]] = {}
		self._intestinal_mode = "attenuate"
		self._intestinal_bg_method = "idw"
		self._intestinal_draw_role = "target"
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
		if self._compact_viewer:
			# Visor de la ventana principal: TAMAÑO FIJO 160x160 px (el recuadro),
			# con ~150x150 px de imagen efectiva dentro. No crece con el panel: el
			# usuario pidió un cine chico y cuadrado de tamaño constante.
			self.preview.setFixedSize(160, 160)
		else:
			self.preview.setMinimumSize(220, 220)
			self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

		self.gate_slider = QSlider(Qt.Orientation.Horizontal)
		self.slice_slider = QSlider(Qt.Orientation.Horizontal)
		if self._compact_viewer:
			# Visor de la ventana principal: sliders CORTOS (como el mockup), no
			# estirados. Ancho fijo para que los controles queden compactos al
			# lado de la imagen.
			for _sl in (self.gate_slider, self.slice_slider):
				_sl.setFixedWidth(120)
				_sl.setMaximumHeight(18)
				_sl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
		else:
			self.gate_slider.setMinimumWidth(180)
			self.gate_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
			self.slice_slider.setMinimumWidth(180)
			self.slice_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
		self.zoom_slider.setMinimumWidth(180)
		self.zoom_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
		self.intestinal_mode_combo = QComboBox()
		self.intestinal_mode_combo.addItem("Atenuar %", "attenuate")
		self.intestinal_mode_combo.addItem("Restar fondo estimado", "subtract")
		self.intestinal_mode_combo.currentIndexChanged.connect(self._on_intestinal_mode_changed)
		self.intestinal_ref_toggle_btn = QPushButton("ROI referencia")
		self.intestinal_ref_toggle_btn.setCheckable(True)
		self.intestinal_ref_toggle_btn.toggled.connect(self._on_intestinal_ref_draw_toggled)
		self.intestinal_ref_clear_btn = QPushButton("Borrar referencias")
		self.intestinal_ref_clear_btn.clicked.connect(self._clear_intestinal_references_current_slice)
		self.intestinal_ref_count_label = QLabel("0 ref.")
		self.intestinal_ref_count_label.setStyleSheet("color:#0e7490;")
		self.intestinal_bg_method_combo = QComboBox()
		self.intestinal_bg_method_combo.addItem("Interpolado (IDW)", "idw")
		self.intestinal_bg_method_combo.addItem("Media simple", "mean")
		self.intestinal_bg_method_combo.currentIndexChanged.connect(self._on_intestinal_bg_method_changed)
		self.intestinal_preview_btn = QPushButton("Antes/después")
		self.intestinal_preview_btn.clicked.connect(self._open_intestinal_preview_dialog)
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
		self.intestinal_mode_combo.setToolTip(
			"Atenuar %: reduce un porcentaje de cuentas (solo mejora el Auto ROI; no cambia la amplitud relativa de la fase).\n"
			"Restar fondo estimado: mide el nivel del intestino en las ROI de referencia y lo resta como componente constante.\n"
			"La resta SÍ recupera vóxeles para el análisis de fase y no desplaza la fase, porque el intestino no late."
		)
		self.intestinal_ref_toggle_btn.setToolTip(
			"Dibuja ROI de REFERENCIA sobre zona de fondo limpia, fuera de la región a corregir.\n"
			"Para un asa intestinal: la 'entrada' y la 'salida' donde se ve sin miocardio encima.\n"
			"Para fondo general: varias zonas repartidas, siempre DENTRO del paciente (nunca sobre aire,\n"
			"que daría mediana ~0 y subestimaría el fondo). Se admiten todas las que quieras."
		)
		self.intestinal_bg_method_combo.setToolTip(
			"Cómo se combina el nivel de las ROI de referencia:\n"
			"Interpolado (IDW): el fondo varía dentro de la zona según la referencia más cercana.\n"
			"Media simple: promedio aritmético de los niveles, aplicado como una única constante.\n"
			"En media simple cada ROI pesa igual, sin importar su tamaño.\n\n"
			"Restar fondo NO es corrección de atenuación (el fondo es aditivo, la atenuación multiplicativa).\n"
			"Ver docs/FUNDAMENTO_MATEMATICO_SUSTRACCION_FONDO.md"
		)
		self.intestinal_preview_btn.setToolTip(
			"Compara lado a lado el corte actual antes y después de la sustracción, "
			"con el mapa de lo restado y el control de calidad de amplitud."
		)
		self.intestinal_ref_clear_btn.setToolTip("Borra las ROI de referencia según el alcance seleccionado.")
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
		self.per_gate_roi_check = QCheckBox("ROI por gate")
		self.per_gate_roi_check.setChecked(False)
		self.per_gate_roi_check.setToolTip(
			"QC por gate: si está activo, la edición de ROI afecta SOLO al gate actual "
			"(cada gate puede tener su propio ROI). Si está apagado, el ROI es común a todos los gates."
		)
		self.per_gate_roi_check.toggled.connect(self._on_per_gate_roi_mode_toggled)
		self.qc_gates_btn = QPushButton("QC gates")
		self.qc_gates_btn.setToolTip("Abre la vista QC de todos los gates del slice actual para editar ROI por gate.")
		self.qc_gates_btn.clicked.connect(self._open_gate_qc_dialog)

		self.play_button = QPushButton("▶ Reproducir")
		self.play_button.clicked.connect(self.toggle_playback)
		self.play_button.setToolTip("Reproduce los gates en tiempo real.")
		self.speed_slider = QSlider(Qt.Orientation.Horizontal)
		self.speed_slider.setRange(50, 600)
		self.speed_slider.setValue(250)
		self.speed_slider.setMaximumHeight(20)
		if self._compact_viewer:
			self.speed_slider.setFixedWidth(120)
			self.speed_slider.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
		else:
			self.speed_slider.setMinimumWidth(180)
			self.speed_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
		self.speed_slider.valueChanged.connect(self._on_speed_change)
		self.speed_label = QLabel("250 ms")
		self.speed_slider.setToolTip("Tiempo por frame: más bajo = más rápido.")
		self.speed_prev_btn = QPushButton("<")
		self.speed_next_btn = QPushButton(">")
		for btn in (self.speed_prev_btn, self.speed_next_btn):
			btn.setFixedWidth(24)
			btn.setMaximumHeight(20)
			btn.setAutoRepeat(True)
			btn.setAutoRepeatDelay(260)
			btn.setAutoRepeatInterval(70)
		self.speed_prev_btn.setToolTip("Más rápido")
		self.speed_next_btn.setToolTip("Más lento")
		self.speed_prev_btn.clicked.connect(lambda: self._step_slider(self.speed_slider, -10))
		self.speed_next_btn.clicked.connect(lambda: self._step_slider(self.speed_slider, 10))

		self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
		self.smooth_slider.setRange(0, 30)
		self.smooth_slider.setValue(0)
		self.smooth_slider.setMaximumHeight(20)
		self.smooth_slider.setMinimumWidth(180)
		self.smooth_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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

		# Ventana (Base/Top): en el visor compacto UN SOLO slider de dos handles
		# (RangeSlider, 0-200%) para ahorrar ancho; en el completo, dos QSlider.
		if self._compact_viewer:
			self.range_slider = RangeSlider()
			self.range_slider.set_values(0, 100)
			self.range_slider.valuesChanged.connect(self._on_range_window_change)
			# Aliases para que el resto del código (auto ventana, labels) siga andando.
			self.window_low_slider = None
			self.window_high_slider = None
		else:
			self.range_slider = None
			self.window_low_slider = QSlider(Qt.Orientation.Horizontal)
			self.window_low_slider.setRange(0, 99)
			self.window_low_slider.setValue(0)
			self.window_low_slider.setMaximumHeight(20)
			self.window_low_slider.setMaximumWidth(220)
			self.window_low_slider.valueChanged.connect(self._on_window_low_change)
			self.window_high_slider = QSlider(Qt.Orientation.Horizontal)
			self.window_high_slider.setRange(1, 100)
			self.window_high_slider.setValue(100)
			self.window_high_slider.setMaximumHeight(20)
			self.window_high_slider.setMaximumWidth(220)
			self.window_high_slider.valueChanged.connect(self._on_window_high_change)
		self.window_low_label = QLabel("0%")
		self.window_high_label = QLabel("100%")

		# Menús secundarios (ROI automático / ROI intestinal): barras flotantes
		# que no empujan el layout. Se construyen siempre; el visor reducido solo
		# usa el de intestino.
		auto_roi_grid = QGridLayout()
		auto_roi_grid.setHorizontalSpacing(8)
		auto_roi_grid.setVerticalSpacing(2)
		auto_roi_grid.addWidget(self.auto_window_btn, 0, 0)
		auto_roi_grid.addWidget(self.auto_roi_btn, 0, 1)
		auto_roi_grid.addWidget(self.auto_roi_all_btn, 0, 2)
		auto_roi_grid.addWidget(self.auto_roi_config_btn, 0, 3)
		auto_roi_grid.addWidget(self.auto_roi_help_btn, 0, 4)
		auto_roi_grid.addWidget(self.auto_roi_method_label, 0, 5)
		auto_roi_grid.addWidget(self.auto_roi_empty_only_check, 0, 6)
		auto_roi_grid.addWidget(self.show_auto_roi_check, 0, 7)
		auto_roi_grid.addWidget(self.per_gate_roi_check, 0, 8)
		auto_roi_grid.addWidget(self.qc_gates_btn, 0, 9)
		auto_roi_btn_menu = self._build_toolbar_button(
			"ROI automático ▾", [auto_roi_grid], key="roi_panel_auto_roi",
			tooltip="Detección automática del corazón por slice/gate, config y ayuda del método.",
		)

		intestinal_grid = QGridLayout()
		intestinal_grid.setHorizontalSpacing(8)
		intestinal_grid.setVerticalSpacing(2)
		intestinal_grid.addWidget(self.intestinal_roi_toggle_btn, 0, 0)
		intestinal_grid.addWidget(self.intestinal_apply_btn, 0, 1)
		intestinal_grid.addWidget(self.intestinal_roi_clear_btn, 0, 2)
		intestinal_grid.addWidget(QLabel("Atenuar int."), 0, 3)
		intestinal_grid.addWidget(self.intestinal_atten_slider, 0, 4)
		intestinal_grid.addWidget(self.intestinal_atten_label, 0, 5)
		intestinal_grid.addWidget(QLabel("Feather"), 1, 0)
		intestinal_grid.addWidget(self.intestinal_feather_slider, 1, 1)
		intestinal_grid.addWidget(self.intestinal_feather_label, 1, 2)
		intestinal_grid.addWidget(QLabel("Alcance int."), 1, 3)
		intestinal_grid.addWidget(self.intestinal_scope_combo, 1, 4)
		intestinal_grid.addWidget(QLabel("Modo"), 2, 0)
		intestinal_grid.addWidget(self.intestinal_mode_combo, 2, 1, 1, 2)
		intestinal_grid.addWidget(self.intestinal_ref_toggle_btn, 2, 3)
		intestinal_grid.addWidget(self.intestinal_ref_clear_btn, 2, 4)
		intestinal_grid.addWidget(self.intestinal_ref_count_label, 2, 5)
		intestinal_grid.addWidget(QLabel("Fondo"), 3, 0)
		intestinal_grid.addWidget(self.intestinal_bg_method_combo, 3, 1, 1, 2)
		intestinal_grid.addWidget(self.intestinal_preview_btn, 3, 3, 1, 2)
		intestinal_btn_menu = self._build_toolbar_button(
			"ROI intestinal ▾", [intestinal_grid], key="roi_panel_intestinal",
			tooltip="Dibujo y atenuación manual del intestino, para no contaminar el Auto ROI del corazón.",
		)

		_lbl_align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
		if self._compact_viewer:
			# Visor reducido (ventana principal): solo lo necesario para MIRAR el
			# cine. Sin slider de zoom (se usa la rueda del mouse), sin smooth, sin
			# matriz ni Auto ROI. Play + ROI intestinal arriba; luego colormap /
			# invertir; y navegación de gate / slice / velocidad.
			self.play_button.setText("▶")
			self.play_button.setMaximumWidth(52)
			nav_grid = QGridLayout()
			nav_grid.setHorizontalSpacing(6)
			nav_grid.setVerticalSpacing(3)
			top_row = QHBoxLayout()
			top_row.setSpacing(6)
			top_row.addWidget(self.play_button)
			top_row.addWidget(intestinal_btn_menu)
			top_row.addStretch(1)
			nav_grid.addLayout(top_row, 0, 0, 1, 4)
			nav_grid.addWidget(QLabel("Colormap"), 1, 0, _lbl_align)
			nav_grid.addWidget(self.cmap_combo, 1, 1, 1, 2)
			nav_grid.addWidget(self.invert_cmap_check, 1, 3)
			nav_grid.addWidget(QLabel("Gate"), 2, 0, _lbl_align)
			nav_grid.addWidget(self.gate_prev_btn, 2, 1, _lbl_align)
			nav_grid.addWidget(self.gate_slider, 2, 2)
			nav_grid.addWidget(self.gate_next_btn, 2, 3)
			nav_grid.addWidget(QLabel("Slice"), 3, 0, _lbl_align)
			nav_grid.addWidget(self.slice_prev_btn, 3, 1, _lbl_align)
			nav_grid.addWidget(self.slice_slider, 3, 2)
			nav_grid.addWidget(self.slice_next_btn, 3, 3)
			nav_grid.addWidget(QLabel("Speed"), 4, 0, _lbl_align)
			nav_grid.addWidget(self.speed_prev_btn, 4, 1, _lbl_align)
			nav_grid.addWidget(self.speed_slider, 4, 2)
			# Botón '>' y el valor '250 ms' juntos en la misma celda (sin gap),
			# para que el número quede pegado al lado del botón.
			_speed_end = QHBoxLayout()
			_speed_end.setContentsMargins(0, 0, 0, 0)
			_speed_end.setSpacing(3)
			_speed_end.addWidget(self.speed_next_btn)
			_speed_end.addWidget(self.speed_label)
			_speed_end_w = QWidget()
			_speed_end_w.setLayout(_speed_end)
			nav_grid.addWidget(_speed_end_w, 4, 3)
			nav_grid.setColumnStretch(2, 1)
		else:
			# --- Grupo esencial (siempre visible): colormap, play, navegación de
			# gate/slice, zoom, velocidad y smooth.
			nav_grid = QGridLayout()
			nav_grid.setHorizontalSpacing(8)
			nav_grid.setVerticalSpacing(2)
			nav_grid.addWidget(QLabel("Colormap"), 0, 0)
			nav_grid.addWidget(self.cmap_combo, 0, 1)
			nav_grid.addWidget(self.invert_cmap_check, 0, 2)
			nav_grid.addWidget(self.play_button, 0, 3)
			nav_grid.addWidget(self.zoom_reset, 0, 4)
			nav_grid.addWidget(auto_roi_btn_menu, 0, 5)
			nav_grid.addWidget(intestinal_btn_menu, 0, 6)
			nav_grid.addWidget(self.gate_label, 1, 0, _lbl_align)
			nav_grid.addWidget(self.gate_prev_btn, 1, 1, _lbl_align)
			nav_grid.addWidget(self.gate_slider, 1, 2)
			nav_grid.addWidget(self.gate_next_btn, 1, 3)
			nav_grid.addWidget(self.slice_label, 1, 4, _lbl_align)
			nav_grid.addWidget(self.slice_prev_btn, 1, 5, _lbl_align)
			nav_grid.addWidget(self.slice_slider, 1, 6)
			nav_grid.addWidget(self.slice_next_btn, 1, 7)
			nav_grid.addWidget(self.matrix_label, 1, 8)
			nav_grid.addWidget(QLabel("Zoom"), 2, 0, _lbl_align)
			nav_grid.addWidget(self.zoom_prev_btn, 2, 1, _lbl_align)
			nav_grid.addWidget(self.zoom_slider, 2, 2)
			nav_grid.addWidget(self.zoom_next_btn, 2, 3)
			nav_grid.addWidget(self.zoom_label, 2, 4)
			nav_grid.addWidget(QLabel("Speed"), 2, 5, _lbl_align)
			nav_grid.addWidget(self.speed_slider, 2, 6)
			nav_grid.addWidget(self.speed_label, 2, 7)
			nav_grid.addWidget(QLabel("Smooth"), 3, 0, _lbl_align)
			nav_grid.addWidget(self.smooth_prev_btn, 3, 1, _lbl_align)
			nav_grid.addWidget(self.smooth_slider, 3, 2)
			nav_grid.addWidget(self.smooth_next_btn, 3, 3)
			nav_grid.addWidget(self.smooth_label, 3, 4)
			nav_grid.setColumnStretch(2, 1)
			nav_grid.setColumnStretch(6, 1)

		if not self._compact_viewer:
			# Visor completo: dos QSlider verticales (Base/Top) con botones reset.
			self.window_low_slider.setOrientation(Qt.Orientation.Vertical)
			self.window_high_slider.setOrientation(Qt.Orientation.Vertical)
			self.window_low_slider.setMinimumHeight(180)
			self.window_high_slider.setMinimumHeight(180)
			self.window_low_slider.setMaximumWidth(18)
			self.window_high_slider.setMaximumWidth(18)
			for _wsl in (self.window_low_slider, self.window_high_slider):
				_wsl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

		# Botones reset Base/Top (en compacto resetean el RangeSlider, en completo
		# los QSlider). Base a la izquierda, Top a la derecha.
		self.window_low_reset_btn = QPushButton("Base")
		self.window_low_reset_btn.setFlat(True)
		self.window_low_reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
		self.window_low_reset_btn.setToolTip("Volver Base a 0%")
		self.window_low_reset_btn.clicked.connect(self._reset_window_low)
		self.window_high_reset_btn = QPushButton("Top")
		self.window_high_reset_btn.setFlat(True)
		self.window_high_reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
		self.window_high_reset_btn.setToolTip("Volver Top a 100%")
		self.window_high_reset_btn.clicked.connect(self._reset_window_high)
		for _rb in (self.window_low_reset_btn, self.window_high_reset_btn):
			_rb.setStyleSheet("QPushButton{border:none;padding:0;color:#1f2937;background:transparent;} QPushButton:hover{color:#2563eb;}")

		if not self._compact_viewer:
			window_panel = QHBoxLayout()
			window_panel.setSpacing(6)
			window_panel.setContentsMargins(0, 2, 0, 2)
			_base_col = QVBoxLayout()
			_base_col.setSpacing(2)
			_base_col.addWidget(self.window_low_reset_btn, 0, Qt.AlignmentFlag.AlignHCenter)
			_base_col.addWidget(self.window_low_slider, 1)
			_base_col.addWidget(self.window_low_label, 0, Qt.AlignmentFlag.AlignHCenter)
			_top_col = QVBoxLayout()
			_top_col.setSpacing(2)
			_top_col.addWidget(self.window_high_reset_btn, 0, Qt.AlignmentFlag.AlignHCenter)
			_top_col.addWidget(self.window_high_slider, 1)
			_top_col.addWidget(self.window_high_label, 0, Qt.AlignmentFlag.AlignHCenter)
			window_panel.addLayout(_base_col)
			window_panel.addLayout(_top_col)
			self.window_panel_widget = QWidget()
			self.window_panel_widget.setLayout(window_panel)
			self.window_panel_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

		preview_row = QHBoxLayout()
		if self._compact_viewer:
			# En el visor compacto la imagen va SOLA en su fila (los sliders Base/Top
			# se agregan DESPUÉS, entre las dos imágenes, en el layout principal).
			# Así la 1ra y la 2da imagen tienen el mismo ancho y quedan alineadas.
			preview_row.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignTop)
			preview_row.addStretch(1)
		else:
			preview_row.addWidget(self.preview, 1)
			preview_row.addWidget(self.window_panel_widget)

		self.controls_panel = QWidget()
		controls_layout = QVBoxLayout()
		controls_layout.setContentsMargins(0, 0, 0, 0)
		controls_layout.setSpacing(2)
		if self._compact_viewer:
			# Los controles van ABAJO, alineados con la parte baja de la imagen (el
			# usuario los pidió corridos hacia abajo, no pegados arriba). El stretch
			# de arriba los empuja hacia el fondo del panel.
			controls_layout.addStretch(1)
			controls_layout.addLayout(nav_grid)
		else:
			controls_layout.addLayout(nav_grid)
		self.controls_panel.setLayout(controls_layout)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(4, 4, 4, 4)
		layout.setSpacing(2)
		if self._compact_viewer:
			# GRILLA 3x3 REAL (QGridLayout), según lo descripto por el usuario:
			#   fila 0: [ "1ra. Fase" | "Base  Top" | "2da. Fase" ]
			#   fila 1: [ imagen 1ra  | sliders    | imagen 2da  ]  ← alto FIJO 160px
			#   fila 2: [ slice/gate  | 0% / 100%  | slice/gate  ]
			# La fila 1 tiene alto fijo compartido, así las 3 celdas (y por ende las
			# dos imágenes y los sliders) quedan EXACTAMENTE a la misma altura.
			TITLE_H = 20
			IMG_H = 160
			BOTTOM_H = 18

			# --- Celda (0,0): título de fase (1ra en el principal, 2da en el compare) ---
			self.phase_title_label = QLabel("2da. Fase" if self._is_compare else "1ra. Fase")
			self.phase_title_label.setStyleSheet("font-weight:bold; color:#1f2937; padding:1px 2px;")
			self.phase_title_label.setFixedHeight(TITLE_H)

			# Ancho FIJO de la columna de sliders (igual en 1ra y 2da fase), para que
			# ninguna se estire más que la otra ni se solape con la imagen vecina.
			SLIDERS_W = 56

			# --- Celda (0,1): botones Base / Top (reset) ---
			_btn_row = QHBoxLayout()
			_btn_row.setContentsMargins(0, 0, 0, 0)
			_btn_row.setSpacing(6)
			_btn_row.addWidget(self.window_low_reset_btn)
			_btn_row.addWidget(self.window_high_reset_btn)
			_btn_w = QWidget()
			_btn_w.setLayout(_btn_row)
			_btn_w.setFixedHeight(TITLE_H)
			_btn_w.setFixedWidth(SLIDERS_W)

			# --- Celda (1,1): UN SOLO RangeSlider (dos handles Base/Top, 0-200%) ---
			_slider_row = QHBoxLayout()
			_slider_row.setContentsMargins(0, 0, 0, 0)
			_slider_row.setSpacing(0)
			self.range_slider.setFixedHeight(IMG_H)
			_slider_row.addWidget(self.range_slider, 1)
			_slider_w = QWidget()
			_slider_w.setLayout(_slider_row)
			_slider_w.setFixedHeight(IMG_H)
			_slider_w.setFixedWidth(SLIDERS_W)

			# --- Celda (2,1): % Base / Top ---
			_lbl_row = QHBoxLayout()
			_lbl_row.setContentsMargins(0, 0, 0, 0)
			_lbl_row.setSpacing(6)
			_lbl_row.addWidget(self.window_low_label)
			_lbl_row.addWidget(self.window_high_label)
			_lbl_w = QWidget()
			_lbl_w.setLayout(_lbl_row)
			_lbl_w.setFixedHeight(BOTTOM_H)
			_lbl_w.setFixedWidth(SLIDERS_W)

			# --- Celda (2,0): slice/gate de la 1ra imagen ---
			pos_row = QHBoxLayout()
			pos_row.setContentsMargins(2, 0, 0, 0)
			pos_row.addWidget(self.slice_label)
			pos_row.addSpacing(14)
			pos_row.addWidget(self.gate_label)
			pos_row.addStretch(1)
			pos_w = QWidget()
			pos_w.setLayout(pos_row)
			pos_w.setFixedHeight(BOTTOM_H)

			# --- Grilla 3x2 (MISMA para 1ra y 2da etapa) ---
			#   fila 0: [ título      | Base  Top ]
			#   fila 1: [ imagen      | sliders   ]  ← alto FIJO 160px
			#   fila 2: [ slice/gate  | 0% / 100% ]
			# Ambos cines muestran sus sliders Base/Top (pedido del usuario).
			grid = QGridLayout()
			grid.setHorizontalSpacing(8)
			grid.setVerticalSpacing(2)
			grid.setContentsMargins(0, 0, 0, 0)
			grid.addWidget(self.phase_title_label, 0, 0)
			grid.addWidget(_btn_w, 0, 1)
			grid.addWidget(self.preview, 1, 0)   # preview fijo 160x160
			grid.addWidget(_slider_w, 1, 1)
			grid.addWidget(pos_w, 2, 0)
			grid.addWidget(_lbl_w, 2, 1)
			grid.setRowMinimumHeight(1, IMG_H)
			grid.setRowStretch(0, 0)
			grid.setRowStretch(1, 0)
			grid.setRowStretch(2, 0)

			# Fila principal: [grilla 1ra etapa][2da etapa][controles][stretch]
			img_ctrls_row = QHBoxLayout()
			img_ctrls_row.setSpacing(8)
			img_ctrls_row.addLayout(grid, 0)
			if self._is_compare:
				# El compare va incrustado en el CineWidget principal: no necesita
				# sus propios controles ni stretch, solo la grilla limpia.
				self._compare_slot = None
				self._compare_widget = None
				self._sliders_col_widget = _slider_w
				layout.addLayout(img_ctrls_row)
			else:
				# Hueco para la 2da etapa (set_compare_viewer la inserta acá, pos 1).
				self._compare_slot = img_ctrls_row
				self._compare_widget = None
				self._sliders_col_widget = _slider_w
					# Controles con ancho FIJO: no se estiran ni empujan el resto fuera de
				# la pantalla al activar la grilla debug o al maximizar.
				self.controls_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
				self.controls_panel.setFixedWidth(300)
				img_ctrls_row.addWidget(self.controls_panel, 0)
				img_ctrls_row.addStretch(1)
				layout.addStretch(1)
				layout.addLayout(img_ctrls_row)
			self.help_label.setVisible(False)
		else:
			layout.addLayout(preview_row)
			layout.addWidget(self.help_label)
			layout.addWidget(self.controls_panel)

		self.preview.roiChanged.connect(self._on_roi_changed)
		self.preview.zoomChanged.connect(self._on_preview_zoom_changed)
		self.preview.middleClicked.connect(self.activated.emit)
		self.preview.exclusionPolygonEdited.connect(self._on_exclusion_polygon_edited)
		self.preview.centerPicked.connect(self.centerPicked.emit)
		# Visor compacto: alto mínimo ajustado a la imagen fija 160px + fila
		# Slice/Gate (~20px) + márgenes. Así no sobra espacio vacío vertical.
		self.setMinimumHeight(190 if self._compact_viewer else 260)
		self.setSizePolicy(self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())
		self.set_active_highlight(False)
		self._refresh_intestinal_apply_button_text()
		self._refresh_intestinal_mode_widgets()
		self._capture_tooltips()

	def set_compare_viewer(self, widget) -> None:
		"""Inserta el visor de la 2da etapa AL LADO de la grilla de la 1ra (entre
		la grilla y los controles), alineado arriba para que las filas coincidan."""
		if not self._compact_viewer or getattr(self, "_compare_slot", None) is None:
			return
		if self._compare_widget is widget:
			return
		# Posición 1: [grilla 1ra][2da etapa][controles][stretch]
		self._compare_slot.insertWidget(1, widget, 0, Qt.AlignmentFlag.AlignTop)
		self._compare_widget = widget

	def set_phase_title(self, text: str) -> None:
		"""Actualiza el rótulo de la 1ra etapa (ej. 'Esfuerzo' / 'Reposo' / '1ra. Fase')."""
		if getattr(self, "phase_title_label", None) is not None:
			self.phase_title_label.setText(str(text))

	def set_debug_grid(self, enabled: bool) -> None:
		"""Modo debug: dibuja bordes rojos de 1px alrededor de cada contenedor del
		layout compacto (imagen, sliders, controles, compare) para ver la grilla
		real en la interfaz. Temporal, para diagnosticar desalineaciones."""
		if not self._compact_viewer:
			return
		border = "border:1px solid red;" if enabled else ""
		targets = [
			getattr(self, "preview", None),
			getattr(self, "_sliders_col_widget", None),
			getattr(self, "controls_panel", None),
			getattr(self, "_compare_widget", None),
			getattr(self, "phase_title_label", None),
		]
		for w in targets:
			if w is None:
				continue
			# Guardar el stylesheet original una sola vez para poder restaurarlo.
			if not hasattr(w, "_dbg_orig_ss"):
				w._dbg_orig_ss = w.styleSheet()
			orig = getattr(w, "_dbg_orig_ss", "")
			# Al activar: estilo original + borde rojo. Al desactivar: solo el original.
			w.setStyleSheet((orig + border) if enabled else orig)

	def _build_toolbar_button(self, title: str, grids: list, key: str, tooltip: str = "") -> QToolButton:
		"""Agrupa una o más QGridLayout de controles secundarios en una
		`FloatingToolbar` (barra flotante movible, horizontal o vertical) en
		vez de un panel colapsable o un menú.

		El botón ocupa siempre el mismo lugar en el layout; la barra flotante
		es una ventana propia que el usuario puede arrastrar donde le quede
		cómoda, así que abrirla/cerrarla NUNCA cambia el tamaño de la imagen
		(a diferencia de un panel que se expande empujando el resto del
		layout)."""
		toolbar = FloatingToolbar(title, key=key, parent=self)
		for grid in grids:
			toolbar.add_layout(grid)
		btn = QToolButton()
		btn.setText(title)
		btn.clicked.connect(lambda: toolbar.toggle_near(btn))
		if tooltip:
			btn.setToolTip(tooltip)
		return btn

	def set_controls_visible(self, visible: bool):
		self._controls_visible = bool(visible)
		self._refresh_ui_visibility()

	def _refresh_ui_visibility(self):
		show = bool(self._controls_visible)
		# En el visor compacto el help_label NO está en ningún layout: si se hace
		# visible Qt lo abre como VENTANA SEPARADA (la 'ventana fantasma' con el
		# texto de ayuda). Solo se muestra en el visor completo, donde sí está
		# agregado al layout.
		if not self._compact_viewer:
			self.help_label.setVisible(show and bool(self._helpers_visible))
		self.controls_panel.setVisible(show)
		# window_panel_widget solo existe en el visor completo; en el compacto el
		# panel de sliders es _sliders_col_widget (la celda del RangeSlider).
		panel = getattr(self, "window_panel_widget", None) or getattr(self, "_sliders_col_widget", None)
		if panel is not None:
			panel.setVisible(show)

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
		if self._compact_viewer:
			# El visor reducido ya curó su propio set de controles.
			return
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

	def set_intestinal_mode(self, mode: str):
		"""Modo de corrección intestinal: 'attenuate' (porcentual) o 'subtract' (fondo estimado)."""
		value = str(mode or "").strip().lower()
		if value not in ("attenuate", "subtract"):
			value = "attenuate"
		self._intestinal_mode = value
		idx = self.intestinal_mode_combo.findData(value)
		if idx >= 0 and self.intestinal_mode_combo.currentIndex() != idx:
			self.intestinal_mode_combo.blockSignals(True)
			self.intestinal_mode_combo.setCurrentIndex(idx)
			self.intestinal_mode_combo.blockSignals(False)
		self._refresh_intestinal_mode_widgets()

	def intestinal_mode(self) -> str:
		return str(self._intestinal_mode)

	def set_intestinal_background_method(self, method: str):
		"""Cómo se combinan los niveles de las ROI de referencia: 'idw' o 'mean'."""
		value = str(method or "").strip().lower()
		if value not in BACKGROUND_METHODS:
			value = "idw"
		self._intestinal_bg_method = value
		idx = self.intestinal_bg_method_combo.findData(value)
		if idx >= 0 and self.intestinal_bg_method_combo.currentIndex() != idx:
			self.intestinal_bg_method_combo.blockSignals(True)
			self.intestinal_bg_method_combo.setCurrentIndex(idx)
			self.intestinal_bg_method_combo.blockSignals(False)

	def intestinal_background_method(self) -> str:
		return str(self._intestinal_bg_method)

	def _refresh_intestinal_mode_widgets(self):
		subtract = self._intestinal_mode == "subtract"
		self.intestinal_ref_toggle_btn.setEnabled(subtract)
		self.intestinal_ref_clear_btn.setEnabled(subtract)
		self.intestinal_ref_count_label.setEnabled(subtract)
		self.intestinal_bg_method_combo.setEnabled(subtract)
		self.intestinal_preview_btn.setEnabled(subtract)
		self.intestinal_atten_slider.setEnabled(not subtract)
		self.intestinal_atten_label.setEnabled(not subtract)
		if not subtract and self.intestinal_ref_toggle_btn.isChecked():
			self.intestinal_ref_toggle_btn.setChecked(False)
		self._refresh_intestinal_ref_label()

	def _refresh_intestinal_ref_label(self):
		refs = self._intestinal_ref_polygons_for_slice(
			self.current_slice_index(), gate_index=self.current_gate_index()
		)
		n = len(refs)
		self.intestinal_ref_count_label.setText(f"{n} ref.")
		if self._intestinal_mode == "subtract" and n == 0:
			self.intestinal_ref_count_label.setStyleSheet("color:#b45309; font-weight:600;")
		else:
			self.intestinal_ref_count_label.setStyleSheet("color:#0e7490;")

	def _intestinal_ref_polygons_for_slice(
		self, slice_index: int, gate_index: int | None = None
	) -> list[list[tuple[float, float]]]:
		sl = int(slice_index)
		if self._intestinal_scope_mode == "gate_slices":
			g = int(self.current_gate_index() if gate_index is None else gate_index)
			polys = self._intestinal_ref_polygons_by_gate.get((g, sl))
			if polys:
				return polys
			for (gg, _ss), any_polys in self._intestinal_ref_polygons_by_gate.items():
				if int(gg) == g and any_polys:
					return any_polys
			return []
		return list(self._intestinal_ref_polygons.get(sl, []))

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

		ref_slice_polygons = []
		for slice_index, polygons in sorted((self._intestinal_ref_polygons or {}).items()):
			for polygon in polygons or []:
				pts = [[float(cy), float(cx)] for cy, cx in (polygon or [])]
				if len(pts) >= 3:
					ref_slice_polygons.append({"slice": int(slice_index), "points": pts})

		ref_gate_polygons = []
		for (gate_index, slice_index), polygons in sorted((self._intestinal_ref_polygons_by_gate or {}).items()):
			for polygon in polygons or []:
				pts = [[float(cy), float(cx)] for cy, cx in (polygon or [])]
				if len(pts) >= 3:
					ref_gate_polygons.append({"gate": int(gate_index), "slice": int(slice_index), "points": pts})

		return {
			"slice_polygons": slice_polygons,
			"gate_polygons": gate_polygons,
			"reference_slice_polygons": ref_slice_polygons,
			"reference_gate_polygons": ref_gate_polygons,
			"mode": str(self._intestinal_mode),
			"background_method": str(self._intestinal_bg_method),
		}

	def set_intestinal_roi_state(self, state: dict | None):
		"""Restaura polígonos de ROI intestinal desde un preset."""
		self._intestinal_roi_polygons = {}
		self._intestinal_roi_polygons_by_gate = {}
		self._intestinal_ref_polygons = {}
		self._intestinal_ref_polygons_by_gate = {}
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
			for item in state.get("reference_slice_polygons", []) or []:
				try:
					slice_index = int(item.get("slice"))
					points = [tuple(float(v) for v in pt[:2]) for pt in (item.get("points") or [])]
				except Exception:
					continue
				if len(points) >= 3:
					self._intestinal_ref_polygons.setdefault(slice_index, []).append(points)
			for item in state.get("reference_gate_polygons", []) or []:
				try:
					gate_index = int(item.get("gate"))
					slice_index = int(item.get("slice"))
					points = [tuple(float(v) for v in pt[:2]) for pt in (item.get("points") or [])]
				except Exception:
					continue
				if len(points) >= 3:
					self._intestinal_ref_polygons_by_gate.setdefault((gate_index, slice_index), []).append(points)
			if state.get("mode"):
				self.set_intestinal_mode(str(state.get("mode")))
			if state.get("background_method"):
				self.set_intestinal_background_method(str(state.get("background_method")))
		self.preview.set_exclusion_polygon(self._intestinal_polygon_for_slice(self.current_slice_index(), gate_index=self.current_gate_index()))
		self.preview.set_reference_polygons(
			self._intestinal_ref_polygons_for_slice(self.current_slice_index(), gate_index=self.current_gate_index())
		)
		self._refresh_intestinal_ref_label()
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

	def roi_for_gate_slice(self, gate_index: int, slice_index: int):
		"""ROI efectivo para un gate/slice.

		Prioridad: ROI manual por gate (si existe) > ROI común por slice.
		"""
		key = (int(gate_index), int(slice_index))
		if key in self._rois_by_gate:
			return self._rois_by_gate[key]
		return self._rois.get(int(slice_index))

	def per_gate_roi_mode_enabled(self) -> bool:
		return bool(self._per_gate_roi_mode)

	def set_per_gate_roi_mode(self, enabled: bool):
		self._per_gate_roi_mode = bool(enabled)
		if self.per_gate_roi_check.isChecked() != self._per_gate_roi_mode:
			self.per_gate_roi_check.blockSignals(True)
			self.per_gate_roi_check.setChecked(self._per_gate_roi_mode)
			self.per_gate_roi_check.blockSignals(False)
		self._update_view()

	def _on_per_gate_roi_mode_toggled(self, checked: bool):
		self.set_per_gate_roi_mode(bool(checked))
		if self._per_gate_roi_mode:
			self.help_label.setText(
				"QC por gate activo: la edición de ROI afecta SOLO al gate actual. "
				"clic = centro | Shift = radio externo | Ctrl = radio interno | clic der = borrar ROI del gate."
			)
		else:
			self.help_label.setText(
				"Mouse: clic izq = centro | Shift+clic = radio externo | Ctrl+clic = radio interno | clic der = borrar ROI | "
				"apex/base sin cavidad: usar 'Borrar internos'"
			)

	def gate_roi_state(self) -> dict[str, object]:
		"""Estado serializable de los ROIs manuales por gate."""
		items = []
		for (gate_index, slice_index), roi in sorted(self._rois_by_gate.items()):
			items.append({
				"gate": int(gate_index),
				"slice": int(slice_index),
				"roi": [float(v) for v in roi],
				"source": str(self._roi_by_gate_source.get((gate_index, slice_index), "manual")),
			})
		return {
			"per_gate_mode": bool(self._per_gate_roi_mode),
			"gate_rois": items,
		}

	def set_gate_roi_state(self, state: dict | None):
		"""Restaura ROIs por gate desde un preset."""
		self._rois_by_gate = {}
		self._roi_by_gate_source = {}
		per_gate_mode = False
		if isinstance(state, dict):
			per_gate_mode = bool(state.get("per_gate_mode", False))
			for item in state.get("gate_rois", []) or []:
				try:
					g = int(item.get("gate"))
					s = int(item.get("slice"))
					roi = tuple(float(v) for v in (item.get("roi") or [])[:4])
				except Exception:
					continue
				if len(roi) == 4:
					self._rois_by_gate[(g, s)] = roi
					self._roi_by_gate_source[(g, s)] = str(item.get("source", "manual"))
		self.set_per_gate_roi_mode(per_gate_mode)
		self._update_view()

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

	def intestinal_target_weights(self, shape: tuple[int, int], n_slices: int, gate_index: int | None = None) -> dict[int, np.ndarray]:
		"""Mapas de peso (ROI a corregir + feather) por corte, para el motor de sustracción."""
		g = self.current_gate_index() if gate_index is None else int(gate_index)
		out: dict[int, np.ndarray] = {}
		for s in range(int(n_slices)):
			poly = self._intestinal_polygon_for_slice(s, gate_index=g)
			if not poly:
				continue
			soft = self._soft_mask_from_polygon(shape, poly)
			if np.any(soft > 0):
				out[int(s)] = soft
		return out

	def intestinal_reference_masks(self, shape: tuple[int, int], n_slices: int, gate_index: int | None = None) -> dict[int, list[np.ndarray]]:
		"""Máscaras booleanas de las ROI de referencia por corte."""
		g = self.current_gate_index() if gate_index is None else int(gate_index)
		out: dict[int, list[np.ndarray]] = {}
		for s in range(int(n_slices)):
			polys = self._intestinal_ref_polygons_for_slice(s, gate_index=g)
			masks = [self._polygon_to_mask(shape, p) for p in polys]
			masks = [m for m in masks if np.any(m)]
			if masks:
				out[int(s)] = masks
		return out

	def has_intestinal_references(self) -> bool:
		return bool(self._intestinal_ref_polygons or self._intestinal_ref_polygons_by_gate)

	def _subtract_intestinal_background(self, img: np.ndarray, slice_index: int, gate_index: int | None = None) -> np.ndarray:
		"""Vista previa de la sustracción sobre un único frame.

		Ojo: acá el fondo se estima sobre el frame mostrado, no sobre el promedio
		de gates. Es una aproximación **solo para el preview visual**. El cálculo
		que alimenta segmentación y fase se hace en `MainWindow.process_current`
		con `core.intestinal_subtraction.apply_intestinal_subtraction`, que estima
		una única vez sobre el promedio de gates. Estimar por gate en el análisis
		metería variación temporal artificial y contaminaría la fase.
		"""
		g = self.current_gate_index() if gate_index is None else int(gate_index)
		poly = self._intestinal_polygon_for_slice(int(slice_index), gate_index=g)
		if not poly:
			return img
		refs = self._intestinal_ref_polygons_for_slice(int(slice_index), gate_index=g)
		ref_masks = [self._polygon_to_mask(img.shape, p) for p in refs]
		ref_masks = [m for m in ref_masks if np.any(m)]
		if not ref_masks:
			return img
		weight = self._soft_mask_from_polygon(img.shape, poly)
		if float(np.max(weight)) <= 0.0:
			return img
		background, info = estimate_background_map(img, ref_masks, method=self._intestinal_bg_method)
		if not info["applicable"]:
			return img
		return np.clip(img - background * weight, 0.0, None)

	def _fixed_scale_pixmap(self, data: np.ndarray, vmin: float, vmax: float, size: int, cmap_name: str | None = None) -> QPixmap:
		"""Pixmap con escala de grises/color **fijada**, para comparar dos imágenes.

		`_array_to_pixmap` renormaliza cada frame a su propio min/max, lo que haría
		que 'antes' y 'después' se vean iguales aunque cambien las cuentas. Acá la
		escala se impone desde afuera.
		"""
		arr = np.asarray(data, dtype=np.float64)
		span = max(1e-8, float(vmax) - float(vmin))
		norm = np.clip((arr - float(vmin)) / span, 0.0, 1.0)
		if cmap_name is None:
			name = str(self.cmap_combo.currentText())
			if self.invert_cmap_check.isChecked():
				name = f"{name}_r"
		else:
			name = str(cmap_name)
		cmap = _resolve_cmap(name)
		rgb8 = (np.asarray(cmap(norm)[..., :3], dtype=np.float32) * 255.0).astype(np.uint8)
		h, w, _ = rgb8.shape
		qimg = QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
		pix = QPixmap.fromImage(qimg.copy())
		return pix.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

	def _intestinal_preview_panel(
		self,
		data: np.ndarray,
		vmin: float,
		vmax: float,
		size: int,
		target_poly,
		ref_polys,
		cmap_name: str | None = None,
	) -> QPixmap:
		"""Panel del diálogo antes/después, con los contornos superpuestos."""
		scaled = self._fixed_scale_pixmap(data, vmin, vmax, size, cmap_name=cmap_name)
		canvas = QPixmap(size, size)
		canvas.fill(QColor("#020617"))
		painter = QPainter(canvas)
		x0 = int((size - scaled.width()) / 2)
		y0 = int((size - scaled.height()) / 2)
		painter.drawPixmap(x0, y0, scaled)
		try:
			h, w = int(np.asarray(data).shape[0]), int(np.asarray(data).shape[1])
			sx = float(scaled.width()) / max(1.0, float(w))
			sy = float(scaled.height()) / max(1.0, float(h))
			painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

			def _draw(points, color: str, style):
				if not points or len(points) < 2:
					return
				painter.setPen(QPen(QColor(color), 2, style))
				# Los polígonos se almacenan como (fila, columna) = (y, x), igual
				# que en el visor principal (_polygon_to_widget). Hay que mapear la
				# columna al eje horizontal y la fila al vertical; invertirlo produce
				# una reflexión sobre la diagonal que se ve como una rotación de 90°.
				poly = QPolygonF([QPointF(x0 + float(px) * sx, y0 + float(py) * sy) for py, px in points])
				painter.drawPolygon(poly)

			for rp in (ref_polys or []):
				_draw(rp, "#22d3ee", Qt.PenStyle.DotLine)
			_draw(target_poly, "#ff4dd2", Qt.PenStyle.SolidLine)
		except Exception:
			pass
		painter.end()
		return canvas

	def _open_intestinal_preview_dialog(self):
		"""Compara el corte actual antes y después de la sustracción de fondo.

		Usa el motor real (`apply_intestinal_subtraction`), que estima el fondo
		una sola vez sobre el promedio de gates. Es decir, lo que se ve acá es
		exactamente lo que va a alimentar segmentación y análisis de fase.
		"""
		if self._cube is None or np.asarray(self._cube).ndim != 4:
			QMessageBox.information(self, "Antes/después", "Cargá un estudio gated primero.")
			return
		if self._intestinal_mode != "subtract":
			QMessageBox.information(self, "Antes/después", "Esta vista aplica al modo 'Restar fondo'.")
			return

		cube = np.asarray(self._cube, dtype=np.float64)
		n_gates, n_slices, h, w = cube.shape
		sl = int(self.slice_slider.value())
		if sl < 0 or sl >= n_slices:
			return
		g = int(self.current_gate_index())
		g = max(0, min(n_gates - 1, g))

		weights = self.intestinal_target_weights((h, w), n_slices, gate_index=g)
		refs = self.intestinal_reference_masks((h, w), n_slices, gate_index=g)
		if sl not in weights:
			QMessageBox.information(
				self,
				"Antes/después",
				"Dibujá primero la ROI de la zona a corregir (la que se superpone con el miocardio) en este corte.",
			)
			return
		if sl not in refs:
			QMessageBox.information(
				self,
				"Antes/después",
				"Dibujá al menos una ROI de referencia sobre el asa donde se ve limpia.\n"
				"Con dos o más, el nivel de fondo sale de combinarlas según el método elegido.",
			)
			return

		sub_cube = cube[:, sl : sl + 1, :, :]
		corrected, info = apply_intestinal_subtraction(
			sub_cube,
			{0: weights[sl]},
			{0: refs[sl]},
			method=self._intestinal_bg_method,
		)
		detail = dict(info.get("per_slice", {}).get(0, {}))
		if not detail.get("applied"):
			QMessageBox.information(self, "Antes/después", str(detail.get("message") or "No se pudo estimar el fondo."))
			return

		before = cube[g, sl]
		after = np.asarray(corrected)[g, 0]
		removed = np.clip(before - after, 0.0, None)
		vmin = 0.0
		vmax = float(np.max(before)) if np.isfinite(np.max(before)) else 1.0
		target_poly = self._intestinal_polygon_for_slice(sl, gate_index=g)
		ref_polys = self._intestinal_ref_polygons_for_slice(sl, gate_index=g)

		dialog = QDialog(self)
		dialog.setWindowTitle(f"Sustracción de fondo — Slice {sl + 1}, gate {g + 1}")
		dialog.setModal(True)
		dialog.resize(1040, 660)
		root = QVBoxLayout(dialog)
		header = QLabel(
			"Comparación con el cálculo real: el fondo se estima una sola vez sobre el promedio de gates "
			"y se resta igual a todos. Contorno magenta = zona corregida; punteado cian = referencias."
		)
		header.setWordWrap(True)
		root.addWidget(header)

		grid = QGridLayout()
		grid.setHorizontalSpacing(8)
		grid.setVerticalSpacing(8)
		panels = [
			("Antes", before, vmin, vmax, None),
			("Después", after, vmin, vmax, None),
			("Restado", removed, 0.0, max(1e-8, float(np.max(removed))), "inferno"),
		]
		for col, (title_txt, data, lo, hi, cmap_name) in enumerate(panels):
			card = QWidget()
			card_layout = QVBoxLayout(card)
			card_layout.setContentsMargins(6, 6, 6, 6)
			card_layout.setSpacing(4)
			card.setStyleSheet("background:#0f172a; border:1px solid #334155; border-radius:6px;")
			title = QLabel(title_txt)
			title.setStyleSheet("color:#e2e8f0; font-weight:600;")
			card_layout.addWidget(title)
			img_label = QLabel()
			img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
			img_label.setPixmap(
				self._intestinal_preview_panel(data, lo, hi, 280, target_poly, ref_polys, cmap_name=cmap_name)
			)
			card_layout.addWidget(img_label)
			scale_txt = (
				f"escala 0–{hi:.0f} cuentas" if cmap_name else f"escala común 0–{vmax:.0f} cuentas"
			)
			sub = QLabel(scale_txt)
			sub.setStyleSheet("color:#94a3b8;")
			card_layout.addWidget(sub)
			grid.addWidget(card, 0, col)
		root.addLayout(grid)

		levels = detail.get("levels") or []
		levels_txt = ", ".join(f"{float(lv):.1f}" for lv in levels) or "—"
		method_txt = "media simple" if str(detail.get("method")) == "mean" else "interpolado (IDW)"
		bg_level = detail.get("background_level")
		bg_txt = f"{float(bg_level):.1f} cuentas/gate (constante)" if bg_level is not None else "variable en la zona"
		amp_before = float(detail.get("rel_amp_before", float("nan")))
		amp_after = float(detail.get("rel_amp_after", float("nan")))
		amp_txt = (
			f"{amp_before:.3f} → {amp_after:.3f}"
			if np.isfinite(amp_before) and np.isfinite(amp_after)
			else "no evaluable"
		)
		lines = [
			f"Referencias: {int(detail.get('n_references', 0))} (niveles: {levels_txt} cuentas/gate)",
			f"Método de fondo: {method_txt} → {bg_txt}",
			f"Cuentas restadas: {float(detail.get('counts_subtracted', 0.0)):,.0f} "
			f"({float(detail.get('subtracted_pct', 0.0)):.1f}% del corte)",
			f"Píxeles llevados a cero: {100.0 * float(detail.get('clipped_fraction', 0.0)):.1f}%",
			f"Amplitud relativa del 1er armónico en la zona: {amp_txt}",
		]
		metrics = QLabel("\n".join(lines))
		metrics.setStyleSheet("color:#cbd5e1;")
		metrics.setToolTip(
			"La amplitud relativa debería SUBIR al restar: el fondo intestinal es continuo (DC) y "
			"diluye la modulación del miocardio. Si baja, la ROI o las referencias están mal puestas."
		)
		root.addWidget(metrics)

		warn_bits = []
		if detail.get("oversubtracted"):
			warn_bits.append(
				"Sobresustracción: demasiados píxeles quedaron en cero. Bajá el nivel de fondo usando "
				"referencias más representativas o achicá la zona a corregir."
			)
		if np.isfinite(amp_before) and np.isfinite(amp_after) and amp_after < amp_before:
			warn_bits.append(
				"La amplitud relativa bajó tras restar. Revisá que las referencias estén sobre intestino "
				"limpio y no sobre miocardio."
			)
		if warn_bits:
			warn = QLabel("⚠ " + "  ".join(warn_bits))
			warn.setWordWrap(True)
			warn.setStyleSheet("color:#b45309; font-weight:600;")
			root.addWidget(warn)

		disclaimer = QLabel(
			"Esta operación resta un fondo aditivo. NO es corrección de atenuación: la atenuación es un factor "
			"multiplicativo y solo se compensa dividiendo por un mapa de atenuación (TC, fuente de transmisión o "
			"método de Chang). Restar fondo puede incluso acentuar el defecto inferior por atenuación. "
			"Ver docs/FUNDAMENTO_MATEMATICO_SUSTRACCION_FONDO.md"
		)
		disclaimer.setWordWrap(True)
		disclaimer.setStyleSheet("color:#94a3b8; font-style:italic;")
		root.addWidget(disclaimer)

		close_btn = QPushButton("Cerrar")
		close_btn.clicked.connect(dialog.accept)
		root.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
		dialog.exec()

	def _attenuate_image_with_intestinal_roi(self, img: np.ndarray, slice_index: int, gate_index: int | None = None) -> np.ndarray:
		img = np.asarray(img, dtype=np.float64)
		if not self._intestinal_apply_enabled:
			return img
		if self._intestinal_mode == "subtract":
			return self._subtract_intestinal_background(img, slice_index, gate_index)
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

	def set_center_pick_mode(self, enabled: bool):
		"""Activa/desactiva el modo de fijar centro de cavidad por clic."""
		self.preview.set_center_pick_mode(bool(enabled))

	def set_manual_centers(self, centers: dict[int, tuple[float, float]] | None):
		"""Propaga los centros manuales al label para dibujarlos por corte."""
		self.preview.set_manual_centers(centers)

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
			self.play_button.setText("⏸" if self._compact_viewer else "⏸ Pausar")
		else:
			self._timer.stop()
			self.play_button.setText("▶" if self._compact_viewer else "▶ Reproducir")
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
		roi = None
		if self._per_gate_roi_mode:
			roi = self.roi_for_gate_slice(gate, sl)
		else:
			roi = self._rois.get(sl)
		if roi is not None and not self.show_auto_roi_check.isChecked():
			src = self._roi_by_gate_source.get((gate, sl)) if self._per_gate_roi_mode else self._roi_source.get(sl)
			if src == "auto":
				roi = None
		self.preview.set_roi(roi)
		self.preview.set_exclusion_polygon(self._intestinal_polygon_for_slice(sl, gate_index=gate))
		self.preview.set_reference_polygons(self._intestinal_ref_polygons_for_slice(sl, gate_index=gate))
		self._refresh_intestinal_ref_label()
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

	def _on_range_window_change(self, low: int, high: int):
		"""RangeSlider de dos handles: base (low) y top (high) en 0-200%."""
		self._window_low = float(low) / 100.0
		self._window_high = float(high) / 100.0
		self._update_view()

	def _reset_window_low(self):
		"""Reset Base a 0% (RangeSlider en compacto, QSlider en completo)."""
		if self.range_slider is not None:
			low, high = self.range_slider.values()
			self.range_slider.set_values(0, high)
		else:
			self.window_low_slider.setValue(0)

	def _reset_window_high(self):
		"""Reset Top a 100% (RangeSlider en compacto, QSlider en completo)."""
		if self.range_slider is not None:
			low, high = self.range_slider.values()
			self.range_slider.set_values(low, 100)
		else:
			self.window_high_slider.setValue(100)

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

		if self.range_slider is not None:
			self.range_slider.set_values(base, top)
		else:
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
		if self._per_gate_roi_mode:
			g = int(self.gate_slider.value())
			key = (g, sl)
			self._rois_by_gate[key] = roi
			self._roi_by_gate_source[key] = "auto"
			self.roiEditedGate.emit(g, sl, roi)
		else:
			self._rois[sl] = roi
			self._roi_source[sl] = "auto"
			self.roiEdited.emit(sl, roi)
		self._update_view()

	def _auto_roi_all_slices(self):
		if self._cube is None:
			return
		empty_only = self.auto_roi_empty_only_check.isChecked()
		n_slices = int(self._cube.shape[1])
		if self._per_gate_roi_mode:
			g = int(self.gate_slider.value())
			for sl in range(n_slices):
				key = (g, sl)
				existing_roi = self._rois_by_gate.get(key)
				source = self._roi_by_gate_source.get(key)
				# "solo vacíos" protege ROIs manuales del gate, pero permite regenerar auto.
				if empty_only and existing_roi is not None and source != "auto":
					continue
				roi = self.estimate_auto_roi_for_slice(sl)
				if roi is None:
					continue
				self._rois_by_gate[key] = roi
				self._roi_by_gate_source[key] = "auto"
				self.roiEditedGate.emit(g, sl, roi)
		else:
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
			"7) Inferior superpuesto: suprime focos calientes periféricos inferiores (hígado/intestino) y luego detecta VI.\n"
			"8) Cavidad dominante: prioriza cavidad central hipocaptante para centrar mejor el ROI en FEVI/sincronía.\n\n"
			"ROI intestino irregular:\n"
			"• Activá 'ROI intestino' y dibujá polígono (doble clic para cerrar).\n"
			"• Ajustá Atenuar % y Feather para bajar cuentas con borde suave.\n\n"
			"Tip clínico: en 22x22, usar primero Robusto u Hot bowel y validar con Comparar ROI."
		)
		QMessageBox.information(self, "SINCRO - Help Auto ROI", msg)

	def set_auto_roi_method(self, method: str):
		key = str(method or "").strip().lower()
		if key not in ("robusto", "clasico", "gradiente", "hotbowel", "percentil_central", "consenso", "inferior_overlap", "cavidad_dominante"):
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
		elif key == "cavidad_dominante":
			self.auto_roi_method_label.setText("Cavidad dominante")
		else:
			self.auto_roi_method_label.setText("Robusto")

	def auto_roi_method(self) -> str:
		return str(self._auto_roi_method)

	def set_refine_cavity_center(self, enabled: bool):
		self._refine_cavity_center = bool(enabled)

	def refine_cavity_center(self) -> bool:
		return bool(self._refine_cavity_center)

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
		if m == "cavidad_dominante":
			return "Cavidad dominante"
		return "Robusto"

	def _auto_roi_from_image_cavidad_dominante(self, img: np.ndarray, low_res: bool):
		"""Prioriza casos con cavidad central hipocaptante (anillo intenso + centro frío)."""
		img = np.asarray(img, dtype=np.float64)
		base = self._auto_roi_from_image_robusto(img, low_res)
		if base is None:
			return None
		cy, cx, ri, ro = (float(v) for v in base)
		h, w = img.shape
		ys, xs = np.ogrid[:h, :w]
		d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

		# Buscar centro cavitario dentro de un disco limitado alrededor del VI.
		inner_disk = d <= max(2.0, 0.74 * ro)
		if int(np.count_nonzero(inner_disk)) < 12:
			return base
		finite = img[np.isfinite(img)]
		if finite.size < 12:
			return base
		p70 = float(np.percentile(finite, 70.0 if low_res else 66.0))
		inv = np.clip(p70 - img, 0.0, None)
		# Prior gaussiano alrededor del centro previo para no irse al fondo negro.
		sig = max(1.6, 0.34 * ro)
		prior = np.exp(-0.5 * (d / max(1e-6, sig)) ** 2)
		wgt = inv * prior
		wgt = np.where(inner_disk, wgt, 0.0)
		sw = float(np.sum(wgt))
		if sw > 1e-8:
			cy_c = float(np.sum(wgt * ys) / sw)
			cx_c = float(np.sum(wgt * xs) / sw)
			# Mezcla robusta: favorece cavidad pero no rompe continuidad entre slices.
			cy = 0.35 * cy + 0.65 * cy_c
			cx = 0.35 * cx + 0.65 * cx_c

		# Ajuste leve de radios para anillo con cavidad bien marcada.
		ro = max(ro, 3.0)
		ri_target = 0.46 * ro
		ri = max(0.0, 0.80 * ri + 0.20 * ri_target)
		if ro <= ri:
			ro = ri + 1.0
		return (float(cy), float(cx), float(ri), float(ro))

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
		if method_key == "cavidad_dominante":
			roi = self._auto_roi_from_image_cavidad_dominante(img, low_res)
		elif method_key == "clasico":
			roi = self._auto_roi_from_image_clasico(img, low_res)
		elif method_key == "gradiente":
			roi = self._auto_roi_from_image_gradiente(img, low_res)
		elif method_key == "hotbowel":
			roi = self._auto_roi_from_image_hotbowel(img, low_res)
		elif method_key == "percentil_central":
			roi = self._auto_roi_from_image_percentil_central(img, low_res)
		elif method_key == "consenso":
			roi = self._auto_roi_from_image_consenso(img, low_res)
		elif method_key == "inferior_overlap":
			roi = self._auto_roi_from_image_inferior_overlap(img, low_res)
		else:
			roi = self._auto_roi_from_image_robusto(img, low_res)
		return self._refine_roi_center(img, roi, low_res=low_res)

	def _refine_roi_center(self, img: np.ndarray, roi, *, low_res: bool):
		"""Corre el centro del ROI del centroide del músculo al de la cavidad.

		Salvo "Cavidad dominante", todos los métodos derivan su centro de
		``center_of_mass`` de la máscara de miocardio, que se sesga hacia el sector
		de mayor captación. Se aplica solo con el refinamiento activado, para que
		el comportamiento histórico siga disponible y sea comparable.
		"""
		if not self._refine_cavity_center or roi is None or len(roi) != 4:
			return roi
		cy, cx, ri, ro = (float(v) for v in roi)
		if not (np.isfinite(cy) and np.isfinite(cx) and np.isfinite(ro)) or ro <= 0.0:
			return roi
		new_cy, new_cx = refine_center_to_cavity(cy, cx, ro, img=img, low_res=low_res)
		return (new_cy, new_cx, ri, ro)

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

		# Nuevo término: premiar cavidad hipocaptante en el centro del anillo.
		cavity = d <= max(0.6, float(ri))
		if int(np.count_nonzero(cavity)) >= 6:
			cavity_mean = float(np.mean(img[cavity]))
		else:
			cavity_mean = ring_mean
		cavity_contrast = max(0.0, ring_mean - cavity_mean)

		score = 2.2 * contrast - 0.95 * center_pen - 0.55 * area_pen - 1.25 * inferior_hot_pen + 1.15 * cavity_contrast

		# Penalizar ROIs que se alejan demasiado cuando hay cavidad bien definida.
		if cavity_contrast > 0.08 and center_pen > 0.62:
			score -= 0.9
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

		method_order = ["robusto", "clasico", "gradiente", "hotbowel", "percentil_central", "consenso", "inferior_overlap", "cavidad_dominante"]
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
				"Inferior superpuesto: reduce impacto de focos inferiores extracardíacos.\n"
				"Cavidad dominante: empuja el centro hacia cavidad hipocaptante (útil cuando anillo rodea un centro frío)."
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
		sl = int(slice_index)
		if self._per_gate_roi_mode:
			g = int(self.gate_slider.value())
			key = (g, sl)
			if roi is None:
				self._rois_by_gate.pop(key, None)
				self._roi_by_gate_source.pop(key, None)
			else:
				self._rois_by_gate[key] = tuple(float(v) for v in roi)
				self._roi_by_gate_source[key] = "manual"
			self.roiEditedGate.emit(g, sl, roi)
		else:
			if roi is None:
				self._rois.pop(sl, None)
				self._roi_source.pop(sl, None)
			else:
				self._rois[sl] = tuple(float(v) for v in roi)
				self._roi_source[sl] = "manual"
			self.roiEdited.emit(sl, roi)

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
		if enabled and self.intestinal_ref_toggle_btn.isChecked():
			self.intestinal_ref_toggle_btn.setChecked(False)
		self._intestinal_draw_role = "target" if enabled else self._intestinal_draw_role
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

	def _on_intestinal_mode_changed(self, _index: int):
		mode = self.intestinal_mode_combo.currentData()
		self._intestinal_mode = str(mode or "attenuate")
		self._refresh_intestinal_mode_widgets()
		self._update_view()

	def _on_intestinal_bg_method_changed(self, _index: int):
		method = self.intestinal_bg_method_combo.currentData()
		self._intestinal_bg_method = str(method or "idw")
		self._update_view()

	def _on_intestinal_ref_draw_toggled(self, checked: bool):
		enabled = bool(checked)
		if enabled and self.intestinal_roi_toggle_btn.isChecked():
			self.intestinal_roi_toggle_btn.setChecked(False)
		self._intestinal_draw_role = "reference" if enabled else "target"
		self.preview.set_exclusion_draw_mode(enabled)
		if enabled:
			self.help_label.setText(
				"Modo ROI REFERENCIA activo: dibujá sobre el asa intestinal donde se ve LIMPIA (sin miocardio encima). "
				"Ideal: una a la entrada y otra a la salida del asa. Doble clic cierra el polígono."
			)
		else:
			self.help_label.setText(
				"Mouse: clic izq = centro | Shift+clic = radio externo | Ctrl+clic = radio interno | clic der = borrar ROI | "
				"apex/base sin cavidad: usar 'Borrar internos'"
			)

	def _store_reference_polygon(self, slice_index: int, gate_index: int, points: list[tuple[float, float]]):
		"""Agrega una ROI de referencia respetando el alcance elegido."""
		if self._intestinal_scope_mode == "all_slices" and self._cube is not None:
			for i in range(int(self._cube.shape[1])):
				self._intestinal_ref_polygons.setdefault(int(i), []).append(list(points))
		elif self._intestinal_scope_mode == "gate_slices":
			if self._cube is not None:
				for i in range(int(self._cube.shape[1])):
					self._intestinal_ref_polygons_by_gate.setdefault((gate_index, int(i)), []).append(list(points))
			else:
				self._intestinal_ref_polygons_by_gate.setdefault((gate_index, slice_index), []).append(list(points))
		else:
			self._intestinal_ref_polygons.setdefault(slice_index, []).append(list(points))

	def _clear_intestinal_references_current_slice(self):
		sl = int(self.slice_slider.value())
		if self._intestinal_scope_mode == "all_slices":
			self._intestinal_ref_polygons = {}
		elif self._intestinal_scope_mode == "gate_slices":
			gate = int(self.current_gate_index())
			self._intestinal_ref_polygons_by_gate = {
				(k, s): p for (k, s), p in self._intestinal_ref_polygons_by_gate.items() if int(k) != gate
			}
		else:
			self._intestinal_ref_polygons.pop(sl, None)
		self.preview.set_reference_polygons([])
		self._refresh_intestinal_ref_label()
		self._update_view()

	def _on_exclusion_polygon_edited(self, slice_index: int, polygon):
		sl = int(slice_index)
		gate = int(self.current_gate_index())

		if self._intestinal_draw_role == "reference":
			# El clic derecho (polygon=None) borra las referencias del alcance actual.
			if polygon is None:
				self._clear_intestinal_references_current_slice()
				return
			pts = [tuple(map(float, p)) for p in (polygon or [])]
			if len(pts) >= 3:
				self._store_reference_polygon(sl, gate, pts)
			# El label de exclusión no debe quedarse con la referencia dibujada:
			# restauramos el polígono real de la zona a corregir.
			self.preview.set_exclusion_polygon(self._intestinal_polygon_for_slice(sl, gate_index=gate))
			self.preview.set_reference_polygons(self._intestinal_ref_polygons_for_slice(sl, gate_index=gate))
			self._refresh_intestinal_ref_label()
			self._update_view()
			return

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

	def _open_gate_qc_dialog(self):
		"""Abre ventana modal con mosaico de todos los gates del slice actual para QC/ROI por gate."""
		if self._cube is None or self._cube.ndim != 4:
			QMessageBox.information(self, "QC gates", "Cargá un estudio gated primero.")
			return
		sl = int(self.slice_slider.value())
		n_slices = int(self._cube.shape[1])
		if sl < 0 or sl >= n_slices:
			QMessageBox.information(self, "QC gates", "Slice actual fuera de rango.")
			return
		dlg = GateQcDialog(self, slice_index=sl)
		dlg.exec()
		# Al cerrar, refrescar la vista principal por si se editaron ROIs.
		self._update_view()


class GateQcDialog(QDialog):
	"""Diálogo modal para QC/ROI por gate: muestra todos los gates del slice en mosaico."""

	def __init__(self, cine_widget: CineWidget, slice_index: int, parent=None):
		super().__init__(parent or cine_widget)
		self._cine = cine_widget
		self._slice_index = int(slice_index)
		self.setWindowTitle(f"QC por gate — Slice {self._slice_index + 1}")
		self.setModal(True)
		self.resize(900, 600)

		layout = QVBoxLayout(self)

		# Info
		info = QLabel(
			f"Slice {self._slice_index + 1} | "
			"Click en un gate para seleccionarlo | "
			"Edición en el panel derecho afecta SOLO al gate seleccionado"
		)
		info.setStyleSheet("color:#666; font-size:10pt; padding:4px;")
		info.setWordWrap(True)
		layout.addWidget(info)

		# Layout principal: mosaico + panel de edición
		main_row = QHBoxLayout()

		# Mosaico de gates (izquierda)
		self.montage = GateMontageLabel()
		self.montage.set_cube(self._cine._cube)
		self.montage.set_slice_index(self._slice_index)
		self.montage.set_current_gate(self._cine.current_gate_index())
		self.montage.set_rois(
			self._cine._rois,
			self._cine._rois_by_gate,
			self._cine._per_gate_roi_mode,
		)
		self.montage.set_display_params(
			self._cine.cmap_combo.currentText(),
			self._cine._smooth_sigma,
			self._cine.invert_cmap_check.isChecked(),
			self._cine._window_low,
			self._cine._window_high,
		)
		self.montage.gateSelected.connect(self._on_gate_selected)
		main_row.addWidget(self.montage, 2)

		# Panel de edición (derecha): preview grande + controles
		edit_panel = QWidget()
		edit_layout = QVBoxLayout(edit_panel)

		self.edit_preview = RoiImageLabel()
		self.edit_preview.setMinimumSize(280, 280)
		self.edit_preview.roiChanged.connect(self._on_edit_roi_changed)
		edit_layout.addWidget(QLabel("Gate seleccionado:"))
		self.gate_label = QLabel("Gate 1")
		self.gate_label.setStyleSheet("font-weight:bold; font-size:12pt; color:#d61f1f;")
		edit_layout.addWidget(self.gate_label)
		edit_layout.addWidget(self.edit_preview)

		# Botones
		btn_row = QHBoxLayout()
		self.apply_all_btn = QPushButton("Aplicar a todos gates")
		self.apply_all_btn.setToolTip("Copia el ROI del gate actual a todos los gates de este slice.")
		self.apply_all_btn.clicked.connect(self._apply_roi_to_all_gates)
		self.clear_btn = QPushButton("Borrar ROI gate")
		self.clear_btn.clicked.connect(self._clear_current_gate_roi)
		btn_row.addWidget(self.apply_all_btn)
		btn_row.addWidget(self.clear_btn)
		edit_layout.addLayout(btn_row)

		main_row.addWidget(edit_panel, 1)
		layout.addLayout(main_row)

		# Botón cerrar
		close_btn = QPushButton("Cerrar")
		close_btn.clicked.connect(self.accept)
		layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

		# Inicializar con gate actual del cine
		self._selected_gate = self._cine.current_gate_index()
		self._refresh_edit_preview()

	def _on_gate_selected(self, gate_index: int):
		self._selected_gate = int(gate_index)
		self.montage.set_current_gate(self._selected_gate)
		self._refresh_edit_preview()

	def _refresh_edit_preview(self):
		if self._cine._cube is None:
			return
		frame = np.asarray(self._cine._cube[self._selected_gate, self._slice_index], dtype=np.float64)
		if self._cine._intestinal_apply_enabled:
			frame = self._cine._attenuate_image_with_intestinal_roi(frame, self._slice_index)
		self.edit_preview.set_slice_index(self._slice_index)
		self.edit_preview.set_frame(
			frame,
			cmap_name=str(self._cine.cmap_combo.currentText()),
			smooth_sigma=self._cine._smooth_sigma,
			invert_cmap=self._cine.invert_cmap_check.isChecked(),
			window_low=self._cine._window_low,
			window_high=self._cine._window_high,
		)
		roi = self._cine.roi_for_gate_slice(self._selected_gate, self._slice_index)
		self.edit_preview.set_roi(roi)
		self.gate_label.setText(f"Gate {self._selected_gate + 1}")
		# Log temporal para depuración
		print(f"[QC-DEBUG] Refresh preview: gate={self._selected_gate + 1}, roi={roi}")

	def _on_edit_roi_changed(self, slice_index: int, roi):
		"""Edita el ROI del gate seleccionado."""
		key = (self._selected_gate, self._slice_index)
		if roi is None:
			self._cine._rois_by_gate.pop(key, None)
			self._cine._roi_by_gate_source.pop(key, None)
		else:
			self._cine._rois_by_gate[key] = tuple(float(v) for v in roi)
			self._cine._roi_by_gate_source[key] = "manual"
		# Activar modo por gate si no estaba
		if not self._cine._per_gate_roi_mode:
			self._cine.set_per_gate_roi_mode(True)
		self._cine.roiEditedGate.emit(self._selected_gate, self._slice_index, roi)
		# Refrescar mosaico y preview
		self.montage.set_rois(
			self._cine._rois,
			self._cine._rois_by_gate,
			self._cine._per_gate_roi_mode,
		)
		self.montage.update()
		self.edit_preview.update()
		# Log temporal para depuración
		print(f"[QC-DEBUG] Gate {self._selected_gate + 1} slice {self._slice_index + 1}: roi={roi}")

	def _apply_roi_to_all_gates(self):
		roi = self._cine.roi_for_gate_slice(self._selected_gate, self._slice_index)
		if roi is None:
			return
		n_gates = int(self._cine._cube.shape[0]) if self._cine._cube is not None else 0
		for g in range(n_gates):
			key = (g, self._slice_index)
			self._cine._rois_by_gate[key] = tuple(roi)
			self._cine._roi_by_gate_source[key] = "manual"
			self._cine.roiEditedGate.emit(g, self._slice_index, roi)
		self.montage.set_rois(
			self._cine._rois,
			self._cine._rois_by_gate,
			self._cine._per_gate_roi_mode,
		)
		self.montage.update()

	def _clear_current_gate_roi(self):
		key = (self._selected_gate, self._slice_index)
		self._cine._rois_by_gate.pop(key, None)
		self._cine._roi_by_gate_source.pop(key, None)
		self._cine.roiEditedGate.emit(self._selected_gate, self._slice_index, None)
		self.edit_preview.set_roi(None)
		self.montage.set_rois(
			self._cine._rois,
			self._cine._rois_by_gate,
			self._cine._per_gate_roi_mode,
		)
		self.montage.update()
