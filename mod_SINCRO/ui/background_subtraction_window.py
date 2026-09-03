"""SINCRO - ui.background_subtraction_window

Popup mínimo para la **sustracción de fondo MANUAL con ROI** sobre el crudo.

Conserva la única parte útil de la vieja ventana de "Preparación / Reconstrucción"
(que se eliminó por ser redundante con la toolbar de PROCESAMIENTO): dibujar un
polígono de fondo sobre el MIP crudo y restar el piso de cuentas medido ahí, con
opción de método (constante / localizado) e impacto (solo visual / toda la cadena).

El motor que realmente alimenta la reconstrucción vive en ``main_window``
(``_raw_bg_spec`` / ``set_raw_background_subtraction`` / ``_apply_raw_bg_to_recon_cube``);
aquí solo está la UI de dibujo del ROI + los controles. Es una ventana no modal,
en vivo/en memoria, con una etapa a la vez (selector Esfuerzo/Reposo).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPixmap, QPen, QPolygonF
from PyQt6.QtWidgets import (
	QComboBox,
	QDialog,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QSizePolicy,
	QVBoxLayout,
	QWidget,
)

try:
	from scipy import ndimage as ndi
except Exception:  # pragma: no cover
	ndi = None

try:
	import matplotlib as _mpl
	import matplotlib.cm as _cm

	try:
		from core.col_registry import register_all_colormaps

		register_all_colormaps()
	except Exception:  # pragma: no cover - registro opcional
		pass
	_MPL_OK = True
except Exception:  # pragma: no cover
	_mpl = None
	_cm = None
	_MPL_OK = False


def _colormap(name: str = "odyssey_cool"):
	if not _MPL_OK:
		return None
	for candidate in (name, "hot", "inferno", "gray"):
		# matplotlib ≥3.9 removió cm.get_cmap; usar el registro nuevo con fallback.
		try:
			return _mpl.colormaps[candidate]
		except Exception:
			try:
				return _cm.get_cmap(candidate)
			except Exception:
				continue
	return None


class MipView(QWidget):
	"""Cine de proyecciones crudas en vivo (QLabel + QImage) con dibujo de ROI."""

	def __init__(self, parent=None):
		super().__init__(parent)
		self._frames: list[np.ndarray] = []
		self._rgba_cache: list[np.ndarray | None] = []
		self._idx = 0
		self._disp_max = 1.0
		self._cmap = _colormap()
		self._playing = True
		self._kind = ""

		# --- Dibujo de ROI de fondo / corazón sobre el crudo ---
		self._draw_mode: str | None = None          # 'background' | 'heart' | None
		self._bg_polygon: list[tuple[float, float]] = []
		self._heart_polygon: list[tuple[float, float]] = []
		self._draft_polygon: list[tuple[float, float]] = []
		self._disp_geom: dict | None = None

		lay = QVBoxLayout(self)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.setSpacing(2)
		self._image_label = QLabel("Sin estudio")
		self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self._image_label.setMinimumSize(240, 240)
		self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		self._image_label.setStyleSheet("background:#0b1220; color:#94a3b8; border:1px solid #1e293b;")
		self._image_label.mousePressEvent = self._on_label_click  # type: ignore[assignment]
		lay.addWidget(self._image_label, 1)
		self._status_label = QLabel("")
		self._status_label.setStyleSheet("color:#94a3b8; font-size:11px;")
		self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		lay.addWidget(self._status_label)

		self._timer = QTimer(self)
		self._timer.timeout.connect(self._advance)
		self._timer.start(66)  # ~15 fps

	def _toggle_play(self, _event):
		self._playing = not self._playing

	# ---------------------------------------- Dibujo de ROI de fondo
	def set_draw_mode(self, mode: str | None):
		"""Activa el modo de dibujo ('background' | 'heart') o lo desactiva (None).

		Al entrar en modo dibujo se pausa la rotación para marcar sobre un frame
		fijo. El polígono en curso se descarta.
		"""
		self._draw_mode = mode
		self._draft_polygon = []
		if mode is not None:
			self._playing = False
		self._render_current()

	def clear_polygons(self):
		self._bg_polygon = []
		self._heart_polygon = []
		self._draft_polygon = []
		self._render_current()

	def background_polygon(self) -> list[tuple[float, float]]:
		return list(self._bg_polygon)

	def heart_polygon(self) -> list[tuple[float, float]]:
		return list(self._heart_polygon)

	def frames_stack(self) -> np.ndarray | None:
		"""Stack (N, H, W) de los frames actuales, o None."""
		if not self._frames:
			return None
		return np.stack([np.asarray(f, dtype=np.float64) for f in self._frames])

	def kind(self) -> str:
		return self._kind

	def _label_to_image(self, x: float, y: float) -> tuple[float, float] | None:
		"""Mapea coords del label (clic) a coords de píxel de la imagen."""
		geom = self._disp_geom
		if not geom:
			return None
		ix = (x - geom["off_x"]) / geom["scale_x"]
		iy = (y - geom["off_y"]) / geom["scale_y"]
		if ix < 0 or iy < 0 or ix > geom["img_w"] or iy > geom["img_h"]:
			return None
		return (ix, iy)

	def _on_label_click(self, event):
		if self._draw_mode is None:
			self._toggle_play(event)
			return
		btn = event.button()
		if btn == Qt.MouseButton.RightButton:
			self._commit_draft()
			return
		pos = event.position() if hasattr(event, "position") else event.pos()
		mapped = self._label_to_image(float(pos.x()), float(pos.y()))
		if mapped is not None:
			self._draft_polygon.append(mapped)
			self._render_current()

	def _commit_draft(self):
		if len(self._draft_polygon) >= 3:
			if self._draw_mode == "background":
				self._bg_polygon = list(self._draft_polygon)
			elif self._draw_mode == "heart":
				self._heart_polygon = list(self._draft_polygon)
		self._draft_polygon = []
		self._draw_mode = None
		self._render_current()

	def set_source(self, kind: str, array: np.ndarray, status: str = ""):
		"""kind='proj' (proyecciones crudas (A,H,W)); otros kinds no dibujan ROI."""
		self._kind = str(kind)
		self._idx = 0
		frames: list[np.ndarray] = []
		if array is not None:
			arr = np.asarray(array, dtype=np.float64)
			if kind == "proj" and arr.ndim == 3:
				frames = [arr[i] for i in range(arr.shape[0])]
		self._frames = frames
		self._rgba_cache = [None] * len(frames)
		if frames:
			stack = np.stack(frames)
			finite = stack[np.isfinite(stack)]
			self._disp_max = float(np.percentile(finite, 99.5)) if finite.size else 1.0
			if self._disp_max <= 0:
				self._disp_max = 1.0
			self._playing = True
		else:
			self._image_label.setText("Sin datos para MIP")
		self._status_label.setText(status)
		self._render_current()

	def clear(self, message: str = "Sin estudio"):
		self._frames = []
		self._rgba_cache = []
		self._kind = ""
		self._image_label.setText(message)
		self._status_label.setText("")

	def _advance(self):
		if not self._frames or not self._playing:
			return
		self._idx = (self._idx + 1) % len(self._frames)
		self._render_current()

	def _rgba_for(self, i: int) -> np.ndarray:
		cached = self._rgba_cache[i]
		if cached is not None:
			return cached
		frame = self._frames[i]
		norm = np.clip(frame / self._disp_max, 0.0, 1.0)
		if self._cmap is not None:
			rgba = (self._cmap(norm) * 255).astype(np.uint8)
		else:
			g = (norm * 255).astype(np.uint8)
			rgba = np.dstack([g, g, g, np.full_like(g, 255)])
		rgba = np.ascontiguousarray(rgba)
		self._rgba_cache[i] = rgba
		return rgba

	def _render_current(self):
		if not self._frames:
			return
		rgba = self._rgba_for(self._idx)
		h, w = rgba.shape[0], rgba.shape[1]
		qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
		pix = QPixmap.fromImage(qimg)
		self._last_rgba = rgba  # mantener el buffer vivo
		scaled = pix.scaled(
			self._image_label.size(),
			Qt.AspectRatioMode.KeepAspectRatio,
			Qt.TransformationMode.SmoothTransformation,
		)
		sw, sh = scaled.width(), scaled.height()
		label_size = self._image_label.size()
		off_x = max(0, (label_size.width() - sw) / 2.0)
		off_y = max(0, (label_size.height() - sh) / 2.0)
		self._disp_geom = {
			"scale_x": sw / float(w) if w else 1.0,
			"scale_y": sh / float(h) if h else 1.0,
			"off_x": off_x,
			"off_y": off_y,
			"img_w": w,
			"img_h": h,
		}
		if self._bg_polygon or self._heart_polygon or self._draft_polygon:
			self._paint_polygons(scaled)
		self._image_label.setPixmap(scaled)

	def _paint_polygons(self, pix: QPixmap):
		"""Dibuja las ROI de fondo/corazón sobre el pixmap escalado."""
		geom = self._disp_geom or {}
		sx = geom.get("scale_x", 1.0)
		sy = geom.get("scale_y", 1.0)

		def _to_qpoly(points):
			poly = QPolygonF()
			for px, py in points:
				poly.append(QPointF(px * sx, py * sy))
			return poly

		painter = QPainter(pix)
		try:
			painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
			# ROI de fondo (cian).
			if len(self._bg_polygon) >= 3:
				painter.setPen(QPen(QColor("#22d3ee"), 2))
				painter.setBrush(QColor(34, 211, 238, 40))
				painter.drawPolygon(_to_qpoly(self._bg_polygon))
			# ROI del corazón (naranja).
			if len(self._heart_polygon) >= 3:
				painter.setPen(QPen(QColor("#fb923c"), 2))
				painter.setBrush(QColor(251, 146, 60, 30))
				painter.drawPolygon(_to_qpoly(self._heart_polygon))
			# Polígono en curso (amarillo punteado + vértices).
			if self._draft_polygon:
				pen = QPen(QColor("#facc15"), 2)
				pen.setStyle(Qt.PenStyle.DashLine)
				painter.setPen(pen)
				painter.setBrush(Qt.BrushStyle.NoBrush)
				if len(self._draft_polygon) >= 2:
					painter.drawPolyline(_to_qpoly(self._draft_polygon))
				painter.setPen(QPen(QColor("#facc15"), 1))
				painter.setBrush(QColor("#facc15"))
				for px, py in self._draft_polygon:
					painter.drawEllipse(QPointF(px * sx, py * sy), 2.5, 2.5)
		finally:
			painter.end()

	def resizeEvent(self, event):  # noqa: N802 (Qt override)
		super().resizeEvent(event)
		self._render_current()


class BackgroundSubtractionWindow(QDialog):
	"""Ventana no modal para la sustracción de fondo manual (ROI) sobre el crudo."""

	def __init__(self, main_window):
		super().__init__(main_window)
		self._main = main_window
		self.setWindowTitle("Sustracción de fondo (ROI) · crudo")
		self.setModal(False)
		self.resize(560, 700)
		# Base cruda y spec por etapa (para poder alternar etapas sin perder estado).
		self._bg_base: dict[str, np.ndarray | None] = {"stress": None, "rest": None}
		self._bg_spec: dict[str, dict | None] = {"stress": None, "rest": None}
		self._proj_status: dict[str, str] = {"stress": "", "rest": ""}

		root = QVBoxLayout(self)
		root.setContentsMargins(8, 8, 8, 8)
		root.setSpacing(6)

		header = QLabel(
			"Sustracción de fondo manual sobre el crudo (proyecciones que rotan). Dibujá la zona "
			"SIN corazón para medir el piso de cuentas; opcionalmente el corazón para el modo localizado. "
			"Clic en la imagen para pausar/reanudar la rotación. Hacela ANTES de reconstruir."
		)
		header.setWordWrap(True)
		header.setStyleSheet("color:#94a3b8;")
		root.addWidget(header)

		stage_row = QHBoxLayout()
		stage_row.setSpacing(4)
		stage_row.addWidget(QLabel("Etapa:"))
		self.stage_combo = QComboBox()
		self.stage_combo.currentIndexChanged.connect(self._on_stage_changed)
		stage_row.addWidget(self.stage_combo, 1)
		root.addLayout(stage_row)

		self.mip = MipView(self)
		root.addWidget(self.mip, 1)

		method_row = QHBoxLayout()
		method_row.setSpacing(4)
		method_row.addWidget(QLabel("Modo:"))
		self.bg_method_combo = QComboBox()
		self.bg_method_combo.addItem("Constante (ROI fondo)", "constant")
		self.bg_method_combo.addItem("Localizado (corazón + fondo)", "localized")
		self.bg_method_combo.currentIndexChanged.connect(lambda *_: self._refresh_bg_widgets())
		method_row.addWidget(self.bg_method_combo, 1)
		root.addLayout(method_row)

		impact_row = QHBoxLayout()
		impact_row.setSpacing(4)
		impact_row.addWidget(QLabel("Impacto:"))
		self.bg_impact_combo = QComboBox()
		self.bg_impact_combo.addItem("Solo visual", "visual")
		self.bg_impact_combo.addItem("Toda la cadena", "chain")
		self.bg_impact_combo.setToolTip(
			"Solo visual: aclara el MIP crudo, no cambia lo que se reconstruye.\n"
			"Toda la cadena: la resta alimenta la reconstrucción de esta etapa."
		)
		impact_row.addWidget(self.bg_impact_combo, 1)
		root.addLayout(impact_row)

		roi_row = QHBoxLayout()
		roi_row.setSpacing(4)
		self.bg_roi_btn = QPushButton("ROI fondo")
		self.bg_roi_btn.setToolTip("Dibujá el polígono de fondo (zona SIN corazón). Clic derecho para cerrar.")
		self.bg_roi_btn.clicked.connect(lambda: self._bg_draw("background"))
		self.heart_roi_btn = QPushButton("ROI corazón")
		self.heart_roi_btn.setToolTip("Solo en modo localizado: marcá el corazón. La resta se limita a esa zona.")
		self.heart_roi_btn.clicked.connect(lambda: self._bg_draw("heart"))
		roi_row.addWidget(self.bg_roi_btn)
		roi_row.addWidget(self.heart_roi_btn)
		root.addLayout(roi_row)

		action_row = QHBoxLayout()
		action_row.setSpacing(4)
		self.bg_apply_btn = QPushButton("Aplicar")
		self.bg_apply_btn.clicked.connect(self._bg_apply)
		self.bg_clear_btn = QPushButton("Limpiar")
		self.bg_clear_btn.clicked.connect(self._bg_clear)
		action_row.addWidget(self.bg_apply_btn)
		action_row.addWidget(self.bg_clear_btn)
		root.addLayout(action_row)

		self.bg_status_label = QLabel("")
		self.bg_status_label.setWordWrap(True)
		self.bg_status_label.setStyleSheet("color:#94a3b8; font-size:10px;")
		root.addWidget(self.bg_status_label)

		bottom = QHBoxLayout()
		self.refresh_btn = QPushButton("↻ Actualizar")
		self.refresh_btn.clicked.connect(self.refresh)
		bottom.addWidget(self.refresh_btn)
		bottom.addStretch(1)
		self.close_btn = QPushButton("Cerrar")
		self.close_btn.clicked.connect(self.close)
		bottom.addWidget(self.close_btn)
		root.addLayout(bottom)

	# ------------------------------------------------------------------ etapas
	def _available_stages(self) -> list[tuple[str, str]]:
		out: list[tuple[str, str]] = []
		for stage, label in (("stress", "Esfuerzo"), ("rest", "Reposo")):
			if self._main._prep_mip_source_for_stage(stage) is not None:
				out.append((stage, label))
		return out

	def _current_stage(self) -> str | None:
		data = self.stage_combo.currentData()
		return str(data) if data else None

	def _on_stage_changed(self, *_a):
		self.mip.clear_polygons()
		self._load_stage_mip()

	def refresh(self):
		"""Reconstruye el selector de etapas y recarga el MIP de la etapa activa."""
		stages = self._available_stages()
		cur = self._current_stage()
		self.stage_combo.blockSignals(True)
		self.stage_combo.clear()
		for stage, label in stages:
			self.stage_combo.addItem(label, stage)
		if cur:
			idx = self.stage_combo.findData(cur)
			if idx >= 0:
				self.stage_combo.setCurrentIndex(idx)
		self.stage_combo.blockSignals(False)
		self._load_stage_mip()

	def _load_stage_mip(self):
		stage = self._current_stage()
		if stage is None:
			self.mip.clear("Sin estudio crudo")
			self._set_controls_enabled(False)
			return
		src = self._main._prep_mip_source_for_stage(stage)
		if src is None:
			self.mip.clear("Sin estudio para esta etapa")
			self._set_controls_enabled(False)
			return
		kind, arr, status = src
		if kind != "proj":
			self.mip.clear(
				"Etapa ya reconstruida.\nLa sustracción de fondo se hace sobre el crudo,\nantes de reconstruir."
			)
			self.set_bg_status("La sustracción aplica sobre el crudo. Hacela antes de reconstruir.")
			self._set_controls_enabled(False)
			return
		self._proj_status[stage] = status
		base = self._bg_base.get(stage)
		self.mip.set_source("proj", base if base is not None else arr, status)
		self._set_controls_enabled(True)
		self._refresh_bg_widgets()

	# ------------------------------------------------------------- UI helpers
	def _set_controls_enabled(self, enabled: bool):
		for w in (self.bg_method_combo, self.bg_impact_combo, self.bg_roi_btn, self.bg_apply_btn, self.bg_clear_btn):
			w.setEnabled(enabled)
		self._refresh_bg_widgets()

	def _refresh_bg_widgets(self):
		localized = self._bg_method() == "localized"
		self.heart_roi_btn.setEnabled(localized and self.mip.kind() == "proj")

	def _bg_method(self) -> str:
		return str(self.bg_method_combo.currentData() or "constant")

	def _bg_impact(self) -> str:
		return str(self.bg_impact_combo.currentData() or "visual")

	def set_bg_status(self, text: str):
		self.bg_status_label.setText(text or "")

	# ------------------------------------------------------ sustracción de fondo
	def _bg_draw(self, mode: str):
		if self.mip.kind() != "proj":
			self.set_bg_status("El fondo se dibuja sobre el crudo (proyecciones que rotan).")
			return
		self.mip.set_draw_mode(mode)
		etiqueta = "fondo (zona SIN corazón)" if mode == "background" else "corazón"
		self.set_bg_status(f"Marcá la ROI de {etiqueta}. Clic para agregar vértices, clic derecho para cerrar.")

	def _bg_clear(self):
		stage = self._current_stage()
		self.mip.clear_polygons()
		if stage is not None:
			base = self._bg_base.get(stage)
			if base is not None:
				self.mip.set_source("proj", base, self._proj_status.get(stage, ""))
			self._bg_base[stage] = None
			self._bg_spec[stage] = None
			if hasattr(self._main, "clear_raw_background_subtraction"):
				try:
					self._main.clear_raw_background_subtraction(stage)
				except Exception:
					pass
		self.set_bg_status("Fondo limpiado.")

	def _bg_apply(self):
		from core.raw_background import (
			measure_background_level,
			polygon_mask,
			subtract_constant,
			subtract_localized,
		)

		stage = self._current_stage()
		if stage is None:
			return
		mip = self.mip
		if mip.kind() != "proj":
			self.set_bg_status("La sustracción aplica sobre el crudo. Hacela antes de reconstruir.")
			return
		base = self._bg_base.get(stage)
		if base is None:
			base = mip.frames_stack()
			if base is None:
				self.set_bg_status("Sin imagen cruda para procesar.")
				return
			self._bg_base[stage] = base
		bg_poly = mip.background_polygon()
		if len(bg_poly) < 3:
			self.set_bg_status("Dibujá primero la ROI de fondo (clic derecho para cerrar).")
			return
		h, w = int(base.shape[-2]), int(base.shape[-1])
		bg_mask = polygon_mask((h, w), bg_poly)
		mean_img = base.mean(axis=0)
		level = measure_background_level(mean_img, bg_mask, stat="median")
		method = self._bg_method()
		if method == "localized":
			heart_poly = mip.heart_polygon()
			if len(heart_poly) < 3:
				self.set_bg_status("Modo localizado: falta dibujar la ROI del corazón.")
				return
			heart_mask = polygon_mask((h, w), heart_poly)
			res = subtract_localized(base, level, heart_mask, feather_px=2.0)
		else:
			res = subtract_constant(base, level)
		subtracted = np.asarray(res.image, dtype=np.float64)
		mip.set_source("proj", subtracted, self._proj_status.get(stage, ""))
		msg = f"Fondo {method}: nivel {level:.1f} · clip {res.clipped_fraction * 100:.0f}%"
		if res.notes:
			msg += "  ⚠ " + res.notes[0]
		impact = self._bg_impact()
		self._bg_spec[stage] = {
			"method": method,
			"level": float(level),
			"impact": impact,
			"bg_polygon": list(bg_poly),
			"heart_polygon": list(mip.heart_polygon()),
			"shape": (h, w),
		}
		if impact == "chain" and hasattr(self._main, "set_raw_background_subtraction"):
			try:
				self._main.set_raw_background_subtraction(stage, subtracted, self._bg_spec[stage])
				msg += " · alimentará la reconstrucción"
			except Exception as exc:  # pragma: no cover - defensivo
				msg += f" · [chain error: {exc}]"
		self.set_bg_status(msg)
