"""SINCRO - ui.preparacion_window

Ventana de **preparación / reconstrucción** de las dos etapas (Esfuerzo / Reposo).

Vive aparte de la ventana principal a propósito: separa el mundo de la
*preparación* (crudo → corrección de movimiento → reconstrucción → reorientación)
del mundo de *análisis/resultados* (fase, métricas, polar, informe), tal como lo
separan Xeleris / ECTb. Es una ventana **no modal**: se deja abierta al costado
mientras se trabaja, y trabaja **en vivo, en memoria** como la de asincronía o el
diálogo de reorientación.

Cada columna (Esfuerzo / Reposo) muestra un **MIP rotatorio en vivo** del estudio:
- Crudo (sin reconstruir): las proyecciones rotan alrededor del paciente (la
  "vista giratoria" natural de la gamma cámara).
- Reconstruido (o cortes SA cargados): MIP rotatorio del volumen (estilo del MIP
  que se ve en el diálogo de reorientación), calculado en memoria.

Los botones de cada columna delegan en el flujo ya probado de la ventana
principal (fijando primero la etapa activa), así se reutiliza toda la lógica de
corrección de movimiento / reconstrucción / reorientación sin duplicarla.
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
	QSplitter,
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


def _rotating_mip_frames(vol: np.ndarray, n_frames: int = 24, max_dim: int = 112) -> list[np.ndarray]:
	"""MIP rotatorio de un volumen 3D (Z, Y, X) girando alrededor del eje vertical."""
	vol = np.asarray(vol, dtype=np.float64)
	if vol.ndim != 3 or vol.size == 0:
		return []
	# Downsample para que el giro sea instantáneo aunque el volumen sea grande.
	factor = max(1, int(np.ceil(max(vol.shape[1], vol.shape[2]) / float(max_dim))))
	if factor > 1:
		vol = vol[:, ::factor, ::factor]
	frames: list[np.ndarray] = []
	for ang in np.linspace(0.0, 360.0, n_frames, endpoint=False):
		if ndi is not None and abs(ang) > 1e-6:
			rot = ndi.rotate(vol, float(ang), axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0)
		else:
			rot = vol
		frames.append(np.asarray(rot.max(axis=1), dtype=np.float64))  # MIP en profundidad → (Z, X)
	return frames


class MipView(QWidget):
	"""Cine de MIP rotatorio en vivo (QLabel + QImage, sin matplotlib canvas)."""

	def __init__(self, parent=None):
		super().__init__(parent)
		self._frames: list[np.ndarray] = []
		self._rgba_cache: list[np.ndarray | None] = []
		self._idx = 0
		self._disp_max = 1.0
		self._cmap = _colormap()
		self._playing = True
		self._kind = ""
		self._overlay_text = ""
		self._overlay_color = "#facc15"

		# --- Dibujo de ROI/VOI de fondo sobre la imagen cruda (realce visual) ---
		self._draw_mode: str | None = None          # 'background' | 'heart' | None
		self._bg_polygon: list[tuple[float, float]] = []      # coords de imagen (x, y)
		self._heart_polygon: list[tuple[float, float]] = []
		self._draft_polygon: list[tuple[float, float]] = []
		self._disp_geom: dict | None = None         # geometría de escalado para mapear clics

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

	# ---------------------------------------- Dibujo de ROI/VOI de fondo
	def set_draw_mode(self, mode: str | None):
		"""Activa el modo de dibujo ('background' | 'heart') o lo desactiva (None).

		Al entrar en modo dibujo se pausa la rotación para poder marcar sobre un
		frame fijo. El polígono en curso se descarta.
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

	def current_frame(self) -> np.ndarray | None:
		"""Frame actualmente mostrado (para medir/mostrar), en cuentas."""
		if not self._frames:
			return None
		return np.asarray(self._frames[self._idx], dtype=np.float64)

	def mean_frame(self) -> np.ndarray | None:
		"""Media de todos los frames (fondo ~independiente del ángulo)."""
		if not self._frames:
			return None
		return np.asarray(np.mean(np.stack(self._frames), axis=0), dtype=np.float64)

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
			# Cierra el polígono en curso.
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


	def set_overlay_label(self, text: str, color: str = "#facc15"):
		"""Rótulo persistente (etapa) dibujado sobre el MIP."""
		self._overlay_text = str(text or "")
		self._overlay_color = color or "#facc15"
		self._render_current()

	def set_source(self, kind: str, array: np.ndarray, status: str = ""):
		"""kind='proj' (proyecciones crudas (A,H,W)) o 'vol' (volumen 3D (Z,Y,X))."""
		self._kind = str(kind)
		self._idx = 0
		frames: list[np.ndarray] = []
		if array is not None:
			arr = np.asarray(array, dtype=np.float64)
			if kind == "proj" and arr.ndim == 3:
				frames = [arr[i] for i in range(arr.shape[0])]
			elif kind == "vol":
				frames = _rotating_mip_frames(arr)
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
		# Geometría de escalado (para mapear clics del label a píxeles de imagen).
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
		if self._overlay_text:
			self._paint_overlay_label(scaled)
		self._image_label.setPixmap(scaled)

	def _paint_polygons(self, pix: QPixmap):
		"""Dibuja las ROI/VOI de fondo sobre el pixmap escalado."""
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

	def _paint_overlay_label(self, pix: QPixmap):
		"""Dibuja un badge redondeado semitransparente con el nombre de la etapa."""
		painter = QPainter(pix)
		try:
			painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
			font = QFont()
			font.setBold(True)
			font.setPointSize(11)
			painter.setFont(font)
			metrics = painter.fontMetrics()
			text = self._overlay_text
			tw = metrics.horizontalAdvance(text)
			th = metrics.height()
			pad_x, pad_y = 8, 4
			rect = QRectF(6, 6, tw + 2 * pad_x, th + 2 * pad_y)
			painter.setPen(Qt.PenStyle.NoPen)
			painter.setBrush(QColor(0, 0, 0, 150))
			painter.drawRoundedRect(rect, 6, 6)
			painter.setPen(QColor(self._overlay_color))
			painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)
		finally:
			painter.end()

	def resizeEvent(self, event):  # noqa: N802 (Qt override)
		super().resizeEvent(event)
		self._render_current()


class _StageColumn(QWidget):
	"""Columna de una etapa: título + MIP en vivo + botonera que delega al main."""

	def __init__(self, window: "PreparacionWindow", stage: str, title: str, parent=None):
		super().__init__(parent)
		self._window = window
		self._stage = stage
		lay = QVBoxLayout(self)
		lay.setContentsMargins(6, 6, 6, 6)
		lay.setSpacing(6)

		self.title_label = QLabel(title)
		self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.title_label.setStyleSheet("font-weight:bold; font-size:14px; color:#e2e8f0;")
		lay.addWidget(self.title_label)
		self._title = title

		self.mip = MipView(self)
		self.mip.set_overlay_label(title, "#ff9a5a" if stage == "stress" else "#5ad1ff")
		lay.addWidget(self.mip, 1)

		# Barra de movimiento (método + eje) — se corrige EN VIVO en esta ventana.
		motion_row = QHBoxLayout()
		motion_row.setSpacing(4)
		motion_row.addWidget(QLabel("Mov:"))
		self.method_combo = QComboBox()
		self.method_combo.addItems(
			["Auto", "Sinusoide", "XCorr", "GammaSync", "Stasis", "Hopkins", "Odyssey", "COM", "Threshold"]
		)
		self.method_combo.setCurrentText("GammaSync")
		self.axis_combo = QComboBox()
		self.axis_combo.addItems(["Y", "X", "XY"])
		motion_row.addWidget(self.method_combo, 1)
		motion_row.addWidget(self.axis_combo)
		lay.addLayout(motion_row)

		btn_row = QHBoxLayout()
		btn_row.setSpacing(4)
		self.motion_btn = QPushButton("Movimiento")
		self.motion_btn.setToolTip("Corrige el movimiento de esta etapa y muestra el cine corregido acá.")
		self.motion_btn.clicked.connect(lambda: self._window._delegate(self._stage, "motion"))
		self.recon_btn = QPushButton("Reconstruir")
		self.recon_btn.setToolTip("Reconstrucción SPECT desde crudo de esta etapa.")
		self.recon_btn.clicked.connect(lambda: self._window._delegate(self._stage, "recon"))
		self.reorient_btn = QPushButton("Reorientar")
		self.reorient_btn.setToolTip("Reorientación Rec/Ref de esta etapa (diálogo rápido con MIP).")
		self.reorient_btn.clicked.connect(lambda: self._window._delegate(self._stage, "reorient"))
		for b in (self.motion_btn, self.recon_btn, self.reorient_btn):
			btn_row.addWidget(b)
		lay.addLayout(btn_row)

		# --- Sustracción de fondo sobre el crudo (realce visual para orientar) ---
		self.bg_toggle_btn = QPushButton("Fondo ▾")
		self.bg_toggle_btn.setCheckable(True)
		self.bg_toggle_btn.setToolTip(
			"Herramienta de sustracción de fondo sobre la imagen cruda.\n"
			"Resta un piso de cuentas medido en una zona SIN corazón para levantar el\n"
			"contraste del miocardio y orientarlo mejor. No es corrección de atenuación."
		)
		self.bg_toggle_btn.toggled.connect(self._on_bg_toggle)
		lay.addWidget(self.bg_toggle_btn)

		self.bg_panel = QWidget()
		bg_lay = QVBoxLayout(self.bg_panel)
		bg_lay.setContentsMargins(2, 2, 2, 2)
		bg_lay.setSpacing(3)

		bg_row1 = QHBoxLayout()
		bg_row1.setSpacing(4)
		bg_row1.addWidget(QLabel("Modo:"))
		self.bg_method_combo = QComboBox()
		self.bg_method_combo.addItem("Constante (ROI fondo)", "constant")
		self.bg_method_combo.addItem("Localizado (corazón + fondo)", "localized")
		self.bg_method_combo.currentIndexChanged.connect(self._on_bg_method_changed)
		bg_row1.addWidget(self.bg_method_combo, 1)
		bg_lay.addLayout(bg_row1)

		bg_row2 = QHBoxLayout()
		bg_row2.setSpacing(4)
		bg_row2.addWidget(QLabel("Impacto:"))
		self.bg_impact_combo = QComboBox()
		self.bg_impact_combo.addItem("Solo visual", "visual")
		self.bg_impact_combo.addItem("Toda la cadena", "chain")
		self.bg_impact_combo.setToolTip(
			"Solo visual: aclara el MIP crudo, no cambia lo que se reconstruye.\n"
			"Toda la cadena: la resta alimenta la reconstrucción/orientación."
		)
		bg_row2.addWidget(self.bg_impact_combo, 1)
		bg_lay.addLayout(bg_row2)

		bg_row3 = QHBoxLayout()
		bg_row3.setSpacing(4)
		self.bg_roi_btn = QPushButton("ROI fondo")
		self.bg_roi_btn.setToolTip("Dibujá el polígono de fondo (zona SIN corazón). Clic derecho para cerrar.")
		self.bg_roi_btn.clicked.connect(lambda: self._window._bg_draw(self._stage, "background"))
		self.heart_roi_btn = QPushButton("ROI corazón")
		self.heart_roi_btn.setToolTip("Solo en modo localizado: marcá el corazón. La resta se limita a esa zona.")
		self.heart_roi_btn.clicked.connect(lambda: self._window._bg_draw(self._stage, "heart"))
		bg_row3.addWidget(self.bg_roi_btn)
		bg_row3.addWidget(self.heart_roi_btn)
		bg_lay.addLayout(bg_row3)

		bg_row4 = QHBoxLayout()
		bg_row4.setSpacing(4)
		self.bg_apply_btn = QPushButton("Aplicar")
		self.bg_apply_btn.clicked.connect(lambda: self._window._bg_apply(self._stage))
		self.bg_clear_btn = QPushButton("Limpiar")
		self.bg_clear_btn.clicked.connect(lambda: self._window._bg_clear(self._stage))
		bg_row4.addWidget(self.bg_apply_btn)
		bg_row4.addWidget(self.bg_clear_btn)
		bg_lay.addLayout(bg_row4)

		self.bg_status_label = QLabel("")
		self.bg_status_label.setWordWrap(True)
		self.bg_status_label.setStyleSheet("color:#94a3b8; font-size:10px;")
		bg_lay.addWidget(self.bg_status_label)

		self.bg_panel.setVisible(False)
		lay.addWidget(self.bg_panel)
		self._refresh_bg_widgets()

		self.status_label = QLabel("")
		self.status_label.setWordWrap(True)
		self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.status_label.setStyleSheet("color:#cbd5e1; font-size:11px;")
		lay.addWidget(self.status_label)

	def set_status(self, text: str):
		self.status_label.setText(text or "")

	def _on_bg_toggle(self, checked: bool):
		self.bg_panel.setVisible(bool(checked))
		self.bg_toggle_btn.setText("Fondo ▴" if checked else "Fondo ▾")

	def _on_bg_method_changed(self, *_a):
		self._refresh_bg_widgets()

	def bg_method(self) -> str:
		return str(self.bg_method_combo.currentData() or "constant")

	def bg_impact(self) -> str:
		return str(self.bg_impact_combo.currentData() or "visual")

	def set_bg_status(self, text: str):
		self.bg_status_label.setText(text or "")

	def _refresh_bg_widgets(self):
		localized = self.bg_method() == "localized"
		self.heart_roi_btn.setEnabled(localized)

	def set_enabled_state(self, enabled: bool):
		for b in (self.motion_btn, self.recon_btn, self.reorient_btn):
			b.setEnabled(enabled)
		self.method_combo.setEnabled(enabled)
		self.axis_combo.setEnabled(enabled)

	def set_state_text(self, state: str):
		colors = {"crudo": "#94a3b8", "reconstruido": "#38bdf8", "reorientado": "#4ade80"}
		color = colors.get(state, "#e2e8f0")
		self.title_label.setText(f"{self._title} · {state}")
		self.title_label.setStyleSheet(f"font-weight:bold; font-size:14px; color:{color};")


class PreparacionWindow(QDialog):
	"""Ventana no modal de preparación dual (Esfuerzo / Reposo) con MIP en vivo."""

	def __init__(self, main_window):
		super().__init__(main_window)
		self._main = main_window
		self.setWindowTitle("Preparación / Reconstrucción · Esfuerzo | Reposo")
		self.setModal(False)
		self.resize(1180, 900)
		# Caché de volumen por etapa: el main tiene slot único de reconstrucción,
		# así que la ventana retiene el mejor volumen de cada etapa (reorientado >
		# reconstruido) para que ambas columnas queden independientes.
		self._stage_vol: dict[str, np.ndarray | None] = {"stress": None, "rest": None}
		self._stage_state: dict[str, str] = {"stress": "crudo", "rest": "crudo"}
		# Proyecciones corregidas EN VIVO por etapa: refresh() debe respetarlas y no
		# volver a mostrar el crudo (P3). Se descartan al cambiar de estudio o al
		# reconstruir (el volumen tiene prioridad).
		self._stage_proj: dict[str, np.ndarray | None] = {"stress": None, "rest": None}
		self._stage_proj_status: dict[str, str] = {"stress": "", "rest": ""}
		self._last_signature = None
		# Fase 1 fusión: la ventana aloja el panel cine_crudo completo del main.
		self._cine_crudo_panel: QWidget | None = None
		self._cine_crudo_attached = False
		# Sustracción de fondo sobre el crudo (realce visual + opción "toda la cadena").
		self._bg_base: dict[str, np.ndarray | None] = {"stress": None, "rest": None}
		self._bg_spec: dict[str, dict | None] = {"stress": None, "rest": None}

		root = QVBoxLayout(self)
		root.setContentsMargins(8, 8, 8, 8)
		root.setSpacing(6)

		header = QLabel(
			"Preparación en vivo (en memoria). Una columna por etapa; funcionan de forma independiente. "
			"El MIP rota automáticamente (clic en la imagen para pausar/reanudar)."
		)
		header.setWordWrap(True)
		header.setStyleSheet("color:#94a3b8;")
		root.addWidget(header)

		cols = QHBoxLayout()
		cols.setSpacing(8)
		self.col_stress = _StageColumn(self, "stress", "Esfuerzo")
		self.col_rest = _StageColumn(self, "rest", "Reposo")
		self._columns = {"stress": self.col_stress, "rest": self.col_rest}
		cols.addWidget(self.col_stress, 1)
		cols.addWidget(self.col_rest, 1)
		cols_widget = QWidget()
		cols_widget.setLayout(cols)

		# Contenedor para el panel cine_crudo reubicado (todos los controles del
		# motor de preparación: corrección de movimiento, reconstrucción, montaje…).
		self._cine_crudo_container = QWidget()
		container_lay = QVBoxLayout(self._cine_crudo_container)
		container_lay.setContentsMargins(0, 0, 0, 0)
		container_lay.setSpacing(0)
		self._cine_crudo_container_layout = container_lay
		self._cine_crudo_placeholder = QLabel(
			"El panel completo de cine crudo se aloja aquí mientras esta ventana está abierta."
		)
		self._cine_crudo_placeholder.setWordWrap(True)
		self._cine_crudo_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self._cine_crudo_placeholder.setStyleSheet("color:#64748b; padding:12px;")
		container_lay.addWidget(self._cine_crudo_placeholder)

		# Splitter vertical: arriba resumen dual (MIP), abajo el motor cine_crudo.
		vsplit = QSplitter(Qt.Orientation.Vertical)
		vsplit.setChildrenCollapsible(False)
		vsplit.addWidget(cols_widget)
		vsplit.addWidget(self._cine_crudo_container)
		vsplit.setStretchFactor(0, 0)
		vsplit.setStretchFactor(1, 1)
		vsplit.setSizes([320, 520])
		root.addWidget(vsplit, 1)

		bottom = QHBoxLayout()
		self.refresh_btn = QPushButton("↻ Actualizar MIP")
		self.refresh_btn.clicked.connect(self.refresh)
		bottom.addWidget(self.refresh_btn)
		bottom.addStretch(1)
		self.close_btn = QPushButton("Cerrar")
		self.close_btn.clicked.connect(self.close)
		bottom.addWidget(self.close_btn)
		root.addLayout(bottom)

	# ---------------------------------------- Fase 1: panel cine_crudo alojado
	def attach_cine_crudo_panel(self):
		"""Reubica el panel cine_crudo del main dentro de esta ventana.

		No duplica lógica ni estado: el mismo widget (con todos sus controles y
		handlers, que viven en el main) se muestra aquí. Al cerrar la ventana se
		devuelve a su pestaña original (detach_cine_crudo_panel)."""
		if self._cine_crudo_attached:
			return
		panel = None
		if hasattr(self._main, "take_cine_crudo_panel"):
			panel = self._main.take_cine_crudo_panel()
		if panel is None:
			return
		self._cine_crudo_placeholder.setVisible(False)
		self._cine_crudo_container_layout.addWidget(panel)
		panel.setVisible(True)
		self._cine_crudo_panel = panel
		self._cine_crudo_attached = True

	def detach_cine_crudo_panel(self):
		"""Devuelve el panel cine_crudo a su pestaña original en el main."""
		if not self._cine_crudo_attached:
			return
		panel = self._cine_crudo_panel
		if panel is not None:
			self._cine_crudo_container_layout.removeWidget(panel)
			panel.setParent(None)
		self._cine_crudo_panel = None
		self._cine_crudo_attached = False
		self._cine_crudo_placeholder.setVisible(True)
		if hasattr(self._main, "restore_cine_crudo_panel"):
			self._main.restore_cine_crudo_panel()

	def closeEvent(self, event):
		self.detach_cine_crudo_panel()
		super().closeEvent(event)

	# -------------------------------------------------- sustracción de fondo (crudo)
	def _bg_draw(self, stage: str, mode: str):
		column = self._columns.get(stage)
		if column is None:
			return
		if column.mip.kind() != "proj":
			column.set_bg_status("El fondo se dibuja sobre el crudo (proyecciones que rotan).")
			return
		column.mip.set_draw_mode(mode)
		etiqueta = "fondo (zona SIN corazón)" if mode == "background" else "corazón"
		column.set_bg_status(f"Marcá la ROI de {etiqueta}. Clic para agregar vértices, clic derecho para cerrar.")

	def _bg_clear(self, stage: str):
		column = self._columns.get(stage)
		if column is None:
			return
		column.mip.clear_polygons()
		base = self._bg_base.get(stage)
		if base is not None:
			column.mip.set_source("proj", base, self._stage_proj_status.get(stage, ""))
		self._bg_base[stage] = None
		self._bg_spec[stage] = None
		column.set_bg_status("Fondo limpiado.")
		main = self._main
		if hasattr(main, "clear_raw_background_subtraction"):
			try:
				main.clear_raw_background_subtraction(stage)
			except Exception:
				pass

	def _bg_apply(self, stage: str):
		from core.raw_background import (
			measure_background_level,
			polygon_mask,
			subtract_constant,
			subtract_localized,
		)

		column = self._columns.get(stage)
		if column is None:
			return
		mip = column.mip
		if mip.kind() != "proj":
			column.set_bg_status("La sustracción aplica sobre el crudo. Hacela antes de reconstruir.")
			return
		base = self._bg_base.get(stage)
		if base is None:
			base = mip.frames_stack()
			if base is None:
				column.set_bg_status("Sin imagen cruda para procesar.")
				return
			self._bg_base[stage] = base
		bg_poly = mip.background_polygon()
		if len(bg_poly) < 3:
			column.set_bg_status("Dibujá primero la ROI de fondo (clic derecho para cerrar).")
			return
		h, w = int(base.shape[-2]), int(base.shape[-1])
		bg_mask = polygon_mask((h, w), bg_poly)
		mean_img = base.mean(axis=0)
		level = measure_background_level(mean_img, bg_mask, stat="median")
		method = column.bg_method()
		if method == "localized":
			heart_poly = mip.heart_polygon()
			if len(heart_poly) < 3:
				column.set_bg_status("Modo localizado: falta dibujar la ROI del corazón.")
				return
			heart_mask = polygon_mask((h, w), heart_poly)
			res = subtract_localized(base, level, heart_mask, feather_px=2.0)
		else:
			res = subtract_constant(base, level)
		subtracted = np.asarray(res.image, dtype=np.float64)
		mip.set_source("proj", subtracted, self._stage_proj_status.get(stage, ""))
		msg = f"Fondo {method}: nivel {level:.1f} · clip {res.clipped_fraction * 100:.0f}%"
		if res.notes:
			msg += "  ⚠ " + res.notes[0]
		column.set_bg_status(msg)
		impact = column.bg_impact()
		self._bg_spec[stage] = {
			"method": method,
			"level": float(level),
			"impact": impact,
			"bg_polygon": list(bg_poly),
			"heart_polygon": list(mip.heart_polygon()),
			"shape": (h, w),
		}
		if impact == "chain":
			main = self._main
			if hasattr(main, "set_raw_background_subtraction"):
				try:
					main.set_raw_background_subtraction(stage, subtracted, self._bg_spec[stage])
					column.set_bg_status(msg + " · alimentará la reconstrucción")
				except Exception as exc:  # pragma: no cover - defensivo
					column.set_bg_status(msg + f" · [chain error: {exc}]")

	# -------------------------------------------------- delegación al main
	def _delegate(self, stage: str, action: str):
		main = self._main
		column = self._columns.get(stage)
		try:
			if action == "motion":
				self._run_motion_live(stage, column)
				return
			if hasattr(main, "_set_active_cine_crudo_stage"):
				main._set_active_cine_crudo_stage(stage)
			if action == "recon":
				if column is not None:
					column.set_status("Reconstruyendo…")
					column.repaint()
				if hasattr(main, "_reconstruct_cine_crudo_raw"):
					main._reconstruct_cine_crudo_raw()
				self._capture_stage_volume(stage)
				if column is not None:
					column.set_status("Reconstrucción lista.")
			elif action == "reorient":
				if hasattr(main, "_open_cine_crudo_reorientation"):
					main._open_cine_crudo_reorientation()
				self._capture_stage_volume(stage)
				if column is not None:
					column.set_status("Reorientación aplicada.")
		except Exception as exc:  # pragma: no cover - defensivo
			if column is not None:
				column.set_status(f"Error: {exc}")
			if hasattr(main, "_log"):
				main._log(f"[ERROR preparación:{action}:{stage}] {exc}")
		self.refresh()

	def _run_motion_live(self, stage: str, column: "_StageColumn | None"):
		"""Corrige el movimiento de la etapa y muestra el cine corregido EN esta ventana."""
		main = self._main
		if column is None or not hasattr(main, "run_stage_motion_live"):
			return
		column.set_status("Aplicando movimiento…")
		column.repaint()
		corrected, result = main.run_stage_motion_live(
			stage,
			method_label=column.method_combo.currentText(),
			axis_label=column.axis_combo.currentText(),
		)
		if corrected is None:
			column.set_status("Sin datos crudos para corregir esta etapa.")
			return
		arr = np.asarray(corrected, dtype=np.float64)
		if arr.ndim == 4:
			arr = arr.sum(axis=0)  # (gates, ángulos, H, W) → proyecciones (ángulos, H, W)
		status = self._motion_status_text(result)
		self._stage_proj[stage] = arr
		self._stage_proj_status[stage] = status
		column.mip.set_source("proj", arr, status)
		column.set_state_text("crudo")
		column.set_status(status)

	@staticmethod
	def _motion_status_text(result) -> str:
		if not isinstance(result, dict):
			return "Movimiento aplicado (cine corregido)."
		method = result.get("method_auto_selected") or result.get("method") or "?"
		max_shift = result.get("max_shift_px")
		axis = result.get("axis_corrected") or ""
		parts = [f"Corregido · {method}"]
		if max_shift is not None:
			parts.append(f"max {max_shift} px")
		if axis:
			parts.append(f"eje {axis}")
		return " · ".join(parts)

	def _capture_stage_volume(self, stage: str):
		"""Retiene el mejor volumen de la etapa desde el slot único del main."""
		main = self._main
		if str(getattr(main, "_cine_crudo_recon_stage", "stress")) != stage:
			return
		reo = getattr(main, "cine_crudo_reoriented_ungated", None)
		if reo is not None:
			vol = np.asarray(reo, dtype=np.float64)
			if vol.ndim == 3 and vol.size:
				self._stage_vol[stage] = vol
				self._stage_state[stage] = "reorientado"
				self._stage_proj[stage] = None  # el volumen manda
				return
		rr = getattr(main, "cine_crudo_recon_result", None)
		if rr is not None:
			vol = np.asarray(getattr(rr, "ungated_volume", None), dtype=np.float64)
			if vol.ndim == 3 and vol.size:
				self._stage_vol[stage] = vol
				self._stage_state[stage] = "reconstruido"
				self._stage_proj[stage] = None  # el volumen manda

	# -------------------------------------------------- refresco del MIP
	def refresh(self):
		main = self._main
		self._invalidate_on_study_change()
		secondary = None
		if hasattr(main, "_secondary_cine_crudo_study"):
			secondary = main._secondary_cine_crudo_study()
		has_secondary = secondary is not None

		self._refresh_column(self.col_stress, "stress", enabled=True)
		self._refresh_column(self.col_rest, "rest", enabled=has_secondary)
		if not has_secondary:
			self.col_rest.mip.clear("Sin segunda etapa cargada")
			self.col_rest.set_state_text("crudo")

	def _study_signature(self):
		main = self._main

		def sig(st):
			if st is None:
				return None
			return (
				str(getattr(st, "patient_id", "")),
				str(getattr(st, "study_date", "")),
				str(getattr(st, "study_time", "")),
			)

		sec = main._secondary_cine_crudo_study() if hasattr(main, "_secondary_cine_crudo_study") else None
		return (sig(getattr(main, "study", None)), sig(sec))

	def _invalidate_on_study_change(self):
		"""Descarta el caché de volúmenes si cambiaron los estudios cargados."""
		signature = self._study_signature()
		if signature != self._last_signature:
			self._stage_vol = {"stress": None, "rest": None}
			self._stage_state = {"stress": "crudo", "rest": "crudo"}
			self._stage_proj = {"stress": None, "rest": None}
			self._stage_proj_status = {"stress": "", "rest": ""}
			self._last_signature = signature

	def _refresh_column(self, column: "_StageColumn", stage: str, enabled: bool):
		column.set_enabled_state(enabled)
		if not enabled:
			return
		# 1) Volumen retenido por la ventana (reorientado/reconstruido de esta etapa).
		cached = self._stage_vol.get(stage)
		if cached is not None:
			state = self._stage_state.get(stage, "reconstruido")
			column.mip.set_source("vol", cached, f"{state.capitalize()} · MIP rotatorio")
			column.set_state_text(state)
			return
		# 1b) Proyecciones corregidas EN VIVO (P3): no volver a mostrar el crudo.
		proj = self._stage_proj.get(stage)
		if proj is not None:
			status = self._stage_proj_status.get(stage, "") or "Movimiento corregido"
			column.mip.set_source("proj", proj, status)
			column.set_state_text("crudo")
			column.set_status(status)
			return
		# 2) Fuente cruda/actual desde el main.
		main = self._main
		src = None
		if hasattr(main, "_prep_mip_source_for_stage"):
			try:
				src = main._prep_mip_source_for_stage(stage)
			except Exception:
				src = None
		if src is None:
			column.mip.clear("Sin datos para esta etapa")
			column.set_state_text("crudo")
			return
		kind, array, status = src
		column.mip.set_source(kind, array, status)
		column.set_state_text("reconstruido" if kind == "vol" else "crudo")
