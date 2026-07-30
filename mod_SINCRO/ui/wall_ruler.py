"""SINCRO - ui.wall_ruler

Cota interactiva para medir el espesor de pared sobre el corte telediastólico.

PARA QUÉ
--------
El método ECTb asume 10 mm de espesor de pared en telediástole. Es una
convención razonable, pero es una convención: en una hipertrofia real la pared
puede andar en 14-16 mm, y en un ventrículo dilatado y adelgazado, en 6-7 mm.
Como ese número es lo que ancla la escala absoluta de los volúmenes, errarle
mucho corre el EDV y el ESV.

Esta herramienta permite **medir la pared sobre la misma imagen en la que el
algoritmo está trabajando** y usar ese valor en lugar del asumido. No reemplaza
a la ecocardiografía: es una cota sobre el SPECT, con la resolución que tiene el
SPECT. Sirve como acotación superior/inferior del valor asumido, no como una
medición de precisión milimétrica.

Es opcional y viene apagada. Si no se toca, el cálculo sigue usando los 10 mm
(o lo que el usuario haya puesto en el campo).

CÓMO SE USA
-----------
- Arrastrar los extremos de la cota para apoyarlos en el borde interno y externo
  de la pared.
- Hacer clic y arrastrar en una zona vacía dibuja una cota nueva desde cero.
- Rueda del mouse: zoom. Botón derecho o del medio arrastrando: desplazar.
- La medición se informa en milímetros reales del paciente y **no cambia con el
  zoom**: el zoom es solo de visualización, la cuenta se hace siempre en
  coordenadas de imagen multiplicadas por el tamaño de píxel del estudio.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget


class WallThicknessRuler(QWidget):
	"""Visor de un corte con una cota de dos puntos arrastrable."""

	#: Distancia medida, en milímetros. Se emite en vivo durante el arrastre.
	measured = pyqtSignal(float)

	#: Radio en píxeles de pantalla para "agarrar" un extremo de la cota.
	HANDLE_GRAB_PX = 12.0
	HANDLE_DRAW_PX = 5.0

	def __init__(self, parent=None):
		super().__init__(parent)
		self._image: np.ndarray | None = None
		self._pixmap: QPixmap | None = None
		self._pixel_mm = 1.0
		self._zoom = 1.0
		self._offset = QPointF(0.0, 0.0)
		self._base_scale = 1.0
		#: Extremos de la cota, en coordenadas de imagen (x=columna, y=fila).
		self._p1 = QPointF(0.0, 0.0)
		self._p2 = QPointF(0.0, 0.0)
		self._dragging: str | None = None
		self._pan_from: QPointF | None = None
		#: Contornos de referencia: lista de (color, puntos en coords de imagen).
		self._contours: list[tuple[QColor, np.ndarray]] = []
		self._show_contours = True
		self._show_ruler = True
		#: Escala de grises en vivo (window/level) e inversión.
		self._invert = False
		self._colormap = "gray"
		self._data_min = 0.0
		self._data_top = 1.0
		#: Nivel y ancho de ventana como fracción de [min, top] (0..1).
		self._wl_level = 0.5
		self._wl_width = 1.0
		#: Forma de la última imagen, para preservar zoom/paneo entre refrescos.
		self._image_shape: tuple[int, int] | None = None

		self.setMinimumHeight(260)
		self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		self.setMouseTracking(True)
		self.setCursor(Qt.CursorShape.CrossCursor)
		self.setToolTip(
			"Arrastrá los extremos de la cota hasta el borde interno y externo de la pared.\n"
			"Clic y arrastre en una zona vacía: dibuja una cota nueva.\n"
			"Rueda: zoom. Botón derecho arrastrando: desplazar la imagen.\n"
			"La medida está en milímetros reales y no cambia con el zoom."
		)

	# ------------------------------------------------------------------
	# Datos
	# ------------------------------------------------------------------

	def set_image(self, image: np.ndarray, pixel_mm: float):
		"""Carga el corte a medir.

		Se normaliza al percentil 99 en vez de al máximo para que un píxel
		caliente aislado no aplaste el resto de la escala de grises.

		Si el corte que llega tiene la misma forma que el anterior (por ejemplo
		un recálculo en vivo o un cambio de gate), se conserva el zoom, el paneo
		y la ventana de grises que el usuario haya ajustado. Solo se reencuadra
		cuando cambia el tamaño de la imagen o es la primera vez.
		"""
		arr = np.asarray(image, dtype=np.float64)
		if arr.ndim != 2 or arr.size == 0:
			self._image = None
			self._pixmap = None
			self._image_shape = None
			self.update()
			return

		same_shape = self._image_shape == arr.shape
		self._image = arr
		self._pixel_mm = max(float(pixel_mm), 1e-6)
		top = float(np.percentile(arr, 99.0))
		if not np.isfinite(top) or top <= 0.0:
			top = float(arr.max()) or 1.0
		self._data_min = float(np.nanmin(arr))
		self._data_top = top if top > self._data_min else self._data_min + 1.0
		self._render_pixmap()
		if not same_shape:
			self._image_shape = arr.shape
			self._fit()
		self.update()

	def _render_pixmap(self):
		"""Reconstruye el pixmap aplicando la ventana de grises, la inversión y la escala de color."""
		if self._image is None:
			self._pixmap = None
			return
		rng = max(self._data_top - self._data_min, 1e-9)
		lo_frac = self._wl_level - self._wl_width / 2.0
		hi_frac = self._wl_level + self._wl_width / 2.0
		lo = self._data_min + lo_frac * rng
		hi = self._data_min + hi_frac * rng
		span = max(hi - lo, 1e-9)
		norm = np.clip((self._image - lo) / span, 0.0, 1.0)
		if self._invert:
			norm = 1.0 - norm
		cmap_name = str(self._colormap or "gray").lower()
		if cmap_name in ("gray", "grey", ""):
			buf = np.ascontiguousarray((norm * 255.0).astype(np.uint8))
			height, width = buf.shape
			qimage = QImage(buf.data, width, height, width, QImage.Format.Format_Grayscale8)
			self._pixmap = QPixmap.fromImage(qimage.copy())
			return
		try:
			from ui.cine_widget import _resolve_cmap

			cmap = _resolve_cmap(self._colormap)
			rgb = np.asarray(cmap(norm)[..., :3], dtype=np.float32)
			rgb8 = np.ascontiguousarray((rgb * 255.0).astype(np.uint8))
			height, width, _ = rgb8.shape
			qimage = QImage(rgb8.data, width, height, 3 * width, QImage.Format.Format_RGB888)
			self._pixmap = QPixmap.fromImage(qimage.copy())
		except Exception:
			# Si el colormap no se puede resolver, se cae a escala de grises.
			buf = np.ascontiguousarray((norm * 255.0).astype(np.uint8))
			height, width = buf.shape
			qimage = QImage(buf.data, width, height, width, QImage.Format.Format_Grayscale8)
			self._pixmap = QPixmap.fromImage(qimage.copy())

	def set_window_level(self, level_frac: float, width_frac: float):
		"""Ajusta la ventana de grises en vivo (fracciones 0..1 de [min, top])."""
		self._wl_level = float(np.clip(level_frac, 0.0, 1.0))
		self._wl_width = float(np.clip(width_frac, 0.02, 2.0))
		self._render_pixmap()
		self.update()

	def set_invert(self, invert: bool):
		"""Invierte la escala de grises (blanco↔negro) en vivo."""
		self._invert = bool(invert)
		self._render_pixmap()
		self.update()

	def set_colormap(self, name: str):
		"""Cambia la escala de color de la vista en vivo."""
		self._colormap = str(name) if name else "gray"
		self._render_pixmap()
		self.update()

	def set_show_ruler(self, visible: bool):
		"""Muestra u oculta la cota sin borrar su posición."""
		self._show_ruler = bool(visible)
		self.update()

	def set_contours(self, contours: list[tuple[QColor, np.ndarray]]):
		"""Contornos de referencia (endocardio, centro de pared, epicardio)."""
		self._contours = list(contours)
		self.update()

	def set_show_contours(self, visible: bool):
		self._show_contours = bool(visible)
		self.update()

	def set_measurement(self, p1: tuple[float, float], p2: tuple[float, float]):
		"""Posiciona la cota (coordenadas de imagen) y emite la medida."""
		self._p1 = QPointF(float(p1[0]), float(p1[1]))
		self._p2 = QPointF(float(p2[0]), float(p2[1]))
		self.update()
		self.measured.emit(self.measurement_mm())

	def measurement_mm(self) -> float:
		delta = self._p2 - self._p1
		length_px = float(np.hypot(delta.x(), delta.y()))
		return length_px * self._pixel_mm

	def reset_view(self):
		self._zoom = 1.0
		self._fit()
		self.update()

	# ------------------------------------------------------------------
	# Transformación imagen <-> pantalla
	# ------------------------------------------------------------------

	def _fit(self):
		"""Calcula la escala base para que el corte entre completo en el widget."""
		if self._pixmap is None:
			return
		w = max(1, self._pixmap.width())
		h = max(1, self._pixmap.height())
		self._base_scale = min(self.width() / w, self.height() / h) * 0.92
		scale = self._base_scale * self._zoom
		self._offset = QPointF(
			(self.width() - w * scale) / 2.0,
			(self.height() - h * scale) / 2.0,
		)

	@property
	def _scale(self) -> float:
		return self._base_scale * self._zoom

	def _to_screen(self, point: QPointF) -> QPointF:
		scale = self._scale
		return QPointF(point.x() * scale + self._offset.x(), point.y() * scale + self._offset.y())

	def _to_image(self, point: QPointF) -> QPointF:
		scale = max(self._scale, 1e-9)
		return QPointF((point.x() - self._offset.x()) / scale, (point.y() - self._offset.y()) / scale)

	def resizeEvent(self, event):  # noqa: N802 - API de Qt
		super().resizeEvent(event)
		self._fit()

	# ------------------------------------------------------------------
	# Interacción
	# ------------------------------------------------------------------

	def mousePressEvent(self, event):  # noqa: N802 - API de Qt
		if self._pixmap is None:
			return
		pos = QPointF(event.position())
		if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
			self._pan_from = pos
			self.setCursor(Qt.CursorShape.ClosedHandCursor)
			return
		if event.button() != Qt.MouseButton.LeftButton:
			return

		if not self._show_ruler:
			# Con la cota oculta, el clic izquierdo no crea ni arrastra la cota.
			return

		near = self._handle_at(pos)
		if near:
			self._dragging = near
		else:
			# Clic en zona vacía: arranca una cota nueva desde acá.
			image_pos = self._to_image(pos)
			self._p1 = image_pos
			self._p2 = QPointF(image_pos)
			self._dragging = "p2"
		self.update()

	def mouseMoveEvent(self, event):  # noqa: N802 - API de Qt
		pos = QPointF(event.position())
		if self._pan_from is not None:
			delta = pos - self._pan_from
			self._offset += delta
			self._pan_from = pos
			self.update()
			return
		if self._dragging is None:
			self.setCursor(
				Qt.CursorShape.SizeAllCursor if self._handle_at(pos) else Qt.CursorShape.CrossCursor
			)
			return
		image_pos = self._to_image(pos)
		if self._dragging == "p1":
			self._p1 = image_pos
		else:
			self._p2 = image_pos
		self.update()
		self.measured.emit(self.measurement_mm())

	def mouseReleaseEvent(self, event):  # noqa: N802 - API de Qt
		self._dragging = None
		self._pan_from = None
		self.setCursor(Qt.CursorShape.CrossCursor)

	def wheelEvent(self, event):  # noqa: N802 - API de Qt
		if self._pixmap is None:
			return
		# Zoom manteniendo fijo el punto de la imagen que está bajo el cursor.
		cursor = QPointF(event.position())
		before = self._to_image(cursor)
		steps = event.angleDelta().y() / 120.0
		self._zoom = float(np.clip(self._zoom * (1.15 ** steps), 0.25, 20.0))
		after = self._to_image(cursor)
		scale = self._scale
		self._offset += QPointF((after.x() - before.x()) * scale, (after.y() - before.y()) * scale)
		self.update()

	def _handle_at(self, screen_pos: QPointF) -> str | None:
		for name, point in (("p1", self._p1), ("p2", self._p2)):
			delta = self._to_screen(point) - screen_pos
			if float(np.hypot(delta.x(), delta.y())) <= self.HANDLE_GRAB_PX:
				return name
		return None

	# ------------------------------------------------------------------
	# Dibujo
	# ------------------------------------------------------------------

	def paintEvent(self, event):  # noqa: N802 - API de Qt
		painter = QPainter(self)
		painter.fillRect(self.rect(), QColor("#0d0d0d"))
		if self._pixmap is None:
			painter.setPen(QColor("#8a8a8a"))
			painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sin corte para medir")
			painter.end()
			return

		painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
		scale = self._scale
		target_w = self._pixmap.width() * scale
		target_h = self._pixmap.height() * scale
		painter.drawPixmap(
			int(self._offset.x()), int(self._offset.y()), int(target_w), int(target_h), self._pixmap
		)

		painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
		if self._show_contours:
			self._draw_contours(painter)
		if self._show_ruler:
			self._draw_ruler(painter)
		painter.end()

	def _draw_contours(self, painter: QPainter):
		for color, points in self._contours:
			if points is None or len(points) < 3:
				continue
			pen = QPen(color, 1.2)
			painter.setPen(pen)
			screen = [self._to_screen(QPointF(float(x), float(y))) for x, y in points]
			for i in range(len(screen)):
				painter.drawLine(screen[i], screen[(i + 1) % len(screen)])

	def _draw_ruler(self, painter: QPainter):
		s1 = self._to_screen(self._p1)
		s2 = self._to_screen(self._p2)
		delta = s2 - s1
		length = float(np.hypot(delta.x(), delta.y()))
		if length < 1e-6:
			return

		# Línea principal con contorno oscuro para que se lea sobre cualquier fondo.
		painter.setPen(QPen(QColor(0, 0, 0, 180), 4))
		painter.drawLine(s1, s2)
		painter.setPen(QPen(QColor("#ffd23f"), 2))
		painter.drawLine(s1, s2)

		# Topes perpendiculares en los extremos, como una cota de plano.
		ux, uy = delta.x() / length, delta.y() / length
		px, py = -uy, ux
		cap = 7.0
		for point in (s1, s2):
			a = QPointF(point.x() + px * cap, point.y() + py * cap)
			b = QPointF(point.x() - px * cap, point.y() - py * cap)
			painter.setPen(QPen(QColor(0, 0, 0, 180), 4))
			painter.drawLine(a, b)
			painter.setPen(QPen(QColor("#ffd23f"), 2))
			painter.drawLine(a, b)

		for point in (s1, s2):
			painter.setBrush(QColor("#ffd23f"))
			painter.setPen(QPen(QColor("#4a3a00"), 1))
			painter.drawEllipse(point, self.HANDLE_DRAW_PX, self.HANDLE_DRAW_PX)

		label = f"{self.measurement_mm():.1f} mm"
		font = QFont(painter.font())
		font.setPointSize(10)
		font.setBold(True)
		painter.setFont(font)
		mid = QPointF((s1.x() + s2.x()) / 2.0 + px * 16.0, (s1.y() + s2.y()) / 2.0 + py * 16.0)
		metrics = painter.fontMetrics()
		rect = metrics.boundingRect(label).adjusted(-5, -3, 5, 3)
		rect.moveCenter(mid.toPoint())
		painter.setPen(Qt.PenStyle.NoPen)
		painter.setBrush(QColor(0, 0, 0, 170))
		painter.drawRoundedRect(rect, 3, 3)
		painter.setPen(QColor("#ffd23f"))
		painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
