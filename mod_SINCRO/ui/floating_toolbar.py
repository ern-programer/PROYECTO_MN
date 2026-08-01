"""Barra de herramientas flotante, movible y con orientación configurable.

Se usa como reemplazo de los menús desplegables (QMenu) para agrupar
controles secundarios (p.ej. "Corrección de movimiento", "ROI automático")
sin que el botón que la abre cambie nunca de tamaño ni empuje el resto del
layout: la barra es una ventana propia (Qt.WindowType.Tool) que flota por
encima de la ventana principal, se puede arrastrar tomándola del título y
se puede alternar entre horizontal/vertical para acomodarla donde moleste
menos. Posición y orientación se recuerdan entre sesiones vía QSettings.
"""
from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt, QSettings
from PyQt6.QtGui import QColor, QGuiApplication, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QSizeGrip, QToolButton, QVBoxLayout, QWidget


class _DragHeader(QWidget):
	"""Franja superior de la barra: al arrastrarla desde una zona libre (no
	sobre los botones) mueve toda la ventana flotante."""

	def __init__(self, toolbar: "FloatingToolbar"):
		super().__init__(toolbar)
		self._toolbar = toolbar
		self._drag_offset: QPoint | None = None
		self._press_global: QPoint | None = None
		self._drag_started = False

	def mousePressEvent(self, event):
		if event.button() == Qt.MouseButton.LeftButton:
			self._press_global = event.globalPosition().toPoint()
			self._drag_started = False
			self._drag_offset = event.globalPosition().toPoint() - self._toolbar.frameGeometry().topLeft()
			event.accept()

	def mouseMoveEvent(self, event):
		if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
			if self._press_global is not None:
				delta = event.globalPosition().toPoint() - self._press_global
				if delta.manhattanLength() > 4:
					self._drag_started = True
			self._toolbar.move(event.globalPosition().toPoint() - self._drag_offset)
			event.accept()

	def mouseReleaseEvent(self, event):
		if event.button() == Qt.MouseButton.LeftButton and not self._drag_started:
			self._toolbar.toggle_collapsed()
		self._drag_offset = None
		self._press_global = None
		self._drag_started = False
		self._toolbar._save_state()

	def paintEvent(self, event):
		super().paintEvent(event)
		p = QPainter(self)
		g = QLinearGradient(0, 0, 0, self.height())
		g.setColorAt(0.0, QColor("#dbdbdb"))
		g.setColorAt(0.5, QColor("#c7c7c7"))
		g.setColorAt(1.0, QColor("#b7b7b7"))
		p.fillRect(self.rect(), g)
		p.setPen(QPen(QColor("#a9a9a9"), 1))
		h = self.height()
		for x in range(-h, self.width(), 8):
			p.drawLine(x, h, x + h, 0)
		p.setPen(QPen(QColor("#8d8d8d"), 1))
		p.drawLine(0, h - 1, self.width(), h - 1)


class FloatingToolbar(QWidget):
	"""Barra de herramientas flotante que agrupa uno o más layouts de
	controles (QHBoxLayout/QGridLayout ya armados) y se abre/cierra con un
	botón trigger, sin afectar el tamaño del panel que la contiene."""

	def __init__(self, title: str, key: str, parent=None):
		super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
		self.setWindowTitle(title)
		self._key = key
		self._settings = QSettings("Gammasys", "GammaSync")
		self._orientation = Qt.Orientation.Horizontal
		self._groups: list = []
		self._widgets: list[QWidget] = []
		self._collapsed = False
		self._rebuilding = False
		self._horizontal_rows = 3

		self.setStyleSheet(
			"FloatingToolbar { background: #b8b8b8; border: 1px solid #888; border-radius: 4px; }"
			" FloatingToolbar QLabel { color: #222; }"
			" QWidget#floatingToolbarHeader {"
			" background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e0e0e0, stop:0.48 #cecece, stop:0.52 #c2c2c2, stop:1 #b7b7b7);"
			" border-top: 1px solid #f6f6f6;"
			" border-bottom: 1px solid #8f8f8f;"
			" }"
			" FloatingToolbar QToolButton { border: 1px solid #9b9b9b; border-radius: 3px; background: #d7d7d7; }"
			" FloatingToolbar QToolButton:hover { background: #ececec; }"
		)

		outer = QVBoxLayout(self)
		outer.setContentsMargins(1, 1, 1, 1)
		outer.setSpacing(0)

		header = _DragHeader(self)
		header.setObjectName("floatingToolbarHeader")
		header_layout = QHBoxLayout(header)
		header_layout.setContentsMargins(6, 2, 2, 2)
		header_layout.setSpacing(2)
		title_lbl = QLabel(title)
		title_lbl.setStyleSheet("font-weight:600; letter-spacing:0.2px;")
		header_layout.addWidget(title_lbl)
		header_layout.addStretch(1)

		orient_btn = QToolButton()
		orient_btn.setText("⇄")
		orient_btn.setToolTip("Cambiar orientación horizontal / vertical")
		orient_btn.setAutoRaise(True)
		orient_btn.clicked.connect(self._toggle_orientation)
		header_layout.addWidget(orient_btn)

		close_btn = QToolButton()
		close_btn.setText("✕")
		close_btn.setToolTip("Cerrar barra")
		close_btn.setAutoRaise(True)
		close_btn.clicked.connect(self.hide)
		header_layout.addWidget(close_btn)

		outer.addWidget(header)

		self._content_host = QWidget()
		self._content_layout = QGridLayout(self._content_host)
		self._content_layout.setContentsMargins(6, 4, 6, 6)
		self._content_layout.setSpacing(8)
		outer.addWidget(self._content_host)

		self._size_grip = QSizeGrip(self)
		self._size_grip.setToolTip("Arrastrar para redimensionar")
		outer.addWidget(self._size_grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
		self.setMinimumSize(300, 110)

		self._restore_state()

	def add_layout(self, layout) -> None:
		"""Agrega un QHBoxLayout/QGridLayout ya armado con sus widgets."""
		self._groups.append(layout)
		self._widgets.extend(self._extract_widgets(layout))
		self._rebuild_content_layout()

	def _extract_widgets(self, layout) -> list[QWidget]:
		"""Extrae widgets de un layout (incluyendo sublayouts) conservando orden."""
		widgets: list[QWidget] = []
		while layout.count():
			item = layout.takeAt(0)
			w = item.widget()
			if w is not None:
				widgets.append(w)
				continue
			inner = item.layout()
			if inner is not None:
				widgets.extend(self._extract_widgets(inner))
		return widgets

	def show_near(self, widget: QWidget) -> None:
		"""Muestra la barra cerca del botón que la abrió (o en la última
		posición recordada, si el usuario ya la movió antes)."""
		if not self.isVisible():
			pos = self._settings.value(f"floating_toolbar/{self._key}/pos", None)
			target = pos if isinstance(pos, QPoint) else None
			# Si la posición guardada quedó fuera de las pantallas actuales
			# (típico al cambiar de monitor), se descarta y se reubica junto
			# al botón que la abre para que nunca aparezca inaccesible.
			if target is None or not self._point_on_screen(target):
				target = widget.mapToGlobal(widget.rect().bottomLeft())
			self.move(self._clamp_to_screen(target))
		self.show()
		self.raise_()
		self.activateWindow()

	def _point_on_screen(self, pos: QPoint) -> bool:
		"""True si el punto cae dentro del área visible de alguna pantalla."""
		for screen in QGuiApplication.screens():
			if screen.availableGeometry().contains(pos):
				return True
		return False

	def _clamp_to_screen(self, pos: QPoint) -> QPoint:
		"""Ajusta la esquina superior-izquierda para que la barra entera quede
		dentro del área visible de la pantalla más cercana al punto."""
		screen = QGuiApplication.screenAt(pos) or QGuiApplication.primaryScreen()
		if screen is None:
			return pos
		area = screen.availableGeometry()
		w = max(self.width(), self.minimumWidth())
		h = max(self.height(), self.minimumHeight())
		x = min(max(pos.x(), area.left()), max(area.left(), area.right() - w))
		y = min(max(pos.y(), area.top()), max(area.top(), area.bottom() - h))
		return QPoint(x, y)

	def toggle_near(self, widget: QWidget) -> None:
		if self.isVisible():
			self.hide()
		else:
			self.show_near(widget)

	def _toggle_orientation(self) -> None:
		self._orientation = (
			Qt.Orientation.Vertical if self._orientation == Qt.Orientation.Horizontal
			else Qt.Orientation.Horizontal
		)
		self._rebuild_content_layout()
		self._save_state()

	def _rebuild_content_layout(self) -> None:
		if self._rebuilding:
			return
		self._rebuilding = True
		old_layout = self._content_layout
		while old_layout.count():
			item = old_layout.takeAt(0)
			child_layout = item.layout()
			if child_layout is not None:
				child_layout.setParent(None)
		QWidget().setLayout(old_layout)

		new_layout = QGridLayout()
		new_layout.setContentsMargins(6, 4, 6, 6)
		new_layout.setSpacing(8)
		self._content_host.setLayout(new_layout)
		self._content_layout = new_layout

		widgets = list(self._widgets)

		if self._orientation == Qt.Orientation.Horizontal:
			rows = self._horizontal_rows
			row_layouts = []
			for row in range(rows):
				rl = QHBoxLayout()
				rl.setSpacing(6)
				row_layouts.append(rl)
				self._content_layout.addLayout(rl, row, 0)
			if widgets:
				for idx, w in enumerate(widgets):
					row_layouts[idx % rows].addWidget(w)
			for rl in row_layouts:
				rl.addStretch(1)
		else:
			for idx, w in enumerate(widgets):
				self._content_layout.addWidget(w, idx, 0)

		# Al alternar orientación, fuerza recálculo para evitar que quede
		# "pegado" el ancho de la orientación anterior.
		self._content_host.updateGeometry()
		self.layout().activate()
		self._update_minimum_size_for_readability()
		hint = self.sizeHint()
		self.resize(hint)
		self._rebuilding = False

	def _update_minimum_size_for_readability(self) -> None:
		"""Evita que la barra se achique hasta volver ilegibles los controles."""
		if self._orientation == Qt.Orientation.Vertical:
			max_w = int(self._content_host.sizeHint().width())
			total_h = int(self._content_host.sizeHint().height())
			self.setMinimumWidth(max(240, max_w + 18))
			self.setMinimumHeight(max(110, min(720, total_h + 54)))
			return

		min_w = max(300, int(self._content_host.minimumSizeHint().width()) + 14)
		min_h = max(110, int(self._content_host.minimumSizeHint().height()) + 56)
		self.setMinimumWidth(max(300, min_w))
		self.setMinimumHeight(max(110, min_h))

	def toggle_collapsed(self) -> None:
		"""Colapsa a solo el header (barra de arriba) o vuelve a expandir.

		Se evita `setFixedHeight` para no dejar la ventana "trabada" en un
		tamaño; en su lugar se libera el mínimo, se ajusta el máximo y se
		redimensiona al `sizeHint` correspondiente.
		"""
		self._collapsed = not self._collapsed
		self._content_host.setVisible(not self._collapsed)
		self._size_grip.setVisible(not self._collapsed)
		header = self.layout().itemAt(0).widget()
		if self._collapsed:
			header_h = header.sizeHint().height() + 2
			# Sin mínimo de contenido: la ventana puede achicarse al header.
			self.setMinimumSize(header.sizeHint().width() + 8, header_h)
			self.setMaximumHeight(header_h)
			self.resize(self.width(), header_h)
		else:
			self.setMaximumHeight(16777215)
			self._update_minimum_size_for_readability()
			self.resize(self.sizeHint())
		self._save_state()

	def _save_state(self) -> None:
		self._settings.setValue(f"floating_toolbar/{self._key}/pos", self.pos())
		self._settings.setValue(
			f"floating_toolbar/{self._key}/orientation",
			"v" if self._orientation == Qt.Orientation.Vertical else "h",
		)

	def _restore_state(self) -> None:
		# Solo se restaura orientación. El estado colapsado NO se persiste a
		# propósito: reabrir siempre expandida evita tamaños heredados raros.
		orient = self._settings.value(f"floating_toolbar/{self._key}/orientation", None)
		if orient == "v" and self._orientation != Qt.Orientation.Vertical:
			self._orientation = Qt.Orientation.Vertical
			self._rebuild_content_layout()
