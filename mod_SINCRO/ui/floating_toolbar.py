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
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget


class _DragHeader(QWidget):
	"""Franja superior de la barra: al arrastrarla desde una zona libre (no
	sobre los botones) mueve toda la ventana flotante."""

	def __init__(self, toolbar: "FloatingToolbar"):
		super().__init__(toolbar)
		self._toolbar = toolbar
		self._drag_offset: QPoint | None = None

	def mousePressEvent(self, event):
		if event.button() == Qt.MouseButton.LeftButton:
			self._drag_offset = event.globalPosition().toPoint() - self._toolbar.frameGeometry().topLeft()
			event.accept()

	def mouseMoveEvent(self, event):
		if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
			self._toolbar.move(event.globalPosition().toPoint() - self._drag_offset)
			event.accept()

	def mouseReleaseEvent(self, event):
		self._drag_offset = None
		self._toolbar._save_state()


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

		self.setStyleSheet(
			"FloatingToolbar { background: #b8b8b8; border: 1px solid #888; border-radius: 4px; }"
			" FloatingToolbar QLabel { color: #222; }"
		)

		outer = QVBoxLayout(self)
		outer.setContentsMargins(1, 1, 1, 1)
		outer.setSpacing(0)

		header = _DragHeader(self)
		header_layout = QHBoxLayout(header)
		header_layout.setContentsMargins(6, 2, 2, 2)
		header_layout.setSpacing(2)
		header_layout.addWidget(QLabel(title))
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
		self._content_layout = QHBoxLayout(self._content_host)
		self._content_layout.setContentsMargins(6, 4, 6, 6)
		self._content_layout.setSpacing(8)
		outer.addWidget(self._content_host)

		self._restore_state()

	def add_layout(self, layout) -> None:
		"""Agrega un QHBoxLayout/QGridLayout ya armado con sus widgets."""
		self._groups.append(layout)
		self._content_layout.addLayout(layout)

	def show_near(self, widget: QWidget) -> None:
		"""Muestra la barra cerca del botón que la abrió (o en la última
		posición recordada, si el usuario ya la movió antes)."""
		if not self.isVisible():
			pos = self._settings.value(f"floating_toolbar/{self._key}/pos", None)
			if pos is not None:
				self.move(pos)
			else:
				self.move(widget.mapToGlobal(widget.rect().bottomLeft()))
		self.show()
		self.raise_()
		self.activateWindow()

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
		old_layout = self._content_layout
		while old_layout.count():
			item = old_layout.takeAt(0)
			child = item.layout()
			if child is not None:
				child.setParent(None)
		QWidget().setLayout(old_layout)

		new_layout = (
			QHBoxLayout() if self._orientation == Qt.Orientation.Horizontal else QVBoxLayout()
		)
		new_layout.setContentsMargins(6, 4, 6, 6)
		new_layout.setSpacing(8)
		self._content_host.setLayout(new_layout)
		self._content_layout = new_layout
		for group in self._groups:
			self._content_layout.addLayout(group)
		self.adjustSize()

	def _save_state(self) -> None:
		self._settings.setValue(f"floating_toolbar/{self._key}/pos", self.pos())
		self._settings.setValue(
			f"floating_toolbar/{self._key}/orientation",
			"v" if self._orientation == Qt.Orientation.Vertical else "h",
		)

	def _restore_state(self) -> None:
		orient = self._settings.value(f"floating_toolbar/{self._key}/orientation", None)
		if orient == "v" and self._orientation != Qt.Orientation.Vertical:
			self._orientation = Qt.Orientation.Vertical
			self._rebuild_content_layout()
