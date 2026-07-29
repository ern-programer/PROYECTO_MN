"""SINCRO - ui.collapsible.

Secciones colapsables para el panel lateral (sidebar) de GammaSync.

PROBLEMA QUE RESUELVE
---------------------
El sidebar acumula muchas cajas de opciones (`QGroupBox`) siempre visibles.
Eso obliga a scrollear mucho y satura visualmente la lectura clínica.

SOLUCIÓN
--------
`CollapsibleSection` envuelve un widget de contenido y convierte su **título
en un botón**: un click expande/colapsa el contenido. El estado se puede
guardar y restaurar (ver `MainWindow._install_collapsible_sidebar_sections`).

RENDIMIENTO
-----------
No hay animaciones ni transiciones: el toggle es un `setVisible()` directo
envuelto en `setUpdatesEnabled(False)` para evitar repintados intermedios.
Es la variante más barata posible; no introduce lag perceptible aunque la
sección contenga decenas de controles.

USO TÍPICO
----------
    # a partir de un QGroupBox ya construido (toma su título automáticamente)
    section = CollapsibleSection.from_group_box(mi_group_box, expanded=False)
    layout.addWidget(section)

    # o con cualquier widget y un título explícito
    section = CollapsibleSection("Procesamiento", mi_widget)

    section.set_expanded(True)          # abrir por código
    section.toggled.connect(on_toggle)  # enterarse de los cambios
    section.key                         # clave estable para persistir estado
"""
from __future__ import annotations

import re

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
	QGroupBox,
	QSizePolicy,
	QToolButton,
	QVBoxLayout,
	QWidget,
)


def slugify_section_key(text: str) -> str:
	"""Devuelve una clave estable y segura para guardar el estado en QSettings.

	Pasa a minúsculas, reemplaza todo lo que no sea alfanumérico por `_` y
	colapsa repeticiones: ``"ROI manual por slice"`` -> ``"roi_manual_por_slice"``.
	Se usa como sufijo de la clave de QSettings, por eso conviene que no tenga
	espacios, acentos ni barras.
	"""
	normalized = (
		str(text)
		.strip()
		.lower()
		.replace("á", "a")
		.replace("é", "e")
		.replace("í", "i")
		.replace("ó", "o")
		.replace("ú", "u")
		.replace("ñ", "n")
	)
	normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
	return normalized or "seccion"


class CollapsibleSection(QWidget):
	"""Caja de opciones con encabezado clickeable que expande/colapsa.

	Parámetros
	----------
	title:
		Texto del encabezado. Es también el botón que dispara el toggle.
	content:
		Widget que se muestra u oculta. Queda reparentado dentro de la sección.
	key:
		Clave estable para persistir el estado. Si se omite se deriva del título.
	expanded:
		Estado inicial. `False` arranca colapsado (solo se ve el encabezado).
	tooltip:
		Ayuda extra que se agrega al tooltip del encabezado.

	Señal
	-----
	toggled(bool):
		Emitida cuando cambia el estado (True = expandido). Útil para persistir.
	"""

	toggled = pyqtSignal(bool)

	#: Flechas usadas en el encabezado según el estado.
	ARROW_EXPANDED = "▾"
	ARROW_COLLAPSED = "▸"

	def __init__(
		self,
		title: str,
		content: QWidget,
		key: str = "",
		expanded: bool = True,
		tooltip: str = "",
		parent: QWidget | None = None,
	) -> None:
		super().__init__(parent)
		self._title = str(title)
		self.key = str(key) or slugify_section_key(self._title)
		self._content = content

		self.header = QToolButton(self)
		self.header.setObjectName("collapsibleHeader")
		self.header.setCheckable(True)
		self.header.setChecked(bool(expanded))
		self.header.setCursor(Qt.CursorShape.PointingHandCursor)
		self.header.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
		self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
		base_tip = f"«{self._title}» — click en el título para expandir o colapsar esta sección."
		self.header.setToolTip(f"{base_tip}\n{tooltip}" if tooltip else base_tip)
		self.header.toggled.connect(self._on_header_toggled)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(0)
		layout.addWidget(self.header)
		if content is not None:
			content.setParent(self)
			layout.addWidget(content)

		self._sync_header_text()
		if content is not None:
			content.setVisible(bool(expanded))

	# ------------------------------------------------------------------ API

	@classmethod
	def from_group_box(
		cls,
		box: QGroupBox,
		key: str = "",
		expanded: bool = True,
		parent: QWidget | None = None,
	) -> "CollapsibleSection":
		"""Envuelve un `QGroupBox` existente reutilizando su título.

		El título del `QGroupBox` se vacía (para no duplicarlo con el
		encabezado) y se le asigna `objectName == "collapsibleContent"`, que la
		hoja de estilos del sidebar usa para dibujarlo sin marco propio.
		El tooltip del box, si lo tenía, se hereda al encabezado.
		"""
		title = box.title().strip()
		tooltip = box.toolTip()
		box.setTitle("")
		box.setObjectName("collapsibleContent")
		return cls(
			title,
			box,
			key=key or slugify_section_key(title),
			expanded=expanded,
			tooltip=tooltip,
			parent=parent,
		)

	@property
	def title(self) -> str:
		"""Título original de la sección (sin la flecha del encabezado)."""
		return self._title

	@property
	def content(self) -> QWidget | None:
		"""Widget de contenido envuelto por la sección."""
		return self._content

	def is_expanded(self) -> bool:
		"""True si el contenido está visible."""
		return bool(self.header.isChecked())

	def set_expanded(self, expanded: bool) -> None:
		"""Expande o colapsa por código (emite `toggled` si hay cambio)."""
		self.header.setChecked(bool(expanded))

	def toggle(self) -> None:
		"""Invierte el estado actual."""
		self.set_expanded(not self.is_expanded())

	def set_badge(self, text: str) -> None:
		"""Agrega un indicador corto al encabezado (ej: cantidad de cambios).

		Pasar cadena vacía lo quita. Sirve para avisar que una sección colapsada
		tiene valores modificados sin obligar a abrirla.
		"""
		self._badge = str(text or "")
		self._sync_header_text()

	# -------------------------------------------------------------- interno

	def _on_header_toggled(self, checked: bool) -> None:
		# setUpdatesEnabled evita el repintado intermedio del relayout: el
		# cambio se ve como un salto instantáneo, sin parpadeo ni lag.
		self.setUpdatesEnabled(False)
		try:
			if self._content is not None:
				self._content.setVisible(bool(checked))
			self._sync_header_text()
		finally:
			self.setUpdatesEnabled(True)
		self.toggled.emit(bool(checked))

	def _sync_header_text(self) -> None:
		arrow = self.ARROW_EXPANDED if self.header.isChecked() else self.ARROW_COLLAPSED
		badge = getattr(self, "_badge", "")
		suffix = f"   {badge}" if badge else ""
		self.header.setText(f"{arrow}  {self._title}{suffix}")
