"""SINCRO - ui.gqc_window

Panel de control de calidad del gating (GQC, *Gated Quality Control*).

QUÉ PROBLEMA RESUELVE
---------------------
Un gated SPECT puede dar números perfectamente "razonables" (FEVI, volúmenes,
fase) y estar arruinado por un problema de adquisición que no se ve en las
imágenes reconstruidas. Los tres clásicos son:

1. **Dropout del último gate.** Los latidos más cortos que la ventana de
   aceptación no llegan a llenar el último gate, que queda con menos cuentas.
2. **Arritmia / rechazo de latidos.** Si el paciente hace extrasístoles, la
   consola descarta latidos y algunos gates quedan con estadística distinta.
3. **Movimiento del paciente.** Se manifiesta como un salto en las cuentas de
   un rango de proyecciones consecutivas: el corazón se sale del campo o entra
   en la sombra de una estructura vecina.

La herramienta clásica para detectarlos es la que arma este panel: **las cuentas
totales de cada proyección, con una curva por gate**. En un estudio sano las
curvas son suaves, paralelas entre sí y están todas al mismo nivel. Cualquier
desviación de eso apunta directamente a la causa:

- una curva sistemáticamente por debajo de las demás → dropout de ese gate;
- un escalón en la misma posición angular en TODAS las curvas → movimiento;
- curvas que se cruzan y separan de forma errática → arritmia o rechazo;
- una caída suave y ancha → atenuación por una estructura (mama, diafragma).

SOBRE LA FUENTE DE DATOS
------------------------
Lo ideal es tener las **proyecciones crudas**, porque el eje X es entonces el
ángulo de adquisición y el análisis es el de la consola. Si el estudio que está
cargado ya viene reconstruido no hay proyecciones, así que el panel cae a un
sustituto: cuentas por **corte** de eje corto, con una curva por gate. Sirve
igual para ver el dropout y la homogeneidad entre gates, pero NO puede detectar
movimiento durante la rotación, porque esa información se perdió al
reconstruir. La ventana lo dice explícitamente para que no se confunda una cosa
con la otra.

USO
---
Se abre desde el panel lateral. Es una ventana no modal: se puede dejar abierta
mientras se trabaja, y se refresca sola al reprocesar.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
	QCheckBox,
	QComboBox,
	QDialog,
	QDoubleSpinBox,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QScrollArea,
	QSizePolicy,
	QVBoxLayout,
	QWidget,
)

from core.gate_dropout import analyze_gate_dropout
from ui.collapsible import CollapsibleSection

#: Paleta de hasta 16 gates. Se eligieron tonos separados en matiz y en
#: luminancia para que se distingan también en una impresión en escala de grises.
GATE_COLORS = [
	"#e6194b", "#3cb44b", "#4363d8", "#f58231",
	"#911eb4", "#008080", "#9a6324", "#800000",
	"#808000", "#000075", "#f032e6", "#46f0f0",
	"#bcf60c", "#fabebe", "#aaffc3", "#808080",
]


class GateCurvesWidget(QWidget):
	"""Curvas de cuentas por proyección (o por corte), una por gate.

	Se dibuja con QPainter y no con matplotlib porque el panel se repinta con
	cada cambio de opción: un canvas de matplotlib metería decenas de
	milisegundos por refresco y el usuario lo notaría.
	"""

	MARGIN_LEFT = 58
	MARGIN_RIGHT = 12
	MARGIN_TOP = 12
	MARGIN_BOTTOM = 30

	def __init__(self, parent=None):
		super().__init__(parent)
		self._curves: np.ndarray = np.zeros((0, 0))
		self._x_label = ""
		self._y_label = ""
		self._x_values: np.ndarray = np.zeros(0)
		self._highlight = -1
		self.setMinimumHeight(240)
		self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

	def set_curves(self, curves, x_values, x_label: str, y_label: str):
		self._curves = np.asarray(curves, dtype=np.float64)
		self._x_values = np.asarray(x_values, dtype=np.float64)
		self._x_label = x_label
		self._y_label = y_label
		self.update()

	def set_highlight(self, gate_index: int):
		"""Resalta un gate (índice 0-based) y apaga el resto. -1 = todos iguales."""
		self._highlight = int(gate_index)
		self.update()

	def paintEvent(self, event):  # noqa: N802 - API de Qt
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
		painter.fillRect(self.rect(), QColor("#ffffff"))

		rect = self.rect().adjusted(
			self.MARGIN_LEFT, self.MARGIN_TOP, -self.MARGIN_RIGHT, -self.MARGIN_BOTTOM
		)
		painter.setPen(QPen(QColor("#c9d2e0"), 1))
		painter.drawRect(rect)

		curves = self._curves
		if curves.ndim != 2 or curves.shape[1] < 2 or not np.isfinite(curves).all():
			painter.setPen(QColor("#8a8a8a"))
			painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Sin datos para graficar")
			painter.end()
			return

		lo = float(curves.min())
		hi = float(curves.max())
		if hi <= lo:
			hi = lo + 1.0
		# Un poco de aire arriba y abajo para que las curvas no toquen el marco.
		pad = 0.06 * (hi - lo)
		lo -= pad
		hi += pad

		n_x = curves.shape[1]
		x_vals = self._x_values if self._x_values.size == n_x else np.arange(n_x, dtype=np.float64)

		def to_px(i: int, value: float):
			x = rect.left() + rect.width() * (i / max(1, n_x - 1))
			y = rect.bottom() - rect.height() * ((value - lo) / (hi - lo))
			return x, y

		# --- grilla horizontal con etiquetas ---
		painter.setFont(QFont("Segoe UI", 7))
		for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
			value = lo + frac * (hi - lo)
			y = rect.bottom() - rect.height() * frac
			painter.setPen(QPen(QColor("#eef1f6"), 1))
			painter.drawLine(int(rect.left()) + 1, int(y), int(rect.right()) - 1, int(y))
			painter.setPen(QColor("#6b7280"))
			painter.drawText(2, int(y) + 4, f"{value:,.0f}".replace(",", "."))

		# --- curvas ---
		n_gates = curves.shape[0]
		for g in range(n_gates):
			color = QColor(GATE_COLORS[g % len(GATE_COLORS)])
			if self._highlight >= 0 and g != self._highlight:
				color.setAlpha(60)
				width = 1
			else:
				width = 2 if self._highlight >= 0 else 1.6
			path = QPainterPath()
			for i in range(n_x):
				x, y = to_px(i, float(curves[g, i]))
				if i == 0:
					path.moveTo(x, y)
				else:
					path.lineTo(x, y)
			painter.setPen(QPen(color, width))
			painter.drawPath(path)

		# --- ejes ---
		painter.setPen(QColor("#6b7280"))
		painter.setFont(QFont("Segoe UI", 7))
		painter.drawText(
			rect.left(), rect.bottom() + 16, rect.width(), 14,
			Qt.AlignmentFlag.AlignCenter, self._x_label,
		)
		if x_vals.size >= 2:
			painter.drawText(rect.left() - 4, rect.bottom() + 16, 40, 14,
			                 Qt.AlignmentFlag.AlignLeft, f"{x_vals[0]:.0f}")
			painter.drawText(rect.right() - 36, rect.bottom() + 16, 40, 14,
			                 Qt.AlignmentFlag.AlignRight, f"{x_vals[-1]:.0f}")
		painter.save()
		painter.translate(12, rect.center().y())
		painter.rotate(-90)
		painter.drawText(-60, 0, 120, 12, Qt.AlignmentFlag.AlignCenter, self._y_label)
		painter.restore()
		painter.end()


class GateLegend(QWidget):
	"""Leyenda de colores por gate, clickeable para resaltar uno."""

	def __init__(self, on_pick, parent=None):
		super().__init__(parent)
		self._on_pick = on_pick
		self._layout = QHBoxLayout(self)
		self._layout.setContentsMargins(0, 0, 0, 0)
		self._layout.setSpacing(3)
		self._buttons: list[QPushButton] = []
		self._selected = -1

	def rebuild(self, n_gates: int):
		while self._layout.count():
			item = self._layout.takeAt(0)
			widget = item.widget()
			if widget is not None:
				widget.deleteLater()
		self._buttons = []
		self._selected = -1

		todos = QPushButton("Todos")
		todos.setCheckable(True)
		todos.setChecked(True)
		todos.setMaximumWidth(52)
		todos.clicked.connect(lambda: self._pick(-1))
		self._layout.addWidget(todos)
		self._buttons.append(todos)

		for g in range(n_gates):
			color = GATE_COLORS[g % len(GATE_COLORS)]
			btn = QPushButton(str(g + 1))
			btn.setCheckable(True)
			btn.setMaximumWidth(30)
			btn.setStyleSheet(
				f"QPushButton {{ border: 2px solid {color}; border-radius: 3px; padding: 1px; }}"
				f"QPushButton:checked {{ background: {color}; color: white; font-weight: 600; }}"
			)
			btn.clicked.connect(lambda _c=False, idx=g: self._pick(idx))
			self._layout.addWidget(btn)
			self._buttons.append(btn)
		self._layout.addStretch(1)

	def _pick(self, index: int):
		self._selected = index
		for i, btn in enumerate(self._buttons):
			btn.setChecked((i - 1) == index)
		self._on_pick(index)


class GQCWindow(QDialog):
	"""Panel de control de calidad del gating. No modal y con refresco en vivo."""

	DEBOUNCE_MS = 120

	def __init__(self, main_window, parent=None):
		super().__init__(parent or main_window)
		self._main = main_window
		self._syncing = False
		self._counts: np.ndarray = np.zeros((0, 0))
		self._source_is_raw = False

		self.setWindowTitle("SINCRO — Control de calidad del gating (GQC)")
		self.setModal(False)
		self.setMinimumSize(620, 600)
		self.resize(720, 780)

		self._recalc_timer = QTimer(self)
		self._recalc_timer.setSingleShot(True)
		self._recalc_timer.timeout.connect(self.recompute)

		self._build_ui()

	# ------------------------------------------------------------------
	# Construcción
	# ------------------------------------------------------------------

	def _build_ui(self):
		outer = QVBoxLayout(self)
		outer.setContentsMargins(8, 8, 8, 8)
		outer.setSpacing(6)

		scroll = QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setFrameShape(QScrollArea.Shape.NoFrame)
		container = QWidget()
		layout = QVBoxLayout(container)
		layout.setContentsMargins(2, 2, 2, 2)
		layout.setSpacing(6)
		scroll.setWidget(container)
		outer.addWidget(scroll, 1)

		container.setStyleSheet(
			"QGroupBox { font-weight: 600; border: 1px solid #d7dce5; border-radius: 7px;"
			" margin-top: 6px; background: white; }"
			"QGroupBox::title { subcontrol-origin: margin; left: 7px; padding: 0 2px; color: #1f3b5b; }"
			"QGroupBox#collapsibleContent { font-weight: 400; border: 1px solid #d7dce5; border-top: none;"
			" border-top-left-radius: 0; border-top-right-radius: 0; margin-top: 0; }"
			"QToolButton#collapsibleHeader { text-align: left; padding: 5px 8px; font-weight: 600;"
			" color: #1f3b5b; background: #eaeff7; border: 1px solid #d7dce5; border-radius: 7px; margin-top: 6px; }"
			"QToolButton#collapsibleHeader:hover { background: #dde6f4; }"
			"QToolButton#collapsibleHeader:checked { background: #e3ebf7;"
			" border-bottom-left-radius: 0; border-bottom-right-radius: 0; }"
			"QLabel { font-size: 11px; }"
		)

		layout.addWidget(self._build_intro_section())
		layout.addWidget(self._build_source_box())
		layout.addWidget(self._build_options_box())
		layout.addWidget(self._build_curves_box())
		layout.addWidget(self._build_findings_box())
		layout.addStretch(1)

		outer.addLayout(self._build_footer())

	def _build_intro_section(self) -> CollapsibleSection:
		box = QGroupBox("Cómo se lee este panel")
		box_layout = QVBoxLayout(box)
		box_layout.setContentsMargins(8, 8, 8, 8)
		text = QLabel(
			"Cada curva son las <b>cuentas totales de cada proyección para un gate</b>. En un estudio "
			"bien adquirido las curvas salen suaves, paralelas y superpuestas: todos los gates "
			"recibieron la misma estadística y el paciente no se movió.<br><br>"
			"<b>Qué buscar:</b><br>"
			"• <b>Una curva por debajo de las demás</b> (casi siempre la del último gate) → <i>dropout</i>: "
			"los latidos cortos no llegaron a llenar ese gate. Se corrige con el escalado del panel principal.<br>"
			"• <b>Un escalón en la misma posición angular en TODAS las curvas</b> → el paciente se movió "
			"en ese tramo de la rotación.<br>"
			"• <b>Curvas que se cruzan de forma errática</b> → arritmia o rechazo de latidos: la estadística "
			"por gate quedó despareja y la fase deja de ser confiable.<br>"
			"• <b>Una caída suave y ancha</b> → atenuación por una estructura vecina (mama, diafragma), "
			"no un problema del gating.<br><br>"
			"<b>Lo que este panel NO puede decir:</b> si el estudio cargado ya viene reconstruido no hay "
			"proyecciones, y entonces el eje horizontal pasa a ser el número de corte. Con eso se ve el "
			"dropout y la homogeneidad entre gates, pero <b>no</b> el movimiento durante la rotación: esa "
			"información se pierde al reconstruir. El recuadro de arriba siempre aclara con qué fuente "
			"se está trabajando."
		)
		text.setWordWrap(True)
		text.setTextFormat(Qt.TextFormat.RichText)
		text.setStyleSheet("color:#35506a; line-height:1.3;")
		box_layout.addWidget(text)
		return CollapsibleSection.from_group_box(box, key="gqc_intro", expanded=False)

	def _build_source_box(self) -> QGroupBox:
		box = QGroupBox("Fuente de datos")
		box_layout = QVBoxLayout(box)
		box_layout.setContentsMargins(8, 8, 8, 8)
		self.source_label = QLabel("—")
		self.source_label.setWordWrap(True)
		self.source_label.setStyleSheet("color:#1f3b5b; font-weight:600;")
		box_layout.addWidget(self.source_label)
		self.source_warning = QLabel("")
		self.source_warning.setWordWrap(True)
		self.source_warning.setStyleSheet("color:#a06000;")
		box_layout.addWidget(self.source_warning)
		return box

	def _build_options_box(self) -> QGroupBox:
		box = QGroupBox("Opciones (se recalcula en vivo)")
		form = QFormLayout(box)
		form.setContentsMargins(8, 8, 8, 8)
		form.setSpacing(5)

		self.normalize_check = QCheckBox("Normalizar cada gate a su promedio")
		self.normalize_check.setToolTip(
			"Divide cada curva por su propio promedio y la expresa en %.\n"
			"Sirve para separar dos cosas que se confunden: si al normalizar las curvas se superponen,\n"
			"la diferencia entre gates era solo de nivel (dropout). Si siguen distintas en FORMA,\n"
			"el problema es otro (movimiento, arritmia, atenuación variable)."
		)

		self.band_check = QCheckBox("Contar solo una banda central de filas")
		self.band_check.setToolTip(
			"Restringe el conteo a una franja horizontal centrada en la imagen.\n"
			"El hígado y el intestino suelen tener más cuentas que el corazón y dominan el total,\n"
			"tapando lo que le pasa al miocardio. Acotando la banda se ve el corazón."
		)

		self.band_frac_spin = QDoubleSpinBox()
		self.band_frac_spin.setRange(0.10, 1.00)
		self.band_frac_spin.setSingleStep(0.05)
		self.band_frac_spin.setDecimals(2)
		self.band_frac_spin.setValue(0.50)
		self.band_frac_spin.setEnabled(False)
		self.band_frac_spin.setToolTip(
			"Alto de la banda como fracción de la imagen. 0.50 = la mitad central."
		)

		self.smooth_combo = QComboBox()
		for label, value in (("sin suavizado", 0), ("3 puntos", 3), ("5 puntos", 5)):
			self.smooth_combo.addItem(label, value)
		self.smooth_combo.setToolTip(
			"Media móvil sobre el eje horizontal, solo para la vista.\n"
			"Ayuda a distinguir un escalón real de la fluctuación estadística, pero un suavizado\n"
			"grande puede disimular un salto corto: si dudás, mirá también sin suavizar."
		)

		form.addRow(self.normalize_check)
		form.addRow(self.band_check)
		form.addRow("Alto de la banda", self.band_frac_spin)
		form.addRow("Suavizado de la vista", self.smooth_combo)

		self.normalize_check.toggled.connect(self._schedule_recompute)
		self.band_check.toggled.connect(self._on_band_toggled)
		self.band_frac_spin.valueChanged.connect(self._schedule_recompute)
		self.smooth_combo.currentIndexChanged.connect(self._schedule_recompute)
		return box

	def _on_band_toggled(self, checked: bool):
		self.band_frac_spin.setEnabled(bool(checked))
		self._schedule_recompute()

	def _build_curves_box(self) -> QGroupBox:
		box = QGroupBox("Cuentas por proyección, una curva por gate")
		box_layout = QVBoxLayout(box)
		box_layout.setContentsMargins(8, 8, 8, 8)
		box_layout.setSpacing(4)

		self.curves = GateCurvesWidget()
		box_layout.addWidget(self.curves, 1)

		self.legend = GateLegend(self._on_gate_picked)
		box_layout.addWidget(self.legend)

		hint = QLabel("Clic en un número para aislar ese gate y compararlo contra el resto.")
		hint.setStyleSheet("color:#6b7280;")
		box_layout.addWidget(hint)
		return box

	def _on_gate_picked(self, index: int):
		self.curves.set_highlight(index)

	def _build_findings_box(self) -> QGroupBox:
		box = QGroupBox("Lectura automática")
		box_layout = QVBoxLayout(box)
		box_layout.setContentsMargins(8, 8, 8, 8)
		box_layout.setSpacing(4)
		self.findings_label = QLabel("—")
		self.findings_label.setWordWrap(True)
		self.findings_label.setTextFormat(Qt.TextFormat.RichText)
		box_layout.addWidget(self.findings_label)
		nota = QLabel(
			"Son indicios calculados sobre las curvas, no un diagnóstico: confirmalos mirando el gráfico."
		)
		nota.setWordWrap(True)
		nota.setStyleSheet("color:#6b7280;")
		box_layout.addWidget(nota)
		return box

	def _build_footer(self) -> QHBoxLayout:
		row = QHBoxLayout()
		self.status_label = QLabel("")
		self.status_label.setStyleSheet("color:#4b7a4b;")
		row.addWidget(self.status_label, 1)

		refresh_btn = QPushButton("Recalcular")
		refresh_btn.clicked.connect(self.recompute)
		row.addWidget(refresh_btn)

		close_btn = QPushButton("Cerrar")
		close_btn.clicked.connect(self.close)
		row.addWidget(close_btn)
		return row

	# ------------------------------------------------------------------
	# Cálculo
	# ------------------------------------------------------------------

	def _schedule_recompute(self):
		if self._syncing:
			return
		self._recalc_timer.start(self.DEBOUNCE_MS)

	def showEvent(self, event):  # noqa: N802 - API de Qt
		super().showEvent(event)
		self.recompute()

	def _resolve_source(self):
		"""Elige la mejor fuente disponible: proyecciones crudas > cubo reconstruido.

		Returns
		-------
		(frames, es_crudo, descripcion) o (None, False, motivo)
		"""
		main = self._main
		# 1) Un crudo que quedó guardado aunque después se haya reconstruido.
		raw_study = getattr(main, "cine_crudo_raw_study_for_recon", None)
		if raw_study is not None and getattr(raw_study, "cube", None) is not None:
			return np.asarray(raw_study.cube, dtype=np.float64), True, "crudo retenido"

		study = getattr(main, "study", None)
		if study is None or getattr(study, "cube", None) is None:
			return None, False, "No hay ningún estudio cargado."

		# 2) El estudio actual, si vino crudo.
		if not bool(getattr(study, "reconstructed", True)):
			return np.asarray(study.cube, dtype=np.float64), True, "estudio crudo cargado"

		# 3) Reconstruido: sustituto por cortes.
		return np.asarray(study.cube, dtype=np.float64), False, "cubo reconstruido"

	def recompute(self):
		"""Recalcula las curvas con las opciones actuales."""
		frames, is_raw, detail = self._resolve_source()
		if frames is None:
			self._show_unavailable(detail)
			return
		if frames.ndim != 4 or frames.shape[0] < 2:
			self._show_unavailable(
				f"Se necesita un cubo gatillado 4D con al menos 2 gates; se encontró {frames.shape}."
			)
			return

		self._source_is_raw = bool(is_raw)
		n_gates, n_x = int(frames.shape[0]), int(frames.shape[1])

		if is_raw:
			self.source_label.setText(
				f"Proyecciones crudas ({detail}): {n_gates} gates × {n_x} ángulos."
			)
			self.source_warning.setText("")
			x_label = "ángulo de proyección"
		else:
			self.source_label.setText(
				f"Cubo reconstruido: {n_gates} gates × {n_x} cortes de eje corto."
			)
			self.source_warning.setText(
				"No hay proyecciones crudas cargadas, así que el eje horizontal son los cortes. "
				"Se puede evaluar el dropout y la homogeneidad entre gates, pero NO el movimiento "
				"durante la rotación."
			)
			x_label = "corte de eje corto"

		counts = self._counts_from(frames)
		self._counts = counts

		display = counts.copy()
		y_label = "cuentas totales"
		if self.normalize_check.isChecked():
			means = display.mean(axis=1, keepdims=True)
			means[means <= 0.0] = 1.0
			display = display / means * 100.0
			y_label = "% del promedio del gate"

		kernel = int(self.smooth_combo.currentData() or 0)
		if kernel >= 3:
			display = self._moving_average(display, kernel)

		self.curves.set_curves(display, np.arange(counts.shape[1]), x_label, y_label)
		self.legend.rebuild(n_gates)
		self.curves.set_highlight(-1)
		self._render_findings(frames, counts)
		self.status_label.setText("Actualizado.")
		self.status_label.setStyleSheet("color:#4b7a4b;")

	def _counts_from(self, frames: np.ndarray) -> np.ndarray:
		"""Cuentas totales por (gate, posición), con la banda central opcional."""
		if self.band_check.isChecked():
			rows = frames.shape[2]
			half = max(1, int(round(rows * float(self.band_frac_spin.value()) / 2.0)))
			center = rows // 2
			lo = max(0, center - half)
			hi = min(rows, center + half)
			frames = frames[:, :, lo:hi, :]
		return frames.sum(axis=(2, 3))

	@staticmethod
	def _moving_average(curves: np.ndarray, kernel: int) -> np.ndarray:
		"""Media móvil sobre el eje horizontal, con bordes replicados.

		Se replican los bordes en vez de envolver de forma circular: aunque las
		proyecciones cubren un arco, el primer y el último ángulo no son vecinos
		en el tiempo de adquisición, y envolverlos inventaría continuidad.
		"""
		k = int(kernel)
		if k < 3 or curves.shape[1] < k:
			return curves
		pad = k // 2
		padded = np.pad(curves, ((0, 0), (pad, pad)), mode="edge")
		window = np.ones(k, dtype=np.float64) / float(k)
		return np.apply_along_axis(lambda row: np.convolve(row, window, mode="valid"), 1, padded)

	def _render_findings(self, frames: np.ndarray, counts: np.ndarray):
		"""Indicios cuantitativos derivados de las curvas."""
		lines: list[str] = []
		totals = counts.sum(axis=1)
		n_gates = totals.size

		# --- dropout del último gate (mismo criterio que la corrección ECTb) ---
		info = analyze_gate_dropout(frames)
		dropout = float(info.get("dropout_pct", 0.0))
		if info.get("significant"):
			lines.append(
				f"<b>Dropout del último gate: {dropout:.1f}%.</b> El gate {n_gates} tiene esa cantidad "
				"menos de cuentas que el gate 1. Conviene dejar activa la corrección del panel principal."
			)
		else:
			lines.append(
				f"Dropout del último gate: {dropout:.1f}% — dentro del ruido estadístico, sin hallazgo."
			)

		# --- dispersión de cuentas entre gates ---
		mean_total = float(totals.mean())
		if mean_total > 0.0:
			cv = float(totals.std() / mean_total * 100.0)
			peor = int(np.argmin(totals))
			if cv > 5.0:
				lines.append(
					f"<b>Dispersión entre gates: {cv:.1f}%.</b> Es alta: el gate con menos estadística es "
					f"el {peor + 1}, con {totals[peor] / mean_total * 100.0:.0f}% del promedio. "
					"Sugiere rechazo de latidos o arritmia durante la adquisición."
				)
			else:
				lines.append(f"Dispersión entre gates: {cv:.1f}% — los gates recibieron estadística pareja.")

		# --- salto común a todos los gates (movimiento) ---
		if self._source_is_raw and counts.shape[1] >= 5:
			# Se busca el salto en la curva PROMEDIO: si el escalón está en todos
			# los gates a la vez, el paciente se movió; si está en uno solo, es del gating.
			mean_curve = counts.mean(axis=0)
			diffs = np.abs(np.diff(mean_curve))
			mad = float(np.median(np.abs(diffs - np.median(diffs))))
			umbral = float(np.median(diffs)) + 6.0 * (mad if mad > 0 else 1.0)
			picos = np.where(diffs > umbral)[0]
			if picos.size:
				donde = ", ".join(str(int(p)) for p in picos[:6])
				lines.append(
					f"<b>Salto brusco de cuentas entre proyecciones consecutivas ({donde}).</b> "
					"Cuando aparece en todos los gates a la vez suele ser movimiento del paciente: "
					"revisalo con el sinograma y la corrección de movimiento."
				)
			else:
				lines.append("Sin saltos bruscos entre proyecciones consecutivas.")
		elif not self._source_is_raw:
			lines.append(
				"<i>Detección de movimiento no disponible: requiere las proyecciones crudas.</i>"
			)

		self.findings_label.setText("<br><br>".join(lines))

	def _show_unavailable(self, message: str):
		self._counts = np.zeros((0, 0))
		self.curves.set_curves(np.zeros((0, 0)), np.zeros(0), "", "")
		self.legend.rebuild(0)
		self.source_label.setText("—")
		self.source_warning.setText("")
		self.findings_label.setText("—")
		self.status_label.setText(message)
		self.status_label.setStyleSheet("color:#a06000;")
