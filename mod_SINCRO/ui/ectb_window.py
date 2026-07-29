"""SINCRO - ui.ectb_window

Ventana de **cuantificación avanzada** basada en el Emory Cardiac Toolbox 4.0.

Vive aparte de la ventana principal a propósito: son opciones de análisis que no
hacen falta en el flujo de todos los días y que, metidas en el panel lateral,
lo saturarían. Se abre desde el botón "Cuantificación ECTb" de la sección
Acciones y se puede dejar abierta al costado mientras se trabaja (es una
ventana no modal).

DISEÑO PARA QUE NO HAYA LAG
---------------------------
- Todos los controles recalculan **en vivo**, pero pasando por un temporizador
  de rebote: mover un spin box diez veces seguidas dispara un solo cálculo.
- El cálculo está vectorizado (`scipy.ndimage.map_coordinates` en un solo paso
  por gate), así que el ciclo completo son milisegundos. El tiempo real medido
  se muestra abajo para que se note si algún parámetro lo encarece.
- Nada de animaciones ni de re-render de imágenes: la curva de volumen se
  dibuja con QPainter directo.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
	QAbstractSpinBox,
	QButtonGroup,
	QCheckBox,
	QComboBox,
	QDialog,
	QDoubleSpinBox,
	QFormLayout,
	QGridLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QRadioButton,
	QScrollArea,
	QSizePolicy,
	QSpinBox,
	QVBoxLayout,
	QWidget,
)

from core.ectb_lv import (
	EF_REGRESSIONS,
	ECTbLVConfig,
	ECTbLVResult,
	analyze_lv_ectb,
	convert_ef_pct,
	regression_equation_text,
)
from ui.collapsible import CollapsibleSection
from ui.wall_ruler import WallThicknessRuler


class VolumeCurveWidget(QWidget):
	"""Curva de volumen ventricular por gate, dibujada con QPainter.

	Se usa QPainter en vez de matplotlib porque este widget se repinta en cada
	cambio de parámetro: un canvas de matplotlib acá metería decenas de
	milisegundos por refresco y se notaría el lag.
	"""

	def __init__(self, parent=None):
		super().__init__(parent)
		self._volumes: np.ndarray = np.zeros(0)
		self._ed_gate = 0
		self._es_gate = 0
		self.setMinimumHeight(170)
		self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		self.setToolTip(
			"Volumen de la cavidad en cada gate del ciclo.\n"
			"El punto más alto es telediástole (ED) y el más bajo, telesístole (ES).\n"
			"Una curva suave y con un solo mínimo indica un gating limpio."
		)

	def set_curve(self, volumes, ed_gate: int, es_gate: int):
		self._volumes = np.asarray(volumes, dtype=np.float64)
		self._ed_gate = int(ed_gate)
		self._es_gate = int(es_gate)
		self.update()

	def paintEvent(self, event):  # noqa: N802 - API de Qt
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
		rect = self.rect().adjusted(38, 12, -12, -24)
		painter.fillRect(self.rect(), QColor("#ffffff"))
		painter.setPen(QPen(QColor("#c9d2e0"), 1))
		painter.drawRect(rect)

		vols = self._volumes
		if vols.size < 2 or not np.isfinite(vols).all():
			painter.setPen(QColor("#8a8a8a"))
			painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Sin datos para graficar")
			painter.end()
			return

		vmin = float(vols.min())
		vmax = float(vols.max())
		span = max(vmax - vmin, 1e-6)
		pad = span * 0.15
		lo = vmin - pad
		hi = vmax + pad

		def to_point(i: int, value: float):
			x = rect.left() + rect.width() * (i / max(1, vols.size - 1))
			y = rect.bottom() - rect.height() * ((value - lo) / (hi - lo))
			return x, y

		# Rejilla horizontal + etiquetas de mL
		small = QFont(painter.font())
		small.setPointSize(7)
		painter.setFont(small)
		for frac in (0.0, 0.5, 1.0):
			value = lo + (hi - lo) * frac
			_, y = to_point(0, value)
			painter.setPen(QPen(QColor("#eef1f6"), 1))
			painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
			painter.setPen(QColor("#6b7280"))
			painter.drawText(2, int(y) + 4, f"{value:5.0f}")

		path = QPainterPath()
		for i, value in enumerate(vols):
			x, y = to_point(i, float(value))
			if i == 0:
				path.moveTo(x, y)
			else:
				path.lineTo(x, y)
		painter.setPen(QPen(QColor("#1f3b5b"), 2))
		painter.drawPath(path)

		for i, value in enumerate(vols):
			x, y = to_point(i, float(value))
			gate = i + 1
			if gate == self._ed_gate:
				color, label = QColor("#2563eb"), "ED"
			elif gate == self._es_gate:
				color, label = QColor("#dc2626"), "ES"
			else:
				color, label = QColor("#94a3b8"), ""
			painter.setBrush(color)
			painter.setPen(QPen(color, 1))
			painter.drawEllipse(int(x) - 3, int(y) - 3, 6, 6)
			if label:
				painter.drawText(int(x) - 8, int(y) - 8, label)
			painter.setPen(QColor("#6b7280"))
			painter.drawText(int(x) - 4, rect.bottom() + 14, str(gate))

		painter.end()


class ECTbWindow(QDialog):
	"""Panel de cuantificación ECTb con recálculo en vivo."""

	#: Se emite cada vez que hay un resultado nuevo, por si la ventana
	#: principal quiere reflejarlo en el resumen.
	resultReady = pyqtSignal(object)

	#: Rebote de los controles, en milisegundos. Suficiente para agrupar el
	#: tecleo/scroll sin que se sienta demorado.
	DEBOUNCE_MS = 120

	def __init__(self, main_window):
		super().__init__(main_window)
		self._main = main_window
		self._result: ECTbLVResult | None = None
		#: Evita que actualizar los controles desde el estado de la app dispare
		#: recálculos y cambios de método en cadena.
		self._syncing = False
		#: Último cubo y spacing usados, para alimentar la cota sin recalcular.
		self._last_cube = None
		self._last_pixel_mm = 1.0
		self._last_slice_mm = 1.0
		#: Espesor asumido antes de pasar a modo medición, para poder volver.
		self._assumed_thickness_mm = 10.0
		self.setWindowTitle("Cuantificación avanzada — Emory Cardiac Toolbox")
		self.setMinimumSize(560, 640)
		self.resize(620, 820)
		# Sin modalidad: se puede seguir usando la ventana principal.
		self.setModal(False)
		self.setWindowFlag(Qt.WindowType.Window, True)

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
		layout.addWidget(self._build_method_box())
		layout.addWidget(self._build_params_box())
		layout.addWidget(self._build_valve_section())
		layout.addWidget(self._build_ruler_section())
		layout.addWidget(self._build_results_box())
		layout.addWidget(self._build_conversion_section())
		layout.addWidget(self._build_curve_box())
		layout.addWidget(self._build_compare_box())
		layout.addStretch(1)

		outer.addLayout(self._build_footer())
		self.sync_from_main()

	def _build_method_box(self) -> QGroupBox:
		"""Selector del método que alimenta resumen, gráficos e informe."""
		box = QGroupBox("Método que usa el informe")
		box_layout = QVBoxLayout(box)
		box_layout.setContentsMargins(8, 8, 8, 8)
		box_layout.setSpacing(4)

		hint = QLabel(
			"Lo que se elija acá es lo que aparece en el resumen, en la curva de volumen y en el PDF. "
			"El otro método sigue disponible para comparar."
		)
		hint.setWordWrap(True)
		hint.setStyleSheet("color:#6b7280;")
		box_layout.addWidget(hint)

		self.method_ectb_radio = QRadioButton("ECTb — máximo de cuentas")
		self.method_ectb_radio.setToolTip(
			"Método del Emory Cardiac Toolbox. Es el que viene por defecto porque sobreestima\n"
			"bastante menos los volúmenes que el anterior.\n"
			"Ubica el borde a partir del máximo de cuentas de la pared, sin umbral ni factor de corrección."
		)
		self.method_threshold_radio = QRadioButton("Anterior — umbral endocárdico")
		self.method_threshold_radio.setToolTip(
			"El método que usábamos antes: busca el primer radio donde la actividad supera un porcentaje\n"
			"del pico y después escala el volumen con un factor empírico.\n"
			"Queda disponible para comparar y para casos donde el método ECTb no resuelva bien."
		)
		self.method_group = QButtonGroup(self)
		self.method_group.addButton(self.method_ectb_radio, 0)
		self.method_group.addButton(self.method_threshold_radio, 1)
		box_layout.addWidget(self.method_ectb_radio)
		box_layout.addWidget(self.method_threshold_radio)

		self.method_warning = QLabel("")
		self.method_warning.setWordWrap(True)
		self.method_warning.setStyleSheet("color:#a06000;")
		box_layout.addWidget(self.method_warning)

		self.method_ectb_radio.toggled.connect(self._on_method_changed)
		return box

	def _on_method_changed(self, _checked: bool = False):
		"""Aplica el método elegido y avisa que el estudio se reprocesa."""
		if self._syncing:
			return
		method = (
			self._main.FEVI_METHOD_ECTB
			if self.method_ectb_radio.isChecked()
			else self._main.FEVI_METHOD_THRESHOLD
		)
		if method == self._main.fevi_method():
			return
		self._main.set_fevi_method(method)
		self.method_warning.setText(
			"Cambiaste el método: se está reprocesando el estudio para regenerar resumen, gráficos e informe."
		)
		QTimer.singleShot(2500, lambda: self.method_warning.setText(""))

	def _build_intro_section(self) -> CollapsibleSection:
		"""Explicación del método, colapsada por defecto."""
		box = QGroupBox("Qué hace este método y en qué se diferencia")
		box_layout = QVBoxLayout(box)
		box_layout.setContentsMargins(8, 8, 8, 8)
		text = QLabel(
			"<b>El problema del método anterior.</b> El borde endocárdico se buscaba como "
			"«el primer radio donde la actividad supera un porcentaje del pico». Ese porcentaje depende "
			"del contraste y del fondo del estudio, y como la FEVI la manda el volumen sistólico, un "
			"corrimiento chico del borde mueve mucho el resultado. Encima hacía falta un factor de "
			"corrección empírico para que los mililitros dieran fisiológicos, y aun así sobreestimaba "
			"los volúmenes.<br><br>"
			"<b>Lo que hace el ECTb.</b> No busca un borde por umbral: busca el <b>máximo de cuentas</b> "
			"de cada perfil radial, que marca el centro de la pared y es mucho más estable porque no "
			"depende del nivel de fondo. Después:<br>"
			"• asume <b>10 mm de espesor en telediástole</b> y pone el endocardio 5 mm hacia adentro del "
			"centro y el epicardio 5 mm hacia afuera;<br>"
			"• deriva el <b>engrosamiento</b> de cada gate del primer armónico de las cuentas máximas "
			"(en SPECT la pared es más fina que la resolución, así que las cuentas suben cuando la pared "
			"engrosa: es el efecto de volumen parcial usado a favor);<br>"
			"• suaviza los radios con <b>mediana 7×7 y después 3×3</b>, que quita píxeles sueltos sin "
			"correr los bordes reales;<br>"
			"• integra el volumen de la cavidad y calcula la <b>masa</b> como volumen de pared × 1.05 g/mL;"
			"<br>"
			"• recorta la base con un <b>plano valvular de dos piezas</b>: perpendicular del lado lateral y "
			"angulado del lado septal, porque el anillo mitral no es perpendicular al eje largo."
			"<br><br>"
			"<b>Qué falta.</b> El ángulo del septum se asume fijo (septal a la izquierda del corte de eje corto), "
			"no se detecta solo a partir del ventrículo derecho. Por eso quedó como parámetro editable."
		)
		text.setWordWrap(True)
		text.setTextFormat(Qt.TextFormat.RichText)
		text.setStyleSheet("color:#35506a; line-height:1.3;")
		box_layout.addWidget(text)
		return CollapsibleSection.from_group_box(box, key="ectb_intro", expanded=False)

	def _build_params_box(self) -> QGroupBox:
		box = QGroupBox("Parámetros (se recalcula en vivo)")
		form = QFormLayout(box)
		form.setContentsMargins(8, 8, 8, 8)
		form.setSpacing(5)

		self.thickness_spin = QDoubleSpinBox()
		self.thickness_spin.setRange(4.0, 20.0)
		self.thickness_spin.setSingleStep(0.5)
		self.thickness_spin.setDecimals(1)
		self.thickness_spin.setValue(10.0)
		self.thickness_spin.setSuffix(" mm")
		self._thickness_tooltip = (
			"Espesor de pared que se asume en telediástole. Es la convención del ECTb y lo que fija la "
			"escala absoluta de los volúmenes.\n"
			"Subirlo mete el endocardio hacia adentro: cavidad más chica, EDV/ESV menores y más masa.\n"
			"Bajarlo hace lo contrario. La FEVI cambia poco porque afecta a ED y ES en el mismo sentido.\n"
			"10 mm es el valor del ECTb. Si tenés una medición de referencia (un ecocardiograma reciente, "
			"por ejemplo) podés escribirla acá, o medirla sobre la imagen con la cota de más abajo."
		)
		self.thickness_spin.setToolTip(self._thickness_tooltip)

		self.angles_combo = QComboBox()
		for value in (32, 48, 64, 96, 128):
			self.angles_combo.addItem(str(value), value)
		self.angles_combo.setCurrentText("64")
		self.angles_combo.setToolTip(
			"Cuántos perfiles radiales se trazan por corte.\n"
			"Más ángulos = contorno más detallado y un poco más lento. 64 es un buen compromiso;\n"
			"por debajo de 32 el contorno se poligoniza y el volumen queda subestimado."
		)

		self.oversample_spin = QDoubleSpinBox()
		self.oversample_spin.setRange(1.0, 8.0)
		self.oversample_spin.setSingleStep(1.0)
		self.oversample_spin.setDecimals(0)
		self.oversample_spin.setValue(4.0)
		self.oversample_spin.setSuffix(" x")
		self.oversample_spin.setToolTip(
			"Muestras por píxel a lo largo de cada rayo.\n"
			"Sube la resolución con la que se ubica el máximo de cuentas antes del refinamiento subpíxel.\n"
			"4x alcanza de sobra; más solo agrega tiempo de cálculo."
		)

		self.median_large_combo = QComboBox()
		for value in (0, 3, 5, 7, 9):
			self.median_large_combo.addItem("sin filtro" if value == 0 else f"{value}×{value}", value)
		self.median_large_combo.setCurrentText("7×7")
		self.median_large_combo.setToolTip(
			"Primera mediana sobre la matriz corte × ángulo.\n"
			"La mediana elimina outliers (un foco hepático que corre un radio) sin desplazar los bordes\n"
			"reales, cosa que un suavizado gaussiano sí haría. 7×7 es lo que usa el ECTb."
		)

		self.median_small_combo = QComboBox()
		for value in (0, 3, 5):
			self.median_small_combo.addItem("sin filtro" if value == 0 else f"{value}×{value}", value)
		self.median_small_combo.setCurrentText("3×3")
		self.median_small_combo.setToolTip(
			"Segunda pasada de mediana, más fina, para pulir el contorno después de la 7×7."
		)

		self.thickening_check = QCheckBox("Aplicar engrosamiento por 1er armónico")
		self.thickening_check.setChecked(True)
		self.thickening_check.setToolTip(
			"Con esto activo el espesor de pared varía gate a gate según las cuentas máximas\n"
			"(la pared engrosa en sístole). Es lo correcto y lo que hace el ECTb.\n"
			"Desactivalo solo para ver cuánto aporta el engrosamiento: con espesor fijo la FEVI\n"
			"sale distinta porque el endocardio no se corre hacia adentro en sístole."
		)

		self.dropout_check = QCheckBox("Usar el cubo corregido por dropout del último gate")
		self.dropout_check.setChecked(True)
		self.dropout_check.setToolTip(
			"Aplica la corrección ECTb 22.8 antes de cuantificar (la misma del panel principal).\n"
			"Sin ella el último gate tiene menos cuentas de la cuenta y la curva de volumen no cierra el ciclo."
		)

		form.addRow("Espesor de pared en ED", self.thickness_spin)
		form.addRow("Perfiles radiales", self.angles_combo)
		form.addRow("Sobremuestreo radial", self.oversample_spin)
		form.addRow("Mediana gruesa", self.median_large_combo)
		form.addRow("Mediana fina", self.median_small_combo)
		form.addRow(self.thickening_check)
		form.addRow(self.dropout_check)

		self.thickness_spin.valueChanged.connect(self._schedule_recompute)
		self.oversample_spin.valueChanged.connect(self._schedule_recompute)
		self.angles_combo.currentIndexChanged.connect(self._schedule_recompute)
		self.median_large_combo.currentIndexChanged.connect(self._schedule_recompute)
		self.median_small_combo.currentIndexChanged.connect(self._schedule_recompute)
		self.thickening_check.toggled.connect(self._schedule_recompute)
		self.dropout_check.toggled.connect(self._schedule_recompute)
		return box

	def _build_valve_section(self) -> CollapsibleSection:
		"""Plano valvular de dos piezas: el corte de la base.

		Va en su propia sección colapsada para no engordar la caja de
		parámetros, pero los controles siguen recalculando en vivo.
		"""
		box = QGroupBox("Plano valvular (corte de la base)")
		box_layout = QVBoxLayout(box)
		box_layout.setContentsMargins(8, 8, 8, 8)
		box_layout.setSpacing(5)

		hint = QLabel(
			"El anillo mitral <b>no</b> es perpendicular al eje largo: del lado septal la cavidad del VI "
			"termina más cerca del ápex (ahí están el tracto de salida aórtico y el septum membranoso, "
			"que no son cavidad), y del lado lateral llega más arriba. Cortando con un plano perpendicular "
			"único hay que elegir entre comerse pared lateral o contar como cavidad lo que no lo es, y por "
			"eso el volumen basal se sobreestima.<br>"
			"El ECTb lo resuelve con un plano de <b>dos piezas</b>: perpendicular en toda la mitad lateral "
			"y angulado en la mitad septal."
		)
		hint.setWordWrap(True)
		hint.setTextFormat(Qt.TextFormat.RichText)
		hint.setStyleSheet("color:#35506a;")
		box_layout.addWidget(hint)

		self.valve_check = QCheckBox("Recortar la base con el plano valvular de dos piezas")
		self.valve_check.setChecked(True)
		self.valve_check.setToolTip(
			"Activo: se aplica el corte angulado del lado septal (comportamiento ECTb).\n"
			"Desactivado: la base se corta con un plano perpendicular único, como hacíamos antes.\n"
			"Desactivalo solo para ver cuánto aporta la corrección: los volúmenes suben."
		)
		box_layout.addWidget(self.valve_check)

		form = QFormLayout()
		form.setContentsMargins(0, 0, 0, 0)
		form.setSpacing(5)

		self.valve_offset_spin = QDoubleSpinBox()
		self.valve_offset_spin.setRange(0.0, 25.0)
		self.valve_offset_spin.setSingleStep(0.5)
		self.valve_offset_spin.setDecimals(1)
		self.valve_offset_spin.setValue(10.0)
		self.valve_offset_spin.setSuffix(" mm")
		self.valve_offset_spin.setToolTip(
			"Cuánto más apical termina la cavidad del lado septal que del lado lateral.\n"
			"Es el 'angulado' del plano. 10 mm es el valor anatómico habitual del anillo mitral.\n"
			"Subirlo saca más volumen de la base septal; en 0 mm el plano queda perpendicular.\n"
			"Cada corte entra con un peso 0-1 según qué fracción suya queda por debajo del plano,\n"
			"así que mover este valor cambia el volumen de forma continua, sin saltos de un corte entero."
		)

		self.valve_angle_spin = QSpinBox()
		self.valve_angle_spin.setRange(0, 359)
		self.valve_angle_spin.setSingleStep(15)
		self.valve_angle_spin.setValue(180)
		self.valve_angle_spin.setSuffix(" °")
		self.valve_angle_spin.setToolTip(
			"Dónde cae el medio del septum en el corte de eje corto.\n"
			"Con la convención de despliegue del módulo (septal a la izquierda, lateral a la derecha,\n"
			"anterior arriba) el septum está en 180°, que es el valor por defecto.\n"
			"Solo hay que tocarlo si el estudio entró con otra orientación."
		)

		form.addRow("Retroceso septal", self.valve_offset_spin)
		form.addRow("Ángulo del septum", self.valve_angle_spin)
		box_layout.addLayout(form)

		self.valve_readout = QLabel("—")
		self.valve_readout.setWordWrap(True)
		self.valve_readout.setStyleSheet("color:#1f3b5b; font-weight:600;")
		box_layout.addWidget(self.valve_readout)

		self.valve_check.toggled.connect(self._on_valve_toggled)
		self.valve_offset_spin.valueChanged.connect(self._schedule_recompute)
		self.valve_angle_spin.valueChanged.connect(self._schedule_recompute)
		return CollapsibleSection.from_group_box(box, key="ectb_valve", expanded=False)

	def _on_valve_toggled(self, checked: bool):
		"""Grisa los parámetros del plano cuando está desactivado."""
		self.valve_offset_spin.setEnabled(bool(checked))
		self.valve_angle_spin.setEnabled(bool(checked))
		self._schedule_recompute()

	def _update_valve_readout(self, result):
		"""Cuánto volumen basal descontó el plano, para poder auditarlo."""
		if result is None or not getattr(result, "available", False):
			self.valve_readout.setText("—")
			return
		if not self.valve_check.isChecked() or self.valve_offset_spin.value() <= 0.0:
			self.valve_readout.setText("Plano perpendicular: no se descuenta volumen basal.")
			return
		removed = float(getattr(result, "valve_removed_ml", 0.0))
		edv_sin = float(result.edv_ml) + removed
		pct = removed / edv_sin * 100.0 if edv_sin > 0.0 else 0.0
		cortes = self.valve_offset_spin.value() / max(1e-6, float(self._last_slice_mm))
		self.valve_readout.setText(
			f"Descontó {removed:.1f} mL del EDV ({pct:.1f}%), equivalente a {cortes:.1f} cortes "
			f"en el medio del septum."
		)

	def _build_ruler_section(self) -> CollapsibleSection:
		"""Cota manual para medir el espesor de pared en telediástole.

		Va colapsada y desactivada: si no se toca nada, el cálculo sigue usando
		el espesor asumido y no hay un solo clic extra en el flujo normal.
		"""
		box = QGroupBox("Medir el espesor de pared a mano (opcional)")
		box_layout = QVBoxLayout(box)
		box_layout.setContentsMargins(8, 8, 8, 8)
		box_layout.setSpacing(5)

		hint = QLabel(
			"Los 10 mm son una convención. Si el ventrículo está hipertrofiado o adelgazado, medir la "
			"pared sobre el corte telediastólico y usar ese valor acota mejor los volúmenes.<br>"
			"<b>Es una cota sobre el SPECT, con la resolución del SPECT:</b> sirve para confirmar o "
			"corregir el valor asumido, no como una medición de precisión milimétrica. Si el paciente "
			"trae un ecocardiograma reciente, ese espesor se puede cargar directamente en el campo de "
			"arriba sin usar esta herramienta."
		)
		hint.setWordWrap(True)
		hint.setTextFormat(Qt.TextFormat.RichText)
		hint.setStyleSheet("color:#6b7280;")
		box_layout.addWidget(hint)

		self.manual_thickness_check = QCheckBox("Usar la medida de la cota en vez del valor asumido")
		self.manual_thickness_check.setChecked(False)
		self.manual_thickness_check.setToolTip(
			"Desactivado (por defecto): se usa el espesor del campo «Espesor de pared en ED».\n"
			"Activado: se usa lo que mida la cota, y el campo pasa a mostrar ese valor en modo lectura.\n"
			"Al desactivarlo se recupera el valor que había antes de medir."
		)
		self.manual_thickness_check.toggled.connect(self._on_manual_thickness_toggled)
		box_layout.addWidget(self.manual_thickness_check)

		controls = QHBoxLayout()
		controls.setSpacing(6)
		controls.addWidget(QLabel("Corte:"))
		self.ruler_slice_spin = QSpinBox()
		self.ruler_slice_spin.setToolTip(
			"Corte del eje corto sobre el que se mide. Se muestra siempre el gate telediastólico,\n"
			"que es el que el método usa como referencia de espesor."
		)
		self.ruler_slice_spin.valueChanged.connect(self._on_ruler_slice_changed)
		controls.addWidget(self.ruler_slice_spin)

		controls.addWidget(QLabel("Ángulo:"))
		self.ruler_angle_spin = QSpinBox()
		self.ruler_angle_spin.setRange(0, 359)
		self.ruler_angle_spin.setSingleStep(15)
		self.ruler_angle_spin.setSuffix("°")
		self.ruler_angle_spin.setToolTip(
			"Gira la cota alrededor del centro del ventrículo para elegir qué pared medir\n"
			"(septal, lateral, anterior, inferior). Después se puede ajustar arrastrando los extremos."
		)
		self.ruler_angle_spin.valueChanged.connect(lambda _v: self._place_ruler())
		controls.addWidget(self.ruler_angle_spin)

		self.ruler_reset_btn = QPushButton("Reubicar")
		self.ruler_reset_btn.setToolTip(
			"Vuelve a apoyar la cota sobre los bordes que calculó el algoritmo, en el ángulo elegido."
		)
		self.ruler_reset_btn.clicked.connect(self._place_ruler)
		controls.addWidget(self.ruler_reset_btn)
		controls.addStretch(1)
		box_layout.addLayout(controls)

		self.ruler_contours_check = QCheckBox("Mostrar los bordes que calculó el algoritmo")
		self.ruler_contours_check.setChecked(True)
		self.ruler_contours_check.setToolTip(
			"Dibuja tres contornos: centro de la pared (amarillo tenue), endocardio (azul) y epicardio (verde).\n"
			"El centro de la pared es una medición real (máximo de cuentas); los otros dos salen de aplicarle\n"
			"el espesor, así que sirven para ver si el espesor asumido cierra con la imagen."
		)
		self.ruler_contours_check.toggled.connect(
			lambda checked: self.ruler.set_show_contours(bool(checked))
		)
		box_layout.addWidget(self.ruler_contours_check)

		self.ruler = WallThicknessRuler()
		self.ruler.measured.connect(self._on_ruler_measured)
		box_layout.addWidget(self.ruler)

		self.ruler_readout = QLabel("Cota: — mm")
		self.ruler_readout.setStyleSheet("font-weight:600; color:#1f3b5b;")
		box_layout.addWidget(self.ruler_readout)

		return CollapsibleSection.from_group_box(box, key="ectb_ruler", expanded=False)

	def _build_conversion_section(self) -> CollapsibleSection:
		"""Conversión de la FEVI a la escala de otro software."""
		box = QGroupBox("Convertir la FEVI a la escala de otro software (opcional)")
		box_layout = QVBoxLayout(box)
		box_layout.setContentsMargins(8, 8, 8, 8)
		box_layout.setSpacing(5)

		hint = QLabel(
			"Cada software pone el borde endocárdico en un lugar apenas distinto, así que la misma "
			"adquisición da FEVI diferentes según el equipo. No es un error de ninguno: son escalas "
			"distintas.<br>"
			"Esto importa cuando el paciente trae un estudio previo informado con otro equipo: comparar "
			"un 58% nuestro contra un 51% de QGS y concluir que «empeoró» es un error de lectura. "
			"Con la regresión publicada se expresa nuestro resultado en la escala del informe anterior "
			"y se los compara de igual a igual."
		)
		hint.setWordWrap(True)
		hint.setTextFormat(Qt.TextFormat.RichText)
		hint.setStyleSheet("color:#6b7280;")
		box_layout.addWidget(hint)

		row = QHBoxLayout()
		row.setSpacing(6)
		self.conversion_check = QCheckBox("Mostrar equivalencia con:")
		self.conversion_check.setChecked(False)
		self.conversion_check.setToolTip(
			"Desactivado por defecto. La FEVI del informe no cambia: esto solo agrega el valor "
			"equivalente para poder compararlo con un estudio previo de otro equipo."
		)
		self.conversion_check.toggled.connect(self._on_conversion_changed)
		row.addWidget(self.conversion_check)

		self.conversion_combo = QComboBox()
		for key, reg in EF_REGRESSIONS.items():
			self.conversion_combo.addItem(reg.label, key)
		self.conversion_combo.setToolTip("Software contra el que se quiere comparar el resultado.")
		self.conversion_combo.currentIndexChanged.connect(self._on_conversion_changed)
		row.addWidget(self.conversion_combo, 1)
		box_layout.addLayout(row)

		self.conversion_result = QLabel("—")
		self.conversion_result.setWordWrap(True)
		self.conversion_result.setStyleSheet("font-weight:600; color:#1f3b5b;")
		box_layout.addWidget(self.conversion_result)

		self.conversion_detail = QLabel("")
		self.conversion_detail.setWordWrap(True)
		self.conversion_detail.setStyleSheet("color:#6b7280;")
		box_layout.addWidget(self.conversion_detail)

		return CollapsibleSection.from_group_box(box, key="ectb_conversion", expanded=False)

	def _build_results_box(self) -> QGroupBox:
		box = QGroupBox("Resultados")
		grid = QGridLayout(box)
		grid.setContentsMargins(8, 8, 8, 8)
		grid.setHorizontalSpacing(10)
		grid.setVerticalSpacing(4)

		self._value_labels: dict[str, QLabel] = {}
		rows = [
			("ef", "FEVI", "Fracción de eyección: (EDV − ESV) / EDV. Normal ≥ 50%."),
			("edv", "EDV", "Volumen telediastólico: la cavidad en su punto más grande."),
			("esv", "ESV", "Volumen telesistólico: la cavidad en su punto más chico. Es el que más pesa en la FEVI."),
			("sv", "Volumen latido", "EDV − ESV: lo que el ventrículo expulsa por latido."),
			("mass", "Masa miocárdica", "Volumen de pared × 1.05 g/mL. Referencia normal ~90-140 g en hombres, ~70-110 g en mujeres."),
			("thick", "Engrosamiento sistólico", "Cuánto engrosa la pared entre ED y ES. Un ventrículo normal engrosa; si da negativo, revisá segmentación o gating."),
			("shape", "Índice de esfericidad ED / ES",
			 "Diámetro de eje corto dividido por la longitud del eje largo.\n"
			 "El VI normal es un elipsoide alargado, así que da bastante menor que 1.\n"
			 "Cuanto más se acerca a 1, más esférico: es un signo de remodelado que puede aparecer\n"
			 "con la FEVI todavía conservada, por eso aporta lo que la FEVI sola no ve.\n"
			 "El punto de corte depende del software y de la población: acá se informa el valor."),
			("gates", "Gates ED / ES", "En qué gate cayó el volumen máximo y el mínimo."),
			("slices", "Cortes usados", "Cortes que entraron a la integración del volumen."),
		]
		for row, (key, title, tip) in enumerate(rows):
			name = QLabel(title)
			name.setToolTip(tip)
			name.setStyleSheet("color:#4b5563;")
			value = QLabel("—")
			value.setToolTip(tip)
			value.setStyleSheet("font-weight:600; color:#1f3b5b;")
			grid.addWidget(name, row, 0)
			grid.addWidget(value, row, 1)
			self._value_labels[key] = value
		grid.setColumnStretch(1, 1)

		self.notes_label = QLabel("")
		self.notes_label.setWordWrap(True)
		self.notes_label.setStyleSheet("color:#a06000;")
		grid.addWidget(self.notes_label, len(rows), 0, 1, 2)
		return box

	def _build_curve_box(self) -> QGroupBox:
		box = QGroupBox("Curva de volumen por gate")
		box_layout = QVBoxLayout(box)
		box_layout.setContentsMargins(8, 8, 8, 8)
		self.curve = VolumeCurveWidget()
		box_layout.addWidget(self.curve)
		return box

	def _build_compare_box(self) -> QGroupBox:
		box = QGroupBox("Los dos métodos, lado a lado")
		box_layout = QVBoxLayout(box)
		box_layout.setContentsMargins(8, 8, 8, 8)
		box_layout.setSpacing(4)
		hint = QLabel(
			"Ambos se calculan siempre, sin importar cuál esté seleccionado arriba, así se puede seguir "
			"comparando en cada estudio."
		)
		hint.setWordWrap(True)
		hint.setStyleSheet("color:#6b7280;")
		box_layout.addWidget(hint)
		self.compare_label = QLabel("—")
		self.compare_label.setWordWrap(True)
		self.compare_label.setStyleSheet("color:#1f3b5b;")
		box_layout.addWidget(self.compare_label)
		return box

	def _build_footer(self) -> QHBoxLayout:
		row = QHBoxLayout()
		row.setSpacing(6)
		self.status_label = QLabel("Sin estudio procesado.")
		self.status_label.setStyleSheet("color:#6b7280;")
		row.addWidget(self.status_label, 1)

		self.apply_btn = QPushButton("Aplicar al informe")
		self.apply_btn.setToolTip(
			"Reprocesa el estudio para que el resumen, los gráficos y el PDF usen estos parámetros.\n"
			"No hace falta tocarlo para explorar: los parámetros ya quedan guardados y se aplican solos\n"
			"la próxima vez que proceses. Este botón es para verlo reflejado ahora mismo."
		)
		self.apply_btn.clicked.connect(self._apply_to_report)
		row.addWidget(self.apply_btn)

		self.reset_btn = QPushButton("Valores ECTb")
		self.reset_btn.setToolTip("Vuelve a los parámetros publicados por el Emory Cardiac Toolbox 4.0.")
		self.reset_btn.clicked.connect(self.reset_to_defaults)
		row.addWidget(self.reset_btn)

		self.refresh_btn = QPushButton("Recalcular")
		self.refresh_btn.setToolTip("Fuerza un recálculo (por ejemplo después de reprocesar el estudio).")
		self.refresh_btn.clicked.connect(self.recompute)
		row.addWidget(self.refresh_btn)

		close_btn = QPushButton("Cerrar")
		close_btn.clicked.connect(self.close)
		row.addWidget(close_btn)
		return row

	def _apply_to_report(self):
		"""Reprocesa el estudio con los parámetros actuales."""
		main = self._main
		main.set_ectb_config(self.current_config())
		if getattr(main, "study", None) is None:
			self.status_label.setText("No hay estudio cargado para reprocesar.")
			return
		main._invalidate_output_cache()
		self.status_label.setText("Reprocesando el estudio con estos parámetros...")
		QTimer.singleShot(0, main.process_current)

	def sync_from_main(self):
		"""Refleja en los controles el método y los parámetros vigentes en la app.

		Se usa al abrir la ventana y cuando un preset cambia la configuración por
		fuera. El flag `_syncing` evita disparar recálculos durante la carga.
		"""
		main = self._main
		self._syncing = True
		try:
			is_ectb = main.fevi_method() == main.FEVI_METHOD_ECTB
			self.method_ectb_radio.setChecked(is_ectb)
			self.method_threshold_radio.setChecked(not is_ectb)

			cfg = main.ectb_config()
			self.thickness_spin.setValue(float(cfg.ed_wall_thickness_mm))
			self.oversample_spin.setValue(float(cfg.radial_oversample))
			self._select_by_data(self.angles_combo, int(cfg.n_angles))
			self._select_by_data(self.median_large_combo, int(cfg.median_kernel_large))
			self._select_by_data(self.median_small_combo, int(cfg.median_kernel_small))
			self.thickening_check.setChecked(bool(cfg.use_thickening))
			self.valve_check.setChecked(bool(cfg.use_valve_plane))
			self.valve_offset_spin.setValue(float(cfg.valve_septal_offset_mm))
			self.valve_angle_spin.setValue(int(round(float(cfg.septal_angle_deg))) % 360)
			self.valve_offset_spin.setEnabled(bool(cfg.use_valve_plane))
			self.valve_angle_spin.setEnabled(bool(cfg.use_valve_plane))

			regression = main.fevi_regression()
			self.conversion_check.setChecked(bool(regression))
			self.conversion_combo.setEnabled(bool(regression))
			if regression:
				self._select_by_data(self.conversion_combo, regression)
		finally:
			self._syncing = False

	@staticmethod
	def _select_by_data(combo: QComboBox, value):
		index = combo.findData(value)
		if index >= 0:
			combo.setCurrentIndex(index)

	# ------------------------------------------------------------------
	# Cálculo
	# ------------------------------------------------------------------

	def current_config(self) -> ECTbLVConfig:
		"""Arma la configuración del motor a partir de los controles."""
		return ECTbLVConfig(
			ed_wall_thickness_mm=float(self.thickness_spin.value()),
			n_angles=int(self.angles_combo.currentData()),
			radial_oversample=float(self.oversample_spin.value()),
			median_kernel_large=int(self.median_large_combo.currentData()),
			median_kernel_small=int(self.median_small_combo.currentData()),
			use_thickening=bool(self.thickening_check.isChecked()),
			use_valve_plane=bool(self.valve_check.isChecked()),
			valve_septal_offset_mm=float(self.valve_offset_spin.value()),
			septal_angle_deg=float(self.valve_angle_spin.value()),
		)

	def reset_to_defaults(self):
		"""Restaura los parámetros publicados por ECTb sin disparar N recálculos."""
		# La medición manual se apaga: si no, pisaría el espesor recién restaurado.
		if self.manual_thickness_check.isChecked():
			self.manual_thickness_check.setChecked(False)
		for widget in (
			self.thickness_spin,
			self.oversample_spin,
			self.angles_combo,
			self.median_large_combo,
			self.median_small_combo,
			self.thickening_check,
			self.dropout_check,
			self.valve_check,
			self.valve_offset_spin,
			self.valve_angle_spin,
		):
			widget.blockSignals(True)
		self.thickness_spin.setValue(10.0)
		self.oversample_spin.setValue(4.0)
		self.angles_combo.setCurrentText("64")
		self.median_large_combo.setCurrentText("7×7")
		self.median_small_combo.setCurrentText("3×3")
		self.thickening_check.setChecked(True)
		self.dropout_check.setChecked(True)
		self.valve_check.setChecked(True)
		self.valve_offset_spin.setValue(10.0)
		self.valve_angle_spin.setValue(180)
		self.valve_offset_spin.setEnabled(True)
		self.valve_angle_spin.setEnabled(True)
		for widget in (
			self.thickness_spin,
			self.oversample_spin,
			self.angles_combo,
			self.median_large_combo,
			self.median_small_combo,
			self.thickening_check,
			self.dropout_check,
			self.valve_check,
			self.valve_offset_spin,
			self.valve_angle_spin,
		):
			widget.blockSignals(False)
		self.recompute()

	def _schedule_recompute(self):
		"""Rebote: agrupa ráfagas de cambios en un solo cálculo."""
		if self._syncing:
			return
		self._recalc_timer.start(self.DEBOUNCE_MS)

	# ------------------------------------------------------------------
	# Cota manual de espesor
	# ------------------------------------------------------------------

	def _on_manual_thickness_toggled(self, checked: bool):
		"""Alterna entre el espesor asumido y el medido con la cota."""
		checked = bool(checked)
		# Se deja habilitado (no gris) para que el valor medido se siga leyendo bien,
		# pero sin flechas y sin edición: la cota es la que manda.
		self.thickness_spin.setReadOnly(checked)
		self.thickness_spin.setButtonSymbols(
			QAbstractSpinBox.ButtonSymbols.NoButtons if checked else QAbstractSpinBox.ButtonSymbols.UpDownArrows
		)
		if checked:
			# Se guarda el asumido para poder volver a él si desactivan la cota.
			self._assumed_thickness_mm = float(self.thickness_spin.value())
			measured = self.ruler.measurement_mm()
			if measured > 0.0:
				self._apply_measured_thickness(measured)
			else:
				self._place_ruler()
			self.thickness_spin.setToolTip(
				"En modo medición este campo muestra lo que marca la cota y no se edita.\n"
				"Desactivá la casilla de abajo para volver a escribirlo a mano."
			)
		else:
			self.thickness_spin.setToolTip(self._thickness_tooltip)
			restored = float(getattr(self, "_assumed_thickness_mm", 10.0))
			if abs(restored - float(self.thickness_spin.value())) > 1e-6:
				self.thickness_spin.setValue(restored)
			else:
				self._schedule_recompute()
		self._update_ruler_readout()

	def _apply_measured_thickness(self, mm: float):
		"""Vuelca la medida de la cota al campo de espesor y recalcula."""
		mm = float(np.clip(mm, self.thickness_spin.minimum(), self.thickness_spin.maximum()))
		self.thickness_spin.blockSignals(True)
		self.thickness_spin.setValue(mm)
		self.thickness_spin.blockSignals(False)
		self._schedule_recompute()

	def _on_ruler_measured(self, mm: float):
		self._update_ruler_readout(mm)
		if self.manual_thickness_check.isChecked():
			self._apply_measured_thickness(mm)

	def _update_ruler_readout(self, mm: float | None = None):
		value = self.ruler.measurement_mm() if mm is None else float(mm)
		if value <= 0.0:
			self.ruler_readout.setText("Cota: — mm")
			return
		if self.manual_thickness_check.isChecked():
			self.ruler_readout.setText(f"Cota: {value:.1f} mm  →  en uso para el cálculo")
		else:
			assumed = float(self.thickness_spin.value())
			self.ruler_readout.setText(
				f"Cota: {value:.1f} mm  (asumido {assumed:.1f} mm, diferencia {value - assumed:+.1f} mm) "
				"— activá la casilla para usarla"
			)

	def _on_ruler_slice_changed(self, _value: int):
		self._refresh_ruler_view()
		self._place_ruler()

	def _refresh_ruler_view(self):
		"""Muestra el corte elegido del gate telediastólico y sus contornos."""
		result = self._result
		cube = self._last_cube
		seg = getattr(self._main, "seg", None)
		if result is None or cube is None or seg is None:
			return
		slice_index = int(self.ruler_slice_spin.value())
		gate_index = max(0, int(result.ed_gate) - 1)
		try:
			plane = np.asarray(cube[gate_index, slice_index], dtype=np.float64)
		except Exception:
			return
		self.ruler.set_image(plane, self._last_pixel_mm)
		self.ruler.set_contours(self._contours_for_slice(result, seg, slice_index, gate_index))
		self.ruler.set_show_contours(bool(self.ruler_contours_check.isChecked()))

	def _contours_for_slice(self, result: ECTbLVResult, seg, slice_index: int, gate_index: int):
		"""Convierte los radios polares del resultado en contornos dibujables."""
		if slice_index not in result.valid_slices:
			return []
		row = result.valid_slices.index(slice_index)
		centers = np.asarray(getattr(seg, "center_per_slice", np.empty((0, 2))), dtype=np.float64)
		if slice_index >= centers.shape[0]:
			return []
		cy, cx = float(centers[slice_index, 0]), float(centers[slice_index, 1])
		px_mm = max(self._last_pixel_mm, 1e-6)

		n_ang = int(result.center_radii_mm.shape[-1])
		angles = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
		out = []
		for color, radii in (
			(QColor(255, 210, 63, 140), result.center_radii_mm[gate_index, row]),
			(QColor(80, 160, 255), result.endo_radii_mm[gate_index, row]),
			(QColor(80, 220, 120), result.epi_radii_mm[gate_index, row]),
		):
			r_px = np.asarray(radii, dtype=np.float64) / px_mm
			xs = cx + r_px * np.cos(angles)
			ys = cy + r_px * np.sin(angles)
			out.append((color, np.stack([xs, ys], axis=1)))
		return out

	def _place_ruler(self):
		"""Apoya la cota sobre los bordes calculados, en el ángulo elegido.

		Arranca desde lo que calculó el algoritmo para que el usuario confirme o
		corrija, en vez de tener que medir de cero.
		"""
		result = self._result
		seg = getattr(self._main, "seg", None)
		if result is None or seg is None:
			return
		slice_index = int(self.ruler_slice_spin.value())
		if slice_index not in result.valid_slices:
			return
		row = result.valid_slices.index(slice_index)
		gate_index = max(0, int(result.ed_gate) - 1)
		centers = np.asarray(getattr(seg, "center_per_slice", np.empty((0, 2))), dtype=np.float64)
		if slice_index >= centers.shape[0]:
			return

		cy, cx = float(centers[slice_index, 0]), float(centers[slice_index, 1])
		px_mm = max(self._last_pixel_mm, 1e-6)
		n_ang = int(result.center_radii_mm.shape[-1])
		angle_deg = float(self.ruler_angle_spin.value())
		idx = int(round(angle_deg / 360.0 * n_ang)) % n_ang
		theta = 2.0 * np.pi * idx / n_ang

		r_in = float(result.endo_radii_mm[gate_index, row, idx]) / px_mm
		r_out = float(result.epi_radii_mm[gate_index, row, idx]) / px_mm
		self.ruler.set_measurement(
			(cx + r_in * np.cos(theta), cy + r_in * np.sin(theta)),
			(cx + r_out * np.cos(theta), cy + r_out * np.sin(theta)),
		)

	# ------------------------------------------------------------------
	# Conversión a la escala de otro software
	# ------------------------------------------------------------------

	def _on_conversion_changed(self, *_args):
		"""Aplica la conversión elegida y la propaga al informe."""
		if self._syncing:
			return
		key = self.conversion_combo.currentData() if self.conversion_check.isChecked() else None
		self.conversion_combo.setEnabled(bool(self.conversion_check.isChecked()))
		self._main.set_fevi_regression(key)
		self._render_conversion()

	def _render_conversion(self):
		result = self._result
		if not self.conversion_check.isChecked():
			self.conversion_result.setText("Desactivado: se informa la FEVI en escala ECTb.")
			self.conversion_detail.setText("")
			return
		if result is None:
			self.conversion_result.setText("—")
			self.conversion_detail.setText("")
			return
		key = str(self.conversion_combo.currentData())
		reg = EF_REGRESSIONS[key]
		converted = convert_ef_pct(result.ef_pct, reg)
		self.conversion_result.setText(
			f"FEVI ECTb {result.ef_pct:.1f} %  →  {converted:.1f} % en escala {reg.label} "
			f"({converted - result.ef_pct:+.1f} puntos)"
		)
		self.conversion_detail.setText(f"{regression_equation_text(reg)}\n{reg.note}")

	def showEvent(self, event):  # noqa: N802 - API de Qt
		super().showEvent(event)
		self.sync_from_main()
		self.recompute()

	def recompute(self):
		"""Recalcula con los parámetros actuales y refresca toda la ventana."""
		main = self._main
		study = getattr(main, "study", None)
		seg = getattr(main, "seg", None)
		if study is None or seg is None:
			self._show_unavailable("Cargá y procesá un estudio en la ventana principal para cuantificar.")
			return

		pixel_spacing = getattr(study, "pixel_spacing", None)
		slice_mm = getattr(study, "z_spacing_mm", None)
		if not pixel_spacing or slice_mm is None:
			self._show_unavailable("El estudio no trae spacing válido: no se pueden calcular mililitros.")
			return

		cube = getattr(study, "cube", None)
		if cube is None:
			self._show_unavailable("El estudio no tiene un cubo gated cargado.")
			return

		if self.dropout_check.isChecked() and hasattr(main, "_apply_gate_dropout_correction"):
			cube, _ = main._apply_gate_dropout_correction(cube, log=False)

		self._last_cube = cube
		self._last_pixel_mm = float(np.mean([abs(float(pixel_spacing[0])), abs(float(pixel_spacing[1]))]))
		self._last_slice_mm = abs(float(slice_mm))

		t0 = perf_counter()
		config = self.current_config()
		# Se guardan en la app para que el próximo procesamiento use estos valores
		# sin obligar al usuario a apretar nada.
		main.set_ectb_config(config)
		try:
			result = analyze_lv_ectb(
				cube,
				seg,
				(float(pixel_spacing[0]), float(pixel_spacing[1])),
				float(slice_mm),
				config,
			)
		except Exception as err:  # el motor no debería fallar, pero no queremos tumbar la ventana
			self._show_unavailable(f"Error al cuantificar: {err}")
			return
		elapsed_ms = (perf_counter() - t0) * 1000.0

		if not result.available:
			self._show_unavailable(result.reason or "No se pudo cuantificar con estos parámetros.")
			return

		self._result = result
		self._render_result(result, elapsed_ms)
		self.resultReady.emit(result)

	def _show_unavailable(self, message: str):
		self._result = None
		for label in self._value_labels.values():
			label.setText("—")
		self.notes_label.setText("")
		self.compare_label.setText("—")
		self.conversion_result.setText("—")
		self.conversion_detail.setText("")
		self.valve_readout.setText("—")
		self.curve.set_curve(np.zeros(0), 0, 0)
		self.status_label.setText(message)
		self.status_label.setStyleSheet("color:#a06000;")

	def _render_result(self, result: ECTbLVResult, elapsed_ms: float):
		self._value_labels["ef"].setText(f"{result.ef_pct:.1f} %")
		self._value_labels["edv"].setText(f"{result.edv_ml:.1f} mL")
		self._value_labels["esv"].setText(f"{result.esv_ml:.1f} mL")
		self._value_labels["sv"].setText(f"{result.sv_ml:.1f} mL")
		self._value_labels["mass"].setText(
			f"{result.myocardial_mass_g:.1f} g  ({result.myocardial_volume_ml:.1f} mL de pared)"
		)
		self._value_labels["thick"].setText(f"{result.thickening_pct:+.1f} %")
		self._value_labels["shape"].setText(
			f"{result.shape_index_ed:.2f} / {result.shape_index_es:.2f}"
			f"   (eje corto {result.short_axis_ed_mm:.0f} mm, eje largo {result.long_axis_mm:.0f} mm)"
		)
		self._value_labels["gates"].setText(f"{result.ed_gate} / {result.es_gate}")
		self._value_labels["slices"].setText(f"{len(result.valid_slices)} de {result.n_slices_total}")
		self.notes_label.setText("\n".join(result.notes))
		self.curve.set_curve(result.gate_volumes_ml, result.ed_gate, result.es_gate)
		self.status_label.setText(f"Recalculado en {elapsed_ms:.0f} ms.")
		self.status_label.setStyleSheet("color:#4b7a4b;")
		self._sync_ruler_range(result)
		self._refresh_ruler_view()
		if self.ruler.measurement_mm() <= 0.0:
			# Primera vez: se apoya la cota sobre los bordes calculados para que el
			# usuario confirme o corrija en vez de medir de cero.
			self._place_ruler()
		self._update_ruler_readout()
		self._update_valve_readout(result)
		self._render_conversion()
		self._render_comparison(result)

	def _sync_ruler_range(self, result: ECTbLVResult):
		"""Ajusta el rango del selector de corte a los cortes que entraron al cálculo."""
		if not result.valid_slices:
			return
		low, high = int(min(result.valid_slices)), int(max(result.valid_slices))
		current = int(self.ruler_slice_spin.value())
		self.ruler_slice_spin.blockSignals(True)
		self.ruler_slice_spin.setRange(low, high)
		if current not in result.valid_slices:
			# Por defecto, el corte medio de los válidos: es el más representativo
			# de la pared y el que menos sufre el efecto de borde de base y ápex.
			self.ruler_slice_spin.setValue(int(result.valid_slices[len(result.valid_slices) // 2]))
		self.ruler_slice_spin.blockSignals(False)

	def _render_comparison(self, result: ECTbLVResult):
		"""Muestra el resultado del método actual al lado del de ECTb."""
		main = self._main
		current = {}
		if hasattr(main, "_estimate_lv_ef_preliminary"):
			try:
				current = main._estimate_lv_ef_preliminary() or {}
			except Exception:
				current = {}
		if not current.get("available"):
			self.compare_label.setText("El método actual no pudo estimar la FEVI para este estudio.")
			return

		ef_old = float(current.get("ef_pct") or 0.0)
		edv_old = float(current.get("edv_ml") or 0.0)
		esv_old = float(current.get("esv_ml") or 0.0)
		uses_ectb = main.fevi_method() == main.FEVI_METHOD_ECTB
		mark_ectb = "▶ " if uses_ectb else "   "
		mark_old = "   " if uses_ectb else "▶ "
		self.compare_label.setText(
			f"{mark_ectb}ECTb (máx. cuentas):  FEVI {result.ef_pct:.1f} %   EDV {result.edv_ml:.1f} mL   "
			f"ESV {result.esv_ml:.1f} mL\n"
			f"{mark_old}Anterior (umbral):  FEVI {ef_old:.1f} %   EDV {edv_old:.1f} mL   ESV {esv_old:.1f} mL\n"
			f"   Diferencia ECTb − anterior:  FEVI {result.ef_pct - ef_old:+.1f} puntos   "
			f"EDV {result.edv_ml - edv_old:+.1f} mL   ESV {result.esv_ml - esv_old:+.1f} mL\n"
			f"   (▶ marca el que está usando el informe)"
		)
