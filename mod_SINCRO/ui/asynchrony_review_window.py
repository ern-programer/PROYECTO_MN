"""SINCRO - ui.asynchrony_review_window.

Vista asincronía LCARS: esfuerzo y reposo a la vez, cada etapa con la ROI del
flujo principal (anillo), los contornos irregulares ECTb (pared) y el
histograma de fase en vivo.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
	QButtonGroup,
	QComboBox,
	QDoubleSpinBox,
	QFormLayout,
	QFrame,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QScrollArea,
	QSlider,
	QSpinBox,
	QVBoxLayout,
	QWidget,
)

from core.ectb_lv import analyze_lv_ectb
from ui.cine_widget import CineWidget, VerticalColorStrip
from ui.epar_dialog import EParDialog
from ui.montage_intestinal_window import LcarsRangeSlider
from viz.perfusion_score_figure import AMBER, BG, BLUE, PANEL, TEXT

try:
	from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
	from matplotlib.figure import Figure
	_MPL_OK = True
except Exception:  # pragma: no cover
	_MPL_OK = False


GREEN = "#90e277"
STAGES = ("stress", "rest")
STAGE_NAMES = {"stress": "ESFUERZO", "rest": "REPOSO", "both": "AMBAS"}

STYLE = f"""
QDialog#asynchronyReviewLcars {{ background:{BG}; color:{TEXT}; }}
QWidget#asyncReviewBody {{ background:{BG}; color:{TEXT};
	font-family:'Arial Narrow','Bahnschrift Condensed','Segoe UI'; }}
QWidget#asyncReviewBody QLabel {{ color:{TEXT}; background:transparent; }}
QWidget#asyncReviewBody QLabel[asyncCaption="true"] {{ color:{BLUE}; font-size:10px; font-weight:800; padding-top:4px; }}
QWidget#asyncReviewBody QLabel#head {{ color:{AMBER}; font-size:13px; font-weight:800; }}
QWidget#asyncReviewBody QLabel#bigValue {{ color:{AMBER}; font-size:22px; font-weight:800; }}
QWidget#asyncReviewBody QLabel#small {{ color:#8a979c; font-size:9px; }}
QFrame#asyncReviewRail {{ background:{PANEL}; border-left:10px solid {BLUE};
	border-top-left-radius:18px; border-bottom-left-radius:18px; }}
QFrame#asyncReadout {{ background:{PANEL}; border:1px solid #343945; border-radius:10px; }}
QFrame#asyncStageRow {{ background:#11151c; border:1px solid #343945; border-radius:10px; }}
QFrame#asyncStageRow[activeStage="true"] {{ border:1px solid {AMBER}; }}
QWidget#asyncReviewBody QPushButton {{ border:none; border-radius:13px; padding:6px 12px;
	color:#172028; font-weight:800; background:{BLUE}; }}
QWidget#asyncReviewBody QPushButton[lcars="true"] {{ text-align:right; }}
QWidget#asyncReviewBody QPushButton:checked {{ background:{AMBER}; }}
QWidget#asyncReviewBody QPushButton:hover {{ background:#9bb6bd; }}
QWidget#asyncReviewBody QPushButton:disabled {{ background:#2a2e36; color:#5b6670; }}
QWidget#asyncReviewBody QPushButton[asyncAction="true"] {{ background:{GREEN}; }}
QWidget#asyncReviewBody QPushButton#reject {{ background:#a7615e; color:white; }}
QWidget#asyncReviewBody QPushButton#mini {{ border-radius:8px; padding:2px 8px; }}
QWidget#asyncReviewBody QComboBox, QWidget#asyncReviewBody QSpinBox,
QWidget#asyncReviewBody QDoubleSpinBox {{
	background:#1b2230; color:{TEXT}; border:1px solid {BLUE}; border-radius:6px; padding:3px 6px; }}
QWidget#asyncReviewBody QComboBox QAbstractItemView {{ background:{PANEL}; color:{TEXT};
	selection-background-color:{AMBER}; selection-color:#172028; }}
QWidget#asyncReviewBody QSlider::groove:horizontal {{ height:6px; background:#2a2e36; border-radius:3px; }}
QWidget#asyncReviewBody QSlider::sub-page:horizontal {{ background:{BLUE}; border-radius:3px; }}
QWidget#asyncReviewBody QSlider::handle:horizontal {{ background:{AMBER}; width:14px;
	margin:-5px 0; border-radius:7px; }}
QScrollArea#asyncReviewScroll {{ background:{PANEL}; border:none; border-radius:10px; }}
QScrollArea#asyncReviewScroll > QWidget > QWidget {{ background:{PANEL}; }}
"""


class _StageView:
	"""Visores, histograma y lectura de una etapa."""

	def __init__(self, stage: str):
		self.stage = stage
		self.frame = None
		self.title = None
		self.note = None
		self.ring = None
		self.wall = None
		self.readout_box = None
		self.psd_value = None
		self.metrics_label = None
		self.cube = None
		self.wall_result = None
		self.wall_reason = ""
		self.px_mm = 1.0
		self.fig = None
		self.ax = None
		self.canvas = None


class AsynchronyReviewWindow(EParDialog):
	"""Comparador visual anillo vs pared/ECTb, esfuerzo y reposo simultáneos."""

	def __init__(self, main_window=None, parent=None):
		super().__init__(parent, title="VISTA ASINCRONÍA · AJUSTE FINO", accent=AMBER,
			settings=getattr(main_window, "_ui_settings", None),
			geometry_key="dialogs/asynchrony_review/geometry")
		self.setObjectName("asynchronyReviewLcars")
		self.setStyleSheet(self.styleSheet() + STYLE)
		self._main = main_window
		self._wall_px_mm = 1.0
		self._show_center_contour = True
		self._show_endo_contour = True
		self._show_epi_contour = True
		self._edit_stage = "stress"
		self._stage_phase_params = {}
		self._nav_ready = False
		self._views = {stage: _StageView(stage) for stage in STAGES}
		self._play_timer = QTimer(self)
		self._play_timer.setInterval(200)
		self._play_timer.timeout.connect(self._advance_gate)
		self.setMinimumSize(1100, 640)
		self.resize(1500, 900)

		body = QWidget(self)
		body.setObjectName("asyncReviewBody")
		root = QHBoxLayout(body)
		root.setContentsMargins(6, 0, 0, 0)
		root.setSpacing(10)
		root.addWidget(self._scroll(self._build_rail(), 230))
		root.addWidget(self._build_center(), 1)
		root.addWidget(self._build_scale_column())
		root.addWidget(self._scroll(self._build_readout(), 270))
		self.content_layout.addWidget(body, 1)
		for button in self.findChildren(QPushButton):
			button.setAutoDefault(False)
		self._update_stage_dependent_controls()
		self._refresh_stage_highlight()

	# ------------------------------------------------------------------
	# Construcción
	# ------------------------------------------------------------------
	@staticmethod
	def _pill(text: str, *, checkable: bool = False, checked: bool = False, tip: str = "") -> QPushButton:
		button = QPushButton(text)
		button.setProperty("lcars", True)
		button.setCheckable(checkable)
		button.setChecked(checked)
		button.setCursor(Qt.CursorShape.PointingHandCursor)
		if tip:
			button.setToolTip(tip)
		return button

	@staticmethod
	def _caption(text: str, name: str = "") -> QLabel:
		label = QLabel(text)
		if name:
			label.setObjectName(name)
		else:
			label.setProperty("asyncCaption", True)
		return label

	@staticmethod
	def _scroll(widget: QWidget, width: int) -> QScrollArea:
		scroll = QScrollArea()
		scroll.setObjectName("asyncReviewScroll")
		scroll.setWidgetResizable(True)
		scroll.setFrameShape(QFrame.Shape.NoFrame)
		scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		scroll.setFixedWidth(width)
		scroll.setWidget(widget)
		return scroll

	def _build_rail(self) -> QFrame:
		rail = QFrame()
		rail.setObjectName("asyncReviewRail")
		lay = QVBoxLayout(rail)
		lay.setContentsMargins(18, 12, 10, 12)
		lay.setSpacing(6)

		lay.addWidget(self._caption("ETAPA A PROCESAR"))
		self.stage_group = QButtonGroup(self)
		self.stage_buttons = {}
		for stage in ("stress", "rest", "both"):
			button = self._pill(STAGE_NAMES[stage], checkable=True, checked=stage == "stress",
				tip="Etapa que reciben Reprocesar, Centrar en cavidad y Limpiar centros. Ambas etapas se ven siempre.")
			button.clicked.connect(lambda _checked=False, st=stage: self._select_edit_stage(st))
			self.stage_group.addButton(button)
			self.stage_buttons[stage] = button
			lay.addWidget(button)

		lay.addSpacing(6)
		lay.addWidget(self._caption("NAVEGACIÓN"))
		self.gate_label = QLabel("GATE —")
		self.gate_slider = QSlider(Qt.Orientation.Horizontal)
		self.gate_slider.valueChanged.connect(self._sync_position)
		self.slice_label = QLabel("CORTE —")
		self.slice_label.setWordWrap(True)
		self.slice_slider = QSlider(Qt.Orientation.Horizontal)
		self.slice_slider.valueChanged.connect(self._sync_position)
		for widget in (self.gate_label, self.gate_slider, self.slice_label, self.slice_slider):
			lay.addWidget(widget)
		self.play_btn = self._pill("▶ REPRODUCIR", checkable=True, tip="Reproduce los gates en ambas etapas.")
		self.play_btn.toggled.connect(self._on_play_toggled)
		self.goto_valid_btn = self._pill("IR A CORTE CON CONTORNOS",
			tip="Los contornos de pared solo se calculan en cortes con miocardio suficiente: salta al válido más cercano.")
		self.goto_valid_btn.clicked.connect(self._goto_nearest_valid_slice)
		lay.addWidget(self.play_btn)
		lay.addWidget(self.goto_valid_btn)

		lay.addSpacing(6)
		lay.addWidget(self._caption("CONTORNOS DE PARED"))
		self.center_check = self._pill("CENTRO", checkable=True, checked=True)
		self.endo_check = self._pill("ENDOCARDIO", checkable=True, checked=True)
		self.epi_check = self._pill("EPICARDIO", checkable=True, checked=True)
		for button in (self.center_check, self.endo_check, self.epi_check):
			button.toggled.connect(self._on_contour_visibility_changed)
			lay.addWidget(button)

		lay.addSpacing(6)
		lay.addWidget(self._caption("CENTRO DEL VI"))
		self.cavity_center_check = self._pill("CENTRAR EN CAVIDAD", checkable=True,
			tip="Apagado: centroide de la máscara de miocardio (se corre hacia el sector que más capta).\n"
				"Encendido: el centro se recalcula sobre la cavidad. Reprocesa la etapa elegida.")
		self.cavity_center_check.toggled.connect(self._on_cavity_center_toggled)
		self.manual_center_check = self._pill("CENTRO MANUAL", checkable=True,
			tip="Clic izq. en el anillo de una etapa fija su centro de cavidad; clic der. lo borra.\n"
				"Mueve el muestreo polar/AHA y los radios (no cambia la FEVI).")
		self.manual_center_check.toggled.connect(self._on_manual_center_mode_toggled)
		self.manual_center_all_check = self._pill("CLIC EN TODOS LOS CORTES", checkable=True)
		self.manual_center_all_check.toggled.connect(self._on_manual_center_all_toggled)
		self.manual_center_clear_btn = self._pill("LIMPIAR CENTROS")
		self.manual_center_clear_btn.clicked.connect(self._on_manual_center_clear)
		for button in (self.cavity_center_check, self.manual_center_check,
				self.manual_center_all_check, self.manual_center_clear_btn):
			lay.addWidget(button)

		lay.addSpacing(6)
		lay.addWidget(self._caption("ROI DEL ANÁLISIS"))
		self.roi_group = QButtonGroup(self)
		self.roi_buttons = {
			"ring": self._pill("ANILLO", checkable=True, checked=True, tip="ROI clásica del panel Anillo."),
			"ectb_wall": self._pill("PARED", checkable=True,
				tip="Pared irregular según el máximo de cuentas (considera PVE). Solo Esfuerzo."),
		}
		for button in self.roi_buttons.values():
			self.roi_group.addButton(button)
			lay.addWidget(button)
		self.apply_btn = self._pill("APLICAR Y REPROCESAR",
			tip="Usa la geometría elegida como ROI del análisis de fase y reprocesa Esfuerzo.")
		self.apply_btn.setProperty("asyncAction", True)
		self.apply_btn.clicked.connect(self._apply_selected_roi_source)
		self.applied_label = self._caption("—", "small")
		lay.addWidget(self.apply_btn)
		lay.addWidget(self.applied_label)
		lay.addStretch(1)
		return rail

	def _build_center(self) -> QWidget:
		host = QWidget()
		lay = QVBoxLayout(host)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.setSpacing(8)
		for stage in STAGES:
			lay.addWidget(self._build_stage_row(stage), 1)
		return host

	def _build_stage_row(self, stage: str) -> QFrame:
		view = self._views[stage]
		view.frame = QFrame()
		view.frame.setObjectName("asyncStageRow")
		lay = QVBoxLayout(view.frame)
		lay.setContentsMargins(10, 6, 10, 8)
		lay.setSpacing(4)
		header = QHBoxLayout()
		view.title = self._caption(STAGE_NAMES[stage], "head")
		view.note = QLabel("")
		header.addWidget(view.title)
		header.addStretch(1)
		header.addWidget(view.note)
		lay.addLayout(header)

		view.ring = CineWidget(self)
		view.ring.set_controls_visible(False)
		view.ring.setToolTip("Anillo de la ROI principal. Con CENTRO MANUAL: clic izq. fija el centro, clic der. lo borra.")
		view.ring.centerPicked.connect(lambda sl, center, st=stage: self._on_center_picked(sl, center, stage=st))
		view.wall = CineWidget(self)
		view.wall.set_controls_visible(False)
		view.wall.setToolTip("Contornos irregulares de pared (centro/endo/epi) por gate.")
		cells = QHBoxLayout()
		cells.setSpacing(8)
		for caption, widget in (
			("ANILLO · ROI PRINCIPAL", view.ring),
			("PARED · CONTORNOS", view.wall),
			("FASE EN VIVO", self._build_phase_canvas(view)),
		):
			cell = QVBoxLayout()
			cell.addWidget(self._caption(caption))
			cell.addWidget(widget, 1)
			cells.addLayout(cell, 1)
		lay.addLayout(cells, 1)
		return view.frame

	@staticmethod
	def _build_phase_canvas(view: _StageView) -> QWidget:
		if not _MPL_OK:
			label = QLabel("matplotlib no disponible: sin gráfico de fase.")
			label.setWordWrap(True)
			label.setAlignment(Qt.AlignmentFlag.AlignCenter)
			return label
		view.fig = Figure(figsize=(3.2, 2.4))
		view.fig.patch.set_facecolor(BG)
		view.ax = view.fig.add_subplot(111)
		view.canvas = FigureCanvas(view.fig)
		view.canvas.setMinimumSize(180, 150)
		return view.canvas

	def _build_scale_column(self) -> QFrame:
		col = QFrame()
		col.setObjectName("asyncReadout")
		col.setFixedWidth(132)
		lay = QVBoxLayout(col)
		lay.setContentsMargins(6, 8, 6, 8)
		lay.addWidget(self._caption("ESCALA"))
		self.cmap_combo = QComboBox()
		main = self._main
		cmaps = list(getattr(main, "_all_cmaps", []) or [])
		if not cmaps and main is not None and hasattr(main, "cmap_combo"):
			cmaps = [main.cmap_combo.itemText(i) for i in range(main.cmap_combo.count())]
		self.cmap_combo.addItems(cmaps or ["french"])
		self.cmap_combo.setToolTip("Escala de color de las cuatro imágenes y del mapa de fase al reprocesar.")
		self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
		lay.addWidget(self.cmap_combo)
		self.interp_combo = QComboBox()
		self.interp_combo.addItems(["Píxel", "Bilineal", "Bicúbico", "Hanning", "Lanczos"])
		self.interp_combo.setCurrentText("Bilineal")
		self.interp_combo.setToolTip("Interpolación al escalar las imágenes.")
		self.interp_combo.currentTextChanged.connect(self._apply_visual)
		lay.addWidget(self.interp_combo)
		top = QHBoxLayout()
		btn_top = QPushButton("TOP")
		btn_top.setObjectName("mini")
		btn_top.setToolTip("Volver Top a 100%")
		btn_top.clicked.connect(lambda: self.range_slider.set_values(self.range_slider.values()[0], 100))
		self.top_label = QLabel("100%")
		top.addWidget(btn_top)
		top.addWidget(self.top_label)
		lay.addLayout(top)
		sliders = QHBoxLayout()
		self.range_slider = LcarsRangeSlider()
		self.range_slider.valuesChanged.connect(self._apply_visual)
		self.color_strip = VerticalColorStrip(self.cmap_combo.currentText())
		sliders.addWidget(self.range_slider, 1)
		sliders.addWidget(self.color_strip)
		lay.addLayout(sliders, 1)
		base = QHBoxLayout()
		btn_base = QPushButton("BASE")
		btn_base.setObjectName("mini")
		btn_base.setToolTip("Volver Base a 0%")
		btn_base.clicked.connect(lambda: self.range_slider.set_values(0, self.range_slider.values()[1]))
		self.base_label = QLabel("0%")
		base.addWidget(btn_base)
		base.addWidget(self.base_label)
		lay.addLayout(base)
		lay.addWidget(self._caption("SUAVIZADO"))
		self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
		self.smooth_slider.setRange(0, 30)
		self.smooth_slider.setToolTip("Suavizado visual (no altera el motor).")
		self.smooth_slider.valueChanged.connect(self._apply_visual)
		self.smooth_label = QLabel("0.0")
		lay.addWidget(self.smooth_slider)
		lay.addWidget(self.smooth_label)
		return col

	def _build_readout(self) -> QFrame:
		panel = QFrame()
		panel.setObjectName("asyncReadout")
		lay = QVBoxLayout(panel)
		lay.setContentsMargins(10, 10, 10, 10)
		lay.setSpacing(6)
		lay.addWidget(self._caption("PARÁMETROS DE FASE"))
		form = QFormLayout()
		form.setContentsMargins(0, 0, 0, 0)
		self.seg_method_combo = QComboBox()
		self.seg_method_combo.addItems(["auto", "threshold", "manual"])
		self.threshold_spin = QDoubleSpinBox()
		self.threshold_spin.setRange(0.01, 0.90)
		self.threshold_spin.setSingleStep(0.01)
		self.sigma_spin = QDoubleSpinBox()
		self.sigma_spin.setRange(0.0, 6.0)
		self.sigma_spin.setSingleStep(0.1)
		self.harmonics_spin = QSpinBox()
		self.harmonics_spin.setRange(1, 4)
		self.amp_filter_spin = QDoubleSpinBox()
		self.amp_filter_spin.setRange(0.01, 0.80)
		self.amp_filter_spin.setSingleStep(0.01)
		form.addRow("Segmentación", self.seg_method_combo)
		form.addRow("Umbral", self.threshold_spin)
		form.addRow("Sigma", self.sigma_spin)
		form.addRow("Armónicos", self.harmonics_spin)
		form.addRow("Filtro amplitud", self.amp_filter_spin)
		lay.addLayout(form)
		self.reprocess_btn = self._pill("REPROCESAR", tip="Vuelca estos parámetros al motor y reprocesa la etapa elegida.")
		self.reprocess_btn.setProperty("asyncAction", True)
		self.reprocess_btn.clicked.connect(self._apply_phase_controls_and_reprocess)
		lay.addWidget(self.reprocess_btn)

		lay.addSpacing(6)
		lay.addWidget(self._caption("RESULTADOS EN VIVO"))
		for stage in STAGES:
			view = self._views[stage]
			view.readout_box = QWidget()
			box = QVBoxLayout(view.readout_box)
			box.setContentsMargins(0, 4, 0, 4)
			box.setSpacing(0)
			box.addWidget(self._caption(STAGE_NAMES[stage], "head"))
			view.psd_value = self._caption("—", "bigValue")
			view.metrics_label = QLabel("Sin resultados: procesá el estudio.")
			view.metrics_label.setWordWrap(True)
			view.metrics_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
			box.addWidget(view.psd_value)
			box.addWidget(view.metrics_label)
			lay.addWidget(view.readout_box)

		lay.addSpacing(6)
		lay.addWidget(self._caption("ESTADO"))
		self.status_label = QLabel("—")
		self.status_label.setWordWrap(True)
		self.status_label.setStyleSheet(f"color:{AMBER};")
		lay.addWidget(self.status_label)
		lay.addStretch(1)
		note = self._caption("Fase, métricas y FEVI vienen del motor principal.", "small")
		note.setWordWrap(True)
		lay.addWidget(note)
		close = self._pill("CERRAR")
		close.setObjectName("reject")
		close.clicked.connect(self.reject)
		lay.addWidget(close)
		return panel

	# ------------------------------------------------------------------
	# Datos por etapa
	# ------------------------------------------------------------------
	def _rest_available(self) -> bool:
		"""¿Hay una 2da etapa (Reposo) cargada?"""
		main = self._main
		return main is not None and getattr(main, "compare_bundle", None) is not None

	def _active_is_rest(self) -> bool:
		"""¿La etapa a procesar es solo Reposo?"""
		return self._edit_stage == "rest" and self._rest_available()

	def _compare_bundle(self):
		main = self._main
		return getattr(main, "compare_bundle", None) if main is not None else None

	def _stage_available(self, stage: str) -> bool:
		if stage == "rest":
			return self._rest_available()
		return self._main is not None and getattr(self._main, "study", None) is not None

	def _stage_study(self, stage: str):
		if stage == "rest":
			bundle = self._compare_bundle()
			return bundle.get("study") if bundle is not None else None
		return getattr(self._main, "study", None) if self._main is not None else None

	def _stage_metrics(self, stage: str):
		main = self._main
		if main is None:
			return None
		if stage == "rest":
			bundle = self._compare_bundle()
			if bundle is None:
				return None
			return bundle.get("metrics") or getattr(main, "compare_metrics", None)
		return getattr(main, "metrics", None)

	def _stage_phase_result(self, stage: str):
		if stage == "rest":
			bundle = self._compare_bundle()
			return bundle.get("phase_result") if bundle is not None else None
		return getattr(self._main, "phase_result", None) if self._main is not None else None

	def _stage_ef(self, stage: str):
		"""FEVI de la etapa (dict con 'available'/'ef_pct')."""
		main = self._main
		if main is None:
			return None
		if stage == "rest":
			bundle = self._compare_bundle()
			ef = bundle.get("ef") if bundle is not None else None
			return ef if ef is not None else getattr(main, "compare_ef", None)
		try:
			return main._estimate_lv_ef()
		except Exception:
			return None

	def _stage_seg_ring(self, stage: str):
		"""Segmentación anular (semilla ECTb) de la etapa."""
		if stage == "rest":
			bundle = self._compare_bundle()
			seg = bundle.get("seg") if bundle is not None else None
			if seg is not None and str(getattr(seg, "method", "")) != "ectb_wall":
				return seg
			return None
		return self._ectb_seed_segmentation()

	def _stage_rois(self, stage: str):
		"""ROIs a dibujar en el panel Anillo de la etapa."""
		main = self._main
		if main is None:
			return {}
		if stage == "rest":
			# El anillo real de Reposo muestra el centrado; el texto de ROIs no se re-deriva.
			seg = self._stage_seg_ring("rest")
			if seg is not None and hasattr(main, "_rois_from_segmentation"):
				try:
					rings = main._rois_from_segmentation(seg)
					if rings:
						return rings
				except Exception:
					pass
			try:
				return main._parse_manual_rois_text(getattr(main, "compare_manual_rois_text", "")) or {}
			except Exception:
				return {}
		try:
			return main._parse_manual_rois() or {}
		except Exception:
			return {}

	def _ectb_seed_segmentation(self):
		"""Segmentación anular que el ECTb usa como semilla en Esfuerzo.

		Con la pared ECTb aplicada, `main.seg` deja de ser un anillo: alimentar al
		ECTb con su propia salida lo deja sin cavidad desde donde tirar rayos.
		"""
		main = self._main
		if main is None:
			return None
		base = getattr(main, "seg_ring_base", None)
		if base is not None and str(getattr(base, "method", "")) != "ectb_wall":
			return base
		seg = getattr(main, "seg", None)
		if seg is not None and str(getattr(seg, "method", "")) == "ectb_wall":
			return None
		return seg

	# ------------------------------------------------------------------
	# Etapa a procesar
	# ------------------------------------------------------------------
	def _select_edit_stage(self, stage: str):
		if stage in ("rest", "both") and not self._rest_available():
			self._set_status("No hay 2da etapa (Reposo) cargada: se trabaja sobre Esfuerzo.")
			stage = "stress"
		self.stage_buttons[stage].setChecked(True)
		if stage == self._edit_stage:
			return
		self._stage_phase_params[self._edit_stage] = self._capture_sidebar_params()
		self._edit_stage = stage
		cached = self._stage_phase_params.get(stage)
		if cached:
			self._apply_sidebar_params(cached)
		self._update_stage_dependent_controls()
		self._refresh_stage_highlight()
		self._set_status(f"Etapa a procesar: {STAGE_NAMES[stage].capitalize()}.", ok=True)

	def _update_stage_dependent_controls(self):
		"""La ROI ECTb como fuente del análisis es exclusiva de Esfuerzo."""
		is_rest = self._active_is_rest()
		for button in (*self.roi_buttons.values(), self.apply_btn):
			button.setEnabled(not is_rest)
		self.apply_btn.setToolTip(
			"Sólo disponible en Esfuerzo. El motor de Reposo usa la ROI de anillo."
			if is_rest else "Usa la geometría elegida como ROI del análisis de fase y reprocesa Esfuerzo."
		)

	def _refresh_stage_controls_enabled(self):
		available = self._rest_available()
		for stage in ("rest", "both"):
			self.stage_buttons[stage].setEnabled(available)
		if not available and self._edit_stage in ("rest", "both"):
			self._edit_stage = "stress"
			self.stage_buttons["stress"].setChecked(True)
			self._update_stage_dependent_controls()
		rest = self._views["rest"]
		rest.frame.setVisible(available)
		rest.readout_box.setVisible(available)
		self._refresh_stage_highlight()

	def _refresh_stage_highlight(self):
		targets = STAGES if self._edit_stage == "both" else (self._edit_stage,)
		for stage, view in self._views.items():
			view.frame.setProperty("activeStage", stage in targets)
			view.frame.style().unpolish(view.frame)
			view.frame.style().polish(view.frame)

	def _capture_sidebar_params(self) -> dict:
		"""Fotografía los parámetros de fase (memoria por etapa)."""
		return {
			"seg_method": self.seg_method_combo.currentText(),
			"threshold": float(self.threshold_spin.value()),
			"sigma": float(self.sigma_spin.value()),
			"harmonics": int(self.harmonics_spin.value()),
			"amp_filter": float(self.amp_filter_spin.value()),
			"cmap": self.cmap_combo.currentText(),
		}

	def _apply_sidebar_params(self, params: dict):
		"""Restaura parámetros de fase memorizados (sin disparar señales)."""
		for widget, key, kind in (
			(self.seg_method_combo, "seg_method", "combo"),
			(self.threshold_spin, "threshold", "spin"),
			(self.sigma_spin, "sigma", "spin"),
			(self.harmonics_spin, "harmonics", "spin"),
			(self.amp_filter_spin, "amp_filter", "spin"),
			(self.cmap_combo, "cmap", "combo"),
		):
			if key not in params:
				continue
			widget.blockSignals(True)
			if kind == "spin":
				widget.setValue(params[key])
			else:
				idx = widget.findText(str(params[key]))
				if idx >= 0:
					widget.setCurrentIndex(idx)
			widget.blockSignals(False)

	# ------------------------------------------------------------------
	# ROI del análisis y centrado
	# ------------------------------------------------------------------
	def _selected_roi_source(self) -> str:
		return next((key for key, button in self.roi_buttons.items() if button.isChecked()), "ring")

	def _refresh_applied_label(self):
		"""Muestra qué geometría usa hoy el análisis principal."""
		main = self._main
		source = str(main.roi_source()) if main is not None and hasattr(main, "roi_source") else "ring"
		button = self.roi_buttons.get(source)
		if button is not None and not button.isChecked():
			button.setChecked(True)
		name = "pared" if source == "ectb_wall" else "anillo clásico"
		self.applied_label.setText(f"Activa: {name}")
		self._sync_cavity_center_check()

	def _sync_cavity_center_check(self):
		"""Refleja el estado real del centrado sin disparar un reproceso."""
		main = self._main
		if main is None or not hasattr(main, "cavity_center_enabled"):
			return
		enabled = bool(main.cavity_center_enabled())
		if bool(self.cavity_center_check.isChecked()) == enabled:
			return
		self.cavity_center_check.blockSignals(True)
		self.cavity_center_check.setChecked(enabled)
		self.cavity_center_check.blockSignals(False)

	def _on_cavity_center_toggled(self, checked: bool):
		"""Cambia el criterio de centrado y reprocesa la etapa elegida."""
		main = self._main
		check = getattr(main, "cavity_center_check", None) if main is not None else None
		if check is None:
			self._set_status("No hay ventana principal para cambiar el centrado.")
			return
		if bool(check.isChecked()) == bool(checked):
			return
		# Sin el auto-reproceso de la principal, que reprocesaría ambas etapas.
		check.blockSignals(True)
		check.setChecked(bool(checked))
		check.blockSignals(False)
		if hasattr(main, "_propagate_cavity_center"):
			main._propagate_cavity_center()
		modo = "cavidad" if bool(checked) else "centroide de miocardio"
		if getattr(main, "study", None) is None:
			self._set_status(f"Centro del VI: {modo}. Cargá un estudio para procesarlo.")
			return
		try:
			main.reprocess_async_stage(self._edit_stage)
		except Exception as exc:
			self._set_status(f"No se pudo reprocesar: {exc}")
			return
		self._set_status(f"Centro del VI: {modo} ({STAGE_NAMES[self._edit_stage].capitalize()}). Reprocesando...", ok=True)

	def _on_manual_center_mode_toggled(self, checked: bool):
		for view in self._views.values():
			if view.ring is not None:
				view.ring.set_center_pick_mode(bool(checked))
		if checked:
			self._set_status("Centro manual activo: clic izq. en la cavidad del anillo de cada etapa (clic der. borra).", ok=True)
		else:
			self._set_status("Centro manual desactivado.")

	def _on_manual_center_all_toggled(self, checked: bool):
		main = self._main
		chk = getattr(main, "manual_center_all_check", None) if main is not None else None
		if chk is not None and bool(chk.isChecked()) != bool(checked):
			chk.setChecked(bool(checked))

	def _on_manual_center_clear(self):
		"""Borra los centros manuales de la etapa elegida (o de ambas)."""
		main = self._main
		targets = STAGES if self._edit_stage == "both" else (self._edit_stage,)
		cleared = False
		for stage in targets:
			name = "_clear_compare_manual_centers" if stage == "rest" else "_clear_manual_centers"
			handler = getattr(main, name, None) if main is not None else None
			if handler is not None and self._stage_available(stage):
				handler()
				cleared = True
		if not cleared:
			self._set_status("No hay ventana principal para limpiar centros.")
			return
		self._push_manual_centers()
		self._set_status("Centros manuales borrados; vuelve el centro automático.", ok=True)

	def _on_center_picked(self, slice_index, center, stage: str | None = None):
		"""Reenvía el clic de centro a la etapa del visor clickeado."""
		main = self._main
		if stage is None:
			stage = "rest" if self._active_is_rest() else "stress"
		name = "_on_compare_center_picked" if stage == "rest" else "_on_center_picked"
		handler = getattr(main, name, None) if main is not None else None
		if handler is None:
			self._set_status("No hay ventana principal para fijar el centro.")
			return
		handler(slice_index, center)
		self._push_manual_centers()

	def _push_manual_centers(self):
		main = self._main
		for stage, view in self._views.items():
			if view.ring is None:
				continue
			attr = "compare_manual_center_per_slice" if stage == "rest" else "manual_center_per_slice"
			view.ring.set_manual_centers(getattr(main, attr, None) if main is not None else None)

	def _sync_manual_center_checks(self):
		main = self._main
		chk = getattr(main, "manual_center_all_check", None) if main is not None else None
		if chk is None:
			return
		enabled = bool(chk.isChecked())
		if bool(self.manual_center_all_check.isChecked()) == enabled:
			return
		self.manual_center_all_check.blockSignals(True)
		self.manual_center_all_check.setChecked(enabled)
		self.manual_center_all_check.blockSignals(False)

	def _apply_selected_roi_source(self):
		"""Fija la geometría elegida en la ventana principal y reprocesa Esfuerzo."""
		main = self._main
		if main is None or not hasattr(main, "set_roi_source"):
			self._set_status("No hay ventana principal para aplicar la ROI.")
			return
		source = self._selected_roi_source()
		main.set_roi_source(source)
		self._refresh_applied_label()
		if getattr(main, "study", None) is None:
			self._set_status("ROI seleccionada. Cargá un estudio en la ventana principal para procesarlo.")
			return
		try:
			main.reprocess_async_stage("stress")
		except Exception as exc:
			self._set_status(f"No se pudo reprocesar: {exc}")
			return
		name = "pared" if source == "ectb_wall" else "anillo clásico"
		self._set_status(f"Análisis reprocesado con {name}.", ok=True)

	# ------------------------------------------------------------------
	# Parámetros de fase
	# ------------------------------------------------------------------
	def _apply_phase_controls_and_reprocess(self):
		main = self._main
		if main is None:
			self._set_status("No hay ventana principal para reprocesar.")
			return
		self._write_phase_controls_to_main()
		if getattr(main, "study", None) is None:
			self._set_status("Parámetros aplicados. Cargá un estudio en la ventana principal.")
			return
		try:
			main.reprocess_async_stage(self._edit_stage)
		except Exception as exc:
			self._set_status(f"No se pudo reprocesar: {exc}")
			return
		self._set_status(f"Reprocesado ({STAGE_NAMES[self._edit_stage].capitalize()}) con los parámetros de fase actuales.", ok=True)

	def _write_phase_controls_to_main(self):
		main = self._main
		if main is None:
			return
		combo = getattr(main, "seg_method", None)
		if combo is not None:
			idx = combo.findText(self.seg_method_combo.currentText())
			if idx >= 0:
				combo.setCurrentIndex(idx)
		for src, dst_name in (
			(self.threshold_spin, "threshold_spin"),
			(self.sigma_spin, "sigma_spin"),
			(self.harmonics_spin, "harmonics_spin"),
			(self.amp_filter_spin, "phase_threshold_spin"),
		):
			dst = getattr(main, dst_name, None)
			if dst is not None:
				dst.setValue(src.value())
		self._push_cmap_to_main(self.cmap_combo.currentText())

	def _push_cmap_to_main(self, text: str):
		main = self._main
		if main is None:
			return
		for attr in ("cmap_combo", "report_cmap_phase", "report_cmap_polar_clinico"):
			combo = getattr(main, attr, None)
			if combo is None:
				continue
			idx = combo.findText(text)
			if idx >= 0 and idx != combo.currentIndex():
				combo.setCurrentIndex(idx)

	def _on_cmap_changed(self, text: str):
		"""Recolorea en vivo las cuatro imágenes; el mapa de fase toma el color al reprocesar."""
		self._apply_visual()
		self._push_cmap_to_main(str(text))

	def _sync_phase_controls_from_main(self):
		"""Refleja los valores del motor; los memorizados de la etapa elegida mandan."""
		main = self._main
		if main is None:
			return
		cached = self._stage_phase_params.get(self._edit_stage)
		if cached:
			self._apply_sidebar_params(cached)
			return
		pairs = [
			(self.seg_method_combo, getattr(main, "seg_method", None), "combo"),
			(self.threshold_spin, getattr(main, "threshold_spin", None), "spin"),
			(self.sigma_spin, getattr(main, "sigma_spin", None), "spin"),
			(self.harmonics_spin, getattr(main, "harmonics_spin", None), "spin"),
			(self.amp_filter_spin, getattr(main, "phase_threshold_spin", None), "spin"),
			(self.cmap_combo, getattr(main, "cmap_combo", None), "combo"),
		]
		for widget, src, kind in pairs:
			if src is None:
				continue
			widget.blockSignals(True)
			if kind == "spin":
				widget.setValue(src.value())
			else:
				idx = widget.findText(src.currentText())
				if idx >= 0:
					widget.setCurrentIndex(idx)
			widget.blockSignals(False)

	# ------------------------------------------------------------------
	# Sincronización con la ventana principal
	# ------------------------------------------------------------------
	def sync_from_main(self):
		"""Actualiza las dos etapas con el estado del motor."""
		if self._main is None:
			for view in self._views.values():
				view.cube = None
				view.ring.set_cube(None)
				view.wall.set_cube(None)
			return
		self._refresh_stage_controls_enabled()
		self._refresh_applied_label()
		self._sync_phase_controls_from_main()
		for stage, view in self._views.items():
			study = self._stage_study(stage) if self._stage_available(stage) else None
			cube = getattr(study, "cube", None) if study is not None else None
			view.cube = cube
			view.ring.set_cube(cube)
			view.wall.set_cube(cube)
			if cube is None:
				view.wall_result = None
				view.ring.set_manual_rois(None)
				view.wall.set_manual_rois(None)
				view.wall.preview.set_overlay_contours([])
				continue
			view.ring.set_manual_rois(self._stage_rois(stage))
			view.wall.set_manual_rois({})
			self._ensure_wall_result(stage)
		self._refresh_nav_ranges()
		self._apply_visual()
		self._sync_position()
		self._push_manual_centers()
		self._sync_manual_center_checks()
		self._refresh_metrics_readout()
		self._refresh_live_phase_graph()
		self._apply_ui_preferences()

	def _apply_ui_preferences(self):
		if self._main is None:
			return
		enable_tooltips = bool(getattr(self._main, "_ui_enable_tooltips", True))
		for view in self._views.values():
			for widget in (view.ring, view.wall):
				widget.set_ui_preferences(show_helpers=False, enable_tooltips=enable_tooltips, compact_controls=True)

	def _apply_visual(self, *_args):
		"""Misma escala de color, ventana, suavizado e interpolación en las cuatro imágenes."""
		cmap = self.cmap_combo.currentText()
		low, high = self.range_slider.values()
		smooth = float(self.smooth_slider.value()) / 10.0
		interp = self.interp_combo.currentText()
		self.top_label.setText(f"{high}%")
		self.base_label.setText(f"{low}%")
		self.smooth_label.setText(f"{smooth:.1f}")
		self.color_strip.set_cmap(cmap)
		for view in self._views.values():
			for widget in (view.ring, view.wall):
				widget.cmap_combo.blockSignals(True)
				widget.cmap_combo.setCurrentText(cmap)
				widget.cmap_combo.blockSignals(False)
				widget._window_low = low / 100.0
				widget._window_high = high / 100.0
				widget._smooth_sigma = smooth
				widget.set_display_interp(interp)
		self._refresh_all_overlays()

	# ------------------------------------------------------------------
	# Navegación compartida
	# ------------------------------------------------------------------
	@staticmethod
	def _clamped_position(shape, gate: int, slice_index: int) -> tuple[int, int]:
		"""Mismo gate/corte en ambas etapas, recortado si una tiene menos (igual que la principal)."""
		return (max(0, min(int(gate), int(shape[0]) - 1)), max(0, min(int(slice_index), int(shape[1]) - 1)))

	def _refresh_nav_ranges(self):
		shapes = [np.shape(view.cube) for view in self._views.values() if view.cube is not None]
		for slider, axis in ((self.gate_slider, 0), (self.slice_slider, 1)):
			top = max((int(shape[axis]) for shape in shapes), default=1) - 1
			slider.blockSignals(True)
			slider.setRange(0, max(0, top))
			slider.blockSignals(False)
		if shapes and not self._nav_ready:
			self.slice_slider.blockSignals(True)
			self.slice_slider.setValue(self.slice_slider.maximum() // 2)
			self.slice_slider.blockSignals(False)
			self._nav_ready = True

	def _sync_position(self, *_args):
		gate, sl = int(self.gate_slider.value()), int(self.slice_slider.value())
		parts = []
		for stage, view in self._views.items():
			if view.cube is None:
				continue
			shape = np.shape(view.cube)
			gg, ss = self._clamped_position(shape, gate, sl)
			for widget in (view.ring, view.wall):
				widget.gate_slider.setValue(gg)
				widget.slice_slider.setValue(ss)
			parts.append(f"{STAGE_NAMES[stage][:3]} {ss + 1}/{shape[1]}")
			self._refresh_wall_overlay(stage)
		self.gate_label.setText(f"GATE {gate + 1}/{self.gate_slider.maximum() + 1}" if parts else "GATE —")
		self.slice_label.setText("CORTE " + " · ".join(parts) if parts else "CORTE —")

	def _on_play_toggled(self, playing: bool):
		if playing and self.gate_slider.maximum() > 0:
			self._play_timer.start()
			self.play_btn.setText("⏸ PAUSA")
			return
		self._play_timer.stop()
		self.play_btn.setText("▶ REPRODUCIR")
		if self.play_btn.isChecked():
			self.play_btn.blockSignals(True)
			self.play_btn.setChecked(False)
			self.play_btn.blockSignals(False)

	def _advance_gate(self):
		count = self.gate_slider.maximum() + 1
		self.gate_slider.setValue((self.gate_slider.value() + 1) % max(1, count))

	def hideEvent(self, event):
		if self.play_btn.isChecked():
			self.play_btn.setChecked(False)
		super().hideEvent(event)

	def _goto_nearest_valid_slice(self):
		"""Lleva ambas etapas al corte más cercano con contornos ECTb de la etapa elegida."""
		stage = "rest" if self._active_is_rest() else "stress"
		view = self._views[stage]
		result = view.wall_result
		if result is None:
			self._set_status("Todavía no hay contornos de pared calculados.")
			return
		valid = [int(v) for v in getattr(result, "valid_slices", ()) or ()]
		if not valid:
			self._set_status("No se encontraron cortes con contornos de pared en este estudio.")
			return
		current = int(view.wall.current_slice_index())
		target = min(valid, key=lambda v: (abs(v - current), v))
		self.slice_slider.setValue(target)
		self._set_status(f"{STAGE_NAMES[stage].capitalize()}: corte {target + 1} con contornos.", ok=True)

	# ------------------------------------------------------------------
	# Resultados y gráfico de fase
	# ------------------------------------------------------------------
	@staticmethod
	def _metrics_summary(metrics, ef) -> tuple[str, str]:
		if not metrics:
			return "—", "Sin resultados: procesá el estudio."

		def _fmt(key, unit=""):
			try:
				return f"{float(metrics.get(key)):.1f}{unit}"
			except (TypeError, ValueError):
				return "N/D"

		fevi = "N/D"
		try:
			if ef and ef.get("available"):
				fevi = f"{float(ef.get('ef_pct')):.1f}%"
		except (TypeError, ValueError):
			fevi = "N/D"
		cls = metrics.get("technical_classification", metrics.get("classification", "N/D"))
		return (
			f"PSD {_fmt('phase_sd', '°')}",
			f"BW {_fmt('bandwidth', '°')} · Entropía {_fmt('entropy_normalized_pct', '%')}\n"
			f"FEVI {fevi}\nPSD técnico: {cls} (no dx)",
		)

	def _refresh_metrics_readout(self):
		for stage, view in self._views.items():
			metrics = self._stage_metrics(stage) if self._stage_available(stage) else None
			ef = self._stage_ef(stage) if metrics else None
			big, detail = self._metrics_summary(metrics, ef)
			view.psd_value.setText(big)
			view.metrics_label.setText(detail)

	def _refresh_live_phase_graph(self):
		for stage in STAGES:
			self._draw_phase_hist(stage)

	def _draw_phase_hist(self, stage: str):
		"""Histograma de fase en vivo, leído de phase_result (no de un PNG)."""
		view = self._views[stage]
		if view.ax is None:
			return
		ax = view.ax
		ax.clear()
		view.fig.patch.set_facecolor(BG)
		ax.set_facecolor(BG)
		for spine in ax.spines.values():
			spine.set_color("#343945")
		ax.tick_params(colors=BLUE, labelsize=7)
		phase_result = self._stage_phase_result(stage) if self._stage_available(stage) else None
		phases = getattr(phase_result, "phases_deg", None) if phase_result is not None else None
		phases = np.asarray([] if phases is None else phases, dtype=np.float64)
		phases = phases[np.isfinite(phases)]
		if phases.size == 0:
			ax.text(0.5, 0.5, "Sin datos de fase", ha="center", va="center", color=BLUE, fontsize=9,
				transform=ax.transAxes)
			ax.set_xticks([])
			ax.set_yticks([])
			view.canvas.draw_idle()
			return
		ax.hist(phases % 360.0, bins=72, range=(0.0, 360.0), color=BLUE, alpha=0.85, edgecolor=BG, linewidth=0.4)
		try:
			from core.metrics import circular_mean_deg
			mean_phase = float(circular_mean_deg(phases))
			ax.axvline(mean_phase, color=AMBER, linewidth=2.0, label=f"Media {mean_phase:.0f}°")
			legend = ax.legend(loc="upper right", fontsize=7, framealpha=0.25)
			for text in legend.get_texts():
				text.set_color(TEXT)
		except Exception:
			pass
		ax.set_xlim(0.0, 360.0)
		ax.set_xticks([0, 90, 180, 270, 360])
		ax.set_xlabel("Fase (°)", color=BLUE, fontsize=8)
		view.fig.subplots_adjust(left=0.16, right=0.97, top=0.94, bottom=0.2)
		view.canvas.draw_idle()

	# ------------------------------------------------------------------
	# Contornos ECTb
	# ------------------------------------------------------------------
	def _on_contour_visibility_changed(self, *_args):
		self._show_center_contour = bool(self.center_check.isChecked())
		self._show_endo_contour = bool(self.endo_check.isChecked())
		self._show_epi_contour = bool(self.epi_check.isChecked())
		self._refresh_all_overlays()

	def _ensure_wall_result(self, stage: str):
		"""Calcula contornos ECTb de la etapa para el overlay irregular por gate/corte."""
		view = self._views[stage]
		main = self._main
		view.wall_result = None
		view.wall_reason = ""
		study = self._stage_study(stage)
		seg = self._stage_seg_ring(stage)
		if main is None or study is None or seg is None:
			view.wall_reason = "Falta estudio o segmentación anular: procesá en la ventana principal."
			return
		pixel_spacing = getattr(study, "pixel_spacing", None)
		slice_mm = getattr(study, "z_spacing_mm", None)
		cube = getattr(study, "cube", None)
		if not pixel_spacing or slice_mm is None or cube is None:
			view.wall_reason = "El estudio no trae spacing o cubo gated válido."
			return
		view.px_mm = max(1e-6, float(np.mean([abs(float(pixel_spacing[0])), abs(float(pixel_spacing[1]))])))
		try:
			check = getattr(main, "gate_dropout_check", None)
			if hasattr(main, "_apply_gate_dropout_correction") and check is not None and check.isChecked():
				cube, _ = main._apply_gate_dropout_correction(cube, log=False)
			cfg = main.ectb_config() if hasattr(main, "ectb_config") else None
			result = analyze_lv_ectb(cube, seg, (float(pixel_spacing[0]), float(pixel_spacing[1])), float(slice_mm), cfg)
			if getattr(result, "available", False):
				view.wall_result = result
			else:
				view.wall_reason = str(getattr(result, "reason", "") or "PARED no pudo cuantificar.")
		except Exception as exc:
			view.wall_reason = f"Error al calcular contornos de pared: {exc}"

	def _refresh_all_overlays(self):
		for stage in STAGES:
			self._refresh_wall_overlay(stage)

	def _refresh_wall_overlay(self, stage: str):
		view = self._views[stage]
		if view.wall is None or view.cube is None:
			return
		result = view.wall_result
		seg = self._stage_seg_ring(stage) if result is not None else None
		if result is None or seg is None:
			view.wall.preview.set_overlay_contours([])
			self._set_note(view, view.wall_reason or "Sin segmentación anular: no hay contornos para dibujar.")
			return
		gate = int(view.wall.current_gate_index())
		sl = int(view.wall.current_slice_index())
		contours = self._contours_for_gate_slice(result, seg, sl, gate, px_mm=view.px_mm)
		view.wall.preview.set_overlay_contours(contours)
		valid = tuple(int(v) for v in getattr(result, "valid_slices", ()) or ())
		if sl not in valid:
			self._set_note(view, f"Corte {sl + 1} sin contornos de pared ({len(valid)} cortes con contorno).")
		elif not contours:
			self._set_note(view, "Contornos ocultos: activá centro, endocardio o epicardio.")
		else:
			self._set_note(view, f"Contornos de pared · corte {sl + 1} · gate {gate + 1}", ok=True)

	@staticmethod
	def _set_note(view: _StageView, message: str, *, ok: bool = False):
		view.note.setText(str(message))
		view.note.setStyleSheet(f"color:{GREEN if ok else AMBER}; font-size:10px;")

	def _set_status(self, message: str, *, ok: bool = False):
		self.status_label.setText(str(message))
		self.status_label.setStyleSheet(f"color:{GREEN if ok else AMBER};")

	def _contours_for_gate_slice(self, result, seg, slice_index: int, gate_index: int, px_mm: float | None = None):
		if slice_index not in result.valid_slices:
			return []
		row = result.valid_slices.index(slice_index)
		centers = np.asarray(getattr(seg, "center_per_slice", np.empty((0, 2))), dtype=np.float64)
		if slice_index >= centers.shape[0]:
			return []
		cy, cx = float(centers[slice_index, 0]), float(centers[slice_index, 1])
		px_mm = max(float(self._wall_px_mm if px_mm is None else px_mm), 1e-6)
		n_ang = int(result.center_radii_mm.shape[-1])
		angles = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
		if gate_index < 0 or gate_index >= int(result.center_radii_mm.shape[0]):
			return []

		out = []
		for key, color, radii in (
			("center", QColor(255, 210, 63, 180), result.center_radii_mm[gate_index, row]),
			("endo", QColor(80, 160, 255), result.endo_radii_mm[gate_index, row]),
			("epi", QColor(80, 220, 120), result.epi_radii_mm[gate_index, row]),
		):
			if key == "center" and not bool(getattr(self, "_show_center_contour", True)):
				continue
			if key == "endo" and not bool(getattr(self, "_show_endo_contour", True)):
				continue
			if key == "epi" and not bool(getattr(self, "_show_epi_contour", True)):
				continue
			r_px = np.asarray(radii, dtype=np.float64) / px_mm
			xs = cx + r_px * np.cos(angles)
			ys = cy + r_px * np.sin(angles)
			out.append((color, np.stack([ys, xs], axis=1)))
		return out
