"""SINCRO - ui.main_window.

Ventana principal con controles de procesamiento y vista previa interactiva.
"""
from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from time import perf_counter

import numpy as np
from PyQt6.QtCore import QSize, Qt, QSettings, QTimer, QEventLoop
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QGuiApplication, QIcon, QMovie, QPixmap, QColor, QImage, QCursor, QPainter, QPen
from PyQt6.QtWidgets import (
	QApplication,
	QFileDialog,
	QCheckBox,
	QComboBox,
	QDoubleSpinBox,
	QDialog,
	QDialogButtonBox,
	QFormLayout,
	QFrame,
	QGroupBox,
	QHBoxLayout,
	QGridLayout,
	QLabel,
	QLineEdit,
	QMainWindow,
	QMenu,
	QMessageBox,
	QPushButton,
	QPlainTextEdit,
	QProgressBar,
	QProgressDialog,
	QScrollArea,
	QSpinBox,
	QSlider,
	QSizePolicy,
	QSplitter,
	QTabWidget,
	QTextEdit,
	QToolButton,
	QVBoxLayout,
	QWidget,
)

from core import dicom_loader
from core import normal_db
from core.col_registry import register_all_colormaps, available_colormaps
from core.aha_segments import (
	SECTOR_TO_SEGMENT_APICAL,
	SECTOR_TO_SEGMENT_BASAL,
	SECTOR_TO_SEGMENT_MEDIO,
	map_to_17_segments,
	phase_by_segment,
	territory_analysis,
)
from core.export_manager import export_all
from core.filling_metrics import compute_filling_metrics, format_pfr, format_tvmax
from core.metric_explanations import explanation_tooltip
from core.gate_dropout import analyze_gate_dropout, correct_last_gate_dropout
from core.perfusion_texture import (
	combine_perfusion_phase,
	perfusion_texture_by_segment,
)
from core.segmental_report import SEGMENT_NAMES, build_segmental_report
from core.stress_rest import compare_stress_rest
from core.perfusion_quant import perfusion_by_segment, perfusion_quant_summary
from core.executive_summary import build_executive_summary
from core.intestinal_subtraction import apply_intestinal_subtraction
from core.ectb_lv import (
	EF_REGRESSIONS,
	ECTbLVConfig,
	analyze_lv_ectb,
	convert_ef_pct,
	wall_segmentation_from_ectb,
)
from core.logging_config import get_logger
from core.metrics import calculate_phase_metrics
from core.phase_analysis import phase_analysis
from core.robustness import (
	bootstrap_phase_metrics,
	calculate_segmental_metrics,
	roi_sensitivity_analysis,
)
from core.segmentation import segment_myocardium
from report.report_generator import generate_report
from report.html_report import generate_html_report
from viz.histogram import build_phase_histogram, save_histogram
from viz.polar_map import (
	build_clinical_phase_panel,
	build_polar_map,
	save_clinical_phase_panel,
	save_polar_map,
)

from core.dual_stage import DualSession
from ui.cine_widget import CineWidget, RangeSlider, VerticalColorStrip
from ui.collapsible import CollapsibleSection, slugify_section_key
from ui.floating_toolbar import FloatingToolbar
from ui.managers import (
	CineManager,
	CompareManager,
	PresetManager,
	ProcessingManager,
	ReportManager,
	ROIManager,
)
from version import __version__


RAW_PHASE_QC_AMP_FILTER = 0.10
CLINICAL_PHASE_AMP_FILTER_DEFAULT = 0.40
LOW_CONFIDENCE_TAIL_DEG = 120.0
LOW_CONFIDENCE_TAIL_WARN_PCT = 5.0


def _stage_prop(stage_key: str, field: str) -> property:
	"""Property que delega un atributo legacy en la etapa fija de la DualSession."""
	def fget(self):
		return getattr(self._dual_session().stage(stage_key), field)

	def fset(self, value):
		setattr(self._dual_session().stage(stage_key), field, value)

	return property(fget, fset)


def _recon_stage_prop(field: str) -> property:
	"""Property que delega en la etapa del slot de recon (`_cine_crudo_recon_stage`).

	Conserva la semántica actual (un solo juego de atributos que sigue a la etapa
	reconstruida) pero con almacenamiento POR ETAPA: reconstruir la segunda etapa
	ya no pisa los resultados de la primera.
	"""
	def fget(self):
		stage = getattr(self, "_cine_crudo_recon_stage", "stress")
		return getattr(self._dual_session().stage(stage), field)

	def fset(self, value):
		stage = getattr(self, "_cine_crudo_recon_stage", "stress")
		setattr(self._dual_session().stage(stage), field, value)

	return property(fget, fset)


class MainWindow(QMainWindow):
	# Registro de layouts de presentación del montaje SA/VLA/HLA.
	# per_strip = cortes visibles por tira/eje (None = todos). panel_in = pulgadas por panel.
	# Agregar un layout nuevo = añadir una entrada acá; el combo se puebla solo.
	MONTAGE_LAYOUTS: dict[str, dict] = {
		"denso": {"label": "Denso (todos)", "per_strip": None, "panel_in": 1.4},
		"grande": {"label": "Grande (16)", "per_strip": 16, "panel_in": 2.15},
		"cuadrante": {"label": "Cuadrante (8)", "per_strip": 8, "panel_in": 1.7},
		"nueve": {"label": "9 cortes", "per_strip": 9, "panel_in": 1.7},
		"ocho": {"label": "8 cortes", "per_strip": 8, "panel_in": 1.7},
	}

	# ── Plan C (PERFU_RyE): atributos legacy delegados en DualSession ──────────
	# Los handlers existentes siguen leyendo/escribiendo los mismos nombres; el
	# almacenamiento real es por etapa (core/dual_stage.StageState), así que
	# reconstruir la segunda etapa ya NO pisa los resultados de la primera.
	# Etapa fija — esfuerzo (primario del cine crudo):
	cine_crudo_motion_result = _stage_prop("stress", "motion_result")
	cine_crudo_corrected_projections = _stage_prop("stress", "corrected_projections")
	cine_crudo_ref_index = _stage_prop("stress", "ref_index")
	# Etapa fija — reposo (secundario "compare"):
	cine_crudo_motion_result_compare = _stage_prop("rest", "motion_result")
	cine_crudo_corrected_projections_compare = _stage_prop("rest", "corrected_projections")
	cine_crudo_ref_index_compare = _stage_prop("rest", "ref_index")
	compare_raw_study = _stage_prop("rest", "raw_study")
	compare_raw_path = _stage_prop("rest", "source_path")
	# Siguen al slot de recon (_cine_crudo_recon_stage), como hasta ahora:
	cine_crudo_recon_result = _recon_stage_prop("recon_result")
	cine_crudo_recon_result_phase = _recon_stage_prop("recon_result_phase")
	cine_crudo_recon_study = _recon_stage_prop("recon_study")
	cine_crudo_raw_study_for_recon = _recon_stage_prop("raw_study_for_recon")
	cine_crudo_cut_study = _recon_stage_prop("cut_study")
	cine_crudo_cut_source_label = _recon_stage_prop("cut_source_label")
	cine_crudo_reoriented_ungated = _recon_stage_prop("reoriented_ungated")
	cine_crudo_reoriented_gated = _recon_stage_prop("reoriented_gated")
	cine_crudo_reoriented_gated_phase = _recon_stage_prop("reoriented_phase")
	cine_crudo_reoriented_mf = _recon_stage_prop("reoriented_mf")
	cine_crudo_reoriented_ct = _recon_stage_prop("reoriented_ct")
	cine_crudo_axes_for_export = _recon_stage_prop("axes")
	cine_crudo_axes_for_export_ungated = _recon_stage_prop("axes_ungated")
	cine_crudo_axes_for_export_mf = _recon_stage_prop("axes_mf")
	cine_crudo_axes_for_export_ct = _recon_stage_prop("axes_ct")
	cine_crudo_cut_thickness_mm = _recon_stage_prop("cut_thickness_mm")
	# Slots del montaje por etapa (ahora vistas del MISMO almacenamiento por etapa;
	# la copia stress/rest del flujo actual pasa a ser idempotente):
	cine_crudo_axes_for_export_stress = _stage_prop("stress", "axes")
	cine_crudo_axes_for_export_rest = _stage_prop("rest", "axes")
	cine_crudo_axes_for_export_ungated_stress = _stage_prop("stress", "axes_ungated")
	cine_crudo_axes_for_export_ungated_rest = _stage_prop("rest", "axes_ungated")
	cine_crudo_axes_for_export_mf_stress = _stage_prop("stress", "axes_mf")
	cine_crudo_axes_for_export_mf_rest = _stage_prop("rest", "axes_mf")
	cine_crudo_cut_thickness_mm_rest = _stage_prop("rest", "cut_thickness_mm")
	cine_crudo_rest_source_label = _stage_prop("rest", "cut_source_label")

	def _dual_session(self) -> DualSession:
		"""Sesión dual de perfusión; se crea perezosa para tolerar accesos tempranos."""
		s = getattr(self, "_session", None)
		if s is None:
			s = DualSession()
			self._session = s
		return s

	def __init__(self, initial_path: str | None = None):
		super().__init__()
		self.setWindowTitle(f"GammaSync v{__version__} - Interfaz de procesado")
		screen = QApplication.primaryScreen()
		if screen is not None:
			available = screen.availableGeometry()
			self.resize(max(1200, int(available.width() * 0.92)), max(800, int(available.height() * 0.90)))
		else:
			self.resize(1500, 920)
		self._set_window_icon()

		self.study = None
		# Flujo ida-y-vuelta (Fase A): registro de pasos + pila deshacer/rehacer.
		from ui.managers.pipeline_history import PipelineHistory
		self.pipeline_history = PipelineHistory()
		self._undo_suspended = False
		self._register_pipeline_steps()
		self.pipeline_history.add_listener(self._refresh_pipeline_step_bar)
		self.axis_companions: dict[str, object] = {}
		self.seg = None
		# Copia de la segmentación ANULAR, intacta aunque `self.seg` se reemplace por
		# la pared ECTb. El ECTb necesita un anillo con cavidad para tirar sus rayos:
		# si se le pasa su propia salida irregular como semilla, deja de encontrar
		# contornos. Toda recalculación de ECTb debe partir de acá.
		self.seg_ring_base = None
		self.intestinal_subtraction_info = None
		# Diagnóstico de calibración FEVI/volumen (barridos [DIAG-*] en el log).
		# Poner en True para reactivar los logs de calibración al evaluar estudios
		# nuevos (sweep de cavity_frac, radios ED/ES, epicardio por gate, FWHM
		# subpíxel, upsampling). Off por defecto en uso clínico normal.
		self.fevi_diag_enabled = False
		self.phase_result_raw = None
		self.phase_result = None
		self.metrics_raw = None
		self.metrics = None
		self.phase_qc = None
		self.aha = None
		self.phase_by_seg = None
		self.territory = None
		# Centro de cavidad fijado por el operador (clic), por corte:
		# {slice_idx: (cy, cx)}. Manda sobre el centroide automático y se propaga
		# a radios, ángulo AHA y fase. Vacío = centro 100% automático.
		self.manual_center_per_slice: dict[int, tuple[float, float]] = {}
		# Estudio de comparación (típicamente REST vs el actual STRESS) para el
		# análisis stress/rest de disincronía (stunning isquémico, Camilletti 2015).
		self.compare_metrics = None
		self.compare_label = None
		self.compare_ef = None
		self.compare_bundle = None
		self.compare_raw_study = None
		self.compare_raw_path = ""
		self.dual_mode_active = False
		# Zoom de la elipse de reorientación bloqueado entre etapas (mismo zoom).
		self._reorient_locked_voi = None
		self._reorient_locked_stage = None
		# Semilla de orientación (eje largo + rango de cortes + espesor) heredada
		# de la 1ra etapa a la 2da. A diferencia del zoom, NO se bloquea: editable.
		self._reorient_seed = None
		self._reorient_seed_stage = None
		self.primary_manual_rois_text = ""
		self.compare_manual_rois_text = ""
		self.primary_manual_rois_autogenerated = False
		self.compare_manual_rois_autogenerated = False
		self._programmatic_manual_rois_update = False
		self.active_cine_source = "primary"
		self.preview_zoom: dict[str, float] = {}
		self.preview_base_sizes: dict[str, QSize] = {}
		self.preview_pixmaps: dict[str, QPixmap] = {}
		self.preview_movies: dict[str, QMovie] = {}
		self.preview_zoom_labels: dict[str, QLabel] = {}
		self.polar_cine_toggle_btn: QToolButton | None = None
		self.polar_perf_view_perf_btn: QToolButton | None = None
		self.polar_perf_view_cine_btn: QToolButton | None = None
		self.polar_view_mode = "perfusion"  # "perfusion" | "cine" dentro de polar_perfusion_directa
		self._report_editor_html = ""  # HTML del editor de informe (si se usó)
		self._cached_exec_html = ""  # Resumen ejecutivo cacheado del pipeline
		# Colormap de pantalla del mapa polar de perfusión (independiente del informe).
		# Default = mismo cmap que el informe para no cambiar el look inicial.
		self.polar_perf_screen_cmap = "odyssey_cool"
		self._polar_perf_cart_cache = None
		self.polar_cine_preview_frames: list[QPixmap] = []
		self.polar_cine_preview_index = 0
		self.polar_cine_playing = False
		self.polar_cine_timer = QTimer(self)
		self.polar_cine_timer.timeout.connect(self._advance_polar_cine_frame)
		# Cine crudo (proyecciones SPECT)
		self.cine_crudo_frames: list[QPixmap] = []
		self.cine_crudo_index = 0
		self.cine_crudo_playing = False
		self.cine_crudo_direction = 1  # para modo rebote
		self.cine_crudo_matrix_txt = ""
		self.cine_crudo_timer = QTimer(self)
		self.cine_crudo_timer.timeout.connect(self._advance_cine_crudo_frame)
		self.cine_crudo_play_btn: QToolButton | None = None
		self.cine_crudo_correct_btn: QToolButton | None = None
		self.cine_crudo_accept_btn: QToolButton | None = None
		self.cine_crudo_reject_btn: QToolButton | None = None
		# True una vez que el usuario confirma la corrección de movimiento con "Aplicar".
		self._cine_crudo_motion_accepted = False
		self.cine_crudo_compare_check: QCheckBox | None = None
		self.cine_crudo_mask_check: QCheckBox | None = None
		self.cine_crudo_seed_btn: QToolButton | None = None
		self.cine_crudo_seed: tuple[float, float] | None = None
		self.cine_crudo_seed_compare: tuple[float, float] | None = None
		self.cine_crudo_seed_mode = False
		# Etapa que reciben las herramientas del crudo en modo dual: "stress" (primario) | "rest" (secundario).
		self._cine_crudo_active_stage = "stress"
		# Con dos etapas cargadas, el pipeline debe ser dual por defecto sin que el
		# operador recuerde seleccionar "Ambas". Un cambio manual del selector es
		# un override temporal y permite procesar una etapa aislada.
		self._dual_pipeline_auto_enabled = True
		self._dual_pipeline_manual_stage_override: str | None = None
		self.cine_crudo_band_upper: float | None = None
		self.cine_crudo_band_lower: float | None = None
		self.cine_crudo_compare_line_y: float | None = None
		self._cine_crudo_drag_marker: str | None = None
		self._cine_crudo_hover_marker: str | None = None
		self._cine_crudo_last_drag_refresh = 0.0
		self.cine_crudo_ref_index: int | None = None
		self.cine_crudo_ref_index_compare: int | None = None
		self.cine_crudo_corrected_projections = None
		self.cine_crudo_corrected_projections_compare = None
		self.cine_crudo_motion_result = None
		self.cine_crudo_motion_result_compare = None
		self.cine_crudo_recon_result = None
		# Pasajero de fase (FBP): volumen paralelo que se genera junto al nítido
		# cuando NÍTIDA (RR) está activa; la fase se calcula sobre él (ver paso 4).
		self.cine_crudo_recon_result_phase = None
		self.cine_crudo_raw_study_for_recon = None
		self._cine_crudo_recon_stage = "stress"
		# Sustracción de fondo sobre el crudo (herramienta de la ventana de
		# reconstrucción). Solo alimenta la recon cuando el impacto es "toda la cadena".
		self._raw_bg_spec: dict[str, dict] = {}
		self.cine_crudo_recon_study = None
		self.cine_crudo_cut_study = None
		self.cine_crudo_cut_source_label = ""
		self.cine_crudo_axes_for_export: dict[str, np.ndarray] = {}
		self.cine_crudo_axes_for_export_ungated: dict[str, np.ndarray] = {}
		# Fuente del montaje clínico: "ungated" (perfusión estática, con Denoise+;
		# la imagen del informe) o "gated" (cine en movimiento). Default ungated.
		self.cine_crudo_montage_source: str = "ungated"
		# Slots duales por etapa para el montaje comparativo stress/rest. El montaje
		# usa estos (no el genérico axes_for_export, que es la etapa mostrada).
		self.cine_crudo_axes_for_export_stress: dict[str, np.ndarray] = {}
		self.cine_crudo_axes_for_export_rest: dict[str, np.ndarray] = {}
		self.cine_crudo_axes_for_export_ungated_stress: dict[str, np.ndarray] = {}
		self.cine_crudo_axes_for_export_ungated_rest: dict[str, np.ndarray] = {}
		self.cine_crudo_rest_source_label = ""
		self.cine_crudo_cut_thickness_mm = 0.0
		self.cine_crudo_cut_thickness_mm_rest = 0.0
		# Límites Base/Ápex por etapa (1-based en UI). Evita que stress/rest
		# compartan accidentalmente los mismos markers en modo dual.
		# apex_1=None significa "hasta el último corte disponible" (se resuelve
		# recién cuando se conoce n_slices de esa etapa).
		self._cine_crudo_cut_limits_by_stage = {
			"stress": {"base_1": 1, "apex_1": None},
			"rest": {"base_1": 1, "apex_1": None},
		}
		# Recorte del montaje: "limits" (markers base/ápex) o "voi" (elipse VOI).
		self.cine_crudo_montage_crop_mode = "limits"
		# Plantillas de presentación del montaje.
		self.cine_crudo_montage_template = "nueve"
		self.cine_crudo_montage_cut_zoom = 1.0
		# Colormap y window level (percentiles) del montaje clínico.
		self.cine_crudo_montage_cmap = "odyssey_cool"
		self.cine_crudo_montage_win_low = 2.0
		self.cine_crudo_montage_win_high = 99.5
		# Motor de color del cine de proyecciones (preview crudo): colormap + ventana
		# 0..200% de p99. Default gris con ventana 0..100% = comportamiento histórico.
		self.cine_crudo_screen_cmap = "gray"
		self.cine_crudo_screen_win_low = 0.0
		self.cine_crudo_screen_win_high = 100.0
		# Modo de ventaneo del montaje: "percentil" (histórico, spinboxes) o
		# "lineal" (motor unificado: normaliza el volumen por min/máx y aplica la
		# ventana 0..200% del RangeSlider, idéntica al cine/diálogo). El modo lineal
		# comparte escala entre cortes (norm por volumen, no por corte).
		self.cine_crudo_montage_win_mode = "percentil"
		self.cine_crudo_montage_lin_low = 0.0   # fracción 0..2 (handle base)
		self.cine_crudo_montage_lin_high = 1.0  # fracción 0..2 (handle top)
		# Centrar cada corte en su casilla (centroide → centro del panel).
		self.cine_crudo_montage_center_cuts = False
		# Filtro visual del montaje clínico (idéntico a la vista de cortes): tipo de
		# interpolación de display + gaussiano extra. No altera datos ni análisis.
		self.cine_crudo_montage_interp = "Bilineal"
		self.cine_crudo_montage_smooth = 0.0
		# Cine del montaje clínico (solo con fuente Gated): recorre los gates en
		# vivo. Default 40 ms (configurable). Independiente del cine de proyecciones.
		self.cine_crudo_montage_cine_playing: bool = False
		self.cine_crudo_montage_cine_frame: int = 0
		self._montage_cine_timer = QTimer(self)
		self._montage_cine_timer.timeout.connect(self._advance_montage_cine_frame)
		# Caché de frames pre-renderizados del cine del montaje: se generan TODOS
		# al dar Play (preload) y luego el timer solo blit-ea el QPixmap (instantáneo).
		self._montage_cine_frames: list = []
		self._montage_cine_frames_sig = None
		# Ventanas por tira/eje (1-based): inicio y cantidad visible.
		self.cine_crudo_stripe_start = {"SA": 1, "VLA": 1, "HLA": 1}
		self.cine_crudo_stripe_count = {"SA": 999, "VLA": 999, "HLA": 999}
		# Navegación independiente por fila del montaje dual. El dict histórico
		# anterior se conserva como fallback para sesiones de una sola etapa.
		self.cine_crudo_stripe_start_by_stage = {
			"ESFUERZO": {"SA": 1, "VLA": 1, "HLA": 1},
			"REPOSO": {"SA": 1, "VLA": 1, "HLA": 1},
		}
		# Offsets de alineación de reposo respecto de esfuerzo (px de corte).
		self.cine_crudo_rest_offset = {"SA": 0, "VLA": 0, "HLA": 0}
		# Rango de frames/gates para visualizar cada corte en el montaje (1-based).
		self.cine_crudo_gate_from = 1
		self.cine_crudo_gate_to = 1
		self._montage_drag_axis: str | None = None
		self._montage_drag_mode: str | None = None
		self._montage_drag_start_x: float | None = None
		self._montage_drag_start_off: int = 0
		self._montage_drag_start_gate: int = 1
		self._montage_drag_selection_key: str | None = None
		# Foco persistente de rueda/teclado: a diferencia de la selección múltiple,
		# siempre hay UNA tira que recibe navegación.
		self._montage_focus_selection_key: str = "ESFUERZO:SA"
		self.cine_crudo_focused_stripe: str = "ESFUERZO:SA"
		self._montage_render_meta: dict = {}
		# Identificadores independientes por fila: "ESFUERZO:SA", "REPOSO:SA".
		# La selección histórica simple se conserva como foco de teclado.
		self.cine_crudo_selected_stripe: str = "SA"
		self.cine_crudo_selected_stripes: set[str] = {"ESFUERZO:SA"}
		self.cine_crudo_focused_stripe: str = "ESFUERZO:SA"
		self._montage_refresh_timer = QTimer(self)
		self._montage_refresh_timer.setSingleShot(True)
		self._montage_refresh_timer.timeout.connect(self._show_cine_crudo_sa_montage)
		# Fast-pass: durante la interacción se rinde a baja resolución (ágil) y al
		# soltar se re-rinde en HQ 512px (nítido). panel px efectivo del lienzo.
		self._montage_panel_px: int = 512
		self._montage_hq_timer = QTimer(self)
		self._montage_hq_timer.setSingleShot(True)
		self._montage_hq_timer.timeout.connect(self._render_montage_hq)
		# Recoloreo: display inmediato en FastTransformation y repaint nítido diferido.
		self._montage_recolor_smooth_timer = QTimer(self)
		self._montage_recolor_smooth_timer.setSingleShot(True)
		self._montage_recolor_smooth_timer.timeout.connect(self._montage_recolor_smooth_repaint)
		# Recompute por rama del QC de reconstrucción: al cambiar un filtro se
		# recomputa SOLO esa rama (ungated o gated) sobre las proyecciones ya
		# corregidas, con debounce para no recalcular en cada tecla del spin.
		self._recon_recompute_ung_timer = QTimer(self)
		self._recon_recompute_ung_timer.setSingleShot(True)
		self._recon_recompute_ung_timer.timeout.connect(self._recompute_recon_branch_ungated)
		self._recon_recompute_gated_timer = QTimer(self)
		self._recon_recompute_gated_timer.setSingleShot(True)
		self._recon_recompute_gated_timer.timeout.connect(self._recompute_recon_branch_gated)
		self._preview_scrollers: dict[str, QScrollArea] = {}
		self._preview_pan_active = False
		self._preview_pan_anchor = None
		self.cine_crudo_preview_mode: str | None = None
		self._cine_crudo_dual_render_meta: dict = {}
		# Contexto visual temporal de ejecución dual (paso/etapa/índice), usado
		# para rotular claramente Esfuerzo vs Reposo durante procesos largos.
		self._cine_crudo_dual_context: dict | None = None
		self._cine_crudo_cut_limits_meta: dict | None = None
		self._tooltips_cache_main: dict[QWidget, str] = {}
		self._ui_show_helpers = True
		self._ui_enable_tooltips = True
		self._ui_compact_controls = False
		self._perfusion_source = self.PERFUSION_SOURCE_ED
		self.compare_axes_preview_frames: list[QPixmap] = []
		self.compare_axes_preview_index = 0
		self.compare_axes_playing = False
		self._compare_axes_dirty_pending = False
		self.compare_interactive_fast_mode = False
		self.compare_axes_cine_timer = QTimer(self)
		self.compare_axes_cine_timer.timeout.connect(self._advance_compare_axes_frame)
		self.compare_axes_refresh_timer = QTimer(self)
		self.compare_axes_refresh_timer.setSingleShot(True)
		self.compare_axes_refresh_timer.timeout.connect(self._refresh_compare_axes_panel_now)
		self._deferred_hq_job: str = ""
		self._deferred_compare_bundle: dict | None = None
		self._deferred_compare_left_label = ""
		self._deferred_compare_right_label = ""
		self._deferred_hq_generation = 0
		self._deferred_hq_running = False
		self._deferred_hq_timer = QTimer(self)
		self._deferred_hq_timer.setSingleShot(True)
		self._deferred_hq_timer.timeout.connect(self._run_deferred_hq_render)
		self._gate_roi_recalc_timer = QTimer(self)
		self._gate_roi_recalc_timer.setSingleShot(True)
		self._gate_roi_recalc_timer.timeout.connect(self._on_gate_roi_recalc_timeout)
		self._lazy_render_pending_tabs: set[str] = set()
		self._cache_study_sig = ""
		self._cache_seg_sig = ""
		self._cache_phase_sig = ""
		self._cache_output_sig = ""
		self._cache_tab_output_sigs: dict[str, str] = {}
		self._last_primary_path = ""
		self._last_browse_dir = ""
		self.advanced_mode_enabled = False
		self._basic_tab_order = [
			"slices_fase",
			"polar_combo",
			"delta_combo",
			"histograma",
			"comparacion_stress_rest",
			"ungated",
			"cine_crudo",
		]
		self._advanced_extra_tab_order = [
			"polar_perfusion_directa",
			"comparacion_ejes",
			"panel_funcional_gated",
			"bullseye_directo",
			"guia_fase_vi",
		]
		# Fase 1 fusión: cuando la ventana de Preparación aloja el panel cine_crudo,
		# la pestaña se saca del QTabWidget y vuelve al cerrar la ventana (reversible).
		self._cine_crudo_reparented = False

		self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output_demo")
		os.makedirs(self.output_dir, exist_ok=True)
		self.compare_output_dir = os.path.join(self.output_dir, "_compare")
		os.makedirs(self.compare_output_dir, exist_ok=True)
		self.presets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "presets")
		os.makedirs(self.presets_dir, exist_ok=True)
		self.presets_path = os.path.join(self.presets_dir, "processing_presets.json")
		self._presets_data = self._load_presets_store()

		central = QWidget()
		self.setCentralWidget(central)

		splitter = QSplitter(Qt.Orientation.Horizontal)
		splitter.setChildrenCollapsible(False)
		splitter.setOpaqueResize(True)
		splitter.setHandleWidth(10)
		left = self._build_sidebar()

		self.file_edit = QLineEdit()
		self.file_edit.setPlaceholderText("Ruta al DICOM gated reconstruido...")
		browse_btn = QPushButton("Abrir...")
		browse_btn.clicked.connect(self._browse_file)

		# Menú de carpetas recientes / favoritas
		self._recent_dirs_btn = QToolButton()
		self._recent_dirs_btn.setText("▼")
		self._recent_dirs_btn.setToolTip("Carpetas recientes y favoritas. Click en una para navegar. 'Guardar esta carpeta' la marca como favorita.")
		self._recent_dirs_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
		self._recent_dirs_menu = QMenu(self)
		self._recent_dirs_btn.setMenu(self._recent_dirs_menu)
		self._rebuild_recent_dirs_menu()
		self._favorite_dir_btn = QToolButton()
		self._favorite_dir_btn.setText("★")
		self._favorite_dir_btn.setToolTip("Guardar como favorita la carpeta actual (la última usada o la del estudio cargado).")
		self._favorite_dir_btn.clicked.connect(self._favorite_current_browse_dir)

		file_row = QHBoxLayout()
		file_row.addWidget(self.file_edit, 1)
		file_row.addWidget(browse_btn)
		file_row.addWidget(self._recent_dirs_btn)
		file_row.addWidget(self._favorite_dir_btn)
		file_box = QGroupBox("Estudio")
		file_box_layout = QVBoxLayout(file_box)
		file_box_layout.addLayout(file_row)
		self._sidebar_layout.addWidget(file_box)

		controls_box = QGroupBox("Procesamiento")
		controls_form = QFormLayout(controls_box)
		self.controls_form = controls_form

		self.seg_method = QComboBox()
		self.seg_method.addItems(["auto", "threshold", "manual"])

		# Qué geometría de ROI alimenta la fase: el anillo circular de siempre o la
		# pared irregular que traza el ECTb siguiendo el máximo de cuentas.
		self.roi_source_combo = QComboBox()
		self.roi_source_combo.addItem("Contornos Irregulares", "ectb_wall")
		self.roi_source_combo.addItem("Anillo (ROI clásica)", "ring")
		self.roi_source_combo.setCurrentIndex(0)

		# El centro del VI se venía tomando como centroide de la máscara de
		# miocardio, que se corre hacia el sector de mayor captación. Esta opción
		# lo recalcula sobre la cavidad. Apagada por defecto porque cambia todos
		# los números derivados (fase, AHA, territorios).
		self.cavity_center_check = QCheckBox("Centrar ROI en la cavidad")
		self.cavity_center_check.setChecked(False)
		self.cavity_center_check.toggled.connect(self._on_cavity_center_toggled)

		# Centro manual: el operador hace clic en la cavidad y ese centro manda
		# sobre el automático. Resuelve los casos donde ningún método auto acierta.
		self.manual_center_check = QCheckBox("Centro manual (clic en cavidad)")
		self.manual_center_check.setChecked(False)
		self.manual_center_check.toggled.connect(self._on_manual_center_mode_toggled)
		self.manual_center_all_check = QCheckBox("Aplicar el clic a todos los cortes")
		self.manual_center_all_check.setChecked(False)
		self.manual_center_clear_btn = QPushButton("Limpiar centros manuales")
		self.manual_center_clear_btn.clicked.connect(self._clear_manual_centers)

		self.threshold_spin = QDoubleSpinBox()
		self.threshold_spin.setRange(0.01, 0.90)
		self.threshold_spin.setSingleStep(0.01)
		self.threshold_spin.setValue(0.35)

		self.sigma_spin = QDoubleSpinBox()
		self.sigma_spin.setRange(0.0, 6.0)
		self.sigma_spin.setSingleStep(0.1)
		self.sigma_spin.setValue(0.0)

		self.harmonics_spin = QSpinBox()
		self.harmonics_spin.setRange(1, 4)
		self.harmonics_spin.setValue(1)

		self.phase_threshold_spin = QDoubleSpinBox()
		self.phase_threshold_spin.setRange(0.01, 0.80)
		self.phase_threshold_spin.setSingleStep(0.01)
		self.phase_threshold_spin.setValue(CLINICAL_PHASE_AMP_FILTER_DEFAULT)

		self.normalize_check = QCheckBox("Normalizar referencia de fase")
		self.normalize_check.setChecked(False)

		# --- Corrección de dropout del último gate (ECTb 22.8) ---
		# Por la variabilidad del R-R, el último gate acumula menos cuentas. Se
		# escala para igualar al primero antes de segmentar y de calcular fase.
		self.gate_dropout_check = QCheckBox("Corregir dropout del último gate")
		self.gate_dropout_check.setChecked(True)
		self.gate_dropout_check.toggled.connect(self._on_gate_dropout_toggled)
		self.gate_dropout_status = QLabel("Sin estudio cargado.")
		self.gate_dropout_status.setWordWrap(True)
		self.gate_dropout_status.setStyleSheet("color:#35506a;")
		self.gate_dropout_help_btn = QToolButton()
		self.gate_dropout_help_btn.setText("?")
		self.gate_dropout_help_btn.setToolTip("Explica qué es el dropout del último gate y cuándo conviene corregirlo.")
		self.gate_dropout_help_btn.clicked.connect(self.show_gate_dropout_help)
		gate_dropout_widget = QWidget()
		gate_dropout_layout = QVBoxLayout(gate_dropout_widget)
		gate_dropout_layout.setContentsMargins(0, 0, 0, 0)
		gate_dropout_layout.setSpacing(2)
		gate_dropout_row = QHBoxLayout()
		gate_dropout_row.setContentsMargins(0, 0, 0, 0)
		gate_dropout_row.setSpacing(4)
		gate_dropout_row.addWidget(self.gate_dropout_check, 1)
		gate_dropout_row.addWidget(self.gate_dropout_help_btn)
		gate_dropout_layout.addLayout(gate_dropout_row)
		gate_dropout_layout.addWidget(self.gate_dropout_status)

		self.global_intestinal_render_check = QCheckBox("Atenuación intestinal visual global")
		self.global_intestinal_render_check.setChecked(True)
		self.global_intestinal_render_check.toggled.connect(self._on_global_intestinal_render_toggled)

		# Base de datos normal (comparación de PSD/BW contra valores publicados).
		self.normal_sex_combo = QComboBox()
		self.normal_sex_combo.addItems(["Hombre", "Mujer"])
		self.normal_protocol_combo = QComboBox()
		self.normal_protocol_combo.addItems(["Stress", "Rest"])
		self.normal_db_combo = QComboBox()
		self.normal_db_combo.addItems(normal_db.available_datasets())

		self.auto_run_check = QCheckBox("Procesar automáticamente al cargar")
		self.auto_run_check.setChecked(True)
		self.auto_run_check.setToolTip("Si está activo, el estudio se procesa apenas se carga con los parámetros actuales.")

		register_all_colormaps()
		self._all_cmaps = available_colormaps()

		self.cmap_combo = QComboBox()
		self.cmap_combo.addItems(self._all_cmaps)
		self.cmap_combo.setCurrentText("french")

		self.visual_style_combo = QComboBox()
		self.visual_style_combo.addItems(["GammaSync", "Clinico"])
		self.visual_style_combo.setCurrentText("GammaSync")

		self.polar_rotation_spin = QSpinBox()
		self.polar_rotation_spin.setRange(-180, 180)
		self.polar_rotation_spin.setSingleStep(5)
		self.polar_rotation_spin.setValue(0)
		self.polar_rotation_spin.setSuffix("°")
		self.polar_perf_smooth_method_combo = QComboBox()
		self.polar_perf_smooth_method_combo.addItems(["Gaussiano", "Butterworth"])
		self.polar_perf_smooth_method_combo.setCurrentText("Gaussiano")
		self.polar_perf_smooth_strength_spin = QDoubleSpinBox()
		self.polar_perf_smooth_strength_spin.setRange(0.0, 16.0)
		self.polar_perf_smooth_strength_spin.setSingleStep(0.25)
		self.polar_perf_smooth_strength_spin.setDecimals(2)
		self.polar_perf_smooth_strength_spin.setValue(8.0)
		polar_perf_smooth_widget = QWidget()
		polar_perf_smooth_layout = QHBoxLayout(polar_perf_smooth_widget)
		polar_perf_smooth_layout.setContentsMargins(0, 0, 0, 0)
		polar_perf_smooth_layout.setSpacing(4)
		polar_perf_smooth_layout.addWidget(self.polar_perf_smooth_method_combo, 1)
		polar_perf_smooth_layout.addWidget(self.polar_perf_smooth_strength_spin)

		self.polar_cine_speed_spin = QSpinBox()
		self.polar_cine_speed_spin.setRange(40, 1000)
		self.polar_cine_speed_spin.setSingleStep(10)
		self.polar_cine_speed_spin.setValue(180)
		self.polar_cine_speed_spin.setSuffix(" ms")

		self.polar_compare_math_combo = QComboBox()
		self.polar_compare_math_combo.addItems(["Ninguna", "Suma", "Resta", "Multiplicación", "División"])
		self.polar_compare_math_combo.setCurrentText("Ninguna")
		self.polar_compare_term_a_combo = QComboBox()
		self.polar_compare_term_a_combo.addItems(["Esfuerzo", "Reposo"])
		self.polar_compare_term_a_combo.setCurrentText("Esfuerzo")
		self.polar_compare_term_b_combo = QComboBox()
		self.polar_compare_term_b_combo.addItems(["Esfuerzo", "Reposo"])
		self.polar_compare_term_b_combo.setCurrentText("Reposo")
		polar_math_terms = QWidget()
		polar_math_terms_layout = QHBoxLayout(polar_math_terms)
		polar_math_terms_layout.setContentsMargins(0, 0, 0, 0)
		polar_math_terms_layout.setSpacing(4)
		polar_math_terms_layout.addWidget(QLabel("A"))
		polar_math_terms_layout.addWidget(self.polar_compare_term_a_combo)
		polar_math_terms_layout.addWidget(QLabel("B"))
		polar_math_terms_layout.addWidget(self.polar_compare_term_b_combo)

		self.export_polar_mp4_check = QCheckBox("Exportar polar cine MP4")
		self.export_polar_mp4_check.setChecked(True)
		self.profile_timing_check = QCheckBox("Log tiempos > 0.5 s")
		self.profile_timing_check.setChecked(True)
		self.realtime_deferred_render_check = QCheckBox("Tiempo real (rápido) + HQ diferido")
		self.realtime_deferred_render_check.setChecked(True)

		self.manual_rois = QPlainTextEdit()
		self.manual_rois.setPlaceholderText(
			"Modo manual: slice,cy,cx,r_inner,r_outer\n"
			"ej: 9,12,11,4,7 | apex sin cavidad: 9,12,11,-,7"
		)
		self.manual_rois.setMaximumHeight(84)
		self.manual_rois.setToolTip("Cada línea define un slice. Tras segmentar en auto/threshold, GammaSync vuelca acá el ROI detectado para poder reproducirlo en manual.")
		self.manual_rois.textChanged.connect(self._on_manual_rois_text_changed)

		controls_form.addRow("Segmentación", self.seg_method)
		controls_form.addRow("ROI de análisis", self.roi_source_combo)
		controls_form.addRow(self.cavity_center_check)
		controls_form.addRow("Threshold", self.threshold_spin)
		controls_form.addRow("Smooth sigma", self.sigma_spin)
		controls_form.addRow("Harmonics", self.harmonics_spin)
		controls_form.addRow("Amplitude filter clínico", self.phase_threshold_spin)
		controls_form.addRow("Colormap fase", self.cmap_combo)
		controls_form.addRow("Estilo visual", self.visual_style_combo)
		controls_form.addRow("Rotación polar", self.polar_rotation_spin)
		controls_form.addRow("Suavizado polar", polar_perf_smooth_widget)
		controls_form.addRow("Velocidad polar cine", self.polar_cine_speed_spin)
		controls_form.addRow("Math polar stress/rest", self.polar_compare_math_combo)
		controls_form.addRow("Términos math", polar_math_terms)
		controls_form.addRow(self.export_polar_mp4_check)
		controls_form.addRow(self.profile_timing_check)
		controls_form.addRow(self.realtime_deferred_render_check)
		controls_form.addRow(self.normalize_check)
		controls_form.addRow(gate_dropout_widget)
		controls_form.addRow(self.global_intestinal_render_check)
		controls_form.addRow("Sexo (DB normal)", self.normal_sex_combo)
		controls_form.addRow("Protocolo (DB normal)", self.normal_protocol_combo)
		controls_form.addRow("DB normal", self.normal_db_combo)
		controls_form.addRow(self.auto_run_check)

		self.seg_method.setToolTip("auto: segmentación automática; threshold: umbral simple; manual: usa los ROIs que dibujes o pegues.")
		self.roi_source_combo.setToolTip(
			"Geometría que alimenta el análisis de fase.\n"
			"Contornos Irregulares (default): traza endo/epi por ángulo con espesor variable, estilo ECTb.\n"
			"Anillo: ROI circular clásica (centro + radio interno/externo).\n"
			"Podés comparar ambas en 'Vista asincronía' antes de decidir."
		)
		self.cavity_center_check.setToolTip(
			"Corrige el centro del VI.\n"
			"Apagado: el centro es el centroide de la máscara de miocardio, que se corre "
			"hacia el sector que más capta (pared lateral caliente, defecto, intestino pegado).\n"
			"Encendido: el centro se recalcula sobre la cavidad (hueco del anillo o zona "
			"hipocaptante interna).\n"
			"Afecta radios, contornos ECTb y la asignación angular a segmentos AHA, "
			"así que cambia los valores de fase. Reprocesa para verlo."
		)
		self.manual_center_check.setToolTip(
			"Fija el centro de la cavidad a mano cuando el automático no acierta.\n"
			"Encendido: hacé clic en el centro de la cavidad sobre la imagen del cine "
			"(clic derecho borra el de ese corte). Ese centro manda sobre el automático "
			"y todo lo que viene después (radios, ángulo AHA, fase) se recalcula respecto de él.\n"
			"Reprocesa solo el/los corte(s) con centro fijado; el resto sigue automático."
		)
		self.manual_center_all_check.setToolTip(
			"Si está encendido, un solo clic aplica ese mismo centro a TODOS los cortes.\n"
			"Útil cuando el corazón está bien alineado. Apagado: el centro es por corte."
		)
		self.manual_center_clear_btn.setToolTip("Borra todos los centros manuales y vuelve al centro automático.")
		self.threshold_spin.setToolTip("Porcentaje del máximo usado para separar miocardio del fondo.")
		self.sigma_spin.setToolTip("Suavizado espacial aplicado antes del threshold en segmentación.")
		self.harmonics_spin.setToolTip("Cantidad de armónicos usados para estabilizar la fase.")
		self.phase_threshold_spin.setToolTip("Filtro clínico robusto de amplitud. El QC crudo se calcula aparte con amp 0.10 y no entra al informe.")
		self.cmap_combo.setToolTip("Colormap cíclico para visualizar fase.")
		self.visual_style_combo.setToolTip("Tema visual de los paneles clínicos (curva FEVI, panel funcional gated y bull's eye).")
		self.polar_rotation_spin.setToolTip("Rota el mapa polar de perfusión continua. Ajustalo para alinear ANT/SEP/LAT/INF a tu convención.")
		self.polar_perf_smooth_method_combo.setToolTip("Método de suavizado para el segundo mapa de polar_perfusion_directa y para el cine polar.")
		self.polar_perf_smooth_strength_spin.setToolTip("Intensidad del suavizado polar. 0 = sin suavizado; valores mayores suavizan más.")
		self.polar_cine_speed_spin.setToolTip("Duración por frame del GIF del cine polar (en milisegundos).")
		self.polar_compare_math_combo.setToolTip("Operación matemática opcional entre mapas polares de esfuerzo/reposo en el cine comparativo.")
		self.polar_compare_term_a_combo.setToolTip("Primer término de la operación A op B.")
		self.polar_compare_term_b_combo.setToolTip("Segundo término de la operación A op B.")
		self.export_polar_mp4_check.setToolTip("Además del GIF, intenta exportar un MP4 del cine polar gatillado.")
		self.profile_timing_check.setToolTip("Registra en el log solo etapas que superan 0.5 s.")
		self.realtime_deferred_render_check.setToolTip("Muestra resultados rápidos primero y completa en alta calidad unos instantes después.")
		self.normalize_check.setToolTip("Resta una referencia global de fase para comparar estudios.")
		self.gate_dropout_check.setToolTip(
			"Escala el último gate para que sume las mismas cuentas que el primero (Emory Cardiac Toolbox 4.0, secc. 22.8).\n"
			"Por la variabilidad del R-R, el último gate siempre queda con menos cuentas: eso es artefacto de adquisición,\n"
			"no fisiología, y sesga tanto la FEVI como la fase (PSD/BW salen más altos de lo real).\n"
			"Es una corrección de ganancia global: no deforma el miocardio ni mueve bordes.\n"
			"Dejalo activo salvo que la consola ya lo haya corregido (en ese caso el dropout medido da ~0%)."
		)
		self.gate_dropout_status.setToolTip("Déficit medido en el último gate y factor aplicado. Se actualiza al cargar o reprocesar el estudio.")
		self.global_intestinal_render_check.setToolTip("Si está activo, muestra la atenuación intestinal en las salidas visuales. Fase, métricas y FEVI se calculan sobre el estudio bruto.")
		self.normal_sex_combo.setToolTip("Sexo del paciente: los valores normales de PSD/BW difieren por sexo (Mukherjee 2016).")
		self.normal_protocol_combo.setToolTip("Protocolo del estudio: stress da PSD/BW mayores que rest.")
		self.normal_db_combo.setToolTip("Base normal publicada usada para lectura vs referencia; los límites son software-dependientes.")

		self._sidebar_layout.addWidget(controls_box)

		# Grupo ECG - Contexto electrocardiográfico
		ecg_box = QGroupBox("ECG (contexto clínico)")
		ecg_form = QFormLayout(ecg_box)

		self.ecg_ritmo_combo = QComboBox()
		self.ecg_ritmo_combo.addItems(["Sinusal", "FA", "Marcapasos", "CRT", "Otro"])
		self.ecg_ritmo_combo.setCurrentText("Sinusal")

		self.ecg_fc_spin = QSpinBox()
		self.ecg_fc_spin.setRange(30, 220)
		self.ecg_fc_spin.setValue(70)
		self.ecg_fc_spin.setSuffix(" lpm")

		self.ecg_qrs_spin = QSpinBox()
		self.ecg_qrs_spin.setRange(60, 250)
		self.ecg_qrs_spin.setValue(90)
		self.ecg_qrs_spin.setSuffix(" ms")

		self.ecg_qt_spin = QSpinBox()
		self.ecg_qt_spin.setRange(250, 600)
		self.ecg_qt_spin.setValue(400)
		self.ecg_qt_spin.setSuffix(" ms")

		self.ecg_bri_check = QCheckBox("BRI (bloqueo rama izquierda)")
		self.ecg_bri_check.setChecked(False)

		self.ecg_brd_check = QCheckBox("BRD (bloqueo rama derecha)")
		self.ecg_brd_check.setChecked(False)

		self.ecg_marcapasos_check = QCheckBox("Marcapasos/CRT")
		self.ecg_marcapasos_check.setChecked(False)

		self.ecg_obs_edit = QLineEdit()
		self.ecg_obs_edit.setPlaceholderText("Observaciones ECG...")

		# Carga de archivo ECG
		self.ecg_file_path = ""
		self.ecg_load_btn = QPushButton("Cargar ECG...")
		self.ecg_load_btn.clicked.connect(self._load_ecg_file)
		self.ecg_load_btn.setToolTip("Cargar archivo ECG (PDF, JPG, PNG) para adjuntar al informe.")

		self.ecg_preview_label = QLabel("Sin ECG cargado")
		self.ecg_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.ecg_preview_label.setStyleSheet("color:#666; font-size:9pt; border:1px solid #444; padding:4px;")
		self.ecg_preview_label.setMaximumHeight(60)
		self.ecg_preview_label.setWordWrap(True)

		self.ecg_clear_btn = QPushButton("Limpiar ECG")
		self.ecg_clear_btn.clicked.connect(self._clear_ecg_file)
		self.ecg_clear_btn.setToolTip("Quitar el archivo ECG cargado.")

		ecg_file_row = QHBoxLayout()
		ecg_file_row.addWidget(self.ecg_load_btn)
		ecg_file_row.addWidget(self.ecg_clear_btn)

		ecg_form.addRow("Ritmo", self.ecg_ritmo_combo)
		ecg_form.addRow("FC", self.ecg_fc_spin)
		ecg_form.addRow("QRS", self.ecg_qrs_spin)
		ecg_form.addRow("QT", self.ecg_qt_spin)
		ecg_form.addRow(self.ecg_bri_check)
		ecg_form.addRow(self.ecg_brd_check)
		ecg_form.addRow(self.ecg_marcapasos_check)
		ecg_form.addRow("Observaciones", self.ecg_obs_edit)
		ecg_form.addRow("Archivo ECG", ecg_file_row)
		ecg_form.addRow("", self.ecg_preview_label)

		self.ecg_ritmo_combo.setToolTip("Ritmo cardíaco dominante en el ECG de 12 derivaciones.")
		self.ecg_fc_spin.setToolTip("Frecuencia cardíaca en latidos por minuto.")
		self.ecg_qrs_spin.setToolTip("Duración del QRS en milisegundos. QRS ≥120ms sugiere disincronía eléctrica.")
		self.ecg_qt_spin.setToolTip("Intervalo QT en milisegundos.")
		self.ecg_bri_check.setToolTip("Bloqueo de rama izquierda del haz de His.")
		self.ecg_brd_check.setToolTip("Bloqueo de rama derecha del haz de His.")
		self.ecg_marcapasos_check.setToolTip("Paciente con marcapasos o resincronizador cardíaco (CRT).")

		self._sidebar_layout.addWidget(ecg_box)

		report_cmap_box = QGroupBox("Escalas informe (por imagen)")
		report_cmap_layout = QGridLayout(report_cmap_box)
		report_cmap_layout.setContentsMargins(6, 6, 6, 6)
		report_cmap_layout.setHorizontalSpacing(4)
		report_cmap_layout.setVerticalSpacing(4)
		report_cmap_layout.setColumnStretch(0, 3)
		report_cmap_layout.setColumnStretch(1, 2)
		report_cmap_info = QLabel("Elegí la escala por pestaña: seguí el orden clínico 1-9 para preparar el informe de forma consistente.")
		report_cmap_info.setWordWrap(True)
		report_cmap_info.setStyleSheet("color:#35506a;")
		report_cmap_layout.addWidget(report_cmap_info, 0, 0, 1, 2)

		def _mk_combo(current: str) -> QComboBox:
			cb = QComboBox()
			cb.addItems(self._all_cmaps)
			if current in self._all_cmaps:
				cb.setCurrentText(current)
			return cb

		# Perfusión/intensidad → odyssey_cool; crudo → gris; fase/amplitud/delta conservan su escala.
		self.report_cmap_slices = _mk_combo("odyssey_cool")
		self.report_cmap_axes = _mk_combo("odyssey_cool")
		self.report_cmap_compare = _mk_combo("odyssey_cool")
		self.report_cmap_panel_axes = _mk_combo("odyssey_cool")
		self.report_cmap_phase = _mk_combo("french")
		self.report_cmap_polar_clinico = _mk_combo("french")
		self.report_cmap_amp = _mk_combo("turbo")
		self.report_cmap_bullseye = _mk_combo("odyssey_cool")
		self.report_cmap_polar_perf = _mk_combo("odyssey_cool")
		self.report_cmap_ungated = _mk_combo("odyssey_cool")
		self.report_cmap_cine_crudo = _mk_combo("gray")
		self.report_cmap_histograma = _mk_combo("hot")
		self.report_cmap_polar_combo = _mk_combo("french")
		self.report_cmap_delta_combo = _mk_combo("french")
		self.report_cmap_stress_rest = _mk_combo("odyssey_cool")
		self.report_cmap_polar_cine = _mk_combo("odyssey_cool")

		def _add_cmap_row(row: int, label_text: str, combo: QComboBox, tip_text: str):
			label = QLabel(label_text)
			label.setToolTip(tip_text)
			combo.setToolTip(tip_text)
			report_cmap_layout.addWidget(label, row, 0)
			report_cmap_layout.addWidget(combo, row, 1)

		_add_cmap_row(1, "[1] slices_fase", self.report_cmap_slices, "Afecta la imagen de la pestaña slices_fase (slice/gate, máscara y superposición).")
		_add_cmap_row(2, "[2] comparacion_ejes (SA/HLA/VLA)", self.report_cmap_axes, "Afecta las imágenes de ejes ortogonales SA/HLA/VLA para informe.")
		_add_cmap_row(3, "[3] comparacion_ejes (grilla)", self.report_cmap_compare, "Afecta la grilla multicorte de comparación entre estudios.")
		_add_cmap_row(4, "[4] panel_funcional_gated (ED/ES)", self.report_cmap_panel_axes, "Afecta las imágenes ED/ES del panel funcional gated.")
		_add_cmap_row(5, "[5] fase (overlay/polar)", self.report_cmap_phase, "Afecta mapas de fase (overlay y polar de fase).")
		_add_cmap_row(6, "[6] polar_clinico", self.report_cmap_polar_clinico, "Afecta el panel polar clínico (histograma + bullseye de fase).")
		_add_cmap_row(7, "[7] panel_funcional_gated (amplitud)", self.report_cmap_amp, "Afecta el mapa de amplitud en el panel funcional gated.")
		_add_cmap_row(8, "[8] bullseye_directo", self.report_cmap_bullseye, "Afecta el bullseye directo de perfusión segmentaria AHA.")
		_add_cmap_row(9, "[9] polar_perfusion_directa", self.report_cmap_polar_perf, "Afecta el mapa polar continuo de perfusión (apex-centro, base-borde).")
		_add_cmap_row(10, "[10] ungated (desgatillado)", self.report_cmap_ungated, "Afecta la grilla de cortes desgatillados (UngRaw / perfusión total).")
		_add_cmap_row(11, "[11] cine_crudo (proyecciones)", self.report_cmap_cine_crudo, "Afecta el cine de proyecciones crudas SPECT (gated y UngGat).")
		_add_cmap_row(12, "[12] histograma", self.report_cmap_histograma, "Afecta el histograma de fase (barras y fondo).")
		_add_cmap_row(13, "[13] polar_combo", self.report_cmap_polar_combo, "Afecta el panel combinado polar AHA + clínico.")
		_add_cmap_row(14, "[14] delta_combo", self.report_cmap_delta_combo, "Afecta los mapas delta stress/rest (signed y abs).")
		_add_cmap_row(15, "[15] stress_vs_rest", self.report_cmap_stress_rest, "Afecta el panel resumen de métricas stress vs rest.")
		_add_cmap_row(16, "[16] polar_cine_montaje", self.report_cmap_polar_cine, "Afecta el cine polar gatillado (montaje por gate).")

		# El grid de escalas del informe ya no vive en el sidebar: se aloja en el
		# diálogo de Configuración (declutter). Estos selectores definen SOLO el
		# color de cada imagen del informe PDF. Se mantiene la referencia viva.
		self._report_cmap_box = report_cmap_box
		report_cmap_box.setVisible(False)

		preset_box = QGroupBox("Presets por paciente")
		preset_layout = QVBoxLayout(preset_box)
		preset_layout.setContentsMargins(6, 6, 6, 6)
		preset_layout.setSpacing(4)
		self.preset_patient_edit = QLineEdit()
		self.preset_patient_edit.setPlaceholderText("Paciente (auto si está vacío)")
		self.preset_name_edit = QLineEdit()
		self.preset_name_edit.setPlaceholderText("Nombre del preset (ej: stress_base)")
		self.preset_combo = QComboBox()
		self.preset_combo.setToolTip("Presets guardados para el paciente actual.")
		preset_layout.addWidget(QLabel("Paciente"))
		preset_layout.addWidget(self.preset_patient_edit)
		preset_layout.addWidget(QLabel("Nombre preset"))
		preset_layout.addWidget(self.preset_name_edit)
		preset_layout.addWidget(QLabel("Presets guardados"))
		preset_layout.addWidget(self.preset_combo)

		preset_actions = QHBoxLayout()
		self.save_preset_btn = QPushButton("Guardar")
		self.save_preset_btn.clicked.connect(self.save_current_preset)
		self.load_preset_btn = QPushButton("Cargar")
		self.load_preset_btn.clicked.connect(self.load_selected_preset)
		self.delete_preset_btn = QPushButton("Borrar")
		self.delete_preset_btn.clicked.connect(self.delete_selected_preset)
		preset_actions.addWidget(self.save_preset_btn)
		preset_actions.addWidget(self.load_preset_btn)
		preset_actions.addWidget(self.delete_preset_btn)
		preset_layout.addLayout(preset_actions)
		self._sidebar_layout.addWidget(preset_box)

		self.helper_box = QGroupBox("Ayuda rápida")
		helper_layout = QVBoxLayout(self.helper_box)
		helper_layout.setContentsMargins(6, 6, 6, 6)
		helper_layout.setSpacing(3)
		helper = QLabel(
			"1. Abrí el estudio y procesá.\n"
			"2. Ajustá ROIs o parámetros si hace falta.\n"
			"3. En apex/base, si no se ve cavidad, dejá r_inner='-' (o usá Borrar internos).\n"
			"4. Replicá ROI o procesá de nuevo para ver cambios."
		)
		helper.setWordWrap(True)
		helper.setStyleSheet("color:#35506a; line-height:1.25;")
		helper_layout.addWidget(helper)
		self.audit_help_btn = QPushButton("Ayuda auditoría/validación")
		self.audit_help_btn.clicked.connect(self.show_audit_validation_help)
		self.audit_help_btn.setToolTip("Explica cálculos, supuestos y recomendaciones clínicas de uso para auditoría.")
		helper_layout.addWidget(self.audit_help_btn)
		self.polar_tech_help_btn = QPushButton("Help técnico mapas polares")
		self.polar_tech_help_btn.clicked.connect(self.show_polar_technical_help)
		self.polar_tech_help_btn.setToolTip("Explica para qué sirve cada mapa polar, fórmulas de sincronía, interpretación y rangos orientativos.")
		helper_layout.addWidget(self.polar_tech_help_btn)
		self.crt_plan_help_btn = QPushButton("Plan implementación CRT (prioridades)")
		self.crt_plan_help_btn.clicked.connect(self.show_crt_implementation_plan)
		self.crt_plan_help_btn.setToolTip("Roadmap clínico-técnico priorizado para acelerar entrega y mejorar robustez de interpretación.")
		helper_layout.addWidget(self.crt_plan_help_btn)
		self.docs_portal_btn = QPushButton("Portal docs")
		self.docs_portal_btn.clicked.connect(self.open_docs_portal)
		self.docs_portal_btn.setToolTip("Abre el portal de documentación HTML (índice de guías e instrucciones).")
		helper_layout.addWidget(self.docs_portal_btn)
		self._sidebar_layout.addWidget(self.helper_box)

		button_box = QGroupBox("Acciones")
		button_row = QGridLayout(button_box)
		button_row.setContentsMargins(6, 6, 6, 6)
		button_row.setHorizontalSpacing(4)
		button_row.setVerticalSpacing(4)
		self.restart_btn = QPushButton("RESTART")
		self.restart_btn.setStyleSheet(
			"QPushButton{background:#dc2626;color:white;font-weight:bold;border:1px solid #b91c1c;"
			"border-radius:4px;padding:4px 8px;} "
			"QPushButton:hover{background:#b91c1c;}"
		)
		self.restart_btn.clicked.connect(self.restart_workspace_state)
		self.restart_btn.setToolTip("Limpia el estado en memoria de la sesión para cargar estudios nuevos desde cero.")
		self.process_btn = QPushButton("Procesar")
		self.process_btn.setStyleSheet(
			"QPushButton{background:#16a34a;color:white;font-weight:bold;border:1px solid #15803d;"
			"border-radius:4px;padding:4px 8px;} "
			"QPushButton:hover{background:#15803d;}"
		)
		self.process_btn.clicked.connect(self.process_current)
		self.process_btn.setToolTip("Recalcula segmentación, fase, métricas, polar map y gráficos.")
		self.open_pdf_btn = QPushButton("Abrir PDF")
		self.open_pdf_btn.clicked.connect(self.open_pdf)
		self.open_pdf_btn.setToolTip("Abre el informe clínico PDF generado.")
		self.save_pdf_as_btn = QPushButton("Guardar PDF")
		self.save_pdf_as_btn.clicked.connect(self.save_pdf_as)
		self.save_pdf_as_btn.setToolTip("Guarda una copia del informe PDF en la ubicación que elijas.")
		# Botón PDF con menú (Abrir / Guardar como)
		self.pdf_menu = QMenu(self)
		self.pdf_menu.addAction("Abrir PDF", self.open_pdf)
		self.pdf_menu.addAction("Guardar PDF como...", self.save_pdf_as)
		self.pdf_btn = QToolButton()
		self.pdf_btn.setText("PDF ▾")
		self.pdf_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
		self.pdf_btn.setMenu(self.pdf_menu)
		self.pdf_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
		self.pdf_btn.setToolTip("Abrir o guardar el informe PDF.")
		self.html_menu = QMenu(self)
		self.html_menu.addAction("Abrir HTML", self.open_html_report)
		self.html_menu.addAction("Guardar HTML como...", self.save_html_as)
		self.html_menu.addSeparator()
		editor_action = self.html_menu.addAction("Editor de informe...", self.open_report_editor)
		editor_action.setEnabled(False)
		editor_action.setToolTip("Se habilita después de reorientar el estudio.")
		self.html_menu.addSeparator()
		self.html_menu.addAction("Verificar integridad HTML...", self.verify_html_integrity)
		self.html_menu.addAction("Limpiar hashes antiguos...", self.cleanup_hash_store)
		self.html_btn = QToolButton()
		self.html_btn.setText("HTML ▾")
		self.html_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
		self.html_btn.setMenu(self.html_menu)
		self.html_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
		self.html_btn.setToolTip("Abrir o guardar el informe HTML autocontenido.")
		self.compare_stress_rest_btn = QPushButton("Comparar Rest/Stress")
		self.compare_stress_rest_btn.clicked.connect(self.load_compare_study)
		self.compare_stress_rest_btn.setToolTip(
			"Carga un segundo estudio (ej: REST) y compara la disincronía (PSD, BW, Kurtosis, Entropy) "
			"contra el estudio actual. Útil para detectar stunning isquémico post-stress (Camilletti 2015)."
		)
		self.apply_roi_all_btn = QPushButton("Replicar ROI")
		self.apply_roi_all_btn.clicked.connect(self.apply_current_roi_to_all_slices)
		self.apply_roi_all_btn.setToolTip("Copia el ROI del slice actual a todos los slices del volumen.")
		self.load_one_or_two_btn = QPushButton("Cargar 1/2 estudios")
		self.load_one_or_two_btn.clicked.connect(self.load_one_or_two_studies)
		self.load_one_or_two_btn.setToolTip("Carga una o dos fases (stress/rest). Opción 'Carpeta inteligente': marcás una carpeta y lee/carga todo (Esfuerzo y Reposo con sus EM, ATT, CT y Scatter).")
		self.preparacion_btn = QPushButton("Preparar / Reconstruir…")
		self.preparacion_btn.clicked.connect(self.open_preparacion_window)
		self.preparacion_btn.setToolTip(
			"Abre la ventana de preparación dual (Esfuerzo | Reposo) con MIP rotatorio en vivo. "
			"Corrección de movimiento, reconstrucción y reorientación por etapa, independientes. "
			"Es no modal: se puede dejar abierta al costado."
		)
		self.ui_config_btn = QPushButton("Configuración")
		self.ui_config_btn.clicked.connect(self.open_ui_preferences_dialog)
		self.ui_config_btn.setToolTip(
			"Panel de Configuración: tema visual (Clásico/Moderno), helpers, tooltips "
			"y modo compacto de botones. Se irán agregando más opciones."
		)
		self.export_ungated_btn = QPushButton("Ungated DCM")
		self.export_ungated_btn.clicked.connect(self._export_ungated_dicom)
		self.export_ungated_btn.setToolTip("Exporta el desgatillado (suma de gates = perfusión total) como DICOM NM no-gated para compartir o releer.")
		self.ectb_window_btn = QPushButton("Cuantificación ECTb")
		self.ectb_window_btn.clicked.connect(self.open_ectb_window)
		self.ectb_window_btn.setToolTip(
			"Abre la ventana de cuantificación del Emory Cardiac Toolbox: FEVI por máximo de cuentas, "
			"volúmenes, masa miocárdica y engrosamiento.\n"
			"Es el método que usa el informe por defecto. Desde ahí también se puede volver al método "
			"anterior (umbral endocárdico), que queda disponible para comparar.\n"
			"Los parámetros recalculan en vivo y los dos métodos se muestran lado a lado.\n"
			"Es no modal: se puede dejar abierta al costado mientras se trabaja acá."
		)
		self.gqc_window_btn = QPushButton("Control de calidad del gating")
		self.gqc_window_btn.clicked.connect(self.open_gqc_window)
		self.gqc_window_btn.setToolTip(
			"Abre el panel GQC: cuentas de cada proyección con una curva por gate.\n"
			"Es la forma clásica de detectar dropout del último gate, arritmia o rechazo de latidos y "
			"movimiento del paciente durante la rotación.\n"
			"Con proyecciones crudas el análisis es completo; con un estudio ya reconstruido cae a un "
			"sustituto por cortes, que no permite ver el movimiento.\n"
			"Es no modal y se recalcula en vivo."
		)
		self.asynchrony_review_btn = QPushButton("Vista asincronía")
		self.asynchrony_review_btn.clicked.connect(self.open_asynchrony_review_window)
		self.asynchrony_review_btn.setToolTip(
			"Abre una vista separada para inspección visual de imagen, ROIs y bordes en modo asincrónico.\n"
			"No reemplaza el flujo actual: conserva la ventana principal y añade una vista de revisión independiente."
		)
		self.lv_3d_btn = QPushButton("VI 3D")
		self.lv_3d_btn.clicked.connect(self.open_lv_3d_window)
		self.lv_3d_btn.setToolTip(
			"Abre el panel 3D del ventrículo: miocardio sólido (isosuperficie del SA reorientado) "
			"a la izquierda y la malla alambre del VI (ECTb) a la derecha, con ED fija y el gate "
			"actual animado en cine sincronizado. Rotación/zoom con el mouse. Exporta GIF/AVI.\n"
			"Requiere estudio gated procesado con FEVI (ECTb) calculado."
		)
		button_row.addWidget(self.process_btn, 0, 0)
		button_row.addWidget(self.pdf_btn, 0, 1)
		button_row.addWidget(self.html_btn, 0, 2)
		button_row.addWidget(self.restart_btn, 0, 3)
		button_row.setColumnStretch(0, 3)
		button_row.setColumnStretch(1, 3)
		button_row.setColumnStretch(2, 3)
		button_row.setColumnStretch(3, 1)
		compare_load_row = QHBoxLayout()
		compare_load_row.setContentsMargins(0, 0, 0, 0)
		compare_load_row.setSpacing(4)
		compare_load_row.addWidget(self.compare_stress_rest_btn, 1)
		compare_load_row.addWidget(self.load_one_or_two_btn, 1)
		button_row.addLayout(compare_load_row, 1, 0, 1, 4)
		button_row.addWidget(self.preparacion_btn, 2, 0, 1, 4)
		ungated_config_row = QHBoxLayout()
		ungated_config_row.setContentsMargins(0, 0, 0, 0)
		ungated_config_row.setSpacing(4)
		ungated_config_row.addWidget(self.export_ungated_btn, 1)
		ungated_config_row.addWidget(self.ui_config_btn, 1)
		button_row.addLayout(ungated_config_row, 3, 0, 1, 4)
		button_row.addWidget(self.ectb_window_btn, 4, 0, 1, 4)
		button_row.addWidget(self.gqc_window_btn, 5, 0, 1, 4)
		button_row.addWidget(self.asynchrony_review_btn, 6, 0, 1, 4)
		button_row.addWidget(self.lv_3d_btn, 7, 0, 1, 4)
		self.amyloid_btn = QPushButton("Amyloidosis Planar")
		self.amyloid_btn.clicked.connect(self.open_amyloid_window)
		self.amyloid_btn.setToolTip(
			"Módulo planar de amiloidosis cardíaca: ROIs, HMR, Perugini, washout y reporte."
		)
		button_row.addWidget(self.amyloid_btn, 8, 0, 1, 4)
		self.amyloid_spect_btn = QPushButton("Amyloidosis SPECT/CT")
		self.amyloid_spect_btn.clicked.connect(self.open_amyloid_spect_window)
		self.amyloid_spect_btn.setToolTip(
			"Abre directamente el flujo AMYLO SPECT / SPECT-CT: reconstrucción, cortes, CT, registro, "
			"sustracción ósea visual y exportación."
		)
		button_row.addWidget(self.amyloid_spect_btn, 9, 0, 1, 4)

		# Ubicar Acciones justo debajo de la versión y la barra de progreso.
		insert_at = self._sidebar_layout.indexOf(self._progress_bar) + 1
		self._sidebar_layout.insertWidget(insert_at, button_box)

		roi_box = QGroupBox("ROI manual por slice")
		roi_layout = QVBoxLayout(roi_box)
		roi_layout.setContentsMargins(6, 6, 6, 6)
		roi_layout.setSpacing(4)
		roi_note = QLabel("Editá ROI en el visor o pegá líneas slice,cy,cx,r_inner,r_outer.")
		roi_note.setWordWrap(True)
		roi_note.setStyleSheet("color:#555;")
		roi_layout.addWidget(roi_note)
		roi_layout.addWidget(self.manual_rois)
		roi_actions_top = QHBoxLayout()
		roi_actions_top.addWidget(self.apply_roi_all_btn)
		roi_actions_top.addStretch(1)
		roi_layout.addLayout(roi_actions_top)

		roi_adjust_note = QLabel("Ajuste Auto ROI desde slice actual: propagá centro/radios al volumen.")
		roi_adjust_note.setWordWrap(True)
		roi_adjust_note.setStyleSheet("color:#555;")
		roi_layout.addWidget(roi_adjust_note)

		roi_delta_grid = QGridLayout()
		roi_delta_grid.setHorizontalSpacing(4)
		roi_delta_grid.setVerticalSpacing(4)
		self.auto_center_gain_slider = QSlider(Qt.Orientation.Horizontal)
		self.auto_center_gain_slider.setRange(0, 200)
		self.auto_center_gain_slider.setValue(100)
		self.auto_center_gain_slider.setToolTip("Multiplica el corrimiento del centro detectado en el slice de referencia. 100% = aplicar el mismo delta.")
		self.auto_center_gain_label = QLabel("100%")
		self.auto_inner_delta_slider = QSlider(Qt.Orientation.Horizontal)
		self.auto_inner_delta_slider.setRange(-50, 50)
		self.auto_inner_delta_slider.setValue(0)
		self.auto_inner_delta_slider.setToolTip("Delta fino extra para el radio interno en pixels. Positivo agranda; negativo achica.")
		self.auto_inner_delta_label = QLabel("+0.0 px")
		self.auto_outer_delta_slider = QSlider(Qt.Orientation.Horizontal)
		self.auto_outer_delta_slider.setRange(-50, 50)
		self.auto_outer_delta_slider.setValue(0)
		self.auto_outer_delta_slider.setToolTip("Delta fino extra para el radio externo en pixels. Positivo agranda; negativo achica.")
		self.auto_outer_delta_label = QLabel("+0.0 px")
		for slider in (self.auto_center_gain_slider, self.auto_inner_delta_slider, self.auto_outer_delta_slider):
			slider.valueChanged.connect(self._update_roi_adjust_labels)
		roi_delta_grid.addWidget(QLabel("Centro"), 0, 0)
		roi_delta_grid.addWidget(self.auto_center_gain_slider, 0, 1)
		roi_delta_grid.addWidget(self.auto_center_gain_label, 0, 2)
		roi_delta_grid.addWidget(QLabel("Interno"), 1, 0)
		roi_delta_grid.addWidget(self.auto_inner_delta_slider, 1, 1)
		roi_delta_grid.addWidget(self.auto_inner_delta_label, 1, 2)
		roi_delta_grid.addWidget(QLabel("Externo"), 2, 0)
		roi_delta_grid.addWidget(self.auto_outer_delta_slider, 2, 1)
		roi_delta_grid.addWidget(self.auto_outer_delta_label, 2, 2)
		self.reset_roi_deltas_btn = QPushButton("Reset")
		self.reset_roi_deltas_btn.clicked.connect(self.reset_roi_adjust_deltas)
		roi_delta_grid.addWidget(self.reset_roi_deltas_btn, 3, 1)
		roi_layout.addLayout(roi_delta_grid)
		self._update_roi_adjust_labels()

		roi_range_row = QHBoxLayout()
		roi_range_row.addWidget(QLabel("Rango +/-"))
		self.auto_adjust_range_spin = QSpinBox()
		self.auto_adjust_range_spin.setRange(-1, 99)
		self.auto_adjust_range_spin.setValue(-1)
		self.auto_adjust_range_spin.setSpecialValueText("todos")
		self.auto_adjust_range_spin.setToolTip("Cuántos slices a cada lado del slice de referencia se ajustan. 'todos' aplica al volumen completo.")
		roi_range_row.addWidget(self.auto_adjust_range_spin)
		roi_range_row.addStretch(1)
		roi_layout.addLayout(roi_range_row)

		roi_adjust_actions = QGridLayout()
		roi_adjust_actions.setHorizontalSpacing(4)
		roi_adjust_actions.setVerticalSpacing(4)
		self.adjust_auto_center_btn = QPushButton("Centro")
		self.adjust_auto_center_btn.clicked.connect(self.adjust_auto_center_all_slices)
		self.adjust_auto_inner_btn = QPushButton("Interno")
		self.adjust_auto_inner_btn.clicked.connect(self.adjust_auto_inner_all_slices)
		self.adjust_auto_outer_btn = QPushButton("Externo")
		self.adjust_auto_outer_btn.clicked.connect(self.adjust_auto_outer_all_slices)
		self.adjust_auto_full_btn = QPushButton("Completo")
		self.adjust_auto_full_btn.clicked.connect(self.adjust_auto_full_all_slices)
		roi_adjust_actions.addWidget(self.adjust_auto_center_btn, 0, 0)
		roi_adjust_actions.addWidget(self.adjust_auto_inner_btn, 0, 1)
		roi_adjust_actions.addWidget(self.adjust_auto_outer_btn, 1, 0)
		roi_adjust_actions.addWidget(self.adjust_auto_full_btn, 1, 1)
		roi_layout.addLayout(roi_adjust_actions)

		roi_actions_mid = QGridLayout()
		roi_actions_mid.setHorizontalSpacing(4)
		roi_actions_mid.setVerticalSpacing(4)
		self.clear_current_roi_btn = QPushButton("Borrar ROI")
		self.clear_current_roi_btn.clicked.connect(self.clear_current_roi)
		self.clear_all_rois_btn = QPushButton("Borrar todos")
		self.clear_all_rois_btn.clicked.connect(self.clear_all_rois)
		self.clear_outer_rois_btn = QPushButton("Borrar externos")
		self.clear_outer_rois_btn.clicked.connect(self.clear_outer_rois)
		roi_actions_mid.addWidget(self.clear_current_roi_btn, 0, 0)
		roi_actions_mid.addWidget(self.clear_all_rois_btn, 0, 1)
		roi_actions_mid.addWidget(self.clear_outer_rois_btn, 1, 0)

		roi_actions_bottom = QGridLayout()
		roi_actions_bottom.setHorizontalSpacing(4)
		roi_actions_bottom.setVerticalSpacing(4)
		self.clear_inner_rois_btn = QPushButton("Borrar internos")
		self.clear_inner_rois_btn.clicked.connect(self.clear_inner_rois)
		self.clear_centers_btn = QPushButton("Borrar centros")
		self.clear_centers_btn.clicked.connect(self.clear_centers)
		self.reset_file_btn = QPushButton("Reset archivo")
		self.reset_file_btn.clicked.connect(self.reset_current_file)
		roi_actions_bottom.addWidget(self.clear_inner_rois_btn, 0, 0)
		roi_actions_bottom.addWidget(self.clear_centers_btn, 0, 1)
		roi_actions_bottom.addWidget(self.reset_file_btn, 1, 0, 1, 2)

		roi_layout.addLayout(roi_actions_mid)
		roi_layout.addLayout(roi_actions_bottom)
		self._sidebar_layout.addWidget(roi_box)

		compare_box = QGroupBox("Comparación ejes")
		compare_layout = QVBoxLayout(compare_box)
		compare_layout.setContentsMargins(6, 6, 6, 6)
		compare_layout.setSpacing(4)
		compare_note = QLabel("Alinea gate/corte para comparar ejes entre estudios.")
		compare_note.setWordWrap(True)
		compare_note.setStyleSheet("color:#555;")
		compare_layout.addWidget(compare_note)
		cine_source_row = QHBoxLayout()
		cine_source_row.addWidget(QLabel("Cine/ROI"))
		self.cine_source_combo = QComboBox()
		self.cine_source_combo.currentIndexChanged.connect(self._on_cine_source_changed)
		cine_source_row.addWidget(self.cine_source_combo, 1)
		self.cine_primary_btn = QToolButton()
		self.cine_primary_btn.setText("Esf.")
		self.cine_primary_btn.clicked.connect(lambda: self._apply_cine_source("primary"))
		self.cine_compare_btn = QToolButton()
		self.cine_compare_btn.setText("Reposo")
		self.cine_compare_btn.clicked.connect(lambda: self._apply_cine_source("compare"))
		cine_source_row.addWidget(self.cine_primary_btn)
		cine_source_row.addWidget(self.cine_compare_btn)
		compare_layout.addLayout(cine_source_row)
		compare_gate_row = QHBoxLayout()
		compare_gate_row.addWidget(QLabel("Gate"))
		self.compare_gate_spin = QSpinBox()
		self.compare_gate_spin.setRange(1, 1)
		self.compare_gate_spin.setValue(1)
		self.compare_gate_spin.setToolTip("Gate usado en la lámina de comparación de ejes.")
		self.compare_gate_spin.valueChanged.connect(self._schedule_compare_axes_refresh)
		compare_gate_row.addWidget(self.compare_gate_spin)
		self.use_cine_compare_btn = QPushButton("Usar cine")
		self.use_cine_compare_btn.clicked.connect(self.use_cine_position_for_comparison)
		compare_gate_row.addWidget(self.use_cine_compare_btn)
		compare_layout.addLayout(compare_gate_row)
		compare_slice_row = QGridLayout()
		self.compare_slice_slider = QSlider(Qt.Orientation.Horizontal)
		self.compare_slice_slider.setRange(0, 100)
		self.compare_slice_slider.setValue(50)
		self.compare_slice_slider.setToolTip("Posición anatómica relativa del corte comparativo. 50% = plano medio.")
		self.compare_slice_slider.sliderPressed.connect(self._on_compare_controls_drag_started)
		self.compare_slice_slider.sliderReleased.connect(self._on_compare_controls_drag_ended)
		self.compare_slice_label = QLabel("50%")
		self.compare_slice_slider.valueChanged.connect(self._update_compare_slice_label)
		self.compare_slice_slider.valueChanged.connect(self._schedule_compare_axes_refresh)
		compare_slice_row.addWidget(QLabel("Corte"), 0, 0)
		compare_slice_row.addWidget(self.compare_slice_slider, 0, 1)
		compare_slice_row.addWidget(self.compare_slice_label, 0, 2)
		compare_layout.addLayout(compare_slice_row)
		compare_offset_row = QGridLayout()
		self.compare_slice_offset_sa_spin = QSpinBox()
		self.compare_slice_offset_sa_spin.setRange(-12, 12)
		self.compare_slice_offset_sa_spin.setValue(0)
		self.compare_slice_offset_sa_spin.setToolTip("Desfase manual para SA cuando no coincide el corte entre stress y rest.")
		self.compare_slice_offset_sa_spin.valueChanged.connect(self._schedule_compare_axes_refresh)
		self.compare_slice_offset_hla_spin = QSpinBox()
		self.compare_slice_offset_hla_spin.setRange(-12, 12)
		self.compare_slice_offset_hla_spin.setValue(0)
		self.compare_slice_offset_hla_spin.setToolTip("Desfase manual para HLA cuando no coincide el corte entre stress y rest.")
		self.compare_slice_offset_hla_spin.valueChanged.connect(self._schedule_compare_axes_refresh)
		self.compare_slice_offset_vla_spin = QSpinBox()
		self.compare_slice_offset_vla_spin.setRange(-12, 12)
		self.compare_slice_offset_vla_spin.setValue(0)
		self.compare_slice_offset_vla_spin.setToolTip("Desfase manual para VLA cuando no coincide el corte entre stress y rest.")
		self.compare_slice_offset_vla_spin.valueChanged.connect(self._schedule_compare_axes_refresh)
		compare_offset_row.addWidget(QLabel("SA"), 0, 0)
		compare_offset_row.addWidget(self.compare_slice_offset_sa_spin, 0, 1)
		compare_offset_row.addWidget(QLabel("HLA"), 1, 0)
		compare_offset_row.addWidget(self.compare_slice_offset_hla_spin, 1, 1)
		compare_offset_row.addWidget(QLabel("VLA"), 2, 0)
		compare_offset_row.addWidget(self.compare_slice_offset_vla_spin, 2, 1)
		compare_layout.addLayout(compare_offset_row)
		self.compare_axes_cmap_combo = QComboBox()
		self.compare_axes_cmap_combo.addItems(self._all_cmaps)
		self.compare_axes_cmap_combo.setCurrentText("odyssey_cool")
		self.compare_axes_cmap_combo.setToolTip("Escala de colores específica de la pestaña comparacion_ejes.")
		self.compare_axes_cmap_combo.currentTextChanged.connect(self._schedule_compare_axes_refresh)
		compare_cmap_row = QHBoxLayout()
		compare_cmap_row.addWidget(QLabel("Colormap"))
		compare_cmap_row.addWidget(self.compare_axes_cmap_combo, 1)
		compare_layout.addLayout(compare_cmap_row)
		compare_window_row = QGridLayout()
		self.compare_window_high_slider = QSlider(Qt.Orientation.Horizontal)
		self.compare_window_high_slider.setRange(1, 100)
		self.compare_window_high_slider.setValue(100)
		self.compare_window_high_slider.sliderPressed.connect(self._on_compare_controls_drag_started)
		self.compare_window_high_slider.sliderReleased.connect(self._on_compare_controls_drag_ended)
		self.compare_window_high_slider.valueChanged.connect(self._on_compare_window_high_change)
		self.compare_window_high_label = QLabel("100%")
		self.compare_window_low_slider = QSlider(Qt.Orientation.Horizontal)
		self.compare_window_low_slider.setRange(0, 99)
		self.compare_window_low_slider.setValue(0)
		self.compare_window_low_slider.sliderPressed.connect(self._on_compare_controls_drag_started)
		self.compare_window_low_slider.sliderReleased.connect(self._on_compare_controls_drag_ended)
		self.compare_window_low_slider.valueChanged.connect(self._on_compare_window_low_change)
		self.compare_window_low_label = QLabel("0%")
		compare_window_row.addWidget(QLabel("Top"), 0, 0)
		compare_window_row.addWidget(self.compare_window_high_slider, 0, 1)
		compare_window_row.addWidget(self.compare_window_high_label, 0, 2)
		compare_window_row.addWidget(QLabel("Base"), 1, 0)
		compare_window_row.addWidget(self.compare_window_low_slider, 1, 1)
		compare_window_row.addWidget(self.compare_window_low_label, 1, 2)
		compare_layout.addLayout(compare_window_row)
		compare_zoom_row = QGridLayout()
		self.compare_axes_zoom_slider = QSlider(Qt.Orientation.Horizontal)
		self.compare_axes_zoom_slider.setRange(100, 300)
		self.compare_axes_zoom_slider.setValue(100)
		self.compare_axes_zoom_slider.setToolTip("Zoom global en comparacion_ejes: agranda todos los cortes SA/HLA/VLA al mismo tiempo.")
		self.compare_axes_zoom_slider.sliderPressed.connect(self._on_compare_controls_drag_started)
		self.compare_axes_zoom_slider.sliderReleased.connect(self._on_compare_controls_drag_ended)
		self.compare_axes_zoom_slider.valueChanged.connect(self._on_compare_axes_zoom_changed)
		self.compare_axes_zoom_label = QLabel("100%")
		compare_zoom_row.addWidget(QLabel("Zoom"), 0, 0)
		compare_zoom_row.addWidget(self.compare_axes_zoom_slider, 0, 1)
		compare_zoom_row.addWidget(self.compare_axes_zoom_label, 0, 2)
		compare_layout.addLayout(compare_zoom_row)
		self.compare_mask_check = QCheckBox("Máscara")
		self.compare_mask_check.setChecked(True)
		self.compare_mask_check.toggled.connect(self._on_compare_mask_toggled)
		compare_layout.addWidget(self.compare_mask_check)
		self.compare_axes_intestinal_mask_check = QCheckBox("ROI intestino")
		self.compare_axes_intestinal_mask_check.setChecked(True)
		self.compare_axes_intestinal_mask_check.setToolTip("Si está activo, aplica la atenuación intestinal visual en comparacion_ejes cuando el toggle visual global está ON.")
		self.compare_axes_intestinal_mask_check.toggled.connect(self._schedule_compare_axes_refresh)
		compare_layout.addWidget(self.compare_axes_intestinal_mask_check)
		compare_quick_row = QHBoxLayout()
		self.compare_axes_preset_clinical_btn = QPushButton("Preset ejes")
		self.compare_axes_preset_clinical_btn.setToolTip("Aplica valores recomendados para lectura clínica rápida de comparacion_ejes.")
		self.compare_axes_preset_clinical_btn.clicked.connect(self._apply_compare_axes_clinical_quick_preset)
		compare_quick_row.addWidget(self.compare_axes_preset_clinical_btn)
		compare_layout.addLayout(compare_quick_row)
		self.compare_fast_drag_check = QCheckBox("Rápido al arrastrar")
		self.compare_fast_drag_check.setChecked(True)
		compare_layout.addWidget(self.compare_fast_drag_check)
		compare_cine_row = QGridLayout()
		self.compare_axes_cine_check = QCheckBox("Cine")
		self.compare_axes_cine_check.setChecked(False)
		self.compare_axes_cine_check.toggled.connect(self._on_compare_axes_cine_toggled)
		self.compare_axes_cine_speed_spin = QSpinBox()
		self.compare_axes_cine_speed_spin.setRange(40, 1000)
		self.compare_axes_cine_speed_spin.setSingleStep(10)
		self.compare_axes_cine_speed_spin.setValue(180)
		self.compare_axes_cine_speed_spin.setSuffix(" ms")
		self.compare_axes_cine_speed_spin.setToolTip("Duración por frame del cine de comparativa.")
		self.compare_axes_cine_speed_spin.valueChanged.connect(self._on_compare_axes_cine_speed_changed)
		self.compare_axes_cine_toggle_btn = QToolButton()
		self.compare_axes_cine_toggle_btn.setText("Play")
		self.compare_axes_cine_toggle_btn.clicked.connect(self._toggle_compare_axes_preview)
		self.compare_axes_cine_restart_btn = QToolButton()
		self.compare_axes_cine_restart_btn.setText("Reset")
		self.compare_axes_cine_restart_btn.clicked.connect(self._restart_compare_axes_preview)
		compare_cine_row.addWidget(self.compare_axes_cine_check, 0, 0, 1, 2)
		compare_cine_row.addWidget(QLabel("Velocidad"), 1, 0)
		compare_cine_row.addWidget(self.compare_axes_cine_speed_spin, 1, 1)
		compare_cine_row.addWidget(self.compare_axes_cine_toggle_btn, 2, 0)
		compare_cine_row.addWidget(self.compare_axes_cine_restart_btn, 2, 1)
		self.compare_axes_export_frames_btn = QToolButton()
		self.compare_axes_export_frames_btn.setText("Export frames")
		self.compare_axes_export_frames_btn.clicked.connect(self.export_compare_axes_frames_debug)
		compare_cine_row.addWidget(self.compare_axes_export_frames_btn, 3, 0, 1, 2)
		compare_layout.addLayout(compare_cine_row)
		self.refresh_compare_btn = QPushButton("Actualizar ejes")
		self.refresh_compare_btn.clicked.connect(self._refresh_compare_axes_panel_now)
		compare_layout.addWidget(self.refresh_compare_btn)
		self._update_compare_slice_label()
		self._update_compare_window_labels()
		self._update_compare_axes_zoom_label()
		self._refresh_cine_source_selector()
		# 'Comparación ejes' (lámina primitiva) fue reemplazada por el Montaje
		# clínico: colormap, window level y export migraron a esa barra. Se
		# mantiene construida (handlers acoplados) pero fuera del sidebar.
		self._compare_axes_box_hidden = compare_box
		compare_box.setVisible(False)

		self.summary_clinical = QTextEdit()
		self.summary_clinical.setReadOnly(True)
		self.summary_clinical.setMinimumHeight(120)
		self.summary_clinical.setPlaceholderText("Aquí aparecerá el resumen clínico cuando proceses un estudio.")
		self.summary_clinical.setToolTip("Resumen clínico: lectura vs DB normal, métricas robustas, volúmenes, FEVI y territorios.")

		self.summary_executive = QTextEdit()
		self.summary_executive.setReadOnly(True)
		self.summary_executive.setMinimumHeight(120)
		self.summary_executive.setPlaceholderText("Aquí aparecerá el resumen ejecutivo (síntesis del hallazgo) cuando proceses un estudio.")
		self.summary_executive.setToolTip("Resumen ejecutivo: síntesis del hallazgo en lenguaje natural (fase, territorios, función). Mismo texto que encabeza el informe PDF.")

		self.summary_technical = QTextEdit()
		self.summary_technical.setReadOnly(True)
		self.summary_technical.setMinimumHeight(120)
		self.summary_technical.setPlaceholderText("Aquí aparecerá el detalle técnico y de procesamiento.")
		self.summary_technical.setToolTip("Detalle técnico: metadata DICOM, parámetros, métricas y notas de QC.")

		self.summary_tabs = QTabWidget()
		self.summary_tabs.addTab(self.summary_executive, "Resumen")
		self.summary_tabs.addTab(self.summary_clinical, "Clínico")
		self.summary_tabs.addTab(self.summary_technical, "Técnico")

		report_box = QGroupBox("Resumen")
		report_layout = QVBoxLayout(report_box)
		report_layout.setContentsMargins(6, 6, 6, 6)
		report_layout.setSpacing(4)
		report_layout.addWidget(self.summary_tabs)
		self._sidebar_layout.addWidget(report_box)

		self.log_box = QTextEdit()
		self.log_box.setReadOnly(True)
		self.log_box.setMinimumHeight(90)
		self.log_box.setPlaceholderText("Eventos y advertencias aparecerán aquí.")
		self.log_box.setToolTip("Mensajes del loader, segmentación y reprocesado.")

		log_box = QGroupBox("Log")
		log_layout = QVBoxLayout(log_box)
		log_layout.setContentsMargins(6, 6, 6, 6)
		log_layout.setSpacing(4)
		log_layout.addWidget(self.log_box)
		self._sidebar_layout.addWidget(log_box)
		self._sidebar_layout.addStretch(1)

		# Cada caja de opciones del sidebar pasa a ser una sección colapsable:
		# el título se convierte en botón. Se hace acá, al final, para no tener
		# que tocar la construcción de cada caja.
		self._install_collapsible_sidebar_sections()

		right = QWidget()
		right_layout = QVBoxLayout(right)
		right_layout.setContentsMargins(0, 0, 0, 0)
		right_layout.setSpacing(0)
		right_layout.addWidget(self._build_pipeline_step_bar())
		right_splitter = QSplitter(Qt.Orientation.Vertical)
		right_splitter.setChildrenCollapsible(False)
		right_splitter.setOpaqueResize(True)
		right_splitter.setHandleWidth(10)

		self.tabs = QTabWidget()
		self.preview_labels: dict[str, QLabel] = {}
		self._tab_widgets: dict[str, QWidget] = {}
		self._tab_titles: dict[str, str] = {}
		self._tab_tooltips: dict[str, str] = {}
		preview_titles = {
			"slices_fase": "slices_fase",
			"polar_combo": "polar",
			"delta_combo": "delta_polar",
			"histograma": "histograma",
			"polar_perfusion_directa": "polar_perfusion_directa",
			"comparacion_ejes": "Montaje clínico",
			"comparacion_stress_rest": "stress_vs_rest",
			"panel_funcional_gated": "Panel funcional gated",
			"bullseye_directo": "bullseye_directo",
			"guia_fase_vi": "Guía para fase VI",
			"ungated": "QC",
			"cine_crudo": "cine_crudo",
		}
		preview_help_texts = {
			"slices_fase": "Vista de referencia del slice/gate medio con máscara y fase superpuesta. Útil para control de calidad de segmentación.",
			"polar_combo": "Panel combinado: mapa polar AHA + panel clínico con histograma/PSD/PHB. Mantiene valores y lectura rápida en una sola pestaña.",
			"histograma": "Histograma de fase global para estimar dispersión temporal (PSD, BW, entropy).",
			"delta_combo": "Panel combinado de los dos mapas delta (con signo y absoluto) para comparar stress/rest en una sola pestaña.",
			"polar_perfusion_directa": "Mapa polar de perfusión (crudo + suavizado en paralelo) con vista Cine gatillado integrada: alterná entre 'Perfusión' (estática) y 'Cine' (evolución por gate, incluye operación esfuerzo/reposo).",
			"comparacion_ejes": "Comparación multicorte por ejes entre estudios para detectar diferencias regionales en el mismo gate.",
			"comparacion_stress_rest": "Resumen de métricas de disincronía stress vs rest (PSD/BW/Kurtosis/Entropy) e interpretación clínica.",
			"panel_funcional_gated": "Panel funcional integrado (ED/ES, fase, amplitud y curvas) para lectura clínica rápida.",
			"bullseye_directo": "Bull's-eye de perfusión segmentaria AHA (17): resumen compacto de intensidad regional.",
			"guia_fase_vi": "Guía para fase VI: bull's-eye doble (fase + perfusión/viabilidad) y tabla segmentaria AHA-17 que cruza cuándo se contrae cada segmento con cuánto capta. Si hay estudio de comparación, muestra reposo y esfuerzo con Δfase en una sola imagen.",
			"ungated": "Desgatillado (UngRaw): suma de todos los gates = perfusión total con máxima estadística. Base para cortes anatómicos y comparación contra RECON del fabricante.",
			"cine_crudo": "Cine de proyecciones crudas SPECT: revisá el movimiento del paciente entre ángulos antes de reconstruir. Selector gated/UngGat, play/pause, velocidad y frame-by-frame.",
		}
		for name in [
			"slices_fase",
			"polar_combo",
			"delta_combo",
			"histograma",
			"polar_perfusion_directa",
			"comparacion_ejes",
			"comparacion_stress_rest",
			"panel_funcional_gated",
			"bullseye_directo",
			"guia_fase_vi",
			"ungated",
			"cine_crudo",
		]:
			tab = QWidget()
			tab_layout = QVBoxLayout(tab)

			toolbar = QHBoxLayout()
			zoom_out = QToolButton()
			zoom_out.setText("-")
			zoom_out.clicked.connect(lambda _=False, n=name: self._zoom_preview(n, -0.10))
			zoom_in = QToolButton()
			zoom_in.setText("+")
			zoom_in.clicked.connect(lambda _=False, n=name: self._zoom_preview(n, +0.10))
			zoom_reset = QToolButton()
			zoom_reset.setText("100%")
			zoom_reset.clicked.connect(lambda _=False, n=name: self._set_preview_zoom(n, 1.0))
			zoom_label = QLabel(f"{int(self._default_preview_zoom(name) * 100)}%")
			zoom_label.setStyleSheet("color:#444;")
			self.preview_zoom_labels[name] = zoom_label
			toolbar.addWidget(QLabel("Zoom"))
			toolbar.addWidget(zoom_out)
			toolbar.addWidget(zoom_in)
			toolbar.addWidget(zoom_reset)
			toolbar.addWidget(zoom_label)
			if name == "polar_perfusion_directa":
				perf_view_btn = QToolButton()
				perf_view_btn.setText("Perfusión")
				perf_view_btn.setCheckable(True)
				perf_view_btn.setChecked(self.polar_view_mode == "perfusion")
				perf_view_btn.setToolTip("Vista estática: crudo + suavizado en paralelo")
				perf_view_btn.clicked.connect(lambda _=False: self._set_polar_view_mode("perfusion"))
				self.polar_perf_view_perf_btn = perf_view_btn
				cine_view_btn = QToolButton()
				cine_view_btn.setText("Cine")
				cine_view_btn.setCheckable(True)
				cine_view_btn.setChecked(self.polar_view_mode == "cine")
				cine_view_btn.setToolTip("Vista cine gatillado: evolución por gate + operación esfuerzo/reposo")
				cine_view_btn.clicked.connect(lambda _=False: self._set_polar_view_mode("cine"))
				self.polar_perf_view_cine_btn = cine_view_btn
				toolbar.addWidget(QLabel("Vista"))
				toolbar.addWidget(perf_view_btn)
				toolbar.addWidget(cine_view_btn)
				play_btn = QToolButton()
				play_btn.setText("Play")
				play_btn.clicked.connect(self._toggle_polar_cine_preview)
				self.polar_cine_toggle_btn = play_btn
				restart_btn = QToolButton()
				restart_btn.setText("Restart")
				restart_btn.clicked.connect(self._restart_polar_cine_preview)
				toolbar.addWidget(play_btn)
				toolbar.addWidget(restart_btn)
			if name == "cine_crudo":
				# --- Fila 1 (toolbar principal): zoom + reproducción + navegación + fuente/modo ---
				self.cine_crudo_play_btn = QToolButton()
				self.cine_crudo_play_btn.setText("Play")
				self.cine_crudo_play_btn.clicked.connect(self._toggle_cine_crudo)
				toolbar.addWidget(self.cine_crudo_play_btn)
				prev_btn = QToolButton()
				prev_btn.setText("|<")
				prev_btn.setToolTip("Frame anterior")
				prev_btn.clicked.connect(lambda _=False: self._step_cine_crudo(-1))
				toolbar.addWidget(prev_btn)
				next_btn = QToolButton()
				next_btn.setText(">|")
				next_btn.setToolTip("Frame siguiente")
				next_btn.clicked.connect(lambda _=False: self._step_cine_crudo(1))
				toolbar.addWidget(next_btn)
				toolbar.addWidget(QLabel("Vel."))
				self.cine_crudo_speed_spin = QSpinBox()
				self.cine_crudo_speed_spin.setRange(40, 1000)
				self.cine_crudo_speed_spin.setSingleStep(20)
				self.cine_crudo_speed_spin.setValue(120)
				self.cine_crudo_speed_spin.setSuffix(" ms")
				self.cine_crudo_speed_spin.valueChanged.connect(self._on_cine_crudo_speed_changed)
				toolbar.addWidget(self.cine_crudo_speed_spin)
				toolbar.addWidget(QLabel("Fuente"))
				self.cine_crudo_source_combo = QComboBox()
				self.cine_crudo_source_combo.addItems(["UngGat", "Gated"])
				self.cine_crudo_source_combo.setCurrentText("UngGat")
				self.cine_crudo_source_combo.setFixedWidth(80)
				self.cine_crudo_source_combo.currentTextChanged.connect(self._on_cine_crudo_source_changed)
				toolbar.addWidget(self.cine_crudo_source_combo)
				toolbar.addWidget(QLabel("Modo"))
				self.cine_crudo_mode_combo = QComboBox()
				self.cine_crudo_mode_combo.addItems(["Continuo", "Rebote"])
				self.cine_crudo_mode_combo.setCurrentText("Rebote")
				self.cine_crudo_mode_combo.setFixedWidth(100)
				self.cine_crudo_mode_combo.setToolTip("Continuo: loop 1→N→1. Rebote: 1→N→1→N (ping-pong).")
				toolbar.addWidget(self.cine_crudo_mode_combo)
				toolbar.addWidget(QLabel("Etapa"))
				self.cine_crudo_stage_combo = QComboBox()
				self.cine_crudo_stage_combo.addItems(["Esfuerzo", "Reposo", "Ambas"])
				self.cine_crudo_stage_combo.setCurrentText("Esfuerzo")
				self.cine_crudo_stage_combo.setFixedWidth(100)
				self.cine_crudo_stage_combo.setToolTip("Con dos fases crudas cargadas, elegí qué etapa procesan las herramientas (corrección, offset, etc.): solo Esfuerzo, solo Reposo, o Ambas. También: CLICK sobre una imagen selecciona esa etapa; CTRL+CLICK selecciona ambas.")
				self.cine_crudo_stage_combo.currentTextChanged.connect(self._on_cine_crudo_stage_combo_changed)
				toolbar.addWidget(self.cine_crudo_stage_combo)
				self.cine_crudo_frame_label = QLabel("--/--")
				self.cine_crudo_frame_label.setStyleSheet("color:#444;")
				self.cine_crudo_frame_label.setFixedWidth(180)
				toolbar.addWidget(self.cine_crudo_frame_label)

				# --- Fila 2 (toolbar motion correction): método + eje + threshold mejorado + acciones ---
				toolbar2 = QHBoxLayout()
				toolbar2.addWidget(QLabel("Método"))
				self.cine_crudo_method_combo = QComboBox()
				self.cine_crudo_method_combo.addItems(["Auto", "Sinusoide", "XCorr", "GammaSync", "Stasis", "Hopkins", "Odyssey", "COM", "Threshold"])
				self.cine_crudo_method_combo.setCurrentText("Sinusoide")
				self.cine_crudo_method_combo.setToolTip("Sinusoide: ajusta la sinusoide de rotación esperada y corrige solo el residuo (movimiento real). Correcto geométricamente para SPECT (default). XCorr: correlación de fase 2D contra una proyección de referencia; usa toda la estructura de la imagen, no depende de threshold ni de aislar el corazón (alternativa cuando el hígado se pega al corazón). GammaSync: selección de órgano automática/click. Stasis: referencia estática (moda, Xeleris). Hopkins: frame más estable (Xeleris). Odyssey: re-proyección iterativa. COM: centro de masa. Threshold: bounding box.")
				toolbar2.addWidget(self.cine_crudo_method_combo)
				toolbar2.addWidget(QLabel("Eje"))
				self.cine_crudo_axis_combo = QComboBox()
				self.cine_crudo_axis_combo.addItems(["Y", "X", "XY"])
				self.cine_crudo_axis_combo.setCurrentText("Y")
				self.cine_crudo_axis_combo.setToolTip("Eje de corrección: solo Y (default clínico), solo X, o ambos.")
				toolbar2.addWidget(self.cine_crudo_axis_combo)

				# Threshold mejorado: botón - , slider amplio, botón + , spin numérico sincronizado
				toolbar2.addWidget(QLabel("Thr"))
				thr_minus = QToolButton()
				thr_minus.setText("−")
				thr_minus.setToolTip("Bajar threshold de a 0.01")
				thr_minus.clicked.connect(lambda _=False: self._step_cine_crudo_threshold(-1))
				toolbar2.addWidget(thr_minus)
				self.cine_crudo_threshold_slider = QSlider(Qt.Orientation.Horizontal)
				self.cine_crudo_threshold_slider.setRange(1, 100)
				self.cine_crudo_threshold_slider.setValue(20)
				self.cine_crudo_threshold_slider.setMinimumWidth(220)
				self.cine_crudo_threshold_slider.setToolTip("Threshold para aislar el corazón (Select Object). Mová el slider y mirá la máscara en vivo.")
				self.cine_crudo_threshold_slider.valueChanged.connect(self._on_cine_crudo_threshold_changed)
				toolbar2.addWidget(self.cine_crudo_threshold_slider, 1)
				thr_plus = QToolButton()
				thr_plus.setText("+")
				thr_plus.setToolTip("Subir threshold de a 0.01")
				thr_plus.clicked.connect(lambda _=False: self._step_cine_crudo_threshold(1))
				toolbar2.addWidget(thr_plus)
				self.cine_crudo_threshold_spin = QDoubleSpinBox()
				self.cine_crudo_threshold_spin.setRange(0.01, 1.00)
				self.cine_crudo_threshold_spin.setSingleStep(0.01)
				self.cine_crudo_threshold_spin.setDecimals(2)
				self.cine_crudo_threshold_spin.setValue(0.20)
				self.cine_crudo_threshold_spin.setMaximumWidth(64)
				self.cine_crudo_threshold_spin.setToolTip("Valor numérico del threshold (sincronizado con el slider).")
				self.cine_crudo_threshold_spin.valueChanged.connect(self._on_cine_crudo_threshold_spin_changed)
				toolbar2.addWidget(self.cine_crudo_threshold_spin)
				toolbar2.addStretch(1)
				self.cine_crudo_grid_btn = QToolButton()
				self.cine_crudo_grid_btn.setText("Grilla pick")
				self.cine_crudo_grid_btn.setToolTip("Grilla de cortes transaxiales con máscara para discriminar corazón de hígado antes del pick (como Odyssey).")
				self.cine_crudo_grid_btn.clicked.connect(self._show_cine_crudo_transaxial_grid)
				toolbar2.addWidget(self.cine_crudo_grid_btn)
				self.cine_crudo_synthetic_btn = QToolButton()
				self.cine_crudo_synthetic_btn.setText("Sintético")
				self.cine_crudo_synthetic_btn.setToolTip("Carga un crudo sintético: corazón con desplazamiento X por rotación, salto Y respiratorio e hígado/intestino inferior intenso. Útil para probar Banda Y y Atenuar hígado.")
				self.cine_crudo_synthetic_btn.clicked.connect(self._load_cine_crudo_synthetic)
				toolbar2.addWidget(self.cine_crudo_synthetic_btn)
				self.cine_crudo_correct_btn = QToolButton()
				self.cine_crudo_correct_btn.setText("Corregir")
				self.cine_crudo_correct_btn.setToolTip("Aplica motion correction con método/eje/threshold seleccionados.")
				self.cine_crudo_correct_btn.clicked.connect(self._apply_cine_crudo_motion_correction)
				self.cine_crudo_correct_btn.setMinimumHeight(90)
				# Comparar / Diferencia / Sinograma: controles de VISUALIZACIÓN de la
				# corrección, junto al botón Corregir (no en ajuste manual).
				self.cine_crudo_compare_check = QCheckBox("Comparar")
				self.cine_crudo_compare_check.setToolTip("Muestra original y corregido en paralelo (original | corregido).")
				self.cine_crudo_compare_check.setEnabled(False)
				self.cine_crudo_compare_check.toggled.connect(self._refresh_cine_crudo_view)
				toolbar2.addWidget(self.cine_crudo_compare_check)
				self.cine_crudo_diff_check = QCheckBox("Diferencia\n(resta scatter)")
				self.cine_crudo_diff_check.setToolTip(
					"Muestra la DIFERENCIA corregido − original (mismo truco que el "
					"preview de scatter EM−SC). Sirve para ver QUÉ movió la corrección "
					"de movimiento. Se activa tras aplicar la corrección.")
				self.cine_crudo_diff_check.setEnabled(False)
				self.cine_crudo_diff_check.toggled.connect(self._refresh_cine_crudo_view)
				toolbar2.addWidget(self.cine_crudo_diff_check)
				self.cine_crudo_sino_check = QCheckBox("Sinograma")
				self.cine_crudo_sino_check.setToolTip("Muestra a la derecha el sinograma; con Comparar activo agrega original/corregido como Odyssey.")
				self.cine_crudo_sino_check.toggled.connect(self._refresh_cine_crudo_view)
				toolbar2.addWidget(self.cine_crudo_sino_check)
				self.cine_crudo_sino_axis_combo = QComboBox()
				self.cine_crudo_sino_axis_combo.addItems(["Sinograma Y", "Sinograma X"])
				self.cine_crudo_sino_axis_combo.setCurrentIndex(1)  # Sinograma X por defecto
				self.cine_crudo_sino_axis_combo.setMaximumWidth(118)
				self.cine_crudo_sino_axis_combo.setToolTip("Y: perfil horizontal por ángulo. X: perfil vertical por ángulo.")
				self.cine_crudo_sino_axis_combo.currentTextChanged.connect(self._refresh_cine_crudo_view)
				toolbar2.addWidget(self.cine_crudo_sino_axis_combo)

				# Aplicar / Rechazar: una vez que la corrección (p.ej. Sinusoide + ajuste
				# fino manual) quedó como el usuario quiere, decide si el pipeline sigue
				# con el crudo corregido (Aplicar) o vuelve al crudo original (Rechazar).
				self.cine_crudo_accept_btn = QToolButton()
				self.cine_crudo_accept_btn.setText("Aplicar")
				self.cine_crudo_accept_btn.setToolTip("Confirma la corrección actual: la reconstrucción y todo el procesamiento usarán las proyecciones corregidas.")
				self.cine_crudo_accept_btn.clicked.connect(self._accept_cine_crudo_motion_correction)
				self.cine_crudo_reject_btn = QToolButton()
				self.cine_crudo_reject_btn.setText("Rechazar")
				self.cine_crudo_reject_btn.setToolTip("Descarta la corrección: se vuelve al crudo original y el procesamiento seguirá sin corrección de movimiento.")
				self.cine_crudo_reject_btn.clicked.connect(self._reject_cine_crudo_motion_correction)
				corr_side = QWidget()
				corr_side_layout = QVBoxLayout(corr_side)
				corr_side_layout.setContentsMargins(0, 0, 0, 0)
				corr_side_layout.setSpacing(3)
				corr_side_layout.addWidget(self.cine_crudo_correct_btn, 1)
				corr_accept_row = QHBoxLayout()
				corr_accept_row.setContentsMargins(0, 0, 0, 0)
				corr_accept_row.setSpacing(3)
				corr_accept_row.addWidget(self.cine_crudo_accept_btn)
				corr_accept_row.addWidget(self.cine_crudo_reject_btn)
				corr_side_layout.addLayout(corr_accept_row)
				self.cine_crudo_correct_side = corr_side

				# --- Fila 3 (selección de órgano): máscara + pick corazón + ROI ---
				toolbar3 = QHBoxLayout()
				self.cine_crudo_mask_check = QCheckBox("Máscara")
				self.cine_crudo_mask_check.setToolTip("Superpone la máscara del threshold sobre la proyección actual en tiempo real.")
				self.cine_crudo_mask_check.toggled.connect(self._refresh_cine_crudo_view)
				toolbar3.addWidget(self.cine_crudo_mask_check)
				self.cine_crudo_seed_btn = QToolButton()
				self.cine_crudo_seed_btn.setText("Elegir corazón")
				self.cine_crudo_seed_btn.setCheckable(True)
				self.cine_crudo_seed_btn.setToolTip("Activá y hacé CLICK en el corazón sobre la imagen. Con 'Radio ROI' > 0 el tracking sigue SOLO una ventana alrededor del corazón (umbral local), así el hígado —aunque tenga más cuentas y sea más grande— queda fuera y no engancha el tracking.")
				self.cine_crudo_seed_btn.toggled.connect(self._on_cine_crudo_seed_mode_toggled)
				toolbar3.addWidget(self.cine_crudo_seed_btn)
				toolbar3.addWidget(QLabel("Radio ROI"))
				self.cine_crudo_roi_spin = QSpinBox()
				self.cine_crudo_roi_spin.setRange(0, 40)
				self.cine_crudo_roi_spin.setValue(6)
				self.cine_crudo_roi_spin.setSuffix(" px")
				self.cine_crudo_roi_spin.setMaximumWidth(72)
				self.cine_crudo_roi_spin.setToolTip("Radio de la ventana de tracking alrededor del corazón (tras el click). 0 = desactivado (usa componente global). 10–16 px suele aislar el corazón del hígado en matriz 64².")
				self.cine_crudo_roi_spin.valueChanged.connect(self._refresh_cine_crudo_view)
				toolbar3.addWidget(self.cine_crudo_roi_spin)
				toolbar3.addWidget(QLabel("Tipo"))
				self.cine_crudo_roi_mode_combo = QComboBox()
				self.cine_crudo_roi_mode_combo.addItems(["Caja", "Banda Y"])
				self.cine_crudo_roi_mode_combo.setMaximumWidth(92)
				self.cine_crudo_roi_mode_combo.setToolTip("Caja: ventana local alrededor del click. Banda Y: dos líneas horizontales upper/lower; usa toda la franja cardíaca y penaliza focos alejados en X para reducir hígado/intestino.")
				self.cine_crudo_roi_mode_combo.currentTextChanged.connect(self._refresh_cine_crudo_view)
				toolbar3.addWidget(self.cine_crudo_roi_mode_combo)
				toolbar3.addWidget(QLabel("Color"))
				self.cine_crudo_marker_color_combo = QComboBox()
				self.cine_crudo_marker_color_combo.addItems(["Arena", "Cian", "Blanco", "Negro", "Magenta", "Verde"])
				self.cine_crudo_marker_color_combo.setMaximumWidth(86)
				self.cine_crudo_marker_color_combo.setToolTip("Color de cruz, markers y línea de frame del sinograma. Cambialo según el colormap/contraste.")
				self.cine_crudo_marker_color_combo.currentTextChanged.connect(self._refresh_cine_crudo_view)
				toolbar3.addWidget(self.cine_crudo_marker_color_combo)
				self.cine_crudo_liver_suppress_check = QCheckBox("Atenuar hígado")
				self.cine_crudo_liver_suppress_check.setToolTip("Atenúa focos hepato-intestinales SOLO para el tracking, guiado por los markers Y. No modifica las cuentas usadas para corregir ni el DICOM exportado.")
				toolbar3.addWidget(self.cine_crudo_liver_suppress_check)
				self.cine_crudo_liver_suppress_spin = QSpinBox()
				self.cine_crudo_liver_suppress_spin.setRange(0, 95)
				self.cine_crudo_liver_suppress_spin.setValue(60)
				self.cine_crudo_liver_suppress_spin.setSuffix(" %")
				self.cine_crudo_liver_suppress_spin.setMaximumWidth(72)
				self.cine_crudo_liver_suppress_spin.setToolTip("Porcentaje de atenuación usado solo en la imagen de tracking. Sugerido: 50–70%.")
				toolbar3.addWidget(self.cine_crudo_liver_suppress_spin)
				toolbar3.addStretch(1)

				# --- Fila 4 (corrección manual): flechas tipo teclado + paso + reset + ajuste fino + referencia + comparar ---
				toolbar4 = QHBoxLayout()
				toolbar4.addWidget(QLabel("Manual"))
				# Flechas en layout tipo teclado:
				#       ↑
				#   ←   ↓   →
				self.cine_crudo_up_btn = QToolButton()
				self.cine_crudo_up_btn.setText("↑")
				self.cine_crudo_up_btn.setToolTip("Subir el frame ACTUAL (shift Y −paso). Corrección manual frame a frame en vivo.")
				self.cine_crudo_up_btn.clicked.connect(lambda _=False: self._nudge_cine_crudo_frame(-self._cine_crudo_nudge_step(), 0.0))
				self.cine_crudo_down_btn = QToolButton()
				self.cine_crudo_down_btn.setText("↓")
				self.cine_crudo_down_btn.setToolTip("Bajar el frame ACTUAL (shift Y +paso).")
				self.cine_crudo_down_btn.clicked.connect(lambda _=False: self._nudge_cine_crudo_frame(self._cine_crudo_nudge_step(), 0.0))
				self.cine_crudo_left_btn = QToolButton()
				self.cine_crudo_left_btn.setText("←")
				self.cine_crudo_left_btn.setToolTip("Mover el frame ACTUAL a la izquierda (shift X −paso).")
				self.cine_crudo_left_btn.clicked.connect(lambda _=False: self._nudge_cine_crudo_frame(0.0, -self._cine_crudo_nudge_step()))
				self.cine_crudo_right_btn = QToolButton()
				self.cine_crudo_right_btn.setText("→")
				self.cine_crudo_right_btn.setToolTip("Mover el frame ACTUAL a la derecha (shift X +paso).")
				self.cine_crudo_right_btn.clicked.connect(lambda _=False: self._nudge_cine_crudo_frame(0.0, self._cine_crudo_nudge_step()))
				arrow_grid = QGridLayout()
				arrow_grid.setContentsMargins(0, 0, 0, 0)
				arrow_grid.setSpacing(2)
				arrow_grid.addWidget(self.cine_crudo_up_btn, 0, 1)
				arrow_grid.addWidget(self.cine_crudo_left_btn, 1, 0)
				arrow_grid.addWidget(self.cine_crudo_down_btn, 1, 1)
				arrow_grid.addWidget(self.cine_crudo_right_btn, 1, 2)
				toolbar4.addLayout(arrow_grid)
				toolbar4.addWidget(QLabel("Paso"))
				self.cine_crudo_nudge_step_spin = QDoubleSpinBox()
				self.cine_crudo_nudge_step_spin.setRange(0.10, 5.0)
				self.cine_crudo_nudge_step_spin.setSingleStep(0.10)
				self.cine_crudo_nudge_step_spin.setDecimals(2)
				self.cine_crudo_nudge_step_spin.setValue(0.50)
				self.cine_crudo_nudge_step_spin.setMaximumWidth(64)
				self.cine_crudo_nudge_step_spin.setToolTip("Tamaño del paso (px) de las flechas de corrección manual.")
				toolbar4.addWidget(self.cine_crudo_nudge_step_spin)
				self.cine_crudo_reset_manual_btn = QToolButton()
				self.cine_crudo_reset_manual_btn.setText("Reset frame")
				self.cine_crudo_reset_manual_btn.setToolTip("Pone el shift del frame actual en 0 (deshace la corrección manual de ese frame).")
				self.cine_crudo_reset_manual_btn.clicked.connect(self._reset_cine_crudo_frame_shift)
				toolbar4.addWidget(self.cine_crudo_reset_manual_btn)
				self.cine_crudo_fine_btn = QToolButton()
				self.cine_crudo_fine_btn.setText("Ajuste fino")
				self.cine_crudo_fine_btn.setToolTip("Edita manualmente shifts Y/X del frame actual para ajuste fino.")
				self.cine_crudo_fine_btn.clicked.connect(self._open_cine_crudo_fine_adjust)
				toolbar4.addWidget(self.cine_crudo_fine_btn)
				self.cine_crudo_set_ref_btn = QToolButton()
				self.cine_crudo_set_ref_btn.setText("Usar frame ref")
				self.cine_crudo_set_ref_btn.setToolTip("Fija el frame actual como referencia (shift=0) para la corrección.")
				self.cine_crudo_set_ref_btn.clicked.connect(self._set_cine_crudo_reference_frame)
				toolbar4.addWidget(self.cine_crudo_set_ref_btn)
				toolbar4.addStretch(1)

				# --- Fila 5 (offset global + curvas): offset X/Y + curvas de shift ---
				toolbar5 = QHBoxLayout()
				toolbar5.addWidget(QLabel("OffY"))
				self.cine_crudo_offset_y_spin = QDoubleSpinBox()
				self.cine_crudo_offset_y_spin.setRange(-10.0, 10.0)
				self.cine_crudo_offset_y_spin.setDecimals(2)
				self.cine_crudo_offset_y_spin.setSingleStep(0.25)
				self.cine_crudo_offset_y_spin.setValue(0.0)
				self.cine_crudo_offset_y_spin.setMaximumWidth(70)
				self.cine_crudo_offset_y_spin.setToolTip("Offset manual global Y (px) para ajuste visual de la corrección.")
				toolbar5.addWidget(self.cine_crudo_offset_y_spin)
				toolbar5.addWidget(QLabel("OffX"))
				self.cine_crudo_offset_x_spin = QDoubleSpinBox()
				self.cine_crudo_offset_x_spin.setRange(-10.0, 10.0)
				self.cine_crudo_offset_x_spin.setDecimals(2)
				self.cine_crudo_offset_x_spin.setSingleStep(0.25)
				self.cine_crudo_offset_x_spin.setValue(0.0)
				self.cine_crudo_offset_x_spin.setMaximumWidth(70)
				self.cine_crudo_offset_x_spin.setToolTip("Offset manual global X (px) para ajuste visual de la corrección.")
				toolbar5.addWidget(self.cine_crudo_offset_x_spin)
				self.cine_crudo_apply_offset_btn = QToolButton()
				self.cine_crudo_apply_offset_btn.setText("Aplicar offset")
				self.cine_crudo_apply_offset_btn.setToolTip("Aplica offset global X/Y a todos los frames corregidos.")
				self.cine_crudo_apply_offset_btn.clicked.connect(self._apply_cine_crudo_manual_offset)
				toolbar5.addWidget(self.cine_crudo_apply_offset_btn)
				self.cine_crudo_shift_plot_btn = QToolButton()
				self.cine_crudo_shift_plot_btn.setText("Curvas shift")
				self.cine_crudo_shift_plot_btn.setToolTip("Muestra curvas X/Y de shift por frame (estilo Xeleris).")
				self.cine_crudo_shift_plot_btn.clicked.connect(self._show_cine_crudo_shift_curves)
				toolbar5.addWidget(self.cine_crudo_shift_plot_btn)
				toolbar5.addStretch(1)

				# --- Exportar/Importar/Visual/DICOM: sección IO separada dentro de Corrección de movimiento ---
				toolbar_export = QHBoxLayout()
				self.cine_crudo_compare_line_check = QCheckBox("Línea ref")
				self.cine_crudo_compare_line_check.setToolTip("Muestra una línea horizontal arrastrable para comparar si el salto quedó alineado entre original/corregido.")
				self.cine_crudo_compare_line_check.toggled.connect(self._refresh_cine_crudo_view)
				toolbar_export.addWidget(self.cine_crudo_compare_line_check)
				self.cine_crudo_export_btn = QToolButton()
				self.cine_crudo_export_btn.setText("Exportar corrección")
				self.cine_crudo_export_btn.setToolTip("Exporta shifts Y/X por frame (CSV) + proyecciones corregidas (.npz) para comparar y calibrar métodos.")
				self.cine_crudo_export_btn.clicked.connect(self._export_cine_crudo_correction)
				toolbar_export.addWidget(self.cine_crudo_export_btn)
				self.cine_crudo_import_btn = QToolButton()
				self.cine_crudo_import_btn.setText("Importar corrección")
				self.cine_crudo_import_btn.setToolTip("Vuelve a cargar una corrección guardada (CSV o NPZ): aplica los shifts Y/X por frame al estudio actual. Podés seguir ajustando con las flechas, Comparar o Grabar DICOM.")
				self.cine_crudo_import_btn.clicked.connect(self._import_cine_crudo_correction)
				toolbar_export.addWidget(self.cine_crudo_import_btn)
				self.cine_crudo_save_visual_btn = QToolButton()
				self.cine_crudo_save_visual_btn.setText("Guardar visual")
				self.cine_crudo_save_visual_btn.setToolTip("Guarda configuración visual de motion correction: color, Banda Y, línea ref, threshold, método/eje y sinograma.")
				self.cine_crudo_save_visual_btn.clicked.connect(self._save_cine_crudo_visual_config)
				toolbar_export.addWidget(self.cine_crudo_save_visual_btn)
				self.cine_crudo_load_visual_btn = QToolButton()
				self.cine_crudo_load_visual_btn.setText("Cargar visual")
				self.cine_crudo_load_visual_btn.setToolTip("Carga una configuración visual guardada para repetir la misma lectura/overlay.")
				self.cine_crudo_load_visual_btn.clicked.connect(self._load_cine_crudo_visual_config)
				toolbar_export.addWidget(self.cine_crudo_load_visual_btn)
				self.cine_crudo_save_dcm_btn = QToolButton()
				self.cine_crudo_save_dcm_btn.setText("Grabar DICOM")
				self.cine_crudo_save_dcm_btn.setToolTip("Graba las proyecciones corregidas como un DICOM GATED TOMO nuevo (misma estructura y geometría que el original, re-cargable por SINCRO o Xeleris).")
				self.cine_crudo_save_dcm_btn.clicked.connect(self._save_cine_crudo_corrected_dicom)
				toolbar_export.addWidget(self.cine_crudo_save_dcm_btn)

				# Botón "Ajuste manual" en toolbar_export (después de Grabar DICOM):
				# agrega un submenú (FloatingToolbar) con toolbar4 + toolbar5.
				self._cine_crudo_ajuste_btn = self._build_toolbar_group_menu(
					"Ajuste manual ▾", [toolbar4, toolbar5],
					key="cine_crudo_ajuste_manual_export",
					tooltip="Nudge manual, comparación visual, offsets y curvas de shift.",
				)
				toolbar_export.addWidget(self._cine_crudo_ajuste_btn)
				toolbar_export.addStretch(1)

				# --- Fila 6 (reconstrucción raw): separada en 3 filas para
				# legibilidad y mejor adaptación en barra flotante.
				toolbar6_r1 = QHBoxLayout()
				toolbar6_r2 = QHBoxLayout()
				toolbar6_r3 = QHBoxLayout()
				toolbar6_r1.addWidget(QLabel("Recon Ung"))
				self.cine_crudo_recon_method_combo = QComboBox()
				self.cine_crudo_recon_method_combo.addItems(["FBP", "MLEM", "OSEM", "OSEM-Adj"])
				self.cine_crudo_recon_method_combo.setCurrentText("FBP")
				self.cine_crudo_recon_method_combo.setMaximumWidth(90)
				self.cine_crudo_recon_method_combo.setToolTip("Método de reconstrucción. OSEM-Adj: proyector ray-driven adyunto (sin shift, más lento). MLEM/OSEM son CPU de referencia.")
				toolbar6_r1.addWidget(self.cine_crudo_recon_method_combo)
				toolbar6_r1.addWidget(QLabel("Ung"))
				self.cine_crudo_ung_filter_combo = QComboBox()
				self.cine_crudo_ung_filter_combo.addItems(["none", "lowpass", "butterworth", "wiener"])
				self.cine_crudo_ung_filter_combo.setCurrentText("butterworth")
				self.cine_crudo_ung_filter_combo.setMaximumWidth(104)
				self.cine_crudo_ung_filter_combo.setToolTip("Filtro rama UngGat/perfusión. FBP: pre-filtro del sinograma (calco Xeleris). OSEM/MLEM: el Butterworth se aplica POST-reconstrucción (suavizado 3D).")
				toolbar6_r1.addWidget(self.cine_crudo_ung_filter_combo)
				self.cine_crudo_ung_cutoff_spin = QDoubleSpinBox()
				self.cine_crudo_ung_cutoff_spin.setRange(0.01, 1.00)
				self.cine_crudo_ung_cutoff_spin.setSingleStep(0.01)
				self.cine_crudo_ung_cutoff_spin.setDecimals(2)
				self.cine_crudo_ung_cutoff_spin.setValue(0.52)
				self.cine_crudo_ung_cutoff_spin.setMaximumWidth(60)
				self.cine_crudo_ung_cutoff_spin.setToolTip("Cutoff normalizado (fracción de Nyquist) del filtro UngGat. Default cardíaco (matriz 64² · órbita 180°) calcado de Xeleris ECToolbox: 0.52. Con OSEM/MLEM el Butterworth se aplica post-recon (no es pre-filtro FBP).")
				toolbar6_r1.addWidget(self.cine_crudo_ung_cutoff_spin)
				self.cine_crudo_ung_order_spin = QSpinBox()
				self.cine_crudo_ung_order_spin.setRange(1, 20)
				self.cine_crudo_ung_order_spin.setValue(5)
				self.cine_crudo_ung_order_spin.setMaximumWidth(50)
				self.cine_crudo_ung_order_spin.setToolTip("Orden del filtro UngGat. Default cardíaco calcado de Xeleris ECToolbox: 5.")
				toolbar6_r1.addWidget(self.cine_crudo_ung_order_spin)
				toolbar6_r1.addWidget(QLabel("Gated"))
				self.cine_crudo_gated_method_combo = QComboBox()
				self.cine_crudo_gated_method_combo.addItems(["FBP", "MLEM", "OSEM", "OSEM-Adj"])
				self.cine_crudo_gated_method_combo.setCurrentText("FBP")
				self.cine_crudo_gated_method_combo.setMaximumWidth(90)
				self.cine_crudo_gated_method_combo.setToolTip("Método de reconstrucción de la rama gated. OSEM-Adj: proyector ray-driven adyunto (sin shift, más lento).")
				toolbar6_r1.addWidget(self.cine_crudo_gated_method_combo)
				self.cine_crudo_gated_filter_combo = QComboBox()
				self.cine_crudo_gated_filter_combo.addItems(["none", "lowpass", "butterworth", "wiener"])
				self.cine_crudo_gated_filter_combo.setCurrentText("butterworth")
				self.cine_crudo_gated_filter_combo.setMaximumWidth(104)
				self.cine_crudo_gated_filter_combo.setToolTip("Filtro rama gated. FBP: pre-filtro del sinograma (calco Xeleris). OSEM/MLEM: el Butterworth se aplica POST-reconstrucción (suavizado 3D).")
				toolbar6_r1.addWidget(self.cine_crudo_gated_filter_combo)
				self.cine_crudo_gated_cutoff_spin = QDoubleSpinBox()
				self.cine_crudo_gated_cutoff_spin.setRange(0.01, 1.00)
				self.cine_crudo_gated_cutoff_spin.setSingleStep(0.01)
				self.cine_crudo_gated_cutoff_spin.setDecimals(2)
				self.cine_crudo_gated_cutoff_spin.setValue(0.40)
				self.cine_crudo_gated_cutoff_spin.setMaximumWidth(60)
				self.cine_crudo_gated_cutoff_spin.setToolTip("Cutoff normalizado (fracción de Nyquist) del filtro gated. Default cardíaco (matriz 64² · órbita 180°) calcado de Xeleris ECToolbox: 0.40. Con OSEM/MLEM el Butterworth se aplica post-recon (no es pre-filtro FBP).")
				toolbar6_r1.addWidget(self.cine_crudo_gated_cutoff_spin)
				self.cine_crudo_gated_order_spin = QSpinBox()
				self.cine_crudo_gated_order_spin.setRange(1, 20)
				self.cine_crudo_gated_order_spin.setValue(10)
				self.cine_crudo_gated_order_spin.setMaximumWidth(50)
				self.cine_crudo_gated_order_spin.setToolTip("Orden del filtro gated. Default cardíaco calcado de Xeleris ECToolbox: 10.")
				toolbar6_r1.addWidget(self.cine_crudo_gated_order_spin)
				toolbar6_r1.addStretch(1)

				# Auto-recompute por rama DESHABILITADO (2026-08-14): al cambiar método
				# (FBP/OSEM/MLEM) o filtros, la imagen NO se actualiza en tiempo real
				# (tarda mucho, sobre todo OSEM). Se actualiza recién al tocar
				# "Recon raw" o "Reconstruir selección". Si se quiere reactivar,
				# descomentar las conexiones de abajo.
				# self.cine_crudo_ung_filter_combo.currentIndexChanged.connect(lambda *_: self._schedule_recon_branch_recompute("ungated"))
				# self.cine_crudo_ung_cutoff_spin.valueChanged.connect(lambda *_: self._schedule_recon_branch_recompute("ungated"))
				# self.cine_crudo_ung_order_spin.valueChanged.connect(lambda *_: self._schedule_recon_branch_recompute("ungated"))
				# self.cine_crudo_recon_method_combo.currentIndexChanged.connect(lambda *_: self._schedule_recon_branch_recompute("ungated"))
				# self.cine_crudo_gated_filter_combo.currentIndexChanged.connect(lambda *_: self._schedule_recon_branch_recompute("gated"))
				# self.cine_crudo_gated_cutoff_spin.valueChanged.connect(lambda *_: self._schedule_recon_branch_recompute("gated"))
				# self.cine_crudo_gated_order_spin.valueChanged.connect(lambda *_: self._schedule_recon_branch_recompute("gated"))
				# self.cine_crudo_gated_method_combo.currentIndexChanged.connect(lambda *_: self._schedule_recon_branch_recompute("gated"))

				# Iter/Sub: gestionados por el diálogo 'Iteraciones ⚙' (spins ocultos:
				# siguen siendo el default global que el diálogo pisa por estudio).
				_lbl_iter = QLabel("Iter")
				_lbl_iter.setVisible(False)
				toolbar6_r2.addWidget(_lbl_iter)
				self.cine_crudo_iter_spin = QSpinBox()
				self.cine_crudo_iter_spin.setRange(1, 30)
				self.cine_crudo_iter_spin.setValue(8)
				self.cine_crudo_iter_spin.setMaximumWidth(50)
				self.cine_crudo_iter_spin.setVisible(False)
				toolbar6_r2.addWidget(self.cine_crudo_iter_spin)
				_lbl_sub = QLabel("Sub")
				_lbl_sub.setVisible(False)
				toolbar6_r2.addWidget(_lbl_sub)
				self.cine_crudo_osem_subsets_spin = QSpinBox()
				self.cine_crudo_osem_subsets_spin.setRange(1, 16)
				self.cine_crudo_osem_subsets_spin.setValue(4)
				self.cine_crudo_osem_subsets_spin.setMaximumWidth(50)
				self.cine_crudo_osem_subsets_spin.setVisible(False)
				toolbar6_r2.addWidget(self.cine_crudo_osem_subsets_spin)
				self.cine_crudo_iter_cfg_btn = QPushButton("Iteraciones ⚙")
				self.cine_crudo_iter_cfg_btn.setMaximumWidth(104)
				self.cine_crudo_iter_cfg_btn.setToolTip(
					"Iteraciones/subsets POR ESTUDIO: ungated y gated de esfuerzo y reposo "
					"por separado (default 8 iter × 4 subsets).")
				self.cine_crudo_iter_cfg_btn.clicked.connect(self._open_iter_config_dialog)
				toolbar6_r2.addWidget(self.cine_crudo_iter_cfg_btn)
				# Fondo: preprocesado del sinograma COMÚN a todo el estudio (ungated +
				# gated comparten las proyecciones crudas). No es de una rama: va acá.
				self.cine_crudo_bg_check = QCheckBox("Fondo")
				self.cine_crudo_bg_check.setChecked(False)
				self.cine_crudo_bg_check.setToolTip(
					"Descuento de fondo automático (pre-recon, en el sinograma, aplica a "
					"TODO el estudio: ungated + gated). Mide el piso en la zona de bajo "
					"fondo (pulmón/tejido; excluye aire y vísceras calientes) y lo resta "
					"de TODA la imagen, incluido el VI. Aumenta el contraste cavidad/pared. "
					"Default OFF.")
				toolbar6_r2.addWidget(self.cine_crudo_bg_check)
				# --- Descuento de SCATTER (EM − k×SC): preprocesado de las
				# proyecciones crudas ANTES de reconstruir. Como Fondo, es COMÚN a
				# todo el estudio (afecta AMBAS ramas: ungated + gated). Por eso va
				# acá, en la fila de funciones comunes, no en una rama.
				self.cine_crudo_scatter_check = QCheckBox("Desc. SC")
				self.cine_crudo_scatter_check.setChecked(False)
				self.cine_crudo_scatter_check.setEnabled(False)
				self.cine_crudo_scatter_check.setToolTip(
					"Descuento de SCATTER dual-energy (P = EM − k×SC) en las proyecciones "
					"ANTES de reconstruir. Afecta a AMBAS ramas (ungated + gated). Solo "
					"disponible si el estudio tiene archivo hermano _SC (p.ej. exportación "
					"Infinia EM/SC) o dual-energy en un archivo. Default OFF.")
				toolbar6_r2.addWidget(self.cine_crudo_scatter_check)
				# Al activar/desactivar o cambiar k: refrescar el cine crudo (preview
				# EM−SC en vivo) y la recon si ya se hizo.
				self.cine_crudo_scatter_check.toggled.connect(self._on_scatter_preview_changed)
				toolbar6_r2.addWidget(QLabel("k"))
				self.cine_crudo_scatter_k_spin = QDoubleSpinBox()
				self.cine_crudo_scatter_k_spin.setRange(0.0, 2.0)
				self.cine_crudo_scatter_k_spin.setSingleStep(0.05)
				self.cine_crudo_scatter_k_spin.setDecimals(2)
				self.cine_crudo_scatter_k_spin.setValue(1.0)
				self.cine_crudo_scatter_k_spin.setMaximumWidth(58)
				self.cine_crudo_scatter_k_spin.setEnabled(False)
				self.cine_crudo_scatter_k_spin.setToolTip(
					"Factor k de la resta de scatter (P = EM - k×SC). 1.0 = resta "
					"directa (dual-window). Con TEW se calcula de los anchos de "
					"ventana. Calibrar con fantoma/estudio real.")
				self.cine_crudo_scatter_k_spin.valueChanged.connect(self._on_scatter_preview_changed)
				toolbar6_r2.addWidget(self.cine_crudo_scatter_k_spin)
				# --- CT/ATT + AC: corrección de atenuación iterativa (OSEM/MLEM). El
				# μ-map es POR ETAPA (cada etapa con su propio CT, no se comparte).
				self.cine_crudo_ct_btn = QPushButton("CT/ATT")
				self.cine_crudo_ct_btn.setMaximumWidth(64)
				self.cine_crudo_ct_btn.setToolTip(
					"Carga CT o mapa de atenuación (ATTMAP/CTAC) para la ETAPA ACTIVA.\n"
					"• ATTMAP exportado por el equipo: se usa como μ-map directo.\n"
					"• CT en HU: se convierte a μ-map con el modelo bilineal CTAC (140 keV).\n"
					"Cada etapa (Esfuerzo/Reposo) requiere su propio CT.")
				self.cine_crudo_ct_btn.clicked.connect(self._load_cine_crudo_ct_attmap)
				toolbar6_r2.addWidget(self.cine_crudo_ct_btn)
				self.cine_crudo_ac_check = QCheckBox("AC")
				self.cine_crudo_ac_check.setChecked(False)
				self.cine_crudo_ac_check.setEnabled(False)
				self.cine_crudo_ac_check.setToolTip(
					"Corrección de atenuación física en la reconstrucción iterativa "
					"(OSEM/MLEM modelan el μ-map en el forward/backprojection). Requiere "
					"CT/ATT cargado para la etapa. FBP no usa AC iterativa. El pasajero "
					"de fase SIEMPRE va sin AC (límites normales calibrados sin AC).")
				toolbar6_r2.addWidget(self.cine_crudo_ac_check)
				self.cine_crudo_ac_qc_btn = QPushButton("QC AC")
				self.cine_crudo_ac_qc_btn.setMaximumWidth(58)
				self.cine_crudo_ac_qc_btn.setToolTip(
					"QC visual de la alineación μ-map ↔ reconstrucción: contornos del "
					"cuerpo (cian) y tejido denso (naranja) del CT/ATT superpuestos al "
					"volumen ungated en axial/coronal/sagital. Requiere recon previa. "
					"Segundo click: vuelve a la vista anterior.")
				self.cine_crudo_ac_qc_btn.clicked.connect(self._show_ac_qc)
				toolbar6_r2.addWidget(self.cine_crudo_ac_qc_btn)
				self.cine_crudo_fusion_btn = QPushButton("Fusión")
				self.cine_crudo_fusion_btn.setMaximumWidth(60)
				self.cine_crudo_fusion_btn.setEnabled(False)
				self.cine_crudo_fusion_btn.setToolTip(
					"Abre el panel de fusión SPECT/CT de PERFUSIÓN (adaptado del de AMYLO) "
					"precargado con el crudo de la etapa activa: recon + cortes ahí, cargar CT, "
					"registrar (Ctrl/Shift/medio), fusionar en cortes tomo o cardiacos. "
					"Después 'Reg→AC' trae el registro acá.")
				self.cine_crudo_fusion_btn.clicked.connect(self._show_ct_fusion_preview)
				toolbar6_r2.addWidget(self.cine_crudo_fusion_btn)
				self.cine_crudo_import_reg_btn = QPushButton("Reg→AC")
				self.cine_crudo_import_reg_btn.setMaximumWidth(64)
				self.cine_crudo_import_reg_btn.setToolTip(
					"Importa el CT/ATT registrado en la ventana de fusión hacia la etapa activa "
					"y habilita AC: la próxima 'Recon raw' aplica la corrección con ESE registro.")
				self.cine_crudo_import_reg_btn.clicked.connect(self._import_amylo_fusion_registration)
				# Redundante: el botón 7 del panel de fusión hace esto automáticamente.
				self.cine_crudo_import_reg_btn.setVisible(False)
				toolbar6_r2.addWidget(self.cine_crudo_import_reg_btn)
				# --- Filtros por rama en DOS filas (ungated / gated): la fila única se
				# iba de pantalla y no permitía activar un filtro en una rama y otro en
				# la otra. UNGATED = perfusión estática (alto conteo) -> NÍTIDA(RR) para
				# nitidez. GATED = movimiento (bajo conteo/gate) -> FBP CLEAN / NITIDA
				# III / NITIDA II. Fondo y las acciones van en las comunes (r2).
				toolbar6_r_filters = QHBoxLayout()   # fila UNGATED
				toolbar6_r_filters_g = QHBoxLayout()  # fila GATED
				toolbar6_r_filters.addWidget(QLabel("<b>Ungated:</b>"))
				self.cine_crudo_nitida_check = QCheckBox("NÍTIDA")
				self.cine_crudo_nitida_check.setChecked(False)
				self.cine_crudo_nitida_check.setToolTip(
					"NÍTIDA (OmniRes): recuperación de resolución dependiente de profundidad "
					"(modela la respuesta colimador-detector según el DICOM). Requiere OSEM/MLEM "
					"(si está en FBP, se fuerza OSEM). Al activarla toma el control de la "
					"reconstrucción: desactiva los filtros de proyección (para no pre-difuminar "
					"lo que va a des-difuminar) y enciende 'Suavizar' (post-filtro que regula el "
					"ruido que amplifica la recuperación de resolución). Opcional; medio-dosis/medio-tiempo.")
				self.cine_crudo_nitida_check.toggled.connect(self._on_nitida_toggled)
				toolbar6_r_filters.addWidget(self.cine_crudo_nitida_check)
				self.cine_crudo_post_check = QCheckBox("Suavizar")
				self.cine_crudo_post_check.setChecked(False)
				self.cine_crudo_post_check.setToolTip(
					"Post-filtro gaussiano 3D opcional (control de ruido tras la reconstrucción). "
					"Es la contraparte de la recuperación de resolución: OSEM+NÍTIDA realza detalle "
					"pero amplifica ruido; este único suavizado lo regula. Default OFF.")
				self.cine_crudo_post_check.toggled.connect(self._on_post_filter_toggled)
				toolbar6_r_filters.addWidget(self.cine_crudo_post_check)
				self.cine_crudo_post_fwhm_spin = QDoubleSpinBox()
				self.cine_crudo_post_fwhm_spin.setRange(0.0, 30.0)
				self.cine_crudo_post_fwhm_spin.setSingleStep(0.5)
				self.cine_crudo_post_fwhm_spin.setDecimals(1)
				self.cine_crudo_post_fwhm_spin.setValue(8.0)
				self.cine_crudo_post_fwhm_spin.setSuffix(" mm")
				self.cine_crudo_post_fwhm_spin.setMaximumWidth(72)
				self.cine_crudo_post_fwhm_spin.setEnabled(False)
				self.cine_crudo_post_fwhm_spin.setToolTip("FWHM del suavizado gaussiano UNGATED [mm]. Típico 6–10 mm.")
				toolbar6_r_filters.addWidget(self.cine_crudo_post_fwhm_spin)
				# Denoise+ UNGATED: denoise de sinograma + realce por resta (abre la
				# cavidad y afina la pared; el ungated también sufre scatter/fondo).
				self.cine_crudo_denoise_plus_check = QCheckBox("Denoise+")
				self.cine_crudo_denoise_plus_check.setChecked(False)
				self.cine_crudo_denoise_plus_check.setToolTip(
					"Denoise+ UNGATED: denoise bilateral del sinograma + realce por resta "
					"(misma idea que FBP_CLEAN pero para el ungated de alto conteo). Abre "
					"la cavidad y afina la pared (el ungated también sufre scatter/fondo "
					"que la rellena). Medido: contraste cavidad/pared 0.68 -> 0.79. Default OFF.")
				toolbar6_r_filters.addWidget(self.cine_crudo_denoise_plus_check)
				toolbar6_r_filters.addWidget(QLabel("k"))
				self.cine_crudo_denoise_plus_slider = QSlider(Qt.Orientation.Horizontal)
				self.cine_crudo_denoise_plus_slider.setRange(0, 50)   # k = 0.00..0.50
				self.cine_crudo_denoise_plus_slider.setValue(20)       # óptimo medido k=0.20
				self.cine_crudo_denoise_plus_slider.setMaximumWidth(80)
				self.cine_crudo_denoise_plus_slider.setToolTip(
					"Factor de realce k del Denoise+ ungated (0.00–0.50). "
					"ÓPTIMO medido ~0.20: abre la cavidad sin comer la pared. "
					"Más de ~0.5 empieza a comer la pared.")
				toolbar6_r_filters.addWidget(self.cine_crudo_denoise_plus_slider)
				self.cine_crudo_denoise_plus_lbl = QLabel("0.20")
				self.cine_crudo_denoise_plus_lbl.setMaximumWidth(36)
				toolbar6_r_filters.addWidget(self.cine_crudo_denoise_plus_lbl)
				self.cine_crudo_denoise_plus_slider.valueChanged.connect(
					lambda v: self.cine_crudo_denoise_plus_lbl.setText(f"{v/100.0:.2f}"))
				# NOTA: Motion-frozen fue retirado de la UI (2026-08-13). El pipeline
				# (RawReconConfig.motion_frozen) y core/motion_frozen.py quedan
				# disponibles y documentados por si sirven para otros cálculos.
				toolbar6_r_filters.addStretch(1)
				# --- Fila GATED ---
				toolbar6_r_filters_g.addWidget(QLabel("<b>Gated:</b>"))
				# --- FBP_CLEAN: denoise Poisson en sinograma + realce por resta ---
				# (banco 023/025/026/027; idea del usuario). Ver core.fbp_clean.
				self.cine_crudo_fbpclean_check = QCheckBox("FBP CLEAN")
				self.cine_crudo_fbpclean_check.setChecked(False)
				self.cine_crudo_fbpclean_check.setToolTip(
					"FBP_CLEAN: denoise Poisson del SINOGRAMA (bilateral σc=0.04) antes del FBP "
					"(ataca las estrías en la raíz, no en la imagen) + realce de cavidad/bordes "
					"por resta de una fracción de la versión muy suavizada (unsharp mask). "
					"Para estudios de mitad de tiempo/dosis. Default OFF.")
				toolbar6_r_filters_g.addWidget(self.cine_crudo_fbpclean_check)
				toolbar6_r_filters_g.addWidget(QLabel("realce"))
				self.cine_crudo_fbpclean_slider = QSlider(Qt.Orientation.Horizontal)
				self.cine_crudo_fbpclean_slider.setRange(30, 70)  # k = 0.30..0.70
				self.cine_crudo_fbpclean_slider.setValue(50)      # default k=0.50
				self.cine_crudo_fbpclean_slider.setMaximumWidth(80)
				self.cine_crudo_fbpclean_slider.setToolTip("Factor de realce k (0.30–0.70). Más k = más realce de cavidad/bordes pero más ruido de fondo. Default 0.50.")
				toolbar6_r_filters_g.addWidget(self.cine_crudo_fbpclean_slider)
				self.cine_crudo_fbpclean_lbl = QLabel("0.50")
				self.cine_crudo_fbpclean_lbl.setMaximumWidth(36)
				toolbar6_r_filters_g.addWidget(self.cine_crudo_fbpclean_lbl)
				self.cine_crudo_fbpclean_slider.valueChanged.connect(
					lambda v: self.cine_crudo_fbpclean_lbl.setText(f"{v/100.0:.2f}"))
				# FBP_CLEAN es una cadena autocontenida (FBP + Butterworth + denoise +
				# realce): anula los filtros de proyección, el método iterativo y el
				# post-gaussiano. Los grisamos visualmente; el ÚNICO que sigue activo
				# es NITIDA II (post-recon gated, otra capa compatible).
				self.cine_crudo_fbpclean_check.toggled.connect(self._refresh_fbpclean_filter_lock)
				self.cine_crudo_nitida3_check = QCheckBox("NITIDA III")
				self.cine_crudo_nitida3_check.setChecked(False)
				self.cine_crudo_nitida3_check.setToolTip(
					"NITIDA III: reconstrucción MAP-OSEM del GATED con Pilar C "
					"(prior de suavidad Huber con beta adaptativo por SNR local). "
					"Limpia la pared (ruido) SIN aplastar el movimiento cardíaco (H1): "
					"medido en estudio real 5s, SNR x1.5 conservando el movimiento. "
					"Para gated de bajo conteo. Es OSEM CPU: tarda más. Default OFF.")
				toolbar6_r_filters_g.addWidget(self.cine_crudo_nitida3_check)
				self.cine_crudo_nitida3_iter_spin = QSpinBox()
				self.cine_crudo_nitida3_iter_spin.setRange(2, 8)
				self.cine_crudo_nitida3_iter_spin.setValue(2)
				self.cine_crudo_nitida3_iter_spin.setMaximumWidth(46)
				self.cine_crudo_nitida3_iter_spin.setToolTip(
					"Iteraciones OSEM de NITIDA III. Con 2 queda granulado/pixelado "
					"(OSEM poco convergente en bajo conteo). Subilo a 4-6 para una recon "
					"más suave. Más iteraciones = más tiempo de cómputo.")
				toolbar6_r_filters_g.addWidget(self.cine_crudo_nitida3_iter_spin)
				# NITIDA 4D (4D-OSEM): prior TEMPORAL entre gates. Reconstruye los gates
				# juntos compartiendo información entre vecinos temporales (sin
				# promediar: no congela el latido). Para gated de bajo conteo.
				self.cine_crudo_nitida4d_check = QCheckBox("NITIDA 4D")
				self.cine_crudo_nitida4d_check.setChecked(False)
				self.cine_crudo_nitida4d_check.setToolTip(
					"NITIDA 4D (4D-OSEM): reconstrucción del GATED con prior TEMPORAL "
					"Huber entre gates vecinos DENTRO del update OSEM. Comparte "
					"estadística entre gates SIN promediar (no congela el latido, a "
					"diferencia del motion-frozen). Limpia el ruido conservando la "
					"contracción. Para gated de bajo conteo. Es OSEM CPU: tarda más. "
					"Default OFF.")
				toolbar6_r_filters_g.addWidget(self.cine_crudo_nitida4d_check)
				toolbar6_r_filters_g.addWidget(QLabel("βt"))
				self.cine_crudo_nitida4d_beta_spin = QDoubleSpinBox()
				self.cine_crudo_nitida4d_beta_spin.setRange(0.0, 1.0)
				self.cine_crudo_nitida4d_beta_spin.setSingleStep(0.1)
				self.cine_crudo_nitida4d_beta_spin.setDecimals(2)
				self.cine_crudo_nitida4d_beta_spin.setValue(0.3)
				self.cine_crudo_nitida4d_beta_spin.setMaximumWidth(58)
				self.cine_crudo_nitida4d_beta_spin.setToolTip(
					"Beta temporal (fuerza del acoplamiento entre gates). 0 = OSEM por "
					"gate independiente. 0.2-0.4 = limpieza moderada conservando el "
					"latido. >0.6 = más limpieza pero riesgo de amplificar el "
					"movimiento (medido en fantoma).")
				toolbar6_r_filters_g.addWidget(self.cine_crudo_nitida4d_beta_spin)
				toolbar6_r_filters_g.addWidget(QLabel("NITIDA II"))
				self.cine_crudo_nitida2_combo = QComboBox()
				self.cine_crudo_nitida2_combo.addItem("Off", "none")
				self.cine_crudo_nitida2_combo.addItem("Temporal", "temporal")
				self.cine_crudo_nitida2_combo.addItem("Espaciotemporal", "spatiotemporal")
				self.cine_crudo_nitida2_combo.setMaximumWidth(120)
				self.cine_crudo_nitida2_combo.setToolTip(
					"NITIDA II: denoiser gated por armónicos temporales (post-recon, solo el gated). "
					"'Temporal' conserva el movimiento cardíaco exacto (bandas de baja frecuencia) y "
					"elimina el ruido de banda alta. 'Espaciotemporal' además limpia el granulado "
					"espacial de la media y suaviza levemente el movimiento. Independiente de NÍTIDA (RR).")
				toolbar6_r_filters_g.addWidget(self.cine_crudo_nitida2_combo)
				# Suavizar GATED (post-filtro por rama, independiente del ungated).
				self.cine_crudo_post_gated_check = QCheckBox("Suavizar")
				self.cine_crudo_post_gated_check.setChecked(False)
				self.cine_crudo_post_gated_check.setToolTip(
					"Post-filtro gaussiano 3D del GATED (control de ruido tras la recon). "
					"Independiente del 'Suavizar' del ungated. Default OFF.")
				self.cine_crudo_post_gated_check.toggled.connect(self._on_post_filter_gated_toggled)
				toolbar6_r_filters_g.addWidget(self.cine_crudo_post_gated_check)
				self.cine_crudo_post_gated_fwhm_spin = QDoubleSpinBox()
				self.cine_crudo_post_gated_fwhm_spin.setRange(0.0, 30.0)
				self.cine_crudo_post_gated_fwhm_spin.setSingleStep(0.5)
				self.cine_crudo_post_gated_fwhm_spin.setDecimals(1)
				self.cine_crudo_post_gated_fwhm_spin.setValue(8.0)
				self.cine_crudo_post_gated_fwhm_spin.setSuffix(" mm")
				self.cine_crudo_post_gated_fwhm_spin.setMaximumWidth(72)
				self.cine_crudo_post_gated_fwhm_spin.setEnabled(False)
				self.cine_crudo_post_gated_fwhm_spin.setToolTip("FWHM del suavizado gaussiano GATED [mm].")
				toolbar6_r_filters_g.addWidget(self.cine_crudo_post_gated_fwhm_spin)
				# Denoise+ GATED: denoise de sinograma + realce por resta para la rama
				# gated, con CUALQUIER método (FBP u OSEM). FBP_CLEAN solo corre si gated
				# es FBP; como el gated ahora defaultea a OSEM, quedaba sin tratamiento
				# de cavidad. Esto abre la cavidad del gated (empata visual al ungated).
				self.cine_crudo_denoise_plus_gated_check = QCheckBox("Denoise+")
				self.cine_crudo_denoise_plus_gated_check.setChecked(False)
				self.cine_crudo_denoise_plus_gated_check.setToolTip(
					"Denoise+ GATED: denoise bilateral del sinograma gated + realce de "
					"cavidad por resta (misma idea que Denoise+ ungated pero para la rama "
					"gated, con CUALQUIER método FBP/OSEM). Abre la cavidad del gated que "
					"a veces se pierde en OSEM. k default 0.50 (bajo conteo tolera más "
					"realce que el ungated). Default OFF.")
				toolbar6_r_filters_g.addWidget(self.cine_crudo_denoise_plus_gated_check)
				toolbar6_r_filters_g.addWidget(QLabel("k"))
				self.cine_crudo_denoise_plus_gated_slider = QSlider(Qt.Orientation.Horizontal)
				self.cine_crudo_denoise_plus_gated_slider.setRange(0, 100)  # k = 0.00..1.00
				self.cine_crudo_denoise_plus_gated_slider.setValue(50)      # default k=0.50
				self.cine_crudo_denoise_plus_gated_slider.setMaximumWidth(80)
				self.cine_crudo_denoise_plus_gated_slider.setToolTip(
					"Factor de realce k del Denoise+ gated (0.00–1.00). Default 0.50 "
					"(calibrado para bajo conteo). Más k = más apertura de cavidad pero "
					"más ruido de fondo.")
				toolbar6_r_filters_g.addWidget(self.cine_crudo_denoise_plus_gated_slider)
				self.cine_crudo_denoise_plus_gated_lbl = QLabel("0.50")
				self.cine_crudo_denoise_plus_gated_lbl.setMaximumWidth(36)
				toolbar6_r_filters_g.addWidget(self.cine_crudo_denoise_plus_gated_lbl)
				self.cine_crudo_denoise_plus_gated_slider.valueChanged.connect(
					lambda v: self.cine_crudo_denoise_plus_gated_lbl.setText(f"{v/100.0:.2f}"))
				toolbar6_r_filters_g.addStretch(1)
				# Selector de etapa para 'Recon raw': Esfuerzo / Reposo / Ambas.
				self.cine_crudo_recon_stage_combo = QComboBox()
				self.cine_crudo_recon_stage_combo.addItem("Esfuerzo", "stress")
				self.cine_crudo_recon_stage_combo.addItem("Reposo", "rest")
				self.cine_crudo_recon_stage_combo.addItem("Ambas", "both")
				self.cine_crudo_recon_stage_combo.setMaximumWidth(86)
				self.cine_crudo_recon_stage_combo.setToolTip("Qué etapa(s) reconstruye 'Recon raw'.")
				toolbar6_r2.addWidget(self.cine_crudo_recon_stage_combo)
				self.cine_crudo_recon_btn = QToolButton()
				self.cine_crudo_recon_btn.setText("Recon raw")
				self.cine_crudo_recon_btn.setToolTip("Reconstruye desde crudo gated la(s) etapa(s) del selector con la corrección actual y muestra QC: UngGat + gates.")
				self.cine_crudo_recon_btn.clicked.connect(self._on_recon_raw_clicked)
				toolbar6_r2.addWidget(self.cine_crudo_recon_btn)
				self.cine_crudo_recon_feta_btn = QToolButton()
				self.cine_crudo_recon_feta_btn.setText("Reconstruir selección")
				self.cine_crudo_recon_feta_btn.setToolTip("Reconstruye SOLO la banda axial (feta) entre las líneas Base/Ápex de esta pantalla. Flujo: 1) 'Recon raw' (FBP rápido) para ver el corazón; 2) ajustá Base/Ápex sobre las líneas rojas; 3) 'Reconstruir selección'. Excluye la actividad extracardíaca de arriba/abajo, es más rápido, y es el volumen con el que se reorienta y analiza de aquí en más.")
				self.cine_crudo_recon_feta_btn.clicked.connect(lambda: self._reconstruct_cine_crudo_raw(feta_only=True))
				toolbar6_r2.addWidget(self.cine_crudo_recon_feta_btn)
				self.cine_crudo_reorient_btn = QToolButton()
				self.cine_crudo_reorient_btn.setText("Reorientar")
				self.cine_crudo_reorient_btn.setToolTip("Abre la reorientación oblicua interactiva (Rec/Ref estilo Xeleris): definí eje largo del VI en vistas anterior/lateral, ROI y límites Base/Ápex, con preview SA/HLA/VLA en vivo.")
				self.cine_crudo_reorient_btn.clicked.connect(self._open_cine_crudo_reorientation)
				self.cine_crudo_reorient_btn.setEnabled(False)
				toolbar6_r2.addWidget(self.cine_crudo_reorient_btn)
				toolbar6_r2.addWidget(QLabel("Base"))
				self.cine_crudo_cut_base_spin = QSpinBox()
				self.cine_crudo_cut_base_spin.setRange(1, 1)
				self.cine_crudo_cut_base_spin.setValue(1)
				self.cine_crudo_cut_base_spin.setMaximumWidth(58)
				self.cine_crudo_cut_base_spin.setEnabled(False)
				self.cine_crudo_cut_base_spin.setToolTip("Primer corte SA a conservar (límite basal). Ajuste tipo líneas de límite Odyssey/Xeleris.")
				self.cine_crudo_cut_base_spin.valueChanged.connect(self._preview_cine_crudo_cut_limits)
				toolbar6_r2.addWidget(self.cine_crudo_cut_base_spin)
				toolbar6_r2.addWidget(QLabel("Ápex"))
				self.cine_crudo_cut_apex_spin = QSpinBox()
				self.cine_crudo_cut_apex_spin.setRange(1, 1)
				self.cine_crudo_cut_apex_spin.setValue(1)
				self.cine_crudo_cut_apex_spin.setMaximumWidth(58)
				self.cine_crudo_cut_apex_spin.setEnabled(False)
				self.cine_crudo_cut_apex_spin.setToolTip("Último corte SA a conservar (límite apical). El cubo SA generado baja a sincronía/FEVI.")
				self.cine_crudo_cut_apex_spin.valueChanged.connect(self._preview_cine_crudo_cut_limits)
				toolbar6_r2.addWidget(self.cine_crudo_cut_apex_spin)
				toolbar6_r2.addWidget(QLabel("Esp"))
				self.cine_crudo_cut_thickness_spin = QSpinBox()
				self.cine_crudo_cut_thickness_spin.setRange(1, 9)
				self.cine_crudo_cut_thickness_spin.setValue(1)
				self.cine_crudo_cut_thickness_spin.setMaximumWidth(50)
				self.cine_crudo_cut_thickness_spin.setEnabled(False)
				self.cine_crudo_cut_thickness_spin.setToolTip("Espesor de corte en píxeles. Cada SA se genera promediando este espesor alrededor del plano seleccionado.")
				self.cine_crudo_cut_thickness_spin.valueChanged.connect(self._preview_cine_crudo_cut_limits)
				toolbar6_r2.addWidget(self.cine_crudo_cut_thickness_spin)
				toolbar6_r2.addWidget(QLabel("Interp cortes"))
				self.cine_crudo_cuts_interp_combo = QComboBox()
				# Continuo nítido→suave (interpolación de display, NO altera datos).
				# Etiqueta visible -> método matplotlib.
				self.cine_crudo_cuts_interp_combo.addItems(["Píxel", "Bilineal", "Bicúbico", "Hanning", "Lanczos"])
				self.cine_crudo_cuts_interp_combo.setCurrentText("Bilineal")
				self.cine_crudo_cuts_interp_combo.setMaximumWidth(90)
				self.cine_crudo_cuts_interp_combo.setToolTip(
					"Tipo de interpolación de VISUALIZACIÓN de los cortes (no altera datos ni análisis).\n"
					"Píxel = vóxel crudo (bloques). Bilineal = intermedio fiel (recomendado).\n"
					"Bicúbico/Hanning/Lanczos = progresivamente más suaves.")
				self.cine_crudo_cuts_interp_combo.currentIndexChanged.connect(lambda *_: self._refresh_cine_crudo_cuts_smoothing())
				toolbar6_r2.addWidget(self.cine_crudo_cuts_interp_combo)
				toolbar6_r2.addWidget(QLabel("Suav. cortes"))
				self.cine_crudo_cuts_smooth_spin = QDoubleSpinBox()
				self.cine_crudo_cuts_smooth_spin.setRange(0.0, 3.0)
				self.cine_crudo_cuts_smooth_spin.setSingleStep(0.2)
				self.cine_crudo_cuts_smooth_spin.setDecimals(1)
				self.cine_crudo_cuts_smooth_spin.setValue(0.0)
				self.cine_crudo_cuts_smooth_spin.setMaximumWidth(56)
				self.cine_crudo_cuts_smooth_spin.setToolTip(
					"Suavizado gaussiano EXTRA [px] de los cortes (post-filtro de display, no altera datos).\n"
					"0.0 = solo la interpolación elegida. >0 = agrega difuminado gaussiano encima.\n"
					"Independiente del tipo de interpolación: combinalos a gusto del médico.")
				self.cine_crudo_cuts_smooth_spin.valueChanged.connect(lambda *_: self._refresh_cine_crudo_cuts_smoothing())
				toolbar6_r2.addWidget(self.cine_crudo_cuts_smooth_spin)
				toolbar6_r2.addStretch(1)

				self.cine_crudo_preview_limits_btn = QToolButton()
				self.cine_crudo_preview_limits_btn.setText("Ver límites")
				self.cine_crudo_preview_limits_btn.setToolTip("Muestra líneas Base/Ápex sobre el volumen reconstruido y cortes SA de referencia antes de generar ejes.")
				self.cine_crudo_preview_limits_btn.clicked.connect(self._preview_cine_crudo_cut_limits)
				self.cine_crudo_preview_limits_btn.setEnabled(False)
				toolbar6_r3.addWidget(self.cine_crudo_preview_limits_btn)
				self.cine_crudo_generate_cuts_btn = QToolButton()
				self.cine_crudo_generate_cuts_btn.setText("Generar cortes")
				self.cine_crudo_generate_cuts_btn.setToolTip("Genera los cortes cardíacos SA/HLA/VLA desde el volumen reconstruido y los muestra en comparacion_ejes.")
				self.cine_crudo_generate_cuts_btn.clicked.connect(self._generate_cine_crudo_cardiac_cuts)
				self.cine_crudo_generate_cuts_btn.setEnabled(False)
				toolbar6_r3.addWidget(self.cine_crudo_generate_cuts_btn)
				self.cine_crudo_save_axes_dcm_btn = QToolButton()
				self.cine_crudo_save_axes_dcm_btn.setText("Guardar ejes DICOM")
				self.cine_crudo_save_axes_dcm_btn.setToolTip("Guarda SA, HLA y VLA generados como DICOM NM gated multiframe para reutilizar en SINCRO u otro software DICOM.")
				self.cine_crudo_save_axes_dcm_btn.clicked.connect(self._save_cine_crudo_axes_dicoms)
				self.cine_crudo_save_axes_dcm_btn.setEnabled(False)
				toolbar6_r3.addWidget(self.cine_crudo_save_axes_dcm_btn)
				self.cine_crudo_process_recon_btn = QToolButton()
				self.cine_crudo_process_recon_btn.setText("Procesar recon")
				self.cine_crudo_process_recon_btn.setToolTip("Procesa fase/FEVI desde cortes SA. Con Esfuerzo+Reposo generados, procesa automáticamente AMBAS etapas y arma la comparación.")
				self.cine_crudo_process_recon_btn.clicked.connect(self._process_cine_crudo_reconstruction)
				self.cine_crudo_process_recon_btn.setEnabled(False)
				toolbar6_r3.addWidget(self.cine_crudo_process_recon_btn)
				self.cine_crudo_copy_rois_to_rest_btn = QToolButton()
				self.cine_crudo_copy_rois_to_rest_btn.setText("ROI E→R")
				self.cine_crudo_copy_rois_to_rest_btn.setToolTip("Copia los ROI manuales de Esfuerzo a Reposo como punto inicial. Después podés ajustar Reposo fino en el cine.")
				self.cine_crudo_copy_rois_to_rest_btn.clicked.connect(self._copy_stress_rois_to_rest)
				self.cine_crudo_copy_rois_to_rest_btn.setEnabled(False)
				toolbar6_r3.addWidget(self.cine_crudo_copy_rois_to_rest_btn)
				toolbar6_r3.addStretch(1)
			if name == "comparacion_ejes":
				# Controles de acción del montaje, centralizados en esta pestaña.
				self._build_montage_toolbar_into(toolbar)
			toolbar.addStretch(1)
			tab_layout.addLayout(toolbar)
			if name == "cine_crudo":
				# Filas 2-7 agrupadas en menús desplegables: la fila 1 (arriba)
				# tiene los controles esenciales de reproducción y queda siempre
				# visible; el resto (corrección de movimiento, ajuste manual,
				# reconstrucción, montaje) vive en un botón con menú. A
				# diferencia de un panel colapsable, el menú es un popup que
				# flota por encima y NUNCA cambia el tamaño de la imagen.
				groups_row = QHBoxLayout()
				groups_row.addWidget(self._build_toolbar_group_menu(
					"Corrección de movimiento ▾", [toolbar2, toolbar3, toolbar_export],
					key="cine_crudo_correccion_movimiento",
					tooltip="Método de corrección, ajuste manual, offsets y exportar/importar/grabar DICOM.",
					side_widget=self.cine_crudo_correct_side,
				))
				groups_row.addWidget(self._build_toolbar_group_menu(
					"Reconstrucción desde crudo ▾", [toolbar6_r1, toolbar6_r2, toolbar6_r_filters, toolbar6_r_filters_g, toolbar6_r3],
					key="cine_crudo_reconstruccion",
					tooltip="Reconstrucción FBP/MLEM/OSEM, filtros de ungated/gated, reorientación y generación de cortes de eje.",
				))
				groups_row.addStretch(1)
				tab_layout.addLayout(groups_row)

			label = QLabel("Sin procesar")
			label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
			label.setMinimumSize(500, 320)
			label.setStyleSheet("background:#111; color:#ddd; border:1px solid #444;")
			label.setScaledContents(False)
			label.setMouseTracking(True)
			helptxt = preview_help_texts.get(name, "")
			if helptxt:
				label.setToolTip(f"{helptxt}\n\nZoom con +/- o 100%.")
			else:
				label.setToolTip("Zoom con los botones +/- o 100% arriba de cada panel.")
			self.preview_labels[name] = label
			self.preview_zoom[name] = self._default_preview_zoom(name)
			if name == "cine_crudo":
				label.mousePressEvent = (lambda e, lbl=label: self._on_cine_crudo_mouse_press_safe(e, lbl))
				label.mouseMoveEvent = (lambda e, lbl=label: self._on_cine_crudo_mouse_move_safe(e, lbl))
				label.mouseReleaseEvent = (lambda e, lbl=label: self._on_cine_crudo_mouse_release_safe(e, lbl))
				label.wheelEvent = (lambda e, lbl=label: self._on_cine_crudo_mouse_wheel_safe(e, lbl))
				label.mouseDoubleClickEvent = (lambda e, lbl=label: self._on_cine_crudo_mouse_double_click_safe(e, lbl))
			if name == "comparacion_ejes":
				# El montaje clínico ahora vive en esta pestaña: además del drag para
				# Base/Ápex, hay que cablear rueda (mover tira / Ctrl+zoom) y doble
				# click (reset de tira), igual que en cine_crudo.
				label.mousePressEvent = (lambda e, lbl=label: self._on_cine_crudo_mouse_press_safe(e, lbl))
				label.mouseMoveEvent = (lambda e, lbl=label: self._on_cine_crudo_mouse_move_safe(e, lbl))
				label.mouseReleaseEvent = (lambda e, lbl=label: self._on_cine_crudo_mouse_release_safe(e, lbl))
				label.wheelEvent = (lambda e, lbl=label: self._on_cine_crudo_mouse_wheel_safe(e, lbl))
				label.mouseDoubleClickEvent = (lambda e, lbl=label: self._on_cine_crudo_mouse_double_click_safe(e, lbl))
			scroller = QScrollArea()
			scroller.setWidgetResizable(False)
			scroller.setWidget(label)
			self._preview_scrollers[name] = scroller
			if name == "comparacion_ejes":
				# Controles de color/ventaneo pegados al panel (antes vivían lejos,
				# en la barra del montaje). Columna vertical a la derecha: cmap +
				# modo Percentil/Lineal + RangeSlider 200% + tira del LUT.
				ce_row = QHBoxLayout()
				ce_row.setContentsMargins(0, 0, 0, 0)
				ce_row.addWidget(scroller, 1)
				ce_row.addWidget(self._build_compare_axes_color_column())
				tab_layout.addLayout(ce_row)
			elif name == "cine_crudo":
				# Motor de color pegado al preview: cmap + RangeSlider 0..200% + tira LUT.
				cc_row = QHBoxLayout()
				cc_row.setContentsMargins(0, 0, 0, 0)
				cc_row.addWidget(scroller, 1)
				cc_row.addWidget(self._build_cine_crudo_color_column())
				tab_layout.addLayout(cc_row)
			elif name == "polar_perfusion_directa":
				# Color de pantalla del mapa polar: solo cmap + tira LUT (sin ventana).
				pp_row = QHBoxLayout()
				pp_row.setContentsMargins(0, 0, 0, 0)
				pp_row.addWidget(scroller, 1)
				pp_row.addWidget(self._build_polar_screen_color_column())
				tab_layout.addLayout(pp_row)
			else:
				tab_layout.addWidget(scroller)
			self._tab_widgets[name] = tab
			self._tab_titles[name] = preview_titles.get(name, name)
			self._tab_tooltips[name] = helptxt or ""

		self._rebuild_tabs_for_mode()
		self.cine = CineWidget(compact_viewer=True)
		self.cine.roiEdited.connect(self._on_cine_roi_changed)
		self.cine.roiEditedGate.connect(self._on_cine_roi_changed_gate)
		self.cine.playStateChanged.connect(self._on_play_state_changed)
		self.cine.activated.connect(lambda: self._on_cine_panel_activated("main"))
		self.cine.centerPicked.connect(self._on_center_picked)
		self.cine.setToolTip("Reproducí el cine, hacé zoom y dibujá ROIs sobre la imagen.")
		self.cine_secondary_source: str | None = None
		# is_compare=True: oculta su título interno y sus sliders Base/Top (son
		# solo de la 1ra etapa), para que las dos imágenes queden alineadas.
		self.cine_compare = CineWidget(compact_viewer=True, is_compare=True)
		self.cine_compare.roiEdited.connect(self._on_cine_compare_roi_changed)
		self.cine_compare.roiEditedGate.connect(self._on_cine_compare_roi_changed_gate)
		self.cine_compare.playStateChanged.connect(self._on_play_state_changed)
		self.cine_compare.activated.connect(lambda: self._on_cine_panel_activated("secondary"))
		self.cine_compare.setToolTip("Segundo visor (otro estudio): editable para ajustar ROI esfuerzo/reposo en paralelo.")
		self.cine.set_controls_visible(True)
		self.cine_compare.set_controls_visible(False)
		# Sincronización: el compare refleja colormap, invertir y velocidad del cine
		# principal. Un solo juego de controles maneja ambas imágenes.
		self.cine.cmap_combo.currentTextChanged.connect(
			self.cine_compare.cmap_combo.setCurrentText
		)
		self.cine.invert_cmap_check.toggled.connect(
			self.cine_compare.invert_cmap_check.setChecked
		)
		self.cine.speed_slider.valueChanged.connect(
			self.cine_compare.speed_slider.setValue
		)
		self.cine.interp_combo.currentTextChanged.connect(
			self.cine_compare.set_display_interp
		)
		self.cine.playStateChanged.connect(
			self._sync_cine_compare_playback
		)
		# Sincronización gate/slice: el mismo corte en ambas etapas para comparar.
		self.cine.gate_slider.valueChanged.connect(
			self.cine_compare.gate_slider.setValue
		)
		self.cine.slice_slider.valueChanged.connect(
			self.cine_compare.slice_slider.setValue
		)
		# Controles de centro manual, a mano justo encima del visor donde se clickea.
		self.manual_center_clear_btn.setText("Limpiar centros")
		# Se agrupan en un widget propio para poder ocultarlos como bloque cuando la
		# sincronía se centraliza en la ventana "vista asincronía".
		self.manual_center_bar = QWidget()
		manual_center_bar_layout = QHBoxLayout(self.manual_center_bar)
		manual_center_bar_layout.setContentsMargins(0, 0, 0, 0)
		manual_center_bar_layout.addWidget(QLabel("Centro:"))
		manual_center_bar_layout.addWidget(self.manual_center_check)
		manual_center_bar_layout.addWidget(self.manual_center_all_check)
		manual_center_bar_layout.addWidget(self.manual_center_clear_btn)
		lower_header = QHBoxLayout()
		lower_header.setContentsMargins(0, 0, 0, 0)
		# El cine_compare (2da etapa) queda SIEMPRE VISIBLE: ya no hay botón
		# Mostrar/Ocultar. El header solo lleva los controles de centro manual.
		lower_header.addWidget(self.manual_center_bar)
		lower_header.addStretch(1)
		# Botón debug: dibuja la grilla (bordes rojos 2px) del layout del cine para
		# diagnosticar desalineaciones. Temporal.
		self.debug_grid_btn = QToolButton()
		self.debug_grid_btn.setText("⊞ Grilla")
		self.debug_grid_btn.setCheckable(True)
		self.debug_grid_btn.setToolTip("Debug: muestra/oculta la grilla del layout (bordes rojos).")
		self.debug_grid_btn.toggled.connect(self._on_debug_grid_toggled)
		self.debug_grid_btn.setVisible(False)

		# Contenedor del cine principal: header (centro/colapsar) + el visor.
		# El segundo visor (cine_compare) ya NO va acá; según el mockup vive al
		# extremo derecho de la banda inferior (ver bottom_hsplit más abajo).
		cine_area = QWidget()
		cine_area.setMinimumWidth(456)  # evita que el splitter comprima las imágenes
		cine_area_layout = QVBoxLayout(cine_area)
		cine_area_layout.setContentsMargins(0, 0, 0, 0)
		cine_area_layout.setSpacing(2)
		cine_area_layout.addLayout(lower_header)
		cine_area_layout.addWidget(self.cine)

		# El cine_compare (2da etapa) va PEGADO al cine principal, con su propia
		# grilla 3x2 (título '2da. Fase' + imagen + sliders + slice/gate), igual
		# que la 1ra. Su título dinámico se actualiza en _refresh_cine_compare_title.
		self.cine.set_compare_viewer(self.cine_compare)

		# Banda inferior (izquierda → derecha):
		#   [ cine principal + 2da etapa (juntos) ] [ Datos Paciente / Resultados ]
		#   [ curvas: histograma de fase + volumen/derivada ]
		self.bottom_hsplit = QSplitter(Qt.Orientation.Horizontal)
		# El cine (1ra+2da) no colapsa para que los sliders no se solapen sobre
		# las imágenes. El mintrack total está controlado por el minimumSize de
		# la ventana (≈1280px), no por los mínimos de cada zona.
		self.bottom_hsplit.setChildrenCollapsible(False)
		self.bottom_hsplit.setHandleWidth(6)
		self.bottom_hsplit.addWidget(cine_area)
		# Bloque clínico único: Resultados + curvas 2×2 y una franja Δ común.
		# Evita que el splitter comprima los cuatro gráficos a una tira angosta.
		results_curves = QWidget()
		results_curves_l = QVBoxLayout(results_curves)
		results_curves_l.setContentsMargins(0, 0, 0, 0)
		results_curves_l.setSpacing(2)
		self.results_curves_split = QSplitter(Qt.Orientation.Horizontal)
		self.results_curves_split.setChildrenCollapsible(False)
		self.results_curves_split.setHandleWidth(6)
		self.results_curves_split.addWidget(self._build_readonly_results_panel())
		self.results_curves_split.addWidget(self._build_curves_panel())
		self.results_curves_split.setStretchFactor(0, 0)
		self.results_curves_split.setStretchFactor(1, 1)
		self.results_curves_split.setSizes([310, 520])
		results_curves_l.addWidget(self.results_curves_split, 1)
		self.main_delta_readout = self._build_main_delta_readout()
		results_curves_l.addWidget(self.main_delta_readout, 0)
		self.bottom_hsplit.addWidget(results_curves)
		self.bottom_hsplit.setStretchFactor(0, 1)
		self.bottom_hsplit.setStretchFactor(1, 2)

		lower_cine_panel = QWidget()
		self._lower_cine_panel = lower_cine_panel
		lower_cine_layout = QVBoxLayout(lower_cine_panel)
		lower_cine_layout.setContentsMargins(0, 0, 0, 0)
		lower_cine_layout.setSpacing(2)
		# Header delgado con toggle para colapsar la banda (cine + resultados +
		# curvas) y dar todo el alto a las pestañas. El right_splitter no colapsa
		# por drag (setChildrenCollapsible(False)); el toggle lo hace por botón.
		lower_band_header = QHBoxLayout()
		lower_band_header.setContentsMargins(4, 0, 4, 0)
		lower_band_header.setSpacing(6)
		self.lower_cine_collapse_btn = QToolButton()
		self.lower_cine_collapse_btn.setText("▾")
		self.lower_cine_collapse_btn.setAutoRaise(True)
		self.lower_cine_collapse_btn.setToolTip("Colapsar/expandir la banda inferior (cine, resultados y curvas).")
		self.lower_cine_collapse_btn.clicked.connect(self._toggle_lower_cine_band)
		lower_band_header.addWidget(self.lower_cine_collapse_btn)
		lower_band_header.addWidget(QLabel("Cine / Resultados / Curvas"))
		lower_band_header.addStretch(1)
		lower_cine_layout.addLayout(lower_band_header)
		lower_cine_layout.addWidget(self.bottom_hsplit)
		self._lower_cine_collapsed = False
		self._right_splitter_saved_sizes = None
		right_splitter.addWidget(self.tabs)
		right_splitter.addWidget(lower_cine_panel)
		right_splitter.setStretchFactor(0, 3)
		right_splitter.setStretchFactor(1, 1)
		right_layout.addWidget(right_splitter)

		splitter.addWidget(left)
		splitter.addWidget(right)
		splitter.setStretchFactor(0, 1)
		splitter.setStretchFactor(1, 4)
		splitter.setSizes([300, 1260])
		right_splitter.setSizes([840, 220])
		# 3 zonas: cine+2da etapa | Datos+Resultados | curvas. La primera es más
		# ancha porque ahora contiene las DOS imágenes (cine + cine_compare).
		# Banda inferior: 25% visores/controles, 75% datos + resultados + curvas.
		self.bottom_hsplit.setSizes([250, 750])
		self.main_splitter = splitter
		self.right_splitter = right_splitter
		self._ui_settings = QSettings("Gammasys", "GammaSync")
		self._load_global_ui_preferences()
		self._restore_window_layout()
		self._restore_sidebar_sections_state()
		self._restore_fevi_settings()
		self._rebuild_recent_dirs_menu()

		layout = QVBoxLayout(central)
		layout.addWidget(splitter)
		self.statusBar().showMessage("Listo")
		self.tabs.currentChanged.connect(self._on_preview_tab_changed)
		self.polar_cine_speed_spin.valueChanged.connect(self._on_polar_cine_speed_changed)
		self.cmap_combo.currentTextChanged.connect(self._on_phase_cmap_changed)
		self.preset_patient_edit.textChanged.connect(lambda _=None: self._refresh_presets_for_current_patient())
		self._on_phase_cmap_changed(self.cmap_combo.currentText())
		self._refresh_presets_for_current_patient()
		self._capture_global_tooltips()
		self._apply_global_ui_preferences()
		self._update_cine_active_border()
		self._hide_sync_controls_in_main()
		self._refresh_readonly_results_panel()
		self._install_undo_shortcuts()
		if initial_path:
			self.file_edit.setText(initial_path)
			if self.auto_run_check.isChecked():
				self.process_auto()
			else:
				self.process_current()

	def _hide_sync_controls_in_main(self):
		"""Oculta en el sidebar principal los controles de sincronía ya duplicados
		en la ventana 'vista asincronía' (ROI de análisis, centrar en cavidad y
		centro manual). Los widgets siguen existiendo y operativos; solo dejan de
		mostrarse acá, porque la sincronía se maneja desde 'vista asincronía'."""
		form = getattr(self, "controls_form", None)
		for w in (getattr(self, "roi_source_combo", None), getattr(self, "cavity_center_check", None)):
			if w is None:
				continue
			if form is not None:
				lbl = form.labelForField(w)
				if lbl is not None:
					lbl.setVisible(False)
			w.setVisible(False)
		bar = getattr(self, "manual_center_bar", None)
		if bar is not None:
			bar.setVisible(False)

	# ==================================================================
	# Flujo ida-y-vuelta (Fase A): pasos + deshacer/rehacer (Ctrl+Z)
	# ==================================================================
	#: Estado que captura/restaura cada acción deshacible (por grupo de paso).
	UNDO_ATTRS_MOTION = (
		"cine_crudo_seed", "cine_crudo_seed_compare",
		"cine_crudo_band_upper", "cine_crudo_band_lower",
		"cine_crudo_compare_line_y", "cine_crudo_ref_index",
		"cine_crudo_ref_index_compare",
	)
	UNDO_ATTRS_CUTS = (
		"cine_crudo_axes_for_export", "cine_crudo_axes_for_export_stress",
		"cine_crudo_axes_for_export_rest", "cine_crudo_cut_thickness_mm",
		"cine_crudo_cut_thickness_mm_rest", "cine_crudo_rest_source_label",
		"cine_crudo_gate_from", "cine_crudo_gate_to",
	)
	#: Pasos "pesados": el atributo se reemplaza entero (no se muta), así que se
	#: guarda por referencia en vez de deepcopy para no duplicar volúmenes en RAM.
	UNDO_ATTRS_RECON = (
		"cine_crudo_recon_result", "cine_crudo_recon_study",
		"cine_crudo_recon_result_phase",
		"cine_crudo_cut_study", "cine_crudo_cut_source_label",
		"cine_crudo_reoriented_gated", "cine_crudo_reoriented_ungated",
		"cine_crudo_reoriented_gated_phase", "cine_crudo_reoriented_mf",
		"cine_crudo_raw_study_for_recon", "_cine_crudo_recon_stage",
		"cine_crudo_preview_mode",
	)
	UNDO_ATTRS_REORIENT = (
		"cine_crudo_reoriented_gated", "cine_crudo_reoriented_ungated",
		"cine_crudo_reoriented_gated_phase", "cine_crudo_reoriented_mf",
		"cine_crudo_reoriented_voi", "cine_crudo_cut_source_label",
	)

	def _register_pipeline_steps(self):
		"""Registra los pasos del pipeline unificado (barra de pasos, Fase B)."""
		for key, label in (
			("crudo", "Crudo"),
			("motion", "Corrección de movimiento"),
			("recon", "Reconstrucción"),
			("reorient", "Reorientación"),
			("cuts", "Generar cortes"),
			("segment", "Segmentación"),
			("phase", "Fase + métricas"),
			("render", "Render / Montaje / Informe"),
		):
			self.pipeline_history.register_step(key, label)

	def _install_undo_shortcuts(self):
		"""Instala Ctrl+Z (deshacer) y Ctrl+Shift+Z / Ctrl+Y (rehacer)."""
		from PyQt6.QtGui import QKeySequence, QShortcut
		self._sc_undo = QShortcut(QKeySequence.StandardKey.Undo, self)
		self._sc_undo.activated.connect(self._undo_pipeline)
		self._sc_redo = QShortcut(QKeySequence.StandardKey.Redo, self)
		self._sc_redo.activated.connect(self._redo_pipeline)
		self._sc_redo2 = QShortcut(QKeySequence("Ctrl+Y"), self)
		self._sc_redo2.activated.connect(self._redo_pipeline)

	def _snapshot_attrs(self, names, deep: bool = True) -> dict:
		"""Captura los atributos indicados para poder restaurarlos.

		`deep=True` copia en profundidad (pasos livianos). `deep=False` guarda
		referencias (pasos pesados cuyo atributo se reemplaza entero, no se muta).
		"""
		import copy
		snap: dict = {}
		for n in names:
			val = getattr(self, n, None)
			if not deep:
				snap[n] = val
				continue
			try:
				snap[n] = copy.deepcopy(val)
			except Exception:
				snap[n] = val  # objetos no copiables: guardar referencia
		return snap

	def _apply_attrs_snapshot(self, snapshot: dict):
		"""Restaura atributos desde un snapshot y refresca la vista."""
		for n, v in snapshot.items():
			setattr(self, n, v)
		self._refresh_after_undo()

	def _commit_undo(self, label: str, attr_names, before, deep: bool = True):
		"""Cierra una acción deshacible: captura el estado posterior y lo apila.

		`before` es el snapshot tomado antes de la acción; si es None (acción
		anidada o undo suspendido) no se registra nada. `deep` debe coincidir con
		el usado al tomar `before` (False para pasos pesados).
		"""
		if before is None:
			return
		after = self._snapshot_attrs(attr_names, deep=deep)
		self.pipeline_history.push(
			label,
			undo=lambda b=before: self._apply_attrs_snapshot(b),
			redo=lambda a=after: self._apply_attrs_snapshot(a),
		)

	def _refresh_after_undo(self):
		"""Refresca la vista tras un deshacer/rehacer, según el modo activo."""
		try:
			if getattr(self, "cine_crudo_preview_mode", None) == "sa_montage":
				self._schedule_montage_refresh(0)
			else:
				self._refresh_cine_crudo_view()
		except Exception:
			pass

	def _undo_pipeline(self):
		label = self.pipeline_history.undo()
		if label:
			self._log(f"Deshacer: {label}")
			self.statusBar().showMessage(f"Deshacer: {label}")
		else:
			self.statusBar().showMessage("Nada para deshacer")

	def _redo_pipeline(self):
		label = self.pipeline_history.redo()
		if label:
			self._log(f"Rehacer: {label}")
			self.statusBar().showMessage(f"Rehacer: {label}")
		else:
			self.statusBar().showMessage("Nada para rehacer")

	# --------------------------------------------------- barra de pasos (Fase B)
	#: Color por estado del paso: (fondo, texto, borde).
	_STEP_COLORS = {
		"empty": ("#e5e7eb", "#6b7280", "#d1d5db"),
		"valid": ("#dcfce7", "#166534", "#86efac"),
		"stale": ("#fef3c7", "#92400e", "#fcd34d"),
	}
	#: Pestaña a la que salta el clic en cada chip de paso.
	_STEP_TABS = {
		"crudo": "cine_crudo", "motion": "cine_crudo", "recon": "cine_crudo",
		"reorient": "cine_crudo", "cuts": "comparacion_ejes", "render": "comparacion_ejes",
		"segment": "slices_fase", "phase": "slices_fase",
	}

	def _build_pipeline_step_bar(self) -> QWidget:
		"""Barra horizontal con el estado de cada paso del pipeline (Fase B)."""
		bar = QWidget()
		bar.setObjectName("pipelineStepBar")
		lay = QHBoxLayout(bar)
		lay.setContentsMargins(8, 3, 8, 3)
		lay.setSpacing(4)
		self._step_chip_labels: dict[str, QPushButton] = {}
		steps = self.pipeline_history.steps()
		for i, st in enumerate(steps):
			if i > 0:
				if st.key == "segment":
					# Divisor Preparación | Análisis.
					div = QLabel("┃")
					div.setStyleSheet("color:#3b82f6; font-size:13pt; font-weight:bold; margin:0 2px;")
					div.setToolTip("← Preparación   |   Análisis →")
					lay.addWidget(div)
				else:
					sep = QLabel("›")
					sep.setStyleSheet("color:#9ca3af; font-size:11pt;")
					lay.addWidget(sep)
			chip = QPushButton(st.label)
			chip.setFlat(True)
			chip.setCursor(Qt.CursorShape.PointingHandCursor)
			chip.clicked.connect(lambda _=False, k=st.key: self._on_step_chip_clicked(k))
			self._step_chip_labels[st.key] = chip
			lay.addWidget(chip)
		lay.addStretch(1)
		self._undo_hint_label = QLabel("")
		self._undo_hint_label.setStyleSheet("color:#6b7280; font-size:8pt;")
		lay.addWidget(self._undo_hint_label)
		self._recompute_btn = QPushButton("↻ Recalcular")
		self._recompute_btn.setToolTip("Reejecuta en cadena los pasos desactualizados (los que se puedan automáticamente).")
		self._recompute_btn.setStyleSheet(
			"QPushButton{background:#fef3c7; color:#92400e; border:1px solid #fcd34d;"
			"border-radius:8px; padding:2px 10px; font-size:9pt; font-weight:600;}"
			"QPushButton:disabled{background:#f3f4f6; color:#9ca3af; border-color:#e5e7eb;}"
		)
		self._recompute_btn.clicked.connect(self._recompute_stale_steps)
		lay.addWidget(self._recompute_btn)
		self._refresh_pipeline_step_bar()
		return bar

	#: Pasos con recómputo automático (sin diálogo). El resto requiere acción manual.
	def _step_recomputers(self) -> dict:
		return {
			"recon": self._reconstruct_cine_crudo_raw,
			"cuts": self._generate_cine_crudo_cardiac_cuts,
		}

	def _recompute_stale_steps(self):
		"""Reejecuta en orden los pasos desactualizados con recómputo automático.

		Se detiene en el primer paso desactualizado que requiere acción manual
		(p.ej. reorientación, que abre un diálogo), avisando al usuario.
		"""
		recomputers = self._step_recomputers()
		ran = 0
		for st in self.pipeline_history.steps():
			if st.status.value != "stale":
				continue
			fn = recomputers.get(st.key)
			if fn is None:
				self._log(f"[RECALCULAR] '{st.label}' requiere acción manual; me detengo ahí.")
				self.statusBar().showMessage(f"Recalcular: '{st.label}' necesita acción manual")
				break
			self._log(f"[RECALCULAR] Reejecutando '{st.label}'...")
			try:
				fn()
				ran += 1
			except Exception as exc:
				self._log(f"[RECALCULAR] '{st.label}' falló: {exc}")
				break
		if ran:
			self.statusBar().showMessage(f"Recalcular: {ran} paso(s) reejecutado(s)")
		elif not self.pipeline_history.stale_steps():
			self.statusBar().showMessage("Recalcular: no hay pasos desactualizados")
		self._refresh_pipeline_step_bar()

	def _on_step_chip_clicked(self, key: str):
		"""Clic en un chip: salta a la vista/pestaña asociada a ese paso.

		Es robusto: si la vista destino no está visible (pestaña avanzada en modo
		básico, o cine_crudo movida a la ventana de Preparación) intenta habilitarla
		antes de navegar, y si no puede avisa en la barra de estado.
		"""
		st = self.pipeline_history.get(key)
		label = st.label if st is not None else key
		tab = self._STEP_TABS.get(key)
		if not tab:
			self.statusBar().showMessage(f"Paso: {label}")
			return
		# cine_crudo puede estar reubicada en la ventana de Preparación.
		if tab == "cine_crudo" and getattr(self, "_cine_crudo_reparented", False):
			try:
				self.open_preparacion_window()
				self.statusBar().showMessage(f"Paso: {label} — en ventana de Preparación")
			except Exception:
				self.statusBar().showMessage(f"'{label}': ventana de Preparación no disponible")
			return
		if self._select_tab_by_title(tab):
			self.statusBar().showMessage(f"Paso: {label}")
		else:
			self.statusBar().showMessage(f"La vista '{tab}' no está disponible en este modo")

	def _refresh_pipeline_step_bar(self):
		"""Repinta la barra de pasos según el estado de PipelineHistory."""
		chips = getattr(self, "_step_chip_labels", None)
		if not chips:
			return
		for st in self.pipeline_history.steps():
			chip = chips.get(st.key)
			if chip is None:
				continue
			bg, fg, border = self._STEP_COLORS.get(st.status.value, self._STEP_COLORS["empty"])
			chip.setStyleSheet(
				f"QPushButton{{background:{bg}; color:{fg}; border:1px solid {border};"
				"border-radius:8px; padding:2px 9px; font-size:9pt; font-weight:600;}"
				f"QPushButton:hover{{border:1px solid #3b82f6;}}"
			)
			tip = {"empty": "pendiente", "valid": "al día", "stale": "desactualizado"}.get(st.status.value, "")
			chip.setToolTip(f"{st.label} — {tip} · clic para ver este paso")
		hint = getattr(self, "_undo_hint_label", None)
		if hint is not None:
			parts = []
			if self.pipeline_history.can_undo():
				parts.append(f"⟲ {self.pipeline_history.peek_undo_label()}")
			if self.pipeline_history.can_redo():
				parts.append(f"⟳ {self.pipeline_history.peek_redo_label()}")
			hint.setText("   ".join(parts))
		btn = getattr(self, "_recompute_btn", None)
		if btn is not None:
			stale_keys = {s.key for s in self.pipeline_history.stale_steps()}
			has_auto = bool(stale_keys & set(self._step_recomputers()))
			btn.setEnabled(has_auto)

	def _mark_step_done(self, key: str, *sig_parts):
		"""Marca un paso como al día; invalida los posteriores solo si sus inputs cambiaron."""
		try:
			new_sig = self.pipeline_history.make_signature(*sig_parts) if sig_parts else ""
			prev = self.pipeline_history.get(key)
			prev_sig = prev.signature if prev is not None else ""
			self.pipeline_history.mark_done(key, new_sig)
			# Invalidación fina: si la firma (inputs) no cambió, el output es el
			# mismo y no hay que desactualizar la cadena posterior.
			if sig_parts and new_sig != prev_sig:
				self.pipeline_history.invalidate_after(key)
		except Exception:
			pass

	def _build_readonly_results_panel(self) -> QWidget:
		"""Construye el panel de solo-lectura de la banda inferior: datos del
		paciente, resultados en vivo (mismas métricas que 'vista asincronía') y
		las dos curvas ya renderizadas (histograma de fase y volumen/derivada).
		No tiene controles: solo refleja lo calculado por el motor."""
		panel = QWidget()
		panel.setMinimumWidth(240)
		panel.setMaximumWidth(420)
		lay = QVBoxLayout(panel)
		lay.setContentsMargins(3, 3, 3, 3)
		lay.setSpacing(3)

		# Según el mockup: "Datos del Paciente" es una caja ancha ARRIBA y
		# "Resultados en vivo" va DEBAJO, a la altura de los controles del cine.
		pat_box = QGroupBox("Datos del Paciente")
		pat_l = QVBoxLayout(pat_box)
		pat_l.setContentsMargins(6, 3, 6, 3)
		self.patient_data_label = QLabel("Sin estudio cargado.")
		self.patient_data_label.setWordWrap(True)
		self.patient_data_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
		self.patient_data_label.setStyleSheet("font-size:9pt; color:#1f2937;")
		pat_l.addWidget(self.patient_data_label)
		lay.addWidget(pat_box, 0)

		res_box = QGroupBox("Resultados en vivo")
		res_l = QVBoxLayout(res_box)
		res_l.setContentsMargins(6, 3, 6, 3)
		self.main_metrics_readout = QLabel("Sin resultados: procesá un estudio.")
		self.main_metrics_readout.setWordWrap(True)
		self.main_metrics_readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
		self.main_metrics_readout.setStyleSheet("font-size:10pt; color:#1f2937;")
		res_l.addWidget(self.main_metrics_readout)
		lay.addWidget(res_box, 1)
		self.readonly_panel = panel
		return panel

	def _build_main_delta_readout(self) -> QLabel:
		"""Franja Δ común para resultados y gráficos en el flujo dual."""
		label = QLabel("")
		label.setWordWrap(False)
		label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
		label.setStyleSheet(
			"font-size:9pt; color:#0f172a; background:#f8fafc; "
			"border-top:1px solid #cbd5e1; padding:4px 7px;"
		)
		label.setVisible(False)
		return label

	def _build_curves_panel(self) -> QWidget:
		"""Zona de curvas clínicas: 1 columna en simple, 2×2 en dual.

		Fila superior: histogramas de fase/asynchrony. Fila inferior: FEVI.
		Las columnas son Esfuerzo y Reposo cuando ambas etapas están disponibles.
		"""
		panel = QWidget()
		panel.setMinimumWidth(220)
		lay = QGridLayout(panel)
		lay.setContentsMargins(3, 3, 3, 3)
		lay.setSpacing(3)
		def _curve_label(text: str) -> QLabel:
			label = QLabel(text)
			label.setAlignment(Qt.AlignmentFlag.AlignCenter)
			label.setMinimumHeight(96)
			label.setStyleSheet("color:#5b6470; background:#0b1220; border:1px solid #26324a;")
			return label
		self.curve_hist_view = _curve_label("Histograma de fase: procesá un estudio.")
		self.curve_hist_compare_view = _curve_label("Histograma Reposo: sin segunda etapa.")
		self.curve_fevi_view = _curve_label("Curva volumen/derivada: modo avanzado.")
		self.curve_fevi_compare_view = _curve_label("Curva FEVI Reposo: sin segunda etapa.")
		lay.addWidget(self.curve_hist_view, 0, 0)
		lay.addWidget(self.curve_hist_compare_view, 0, 1)
		lay.addWidget(self.curve_fevi_view, 1, 0)
		lay.addWidget(self.curve_fevi_compare_view, 1, 1)
		lay.setColumnStretch(0, 1)
		lay.setColumnStretch(1, 1)
		lay.setRowStretch(0, 1)
		lay.setRowStretch(1, 1)
		self.curves_panel = panel
		return panel

	def _load_curve_pixmap(self, label: QLabel, filename: str, placeholder: str) -> None:
		"""Carga un PNG ya renderizado (desde output_dir) en un QLabel escalado al
		ancho disponible. Si el archivo no existe, deja un placeholder."""
		try:
			path = os.path.join(self.output_dir, filename)
		except Exception:
			path = ""
		if not path or not os.path.isfile(path):
			label.setPixmap(QPixmap())
			label.setText(placeholder)
			return
		pix = QPixmap(path)
		if pix.isNull():
			label.setPixmap(QPixmap())
			label.setText(placeholder)
			return
		target_w = max(240, label.width() - 4)
		label.setText("")
		label.setPixmap(pix.scaledToWidth(target_w, Qt.TransformationMode.SmoothTransformation))

	def _load_curve_pixmap_from_path(self, label: QLabel, path: str, placeholder: str) -> None:
		"""Carga una curva desde una ruta explícita (salida de la 2da etapa)."""
		if not path or not os.path.isfile(path):
			label.setPixmap(QPixmap())
			label.setText(placeholder)
			return
		pix = QPixmap(path)
		if pix.isNull():
			label.setPixmap(QPixmap())
			label.setText(placeholder)
			return
		label.setText("")
		label.setPixmap(pix.scaledToWidth(max(200, label.width() - 4), Qt.TransformationMode.SmoothTransformation))

	def _render_empty_curve(self, label: QLabel, xlabel: str, ylabel: str, filename: str) -> None:
		"""Dibuja unos ejes X/Y vacíos (sin datos) en `label`, para no dejar
		el gráfico anterior 'pegado' cuando no hay estudio procesado."""
		try:
			import matplotlib
			matplotlib.use("Agg")
			import matplotlib.pyplot as plt
			bg, fg, grid = "#0b1220", "#5b6470", "#26324a"
			fig, ax = plt.subplots(figsize=(5, 2.2), facecolor=bg)
			ax.set_facecolor(bg)
			ax.set_xlim(0, 1)
			ax.set_ylim(0, 1)
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_xlabel(xlabel, color=fg, fontsize=9)
			ax.set_ylabel(ylabel, color=fg, fontsize=9)
			for spine in ax.spines.values():
				spine.set_color(grid)
			ax.grid(True, color=grid, alpha=0.35)
			out = os.path.join(self.output_dir, filename)
			fig.tight_layout()
			fig.savefig(out, dpi=140, facecolor=bg, bbox_inches="tight")
			plt.close(fig)
			pix = QPixmap(out)
			target_w = max(240, label.width() - 4)
			label.setText("")
			label.setPixmap(pix.scaledToWidth(target_w, Qt.TransformationMode.SmoothTransformation))
		except Exception:
			label.setPixmap(QPixmap())
			label.setText("")

	def _visual_style_dict(self) -> dict:
		"""Paleta de los paneles clínicos (curva FEVI, panel funcional gated,
		bull's eye) según el tema visual seleccionado. Fuente única de verdad."""
		style_catalog = {
			"clinico": {
				"fig_bg": "#050811",
				"ax_bg": "#0a1424",
				"grid": "#1f3a5f",
				"fg": "#dbeafe",
				"subtle": "#93c5fd",
				"vol": "#fde047",
				"deriv": "#60a5fa",
				"ed": "#86efac",
				"es": "#fca5a5",
				"amp_cmap": "viridis",
				"bull_cmap": "plasma",
			},
			"gammasync": {
				"fig_bg": "#f8fafc",
				"ax_bg": "#ffffff",
				"grid": "#cbd5e1",
				"fg": "#0f172a",
				"subtle": "#475569",
				"vol": "#b45309",
				"deriv": "#0f766e",
				"ed": "#0ea5e9",
				"es": "#e11d48",
				"amp_cmap": "turbo",
				"bull_cmap": "turbo",
			},
		}
		style_name = str(self.visual_style_combo.currentText()).strip().lower() if hasattr(self, "visual_style_combo") else "clinico"
		if style_name not in style_catalog:
			style_name = "clinico"
		return style_catalog[style_name]

	def _render_fevi_curve_panel(self, label: QLabel, ef: dict | None, output_filename: str = "curva_fevi_panel.png", title_suffix: str = "") -> bool:
		"""Dibuja la curva volumen/gate (FEVI) autónoma para la banda inferior,
		replicando el panel 'Time/Volume y derivada' (mismo estilo y colores que
		Panel funcional gated). Devuelve True si pudo dibujar datos."""
		if not ef or not bool(ef.get("available")):
			return False
		try:
			gate_volumes = np.asarray(ef.get("gate_volumes_ml", []), dtype=np.float64)
			if gate_volumes.size < 2 or not np.isfinite(gate_volumes).any():
				return False
			import matplotlib
			matplotlib.use("Agg")
			import matplotlib.pyplot as plt
			style = self._visual_style_dict()
			t_gate = np.arange(1, gate_volumes.size + 1)
			dv = np.gradient(gate_volumes)
			fig, ax = plt.subplots(figsize=(5, 2.4), facecolor=style["fig_bg"])
			ax.set_facecolor(style["ax_bg"])
			ax.plot(t_gate, gate_volumes, color=style["vol"], linewidth=2.2, marker="o", markersize=4, label="Volumen")
			ax2 = ax.twinx()
			ax2.plot(t_gate, dv, color=style["deriv"], linewidth=1.8, label="dV/dgate")
			ax2.set_ylabel("dV/dgate", color=style["deriv"], fontsize=9)
			ax2.tick_params(axis="y", colors=style["deriv"], labelsize=7)
			ed_gate = int(ef.get("ed_gate", 1))
			es_gate = int(ef.get("es_gate", 1))
			ax.axvline(ed_gate, color=style["ed"], linestyle="--", linewidth=1.2)
			ax.axvline(es_gate, color=style["es"], linestyle="--", linewidth=1.2)
			vol_max = float(np.nanmax(gate_volumes)) if np.isfinite(gate_volumes).any() else 1.0
			ax.set_ylim(0.0, vol_max * 1.15)
			ef_txt = ""
			try:
				ef_val = float(ef.get("ef_pct"))
				if np.isfinite(ef_val):
					ef_txt = f" — FEVI {ef_val:.1f}%"
			except (TypeError, ValueError):
				pass
			ax.set_title(f"Time/Volume y derivada{title_suffix}{ef_txt}", color=style["fg"], fontsize=10, fontweight="bold")
			ax.set_xlabel("Gate", color=style["subtle"], fontsize=9)
			ax.set_ylabel("Volumen (mL)", color=style["vol"], fontsize=9)
			ax.tick_params(axis="x", colors=style["subtle"], labelsize=7)
			ax.tick_params(axis="y", colors=style["vol"], labelsize=7)
			for spine in ax.spines.values():
				spine.set_color(style["grid"])
			ax.grid(True, color=style["grid"], alpha=0.45)
			out = os.path.join(self.output_dir, output_filename)
			fig.tight_layout()
			fig.savefig(out, dpi=140, facecolor=style["fig_bg"], bbox_inches="tight")
			plt.close(fig)
			pix = QPixmap(out)
			target_w = max(240, label.width() - 4)
			label.setText("")
			label.setPixmap(pix.scaledToWidth(target_w, Qt.TransformationMode.SmoothTransformation))
			return True
		except Exception:
			return False

	def _render_phase_histogram_panel(self, label: QLabel, phase_result, metrics: dict | None, output_filename: str, title: str) -> bool:
		"""Render liviano de histograma para una etapa desde sus datos en memoria."""
		if phase_result is None or not isinstance(metrics, dict):
			return False
		try:
			from viz.histogram import build_phase_histogram, save_histogram
			fig = build_phase_histogram(
				np.asarray(phase_result.phases_deg, dtype=np.float64),
				metrics=metrics, bins=72, title=title,
			)
			out = os.path.join(self.output_dir, output_filename)
			save_histogram(fig, out, dpi=125)
			import matplotlib.pyplot as plt
			plt.close(fig)
			self._load_curve_pixmap_from_path(label, out, "Histograma: sin datos.")
			return True
		except Exception as exc:
			self._log(f"[WARN] Histograma de etapa no renderizado: {exc}")
			return False

	def _format_async_metrics_lines(self, metrics: dict, ef_pct=None, ef: dict | None = None) -> list[str]:
		"""Bloque de métricas de asincronía con la nomenclatura del ECTb 4.0
		(Fase máximo, SD, Ancho de banda, Sesgo, Curtosis) + Entropy y FEVI opcional.
		'Fase máximo' es el pico/moda del histograma (peak_phase), NO la media."""
		def m(key: str, decimals: int = 1) -> str:
			val = metrics.get(key) if isinstance(metrics, dict) else None
			try:
				return f"{float(val):.{decimals}f}"
			except (TypeError, ValueError):
				return "N/D"

		lines = [
			f"Fase máximo: {m('peak_phase')}°",
			f"SD: {m('phase_sd')}°",
			f"Ancho de banda: {m('bandwidth')}°",
			f"Sesgo: {m('skewness', 2)}",
			f"Curtosis: {m('kurtosis', 2)}",
			f"Entropy: {m('entropy_normalized_pct')} %",
		]
		if ef_pct is not None:
			try:
				lines.append(f"<b>FEVI: {float(ef_pct):.1f} %</b>")
			except (TypeError, ValueError):
				pass
		# Función diastólica (ECTb): PFR y TVmáx desde la curva de volumen.
		if isinstance(ef, dict) and ef.get("pfr_text"):
			lines.append(f"PFR: {ef.get('pfr_text')}")
			if ef.get("tvmax_text"):
				lines.append(f"TVmáx: {ef.get('tvmax_text')}")
		return lines

	def _format_async_delta_lines(self, primary: dict, compare: dict, primary_ef: dict | None = None, compare_ef: dict | None = None) -> list[str]:
		"""Deltas clínicos Esfuerzo − Reposo para el resumen dual."""
		if not isinstance(primary, dict) or not isinstance(compare, dict):
			return []
		def _delta(key: str, decimals: int = 1, suffix: str = "") -> str | None:
			try:
				a, b = float(primary.get(key)), float(compare.get(key))
				if np.isfinite(a) and np.isfinite(b):
					return f"Δ {key}: {a - b:+.{decimals}f}{suffix}"
			except (TypeError, ValueError):
				pass
			return None
		items = [
			_delta("peak_phase", 1, "°"),
			_delta("phase_sd", 1, "°"),
			_delta("bandwidth", 1, "°"),
			_delta("entropy_normalized_pct", 1, "%"),
		]
		try:
			a = float((primary_ef or {}).get("ef_pct"))
			b = float((compare_ef or {}).get("ef_pct"))
			if np.isfinite(a) and np.isfinite(b):
				items.append(f"Δ FEVI: {a - b:+.1f}%")
		except (TypeError, ValueError):
			pass
		items = [item for item in items if item]
		return ["<b>Δ Esfuerzo − Reposo</b>", *items] if items else []


	def _format_processing_info_lines(self) -> list[str]:
		"""Genera líneas compactas con TODOS los filtros, correcciones y parámetros aplicados."""
		study = getattr(self, "study", None)
		if study is None:
			return []
		parts: list[str] = []

		# --- Lectura segura de widgets ---
		def _w(name: str):
			return getattr(self, name, None)

		def _combo_text(name: str, default: str = "") -> str:
			w = _w(name)
			return str(w.currentText()).strip() if w is not None else default

		def _spin_val(name: str, default=0):
			w = _w(name)
			return w.value() if w is not None else default

		recon_method = _combo_text("cine_crudo_recon_method_combo", "").upper()
		nitida = bool(_w("cine_crudo_nitida_check") and _w("cine_crudo_nitida_check").isChecked())
		n_iter = int(_spin_val("cine_crudo_iter_spin", 0))
		n_sub = int(_spin_val("cine_crudo_osem_subsets_spin", 0))
		post_on = bool(_w("cine_crudo_post_check") and _w("cine_crudo_post_check").isChecked())
		post_fwhm = float(_spin_val("cine_crudo_post_fwhm_spin", 0.0))
		ung_kind = _combo_text("cine_crudo_ung_filter_combo", "butterworth")
		ung_cut = float(_spin_val("cine_crudo_ung_cutoff_spin", 0.52))
		ung_ord = int(_spin_val("cine_crudo_ung_order_spin", 5))
		gat_kind = _combo_text("cine_crudo_gated_filter_combo", "butterworth")
		gat_cut = float(_spin_val("cine_crudo_gated_cutoff_spin", 0.40))
		gat_ord = int(_spin_val("cine_crudo_gated_order_spin", 10))

		reconstructed_flag = bool(getattr(study, "reconstructed", True))
		motion = getattr(self, "cine_crudo_motion_result", None)
		recon_from_raw = motion is not None

		# En "Resultados en vivo" se muestran solo datos clínicos accionables.
		# Reconstrucción, filtros y correcciones permanecen disponibles en los
		# controles/Log e informe técnico, pero no ocupan este resumen.

		# Los parámetros técnicos de análisis se consultan en Configuración e
		# informe técnico; no pertenecen al resumen clínico "en vivo".
		return parts

	def _refresh_readonly_results_panel(self) -> None:
		"""Refresca el panel de solo-lectura de la banda inferior tras procesar."""
		if getattr(self, "patient_data_label", None) is None:
			return
		# --- Datos del paciente ---
		study = getattr(self, "study", None)
		second = self._second_stage_study()
		if study is None:
			self.patient_data_label.setText("Sin estudio cargado.")
		elif second is None:
			ctx = self._study_context()

			def g(attr: str) -> str:
				return str(getattr(study, attr, "") or "").strip() or "N/D"

			birth = self._format_dicom_date(str(getattr(study, "patient_birth_date", "") or "")) or "N/D"
			stime_raw = str(getattr(study, "study_time", "") or "").strip()
			if len(stime_raw) >= 4 and stime_raw[:4].isdigit():
				stime = f"{stime_raw[:2]}:{stime_raw[2:4]}"
			else:
				stime = ""
			self.patient_data_label.setText(
				f"<b>{ctx['patient_name']}</b> (ID: {ctx['patient_id']})<br>"
				f"Sexo: {g('patient_sex')} &nbsp;|&nbsp; Nac.: {birth}<br>"
				f"Estudio: {ctx['study_date']} {stime}<br>"
				f"Accession: {g('accession_number')}<br>"
				f"Desc.: {g('study_description')}<br>"
				f"Serie: {g('series_description')}<br>"
				f"Tipo: {ctx['phase']}"
			)
		else:
			# Dos etapas del MISMO paciente (garantizado por el guard de identidad
			# al cargar). Identidad una sola vez + dos columnas Esfuerzo/Reposo.
			ctx1 = self._study_context(study_obj=study)
			ctx2 = self._study_context(study_obj=second)

			def gg(st, attr: str) -> str:
				return str(getattr(st, attr, "") or "").strip() or "N/D"

			def stime_of(st) -> str:
				raw = str(getattr(st, "study_time", "") or "").strip()
				return f"{raw[:2]}:{raw[2:4]}" if len(raw) >= 4 and raw[:4].isdigit() else ""

			def stage_label(st, ctx, fallback: str) -> str:
				lab = self._cine_crudo_stage_display(st)
				if lab:
					return lab
				if ctx.get("phase") in ("Reposo", "Esfuerzo"):
					return ctx["phase"]
				return fallback

			lbl1 = stage_label(study, ctx1, "Etapa 1")
			lbl2 = stage_label(second, ctx2, "Etapa 2")
			# Evitar dos etiquetas iguales (ambas "Etapa" o ambas mismo tipo).
			if lbl2 == lbl1:
				lbl2 = "Reposo" if lbl1 == "Esfuerzo" else ("Esfuerzo" if lbl1 == "Reposo" else "Etapa 2")
			birth = self._format_dicom_date(str(getattr(study, "patient_birth_date", "") or "")) or "N/D"
			cell = "padding:1px 8px 1px 0;"
			self.patient_data_label.setText(
				f"<b>{ctx1['patient_name']}</b> (ID: {ctx1['patient_id']})<br>"
				f"Sexo: {gg(study, 'patient_sex')} &nbsp;|&nbsp; Nac.: {birth}<br>"
				f"<table style='border-spacing:0;'>"
				f"<tr><td style='{cell}'></td>"
				f"<td style='{cell}'><b>{lbl1}</b></td>"
				f"<td style='{cell}'><b>{lbl2}</b></td></tr>"
				f"<tr><td style='{cell}'>Fecha:</td>"
				f"<td style='{cell}'>{ctx1['study_date']} {stime_of(study)}</td>"
				f"<td style='{cell}'>{ctx2['study_date']} {stime_of(second)}</td></tr>"
				f"<tr><td style='{cell}'>Serie:</td>"
				f"<td style='{cell}'>{gg(study, 'series_description')}</td>"
				f"<td style='{cell}'>{gg(second, 'series_description')}</td></tr>"
				f"</table>"
			)

		# --- Resultados en vivo ---
		# EF calculado una sola vez y reutilizado por el readout y la curva FEVI.
		ef = None
		if getattr(self, "study", None) is not None and getattr(self, "phase_result", None) is not None:
			try:
				ef = self._estimate_lv_ef()
			except Exception:
				ef = None
		metrics = getattr(self, "metrics", None)
		if not metrics:
			self.main_metrics_readout.setText("Sin resultados: procesá el estudio.")
			self.main_delta_readout.setText("")
			self.main_delta_readout.setVisible(False)
		else:
			ef_pct = ef.get("ef_pct") if isinstance(ef, dict) else None
			compare_metrics = getattr(self, "compare_metrics", None)
			compare_ef = getattr(self, "compare_ef", None)
			# Determinar por datos qué etapa es la primaria visual y cuál es la
			# opuesta. No usar solo _cine_crudo_recon_stage: tras el orquestador
			# dual puede quedar apuntando a la última etapa aunque self.study sea la
			# primera, y eso terminaba consultando la misma etapa dos veces.
			other = None
			try:
				sess = self._dual_session()
				if self.study is sess.stage("stress").cut_study:
					other = sess.stage("rest")
				elif self.study is sess.stage("rest").cut_study:
					other = sess.stage("stress")
				else:
					active_stage = str(getattr(self, "_cine_crudo_recon_stage", "stress") or "stress")
					other = sess.stage("rest" if active_stage == "stress" else "stress")
			except Exception:
				other = None
			# La UI no debe depender exclusivamente de compare_bundle: en el flujo
			# dual los resultados de ambas etapas viven canónicamente en DualSession.
			# Si el render/HQ diferido todavía no instaló el bundle, mostrar igual la
			# segunda etapa apenas estén sus métricas disponibles.
			try:
				# La etapa primaria real es la que alimenta self.study (stress en el
				# flujo dual normal); buscar SIEMPRE su complementaria, sin depender
				# del _recon_stage que puede quedar apuntando al último paso (reposo).
				dual_rest = self._dual_session().stage("rest")
				dual_stress = self._dual_session().stage("stress")
				if dual_rest.cut_study is self.study:
					dual_other = dual_stress
				else:
					dual_other = dual_rest
				if compare_metrics is None and dual_other.metrics is not None:
					compare_metrics = dual_other.metrics
					compare_ef = dual_other.ef
			except Exception:
				pass
			if compare_metrics is None:
				try:
					if other is not None and other.metrics is not None:
						compare_metrics = other.metrics
						compare_ef = other.ef
				except Exception:
					pass
			if compare_metrics is not None:
				# Dos etapas apiladas como la pantalla Asincronía VI del ECTb (b_07):
				# etapa procesada arriba, etapa de comparación abajo, + delta.
				try:
					primary_phase = str(self._study_context().get("phase", "") or "").strip()
				except Exception:
					primary_phase = ""
				primary_label = primary_phase or "Etapa 1"
				second_phase = self._second_phase_label()
				compare_label = (
					second_phase
					if second_phase in ("Reposo", "Esfuerzo")
					else str(self.compare_label or "Etapa 2").strip()
				)
				# Dos columnas equivalentes: lectura simultánea Esfuerzo | Reposo.
				# Mantiene los mismos campos por etapa para comparar sin buscar entre
				# bloques apilados de texto.
				left_lines = "<br>".join(self._format_async_metrics_lines(metrics, ef_pct, ef))
				right_lines = "<br>".join(self._format_async_metrics_lines(
					compare_metrics,
					compare_ef.get("ef_pct") if isinstance(compare_ef, dict) else None,
					compare_ef,
				))
				text = (
					"<table style='border-spacing:0; width:100%;'>"
					f"<tr><td style='vertical-align:top; padding-right:14px;'><b>{primary_label}</b><br>{left_lines}</td>"
					f"<td style='vertical-align:top;'><b>{compare_label}</b><br>{right_lines}</td></tr>"
					"</table>"
				)
				delta = self._format_async_delta_lines(metrics, compare_metrics, ef, compare_ef)
				if delta:
					self.main_delta_readout.setText(" &nbsp;|&nbsp; ".join(delta))
					self.main_delta_readout.setVisible(True)
				else:
					self.main_delta_readout.setText("")
					self.main_delta_readout.setVisible(False)
				self.main_metrics_readout.setText(text)
			else:
				self.main_delta_readout.setText("")
				self.main_delta_readout.setVisible(False)
				self.main_metrics_readout.setText(
					"<br>".join(self._format_async_metrics_lines(metrics, ef_pct, ef))
				)
			# Ayuda consultable (piloto): explicación de PFR/TVmáx al pasar el mouse.
			if isinstance(ef, dict) and ef.get("pfr_text"):
				self.main_metrics_readout.setToolTip(
					explanation_tooltip("pfr") + "\n\n" + explanation_tooltip("tvmax")
				)
			else:
				self.main_metrics_readout.setToolTip("")
		# --- Curvas ya renderizadas ---
		# Asincronía (histograma de fase): solo si hay resultado actual; si no,
		# ejes vacíos para no mostrar el último gráfico que quedó.
		hist_path = os.path.join(self.output_dir, "histograma.png")
		# El histograma de output puede ser un QC combinado crudo/clínico. Para la
		# grilla dual, renderizar SIEMPRE cada etapa desde sus fases clínicas para
		# no mezclar Esfuerzo+Reposo en la celda izquierda.
		if getattr(self, "phase_result", None) is not None and isinstance(metrics, dict):
			self._render_phase_histogram_panel(
				self.curve_hist_view, self.phase_result, metrics,
				"histograma_esfuerzo_panel.png", "Histograma de fase · Esfuerzo",
			)
		else:
			self._render_empty_curve(self.curve_hist_view, "Fase (°)", "Frecuencia", "_empty_hist.png")
		# Segunda columna: resultados/curvas de la etapa opuesta. Usar primero el
		# bundle ya renderizado; fallback a DualSession para no esperar a HQ.
		comp_bundle = getattr(self, "compare_bundle", None)
		comp_metrics_curve = getattr(self, "compare_metrics", None)
		comp_ef_curve = getattr(self, "compare_ef", None)
		try:
			dual_rest = self._dual_session().stage("rest")
			dual_stress = self._dual_session().stage("stress")
			dual_other_curve = dual_stress if dual_rest.cut_study is self.study else dual_rest
			if comp_metrics_curve is None and dual_other_curve.metrics is not None:
				comp_metrics_curve = dual_other_curve.metrics
				comp_ef_curve = dual_other_curve.ef
				comp_phase_curve_fallback = dual_other_curve.phase
			else:
				comp_phase_curve_fallback = None
		except Exception:
			comp_phase_curve_fallback = None
		if comp_metrics_curve is None:
			try:
				active_stage = str(getattr(self, "_cine_crudo_recon_stage", "stress") or "stress")
				other = self._dual_session().stage("rest" if active_stage == "stress" else "stress")
				comp_metrics_curve = other.metrics
				comp_ef_curve = other.ef
			except Exception:
				pass
		if comp_metrics_curve is not None:
			# El compare_output puede llegar vacío hasta que termine HQ. Para que
			# nunca haya panel oscuro, renderizar el histograma de Reposo desde la
			# fase/métricas en memoria (misma fuente que la curva FEVI).
			comp_phase_curve = comp_bundle.get("phase_result") if isinstance(comp_bundle, dict) else None
			if comp_phase_curve is None:
				try:
					comp_phase_curve = comp_phase_curve_fallback
					if comp_phase_curve is None:
						active_stage = str(getattr(self, "_cine_crudo_recon_stage", "stress") or "stress")
						comp_phase_curve = self._dual_session().stage("rest" if active_stage == "stress" else "stress").phase
				except Exception:
					comp_phase_curve = None
			if not self._render_phase_histogram_panel(
				self.curve_hist_compare_view, comp_phase_curve, comp_metrics_curve,
				"histograma_reposo_panel.png", "Histograma de fase · Reposo",
			):
				self._render_empty_curve(self.curve_hist_compare_view, "Fase (°)", "Frecuencia", "_empty_hist_reposo.png")
			if not self._render_fevi_curve_panel(
				self.curve_fevi_compare_view, comp_ef_curve,
				output_filename="curva_fevi_reposo_panel.png", title_suffix=" · Reposo",
			):
				self._render_empty_curve(self.curve_fevi_compare_view, "Gate", "Volumen (mL)", "_empty_fevi_reposo.png")
		else:
			self._render_empty_curve(self.curve_hist_compare_view, "Fase (°)", "Frecuencia", "_empty_hist_reposo.png")
			self._render_empty_curve(self.curve_fevi_compare_view, "Gate", "Volumen (mL)", "_empty_fevi_reposo.png")
		# FEVI (volumen/gate): curva autónoma desde el EF ya calculado (funciona en
		# básico y avanzado, sin depender del pipeline pesado). Si no hay datos
		# suficientes, ejes vacíos.
		if not self._render_fevi_curve_panel(self.curve_fevi_view, ef, title_suffix=" · Esfuerzo"):
			self._render_empty_curve(self.curve_fevi_view, "Gate", "Volumen (mL)", "_empty_fevi.png")
		# Título de la 2da. etapa sobre el cine_compare (reposo/esfuerzo según la
		# 1ra. cargada); se actualiza cada vez que se reprocesa/carga.
		self._refresh_cine_compare_title()

	def _tab_name_from_title(self, title: str) -> str | None:
		for name, tab_title in self._tab_titles.items():
			if tab_title == str(title):
				return name
		return None

	def _active_tab_name(self) -> str | None:
		idx = int(self.tabs.currentIndex()) if self.tabs is not None else -1
		if idx < 0:
			return None
		return self._tab_name_from_title(self.tabs.tabText(idx))

	def _default_preview_tabs(self) -> set[str]:
		tabs = {"slices_fase", "polar_combo", "delta_combo", "histograma", "comparacion_stress_rest", "ungated"}
		active = self._active_tab_name()
		if active:
			tabs.add(active)
		return tabs

	def _tab_required_outputs(self, name: str) -> tuple[str, ...]:
		mapping = {
			"slices_fase": ("slices_fase.png",),
			"polar_combo": ("polar_combo.png",),
			"delta_combo": ("delta_combo.png",),
			"histograma": ("histograma.png",),
			"comparacion_stress_rest": ("comparacion_stress_rest.png",),
			"comparacion_ejes": ("comparacion_ejes.png",),
			"curva_fevi": ("curva_fevi.png",),
			"panel_funcional_gated": ("panel_funcional_gated.png",),
			"bullseye_directo": ("bullseye_directo.png",),
			"guia_fase_vi": ("guia_fase_vi.png",),
			"polar_perfusion_directa": ("polar_perfusion_directa.png",),
		}
		return mapping.get(str(name), tuple())

	def _is_tab_render_ready(self, name: str) -> bool:
		files = self._tab_required_outputs(name)
		if not files:
			return True
		return all(os.path.exists(os.path.join(self.output_dir, fname)) for fname in files)

	def _request_lazy_tab_render(self, tab_name: str, reason: str = ""):
		if self.study is None or self.seg is None or self.phase_result is None:
			return
		heavy_tabs = {
			"comparacion_ejes",
			"curva_fevi",
			"panel_funcional_gated",
			"bullseye_directo",
			"guia_fase_vi",
			"polar_perfusion_directa",
		}
		tab_name = str(tab_name or "")
		if tab_name not in heavy_tabs:
			return
		if self._is_tab_render_ready(tab_name):
			self._load_preview(tab_name)
			return
		self._lazy_render_pending_tabs.add(tab_name)
		if self._deferred_hq_running:
			self._log(f"Lazy render encolado ({tab_name}) mientras HQ está en curso.")
			return
		pending = set(self._lazy_render_pending_tabs)
		self._lazy_render_pending_tabs.clear()
		if not pending:
			return
		msg = f"Render bajo demanda: {', '.join(sorted(pending))}"
		if reason:
			msg += f" ({reason})"
		self._set_progress(86, msg)
		try:
			if self.compare_bundle is not None:
				self._write_outputs_for_bundle(self.compare_bundle, self.compare_output_dir, target_tabs=pending)
			self._write_outputs(target_tabs=pending)
			if self.compare_bundle is not None:
				left_label, right_label = self._dual_compare_labels()
				self._compose_dual_tab_images(left_label, right_label, target_tabs=pending)
			self._load_previews_selected(pending)
		except Exception as exc:
			self._log(f"[WARN] Lazy render falló ({', '.join(sorted(pending))}): {exc}")

	def _log(self, message: str):
		self.log_box.append(message)

	def _restore_window_layout(self):
		geom = self._ui_settings.value("window_geometry", None)
		if geom is not None:
			self.restoreGeometry(geom)
			# Tras restaurar, clamar a la pantalla y si aun así no entra, abrir
			# maximizada. Sin esto, la ventana abre estirada fuera de la pantalla y
			# las filas del cine se solapan hasta que el usuario maximiza a mano.
			QTimer.singleShot(0, self._clamp_or_maximize)
		# Clave versionada v2: el sidebar pasó a arrancar en su ancho mínimo (~300px)
		# para no estirar las filas de botones, así que se descarta el ancho guardado viejo.
		main_state = self._ui_settings.value("main_splitter_state_v2", None)
		if main_state is not None:
			self.main_splitter.restoreState(main_state)
		# Clave versionada: el layout cambió (75/25 + panel + cine compacto), así que
		# se ignora el estado guardado con la clave vieja para aplicar el nuevo ratio.
		right_state = self._ui_settings.value("right_splitter_state_v3", None)
		if right_state is not None:
			self.right_splitter.restoreState(right_state)
		# Clave versionada v7: nuevo reparto 25/75 entre cine y bloque clínico.
		# Se ignora el estado v6 para que el cambio se aplique desde el arranque.
		bottom_state = self._ui_settings.value("bottom_hsplit_state_v7", None)
		if bottom_state is not None:
			self.bottom_hsplit.restoreState(bottom_state)

	def _second_phase_label(self) -> str:
		"""Etiqueta de la 2da. Fase = la OTRA etapa respecto de la cargada.
		La fase se lee de la metadata/nombre del archivo (REST→Reposo,
		STRESS→Esfuerzo). Si la 1ra. es Reposo, la 2da. es Esfuerzo y viceversa.
		Si no se puede determinar, se usa la genérica '2da. Fase'."""
		try:
			phase = self._study_context().get("phase", "")
		except Exception:
			phase = ""
		if phase == "Reposo":
			return "Esfuerzo"
		if phase == "Esfuerzo":
			return "Reposo"
		return "2da. Fase"

	def _refresh_cine_compare_title(self) -> None:
		"""Actualiza los rótulos de fase de ambos cines: la 2da. etapa en el
		cine_compare (reposo/esfuerzo según la 1ra. cargada) y la 1ra. en el cine."""
		second = self._second_phase_label()
		if getattr(self, "cine_compare", None) is not None:
			self.cine_compare.set_phase_title(second)
			self.cine_compare.set_image_overlay_label(second, color=self._phase_overlay_color(second))
		if getattr(self, "cine", None) is not None:
			try:
				phase = self._study_context().get("phase", "")
			except Exception:
				phase = ""
			first = phase if phase in ("Reposo", "Esfuerzo") else "1ra. Fase"
			self.cine.set_phase_title(first)
			self.cine.set_image_overlay_label(first, color=self._phase_overlay_color(first))

	@staticmethod
	def _phase_overlay_color(label: str) -> str:
		"""Color del rótulo sobre la imagen según la etapa, para distinguirlas
		de un vistazo: Esfuerzo en naranja, Reposo en celeste."""
		if label == "Esfuerzo":
			return "#ff9a5a"
		if label == "Reposo":
			return "#5ad1ff"
		return "#facc15"

	# ---------------------------------------------------------------------
	# Grupos de controles en menú desplegable dentro de pestañas (cine_crudo)
	# ---------------------------------------------------------------------
	#
	# Se probó envolver las filas de toolbar en CollapsibleSection, pero al
	# expandir/colapsar cambiaba el tamaño de la imagen (empujaba el layout).
	# También se probó un menú desplegable (QMenu), pero no se podía mover ni
	# reacomodar. En su lugar, se agrupan en una `FloatingToolbar`: una barra
	# propia, movible con el mouse y con orientación horizontal/vertical
	# configurable, que el usuario deja donde le resulte más cómoda. El botón
	# que la abre ocupa siempre el mismo lugar, así que nunca cambia el
	# tamaño del resto de la pestaña.

	def _build_toolbar_group_menu(self, title: str, hlayouts: list, key: str, tooltip: str = "", side_widget: QWidget | None = None) -> QToolButton:
		"""Agrupa una o más QHBoxLayout de filas de controles en una
		`FloatingToolbar`, para bajar el ruido visual de paneles con muchas
		filas de toolbar (p.ej. la pestaña cine_crudo) sin mover el resto del
		layout. Si se provee `side_widget`, se coloca a la derecha con rowspan."""
		toolbar = FloatingToolbar(title, key=key, side_widget=side_widget, parent=self)
		for hl in hlayouts:
			toolbar.add_layout(hl)
		btn = QToolButton()
		btn.setText(title)
		btn.clicked.connect(lambda: toolbar.toggle_near(btn))
		if tooltip:
			btn.setToolTip(tooltip)
		# Guardar (barra, botón) por clave para poder abrir/cerrar por código
		# (p.ej. tras Aplicar/Rechazar la corrección de movimiento).
		if not hasattr(self, "_toolbar_group_menus"):
			self._toolbar_group_menus = {}
		self._toolbar_group_menus[key] = (toolbar, btn)
		return btn

	# ---------------------------------------------------------------------
	# Sidebar colapsable
	# ---------------------------------------------------------------------
	#
	# El sidebar se construye con una lista de QGroupBox. En vez de reescribir
	# esa construcción, al final se recorre el layout y cada QGroupBox se
	# reemplaza in-place por un CollapsibleSection que lo envuelve. Así el
	# título de cada caja pasa a ser un botón expandir/colapsar y no hace falta
	# tocar ninguna de las cajas existentes ni las futuras.

	#: Secciones que arrancan abiertas la primera vez (flujo de trabajo básico).
	#: El resto arranca colapsado para bajar el ruido visual. Después de la
	#: primera vez manda lo que el usuario haya dejado guardado en QSettings.
	SIDEBAR_DEFAULT_EXPANDED = ("estudio", "acciones", "resumen")

	def _install_collapsible_sidebar_sections(self):
		"""Convierte cada QGroupBox del sidebar en una sección colapsable.

		Reemplaza cada caja por un `CollapsibleSection` en la misma posición y
		agrega arriba de todo una fila con "Expandir todo" / "Colapsar todo".
		Guarda el mapa `clave -> sección` en `self._sidebar_sections` para poder
		abrir/cerrar por código y para persistir el estado.
		"""
		self._sidebar_sections: dict[str, CollapsibleSection] = {}
		self._sidebar_section_by_content: dict[QWidget, CollapsibleSection] = {}
		layout = self._sidebar_layout

		boxes = []
		for i in range(layout.count()):
			item = layout.itemAt(i)
			widget = item.widget() if item is not None else None
			if isinstance(widget, QGroupBox) and widget.title().strip():
				boxes.append(widget)
		if not boxes:
			return

		first_index = layout.indexOf(boxes[0])
		for box in boxes:
			index = layout.indexOf(box)
			if index < 0:
				continue
			key = slugify_section_key(box.title())
			expanded = key in self.SIDEBAR_DEFAULT_EXPANDED
			# takeAt saca la caja del layout sin destruirla; el constructor de
			# CollapsibleSection la reparenta dentro de la sección.
			layout.takeAt(index)
			section = CollapsibleSection.from_group_box(box, key=key, expanded=expanded)
			section.toggled.connect(
				lambda checked, k=key: self._on_sidebar_section_toggled(k, checked)
			)
			layout.insertWidget(index, section)
			self._sidebar_sections[key] = section
			self._sidebar_section_by_content[box] = section

		layout.insertWidget(first_index, self._build_sidebar_sections_toolbar())

	def _build_sidebar_sections_toolbar(self) -> QWidget:
		"""Fila de atajos para abrir/cerrar todas las secciones del sidebar."""
		bar = QWidget()
		row = QHBoxLayout(bar)
		row.setContentsMargins(0, 2, 0, 2)
		row.setSpacing(4)
		hint = QLabel("Secciones")
		hint.setStyleSheet("color:#4b5563; font-weight:600;")
		hint.setToolTip(
			"Cada título de abajo es un botón: click para expandir o colapsar esa sección.\n"
			"El estado se recuerda entre sesiones."
		)
		row.addWidget(hint)
		row.addStretch(1)
		self.expand_all_sections_btn = QToolButton()
		self.expand_all_sections_btn.setText("Expandir todo")
		self.expand_all_sections_btn.setToolTip("Abre todas las secciones del panel lateral.")
		self.expand_all_sections_btn.clicked.connect(lambda: self._set_all_sidebar_sections(True))
		row.addWidget(self.expand_all_sections_btn)
		self.collapse_all_sections_btn = QToolButton()
		self.collapse_all_sections_btn.setText("Colapsar todo")
		self.collapse_all_sections_btn.setToolTip(
			"Cierra todas las secciones y deja solo los títulos. Útil para trabajar con el visor a pantalla ancha."
		)
		self.collapse_all_sections_btn.clicked.connect(lambda: self._set_all_sidebar_sections(False))
		row.addWidget(self.collapse_all_sections_btn)
		return bar

	def _set_all_sidebar_sections(self, expanded: bool):
		"""Expande o colapsa todas las secciones de una sola pasada.

		Se desactivan los updates del sidebar durante el bucle para que Qt haga
		un solo relayout/repintado en vez de uno por sección.
		"""
		sections = getattr(self, "_sidebar_sections", {})
		if not sections:
			return
		container = self._sidebar_layout.parentWidget()
		if container is not None:
			container.setUpdatesEnabled(False)
		try:
			for section in sections.values():
				section.set_expanded(bool(expanded))
		finally:
			if container is not None:
				container.setUpdatesEnabled(True)

	def _sidebar_section(self, key_or_widget) -> CollapsibleSection | None:
		"""Devuelve la sección por clave (`"log"`) o por su widget de contenido."""
		if isinstance(key_or_widget, QWidget):
			return getattr(self, "_sidebar_section_by_content", {}).get(key_or_widget)
		return getattr(self, "_sidebar_sections", {}).get(str(key_or_widget))

	def _on_sidebar_section_toggled(self, key: str, expanded: bool):
		"""Persiste el estado de una sección apenas el usuario la abre/cierra."""
		settings = getattr(self, "_ui_settings", None)
		if settings is None:
			return
		settings.setValue(f"sidebar/section_{key}", bool(expanded))

	def _restore_sidebar_sections_state(self):
		"""Restaura desde QSettings qué secciones quedaron abiertas o cerradas."""
		settings = getattr(self, "_ui_settings", None)
		sections = getattr(self, "_sidebar_sections", {})
		if settings is None or not sections:
			return
		container = self._sidebar_layout.parentWidget()
		if container is not None:
			container.setUpdatesEnabled(False)
		try:
			for key, section in sections.items():
				stored = settings.value(f"sidebar/section_{key}", None)
				if stored is None:
					continue
				section.set_expanded(bool(stored in (True, "true", "True", 1, "1")))
		finally:
			if container is not None:
				container.setUpdatesEnabled(True)

	def _save_sidebar_sections_state(self):
		"""Vuelca el estado actual de todas las secciones a QSettings."""
		settings = getattr(self, "_ui_settings", None)
		sections = getattr(self, "_sidebar_sections", {})
		if settings is None or not sections:
			return
		for key, section in sections.items():
			settings.setValue(f"sidebar/section_{key}", bool(section.is_expanded()))

	def _restore_fevi_settings(self):
		"""Recupera el método de FEVI y los parámetros ECTb de la sesión anterior."""
		settings = getattr(self, "_ui_settings", None)
		self._fevi_method = self.FEVI_METHOD_ECTB
		self._ectb_config = ECTbLVConfig()
		self._fevi_regression = None
		if settings is None:
			return
		method = str(settings.value("fevi/method", self.FEVI_METHOD_ECTB) or self.FEVI_METHOD_ECTB)
		if method in self.FEVI_METHOD_LABELS:
			self._fevi_method = method
		regression = str(settings.value("fevi/regression", "") or "")
		self._fevi_regression = regression if regression in EF_REGRESSIONS else None
		defaults = ECTbLVConfig()
		try:
			self._ectb_config = ECTbLVConfig(
				ed_wall_thickness_mm=float(settings.value("fevi/ectb_wall_mm", defaults.ed_wall_thickness_mm)),
				n_angles=int(settings.value("fevi/ectb_angles", defaults.n_angles)),
				radial_oversample=float(settings.value("fevi/ectb_oversample", defaults.radial_oversample)),
				median_kernel_large=int(settings.value("fevi/ectb_median_large", defaults.median_kernel_large)),
				median_kernel_small=int(settings.value("fevi/ectb_median_small", defaults.median_kernel_small)),
				use_thickening=str(settings.value("fevi/ectb_thickening", "true")).lower() in ("true", "1"),
				use_valve_plane=str(settings.value("fevi/ectb_valve_plane", "true")).lower() in ("true", "1"),
				valve_septal_offset_mm=float(settings.value("fevi/ectb_valve_offset_mm", defaults.valve_septal_offset_mm)),
				septal_angle_deg=float(settings.value("fevi/ectb_septal_angle", defaults.septal_angle_deg)),
			)
		except Exception:
			self._ectb_config = defaults

	def _save_fevi_settings(self):
		"""Persiste método de FEVI y parámetros ECTb."""
		settings = getattr(self, "_ui_settings", None)
		if settings is None:
			return
		cfg = self.ectb_config()
		settings.setValue("fevi/method", self.fevi_method())
		settings.setValue("fevi/regression", self.fevi_regression() or "")
		settings.setValue("fevi/ectb_wall_mm", float(cfg.ed_wall_thickness_mm))
		settings.setValue("fevi/ectb_angles", int(cfg.n_angles))
		settings.setValue("fevi/ectb_oversample", float(cfg.radial_oversample))
		settings.setValue("fevi/ectb_median_large", int(cfg.median_kernel_large))
		settings.setValue("fevi/ectb_median_small", int(cfg.median_kernel_small))
		settings.setValue("fevi/ectb_thickening", bool(cfg.use_thickening))
		settings.setValue("fevi/ectb_valve_plane", bool(cfg.use_valve_plane))
		settings.setValue("fevi/ectb_valve_offset_mm", float(cfg.valve_septal_offset_mm))
		settings.setValue("fevi/ectb_septal_angle", float(cfg.septal_angle_deg))

	def _save_window_layout(self):
		self._ui_settings.setValue("window_geometry", self.saveGeometry())
		self._ui_settings.setValue("main_splitter_state_v2", self.main_splitter.saveState())
		self._ui_settings.setValue("right_splitter_state_v3", self.right_splitter.saveState())
		self._ui_settings.setValue("bottom_hsplit_state_v7", self.bottom_hsplit.saveState())
		self._save_sidebar_sections_state()
		self._save_fevi_settings()
		self._ui_settings.sync()

	def _load_global_ui_preferences(self):
		self._ui_show_helpers = bool(self._ui_settings.value("ui/show_helpers", True, type=bool))
		self._ui_enable_tooltips = bool(self._ui_settings.value("ui/enable_tooltips", True, type=bool))
		self._ui_compact_controls = bool(self._ui_settings.value("ui/compact_controls", False, type=bool))
		self._dual_pipeline_auto_enabled = bool(self._ui_settings.value("pipeline/dual_auto_enabled", True, type=bool))
		src = str(self._ui_settings.value("analysis/perfusion_source", self.PERFUSION_SOURCE_ED))
		self._perfusion_source = src if src in self.PERFUSION_SOURCE_LABELS else self.PERFUSION_SOURCE_ED

	def _save_global_ui_preferences(self):
		self._ui_settings.setValue("ui/show_helpers", bool(self._ui_show_helpers))
		self._ui_settings.setValue("ui/enable_tooltips", bool(self._ui_enable_tooltips))
		self._ui_settings.setValue("ui/compact_controls", bool(self._ui_compact_controls))
		self._ui_settings.setValue("pipeline/dual_auto_enabled", bool(getattr(self, "_dual_pipeline_auto_enabled", True)))
		self._ui_settings.setValue("analysis/perfusion_source", self.perfusion_source())
		self._ui_settings.sync()

	def _capture_global_tooltips(self):
		for w in self.findChildren(QWidget):
			tip = w.toolTip()
			if tip:
				self._tooltips_cache_main[w] = tip

	def _apply_global_tooltips(self):
		enabled = bool(self._ui_enable_tooltips)
		for w, tip in list(self._tooltips_cache_main.items()):
			if w is None:
				continue
			w.setToolTip(tip if enabled else "")
		for i in range(self.tabs.count()):
			title = self.tabs.tabText(i)
			tip = self._tab_tooltips.get({v: k for k, v in self._tab_titles.items()}.get(title, ""), "")
			self.tabs.setTabToolTip(i, tip if enabled else "")

	def _apply_global_ui_preferences(self):
		if hasattr(self, "helper_box"):
			# Si la caja quedó envuelta en una sección colapsable hay que ocultar
			# la sección entera; si no, quedaría el encabezado suelto sin contenido.
			target = self._sidebar_section(self.helper_box) or self.helper_box
			target.setVisible(bool(self._ui_show_helpers))
		self.cine.set_ui_preferences(
			show_helpers=bool(self._ui_show_helpers),
			enable_tooltips=bool(self._ui_enable_tooltips),
			compact_controls=bool(self._ui_compact_controls),
		)
		self.cine_compare.set_ui_preferences(
			show_helpers=bool(self._ui_show_helpers),
			enable_tooltips=bool(self._ui_enable_tooltips),
			compact_controls=bool(self._ui_compact_controls),
		)
		self._apply_global_tooltips()

	def open_ectb_window(self):
		"""Abre (o trae al frente) la ventana de cuantificación ECTb.

		La ventana se crea una sola vez y se reutiliza, así conserva los
		parámetros que el usuario haya ajustado. El import es diferido para no
		pagar el costo de construirla en el arranque de la aplicación.
		"""
		window = getattr(self, "_ectb_window", None)
		if window is None:
			from ui.ectb_window import ECTbWindow

			window = ECTbWindow(self)
			self._ectb_window = window
		window.show()
		window.raise_()
		window.activateWindow()
		window.recompute()

	def _refresh_ectb_window(self):
		"""Recalcula la ventana ECTb si está abierta (tras reprocesar el estudio)."""
		window = getattr(self, "_ectb_window", None)
		if window is not None and window.isVisible():
			window.recompute()

	def _sync_ectb_window_controls(self):
		"""Vuelca la configuración vigente a los controles de la ventana ECTb."""
		window = getattr(self, "_ectb_window", None)
		if window is not None:
			window.sync_from_main()

	def open_preparacion_window(self):
		"""Abre (o trae al frente) la ventana de preparación dual (Esfuerzo/Reposo).

		Ventana no modal, en vivo/en memoria, con MIP rotatorio por etapa. Reutiliza
		el flujo probado de corrección de movimiento / reconstrucción / reorientación.
		"""
		window = getattr(self, "_preparacion_window", None)
		if window is None:
			from ui.preparacion_window import PreparacionWindow

			window = PreparacionWindow(self)
			self._preparacion_window = window
		window.show()
		window.raise_()
		window.activateWindow()
		# Fase 1 fusión: la ventana aloja el panel cine_crudo completo (motor real).
		if hasattr(window, "attach_cine_crudo_panel"):
			window.attach_cine_crudo_panel()
		window.refresh()

	def take_cine_crudo_panel(self):
		"""Saca la pestaña cine_crudo del QTabWidget para alojarla en otra ventana
		(ventana de Preparación). Reversible con restore_cine_crudo_panel()."""
		widget = self._tab_widgets.get("cine_crudo")
		if widget is None:
			return None
		self._cine_crudo_reparented = True
		idx = self.tabs.indexOf(widget)
		if idx >= 0:
			self.tabs.removeTab(idx)
		return widget

	def restore_cine_crudo_panel(self):
		"""Devuelve la pestaña cine_crudo al QTabWidget (tras cerrar la ventana host)."""
		if not getattr(self, "_cine_crudo_reparented", False):
			return
		self._cine_crudo_reparented = False
		widget = self._tab_widgets.get("cine_crudo")
		if widget is not None:
			widget.setParent(None)
		self._rebuild_tabs_for_mode()

	def _refresh_preparacion_window(self):
		"""Refresca el MIP de la ventana de preparación si está abierta."""
		window = getattr(self, "_preparacion_window", None)
		if window is not None and window.isVisible():
			window.refresh()

	def _prep_mip_source_for_stage(self, stage: str):
		"""Fuente para el MIP de una etapa: ('vol', vol3D, status) o ('proj', proj, status).

		- Reconstruido (o cortes SA cargados) → volumen 3D para MIP rotatorio.
		- Crudo sin reconstruir → proyecciones (A,H,W) como vista giratoria natural.
		Devuelve None si no hay estudio para esa etapa.
		"""
		if stage == "rest":
			study = self._secondary_cine_crudo_study()
		else:
			study = getattr(self, "cine_crudo_raw_study_for_recon", None) or self.study
		if study is None:
			return None
		# 1) Volumen reconstruido, si la reconstrucción vigente es de esta etapa.
		rr = getattr(self, "cine_crudo_recon_result", None)
		if rr is not None and str(getattr(self, "_cine_crudo_recon_stage", "stress")) == stage:
			vol = np.asarray(getattr(rr, "ungated_volume", None), dtype=np.float64)
			if vol.ndim == 3 and vol.size:
				return ("vol", vol, "Reconstruido · MIP rotatorio")
		cube = getattr(study, "cube", None)
		if cube is None:
			return None
		cube = np.asarray(cube, dtype=np.float64)
		if cube.size == 0:
			return None
		reconstructed = bool(getattr(study, "reconstructed", True))
		if not reconstructed:
			proj = cube.sum(axis=0) if cube.ndim == 4 else cube  # (gates,A,H,W)→(A,H,W)
			return ("proj", np.asarray(proj, dtype=np.float64), "Crudo · proyecciones giratorias")
		vol = cube.mean(axis=0) if cube.ndim == 4 else cube  # (gates,Z,H,W)→(Z,H,W)
		return ("vol", np.asarray(vol, dtype=np.float64), "Cortes SA · MIP rotatorio")

	def open_gqc_window(self):
		"""Abre (o trae al frente) el panel de control de calidad del gating."""
		window = getattr(self, "_gqc_window", None)
		if window is None:
			from ui.gqc_window import GQCWindow

			window = GQCWindow(self)
			self._gqc_window = window
		window.show()
		window.raise_()
		window.activateWindow()
		window.recompute()

	def open_asynchrony_review_window(self):
		"""Abre una vista separada de inspección visual de ROIs y bordes."""
		window = getattr(self, "_asynchrony_review_window", None)
		if window is None:
			from ui.asynchrony_review_window import AsynchronyReviewWindow

			window = AsynchronyReviewWindow(self)
			self._asynchrony_review_window = window
		window.show()
		window.raise_()
		window.activateWindow()
		window.sync_from_main()
		return window

	def _polar_perfusion_map_for_3d(self):
		"""Construye el mapa polar numérico para 3D aunque el render avanzado esté omitido."""
		cache = getattr(self, "_polar_perf_cart_cache", None) or {}
		cached = cache.get("polar_map")
		if cached is not None and np.asarray(cached).size:
			return np.asarray(cached, dtype=np.float64)
		if self.study is None or self.seg is None:
			return None
		cube = np.asarray(getattr(self.study, "cube", None), dtype=np.float64)
		mask = np.asarray(getattr(self.seg, "mask", None), dtype=bool)
		centers = np.asarray(getattr(self.seg, "center_per_slice", None), dtype=np.float64)
		if cube.ndim != 4 or mask.ndim != 3 or cube.shape[1:] != mask.shape:
			return None
		if centers.shape != (mask.shape[0], 2):
			return None

		ungated = cube.sum(axis=0)
		valid_slices = np.where(mask.reshape(mask.shape[0], -1).any(axis=1))[0]
		profiles = []
		for s in valid_slices:
			mask_s = mask[int(s)]
			cy, cx = centers[int(s)]
			if not (np.isfinite(cy) and np.isfinite(cx)):
				continue
			ys, xs = np.nonzero(mask_s)
			if ys.size == 0:
				continue
			vals = ungated[int(s), ys, xs]
			ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360.0) % 360.0
			bins = np.floor(ang).astype(np.int32) % 360
			profile = np.full(360, np.nan, dtype=np.float64)
			for b in range(360):
				vb = vals[bins == b]
				if vb.size:
					profile[b] = float(np.percentile(vb, 70))
			finite = np.isfinite(profile)
			if not finite.any():
				continue
			if not finite.all():
				x = np.arange(360)
				xv = x[finite]
				yv = profile[finite]
				profile[~finite] = np.interp(
					x[~finite], np.concatenate([xv - 360, xv, xv + 360]),
					np.concatenate([yv, yv, yv]),
				)
			profiles.append(profile)
		if len(profiles) < 2:
			return None

		profiles_arr = np.asarray(profiles, dtype=np.float64)
		nr, nt = 220, 360
		polar_map = np.empty((nr, nt), dtype=np.float64)
		for ir in range(nr):
			t = (ir / max(1, nr - 1)) * (profiles_arr.shape[0] - 1)
			i0 = int(np.floor(t))
			i1 = min(i0 + 1, profiles_arr.shape[0] - 1)
			a = float(t - i0)
			polar_map[ir] = (1.0 - a) * profiles_arr[i0] + a * profiles_arr[i1]
		mx = float(np.nanmax(polar_map))
		if not np.isfinite(mx) or mx <= 0.0:
			return None
		polar_map = np.clip(polar_map / mx, 0.0, 1.0)
		try:
			from scipy.ndimage import gaussian_filter
			strength = float(self.polar_perf_smooth_strength_spin.value())
			polar_map = gaussian_filter(
				polar_map, sigma=(max(0.05, strength), max(0.05, strength * 0.60)),
				mode=("nearest", "wrap"),
			)
		except Exception:
			pass
		rotation = int(getattr(self, "polar_rotation_spin", None).value()) if hasattr(self, "polar_rotation_spin") else 0
		if rotation:
			polar_map = np.roll(polar_map, shift=rotation % 360, axis=1)
		if getattr(self, "_polar_perf_cart_cache", None) is None:
			self._polar_perf_cart_cache = {}
		self._polar_perf_cart_cache["polar_map"] = polar_map
		return polar_map

	def open_lv_3d_window(self):
		"""Abre el panel 3D del VI (miocardio sólido + malla alambre ECTb).

		Necesita: estudio gated procesado (cube reorientado a eje corto) + FEVI
		ECTb calculado (el resultado completo queda cacheado en
		``self._ectb_last_result`` la última vez que corrió ``_estimate_lv_ef_ectb``).
		Si el cache no existe (p.ej. recién cargado y nunca se refrescó el resumen),
		se calcula al vuelo acá.
		"""
		if self.study is None:
			self._log("[3D] No hay estudio cargado.")
			return None
		res = getattr(self, "_ectb_last_result", None)
		if res is None or not getattr(res, "available", False):
			# Calcular FEVI ECTb al vuelo (llena el cache si tiene éxito).
			self._estimate_lv_ef_ectb()
			res = getattr(self, "_ectb_last_result", None)
		if res is None or not getattr(res, "available", False):
			reason = str(getattr(res, "reason", "") or "FEVI ECTb no disponible")
			self._log(f"[3D] No se puede abrir el panel 3D: {reason}")
			QMessageBox.information(
				self, "SINCRO — VI 3D",
				"El panel 3D necesita un estudio gated procesado con FEVI (ECTb) calculado.\n"
				f"Motivo: {reason}",
			)
			return None

		cube = getattr(self.study, "cube", None)
		pixel_spacing = getattr(self.study, "pixel_spacing", None) or (1.0, 1.0)
		slice_mm = float(getattr(self.study, "z_spacing_mm", None) or 1.0)
		spacing = (slice_mm, float(pixel_spacing[0]), float(pixel_spacing[1]))

		# El 3D necesita el volumen RECONSTRUIDO (no el crudo). Si el estudio es
		# crudo y nunca se reconstruyó, no hay volumen para samplear actividad.
		recon = getattr(self, "cine_crudo_recon_result", None)
		if recon is None or getattr(recon, "gated_volume", None) is None:
			self._log("[3D] No hay volumen reconstruido. Reconstruí el estudio primero.")
			QMessageBox.information(
				self, "SINCRO — VI 3D",
				"El panel 3D necesita un estudio RECONSTRUIDO (no crudo).\n"
				"Usá 'Recon raw' o 'Reconstruir selección' primero, después abrí el 3D."
			)
			return None

		from core.lv_mesh import lv_meshes_from_ectb
		try:
			# Shape del volumen SA reorientado (para alinear la cáscara al samplear).
			volume_shape = None
			if recon is not None and getattr(recon, "gated_volume", None) is not None:
				volume_shape = tuple(int(v) for v in recon.gated_volume.shape[1:])
			lv = lv_meshes_from_ectb(
				res, slice_mm, surface="endo",
				seg=self.seg, pixel_mm=(float(pixel_spacing[0]), float(pixel_spacing[1])),
				volume_shape=volume_shape,
			)
		except Exception as exc:
			self._log(f"[3D] No se pudo construir la malla del VI: {exc}")
			return None

		from ui.lv_3d_panel import LV3DDialog
		cmap = str(getattr(self, "cine_crudo_screen_cmap", "odyssey_cool") or "odyssey_cool")
		# Volumen de actividad para mapear sobre la cáscara: el GATED reconstruido
		# (suma de gates = ungated) tiene mejor SNR y la misma geometría de pared.
		myo_volume = None
		if recon is not None and getattr(recon, "gated_volume", None) is not None:
			import numpy as _np
			myo_volume = _np.asarray(recon.gated_volume, dtype=_np.float64).sum(axis=0)

		# Usar el mapa polar REAL ya calculado por el pipeline. No fabricar un
		# mapa sintético al abrir 3D: además de no representar al paciente, ese
		# cálculo podía dejar la ventana pesada o inestable.
		polar_map = self._polar_perfusion_map_for_3d()

		dlg = LV3DDialog(
			self,
			lv_meshes=lv,
			myo_volume=myo_volume,
			spacing_mm=spacing,
			cmap_name=cmap,
			ectb_result=res,
			seg=self.seg,
			pixel_mm=(float(pixel_spacing[0]), float(pixel_spacing[1])),
			polar_map=polar_map,
		)
		dlg.show()
		dlg.raise_()
		dlg.activateWindow()
		self._lv_3d_window = dlg
		return dlg

	def open_amyloid_window(self):
		"""Abre AMYLO vacío; la carga 1 h/3 h se realiza dentro del módulo."""
		from ui.amyloid_window import AmyloidWindow
		dlg = AmyloidWindow(self)
		dlg.show()
		dlg.raise_()
		dlg.activateWindow()
		self._amyloid_window = dlg
		return dlg

	def open_amyloid_spect_window(self):
		"""Abre directamente el flujo AMYLO SPECT / SPECT-CT."""
		from ui.amyloid_spect_panel import AmyloidSpectPanel
		# Sin parent: ventana top-level con entrada propia en la barra de tareas
		dlg = AmyloidSpectPanel(None)
		dlg.show()
		dlg.raise_()
		dlg.activateWindow()
		self._amyloid_spect_window = dlg
		return dlg



	def _amyloid_2d_image(self):
		"""Extrae una imagen 2D del estudio para amiloidosis.

        - Si es planar (1 gate, 1 slice): usa el único frame.
		- Si es SPECT 3D (1 gate, n_slices): usa el MIP maximo.
		- Si es gated: usa la proyección sumada.
		"""
		if self.study is None:
			return None
		cube = np.asarray(self.study.cube, dtype=np.float64)
		if cube.ndim != 4:
			return None
		n_gates, n_slices, rows, cols = cube.shape
		if n_gates == 1 and n_slices == 1:
			return cube[0, 0]
		if n_slices > 1:
			return cube.max(axis=1) if cube.ndim == 4 else None
		if n_gates > 1 and n_slices == 1:
			return cube[:, 0].sum(axis=0)
		return None

	def open_planar_study(self):
		"""Carga una imagen planar estática (no gated) y abre la ventana de amiloidosis.

        Diferente a load_one_or_two_studies: no pasa por process_current, no intenta
        segmentar ni calcular fase/FEVI. Carga directa para imágenes planar.
		"""
		paths = self._select_dicom_paths(
			title="Seleccionar imagen planar (estática)",
			allow_multiple=False,
		)
		if not paths:
			return
		path = paths[0]
		try:
			from core.dicom_loader import load
			study = load(path, verbose=False)
			if study is None:
				QMessageBox.warning(self, "SINCRO", "No se pudo cargar el archivo.")
				return
			img = self._amyloid_2d_image_from_study(study)
			if img is None:
				QMessageBox.information(
					self, "SINCRO — Amyloidosis",
					"El archivo no contiene una imagen planar válida.\n"
					"Se espera una imagen 2D estática (1 frame, no gated)."
				)
				return
			self.study = study
			from ui.amyloid_window import AmyloidWindow
			dlg = AmyloidWindow(self, image=img, study=study)
			dlg.show()
			dlg.raise_()
			dlg.activateWindow()
			self._amyloid_window = dlg
			self._log(f"Planar cargado: {study.series_description or path}")
		except Exception as exc:
			QMessageBox.critical(self, "SINCRO", f"Error al cargar planar:\n{exc}")

	def _amyloid_2d_image_from_study(self, study):
		"""Extrae la imagen 2D de un estudio cargado (sin asumir self.study)."""
		cube = np.asarray(study.cube, dtype=np.float64)
		if cube.ndim != 4:
			return None
		n_gates, n_slices, rows, cols = cube.shape
		if n_gates == 1 and n_slices == 1:
			return cube[0, 0]
		if n_slices > 1:
			return cube.max(axis=1) if cube.ndim == 4 else None
		if n_gates > 1 and n_slices == 1:
			return cube[:, 0].sum(axis=0)
		return None

	def _refresh_gqc_window(self):
		"""Recalcula el panel GQC si está abierto (tras cargar o reprocesar)."""
		window = getattr(self, "_gqc_window", None)
		if window is not None and window.isVisible():
			window.recompute()

	def _refresh_asynchrony_review_window(self):
		"""Sincroniza la vista de inspección si está abierta."""
		window = getattr(self, "_asynchrony_review_window", None)
		if window is not None and window.isVisible():
			window.sync_from_main()

	def open_ui_preferences_dialog(self):
		"""Panel de Configuración de la aplicación.

		Reúne (y seguirá reuniendo, migración gradual) las opciones de
		configuración. Hoy: selección de tema visual (Clásico/Moderno) + las
		preferencias de interfaz que antes estaban en "Config UI".
		"""
		from ui import theme_manager

		dlg = QDialog(self)
		dlg.setWindowTitle("Configuración")
		root = QVBoxLayout(dlg)

		# --- Apariencia: selector de tema ---
		appearance_box = QGroupBox("Apariencia")
		appearance_l = QFormLayout(appearance_box)
		theme_combo = QComboBox()
		for tid, label in theme_manager.AVAILABLE_THEMES:
			theme_combo.addItem(label, tid)
		cur_theme = theme_manager.current_theme()
		idx = theme_combo.findData(cur_theme)
		if idx >= 0:
			theme_combo.setCurrentIndex(idx)
		theme_combo.setToolTip(
			"Clásico: estilo nativo de Qt (como estaba la app).\n"
			"Moderno: hoja de estilo con acento azul GammaSync, tarjetas redondeadas, etc."
		)
		appearance_l.addRow("Tema visual:", theme_combo)
		theme_note = QLabel("El cambio de tema se aplica al instante.")
		theme_note.setWordWrap(True)
		theme_note.setStyleSheet("color:#6b7280; font-size:8pt;")
		appearance_l.addRow(theme_note)
		root.addWidget(appearance_box)

		# --- Interfaz: helpers, tooltips, modo compacto ---
		ui_box = QGroupBox("Interfaz")
		ui_l = QVBoxLayout(ui_box)
		msg = QLabel("Preferencias globales de interfaz para simplificar controles y ayuda visual.")
		msg.setWordWrap(True)
		ui_l.addWidget(msg)
		show_helpers = QCheckBox("Mostrar helpers / ayuda rápida")
		show_helpers.setChecked(bool(self._ui_show_helpers))
		enable_tooltips = QCheckBox("Habilitar tooltips")
		enable_tooltips.setChecked(bool(self._ui_enable_tooltips))
		compact_controls = QCheckBox("Modo compacto (ocultar botones secundarios)")
		compact_controls.setChecked(bool(self._ui_compact_controls))
		dual_pipeline_auto = QCheckBox("Con dos etapas, procesar Ambas automáticamente")
		dual_pipeline_auto.setChecked(bool(getattr(self, "_dual_pipeline_auto_enabled", True)))
		dual_pipeline_auto.setToolTip(
			"Al cargar Esfuerzo y Reposo, bloquea el pipeline en Ambas por defecto: "
			"motion, reconstrucción, reorientación, cortes y fase/FEVI. Elegir "
			"Esfuerzo o Reposo en el selector sigue permitiendo una corrección puntual."
		)
		ui_l.addWidget(show_helpers)
		ui_l.addWidget(enable_tooltips)
		ui_l.addWidget(compact_controls)
		ui_l.addWidget(dual_pipeline_auto)
		root.addWidget(ui_box)

		# --- Análisis: fuente de perfusión segmentaria ---
		analysis_box = QGroupBox("Análisis")
		analysis_l = QFormLayout(analysis_box)
		perfusion_combo = QComboBox()
		for pid, plabel in self.PERFUSION_SOURCE_LABELS.items():
			perfusion_combo.addItem(plabel, pid)
		p_idx = perfusion_combo.findData(self.perfusion_source())
		if p_idx >= 0:
			perfusion_combo.setCurrentIndex(p_idx)
		perfusion_combo.setToolTip(
			"Imagen que alimenta la perfusión y viabilidad por segmento (panel de fase VI).\n"
			"Gate ED (fin de diástole) es el estándar de lectura de perfusión.\n"
			"Media de gates suma cuentas (menos ruido) pero mezcla la fase del ciclo."
		)
		analysis_l.addRow("Fuente de perfusión:", perfusion_combo)
		perfusion_note = QLabel("El gate ED (fin de diástole) es el estándar; cambiarlo reprocesa el panel segmentario.")
		perfusion_note.setWordWrap(True)
		perfusion_note.setStyleSheet("color:#6b7280; font-size:8pt;")
		analysis_l.addRow(perfusion_note)
		root.addWidget(analysis_box)

		# --- Escalas de color de las imágenes del informe ---
		# El grid (16 combos) vive en self._report_cmap_box; se aloja acá dentro de
		# un scroll y se saca del diálogo al cerrar para que no se destruya con él.
		report_cmap_scroll = QScrollArea()
		report_cmap_scroll.setWidgetResizable(True)
		report_cmap_scroll.setMinimumHeight(220)
		self._report_cmap_box.setVisible(True)
		report_cmap_scroll.setWidget(self._report_cmap_box)
		root.addWidget(report_cmap_scroll)
		dlg.finished.connect(
			lambda _=0: (self._report_cmap_box.setParent(None), self._report_cmap_box.setVisible(False))
		)

		# --- Carpeta de salida ---
		output_box = QGroupBox("Carpeta de salida")
		output_l = QVBoxLayout(output_box)
		output_btn = QPushButton("Abrir carpeta de salida")
		output_btn.clicked.connect(self.open_output_folder)
		output_btn.setToolTip("Abre el explorador en la carpeta donde se guardan los PNG, PDF y demás salidas.")
		output_l.addWidget(output_btn)
		root.addWidget(output_box)

		# --- Integridad de informes HTML ---
		integrity_box = QGroupBox("Integridad de informes HTML")
		integrity_l = QFormLayout(integrity_box)
		_saved_mf = int(self._ui_settings.value("integrity/hash_max_files", 200)) if hasattr(self, "_ui_settings") else 200
		_saved_md = int(self._ui_settings.value("integrity/hash_max_days", 90)) if hasattr(self, "_ui_settings") else 90
		hash_max_files = QSpinBox()
		hash_max_files.setRange(0, 9999)
		hash_max_files.setValue(_saved_mf)
		hash_max_files.setSuffix(" archivos")
		hash_max_files.setSpecialValueText("Sin límite")
		hash_max_files.setToolTip("Cantidad máxima de hashes SHA-256 a conservar. Los más antiguos se eliminan automáticamente.")
		integrity_l.addRow("Retención (archivos):", hash_max_files)
		hash_max_days = QSpinBox()
		hash_max_days.setRange(0, 3650)
		hash_max_days.setValue(_saved_md)
		hash_max_days.setSuffix(" días")
		hash_max_days.setSpecialValueText("Sin límite")
		hash_max_days.setToolTip("Días de retención de hashes. Los más antiguos se eliminan automáticamente.")
		integrity_l.addRow("Retención (días):", hash_max_days)
		hash_info = QLabel(
			f"Almacén: report_hashes/ · {self._hash_store_count()} hashes actuales"
		)
		hash_info.setStyleSheet("color:#6b7280; font-size:8pt;")
		integrity_l.addRow(hash_info)
		root.addWidget(integrity_box)

		# Aplicar el tema en vivo al cambiar el combo (aunque se cancele el diálogo,
		# ya queda aplicado el tema elegido; se persiste solo al Aceptar).
		def _on_theme_changed(_idx: int):
			tid = theme_combo.currentData()
			app = QApplication.instance()
			if app is not None and tid:
				theme_manager.apply_theme(app, tid)
		theme_combo.currentIndexChanged.connect(_on_theme_changed)

		buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
		buttons.accepted.connect(dlg.accept)
		buttons.rejected.connect(dlg.reject)
		root.addWidget(buttons)

		prev_theme = cur_theme
		if dlg.exec() != int(QDialog.DialogCode.Accepted):
			# Cancelado: restaurar el tema previo (por si se estuvo previsualizando).
			app = QApplication.instance()
			if app is not None:
				theme_manager.apply_theme(app, prev_theme)
			return

		# Tema
		chosen_theme = theme_combo.currentData() or theme_manager.DEFAULT_THEME
		theme_manager.save_theme(chosen_theme)
		app = QApplication.instance()
		if app is not None:
			theme_manager.apply_theme(app, chosen_theme)

		# Interfaz
		self._ui_show_helpers = bool(show_helpers.isChecked())
		self._ui_enable_tooltips = bool(enable_tooltips.isChecked())
		self._ui_compact_controls = bool(compact_controls.isChecked())
		self._dual_pipeline_auto_enabled = bool(dual_pipeline_auto.isChecked())
		self._apply_global_ui_preferences()
		self._save_global_ui_preferences()

		# Análisis: fuente de perfusión segmentaria
		chosen_source = perfusion_combo.currentData() or self.PERFUSION_SOURCE_ED
		if chosen_source != self.perfusion_source():
			self._perfusion_source = chosen_source
			self._save_global_ui_preferences()
			self._log(f"[PERFUSIÓN] Fuente del panel segmentario: {self.perfusion_source_label()}")
			self._refresh_readonly_results_panel()

		# Integridad: retención de hashes.
		settings = getattr(self, "_ui_settings", None)
		if settings:
			settings.setValue("integrity/hash_max_files", int(hash_max_files.value()))
			settings.setValue("integrity/hash_max_days", int(hash_max_days.value()))
			settings.sync()

		self.statusBar().showMessage("Configuración aplicada")

	def closeEvent(self, event):
		self._save_window_layout()
		if self._last_browse_dir:
			settings = getattr(self, "_ui_settings", None)
			if settings:
				settings.setValue("paths/last_dicom_dir", self._last_browse_dir)
				settings.sync()
		if self._check_unsaved_study():
			event.ignore()
			return
		super().closeEvent(event)

	def _check_unsaved_study(self) -> bool:
		if getattr(self, "study", None) is None or getattr(self, "metrics", None) is None:
			return False
		try:
			pdf_path = os.path.join(self.output_dir, "informe_sincro.pdf")
			if os.path.isfile(pdf_path):
				return False
		except Exception:
			pass
		btn = QMessageBox.question(
			self, "Estudio sin guardar",
			"El estudio actual NO tiene un PDF guardado.\n\n"
			"¿Guardar el informe antes de salir?",
			QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
			QMessageBox.StandardButton.Cancel,
		)
		if btn == QMessageBox.StandardButton.Yes:
			self.save_pdf_as()
			return False
		if btn == QMessageBox.StandardButton.No:
			return False
		return True

	def _load_presets_store(self) -> dict:
		if not os.path.exists(self.presets_path):
			return {}
		try:
			with open(self.presets_path, "r", encoding="utf-8") as fh:
				data = json.load(fh)
				return data if isinstance(data, dict) else {}
		except Exception:
			return {}

	def _save_presets_store(self):
		with open(self.presets_path, "w", encoding="utf-8") as fh:
			json.dump(self._presets_data, fh, ensure_ascii=False, indent=2)

	def _current_patient_key(self) -> str:
		manual = self.preset_patient_edit.text().strip()
		if manual:
			return manual
		if self.study is not None:
			desc = str(getattr(self.study, "study_description", "") or "").strip()
			if desc:
				return desc
		path = self.file_edit.text().strip()
		if path:
			return os.path.splitext(os.path.basename(path))[0]
		return "paciente_sin_nombre"

	def _refresh_presets_for_current_patient(self):
		patient = self._current_patient_key()
		self.preset_combo.blockSignals(True)
		self.preset_combo.clear()
		presets = sorted((self._presets_data.get(patient) or {}).keys())
		self.preset_combo.addItems(presets)
		self.preset_combo.blockSignals(False)

	def _collect_processing_params(self) -> dict:
		active_auto_roi_method = self.cine.auto_roi_method()
		atten_pct, feather_px = self.cine.intestinal_params()
		intestinal_scope = self.cine.intestinal_scope()
		intestinal_apply_enabled = self.cine.intestinal_apply_enabled()
		intestinal_roi_state = self.cine.intestinal_roi_state()
		gate_roi_state = self.cine.gate_roi_state()
		if self.active_cine_source == "compare":
			active_auto_roi_method = self.cine_compare.auto_roi_method()
			atten_pct, feather_px = self.cine_compare.intestinal_params()
			intestinal_scope = self.cine_compare.intestinal_scope()
			intestinal_apply_enabled = self.cine_compare.intestinal_apply_enabled()
			intestinal_roi_state = self.cine_compare.intestinal_roi_state()
			gate_roi_state = self.cine_compare.gate_roi_state()
		return {
			"seg_method": str(self.seg_method.currentText()),
			"roi_source": self.roi_source(),
			"cavity_center": bool(self.cavity_center_enabled()),
			"threshold": float(self.threshold_spin.value()),
			"smooth_sigma": float(self.sigma_spin.value()),
			"harmonics": int(self.harmonics_spin.value()),
			"amp_filter": float(self.phase_threshold_spin.value()),
			"normalize_reference": bool(self.normalize_check.isChecked()),
			"gate_dropout_correction": bool(self.gate_dropout_check.isChecked()),
			"fevi_method": str(self.fevi_method()),
			"ectb_wall_mm": float(self.ectb_config().ed_wall_thickness_mm),
			"ectb_valve_plane": bool(self.ectb_config().use_valve_plane),
			"ectb_valve_offset_mm": float(self.ectb_config().valve_septal_offset_mm),
			"phase_cmap": str(self.cmap_combo.currentText()),
			"visual_style": str(self.visual_style_combo.currentText()),
			"polar_rotation_deg": int(self.polar_rotation_spin.value()),
			"polar_perf_smooth_method": str(self.polar_perf_smooth_method_combo.currentText()),
			"polar_perf_smooth_strength": float(self.polar_perf_smooth_strength_spin.value()),
			"polar_cine_speed_ms": int(self.polar_cine_speed_spin.value()),
			"polar_compare_math_op": str(self.polar_compare_math_combo.currentText()),
			"polar_compare_math_a": str(self.polar_compare_term_a_combo.currentText()),
			"polar_compare_math_b": str(self.polar_compare_term_b_combo.currentText()),
			"export_polar_mp4": bool(self.export_polar_mp4_check.isChecked()),
			"realtime_deferred_render": bool(self.realtime_deferred_render_check.isChecked()),
			"report_cmap_slices": str(self.report_cmap_slices.currentText()),
			"report_cmap_axes": str(self.report_cmap_axes.currentText()),
			"report_cmap_compare": str(self.report_cmap_compare.currentText()),
			"report_cmap_panel_axes": str(self.report_cmap_panel_axes.currentText()),
			"report_cmap_phase": str(self.report_cmap_phase.currentText()),
			"report_cmap_polar_clinico": str(self.report_cmap_polar_clinico.currentText()),
			"report_cmap_amp": str(self.report_cmap_amp.currentText()),
			"report_cmap_bullseye": str(self.report_cmap_bullseye.currentText()),
			"report_cmap_polar_perf": str(self.report_cmap_polar_perf.currentText()),
			"auto_run": bool(self.auto_run_check.isChecked()),
			"auto_center_gain": int(self.auto_center_gain_slider.value()),
			"auto_inner_delta": int(self.auto_inner_delta_slider.value()),
			"auto_outer_delta": int(self.auto_outer_delta_slider.value()),
			"auto_adjust_range": int(self.auto_adjust_range_spin.value()),
			"compare_gate": int(self.compare_gate_spin.value()),
			"compare_slice_pct": int(self.compare_slice_slider.value()),
			"compare_slice_offset_sa": int(self.compare_slice_offset_sa_spin.value()),
			"compare_slice_offset_hla": int(self.compare_slice_offset_hla_spin.value()),
			"compare_slice_offset_vla": int(self.compare_slice_offset_vla_spin.value()),
			"compare_axes_cmap": str(self.compare_axes_cmap_combo.currentText()),
			"compare_axes_cine": bool(self.compare_axes_cine_check.isChecked()),
			"compare_axes_cine_speed_ms": int(self.compare_axes_cine_speed_spin.value()),
			"compare_fast_drag": bool(self.compare_fast_drag_check.isChecked()),
			"compare_window_top": int(self.compare_window_high_slider.value()),
			"compare_window_base": int(self.compare_window_low_slider.value()),
			"compare_show_mask": bool(self.compare_mask_check.isChecked()),
			"compare_axes_zoom_pct": int(self.compare_axes_zoom_slider.value()),
			"compare_axes_use_intestinal_mask": bool(self.compare_axes_intestinal_mask_check.isChecked()),
			"global_intestinal_render": bool(self.global_intestinal_render_check.isChecked()),
			"normal_sex": str(self.normal_sex_combo.currentText()),
			"normal_protocol": str(self.normal_protocol_combo.currentText()),
			"normal_db": str(self.normal_db_combo.currentText()),
			"auto_roi_method": str(active_auto_roi_method),
			"intestinal_attenuation_pct": int(atten_pct),
			"intestinal_feather_px": int(feather_px),
			"intestinal_scope": str(intestinal_scope),
			"intestinal_apply_enabled": bool(intestinal_apply_enabled),
			"intestinal_roi_state": intestinal_roi_state,
			"gate_roi_state": gate_roi_state,
			"ui_show_helpers": bool(self._ui_show_helpers),
			"ui_enable_tooltips": bool(self._ui_enable_tooltips),
			"ui_compact_controls": bool(self._ui_compact_controls),
			"manual_rois_text": self.manual_rois.toPlainText(),
			"ecg_ritmo": str(self.ecg_ritmo_combo.currentText()),
			"ecg_fc": int(self.ecg_fc_spin.value()),
			"ecg_qrs": int(self.ecg_qrs_spin.value()),
			"ecg_qt": int(self.ecg_qt_spin.value()),
			"ecg_bri": bool(self.ecg_bri_check.isChecked()),
			"ecg_brd": bool(self.ecg_brd_check.isChecked()),
			"ecg_marcapasos": bool(self.ecg_marcapasos_check.isChecked()),
			"ecg_observaciones": str(self.ecg_obs_edit.text()),
			"ecg_file_path": str(self.ecg_file_path),
			"cine_crudo_visual_config": self._collect_cine_crudo_visual_config(),
			"updated_at": datetime.now().isoformat(timespec="seconds"),
		}

	def _apply_processing_params(self, params: dict):
		if "seg_method" in params:
			self.seg_method.setCurrentText(str(params["seg_method"]))
		if "roi_source" in params:
			self.set_roi_source(str(params["roi_source"]))
		if "cavity_center" in params:
			# blockSignals: process_current() ya corre al terminar de aplicar el preset.
			self.cavity_center_check.blockSignals(True)
			self.cavity_center_check.setChecked(bool(params["cavity_center"]))
			self.cavity_center_check.blockSignals(False)
			self._propagate_cavity_center()
		if "threshold" in params:
			self.threshold_spin.setValue(float(params["threshold"]))
		if "smooth_sigma" in params:
			self.sigma_spin.setValue(float(params["smooth_sigma"]))
		if "harmonics" in params:
			self.harmonics_spin.setValue(int(params["harmonics"]))
		if "amp_filter" in params:
			self.phase_threshold_spin.setValue(float(params["amp_filter"]))
		if "normalize_reference" in params:
			self.normalize_check.setChecked(bool(params["normalize_reference"]))
		if "gate_dropout_correction" in params:
			# blockSignals: al cargar un preset no queremos disparar un reproceso
			# extra; process_current() ya corre después con todos los parámetros.
			self.gate_dropout_check.blockSignals(True)
			self.gate_dropout_check.setChecked(bool(params["gate_dropout_correction"]))
			self.gate_dropout_check.blockSignals(False)
			self._refresh_gate_dropout_status()
		if "fevi_method" in params:
			# reprocess=False: process_current() ya corre después de aplicar el preset.
			self.set_fevi_method(str(params["fevi_method"]), reprocess=False)
		if "ectb_wall_mm" in params:
			cfg = self.ectb_config()
			cfg.ed_wall_thickness_mm = float(params["ectb_wall_mm"])
			self.set_ectb_config(cfg)
			self._sync_ectb_window_controls()
		if "ectb_valve_plane" in params or "ectb_valve_offset_mm" in params:
			cfg = self.ectb_config()
			if "ectb_valve_plane" in params:
				cfg.use_valve_plane = bool(params["ectb_valve_plane"])
			if "ectb_valve_offset_mm" in params:
				cfg.valve_septal_offset_mm = float(params["ectb_valve_offset_mm"])
			self.set_ectb_config(cfg)
			self._sync_ectb_window_controls()
		if "phase_cmap" in params:
			self.cmap_combo.setCurrentText(str(params["phase_cmap"]))
		if "visual_style" in params:
			style_value = str(params["visual_style"])
			if "like" in style_value.lower():
				style_value = "Clinico"
			self.visual_style_combo.setCurrentText(style_value)
		if "polar_rotation_deg" in params:
			self.polar_rotation_spin.setValue(int(params["polar_rotation_deg"]))
		if "polar_perf_smooth_method" in params:
			self.polar_perf_smooth_method_combo.setCurrentText(str(params["polar_perf_smooth_method"]))
		if "polar_perf_smooth_strength" in params:
			self.polar_perf_smooth_strength_spin.setValue(float(params["polar_perf_smooth_strength"]))
		if "polar_cine_speed_ms" in params:
			self.polar_cine_speed_spin.setValue(int(params["polar_cine_speed_ms"]))
		if "polar_compare_math_op" in params:
			self.polar_compare_math_combo.setCurrentText(str(params["polar_compare_math_op"]))
		if "polar_compare_math_a" in params:
			self.polar_compare_term_a_combo.setCurrentText(str(params["polar_compare_math_a"]))
		if "polar_compare_math_b" in params:
			self.polar_compare_term_b_combo.setCurrentText(str(params["polar_compare_math_b"]))
		if "export_polar_mp4" in params:
			self.export_polar_mp4_check.setChecked(bool(params["export_polar_mp4"]))
		if "realtime_deferred_render" in params:
			self.realtime_deferred_render_check.setChecked(bool(params["realtime_deferred_render"]))
		if "report_cmap_slices" in params:
			self.report_cmap_slices.setCurrentText(str(params["report_cmap_slices"]))
		if "report_cmap_axes" in params:
			self.report_cmap_axes.setCurrentText(str(params["report_cmap_axes"]))
		if "report_cmap_compare" in params:
			self.report_cmap_compare.setCurrentText(str(params["report_cmap_compare"]))
		if "report_cmap_panel_axes" in params:
			self.report_cmap_panel_axes.setCurrentText(str(params["report_cmap_panel_axes"]))
		if "report_cmap_phase" in params:
			self.report_cmap_phase.setCurrentText(str(params["report_cmap_phase"]))
		if "report_cmap_polar_clinico" in params:
			self.report_cmap_polar_clinico.setCurrentText(str(params["report_cmap_polar_clinico"]))
		if "report_cmap_amp" in params:
			self.report_cmap_amp.setCurrentText(str(params["report_cmap_amp"]))
		if "report_cmap_bullseye" in params:
			self.report_cmap_bullseye.setCurrentText(str(params["report_cmap_bullseye"]))
		if "report_cmap_polar_perf" in params:
			self.report_cmap_polar_perf.setCurrentText(str(params["report_cmap_polar_perf"]))
		if "auto_run" in params:
			self.auto_run_check.setChecked(bool(params["auto_run"]))
		if "auto_center_gain" in params:
			self.auto_center_gain_slider.setValue(int(params["auto_center_gain"]))
		if "auto_inner_delta" in params:
			self.auto_inner_delta_slider.setValue(int(params["auto_inner_delta"]))
		if "auto_outer_delta" in params:
			self.auto_outer_delta_slider.setValue(int(params["auto_outer_delta"]))
		if "auto_adjust_range" in params:
			self.auto_adjust_range_spin.setValue(int(params["auto_adjust_range"]))
		if "compare_gate" in params:
			self.compare_gate_spin.setValue(int(params["compare_gate"]))
		if "compare_slice_pct" in params:
			self.compare_slice_slider.setValue(int(params["compare_slice_pct"]))
		if "compare_slice_offset_sa" in params:
			self.compare_slice_offset_sa_spin.setValue(int(params["compare_slice_offset_sa"]))
		if "compare_slice_offset_hla" in params:
			self.compare_slice_offset_hla_spin.setValue(int(params["compare_slice_offset_hla"]))
		if "compare_slice_offset_vla" in params:
			self.compare_slice_offset_vla_spin.setValue(int(params["compare_slice_offset_vla"]))
		if "compare_axes_cmap" in params:
			self.compare_axes_cmap_combo.setCurrentText(str(params["compare_axes_cmap"]))
		if "compare_axes_cine" in params:
			self.compare_axes_cine_check.setChecked(bool(params["compare_axes_cine"]))
		if "compare_axes_cine_speed_ms" in params:
			self.compare_axes_cine_speed_spin.setValue(int(params["compare_axes_cine_speed_ms"]))
		if "compare_fast_drag" in params:
			self.compare_fast_drag_check.setChecked(bool(params["compare_fast_drag"]))
		if "compare_window_top" in params:
			self.compare_window_high_slider.setValue(int(params["compare_window_top"]))
		if "compare_window_base" in params:
			self.compare_window_low_slider.setValue(int(params["compare_window_base"]))
		if "compare_show_mask" in params:
			self.compare_mask_check.setChecked(bool(params["compare_show_mask"]))
		if "compare_axes_zoom_pct" in params:
			self.compare_axes_zoom_slider.setValue(int(params["compare_axes_zoom_pct"]))
		if "compare_axes_use_intestinal_mask" in params:
			self.compare_axes_intestinal_mask_check.setChecked(bool(params["compare_axes_use_intestinal_mask"]))
		if "global_intestinal_render" in params:
			self.global_intestinal_render_check.setChecked(bool(params["global_intestinal_render"]))
		if "normal_sex" in params:
			self.normal_sex_combo.setCurrentText(str(params["normal_sex"]))
		if "normal_protocol" in params:
			self.normal_protocol_combo.setCurrentText(str(params["normal_protocol"]))
		if "normal_db" in params:
			self.normal_db_combo.setCurrentText(str(params["normal_db"]))
		if "auto_roi_method" in params:
			method = str(params["auto_roi_method"])
			self.cine.set_auto_roi_method(method)
			self.cine_compare.set_auto_roi_method(method)
		if "intestinal_attenuation_pct" in params or "intestinal_feather_px" in params:
			atten_pct = int(params.get("intestinal_attenuation_pct", self.cine.intestinal_params()[0]))
			feather_px = int(params.get("intestinal_feather_px", self.cine.intestinal_params()[1]))
			self.cine.set_intestinal_params(atten_pct, feather_px)
			self.cine_compare.set_intestinal_params(atten_pct, feather_px)
		if "intestinal_scope" in params:
			scope = str(params.get("intestinal_scope", "slice"))
			self.cine.set_intestinal_scope(scope)
			self.cine_compare.set_intestinal_scope(scope)
		if "intestinal_apply_enabled" in params:
			apply_on = bool(params.get("intestinal_apply_enabled", False))
			self.cine.set_intestinal_apply_enabled(apply_on)
			self.cine_compare.set_intestinal_apply_enabled(apply_on)
		if "intestinal_roi_state" in params:
			state = params.get("intestinal_roi_state")
			self.cine.set_intestinal_roi_state(state if isinstance(state, dict) else None)
			self.cine_compare.set_intestinal_roi_state(state if isinstance(state, dict) else None)
		if "gate_roi_state" in params:
			state = params.get("gate_roi_state")
			self.cine.set_gate_roi_state(state if isinstance(state, dict) else None)
			self.cine_compare.set_gate_roi_state(state if isinstance(state, dict) else None)
		if "ui_show_helpers" in params:
			self._ui_show_helpers = bool(params["ui_show_helpers"])
		if "ui_enable_tooltips" in params:
			self._ui_enable_tooltips = bool(params["ui_enable_tooltips"])
		if "ui_compact_controls" in params:
			self._ui_compact_controls = bool(params["ui_compact_controls"])
		self._apply_global_ui_preferences()
		self._save_global_ui_preferences()
		if "manual_rois_text" in params:
			self.manual_rois.setPlainText(str(params["manual_rois_text"]))
			self.cine.set_manual_rois(self._parse_manual_rois())
		if "ecg_ritmo" in params:
			self.ecg_ritmo_combo.setCurrentText(str(params["ecg_ritmo"]))
		if "ecg_fc" in params:
			self.ecg_fc_spin.setValue(int(params["ecg_fc"]))
		if "ecg_qrs" in params:
			self.ecg_qrs_spin.setValue(int(params["ecg_qrs"]))
		if "ecg_qt" in params:
			self.ecg_qt_spin.setValue(int(params["ecg_qt"]))
		if "ecg_bri" in params:
			self.ecg_bri_check.setChecked(bool(params["ecg_bri"]))
		if "ecg_brd" in params:
			self.ecg_brd_check.setChecked(bool(params["ecg_brd"]))
		if "ecg_marcapasos" in params:
			self.ecg_marcapasos_check.setChecked(bool(params["ecg_marcapasos"]))
		if "ecg_observaciones" in params:
			self.ecg_obs_edit.setText(str(params["ecg_observaciones"]))
		if "ecg_file_path" in params:
			self.ecg_file_path = str(params["ecg_file_path"])
			if self.ecg_file_path:
				self.ecg_preview_label.setText(f"ECG adjuntado: {os.path.basename(self.ecg_file_path)}")
			else:
				self.ecg_preview_label.setText("Sin ECG cargado")
		if "cine_crudo_visual_config" in params:
			cfg = params.get("cine_crudo_visual_config")
			if isinstance(cfg, dict):
				self._apply_cine_crudo_visual_config(cfg, refresh=True)

	def save_current_preset(self):
		patient = self._current_patient_key()
		name = self.preset_name_edit.text().strip()
		if not name:
			QMessageBox.information(self, "SINCRO", "Ingresá un nombre de preset.")
			return
		self._presets_data.setdefault(patient, {})[name] = self._collect_processing_params()
		self._save_presets_store()
		self._refresh_presets_for_current_patient()
		self.preset_combo.setCurrentText(name)
		self._log(f"Preset guardado: paciente={patient}, preset={name}")
		self.statusBar().showMessage(f"Preset '{name}' guardado para '{patient}'.")

	def load_selected_preset(self):
		patient = self._current_patient_key()
		name = self.preset_combo.currentText().strip()
		if not name:
			QMessageBox.information(self, "SINCRO", "No hay preset seleccionado.")
			return
		params = ((self._presets_data.get(patient) or {}).get(name) or None)
		if params is None:
			QMessageBox.warning(self, "SINCRO", "No se encontró el preset para este paciente.")
			return
		self._apply_processing_params(params)
		self.preset_name_edit.setText(name)
		self._log(f"Preset cargado: paciente={patient}, preset={name}")
		self.statusBar().showMessage(f"Preset '{name}' cargado.")

	def delete_selected_preset(self):
		patient = self._current_patient_key()
		name = self.preset_combo.currentText().strip()
		if not name:
			QMessageBox.information(self, "SINCRO", "No hay preset seleccionado.")
			return
		patient_presets = self._presets_data.get(patient) or {}
		if name not in patient_presets:
			QMessageBox.warning(self, "SINCRO", "No se encontró el preset para borrar.")
			return
		del patient_presets[name]
		if not patient_presets and patient in self._presets_data:
			del self._presets_data[patient]
		self._save_presets_store()
		self._refresh_presets_for_current_patient()
		self._log(f"Preset borrado: paciente={patient}, preset={name}")
		self.statusBar().showMessage(f"Preset '{name}' borrado.")

	def _on_phase_cmap_changed(self, name: str):
		idx = self.cine.cmap_combo.findText(str(name))
		if idx >= 0 and self.cine.cmap_combo.currentIndex() != idx:
			self.cine.cmap_combo.setCurrentIndex(idx)

	def _set_window_icon(self):
		assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
		icon_path = os.path.join(assets_dir, "logo_gammasync_256.png")
		if os.path.exists(icon_path):
			self.setWindowIcon(QIcon(icon_path))

	def _build_sidebar(self) -> QWidget:
		sidebar = QWidget()
		sidebar.setObjectName("sincroSidebar")
		sidebar.setMinimumWidth(300)
		sidebar.setMaximumWidth(560)
		sidebar.setStyleSheet(
			"#sincroSidebar { background: #f7f8fb; border-right: 1px solid #d7dce5; }"
			"QGroupBox { font-weight: 600; border: 1px solid #d7dce5; border-radius: 7px; margin-top: 6px; background: white; }"
			"QGroupBox::title { subcontrol-origin: margin; left: 7px; padding: 0 2px; color: #1f3b5b; }"
			# El contenido de una seccion colapsable no lleva marco propio: el marco
			# y el titulo los aporta el encabezado clickeable (QToolButton).
			"QGroupBox#collapsibleContent { font-weight: 400; border: 1px solid #d7dce5; border-top: none;"
			" border-top-left-radius: 0; border-top-right-radius: 0; margin-top: 0; background: white; }"
			"QToolButton#collapsibleHeader { text-align: left; padding: 5px 8px; font-weight: 600; font-size: 11px;"
			" color: #1f3b5b; background: #eaeff7; border: 1px solid #d7dce5; border-radius: 7px; margin-top: 6px; }"
			"QToolButton#collapsibleHeader:hover { background: #dde6f4; }"
			"QToolButton#collapsibleHeader:checked { background: #e3ebf7; border-bottom-left-radius: 0;"
			" border-bottom-right-radius: 0; }"
			"QPushButton { padding: 4px 7px; font-size: 11px; }"
			"QLabel { font-size: 11px; }"
			"QTextEdit, QPlainTextEdit, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox { background: white; }"
		)
		scroll = QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setFrameShape(QFrame.Shape.NoFrame)
		scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
		container = QWidget()
		self._sidebar_layout = QVBoxLayout(container)
		self._sidebar_layout.setContentsMargins(5, 5, 5, 5)
		self._sidebar_layout.setSpacing(4)

		banner = QLabel()
		banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
		banner.setStyleSheet("background: transparent; border: none;")
		assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
		banner_path = os.path.join(assets_dir, "logo_gammasync_banner.png")
		if os.path.exists(banner_path):
			pix = QPixmap(banner_path)
			banner.setPixmap(pix.scaledToWidth(230, Qt.TransformationMode.SmoothTransformation))
		else:
			banner.setText("GammaSync")
			banner.setStyleSheet("font-size: 18px; font-weight: 700; color: #1f3b5b;")
		self._sidebar_layout.addWidget(banner)
		version_label = QLabel(f"Versión v{__version__}")
		version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		version_label.setStyleSheet("color:#4b5563; font-size:10px; font-weight:600;")
		self._sidebar_layout.addWidget(version_label)

		self._progress_bar = QProgressBar()
		self._progress_bar.setRange(0, 100)
		self._progress_bar.setValue(0)
		self._progress_bar.setMinimumHeight(18)
		self._progress_bar.setTextVisible(True)
		self._progress_bar.setFormat("Listo")
		self._progress_bar.setStyleSheet(
			"QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; background: #222; height: 16px; }"
			" QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0066cc, stop:1 #00cc88); border-radius: 3px; }"
		)
		self._sidebar_layout.addWidget(self._progress_bar)

		scroll.setWidget(container)
		layout = QVBoxLayout(sidebar)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(scroll)
		return sidebar

	def _browse_file(self):
		paths = self._select_dicom_paths(
			title="Abrir DICOM gated",
			allow_multiple=False,
		)
		if paths:
			self.file_edit.setText(paths[0])
			if not self.preset_patient_edit.text().strip():
				self._refresh_presets_for_current_patient()
			if self.auto_run_check.isChecked():
				self.process_auto()

	def _default_dicom_start_dir(self) -> str:
		"""Devuelve la carpeta inicial común para todos los diálogos DICOM."""
		settings = getattr(self, "_ui_settings", None)
		candidates = [
			self._last_browse_dir,
			str(settings.value("paths/last_dicom_dir", "")) if settings else "",
			os.path.dirname(self.file_edit.text().strip()) if getattr(self, "file_edit", None) is not None else "",
			getattr(self, "output_dir", ""),
		]
		for c in candidates:
			if c and os.path.isdir(c):
				return c
		return ""

	def _select_dicom_paths(
		self,
		*,
		title: str,
		allow_multiple: bool,
		max_files: int | None = None,
		start_dir_override: str = "",
	) -> list[str]:
		"""Diálogo común DICOM para todos los flujos (abrir, comparar, 1/2 estudios)."""
		start_dir = start_dir_override if start_dir_override and os.path.isdir(start_dir_override) else self._default_dicom_start_dir()
		flt = "DICOM (*.dcm *.DCM *.dicom *.DICOM *.ima *.IMA);;Todos (*.*)"
		if allow_multiple:
			paths, _ = QFileDialog.getOpenFileNames(self, title, start_dir, flt)
		else:
			path, _ = QFileDialog.getOpenFileName(self, title, start_dir, flt)
			paths = [path] if path else []
		valid_paths = [p for p in paths if p and os.path.exists(p)]
		if max_files is not None and len(valid_paths) > int(max_files):
			valid_paths = valid_paths[: int(max_files)]
		if valid_paths:
			self._record_browse_dir(os.path.dirname(valid_paths[0]))
		return valid_paths

	def _record_browse_dir(self, dirname: str):
		"""Registra la carpeta usada y actualiza memoria + persistencia."""
		if not dirname or not os.path.isdir(dirname):
			return
		self._last_browse_dir = dirname
		settings = getattr(self, "_ui_settings", None)
		if settings:
			settings.setValue("paths/last_dicom_dir", dirname)
			settings.sync()
		# Agregar a recientes (max 10, sin duplicados, al tope)
		recents = self._get_recent_dirs()
		if dirname in recents:
			recents.remove(dirname)
		recents.insert(0, dirname)
		recents = recents[:10]
		self._set_recent_dirs(recents)
		self._rebuild_recent_dirs_menu()

	def _get_recent_dirs(self) -> list[str]:
		"""Recupera la lista de carpetas recientes desde QSettings."""
		settings = getattr(self, "_ui_settings", None)
		if not settings:
			return []
		val = settings.value("paths/recent_dirs", [])
		if isinstance(val, list):
			return [str(v) for v in val if isinstance(v, str) and os.path.isdir(v)]
		return []

	def _set_recent_dirs(self, dirs: list[str]):
		"""Persiste la lista de recientes."""
		settings = getattr(self, "_ui_settings", None)
		if settings:
			settings.setValue("paths/recent_dirs", dirs)
			settings.sync()

	def _get_favorite_dirs(self) -> list[str]:
		"""Recupera las carpetas favoritas desde QSettings."""
		settings = getattr(self, "_ui_settings", None)
		if not settings:
			return []
		val = settings.value("paths/favorite_dirs", [])
		if isinstance(val, list):
			return [str(v) for v in val if isinstance(v, str) and os.path.isdir(v)]
		return []

	def _set_favorite_dirs(self, dirs: list[str]):
		"""Persiste las carpetas favoritas."""
		settings = getattr(self, "_ui_settings", None)
		if settings:
			settings.setValue("paths/favorite_dirs", dirs)
			settings.sync()

	def _add_favorite_dir(self, dirname: str):
		"""Marca una carpeta como favorita."""
		if not dirname or not os.path.isdir(dirname):
			return
		favs = self._get_favorite_dirs()
		if dirname not in favs:
			favs.append(dirname)
			self._set_favorite_dirs(favs)
			self._rebuild_recent_dirs_menu()
			self._log(f"Carpeta favorita: {dirname}")

	def _remove_favorite_dir(self, dirname: str):
		"""Quita una carpeta de favoritos."""
		favs = self._get_favorite_dirs()
		if dirname in favs:
			favs.remove(dirname)
			self._set_favorite_dirs(favs)
			self._rebuild_recent_dirs_menu()

	def _favorite_current_browse_dir(self):
		"""Marca como favorita la carpeta actual sin pasar por el menú desplegable."""
		dirname = self._last_browse_dir
		if not dirname:
			path_txt = self.file_edit.text().strip() if getattr(self, "file_edit", None) is not None else ""
			dirname = os.path.dirname(path_txt) if path_txt else ""
		if not dirname or not os.path.isdir(dirname):
			QMessageBox.information(self, "SINCRO", "No hay carpeta válida para marcar como favorita todavía.")
			return
		self._add_favorite_dir(dirname)
		self.statusBar().showMessage(f"Favorita guardada: {dirname}")

	def _rebuild_recent_dirs_menu(self):
		"""Reconstruye el menú desplegable de carpetas recientes y favoritas."""
		self._recent_dirs_menu.clear()
		favs = self._get_favorite_dirs()
		recents = self._get_recent_dirs()
		last = self._last_browse_dir

		if favs:
			for d in favs:
				action = self._recent_dirs_menu.addAction(f"★ {d}")
				action.setData(d)
				action.triggered.connect(lambda checked, p=d: self._browse_to_dir(p))
			self._recent_dirs_menu.addSeparator()

		if recents:
			for d in recents[:8]:  # max 8 en el menu
				if d in favs:
					continue  # ya esta arriba
				star = " ★" if d == last else ""
				action = self._recent_dirs_menu.addAction(f"{d}{star}")
				action.setData(d)
				action.triggered.connect(lambda checked, p=d: self._browse_to_dir(p))
			self._recent_dirs_menu.addSeparator()

		if last and last not in favs:
			add_action = self._recent_dirs_menu.addAction(f"Guardar esta carpeta como favorita")
			add_action.triggered.connect(lambda: self._add_favorite_dir(last))
		if favs:
			self._recent_dirs_menu.addSeparator()
			for d in favs:
				rm_action = self._recent_dirs_menu.addAction(f"✕ Quitar favorito: {os.path.basename(d)}")
				rm_action.triggered.connect(lambda checked, p=d: self._remove_favorite_dir(p))

		if not favs and not recents:
			empty = self._recent_dirs_menu.addAction("(sin carpetas recientes)")
			empty.setEnabled(False)

	def _browse_to_dir(self, dirname: str):
		"""Abre el diálogo de archivo en la carpeta indicada."""
		if not dirname or not os.path.isdir(dirname):
			return
		paths = self._select_dicom_paths(
			title="Abrir DICOM gated",
			allow_multiple=False,
			start_dir_override=dirname,
		)
		if paths:
			self.file_edit.setText(paths[0])
			if not self.preset_patient_edit.text().strip():
				self._refresh_presets_for_current_patient()
			if self.auto_run_check.isChecked():
				self.process_auto()

	def _manual_ecg_data(self):
		from core.ecg_extractor import ECGData
		return ECGData(
			ritmo=str(self.ecg_ritmo_combo.currentText()),
			fc=int(self.ecg_fc_spin.value()),
			qrs_ms=int(self.ecg_qrs_spin.value()),
			qt_ms=int(self.ecg_qt_spin.value()),
			bri=bool(self.ecg_bri_check.isChecked()),
			brd=bool(self.ecg_brd_check.isChecked()),
			marcapasos=bool(self.ecg_marcapasos_check.isChecked()),
			observaciones=str(self.ecg_obs_edit.text()),
			fuente="manual",
		)

	def _study_n_gates(self) -> int:
		st = getattr(self, "study", None)
		if st is None:
			return 0
		try:
			return int(getattr(st, "n_gates", 0) or 0)
		except (TypeError, ValueError):
			return 0

	def _apply_gated_controls_state(self):
		"""Habilita/deshabilita los controles que exigen gatillado (>=3 gates).

		En estudios ungated (FEVI/asincronía/fase no calculables) deja disponibles
		reconstrucción, cine, QC y NITIDA, y desactiva lo clínico dependiente del gating.
		También deshabilita la fuente Gated del montaje y todos los controles gated
		de la barra de reconstrucción (para no generar confusiones).
		"""
		gated = self._study_n_gates() >= 3
		for attr in ("asynchrony_review_btn", "ectb_window_btn"):
			btn = getattr(self, attr, None)
			if btn is None:
				continue
			if not hasattr(btn, "_orig_tooltip"):
				btn._orig_tooltip = btn.toolTip()
			btn.setEnabled(gated)
			btn.setToolTip(
				btn._orig_tooltip if gated
				else "Requiere estudio gatillado (≥3 gates). Estudio ungated: FEVI/asincronía no disponibles."
			)
		# Fuente Gated del montaje: deshabilitar si no hay gates (no hay cine).
		if hasattr(self, "cine_crudo_montage_source_combo"):
			idx = self.cine_crudo_montage_source_combo.findData("gated")
			if idx >= 0:
				self.cine_crudo_montage_source_combo.model().item(idx).setEnabled(gated)
				if not gated and str(getattr(self, "cine_crudo_montage_source", "ungated")) == "gated":
					self.cine_crudo_montage_source_combo.setCurrentIndex(0)  # volver a Ungated
		# Controles GATED de la reconstrucción: deshabilitar todos si no hay gates.
		gated_widgets = (
			"cine_crudo_fbpclean_check", "cine_crudo_fbpclean_slider",
			"cine_crudo_nitida3_check", "cine_crudo_nitida3_iter_spin",
			"cine_crudo_nitida4d_check", "cine_crudo_nitida4d_beta_spin",
			"cine_crudo_nitida2_combo",
			"cine_crudo_post_gated_check", "cine_crudo_post_gated_fwhm_spin",
			"cine_crudo_denoise_plus_gated_check", "cine_crudo_denoise_plus_gated_slider",
		)
		for attr in gated_widgets:
			w = getattr(self, attr, None)
			if w is not None:
				w.setEnabled(gated)

	def _handle_raw_projections_loaded(self, path: str, t_total: float):
		"""Maneja un estudio crudo (proyecciones gated): genera panel QC y muestra info/gating."""
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		from core.raw_projections import build_sinograms, center_of_mass_tracking

		# --- Ventana de scatter hermana (EM/SC): si el loader la adjuntó, ---
		# --- habilitar el control y PREGUNTAR al usuario si quiere usarla. ---
		_sc = getattr(self.study, "scatter_projections", None)
		_sc_rest = getattr(self._secondary_cine_crudo_study(), "scatter_projections", None)
		if hasattr(self, "cine_crudo_scatter_check"):
			if _sc is not None or _sc_rest is not None:
				self.cine_crudo_scatter_check.setEnabled(True)
				self.cine_crudo_scatter_k_spin.setEnabled(True)
				# Si el DICOM trae las ventanas de energía, el loader calculó el
				# k de TEW (W_EM / (2*W_SC)). Usarlo como default del spin: es
				# físicamente mejor que el 1.0 a ciegas. El usuario puede cambiarlo.
				_k_tew = getattr(self.study, "scatter_k_tew", None)
				if _k_tew is not None and float(_k_tew) > 0:
					self.cine_crudo_scatter_k_spin.setValue(float(_k_tew))
				sc_name_top = os.path.basename(str(getattr(self.study, "scatter_path", "") or "?_SC")) if _sc is not None else "N/D"
				sc_name_bot = os.path.basename(str(getattr(self._secondary_cine_crudo_study(), "scatter_path", "") or "?_SC")) if _sc_rest is not None else "N/D"
				_k_msg = f" (k TEW={float(_k_tew):.3f} de las ventanas del DICOM)" if _k_tew else ""
				if _sc is not None and _sc_rest is not None:
					self._log(f"Ventanas de scatter detectadas en ambas etapas: stress={sc_name_top} | rest={sc_name_bot}{_k_msg}.")
				else:
					self._log(f"Ventana de scatter detectada: {sc_name_top if _sc is not None else sc_name_bot} (misma geometría que EM){_k_msg}.")
				ans = QMessageBox.question(
					self, "SINCRO — Scatter EM/SC",
					(
						"Se detectó SCATTER hermano en la(s) etapa(s) crudas:\n"
						f"• Stress: {sc_name_top}\n"
						f"• Rest: {sc_name_bot}\n\n"
						"¿Usarlo para la corrección de scatter en la reconstrucción?\n"
						"(P = EM − k×SC, pre-recon. Podés cambiarlo con el checkbox 'Desc. SC'.)"
					)
					,
					QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
					QMessageBox.StandardButton.Yes,
				)
				self.cine_crudo_scatter_check.setChecked(ans == QMessageBox.StandardButton.Yes)
			else:
				self.cine_crudo_scatter_check.setEnabled(False)
				self.cine_crudo_scatter_check.setChecked(False)
				self.cine_crudo_scatter_k_spin.setEnabled(False)

		projections = np.asarray(self.study.cube, dtype=np.float64)  # (gates, angles, H, W)
		n_gates, n_angles = int(projections.shape[0]), int(projections.shape[1])
		gating = getattr(self.study, "gating_info", {}) or {}

		self._set_progress(40, "Generando QC de proyecciones crudas...")
		sh, sv = build_sinograms(projections)
		ty = center_of_mass_tracking(projections, axis="y")
		tx = center_of_mass_tracking(projections, axis="x")

		fig, axes = plt.subplots(2, 3, figsize=(15, 8))
		fig.patch.set_facecolor("#0b1220")
		# Sinograma VERTICAL: ángulo en eje Y (vertical), posición axial en X.
		# sh es (n_angles, H): filas=ángulos → se ve vertical (alto), como Odyssey/Xeleris.
		axes[0, 0].imshow(sh, cmap="gray", aspect="auto")
		axes[0, 0].set_title("Sinograma VERTICAL (perfil axial)", color="white")
		axes[0, 0].set_xlabel("posición axial (px)", color="white")
		axes[0, 0].set_ylabel("ángulo de proyección", color="white")
		# Sinograma HORIZONTAL: ángulo en eje X, posición horizontal en Y.
		# sv es (n_angles, W) → transponer para ángulo en X.
		axes[1, 0].imshow(sv.T, cmap="gray", aspect="auto")
		axes[1, 0].set_title("Sinograma HORIZONTAL (perfil transversal)", color="white")
		axes[1, 0].set_xlabel("ángulo de proyección", color="white")
		axes[1, 0].set_ylabel("posición horizontal (px)", color="white")

		summed = projections.sum(axis=0)
		pos = [(0, 1, 0), (0, 2, n_angles // 3), (1, 1, 2 * n_angles // 3)]
		for r, c, a in pos:
			axes[r, c].imshow(summed[a], cmap="gray")
			axes[r, c].set_title(f"Proyeccion ang {a}", color="white")
			axes[r, c].axis("off")

		axc = axes[1, 2]
		ang = np.arange(n_angles)
		axc.plot(ang, ty["com_series"], "o-", color="cyan", label="COM Y", ms=4)
		axc.plot(ang, tx["com_series"], "s-", color="orange", label="COM X", ms=4)
		out_y = np.where(ty["outliers"])[0]
		if out_y.size:
			axc.plot(out_y, ty["com_series"][out_y], "r*", ms=13, label=f"outliers Y ({ty['n_outliers']})")
		axc.set_title(f"COM tracking: mov Y={ty['motion_suspected']} (max {ty['max_shift_px']}px)", color="white", fontsize=10)
		axc.set_xlabel("angulo"); axc.set_ylabel("centro de masa (px)")
		axc.legend(fontsize=8); axc.grid(alpha=0.3)

		for ax in axes.ravel():
			ax.set_facecolor("#0b1220")
			ax.tick_params(colors="white")
			for s in ax.spines.values():
				s.set_color("#334155")

		ctx_label = self._study_context_label(path_override=path, study_obj=self.study)
		fc = gating.get("heart_rate") or gating.get("heart_rate_est") or "N/D"
		fig.suptitle(
			f"QC Crudo Gated — {ctx_label} | {n_gates} gates × {n_angles} ángulos | FC {fc} lpm",
			color="white", fontsize=12, fontweight="bold",
		)
		fig.tight_layout(rect=[0, 0, 1, 0.95])
		self._stamp_export_figure(fig, None)
		out_png = os.path.join(self.output_dir, "qc_crudo_proyecciones.png")
		fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
		plt.close(fig)

		# Grilla de cortes transaxiales (discriminar corazón de hígado, como Odyssey).
		# Los cortes son las proyecciones sumadas (UngGat) en grilla, con threshold aplicado,
		# para que el usuario vea todas las vistas y elija en cuál hacer pick en el corazón.
		try:
			from core.raw_projections import ungate_projections
			unggat = ungate_projections(projections)  # (angles, H, W)
			thr_grid = 0.20
			fig2, axes2 = plt.subplots(4, 8, figsize=(16, 8))
			fig2.patch.set_facecolor("#0b1220")
			p99g = float(np.percentile(unggat, 99.0)) or 1.0
			for idx, ax in enumerate(axes2.ravel()):
				ax.axis("off")
				ax.set_facecolor("#0b1220")
				if idx >= n_angles:
					continue
				img = unggat[idx]
				img_n = np.clip(img / p99g, 0, 1)
				ax.imshow(img_n, cmap="gray")
				# Máscara por threshold en naranja (como Odyssey) para discriminar órganos.
				mask = img > (thr_grid * img.max()) if img.max() > 0 else np.zeros_like(img, dtype=bool)
				ov = np.ma.masked_where(~mask, mask)
				ax.imshow(ov, cmap="autumn", alpha=0.55)
				counts = float(img.sum())
				ax.set_title(f"#{idx} · {counts:.0f}c", color="white", fontsize=7, pad=1)
			fig2.suptitle(
				f"Cortes transaxiales (UngGat) + threshold {thr_grid:.2f} — elegí el frame y hacé pick en el corazón (evitar hígado) | {ctx_label}",
				color="white", fontsize=11, fontweight="bold",
			)
			fig2.tight_layout(rect=[0, 0, 1, 0.94])
			grid_png = os.path.join(self.output_dir, "cortes_pick_corazon.png")
			fig2.savefig(grid_png, dpi=130, bbox_inches="tight", facecolor=fig2.get_facecolor())
			plt.close(fig2)
			if "ungated" in self.preview_labels:
				pix = QPixmap(grid_png)
				self.preview_pixmaps["ungated"] = pix
				self.preview_base_sizes["ungated"] = pix.size()
				self._apply_preview_zoom("ungated")
		except Exception as exc:
			self._log(f"[WARN] No se pudo generar grilla de cortes: {exc}")

		# Cine del crudo (proyecciones por ángulo) y del UngGat (desgatillado alta estadística)
		try:
			from core.raw_projections import ungate_projections
			from PIL import Image
			unggat = ungate_projections(projections)  # (angles, H, W) suma gates

			def _norm_frame(img):
				img = np.asarray(img, dtype=np.float64)
				mx = float(img.max()) if img.size else 1.0
				if mx <= 0:
					mx = 1.0
				return (np.clip(img / mx, 0, 1) * 255).astype(np.uint8)

			# GIF crudo gated: primer gate (o gate medio) por ángulo
			gate_mid = n_gates // 2
			frames_crudo = [Image.fromarray(_norm_frame(projections[gate_mid, a])).convert("P") for a in range(n_angles)]
			gif_crudo = os.path.join(self.output_dir, "cine_crudo_gated.gif")
			frames_crudo[0].save(gif_crudo, save_all=True, append_images=frames_crudo[1:], duration=120, loop=0)

			# GIF UngGat (suma gates, alta estadística)
			frames_unggat = [Image.fromarray(_norm_frame(unggat[a])).convert("P") for a in range(n_angles)]
			gif_unggat = os.path.join(self.output_dir, "cine_unggat.gif")
			frames_unggat[0].save(gif_unggat, save_all=True, append_images=frames_unggat[1:], duration=120, loop=0)
			self._log(f"Cines generados: cine_crudo_gated.gif (gate {gate_mid}) + cine_unggat.gif (UngGat {n_gates}× cuentas)")
		except Exception as exc:
			self._log(f"[WARN] No se pudo generar cine del crudo: {exc}")

		# Mostrar el panel en la pestaña ungated (reutilizada como visor QC crudo)
		if "ungated" in self.preview_labels:
			pix = QPixmap(out_png)
			self.preview_pixmaps["ungated"] = pix
			self.preview_base_sizes["ungated"] = pix.size()
			self._apply_preview_zoom("ungated")

		# Cargar cine del crudo en la pestaña cine_crudo (UngGat por defecto, alta estadística)
		if "cine_crudo" in self.preview_labels:
			try:
				source = str(self.cine_crudo_source_combo.currentText()) if hasattr(self, "cine_crudo_source_combo") else "UngGat"
				self._load_cine_crudo_frames(source)
				self._select_tab_by_title("cine_crudo")
			except Exception as exc:
				self._log(f"[WARN] No se pudo cargar cine crudo: {exc}")
				self._select_tab_by_title("ungated")
		else:
			self._select_tab_by_title("ungated")

		mov_txt = "MOVIMIENTO detectado" if ty["motion_suspected"] else "sin movimiento significativo"
		self._log(
			f"CRUDO cargado: {n_gates} gates × {n_angles} ángulos | FC {fc} lpm | "
			f"{mov_txt} (max shift {ty['max_shift_px']}px Y, {tx['max_shift_px']}px X). "
		 f"Panel QC: qc_crudo_proyecciones.png"
		)
		self._set_progress(100, "Crudo cargado (QC listo)")
		self.statusBar().showMessage("Crudo cargado: QC de proyecciones listo")
		self._refresh_readonly_results_panel()

		QMessageBox.information(
			self,
			"Estudio crudo (proyecciones gated)",
			f"Se cargó el estudio CRUDO: {n_gates} gates × {n_angles} ángulos.\n\n"
			f"• FC adquisición: {fc} lpm\n"
			f"• Motion tracking: {mov_txt} (max {ty['max_shift_px']}px)\n"
			f"• Panel QC generado en la pestaña 'ungated'.\n\n"
			"El análisis de fase (segmentación/PSD/BW) requiere reconstrucción. "
			"Próximo paso del pipeline: motion correction y reconstrucción gate-por-gate.",
		)
		try:
			get_logger().log_processing_end(path, perf_counter() - t_total, {"mode": "raw_projections"})
		except Exception:
			pass

	def _preload_acquisition_ecg(self):
		"""Precarga datos del ECG de adquisición (3 derivaciones) embebidos en el DICOM SPECT."""
		gating = getattr(self.study, "gating_info", None) or {}
		if not gating:
			return
		aplicado = []
		fc = gating.get("heart_rate") or gating.get("heart_rate_est")
		# Ignorar FC absurda (placeholder GE: RR=1ms → FC=60000)
		if fc and 25 <= int(fc) <= 250:
			self.ecg_fc_spin.setValue(int(fc))
			aplicado.append(f"FC={fc} lpm")
		elif fc:
			aplicado.append("FC no confiable (placeholder en DICOM)")
		if gating.get("rr_mean_ms") and not gating.get("rr_placeholder"):
			rr_txt = f"RR medio {gating['rr_mean_ms']:.0f} ms"
			if gating.get("rr_cv_pct") is not None:
				rr_txt += f" (CV {gating['rr_cv_pct']:.1f}%)"
			aplicado.append(rr_txt)
			if gating.get("rr_variability_flag") == "alta":
				self.ecg_obs_edit.setText(
					"Variabilidad RR alta en adquisición (posible FA/extrasístoles); interpretar fase con cautela."
				)
		if gating.get("trigger_window_pct") is not None:
			aplicado.append(f"ventana trigger {gating['trigger_window_pct']:.0f}%")
		if not self.ecg_obs_edit.text().strip() and aplicado:
			self.ecg_obs_edit.setText("ECG adquisición (3 deriv.): " + "; ".join(aplicado))
		self.ecg_preview_label.setText("Precargado desde adquisición (3 derivaciones): " + "; ".join(aplicado))
		self._log(f"ECG adquisición precargado desde DICOM: {'; '.join(aplicado)}")

	def _apply_extracted_ecg(self, data):
		if data.ritmo and data.ritmo != "No especificado":
			self.ecg_ritmo_combo.setCurrentText(data.ritmo)
		if data.fc > 0:
			self.ecg_fc_spin.setValue(int(data.fc))
		if data.qrs_ms > 0:
			self.ecg_qrs_spin.setValue(int(data.qrs_ms))
		if data.qt_ms > 0:
			self.ecg_qt_spin.setValue(int(data.qt_ms))
		self.ecg_bri_check.setChecked(bool(data.bri))
		self.ecg_brd_check.setChecked(bool(data.brd))
		self.ecg_marcapasos_check.setChecked(bool(data.marcapasos))
		if data.observaciones:
			self.ecg_obs_edit.setText(str(data.observaciones))

	def _load_ecg_file(self):
		from core.ecg_extractor import compare_ecg_data, extract_ecg
		path, _ = QFileDialog.getOpenFileName(
			self,
			"Cargar ECG de 12 derivaciones",
			"",
			"ECG (*.pdf *.scp *.dcm *.dicom);;PDF (*.pdf);;SCP-ECG (*.scp);;DICOM (*.dcm *.dicom);;Todos (*.*)",
		)
		if not path:
			return
		try:
			extracted = extract_ecg(path)
		except Exception as exc:
			QMessageBox.warning(
				self,
				"ECG",
				f"No se pudieron extraer datos automáticos del ECG:\n{exc}\n\nPodés cargar los valores manualmente.",
			)
			self.ecg_file_path = path
			self.ecg_preview_label.setText(f"ECG adjuntado (sin extracción): {os.path.basename(path)}")
			return

		self.ecg_file_path = path
		manual = self._manual_ecg_data()
		comparison = compare_ecg_data(manual, extracted)

		# Contraste adicional contra ECG de adquisición (3 derivaciones) si existe
		gating = getattr(self.study, "gating_info", None) or {} if self.study is not None else {}
		fc_adq = gating.get("heart_rate") or gating.get("heart_rate_est")
		if fc_adq and extracted.fc > 0:
			diff_fc_adq = abs(int(fc_adq) - int(extracted.fc))
			if diff_fc_adq > 10:
				comparison["differences"].append({
					"field": "fc (3 deriv. adquisición)",
					"manual": fc_adq,
					"extracted": extracted.fc,
					"diff": diff_fc_adq,
					"significant": diff_fc_adq > 20,
				})
				comparison["has_differences"] = True

		resumen = (
			f"Extraído ({extracted.fuente}, confianza {extracted.confianza}): "
			f"ritmo={extracted.ritmo or 'N/D'} | FC={extracted.fc or 'N/D'} | "
			f"QRS={extracted.qrs_ms or 'N/D'}ms | QT={extracted.qt_ms or 'N/D'}ms | "
			f"BRI={'sí' if extracted.bri else 'no'} | BRD={'sí' if extracted.brd else 'no'} | "
			f"MP={'sí' if extracted.marcapasos else 'no'}"
		)

		if comparison["has_differences"]:
			lineas = ["Se encontraron diferencias entre los valores manuales y los extraídos del ECG:", ""]
			for d in comparison["differences"]:
				marca = " [SIGNIFICATIVO]" if d.get("significant") else ""
				lineas.append(f"• {d['field']}: manual={d['manual']} vs ECG={d['extracted']}{marca}")
			lineas += ["", "¿Aplicar los valores extraídos del ECG? (Sí = usar ECG, No = conservar manuales)"]
			resp = QMessageBox.question(
				self,
				"ECG: diferencias detectadas",
				"\n".join(lineas),
				QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
				QMessageBox.StandardButton.Yes,
			)
			if resp == QMessageBox.StandardButton.Yes:
				self._apply_extracted_ecg(extracted)
				self._log(f"ECG cargado y aplicado desde {os.path.basename(path)} ({comparison['n_significant']} diferencias significativas).")
			else:
				self._log(f"ECG adjuntado desde {os.path.basename(path)}; se conservaron los valores manuales.")
		else:
			self._apply_extracted_ecg(extracted)
			self._log(f"ECG cargado desde {os.path.basename(path)}; sin diferencias con valores manuales.")

		self.ecg_preview_label.setText(resumen)

	def _clear_ecg_file(self):
		self.ecg_file_path = ""
		self.ecg_preview_label.setText("Sin ECG cargado")
		self._log("ECG desvinculado del estudio.")

	def _find_axis_companion_path(self, sa_path: str, axis_code: str) -> str | None:
		base = os.path.basename(sa_path)
		axis_code = str(axis_code).upper()
		if "_SA" not in base.upper():
			return None
		candidate = base.upper().replace("_SA", f"_{axis_code}")
		dir_path = os.path.dirname(sa_path)
		for name in os.listdir(dir_path):
			if name.upper() == candidate:
				return os.path.join(dir_path, name)
		return None

	def _load_axis_companions(self, sa_path: str) -> dict[str, object]:
		companions: dict[str, object] = {}
		for axis_code in ("HLA", "VLA"):
			axis_path = self._find_axis_companion_path(sa_path, axis_code)
			if not axis_path or not os.path.exists(axis_path):
				continue
			try:
				companions[axis_code] = dicom_loader.load(axis_path, verbose=False)
			except Exception as exc:
				self._log(f"No se pudo cargar serie {axis_code} original: {exc}")
		return companions

	def _parse_manual_rois_text(self, raw_text: str) -> dict[int, tuple[float, float, float, float]]:
		rois: dict[int, tuple[float, float, float, float]] = {}
		for raw in str(raw_text or "").splitlines():
			line = raw.strip()
			if not line or line.startswith("#"):
				continue
			parts = [p.strip() for p in line.split(",")]
			if len(parts) != 5:
				continue
			try:
				s = int(parts[0])
				cy = float(parts[1])
				cx = float(parts[2])
				ri = float("nan") if parts[3] in ("", "-", "na", "n/a") else float(parts[3])
				ro = float(parts[4])
			except ValueError:
				continue
			rois[s] = (cy, cx, ri, ro)
		return rois

	def _parse_manual_rois(self) -> dict[int, tuple[float, float, float, float]]:
		return self._parse_manual_rois_text(self.manual_rois.toPlainText())

	def _format_manual_rois(self, rois: dict[int, tuple[float, float, float, float]]) -> str:
		lines = []
		for slice_index in sorted(rois):
			cy, cx, r_inner, r_outer = rois[slice_index]
			ri_txt = "-" if not np.isfinite(float(r_inner)) else f"{float(r_inner):.1f}"
			lines.append(f"{slice_index},{cy:.1f},{cx:.1f},{ri_txt},{r_outer:.1f}")
		return "\n".join(lines)

	def _rois_from_segmentation(self, seg_obj) -> dict[int, tuple[float, float, float, float]]:
		centers = np.asarray(getattr(seg_obj, "center_per_slice", np.empty((0, 2))), dtype=np.float64)
		inner = np.asarray(getattr(seg_obj, "inner_radius", np.empty((0,))), dtype=np.float64)
		outer = np.asarray(getattr(seg_obj, "outer_radius", np.empty((0,))), dtype=np.float64)
		if centers.ndim != 2 or centers.shape[1] != 2 or inner.shape[0] != centers.shape[0] or outer.shape[0] != centers.shape[0]:
			return {}
		rois: dict[int, tuple[float, float, float, float]] = {}
		for slice_index in range(int(centers.shape[0])):
			cy = float(centers[slice_index, 0])
			cx = float(centers[slice_index, 1])
			r_inner = float(inner[slice_index])
			r_outer = float(outer[slice_index])
			if not np.isfinite(cy) or not np.isfinite(cx) or not np.isfinite(r_outer) or r_outer <= 0.0:
				continue
			rois[int(slice_index)] = (cy, cx, r_inner if np.isfinite(r_inner) else float("nan"), r_outer)
		return rois

	def _set_manual_rois_text(self, text: str, *, autogenerated: bool = False):
		self._programmatic_manual_rois_update = True
		try:
			self.manual_rois.blockSignals(True)
			self.manual_rois.setPlainText(str(text or ""))
			self.manual_rois.blockSignals(False)
		finally:
			self._programmatic_manual_rois_update = False
		if self.active_cine_source == "compare":
			self.compare_manual_rois_text = str(text or "")
			self.compare_manual_rois_autogenerated = bool(autogenerated)
		else:
			self.primary_manual_rois_text = str(text or "")
			self.primary_manual_rois_autogenerated = bool(autogenerated)

	def _on_manual_rois_text_changed(self):
		if self._programmatic_manual_rois_update:
			return
		self._save_active_manual_rois_text()
		if self.active_cine_source == "compare":
			self.compare_manual_rois_autogenerated = False
		else:
			self.primary_manual_rois_autogenerated = False

	def roi_source(self) -> str:
		"""Geometría de ROI elegida para el análisis: 'ring' o 'ectb_wall'."""
		combo = getattr(self, "roi_source_combo", None)
		if combo is None:
			return "ring"
		value = combo.currentData()
		return str(value) if value else "ring"

	def set_roi_source(self, source: str) -> bool:
		"""Selecciona la geometría de ROI. Devuelve True si cambió."""
		combo = getattr(self, "roi_source_combo", None)
		if combo is None:
			return False
		index = combo.findData(str(source))
		if index < 0 or index == combo.currentIndex():
			return False
		combo.setCurrentIndex(index)
		return True

	def cavity_center_enabled(self) -> bool:
		"""True si el centro del VI se refina sobre la cavidad en vez del músculo."""
		check = getattr(self, "cavity_center_check", None)
		return bool(check is not None and check.isChecked())

	def set_cavity_center_enabled(self, enabled: bool) -> bool:
		"""Activa/desactiva el centrado en cavidad. Devuelve True si cambió."""
		check = getattr(self, "cavity_center_check", None)
		if check is None or bool(check.isChecked()) == bool(enabled):
			return False
		check.setChecked(bool(enabled))
		return True

	def _propagate_cavity_center(self):
		"""Baja el flag a los visores para que el Auto ROI dibujado coincida."""
		enabled = self.cavity_center_enabled()
		for name in ("cine", "cine_compare"):
			widget = getattr(self, name, None)
			setter = getattr(widget, "set_refine_cavity_center", None)
			if callable(setter):
				setter(enabled)

	def _on_cavity_center_toggled(self, checked: bool):
		"""Reprocesa al cambiar el criterio de centrado.

		Mover el centro cambia radios, contornos ECTb y la asignación angular a
		segmentos, así que invalida segmentación y fase por completo.
		"""
		self._propagate_cavity_center()
		state = "cavidad" if bool(checked) else "centroide de miocardio"
		self.statusBar().showMessage(f"Centro del VI: {state}")
		if self.study is None or not bool(getattr(self.study, "reconstructed", True)):
			return
		self._cache_seg_sig = ""
		self._cache_phase_sig = ""
		self._invalidate_output_cache()
		QTimer.singleShot(0, self.process_current)

	def _on_manual_center_mode_toggled(self, checked: bool):
		"""Activa el modo de fijar el centro de cavidad por clic en el cine."""
		cine = getattr(self, "cine", None)
		setter = getattr(cine, "set_center_pick_mode", None)
		if callable(setter):
			setter(bool(checked))
		if checked:
			self.statusBar().showMessage(
				"Centro manual activo: hacé clic en el centro de la cavidad (clic derecho borra)."
			)
		else:
			self.statusBar().showMessage("Centro manual desactivado.")

	def _clear_manual_centers(self):
		"""Borra todos los centros manuales y vuelve al centro automático."""
		if not self.manual_center_per_slice:
			self.statusBar().showMessage("No había centros manuales.")
			return
		self.manual_center_per_slice = {}
		self._push_manual_centers_to_cine()
		self.statusBar().showMessage("Centros manuales borrados; vuelve el centro automático.")
		if self.study is None or not bool(getattr(self.study, "reconstructed", True)):
			return
		self._cache_seg_sig = ""
		self._cache_phase_sig = ""
		self._invalidate_output_cache()
		QTimer.singleShot(0, self.process_current)

	def _on_center_picked(self, slice_index: int, center):
		"""Guarda (o borra) el centro de cavidad que el operador marcó por clic.

		center = (cy, cx) fija; center = None borra el de ese corte. Si está
		activo 'Aplicar a todos los cortes', el clic se propaga a toda la pila.
		"""
		try:
			s = int(slice_index)
		except (TypeError, ValueError):
			return
		apply_all = bool(getattr(self, "manual_center_all_check", None) and self.manual_center_all_check.isChecked())
		n_slices = None
		if self.study is not None and getattr(self.study, "cube", None) is not None:
			n_slices = int(self.study.cube.shape[1])

		if center is None:
			if apply_all:
				self.manual_center_per_slice = {}
			else:
				self.manual_center_per_slice.pop(s, None)
		else:
			cy, cx = float(center[0]), float(center[1])
			if apply_all and n_slices:
				self.manual_center_per_slice = {k: (cy, cx) for k in range(n_slices)}
			else:
				self.manual_center_per_slice[s] = (cy, cx)

		count = len(self.manual_center_per_slice)
		self._push_manual_centers_to_cine()
		self.statusBar().showMessage(
			f"Centro manual {'borrado' if center is None else 'fijado'} · {count} corte(s) con centro fijo."
		)
		if self.study is None or not bool(getattr(self.study, "reconstructed", True)):
			return
		self._cache_seg_sig = ""
		self._cache_phase_sig = ""
		self._invalidate_output_cache()
		QTimer.singleShot(0, self.process_current)

	def _manual_center_override_array(self, n_slices: int):
		"""Arma el array (n_slices, 2) de override para segment_myocardium, con
		NaN donde no hay centro manual. Devuelve None si no hay ninguno."""
		if not self.manual_center_per_slice:
			return None
		ov = np.full((int(n_slices), 2), np.nan, dtype=np.float64)
		for s, c in self.manual_center_per_slice.items():
			if 0 <= int(s) < int(n_slices) and c is not None:
				ov[int(s), 0] = float(c[0])
				ov[int(s), 1] = float(c[1])
		return ov

	def _push_manual_centers_to_cine(self):
		"""Envía el dict de centros manuales a los visores para dibujar el marcador."""
		for cine in (getattr(self, "cine", None), getattr(self, "cine_compare", None)):
			setter = getattr(cine, "set_manual_centers", None)
			if callable(setter):
				setter(self.manual_center_per_slice)

	def _log_cavity_center_shift(self, seg):
		"""Informa cuánto movió el centro el refinamiento en cavidad.

		Sin esto no hay forma de saber si la opción hizo algo: un corrimiento de
		medio píxel es invisible a ojo pero mueve la asignación angular.
		"""
		if not self.cavity_center_enabled():
			return
		shift = getattr(seg, "center_shift_px", None)
		if shift is None:
			return
		shift = np.asarray(shift, dtype=np.float64)
		valid = shift[np.isfinite(shift)]
		if valid.size == 0:
			self._log("Centrado en cavidad activo, pero ningún corte pudo refinarse.")
			return
		mean_shift = float(np.mean(valid))
		max_shift = float(np.max(valid))
		worst = int(np.nanargmax(np.where(np.isfinite(shift), shift, -np.inf)))
		moved = int(np.count_nonzero(valid > 0.05))
		self._log(
			f"Centrado en cavidad: {moved} de {valid.size} cortes movidos; "
			f"desplazamiento medio {mean_shift:.2f} px, máximo {max_shift:.2f} px (corte {worst + 1})."
		)

	def _apply_ectb_wall_segmentation(self, cube) -> bool:
		"""Reemplaza la ROI anular por la pared irregular que traza el ECTb.

		La segmentación clásica queda como semilla (aporta centro y extensión
		por corte, que es lo que el ECTb necesita para tirar los rayos); sobre
		esa base se recalculan endocardio y epicardio ángulo por ángulo y con
		eso se rearma la máscara. Si el ECTb no puede cuantificar, se conserva
		la ROI anular y se avisa, en vez de dejar el análisis sin máscara.
		"""
		study = self.study
		seg = self.seg
		if study is None or seg is None:
			return False
		pixel_spacing = getattr(study, "pixel_spacing", None)
		slice_mm = getattr(study, "z_spacing_mm", None)
		if not pixel_spacing or slice_mm is None:
			self._log("[WARN] Contornos Irregulares no aplicados: el estudio no trae spacing.")
			return False

		try:
			pixel_mm = (float(pixel_spacing[0]), float(pixel_spacing[1]))
			result = analyze_lv_ectb(cube, seg, pixel_mm, float(slice_mm), self.ectb_config())
			if not getattr(result, "available", False):
				reason = str(getattr(result, "reason", "") or "sin motivo informado")
				self._log(f"[WARN] Contornos Irregulares no aplicados ({reason}). Se mantiene la ROI anular.")
				return False
			wall_seg = wall_segmentation_from_ectb(result, seg, pixel_mm)
		except Exception as exc:
			self._log(f"[WARN] Contornos Irregulares no aplicados: {exc}. Se mantiene la ROI anular.")
			return False

		if wall_seg is None:
			self._log("[WARN] Contornos Irregulares no aplicados: los contornos no generaron máscara. Se mantiene la ROI anular.")
			return False

		ring_voxels = int(getattr(seg, "n_voxels", 0))
		self.seg_ring_base = seg
		self.seg = wall_seg
		self._log(
			f"ROI de análisis = Contornos Irregulares: {wall_seg.n_voxels} voxels "
			f"en {len(result.valid_slices)} de {result.n_slices_total} cortes "
			f"(la ROI anular tenía {ring_voxels})."
		)
		return True

	def _expose_segmentation_rois(self, seg_method: str):
		if self.seg is None:
			return
		rois = self._rois_from_segmentation(self.seg)
		if not rois:
			return
		formatted = self._format_manual_rois(rois)
		if str(seg_method) in ("auto", "threshold"):
			self._set_manual_rois_text(formatted, autogenerated=True)
			self.cine.set_manual_rois(rois)
			self._log(f"ROI {seg_method} visible/reproducible: {len(rois)} slices. Para fijarlo, cambiá Segmentación a manual y reprocesá.")
		else:
			self._set_manual_rois_text(formatted, autogenerated=False)

	def _is_roi_valid_for_manual(self, roi: tuple[float, float, float, float] | None) -> bool:
		if roi is None or len(roi) != 4:
			return False
		cy, cx, r_inner, r_outer = (float(v) for v in roi)
		if not np.isfinite(cy) or not np.isfinite(cx):
			return False
		if not np.isfinite(r_outer) or r_outer <= 0.0:
			return False
		if np.isfinite(r_inner) and r_inner < 0.0:
			return False
		if np.isfinite(r_inner) and r_outer <= r_inner:
			return False
		return True

	def _sync_manual_rois(self, rois: dict[int, tuple[float, float, float, float]], message: str | None = None):
		formatted = self._format_manual_rois(rois)
		self._set_manual_rois_text(formatted, autogenerated=False)
		if self.active_cine_source == "compare":
			self.compare_manual_rois_text = formatted
		else:
			self.primary_manual_rois_text = formatted
		if self.active_cine_source == "compare":
			self.cine_compare.set_manual_rois(rois)
		else:
			self.cine.set_manual_rois(rois)
		self._refresh_dual_cine_views(preserve_position=True)
		if message:
			self._log(message)

	def _refresh_cine_source_selector(self):
		self.cine_source_combo.blockSignals(True)
		self.cine_source_combo.clear()
		primary_label = "Esfuerzo / principal"
		if self.file_edit.text().strip():
			primary_label = f"Esfuerzo / {os.path.splitext(os.path.basename(self.file_edit.text().strip()))[0]}"
		self.cine_source_combo.addItem(primary_label, "primary")
		has_secondary = self.compare_bundle is not None or self.compare_raw_study is not None
		if has_secondary:
			compare_label = self.compare_label or os.path.splitext(os.path.basename(self.compare_raw_path or "comparacion"))[0]
			self.cine_source_combo.addItem(f"Reposo / {compare_label}", "compare")
			target_index = 1 if self.active_cine_source == "compare" else 0
		else:
			self.active_cine_source = "primary"
			target_index = 0
		self.cine_source_combo.setCurrentIndex(target_index)
		self.cine_source_combo.setEnabled(has_secondary)
		self.cine_primary_btn.setEnabled(self.study is not None)
		self.cine_compare_btn.setEnabled(has_secondary)
		self.cine_source_combo.blockSignals(False)
		self._update_patient_banner()

	def _load_manual_rois_text_for_source(self, source: str) -> str:
		return self.compare_manual_rois_text if source == "compare" else self.primary_manual_rois_text

	def _save_manual_rois_text_for_source(self, source: str, text: str):
		if source == "compare":
			self.compare_manual_rois_text = str(text)
		else:
			self.primary_manual_rois_text = str(text)

	def _cube_for_source(self, source: str):
		if source == "compare" and self.compare_bundle is not None:
			return self.compare_bundle["study"].cube
		if source == "compare" and self.compare_raw_study is not None:
			return self.compare_raw_study.cube
		if self.study is not None:
			return self.study.cube
		return None

	def _refresh_dual_cine_views(self, *, preserve_position: bool = True, preferred_gate: int | None = None, preferred_slice: int | None = None):
		main_cube = self._cube_for_source("primary")
		# Los crudos (sin reconstruir) NO deben hidratar los cines de asincronía:
		# esos viewers son solo para cortes SA ya reconstruidos.
		if self.study is not None and not bool(getattr(self.study, "reconstructed", True)):
			main_cube = None
		main_rois = self._parse_manual_rois_text(self._load_manual_rois_text_for_source("primary"))
		main_gate = self.cine.current_gate_index()
		main_slice = self.cine.current_slice_index()
		self.cine.set_manual_rois(main_rois)
		self.cine.set_cube(main_cube)
		if main_cube is not None:
			if preserve_position:
				gate_idx = min(main_gate, int(main_cube.shape[0]) - 1)
				slice_idx = min(main_slice, int(main_cube.shape[1]) - 1)
			else:
				gate_idx = int(main_cube.shape[0] // 2) if preferred_gate is None else int(preferred_gate)
				slice_idx = int(main_cube.shape[1] // 2) if preferred_slice is None else int(preferred_slice)
				gate_idx = max(0, min(gate_idx, int(main_cube.shape[0]) - 1))
				slice_idx = max(0, min(slice_idx, int(main_cube.shape[1]) - 1))
			self.cine.gate_slider.setValue(gate_idx)
			self.cine.slice_slider.setValue(slice_idx)

		has_secondary = self.compare_bundle is not None
		if not has_secondary:
			self.cine_secondary_source = None
			self.cine_compare.set_cube(None)
			self.active_cine_source = "primary"
			self.cine.set_controls_visible(True)
			self.cine_compare.set_controls_visible(False)
			self._update_cine_active_border()
			return

		self.cine_secondary_source = "compare"
		other_cube = self._cube_for_source("compare")
		other_rois = self._parse_manual_rois_text(self._load_manual_rois_text_for_source("compare"))
		other_gate = self.cine_compare.current_gate_index()
		other_slice = self.cine_compare.current_slice_index()
		self.cine_compare.set_manual_rois(other_rois)
		self.cine_compare.set_cube(other_cube)
		if other_cube is not None:
			if preserve_position:
				gate_idx = min(other_gate, int(other_cube.shape[0]) - 1)
				slice_idx = min(other_slice, int(other_cube.shape[1]) - 1)
			else:
				gate_base = self.cine.current_gate_index() if preferred_gate is None else int(preferred_gate)
				slice_base = self.cine.current_slice_index() if preferred_slice is None else int(preferred_slice)
				gate_idx = max(0, min(gate_base, int(other_cube.shape[0]) - 1))
				slice_idx = max(0, min(slice_base, int(other_cube.shape[1]) - 1))
			self.cine_compare.gate_slider.setValue(gate_idx)
			self.cine_compare.slice_slider.setValue(slice_idx)
		self._update_cine_active_border()

	def _update_cine_active_border(self):
		if self.compare_bundle is None and self.compare_raw_study is None:
			self.cine.set_active_highlight(True)
			self.cine_compare.set_active_highlight(False)
			return
		main_active = self.active_cine_source != "compare"
		self.cine.set_active_highlight(main_active)
		self.cine_compare.set_active_highlight(not main_active)

	def _on_cine_panel_activated(self, panel: str):
		if panel == "secondary" and (self.compare_bundle is not None or self.compare_raw_study is not None):
			self._apply_cine_source("compare", preserve_position=True)
			self.statusBar().showMessage("Visor activo: Reposo")
			return
		self._apply_cine_source("primary", preserve_position=True)
		self.statusBar().showMessage("Visor activo: Esfuerzo")

	def _on_debug_grid_toggled(self, checked: bool):
		"""Activa/desactiva la grilla de debug (bordes rojos) en ambos cines."""
		for cine in (self.cine, self.cine_compare):
			if cine is not None and hasattr(cine, "set_debug_grid"):
				cine.set_debug_grid(checked)

	def _current_cine_cube(self):
		if self.active_cine_source == "compare" and self.compare_bundle is not None:
			return self.compare_bundle["study"].cube
		if self.study is not None:
			return self.study.cube
		return None

	def _secondary_cine_crudo_study(self):
		"""Devuelve el estudio secundario usable como CRUDO para la pestaña cine_crudo."""
		primary = getattr(self, "study", None)
		def _is_same_study(a, b) -> bool:
			if a is None or b is None:
				return False
			if a is b:
				return True
			# OJO: StudyInstanceUID puede ser el mismo para esfuerzo/reposo (series
			# distintas del mismo estudio). No usarlo para deduplicar acá.
			try:
				ap = str(getattr(a, "source_path", "") or getattr(a, "path", "") or "")
				bp = str(getattr(b, "source_path", "") or getattr(b, "path", "") or "")
				if ap and bp and os.path.normcase(ap) == os.path.normcase(bp):
					return True
			except Exception:
				pass
			try:
				aser = str(getattr(a, "series_instance_uid", "") or "")
				bser = str(getattr(b, "series_instance_uid", "") or "")
				if aser and bser and aser == bser:
					return True
			except Exception:
				pass
			return False
		if self.compare_raw_study is not None and not bool(getattr(self.compare_raw_study, "reconstructed", True)):
			if not _is_same_study(self.compare_raw_study, primary):
				return self.compare_raw_study
		if self.compare_bundle is not None:
			st = self.compare_bundle.get("study")
			if st is not None and not bool(getattr(st, "reconstructed", True)):
				if not _is_same_study(st, primary):
					return st
		return None

	def _save_active_manual_rois_text(self):
		current_text = self.manual_rois.toPlainText()
		if self.active_cine_source == "compare":
			self.compare_manual_rois_text = current_text
		else:
			self.primary_manual_rois_text = current_text

	def _apply_cine_source(self, source: str, *, preserve_position: bool = True, preferred_gate: int | None = None, preferred_slice: int | None = None):
		# El método Auto ROI es una preferencia global de workflow; se sincroniza
		# al alternar visores para mantener coherencia clínica.
		active_method = self.cine.auto_roi_method()
		active_atten, active_feather = self.cine.intestinal_params()
		active_intestinal_scope = self.cine.intestinal_scope()
		if self.active_cine_source == "compare":
			active_method = self.cine_compare.auto_roi_method()
			active_atten, active_feather = self.cine_compare.intestinal_params()
			active_intestinal_scope = self.cine_compare.intestinal_scope()
		self.cine.set_auto_roi_method(active_method)
		self.cine_compare.set_auto_roi_method(active_method)
		self.cine.set_intestinal_params(active_atten, active_feather)
		self.cine_compare.set_intestinal_params(active_atten, active_feather)
		self.cine.set_intestinal_scope(active_intestinal_scope)
		self.cine_compare.set_intestinal_scope(active_intestinal_scope)

		self._save_active_manual_rois_text()
		has_secondary = self.compare_bundle is not None or self.compare_raw_study is not None
		self.active_cine_source = "compare" if source == "compare" and has_secondary else "primary"
		if self.active_cine_source == "compare":
			autogenerated = bool(self.compare_manual_rois_autogenerated)
		else:
			autogenerated = bool(self.primary_manual_rois_autogenerated)
		self._set_manual_rois_text(self._load_manual_rois_text_for_source(self.active_cine_source), autogenerated=autogenerated)
		main_controls = self.active_cine_source == "primary"
		self.cine.set_controls_visible(main_controls)
		self.cine_compare.set_controls_visible(not main_controls and has_secondary)
		self._refresh_dual_cine_views(
			preserve_position=preserve_position,
			preferred_gate=preferred_gate,
			preferred_slice=preferred_slice,
		)
		self._update_cine_active_border()
		self._clamp_window_to_screen()

	def _clamp_or_maximize(self):
		"""Si la ventana restaurada no entra en la pantalla disponible, abre
		maximizada; si entra pero sobresale, la clama al área visible."""
		if self.isFullScreen() or self.isMaximized():
			return
		screen = self.screen() or QApplication.primaryScreen()
		if screen is None:
			return
		available = screen.availableGeometry()
		geom = self.geometry()
		if geom.width() > available.width() or geom.height() > available.height():
			self.showMaximized()
		else:
			self._clamp_window_to_screen()

	def _clamp_window_to_screen(self):
		"""Evita crecimiento horizontal fuera del monitor activo.

		Algunos cambios de visibilidad/layout (p.ej. alternar cine en asincronia)
		pueden hacer que Qt intente ensanchar la ventana para satisfacer nuevos
		size hints. Este clamp mantiene la geometria dentro del area visible.
		"""
		if self.isFullScreen() or self.isMaximized():
			return
		screen = self.screen() or QApplication.primaryScreen()
		if screen is None:
			return
		available = screen.availableGeometry()
		geom = self.geometry()
		new_w = min(geom.width(), available.width())
		new_h = min(geom.height(), available.height())
		max_x = available.left() + max(0, available.width() - new_w)
		max_y = available.top() + max(0, available.height() - new_h)
		new_x = min(max(geom.x(), available.left()), max_x)
		new_y = min(max(geom.y(), available.top()), max_y)
		if (
			new_w != geom.width()
			or new_h != geom.height()
			or new_x != geom.x()
			or new_y != geom.y()
		):
			self.setGeometry(new_x, new_y, new_w, new_h)

	def _on_cine_source_changed(self, index: int):
		if index < 0:
			return
		source = self.cine_source_combo.itemData(index)
		self._apply_cine_source(str(source or "primary"))

	def _update_roi_adjust_labels(self):
		self.auto_center_gain_label.setText(f"{int(self.auto_center_gain_slider.value())}%")
		self.auto_inner_delta_label.setText(f"{self.auto_inner_delta_slider.value() / 10.0:+.1f} px")
		self.auto_outer_delta_label.setText(f"{self.auto_outer_delta_slider.value() / 10.0:+.1f} px")

	def reset_roi_adjust_deltas(self):
		self.auto_center_gain_slider.setValue(100)
		self.auto_inner_delta_slider.setValue(0)
		self.auto_outer_delta_slider.setValue(0)

	def _update_compare_slice_label(self):
		self.compare_slice_label.setText(f"{int(self.compare_slice_slider.value())}%")

	def _update_compare_window_labels(self):
		self.compare_window_high_label.setText(f"{int(self.compare_window_high_slider.value())}%")
		self.compare_window_low_label.setText(f"{int(self.compare_window_low_slider.value())}%")

	def _update_compare_axes_zoom_label(self):
		self.compare_axes_zoom_label.setText(f"{int(self.compare_axes_zoom_slider.value())}%")

	def _schedule_compare_axes_refresh(self):
		self.compare_axes_refresh_timer.start(180)

	def _on_compare_window_high_change(self, value: int):
		if int(value) <= int(self.compare_window_low_slider.value()):
			self.compare_window_low_slider.blockSignals(True)
			self.compare_window_low_slider.setValue(max(0, int(value) - 1))
			self.compare_window_low_slider.blockSignals(False)
		self._update_compare_window_labels()
		self._schedule_compare_axes_refresh()

	def _on_compare_window_low_change(self, value: int):
		if int(value) >= int(self.compare_window_high_slider.value()):
			self.compare_window_high_slider.blockSignals(True)
			self.compare_window_high_slider.setValue(min(100, int(value) + 1))
			self.compare_window_high_slider.blockSignals(False)
		self._update_compare_window_labels()
		self._schedule_compare_axes_refresh()

	def _on_compare_axes_zoom_changed(self, value: int):
		self._update_compare_axes_zoom_label()
		self._schedule_compare_axes_refresh()

	def _on_global_intestinal_render_toggled(self, checked: bool):
		self._invalidate_output_cache()
		if self.study is not None and self.phase_result is not None:
			self._set_progress(82, "Actualizando salidas visuales con ROI intestinal...")
			self._write_outputs()
			if self.compare_bundle is not None and self.advanced_mode_enabled:
				self._write_outputs_for_bundle(self.compare_bundle, self.compare_output_dir)
				left_label, right_label = self._dual_compare_labels()
				self._compose_dual_tab_images(left_label, right_label)
			self._load_previews()
			self._set_progress(100, "Procesamiento completo")
		state = "ON" if bool(checked) else "OFF"
		self.statusBar().showMessage(f"Atenuación intestinal visual global: {state}")

	# ------------------------------------------------------------------
	# Dropout de cuentas del último gate (ECTb 4.0, Technical Overview 22.8)
	# ------------------------------------------------------------------

	def gate_dropout_enabled(self) -> bool:
		"""True si el usuario dejó activa la corrección de dropout del último gate."""
		return bool(getattr(self, "gate_dropout_check", None) is not None and self.gate_dropout_check.isChecked())

	def _apply_gate_dropout_correction(self, cube, label: str = "", log: bool = True):
		"""Devuelve el cubo con el último gate escalado según ECTb 22.8.

		Se llama justo antes de segmentar y de analizar fase. Si la opción está
		apagada o el déficit es despreciable, devuelve el cubo tal cual (sin
		copiarlo) para no pagar memoria ni tiempo de más.

		Returns
		-------
		(cube, info) — `info` es el dict de `analyze_gate_dropout` (o None si
		el cubo no era 4D / la opción estaba apagada).
		"""
		arr = np.asarray(cube, dtype=np.float64)
		if arr.ndim != 4 or not self.gate_dropout_enabled():
			return arr, None
		try:
			corrected, info = correct_last_gate_dropout(arr)
		except ValueError:
			return arr, None
		suffix = f" [{label}]" if label else ""
		if info.get("applied"):
			if log:
				self._log(f"Dropout último gate{suffix}: {info['message']}")
			return corrected, info
		return arr, info

	def _refresh_gate_dropout_status(self, info: dict | None = None):
		"""Actualiza el cartel de estado del dropout en el sidebar.

		Si no se pasa `info`, lo recalcula sobre el estudio actual. La medición
		es un `sum` sobre el cubo: barata, se puede llamar en cada refresco.
		"""
		label = getattr(self, "gate_dropout_status", None)
		if label is None:
			return
		if info is None:
			cube = getattr(self.study, "cube", None) if self.study is not None else None
			if cube is None or np.asarray(cube).ndim != 4:
				label.setText("Sin estudio gated cargado.")
				label.setStyleSheet("color:#7a7a7a;")
				return
			try:
				info = analyze_gate_dropout(cube)
			except ValueError:
				label.setText("El estudio actual no es un gated 4D.")
				label.setStyleSheet("color:#7a7a7a;")
				return

		enabled = self.gate_dropout_enabled()
		prefix = "" if enabled else "[corrección OFF] "
		label.setText(prefix + str(info.get("message", "")))
		if info.get("clipped"):
			color = "#b03030"
		elif info.get("significant"):
			color = "#1f3b5b" if enabled else "#a06000"
		else:
			color = "#4b7a4b"
		label.setStyleSheet(f"color:{color};")

	def _on_gate_dropout_toggled(self, checked: bool):
		"""Reprocesa en vivo al prender/apagar la corrección.

		El cambio invalida segmentación y fase (el cubo de entrada cambió), así
		que hay que rehacer el pipeline. Se hace con un disparo diferido para
		que la UI pinte el checkbox antes de arrancar el cálculo.
		"""
		self._refresh_gate_dropout_status()
		state = "ON" if bool(checked) else "OFF"
		self.statusBar().showMessage(f"Corrección de dropout del último gate: {state}")
		if self.study is None or not bool(getattr(self.study, "reconstructed", True)):
			return
		self._cache_seg_sig = ""
		self._cache_phase_sig = ""
		self._invalidate_output_cache()
		QTimer.singleShot(0, self.process_current)

	def show_gate_dropout_help(self):
		"""Ventana de ayuda del control de dropout del último gate."""
		QMessageBox.information(
			self,
			"Dropout del último gate",
			"QUÉ ES\n"
			"En un gated SPECT el ciclo R-R se divide en 8 o 16 gates. Como el R-R no es constante "
			"latido a latido, los latidos más cortos terminan antes de llenar el último gate. "
			"Resultado: el último gate acumula sistemáticamente menos cuentas que los demás.\n\n"
			"POR QUÉ IMPORTA\n"
			"• FEVI: la curva de volumen no cierra el ciclo y el volumen telediastólico final "
			"queda subestimado.\n"
			"• Fase: la caída artificial del último punto de la curva de actividad mete un escalón "
			"que corre la fase estimada y ensancha la distribución, así que PSD y Bandwidth salen "
			"más altos de lo real (falso positivo de disincronía).\n\n"
			"QUÉ HACE LA CORRECCIÓN\n"
			"Escala todas las muestras del último gate por un único factor, de forma que su suma "
			"total iguale la del primer gate:\n\n"
			"    factor = cuentas(gate 1) / cuentas(último gate)\n\n"
			"Es una corrección de ganancia global: NO deforma el miocardio, NO mueve bordes, NO "
			"cambia la forma espacial. Solo restituye la estadística perdida.\n\n"
			"ORIGEN\n"
			"Emory Cardiac Toolbox 4.0 — Technical Overview, sección 22.8. Es el mismo paso que "
			"aplica el ECTb antes de calcular volúmenes y fase.\n\n"
			"EL SUPUESTO\n"
			"Se asume que el primer y el último gate son adyacentes en el ciclo (ambos en "
			"telediastole) y que por lo tanto deberían tener cuentas comparables. Es razonable en un "
			"gated normal, pero no exacto: si el último gate tuviera legítimamente algo menos de "
			"cuentas, la corrección mete un error pequeño en sentido contrario. Ese residuo es mucho "
			"menor que el dropout que corrige (10-20% de déficit por R-R vs pocos % de diferencia "
			"fisiológica), por eso el ECTb lo aplica por defecto.\n\n"
			"CUÁNDO APAGARLA\n"
			"• Si el estudio ya viene corregido por la consola: el cartel del sidebar va a mostrar "
			"un dropout de ~0% y la corrección no se aplica sola.\n"
			"• Si querés comparar A/B el efecto de la corrección sobre PSD/BW/FEVI.\n\n"
			"SEGURIDAD\n"
			"Si el último gate quedó con menos de la mitad de las cuentas, el factor se recorta a "
			"2.0 y aparece un aviso en rojo: amplificar más allá de eso solo amplifica ruido, y "
			"conviene revisar la ventana de aceptación del R-R de la adquisición.",
		)

	def _apply_compare_axes_clinical_quick_preset(self):
		self.compare_axes_zoom_slider.setValue(140)
		self.compare_axes_intestinal_mask_check.setChecked(True)
		self.compare_window_high_slider.setValue(98)
		self.compare_window_low_slider.setValue(5)
		self.compare_mask_check.setChecked(True)
		self.compare_axes_cmap_combo.setCurrentText("hot")
		self._schedule_compare_axes_refresh()
		self.statusBar().showMessage("Preset clínico aplicado en comparacion_ejes (zoom 140%, ROI intestino ON, ventana 98/5).")

	def _comparison_gate_index(self) -> int:
		if self.study is None:
			return 0
		return max(0, min(int(self.study.cube.shape[0]) - 1, int(self.compare_gate_spin.value()) - 1))

	def _comparison_fraction(self) -> float:
		return max(0.0, min(1.0, float(self.compare_slice_slider.value()) / 100.0))

	def use_cine_position_for_comparison(self):
		if self.study is None:
			QMessageBox.information(self, "SINCRO", "Primero cargá un estudio.")
			return
		self.compare_gate_spin.setValue(self.cine.current_gate_index() + 1)
		n_slices = max(1, int(self.study.cube.shape[1]) - 1)
		slice_pct = int(round(100.0 * self.cine.current_slice_index() / n_slices))
		self.compare_slice_slider.setValue(slice_pct)
		if self.phase_result is not None:
			self._write_outputs()
			self._load_previews()
			idx = self.tabs.indexOf(self.tabs.findChild(QWidget, "comparacion_ejes"))
			if idx < 0:
				for i in range(self.tabs.count()):
					if self.tabs.tabText(i) == "comparacion_ejes":
						idx = i
						break
			if idx >= 0:
				self.tabs.setCurrentIndex(idx)
		self.statusBar().showMessage("Comparación alineada con gate/slice actuales del cine.")

	def _on_compare_mask_toggled(self, checked: bool):
		self._schedule_compare_axes_refresh()

	def _on_compare_controls_drag_started(self):
		if self.compare_fast_drag_check.isChecked():
			self.compare_interactive_fast_mode = True

	def _on_compare_controls_drag_ended(self):
		if self.compare_interactive_fast_mode:
			self.compare_interactive_fast_mode = False
			self._refresh_compare_axes_panel_now()

	def _advance_compare_axes_frame(self):
		if not self.compare_axes_preview_frames:
			self.compare_axes_cine_timer.stop()
			self.compare_axes_playing = False
			self._update_compare_axes_toggle_text(enabled=False)
			return
		self.compare_axes_preview_index = (int(self.compare_axes_preview_index) + 1) % max(1, len(self.compare_axes_preview_frames))
		self._set_compare_axes_memory_frame(self.compare_axes_preview_index)

	def _set_compare_axes_memory_frame(self, index: int):
		if not self.compare_axes_preview_frames:
			return
		idx = max(0, min(int(index), len(self.compare_axes_preview_frames) - 1))
		self.compare_axes_preview_index = idx
		pix = self.compare_axes_preview_frames[idx]
		self.preview_pixmaps["comparacion_ejes"] = pix
		self.preview_base_sizes["comparacion_ejes"] = pix.size()
		self._apply_preview_zoom("comparacion_ejes")

	def _rgb_frame_to_qpixmap(self, rgb: np.ndarray) -> QPixmap:
		arr = np.ascontiguousarray(rgb, dtype=np.uint8)
		h, w, _ = arr.shape
		qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
		return QPixmap.fromImage(qimg.copy())

	def export_compare_axes_frames_debug(self):
		if not self.compare_axes_preview_frames:
			if self.study is None or self.seg is None:
				QMessageBox.information(self, "SINCRO", "Primero cargá/procesá un estudio para exportar frames.")
				return
			prev_fast = self.compare_interactive_fast_mode
			self.compare_interactive_fast_mode = False
			self._refresh_compare_axes_panel_now()
			self.compare_interactive_fast_mode = prev_fast
		if not self.compare_axes_preview_frames:
			QMessageBox.information(self, "SINCRO", "No hay frames de comparacion_ejes para exportar.")
			return
		stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		folder = os.path.join(self.output_dir, f"_debug_compare_axes_frames_{stamp}")
		os.makedirs(folder, exist_ok=True)
		saved_paths: list[str] = []
		for i, pix in enumerate(self.compare_axes_preview_frames):
			p = os.path.join(folder, f"frame_{i:03d}.png")
			pix.save(p, "PNG")
			saved_paths.append(p)
		static_path = os.path.join(self.output_dir, "comparacion_ejes.png")
		if os.path.exists(static_path):
			QPixmap(static_path).save(os.path.join(folder, "comparacion_ejes_static.png"), "PNG")
		try:
			from PIL import Image
		except Exception:
			Image = None
		if Image is not None and saved_paths:
			try:
				ims = [Image.open(p).convert("RGB") for p in saved_paths]
				ims[0].save(
					os.path.join(folder, "comparacion_ejes_debug.gif"),
					save_all=True,
					append_images=ims[1:],
					duration=max(40, int(self.compare_axes_cine_speed_spin.value())),
					loop=0,
					disposal=2,
					optimize=False,
				)
			except Exception as exc:
				self._log(f"[WARN] No se pudo exportar GIF debug de comparacion_ejes: {exc}")
		self._log(f"Frames comparacion_ejes exportados: {folder}")
		QMessageBox.information(self, "SINCRO", f"Frames exportados en:\n{folder}")

	def _refresh_compare_axes_panel_now(self):
		# Grilla comparacion_ejes DEPRECADA (reemplazada por el Montaje clínico).
		# No se regenera; la tab queda a cargo del montaje.
		return

	def _on_preview_tab_changed(self, index: int):
		if index < 0:
			return
		title = self.tabs.tabText(index)
		tab_name = self._tab_name_from_title(title)
		# Montaje clínico: se renderiza al entrar (ya no hay botón "Ver montaje").
		# El resto de acciones (layout, zoom, gates) ocurre en vivo. Va antes del
		# guard study/seg porque el crudo puede tener cortes sin segmentación.
		# Solo se re-renderiza si algo del montaje cambió desde el último render.
		if tab_name == "comparacion_ejes" and self.cine_crudo_axes_for_export:
			sig = self._montage_signature()
			already = (
				self.cine_crudo_preview_mode == "sa_montage"
				and self.preview_pixmaps.get("comparacion_ejes") is not None
				and getattr(self, "_montage_last_signature", None) == sig
			)
			if already:
				self._apply_preview_zoom("comparacion_ejes")
			else:
				self._show_cine_crudo_sa_montage()
			return
		if self.study is None or self.seg is None:
			return
		if tab_name:
			self._request_lazy_tab_render(tab_name, reason="apertura de pestaña")

	def _refresh_compare_axes_panel(self):
		self._refresh_compare_axes_panel_now()

	def _mutate_manual_rois(self, transform, message: str):
		rois = self._parse_manual_rois()
		new_rois = transform(rois)
		self._sync_manual_rois(new_rois)
		self._log(message)
		self.statusBar().showMessage(message)

	def _on_cine_roi_changed(self, slice_index: int, roi):
		current = self._parse_manual_rois()
		if roi is None:
			current.pop(int(slice_index), None)
		else:
			current[int(slice_index)] = tuple(float(v) for v in roi)
		self._sync_manual_rois(current)
		if self.seg_method.currentText() != "manual":
			self.seg_method.setCurrentText("manual")
			self._log("ROI detectada: se activó Segmentación=manual.")
		self.statusBar().showMessage(f"ROI actualizada en slice {slice_index + 1}. Reprocesá para aplicar cambios.")

	def _on_cine_roi_changed_gate(self, gate_index: int, slice_index: int, roi):
		"""QC por gate: ROI manual específica de un gate. No altera el ROI común por slice."""
		self._invalidate_output_cache()
		self._schedule_gate_roi_recalc()
		msg = (
			f"ROI (gate {int(gate_index) + 1}) actualizada en slice {int(slice_index) + 1}. "
			"FEVI se actualizará con QC por gate."
			if roi is not None
			else f"ROI (gate {int(gate_index) + 1}) borrada en slice {int(slice_index) + 1}."
		)
		self._log(msg)
		self.statusBar().showMessage(msg)

	def _on_cine_compare_roi_changed_gate(self, gate_index: int, slice_index: int, roi):
		self._invalidate_output_cache()
		self._schedule_gate_roi_recalc()
		msg = (
			f"ROI comparativa (gate {int(gate_index) + 1}) actualizada en slice {int(slice_index) + 1}."
			if roi is not None
			else f"ROI comparativa (gate {int(gate_index) + 1}) borrada en slice {int(slice_index) + 1}."
		)
		self._log(msg)
		self.statusBar().showMessage(msg)

	def _schedule_gate_roi_recalc(self):
		"""Agrupa ediciones ROI por gate y refresca FEVI/salidas sin reproceso completo."""
		if self.study is None or self.seg is None or self.phase_result is None:
			return
		self._gate_roi_recalc_timer.start(220)

	def _on_gate_roi_recalc_timeout(self):
		if self.study is None or self.seg is None or self.phase_result is None:
			return
		try:
			self._refresh_summary()
			target_tabs = {"curva_fevi", "panel_funcional_gated", "bullseye_directo"}
			self._write_outputs(target_tabs=target_tabs)
			if self.compare_bundle is not None:
				left_label, right_label = self._dual_compare_labels()
				self._compose_dual_tab_images(left_label, right_label, target_tabs=target_tabs)
			self._load_previews_selected(target_tabs)
			visual_payload = self._collect_visual_signature_payload()
			visual_payload["phase"] = self._cache_phase_sig
			self._cache_output_sig = self._hash_payload(visual_payload)
			self.statusBar().showMessage("FEVI/QC por gate actualizados.")
		except Exception as exc:
			self._log(f"[WARN] No se pudo refrescar FEVI tras editar ROI por gate: {exc}")

	def _on_cine_compare_roi_changed(self, slice_index: int, roi):
		source = self.cine_secondary_source
		if source not in ("primary", "compare"):
			return
		current = self._parse_manual_rois_text(self._load_manual_rois_text_for_source(source))
		if roi is None:
			current.pop(int(slice_index), None)
		else:
			current[int(slice_index)] = tuple(float(v) for v in roi)
		formatted = self._format_manual_rois(current)
		self._save_manual_rois_text_for_source(source, formatted)
		if self.active_cine_source == source:
			self.manual_rois.blockSignals(True)
			self.manual_rois.setPlainText(formatted)
			self.manual_rois.blockSignals(False)
		if self.seg_method.currentText() != "manual":
			self.seg_method.setCurrentText("manual")
			self._log("ROI detectada: se activó Segmentación=manual.")
		self.statusBar().showMessage(f"ROI ({'reposo' if source == 'compare' else 'esfuerzo'}) actualizada en slice {slice_index + 1}.")

	def _on_play_state_changed(self, playing: bool):
		self.statusBar().showMessage("Cine en reproducción" if playing else "Cine en pausa")

	def _sync_cine_compare_playback(self, playing: bool):
		"""Sincroniza el play/pausa del cine_compare con el principal, evitando
		bucle (solo llama toggle si los estados no coinciden)."""
		if getattr(self, "cine_compare", None) is None:
			return
		if playing != self.cine_compare._playing:
			self.cine_compare.toggle_playback()

	def process_auto(self):
		self.seg_method.setCurrentText("auto")
		self.process_current()

	def apply_current_roi_to_all_slices(self):
		if self.study is None:
			QMessageBox.information(self, "SINCRO", "Primero cargá un estudio.")
			return
		current_roi = self.cine.preview.roi()
		if current_roi is None:
			QMessageBox.information(self, "SINCRO", "No hay ROI actual para replicar. Dibujá uno en el cine primero.")
			return
		if not self._is_roi_valid_for_manual(current_roi):
			QMessageBox.information(self, "SINCRO", "El ROI actual no es válido para replicar.")
			return
		manual_rois = self._parse_manual_rois()
		for slice_index in range(self.study.cube.shape[1]):
			manual_rois[slice_index] = tuple(float(v) for v in current_roi)
		self._sync_manual_rois(manual_rois)
		self.seg_method.setCurrentText("manual")
		self._log("ROI replicado a todos los slices; Segmentación=manual activado.")
		self.statusBar().showMessage("ROI replicado a todos los slices.")

	def _apply_reference_auto_adjustment(self, *, adjust_center: bool, adjust_inner: bool, adjust_outer: bool, label: str):
		if self.study is None:
			QMessageBox.information(self, "SINCRO", "Primero cargá un estudio.")
			return
		reference_slice = self.cine.current_slice_index()
		reference_roi = self.cine.preview.roi()
		if not self._is_roi_valid_for_manual(reference_roi):
			QMessageBox.information(
				self,
				"SINCRO",
				"Primero ajustá manualmente el ROI del slice de referencia y luego propagá el ajuste.",
			)
			return

		adjusted_rois = self.cine.build_adjusted_auto_rois(
			reference_slice,
			tuple(float(v) for v in reference_roi),
			adjust_center=adjust_center,
			adjust_inner=adjust_inner,
			adjust_outer=adjust_outer,
			center_gain=float(self.auto_center_gain_slider.value()) / 100.0,
			inner_extra=float(self.auto_inner_delta_slider.value()) / 10.0,
			outer_extra=float(self.auto_outer_delta_slider.value()) / 10.0,
			max_distance=int(self.auto_adjust_range_spin.value()),  # Usar el rango configurado en la propagación si aún no estaba conectado
		)
		if not adjusted_rois:
			QMessageBox.information(
				self,
				"SINCRO",
				"No pude construir Auto ROI de referencia en este estudio. Probá con otro slice o ajustá el umbral visualmente.",
			)
			return

		self._sync_manual_rois(adjusted_rois)
		self.seg_method.setCurrentText("manual")
		message = f"{label} propagado desde slice {reference_slice + 1} a {len(adjusted_rois)} slices."
		self._log(message)
		self.statusBar().showMessage(message)

	def adjust_auto_center_all_slices(self):
		self._apply_reference_auto_adjustment(
			adjust_center=True,
			adjust_inner=False,
			adjust_outer=False,
			label="Ajuste de centro",
		)

	def adjust_auto_inner_all_slices(self):
		self._apply_reference_auto_adjustment(
			adjust_center=False,
			adjust_inner=True,
			adjust_outer=False,
			label="Ajuste de radio interno",
		)

	def adjust_auto_outer_all_slices(self):
		self._apply_reference_auto_adjustment(
			adjust_center=False,
			adjust_inner=False,
			adjust_outer=True,
			label="Ajuste de radio externo",
		)

	def adjust_auto_full_all_slices(self):
		self._apply_reference_auto_adjustment(
			adjust_center=True,
			adjust_inner=True,
			adjust_outer=True,
			label="Ajuste completo de Auto ROI",
		)

	def clear_current_roi(self):
		if self.study is None:
			QMessageBox.information(self, "SINCRO", "Primero cargá un estudio.")
			return
		slice_index = self.cine.current_slice_index()
		manual_rois = self._parse_manual_rois()
		if slice_index not in manual_rois:
			self.statusBar().showMessage(f"No había ROI en el slice {slice_index + 1}.")
			return
		manual_rois.pop(slice_index, None)
		self._sync_manual_rois(manual_rois)
		self.statusBar().showMessage(f"ROI borrado en slice {slice_index + 1}.")
		self._log(f"ROI borrado en slice {slice_index + 1}.")

	def clear_all_rois(self):
		if self.study is None:
			QMessageBox.information(self, "SINCRO", "Primero cargá un estudio.")
			return
		self._sync_manual_rois({})
		self.statusBar().showMessage("Se borraron todos los ROIs.")
		self._log("Se borraron todos los ROIs.")

	def clear_outer_rois(self):
		manual_rois = self._parse_manual_rois()
		if not manual_rois:
			QMessageBox.information(self, "SINCRO", "No hay ROIs para modificar.")
			return
		for slice_index, roi in list(manual_rois.items()):
			cy, cx, r_inner, _r_outer = roi
			manual_rois[slice_index] = (cy, cx, r_inner, float("nan"))
		self._sync_manual_rois(manual_rois)
		self.statusBar().showMessage("Se borraron los ROIs externos.")
		self._log("Se borraron los ROIs externos.")

	def clear_inner_rois(self):
		manual_rois = self._parse_manual_rois()
		if not manual_rois:
			QMessageBox.information(self, "SINCRO", "No hay ROIs para modificar.")
			return
		for slice_index, roi in list(manual_rois.items()):
			cy, cx, _r_inner, r_outer = roi
			manual_rois[slice_index] = (cy, cx, float("nan"), r_outer)
		self._sync_manual_rois(manual_rois)
		self.statusBar().showMessage("Se borraron los ROIs internos.")
		self._log("Se borraron los ROIs internos.")

	def clear_centers(self):
		manual_rois = self._parse_manual_rois()
		if not manual_rois:
			QMessageBox.information(self, "SINCRO", "No hay ROIs para modificar.")
			return
		for slice_index, roi in list(manual_rois.items()):
			cy, cx, r_inner, r_outer = roi
			manual_rois[slice_index] = (float("nan"), float("nan"), r_inner, r_outer)
		self._sync_manual_rois(manual_rois)
		self.statusBar().showMessage("Se borraron los centros de los ROIs.")
		self._log("Se borraron los centros de los ROIs.")

	def reset_current_file(self):
		if self.study is None and not self.file_edit.text().strip():
			QMessageBox.information(self, "SINCRO", "Primero cargá un estudio.")
			return
		self._sync_manual_rois({})
		if self.seg_method.currentText() == "manual":
			self.seg_method.setCurrentText("auto")
		self.cine.stop_playback()
		self.process_current()

	def _stop_all_session_timers(self):
		"""Detiene todos los timers/reproducciones asociados a un estudio."""
		try:
			self.cine.stop_playback()
		except Exception:
			pass
		for timer_name in (
			"polar_cine_timer",
			"cine_crudo_timer",
			"_montage_cine_timer",
			"_montage_refresh_timer",
			"_montage_hq_timer",
			"_montage_recolor_smooth_timer",
			"compare_axes_cine_timer",
			"compare_axes_refresh_timer",
			"_deferred_hq_timer",
			"_gate_roi_recalc_timer",
		):
			timer = getattr(self, timer_name, None)
			if timer is not None:
				try:
					timer.stop()
				except Exception:
					pass

	def _reset_session_data(self):
		"""Restablece TODO el estado de datos en memoria a como está al abrir SINCRO.

		Limpia el estudio principal, la comparación, la fase/segmentación, los ROIs
		manuales, todo el pipeline de cine crudo (proyecciones, corrección de
		movimiento, reconstrucción, montajes) y los trabajos diferidos. NO toca
		widgets ni preferencias de UI (que en un arranque fresco se recargan de
		QSettings)."""
		self._stop_all_session_timers()
		self._clear_compare_state()
		# Sesión dual: limpiar AMBAS etapas (los `= None` legacy de abajo solo
		# alcanzan la etapa del slot activo vía properties).
		self._dual_session().clear()

		# --- Estudio principal y derivados de fase/segmentación ---
		self.study = None
		self.axis_companions = {}
		self.seg = None
		self.seg_ring_base = None
		self.intestinal_subtraction_info = None
		self.phase_result_raw = None
		self.phase_result = None
		self.metrics_raw = None
		self.metrics = None
		self.phase_qc = None
		self.aha = None
		self.phase_by_seg = None
		self.territory = None

		# --- ROIs manuales y centros ---
		self.primary_manual_rois_text = ""
		self.compare_manual_rois_text = ""
		self.primary_manual_rois_autogenerated = False
		self.compare_manual_rois_autogenerated = False
		self._programmatic_manual_rois_update = False
		self.manual_center_per_slice = {}

		# --- Previews y cine polar ---
		self.polar_view_mode = "perfusion"
		self.polar_cine_preview_frames = []
		self.polar_cine_preview_index = 0
		self.polar_cine_playing = False
		self._polar_perf_cart_cache = None
		self._polar_cine_cart_cache = None
		self.preview_pixmaps.clear()
		self.preview_base_sizes.clear()

		# --- Cine crudo (proyecciones / movimiento / reconstrucción / montaje) ---
		self.cine_crudo_frames = []
		self.cine_crudo_index = 0
		self.cine_crudo_playing = False
		self.cine_crudo_direction = 1
		self._stop_montage_cine()
		if hasattr(self, "cine_crudo_scatter_check"):
			self.cine_crudo_scatter_check.setEnabled(False)
			self.cine_crudo_scatter_check.setChecked(False)
			self.cine_crudo_scatter_k_spin.setEnabled(False)
		if getattr(self, "cine_crudo_diff_check", None) is not None:
			self.cine_crudo_diff_check.setEnabled(False)
			self.cine_crudo_diff_check.setChecked(False)
		self.cine_crudo_matrix_txt = ""
		self.cine_crudo_seed = None
		self.cine_crudo_seed_compare = None
		self.cine_crudo_seed_mode = False
		self._cine_crudo_active_stage = "stress"
		self._dual_pipeline_manual_stage_override = None
		self.cine_crudo_band_upper = None
		self.cine_crudo_band_lower = None
		self.cine_crudo_compare_line_y = None
		self._cine_crudo_drag_marker = None
		self._cine_crudo_hover_marker = None
		self._cine_crudo_last_drag_refresh = 0.0
		self.cine_crudo_ref_index = None
		self.cine_crudo_ref_index_compare = None
		self.cine_crudo_corrected_projections = None
		self.cine_crudo_corrected_projections_compare = None
		self.cine_crudo_motion_result = None
		self.cine_crudo_motion_result_compare = None
		self.cine_crudo_recon_result = None
		self.cine_crudo_recon_result_phase = None
		self.cine_crudo_raw_study_for_recon = None
		self._cine_crudo_recon_stage = "stress"
		self._reorient_locked_voi = None
		self._reorient_locked_stage = None
		self._reorient_seed = None
		self._reorient_seed_stage = None
		self.cine_crudo_recon_study = None
		self.cine_crudo_cut_study = None
		self.cine_crudo_cut_source_label = ""
		self.cine_crudo_axes_for_export = {}
		self.cine_crudo_axes_for_export_ungated = {}
		self.cine_crudo_axes_for_export_stress = {}
		self.cine_crudo_axes_for_export_rest = {}
		self.cine_crudo_axes_for_export_ungated_stress = {}
		self.cine_crudo_axes_for_export_ungated_rest = {}
		self.cine_crudo_axes_for_export_mf = {}
		self.cine_crudo_axes_for_export_mf_stress = {}
		self.cine_crudo_axes_for_export_mf_rest = {}
		self.cine_crudo_rest_source_label = ""
		self.cine_crudo_cut_thickness_mm = 0.0
		self.cine_crudo_cut_thickness_mm_rest = 0.0
		self._cine_crudo_cut_limits_by_stage = {
			"stress": {"base_1": 1, "apex_1": None},
			"rest": {"base_1": 1, "apex_1": None},
		}
		self.cine_crudo_montage_crop_mode = "limits"
		self.cine_crudo_montage_template = "nueve"
		self.cine_crudo_montage_cut_zoom = 1.0
		self.cine_crudo_stripe_start = {"SA": 1, "VLA": 1, "HLA": 1}
		self.cine_crudo_stripe_count = {"SA": 999, "VLA": 999, "HLA": 999}
		self.cine_crudo_stripe_start_by_stage = {
			"ESFUERZO": {"SA": 1, "VLA": 1, "HLA": 1},
			"REPOSO": {"SA": 1, "VLA": 1, "HLA": 1},
		}
		self.cine_crudo_rest_offset = {"SA": 0, "VLA": 0, "HLA": 0}
		self.cine_crudo_gate_from = 1
		self.cine_crudo_gate_to = 1
		self._montage_drag_axis = None
		self._montage_drag_mode = None
		self._montage_drag_start_x = None
		self._montage_drag_start_off = 0
		self._montage_drag_start_gate = 1
		self._montage_drag_selection_key = None
		self._montage_focus_selection_key = "ESFUERZO:SA"
		self.cine_crudo_focused_stripe = "ESFUERZO:SA"
		self._montage_render_meta = {}
		self.cine_crudo_selected_stripe = "SA"
		self.cine_crudo_selected_stripes = {"ESFUERZO:SA"}
		self.cine_crudo_focused_stripe = "ESFUERZO:SA"
		self.cine_crudo_preview_mode = None
		self._last_cine_crudo_preview_mode = None
		self._cine_crudo_dual_render_meta = {}
		self._cine_crudo_cut_limits_meta = None
		self._cuts_qc_pix_by_stage = {}
		self._preview_pan_active = False
		self._preview_pan_anchor = None

		# --- Comparación de ejes ---
		self.compare_axes_preview_frames = []
		self.compare_axes_preview_index = 0
		self.compare_axes_playing = False
		self.compare_interactive_fast_mode = False

		# --- Trabajos diferidos (HQ / lazy render) ---
		self._deferred_hq_job = ""
		self._deferred_compare_bundle = None
		self._deferred_compare_left_label = ""
		self._deferred_compare_right_label = ""
		self._deferred_hq_running = False
		self._deferred_hq_generation = int(getattr(self, "_deferred_hq_generation", 0)) + 1
		self._lazy_render_pending_tabs = set()

		# --- Caché de firmas / rutas ---
		self._output_study_path_override = ""
		self._last_primary_path = ""
		self._cache_study_sig = ""
		self._cache_seg_sig = ""
		self._cache_phase_sig = ""
		self._invalidate_output_cache()
		# Flujo ida-y-vuelta: estudio nuevo ⇒ pila de undo y estados de pasos a cero
		# (evita que un Ctrl+Z restaure atributos de otro estudio).
		if hasattr(self, "pipeline_history"):
			self.pipeline_history.reset()

	def restart_workspace_state(self):
		self._reset_session_data()
		self.file_edit.clear()
		self._sync_manual_rois({})
		self.manual_rois.clear()
		self._push_manual_centers_to_cine()
		if self.seg_method.currentText() == "manual":
			self.seg_method.setCurrentText("auto")
		self.summary_clinical.clear()
		self.summary_technical.clear()
		self.summary_executive.clear()
		self._refresh_readonly_results_panel()
		for movie in list(self.preview_movies.values()):
			movie.stop()
		self.preview_movies.clear()
		self.preview_pixmaps.clear()
		for name, label in self.preview_labels.items():
			self.preview_zoom[name] = self._default_preview_zoom(name)
			label.clear()
			label.setText("Sin procesar")
			if name in self.preview_zoom_labels:
				self.preview_zoom_labels[name].setText(f"{int(self.preview_zoom[name] * 100)}%")
		try:
			self.gate_dropout_status.setText("Sin estudio cargado.")
		except Exception:
			pass
		self.cine.set_cube(None)
		self._refresh_cine_source_selector()
		self._progress_bar.setValue(0)
		self._progress_bar.setFormat("Listo")
		self.log_box.clear()
		self._log("RESTART: sesión limpia, lista para cargar estudios nuevos.")
		self._last_primary_path = ""
		self.statusBar().showMessage("Sesión reiniciada")

	def _hash_payload(self, payload: dict) -> str:
		blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
		return hashlib.sha1(blob.encode("utf-8")).hexdigest()

	def _invalidate_output_cache(self):
		self._cache_output_sig = ""
		self._cache_tab_output_sigs.clear()

	def _log_timing_if_slow(self, label: str, t0: float, *, threshold_sec: float = 0.5) -> float:
		elapsed = float(perf_counter() - float(t0))
		if elapsed >= float(threshold_sec) and bool(self.profile_timing_check.isChecked()):
			self._log(f"[TIEMPO] {label}: {elapsed:.2f}s")
		return elapsed

	def _build_study_signature(self, path: str) -> str:
		full = os.path.abspath(path)
		try:
			st = os.stat(full)
			return f"{full}|{int(st.st_mtime_ns)}|{int(st.st_size)}"
		except OSError:
			return f"{full}|missing"

	def _serialize_manual_rois(self, rois: dict[int, tuple[float, float, float, float]]) -> list[list[float]]:
		serial: list[list[float]] = []
		for sidx in sorted(rois.keys()):
			cy, cx, rin, rout = rois[sidx]
			serial.append([int(sidx), float(cy), float(cx), float(rin), float(rout)])
		return serial

	def _intestinal_signature_for_widget(self, cine_widget: CineWidget | None) -> dict:
		if cine_widget is None:
			return {
				"global_render": bool(self.global_intestinal_render_check.isChecked()),
				"enabled": False,
				"atten": 0,
				"feather": 0,
				"scope": "slice",
				"poly_s": "none",
				"poly_g": "none",
				"mode": "attenuate",
				"bg_method": "idw",
				"ref_s": "none",
				"ref_g": "none",
			}
		atten, feather = cine_widget.intestinal_params()
		scope = cine_widget.intestinal_scope()
		enabled = cine_widget.intestinal_apply_enabled()
		poly_slice = getattr(cine_widget, "_intestinal_roi_polygons", {}) or {}
		poly_gate = getattr(cine_widget, "_intestinal_roi_polygons_by_gate", {}) or {}
		try:
			poly_s_items = sorted((int(k), [(float(a), float(b)) for a, b in (v or [])]) for k, v in poly_slice.items())
		except Exception:
			poly_s_items = []
		try:
			poly_g_items = sorted(((int(g), int(s)), [(float(a), float(b)) for a, b in (v or [])]) for (g, s), v in poly_gate.items())
		except Exception:
			poly_g_items = []
		ref_slice = getattr(cine_widget, "_intestinal_ref_polygons", {}) or {}
		ref_gate = getattr(cine_widget, "_intestinal_ref_polygons_by_gate", {}) or {}
		try:
			ref_s_items = sorted(
				(int(k), [[(float(a), float(b)) for a, b in (poly or [])] for poly in (v or [])])
				for k, v in ref_slice.items()
			)
		except Exception:
			ref_s_items = []
		try:
			ref_g_items = sorted(
				((int(g), int(s)), [[(float(a), float(b)) for a, b in (poly or [])] for poly in (v or [])])
				for (g, s), v in ref_gate.items()
			)
		except Exception:
			ref_g_items = []
		return {
			"global_render": bool(self.global_intestinal_render_check.isChecked()),
			"enabled": bool(enabled),
			"atten": int(atten),
			"feather": int(feather),
			"scope": str(scope),
			"poly_s": self._hash_payload({"items": poly_s_items}) if poly_s_items else "none",
			"poly_g": self._hash_payload({"items": poly_g_items}) if poly_g_items else "none",
			"mode": str(cine_widget.intestinal_mode()) if hasattr(cine_widget, "intestinal_mode") else "attenuate",
			"bg_method": (
				str(cine_widget.intestinal_background_method())
				if hasattr(cine_widget, "intestinal_background_method")
				else "idw"
			),
			"ref_s": self._hash_payload({"items": ref_s_items}) if ref_s_items else "none",
			"ref_g": self._hash_payload({"items": ref_g_items}) if ref_g_items else "none",
		}

	def _gate_roi_signature_for_widget(self, cine_widget: CineWidget | None) -> dict:
		if cine_widget is None:
			return {"per_gate_mode": False, "items": "none"}
		state = cine_widget.gate_roi_state() if hasattr(cine_widget, "gate_roi_state") else {}
		per_gate_mode = bool(state.get("per_gate_mode", False)) if isinstance(state, dict) else False
		items = state.get("gate_rois", []) if isinstance(state, dict) else []
		norm_items = []
		for item in items or []:
			try:
				g = int(item.get("gate"))
				s = int(item.get("slice"))
				roi = [float(v) for v in (item.get("roi") or [])[:4]]
			except Exception:
				continue
			if len(roi) == 4:
				norm_items.append((g, s, roi[0], roi[1], roi[2], roi[3]))
		norm_items.sort()
		return {
			"per_gate_mode": per_gate_mode,
			"items": self._hash_payload({"items": norm_items}) if norm_items else "none",
		}

	def _collect_visual_signature_payload(self) -> dict:
		return {
			"visual_style": str(self.visual_style_combo.currentText()),
			"polar_rotation_deg": int(self.polar_rotation_spin.value()),
			"polar_perf_smooth_method": str(self.polar_perf_smooth_method_combo.currentText()),
			"polar_perf_smooth_strength": float(self.polar_perf_smooth_strength_spin.value()),
			"polar_cine_speed_ms": int(self.polar_cine_speed_spin.value()),
			"polar_compare_math_op": str(self.polar_compare_math_combo.currentText()),
			"polar_compare_math_a": str(self.polar_compare_term_a_combo.currentText()),
			"polar_compare_math_b": str(self.polar_compare_term_b_combo.currentText()),
			"export_polar_mp4": bool(self.export_polar_mp4_check.isChecked()),
			"realtime_deferred_render": bool(self.realtime_deferred_render_check.isChecked()),
			"report_cmap_slices": str(self.report_cmap_slices.currentText()),
			"report_cmap_axes": str(self.report_cmap_axes.currentText()),
			"report_cmap_compare": str(self.report_cmap_compare.currentText()),
			"report_cmap_panel_axes": str(self.report_cmap_panel_axes.currentText()),
			"report_cmap_phase": str(self.report_cmap_phase.currentText()),
			"report_cmap_polar_clinico": str(self.report_cmap_polar_clinico.currentText()),
			"report_cmap_amp": str(self.report_cmap_amp.currentText()),
			"report_cmap_bullseye": str(self.report_cmap_bullseye.currentText()),
			"report_cmap_polar_perf": str(self.report_cmap_polar_perf.currentText()),
			"compare_active": bool(self.compare_bundle is not None),
			"compare_axes_zoom_pct": int(self.compare_axes_zoom_slider.value()),
			"compare_axes_use_intestinal_mask": bool(self.compare_axes_intestinal_mask_check.isChecked()),
			"global_intestinal_render": bool(self.global_intestinal_render_check.isChecked()),
			"intestinal_primary": self._intestinal_signature_for_widget(self.cine),
			"intestinal_compare": self._intestinal_signature_for_widget(self.cine_compare),
			"gate_roi_primary": self._gate_roi_signature_for_widget(self.cine),
			"gate_roi_compare": self._gate_roi_signature_for_widget(self.cine_compare),
		}

	def _apply_intestinal_mask_to_cube(self, cube: np.ndarray, cine_widget: CineWidget | None, *, require_global_visual: bool = True) -> np.ndarray:
		arr = np.asarray(cube, dtype=np.float64)
		if arr.ndim != 4 or cine_widget is None:
			return arr
		if require_global_visual and not bool(self.global_intestinal_render_check.isChecked()):
			return arr
		if not cine_widget.intestinal_apply_enabled():
			return arr
		out = np.array(arr, dtype=np.float64, copy=True)
		for g in range(int(out.shape[0])):
			out[g] = cine_widget.apply_intestinal_mask_to_gate_volume(out[g], gate_index=g)
		return out

	def _apply_intestinal_subtraction_to_cube(
		self, cube: np.ndarray, cine_widget: CineWidget | None
	) -> tuple[np.ndarray, dict | None]:
		"""Sustracción de fondo intestinal (modo 'subtract') sobre el cubo completo.

		A diferencia de la atenuación porcentual, que solo sirve para que el Auto ROI
		no se enganche con el asa, esta corrección **sí** tiene que alimentar el
		análisis de fase: restar la componente DC del intestino es lo único que
		mejora la amplitud relativa que mira el filtro de amplitud. Ver el docstring
		de `core.intestinal_subtraction` para el desarrollo.
		"""
		arr = np.asarray(cube, dtype=np.float64)
		if arr.ndim != 4 or cine_widget is None:
			return arr, None
		if not cine_widget.intestinal_apply_enabled():
			return arr, None
		if cine_widget.intestinal_mode() != "subtract":
			return arr, None

		shape = (int(arr.shape[2]), int(arr.shape[3]))
		n_slices = int(arr.shape[1])
		targets = cine_widget.intestinal_target_weights(shape, n_slices)
		if not targets:
			return arr, None
		references = cine_widget.intestinal_reference_masks(shape, n_slices)
		method = (
			cine_widget.intestinal_background_method()
			if hasattr(cine_widget, "intestinal_background_method")
			else "idw"
		)
		return apply_intestinal_subtraction(arr, targets, references, method=method)

	def _log_intestinal_subtraction(self, info: dict | None):
		if not info:
			return
		message = str(info.get("message") or "").strip()
		if message:
			self._log(message)
		method = str(info.get("method") or "").strip().lower()
		if method:
			self._log(
				"Fondo estimado por media simple de las referencias (nivel constante)."
				if method == "mean"
				else "Fondo estimado por interpolación de distancia inversa entre referencias."
			)
		per_slice = info.get("per_slice") or {}
		mejoras = [
			(int(s), float(d.get("rel_amp_before") or 0.0), float(d.get("rel_amp_after") or 0.0))
			for s, d in per_slice.items()
			if d.get("applied") and np.isfinite(d.get("rel_amp_after", np.nan))
		]
		if mejoras:
			# Control de calidad: lo que queda tras restar TIENE que latir. Si la
			# amplitud relativa no sube, ahí había intestino, no miocardio
			# recuperable, y la corrección no sirvió.
			subieron = sum(1 for _s, a, b in mejoras if b > a)
			detalle = ", ".join(f"corte {s}: {a:.3f}→{b:.3f}" for s, a, b in sorted(mejoras)[:6])
			self._log(
				f"QC sustracción de fondo: amplitud relativa mejoró en {subieron}/{len(mejoras)} corte(s) "
				f"({detalle}). Si no mejora, lo que había ahí era fondo y no miocardio recuperable."
			)
		if info.get("slices_oversubtracted"):
			self._log(
				"ADVERTENCIA: posible sobre-sustracción. Revisá el nivel de las ROI de referencia: "
				"restar de más fabrica un defecto inferior que no existe."
			)

	def _intestinal_export_stamp_text(self, cine_widget: CineWidget | None = None) -> str:
		widget = cine_widget
		if widget is None:
			widget = getattr(self, "_output_cine_widget_override", None)
		if widget is None:
			widget = self.cine
		global_on = bool(self.global_intestinal_render_check.isChecked())
		local_on = bool(widget is not None and widget.intestinal_apply_enabled())
		applied = bool(global_on and local_on)
		atten_pct = 0
		feather_px = 0
		if widget is not None:
			atten_pct, feather_px = widget.intestinal_params()
		return (
			f"ROI intestino | global {'ON' if global_on else 'OFF'} | visor {'ON' if local_on else 'OFF'} | "
			f"aplicado {'ON' if applied else 'OFF'} | atten {int(atten_pct)}% | feather {int(feather_px)}px"
		)

	def _stamp_export_figure(self, fig, cine_widget: CineWidget | None = None):
		try:
			fig.text(
				0.995,
				0.004,
				self._intestinal_export_stamp_text(cine_widget),
				ha="right",
				va="bottom",
				fontsize=7.2,
				color="#f8fafc",
				bbox=dict(boxstyle="round,pad=0.22", facecolor="black", edgecolor="#334155", alpha=0.60),
			)
		except Exception:
			pass

	def _annotate_phase_metrics(self, metrics: dict, phase_result, amp_filter: float, label: str) -> dict:
		out = dict(metrics or {})
		out["amp_filter"] = round(float(amp_filter), 2)
		out["amp_label"] = str(label)
		out["n_voxels_kept"] = int(getattr(phase_result, "n_voxels_kept", out.get("n_voxels", 0)))
		out["n_voxels_total"] = int(getattr(phase_result, "n_voxels_total", out.get("n_voxels", 0)))
		return out

	def _build_phase_qc(self, raw_result, clinical_result, raw_metrics: dict, clinical_metrics: dict) -> dict:
		if raw_result is None or clinical_result is None:
			return {}
		raw_phases = np.asarray(getattr(raw_result, "phases_deg", []), dtype=np.float64)
		raw_amps = np.asarray(getattr(raw_result, "amplitudes", []), dtype=np.float64)
		if raw_phases.size == 0 or raw_amps.size != raw_phases.size:
			return {}
		clinical_filter = float(getattr(clinical_result, "amplitude_threshold_frac", CLINICAL_PHASE_AMP_FILTER_DEFAULT))
		amp_max = float(np.nanmax(raw_amps)) if raw_amps.size else 0.0
		low_amp = raw_amps < (clinical_filter * amp_max) if amp_max > 0.0 else np.zeros(raw_amps.shape, dtype=bool)
		clinical_mean = float(clinical_metrics.get("mean_phase", raw_metrics.get("mean_phase", 0.0)))
		centered = (raw_phases - clinical_mean + 180.0) % 360.0 - 180.0
		late_tail = np.abs(centered) > LOW_CONFIDENCE_TAIL_DEG
		low_tail = low_amp & late_tail
		low_tail_pct = float(np.mean(low_tail) * 100.0) if raw_phases.size else 0.0
		low_tail_n = int(np.count_nonzero(low_tail))
		class_changed = str(raw_metrics.get("classification", "")) != str(clinical_metrics.get("classification", ""))
		warn = bool(low_tail_pct >= LOW_CONFIDENCE_TAIL_WARN_PCT or class_changed)
		return {
			"raw_filter": float(RAW_PHASE_QC_AMP_FILTER),
			"clinical_filter": round(clinical_filter, 2),
			"raw_classification": str(raw_metrics.get("classification", "N/D")),
			"clinical_classification": str(clinical_metrics.get("classification", "N/D")),
			"class_changed": class_changed,
			"low_confidence_tail_pct": round(low_tail_pct, 1),
			"low_confidence_tail_n": low_tail_n,
			"raw_voxels": int(getattr(raw_result, "n_voxels_kept", raw_phases.size)),
			"clinical_voxels": int(getattr(clinical_result, "n_voxels_kept", 0)),
			"total_voxels": int(getattr(raw_result, "n_voxels_total", raw_phases.size)),
			"phase_passenger": bool(getattr(self, "phase_used_passenger", False)),
			"warn": warn,
		}

	def _phase_qc_note(self, qc: dict | None = None) -> str:
		qc = qc or self.phase_qc or {}
		if not qc:
			return ""
		parts = []
		if qc.get("phase_passenger"):
			parts.append("fase sobre FBP (pasajero)")
		else:
			parts.append("fase sobre volumen VISIBLE (sin pasajero)")
		if qc.get("class_changed"):
			parts.append(f"cambio {qc.get('raw_classification')}→{qc.get('clinical_classification')}")
		if float(qc.get("low_confidence_tail_pct", 0.0)) >= LOW_CONFIDENCE_TAIL_WARN_PCT:
			parts.append(f"cola baja amplitud {qc.get('low_confidence_tail_pct')}%")
		return "; ".join(parts)

	def _attach_robustness_metrics(self):
		if self.metrics is None or self.phase_result is None or self.phase_by_seg is None or self.study is None or self.seg is None:
			return
		self.metrics["mode"] = "voxel"
		self.metrics["segmental_aha"] = calculate_segmental_metrics(self.phase_by_seg)
		self.metrics["bootstrap"] = bootstrap_phase_metrics(
			self.phase_result.phases_deg,
			n_iter=500,
			sample_frac=0.80,
			seed=20260720,
		)
		self.metrics["roi_sensitivity"] = roi_sensitivity_analysis(
			self.study.cube,
			self.seg,
			harmonics=int(self.harmonics_spin.value()),
			amplitude_threshold_frac=float(self.phase_threshold_spin.value()),
			normalize_reference=bool(self.normalize_check.isChecked()),
			delta_px=1.0,
		)

	def process_current(self):
		path = self.file_edit.text().strip()
		if not path:
			QMessageBox.warning(self, "SINCRO", "Seleccioná un archivo DICOM primero.")
			return
		if not os.path.exists(path):
			QMessageBox.warning(self, "SINCRO", f"No existe el archivo:\n{path}")
			return

		try:
			t_total = perf_counter()
			primary_abs = os.path.abspath(path)
			preserved_compare_path = ""
			if self.compare_bundle is not None:
				preserved_compare_path = str(self.compare_bundle.get("path", "") or "").strip()
			if self._last_primary_path and os.path.abspath(self._last_primary_path) != primary_abs:
				# Si cambia el estudio primario, se limpia el contexto compare previo.
				self._clear_compare_state()
				preserved_compare_path = ""
			if preserved_compare_path and os.path.abspath(preserved_compare_path) == primary_abs:
				preserved_compare_path = ""
			study_sig = self._build_study_signature(path)
			reuse_study = self.study is not None and self._cache_study_sig == study_sig
			if reuse_study:
				self._set_progress(12, "Reutilizando DICOM en memoria...")
				self._log("Cache: estudio sin cambios, se reutiliza carga DICOM.")
			else:
				t_stage = perf_counter()
				self._set_progress(5, "Cargando DICOM...")
				self._log(f"Cargando: {path}")
				self.study = dicom_loader.load(path, verbose=False)
				self._set_progress(15, "Series originales...")
				self.axis_companions = self._load_axis_companions(path)
				self._log_timing_if_slow("Carga DICOM + series compañeras", t_stage)
				self._cache_study_sig = study_sig
				self._cache_seg_sig = ""
				self._cache_phase_sig = ""
				self._invalidate_output_cache()
				self._preload_acquisition_ecg()
				# Estudio nuevo en memoria ⇒ resetear pila de undo y estados de pasos.
				self.pipeline_history.reset()
			# --- Modo crudo: proyecciones (no reconstruido) → panel QC + cine + gating ---
			self._apply_gated_controls_state()
			if not bool(getattr(self.study, "reconstructed", True)):
				self._handle_raw_projections_loaded(path, t_total)
				return
			if int(np.asarray(self.study.cube).shape[0]) < 3:
				self._log("Estudio reconstruido sin gatillado suficiente (<3 gates): FEVI/asincronía/fase no disponibles.")
				QMessageBox.information(
					self, "SINCRO",
					"Estudio sin gatillado suficiente (<3 gates).\n\n"
					"FEVI, asincronía y análisis de fase no están disponibles para estudios ungated.\n"
					"Reconstrucción, cine, QC y NITIDA sí están disponibles."
				)
				return
			self.compare_gate_spin.setRange(1, max(1, int(self.study.cube.shape[0])))
			self.compare_gate_spin.setValue(max(1, int(self.study.cube.shape[0] // 2) + 1))
			if self.axis_companions:
				loaded = ", ".join(sorted(self.axis_companions.keys()))
				self._log(f"Series originales detectadas para comparación: {loaded}.")
			if not self.preset_patient_edit.text().strip():
				self._refresh_presets_for_current_patient()

			seg_method = str(self.seg_method.currentText())
			roi_text_autogenerated = bool(self.primary_manual_rois_autogenerated)
			self.primary_manual_rois_text = self.manual_rois.toPlainText()
			parsed_rois = self._parse_manual_rois_text(self.primary_manual_rois_text)
			valid_rois = {
				slice_index: roi
				for slice_index, roi in parsed_rois.items()
				if self._is_roi_valid_for_manual(roi)
			}
			if len(valid_rois) != len(parsed_rois):
				self._log("Se ignoraron ROIs incompletas o inválidas.")
			parsed_rois = valid_rois
			if roi_text_autogenerated and seg_method != "manual":
				parsed_rois = {}
			if parsed_rois and seg_method != "manual":
				seg_method = "manual"
				self.seg_method.setCurrentText("manual")
				self._log("Se detectaron ROIs manuales: cambiando Segmentación a manual.")
			if seg_method == "manual" and not parsed_rois:
				QMessageBox.warning(self, "SINCRO", "Modo manual activo pero no hay ROIs definidos. Dibujá ROI o cambiá a auto/threshold.")
				return
			manual_rois = parsed_rois if seg_method == "manual" else None
			# Corrección de dropout del último gate ANTES de segmentar y de analizar
			# fase: si no, el déficit del último gate contamina la máscara y la FFT.
			cube_corrected, dropout_info = self._apply_gate_dropout_correction(self.study.cube, "principal")
			self._refresh_gate_dropout_status(dropout_info)
			# La sustracción de fondo intestinal va ANTES de separar los cubos y
			# alimenta a los dos: a diferencia de la atenuación porcentual, restar
			# la componente DC del intestino es lo único que mejora la amplitud
			# relativa que evalúa el filtro de amplitud de la fase.
			cube_corrected, intestinal_sub_info = self._apply_intestinal_subtraction_to_cube(
				cube_corrected, self.cine
			)
			self.intestinal_subtraction_info = intestinal_sub_info
			self._log_intestinal_subtraction(intestinal_sub_info)
			cube_for_segmentation = self._apply_intestinal_mask_to_cube(cube_corrected, self.cine, require_global_visual=False)
			cube_for_analysis = cube_corrected
			# Pasajero de fase: si el estudio trae un cubo FBP paralelo (cube_phase),
			# la FASE se calcula sobre él (los límites normales Emory/Xeleris están
			# calibrados sobre FBP-Butterworth, no sobre NÍTIDA/RR). La SEGMENTACIÓN
			# y la máscara siguen saliendo del cubo visible. Se aplican las MISMAS
			# correcciones (dropout + sustracción intestinal) al pasajero.
			phase_passenger_active = False
			study_cube_phase = getattr(self.study, "cube_phase", None)
			if study_cube_phase is not None:
				try:
					cube_phase_base = np.asarray(study_cube_phase, dtype=np.float64)
					if cube_phase_base.shape == np.asarray(self.study.cube).shape:
						cube_phase_corr, _ = self._apply_gate_dropout_correction(
							cube_phase_base, "fase (pasajero FBP)", log=False
						)
						cube_phase_corr, _ = self._apply_intestinal_subtraction_to_cube(
							cube_phase_corr, self.cine
						)
						cube_for_analysis = cube_phase_corr
						phase_passenger_active = True
						self._log("Fase calculada sobre pasajero FBP (cube_phase).")
					else:
						self._log(f"[FASE][WARN] Pasajero FBP descartado en análisis: shape {cube_phase_base.shape} != visible {np.asarray(self.study.cube).shape}. Fase sobre volumen VISIBLE.")
				except Exception as exc:
					phase_passenger_active = False
					self._log(f"[FASE][WARN] Pasajero FBP inutilizable en análisis: {exc}. Fase sobre volumen VISIBLE.")
			elif getattr(self, "cine_crudo_cut_study", None) is self.study:
				self._log("[FASE][WARN] El estudio promovido desde el crudo NO trae cube_phase: fase sobre volumen VISIBLE (filtro-dependiente).")
			self.phase_used_passenger = bool(phase_passenger_active)
			intestinal_sig_primary = self._intestinal_signature_for_widget(self.cine)
			intestinal_sig_primary["global_render"] = "ignored_for_segmentation"
			seg_payload = {
				"study": study_sig,
				"method": seg_method,
				"roi_source": self.roi_source(),
				"cavity_center": bool(self.cavity_center_enabled()),
				"threshold": round(float(self.threshold_spin.value()), 5),
				"sigma": round(float(self.sigma_spin.value()), 5),
				"gate_dropout": bool(self.gate_dropout_enabled()),
				"intestinal": intestinal_sig_primary,
				"manual_rois": self._serialize_manual_rois(manual_rois or {}),
				"manual_center": sorted(
					(int(s), round(float(c[0]), 3), round(float(c[1]), 3))
					for s, c in self.manual_center_per_slice.items() if c is not None
				),
			}
			seg_sig = self._hash_payload(seg_payload)
			if self.seg is None or seg_sig != self._cache_seg_sig:
				t_stage = perf_counter()
				self._set_progress(30, "Segmentando miocardio...")
				center_override = self._manual_center_override_array(cube_for_segmentation.shape[1])
				self.seg = segment_myocardium(
					cube_for_segmentation,
					method=seg_method,
					threshold_frac=float(self.threshold_spin.value()),
					smooth_sigma=float(self.sigma_spin.value()),
					manual_rois=manual_rois,
					refine_cavity_center=self.cavity_center_enabled(),
					center_override_per_slice=center_override,
				)
				self.seg_ring_base = self.seg
				self._log_cavity_center_shift(self.seg)
				if self.roi_source() == "ectb_wall":
					self._set_progress(38, "Reemplazando ROI por Contornos Irregulares...")
					self._apply_ectb_wall_segmentation(cube_for_segmentation)
				self._cache_seg_sig = seg_sig
				self._cache_phase_sig = ""
				self._invalidate_output_cache()
				self._log_timing_if_slow("Segmentación", t_stage)
			else:
				self._set_progress(30, "Segmentación sin cambios (cache)...")
				self._log("Cache: segmentación reutilizada.")
			self._mark_step_done("segment", self._cache_seg_sig)

			clinical_amp_filter = float(self.phase_threshold_spin.value())
			phase_payload = {
				"seg": self._cache_seg_sig,
				"harmonics": int(self.harmonics_spin.value()),
				"raw_amp_filter": round(float(RAW_PHASE_QC_AMP_FILTER), 5),
				"clinical_amp_filter": round(clinical_amp_filter, 5),
				"gate_dropout": bool(self.gate_dropout_enabled()),
				"normalize_reference": bool(self.normalize_check.isChecked()),
				"phase_passenger": bool(phase_passenger_active),
			}
			phase_sig = self._hash_payload(phase_payload)
			if self.phase_result is None or phase_sig != self._cache_phase_sig:
				t_stage = perf_counter()
				self._set_progress(50, "Análisis de fase crudo y clínico robusto...")
				self.phase_result_raw = phase_analysis(
					cube_for_analysis,
					self.seg.mask,
					harmonics=int(self.harmonics_spin.value()),
					amplitude_threshold_frac=float(RAW_PHASE_QC_AMP_FILTER),
					normalize_reference=self.normalize_check.isChecked(),
				)
				self.phase_result = phase_analysis(
					cube_for_analysis,
					self.seg.mask,
					harmonics=int(self.harmonics_spin.value()),
					amplitude_threshold_frac=clinical_amp_filter,
					normalize_reference=self.normalize_check.isChecked(),
				)
				self._set_progress(65, "Métricas y segmentos AHA...")
				self.metrics_raw = self._annotate_phase_metrics(
					calculate_phase_metrics(self.phase_result_raw.phases_deg),
					self.phase_result_raw,
					RAW_PHASE_QC_AMP_FILTER,
					"crudo ROI",
				)
				self.metrics = self._annotate_phase_metrics(
					calculate_phase_metrics(self.phase_result.phases_deg),
					self.phase_result,
					clinical_amp_filter,
					"clínico robusto",
				)
				self.phase_qc = self._build_phase_qc(self.phase_result_raw, self.phase_result, self.metrics_raw, self.metrics)
				self.aha = map_to_17_segments(self.seg)
				self.phase_by_seg = phase_by_segment(self.phase_result.phase_map, self.aha)
				self.territory = territory_analysis(self.phase_by_seg)
				self._attach_robustness_metrics()
				self._cache_phase_sig = phase_sig
				self._invalidate_output_cache()
				self._log_timing_if_slow("Análisis de fase + métricas AHA", t_stage)
			else:
				self._set_progress(65, "Fase/métricas sin cambios (cache)...")
				self._log("Cache: fase, métricas y segmentación AHA reutilizadas.")
				if self.phase_result_raw is None:
					self.phase_result_raw = phase_analysis(
						cube_for_analysis,
						self.seg.mask,
						harmonics=int(self.harmonics_spin.value()),
						amplitude_threshold_frac=float(RAW_PHASE_QC_AMP_FILTER),
						normalize_reference=self.normalize_check.isChecked(),
					)
				if self.metrics_raw is None:
					self.metrics_raw = self._annotate_phase_metrics(
						calculate_phase_metrics(self.phase_result_raw.phases_deg),
						self.phase_result_raw,
						RAW_PHASE_QC_AMP_FILTER,
						"crudo ROI",
					)
				if self.metrics is None:
					self.metrics = self._annotate_phase_metrics(
						calculate_phase_metrics(self.phase_result.phases_deg),
						self.phase_result,
						clinical_amp_filter,
						"clínico robusto",
					)
				if self.phase_qc is None:
					self.phase_qc = self._build_phase_qc(self.phase_result_raw, self.phase_result, self.metrics_raw, self.metrics)
				if self.aha is None:
					self.aha = map_to_17_segments(self.seg)
				if self.phase_by_seg is None:
					self.phase_by_seg = phase_by_segment(self.phase_result.phase_map, self.aha)
				if self.territory is None:
					self.territory = territory_analysis(self.phase_by_seg)
				if not self.metrics.get("bootstrap") or not self.metrics.get("roi_sensitivity") or not self.metrics.get("segmental_aha"):
					self._attach_robustness_metrics()
			self._mark_step_done("phase", self._cache_phase_sig)
			preferred_gate_idx = int(self.study.cube.shape[0] // 2)
			preferred_slice_idx = int(self.study.cube.shape[1] // 2)

			self._set_progress(75, "Preparando cine...")
			self._expose_segmentation_rois(seg_method)
			self.active_cine_source = "primary"
			self._refresh_cine_source_selector()
			self._apply_cine_source(
				"primary",
				preserve_position=False,
				preferred_gate=preferred_gate_idx,
				preferred_slice=preferred_slice_idx,
			)
			self.cine.set_smooth_sigma(float(self.sigma_spin.value()))
			visual_payload = self._collect_visual_signature_payload()
			visual_payload["phase"] = self._cache_phase_sig
			output_sig = self._hash_payload(visual_payload)
			if output_sig != self._cache_output_sig:
				t_stage = perf_counter()
				if bool(self.realtime_deferred_render_check.isChecked()) and bool(self.advanced_mode_enabled):
					self._set_progress(80, "Generando vista rápida...")
					prev_adv = bool(self.advanced_mode_enabled)
					self.advanced_mode_enabled = False
					try:
						self._write_outputs()
					finally:
						self.advanced_mode_enabled = prev_adv
					self._cache_output_sig = ""
					self._schedule_deferred_hq_render("primary", delay_ms=280)
					self._log("Modo tiempo real: salida rápida lista; HQ diferido en progreso.")
				else:
					self._set_progress(80, "Generando imágenes...")
					self._write_outputs()
					self._cache_output_sig = output_sig
				self._log_timing_if_slow("Generación de salidas visuales", t_stage)
			else:
				self._set_progress(80, "Imágenes sin cambios (cache)...")
				self._log("Cache: se omitió regeneración de imágenes (parámetros visuales sin cambios).")
			self._last_primary_path = primary_abs

			if preserved_compare_path and os.path.exists(preserved_compare_path):
				t_stage = perf_counter()
				self._log("Reprocesando automáticamente el estudio de comparación cargado previamente...")
				self._load_compare_study_from_path(preserved_compare_path)
				self._log_timing_if_slow("Reproceso estudio de comparación", t_stage)
				self._log_timing_if_slow("Proceso total", t_total)
				return
			self._refresh_summary()
			self._set_progress(90, "Cargando previews...")
			t_stage = perf_counter()
			self._load_previews_selected(self._default_preview_tabs())
			self._log_timing_if_slow("Carga de previews", t_stage)
			self._select_tab_by_title("histograma")
			self._set_progress(100, "Procesamiento completo")
			self._log_timing_if_slow("Proceso total", t_total)
			self._mark_step_done("render", output_sig)
			self.statusBar().showMessage("Procesamiento completo")

			# Logging estructurado
			try:
				logger = get_logger()
				logger.log_processing_end(
					path,
					perf_counter() - t_total,
					self.metrics or {},
				)
			except Exception:
				pass

			# Exportación automática JSON/CSV
			try:
				self._export_structured_results()
			except Exception as exc:
				self._log(f"[WARN] Exportación estructurada falló: {exc}")

			# La ventana de cuantificación ECTb, si está abierta, muestra datos
			# del estudio anterior hasta que se le avisa.
			try:
				self._refresh_ectb_window()
			except Exception as exc:
				self._log(f"[WARN] Refresco de la ventana ECTb falló: {exc}")
			try:
				self._refresh_gqc_window()
			except Exception as exc:
				self._log(f"[WARN] Refresco del panel GQC falló: {exc}")
			try:
				self._refresh_asynchrony_review_window()
			except Exception as exc:
				self._log(f"[WARN] Refresco de la vista asincrónica falló: {exc}")
			try:
				self._refresh_readonly_results_panel()
			except Exception as exc:
				self._log(f"[WARN] Refresco del panel de resultados falló: {exc}")
		except Exception as exc:
			self._set_progress(0, "Error")
			self.statusBar().showMessage("Error")
			QMessageBox.critical(self, "Error de procesamiento", str(exc))
			self._log(f"[ERROR] {exc}")
			try:
				logger = get_logger()
				logger.log_error(exc, context={"study_path": path})
			except Exception:
				pass

	def _set_progress(self, value: int, label: str = ""):
		self._progress_bar.setValue(value)
		if label:
			self._progress_bar.setFormat(label)
			self.statusBar().showMessage(label)
		QApplication.processEvents()

	def _schedule_deferred_hq_render(
		self,
		job: str,
		*,
		delay_ms: int = 260,
		compare_bundle: dict | None = None,
		left_label: str = "",
		right_label: str = "",
	):
		self._deferred_hq_job = str(job or "").strip().lower()
		self._deferred_compare_bundle = compare_bundle
		self._deferred_compare_left_label = str(left_label or "")
		self._deferred_compare_right_label = str(right_label or "")
		self._deferred_hq_generation = int(self._deferred_hq_generation) + 1
		self._deferred_hq_timer.start(max(120, int(delay_ms)))

	def _run_deferred_hq_render(self):
		if self._deferred_hq_running:
			return
		self._deferred_hq_running = True
		run_generation = int(self._deferred_hq_generation)
		job = str(self._deferred_hq_job or "").strip().lower()
		self._deferred_hq_job = ""
		try:
			if job == "primary":
				if self.study is None or self.phase_result is None or not bool(self.advanced_mode_enabled):
					return
				if run_generation != int(self._deferred_hq_generation):
					self._log("Render HQ principal omitido por versión más nueva en cola.")
					return
				try:
					self._set_progress(88, "Render HQ diferido (principal)...")
					self._write_outputs()
					visual_payload = self._collect_visual_signature_payload()
					visual_payload["phase"] = self._cache_phase_sig
					self._cache_output_sig = self._hash_payload(visual_payload)
					self._load_previews_selected(self._default_preview_tabs())
					self._set_progress(100, "Procesamiento completo")
					self.statusBar().showMessage("Render HQ principal completado")
				except Exception as exc:
					self._log(f"[WARN] Render HQ diferido (principal) falló: {exc}")
				return
			if job == "compare":
				bundle = self._deferred_compare_bundle
				if bundle is None or self.study is None:
					return
				if run_generation != int(self._deferred_hq_generation):
					self._log("Render HQ comparación omitido por versión más nueva en cola.")
					return
				try:
					self._set_progress(88, "Render HQ diferido (comparación)...")
					self._run_compare_hq_pipeline(
						bundle,
						left_label=self._deferred_compare_left_label,
						right_label=self._deferred_compare_right_label,
						deferred=True,
					)
					self._set_progress(100, "Procesamiento completo")
				except Exception as exc:
					self._log(f"[WARN] Render HQ diferido (comparación) falló: {exc}")
		finally:
			self._deferred_hq_running = False
			if self._lazy_render_pending_tabs and self.study is not None and self.seg is not None and self.phase_result is not None:
				pending = set(self._lazy_render_pending_tabs)
				self._lazy_render_pending_tabs.clear()
				if pending:
					self._set_progress(86, f"Render bajo demanda: {', '.join(sorted(pending))} (post HQ)")
					try:
						if self.compare_bundle is not None:
							self._write_outputs_for_bundle(self.compare_bundle, self.compare_output_dir, target_tabs=pending)
						self._write_outputs(target_tabs=pending)
						if self.compare_bundle is not None:
							left_label, right_label = self._dual_compare_labels()
							self._compose_dual_tab_images(left_label, right_label, target_tabs=pending)
						self._load_previews_selected(pending)
					except Exception as exc:
						self._log(f"[WARN] Lazy render falló ({', '.join(sorted(pending))}): {exc}")

	def _effective_voxel_volume_ml(self) -> float | None:
		if self.study is None:
			return None
		px = getattr(self.study, "pixel_spacing", None)
		z_mm = getattr(self.study, "z_spacing_mm", None)
		if not px or z_mm is None:
			return None
		try:
			dx_mm = float(px[0])
			dy_mm = float(px[1])
			dz_mm = float(z_mm)
		except Exception:
			return None
		if dx_mm <= 0.0 or dy_mm <= 0.0 or dz_mm <= 0.0:
			return None
		# 1 mL = 1000 mm^3
		return (dx_mm * dy_mm * dz_mm) / 1000.0

	def _preferred_cine_slice_index(self, seg_obj) -> int:
		if seg_obj is None:
			return 0
		mask = np.asarray(getattr(seg_obj, "mask", np.empty((0,))), dtype=bool)
		if mask.ndim != 3 or mask.shape[0] <= 0:
			return 0
		areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
		if np.any(areas > 0):
			return int(np.argmax(areas))
		return int(mask.shape[0] // 2)

	def _compute_volumes_ml(self) -> dict[str, float | None]:
		if self.seg is None:
			return {
				"voxel_ml": None,
				"myocardial_ml": None,
				"cavity_ml": None,
				"lv_total_ml": None,
				"cavity_to_myo_ratio": None,
			}

		voxel_ml = self._effective_voxel_volume_ml()
		if voxel_ml is None:
			return {
				"voxel_ml": None,
				"myocardial_ml": None,
				"cavity_ml": None,
				"lv_total_ml": None,
				"cavity_to_myo_ratio": None,
			}

		# --- Volumen miocárdico (fix bug ~647 mL) --------------------------------
		# ANTES: se contaba TODA la máscara de segmentación (self.seg.mask), que con
		# umbral 0.35 capta un anillo grueso + arrastre de fondo (~2469 voxels →
		# ~647 mL, físicamente imposible: el miocardio del VI son ~85-250 mL).
		# AHORA: se prefiere la máscara CLÍNICA filtrada por amplitud (la misma que
		# usa el análisis de fase, phase_result.amplitude_map > 0, ~386 voxels), que
		# representa el miocardio con señal cardíaca real. Fallback a seg.mask solo
		# si no hay phase_result. Se marca fuera de rango fisiológico como guardrail.
		myo_voxels_clin = 0
		amp_map = getattr(getattr(self, "phase_result", None), "amplitude_map", None)
		if amp_map is not None:
			amp_arr = np.asarray(amp_map, dtype=np.float64)
			myo_voxels_clin = int(np.count_nonzero(amp_arr > 0.0))
		if myo_voxels_clin > 0:
			myocardial_voxels = myo_voxels_clin
			myo_source = "clinical_amp"
		else:
			myocardial_voxels = int(np.count_nonzero(self.seg.mask))
			myo_source = "seg_mask"
		myocardial_ml = float(myocardial_voxels) * voxel_ml
		# Guardrail físico: el miocardio del VI raramente excede ~250 mL.
		myo_physiologic = bool(40.0 <= myocardial_ml <= 260.0)

		cavity_ml = None
		centers = np.asarray(getattr(self.seg, "center_per_slice", np.empty((0, 2))), dtype=np.float64)
		inner = np.asarray(getattr(self.seg, "inner_radius", np.empty((0,))), dtype=np.float64)
		if centers.ndim == 2 and centers.shape[0] == inner.shape[0] and centers.shape[0] > 0:
			h = int(self.study.cube.shape[2])
			w = int(self.study.cube.shape[3])
			ys, xs = np.ogrid[:h, :w]
			cavity_voxels = 0
			for s in range(inner.shape[0]):
				r = float(inner[s])
				cy = float(centers[s, 0]) if np.isfinite(centers[s, 0]) else np.nan
				cx = float(centers[s, 1]) if np.isfinite(centers[s, 1]) else np.nan
				if not np.isfinite(cy) or not np.isfinite(cx) or not np.isfinite(r) or r <= 0.0:
					continue
				d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
				cavity_voxels += int(np.count_nonzero(d <= r))
			if cavity_voxels > 0:
				cavity_ml = float(cavity_voxels) * voxel_ml

		lv_total_ml = None
		cavity_to_myo_ratio = None
		if cavity_ml is not None:
			lv_total_ml = float(cavity_ml + myocardial_ml)
		if myocardial_ml > 0.0 and cavity_ml is not None:
			cavity_to_myo_ratio = float(cavity_ml / myocardial_ml)

		return {
			"voxel_ml": float(voxel_ml),
			"myocardial_ml": float(myocardial_ml),
			"myocardial_voxels": int(myocardial_voxels),
			"myo_source": str(myo_source),
			"myo_physiologic": bool(myo_physiologic),
			"cavity_ml": float(cavity_ml) if cavity_ml is not None else None,
			"lv_total_ml": float(lv_total_ml) if lv_total_ml is not None else None,
			"cavity_to_myo_ratio": float(cavity_to_myo_ratio) if cavity_to_myo_ratio is not None else None,
		}

	def _estimate_lv_ef_preliminary(self) -> dict[str, object | None]:
		if self.study is None or self.seg is None:
			return {"available": False}

		voxel_ml = self._effective_voxel_volume_ml()
		if voxel_ml is None:
			return {"available": False}

		# El mismo cubo corregido que usó fase/segmentación: si el último gate
		# arrastra el dropout de R-R, la curva de volumen no cierra y la FEVI sale
		# sesgada (ECTb 22.8).
		cube, _ = self._apply_gate_dropout_correction(self.study.cube, log=False)
		if cube.ndim != 4 or cube.shape[0] < 2:
			return {"available": False}

		centers = np.asarray(getattr(self.seg, "center_per_slice", np.empty((0, 2))), dtype=np.float64)
		inner = np.asarray(getattr(self.seg, "inner_radius", np.empty((0,))), dtype=np.float64)
		outer = np.asarray(getattr(self.seg, "outer_radius", np.empty((0,))), dtype=np.float64)
		n_slices = int(cube.shape[1])
		if centers.shape[0] != n_slices or inner.shape[0] != n_slices or outer.shape[0] != n_slices:
			return {"available": False}

		mask_all = np.asarray(getattr(self.seg, "mask", np.empty((0,))), dtype=bool)
		h = int(cube.shape[2])
		w = int(cube.shape[3])
		ys, xs = np.ogrid[:h, :w]
		n_gates = int(cube.shape[0])

		# --- Método angular de borde endocárdico (tipo QGS/Emory) ------------------
		# Para cada slice válido y cada gate se trazan N perfiles radiales desde un
		# centro RECENTRADO por gate (la cavidad se desplaza entre ED y ES, no solo
		# se contrae). El borde endocárdico en cada ángulo es el primer radio donde
		# la actividad supera un umbral relativo al pico miocárdico. El área de la
		# cavidad es el polígono encerrado por esos radios: 0.5 * Σ r² * dθ.
		#
		# Esto captura la CONTRACCIÓN real (ED grande, ES chico) en vez de contar
		# píxeles bajo umbral en un disco fijo (lo anterior daba EF ~15-21% porque
		# aplastaba la curva). Validado contra el estudio Xeleris: EF ~73% coincide
		# con Emory; el ciclo de volumen es fisiológico.
		cavity_frac = 0.45
		# Corrección basal: escala el radio endocárdico para que los volúmenes
		# absolutos (EDV/ESV) sean fisiológicos (~85-110 mL), alineados con GE/ECTb.
		# NO altera el EF (es un factor de escala sobre el radio, la relación
		# EDV/ESV se conserva). Validado contra estudio Xeleris.
		basal_pad = 0.30
		n_ang = 48

		# QC por gate: si el usuario editó ROIs manuales por gate en el cine, usar
		# ese centro/radio externo como geometría base del slice/gate en vez de la
		# segmentación automática común. Así la edición manual por gate afecta la
		# FEVI (p.ej. bajar FEVI agrandando el ROI de sístole).
		per_gate_rois = {}
		try:
			if hasattr(self, "cine") and self.cine is not None and self.cine.per_gate_roi_mode_enabled():
				per_gate_rois = dict(getattr(self.cine, "_rois_by_gate", {}) or {})
		except Exception:
			per_gate_rois = {}

		# Slices válidos: donde el anillo miocárdico es sustancial (evita base
		# abierta y apex sin cavidad, que inflan el volumen y matan el EF).
		if mask_all.shape[0] == n_slices:
			myo_area = mask_all.reshape(n_slices, -1).sum(axis=1)
		else:
			myo_area = np.zeros((n_slices,), dtype=np.float64)
		max_area = float(myo_area.max()) if myo_area.size else 0.0
		if max_area <= 0.0:
			return {"available": False}
		valid_s = [
			s for s in range(n_slices)
			if myo_area[s] >= 0.30 * max_area
			and np.isfinite(outer[s]) and outer[s] > 2.0
			and np.isfinite(centers[s, 0]) and np.isfinite(centers[s, 1])
		]
		if len(valid_s) < max(3, n_slices // 4):
			return {"available": False}

		angles = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
		sin_a = np.sin(angles)
		cos_a = np.cos(angles)
		dtheta = 2.0 * np.pi / n_ang

		def _cavity_area_per_gate(frac: float) -> np.ndarray:
			"""Área endocárdica por gate (px²) para un cavity_frac dado.

			Parametrizado para poder barrer distintos umbrales de borde en
			diagnóstico sin duplicar el bucle. El resto de la geometría
			(centros, outer, recentrado por gate, basal_pad) es idéntica.
			"""
			area = np.zeros((n_gates,), dtype=np.float64)
			for s in valid_s:
				cy0 = float(centers[s, 0])
				cx0 = float(centers[s, 1])
				ro0 = float(outer[s])
				for g in range(n_gates):
					# QC por gate: si hay ROI manual para este (gate, slice), usar
					# su centro y radio externo como geometría base.
					cy_g, cx_g, ro_g = cy0, cx0, ro0
					if per_gate_rois:
						roi_g = per_gate_rois.get((g, s))
						if roi_g is not None and len(roi_g) >= 4:
							try:
								cy_g = float(roi_g[0])
								cx_g = float(roi_g[1])
								if np.isfinite(roi_g[3]) and roi_g[3] > 0.0:
									ro_g = float(roi_g[3])
							except Exception:
								pass
					r_line = np.linspace(0.0, ro_g * 1.1, int(ro_g * 2) + 4)
					img = cube[g, s]
					d0 = np.sqrt((ys - cy_g) ** 2 + (xs - cx_g) ** 2)
					ring = (d0 >= ro_g * 0.5) & (d0 <= ro_g)
					peak = float(np.percentile(img[ring], 80)) if np.any(ring) else 0.0
					if peak <= 0.0:
						continue
					thr = frac * peak
					# Recentrado por gate: centroide de baja actividad cerca del centro.
					low = (d0 <= ro_g * 0.7) & (img < thr)
					if np.count_nonzero(low) >= 3:
						yy_l, xx_l = np.nonzero(low)
						cyg = float(yy_l.mean())
						cxg = float(xx_l.mean())
					else:
						cyg, cxg = cy_g, cx_g
					# Radio endocárdico por ángulo.
					r_endo = np.zeros((n_ang,), dtype=np.float64)
					for ai in range(n_ang):
						sy = cyg + r_line * sin_a[ai]
						sx = cxg + r_line * cos_a[ai]
						iy = np.clip(np.round(sy).astype(np.int32), 0, h - 1)
						ix = np.clip(np.round(sx).astype(np.int32), 0, w - 1)
						line_vals = img[iy, ix]
						above = np.where(line_vals >= thr)[0]
						r_endo[ai] = r_line[above[0]] if above.size else 0.0
					if basal_pad > 0.0:
						r_endo = r_endo * (1.0 + basal_pad)
					area[g] += float(0.5 * np.sum(r_endo ** 2) * dtheta)
			return area

		gate_cavity_area = _cavity_area_per_gate(cavity_frac)

		gate_volumes_ml = gate_cavity_area * float(voxel_ml)
		if gate_volumes_ml.size < 2 or not np.isfinite(gate_volumes_ml).all():
			return {"available": False}

		# Suavizado temporal circular (1-4-1): quita jitter sin perder el min/max
		# real del ciclo cardíaco (periódico).
		def _smooth_cyclic(v: np.ndarray) -> np.ndarray:
			if v.size < 3:
				return v
			prev = np.roll(v, 1)
			nxt = np.roll(v, -1)
			return (prev + 4.0 * v + nxt) / 6.0

		gate_volumes_ml = _smooth_cyclic(gate_volumes_ml)

		ed_idx = int(np.argmax(gate_volumes_ml))  # diástole = volumen máximo
		es_idx = int(np.argmin(gate_volumes_ml))  # sístole = volumen mínimo
		edv = float(gate_volumes_ml[ed_idx])
		esv = float(gate_volumes_ml[es_idx])
		if edv <= 0.0:
			return {"available": False}

		ef = float((edv - esv) / edv * 100.0)
		sv = float(edv - esv)

		# En uso clínico normal se omiten los diagnósticos [DIAG-*] para evitar
		# costo de cómputo extra. Se reactivan poniendo self.fevi_diag_enabled=True.
		if not bool(getattr(self, "fevi_diag_enabled", False)):
			return {
				"available": True,
				"method": "preliminar_endo_angular_gate",
				"valid_slices": int(len(valid_s)),
				"cavity_frac": float(cavity_frac),
				"basal_pad": float(basal_pad),
				"edv_ml": edv,
				"esv_ml": esv,
				"sv_ml": sv,
				"ef_pct": ef,
				"ed_gate": int(ed_idx + 1),
				"es_gate": int(es_idx + 1),
				"gate_volumes_ml": gate_volumes_ml,
			}

		# --- DIAGNÓSTICO TEMPORAL (Fase 1: calibración volumen/FEVI) ---------------
		# Vuelca los números crudos para separar "voxel mal calibrado" de "polígono
		# de cavidad demasiado chico". Quitar tras validar contra Xeleris.
		try:
			px = getattr(self.study, "pixel_spacing", None)
			z_mm = getattr(self.study, "z_spacing_mm", None)
			px_txt = f"{float(px[0]):.3f}x{float(px[1]):.3f}" if px else "None"
			area_ed = float(gate_cavity_area[ed_idx])
			area_es = float(gate_cavity_area[es_idx])
			vols_txt = ", ".join(f"{v:.1f}" for v in gate_volumes_ml)
			self._log(
				"[DIAG-VOL] "
				f"voxel_ml={voxel_ml:.5f} | pixel_spacing={px_txt} mm | z_spacing={z_mm} mm | "
				f"n_slices={n_slices} valid_slices={len(valid_s)} n_gates={n_gates} | "
				f"cavity_frac={cavity_frac} basal_pad={basal_pad} n_ang={n_ang} | "
				f"area_ED(px²)={area_ed:.1f} area_ES(px²)={area_es:.1f} | "
				f"EDV={edv:.1f} ESV={esv:.1f} SV={sv:.1f} mL EF={ef:.1f}% "
				f"(ed_gate={ed_idx + 1} es_gate={es_idx + 1}) | "
				f"gate_volumes_ml=[{vols_txt}]"
			)
		except Exception as _diag_err:  # pragma: no cover - solo diagnóstico
			self._log(f"[DIAG-VOL] error al loggear diagnóstico: {_diag_err}")

		# --- BARRIDO cavity_frac (Fase 3: calibrar borde endocárdico / ESV) -------
		# La FEVI alta viene del ESV bajo: en sístole el umbral fijo "cierra" la
		# cavidad de más. Se recalcula EDV/ESV/EF para varios umbrales SIN alterar
		# el resultado real (solo diagnóstico) para elegir con datos el que
		# reproduzca Xeleris (EDV~110, ESV~51, EF~53%). Quitar tras calibrar.
		try:
			sweep_rows = []
			for frac in (0.30, 0.35, 0.40, 0.45, 0.50):
				area_sw = _cavity_area_per_gate(float(frac))
				vols_sw = _smooth_cyclic(area_sw * float(voxel_ml))
				if vols_sw.size < 2 or not np.isfinite(vols_sw).all():
					continue
				edv_sw = float(vols_sw.max())
				esv_sw = float(vols_sw.min())
				if edv_sw <= 0.0:
					continue
				ef_sw = float((edv_sw - esv_sw) / edv_sw * 100.0)
				sweep_rows.append(
					f"frac={frac:.2f}: EDV={edv_sw:.1f} ESV={esv_sw:.1f} EF={ef_sw:.1f}%"
				)
			if sweep_rows:
				self._log("[DIAG-SWEEP] " + " | ".join(sweep_rows) + " (ref Xeleris: EDV~110 ESV~51 EF~53%)")
		except Exception as _sweep_err:  # pragma: no cover - solo diagnóstico
			self._log(f"[DIAG-SWEEP] error en barrido: {_sweep_err}")

		# --- RADIOS y GROSOR DE PARED ED vs ES (Fase 3: confirmar volumen parcial) -
		# Si en ES el radio endocárdico se subestima MIENTRAS la pared engrosa
		# mucho, es efecto de volumen parcial (miocardio brillante invade la
		# cavidad chica) → ESV artificialmente bajo → FEVI alta. Mide radio endo
		# medio (px) y grosor de pared (outer - endo) promediados sobre slices
		# válidos, en el gate ED y ES. Solo diagnóstico. Quitar tras decidir.
		try:
			def _radii_stats(g: int) -> tuple[float, float, float]:
				r_endo_acc = []
				r_outer_acc = []
				for s in valid_s:
					cy0 = float(centers[s, 0])
					cx0 = float(centers[s, 1])
					ro0 = float(outer[s])
					r_line = np.linspace(0.0, ro0 * 1.1, int(ro0 * 2) + 4)
					img = cube[g, s]
					d0 = np.sqrt((ys - cy0) ** 2 + (xs - cx0) ** 2)
					ring = (d0 >= ro0 * 0.5) & (d0 <= ro0)
					peak = float(np.percentile(img[ring], 80)) if np.any(ring) else 0.0
					if peak <= 0.0:
						continue
					thr = cavity_frac * peak
					low = (d0 <= ro0 * 0.7) & (img < thr)
					if np.count_nonzero(low) >= 3:
						yy_l, xx_l = np.nonzero(low)
						cyg = float(yy_l.mean())
						cxg = float(xx_l.mean())
					else:
						cyg, cxg = cy0, cx0
					r_endo = np.zeros((n_ang,), dtype=np.float64)
					for ai in range(n_ang):
						sy = cyg + r_line * sin_a[ai]
						sx = cxg + r_line * cos_a[ai]
						iy = np.clip(np.round(sy).astype(np.int32), 0, h - 1)
						ix = np.clip(np.round(sx).astype(np.int32), 0, w - 1)
						line_vals = img[iy, ix]
						above = np.where(line_vals >= thr)[0]
						r_endo[ai] = r_line[above[0]] if above.size else 0.0
					r_endo_acc.append(float(np.mean(r_endo)))
					r_outer_acc.append(ro0)
				if not r_endo_acc:
					return (np.nan, np.nan, np.nan)
				endo_m = float(np.mean(r_endo_acc))
				outer_m = float(np.mean(r_outer_acc))
				return (endo_m, outer_m, outer_m - endo_m)

			endo_ed, outer_ed, wall_ed = _radii_stats(ed_idx)
			endo_es, outer_es, wall_es = _radii_stats(es_idx)
			# Ratio de contracción radial: cuánto se achica el radio endo ED→ES.
			# Si es muy bajo (<0.6) hay sobre-contracción aparente (volumen parcial).
			endo_ratio = float(endo_es / endo_ed) if endo_ed > 0 else np.nan
			wall_thicken = float(wall_es / wall_ed) if wall_ed > 0 else np.nan
			self._log(
				"[DIAG-RADII] "
				f"ED(gate{ed_idx + 1}): r_endo={endo_ed:.2f}px r_outer={outer_ed:.2f}px pared={wall_ed:.2f}px | "
				f"ES(gate{es_idx + 1}): r_endo={endo_es:.2f}px r_outer={outer_es:.2f}px pared={wall_es:.2f}px | "
				f"contracción_radial(ES/ED)={endo_ratio:.2f} engrosamiento_pared(ES/ED)={wall_thicken:.2f} "
				"(volumen parcial probable si contracción<0.6 y engrosamiento>1.4)"
			)
		except Exception as _radii_err:  # pragma: no cover - solo diagnóstico
			self._log(f"[DIAG-RADII] error en radios: {_radii_err}")

		# --- UMBRAL ADAPTATIVO EN ES (Fase 3, Opción A) ---------------------------
		# Prueba: reducir cavity_frac SOLO en el gate de menor volumen (sístole)
		# para compensar el volumen parcial que "cierra" la cavidad de más.
		# Recalcula el área del gate ES con fracción reducida y ve si el ESV sube
		# hacia ~51 sin tocar el EDV. Solo diagnóstico (no altera el resultado).
		try:
			base_area = _cavity_area_per_gate(cavity_frac)  # área por gate con frac base
			es_g = int(np.argmin(_smooth_cyclic(base_area * float(voxel_ml))))
			adapt_rows = []
			for es_frac in (0.45, 0.40, 0.35, 0.30, 0.25, 0.20):
				# Área de todos los gates con frac base, pero el gate ES recalculado
				# con es_frac (más bajo → borde endo más afuera → cavidad ES mayor).
				area_es_only = _cavity_area_per_gate(float(es_frac))
				mixed = base_area.copy()
				mixed[es_g] = area_es_only[es_g]
				vols_mixed = _smooth_cyclic(mixed * float(voxel_ml))
				if vols_mixed.size < 2 or not np.isfinite(vols_mixed).all():
					continue
				edv_m = float(vols_mixed.max())
				esv_m = float(vols_mixed.min())
				if edv_m <= 0.0:
					continue
				ef_m = float((edv_m - esv_m) / edv_m * 100.0)
				adapt_rows.append(f"es_frac={es_frac:.2f}: EDV={edv_m:.1f} ESV={esv_m:.1f} EF={ef_m:.1f}%")
			if adapt_rows:
				self._log(
					f"[DIAG-ADAPT-ES] gate_ES={es_g + 1} (frac base={cavity_frac}) | "
					+ " | ".join(adapt_rows)
					+ " (objetivo Xeleris: ESV~51 EF~53%)"
				)
		except Exception as _adapt_err:  # pragma: no cover - solo diagnóstico
			self._log(f"[DIAG-ADAPT-ES] error: {_adapt_err}")

		# --- EPICARDIO CONGELADO vs POR-GATE (Fase 3: raíz del problema) ----------
		# La segmentación se calcula sobre cube.mean(axis=0) → outer[s] (epicardio)
		# es ÚNICO para los 8 gates. Aquí se mide el radio EPICÁRDICO recalculado
		# POR GATE (umbral relativo al pico de cada gate) para cuantificar cuánto
		# se está perdiendo al congelarlo. Si el outer real cambia entre ED y ES,
		# confirma que hay que segmentar por gate (como QGS/Xeleris). Solo diag.
		try:
			def _outer_mean_per_gate(g: int) -> float:
				acc = []
				for s in valid_s:
					cy0 = float(centers[s, 0])
					cx0 = float(centers[s, 1])
					ro0 = float(outer[s])
					img = cube[g, s]
					d0 = np.sqrt((ys - cy0) ** 2 + (xs - cx0) ** 2)
					ring = (d0 >= ro0 * 0.5) & (d0 <= ro0)
					peak = float(np.percentile(img[ring], 80)) if np.any(ring) else 0.0
					if peak <= 0.0:
						continue
					# Epicardio ≈ radio donde la actividad cae por debajo del 50% del
					# pico yendo hacia afuera (borde externo del anillo).
					thr_epi = 0.50 * peak
					r_line = np.linspace(0.0, ro0 * 1.6, int(ro0 * 3) + 6)
					r_out_ang = []
					for ai in range(0, n_ang, 4):  # submuestreo angular para velocidad
						sy = cy0 + r_line * sin_a[ai]
						sx = cx0 + r_line * cos_a[ai]
						iy = np.clip(np.round(sy).astype(np.int32), 0, h - 1)
						ix = np.clip(np.round(sx).astype(np.int32), 0, w - 1)
						vals = img[iy, ix]
						above = np.where(vals >= thr_epi)[0]
						r_out_ang.append(r_line[above[-1]] if above.size else ro0)
					if r_out_ang:
						acc.append(float(np.mean(r_out_ang)))
				return float(np.mean(acc)) if acc else np.nan

			outer_ed_g = _outer_mean_per_gate(ed_idx)
			outer_es_g = _outer_mean_per_gate(es_idx)
			outer_frozen = float(np.nanmean(outer[valid_s])) if len(valid_s) else np.nan
			epi_ratio = float(outer_es_g / outer_ed_g) if outer_ed_g and np.isfinite(outer_ed_g) else np.nan
			self._log(
				"[DIAG-EPI] "
				f"outer_CONGELADO(prom)={outer_frozen:.2f}px | "
				f"outer_POR-GATE ED(gate{ed_idx + 1})={outer_ed_g:.2f}px ES(gate{es_idx + 1})={outer_es_g:.2f}px | "
				f"epi_contracción(ES/ED)={epi_ratio:.2f} "
				"(si epi_contracción<0.9 el epicardio SÍ se mueve → segmentar por gate mejoraría)"
			)
		except Exception as _epi_err:  # pragma: no cover - solo diagnóstico
			self._log(f"[DIAG-EPI] error: {_epi_err}")

		# --- FWHM SUBPÍXEL (Fase 3, Opción 2: raíz de la FEVI alta) ---------------
		# El método actual toma "primer punto ≥ frac·pico" → radios endo de 1-2 px
		# (subestimados) y al elevar al cuadrado dispara la FEVI. Aquí se define el
		# borde endocárdico por FWHM: donde el perfil radial cruza el 50% del PICO
		# LOCAL del miocardio, interpolando linealmente entre píxeles vecinos
		# (precisión subpíxel). Da radios más grandes y un ratio ES/ED más
		# fisiológico. Recalcula EDV/ESV/EF con este criterio (solo diagnóstico).
		try:
			def _endo_area_fwhm_per_gate(fwhm_frac: float) -> np.ndarray:
				"""Área endocárdica por gate usando cruce FWHM subpíxel del perfil."""
				area = np.zeros((n_gates,), dtype=np.float64)
				for s in valid_s:
					cy0 = float(centers[s, 0])
					cx0 = float(centers[s, 1])
					ro0 = float(outer[s])
					# Perfil fino: 4 muestras por píxel para resolver subpíxel.
					r_line = np.linspace(0.0, ro0 * 1.2, int(ro0 * 4) + 8)
					for g in range(n_gates):
						img = cube[g, s]
						d0 = np.sqrt((ys - cy0) ** 2 + (xs - cx0) ** 2)
						ring = (d0 >= ro0 * 0.5) & (d0 <= ro0)
						peak_ring = float(np.percentile(img[ring], 80)) if np.any(ring) else 0.0
						if peak_ring <= 0.0:
							continue
						# Recentrado por gate (igual criterio que el método actual).
						thr_lo = 0.45 * peak_ring
						low = (d0 <= ro0 * 0.7) & (img < thr_lo)
						if np.count_nonzero(low) >= 3:
							yy_l, xx_l = np.nonzero(low)
							cyg = float(yy_l.mean())
							cxg = float(xx_l.mean())
						else:
							cyg, cxg = cy0, cx0
						r_endo = np.zeros((n_ang,), dtype=np.float64)
						for ai in range(n_ang):
							sy = cyg + r_line * sin_a[ai]
							sx = cxg + r_line * cos_a[ai]
							iy = np.clip(np.round(sy).astype(np.int32), 0, h - 1)
							ix = np.clip(np.round(sx).astype(np.int32), 0, w - 1)
							prof = img[iy, ix].astype(np.float64)
							# Pico LOCAL del miocardio a lo largo de este rayo.
							pk_i = int(np.argmax(prof))
							pk_val = float(prof[pk_i])
							if pk_val <= 0.0 or pk_i == 0:
								r_endo[ai] = 0.0
								continue
							target = fwhm_frac * pk_val
							# Buscar el cruce del 50% del pico SUBIENDO (endocardio):
							# el último índice antes del pico donde prof < target.
							r_cross = 0.0
							for k in range(pk_i, 0, -1):
								if prof[k - 1] < target <= prof[k]:
									# Interpolación lineal subpíxel entre k-1 y k.
									denom = prof[k] - prof[k - 1]
									frac_k = (target - prof[k - 1]) / denom if denom > 0 else 0.0
									r_cross = r_line[k - 1] + frac_k * (r_line[k] - r_line[k - 1])
									break
							r_endo[ai] = r_cross
						if basal_pad > 0.0:
							r_endo = r_endo * (1.0 + basal_pad)
						area[g] += float(0.5 * np.sum(r_endo ** 2) * dtheta)
				return area

			fwhm_rows = []
			r_endo_dbg = {}
			for ff in (0.50,):
				area_fw = _endo_area_fwhm_per_gate(float(ff))
				vols_fw = _smooth_cyclic(area_fw * float(voxel_ml))
				if vols_fw.size < 2 or not np.isfinite(vols_fw).all():
					continue
				edv_fw = float(vols_fw.max())
				esv_fw = float(vols_fw.min())
				if edv_fw <= 0.0:
					continue
				ef_fw = float((edv_fw - esv_fw) / edv_fw * 100.0)
				vols_txt_fw = ", ".join(f"{v:.1f}" for v in vols_fw)
				fwhm_rows.append(
					f"fwhm={ff:.2f}: EDV={edv_fw:.1f} ESV={esv_fw:.1f} EF={ef_fw:.1f}% "
					f"vols=[{vols_txt_fw}]"
				)
			if fwhm_rows:
				self._log(
					"[DIAG-FWHM] " + " | ".join(fwhm_rows)
					+ f" || ACTUAL: EDV={edv:.1f} ESV={esv:.1f} EF={ef:.1f}% "
					+ "(ref Xeleris: EDV~110 ESV~51 EF~53%)"
				)
		except Exception as _fwhm_err:  # pragma: no cover - solo diagnóstico
			self._log(f"[DIAG-FWHM] error: {_fwhm_err}")

		# --- UPSAMPLING antes de medir (Fase 3, Opción A: resolución en ES) --------
		# A 28×28 la cavidad en ES ocupa ~1 px de radio → sin señal para medir ESV.
		# Se interpola el cubo (H,W) a mayor resolución ANTES de trazar los rayos,
		# dando al algoritmo de borde píxeles subvoxel reales (como el modelo 3D de
		# Xeleris). El voxel_ml se divide por el factor² (mismo volumen físico). Se
		# prueban factores hacia 56 (×2) y 64 y se mide la DEFORMACIÓN de aspecto
		# (rows/cols) para descartar distorsión. Solo diagnóstico.
		try:
			from scipy.ndimage import zoom as _diag_zoom

			def _ef_on_upsampled(target_hw: int) -> tuple:
				fy = float(target_hw) / float(h)
				fx = float(target_hw) / float(w)
				# Deformación: cuánto difieren los factores de escala en cada eje.
				deform_pct = abs(fy - fx) / max(fy, fx) * 100.0
				# Reescala solo el plano (H,W) de cada gate/slice (order=3 cúbico).
				big = np.zeros((n_gates, n_slices, target_hw, target_hw), dtype=np.float64)
				for g in range(n_gates):
					for s in range(n_slices):
						big[g, s] = _diag_zoom(cube[g, s], (fy, fx), order=3, mode="nearest")
				big = np.clip(big, 0.0, None)
				# Centros y radios escalados al nuevo grid.
				centers_b = centers.copy()
				centers_b[:, 0] = centers[:, 0] * fy
				centers_b[:, 1] = centers[:, 1] * fx
				outer_b = outer * ((fy + fx) / 2.0)
				hb = wb = int(target_hw)
				ysb, xsb = np.ogrid[:hb, :wb]
				# voxel_ml corregido: mismo volumen físico repartido en más voxels.
				voxel_ml_b = float(voxel_ml) / (fy * fx)
				area_b = np.zeros((n_gates,), dtype=np.float64)
				for s in valid_s:
					cy0 = float(centers_b[s, 0]); cx0 = float(centers_b[s, 1])
					ro0 = float(outer_b[s])
					r_line = np.linspace(0.0, ro0 * 1.1, int(ro0 * 2) + 4)
					for g in range(n_gates):
						img = big[g, s]
						d0 = np.sqrt((ysb - cy0) ** 2 + (xsb - cx0) ** 2)
						ring = (d0 >= ro0 * 0.5) & (d0 <= ro0)
						peak = float(np.percentile(img[ring], 80)) if np.any(ring) else 0.0
						if peak <= 0.0:
							continue
						thr = cavity_frac * peak
						low = (d0 <= ro0 * 0.7) & (img < thr)
						if np.count_nonzero(low) >= 3:
							yy_l, xx_l = np.nonzero(low)
							cyg = float(yy_l.mean()); cxg = float(xx_l.mean())
						else:
							cyg, cxg = cy0, cx0
						r_endo = np.zeros((n_ang,), dtype=np.float64)
						for ai in range(n_ang):
							sy = cyg + r_line * sin_a[ai]
							sx = cxg + r_line * cos_a[ai]
							iy = np.clip(np.round(sy).astype(np.int32), 0, hb - 1)
							ix = np.clip(np.round(sx).astype(np.int32), 0, wb - 1)
							line_vals = img[iy, ix]
							above = np.where(line_vals >= thr)[0]
							r_endo[ai] = r_line[above[0]] if above.size else 0.0
						if basal_pad > 0.0:
							r_endo = r_endo * (1.0 + basal_pad)
						area_b[g] += float(0.5 * np.sum(r_endo ** 2) * dtheta)
				vols_b = _smooth_cyclic(area_b * voxel_ml_b)
				edv_b = float(vols_b.max()); esv_b = float(vols_b.min())
				ef_b = float((edv_b - esv_b) / edv_b * 100.0) if edv_b > 0 else np.nan
				# Radio endo medio ED/ES en el grid grande (para ver si ya hay señal).
				return edv_b, esv_b, ef_b, deform_pct

			up_rows = []
			for target in (56, 64):
				edv_b, esv_b, ef_b, deform = _ef_on_upsampled(int(target))
				up_rows.append(
					f"{target}x{target}: EDV={edv_b:.1f} ESV={esv_b:.1f} EF={ef_b:.1f}% "
					f"deform={deform:.1f}%"
				)
			if up_rows:
				self._log(
					"[DIAG-UPSAMPLE] " + " | ".join(up_rows)
					+ f" || ACTUAL(28x28): EDV={edv:.1f} ESV={esv:.1f} EF={ef:.1f}% "
					+ "(ref Xeleris: EDV~110 ESV~51 EF~53%; deform<3% = sin distorsión)"
				)
		except Exception as _up_err:  # pragma: no cover - solo diagnóstico
			self._log(f"[DIAG-UPSAMPLE] error: {_up_err}")

		return {
			"available": True,
			"method": "preliminar_endo_angular_gate",
			"valid_slices": int(len(valid_s)),
			"cavity_frac": float(cavity_frac),
			"basal_pad": float(basal_pad),
			"edv_ml": edv,
			"esv_ml": esv,
			"sv_ml": sv,
			"ef_pct": ef,
			"ed_gate": int(ed_idx + 1),
			"es_gate": int(es_idx + 1),
			"gate_volumes_ml": gate_volumes_ml,
		}

	# ------------------------------------------------------------------
	# Selección del método de cuantificación de FEVI
	# ------------------------------------------------------------------

	#: Método por máximo de cuentas del Emory Cardiac Toolbox. Es el que usa el
	#: informe: sobreestima bastante menos que el de umbral.
	FEVI_METHOD_ECTB = "ectb"
	#: Método original por umbral de actividad + factor de corrección basal.
	#: Se conserva seleccionable para poder comparar.
	FEVI_METHOD_THRESHOLD = "umbral"

	FEVI_METHOD_LABELS = {
		FEVI_METHOD_ECTB: "ECTb (máximo de cuentas)",
		FEVI_METHOD_THRESHOLD: "Anterior (umbral endocárdico)",
	}

	#: Fuente de la imagen de perfusión para la tabla/bull's eye segmentario.
	#: El gate ED (fin de diástole) es el estándar de lectura de perfusión.
	PERFUSION_SOURCE_ED = "ed_gate"
	#: Alternativa: media de todos los gates (más cuentas, menos ruido).
	PERFUSION_SOURCE_MEAN = "mean_gates"

	PERFUSION_SOURCE_LABELS = {
		PERFUSION_SOURCE_ED: "Gate ED (fin de diástole) — estándar",
		PERFUSION_SOURCE_MEAN: "Media de todos los gates",
	}

	def perfusion_source(self) -> str:
		"""Fuente de perfusión para el panel segmentario (gate ED por defecto)."""
		src = str(getattr(self, "_perfusion_source", self.PERFUSION_SOURCE_ED))
		return src if src in self.PERFUSION_SOURCE_LABELS else self.PERFUSION_SOURCE_ED

	def perfusion_source_label(self) -> str:
		return self.PERFUSION_SOURCE_LABELS[self.perfusion_source()]

	def fevi_method(self) -> str:
		"""Método de FEVI que alimenta resumen, gráficos e informe."""
		method = str(getattr(self, "_fevi_method", self.FEVI_METHOD_ECTB))
		return method if method in self.FEVI_METHOD_LABELS else self.FEVI_METHOD_ECTB

	def fevi_method_label(self) -> str:
		return self.FEVI_METHOD_LABELS[self.fevi_method()]

	def set_fevi_method(self, method: str, *, reprocess: bool = True):
		"""Cambia el método de FEVI del informe y regenera las salidas.

		Los gráficos y el PDF ya emitidos quedan con el método anterior, así que
		se invalida la caché de salidas y se reprocesa.
		"""
		method = method if method in self.FEVI_METHOD_LABELS else self.FEVI_METHOD_ECTB
		if method == getattr(self, "_fevi_method", None):
			return
		self._fevi_method = method
		self._log(f"[FEVI] Método del informe: {self.fevi_method_label()}")
		self.statusBar().showMessage(f"FEVI del informe: {self.fevi_method_label()}")
		self._invalidate_output_cache()
		if reprocess and self.study is not None and self.seg is not None:
			QTimer.singleShot(0, self.process_current)

	def ectb_config(self) -> ECTbLVConfig:
		"""Parámetros ECTb vigentes (los que se ajustan en la ventana ECTb)."""
		cfg = getattr(self, "_ectb_config", None)
		return cfg if isinstance(cfg, ECTbLVConfig) else ECTbLVConfig()

	def set_ectb_config(self, config: ECTbLVConfig):
		self._ectb_config = config

	def fevi_regression(self) -> str | None:
		"""Regresión de conversión activa, o None si se informa en escala ECTb."""
		key = getattr(self, "_fevi_regression", None)
		return key if key in EF_REGRESSIONS else None

	def set_fevi_regression(self, key: str | None):
		"""Elige contra qué software se expresa la FEVI equivalente.

		No cambia la FEVI del informe: solo agrega el valor convertido, para poder
		compararlo con un estudio previo hecho en otro equipo.
		"""
		key = key if key in EF_REGRESSIONS else None
		if key == getattr(self, "_fevi_regression", None):
			return
		self._fevi_regression = key
		if key:
			self._log(f"[FEVI] Equivalencia informada en escala: {EF_REGRESSIONS[key].label}")
		else:
			self._log("[FEVI] Equivalencia con otro software desactivada.")
		self._invalidate_output_cache()
		if self.study is not None and self.metrics is not None:
			self._refresh_summary()

	def _estimate_lv_ef_ectb(self) -> dict[str, object | None]:
		"""FEVI por el método ECTb, en el mismo formato que el estimador anterior.

		Devolver el mismo diccionario permite que resumen, gráficos, exportación
		y PDF funcionen igual con cualquiera de los dos métodos.
		"""
		def unavailable(reason: str) -> dict[str, object | None]:
			return {"available": False, "method": "ectb_max_counts", "reason": reason}

		if self.study is None or self.seg is None:
			return unavailable("Falta estudio o segmentación.")
		pixel_spacing = getattr(self.study, "pixel_spacing", None)
		slice_mm = getattr(self.study, "z_spacing_mm", None)
		if not pixel_spacing or slice_mm is None:
			return unavailable("El estudio no trae spacing válido.")
		cube = getattr(self.study, "cube", None)
		if cube is None:
			return unavailable("El estudio no tiene cubo gated.")

		try:
			cube, _ = self._apply_gate_dropout_correction(cube, log=False)
			res = analyze_lv_ectb(
				cube,
				self.seg,
				(float(pixel_spacing[0]), float(pixel_spacing[1])),
				float(slice_mm),
				self.ectb_config(),
			)
		except Exception as exc:
			self._log(f"[WARN] FEVI ECTb falló: {exc}")
			return unavailable(str(exc))

		if not res.available:
			return unavailable(res.reason)

		# Cachear el resultado completo (con radios endo/epi por gate) para el
		# panel 3D del VI: la malla alambre ED/ES se construye de acá.
		self._ectb_last_result = res

		payload: dict[str, object | None] = {
			"available": True,
			"method": "ectb_max_counts",
			"valid_slices": int(len(res.valid_slices)),
			"edv_ml": float(res.edv_ml),
			"esv_ml": float(res.esv_ml),
			"sv_ml": float(res.sv_ml),
			"ef_pct": float(res.ef_pct),
			"ed_gate": int(res.ed_gate),
			"es_gate": int(res.es_gate),
			"gate_volumes_ml": [float(v) for v in res.gate_volumes_ml],
			"myocardial_volume_ml": float(res.myocardial_volume_ml),
			"myocardial_mass_g": float(res.myocardial_mass_g),
			"thickening_pct": float(res.thickening_pct),
			"wall_thickness_ed_mm": float(res.wall_thickness_ed_mm),
			"wall_thickness_es_mm": float(res.wall_thickness_es_mm),
			"ed_wall_thickness_mm": float(res.config.ed_wall_thickness_mm),
			"valve_plane": bool(res.config.use_valve_plane),
			"valve_offset_mm": float(res.config.valve_septal_offset_mm),
			"valve_removed_ml": float(res.valve_removed_ml),
			"shape_index_ed": float(res.shape_index_ed),
			"shape_index_es": float(res.shape_index_es),
			"long_axis_mm": float(res.long_axis_mm),
			"short_axis_ed_mm": float(res.short_axis_ed_mm),
			"short_axis_es_mm": float(res.short_axis_es_mm),
			"notes": list(res.notes),
		}
		regression = self.fevi_regression()
		if regression:
			reg = EF_REGRESSIONS[regression]
			payload["regression_key"] = regression
			payload["regression_label"] = reg.label
			payload["ef_pct_converted"] = float(convert_ef_pct(res.ef_pct, reg))
		return payload
		return payload

	def _effective_rr_ms(self) -> float | None:
		"""Intervalo RR medio (ms) más confiable disponible, o None.

		Prioridad: RR medido en adquisición (si no es placeholder) → FC del DICOM
		→ FC que el usuario fijó en el panel ECG. Solo devuelve valores fisiológicos.
		"""
		gating = getattr(self.study, "gating_info", None) or {} if self.study is not None else {}
		rr = gating.get("rr_mean_ms")
		if rr and not gating.get("rr_placeholder"):
			try:
				rr_val = float(rr)
				if 250.0 <= rr_val <= 2500.0:
					return rr_val
			except (TypeError, ValueError):
				pass
		fc = gating.get("heart_rate") or gating.get("heart_rate_est")
		try:
			fc_val = int(fc) if fc else 0
		except (TypeError, ValueError):
			fc_val = 0
		if not (25 <= fc_val <= 250) and hasattr(self, "ecg_fc_spin"):
			try:
				fc_val = int(self.ecg_fc_spin.value())
			except Exception:
				fc_val = 0
		if 25 <= fc_val <= 250:
			return 60000.0 / float(fc_val)
		return None

	def _augment_with_filling_metrics(self, ef: dict[str, object | None]) -> dict[str, object | None]:
		"""Inyecta PFR y TVmáx (función diastólica) en el dict de FEVI.

		Reutiliza la curva de volumen por gate que el estimador ya calculó, así
		el readout, el resumen y el PDF los ven con las mismas claves 'pfr_*'/'tpfr_*'.
		"""
		if not ef or not bool(ef.get("available")):
			return ef
		try:
			fm = compute_filling_metrics(
				ef.get("gate_volumes_ml", []),
				float(ef.get("edv_ml", 0.0) or 0.0),
				int(ef.get("es_gate", 1) or 1),
				self._effective_rr_ms(),
			)
		except Exception as exc:  # pragma: no cover - defensivo
			self._log(f"[WARN] PFR/TVmáx falló: {exc}")
			return ef
		if fm.get("available"):
			ef.update({
				"pfr_edv_per_rr": fm.get("pfr_edv_per_rr"),
				"pfr_edv_per_s": fm.get("pfr_edv_per_s"),
				"tpfr_pct_rr": fm.get("tpfr_pct_rr"),
				"tpfr_ms": fm.get("tpfr_ms"),
				"pfr_gate": fm.get("pfr_gate"),
				"pfr_text": format_pfr(fm),
				"tvmax_text": format_tvmax(fm),
			})
		return ef

	def _estimate_lv_ef(self) -> dict[str, object | None]:
		"""Punto único de entrada para la FEVI del informe.

		Si el método ECTb no puede resolver (segmentación pobre, spacing raro),
		cae al método anterior en vez de dejar el informe sin FEVI.
		"""
		if self.fevi_method() == self.FEVI_METHOD_THRESHOLD:
			return self._augment_with_filling_metrics(self._estimate_lv_ef_preliminary())
		result = self._estimate_lv_ef_ectb()
		if result.get("available"):
			return self._augment_with_filling_metrics(result)
		fallback = self._estimate_lv_ef_preliminary()
		if fallback.get("available"):
			fallback["fallback_from"] = "ectb_max_counts"
			fallback["fallback_reason"] = str(result.get("reason") or "")
			self._log(
				"[FEVI] ECTb no pudo cuantificar "
				f"({fallback['fallback_reason']}); se usó el método anterior."
			)
		return self._augment_with_filling_metrics(fallback)

	def _harmonize_volumes_with_ef(self, vol: dict[str, float | None], ef: dict[str, object | None]) -> dict[str, float | None]:
		"""Unifica la cavidad reportada con EDV cuando FEVI está disponible.

		Evita inconsistencias entre:
		- "Cavidad" calculada por inner_radius estático (puede sobre/infra-estimar), y
		- EDV/ESV dinámicos por gate del motor FEVI.

		Regla: si hay FEVI disponible, usar EDV (gate de volumen máximo) como
		volumen de cavidad de referencia para el resumen/informe.
		"""
		out = dict(vol or {})
		if not ef or not bool(ef.get("available")):
			return out
		try:
			edv = float(ef.get("edv_ml", np.nan))
		except Exception:
			edv = np.nan
		if not np.isfinite(edv) or edv <= 0.0:
			return out

		out["cavity_ml"] = float(edv)
		myo = out.get("myocardial_ml", None)
		if myo is not None:
			try:
				myo_f = float(myo)
			except Exception:
				myo_f = np.nan
			if np.isfinite(myo_f) and myo_f > 0.0:
				out["lv_total_ml"] = float(myo_f + edv)
				out["cavity_to_myo_ratio"] = float(edv / myo_f)
		out["cavity_source"] = "edv_gate_max"
		return out

	def _polar_compare_operation_text(self) -> str:
		op_name = str(self.polar_compare_math_combo.currentText())
		if op_name == "Ninguna":
			return ""
		a_name = str(self.polar_compare_term_a_combo.currentText())
		b_name = str(self.polar_compare_term_b_combo.currentText())
		symbol = {
			"Suma": "+",
			"Resta": "-",
			"Multiplicación": "*",
			"División": "/",
		}.get(op_name, op_name)
		return f"{a_name} {symbol} {b_name} ({op_name})"

	def _phase_label_from_path(self, path_text: str, fallback: str = "Estudio") -> str:
		u = os.path.basename(str(path_text or "")).upper()
		if "REST" in u:
			return "Reposo"
		if "STRESS" in u:
			return "Esfuerzo"
		return fallback

	def _format_dicom_date(self, raw: str) -> str:
		val = str(raw or "").strip()
		if len(val) == 8 and val.isdigit():
			return f"{val[6:8]}/{val[4:6]}/{val[0:4]}"
		return val or "N/D"

	def _study_context(self, *, path_override: str | None = None, study_obj=None) -> dict[str, str]:
		study_ref = study_obj if study_obj is not None else self.study
		path_txt = str(path_override if path_override is not None else (self.file_edit.text().strip() if self.file_edit is not None else ""))
		# Etapa desde metadata DICOM primero; el nombre de archivo es solo fallback.
		phase = self._cine_crudo_stage_display(study_ref) or self._phase_label_from_path(path_txt, "Estudio")
		patient_name = str(getattr(study_ref, "patient_name", "") or "").strip()
		patient_id = str(getattr(study_ref, "patient_id", "") or "").strip()
		study_date = self._format_dicom_date(str(getattr(study_ref, "study_date", "") or ""))
		desc = str(getattr(study_ref, "study_description", "") or "").strip()
		if not patient_name:
			patient_name = desc or os.path.splitext(os.path.basename(path_txt))[0] or "Paciente N/D"
		return {
			"phase": phase,
			"patient_name": patient_name,
			"patient_id": patient_id or "N/D",
			"study_date": study_date,
		}

	def _study_context_label(self, *, path_override: str | None = None, study_obj=None) -> str:
		ctx = self._study_context(path_override=path_override, study_obj=study_obj)
		return f"{ctx['phase']} | {ctx['patient_name']} | ID {ctx['patient_id']} | Fecha {ctx['study_date']}"

	def _patient_banner_text(self, stage: str | None = None, include_stage: bool = True) -> str:
		"""Paciente · ID · Fecha · ETAPA — la etapa sale del DICOM (fallback: slot stress/rest)."""
		stg = str(stage or getattr(self, "_cine_crudo_active_stage", "stress") or "stress")
		study = self.study
		if stg == "rest":
			study = self._secondary_cine_crudo_study() or self.study
		try:
			if stg in ("stress", "rest"):
				st = self._dual_session().stage(stg)
				study = getattr(st, "raw_study_for_recon", None) or getattr(st, "raw_study", None) or study
		except Exception:
			pass
		name = str(getattr(study, "patient_name", "") or "").strip().replace("^", " ") or "Paciente N/D"
		pid = str(getattr(study, "patient_id", "") or "").strip()
		date = self._format_dicom_date(str(getattr(study, "study_date", "") or ""))
		if stg == "both":
			stage_txt = "ESFUERZO + REPOSO"
		else:
			stage_txt = (self._cine_crudo_stage_display(study) or ("Reposo" if stg == "rest" else "Esfuerzo")).upper()
		parts = [name]
		if pid:
			parts.append(f"ID {pid}")
		parts.append(date)
		if include_stage:
			parts.append(stage_txt)
		return " · ".join(parts)

	def _update_patient_banner(self):
		"""Banner permanente en la barra de estado: paciente + fecha + etapa activa."""
		lbl = getattr(self, "_patient_banner_lbl", None)
		if lbl is None:
			lbl = QLabel("")
			lbl.setStyleSheet("color:#fbbf24; font-weight:bold; padding:0 10px;")
			self.statusBar().addPermanentWidget(lbl)
			self._patient_banner_lbl = lbl
		try:
			lbl.setText(self._patient_banner_text() if self.study is not None else "")
		except Exception:
			lbl.setText("")

	def _dual_compare_labels(self) -> tuple[str, str]:
		"""Rótulos de renders comparativos: SIEMPRE Esfuerzo/Reposo (DICOM), nunca el nombre del archivo."""
		left = self._cine_crudo_stage_display(self.study) or "Esfuerzo"
		try:
			cmp_study = self._second_stage_study()
		except Exception:
			cmp_study = None
		right = self._cine_crudo_stage_display(cmp_study) or "Reposo"
		if left == right:
			left, right = "Esfuerzo", "Reposo"
		return left, right

	def _second_stage_study(self):
		"""Estudio de la 2da etapa cargada (reconstruido o crudo), o None."""
		if self.compare_bundle is not None:
			return self.compare_bundle.get("study")
		if getattr(self, "compare_raw_study", None) is not None:
			return self.compare_raw_study
		return None

	@staticmethod
	def _normalize_patient_token(value) -> str:
		return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())

	def _check_second_stage_patient(self, comp_study) -> bool:
		"""Valida que la 2da etapa sea del MISMO paciente que la 1ra.

		Devuelve True si se permite continuar. Bloquea si la metadata indica
		positivamente pacientes distintos; si es ambigua (faltan datos), pide
		confirmación al usuario. (La correlación de estudios del mismo paciente
		en distintas fechas se habilitará más adelante.)
		"""
		if self.study is None or comp_study is None:
			return True
		id_a = self._normalize_patient_token(getattr(self.study, "patient_id", ""))
		id_b = self._normalize_patient_token(getattr(comp_study, "patient_id", ""))
		name_a = self._normalize_patient_token(getattr(self.study, "patient_name", ""))
		name_b = self._normalize_patient_token(getattr(comp_study, "patient_name", ""))
		a = self._study_context(study_obj=self.study)
		b = self._study_context(study_obj=comp_study)
		# Mismo paciente confirmado (por ID; fallback nombre si falta algún ID).
		same = (id_a and id_b and id_a == id_b) or (not (id_a and id_b) and name_a and name_b and name_a == name_b)
		if same:
			return True
		# Distinto confirmado: ambos IDs presentes y difieren, o ambos nombres difieren.
		positively_different = (bool(id_a) and bool(id_b) and id_a != id_b) or (bool(name_a) and bool(name_b) and name_a != name_b)
		if positively_different:
			QMessageBox.critical(
				self, "Pacientes distintos",
				"No se puede cargar la segunda etapa: parece pertenecer a OTRO paciente.\n\n"
				f"1ra etapa: {a['patient_name']} (ID {a['patient_id']})\n"
				f"2da etapa: {b['patient_name']} (ID {b['patient_id']})\n\n"
				"Esfuerzo y reposo deben ser del mismo paciente. La correlación de estudios "
				"del mismo paciente en distintas fechas se habilitará más adelante.",
			)
			self._log(
				f"[BLOQUEADO] 2da etapa de otro paciente: '{a['patient_name']}' (ID {a['patient_id']}) "
				f"vs '{b['patient_name']}' (ID {b['patient_id']})."
			)
			return False
		# Ambiguo (falta ID/nombre en alguno): pedir confirmación explícita.
		btn = QMessageBox.question(
			self, "Confirmar paciente",
			"No se pudo verificar por metadata que ambas etapas sean del mismo paciente.\n\n"
			f"1ra etapa: {a['patient_name']} (ID {a['patient_id']})\n"
			f"2da etapa: {b['patient_name']} (ID {b['patient_id']})\n\n"
			"¿Confirmás que corresponden al mismo paciente y querés continuar?",
			QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
			QMessageBox.StandardButton.No,
		)
		if btn != QMessageBox.StandardButton.Yes:
			self._log("2da etapa cancelada por el usuario (identidad de paciente no verificable).")
			return False
		self._log("2da etapa aceptada manualmente pese a metadata de paciente incompleta.")
		return True

	@staticmethod
	def _acquisition_pixel_scale(study) -> float | None:
		"""Zoom de adquisición = tamaño de píxel (mm/px) del estudio.

		Devuelve la media geométrica del PixelSpacing (mm/px) o None si no hay dato.
		Un píxel más chico (mm/px menor) equivale a mayor zoom de adquisición: el
		corazón ocupa más píxeles y puede simular una dilatación falsa si se compara
		contra una etapa adquirida con otro zoom.
		"""
		px = getattr(study, "pixel_spacing", None)
		if not px:
			return None
		try:
			sy = float(px[0])
			sx = float(px[1]) if len(px) > 1 else sy
		except (TypeError, ValueError, IndexError):
			return None
		if not (np.isfinite(sy) and np.isfinite(sx)) or sy <= 0.0 or sx <= 0.0:
			return None
		return float(np.sqrt(sy * sx))

	def _apply_zoom_rescale(self, study, factor: float, *, match_scale: float, stage_label: str) -> bool:
		"""Reescala espacialmente el cubo de `study` por `factor` para empatar el zoom.

		El cubo es 4D ``(gates, slices/ángulos, H, W)``; se interpolan sólo H y W.
		Tras el reescalado, el PixelSpacing se iguala al de la etapa de referencia y
		se registra `study.zoom_correction` para que quede grabado y se exporte en DICOM.
		"""
		try:
			from scipy.ndimage import zoom as ndi_zoom
		except Exception as exc:  # pragma: no cover
			self._log(f"[WARN] scipy no disponible; no se pudo reescalar zoom: {exc}")
			return False
		cube = np.asarray(getattr(study, "cube", None), dtype=np.float64)
		if cube.ndim != 4:
			self._log(f"[WARN] Cubo no 4D ({cube.shape}); no se aplica reescalado de zoom.")
			return False
		orig_px = getattr(study, "pixel_spacing", None)
		try:
			new_cube = ndi_zoom(cube, (1.0, 1.0, factor, factor), order=1, prefilter=False)
		except Exception as exc:
			self._log(f"[WARN] Falló el reescalado del cubo: {exc}")
			return False
		study.cube = new_cube
		study.rows = int(new_cube.shape[2])
		if not getattr(study, "was_montage", False):
			study.cols = int(new_cube.shape[3])
		study.pixel_spacing = (float(match_scale), float(match_scale))
		study.zoom_correction = {
			"applied": True,
			"factor": float(factor),
			"matched_scale_mm": float(match_scale),
			"original_pixel_spacing": tuple(orig_px) if orig_px else None,
			"stage": stage_label,
		}
		note = (
			f"Zoom empatado con {stage_label}: reescalado ×{factor:.3f}; "
			f"PixelSpacing → {match_scale:.3f} mm/px (cubo {cube.shape[2]}×{cube.shape[3]} → "
			f"{new_cube.shape[2]}×{new_cube.shape[3]})."
		)
		try:
			if isinstance(getattr(study, "notes", None), list):
				study.notes.append(note)
		except Exception:
			pass
		self._log(f"[ZOOM] {note}")
		return True

	def _check_stage_zoom_consistency(self, comp_study) -> bool:
		"""Chequea que el zoom de adquisición (PixelSpacing) coincida entre etapas.

		Si difieren, avisa del riesgo de falso positivo de dilatación del VI y ofrece
		aplicar un factor de reescalado a la 2da etapa para empatar el zoom. No bloquea:
		el usuario puede continuar sin corregir. Devuelve False sólo si el usuario
		cancela la carga.
		"""
		if self.study is None or comp_study is None:
			return True
		scale_primary = self._acquisition_pixel_scale(self.study)
		scale_secondary = self._acquisition_pixel_scale(comp_study)
		if scale_primary is None or scale_secondary is None:
			self._log(
				"[ZOOM] No se pudo verificar el zoom de adquisición (falta PixelSpacing en alguna etapa)."
			)
			return True
		# Tolerancia 0.5 %: mismo zoom efectivo.
		if abs(scale_secondary - scale_primary) <= 0.005 * max(scale_primary, scale_secondary):
			self._log(
				f"[ZOOM] Zoom de adquisición coincidente: "
				f"1ra={scale_primary:.3f} mm/px · 2da={scale_secondary:.3f} mm/px."
			)
			return True
		# Factor a aplicar a la 2da etapa para que su píxel iguale al de la 1ra.
		factor = float(scale_secondary / scale_primary)
		stage_ref = "Esfuerzo (1ra etapa)"
		bigger = "2da etapa" if scale_secondary < scale_primary else "1ra etapa"
		box = QMessageBox(self)
		box.setIcon(QMessageBox.Icon.Warning)
		box.setWindowTitle("Zoom de adquisición distinto")
		box.setText(
			"El zoom de adquisición NO coincide entre las dos etapas.\n\n"
			f"1ra etapa (referencia): {scale_primary:.3f} mm/px\n"
			f"2da etapa:              {scale_secondary:.3f} mm/px\n\n"
			f"El corazón aparece más grande en la {bigger}, lo que puede simular una "
			"dilatación falsa del VI al comparar. Se puede reescalar la 2da etapa para "
			f"empatar el zoom de la referencia (factor sugerido ×{factor:.3f})."
		)
		apply_btn = box.addButton("Empatar zoom (reescalar 2da)", QMessageBox.ButtonRole.AcceptRole)
		keep_btn = box.addButton("Continuar sin corregir", QMessageBox.ButtonRole.RejectRole)
		cancel_btn = box.addButton("Cancelar carga", QMessageBox.ButtonRole.DestructiveRole)
		box.setDefaultButton(apply_btn)
		box.exec()
		clicked = box.clickedButton()
		if clicked is cancel_btn:
			self._log("[ZOOM] Carga de 2da etapa cancelada por discrepancia de zoom.")
			return False
		if clicked is apply_btn:
			ok = self._apply_zoom_rescale(
				comp_study, factor, match_scale=scale_primary, stage_label=stage_ref,
			)
			if not ok:
				QMessageBox.warning(
					self, "No se pudo reescalar",
					"No se pudo aplicar el reescalado automático. Se continúa sin corregir; "
					"tener en cuenta la diferencia de zoom al interpretar el tamaño del VI.",
				)
			return True
		# Continuar sin corregir: dejar constancia.
		try:
			if isinstance(getattr(comp_study, "notes", None), list):
				comp_study.notes.append(
					f"ADVERTENCIA: zoom de adquisición distinto ({scale_secondary:.3f} vs "
					f"{scale_primary:.3f} mm/px) — riesgo de falsa dilatación del VI."
				)
		except Exception:
			pass
		comp_study.zoom_correction = {
			"applied": False,
			"factor": factor,
			"matched_scale_mm": scale_primary,
			"secondary_scale_mm": scale_secondary,
			"stage": stage_ref,
		}
		self._log(
			f"[ZOOM] Usuario continúa sin corregir: 1ra={scale_primary:.3f} vs "
			f"2da={scale_secondary:.3f} mm/px (factor sugerido ×{factor:.3f})."
		)
		return True

	def _normal_db_context(self) -> tuple[str, str, str, dict]:
		sex = "male" if self.normal_sex_combo.currentText() == "Hombre" else "female"
		protocol = "stress" if self.normal_protocol_combo.currentText() == "Stress" else "rest"
		dataset = self.normal_db_combo.currentText()
		nd = normal_db.evaluate(
			float(self.metrics.get("phase_sd", 0.0)),
			float(self.metrics.get("bandwidth", 0.0)),
			entropy_normalized_pct=float(self.metrics.get("entropy_normalized_pct", np.nan)),
			dataset=dataset,
			sex=sex,
			protocol=protocol,
		)
		return dataset, sex, protocol, nd

	def _format_normal_metric_line(self, metric: dict, label: str, unit: str) -> str:
		if not metric.get("available"):
			return f"  {label}: sin referencia en la DB"
		flag = "ANORMAL" if metric.get("abnormal") else "normal"
		value = float(metric.get("value", np.nan))
		cutoff = metric.get("cutoff")
		mean = metric.get("mean")
		sd = metric.get("sd")
		zt = f"{metric['z']:+.1f}" if metric.get("z") is not None else "n/d"
		if mean is not None and sd is not None:
			ref_txt = f"normal {float(mean):.1f}±{float(sd):.1f}"
		else:
			ref_txt = "normal según límite superior"
		cut_txt = f"cutoff {float(cutoff):.1f}{unit}" if cutoff is not None else "cutoff n/d"
		return f"  {label}: {value:.1f}{unit} | {ref_txt} | {cut_txt} | z={zt} → {flag}"

	def _write_ungated_output(self):
		"""Genera la imagen de perfusión desgatillada (UngRaw) en el directorio de salida."""
		if self.study is None:
			return
		from core.ungating import ungate, ungate_stats
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		ug = ungate(self.study.cube)
		stats = ungate_stats(self.study.cube)
		n_slices = ug.shape[0]
		cols = int(np.ceil(np.sqrt(n_slices)))
		rows = int(np.ceil(n_slices / cols))

		fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
		axes = np.atleast_1d(axes).ravel()
		vmax = float(ug.max()) if ug.size else 1.0
		for idx in range(rows * cols):
			ax = axes[idx]
			ax.axis("off")
			if idx < n_slices:
				ax.imshow(ug[idx], cmap=str(self.report_cmap_polar_perf.currentText()), vmin=0, vmax=vmax)
				ax.set_title(f"{idx + 1}", fontsize=7, color="white", pad=1)
		ctx_label = self._study_context_label(
			path_override=str(getattr(self, "_output_study_path_override", "") or self.file_edit.text().strip()),
			study_obj=self.study,
		)
		fig.suptitle(
			f"Desgatillado (UngRaw) — {ctx_label}\n"
			f"counts totales {stats.get('total_counts', 0):.0f} | CV gates {stats.get('counts_per_gate_cv_pct', 0):.1f}%",
			fontsize=10, fontweight="bold", color="white",
		)
		fig.patch.set_facecolor("#0b1220")
		fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
		self._stamp_export_figure(fig, None)
		out_path = os.path.join(self.output_dir, "perfusion_ungated.png")
		fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
		plt.close(fig)
		self._log(f"Desgatillado generado: perfusion_ungated.png (counts {stats.get('total_counts', 0):.0f})")

	def _export_ungated_dicom(self):
		"""Exporta el desgatillado como DICOM NM no-gated (UngRaw) para compartir/releer."""
		if self.study is None:
			QMessageBox.information(self, "SINCRO", "Primero cargá un estudio.")
			return
		from core.ungating import ungate
		from core.dicom_export import save_ungated_dicom
		default_name = os.path.join(self.output_dir, "ungated_perfusion.dcm")
		path, _ = QFileDialog.getSaveFileName(
			self,
			"Guardar desgatillado DICOM",
			default_name,
			"DICOM (*.dcm);;Todos (*.*)",
		)
		if not path:
			return
		try:
			ug = ungate(self.study.cube)
			save_ungated_dicom(ug, path, source_study=self.study)
			self._log(f"Desgatillado DICOM guardado: {path}")
			self.statusBar().showMessage("Desgatillado DICOM guardado.")
			try:
				get_logger().log_export("dcm_ungated", path)
			except Exception:
				pass
		except Exception as exc:
			QMessageBox.critical(self, "Error exportando DICOM", str(exc))
			self._log(f"[ERROR] Exportar desgatillado: {exc}")

	def _compute_perfusion_texture(self):
		"""Textura GLCM de perfusión por segmento AHA + tabla combinada perfusión+fase.

		Perfusión = media de gates del cubo actual (short-axis). Devuelve
		(texture_by_seg, perfusion_phase_rows) o (None, None) si falta info.
		"""
		if self.aha is None or self.study is None or not isinstance(self.phase_by_seg, dict):
			return None, None
		segment_map = getattr(self.aha, "segment_map", None)
		if segment_map is None:
			return None, None
		cube = self.study.cube
		if cube is None or cube.ndim != 4:
			return None, None
		perfusion = np.asarray(cube, dtype=np.float64).mean(axis=0)  # (n_slices, H, W)
		if perfusion.shape != segment_map.shape:
			return None, None
		texture_by_seg = perfusion_texture_by_segment(perfusion, segment_map)
		perfusion_phase_rows = combine_perfusion_phase(texture_by_seg, self.phase_by_seg)
		return texture_by_seg, perfusion_phase_rows

	def _rest_stage_state(self):
		"""StageState de reposo si existe y no es el mismo estudio primario."""
		try:
			state = self._dual_session().stage("rest")
		except Exception:
			return None
		if state is None or state.cut_study is None or state.cut_study is self.study:
			return None
		return state

	def _stress_rest_for_reports(self, ef):
		"""Comparación stress-rest para informes.

		Usa compare_bundle si está; si el bundle fue regenerado/limpiado por el
		render diferido, cae al StageState canónico de reposo (que persiste
		metrics/territory/ef). Sin esto, el informe perdía la FEVI de reposo.
		"""
		if self.metrics is None:
			return None
		rest_metrics = rest_territory = rest_ef = None
		if self.compare_bundle is not None:
			rest_metrics = self.compare_bundle.get("metrics")
			rest_territory = self.compare_bundle.get("territory")
			rest_ef = getattr(self, "compare_ef", None) or self.compare_bundle.get("ef")
		if rest_metrics is None:
			rest_metrics = getattr(self, "compare_metrics", None)
		state = self._rest_stage_state()
		if state is not None:
			if rest_metrics is None:
				rest_metrics = state.metrics
			if rest_territory is None:
				rest_territory = state.territory
			if not (rest_ef and rest_ef.get("available")):
				rest_ef = state.ef or rest_ef
		if not rest_metrics:
			return None
		return compare_stress_rest(
			self.metrics,
			rest_metrics,
			self.territory,
			rest_territory,
			ef,
			rest_ef,
		)

	def _compute_perfusion_quant(self):
		"""Cuantificación relativa de perfusión por segmento AHA (esf. + rep. si hay)."""
		if self.study is None or self.aha is None:
			return None
		seg_map = getattr(self.aha, "segment_map", None)
		cube = getattr(self.study, "cube", None)
		if seg_map is None or cube is None or np.asarray(cube).ndim != 4:
			return None
		stress_perf = np.asarray(cube, dtype=np.float64).mean(axis=0)
		if stress_perf.shape != np.asarray(seg_map).shape:
			return None
		stress_pct = perfusion_by_segment(stress_perf, seg_map)
		rest_pct = None
		rest_study = rest_aha = None
		if self.compare_bundle is not None:
			rest_study = self.compare_bundle.get("study")
			rest_aha = self.compare_bundle.get("aha")
		if rest_study is None:
			state = self._rest_stage_state()
			if state is not None and state.seg is not None:
				rest_study = state.cut_study
				try:
					rest_aha = map_to_17_segments(state.seg)
				except Exception:
					rest_aha = None
		if rest_study is not None and rest_aha is not None:
			r_map = getattr(rest_aha, "segment_map", None)
			r_cube = getattr(rest_study, "cube", None)
			if r_map is not None and r_cube is not None and np.asarray(r_cube).ndim == 4:
				r_perf = np.asarray(r_cube, dtype=np.float64).mean(axis=0)
				if r_perf.shape == np.asarray(r_map).shape:
					rest_pct = perfusion_by_segment(r_perf, r_map)
		return perfusion_quant_summary(stress_pct, rest_pct)

	def _export_structured_results(self):
		"""Exporta resultados a JSON/CSV/Excel en el directorio de salida."""
		if self.study is None or self.metrics is None:
			return

		# Metadatos del estudio
		ctx = self._study_context(
			path_override=str(getattr(self, "_output_study_path_override", "") or self.file_edit.text().strip()),
			study_obj=self.study,
		)
		study_meta = {
			"patient_name": ctx["patient_name"],
			"patient_id": ctx["patient_id"],
			"patient_sex": str(getattr(self.study, "patient_sex", "") or ""),
			"study_date": ctx["study_date"],
			"study_description": str(getattr(self.study, "study_description", "") or ""),
			"series_description": str(getattr(self.study, "series_description", "") or ""),
			"dimensions": f"{self.study.cube.shape[0]}x{self.study.cube.shape[1]}x{self.study.cube.shape[2]}x{self.study.cube.shape[3]}",
		}

		# Info de segmentación
		seg_info = {
			"method": str(getattr(self.seg, "method", "N/D")),
			"n_voxels": int(getattr(self.seg, "n_voxels", 0)),
			"n_slices": int(self.study.cube.shape[1]),
		}

		# Parámetros de procesamiento
		proc_params = {
			"seg_method": str(self.seg_method.currentText()),
			"roi_source": self.roi_source(),
			"cavity_center": bool(self.cavity_center_enabled()),
			"threshold": float(self.threshold_spin.value()),
			"smooth_sigma": float(self.sigma_spin.value()),
			"harmonics": int(self.harmonics_spin.value()),
			"amp_filter": float(self.phase_threshold_spin.value()),
		}

		# Robustez
		robustness = {
			"segmental_aha": self.metrics.get("segmental_aha"),
			"bootstrap": self.metrics.get("bootstrap"),
			"roi_sensitivity": self.metrics.get("roi_sensitivity"),
		}

		# Evaluación DB normal
		try:
			dataset, sex, protocol, nd = self._normal_db_context()
			normal_db_eval = nd
		except Exception:
			normal_db_eval = None

		# QC
		qc_info = self.phase_qc if self.phase_qc else None

		# Datos de investigación por segmento AHA / territorio / textura / stress-rest.
		phase_by_seg = self.phase_by_seg if isinstance(self.phase_by_seg, dict) else None
		territory = self.territory if isinstance(self.territory, dict) else None
		n_per_segment = getattr(self.aha, "n_per_segment", None) if self.aha is not None else None
		texture_by_seg = None
		perfusion_phase_rows = None
		stress_rest = None
		try:
			texture_by_seg, perfusion_phase_rows = self._compute_perfusion_texture()
		except Exception as exc:
			self._log(f"Textura de perfusión no disponible para export: {exc}")
		try:
			stress_rest = self._stress_rest_for_reports(ef)
		except Exception as exc:
			self._log(f"Comparación stress-rest no disponible para export: {exc}")

		# Nombre base
		patient_id = study_meta.get("patient_id", "unknown")
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		base_name = f"gammasync_{patient_id}_{timestamp}"

		# Exportar
		results = export_all(
			self.output_dir,
			study_meta,
			self.metrics,
			seg_info,
			proc_params,
			robustness,
			normal_db_eval,
			qc_info,
			base_name,
			phase_by_seg,
			territory,
			n_per_segment,
			texture_by_seg,
			perfusion_phase_rows,
			stress_rest,
		)

		# Log
		try:
			logger = get_logger()
			for fmt, path in results.items():
				if path:
					logger.log_export(fmt, path)
		except Exception:
			pass

		self._log(f"Exportación estructurada: {', '.join(k for k, v in results.items() if v)}")

	def _refresh_summary(self):
		if self.study is None or self.metrics is None:
			return

		vol = self._compute_volumes_ml()
		ef = self._estimate_lv_ef()
		vol = self._harmonize_volumes_with_ef(vol, ef)
		ctx = self._study_context(
			path_override=str(getattr(self, "_output_study_path_override", "") or self.file_edit.text().strip()),
			study_obj=self.study,
		)
		# El resumen técnico termina más abajo, pero el box "Resultados en vivo"
		# y sus curvas deben refrescarse EN ESTA llamada. Sin esto, al completar
		# reposo desde memoria quedaban mostrando el render previo de esfuerzo.
		self._refresh_readonly_results_panel()

		clinical = []
		clinical.append(f"Visualizando: {ctx['phase']}")
		clinical.append(f"Paciente: {ctx['patient_name']}  |  ID: {ctx['patient_id']}  |  Fecha: {ctx['study_date']}")
		clinical.append("")
		clinical.append(f"Resultado robusto GammaSync (amp {self.metrics.get('amp_filter', CLINICAL_PHASE_AMP_FILTER_DEFAULT):.2f})")
		clinical.append(f"  Clase PSD técnica: {self.metrics.get('technical_classification', self.metrics.get('classification'))} (orientativa, no diagnóstica)")
		clinical.append(f"  Phase SD: {self.metrics.get('phase_sd')}°")
		clinical.append(f"  Bandwidth: {self.metrics.get('bandwidth')}°")
		clinical.append(f"  Entropy Shannon: {self.metrics.get('entropy_shannon_bits', self.metrics.get('entropy'))} bits")
		clinical.append(f"  Entropy normalizada: {self.metrics.get('entropy_normalized_pct', 'N/D')}%")
		segm = self.metrics.get("segmental_aha") or {}
		if segm.get("available"):
			clinical.append(f"  Modo segmentario AHA: PSD {segm.get('phase_sd')}° | BW {segm.get('bandwidth')}° | n={segm.get('n_segments')} segmentos")
		boot = self.metrics.get("bootstrap") or {}
		if boot.get("available"):
			psd_boot = boot.get("phase_sd", {})
			bw_boot = boot.get("bandwidth", {})
			clinical.append(
				f"  Bootstrap voxel IC95: PSD {psd_boot.get('ci95_low')}–{psd_boot.get('ci95_high')}° | "
				f"BW {bw_boot.get('ci95_low')}–{bw_boot.get('ci95_high')}° ({boot.get('n_iter')} muestras)"
			)
		roi_sens = self.metrics.get("roi_sensitivity") or {}
		if roi_sens.get("available"):
			clinical.append(
				f"  Sensibilidad ROI ±{roi_sens.get('delta_px')} px: PSD {roi_sens.get('phase_sd_min')}–{roi_sens.get('phase_sd_max')}° | "
				f"BW {roi_sens.get('bandwidth_min')}–{roi_sens.get('bandwidth_max')}°"
			)
			if roi_sens.get("warn"):
				clinical.append("  QC: métricas sensibles a pequeñas variaciones de ROI; revisar máscara antes de concluir.")
		if self.metrics_raw is not None and self.phase_qc:
			clinical.append("")
			clinical.append("Control QC en pantalla")
			clinical.append(
				f"  Resultado crudo amp {self.metrics_raw.get('amp_filter', RAW_PHASE_QC_AMP_FILTER):.2f}: "
				f"{self.metrics_raw.get('classification')} "
				f"(PSD {self.metrics_raw.get('phase_sd')}°, BW {self.metrics_raw.get('bandwidth')}°)"
			)
			if self.phase_qc.get("class_changed"):
				clinical.append(
					f"  Cambio de clase: {self.phase_qc.get('raw_classification')} → {self.phase_qc.get('clinical_classification')}"
				)
			if self.phase_qc.get("warn"):
				clinical.append("  Interpretación QC: posible sobreestimación cruda por voxels de baja amplitud.")
			clinical.append("  Nota: el resultado crudo/QC no se incluye en el informe PDF.")
		clinical.append("")

		# Comparación contra base de datos normal (por sexo/protocolo/software).
		try:
			dataset, _sex, _protocol, nd = self._normal_db_context()
			clinical.append(f"Interpretación clínica vs DB [{dataset} · {self.normal_sex_combo.currentText()} · {self.normal_protocol_combo.currentText()}]")
			clinical.append(self._format_normal_metric_line(nd["metrics"].get("phase_sd", {}), "PSD", "°"))
			clinical.append(self._format_normal_metric_line(nd["metrics"].get("bandwidth", {}), "BW", "°"))
			if "entropy_normalized_pct" in nd["metrics"]:
				clinical.append(self._format_normal_metric_line(nd["metrics"].get("entropy_normalized_pct", {}), "Entropy norm.", "%"))
			clinical.append(f"  Lectura vs DB: {'fuera de referencia' if nd.get('dyssynchrony') else 'dentro de referencia'}")
			clinical.append("  Alcance: asincronía mecánica intraventricular del VI; correlacionar con QRS, FEVI, perfusión, viabilidad y clínica.")
		except Exception:
			clinical.append("Vs DB normal: no disponible")
		clinical.append("")
		clinical.append("Volúmenes")
		if vol["myocardial_ml"] is not None:
			myo_src = str(vol.get("myo_source", ""))
			src_txt = " (máscara clínica por amplitud)" if myo_src == "clinical_amp" else " (máscara de segmentación)"
			clinical.append(f"  Miocardio: {vol['myocardial_ml']:.2f} mL{src_txt}")
			if not vol.get("myo_physiologic", True):
				clinical.append(
					"  ⚠ Volumen miocárdico fuera de rango fisiológico esperado (~85–250 mL): "
					"revisar máscara/umbral de segmentación antes de interpretar."
				)
		if vol["cavity_ml"] is not None:
			cav_src = str(vol.get("cavity_source", ""))
			cav_txt = " (referencia EDV, gate de volumen máximo)" if cav_src == "edv_gate_max" else ""
			clinical.append(f"  Cavidad: {vol['cavity_ml']:.2f} mL{cav_txt}")
		if vol["lv_total_ml"] is not None:
			clinical.append(f"  Total VI: {vol['lv_total_ml']:.2f} mL")
		if vol["cavity_to_myo_ratio"] is not None:
			clinical.append(f"  Índice cavidad/miocardio: {vol['cavity_to_myo_ratio']:.3f}")
		if vol["myocardial_ml"] is None:
			clinical.append("  No disponibles (faltan metadatos geométricos DICOM).")

		if self.compare_metrics is not None:
			clinical.append("")
			clinical.append(f"Comparación disincronía vs {self.compare_label or 'otro estudio'}")
			math_text = self._polar_compare_operation_text()
			if math_text:
				clinical.append(f"  Operación polar aplicada: {math_text}")
			d_psd = float(self.metrics.get("phase_sd", 0.0)) - float(self.compare_metrics.get("phase_sd", 0.0))
			d_bw = float(self.metrics.get("bandwidth", 0.0)) - float(self.compare_metrics.get("bandwidth", 0.0))
			clinical.append(f"  Δ Phase SD: {d_psd:+.2f}°   Δ Bandwidth: {d_bw:+.2f}°")
			if d_psd > 3.0 and d_bw > 8.0:
				clinical.append("  → Δ marcado: posible stunning isquémico post-stress.")
			elif abs(d_psd) <= 3.0 and abs(d_bw) <= 8.0:
				clinical.append("  → Sincronía estable entre estudios.")
			else:
				clinical.append("  → Diferencia intermedia: correlacionar con clínica.")

		clinical.append("")
		clinical.append(f"FEVI — método: {self.fevi_method_label()}")
		if ef.get("available"):
			clinical.append(f"  EDV: {float(ef['edv_ml']):.2f} mL (gate {int(ef['ed_gate'])})")
			clinical.append(f"  ESV: {float(ef['esv_ml']):.2f} mL (gate {int(ef['es_gate'])})")
			clinical.append(f"  SV: {float(ef['sv_ml']):.2f} mL")
			clinical.append(f"  FEVI: {float(ef['ef_pct']):.1f}%")
			if ef.get("pfr_text"):
				clinical.append(f"  PFR (llenado pico): {ef.get('pfr_text')}")
			if ef.get("tvmax_text"):
				clinical.append(f"  TVmáx (tiempo a pico llenado): {ef.get('tvmax_text')}")
			if ef.get("ef_pct_converted") is not None:
				clinical.append(
					f"  Equivalente en escala {ef.get('regression_label')}: "
					f"{float(ef['ef_pct_converted']):.1f}% "
					"(solo para comparar con informes de ese equipo)"
				)
			if ef.get("ed_wall_thickness_mm") is not None:
				clinical.append(f"  Espesor de pared en ED usado: {float(ef['ed_wall_thickness_mm']):.1f} mm")
			if ef.get("valve_removed_ml"):
				clinical.append(
					f"  Plano valvular de 2 piezas ({float(ef.get('valve_offset_mm', 0.0)):.1f} mm septal): "
					f"-{float(ef['valve_removed_ml']):.1f} mL de base"
				)
			elif ef.get("valve_plane") is False:
				clinical.append("  Plano valvular desactivado: base cortada con plano perpendicular")
			if ef.get("myocardial_mass_g") is not None:
				clinical.append(f"  Masa miocárdica: {float(ef['myocardial_mass_g']):.1f} g")
			if ef.get("thickening_pct") is not None:
				thk_ed = ef.get("wall_thickness_ed_mm")
				thk_es = ef.get("wall_thickness_es_mm")
				if thk_ed is not None and thk_es is not None:
					clinical.append(
						f"  Engrosamiento sistólico (ED→ES): {float(ef['thickening_pct']):+.1f}% "
						f"({float(thk_ed):.1f} → {float(thk_es):.1f} mm)"
					)
				else:
					clinical.append(f"  Engrosamiento sistólico: {float(ef['thickening_pct']):+.1f}%")
				compare_ef = getattr(self, "compare_ef", None)
				if compare_ef and compare_ef.get("available") and compare_ef.get("thickening_pct") is not None:
					d_thick = float(ef["thickening_pct"]) - float(compare_ef["thickening_pct"])
					cur_lbl = ctx.get("phase", "Actual")
					cmp_lbl = self.compare_label or "Comparación"
					clinical.append(
						f"  Engrosamiento {cur_lbl} vs {cmp_lbl}: {float(ef['thickening_pct']):+.1f}% vs "
						f"{float(compare_ef['thickening_pct']):+.1f}%   (Δ {d_thick:+.1f} puntos)"
					)
					if d_thick < -10.0:
						clinical.append(
							"    → El engrosamiento cayó marcadamente respecto de la otra etapa: "
							"correlacionar con perfusión regional (posible isquemia/stunning)."
						)
			if ef.get("shape_index_ed") is not None:
				clinical.append(
					f"  Índice de esfericidad ED/ES: {float(ef['shape_index_ed']):.2f} / "
					f"{float(ef['shape_index_es']):.2f} "
					f"(eje corto {float(ef.get('short_axis_ed_mm', 0.0)):.0f} mm / "
					f"eje largo {float(ef.get('long_axis_mm', 0.0)):.0f} mm). "
					"Más cerca de 1 = ventrículo más esférico (remodelado)."
				)
			if ef.get("fallback_from"):
				clinical.append(
					f"  Aviso: el método ECTb no pudo cuantificar ({ef.get('fallback_reason')}); "
					"se usó el método anterior."
				)
			clinical.append("  Nota: estimación preliminar de investigación.")
			# Aviso base/ápex: la elección del límite basal afecta MUCHO EDV/ESV/FEVI.
			# Un corte basal más bajo incluye más volumen (FEVI más baja) pero puede
			# contaminar la fase con actividad valvular. Se informa el límite usado.
			try:
				n_sa = int(getattr(self.study, "cube", np.empty((0,))).shape[1]) if self.study is not None else 0
				base_v = int(self.cine_crudo_cut_base_spin.value()) if hasattr(self, "cine_crudo_cut_base_spin") else None
				apex_v = int(self.cine_crudo_cut_apex_spin.value()) if hasattr(self, "cine_crudo_cut_apex_spin") else None
				if base_v is not None and apex_v is not None:
					clinical.append(
						f"  Límite SA usado: base={base_v} ápex={apex_v} ({n_sa} cortes válidos)."
					)
				clinical.append(
					"  Sensibilidad base/ápex: el límite basal afecta fuertemente EDV/ESV/FEVI; "
					"bajarlo incluye más volumen basal (FEVI menor) pero puede añadir ruido de fase valvular. "
					"Ajustar base/ápex y reprocesar si la FEVI no es plausible."
				)
			except Exception:
				pass
			# Aviso QC por gate: si el usuario editó ROIs manuales por gate, la FEVI
			# usa esa geometría manual en esos gates (en vez de la automática).
			try:
				if hasattr(self, "cine") and self.cine is not None and self.cine.per_gate_roi_mode_enabled():
					n_gate_rois = len(getattr(self.cine, "_rois_by_gate", {}) or {})
					if n_gate_rois > 0:
						clinical.append(
							f"  QC por gate: activo ({n_gate_rois} ROIs manuales por gate). "
							"La FEVI usa la geometría manual editada en esos gates."
						)
			except Exception:
				pass
		else:
			clinical.append("  No disponible (segmentación/metadata insuficiente).")

		clinical.append("")
		clinical.append("Territorios coronarios")
		for name, data in self.territory.items():
			clinical.append(f"  {name}: mean={data['mean']:.1f}°, SD={data['std']:.1f}°, n={data['n']}")

		# Delta stress-rest de fase (solo si hay estudio de comparación cargado).
		# Es la señal con mayor respaldo predictivo (Fukumoto 2025 entropy /
		# Tanaka 2025 bandwidth). Ver core.stress_rest.
		if self.compare_bundle is not None:
			try:
				sr = compare_stress_rest(
					self.metrics,
					self.compare_bundle.get("metrics"),
					self.territory,
					self.compare_bundle.get("territory"),
				)
				if sr.get("available"):
					d = sr["deltas"]
					clinical.append("")
					clinical.append("Delta stress-rest de fase (esfuerzo - reposo)")
					clinical.append(
						f"  Phase SD: {d.get('phase_sd', float('nan')):+.1f}°  |  "
						f"Bandwidth: {d.get('bandwidth', float('nan')):+.1f}°  |  "
						f"Entropy: {d.get('entropy_normalized_pct', float('nan')):+.1f}%"
					)
					for note in sr.get("notes", []):
						clinical.append(f"  • {note}")
			except Exception:
				pass

		technical = []
		technical.append("Identificación")
		technical.append(f"  Fase visualizada: {ctx['phase']}")
		technical.append(f"  Paciente: {ctx['patient_name']}")
		technical.append(f"  Patient ID: {ctx['patient_id']}")
		technical.append(f"  Fecha estudio: {ctx['study_date']}")
		technical.append("")
		technical.append("Estudio cargado")
		technical.append(self.study.summary())
		technical.append("")
		technical.append("Parámetros de procesamiento")
		technical.append(f"  Segmentación: {self.seg.method}")
		technical.append(f"  Threshold: {self.threshold_spin.value():.2f}")
		technical.append(f"  Smooth sigma: {self.sigma_spin.value():.1f}")
		technical.append(f"  Harmonics: {self.harmonics_spin.value()}")
		technical.append(f"  Amp filter clínico: {self.phase_threshold_spin.value():.2f}")
		technical.append(f"  Amp filter crudo QC: {RAW_PHASE_QC_AMP_FILTER:.2f}")
		technical.append(f"  Normalize reference: {'sí' if self.normalize_check.isChecked() else 'no'}")
		if vol["voxel_ml"] is not None:
			technical.append(f"  Volumen voxel: {vol['voxel_ml']:.4f} mL")
		technical.append("")
		technical.append("Métricas técnicas")
		for key in ["mean_phase", "phase_sd", "bandwidth", "entropy_shannon_bits", "entropy_normalized_pct", "skewness", "kurtosis", "asynchrony_index", "peak_phase", "peak_width", "latest_activation_phase", "technical_classification"]:
			technical.append(f"  {key}: {self.metrics.get(key)}")
		segm = self.metrics.get("segmental_aha") or {}
		if segm.get("available"):
			technical.append("  Segmentario AHA:")
			for key in ["n_segments", "phase_sd", "bandwidth", "entropy_normalized_pct", "technical_classification"]:
				technical.append(f"    {key}: {segm.get(key)}")
		boot = self.metrics.get("bootstrap") or {}
		if boot.get("available"):
			technical.append(f"  Bootstrap voxel: n={boot.get('n_iter')} sample={boot.get('sample_size')}/{boot.get('n_voxels')}")
			technical.append(f"    PSD IC95: {boot.get('phase_sd', {}).get('ci95_low')}–{boot.get('phase_sd', {}).get('ci95_high')}°")
			technical.append(f"    BW IC95: {boot.get('bandwidth', {}).get('ci95_low')}–{boot.get('bandwidth', {}).get('ci95_high')}°")
		roi_sens = self.metrics.get("roi_sensitivity") or {}
		if roi_sens.get("available"):
			technical.append(f"  Sensibilidad ROI: warn={roi_sens.get('warn')} ΔPSDmax={roi_sens.get('max_phase_sd_delta')}° ΔBWmax={roi_sens.get('max_bandwidth_delta')}°")
			for row in roi_sens.get("variants", []):
				if "error" in row:
					technical.append(f"    {row.get('label')}: ERROR {row.get('error')}")
				else:
					technical.append(f"    {row.get('label')}: PSD {float(row.get('phase_sd', np.nan)):.1f}° | BW {float(row.get('bandwidth', np.nan)):.1f}° | vox {row.get('phase_voxels')}")
		if self.metrics_raw is not None:
			technical.append("")
			technical.append("QC crudo (solo pantalla; no informe)")
			for key in ["phase_sd", "bandwidth", "entropy_shannon_bits", "entropy_normalized_pct", "peak_phase", "technical_classification", "n_voxels_kept", "n_voxels_total"]:
				technical.append(f"  raw_{key}: {self.metrics_raw.get(key)}")
		if self.phase_qc:
			technical.append("")
			technical.append("QC estabilidad por amplitud")
			for key in ["raw_filter", "clinical_filter", "class_changed", "low_confidence_tail_pct", "low_confidence_tail_n", "raw_voxels", "clinical_voxels", "total_voxels", "warn"]:
				technical.append(f"  {key}: {self.phase_qc.get(key)}")

		self.summary_clinical.setPlainText("\n".join(clinical))
		self.summary_technical.setPlainText("\n".join(technical))

		# Resumen ejecutivo (síntesis del hallazgo en lenguaje natural).
		try:
			exec_db_eval = None
			try:
				_dataset, _sex, _protocol, exec_db_eval = self._normal_db_context()
			except Exception:
				exec_db_eval = None
			exec_summary = build_executive_summary(
				metrics=self.metrics,
				ef=ef,
				territory=self.territory,
				volumes=vol,
				phase_label=str(ctx.get("phase", "Estudio")),
				db_eval=exec_db_eval,
			)
			self.summary_executive.setPlainText(exec_summary.get("plain_text", ""))
		except Exception:
			self.summary_executive.setPlainText("")

	def _perfusion_image_for_cube(self, cube, ed_gate):
		"""Imagen de perfusión 3D (n_slices,H,W) para el bull's-eye segmentario,
		según la fuente configurada en Configuración → Análisis.

		- Gate ED (estándar): usa el gate de fin de diástole (mejor definición del
		  borde endocárdico, convención de consolas clínicas).
		- Media de gates: promedia todos los gates (mayor estadística de conteos).
		"""
		import numpy as _np
		arr = _np.asarray(cube, dtype=_np.float64)
		if arr.ndim != 4 or arr.shape[0] == 0:
			return None
		if self.perfusion_source() == self.PERFUSION_SOURCE_MEAN:
			return arr.mean(axis=0)
		g = (int(ed_gate) - 1) if ed_gate else 0
		g = max(0, min(arr.shape[0] - 1, g))
		return arr[g]

	def _render_guia_fase_vi(self, style, primary_cube_render, active_cine_widget, ef, cmap_bullseye):
		"""Panel "Guía para fase VI": bull's-eye doble (fase + perfusión/viabilidad)
		y tabla segmentaria AHA-17. Si hay estudio de comparación, dibuja reposo y
		esfuerzo en la misma imagen con Δfase. Estilo propio (panel funcional)."""
		import numpy as _np
		import matplotlib
		import matplotlib.pyplot as plt
		from matplotlib.patches import Circle, Wedge
		from matplotlib.colors import Normalize

		if not isinstance(self.phase_by_seg, dict) or self.aha is None:
			return

		def _circular_delta(cur, ref):
			return float(((float(cur) - float(ref) + 180.0) % 360.0) - 180.0)

		# --- Datos por etapa -------------------------------------------------
		stages: list[dict] = []
		# Etapa primaria.
		seg_map_primary = _np.asarray(self.aha.segment_map, dtype=_np.int32)
		perf_primary = self._perfusion_image_for_cube(primary_cube_render, ef.get("ed_gate"))
		report_primary = build_segmental_report(
			self.phase_by_seg, perf_primary, seg_map_primary,
		)
		stages.append({"label": self._guia_stage_label(is_compare=False), "report": report_primary})

		# Etapa de comparación (si existe).
		if self.compare_bundle is not None:
			try:
				comp_phase = self.compare_bundle.get("phase_by_seg")
				comp_aha = self.compare_bundle.get("aha")
				comp_study = self.compare_bundle.get("study")
				comp_ef = getattr(self, "compare_ef", None) or {}
				if isinstance(comp_phase, dict) and comp_aha is not None and comp_study is not None:
					seg_map_comp = _np.asarray(comp_aha.segment_map, dtype=_np.int32)
					perf_comp = self._perfusion_image_for_cube(comp_study.cube, comp_ef.get("ed_gate"))
					report_comp = build_segmental_report(comp_phase, perf_comp, seg_map_comp)
					stages.append({"label": self._guia_stage_label(is_compare=True), "report": report_comp})
			except Exception as exc:
				self._log(f"Guía fase VI: etapa de comparación omitida ({exc}).")

		dual = len(stages) == 2

		# --- Geometría del bull's-eye (AHA-17) -------------------------------
		rings = [
			([1, 2, 3, 4, 5, 6], 0.68, 0.98, 90.0),
			([7, 8, 9, 10, 11, 12], 0.40, 0.68, 90.0),
			([13, 14, 15, 16], 0.18, 0.40, 45.0),
		]

		def _draw_bullseye(ax, values_by_seg, cmap, norm, *, latest_seg=None, fmt="{:.0f}", nan_face=(0.22, 0.24, 0.28, 1.0)):
			ax.set_xlim(-1.10, 1.10)
			ax.set_ylim(-1.10, 1.10)
			ax.set_aspect("equal")
			ax.axis("off")

			def _face(seg_id):
				v = values_by_seg.get(int(seg_id), _np.nan)
				if v is None or not _np.isfinite(v):
					return nan_face
				return cmap(norm(float(v)))

			def _edge(seg_id):
				return ("#ffd24a" if (latest_seg is not None and int(seg_id) == int(latest_seg)) else style["grid"])

			def _lw(seg_id):
				return 2.8 if (latest_seg is not None and int(seg_id) == int(latest_seg)) else 1.3

			for seg_ids, r_in, r_out, start in rings:
				n = len(seg_ids)
				for i, sid in enumerate(seg_ids):
					t1 = start - (i + 1) * (360.0 / n)
					t2 = start - i * (360.0 / n)
					ax.add_patch(Wedge((0.0, 0.0), r_out, t1, t2, width=r_out - r_in,
						facecolor=_face(sid), edgecolor=_edge(sid), linewidth=_lw(sid)))
					mid_a = _np.deg2rad((t1 + t2) * 0.5)
					r_t = (r_in + r_out) * 0.5
					v = values_by_seg.get(int(sid), _np.nan)
					txt = fmt.format(float(v)) if (v is not None and _np.isfinite(v)) else "—"
					ax.text(r_t * _np.cos(mid_a), r_t * _np.sin(mid_a), txt,
						color=style["fg"], fontsize=7.2, ha="center", va="center", fontweight="bold")
			# Ápice (segmento 17).
			ax.add_patch(Circle((0.0, 0.0), radius=0.18, facecolor=_face(17),
				edgecolor=_edge(17), linewidth=_lw(17)))
			v17 = values_by_seg.get(17, _np.nan)
			t17 = fmt.format(float(v17)) if (v17 is not None and _np.isfinite(v17)) else "—"
			ax.text(0.0, 0.0, t17, color=style["fg"], fontsize=7.2, ha="center", va="center", fontweight="bold")

		# --- Figura ----------------------------------------------------------
		n_bulls = 2 * len(stages)  # fase + perfusión por etapa
		fig = plt.figure(figsize=(6.6 if not dual else 12.6, 10.5), facecolor=style["fig_bg"])
		gs = fig.add_gridspec(2, n_bulls, height_ratios=[1.0, 1.3], hspace=0.22, wspace=0.12)

		phase_cmap = matplotlib.colormaps.get_cmap("twilight")
		phase_norm = Normalize(vmin=0.0, vmax=360.0)
		perf_cmap = matplotlib.colormaps.get_cmap(cmap_bullseye)
		perf_norm = Normalize(vmin=0.0, vmax=100.0)

		col = 0
		for stage in stages:
			rep = stage["report"]
			phase_by = {r["segment"]: r["phase_deg"] for r in rep["rows"]}
			perf_by = {r["segment"]: r["perfusion_pct"] for r in rep["rows"]}
			latest = rep.get("latest_segment")

			ax_ph = fig.add_subplot(gs[0, col])
			ax_ph.set_facecolor(style["fig_bg"])
			_draw_bullseye(ax_ph, phase_by, phase_cmap, phase_norm, latest_seg=latest, fmt="{:.0f}")
			ax_ph.set_title(f"{stage['label']} — Fase (°)", color=style["fg"], fontsize=11, fontweight="bold", pad=6)
			col += 1

			ax_pf = fig.add_subplot(gs[0, col])
			ax_pf.set_facecolor(style["fig_bg"])
			_draw_bullseye(ax_pf, perf_by, perf_cmap, perf_norm, fmt="{:.0f}")
			ax_pf.set_title(f"{stage['label']} — Perfusión (%)", color=style["fg"], fontsize=11, fontweight="bold", pad=6)
			col += 1

		# --- Tabla segmentaria ----------------------------------------------
		ax_tbl = fig.add_subplot(gs[1, :])
		ax_tbl.axis("off")
		terr_colors = {"LAD": "#3a2f18", "LCx": "#18303a", "RCA": "#2a183a", "N/D": style["ax_bg"]}

		rows_primary = {r["segment"]: r for r in stages[0]["report"]["rows"]}
		rows_compare = {r["segment"]: r for r in stages[1]["report"]["rows"]} if dual else {}
		latest_primary = stages[0]["report"].get("latest_segment")

		if dual:
			headers = ["Seg", "Nombre", "Terr.", "Fase R°", "Fase E°", "Δfase°", "Perf R%", "Perf E%", "Viab E"]
		else:
			headers = ["Seg", "Nombre", "Territorio", "Fase °", "Perf %", "Viabilidad"]

		def _fmt_num(v, suf=""):
			return (f"{float(v):.0f}{suf}") if (v is not None and _np.isfinite(v)) else "—"

		table_rows = []
		cell_bg = []
		for seg_id in range(1, 18):
			rp = rows_primary.get(seg_id, {})
			terr = rp.get("territory", "N/D")
			base_bg = terr_colors.get(terr, style["ax_bg"])
			if dual:
				rc = rows_compare.get(seg_id, {})
				dphase = _circular_delta(rc.get("phase_deg", _np.nan), rp.get("phase_deg", _np.nan)) \
					if (_np.isfinite(rp.get("phase_deg", _np.nan)) and _np.isfinite(rc.get("phase_deg", _np.nan))) else _np.nan
				row = [
					str(seg_id), rp.get("name", str(seg_id)), terr,
					_fmt_num(rp.get("phase_deg", _np.nan)),
					_fmt_num(rc.get("phase_deg", _np.nan)),
					(f"{dphase:+.0f}" if _np.isfinite(dphase) else "—"),
					_fmt_num(rp.get("perfusion_pct", _np.nan)),
					_fmt_num(rc.get("perfusion_pct", _np.nan)),
					rc.get("viability", "N/D"),
				]
			else:
				row = [
					str(seg_id), rp.get("name", str(seg_id)), terr,
					_fmt_num(rp.get("phase_deg", _np.nan)),
					_fmt_num(rp.get("perfusion_pct", _np.nan)),
					rp.get("viability", "N/D"),
				]
			table_rows.append(row)
			cell_bg.append(base_bg)

		table = ax_tbl.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
		table.auto_set_font_size(False)
		table.set_fontsize(8.0)
		table.scale(1.0, 1.35)
		n_cols = len(headers)
		for (r_idx, c_idx), cell in table.get_celld().items():
			cell.set_edgecolor("#4a5568")
			cell.set_linewidth(0.8)
			if r_idx == 0:
				cell.set_facecolor("#1a3a5c")
				cell.set_text_props(color="#ffffff", fontweight="bold", fontsize=8.5)
			else:
				seg_id = r_idx  # fila r_idx (1..17) → segmento
				bg = cell_bg[r_idx - 1]
				if latest_primary is not None and seg_id == int(latest_primary):
					cell.set_facecolor("#4a3a12")
					cell.set_text_props(color="#ffe9a8", fontweight="bold")
				else:
					cell.set_facecolor(bg)
					cell.set_text_props(color="#e2e8f0", fontweight="normal")

		# Título + leyenda.
		src_label = self.perfusion_source_label()
		fig.suptitle(
			f"Guía para fase VI — cruce Fase × Perfusión (AHA-17)   ·   fuente perfusión: {src_label}",
			color=style["fg"], fontsize=12.5, fontweight="bold",
		)
		legend = (
			"Borde dorado = segmento de activación más tardía (fase máxima).   "
			"Viabilidad por % del máximo segmentario: ≥70% viable · 50–70% dudosa · <50% no viable."
		)
		if dual:
			legend += "   Δfase = esfuerzo − reposo (circular)."
		fig.text(0.5, 0.012, legend, ha="center", va="bottom", color=style["subtle"], fontsize=8.2)

		self._stamp_export_figure(fig, active_cine_widget)
		fig.savefig(os.path.join(self.output_dir, "guia_fase_vi.png"), dpi=160,
			bbox_inches="tight", facecolor=fig.get_facecolor())
		plt.close(fig)

	def _guia_stage_label(self, *, is_compare: bool) -> str:
		"""Etiqueta de etapa (Reposo/Esfuerzo) para el panel Guía fase VI. Usa la
		operación marcada en el visor cuando está disponible; cae a genérico."""
		try:
			if self.compare_bundle is None:
				return "Estudio"
			# Convención del módulo: la 1ra etapa suele ser reposo y la 2da esfuerzo.
			return "Esfuerzo" if is_compare else "Reposo"
		except Exception:
			return "Comparación" if is_compare else "Estudio"

	def _write_outputs(self, target_tabs: set[str] | None = None):
		if self.study is None or self.phase_result is None:
			return
		target_tabs_set = None if target_tabs is None else {str(x) for x in target_tabs}
		active_cine_widget = getattr(self, "_output_cine_widget_override", None)
		if active_cine_widget is None:
			active_cine_widget = self.cine
		study_cube_render = self._apply_intestinal_mask_to_cube(self.study.cube, active_cine_widget)

		mid_slice = study_cube_render.shape[1] // 2
		mid_gate = study_cube_render.shape[0] // 2
		frame = study_cube_render[mid_gate, mid_slice]
		frame_norm = frame / (frame.max() + 1e-8)

		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		cmap_slices = str(self.report_cmap_slices.currentText())
		cmap_axes = str(self.report_cmap_axes.currentText())
		cmap_compare = str(self.report_cmap_compare.currentText())
		cmap_panel_axes = str(self.report_cmap_panel_axes.currentText())
		cmap_phase_report = str(self.report_cmap_phase.currentText())
		cmap_polar_clinico = str(self.report_cmap_polar_clinico.currentText())
		cmap_amp_report = str(self.report_cmap_amp.currentText())
		cmap_bullseye = str(self.report_cmap_bullseye.currentText())
		cmap_polar_perf = str(self.report_cmap_polar_perf.currentText())
		current_path_for_label = str(getattr(self, "_output_study_path_override", "") or self.file_edit.text().strip())
		study_context_label = self._study_context_label(path_override=current_path_for_label, study_obj=self.study)

		def _digest_mapping(data: dict | None) -> str:
			if not data:
				return "none"
			try:
				items = sorted((int(k), float(v)) for k, v in data.items())
			except Exception:
				items = sorted((str(k), str(v)) for k, v in data.items())
			return self._hash_payload({"items": items})

		compare_payload = {
			"active": bool(self.compare_bundle is not None),
			"path": str(self.compare_bundle.get("path", "")) if self.compare_bundle is not None else "",
			"primary_phase_seg": _digest_mapping(self.phase_by_seg),
			"compare_phase_seg": _digest_mapping(self.compare_bundle.get("phase_by_seg")) if self.compare_bundle is not None else "none",
		}
		base_payload = {
			"study": self._cache_study_sig,
			"seg": self._cache_seg_sig,
			"phase": self._cache_phase_sig,
			"compare": compare_payload,
		}
		tab_payloads = {
			"comparacion_ejes": {
				**base_payload,
				"cmap_compare": cmap_compare,
				"gate": int(self.compare_gate_spin.value()),
				"offset_sa": int(self.compare_slice_offset_sa_spin.value()),
				"offset_hla": int(self.compare_slice_offset_hla_spin.value()),
				"offset_vla": int(self.compare_slice_offset_vla_spin.value()),
				"window_lo": int(self.compare_window_low_slider.value()),
				"window_hi": int(self.compare_window_high_slider.value()),
			},
			"curva_fevi": {
				**base_payload,
				"visual_style": str(self.visual_style_combo.currentText()),
			},
			"panel_funcional_gated": {
				**base_payload,
				"visual_style": str(self.visual_style_combo.currentText()),
				"cmap_panel_axes": cmap_panel_axes,
				"cmap_phase": cmap_phase_report,
				"cmap_amp": cmap_amp_report,
			},
			"bullseye_directo": {
				**base_payload,
				"visual_style": str(self.visual_style_combo.currentText()),
				"cmap_bullseye": cmap_bullseye,
			},
			"guia_fase_vi": {
				**base_payload,
				"visual_style": str(self.visual_style_combo.currentText()),
				"cmap_bullseye": cmap_bullseye,
				"perfusion_source": str(self.perfusion_source()),
			},
			"polar_perfusion_directa": {
				**base_payload,
				"rotation": int(self.polar_rotation_spin.value()),
				"smooth_method": str(self.polar_perf_smooth_method_combo.currentText()),
				"smooth_strength": float(self.polar_perf_smooth_strength_spin.value()),
				"cmap_polar_perf": cmap_polar_perf,
				"cine_speed": int(self.polar_cine_speed_spin.value()),
				"export_mp4": bool(self.export_polar_mp4_check.isChecked()),
				"math_op": str(self.polar_compare_math_combo.currentText()),
				"math_a": str(self.polar_compare_term_a_combo.currentText()),
				"math_b": str(self.polar_compare_term_b_combo.currentText()),
			},
		}
		need_tab_render: dict[str, bool] = {}
		for tab_name, payload in tab_payloads.items():
			sig = self._hash_payload(payload)
			need_tab_render[tab_name] = self._cache_tab_output_sigs.get(tab_name) != sig
			self._cache_tab_output_sigs[tab_name] = sig
		need_tab_render["curva_fevi"] = False
		if target_tabs_set is not None:
			for tab_name in list(need_tab_render.keys()):
				need_tab_render[tab_name] = bool(tab_name in target_tabs_set)
		render_compare_axes = bool(need_tab_render.get("comparacion_ejes", True))
		render_curva_fevi = bool(need_tab_render.get("curva_fevi", True))
		render_panel_funcional = bool(need_tab_render.get("panel_funcional_gated", True))
		render_slices = target_tabs_set is None or "slices_fase" in target_tabs_set
		render_polar_combo = target_tabs_set is None or "polar_combo" in target_tabs_set
		render_delta_combo = target_tabs_set is None or "delta_combo" in target_tabs_set
		render_histograma = target_tabs_set is None or "histograma" in target_tabs_set

		# Opción A: la corrida completa (target_tabs_set is None) es rápida y omite
		# las pesadas; el render por-pestaña (target_tabs) sí las genera al entrar.
		advanced_mode = bool(self.advanced_mode_enabled)
		if not advanced_mode and target_tabs_set is None:
			for heavy_tab in (
				"comparacion_ejes",
				"curva_fevi",
				"panel_funcional_gated",
				"bullseye_directo",
				"guia_fase_vi",
				"polar_perfusion_directa",
			):
				need_tab_render[heavy_tab] = False
			for fname in (
				"ejes_ortogonales.png",
				"curva_tac.png",
				"comparacion_ejes.png",
				"curva_fevi.png",
				"ventriculograma.png",
				"panel_funcional_gated.png",
				"bullseye_directo.png",
				"guia_fase_vi.png",
				"polar_perfusion_directa.png",
				"polar_cine_montaje.png",
				"polar_cine.gif",
				"polar_cine.mp4",
			):
				fpath = os.path.join(self.output_dir, fname)
				if os.path.exists(fpath):
					try:
						os.remove(fpath)
					except OSError:
						pass
			self.compare_axes_preview_frames = []
			self.compare_axes_preview_index = 0
			self.compare_axes_playing = False
			self.compare_axes_cine_timer.stop()
			self.polar_cine_preview_frames = []
			self.polar_cine_preview_index = 0
			self.polar_cine_playing = False
			self.polar_cine_timer.stop()
			self._log("Modo rápido asincronía: se omiten comparacion_ejes, curva_fevi y polar perfusión/cine.")

		fig, axes = plt.subplots(1, 3, figsize=(15, 5))
		for ax in axes:
			ax.set_xticks([])
			ax.set_yticks([])

		axes[0].imshow(frame_norm, cmap=cmap_slices)
		axes[0].set_title(f"Slice {mid_slice}, Gate {mid_gate}")

		axes[1].imshow(frame_norm, cmap=cmap_slices)
		mask_slice = self.seg.mask[mid_slice].astype(float)
		overlay = np.zeros((*mask_slice.shape, 4))
		overlay[..., 0] = 1.0
		overlay[..., 3] = mask_slice * 0.45
		axes[1].imshow(overlay)
		axes[1].set_title("Máscara miocardio")

		axes[2].imshow(frame_norm, cmap=cmap_slices)
		phase_slice = self.phase_result.phase_map[mid_slice].copy()
		valid = np.isfinite(phase_slice)
		if valid.any():
			from viz.colormaps import phase_to_rgb
			rgb = phase_to_rgb(phase_slice[valid], cmap_name=cmap_phase_report)
			pm_overlay = np.zeros((*phase_slice.shape, 4))
			pm_overlay[valid, :3] = rgb
			pm_overlay[valid, 3] = 0.75
			axes[2].imshow(pm_overlay)
		axes[2].set_title("Fase superpuesta")

		fig.suptitle(f"SINCRO — Vista principal — {study_context_label}", fontsize=12.5, fontweight="bold")
		self._stamp_export_figure(fig, active_cine_widget)
		fig.tight_layout()
		fig.savefig(os.path.join(self.output_dir, "slices_fase.png"), dpi=150, bbox_inches="tight")
		plt.close(fig)

		pm = build_polar_map(self.phase_by_seg, cmap_name=cmap_phase_report, title=f"Phase Polar Map — {study_context_label}")
		pm.fig.text(
			0.02,
			0.02,
			"Qué muestra: distribución regional de fase (AHA 17). Uso clínico: identificar patrón y extensión de disincronía intraventricular.",
			fontsize=8.8,
			color="#334155",
			ha="left",
			va="bottom",
		)
		self._stamp_export_figure(pm.fig, active_cine_widget)
		save_polar_map(pm, os.path.join(self.output_dir, "polar_map.png"), dpi=150)
		plt.close(pm.fig)

		if render_delta_combo and self.compare_bundle is not None and self.compare_bundle.get("phase_by_seg") and self.study is not self.compare_bundle.get("study"):
			from matplotlib.cm import ScalarMappable
			from matplotlib.colors import Normalize
			from matplotlib.patches import Circle, Wedge

			def _circular_delta_deg(current_deg: float, reference_deg: float) -> float:
				return float(((float(current_deg) - float(reference_deg) + 180.0) % 360.0) - 180.0)

			def _render_numeric_polar_map(values_by_seg: dict[int, float], *, cmap_name: str, title: str, vmin: float, vmax: float, output_name: str, tick_labels: list[str], legend_text: str = ""):
				fig, ax = plt.subplots(figsize=(7.5, 7.0))
				ax.set_aspect("equal")
				ax.axis("off")

				rings = [
					(0.75, 1.00, 60.0, SECTOR_TO_SEGMENT_BASAL),
					(0.50, 0.75, 60.0, SECTOR_TO_SEGMENT_MEDIO),
					(0.25, 0.50, 90.0, SECTOR_TO_SEGMENT_APICAL),
				]
				cmap = matplotlib.colormaps.get_cmap(cmap_name)
				norm = Normalize(vmin=float(vmin), vmax=float(vmax))

				for r_in, r_out, step, lut in rings:
					n = len(lut)
					for k in range(n):
						t1 = float(k * step)
						t2 = float((k + 1) * step)
						seg_id = int(lut[k])
						val = float(values_by_seg.get(seg_id, np.nan))
						color = (0.35, 0.35, 0.35)
						if np.isfinite(val):
							color = cmap(norm(val))
						wedge = Wedge(
							(0.0, 0.0),
							r_out,
							t1,
							t2,
							width=(r_out - r_in),
							facecolor=color,
							edgecolor="white",
							linewidth=1.2,
						)
						ax.add_patch(wedge)
						x = (r_in + r_out) / 2.0 * np.cos(np.deg2rad((t1 + t2) / 2.0))
						y = (r_in + r_out) / 2.0 * np.sin(np.deg2rad((t1 + t2) / 2.0))
						if np.isfinite(val):
							ax.text(x, y, f"{seg_id}\n{val:+.0f}" if vmin < 0 else f"{seg_id}\n{val:.0f}", ha="center", va="center", fontsize=8, color="black")
						else:
							ax.text(x, y, f"{seg_id}\n--", ha="center", va="center", fontsize=8, color="black")

				apex_val = float(values_by_seg.get(17, np.nan))
				apex_color = (0.35, 0.35, 0.35)
				if np.isfinite(apex_val):
					apex_color = cmap(norm(apex_val))
				apex = Circle((0.0, 0.0), 0.25, facecolor=apex_color, edgecolor="white", linewidth=1.2)
				ax.add_patch(apex)
				if np.isfinite(apex_val):
					ax.text(0.0, 0.0, f"17\n{apex_val:+.0f}" if vmin < 0 else f"17\n{apex_val:.0f}", ha="center", va="center", fontsize=8, color="black")
				else:
					ax.text(0.0, 0.0, "17\n--", ha="center", va="center", fontsize=8, color="black")

				ax.set_xlim(-1.15, 1.35)
				ax.set_ylim(-1.15, 1.15)
				ax.set_title(title)
				sm = ScalarMappable(norm=norm, cmap=cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.04)
				cbar.set_label("Δ fase (°)")
				if tick_labels:
					ticks = np.linspace(float(vmin), float(vmax), num=len(tick_labels))
					cbar.set_ticks(ticks)
					cbar.set_ticklabels(tick_labels)
				if legend_text:
					fig.text(0.02, 0.02, legend_text, fontsize=9, color="#334155", ha="left", va="bottom")
				self._stamp_export_figure(fig, active_cine_widget)
				fig.tight_layout()
				fig.savefig(os.path.join(self.output_dir, output_name), dpi=150, bbox_inches="tight")
				plt.close(fig)

			compare_phase_by_seg = self.compare_bundle["phase_by_seg"]
			delta_signed: dict[int, float] = {}
			delta_abs: dict[int, float] = {}
			for seg_id in sorted(set(self.phase_by_seg.keys()) | set(compare_phase_by_seg.keys())):
				cur_val = self.phase_by_seg.get(int(seg_id), np.nan)
				ref_val = compare_phase_by_seg.get(int(seg_id), np.nan)
				if np.isfinite(cur_val) and np.isfinite(ref_val):
					delta = _circular_delta_deg(cur_val, ref_val)
					delta_signed[int(seg_id)] = delta
					delta_abs[int(seg_id)] = abs(delta)
				else:
					delta_signed[int(seg_id)] = np.nan
					delta_abs[int(seg_id)] = np.nan

			_render_numeric_polar_map(
				delta_signed,
				cmap_name="french",
				title="Delta polar map: esfuerzo - reposo (circular signed)",
				vmin=-180.0,
				vmax=180.0,
				output_name="polar_map_delta_signed.png",
				tick_labels=["-180", "-120", "-60", "0", "60", "120", "180"],
				legend_text="Qué es: Δsigned = esfuerzo - reposo (circular). Uso: dirección del cambio; negativo=atraso relativo, positivo=adelanto relativo.",
			)
			_render_numeric_polar_map(
				delta_abs,
				cmap_name="hot",
				title="Delta polar map: |esfuerzo - reposo|",
				vmin=0.0,
				vmax=180.0,
				output_name="polar_map_absdiff.png",
				tick_labels=["0", "30", "60", "90", "120", "150", "180"],
				legend_text="Qué es: |esfuerzo - reposo|. Uso: magnitud regional del cambio sin dirección (hotspots dinámicos).",
			)

		from PIL import Image

		def _compose_polar_combo():
			paths = [
				os.path.join(self.output_dir, "polar_map.png"),
				os.path.join(self.output_dir, "polar_clinico.png"),
			]
			images = [Image.open(p).convert("RGB") for p in paths if os.path.exists(p)]
			if not images:
				return
			max_w = max(im.width for im in images)
			pad = 18
			bg = (8, 12, 18)
			headers = ["Polar map", "Polar clínico"]
			header_h = 34
			prepared: list[Image.Image] = []
			for im in images:
				if im.width != max_w:
					scale = max_w / float(im.width)
					im = im.resize((max_w, max(1, int(round(im.height * scale)))))
				prepared.append(im)
			total_h = pad + sum(im.height + header_h + pad for im in prepared)
			canvas = Image.new("RGB", (max_w + pad * 2, total_h), color=bg)
			y = pad
			from PIL import ImageDraw
			draw = ImageDraw.Draw(canvas)
			for idx, im in enumerate(prepared):
				draw.rounded_rectangle([pad, y, pad + max_w, y + header_h - 4], radius=6, fill=(25, 35, 50), outline=(75, 105, 140), width=1)
				draw.text((pad + 10, y + 7), headers[idx], fill=(235, 242, 255))
				y += header_h
				canvas.paste(im, (pad, y))
				y += im.height + pad
			canvas.save(os.path.join(self.output_dir, "polar_combo.png"))

		def _compose_delta_combo():
			paths = [
				os.path.join(self.output_dir, "polar_map_delta_signed.png"),
				os.path.join(self.output_dir, "polar_map_absdiff.png"),
			]
			images = [Image.open(p).convert("RGB") for p in paths if os.path.exists(p)]
			if len(images) < 2:
				return
			max_h = max(im.height for im in images)
			pad = 18
			bg = (8, 12, 18)
			headers = ["Δsigned", "Δabs"]
			header_h = 34
			prepared: list[Image.Image] = []
			for im in images:
				if im.height != max_h:
					scale = max_h / float(im.height)
					im = im.resize((max(1, int(round(im.width * scale))), max_h))
				prepared.append(im)
			total_w = pad + sum(im.width + pad for im in prepared)
			canvas = Image.new("RGB", (total_w, max_h + header_h + pad * 2), color=bg)
			from PIL import ImageDraw
			draw = ImageDraw.Draw(canvas)
			x = pad
			for idx, im in enumerate(prepared):
				draw.rounded_rectangle([x, pad, x + im.width, pad + header_h - 4], radius=6, fill=(25, 35, 50), outline=(75, 105, 140), width=1)
				draw.text((x + 10, pad + 7), headers[idx], fill=(235, 242, 255))
				canvas.paste(im, (x, pad + header_h))
				x += im.width + pad
			canvas.save(os.path.join(self.output_dir, "delta_combo.png"))

		if render_histograma:
			if self.phase_result_raw is not None and self.metrics_raw is not None:
				hfig = build_phase_histogram(
					self.phase_result_raw.phases_deg,
					metrics=self.metrics_raw,
					bins=72,
					title=f"Phase Histogram QC — {study_context_label}",
					comparison_phases_deg=self.phase_result.phases_deg,
					comparison_metrics=self.metrics,
					primary_label=f"Crudo amp {RAW_PHASE_QC_AMP_FILTER:.2f}",
					comparison_label=f"Clínico amp {float(self.metrics.get('amp_filter', self.phase_threshold_spin.value())):.2f}",
					qc_note=self._phase_qc_note(),
				)
			else:
				hfig = build_phase_histogram(self.phase_result.phases_deg, metrics=self.metrics, bins=72, title=f"Phase Histogram — {study_context_label}")
			self._stamp_export_figure(hfig, active_cine_widget)
			save_histogram(hfig, os.path.join(self.output_dir, "histograma.png"), dpi=150)
			plt.close(hfig)
		cfig = build_clinical_phase_panel(
			self.phase_by_seg,
			self.phase_result.phases_deg,
			metrics=self.metrics,
			cmap_name=cmap_polar_clinico,
			title=f"Panel polar clínico (histograma + fase) — {study_context_label}",
		)
		self._stamp_export_figure(cfig, active_cine_widget)
		save_clinical_phase_panel(cfig, os.path.join(self.output_dir, "polar_clinico.png"), dpi=150)
		plt.close(cfig)
		_compose_polar_combo()
		_compose_delta_combo()

		# Opción A: en la corrida completa se corta acá (rápido); el render por-pestaña
		# (target_tabs) continúa para generar la pesada solicitada bajo demanda.
		if not advanced_mode and target_tabs_set is None:
			self._log("Modo básico: se omite render avanzado (ejes, panel funcional, perfusión directa y cine polar).")
			return

		def _oriented_axes_views(gate_index: int):
			from core.cardiac_reorientation import hla_slice as _hla_slice, vla_slice as _vla_slice
			vol_gate = np.asarray(study_cube_render[int(gate_index)], dtype=np.float64)
			sa_local = vol_gate[mid_slice]

			def _axis_plane(axis_code: str, prefer_original: bool):
				if prefer_original and self.axis_companions.get(axis_code) is not None:
					axis_study = self.axis_companions[axis_code]
					return np.asarray(
						axis_study.cube[
							int(gate_index),
							min(int(axis_study.cube.shape[1] // 2), int(axis_study.cube.shape[1] - 1)),
						],
						dtype=np.float64,
					), True
				if axis_code == "HLA":
					return vol_gate[:, vol_gate.shape[1] // 2, :], False
				return vol_gate[:, :, vol_gate.shape[2] // 2], False

			hla_local, hla_original = _axis_plane("HLA", prefer_original=True)
			vla_local, vla_original = _axis_plane("VLA", prefer_original=True)

			# Si no hay companions HLA/VLA originales, derivar desde SA con la
			# convención anatómica canónica del motor (cardiac_reorientation).
			if not hla_original:
				hla_local = _hla_slice(vol_gate, int(vol_gate.shape[1] // 2))
			if not vla_original:
				vla_local = _vla_slice(vol_gate, int(vol_gate.shape[2] // 2))

			hla_view_local = _norm(hla_local)
			vla_view_local = _norm(vla_local)
			return _norm(sa_local), hla_view_local, vla_view_local, hla_original, vla_original

		def _annotate_axis(ax, top: str, bottom: str, left: str, right: str):
			label_style = dict(
				transform=ax.transAxes,
				fontsize=8,
				fontweight="bold",
				color="#d7f0ff",
				bbox=dict(boxstyle="round,pad=0.18", facecolor="black", edgecolor="#8ad0ff", alpha=0.65),
			)
			ax.text(0.50, 0.98, top, ha="center", va="top", **label_style)
			ax.text(0.50, 0.02, bottom, ha="center", va="bottom", **label_style)
			ax.text(0.02, 0.50, left, ha="left", va="center", rotation=90, **label_style)
			ax.text(0.98, 0.50, right, ha="right", va="center", rotation=270, **label_style)

		def _norm(img):
			arr = np.asarray(img, dtype=np.float64)
			mx = float(np.nanmax(arr)) if arr.size else 0.0
			return arr / (mx + 1e-8)

		try:
			from scipy.ndimage import zoom as ndi_zoom
		except Exception:
			ndi_zoom = None

		def _windowed_panel(img2d: np.ndarray) -> np.ndarray:
			lo = float(self.compare_window_low_slider.value()) / 100.0
			hi = float(self.compare_window_high_slider.value()) / 100.0
			hi = max(hi, lo + 0.01)
			arr = _norm(img2d)
			return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

		def _zoom_center_panel(img2d: np.ndarray, *, zoom_factor: float) -> np.ndarray:
			arr = np.asarray(img2d, dtype=np.float64)
			if arr.ndim != 2 or zoom_factor <= 1.001:
				return arr
			h, w = arr.shape
			crop_h = max(8, int(round(h / zoom_factor)))
			crop_w = max(8, int(round(w / zoom_factor)))
			y0 = max(0, (h - crop_h) // 2)
			x0 = max(0, (w - crop_w) // 2)
			crop = arr[y0:y0 + crop_h, x0:x0 + crop_w]
			if crop.size == 0:
				return arr
			if ndi_zoom is not None:
				scale_y = float(h) / float(max(1, crop_h))
				scale_x = float(w) / float(max(1, crop_w))
				up = ndi_zoom(crop, (scale_y, scale_x), order=3, mode="nearest", prefilter=True)
			else:
				zy = max(1, int(np.ceil(h / max(1, crop_h))))
				zx = max(1, int(np.ceil(w / max(1, crop_w))))
				up = np.repeat(np.repeat(crop, zy, axis=0), zx, axis=1)
			if up.shape[0] < h or up.shape[1] < w:
				pad_y = max(0, h - up.shape[0])
				pad_x = max(0, w - up.shape[1])
				up = np.pad(up, ((0, pad_y), (0, pad_x)), mode="edge")
			return up[:h, :w]

		sa, hla_view, vla_view, hla_original_mid, vla_original_mid = _oriented_axes_views(mid_gate)

		if target_tabs_set is None or "comparacion_ejes" in target_tabs_set:
			fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.8))
			for ax in axes2:
				ax.set_xticks([])
				ax.set_yticks([])

			axes2[0].imshow(sa, cmap=cmap_axes)
			axes2[0].set_title(f"SA (slice {mid_slice + 1})")
			_annotate_axis(axes2[0], "ANT", "INF", "SEP", "LAT")
			cmp_frac_sa = self._comparison_fraction()
			sa_h = int(sa.shape[0])
			sa_w = int(sa.shape[1])
			y_cmp = min(max(0, int(round(cmp_frac_sa * max(0, sa_h - 1)))), sa_h - 1)
			x_cmp = min(max(0, int(round(cmp_frac_sa * max(0, sa_w - 1)))), sa_w - 1)
			axes2[0].axhline(y_cmp, color="#7cf29a", linestyle="--", linewidth=1.2)
			axes2[0].axvline(x_cmp, color="#7cf29a", linestyle="--", linewidth=1.2)
			axes2[0].text(0.03, 0.05, f"Corte cmp {int(round(cmp_frac_sa * 100.0))}%", transform=axes2[0].transAxes, fontsize=8, color="#7cf29a", fontweight="bold")
			axes2[1].imshow(hla_view, cmap=cmap_axes, aspect="equal")
			axes2[1].set_title("HLA (horizontal long axis)")
			_annotate_axis(axes2[1], "BASE", "APEX", "ANT", "INF")
			axes2[2].imshow(vla_view, cmap=cmap_axes, aspect="equal")
			axes2[2].set_title("VLA (vertical long axis)")
			_annotate_axis(axes2[2], "BASE", "APEX", "SEP", "LAT")
			if hla_original_mid:
				axes2[1].text(0.03, 0.05, "ORIGINAL", transform=axes2[1].transAxes, fontsize=8, color="#ffe082", fontweight="bold")
			if vla_original_mid:
				axes2[2].text(0.03, 0.05, "ORIGINAL", transform=axes2[2].transAxes, fontsize=8, color="#ffe082", fontweight="bold")
			fig2.suptitle(f"Ejes cardíacos ortogonales — Gate {mid_gate + 1} — {study_context_label}", fontsize=12, fontweight="bold")
			self._stamp_export_figure(fig2, active_cine_widget)
			fig2.tight_layout()
			fig2.savefig(os.path.join(self.output_dir, "ejes_ortogonales.png"), dpi=150, bbox_inches="tight")
			plt.close(fig2)

		if render_compare_axes:
			# Grilla comparacion_ejes (16 cortes/eje) DEPRECADA: reemplazada por el
			# Montaje clínico. Ya no se regenera aquí (evita render lento y mal orientado).
			pass

		ef = self._estimate_lv_ef()
		n_gates = int(study_cube_render.shape[0])
		if ef.get("available"):
			ed_gate = max(0, min(n_gates - 1, int(ef["ed_gate"]) - 1))
			es_gate = max(0, min(n_gates - 1, int(ef["es_gate"]) - 1))
		else:
			ed_gate = mid_gate
			es_gate = (mid_gate + max(1, n_gates // 2)) % n_gates

		style_name = str(self.visual_style_combo.currentText()).strip().lower()
		style = self._visual_style_dict()

		panel_zoom = max(1.0, float(self.compare_axes_zoom_slider.value()) / 100.0)
		sa_ed_raw, hla_ed_raw, vla_ed_raw, _ed_hla_original, _ed_vla_original = _oriented_axes_views(ed_gate)
		sa_es_raw, hla_es_raw, vla_es_raw, _es_hla_original, _es_vla_original = _oriented_axes_views(es_gate)
		sa_ed = _zoom_center_panel(_windowed_panel(sa_ed_raw), zoom_factor=panel_zoom)
		hla_ed = _zoom_center_panel(_windowed_panel(hla_ed_raw), zoom_factor=panel_zoom)
		vla_ed = _zoom_center_panel(_windowed_panel(vla_ed_raw), zoom_factor=panel_zoom)
		sa_es = _zoom_center_panel(_windowed_panel(sa_es_raw), zoom_factor=panel_zoom)
		hla_es = _zoom_center_panel(_windowed_panel(hla_es_raw), zoom_factor=panel_zoom)
		vla_es = _zoom_center_panel(_windowed_panel(vla_es_raw), zoom_factor=panel_zoom)

		fig4, axes4 = plt.subplots(2, 3, figsize=(14, 8.2))
		for ax in axes4.ravel():
			ax.set_xticks([])
			ax.set_yticks([])

		axes4[0, 0].imshow(sa_ed, cmap=cmap_panel_axes, interpolation="bicubic")
		axes4[0, 0].set_title(f"A) ED - SHORT AXIS (Gate {ed_gate + 1})", fontsize=10)
		_annotate_axis(axes4[0, 0], "ANT", "INF", "SEP", "LAT")
		axes4[0, 1].imshow(hla_ed, cmap=cmap_panel_axes, aspect="equal", interpolation="bicubic")
		axes4[0, 1].set_title("A) ED - HORIZONTAL AXIS (HLA)", fontsize=10)
		_annotate_axis(axes4[0, 1], "BASE", "APEX", "ANT", "INF")
		axes4[0, 2].imshow(vla_ed, cmap=cmap_panel_axes, aspect="equal", interpolation="bicubic")
		axes4[0, 2].set_title("A) ED - VERTICAL AXIS (VLA)", fontsize=10)
		_annotate_axis(axes4[0, 2], "BASE", "APEX", "SEP", "LAT")

		axes4[1, 0].imshow(sa_es, cmap=cmap_panel_axes, interpolation="bicubic")
		axes4[1, 0].set_title(f"B) ES - SHORT AXIS (Gate {es_gate + 1})", fontsize=10)
		_annotate_axis(axes4[1, 0], "ANT", "INF", "SEP", "LAT")
		axes4[1, 1].imshow(hla_es, cmap=cmap_panel_axes, aspect="equal", interpolation="bicubic")
		axes4[1, 1].set_title("B) ES - HORIZONTAL AXIS (HLA)", fontsize=10)
		_annotate_axis(axes4[1, 1], "BASE", "APEX", "ANT", "INF")
		axes4[1, 2].imshow(vla_es, cmap=cmap_panel_axes, aspect="equal", interpolation="bicubic")
		axes4[1, 2].set_title("B) ES - VERTICAL AXIS (VLA)", fontsize=10)
		_annotate_axis(axes4[1, 2], "BASE", "APEX", "SEP", "LAT")

		fig4.suptitle(
			f"Panel clínico por convención (A=diástole, B=sístole) — SA/HLA/VLA — {study_context_label}",
			fontsize=13,
			fontweight="bold",
		)
		self._stamp_export_figure(fig4, active_cine_widget)
		fig4.tight_layout()
		fig4.savefig(os.path.join(self.output_dir, "panel_clinico_convencion.png"), dpi=150, bbox_inches="tight")
		plt.close(fig4)

		if target_tabs_set is None or "curva_fevi" in target_tabs_set:
			fig3, ax3 = plt.subplots(figsize=(10, 4))
			mean_per_gate = np.array([float(study_cube_render[gi][self.seg.mask].mean()) for gi in range(study_cube_render.shape[0])])
			mean_per_gate = mean_per_gate / (mean_per_gate.max() + 1e-8)
			ax3.plot(np.arange(study_cube_render.shape[0]), mean_per_gate, "o-", color="#2c7fb8")
			ax3.set_title("Curva de actividad miocárdica por gate")
			ax3.set_xlabel("Gate")
			ax3.set_ylabel("Intensidad normalizada")
			ax3.grid(True, alpha=0.3)
			self._stamp_export_figure(fig3, active_cine_widget)
			fig3.tight_layout()
			fig3.savefig(os.path.join(self.output_dir, "curva_tac.png"), dpi=150, bbox_inches="tight")
			plt.close(fig3)

		if render_curva_fevi:
			fig_ef, ax_ef = plt.subplots(figsize=(10, 4.4), facecolor=style["fig_bg"])
			ax_ef.set_facecolor(style["ax_bg"])
			gate_axis = np.arange(study_cube_render.shape[0]) + 1
			t_pct = np.linspace(0.0, 100.0, study_cube_render.shape[0], endpoint=False)
			if ef.get("available"):
				gate_volumes = np.asarray(ef.get("gate_volumes_ml", []), dtype=np.float64)
				if gate_volumes.size != gate_axis.size:
					gate_volumes = np.full_like(gate_axis, np.nan, dtype=np.float64)
				dv_dt = np.gradient(gate_volumes)
				ax_ef.plot(gate_axis, gate_volumes, "o-", color=style["vol"], linewidth=2.0, markersize=4.5, label="Volumen VI (mL)")
				ax_ef_2 = ax_ef.twinx()
				ax_ef_2.plot(gate_axis, dv_dt, "-", color=style["deriv"], linewidth=1.7, label="dV/dgate")
				ax_ef_2.set_ylabel("dV/dgate", color=style["deriv"])
				ax_ef_2.tick_params(axis="y", colors=style["deriv"])
				ed_gate = int(ef.get("ed_gate", 1))
				es_gate = int(ef.get("es_gate", 1))
				ax_ef.axvline(ed_gate, color=style["ed"], linestyle="--", linewidth=1.2)
				ax_ef.axvline(es_gate, color=style["es"], linestyle="--", linewidth=1.2)
				ax_ef.text(ed_gate, float(np.nanmax(gate_volumes)) * 1.01, "ED", color=style["ed"], ha="center", va="bottom", fontsize=9, fontweight="bold")
				ax_ef.text(es_gate, float(np.nanmax(gate_volumes)) * 1.01, "ES", color=style["es"], ha="center", va="bottom", fontsize=9, fontweight="bold")
				ax_ef.set_title(
					f"Curva de volumen por gate — FEVI {float(ef.get('ef_pct', 0.0)):.1f}% "
					f"({self.fevi_method_label()})",
					color=style["fg"], fontsize=12, fontweight="bold",
				)
				ax_ef.set_ylabel("Volumen estimado (mL)", color=style["fg"])
				ax_ef.tick_params(axis="x", colors=style["subtle"])
				ax_ef.tick_params(axis="y", colors=style["vol"])
				ax_ef.grid(True, color=style["grid"], alpha=0.45)
				ax_ef.set_xlabel("Gate", color=style["subtle"])
				vol_max_ef = float(np.nanmax(gate_volumes)) if gate_volumes.size and np.isfinite(gate_volumes).any() else 1.0
				ax_ef.set_ylim(0.0, vol_max_ef * 1.15)
				ax_top = ax_ef.twiny()
				ax_top.set_xlim(ax_ef.get_xlim())
				ax_top.set_xticks(gate_axis)
				ax_top.set_xticklabels([f"{int(v)}" for v in t_pct], fontsize=8)
				ax_top.set_xlabel("% ciclo", color=style["subtle"])
				ax_top.tick_params(axis="x", colors=style["subtle"])
			else:
				ax_ef.plot([], [])
				ax_ef.text(
					0.5,
					0.5,
					"FEVI preliminar no disponible\n(segmentación/metadata insuficiente)",
					ha="center",
					va="center",
					transform=ax_ef.transAxes,
					fontsize=11,
					color=style["fg"],
				)
				ax_ef.set_title("Curva FEVI preliminar por gate", color=style["fg"], fontsize=12, fontweight="bold")
				ax_ef.set_xlabel("Gate", color=style["subtle"])
				ax_ef.set_ylabel("Volumen estimado (mL)", color=style["fg"])
				ax_ef.tick_params(axis="x", colors=style["subtle"])
				ax_ef.tick_params(axis="y", colors=style["fg"])
				ax_ef.grid(True, color=style["grid"], alpha=0.45)
			self._stamp_export_figure(fig_ef, active_cine_widget)
			fig_ef.tight_layout()
			fig_ef.savefig(os.path.join(self.output_dir, "curva_fevi.png"), dpi=160, bbox_inches="tight", facecolor=fig_ef.get_facecolor())
			plt.close(fig_ef)
		else:
			self._log("Cache tab: curva_fevi sin cambios, se omite regeneración.")

		# Panel funcional gated SPECT: ED/ES + mapas + curvas de volumen/fase.
		fig_v = plt.figure(figsize=(14.0, 8.4), facecolor=style["fig_bg"])
		gs = fig_v.add_gridspec(3, 4, width_ratios=[1.1, 1.1, 1.45, 1.15], hspace=0.28, wspace=0.22)
		ax_ed_sa = fig_v.add_subplot(gs[0, 0])
		ax_es_sa = fig_v.add_subplot(gs[1, 0])
		ax_ed_hla = fig_v.add_subplot(gs[0, 1])
		ax_es_hla = fig_v.add_subplot(gs[1, 1])
		ax_phase = fig_v.add_subplot(gs[2, 0])
		ax_amp = fig_v.add_subplot(gs[2, 1])
		ax_results = fig_v.add_subplot(gs[0, 2:4])
		ax_curve = fig_v.add_subplot(gs[1, 2:4])
		ax_metrics = fig_v.add_subplot(gs[2, 2:4])

		for ax in [ax_ed_sa, ax_es_sa, ax_ed_hla, ax_es_hla, ax_phase, ax_amp, ax_results, ax_curve, ax_metrics]:
			ax.set_facecolor(style["ax_bg"])
			for spine in ax.spines.values():
				spine.set_color(style["grid"])

		for ax in [ax_ed_sa, ax_es_sa, ax_ed_hla, ax_es_hla, ax_phase, ax_amp]:
			ax.set_xticks([])
			ax.set_yticks([])

		ax_ed_sa.imshow(sa_ed, cmap=cmap_panel_axes, interpolation="lanczos", resample=True)
		ax_ed_sa.set_title(f"ED SA (gate {ed_gate + 1})", color=style["fg"], fontsize=9)
		ax_es_sa.imshow(sa_es, cmap=cmap_panel_axes, interpolation="lanczos", resample=True)
		ax_es_sa.set_title(f"ES SA (gate {es_gate + 1})", color=style["fg"], fontsize=9)
		ax_ed_hla.imshow(hla_ed, cmap=cmap_panel_axes, aspect="equal", interpolation="lanczos", resample=True)
		ax_ed_hla.set_title("ED HLA", color=style["fg"], fontsize=9)
		ax_es_hla.imshow(hla_es, cmap=cmap_panel_axes, aspect="equal", interpolation="lanczos", resample=True)
		ax_es_hla.set_title("ES HLA", color=style["fg"], fontsize=9)

		from viz.colormaps import phase_to_rgb

		phase_mid = np.asarray(self.phase_result.phase_map[mid_slice], dtype=np.float64)
		amp_mid = np.asarray(self.phase_result.amplitude_map[mid_slice], dtype=np.float64)
		amp_show = amp_mid / (float(np.nanmax(amp_mid)) + 1e-8)
		phase_rgb = phase_to_rgb(phase_mid, cmap_name=cmap_phase_report, nan_color=(0.05, 0.07, 0.10))
		ax_phase.imshow(phase_rgb)
		ax_phase.set_title("Mapa de fase", color=style["fg"], fontsize=9)
		ax_amp.imshow(amp_show, cmap=cmap_amp_report, vmin=0.0, vmax=1.0)
		ax_amp.set_title("Mapa de amplitud", color=style["fg"], fontsize=9)

		t_gate = np.arange(1, n_gates + 1)
		if ef.get("available"):
			v = np.asarray(ef.get("gate_volumes_ml", []), dtype=np.float64)
			if v.size != t_gate.size:
				v = np.full_like(t_gate, np.nan, dtype=np.float64)
			dv = np.gradient(v)
			ax_curve.plot(t_gate, v, color=style["vol"], linewidth=2.2, marker="o", markersize=4, label="Volumen")
			ax_curve_2 = ax_curve.twinx()
			ax_curve_2.plot(t_gate, dv, color=style["deriv"], linewidth=1.8, label="dV/dgate")
			ax_curve_2.tick_params(axis="y", colors=style["deriv"])
			ax_curve_2.set_ylabel("dV/dgate", color=style["deriv"])
			ax_curve.axvline(ed_gate + 1, color=style["ed"], linestyle="--", linewidth=1.2)
			ax_curve.axvline(es_gate + 1, color=style["es"], linestyle="--", linewidth=1.2)
			vol_max_curve = float(np.nanmax(v)) if v.size and np.isfinite(v).any() else 1.0
			ax_curve.set_ylim(0.0, vol_max_curve * 1.15)
		else:
			ax_curve.plot([], [])
			ax_curve.text(0.5, 0.5, "Sin FEVI preliminar", transform=ax_curve.transAxes, ha="center", va="center", color=style["fg"])

		ax_curve.set_title("Time/Volume y derivada", color=style["fg"], fontsize=10, fontweight="bold")
		ax_curve.set_xlabel("Gate", color=style["subtle"])
		ax_curve.set_ylabel("Volumen (mL)", color=style["vol"])
		ax_curve.tick_params(axis="x", colors=style["subtle"])
		ax_curve.tick_params(axis="y", colors=style["vol"])
		ax_curve.grid(True, color=style["grid"], alpha=0.45)

		phase_seg_ids = np.array(sorted(int(k) for k in self.phase_by_seg.keys()), dtype=np.int32)
		phase_seg_vals = np.array([float(self.phase_by_seg[int(k)]) for k in phase_seg_ids], dtype=np.float64) if phase_seg_ids.size else np.array([], dtype=np.float64)
		if phase_seg_ids.size:
			ax_metrics.plot(phase_seg_ids, phase_seg_vals, color=style["deriv"], linewidth=1.8, marker="o", markersize=4)
			ax_metrics.axhline(float(self.metrics.get("mean_phase", np.nan)), color=style["ed"], linestyle="--", linewidth=1.1)
			ax_metrics.set_xlim(1, 17)
			ax_metrics.set_xticks(np.arange(1, 18, 2))
			ax_metrics.set_ylim(0, 360)
			ax_metrics.set_yticks(np.arange(0, 361, 90))
			ax_metrics.set_title("Curva de fase por segmento AHA", color=style["fg"], fontsize=10, fontweight="bold")
			ax_metrics.set_xlabel("Segmento AHA", color=style["subtle"])
			ax_metrics.set_ylabel("Fase (°)", color=style["deriv"])
			ax_metrics.tick_params(axis="x", colors=style["subtle"])
			ax_metrics.tick_params(axis="y", colors=style["deriv"])
			ax_metrics.grid(True, color=style["grid"], alpha=0.35)
		else:
			ax_metrics.set_xticks([])
			ax_metrics.set_yticks([])
			ax_metrics.text(0.5, 0.5, "Sin datos de fase por segmento", transform=ax_metrics.transAxes, ha="center", va="center", color=style["fg"])

		metrics_lines = [
			f"PSD técnico: {self.metrics.get('technical_classification', self.metrics.get('classification'))}",
			f"Phase SD: {float(self.metrics.get('phase_sd', np.nan)):.1f}°",
			f"Bandwidth: {float(self.metrics.get('bandwidth', np.nan)):.1f}°",
			f"Entropy: {float(self.metrics.get('entropy_normalized_pct', np.nan)):.1f}%",
		]
		if ef.get("available"):
			metrics_lines.extend([
				f"EDV: {float(ef.get('edv_ml', np.nan)):.1f} mL",
				f"ESV: {float(ef.get('esv_ml', np.nan)):.1f} mL",
				f"FEVI: {float(ef.get('ef_pct', np.nan)):.1f}%",
			])
		else:
			metrics_lines.append("FEVI: no disponible")
		ax_results.set_xticks([])
		ax_results.set_yticks([])
		ax_results.set_xlim(0.0, 1.0)
		ax_results.set_ylim(0.0, 1.0)
		ax_results.text(0.02, 0.92, "Resultados", transform=ax_results.transAxes, va="top", ha="left", color=style["fg"], fontsize=11.5, fontweight="bold")
		result_items = [
			("PSD técnico", f"{self.metrics.get('technical_classification', self.metrics.get('classification'))} (no dx)"),
			("Phase SD", f"{float(self.metrics.get('phase_sd', np.nan)):.1f}°"),
			("Bandwidth", f"{float(self.metrics.get('bandwidth', np.nan)):.1f}°"),
			("Entropy", f"{float(self.metrics.get('entropy_normalized_pct', np.nan)):.1f}%"),
		]
		if ef.get("available"):
			result_items.extend([
				("EDV", f"{float(ef.get('edv_ml', np.nan)):.1f} mL"),
				("ESV", f"{float(ef.get('esv_ml', np.nan)):.1f} mL"),
			])
			fevi_value_txt = f"{float(ef.get('ef_pct', np.nan)):.1f}%"
		else:
			fevi_value_txt = "no disponible"
		for idx, (label, value) in enumerate(result_items):
			col = idx % 2
			row = idx // 2
			x = 0.04 + col * 0.48
			y = 0.62 - row * 0.24
			ax_results.text(x, y + 0.11, label, transform=ax_results.transAxes, va="bottom", ha="left", color=style["subtle"], fontsize=7.8, fontweight="bold")
			ax_results.text(x, y, value, transform=ax_results.transAxes, va="bottom", ha="left", color=style["fg"], fontsize=11.5, fontweight="bold")

		# FEVI destacado (grande) en el panel derecho, para lectura rápida clínica.
		ax_results.text(
			0.82, 0.58, "FEVI",
			transform=ax_results.transAxes,
			ha="center", va="center",
			color=style["subtle"],
			fontsize=10.5, fontweight="bold",
		)
		ax_results.text(
			0.82, 0.45, fevi_value_txt,
			transform=ax_results.transAxes,
			ha="center", va="center",
			color=style["fg"],
			fontsize=24, fontweight="bold",
			bbox=dict(boxstyle="round,pad=0.28", facecolor=style["ax_bg"], edgecolor=style["grid"], linewidth=1.1, alpha=0.95),
		)

		fig_v.suptitle(
			f"Panel funcional gated — {study_context_label} (estilo clínico: {self.visual_style_combo.currentText()})",
			color=style["fg"],
			fontsize=13,
			fontweight="bold",
		)
		self._stamp_export_figure(fig_v, active_cine_widget)
		if render_panel_funcional:
			panel_path = os.path.join(self.output_dir, "panel_funcional_gated.png")
			fig_v.savefig(panel_path, dpi=155, bbox_inches="tight", facecolor=fig_v.get_facecolor())
			legacy_path = os.path.join(self.output_dir, "ventriculograma.png")
			try:
				if not os.path.exists(legacy_path):
					import shutil
					shutil.copyfile(panel_path, legacy_path)
			except OSError:
				pass
		else:
			self._log("Cache tab: panel_funcional_gated sin cambios, se omite regeneración.")
		plt.close(fig_v)

		# Bull's eye directo de perfusión (colores de intensidad), inspirado en consolas clínicas.
		from matplotlib.patches import Circle, Wedge
		seg_map = np.asarray(self.aha.segment_map, dtype=np.int32)
		mid_gate_cube = np.asarray(study_cube_render[mid_gate], dtype=np.float64)
		mx = float(np.nanmax(mid_gate_cube)) if mid_gate_cube.size else 0.0
		uptake_norm = mid_gate_cube / (mx + 1e-8)
		seg_uptake: dict[int, float] = {}
		for seg_id in range(1, 18):
			vals = uptake_norm[seg_map == seg_id]
			vals = vals[np.isfinite(vals)]
			if vals.size:
				seg_uptake[seg_id] = float(np.median(vals))
			else:
				seg_uptake[seg_id] = np.nan

		fig_b, ax_b = plt.subplots(figsize=(7.2, 7.2), facecolor=style["fig_bg"])
		ax_b.set_facecolor(style["fig_bg"])
		ax_b.set_xlim(-1.08, 1.08)
		ax_b.set_ylim(-1.08, 1.08)
		ax_b.set_aspect("equal")
		ax_b.axis("off")
		cmap_b = matplotlib.colormaps.get(cmap_bullseye)

		def _segment_color(seg_id: int):
			v = seg_uptake.get(int(seg_id), np.nan)
			if not np.isfinite(v):
				return (0.25, 0.25, 0.28, 1.0)
			v = float(np.clip(v, 0.0, 1.0))
			return cmap_b(v)

		def _draw_ring(seg_ids: list[int], r_inner: float, r_outer: float, start_deg: float = 90.0):
			n = len(seg_ids)
			for i, sid in enumerate(seg_ids):
				theta1 = start_deg - (i + 1) * (360.0 / n)
				theta2 = start_deg - i * (360.0 / n)
				wedge = Wedge((0.0, 0.0), r_outer, theta1, theta2, width=r_outer - r_inner, facecolor=_segment_color(sid), edgecolor=style["grid"], linewidth=1.4)
				ax_b.add_patch(wedge)
				mid_a = np.deg2rad((theta1 + theta2) * 0.5)
				r_t = (r_inner + r_outer) * 0.5
				ax_b.text(r_t * np.cos(mid_a), r_t * np.sin(mid_a), str(sid), color=style["fg"], fontsize=8, ha="center", va="center", fontweight="bold")

		_draw_ring([1, 2, 3, 4, 5, 6], 0.68, 0.98, start_deg=90.0)
		_draw_ring([7, 8, 9, 10, 11, 12], 0.40, 0.68, start_deg=90.0)
		_draw_ring([13, 14, 15, 16], 0.18, 0.40, start_deg=45.0)
		apex = Circle((0.0, 0.0), radius=0.18, facecolor=_segment_color(17), edgecolor=style["grid"], linewidth=1.4)
		ax_b.add_patch(apex)
		ax_b.text(0.0, 0.0, "17", color=style["fg"], fontsize=8, ha="center", va="center", fontweight="bold")

		ax_b.text(0.0, 1.04, f"Bull's eye perfusión directa ({self.visual_style_combo.currentText()})", ha="center", va="bottom", color=style["fg"], fontsize=12, fontweight="bold")
		ax_b.text(0.0, -1.02, "Colores de intensidad normalizada (gate medio)", ha="center", va="top", color=style["subtle"], fontsize=9)
		ax_b.text(0.0, -1.09, "Uso clínico: resumen segmentario AHA rápido para detectar regiones de hipocaptación.", ha="center", va="top", color=style["subtle"], fontsize=8.4)
		self._stamp_export_figure(fig_b, active_cine_widget)
		fig_b.savefig(os.path.join(self.output_dir, "bullseye_directo.png"), dpi=170, bbox_inches="tight", facecolor=fig_b.get_facecolor())
		plt.close(fig_b)

		# Guía para fase VI: bull's-eye doble (fase + perfusión/viabilidad) + tabla
		# segmentaria AHA-17, con reposo y esfuerzo en la misma imagen si hay
		# estudio de comparación. Diseño propio (estilo del panel funcional).
		if need_tab_render.get("guia_fase_vi", True):
			try:
				self._render_guia_fase_vi(style, study_cube_render, active_cine_widget, ef, cmap_bullseye)
			except Exception as exc:
				self._log(f"Guía para fase VI: no se pudo renderizar ({exc}).")

		# Mapa polar continuo de perfusión ("aplastado" apex->base), complementario al bull's eye por segmentos.
		from scipy.ndimage import gaussian_filter

		def _fill_profile_nans_circular(profile: np.ndarray) -> np.ndarray:
			p = np.asarray(profile, dtype=np.float64).copy()
			if p.size == 0:
				return p
			valid = np.isfinite(p)
			if valid.all():
				return p
			if not valid.any():
				return np.zeros_like(p, dtype=np.float64)
			x = np.arange(p.size)
			xv = x[valid]
			yv = p[valid]
			x_ext = np.concatenate([xv - p.size, xv, xv + p.size])
			y_ext = np.concatenate([yv, yv, yv])
			p[~valid] = np.interp(x[~valid], x_ext, y_ext)
			return p

		def _slice_angular_profile(s_idx: int) -> np.ndarray | None:
			img = np.asarray(mid_gate_cube[int(s_idx)], dtype=np.float64)
			mask_s = np.asarray(self.seg.mask[int(s_idx)], dtype=bool)
			if not np.any(mask_s):
				return None
			cy, cx = self.seg.center_per_slice[int(s_idx)]
			if not (np.isfinite(cy) and np.isfinite(cx)):
				ys0, xs0 = np.nonzero(mask_s)
				if ys0.size == 0:
					return None
				cy = float(np.mean(ys0))
				cx = float(np.mean(xs0))
			ys, xs = np.nonzero(mask_s)
			vals = img[ys, xs]
			ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360.0) % 360.0
			bins = np.floor(ang).astype(np.int32) % 360
			prof = np.full((360,), np.nan, dtype=np.float64)
			for b in range(360):
				vb = vals[bins == b]
				if vb.size:
					prof[b] = float(np.percentile(vb, 70))
			return _fill_profile_nans_circular(prof)

		def _slice_angular_profile_from_gate(gate_cube: np.ndarray, seg_obj, s_idx: int) -> np.ndarray | None:
			img = np.asarray(gate_cube[int(s_idx)], dtype=np.float64)
			mask_s = np.asarray(seg_obj.mask[int(s_idx)], dtype=bool)
			if not np.any(mask_s):
				return None
			cy, cx = seg_obj.center_per_slice[int(s_idx)]
			if not (np.isfinite(cy) and np.isfinite(cx)):
				ys0, xs0 = np.nonzero(mask_s)
				if ys0.size == 0:
					return None
				cy = float(np.mean(ys0))
				cx = float(np.mean(xs0))
			ys, xs = np.nonzero(mask_s)
			vals = img[ys, xs]
			ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360.0) % 360.0
			bins = np.floor(ang).astype(np.int32) % 360
			prof = np.full((360,), np.nan, dtype=np.float64)
			for b in range(360):
				vb = vals[bins == b]
				if vb.size:
					prof[b] = float(np.percentile(vb, 70))
			return _fill_profile_nans_circular(prof)

		apex_to_base = list(getattr(self.aha, "apex_to_base_order", []) or [])
		if not apex_to_base:
			apex_to_base = [int(s) for s in np.where(self.seg.mask.reshape(self.seg.mask.shape[0], -1).any(axis=1))[0].tolist()]
		profiles = []
		for s in apex_to_base:
			p = _slice_angular_profile(int(s))
			if p is not None:
				profiles.append(p)

		if len(profiles) >= 2 and need_tab_render.get("polar_perfusion_directa", True):
			perf_bg = "#000000"
			perf_grid = "#7f8a9a"
			perf_fg = "#f3f4f6"
			perf_subtle = "#9ca3af"
			profiles_arr = np.asarray(profiles, dtype=np.float64)
			nr, nt = 220, 360
			rotation_deg = int(self.polar_rotation_spin.value())
			rotation_bins = int(np.round(rotation_deg)) % 360
			polar_map = np.zeros((nr, nt), dtype=np.float64)
			for ir in range(nr):
				t = (ir / max(1, nr - 1)) * (profiles_arr.shape[0] - 1)
				i0 = int(np.floor(t))
				i1 = min(i0 + 1, profiles_arr.shape[0] - 1)
				a = float(t - i0)
				polar_map[ir] = (1.0 - a) * profiles_arr[i0] + a * profiles_arr[i1]
			if rotation_bins:
				polar_map = np.roll(polar_map, shift=rotation_bins, axis=1)

			mx_pm = float(np.nanmax(polar_map)) if np.isfinite(polar_map).any() else 0.0
			polar_map = polar_map / (mx_pm + 1e-8)
			smooth_method = str(self.polar_perf_smooth_method_combo.currentText()).strip().lower()
			smooth_strength = float(self.polar_perf_smooth_strength_spin.value())

			def _smooth_polar_perfusion_map(pm: np.ndarray) -> np.ndarray:
				arr = np.asarray(pm, dtype=np.float64)
				if arr.size == 0 or smooth_strength <= 0.001:
					return arr
				if smooth_method.startswith("butter"):
					h, w = arr.shape
					fy = np.fft.fftfreq(h).reshape(-1, 1)
					fx = np.fft.fftfreq(w).reshape(1, -1)
					rr = np.sqrt(fy * fy + fx * fx)
					cutoff = float(np.clip(0.42 / (1.0 + 0.38 * smooth_strength), 0.035, 0.45))
					order = 2.0
					transfer = 1.0 / (1.0 + np.power(rr / max(cutoff, 1e-6), 2.0 * order))
					smoothed = np.real(np.fft.ifft2(np.fft.fft2(arr) * transfer))
					return np.clip(smoothed, 0.0, 1.0)
				sigma_radial = max(0.05, smooth_strength)
				sigma_angular = max(0.05, smooth_strength * 0.60)
				return np.clip(gaussian_filter(arr, sigma=(sigma_radial, sigma_angular)), 0.0, 1.0)

			polar_map_smooth = _smooth_polar_perfusion_map(polar_map)

			def _polar_to_cartesian(pm: np.ndarray, size: int = 480) -> np.ndarray:
				canvas = np.full((size, size), np.nan, dtype=np.float64)
				yy, xx = np.indices((size, size), dtype=np.float64)
				cxp = (size - 1) / 2.0
				cyp = (size - 1) / 2.0
				xn = (xx - cxp) / max(1.0, cxp)
				yn = (yy - cyp) / max(1.0, cyp)
				rr = np.sqrt(xn**2 + yn**2)
				inside = rr <= 1.0
				ang = (np.degrees(np.arctan2(yn, xn)) + 360.0) % 360.0
				ri = np.clip((rr * (pm.shape[0] - 1)).astype(np.int32), 0, pm.shape[0] - 1)
				ti = np.clip(np.floor(ang).astype(np.int32), 0, pm.shape[1] - 1)
				canvas[inside] = pm[ri[inside], ti[inside]]
				return canvas

			cart_raw = _polar_to_cartesian(polar_map)
			cart_smooth = _polar_to_cartesian(polar_map_smooth)
			# Cache para recoloreo en pantalla (cmap independiente del informe): guarda
			# los mapas cartesianos float 0..1 ya calculados; el preview los re-rinde
			# con self.polar_perf_screen_cmap sin regenerar el pipeline ni tocar disco.
			self._polar_perf_cart_cache = {
				"raw": cart_raw,
				"smooth": cart_smooth,
				"polar_map": polar_map_smooth,
				"label": str(study_context_label),
				"rotation_deg": int(rotation_deg),
				"smooth_desc": f"{self.polar_perf_smooth_method_combo.currentText()} {smooth_strength:.2f}",
			}

			def _annotate_polar_guides(ax, canvas_size: int):
				c = canvas_size * 0.5
				r = canvas_size * 0.5
				for frac in (0.25, 0.50, 0.75, 1.0):
					ax.add_patch(
						plt.Circle(
							(c, c),
							radius=r * frac,
							fill=False,
							color=perf_grid,
							linewidth=0.8,
							alpha=0.75,
						)
					)
				# Cruces anatómicas simplificadas para lectura rápida clínica.
				ax.plot([c - r, c + r], [c, c], color=perf_grid, linewidth=0.8, alpha=0.8)
				ax.plot([c, c], [c - r, c + r], color=perf_grid, linewidth=0.8, alpha=0.8)
				ax.text(c, c - r * 1.03, "ANT", ha="center", va="bottom", color=perf_fg, fontsize=8, fontweight="bold")
				ax.text(c + r * 1.03, c, "LAT", ha="left", va="center", color=perf_fg, fontsize=8, fontweight="bold")
				ax.text(c, c + r * 1.03, "INF", ha="center", va="top", color=perf_fg, fontsize=8, fontweight="bold")
				ax.text(c - r * 1.03, c, "SEP", ha="right", va="center", color=perf_fg, fontsize=8, fontweight="bold")
				ax.text(c, c, "APEX", ha="center", va="center", color=perf_fg, fontsize=7, fontweight="bold")
				ax.text(c, c + r * 0.98, "BASE", ha="center", va="top", color=perf_subtle, fontsize=7, fontweight="bold")

			fig_pp, axes_pp = plt.subplots(1, 2, figsize=(12.0, 6.0), facecolor=perf_bg)
			for ax_pp, img_pp, ttl in [
				(axes_pp[0], cart_raw, "Perfusión polar directa (crudo)"),
				(axes_pp[1], cart_smooth, f"Perfusión polar directa ({self.polar_perf_smooth_method_combo.currentText()} {smooth_strength:.2f})"),
			]:
				ax_pp.set_facecolor(perf_bg)
				ax_pp.set_aspect("equal")
				ax_pp.set_xticks([])
				ax_pp.set_yticks([])
				im_pp = ax_pp.imshow(img_pp, cmap=cmap_polar_perf, vmin=0.0, vmax=1.0)
				_annotate_polar_guides(ax_pp, int(img_pp.shape[0]))
				ax_pp.set_title(ttl, color=perf_fg, fontsize=10, fontweight="bold")
				cbar = fig_pp.colorbar(im_pp, ax=ax_pp, fraction=0.046, pad=0.03)
				cbar.set_ticks([])
				cbar.outline.set_edgecolor("white")
				cbar.ax.set_facecolor(perf_bg)
			fig_pp.suptitle(f"Mapa polar de perfusión (apex en centro, base en borde) — {study_context_label} — rotación {rotation_deg:+d}°", color=perf_fg, fontsize=11.5, fontweight="bold")
			fig_pp.text(0.5, 0.02, "Reconstrucción polar continua desde short-axis: 'aplastado' apex->base", ha="center", color=perf_subtle, fontsize=8.6)
			self._stamp_export_figure(fig_pp, active_cine_widget)
			fig_pp.savefig(os.path.join(self.output_dir, "polar_perfusion_directa.png"), dpi=185, bbox_inches="tight", facecolor=fig_pp.get_facecolor())
			plt.close(fig_pp)

			# PNG solo con el mapa suavizado (para el informe HTML).
			try:
				fig_smooth = plt.figure(figsize=(7.0, 7.0), facecolor=perf_bg)
				ax_s = fig_smooth.add_axes([0.02, 0.02, 0.96, 0.96])
				ax_s.set_facecolor(perf_bg)
				ax_s.set_aspect("equal")
				ax_s.set_xticks([])
				ax_s.set_yticks([])
				im_s = ax_s.imshow(cart_smooth, cmap=cmap_polar_perf, vmin=0.0, vmax=1.0)
				_annotate_polar_guides(ax_s, int(cart_smooth.shape[0]))
				cbar_s = fig_smooth.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.03)
				cbar_s.set_ticks([])
				cbar_s.outline.set_edgecolor("white")
				cbar_s.ax.set_facecolor(perf_bg)
				fig_smooth.savefig(os.path.join(self.output_dir, "polar_perfusion_smooth.png"), dpi=150, bbox_inches="tight", facecolor=fig_smooth.get_facecolor())
				plt.close(fig_smooth)
			except Exception:
				pass

			# Cine polar gatillado por gate: genera GIF y un montaje estático para preview/PDF.
			try:
				from PIL import Image
			except Exception:
				Image = None

			def _render_gate_frame(study_cube_all: np.ndarray, seg_obj, apex_order, gate_index: int, label_text: str):
				gate_cube = np.asarray(study_cube_all[int(gate_index)], dtype=np.float64)
				profiles_g = []
				for s in apex_order:
					pg = _slice_angular_profile_from_gate(gate_cube, seg_obj, int(s))
					if pg is not None:
						profiles_g.append(pg)
				if len(profiles_g) < 2:
					return None, None
				arr_g = np.asarray(profiles_g, dtype=np.float64)
				pm_g = np.zeros((nr, nt), dtype=np.float64)
				for ir in range(nr):
					t = (ir / max(1, nr - 1)) * (arr_g.shape[0] - 1)
					i0 = int(np.floor(t))
					i1 = min(i0 + 1, arr_g.shape[0] - 1)
					a = float(t - i0)
					pm_g[ir] = (1.0 - a) * arr_g[i0] + a * arr_g[i1]
				if rotation_bins:
					pm_g = np.roll(pm_g, shift=rotation_bins, axis=1)
				mx_g = float(np.nanmax(pm_g)) if np.isfinite(pm_g).any() else 0.0
				pm_g = pm_g / (mx_g + 1e-8)
				pm_g = _smooth_polar_perfusion_map(pm_g)
				cart_g = _polar_to_cartesian(pm_g)
				fig_g, ax_g = plt.subplots(1, 1, figsize=(5.2, 5.2), facecolor=perf_bg)
				ax_g.set_facecolor(perf_bg)
				ax_g.set_aspect("equal")
				ax_g.set_xticks([])
				ax_g.set_yticks([])
				ax_g.imshow(cart_g, cmap=cmap_polar_perf, vmin=0.0, vmax=1.0)
				_annotate_polar_guides(ax_g, int(cart_g.shape[0]))
				ax_g.set_title(f"{label_text} gate {gate_index + 1}/{int(study_cube_all.shape[0])}", color=perf_fg, fontsize=10, fontweight="bold")
				fig_g.tight_layout()
				fig_g.canvas.draw()
				w, h = fig_g.canvas.get_width_height()
				buf = np.frombuffer(fig_g.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
				plt.close(fig_g)
				return buf, pm_g

			def _math_map(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray | None:
				if op == "Ninguna":
					return None
				if op == "Suma":
					return np.clip(a + b, 0.0, 1.0)
				if op == "Resta":
					return np.clip(a - b, 0.0, 1.0)
				if op == "Multiplicación":
					return np.clip(a * b, 0.0, 1.0)
				if op == "División":
					div = a / np.maximum(b, 1e-6)
					mx = float(np.nanmax(div)) if np.isfinite(div).any() else 0.0
					return np.clip(div / (mx + 1e-8), 0.0, 1.0)
				return None

			def _render_math_panel(pm_map: np.ndarray, gate_index: int, label_text: str):
				cart_m = _polar_to_cartesian(pm_map)
				fig_mx, ax_mx = plt.subplots(1, 1, figsize=(5.2, 5.2), facecolor=perf_bg)
				ax_mx.set_facecolor(perf_bg)
				ax_mx.set_aspect("equal")
				ax_mx.set_xticks([])
				ax_mx.set_yticks([])
				ax_mx.imshow(cart_m, cmap=cmap_polar_perf, vmin=0.0, vmax=1.0)
				_annotate_polar_guides(ax_mx, int(cart_m.shape[0]))
				ax_mx.set_title(f"{label_text} gate {gate_index + 1}", color=perf_fg, fontsize=10, fontweight="bold")
				fig_mx.tight_layout()
				fig_mx.canvas.draw()
				w, h = fig_mx.canvas.get_width_height()
				buf = np.frombuffer(fig_mx.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
				plt.close(fig_mx)
				return buf

			def _phase_label_from_path(path_text: str, fallback: str) -> str:
				u = os.path.basename(str(path_text)).upper()
				if "REST" in u:
					return "Reposo"
				if "STRESS" in u:
					return "Esfuerzo"
				return fallback

			active_primary_path = str(getattr(self, "_output_study_path_override", "") or self.file_edit.text().strip())
			primary_phase_label = _phase_label_from_path(active_primary_path, "Estudio")
			compare_phase_label = _phase_label_from_path(
				str(self.compare_bundle.get("path", "")) if self.compare_bundle is not None else "",
				"Comparación",
			)

			primary_frames: list[np.ndarray] = []
			compare_frames: list[np.ndarray] = []
			# Cache de mapas polares por gate para recolorear el cine en pantalla.
			cine_cart_frames: list[list[dict]] = []
			frame_count = int(study_cube_render.shape[0])
			compare_cube_render = None
			if self.compare_bundle is not None and self.compare_bundle.get("study") is not None:
				compare_cube_render = self._apply_intestinal_mask_to_cube(self.compare_bundle["study"].cube, self.cine_compare)
				frame_count = min(frame_count, int(compare_cube_render.shape[0]))
				compare_apex_to_base = list(getattr(self.compare_bundle.get("aha"), "apex_to_base_order", []) or [])
				if not compare_apex_to_base:
					compare_apex_to_base = [int(s) for s in np.where(np.asarray(self.compare_bundle["seg"].mask).reshape(self.compare_bundle["seg"].mask.shape[0], -1).any(axis=1))[0].tolist()]
			else:
				compare_apex_to_base = []

			for g in range(frame_count):
				p_frame, p_pm = _render_gate_frame(study_cube_render, self.seg, apex_to_base, g, primary_phase_label)
				if p_frame is None:
					continue
				primary_frames.append(p_frame)
				primary_title = f"{primary_phase_label} gate {g + 1}/{int(study_cube_render.shape[0])}"
				if self.compare_bundle is not None and self.compare_bundle.get("study") is not None:
					r_frame, r_pm = _render_gate_frame(compare_cube_render, self.compare_bundle["seg"], compare_apex_to_base, g, compare_phase_label)
					if r_frame is None:
						compare_frames.append(p_frame)
						cine_cart_frames.append([{"pm": np.asarray(p_pm, dtype=np.float32), "title": primary_title}])
					else:
						gap = np.full((p_frame.shape[0], 28, 3), 12, dtype=np.uint8)
						panels = [p_frame, gap, r_frame]
						rest_title = f"{compare_phase_label} gate {g + 1}/{int(compare_cube_render.shape[0])}"
						cache_panels = [
							{"pm": np.asarray(p_pm, dtype=np.float32), "title": primary_title},
							{"pm": np.asarray(r_pm, dtype=np.float32), "title": rest_title},
						]
						op_name = str(self.polar_compare_math_combo.currentText())
						if op_name != "Ninguna" and p_pm is not None and r_pm is not None:
							a_name = str(self.polar_compare_term_a_combo.currentText())
							b_name = str(self.polar_compare_term_b_combo.currentText())
							a_map = p_pm if a_name == "Esfuerzo" else r_pm
							b_map = p_pm if b_name == "Esfuerzo" else r_pm
							pm_math = _math_map(a_map, b_map, op_name)
							if pm_math is not None:
								math_label = f"{a_name} {op_name} {b_name}"
								m_frame = _render_math_panel(pm_math, g, math_label)
								panels.extend([gap, m_frame])
								cache_panels.append({"pm": np.asarray(pm_math, dtype=np.float32), "title": f"{math_label} gate {g + 1}"})
						compare_frames.append(np.concatenate(panels, axis=1))
						cine_cart_frames.append(cache_panels)
				else:
					compare_frames.append(p_frame)
					cine_cart_frames.append([{"pm": np.asarray(p_pm, dtype=np.float32), "title": primary_title}])

			gate_frames = compare_frames
			self._polar_cine_cart_cache = {"frames": cine_cart_frames, "disk_cmap": cmap_polar_perf} if cine_cart_frames else None

			if gate_frames:
				polar_cine_ms = int(self.polar_cine_speed_spin.value())
				export_mp4 = bool(self.export_polar_mp4_check.isChecked())
				active_math_text = ""
				if self.compare_bundle is not None:
					active_math_text = self._polar_compare_operation_text()
				if Image is not None:
					pil_frames = [Image.fromarray(frm) for frm in gate_frames]
					pil_frames[0].save(
						os.path.join(self.output_dir, "polar_cine.gif"),
						save_all=True,
						append_images=pil_frames[1:],
						duration=polar_cine_ms,
						loop=0,
						disposal=2,
						optimize=False,
					)

				if export_mp4:
					fps = max(1.0, 1000.0 / max(1, polar_cine_ms))
					mp4_path = os.path.join(self.output_dir, "polar_cine.mp4")
					mp4_done = False
					try:
						import cv2
						h, w = gate_frames[0].shape[:2]
						writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(w), int(h)))
						for frm in gate_frames:
							writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
						writer.release()
						mp4_done = True
					except Exception:
						mp4_done = False

					if not mp4_done:
						try:
							import imageio.v2 as imageio
							imageio.mimsave(mp4_path, gate_frames, fps=float(fps))
							mp4_done = True
						except Exception:
							mp4_done = False
					if not mp4_done:
						self._log("[WARN] No se pudo exportar polar_cine.mp4 (faltan códecs/librerías).")

				# Montaje estático para preview/PDF
				n_show = min(8, len(gate_frames))
				idx = np.linspace(0, len(gate_frames) - 1, n_show).astype(int)
				fig_m, axes_m = plt.subplots(2, int(np.ceil(n_show / 2.0)), figsize=(12, 6.2), facecolor=perf_bg)
				axes_arr = np.atleast_1d(axes_m).ravel()
				for i, ax in enumerate(axes_arr):
					ax.set_facecolor(perf_bg)
					ax.set_xticks([])
					ax.set_yticks([])
					if i < n_show:
						ax.imshow(gate_frames[int(idx[i])])
						ax.set_title(f"Gate {int(idx[i]) + 1}", color=perf_fg, fontsize=9)
					else:
						ax.axis("off")
				fig_m.suptitle(f"Polar cine gatillado (muestra de gates) — {study_context_label}", color=perf_fg, fontsize=11.5, fontweight="bold")
				if active_math_text:
					fig_m.text(
						0.5,
						0.055,
						f"Operación stress/rest aplicada: {active_math_text}",
						ha="center",
						color=perf_fg,
						fontsize=8.7,
						fontweight="bold",
					)
				fig_m.text(0.5, 0.02, "Uso clínico: evaluar dinámica temporal del patrón polar; en stress/rest comparar evolución de sincronía por gate.", ha="center", color=perf_subtle, fontsize=8.2)
				self._stamp_export_figure(fig_m, active_cine_widget)
				fig_m.savefig(os.path.join(self.output_dir, "polar_cine_montaje.png"), dpi=160, bbox_inches="tight", facecolor=fig_m.get_facecolor())
				plt.close(fig_m)

	def _write_compare_axes_panel(self, cmap_compare: str = "hot", build_cine: bool | None = None):
		if self.study is None or self.seg is None:
			return
		import matplotlib.pyplot as plt
		try:
			from scipy.ndimage import zoom as ndi_zoom
		except Exception:
			ndi_zoom = None
		fast_mode = bool(self.compare_fast_drag_check.isChecked() and self.compare_interactive_fast_mode)
		render_dpi = 130 if fast_mode else 240
		interp_mode = "nearest" if fast_mode else "lanczos"
		zoom_factor = max(1.0, float(self.compare_axes_zoom_slider.value()) / 100.0)
		apply_intestinal = bool(self.global_intestinal_render_check.isChecked() and self.compare_axes_intestinal_mask_check.isChecked())

		def _norm(img):
			arr = np.asarray(img, dtype=np.float64)
			mx = float(np.nanmax(arr)) if arr.size else 0.0
			return arr / (mx + 1e-8)

		def _windowed(img):
			lo = float(self.compare_window_low_slider.value()) / 100.0
			hi = float(self.compare_window_high_slider.value()) / 100.0
			hi = max(hi, lo + 0.01)
			arr = _norm(img)
			return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

		def _zoom_center(img2d: np.ndarray) -> np.ndarray:
			arr = np.asarray(img2d, dtype=np.float64)
			if arr.ndim != 2 or zoom_factor <= 1.001:
				return arr
			h, w = arr.shape
			crop_h = max(8, int(round(h / zoom_factor)))
			crop_w = max(8, int(round(w / zoom_factor)))
			y0 = max(0, (h - crop_h) // 2)
			x0 = max(0, (w - crop_w) // 2)
			crop = arr[y0:y0 + crop_h, x0:x0 + crop_w]
			if crop.size == 0:
				return arr
			if (not fast_mode) and (ndi_zoom is not None):
				scale_y = float(h) / float(max(1, crop_h))
				scale_x = float(w) / float(max(1, crop_w))
				up = ndi_zoom(crop, (scale_y, scale_x), order=3, mode="nearest", prefilter=True)
			else:
				zy = max(1, int(np.ceil(h / max(1, crop_h))))
				zx = max(1, int(np.ceil(w / max(1, crop_w))))
				up = np.repeat(np.repeat(crop, zy, axis=0), zx, axis=1)
			if up.shape[0] < h or up.shape[1] < w:
				pad_y = max(0, h - up.shape[0])
				pad_x = max(0, w - up.shape[1])
				up = np.pad(up, ((0, pad_y), (0, pad_x)), mode="edge")
			return up[:h, :w]

		def _sample_even(indices: list[int], n_cols: int) -> list[int]:
			if not indices:
				return []
			if len(indices) <= n_cols:
				return indices
			idx = np.linspace(0, len(indices) - 1, n_cols).astype(int)
			return [indices[int(i)] for i in idx]

		def _shift_index(value: int, shift: int, limit: int) -> int:
			return int(max(0, min(max(0, limit - 1), int(value) + int(shift))))

		def _axis_orient(img2d: np.ndarray, axis_name: str) -> np.ndarray:
			img2d = _zoom_center(img2d)
			if axis_name == "HLA":
				return _windowed(img2d)
			if axis_name == "VLA":
				return np.rot90(_windowed(img2d), k=3)
			return _windowed(img2d)

		def _mask_orient(mask2d: np.ndarray, axis_name: str) -> np.ndarray:
			arr = np.asarray(mask2d, dtype=np.float64)
			if axis_name == "HLA":
				return arr
			if axis_name == "VLA":
				return np.rot90(arr, k=3)
			return arr

		def _annotate_orientation(ax, axis_name: str):
			style = dict(
				transform=ax.transAxes,
				fontsize=5.8,
				fontweight="bold",
				color="#e2e8f0",
				bbox=dict(boxstyle="round,pad=0.10", facecolor="black", edgecolor="#00bcd4", alpha=0.55),
			)
			if axis_name == "SA":
				ax.text(0.50, 0.98, "ANT", ha="center", va="top", **style)
				ax.text(0.50, 0.02, "INF", ha="center", va="bottom", **style)
				ax.text(0.02, 0.50, "SEPT", ha="left", va="center", rotation=90, **style)
				ax.text(0.98, 0.50, "LAT", ha="right", va="center", rotation=270, **style)
			elif axis_name == "HLA":
				ax.text(0.50, 0.98, "APEX", ha="center", va="top", **style)
				ax.text(0.50, 0.02, "BASE", ha="center", va="bottom", **style)
				ax.text(0.02, 0.50, "SEPT", ha="left", va="center", rotation=90, **style)
				ax.text(0.98, 0.50, "LAT", ha="right", va="center", rotation=270, **style)
			else:
				ax.text(0.50, 0.98, "ANT", ha="center", va="top", **style)
				ax.text(0.50, 0.02, "INF", ha="center", va="bottom", **style)
				ax.text(0.02, 0.50, "BASE", ha="left", va="center", rotation=90, **style)
				ax.text(0.98, 0.50, "APEX", ha="right", va="center", rotation=270, **style)

		def _phase_caption(path_text: str, fallback: str) -> str:
			u = os.path.basename(path_text).upper()
			if "STRESS" in u:
				return "ESFUERZO"
			if "REST" in u:
				return "REPOSO"
			return fallback

		def _extract_rows(study, seg, path_text: str, cine_widget: CineWidget | None, gate_override: int | None = None):
			gate = max(0, min(int(study.cube.shape[0]) - 1, int(self.compare_gate_spin.value()) - 1 if gate_override is None else int(gate_override)))
			vol_gate = np.asarray(study.cube[int(gate)], dtype=np.float64)
			if apply_intestinal and cine_widget is not None:
				try:
					vol_gate = cine_widget.apply_intestinal_mask_to_gate_volume(vol_gate, gate_index=int(gate))
				except Exception:
					pass
			mask3d = np.asarray(seg.mask, dtype=bool)
			n_slices, h, w = vol_gate.shape
			rows = []
			for axis_name in ("SA", "VLA", "HLA"):
				if axis_name == "SA":
					offset = int(self.compare_slice_offset_sa_spin.value())
					valid_sa = [int(s) for s in np.where(mask3d.reshape(n_slices, -1).any(axis=1))[0].tolist()]
					if not valid_sa:
						valid_sa = list(range(n_slices))
					sa_idx = _sample_even(valid_sa, 16)
					if len(sa_idx) < 16:
						last = sa_idx[-1] if sa_idx else 0
						sa_idx = (sa_idx + [last] * 16)[:16]
					sa_idx = [_shift_index(s_idx, offset, n_slices) for s_idx in sa_idx]
					row_source = sa_idx
				elif axis_name == "HLA":
					offset = int(self.compare_slice_offset_hla_spin.value())
					row_source = [_shift_index(v, offset, h) for v in np.linspace(max(0, int(0.08 * h)), max(0, int(0.92 * h)), 16).astype(int).tolist()]
				else:
					offset = int(self.compare_slice_offset_vla_spin.value())
					row_source = [_shift_index(v, offset, w) for v in np.linspace(max(0, int(0.08 * w)), max(0, int(0.92 * w)), 16).astype(int).tolist()]
				row_imgs = []
				row_masks = []
				row_titles = []
				for col in range(16):
					if axis_name == "SA":
						s_idx = int(row_source[col])
						row_imgs.append(_axis_orient(vol_gate[s_idx], "SA"))
						row_masks.append(_mask_orient(mask3d[s_idx].astype(np.float64), "SA"))
						row_titles.append(f"SA {s_idx + 1}")
					elif axis_name == "HLA":
						yv = int(row_source[col])
						row_imgs.append(_axis_orient(vol_gate[:, yv, :], "HLA"))
						row_masks.append(_mask_orient(mask3d[:, yv, :].astype(np.float64), "HLA"))
						row_titles.append(f"HLA {yv + 1}")
					else:
						xv = int(row_source[col])
						row_imgs.append(_axis_orient(vol_gate[:, :, xv], "VLA"))
						row_masks.append(_mask_orient(mask3d[:, :, xv].astype(np.float64), "VLA"))
						row_titles.append(f"VLA {xv + 1}")
				rows.append((axis_name, row_imgs, row_masks, row_titles))
			return gate, rows, _phase_caption(path_text, "ESTUDIO")

		def _build_compare_figure(primary_gate_override: int | None = None, secondary_gate_override: int | None = None):
			show_mask = bool(self.compare_mask_check.isChecked())
			primary_gate, primary_rows, primary_phase = _extract_rows(
				self.study,
				self.seg,
				self.file_edit.text().strip(),
				self.cine,
				gate_override=primary_gate_override,
			)
			if self.compare_bundle is not None:
				secondary_gate, secondary_rows, secondary_phase = _extract_rows(
					self.compare_bundle["study"],
					self.compare_bundle["seg"],
					str(self.compare_bundle.get("path", self.compare_label or "comparacion")),
					self.cine_compare,
					gate_override=secondary_gate_override,
				)
			else:
				secondary_gate, secondary_rows, secondary_phase = None, [], None

			row_specs = []
			for idx, axis_name in enumerate(("SA", "VLA", "HLA")):
				row_specs.append((axis_name, primary_phase, primary_rows[idx][1], primary_rows[idx][2], primary_rows[idx][3]))
				if secondary_rows:
					row_specs.append((axis_name, secondary_phase, secondary_rows[idx][1], secondary_rows[idx][2], secondary_rows[idx][3]))

			fig, axes = plt.subplots(len(row_specs), 16, figsize=(24, 1.8 * len(row_specs) + 1.2), facecolor="#04070f")
			axes = np.asarray(axes)
			if axes.ndim == 1:
				axes = axes.reshape(1, -1)

			for r, (axis_name, phase_name, row_imgs, row_masks, row_titles) in enumerate(row_specs):
				for c in range(16):
					ax = axes[r, c]
					ax.set_aspect("equal", adjustable="box")
					ax.imshow(row_imgs[c], cmap=cmap_compare, aspect="equal", interpolation=interp_mode, resample=True)
					if show_mask and np.any(row_masks[c] > 0):
						ax.contour(row_masks[c], levels=[0.5], colors=["#ffffff"], linewidths=1.0 if fast_mode else 1.4)
						ax.contour(row_masks[c], levels=[0.5], colors=["#00e5ff"], linewidths=0.5 if fast_mode else 0.7)
					ax.set_xticks([])
					ax.set_yticks([])
					ax.set_facecolor("#000000")
					ax.set_title(row_titles[c], fontsize=6.5, color="#cbd5e1", pad=1.0)
					if c == 0:
						_annotate_orientation(ax, axis_name)
				axes[r, 0].text(
					-0.16,
					0.5,
					f"{axis_name}: {phase_name}",
					transform=axes[r, 0].transAxes,
					rotation=90,
					va="center",
					ha="right",
					fontsize=8.5,
					color="#93c5fd",
					fontweight="bold",
				)

			gate_text = f"Gate esfuerzo {int(primary_gate) + 1}"
			if secondary_gate is not None:
				gate_text += f" | Gate reposo {int(secondary_gate) + 1}"
			mask_txt = "ON" if show_mask else "OFF"
			int_txt = "ON" if apply_intestinal else "OFF"
			fig.suptitle(
				f"Comparativa de ejes clínica — 16 cortes por eje — {gate_text} — Máscara {mask_txt} — ROI intestino {int_txt} — Zoom {int(self.compare_axes_zoom_slider.value())}% — Top {int(self.compare_window_high_slider.value())}% / Base {int(self.compare_window_low_slider.value())}%",
				fontsize=12,
				fontweight="bold",
				color="#f8fafc",
			)
			fig.tight_layout(rect=(0.02, 0.02, 1, 0.94))
			return fig

		fig = _build_compare_figure()
		self._stamp_export_figure(fig, self.cine)
		fig.savefig(os.path.join(self.output_dir, "comparacion_ejes.png"), dpi=render_dpi, bbox_inches="tight")
		if os.path.exists(os.path.join(self.output_dir, "comparacion_ejes.gif")):
			try:
				os.remove(os.path.join(self.output_dir, "comparacion_ejes.gif"))
			except OSError:
				pass
		if build_cine is None:
			build_cine = bool(self.compare_axes_cine_check.isChecked()) and (not fast_mode) and self._is_tab_active("comparacion_ejes")
		if bool(build_cine) and bool(self.compare_axes_cine_check.isChecked()) and (not fast_mode):
			primary_gate_count = int(self.study.cube.shape[0]) if self.study is not None else 0
			secondary_gate_count = int(self.compare_bundle["study"].cube.shape[0]) if self.compare_bundle is not None else primary_gate_count
			frame_count = max(1, min(primary_gate_count, secondary_gate_count))
			frames: list[QPixmap] = []
			for gate_index in range(frame_count):
				frame_fig = _build_compare_figure(gate_index, gate_index if self.compare_bundle is not None else None)
				frame_fig.canvas.draw()
				w, h = frame_fig.canvas.get_width_height()
				buf = np.frombuffer(frame_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
				frames.append(self._rgb_frame_to_qpixmap(buf))
				plt.close(frame_fig)
			if frames:
				self.compare_axes_preview_frames = frames
				self.compare_axes_preview_index = 0
				self.compare_axes_cine_timer.setInterval(max(40, int(self.compare_axes_cine_speed_spin.value())))
				if self.compare_axes_playing:
					self.compare_axes_cine_timer.start()
				else:
					self.compare_axes_cine_timer.stop()
		else:
			self.compare_axes_preview_frames = []
			self.compare_axes_preview_index = 0
			self.compare_axes_playing = False
			self.compare_axes_cine_timer.stop()
		plt.close(fig)

	def _load_preview(self, name: str):
		if name == "polar_perfusion_directa" and self.polar_view_mode == "cine":
			self._load_polar_cine_preview()
			return
		if name == "polar_perfusion_directa" and getattr(self, "_polar_perf_cart_cache", None) and self.compare_bundle is None:
			# Vista estática con el cmap de pantalla (independiente del informe):
			# recolorea en memoria desde la caché en vez de cargar el PNG de disco.
			# En dual NO: el PNG de disco es el compuesto Esfuerzo | Reposo.
			if self._rerender_polar_perfusion_screen():
				return
		if name == "comparacion_ejes":
			self._load_compare_axes_preview()
			return
		# QC (ex-ungated): conserva el panel QC crudo cargado al abrir el estudio;
		# no se re-renderiza ni se sobrescribe con la grilla desgatillada.
		if name == "ungated":
			return
		fname = f"{name}.png"
		path = os.path.join(self.output_dir, fname)
		label = self.preview_labels[name]
		if os.path.exists(path):
			pix = QPixmap(path)
			self.preview_pixmaps[name] = pix
			self.preview_base_sizes[name] = pix.size()
			self._apply_preview_zoom(name)
		else:
			self.preview_pixmaps.pop(name, None)
			self.preview_base_sizes.pop(name, None)
			label.setText("Sin imagen")

	def _load_previews(self):
		for name in self.preview_labels:
			self._load_preview(name)

	def _load_previews_selected(self, names):
		for name in names:
			if name in self.preview_labels:
				self._load_preview(name)

	def _load_polar_cine_preview(self):
		name = "polar_perfusion_directa"
		label = self.preview_labels[name]
		gif_path = os.path.join(self.output_dir, "polar_cine.gif")
		png_path = os.path.join(self.output_dir, "polar_cine_montaje.png")
		self.polar_cine_timer.stop()
		self.polar_cine_playing = False
		movie = self.preview_movies.pop(name, None)
		if movie is not None:
			movie.stop()
			label.clear()
			label.setMovie(None)
		# Recolor en memoria SOLO si el cmap de pantalla difiere del que generó el GIF
		# (informe intacto); si coinciden, cargar el GIF de disco es instantáneo.
		_cine_cache = getattr(self, "_polar_cine_cart_cache", None)
		if _cine_cache and str(getattr(self, "polar_perf_screen_cmap", "")) != str(_cine_cache.get("disk_cmap", "")):
			if self._rebuild_polar_cine_frames_screen():
				return
		if os.path.exists(gif_path):
			frames: list[QPixmap] = []
			duration_ms = int(self.polar_cine_speed_spin.value())
			try:
				from PIL import Image, ImageSequence
				with Image.open(gif_path) as im:
					duration_ms = int(im.info.get("duration", duration_ms))
					for frm in ImageSequence.Iterator(im):
						rgb = np.asarray(frm.convert("RGB"), dtype=np.uint8)
						frames.append(self._rgb_frame_to_qpixmap(rgb))
			except Exception:
				frames = []
			if frames:
				self.polar_cine_preview_frames = frames
				self.polar_cine_preview_index = 0
				self.polar_cine_timer.setInterval(max(40, int(duration_ms)))
				self._set_polar_cine_memory_frame(0)
				self._update_polar_cine_toggle_text(enabled=True)
				return
			self.polar_cine_preview_frames = []
			self.polar_cine_preview_index = 0
		if os.path.exists(png_path):
			pix = QPixmap(png_path)
			self.preview_pixmaps[name] = pix
			self.preview_base_sizes[name] = pix.size()
			self._update_polar_cine_toggle_text(enabled=False)
			self._apply_preview_zoom(name)
		else:
			self.preview_pixmaps.pop(name, None)
			self.preview_base_sizes.pop(name, None)
			self._update_polar_cine_toggle_text(enabled=False)
			label.setText("Sin cine polar")

	def _load_compare_axes_preview(self):
		# La tab comparacion_ejes ahora aloja el Montaje clínico SA/VLA/HLA.
		# Si hay un montaje generado, se muestra; si no, la tab queda vacía.
		name = "comparacion_ejes"
		label = self.preview_labels[name]
		montage_png = os.path.join(self.output_dir, "sa_montage.png")
		if str(getattr(self, "cine_crudo_preview_mode", "")) == "sa_montage" and os.path.isfile(montage_png):
			pix = QPixmap(montage_png)
			if not pix.isNull():
				self.preview_pixmaps[name] = pix
				self.preview_base_sizes[name] = pix.size()
				self._apply_preview_zoom(name)
				return
		# Sin montaje: no mostrar nada.
		movie = self.preview_movies.pop(name, None)
		if movie is not None:
			movie.stop()
			label.setMovie(None)
		self.preview_pixmaps.pop(name, None)
		self.preview_base_sizes.pop(name, None)
		self._update_compare_axes_toggle_text(enabled=False)
		label.clear()
		label.setText("")

	def _write_raw_mip_views_for_pdf(self):
		"""Genera tres proyecciones planares del estudio EN CRUDO (tórax completo) para
		el informe: anterior (AP), oblicua anterior izquierda (OAI 45°) y lateral
		izquierda. Cuando hay datos crudos usa los frames de adquisición reales
		(proyecciones por ángulo); si sólo hay volumen reconstruido, reproyecta.
		Robusto: si falla, no interrumpe el PDF."""
		if self.study is None:
			return
		try:
			import matplotlib
			matplotlib.use("Agg")
			import matplotlib.pyplot as plt
			raw = getattr(self, "cine_crudo_raw_study_for_recon", None) or self.study
			cube = np.asarray(getattr(raw, "cube", None), dtype=np.float64)
			if cube.size == 0:
				return
			start = getattr(raw, "start_angle", None)
			step = getattr(raw, "angular_step", None)
			rot = str(getattr(raw, "rotation_direction", "") or "")
			arc = getattr(raw, "scan_arc", None)
			is_raw_proj = (not bool(getattr(raw, "reconstructed", True))) and cube.ndim == 4

			if is_raw_proj:
				# Crudo real: cada frame por ángulo es una proyección planar del tórax.
				from core.raw_projections import ungate_projections
				ung = ungate_projections(cube)  # (angles, H, W)
				n_ang = int(ung.shape[0])
				# Grados por frame, avanzando en el sentido de giro.
				if step:
					deg_per_frame = abs(float(step))
				elif arc and n_ang > 1:
					deg_per_frame = abs(float(arc)) / float(n_ang - 1)
				else:
					deg_per_frame = 180.0 / max(1, n_ang - 1)

				def _frame(offset_deg):
					idx = int(round(float(offset_deg) / max(deg_per_frame, 1e-6)))
					return ung[int(np.clip(idx, 0, n_ang - 1))]

				# Offsets desde StartAngle (RAO 45°), avanzando en el sentido de giro:
				# AP=+45°, OAI 45° (LAO 45°)=+90°, lateral izq (LAO 90°)=+135°.
				specs = [
					(_frame(45.0), "raw_ap_mip.png", "AP (anterior)"),
					(_frame(90.0), "raw_oai_mip.png", "OAI 45°"),
					(_frame(135.0), "raw_ll_mip.png", "Lateral izq"),
				]
			else:
				# Sólo volumen reconstruido: reproyectar a los ángulos anatómicos.
				ung = cube.mean(axis=0) if cube.ndim == 4 else cube  # (z, y, x)
				from core.spect_geometry import reproject_view
				sign = -1.0 if rot.upper().startswith("CW") else 1.0
				if start is not None and ung.ndim == 3:
					ap = reproject_view(ung, (float(start) + sign * 45.0) % 360.0)
					oai = reproject_view(ung, (float(start) + sign * 90.0) % 360.0)
					ll = reproject_view(ung, (float(start) + sign * 135.0) % 360.0)
				else:
					ap = ung.sum(axis=1) if ung.ndim == 3 else ung
					oai = ap
					ll = ung.sum(axis=2) if ung.ndim == 3 else ung
				specs = [
					(ap, "raw_ap_mip.png", "AP (anterior)"),
					(oai, "raw_oai_mip.png", "OAI 45°"),
					(ll, "raw_ll_mip.png", "Lateral izq"),
				]

			cmap_axes = str(self.report_cmap_axes.currentText())
			for arr, fname, title in specs:
				a = np.asarray(arr, dtype=np.float64)
				p99 = float(np.percentile(a, 99.0)) if a.size else 0.0
				a = np.clip(a / max(p99, 1e-8), 0.0, 1.0)
				fig, axm = plt.subplots(figsize=(3.2, 4.4))
				axm.set_facecolor("#020611")
				axm.imshow(a, cmap=cmap_axes, interpolation="bicubic", aspect="equal")
				axm.set_title(title, fontsize=10, fontweight="bold")
				axm.set_xticks([])
				axm.set_yticks([])
				fig.tight_layout()
				fig.savefig(os.path.join(self.output_dir, fname), dpi=150, bbox_inches="tight")
				plt.close(fig)
		except Exception as exc:
			self._log(f"[WARN] Proyecciones AP/OAI/Lateral no generadas para PDF: {exc}")

	def _write_filtered_mip_views_for_pdf(self):
		"""Genera las mismas 3 proyecciones (AP, OAI 45°, Lat.izq) pero del volumen
		reconstruido CON los filtros aplicados (Denoise+, NITIDA, Suavizar, etc.)."""
		recon = getattr(self, "cine_crudo_recon_result", None)
		if recon is None or getattr(recon, "gated_volume", None) is None:
			return
		try:
			import matplotlib
			matplotlib.use("Agg")
			import matplotlib.pyplot as plt
			from core.spect_geometry import reproject_view

			vol = np.asarray(recon.gated_volume, dtype=np.float64)  # (gates, z, y, x)
			ung = vol.sum(axis=0) if vol.ndim == 4 else vol  # (z, y, x)
			raw = getattr(self, "cine_crudo_raw_study_for_recon", None) or self.study
			start = getattr(raw, "start_angle", None)
			rot = str(getattr(raw, "rotation_direction", "") or "")
			sign = -1.0 if rot.upper().startswith("CW") else 1.0

			if start is not None and ung.ndim == 3:
				ap = reproject_view(ung, (float(start) + sign * 45.0) % 360.0)
				oai = reproject_view(ung, (float(start) + sign * 90.0) % 360.0)
				ll = reproject_view(ung, (float(start) + sign * 135.0) % 360.0)
			else:
				ap = ung.sum(axis=1) if ung.ndim == 3 else ung
				oai = ap
				ll = ung.sum(axis=2) if ung.ndim == 3 else ung

			cmap_axes = str(self.report_cmap_axes.currentText())
			for arr, fname, title in [
				(ap, "filtered_ap_mip.png", "AP (filtrada)"),
				(oai, "filtered_oai_mip.png", "OAI 45° (filtrada)"),
				(ll, "filtered_ll_mip.png", "Lateral izq (filtrada)"),
			]:
				a = np.asarray(arr, dtype=np.float64)
				p99 = float(np.percentile(a, 99.0)) if a.size else 0.0
				a = np.clip(a / max(p99, 1e-8), 0.0, 1.0)
				fig, axm = plt.subplots(figsize=(3.2, 4.4))
				axm.set_facecolor("#020611")
				axm.imshow(a, cmap=cmap_axes, interpolation="bicubic", aspect="equal")
				axm.set_title(title, fontsize=10, fontweight="bold")
				axm.set_xticks([])
				axm.set_yticks([])
				fig.tight_layout()
				fig.savefig(os.path.join(self.output_dir, fname), dpi=150, bbox_inches="tight")
				plt.close(fig)
		except Exception as exc:
			self._log(f"[WARN] MIPs filtradas no generadas: {exc}")

	def _ensure_pdf_extra_images(self):
		"""Fuerza las imágenes que en modo básico no se generan en la corrida rápida
		pero deben ir sí o sí al PDF (guía fase VI, perfusión polar directa), genera
		las MIP AP/Lateral y persiste el montaje clínico si está en memoria."""
		self._write_raw_mip_views_for_pdf()
		try:
			self._write_outputs(target_tabs={"guia_fase_vi", "polar_perfusion_directa"})
		except Exception as exc:
			self._log(f"[WARN] No se pudieron generar imágenes pesadas para PDF: {exc}")
		# Montaje clínico: si hay pixmap en memoria (pestaña ya vista), persistirlo.
		try:
			mont_pix = self.preview_pixmaps.get("comparacion_ejes")
			mont_png = os.path.join(self.output_dir, "sa_montage.png")
			if (
				mont_pix is not None
				and not mont_pix.isNull()
				and str(getattr(self, "cine_crudo_preview_mode", "")) == "sa_montage"
			):
				mont_pix.save(mont_png, "PNG")
		except Exception:
			pass
		# Forzar generación del GIF del montaje cine (para el informe HTML).
		try:
			# Asegurar que el montaje esté renderizado antes de generar el GIF.
			if str(getattr(self, "cine_crudo_preview_mode", "")) != "sa_montage":
				self.cine_crudo_preview_mode = "sa_montage"
				self._show_cine_crudo_sa_montage()
			self._ensure_montage_cine_frames()
			frames = getattr(self, "_montage_cine_frames", None) or []
			if len(frames) >= 2:
				from PIL import Image
				pil_frames = []
				for fpix in frames:
					img = fpix.toImage()
					buf = img.bits().asstring(img.sizeInBytes())
					pil_frames.append(Image.frombuffer("RGBA", (img.width(), img.height()), buf, "raw", "BGRA"))
				gif_path = os.path.join(self.output_dir, "sa_montage_cine.gif")
				pil_frames[0].save(
					gif_path, save_all=True, append_images=pil_frames[1:],
					duration=int(self.polar_cine_speed_spin.value()), loop=0,
				)
				self._log(f"GIF montaje cine generado: {len(frames)} frames")
			else:
				self._log("[INFO] GIF montaje cine no generado: no hay suficientes frames.")
		except Exception as exc:
			self._log(f"[WARN] GIF montaje cine no generado: {exc}")
		# Capturar vistas 3D si el panel está abierto.
		try:
			lv3d = getattr(self, "_lv_3d_window", None)
			if lv3d is not None and lv3d.isVisible():
				lv3d.save_report_views(self.output_dir)
				self._log("Vistas 3D capturadas para informe.")
		except Exception as exc:
			self._log(f"[WARN] Captura 3D no disponible: {exc}")

	def _generate_pdf_report(self):
		if self.study is None or self.seg is None or self.metrics is None or self.territory is None:
			return
		pdf_path = os.path.join(self.output_dir, "informe_sincro.pdf")
		params = {
			"threshold": float(self.threshold_spin.value()),
			"smooth_sigma": float(self.sigma_spin.value()),
			"harmonics": int(self.harmonics_spin.value()),
			"amp_filter": float(self.metrics.get("amp_filter", self.phase_threshold_spin.value())),
			"visual_style": str(self.visual_style_combo.currentText()),
			"polar_rotation_deg": int(self.polar_rotation_spin.value()),
			"polar_cine_speed_ms": int(self.polar_cine_speed_spin.value()),
			"export_polar_mp4": bool(self.export_polar_mp4_check.isChecked()),
			"report_cmap_slices": str(self.report_cmap_slices.currentText()),
			"report_cmap_axes": str(self.report_cmap_axes.currentText()),
			"report_cmap_compare": str(self.report_cmap_compare.currentText()),
			"report_cmap_panel_axes": str(self.report_cmap_panel_axes.currentText()),
			"report_cmap_phase": str(self.report_cmap_phase.currentText()),
			"report_cmap_polar_clinico": str(self.report_cmap_polar_clinico.currentText()),
			"report_cmap_amp": str(self.report_cmap_amp.currentText()),
			"report_cmap_bullseye": str(self.report_cmap_bullseye.currentText()),
			"report_cmap_polar_perf": str(self.report_cmap_polar_perf.currentText()),
			"intestinal_subtraction": self.intestinal_subtraction_info,
			"ecg_ritmo": str(self.ecg_ritmo_combo.currentText()),
			"ecg_fc": int(self.ecg_fc_spin.value()),
			"ecg_qrs": int(self.ecg_qrs_spin.value()),
			"ecg_qt": int(self.ecg_qt_spin.value()),
			"ecg_bri": bool(self.ecg_bri_check.isChecked()),
			"ecg_brd": bool(self.ecg_brd_check.isChecked()),
			"ecg_marcapasos": bool(self.ecg_marcapasos_check.isChecked()),
			"ecg_observaciones": str(getattr(self, "ecg_observaciones_text", "")),
			"ecg_file_path": str(getattr(self, "ecg_file_path", "")),
		}
		vol = self._compute_volumes_ml()
		ef = self._estimate_lv_ef()
		vol = self._harmonize_volumes_with_ef(vol, ef)
		# Dato experimental: volumen miocárdico VI desde la máscara de fusión CT
		# (segmentación CT corregida por el usuario). Se compara en el informe
		# contra la masa/volumen miocárdico del gated SPECT.
		try:
			_panel = getattr(self, "_perfusion_fusion_panel", None)
			if _panel is not None and hasattr(_panel, "get_ct_mask_volume_ml"):
				_ct_myo_ml = _panel.get_ct_mask_volume_ml()
				if _ct_myo_ml is not None:
					vol["ct_mask_myo_ml"] = float(_ct_myo_ml)
		except Exception:
			pass
		if ef.get("available") and ef.get("thickening_pct") is not None:
			compare_ef = getattr(self, "compare_ef", None)
			if compare_ef and compare_ef.get("available") and compare_ef.get("thickening_pct") is not None:
				ef = dict(ef)
				ef["compare_thickening_pct"] = float(compare_ef["thickening_pct"])
				ef["compare_label"] = self.compare_label or "Comparación"
		report_metrics = dict(self.metrics)
		try:
			dataset, sex, protocol, nd = self._normal_db_context()
			report_metrics["normal_db_eval"] = nd
			report_metrics["normal_db_dataset"] = dataset
			report_metrics["normal_db_sex"] = sex
			report_metrics["normal_db_protocol"] = protocol
		except Exception:
			pass
		# Territorio/textura/stress-rest para el PDF (mismos datos que el export crudo).
		stress_rest = None
		perfusion_phase_rows = None
		try:
			_texture, perfusion_phase_rows = self._compute_perfusion_texture()
		except Exception as exc:
			self._log(f"Textura de perfusión no disponible para PDF: {exc}")
		try:
			stress_rest = self._stress_rest_for_reports(ef)
		except Exception as exc:
			self._log(f"Comparación stress-rest no disponible para PDF: {exc}")
		perfusion_quant = None
		try:
			perfusion_quant = self._compute_perfusion_quant()
		except Exception as exc:
			self._log(f"Cuantificación de perfusión no disponible: {exc}")
		try:
			self._ensure_pdf_extra_images()
		except Exception as exc:
			self._log(f"[WARN] Imágenes extra del PDF no generadas: {exc}")
		try:
			generate_report(
				output_pdf=pdf_path,
				output_dir=self.output_dir,
				study=self.study,
				seg=self.seg,
				metrics=report_metrics,
				territory=self.territory,
				processing_params=params,
				volumes=vol,
				ef=ef,
				stress_rest=stress_rest,
				perfusion_phase_rows=perfusion_phase_rows,
				perfusion_quant=perfusion_quant,
			)
			self._log(f"PDF actualizado: {pdf_path}")
		except Exception as exc:
			self._log(f"[WARN] No se pudo generar PDF integrado: {exc}")

		# Generar informe HTML autocontenido.
		try:
			html_path = os.path.join(self.output_dir, "informe_sincro.html")
			_, exec_html_out, hash_entry = generate_html_report(
				output_html=html_path,
				output_dir=self.output_dir,
				study=self.study,
				seg=self.seg,
				metrics=report_metrics,
				territory=self.territory,
				processing_params=params,
				volumes=vol,
				ef=ef,
				stress_rest=stress_rest,
				perfusion_phase_rows=perfusion_phase_rows,
				perfusion_quant=perfusion_quant,
				editor_html=getattr(self, "_report_editor_html", ""),
				hash_max_files=int(self._ui_settings.value("integrity/hash_max_files", 200)) if hasattr(self, "_ui_settings") else 200,
				hash_max_days=int(self._ui_settings.value("integrity/hash_max_days", 90)) if hasattr(self, "_ui_settings") else 90,
			)
			self._cached_exec_html = exec_html_out or ""
			if hash_entry:
				self._log(f"Hash SHA-256 registrado: {hash_entry.get('sha256', '')[:16]}...")
			self._log(f"HTML actualizado: {html_path}")
		except Exception as exc:
			self._log(f"[WARN] No se pudo generar HTML integrado: {exc}")

	def _ensure_reports_generated(self):
		if self.study is None or self.seg is None or self.metrics is None or self.territory is None:
			QMessageBox.information(self, "SINCRO", "Primero procesá un estudio para generar informes.")
			return False
		self._set_progress(92, "Generando informes (PDF + HTML)...")
		self._generate_pdf_report()
		self._set_progress(100, "Informes listos")
		return True

	def _apply_preview_zoom(self, name: str, fast: bool = False):
		label = self.preview_labels[name]
		scroller = self._preview_scrollers.get(name)
		anchor = None
		if scroller is not None:
			hb = scroller.horizontalScrollBar()
			vb = scroller.verticalScrollBar()
			if hb is not None and vb is not None:
				anchor = (hb.value(), vb.value())
		movie = self.preview_movies.get(name)
		if movie is not None and movie.isValid():
			base_size = self.preview_base_sizes.get(name)
			if base_size is None or base_size.isEmpty():
				base_size = movie.currentPixmap().size()
				if base_size.isEmpty():
					base_size = movie.frameRect().size()
				if base_size.isEmpty():
					base_size = QSize(500, 320)
				self.preview_base_sizes[name] = base_size
			zoom = max(0.20, min(5.00, self.preview_zoom.get(name, 1.0)))
			w = max(1, int(base_size.width() * zoom))
			h = max(1, int(base_size.height() * zoom))
			movie.setScaledSize(QSize(w, h))
			label.setMinimumSize(w, h)
			label.resize(w, h)
			if name in self.preview_zoom_labels:
				self.preview_zoom_labels[name].setText(f"{int(zoom * 100)}%")
			self._restore_preview_anchor(name, anchor)
			return
		pix = self.preview_pixmaps.get(name)
		if pix is None or pix.isNull():
			label.setText("Sin imagen")
			return
		base_size = self.preview_base_sizes.get(name)
		if base_size is None or base_size.isEmpty():
			base_size = pix.size()
			self.preview_base_sizes[name] = base_size
		zoom = max(0.20, min(5.00, self.preview_zoom.get(name, 1.0)))
		w = max(1, int(base_size.width() * zoom))
		h = max(1, int(base_size.height() * zoom))
		mode = Qt.TransformationMode.FastTransformation if fast else Qt.TransformationMode.SmoothTransformation
		scaled = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, mode)
		label.setPixmap(scaled)
		label.setMinimumSize(scaled.size())
		label.resize(scaled.size())
		if name in self.preview_zoom_labels:
			self.preview_zoom_labels[name].setText(f"{int(zoom * 100)}%")
		self._restore_preview_anchor(name, anchor)

	def _restore_preview_anchor(self, name: str, anchor):
		if anchor is None:
			return
		scroller = self._preview_scrollers.get(name)
		if scroller is None:
			return
		hb = scroller.horizontalScrollBar()
		vb = scroller.verticalScrollBar()
		if hb is not None:
			hb.setValue(int(anchor[0]))
		if vb is not None:
			vb.setValue(int(anchor[1]))

	def _default_preview_zoom(self, name: str) -> float:
		"""Zoom inicial por pestaña. Cada vista arranca en el nivel que mejor la
		muestra; siempre se puede cambiar en vivo con +/- o el slider."""
		defaults = {
			"slices_fase": 0.5,
			"polar_combo": 0.3,
			"delta_combo": 0.5,
			"histograma": 0.7,
			"ungated": 0.6,
			"cine_crudo": 5.0,
			"polar_perfusion_directa": 0.8,
			"comparacion_ejes": 0.2,
			"panel_funcional_gated": 0.5,
			"bullseye_directo": 0.5,
			"guia_fase_vi": 0.5,
		}
		return defaults.get(str(name), 0.5)

	def _zoom_preview(self, name: str, delta: float):
		current = self.preview_zoom.get(name, 1.0)
		self._set_preview_zoom(name, current + delta)

	def _set_preview_zoom(self, name: str, value: float):
		self.preview_zoom[name] = max(0.20, min(5.00, float(value)))
		self._apply_preview_zoom(name)

	def _build_polar_screen_color_column(self) -> QWidget:
		"""Columna de color pegada al preview del mapa polar de perfusión: solo cmap
		+ tira vertical del LUT (SIN RangeSlider). Afecta únicamente la vista en
		pantalla; el color del informe se configura aparte en 'Escalas informe'."""
		col = QWidget()
		col.setMaximumWidth(132)
		v = QVBoxLayout(col)
		v.setContentsMargins(4, 4, 4, 4)
		v.setSpacing(4)
		v.addWidget(QLabel("Escala"))
		self.polar_screen_cmap_combo = QComboBox()
		self.polar_screen_cmap_combo.addItems(self._all_cmaps)
		self.polar_screen_cmap_combo.setCurrentText(self.polar_perf_screen_cmap)
		self.polar_perf_screen_cmap = str(self.polar_screen_cmap_combo.currentText())
		self.polar_screen_cmap_combo.setToolTip("Escala de colores del mapa polar en pantalla (no afecta el informe).")
		self.polar_screen_cmap_combo.currentTextChanged.connect(self._on_polar_screen_cmap_changed)
		v.addWidget(self.polar_screen_cmap_combo)
		self.polar_screen_color_strip = VerticalColorStrip(self.polar_perf_screen_cmap)
		v.addWidget(self.polar_screen_color_strip, 1)
		return col

	def _on_polar_screen_cmap_changed(self, name):
		self.polar_perf_screen_cmap = str(name)
		strip = getattr(self, "polar_screen_color_strip", None)
		if strip is not None:
			strip.set_cmap(self.polar_perf_screen_cmap)
		if self.polar_view_mode == "cine":
			self._rebuild_polar_cine_frames_screen()
		else:
			self._rerender_polar_perfusion_screen()

	def _draw_polar_guides(self, ax, canvas_size: int):
		import matplotlib.pyplot as plt

		perf_grid = "#7f8a9a"
		perf_fg = "#f3f4f6"
		perf_subtle = "#9ca3af"
		c = canvas_size * 0.5
		r = canvas_size * 0.5
		for frac in (0.25, 0.50, 0.75, 1.0):
			ax.add_patch(plt.Circle((c, c), radius=r * frac, fill=False, color=perf_grid, linewidth=0.8, alpha=0.75))
		ax.plot([c - r, c + r], [c, c], color=perf_grid, linewidth=0.8, alpha=0.8)
		ax.plot([c, c], [c - r, c + r], color=perf_grid, linewidth=0.8, alpha=0.8)
		ax.text(c, c - r * 1.03, "ANT", ha="center", va="bottom", color=perf_fg, fontsize=8, fontweight="bold")
		ax.text(c + r * 1.03, c, "LAT", ha="left", va="center", color=perf_fg, fontsize=8, fontweight="bold")
		ax.text(c, c + r * 1.03, "INF", ha="center", va="top", color=perf_fg, fontsize=8, fontweight="bold")
		ax.text(c - r * 1.03, c, "SEP", ha="right", va="center", color=perf_fg, fontsize=8, fontweight="bold")
		ax.text(c, c, "APEX", ha="center", va="center", color=perf_fg, fontsize=7, fontweight="bold")
		ax.text(c, c + r * 0.98, "BASE", ha="center", va="top", color=perf_subtle, fontsize=7, fontweight="bold")

	def _polar_pm_to_cartesian(self, pm: np.ndarray, size: int = 480) -> np.ndarray:
		pm = np.asarray(pm, dtype=np.float64)
		canvas = np.full((size, size), np.nan, dtype=np.float64)
		yy, xx = np.indices((size, size), dtype=np.float64)
		cxp = (size - 1) / 2.0
		cyp = (size - 1) / 2.0
		xn = (xx - cxp) / max(1.0, cxp)
		yn = (yy - cyp) / max(1.0, cyp)
		rr = np.sqrt(xn**2 + yn**2)
		inside = rr <= 1.0
		ang = (np.degrees(np.arctan2(yn, xn)) + 360.0) % 360.0
		ri = np.clip((rr * (pm.shape[0] - 1)).astype(np.int32), 0, pm.shape[0] - 1)
		ti = np.clip(np.floor(ang).astype(np.int32), 0, pm.shape[1] - 1)
		canvas[inside] = pm[ri[inside], ti[inside]]
		return canvas

	def _render_polar_cart_panel(self, cart: np.ndarray, title: str, cmap_name: str) -> np.ndarray:
		import matplotlib.pyplot as plt

		perf_bg = "#000000"
		perf_fg = "#f3f4f6"
		fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.2), facecolor=perf_bg)
		ax.set_facecolor(perf_bg)
		ax.set_aspect("equal")
		ax.set_xticks([])
		ax.set_yticks([])
		ax.imshow(cart, cmap=cmap_name, vmin=0.0, vmax=1.0)
		self._draw_polar_guides(ax, int(cart.shape[0]))
		ax.set_title(title, color=perf_fg, fontsize=10, fontweight="bold")
		fig.tight_layout()
		fig.canvas.draw()
		w, h = fig.canvas.get_width_height()
		buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
		plt.close(fig)
		return buf

	def _rebuild_polar_cine_frames_screen(self) -> bool:
		"""Reconstruye los frames del cine polar EN MEMORIA con el cmap de pantalla
		(self.polar_perf_screen_cmap), desde la cache de mapas polares por gate. No
		toca el GIF/montaje de disco (el informe conserva su propio cmap)."""
		cache = getattr(self, "_polar_cine_cart_cache", None)
		if not cache or not cache.get("frames"):
			return False
		if "polar_perfusion_directa" not in self.preview_labels:
			return False
		try:
			cmap_name = str(getattr(self, "polar_perf_screen_cmap", "odyssey_cool") or "odyssey_cool")
			frames_out: list[QPixmap] = []
			for panels in cache["frames"]:
				bufs = []
				for pnl in panels:
					cart = self._polar_pm_to_cartesian(pnl["pm"])
					bufs.append(self._render_polar_cart_panel(cart, str(pnl.get("title", "")), cmap_name))
				if not bufs:
					continue
				if len(bufs) == 1:
					frame_rgb = bufs[0]
				else:
					gap = np.full((bufs[0].shape[0], 28, 3), 12, dtype=np.uint8)
					parts = []
					for i, b in enumerate(bufs):
						if i > 0:
							parts.append(gap)
						parts.append(b)
					frame_rgb = np.concatenate(parts, axis=1)
				frames_out.append(self._rgb_frame_to_qpixmap(frame_rgb))
			if not frames_out:
				return False
			self.polar_cine_preview_frames = frames_out
			self.polar_cine_preview_index = 0
			self.polar_cine_timer.setInterval(max(40, int(self.polar_cine_speed_spin.value())))
			self._set_polar_cine_memory_frame(0)
			self._update_polar_cine_toggle_text(enabled=True)
			return True
		except Exception:
			return False

	def _rerender_polar_perfusion_screen(self) -> bool:
		"""Recolorea en memoria el mapa polar de perfusión (vista estática) usando
		self.polar_perf_screen_cmap, desde la caché de mapas cartesianos. No escribe
		en disco, así que el PNG del informe (con su propio cmap) queda intacto."""
		cache = getattr(self, "_polar_perf_cart_cache", None)
		if not cache or "polar_perfusion_directa" not in self.preview_labels:
			return False
		if self.polar_view_mode != "perfusion":
			return False
		try:
			import matplotlib.pyplot as plt

			cmap_name = str(getattr(self, "polar_perf_screen_cmap", "odyssey_cool") or "odyssey_cool")
			cart_raw = cache["raw"]
			cart_smooth = cache["smooth"]
			label = str(cache.get("label", ""))
			rotation_deg = int(cache.get("rotation_deg", 0))
			smooth_desc = str(cache.get("smooth_desc", ""))
			perf_bg = "#000000"
			perf_fg = "#f3f4f6"
			perf_subtle = "#9ca3af"

			fig_pp, axes_pp = plt.subplots(1, 2, figsize=(12.0, 6.0), facecolor=perf_bg)
			for ax_pp, img_pp, ttl in [
				(axes_pp[0], cart_raw, "Perfusión polar directa (crudo)"),
				(axes_pp[1], cart_smooth, f"Perfusión polar directa ({smooth_desc})"),
			]:
				ax_pp.set_facecolor(perf_bg)
				ax_pp.set_aspect("equal")
				ax_pp.set_xticks([])
				ax_pp.set_yticks([])
				ax_pp.imshow(img_pp, cmap=cmap_name, vmin=0.0, vmax=1.0)
				self._draw_polar_guides(ax_pp, int(img_pp.shape[0]))
				ax_pp.set_title(ttl, color=perf_fg, fontsize=10, fontweight="bold")
			fig_pp.suptitle(f"Mapa polar de perfusión (apex en centro, base en borde) — {label} — rotación {rotation_deg:+d}°", color=perf_fg, fontsize=11.5, fontweight="bold")
			fig_pp.text(0.5, 0.02, "Reconstrucción polar continua desde short-axis: 'aplastado' apex->base", ha="center", color=perf_subtle, fontsize=8.6)
			fig_pp.canvas.draw()
			w, h = fig_pp.canvas.get_width_height()
			buf = np.frombuffer(fig_pp.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
			plt.close(fig_pp)
			pix = self._rgb_frame_to_qpixmap(buf)
			self.preview_pixmaps["polar_perfusion_directa"] = pix
			self.preview_base_sizes["polar_perfusion_directa"] = pix.size()
			self._apply_preview_zoom("polar_perfusion_directa")
			return True
		except Exception:
			return False

	def _set_polar_view_mode(self, mode: str):
		mode = "cine" if mode == "cine" else "perfusion"
		self.polar_view_mode = mode
		if self.polar_perf_view_perf_btn is not None:
			self.polar_perf_view_perf_btn.setChecked(mode == "perfusion")
		if self.polar_perf_view_cine_btn is not None:
			self.polar_perf_view_cine_btn.setChecked(mode == "cine")
		if mode == "perfusion":
			self.polar_cine_timer.stop()
			self.polar_cine_playing = False
			self._update_polar_cine_toggle_text(enabled=bool(self.polar_cine_preview_frames))
		self._load_preview("polar_perfusion_directa")

	def _toggle_polar_cine_preview(self):
		if self.polar_view_mode != "cine":
			self._set_polar_view_mode("cine")
		if not self.polar_cine_preview_frames:
			self._update_polar_cine_toggle_text(enabled=False)
			return
		self.polar_cine_playing = not self.polar_cine_playing
		if self.polar_cine_playing:
			self.polar_cine_timer.start()
		else:
			self.polar_cine_timer.stop()
		self._update_polar_cine_toggle_text(enabled=True)

	def _restart_polar_cine_preview(self):
		if self.polar_view_mode != "cine":
			self._set_polar_view_mode("cine")
		if not self.polar_cine_preview_frames:
			self._update_polar_cine_toggle_text(enabled=False)
			return
		self.polar_cine_preview_index = 0
		self._set_polar_cine_memory_frame(0)
		self._update_polar_cine_toggle_text(enabled=True)

	def _toggle_compare_axes_preview(self):
		if not self.compare_axes_preview_frames:
			self._update_compare_axes_toggle_text(enabled=False)
			return
		self.compare_axes_playing = not self.compare_axes_playing
		if self.compare_axes_playing:
			self.compare_axes_cine_timer.setInterval(max(40, int(self.compare_axes_cine_speed_spin.value())))
			self.compare_axes_cine_timer.start()
		else:
			self.compare_axes_cine_timer.stop()
		self._update_compare_axes_toggle_text(enabled=True)

	def _restart_compare_axes_preview(self):
		if not self.compare_axes_preview_frames:
			self._update_compare_axes_toggle_text(enabled=False)
			return
		self.compare_axes_preview_index = 0
		self._set_compare_axes_memory_frame(0)
		self._update_compare_axes_toggle_text(enabled=True)

	def _update_polar_cine_toggle_text(self, enabled: bool = True):
		if self.polar_cine_toggle_btn is None:
			return
		self.polar_cine_toggle_btn.setEnabled(enabled)
		if not enabled:
			self.polar_cine_toggle_btn.setText("▶")
			return
		if not self.polar_cine_preview_frames:
			self.polar_cine_toggle_btn.setText("▶")
			return
		if self.polar_cine_playing:
			self.polar_cine_toggle_btn.setText("⏸")
			self.polar_cine_toggle_btn.setToolTip("Pausar")
		else:
			self.polar_cine_toggle_btn.setText("▶")
			self.polar_cine_toggle_btn.setToolTip("Reproducir")

	def _set_polar_cine_memory_frame(self, index: int):
		if not self.polar_cine_preview_frames:
			return
		idx = max(0, min(int(index), len(self.polar_cine_preview_frames) - 1))
		self.polar_cine_preview_index = idx
		pix = self.polar_cine_preview_frames[idx]
		self.preview_pixmaps["polar_perfusion_directa"] = pix
		self.preview_base_sizes["polar_perfusion_directa"] = pix.size()
		self._apply_preview_zoom("polar_perfusion_directa")

	def _advance_polar_cine_frame(self):
		if not self.polar_cine_preview_frames:
			self.polar_cine_timer.stop()
			self.polar_cine_playing = False
			self._update_polar_cine_toggle_text(enabled=False)
			return
		self.polar_cine_preview_index = (int(self.polar_cine_preview_index) + 1) % max(1, len(self.polar_cine_preview_frames))
		self._set_polar_cine_memory_frame(self.polar_cine_preview_index)

	def _on_polar_cine_speed_changed(self, value: int):
		self.polar_cine_timer.setInterval(max(40, int(value)))

	# --- Cine crudo (proyecciones SPECT) ---
	def _rgb_frame_to_qpixmap_raw(self, rgb: np.ndarray) -> QPixmap:
		h, w = rgb.shape[:2]
		if rgb.ndim == 2:
			qimg = QImage(rgb.data, w, h, w, QImage.Format.Format_Grayscale8)
		else:
			qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
		return QPixmap.fromImage(qimg.copy())

	def _cine_crudo_band_bounds(self, H: int) -> tuple[float, float] | None:
		if self.cine_crudo_seed is None:
			return None
		if self.cine_crudo_band_upper is not None and self.cine_crudo_band_lower is not None:
			y0 = float(self.cine_crudo_band_upper)
			y1 = float(self.cine_crudo_band_lower)
		else:
			r = float(self.cine_crudo_roi_spin.value()) if hasattr(self, "cine_crudo_roi_spin") else 0.0
			y0 = float(self.cine_crudo_seed[0]) - r
			y1 = float(self.cine_crudo_seed[0]) + r
		y0, y1 = sorted((y0, y1))
		return float(np.clip(y0, 0, H - 1)), float(np.clip(y1, 0, H - 1))

	def _cine_crudo_event_to_matrix(self, event):
		label = self.preview_labels.get("cine_crudo")
		pix = self.preview_pixmaps.get("cine_crudo")
		if label is None or pix is None or pix.isNull() or self.study is None:
			return None
		projections = np.asarray(self.study.cube, dtype=np.float64)
		H, W = int(projections.shape[2]), int(projections.shape[3])
		lw, lh = label.width(), label.height()
		pw, ph = pix.width(), pix.height()
		shown = label.pixmap()
		if shown is not None and not shown.isNull():
			lw, lh = shown.width(), shown.height()
		scale = min(lw / max(1, pw), lh / max(1, ph))
		dw, dh = pw * scale, ph * scale
		x0 = (label.width() - dw) / 2.0
		y0 = (label.height() - dh) / 2.0
		rx = (event.pos().x() - x0) / max(1e-6, scale)
		ry = (event.pos().y() - y0) / max(1e-6, scale)
		return float(np.clip(ry, 0, H - 1)), float(np.clip(rx, 0, W - 1)), H, W, float(rx)

	def _cine_crudo_marker_at_event(self, event) -> str | None:
		pos = self._cine_crudo_event_to_matrix(event)
		if pos is None:
			return None
		ry0, _rx0, H_map, W_map, rx_raw = pos
		show_ref = bool(hasattr(self, "cine_crudo_compare_line_check") and self.cine_crudo_compare_line_check.isChecked())
		line_y = self.cine_crudo_compare_line_y
		if line_y is None:
			line_y = float(self.cine_crudo_seed[0]) if self.cine_crudo_seed is not None else 0.5 * float(H_map - 1)
		if not self.cine_crudo_seed_mode and show_ref and rx_raw >= float(W_map):
			if abs(ry0 - float(line_y)) <= 6.0:
				return "compare_line"
		roi_mode = str(self.cine_crudo_roi_mode_combo.currentText()).lower() if hasattr(self, "cine_crudo_roi_mode_combo") else "caja"
		bounds = self._cine_crudo_band_bounds(H_map) if "banda" in roi_mode else None
		if not self.cine_crudo_seed_mode and bounds is not None and rx_raw < float(W_map):
			yu, yl = bounds
			if abs(ry0 - yu) <= 5.0:
				return "upper"
			if abs(ry0 - yl) <= 5.0:
				return "lower"
		if not self.cine_crudo_seed_mode and show_ref:
			if abs(ry0 - float(line_y)) <= 6.0:
				return "compare_line"
		return None

	def _cine_crudo_set_drag_status(self, marker: str | None):
		label = marker or "—"
		if marker == "compare_line":
			label = "ref"
		self._cine_crudo_hover_marker = marker
		preview = self.preview_labels.get("cine_crudo") if hasattr(self, "preview_labels") else None
		if preview is not None:
			preview.setCursor(QCursor(Qt.CursorShape.SizeVerCursor if marker else Qt.CursorShape.ArrowCursor))

	def _collect_cine_crudo_visual_config(self) -> dict:
		return {
			"version": 1,
			"method": str(self.cine_crudo_method_combo.currentText()) if hasattr(self, "cine_crudo_method_combo") else "Sinusoide",
			"axis": str(self.cine_crudo_axis_combo.currentText()) if hasattr(self, "cine_crudo_axis_combo") else "Y",
			"threshold": float(self._cine_crudo_threshold_value()),
			"roi_mode": str(self.cine_crudo_roi_mode_combo.currentText()) if hasattr(self, "cine_crudo_roi_mode_combo") else "Caja",
			"roi_radius": int(self.cine_crudo_roi_spin.value()) if hasattr(self, "cine_crudo_roi_spin") else 12,
			"marker_color": str(self.cine_crudo_marker_color_combo.currentText()) if hasattr(self, "cine_crudo_marker_color_combo") else "Arena",
			"band_upper": None if self.cine_crudo_band_upper is None else float(self.cine_crudo_band_upper),
			"band_lower": None if self.cine_crudo_band_lower is None else float(self.cine_crudo_band_lower),
			"reference_line_y": None if self.cine_crudo_compare_line_y is None else float(self.cine_crudo_compare_line_y),
			"show_reference_line": bool(hasattr(self, "cine_crudo_compare_line_check") and self.cine_crudo_compare_line_check.isChecked()),
			"show_sinogram": bool(hasattr(self, "cine_crudo_sino_check") and self.cine_crudo_sino_check.isChecked()),
			"sinogram_axis": str(self.cine_crudo_sino_axis_combo.currentText()) if hasattr(self, "cine_crudo_sino_axis_combo") else "Sinograma Y",
			"show_mask": bool(self.cine_crudo_mask_check is not None and self.cine_crudo_mask_check.isChecked()),
			"liver_suppression_enabled": bool(hasattr(self, "cine_crudo_liver_suppress_check") and self.cine_crudo_liver_suppress_check.isChecked()),
			"liver_suppression_pct": int(self.cine_crudo_liver_suppress_spin.value()) if hasattr(self, "cine_crudo_liver_suppress_spin") else 60,
		}

	def _apply_cine_crudo_visual_config(self, cfg: dict, refresh: bool = True):
		if hasattr(self, "cine_crudo_method_combo") and "method" in cfg:
			self.cine_crudo_method_combo.setCurrentText(str(cfg["method"]))
		if hasattr(self, "cine_crudo_axis_combo") and "axis" in cfg:
			self.cine_crudo_axis_combo.setCurrentText(str(cfg["axis"]))
		if "threshold" in cfg and hasattr(self, "cine_crudo_threshold_slider"):
			thr = int(round(float(cfg["threshold"]) * 100.0))
			self.cine_crudo_threshold_slider.setValue(int(np.clip(thr, 1, 100)))
		if hasattr(self, "cine_crudo_roi_mode_combo") and "roi_mode" in cfg:
			self.cine_crudo_roi_mode_combo.setCurrentText(str(cfg["roi_mode"]))
		if hasattr(self, "cine_crudo_roi_spin") and "roi_radius" in cfg:
			self.cine_crudo_roi_spin.setValue(int(np.clip(int(cfg["roi_radius"]), self.cine_crudo_roi_spin.minimum(), self.cine_crudo_roi_spin.maximum())))
		if hasattr(self, "cine_crudo_marker_color_combo") and "marker_color" in cfg:
			self.cine_crudo_marker_color_combo.setCurrentText(str(cfg["marker_color"]))
		self.cine_crudo_band_upper = None if cfg.get("band_upper") is None else float(cfg.get("band_upper"))
		self.cine_crudo_band_lower = None if cfg.get("band_lower") is None else float(cfg.get("band_lower"))
		self.cine_crudo_compare_line_y = None if cfg.get("reference_line_y") is None else float(cfg.get("reference_line_y"))
		if hasattr(self, "cine_crudo_compare_line_check") and "show_reference_line" in cfg:
			self.cine_crudo_compare_line_check.setChecked(bool(cfg["show_reference_line"]))
		if hasattr(self, "cine_crudo_sino_check") and "show_sinogram" in cfg:
			self.cine_crudo_sino_check.setChecked(bool(cfg["show_sinogram"]))
		if hasattr(self, "cine_crudo_sino_axis_combo") and "sinogram_axis" in cfg:
			self.cine_crudo_sino_axis_combo.setCurrentText(str(cfg["sinogram_axis"]))
		if self.cine_crudo_mask_check is not None and "show_mask" in cfg:
			self.cine_crudo_mask_check.setChecked(bool(cfg["show_mask"]))
		if hasattr(self, "cine_crudo_liver_suppress_check") and "liver_suppression_enabled" in cfg:
			self.cine_crudo_liver_suppress_check.setChecked(bool(cfg["liver_suppression_enabled"]))
		if hasattr(self, "cine_crudo_liver_suppress_spin") and "liver_suppression_pct" in cfg:
			self.cine_crudo_liver_suppress_spin.setValue(int(np.clip(int(cfg["liver_suppression_pct"]), self.cine_crudo_liver_suppress_spin.minimum(), self.cine_crudo_liver_suppress_spin.maximum())))
		if refresh:
			self._refresh_cine_crudo_view()

	def _save_cine_crudo_visual_config(self):
		try:
			default_path = os.path.join(self.presets_dir, "cine_crudo_visual_config.json")
			path, _flt = QFileDialog.getSaveFileName(
				self, "Guardar configuración visual cine_crudo", default_path,
				"Configuración JSON (*.json);;Todos los archivos (*.*)",
			)
			if not path:
				return
			base, ext = os.path.splitext(path)
			if ext.lower() != ".json":
				path = base + ".json"
			payload = self._collect_cine_crudo_visual_config()
			payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
			payload["study_uid"] = str(getattr(self.study, "study_instance_uid", "") or "") if self.study is not None else ""
			payload["patient_id"] = str(getattr(self.study, "patient_id", "") or "") if self.study is not None else ""
			with open(path, "wb") as fh:
				fh.write((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
			self._log(f"Configuración visual cine_crudo guardada: {os.path.basename(path)}")
			self.statusBar().showMessage("Configuración visual cine_crudo guardada")
		except Exception as exc:
			self._log(f"[WARN] Guardar configuración visual falló: {exc}")
			QMessageBox.warning(self, "SINCRO", f"No se pudo guardar la configuración visual:\n{exc}")

	def _load_cine_crudo_visual_config(self):
		try:
			path, _flt = QFileDialog.getOpenFileName(
				self, "Cargar configuración visual cine_crudo", self.presets_dir,
				"Configuración JSON (*.json);;Todos los archivos (*.*)",
			)
			if not path:
				return
			with open(path, "rb") as fh:
				cfg = json.loads(fh.read().decode("utf-8", errors="replace"))
			if not isinstance(cfg, dict):
				raise ValueError("El archivo no contiene un objeto JSON válido.")
			self._apply_cine_crudo_visual_config(cfg, refresh=True)
			self._log(f"Configuración visual cine_crudo cargada: {os.path.basename(path)}")
			self.statusBar().showMessage("Configuración visual cine_crudo cargada")
		except Exception as exc:
			self._log(f"[WARN] Cargar configuración visual falló: {exc}")
			QMessageBox.warning(self, "SINCRO", f"No se pudo cargar la configuración visual:\n{exc}")

	def _cine_crudo_marker_color(self) -> np.ndarray:
		name = str(self.cine_crudo_marker_color_combo.currentText()).lower() if hasattr(self, "cine_crudo_marker_color_combo") else "arena"
		colors = {
			"cian": (80, 220, 235),
			"blanco": (245, 245, 235),
			"negro": (15, 18, 20),
			"magenta": (230, 90, 190),
			"verde": (95, 210, 130),
			"arena": (188, 174, 122),
		}
		return np.array(colors.get(name, colors["arena"]), dtype=np.float64)

	def _draw_cine_crudo_reference_line(self, rgb: np.ndarray, H: int) -> np.ndarray:
		show_line = bool(hasattr(self, "cine_crudo_compare_line_check") and self.cine_crudo_compare_line_check.isChecked())
		if not show_line:
			return rgb
		if self.cine_crudo_compare_line_y is None:
			if self.cine_crudo_seed is not None:
				self.cine_crudo_compare_line_y = float(self.cine_crudo_seed[0])
			else:
				self.cine_crudo_compare_line_y = 0.5 * float(H - 1)
		y = int(np.clip(round(float(self.cine_crudo_compare_line_y)), 0, max(0, rgb.shape[0] - 1)))
		marker = self._cine_crudo_marker_color()
		out = rgb.copy()
		out[y, :, :] = (0.18 * out[y, :, :].astype(np.float64) + 0.82 * marker).astype(np.uint8)
		if y + 1 < out.shape[0]:
			out[y + 1, :, :] = (0.58 * out[y + 1, :, :].astype(np.float64) + 0.42 * marker).astype(np.uint8)
		return out

	def _append_cine_crudo_sinogram_panel(self, rgb: np.ndarray, frames_arr: np.ndarray, corrected_frames_arr: np.ndarray | None, frame_idx: int) -> np.ndarray:
		show_sino = bool(hasattr(self, "cine_crudo_sino_check") and self.cine_crudo_sino_check.isChecked())
		if not show_sino:
			return rgb
		try:
			import matplotlib.cm as _cm
			cmap = _cm.get_cmap("gray")
			sino_axis = str(self.cine_crudo_sino_axis_combo.currentText()).lower() if hasattr(self, "cine_crudo_sino_axis_combo") else "sinograma y"
			profile_axis = 1 if "x" in sino_axis else 2
			profiles = [np.asarray(frames_arr, dtype=np.float64).sum(axis=profile_axis)]  # (ang, Y) o (ang, X)
			if corrected_frames_arr is not None:
				profiles.append(np.asarray(corrected_frames_arr, dtype=np.float64).sum(axis=profile_axis))
			p99 = max(float(np.percentile(p, 99.0)) or 1.0 for p in profiles)
			panels = []
			marker = self._cine_crudo_marker_color().astype(np.uint8)
			for prof in profiles:
				img = np.clip(prof / max(1e-6, p99), 0, 1)
				panel = (np.asarray(cmap(img)[..., :3]) * 255).astype(np.uint8)  # vertical: ang rows × Y cols
				row = int(np.clip(frame_idx, 0, panel.shape[0] - 1))
				panel[row, :, :] = marker
				if profile_axis == 2:
					bounds = self._cine_crudo_band_bounds(int(frames_arr.shape[1]))
					if bounds is not None:
						for yy in bounds:
							col = int(np.clip(round(float(yy)), 0, panel.shape[1] - 1))
							panel[:, col, :] = (0.28 * panel[:, col, :].astype(np.float64) + 0.72 * marker).astype(np.uint8)
					if self.cine_crudo_compare_line_y is not None and bool(hasattr(self, "cine_crudo_compare_line_check") and self.cine_crudo_compare_line_check.isChecked()):
						col = int(np.clip(round(float(self.cine_crudo_compare_line_y)), 0, panel.shape[1] - 1))
						panel[:, col, :] = (0.16 * panel[:, col, :].astype(np.float64) + 0.84 * marker).astype(np.uint8)
				panels.append(panel)
			sino = np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0]
			gap = np.full((rgb.shape[0], 4, 3), 18, dtype=np.uint8)
			if sino.shape[0] != rgb.shape[0]:
				rep = max(1, int(np.ceil(rgb.shape[0] / max(1, sino.shape[0]))))
				sino = np.repeat(sino, rep, axis=0)[:rgb.shape[0]]
			return np.concatenate([rgb, gap, sino], axis=1)
		except Exception:
			return rgb

	def _scale_seed_for_study(self, target_study):
		"""Escala el seed seleccionado sobre el estudio principal al tamaño del estudio target."""
		if self.cine_crudo_seed is None or self.study is None or target_study is None:
			return self.cine_crudo_seed
		try:
			src_cube = np.asarray(self.study.cube, dtype=np.float64)
			dst_cube = np.asarray(target_study.cube, dtype=np.float64)
			src_h, src_w = int(src_cube.shape[2]), int(src_cube.shape[3])
			dst_h, dst_w = int(dst_cube.shape[2]), int(dst_cube.shape[3])
			sy, sx = float(self.cine_crudo_seed[0]), float(self.cine_crudo_seed[1])
			sy2 = sy * (float(dst_h) / max(1.0, float(src_h)))
			sx2 = sx * (float(dst_w) / max(1.0, float(src_w)))
			return float(np.clip(sy2, 0.0, max(0, dst_h - 1))), float(np.clip(sx2, 0.0, max(0, dst_w - 1)))
		except Exception:
			return self.cine_crudo_seed

	def _build_cine_crudo_frames_for_study(self, study_obj, corrected_projections, source: str, stage_label: str, seed_stage=None):
		"""Genera frames QPixmap de un estudio crudo para el preview superior."""
		projections = np.asarray(study_obj.cube, dtype=np.float64)  # (gates, angles, H, W)
		n_gates, n_angles, H, W = projections.shape
		# Preview rápido EM−SC: si el checkbox "Desc. SC" está activado y el
		# estudio trajo scatter, restar k×SC de las proyecciones SOLO para el
		# display del cine (no altera los datos ni la recon). Si está OFF, se ve
		# el crudo EM solo. El SC no gatillado se reparte como SC/N_gates.
		_sc_on = bool(getattr(self, "cine_crudo_scatter_check", None) is not None
					and self.cine_crudo_scatter_check.isEnabled()
					and self.cine_crudo_scatter_check.isChecked())
		_sc = getattr(study_obj, "scatter_projections", None)
		if _sc_on and _sc is not None:
			_k = float(self.cine_crudo_scatter_k_spin.value()) if getattr(self, "cine_crudo_scatter_k_spin", None) is not None else 1.0
			_sc_arr = np.asarray(_sc, dtype=np.float64)
			if _sc_arr.shape[0] == 1 and n_gates >= 2:
				_sc_arr = np.repeat(_sc_arr / float(n_gates), n_gates, axis=0)
			if _sc_arr.shape == projections.shape:
				projections = np.clip(projections - _k * _sc_arr, 0.0, None)
				matrix_txt_sc = f" [EM−{_k:.2f}×SC preview]"
			else:
				matrix_txt_sc = ""
		else:
			matrix_txt_sc = ""
		if source == "UngGat":
			from core.raw_projections import ungate_projections
			frames_arr = ungate_projections(projections)
			matrix_txt = f"{stage_label}: UngGat {n_angles}áng × {H}×{W}px (suma {n_gates} gates){matrix_txt_sc}"
		else:
			gate_mid = n_gates // 2
			frames_arr = projections[gate_mid]
			matrix_txt = f"{stage_label}: Gated gate {gate_mid + 1}/{n_gates} · {n_angles}áng × {H}×{W}px{matrix_txt_sc}"

		counts = frames_arr.sum(axis=(1, 2)).astype(np.float64) if frames_arr.ndim == 3 else np.zeros((frames_arr.shape[0],), dtype=np.float64)
		compare_on = bool(self.cine_crudo_compare_check is not None and self.cine_crudo_compare_check.isChecked())
		diff_on = bool(getattr(self, "cine_crudo_diff_check", None) is not None
					and self.cine_crudo_diff_check.isEnabled()
					and self.cine_crudo_diff_check.isChecked())
		corrected_frames_arr = None
		if (compare_on or diff_on) and corrected_projections is not None:
			corr = np.asarray(corrected_projections, dtype=np.float64)
			if source == "UngGat":
				from core.raw_projections import ungate_projections
				corrected_frames_arr = ungate_projections(corr)
			else:
				gate_mid = corr.shape[0] // 2
				corrected_frames_arr = corr[gate_mid]
			if compare_on and not diff_on:
				matrix_txt += " | original|corregido"
		# Vista de DIFERENCIA (corregido − original): mismo truco visual que el
		# preview de scatter EM−SC. Muestra QUÉ movió la corrección de movimiento.
		# La diferencia puede ser negativa (corregido < original) -> se centra en 0
		# con un colormap divergente y ventana simétrica.
		diff_frames_arr = None
		if diff_on and corrected_frames_arr is not None and corrected_frames_arr.shape == frames_arr.shape:
			diff_frames_arr = corrected_frames_arr - frames_arr
			matrix_txt += " [corregido−original]"

		p99 = float(np.percentile(frames_arr, 99.0)) or 1.0
		if corrected_frames_arr is not None:
			p99 = max(p99, float(np.percentile(corrected_frames_arr, 99.0)) or 1.0)
		if diff_frames_arr is not None:
			p99 = max(p99, float(np.percentile(np.abs(diff_frames_arr), 99.0)) or 1.0)
		if p99 <= 0:
			p99 = 1.0

		mask_on = bool(self.cine_crudo_mask_check is not None and self.cine_crudo_mask_check.isChecked())
		thr_val = self._cine_crudo_threshold_value()
		if seed_stage is None:
			seed_stage = self._scale_seed_for_study(study_obj)
		frames: list[QPixmap] = []
		# Motor de color del preview crudo: colormap + ventana 0..200% de p99.
		screen_cmap = str(getattr(self, "cine_crudo_screen_cmap", "gray") or "gray")
		win_lo = float(getattr(self, "cine_crudo_screen_win_low", 0.0)) / 100.0 * p99
		win_hi = float(getattr(self, "cine_crudo_screen_win_high", 100.0)) / 100.0 * p99
		win_den = (win_hi - win_lo) if (win_hi - win_lo) > 1e-9 else 1e-9
		try:
			import matplotlib
			try:
				cmap = matplotlib.colormaps[screen_cmap]
			except Exception:
				cmap = matplotlib.colormaps["gray"]
			for a in range(frames_arr.shape[0]):
				img = np.clip((frames_arr[a] - win_lo) / win_den, 0, 1)
				rgb = (np.asarray(cmap(img)[..., :3]) * 255).astype(np.uint8)
				if mask_on:
					frame_img = frames_arr[a]
					fmax = float(frame_img.max()) if frame_img.size else 0.0
					mask = frame_img > (thr_val * fmax) if fmax > 0 else np.zeros_like(frame_img, dtype=bool)
					method_now = str(self.cine_crudo_method_combo.currentText()).lower() if hasattr(self, "cine_crudo_method_combo") else "gammasync"
					if method_now == "gammasync":
						from core.raw_projections import _select_organ_component
						mask = _select_organ_component(mask, seed=seed_stage, auto=(seed_stage is None))
					rgb = rgb.copy()
					rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.array([255, 255, 255])).astype(np.uint8)

				roi_r = int(self.cine_crudo_roi_spin.value()) if hasattr(self, "cine_crudo_roi_spin") else 0
				roi_mode = str(self.cine_crudo_roi_mode_combo.currentText()).lower() if hasattr(self, "cine_crudo_roi_mode_combo") else "caja"
				if seed_stage is not None:
					rgb = rgb.copy()
					sy, sx = int(round(seed_stage[0])), int(round(seed_stage[1]))
					H0, W0 = rgb.shape[0], rgb.shape[1]
					marker = self._cine_crudo_marker_color()

					def _hline(yy, xa, xb):
						yy = int(np.clip(yy, 0, H0 - 1))
						xa = int(np.clip(xa, 0, W0 - 1)); xb = int(np.clip(xb, 0, W0 - 1))
						rgb[yy, xa:xb + 1] = (0.35 * rgb[yy, xa:xb + 1].astype(np.float64) + 0.65 * marker).astype(np.uint8)

					def _vline(xx, ya, yb):
						xx = int(np.clip(xx, 0, W0 - 1))
						ya = int(np.clip(ya, 0, H0 - 1)); yb = int(np.clip(yb, 0, H0 - 1))
						rgb[ya:yb + 1, xx] = (0.35 * rgb[ya:yb + 1, xx].astype(np.float64) + 0.65 * marker).astype(np.uint8)

					if roi_r > 0:
						if "banda" in roi_mode:
							bounds = self._cine_crudo_band_bounds(H0)
							if bounds is not None:
								y0, y1 = bounds
								_hline(y0, 0, W0 - 1)
								_hline(y1, 0, W0 - 1)
						else:
							y0, y1 = sy - roi_r, sy + roi_r
							x0, x1 = sx - roi_r, sx + roi_r
							_hline(y0, x0, x1); _hline(y1, x0, x1)
							_vline(x0, y0, y1); _vline(x1, y0, y1)
					_hline(sy, sx - 3, sx + 3)
					_vline(sx, sy - 3, sy + 3)

				if corrected_frames_arr is not None and not diff_on:
					img_corr = np.clip((corrected_frames_arr[a] - win_lo) / win_den, 0, 1)
					rgb_corr = (np.asarray(cmap(img_corr)[..., :3]) * 255).astype(np.uint8)
					rgb = np.concatenate([rgb, rgb_corr], axis=1)
				if diff_frames_arr is not None:
					# Diferencia corregido−original: colormap divergente centrado en 0.
					# Positivo (corregido > original) = un color; negativo = otro.
					img_diff = diff_frames_arr[a]
					abs99 = float(np.percentile(np.abs(img_diff), 99.0)) or 1.0
					if abs99 <= 0:
						abs99 = 1.0
					# Normalizar a [-1, 1] y mapear a [0, 1] para el colormap.
					norm = np.clip(img_diff / abs99, -1.0, 1.0)
					disp = (norm + 1.0) / 2.0
					try:
						diff_cmap = matplotlib.colormaps["RdBu_r"]
					except Exception:
						diff_cmap = matplotlib.colormaps["coolwarm"]
					rgb_diff = (np.asarray(diff_cmap(disp)[..., :3]) * 255).astype(np.uint8)
					rgb = np.concatenate([rgb, rgb_diff], axis=1)
				rgb = self._append_cine_crudo_sinogram_panel(rgb, frames_arr, corrected_frames_arr, a)
				rgb = self._draw_cine_crudo_reference_line(rgb, H)
				frames.append(self._rgb_frame_to_qpixmap_raw(rgb))
		except Exception:
			for a in range(frames_arr.shape[0]):
				img = (np.clip((frames_arr[a] - win_lo) / win_den, 0, 1) * 255).astype(np.uint8)
				if corrected_frames_arr is not None:
					img_corr = (np.clip((corrected_frames_arr[a] - win_lo) / win_den, 0, 1) * 255).astype(np.uint8)
					img = np.concatenate([img, img_corr], axis=1)
				frames.append(self._rgb_frame_to_qpixmap_raw(img))
		return frames, counts, matrix_txt

	def _stack_cine_crudo_dual_pixmaps(self, top_pix: QPixmap, bottom_pix: QPixmap, top_label: str, bottom_label: str, active_stage: str = "stress") -> QPixmap:
		"""Compone dos paneles en vertical con títulos: stress arriba / rest abajo. Resalta la etapa activa."""
		top_w, top_h = int(top_pix.width()), int(top_pix.height())
		bottom_w, bottom_h = int(bottom_pix.width()), int(bottom_pix.height())
		gap = 8
		fs = int(max(7, min(11, round(min(top_h, bottom_h) / 9.0))))
		bar_h = fs + 8
		w = max(top_w, bottom_w)
		split_y = bar_h + top_h + gap
		h = split_y + bar_h + bottom_h
		top_x = (w - top_w) // 2
		bottom_x = (w - bottom_w) // 2
		top_active = active_stage in ("stress", "both")
		bottom_active = active_stage in ("rest", "both")
		canvas = QPixmap(w, h)
		canvas.fill(QColor("#050912"))
		p = QPainter(canvas)
		f = p.font()
		f.setPointSize(fs)
		p.setFont(f)
		p.fillRect(0, 0, w, bar_h, QColor("#7f1d1d" if top_active else "#0f172a"))
		p.fillRect(0, split_y, w, bar_h, QColor("#7f1d1d" if bottom_active else "#0f172a"))
		p.setPen(QColor("#e2e8f0"))
		p.drawText(6, fs + 4, top_label)
		p.drawText(6, split_y + fs + 4, bottom_label)
		p.drawPixmap(top_x, bar_h, top_pix)
		p.drawPixmap(bottom_x, split_y + bar_h, bottom_pix)
		p.setPen(QPen(QColor("#ef4444"), 3))
		if top_active:
			p.drawRect(top_x + 1, bar_h + 1, max(1, top_w - 2), max(1, top_h - 2))
		if bottom_active:
			p.drawRect(bottom_x + 1, split_y + bar_h + 1, max(1, bottom_w - 2), max(1, bottom_h - 2))
		p.end()
		self._cine_crudo_dual_render_meta = {
			"enabled": True,
			"bar_h": int(bar_h),
			"gap": int(gap),
			"split_y": int(split_y),
			"top_h": int(top_h),
			"bottom_h": int(bottom_h),
			"top_w": int(top_w),
			"bottom_w": int(bottom_w),
			"canvas_w": int(w),
			"active_stage": active_stage if active_stage in ("stress", "rest", "both") else "stress",
		}
		return canvas

	def _set_active_cine_crudo_stage(self, stage: str, *, refresh_view: bool = True, force: bool = False):
		"""Selecciona qué etapa(s) (stress/rest/both) reciben las herramientas del crudo.

		``force=True``: no degradar a 'stress' aunque el detector de estudio
		secundario devuelva None (lo usa el orquestador dual, que ya validó las
		etapas al armar la lista).
		"""
		prev_stage = getattr(self, "_cine_crudo_active_stage", "stress")
		# Guardar los límites del stage saliente antes de cambiar (solo si los
		# spins realmente muestran esa etapa; evita contaminación cruzada).
		if prev_stage in ("stress", "rest"):
			self._cine_crudo_capture_limits_from_spins(prev_stage)
		if stage not in ("stress", "rest", "both"):
			stage = "stress"
		if not force and self._secondary_cine_crudo_study() is None:
			stage = "stress"
		changed = stage != prev_stage
		self._cine_crudo_active_stage = stage
		if stage in ("stress", "rest"):
			self._cine_crudo_recon_stage = stage
			# Cargar en UI los límites propios de la etapa entrante.
			self._cine_crudo_apply_stage_limits_to_spins(stage)
		if hasattr(self, "cine_crudo_stage_combo") and self.cine_crudo_stage_combo is not None:
			label = {"stress": "Esfuerzo", "rest": "Reposo", "both": "Ambas"}[stage]
			if self.cine_crudo_stage_combo.currentText() != label:
				self.cine_crudo_stage_combo.blockSignals(True)
				self.cine_crudo_stage_combo.setCurrentText(label)
				self.cine_crudo_stage_combo.blockSignals(False)
		if changed:
			etapa = {"stress": "Esfuerzo", "rest": "Reposo", "both": "Ambas (esfuerzo + reposo)"}[stage]
			self._log(f"Etapa activa del crudo: {etapa} — las herramientas actúan sobre esa(s) etapa(s).")
		self._update_patient_banner()
		if refresh_view:
			# Si estamos en la pantalla de límites (Base/Ápex), NO volver al cine:
			# re-renderizar la misma pantalla con la nueva etapa activa.
			if getattr(self, "cine_crudo_preview_mode", None) == "cut_limits":
				if stage in ("stress", "rest") and self._dual_session().stage(stage).recon_result is not None:
					self._cine_crudo_recon_stage = stage
				self._preview_cine_crudo_cut_limits()
			else:
				self._refresh_cine_crudo_view()

	def _on_cine_crudo_stage_combo_changed(self, text: str):
		t = str(text).lower()
		if "amb" in t:
			stage = "both"
		elif "repos" in t or "rest" in t:
			stage = "rest"
		else:
			stage = "stress"
		# Cambiar explícitamente el selector es una intención clínica: habilita
		# temporalmente procesar una sola etapa aun con dual-auto encendido.
		self._dual_pipeline_manual_stage_override = None if stage == "both" else stage
		self._set_active_cine_crudo_stage(stage)

	def _cine_crudo_process_stress(self) -> bool:
		return getattr(self, "_cine_crudo_active_stage", "stress") in ("stress", "both")

	def _cine_crudo_process_rest(self) -> bool:
		return getattr(self, "_cine_crudo_active_stage", "stress") in ("rest", "both") and self._secondary_cine_crudo_study() is not None

	def _cine_crudo_target_stages(self) -> list[str]:
		"""Etapas efectivas para ejecutar una herramienta del crudo.

		Si el selector está en "Ambas" y existe estudio secundario, se procesa en
		orden clínico Esfuerzo → Reposo. Si no hay segunda etapa, cae a Esfuerzo.
		"""
		active = getattr(self, "_cine_crudo_active_stage", "stress")
		has_dual = self._secondary_cine_crudo_study() is not None
		manual = getattr(self, "_dual_pipeline_manual_stage_override", None)
		# Regla por defecto: dos estudios → pipeline para ambas etapas. Solo se
		# sale de ella mediante selección explícita de una etapa o configuración.
		if has_dual and bool(getattr(self, "_dual_pipeline_auto_enabled", True)) and manual not in ("stress", "rest"):
			return ["stress", "rest"]
		if active == "both" and has_dual:
			return ["stress", "rest"]
		if active == "rest" and has_dual:
			return ["rest"]
		return ["stress"]

	def _run_cine_crudo_stage_orchestrator(self, step_label: str, runner):
		"""Orquesta una acción por etapa, incluyendo modo "Ambas" sin duplicar UI."""
		stages = self._cine_crudo_target_stages()
		if not stages:
			return False
		if len(stages) == 1:
			return runner(stages[0])

		prev_active = getattr(self, "_cine_crudo_active_stage", "stress")
		prev_recon_stage = getattr(self, "_cine_crudo_recon_stage", "stress")
		self._log(f"[DUAL] {step_label}: ejecución en Ambas (Esfuerzo → Reposo).")
		ok_all = True
		was_both = (prev_active == "both")
		try:
			for idx, stage in enumerate(stages, start=1):
				stage_txt = "Esfuerzo" if stage == "stress" else "Reposo"
				# Cambiar etapa usando el setter central para mantener sincronizados
				# los límites Base/Ápex por etapa y evitar mezclas entre stress/rest.
				# force=True: la lista de etapas ya fue validada; sin esto, un
				# falso negativo del detector de secundario degradaba 'rest' a
				# 'stress' y se reconstruía DOS veces la misma etapa.
				self._set_active_cine_crudo_stage(stage, refresh_view=False, force=True)
				self._cine_crudo_recon_stage = stage
				self._cine_crudo_dual_context = {
					"step": str(step_label),
					"stage": str(stage),
					"idx": int(idx),
					"total": int(len(stages)),
				}
				self._log(f"[DUAL] {step_label}: etapa {idx}/{len(stages)} → {stage_txt}.")
				try:
					self.statusBar().showMessage(f"{step_label} dual · {stage_txt} ({idx}/{len(stages)})", 5000)
				except Exception:
					pass
				ok = runner(stage)
				if ok is False:
					ok_all = False
					self._log(f"[DUAL] {step_label}: secuencia detenida en {stage_txt}.")
					break
		finally:
			self._cine_crudo_dual_context = None
			self._cine_crudo_recon_stage = prev_recon_stage
			# Importante: al volver a "Ambas" NO refrescar el cine crudo, para no
			# pisar el preview de reconstrucción/cortes que dejó la secuencia.
			self._set_active_cine_crudo_stage(prev_active, refresh_view=False)
		# Al terminar en "Ambas" NO forzar refresh de cine crudo: eso pisa la
		# pantalla de reconstrucción/cortes y genera el "salto" visual no deseado.
		if was_both and ok_all:
			self._log(f"[DUAL] {step_label}: completado (Esfuerzo+Reposo). Se conserva la vista actual.")
			# Para reorientación/cortes en modo dual, conviene dejar una vista
			# comparativa inmediata (pantalla partida) para validar coherencia
			# stress/rest sin pasos extra.
			if str(step_label).lower() in {"reorientación", "reorientacion", "generar cortes"}:
				try:
					self._set_active_cine_crudo_stage("both", refresh_view=False)
					self._show_cine_crudo_sa_montage()
				except Exception as exc:
					self._log(f"[WARN] No se pudo abrir el montaje dual tras {step_label}: {exc}")
			try:
				self.statusBar().showMessage(f"{step_label} dual completo · Esfuerzo + Reposo", 7000)
			except Exception:
				pass
		return ok_all

	def _detect_stage_from_study(self, study) -> str | None:
		"""Heurística: infiere 'stress'/'rest' desde metadata DICOM. None si es ambiguo."""
		if study is None:
			return None
		parts = []
		for attr in ("series_description", "study_description", "protocol_name"):
			val = getattr(study, attr, None)
			if val:
				parts.append(str(val))
		it = getattr(study, "image_type", None)
		if it:
			parts.append(" ".join(str(x) for x in it) if isinstance(it, (list, tuple)) else str(it))
		text = " ".join(parts).lower()
		if not text:
			return None
		stress_kw = ("stress", "esfuerzo", "ejercicio", "exercise", "dipirid", "dipyrid", "adenos", "dobutam", "regaden", "persantin")
		rest_kw = ("rest", "reposo", "basal", "resting")
		has_stress = any(k in text for k in stress_kw)
		has_rest = any(k in text for k in rest_kw)
		if has_stress and not has_rest:
			return "stress"
		if has_rest and not has_stress:
			return "rest"
		return None

	def _cine_crudo_stage_display(self, study) -> str | None:
		"""Nombre de etapa para rotular ('Esfuerzo'/'Reposo') si la metadata lo permite."""
		st = self._detect_stage_from_study(study)
		if st == "stress":
			return "Esfuerzo"
		if st == "rest":
			return "Reposo"
		return None

	def _cine_crudo_event_stage_and_matrix(self, event):
		"""Mapea un click sobre cine_crudo a (etapa, y, x, H, W) del estudio correspondiente (dual-aware)."""
		label = self.preview_labels.get("cine_crudo")
		pix = self.preview_pixmaps.get("cine_crudo")
		if label is None or pix is None or pix.isNull() or self.study is None:
			return None
		lw, lh = label.width(), label.height()
		pw, ph = pix.width(), pix.height()
		shown = label.pixmap()
		if shown is not None and not shown.isNull():
			lw, lh = shown.width(), shown.height()
		scale = min(lw / max(1, pw), lh / max(1, ph))
		dw, dh = pw * scale, ph * scale
		x0 = (label.width() - dw) / 2.0
		y0 = (label.height() - dh) / 2.0
		cx = (event.pos().x() - x0) / max(1e-6, scale)
		cy = (event.pos().y() - y0) / max(1e-6, scale)
		meta = getattr(self, "_cine_crudo_dual_render_meta", None) or {}
		if meta.get("enabled"):
			bar_h = int(meta.get("bar_h", 22))
			split_y = int(meta.get("split_y", 0))
			canvas_w = int(meta.get("canvas_w", pw))
			if cy < split_y:
				stage = "stress"
				study_obj = self.study
				y_local = cy - bar_h
				x_off = (canvas_w - int(meta.get("top_w", pw))) / 2.0
			else:
				stage = "rest"
				study_obj = self._secondary_cine_crudo_study() or self.study
				y_local = cy - (split_y + bar_h)
				x_off = (canvas_w - int(meta.get("bottom_w", pw))) / 2.0
			cube = np.asarray(study_obj.cube, dtype=np.float64)
			H, W = int(cube.shape[2]), int(cube.shape[3])
			return stage, float(np.clip(y_local, 0, H - 1)), float(np.clip(cx - x_off, 0, W - 1)), H, W
		cube = np.asarray(self.study.cube, dtype=np.float64)
		H, W = int(cube.shape[2]), int(cube.shape[3])
		return "stress", float(np.clip(cy, 0, H - 1)), float(np.clip(cx, 0, W - 1)), H, W

	def _load_cine_crudo_frames(self, source: str = "UngGat"):
		"""Carga frames del cine crudo desde proyecciones en memoria (UngGat o gated)."""
		self.cine_crudo_preview_mode = None
		self.cine_crudo_timer.stop()
		self.cine_crudo_playing = False
		self.cine_crudo_frames = []
		self.cine_crudo_index = 0
		self._cine_crudo_dual_render_meta = {}
		if self.study is None or bool(getattr(self.study, "reconstructed", True)):
			return
		primary_frames, primary_counts, primary_txt = self._build_cine_crudo_frames_for_study(
			self.study,
			self.cine_crudo_corrected_projections,
			source,
			"Stress",
			seed_stage=self.cine_crudo_seed,
		)
		self.cine_crudo_counts = primary_counts
		self.cine_crudo_matrix_txt = primary_txt

		secondary_study = self._secondary_cine_crudo_study()
		if secondary_study is not None:
			seed_secondary = self.cine_crudo_seed_compare if self.cine_crudo_seed_compare is not None else self._scale_seed_for_study(secondary_study)
			compare_frames, _compare_counts, compare_txt = self._build_cine_crudo_frames_for_study(
				secondary_study,
				self.cine_crudo_corrected_projections_compare,
				source,
				"Rest",
				seed_stage=seed_secondary,
			)
			active = getattr(self, "_cine_crudo_active_stage", "stress")
			top_name = self._cine_crudo_stage_display(self.study) or "Esfuerzo"
			bottom_name = self._cine_crudo_stage_display(self._secondary_cine_crudo_study()) or "Reposo"
			top_label = top_name + (" ●" if active in ("stress", "both") else "")
			bottom_label = bottom_name + (" ●" if active in ("rest", "both") else "")
			n = min(len(primary_frames), len(compare_frames))
			stacked = []
			for i in range(n):
				stacked.append(self._stack_cine_crudo_dual_pixmaps(primary_frames[i], compare_frames[i], top_label, bottom_label, active_stage=active))
			self.cine_crudo_frames = stacked
			self.cine_crudo_matrix_txt = f"{primary_txt} || {compare_txt}"
		else:
			self.cine_crudo_frames = primary_frames
		self.cine_crudo_timer.setInterval(max(40, int(self.cine_crudo_speed_spin.value() if hasattr(self, "cine_crudo_speed_spin") else 120)))
		# Preservar el frame actual (el que el usuario eligió con las flechas) en vez de resetear a 0.
		self._set_cine_crudo_frame(int(getattr(self, "_cine_crudo_current_frame", 0)))

	def _apply_cine_crudo_motion_correction(self):
		if self.study is None or bool(getattr(self.study, "reconstructed", True)):
			return
		try:
			self._set_progress(55, "Aplicando motion correction al crudo...")
			method = str(self.cine_crudo_method_combo.currentText()).lower() if hasattr(self, "cine_crudo_method_combo") else "gammasync"
			if method == "sinusoide":
				method = "sinusoid"
			axis = str(self.cine_crudo_axis_combo.currentText()).lower() if hasattr(self, "cine_crudo_axis_combo") else "y"
			threshold = self._cine_crudo_threshold_value()
			seed = self.cine_crudo_seed
			roi_radius = float(self.cine_crudo_roi_spin.value()) if hasattr(self, "cine_crudo_roi_spin") else 0.0
			roi_mode = "band" if hasattr(self, "cine_crudo_roi_mode_combo") and "banda" in str(self.cine_crudo_roi_mode_combo.currentText()).lower() else "box"
			if roi_mode == "band" and seed is not None:
				bounds = self._cine_crudo_band_bounds(int(np.asarray(self.study.cube).shape[2]))
				if bounds is not None:
					y0, y1 = bounds
					roi_radius = max(1.0, 0.5 * float(y1 - y0))
					seed = (0.5 * float(y0 + y1), float(seed[1]))
			liver_suppression = 0.0
			if hasattr(self, "cine_crudo_liver_suppress_check") and self.cine_crudo_liver_suppress_check.isChecked():
				liver_suppression = float(self.cine_crudo_liver_suppress_spin.value()) / 100.0
			ref_idx = int(self.cine_crudo_ref_index if self.cine_crudo_ref_index is not None else getattr(self, "_cine_crudo_current_frame", 0))

			do_stress = self._cine_crudo_process_stress()
			do_rest = self._cine_crudo_process_rest()
			if do_stress:
				result, corrected, jy_before, jy_after, jx_before, jx_after = self._run_motion_correction_for_study(
					self.study,
					method=method,
					axis=axis,
					threshold=threshold,
					seed=seed,
					roi_radius=roi_radius,
					roi_mode=roi_mode,
					liver_suppression=liver_suppression,
					ref_idx=ref_idx,
				)
				self.cine_crudo_motion_result = result
				self.cine_crudo_corrected_projections = corrected
				self._log(
					f"Motion correction (esfuerzo): método {result.get('method_auto_selected') or result.get('method')} | "
					f"jitter Y {jy_before:.2f}→{jy_after:.2f} · X {jx_before:.2f}→{jx_after:.2f} | "
					f"max shift {result.get('max_shift_px')} px | axis={result.get('axis_corrected')}"
				)

			secondary_study = self._secondary_cine_crudo_study()
			if do_rest and secondary_study is not None:
				seed_compare = self.cine_crudo_seed_compare if self.cine_crudo_seed_compare is not None else self._scale_seed_for_study(secondary_study)
				result_c, corrected_c, *_ = self._run_motion_correction_for_study(
					secondary_study,
					method=method,
					axis=axis,
					threshold=threshold,
					seed=seed_compare,
					roi_radius=roi_radius,
					roi_mode=roi_mode,
					liver_suppression=liver_suppression,
					ref_idx=ref_idx,
				)
				self.cine_crudo_motion_result_compare = result_c
				self.cine_crudo_corrected_projections_compare = corrected_c
				self.cine_crudo_ref_index_compare = ref_idx
				self._log("Motion correction (reposo) aplicada.")

			if self.cine_crudo_compare_check is not None:
				self.cine_crudo_compare_check.setEnabled(True)
				self.cine_crudo_compare_check.setChecked(True)
			if getattr(self, "cine_crudo_diff_check", None) is not None:
				self.cine_crudo_diff_check.setEnabled(True)
			if do_stress and do_rest and secondary_study is not None:
				self._log("Dual crudo: corrección en vivo aplicada a esfuerzo y reposo en simultáneo.")
			self._refresh_cine_crudo_view()
			self._set_progress(100, "Motion correction lista")
		except Exception as exc:
			self._log(f"[WARN] Motion correction falló: {exc}")
			self._set_progress(100, "Crudo cargado")

	def _accept_cine_crudo_motion_correction(self):
		"""Confirma la corrección de movimiento: la recon/procesamiento usarán el crudo corregido."""
		has_stress = self.cine_crudo_motion_result is not None and self.cine_crudo_corrected_projections is not None
		has_rest = (
			getattr(self, "cine_crudo_motion_result_compare", None) is not None
			and getattr(self, "cine_crudo_corrected_projections_compare", None) is not None
		)
		if not has_stress and not has_rest:
			QMessageBox.information(
				self, "SINCRO",
				"No hay corrección para aplicar. Corregí primero con un método (p.ej. Sinusoide) "
				"y, si querés, ajustá manualmente antes de aplicar.",
			)
			return
		self._cine_crudo_motion_accepted = True
		etapas = []
		if has_stress:
			etapas.append("esfuerzo")
		if has_rest:
			etapas.append("reposo")
		self._log(
			f"Corrección de movimiento APLICADA ({' y '.join(etapas)}): la reconstrucción y el "
			"procesamiento usarán el crudo corregido."
		)
		try:
			self.statusBar().showMessage("Corrección de movimiento aplicada · se usará el crudo corregido", 6000)
		except Exception:
			pass
		self._start_cine_crudo_recon_flow()

	def _reject_cine_crudo_motion_correction(self):
		"""Descarta la corrección de movimiento: se vuelve al crudo original para todo el pipeline."""
		had = (
			self.cine_crudo_motion_result is not None
			or getattr(self, "cine_crudo_motion_result_compare", None) is not None
		)
		if not had:
			QMessageBox.information(
				self, "SINCRO",
				"No hay corrección activa para rechazar; ya se está usando el crudo original.",
			)
			return
		self.cine_crudo_motion_result = None
		self.cine_crudo_corrected_projections = None
		self.cine_crudo_motion_result_compare = None
		self.cine_crudo_corrected_projections_compare = None
		self._cine_crudo_motion_accepted = False
		self._log("Corrección de movimiento RECHAZADA: se continúa con el crudo original (sin corrección).")
		# Recargar el cine desde el crudo original para que se vea sin la corrección.
		source = str(self.cine_crudo_source_combo.currentText()) if hasattr(self, "cine_crudo_source_combo") else "UngGat"
		try:
			self._load_cine_crudo_frames(source)
			self._refresh_cine_crudo_view()
		except Exception as exc:
			self._log(f"[WARN] No se pudo recargar el cine crudo tras rechazar: {exc}")
		try:
			self.statusBar().showMessage("Corrección descartada · se sigue con el crudo original", 6000)
		except Exception:
			pass
		self._start_cine_crudo_recon_flow()

	def _start_cine_crudo_recon_flow(self):
		"""Tras Aplicar/Rechazar: cierra la barra de corrección, abre la de reconstrucción
		y lanza una FBP filtrada (la más barata/genérica) como punto de partida.

		El usuario sigue ajustando los controles (filtros, NÍTIDA, etc.) desde ahí. La
		reconstrucción trabaja sobre el crudo corregido o el original según el estado que
		dejaron Aplicar/Rechazar (lo resuelve `_cine_crudo_recon_target`)."""
		menus = getattr(self, "_toolbar_group_menus", {})
		try:
			self._select_tab_by_title("cine_crudo")
		except Exception:
			pass
		corr = menus.get("cine_crudo_correccion_movimiento")
		if corr is not None:
			corr[0].hide()
		# Punto de partida: OSEM (fondo limpio, sin halo ni streaks del FBP), sin NÍTIDA.
		if hasattr(self, "cine_crudo_recon_method_combo") and self.cine_crudo_recon_method_combo is not None:
			self.cine_crudo_recon_method_combo.setCurrentText("OSEM")
		if hasattr(self, "cine_crudo_nitida_check") and self.cine_crudo_nitida_check is not None:
			self.cine_crudo_nitida_check.setChecked(False)
		rec = menus.get("cine_crudo_reconstruccion")
		if rec is not None:
			rec[0].show_near(rec[1])
		self._reconstruct_cine_crudo_raw()

	def run_stage_motion_live(self, stage: str, *, method_label: str | None = None, axis_label: str | None = None):
		"""Ejecuta motion correction de una etapa y devuelve (proyecciones_corregidas, result).

		Reutiliza todo el flujo probado (`_apply_cine_crudo_motion_correction`): seed,
		ROI, umbral, supresión de hígado y selección de etapa. Pensado para que la
		ventana de Preparación muestre el cine corregido en vivo, sin duplicar lógica.
		Devuelve (None, None) si no aplica.
		"""
		self._set_active_cine_crudo_stage(stage)
		# La etapa efectiva puede quedar forzada a 'stress' si no hay 2da etapa cruda.
		effective = getattr(self, "_cine_crudo_active_stage", "stress")
		if method_label and hasattr(self, "cine_crudo_method_combo") and self.cine_crudo_method_combo is not None:
			idx = self.cine_crudo_method_combo.findText(method_label)
			if idx >= 0:
				self.cine_crudo_method_combo.setCurrentIndex(idx)
		if axis_label and hasattr(self, "cine_crudo_axis_combo") and self.cine_crudo_axis_combo is not None:
			idx = self.cine_crudo_axis_combo.findText(axis_label)
			if idx >= 0:
				self.cine_crudo_axis_combo.setCurrentIndex(idx)
		self._apply_cine_crudo_motion_correction()
		if effective == "rest":
			return self.cine_crudo_corrected_projections_compare, self.cine_crudo_motion_result_compare
		return self.cine_crudo_corrected_projections, self.cine_crudo_motion_result

	def _run_motion_correction_for_study(self, study_obj, *, method: str, axis: str, threshold: float, seed, roi_radius: float, roi_mode: str, liver_suppression: float, ref_idx: int):
		"""Ejecuta motion correction sobre un estudio crudo individual y devuelve métricas de jitter."""
		from core.raw_projections import center_of_mass_tracking, motion_correct_projections

		projections = np.asarray(study_obj.cube, dtype=np.float64)
		angles = getattr(study_obj, "angles_deg", None)

		def _jitter(pr, ax):
			c = np.asarray(center_of_mass_tracking(np.asarray(pr, dtype=np.float64), axis=ax).get("com_series", []), dtype=np.float64)
			v = np.isfinite(c)
			if v.sum() < 3:
				return 0.0
			return float(np.std(np.diff(c[v])))

		if method == "auto":
			candidates = ["sinusoid", "xcorr", "com", "odyssey", "stasis", "hopkins", "gammasync", "threshold"]
			best_result = None
			best_method = None
			best_score = 1e18
			for m in candidates:
				res_m = motion_correct_projections(
					projections,
					axis=axis,
					method=m,
					threshold_frac=threshold,
					seed=seed,
					max_abs_shift_px=4.0,
					smooth_sigma=1.0,
					ref_index=ref_idx,
					angles_deg=angles,
					roi_radius=roi_radius,
					roi_mode=roi_mode,
					liver_suppression_frac=liver_suppression,
				)
				corr_m = np.asarray(res_m.get("corrected"), dtype=np.float64)
				score = 1.00 * _jitter(corr_m, "y") + 0.60 * _jitter(corr_m, "x")
				if score < best_score:
					best_score = score
					best_result = res_m
					best_method = m
			result = best_result
			if result is not None:
				result["method_auto_selected"] = best_method
		else:
			result = motion_correct_projections(
				projections,
				axis=axis,
				method=method,
				threshold_frac=threshold,
				seed=seed,
				max_abs_shift_px=4.0,
				smooth_sigma=1.0,
				ref_index=ref_idx,
				angles_deg=angles,
				roi_radius=roi_radius,
				roi_mode=roi_mode,
				liver_suppression_frac=liver_suppression,
			)

		corrected = np.asarray(result.get("corrected"), dtype=np.float64)
		jy_before = _jitter(projections, "y")
		jy_after = _jitter(corrected, "y")
		jx_before = _jitter(projections, "x")
		jx_after = _jitter(corrected, "x")
		worse = (jy_after > 1.10 * max(jy_before, 1e-6)) and (jy_after + jx_after > jy_before + jx_before)
		if worse and not bool(result.get("manual_override")):
			corrected = projections.copy()
			result["corrected"] = corrected
			result["applied_shifts_y"] = np.zeros((int(projections.shape[1]),), dtype=np.float64)
			result["applied_shifts_x"] = np.zeros((int(projections.shape[1]),), dtype=np.float64)
			jy_after, jx_after = jy_before, jx_before
		return result, corrected, jy_before, jy_after, jx_before, jx_after

	def _apply_cine_crudo_manual_offset(self):
		"""Aplica offset manual global X/Y a la corrección ya calculada en la(s) etapa(s) activa(s)."""
		do_s = self._cine_crudo_process_stress() and self.cine_crudo_motion_result is not None
		do_r = self._cine_crudo_process_rest() and self.cine_crudo_motion_result_compare is not None
		if self.study is None or (not do_s and not do_r):
			QMessageBox.information(self, "SINCRO", "Primero ejecutá una corrección automática en la(s) etapa(s) seleccionada(s).")
			return
		try:
			from core.raw_projections import apply_shifts_to_projections
			offy = float(self.cine_crudo_offset_y_spin.value()) if hasattr(self, "cine_crudo_offset_y_spin") else 0.0
			offx = float(self.cine_crudo_offset_x_spin.value()) if hasattr(self, "cine_crudo_offset_x_spin") else 0.0
			if do_s:
				projections = np.asarray(self.study.cube, dtype=np.float64)
				sy = np.asarray(self.cine_crudo_motion_result.get("applied_shifts_y", np.zeros((projections.shape[1],))), dtype=np.float64)
				sx = np.asarray(self.cine_crudo_motion_result.get("applied_shifts_x", np.zeros((projections.shape[1],))), dtype=np.float64)
				sy2 = sy + offy
				sx2 = sx + offx
				corr = apply_shifts_to_projections(projections, sy2, sx2)
				self.cine_crudo_corrected_projections = np.asarray(corr, dtype=np.float64)
				self.cine_crudo_motion_result["corrected"] = self.cine_crudo_corrected_projections
				self.cine_crudo_motion_result["applied_shifts_y"] = sy2
				self.cine_crudo_motion_result["applied_shifts_x"] = sx2
			if do_r:
				secondary_study = self._secondary_cine_crudo_study()
				projections_c = np.asarray(secondary_study.cube, dtype=np.float64)
				sy_c = np.asarray(self.cine_crudo_motion_result_compare.get("applied_shifts_y", np.zeros((projections_c.shape[1],))), dtype=np.float64)
				sx_c = np.asarray(self.cine_crudo_motion_result_compare.get("applied_shifts_x", np.zeros((projections_c.shape[1],))), dtype=np.float64)
				sy2_c = sy_c + offy
				sx2_c = sx_c + offx
				corr_c = apply_shifts_to_projections(projections_c, sy2_c, sx2_c)
				self.cine_crudo_corrected_projections_compare = np.asarray(corr_c, dtype=np.float64)
				self.cine_crudo_motion_result_compare["corrected"] = self.cine_crudo_corrected_projections_compare
				self.cine_crudo_motion_result_compare["applied_shifts_y"] = sy2_c
				self.cine_crudo_motion_result_compare["applied_shifts_x"] = sx2_c
			self._refresh_cine_crudo_view()
			self._log(f"Offset manual aplicado: OffY={offy:+.2f}px OffX={offx:+.2f}px")
		except Exception as exc:
			self._log(f"[WARN] Aplicar offset manual falló: {exc}")

	def _set_cine_crudo_reference_frame(self):
		"""Fija el frame actual como referencia de corrección (shift=0 en ese frame)."""
		if not self.cine_crudo_frames:
			QMessageBox.information(self, "SINCRO", "Primero cargá el cine crudo.")
			return
		self.cine_crudo_ref_index = int(getattr(self, "_cine_crudo_current_frame", self.cine_crudo_index))
		self.cine_crudo_ref_index_compare = self.cine_crudo_ref_index
		self._log(f"Frame de referencia fijado: {self.cine_crudo_ref_index}. La próxima corrección ancla shift=0 en ese frame.")

	def _cine_crudo_nudge_step(self) -> float:
		"""Tamaño del paso (px) de las flechas de corrección manual."""
		if hasattr(self, "cine_crudo_nudge_step_spin"):
			return float(self.cine_crudo_nudge_step_spin.value())
		return 0.5

	def _cine_crudo_ensure_shift_arrays(self):
		"""Garantiza que exista un motion_result con arrays de shift por frame (para edición manual)."""
		projections = np.asarray(self.study.cube, dtype=np.float64)
		n_angles = int(projections.shape[1])
		if self.cine_crudo_motion_result is None:
			self.cine_crudo_motion_result = {
				"corrected": projections.copy(),
				"applied_shifts_y": np.zeros((n_angles,), dtype=np.float64),
				"applied_shifts_x": np.zeros((n_angles,), dtype=np.float64),
				"method": "manual",
				"manual_edited": True,
			}
			self.cine_crudo_corrected_projections = projections.copy()
		return projections, n_angles

	def _nudge_cine_crudo_frame(self, dy: float, dx: float):
		"""Corrige manualmente SOLO el frame actual en la(s) etapa(s) activa(s): suma dy/dx a su shift."""
		if self.study is None or bool(getattr(self.study, "reconstructed", True)):
			return
		if not self.cine_crudo_frames:
			QMessageBox.information(self, "SINCRO", "Primero cargá el cine del crudo.")
			return
		do_s = self._cine_crudo_process_stress()
		do_r = self._cine_crudo_process_rest()
		try:
			from core.raw_projections import apply_shifts_to_projections
			idx_global = int(getattr(self, "_cine_crudo_current_frame", self.cine_crudo_index))
			if do_s:
				projections, n_angles = self._cine_crudo_ensure_shift_arrays()
				idx = idx_global % n_angles
				sy = np.asarray(self.cine_crudo_motion_result.get("applied_shifts_y", np.zeros((n_angles,))), dtype=np.float64).copy()
				sx = np.asarray(self.cine_crudo_motion_result.get("applied_shifts_x", np.zeros((n_angles,))), dtype=np.float64).copy()
				if sy.size != n_angles:
					sy = np.zeros((n_angles,), dtype=np.float64)
				if sx.size != n_angles:
					sx = np.zeros((n_angles,), dtype=np.float64)
				sy[idx] += float(dy)
				sx[idx] += float(dx)
				corr = apply_shifts_to_projections(projections, sy, sx)
				self.cine_crudo_corrected_projections = np.asarray(corr, dtype=np.float64)
				self.cine_crudo_motion_result["applied_shifts_y"] = sy
				self.cine_crudo_motion_result["applied_shifts_x"] = sx
				self.cine_crudo_motion_result["corrected"] = self.cine_crudo_corrected_projections
				self.cine_crudo_motion_result["manual_edited"] = True
				self._log(f"Manual (esfuerzo) frame {idx}: shift Y={sy[idx]:+.2f} X={sx[idx]:+.2f} px (Δ Y={dy:+.2f} X={dx:+.2f})")
			secondary_study = self._secondary_cine_crudo_study()
			if do_r and secondary_study is not None:
				projections_c = np.asarray(secondary_study.cube, dtype=np.float64)
				n_angles_c = int(projections_c.shape[1])
				if self.cine_crudo_motion_result_compare is None:
					self.cine_crudo_motion_result_compare = {
						"corrected": projections_c.copy(),
						"applied_shifts_y": np.zeros((n_angles_c,), dtype=np.float64),
						"applied_shifts_x": np.zeros((n_angles_c,), dtype=np.float64),
						"method": "manual",
						"manual_edited": True,
					}
				idx_c = idx_global % n_angles_c
				sy_c = np.asarray(self.cine_crudo_motion_result_compare.get("applied_shifts_y", np.zeros((n_angles_c,))), dtype=np.float64).copy()
				sx_c = np.asarray(self.cine_crudo_motion_result_compare.get("applied_shifts_x", np.zeros((n_angles_c,))), dtype=np.float64).copy()
				if sy_c.size != n_angles_c:
					sy_c = np.zeros((n_angles_c,), dtype=np.float64)
				if sx_c.size != n_angles_c:
					sx_c = np.zeros((n_angles_c,), dtype=np.float64)
				sy_c[idx_c] += float(dy)
				sx_c[idx_c] += float(dx)
				corr_c = apply_shifts_to_projections(projections_c, sy_c, sx_c)
				self.cine_crudo_corrected_projections_compare = np.asarray(corr_c, dtype=np.float64)
				self.cine_crudo_motion_result_compare["applied_shifts_y"] = sy_c
				self.cine_crudo_motion_result_compare["applied_shifts_x"] = sx_c
				self.cine_crudo_motion_result_compare["corrected"] = self.cine_crudo_corrected_projections_compare
				self.cine_crudo_motion_result_compare["manual_edited"] = True
				self._log(f"Manual (reposo) frame {idx_c}: shift Y={sy_c[idx_c]:+.2f} X={sx_c[idx_c]:+.2f} px (Δ Y={dy:+.2f} X={dx:+.2f})")
			if self.cine_crudo_compare_check is not None:
				self.cine_crudo_compare_check.setEnabled(True)
			self._refresh_cine_crudo_view()
			self._set_cine_crudo_frame(idx_global)
		except Exception as exc:
			self._log(f"[WARN] Ajuste manual falló: {exc}")

	def _reset_cine_crudo_frame_shift(self):
		"""Pone el shift del frame actual en 0 en la(s) etapa(s) activa(s)."""
		do_s = self._cine_crudo_process_stress() and self.cine_crudo_motion_result is not None
		do_r = self._cine_crudo_process_rest() and self.cine_crudo_motion_result_compare is not None
		if self.study is None or not self.cine_crudo_frames or (not do_s and not do_r):
			return
		try:
			from core.raw_projections import apply_shifts_to_projections
			idx_global = int(getattr(self, "_cine_crudo_current_frame", self.cine_crudo_index))
			if do_s:
				projections = np.asarray(self.study.cube, dtype=np.float64)
				n_angles = int(projections.shape[1])
				idx = idx_global % n_angles
				sy = np.asarray(self.cine_crudo_motion_result.get("applied_shifts_y", np.zeros((n_angles,))), dtype=np.float64).copy()
				sx = np.asarray(self.cine_crudo_motion_result.get("applied_shifts_x", np.zeros((n_angles,))), dtype=np.float64).copy()
				if sy.size == n_angles:
					sy[idx] = 0.0
				if sx.size == n_angles:
					sx[idx] = 0.0
				corr = apply_shifts_to_projections(projections, sy, sx)
				self.cine_crudo_corrected_projections = np.asarray(corr, dtype=np.float64)
				self.cine_crudo_motion_result["applied_shifts_y"] = sy
				self.cine_crudo_motion_result["applied_shifts_x"] = sx
				self.cine_crudo_motion_result["corrected"] = self.cine_crudo_corrected_projections
			secondary_study = self._secondary_cine_crudo_study()
			if do_r and secondary_study is not None:
				projections_c = np.asarray(secondary_study.cube, dtype=np.float64)
				n_angles_c = int(projections_c.shape[1])
				idx_c = idx_global % n_angles_c
				sy_c = np.asarray(self.cine_crudo_motion_result_compare.get("applied_shifts_y", np.zeros((n_angles_c,))), dtype=np.float64).copy()
				sx_c = np.asarray(self.cine_crudo_motion_result_compare.get("applied_shifts_x", np.zeros((n_angles_c,))), dtype=np.float64).copy()
				if sy_c.size == n_angles_c:
					sy_c[idx_c] = 0.0
				if sx_c.size == n_angles_c:
					sx_c[idx_c] = 0.0
				corr_c = apply_shifts_to_projections(projections_c, sy_c, sx_c)
				self.cine_crudo_corrected_projections_compare = np.asarray(corr_c, dtype=np.float64)
				self.cine_crudo_motion_result_compare["applied_shifts_y"] = sy_c
				self.cine_crudo_motion_result_compare["applied_shifts_x"] = sx_c
				self.cine_crudo_motion_result_compare["corrected"] = self.cine_crudo_corrected_projections_compare
			self._refresh_cine_crudo_view()
			self._set_cine_crudo_frame(idx_global)
			self._log(f"Frame {idx_global}: shift reseteado a 0 en la(s) etapa(s) activa(s).")
		except Exception as exc:
			self._log(f"[WARN] Reset de frame falló: {exc}")

	def _export_cine_crudo_correction(self):
		"""Exporta shifts Y/X por frame (CSV) + proyecciones corregidas (.npz) para comparar/calibrar métodos."""
		if self.cine_crudo_motion_result is None:
			QMessageBox.information(self, "SINCRO", "Primero ejecutá una corrección (automática o manual con flechas).")
			return
		try:
			from PyQt6.QtWidgets import QFileDialog
			sy = np.asarray(self.cine_crudo_motion_result.get("applied_shifts_y", []), dtype=np.float64)
			sx = np.asarray(self.cine_crudo_motion_result.get("applied_shifts_x", []), dtype=np.float64)
			n = int(max(sy.size, sx.size))
			if n == 0:
				QMessageBox.information(self, "SINCRO", "No hay shifts para exportar.")
				return
			if sy.size != n:
				sy = np.zeros((n,), dtype=np.float64)
			if sx.size != n:
				sx = np.zeros((n,), dtype=np.float64)
			method = str(self.cine_crudo_motion_result.get("method_auto_selected") or self.cine_crudo_motion_result.get("method") or "manual")
			default_base = os.path.join(self.output_dir, f"motion_correction_{method}")
			path, _flt = QFileDialog.getSaveFileName(
				self, "Exportar corrección de movimiento", default_base,
				"CSV de shifts (*.csv);;Todos los archivos (*.*)",
			)
			if not path:
				return
			base, ext = os.path.splitext(path)
			csv_path = path if ext.lower() == ".csv" else base + ".csv"
			# Identidad del estudio (para validar en la importación que corresponde al mismo paciente/file).
			pid = str(getattr(self.study, "patient_id", "") or "")
			pname = str(getattr(self.study, "patient_name", "") or "")
			suid = str(getattr(self.study, "study_instance_uid", "") or "")
			src = str(getattr(self.study, "source_path", "") or "")
			src_name = os.path.basename(src)
			# CSV: cabecera de identidad (# ...) + frame, angulo_deg, shift_y_px, shift_x_px
			angles = getattr(self.study, "angles_deg", None)
			lines = [
				f"# patient_id={pid}",
				f"# patient_name={pname}",
				f"# study_uid={suid}",
				f"# source_file={src_name}",
				f"# n_angles={n}",
				"frame,angle_deg,shift_y_px,shift_x_px",
			]
			for i in range(n):
				ang = float(angles[i]) if (angles is not None and i < len(angles)) else float("nan")
				lines.append(f"{i},{ang:.3f},{sy[i]:.4f},{sx[i]:.4f}")
			csv_text = "\n".join(lines) + "\n"
			with open(csv_path, "wb") as fh:
				fh.write(csv_text.encode("utf-8"))
			# NPZ: shifts + proyecciones corregidas (para reconstruir/comparar) + identidad.
			npz_path = base + ".npz"
			corrected = np.asarray(self.cine_crudo_corrected_projections, dtype=np.float32) if self.cine_crudo_corrected_projections is not None else None
			save_kwargs = {
				"shifts_y": sy.astype(np.float32),
				"shifts_x": sx.astype(np.float32),
				"method": np.array(method),
				"ref_index": np.array(self.cine_crudo_ref_index if self.cine_crudo_ref_index is not None else -1),
				"patient_id": np.array(pid),
				"patient_name": np.array(pname),
				"study_uid": np.array(suid),
				"source_file": np.array(src_name),
				"n_angles": np.array(n),
			}
			if corrected is not None:
				save_kwargs["corrected"] = corrected
			np.savez_compressed(npz_path, **save_kwargs)
			self._log(f"Corrección exportada: {os.path.basename(csv_path)} + {os.path.basename(npz_path)} (método {method}, {n} frames).")
			QMessageBox.information(
				self, "SINCRO",
				f"Exportado:\n• {csv_path}\n• {npz_path}\n\nShifts Y/X por frame y proyecciones corregidas para comparar/calibrar.",
			)
		except Exception as exc:
			self._log(f"[WARN] Exportar corrección falló: {exc}")

	def _import_cine_crudo_correction(self):
		"""Re-lee una corrección guardada (CSV o NPZ) y aplica los shifts Y/X al estudio actual."""
		if self.study is None or bool(getattr(self.study, "reconstructed", True)):
			QMessageBox.information(self, "SINCRO", "Cargá primero el crudo gated (proyecciones) al que aplicar la corrección.")
			return
		try:
			from PyQt6.QtWidgets import QFileDialog
			from core.raw_projections import apply_shifts_to_projections
			path, _flt = QFileDialog.getOpenFileName(
				self, "Importar corrección", self.output_dir,
				"Corrección (*.csv *.npz);;CSV de shifts (*.csv);;NPZ (*.npz);;Todos los archivos (*.*)",
			)
			if not path:
				return
			projections = np.asarray(self.study.cube, dtype=np.float64)
			n_angles = int(projections.shape[1])
			ext = os.path.splitext(path)[1].lower()
			method = "importado"
			ref_index = None
			ident: dict = {}
			if ext == ".npz":
				data = np.load(path, allow_pickle=True)
				sy = np.asarray(data["shifts_y"], dtype=np.float64) if "shifts_y" in data else np.zeros((n_angles,))
				sx = np.asarray(data["shifts_x"], dtype=np.float64) if "shifts_x" in data else np.zeros((n_angles,))
				if "method" in data:
					method = f"importado ({str(data['method'])})"
				if "ref_index" in data:
					ri = int(data["ref_index"])
					ref_index = ri if ri >= 0 else None
				for k in ("patient_id", "patient_name", "study_uid", "source_file", "n_angles"):
					if k in data:
						ident[k] = str(data[k])
			else:
				# CSV: comentarios de identidad (# key=value) + frame,angle_deg,shift_y_px,shift_x_px
				with open(path, "rb") as fh:
					text = fh.read().decode("utf-8", errors="replace")
				all_lines = [ln for ln in text.splitlines() if ln.strip()]
				rows = []
				for ln in all_lines:
					s = ln.strip()
					if s.startswith("#"):
						body = s[1:].strip()
						if "=" in body:
							k, v = body.split("=", 1)
							ident[k.strip()] = v.strip()
						continue
					if s.lower().startswith("frame"):
						continue
					rows.append(s)
				sy = np.zeros((n_angles,), dtype=np.float64)
				sx = np.zeros((n_angles,), dtype=np.float64)
				for ln in rows:
					parts = ln.split(",")
					if len(parts) < 4:
						continue
					try:
						fi = int(float(parts[0]))
					except ValueError:
						continue
					if 0 <= fi < n_angles:
						sy[fi] = float(parts[2])
						sx[fi] = float(parts[3])
			# --- Validar que la corrección corresponde a ESTE paciente/estudio ---
			cur_pid = str(getattr(self.study, "patient_id", "") or "")
			cur_suid = str(getattr(self.study, "study_instance_uid", "") or "")
			cur_src = os.path.basename(str(getattr(self.study, "source_path", "") or ""))
			reasons: list[str] = []
			saved_suid = ident.get("study_uid", "")
			saved_pid = ident.get("patient_id", "")
			saved_src = ident.get("source_file", "")
			saved_na = ident.get("n_angles", "")
			if saved_suid and cur_suid and saved_suid != cur_suid:
				reasons.append(f"StudyInstanceUID distinto (guardado ≠ actual)")
			if saved_pid and cur_pid and saved_pid != cur_pid:
				reasons.append(f"PatientID: guardado '{saved_pid}' ≠ actual '{cur_pid}'")
			if saved_na:
				try:
					if int(float(saved_na)) != n_angles:
						reasons.append(f"Nº de ángulos: guardado {saved_na} ≠ actual {n_angles}")
				except ValueError:
					pass
			has_ident = bool(saved_suid or saved_pid or saved_src or saved_na)
			if reasons:
				QMessageBox.critical(
					self, "SINCRO — corrección no corresponde",
					"La corrección guardada NO corresponde a este estudio:\n\n• "
					+ "\n• ".join(reasons)
					+ f"\n\nPaciente/archivo actual: {cur_pid or '—'} / {cur_src or '—'}\n"
					+ f"Corrección guardada: {saved_pid or '—'} / {saved_src or '—'}\n\nNo se cargó nada.",
				)
				self._log(f"[BLOQUEADO] Importar corrección: no corresponde al estudio actual — {'; '.join(reasons)}.")
				return
			if not has_ident:
				resp = QMessageBox.question(
					self, "SINCRO — identidad no verificable",
					"El archivo de corrección no contiene datos de identidad (paciente/estudio), "
					"probablemente es de una versión anterior.\n\nNo puedo verificar que corresponda a este estudio. "
					"¿Cargar de todos modos?",
					QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
					QMessageBox.StandardButton.No,
				)
				if resp != QMessageBox.StandardButton.Yes:
					self._log("Importar corrección cancelado: identidad no verificable.")
					return
			# Ajustar longitud al estudio actual.
			if sy.size != n_angles:
				tmp = np.zeros((n_angles,), dtype=np.float64); tmp[:min(n_angles, sy.size)] = sy[:min(n_angles, sy.size)]; sy = tmp
			if sx.size != n_angles:
				tmp = np.zeros((n_angles,), dtype=np.float64); tmp[:min(n_angles, sx.size)] = sx[:min(n_angles, sx.size)]; sx = tmp
			corr = apply_shifts_to_projections(projections, sy, sx)
			self.cine_crudo_corrected_projections = np.asarray(corr, dtype=np.float64)
			self.cine_crudo_ref_index = ref_index
			self.cine_crudo_motion_result = {
				"corrected": self.cine_crudo_corrected_projections,
				"applied_shifts_y": sy,
				"applied_shifts_x": sx,
				"method": method,
				"manual_edited": True,
				"ref_index": ref_index if ref_index is not None else -1,
			}
			if self.cine_crudo_compare_check is not None:
				self.cine_crudo_compare_check.setEnabled(True)
			source = str(self.cine_crudo_source_combo.currentText()) if hasattr(self, "cine_crudo_source_combo") else "UngGat"
			self._load_cine_crudo_frames(source)
			self._refresh_cine_crudo_view()
			self._log(f"Corrección importada de {os.path.basename(path)} ({method}): shift máx Y={np.abs(sy).max():.2f} X={np.abs(sx).max():.2f} px. Podés seguir ajustando con las flechas, Comparar o Grabar DICOM.")
			QMessageBox.information(
				self, "SINCRO",
				f"Corrección importada:\n• {os.path.basename(path)}\n\nShifts aplicados al estudio actual ({n_angles} frames). Activá 'Comparar' para ver original|corregido, seguí ajustando con las flechas o grabá el DICOM corregido.",
			)
		except Exception as exc:
			self._log(f"[WARN] Importar corrección falló: {exc}")
			QMessageBox.warning(self, "SINCRO", f"No se pudo importar la corrección:\n{exc}")

	def _save_cine_crudo_corrected_dicom(self):
		"""Graba las proyecciones corregidas como un DICOM GATED TOMO nuevo (re-cargable)."""
		if self.cine_crudo_corrected_projections is None:
			QMessageBox.information(self, "SINCRO", "Primero ejecutá una corrección (automática o manual con flechas).")
			return
		source_path = str(getattr(self.study, "source_path", "") or "")
		if not source_path or not os.path.exists(source_path):
			QMessageBox.warning(self, "SINCRO", "No encuentro el DICOM original para preservar la cabecera. Grabá desde el estudio cargado.")
			return
		try:
			from PyQt6.QtWidgets import QFileDialog
			from core.raw_projections import save_corrected_projections_dicom
			method = str(self.cine_crudo_motion_result.get("method_auto_selected") or self.cine_crudo_motion_result.get("method") or "manual") if self.cine_crudo_motion_result else "manual"
			base_name = os.path.splitext(os.path.basename(source_path))[0]
			default_path = os.path.join(self.output_dir, f"{base_name}_MOTIONCORR_{method}.dcm")
			path, _flt = QFileDialog.getSaveFileName(
				self, "Grabar DICOM corregido", default_path,
				"DICOM (*.dcm);;Todos los archivos (*.*)",
			)
			if not path:
				return
			base, ext = os.path.splitext(path)
			if ext.lower() != ".dcm":
				path = base + ".dcm"
			out = save_corrected_projections_dicom(
				source_path,
				np.asarray(self.cine_crudo_corrected_projections, dtype=np.float64),
				path,
				series_description_suffix=f"MOTION CORR {method}",
			)
			self._log(f"DICOM corregido grabado: {os.path.basename(out)} (método {method}).")
			QMessageBox.information(
				self, "SINCRO",
				f"DICOM corregido grabado:\n• {out}\n\nSerie nueva GATED TOMO (misma geometría que el original), re-cargable por SINCRO o Xeleris.",
			)
		except Exception as exc:
			self._log(f"[WARN] Grabar DICOM corregido falló: {exc}")
			QMessageBox.warning(self, "SINCRO", f"No se pudo grabar el DICOM:\n{exc}")

	def _cine_crudo_recon_filter_config(self, branch: str):
		from core.raw_reconstruction import ProjectionFilterConfig

		if branch == "ungated":
			kind = str(self.cine_crudo_ung_filter_combo.currentText()) if hasattr(self, "cine_crudo_ung_filter_combo") else "butterworth"
			cutoff = float(self.cine_crudo_ung_cutoff_spin.value()) if hasattr(self, "cine_crudo_ung_cutoff_spin") else 0.52
			order = int(self.cine_crudo_ung_order_spin.value()) if hasattr(self, "cine_crudo_ung_order_spin") else 5
		else:
			kind = str(self.cine_crudo_gated_filter_combo.currentText()) if hasattr(self, "cine_crudo_gated_filter_combo") else "butterworth"
			cutoff = float(self.cine_crudo_gated_cutoff_spin.value()) if hasattr(self, "cine_crudo_gated_cutoff_spin") else 0.40
			order = int(self.cine_crudo_gated_order_spin.value()) if hasattr(self, "cine_crudo_gated_order_spin") else 10
		return ProjectionFilterConfig(kind=kind, cutoff=cutoff, order=order)

	def _load_cine_crudo_ct_attmap(self):
		"""Carga ATTMAP o CT para la etapa elegida y deja el μ-map nativo en StageState.

		Cuatro combinaciones explícitas: ATTMAP/CT × Esfuerzo/Reposo, o carpeta
		inteligente que detecta SPECT crudo + ATT + CT de ambas etapas de una.
		"""
		active = str(getattr(self, "_cine_crudo_recon_stage", None) or self._cine_crudo_active_stage_or_default())
		dlg = QDialog(self)
		dlg.setWindowTitle("SINCRO · CT/ATT para corrección de atenuación")
		lay = QVBoxLayout(dlg)
		b_smart = QPushButton("🔍 Carpeta inteligente (SPECT + ATT + CT de ambas etapas)")
		b_smart.setToolTip(
			"Escanea una carpeta y clasifica automáticamente por metadata DICOM y nombre: "
			"crudos SPECT (esfuerzo/reposo), ATT maps y CTs de cada etapa. Carga todo de una.")
		b_smart.setStyleSheet("font-weight:bold; padding:6px;")
		lay.addWidget(b_smart)
		lay.addWidget(QLabel("— o carga manual (cada etapa usa su PROPIO CT/ATTMAP):"))
		row1 = QHBoxLayout()
		row1.addWidget(QLabel("Etapa:"))
		stage_combo = QComboBox()
		stage_combo.addItem("Esfuerzo", "stress")
		stage_combo.addItem("Reposo", "rest")
		stage_combo.setCurrentIndex(1 if active == "rest" else 0)
		row1.addWidget(stage_combo, 1)
		lay.addLayout(row1)
		row2 = QHBoxLayout()
		row2.addWidget(QLabel("Tipo:"))
		type_combo = QComboBox()
		type_combo.addItem("ATTMAP (μ-map exportado por el equipo)", "att")
		type_combo.addItem("CT (HU → μ bilineal 140 keV)", "ct")
		row2.addWidget(type_combo, 1)
		lay.addLayout(row2)
		picked = {"path": "", "smart": False}
		btns = QHBoxLayout()
		b_file = QPushButton("Archivo...")
		b_dir = QPushButton("Carpeta...")
		b_cancel = QPushButton("Cancelar")

		def _pick_file():
			p, _ = QFileDialog.getOpenFileName(dlg, "CT/ATTMAP · archivo DICOM", "", "DICOM (*.dcm *.ima);;Todos (*.*)")
			if p:
				picked["path"] = p
				dlg.accept()

		def _pick_dir():
			p = QFileDialog.getExistingDirectory(dlg, "CT/ATTMAP · carpeta de serie")
			if p:
				picked["path"] = p
				dlg.accept()

		def _pick_smart():
			picked["smart"] = True
			dlg.accept()

		b_smart.clicked.connect(_pick_smart)
		b_file.clicked.connect(_pick_file)
		b_dir.clicked.connect(_pick_dir)
		b_cancel.clicked.connect(dlg.reject)
		btns.addWidget(b_file)
		btns.addWidget(b_dir)
		btns.addWidget(b_cancel)
		lay.addLayout(btns)
		if dlg.exec() != QDialog.DialogCode.Accepted:
			return
		if picked["smart"]:
			self._smart_load_ct_att_folder()
			return
		if not picked["path"]:
			return
		self._load_ct_att_for_stage(str(stage_combo.currentData()), picked["path"], str(type_combo.currentData()))

	def _load_ct_att_for_stage(self, stage: str, path: str, kind: str, *,
							   series_uid: str | None = None, mu_only: bool = False) -> bool:
		"""Carga CT o ATT en la etapa. mu_only: solo pisa el μ-map (conserva el CT display ya cargado)."""
		from core.ct_fusion import (
			load_attenuation_map_from_path,
			load_ct_volume_from_path,
			mu_map_from_ct_hu,
			validate_mu_map,
		)

		stage = "rest" if str(stage) == "rest" else "stress"
		stage_txt = "ESFUERZO" if stage == "stress" else "REPOSO"
		try:
			conv_notes: list[str] = []
			if kind == "ct":
				res = load_ct_volume_from_path(path, series_uid=series_uid)
				mu, conv_notes = mu_map_from_ct_hu(np.asarray(res.volume, dtype=np.float64))
				source = "ct_bilineal"
			else:
				res = load_attenuation_map_from_path(path)
				vol = np.asarray(res.volume, dtype=np.float64)
				if float(np.nanmin(vol)) < -200.0:
					mu, conv_notes = mu_map_from_ct_hu(vol)
					conv_notes.append("La serie elegída como ATTMAP contenía HU de CT: se convirtió con la bilineal.")
					source = "ct_bilineal"
				else:
					mu = np.clip(vol, 0.0, None)
					q99 = float(np.percentile(mu, 99.0)) if mu.size else 0.0
					if q99 > 2.0:
						# Export en enteros escalados: normalizar p99 → μ agua.
						mu = mu / q99 * 0.154
						conv_notes.append(f"ATTMAP reescalado: p99={q99:.1f} → 0.154/cm (μ agua).")
					source = "att_export"
			ok, qc_notes = validate_mu_map(mu)
			for n in list(getattr(res, "notes", []) or []) + conv_notes + qc_notes:
				self._log(f"[AC] {n}")
			if not ok:
				QMessageBox.warning(
					self, "SINCRO",
					"El μ-map cargado no pasó el QC (ver log — posible export vacío/roto).\n"
					"Probá cargando el CT y usá la conversión bilineal.",
				)
				return False
			st = self._dual_session().stage(stage)
			st.mu_map_native = mu
			st.mu_map_spacing_zyx = getattr(res, "spacing_zyx", None)
			st.mu_map_source = source
			st.mu_map_description = str(getattr(res, "series_description", "") or "")
			st.mu_map_recon_grid = None
			st.mu_map_shift_zyx = None
			if not mu_only:
				st.ct_path = str(path)
				st.ct_volume_native = np.asarray(res.volume, dtype=np.float64)
				st.ct_affine_ijk_to_lps = getattr(res, "affine_ijk_to_lps", None)
				st.ct_spacing_zyx = getattr(res, "spacing_zyx", None)
			if getattr(self, "cine_crudo_ac_check", None) is not None:
				self.cine_crudo_ac_check.setEnabled(True)
				self.cine_crudo_ac_check.setChecked(True)
			if getattr(self, "cine_crudo_fusion_btn", None) is not None:
				self._refresh_fusion_btn_state()
			src_txt = "ATTMAP export" if source == "att_export" else "CT→μ bilineal (140 keV)"
			self._log(
				f"[AC] μ-map cargado para {stage_txt}: {mu.shape}, fuente={src_txt}, "
				f"spacing={st.mu_map_spacing_zyx}, serie='{st.mu_map_description}'."
				+ (" (solo μ: el CT display de la etapa se conserva)" if mu_only else "")
			)
			return True
		except Exception as exc:
			QMessageBox.warning(self, "SINCRO", f"No se pudo cargar CT/ATTMAP ({stage_txt}):\n{exc}")
			return False

	@staticmethod
	def _stage_from_dicom_text(text: str) -> str | None:
		t = str(text or "").lower()
		stress_kw = ("stress", "esfuerzo", "ejercicio", "exercise", "dipirid", "dipyrid", "adenos", "dobutam", "regaden", "persantin", "_str", "str_")
		rest_kw = ("rest", "reposo", "basal", "resting")
		has_s = any(k in t for k in stress_kw)
		has_r = any(k in t for k in rest_kw)
		if has_s and not has_r:
			return "stress"
		if has_r and not has_s:
			return "rest"
		return None

	def _smart_load_ct_att_folder(self):
		"""Escanea una carpeta y carga SPECT crudo + ATT + CT de cada etapa de una sola vez."""
		import pydicom

		folder = QFileDialog.getExistingDirectory(self, "Carpeta con SPECT + CT/ATT del paciente")
		if not folder:
			return
		self._log(f"[SMART] Escaneando {folder} ...")
		series: dict[str, dict] = {}
		for base, _dirs, files in os.walk(folder):
			for fn in files:
				if fn.lower().endswith((".png", ".jpg", ".txt", ".pdf", ".json", ".xml", ".csv")):
					continue
				fp = os.path.join(base, fn)
				try:
					ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
					uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
					if not uid:
						continue
					info = series.setdefault(uid, {
						"files": [], "modality": str(getattr(ds, "Modality", "") or ""),
						"desc": str(getattr(ds, "SeriesDescription", "") or ""),
						"protocol": str(getattr(ds, "ProtocolName", "") or ""),
						"image_type": " ".join(str(x) for x in (getattr(ds, "ImageType", None) or [])),
						"frames": int(getattr(ds, "NumberOfFrames", 1) or 1),
					})
					info["files"].append(fp)
				except Exception:
					continue
		if not series:
			QMessageBox.information(self, "SINCRO", "No se encontraron DICOMs en la carpeta.")
			return

		def _classify(info) -> str | None:
			text = f"{info['desc']} {info['protocol']} {info['image_type']} {os.path.basename(info['files'][0])}".lower()
			base_up = os.path.basename(info['files'][0]).upper()
			stem_up = os.path.splitext(base_up)[0]
			if "localizer" in text or "scout" in text:
				return None
			if any(k in text for k in ("atten", "attmap", "att map", "att_", "_att", "mu map", "mumap", "umap", "transmission")):
				return "att"
			if info["modality"].upper() == "CT":
				return "ct"
			if info["modality"].upper() == "NM" and "tomo" in text and "recon" not in text:
				# Ventana de scatter hermana (token _SC_ / _SC): rama propia.
				if "_SC_" in base_up or stem_up.endswith("_SC") or "scatter" in text:
					return "sc"
				return "raw"
			return None

		detected: dict[tuple, dict] = {}  # (kind, stage) -> info
		unresolved = []
		for uid, info in series.items():
			kind = _classify(info)
			if kind is None:
				continue
			info["uid"] = uid
			text = f"{info['desc']} {info['protocol']} {os.path.basename(info['files'][0])}"
			stage = self._stage_from_dicom_text(text)
			if stage is None:
				unresolved.append((kind, info))
				continue
			key = (kind, stage)
			# Ante duplicados: la serie con más cortes/frames gana.
			prev = detected.get(key)
			if prev is None or len(info["files"]) * info["frames"] > len(prev["files"]) * prev["frames"]:
				detected[key] = info
		# Resolver etapa por descarte: si de un tipo hay una etapa ocupada y una serie sin etapa.
		for kind, info in unresolved:
			for stage in ("stress", "rest"):
				if (kind, stage) not in detected:
					detected[(kind, stage)] = info
					self._log(f"[SMART] '{info['desc']}' sin etapa clara: asignada a {stage} por descarte.")
					break

		if not detected:
			QMessageBox.information(self, "SINCRO", "No se reconocieron series SPECT/ATT/CT en la carpeta (ver log).")
			return
		nombres = {"raw": "SPECT crudo", "att": "ATT map", "ct": "CT", "sc": "Scatter (SC)"}
		lines = []
		for (kind, stage), info in sorted(detected.items()):
			stage_txt = "Esfuerzo" if stage == "stress" else "Reposo"
			lines.append(f"• {nombres[kind]} {stage_txt}: '{info['desc'] or os.path.basename(info['files'][0])}' ({len(info['files'])} arch, {info['frames']} frames)")
		resp = QMessageBox.question(
			self, "SINCRO — Carpeta inteligente",
			"Se detectó:\n\n" + "\n".join(lines) + "\n\n¿Cargar todo?",
			QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
			QMessageBox.StandardButton.Yes,
		)
		if resp != QMessageBox.StandardButton.Yes:
			return

		# 1) SPECT crudos (solo si aún no hay estudio cargado: no pisar trabajo hecho).
		raw_s = detected.get(("raw", "stress"))
		raw_r = detected.get(("raw", "rest"))
		if self.study is None and (raw_s or raw_r):
			primary = raw_s or raw_r
			self.file_edit.setText(primary["files"][0])
			self.process_current()
			if raw_s and raw_r and self.study is not None and not bool(getattr(self.study, "reconstructed", True)):
				self._load_compare_raw_study_from_path(raw_r["files"][0])
		elif raw_s or raw_r:
			self._log("[SMART] Ya hay estudio cargado: los SPECT crudos detectados NO se recargan.")

		# 1b) Scatter: el loader lo adjunta solo si sigue el patrón _EM_/_SC_; si no,
		# adjuntarlo acá a mano y dejar la corrección ACTIVADA.
		for stage, study_obj in (("stress", self.study), ("rest", self._secondary_cine_crudo_study())):
			sc = detected.get(("sc", stage))
			if sc is None or study_obj is None:
				continue
			if getattr(study_obj, "scatter_projections", None) is not None:
				continue
			try:
				from core.raw_projections import load_raw_projections
				sc_raw = load_raw_projections(sc["files"][0], _skip_scatter=True)
				if np.asarray(sc_raw.projections).shape == np.asarray(study_obj.cube).shape:
					study_obj.scatter_projections = np.asarray(sc_raw.projections, dtype=np.float64)
					study_obj.scatter_path = sc["files"][0]
					self._log(f"[SMART] Scatter adjuntado a {stage}: {os.path.basename(sc['files'][0])}.")
				else:
					self._log(f"[SMART][WARN] SC de {stage} con shape distinto al EM: no se adjunta.")
			except Exception as exc:
				self._log(f"[SMART][WARN] No pude adjuntar SC de {stage}: {exc}")
		if any(detected.get(("sc", s)) for s in ("stress", "rest")) and getattr(self, "cine_crudo_scatter_check", None) is not None:
			if getattr(self.study, "scatter_projections", None) is not None or getattr(self._secondary_cine_crudo_study(), "scatter_projections", None) is not None:
				self.cine_crudo_scatter_check.setEnabled(True)
				self.cine_crudo_scatter_check.setChecked(True)
				self.cine_crudo_scatter_k_spin.setEnabled(True)
				self._log("[SMART] Corrección de scatter ACTIVADA (P = EM − k×SC).")

		# 2) CT y ATT por etapa: CT da display alta res; ATT (si existe) pisa el μ oficial.
		for stage in ("stress", "rest"):
			ct = detected.get(("ct", stage))
			att = detected.get(("att", stage))
			if ct is not None:
				self._load_ct_att_for_stage(stage, ct["files"][0], "ct", series_uid=ct["uid"])
			if att is not None:
				self._load_ct_att_for_stage(stage, att["files"][0], "att", mu_only=ct is not None)
		self.statusBar().showMessage("✓ Carpeta inteligente cargada (ver log para el detalle).", 10000)

	def _cine_crudo_active_stage_or_default(self) -> str:
		stage = str(getattr(self, "_cine_crudo_active_stage", "stress") or "stress")
		return stage if stage in ("stress", "rest") else "stress"

	def _stage_mu_map_for_recon(self, stage: str, projections: np.ndarray, raw_study=None):
		"""μ-map de la etapa remuestreado a la grilla de recon (H,W,W) o (None, None).

		Remuestreo por spacing + center crop; si hay una recon previa de la misma
		etapa con la misma grilla, refina la traslación por NCC contra el ungated
		(registro fino Fase 2). fill=0 (aire). Cachea el resultado para el QC.
		"""
		proj = np.asarray(projections)
		height, width = int(proj.shape[2]), int(proj.shape[3])
		st = self._dual_session().stage("rest" if str(stage) == "rest" else "stress")
		refine_to = None
		prev = getattr(st, "recon_result", None)
		if prev is not None:
			prev_ung = getattr(prev, "ungated_volume", None)
			if prev_ung is not None and tuple(np.asarray(prev_ung).shape) == (height, width, width):
				refine_to = np.asarray(prev_ung, dtype=np.float64)
		return self._stage_mu_map_to_grid(stage, (height, width, width), raw_study=raw_study, refine_to=refine_to)

	def _stage_mu_map_to_grid(self, stage: str, target_shape, raw_study=None, refine_to=None):
		"""Remuestrea (y opcionalmente registra por NCC) el μ-map de la etapa a target_shape."""
		from core.ct_fusion import resample_volume_to_spect_grid, validate_mu_map, refine_ct_to_spect_translation

		st = self._dual_session().stage("rest" if str(stage) == "rest" else "stress")
		mu_native = st.mu_map_native
		if mu_native is None:
			self._log(f"[AC][WARN] AC pedida pero la etapa {stage} no tiene CT/ATT cargado: reconstruyo SIN AC.")
			return None, None
		try:
			height, width = int(target_shape[0]), int(target_shape[1])
			raw_study = raw_study or st.raw_study_for_recon or st.raw_study or self.study
			ps = getattr(raw_study, "pixel_spacing", None) if raw_study is not None else None
			px_mm = float(ps[0]) if ps else 6.4
			target = np.zeros(tuple(int(v) for v in target_shape), dtype=np.float64)
			mu_rs, notes = resample_volume_to_spect_grid(
				np.asarray(mu_native, dtype=np.float64), target,
				source_spacing_zyx=st.mu_map_spacing_zyx,
				spect_spacing_zyx=(px_mm, px_mm, px_mm),
				fill_value=0.0, order=1,
			)
			for n in notes:
				self._log(f"[AC] {n}")
			flips = getattr(st, "mu_map_flip_zyx", None)
			if flips and any(flips):
				for axis_i, do_flip in enumerate(flips):
					if do_flip:
						mu_rs = np.flip(mu_rs, axis=axis_i)
				mu_rs = np.ascontiguousarray(mu_rs)
				self._log(f"[AC] Espejos del CT aplicados al μ-map (visor de fusión): z/y/x={flips}.")
			manual = getattr(st, "mu_map_manual_shift_zyx", None)
			if manual is not None:
				# Registro explícito (visor de fusión o import del panel): NCC NUNCA
				# debe re-desplazarlo — incluso con Δ=(0,0,0) (μ-map YA registrado).
				if any(abs(float(v)) > 1e-6 for v in manual):
					import scipy.ndimage as ndi_local
					mu_rs = ndi_local.shift(mu_rs, shift=tuple(float(v) for v in manual), order=1, mode="nearest")
				st.mu_map_shift_zyx = tuple(float(v) for v in manual)
				self._log(f"[AC] Registro explícito aplicado: Δ z/y/x={manual} vox. NCC omitido.")
			elif refine_to is not None and np.asarray(refine_to).shape == mu_rs.shape:
				try:
					mu_rs, shift, rnotes = refine_ct_to_spect_translation(
						mu_rs, np.asarray(refine_to, dtype=np.float64),
						search_radius_zyx=(3, 6, 6),
					)
					st.mu_map_shift_zyx = tuple(float(v) for v in shift)
					for n in rnotes:
						self._log(f"[AC] {n}")
				except Exception as exc:
					self._log(f"[AC][WARN] Refinamiento NCC falló ({exc}); uso remuestreo por spacing solo.")
			else:
				st.mu_map_shift_zyx = None
				self._log("[AC] Sin recon previa de la etapa: remuestreo por spacing sin registro fino (se refina en la próxima recon).")
			ok, qc_notes = validate_mu_map(mu_rs)
			for n in qc_notes:
				self._log(f"[AC] {n}")
			if not ok:
				self._log("[AC][WARN] μ-map remuestreado no pasó QC: reconstruyo SIN AC.")
				return None, None
			st.mu_map_recon_grid = mu_rs
			self._log(
				f"[AC] μ-map {stage} listo en grilla de recon {mu_rs.shape} "
				f"(px={px_mm:.2f} mm). Verificá la alineación con 'QC AC'."
			)
			return mu_rs, px_mm / 10.0
		except Exception as exc:
			self._log(f"[AC][WARN] Remuestreo del μ-map falló ({exc}): reconstruyo SIN AC.")
			return None, None

	def _show_ac_qc(self):
		"""QC visual DUAL de alineación μ-map ↔ recon (esfuerzo+reposo). Toggle: 2do click restaura."""
		if getattr(self, "cine_crudo_preview_mode", None) == "ac_qc":
			prev = getattr(self, "_ac_qc_prev_state", None) or {}
			self._ac_qc_prev_state = None
			self.cine_crudo_preview_mode = prev.get("mode")
			self._ac_qc_active = False
			self._refresh_ac_qc_btn_state()
			pix = prev.get("pix")
			if pix is not None and not pix.isNull() and "cine_crudo" in self.preview_labels:
				self.preview_pixmaps["cine_crudo"] = pix
				self.preview_base_sizes["cine_crudo"] = pix.size()
				self._apply_preview_zoom("cine_crudo")
			else:
				self._refresh_cine_crudo_view()
			return
		# Reunir las etapas que tienen recon + μ-map (dual si ambas están listas).
		panels = []
		for stage in ("stress", "rest"):
			st = self._dual_session().stage(stage)
			result = getattr(st, "recon_result", None)
			if result is None:
				continue
			ung = np.asarray(result.ungated_volume, dtype=np.float64)
			mu = getattr(st, "mu_map_recon_grid", None)
			if mu is None or np.asarray(mu).shape != ung.shape:
				mu, _ = self._stage_mu_map_to_grid(stage, ung.shape, refine_to=ung)
			if mu is None:
				continue
			panels.append((stage, ung, np.asarray(mu, dtype=np.float64)))
		if not panels:
			QMessageBox.information(self, "SINCRO", "Primero reconstruí con 'Recon raw' y cargá CT/ATT para tener QC.")
			return
		self._ac_qc_panels = panels
		self._ac_qc_frac = float(getattr(self, "_ac_qc_frac", 0.5))
		self._ac_qc_active = True
		self._render_ac_qc()
		self._refresh_ac_qc_btn_state()
		self._select_tab_by_title("cine_crudo")
		self._log(f"[AC] QC DUAL generado ({len(panels)} etapa/s). Rueda del mouse = navegar cortes.")

	def _refresh_ac_qc_btn_state(self):
		btn = getattr(self, "cine_crudo_ac_qc_btn", None)
		if btn is None:
			return
		active = bool(getattr(self, "_ac_qc_active", False))
		btn.setStyleSheet("background-color:#15803d; color:white; font-weight:bold;" if active else "")

	def _render_ac_qc(self):
		"""Dibuja el QC AC de todas las etapas al frac de corte actual (navegable con rueda)."""
		panels = getattr(self, "_ac_qc_panels", None)
		if not panels:
			return
		try:
			import matplotlib.pyplot as plt
			frac = float(np.clip(getattr(self, "_ac_qc_frac", 0.5), 0.02, 0.98))
			nrows = len(panels)
			fig, axes = plt.subplots(nrows, 3, figsize=(12, 4.2 * nrows), squeeze=False)
			fig.patch.set_facecolor("#0b1220")
			for r, (stage, ung, mu) in enumerate(panels):
				zc = int(np.clip(round(frac * (ung.shape[0] - 1)), 0, ung.shape[0] - 1))
				yc = int(np.clip(round(frac * (ung.shape[1] - 1)), 0, ung.shape[1] - 1))
				xc = int(np.clip(round(frac * (ung.shape[2] - 1)), 0, ung.shape[2] - 1))
				views = [
					("Axial", ung[zc], mu[zc]),
					("Coronal", ung[:, yc, :], mu[:, yc, :]),
					("Sagital", ung[:, :, xc], mu[:, :, xc]),
				]
				st = self._dual_session().stage(stage)
				shift = getattr(st, "mu_map_shift_zyx", None)
				stage_txt = "ESFUERZO" if stage == "stress" else "REPOSO"
				for c, (title, sp_sl, mu_sl) in enumerate(views):
					ax = axes[r][c]
					sp_n = sp_sl / max(float(np.percentile(sp_sl, 99.5)), 1e-9)
					ax.imshow(np.clip(sp_n, 0, 1), cmap="gray", interpolation="bicubic")
					ax.contour(mu_sl, levels=[0.05], colors=["cyan"], linewidths=1.0)
					ax.contour(mu_sl, levels=[0.13], colors=["orange"], linewidths=0.8)
					lbl = f"{stage_txt} · {title}" if c == 0 else title
					ax.set_title(lbl, color="white", fontsize=10)
					ax.axis("off")
			fig.suptitle(
				f"QC AC · contornos μ-map (cian=cuerpo, naranja=denso) sobre recon ungated · corte {int(frac * 100)}% "
				"— rueda del mouse para navegar",
				color="white", fontsize=11,
			)
			fig.tight_layout(rect=[0, 0, 1, 0.95])
			out_png = os.path.join(self.output_dir, "ac_qc_dual.png")
			fig.savefig(out_png, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
			plt.close(fig)
			if "cine_crudo" in self.preview_labels:
				if getattr(self, "cine_crudo_preview_mode", None) != "ac_qc":
					self._ac_qc_prev_state = {
						"mode": getattr(self, "cine_crudo_preview_mode", None),
						"pix": self.preview_pixmaps.get("cine_crudo"),
					}
				pix = QPixmap(out_png)
				self.preview_pixmaps["cine_crudo"] = pix
				self.preview_base_sizes["cine_crudo"] = pix.size()
				self.cine_crudo_preview_mode = "ac_qc"
				self._set_preview_zoom("cine_crudo", 1.0)
				self._apply_preview_zoom("cine_crudo")
		except Exception as exc:
			self._log(f"[AC][WARN] QC AC falló: {exc}")

	def _stage_ct_on_recon_grid(self, stage: str, target_shape):
		"""CT de display de la etapa remuestreado (+Δ NCC si existe) a target_shape, o None."""
		import scipy.ndimage as ndi_local
		from core.ct_fusion import resample_volume_to_spect_grid

		st = self._dual_session().stage("rest" if str(stage) == "rest" else "stress")
		ct_native = getattr(st, "ct_volume_native", None)
		if ct_native is None:
			ct_native = getattr(st, "mu_map_native", None)
		if ct_native is None:
			return None
		raw_study = st.raw_study_for_recon or st.raw_study or self.study
		ps = getattr(raw_study, "pixel_spacing", None) if raw_study is not None else None
		px_mm = float(ps[0]) if ps else 6.4
		ct = np.asarray(ct_native, dtype=np.float64)
		ct_rs, _notes = resample_volume_to_spect_grid(
			ct, np.zeros(tuple(int(v) for v in target_shape), dtype=np.float64),
			source_spacing_zyx=getattr(st, "ct_spacing_zyx", None) or st.mu_map_spacing_zyx,
			spect_spacing_zyx=(px_mm, px_mm, px_mm),
			fill_value=float(np.min(ct)), order=1,
		)
		flips = getattr(st, "mu_map_flip_zyx", None)
		if flips and any(flips):
			for axis_i, do_flip in enumerate(flips):
				if do_flip:
					ct_rs = np.flip(ct_rs, axis=axis_i)
			ct_rs = np.ascontiguousarray(ct_rs)
		shift = getattr(st, "mu_map_manual_shift_zyx", None) or getattr(st, "mu_map_shift_zyx", None)
		if shift:
			ct_rs = ndi_local.shift(ct_rs, shift=tuple(float(v) for v in shift), order=1, mode="nearest")
		return ct_rs

	def _refresh_fusion_btn_state(self):
		"""Fusión se habilita cuando ALGUNA etapa tiene SPECT reconstruido Y CT/ATT."""
		btn = getattr(self, "cine_crudo_fusion_btn", None)
		if btn is None:
			return
		ok = False
		try:
			for stg in ("stress", "rest"):
				st = self._dual_session().stage(stg)
				has_ct = getattr(st, "ct_volume_native", None) is not None or getattr(st, "mu_map_native", None) is not None
				if getattr(st, "recon_result", None) is not None and has_ct:
					ok = True
					break
		except Exception:
			ok = False
		btn.setEnabled(ok)

	def _show_ct_fusion_preview(self):
		"""Abre el panel de fusión SPECT/CT de PERFUSIÓN (copia adaptada del de AMYLO).

		Flujo correcto (igual que AMYLO): recon del SPECT → superponer/registrar el
		CT sobre los cortes reconstruidos (tomo o cardiacos) → 'Reg→AC' trae ese
		registro a la etapa → re-Recon raw con AC aplica la corrección física.
		"""
		try:
			from ui.perfusion_fusion_panel import PerfusionFusionPanel
		except Exception as exc:
			QMessageBox.warning(self, "SINCRO", f"No se pudo abrir la ventana de fusión:\n{exc}")
			return
		stage = str(getattr(self, "_cine_crudo_recon_stage", None) or self._cine_crudo_active_stage_or_default())
		if stage not in ("stress", "rest"):
			stage = "stress"
		st = self._dual_session().stage(stage)
		result = getattr(st, "recon_result", None)
		ct_native = getattr(st, "ct_volume_native", None)
		if ct_native is None:
			ct_native = getattr(st, "mu_map_native", None)
		if result is None or ct_native is None:
			QMessageBox.information(
				self, "SINCRO",
				"La fusión necesita el SPECT RECONSTRUIDO y el CT/ATT de la etapa:\n"
				"1) Recon raw  2) botón CT/ATT. Con ambos cargados se habilita Fusión.",
			)
			return
		panel = getattr(self, "_perfusion_fusion_panel", None)
		if panel is None:
			# Top-level SIN parent: minimiza normal en Windows (regla Qt owned-window).
			panel = PerfusionFusionPanel(None)
			self._perfusion_fusion_panel = panel
		panel._perfusion_apply_cb = self._import_amylo_fusion_registration
		panel._perfusion_stage_change_cb = self._on_fusion_panel_stage_changed
		raw_study = st.raw_study_for_recon or st.raw_study or self.study
		ps = getattr(raw_study, "pixel_spacing", None) if raw_study is not None else None
		px_mm = float(ps[0]) if ps else 6.4
		spect_sigs = getattr(panel, "_perfusion_spect_sig", None)
		if spect_sigs is None:
			spect_sigs = {}
			panel._perfusion_spect_sig = spect_sigs
		has_session = (
			stage in getattr(panel, "_perfusion_stage_sessions", {})
			or getattr(panel, "_perfusion_current_stage", None) == stage
		)
		if has_session:
			# La sesión en memoria de la etapa MANDA (registro/máscara/nudges del
			# usuario intactos). Solo refrescar el SPECT si hay recon nueva (AC).
			try:
				if spect_sigs.get(stage) != id(result):
					panel.update_perfusion_spect(stage, result.ungated_volume, (px_mm, px_mm, px_mm))
					spect_sigs[stage] = id(result)
					self._log(f"[FUSION] SPECT de {stage} refrescado en el panel (registro CT conservado).")
				if getattr(panel, "_perfusion_current_stage", None) != stage:
					idx = panel._perfusion_stage_combo.findData(stage)
					if idx >= 0:
						panel._perfusion_stage_combo.setCurrentIndex(idx)  # dispara restore de sesión
			except Exception as exc:
				self._log(f"[FUSION][WARN] Refresh de sesión falló: {exc}")
		else:
			try:
				_t0 = perf_counter()
				panel.set_perfusion_inputs(
					spect_volume=result.ungated_volume,
					spect_spacing_zyx=(px_mm, px_mm, px_mm),
					ct_volume=ct_native,
					ct_spacing_zyx=getattr(st, "ct_spacing_zyx", None) or getattr(st, "mu_map_spacing_zyx", None),
					ct_affine=getattr(st, "ct_affine_ijk_to_lps", None),
					cardiac_axes=getattr(st, "axes_ungated", None) or None,
					stage=stage,
					source_label=str(getattr(st, "source_path", "") or ""),
				)
				self._log(f"[PERF] precarga panel de fusión ({stage}): {perf_counter() - _t0:.1f}s")
				spect_sigs[stage] = id(result)
				if getattr(st, "ct_spacing_zyx", None) is None and np.asarray(ct_native).shape[1] > 128:
					self._log("[FUSION][WARN] CT de alta resolución SIN spacing propio guardado: el remuestreo puede "
						"quedar mal posicionado. Recargá el CT con el botón CT/ATT.")
				if getattr(st, "ct_affine_ijk_to_lps", None) is None:
					self._log("[FUSION][WARN] El CT de la etapa NO tiene affine DICOM guardado (cargado antes del fix): "
						"recargalo con el botón CT/ATT para que 'CT nativa' y la CT registrada tengan la MISMA orientación.")
				self._log(
					f"[FUSION] SPECT reconstruido + CT de {stage} transferidos al panel; "
					"registro automático ejecutado — la fusión queda visible de entrada."
				)
			except Exception as exc:
				self._log(f"[FUSION][WARN] No pude precargar SPECT+CT en el panel: {exc}")
		try:
			panel.set_perfusion_header(stage, self._patient_banner_text(stage=stage))
		except Exception:
			pass
		panel.show()
		panel.raise_()
		panel.activateWindow()

	def _on_fusion_panel_stage_changed(self, stage: str):
		"""El combo Etapa del panel de fusión pidió cargar la otra etapa."""
		panel = getattr(self, "_perfusion_fusion_panel", None)
		if panel is None:
			return
		stage = "rest" if str(stage) == "rest" else "stress"
		st = self._dual_session().stage(stage)
		result = getattr(st, "recon_result", None)
		ct_native = getattr(st, "ct_volume_native", None)
		if ct_native is None:
			ct_native = getattr(st, "mu_map_native", None)
		if result is None or ct_native is None:
			stage_txt = "REPOSO" if stage == "rest" else "ESFUERZO"
			faltan = []
			if result is None:
				faltan.append("reconstruir (Recon raw)")
			if ct_native is None:
				faltan.append("cargar CT/ATT (botón CT/ATT, eligiendo esa etapa)")
			panel._status.setText(f"Etapa {stage_txt}: falta {' y '.join(faltan)} en el flujo de perfusión.")
			self._log(f"[FUSION] Cambio a {stage_txt} rechazado: falta {' y '.join(faltan)}.")
			return
		try:
			raw_study = st.raw_study_for_recon or st.raw_study or self.study
			ps = getattr(raw_study, "pixel_spacing", None) if raw_study is not None else None
			px_mm = float(ps[0]) if ps else 6.4
			panel.set_perfusion_inputs(
				spect_volume=result.ungated_volume,
				spect_spacing_zyx=(px_mm, px_mm, px_mm),
				ct_volume=ct_native,
				ct_spacing_zyx=getattr(st, "ct_spacing_zyx", None) or getattr(st, "mu_map_spacing_zyx", None),
				ct_affine=getattr(st, "ct_affine_ijk_to_lps", None),
				cardiac_axes=getattr(st, "axes_ungated", None) or None,
				stage=stage,
				source_label=str(getattr(st, "source_path", "") or ""),
			)
			sigs = getattr(panel, "_perfusion_spect_sig", None)
			if sigs is None:
				sigs = {}
				panel._perfusion_spect_sig = sigs
			sigs[stage] = id(result)
			try:
				panel.set_perfusion_header(stage, self._patient_banner_text(stage=stage))
			except Exception:
				pass
			self._log(f"[FUSION] Panel cambiado a etapa {stage}: SPECT+CT cargados y registrados (primera vez).")
		except Exception as exc:
			self._log(f"[FUSION][WARN] Cambio de etapa falló: {exc}")

	def _suggest_feta_limits_from_fusion(self, stage: str, margin: int = 2) -> None:
		"""Pre-carga los límites Base/Ápex de la feta desde la máscara del corazón
		ya fusionada (rango axial Z + margen). Es una sugerencia editable: no
		reconstruye ni pisa una selección que el usuario haya tocado a mano."""
		panel = getattr(self, "_perfusion_fusion_panel", None)
		if panel is None or not hasattr(panel, "get_heart_axial_bounds"):
			return
		bounds = panel.get_heart_axial_bounds()
		if not bounds:
			return
		z_min, z_max = int(bounds[0]), int(bounds[1])
		try:
			res = self._dual_session().stage(stage).recon_result
			n = int(np.asarray(res.gated_volume).shape[1]) if res is not None else None
		except Exception:
			n = None
		base_1 = z_min + 1 - margin
		apex_1 = z_max + 1 + margin
		if n:
			base_1 = int(np.clip(base_1, 1, n))
			apex_1 = int(np.clip(apex_1, 1, n))
		else:
			base_1 = max(1, base_1)
			apex_1 = max(1, apex_1)
		b, a = self._cine_crudo_stage_limits_set(stage, base_1, apex_1, n)
		# Reflejar en los spins si esa etapa es la activa en pantalla.
		if str(getattr(self, "_cine_crudo_recon_stage", "stress")) == stage:
			for spin_name, val in (("cine_crudo_cut_base_spin", b), ("cine_crudo_cut_apex_spin", a)):
				spin = getattr(self, spin_name, None)
				if spin is not None:
					spin.blockSignals(True)
					spin.setValue(int(val))
					spin.blockSignals(False)
		stage_txt = "REPOSO" if stage == "rest" else "ESFUERZO"
		self._log(
			f"[FETA] Sugerencia de límites Base/Ápex para {stage_txt} desde la máscara "
			f"fusionada: z=[{b},{a}] (margen ±{margin}). Editable; no re-reconstruye solo."
		)

	def _import_amylo_fusion_registration(self, stage: str | None = None):
		"""Trae el CT/ATT registrado en el panel de fusión a la etapa indicada (para AC)."""
		from core.ct_fusion import mu_map_from_ct_hu, validate_mu_map
		panel = getattr(self, "_perfusion_fusion_panel", None)
		if panel is None:
			QMessageBox.information(self, "SINCRO", "Primero abrí 'Fusión' y registrá el CT ahí.")
			return
		if stage not in ("stress", "rest"):
			stage = str(getattr(self, "_cine_crudo_recon_stage", None) or self._cine_crudo_active_stage_or_default())
		if stage not in ("stress", "rest"):
			stage = "stress"
		self._log(f"[AC] Importando registro del panel de fusión hacia {stage}...")
		try:
			att = getattr(panel, "_att_map_registered", None)
			ct = getattr(panel, "_ct_registered", None)
			conv_notes: list[str] = []
			if att is not None:
				mu = np.clip(np.asarray(att, dtype=np.float64), 0.0, None)
				q99 = float(np.percentile(mu, 99.0)) if mu.size else 0.0
				if q99 > 2.0:
					mu = mu / q99 * 0.154
					conv_notes.append(f"ATT registrado reescalado: p99={q99:.1f} → 0.154/cm.")
				source = "att_export"
			elif ct is not None:
				mu, conv_notes = mu_map_from_ct_hu(np.asarray(ct, dtype=np.float64))
				source = "ct_bilineal"
			else:
				QMessageBox.information(self, "SINCRO", "La ventana de fusión no tiene CT/ATT registrado todavía.")
				return
			ok, qc_notes = validate_mu_map(mu)
			for n in conv_notes + qc_notes:
				self._log(f"[AC] {n}")
			if not ok:
				QMessageBox.warning(self, "SINCRO", "El μ-map registrado no pasó el QC (ver log).")
				return
			st = self._dual_session().stage(stage)
			# ¿Cambió realmente el registro? Solo saltear la re-recon si además la
			# recon vigente YA se hizo con AC (si no, hay que reconstruir igual).
			prev_mu = getattr(st, "mu_map_native", None)
			recon_has_ac = bool(getattr(getattr(getattr(st, "recon_result", None), "config", None), "attenuation_correction", False))
			unchanged = (
				recon_has_ac
				and prev_mu is not None
				and np.asarray(prev_mu).shape == mu.shape
				and bool(np.allclose(np.asarray(prev_mu), mu, atol=1e-9))
			)
			st.mu_map_native = mu
			# NO pisar el CT display nativo (alta resolución + affine) con la versión
			# registrada de baja resolución: solo si la etapa no tenía CT.
			if getattr(st, "ct_volume_native", None) is None:
				st.ct_volume_native = np.asarray(ct, dtype=np.float64) if ct is not None else mu
			# Preservar el spacing NATIVO del CT antes de pisar el del μ-map con el
			# de la grilla SPECT (si no, el próximo preload remuestrea el CT nativo
			# con spacing equivocado y "queda en cualquier lado").
			if getattr(st, "ct_spacing_zyx", None) is None:
				st.ct_spacing_zyx = getattr(st, "mu_map_spacing_zyx", None)
			sp_spacing = getattr(panel, "_spect_spacing_zyx", None)
			st.mu_map_spacing_zyx = tuple(float(v) for v in sp_spacing) if sp_spacing else None
			st.mu_map_source = source
			st.mu_map_description = "registrado en ventana de fusión AMYLO"
			# Ya viene registrado sobre la grilla SPECT: sin NCC ni ajustes extra.
			st.mu_map_manual_shift_zyx = (0.0, 0.0, 0.0)
			st.mu_map_flip_zyx = None
			st.mu_map_recon_grid = None
			if getattr(self, "cine_crudo_ac_check", None) is not None:
				self.cine_crudo_ac_check.setEnabled(True)
				self.cine_crudo_ac_check.setChecked(True)
			# Sugerir límites Base/Ápex de la feta desde la máscara ya fusionada
			# (editable; no fuerza re-recon). Solo con CT/fusión presente.
			try:
				self._suggest_feta_limits_from_fusion(stage)
			except Exception as exc:
				self._log(f"[FETA][WARN] No pude sugerir límites desde la fusión: {exc}")
			if unchanged:
				self._log(f"[AC] Registro importado idéntico al vigente en {stage}: no re-reconstruyo.")
				return
			self._log(
				f"[AC] Registro importado del panel para {stage}: μ-map {mu.shape} "
				f"({source}), spacing={st.mu_map_spacing_zyx}. Reconstruyendo con AC automáticamente..."
			)
			# La AC física vive DENTRO del motor iterativo: FBP la ignoraría y "no pasa nada".
			try:
				for _cname in ("cine_crudo_recon_method_combo", "cine_crudo_gated_method_combo"):
					combo = getattr(self, _cname, None)
					if combo is not None and str(combo.currentText()).strip().lower() == "fbp":
						combo.setCurrentText("OSEM")
						_rama = "gated" if "gated" in _cname else "ungated"
						self._log(f"[AC] Método {_rama} FBP no modela AC: cambiado automáticamente a OSEM.")
			except Exception:
				pass
			try:
				self._reconstruct_cine_crudo_raw(_force_stage=stage)
			except Exception as exc:
				self._log(f"[AC][WARN] Re-recon automática falló ({exc}): tocá 'Recon raw' manualmente.")
			else:
				stage_txt = "ESFUERZO" if stage == "stress" else "REPOSO"
				self.statusBar().showMessage(
					f"✓ AC aplicada a {stage_txt}: recon OSEM con μ-map registrado en la fusión.", 15000
				)
				# Si la OTRA etapa está lista y aún sin AC: volver al panel apuntando ahí.
				other = "rest" if stage == "stress" else "stress"
				ost = self._dual_session().stage(other)
				other_ready = getattr(ost, "recon_result", None) is not None and (
					getattr(ost, "ct_volume_native", None) is not None
					or getattr(ost, "mu_map_native", None) is not None
				)
				applied = getattr(panel, "_perfusion_ac_applied", None) or set()
				try:
					if other_ready and other not in applied:
						idx = panel._perfusion_stage_combo.findData(other)
						if idx >= 0:
							panel._perfusion_stage_combo.setCurrentIndex(idx)
						panel.show()
						panel.raise_()
						panel.activateWindow()
						other_txt = "REPOSO" if other == "rest" else "ESFUERZO"
						panel._status.setText(f"Etapa {other_txt}: registrá/verificá la fusión y aplicá el paso 7 acá también.")
					else:
						panel.close()
				except Exception:
					pass
		except Exception as exc:
			QMessageBox.warning(self, "SINCRO", f"No se pudo importar el registro:\n{exc}")

	def _show_ct_fusion_preview_static(self, stage: str | None = None):
		"""Render estático de fusión en grilla de recon (PNG para preview/informe)."""
		import scipy.ndimage as ndi_local

		if stage is None:
			stage = str(getattr(self, "_cine_crudo_recon_stage", None) or self._cine_crudo_active_stage_or_default())
		if stage not in ("stress", "rest"):
			stage = "stress"
		st = self._dual_session().stage(stage)
		result = getattr(st, "recon_result", None) or getattr(self, "cine_crudo_recon_result", None)
		if result is None:
			QMessageBox.information(self, "SINCRO", "Primero reconstruí con 'Recon raw'.")
			return
		ct_native = getattr(st, "ct_volume_native", None)
		if ct_native is None:
			ct_native = getattr(st, "mu_map_native", None)
		if ct_native is None:
			QMessageBox.information(self, "SINCRO", "Cargá CT/ATT para esta etapa primero.")
			return
		try:
			from core.ct_fusion import resample_volume_to_spect_grid
			import matplotlib.pyplot as plt
			import matplotlib as mpl

			ung = np.asarray(result.ungated_volume, dtype=np.float64)
			raw_study = st.raw_study_for_recon or st.raw_study or self.study
			ps = getattr(raw_study, "pixel_spacing", None) if raw_study is not None else None
			px_mm = float(ps[0]) if ps else 6.4
			ct = np.asarray(ct_native, dtype=np.float64)
			ct_rs, notes = resample_volume_to_spect_grid(
				ct, np.zeros_like(ung),
				source_spacing_zyx=st.mu_map_spacing_zyx,
				spect_spacing_zyx=(px_mm, px_mm, px_mm),
				fill_value=float(np.min(ct)), order=1,
			)
			for n in notes:
				self._log(f"[FUSION] {n}")
			shift = getattr(st, "mu_map_shift_zyx", None)
			if shift:
				ct_rs = ndi_local.shift(ct_rs, shift=tuple(float(v) for v in shift), order=1, mode="nearest")
				self._log(f"[FUSION] Aplicado Δ del registro NCC: z/y/x={shift}.")
			# Ventana anatómica: HU tejido blando si es CT; percentiles si es μ.
			if float(np.min(ct_rs)) < -200.0:
				ct_disp = np.clip((ct_rs + 200.0) / 500.0, 0.0, 1.0)
			else:
				lo, hi = np.percentile(ct_rs, (1.0, 99.5))
				ct_disp = np.clip((ct_rs - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
			sp_n = np.clip(ung / max(float(np.percentile(ung, 99.5)), 1e-9), 0.0, 1.0)
			try:
				cmap = mpl.colormaps[str(getattr(self, "cine_crudo_screen_cmap", "") or "hot")]
			except Exception:
				cmap = mpl.colormaps["hot"]

			zs, ys, xs = ung.shape
			fracs = (0.35, 0.50, 0.65)
			rows = [
				("Axial", [(ct_disp[int(f * zs)], sp_n[int(f * zs)]) for f in fracs]),
				("Coronal", [(ct_disp[:, int(f * ys), :], sp_n[:, int(f * ys), :]) for f in fracs]),
				("Sagital", [(ct_disp[:, :, int(f * xs)], sp_n[:, :, int(f * xs)]) for f in fracs]),
			]
			fig, axes = plt.subplots(3, 3, figsize=(11, 11))
			fig.patch.set_facecolor("#0b1220")
			for r, (title, panels) in enumerate(rows):
				for c, (ct_sl, sp_sl) in enumerate(panels):
					ax = axes[r][c]
					base = np.stack([ct_sl] * 3, axis=-1)
					over = cmap(sp_sl)[..., :3]
					alpha = (np.clip(sp_sl, 0.0, 1.0) ** 0.7 * 0.65)[..., None]
					ax.imshow(np.clip(base * (1 - alpha) + over * alpha, 0, 1), interpolation="bicubic")
					if c == 0:
						ax.set_ylabel(title, color="white", fontsize=10)
					ax.set_xticks([]); ax.set_yticks([])
					for s in ax.spines.values():
						s.set_color("#334155")
			stage_txt = "ESFUERZO" if stage == "stress" else "REPOSO"
			src_txt = "CT (HU)" if float(np.min(np.asarray(ct_native))) < -200.0 else "ATT (μ)"
			fig.suptitle(
				f"Fusión SPECT/CT · {stage_txt} · anatomía={src_txt} · grilla recon {ung.shape}",
				color="white", fontsize=12,
			)
			fig.tight_layout(rect=[0, 0, 1, 0.95])
			out_png = os.path.join(self.output_dir, f"ct_fusion_{stage}.png")
			fig.savefig(out_png, dpi=125, bbox_inches="tight", facecolor=fig.get_facecolor())
			plt.close(fig)
			if "cine_crudo" in self.preview_labels:
				pix = QPixmap(out_png)
				self.preview_pixmaps["cine_crudo"] = pix
				self.preview_base_sizes["cine_crudo"] = pix.size()
				self.cine_crudo_preview_mode = "ct_fusion"
				self._set_preview_zoom("cine_crudo", 1.0)
				self._apply_preview_zoom("cine_crudo")
				self._select_tab_by_title("cine_crudo")
			self._log(f"[FUSION] Fusión SPECT/CT generada: {out_png}")
		except Exception as exc:
			self._log(f"[FUSION][WARN] Fusión falló: {exc}")

	def _render_sa_fusion(self, stage: str, axes_ct: dict, axes_perf: dict):
		"""Fusión en ejes cardiacos: perfusión SA (color) sobre CT SA (gris)."""
		import matplotlib.pyplot as plt
		import matplotlib as mpl

		sa_ct = np.asarray(axes_ct["SA"], dtype=np.float64)
		sa_pf = np.asarray(axes_perf["SA"], dtype=np.float64)
		if sa_ct.ndim == 4:
			sa_ct = sa_ct[0]
		if sa_pf.ndim == 4:
			sa_pf = sa_pf.sum(axis=0) if sa_pf.shape[0] > 1 else sa_pf[0]
		n = min(int(sa_ct.shape[0]), int(sa_pf.shape[0]))
		if n < 1:
			raise ValueError("sin cortes SA")
		if float(np.min(sa_ct)) < -200.0:
			ct_disp = np.clip((sa_ct + 200.0) / 500.0, 0.0, 1.0)
		else:
			lo, hi = np.percentile(sa_ct, (1.0, 99.5))
			ct_disp = np.clip((sa_ct - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
		pf_n = np.clip(sa_pf / max(float(np.percentile(sa_pf, 99.5)), 1e-9), 0.0, 1.0)
		try:
			cmap = mpl.colormaps[str(getattr(self, "cine_crudo_screen_cmap", "") or "hot")]
		except Exception:
			cmap = mpl.colormaps["hot"]
		cols = 4
		show_idx = list(range(n))[:12]
		rows_n = int(np.ceil(len(show_idx) / cols))
		fig, axes = plt.subplots(rows_n, cols, figsize=(3.0 * cols, 3.0 * rows_n))
		fig.patch.set_facecolor("#0b1220")
		axes = np.atleast_2d(axes)
		for i, ax in enumerate(axes.flat):
			ax.axis("off")
			if i >= len(show_idx):
				continue
			k = show_idx[i]
			base = np.stack([ct_disp[k]] * 3, axis=-1)
			over = cmap(pf_n[k])[..., :3]
			alpha = (np.clip(pf_n[k], 0.0, 1.0) ** 0.7 * 0.65)[..., None]
			ax.imshow(np.clip(base * (1 - alpha) + over * alpha, 0, 1), interpolation="bicubic")
			ax.set_title(f"SA {k + 1}", color="#94a3b8", fontsize=8)
		stage_txt = "ESFUERZO" if stage == "stress" else "REPOSO"
		fig.suptitle(
			f"Fusión en ejes cardiacos · {stage_txt} · perfusión SA sobre CT (base→ápex)",
			color="white", fontsize=12,
		)
		fig.tight_layout(rect=[0, 0, 1, 0.94])
		out_png = os.path.join(self.output_dir, f"ct_fusion_sa_{stage}.png")
		fig.savefig(out_png, dpi=125, bbox_inches="tight", facecolor=fig.get_facecolor())
		plt.close(fig)
		if "cine_crudo" in self.preview_labels:
			pix = QPixmap(out_png)
			self.preview_pixmaps["cine_crudo"] = pix
			self.preview_base_sizes["cine_crudo"] = pix.size()
			self.cine_crudo_preview_mode = "ct_fusion"
			self._set_preview_zoom("cine_crudo", 1.0)
			self._apply_preview_zoom("cine_crudo")
			self._select_tab_by_title("cine_crudo")
		self._log(f"[FUSION] Fusión SA generada: {out_png}")

	def _load_iter_cfg_by_stage(self):
		"""Recupera la config de iteraciones por estudio persistida en QSettings."""
		cfgs = getattr(self, "_iter_cfg_by_stage", None)
		if cfgs is not None:
			return cfgs
		g_it = int(self.cine_crudo_iter_spin.value()) if hasattr(self, "cine_crudo_iter_spin") else 8
		g_su = int(self.cine_crudo_osem_subsets_spin.value()) if hasattr(self, "cine_crudo_osem_subsets_spin") else 4
		cfgs = {(s, b): [g_it, g_su] for s in ("stress", "rest") for b in ("ungated", "gated")}
		try:
			blob = self._ui_settings.value("recon/iter_cfg_by_stage", "")
			if blob:
				stored = json.loads(blob)
				for k, v in stored.items():
					s, b = k.split("|", 1)
					if (s, b) in cfgs and isinstance(v, (list, tuple)) and len(v) == 2:
						cfgs[(s, b)] = [int(v[0]), int(v[1])]
		except Exception:
			pass
		self._iter_cfg_by_stage = cfgs
		return cfgs

	def _save_iter_cfg_by_stage(self):
		"""Persiste la config de iteraciones por estudio en QSettings."""
		try:
			cfgs = getattr(self, "_iter_cfg_by_stage", None) or {}
			stored = {f"{s}|{b}": [int(v[0]), int(v[1])] for (s, b), v in cfgs.items()}
			self._ui_settings.setValue("recon/iter_cfg_by_stage", json.dumps(stored))
		except Exception:
			pass

	def _open_iter_config_dialog(self):
		"""Iteraciones/subsets POR ESTUDIO: ungated/gated × esfuerzo/reposo."""
		cfgs = self._load_iter_cfg_by_stage()
		dlg = QDialog(self)
		dlg.setWindowTitle("SINCRO — Iteraciones por estudio (OSEM/MLEM)")
		grid = QGridLayout(dlg)
		grid.addWidget(QLabel("<b>Estudio</b>"), 0, 0)
		grid.addWidget(QLabel("<b>Iter</b>"), 0, 1)
		grid.addWidget(QLabel("<b>Subsets</b>"), 0, 2)
		rows = [
			("stress", "ungated", "Esfuerzo · Ungated"),
			("stress", "gated", "Esfuerzo · Gated"),
			("rest", "ungated", "Reposo · Ungated"),
			("rest", "gated", "Reposo · Gated"),
		]
		spins = {}
		for r, (s, b, label) in enumerate(rows, start=1):
			grid.addWidget(QLabel(label), r, 0)
			it_spin = QSpinBox(); it_spin.setRange(1, 30); it_spin.setValue(int(cfgs[(s, b)][0]))
			su_spin = QSpinBox(); su_spin.setRange(1, 16); su_spin.setValue(int(cfgs[(s, b)][1]))
			grid.addWidget(it_spin, r, 1)
			grid.addWidget(su_spin, r, 2)
			spins[(s, b)] = (it_spin, su_spin)
		btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
		btns.accepted.connect(dlg.accept)
		btns.rejected.connect(dlg.reject)
		grid.addWidget(btns, len(rows) + 1, 0, 1, 3)
		if dlg.exec() != QDialog.DialogCode.Accepted:
			return
		for key, (it_spin, su_spin) in spins.items():
			cfgs[key] = [int(it_spin.value()), int(su_spin.value())]
		self._save_iter_cfg_by_stage()
		resumen = " · ".join(
			f"{'Esf' if s == 'stress' else 'Rep'}-{b[:3]}={cfgs[(s, b)][0]}it/{cfgs[(s, b)][1]}ss"
			for s, b, _ in rows
		)
		self._log(f"[RECON] Iteraciones por estudio: {resumen}")

	def _apply_stage_iter_overrides(self, cfg, stage: str):
		"""Pisa iter/subsets del cfg con la config por estudio (si el usuario la definió)."""
		cfgs = self._load_iter_cfg_by_stage()
		if not cfgs:
			return cfg
		from dataclasses import replace as _dc_replace
		s = "rest" if str(stage) == "rest" else "stress"
		ung = cfgs.get((s, "ungated"))
		gat = cfgs.get((s, "gated"))
		kwargs = {}
		if ung:
			kwargs["iterative_iterations"] = int(ung[0])
			kwargs["osem_subsets"] = int(ung[1])
		if gat:
			kwargs["gated_iterations"] = int(gat[0])
			kwargs["gated_osem_subsets"] = int(gat[1])
		return _dc_replace(cfg, **kwargs) if kwargs else cfg

	def _cine_crudo_recon_config(self, study=None):
		from core.raw_reconstruction import RawReconConfig

		method = str(self.cine_crudo_recon_method_combo.currentText()).strip().lower() if hasattr(self, "cine_crudo_recon_method_combo") else "fbp"
		self._use_adjoint_osem = (method == "osem-adj")
		if method == "osem-adj":
			method = "osem"  # el config solo conoce fbp/osem/mlem
		# NÍTIDA (OmniRes): recuperación de resolución dependiente de profundidad.
		# Es una OPCIÓN de reconstrucción (modela la PSF del colimador dentro del
		# OSEM/MLEM), NO un pre-filtro del crudo: se aplica DESPUÉS del motion
		# correction, en el paso de reconstrucción. Si no hay colimador legible o
		# no está el método iterativo, se auto-corrige a OSEM y se avisa.
		nitida = bool(self.cine_crudo_nitida_check.isChecked()) if hasattr(self, "cine_crudo_nitida_check") and self.cine_crudo_nitida_check is not None else False
		psf_model = None
		if nitida:
			psf_model = self._build_nitida_psf(study)
			if psf_model is None:
				self._log("NÍTIDA (OmniRes): no pude construir la PSF (colimador/geometría no legibles del DICOM). Reconstruyo sin RR.")
				nitida = False
			elif method not in {"osem", "mlem"}:
				self._log("NÍTIDA (OmniRes) requiere OSEM/MLEM en el UNGATED: fuerzo método OSEM para la recuperación de resolución.")
				method = "osem"
		# NÍTIDA es SOLO-UNGATED (perfusión estática nítida). El gated NO la usa:
		# en bajo conteo/gate la RR amplifica ruido; para gated está NITIDA III.
		rr_active = bool(nitida and psf_model is not None)
		# Método independiente de la rama gated. NÍTIDA ya NO fuerza OSEM en gated
		# (es solo-ungated): el gated conserva su propio método/filtro.
		gated_method = str(self.cine_crudo_gated_method_combo.currentText()).strip().lower() if hasattr(self, "cine_crudo_gated_method_combo") and self.cine_crudo_gated_method_combo is not None else method
		# Con OSEM/MLEM el Butterworth del combo se aplica POST-recon (3D) dentro
		# del motor: ya no pelea con la PSF de NÍTIDA ni rompe el modelo Poisson.
		# Solo lowpass/wiener siguen siendo pre-filtro: con NÍTIDA se anulan.
		ungated_filter = self._cine_crudo_recon_filter_config("ungated")
		gated_filter = self._cine_crudo_recon_filter_config("gated")
		if rr_active and str(ungated_filter.kind).strip().lower() in {"lowpass", "wiener"}:
			from core.raw_reconstruction import ProjectionFilterConfig
			self._log(f"NÍTIDA (OmniRes): el pre-filtro {ungated_filter.kind} del UngGat pelea con la PSF; lo anulo (usá butterworth, que va post-recon).")
			ungated_filter = ProjectionFilterConfig(kind="none", cutoff=0.5, order=1)
		# Suavizar POR RAMA: ungated y gated independientes.
		post_sigma_ung = self._cine_crudo_post_filter_sigma_px(study)
		post_sigma_gat = self._cine_crudo_post_filter_gated_sigma_px(study)
		# FBP_CLEAN: denoise en sinograma + realce por resta. Es una cadena
		# autocontenida (FBP + Butterworth + denoise + realce): NO debe sumarse al
		# post-filtro gaussiano ("Suavizar") ni pelear con OSEM. Si está activa,
		# anulamos el post-gaussiano (que difumina/engorda) y forzamos FBP. Solo
		# NITIDA II puede coexistir (actúa post-recon en el gated, otra capa).
		fbc_on = bool(getattr(self, "cine_crudo_fbpclean_check", None) is not None
					and self.cine_crudo_fbpclean_check.isChecked())
		if fbc_on:
			if method not in {"fbp"}:
				self._log("FBP_CLEAN es una cadena FBP: fuerzo método FBP (ignoro OSEM/MLEM).")
				method = "fbp"
				if str(gated_method).lower() in {"osem", "mlem"}:
					gated_method = "fbp"
			if post_sigma_gat > 0.0:
				self._log("FBP_CLEAN activa: anulo el post-filtro gaussiano GATED (el realce ya controla el ruido).")
				post_sigma_gat = 0.0
		fbc_sigma = 0.04 if (fbc_on and method == "fbp") else 0.0
		fbc_k = float(self.cine_crudo_fbpclean_slider.value()) / 100.0 if getattr(self, "cine_crudo_fbpclean_slider", None) is not None else 0.5
		# NITIDA II: denoiser gated temporal/espaciotemporal por armónicos (post-recon).
		# Independiente de NÍTIDA (RR) y compatible con FBP_CLEAN (otra capa, gated).
		nitida2_mode = "none"
		if getattr(self, "cine_crudo_nitida2_combo", None) is not None:
			nitida2_mode = str(self.cine_crudo_nitida2_combo.currentData() or "none")
		iters = int(self.cine_crudo_iter_spin.value()) if hasattr(self, "cine_crudo_iter_spin") else 3
		if rr_active and iters < 8:
			self._log(f"NÍTIDA (OmniRes): {iters} iteraciones no alcanzan para converger la RR; uso 8.")
			iters = 8
		return RawReconConfig(
			reconstruction_method=method,
			gated_method=gated_method,
			ungated_filter=ungated_filter,
			gated_filter=gated_filter,
			iterative_iterations=iters,
			osem_subsets=int(self.cine_crudo_osem_subsets_spin.value()) if hasattr(self, "cine_crudo_osem_subsets_spin") else 6,
			display_slice_step_px=2,
			resolution_recovery=False,
			rr_ungated=rr_active,
			rr_gated=False,
			psf_model=psf_model,
			post_filter_sigma_px=0.0,
			post_filter_sigma_ungated_px=post_sigma_ung,
			post_filter_sigma_gated_px=post_sigma_gat,
			nitida2_mode=nitida2_mode,
			fbp_clean_sigma_color=fbc_sigma,
			fbp_clean_sharpen_k=fbc_k,
			background_subtract=bool(getattr(self, "cine_crudo_bg_check", None) is not None
								and self.cine_crudo_bg_check.isChecked()),
			attenuation_correction=bool(getattr(self, "cine_crudo_ac_check", None) is not None
								and self.cine_crudo_ac_check.isChecked()),
			nitida3_enabled=bool(getattr(self, "cine_crudo_nitida3_check", None) is not None
							and self.cine_crudo_nitida3_check.isChecked()),
			nitida3_iterations=int(self.cine_crudo_nitida3_iter_spin.value()) if getattr(self, "cine_crudo_nitida3_iter_spin", None) is not None else 2,
			nitida4d_enabled=bool(getattr(self, "cine_crudo_nitida4d_check", None) is not None
							and self.cine_crudo_nitida4d_check.isChecked()),
			nitida4d_beta_temporal=(float(self.cine_crudo_nitida4d_beta_spin.value())
								if getattr(self, "cine_crudo_nitida4d_beta_spin", None) is not None else 0.3),
			ungated_denoise_plus=bool(getattr(self, "cine_crudo_denoise_plus_check", None) is not None
								and self.cine_crudo_denoise_plus_check.isChecked()),
			ungated_denoise_plus_k=(float(self.cine_crudo_denoise_plus_slider.value()) / 100.0
								if getattr(self, "cine_crudo_denoise_plus_slider", None) is not None else 0.20),
			gated_denoise_plus=bool(getattr(self, "cine_crudo_denoise_plus_gated_check", None) is not None
								and self.cine_crudo_denoise_plus_gated_check.isChecked()),
			gated_denoise_plus_k=(float(self.cine_crudo_denoise_plus_gated_slider.value()) / 100.0
								if getattr(self, "cine_crudo_denoise_plus_gated_slider", None) is not None else 0.50),
			scatter_subtract=bool(getattr(self, "cine_crudo_scatter_check", None) is not None
							and self.cine_crudo_scatter_check.isEnabled()
							and self.cine_crudo_scatter_check.isChecked()),
			scatter_k=(float(self.cine_crudo_scatter_k_spin.value())
						if getattr(self, "cine_crudo_scatter_k_spin", None) is not None else 1.0),
		)

	# --- Pasajero de fase (FBP) ---
	# La FASE se calcula SIEMPRE sobre un volumen FBP paralelo ("pasajero") con
	# filtros y postfiltro FIJOS: los límites normales de disincronía (Emory/
	# Xeleris) están calibrados sobre FBP-Butterworth y las recon iterativas/RR
	# inflan el Phase SD (medido: 8.1°→20.7°, NORMAL→MILD). FEVI/perfusión usan
	# la recon del usuario (mejor cavidad); la fase queda reproducible entre
	# configuraciones y entre etapas. Override: presets/phase_passenger_config.json.
	PHASE_PASSENGER_DEFAULT_UNG = ("butterworth", 0.52, 5)
	PHASE_PASSENGER_DEFAULT_GATED = ("butterworth", 0.40, 10)
	#: Postfiltro gaussiano ESTÁNDAR del pasajero (FWHM mm). Fijo a propósito:
	#: NO hereda el "Suavizar" del usuario para que la fase no varíe con la UI.
	PHASE_PASSENGER_POST_FWHM_MM = 8.0

	def _phase_passenger_filters(self):
		"""(ung, gated) filtros FBP del pasajero de fase. Override opcional vía preset JSON."""
		ung = self.PHASE_PASSENGER_DEFAULT_UNG
		gated = self.PHASE_PASSENGER_DEFAULT_GATED
		try:
			path = os.path.join(self.presets_dir, "phase_passenger_config.json")
			if os.path.isfile(path):
				with open(path, "r", encoding="utf-8") as fh:
					data = json.load(fh)

				def _f(node, default):
					if not isinstance(node, dict):
						return default
					return (
						str(node.get("kind", default[0])),
						float(node.get("cutoff", default[1])),
						int(node.get("order", default[2])),
					)

				ung = _f(data.get("ungated"), ung)
				gated = _f(data.get("gated"), gated)
		except Exception as exc:
			self._log(f"[WARN] phase_passenger_config.json ilegible; uso filtros default: {exc}")
		return ung, gated

	def _phase_passenger_recon_config(self, study=None):
		"""Config FBP pura para el volumen de fase paralelo (mismo protocolo Xeleris)."""
		from core.raw_reconstruction import RawReconConfig, ProjectionFilterConfig

		ung, gated = self._phase_passenger_filters()
		# Postfiltro ESTÁNDAR fijo (no hereda 'Suavizar'): fase 100% reproducible
		# entre configuraciones de usuario y entre etapas esfuerzo/reposo.
		study = study or getattr(self, "cine_crudo_raw_study_for_recon", None) or self.study
		ps = getattr(study, "pixel_spacing", None) if study is not None else None
		pixel_mm = float(ps[0]) if ps else 6.4
		post_sigma_px = (self.PHASE_PASSENGER_POST_FWHM_MM / 2.354820045) / max(pixel_mm, 1e-6)
		return RawReconConfig(
			reconstruction_method="fbp",
			ungated_filter=ProjectionFilterConfig(kind=ung[0], cutoff=ung[1], order=ung[2]),
			gated_filter=ProjectionFilterConfig(kind=gated[0], cutoff=gated[1], order=gated[2]),
			display_slice_step_px=2,
			resolution_recovery=False,
			psf_model=None,
			post_filter_sigma_px=post_sigma_px,
		)

	def _cine_crudo_post_filter_sigma_px(self, study=None) -> float:
		"""Sigma en píxeles del post-filtro gaussiano segun la casilla 'Suavizar'.

		Convierte el FWHM en mm (control de la UI) a sigma en píxeles usando el
		pixel spacing del estudio. 0.0 = post-filtro desactivado.
		"""
		if not (hasattr(self, "cine_crudo_post_check") and self.cine_crudo_post_check is not None
				and self.cine_crudo_post_check.isChecked()):
			return 0.0
		fwhm_mm = float(self.cine_crudo_post_fwhm_spin.value()) if hasattr(self, "cine_crudo_post_fwhm_spin") else 0.0
		if fwhm_mm <= 0.0:
			return 0.0
		study = study or getattr(self, "cine_crudo_raw_study_for_recon", None) or self.study
		ps = getattr(study, "pixel_spacing", None) if study is not None else None
		pixel_mm = float(ps[0]) if ps else 6.4
		sigma_px = (fwhm_mm / 2.354820045) / max(pixel_mm, 1e-6)
		self._log(f"Post-filtro: suavizado gaussiano FWHM={fwhm_mm:.1f} mm -> sigma={sigma_px:.2f} px (pixel={pixel_mm:.2f} mm).")
		return sigma_px

	def _cine_crudo_post_filter_gated_sigma_px(self, study=None) -> float:
		"""Sigma en píxeles del post-filtro gaussiano GATED (checkbox 'Suavizar' gated)."""
		if not (hasattr(self, "cine_crudo_post_gated_check") and self.cine_crudo_post_gated_check is not None
				and self.cine_crudo_post_gated_check.isChecked()):
			return 0.0
		fwhm_mm = float(self.cine_crudo_post_gated_fwhm_spin.value()) if hasattr(self, "cine_crudo_post_gated_fwhm_spin") else 0.0
		if fwhm_mm <= 0.0:
			return 0.0
		study = study or getattr(self, "cine_crudo_raw_study_for_recon", None) or self.study
		ps = getattr(study, "pixel_spacing", None) if study is not None else None
		pixel_mm = float(ps[0]) if ps else 6.4
		return (fwhm_mm / 2.354820045) / max(pixel_mm, 1e-6)

	def _on_nitida_toggled(self, checked: bool):
		"""NÍTIDA (solo-ungated) toma el control de la rama UNGATED: desactiva su
		filtro de proyección y su selector de método, y enciende 'Suavizar' ungated.
		El GATED queda intacto (NÍTIDA no lo toca; para gated está NITIDA III)."""
		ungated_widgets = [
			"cine_crudo_ung_filter_combo", "cine_crudo_ung_cutoff_spin", "cine_crudo_ung_order_spin",
			# NÍTIDA fuerza OSEM en el ungated: su selector de método no aplica.
			"cine_crudo_recon_method_combo",
		]
		for name in ungated_widgets:
			w = getattr(self, name, None)
			if w is not None:
				w.setEnabled(not checked)
		if checked and hasattr(self, "cine_crudo_post_check") and self.cine_crudo_post_check is not None:
			# OSEM+RR sin post-filtro sale ruidoso/manchado. La receta validada
			# (iteraciones bajas + suavizado ~8 mm) iguala la calidad; por eso NÍTIDA
			# enciende 'Suavizar' ungated y NO toca las iteraciones.
			if not self.cine_crudo_post_check.isChecked():
				self.cine_crudo_post_check.setChecked(True)
		if checked:
			self._log("NÍTIDA activa (ungated): filtro de proyección ungated desactivado; 'Suavizar' ungated encendido. El gated conserva su método/filtro.")

	def _on_post_filter_toggled(self, checked: bool):
		if hasattr(self, "cine_crudo_post_fwhm_spin") and self.cine_crudo_post_fwhm_spin is not None:
			self.cine_crudo_post_fwhm_spin.setEnabled(bool(checked))

	def _on_post_filter_gated_toggled(self, checked: bool):
		if hasattr(self, "cine_crudo_post_gated_fwhm_spin") and self.cine_crudo_post_gated_fwhm_spin is not None:
			self.cine_crudo_post_gated_fwhm_spin.setEnabled(bool(checked))

	def _refresh_fbpclean_filter_lock(self):
		"""Grisa los controles GATED cuando FBP_CLEAN está activo.

		FBP_CLEAN es una cadena autocontenida del GATED (FBP + Butterworth + denoise
		+ realce): anula el filtro de proyección gated, fuerza FBP gated y anula el
		post-gaussiano gated. Esos controles quedan deshabilitados. El UNGATED no se
		toca (FBP_CLEAN no lo afecta). NITIDA II y NITIDA III quedan habilitados
		(otra capa gated, compatibles). 'Fondo' también (preprocesado del sinograma).
		"""
		fbc_on = bool(getattr(self, "cine_crudo_fbpclean_check", None) is not None
					and self.cine_crudo_fbpclean_check.isChecked())
		for name in (
			"cine_crudo_gated_method_combo",
			"cine_crudo_gated_filter_combo", "cine_crudo_gated_cutoff_spin", "cine_crudo_gated_order_spin",
			"cine_crudo_post_gated_check",
		):
			w = getattr(self, name, None)
			if w is not None:
				w.setEnabled(not fbc_on)
		post_g_spin = getattr(self, "cine_crudo_post_gated_fwhm_spin", None)
		if post_g_spin is not None:
			post_g_spin.setEnabled(
				(not fbc_on)
				and bool(getattr(self, "cine_crudo_post_gated_check", None) is not None
						 and self.cine_crudo_post_gated_check.isChecked())
			)

	def _build_nitida_psf(self, study):
		"""Construye el PsfModel (NÍTIDA/OmniRes) a partir del colimador y la geometría del estudio.

		Multi-fabricante: identifica el colimador vía DICOM (fabricante + nombre/tipo)
		contra la base ``collimator_specs`` y calcula la PSF dependiente de profundidad
		con el radio de órbita y el pixel del estudio. Devuelve None si no hay datos
		suficientes para un modelo físico honesto.
		"""
		study = study or getattr(self, "cine_crudo_raw_study_for_recon", None) or self.study
		if study is None:
			return None
		try:
			from core.collimator_specs import lookup_collimator
			from core.resolution_recovery import PsfModel

			spec = lookup_collimator(
				getattr(study, "manufacturer", "") or "",
				getattr(study, "collimator_name", "") or "",
				getattr(study, "collimator_type", "") or "",
			)
			if spec is None:
				return None
			# pixel: prioridad al pixel_spacing del DICOM; fallback razonable.
			ps = getattr(study, "pixel_spacing", None)
			pixel_mm = float(ps[0]) if ps else 6.4
			radius_mm = getattr(study, "radius_mm", None)
			if radius_mm is None or float(radius_mm) <= 0.0:
				radius_mm = 250.0  # fallback conservador (órbita típica cardíaca)
				self._log("NÍTIDA (OmniRes): radio de órbita ausente en el DICOM; uso 250 mm por defecto.")
			psf = PsfModel.from_collimator(spec, radius_mm=float(radius_mm), pixel_mm=pixel_mm)
			self._log(
				f"NÍTIDA (OmniRes): colimador {spec.manufacturer} {spec.name} [{spec.geometry}] · "
				f"radio={float(radius_mm):.0f} mm · pixel={pixel_mm:.2f} mm · FWHM_int={spec.intrinsic_fwhm_mm:.1f} mm."
			)
			return psf
		except Exception as exc:  # pragma: no cover - defensivo en UI
			self._log(f"NÍTIDA (OmniRes): error construyendo PSF ({exc}). Reconstruyo sin RR.")
			return None

	def _identity_cine_crudo_motion_result(self, projections: np.ndarray, method: str) -> dict:
		n_angles = int(np.asarray(projections).shape[1])
		return {
			"corrected": np.asarray(projections, dtype=np.float64),
			"applied_shifts_y": np.zeros((n_angles,), dtype=np.float64),
			"applied_shifts_x": np.zeros((n_angles,), dtype=np.float64),
			"method": method,
			"axis_corrected": "none",
			"max_shift_px": 0.0,
		}

	def _schedule_recon_branch_recompute(self, branch: str) -> None:
		"""Dispara (con debounce) el recompute de una rama tras cambiar su filtro."""
		if getattr(self, "cine_crudo_recon_result", None) is None:
			return
		# NÍTIDA anula el filtro de proyección de SU rama: no hay recompute por filtro
		# en esa rama. Con NÍTIDA solo-ungated, el gated SÍ puede recomputar.
		cfg = getattr(self.cine_crudo_recon_result, "config", None)
		if cfg is not None:
			_rr_g = bool(getattr(cfg, "resolution_recovery", False))
			_rr_ung = _rr_g if getattr(cfg, "rr_ungated", None) is None else bool(cfg.rr_ungated)
			_rr_gat = _rr_g if getattr(cfg, "rr_gated", None) is None else bool(cfg.rr_gated)
			if (branch == "ungated" and _rr_ung) or (branch == "gated" and _rr_gat):
				return
		if branch == "ungated":
			self._recon_recompute_ung_timer.start(350)
		else:
			self._recon_recompute_gated_timer.start(350)

	def _recompute_recon_branch_ungated(self) -> None:
		self._recompute_recon_branch("ungated")

	def _recompute_recon_branch_gated(self) -> None:
		self._recompute_recon_branch("gated")

	def _recompute_recon_branch(self, branch: str) -> None:
		"""Recomputa SOLO una rama (ungated o gated) con el filtro/método actual de la UI.

		Reutiliza las proyecciones ya corregidas por motion correction guardadas en
		``result`` (no rehace el pipeline entero) y reaplica el mismo flip L/R y
		post-filtro que el pipeline. Cada rama tiene su propio método (FBP/MLEM/OSEM).
		Refresca la vista dual de límites al terminar.
		"""
		from dataclasses import replace as _dc_replace

		result = getattr(self, "cine_crudo_recon_result", None)
		if result is None:
			return
		cfg = result.config
		if bool(getattr(cfg, "attenuation_correction", False)):
			self._log("[AC] Recompute por rama no soporta AC iterativa: usá 'Recon raw' completo para conservar la corrección.")
			return
		_rr_g = bool(getattr(cfg, "resolution_recovery", False))
		_rr_ung = _rr_g if getattr(cfg, "rr_ungated", None) is None else bool(cfg.rr_ungated)
		_rr_gat = _rr_g if getattr(cfg, "rr_gated", None) is None else bool(cfg.rr_gated)
		if (branch == "ungated" and _rr_ung) or (branch == "gated" and _rr_gat):
			return  # NÍTIDA en esta rama: filtro de proyección desactivado por diseño
		# Método propio de la rama desde su combo.
		if branch == "ungated":
			method = str(self.cine_crudo_recon_method_combo.currentText()).strip().lower() if hasattr(self, "cine_crudo_recon_method_combo") else "fbp"
		else:
			method = str(self.cine_crudo_gated_method_combo.currentText()).strip().lower() if hasattr(self, "cine_crudo_gated_method_combo") else "fbp"
		if method not in {"fbp", "mlem", "osem"}:
			return
		from scipy.ndimage import gaussian_filter
		from core.raw_reconstruction import (
			reconstruct_projection_volume,
			reconstruct_gated_projection_volume,
			_detect_rotation_ccw,
			_FLIP_X_ON_CCW,
		)

		raw_study = getattr(self, "cine_crudo_raw_study_for_recon", None) or getattr(self, "study", None)
		angles = getattr(raw_study, "angles_deg", None) if raw_study is not None else None
		subsets = int(cfg.osem_subsets) if method == "osem" else 1
		ccw = _detect_rotation_ccw(angles)
		flip_x = True if ccw is None else (bool(ccw) == _FLIP_X_ON_CCW)
		# Post-filtro por rama (fallback al global si la por-rama es None).
		_post_global = float(getattr(cfg, "post_filter_sigma_px", 0.0) or 0.0)
		if branch == "ungated":
			post_sigma = _post_global if getattr(cfg, "post_filter_sigma_ungated_px", None) is None else float(cfg.post_filter_sigma_ungated_px)
		else:
			post_sigma = _post_global if getattr(cfg, "post_filter_sigma_gated_px", None) is None else float(cfg.post_filter_sigma_gated_px)
		new_filter = self._cine_crudo_recon_filter_config(branch)
		try:
			if branch == "ungated":
				vol = reconstruct_projection_volume(
					np.asarray(result.ungated_projections, dtype=np.float64), angles,
					method=method, projection_filter=new_filter, fbp_filter_name=cfg.fbp_filter_name,
					iterations=int(cfg.iterative_iterations), subsets=subsets,
				)
				if flip_x:
					vol = np.ascontiguousarray(np.flip(vol, axis=-1))
				if post_sigma > 0.05:
					vol = gaussian_filter(vol, sigma=post_sigma, mode="constant")
				result.ungated_volume = vol
				result.config = _dc_replace(cfg, reconstruction_method=method, ungated_filter=new_filter)
			else:
				gated = reconstruct_gated_projection_volume(
					np.asarray(result.corrected_projections, dtype=np.float64), angles,
					method=method, projection_filter=new_filter, fbp_filter_name=cfg.fbp_filter_name,
					iterations=int(cfg.iterative_iterations), subsets=subsets,
				)
				if flip_x:
					gated = np.ascontiguousarray(np.flip(gated, axis=-1))
				if post_sigma > 0.05:
					for g in range(gated.shape[0]):
						gated[g] = gaussian_filter(gated[g], sigma=post_sigma, mode="constant")
				result.gated_volume = gated
				result.config = _dc_replace(cfg, gated_method=method, gated_filter=new_filter)
		except Exception as exc:
			self._log(f"[WARN] Recompute rama {branch} falló: {exc}")
			return
		self._log(
			f"Recompute {branch}: {method.upper()} · filtro={new_filter.kind} {new_filter.cutoff:.2f}/{new_filter.order}. "
			"La otra rama se conserva."
		)
		self._preview_cine_crudo_cut_limits()


	def _cine_crudo_recon_target(self):
		"""Devuelve (study, motion_result, corrected, stage) de la etapa a reconstruir.

		Esfuerzo→estudio primario; Reposo→estudio secundario. La reconstrucción y
		reorientación son por etapa (modal): con 'Ambas' se procesa esfuerzo primero
		(reconstruí/reorientá una, marcá como reposo, y elegí la otra para el montaje).
		"""
		stage = getattr(self, "_cine_crudo_active_stage", "stress")
		secondary = self._secondary_cine_crudo_study()
		if stage == "rest" and secondary is not None:
			return secondary, self.cine_crudo_motion_result_compare, self.cine_crudo_corrected_projections_compare, "rest"
		return (self.cine_crudo_raw_study_for_recon or self.study), self.cine_crudo_motion_result, self.cine_crudo_corrected_projections, "stress"

	# ------------------------------------------------ sustracción de fondo (crudo → cadena)
	def set_raw_background_subtraction(self, stage: str, projections, spec: dict):
		"""Registra una sustracción de fondo del crudo para que alimente la reconstrucción.

		La invoca la ventana de reconstrucción cuando el impacto elegido es
		"toda la cadena". ``projections`` es la imagen ungated ya restada (solo para
		referencia visual); lo que consume la recon es ``spec`` (método, nivel,
		polígonos), reaplicado sobre el cubo gated con el escalado por gate.
		"""
		if not spec:
			return
		self._raw_bg_spec[str(stage)] = dict(spec)
		try:
			self._log(
				f"[fondo crudo→cadena] {stage}: modo={spec.get('method')} nivel={float(spec.get('level', 0.0)):.1f}. "
				"Se aplicará al reconstruir esta etapa."
			)
		except Exception:
			pass

	def clear_raw_background_subtraction(self, stage: str | None = None):
		"""Descarta la sustracción de fondo registrada (una etapa o todas)."""
		if not getattr(self, "_raw_bg_spec", None):
			return
		if stage is None:
			self._raw_bg_spec.clear()
		else:
			self._raw_bg_spec.pop(str(stage), None)

	def _apply_raw_bg_to_recon_cube(self, projections: np.ndarray, stage: str) -> np.ndarray:
		"""Aplica la sustracción de fondo registrada al cubo de entrada de la recon.

		El nivel se midió sobre proyecciones ungated (suma de gates); para el cubo
		gated (n_gates, n_ángulos, H, W) se resta ``nivel / n_gates`` por gate, de
		modo que la suma sobre gates coincida con lo que se ve en el MIP crudo.
		"""
		spec = getattr(self, "_raw_bg_spec", {}).get(str(stage))
		if not spec or str(spec.get("impact")) != "chain":
			return projections
		arr = np.asarray(projections, dtype=np.float64)
		if arr.ndim < 2:
			return arr
		from core.raw_background import polygon_mask, subtract_constant, subtract_localized

		level = float(spec.get("level", 0.0))
		method = str(spec.get("method", "constant"))
		h, w = int(arr.shape[-2]), int(arr.shape[-1])
		if arr.ndim == 4:
			n_gates = max(1, int(arr.shape[0]))
			eff_level = level / n_gates
		else:
			eff_level = level
		heart_poly = spec.get("heart_polygon") or []
		if method == "localized" and len(heart_poly) >= 3:
			heart_mask = polygon_mask((h, w), heart_poly)
			res = subtract_localized(arr, eff_level, heart_mask, feather_px=2.0)
		else:
			res = subtract_constant(arr, eff_level)
		self._log(
			f"[fondo crudo→recon] {stage}: resta {method} nivel_efectivo={eff_level:.2f} "
			f"sobre cubo {arr.shape} · clip {res.clipped_fraction * 100:.0f}%."
		)
		return res.image

	def _on_recon_raw_clicked(self):
		"""Recon raw según el selector Esfuerzo/Reposo/Ambas de la toolbar."""
		combo = getattr(self, "cine_crudo_recon_stage_combo", None)
		stage = str(combo.currentData()) if combo is not None else "stress"
		if stage in ("stress", "rest", "both"):
			self._set_active_cine_crudo_stage(stage, refresh_view=False, force=True)
		return self._reconstruct_cine_crudo_raw()

	def _reconstruct_cine_crudo_raw(self, feta_only: bool = False, _force_stage: str | None = None):
		"""Reconstruye desde crudo la etapa seleccionada (Esfuerzo=primario / Reposo=secundario).

		``feta_only``: si True, reconstruye SOLO la banda axial (feta) delimitada por
		los markers Base/Ápex — excluye actividad extracardíaca de arriba/abajo y es
		más rápido. El volumen resultante es el de trabajo (reorientación/análisis).
		"""
		if _force_stage is None:
			stages = self._cine_crudo_target_stages()
			if len(stages) > 1:
				return self._run_cine_crudo_stage_orchestrator(
					"Recon raw",
					lambda stage: self._reconstruct_cine_crudo_raw(feta_only=feta_only, _force_stage=stage),
				)
			_force_stage = stages[0]
		if _force_stage:
			self._set_active_cine_crudo_stage(str(_force_stage), refresh_view=False, force=True)
		raw_study, motion_result, corrected, stage = self._cine_crudo_recon_target()
		if raw_study is None:
			QMessageBox.information(self, "SINCRO", "Cargá un estudio crudo gated en cine_crudo primero.")
			return False		# Fijar el estudio activo de reconstrucción para todo el pipeline downstream
		# (reorientación, cortes, metadatos) según la etapa elegida.
		self.cine_crudo_raw_study_for_recon = raw_study
		self._cine_crudo_recon_stage = stage
		# Detener el cine del crudo: si sigue corriendo, el timer pisaría la imagen de
		# reconstrucción en la pestaña (incluso durante el diálogo modal por processEvents).
		self.cine_crudo_timer.stop()
		self.cine_crudo_playing = False
		self._update_cine_crudo_toggle_text()
		_undo_before = None if getattr(self, "_undo_suspended", False) else self._snapshot_attrs(self.UNDO_ATTRS_RECON, deep=False)
		try:
			from core.raw_reconstruction import reconstruct_raw_gated_pipeline

			projections = np.asarray(raw_study.cube, dtype=np.float64)
			projections = self._apply_raw_bg_to_recon_cube(projections, stage)
			angles = getattr(raw_study, "angles_deg", None)
			cfg = self._cine_crudo_recon_config(raw_study)
			cfg = self._apply_stage_iter_overrides(cfg, stage)
			feta_txt = ""
			if feta_only:
				# La feta se define con los markers Base/Ápex de ESTA pantalla, que
				# operan sobre el eje axial z de la recon (= altura del detector).
				# Requiere una recon previa (FBP default) para tener las líneas rojas.
				if self.cine_crudo_recon_result is None:
					QMessageBox.information(self, "SINCRO", "Reconstruir selección: primero tocá 'Recon raw' (FBP rápido), ajustá las líneas Base/Ápex sobre el corazón y recién ahí reconstruí la selección.")
					self._set_progress(100, "Reconstruir selección: falta recon base")
					return False
				height = int(projections.shape[2])
				z0, z1 = self._cine_crudo_cut_bounds(height)
				from dataclasses import replace as _dc_replace
				cfg = _dc_replace(cfg, recon_slice_range=(z0, z1))
				feta_txt = f" · feta z=[{z0},{z1}]/{height}"
				self._log(f"Reconstruir selección: feta axial z=[{z0},{z1}] de {height} (markers Base/Ápex).")
			etapa_txt = "reposo" if stage == "rest" else "esfuerzo"
			etapa_caps = "REPOSO" if stage == "rest" else "ESFUERZO"
			dual_ctx = getattr(self, "_cine_crudo_dual_context", None) or {}
			dual_tag = ""
			if int(dual_ctx.get("total", 1)) > 1:
				dual_tag = f" ({int(dual_ctx.get('idx', 1))}/{int(dual_ctx.get('total', 2))})"
			if motion_result is not None and corrected is not None:
				motion = dict(motion_result)
				source_label = f"{etapa_txt} · corregido por motion correction{feta_txt}"
			else:
				motion = self._identity_cine_crudo_motion_result(projections, "sin_correccion")
				source_label = f"{etapa_txt} · crudo original sin correccion{feta_txt}"

			if cfg.reconstruction_method.lower() in {"mlem", "osem"} and projections.shape[-1] >= 64:
				self._log("[INFO] MLEM/OSEM CPU en matriz real puede tardar; para pruebas rápidas usá Iter=1-2.")
			self._set_progress(45, f"Reconstruyendo raw ({cfg.reconstruction_method.upper()})...")

			# AC: μ-map de la ETAPA remuestreado a la grilla de recon (o None).
			ac_mu_map, ac_px_cm = (None, None)
			if bool(getattr(cfg, "attenuation_correction", False)):
				ac_mu_map, ac_px_cm = self._stage_mu_map_for_recon(stage, projections, raw_study)

			nitida_on = bool(
				(getattr(cfg, "resolution_recovery", False) or getattr(cfg, "rr_ungated", False))
				and getattr(cfg, "psf_model", None) is not None
			)
			titulo_base = "NÍTIDA (OmniRes) — recuperación de resolución" if nitida_on else f"Reconstrucción {cfg.reconstruction_method.upper()}"
			titulo = f"{titulo_base} · {etapa_caps}{dual_tag}"
			recon_dialog = QProgressDialog(
				f"{titulo}\nEsto puede tardar según iteraciones y tamaño de matriz…",
				None, 0, 100, self,
			)
			recon_dialog.setWindowTitle(f"SINCRO · Reconstruyendo · {etapa_caps}{dual_tag}")
			recon_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
			recon_dialog.setMinimumWidth(420)
			recon_dialog.setMinimumDuration(0)
			recon_dialog.setAutoClose(False)
			recon_dialog.setAutoReset(False)
			recon_dialog.setCancelButton(None)
			recon_dialog.setValue(0)
			recon_dialog.show()
			QApplication.processEvents()

			def _recon_progress(fraction: float, message: str = "") -> None:
				# Mapea el avance interno (0..1) del pipeline al tramo 45-99% de la barra
				# lateral y al diálogo modal en primer plano.
				frac = max(0.0, min(1.0, float(fraction)))
				pct = int(round(45 + 54 * frac))
				msg = message or f"Reconstruyendo raw ({cfg.reconstruction_method.upper()})..."
				self._set_progress(min(99, pct), msg)
				recon_dialog.setLabelText(f"{titulo}\n{msg}")
				recon_dialog.setValue(int(round(100 * frac)))
				QApplication.processEvents()

			try:
				result = reconstruct_raw_gated_pipeline(
					projections, angles, motion_result=motion, config=cfg,
					progress_callback=_recon_progress,
					scatter_projections=getattr(raw_study, "scatter_projections", None),
					attenuation_mu_map=ac_mu_map,
					attenuation_pixel_size_cm=ac_px_cm,
				)
			finally:
				recon_dialog.close()
				recon_dialog.deleteLater()
			self.cine_crudo_recon_result = result
			# Las notas de AC del motor son la única evidencia de si la corrección
			# realmente aplicó (p.ej. FBP no la usa): mostrarlas SIEMPRE en el log.
			for _n in list(getattr(result, "notes", []) or []):
				if "AC " in _n or "AC iterativa" in _n or "ATT MAP" in _n:
					self._log(f"[AC] {_n}")

			# OSEM-Adj: reemplazar el volumen ungated con la recon adyunta.
			if getattr(self, "_use_adjoint_osem", False):
				try:
					self._set_progress(85, "OSEM adyunto ray-driven...")
					from core.osem_adjoint import osem_adjoint_reconstruct_slice
					proj_adj = np.asarray(projections, dtype=np.float64)
					angles_adj = np.asarray(angles, dtype=np.float64)
					if proj_adj.ndim == 4:
						# gated: sumar gates para ungated
						proj_ung = proj_adj.sum(axis=0)
					else:
						proj_ung = proj_adj
					n_slices_adj = proj_ung.shape[1]
					det_size = proj_ung.shape[2]
					recon_vol = np.zeros((n_slices_adj, det_size, det_size), dtype=np.float64)
					for sl in range(n_slices_adj):
						sino = proj_ung[:, sl, :].T
						recon_vol[sl] = osem_adjoint_reconstruct_slice(
							sino, angles_adj, output_size=det_size,
							iterations=int(cfg.iterations), subsets=int(cfg.osem_subsets),
						)
						self._set_progress(85 + 10 * (sl + 1) / max(n_slices_adj, 1))
						QApplication.processEvents()
					result.ungated_volume = recon_vol
					self._log(f"OSEM-Adj aplicado: {n_slices_adj} slices, {int(cfg.iterations)} iter, {int(cfg.osem_subsets)} subsets.")
				except Exception as exc:
					self._log(f"[WARN] OSEM-Adj falló, conservo recon estándar: {exc}")

			if feta_only:
				try:
					self._dump_feta_for_harness(result, raw_study, angles, cfg, z0, z1, stage)
				except Exception as exc:
					self._log(f"[WARN] No pude volcar la feta para el harness: {exc}")
			# Pasajero de fase (FBP): SIEMPRE. La fase se calcula sobre un volumen
			# FBP-Butterworth fijo de la MISMA geometría (mismos shifts de motion y
			# proyecciones bg-restadas). Perfusión/FEVI usan la recon del usuario.
			self.cine_crudo_recon_result_phase = None
			try:
				self._set_progress(99, "Pasajero de fase (FBP)...")
				phase_cfg = self._phase_passenger_recon_config(raw_study)
				if getattr(cfg, "recon_slice_range", None) is not None:
					from dataclasses import replace as _dc_replace
					phase_cfg = _dc_replace(phase_cfg, recon_slice_range=cfg.recon_slice_range)
				phase_result = reconstruct_raw_gated_pipeline(
					projections, angles, motion_result=motion, config=phase_cfg
				)
				self.cine_crudo_recon_result_phase = phase_result
				self._log(
					"Pasajero de fase FBP generado: "
					f"UngGat={phase_cfg.ungated_filter.kind} {phase_cfg.ungated_filter.cutoff:.2f}/{phase_cfg.ungated_filter.order}; "
					f"Gated={phase_cfg.gated_filter.kind} {phase_cfg.gated_filter.cutoff:.2f}/{phase_cfg.gated_filter.order}; "
					f"post-filtro sigma={phase_cfg.post_filter_sigma_px:.2f}px; "
					f"volumen={phase_result.gated_volume.shape}. La fase se calculará sobre este volumen."
				)
			except Exception as exc:
				self.cine_crudo_recon_result_phase = None
				self._log(f"[WARN] Pasajero de fase FBP no generado; la fase caerá al volumen visible: {exc}")

			self.cine_crudo_recon_study = None
			self.cine_crudo_cut_study = None
			self.cine_crudo_cut_source_label = source_label
			self.cine_crudo_reoriented_gated = None
			self.cine_crudo_reoriented_ungated = None
			self.cine_crudo_reoriented_gated_phase = None
			self.cine_crudo_reoriented_ct = None
			self.cine_crudo_reoriented_mf = None
			if hasattr(self, "cine_crudo_reorient_btn"):
				self.cine_crudo_reorient_btn.setEnabled(True)
			if hasattr(self, "cine_crudo_process_recon_btn"):
				self.cine_crudo_process_recon_btn.setEnabled(False)
			n_slices = int(result.gated_volume.shape[1])
			if hasattr(self, "cine_crudo_cut_base_spin") and hasattr(self, "cine_crudo_cut_apex_spin"):
				self.cine_crudo_cut_base_spin.setEnabled(True)
				self.cine_crudo_cut_apex_spin.setEnabled(True)
				self.cine_crudo_cut_base_spin.setRange(1, max(1, n_slices))
				self.cine_crudo_cut_apex_spin.setRange(1, max(1, n_slices))
				# Con feta, preservar la selección Base/Ápex del usuario (la feta se
				# reconstruyó justo en esa banda); solo resetear a full en Recon raw.
				if not feta_only:
					self.cine_crudo_cut_base_spin.setValue(1)
					self.cine_crudo_cut_apex_spin.setValue(max(1, n_slices))
					self._cine_crudo_stage_limits_set(stage, 1, max(1, n_slices), n_slices)
				else:
					self._cine_crudo_capture_limits_from_spins(stage)
			if hasattr(self, "cine_crudo_cut_thickness_spin"):
				self.cine_crudo_cut_thickness_spin.setEnabled(True)
			if hasattr(self, "cine_crudo_preview_limits_btn"):
				self.cine_crudo_preview_limits_btn.setEnabled(True)
			if hasattr(self, "cine_crudo_generate_cuts_btn"):
				self.cine_crudo_generate_cuts_btn.setEnabled(True)
			self._preview_cine_crudo_cut_limits()
			self._log(
				f"Recon raw lista: fuente={source_label}; método={cfg.reconstruction_method.upper()}; "
				f"UngGat={cfg.ungated_filter.kind} {cfg.ungated_filter.cutoff:.2f}/{cfg.ungated_filter.order}; "
				f"Gated={cfg.gated_filter.kind} {cfg.gated_filter.cutoff:.2f}/{cfg.gated_filter.order}; "
				f"volumen gated={result.gated_volume.shape}. Ajustá límites Base/Ápex y tocá Generar cortes."
			)
			self._set_progress(100, "Recon raw lista; definí límites de cortes")
			self._commit_undo("Reconstrucción", self.UNDO_ATTRS_RECON, _undo_before, deep=False)
			self._mark_step_done("crudo")
			self._mark_step_done("recon", cfg.reconstruction_method, getattr(result.gated_volume, "shape", None))
			try:
				self._refresh_fusion_btn_state()
			except Exception:
				pass
			return True
		except Exception as exc:
			self._log(f"[ERROR] Recon raw falló: {exc}")
			self._set_progress(100, "Recon raw falló")
			QMessageBox.warning(self, "SINCRO", f"No se pudo reconstruir desde crudo:\n{exc}")
			return False

	def _dump_feta_for_harness(self, result, raw_study, angles, cfg, z0: int, z1: int, stage: str) -> None:
		"""Vuelca la feta reconstruida + proyecciones corregidas + geometría a disco.

		El harness de NÍTIDA (proceso aparte) lee estos archivos para re-reconstruir
		la MISMA banda con distintos prefiltros/iteraciones/post-filtros y comparar,
		sin depender de re-hacer el motion correction ni de adivinar la banda.
		"""
		import json
		out_dir = os.path.join(self.output_dir, "_feta_harness")
		os.makedirs(out_dir, exist_ok=True)
		ung_proj = np.asarray(result.ungated_projections, dtype=np.float32)
		corr_proj = np.asarray(result.corrected_projections, dtype=np.float32)
		ung_vol = np.asarray(result.ungated_volume, dtype=np.float32)
		gated_vol = np.asarray(result.gated_volume, dtype=np.float32)
		ang_src = angles if angles is not None else getattr(raw_study, "angles_deg", None)
		ang = np.asarray(ang_src, dtype=np.float64)
		np.save(os.path.join(out_dir, "ungated_projections.npy"), ung_proj)
		np.save(os.path.join(out_dir, "corrected_projections.npy"), corr_proj)
		np.save(os.path.join(out_dir, "ungated_volume.npy"), ung_vol)
		np.save(os.path.join(out_dir, "gated_volume.npy"), gated_vol)
		np.save(os.path.join(out_dir, "angles_deg.npy"), ang)
		px = getattr(raw_study, "pixel_mm", None)
		meta = {
			"z0": int(z0), "z1": int(z1), "stage": str(stage),
			"pixel_mm": float(px) if px else None,
			"radius_mm": float(getattr(raw_study, "radius_mm", 0.0) or 0.0) or None,
			"manufacturer": str(getattr(raw_study, "manufacturer", "") or ""),
			"collimator_name": str(getattr(raw_study, "collimator_name", "") or ""),
			"collimator_type": str(getattr(raw_study, "collimator_type", "") or ""),
			"ungated_shape": list(ung_proj.shape),
			"gated_shape": list(gated_vol.shape),
			"method": str(cfg.reconstruction_method),
			"iterations": int(cfg.iterative_iterations),
			"subsets": int(cfg.osem_subsets),
		}
		with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
			json.dump(meta, fh, ensure_ascii=False, indent=2)
		self._log(f"Feta volcada para harness -> {out_dir}  (z=[{z0},{z1}], ung_proj {ung_proj.shape})")

	def _dump_reorient_for_harness(self, dlg) -> None:
		"""Vuelca la orientación REAL aprobada por el usuario (eje largo + centro) y
		los volúmenes ya reorientados, para que el harness compare FBP vs NÍTIDA
		sobre la dona SA que el usuario definió (no sobre un eje automático).
		"""
		import json
		out_dir = os.path.join(self.output_dir, "_feta_harness")
		os.makedirs(out_dir, exist_ok=True)
		u = getattr(dlg, "result_long_axis", None)
		center = getattr(dlg, "result_center", None)
		try:
			rz, ry, rx = dlg._voi_semiaxes()
		except Exception:
			rz = ry = rx = None
		payload = {
			"long_axis": ([float(x) for x in np.asarray(u, dtype=np.float64).ravel()[:3]] if u is not None else None),
			"center": ([float(x) for x in np.asarray(center, dtype=np.float64).ravel()[:3]] if center is not None else None),
			"base_k": int(getattr(dlg, "base_k", 0)),
			"apex_k": int(getattr(dlg, "apex_k", 0)),
			"thickness": int(getattr(dlg, "thickness", 1)),
			"out_size": int(getattr(dlg, "result_out_size", 0)),
			"voi_semiaxes": ([float(rz), float(ry), float(rx)] if rz is not None else None),
		}
		if getattr(self, "cine_crudo_reoriented_ungated", None) is not None:
			np.save(os.path.join(out_dir, "reoriented_ungated.npy"),
					np.asarray(self.cine_crudo_reoriented_ungated, dtype=np.float32))
		with open(os.path.join(out_dir, "reorient.json"), "w", encoding="utf-8") as fh:
			json.dump(payload, fh, ensure_ascii=False, indent=2)
		self._log(f"Reorientación volcada para harness -> {out_dir}\\reorient.json  (eje={payload['long_axis']})")

	def _cine_crudo_stage_limits_get(self, stage: str | None = None, n_slices: int | None = None) -> tuple[int, int]:
		"""Límites Base/Ápex 1-based de una etapa, con clamp al tamaño actual."""
		st = str(stage or getattr(self, "_cine_crudo_recon_stage", "stress") or "stress").lower()
		if st not in ("stress", "rest"):
			st = "stress"
		by_stage = getattr(self, "_cine_crudo_cut_limits_by_stage", None)
		if not isinstance(by_stage, dict):
			by_stage = {
				"stress": {"base_1": 1, "apex_1": 1},
				"rest": {"base_1": 1, "apex_1": 1},
			}
			self._cine_crudo_cut_limits_by_stage = by_stage
		vals = by_stage.get(st)
		if not isinstance(vals, dict):
			vals = {"base_1": 1, "apex_1": None}
			by_stage[st] = vals
		if n_slices is None:
			try:
				res = self._dual_session().stage(st).recon_result
				n_slices = int(np.asarray(res.gated_volume).shape[1]) if res is not None else None
			except Exception:
				n_slices = None
		# n desconocido: NO clampear (evita "trabar" los límites en 1 cuando la
		# etapa aún no tiene recon). Se devuelve lo guardado tal cual.
		if n_slices is None:
			base_raw = int(vals.get("base_1", 1) or 1)
			apex_raw = vals.get("apex_1", None)
			apex_raw = int(apex_raw) if apex_raw else max(base_raw, 1)
			return (min(base_raw, apex_raw), max(base_raw, apex_raw))
		n = max(1, int(n_slices))
		base_1 = int(np.clip(int(vals.get("base_1", 1) or 1), 1, n))
		apex_raw = vals.get("apex_1", None)
		apex_1 = int(np.clip(int(apex_raw), 1, n)) if apex_raw else n
		if base_1 > apex_1:
			base_1, apex_1 = apex_1, base_1
		# Lectura NO destructiva: no re-escribir vals acá (el clamp con un n
		# transitorio incorrecto dejaba los markers pegados).
		return int(base_1), int(apex_1)

	def _cine_crudo_stage_limits_set(self, stage: str | None, base_1: int, apex_1: int, n_slices: int | None = None) -> tuple[int, int]:
		"""Guarda límites Base/Ápex 1-based por etapa (stress/rest)."""
		st = str(stage or getattr(self, "_cine_crudo_recon_stage", "stress") or "stress").lower()
		if st not in ("stress", "rest"):
			st = "stress"
		if n_slices is None:
			try:
				res = self._dual_session().stage(st).recon_result
				n_slices = int(np.asarray(res.gated_volume).shape[1]) if res is not None else None
			except Exception:
				n_slices = None
		b = max(1, int(base_1))
		a = max(1, int(apex_1))
		if n_slices is not None:
			n = max(1, int(n_slices))
			b = int(np.clip(b, 1, n))
			a = int(np.clip(a, 1, n))
		if b > a:
			b, a = a, b
		by_stage = getattr(self, "_cine_crudo_cut_limits_by_stage", None)
		if not isinstance(by_stage, dict):
			by_stage = {}
			self._cine_crudo_cut_limits_by_stage = by_stage
		by_stage.setdefault("stress", {"base_1": 1, "apex_1": None})
		by_stage.setdefault("rest", {"base_1": 1, "apex_1": None})
		by_stage[st] = {"base_1": int(b), "apex_1": int(a)}
		return int(b), int(a)

	def _cine_crudo_capture_limits_from_spins(self, stage: str | None = None) -> None:
		"""Persiste a etapa activa los valores de los spins Base/Ápex."""
		if not (hasattr(self, "cine_crudo_cut_base_spin") and hasattr(self, "cine_crudo_cut_apex_spin")):
			return
		st = str(stage or getattr(self, "_cine_crudo_recon_stage", "stress") or "stress").lower()
		if st not in ("stress", "rest"):
			st = "stress"
		try:
			res = self._dual_session().stage(st).recon_result
			n = int(np.asarray(res.gated_volume).shape[1]) if res is not None else None
		except Exception:
			n = None
		if n is None:
			# Etapa sin recon: los spins NO muestran esta etapa; no capturar
			# (evita pisar los límites guardados con valores ajenos).
			return
		self._cine_crudo_stage_limits_set(st, int(self.cine_crudo_cut_base_spin.value()), int(self.cine_crudo_cut_apex_spin.value()), n)

	def _cine_crudo_apply_stage_limits_to_spins(self, stage: str | None = None, n_slices: int | None = None) -> None:
		"""Carga en los spins los límites guardados de una etapa."""
		if not (hasattr(self, "cine_crudo_cut_base_spin") and hasattr(self, "cine_crudo_cut_apex_spin")):
			return
		st = str(stage or getattr(self, "_cine_crudo_recon_stage", "stress") or "stress").lower()
		if st not in ("stress", "rest"):
			st = "stress"
		if n_slices is None:
			try:
				res = self._dual_session().stage(st).recon_result
				n_slices = int(np.asarray(res.gated_volume).shape[1]) if res is not None else None
			except Exception:
				n_slices = None
		if n_slices is None:
			# Etapa sin recon: no tocar los spins (evita clampear/contaminar con
			# el rango de la otra etapa).
			return
		b, a = self._cine_crudo_stage_limits_get(st, n_slices)
		n = max(1, int(n_slices))
		self.cine_crudo_cut_base_spin.blockSignals(True)
		self.cine_crudo_cut_apex_spin.blockSignals(True)
		# Ajustar el RANGO primero: si quedó el de la otra etapa, QSpinBox
		# clampa el valor al setValue y corrompe los límites guardados.
		self.cine_crudo_cut_base_spin.setRange(1, n)
		self.cine_crudo_cut_apex_spin.setRange(1, n)
		self.cine_crudo_cut_base_spin.setValue(int(b))
		self.cine_crudo_cut_apex_spin.setValue(int(a))
		self.cine_crudo_cut_base_spin.blockSignals(False)
		self.cine_crudo_cut_apex_spin.blockSignals(False)

	def _cine_crudo_cut_bounds(self, n_slices: int) -> tuple[int, int]:
		stage = getattr(self, "_cine_crudo_recon_stage", "stress")
		base, apex = self._cine_crudo_stage_limits_get(stage, int(n_slices))
		z0 = int(np.clip(min(base, apex) - 1, 0, max(0, int(n_slices) - 1)))
		z1 = int(np.clip(max(base, apex) - 1, 0, max(0, int(n_slices) - 1)))
		if z1 <= z0:
			z1 = min(max(0, int(n_slices) - 1), z0 + 1)
		return z0, z1

	def _cine_crudo_cut_thickness_px(self) -> int:
		if hasattr(self, "cine_crudo_cut_thickness_spin"):
			return max(1, int(self.cine_crudo_cut_thickness_spin.value()))
		return 1

	def _thickened_sa_volume(self, volume: np.ndarray, z0: int, z1: int, thickness_px: int) -> np.ndarray:
		vol = np.asarray(volume, dtype=np.float64)
		if vol.ndim != 3:
			raise ValueError(f"volume debe ser 3D; recibió {vol.shape}")
		thick = max(1, int(thickness_px))
		half_low = (thick - 1) // 2
		half_high = thick // 2
		slices = []
		for z in range(int(z0), int(z1) + 1):
			lo = max(0, z - half_low)
			hi = min(vol.shape[0] - 1, z + half_high)
			slices.append(vol[lo:hi + 1].mean(axis=0))
		return np.stack(slices, axis=0)

	def _thickened_sa_cube(self, cube: np.ndarray, z0: int, z1: int, thickness_px: int) -> np.ndarray:
		arr = np.asarray(cube, dtype=np.float64)
		if arr.ndim != 4:
			raise ValueError(f"cube debe ser 4D; recibió {arr.shape}")
		return np.stack([self._thickened_sa_volume(arr[g], z0, z1, thickness_px) for g in range(arr.shape[0])], axis=0)

	def _write_cine_crudo_limits_qc(self, result, z0: int, z1: int, thickness_px: int, active_marker: str | None = None, dpi: int = 150) -> tuple[str, dict]:
		import matplotlib.pyplot as plt

		ung = np.asarray(result.ungated_volume, dtype=np.float64)
		gated_all = np.asarray(result.gated_volume, dtype=np.float64)
		# Si el estudio NO es gated (1 gate), las columnas de la derecha (gated)
		# muestran el mismo volumen ungated con un texto indicativo, para no generar
		# confusión (no hay gates que mostrar).
		is_gated = self._study_n_gates() >= 3
		if not is_gated:
			gated_ed = ung  # mismo volumen, se etiqueta como "UNGATED (sin gates)"
		else:
			# Panel gated: gate ED (fin de diástole = gate 1) por convención.
			gated_ed = gated_all[0] if (gated_all.ndim == 4 and gated_all.shape[0] > 0) else ung
		mid_y = int(np.clip(ung.shape[1] // 2, 0, ung.shape[1] - 1))
		mid_x = int(np.clip(ung.shape[2] // 2, 0, ung.shape[2] - 1))
		mid_z = int(np.clip((int(z0) + int(z1)) // 2, 0, ung.shape[0] - 1))

		# --- Vistas de referencia orientadas (resolvedor multi-cámara), resueltas
		# una sola vez y reutilizadas para ambas ramas (misma geometría). ---
		title_ap = "Vista longitudinal Y · límites"
		title_ll = "Vista longitudinal X · límites"
		ori = None
		orient_note = None
		try:
			raw_study = getattr(self, "cine_crudo_raw_study_for_recon", None) or getattr(self, "study", None)
			if raw_study is not None:
				from core.orientation_resolver import resolve_orientation
				ori = resolve_orientation(
					manufacturer=str(getattr(raw_study, "manufacturer", "") or ""),
					model=str(getattr(raw_study, "model", "") or ""),
					patient_position=str(getattr(raw_study, "patient_position", "") or ""),
					start_angle=getattr(raw_study, "start_angle", None),
					rotation_direction=str(getattr(raw_study, "rotation_direction", "") or ""),
					scan_arc=getattr(raw_study, "scan_arc", None),
					detector_iop=getattr(raw_study, "detector_iop", None),
				)
				if ori.anterior_angle_deg is not None and ori.left_lateral_angle_deg is not None:
					title_ap = f"Anterior (AP) · {ori.anterior_angle_deg:.0f}°"
					title_ll = f"Lateral izq · {ori.left_lateral_angle_deg:.0f}°"
					orient_note = f"Orientación: perfil '{ori.profile_key}' (fuente {ori.source})"
					if not ori.calibrated:
						orient_note += " — sin calibrar, verificar a ojo"
					self._log(f"[ORIENT] {orient_note}. " + " | ".join(ori.notes))
				else:
					ori = None
		except Exception as exc:
			ori = None
			self._log(f"[WARN] Resolvedor de orientación no aplicado (fallback a cortes crudos): {exc}")

		def _long_views(vol3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
			v = np.asarray(vol3d, dtype=np.float64)
			ly = v[:, mid_y, :]
			lx = v[:, :, mid_x]
			if ori is not None:
				from core.spect_geometry import reproject_view
				va = reproject_view(v, float(ori.anterior_angle_deg))
				vl = reproject_view(v, float(ori.left_lateral_angle_deg))
				if ori.mirror_ap_lr:
					va = va[:, ::-1]
				if ori.mirror_ll_lr:
					vl = vl[:, ::-1]
				ly, lx = va, vl
			return ly, lx

		# Motor de color: usar el cmap y ventana del preview (cableado con la UI).
		# Default odyssey_cool para esta ventana (más informativo que gris para
		# elegir límites base/ápex sobre el miocardio).
		qc_cmap = str(getattr(self, "cine_crudo_screen_cmap", "odyssey_cool") or "odyssey_cool")
		if qc_cmap == "gray":
			qc_cmap = "odyssey_cool"
		win_hi_pct = float(getattr(self, "cine_crudo_screen_win_high", 99.0))

		def _norm(img2d: np.ndarray) -> np.ndarray:
			arr = np.asarray(img2d, dtype=np.float64)
			p99 = float(np.percentile(arr, win_hi_pct)) if arr.size else 0.0
			return np.clip(arr / max(p99, 1e-8), 0.0, 1.0)

		ung_ly, ung_lx = _long_views(ung)
		g_ly, g_lx = _long_views(gated_ed)
		ung_sa_base = self._thickened_sa_volume(ung, int(z0), int(z0), thickness_px)[0]
		ung_sa_mid = self._thickened_sa_volume(ung, mid_z, mid_z, thickness_px)[0]
		ung_sa_apex = self._thickened_sa_volume(ung, int(z1), int(z1), thickness_px)[0]
		g_sa_base = self._thickened_sa_volume(gated_ed, int(z0), int(z0), thickness_px)[0]
		g_sa_mid = self._thickened_sa_volume(gated_ed, mid_z, mid_z, thickness_px)[0]
		g_sa_apex = self._thickened_sa_volume(gated_ed, int(z1), int(z1), thickness_px)[0]

		cfg = result.config
		ung_method = str(cfg.reconstruction_method).upper()
		gated_method = str(getattr(cfg, "gated_method", None) or cfg.reconstruction_method).upper()

		# Ungated (cols 0-2) | Gated ED (cols 3-5). Fila 0 = longitudinales + info;
		# fila 1 = SA Base/medio/Ápex.
		fig, axes = plt.subplots(2, 6, figsize=(20.5, 7.2))
		for ax in axes.ravel():
			ax.axis("off")
			ax.set_facecolor("#0b1220")

		axes[0, 0].imshow(_norm(ung_ly), cmap=qc_cmap, aspect="auto")
		axes[0, 0].set_title(f"UngGat · {title_ap}", color="white", fontsize=9, fontweight="bold")
		axes[0, 1].imshow(_norm(ung_lx), cmap=qc_cmap, aspect="auto")
		axes[0, 1].set_title(f"UngGat · {title_ll}", color="white", fontsize=9, fontweight="bold")
		axes[0, 3].imshow(_norm(g_ly), cmap=qc_cmap, aspect="auto")
		axes[0, 3].set_title(f"{'Gated ED' if is_gated else 'UNGATED (sin gates)'} · {title_ap}", color="#9fd0ff" if is_gated else "#888888", fontsize=9, fontweight="bold")
		axes[0, 4].imshow(_norm(g_lx), cmap=qc_cmap, aspect="auto")
		axes[0, 4].set_title(f"{'Gated ED' if is_gated else 'UNGATED (sin gates)'} · {title_ll}", color="#9fd0ff" if is_gated else "#888888", fontsize=9, fontweight="bold")

		for ax in (axes[0, 0], axes[0, 1], axes[0, 3], axes[0, 4]):
			ax.axis("on")
			ax.set_xticks([])
			ax.set_yticks([])
			col_base = "#ffd84d" if active_marker == "base" else "#ff4040"
			col_apex = "#ffd84d" if active_marker == "apex" else "#ff4040"
			lw_base = 2.4 if active_marker == "base" else 1.8
			lw_apex = 2.4 if active_marker == "apex" else 1.8
			ax.axhline(int(z0), color=col_base, linewidth=lw_base)
			ax.axhline(int(z1), color=col_apex, linewidth=lw_apex)
			ax.axhline(mid_z, color="#66ff66", linewidth=1.2, linestyle="--")
			ax.text(0.02, 0.05, f"Base {z0 + 1}  Ápex {z1 + 1}  Esp {thickness_px}px", transform=ax.transAxes, color="#7cf29a", fontsize=8, fontweight="bold")

		sa_panels = [
			(axes[1, 0], "SA Base", ung_sa_base, z0), (axes[1, 1], "SA medio", ung_sa_mid, mid_z), (axes[1, 2], "SA Ápex", ung_sa_apex, z1),
			(axes[1, 3], "SA Base", g_sa_base, z0), (axes[1, 4], "SA medio", g_sa_mid, mid_z), (axes[1, 5], "SA Ápex", g_sa_apex, z1),
		]
		for ax, title, img, z in sa_panels:
			ax.imshow(_norm(img), cmap=qc_cmap, vmin=0.0, vmax=1.0)
			ax.set_title(title, color="white", fontsize=9, fontweight="bold")
			ax.text(0.03, 0.05, f"SA {int(z) + 1}", transform=ax.transAxes, color="#7cf29a", fontsize=8, fontweight="bold")

		axes[0, 2].text(0.5, 0.62, f"UNGATED\n{ung_method} · {cfg.ungated_filter.kind}\n{cfg.ungated_filter.cutoff:.2f}/{cfg.ungated_filter.order}",
			ha="center", va="center", color="white", fontsize=9, fontweight="bold")
		axes[0, 2].text(0.5, 0.20, "Ajustá Base / Ápex / Esp\ny mirá las líneas rojas", ha="center", va="center", color="#7cf29a", fontsize=8)
		if orient_note:
			axes[0, 2].text(0.5, 0.02, orient_note, ha="center", va="bottom", color="#7cf29a", fontsize=6.5, wrap=True)
		axes[0, 5].text(0.5, 0.62, f"{'GATED (ED)' if is_gated else 'UNGATED'}\n{gated_method} · {cfg.gated_filter.kind}\n{cfg.gated_filter.cutoff:.2f}/{cfg.gated_filter.order}",
			ha="center", va="center", color="#9fd0ff" if is_gated else "#888888", fontsize=9, fontweight="bold")
		axes[0, 5].text(0.5, 0.20, "Elegí método/filtro gated:\nactualiza en vivo" if is_gated else "Estudio UNGATED\n(sin gates para animar)", ha="center", va="center", color="#7cf29a" if is_gated else "#888888", fontsize=8)

		fig.patch.set_facecolor("#0b1220")
		fig.suptitle("Selección de límites para cortes cardíacos — UngGat | Gated ED", color="white", fontsize=11, fontweight="bold")
		fig.tight_layout(rect=[0, 0, 1, 0.92])
		# Metadatos geométricos exactos de los paneles longitudinales interactivos
		# (fila superior, ambas ramas) en coordenadas normalizadas de figura, para
		# mapear mouse->slice. Se puede arrastrar Base/Ápex sobre cualquiera de los 4.
		def _bounds(ax) -> dict:
			b = ax.get_position().bounds
			return {"x0": float(b[0]), "y0": float(b[1]), "w": float(b[2]), "h": float(b[3])}

		meta = {
			"n_slices": int(ung.shape[0]),
			"z0": int(z0),
			"z1": int(z1),
			"thickness": int(thickness_px),
			"top_axes": {
				"left": _bounds(axes[0, 0]),
				"mid": _bounds(axes[0, 1]),
				"gleft": _bounds(axes[0, 3]),
				"gmid": _bounds(axes[0, 4]),
			},
		}
		out_png = os.path.join(self.output_dir, "raw_cut_limits_qc.png")
		# Importante: no usar bbox_inches='tight' porque recorta márgenes y rompe
		# el mapeo lineal de coordenadas de mouse sobre la imagen renderizada.
		fig.savefig(out_png, dpi=int(dpi), facecolor=fig.get_facecolor())
		plt.close(fig)
		return out_png, meta


	def _preview_cine_crudo_cut_limits(self, active_marker: str | None = None, fast: bool = False):
		if self.cine_crudo_recon_result is None:
			self._cine_crudo_cut_limits_meta = None
			return
		try:
			# Si el usuario tocó los spins, persistir esos valores en la etapa activa
			# antes de recomputar los límites (estado por etapa en modo dual).
			self._cine_crudo_capture_limits_from_spins(getattr(self, "_cine_crudo_recon_stage", "stress"))
			result = self.cine_crudo_recon_result
			z0, z1 = self._cine_crudo_cut_bounds(int(np.asarray(result.gated_volume).shape[1]))
			thickness = self._cine_crudo_cut_thickness_px()
			# Vista DUAL de límites: si ambas etapas ya están reconstruidas, mostrar
			# ESFUERZO arriba y REPOSO abajo (la 2da etapa NO pisa a la 1ra).
			sess = self._dual_session()
			stress_res = sess.stage("stress").recon_result
			rest_res = sess.stage("rest").recon_result
			dual_view = stress_res is not None and rest_res is not None
			out_pix = None
			# Fast-pass interactivo (drag de markers): renderiza SOLO la etapa
			# activa a DPI bajo y reutiliza el pixmap cacheado de la otra etapa.
			# Al soltar (o tras una pausa) se re-renderiza en HQ.
			render_dpi = 80 if fast else 150
			pix_cache = getattr(self, "_limits_stage_pix_cache", None)
			meta_cache = getattr(self, "_limits_stage_meta_cache", None)
			if not isinstance(pix_cache, dict):
				pix_cache = {}
				self._limits_stage_pix_cache = pix_cache
			if not isinstance(meta_cache, dict):
				meta_cache = {}
				self._limits_stage_meta_cache = meta_cache
			if dual_view:
				prev_stage = getattr(self, "_cine_crudo_recon_stage", "stress")
				active_stage = getattr(self, "_cine_crudo_active_stage", "stress")
				if prev_stage in ("stress", "rest"):
					active = prev_stage
				elif active_stage in ("stress", "rest"):
					active = str(active_stage)
				else:
					active = "stress"
				stage_pix: dict[str, QPixmap] = {}
				stage_meta: dict[str, dict] = {}
				try:
					for st, res in (("stress", stress_res), ("rest", rest_res)):
						# Fast: la etapa NO activa se sirve del caché (no cambió).
						if fast and st != active and st in pix_cache and st in meta_cache:
							stage_pix[st] = pix_cache[st]
							stage_meta[st] = dict(meta_cache[st])
							continue
						self._cine_crudo_recon_stage = st
						zz0, zz1 = self._cine_crudo_cut_bounds(int(np.asarray(res.gated_volume).shape[1]))
						png_st, meta_st = self._write_cine_crudo_limits_qc(
							res, zz0, zz1, thickness,
							active_marker=active_marker if st == active else None,
							dpi=render_dpi if st == active else 150,
						)
						stage_pix[st] = QPixmap(png_st)
						stage_meta[st] = dict(meta_st)
						# Cachear solo renders HQ (los fast son transitorios).
						if not fast or st != active:
							pix_cache[st] = stage_pix[st]
							meta_cache[st] = dict(meta_st)
				finally:
					self._cine_crudo_recon_stage = prev_stage
				# En fast, el pixmap activo puede tener otra resolución que el
				# cacheado; igualar ancho para que el stacking no "salte".
				if fast:
					ref = stage_pix["stress" if active == "rest" else "rest"]
					act = stage_pix[active]
					if not ref.isNull() and not act.isNull() and act.width() != ref.width():
						stage_pix[active] = act.scaledToWidth(ref.width(), Qt.TransformationMode.FastTransformation)
				top_pix, bottom_pix = stage_pix["stress"], stage_pix["rest"]
				top_label = "ESFUERZO — límites" + (" ●" if active == "stress" else "")
				bottom_label = "REPOSO — límites" + (" ●" if active == "rest" else "")
				out_pix = self._stack_cine_crudo_dual_pixmaps(top_pix, bottom_pix, top_label, bottom_label, active_stage=active)
				dmeta = dict(getattr(self, "_cine_crudo_dual_render_meta", None) or {})
				meta = dict(stage_meta[active])
				bar_h = float(dmeta.get("bar_h", 22))
				canvas_w = float(max(1, out_pix.width()))
				canvas_h = float(max(1, out_pix.height()))
				split_y = float(dmeta.get("split_y", 0))
				# Regiones en FRACCIONES del canvas (independientes del zoom con el
				# que se muestre el pixmap): fx/fy = esquina sup-izq, fw/fh = tamaño.
				regions = {
					"stress": {
						"fx": ((canvas_w - float(top_pix.width())) / 2.0) / canvas_w,
						"fy": bar_h / canvas_h,
						"fw": float(top_pix.width()) / canvas_w,
						"fh": float(top_pix.height()) / canvas_h,
						"fy_bar": 0.0,  # su barra de título arranca en 0
					},
					"rest": {
						"fx": ((canvas_w - float(bottom_pix.width())) / 2.0) / canvas_w,
						"fy": (split_y + bar_h) / canvas_h,
						"fw": float(bottom_pix.width()) / canvas_w,
						"fh": float(bottom_pix.height()) / canvas_h,
						"fy_bar": split_y / canvas_h,
					},
				}
				meta["dual_overlay"] = dict(regions[active])
				meta["dual_regions"] = regions
				meta["dual_active"] = active
				meta["dual_stage_meta"] = {
					"stress": dict(stage_meta.get("stress", {})),
					"rest": dict(stage_meta.get("rest", {})),
				}
				self._cine_crudo_cut_limits_meta = meta
			else:
				out_png, meta = self._write_cine_crudo_limits_qc(result, z0, z1, thickness, active_marker=active_marker)
				self._cine_crudo_cut_limits_meta = dict(meta)
				out_pix = QPixmap(out_png)
			self.cine_crudo_preview_mode = "cut_limits"
			# Al entrar a la pantalla de markers (Base/Ápex) la mostramos a zoom 40%,
			# para ver el volumen completo y colocar las líneas sin paneo. No pisa un
			# re-render posterior (drag/wheel): solo se fija al ENTRAR al modo.
			if getattr(self, "_last_cine_crudo_preview_mode", None) != "cut_limits":
				self.preview_zoom["cine_crudo"] = 0.4
				self.preview_zoom["comparacion_ejes"] = 0.4
			self._last_cine_crudo_preview_mode = "cut_limits"
			for tab_name in ("comparacion_ejes", "cine_crudo"):
				if tab_name in self.preview_labels:
					pix = QPixmap(out_pix)
					self.preview_pixmaps[tab_name] = pix
					self.preview_base_sizes[tab_name] = pix.size()
					self.preview_labels[tab_name].setToolTip(
						"Montaje interactivo:\n"
						"• Click: selecciona tira activa (roja).\n"
						"• Rueda: desplaza tira activa.\n"
						"• Ctrl+rueda: zoom parejo global.\n"
						"• Botón medio o Alt+drag: pan/recentrar.\n"
						"• Doble click: reset de tira activa.\n"
						"• Flechas: navegación de tiras (↑/↓ cambia eje, ←/→ desplaza)."
					)
					self._apply_preview_zoom(tab_name)
			self._select_tab_by_title("cine_crudo")
		except Exception as exc:
			self._cine_crudo_cut_limits_meta = None
			self._log(f"[WARN] Preview límites falló: {exc}")

	def _cine_crudo_limits_canvas_frac(self, event, source_label=None):
		"""Posición del evento como FRACCIÓN (0..1) del pixmap mostrado.

		Independiente del zoom: el pixmap del preview puede estar re-escalado
		(_apply_preview_zoom), así que solo las fracciones son estables.
		"""
		label = source_label
		if label is None:
			label = event.widget() if hasattr(event, "widget") else None
		if label is None or label not in self.preview_labels.values():
			return None
		shown = label.pixmap() if hasattr(label, "pixmap") else None
		if shown is None or shown.isNull():
			return None
		lw = max(1, int(label.width()))
		lh = max(1, int(label.height()))
		pw = int(shown.width())
		ph = int(shown.height())
		scale = min(lw / max(1, pw), lh / max(1, ph))
		xo = (lw - pw * scale) / 2.0
		yo = (lh - ph * scale) / 2.0
		x_img = (float(event.pos().x()) - xo) / max(1e-6, scale)
		y_img = (float(event.pos().y()) - yo) / max(1e-6, scale)
		if not (0.0 <= x_img <= float(pw - 1) and 0.0 <= y_img <= float(ph - 1)):
			return None
		return x_img / max(1.0, float(pw - 1)), y_img / max(1.0, float(ph - 1))

	def _montage_selection_key_at_event(self, event, source_label=None) -> str | None:
		"""Devuelve la fila exacta ``ETAPA:EJE`` bajo el puntero en el montaje."""
		label = source_label or (event.widget() if hasattr(event, "widget") else None)
		if label is None:
			return None
		# QLabel se redimensiona al pixmap ya escalado; usar directamente su
		# tamaño elimina el error de letterbox de _cine_crudo_limits_canvas_frac
		# que hacía caer todo click sobre la fila 0.
		lh = max(1.0, float(label.height()))
		y_shown = float(event.pos().y())
		if not 0.0 <= y_shown < lh:
			return None
		cache = getattr(self, "_montage_gray_cache", {}) or {}
		geom = cache.get("geom", ())
		rows_meta = cache.get("rows_meta", [])
		if len(geom) < 9 or not rows_meta:
			return None
		# ``geom`` contiene 11 valores en el compositor actual (incluye W/H y
		# REF_W). Desempaquetar los 10 primeros; el código anterior pedía 10
		# valores pero recibía ``geom[:9]`` y por eso TODOS los clicks fallaban.
		_panel, _pad, _title_h, _left, top, _cell_w, cell_h, _scale, _w, canvas_h = geom[:10]
		# Preview: pixmap original (canvas_h) → QLabel mostrado (label.height).
		y = y_shown * float(canvas_h) / lh
		row_idx = int((int(y) - int(top)) // max(1, int(cell_h)))
		if row_idx < 0 or row_idx >= len(rows_meta):
			return None
		row = rows_meta[row_idx]
		return str(row.get("selection_key", f"{row.get('tag') or 'ESFUERZO'}:{row.get('prefix', 'SA')}"))

	def _cine_crudo_limits_stage_at_event(self, event, source_label=None) -> str | None:
		"""En vista dual de límites: etapa ('stress'/'rest') bajo el puntero, o None."""
		meta = self._cine_crudo_cut_limits_meta
		if not meta or not isinstance(meta.get("dual_regions"), dict):
			return None
		pos = self._cine_crudo_limits_canvas_frac(event, source_label=source_label)
		if pos is None:
			return None
		fx, fy = pos
		for st, r in meta["dual_regions"].items():
			x0 = float(r.get("fx", 0.0)) - 0.01
			y0 = float(r.get("fy_bar", r.get("fy", 0.0))) - 0.01  # incluye barra título
			x1 = float(r.get("fx", 0.0)) + float(r.get("fw", 1.0)) + 0.01
			y1 = float(r.get("fy", 0.0)) + float(r.get("fh", 1.0)) + 0.01
			if x0 <= fx <= x1 and y0 <= fy <= y1:
				return str(st)
		return None

	def _cine_crudo_cut_limits_event_to_slice(self, event, source_label=None):
		"""Mapea click/drag en preview de límites a índice de slice (k).

		Funciona sobre la imagen renderizada en `comparacion_ejes` / `cine_crudo`
		durante `cine_crudo_preview_mode == 'cut_limits'`.
		"""
		meta = self._cine_crudo_cut_limits_meta
		if not meta:
			return None
		# Fracción 0..1 sobre el pixmap mostrado (robusto al zoom).
		frac = self._cine_crudo_limits_canvas_frac(event, source_label=source_label)
		if frac is None:
			return None
		fx, fy = frac
		# Vista dual (Esfuerzo arriba / Reposo abajo): remapear a la sub-imagen de
		# la etapa ACTIVA usando fracciones del canvas; clicks fuera → None.
		overlay = meta.get("dual_overlay") if isinstance(meta, dict) else None
		if overlay and "fx" in overlay:
			ox, oy = float(overlay.get("fx", 0.0)), float(overlay.get("fy", 0.0))
			ow, oh = float(overlay.get("fw", 1.0)), float(overlay.get("fh", 1.0))
			if not (ox <= fx <= ox + ow and oy <= fy <= oy + oh):
				return None
			xn = (fx - ox) / max(1e-6, ow)
			yn_img = (fy - oy) / max(1e-6, oh)
		else:
			xn = fx
			yn_img = fy
		# yb_fig: 0 abajo, 1 arriba (coords normalizadas de Matplotlib Figure)
		yb_fig = 1.0 - yn_img
		top = meta.get("top_axes", {})
		in_top_axis = False
		y0_match = 0.0
		h_match = 1.0
		for ax in top.keys():
			r = top.get(ax, {})
			x0 = float(r.get("x0", 0.0)); y0 = float(r.get("y0", 0.0))
			w = float(r.get("w", 0.0)); h = float(r.get("h", 0.0))
			# pequeño margen para que sea más fácil "agarrar" la línea
			if (x0 - 0.01) <= xn <= (x0 + w + 0.01) and (y0 - 0.01) <= yb_fig <= (y0 + h + 0.01):
				in_top_axis = True
				y0_match = y0
				h_match = h
				break
		if not in_top_axis:
			return None
		nz = int(meta.get("n_slices", 1))
		# En figura Matplotlib, y crece hacia arriba; para filas de imagen (z),
		# queremos 0 en el borde superior del eje y 1 en el inferior.
		y_rel = ((y0_match + h_match) - yb_fig) / max(1e-6, h_match)
		z = int(np.clip(round(y_rel * max(0, nz - 1)), 0, max(0, nz - 1)))
		return z

	def _cine_crudo_marker_at_limits_event(self, event, source_label=None) -> str | None:
		"""Detecta si el puntero está cerca de la línea Base/Ápex en preview límites."""
		meta = self._cine_crudo_cut_limits_meta
		if not meta:
			return None
		z = self._cine_crudo_cut_limits_event_to_slice(event, source_label=source_label)
		if z is None:
			return None
		z0 = int(meta.get("z0", 0))
		z1 = int(meta.get("z1", 0))
		if abs(z - z0) <= 2:
			return "base"
		if abs(z - z1) <= 2:
			return "apex"
		return None

	def _update_cine_crudo_cut_spins_from_drag(self, marker: str, z: int):
		if marker not in {"base", "apex"}:
			return
		if self.cine_crudo_recon_result is None:
			return
		stage = str(getattr(self, "_cine_crudo_recon_stage", "stress") or "stress")
		nz = int(np.asarray(self.cine_crudo_recon_result.gated_volume).shape[1])
		z = int(np.clip(z, 0, max(0, nz - 1)))
		base_1, apex_1 = self._cine_crudo_stage_limits_get(stage, nz)
		if marker == "base":
			base_1 = z + 1
		else:
			apex_1 = z + 1
		# Mantener coherencia base<=ápex sin bloquear inversión del usuario.
		if base_1 > apex_1:
			if marker == "base":
				apex_1 = base_1
			else:
				base_1 = apex_1
		base_1, apex_1 = self._cine_crudo_stage_limits_set(stage, base_1, apex_1, nz)
		if hasattr(self, "cine_crudo_cut_base_spin") and hasattr(self, "cine_crudo_cut_apex_spin"):
			self.cine_crudo_cut_base_spin.blockSignals(True)
			self.cine_crudo_cut_apex_spin.blockSignals(True)
			self.cine_crudo_cut_base_spin.setValue(int(base_1))
			self.cine_crudo_cut_apex_spin.setValue(int(apex_1))
			self.cine_crudo_cut_base_spin.blockSignals(False)
			self.cine_crudo_cut_apex_spin.blockSignals(False)
		# Durante arrastre continuo (self._cine_crudo_drag_marker activo) usar el
		# fast-pass (DPI bajo + caché de la otra etapa) y programar re-render HQ
		# al soltar/pausar, para que el movimiento sea fluido.
		dragging = getattr(self, "_cine_crudo_drag_marker", None) in {"base", "apex"}
		self._preview_cine_crudo_cut_limits(active_marker=marker, fast=dragging)
		if dragging:
			self._schedule_cut_limits_hq_render()

	def _schedule_cut_limits_hq_render(self):
		"""Re-render HQ diferido de la pantalla de límites tras interacción."""
		timer = getattr(self, "_cut_limits_hq_timer", None)
		if timer is None:
			timer = QTimer(self)
			timer.setSingleShot(True)
			timer.timeout.connect(self._render_cut_limits_hq)
			self._cut_limits_hq_timer = timer
		timer.start(140)

	def _render_cut_limits_hq(self):
		if getattr(self, "cine_crudo_preview_mode", None) != "cut_limits":
			return
		if getattr(self, "cine_crudo_recon_result", None) is None:
			return
		try:
			self._preview_cine_crudo_cut_limits(fast=False)
		except Exception as exc:
			self._log(f"[WARN] Re-render HQ de límites falló: {exc}")

	def _heart_crop_window(self, ung_full: np.ndarray, z0: int, z1: int) -> tuple[int, int, int, int]:
		"""Centro y radio del corazón (y0,y1,x0,x1) desde el slab base→ápex."""
		vol = np.asarray(ung_full, dtype=np.float64)
		z0i = int(np.clip(z0, 0, vol.shape[0] - 1))
		z1i = int(np.clip(z1, 0, vol.shape[0] - 1))
		slab = vol[min(z0i, z1i):max(z0i, z1i) + 1].sum(axis=0)
		H, W = slab.shape
		mx = float(slab.max()) if slab.size else 0.0
		if mx <= 0.0:
			return 0, H - 1, 0, W - 1
		mask = slab > 0.35 * mx
		ys, xs = np.where(mask)
		if ys.size < 4:
			cy, cx = H // 2, W // 2
			r = max(12, min(H, W) // 4)
		else:
			cy = int(round(float(ys.mean())))
			cx = int(round(float(xs.mean())))
			ry = int(np.ceil((ys.max() - ys.min()) / 2.0))
			rx = int(np.ceil((xs.max() - xs.min()) / 2.0))
			r = int(np.clip(max(ry, rx) + 6, 12, max(H, W) // 2))
		y0 = int(np.clip(cy - r, 0, H - 1))
		y1 = int(np.clip(cy + r, 0, H - 1))
		x0 = int(np.clip(cx - r, 0, W - 1))
		x1 = int(np.clip(cx + r, 0, W - 1))
		return y0, y1, x0, x1

	def _write_cine_crudo_cuts_qc(self, ung_full: np.ndarray, z0: int, z1: int) -> str:
		import matplotlib.pyplot as plt

		vol = np.asarray(ung_full, dtype=np.float64)
		n_slices = int(vol.shape[0])
		z0 = int(np.clip(z0, 0, n_slices - 1))
		z1 = int(np.clip(z1, 0, n_slices - 1))
		mid_slice = int(np.clip((z0 + z1) // 2, 0, n_slices - 1))
		thickness = self._cine_crudo_cut_thickness_px()
		# Guarda los argumentos para re-render en vivo al mover "Suav. cortes".
		self._cine_crudo_cuts_qc_args = (np.array(vol, copy=True), int(z0), int(z1))

		# Interpolación de VISUALIZACIÓN (no toca datos) y gaussiano extra, ambos
		# independientes: el combo da el continuo nítido→suave, el spin agrega
		# difuminado gaussiano encima si el médico lo quiere.
		_interp_map = {"Píxel": "nearest", "Bilineal": "bilinear", "Bicúbico": "bicubic", "Hanning": "hanning", "Lanczos": "lanczos"}
		interp = "bilinear"
		if hasattr(self, "cine_crudo_cuts_interp_combo") and self.cine_crudo_cuts_interp_combo is not None:
			interp = _interp_map.get(self.cine_crudo_cuts_interp_combo.currentText(), "bilinear")
		cuts_smooth = 0.0
		if hasattr(self, "cine_crudo_cuts_smooth_spin") and self.cine_crudo_cuts_smooth_spin is not None:
			cuts_smooth = float(self.cine_crudo_cuts_smooth_spin.value())

		try:
			from scipy.ndimage import gaussian_filter
		except Exception:
			gaussian_filter = None

		# Motor de color: usar el cmap y ventana del preview (cableado con la UI).
		# Default odyssey_cool para esta ventana (igual que Selección de límites).
		qc_cmap = str(getattr(self, "cine_crudo_screen_cmap", "odyssey_cool") or "odyssey_cool")
		if qc_cmap == "gray":
			qc_cmap = "odyssey_cool"
		win_lo_pct = float(getattr(self, "cine_crudo_screen_win_low", 5.0))
		win_hi_pct = float(getattr(self, "cine_crudo_screen_win_high", 99.5))

		def _norm(img2d: np.ndarray) -> np.ndarray:
			arr = np.asarray(img2d, dtype=np.float64)
			if gaussian_filter is not None and arr.ndim == 2 and cuts_smooth > 0.0:
				arr = gaussian_filter(arr, sigma=cuts_smooth)
			p99 = float(np.percentile(arr, win_hi_pct)) if arr.size else 0.0
			p5 = float(np.percentile(arr, win_lo_pct)) if arr.size else 0.0
			return np.clip((arr - p5) / max(p99 - p5, 1e-8), 0.0, 1.0)

		# El volumen recibido ya está SA-alineado (reorientado): axis 0 = k
		# (base→ápex), axis 1 = j (vertical anat.), axis 2 = i (horizontal anat.).
		# Se usan los extractores anatómicos (convención Xeleris/Odyssey) para
		# que esta QC coincida EXACTAMENTE con los cortes generados y el diálogo.
		from core.cardiac_reorientation import hla_slice, sa_slice, vla_slice

		jmid = int(np.clip(vol.shape[1] // 2, 0, vol.shape[1] - 1))
		imid = int(np.clip(vol.shape[2] // 2, 0, vol.shape[2] - 1))

		# SA en el medio del slab base→ápex (orientación anatómica fija).
		sa_crop = sa_slice(vol, mid_slice)
		# Ejes largos completos (altura real base→ápex).
		hla_view = hla_slice(vol, jmid)   # APEX↑ BASE↓, SEP← LAT→
		vla_view = vla_slice(vol, imid)   # ANT↑ INF↓, BASE← APEX→
		# Recortes base→ápex de los ejes largos (líneas de límite luego).
		hla_cut = hla_view
		vla_cut = vla_view

		fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.3), gridspec_kw={"height_ratios": [1.05, 1.0]})
		for ax in axes.ravel():
			ax.axis("off")
			ax.set_facecolor("#020611")

		# Fila superior: localización SA + límites en los ejes largos (líneas rojas base/ápex).
		axes[0, 0].imshow(_norm(sa_crop), cmap=qc_cmap, vmin=0.0, vmax=1.0, interpolation=interp)
		axes[0, 0].set_title("Localización SA", color="white", fontsize=9, fontweight="bold")
		axes[0, 0].axhline((sa_crop.shape[0] - 1) / 2.0, color="#40ff5a", linewidth=1.0)
		axes[0, 0].axvline((sa_crop.shape[1] - 1) / 2.0, color="#40ff5a", linewidth=1.0)
		axes[0, 0].text(0.03, 0.05, f"SA {mid_slice + 1}", transform=axes[0, 0].transAxes, color="#7cf29a", fontsize=8, fontweight="bold")

		nk = int(vol.shape[0])
		# VLA: base→ápex en columnas (BASE izq / APEX der) → líneas verticales.
		axes[0, 1].imshow(_norm(vla_view), cmap=qc_cmap, vmin=0.0, vmax=1.0, aspect="auto", interpolation=interp)
		axes[0, 1].set_title("VLA limits · ANT↑ BASE←", color="white", fontsize=9, fontweight="bold")
		axes[0, 1].axvline(z0, color="#ff3333", linewidth=1.6)
		axes[0, 1].axvline(z1, color="#ff3333", linewidth=1.6)
		axes[0, 1].axvline(mid_slice, color="#40ff5a", linewidth=1.0, linestyle="--")
		axes[0, 1].text(0.03, 0.05, f"Base {z0 + 1}  Ápex {z1 + 1}  Esp {thickness}px", transform=axes[0, 1].transAxes, color="#7cf29a", fontsize=8, fontweight="bold")
		# HLA: APEX arriba / BASE abajo (fila k invertida) → líneas horizontales
		# en coordenada de fila invertida k' = (nk-1) - k.
		axes[0, 2].imshow(_norm(hla_view), cmap=qc_cmap, vmin=0.0, vmax=1.0, aspect="auto", interpolation=interp)
		axes[0, 2].set_title("HLA limits · APEX↑ SEP←", color="white", fontsize=9, fontweight="bold")
		axes[0, 2].axhline((nk - 1) - z0, color="#ff3333", linewidth=1.6)
		axes[0, 2].axhline((nk - 1) - z1, color="#ff3333", linewidth=1.6)
		axes[0, 2].axhline((nk - 1) - mid_slice, color="#40ff5a", linewidth=1.0, linestyle="--")
		axes[0, 2].text(0.03, 0.05, f"Base {z0 + 1}  Ápex {z1 + 1}  Esp {thickness}px", transform=axes[0, 2].transAxes, color="#7cf29a", fontsize=8, fontweight="bold")

		# Fila inferior: cortes generados con convención anatómica Xeleris/Odyssey.
		for ax, title, img, marker in [
			(axes[1, 0], "VLA", vla_cut, "ANT↑ · BASE← APEX→"),
			(axes[1, 1], "HLA", hla_cut, "APEX↑ · SEP← LAT→"),
			(axes[1, 2], "SA", sa_crop, "ANT↑ · SEP← LAT→"),
		]:
			ax.imshow(_norm(img), cmap=qc_cmap, vmin=0.0, vmax=1.0, aspect="auto" if title != "SA" else "equal", interpolation=interp)
			ax.set_title(title, color="white", fontsize=10, fontweight="bold")
			ax.text(0.03, 0.05, marker, transform=ax.transAxes, color="#e8f5e9", fontsize=8, fontweight="bold")
		fig.patch.set_facecolor("#0b1220")
		fig.suptitle(
			f"Cortes generados · límites Base={z0 + 1} Ápex={z1 + 1} · Esp {thickness}px · "
			f"fuente: {self.cine_crudo_cut_source_label or 'raw recon'}",
			color="white", fontsize=11, fontweight="bold",
		)
		fig.tight_layout(rect=[0, 0, 1, 0.91], pad=1.0)
		out_png = os.path.join(self.output_dir, "raw_generated_axes_qc.png")
		fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
		plt.close(fig)
		return out_png

	def _refresh_cine_crudo_cuts_smoothing(self) -> None:
		"""Re-renderiza el QC de cortes al mover 'Suav. cortes' (solo display).

		No recomputa el volumen ni el análisis: reusa los cortes ya generados y
		aplica el post-filtro de visualización elegido. Silencioso si aún no hay
		cortes generados en pantalla.
		"""
		if getattr(self, "cine_crudo_preview_mode", None) != "generated_cuts":
			return
		args = getattr(self, "_cine_crudo_cuts_qc_args", None)
		if not args:
			return
		ung_vol, z0, z1 = args
		try:
			out_png = self._write_cine_crudo_cuts_qc(np.asarray(ung_vol, dtype=np.float64), int(z0), int(z1))
		except Exception as exc:
			self._log(f"[WARN] Re-render de suavizado de cortes falló: {exc}")
			return
		for tab_name in ("comparacion_ejes", "cine_crudo"):
			if tab_name in self.preview_labels:
				pix = QPixmap(out_png)
				self.preview_pixmaps[tab_name] = pix
				self.preview_base_sizes[tab_name] = pix.size()
				self._apply_preview_zoom(tab_name)

	def _refresh_cine_crudo_cuts_color(self) -> None:
		"""Re-renderiza el QC de cortes al cambiar cmap/ventana del motor de color.

		Mismo mecanismo que _refresh_cine_crudo_cuts_smoothing pero para cambios
		de colormap o ventana (Base/Top).
		"""
		if getattr(self, "cine_crudo_preview_mode", None) != "generated_cuts":
			return
		args = getattr(self, "_cine_crudo_cuts_qc_args", None)
		if not args:
			return
		ung_vol, z0, z1 = args
		try:
			out_png = self._write_cine_crudo_cuts_qc(np.asarray(ung_vol, dtype=np.float64), int(z0), int(z1))
		except Exception as exc:
			self._log(f"[WARN] Re-render de color de cortes falló: {exc}")
			return
		for tab_name in ("comparacion_ejes", "cine_crudo"):
			if tab_name in self.preview_labels:
				pix = QPixmap(out_png)
				self.preview_pixmaps[tab_name] = pix
				self.preview_base_sizes[tab_name] = pix.size()
				self._apply_preview_zoom(tab_name)

	def _refresh_cine_crudo_cut_limits_color(self) -> None:
		"""Re-renderiza la QC de límites al cambiar cmap/ventana del motor de color.

		Solo si estamos en modo cut_limits (Selección de límites para cortes).
		"""
		if getattr(self, "cine_crudo_preview_mode", None) != "cut_limits":
			return
		if getattr(self, "cine_crudo_recon_result", None) is None:
			return
		try:
			self._preview_cine_crudo_cut_limits()
		except Exception as exc:
			self._log(f"[WARN] Re-render de color de límites falló: {exc}")

	def _reorient_locked_voi_for_stage(self):
		"""Devuelve el VOI (semiejes) bloqueado si otra etapa ya reorientó, o None.

		El zoom de la elipse se fija con la PRIMERA reorientación de la sesión; toda
		reorientación posterior (sea la etapa que sea) arranca con esa misma
		magnificación para no simular dilatación del VI. Que ambas etapas usen el
		mismo zoom prima sobre poder re-editarlo libremente.
		"""
		locked = getattr(self, "_reorient_locked_voi", None)
		if not locked:
			return None
		return dict(locked)

	def _register_reorient_zoom_lock(self, dlg):
		"""Registra el zoom (semiejes de la elipse) de la primera reorientación."""
		voi = {
			"rz": float(getattr(dlg, "_voi_rz", 0.0)),
			"ry": float(getattr(dlg, "_voi_ry", 0.0)),
			"rx": float(getattr(dlg, "_voi_rx", 0.0)),
		}
		current_stage = getattr(self, "_cine_crudo_recon_stage", None)
		stage_txt = {"stress": "Esfuerzo", "rest": "Reposo"}.get(str(current_stage), str(current_stage))
		if not getattr(self, "_reorient_locked_voi", None):
			self._reorient_locked_voi = voi
			self._reorient_locked_stage = current_stage
			self._log(
				f"[ZOOM] Zoom de reorientación FIJADO por la etapa '{stage_txt}' "
				f"(semiejes rz={voi['rz']:.1f} ry={voi['ry']:.1f} rx={voi['rx']:.1f}); "
				"la otra etapa arrancará con este mismo zoom bloqueado."
			)
		else:
			locked = self._reorient_locked_voi
			lock_stage_txt = {"stress": "Esfuerzo", "rest": "Reposo"}.get(
				str(getattr(self, "_reorient_locked_stage", None)),
				str(getattr(self, "_reorient_locked_stage", None)),
			)
			self._log(
				f"[ZOOM] Reorientación de '{stage_txt}' con zoom bloqueado por '{lock_stage_txt}' "
				f"(rz={locked['rz']:.1f} ry={locked['ry']:.1f} rx={locked['rx']:.1f})."
			)

	def _reorient_seed_for_stage(self):
		"""Semilla de orientación (eje largo + rango de cortes + espesor) heredada.

		La primera reorientación de la sesión define el punto de partida; la otra
		etapa arranca con esos mismos valores pero editables (no bloqueados)."""
		seed = getattr(self, "_reorient_seed", None)
		if not seed:
			return None
		return dict(seed)

	def _register_reorient_seed(self, dlg):
		"""Registra la orientación de la primera reorientación como semilla editable."""
		u = getattr(dlg, "result_long_axis", None)
		half = float(getattr(dlg, "result_half_length", 0.0) or 0.0)
		out = int(getattr(dlg, "result_out_size", 0) or 0)
		if u is None or half <= 0.0 or out <= 0:
			return
		try:
			u_list = [float(x) for x in np.asarray(u, dtype=np.float64).ravel()[:3]]
		except (TypeError, ValueError):
			return
		seed = {
			"long_axis": u_list,
			"half_length": half,
			"base_frac": float(dlg.base_k) / max(1, out),
			"apex_frac": float(dlg.apex_k) / max(1, out),
			"thickness": int(getattr(dlg, "thickness", 1) or 1),
		}
		current_stage = getattr(self, "_cine_crudo_recon_stage", None)
		stage_txt = {"stress": "Esfuerzo", "rest": "Reposo"}.get(str(current_stage), str(current_stage))
		if not getattr(self, "_reorient_seed", None):
			self._reorient_seed = seed
			self._reorient_seed_stage = current_stage
			self._log(
				f"[REORIENT] Orientación de referencia FIJADA por la etapa '{stage_txt}' "
				f"(eje largo + Base/Ápex + espesor); la otra etapa arrancará igual, pero editable."
			)

	def _build_reorient_dialog_kwargs(self, stage: str) -> dict | None:
		"""Arma los kwargs del CardiacReorientationDialog para una etapa dada."""
		st = self._dual_session().stage(stage)
		result = st.recon_result
		if result is None:
			return None
		ung_vol = np.asarray(result.ungated_volume, dtype=np.float64)
		gated_vol = np.asarray(result.gated_volume, dtype=np.float64)
		raw_study = st.raw_study_for_recon or st.raw_study or self.study
		geometry = None
		if raw_study is not None:
			try:
				from core.spect_geometry import SpectGeometry
				geometry = SpectGeometry(
					patient_position=str(getattr(raw_study, "patient_position", "") or ""),
					start_angle=getattr(raw_study, "start_angle", None),
					angular_step=getattr(raw_study, "angular_step", None),
					rotation_direction=str(getattr(raw_study, "rotation_direction", "") or ""),
					scan_arc=getattr(raw_study, "scan_arc", None),
					n_angles=int(getattr(raw_study, "n_slices", 0) or 0),
				)
			except Exception as exc:
				self._log(f"[WARN] No se pudo derivar geometría de adquisición ({stage}): {exc}")
				geometry = None
		_reo_px = getattr(raw_study, "pixel_spacing", None) if raw_study is not None else None
		_reo_voxel_mm = float(_reo_px[0]) if _reo_px else None
		phase_res = st.recon_result_phase
		ct_on_grid = None
		try:
			ct_on_grid = self._stage_ct_on_recon_grid(stage, ung_vol.shape)
		except Exception as exc:
			self._log(f"[FUSION][WARN] CT no disponible para reorientar ({stage}): {exc}")
		return {
			"ungated_volume": ung_vol,
			"gated_volume": gated_vol,
			"source_label": st.cut_source_label or "raw recon",
			"geometry": geometry,
			"voxel_mm": _reo_voxel_mm,
			"locked_voi": self._reorient_locked_voi_for_stage(),
			"initial_orientation": self._reorient_seed_for_stage(),
			"ct_volume": ct_on_grid,
			"phase_gated_volume": (
				np.asarray(phase_res.gated_volume, dtype=np.float64)
				if phase_res is not None else None
			),
			"motion_frozen_volume": (
				np.asarray(result.ungated_volume_mf, dtype=np.float64)
				if getattr(result, "ungated_volume_mf", None) is not None else None
			),
		}

	def _open_cine_crudo_dual_reorientation(self) -> bool:
		"""Reorientación EN PARALELO (pantalla partida Esfuerzo | Reposo)."""
		kw_stress = self._build_reorient_dialog_kwargs("stress")
		kw_rest = self._build_reorient_dialog_kwargs("rest")
		missing = [txt for txt, kw in (("Esfuerzo", kw_stress), ("Reposo", kw_rest)) if kw is None]
		if missing:
			QMessageBox.information(
				self, "SINCRO",
				f"Reorientación dual: falta reconstruir {' y '.join(missing)}.\n"
				"Ejecutá 'Recon raw' en Ambas primero.",
			)
			return False
		try:
			from ui.reorientation_dual_dialog import DualCardiacReorientationDialog
		except Exception as exc:
			QMessageBox.warning(self, "SINCRO", f"No se pudo abrir la reorientación dual:\n{exc}")
			return False
		_undo_group = self.UNDO_ATTRS_REORIENT + self.UNDO_ATTRS_CUTS
		_undo_before = None if getattr(self, "_undo_suspended", False) else self._snapshot_attrs(_undo_group, deep=False)
		self._log("[REORIENT-DUAL] Abriendo reorientación en paralelo (Esfuerzo | Reposo).")
		dlg = DualCardiacReorientationDialog(kw_stress, kw_rest, parent=self)
		if dlg.exec() != QDialog.DialogCode.Accepted:
			return False
		prev_active = getattr(self, "_cine_crudo_active_stage", "both")
		prev_recon = getattr(self, "_cine_crudo_recon_stage", "stress")
		ok_all = True
		try:
			for stage, panel in (("stress", dlg.panel_stress), ("rest", dlg.panel_rest)):
				if panel.reoriented_gated is None:
					ok_all = False
					continue
				self._set_active_cine_crudo_stage(stage, refresh_view=False, force=True)
				self._cine_crudo_recon_stage = stage
				if not self._apply_reorientation_result(panel, generate_cuts=True):
					ok_all = False
		finally:
			self._cine_crudo_recon_stage = prev_recon
			self._set_active_cine_crudo_stage(prev_active, refresh_view=False, force=True)
		self._commit_undo("Reorientación dual", _undo_group, _undo_before, deep=False)
		if ok_all:
			self._log("[REORIENT-DUAL] Ambas etapas reorientadas con elipse igualada. Cortes generados.")
			try:
				self._show_cine_crudo_sa_montage()
			except Exception as exc:
				self._log(f"[WARN] Montaje dual post-reorientación falló: {exc}")
		return ok_all

	def _open_cine_crudo_reorientation(self, _force_stage: str | None = None):
		"""Abre el diálogo interactivo de reorientación oblicua (Rec/Ref estilo Xeleris)."""
		# La señal clicked(bool) de Qt pasa False como primer posicional: cualquier
		# valor no-etapa se normaliza a None para que la rama dual se evalúe.
		if _force_stage not in ("stress", "rest"):
			_force_stage = None
		if _force_stage is None:
			# Criterio robusto para DUAL: ambas etapas RECONSTRUIDAS en la sesión
			# (independiente del selector Etapa, que puede haberse degradado a
			# 'stress' por un falso negativo del detector de secundario).
			sess = self._dual_session()
			both_recon = (
				sess.stage("stress").recon_result is not None
				and sess.stage("rest").recon_result is not None
			)
			if both_recon:
				# Con dos etapas: ventana DUAL en paralelo (pantalla partida).
				return self._open_cine_crudo_dual_reorientation()
			stages = self._cine_crudo_target_stages()
			_force_stage = stages[0]
		if _force_stage:
			self._set_active_cine_crudo_stage(str(_force_stage), refresh_view=False, force=True)
			self._cine_crudo_recon_stage = str(_force_stage)
		if self.cine_crudo_recon_result is None:
			QMessageBox.information(self, "SINCRO", "Primero reconstruí el crudo con Recon raw.")
			return False
		try:
			from ui.reorientation_dialog import CardiacReorientationDialog
		except Exception as exc:
			QMessageBox.warning(self, "SINCRO", f"No se pudo abrir la reorientación:\n{exc}")
			return False
		_undo_group = self.UNDO_ATTRS_REORIENT + self.UNDO_ATTRS_CUTS
		_undo_before = None if getattr(self, "_undo_suspended", False) else self._snapshot_attrs(_undo_group, deep=False)
		stage_now = str(getattr(self, "_cine_crudo_recon_stage", "stress") or "stress")
		kw = self._build_reorient_dialog_kwargs(stage_now)
		if kw is None:
			QMessageBox.information(self, "SINCRO", "Primero reconstruí el crudo con Recon raw.")
			return False
		stage_txt = {"stress": "Esfuerzo", "rest": "Reposo"}.get(stage_now, stage_now)
		self._log(f"[REORIENT] Reorientando la etapa '{stage_txt}'.")
		dlg = CardiacReorientationDialog(parent=self, **kw)
		try:
			dlg.setWindowTitle(f"Reorientación oblicua · {'ESFUERZO' if stage_now == 'stress' else 'REPOSO'}")
		except Exception:
			pass
		if dlg.exec() != QDialog.DialogCode.Accepted or dlg.reoriented_gated is None:
			return False
		ok = self._apply_reorientation_result(dlg, generate_cuts=True)
		self._commit_undo("Reorientación", _undo_group, _undo_before, deep=False)
		return ok

	def _apply_reorientation_result(self, dlg, generate_cuts: bool = True) -> bool:
		"""Aplica el resultado de un diálogo/panel de reorientación a la etapa activa."""
		if getattr(dlg, "reoriented_gated", None) is None:
			return False
		self.cine_crudo_reoriented_gated = np.asarray(dlg.reoriented_gated, dtype=np.float64)
		self.cine_crudo_reoriented_ungated = (
			np.asarray(dlg.reoriented_ungated, dtype=np.float64)
			if dlg.reoriented_ungated is not None else None
		)
		self.cine_crudo_reoriented_mf = (
			np.asarray(dlg.reoriented_mf, dtype=np.float64)
			if getattr(dlg, "reoriented_mf", None) is not None else None
		)
		# Pasajero de fase reorientado con la misma transformación (o None si no hubo).
		self.cine_crudo_reoriented_gated_phase = (
			np.asarray(dlg.reoriented_gated_phase, dtype=np.float64)
			if getattr(dlg, "reoriented_gated_phase", None) is not None else None
		)
		# CT registrado reorientado (mismo reslice, sin VOI) para fusión en ejes.
		self.cine_crudo_reoriented_ct = (
			np.asarray(dlg.reoriented_ct, dtype=np.float64)
			if getattr(dlg, "reoriented_ct", None) is not None else None
		)
		n = int(self.cine_crudo_reoriented_gated.shape[1])
		base_1 = int(np.clip(dlg.base_k + 1, 1, n))
		apex_1 = int(np.clip(dlg.apex_k + 1, 1, n))
		if hasattr(self, "cine_crudo_cut_base_spin") and hasattr(self, "cine_crudo_cut_apex_spin"):
			for sp in (self.cine_crudo_cut_base_spin, self.cine_crudo_cut_apex_spin):
				sp.blockSignals(True)
				sp.setEnabled(True)
				sp.setRange(1, max(1, n))
			self.cine_crudo_cut_base_spin.setValue(base_1)
			self.cine_crudo_cut_apex_spin.setValue(apex_1)
			for sp in (self.cine_crudo_cut_base_spin, self.cine_crudo_cut_apex_spin):
				sp.blockSignals(False)
		self._cine_crudo_stage_limits_set(getattr(self, "_cine_crudo_recon_stage", "stress"), base_1, apex_1, n)
		if hasattr(self, "cine_crudo_cut_thickness_spin"):
			self.cine_crudo_cut_thickness_spin.setEnabled(True)
			self.cine_crudo_cut_thickness_spin.setValue(int(getattr(dlg, "thickness", 1)))
		# Guardar VOI elíptica de la reorientación (para recorte "Elipse VOI").
		try:
			self.cine_crudo_reoriented_voi = {
				"cz": float(getattr(dlg, "_voi_cz", 0.0)),
				"rz": float(getattr(dlg, "_voi_rz", 0.0)),
			}
		except Exception:
			self.cine_crudo_reoriented_voi = None
		# Bloqueo de zoom entre etapas: registrar los semiejes de la elipse (la
		# magnificación) de la PRIMERA reorientación para forzarlos en la 2da etapa.
		self._register_reorient_zoom_lock(dlg)
		# Semilla de orientación (eje largo + rango de cortes + espesor): la 2da
		# etapa arrancará igual que esta, pero podrá corregirla si hace falta.
		self._register_reorient_seed(dlg)
		# Opción A del harness NÍTIDA: volcar el eje largo + centro + volúmenes
		# reorientados a disco, para que el diagnóstico use la orientación REAL
		# aprobada por el usuario (no un eje automático).
		try:
			self._dump_reorient_for_harness(dlg)
		except Exception as exc:
			self._log(f"[WARN] No pude volcar la reorientación para el harness: {exc}")
		self.cine_crudo_cut_source_label = (self.cine_crudo_cut_source_label or "raw recon") + " · reorientado"
		self._log(
			f"Reorientación aplicada: azimut/elevación oblicuos; volumen SA-alineado {self.cine_crudo_reoriented_gated.shape}; "
			f"Base={base_1} Ápex={apex_1}. Generando cortes."
		)
		# Marcar reorient al día ANTES de regenerar cortes: así invalidate_after
		# desactualiza los posteriores y luego 'cuts' se re-marca al día.
		self._mark_step_done("reorient", getattr(self.cine_crudo_reoriented_gated, "shape", None), base_1, apex_1)
		if generate_cuts:
			_prev_suspended = getattr(self, "_undo_suspended", False)
			self._undo_suspended = True
			try:
				# Forzar la etapa actual: NO pasar por el orquestador "Ambas"
				# (en el flujo dual cada panel aplica sus propios cortes).
				self._generate_cine_crudo_cardiac_cuts(
					_force_stage=str(getattr(self, "_cine_crudo_recon_stage", "stress") or "stress")
				)
			finally:
				self._undo_suspended = _prev_suspended
		# Habilitar el editor de informe ahora que hay datos reorientados.
		if hasattr(self, "html_menu"):
			for action in self.html_menu.actions():
				if "Editor" in action.text():
					action.setEnabled(True)
					break
		return True

	def _generate_cine_crudo_cardiac_cuts(self, _force_stage: str | None = None):
		"""Genera los cortes cardíacos desde el volumen reconstruido; SA alimenta fase/FEVI."""
		# clicked(bool) de Qt pasa False posicional: normalizar a None.
		if _force_stage not in ("stress", "rest"):
			_force_stage = None
		if _force_stage is None:
			stages = self._cine_crudo_target_stages()
			if len(stages) > 1:
				# Evitar mezcla inconsistente: si una etapa ya fue reorientada y la otra
				# no, NO auto-generar cortes en "Ambas". Primero completar la
				# reorientación faltante para mantener comparabilidad clínica.
				try:
					sess = self._dual_session()
					reo_by_stage = {st: bool(getattr(sess.stage(st), "reoriented_gated", None) is not None) for st in stages}
					if any(reo_by_stage.values()) and not all(reo_by_stage.values()):
						pend = ["Esfuerzo" if st == "stress" else "Reposo" for st, ok in reo_by_stage.items() if not ok]
						QMessageBox.information(
							self,
							"SINCRO",
							"Reorientación dual incompleta.\n\n"
							f"Falta reorientar: {', '.join(pend)}.\n"
							"Para evitar que una etapa se genere automáticamente sin reorientación equivalente, "
							"completá Reorientación en ambas etapas y luego generá cortes.",
						)
						return False
				except Exception:
					pass
				return self._run_cine_crudo_stage_orchestrator(
					"Generar cortes",
					lambda stage: self._generate_cine_crudo_cardiac_cuts(_force_stage=stage),
				)
			_force_stage = stages[0]
		if _force_stage:
			self._set_active_cine_crudo_stage(str(_force_stage), refresh_view=False, force=True)
			self._cine_crudo_recon_stage = str(_force_stage)
		if self.cine_crudo_recon_result is None:
			QMessageBox.information(self, "SINCRO", "Primero reconstruí el crudo con Recon raw.")
			return False
		_undo_before = None if getattr(self, "_undo_suspended", False) else self._snapshot_attrs(self.UNDO_ATTRS_CUTS)
		try:
			result = self.cine_crudo_recon_result
			raw_study = self.cine_crudo_raw_study_for_recon or self.study
			gated_vol = np.asarray(result.gated_volume, dtype=np.float64)
			ung_vol = np.asarray(result.ungated_volume, dtype=np.float64)
			# Si hubo reorientación oblicua, usar el volumen SA-alineado.
			if getattr(self, "cine_crudo_reoriented_gated", None) is not None:
				gated_vol = np.asarray(self.cine_crudo_reoriented_gated, dtype=np.float64)
				if getattr(self, "cine_crudo_reoriented_ungated", None) is not None:
					ung_vol = np.asarray(self.cine_crudo_reoriented_ungated, dtype=np.float64)
			z0, z1 = self._cine_crudo_cut_bounds(int(gated_vol.shape[1]))
			thickness = self._cine_crudo_cut_thickness_px()
			# Volumen SA-alineado recortado base→ápex, ejes (g, k, j, i).
			reo_cube = self._thickened_sa_cube(gated_vol, z0, z1, thickness)
			# Cortes anatómicos con convención Xeleris/Odyssey (única fuente de
			# verdad de orientación; SA alimenta fase/FEVI).
			from core.cardiac_reorientation import anatomical_cuts_gated
			cuts = anatomical_cuts_gated(reo_cube)
			sa_cube = np.ascontiguousarray(cuts["sa"])
			hla_cube = np.ascontiguousarray(cuts["hla"])
			vla_cube = np.ascontiguousarray(cuts["vla"])
			# Cortes del UNGATED (imagen de perfusión estática, con Denoise+ si está
			# activo): mismo recorte base→ápex y mismos cortes anatómicos. El ungated
			# es 3D (sin gates): se envuelve con un eje de gate de tamaño 1. Es la
			# imagen que va al INFORME y la que se ve "espectacular" en la recon.
			ung_cube4 = self._thickened_sa_cube(ung_vol[None, ...], z0, z1, thickness)
			cuts_u = anatomical_cuts_gated(ung_cube4)
			self.cine_crudo_axes_for_export_ungated = {
				"SA": np.ascontiguousarray(cuts_u["sa"]),
				"HLA": np.ascontiguousarray(cuts_u["hla"]),
				"VLA": np.ascontiguousarray(cuts_u["vla"]),
			}
			# Cortes del MOTION-FROZEN (si el pipeline lo generó): mismo recorte y
			# mismos cortes anatómicos. Si hubo reorientación, usar el volumen MF
			# reorientado (misma transformación que el ungated); si no, el original.
			mf_vol = getattr(self, "cine_crudo_reoriented_mf", None)
			if mf_vol is None:
				mf_vol = getattr(self.cine_crudo_recon_result, "ungated_volume_mf", None) if getattr(self, "cine_crudo_recon_result", None) is not None else None
			if mf_vol is not None:
				mf_vol = np.asarray(mf_vol, dtype=np.float64)
				mf_cube4 = self._thickened_sa_cube(mf_vol[None, ...], z0, z1, thickness)
				cuts_mf = anatomical_cuts_gated(mf_cube4)
				self.cine_crudo_axes_for_export_mf = {
					"SA": np.ascontiguousarray(cuts_mf["sa"]),
					"HLA": np.ascontiguousarray(cuts_mf["hla"]),
					"VLA": np.ascontiguousarray(cuts_mf["vla"]),
				}
				self._log(f"Cortes Motion-frozen generados: SA {cuts_mf['sa'].shape}, HLA {cuts_mf['hla'].shape}, VLA {cuts_mf['vla'].shape}")
			else:
				self.cine_crudo_axes_for_export_mf = {}
				self._log("Motion-frozen: no hay volumen MF (¿checkbox activado al reconstruir?)")
			# Cortes del CT registrado reorientado (fusión en ejes cardiacos): mismo
			# recorte base→ápex y mismos cortes anatómicos que el visible.
			ct_reo = getattr(self, "cine_crudo_reoriented_ct", None)
			if ct_reo is not None:
				try:
					ct_reo = np.asarray(ct_reo, dtype=np.float64)
					ct_cube4 = self._thickened_sa_cube(ct_reo[None, ...], z0, z1, thickness)
					cuts_ct = anatomical_cuts_gated(ct_cube4)
					self.cine_crudo_axes_for_export_ct = {
						"SA": np.ascontiguousarray(cuts_ct["sa"]),
						"HLA": np.ascontiguousarray(cuts_ct["hla"]),
						"VLA": np.ascontiguousarray(cuts_ct["vla"]),
					}
					self._log(f"[FUSION] Cortes CT en ejes cardiacos generados: SA {cuts_ct['sa'].shape}.")
				except Exception as exc:
					self.cine_crudo_axes_for_export_ct = {}
					self._log(f"[FUSION][WARN] Cortes CT fallaron: {exc}")
			else:
				self.cine_crudo_axes_for_export_ct = {}
			if sa_cube.shape[1] < 2:
				QMessageBox.information(self, "SINCRO", "Los límites deben dejar al menos 2 cortes SA.")
				return False
			# Pasajero de fase (FBP): mismo recorte base→ápex y mismos cortes SA que
			# el visible. Se usa como base del análisis de FASE (ver paso 4). El
			# fallback al pasajero SIN reorientar solo vale si NO hubo reorientación
			# (si la hubo, cortar el crudo daría anatomía equivocada).
			sa_cube_phase = None
			reo_done = getattr(self, "cine_crudo_reoriented_gated", None) is not None
			phase_vol = getattr(self, "cine_crudo_reoriented_gated_phase", None)
			phase_src = "reorientado"
			if phase_vol is None:
				if reo_done:
					self._log("[FASE][WARN] Hay reorientación pero el pasajero FBP no fue reorientado: la fase caerá al volumen visible (filtro-dependiente).")
				elif getattr(self, "cine_crudo_recon_result_phase", None) is not None:
					phase_vol = getattr(self.cine_crudo_recon_result_phase, "gated_volume", None)
					phase_src = "recon (sin reorientar)"
				else:
					self._log("[FASE][WARN] Sin pasajero FBP disponible al generar cortes: la fase se calculará sobre el volumen visible (filtro-dependiente).")
			if phase_vol is not None:
				try:
					phase_vol = np.asarray(phase_vol, dtype=np.float64)
					if phase_vol.shape == gated_vol.shape:
						reo_cube_p = self._thickened_sa_cube(phase_vol, z0, z1, thickness)
						sa_cube_phase = np.ascontiguousarray(anatomical_cuts_gated(reo_cube_p)["sa"])
						self._log(f"[FASE] Cortes SA del pasajero FBP generados ({phase_src}): {sa_cube_phase.shape}.")
					else:
						self._log(f"[FASE][WARN] Pasajero FBP descartado por shape: pasajero {phase_vol.shape} vs visible {gated_vol.shape}. La fase caerá al volumen visible.")
				except Exception as exc:
					sa_cube_phase = None
					self._log(f"[FASE][WARN] Corte del pasajero FBP falló: {exc}. La fase caerá al volumen visible.")
			out_png = self._write_cine_crudo_cuts_qc(ung_vol, z0, z1)
			self.cine_crudo_preview_mode = "generated_cuts"
			# QC dual: cachear el QC por etapa y, si ambas ya generaron cortes,
			# mostrar ESFUERZO arriba y REPOSO abajo (la 2da no tapa a la 1ra).
			stage_now = str(getattr(self, "_cine_crudo_recon_stage", "stress") or "stress")
			qc_cache = getattr(self, "_cuts_qc_pix_by_stage", None)
			if not isinstance(qc_cache, dict):
				qc_cache = {}
				self._cuts_qc_pix_by_stage = qc_cache
			qc_cache[stage_now] = QPixmap(out_png)
			if "stress" in qc_cache and "rest" in qc_cache and not qc_cache["stress"].isNull() and not qc_cache["rest"].isNull():
				display_pix = self._stack_cine_crudo_dual_pixmaps(
					qc_cache["stress"], qc_cache["rest"],
					"ESFUERZO — cortes generados", "REPOSO — cortes generados",
					active_stage=stage_now,
				)
			else:
				display_pix = QPixmap(out_png)
			for tab_name in ("comparacion_ejes", "cine_crudo"):
				if tab_name in self.preview_labels:
					pix = QPixmap(display_pix)
					self.preview_pixmaps[tab_name] = pix
					self.preview_base_sizes[tab_name] = pix.size()
					self._apply_preview_zoom(tab_name)
			self._select_tab_by_title("cine_crudo")
			# Espesor de corte físico. En un SPECT reconstruido desde proyecciones el
			# volumen es ISOTRÓPICO: el espesor Z = pixel spacing en plano. Si el
			# estudio crudo no trae z_spacing/slice_thickness DICOM (lo normal en
			# proyecciones), NO hay que caer a 1.0 mm (placeholder que aplasta los
			# volúmenes ~6× y da EDV/ESV irreales). Se usa el pixel spacing como
			# fallback físico correcto. Validado contra Xeleris (EDV/ESV coherentes).
			raw_px = getattr(raw_study, "pixel_spacing", None)
			try:
				iso_px_mm = float(raw_px[0]) if raw_px else 0.0
			except Exception:
				iso_px_mm = 0.0
			src_z_mm = float(
				getattr(raw_study, "z_spacing_mm", None)
				or getattr(raw_study, "slice_thickness_mm", None)
				or (iso_px_mm if iso_px_mm > 0.0 else 1.0)
			)
			cut_thickness_mm = src_z_mm * float(thickness)
			self.cine_crudo_cut_study = dicom_loader.GatedStudy(
				cube=sa_cube,
				cube_phase=sa_cube_phase,
				n_gates=int(sa_cube.shape[0]),
				n_slices=int(sa_cube.shape[1]),
				rows=int(sa_cube.shape[2]),
				cols=int(sa_cube.shape[3]),
				pixel_spacing=getattr(raw_study, "pixel_spacing", None),
				source_path=str(getattr(raw_study, "source_path", "") or self.file_edit.text().strip()),
				z_spacing_mm=src_z_mm,
				slice_thickness_mm=cut_thickness_mm,
				spacing_between_slices_mm=src_z_mm,
				image_type=["DERIVED", "RECON", "GATED TOMO", "GAMMASYNC SA CUTS"],
				series_description=(str(getattr(raw_study, "series_description", "") or "RAW") + " | GammaSync SA cuts"),
				study_description=str(getattr(raw_study, "study_description", "") or ""),
				patient_name=str(getattr(raw_study, "patient_name", "") or ""),
				patient_id=str(getattr(raw_study, "patient_id", "") or ""),
				patient_sex=str(getattr(raw_study, "patient_sex", "") or ""),
				patient_birth_date=str(getattr(raw_study, "patient_birth_date", "") or ""),
				study_date=str(getattr(raw_study, "study_date", "") or ""),
				study_time=str(getattr(raw_study, "study_time", "") or ""),
				accession_number=str(getattr(raw_study, "accession_number", "") or ""),
				study_instance_uid=str(getattr(raw_study, "study_instance_uid", "") or ""),
				reconstructed=True,
				qc_first_harmonic=float(getattr(raw_study, "qc_first_harmonic", 0.0) or 0.0),
				qc_passed=bool(getattr(raw_study, "qc_passed", False)),
				gating_info=dict(getattr(raw_study, "gating_info", {}) or {}),
				notes=list(getattr(raw_study, "notes", []) or []) + list(result.notes) + [
					f"Recon raw fuente: {self.cine_crudo_cut_source_label}",
					f"SA cuts limits: base={z0 + 1}, apex={z1 + 1}, thickness_px={thickness}",
				],
			)
			self.cine_crudo_axes_for_export = {"SA": sa_cube, "HLA": hla_cube, "VLA": vla_cube}
			# Escala física: mm por corte SA (para montaje a escala real).
			px = getattr(raw_study, "pixel_spacing", None)
			try:
				px_mm = float(px[0]) if px is not None and len(px) else 0.0
			except Exception:
				px_mm = 0.0
			self.cine_crudo_axes_pixel_mm = px_mm
			self.cine_crudo_cut_thickness_mm = cut_thickness_mm
			self.cine_crudo_montage_bounds = (int(z0), int(z1))
			# Enrutar los cortes al slot de la etapa reconstruida (dual real, sin
			# necesidad del botón "Marcar como reposo"): esfuerzo→_stress, reposo→_rest.
			if getattr(self, "_cine_crudo_recon_stage", "stress") == "rest":
				self.cine_crudo_axes_for_export_rest = {k: np.array(v, copy=True) for k, v in self.cine_crudo_axes_for_export.items()}
				self.cine_crudo_axes_for_export_ungated_rest = {k: np.array(v, copy=True) for k, v in self.cine_crudo_axes_for_export_ungated.items()}
				self.cine_crudo_axes_for_export_mf_rest = {k: np.array(v, copy=True) for k, v in getattr(self, "cine_crudo_axes_for_export_mf", {}).items()}
				self.cine_crudo_rest_source_label = self.cine_crudo_cut_source_label or "raw recon"
				self.cine_crudo_cut_thickness_mm_rest = float(cut_thickness_mm or 0.0)
				self._log("Cortes de REPOSO guardados automáticamente para el montaje comparativo.")
			else:
				self.cine_crudo_axes_for_export_stress = {k: np.array(v, copy=True) for k, v in self.cine_crudo_axes_for_export.items()}
				self.cine_crudo_axes_for_export_ungated_stress = {k: np.array(v, copy=True) for k, v in self.cine_crudo_axes_for_export_ungated.items()}
				self.cine_crudo_axes_for_export_mf_stress = {k: np.array(v, copy=True) for k, v in getattr(self, "cine_crudo_axes_for_export_mf", {}).items()}
				self._log("Cortes de ESFUERZO guardados automáticamente para el montaje comparativo.")
			# Inicializar rango de gates para montaje (todo el ciclo por defecto).
			n_gates_out = int(sa_cube.shape[0])
			self.cine_crudo_gate_from = 1
			self.cine_crudo_gate_to = max(1, n_gates_out)
			if hasattr(self, "cine_crudo_gate_from_spin") and hasattr(self, "cine_crudo_gate_to_spin"):
				self.cine_crudo_gate_from_spin.blockSignals(True)
				self.cine_crudo_gate_to_spin.blockSignals(True)
				self.cine_crudo_gate_from_spin.setRange(1, max(1, n_gates_out))
				self.cine_crudo_gate_to_spin.setRange(1, max(1, n_gates_out))
				self.cine_crudo_gate_from_spin.setValue(1)
				self.cine_crudo_gate_to_spin.setValue(max(1, n_gates_out))
				self.cine_crudo_gate_from_spin.setEnabled(True)
				self.cine_crudo_gate_to_spin.setEnabled(True)
				self.cine_crudo_gate_from_spin.blockSignals(False)
				self.cine_crudo_gate_to_spin.blockSignals(False)
			if hasattr(self, "cine_crudo_gate_all_btn"):
				self.cine_crudo_gate_all_btn.setEnabled(True)
			if hasattr(self, "cine_crudo_process_recon_btn"):
				self.cine_crudo_process_recon_btn.setEnabled(True)
			if hasattr(self, "cine_crudo_copy_rois_to_rest_btn"):
				# Se habilita recién cuando ambas etapas tienen cortes SA en memoria.
				sess = self._dual_session()
				self.cine_crudo_copy_rois_to_rest_btn.setEnabled(
					sess.stage("stress").cut_study is not None
					and sess.stage("rest").cut_study is not None
				)
			if hasattr(self, "cine_crudo_save_axes_dcm_btn"):
				self.cine_crudo_save_axes_dcm_btn.setEnabled(True)
			if hasattr(self, "cine_crudo_montage_export_btn"):
				self.cine_crudo_montage_export_btn.setEnabled(True)
			if hasattr(self, "cine_crudo_mark_rest_btn"):
				self.cine_crudo_mark_rest_btn.setEnabled(True)
			self._log(f"Cortes generados: SA {z0 + 1}..{z1 + 1} ({sa_cube.shape[1]} cortes, espesor {thickness}px = {cut_thickness_mm:.2f}mm). HLA/VLA visibles en comparacion_ejes. Ahora podés Procesar recon o Guardar ejes DICOM.")
			self._set_progress(100, "Cortes SA/HLA/VLA generados")
			self._commit_undo("Generar cortes", self.UNDO_ATTRS_CUTS, _undo_before)
			self._mark_step_done("cuts", z0, z1, thickness)
			return True
		except Exception as exc:
			self._log(f"[ERROR] Generar cortes falló: {exc}")
			QMessageBox.warning(self, "SINCRO", f"No se pudieron generar los cortes:\n{exc}")
			return False

	def _montage_cmap_lut(self, name):
		"""LUT uint8 (256,3) del colormap (incluye los .col registrados)."""
		import matplotlib
		try:
			cmap = matplotlib.colormaps[str(name)]
		except Exception:
			try:
				cmap = matplotlib.colormaps["gray"]
			except Exception:
				cmap = None
		if cmap is None:
			g = np.linspace(0, 255, 256).astype(np.uint8)
			return np.stack([g, g, g], axis=1)
		rgba = cmap(np.linspace(0.0, 1.0, 256))
		return (np.clip(np.asarray(rgba)[:, :3], 0.0, 1.0) * 255.0).astype(np.uint8)

	def _composite_montage_pixmap(self, rows_data, cols, cmap_name, suptitle, ref_views=None):
		"""Compone el montaje en memoria y cachea el lienzo GRIS + máscara para que
		cambiar el colormap solo re-aplique el LUT (recoloreo casi instantáneo).
		Interpola en gris de 1 canal (más suave y ~3x más barato que en RGB).
		Cada corte llega ya normalizado/ventaneado a 0..1 con escala compartida.

		``ref_views`` (opcional): dict {"SA": arr2d, "VLA": arr2d, "HLA": arr2d}
		con las vistas MIP de referencia para la columna izquierda. Si es None o
		falta alguna, la columna queda vacía (comportamiento previo)."""
		from PIL import Image
		rows = max(1, len(rows_data))
		cols = max(1, int(cols))
		# Filtro visual del montaje (idéntico a la vista de cortes): interpolación de
		# resample + gaussiano extra. Solo display, no altera datos ni análisis.
		_pil_resample = {
			"Píxel": Image.NEAREST, "Bilineal": Image.BILINEAR, "Bicúbico": Image.BICUBIC,
			"Hanning": Image.HAMMING, "Lanczos": Image.LANCZOS,
		}
		resample = _pil_resample.get(str(getattr(self, "cine_crudo_montage_interp", "Bilineal")), Image.BILINEAR)
		montage_smooth = float(getattr(self, "cine_crudo_montage_smooth", 0.0) or 0.0)
		_gauss = None
		if montage_smooth > 0.0:
			try:
				from scipy.ndimage import gaussian_filter as _gauss
			except Exception:
				_gauss = None
		# Panel px: 512 en HQ (final/export), menor durante la interacción (fast-pass).
		PANEL = max(120, int(getattr(self, "_montage_panel_px", 512)))
		f = PANEL / 150.0
		PAD = max(2, round(4 * f))
		TITLE_H = round(16 * f)
		# Columna de referencia: ancho fijo a la izquierda (mismo alto que un panel).
		REF_W = round(76 * f)
		LEFT = round(76 * f) + REF_W  # espacio para rótulos de eje + columna ref
		TOP = round(42 * f)
		cell_w = PANEL + PAD
		cell_h = TITLE_H + PANEL + PAD
		W = LEFT + cols * cell_w + PAD
		H = TOP + rows * cell_h + PAD
		gray = np.zeros((H, W), dtype=np.uint8)
		mask = np.zeros((H, W), dtype=bool)

		# --- Columna de referencia: MIP por fila, con línea del corte actual ---
		if ref_views:
			for r, row in enumerate(rows_data):
				prefix = row["prefix"]
				if prefix not in ref_views:
					continue
				ref_img = np.asarray(ref_views[prefix], dtype=np.float64)
				if ref_img.ndim != 2:
					continue
				# Normalizar a 0..1 con la misma ventana que el montaje.
				p99 = float(np.percentile(ref_img, 99.0)) or 1.0
				p2 = float(np.percentile(ref_img, 2.0)) or 0.0
				ref_norm = np.clip((ref_img - p2) / max(p99 - p2, 1e-8), 0.0, 1.0)
				# Resize al tamaño del panel (cuadrado, letterbox si no es cuadrado).
				ih, iw = int(ref_norm.shape[0]), int(ref_norm.shape[1])
				scale = min(PANEL / max(1, iw), PANEL / max(1, ih))
				nw = max(1, int(round(iw * scale)))
				nh = max(1, int(round(ih * scale)))
				try:
					rimg = np.asarray(Image.fromarray(ref_norm, mode="F").resize((nw, nh), resample))
				except Exception:
					rimg = ref_norm[:nh, :nw]
				idx8 = np.clip(np.asarray(rimg) * 255.0, 0.0, 255.0).astype(np.uint8)
				y0 = TOP + r * cell_h + TITLE_H
				x0 = round(76 * f)  # inicio de la columna de referencia
				oy = y0 + (PANEL - nh) // 2
				ox = x0 + (REF_W - nw) // 2
				gray[oy:oy + nh, ox:ox + nw] = idx8[:nh, :nw]
				mask[oy:oy + nh, ox:ox + nw] = True
				# Guardar la geometría de la referencia para pintar las rayitas después.
				row["_ref_geom"] = (x0, y0, REF_W, PANEL, ox, oy, nw, nh)
				# Calcular las posiciones de las rayitas (bandoneón) en la referencia.
				# Cada corte de la fila tiene una rayita; la actual se destaca.
				idxs = row.get("idxs", [])
				if idxs and len(idxs) > 1:
					# Mapear cada índice de corte a una posición en la referencia.
					# Para SA: el eje de la fila es el eje 0 del volumen (transaxial),
					# la referencia es un MIP transaxial -> la posición es vertical (y).
					# Para VLA/HLA: el eje de la fila es el eje 1 o 2 (longitudinal),
					# la referencia es un MIP coronal/sagital -> la posición es horizontal (x).
					k_min, k_max = float(min(idxs)), float(max(idxs))
					rayitas = []
					for k in idxs:
						frac = (float(k) - k_min) / max(k_max - k_min, 1e-6)
						if row["prefix"] == "SA":
							# Posición vertical en la referencia (y).
							y_pos = oy + frac * nh
							rayitas.append((ox, int(y_pos), ox + nw, int(y_pos)))
						else:
							# Posición horizontal en la referencia (x).
							x_pos = ox + frac * nw
							rayitas.append((int(x_pos), oy, int(x_pos), oy + nh))
					row["_rayitas"] = rayitas

		panel_boxes = []
		for r, row in enumerate(rows_data):
			for p in row["panels"]:
				c = int(p["col"])
				if c >= cols:
					continue
				img = np.clip(np.asarray(p["img"], dtype=np.float32), 0.0, 1.0)
				if _gauss is not None:
					img = np.clip(_gauss(img, sigma=montage_smooth), 0.0, 1.0).astype(np.float32)
				ih, iw = int(img.shape[0]), int(img.shape[1])
				# Preservar relación de aspecto (evita VLA/HLA estirados); letterbox negro centrado.
				scale = min(PANEL / max(1, iw), PANEL / max(1, ih))
				nw = max(1, int(round(iw * scale)))
				nh = max(1, int(round(ih * scale)))
				try:
					# Interpolar en float (mode 'F') evita bandas de cuantización.
					rimg = np.asarray(Image.fromarray(img, mode="F").resize((nw, nh), resample))
				except Exception:
					rimg = img[:nh, :nw]
				idx8 = np.clip(np.asarray(rimg) * 255.0, 0.0, 255.0).astype(np.uint8)
				y0 = TOP + r * cell_h + TITLE_H
				x0 = LEFT + c * cell_w
				oy = y0 + (PANEL - nh) // 2
				ox = x0 + (PANEL - nw) // 2
				gray[oy:oy + nh, ox:ox + nw] = idx8[:nh, :nw]
				mask[oy:oy + nh, ox:ox + nw] = True
				panel_boxes.append((x0, y0, p.get("title", ""), p.get("corners")))

		rows_meta = [
			{
				"tag": row["tag"],
				"prefix": row["prefix"],
				"selection_key": row.get("selection_key", f"{row['tag'] or 'ESFUERZO'}:{row['prefix']}"),
				"selected": bool(row["selected"]),
				"used_cols": int(row["used_cols"]),
				"_ref_geom": row.get("_ref_geom"),
				"_rayitas": row.get("_rayitas"),
			}
			for row in rows_data
		]
		self._montage_gray_cache = {
			"gray": gray, "mask": mask, "panel_boxes": panel_boxes, "rows_meta": rows_meta,
			"geom": (PANEL, PAD, TITLE_H, LEFT, TOP, cell_w, cell_h, f, W, H, REF_W),
			"suptitle": str(suptitle),
		}
		return self._montage_pix_from_gray(cmap_name)

	def _montage_pix_from_gray(self, cmap_name):
		"""Aplica el LUT al lienzo gris cacheado y pinta rótulos. Sin pipeline ni
		resize: recolorear el montaje cuesta solo un lookup vectorizado."""
		cache = getattr(self, "_montage_gray_cache", None)
		if not cache:
			return None
		lut = self._montage_cmap_lut(cmap_name)
		gray = cache["gray"]
		mask = cache["mask"]
		H, W = gray.shape
		rgb = np.zeros((H, W, 3), dtype=np.uint8)
		rgb[mask] = lut[gray[mask]]
		buf = rgb.tobytes()
		qimg = QImage(buf, W, H, 3 * W, QImage.Format.Format_RGB888)
		pix = QPixmap.fromImage(qimg)
		self._paint_montage_overlays(pix, cache)
		return pix

	def _refresh_montage_selection_overlay(self) -> None:
		"""Redibuja SOLO los bordes de selección sobre el montaje cacheado.

		No vuelve a calcular cortes, normalización, LUT ni compositor. Actualiza
		la capa de overlay QPainter desde ``rows_meta`` y blitea el QPixmap al
		QLabel, por lo que el click se refleja inmediatamente.
		"""
		cache = getattr(self, "_montage_gray_cache", None)
		if not cache:
			return
		selected = set(getattr(self, "cine_crudo_selected_stripes", set()) or set())
		for row in cache.get("rows_meta", []):
			key = str(row.get("selection_key", f"{row.get('tag') or 'ESFUERZO'}:{row.get('prefix', 'SA')}"))
			row["selected"] = key in selected
		self._recolor_montage_from_cache(str(getattr(self, "cine_crudo_montage_cmap", "odyssey_cool")))

	def _paint_montage_overlays(self, pix, cache):
		"""Título, rótulos de eje rotados, recuadro de tira activa, títulos de panel
		y esquinas anatómicas sobre el pixmap ya coloreado. Si hay columna de
		referencia, dibuja la línea indicativa del corte actual."""
		from PyQt6.QtGui import QFont
		from PyQt6.QtCore import QRect
		geom = cache["geom"]
		if len(geom) == 11:
			PANEL, PAD, TITLE_H, LEFT, TOP, cell_w, cell_h, f, W, H, REF_W = geom
		else:
			PANEL, PAD, TITLE_H, LEFT, TOP, cell_w, cell_h, f, W, H = geom
			REF_W = 0
		panel_boxes = cache["panel_boxes"]
		rows_meta = cache["rows_meta"]
		painter = QPainter(pix)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
		painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
		align_left = int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
		align_center = int(Qt.AlignmentFlag.AlignCenter)
		align_hcenter = int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

		painter.setFont(QFont("Segoe UI", max(8, round(10 * f)), QFont.Weight.Bold))
		painter.setPen(QColor("#ffffff"))
		painter.drawText(QRect(LEFT, round(4 * f), W - LEFT - PAD, TOP - round(8 * f)), align_left, str(cache["suptitle"]))

		f_panel = QFont("Segoe UI", max(7, round(8 * f)), QFont.Weight.Bold)
		f_corner = QFont("Segoe UI", max(6, round(7 * f)), QFont.Weight.Bold)
		f_axis = QFont("Segoe UI", max(7, round(9 * f)), QFont.Weight.Bold)
		for r, row in enumerate(rows_meta):
			used = max(1, int(row["used_cols"]))
			row_top = TOP + r * cell_h
			painter.save()
			painter.setFont(f_axis)
			painter.setPen(QColor("#ff4040") if row["selected"] else QColor("#7cf29a"))
			painter.translate(round(18 * f), row_top + (TITLE_H + PANEL) / 2)
			painter.rotate(-90)
			painter.drawText(QRect(-PANEL // 2, round(-12 * f), PANEL, round(24 * f)), align_center, f"{row['tag']} {row['prefix']}".strip())
			painter.restore()
			if row["selected"]:
				painter.setPen(QPen(QColor("#ff4040"), max(1, round(f))))
				painter.drawRect(QRect(LEFT - 1, row_top + TITLE_H - 1, used * cell_w - PAD + 1, PANEL + 1))
			# Línea indicativa en la columna de referencia (si existe).
			ref_geom = row.get("_ref_geom")
			if ref_geom is not None and REF_W > 0:
				x0, y0, ref_w, ref_h, ox, oy, nw, nh = ref_geom
				# Rayitas del bandoneón: cada corte de la fila tiene su marca.
				rayitas = row.get("_rayitas") or []
				if rayitas:
					painter.setPen(QPen(QColor("#7cf29a"), max(1, round(0.7 * f))))
					for (rx0, ry0, rx1, ry1) in rayitas:
						painter.drawLine(rx0, ry0, rx1, ry1)
					# La rayita del corte actual (centro de la ventana visible) se destaca.
					mid = len(rayitas) // 2
					if 0 <= mid < len(rayitas):
						rx0, ry0, rx1, ry1 = rayitas[mid]
						painter.setPen(QPen(QColor("#ffb020"), max(1, round(1.8 * f))))
						painter.drawLine(rx0, ry0, rx1, ry1)
				else:
					# Fallback: una sola línea central si no hay rayitas.
					cx = ox + nw // 2
					cy = oy + nh // 2
					painter.setPen(QPen(QColor("#ffb020"), max(1, round(1.5 * f))))
					if row["prefix"] == "SA":
						painter.drawLine(ox, cy, ox + nw, cy)
					else:
						painter.drawLine(cx, oy, cx, oy + nh)
				# Marco sutil alrededor de la referencia.
				painter.setPen(QPen(QColor("#4a6a8a"), max(1, round(0.8 * f))))
				painter.drawRect(ox - 1, oy - 1, nw + 2, nh + 2)

		for (x0, y0, title, corners) in panel_boxes:
			painter.setFont(f_panel)
			painter.setPen(QColor("#ffffff"))
			painter.drawText(QRect(x0, y0 - TITLE_H, PANEL, TITLE_H), align_hcenter, str(title))
			if corners:
				painter.setFont(f_corner)
				painter.setPen(QColor("#9cdcff"))
				for (tx, ty, txt) in corners:
					painter.drawText(x0 + int(tx * PANEL), y0 + int((1.0 - ty) * PANEL), str(txt))
		painter.end()

	def _montage_display_base_size(self, pix):
		"""Tamaño base del preview normalizado al equivalente 512px: así el tamaño
		en pantalla es el mismo en fast-pass (256) y en HQ (512), sin salto de zoom."""
		panel = max(1, int(getattr(self, "_montage_panel_px", 512)))
		factor = 512.0 / panel
		return QSize(max(1, round(pix.width() * factor)), max(1, round(pix.height() * factor)))

	def _recolor_montage_from_cache(self, cmap_name):
		"""Recoloreo rápido (solo cambió el colormap): reusa el lienzo gris cacheado
		y actualiza el preview sin re-ejecutar el pipeline del montaje."""
		pix = self._montage_pix_from_gray(cmap_name)
		if pix is None:
			self._show_cine_crudo_sa_montage()
			return
		# Persistir el PNG solo en HQ: evita dejar un archivo en baja resolución en disco.
		if int(getattr(self, "_montage_panel_px", 512)) >= 512:
			try:
				pix.save(os.path.join(self.output_dir, "sa_montage.png"), "PNG")
			except Exception:
				pass
			# Generar GIF del montaje cine si hay frames cacheados.
			try:
				frames = getattr(self, "_montage_cine_frames", None) or []
				if len(frames) >= 2:
					from PIL import Image
					pil_frames = []
					for fpix in frames:
						img = fpix.toImage()
						buf = img.bits().asstring(img.sizeInBytes())
						pil_frames.append(Image.frombuffer("RGBA", (img.width(), img.height()), buf, "raw", "BGRA"))
					gif_path = os.path.join(self.output_dir, "sa_montage_cine.gif")
					pil_frames[0].save(
						gif_path, save_all=True, append_images=pil_frames[1:],
						duration=int(self.polar_cine_speed_spin.value()), loop=0,
					)
			except Exception:
				pass
		if "comparacion_ejes" in self.preview_labels:
			self.preview_pixmaps["comparacion_ejes"] = pix
			self.preview_base_sizes["comparacion_ejes"] = self._montage_display_base_size(pix)
			# Display inmediato en FastTransformation (ágil) y repaint nítido diferido.
			self._apply_preview_zoom("comparacion_ejes", fast=True)
			if hasattr(self, "_montage_recolor_smooth_timer"):
				self._montage_recolor_smooth_timer.start(120)

	def _montage_recolor_smooth_repaint(self):
		"""Repaint nítido (SmoothTransformation) del montaje tras el recoloreo fast."""
		if self.cine_crudo_preview_mode == "sa_montage" and "comparacion_ejes" in self.preview_labels:
			self._apply_preview_zoom("comparacion_ejes")

	def _montage_signature(self):
		"""Firma del estado que afecta el render del montaje. Si no cambia, no se
		vuelve a renderizar al reentrar a la pestaña Montaje clínico."""
		def _ids(d):
			try:
				return tuple(sorted((str(k), id(v)) for k, v in (d or {}).items()))
			except Exception:
				return ()
		return (
			_ids(self.cine_crudo_axes_for_export),
			_ids(getattr(self, "cine_crudo_axes_for_export_stress", None)),
			_ids(getattr(self, "cine_crudo_axes_for_export_rest", None)),
			int(getattr(self, "cine_crudo_gate_from", 1) or 1),
			int(getattr(self, "cine_crudo_gate_to", 1) or 1),
			str(getattr(self, "cine_crudo_montage_template", "denso")),
			float(getattr(self, "cine_crudo_montage_cut_zoom", 1.0) or 1.0),
			str(getattr(self, "cine_crudo_montage_cmap", "")),
			bool(getattr(self, "cine_crudo_montage_center_cuts", False)),
			str(getattr(self, "cine_crudo_montage_interp", "Bilineal")),
			float(getattr(self, "cine_crudo_montage_smooth", 0.0) or 0.0),
			str(getattr(self, "cine_crudo_montage_crop_mode", "limits")),
			str(getattr(self, "cine_crudo_montage_win_mode", "percentil")),
			float(getattr(self, "cine_crudo_montage_win_low", 2.0) or 0.0),
			float(getattr(self, "cine_crudo_montage_win_high", 99.5) or 0.0),
			float(getattr(self, "cine_crudo_montage_lin_low", 0.0) or 0.0),
			float(getattr(self, "cine_crudo_montage_lin_high", 1.0) or 0.0),
			tuple(sorted((str(k), int(v)) for k, v in (getattr(self, "cine_crudo_rest_offset", {}) or {}).items())),
			tuple(sorted(
				(str(stage), str(axis), int(start))
				for stage, values in (getattr(self, "cine_crudo_stripe_start_by_stage", {}) or {}).items()
				for axis, start in (values or {}).items()
			)),
			tuple(sorted(getattr(self, "cine_crudo_selected_stripes", set()) or set())),
			str(getattr(self, "cine_crudo_focused_stripe", "ESFUERZO:SA")),
			tuple(sorted((str(k), int(v)) for k, v in (getattr(self, "cine_crudo_stripe_count", {}) or {}).items())),
		)

	def _montage_axes_for_source(self):
		"""Resuelve (stress_axes, rest_axes) según la fuente del montaje.

		"ungated" (default): cortes del ungated (perfusión estática, con Denoise+;
		la imagen del informe). "gated": cortes del gated (cine en movimiento).
		"motion_frozen": cortes del volumen 4D alineado y promediado (si existe).
		Devuelve dicts SA/HLA/VLA o {} si no hay.
		"""
		src = str(getattr(self, "cine_crudo_montage_source", "ungated") or "ungated")
		if src == "gated":
			stress = self.cine_crudo_axes_for_export_stress or self.cine_crudo_axes_for_export
			rest = self.cine_crudo_axes_for_export_rest
		else:
			stress = (getattr(self, "cine_crudo_axes_for_export_ungated_stress", None)
					or getattr(self, "cine_crudo_axes_for_export_ungated", None) or {})
			rest = getattr(self, "cine_crudo_axes_for_export_ungated_rest", None) or {}
		return stress, rest

	def _update_montage_cine_controls(self):
		"""Habilita/deshabilita los controles de cine del montaje según la fuente.

		El cine tiene sentido solo con la fuente Gated (varios gates). Con Ungated
		o Motion-frozen (imágenes estáticas) los controles quedan deshabilitados.
		"""
		src = str(getattr(self, "cine_crudo_montage_source", "ungated"))
		has_gates = src == "gated"
		for wname in (
			"cine_crudo_montage_cine_play_btn",
			"cine_crudo_montage_cine_prev_btn",
			"cine_crudo_montage_cine_next_btn",
			"cine_crudo_montage_cine_speed_spin",
		):
			w = getattr(self, wname, None)
			if w is not None:
				try:
					w.setEnabled(bool(has_gates))
				except Exception:
					pass

	def _update_montage_cine_toggle_text(self):
		btn = getattr(self, "cine_crudo_montage_cine_play_btn", None)
		if btn is not None:
			btn.setText("⏸" if bool(getattr(self, "cine_crudo_montage_cine_playing", False)) else "▶")

	def _montage_cine_range(self) -> tuple[int, int]:
		"""Rango de gates (1-based, inclusivo) que recorre el cine del montaje."""
		g0 = int(getattr(self, "cine_crudo_gate_from", 1) or 1)
		g1 = int(getattr(self, "cine_crudo_gate_to", 1) or 1)
		lo, hi = (g0, g1) if g0 <= g1 else (g1, g0)
		return max(1, lo), max(1, hi)

	def _toggle_montage_cine(self):
		src = str(getattr(self, "cine_crudo_montage_source", "ungated"))
		if src != "gated":
			return
		if not (getattr(self, "cine_crudo_axes_for_export_stress", None) or getattr(self, "cine_crudo_axes_for_export", None)):
			return
		self.cine_crudo_montage_cine_playing = not bool(getattr(self, "cine_crudo_montage_cine_playing", False))
		if self.cine_crudo_montage_cine_playing:
			if not self._ensure_montage_cine_frames():
				self.cine_crudo_montage_cine_playing = False
				self._update_montage_cine_toggle_text()
				return
			ms = int(self.cine_crudo_montage_cine_speed_spin.value()) if hasattr(self, "cine_crudo_montage_cine_speed_spin") else 40
			self._montage_cine_timer.setInterval(max(40, ms))
			self._montage_cine_timer.start()
		else:
			self._montage_cine_timer.stop()
		self._update_montage_cine_toggle_text()

	def _montage_cine_signature(self):
		"""Firma del estado que define los frames del cine: si cambia, hay que re-pre-renderizar."""
		def _ids(d):
			try:
				return tuple(sorted((str(k), id(v)) for k, v in (d or {}).items()))
			except Exception:
				return ()
		return (
			_ids(self._montage_axes_for_source()[0]),
			_ids(self._montage_axes_for_source()[1]),
			int(getattr(self, "cine_crudo_gate_from", 1) or 1),
			int(getattr(self, "cine_crudo_gate_to", 1) or 1),
			str(getattr(self, "cine_crudo_montage_template", "denso")),
			float(getattr(self, "cine_crudo_montage_cut_zoom", 1.0) or 1.0),
			str(getattr(self, "cine_crudo_montage_cmap", "")),
			bool(getattr(self, "cine_crudo_montage_center_cuts", False)),
			str(getattr(self, "cine_crudo_montage_interp", "Bilineal")),
			float(getattr(self, "cine_crudo_montage_smooth", 0.0) or 0.0),
			str(getattr(self, "cine_crudo_montage_win_mode", "percentil")),
		)

	def _ensure_montage_cine_frames(self) -> bool:
		"""Pre-renderiza TODOS los gates del cine a QPixmap (una sola vez, en HQ).

		La lentitud original venía de recomponer todo el montaje en cada tick del
		timer. Acá se renderizan los N frames UNA vez (al Play o al primer step) y se
		cachean; después cada frame solo blit-ea el QPixmap ya compuesto (instantáneo).
		"""
		sig = self._montage_cine_signature()
		if getattr(self, "_montage_cine_frames", None) and getattr(self, "_montage_cine_frames_sig", None) == sig:
			return True
		lo, hi = self._montage_cine_range()
		span = max(1, hi - lo + 1)
		prev_playing = bool(getattr(self, "cine_crudo_montage_cine_playing", False))
		prev_frame = int(getattr(self, "cine_crudo_montage_cine_frame", 0))
		prev_panel = int(getattr(self, "_montage_panel_px", 512))
		self._montage_panel_px = 512  # HQ para los frames cacheados
		self.cine_crudo_montage_cine_playing = True  # para que el render tome gate único
		frames = []
		# Countdown de película vieja: overlay flotante 8→1 durante el preload.
		countdown_overlay = self._create_montage_cine_countdown_overlay(span)
		try:
			QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
			for i in range(span):
				self._update_montage_countdown_overlay(countdown_overlay, span - i)
				self.cine_crudo_montage_cine_frame = i
				self._montage_last_signature = None
				self._show_cine_crudo_sa_montage()
				pix = self.preview_pixmaps.get("comparacion_ejes")
				if pix is None:
					return False
				frames.append(pix)
				QApplication.processEvents()  # no congelar la UI durante el preload
		except Exception as exc:
			self._log(f"[WARN] Preload del cine del montaje falló: {exc}")
			return False
		finally:
			self._remove_montage_countdown_overlay(countdown_overlay)
			try:
				QApplication.restoreOverrideCursor()
			except Exception:
				pass
			self._montage_panel_px = prev_panel
			self.cine_crudo_montage_cine_playing = prev_playing
			self.cine_crudo_montage_cine_frame = prev_frame
		self._montage_cine_frames = frames
		self._montage_cine_frames_sig = sig
		self._log(f"Cine del montaje: {span} frames pre-renderizados en memoria.")
		return True

	def _create_montage_cine_countdown_overlay(self, span: int):
		"""Crea el overlay de countdown discreto (esquina superior izquierda, chico)."""
		from PyQt6.QtWidgets import QLabel
		from PyQt6.QtCore import Qt
		from PyQt6.QtGui import QFont

		label = self.preview_labels.get("comparacion_ejes")
		if label is None:
			return None
		# Overlay discreto: esquina superior izquierda, fondo casi transparente.
		overlay = QLabel(str(span), label)
		overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
		overlay.setStyleSheet(
			"color: #ffb020; background: rgba(0,0,0,60); padding: 2px 6px; border-radius: 3px;"
		)
		overlay.setFont(QFont("Arial", 24, QFont.Weight.Bold))
		overlay.adjustSize()
		overlay.move(6, 6)  # esquina superior izquierda con un pequeño margen
		overlay.show()
		overlay.raise_()
		return overlay

	def _update_montage_countdown_overlay(self, overlay, value: int):
		"""Actualiza el número del countdown."""
		if overlay is None:
			return
		overlay.setText(str(max(1, int(value))))
		overlay.adjustSize()

	def _remove_montage_countdown_overlay(self, overlay):
		"""Elimina el overlay de countdown."""
		if overlay is not None:
			overlay.hide()
			overlay.deleteLater()

	def _blit_montage_cine_frame(self):
		"""Muestra el frame actual del cine blit-eando el QPixmap cacheado (sin recomponer)."""
		frames = getattr(self, "_montage_cine_frames", None) or []
		if not frames:
			return
		idx = int(getattr(self, "cine_crudo_montage_cine_frame", 0)) % len(frames)
		self.cine_crudo_montage_cine_frame = idx
		pix = frames[idx]
		if "comparacion_ejes" in self.preview_labels:
			self.cine_crudo_preview_mode = "sa_montage"
			self.preview_pixmaps["comparacion_ejes"] = pix
			self.preview_base_sizes["comparacion_ejes"] = self._montage_display_base_size(pix)
			self._apply_preview_zoom("comparacion_ejes", fast=False)
			self._select_tab_by_title("comparacion_ejes")

	def _stop_montage_cine(self):
		"""Detiene el cine del montaje, libera el caché de frames y vuelve a la suma de gates."""
		try:
			self._montage_cine_timer.stop()
		except Exception:
			pass
		self.cine_crudo_montage_cine_playing = False
		self.cine_crudo_montage_cine_frame = 0
		self._montage_cine_frames = []
		self._montage_cine_frames_sig = None
		self._update_montage_cine_toggle_text()

	def _advance_montage_cine_frame(self):
		if not bool(getattr(self, "cine_crudo_montage_cine_playing", False)):
			self._montage_cine_timer.stop()
			return
		frames = getattr(self, "_montage_cine_frames", None) or []
		if not frames:
			self._montage_cine_timer.stop()
			return
		self.cine_crudo_montage_cine_frame = (int(getattr(self, "cine_crudo_montage_cine_frame", 0)) + 1) % len(frames)
		self._blit_montage_cine_frame()

	def _step_montage_cine(self, delta: int):
		"""Navega gate a gate (detiene el play para dar control manual)."""
		if str(getattr(self, "cine_crudo_montage_source", "ungated")) != "gated":
			return
		self._montage_cine_timer.stop()
		self.cine_crudo_montage_cine_playing = False
		self._update_montage_cine_toggle_text()
		if not self._ensure_montage_cine_frames():
			return
		frames = getattr(self, "_montage_cine_frames", None) or []
		if not frames:
			return
		cur = int(getattr(self, "cine_crudo_montage_cine_frame", 0))
		self.cine_crudo_montage_cine_frame = int((cur + int(delta)) % len(frames))
		self._blit_montage_cine_frame()

	def _on_montage_cine_speed_changed(self, value: int):
		self._montage_cine_timer.setInterval(max(40, int(value)))

	def _show_cine_crudo_sa_montage(self):

		"""Montaje clínico de TODOS los cortes SA/VLA/HLA (estilo MyoVation/Xeleris).

		Usa los cubos ya orientados anatómicamente por `anatomical_cuts_gated`
		(misma fuente que "cortes generados", así coincide la rotación). Filas:
		SA (ápex→base), VLA (sep→lat) y HLA (inf→ant), paneles grandes cuadrados.
		"""
		# Si el cine del montaje está activo y ya hay frames pre-renderizados, solo
		# blit-ear el frame actual: NO recomponer todo el montaje en cada tick.
		if (
			str(getattr(self, "cine_crudo_montage_source", "ungated")) == "gated"
			and (bool(getattr(self, "cine_crudo_montage_cine_playing", False)) or int(getattr(self, "cine_crudo_montage_cine_frame", 0)) > 0)
			and getattr(self, "_montage_cine_frames", None)
			and getattr(self, "_montage_cine_frames_sig", None) == self._montage_cine_signature()
		):
			self._blit_montage_cine_frame()
			return
		stress_axes, rest_axes = self._montage_axes_for_source()
		if not stress_axes:
			QMessageBox.information(self, "SINCRO", "Primero generá los cortes con 'Generar cortes'.")
			return
		try:
			import matplotlib.pyplot as plt

			# Rango de gates activo (1-based en UI).
			gate_from = int(getattr(self, "cine_crudo_gate_from", 1) or 1)
			gate_to = int(getattr(self, "cine_crudo_gate_to", 1) or 1)
			# Cine del montaje: solo con fuente Gated, y si hay play o se navegó a un
			# frame (step). Cuando está activo se muestra un solo gate en vez de la suma.
			_cine_gate = None
			if str(getattr(self, "cine_crudo_montage_source", "ungated")) == "gated":
				if bool(getattr(self, "cine_crudo_montage_cine_playing", False)) or int(getattr(self, "cine_crudo_montage_cine_frame", 0)) > 0:
					_cine_gate = int(getattr(self, "cine_crudo_montage_cine_frame", 0)) + 1
			template_mode = str(getattr(self, "cine_crudo_montage_template", "denso") or "denso")
			cut_zoom = float(getattr(self, "cine_crudo_montage_cut_zoom", 1.0) or 1.0)
			montage_cmap = str(getattr(self, "cine_crudo_montage_cmap", "odyssey_cool") or "odyssey_cool")
			layout_cfg = self.MONTAGE_LAYOUTS.get(template_mode, self.MONTAGE_LAYOUTS["denso"])
			per_strip = layout_cfg.get("per_strip")

			def _template_defaults(total: int) -> int:
				if per_strip is None:
					return max(1, int(total))
				return max(1, min(int(per_strip), int(total)))

			def _zoom_cut(img2d: np.ndarray, factor: float) -> np.ndarray:
				arr = np.asarray(img2d, dtype=np.float64)
				if arr.ndim != 2 or factor <= 1.001:
					return arr
				h, w = arr.shape
				ch = max(4, int(round(h / factor)))
				cw = max(4, int(round(w / factor)))
				y0 = max(0, (h - ch) // 2)
				x0 = max(0, (w - cw) // 2)
				crop = arr[y0:y0 + ch, x0:x0 + cw]
				if crop.size == 0:
					return arr
				try:
					from scipy.ndimage import zoom as _ndi_zoom
					up = _ndi_zoom(crop, (float(h) / max(1, ch), float(w) / max(1, cw)), order=3, mode="nearest", prefilter=True)
				except Exception:
					zy = max(1, int(np.ceil(h / max(1, ch))))
					zx = max(1, int(np.ceil(w / max(1, cw))))
					up = np.repeat(np.repeat(crop, zy, axis=0), zx, axis=1)
				if up.shape[0] < h or up.shape[1] < w:
					up = np.pad(up, ((0, max(0, h - up.shape[0])), (0, max(0, w - up.shape[1]))), mode="edge")
				return np.asarray(up[:h, :w], dtype=np.float64)

			center_cuts = bool(getattr(self, "cine_crudo_montage_center_cuts", False))

			def _center_cut(img2d: np.ndarray) -> np.ndarray:
				"""Desplaza el corte para que su centroide de intensidad quede en el
				centro del panel (roll circular, sin cambiar el tamaño)."""
				arr = np.asarray(img2d, dtype=np.float64)
				if arr.ndim != 2:
					return arr
				h, w = arr.shape
				tot = float(arr.sum())
				if tot <= 1e-8:
					return arr
				ys = np.arange(h, dtype=np.float64)
				xs = np.arange(w, dtype=np.float64)
				cy = float((arr.sum(axis=1) * ys).sum() / tot)
				cx = float((arr.sum(axis=0) * xs).sum() / tot)
				sy = int(round((h - 1) / 2.0 - cy))
				sx = int(round((w - 1) / 2.0 - cx))
				if sy == 0 and sx == 0:
					return arr
				return np.roll(np.roll(arr, sy, axis=0), sx, axis=1)

			def _norm_vol(cube4d):
				arr4 = np.asarray(cube4d, dtype=np.float64)
				if arr4.ndim != 4 or arr4.shape[0] <= 0:
					return np.zeros((1, 1, 1), dtype=np.float64)
				g0 = int(np.clip(min(gate_from, gate_to) - 1, 0, arr4.shape[0] - 1))
				g1 = int(np.clip(max(gate_from, gate_to) - 1, 0, arr4.shape[0] - 1))
				if _cine_gate is not None:
					# Cine activo: mostrar un solo gate (sin sumar), para ver el latido.
					# Cada gate tiene escala distinta -> invalidar la firma cacheada para
					# que el render NUNCA se saltee por firma igual entre frames.
					self._montage_last_signature = None
					gg = int(np.clip(int(_cine_gate) - 1, g0, g1))
					v = np.asarray(arr4[gg], dtype=np.float64)
				else:
					v = np.asarray(arr4[g0:g1 + 1], dtype=np.float64).sum(axis=0)
				mode = str(getattr(self, "cine_crudo_montage_win_mode", "percentil") or "percentil")
				if mode == "lineal":
					# Motor unificado: normaliza por min/máx del volumen (escala
					# compartida entre cortes) y aplica la ventana 0..200% del
					# RangeSlider, idéntica a render_array_rgb del cine/diálogo.
					vmin = float(v.min()) if v.size else 0.0
					vmax = float(v.max()) if v.size else 1.0
					norm = (v - vmin) / max(vmax - vmin, 1e-8) if vmax > vmin else np.zeros_like(v)
					w0 = max(0.0, float(getattr(self, "cine_crudo_montage_lin_low", 0.0)))
					w1 = max(0.0, min(2.0, float(getattr(self, "cine_crudo_montage_lin_high", 1.0))))
					if w1 <= w0:
						w1 = min(2.0, w0 + 0.01)
					return np.clip((norm - w0) / max(1e-8, (w1 - w0)), 0.0, 1.0)
				win_lo = float(getattr(self, "cine_crudo_montage_win_low", 2.0) or 0.0)
				win_hi = float(getattr(self, "cine_crudo_montage_win_high", 99.5) or 100.0)
				if win_hi <= win_lo:
					win_hi = min(100.0, win_lo + 1.0)
				p99 = float(np.percentile(v, win_hi)) if v.size else 1.0
				p2 = float(np.percentile(v, win_lo)) if v.size else 0.0
				return np.clip((v - p2) / max(p99 - p2, 1e-8), 0.0, 1.0)

			def _voi_bounds(nk):
				"""Rango base→ápex (k0,k1) según modo de recorte."""
				if self.cine_crudo_montage_crop_mode == "voi" and getattr(self, "cine_crudo_reoriented_voi", None):
					voi = self.cine_crudo_reoriented_voi
					cz = float(voi.get("cz", nk / 2.0))
					rz = float(voi.get("rz", nk / 2.0))
					k0 = int(np.clip(round(cz - rz), 0, nk - 1))
					k1 = int(np.clip(round(cz + rz), 0, nk - 1))
					if k1 <= k0:
						k1 = min(nk - 1, k0 + 1)
					return k0, k1
				return 0, nk - 1  # el cubo SA ya viene recortado a límites

			def _build_rows(ax_cubes, rest_offsets=None, stage_tag="ESFUERZO"):
				sa_cube = np.asarray(ax_cubes.get("SA", []), dtype=np.float64)
				sa_v = _norm_vol(sa_cube)
				vla_v = _norm_vol(np.asarray(ax_cubes.get("VLA", sa_cube)))
				hla_v = _norm_vol(np.asarray(ax_cubes.get("HLA", sa_cube)))
				# Rango base→ápex coherente para los tres ejes.
				k0, k1 = _voi_bounds(int(sa_v.shape[0]))
				# VLA/HLA recorren el eje largo → mapear el mismo % de rango.
				def _range_map(nsrc):
					if nsrc <= 1:
						return [0]
					lo = int(round(k0 / max(1, sa_v.shape[0] - 1) * (nsrc - 1)))
					hi = int(round(k1 / max(1, sa_v.shape[0] - 1) * (nsrc - 1)))
					lo, hi = sorted((lo, hi))
					return list(range(lo, hi + 1))
				sa_idx = list(range(k0, k1 + 1))[::-1]  # ápex primero
				vla_idx = _range_map(int(vla_v.shape[0]))
				hla_idx = _range_map(int(hla_v.shape[0]))
				off = rest_offsets or {"SA": 0, "VLA": 0, "HLA": 0}
				sa_idx = [int(np.clip(k + off["SA"], 0, sa_v.shape[0] - 1)) for k in sa_idx]
				vla_idx = [int(np.clip(k + off["VLA"], 0, vla_v.shape[0] - 1)) for k in vla_idx]
				hla_idx = [int(np.clip(k + off["HLA"], 0, hla_v.shape[0] - 1)) for k in hla_idx]

				def _window(axis_name: str, idxs: list[int]) -> list[int]:
					if not idxs:
						return idxs
					starts_by_stage = getattr(self, "cine_crudo_stripe_start_by_stage", {}) or {}
					stage_starts = starts_by_stage.setdefault(stage_tag, {"SA": 1, "VLA": 1, "HLA": 1})
					start_1 = int(stage_starts.get(axis_name, 1) or 1)
					count_cfg = int(getattr(self, "cine_crudo_stripe_count", {}).get(axis_name, 999) or 999)
					count = _template_defaults(len(idxs)) if count_cfg >= 999 else max(1, min(count_cfg, len(idxs)))
					start = int(np.clip(start_1 - 1, 0, max(0, len(idxs) - 1)))
					if start + count > len(idxs):
						start = max(0, len(idxs) - count)
					# Persistir start efectivo por etapa para drag/rueda independientes.
					stage_starts[axis_name] = start + 1
					return idxs[start:start + count]

				sa_idx = _window("SA", sa_idx)
				vla_idx = _window("VLA", vla_idx)
				hla_idx = _window("HLA", hla_idx)
				return [(sa_v, sa_idx, "SA"), (vla_v, vla_idx, "VLA"), (hla_v, hla_idx, "HLA")]

			stress_rows = _build_rows(stress_axes, stage_tag="ESFUERZO")
			has_rest = bool(rest_axes)
			rest_rows = _build_rows(rest_axes, self.cine_crudo_rest_offset, stage_tag="REPOSO") if has_rest else None

			thickness = self._cine_crudo_cut_thickness_px()
			th_mm = float(getattr(self, "cine_crudo_cut_thickness_mm", 0.0) or 0.0)
			px_mm = float(getattr(self, "cine_crudo_axes_pixel_mm", 0.0) or 0.0)
			bounds = getattr(self, "cine_crudo_montage_bounds", None)

			# Columnas = máximo de cortes en cualquier fila (ventana seleccionada por tira).
			all_rows = list(stress_rows) + (list(rest_rows) if rest_rows else [])
			cols = max((len(idxs) for _, idxs, _ in all_rows), default=1)
			cols = max(cols, 1)
			block_rows = 3
			n_blocks = 2 if has_rest else 1
			rows = block_rows * n_blocks

			corner_map = {
				"SA": [(0.32, 0.90, "ANT"), (0.02, 0.45, "SEP"), (0.80, 0.45, "LAT"), (0.32, 0.03, "INF")],
				"VLA": [(0.32, 0.90, "ANT"), (0.02, 0.45, "BASE"), (0.76, 0.45, "APEX"), (0.32, 0.03, "INF")],
				"HLA": [(0.32, 0.90, "APEX"), (0.02, 0.45, "SEP"), (0.80, 0.45, "LAT"), (0.32, 0.03, "BASE")],
			}

			# Orden de filas: con reposo, intercalado por eje (stress/rest juntos):
			# ESFUERZO SA · REPOSO SA · ESFUERZO VLA · REPOSO VLA · ESFUERZO HLA · REPOSO HLA.
			ordered_rows: list[tuple] = []
			if rest_rows:
				for s_row, r_row in zip(stress_rows, rest_rows):
					ordered_rows.append((s_row[0], s_row[1], s_row[2], "ESFUERZO"))
					ordered_rows.append((r_row[0], r_row[1], r_row[2], "REPOSO"))
			else:
				for s_row in stress_rows:
					ordered_rows.append((s_row[0], s_row[1], s_row[2], ""))

			# Cortes finales (0..1, ya ventaneados/zoom/centrados) para el compositor.
			selected_stripes = set(getattr(self, "cine_crudo_selected_stripes", set()) or set())
			# Compatibilidad: sesiones previas sin selección por etapa.
			if not selected_stripes:
				selected_stripes = {f"ESFUERZO:{getattr(self, 'cine_crudo_selected_stripe', 'SA')}"}
			rows_data = []
			for (vol, idxs, prefix, tag) in ordered_rows:
				panels = []
				for c, k in enumerate(idxs):
					img = vol[int(np.clip(k, 0, vol.shape[0] - 1))]
					img = _zoom_cut(img, cut_zoom)
					if center_cuts:
						img = _center_cut(img)
					panels.append({
						"col": c,
						"img": np.asarray(img, dtype=np.float64),
						"title": f"{prefix} {k + 1}",
						"corners": corner_map.get(prefix) if c == 0 else None,
					})
				rows_data.append({
					"prefix": prefix,
					"tag": tag,
					"selection_key": f"{tag or 'ESFUERZO'}:{prefix}",
					"selected": f"{tag or 'ESFUERZO'}:{prefix}" in selected_stripes,
					"used_cols": len(idxs),
					"panels": panels,
					"idxs": list(idxs),  # posiciones de los cortes para las rayitas del bandoneón
				})

			# --- Vistas de referencia para la columna izquierda ---
			# PENDIENTE (2026-08-14): la columna de referencia con rayitas tipo
			# bandoneón (Odyssey) queda COMENTADA porque el render del compositor
			# degrada la imagen (re-normaliza y reescala en el lienzo gris, dando un
			# efecto "QR" incluso copiando el corte del medio del montaje). La
			# solución correcta requiere refactor del compositor para aceptar
			# QPixmaps directamente en la columna (no arrays re-procesados).
			# Código conservado en git history (rama FBP_POCO_ORTODOXO, ~v1.46.0).
			ref_views = {}

			# Metadatos para drag en vivo de tiras por eje.
			self._montage_render_meta = {
				"rows_total": int(rows),
				"cols": int(cols),
				"has_rest": bool(has_rest),
				"template": template_mode,
			}

			th_txt = f"{thickness}px" + (f" = {th_mm:.2f}mm" if th_mm > 0 else "")
			scale_txt = f" · pixel {px_mm:.2f}mm" if px_mm > 0 else ""
			bnd_txt = f" · Base {bounds[0] + 1}→Ápex {bounds[1] + 1}" if bounds else ""
			crop_txt = "Elipse VOI" if self.cine_crudo_montage_crop_mode == "voi" else "Límites"
			if _cine_gate is not None:
				gate_txt = f" · gate {_cine_gate} (cine)"
			else:
				gate_txt = f" · gates {min(gate_from, gate_to)}→{max(gate_from, gate_to)}"
			tpl_txt = f" · layout {layout_cfg.get('label', template_mode)}"
			zoom_txt = f" · zoom corte x{cut_zoom:.2f}"
			n_sa_stress = len(stress_rows[0][1])
			suptitle = (
				f"{self._patient_banner_text(include_stage=False)} — Montaje clínico{bnd_txt} · Esp {th_txt}{scale_txt} · recorte {crop_txt}{gate_txt}{tpl_txt}{zoom_txt} · "
				f"SA {n_sa_stress} cortes" + (" · ESFUERZO/REPOSO" if has_rest else "")
			)

			# Render en memoria (numpy RGB + QPainter): sin matplotlib ni PNG en cada cambio.
			pix = self._composite_montage_pixmap(rows_data, int(cols), montage_cmap, suptitle, ref_views=ref_views)
			self.cine_crudo_preview_mode = "sa_montage"
			# Firma del estado ya renderizado: al reentrar no se re-renderiza si no cambió.
			self._montage_last_signature = self._montage_signature()
			# Escribir sa_montage.png solo en HQ (reload al cambiar de pestaña y "Guardar PNG").
			if int(getattr(self, "_montage_panel_px", 512)) >= 512:
				try:
					pix.save(os.path.join(self.output_dir, "sa_montage.png"), "PNG")
				except Exception:
					pass
			if "comparacion_ejes" in self.preview_labels:
				self.preview_pixmaps["comparacion_ejes"] = pix
				self.preview_base_sizes["comparacion_ejes"] = self._montage_display_base_size(pix)
				# En fast-pass (interacción) escala rápido; el settle HQ 512 reescala nítido.
				fast_display = int(getattr(self, "_montage_panel_px", 512)) < 512
				self._apply_preview_zoom("comparacion_ejes", fast=fast_display)
			self._select_tab_by_title("comparacion_ejes")
			self._log(
				f"Montaje generado: SA {n_sa_stress} cortes · Esp {th_txt} · recorte {crop_txt} · gates {min(gate_from, gate_to)}→{max(gate_from, gate_to)} · template {template_mode}"
				+ (" · doble fila ESFUERZO/REPOSO" if has_rest else "")
			)
		except Exception as exc:
			self._log(f"[ERROR] Montaje falló: {exc}")
			QMessageBox.warning(self, "SINCRO", f"No se pudo generar el montaje:\n{exc}")

	def _on_montage_crop_mode_changed(self, idx):
		self.cine_crudo_montage_crop_mode = "voi" if int(idx) == 1 else "limits"
		if self.cine_crudo_preview_mode == "sa_montage":
			self._schedule_montage_refresh(0)

	def _on_montage_template_changed(self, _idx):
		if hasattr(self, "cine_crudo_montage_template_combo"):
			key = self.cine_crudo_montage_template_combo.currentData()
			if key in self.MONTAGE_LAYOUTS:
				self.cine_crudo_montage_template = str(key)
		if self.cine_crudo_preview_mode == "sa_montage":
			self._schedule_montage_refresh(0)

	def _on_montage_source_changed(self, _idx):
		if hasattr(self, "cine_crudo_montage_source_combo"):
			self.cine_crudo_montage_source = str(self.cine_crudo_montage_source_combo.currentData() or "ungated")
		# El cine solo aplica a la fuente Gated: al cambiar, lo detenemos y
		# habilitamos/deshabilitamos sus controles.
		self._stop_montage_cine()
		self._update_montage_cine_controls()
		# La fuente cambia los cortes: hay que re-renderizar el montaje completo
		# (no es solo un recolor). Forzamos el render aunque la firma no cambie.
		if self.cine_crudo_preview_mode == "sa_montage":
			self._montage_last_signature = None
			self._schedule_montage_refresh(0)

	def _on_montage_cut_zoom_changed(self, value):
		self.cine_crudo_montage_cut_zoom = float(max(1.0, min(2.5, float(value))))
		if self.cine_crudo_preview_mode == "sa_montage":
			self._schedule_montage_refresh(0)

	def _on_montage_cmap_changed(self, name):
		self._set_montage_cmap(str(name))

	def _set_montage_cmap(self, name: str):
		"""Fija el colormap del montaje y sincroniza los dos combos (barra del
		montaje + columna de comparacion_ejes) y la tira del LUT."""
		name = str(name)
		self.cine_crudo_montage_cmap = name
		for combo_name in ("cine_crudo_montage_cmap_combo", "compare_axes_color_cmap_combo"):
			combo = getattr(self, combo_name, None)
			if combo is not None and combo.currentText() != name:
				combo.blockSignals(True)
				idx = combo.findText(name)
				if idx >= 0:
					combo.setCurrentIndex(idx)
				combo.blockSignals(False)
		strip = getattr(self, "compare_axes_color_strip", None)
		if strip is not None:
			strip.set_cmap(name)
		if self.cine_crudo_preview_mode == "sa_montage":
			# Solo cambió el color: recolorear el lienzo gris cacheado (sin pipeline).
			if getattr(self, "_montage_gray_cache", None):
				self._recolor_montage_from_cache(name)
			else:
				self._schedule_montage_refresh(0)

	def _build_compare_axes_color_column(self) -> QWidget:
		"""Columna de controles de color/ventaneo pegada al panel de comparacion_ejes.

		cmap + toggle Percentil/Lineal + RangeSlider 200% (con botones Top/Base) +
		tira vertical del LUT. En modo Lineal, el RangeSlider maneja la ventana con
		el mismo motor 0..200% del cine/diálogo (norm por volumen = escala compartida)."""
		col = QWidget()
		col.setMaximumWidth(132)
		v = QVBoxLayout(col)
		v.setContentsMargins(4, 4, 4, 4)
		v.setSpacing(4)

		v.addWidget(QLabel("Escala"))
		self.compare_axes_color_cmap_combo = QComboBox()
		self.compare_axes_color_cmap_combo.addItems(self._all_cmaps)
		self.compare_axes_color_cmap_combo.setCurrentText(self.cine_crudo_montage_cmap)
		self.compare_axes_color_cmap_combo.setToolTip("Escala de colores del montaje (compartida con la barra del montaje).")
		self.compare_axes_color_cmap_combo.currentTextChanged.connect(self._on_montage_cmap_changed)
		v.addWidget(self.compare_axes_color_cmap_combo)

		self.compare_axes_win_mode_combo = QComboBox()
		self.compare_axes_win_mode_combo.addItem("Percentil", "percentil")
		self.compare_axes_win_mode_combo.addItem("Lineal", "lineal")
		_mi = self.compare_axes_win_mode_combo.findData(self.cine_crudo_montage_win_mode)
		if _mi >= 0:
			self.compare_axes_win_mode_combo.setCurrentIndex(_mi)
		self.compare_axes_win_mode_combo.setToolTip(
			"Percentil: ventana por percentiles (histórico, spinboxes del montaje).\n"
			"Lineal: normaliza por min/máx y aplica el RangeSlider 0–200% (motor unificado)."
		)
		self.compare_axes_win_mode_combo.currentIndexChanged.connect(self._on_compare_axes_win_mode_changed)
		v.addWidget(self.compare_axes_win_mode_combo)

		_btn_css = (
			"QPushButton{font-weight:bold;font-size:8pt;border:1px solid #94a3b8;"
			"border-radius:3px;padding:1px 3px;color:#1e293b;background:#e2e8f0;}"
			"QPushButton:hover{background:#cbd5e1;color:#2563eb;}"
		)
		slider_col = QVBoxLayout()
		slider_col.setContentsMargins(0, 0, 0, 0)
		slider_col.setSpacing(2)
		top_row = QHBoxLayout()
		top_row.setContentsMargins(0, 0, 0, 0)
		self.compare_axes_btn_top = QPushButton("Top")
		self.compare_axes_btn_top.setStyleSheet(_btn_css)
		self.compare_axes_btn_top.setCursor(Qt.CursorShape.PointingHandCursor)
		self.compare_axes_btn_top.setToolTip("Volver Top a 100%")
		self.compare_axes_btn_top.clicked.connect(self._reset_compare_axes_window_high)
		self.compare_axes_lbl_top = QLabel("100%")
		top_row.addStretch(1)
		top_row.addWidget(self.compare_axes_btn_top)
		top_row.addWidget(self.compare_axes_lbl_top)
		top_row.addStretch(1)
		slider_col.addLayout(top_row)

		self.compare_axes_range_slider = RangeSlider()
		self.compare_axes_range_slider.valuesChanged.connect(self._on_compare_axes_window_changed)

		base_row = QHBoxLayout()
		base_row.setContentsMargins(0, 0, 0, 0)
		self.compare_axes_btn_base = QPushButton("Base")
		self.compare_axes_btn_base.setStyleSheet(_btn_css)
		self.compare_axes_btn_base.setCursor(Qt.CursorShape.PointingHandCursor)
		self.compare_axes_btn_base.setToolTip("Volver Base a 0%")
		self.compare_axes_btn_base.clicked.connect(self._reset_compare_axes_window_low)
		self.compare_axes_lbl_base = QLabel("0%")
		base_row.addStretch(1)
		base_row.addWidget(self.compare_axes_btn_base)
		base_row.addWidget(self.compare_axes_lbl_base)
		base_row.addStretch(1)

		slider_row = QHBoxLayout()
		slider_row.setContentsMargins(0, 0, 0, 0)
		slider_col.addWidget(self.compare_axes_range_slider, 1)
		slider_col.addLayout(base_row)
		slider_row.addLayout(slider_col, 1)
		self.compare_axes_color_strip = VerticalColorStrip(self.cine_crudo_montage_cmap)
		slider_row.addWidget(self.compare_axes_color_strip)
		v.addLayout(slider_row, 1)

		self._compare_axes_slider_from_state()
		return col

	def _on_compare_axes_win_mode_changed(self, _idx):
		combo = getattr(self, "compare_axes_win_mode_combo", None)
		if combo is not None:
			self.cine_crudo_montage_win_mode = str(combo.currentData() or "percentil")
		# Reflejar en el slider los valores guardados del modo recién activado.
		self._compare_axes_slider_from_state()
		if self.cine_crudo_preview_mode == "sa_montage":
			self._schedule_montage_refresh(0)

	def _compare_axes_slider_from_state(self):
		"""Vuelca al RangeSlider los valores guardados del modo activo (sin señales) y
		actualiza etiquetas/tooltip. Slider 0..200; en Percentil el valor es pct×2
		(0..100), en Lineal es fracción directa (0..200%)."""
		slider = getattr(self, "compare_axes_range_slider", None)
		if slider is None:
			return
		mode = str(getattr(self, "cine_crudo_montage_win_mode", "percentil"))
		if mode == "lineal":
			lo = int(round(float(self.cine_crudo_montage_lin_low) * 100))
			hi = int(round(float(self.cine_crudo_montage_lin_high) * 100))
			tip = "Ventana lineal 0–200% (Base abajo, Top arriba; >100% desatura)."
		else:
			lo = int(round(float(self.cine_crudo_montage_win_low) * 2))
			hi = int(round(float(self.cine_crudo_montage_win_high) * 2))
			tip = "Ventana por percentiles 0–100% (Base = fondo, Top = saturación)."
		slider.blockSignals(True)
		slider.set_values(lo, hi)
		slider.blockSignals(False)
		slider.setToolTip(tip)
		self._update_compare_axes_window_labels(*slider.values())

	def _update_compare_axes_window_labels(self, low, high):
		mode = str(getattr(self, "cine_crudo_montage_win_mode", "percentil"))
		if mode == "lineal":
			top_txt, base_txt = f"{int(high)}%", f"{int(low)}%"
		else:
			top_txt, base_txt = f"{int(high) / 2:.1f}%", f"{int(low) / 2:.1f}%"
		if getattr(self, "compare_axes_lbl_top", None) is not None:
			self.compare_axes_lbl_top.setText(top_txt)
		if getattr(self, "compare_axes_lbl_base", None) is not None:
			self.compare_axes_lbl_base.setText(base_txt)

	def _on_compare_axes_window_changed(self, low, high):
		mode = str(getattr(self, "cine_crudo_montage_win_mode", "percentil"))
		if mode == "lineal":
			self.cine_crudo_montage_lin_low = int(low) / 100.0
			self.cine_crudo_montage_lin_high = int(high) / 100.0
		else:
			self.cine_crudo_montage_win_low = int(low) / 2.0
			self.cine_crudo_montage_win_high = int(high) / 2.0
		self._update_compare_axes_window_labels(low, high)
		if self.cine_crudo_preview_mode == "sa_montage":
			# Fast-pass: rinde baja-res al arrastrar y HQ al soltar (sin lag de debounce).
			self._schedule_montage_refresh(0, fast=True)

	def _reset_compare_axes_window_high(self):
		slider = getattr(self, "compare_axes_range_slider", None)
		if slider is not None:
			low, _ = slider.values()
			top = 100 if str(getattr(self, "cine_crudo_montage_win_mode", "percentil")) == "lineal" else 200
			slider.set_values(low, top)

	def _reset_compare_axes_window_low(self):
		slider = getattr(self, "compare_axes_range_slider", None)
		if slider is not None:
			_, high = slider.values()
			slider.set_values(0, high)

	def _build_cine_crudo_color_column(self) -> QWidget:
		"""Columna de color/ventaneo pegada al preview del cine crudo (proyecciones).

		cmap + RangeSlider 0..200% de p99 (con botones Top/Base) + tira vertical del
		LUT. Solo afecta la visualización en pantalla del cine de proyecciones; no
		toca el informe ni el montaje clínico."""
		col = QWidget()
		col.setMaximumWidth(132)
		v = QVBoxLayout(col)
		v.setContentsMargins(4, 4, 4, 4)
		v.setSpacing(4)

		v.addWidget(QLabel("Escala"))
		self.cine_crudo_screen_cmap_combo = QComboBox()
		self.cine_crudo_screen_cmap_combo.addItems(self._all_cmaps)
		self.cine_crudo_screen_cmap_combo.setCurrentText(self.cine_crudo_screen_cmap)
		# Sincronizar el estado con lo que realmente quedó seleccionado (por si "gray"
		# no estuviera en la lista, evitar desincronización combo/estado).
		self.cine_crudo_screen_cmap = str(self.cine_crudo_screen_cmap_combo.currentText())
		self.cine_crudo_screen_cmap_combo.setToolTip("Escala de colores del cine de proyecciones (solo pantalla).")
		self.cine_crudo_screen_cmap_combo.currentTextChanged.connect(self._on_cine_crudo_screen_cmap_changed)
		v.addWidget(self.cine_crudo_screen_cmap_combo)

		_btn_css = (
			"QPushButton{font-weight:bold;font-size:8pt;border:1px solid #94a3b8;"
			"border-radius:3px;padding:1px 3px;color:#1e293b;background:#e2e8f0;}"
			"QPushButton:hover{background:#cbd5e1;color:#2563eb;}"
		)
		slider_col = QVBoxLayout()
		slider_col.setContentsMargins(0, 0, 0, 0)
		slider_col.setSpacing(2)
		top_row = QHBoxLayout()
		top_row.setContentsMargins(0, 0, 0, 0)
		self.cine_crudo_screen_btn_top = QPushButton("Top")
		self.cine_crudo_screen_btn_top.setStyleSheet(_btn_css)
		self.cine_crudo_screen_btn_top.setCursor(Qt.CursorShape.PointingHandCursor)
		self.cine_crudo_screen_btn_top.setToolTip("Volver Top a 100%")
		self.cine_crudo_screen_btn_top.clicked.connect(self._reset_cine_crudo_screen_window_high)
		self.cine_crudo_screen_lbl_top = QLabel("100%")
		top_row.addStretch(1)
		top_row.addWidget(self.cine_crudo_screen_btn_top)
		top_row.addWidget(self.cine_crudo_screen_lbl_top)
		top_row.addStretch(1)
		slider_col.addLayout(top_row)

		self.cine_crudo_screen_range_slider = RangeSlider()
		self.cine_crudo_screen_range_slider.setToolTip("Ventana 0–200% de p99 (Base abajo, Top arriba; >100% oscurece).")
		self.cine_crudo_screen_range_slider.valuesChanged.connect(self._on_cine_crudo_screen_window_changed)

		base_row = QHBoxLayout()
		base_row.setContentsMargins(0, 0, 0, 0)
		self.cine_crudo_screen_btn_base = QPushButton("Base")
		self.cine_crudo_screen_btn_base.setStyleSheet(_btn_css)
		self.cine_crudo_screen_btn_base.setCursor(Qt.CursorShape.PointingHandCursor)
		self.cine_crudo_screen_btn_base.setToolTip("Volver Base a 0%")
		self.cine_crudo_screen_btn_base.clicked.connect(self._reset_cine_crudo_screen_window_low)
		self.cine_crudo_screen_lbl_base = QLabel("0%")
		base_row.addStretch(1)
		base_row.addWidget(self.cine_crudo_screen_btn_base)
		base_row.addWidget(self.cine_crudo_screen_lbl_base)
		base_row.addStretch(1)

		slider_row = QHBoxLayout()
		slider_row.setContentsMargins(0, 0, 0, 0)
		slider_col.addWidget(self.cine_crudo_screen_range_slider, 1)
		slider_col.addLayout(base_row)
		slider_row.addLayout(slider_col, 1)
		self.cine_crudo_screen_color_strip = VerticalColorStrip(self.cine_crudo_screen_cmap)
		slider_row.addWidget(self.cine_crudo_screen_color_strip)
		v.addLayout(slider_row, 1)

		self._cine_crudo_screen_slider_from_state()
		return col

	def _cine_crudo_screen_slider_from_state(self):
		slider = getattr(self, "cine_crudo_screen_range_slider", None)
		if slider is None:
			return
		lo = int(round(float(self.cine_crudo_screen_win_low)))
		hi = int(round(float(self.cine_crudo_screen_win_high)))
		slider.blockSignals(True)
		slider.set_values(lo, hi)
		slider.blockSignals(False)
		self._update_cine_crudo_screen_labels(*slider.values())

	def _update_cine_crudo_screen_labels(self, low, high):
		if getattr(self, "cine_crudo_screen_lbl_top", None) is not None:
			self.cine_crudo_screen_lbl_top.setText(f"{int(high)}%")
		if getattr(self, "cine_crudo_screen_lbl_base", None) is not None:
			self.cine_crudo_screen_lbl_base.setText(f"{int(low)}%")

	def _refresh_cine_crudo_projection_colors(self):
		"""Re-colorea SOLO el cine de proyecciones (preview crudo) tras cambiar
		colormap/ventana; no toca montaje ni cortes generados."""
		if self.study is None or bool(getattr(self.study, "reconstructed", True)):
			return
		if self.cine_crudo_preview_mode is not None:
			return
		source = str(self.cine_crudo_source_combo.currentText()) if hasattr(self, "cine_crudo_source_combo") else "UngGat"
		self._load_cine_crudo_frames(source)

	def _on_cine_crudo_screen_cmap_changed(self, name):
		self.cine_crudo_screen_cmap = str(name)
		strip = getattr(self, "cine_crudo_screen_color_strip", None)
		if strip is not None:
			strip.set_cmap(self.cine_crudo_screen_cmap)
		self._refresh_cine_crudo_projection_colors()
		self._refresh_cine_crudo_cuts_color()
		self._refresh_cine_crudo_cut_limits_color()

	def _on_cine_crudo_screen_window_changed(self, low, high):
		self.cine_crudo_screen_win_low = float(low)
		self.cine_crudo_screen_win_high = float(high)
		self._update_cine_crudo_screen_labels(low, high)
		self._refresh_cine_crudo_projection_colors()
		self._refresh_cine_crudo_cuts_color()
		self._refresh_cine_crudo_cut_limits_color()

	def _reset_cine_crudo_screen_window_high(self):
		slider = getattr(self, "cine_crudo_screen_range_slider", None)
		if slider is not None:
			low, _ = slider.values()
			slider.set_values(low, 100)

	def _reset_cine_crudo_screen_window_low(self):
		slider = getattr(self, "cine_crudo_screen_range_slider", None)
		if slider is not None:
			_, high = slider.values()
			slider.set_values(0, high)

	def _build_montage_toolbar_into(self, toolbar):
		"""Controles de acción del montaje clínico, centralizados en la pestaña
		Montaje clínico. El montaje se renderiza al entrar a la pestaña; el resto
		de acciones (layout, zoom, gates) ocurre en vivo."""
		toolbar.addWidget(QLabel("Fuente"))
		self.cine_crudo_montage_source_combo = QComboBox()
		self.cine_crudo_montage_source_combo.addItem("Ungated (perfusión)", "ungated")
		self.cine_crudo_montage_source_combo.addItem("Gated (cine)", "gated")
		# NOTA: la fuente "Motion-frozen" fue retirada del combo (2026-08-13). El
		# código del pipeline queda disponible (ver core/motion_frozen.py).
		_cur_src = self.cine_crudo_montage_source_combo.findData(getattr(self, "cine_crudo_montage_source", "ungated"))
		if _cur_src >= 0:
			self.cine_crudo_montage_source_combo.setCurrentIndex(_cur_src)
		self.cine_crudo_montage_source_combo.setToolTip(
			"Fuente de los cortes del montaje. Ungated = perfusión estática (con "
			"Denoise+ si está activo; la imagen del informe). Gated = cine en "
			"movimiento (suma de gates).")
		self.cine_crudo_montage_source_combo.currentIndexChanged.connect(self._on_montage_source_changed)
		toolbar.addWidget(self.cine_crudo_montage_source_combo)

		toolbar.addWidget(QLabel("Template"))
		self.cine_crudo_montage_template_combo = QComboBox()
		for _lay_key, _lay_cfg in self.MONTAGE_LAYOUTS.items():
			self.cine_crudo_montage_template_combo.addItem(_lay_cfg["label"], _lay_key)
		_cur_lay = self.cine_crudo_montage_template_combo.findData(self.cine_crudo_montage_template)
		if _cur_lay >= 0:
			self.cine_crudo_montage_template_combo.setCurrentIndex(_cur_lay)
		self.cine_crudo_montage_template_combo.setToolTip("Layout de presentación del montaje: cortes por tira (SA/VLA/HLA). Denso = todos.")
		self.cine_crudo_montage_template_combo.currentIndexChanged.connect(self._on_montage_template_changed)
		toolbar.addWidget(self.cine_crudo_montage_template_combo)

		toolbar.addWidget(QLabel("Zoom corte"))
		self.cine_crudo_cut_zoom_spin = QDoubleSpinBox()
		self.cine_crudo_cut_zoom_spin.setRange(1.00, 2.50)
		self.cine_crudo_cut_zoom_spin.setSingleStep(0.05)
		self.cine_crudo_cut_zoom_spin.setDecimals(2)
		self.cine_crudo_cut_zoom_spin.setValue(1.00)
		self.cine_crudo_cut_zoom_spin.setMaximumWidth(70)
		self.cine_crudo_cut_zoom_spin.setToolTip("Agrandar interno de cada corte (mismo factor para todos los paneles). 1.00 = original.")
		self.cine_crudo_cut_zoom_spin.valueChanged.connect(self._on_montage_cut_zoom_changed)
		toolbar.addWidget(self.cine_crudo_cut_zoom_spin)

		toolbar.addWidget(QLabel("Frames"))
		self.cine_crudo_gate_from_spin = QSpinBox()
		self.cine_crudo_gate_from_spin.setRange(1, 1)
		self.cine_crudo_gate_from_spin.setValue(1)
		self.cine_crudo_gate_from_spin.setMaximumWidth(56)
		self.cine_crudo_gate_from_spin.setEnabled(False)
		self.cine_crudo_gate_from_spin.setToolTip("Gate inicial (1-based) para el montaje. Se aplica en vivo. También podés arrastrar/cambiar con rueda en el panel.")
		self.cine_crudo_gate_from_spin.valueChanged.connect(self._on_montage_gate_range_changed)
		toolbar.addWidget(self.cine_crudo_gate_from_spin)
		toolbar.addWidget(QLabel("→"))
		self.cine_crudo_gate_to_spin = QSpinBox()
		self.cine_crudo_gate_to_spin.setRange(1, 1)
		self.cine_crudo_gate_to_spin.setValue(1)
		self.cine_crudo_gate_to_spin.setMaximumWidth(56)
		self.cine_crudo_gate_to_spin.setEnabled(False)
		self.cine_crudo_gate_to_spin.setToolTip("Gate final (1-based) para el montaje. Se aplica en vivo. También podés arrastrar/cambiar con rueda en el panel.")
		self.cine_crudo_gate_to_spin.valueChanged.connect(self._on_montage_gate_range_changed)
		toolbar.addWidget(self.cine_crudo_gate_to_spin)
		self.cine_crudo_gate_all_btn = QToolButton()
		self.cine_crudo_gate_all_btn.setText("Todo")
		self.cine_crudo_gate_all_btn.setEnabled(False)
		self.cine_crudo_gate_all_btn.setToolTip("Usa todos los gates para el montaje (click rápido).")
		self.cine_crudo_gate_all_btn.clicked.connect(self._set_montage_gate_full_range)
		toolbar.addWidget(self.cine_crudo_gate_all_btn)

		# --- Cine del montaje (solo fuente Gated): play/step/velocidad ---
		_is_gated_src = str(getattr(self, "cine_crudo_montage_source", "ungated")) == "gated"
		self.cine_crudo_montage_cine_play_btn = QToolButton()
		self.cine_crudo_montage_cine_play_btn.setText("▶")
		self.cine_crudo_montage_cine_play_btn.setEnabled(_is_gated_src)
		self.cine_crudo_montage_cine_play_btn.setToolTip(
			"Cine del montaje (solo fuente Gated): recorre los gates en vivo para "
			"ver el latido. Velocidad configurable (default 40 ms).")
		self.cine_crudo_montage_cine_play_btn.clicked.connect(self._toggle_montage_cine)
		toolbar.addWidget(self.cine_crudo_montage_cine_play_btn)
		self.cine_crudo_montage_cine_prev_btn = QToolButton()
		self.cine_crudo_montage_cine_prev_btn.setText("|<")
		self.cine_crudo_montage_cine_prev_btn.setEnabled(_is_gated_src)
		self.cine_crudo_montage_cine_prev_btn.setToolTip("Gate anterior (montaje cine)")
		self.cine_crudo_montage_cine_prev_btn.clicked.connect(lambda _=False: self._step_montage_cine(-1))
		toolbar.addWidget(self.cine_crudo_montage_cine_prev_btn)
		self.cine_crudo_montage_cine_next_btn = QToolButton()
		self.cine_crudo_montage_cine_next_btn.setText(">|")
		self.cine_crudo_montage_cine_next_btn.setEnabled(_is_gated_src)
		self.cine_crudo_montage_cine_next_btn.setToolTip("Gate siguiente (montaje cine)")
		self.cine_crudo_montage_cine_next_btn.clicked.connect(lambda _=False: self._step_montage_cine(1))
		toolbar.addWidget(self.cine_crudo_montage_cine_next_btn)
		toolbar.addWidget(QLabel("Vel."))
		self.cine_crudo_montage_cine_speed_spin = QSpinBox()
		self.cine_crudo_montage_cine_speed_spin.setRange(40, 500)
		self.cine_crudo_montage_cine_speed_spin.setSingleStep(10)
		self.cine_crudo_montage_cine_speed_spin.setValue(40)
		self.cine_crudo_montage_cine_speed_spin.setSuffix(" ms")
		self.cine_crudo_montage_cine_speed_spin.setEnabled(_is_gated_src)
		self.cine_crudo_montage_cine_speed_spin.setMaximumWidth(74)
		self.cine_crudo_montage_cine_speed_spin.setToolTip("Intervalo entre gates del cine del montaje (default 40 ms).")
		self.cine_crudo_montage_cine_speed_spin.valueChanged.connect(self._on_montage_cine_speed_changed)
		toolbar.addWidget(self.cine_crudo_montage_cine_speed_spin)

		self.cine_crudo_montage_center_btn = QToolButton()
		self.cine_crudo_montage_center_btn.setText("Centrar")
		self.cine_crudo_montage_center_btn.setCheckable(True)
		self.cine_crudo_montage_center_btn.setChecked(bool(getattr(self, "cine_crudo_montage_center_cuts", False)))
		self.cine_crudo_montage_center_btn.setToolTip("Centra cada corte en su casilla (centroide de intensidad → centro del panel).")
		self.cine_crudo_montage_center_btn.toggled.connect(self._on_montage_center_toggled)
		toolbar.addWidget(self.cine_crudo_montage_center_btn)

		toolbar.addWidget(QLabel("Interp"))
		self.cine_crudo_montage_interp_combo = QComboBox()
		self.cine_crudo_montage_interp_combo.addItems(["Píxel", "Bilineal", "Bicúbico", "Hanning", "Lanczos"])
		self.cine_crudo_montage_interp_combo.setCurrentText(str(getattr(self, "cine_crudo_montage_interp", "Bilineal")))
		self.cine_crudo_montage_interp_combo.setMaximumWidth(90)
		self.cine_crudo_montage_interp_combo.setToolTip(
			"Interpolación de VISUALIZACIÓN del montaje del informe (no altera datos ni análisis).\n"
			"Píxel = vóxel crudo. Bilineal = intermedio fiel (recomendado).\n"
			"Bicúbico/Hanning/Lanczos = progresivamente más suaves.")
		self.cine_crudo_montage_interp_combo.currentIndexChanged.connect(self._on_montage_interp_changed)
		toolbar.addWidget(self.cine_crudo_montage_interp_combo)
		toolbar.addWidget(QLabel("Suav."))
		self.cine_crudo_montage_smooth_spin = QDoubleSpinBox()
		self.cine_crudo_montage_smooth_spin.setRange(0.0, 3.0)
		self.cine_crudo_montage_smooth_spin.setSingleStep(0.2)
		self.cine_crudo_montage_smooth_spin.setDecimals(1)
		self.cine_crudo_montage_smooth_spin.setValue(float(getattr(self, "cine_crudo_montage_smooth", 0.0)))
		self.cine_crudo_montage_smooth_spin.setMaximumWidth(56)
		self.cine_crudo_montage_smooth_spin.setToolTip(
			"Suavizado gaussiano EXTRA [px] del montaje (post-filtro de display, no altera datos).\n"
			"0.0 = solo la interpolación elegida. >0 = agrega difuminado gaussiano encima.")
		self.cine_crudo_montage_smooth_spin.valueChanged.connect(self._on_montage_smooth_changed)
		toolbar.addWidget(self.cine_crudo_montage_smooth_spin)

		self.cine_crudo_montage_export_btn = QToolButton()
		self.cine_crudo_montage_export_btn.setText("Guardar PNG")
		self.cine_crudo_montage_export_btn.setToolTip("Exporta el montaje clínico actual como PNG en la ubicación que elijas.")
		self.cine_crudo_montage_export_btn.clicked.connect(self._export_cine_crudo_montage_png)
		self.cine_crudo_montage_export_btn.setEnabled(False)
		toolbar.addWidget(self.cine_crudo_montage_export_btn)

		self.cine_crudo_tips_btn = QToolButton()
		self.cine_crudo_tips_btn.setText("Tips")
		self.cine_crudo_tips_btn.setToolTip("Guía rápida de atajos y controles del montaje (click, rueda, flechas, zoom y pan).")
		self.cine_crudo_tips_btn.clicked.connect(self._show_cine_crudo_montage_tips)
		toolbar.addWidget(self.cine_crudo_tips_btn)

	def _on_montage_center_toggled(self, checked):
		self.cine_crudo_montage_center_cuts = bool(checked)
		if self.cine_crudo_preview_mode == "sa_montage":
			self._schedule_montage_refresh(0)

	def _on_montage_interp_changed(self, _idx):
		if hasattr(self, "cine_crudo_montage_interp_combo"):
			self.cine_crudo_montage_interp = str(self.cine_crudo_montage_interp_combo.currentText())
		if self.cine_crudo_preview_mode == "sa_montage":
			self._schedule_montage_refresh(0)

	def _on_montage_smooth_changed(self, value):
		self.cine_crudo_montage_smooth = float(value)
		if self.cine_crudo_preview_mode == "sa_montage":
			self._schedule_montage_refresh(0)

	def _on_montage_window_changed(self, _value):
		if hasattr(self, "cine_crudo_montage_win_low_spin"):
			self.cine_crudo_montage_win_low = float(self.cine_crudo_montage_win_low_spin.value())
		if hasattr(self, "cine_crudo_montage_win_high_spin"):
			self.cine_crudo_montage_win_high = float(self.cine_crudo_montage_win_high_spin.value())
		if self.cine_crudo_preview_mode == "sa_montage":
			self._schedule_montage_refresh(0)

	def _export_cine_crudo_montage_png(self):
		"""Exporta el montaje clínico actual como PNG donde elija el usuario."""
		if not self.cine_crudo_axes_for_export:
			QMessageBox.information(self, "SINCRO", "Primero generá los cortes y el montaje ('Ver montaje').")
			return
		src = os.path.join(self.output_dir, "sa_montage.png")
		if self.cine_crudo_preview_mode != "sa_montage" or not os.path.exists(src):
			# Forzar HQ 512: el PNG solo se escribe en render HQ.
			self._montage_panel_px = 512
			self._show_cine_crudo_sa_montage()
		if not os.path.exists(src):
			QMessageBox.warning(self, "SINCRO", "No hay montaje para exportar. Generalo primero con 'Ver montaje'.")
			return
		suggested = os.path.join(self.output_dir, "montaje_clinico.png")
		path, _flt = QFileDialog.getSaveFileName(self, "Guardar montaje clínico", suggested, "Imagen PNG (*.png)")
		if not path:
			return
		if not path.lower().endswith(".png"):
			path += ".png"
		try:
			import shutil
			shutil.copyfile(src, path)
			self._log(f"Montaje exportado a: {path}")
			QMessageBox.information(self, "SINCRO", f"Montaje guardado en:\n{path}")
		except Exception as exc:
			self._log(f"[ERROR] Export montaje: {exc}")
			QMessageBox.warning(self, "SINCRO", f"No se pudo guardar el montaje:\n{exc}")

	def _on_montage_gate_range_changed(self, _value):
		"""Actualiza rango de gates del montaje en vivo (click-only friendly)."""
		if not (hasattr(self, "cine_crudo_gate_from_spin") and hasattr(self, "cine_crudo_gate_to_spin")):
			return
		g0 = int(self.cine_crudo_gate_from_spin.value())
		g1 = int(self.cine_crudo_gate_to_spin.value())
		if g0 > g1:
			# Mantener rango válido sin fricción: mover el extremo opuesto según cuál cambió.
			sender = self.sender()
			if sender is self.cine_crudo_gate_from_spin:
				self.cine_crudo_gate_to_spin.blockSignals(True)
				self.cine_crudo_gate_to_spin.setValue(g0)
				self.cine_crudo_gate_to_spin.blockSignals(False)
				g1 = g0
			else:
				self.cine_crudo_gate_from_spin.blockSignals(True)
				self.cine_crudo_gate_from_spin.setValue(g1)
				self.cine_crudo_gate_from_spin.blockSignals(False)
				g0 = g1
		self.cine_crudo_gate_from = int(g0)
		self.cine_crudo_gate_to = int(g1)
		if self.cine_crudo_preview_mode == "sa_montage":
			self._schedule_montage_refresh(8, fast=True)

	def _set_montage_gate_full_range(self):
		"""Acceso rápido: usar todos los gates disponibles en el montaje."""
		if not self.cine_crudo_axes_for_export:
			return
		sa = np.asarray(self.cine_crudo_axes_for_export.get("SA", []), dtype=np.float64)
		n = int(sa.shape[0]) if sa.ndim == 4 else 1
		if hasattr(self, "cine_crudo_gate_from_spin") and hasattr(self, "cine_crudo_gate_to_spin"):
			self.cine_crudo_gate_from_spin.blockSignals(True)
			self.cine_crudo_gate_to_spin.blockSignals(True)
			self.cine_crudo_gate_from_spin.setRange(1, max(1, n))
			self.cine_crudo_gate_to_spin.setRange(1, max(1, n))
			self.cine_crudo_gate_from_spin.setValue(1)
			self.cine_crudo_gate_to_spin.setValue(max(1, n))
			self.cine_crudo_gate_from_spin.blockSignals(False)
			self.cine_crudo_gate_to_spin.blockSignals(False)
		self.cine_crudo_gate_from = 1
		self.cine_crudo_gate_to = max(1, n)
		if self.cine_crudo_preview_mode == "sa_montage":
			self._schedule_montage_refresh(0)

	def _on_rest_offset_changed(self, axis, value):
		if axis in self.cine_crudo_rest_offset:
			self.cine_crudo_rest_offset[axis] = int(value)
		if self.cine_crudo_preview_mode == "sa_montage" and self.cine_crudo_axes_for_export_rest:
			self._schedule_montage_refresh(8, fast=True)

	def _mark_cine_crudo_as_rest(self):
		"""Guarda los cortes actuales como estudio de REPOSO para el montaje comparativo."""
		if not self.cine_crudo_axes_for_export:
			QMessageBox.information(self, "SINCRO", "Primero generá los cortes con 'Generar cortes'.")
			return
		self.cine_crudo_axes_for_export_rest = {k: np.array(v, copy=True) for k, v in self.cine_crudo_axes_for_export.items()}
		self.cine_crudo_rest_source_label = self.cine_crudo_cut_source_label or "raw recon"
		self.cine_crudo_cut_thickness_mm_rest = float(getattr(self, "cine_crudo_cut_thickness_mm", 0.0) or 0.0)
		self._log(
			f"Cortes actuales marcados como REPOSO ({self.cine_crudo_rest_source_label}). "
			"Ahora generá/reorientá el ESFUERZO y tocá 'Ver montaje' para la comparación doble fila."
		)
		QMessageBox.information(self, "SINCRO", "Cortes guardados como REPOSO.\n\nAhora generá el ESFUERZO y volvé a 'Ver montaje'.")

	def _save_cine_crudo_axes_dicoms(self):
		"""Guarda SA/HLA/VLA generados como tres DICOM gated multiframe."""
		if not self.cine_crudo_axes_for_export:
			QMessageBox.information(self, "SINCRO", "Primero generá los cortes SA/HLA/VLA.")
			return
		try:
			from PyQt6.QtWidgets import QFileDialog
			from core.dicom_export import save_cardiac_axes_dicoms

			source = self.cine_crudo_cut_study or self.cine_crudo_raw_study_for_recon or self.study
			base_patient = str(getattr(source, "patient_id", "") or getattr(source, "patient_name", "") or "study")
			base_patient = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_patient)[:32] or "study"
			folder = QFileDialog.getExistingDirectory(self, "Guardar ejes DICOM", self.output_dir)
			if not folder:
				return
			paths = save_cardiac_axes_dicoms(
				self.cine_crudo_axes_for_export,
				folder,
				source_study=source,
				base_name=f"GammaSync_{base_patient}_axes",
				slice_thickness_mm=float(self.cine_crudo_cut_study.slice_thickness_mm or self.cine_crudo_cut_study.z_spacing_mm or 1.0) if self.cine_crudo_cut_study is not None else None,
				extra_description="SA/HLA/VLA",
			)
			self._log("Ejes DICOM guardados: " + "; ".join(f"{k}={os.path.basename(v)}" for k, v in paths.items()))
			QMessageBox.information(
				self,
				"SINCRO",
				"Ejes DICOM guardados:\n" + "\n".join(f"• {axis}: {path}" for axis, path in paths.items()),
			)
		except Exception as exc:
			self._log(f"[ERROR] Guardar ejes DICOM falló: {exc}")
			QMessageBox.warning(self, "SINCRO", f"No se pudieron guardar los ejes DICOM:\n{exc}")

	def _copy_stress_rois_to_rest(self):
		"""Copia ROI manuales de esfuerzo a reposo como semilla editable."""
		stress_state = self._dual_session().stage("stress")
		rest_state = self._dual_session().stage("rest")
		if stress_state.cut_study is None or rest_state.cut_study is None:
			QMessageBox.information(self, "SINCRO", "Primero generá cortes para Esfuerzo y Reposo.")
			return
		text = str(self.primary_manual_rois_text or self.manual_rois.toPlainText() or "").strip()
		if not text:
			QMessageBox.information(self, "SINCRO", "Primero definí o procesá los ROI de Esfuerzo.")
			return
		rois = self._parse_manual_rois_text(text)
		if not rois:
			QMessageBox.information(self, "SINCRO", "No hay ROI válidos de Esfuerzo para copiar.")
			return
		# Mantener solo slices existentes en Reposo; la copia queda manual/editable.
		try:
			n_rest = int(np.asarray(rest_state.cut_study.cube).shape[1])
		except Exception:
			n_rest = 0
		copied = {int(k): tuple(float(v) for v in roi) for k, roi in rois.items() if 0 <= int(k) < n_rest}
		if not copied:
			QMessageBox.information(self, "SINCRO", "Los ROI de Esfuerzo no coinciden con los cortes disponibles de Reposo.")
			return
		self.compare_manual_rois_text = self._format_manual_rois(copied)
		self.compare_manual_rois_autogenerated = False
		# Copiar por texto no actualizaba el cine secundario si estaba visible como
		# panel paralelo (active_cine_source suele seguir en primary). Cargar los
		# ROI en el widget SIEMPRE y forzar Segmentación=manual: así la primera
		# corrida usa la semilla copiada, no la máscara automática de toda la imagen.
		self.cine_compare.set_manual_rois(copied)
		if self.seg_method.currentText() != "manual":
			self.seg_method.setCurrentText("manual")
			self._log("[DUAL] ROI E→R: Segmentación cambiada a manual para respetar la copia.")
		if self.active_cine_source == "compare":
			self._set_manual_rois_text(self.compare_manual_rois_text, autogenerated=False)
		self._log(f"[DUAL] ROI Esfuerzo→Reposo copiados ({len(copied)} cortes). Ajuste fino de Reposo habilitado.")
		self.statusBar().showMessage("ROI copiados a Reposo; ajustá fino y procesá ambas etapas.", 5000)

	def _process_cine_crudo_reconstruction(self, _force_stage: str | None = None):
		"""Procesa fase/FEVI desde cortes SA; con dos etapas, procesa ambas en orden."""
		if _force_stage not in ("stress", "rest"):
			_force_stage = None
		if _force_stage is None:
			sess = self._dual_session()
			if sess.stage("stress").cut_study is not None and sess.stage("rest").cut_study is not None:
				self._log("[DUAL] Procesar recon: Esfuerzo → Reposo (fase + FEVI).")
				# Esfuerzo es el primario visual; se procesa primero y reposo se
				# incorpora automáticamente como compare_bundle desde memoria.
				return self._process_cine_crudo_reconstruction(_force_stage="stress")
		if _force_stage is not None:
			self._set_active_cine_crudo_stage(_force_stage, refresh_view=False, force=True)
			self._cine_crudo_recon_stage = _force_stage
		# IMPORTANTE: recién después de seleccionar la etapa consultar el property
		# compatiblizado. Antes se consultaba el slot dejado por la última etapa
		# (usualmente Reposo), por lo que 'Procesar recon' podía promover la etapa
		# equivocada y duplicarla en FEVI/montaje.
		if self.cine_crudo_cut_study is None:
			QMessageBox.information(self, "SINCRO", "Primero tocá Generar cortes y revisá SA/HLA/VLA en comparacion_ejes.")
			return
		path = self.file_edit.text().strip()
		if not path or not os.path.exists(path):
			QMessageBox.information(self, "SINCRO", "Para procesar fase/FEVI desde la reconstrucción necesitás un DICOM cargado con path válido.")
			return
		self.study = self.cine_crudo_cut_study
		self._cache_study_sig = self._build_study_signature(path)
		self._cache_seg_sig = ""
		self._cache_phase_sig = ""
		self._invalidate_output_cache()
		self._log("Procesando sincronía/FEVI desde los cortes SA generados en cine_crudo.")
		self.process_current()
		# Guardar los resultados clínicos recién calculados en la etapa que los
		# originó (StageState no tenía que depender del UI global para FEVI/fase).
		stage_now = str(getattr(self, "_cine_crudo_recon_stage", "stress") or "stress")
		stage_state = self._dual_session().stage(stage_now)
		stage_state.seg = self.seg
		stage_state.phase = self.phase_result
		stage_state.metrics = self.metrics
		stage_state.metrics_raw = self.metrics_raw
		stage_state.phase_by_seg = self.phase_by_seg
		stage_state.territory = self.territory
		stage_state.ef = self._estimate_ef_for(self.study, self.seg)
		# Plan C Fase 3: si la etapa opuesta ya tiene cortes SA en memoria,
		# levantar comparación procesada sin depender de un DICOM reconstruido en disco.
		primary_stage = "rest" if getattr(self, "_cine_crudo_recon_stage", "stress") == "rest" else "stress"
		compare_stage = "stress" if primary_stage == "rest" else "rest"
		if self._dual_session().stage(compare_stage).cut_study is not None:
			try:
				# En el primer procesamiento dual, usar la ROI de Esfuerzo como
				# semilla editable para Reposo, igual que la reorientación. El usuario
				# puede ajustar fino la segunda etapa antes de reprocesarla.
				if primary_stage == "stress" and compare_stage == "rest" and not str(self.compare_manual_rois_text or "").strip():
					self._copy_stress_rois_to_rest()
				if self._load_compare_bundle_from_stage_memory(compare_stage):
					other_state = self._dual_session().stage(compare_stage)
					self._log(
						f"[DUAL] Fase + FEVI de {('Reposo' if compare_stage == 'rest' else 'Esfuerzo')} "
						f"calculadas automáticamente desde memoria; comparación stress/rest activa "
						f"(fase={'OK' if other_state.phase is not None else 'N/D'}, "
						f"métricas={'OK' if other_state.metrics is not None else 'N/D'}, "
						f"FEVI={'OK' if other_state.ef is not None else 'N/D'})."
					)
				else:
					self._log(f"[WARN] [DUAL] No se pudo procesar {compare_stage} desde memoria; no habrá segunda columna clínica.")
				# Refrescar DESPUÉS de poblar compare_bundle/DualSession, no solo al
				# terminar process_current(), que corrió antes de esta segunda etapa.
				self._refresh_summary()
			except Exception as exc:
				self._log(f"[WARN] No se pudo cargar comparación dual desde memoria: {exc}")

	def _show_cine_crudo_shift_curves(self):
		"""Muestra curvas de shifts X/Y vs frame (estilo Xeleris) para depurar la corrección."""
		if self.cine_crudo_motion_result is None:
			QMessageBox.information(self, "SINCRO", "Primero ejecutá una corrección.")
			return
		try:
			import matplotlib.pyplot as plt
			sy = np.asarray(self.cine_crudo_motion_result.get("applied_shifts_y", []), dtype=np.float64)
			sx = np.asarray(self.cine_crudo_motion_result.get("applied_shifts_x", []), dtype=np.float64)
			n = int(max(len(sy), len(sx)))
			if n == 0:
				QMessageBox.information(self, "SINCRO", "No hay shifts para graficar.")
				return
			x = np.arange(n)
			fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
			fig.patch.set_facecolor("#0b1220")
			axes[0].plot(x, sy, "-o", color="gold", ms=3)
			axes[0].axhline(0.0, color="#94a3b8", lw=1, alpha=0.6)
			axes[0].set_ylabel("Shift Y (px)", color="white")
			axes[0].set_title("Y Shifts vs Frame", color="white", fontsize=10)
			axes[0].grid(alpha=0.25)
			axes[1].plot(x, sx, "-o", color="cyan", ms=3)
			axes[1].axhline(0.0, color="#94a3b8", lw=1, alpha=0.6)
			axes[1].set_ylabel("Shift X (px)", color="white")
			axes[1].set_xlabel("Frame", color="white")
			axes[1].set_title("X Shifts vs Frame", color="white", fontsize=10)
			axes[1].grid(alpha=0.25)
			for ax in axes:
				ax.set_facecolor("#0b1220")
				ax.tick_params(colors="white")
				for s in ax.spines.values():
					s.set_color("#334155")
			ref_txt = str(self.cine_crudo_ref_index) if self.cine_crudo_ref_index is not None else "auto(frame actual)"
			meth = str(self.cine_crudo_motion_result.get("method_auto_selected") or self.cine_crudo_motion_result.get("method") or "?")
			fig.suptitle(f"Curvas de shift — método {meth} | ref {ref_txt}", color="white", fontsize=11, fontweight="bold")
			fig.tight_layout(rect=[0, 0, 1, 0.95])
			out_png = os.path.join(self.output_dir, "motion_shift_curves.png")
			fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
			plt.close(fig)
			if "ungated" in self.preview_labels:
				pix = QPixmap(out_png)
				self.preview_pixmaps["ungated"] = pix
				self.preview_base_sizes["ungated"] = pix.size()
				self._apply_preview_zoom("ungated")
				self._select_tab_by_title("ungated")
			self._log("Curvas de shift generadas: motion_shift_curves.png")
		except Exception as exc:
			self._log(f"[WARN] Curvas de shift fallaron: {exc}")

	def _refresh_cine_crudo_view(self):
		# Al refrescar el cine crudo (máscara/pick/linea ref/corrección), volver al
		# modo interactivo de proyecciones y limpiar estados de previews especiales
		# (cut_limits/sa_montage/generated), que bloquean handlers de mouse.
		self.cine_crudo_preview_mode = None
		self._cine_crudo_drag_marker = None
		self._montage_drag_axis = None
		self._montage_drag_mode = None
		self._montage_drag_start_x = None
		source = str(self.cine_crudo_source_combo.currentText()) if hasattr(self, "cine_crudo_source_combo") else "UngGat"
		self._load_cine_crudo_frames(source)

	def _load_cine_crudo_synthetic(self):
		"""Carga un crudo sintético para probar motion correction sin DICOM externo."""
		try:
			from core.raw_projections import center_of_mass_tracking, make_synthetic_raw_motion_projections

			raw = make_synthetic_raw_motion_projections()
			self.study = dicom_loader.GatedStudy(
				cube=raw.projections,
				n_gates=raw.n_gates,
				n_slices=raw.n_angles,
				rows=raw.rows,
				cols=raw.cols,
				pixel_spacing=(6.4, 6.4),
				source_path="",
				image_type=["SYNTHETIC", "GATED TOMO"],
				series_description=raw.series_description,
				study_description=raw.study_description,
				patient_name=raw.patient_name,
				patient_id=raw.patient_id,
				study_instance_uid="SYNTHETIC.RAW.MOTION",
				reconstructed=False,
				qc_first_harmonic=0.0,
				qc_passed=False,
				gating_info=raw.gating_info,
				notes=raw.notes,
			)
			self.study.angles_deg = raw.angles_deg

			self.cine_crudo_seed = None
			self.cine_crudo_seed_mode = False
			self.cine_crudo_band_upper = None
			self.cine_crudo_band_lower = None
			self.cine_crudo_compare_line_y = None
			self._cine_crudo_drag_marker = None
			self.cine_crudo_ref_index = None
			self.cine_crudo_corrected_projections = None
			self.cine_crudo_motion_result = None
			if self.cine_crudo_compare_check is not None:
				self.cine_crudo_compare_check.setChecked(False)
				self.cine_crudo_compare_check.setEnabled(False)
			if hasattr(self, "cine_crudo_source_combo"):
				self.cine_crudo_source_combo.setCurrentText("UngGat")
			if hasattr(self, "cine_crudo_method_combo"):
				self.cine_crudo_method_combo.setCurrentText("Sinusoide")
			if hasattr(self, "cine_crudo_axis_combo"):
				self.cine_crudo_axis_combo.setCurrentText("Y")
			if hasattr(self, "cine_crudo_roi_mode_combo"):
				self.cine_crudo_roi_mode_combo.setCurrentText("Banda Y")
			if hasattr(self, "cine_crudo_roi_spin"):
				self.cine_crudo_roi_spin.setValue(8)
			if hasattr(self, "cine_crudo_liver_suppress_check"):
				self.cine_crudo_liver_suppress_check.setChecked(True)
			if hasattr(self, "cine_crudo_liver_suppress_spin"):
				self.cine_crudo_liver_suppress_spin.setValue(60)

			self._load_cine_crudo_frames("UngGat")
			self._select_tab_by_title("cine_crudo")
			ty = center_of_mass_tracking(raw.projections, axis="y")
			tx = center_of_mass_tracking(raw.projections, axis="x")
			self._log(
				"Sintético raw cargado: 8 gates × 60 ángulos × 64×64 | "
				"corazón con X rotacional + saltos Y, hígado/intestino inferior intenso. "
				f"COM inicial max Y={ty.get('max_shift_px')}px X={tx.get('max_shift_px')}px. "
				"Sugerido: Elegir corazón, Banda Y, Atenuar hígado 60%, Corregir."
			)
			self._set_progress(100, "Sintético raw cargado")
			self.statusBar().showMessage("Sintético raw cargado para pruebas de motion correction")
		except Exception as exc:
			self._log(f"[WARN] No se pudo cargar sintético raw: {exc}")
			QMessageBox.warning(self, "SINCRO", f"No se pudo cargar el sintético raw:\n{exc}")

	def _open_cine_crudo_fine_adjust(self):
		if self.study is None or bool(getattr(self.study, "reconstructed", True)):
			return
		if not self.cine_crudo_frames:
			QMessageBox.information(self, "SINCRO", "Primero cargá el cine del crudo.")
			return
		projections = np.asarray(self.study.cube, dtype=np.float64)
		n_angles = int(projections.shape[1])
		current = int(self.cine_crudo_index) % max(1, len(self.cine_crudo_frames))
		result = self.cine_crudo_motion_result or {}
		sy = np.asarray(result.get("applied_shifts_y", np.zeros((n_angles,), dtype=np.float64)), dtype=np.float64)
		sx = np.asarray(result.get("applied_shifts_x", np.zeros((n_angles,), dtype=np.float64)), dtype=np.float64)
		dialog = QDialog(self)
		dialog.setWindowTitle(f"Ajuste fino motion correction — ángulo {current}")
		form = QFormLayout(dialog)
		sy_spin = QDoubleSpinBox()
		sy_spin.setRange(-30.0, 30.0)
		sy_spin.setDecimals(2)
		sy_spin.setSingleStep(0.25)
		sy_spin.setValue(float(sy[current]) if current < sy.size else 0.0)
		sx_spin = QDoubleSpinBox()
		sx_spin.setRange(-30.0, 30.0)
		sx_spin.setDecimals(2)
		sx_spin.setSingleStep(0.25)
		sx_spin.setValue(float(sx[current]) if current < sx.size else 0.0)
		form.addRow("Shift Y (px)", sy_spin)
		form.addRow("Shift X (px)", sx_spin)
		buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
		form.addRow(buttons)
		buttons.accepted.connect(dialog.accept)
		buttons.rejected.connect(dialog.reject)
		if dialog.exec() != QDialog.DialogCode.Accepted:
			return
		sy[current] = float(sy_spin.value())
		sx[current] = float(sx_spin.value())
		try:
			from core.raw_projections import motion_correct_projections
			method = str(self.cine_crudo_method_combo.currentText()).lower() if hasattr(self, "cine_crudo_method_combo") else "com"
			axis = str(self.cine_crudo_axis_combo.currentText()).lower() if hasattr(self, "cine_crudo_axis_combo") else "y"
			threshold = self._cine_crudo_threshold_value()
			result = motion_correct_projections(
				projections,
				axis=axis,
				method=method,
				threshold_frac=threshold,
				manual_shifts_y=sy,
				manual_shifts_x=sx,
			)
			self.cine_crudo_motion_result = result
			self.cine_crudo_corrected_projections = np.asarray(result.get("corrected"), dtype=np.float64)
			if self.cine_crudo_compare_check is not None:
				self.cine_crudo_compare_check.setEnabled(True)
			self._log(f"Ajuste fino aplicado en ángulo {current}: shiftY={sy[current]:+.2f}px shiftX={sx[current]:+.2f}px")
			self._refresh_cine_crudo_view()
		except Exception as exc:
			self._log(f"[WARN] Ajuste fino falló: {exc}")

	def _set_cine_crudo_frame(self, idx: int):
		if not self.cine_crudo_frames:
			if "cine_crudo" in self.preview_labels:
				self.preview_labels["cine_crudo"].setText("Cargá un estudio crudo (proyecciones gated)")
			return
		n = len(self.cine_crudo_frames)
		self.cine_crudo_index = int(idx) % n
		# Guardar el frame actual para que el threshold/máscara se aplique sobre él (no siempre frame 0).
		self._cine_crudo_current_frame = self.cine_crudo_index
		pix = self.cine_crudo_frames[self.cine_crudo_index]
		self.preview_pixmaps["cine_crudo"] = pix
		self.preview_base_sizes["cine_crudo"] = pix.size()
		self._apply_preview_zoom("cine_crudo")
		if hasattr(self, "cine_crudo_frame_label"):
			counts_txt = ""
			counts = getattr(self, "cine_crudo_counts", None)
			if counts is not None and self.cine_crudo_index < len(counts):
				counts_txt = f" · {int(counts[self.cine_crudo_index]):,} cts"
			matrix_txt = f" · {self.cine_crudo_matrix_txt}" if self.cine_crudo_matrix_txt else ""
			self.cine_crudo_frame_label.setText(f"Img {self.cine_crudo_index + 1:02d}/{n}{counts_txt}{matrix_txt}")

	def _advance_cine_crudo_frame(self):
		if not self.cine_crudo_frames:
			self.cine_crudo_timer.stop()
			self.cine_crudo_playing = False
			self._update_cine_crudo_toggle_text()
			return
		# Si estamos mostrando la reconstrucción, no pisar la imagen con frames del cine.
		if str(getattr(self, "cine_crudo_preview_mode", None)) == "recon_qc":
			self.cine_crudo_timer.stop()
			self.cine_crudo_playing = False
			self._update_cine_crudo_toggle_text()
			return
		n = len(self.cine_crudo_frames)
		modo = str(self.cine_crudo_mode_combo.currentText()) if hasattr(self, "cine_crudo_mode_combo") else "Continuo"
		if modo == "Rebote":
			next_idx = self.cine_crudo_index + self.cine_crudo_direction
			if next_idx >= n - 1:
				next_idx = n - 1
				self.cine_crudo_direction = -1
			elif next_idx <= 0:
				next_idx = 0
				self.cine_crudo_direction = 1
			self._set_cine_crudo_frame(next_idx)
		else:  # Continuo (loop)
			self._set_cine_crudo_frame((self.cine_crudo_index + 1) % n)

	def _step_cine_crudo(self, delta: int):
		if not self.cine_crudo_frames:
			return
		self._set_cine_crudo_frame((self.cine_crudo_index + int(delta)) % len(self.cine_crudo_frames))
		# Si la máscara está activa, regenerar para que se aplique sobre el frame recién elegido.
		if self.cine_crudo_mask_check is not None and self.cine_crudo_mask_check.isChecked():
			self._load_cine_crudo_frames(
				str(self.cine_crudo_source_combo.currentText()) if hasattr(self, "cine_crudo_source_combo") else "UngGat"
			)

	def _toggle_cine_crudo(self):
		if not self.cine_crudo_frames:
			return
		# Si venimos de mostrar la reconstrucción (imagen estática) y el usuario pide
		# play, volver al cine del crudo: restaurar modo y zoom propios del cine.
		if str(getattr(self, "cine_crudo_preview_mode", None)) == "recon_qc":
			self.cine_crudo_preview_mode = None
			self.preview_zoom["cine_crudo"] = self._default_preview_zoom("cine_crudo")
			self._set_cine_crudo_frame(int(getattr(self, "cine_crudo_index", 0)))
		self.cine_crudo_playing = not bool(self.cine_crudo_playing)
		if self.cine_crudo_playing:
			self.cine_crudo_timer.start()
		else:
			self.cine_crudo_timer.stop()
		self._update_cine_crudo_toggle_text()

	def _update_cine_crudo_toggle_text(self):
		if self.cine_crudo_play_btn is not None:
			self.cine_crudo_play_btn.setText("⏸" if self.cine_crudo_playing else "▶")

	def _on_cine_crudo_speed_changed(self, value: int):
		self.cine_crudo_timer.setInterval(max(40, int(value)))

	def _show_cine_crudo_transaxial_grid(self):
		"""Grilla de cortes transaxiales con máscara para discriminar corazón de hígado antes del pick (estilo Odyssey)."""
		if self.study is None or bool(getattr(self.study, "reconstructed", True)):
			QMessageBox.information(self, "SINCRO", "Cargá un estudio crudo primero.")
			return
		try:
			import matplotlib.pyplot as plt
			from core.raw_projections import reconstruct_transaxial_slices

			projections = np.asarray(self.study.cube, dtype=np.float64)
			angles_deg = getattr(self.study, "angles_deg", None)
			# Cortes transaxiales anatómicos del bruto (FBP rápido), como los "cortes rápidos" de Odyssey.
			vol = reconstruct_transaxial_slices(projections, angles_deg)
			n_slices = vol.shape[0]
			thr = self._cine_crudo_threshold_value()

			# Grilla de cortes transaxiales con máscara. Limitar a 64 subplots para evitar saturar matplotlib.
			max_plots = 64
			if n_slices > max_plots:
				step = int(np.ceil(n_slices / max_plots))
				vol = vol[::step]
				n_slices = vol.shape[0]
				self._log(f"Grilla pick: submuestreo cada {step} cortes ({n_slices} de {vol.shape[0]}) para performance.")
			cols = int(np.ceil(np.sqrt(n_slices)))
			rows = int(np.ceil(n_slices / cols))
			fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6))
			axes = np.atleast_1d(axes).ravel()
			p99 = float(np.percentile(vol, 99.0)) or 1.0
			for idx in range(rows * cols):
				ax = axes[idx]
				ax.axis("off")
				ax.set_facecolor("#0b1220")
				if idx < n_slices:
					img = vol[idx]
					ax.imshow(img, cmap="odyssey_cool", vmin=0, vmax=p99)
					mask = img > (thr * img.max()) if img.max() > 0 else np.zeros_like(img, dtype=bool)
					# Contorno de la máscara solo si tiene píxeles (evita crash con máscara vacía).
					if mask.any():
						try:
							ax.contour(mask.astype(np.uint8), levels=[0.5], colors="spring", linewidths=0.8)
						except Exception:
							pass
					ax.set_title(f"{idx}", fontsize=7, color="white", pad=1)
			ctx_label = self._study_context_label(
				path_override=str(getattr(self, "_output_study_path_override", "") or self.file_edit.text().strip()),
				study_obj=self.study,
			)
			fig.suptitle(
				f"Grilla transaxial pick (FBP) — {ctx_label} | thr {thr:.2f} | "
				"Discriminá corazón de hígado y hacé pick con 'Elegir corazón' en cine_crudo",
				color="white", fontsize=10.5, fontweight="bold",
			)
			fig.patch.set_facecolor("#0b1220")
			fig.tight_layout(rect=[0, 0, 1, 0.94])
			out_png = os.path.join(self.output_dir, "grilla_pick_transaxial.png")
			fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
			plt.close(fig)
			if "ungated" in self.preview_labels:
				pix = QPixmap(out_png)
				self.preview_pixmaps["ungated"] = pix
				self.preview_base_sizes["ungated"] = pix.size()
				self._apply_preview_zoom("ungated")
				self._select_tab_by_title("ungated")
			self._log(f"Grilla transaxial pick generada (thr {thr:.2f}, {n_slices} cortes) → pestaña ungated.")
		except Exception as exc:
			self._log(f"[ERROR] Grilla pick falló: {exc}")
			QMessageBox.warning(self, "SINCRO", f"No se pudo generar la grilla pick:\n{exc}")

	def _cine_crudo_threshold_value(self) -> float:
		if hasattr(self, "cine_crudo_threshold_slider"):
			return float(self.cine_crudo_threshold_slider.value()) / 100.0
		return 0.20

	def _on_cine_crudo_threshold_changed(self, value: int):
		thr = float(value) / 100.0
		if hasattr(self, "cine_crudo_threshold_spin"):
			self.cine_crudo_threshold_spin.blockSignals(True)
			self.cine_crudo_threshold_spin.setValue(thr)
			self.cine_crudo_threshold_spin.blockSignals(False)
		# Actualización en tiempo real: solo regenerar frames si la máscara está activa.
		if self.cine_crudo_mask_check is not None and self.cine_crudo_mask_check.isChecked():
			self._load_cine_crudo_frames(
				str(self.cine_crudo_source_combo.currentText()) if hasattr(self, "cine_crudo_source_combo") else "UngGat"
			)

	def _on_cine_crudo_threshold_spin_changed(self, value: float):
		if hasattr(self, "cine_crudo_threshold_slider"):
			self.cine_crudo_threshold_slider.blockSignals(True)
			self.cine_crudo_threshold_slider.setValue(int(round(float(value) * 100.0)))
			self.cine_crudo_threshold_slider.blockSignals(False)
		if self.cine_crudo_mask_check is not None and self.cine_crudo_mask_check.isChecked():
			self._load_cine_crudo_frames(
				str(self.cine_crudo_source_combo.currentText()) if hasattr(self, "cine_crudo_source_combo") else "UngGat"
			)

	def _step_cine_crudo_threshold(self, delta: int):
		if not hasattr(self, "cine_crudo_threshold_slider"):
			return
		self.cine_crudo_threshold_slider.setValue(self.cine_crudo_threshold_slider.value() + int(delta))

	def _on_cine_crudo_seed_mode_toggled(self, checked: bool):
		self.cine_crudo_seed_mode = bool(checked)
		if not checked:
			if getattr(self, "_cine_crudo_active_stage", "stress") == "rest":
				self.cine_crudo_seed_compare = None
			else:
				self.cine_crudo_seed = None
			self.cine_crudo_band_upper = None
			self.cine_crudo_band_lower = None
			self.cine_crudo_compare_line_y = None
			self._cine_crudo_drag_marker = None
			self._log("Selección de órgano: modo automático (seed limpiado en la etapa activa).")
			self._refresh_cine_crudo_view()
		else:
			self._log("Selección de órgano: hacé CLICK en el corazón sobre la imagen (se fija en la etapa que clickees).")

	def _on_cine_crudo_mouse_press_safe(self, event, source_label=None):
		# Pan de imagen con botón medio o Alt+arrastre (fluido, sin rerender).
		if getattr(event, "button", lambda: None)() == Qt.MouseButton.MiddleButton or (
			getattr(event, "button", lambda: None)() == Qt.MouseButton.LeftButton and
			bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
		):
			lbl = source_label or (event.widget() if hasattr(event, "widget") else None)
			name = None
			if lbl is self.preview_labels.get("cine_crudo"):
				name = "cine_crudo"
			elif lbl is self.preview_labels.get("comparacion_ejes"):
				name = "comparacion_ejes"
			if name is not None:
				s = self._preview_scrollers.get(name)
				if s is not None:
					hb = s.horizontalScrollBar(); vb = s.verticalScrollBar()
					self._preview_pan_active = True
					self._preview_pan_anchor = (name, int(event.pos().x()), int(event.pos().y()), hb.value() if hb else 0, vb.value() if vb else 0)
					event.accept(); return
		if self.cine_crudo_preview_mode in {"recon_qc", "generated_cuts"}:
			event.accept()
			return
		if self.cine_crudo_preview_mode == "sa_montage":
			# 1) Click en el panel selecciona tira (SA/VLA/HLA) y habilita drag vivo
			# para desplazar su ventana de cortes (start).
			try:
				lbl = source_label or (event.widget() if hasattr(event, "widget") else None)
				cache = getattr(self, "_montage_gray_cache", {}) or {}
				rows_meta = cache.get("rows_meta", [])
				selection_key = self._montage_selection_key_at_event(event, source_label=lbl)
				if selection_key is None or not rows_meta:
					raise ValueError("sin geometría de montaje")
				row = next((r for r in rows_meta if str(r.get("selection_key", "")) == selection_key), None)
				if row is None:
					raise ValueError("fila de montaje no encontrada")
				axis_click = str(row.get("prefix", "SA"))
				ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
				selected = set(getattr(self, "cine_crudo_selected_stripes", set()) or set())
				if ctrl:
					if selection_key in selected:
						selected.remove(selection_key)
					else:
						selected.add(selection_key)
				else:
					selected = {selection_key}
				self.cine_crudo_selected_stripes = selected or {selection_key}
				self.cine_crudo_focused_stripe = selection_key
				self._montage_focus_selection_key = selection_key
				self._montage_drag_axis = axis_click
				self._montage_drag_selection_key = selection_key
				self.cine_crudo_selected_stripe = axis_click
				self._montage_drag_start_x = float(event.pos().x())
				self._montage_drag_start_off = int(
					(getattr(self, "cine_crudo_stripe_start_by_stage", {}) or {})
					.get(str(row.get("tag", "") or "ESFUERZO").upper(), {})
					.get(axis_click, 1)
				)
				# Para selección (sin mover cortes) redibujar SOLO el overlay desde
				# caché: feedback visual inmediato, sin re-render de imágenes.
				self._refresh_montage_selection_overlay()
				self._log(f"Montaje: tira activa {selection_key}" + (" (selección múltiple)" if ctrl else ""))
				self.statusBar().showMessage(f"Montaje: tira activa {selection_key}", 1500)
			except Exception as exc:
				self._log(f"[WARN] Click de tira no resuelto: {exc}")

			# Drag horizontal en montaje: la fila se resolvió arriba con la geometría
			# real. No usar mitades/tercios del QLabel: con zoom/plantillas eso hacía
			# que un click sobre una tira seleccionara otra.
			# La navegación por drag queda asociada SOLO a la tira clickeada.
			if self.cine_crudo_axes_for_export_rest:
				try:
					row_tag = str(row.get("tag", "") or "ESFUERZO").upper()
					# No convertir un click en drag de offset automáticamente: eso
					# sobrescribía el estado de la fila y ocultaba la selección. El
					# drag normal navega la tira focal, igual en ambas etapas.
					self._montage_drag_mode = None
					self._montage_drag_start_x = float(event.pos().x())
					self._montage_drag_start_off = int(
						(getattr(self, "cine_crudo_stripe_start_by_stage", {}) or {})
						.get(row_tag, {}).get(axis_click, 1)
					)
				except Exception:
					self._montage_drag_axis = None
					self._montage_drag_mode = None
			else:
				# Sin reposo: usar bloque único para drag de rango de gates.
				try:
					lbl = source_label or (event.widget() if hasattr(event, "widget") else None)
					if hasattr(self, "cine_crudo_gate_from_spin") and hasattr(self, "cine_crudo_gate_to_spin"):
						g_from = int(self.cine_crudo_gate_from_spin.value())
						g_to = int(self.cine_crudo_gate_to_spin.value())
						self._montage_drag_start_x = float(event.pos().x())
						if abs(float(event.pos().x()) - 0.15 * float(lbl.width() if lbl else 1)) < abs(float(event.pos().x()) - 0.85 * float(lbl.width() if lbl else 1)):
							self._montage_drag_mode = "gate_from"
							self._montage_drag_start_gate = g_from
						else:
							self._montage_drag_mode = "gate_to"
							self._montage_drag_start_gate = g_to
				except Exception:
					self._montage_drag_mode = None
			event.accept()
			return
		if self.cine_crudo_preview_mode == "cut_limits":
			try:
				# Vista dual: click sobre la etapa NO activa la selecciona (recuadro
				# rojo, como amyloidosis planar) sin editar markers en ese click.
				meta = self._cine_crudo_cut_limits_meta or {}
				clicked_stage = self._cine_crudo_limits_stage_at_event(event, source_label=source_label)
				if clicked_stage is not None and clicked_stage != str(meta.get("dual_active", "")):
					self._cine_crudo_drag_marker = None
					self._cine_crudo_drag_last_z = None
					self._cine_crudo_recon_stage = clicked_stage
					self._set_active_cine_crudo_stage(clicked_stage, refresh_view=False)
					self._preview_cine_crudo_cut_limits()
					stage_txt = "Esfuerzo" if clicked_stage == "stress" else "Reposo"
					self.statusBar().showMessage(f"Etapa activa: {stage_txt} — los markers Base/Ápex editan esta imagen", 5000)
					event.accept()
					return
				z = self._cine_crudo_cut_limits_event_to_slice(event, source_label=source_label)
				mk = self._cine_crudo_marker_at_limits_event(event, source_label=source_label)
				# UX robusta: si el click cae en zona válida del panel activo pero no
				# "tocó" exactamente la línea, enganchar el marcador más cercano.
				if mk is None and z is not None:
					try:
						z0 = int(meta.get("z0", 0))
						z1 = int(meta.get("z1", 0))
						mk = "base" if abs(int(z) - z0) <= abs(int(z) - z1) else "apex"
					except Exception:
						mk = None
				self._cine_crudo_drag_marker = mk
				self._cine_crudo_drag_last_z = None
				if mk in {"base", "apex"} and z is not None:
					# Reposiciona en el click y deja arrastre continuo en move.
					self._update_cine_crudo_cut_spins_from_drag(mk, int(z))
				preview = source_label
				if preview is None:
					preview = event.widget() if hasattr(event, "widget") else None
				if preview is not None:
					preview.setCursor(QCursor(Qt.CursorShape.SizeVerCursor if mk else Qt.CursorShape.ArrowCursor))
			except Exception:
				self._cine_crudo_drag_marker = None
			event.accept()
			return
		try:
			self._on_cine_crudo_image_clicked(event)
		except Exception as exc:
			self._cine_crudo_drag_marker = None
			self._log(f"[WARN] Evento mouse cine_crudo (press) falló: {exc}")

	def _on_cine_crudo_mouse_move_safe(self, event, source_label=None):
		if self._preview_pan_active and self._preview_pan_anchor is not None:
			try:
				name, x0, y0, hx0, vy0 = self._preview_pan_anchor
				s = self._preview_scrollers.get(name)
				if s is not None:
					hb = s.horizontalScrollBar(); vb = s.verticalScrollBar()
					dx = int(event.pos().x()) - int(x0)
					dy = int(event.pos().y()) - int(y0)
					if hb is not None:
						hb.setValue(int(hx0 - dx))
					if vb is not None:
						vb.setValue(int(vy0 - dy))
			except Exception:
				pass
			event.accept(); return
		if self.cine_crudo_preview_mode in {"recon_qc", "generated_cuts"}:
			event.accept()
			return
		if self.cine_crudo_preview_mode == "sa_montage":
			if self._montage_drag_mode in {"gate_from", "gate_to"} and self._montage_drag_start_x is not None:
				try:
					lbl = source_label or (event.widget() if hasattr(event, "widget") else None)
					w = float(lbl.width() if lbl else 1)
					sa = np.asarray(self.cine_crudo_axes_for_export.get("SA", []), dtype=np.float64)
					n_gates = int(sa.shape[0]) if sa.ndim == 4 else 1
					px_per_gate = max(1.0, w / max(1, n_gates))
					dg = int(round((float(event.pos().x()) - self._montage_drag_start_x) / px_per_gate))
					new_gate = int(np.clip(self._montage_drag_start_gate + dg, 1, max(1, n_gates)))
					if self._montage_drag_mode == "gate_from" and hasattr(self, "cine_crudo_gate_from_spin"):
						self.cine_crudo_gate_from_spin.setValue(new_gate)
					elif self._montage_drag_mode == "gate_to" and hasattr(self, "cine_crudo_gate_to_spin"):
						self.cine_crudo_gate_to_spin.setValue(new_gate)
				except Exception:
					pass
			elif self._montage_drag_mode == "rest_offset" and self._montage_drag_axis and self._montage_drag_start_x is not None:
				try:
					lbl = source_label or (event.widget() if hasattr(event, "widget") else None)
					w = float(lbl.width() if lbl else 1)
					sa_idx = self.cine_crudo_axes_for_export_rest.get("SA")
					ncols = int(np.asarray(sa_idx).shape[1]) if sa_idx is not None else 12
					px_per_col = max(1.0, w / max(1, ncols))
					dcols = int(round((float(event.pos().x()) - self._montage_drag_start_x) / px_per_col))
					new_off = self._montage_drag_start_off + dcols
					spin = {"SA": self.cine_crudo_rest_off_sa_spin, "VLA": self.cine_crudo_rest_off_vla_spin, "HLA": self.cine_crudo_rest_off_hla_spin}[self._montage_drag_axis]
					spin.setValue(int(np.clip(new_off, -40, 40)))
				except Exception:
					pass
			elif self._montage_drag_axis and self._montage_drag_start_x is not None:
				# Drag de la tira seleccionada: desplaza ventana start por eje.
				try:
					lbl = source_label or (event.widget() if hasattr(event, "widget") else None)
					w = float(lbl.width() if lbl else 1)
					# Aproximar número de columnas visibles del render actual.
					cols = int(getattr(self, "_montage_render_meta", {}).get("cols", 1) or 1)
					px_per_col = max(1.0, w / max(1, cols))
					dcols = int(round((float(event.pos().x()) - self._montage_drag_start_x) / px_per_col))
					axis_name = str(self._montage_drag_axis)
					cur = int(self._montage_drag_start_off)
					new_start = max(1, cur - dcols)
					key = str(getattr(self, "_montage_drag_selection_key", "") or "")
					stage_tag = key.split(":", 1)[0] if ":" in key else "ESFUERZO"
					starts = (getattr(self, "cine_crudo_stripe_start_by_stage", {}) or {}).setdefault(
						stage_tag, {"SA": 1, "VLA": 1, "HLA": 1}
					)
					starts[axis_name] = int(new_start)
					self._schedule_montage_refresh(8, fast=True)
				except Exception:
					pass
			event.accept()
			return
		if self.cine_crudo_preview_mode == "cut_limits":
			try:
				if self._cine_crudo_drag_marker in {"base", "apex"}:
					z = self._cine_crudo_cut_limits_event_to_slice(event, source_label=source_label)
					# Solo re-renderizar si el slice destino cambió: mover el mouse
					# dentro del mismo píxel de corte no debe forzar un re-render
					# (evita lag por renders redundantes durante el arrastre).
					if z is not None:
						last = getattr(self, "_cine_crudo_drag_last_z", None)
						if last is None or int(z) != int(last):
							self._cine_crudo_drag_last_z = int(z)
							self._update_cine_crudo_cut_spins_from_drag(self._cine_crudo_drag_marker, z)
				else:
					mk = self._cine_crudo_marker_at_limits_event(event, source_label=source_label)
					preview = source_label
					if preview is None:
						preview = event.widget() if hasattr(event, "widget") else None
					if preview is not None:
						preview.setCursor(QCursor(Qt.CursorShape.SizeVerCursor if mk else Qt.CursorShape.ArrowCursor))
			except Exception as exc:
				self._cine_crudo_drag_marker = None
				self._log(f"[WARN] Drag límites (move) falló: {exc}")
			event.accept()
			return
		try:
			self._on_cine_crudo_image_dragged(event)
		except Exception as exc:
			self._cine_crudo_drag_marker = None
			self._cine_crudo_set_drag_status(None)
			self._log(f"[WARN] Evento mouse cine_crudo (drag) falló: {exc}")

	def _on_cine_crudo_mouse_release_safe(self, event, source_label=None):
		if self._preview_pan_active:
			self._preview_pan_active = False
			self._preview_pan_anchor = None
			event.accept(); return
		if self.cine_crudo_preview_mode in {"recon_qc", "generated_cuts"}:
			event.accept()
			return
		if self.cine_crudo_preview_mode == "sa_montage":
			self._montage_drag_axis = None
			self._montage_drag_mode = None
			self._montage_drag_start_x = None
			self._montage_drag_selection_key = None
			event.accept()
			return
		if self.cine_crudo_preview_mode == "cut_limits":
			self._cine_crudo_drag_marker = None
			self._cine_crudo_drag_last_z = None
			self._schedule_cut_limits_hq_render()
			preview = source_label
			if preview is None:
				preview = event.widget() if hasattr(event, "widget") else None
			if preview is not None:
				preview.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
			event.accept()
			return
		try:
			self._on_cine_crudo_image_released(event)
		except Exception as exc:
			self._cine_crudo_drag_marker = None
			self._cine_crudo_set_drag_status(None)
			self._log(f"[WARN] Evento mouse cine_crudo (release) falló: {exc}")

	def _on_cine_crudo_mouse_wheel_safe(self, event, source_label=None):
		# Ctrl+rueda = zoom parejo en toda la pestaña (sin rerender).
		mods = event.modifiers() if hasattr(event, "modifiers") else Qt.KeyboardModifier.NoModifier
		if bool(mods & Qt.KeyboardModifier.ControlModifier):
			delta = int(event.angleDelta().y()) if hasattr(event, "angleDelta") else 0
			step = 0.08 if delta > 0 else (-0.08 if delta < 0 else 0.0)
			if step != 0.0:
				lbl = source_label or (event.widget() if hasattr(event, "widget") else None)
				name = "cine_crudo" if lbl is self.preview_labels.get("cine_crudo") else "comparacion_ejes"
				self._set_preview_zoom(name, self.preview_zoom.get(name, 0.5) + step)
				event.accept()
			return
		# QC AC dual: la rueda navega los cortes (frac 0..1) de todas las etapas.
		if getattr(self, "cine_crudo_preview_mode", None) == "ac_qc" and getattr(self, "_ac_qc_panels", None):
			delta = int(event.angleDelta().y()) if hasattr(event, "angleDelta") else 0
			if delta != 0:
				self._ac_qc_frac = float(np.clip(getattr(self, "_ac_qc_frac", 0.5) + (0.03 if delta > 0 else -0.03), 0.02, 0.98))
				self._render_ac_qc()
				event.accept()
			return
		if self.cine_crudo_preview_mode != "sa_montage":
			return
		try:
			delta = int(event.angleDelta().y()) if hasattr(event, "angleDelta") else 0
			step = 1 if delta > 0 else (-1 if delta < 0 else 0)
			if step == 0:
				return
			# Rueda: mueve TODAS las filas seleccionadas con Ctrl+click. Sin
			# selección múltiple conserva el foco de la última tira clickeada.
			keys = set(getattr(self, "cine_crudo_selected_stripes", set()) or set())
			if not keys:
				keys = {str(getattr(self, "cine_crudo_focused_stripe", "") or "ESFUERZO:SA")}
			for key in keys:
				stage_tag, axis = key.split(":", 1) if ":" in key else ("ESFUERZO", "SA")
				starts = (getattr(self, "cine_crudo_stripe_start_by_stage", {}) or {}).setdefault(
					stage_tag, {"SA": 1, "VLA": 1, "HLA": 1}
				)
				cur = int(starts.get(axis, 1) or 1)
				starts[axis] = max(1, cur - step)
			self._schedule_montage_refresh(10, fast=True)
			event.accept()
		except Exception as exc:
			self._log(f"[WARN] Rueda en montaje falló: {exc}")

	def keyPressEvent(self, event):
		# Atajos de teclado para manejo fino de tiras en montaje.
		if self.cine_crudo_preview_mode == "sa_montage":
			key = event.key()
			mods = event.modifiers()
			selection_keys = set(getattr(self, "cine_crudo_selected_stripes", set()) or set())
			if not selection_keys:
				selection_keys = {str(getattr(self, "cine_crudo_focused_stripe", "") or "ESFUERZO:SA")}
			selection_key = str(getattr(self, "cine_crudo_focused_stripe", "") or next(iter(selection_keys)))
			stage_tag, axis = selection_key.split(":", 1) if ":" in selection_key else ("ESFUERZO", "SA")
			step = 3 if bool(mods & Qt.KeyboardModifier.ShiftModifier) else 1
			if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
				for selected_key in selection_keys:
					st, ax = selected_key.split(":", 1) if ":" in selected_key else ("ESFUERZO", "SA")
					starts = (getattr(self, "cine_crudo_stripe_start_by_stage", {}) or {}).setdefault(
						st, {"SA": 1, "VLA": 1, "HLA": 1}
					)
					cur = int(starts.get(ax, 1) or 1)
					starts[ax] = max(1, cur + step if key == Qt.Key.Key_Left else cur - step)
				self._schedule_montage_refresh(8, fast=True)
				self.statusBar().showMessage(f"Montaje: {len(selection_keys)} tira(s) desplazada(s) · foco {stage_tag} {axis}", 1200)
				event.accept()
				return
			if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
				order = ["SA", "VLA", "HLA"]
				idx = order.index(axis) if axis in order else 0
				idx = (idx - 1) % 3 if key == Qt.Key.Key_Up else (idx + 1) % 3
				self.cine_crudo_selected_stripe = order[idx]
				self._schedule_montage_refresh(0)
				self.statusBar().showMessage(f"Montaje: tira activa {self.cine_crudo_selected_stripe}", 1200)
				event.accept()
				return
			if key in (Qt.Key.Key_R, Qt.Key.Key_Home):
				starts[axis] = 1
				self._schedule_montage_refresh(0)
				event.accept()
				return
		super().keyPressEvent(event)

	def _on_cine_crudo_mouse_double_click_safe(self, event, source_label=None):
		if self.cine_crudo_preview_mode != "sa_montage":
			return
		try:
			self.statusBar().showMessage("Montaje: doble click = reset tira activa", 1800)
			# Reset de la tira seleccionada al inicio de ventana.
			selection_key = str(getattr(self, "cine_crudo_focused_stripe", "") or next(iter(getattr(self, "cine_crudo_selected_stripes", set()) or set()), "ESFUERZO:SA"))
			stage_tag, axis = selection_key.split(":", 1) if ":" in selection_key else ("ESFUERZO", "SA")
			starts = (getattr(self, "cine_crudo_stripe_start_by_stage", {}) or {}).setdefault(
				stage_tag, {"SA": 1, "VLA": 1, "HLA": 1}
			)
			starts[axis] = 1
			self._schedule_montage_refresh(0)
			self._log(f"Montaje: reset de tira {stage_tag} {axis} (start=1).")
			event.accept()
		except Exception as exc:
			self._log(f"[WARN] Doble click en montaje falló: {exc}")

	def _schedule_montage_refresh(self, delay_ms: int = 20, fast: bool = False):
		if self.cine_crudo_preview_mode != "sa_montage":
			return
		# Fast-pass: interacción continua (rueda/ventana/drag) rinde a baja resolución
		# y agenda un re-render HQ 512px cuando el usuario suelta (~180ms de idle).
		if fast:
			self._montage_panel_px = int(getattr(self, "_MONTAGE_PANEL_FAST", 256))
			if hasattr(self, "_montage_hq_timer"):
				self._montage_hq_timer.start(180)
		else:
			self._montage_panel_px = 512
			if hasattr(self, "_montage_hq_timer"):
				self._montage_hq_timer.stop()
		if hasattr(self, "_montage_refresh_timer"):
			self._montage_refresh_timer.start(max(0, int(delay_ms)))
		else:
			self._show_cine_crudo_sa_montage()

	def _render_montage_hq(self):
		"""Re-render nítido 512px tras terminar la interacción (fast-pass settle)."""
		if self.cine_crudo_preview_mode != "sa_montage":
			return
		self._montage_panel_px = 512
		self._show_cine_crudo_sa_montage()

	def _show_cine_crudo_montage_tips(self):
		msg = (
			"Montaje SA/VLA/HLA — Tips rápidos\n\n"
			"Mouse en la imagen:\n"
			"• Click simple: selecciona la tira activa (SA/VLA/HLA).\n"
			"• Rueda: mueve solo la tira activa (desplaza cortes visibles).\n"
			"• Doble click: resetea la tira activa (vuelve al inicio).\n"
			"• Ctrl + rueda: zoom parejo en toda la vista (mismo zoom para todos los cortes).\n"
			"• Botón medio arrastrar (o Alt + arrastre): pan para recentrar sin rerender.\n\n"
			"Teclado (montaje activo):\n"
			"• Flecha izquierda/derecha: mueve tira activa.\n"
			"• Shift + izquierda/derecha: paso rápido.\n"
			"• Flecha arriba/abajo: cambia tira activa.\n"
			"• R o Home: reset tira activa.\n\n"
			"Controles de barra:\n"
			"• Layout (Denso/Grande/9/8…): cortes por tira (SA/VLA/HLA).\n"
			"• Frames desde→hasta: rango de gates para sumar/mostrar.\n"
			"• Offsets SA/VLA/HLA: alinea reposo contra esfuerzo.\n"
			"• Con reposo, las filas se intercalan por eje: Esf SA · Rep SA · Esf VLA · Rep VLA · Esf HLA · Rep HLA.\n"
		)
		QMessageBox.information(self, "SINCRO - Tips de montaje", msg)

	def _on_cine_crudo_image_clicked(self, event):
		"""Captura el click del usuario sobre la imagen de cine_crudo para elegir órgano y/o etapa activa."""
		pos = self._cine_crudo_event_to_matrix(event)
		if pos is None:
			return
		ry0, rx0, H_map, W_map, rx_raw = pos
		marker = self._cine_crudo_marker_at_event(event)
		if marker is not None:
			self._cine_crudo_drag_marker = marker
			self._cine_crudo_set_drag_status(marker)
			return
		dual = bool((getattr(self, "_cine_crudo_dual_render_meta", None) or {}).get("enabled"))
		try:
			ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
		except Exception:
			ctrl = False
		if not self.cine_crudo_seed_mode:
			# En modo dual, un click simple selecciona la etapa activa; Ctrl+click selecciona ambas.
			if dual:
				sm = self._cine_crudo_event_stage_and_matrix(event)
				if sm is not None:
					self._set_active_cine_crudo_stage("both" if ctrl else sm[0])
			return
		if self.study is None or bool(getattr(self.study, "reconstructed", True)):
			return
		sm = self._cine_crudo_event_stage_and_matrix(event)
		if sm is None:
			return
		stage, ry, rx, H_stage, W_stage = sm
		_undo_before = None if getattr(self, "_undo_suspended", False) else self._snapshot_attrs(self.UNDO_ATTRS_MOTION)
		self.cine_crudo_seed_mode = False
		# Despresionar el botón SIN emitir toggled(): de lo contrario
		# _on_cine_crudo_seed_mode_toggled(False) borraría el seed recién fijado.
		self.cine_crudo_seed_btn.blockSignals(True)
		self.cine_crudo_seed_btn.setChecked(False)
		self.cine_crudo_seed_btn.blockSignals(False)
		if stage == "rest":
			self.cine_crudo_seed_compare = (ry, rx)
		else:
			self.cine_crudo_seed = (ry, rx)  # (y, x) en coordenadas de matriz
			if self.cine_crudo_compare_line_y is None:
				self.cine_crudo_compare_line_y = float(ry)
			if hasattr(self, "cine_crudo_roi_mode_combo") and "banda" in str(self.cine_crudo_roi_mode_combo.currentText()).lower():
				r = float(self.cine_crudo_roi_spin.value()) if hasattr(self, "cine_crudo_roi_spin") else 8.0
				self.cine_crudo_band_upper = float(np.clip(ry - r, 0, H_stage - 1))
				self.cine_crudo_band_lower = float(np.clip(ry + r, 0, H_stage - 1))
		self._cine_crudo_set_drag_status(None)
		etapa_txt = "Rest (reposo)" if stage == "rest" else "Stress (esfuerzo)"
		self._log(f"Corazón fijado en {etapa_txt}: (y={ry:.1f}, x={rx:.1f}) — el tracking seguirá esa componente.")
		self._set_active_cine_crudo_stage("both" if ctrl else stage)
		self._commit_undo("Fijar corazón (motion)", self.UNDO_ATTRS_MOTION, _undo_before)
		self._mark_step_done("crudo")
		self._mark_step_done("motion", getattr(self, "cine_crudo_seed", None), getattr(self, "cine_crudo_band_upper", None), getattr(self, "cine_crudo_band_lower", None))

	def _on_cine_crudo_image_dragged(self, event):
		if self._cine_crudo_drag_marker not in ("upper", "lower", "compare_line"):
			self._cine_crudo_set_drag_status(self._cine_crudo_marker_at_event(event))
			return
		self._cine_crudo_set_drag_status(self._cine_crudo_drag_marker)
		pos = self._cine_crudo_event_to_matrix(event)
		if pos is None:
			return
		ry, _rx, H, _W, _rx_raw = pos
		margin = 2.0
		if self._cine_crudo_drag_marker == "compare_line":
			self.cine_crudo_compare_line_y = float(np.clip(ry, 0, H - 1.0))
			now = perf_counter()
			if now - float(getattr(self, "_cine_crudo_last_drag_refresh", 0.0)) > 0.06:
				self._cine_crudo_last_drag_refresh = now
				self._refresh_cine_crudo_view()
			return
		if self._cine_crudo_drag_marker == "upper":
			lower = self.cine_crudo_band_lower
			if lower is None and self.cine_crudo_seed is not None:
				lower = float(self.cine_crudo_seed[0]) + float(self.cine_crudo_roi_spin.value())
			limit = float(lower) - margin if lower is not None else H - 1
			self.cine_crudo_band_upper = float(np.clip(ry, 0, max(0.0, limit)))
		else:
			upper = self.cine_crudo_band_upper
			if upper is None and self.cine_crudo_seed is not None:
				upper = float(self.cine_crudo_seed[0]) - float(self.cine_crudo_roi_spin.value())
			limit = float(upper) + margin if upper is not None else 0.0
			self.cine_crudo_band_lower = float(np.clip(ry, min(H - 1.0, limit), H - 1.0))
		bounds = self._cine_crudo_band_bounds(H)
		if bounds is not None and hasattr(self, "cine_crudo_roi_spin"):
			y0, y1 = bounds
			self.cine_crudo_roi_spin.blockSignals(True)
			self.cine_crudo_roi_spin.setValue(int(round(max(1.0, 0.5 * (y1 - y0)))))
			self.cine_crudo_roi_spin.blockSignals(False)
		now = perf_counter()
		if now - float(getattr(self, "_cine_crudo_last_drag_refresh", 0.0)) > 0.06:
			self._cine_crudo_last_drag_refresh = now
			self._refresh_cine_crudo_view()

	def _on_cine_crudo_image_released(self, event):
		if self._cine_crudo_drag_marker in ("upper", "lower"):
			upper = float(self.cine_crudo_band_upper) if self.cine_crudo_band_upper is not None else float("nan")
			lower = float(self.cine_crudo_band_lower) if self.cine_crudo_band_lower is not None else float("nan")
			self._log(
				f"Markers Banda Y ajustados: upper={upper:.1f}, "
				f"lower={lower:.1f}."
			)
		elif self._cine_crudo_drag_marker == "compare_line":
			line_y = float(self.cine_crudo_compare_line_y) if self.cine_crudo_compare_line_y is not None else float("nan")
			self._log(f"Línea de referencia ajustada: y={line_y:.1f}.")
		self._cine_crudo_drag_marker = None
		self._cine_crudo_set_drag_status(self._cine_crudo_marker_at_event(event))
		self._refresh_cine_crudo_view()

	def _on_cine_crudo_source_changed(self, source: str):
		# Cambiar fuente debe devolver siempre la interacción completa de motion
		# correction (máscara, elegir corazón, banda y línea de referencia).
		self.cine_crudo_preview_mode = None
		self._cine_crudo_drag_marker = None
		self._montage_drag_axis = None
		self._montage_drag_mode = None
		self._montage_drag_start_x = None
		self._load_cine_crudo_frames(str(source))
		self._log(f"Cine crudo: fuente {source} ({len(self.cine_crudo_frames)} frames).")

	def _on_scatter_preview_changed(self, *_args):
		"""Refresca el cine crudo al cambiar el checkbox o k de scatter (preview EM−SC)."""
		if self.study is None or bool(getattr(self.study, "reconstructed", True)):
			return
		# Solo refrescar si estamos en modo cine crudo (no recon/montaje).
		if str(getattr(self, "cine_crudo_preview_mode", None)) in ("recon_qc", "sa_montage", "generated_cuts"):
			return
		source = str(self.cine_crudo_source_combo.currentText()) if hasattr(self, "cine_crudo_source_combo") else "UngGat"
		self._load_cine_crudo_frames(source)
		on = bool(getattr(self, "cine_crudo_scatter_check", None) is not None and self.cine_crudo_scatter_check.isChecked())
		k = float(self.cine_crudo_scatter_k_spin.value()) if hasattr(self, "cine_crudo_scatter_k_spin") else 1.0
		self._log(f"Preview scatter: {'EM−%.2f×SC' % k if on else 'EM solo'}.")

	def _toggle_lower_cine_band(self):
		"""Colapsa/expande la banda inferior dejando solo su header delgado."""
		splitter = getattr(self, "right_splitter", None)
		if splitter is None:
			return
		self._lower_cine_collapsed = not bool(getattr(self, "_lower_cine_collapsed", False))
		if self._lower_cine_collapsed:
			self._right_splitter_saved_sizes = splitter.sizes()
			self.bottom_hsplit.setVisible(False)
			self.lower_cine_collapse_btn.setText("▸")
			total = sum(splitter.sizes())
			header_h = max(28, self._lower_cine_panel.sizeHint().height())
			splitter.setSizes([max(0, total - header_h), header_h])
		else:
			self.bottom_hsplit.setVisible(True)
			self.lower_cine_collapse_btn.setText("▾")
			if self._right_splitter_saved_sizes:
				splitter.setSizes(self._right_splitter_saved_sizes)

	def _rebuild_tabs_for_mode(self):
		current_title = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() > 0 else ""
		while self.tabs.count() > 0:
			self.tabs.removeTab(0)
		# Opción A: todas las pestañas siempre visibles; las pesadas se renderizan
		# perezosamente la primera vez que se entra en ellas (_request_lazy_tab_render).
		order = list(self._basic_tab_order)
		order.extend(self._advanced_extra_tab_order)
		for name in order:
			# Si el panel cine_crudo está reubicado en la ventana de Preparación,
			# no lo agregamos como pestaña (vive allá hasta que se cierre esa ventana).
			if name == "cine_crudo" and getattr(self, "_cine_crudo_reparented", False):
				continue
			widget = self._tab_widgets.get(name)
			if widget is None:
				continue
			title = self._tab_titles.get(name, name)
			tip = self._tab_tooltips.get(name, "")
			self.tabs.addTab(widget, title)
			self.tabs.setTabToolTip(self.tabs.count() - 1, tip)
		if current_title:
			for i in range(self.tabs.count()):
				if self.tabs.tabText(i) == current_title:
					self.tabs.setCurrentIndex(i)
					break

	def toggle_advanced_mode(self):
		# Opción A: el modo avanzado se eliminó. Las pestañas están siempre visibles
		# y las pesadas se renderizan bajo demanda al entrar en ellas. Este método se
		# conserva como no-op por compatibilidad con llamadas históricas.
		return

	def _is_tab_active(self, title: str) -> bool:
		idx = int(self.tabs.currentIndex()) if self.tabs is not None else -1
		return idx >= 0 and self.tabs.tabText(idx) == str(title)

	def _on_compare_axes_cine_toggled(self, checked: bool):
		self.compare_axes_playing = False
		self.compare_axes_cine_timer.stop()
		if not checked:
			self._load_compare_axes_preview()
			return
		self._schedule_compare_axes_refresh()

	def _on_compare_axes_cine_speed_changed(self, value: int):
		self.compare_axes_cine_timer.setInterval(max(40, int(value)))
		if self.compare_axes_cine_check.isChecked() and self.compare_axes_preview_frames:
			self._update_compare_axes_toggle_text(enabled=True)

	def _update_compare_axes_toggle_text(self, enabled: bool = True):
		if self.compare_axes_cine_toggle_btn is None:
			return
		self.compare_axes_cine_toggle_btn.setEnabled(enabled)
		if not enabled:
			self.compare_axes_cine_toggle_btn.setText("▶ ⏸")
			return
		if not self.compare_axes_preview_frames:
			self.compare_axes_cine_toggle_btn.setText("▶ ⏸")
			return
		if self.compare_axes_playing:
			self.compare_axes_cine_toggle_btn.setText("⏸")
		else:
			self.compare_axes_cine_toggle_btn.setText("▶")

	def resizeEvent(self, event):
		super().resizeEvent(event)
		for name in list(self.preview_labels.keys()):
			if name in self.preview_pixmaps:
				self._apply_preview_zoom(name)

	def show_audit_validation_help(self):
		doc_path = os.path.join(
			os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
			"docs",
			"GUIA_AUDITORIA_VALIDACION_CALCULOS.md",
		)
		summary = (
			"Guía rápida para auditoría y validación:\n\n"
			"1) Segmentación manual usa ROI por slice (centro, radio interno y externo).\n"
			"2) En apex/base sin cavidad visible, se permite r_inner='-' (sin interno).\n"
			"3) Eso impacta en volúmenes: puede subir volumen miocárdico y bajar volumen de cavidad en esos slices.\n"
			"4) La FE preliminar excluye slices sin radio interno finito, para evitar sesgo por cavidad no visible.\n"
			"5) Métricas de fase y mapa AHA se calculan sobre la máscara segmentada final.\n\n"
			"Recomendación práctica:\n"
			"- Apex/base sin cavidad: usar sin interno solo cuando no hay luz ventricular distinguible.\n"
			"- Slice medio con cavidad visible: usar anillo completo (interno y externo)."
		)
		msg = QMessageBox(self)
		msg.setWindowTitle("SINCRO - Auditoría y validación")
		msg.setIcon(QMessageBox.Icon.Information)
		msg.setText(summary)
		msg.setStandardButtons(QMessageBox.StandardButton.Ok)
		open_doc_btn = msg.addButton("Abrir guía técnica", QMessageBox.ButtonRole.ActionRole)
		msg.exec()
		if msg.clickedButton() is open_doc_btn:
			if os.path.exists(doc_path):
				QDesktopServices.openUrl(QUrl.fromLocalFile(doc_path))
			else:
				QMessageBox.information(self, "SINCRO", "No se encontró la guía técnica en docs.")

	def show_polar_technical_help(self):
		summary = (
			"Guía técnica rápida de mapas polares y sincronía:\n\n"
			"1) polar_map: distribución de fase AHA (17 segmentos), útil para patrón regional de disincronía.\n"
			"2) polar_clinico: panel estilo estación (histograma+bullseye) con PSD/PHB para lectura rápida.\n"
			"3) polar_map_Δsigned: Δ circular (esfuerzo-reposo), conserva dirección (adelanto/atraso relativo).\n"
			"4) polar_map_Δabs: magnitud |Δ| sin dirección, útil para localizar hotspots dinámicos.\n"
			"5) polar_perfusion_directa: perfusión polar continua (apex centro, base borde) + cine gate-a-gate integrado con operación esfuerzo/reposo.\n"
			"6) bullseye_directo: resumen segmentario AHA rápido de intensidad regional.\n\n"
			"Fórmulas clave:\n"
			"• Δsigned = ((φ_esfuerzo - φ_reposo + 180) mod 360) - 180\n"
			"• Δabs = |Δsigned|\n"
			"• PSD/BW/Entropy: a mayor dispersión, mayor asincronía probable; comparar contra DB del software correspondiente.\n\n"
			"Checklist de interpretación segura (uso recomendado):\n"
			"• Integrar fase + perfusión + cine + clínica, nunca un único mapa aislado.\n"
			"• Estratificar por QRS/morfología (BRI/BRD/IVCD) antes de sugerir respuesta a TRC.\n"
			"• Si calidad de segmentación/gating es dudosa, etiquetar resultado como orientativo.\n"
			"• Diferenciar hallazgo mecánico de recomendación terapéutica final.\n\n"
			"Fuentes y evidencia:\n"
			"• Priorizar guías/revisiones independientes (PMC/SAC/CONAREC).\n"
			"• Material de fabricantes puede ayudar en UI/flujo, pero no debe limitar criterios clínicos.\n\n"
			"Rangos orientativos reportados:\n"
			"• Phase SD: 11-14°\n"
			"• Bandwidth: 42-49°\n"
			"• Entropy: reportar Shannon en bits y normalizada en % cuando se compare con literatura.\n\n"
			"Nota: valores orientativos, siempre integrar con perfusión, cine, clínica y validación local."
		)
		QMessageBox.information(self, "SINCRO - Help técnico mapas polares", summary)

	def show_crt_implementation_plan(self):
		summary = (
			"Plan de implementación priorizado (rápido y clínicamente sólido):\n\n"
			"Prioridad ALTA (hacer primero):\n"
			"1) Capa de estratificación eléctrica (QRS y morfología) en resumen clínico y PDF.\n"
			"2) Banderas de calidad del estudio (segmentación, gating, artefacto) con warning visible.\n"
			"3) Separar en reporte: hallazgo mecánico vs sugerencia clínica vs advertencia de uso.\n\n"
			"Prioridad MEDIA:\n"
			"4) Checklist de no respondedor TRC (carga BiV, AV/VV, FA/EV, cicatriz).\n"
			"5) Export estructurado para seguimiento longitudinal y auditoría.\n\n"
			"Para acelerar proceso YA (impacto directo):\n"
			"• Mantener generación pesada on-demand (PDF/cines) y sólo en pestaña activa.\n"
			"• Cachear renders por combinación de parámetros (evita recomputar figuras idénticas).\n"
			"• Reprocesar incremental: si cambió zoom/rotación, no recalcular fase/segmentación.\n"
			"• Añadir pre-ajustes clínicos (rápido/calidad) con un clic para reducir ajustes manuales."
		)
		QMessageBox.information(self, "SINCRO - Plan implementación CRT", summary)

	def load_compare_study(self):
		"""Carga un segundo estudio gated (típicamente REST) y calcula sus métricas
		de fase para comparar disincronía contra el estudio actual (típicamente
		STRESS). Base clínica: Camilletti/Erriest 2015 (Hospital Italiano La Plata):
		la isquemia post-stress produce disincronía transitoria por stunning, que se
		manifiesta como aumento de BW/PSD en stress respecto de rest.
		"""
		if self.study is None or self.metrics is None:
			QMessageBox.warning(self, "SINCRO", "Primero procesá el estudio actual (STRESS).")
			return
		if self._check_unsaved_study():
			return
		paths = self._select_dicom_paths(
			title="Seleccionar estudio de comparación (ej: REST)",
			allow_multiple=False,
		)
		if not paths:
			return
		self._load_compare_study_from_path(paths[0])

	def load_one_or_two_studies(self):
		# Ofrecer carga inteligente por carpeta (EM+ATT+CT+SC de ambas etapas) o
		# la selección manual de 1/2 archivos de siempre.
		box = QMessageBox(self)
		box.setWindowTitle("SINCRO — Cargar estudios")
		box.setIcon(QMessageBox.Icon.Question)
		box.setText("¿Cómo querés cargar?")
		box.setInformativeText(
			"• Carpeta inteligente: marcás una carpeta y lee/carga todo lo que haya "
			"(Esfuerzo y Reposo, con sus EM, ATT, CT y Scatter).\n"
			"• Elegir archivos: seleccionás 1 o 2 estudios a mano."
		)
		btn_smart = box.addButton("🔍 Carpeta inteligente", QMessageBox.ButtonRole.AcceptRole)
		btn_files = box.addButton("Elegir archivos", QMessageBox.ButtonRole.ActionRole)
		box.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
		box.setDefaultButton(btn_smart)
		box.exec()
		clicked = box.clickedButton()
		if clicked is btn_smart:
			self._smart_load_ct_att_folder()
			return
		if clicked is not btn_files:
			return
		paths = self._select_dicom_paths(
			title="Seleccionar uno o dos estudios (stress/rest)",
			allow_multiple=True,
		)
		if not paths:
			return
		valid_paths = [p for p in paths if os.path.exists(p)]
		if not valid_paths:
			QMessageBox.warning(self, "SINCRO", "No se seleccionaron archivos válidos.")
			return
		if len(valid_paths) > 2:
			QMessageBox.information(self, "SINCRO", "Se usarán solo los primeros 2 archivos seleccionados.")
			valid_paths = valid_paths[:2]

		if len(valid_paths) == 1:
			self.file_edit.setText(valid_paths[0])
			self.process_current()
			return

		def _stage_from_metadata(path_text: str) -> str | None:
			"""Lee metadata DICOM liviana (sin pixel data) para inferir stress/rest."""
			try:
				import pydicom
				ds = pydicom.dcmread(path_text, stop_before_pixels=True, force=True)
				parts = []
				for tag in ("SeriesDescription", "StudyDescription", "ProtocolName"):
					val = getattr(ds, tag, None)
					if val:
						parts.append(str(val))
				it = getattr(ds, "ImageType", None)
				if it:
					parts.append(" ".join(str(x) for x in it))
				text = " ".join(parts).lower()
				if not text:
					return None
				stress_kw = ("stress", "esfuerzo", "ejercicio", "exercise", "dipirid", "dipyrid", "adenos", "dobutam", "regaden", "persantin")
				rest_kw = ("rest", "reposo", "basal", "resting")
				has_stress = any(k in text for k in stress_kw)
				has_rest = any(k in text for k in rest_kw)
				if has_stress and not has_rest:
					return "stress"
				if has_rest and not has_stress:
					return "rest"
			except Exception:
				pass
			return None

		def _score_stress(path_text: str) -> int:
			# Prioridad 1: metadata DICOM (SeriesDescription/ProtocolName/ImageType).
			meta_stage = _stage_from_metadata(path_text)
			if meta_stage == "stress":
				return 10
			if meta_stage == "rest":
				return -10
			# Fallback: nombre de archivo.
			u = os.path.basename(path_text).upper()
			score = 0
			if "STRESS" in u:
				score += 2
			if "REST" in u:
				score -= 1
			return score

		meta_1 = _stage_from_metadata(valid_paths[0])
		meta_2 = _stage_from_metadata(valid_paths[1])
		if meta_1 or meta_2:
			self._log(
				f"Etapas por metadata DICOM: {os.path.basename(valid_paths[0])} → {meta_1 or 'indeterminada'} | "
				f"{os.path.basename(valid_paths[1])} → {meta_2 or 'indeterminada'}. "
				"Esfuerzo se carga como primario (arriba)."
			)
		primary_path = max(valid_paths, key=_score_stress)
		compare_path = valid_paths[0] if valid_paths[1] == primary_path else valid_paths[1]
		self.file_edit.setText(primary_path)
		self.process_current()
		if self.study is None:
			return
		if not bool(getattr(self.study, "reconstructed", True)):
			self._load_compare_raw_study_from_path(compare_path)
			return
		if self.metrics is not None:
			self._load_compare_study_from_path(compare_path)

	def _load_compare_raw_study_from_path(self, path: str):
		"""Carga un segundo estudio CRUDO para edición/comparación en cine, sin pipeline de fase."""
		try:
			comp_study = dicom_loader.load(path, verbose=False)
			if bool(getattr(comp_study, "reconstructed", True)):
				self._log("Comparación cruda: el segundo estudio parece reconstruido; usando flujo estándar de comparación.")
				self._load_compare_study_from_path(path)
				return
			if not self._check_second_stage_patient(comp_study):
				return
			if not self._check_stage_zoom_consistency(comp_study):
				return
			self.compare_raw_study = comp_study
			self.compare_raw_path = str(path)
			self.cine_crudo_corrected_projections_compare = None
			self.cine_crudo_motion_result_compare = None
			self.cine_crudo_ref_index_compare = None
			self.compare_bundle = None
			self.compare_metrics = None
			self.compare_ef = None
			self.compare_label = os.path.splitext(os.path.basename(path))[0]
			self.dual_mode_active = True
			self._refresh_cine_source_selector()
			self._apply_cine_source("primary", preserve_position=True)
			# Con dos etapas crudas cargadas, arrancar con el selector en "Ambas":
			# las herramientas (corrección/recon/reorient/cortes) procesan las dos.
			self._set_active_cine_crudo_stage("both", refresh_view=False)
			# Al cargar la segunda etapa cruda, refrescar de inmediato la pestaña
			# superior para que muestre Stress/Rest apilados.
			self._refresh_cine_crudo_view()
			self._set_progress(100, "Crudo dual listo")
			self.statusBar().showMessage(f"Comparación cruda cargada: {self.compare_label}")
			self._log("Dual crudo: segunda etapa cargada en cine secundario para edición desde cero.")
			# Pista de etapa por metadata (solo informativa; el selector manda).
			st_top = self._cine_crudo_stage_display(self.study)
			st_bot = self._cine_crudo_stage_display(comp_study)
			if st_top or st_bot:
				self._log(f"Etapa detectada por metadata → arriba: {st_top or 'indeterminada'} · abajo: {st_bot or 'indeterminada'} (el selector Etapa sigue mandando).")
			self._refresh_preparacion_window()
		except Exception as exc:
			self._log(f"[ERROR compare raw] {exc}")
			QMessageBox.critical(self, "Error de comparación cruda", str(exc))

	def _process_secondary_bundle(self, path: str, preloaded_study=None) -> dict:
		comp_study = preloaded_study if preloaded_study is not None else dicom_loader.load(path, verbose=False)
		comp_axis = {}
		if path:
			try:
				comp_axis = self._load_axis_companions(path)
			except Exception:
				comp_axis = {}
		comp_cube_corrected, _ = self._apply_gate_dropout_correction(comp_study.cube, "comparación")
		comp_cube_corrected, comp_intestinal_info = self._apply_intestinal_subtraction_to_cube(
			comp_cube_corrected, self.cine_compare
		)
		self._log_intestinal_subtraction(comp_intestinal_info)
		comp_cube_for_segmentation = self._apply_intestinal_mask_to_cube(comp_cube_corrected, self.cine_compare, require_global_visual=False)
		comp_cube_for_analysis = comp_cube_corrected
		# Pasajero de fase de la 2da etapa: mismas correcciones que el visible.
		comp_cube_phase = getattr(comp_study, "cube_phase", None)
		if comp_cube_phase is not None:
			try:
				comp_phase_base = np.asarray(comp_cube_phase, dtype=np.float64)
				if comp_phase_base.shape == np.asarray(comp_study.cube).shape:
					comp_phase_corr, _ = self._apply_gate_dropout_correction(
						comp_phase_base, "comparación fase (pasajero FBP)", log=False
					)
					comp_phase_corr, _ = self._apply_intestinal_subtraction_to_cube(
						comp_phase_corr, self.cine_compare
					)
					comp_cube_for_analysis = comp_phase_corr
					self._log("Fase de la 2da etapa calculada sobre pasajero FBP (cube_phase).")
			except Exception:
				pass
		seg_method = "auto"
		manual_rois = None
		parsed_compare_rois = self._parse_manual_rois_text(self.compare_manual_rois_text)
		valid_compare_rois = {
			slice_index: roi
			for slice_index, roi in parsed_compare_rois.items()
			if self._is_roi_valid_for_manual(roi)
		}
		if str(self.seg_method.currentText()) == "manual" and valid_compare_rois:
			seg_method = "manual"
			manual_rois = valid_compare_rois
		comp_seg = segment_myocardium(
			comp_cube_for_segmentation,
			method=seg_method,
			threshold_frac=float(self.threshold_spin.value()),
			smooth_sigma=float(self.sigma_spin.value()),
			manual_rois=manual_rois,
		)
		comp_phase = phase_analysis(
			comp_cube_for_analysis,
			comp_seg.mask,
			harmonics=int(self.harmonics_spin.value()),
			amplitude_threshold_frac=float(self.phase_threshold_spin.value()),
			normalize_reference=self.normalize_check.isChecked(),
		)
		comp_phase_raw = phase_analysis(
			comp_cube_for_analysis,
			comp_seg.mask,
			harmonics=int(self.harmonics_spin.value()),
			amplitude_threshold_frac=float(RAW_PHASE_QC_AMP_FILTER),
			normalize_reference=self.normalize_check.isChecked(),
		)
		comp_metrics_raw = self._annotate_phase_metrics(
			calculate_phase_metrics(comp_phase_raw.phases_deg),
			comp_phase_raw,
			RAW_PHASE_QC_AMP_FILTER,
			"crudo ROI",
		)
		comp_metrics = self._annotate_phase_metrics(
			calculate_phase_metrics(comp_phase.phases_deg),
			comp_phase,
			float(self.phase_threshold_spin.value()),
			"clínico robusto",
		)
		comp_phase_qc = self._build_phase_qc(comp_phase_raw, comp_phase, comp_metrics_raw, comp_metrics)
		comp_aha = map_to_17_segments(comp_seg)
		comp_phase_by_seg = phase_by_segment(comp_phase.phase_map, comp_aha)
		comp_territory = territory_analysis(comp_phase_by_seg)
		comp_ef = self._estimate_ef_for(comp_study, comp_seg)
		return {
			"path": path,
			"label": os.path.splitext(os.path.basename(path))[0],
			"study": comp_study,
			"axis_companions": comp_axis,
			"seg": comp_seg,
			"phase_result": comp_phase,
			"phase_result_raw": comp_phase_raw,
			"metrics": comp_metrics,
			"metrics_raw": comp_metrics_raw,
			"phase_qc": comp_phase_qc,
			"aha": comp_aha,
			"phase_by_seg": comp_phase_by_seg,
			"territory": comp_territory,
			"ef": comp_ef,
			"manual_rois_text": self.compare_manual_rois_text,
		}

	def _load_compare_bundle_from_stage_memory(self, stage: str) -> bool:
		"""Carga comparación procesada desde una etapa dual ya preparada en memoria."""
		stage_key = "rest" if str(stage) == "rest" else "stress"
		stage_state = self._dual_session().stage(stage_key)
		mem_study = stage_state.cut_study
		if mem_study is None or self.study is None or self.metrics is None:
			return False
		if mem_study is self.study:
			return False
		try:
			n_g = int(np.asarray(mem_study.cube).shape[0])
			if n_g < 3:
				self._log(f"[DUAL] Comparación desde memoria omitida: etapa {stage_key} sin gating suficiente (<3).")
				return False
		except Exception:
			pass

		stage_label = "Reposo" if stage_key == "rest" else "Esfuerzo"
		pseudo_path = str(stage_state.source_path or stage_state.cut_source_label or f"{stage_key}_memory")
		self._set_progress(78, f"Procesando comparación desde {stage_label} (memoria)...")
		bundle = self._process_secondary_bundle(pseudo_path, preloaded_study=mem_study)
		bundle["path"] = str(stage_state.source_path or "")
		bundle["label"] = str(stage_state.label or stage_label)
		# Persistir SIEMPRE el análisis secundario en el estado canónico de la
		# etapa. Antes solo vivía en compare_bundle; si el render diferido lo
		# reemplazaba/limpiaba, Resultados en vivo y la grilla 2×2 quedaban vacíos.
		stage_state.seg = bundle["seg"]
		stage_state.phase = bundle["phase_result"]
		stage_state.metrics = bundle["metrics"]
		stage_state.metrics_raw = bundle.get("metrics_raw")
		stage_state.phase_by_seg = bundle["phase_by_seg"]
		stage_state.territory = bundle["territory"]
		stage_state.ef = bundle["ef"]
		# Persistir el análisis secundario en el StageState canónico ANTES de
		# instalar/renderizar compare_bundle. El bundle puede regenerarse o quedar
		# temporalmente ausente durante el render HQ, pero Resultados en vivo y la
		# grilla 2×2 siempre deben conservar Reposo.
		stage_state.seg = bundle.get("seg")
		stage_state.phase = bundle.get("phase_result")
		stage_state.metrics = bundle.get("metrics")
		stage_state.metrics_raw = bundle.get("metrics_raw")
		stage_state.phase_by_seg = bundle.get("phase_by_seg")
		stage_state.territory = bundle.get("territory")
		stage_state.ef = bundle.get("ef")

		self.compare_bundle = bundle
		self.compare_metrics = bundle["metrics"]
		self.compare_ef = bundle["ef"]
		self.compare_label = bundle["label"]
		# NO tocar compare_raw_study: es property → rest.raw_study en DualSession.
		# Anularlo dejaba el reposo sin crudo (selector bloqueado, re-recon imposible).
		self.compare_raw_path = str(stage_state.source_path or "")
		self._refresh_cine_source_selector()
		self._apply_cine_source("primary", preserve_position=True)

		left_label, right_label = self._dual_compare_labels()
		if bool(self.realtime_deferred_render_check.isChecked()):
			prev_fast = bool(self.compare_interactive_fast_mode)
			self.compare_interactive_fast_mode = True
			try:
				self._write_compare_stress_rest()
			finally:
				self.compare_interactive_fast_mode = prev_fast
			self.dual_mode_active = True
			self._load_previews_selected(self._default_preview_tabs())
			self._refresh_summary()
			self._select_tab_by_title("histograma")
			self._schedule_deferred_hq_render(
				"compare",
				delay_ms=320,
				compare_bundle=bundle,
				left_label=left_label,
				right_label=right_label,
			)
			self._set_progress(92, f"Comparación rápida lista ({stage_label} memoria); HQ diferido...")
			self.statusBar().showMessage(f"Comparación rápida cargada desde {stage_label} (memoria)")
			self._log(f"[DUAL] Comparación rápida cargada desde {stage_label} en memoria; HQ diferido en progreso.")
			return True

		self._set_progress(82, f"Generando comparación HQ desde {stage_label} (memoria)...")
		self._run_compare_hq_pipeline(bundle, left_label=left_label, right_label=right_label, deferred=False)
		self._set_progress(100, "Comparación lista")
		self._log(f"[DUAL] Comparación HQ cargada desde {stage_label} en memoria.")
		return True

	def _write_outputs_for_bundle(self, bundle: dict, target_dir: str, target_tabs: set[str] | None = None):
		os.makedirs(target_dir, exist_ok=True)
		saved_output_dir = self.output_dir
		saved_study = self.study
		saved_axis = self.axis_companions
		saved_seg = self.seg
		saved_phase = self.phase_result
		saved_phase_raw = self.phase_result_raw
		saved_metrics = self.metrics
		saved_metrics_raw = self.metrics_raw
		saved_phase_qc = self.phase_qc
		saved_aha = self.aha
		saved_phase_by_seg = self.phase_by_seg
		saved_territory = self.territory
		saved_compare_bundle = self.compare_bundle
		saved_output_path_override = getattr(self, "_output_study_path_override", None)
		saved_output_cine_override = getattr(self, "_output_cine_widget_override", None)
		try:
			self.output_dir = target_dir
			self.study = bundle["study"]
			self.axis_companions = bundle["axis_companions"]
			self.seg = bundle["seg"]
			self.phase_result = bundle["phase_result"]
			self.phase_result_raw = bundle.get("phase_result_raw")
			self.metrics = bundle["metrics"]
			self.metrics_raw = bundle.get("metrics_raw")
			self.phase_qc = bundle.get("phase_qc")
			self.aha = bundle["aha"]
			self.phase_by_seg = bundle["phase_by_seg"]
			self.territory = bundle["territory"]
			# Al renderizar el bundle secundario, evitar compararlo consigo mismo.
			self.compare_bundle = None
			self._output_study_path_override = str(bundle.get("path", ""))
			self._output_cine_widget_override = self.cine_compare
			self._write_outputs(target_tabs=target_tabs)
		finally:
			self.output_dir = saved_output_dir
			self.study = saved_study
			self.axis_companions = saved_axis
			self.seg = saved_seg
			self.phase_result = saved_phase
			self.phase_result_raw = saved_phase_raw
			self.metrics = saved_metrics
			self.metrics_raw = saved_metrics_raw
			self.phase_qc = saved_phase_qc
			self.aha = saved_aha
			self.phase_by_seg = saved_phase_by_seg
			self.territory = saved_territory
			self.compare_bundle = saved_compare_bundle
			self._output_study_path_override = saved_output_path_override
			self._output_cine_widget_override = saved_output_cine_override

	def _compose_dual_tab_images(self, left_label: str, right_label: str, target_tabs: set[str] | None = None):
		import matplotlib.pyplot as plt

		names = list(self.preview_labels.keys())
		if target_tabs is not None:
			names = [n for n in names if n in set(target_tabs)]
		for name in names:
			if name in ("comparacion_stress_rest", "comparacion_ejes", "guia_fase_vi"):
				continue
			left_path = os.path.join(self.output_dir, f"{name}.png")
			right_path = os.path.join(self.compare_output_dir, f"{name}.png")
			if not (os.path.exists(left_path) and os.path.exists(right_path)):
				continue
			try:
				left_img = plt.imread(left_path)
				right_img = plt.imread(right_path)
			except Exception:
				continue
			fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f172a")
			for ax, img, title in (
				(axes[0], left_img, left_label),
				(axes[1], right_img, right_label),
			):
				ax.imshow(img)
				ax.set_xticks([])
				ax.set_yticks([])
				ax.set_title(title, color="#e2e8f0", fontsize=11, fontweight="bold")
			fig.suptitle(f"Comparativa {name} — {self._patient_banner_text(include_stage=False)}", color="#f8fafc", fontsize=12, fontweight="bold")
			fig.tight_layout(rect=(0, 0, 1, 0.95))
			fig.savefig(left_path, dpi=150, bbox_inches="tight")
			plt.close(fig)

	def _compose_dual_polar_cine_gif(self):
		left_gif = os.path.join(self.output_dir, "polar_cine.gif")
		right_gif = os.path.join(self.compare_output_dir, "polar_cine.gif")
		if not (os.path.exists(left_gif) and os.path.exists(right_gif)):
			return
		try:
			from PIL import Image, ImageSequence
		except Exception:
			return
		try:
			with Image.open(left_gif) as left_im, Image.open(right_gif) as right_im:
				left_frames = [frm.convert("RGB") for frm in ImageSequence.Iterator(left_im)]
				right_frames = [frm.convert("RGB") for frm in ImageSequence.Iterator(right_im)]
				if not left_frames or not right_frames:
					return
				count = min(len(left_frames), len(right_frames))
				composed = []
				for idx in range(count):
					lf = left_frames[idx]
					rf = right_frames[idx]
					h = max(lf.height, rf.height)
					resampling_bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
					if lf.height != h:
						new_w = max(1, int(round(lf.width * (h / max(1, lf.height)))))
						lf = lf.resize((new_w, h), resampling_bilinear)
					if rf.height != h:
						new_w = max(1, int(round(rf.width * (h / max(1, rf.height)))))
						rf = rf.resize((new_w, h), resampling_bilinear)
					gap = Image.new("RGB", (28, h), color=(4, 7, 15))
					canvas = Image.new("RGB", (lf.width + gap.width + rf.width, h), color=(4, 7, 15))
					canvas.paste(lf, (0, 0))
					canvas.paste(gap, (lf.width, 0))
					canvas.paste(rf, (lf.width + gap.width, 0))
					composed.append(canvas)
				duration_ms = int(left_im.info.get("duration", int(self.polar_cine_speed_spin.value())))
				composed[0].save(
					left_gif,
					save_all=True,
					append_images=composed[1:],
					duration=duration_ms,
					loop=0,
					disposal=2,
					optimize=False,
				)
		except Exception as exc:
			self._log(f"[WARN] No se pudo componer polar cine comparativo: {exc}")

	def _clear_compare_state(self):
		self.compare_metrics = None
		self.compare_label = None
		self.compare_ef = None
		self.compare_bundle = None
		self.compare_raw_study = None
		self.compare_raw_path = ""
		self.cine_crudo_corrected_projections_compare = None
		self.cine_crudo_motion_result_compare = None
		self.cine_crudo_ref_index_compare = None
		self.compare_manual_rois_text = ""
		self.dual_mode_active = False
		self.active_cine_source = "primary"
		self._refresh_cine_source_selector()
		cmp_path = os.path.join(self.output_dir, "comparacion_stress_rest.png")
		if os.path.exists(cmp_path):
			try:
				os.remove(cmp_path)
			except OSError:
				pass
		self._invalidate_output_cache()

	def _run_compare_hq_pipeline(self, bundle: dict, *, left_label: str, right_label: str, deferred: bool = False, target_tabs: set[str] | None = None):
		self._write_outputs_for_bundle(bundle, self.compare_output_dir, target_tabs=target_tabs)
		# Re-generar salidas del estudio principal con compare_bundle activo para
		# crear mapas delta (polar_map_Δsigned / polar_map_Δabs).
		self._write_outputs(target_tabs=target_tabs)
		self._compose_dual_tab_images(left_label, right_label, target_tabs=target_tabs)
		# polar_cine ya se genera compuesto dentro de _write_outputs cuando hay compare_bundle.
		# Evitamos recomponer de nuevo para no duplicar paneles (p.ej. Reposo repetido).
		self._write_compare_stress_rest()
		self.dual_mode_active = True
		self._load_previews_selected(self._default_preview_tabs())
		self._refresh_summary()
		self._select_tab_by_title("histograma")
		if deferred:
			self.statusBar().showMessage(f"Render HQ de comparación completado: {self.compare_label}")
		else:
			self.statusBar().showMessage(f"Comparación cargada: {self.compare_label}")

	def _load_compare_study_from_path(self, path: str):
		try:
			t_total = perf_counter()
			self._set_progress(10, "Cargando y procesando estudio de comparación...")
			t_stage = perf_counter()
			# Precargar para validar identidad de paciente ANTES del pipeline pesado.
			preloaded = dicom_loader.load(path, verbose=False)
			if not self._check_second_stage_patient(preloaded):
				self._set_progress(0, "Cancelado")
				return
			if not self._check_stage_zoom_consistency(preloaded):
				self._set_progress(0, "Cancelado")
				return
			bundle = self._process_secondary_bundle(path, preloaded_study=preloaded)
			self._log_timing_if_slow("Comparación: carga DICOM + segmentación + fase", t_stage)
			self.compare_bundle = bundle
			self.compare_metrics = bundle["metrics"]
			self.compare_ef = bundle["ef"]
			self.compare_label = bundle["label"]
			self._refresh_cine_source_selector()
			# Hidratar ambos cines de inmediato para que el segundo visor quede editable.
			self._apply_cine_source("primary", preserve_position=True)
			left_label, right_label = self._dual_compare_labels()

			if bool(self.realtime_deferred_render_check.isChecked()):
				self._set_progress(75, "Vista rápida de comparación...")
				prev_fast = bool(self.compare_interactive_fast_mode)
				self.compare_interactive_fast_mode = True
				try:
					self._write_compare_stress_rest()
				finally:
					self.compare_interactive_fast_mode = prev_fast
				self.dual_mode_active = True
				self._load_previews_selected(self._default_preview_tabs())
				self._refresh_summary()
				self._select_tab_by_title("histograma")
				self._set_progress(92, "Vista rápida lista (HQ diferido)...")
				self._schedule_deferred_hq_render(
					"compare",
					delay_ms=320,
					compare_bundle=bundle,
					left_label=left_label,
					right_label=right_label,
				)
				self._log("Modo tiempo real: comparación rápida lista; HQ diferido en progreso.")
				self._log_timing_if_slow("Comparación: total", t_total)
				self.statusBar().showMessage(f"Comparación rápida cargada: {self.compare_label}")
				return

			self._set_progress(75, "Generando salidas comparativas en todas las pestañas...")
			t_stage = perf_counter()
			self._run_compare_hq_pipeline(bundle, left_label=left_label, right_label=right_label, deferred=False)
			self._log_timing_if_slow("Comparación: render estudio secundario", t_stage)
			self._log_timing_if_slow("Comparación: re-render estudio principal con deltas", t_stage)
			self._set_progress(100, "Comparación lista")
			self._log_timing_if_slow("Comparación: total", t_total)
			self._log(f"Comparación cargada: {self.compare_label}")
			self.statusBar().showMessage(f"Comparación HQ cargada: {self.compare_label}")
		except Exception as exc:
			self._set_progress(0, "Error")
			QMessageBox.critical(self, "Error de comparación", str(exc))
			self._log(f"[ERROR compare] {exc}")

	def _estimate_ef_for(self, study, seg) -> dict:
		"""Corre el estimador de EF sobre un (study, seg) arbitrario sin perder el
		estado actual. Reusa _estimate_lv_ef_preliminary temporalmente."""
		saved_study, saved_seg = self.study, self.seg
		try:
			self.study, self.seg = study, seg
			return self._estimate_lv_ef()
		finally:
			self.study, self.seg = saved_study, saved_seg

	def _select_tab_by_title(self, title: str) -> bool:
		# Acepta el título visible o la clave interna (p.ej. "comparacion_ejes").
		target = self._tab_titles.get(str(title), str(title))
		for i in range(self.tabs.count()):
			if self.tabs.tabText(i) == target:
				self.tabs.setCurrentIndex(i)
				return True
		return False

	def _write_compare_stress_rest(self):
		"""Genera comparacion_stress_rest.png: panel comparativo de métricas de
		disincronía (actual vs comparación) con Δ e interpretación de stunning."""
		if self.metrics is None or self.compare_metrics is None:
			return
		import matplotlib.pyplot as plt

		cur_label = os.path.splitext(os.path.basename(self.file_edit.text().strip()))[0] or "Actual"
		cmp_label = self.compare_label or "Comparación"

		keys = [
			("phase_sd", "Phase SD (°)", "menor = más sincrónico"),
			("bandwidth", "Bandwidth (°)", "menor = más sincrónico"),
			("kurtosis", "Kurtosis", "mayor = más sincrónico"),
			("entropy_normalized_pct", "Entropy norm. (%)", "menor = más sincrónico"),
		]
		cur_vals = [float(self.metrics.get(k, 0.0)) for k, _, _ in keys]
		cmp_vals = [float(self.compare_metrics.get(k, 0.0)) for k, _, _ in keys]
		deltas = [c - r for c, r in zip(cur_vals, cmp_vals)]

		fig, (ax_bar, ax_txt) = plt.subplots(1, 2, figsize=(13, 6.0), gridspec_kw={"width_ratios": [1.4, 1.0]})
		x = np.arange(len(keys))
		width = 0.38
		ax_bar.bar(x - width / 2, cur_vals, width, label=cur_label, color="#d9534f")
		ax_bar.bar(x + width / 2, cmp_vals, width, label=cmp_label, color="#0275d8")
		ax_bar.set_xticks(x)
		ax_bar.set_xticklabels([lbl for _, lbl, _ in keys], fontsize=9)
		ax_bar.set_title("Disincronía: comparación entre estudios", fontsize=12, fontweight="bold")
		ax_bar.legend()
		ax_bar.grid(True, axis="y", alpha=0.3)
		for xi, (cv, rv) in enumerate(zip(cur_vals, cmp_vals)):
			ax_bar.text(xi - width / 2, cv, f"{cv:.1f}", ha="center", va="bottom", fontsize=8)
			ax_bar.text(xi + width / 2, rv, f"{rv:.1f}", ha="center", va="bottom", fontsize=8)

		# Panel de texto: tabla de Δ + interpretación clínica.
		ax_txt.axis("off")
		lines = [f"{cur_label}  vs  {cmp_label}", ""]
		for (k, lbl, _), cv, rv, dv in zip(keys, cur_vals, cmp_vals, deltas):
			lines.append(f"{lbl:<16} {cv:7.2f}  {rv:7.2f}   Δ {dv:+.2f}")
		lines.append("")
		# Interpretación de stunning: si el estudio actual (stress) tiene PSD y BW
		# claramente mayores que el de comparación (rest), sugiere disincronía
		# transitoria post-stress (stunning isquémico) — Camilletti 2015.
		d_psd = deltas[0]
		d_bw = deltas[1]
		psd_cur = cur_vals[0]
		if d_psd > 3.0 and d_bw > 8.0:
			interp = (
				"Δ positivo marcado en PSD y BW:\n"
				"sugiere DISINCRONÍA POST-STRESS\n"
				"(posible stunning isquémico).\n"
				"Revisar perfusión regional."
			)
			color = "#d9534f"
		elif abs(d_psd) <= 3.0 and abs(d_bw) <= 8.0:
			interp = (
				"Diferencias pequeñas entre estudios:\n"
				"sincronía estable, sin stunning\n"
				"significativo aparente."
			)
			color = "#5cb85c"
		else:
			interp = (
				"Diferencias intermedias:\n"
				"correlacionar con clínica y\n"
				"perfusión regional."
			)
			color = "#f0ad4e"
		ax_txt.text(0.0, 0.95, "\n".join(lines), family="monospace", fontsize=10, va="top")
		ax_txt.text(0.0, 0.42, interp, fontsize=10.5, va="top", color=color, fontweight="bold")
		ax_txt.text(
			0.0, 0.10,
			"Base: Camilletti/Erriest 2015 (ASNC).\nCutoffs Δ orientativos, no diagnósticos.",
			fontsize=8, va="top", color="#666",
		)

		fig.suptitle("Comparación de disincronía entre estudios (stress vs rest)", fontsize=13, fontweight="bold")
		self._stamp_export_figure(fig, self.cine)
		fig.tight_layout(rect=(0, 0, 1, 0.96))
		fig.savefig(os.path.join(self.output_dir, "comparacion_stress_rest.png"), dpi=160, bbox_inches="tight")
		plt.close(fig)

	def open_output_folder(self):
		QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_dir))

	def open_docs_portal(self):
		docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
		portal_path = os.path.join(docs_dir, "index.html")
		fallback_path = os.path.join(docs_dir, "GUIA_AUDITORIA_VALIDACION_CALCULOS.html")
		if os.path.exists(portal_path):
			QDesktopServices.openUrl(QUrl.fromLocalFile(portal_path))
			return
		if os.path.exists(fallback_path):
			QDesktopServices.openUrl(QUrl.fromLocalFile(fallback_path))
			return
		QMessageBox.information(self, "SINCRO", "No se encontró documentación HTML en docs.")

	def open_pdf(self):
		if not self._ensure_reports_generated():
			return
		pdf_path = os.path.join(self.output_dir, "informe_sincro.pdf")
		if not os.path.exists(pdf_path):
			QMessageBox.information(self, "SINCRO", "Todavía no hay PDF generado en output_demo.")
			return
		QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_path))

	def open_html_report(self):
		if not self._ensure_reports_generated():
			return
		html_path = os.path.join(self.output_dir, "informe_sincro.html")
		if not os.path.exists(html_path):
			QMessageBox.information(self, "SINCRO", "Todavía no hay HTML generado. Procesá un estudio primero.")
			return
		QDesktopServices.openUrl(QUrl.fromLocalFile(html_path))

	def save_html_as(self):
		import shutil
		if not self._ensure_reports_generated():
			return
		html_path = os.path.join(self.output_dir, "informe_sincro.html")
		if not os.path.exists(html_path):
			QMessageBox.information(self, "SINCRO", "Todavía no hay HTML generado. Procesá un estudio primero.")
			return
		# Nombre sugerido con paciente y fecha.
		patient = str(getattr(self.study, "patient_name", "") or "").strip().replace(" ", "_")
		patient_id = str(getattr(self.study, "patient_id", "") or "").strip()
		study_date = str(getattr(self.study, "study_date", "") or "").strip()
		if len(study_date) == 8 and study_date.isdigit():
			study_date = f"{study_date[6:8]}-{study_date[4:6]}-{study_date[0:4]}"
		parts = ["SINCRO"]
		if patient:
			parts.append(patient)
		elif patient_id:
			parts.append(patient_id)
		if study_date:
			parts.append(study_date)
		suggested = "_".join(parts) + ".html"
		dest, _ = QFileDialog.getSaveFileName(
			self,
			"Guardar informe HTML como...",
			suggested,
			"Archivos HTML (*.html);;Todos (*.*)",
		)
		if not dest:
			return
		try:
			shutil.copy2(html_path, dest)
			self._log(f"HTML guardado en: {dest}")
			self.statusBar().showMessage(f"HTML guardado en: {dest}")
		except Exception as exc:
			QMessageBox.critical(self, "SINCRO", f"No se pudo guardar el HTML:\n{exc}")

	def _hash_store_count(self) -> int:
		"""Retorna la cantidad de hashes almacenados."""
		try:
			from report.hash_store import HashStore
			return HashStore().count()
		except Exception:
			return 0

	def verify_html_integrity(self):
		"""Verifica la integridad de un archivo HTML contra su hash registrado."""
		path, _ = QFileDialog.getOpenFileName(
			self, "Seleccionar HTML para verificar...",
			self.output_dir,
			"Archivos HTML (*.html);;Todos (*.*)",
		)
		if not path:
			return
		try:
			from report.hash_store import HashStore
			store = HashStore()
			ok, msg = store.verify(path)
			if ok:
				QMessageBox.information(self, "SINCRO — Verificación de integridad", msg)
			else:
				QMessageBox.warning(self, "SINCRO — Verificación de integridad", msg)
		except Exception as exc:
			QMessageBox.critical(self, "SINCRO", f"Error al verificar:\n{exc}")

	def cleanup_hash_store(self):
		"""Limpia hashes antiguos del almacén según retención configurada."""
		try:
			from report.hash_store import HashStore
			store = HashStore()
			settings = getattr(self, "_ui_settings", None)
			mf = int(settings.value("integrity/hash_max_files", 200)) if settings else 200
			md = int(settings.value("integrity/hash_max_days", 90)) if settings else 90
			n = store.count()
			removed = store.cleanup(max_files=mf, max_days=md)
			remaining = store.count()
			mf_txt = str(mf) if mf > 0 else "sin límite"
			md_txt = f"{md} días" if md > 0 else "sin límite"
			QMessageBox.information(
				self, "SINCRO — Limpieza de hashes",
				f"Hashes antes: {n}\nEliminados: {removed}\nRestantes: {remaining}\n"
				f"Retención: {mf_txt} / {md_txt}.",
			)
		except Exception as exc:
			QMessageBox.critical(self, "SINCRO", f"Error al limpiar:\n{exc}")

	def open_report_editor(self):
		"""Abre el editor de informe clínico con formato rico."""
		if self.study is None:
			QMessageBox.information(self, "SINCRO", "Cargá un estudio primero.")
			return
		# Usar el resumen ejecutivo cacheado del pipeline (mismo texto que el HTML).
		exec_html = getattr(self, "_cached_exec_html", "")
		if not exec_html:
			# Fallback: generar si no hay caché (ej: editor antes de generar HTML).
			try:
				from core.executive_summary import build_executive_summary
				vol = self._compute_volumes_ml()
				ef = self._estimate_lv_ef()
				summary = build_executive_summary(
					metrics=self.metrics, ef=ef, territory=self.territory,
					volumes=vol, phase_label="Estudio",
				)
				if summary.get("available"):
					sections = summary.get("sections", [])
					exec_html = "".join(f"<p><b>{s['title']}.</b> {s['text']}</p>" for s in sections)
			except Exception:
				pass
		from report.report_editor import ReportEditorDialog
		patient_name = str(getattr(self.study, "patient_name", "") or "").strip() or "Paciente"
		study_desc = str(getattr(self.study, "study_description", "") or "")
		dlg = ReportEditorDialog(
			self,
			exec_summary=exec_html,
			patient_name=patient_name,
			study_desc=study_desc,
		)
		if dlg.exec():
			self._report_editor_html = dlg.get_html()
			self._log("Informe del editor guardado en memoria. Se incluirá en el próximo HTML.")

	def save_pdf_as(self):
		import shutil
		if not self._ensure_reports_generated():
			return
		pdf_path = os.path.join(self.output_dir, "informe_sincro.pdf")
		if not os.path.exists(pdf_path):
			QMessageBox.information(self, "SINCRO", "Todavía no hay PDF generado. Procesá un estudio primero.")
			return
		dest, _ = QFileDialog.getSaveFileName(
			self,
			"Guardar informe PDF como...",
			"informe_sincro.pdf",
			"Archivos PDF (*.pdf);;Todos (*.*)",
		)
		if not dest:
			return
		try:
			shutil.copy2(pdf_path, dest)
			self._log(f"PDF guardado en: {dest}")
			self.statusBar().showMessage(f"PDF guardado en: {dest}")
		except Exception as exc:
			QMessageBox.critical(self, "SINCRO", f"No se pudo guardar el PDF:\n{exc}")

	def open_polar_map(self):
		pm_path = os.path.join(self.output_dir, "polar_map.png")
		if not os.path.exists(pm_path):
			QMessageBox.information(self, "SINCRO", "Todavía no hay polar map generado. Procesá un estudio primero.")
			return
		QDesktopServices.openUrl(QUrl.fromLocalFile(pm_path))
