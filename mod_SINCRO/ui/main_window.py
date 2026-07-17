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
from PyQt6.QtCore import QSize, Qt, QSettings, QTimer
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QIcon, QMovie, QPixmap, QColor, QImage
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
	QMessageBox,
	QPushButton,
	QPlainTextEdit,
	QProgressBar,
	QScrollArea,
	QSpinBox,
	QSlider,
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
from core.metrics import calculate_phase_metrics
from core.phase_analysis import phase_analysis
from core.segmentation import segment_myocardium
from report.report_generator import generate_report
from viz.histogram import build_phase_histogram, save_histogram
from viz.polar_map import (
	build_clinical_phase_panel,
	build_polar_map,
	save_clinical_phase_panel,
	save_polar_map,
)

from ui.cine_widget import CineWidget
from version import __version__


class MainWindow(QMainWindow):
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
		self.axis_companions: dict[str, object] = {}
		self.seg = None
		self.phase_result = None
		self.metrics = None
		self.aha = None
		self.phase_by_seg = None
		self.territory = None
		# Estudio de comparación (típicamente REST vs el actual STRESS) para el
		# análisis stress/rest de disincronía (stunning isquémico, Camilletti 2015).
		self.compare_metrics = None
		self.compare_label = None
		self.compare_ef = None
		self.compare_bundle = None
		self.dual_mode_active = False
		self.primary_manual_rois_text = ""
		self.compare_manual_rois_text = ""
		self.active_cine_source = "primary"
		self.preview_zoom: dict[str, float] = {}
		self.preview_base_sizes: dict[str, QSize] = {}
		self.preview_pixmaps: dict[str, QPixmap] = {}
		self.preview_movies: dict[str, QMovie] = {}
		self.preview_zoom_labels: dict[str, QLabel] = {}
		self.polar_cine_toggle_btn: QToolButton | None = None
		self.polar_cine_preview_frames: list[QPixmap] = []
		self.polar_cine_preview_index = 0
		self.polar_cine_playing = False
		self.polar_cine_timer = QTimer(self)
		self.polar_cine_timer.timeout.connect(self._advance_polar_cine_frame)
		self._tooltips_cache_main: dict[QWidget, str] = {}
		self._ui_show_helpers = True
		self._ui_enable_tooltips = True
		self._ui_compact_controls = False
		self.compare_axes_preview_frames: list[QPixmap] = []
		self.compare_axes_preview_index = 0
		self.compare_axes_playing = False
		self.compare_interactive_fast_mode = False
		self.compare_axes_cine_timer = QTimer(self)
		self.compare_axes_cine_timer.timeout.connect(self._advance_compare_axes_frame)
		self.compare_axes_refresh_timer = QTimer(self)
		self.compare_axes_refresh_timer.setSingleShot(True)
		self.compare_axes_refresh_timer.timeout.connect(self._refresh_compare_axes_panel_now)
		self._cache_study_sig = ""
		self._cache_seg_sig = ""
		self._cache_phase_sig = ""
		self._cache_output_sig = ""
		self._cache_tab_output_sigs: dict[str, str] = {}
		self._last_primary_path = ""
		self.advanced_mode_enabled = False
		self._basic_tab_order = [
			"slices_fase",
			"polar_combo",
			"delta_combo",
			"histograma",
			"comparacion_stress_rest",
		]
		self._advanced_extra_tab_order = [
			"polar_perfusion_directa",
			"polar_cine_montaje",
			"comparacion_ejes",
			"ventriculograma",
			"bullseye_directo",
		]

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

		file_row = QHBoxLayout()
		file_row.addWidget(self.file_edit, 1)
		file_row.addWidget(browse_btn)
		file_box = QGroupBox("Estudio")
		file_box_layout = QVBoxLayout(file_box)
		file_box_layout.addLayout(file_row)
		self._sidebar_layout.addWidget(file_box)

		controls_box = QGroupBox("Procesamiento")
		controls_form = QFormLayout(controls_box)

		self.seg_method = QComboBox()
		self.seg_method.addItems(["auto", "threshold", "manual"])

		self.threshold_spin = QDoubleSpinBox()
		self.threshold_spin.setRange(0.01, 0.90)
		self.threshold_spin.setSingleStep(0.01)
		self.threshold_spin.setValue(0.35)

		self.sigma_spin = QDoubleSpinBox()
		self.sigma_spin.setRange(0.0, 6.0)
		self.sigma_spin.setSingleStep(0.1)
		self.sigma_spin.setValue(1.0)

		self.harmonics_spin = QSpinBox()
		self.harmonics_spin.setRange(1, 4)
		self.harmonics_spin.setValue(1)

		self.phase_threshold_spin = QDoubleSpinBox()
		self.phase_threshold_spin.setRange(0.01, 0.50)
		self.phase_threshold_spin.setSingleStep(0.01)
		self.phase_threshold_spin.setValue(0.10)

		self.normalize_check = QCheckBox("Normalizar referencia de fase")
		self.normalize_check.setChecked(False)

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

		self.manual_rois = QPlainTextEdit()
		self.manual_rois.setPlaceholderText(
			"Modo manual: slice,cy,cx,r_inner,r_outer\n"
			"ej: 9,12,11,4,7 | apex sin cavidad: 9,12,11,-,7"
		)
		self.manual_rois.setMaximumHeight(84)
		self.manual_rois.setToolTip("Cada línea define un slice. En apex/base podés usar r_inner='-' para indicar sin cavidad visible.")

		controls_form.addRow("Segmentación", self.seg_method)
		controls_form.addRow("Threshold", self.threshold_spin)
		controls_form.addRow("Smooth sigma", self.sigma_spin)
		controls_form.addRow("Harmonics", self.harmonics_spin)
		controls_form.addRow("Amplitude filter", self.phase_threshold_spin)
		controls_form.addRow("Colormap fase", self.cmap_combo)
		controls_form.addRow("Estilo visual", self.visual_style_combo)
		controls_form.addRow("Rotación polar", self.polar_rotation_spin)
		controls_form.addRow("Velocidad polar cine", self.polar_cine_speed_spin)
		controls_form.addRow("Math polar stress/rest", self.polar_compare_math_combo)
		controls_form.addRow("Términos math", polar_math_terms)
		controls_form.addRow(self.export_polar_mp4_check)
		controls_form.addRow(self.profile_timing_check)
		controls_form.addRow(self.normalize_check)
		controls_form.addRow("Sexo (DB normal)", self.normal_sex_combo)
		controls_form.addRow("Protocolo (DB normal)", self.normal_protocol_combo)
		controls_form.addRow("DB normal", self.normal_db_combo)
		controls_form.addRow(self.auto_run_check)

		self.seg_method.setToolTip("auto: segmentación automática; threshold: umbral simple; manual: usa los ROIs que dibujes o pegues.")
		self.threshold_spin.setToolTip("Porcentaje del máximo usado para separar miocardio del fondo.")
		self.sigma_spin.setToolTip("Suavizado espacial aplicado antes del threshold en segmentación.")
		self.harmonics_spin.setToolTip("Cantidad de armónicos usados para estabilizar la fase.")
		self.phase_threshold_spin.setToolTip("Filtro de amplitud: descarta voxels débiles o ruidosos.")
		self.cmap_combo.setToolTip("Colormap cíclico para visualizar fase.")
		self.visual_style_combo.setToolTip("Tema visual de los paneles clínicos (curva FEVI, panel funcional gated y bull's eye).")
		self.polar_rotation_spin.setToolTip("Rota el mapa polar de perfusión continua. Ajustalo para alinear ANT/SEP/LAT/INF a tu convención.")
		self.polar_cine_speed_spin.setToolTip("Duración por frame del GIF del cine polar (en milisegundos).")
		self.polar_compare_math_combo.setToolTip("Operación matemática opcional entre mapas polares de esfuerzo/reposo en el cine comparativo.")
		self.polar_compare_term_a_combo.setToolTip("Primer término de la operación A op B.")
		self.polar_compare_term_b_combo.setToolTip("Segundo término de la operación A op B.")
		self.export_polar_mp4_check.setToolTip("Además del GIF, intenta exportar un MP4 del cine polar gatillado.")
		self.profile_timing_check.setToolTip("Registra en el log solo etapas que superan 0.5 s.")
		self.normalize_check.setToolTip("Resta una referencia global de fase para comparar estudios.")
		self.normal_sex_combo.setToolTip("Sexo del paciente: los valores normales de PSD/BW difieren por sexo (Mukherjee 2016).")
		self.normal_protocol_combo.setToolTip("Protocolo del estudio: stress da PSD/BW mayores que rest.")
		self.normal_db_combo.setToolTip("Base de datos normal usada para z-score y flag de disincronía (media + 2 SD).")

		self._sidebar_layout.addWidget(controls_box)

		report_cmap_box = QGroupBox("Escalas informe (por imagen)")
		report_cmap_layout = QGridLayout(report_cmap_box)
		report_cmap_layout.setContentsMargins(6, 6, 6, 6)
		report_cmap_layout.setHorizontalSpacing(4)
		report_cmap_layout.setVerticalSpacing(4)

		def _mk_combo(current: str) -> QComboBox:
			cb = QComboBox()
			cb.addItems(self._all_cmaps)
			if current in self._all_cmaps:
				cb.setCurrentText(current)
			return cb

		self.report_cmap_slices = _mk_combo("hot")
		self.report_cmap_axes = _mk_combo("hot")
		self.report_cmap_compare = _mk_combo("hot")
		self.report_cmap_panel_axes = _mk_combo("hot")
		self.report_cmap_phase = _mk_combo("french")
		self.report_cmap_polar_clinico = _mk_combo("french")
		self.report_cmap_amp = _mk_combo("turbo")
		self.report_cmap_bullseye = _mk_combo("turbo")
		self.report_cmap_polar_perf = _mk_combo("perf_clinical")

		report_cmap_layout.addWidget(QLabel("slices_fase"), 0, 0)
		report_cmap_layout.addWidget(self.report_cmap_slices, 0, 1)
		report_cmap_layout.addWidget(QLabel("ejes_ortogonales"), 1, 0)
		report_cmap_layout.addWidget(self.report_cmap_axes, 1, 1)
		report_cmap_layout.addWidget(QLabel("comparacion_ejes"), 2, 0)
		report_cmap_layout.addWidget(self.report_cmap_compare, 2, 1)
		report_cmap_layout.addWidget(QLabel("panel_funcional (ED/ES)"), 3, 0)
		report_cmap_layout.addWidget(self.report_cmap_panel_axes, 3, 1)
		report_cmap_layout.addWidget(QLabel("fase (overlay/polar)"), 4, 0)
		report_cmap_layout.addWidget(self.report_cmap_phase, 4, 1)
		report_cmap_layout.addWidget(QLabel("polar_clinico"), 5, 0)
		report_cmap_layout.addWidget(self.report_cmap_polar_clinico, 5, 1)
		report_cmap_layout.addWidget(QLabel("amplitud"), 6, 0)
		report_cmap_layout.addWidget(self.report_cmap_amp, 6, 1)
		report_cmap_layout.addWidget(QLabel("bullseye_directo"), 7, 0)
		report_cmap_layout.addWidget(self.report_cmap_bullseye, 7, 1)
		report_cmap_layout.addWidget(QLabel("polar_perfusion_directa"), 8, 0)
		report_cmap_layout.addWidget(self.report_cmap_polar_perf, 8, 1)

		self._sidebar_layout.addWidget(report_cmap_box)

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
		self.restart_btn.clicked.connect(self.restart_workspace_state)
		self.restart_btn.setToolTip("Limpia el estado en memoria de la sesión para cargar estudios nuevos desde cero.")
		self.process_btn = QPushButton("Procesar")
		self.process_btn.clicked.connect(self.process_current)
		self.process_btn.setToolTip("Recalcula segmentación, fase, métricas, polar map y gráficos.")
		self.auto_btn = QPushButton("Auto")
		self.auto_btn.clicked.connect(self.process_auto)
		self.auto_btn.setToolTip("Fuerza modo automático y procesa todo sin tocar manualmente el flujo.")
		self.open_folder_btn = QPushButton("Carpeta")
		self.open_folder_btn.clicked.connect(self.open_output_folder)
		self.open_folder_btn.setToolTip("Abre la carpeta con los PNG y el PDF generados.")
		self.open_pdf_btn = QPushButton("Abrir PDF")
		self.open_pdf_btn.clicked.connect(self.open_pdf)
		self.open_pdf_btn.setToolTip("Abre el informe clínico PDF generado.")
		self.save_pdf_as_btn = QPushButton("Guardar PDF como...")
		self.save_pdf_as_btn.clicked.connect(self.save_pdf_as)
		self.save_pdf_as_btn.setToolTip("Guarda una copia del informe PDF en la ubicación que elijas.")
		self.compare_stress_rest_btn = QPushButton("Comparar con Rest/Stress...")
		self.compare_stress_rest_btn.clicked.connect(self.load_compare_study)
		self.compare_stress_rest_btn.setToolTip(
			"Carga un segundo estudio (ej: REST) y compara la disincronía (PSD, BW, Kurtosis, Entropy) "
			"contra el estudio actual. Útil para detectar stunning isquémico post-stress (Camilletti 2015)."
		)
		self.apply_roi_all_btn = QPushButton("Replicar ROI a todos")
		self.apply_roi_all_btn.clicked.connect(self.apply_current_roi_to_all_slices)
		self.apply_roi_all_btn.setToolTip("Copia el ROI del slice actual a todos los slices del volumen.")
		self.load_one_or_two_btn = QPushButton("Cargar 1 o 2 estudios...")
		self.load_one_or_two_btn.clicked.connect(self.load_one_or_two_studies)
		self.load_one_or_two_btn.setToolTip("Permite cargar y procesar una fase (stress o rest) o dos fases para comparación integral.")
		self.advanced_toggle_btn = QPushButton("AVANZADO...")
		self.advanced_toggle_btn.clicked.connect(self.toggle_advanced_mode)
		self.advanced_toggle_btn.setToolTip("Activa paneles y render pesado. En básico se prioriza velocidad para asincronía.")
		self.ui_config_btn = QPushButton("Config UI...")
		self.ui_config_btn.clicked.connect(self.open_ui_preferences_dialog)
		self.ui_config_btn.setToolTip("Configura helpers, tooltips y modo compacto de botones.")
		button_row.addWidget(self.restart_btn, 0, 0, 1, 2)
		button_row.addWidget(self.process_btn, 1, 0)
		button_row.addWidget(self.auto_btn, 1, 1)
		button_row.addWidget(self.open_folder_btn, 2, 0)
		button_row.addWidget(self.open_pdf_btn, 2, 1)
		button_row.addWidget(self.save_pdf_as_btn, 3, 0, 1, 2)
		button_row.addWidget(self.compare_stress_rest_btn, 4, 0, 1, 2)
		button_row.addWidget(self.load_one_or_two_btn, 5, 0, 1, 2)
		button_row.addWidget(self.advanced_toggle_btn, 6, 0, 1, 2)
		button_row.addWidget(self.ui_config_btn, 7, 0, 1, 2)
		# Ubicar Acciones arriba de Procesamiento para tener comandos a la vista.
		self._sidebar_layout.insertWidget(1, button_box)

		roi_box = QGroupBox("ROI manual por slice")
		roi_layout = QVBoxLayout(roi_box)
		roi_layout.setContentsMargins(6, 6, 6, 6)
		roi_layout.setSpacing(4)
		roi_note = QLabel("Usá el visor para editar el ROI. También podés pegar líneas slice,cy,cx,r_inner,r_outer.")
		roi_note.setWordWrap(True)
		roi_note.setStyleSheet("color:#555;")
		roi_layout.addWidget(roi_note)
		roi_layout.addWidget(self.manual_rois)
		roi_actions_top = QHBoxLayout()
		roi_actions_top.addWidget(self.apply_roi_all_btn)
		roi_actions_top.addStretch(1)
		roi_layout.addLayout(roi_actions_top)

		roi_adjust_note = QLabel(
			"Ajuste Auto ROI desde slice actual: corregí un slice y propagá solo centro, radio interno, radio externo o todo el ajuste al volumen."
		)
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
		self.reset_roi_deltas_btn = QPushButton("Reset deltas")
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
		self.adjust_auto_center_btn = QPushButton("Centro -> todos")
		self.adjust_auto_center_btn.clicked.connect(self.adjust_auto_center_all_slices)
		self.adjust_auto_inner_btn = QPushButton("Interno -> todos")
		self.adjust_auto_inner_btn.clicked.connect(self.adjust_auto_inner_all_slices)
		self.adjust_auto_outer_btn = QPushButton("Externo -> todos")
		self.adjust_auto_outer_btn.clicked.connect(self.adjust_auto_outer_all_slices)
		self.adjust_auto_full_btn = QPushButton("Completo -> todos")
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

		compare_box = QGroupBox("Comparación original vs reconstruido")
		compare_layout = QVBoxLayout(compare_box)
		compare_layout.setContentsMargins(6, 6, 6, 6)
		compare_layout.setSpacing(4)
		compare_note = QLabel("Define el mismo gate y el mismo corte anatómico relativo para comparar HLA/VLA original vs reconstruido.")
		compare_note.setWordWrap(True)
		compare_note.setStyleSheet("color:#555;")
		compare_layout.addWidget(compare_note)
		cine_source_row = QHBoxLayout()
		cine_source_row.addWidget(QLabel("Cine/ROI"))
		self.cine_source_combo = QComboBox()
		self.cine_source_combo.currentIndexChanged.connect(self._on_cine_source_changed)
		cine_source_row.addWidget(self.cine_source_combo, 1)
		self.cine_primary_btn = QToolButton()
		self.cine_primary_btn.setText("SA Esfuerzo")
		self.cine_primary_btn.clicked.connect(lambda: self._apply_cine_source("primary"))
		self.cine_compare_btn = QToolButton()
		self.cine_compare_btn.setText("SA Reposo")
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
		self.use_cine_compare_btn = QPushButton("Usar gate/slice")
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
		compare_slice_row.addWidget(QLabel("Corte anatómico"), 0, 0)
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
		compare_offset_row.addWidget(QLabel("Desfase SA"), 0, 0)
		compare_offset_row.addWidget(self.compare_slice_offset_sa_spin, 0, 1)
		compare_offset_row.addWidget(QLabel("Desfase HLA"), 1, 0)
		compare_offset_row.addWidget(self.compare_slice_offset_hla_spin, 1, 1)
		compare_offset_row.addWidget(QLabel("Desfase VLA"), 2, 0)
		compare_offset_row.addWidget(self.compare_slice_offset_vla_spin, 2, 1)
		compare_layout.addLayout(compare_offset_row)
		self.compare_axes_cmap_combo = QComboBox()
		self.compare_axes_cmap_combo.addItems(self._all_cmaps)
		self.compare_axes_cmap_combo.setCurrentText("hot")
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
		self.compare_mask_check = QCheckBox("Mostrar máscara en comparativa")
		self.compare_mask_check.setChecked(True)
		self.compare_mask_check.toggled.connect(self._on_compare_mask_toggled)
		compare_layout.addWidget(self.compare_mask_check)
		self.compare_fast_drag_check = QCheckBox("Modo rápido al arrastrar (alta calidad al soltar)")
		self.compare_fast_drag_check.setChecked(True)
		compare_layout.addWidget(self.compare_fast_drag_check)
		compare_cine_row = QGridLayout()
		self.compare_axes_cine_check = QCheckBox("Cine en comparativa")
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
		self.compare_axes_cine_restart_btn.setText("Restart")
		self.compare_axes_cine_restart_btn.clicked.connect(self._restart_compare_axes_preview)
		compare_cine_row.addWidget(self.compare_axes_cine_check, 0, 0, 1, 2)
		compare_cine_row.addWidget(QLabel("Velocidad"), 1, 0)
		compare_cine_row.addWidget(self.compare_axes_cine_speed_spin, 1, 1)
		compare_cine_row.addWidget(self.compare_axes_cine_toggle_btn, 2, 0)
		compare_cine_row.addWidget(self.compare_axes_cine_restart_btn, 2, 1)
		self.compare_axes_export_frames_btn = QToolButton()
		self.compare_axes_export_frames_btn.setText("Exportar frames")
		self.compare_axes_export_frames_btn.clicked.connect(self.export_compare_axes_frames_debug)
		compare_cine_row.addWidget(self.compare_axes_export_frames_btn, 3, 0, 1, 2)
		compare_layout.addLayout(compare_cine_row)
		self.refresh_compare_btn = QPushButton("Actualizar comparativa")
		self.refresh_compare_btn.clicked.connect(self._refresh_compare_axes_panel_now)
		compare_layout.addWidget(self.refresh_compare_btn)
		self._update_compare_slice_label()
		self._update_compare_window_labels()
		self._refresh_cine_source_selector()
		self._sidebar_layout.addWidget(compare_box)

		self.summary_clinical = QTextEdit()
		self.summary_clinical.setReadOnly(True)
		self.summary_clinical.setMinimumHeight(120)
		self.summary_clinical.setPlaceholderText("Aquí aparecerá el resumen clínico cuando proceses un estudio.")
		self.summary_clinical.setToolTip("Resumen clínico: clasificación, volúmenes, FEVI y territorios.")

		self.summary_technical = QTextEdit()
		self.summary_technical.setReadOnly(True)
		self.summary_technical.setMinimumHeight(120)
		self.summary_technical.setPlaceholderText("Aquí aparecerá el detalle técnico y de procesamiento.")
		self.summary_technical.setToolTip("Detalle técnico: metadata DICOM, parámetros, métricas y notas de QC.")

		self.summary_tabs = QTabWidget()
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

		right = QWidget()
		right_layout = QVBoxLayout(right)
		right_layout.setContentsMargins(0, 0, 0, 0)
		right_layout.setSpacing(0)
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
			"polar_cine_montaje": "polar_cine_montaje",
			"comparacion_ejes": "comparacion_ejes",
			"comparacion_stress_rest": "stress_vs_rest",
			"ventriculograma": "panel_funcional_gated",
			"bullseye_directo": "bullseye_directo",
		}
		preview_help_texts = {
			"slices_fase": "Vista de referencia del slice/gate medio con máscara y fase superpuesta. Útil para control de calidad de segmentación.",
			"polar_combo": "Panel combinado: mapa polar AHA + panel clínico con histograma/PSD/PHB. Mantiene valores y lectura rápida en una sola pestaña.",
			"histograma": "Histograma de fase global para estimar dispersión temporal (PSD, BW, entropy).",
			"delta_combo": "Panel combinado de los dos mapas delta (con signo y absoluto) para comparar stress/rest en una sola pestaña.",
			"polar_perfusion_directa": "Mapa polar continuo de perfusión (intensidad normalizada): complementa fase para analizar heterogeneidad perfusional apex-base.",
			"polar_cine_montaje": "Cine polar gatillado: evolución temporal por gate del patrón polar; en dual-mode permite lectura dinámica stress/rest.",
			"comparacion_ejes": "Comparación multicorte por ejes entre estudios para detectar diferencias regionales en el mismo gate.",
			"comparacion_stress_rest": "Resumen de métricas de disincronía stress vs rest (PSD/BW/Kurtosis/Entropy) e interpretación clínica.",
			"ventriculograma": "Panel funcional integrado (ED/ES, fase, amplitud y curvas) para lectura clínica rápida.",
			"bullseye_directo": "Bull's-eye de perfusión segmentaria AHA (17): resumen compacto de intensidad regional.",
		}
		for name in [
			"slices_fase",
			"polar_combo",
			"delta_combo",
			"histograma",
			"polar_perfusion_directa",
			"polar_cine_montaje",
			"comparacion_ejes",
			"comparacion_stress_rest",
			"ventriculograma",
			"bullseye_directo",
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
			zoom_label = QLabel("100%")
			zoom_label.setStyleSheet("color:#444;")
			self.preview_zoom_labels[name] = zoom_label
			toolbar.addWidget(QLabel("Zoom"))
			toolbar.addWidget(zoom_out)
			toolbar.addWidget(zoom_in)
			toolbar.addWidget(zoom_reset)
			toolbar.addWidget(zoom_label)
			if name == "polar_cine_montaje":
				play_btn = QToolButton()
				play_btn.setText("Play")
				play_btn.clicked.connect(self._toggle_polar_cine_preview)
				self.polar_cine_toggle_btn = play_btn
				restart_btn = QToolButton()
				restart_btn.setText("Restart")
				restart_btn.clicked.connect(self._restart_polar_cine_preview)
				toolbar.addWidget(play_btn)
				toolbar.addWidget(restart_btn)
			toolbar.addStretch(1)
			tab_layout.addLayout(toolbar)

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
			self.preview_zoom[name] = 1.0
			scroller = QScrollArea()
			scroller.setWidgetResizable(False)
			scroller.setWidget(label)
			tab_layout.addWidget(scroller)
			self._tab_widgets[name] = tab
			self._tab_titles[name] = preview_titles.get(name, name)
			self._tab_tooltips[name] = helptxt or ""

		self._rebuild_tabs_for_mode()
		self.cine = CineWidget()
		self.cine.roiEdited.connect(self._on_cine_roi_changed)
		self.cine.playStateChanged.connect(self._on_play_state_changed)
		self.cine.activated.connect(lambda: self._on_cine_panel_activated("main"))
		self.cine.setToolTip("Reproducí el cine, hacé zoom y dibujá ROIs sobre la imagen.")
		self.cine_secondary_source: str | None = None
		self.cine_compare = CineWidget()
		self.cine_compare.roiEdited.connect(self._on_cine_compare_roi_changed)
		self.cine_compare.playStateChanged.connect(self._on_play_state_changed)
		self.cine_compare.activated.connect(lambda: self._on_cine_panel_activated("secondary"))
		self.cine_compare.setToolTip("Segundo visor (otro estudio): editable para ajustar ROI esfuerzo/reposo en paralelo.")
		self.cine.set_controls_visible(True)
		self.cine_compare.set_controls_visible(False)
		lower_cine_panel = QWidget()
		lower_cine_layout = QHBoxLayout(lower_cine_panel)
		lower_cine_layout.setContentsMargins(0, 0, 0, 0)
		lower_cine_layout.setSpacing(6)
		lower_cine_layout.addWidget(self.cine, 1)
		lower_cine_layout.addWidget(self.cine_compare, 1)
		right_splitter.addWidget(self.tabs)
		right_splitter.addWidget(lower_cine_panel)
		right_splitter.setStretchFactor(0, 3)
		right_splitter.setStretchFactor(1, 1)
		right_layout.addWidget(right_splitter)

		splitter.addWidget(left)
		splitter.addWidget(right)
		splitter.setStretchFactor(0, 1)
		splitter.setStretchFactor(1, 4)
		splitter.setSizes([420, 1140])
		right_splitter.setSizes([920, 140])
		self.main_splitter = splitter
		self.right_splitter = right_splitter
		self._ui_settings = QSettings("Gammasys", "GammaSync")
		self._load_global_ui_preferences()
		self._restore_window_layout()

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

		if initial_path:
			self.file_edit.setText(initial_path)
			if self.auto_run_check.isChecked():
				self.process_auto()
			else:
				self.process_current()

	def _log(self, message: str):
		self.log_box.append(message)

	def _restore_window_layout(self):
		geom = self._ui_settings.value("window_geometry", None)
		if geom is not None:
			self.restoreGeometry(geom)
		main_state = self._ui_settings.value("main_splitter_state", None)
		if main_state is not None:
			self.main_splitter.restoreState(main_state)
		right_state = self._ui_settings.value("right_splitter_state", None)
		if right_state is not None:
			self.right_splitter.restoreState(right_state)

	def _save_window_layout(self):
		self._ui_settings.setValue("window_geometry", self.saveGeometry())
		self._ui_settings.setValue("main_splitter_state", self.main_splitter.saveState())
		self._ui_settings.setValue("right_splitter_state", self.right_splitter.saveState())
		self._ui_settings.sync()

	def _load_global_ui_preferences(self):
		self._ui_show_helpers = bool(self._ui_settings.value("ui/show_helpers", True, type=bool))
		self._ui_enable_tooltips = bool(self._ui_settings.value("ui/enable_tooltips", True, type=bool))
		self._ui_compact_controls = bool(self._ui_settings.value("ui/compact_controls", False, type=bool))

	def _save_global_ui_preferences(self):
		self._ui_settings.setValue("ui/show_helpers", bool(self._ui_show_helpers))
		self._ui_settings.setValue("ui/enable_tooltips", bool(self._ui_enable_tooltips))
		self._ui_settings.setValue("ui/compact_controls", bool(self._ui_compact_controls))
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
			self.helper_box.setVisible(bool(self._ui_show_helpers))
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

	def open_ui_preferences_dialog(self):
		dlg = QDialog(self)
		dlg.setWindowTitle("Configuración UI")
		root = QVBoxLayout(dlg)
		msg = QLabel("Preferencias globales de interfaz para simplificar controles y ayuda visual.")
		msg.setWordWrap(True)
		root.addWidget(msg)
		show_helpers = QCheckBox("Mostrar helpers / ayuda rápida")
		show_helpers.setChecked(bool(self._ui_show_helpers))
		enable_tooltips = QCheckBox("Habilitar tooltips")
		enable_tooltips.setChecked(bool(self._ui_enable_tooltips))
		compact_controls = QCheckBox("Modo compacto (ocultar botones secundarios)")
		compact_controls.setChecked(bool(self._ui_compact_controls))
		root.addWidget(show_helpers)
		root.addWidget(enable_tooltips)
		root.addWidget(compact_controls)
		buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
		buttons.accepted.connect(dlg.accept)
		buttons.rejected.connect(dlg.reject)
		root.addWidget(buttons)
		if dlg.exec() != int(QDialog.DialogCode.Accepted):
			return
		self._ui_show_helpers = bool(show_helpers.isChecked())
		self._ui_enable_tooltips = bool(enable_tooltips.isChecked())
		self._ui_compact_controls = bool(compact_controls.isChecked())
		self._apply_global_ui_preferences()
		self._save_global_ui_preferences()
		self.statusBar().showMessage("Configuración UI aplicada")

	def closeEvent(self, event):
		self._save_window_layout()
		super().closeEvent(event)

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
		if self.active_cine_source == "compare":
			active_auto_roi_method = self.cine_compare.auto_roi_method()
			atten_pct, feather_px = self.cine_compare.intestinal_params()
			intestinal_scope = self.cine_compare.intestinal_scope()
			intestinal_apply_enabled = self.cine_compare.intestinal_apply_enabled()
		return {
			"seg_method": str(self.seg_method.currentText()),
			"threshold": float(self.threshold_spin.value()),
			"smooth_sigma": float(self.sigma_spin.value()),
			"harmonics": int(self.harmonics_spin.value()),
			"amp_filter": float(self.phase_threshold_spin.value()),
			"normalize_reference": bool(self.normalize_check.isChecked()),
			"phase_cmap": str(self.cmap_combo.currentText()),
			"visual_style": str(self.visual_style_combo.currentText()),
			"polar_rotation_deg": int(self.polar_rotation_spin.value()),
			"polar_cine_speed_ms": int(self.polar_cine_speed_spin.value()),
			"polar_compare_math_op": str(self.polar_compare_math_combo.currentText()),
			"polar_compare_math_a": str(self.polar_compare_term_a_combo.currentText()),
			"polar_compare_math_b": str(self.polar_compare_term_b_combo.currentText()),
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
			"normal_sex": str(self.normal_sex_combo.currentText()),
			"normal_protocol": str(self.normal_protocol_combo.currentText()),
			"normal_db": str(self.normal_db_combo.currentText()),
			"auto_roi_method": str(active_auto_roi_method),
			"intestinal_attenuation_pct": int(atten_pct),
			"intestinal_feather_px": int(feather_px),
			"intestinal_scope": str(intestinal_scope),
			"intestinal_apply_enabled": bool(intestinal_apply_enabled),
			"ui_show_helpers": bool(self._ui_show_helpers),
			"ui_enable_tooltips": bool(self._ui_enable_tooltips),
			"ui_compact_controls": bool(self._ui_compact_controls),
			"manual_rois_text": self.manual_rois.toPlainText(),
			"updated_at": datetime.now().isoformat(timespec="seconds"),
		}

	def _apply_processing_params(self, params: dict):
		if "seg_method" in params:
			self.seg_method.setCurrentText(str(params["seg_method"]))
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
		if "phase_cmap" in params:
			self.cmap_combo.setCurrentText(str(params["phase_cmap"]))
		if "visual_style" in params:
			style_value = str(params["visual_style"])
			if "like" in style_value.lower():
				style_value = "Clinico"
			self.visual_style_combo.setCurrentText(style_value)
		if "polar_rotation_deg" in params:
			self.polar_rotation_spin.setValue(int(params["polar_rotation_deg"]))
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
		sidebar.setMinimumWidth(360)
		sidebar.setMaximumWidth(560)
		sidebar.setStyleSheet(
			"#sincroSidebar { background: #f7f8fb; border-right: 1px solid #d7dce5; }"
			"QGroupBox { font-weight: 600; border: 1px solid #d7dce5; border-radius: 7px; margin-top: 6px; background: white; }"
			"QGroupBox::title { subcontrol-origin: margin; left: 7px; padding: 0 2px; color: #1f3b5b; }"
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
		path, _ = QFileDialog.getOpenFileName(self, "Abrir DICOM gated", "", "DICOM (*.dcm *.dicom);;Todos (*.*)")
		if path:
			self.file_edit.setText(path)
			if not self.preset_patient_edit.text().strip():
				self._refresh_presets_for_current_patient()
			if self.auto_run_check.isChecked():
				self.process_auto()

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
		self.manual_rois.blockSignals(True)
		self.manual_rois.setPlainText(formatted)
		self.manual_rois.blockSignals(False)
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
		if self.compare_bundle is not None:
			compare_label = self.compare_label or "Reposo / comparativo"
			self.cine_source_combo.addItem(f"Reposo / {compare_label}", "compare")
			target_index = 1 if self.active_cine_source == "compare" else 0
		else:
			self.active_cine_source = "primary"
			target_index = 0
		self.cine_source_combo.setCurrentIndex(target_index)
		self.cine_source_combo.setEnabled(self.compare_bundle is not None)
		self.cine_primary_btn.setEnabled(self.study is not None)
		self.cine_compare_btn.setEnabled(self.compare_bundle is not None)
		self.cine_source_combo.blockSignals(False)

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
		if self.study is not None:
			return self.study.cube
		return None

	def _refresh_dual_cine_views(self, *, preserve_position: bool = True, preferred_gate: int | None = None, preferred_slice: int | None = None):
		main_cube = self._cube_for_source("primary")
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

		if self.compare_bundle is None:
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
		if self.compare_bundle is None:
			self.cine.set_active_highlight(True)
			self.cine_compare.set_active_highlight(False)
			return
		main_active = self.active_cine_source != "compare"
		self.cine.set_active_highlight(main_active)
		self.cine_compare.set_active_highlight(not main_active)

	def _on_cine_panel_activated(self, panel: str):
		if panel == "secondary" and self.compare_bundle is not None:
			self._apply_cine_source("compare", preserve_position=True)
			self.statusBar().showMessage("Visor activo: Reposo")
			return
		self._apply_cine_source("primary", preserve_position=True)
		self.statusBar().showMessage("Visor activo: Esfuerzo")

	def _current_cine_cube(self):
		if self.active_cine_source == "compare" and self.compare_bundle is not None:
			return self.compare_bundle["study"].cube
		if self.study is not None:
			return self.study.cube
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
		active_intestinal_apply = self.cine.intestinal_apply_enabled()
		if self.active_cine_source == "compare":
			active_method = self.cine_compare.auto_roi_method()
			active_atten, active_feather = self.cine_compare.intestinal_params()
			active_intestinal_scope = self.cine_compare.intestinal_scope()
			active_intestinal_apply = self.cine_compare.intestinal_apply_enabled()
		self.cine.set_auto_roi_method(active_method)
		self.cine_compare.set_auto_roi_method(active_method)
		self.cine.set_intestinal_params(active_atten, active_feather)
		self.cine_compare.set_intestinal_params(active_atten, active_feather)
		self.cine.set_intestinal_scope(active_intestinal_scope)
		self.cine_compare.set_intestinal_scope(active_intestinal_scope)
		self.cine.set_intestinal_apply_enabled(active_intestinal_apply)
		self.cine_compare.set_intestinal_apply_enabled(active_intestinal_apply)

		self._save_active_manual_rois_text()
		self.active_cine_source = "compare" if source == "compare" and self.compare_bundle is not None else "primary"
		self.manual_rois.blockSignals(True)
		self.manual_rois.setPlainText(self._load_manual_rois_text_for_source(self.active_cine_source))
		self.manual_rois.blockSignals(False)
		main_controls = self.active_cine_source == "primary"
		self.cine.set_controls_visible(main_controls)
		self.cine_compare.set_controls_visible(not main_controls and self.compare_bundle is not None)
		self._refresh_dual_cine_views(
			preserve_position=preserve_position,
			preferred_gate=preferred_gate,
			preferred_slice=preferred_slice,
		)
		self._update_cine_active_border()

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
		if self.study is None or self.seg is None:
			return
		try:
			self._write_compare_axes_panel(cmap_compare=str(self.compare_axes_cmap_combo.currentText()))
			self._load_preview("comparacion_ejes")
			self.statusBar().showMessage("Comparativa de ejes actualizada")
		except Exception as exc:
			self._log(f"[WARN] No se pudo actualizar comparativa de ejes: {exc}")

	def _on_preview_tab_changed(self, index: int):
		if index < 0 or self.study is None or self.seg is None:
			return
		title = self.tabs.tabText(index)
		if title == "comparacion_ejes" and self.compare_axes_cine_check.isChecked() and not self.compare_axes_preview_frames:
			self._set_progress(88, "Generando cine de comparacion_ejes...")
			try:
				self._write_compare_axes_panel(cmap_compare=str(self.compare_axes_cmap_combo.currentText()), build_cine=True)
				self._load_preview("comparacion_ejes")
			finally:
				self._set_progress(100, "Procesamiento completo")

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

	def restart_workspace_state(self):
		self.cine.stop_playback()
		self.polar_cine_timer.stop()
		self.polar_cine_playing = False
		self.polar_cine_preview_frames = []
		self.polar_cine_preview_index = 0
		self.compare_axes_cine_timer.stop()
		self.compare_axes_playing = False
		self.compare_axes_preview_frames = []
		self.compare_axes_preview_index = 0
		self._clear_compare_state()
		self.study = None
		self.axis_companions = {}
		self.seg = None
		self.phase_result = None
		self.metrics = None
		self.aha = None
		self.phase_by_seg = None
		self.territory = None
		self.file_edit.clear()
		self.primary_manual_rois_text = ""
		self.compare_manual_rois_text = ""
		self._sync_manual_rois({})
		self.manual_rois.clear()
		if self.seg_method.currentText() == "manual":
			self.seg_method.setCurrentText("auto")
		self.summary_clinical.clear()
		self.summary_technical.clear()
		for movie in list(self.preview_movies.values()):
			movie.stop()
		self.preview_movies.clear()
		self.preview_pixmaps.clear()
		for name, label in self.preview_labels.items():
			self.preview_zoom[name] = 1.0
			label.clear()
			label.setText("Sin procesar")
			if name in self.preview_zoom_labels:
				self.preview_zoom_labels[name].setText("100%")
		self.cine.set_cube(None)
		self._refresh_cine_source_selector()
		self._progress_bar.setValue(0)
		self._progress_bar.setFormat("Listo")
		self.log_box.clear()
		self._log("RESTART: sesión limpia, lista para cargar estudios nuevos.")
		self._cache_study_sig = ""
		self._cache_seg_sig = ""
		self._cache_phase_sig = ""
		self._invalidate_output_cache()
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

	def _collect_visual_signature_payload(self) -> dict:
		return {
			"visual_style": str(self.visual_style_combo.currentText()),
			"polar_rotation_deg": int(self.polar_rotation_spin.value()),
			"polar_cine_speed_ms": int(self.polar_cine_speed_spin.value()),
			"polar_compare_math_op": str(self.polar_compare_math_combo.currentText()),
			"polar_compare_math_a": str(self.polar_compare_term_a_combo.currentText()),
			"polar_compare_math_b": str(self.polar_compare_term_b_combo.currentText()),
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
			"compare_active": bool(self.compare_bundle is not None),
		}

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
			self.compare_gate_spin.setRange(1, max(1, int(self.study.cube.shape[0])))
			self.compare_gate_spin.setValue(max(1, int(self.study.cube.shape[0] // 2) + 1))
			if self.axis_companions:
				loaded = ", ".join(sorted(self.axis_companions.keys()))
				self._log(f"Series originales detectadas para comparación: {loaded}.")
			if not self.preset_patient_edit.text().strip():
				self._refresh_presets_for_current_patient()

			seg_method = str(self.seg_method.currentText())
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
			if parsed_rois and seg_method != "manual":
				seg_method = "manual"
				self.seg_method.setCurrentText("manual")
				self._log("Se detectaron ROIs manuales: cambiando Segmentación a manual.")
			if seg_method == "manual" and not parsed_rois:
				QMessageBox.warning(self, "SINCRO", "Modo manual activo pero no hay ROIs definidos. Dibujá ROI o cambiá a auto/threshold.")
				return
			manual_rois = parsed_rois if seg_method == "manual" else None
			seg_payload = {
				"study": study_sig,
				"method": seg_method,
				"threshold": round(float(self.threshold_spin.value()), 5),
				"sigma": round(float(self.sigma_spin.value()), 5),
				"manual_rois": self._serialize_manual_rois(manual_rois or {}),
			}
			seg_sig = self._hash_payload(seg_payload)
			if self.seg is None or seg_sig != self._cache_seg_sig:
				t_stage = perf_counter()
				self._set_progress(30, "Segmentando miocardio...")
				self.seg = segment_myocardium(
					self.study.cube,
					method=seg_method,
					threshold_frac=float(self.threshold_spin.value()),
					smooth_sigma=float(self.sigma_spin.value()),
					manual_rois=manual_rois,
				)
				self._cache_seg_sig = seg_sig
				self._cache_phase_sig = ""
				self._invalidate_output_cache()
				self._log_timing_if_slow("Segmentación", t_stage)
			else:
				self._set_progress(30, "Segmentación sin cambios (cache)...")
				self._log("Cache: segmentación reutilizada.")

			phase_payload = {
				"seg": self._cache_seg_sig,
				"harmonics": int(self.harmonics_spin.value()),
				"amp_filter": round(float(self.phase_threshold_spin.value()), 5),
				"normalize_reference": bool(self.normalize_check.isChecked()),
			}
			phase_sig = self._hash_payload(phase_payload)
			if self.phase_result is None or phase_sig != self._cache_phase_sig:
				t_stage = perf_counter()
				self._set_progress(50, "Análisis de fase...")
				self.phase_result = phase_analysis(
					self.study.cube,
					self.seg.mask,
					harmonics=int(self.harmonics_spin.value()),
					amplitude_threshold_frac=float(self.phase_threshold_spin.value()),
					normalize_reference=self.normalize_check.isChecked(),
				)
				self._set_progress(65, "Métricas y segmentos AHA...")
				self.metrics = calculate_phase_metrics(self.phase_result.phases_deg)
				self.aha = map_to_17_segments(self.seg)
				self.phase_by_seg = phase_by_segment(self.phase_result.phase_map, self.aha)
				self.territory = territory_analysis(self.phase_by_seg)
				self._cache_phase_sig = phase_sig
				self._invalidate_output_cache()
				self._log_timing_if_slow("Análisis de fase + métricas AHA", t_stage)
			else:
				self._set_progress(65, "Fase/métricas sin cambios (cache)...")
				self._log("Cache: fase, métricas y segmentación AHA reutilizadas.")
				if self.metrics is None:
					self.metrics = calculate_phase_metrics(self.phase_result.phases_deg)
				if self.aha is None:
					self.aha = map_to_17_segments(self.seg)
				if self.phase_by_seg is None:
					self.phase_by_seg = phase_by_segment(self.phase_result.phase_map, self.aha)
				if self.territory is None:
					self.territory = territory_analysis(self.phase_by_seg)
			preferred_gate_idx = int(self.study.cube.shape[0] // 2)
			preferred_slice_idx = int(self.study.cube.shape[1] // 2)

			self._set_progress(75, "Preparando cine...")
			self.primary_manual_rois_text = self._format_manual_rois(manual_rois or {}) if manual_rois else ""
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
			self._load_previews()
			self._log_timing_if_slow("Carga de previews", t_stage)
			self._select_tab_by_title("histograma")
			self._set_progress(100, "Procesamiento completo")
			self._log_timing_if_slow("Proceso total", t_total)
			self.statusBar().showMessage("Procesamiento completo")
		except Exception as exc:
			self._set_progress(0, "Error")
			self.statusBar().showMessage("Error")
			QMessageBox.critical(self, "Error de procesamiento", str(exc))
			self._log(f"[ERROR] {exc}")

	def _set_progress(self, value: int, label: str = ""):
		self._progress_bar.setValue(value)
		if label:
			self._progress_bar.setFormat(label)
			self.statusBar().showMessage(label)
		QApplication.processEvents()

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

		myocardial_ml = float(np.count_nonzero(self.seg.mask)) * voxel_ml

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

		cube = np.asarray(self.study.cube, dtype=np.float64)
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
		gate_cavity_area = np.zeros((n_gates,), dtype=np.float64)

		for s in valid_s:
			cy0 = float(centers[s, 0])
			cx0 = float(centers[s, 1])
			ro0 = float(outer[s])
			r_line = np.linspace(0.0, ro0 * 1.1, int(ro0 * 2) + 4)
			for g in range(n_gates):
				img = cube[g, s]
				d0 = np.sqrt((ys - cy0) ** 2 + (xs - cx0) ** 2)
				ring = (d0 >= ro0 * 0.5) & (d0 <= ro0)
				peak = float(np.percentile(img[ring], 80)) if np.any(ring) else 0.0
				if peak <= 0.0:
					continue
				thr = cavity_frac * peak
				# Recentrado por gate: centroide de baja actividad cerca del centro.
				low = (d0 <= ro0 * 0.7) & (img < thr)
				if np.count_nonzero(low) >= 3:
					yy_l, xx_l = np.nonzero(low)
					cyg = float(yy_l.mean())
					cxg = float(xx_l.mean())
				else:
					cyg, cxg = cy0, cx0
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
				gate_cavity_area[g] += float(0.5 * np.sum(r_endo ** 2) * dtheta)

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
		phase = self._phase_label_from_path(path_txt, "Estudio")
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

	def _refresh_summary(self):
		if self.study is None or self.metrics is None:
			return

		vol = self._compute_volumes_ml()
		ef = self._estimate_lv_ef_preliminary()
		ctx = self._study_context(
			path_override=str(getattr(self, "_output_study_path_override", "") or self.file_edit.text().strip()),
			study_obj=self.study,
		)

		clinical = []
		clinical.append(f"Visualizando: {ctx['phase']}")
		clinical.append(f"Paciente: {ctx['patient_name']}  |  ID: {ctx['patient_id']}  |  Fecha: {ctx['study_date']}")
		clinical.append("")
		clinical.append("Resultado clínico")
		clinical.append(f"  Clasificación de disincronía: {self.metrics.get('classification')}")
		clinical.append(f"  Phase SD: {self.metrics.get('phase_sd')}°")
		clinical.append(f"  Bandwidth: {self.metrics.get('bandwidth')}°")
		clinical.append(f"  Entropy: {self.metrics.get('entropy')}")
		clinical.append("")

		# Comparación contra base de datos normal (por sexo/protocolo).
		sex = "male" if self.normal_sex_combo.currentText() == "Hombre" else "female"
		protocol = "stress" if self.normal_protocol_combo.currentText() == "Stress" else "rest"
		dataset = self.normal_db_combo.currentText()
		try:
			nd = normal_db.evaluate(
				float(self.metrics.get("phase_sd", 0.0)),
				float(self.metrics.get("bandwidth", 0.0)),
				dataset=dataset,
				sex=sex,
				protocol=protocol,
			)
			clinical.append(f"Vs DB normal [{dataset} · {self.normal_sex_combo.currentText()} · {self.normal_protocol_combo.currentText()}]")
			for mkey, mlabel in (("phase_sd", "PSD"), ("bandwidth", "BW")):
				m = nd["metrics"].get(mkey, {})
				if not m.get("available"):
					clinical.append(f"  {mlabel}: sin referencia en la DB")
					continue
				flag = "ANORMAL" if m["abnormal"] else "normal"
				zt = f"{m['z']:+.1f}" if m.get("z") is not None else "n/d"
				clinical.append(
					f"  {mlabel}: {m['value']:.1f}° | normal {m['mean']:.1f}±{m['sd']:.1f} | cutoff {m['cutoff']:.1f}° | z={zt} → {flag}"
				)
			clinical.append(f"  Disincronía vs DB: {'SÍ' if nd['dyssynchrony'] else 'no'}")
		except Exception:
			clinical.append("Vs DB normal: no disponible")
		clinical.append("")
		clinical.append("Volúmenes")
		if vol["myocardial_ml"] is not None:
			clinical.append(f"  Miocardio: {vol['myocardial_ml']:.2f} mL")
		if vol["cavity_ml"] is not None:
			clinical.append(f"  Cavidad: {vol['cavity_ml']:.2f} mL")
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
		clinical.append("FEVI preliminar")
		if ef.get("available"):
			clinical.append(f"  EDV: {float(ef['edv_ml']):.2f} mL (gate {int(ef['ed_gate'])})")
			clinical.append(f"  ESV: {float(ef['esv_ml']):.2f} mL (gate {int(ef['es_gate'])})")
			clinical.append(f"  SV: {float(ef['sv_ml']):.2f} mL")
			clinical.append(f"  FEVI: {float(ef['ef_pct']):.1f}%")
			clinical.append("  Nota: estimación preliminar de investigación.")
		else:
			clinical.append("  No disponible (segmentación/metadata insuficiente).")

		clinical.append("")
		clinical.append("Territorios coronarios")
		for name, data in self.territory.items():
			clinical.append(f"  {name}: mean={data['mean']:.1f}°, SD={data['std']:.1f}°, n={data['n']}")

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
		technical.append(f"  Amp filter: {self.phase_threshold_spin.value():.2f}")
		technical.append(f"  Normalize reference: {'sí' if self.normalize_check.isChecked() else 'no'}")
		if vol["voxel_ml"] is not None:
			technical.append(f"  Volumen voxel: {vol['voxel_ml']:.4f} mL")
		technical.append("")
		technical.append("Métricas técnicas")
		for key in ["mean_phase", "phase_sd", "bandwidth", "entropy", "asynchrony_index", "peak_phase", "peak_width", "latest_activation_phase", "classification"]:
			technical.append(f"  {key}: {self.metrics.get(key)}")

		self.summary_clinical.setPlainText("\n".join(clinical))
		self.summary_technical.setPlainText("\n".join(technical))

	def _write_outputs(self):
		if self.study is None or self.phase_result is None:
			return

		mid_slice = self.study.cube.shape[1] // 2
		mid_gate = self.study.cube.shape[0] // 2
		frame = self.study.cube[mid_gate, mid_slice]
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
			"ventriculograma": {
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
			"polar_perfusion_directa": {
				**base_payload,
				"rotation": int(self.polar_rotation_spin.value()),
				"cmap_polar_perf": cmap_polar_perf,
			},
			"polar_cine_montaje": {
				**base_payload,
				"rotation": int(self.polar_rotation_spin.value()),
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

		advanced_mode = bool(self.advanced_mode_enabled)
		if not advanced_mode:
			for heavy_tab in (
				"comparacion_ejes",
				"curva_fevi",
				"ventriculograma",
				"bullseye_directo",
				"polar_perfusion_directa",
				"polar_cine_montaje",
			):
				need_tab_render[heavy_tab] = False
			for fname in (
				"ejes_ortogonales.png",
				"curva_tac.png",
				"comparacion_ejes.png",
				"curva_fevi.png",
				"ventriculograma.png",
				"bullseye_directo.png",
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
		save_polar_map(pm, os.path.join(self.output_dir, "polar_map.png"), dpi=150)
		plt.close(pm.fig)

		if self.compare_bundle is not None and self.compare_bundle.get("phase_by_seg") and self.study is not self.compare_bundle.get("study"):
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

		hfig = build_phase_histogram(self.phase_result.phases_deg, metrics=self.metrics, bins=72, title=f"Phase Histogram — {study_context_label}")
		save_histogram(hfig, os.path.join(self.output_dir, "histograma.png"), dpi=150)
		plt.close(hfig)
		_compose_polar_combo()
		_compose_delta_combo()

		cfig = build_clinical_phase_panel(
			self.phase_by_seg,
			self.phase_result.phases_deg,
			metrics=self.metrics,
			cmap_name=cmap_polar_clinico,
			title=f"Panel polar clínico (histograma + fase) — {study_context_label}",
		)
		save_clinical_phase_panel(cfig, os.path.join(self.output_dir, "polar_clinico.png"), dpi=150)
		plt.close(cfig)

		if not advanced_mode:
			self._log("Modo básico: se omite render avanzado (ejes, panel funcional, perfusión directa y cine polar).")
			return

		def _oriented_axes_views(gate_index: int):
			vol_gate = self.study.cube[int(gate_index)].astype(np.float64)
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

			# Convención visual clínica solicitada:
			# - HLA vertical con cara inferior hacia la izquierda.
			# - VLA con cara inferior hacia abajo.
			hla_view_local = np.fliplr(np.rot90(_norm(hla_local), k=1))
			vla_view_local = np.flipud(np.rot90(_norm(vla_local), k=-1))
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

		sa, hla_view, vla_view, hla_original_mid, vla_original_mid = _oriented_axes_views(mid_gate)

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
		axes2[1].imshow(hla_view, cmap=cmap_axes, aspect="auto")
		axes2[1].set_title("HLA (horizontal long axis)")
		_annotate_axis(axes2[1], "BASE", "APEX", "ANT", "INF")
		axes2[2].imshow(vla_view, cmap=cmap_axes, aspect="auto")
		axes2[2].set_title("VLA (vertical long axis)")
		_annotate_axis(axes2[2], "BASE", "APEX", "SEP", "LAT")
		if hla_original_mid:
			axes2[1].text(0.03, 0.05, "ORIGINAL", transform=axes2[1].transAxes, fontsize=8, color="#ffe082", fontweight="bold")
		if vla_original_mid:
			axes2[2].text(0.03, 0.05, "ORIGINAL", transform=axes2[2].transAxes, fontsize=8, color="#ffe082", fontweight="bold")
		fig2.suptitle(f"Ejes cardíacos ortogonales — Gate {mid_gate + 1} — {study_context_label}", fontsize=12, fontweight="bold")
		fig2.tight_layout()
		fig2.savefig(os.path.join(self.output_dir, "ejes_ortogonales.png"), dpi=150, bbox_inches="tight")
		plt.close(fig2)

		if need_tab_render.get("comparacion_ejes", True):
			self._write_compare_axes_panel(cmap_compare=cmap_compare, build_cine=False)
		else:
			self._log("Cache tab: comparacion_ejes sin cambios, se omite regeneración.")

		ef = self._estimate_lv_ef_preliminary()
		n_gates = int(self.study.cube.shape[0])
		if ef.get("available"):
			ed_gate = max(0, min(n_gates - 1, int(ef["ed_gate"]) - 1))
			es_gate = max(0, min(n_gates - 1, int(ef["es_gate"]) - 1))
		else:
			ed_gate = mid_gate
			es_gate = (mid_gate + max(1, n_gates // 2)) % n_gates

		style_name = str(self.visual_style_combo.currentText()).strip().lower()
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
		if style_name not in style_catalog:
			style_name = "clinico"
		style = style_catalog.get(style_name, style_catalog["clinico"])

		sa_ed, hla_ed, vla_ed, _ed_hla_original, _ed_vla_original = _oriented_axes_views(ed_gate)
		sa_es, hla_es, vla_es, _es_hla_original, _es_vla_original = _oriented_axes_views(es_gate)

		fig4, axes4 = plt.subplots(2, 3, figsize=(14, 8.2))
		for ax in axes4.ravel():
			ax.set_xticks([])
			ax.set_yticks([])

		axes4[0, 0].imshow(sa_ed, cmap=cmap_panel_axes)
		axes4[0, 0].set_title(f"A) ED - SHORT AXIS (Gate {ed_gate + 1})", fontsize=10)
		_annotate_axis(axes4[0, 0], "ANT", "INF", "SEP", "LAT")
		axes4[0, 1].imshow(hla_ed, cmap=cmap_panel_axes, aspect="auto")
		axes4[0, 1].set_title("A) ED - HORIZONTAL AXIS (HLA)", fontsize=10)
		_annotate_axis(axes4[0, 1], "BASE", "APEX", "ANT", "INF")
		axes4[0, 2].imshow(vla_ed, cmap=cmap_panel_axes, aspect="auto")
		axes4[0, 2].set_title("A) ED - VERTICAL AXIS (VLA)", fontsize=10)
		_annotate_axis(axes4[0, 2], "BASE", "APEX", "SEP", "LAT")

		axes4[1, 0].imshow(sa_es, cmap=cmap_panel_axes)
		axes4[1, 0].set_title(f"B) ES - SHORT AXIS (Gate {es_gate + 1})", fontsize=10)
		_annotate_axis(axes4[1, 0], "ANT", "INF", "SEP", "LAT")
		axes4[1, 1].imshow(hla_es, cmap=cmap_panel_axes, aspect="auto")
		axes4[1, 1].set_title("B) ES - HORIZONTAL AXIS (HLA)", fontsize=10)
		_annotate_axis(axes4[1, 1], "BASE", "APEX", "ANT", "INF")
		axes4[1, 2].imshow(vla_es, cmap=cmap_panel_axes, aspect="auto")
		axes4[1, 2].set_title("B) ES - VERTICAL AXIS (VLA)", fontsize=10)
		_annotate_axis(axes4[1, 2], "BASE", "APEX", "SEP", "LAT")

		fig4.suptitle(
			f"Panel clínico por convención (A=diástole, B=sístole) — SA/HLA/VLA — {study_context_label}",
			fontsize=13,
			fontweight="bold",
		)
		fig4.tight_layout()
		fig4.savefig(os.path.join(self.output_dir, "panel_clinico_convencion.png"), dpi=150, bbox_inches="tight")
		plt.close(fig4)

		fig3, ax3 = plt.subplots(figsize=(10, 4))
		mean_per_gate = np.array([float(self.study.cube[gi][self.seg.mask].mean()) for gi in range(self.study.cube.shape[0])])
		mean_per_gate = mean_per_gate / (mean_per_gate.max() + 1e-8)
		ax3.plot(np.arange(self.study.cube.shape[0]), mean_per_gate, "o-", color="#2c7fb8")
		ax3.set_title("Curva de actividad miocárdica por gate")
		ax3.set_xlabel("Gate")
		ax3.set_ylabel("Intensidad normalizada")
		ax3.grid(True, alpha=0.3)
		fig3.tight_layout()
		fig3.savefig(os.path.join(self.output_dir, "curva_tac.png"), dpi=150, bbox_inches="tight")
		plt.close(fig3)

		if need_tab_render.get("curva_fevi", True):
			fig_ef, ax_ef = plt.subplots(figsize=(10, 4.4), facecolor=style["fig_bg"])
			ax_ef.set_facecolor(style["ax_bg"])
			gate_axis = np.arange(self.study.cube.shape[0]) + 1
			t_pct = np.linspace(0.0, 100.0, self.study.cube.shape[0], endpoint=False)
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
				ax_ef.set_title(f"Curva FEVI preliminar por gate — FEVI {float(ef.get('ef_pct', 0.0)):.1f}%", color=style["fg"], fontsize=12, fontweight="bold")
				ax_ef.set_ylabel("Volumen estimado (mL)", color=style["fg"])
				ax_ef.tick_params(axis="x", colors=style["subtle"])
				ax_ef.tick_params(axis="y", colors=style["vol"])
				ax_ef.grid(True, color=style["grid"], alpha=0.45)
				ax_ef.set_xlabel("Gate", color=style["subtle"])
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
		ax_curve = fig_v.add_subplot(gs[0:2, 2:4])
		ax_metrics = fig_v.add_subplot(gs[2, 2:4])

		for ax in [ax_ed_sa, ax_es_sa, ax_ed_hla, ax_es_hla, ax_phase, ax_amp, ax_curve, ax_metrics]:
			ax.set_facecolor(style["ax_bg"])
			for spine in ax.spines.values():
				spine.set_color(style["grid"])

		for ax in [ax_ed_sa, ax_es_sa, ax_ed_hla, ax_es_hla, ax_phase, ax_amp]:
			ax.set_xticks([])
			ax.set_yticks([])

		ax_ed_sa.imshow(sa_ed, cmap=cmap_panel_axes)
		ax_ed_sa.set_title(f"ED SA (gate {ed_gate + 1})", color=style["fg"], fontsize=9)
		ax_es_sa.imshow(sa_es, cmap=cmap_panel_axes)
		ax_es_sa.set_title(f"ES SA (gate {es_gate + 1})", color=style["fg"], fontsize=9)
		ax_ed_hla.imshow(hla_ed, cmap=cmap_panel_axes, aspect="auto")
		ax_ed_hla.set_title("ED HLA", color=style["fg"], fontsize=9)
		ax_es_hla.imshow(hla_es, cmap=cmap_panel_axes, aspect="auto")
		ax_es_hla.set_title("ES HLA", color=style["fg"], fontsize=9)

		from viz.colormaps import phase_to_rgb

		phase_mid = np.asarray(self.phase_result.phase_map[mid_slice], dtype=np.float64)
		amp_mid = np.asarray(self.phase_result.amplitude_map[mid_slice], dtype=np.float64)
		phase_show = np.where(np.isfinite(phase_mid), phase_mid, 0.0)
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
			f"Clasificación: {self.metrics.get('classification')}",
			f"Phase SD: {float(self.metrics.get('phase_sd', np.nan)):.1f}°",
			f"Bandwidth: {float(self.metrics.get('bandwidth', np.nan)):.1f}°",
			f"Entropy: {float(self.metrics.get('entropy', np.nan)):.3f}",
		]
		if ef.get("available"):
			metrics_lines.extend([
				f"EDV: {float(ef.get('edv_ml', np.nan)):.1f} mL",
				f"ESV: {float(ef.get('esv_ml', np.nan)):.1f} mL",
				f"FEVI: {float(ef.get('ef_pct', np.nan)):.1f}%",
			])
		else:
			metrics_lines.append("FEVI: no disponible")
		ax_metrics.text(
			0.98,
			0.96,
			"\n".join(metrics_lines),
			transform=ax_metrics.transAxes,
			va="top",
			ha="right",
			color=style["fg"],
			fontsize=8.6,
			bbox=dict(boxstyle="round,pad=0.45", facecolor=style["ax_bg"], edgecolor=style["grid"], alpha=0.95),
		)

		fig_v.suptitle(
			f"Panel funcional gated SPECT — {study_context_label} (estilo clínico: {self.visual_style_combo.currentText()})",
			color=style["fg"],
			fontsize=13,
			fontweight="bold",
		)
		fig_v.savefig(os.path.join(self.output_dir, "ventriculograma.png"), dpi=155, bbox_inches="tight", facecolor=fig_v.get_facecolor())
		plt.close(fig_v)

		# Bull's eye directo de perfusión (colores de intensidad), inspirado en consolas clínicas.
		from matplotlib.patches import Circle, Wedge
		seg_map = np.asarray(self.aha.segment_map, dtype=np.int32)
		mid_gate_cube = np.asarray(self.study.cube[mid_gate], dtype=np.float64)
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
		fig_b.savefig(os.path.join(self.output_dir, "bullseye_directo.png"), dpi=170, bbox_inches="tight", facecolor=fig_b.get_facecolor())
		plt.close(fig_b)

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

		if len(profiles) >= 2 and (need_tab_render.get("polar_perfusion_directa", True) or need_tab_render.get("polar_cine_montaje", True)):
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
			polar_map_smooth = gaussian_filter(polar_map, sigma=(2.0, 1.2))

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
				(axes_pp[1], cart_smooth, "Perfusión polar directa (suavizado)"),
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
			fig_pp.savefig(os.path.join(self.output_dir, "polar_perfusion_directa.png"), dpi=185, bbox_inches="tight", facecolor=fig_pp.get_facecolor())
			plt.close(fig_pp)

			# Cine polar gatillado por gate: genera GIF y un montaje estático para preview/PDF.
			try:
				from PIL import Image
			except Exception:
				Image = None

			def _render_gate_frame(study_obj, seg_obj, apex_order, gate_index: int, label_text: str):
				gate_cube = np.asarray(study_obj.cube[int(gate_index)], dtype=np.float64)
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
				pm_g = gaussian_filter(pm_g, sigma=(1.7, 1.1))
				cart_g = _polar_to_cartesian(pm_g)
				fig_g, ax_g = plt.subplots(1, 1, figsize=(5.2, 5.2), facecolor=perf_bg)
				ax_g.set_facecolor(perf_bg)
				ax_g.set_aspect("equal")
				ax_g.set_xticks([])
				ax_g.set_yticks([])
				ax_g.imshow(cart_g, cmap=cmap_polar_perf, vmin=0.0, vmax=1.0)
				_annotate_polar_guides(ax_g, int(cart_g.shape[0]))
				ax_g.set_title(f"{label_text} gate {gate_index + 1}/{int(study_obj.cube.shape[0])}", color=perf_fg, fontsize=10, fontweight="bold")
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
			frame_count = int(self.study.cube.shape[0])
			if self.compare_bundle is not None and self.compare_bundle.get("study") is not None:
				frame_count = min(frame_count, int(self.compare_bundle["study"].cube.shape[0]))
				compare_apex_to_base = list(getattr(self.compare_bundle.get("aha"), "apex_to_base_order", []) or [])
				if not compare_apex_to_base:
					compare_apex_to_base = [int(s) for s in np.where(np.asarray(self.compare_bundle["seg"].mask).reshape(self.compare_bundle["seg"].mask.shape[0], -1).any(axis=1))[0].tolist()]
			else:
				compare_apex_to_base = []

			for g in range(frame_count):
				p_frame, p_pm = _render_gate_frame(self.study, self.seg, apex_to_base, g, primary_phase_label)
				if p_frame is None:
					continue
				primary_frames.append(p_frame)
				if self.compare_bundle is not None and self.compare_bundle.get("study") is not None:
					r_frame, r_pm = _render_gate_frame(self.compare_bundle["study"], self.compare_bundle["seg"], compare_apex_to_base, g, compare_phase_label)
					if r_frame is None:
						compare_frames.append(p_frame)
					else:
						gap = np.full((p_frame.shape[0], 28, 3), 12, dtype=np.uint8)
						panels = [p_frame, gap, r_frame]
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
						compare_frames.append(np.concatenate(panels, axis=1))
				else:
					compare_frames.append(p_frame)

			gate_frames = compare_frames

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
				fig_m.savefig(os.path.join(self.output_dir, "polar_cine_montaje.png"), dpi=160, bbox_inches="tight", facecolor=fig_m.get_facecolor())
				plt.close(fig_m)

	def _write_compare_axes_panel(self, cmap_compare: str = "hot", build_cine: bool | None = None):
		if self.study is None or self.seg is None:
			return
		import matplotlib.pyplot as plt
		fast_mode = bool(self.compare_fast_drag_check.isChecked() and self.compare_interactive_fast_mode)
		render_dpi = 130 if fast_mode else 240
		interp_mode = "nearest" if fast_mode else "lanczos"

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

		def _extract_rows(study, seg, path_text: str, gate_override: int | None = None):
			gate = max(0, min(int(study.cube.shape[0]) - 1, int(self.compare_gate_spin.value()) - 1 if gate_override is None else int(gate_override)))
			vol_gate = np.asarray(study.cube[int(gate)], dtype=np.float64)
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
			primary_gate, primary_rows, primary_phase = _extract_rows(self.study, self.seg, self.file_edit.text().strip(), gate_override=primary_gate_override)
			if self.compare_bundle is not None:
				secondary_gate, secondary_rows, secondary_phase = _extract_rows(
					self.compare_bundle["study"],
					self.compare_bundle["seg"],
					str(self.compare_bundle.get("path", self.compare_label or "comparacion")),
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
			fig.suptitle(
				f"Comparativa de ejes clínica — 16 cortes por eje — {gate_text} — Máscara {mask_txt} — Top {int(self.compare_window_high_slider.value())}% / Base {int(self.compare_window_low_slider.value())}%",
				fontsize=12,
				fontweight="bold",
				color="#f8fafc",
			)
			fig.tight_layout(rect=(0.02, 0.02, 1, 0.94))
			return fig

		fig = _build_compare_figure()
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
		if name == "polar_cine_montaje":
			self._load_polar_cine_preview()
			return
		if name == "comparacion_ejes":
			self._load_compare_axes_preview()
			return
		path = os.path.join(self.output_dir, f"{name}.png")
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

	def _load_polar_cine_preview(self):
		name = "polar_cine_montaje"
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
		name = "comparacion_ejes"
		label = self.preview_labels[name]
		png_path = os.path.join(self.output_dir, "comparacion_ejes.png")
		movie = self.preview_movies.pop(name, None)
		if movie is not None:
			movie.stop()
			label.clear()
			label.setMovie(None)
		if self.compare_axes_cine_check.isChecked() and self.compare_axes_preview_frames:
			self._set_compare_axes_memory_frame(self.compare_axes_preview_index)
			self._update_compare_axes_toggle_text(enabled=True)
			return
		if os.path.exists(png_path):
			pix = QPixmap(png_path)
			self.preview_pixmaps[name] = pix
			self.preview_base_sizes[name] = pix.size()
			self._update_compare_axes_toggle_text(enabled=False)
			self._apply_preview_zoom(name)
		else:
			self.preview_pixmaps.pop(name, None)
			self.preview_base_sizes.pop(name, None)
			self._update_compare_axes_toggle_text(enabled=False)
			label.setText("Sin comparativa")

	def _generate_pdf_report(self):
		if self.study is None or self.seg is None or self.metrics is None or self.territory is None:
			return
		pdf_path = os.path.join(self.output_dir, "informe_sincro.pdf")
		params = {
			"threshold": float(self.threshold_spin.value()),
			"smooth_sigma": float(self.sigma_spin.value()),
			"harmonics": int(self.harmonics_spin.value()),
			"amp_filter": float(self.phase_threshold_spin.value()),
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
		}
		vol = self._compute_volumes_ml()
		ef = self._estimate_lv_ef_preliminary()
		try:
			generate_report(
				output_pdf=pdf_path,
				output_dir=self.output_dir,
				study=self.study,
				seg=self.seg,
				metrics=self.metrics,
				territory=self.territory,
				processing_params=params,
				volumes=vol,
				ef=ef,
			)
			self._log(f"PDF actualizado: {pdf_path}")
		except Exception as exc:
			self._log(f"[WARN] No se pudo generar PDF integrado: {exc}")

	def _ensure_reports_generated(self):
		if self.study is None or self.seg is None or self.metrics is None or self.territory is None:
			QMessageBox.information(self, "SINCRO", "Primero procesá un estudio para generar informes.")
			return False
		self._set_progress(92, "Generando informe PDF...")
		self._generate_pdf_report()
		self._set_progress(100, "Informes listos")
		return True

	def _apply_preview_zoom(self, name: str):
		label = self.preview_labels[name]
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
			zoom = max(0.20, min(4.00, self.preview_zoom.get(name, 1.0)))
			w = max(1, int(base_size.width() * zoom))
			h = max(1, int(base_size.height() * zoom))
			movie.setScaledSize(QSize(w, h))
			label.setMinimumSize(w, h)
			label.resize(w, h)
			if name in self.preview_zoom_labels:
				self.preview_zoom_labels[name].setText(f"{int(zoom * 100)}%")
			return
		pix = self.preview_pixmaps.get(name)
		if pix is None or pix.isNull():
			label.setText("Sin imagen")
			return
		base_size = self.preview_base_sizes.get(name)
		if base_size is None or base_size.isEmpty():
			base_size = pix.size()
			self.preview_base_sizes[name] = base_size
		zoom = max(0.20, min(4.00, self.preview_zoom.get(name, 1.0)))
		w = max(1, int(base_size.width() * zoom))
		h = max(1, int(base_size.height() * zoom))
		scaled = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
		label.setPixmap(scaled)
		label.setMinimumSize(scaled.size())
		label.resize(scaled.size())
		if name in self.preview_zoom_labels:
			self.preview_zoom_labels[name].setText(f"{int(zoom * 100)}%")

	def _zoom_preview(self, name: str, delta: float):
		current = self.preview_zoom.get(name, 1.0)
		self._set_preview_zoom(name, current + delta)

	def _set_preview_zoom(self, name: str, value: float):
		self.preview_zoom[name] = max(0.20, min(4.00, float(value)))
		self._apply_preview_zoom(name)

	def _toggle_polar_cine_preview(self):
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
			self.polar_cine_toggle_btn.setText("Play/Pause")
			return
		if not self.polar_cine_preview_frames:
			self.polar_cine_toggle_btn.setText("Play/Pause")
			return
		if self.polar_cine_playing:
			self.polar_cine_toggle_btn.setText("Pause")
		else:
			self.polar_cine_toggle_btn.setText("Play")

	def _set_polar_cine_memory_frame(self, index: int):
		if not self.polar_cine_preview_frames:
			return
		idx = max(0, min(int(index), len(self.polar_cine_preview_frames) - 1))
		self.polar_cine_preview_index = idx
		pix = self.polar_cine_preview_frames[idx]
		self.preview_pixmaps["polar_cine_montaje"] = pix
		self.preview_base_sizes["polar_cine_montaje"] = pix.size()
		self._apply_preview_zoom("polar_cine_montaje")

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

	def _rebuild_tabs_for_mode(self):
		current_title = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() > 0 else ""
		while self.tabs.count() > 0:
			self.tabs.removeTab(0)
		order = list(self._basic_tab_order)
		if self.advanced_mode_enabled:
			order.extend(self._advanced_extra_tab_order)
		for name in order:
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
		self.advanced_mode_enabled = not bool(self.advanced_mode_enabled)
		if self.advanced_mode_enabled:
			self.advanced_toggle_btn.setText("BÁSICO")
			self._log("Modo avanzado activado: se habilitan paneles y render pesado bajo demanda.")
		else:
			self.advanced_toggle_btn.setText("AVANZADO...")
			self._log("Modo básico activado: foco en asincronía con render rápido.")
		self._rebuild_tabs_for_mode()
		if self.study is not None and self.phase_result is not None:
			must_render = self.advanced_mode_enabled and not all(
				os.path.exists(os.path.join(self.output_dir, name))
				for name in (
					"polar_perfusion_directa.png",
					"polar_cine_montaje.png",
					"comparacion_ejes.png",
					"ventriculograma.png",
					"bullseye_directo.png",
				)
			)
			# Si hay comparación cargada, al entrar a avanzado hay que garantizar
			# también las salidas avanzadas del bundle secundario (reposo) para evitar
			# pestañas vacías o mostrando solo esfuerzo.
			must_render_compare = False
			if self.advanced_mode_enabled and self.compare_bundle is not None:
				must_render_compare = not all(
					os.path.exists(os.path.join(self.compare_output_dir, name))
					for name in (
						"polar_perfusion_directa.png",
						"polar_cine_montaje.png",
						"comparacion_ejes.png",
						"ventriculograma.png",
						"bullseye_directo.png",
					)
				)
			if must_render:
				self._set_progress(78, "Actualizando modo de visualización...")
				self._write_outputs()
			if self.advanced_mode_enabled and self.compare_bundle is not None and (must_render or must_render_compare):
				self._set_progress(84, "Actualizando comparación en pestañas avanzadas...")
				self.statusBar().showMessage("Regenerando avanzado (esfuerzo + reposo)...")
				self._log("Modo avanzado: regenerando paneles de esfuerzo + reposo.")
				self._write_outputs_for_bundle(self.compare_bundle, self.compare_output_dir)
				# Re-generar principal con compare activo para mantener mapas delta y
				# luego recomponer vistas lado a lado.
				self._write_outputs()
				left_label = os.path.splitext(os.path.basename(self.file_edit.text().strip()))[0] or "Actual"
				right_label = self.compare_label or "Comparación"
				self._compose_dual_tab_images(left_label, right_label)
			self._load_previews()
			self._set_progress(100, "Modo actualizado")

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
			self.compare_axes_cine_toggle_btn.setText("Play/Pause")
			return
		if not self.compare_axes_preview_frames:
			self.compare_axes_cine_toggle_btn.setText("Play/Pause")
			return
		if self.compare_axes_playing:
			self.compare_axes_cine_toggle_btn.setText("Pause")
		else:
			self.compare_axes_cine_toggle_btn.setText("Play")

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
			"5) polar_perfusion_directa: perfusión polar continua (apex centro, base borde), complementa fase.\n"
			"6) bullseye_directo: resumen segmentario AHA rápido de intensidad regional.\n"
			"7) polar_cine_montaje: dinámica gate-a-gate del patrón polar.\n\n"
			"Fórmulas clave:\n"
			"• Δsigned = ((φ_esfuerzo - φ_reposo + 180) mod 360) - 180\n"
			"• Δabs = |Δsigned|\n"
			"• PSD/BW/Entropy: a mayor dispersión, mayor asincronía probable.\n\n"
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
			"• Entropy: ~3.2\n\n"
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
		path, _ = QFileDialog.getOpenFileName(
			self,
			"Seleccionar estudio de comparación (ej: REST)",
			os.path.dirname(self.file_edit.text().strip() or self.output_dir),
			"DICOM (*.dcm *.DCM *.ima *.IMA);;Todos (*.*)",
		)
		if not path:
			return
		self._load_compare_study_from_path(path)

	def load_one_or_two_studies(self):
		paths, _ = QFileDialog.getOpenFileNames(
			self,
			"Seleccionar uno o dos estudios (stress/rest)",
			os.path.dirname(self.file_edit.text().strip() or self.output_dir),
			"DICOM (*.dcm *.DCM *.ima *.IMA);;Todos (*.*)",
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

		def _score_stress(path_text: str) -> int:
			u = os.path.basename(path_text).upper()
			score = 0
			if "STRESS" in u:
				score += 2
			if "REST" in u:
				score -= 1
			return score

		primary_path = max(valid_paths, key=_score_stress)
		compare_path = valid_paths[0] if valid_paths[1] == primary_path else valid_paths[1]
		self.file_edit.setText(primary_path)
		self.process_current()
		if self.study is not None and self.metrics is not None:
			self._load_compare_study_from_path(compare_path)

	def _process_secondary_bundle(self, path: str) -> dict:
		comp_study = dicom_loader.load(path, verbose=False)
		comp_axis = self._load_axis_companions(path)
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
			comp_study.cube,
			method=seg_method,
			threshold_frac=float(self.threshold_spin.value()),
			smooth_sigma=float(self.sigma_spin.value()),
			manual_rois=manual_rois,
		)
		comp_phase = phase_analysis(
			comp_study.cube,
			comp_seg.mask,
			harmonics=int(self.harmonics_spin.value()),
			amplitude_threshold_frac=float(self.phase_threshold_spin.value()),
			normalize_reference=self.normalize_check.isChecked(),
		)
		comp_metrics = calculate_phase_metrics(comp_phase.phases_deg)
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
			"metrics": comp_metrics,
			"aha": comp_aha,
			"phase_by_seg": comp_phase_by_seg,
			"territory": comp_territory,
			"ef": comp_ef,
			"manual_rois_text": self.compare_manual_rois_text,
		}

	def _write_outputs_for_bundle(self, bundle: dict, target_dir: str):
		os.makedirs(target_dir, exist_ok=True)
		saved_output_dir = self.output_dir
		saved_study = self.study
		saved_axis = self.axis_companions
		saved_seg = self.seg
		saved_phase = self.phase_result
		saved_metrics = self.metrics
		saved_aha = self.aha
		saved_phase_by_seg = self.phase_by_seg
		saved_territory = self.territory
		saved_compare_bundle = self.compare_bundle
		saved_output_path_override = getattr(self, "_output_study_path_override", None)
		try:
			self.output_dir = target_dir
			self.study = bundle["study"]
			self.axis_companions = bundle["axis_companions"]
			self.seg = bundle["seg"]
			self.phase_result = bundle["phase_result"]
			self.metrics = bundle["metrics"]
			self.aha = bundle["aha"]
			self.phase_by_seg = bundle["phase_by_seg"]
			self.territory = bundle["territory"]
			# Al renderizar el bundle secundario, evitar compararlo consigo mismo.
			self.compare_bundle = None
			self._output_study_path_override = str(bundle.get("path", ""))
			self._write_outputs()
		finally:
			self.output_dir = saved_output_dir
			self.study = saved_study
			self.axis_companions = saved_axis
			self.seg = saved_seg
			self.phase_result = saved_phase
			self.metrics = saved_metrics
			self.aha = saved_aha
			self.phase_by_seg = saved_phase_by_seg
			self.territory = saved_territory
			self.compare_bundle = saved_compare_bundle
			self._output_study_path_override = saved_output_path_override

	def _compose_dual_tab_images(self, left_label: str, right_label: str):
		import matplotlib.pyplot as plt

		for name in self.preview_labels:
			if name in ("comparacion_stress_rest", "comparacion_ejes", "polar_cine_montaje"):
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
			fig.suptitle(f"Comparativa {name}", color="#f8fafc", fontsize=12, fontweight="bold")
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

	def _load_compare_study_from_path(self, path: str):
		try:
			self._set_progress(10, "Cargando y procesando estudio de comparación...")
			bundle = self._process_secondary_bundle(path)
			self.compare_bundle = bundle
			self.compare_metrics = bundle["metrics"]
			self.compare_ef = bundle["ef"]
			self.compare_label = bundle["label"]
			self._refresh_cine_source_selector()

			self._set_progress(75, "Generando salidas comparativas en todas las pestañas...")
			self._write_outputs_for_bundle(bundle, self.compare_output_dir)
			# Re-generar salidas del estudio principal con compare_bundle activo para
			# crear mapas delta (polar_map_Δsigned / polar_map_Δabs).
			self._write_outputs()
			left_label = os.path.splitext(os.path.basename(self.file_edit.text().strip()))[0] or "Actual"
			right_label = self.compare_label or "Comparación"
			self._compose_dual_tab_images(left_label, right_label)
			# polar_cine ya se genera compuesto dentro de _write_outputs cuando hay compare_bundle.
			# Evitamos recomponer de nuevo para no duplicar paneles (p.ej. Reposo repetido).
			self._write_compare_axes_panel(cmap_compare=str(self.compare_axes_cmap_combo.currentText()), build_cine=False)
			self._write_compare_stress_rest()
			self.dual_mode_active = True
			self._load_previews()
			self._refresh_summary()
			self._select_tab_by_title("histograma")
			self._set_progress(100, "Comparación lista")
			self._log(f"Comparación cargada: {self.compare_label}")
			self.statusBar().showMessage(f"Comparación cargada: {self.compare_label}")
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
			return self._estimate_lv_ef_preliminary()
		finally:
			self.study, self.seg = saved_study, saved_seg

	def _select_tab_by_title(self, title: str):
		for i in range(self.tabs.count()):
			if self.tabs.tabText(i) == title:
				self.tabs.setCurrentIndex(i)
				return

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
			("entropy", "Entropy", "menor = más sincrónico"),
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
