"""SINCRO - ui.main_window.

Ventana principal con controles de procesamiento y vista previa interactiva.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QIcon, QMovie, QPixmap
from PyQt6.QtWidgets import (
	QApplication,
	QFileDialog,
	QCheckBox,
	QComboBox,
	QDoubleSpinBox,
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
from core.aha_segments import map_to_17_segments, phase_by_segment, territory_analysis
from core.metrics import calculate_phase_metrics
from core.phase_analysis import phase_analysis
from core.segmentation import segment_myocardium
from report.report_generator import generate_report
from viz.histogram import build_phase_histogram, save_histogram
from viz.polar_map import build_polar_map, save_polar_map

from ui.cine_widget import CineWidget


class MainWindow(QMainWindow):
	def __init__(self, initial_path: str | None = None):
		super().__init__()
		self.setWindowTitle("GammaSync - Interfaz de procesado")
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
		self.preview_zoom: dict[str, float] = {}
		self.preview_pixmaps: dict[str, QPixmap] = {}
		self.preview_movies: dict[str, QMovie] = {}
		self.preview_zoom_labels: dict[str, QLabel] = {}
		self.polar_cine_toggle_btn: QToolButton | None = None

		self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output_demo")
		os.makedirs(self.output_dir, exist_ok=True)
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

		self.auto_run_check = QCheckBox("Procesar automáticamente al cargar")
		self.auto_run_check.setChecked(True)
		self.auto_run_check.setToolTip("Si está activo, el estudio se procesa apenas se carga con los parámetros actuales.")

		self.cmap_combo = QComboBox()
		self.cmap_combo.addItems(["hsv", "twilight", "twilight_shifted", "cool", "prism", "french"])
		self.cmap_combo.setCurrentText("french")

		self.visual_style_combo = QComboBox()
		self.visual_style_combo.addItems(["GammaSync", "QGS-like"])
		self.visual_style_combo.setCurrentText("QGS-like")

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

		self.export_polar_mp4_check = QCheckBox("Exportar polar cine MP4")
		self.export_polar_mp4_check.setChecked(True)

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
		controls_form.addRow(self.export_polar_mp4_check)
		controls_form.addRow(self.normalize_check)
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
		self.export_polar_mp4_check.setToolTip("Además del GIF, intenta exportar un MP4 del cine polar gatillado.")
		self.normalize_check.setToolTip("Resta una referencia global de fase para comparar estudios.")

		self._sidebar_layout.addWidget(controls_box)

		report_cmap_box = QGroupBox("Escalas informe (por imagen)")
		report_cmap_layout = QGridLayout(report_cmap_box)
		report_cmap_layout.setContentsMargins(6, 6, 6, 6)
		report_cmap_layout.setHorizontalSpacing(4)
		report_cmap_layout.setVerticalSpacing(4)

		def _mk_combo(options: list[str], current: str) -> QComboBox:
			cb = QComboBox()
			cb.addItems(options)
			cb.setCurrentText(current)
			return cb

		intensity_opts = ["hot", "inferno", "magma", "turbo", "viridis", "plasma", "gray", "bone", "cividis"]
		perfusion_opts = ["turbo", "inferno", "magma", "plasma", "viridis", "cividis", "hot"]
		amp_opts = ["turbo", "viridis", "plasma", "magma", "inferno", "cividis"]
		phase_opts = ["hsv", "twilight", "twilight_shifted", "cool", "prism", "french"]

		self.report_cmap_slices = _mk_combo(intensity_opts, "hot")
		self.report_cmap_axes = _mk_combo(intensity_opts, "hot")
		self.report_cmap_compare = _mk_combo(intensity_opts, "hot")
		self.report_cmap_panel_axes = _mk_combo(intensity_opts, "hot")
		self.report_cmap_phase = _mk_combo(phase_opts, "french")
		self.report_cmap_amp = _mk_combo(amp_opts, "turbo")
		self.report_cmap_bullseye = _mk_combo(perfusion_opts, "turbo")
		self.report_cmap_polar_perf = _mk_combo(perfusion_opts, "turbo")

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
		report_cmap_layout.addWidget(QLabel("amplitud"), 5, 0)
		report_cmap_layout.addWidget(self.report_cmap_amp, 5, 1)
		report_cmap_layout.addWidget(QLabel("bullseye_directo"), 6, 0)
		report_cmap_layout.addWidget(self.report_cmap_bullseye, 6, 1)
		report_cmap_layout.addWidget(QLabel("polar_perfusion_directa"), 7, 0)
		report_cmap_layout.addWidget(self.report_cmap_polar_perf, 7, 1)

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

		helper_box = QGroupBox("Ayuda rápida")
		helper_layout = QVBoxLayout(helper_box)
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
		self.docs_portal_btn = QPushButton("Portal docs")
		self.docs_portal_btn.clicked.connect(self.open_docs_portal)
		self.docs_portal_btn.setToolTip("Abre el portal de documentación HTML (índice de guías e instrucciones).")
		helper_layout.addWidget(self.docs_portal_btn)
		self._sidebar_layout.addWidget(helper_box)

		button_box = QGroupBox("Acciones")
		button_row = QGridLayout(button_box)
		button_row.setContentsMargins(6, 6, 6, 6)
		button_row.setHorizontalSpacing(4)
		button_row.setVerticalSpacing(4)
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
		self.open_polar_btn = QPushButton("Polar map")
		self.open_polar_btn.clicked.connect(self.open_polar_map)
		self.open_polar_btn.setToolTip("Abre la imagen del mapa polar del estudio procesado.")
		self.apply_roi_all_btn = QPushButton("Replicar ROI a todos")
		self.apply_roi_all_btn.clicked.connect(self.apply_current_roi_to_all_slices)
		self.apply_roi_all_btn.setToolTip("Copia el ROI del slice actual a todos los slices del volumen.")
		button_row.addWidget(self.process_btn, 0, 0)
		button_row.addWidget(self.auto_btn, 0, 1)
		button_row.addWidget(self.open_folder_btn, 1, 0)
		button_row.addWidget(self.open_pdf_btn, 1, 1)
		button_row.addWidget(self.open_polar_btn, 2, 0, 1, 2)
		self._sidebar_layout.addWidget(button_box)

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
		compare_gate_row = QHBoxLayout()
		compare_gate_row.addWidget(QLabel("Gate"))
		self.compare_gate_spin = QSpinBox()
		self.compare_gate_spin.setRange(1, 1)
		self.compare_gate_spin.setValue(1)
		self.compare_gate_spin.setToolTip("Gate usado en la lámina de comparación de ejes.")
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
		self.compare_slice_label = QLabel("50%")
		self.compare_slice_slider.valueChanged.connect(self._update_compare_slice_label)
		compare_slice_row.addWidget(QLabel("Corte anatómico"), 0, 0)
		compare_slice_row.addWidget(self.compare_slice_slider, 0, 1)
		compare_slice_row.addWidget(self.compare_slice_label, 0, 2)
		compare_layout.addLayout(compare_slice_row)
		self._update_compare_slice_label()
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
		preview_titles = {
			"slices_fase": "slices_fase",
			"polar_map": "polar_map",
			"polar_perfusion_directa": "polar_perfusion_directa",
			"polar_cine_montaje": "polar_cine_montaje",
			"histograma": "histograma",
			"ejes_ortogonales": "ejes_ortogonales",
			"comparacion_ejes": "comparacion_ejes",
			"curva_fevi": "curva_fevi",
			"curva_tac": "curva_tac",
			"ventriculograma": "panel_funcional_gated",
			"bullseye_directo": "bullseye_directo",
		}
		for name in [
			"slices_fase",
			"polar_map",
			"polar_perfusion_directa",
			"polar_cine_montaje",
			"histograma",
			"ejes_ortogonales",
			"comparacion_ejes",
			"curva_fevi",
			"curva_tac",
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
			label.setToolTip("Zoom con los botones +/- o 100% arriba de cada panel.")
			self.preview_labels[name] = label
			self.preview_zoom[name] = 1.0
			scroller = QScrollArea()
			scroller.setWidgetResizable(False)
			scroller.setWidget(label)
			tab_layout.addWidget(scroller)
			self.tabs.addTab(tab, preview_titles.get(name, name))
		self.cine = CineWidget()
		self.cine.roiEdited.connect(self._on_cine_roi_changed)
		self.cine.playStateChanged.connect(self._on_play_state_changed)
		self.cine.setToolTip("Reproducí el cine, hacé zoom y dibujá ROIs sobre la imagen.")
		right_splitter.addWidget(self.tabs)
		right_splitter.addWidget(self.cine)
		right_splitter.setStretchFactor(0, 3)
		right_splitter.setStretchFactor(1, 1)
		right_layout.addWidget(right_splitter)

		splitter.addWidget(left)
		splitter.addWidget(right)
		splitter.setStretchFactor(0, 1)
		splitter.setStretchFactor(1, 4)
		splitter.setSizes([220, 1340])
		right_splitter.setSizes([920, 140])

		layout = QVBoxLayout(central)
		layout.addWidget(splitter)

		self.statusBar().showMessage("Listo")
		self.cmap_combo.currentTextChanged.connect(self._on_phase_cmap_changed)
		self.preset_patient_edit.textChanged.connect(lambda _=None: self._refresh_presets_for_current_patient())
		self._on_phase_cmap_changed(self.cmap_combo.currentText())
		self._refresh_presets_for_current_patient()

		if initial_path:
			self.file_edit.setText(initial_path)
			if self.auto_run_check.isChecked():
				self.process_auto()
			else:
				self.process_current()

	def _log(self, message: str):
		self.log_box.append(message)

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
			"export_polar_mp4": bool(self.export_polar_mp4_check.isChecked()),
			"report_cmap_slices": str(self.report_cmap_slices.currentText()),
			"report_cmap_axes": str(self.report_cmap_axes.currentText()),
			"report_cmap_compare": str(self.report_cmap_compare.currentText()),
			"report_cmap_panel_axes": str(self.report_cmap_panel_axes.currentText()),
			"report_cmap_phase": str(self.report_cmap_phase.currentText()),
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
			if style_value == "Xeleris-like":
				style_value = "QGS-like"
			self.visual_style_combo.setCurrentText(style_value)
		if "polar_rotation_deg" in params:
			self.polar_rotation_spin.setValue(int(params["polar_rotation_deg"]))
		if "polar_cine_speed_ms" in params:
			self.polar_cine_speed_spin.setValue(int(params["polar_cine_speed_ms"]))
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
		sidebar.setMinimumWidth(210)
		sidebar.setMaximumWidth(280)
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

	def _parse_manual_rois(self) -> dict[int, tuple[float, float, float, float]]:
		rois: dict[int, tuple[float, float, float, float]] = {}
		for raw in self.manual_rois.toPlainText().splitlines():
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
		self.manual_rois.blockSignals(True)
		self.manual_rois.setPlainText(self._format_manual_rois(rois))
		self.manual_rois.blockSignals(False)
		self.cine.set_manual_rois(rois)
		if message:
			self._log(message)

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

	def process_current(self):
		path = self.file_edit.text().strip()
		if not path:
			QMessageBox.warning(self, "SINCRO", "Seleccioná un archivo DICOM primero.")
			return
		if not os.path.exists(path):
			QMessageBox.warning(self, "SINCRO", f"No existe el archivo:\n{path}")
			return

		try:
			self.statusBar().showMessage("Cargando estudio...")
			self._log(f"Cargando: {path}")
			self.study = dicom_loader.load(path, verbose=False)
			self.axis_companions = self._load_axis_companions(path)
			self.compare_gate_spin.setRange(1, max(1, int(self.study.cube.shape[0])))
			self.compare_gate_spin.setValue(max(1, int(self.study.cube.shape[0] // 2) + 1))
			if self.axis_companions:
				loaded = ", ".join(sorted(self.axis_companions.keys()))
				self._log(f"Series originales detectadas para comparación: {loaded}.")
			if not self.preset_patient_edit.text().strip():
				self._refresh_presets_for_current_patient()

			seg_method = str(self.seg_method.currentText())
			parsed_rois = self._parse_manual_rois()
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
			self.seg = segment_myocardium(
				self.study.cube,
				method=seg_method,
				threshold_frac=float(self.threshold_spin.value()),
				smooth_sigma=float(self.sigma_spin.value()),
				manual_rois=manual_rois,
			)

			self.phase_result = phase_analysis(
				self.study.cube,
				self.seg.mask,
				harmonics=int(self.harmonics_spin.value()),
				amplitude_threshold_frac=float(self.phase_threshold_spin.value()),
				normalize_reference=self.normalize_check.isChecked(),
			)

			self.metrics = calculate_phase_metrics(self.phase_result.phases_deg)
			self.aha = map_to_17_segments(self.seg)
			self.phase_by_seg = phase_by_segment(self.phase_result.phase_map, self.aha)
			self.territory = territory_analysis(self.phase_by_seg)

			self.cine.set_manual_rois(manual_rois or {})
			self.cine.set_smooth_sigma(float(self.sigma_spin.value()))
			self.cine.set_cube(self.study.cube)
			self._write_outputs()
			self._generate_pdf_report()
			self._refresh_summary()
			self._load_previews()
			self.statusBar().showMessage("Procesamiento completo")
		except Exception as exc:
			self.statusBar().showMessage("Error")
			QMessageBox.critical(self, "Error de procesamiento", str(exc))
			self._log(f"[ERROR] {exc}")

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

		h = int(cube.shape[2])
		w = int(cube.shape[3])
		ys, xs = np.ogrid[:h, :w]
		valid_slices = 0
		gate_cavity_voxels = np.zeros((cube.shape[0],), dtype=np.float64)

		for s in range(n_slices):
			cy = float(centers[s, 0]) if np.isfinite(centers[s, 0]) else np.nan
			cx = float(centers[s, 1]) if np.isfinite(centers[s, 1]) else np.nan
			ri0 = float(inner[s]) if np.isfinite(inner[s]) else np.nan
			ro0 = float(outer[s]) if np.isfinite(outer[s]) else np.nan
			if not np.isfinite(cy) or not np.isfinite(cx) or not np.isfinite(ri0) or ri0 <= 1.0:
				continue
			if not np.isfinite(ro0) or ro0 <= ri0:
				ro0 = ri0 + 2.0

			d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
			r_int = np.floor(d).astype(np.int32)
			r_max = int(min(np.max(r_int), max(3.0, ro0 + 6.0)))
			if r_max < 3:
				continue

			r_lo = max(1, int(round(ri0 * 0.45)))
			r_hi = min(r_max - 1, int(round(max(ri0 + 2.0, (ri0 + ro0) * 0.55))))
			if r_hi <= r_lo + 1:
				continue

			for g in range(cube.shape[0]):
				img = cube[g, s]
				profile = np.zeros((r_max + 1,), dtype=np.float64)
				for rr in range(r_max + 1):
					m = r_int == rr
					if np.any(m):
						profile[rr] = float(np.mean(img[m]))
				if not np.isfinite(profile).any():
					continue
				grad = np.diff(profile)
				window = grad[r_lo:r_hi]
				if window.size < 2:
					r_est = ri0
				else:
					r_est = float(r_lo + int(np.argmax(window)))
				# Restricción para mantener estabilidad entre gates y evitar saltos espurios.
				r_est = float(np.clip(r_est, max(1.0, ri0 * 0.60), max(ri0 * 1.45, ri0 + 1.5)))
				cavity_mask = d <= r_est
				gate_cavity_voxels[g] += float(np.count_nonzero(cavity_mask))

			valid_slices += 1

		if valid_slices < max(3, n_slices // 4):
			return {"available": False}

		gate_volumes_ml = gate_cavity_voxels * float(voxel_ml)
		if gate_volumes_ml.size < 2 or not np.isfinite(gate_volumes_ml).all():
			return {"available": False}

		ed_idx = int(np.argmax(gate_volumes_ml))
		es_idx = int(np.argmin(gate_volumes_ml))
		edv = float(gate_volumes_ml[ed_idx])
		esv = float(gate_volumes_ml[es_idx])
		if edv <= 0.0:
			return {"available": False}

		ef = float((edv - esv) / edv * 100.0)
		sv = float(edv - esv)
		return {
			"available": True,
			"method": "preliminar_radial_gate",
			"valid_slices": int(valid_slices),
			"edv_ml": edv,
			"esv_ml": esv,
			"sv_ml": sv,
			"ef_pct": ef,
			"ed_gate": int(ed_idx + 1),
			"es_gate": int(es_idx + 1),
			"gate_volumes_ml": gate_volumes_ml,
		}

	def _refresh_summary(self):
		if self.study is None or self.metrics is None:
			return

		vol = self._compute_volumes_ml()
		ef = self._estimate_lv_ef_preliminary()

		clinical = []
		clinical.append("Resultado clínico")
		clinical.append(f"  Clasificación de disincronía: {self.metrics.get('classification')}")
		clinical.append(f"  Phase SD: {self.metrics.get('phase_sd')}°")
		clinical.append(f"  Bandwidth: {self.metrics.get('bandwidth')}°")
		clinical.append(f"  Entropy: {self.metrics.get('entropy')}")
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
		cmap_amp_report = str(self.report_cmap_amp.currentText())
		cmap_bullseye = str(self.report_cmap_bullseye.currentText())
		cmap_polar_perf = str(self.report_cmap_polar_perf.currentText())

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

		fig.suptitle("SINCRO — Vista principal", fontsize=14, fontweight="bold")
		fig.tight_layout()
		fig.savefig(os.path.join(self.output_dir, "slices_fase.png"), dpi=150, bbox_inches="tight")
		plt.close(fig)

		pm = build_polar_map(self.phase_by_seg, cmap_name=cmap_phase_report, title="Phase Polar Map")
		save_polar_map(pm, os.path.join(self.output_dir, "polar_map.png"), dpi=150)
		plt.close(pm.fig)

		hfig = build_phase_histogram(self.phase_result.phases_deg, metrics=self.metrics, bins=72, title="Phase Histogram")
		save_histogram(hfig, os.path.join(self.output_dir, "histograma.png"), dpi=150)
		plt.close(hfig)

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
		fig2.suptitle(f"Ejes cardíacos ortogonales — Gate {mid_gate + 1}", fontsize=13, fontweight="bold")
		fig2.tight_layout()
		fig2.savefig(os.path.join(self.output_dir, "ejes_ortogonales.png"), dpi=150, bbox_inches="tight")
		plt.close(fig2)

		if self.axis_companions:
			cmp_gate = self._comparison_gate_index()
			cmp_frac = self._comparison_fraction()
			cmp_vol_gate = self.study.cube[int(cmp_gate)].astype(np.float64)
			hla_recon_idx = min(int(round(cmp_frac * max(0, cmp_vol_gate.shape[1] - 1))), int(cmp_vol_gate.shape[1] - 1))
			vla_recon_idx = min(int(round(cmp_frac * max(0, cmp_vol_gate.shape[2] - 1))), int(cmp_vol_gate.shape[2] - 1))
			if self.axis_companions.get("HLA") is not None:
				hla_original_idx = min(
					int(round(cmp_frac * max(0, self.axis_companions["HLA"].cube.shape[1] - 1))),
					int(self.axis_companions["HLA"].cube.shape[1] - 1),
				)
				hla_original_img = np.asarray(self.axis_companions["HLA"].cube[int(cmp_gate), hla_original_idx], dtype=np.float64)
				hla_original_view = np.fliplr(np.rot90(_norm(hla_original_img), k=1))
			else:
				hla_original_idx = hla_recon_idx
				hla_original_view = hla_view
			if self.axis_companions.get("VLA") is not None:
				vla_original_idx = min(
					int(round(cmp_frac * max(0, self.axis_companions["VLA"].cube.shape[1] - 1))),
					int(self.axis_companions["VLA"].cube.shape[1] - 1),
				)
				vla_original_img = np.asarray(self.axis_companions["VLA"].cube[int(cmp_gate), vla_original_idx], dtype=np.float64)
				vla_original_view = np.flipud(np.rot90(_norm(vla_original_img), k=-1))
			else:
				vla_original_idx = vla_recon_idx
				vla_original_view = vla_view
			hla_recon = np.fliplr(np.rot90(_norm(cmp_vol_gate[:, hla_recon_idx, :]), k=1))
			vla_recon = np.flipud(np.rot90(_norm(cmp_vol_gate[:, :, vla_recon_idx]), k=-1))
			fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(10, 8))
			for ax in axes_cmp.ravel():
				ax.set_xticks([])
				ax.set_yticks([])
			axes_cmp[0, 0].imshow(hla_original_view, cmap=cmap_compare, aspect="auto")
			axes_cmp[0, 0].set_title(f"HLA original (gate {cmp_gate + 1}, corte {hla_original_idx + 1})")
			_annotate_axis(axes_cmp[0, 0], "BASE", "APEX", "ANT", "INF")
			axes_cmp[0, 1].imshow(hla_recon, cmap=cmap_compare, aspect="auto")
			axes_cmp[0, 1].set_title(f"HLA reconstruido desde SA (plano {hla_recon_idx + 1})")
			_annotate_axis(axes_cmp[0, 1], "BASE", "APEX", "ANT", "INF")
			axes_cmp[1, 0].imshow(vla_original_view, cmap=cmap_compare, aspect="auto")
			axes_cmp[1, 0].set_title(f"VLA original (gate {cmp_gate + 1}, corte {vla_original_idx + 1})")
			_annotate_axis(axes_cmp[1, 0], "BASE", "APEX", "SEP", "LAT")
			axes_cmp[1, 1].imshow(vla_recon, cmap=cmap_compare, aspect="auto")
			axes_cmp[1, 1].set_title(f"VLA reconstruido desde SA (plano {vla_recon_idx + 1})")
			_annotate_axis(axes_cmp[1, 1], "BASE", "APEX", "SEP", "LAT")
			fig_cmp.suptitle(
				f"Comparación original vs reconstruido — Gate {cmp_gate + 1}, corte relativo {int(round(cmp_frac * 100.0))}%",
				fontsize=13,
				fontweight="bold",
			)
			fig_cmp.tight_layout()
			fig_cmp.savefig(os.path.join(self.output_dir, "comparacion_ejes.png"), dpi=150, bbox_inches="tight")
			plt.close(fig_cmp)

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
			"qgs-like": {
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
		style = style_catalog.get(style_name, style_catalog["qgs-like"])

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
			"Panel clínico por convención (A=diástole, B=sístole) — SA/HLA/VLA",
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
			f"Panel funcional gated SPECT (estilo clínico: {self.visual_style_combo.currentText()})",
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

		def _slice_angular_profile_from_gate(gate_cube: np.ndarray, s_idx: int) -> np.ndarray | None:
			img = np.asarray(gate_cube[int(s_idx)], dtype=np.float64)
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

		apex_to_base = list(getattr(self.aha, "apex_to_base_order", []) or [])
		if not apex_to_base:
			apex_to_base = [int(s) for s in np.where(self.seg.mask.reshape(self.seg.mask.shape[0], -1).any(axis=1))[0].tolist()]
		profiles = []
		for s in apex_to_base:
			p = _slice_angular_profile(int(s))
			if p is not None:
				profiles.append(p)

		if len(profiles) >= 2:
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
							color=style["grid"],
							linewidth=0.8,
							alpha=0.75,
						)
					)
				# Cruces anatómicas simplificadas para lectura rápida clínica.
				ax.plot([c - r, c + r], [c, c], color=style["grid"], linewidth=0.8, alpha=0.8)
				ax.plot([c, c], [c - r, c + r], color=style["grid"], linewidth=0.8, alpha=0.8)
				ax.text(c, c - r * 1.03, "ANT", ha="center", va="bottom", color=style["fg"], fontsize=8, fontweight="bold")
				ax.text(c + r * 1.03, c, "LAT", ha="left", va="center", color=style["fg"], fontsize=8, fontweight="bold")
				ax.text(c, c + r * 1.03, "INF", ha="center", va="top", color=style["fg"], fontsize=8, fontweight="bold")
				ax.text(c - r * 1.03, c, "SEP", ha="right", va="center", color=style["fg"], fontsize=8, fontweight="bold")
				ax.text(c, c, "APEX", ha="center", va="center", color=style["fg"], fontsize=7, fontweight="bold")
				ax.text(c, c + r * 0.98, "BASE", ha="center", va="top", color=style["subtle"], fontsize=7, fontweight="bold")

			fig_pp, axes_pp = plt.subplots(1, 2, figsize=(12, 6.2), facecolor=style["fig_bg"])
			for ax, img_pp, ttl in [
				(axes_pp[0], cart_raw, "Perfusión polar directa (crudo)"),
				(axes_pp[1], cart_smooth, "Perfusión polar directa (suavizado)"),
			]:
				ax.set_facecolor(style["ax_bg"])
				ax.set_aspect("equal")
				ax.set_xticks([])
				ax.set_yticks([])
				ax.imshow(img_pp, cmap=cmap_polar_perf, vmin=0.0, vmax=1.0)
				_annotate_polar_guides(ax, int(img_pp.shape[0]))
				ax.set_title(ttl, color=style["fg"], fontsize=10, fontweight="bold")

			fig_pp.suptitle(
				f"Mapa polar de perfusión (apex en centro, base en borde) — {self.visual_style_combo.currentText()} | rotación {rotation_deg:+d}°",
				color=style["fg"],
				fontsize=12,
				fontweight="bold",
			)
			fig_pp.text(0.5, 0.04, "Reconstrucción polar continua desde short-axis: \"aplastado\" apex->base", ha="center", color=style["subtle"], fontsize=9)
			fig_pp.savefig(os.path.join(self.output_dir, "polar_perfusion_directa.png"), dpi=170, bbox_inches="tight", facecolor=fig_pp.get_facecolor())
			plt.close(fig_pp)

			# Cine polar gatillado por gate: genera GIF y un montaje estático para preview/PDF.
			try:
				from PIL import Image
			except Exception:
				Image = None

			gate_frames: list[np.ndarray] = []
			for g in range(int(self.study.cube.shape[0])):
				gate_cube = np.asarray(self.study.cube[int(g)], dtype=np.float64)
				profiles_g = []
				for s in apex_to_base:
					pg = _slice_angular_profile_from_gate(gate_cube, int(s))
					if pg is not None:
						profiles_g.append(pg)
				if len(profiles_g) < 2:
					continue
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

				fig_g, ax_g = plt.subplots(1, 1, figsize=(5.2, 5.2), facecolor=style["fig_bg"])
				ax_g.set_facecolor(style["ax_bg"])
				ax_g.set_aspect("equal")
				ax_g.set_xticks([])
				ax_g.set_yticks([])
				ax_g.imshow(cart_g, cmap=cmap_polar_perf, vmin=0.0, vmax=1.0)
				_annotate_polar_guides(ax_g, int(cart_g.shape[0]))
				ax_g.set_title(f"Polar cine gate {g + 1}/{self.study.cube.shape[0]}", color=style["fg"], fontsize=10, fontweight="bold")
				fig_g.tight_layout()
				fig_g.canvas.draw()
				w, h = fig_g.canvas.get_width_height()
				buf = np.frombuffer(fig_g.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
				gate_frames.append(buf)
				plt.close(fig_g)

			if gate_frames:
				polar_cine_ms = int(self.polar_cine_speed_spin.value())
				export_mp4 = bool(self.export_polar_mp4_check.isChecked())
				if Image is not None:
					pil_frames = [Image.fromarray(frm) for frm in gate_frames]
					pil_frames[0].save(
						os.path.join(self.output_dir, "polar_cine.gif"),
						save_all=True,
						append_images=pil_frames[1:],
						duration=polar_cine_ms,
						loop=0,
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
				fig_m, axes_m = plt.subplots(2, int(np.ceil(n_show / 2.0)), figsize=(12, 6.2), facecolor=style["fig_bg"])
				axes_arr = np.atleast_1d(axes_m).ravel()
				for i, ax in enumerate(axes_arr):
					ax.set_facecolor(style["ax_bg"])
					ax.set_xticks([])
					ax.set_yticks([])
					if i < n_show:
						ax.imshow(gate_frames[int(idx[i])])
						ax.set_title(f"Gate {int(idx[i]) + 1}", color=style["fg"], fontsize=9)
					else:
						ax.axis("off")
				fig_m.suptitle("Polar cine gatillado (muestra de gates)", color=style["fg"], fontsize=12, fontweight="bold")
				fig_m.savefig(os.path.join(self.output_dir, "polar_cine_montaje.png"), dpi=160, bbox_inches="tight", facecolor=fig_m.get_facecolor())
				plt.close(fig_m)

	def _load_previews(self):
		for name in self.preview_labels:
			if name == "polar_cine_montaje":
				self._load_polar_cine_preview()
				continue
			path = os.path.join(self.output_dir, f"{name}.png")
			label = self.preview_labels[name]
			if os.path.exists(path):
				pix = QPixmap(path)
				self.preview_pixmaps[name] = pix
				self._apply_preview_zoom(name)
			else:
				self.preview_pixmaps.pop(name, None)
				label.setText("Sin imagen")

	def _load_polar_cine_preview(self):
		name = "polar_cine_montaje"
		label = self.preview_labels[name]
		gif_path = os.path.join(self.output_dir, "polar_cine.gif")
		png_path = os.path.join(self.output_dir, "polar_cine_montaje.png")
		movie = self.preview_movies.pop(name, None)
		if movie is not None:
			movie.stop()
			label.clear()
			label.setMovie(None)
		if os.path.exists(gif_path):
			movie = QMovie(gif_path)
			if movie.isValid():
				self.preview_movies[name] = movie
				self.preview_pixmaps.pop(name, None)
				label.setText("")
				label.setMovie(movie)
				movie.start()
				self._update_polar_cine_toggle_text()
				self._apply_preview_zoom(name)
				return
		if os.path.exists(png_path):
			pix = QPixmap(png_path)
			self.preview_pixmaps[name] = pix
			self._update_polar_cine_toggle_text(enabled=False)
			self._apply_preview_zoom(name)
		else:
			self.preview_pixmaps.pop(name, None)
			self._update_polar_cine_toggle_text(enabled=False)
			label.setText("Sin cine polar")

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

	def _apply_preview_zoom(self, name: str):
		label = self.preview_labels[name]
		movie = self.preview_movies.get(name)
		if movie is not None and movie.isValid():
			base_size = movie.currentPixmap().size()
			if base_size.isEmpty():
				base_size = movie.frameRect().size()
			if base_size.isEmpty():
				base_size = QSize(500, 320)
			zoom = max(0.20, min(4.00, self.preview_zoom.get(name, 1.0)))
			w = max(1, int(base_size.width() * zoom))
			h = max(1, int(base_size.height() * zoom))
			movie.setScaledSize(QSize(w, h))
			label.resize(w, h)
			if name in self.preview_zoom_labels:
				self.preview_zoom_labels[name].setText(f"{int(zoom * 100)}%")
			return
		pix = self.preview_pixmaps.get(name)
		if pix is None or pix.isNull():
			label.setText("Sin imagen")
			return
		zoom = max(0.20, min(4.00, self.preview_zoom.get(name, 1.0)))
		w = max(1, int(pix.width() * zoom))
		h = max(1, int(pix.height() * zoom))
		scaled = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
		label.setPixmap(scaled)
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
		movie = self.preview_movies.get("polar_cine_montaje")
		if movie is not None and movie.isValid():
			state = movie.state()
			if state == QMovie.MovieState.Running:
				movie.setPaused(True)
			else:
				movie.start()
				movie.setPaused(False)
		self._update_polar_cine_toggle_text()

	def _restart_polar_cine_preview(self):
		movie = self.preview_movies.get("polar_cine_montaje")
		if movie is not None and movie.isValid():
			movie.stop()
			movie.start()
		self._update_polar_cine_toggle_text()

	def _update_polar_cine_toggle_text(self, enabled: bool = True):
		if self.polar_cine_toggle_btn is None:
			return
		self.polar_cine_toggle_btn.setEnabled(enabled)
		if not enabled:
			self.polar_cine_toggle_btn.setText("Play/Pause")
			return
		movie = self.preview_movies.get("polar_cine_montaje")
		if movie is None or not movie.isValid():
			self.polar_cine_toggle_btn.setText("Play/Pause")
			return
		if movie.state() == QMovie.MovieState.Running:
			self.polar_cine_toggle_btn.setText("Pause")
		else:
			self.polar_cine_toggle_btn.setText("Play")

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
		pdf_path = os.path.join(self.output_dir, "informe_sincro.pdf")
		if not os.path.exists(pdf_path):
			QMessageBox.information(self, "SINCRO", "Todavía no hay PDF generado en output_demo.")
			return
		QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_path))

	def open_polar_map(self):
		pm_path = os.path.join(self.output_dir, "polar_map.png")
		if not os.path.exists(pm_path):
			QMessageBox.information(self, "SINCRO", "Todavía no hay polar map generado. Procesá un estudio primero.")
			return
		QDesktopServices.openUrl(QUrl.fromLocalFile(pm_path))
