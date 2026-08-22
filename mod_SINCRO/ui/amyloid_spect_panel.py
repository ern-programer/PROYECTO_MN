# -*- coding: utf-8 -*-
"""Panel AMYLO SPECT 3D (fase 2, experimental)."""
from __future__ import annotations

import os
import base64
import numpy as np

from PyQt6.QtCore import QObject, Qt, QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QTextEdit,
    QCheckBox,
    QGridLayout,
    QSlider,
    QProgressBar,
    QGroupBox,
    QDoubleSpinBox,
    QSpinBox,
)

from scipy import ndimage as ndi

from core.col_registry import available_colormaps, register_all_colormaps
from ui.cine_widget import RangeSlider

from core.amyloid_spect import (
    run_amyloid_spect_analysis,
    reconstruct_amyloid_with_perf_pipeline,
    export_amyloid_cardiac_axes_dicom,
    apply_visual_bone_suppression,
    central_slices_preview,
    load_ct_volume_from_path,
    load_attenuation_map_from_path,
    apply_attenuation_correction_prototype,
    apply_attenuation_correction_chang,
    resample_volume_to_spect_grid,
    register_ct_to_spect_rigid,
    align_ct_orientation_to_spect,
    refine_ct_to_spect_translation,
)


class _AmyloidReconWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        dicom_path: str,
        recon_config,
        cuts_mode: str,
        attenuation_mu_map: np.ndarray | None = None,
        attenuation_pixel_size_cm: float | None = None,
    ):
        super().__init__()
        self._dicom_path = dicom_path
        self._recon_config = recon_config
        self._cuts_mode = cuts_mode
        self._attenuation_mu_map = None if attenuation_mu_map is None else np.asarray(attenuation_mu_map, dtype=np.float64)
        self._attenuation_pixel_size_cm = attenuation_pixel_size_cm
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            def _progress(frac: float, message: str):
                if self._cancel_requested:
                    raise RuntimeError("Reconstrucción cancelada por el usuario.")
                pct = int(round(100.0 * max(0.0, min(1.0, float(frac)))))
                self.progress.emit(pct, str(message or "Procesando..."))

            bundle = reconstruct_amyloid_with_perf_pipeline(
                self._dicom_path,
                recon_config=self._recon_config,
                cuts_mode=self._cuts_mode,
                attenuation_mu_map=self._attenuation_mu_map,
                attenuation_pixel_size_cm=self._attenuation_pixel_size_cm,
                progress_callback=_progress,
            )
            self.finished.emit(bundle)
        except Exception as exc:
            self.failed.emit(str(exc))


class AmyloidSpectPanel(QDialog):
    """UI mínima para flujo AMYLO SPECT 3D."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SINCRO — AMYLO SPECT 3D (experimental)")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        self.resize(1060, 700)
        self.setMinimumSize(920, 580)
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

        self._analysis = None
        self._recon_bundle = None
        self._current_volume = None
        self._base_spect_volume = None
        self._ct_volume = None
        self._ct_registered = None
        self._ct_auto_registered = None
        self._att_map_volume = None
        self._att_map_registered = None
        self._att_spacing_zyx = None
        self._att_affine_ijk_to_lps = None
        self._att_path = ""
        self._spect_spacing_zyx = None
        self._ct_spacing_zyx = None
        self._spect_affine_ijk_to_lps = None
        self._ct_affine_ijk_to_lps = None
        self._current_spect_path = ""
        self._ct_path = ""
        self._bone_mask = None
        self._settings = QSettings("GAMMASYS", "SINCRO_AMYLO_SPECT")
        self._recon_thread = None
        self._recon_worker = None
        self._pending_recon_cfg = None
        self._pending_recon_preset_label = ""
        self._pending_recon_cuts_mode = ""
        self._pending_recon_ac_iter = False
        self._study_is_gated = False
        self._slice_idx = {"axial": 0, "coronal": 0, "sagittal": 0}
        self._spect_view_offset = {"axial": 0, "coronal": 0, "sagittal": 0}
        self._ct_view_offset = {"axial": 0, "coronal": 0, "sagittal": 0}
        self._spect_zoom_pct = 100
        self._ct_zoom_pct = 100
        self._spect_pan_px = {"axial": [0, 0], "coronal": [0, 0], "sagittal": [0, 0]}
        self._ct_pan_px = {"axial": [0, 0], "coronal": [0, 0], "sagittal": [0, 0]}
        self._drag_state = None
        # Sensibilidad del arrastre triangulado CT (Ctrl+drag).
        # <1.0 = más suave/fino.
        self._ct_drag_sensitivity = 0.35
        self._spect_win_low = 0
        self._spect_win_high = 100
        self._fusion_pct = 55
        self._ct_window = "bone"

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        flow_box = QGroupBox("Flujo clínico AMYLO SPECT / SPECT-CT")
        flow = QGridLayout(flow_box)
        flow.setContentsMargins(8, 6, 8, 6)
        flow.setHorizontalSpacing(8)
        flow.setVerticalSpacing(4)

        self._btn_load = QPushButton("1. Cargar SPECT")
        self._btn_load.clicked.connect(self._load_spect)
        self._btn_load.setToolTip("Carga el DICOM SPECT AMYLO. Puede ser crudo o reconstruido; no requiere gating.")
        flow.addWidget(self._btn_load, 0, 0)

        flow.addWidget(QLabel("Preset:"), 0, 1)
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Manual", "manual")
        self._preset_combo.addItem("AMYLO 360 estándar 128", "amylo360_std128")
        self._preset_combo.addItem("AMYLO 360 alta definición", "amylo360_hd")
        self._preset_combo.setToolTip(
            "Presets PYP/SPECT 360. 128x128 es recomendación de adquisición; "
            "si el DICOM trae otra matriz, se reconstruye en la matriz adquirida."
        )
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        flow.addWidget(self._preset_combo, 0, 2)

        flow.addWidget(QLabel("Recon:"), 0, 3)
        self._recon_combo = QComboBox()
        self._recon_combo.addItem("FBP", "fbp")
        self._recon_combo.addItem("OSEM", "osem")
        self._recon_combo.addItem("MLEM", "mlem")
        self._recon_combo.currentIndexChanged.connect(self._on_top_recon_changed)
        flow.addWidget(self._recon_combo, 0, 4)

        flow.addWidget(QLabel("Cortes:"), 0, 5)
        self._cuts_mode_combo = QComboBox()
        self._cuts_mode_combo.addItem("Mixto", "mixed")
        self._cuts_mode_combo.addItem("Tomográficos", "tomo")
        self._cuts_mode_combo.addItem("Cardíacos", "cardiac")
        self._cuts_mode_combo.setToolTip("Selecciona qué familia de cortes generar en recon AMYLO.")
        self._cuts_mode_combo.currentIndexChanged.connect(self._persist_ui_state)
        flow.addWidget(self._cuts_mode_combo, 0, 6)

        self._btn_recon_pipeline = QPushButton("2. Reconstruir + cortes")
        self._btn_recon_pipeline.clicked.connect(self._reconstruct_with_perf_pipeline)
        self._btn_recon_pipeline.setEnabled(False)
        self._btn_recon_pipeline.setToolTip("Reconstruye el SPECT y genera cortes tomográficos; si hay gating, también SA/HLA/VLA.")
        flow.addWidget(self._btn_recon_pipeline, 0, 7)

        self._btn_cancel_recon = QPushButton("Cancelar")
        self._btn_cancel_recon.clicked.connect(self._cancel_reconstruction)
        self._btn_cancel_recon.setEnabled(False)
        flow.addWidget(self._btn_cancel_recon, 0, 8)

        self._ct_check = QCheckBox("Usar CT para sustracción ósea (si hay)")
        self._ct_check.setToolTip("Si hay CT cargado/registrado, usa el CT para definir máscara ósea visual.")
        flow.addWidget(self._ct_check, 1, 0, 1, 2)

        self._btn_load_ct = QPushButton("3a. Cargar CT")
        self._btn_load_ct.clicked.connect(self._load_ct)
        self._btn_load_ct.setEnabled(False)
        flow.addWidget(self._btn_load_ct, 1, 2)

        self._btn_load_ct_dir = QPushButton("3b. CT carpeta")
        self._btn_load_ct_dir.clicked.connect(self._load_ct_dir)
        self._btn_load_ct_dir.setEnabled(False)
        flow.addWidget(self._btn_load_ct_dir, 1, 3)

        self._btn_register = QPushButton("4. Registrar CT↔SPECT")
        self._btn_register.clicked.connect(self._register_ct_to_spect)
        self._btn_register.setEnabled(False)
        flow.addWidget(self._btn_register, 1, 4, 1, 2)

        self._btn_load_att = QPushButton("4b. Cargar ATT MAP")
        self._btn_load_att.clicked.connect(self._load_att_map)
        self._btn_load_att.setEnabled(False)
        flow.addWidget(self._btn_load_att, 2, 0, 1, 2)

        self._btn_apply_ac = QPushButton("4c. Aplicar AC (Chang)")
        self._btn_apply_ac.clicked.connect(self._apply_ac_prototype)
        self._btn_apply_ac.setEnabled(False)
        self._btn_apply_ac.setToolTip("Aplica corrección por atenuación tipo Chang (slice-wise) usando ATT MAP remuestreado a grilla SPECT.")
        flow.addWidget(self._btn_apply_ac, 2, 2, 1, 2)

        self._btn_bone = QPushButton("5. Sustracción ósea visual")
        self._btn_bone.clicked.connect(self._apply_bone_suppression)
        self._btn_bone.setEnabled(False)
        flow.addWidget(self._btn_bone, 1, 6, 1, 2)

        self._btn_export_axes_dcm = QPushButton("6. Exportar ejes DICOM")
        self._btn_export_axes_dcm.clicked.connect(self._export_axes_dicom)
        self._btn_export_axes_dcm.setEnabled(False)
        flow.addWidget(self._btn_export_axes_dcm, 1, 8)
        flow.setColumnStretch(9, 1)
        root.addWidget(flow_box)

        recon_box = QGroupBox("Reconstrucción y filtros")
        recon_grid = QGridLayout(recon_box)
        recon_grid.setContentsMargins(8, 6, 8, 6)
        recon_grid.setHorizontalSpacing(8)
        recon_grid.setVerticalSpacing(4)

        self._ung_filter_combo = self._mk_filter_combo()
        self._ung_cutoff_spin = self._mk_cutoff_spin(0.45)
        self._ung_order_spin = self._mk_order_spin(8)
        self._gated_filter_combo = self._mk_filter_combo()
        self._gated_cutoff_spin = self._mk_cutoff_spin(0.40)
        self._gated_order_spin = self._mk_order_spin(10)
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 30)
        self._iter_spin.setValue(3)
        self._iter_spin.setToolTip("Iteraciones para OSEM/MLEM en modo Manual.")
        self._subsets_spin = QSpinBox()
        self._subsets_spin.setRange(1, 32)
        self._subsets_spin.setValue(6)
        self._subsets_spin.setToolTip("Subsets para OSEM. En MLEM se ignora y se usa 1 subset.")

        for widget in (
            self._ung_filter_combo,
            self._ung_cutoff_spin,
            self._ung_order_spin,
            self._gated_filter_combo,
            self._gated_cutoff_spin,
            self._gated_order_spin,
            self._iter_spin,
            self._subsets_spin,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._persist_ui_state)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._persist_ui_state)

        recon_grid.addWidget(QLabel("Ungated"), 0, 0)
        self._ung_method_combo = QComboBox()
        self._ung_method_combo.addItem("FBP", "fbp")
        self._ung_method_combo.addItem("MLEM", "mlem")
        self._ung_method_combo.addItem("OSEM", "osem")
        self._ung_method_combo.setMaximumWidth(90)
        self._ung_method_combo.setToolTip("Método de reconstrucción para el volumen estático AMYLO.")
        self._ung_method_combo.currentIndexChanged.connect(self._on_manual_method_changed)
        recon_grid.addWidget(self._ung_method_combo, 0, 1)
        recon_grid.addWidget(self._ung_filter_combo, 0, 2)
        recon_grid.addWidget(QLabel("cutoff"), 0, 3)
        recon_grid.addWidget(self._ung_cutoff_spin, 0, 4)
        recon_grid.addWidget(QLabel("orden"), 0, 5)
        recon_grid.addWidget(self._ung_order_spin, 0, 6)
        recon_grid.addWidget(QLabel("Iter"), 0, 7)
        recon_grid.addWidget(self._iter_spin, 0, 8)
        recon_grid.addWidget(QLabel("Subsets"), 0, 9)
        recon_grid.addWidget(self._subsets_spin, 0, 10)

        self._gated_title_lbl = QLabel("Gated")
        recon_grid.addWidget(self._gated_title_lbl, 1, 0)
        self._gated_method_combo = QComboBox()
        self._gated_method_combo.addItem("FBP", "fbp")
        self._gated_method_combo.addItem("MLEM", "mlem")
        self._gated_method_combo.addItem("OSEM", "osem")
        self._gated_method_combo.setMaximumWidth(90)
        self._gated_method_combo.setToolTip("Método de reconstrucción gate-by-gate. Se desactiva en AMYLO no gatillado.")
        recon_grid.addWidget(self._gated_method_combo, 1, 1)
        recon_grid.addWidget(self._gated_filter_combo, 1, 2)
        recon_grid.addWidget(QLabel("cutoff"), 1, 3)
        recon_grid.addWidget(self._gated_cutoff_spin, 1, 4)
        recon_grid.addWidget(QLabel("orden"), 1, 5)
        recon_grid.addWidget(self._gated_order_spin, 1, 6)

        self._view_combo = QComboBox()
        self._view_combo.addItem("MPR volumen actual", "mpr")
        self._view_combo.addItem("Cortes tomográficos", "tomo")
        self._view_combo.addItem("Cortes cardíacos SA/HLA/VLA", "cardiac")
        self._view_combo.setEnabled(False)
        self._view_combo.currentIndexChanged.connect(self._render_selected_view)
        recon_grid.addWidget(QLabel("Vista"), 1, 7)
        recon_grid.addWidget(self._view_combo, 1, 8, 1, 3)

        self._bg_check = QCheckBox("Fondo")
        self._bg_check.setToolTip("Descuento de fondo automático en el sinograma antes de reconstruir.")
        self._scatter_check = QCheckBox("Desc. SC")
        self._scatter_check.setToolTip("Resta scatter si el loader encontró proyecciones SC asociadas.")
        self._scatter_k_spin = QDoubleSpinBox()
        self._scatter_k_spin.setRange(0.0, 2.0)
        self._scatter_k_spin.setSingleStep(0.05)
        self._scatter_k_spin.setDecimals(2)
        self._scatter_k_spin.setValue(1.0)
        self._post_check = QCheckBox("Suavizar")
        self._post_check.setToolTip("Post-filtro gaussiano 3D del volumen AMYLO.")
        self._post_sigma_spin = QDoubleSpinBox()
        self._post_sigma_spin.setRange(0.0, 6.0)
        self._post_sigma_spin.setSingleStep(0.1)
        self._post_sigma_spin.setDecimals(1)
        self._post_sigma_spin.setValue(0.5)
        self._denoise_plus_check = QCheckBox("Denoise+")
        self._denoise_plus_check.setToolTip("Denoise de sinograma + realce por resta en la rama ungated.")
        self._denoise_plus_k_spin = QDoubleSpinBox()
        self._denoise_plus_k_spin.setRange(0.0, 0.70)
        self._denoise_plus_k_spin.setSingleStep(0.05)
        self._denoise_plus_k_spin.setDecimals(2)
        self._denoise_plus_k_spin.setValue(0.20)
        self._ac_iter_check = QCheckBox("AC iterativa")
        self._ac_iter_check.setToolTip("Activa corrección por atenuación dentro del update OSEM/MLEM (requiere ATT MAP cargado).")
        self._ac_mu_scale_spin = QDoubleSpinBox()
        self._ac_mu_scale_spin.setRange(0.10, 3.00)
        self._ac_mu_scale_spin.setSingleStep(0.05)
        self._ac_mu_scale_spin.setDecimals(2)
        self._ac_mu_scale_spin.setValue(1.00)
        for widget in (self._bg_check, self._scatter_check, self._scatter_k_spin, self._post_check, self._post_sigma_spin, self._denoise_plus_check, self._denoise_plus_k_spin, self._ac_iter_check, self._ac_mu_scale_spin, self._gated_method_combo):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._persist_ui_state)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self._persist_ui_state)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._persist_ui_state)
        recon_grid.addWidget(self._bg_check, 2, 0)
        recon_grid.addWidget(self._scatter_check, 2, 1)
        recon_grid.addWidget(QLabel("k SC"), 2, 2)
        recon_grid.addWidget(self._scatter_k_spin, 2, 3)
        recon_grid.addWidget(self._post_check, 2, 4)
        recon_grid.addWidget(self._post_sigma_spin, 2, 5)
        recon_grid.addWidget(self._denoise_plus_check, 2, 6)
        recon_grid.addWidget(QLabel("k"), 2, 7)
        recon_grid.addWidget(self._denoise_plus_k_spin, 2, 8)
        recon_grid.addWidget(self._ac_iter_check, 2, 9)
        recon_grid.addWidget(QLabel("μ-scale"), 2, 10)
        recon_grid.addWidget(self._ac_mu_scale_spin, 2, 11)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("Listo")
        recon_grid.addWidget(self._progress, 3, 0, 1, 11)
        root.addWidget(recon_box)

        self._status = QLabel("Cargar un DICOM SPECT para iniciar.")
        self._status.setStyleSheet("color:#93c5fd; font-size:11px;")
        root.addWidget(self._status)

        grid = QGridLayout()
        self._axial_lbl = self._mk_image_label("Axial")
        self._cor_lbl = self._mk_image_label("Coronal")
        self._sag_lbl = self._mk_image_label("Sagital")

        grid.addWidget(self._axial_lbl, 0, 0)
        grid.addWidget(self._cor_lbl, 0, 1)
        grid.addWidget(self._sag_lbl, 0, 2)
        root.addLayout(grid, 1)

        blend_row = QHBoxLayout()
        blend_row.addWidget(QLabel("Overlay óseo:"))
        self._blend_slider = QSlider(Qt.Orientation.Horizontal)
        self._blend_slider.setRange(0, 100)
        self._blend_slider.setValue(35)
        self._blend_slider.setToolTip("0% sin overlay · 100% overlay máximo")
        self._blend_slider.valueChanged.connect(self._on_visual_controls_changed)
        self._blend_slider.setEnabled(False)
        blend_row.addWidget(self._blend_slider, 1)
        self._blend_lbl = QLabel("35%")
        self._blend_lbl.setStyleSheet("color:#94a3b8;")
        blend_row.addWidget(self._blend_lbl)
        root.addLayout(blend_row)

        qc_row = QHBoxLayout()
        qc_row.addWidget(QLabel("QC registro:"))
        self._qc_mode = QComboBox()
        self._qc_mode.addItem("Off", "off")
        self._qc_mode.addItem("Fusión", "fusion")
        self._qc_mode.addItem("Bordes SPECT/CT", "edges")
        self._qc_mode.addItem("Split", "split")
        self._qc_mode.addItem("Checkerboard", "checker")
        self._qc_mode.addItem("Contornos CT", "contours")
        self._qc_mode.currentIndexChanged.connect(self._on_visual_controls_changed)
        self._qc_mode.setEnabled(False)
        qc_row.addWidget(self._qc_mode)
        qc_row.addWidget(QLabel("Split %:"))
        self._qc_split_slider = QSlider(Qt.Orientation.Horizontal)
        self._qc_split_slider.setRange(10, 90)
        self._qc_split_slider.setValue(50)
        self._qc_split_slider.valueChanged.connect(self._on_visual_controls_changed)
        self._qc_split_slider.setEnabled(False)
        qc_row.addWidget(self._qc_split_slider, 1)
        self._qc_split_lbl = QLabel("50%")
        self._qc_split_lbl.setStyleSheet("color:#94a3b8;")
        qc_row.addWidget(self._qc_split_lbl)
        qc_row.addWidget(QLabel("Fusión %:"))
        self._fusion_slider = QSlider(Qt.Orientation.Horizontal)
        self._fusion_slider.setRange(0, 100)
        self._fusion_slider.setValue(55)
        self._fusion_slider.setToolTip("0% CT solamente · 100% SPECT coloreado. Ajusta la mezcla de la fusión.")
        self._fusion_slider.valueChanged.connect(self._on_fusion_slider_changed)
        self._fusion_slider.setEnabled(False)
        qc_row.addWidget(self._fusion_slider, 1)
        self._fusion_lbl = QLabel("55%")
        self._fusion_lbl.setStyleSheet("color:#94a3b8;")
        qc_row.addWidget(self._fusion_lbl)
        root.addLayout(qc_row)

        slice_row = QHBoxLayout()
        slice_row.addWidget(QLabel("Cortes z/y/x:"))
        self._slice_z = QSlider(Qt.Orientation.Horizontal)
        self._slice_y = QSlider(Qt.Orientation.Horizontal)
        self._slice_x = QSlider(Qt.Orientation.Horizontal)
        self._slice_z_lbl = QLabel("z -")
        self._slice_y_lbl = QLabel("y -")
        self._slice_x_lbl = QLabel("x -")
        for slider in (self._slice_z, self._slice_y, self._slice_x):
            slider.setRange(0, 0)
            slider.setEnabled(False)
            slider.valueChanged.connect(self._on_slice_slider_changed)
        slice_row.addWidget(self._slice_z_lbl)
        slice_row.addWidget(self._slice_z, 1)
        slice_row.addWidget(self._slice_y_lbl)
        slice_row.addWidget(self._slice_y, 1)
        slice_row.addWidget(self._slice_x_lbl)
        slice_row.addWidget(self._slice_x, 1)
        root.addLayout(slice_row)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom visual SPECT/CT:"))
        self._spect_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._spect_zoom_slider.setRange(50, 200)
        self._spect_zoom_slider.setValue(100)
        self._spect_zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self._spect_zoom_lbl = QLabel("SPECT 100%")
        self._ct_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._ct_zoom_slider.setRange(50, 200)
        self._ct_zoom_slider.setValue(100)
        self._ct_zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self._ct_zoom_lbl = QLabel("CT 100%")
        zoom_row.addWidget(self._spect_zoom_lbl)
        zoom_row.addWidget(self._spect_zoom_slider, 1)
        zoom_row.addWidget(self._ct_zoom_lbl)
        zoom_row.addWidget(self._ct_zoom_slider, 1)
        root.addLayout(zoom_row)

        visual_row = QHBoxLayout()
        visual_row.addWidget(QLabel("Color SPECT:"))
        register_all_colormaps()
        self._spect_cmap_combo = QComboBox()
        self._spect_cmap_combo.addItems(available_colormaps())
        if self._spect_cmap_combo.findText("hot") >= 0:
            self._spect_cmap_combo.setCurrentText("hot")
        self._spect_cmap_combo.currentIndexChanged.connect(self._render_current_with_overlay)
        visual_row.addWidget(self._spect_cmap_combo)
        self._btn_range_base = QPushButton("Base 0")
        self._btn_range_base.clicked.connect(lambda: self._set_spect_range(0, self._spect_win_high))
        self._btn_range_top = QPushButton("Top 100")
        self._btn_range_top.clicked.connect(lambda: self._set_spect_range(self._spect_win_low, 100))
        visual_row.addWidget(self._btn_range_base)
        visual_row.addWidget(self._btn_range_top)
        visual_row.addWidget(QLabel("Ventana CT:"))
        self._ct_window_combo = QComboBox()
        self._ct_window_combo.addItem("Ósea", "bone")
        self._ct_window_combo.addItem("Partes blandas", "soft")
        self._ct_window_combo.addItem("Pulmón", "lung")
        self._ct_window_combo.addItem("Completa", "full")
        self._ct_window_combo.currentIndexChanged.connect(self._on_ct_window_changed)
        visual_row.addWidget(self._ct_window_combo)
        visual_row.addStretch(1)
        root.addLayout(visual_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Rango SPECT 0–200%:"))
        self._spect_range_slider = RangeSlider()
        self._spect_range_slider.setMinimumHeight(90)
        self._spect_range_slider.valuesChanged.connect(self._on_spect_range_changed)
        range_row.addWidget(self._spect_range_slider)
        self._spect_range_lbl = QLabel("Base 0% · Top 100%")
        range_row.addWidget(self._spect_range_lbl)
        range_row.addStretch(1)
        root.addLayout(range_row)

        nudge_row = QHBoxLayout()
        nudge_row.addWidget(QLabel("Ajuste CT Δ z/y/x:"))
        self._nudge_z = self._mk_nudge_spin()
        self._nudge_y = self._mk_nudge_spin()
        self._nudge_x = self._mk_nudge_spin()
        for spin in (self._nudge_z, self._nudge_y, self._nudge_x):
            spin.valueChanged.connect(self._apply_ct_nudge)
            spin.setEnabled(False)
        nudge_row.addWidget(self._nudge_z)
        nudge_row.addWidget(self._nudge_y)
        nudge_row.addWidget(self._nudge_x)
        self._btn_reset_nudge = QPushButton("Reset ajuste")
        self._btn_reset_nudge.clicked.connect(self._reset_ct_nudge)
        self._btn_reset_nudge.setEnabled(False)
        nudge_row.addWidget(self._btn_reset_nudge)
        self._btn_reset_offsets = QPushButton("Reset offsets vista")
        self._btn_reset_offsets.clicked.connect(self._reset_view_offsets)
        self._btn_reset_offsets.setToolTip("Resetea offsets relativos y zoom visual SPECT/CT.")
        nudge_row.addWidget(self._btn_reset_offsets)
        nudge_row.addStretch(1)
        root.addLayout(nudge_row)

        self._metrics = QTextEdit()
        self._metrics.setReadOnly(True)
        self._metrics.setStyleSheet("background:#0f172a; color:#e2e8f0; border:1px solid #334155;")
        root.addWidget(self._metrics, 1)

        footer = QLabel(
            "Módulo experimental fase 2: métricas 3D proxy y sustracción ósea visual de apoyo. "
            "No constituye interpretación diagnóstica automática."
        )
        footer.setWordWrap(True)
        footer.setStyleSheet("color:#94a3b8; font-size:10px;")
        root.addWidget(footer)
        self._restore_global_ui_state()

    @staticmethod
    def _mk_filter_combo() -> QComboBox:
        combo = QComboBox()
        combo.addItem("Sin filtro", "none")
        combo.addItem("Butterworth", "butterworth")
        combo.addItem("Low-pass", "lowpass")
        combo.addItem("Wiener", "wiener")
        return combo

    @staticmethod
    def _mk_cutoff_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 1.00)
        spin.setSingleStep(0.01)
        spin.setDecimals(2)
        spin.setValue(float(value))
        spin.setToolTip("Frecuencia normalizada respecto de Nyquist.")
        return spin

    @staticmethod
    def _mk_order_spin(value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, 20)
        spin.setValue(int(value))
        return spin

    @staticmethod
    def _mk_nudge_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-64.0, 64.0)
        spin.setSingleStep(0.5)
        spin.setDecimals(1)
        spin.setSuffix(" px")
        spin.setToolTip("Ajuste manual post-registro CT→SPECT en píxeles de la grilla SPECT.")
        return spin

    @staticmethod
    def _settings_id(path: str) -> str:
        raw = os.path.abspath(str(path or "")).encode("utf-8", errors="ignore")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") or "sin_estudio"

    def _study_settings_prefix(self) -> str:
        return f"studies/{self._settings_id(self._current_spect_path)}"

    @staticmethod
    def _set_combo_by_data(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _persist_ui_state(self):
        if not hasattr(self, "_settings"):
            return
        cuts_mode = str(self._cuts_mode_combo.currentData() or "mixed")
        qc_mode = str(self._qc_mode.currentData() or "off")
        preset = str(self._preset_combo.currentData() or "manual")
        split = int(self._qc_split_slider.value())
        overlay = int(self._blend_slider.value())
        fusion = int(self._fusion_slider.value()) if hasattr(self, "_fusion_slider") else int(getattr(self, "_fusion_pct", 55))
        self._settings.setValue("global/preset", preset)
        self._settings.setValue("global/ung_method", str(self._ung_method_combo.currentData() or "fbp"))
        self._settings.setValue("global/gated_method", str(self._gated_method_combo.currentData() or "fbp"))
        self._settings.setValue("global/cuts_mode", cuts_mode)
        self._settings.setValue("global/qc_mode", qc_mode)
        self._settings.setValue("global/split_pct", split)
        self._settings.setValue("global/overlay_pct", overlay)
        self._settings.setValue("global/fusion_pct", fusion)
        self._settings.setValue("global/ung_filter", str(self._ung_filter_combo.currentData() or "butterworth"))
        self._settings.setValue("global/ung_cutoff", float(self._ung_cutoff_spin.value()))
        self._settings.setValue("global/ung_order", int(self._ung_order_spin.value()))
        self._settings.setValue("global/gated_filter", str(self._gated_filter_combo.currentData() or "butterworth"))
        self._settings.setValue("global/gated_cutoff", float(self._gated_cutoff_spin.value()))
        self._settings.setValue("global/gated_order", int(self._gated_order_spin.value()))
        self._settings.setValue("global/iter", int(self._iter_spin.value()))
        self._settings.setValue("global/subsets", int(self._subsets_spin.value()))
        self._settings.setValue("global/background_subtract", bool(self._bg_check.isChecked()))
        self._settings.setValue("global/scatter_subtract", bool(self._scatter_check.isChecked()))
        self._settings.setValue("global/scatter_k", float(self._scatter_k_spin.value()))
        self._settings.setValue("global/post_filter", bool(self._post_check.isChecked()))
        self._settings.setValue("global/post_sigma", float(self._post_sigma_spin.value()))
        self._settings.setValue("global/denoise_plus", bool(self._denoise_plus_check.isChecked()))
        self._settings.setValue("global/denoise_plus_k", float(self._denoise_plus_k_spin.value()))
        self._settings.setValue("global/ac_iter", bool(self._ac_iter_check.isChecked()))
        self._settings.setValue("global/ac_mu_scale", float(self._ac_mu_scale_spin.value()))
        if self._current_spect_path:
            prefix = self._study_settings_prefix()
            self._settings.setValue(f"{prefix}/preset", preset)
            self._settings.setValue(f"{prefix}/cuts_mode", cuts_mode)
            self._settings.setValue(f"{prefix}/qc_mode", qc_mode)
            self._settings.setValue(f"{prefix}/split_pct", split)
            self._settings.setValue(f"{prefix}/overlay_pct", overlay)
            self._settings.setValue(f"{prefix}/fusion_pct", fusion)
            if self._ct_path:
                self._settings.setValue(f"{prefix}/ct_path", self._ct_path)

    def _restore_global_ui_state(self):
        self._set_combo_by_data(self._preset_combo, str(self._settings.value("global/preset", "amylo360_std128") or "amylo360_std128"))
        self._set_combo_by_data(self._ung_method_combo, str(self._settings.value("global/ung_method", "fbp") or "fbp"))
        self._set_combo_by_data(self._gated_method_combo, str(self._settings.value("global/gated_method", "fbp") or "fbp"))
        self._set_combo_by_data(self._cuts_mode_combo, str(self._settings.value("global/cuts_mode", "mixed") or "mixed"))
        self._set_combo_by_data(self._qc_mode, str(self._settings.value("global/qc_mode", "off") or "off"))
        self._set_combo_by_data(self._ung_filter_combo, str(self._settings.value("global/ung_filter", "butterworth") or "butterworth"))
        self._ung_cutoff_spin.setValue(float(self._settings.value("global/ung_cutoff", 0.45) or 0.45))
        self._ung_order_spin.setValue(int(self._settings.value("global/ung_order", 8) or 8))
        self._set_combo_by_data(self._gated_filter_combo, str(self._settings.value("global/gated_filter", "butterworth") or "butterworth"))
        self._gated_cutoff_spin.setValue(float(self._settings.value("global/gated_cutoff", 0.40) or 0.40))
        self._gated_order_spin.setValue(int(self._settings.value("global/gated_order", 10) or 10))
        self._iter_spin.setValue(int(self._settings.value("global/iter", 3) or 3))
        self._subsets_spin.setValue(int(self._settings.value("global/subsets", 6) or 6))
        self._bg_check.setChecked(str(self._settings.value("global/background_subtract", "false")).lower() == "true")
        self._scatter_check.setChecked(str(self._settings.value("global/scatter_subtract", "false")).lower() == "true")
        self._scatter_k_spin.setValue(float(self._settings.value("global/scatter_k", 1.0) or 1.0))
        self._post_check.setChecked(str(self._settings.value("global/post_filter", "false")).lower() == "true")
        self._post_sigma_spin.setValue(float(self._settings.value("global/post_sigma", 0.5) or 0.5))
        self._denoise_plus_check.setChecked(str(self._settings.value("global/denoise_plus", "false")).lower() == "true")
        self._denoise_plus_k_spin.setValue(float(self._settings.value("global/denoise_plus_k", 0.20) or 0.20))
        self._ac_iter_check.setChecked(str(self._settings.value("global/ac_iter", "false")).lower() == "true")
        self._ac_mu_scale_spin.setValue(float(self._settings.value("global/ac_mu_scale", 1.0) or 1.0))
        self._qc_split_slider.setValue(int(self._settings.value("global/split_pct", 50) or 50))
        self._blend_slider.setValue(int(self._settings.value("global/overlay_pct", 35) or 35))
        fusion = int(self._settings.value("global/fusion_pct", 55) or 55)
        self._fusion_slider.setValue(fusion)
        self._fusion_pct = fusion
        self._on_preset_changed()

    def _restore_study_ui_state(self):
        if not self._current_spect_path:
            return
        prefix = self._study_settings_prefix()
        self._set_combo_by_data(
            self._preset_combo,
            str(self._settings.value(f"{prefix}/preset", self._preset_combo.currentData()) or "amylo360_std128"),
        )
        self._set_combo_by_data(
            self._cuts_mode_combo,
            str(self._settings.value(f"{prefix}/cuts_mode", self._cuts_mode_combo.currentData()) or "mixed"),
        )
        self._set_combo_by_data(
            self._qc_mode,
            str(self._settings.value(f"{prefix}/qc_mode", self._qc_mode.currentData()) or "off"),
        )
        self._qc_split_slider.setValue(int(self._settings.value(f"{prefix}/split_pct", self._qc_split_slider.value()) or 50))
        self._blend_slider.setValue(int(self._settings.value(f"{prefix}/overlay_pct", self._blend_slider.value()) or 35))
        fusion = int(self._settings.value(f"{prefix}/fusion_pct", self._fusion_slider.value()) or 55)
        self._fusion_slider.setValue(fusion)
        self._fusion_pct = fusion
        self._ct_path = str(self._settings.value(f"{prefix}/ct_path", "") or "")
        self._on_preset_changed()

    def _on_preset_changed(self):
        preset = str(self._preset_combo.currentData() or "manual")
        manual = preset == "manual"
        self._recon_combo.setEnabled(manual)
        if preset == "amylo360_std128":
            self._set_combo_by_data(self._recon_combo, "fbp")
            self._set_combo_by_data(self._ung_method_combo, "fbp")
            self._set_combo_by_data(self._gated_method_combo, "fbp")
            self._set_combo_by_data(self._ung_filter_combo, "butterworth")
            self._ung_cutoff_spin.setValue(0.45)
            self._ung_order_spin.setValue(8)
            self._set_combo_by_data(self._gated_filter_combo, "butterworth")
            self._gated_cutoff_spin.setValue(0.40)
            self._gated_order_spin.setValue(10)
            self._iter_spin.setValue(3)
            self._subsets_spin.setValue(6)
            self._bg_check.setChecked(False)
            self._scatter_check.setChecked(False)
            self._post_check.setChecked(True)
            self._post_sigma_spin.setValue(0.45)
            self._denoise_plus_check.setChecked(False)
            self._ac_iter_check.setChecked(False)
            self._ac_mu_scale_spin.setValue(1.00)
        elif preset == "amylo360_hd":
            self._set_combo_by_data(self._recon_combo, "osem")
            self._set_combo_by_data(self._ung_method_combo, "osem")
            self._set_combo_by_data(self._gated_method_combo, "osem")
            self._set_combo_by_data(self._ung_filter_combo, "none")
            self._ung_cutoff_spin.setValue(0.50)
            self._ung_order_spin.setValue(1)
            self._set_combo_by_data(self._gated_filter_combo, "none")
            self._gated_cutoff_spin.setValue(0.50)
            self._gated_order_spin.setValue(1)
            self._iter_spin.setValue(3)
            self._subsets_spin.setValue(6)
            self._bg_check.setChecked(False)
            self._scatter_check.setChecked(False)
            self._post_check.setChecked(True)
            self._post_sigma_spin.setValue(0.35)
            self._denoise_plus_check.setChecked(True)
            self._denoise_plus_k_spin.setValue(0.20)
            self._ac_iter_check.setChecked(True)
            self._ac_mu_scale_spin.setValue(1.00)
        for widget in (
            self._ung_method_combo,
            self._ung_filter_combo,
            self._ung_cutoff_spin,
            self._ung_order_spin,
            self._gated_method_combo,
            self._gated_filter_combo,
            self._gated_cutoff_spin,
            self._gated_order_spin,
            self._iter_spin,
            self._subsets_spin,
            self._bg_check,
            self._scatter_check,
            self._scatter_k_spin,
            self._post_check,
            self._post_sigma_spin,
            self._denoise_plus_check,
            self._denoise_plus_k_spin,
        ):
            widget.setEnabled(manual)
        self._refresh_gated_controls()
        self._persist_ui_state()

    def _on_manual_method_changed(self, *_):
        method = str(self._ung_method_combo.currentData() or "fbp")
        self._set_combo_by_data(self._recon_combo, method)
        self._persist_ui_state()

    def _on_top_recon_changed(self, *_):
        if not hasattr(self, "_ung_method_combo"):
            return
        method = str(self._recon_combo.currentData() or "fbp")
        self._set_combo_by_data(self._ung_method_combo, method)
        self._persist_ui_state()

    def _refresh_gated_controls(self):
        gated_enabled = bool(self._study_is_gated and str(self._preset_combo.currentData() or "manual") == "manual")
        for widget in (self._gated_method_combo, self._gated_filter_combo, self._gated_cutoff_spin, self._gated_order_spin):
            widget.setEnabled(gated_enabled)
        self._gated_title_lbl.setText("Gated" if self._study_is_gated else "Gated (no disponible)")
        cardiac_idx = self._cuts_mode_combo.findData("cardiac")
        if cardiac_idx >= 0:
            model_item = self._cuts_mode_combo.model().item(cardiac_idx)
            if model_item is not None:
                model_item.setEnabled(bool(self._study_is_gated))
        if not self._study_is_gated and str(self._cuts_mode_combo.currentData() or "mixed") == "cardiac":
            self._set_combo_by_data(self._cuts_mode_combo, "tomo")

    def _build_recon_config(self):
        from core.raw_reconstruction import ProjectionFilterConfig, RawReconConfig

        preset = str(self._preset_combo.currentData() or "manual")
        method = str(self._ung_method_combo.currentData() or self._recon_combo.currentData() or "fbp")
        gated_method = str(self._gated_method_combo.currentData() or method) if self._study_is_gated else method
        if preset == "amylo360_std128":
            return RawReconConfig(
                reconstruction_method="fbp",
                gated_method="fbp",
                ungated_filter=ProjectionFilterConfig("butterworth", 0.45, 8),
                gated_filter=ProjectionFilterConfig("butterworth", 0.40, 10),
                post_filter_sigma_ungated_px=0.45,
                post_filter_sigma_gated_px=0.60,
                display_slice_step_px=2,
            )
        if preset == "amylo360_hd":
            return RawReconConfig(
                reconstruction_method="osem",
                gated_method="osem",
                ungated_filter=ProjectionFilterConfig("none", 0.50, 1),
                gated_filter=ProjectionFilterConfig("none", 0.50, 1),
                iterative_iterations=3,
                osem_subsets=6,
                post_filter_sigma_ungated_px=0.35,
                post_filter_sigma_gated_px=0.50,
                ungated_denoise_plus=True,
                ungated_denoise_plus_k=0.20,
                gated_denoise_plus=True,
                gated_denoise_plus_k=0.45,
                display_slice_step_px=1,
            )
        return RawReconConfig(
            reconstruction_method=method,
            gated_method=gated_method,
            ungated_filter=ProjectionFilterConfig(
                str(self._ung_filter_combo.currentData() or "butterworth"),
                float(self._ung_cutoff_spin.value()),
                int(self._ung_order_spin.value()),
            ),
            gated_filter=ProjectionFilterConfig(
                str(self._gated_filter_combo.currentData() or "butterworth"),
                float(self._gated_cutoff_spin.value()),
                int(self._gated_order_spin.value()),
            ),
            iterative_iterations=int(self._iter_spin.value()),
            osem_subsets=int(self._subsets_spin.value()),
            background_subtract=bool(self._bg_check.isChecked()),
            scatter_subtract=bool(self._scatter_check.isChecked()),
            scatter_k=float(self._scatter_k_spin.value()),
            post_filter_sigma_ungated_px=float(self._post_sigma_spin.value()) if self._post_check.isChecked() else 0.0,
            post_filter_sigma_gated_px=float(self._post_sigma_spin.value()) if self._study_is_gated and self._post_check.isChecked() else 0.0,
            ungated_denoise_plus=bool(self._denoise_plus_check.isChecked()),
            ungated_denoise_plus_k=float(self._denoise_plus_k_spin.value()),
            gated_denoise_plus=False,
            attenuation_correction=bool(self._ac_iter_check.isChecked()),
            attenuation_mu_scale=float(self._ac_mu_scale_spin.value()),
        )

    def _current_preset_label(self) -> str:
        return str(self._preset_combo.currentText() or "Manual")

    @staticmethod
    def _matrix_recommendation(shape: tuple[int, ...]) -> str:
        if len(shape) < 2:
            return "Matriz no determinada."
        rows, cols = int(shape[-2]), int(shape[-1])
        if rows == 128 and cols == 128:
            return "Matriz 128x128: objetivo recomendado para AMYLO/CT."
        if rows < 128 or cols < 128:
            return f"Matriz {rows}x{cols}: funciona, pero 128x128 facilita correlación con CT."
        return f"Matriz {rows}x{cols}: alta densidad de muestreo; vigilar ruido/tiempo de reconstrucción."

    @staticmethod
    def _fov_text(shape, spacing) -> str:
        if shape is None:
            return "shape=N/D spacing=N/D FOV=N/D"
        if spacing is None:
            return f"shape={tuple(shape)} spacing=N/D FOV=N/D"
        fov = tuple(float(shape[i]) * float(spacing[i]) for i in range(3))
        return (
            f"shape={tuple(shape)} spacing z/y/x="
            f"{spacing[0]:.3f}/{spacing[1]:.3f}/{spacing[2]:.3f} mm "
            f"FOV={fov[0]:.1f}/{fov[1]:.1f}/{fov[2]:.1f} mm"
        )

    def _append_grid_report(self):
        sp_shape = tuple(np.asarray(self._base_spect_volume).shape) if self._base_spect_volume is not None else None
        ct_shape = tuple(np.asarray(self._ct_volume).shape) if self._ct_volume is not None else None
        reg_shape = tuple(np.asarray(self._ct_registered).shape) if self._ct_registered is not None else None
        self._metrics.append("\n--- Grilla / FOV ---")
        self._metrics.append(f"SPECT: {self._fov_text(sp_shape, self._spect_spacing_zyx)}")
        self._metrics.append(f"CT original: {self._fov_text(ct_shape, self._ct_spacing_zyx)}")
        self._metrics.append(
            "Geometría DICOM: "
            f"SPECT={'OK' if self._spect_affine_ijk_to_lps is not None else 'N/D'} · "
            f"CT={'OK' if self._ct_affine_ijk_to_lps is not None else 'N/D'}"
        )
        if reg_shape is not None:
            self._metrics.append(f"CT registrado: shape={reg_shape} (misma grilla de display que SPECT)")

    def _on_visual_controls_changed(self):
        self._persist_ui_state()
        self._render_current_with_overlay()

    def _on_fusion_slider_changed(self, value: int):
        self._fusion_pct = int(value)
        self._fusion_lbl.setText(f"{int(value)}%")
        self._persist_ui_state()
        self._render_current_with_overlay()

    def _task_progress_start(self, message: str):
        self._progress.setValue(3)
        self._progress.setFormat(f"3% · {message}")
        QApplication.processEvents()

    def _task_progress_step(self, value: int, message: str):
        self._progress.setValue(int(np.clip(value, 0, 100)))
        self._progress.setFormat(f"{int(np.clip(value, 0, 100))}% · {message}")
        QApplication.processEvents()

    def _task_progress_done(self, message: str):
        self._progress.setValue(100)
        self._progress.setFormat(f"100% · {message}")
        QApplication.processEvents()

    def _mk_image_label(self, title: str) -> QLabel:
        lbl = QLabel(title)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setMinimumSize(300, 250)
        lbl.setStyleSheet("background:#0b1220; color:#94a3b8; border:1px solid #334155;")
        lbl.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if title.lower().startswith("ax"):
            lbl.wheelEvent = lambda ev, axis="axial": self._on_image_wheel(ev, axis)
            lbl.mousePressEvent = lambda ev, axis="axial": self._on_image_mouse_press(ev, axis)
            lbl.mouseMoveEvent = lambda ev, axis="axial": self._on_image_mouse_move(ev, axis)
            lbl.mouseReleaseEvent = lambda ev, axis="axial": self._on_image_mouse_release(ev, axis)
        elif title.lower().startswith("cor"):
            lbl.wheelEvent = lambda ev, axis="coronal": self._on_image_wheel(ev, axis)
            lbl.mousePressEvent = lambda ev, axis="coronal": self._on_image_mouse_press(ev, axis)
            lbl.mouseMoveEvent = lambda ev, axis="coronal": self._on_image_mouse_move(ev, axis)
            lbl.mouseReleaseEvent = lambda ev, axis="coronal": self._on_image_mouse_release(ev, axis)
        else:
            lbl.wheelEvent = lambda ev, axis="sagittal": self._on_image_wheel(ev, axis)
            lbl.mousePressEvent = lambda ev, axis="sagittal": self._on_image_mouse_press(ev, axis)
            lbl.mouseMoveEvent = lambda ev, axis="sagittal": self._on_image_mouse_move(ev, axis)
            lbl.mouseReleaseEvent = lambda ev, axis="sagittal": self._on_image_mouse_release(ev, axis)
        return lbl

    @staticmethod
    def _arr_to_pixmap(arr: np.ndarray) -> QPixmap:
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError("Se esperaba imagen 2D")
        mn, mx = float(np.min(a)), float(np.max(a))
        if mx - mn < 1e-9:
            norm = np.zeros_like(a, dtype=np.uint8)
        else:
            norm = np.clip((a - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([norm, norm, norm], axis=-1)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        mn, mx = float(np.min(a)), float(np.max(a))
        if mx - mn < 1e-9:
            return np.zeros_like(a, dtype=np.float64)
        return np.clip((a - mn) / (mx - mn), 0.0, 1.0)

    def _window_spect(self, arr: np.ndarray) -> np.ndarray:
        a = self._normalize(arr)
        lo = float(self._spect_win_low) / 100.0
        hi = float(self._spect_win_high) / 100.0
        if hi <= lo:
            hi = lo + 0.01
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    def _window_ct(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        mode = str(self._ct_window or "bone")
        if mode == "soft":
            lo, hi = -160.0, 240.0
        elif mode == "lung":
            lo, hi = -1000.0, 200.0
        elif mode == "full":
            lo, hi = float(np.percentile(a, 1.0)), float(np.percentile(a, 99.0))
        else:
            lo, hi = -200.0, 1000.0
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    def _apply_cmap(self, img: np.ndarray) -> np.ndarray:
        a = np.asarray(img, dtype=np.float64)
        cmap_name = str(self._spect_cmap_combo.currentText() or "hot") if hasattr(self, "_spect_cmap_combo") else "hot"
        try:
            import matplotlib as mpl
            cmap = mpl.colormaps[cmap_name]
            return np.asarray(cmap(np.clip(a, 0.0, 1.0))[..., :3], dtype=np.float64)
        except Exception:
            return np.stack([a, np.clip(a * 0.75, 0, 1), np.zeros_like(a)], axis=-1)

    @staticmethod
    def _pan_2d_center(img: np.ndarray, pan_yx: list[int] | tuple[int, int]) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim not in (2, 3):
            return arr
        dy, dx = int(pan_yx[0]), int(pan_yx[1])
        if dy == 0 and dx == 0:
            return arr
        return ndi.shift(arr, shift=(dy, dx, 0) if arr.ndim == 3 else (dy, dx), order=1, mode="constant", cval=0.0)

    @staticmethod
    def _zoom_2d_center(img: np.ndarray, zoom_pct: int) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim != 2:
            return arr
        z = max(0.05, float(zoom_pct) / 100.0)
        if abs(z - 1.0) < 1e-6:
            return arr
        out_shape = arr.shape
        scaled = ndi.zoom(arr, z, order=1)
        result = np.zeros(out_shape, dtype=np.float64)
        src_slices = []
        dst_slices = []
        for src_len, dst_len in zip(scaled.shape, out_shape):
            if src_len <= dst_len:
                src0 = 0
                src1 = src_len
                dst0 = (dst_len - src_len) // 2
                dst1 = dst0 + src_len
            else:
                src0 = (src_len - dst_len) // 2
                src1 = src0 + dst_len
                dst0 = 0
                dst1 = dst_len
            src_slices.append(slice(src0, src1))
            dst_slices.append(slice(dst0, dst1))
        result[tuple(dst_slices)] = scaled[tuple(src_slices)]
        return result

    @classmethod
    def _make_overlay_rgb(cls, base2d: np.ndarray, mask2d: np.ndarray | None, alpha: float) -> np.ndarray:
        g = cls._normalize(base2d)
        rgb = np.stack([g, g, g], axis=-1)
        if mask2d is None:
            return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        m = np.asarray(mask2d, dtype=bool)
        a = float(np.clip(alpha, 0.0, 1.0))
        if np.any(m) and a > 0.0:
            rgb[m, 0] = (1.0 - a) * rgb[m, 0] + a * 1.0
            rgb[m, 1] = (1.0 - a) * rgb[m, 1] + a * 0.25
            rgb[m, 2] = (1.0 - a) * rgb[m, 2] + a * 0.25
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def _make_qc_rgb(self, spect2d: np.ndarray, ct2d: np.ndarray, mode: str, split_pct: int) -> np.ndarray:
        sp = np.clip(np.asarray(spect2d, dtype=np.float64), 0.0, 1.0)
        ct = np.clip(np.asarray(ct2d, dtype=np.float64), 0.0, 1.0)
        rgb_sp = self._apply_cmap(sp)
        rgb_ct = np.stack([ct, ct, ct], axis=-1)
        mode = str(mode or "off").lower()

        if mode == "fusion":
            mix = float(np.clip(getattr(self, "_fusion_pct", 55), 0, 100)) / 100.0
            # Alpha espacial: el porcentaje de fusión define cuánto pesa el SPECT,
            # pero la señal baja queda translúcida para no tapar CT de fondo.
            alpha = np.clip(sp * 1.35, 0.0, 1.0)[..., None] * mix
            out = (1.0 - alpha) * rgb_ct + alpha * rgb_sp
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        if mode == "edges":
            out = rgb_ct.copy()
            sp_edge = self._edge_mask(sp, 88.0)
            ct_edge = self._edge_mask(ct, 88.0)
            out[ct_edge, 0] = 0.1
            out[ct_edge, 1] = 1.0
            out[ct_edge, 2] = 0.25
            out[sp_edge, 0] = 1.0
            out[sp_edge, 1] = 0.15
            out[sp_edge, 2] = 0.1
            both = sp_edge & ct_edge
            out[both, 0] = 1.0
            out[both, 1] = 1.0
            out[both, 2] = 0.1
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        if mode == "split":
            x = int(np.clip(round(sp.shape[1] * (float(split_pct) / 100.0)), 1, sp.shape[1] - 1))
            out = rgb_sp.copy()
            out[:, x:, :] = rgb_ct[:, x:, :]
            out[:, max(0, x - 1):min(sp.shape[1], x + 1), 0] = 1.0
            out[:, max(0, x - 1):min(sp.shape[1], x + 1), 1] = 0.8
            out[:, max(0, x - 1):min(sp.shape[1], x + 1), 2] = 0.2
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        if mode == "checker":
            out = rgb_sp.copy()
            tile = 14
            yy, xx = np.indices(sp.shape)
            checker = ((yy // tile) + (xx // tile)) % 2 == 0
            out[checker] = rgb_ct[checker]
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        if mode == "contours":
            out = rgb_sp.copy()
            gy, gx = np.gradient(ct)
            edge = np.hypot(gx, gy)
            thr = float(np.percentile(edge, 92.0))
            em = edge >= thr
            out[em, 0] = 0.2
            out[em, 1] = 1.0
            out[em, 2] = 0.3
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        return np.clip(rgb_sp * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _edge_mask(img: np.ndarray, percentile: float = 90.0) -> np.ndarray:
        a = np.asarray(img, dtype=np.float64)
        if a.ndim != 2 or a.size == 0:
            return np.zeros_like(a, dtype=bool)
        gy, gx = np.gradient(a)
        edge = np.hypot(gx, gy)
        if not np.any(edge > 0):
            return np.zeros_like(a, dtype=bool)
        return edge >= float(np.percentile(edge, percentile))

    @staticmethod
    def _rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    @staticmethod
    def _blend_mask_over_rgb(rgb: np.ndarray, mask2d: np.ndarray | None, alpha: float) -> np.ndarray:
        arr = np.asarray(rgb, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("RGB inválido para blend de máscara")
        if mask2d is None:
            return np.clip(arr, 0, 255).astype(np.uint8)
        m = np.asarray(mask2d, dtype=bool)
        a = float(np.clip(alpha, 0.0, 1.0))
        if np.any(m) and a > 0.0:
            arr[m, 0] = (1.0 - a) * arr[m, 0] + a * 255.0
            arr[m, 1] = (1.0 - a) * arr[m, 1] + a * 64.0
            arr[m, 2] = (1.0 - a) * arr[m, 2] + a * 64.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _render_preview(self, volume: np.ndarray):
        pv = self._slices_preview_at(volume)
        ax = self._arr_to_pixmap(pv["axial"]).scaled(
            self._axial_lbl.width(), self._axial_lbl.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        co = self._arr_to_pixmap(pv["coronal"]).scaled(
            self._cor_lbl.width(), self._cor_lbl.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        sa = self._arr_to_pixmap(pv["sagittal"]).scaled(
            self._sag_lbl.width(), self._sag_lbl.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._axial_lbl.setPixmap(ax)
        self._cor_lbl.setPixmap(co)
        self._sag_lbl.setPixmap(sa)

    def _update_slice_controls(self):
        if self._current_volume is None:
            return
        shape = tuple(int(v) for v in np.asarray(self._current_volume).shape[:3])
        sliders = ((self._slice_z, "axial", shape[0], self._slice_z_lbl, "z"),
                   (self._slice_y, "coronal", shape[1], self._slice_y_lbl, "y"),
                   (self._slice_x, "sagittal", shape[2], self._slice_x_lbl, "x"))
        for slider, key, n, lbl, name in sliders:
            slider.blockSignals(True)
            slider.setRange(0, max(0, n - 1))
            val = self._slice_idx.get(key, n // 2)
            if val <= 0 or val >= n:
                val = n // 2
            self._slice_idx[key] = int(val)
            slider.setValue(int(val))
            slider.setEnabled(n > 1)
            lbl.setText(f"{name} {int(val) + 1}/{n}")
            slider.blockSignals(False)

    def _on_slice_slider_changed(self):
        self._slice_idx["axial"] = int(self._slice_z.value())
        self._slice_idx["coronal"] = int(self._slice_y.value())
        self._slice_idx["sagittal"] = int(self._slice_x.value())
        if self._current_volume is not None:
            shape = tuple(int(v) for v in np.asarray(self._current_volume).shape[:3])
            self._slice_z_lbl.setText(f"z {self._slice_idx['axial'] + 1}/{shape[0]}")
            self._slice_y_lbl.setText(f"y {self._slice_idx['coronal'] + 1}/{shape[1]}")
            self._slice_x_lbl.setText(f"x {self._slice_idx['sagittal'] + 1}/{shape[2]}")
        self._render_current_with_overlay()

    def _on_zoom_changed(self):
        self._spect_zoom_pct = int(self._spect_zoom_slider.value())
        self._ct_zoom_pct = int(self._ct_zoom_slider.value())
        self._spect_zoom_lbl.setText(f"SPECT {self._spect_zoom_pct}%")
        self._ct_zoom_lbl.setText(f"CT {self._ct_zoom_pct}%")
        self._render_current_with_overlay()

    def _on_spect_range_changed(self, low: int, high: int):
        self._spect_win_low = int(low)
        self._spect_win_high = int(high)
        self._spect_range_lbl.setText(f"Base {self._spect_win_low}% · Top {self._spect_win_high}%")
        self._render_current_with_overlay()

    def _set_spect_range(self, low: int, high: int):
        self._spect_range_slider.set_values(int(low), int(high))

    def _on_ct_window_changed(self):
        self._ct_window = str(self._ct_window_combo.currentData() or "bone")
        self._render_current_with_overlay()

    def _on_image_wheel(self, event, axis: str):
        delta = int(event.angleDelta().y())
        if delta == 0:
            event.accept()
            return
        step = 1 if delta > 0 else -1
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if ctrl:
            self._ct_zoom_slider.setValue(int(np.clip(self._ct_zoom_slider.value() + step * 5, 50, 200)))
            self._status.setText(f"Ctrl+rueda: zoom CT {self._ct_zoom_slider.value()}%")
            self._render_current_with_overlay()
            event.accept()
            return
        if shift:
            self._spect_zoom_slider.setValue(int(np.clip(self._spect_zoom_slider.value() + step * 5, 50, 200)))
            self._status.setText(f"Shift+rueda: zoom SPECT {self._spect_zoom_slider.value()}%")
            self._render_current_with_overlay()
            event.accept()
            return
        slider = {"axial": self._slice_z, "coronal": self._slice_y, "sagittal": self._slice_x}.get(axis)
        if slider is not None and slider.isEnabled():
            slider.setValue(int(np.clip(slider.value() + step, slider.minimum(), slider.maximum())))
        event.accept()

    def _on_image_mouse_press(self, event, axis: str):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if not (ctrl or shift):
            return
        self._drag_state = {
            "axis": axis,
            "target": "ct" if ctrl else "spect",
            "pos": event.position().toPoint(),
        }
        event.accept()

    def _on_image_mouse_move(self, event, axis: str):
        if not self._drag_state or self._drag_state.get("axis") != axis:
            return
        pos = event.position().toPoint()
        old = self._drag_state["pos"]
        dx = int(pos.x() - old.x())
        dy = int(pos.y() - old.y())
        if dx == 0 and dy == 0:
            return
        target = self._drag_state.get("target")
        if target == "ct" and self._ct_auto_registered is not None:
            dz, dyw, dxw = self._drag_delta_to_world_zyx(axis, dx, dy)
            self._nudge_z.setValue(float(np.clip(self._nudge_z.value() + dz, -64.0, 64.0)))
            self._nudge_y.setValue(float(np.clip(self._nudge_y.value() + dyw, -64.0, 64.0)))
            self._nudge_x.setValue(float(np.clip(self._nudge_x.value() + dxw, -64.0, 64.0)))
            self._status.setText(
                "Ctrl+arrastre (triangulado): "
                f"CT Δ(z,y,x)=({self._nudge_z.value():.1f},{self._nudge_y.value():.1f},{self._nudge_x.value():.1f}) px"
            )
        else:
            pan = self._spect_pan_px
            pan[axis][0] += dy
            pan[axis][1] += dx
            self._status.setText(
                f"Shift+arrastre: SPECT pan {axis}=({pan[axis][0]},{pan[axis][1]}) px"
            )
            self._render_current_with_overlay()
        self._drag_state["pos"] = pos
        event.accept()

    def _on_image_mouse_release(self, event, axis: str):
        self._drag_state = None
        event.accept()

    def _axis_label(self, axis: str) -> QLabel:
        if axis == "axial":
            return self._axial_lbl
        if axis == "coronal":
            return self._cor_lbl
        return self._sag_lbl

    def _drag_delta_to_world_zyx(self, axis: str, dx_px: int, dy_px: int) -> tuple[float, float, float]:
        """Convierte delta de mouse en una vista 2D a delta global z/y/x (px volumen)."""
        vol = self._base_spect_volume if self._base_spect_volume is not None else self._current_volume
        if vol is None:
            return (0.0, 0.0, 0.0)
        shape = tuple(int(v) for v in np.asarray(vol).shape)
        if len(shape) != 3:
            return (0.0, 0.0, 0.0)

        lbl = self._axis_label(axis)
        pm = lbl.pixmap()
        w = int(pm.width()) if pm is not None else max(1, int(lbl.width()))
        h = int(pm.height()) if pm is not None else max(1, int(lbl.height()))
        w = max(1, w)
        h = max(1, h)

        sens = max(0.05, float(getattr(self, "_ct_drag_sensitivity", 1.0)))

        if axis == "axial":
            sx = float(shape[2]) / float(w)
            sy = float(shape[1]) / float(h)
            return (0.0, float(dy_px) * sy * sens, float(dx_px) * sx * sens)
        if axis == "coronal":
            sx = float(shape[2]) / float(w)
            sz = float(shape[0]) / float(h)
            return (float(dy_px) * sz * sens, 0.0, float(dx_px) * sx * sens)
        # sagittal: horizontal=y, vertical=z
        sy = float(shape[1]) / float(w)
        sz = float(shape[0]) / float(h)
        return (float(dy_px) * sz * sens, float(dx_px) * sy * sens, 0.0)

    def _slices_preview_at(self, volume: np.ndarray) -> dict[str, np.ndarray]:
        vol = np.asarray(volume, dtype=np.float64)
        if vol.ndim != 3:
            return central_slices_preview(vol)
        z = int(np.clip(self._slice_idx.get("axial", vol.shape[0] // 2), 0, vol.shape[0] - 1))
        y = int(np.clip(self._slice_idx.get("coronal", vol.shape[1] // 2), 0, vol.shape[1] - 1))
        x = int(np.clip(self._slice_idx.get("sagittal", vol.shape[2] // 2), 0, vol.shape[2] - 1))
        return {
            "axial": self._pan_2d_center(self._zoom_2d_center(self._window_spect(vol[int(np.clip(z + self._spect_view_offset.get("axial", 0), 0, vol.shape[0] - 1))]), self._spect_zoom_pct), self._spect_pan_px["axial"]),
            "coronal": self._pan_2d_center(self._zoom_2d_center(self._window_spect(vol[:, int(np.clip(y + self._spect_view_offset.get("coronal", 0), 0, vol.shape[1] - 1)), :]), self._spect_zoom_pct), self._spect_pan_px["coronal"]),
            "sagittal": self._pan_2d_center(self._zoom_2d_center(self._window_spect(vol[:, :, int(np.clip(x + self._spect_view_offset.get("sagittal", 0), 0, vol.shape[2] - 1))]), self._spect_zoom_pct), self._spect_pan_px["sagittal"]),
        }

    def _reset_view_offsets(self):
        self._spect_view_offset = {"axial": 0, "coronal": 0, "sagittal": 0}
        self._ct_view_offset = {"axial": 0, "coronal": 0, "sagittal": 0}
        self._spect_pan_px = {"axial": [0, 0], "coronal": [0, 0], "sagittal": [0, 0]}
        self._ct_pan_px = {"axial": [0, 0], "coronal": [0, 0], "sagittal": [0, 0]}
        self._spect_zoom_slider.blockSignals(True)
        self._ct_zoom_slider.blockSignals(True)
        self._spect_zoom_slider.setValue(100)
        self._ct_zoom_slider.setValue(100)
        self._spect_zoom_slider.blockSignals(False)
        self._ct_zoom_slider.blockSignals(False)
        self._spect_range_slider.set_values(0, 100)
        self._spect_zoom_pct = 100
        self._ct_zoom_pct = 100
        self._spect_win_low = 0
        self._spect_win_high = 100
        self._spect_zoom_lbl.setText("SPECT 100%")
        self._ct_zoom_lbl.setText("CT 100%")
        self._status.setText("Offsets, pan, zoom y rango visual SPECT/CT reseteados.")
        self._render_current_with_overlay()

    def _ct_slices_preview_at(self, volume: np.ndarray) -> dict[str, np.ndarray]:
        vol = np.asarray(volume, dtype=np.float64)
        if vol.ndim != 3:
            return self._slices_preview_at(vol)
        z = int(np.clip(self._slice_idx.get("axial", vol.shape[0] // 2) + self._ct_view_offset.get("axial", 0), 0, vol.shape[0] - 1))
        y = int(np.clip(self._slice_idx.get("coronal", vol.shape[1] // 2) + self._ct_view_offset.get("coronal", 0), 0, vol.shape[1] - 1))
        # El eje AP/PA del CT queda invertido respecto del SPECT en esta exportación:
        # al pedir el corte y=N del SPECT, el CT equivalente está en y espejado.
        y_ct = int(vol.shape[1] - 1 - y)
        x = int(np.clip(self._slice_idx.get("sagittal", vol.shape[2] // 2) + self._ct_view_offset.get("sagittal", 0), 0, vol.shape[2] - 1))
        axial = self._window_ct(vol[z])
        coronal = self._window_ct(vol[:, y_ct, :])
        sagittal = self._window_ct(vol[:, :, x])

        # Corrección visual por eje (consenso clínico actual):
        # - CT y global AP/PA espejado frente al SPECT.
        # - Axial: flip vertical.
        # - Eje X (sagital): flip horizontal.
        # - Coronal (eje Y): usa y_ct espejado, sin flip extra en pantalla.
        axial = np.flipud(axial)
        sagittal = np.fliplr(sagittal)

        return {
            "axial": self._pan_2d_center(self._zoom_2d_center(axial, self._ct_zoom_pct), self._ct_pan_px["axial"]),
            "coronal": self._pan_2d_center(self._zoom_2d_center(coronal, self._ct_zoom_pct), self._ct_pan_px["coronal"]),
            "sagittal": self._pan_2d_center(self._zoom_2d_center(sagittal, self._ct_zoom_pct), self._ct_pan_px["sagittal"]),
        }

    def _render_triplet(self, left_title: str, left_arr: np.ndarray, mid_title: str, mid_arr: np.ndarray, right_title: str, right_arr: np.ndarray):
        for lbl, title, arr in (
            (self._axial_lbl, left_title, left_arr),
            (self._cor_lbl, mid_title, mid_arr),
            (self._sag_lbl, right_title, right_arr),
        ):
            pix = self._arr_to_pixmap(self._as_display_2d(arr)).scaled(
                lbl.width(), lbl.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl.setPixmap(pix)
            lbl.setToolTip(title)

    @staticmethod
    def _as_display_2d(arr: np.ndarray) -> np.ndarray:
        img = np.asarray(arr, dtype=np.float64)
        if img.ndim == 4:
            img = np.mean(img, axis=0)
        while img.ndim > 2:
            img = img[img.shape[0] // 2]
        if img.ndim != 2:
            raise ValueError(f"Imagen inválida para display: {img.shape}")
        return img

    @classmethod
    def _montage_2d(cls, arr: np.ndarray, max_slices: int = 12) -> np.ndarray:
        vol = np.asarray(arr, dtype=np.float64)
        if vol.ndim == 4:
            vol = np.mean(vol, axis=0)
        if vol.ndim == 2:
            return vol
        while vol.ndim > 3:
            vol = vol[vol.shape[0] // 2]
        if vol.ndim != 3:
            return cls._as_display_2d(vol)
        n = int(vol.shape[0])
        count = min(max_slices, n)
        idxs = np.linspace(0, n - 1, count).round().astype(int) if count > 1 else np.array([n // 2])
        tiles = [cls._safe_pad_2d(cls._normalize(vol[int(i)])) for i in idxs]
        cols = min(4, len(tiles))
        rows = int(np.ceil(len(tiles) / cols))
        h = max(t.shape[0] for t in tiles)
        w = max(t.shape[1] for t in tiles)
        canvas = np.zeros((rows * h, cols * w), dtype=np.float64)
        for k, tile in enumerate(tiles):
            r = k // cols
            c = k % cols
            canvas[r * h:r * h + tile.shape[0], c * w:c * w + tile.shape[1]] = tile
        return canvas

    @staticmethod
    def _safe_pad_2d(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Tile inválido: {arr.shape}")
        return arr

    def _render_selected_view(self):
        mode = str(self._view_combo.currentData() or "mpr") if hasattr(self, "_view_combo") else "mpr"
        qc_mode = str(self._qc_mode.currentData() or "off") if hasattr(self, "_qc_mode") else "off"
        if self._ct_registered is not None and qc_mode != "off":
            self._render_current_with_overlay()
            return
        if mode == "tomo" and self._recon_bundle is not None and self._recon_bundle.tomo_cuts:
            if self._current_volume is not None:
                self._render_preview(self._current_volume)
            else:
                cuts = self._recon_bundle.tomo_cuts
                self._render_triplet("Axial tomo", cuts["axial"], "Coronal tomo", cuts["coronal"], "Sagital tomo", cuts["sagittal"])
            return
        if mode == "cardiac" and self._recon_bundle is not None and self._recon_bundle.cardiac_axes:
            axes = self._recon_bundle.cardiac_axes
            self._render_triplet(
                "SA montage",
                self._montage_2d(axes["SA"]),
                "HLA montage",
                self._montage_2d(axes["HLA"]),
                "VLA montage",
                self._montage_2d(axes["VLA"]),
            )
            return
        self._render_current_with_overlay()

    def _render_current_with_overlay(self):
        if self._current_volume is None:
            return
        alpha = float(self._blend_slider.value()) / 100.0
        self._blend_lbl.setText(f"{self._blend_slider.value()}%")
        self._qc_split_lbl.setText(f"{self._qc_split_slider.value()}%")
        self._fusion_pct = int(self._fusion_slider.value()) if hasattr(self, "_fusion_slider") else int(getattr(self, "_fusion_pct", 55))
        if hasattr(self, "_fusion_lbl"):
            self._fusion_lbl.setText(f"{self._fusion_pct}%")

        pv = self._slices_preview_at(self._current_volume)
        mode = str(self._qc_mode.currentData() or "off")

        ct_prev = None
        if self._ct_registered is not None and mode != "off":
            ct_prev = self._ct_slices_preview_at(np.asarray(self._ct_registered, dtype=np.float64))

        if ct_prev is not None:
            ax_rgb = self._make_qc_rgb(pv["axial"], ct_prev["axial"], mode, self._qc_split_slider.value())
            co_rgb = self._make_qc_rgb(pv["coronal"], ct_prev["coronal"], mode, self._qc_split_slider.value())
            sa_rgb = self._make_qc_rgb(pv["sagittal"], ct_prev["sagittal"], mode, self._qc_split_slider.value())
            if self._bone_mask is not None and alpha > 0.0:
                bm = self._slices_preview_at(np.asarray(self._bone_mask, dtype=np.float64))
                ax_rgb = self._blend_mask_over_rgb(ax_rgb, bm.get("axial"), alpha)
                co_rgb = self._blend_mask_over_rgb(co_rgb, bm.get("coronal"), alpha)
                sa_rgb = self._blend_mask_over_rgb(sa_rgb, bm.get("sagittal"), alpha)
        elif self._bone_mask is not None:
            bm = self._slices_preview_at(np.asarray(self._bone_mask, dtype=np.float64))
            ax_rgb = self._make_overlay_rgb(pv["axial"], bm.get("axial"), alpha)
            co_rgb = self._make_overlay_rgb(pv["coronal"], bm.get("coronal"), alpha)
            sa_rgb = self._make_overlay_rgb(pv["sagittal"], bm.get("sagittal"), alpha)
        else:
            self._render_preview(self._current_volume)
            return

        ax = self._rgb_to_pixmap(ax_rgb).scaled(
            self._axial_lbl.width(), self._axial_lbl.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        co = self._rgb_to_pixmap(co_rgb).scaled(
            self._cor_lbl.width(), self._cor_lbl.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        sa = self._rgb_to_pixmap(sa_rgb).scaled(
            self._sag_lbl.width(), self._sag_lbl.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._axial_lbl.setPixmap(ax)
        self._cor_lbl.setPixmap(co)
        self._sag_lbl.setPixmap(sa)

    def _load_spect(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar DICOM SPECT",
            os.path.expanduser("~"),
            "DICOM (*.dcm *.DCM);;Todos (*)",
        )
        if not path:
            return

        try:
            self._task_progress_start("Cargando SPECT...")
            # Carga inicial siempre rápida en FBP para feedback inmediato en UI.
            method = "fbp"
            selected_method = str(self._recon_combo.currentData() or "fbp")
            self._analysis = run_amyloid_spect_analysis(path, recon_method=method)
            self._task_progress_step(55, "Preparando volumen base...")
            self._study_is_gated = int(getattr(self._analysis, "n_gates", 1) or 1) >= 2
            self._spect_spacing_zyx = getattr(self._analysis, "spacing_zyx", None)
            self._spect_affine_ijk_to_lps = getattr(self._analysis, "affine_ijk_to_lps", None)
            self._current_spect_path = path
            self._restore_study_ui_state()
            self._current_volume = np.asarray(self._analysis.volume, dtype=np.float64)
            self._base_spect_volume = np.asarray(self._analysis.volume, dtype=np.float64)
            self._bone_mask = None
            self._qc_mode.setEnabled(False)
            self._qc_split_slider.setEnabled(False)
            self._fusion_slider.setEnabled(False)
            self._update_slice_controls()
            self._render_preview(self._current_volume)
            self._btn_bone.setEnabled(True)
            self._btn_recon_pipeline.setEnabled(True)
            self._btn_load_ct.setEnabled(True)
            self._btn_load_ct_dir.setEnabled(True)
            self._btn_load_att.setEnabled(True)
            self._btn_register.setEnabled(self._ct_volume is not None)
            self._btn_apply_ac.setEnabled(self._att_map_volume is not None)
            self._btn_export_axes_dcm.setEnabled(False)
            self._blend_slider.setEnabled(False)
            self._refresh_gated_controls()
            self._task_progress_step(88, "Renderizando vistas...")
            self._status.setText(
                f"SPECT cargado: {'crudo' if self._analysis.was_raw else 'reconstruido'} · "
                f"shape {self._current_volume.shape} · "
                f"{'gatillado' if self._study_is_gated else 'no gatillado'}"
            )
            self._write_metrics(self._analysis.metrics, self._analysis.notes)
            if selected_method != "fbp":
                self._metrics.append(
                    f"- Carga inicial forzada a FBP rápido (selección actual: {selected_method.upper()}). "
                    "Para método final usar 'Recon + cortes'."
                )
            if self._ct_path:
                self._metrics.append(f"\nCT asociado guardado: {self._ct_path}")
                self._metrics.append(self._matrix_recommendation(self._current_volume.shape))
            self._persist_ui_state()
            self._task_progress_done("SPECT listo")
        except Exception as exc:
            self._progress.setFormat("Error")
            self._status.setText(f"Error cargando SPECT: {exc}")
            self._metrics.setPlainText(f"Error:\n{exc}")

    def _reconstruct_with_perf_pipeline(self):
        if self._analysis is None:
            self._status.setText("Cargar primero un DICOM SPECT.")
            return
        if self._recon_thread is not None:
            self._status.setText("Reconstrucción en curso...")
            return
        try:
            cfg = self._build_recon_config()
            cuts_mode = str(self._cuts_mode_combo.currentData() or "mixed")
            if not self._study_is_gated and cuts_mode == "cardiac":
                cuts_mode = "tomo"
                self._set_combo_by_data(self._cuts_mode_combo, "tomo")
                self._metrics.append(
                    "\nAMYLO no gatillado: modo 'Cardíacos' no disponible; se usan cortes tomográficos."
                )
            self._pending_recon_cfg = cfg
            self._pending_recon_preset_label = self._current_preset_label()
            self._pending_recon_cuts_mode = cuts_mode
            ac_iter_enabled = bool(getattr(cfg, "attenuation_correction", False))
            self._pending_recon_ac_iter = ac_iter_enabled

            att_rs_for_iter = None
            att_px_cm = None
            if ac_iter_enabled:
                if self._att_map_volume is None:
                    self._status.setText("AC iterativa: cargar primero ATT MAP o desactivar el toggle.")
                    return
                att_rs_for_iter, notes_rs = resample_volume_to_spect_grid(
                    self._att_map_volume,
                    np.asarray(self._analysis.volume, dtype=np.float64),
                    source_spacing_zyx=self._att_spacing_zyx,
                    spect_spacing_zyx=self._spect_spacing_zyx,
                    source_affine_ijk_to_lps=self._att_affine_ijk_to_lps,
                    spect_affine_ijk_to_lps=self._spect_affine_ijk_to_lps,
                    fill_value=0.0,
                )
                self._att_map_registered = np.asarray(att_rs_for_iter, dtype=np.float64)
                px_mm = float(self._spect_spacing_zyx[2]) if self._spect_spacing_zyx is not None else 6.8
                att_px_cm = max(1e-4, px_mm / 10.0)
                self._metrics.append("\n--- AC iterativa (pre-recon) ---")
                for n in notes_rs:
                    self._metrics.append(f"- {n}")
                self._metrics.append(
                    f"- ATT MAP para iterativa listo: shape={self._att_map_registered.shape}, px={px_mm:.3f} mm, μ-scale={float(getattr(cfg, 'attenuation_mu_scale', 1.0)):.2f}"
                )

            self._set_recon_busy(True)
            self._recon_thread = QThread(self)
            self._recon_worker = _AmyloidReconWorker(
                self._analysis.source_path,
                cfg,
                cuts_mode,
                attenuation_mu_map=att_rs_for_iter,
                attenuation_pixel_size_cm=att_px_cm,
            )
            self._recon_worker.moveToThread(self._recon_thread)
            self._recon_thread.started.connect(self._recon_worker.run)
            self._recon_worker.progress.connect(self._on_recon_progress)
            self._recon_worker.finished.connect(self._on_recon_finished)
            self._recon_worker.failed.connect(self._on_recon_failed)
            self._recon_worker.finished.connect(self._recon_thread.quit)
            self._recon_worker.failed.connect(self._recon_thread.quit)
            self._recon_thread.finished.connect(self._cleanup_recon_thread)
            self._recon_thread.start()
        except Exception as exc:
            self._set_recon_busy(False)
            self._status.setText(f"Error iniciando recon+cortes: {exc}")

    def _cancel_reconstruction(self):
        if self._recon_worker is not None:
            self._recon_worker.cancel()
            self._status.setText("Cancelando reconstrucción...")
            self._progress.setFormat("Cancelando...")

    def _on_recon_progress(self, value: int, message: str):
        self._progress.setValue(int(value))
        self._progress.setFormat(f"{int(value)}% · {message}")
        self._status.setText(message)

    def _on_recon_finished(self, bundle):
        try:
            self._recon_bundle = bundle
            self._base_spect_volume = np.asarray(self._recon_bundle.ungated_volume, dtype=np.float64)
            self._current_volume = np.asarray(self._recon_bundle.ungated_volume, dtype=np.float64)
            self._spect_spacing_zyx = getattr(self._recon_bundle, "spacing_zyx", self._spect_spacing_zyx)
            self._spect_affine_ijk_to_lps = getattr(self._recon_bundle, "affine_ijk_to_lps", self._spect_affine_ijk_to_lps)
            self._bone_mask = None
            self._update_slice_controls()
            self._btn_export_axes_dcm.setEnabled(bool(self._recon_bundle.cardiac_axes))
            self._view_combo.setEnabled(bool(self._recon_bundle.tomo_cuts or self._recon_bundle.cardiac_axes))
            if self._recon_bundle.cardiac_axes:
                self._set_combo_by_data(self._view_combo, "cardiac")
            elif self._recon_bundle.tomo_cuts:
                self._set_combo_by_data(self._view_combo, "tomo")
            else:
                self._set_combo_by_data(self._view_combo, "mpr")
            self._render_selected_view()

            notes = list(self._recon_bundle.notes)
            sa_shape = self._recon_bundle.cardiac_axes.get("SA", np.zeros((0,))).shape if self._recon_bundle.cardiac_axes else ()
            tomo_keys = ", ".join(self._recon_bundle.tomo_cuts.keys()) if self._recon_bundle.tomo_cuts else "-"
            self._status.setText(
                f"Pipeline perfusión listo · {self._pending_recon_preset_label} · modo={self._pending_recon_cuts_mode} · "
                f"ungated {self._current_volume.shape} · SA {sa_shape}"
            )
            self._metrics.append("\n--- Recon + cortes (perfusión) ---")
            self._metrics.append(f"- preset: {self._pending_recon_preset_label}")
            cfg = self._pending_recon_cfg
            self._metrics.append(
                f"- config: método={cfg.reconstruction_method.upper()}, "
                f"gated={str(cfg.gated_method or cfg.reconstruction_method).upper()}, "
                f"iter={int(cfg.iterative_iterations)}, subsets={int(cfg.osem_subsets)}"
            )
            self._metrics.append(
                f"- AC iterativa: {'ON' if self._pending_recon_ac_iter else 'OFF'}"
            )
            self._metrics.append(f"- modo cortes: {self._pending_recon_cuts_mode}")
            self._metrics.append(f"- tomográficos: {tomo_keys}")
            self._metrics.append(f"- {self._matrix_recommendation(self._current_volume.shape)}")
            self._append_grid_report()
            for n in notes:
                self._metrics.append(f"- {n}")
            self._persist_ui_state()
            self._progress.setValue(100)
            self._progress.setFormat("100% · Reconstrucción y cortes listos")
            self._set_recon_busy(False)
        except Exception as exc:
            self._on_recon_failed(str(exc))

    def _on_recon_failed(self, message: str):
        self._set_recon_busy(False)
        if "cancelada" in str(message).lower():
            self._status.setText("Reconstrucción cancelada.")
            self._progress.setFormat("Cancelada")
        else:
            self._status.setText(f"Error en recon+cortes: {message}")
            self._progress.setFormat("Error")

    def _cleanup_recon_thread(self):
        if self._recon_worker is not None:
            self._recon_worker.deleteLater()
        if self._recon_thread is not None:
            self._recon_thread.deleteLater()
        self._recon_worker = None
        self._recon_thread = None

    def _set_recon_busy(self, busy: bool):
        for widget in (
            self._btn_load,
            self._preset_combo,
            self._recon_combo,
            self._cuts_mode_combo,
            self._btn_load_ct,
            self._btn_load_ct_dir,
            self._btn_register,
            self._btn_load_att,
            self._btn_apply_ac,
            self._btn_bone,
            self._btn_recon_pipeline,
            self._btn_export_axes_dcm,
        ):
            widget.setEnabled(not busy and widget.isEnabled())
        if busy:
            self._progress.setValue(0)
            self._progress.setFormat("0% · Preparando reconstrucción...")
            self._btn_cancel_recon.setEnabled(True)
        else:
            self._btn_load.setEnabled(True)
            self._preset_combo.setEnabled(True)
            self._cuts_mode_combo.setEnabled(True)
            self._on_preset_changed()
            self._btn_load_ct.setEnabled(self._analysis is not None)
            self._btn_load_ct_dir.setEnabled(self._analysis is not None)
            self._btn_load_att.setEnabled(self._analysis is not None)
            self._btn_register.setEnabled(self._ct_volume is not None and self._base_spect_volume is not None)
            self._btn_apply_ac.setEnabled(self._att_map_volume is not None and self._base_spect_volume is not None)
            self._btn_bone.setEnabled(self._current_volume is not None)
            self._btn_recon_pipeline.setEnabled(self._analysis is not None)
            self._btn_export_axes_dcm.setEnabled(self._recon_bundle is not None and bool(self._recon_bundle.cardiac_axes))
            self._btn_cancel_recon.setEnabled(False)

    def _export_axes_dicom(self):
        if self._recon_bundle is None or not self._recon_bundle.cardiac_axes:
            self._status.setText("No hay ejes cardíacos para exportar.")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta para ejes DICOM",
            os.path.expanduser("~"),
        )
        if not out_dir:
            return
        try:
            exported = export_amyloid_cardiac_axes_dicom(
                self._recon_bundle,
                out_dir,
                base_name="AMYLO_SPECT",
            )
            if exported:
                self._metrics.append("\n--- Export DICOM ejes ---")
                for k, v in exported.items():
                    self._metrics.append(f"- {k}: {v}")
                self._status.setText(f"Ejes DICOM exportados ({len(exported)}).")
            else:
                self._status.setText("No se exportó: no hay ejes cardíacos disponibles.")
        except Exception as exc:
            self._status.setText(f"Error exportando ejes DICOM: {exc}")

    def _load_ct(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar DICOM CT",
            os.path.expanduser("~"),
            "DICOM (*.dcm *.DCM);;Todos (*)",
        )
        if not path:
            return
        self._load_ct_path(path)

    def _load_ct_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta con serie CT",
            os.path.dirname(self._ct_path) if self._ct_path else os.path.expanduser("~"),
        )
        if not path:
            return
        self._load_ct_path(path)

    def _load_ct_path(self, path: str):
        try:
            self._task_progress_start("Cargando CT...")
            ct = load_ct_volume_from_path(path)
            self._task_progress_step(60, "Remapeando CT y actualizando estado...")
            self._ct_volume = np.asarray(ct.volume, dtype=np.float64)
            self._ct_spacing_zyx = getattr(ct, "spacing_zyx", None)
            self._ct_affine_ijk_to_lps = getattr(ct, "affine_ijk_to_lps", None)
            self._ct_path = ct.source_path
            self._ct_registered = None
            self._ct_auto_registered = None
            self._reset_ct_nudge(update_view=False)
            self._qc_mode.setEnabled(False)
            self._qc_split_slider.setEnabled(False)
            self._fusion_slider.setEnabled(False)
            self._btn_register.setEnabled(self._base_spect_volume is not None)
            self._status.setText(f"CT cargado · {ct.series_description} · shape {self._ct_volume.shape}")
            self._metrics.append("\n--- CT cargado ---")
            for note in ct.notes:
                self._metrics.append(f"- {note}")
            self._append_grid_report()
            self._metrics.append("Ejecutar 'Registrar CT↔SPECT'.")
            self._persist_ui_state()
            self._task_progress_done("CT listo")
        except Exception as exc:
            self._progress.setFormat("Error")
            self._status.setText(f"Error cargando CT: {exc}")

    def _load_att_map(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar DICOM ATT MAP",
            os.path.dirname(self._att_path) if self._att_path else os.path.expanduser("~"),
            "DICOM (*.dcm *.DCM);;Todos (*)",
        )
        if not path:
            return
        try:
            self._task_progress_start("Cargando ATT MAP...")
            att = load_attenuation_map_from_path(path)
            self._task_progress_step(70, "Actualizando ATT MAP en estado...")
            self._att_map_volume = np.asarray(att.volume, dtype=np.float64)
            self._att_map_registered = None
            self._att_spacing_zyx = getattr(att, "spacing_zyx", None)
            self._att_affine_ijk_to_lps = getattr(att, "affine_ijk_to_lps", None)
            self._att_path = att.source_path
            self._btn_apply_ac.setEnabled(self._base_spect_volume is not None)
            self._status.setText(f"ATT MAP cargado · {att.series_description} · shape {self._att_map_volume.shape}")
            self._metrics.append("\n--- ATT MAP cargado ---")
            for note in att.notes:
                self._metrics.append(f"- {note}")
            self._task_progress_done("ATT MAP listo")
        except Exception as exc:
            self._progress.setFormat("Error")
            self._status.setText(f"Error cargando ATT MAP: {exc}")

    def _apply_ac_prototype(self):
        if self._base_spect_volume is None:
            self._status.setText("Cargar/reconstruir primero un SPECT.")
            return
        if self._att_map_volume is None:
            self._status.setText("Cargar primero ATT MAP.")
            return
        try:
            self._task_progress_start("Aplicando AC (prototipo/Chang)...")
            att_rs, notes_rs = resample_volume_to_spect_grid(
                self._att_map_volume,
                self._base_spect_volume,
                source_spacing_zyx=self._att_spacing_zyx,
                spect_spacing_zyx=self._spect_spacing_zyx,
                source_affine_ijk_to_lps=self._att_affine_ijk_to_lps,
                spect_affine_ijk_to_lps=self._spect_affine_ijk_to_lps,
                fill_value=0.0,
            )
            self._task_progress_step(45, "Calculando AC prototipo...")
            self._att_map_registered = np.asarray(att_rs, dtype=np.float64)
            corrected_proto, notes_proto = apply_attenuation_correction_prototype(
                self._base_spect_volume,
                self._att_map_registered,
                mu_scale=0.12,
            )
            self._task_progress_step(70, "Calculando AC Chang...")
            corrected, notes_ac = apply_attenuation_correction_chang(
                self._base_spect_volume,
                self._att_map_registered,
                spect_spacing_zyx=self._spect_spacing_zyx,
                mu_scale=1.0,
                n_angles=36,
            )
            self._task_progress_step(90, "Renderizando volumen corregido...")
            self._current_volume = np.asarray(corrected, dtype=np.float64)
            self._bone_mask = None
            self._update_slice_controls()
            self._render_current_with_overlay()
            self._status.setText("AC Chang aplicada con ATT MAP (experimental).")
            self._metrics.append("\n--- Corrección por atenuación (prototipo) ---")
            for n in notes_rs:
                self._metrics.append(f"- {n}")
            for n in notes_proto:
                self._metrics.append(f"- [AC prototipo] {n}")
            for n in notes_ac:
                self._metrics.append(f"- [AC Chang] {n}")
            self._task_progress_done("AC aplicada")
        except Exception as exc:
            self._progress.setFormat("Error")
            self._status.setText(f"Error aplicando AC prototipo: {exc}")

    def _register_ct_to_spect(self):
        if self._ct_volume is None or self._base_spect_volume is None:
            self._status.setText("Cargar primero SPECT y CT.")
            return
        try:
            self._task_progress_start("Registrando CT↔SPECT...")
            ct_reg, shift_zyx, notes = register_ct_to_spect_rigid(
                self._ct_volume,
                self._base_spect_volume,
                ct_spacing_zyx=self._ct_spacing_zyx,
                spect_spacing_zyx=self._spect_spacing_zyx,
                ct_affine_ijk_to_lps=self._ct_affine_ijk_to_lps,
                spect_affine_ijk_to_lps=self._spect_affine_ijk_to_lps,
                refine_ncc=True,
                ncc_search_radius_zyx=(2, 4, 4),
            )
            self._task_progress_step(45, "Ajustando orientación CT...")
            ct_reg, orient_flags, orient_notes = align_ct_orientation_to_spect(
                ct_reg,
                self._base_spect_volume,
                try_flip_x=True,
                try_flip_y=True,
                try_flip_z=False,
                try_flip_xy=True,
                min_score_gain=0.03,
                min_abs_score=0.05,
            )
            self._task_progress_step(70, "Refinando traslación fina...")
            ct_reg, fine_shift_zyx, fine_notes = refine_ct_to_spect_translation(
                ct_reg,
                self._base_spect_volume,
                search_radius_zyx=(3, 8, 8),
                ct_bone_hu_threshold=200.0,
                spect_focus_percentile=85.0,
            )
            self._task_progress_step(90, "Renderizando registro...")
            self._ct_auto_registered = np.asarray(ct_reg, dtype=np.float64)
            self._ct_registered = np.asarray(ct_reg, dtype=np.float64)
            for spin in (self._nudge_z, self._nudge_y, self._nudge_x):
                spin.blockSignals(True)
                spin.setValue(0.0)
                spin.blockSignals(False)
                spin.setEnabled(True)
            self._btn_reset_nudge.setEnabled(True)
            self._qc_mode.setEnabled(True)
            self._qc_split_slider.setEnabled(True)
            self._fusion_slider.setEnabled(True)
            self._set_combo_by_data(self._qc_mode, "fusion")
            self._status.setText(
                "Registro CT↔SPECT listo "
                f"Δ(z,y,x)=({shift_zyx[0]:.1f},{shift_zyx[1]:.1f},{shift_zyx[2]:.1f})"
            )
            self._metrics.append("\n--- Registro CT↔SPECT ---")
            for n in notes:
                self._metrics.append(n)
            for n in orient_notes:
                self._metrics.append(n)
            for n in fine_notes:
                self._metrics.append(n)
            self._metrics.append(
                "Auto-orientación CT aplicada: "
                f"rot={int(orient_flags.get('rot_k', 0))*90}°, "
                f"flip_z={bool(orient_flags.get('flip_z', False))}, "
                f"flip_y={bool(orient_flags.get('flip_y', False))}, "
                f"flip_x={bool(orient_flags.get('flip_x', False))}."
            )
            self._metrics.append(
                "Refinamiento fino incremental Δ(z,y,x)="
                f"({float(fine_shift_zyx[0]):.1f},{float(fine_shift_zyx[1]):.1f},{float(fine_shift_zyx[2]):.1f}) px."
            )
            self._append_grid_report()
            self._render_current_with_overlay()
            self._task_progress_done("Registro CT↔SPECT listo")
        except Exception as exc:
            self._progress.setFormat("Error")
            self._status.setText(f"Error en registro CT↔SPECT: {exc}")

    def _apply_ct_nudge(self):
        if self._ct_auto_registered is None:
            return
        shift = (float(self._nudge_z.value()), float(self._nudge_y.value()), float(self._nudge_x.value()))
        self._ct_registered = ndi.shift(self._ct_auto_registered, shift=shift, order=1, mode="nearest")
        self._status.setText(f"Ajuste CT manual aplicado Δ(z,y,x)=({shift[0]:.1f},{shift[1]:.1f},{shift[2]:.1f}) px")
        self._render_current_with_overlay()
        self._persist_ui_state()

    def _reset_ct_nudge(self, update_view: bool = True):
        for spin in (self._nudge_z, self._nudge_y, self._nudge_x):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        if self._ct_auto_registered is not None:
            self._ct_registered = np.asarray(self._ct_auto_registered, dtype=np.float64)
        if update_view:
            self._render_current_with_overlay()
            self._persist_ui_state()

    def _apply_bone_suppression(self):
        if self._current_volume is None:
            return
        try:
            self._task_progress_start("Aplicando sustracción ósea...")
            ct_vol = None
            if self._ct_check.isChecked():
                ct_vol = self._ct_registered if self._ct_registered is not None else self._ct_volume
            res = apply_visual_bone_suppression(
                self._base_spect_volume if self._base_spect_volume is not None else self._current_volume,
                ct_volume=ct_vol,
            )
            self._task_progress_step(80, "Renderizando sustracción ósea...")
            self._current_volume = np.asarray(res.enhanced_volume, dtype=np.float64)
            self._bone_mask = np.asarray(res.bone_mask, dtype=np.uint8)
            self._blend_slider.setEnabled(True)
            self._render_current_with_overlay()
            self._status.setText(f"Sustracción ósea visual aplicada ({res.method}).")
            self._metrics.append("\n--- Sustracción ósea ---")
            self._task_progress_done("Sustracción ósea lista")
            self._metrics.append("\n".join(res.notes))
        except Exception as exc:
            self._status.setText(f"Error en sustracción ósea: {exc}")

    def _write_metrics(self, metrics: dict, notes: list[str]):
        txt = ["Métricas 3D proxy (experimental):"]
        for k in ("lv_mean", "lv_peak", "bg_mean", "ratio_lv_bg", "heterogeneity_cv", "p80_threshold"):
            if k in metrics:
                txt.append(f"- {k}: {float(metrics[k]):.4f}")
        txt.append("")
        txt.append("Notas:")
        for n in notes:
            txt.append(f"- {n}")
        self._metrics.setPlainText("\n".join(txt))
