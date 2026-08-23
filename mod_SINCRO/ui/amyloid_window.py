# -*- coding: utf-8 -*-
"""Ventana de amiloidosis cardíaca: visor de cuadrantes + análisis ROI + HMR + Perugini.

Dos modos:
1. Visor de cuadrantes: layouts de 4/8/9/12/16 imágenes con selección, colormaps y filtros.
2. Análisis ROI: imagen planar + ROIs draggable + HMR + Perugini score.

Referencias:
- HMR ≥1.5: POSITIVO (sugiere ATTR).
- HMR 1.0–1.5: EQUÍVOCO (complementar con SPECT o repeat a 3h).
- HMR <1.0: NEGATIVO.
- Perugini: score visual 0-3 con referencia integrada.
"""
from __future__ import annotations

import numpy as np
import base64
import json
from datetime import datetime, date, time, timedelta

from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QSettings
from PyQt6.QtGui import QMouseEvent, QPainter, QPen, QColor, QBrush, QPixmap, QImage
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QWidget, QSizePolicy, QMessageBox, QStackedWidget,
    QSlider, QFrame, QFileDialog, QTextEdit, QInputDialog, QCheckBox, QDoubleSpinBox,
)
import os

from core.amyloid_planar import ROICircle, compute_hmr, PERUGINI_SCORES
from core.amyloid_layouts import (
    LAYOUT_NAMES,
    layout_3q, layout_4q, layout_6q, layout_8q, layout_9q, layout_12q, layout_12q_3x4, layout_16q,
)
from ui.quadrant_viewer import QuadrantViewer
from ui.anatomical_3d_panel import Anatomical3DPanel
from ui.amyloid_spect_panel import AmyloidSpectPanel


class ROIDragWidget(QWidget):
    """Widget con dos ROIs circulares draggable sobre la imagen."""

    roiChanged = pyqtSignal(int, float, float, float)  # roi_id, cy, cx, radius

    def __init__(self, image: np.ndarray):
        super().__init__()
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image = np.asarray(image, dtype=np.float64)
        self._pixmap = None
        self._zoom = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._rois = [
            {"cy": 0.4 * image.shape[0], "cx": 0.4 * image.shape[1], "radius": 12.0, "color": "#ff6666", "name": "Corazón"},
            {"cy": 0.6 * image.shape[0], "cx": 0.6 * image.shape[1], "radius": 12.0, "color": "#38bdf8", "name": "Mediastino"},
        ]
        self._show_aux_rois = False
        self._show_mirror_roi = False
        self._drag_roi = -1

    def set_aux_rois_visible(self, visible: bool):
        self._show_aux_rois = bool(visible)
        self.update()

    def set_mirror_roi_visible(self, visible: bool):
        self._show_mirror_roi = bool(visible)
        self.update()

    def _is_roi_visible(self, idx: int) -> bool:
        return idx < 2 or self._show_aux_rois

    def set_zoom(self, zoom: float):
        self._zoom = max(0.2, min(20.0, zoom))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#0b1220"))

        h, w = self._image.shape
        ww, wh = self.width(), self.height()
        scale = min(ww / max(1, w), wh / max(1, h)) * self._zoom
        img_w, img_h = int(w * scale), int(h * scale)
        ox = (ww - img_w) // 2
        oy = (wh - img_h) // 2

        # Normalizar la imagen a 0..1 para renderizar.
        norm = self._image / max(float(self._image.max()), 1e-8) if self._image.size else self._image
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
        rgb[..., 1] = rgb[..., 0]
        rgb[..., 2] = rgb[..., 0]
        from PyQt6.QtGui import QImage, QPixmap
        qimg = QImage(rgb.tobytes(), w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        # Aplicar colormap o escala si se desea (por ahora gris).
        painter.drawPixmap(ox, oy, img_w, img_h, pix)

        # Dibujar ROIs.
        for i, roi in enumerate(self._rois):
            if not self._is_roi_visible(i):
                continue
            rcx = ox + roi["cx"] * scale
            rcy = oy + roi["cy"] * scale
            rr = roi["radius"] * scale
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            color = QColor(roi["color"])
            if i == self._drag_roi:
                painter.setPen(QPen(color, 3, Qt.PenStyle.SolidLine))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(rcx, rcy), rr, rr)
            else:
                style = Qt.PenStyle.DashLine if i < 2 else Qt.PenStyle.DotLine
                painter.setPen(QPen(color, 2.0, style))
                fill_alpha = 60 if i < 2 else 95
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), fill_alpha)))
                painter.drawEllipse(QPointF(rcx, rcy), rr, rr)
            # Centro de ROI para referencia visual fina.
            painter.setPen(QPen(color, 1.6, Qt.PenStyle.SolidLine))
            painter.drawLine(int(rcx - 4), int(rcy), int(rcx + 4), int(rcy))
            painter.drawLine(int(rcx), int(rcy - 4), int(rcx), int(rcy + 4))
            # Etiqueta.
            painter.setPen(QPen(color, 1.5))
            painter.drawText(int(rcx + rr + 5), int(rcy), roi["name"])

        # Leyenda breve cuando ROIs auxiliares están activas.
        if self._show_aux_rois:
            legend_x = 8
            legend_y = 8
            painter.setPen(QPen(QColor("#cbd5e1"), 1.0))
            painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
            painter.drawRoundedRect(legend_x, legend_y, 260, 72, 6, 6)
            painter.setPen(QPen(QColor("#fbbf24"), 1.5))
            painter.drawText(legend_x + 10, legend_y + 17, "Esternón: amarillo")
            painter.setPen(QPen(QColor("#a78bfa"), 1.5))
            painter.drawText(legend_x + 10, legend_y + 33, "Costilla: violeta")
            painter.setPen(QPen(QColor("#34d399"), 1.5))
            painter.drawText(legend_x + 10, legend_y + 49, "Fondo est. 1: verde")
            painter.setPen(QPen(QColor("#22d3ee"), 1.5))
            painter.drawText(legend_x + 10, legend_y + 65, "Fondo est. 2: cian")

        # Vista previa ROI espejo de costilla (izquierda), usada por EXCLUDE_BONE.
        if self._show_mirror_roi and len(self._rois) >= 4:
            rib = self._rois[3]
            m_cx = (w - 1) - float(rib["cx"])
            m_cy = float(rib["cy"])
            m_r = float(rib["radius"])
            m_rcx = ox + m_cx * scale
            m_rcy = oy + m_cy * scale
            m_rr = m_r * scale
            mirror_color = QColor("#c084fc")
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(mirror_color, 2.0, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(m_rcx, m_rcy), m_rr, m_rr)
            painter.setPen(QPen(mirror_color, 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(int(m_rcx - 4), int(m_rcy), int(m_rcx + 4), int(m_rcy))
            painter.drawLine(int(m_rcx), int(m_rcy - 4), int(m_rcx), int(m_rcy + 4))
            painter.setPen(QPen(mirror_color, 1.5))
            painter.drawText(int(m_rcx + m_rr + 5), int(m_rcy), "Costilla espejo")

    def mousePressEvent(self, event: QMouseEvent):
        for i, roi in enumerate(self._rois):
            if not self._is_roi_visible(i):
                continue
            rcx = event.position().x()
            rcy = event.position().y()
            scale = self._scale()
            ox = (self.width() - self._image.shape[1] * scale) // 2
            oy = (self.height() - self._image.shape[0] * scale) // 2
            dist = np.sqrt((rcx - ox - roi["cx"] * scale) ** 2 + (rcy - oy - roi["cy"] * scale) ** 2)
            if dist < roi["radius"] * 1.3 * scale:
                self._drag_roi = i
                break
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_roi < 0:
            return
        scale = self._scale()
        ox = (self.width() - self._image.shape[1] * scale) // 2
        oy = (self.height() - self._image.shape[0] * scale) // 2
        self._rois[self._drag_roi]["cx"] = (event.position().x() - ox) / scale
        self._rois[self._drag_roi]["cy"] = (event.position().y() - oy) / scale
        self.roiChanged.emit(self._drag_roi, self._rois[self._drag_roi]["cy"], self._rois[self._drag_roi]["cx"], self._rois[self._drag_roi]["radius"])
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_roi = -1

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        # Doble clic = no hacer nada (la rueda ajusta el radio).
        pass

    def wheelEvent(self, event):
        """Rueda del mouse = ajustar radio del ROI bajo el cursor (ambos ROIs juntos)."""
        scale = self._scale()
        ox = (self.width() - self._image.shape[1] * scale) // 2
        oy = (self.height() - self._image.shape[0] * scale) // 2
        delta = 1 if event.angleDelta().y() > 0 else -1
        for i, roi in enumerate(self._rois):
            if not self._is_roi_visible(i):
                continue
            rcx = event.position().x()
            rcy = event.position().y()
            dist = np.sqrt((rcx - ox - roi["cx"] * scale) ** 2 + (rcy - oy - roi["cy"] * scale) ** 2)
            if dist < roi["radius"] * 1.5 * scale:
                new_radius = max(3.0, min(64.0, roi["radius"] + delta * 1.0))
                if i < 2:
                    # Corazón + mediastino mantienen radio común.
                    self._rois[0]["radius"] = new_radius
                    self._rois[1]["radius"] = new_radius
                    self.roiChanged.emit(i, self._rois[i]["cy"], self._rois[i]["cx"], new_radius)
                else:
                    self._rois[i]["radius"] = new_radius
                    self.roiChanged.emit(i, self._rois[i]["cy"], self._rois[i]["cx"], new_radius)
                break
        self.update()

    def _scale(self) -> float:
        h, w = self._image.shape
        ww, wh = self.width(), self.height()
        return min(ww / max(1, w), wh / max(1, h)) * self._zoom


class AmyloidWindow(QDialog):
    """Ventana de amiloidosis: visor de cuadrantes + análisis ROI + HMR + Perugini."""

    # Layouts disponibles: nombre → función constructora.
    _LAYOUT_BUILDERS = {
        3: layout_3q,
        4: layout_4q,
        6: layout_6q,
        8: layout_8q,
        9: layout_9q,
        12: layout_12q,
        1234: layout_12q_3x4,
        16: layout_16q,
    }

    def __init__(self, parent=None, image=None, study=None):
        super().__init__(parent)
        self.setWindowTitle("SINCRO — Amiloidosis")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        self.resize(1040, 660)
        self.setMinimumSize(900, 560)
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)
        self._image = image          # imagen renderizada (puede ser RGB)
        self._original_image = image  # imagen original 2D para análisis ROI
        self._study = study
        self._metadata = {
            "patient": str(getattr(study, "patient_name", "") or "N/D"),
            "date": str(getattr(study, "study_date", "") or "N/D"),
            "series": str(getattr(study, "series_description", "") or "N/D"),
        }
        # Fuente canónica independiente del layout. Cada tiempo conserva sus tres
        # vistas aunque el usuario cambie la grilla, pagine o vuelva del análisis.
        self._time_images: dict[str, dict[str, dict | None]] = {
            "1h": {"ap": None, "oai": None, "lat": None},
            "3h": {"ap": None, "oai": None, "lat": None},
        }
        self._current_layout_n = 4
        self._current_mode = "visor"  # "visor" | "analisis"
        self._page_offset = 0  # índice de inicio de la página actual
        self._processed_images: dict[str, dict[str, np.ndarray]] = {"1h": {}, "3h": {}}
        self._roi_state: dict[str, list[dict] | None] = {"1h": None, "3h": None}
        self._roi_state_oai: dict[str, list[dict] | None] = {"1h": None, "3h": None}
        self._perugini_by_time: dict[str, int] = {}
        self._perugini_confirmed_by_time: dict[str, bool] = {}
        self._qbone_mode_by_time: dict[str, str] = {"1h": "auto", "3h": "auto"}
        self._time_hours_by_label: dict[str, float] = {"1h": 1.0, "3h": 3.0}
        self._active_time: str | None = None
        self._active_view_role: str = "ap"
        self._washout_data: dict[str, dict] = {}  # tiempo_label → {hmr, heart_counts, mediastinum_counts}
        self._oai_washout_data: dict[str, dict] = {}  # tiempo_label → {heart_counts, heart_area_px}
        self._early_dynamic: dict | None = None
        self._kinetic_result = None
        self._quadrant_state: dict[int, dict] = {}
        self._linked_spect_ct = None
        self._layout_12q3x4_lat_hidden = False
        self._exclude_bone_enabled = False
        self._exclude_bone_method = "mean_subtract"
        self._exclude_bone_asym_thresh = 0.35
        self._exclude_sternum_enabled = False
        self._exclude_sternum_asym_thresh = 0.35
        self._use_scatter_planar = False
        self._scatter_planar_k = 1.0
        self._result_view_mode = "original"

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Info del paciente ──────────────────────────────────────
        patient = self._metadata["patient"]
        date = self._metadata["date"]
        series = self._metadata["series"]
        self._info_lbl = QLabel(f"Paciente: {patient}  ·  Fecha: {date}  ·  Serie: {series}")
        self._info_lbl.setStyleSheet("font-size: 11px; color: #94a3b8; padding: 2px 0;")
        root.addWidget(self._info_lbl)

        # ── Toolbar ────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        # Selector de layout.
        toolbar.addWidget(QLabel("Layout:"))
        self._layout_combo = QComboBox()
        for key, name in LAYOUT_NAMES:
            if key == "12q_3x4":
                n = 1234
            else:
                n = int(key.replace("q", ""))
            self._layout_combo.addItem(name, n)
        self._layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        toolbar.addWidget(self._layout_combo)

        toolbar.addWidget(QLabel("Plantilla informe:"))
        self._report_template_combo = QComboBox()
        self._report_template_combo.addItem("Auto", "auto")
        self._report_template_combo.addItem("AMILO Clínico Completo", "AMILO Clínico Completo")
        self._report_template_combo.addItem("AMILO Planar", "AMILO Planar")
        self._report_template_combo.addItem("AMILO SPECT", "AMILO SPECT")
        self._report_template_combo.addItem("AMILO Básico", "AMILO Básico")
        self._report_template_combo.currentIndexChanged.connect(lambda _=0: self._persist_user_state())
        toolbar.addWidget(self._report_template_combo)

        self._btn_report_template_preview = QPushButton("Preview plantillas")
        self._btn_report_template_preview.setToolTip("Vista previa visual para elegir plantilla de informe")
        self._btn_report_template_preview.clicked.connect(self._open_report_template_preview)
        toolbar.addWidget(self._btn_report_template_preview)

        self._btn_report_template_matrix = QPushButton("Matriz escenarios")
        self._btn_report_template_matrix.setToolTip("Compara en una sola vista cómo se arma el informe por escenario clínico")
        self._btn_report_template_matrix.clicked.connect(self._open_report_template_matrix)
        toolbar.addWidget(self._btn_report_template_matrix)

        btn_load_1h = QPushButton("Cargar imágenes 1h")
        btn_load_1h.clicked.connect(lambda: self._load_time_images("1h"))
        toolbar.addWidget(btn_load_1h)
        btn_load_3h = QPushButton("Cargar imágenes 3h")
        btn_load_3h.clicked.connect(lambda: self._load_time_images("3h"))
        toolbar.addWidget(btn_load_3h)
        btn_load_auto = QPushButton("Washout automático (metadata)")
        btn_load_auto.setToolTip("Detecta 1h/3h automáticamente desde metadata DICOM (mantiene modo manual disponible)")
        btn_load_auto.clicked.connect(self._load_washout_auto)
        toolbar.addWidget(btn_load_auto)
        btn_load_dynamic = QPushButton("Cargar dinámico temprano")
        btn_load_dynamic.setToolTip("Método experimental: dinámico 0–5 min para TAC y localización cardíaca")
        btn_load_dynamic.clicked.connect(self._load_early_dynamic)
        toolbar.addWidget(btn_load_dynamic)
        btn_kinetic_help = QPushButton("Ayuda cinética")
        btn_kinetic_help.clicked.connect(self._show_kinetic_help)
        toolbar.addWidget(btn_kinetic_help)
        btn_anatomical_3d = QPushButton("3D anatómico")
        btn_anatomical_3d.setToolTip("Visualización anatómica 3D (experimental)")
        btn_anatomical_3d.clicked.connect(self._open_anatomical_3d)
        toolbar.addWidget(btn_anatomical_3d)
        btn_spect_3d = QPushButton("SPECT 3D")
        btn_spect_3d.setToolTip("Análisis SPECT amiloidosis 3D (fase 2, experimental)")
        btn_spect_3d.clicked.connect(self._open_amyloid_spect_3d)
        toolbar.addWidget(btn_spect_3d)

        toolbar.addStretch(1)

        # Botón modo análisis (solo visible si hay imagen single).
        self._btn_mode = QPushButton("Análisis ROI →")
        self._btn_mode.clicked.connect(self._toggle_mode)
        toolbar.addWidget(self._btn_mode)

        root.addLayout(toolbar)

        # ── Línea separadora ───────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #334155;")
        root.addWidget(sep)

        # ── Contenido principal: stacked widget ────────────────────
        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        # ── Página 0: Visor de cuadrantes ──────────────────────────
        page_visor = QWidget()
        visor_layout = QHBoxLayout(page_visor)
        visor_layout.setContentsMargins(0, 0, 0, 0)
        visor_layout.setSpacing(6)

        self._quadrant_viewer = QuadrantViewer()
        self._quadrant_viewer.quadrantSelected.connect(self._on_quadrant_selected)
        self._quadrant_viewer.quadrantLabelEditRequested.connect(self._edit_quadrant_label)
        visor_layout.addWidget(self._quadrant_viewer, 1)

        # Sidebar de controles del cuadrante seleccionado.
        sidebar = QVBoxLayout()
        sidebar.setSpacing(6)
        self._lbl_css = "color: #e2e8f0;"

        lbl = QLabel("Cuadrante seleccionado:")
        lbl.setStyleSheet(self._lbl_css)
        sidebar.addWidget(lbl)
        self._lbl_sel_quad = QLabel("#1")
        self._lbl_sel_quad.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")
        sidebar.addWidget(self._lbl_sel_quad)

        lbl = QLabel("Colormap:")
        lbl.setStyleSheet(self._lbl_css)
        sidebar.addWidget(lbl)
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["grey", "hot", "cool", "viridis"])
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        sidebar.addWidget(self._cmap_combo)

        lbl = QLabel("Ventana baja (%):")
        lbl.setStyleSheet(self._lbl_css)
        sidebar.addWidget(lbl)
        self._win_low_slider = QSlider(Qt.Orientation.Horizontal)
        self._win_low_slider.setRange(0, 95)
        self._win_low_slider.setValue(0)
        self._win_low_slider.valueChanged.connect(self._on_window_changed)
        sidebar.addWidget(self._win_low_slider)

        lbl = QLabel("Ventana alta (%):")
        lbl.setStyleSheet(self._lbl_css)
        sidebar.addWidget(lbl)
        self._win_high_slider = QSlider(Qt.Orientation.Horizontal)
        self._win_high_slider.setRange(5, 100)
        self._win_high_slider.setValue(100)
        self._win_high_slider.valueChanged.connect(self._on_window_changed)
        sidebar.addWidget(self._win_high_slider)

        lbl = QLabel("Filtros:")
        lbl.setStyleSheet(self._lbl_css)
        sidebar.addWidget(lbl)
        btn_smooth = QPushButton("Suavizar")
        btn_smooth.clicked.connect(lambda: self._toggle_filter("smooth"))
        sidebar.addWidget(btn_smooth)
        btn_invert = QPushButton("Invertir")
        btn_invert.clicked.connect(lambda: self._toggle_filter("invert"))
        sidebar.addWidget(btn_invert)
        btn_eq = QPushButton("Ecualizar")
        btn_eq.clicked.connect(lambda: self._toggle_filter("equalize"))
        sidebar.addWidget(btn_eq)

        sidebar.addStretch(1)

        # Botón para borrar el cuadrante seleccionado.
        self._btn_delete = QPushButton("Borrar cuadrante")
        self._btn_delete.setStyleSheet("background: #dc2626; color: white; font-weight: bold; padding: 4px;")
        self._btn_delete.clicked.connect(self._delete_selected_quadrant)
        sidebar.addWidget(self._btn_delete)

        # Swap: intercambiar imágenes entre cuadrantes.
        self._swap_mode = False
        self._swap_first = -1
        self._btn_swap = QPushButton("Swap")
        self._btn_swap.setStyleSheet("background: #d97706; color: white; font-weight: bold; padding: 4px;")
        self._btn_swap.clicked.connect(self._toggle_swap_mode)
        sidebar.addWidget(self._btn_swap)

        # Botón para abrir la imagen seleccionada en modo análisis ROI.
        self._btn_analyze = QPushButton("Analizar ROI")
        self._btn_analyze.clicked.connect(self._analyze_selected)
        sidebar.addWidget(self._btn_analyze)

        self._lbl_filters_active = QLabel("Activos: ninguno")
        self._lbl_filters_active.setStyleSheet("color:#94a3b8; font-size:10px;")
        self._lbl_filters_active.setWordWrap(True)
        sidebar.addWidget(self._lbl_filters_active)

        self._washout_preview = QLabel()
        self._washout_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._washout_preview.setWordWrap(True)
        self._washout_preview.setMinimumHeight(90)
        self._washout_preview.setStyleSheet("color:#94a3b8; border:1px solid #334155; padding:4px;")
        sidebar.addWidget(self._washout_preview)
        self._kinetic_status = QLabel("Cinética experimental: dinámico temprano no cargado.")
        self._kinetic_status.setWordWrap(True)
        self._kinetic_status.setStyleSheet("color:#94a3b8; border:1px solid #334155; padding:4px; font-size:10px;")
        sidebar.addWidget(self._kinetic_status)

        # Bloque AL (obligatorio para validar ATTR-CM en informe).
        lbl_al = QLabel("Exclusión AL (informe):")
        lbl_al.setStyleSheet(self._lbl_css)
        sidebar.addWidget(lbl_al)

        combo_css = (
            "QComboBox { background: #1e293b; color: #e2e8f0; border: 1px solid #475569; "
            "padding: 3px; border-radius: 4px; font-size: 11px; } "
            "QComboBox QAbstractItemView { background: #1e293b; color: #e2e8f0; selection-background-color: #2563eb; }"
        )

        self._al_status_combo = QComboBox()
        self._al_status_combo.addItem("PENDIENTE / NO INFORMADO", "pending")
        self._al_status_combo.addItem("EXCLUIDA", "excluded")
        self._al_status_combo.addItem("NO EXCLUIDA", "not_excluded")
        self._al_status_combo.setStyleSheet(combo_css)
        self._al_status_combo.currentIndexChanged.connect(lambda _=0: self._persist_user_state())
        sidebar.addWidget(self._al_status_combo)

        self._free_light_chain_combo = QComboBox()
        self._free_light_chain_combo.addItem("Cadenas livianas: No informado", "unknown")
        self._free_light_chain_combo.addItem("Cadenas livianas: Normales", "normal")
        self._free_light_chain_combo.addItem("Cadenas livianas: Alteradas", "abnormal")
        self._free_light_chain_combo.setStyleSheet(combo_css)
        self._free_light_chain_combo.currentIndexChanged.connect(lambda _=0: self._persist_user_state())
        sidebar.addWidget(self._free_light_chain_combo)

        self._immunofix_combo = QComboBox()
        self._immunofix_combo.addItem("Inmunofijación: No informada", "unknown")
        self._immunofix_combo.addItem("Inmunofijación: Negativa", "negative")
        self._immunofix_combo.addItem("Inmunofijación: Positiva", "positive")
        self._immunofix_combo.setStyleSheet(combo_css)
        self._immunofix_combo.currentIndexChanged.connect(lambda _=0: self._persist_user_state())
        sidebar.addWidget(self._immunofix_combo)

        # Paginación: anterior/siguiente.
        pag_layout = QHBoxLayout()
        btn_prev = QPushButton("◀")
        btn_prev.clicked.connect(self._prev_page)
        pag_layout.addWidget(btn_prev)
        self._lbl_pagination = QLabel("Página 1/1")
        self._lbl_pagination.setStyleSheet("color: #94a3b8; font-size: 11px;")
        pag_layout.addWidget(self._lbl_pagination)
        btn_next = QPushButton("▶")
        btn_next.clicked.connect(self._next_page)
        pag_layout.addWidget(btn_next)
        sidebar.addLayout(pag_layout)

        sidebar_frame = QFrame()
        sidebar_frame.setLayout(sidebar)
        sidebar_frame.setFixedWidth(260)
        sidebar_frame.setStyleSheet("QFrame { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 8px; }")
        visor_layout.addWidget(sidebar_frame)

        self._stack.addWidget(page_visor)

        # ── Página 1: Análisis ROI (modo clásico) ──────────────────
        page_analysis = QWidget()
        page_analysis.setStyleSheet("QLabel { color:#000000; } QCheckBox { color:#000000; }")
        analysis_layout = QVBoxLayout(page_analysis)
        analysis_layout.setContentsMargins(0, 0, 0, 0)

        self._roi_widget = ROIDragWidget(image if image is not None else np.zeros((64, 64)))
        self._roi_widget.roiChanged.connect(self._update_hmr)
        analysis_layout.addWidget(self._roi_widget, 1)

        self._lbl_hmr = QLabel("HMR = N/D")
        self._lbl_hmr.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._lbl_hmr.setMinimumHeight(24)
        self._lbl_hmr.setStyleSheet("font-size:16px; font-weight:bold; color:#ffffff; background:#000000; padding:4px 8px;")
        analysis_layout.addWidget(self._lbl_hmr)

        self._lbl_class = QLabel("")
        self._lbl_class.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._lbl_class.setMinimumHeight(22)
        self._lbl_class.setStyleSheet("font-size:12px; font-weight:600; color:#ffffff; background:#000000; padding:3px 8px;")
        analysis_layout.addWidget(self._lbl_class)

        # Perugini.
        self._perugini_combo = QComboBox()
        for score, desc in PERUGINI_SCORES.items():
            self._perugini_combo.addItem(f"{score} — {desc}", score)
        try:
            if image is not None:
                roi_h = ROICircle(cy=0.4*image.shape[0], cx=0.4*image.shape[1], radius=12.0)
                roi_m = ROICircle(cy=0.6*image.shape[0], cx=0.6*image.shape[1], radius=12.0)
                result = compute_hmr(image, roi_h, roi_m)
                suggested = self._suggest_perugini_from_hmr(result.hmr)
                self._set_perugini_score(suggested)
        except Exception:
            self._perugini_combo.setCurrentIndex(0)
        analysis_layout.addWidget(self._perugini_combo)

        self._perugini_confirm_chk = QCheckBox("Perugini confirmado manualmente")
        self._perugini_confirm_chk.setStyleSheet("color: #000000; font-size: 11px;")
        self._perugini_confirm_chk.setChecked(False)
        analysis_layout.addWidget(self._perugini_confirm_chk)

        # Selector de tiempo para washout.
        time_row = QHBoxLayout()
        lbl_time = QLabel("Tiempo:")
        lbl_time.setStyleSheet("color:#000000;")
        time_row.addWidget(lbl_time)
        self._time_combo = QComboBox()
        self._time_combo.addItems(["1h", "3h"])
        self._time_combo.setEnabled(False)
        self._time_combo.setStyleSheet("QComboBox { background:#ffffff; color:#000000; border:1px solid #7a7a7a; padding:4px; border-radius:4px; } QComboBox QAbstractItemView { background:#ffffff; color:#000000; selection-background-color:#dbeafe; }")
        time_row.addWidget(self._time_combo)

        lbl_view_role = QLabel("Vista:")
        lbl_view_role.setStyleSheet("color:#000000;")
        time_row.addWidget(lbl_view_role)
        self._view_role_combo = QComboBox()
        self._view_role_combo.addItem("AP (HMR)", "ap")
        self._view_role_combo.addItem("OAI (washout opcional)", "oai")
        self._view_role_combo.setStyleSheet("QComboBox { background:#ffffff; color:#000000; border:1px solid #7a7a7a; padding:4px; border-radius:4px; } QComboBox QAbstractItemView { background:#ffffff; color:#000000; selection-background-color:#dbeafe; }")
        self._view_role_combo.currentIndexChanged.connect(self._on_view_role_changed)
        time_row.addWidget(self._view_role_combo)
        analysis_layout.addLayout(time_row)

        # Selector de filtro visual (solo para posicionamiento de ROIs).
        from core.amyloid_planar import VISUAL_FILTERS
        filter_row = QHBoxLayout()
        lbl_filt = QLabel("Filtro visual:")
        lbl_filt.setStyleSheet("color:#000000;")
        filter_row.addWidget(lbl_filt)
        self._filter_combo = QComboBox()
        for key, (name, _) in VISUAL_FILTERS.items():
            self._filter_combo.addItem(name, key)
        self._filter_combo.addItem("Dinámico temprano (localización)", "early_dynamic")
        self._filter_combo.setStyleSheet("QComboBox { background:#ffffff; color:#000000; border:1px solid #7a7a7a; padding:4px; border-radius:4px; } QComboBox QAbstractItemView { background:#ffffff; color:#000000; selection-background-color:#dbeafe; }")
        self._filter_combo.currentIndexChanged.connect(self._on_visual_filter_changed)
        filter_row.addWidget(self._filter_combo)
        analysis_layout.addLayout(filter_row)

        # Selector de imagen base para visualización ROI.
        result_row = QHBoxLayout()
        lbl_result = QLabel("Ver imagen:")
        lbl_result.setStyleSheet("color:#000000;")
        result_row.addWidget(lbl_result)
        self._result_view_combo = QComboBox()
        self._result_view_combo.addItem("Original", "original")
        self._result_view_combo.addItem("Corregida (filtros)", "corrected")
        self._result_view_combo.addItem("Diferencia (Original - Corregida)", "difference")
        self._result_view_combo.setStyleSheet("QComboBox { background:#ffffff; color:#000000; border:1px solid #7a7a7a; padding:4px; border-radius:4px; } QComboBox QAbstractItemView { background:#ffffff; color:#000000; selection-background-color:#dbeafe; }")
        self._result_view_combo.currentIndexChanged.connect(self._on_result_view_mode_changed)
        result_row.addWidget(self._result_view_combo)
        analysis_layout.addLayout(result_row)

        # Q_bone: modo automático/manual.
        qbone_row = QHBoxLayout()
        lbl_qbone_mode = QLabel("Q_bone:")
        lbl_qbone_mode.setStyleSheet("color:#000000;")
        qbone_row.addWidget(lbl_qbone_mode)
        self._qbone_mode_combo = QComboBox()
        self._qbone_mode_combo.addItem("Automático", "auto")
        self._qbone_mode_combo.addItem("Manual (ROIs esternón/costilla)", "manual")
        self._qbone_mode_combo.setStyleSheet("QComboBox { background:#ffffff; color:#000000; border:1px solid #7a7a7a; padding:4px; border-radius:4px; } QComboBox QAbstractItemView { background:#ffffff; color:#000000; selection-background-color:#dbeafe; }")
        self._qbone_mode_combo.currentIndexChanged.connect(self._on_qbone_mode_changed)
        qbone_row.addWidget(self._qbone_mode_combo)
        analysis_layout.addLayout(qbone_row)

        self._qbone_hint_lbl = QLabel("Q_bone automático: esternón/costilla se estiman desde ROI cardíaco.")
        self._qbone_hint_lbl.setStyleSheet("font-size:10px; color:#000000; padding:2px;")
        self._qbone_hint_lbl.setWordWrap(True)
        analysis_layout.addWidget(self._qbone_hint_lbl)

        # EXCLUDE_BONE experimental (corrección costal + scatter planar opcional)
        self._exclude_bone_chk = QCheckBox("Filtro costal (EXCLUDE_BONE)")
        self._exclude_bone_chk.setStyleSheet("color:#000000; font-size:11px; font-weight:600;")
        self._exclude_bone_chk.setToolTip("Resta fondo óseo costal con ROI costilla derecha y ROI espejo izquierda")
        self._exclude_bone_chk.toggled.connect(self._on_exclude_bone_changed)
        analysis_layout.addWidget(self._exclude_bone_chk)

        eb_row = QHBoxLayout()
        lbl_eb_method = QLabel("Método costal:")
        lbl_eb_method.setStyleSheet("color:#000000; font-size:11px;")
        eb_row.addWidget(lbl_eb_method)
        self._exclude_bone_method_combo = QComboBox()
        self._exclude_bone_method_combo.addItem("Resta media costal", "mean_subtract")
        self._exclude_bone_method_combo.addItem("Escala por hotspot", "scaled_hotspot")
        self._exclude_bone_method_combo.currentIndexChanged.connect(self._on_exclude_bone_changed)
        eb_row.addWidget(self._exclude_bone_method_combo, 1)
        analysis_layout.addLayout(eb_row)

        self._exclude_sternum_chk = QCheckBox("Filtro esternón (2 fondos)")
        self._exclude_sternum_chk.setStyleSheet("color:#000000; font-size:11px; font-weight:600;")
        self._exclude_sternum_chk.setToolTip("Corrige actividad esternal usando ROI esternón menos promedio de Fondo est. 1/2")
        self._exclude_sternum_chk.toggled.connect(self._on_exclude_bone_changed)
        analysis_layout.addWidget(self._exclude_sternum_chk)

        eb_row2 = QHBoxLayout()
        lbl_asym_rib = QLabel("Asimetría máx costal:")
        lbl_asym_rib.setStyleSheet("color:#000000; font-size:11px;")
        eb_row2.addWidget(lbl_asym_rib)
        self._exclude_bone_asym_spin = QDoubleSpinBox()
        self._exclude_bone_asym_spin.setRange(0.05, 1.50)
        self._exclude_bone_asym_spin.setSingleStep(0.05)
        self._exclude_bone_asym_spin.setDecimals(2)
        self._exclude_bone_asym_spin.setValue(0.35)
        self._exclude_bone_asym_spin.valueChanged.connect(self._on_exclude_bone_changed)
        eb_row2.addWidget(self._exclude_bone_asym_spin)
        lbl_sc_k = QLabel("Scatter k:")
        lbl_sc_k.setStyleSheet("color:#000000; font-size:11px;")
        eb_row2.addWidget(lbl_sc_k)
        self._scatter_k_spin = QDoubleSpinBox()
        self._scatter_k_spin.setRange(0.0, 2.0)
        self._scatter_k_spin.setSingleStep(0.05)
        self._scatter_k_spin.setDecimals(2)
        self._scatter_k_spin.setValue(1.00)
        self._scatter_k_spin.valueChanged.connect(self._on_exclude_bone_changed)
        eb_row2.addWidget(self._scatter_k_spin)
        analysis_layout.addLayout(eb_row2)

        eb_row3 = QHBoxLayout()
        lbl_asym_st = QLabel("Asimetría máx esternón:")
        lbl_asym_st.setStyleSheet("color:#000000; font-size:11px;")
        eb_row3.addWidget(lbl_asym_st)
        self._exclude_sternum_asym_spin = QDoubleSpinBox()
        self._exclude_sternum_asym_spin.setRange(0.05, 1.50)
        self._exclude_sternum_asym_spin.setSingleStep(0.05)
        self._exclude_sternum_asym_spin.setDecimals(2)
        self._exclude_sternum_asym_spin.setValue(0.35)
        self._exclude_sternum_asym_spin.valueChanged.connect(self._on_exclude_bone_changed)
        eb_row3.addWidget(self._exclude_sternum_asym_spin)
        analysis_layout.addLayout(eb_row3)

        self._scatter_planar_chk = QCheckBox("Usar SCATTER planar (si existe archivo SC)")
        self._scatter_planar_chk.setStyleSheet("color:#000000; font-size:11px;")
        self._scatter_planar_chk.toggled.connect(self._on_exclude_bone_changed)
        analysis_layout.addWidget(self._scatter_planar_chk)

        self._show_mirror_roi_chk = QCheckBox("Mostrar ROI espejo costilla (preview filtro costal)")
        self._show_mirror_roi_chk.setStyleSheet("color:#000000; font-size:11px;")
        self._show_mirror_roi_chk.setChecked(True)
        self._show_mirror_roi_chk.toggled.connect(self._on_exclude_bone_changed)
        analysis_layout.addWidget(self._show_mirror_roi_chk)

        # Botón Aplicar: renderiza ROIs + HMR y asigna al cuadrante AP+ROIs.
        btn_apply = QPushButton("Aplicar ROIs al cuadrante")
        btn_apply.setStyleSheet("font-size: 13px; font-weight: bold; padding: 8px; background: #2563eb; color: white; border-radius: 6px;")
        btn_apply.clicked.connect(self._apply_rois_to_quadrant)
        analysis_layout.addWidget(btn_apply)

        # Label de estado de washout.
        self._lbl_filter_summary = QLabel("")
        self._lbl_filter_summary.setStyleSheet("font-size:10px; color:#000000; background:#f3f4f6; padding:6px 10px; border:1px solid #d1d5db; border-radius:999px;")
        self._lbl_filter_summary.setWordWrap(True)
        analysis_layout.addWidget(self._lbl_filter_summary)

        self._lbl_washout_status = QLabel("")
        self._lbl_washout_status.setStyleSheet("font-size:10px; color:#000000; background:#eef2ff; padding:6px 10px; border:1px solid #c7d2fe; border-radius:999px;")
        self._lbl_washout_status.setWordWrap(True)
        analysis_layout.addWidget(self._lbl_washout_status)

        self._stack.addWidget(page_analysis)

        # ── Botones inferiores ─────────────────────────────────────
        btns = QHBoxLayout()
        btn_reset = QPushButton("Reset ROIs")
        btn_reset.clicked.connect(self._reset_rois)
        btns.addWidget(btn_reset)
        btns.addStretch(1)
        btn_report = QPushButton("Generar Informe")
        btn_report.clicked.connect(self._generate_report)
        btns.addWidget(btn_report)

        btns.addWidget(QLabel("Salida:"))
        self._report_output_combo = QComboBox()
        self._report_output_combo.addItem("PDF + HTML", "both")
        self._report_output_combo.addItem("Solo PDF", "pdf")
        self._report_output_combo.addItem("Solo HTML", "html")
        self._report_output_combo.currentIndexChanged.connect(lambda _=0: self._persist_user_state())
        btns.addWidget(self._report_output_combo)

        self._report_output_dir_btn = QPushButton("Carpeta informe")
        self._report_output_dir_btn.setToolTip("Elegir carpeta de salida para PDF/HTML")
        self._report_output_dir_btn.clicked.connect(self._select_report_output_dir)
        btns.addWidget(self._report_output_dir_btn)

        btn_export = QPushButton("Exportar PNG/JPG")
        btn_export.clicked.connect(self._export_png_jpg)
        btns.addWidget(btn_export)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_close)
        root.addLayout(btns)

        # ── Inicializar ────────────────────────────────────────────
        if image is not None:
            self._time_images["1h"]["ap"] = {"image": np.asarray(image, dtype=np.float64), "label": "AP", "path": ""}
            self._active_time = "1h"
        self._ensure_aux_rois()
        self._restore_user_state()
        self._load_spect_ct_bridge_state()
        self._rebuild_layout()
        self._update_washout_preview()
        if image is not None:
            self._update_hmr(0, 0, 0, 0)

    def _settings_group_key(self) -> str:
        patient = str(self._metadata.get("patient") or "N_D")
        date = str(self._metadata.get("date") or "N_D")
        series = str(self._metadata.get("series") or "N_D")
        key = f"{patient}|{date}|{series}"
        return "".join(ch if ch.isalnum() or ch in "._-|" else "_" for ch in key)

    def _settings(self) -> QSettings:
        return QSettings("GAMMASYS", "SINCRO_AMYLO")

    def _load_spect_ct_bridge_state(self):
        self._linked_spect_ct = None
        try:
            bridge = QSettings("GAMMASYS", "SINCRO_AMYLO_BRIDGE")
            raw = bridge.value("last_spect_ct_session_json", "")
            if not raw:
                return
            payload = json.loads(str(raw))
            profile = dict(payload.get("profile") or {})
            if str(payload.get("workflow_tag") or "") != "perf_spect_ct":
                return
            bridge_pid = str(profile.get("patient_id") or "").strip().lower()
            bridge_pname = str(profile.get("patient_name") or "").strip().lower()
            bridge_sdate = str(profile.get("study_date") or "").strip()
            local_pid = str(self._metadata.get("patient_id") or "").strip().lower()
            local_pname = str(self._metadata.get("patient") or "").strip().lower()
            local_date = str(self._metadata.get("date") or "").strip()
            # Si el módulo se abrió sin planar cargado (sin identidad local),
            # permitir reutilizar la última sesión SPECT/CT.
            if (not local_pid and not local_pname) and (not local_date or local_date == "N/D"):
                self._linked_spect_ct = payload
                return
            patient_ok = (bridge_pid and local_pid and bridge_pid == local_pid) or (bridge_pname and local_pname and bridge_pname == local_pname)
            date_ok = (not bridge_sdate) or (not local_date) or (bridge_sdate == local_date)
            if not patient_ok or not date_ok:
                return
            self._linked_spect_ct = payload
        except Exception:
            self._linked_spect_ct = None

    @staticmethod
    def _suggest_perugini_from_hmr(hmr: float) -> int:
        """Sugerencia heurística de Perugini basada en HMR (no reemplaza lectura visual)."""
        if hmr >= 1.5:
            return 3
        if hmr >= 1.2:
            return 2
        if hmr >= 1.0:
            return 1
        return 0

    def _set_perugini_score(self, score: int):
        idx = self._perugini_combo.findData(int(score))
        if idx >= 0:
            self._perugini_combo.setCurrentIndex(idx)

    def _ensure_aux_rois(self):
        if len(self._roi_widget._rois) >= 6:
            return
        if self._original_image is None:
            h, w = 64, 64
        else:
            h, w = self._original_image.shape
        base_r = self._roi_widget._rois[0]["radius"] if self._roi_widget._rois else 12.0
        if len(self._roi_widget._rois) < 3:
            self._roi_widget._rois.append(
                {"cy": 0.25 * h, "cx": 0.50 * w, "radius": base_r * 0.6, "color": "#fbbf24", "name": "Esternón"}
            )
        if len(self._roi_widget._rois) < 4:
            self._roi_widget._rois.append(
                {"cy": 0.60 * h, "cx": 0.20 * w, "radius": base_r * 0.5, "color": "#a78bfa", "name": "Costilla"}
            )
        if len(self._roi_widget._rois) < 5:
            self._roi_widget._rois.append(
                {"cy": 0.25 * h, "cx": 0.42 * w, "radius": base_r * 0.45, "color": "#34d399", "name": "Fondo est. 1"}
            )
        if len(self._roi_widget._rois) < 6:
            self._roi_widget._rois.append(
                {"cy": 0.25 * h, "cx": 0.58 * w, "radius": base_r * 0.45, "color": "#22d3ee", "name": "Fondo est. 2"}
            )

    def _serialize_rois(self, rois: list[dict] | None) -> list[dict] | None:
        if not rois:
            return None
        out: list[dict] = []
        for roi in rois:
            out.append({
                "cy": float(roi.get("cy", 0.0)),
                "cx": float(roi.get("cx", 0.0)),
                "radius": float(roi.get("radius", 0.0)),
                "color": str(roi.get("color", "#ffffff")),
                "name": str(roi.get("name", "ROI")),
            })
        return out

    @staticmethod
    def _al_status_text_from_code(code: str) -> str:
        mapping = {
            "excluded": "EXCLUIDA",
            "not_excluded": "NO EXCLUIDA",
            "pending": "PENDIENTE / NO INFORMADO",
        }
        return mapping.get(str(code or "").strip(), "PENDIENTE / NO INFORMADO")

    @staticmethod
    def _flc_text_from_code(code: str) -> str:
        mapping = {
            "normal": "Normales",
            "abnormal": "Alteradas",
            "unknown": "No informado",
        }
        return mapping.get(str(code or "").strip(), "No informado")

    @staticmethod
    def _immunofix_text_from_code(code: str) -> str:
        mapping = {
            "negative": "Negativa",
            "positive": "Positiva",
            "unknown": "No informada",
        }
        return mapping.get(str(code or "").strip(), "No informada")

    def _persist_user_state(self):
        try:
            self._ensure_aux_rois()
            key = self._settings_group_key()
            settings = self._settings()
            settings.beginGroup(f"amyloid/{key}")
            payload = {
                "roi_state": {
                    "1h": self._serialize_rois(self._roi_state.get("1h")),
                    "3h": self._serialize_rois(self._roi_state.get("3h")),
                },
                "roi_state_oai": {
                    "1h": self._serialize_rois(self._roi_state_oai.get("1h")),
                    "3h": self._serialize_rois(self._roi_state_oai.get("3h")),
                },
                "qbone_mode_by_time": {
                    "1h": str(self._qbone_mode_by_time.get("1h", "auto")),
                    "3h": str(self._qbone_mode_by_time.get("3h", "auto")),
                },
                "perugini_by_time": {
                    "1h": int(self._perugini_by_time.get("1h", 0)),
                    "3h": int(self._perugini_by_time.get("3h", 0)),
                },
                "perugini_confirmed_by_time": {
                    "1h": bool(self._perugini_confirmed_by_time.get("1h", False)),
                    "3h": bool(self._perugini_confirmed_by_time.get("3h", False)),
                },
                "quadrant_state": self._quadrant_state,
                "layout_n": int(self._current_layout_n),
                "active_time": str(self._active_time or "1h"),
                "time_hours_by_label": {
                    "1h": float(self._time_hours_by_label.get("1h", 1.0)),
                    "3h": float(self._time_hours_by_label.get("3h", 3.0)),
                },
                "report_template": str(self._report_template_combo.currentData() or "auto"),
                "al_status": str(self._al_status_combo.currentData() or "pending"),
                "free_light_chain": str(self._free_light_chain_combo.currentData() or "unknown"),
                "immunofixation": str(self._immunofix_combo.currentData() or "unknown"),
                "oai_washout_data": {
                    "1h": dict(self._oai_washout_data.get("1h") or {}),
                    "3h": dict(self._oai_washout_data.get("3h") or {}),
                },
                "active_view_role": str(self._active_view_role or "ap"),
                "report_output_mode": str(self._report_output_combo.currentData() or "both"),
                "report_output_dir": str(getattr(self, "_report_output_dir", "") or ""),
                "exclude_bone_enabled": bool(self._exclude_bone_chk.isChecked()),
                "exclude_bone_method": str(self._exclude_bone_method_combo.currentData() or "mean_subtract"),
                "exclude_bone_asym_thresh": float(self._exclude_bone_asym_spin.value()),
                "exclude_sternum_enabled": bool(self._exclude_sternum_chk.isChecked()),
                "exclude_sternum_asym_thresh": float(self._exclude_sternum_asym_spin.value()),
                "use_scatter_planar": bool(self._scatter_planar_chk.isChecked()),
                "scatter_planar_k": float(self._scatter_k_spin.value()),
                "show_mirror_roi_preview": bool(getattr(self, "_show_mirror_roi_chk", None) and self._show_mirror_roi_chk.isChecked()),
                "result_view_mode": str(self._result_view_combo.currentData() or "original"),
            }
            settings.setValue("state_json", json.dumps(payload, ensure_ascii=False))
            settings.endGroup()
            settings.sync()
        except Exception:
            pass

    def _restore_user_state(self):
        try:
            key = self._settings_group_key()
            settings = self._settings()
            settings.beginGroup(f"amyloid/{key}")
            raw = settings.value("state_json", "")
            settings.endGroup()
            if not raw:
                return
            payload = json.loads(str(raw))
            roi_state = payload.get("roi_state", {}) or {}
            for time_label in ("1h", "3h"):
                rois = roi_state.get(time_label)
                if rois:
                    self._roi_state[time_label] = self._serialize_rois(rois)
            roi_state_oai = payload.get("roi_state_oai", {}) or {}
            for time_label in ("1h", "3h"):
                rois_oai = roi_state_oai.get(time_label)
                if rois_oai:
                    self._roi_state_oai[time_label] = self._serialize_rois(rois_oai)
            qmode = payload.get("qbone_mode_by_time", {}) or {}
            self._qbone_mode_by_time["1h"] = str(qmode.get("1h", "auto"))
            self._qbone_mode_by_time["3h"] = str(qmode.get("3h", "auto"))
            pvals = payload.get("perugini_by_time", {}) or {}
            self._perugini_by_time["1h"] = int(pvals.get("1h", self._perugini_by_time.get("1h", 0)))
            self._perugini_by_time["3h"] = int(pvals.get("3h", self._perugini_by_time.get("3h", 0)))
            pconf = payload.get("perugini_confirmed_by_time", {}) or {}
            self._perugini_confirmed_by_time["1h"] = bool(pconf.get("1h", False))
            self._perugini_confirmed_by_time["3h"] = bool(pconf.get("3h", False))
            self._quadrant_state = payload.get("quadrant_state", {}) or {}
            th = payload.get("time_hours_by_label", {}) or {}
            self._time_hours_by_label["1h"] = float(th.get("1h", 1.0))
            self._time_hours_by_label["3h"] = float(th.get("3h", 3.0))
            saved_layout = int(payload.get("layout_n", self._current_layout_n))
            idx = self._layout_combo.findData(saved_layout)
            if idx >= 0:
                self._layout_combo.blockSignals(True)
                self._layout_combo.setCurrentIndex(idx)
                self._layout_combo.blockSignals(False)
                self._current_layout_n = saved_layout
            self._active_time = str(payload.get("active_time", self._active_time or "1h"))

            saved_tpl = str(payload.get("report_template", "auto") or "auto")
            idx_tpl = self._report_template_combo.findData(saved_tpl)
            if idx_tpl >= 0:
                self._report_template_combo.blockSignals(True)
                self._report_template_combo.setCurrentIndex(idx_tpl)
                self._report_template_combo.blockSignals(False)

            saved_al = str(payload.get("al_status", "pending") or "pending")
            idx_al = self._al_status_combo.findData(saved_al)
            if idx_al >= 0:
                self._al_status_combo.blockSignals(True)
                self._al_status_combo.setCurrentIndex(idx_al)
                self._al_status_combo.blockSignals(False)

            saved_flc = str(payload.get("free_light_chain", "unknown") or "unknown")
            idx_flc = self._free_light_chain_combo.findData(saved_flc)
            if idx_flc >= 0:
                self._free_light_chain_combo.blockSignals(True)
                self._free_light_chain_combo.setCurrentIndex(idx_flc)
                self._free_light_chain_combo.blockSignals(False)

            saved_if = str(payload.get("immunofixation", "unknown") or "unknown")
            idx_if = self._immunofix_combo.findData(saved_if)
            if idx_if >= 0:
                self._immunofix_combo.blockSignals(True)
                self._immunofix_combo.setCurrentIndex(idx_if)
                self._immunofix_combo.blockSignals(False)

            oai_data = payload.get("oai_washout_data", {}) or {}
            self._oai_washout_data["1h"] = dict(oai_data.get("1h") or {})
            self._oai_washout_data["3h"] = dict(oai_data.get("3h") or {})

            saved_role = str(payload.get("active_view_role", "ap") or "ap")
            idx_role = self._view_role_combo.findData(saved_role)
            if idx_role >= 0:
                self._view_role_combo.blockSignals(True)
                self._view_role_combo.setCurrentIndex(idx_role)
                self._view_role_combo.blockSignals(False)
                self._active_view_role = saved_role

            saved_out_mode = str(payload.get("report_output_mode", "both") or "both")
            idx_out = self._report_output_combo.findData(saved_out_mode)
            if idx_out >= 0:
                self._report_output_combo.blockSignals(True)
                self._report_output_combo.setCurrentIndex(idx_out)
                self._report_output_combo.blockSignals(False)
            self._report_output_dir = str(payload.get("report_output_dir", "") or "")

            self._exclude_bone_chk.setChecked(bool(payload.get("exclude_bone_enabled", False)))
            idx_eb = self._exclude_bone_method_combo.findData(str(payload.get("exclude_bone_method", "mean_subtract")))
            if idx_eb >= 0:
                self._exclude_bone_method_combo.setCurrentIndex(idx_eb)
            self._exclude_bone_asym_spin.setValue(float(payload.get("exclude_bone_asym_thresh", 0.35) or 0.35))
            self._exclude_sternum_chk.setChecked(bool(payload.get("exclude_sternum_enabled", False)))
            self._exclude_sternum_asym_spin.setValue(float(payload.get("exclude_sternum_asym_thresh", 0.35) or 0.35))
            self._scatter_planar_chk.setChecked(bool(payload.get("use_scatter_planar", False)))
            self._scatter_k_spin.setValue(float(payload.get("scatter_planar_k", 1.0) or 1.0))
            self._show_mirror_roi_chk.setChecked(bool(payload.get("show_mirror_roi_preview", True)))
            saved_view_mode = str(payload.get("result_view_mode", "original") or "original")
            idx_view_mode = self._result_view_combo.findData(saved_view_mode)
            if idx_view_mode >= 0:
                self._result_view_combo.blockSignals(True)
                self._result_view_combo.setCurrentIndex(idx_view_mode)
                self._result_view_combo.blockSignals(False)
                self._result_view_mode = saved_view_mode

            self._sync_qbone_mode_ui()
        except Exception:
            pass

    def closeEvent(self, event):
        self._persist_user_state()
        super().closeEvent(event)

    # ── Modo ────────────────────────────────────────────────────────

    def _toggle_mode(self):
        """Alterna entre visor de cuadrantes y análisis ROI."""
        if self._current_mode == "visor":
            self._current_mode = "analisis"
            self._stack.setCurrentIndex(1)
            self._btn_mode.setText("← Visor cuadrantes")
        else:
            if self._active_time in self._roi_state and self._original_image is not None:
                self._roi_state[self._active_time] = [dict(roi) for roi in self._roi_widget._rois]
            self._current_mode = "visor"
            self._stack.setCurrentIndex(0)
            self._btn_mode.setText("Análisis ROI →")

    # ── Layout ──────────────────────────────────────────────────────

    def _on_layout_changed(self, idx: int):
        n = self._layout_combo.currentData()
        if n and n != self._current_layout_n:
            self._current_layout_n = n
            self._page_offset = 0  # reiniciar paginación al cambiar layout
            self._rebuild_layout(force_layout=n)

    def _rebuild_layout(self, force_layout: int | None = None):
        """Reconstruye la presentación desde las fuentes canónicas 1 h/3 h."""
        n = force_layout or self._current_layout_n
        self._current_layout_n = n
        self._layout_12q3x4_lat_hidden = False

        def _slots(time_label: str) -> tuple[list[np.ndarray | None], list[str]]:
            source = self._time_images[time_label]
            ap_entry = source["ap"]
            ap = ap_entry["image"] if ap_entry else None
            processed = self._processed_images[time_label]
            corr = processed.get("corr", processed.get("clean", ap))
            corr_meta = processed.get("corr_meta", {}) if isinstance(processed.get("corr_meta"), dict) else {}
            corr_used = bool(
                corr_meta.get("rib_filter_used")
                or corr_meta.get("sternum_filter_used")
                or corr_meta.get("scatter_used")
            )
            images = [
                processed.get("roi", ap), corr,
                source["oai"]["image"] if source["oai"] else None,
                source["lat"]["image"] if source["lat"] else None,
            ]
            ap_lbl = self._build_label_for_role(source.get("ap"), "ap", time_label)
            oai_lbl = self._build_label_for_role(source.get("oai"), "oai", time_label)
            lat_lbl = self._build_label_for_role(source.get("lat"), "lat", time_label)
            labels = [
                (ap_lbl + " + ROIs") if "roi" in processed else (ap_lbl + " cuantificación"),
                (ap_lbl + " corregido (filtros)") if corr_used else (ap_lbl + " limpio"),
                oai_lbl,
                lat_lbl,
            ]
            return images, labels

        imgs_1h, labels_1h = _slots("1h")
        imgs_3h, labels_3h = _slots("3h")
        imgs = imgs_1h + imgs_3h
        labels = labels_1h + labels_3h
        layout_labels = labels

        if n == 3:
            layout = layout_3q(
                ap_roi=imgs[0] if len(imgs) > 0 else None,
                ap_clean=imgs[1] if len(imgs) > 1 else None,
                oai=imgs[2] if len(imgs) > 2 else None,
                ap_label="AP cuantificación (1h)",
                oai_label=labels[2] if len(labels) > 2 else "OAI 45°",
            )
            layout.quadrants[0].label = labels[0]
            layout.quadrants[1].label = labels[1]
            layout.quadrants[2].label = labels[2] if len(labels) > 2 else "OAI 45°"
        elif n == 4:
            layout = layout_4q(
                ap_roi=imgs[0] if len(imgs) > 0 else None,
                ap_clean=imgs[1] if len(imgs) > 1 else None,
                oai=imgs[2] if len(imgs) > 2 else None,
                lat=imgs[3] if len(imgs) > 3 else None,
                ap_label="AP cuantificación (1h)",
                oai_label=labels[2] if len(labels) > 2 else "OAI 45°",
                lat_label=labels[3] if len(labels) > 3 else "LAT. IZQ.",
            )
            layout.quadrants[0].label = labels[0]
            layout.quadrants[1].label = labels[1]
        elif n == 6:
            layout = layout_6q(
                images_1h=imgs_1h[:3],
                images_3h=imgs_3h[:3],
                labels=["AP + ROIs", "AP limpio", "OAI"],
            )
        elif n == 8:
            layout = layout_8q(
                images_1h=imgs_1h,
                images_3h=imgs_3h,
                labels=["AP + ROIs", "AP limpio", "OAI", "LAT. IZQ."],
            )
        elif n == 9:
            layout = layout_9q(images=imgs, labels=labels[:3] if len(labels) >= 3 else None)
        elif n == 12:
            layout = layout_12q(images=imgs, labels=labels[:3] if len(labels) >= 3 else None)
        elif n == 1234:
            has_lat_any = any(self._time_images.get(t, {}).get("lat") is not None for t in ("1h", "3h"))
            if has_lat_any:
                layout = layout_12q_3x4(images=imgs, labels=labels[:4] if len(labels) >= 4 else None)
            else:
                # Si no hay LAT en ningún tiempo, compactar visualmente a 3 columnas.
                imgs_compact = [
                    imgs_1h[0], imgs_1h[1], imgs_1h[2],
                    imgs_3h[0], imgs_3h[1], imgs_3h[2],
                    None, None, None,
                ]
                layout = layout_9q(images=imgs_compact, labels=labels_1h[:3] if len(labels_1h) >= 3 else None)
                layout_labels = [
                    labels_1h[0], labels_1h[1], labels_1h[2],
                    labels_3h[0], labels_3h[1], labels_3h[2],
                    "", "", "",
                ]
                self._layout_12q3x4_lat_hidden = True
        elif n == 16:
            layout = layout_16q(images=imgs, labels=labels[:4] if len(labels) >= 4 else None)
        else:
            return
        for idx, label in enumerate(layout_labels):
            if idx < len(layout.quadrants):
                layout.quadrants[idx].label = label
        self._quadrant_viewer.set_layout(layout)
        self._restore_quadrant_states()
        self._quadrant_viewer._rebuild_pixmaps()
        self._quadrant_viewer.update()
        self._on_quadrant_selected(0)
        # Actualizar info de paginación.
        self._lbl_pagination.setText("Página 1/1")

    def get_layout_images(self) -> list[np.ndarray]:
        """Devuelve las imágenes del layout actual como arrays RGB."""
        layout = self._quadrant_viewer._layout
        if layout is None:
            return []
        images = []
        for q in layout.quadrants:
            if q.image is None:
                continue
            img = np.asarray(q.image, dtype=np.float64)
            if img.ndim == 3:
                # Ya es RGB.
                images.append(img)
            else:
                # Convertir 2D a RGB.
                h, w = img.shape
                norm = img / max(float(img.max()), 1e-8) if img.size else img
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                rgb[..., 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
                rgb[..., 1] = rgb[..., 0]
                rgb[..., 2] = rgb[..., 0]
                images.append(rgb)
        return images

    def get_layout_composite_image(self) -> np.ndarray | None:
        """Renderiza todo el layout como una única imagen PIL (grilla rows×cols)."""
        from PIL import Image as PILImage, ImageDraw, ImageFont

        layout = self._quadrant_viewer._layout
        if layout is None or not layout.quadrants:
            return None

        rows, cols = layout.rows, layout.cols

        # Recopilar imágenes y labels.
        items = []
        for q in layout.quadrants:
            if q.image is None:
                items.append(None)
                continue
            img = np.asarray(q.image, dtype=np.float64)
            if img.ndim == 2:
                h, w = img.shape
                norm = img / max(float(img.max()), 1e-8) if img.size else img
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                rgb[..., 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
                rgb[..., 1] = rgb[..., 0]
                rgb[..., 2] = rgb[..., 0]
                items.append((rgb, q.label))
            else:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                items.append((img, q.label))

        # Determinar tamaño de celda (máximo de todas las imágenes).
        max_w, max_h = 0, 0
        for item in items:
            if item is not None:
                arr, _ = item
                max_w = max(max_w, arr.shape[1])
                max_h = max(max_h, arr.shape[0])

        if max_w == 0 or max_h == 0:
            return None

        cell_w = max_w
        cell_h = max_h
        label_h = 24
        border = 2
        pad = 4

        total_w = cols * cell_w + (cols + 1) * border + 2 * pad
        total_h = rows * (cell_h + label_h) + (rows + 1) * border + 2 * pad

        composite = PILImage.new("RGB", (total_w, total_h), (15, 23, 42))
        draw = ImageDraw.Draw(composite)

        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

        for idx, item in enumerate(items):
            row = idx // cols
            col = idx % cols

            x0 = pad + col * (cell_w + border) + border
            y0 = pad + row * (cell_h + label_h + border) + border

            if item is None:
                draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1],
                               outline=(71, 85, 105), width=1)
                continue

            arr, label = item
            pil_img = PILImage.fromarray(arr)

            iw, ih = pil_img.size
            scale = min(cell_w / max(1, iw), cell_h / max(1, ih))
            new_w, new_h = int(iw * scale), int(ih * scale)
            pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

            ox = x0 + (cell_w - new_w) // 2
            oy = y0 + (cell_h - new_h) // 2
            composite.paste(pil_img, (ox, oy))

            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1],
                           outline=(71, 85, 105), width=1)

            lbl = label if len(label) <= 40 else label[:37] + "..."
            bbox = draw.textbbox((0, 0), lbl, font=font)
            tw = bbox[2] - bbox[0]
            lx = x0 + (cell_w - tw) // 2
            ly = y0 + cell_h + 2
            draw.text((lx, ly), lbl, fill=(148, 163, 184), font=font)

        return np.asarray(composite)

    # ── Cargar imágenes ─────────────────────────────────────────────

    @staticmethod
    def _dicom_frame_durations_s(ds, n_frames: int) -> np.ndarray:
        """Extrae duración de cada frame dinámico en segundos."""
        vector = getattr(ds, "FrameTimeVector", None)
        if vector is not None:
            values = np.asarray(vector, dtype=np.float64).reshape(-1) / 1000.0
            if values.size == n_frames and np.all(values > 0):
                return values
        frame_time = getattr(ds, "FrameTime", None)
        if frame_time is not None and float(frame_time) > 0:
            return np.full(n_frames, float(frame_time) / 1000.0)
        actual = getattr(ds, "ActualFrameDuration", None)
        if actual is not None and float(actual) > 0:
            return np.full(n_frames, float(actual) / 1000.0)
        raise ValueError("El DICOM dinámico no informa FrameTime/FrameTimeVector")

    @staticmethod
    def _dicom_static_duration_s(ds) -> float | None:
        """Extrae duración de la adquisición estática si está disponible."""
        actual = getattr(ds, "ActualFrameDuration", None)
        if actual is not None and float(actual) > 0:
            return float(actual) / 1000.0
        frame_time = getattr(ds, "FrameTime", None)
        if frame_time is not None and float(frame_time) > 0:
            return float(frame_time) / 1000.0
        return None

    @staticmethod
    def _parse_dicom_tm(tm_val) -> time | None:
        """Parsea TM DICOM (HHMMSS.frac) a objeto time."""
        if tm_val is None:
            return None
        s = str(tm_val).strip()
        if not s:
            return None
        s = s.split(".")[0]
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) < 2:
            return None
        hh = int(digits[0:2])
        mm = int(digits[2:4]) if len(digits) >= 4 else 0
        ss = int(digits[4:6]) if len(digits) >= 6 else 0
        if hh > 23 or mm > 59 or ss > 59:
            return None
        return time(hour=hh, minute=mm, second=ss)

    @staticmethod
    def _parse_dicom_dt(dt_val) -> datetime | None:
        """Parsea DT DICOM (YYYYMMDDHHMMSS.frac) a datetime."""
        if dt_val is None:
            return None
        s = str(dt_val).strip()
        if not s:
            return None
        s = s.split("+")[0].split("-")[0].split(".")[0]
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) < 8:
            return None
        try:
            y = int(digits[0:4])
            m = int(digits[4:6])
            d = int(digits[6:8])
            hh = int(digits[8:10]) if len(digits) >= 10 else 0
            mm = int(digits[10:12]) if len(digits) >= 12 else 0
            ss = int(digits[12:14]) if len(digits) >= 14 else 0
            return datetime(y, m, d, hh, mm, ss)
        except Exception:
            return None

    def _dicom_acquisition_datetime(self, ds) -> datetime | None:
        """Intenta construir datetime de adquisición con prioridad a tags temporales robustos."""
        for dt_attr in (
            "AcquisitionDateTime",
            "SeriesDateTime",
            "StudyDateTime",
            "RadiopharmaceuticalStartDateTime",
        ):
            dt_obj = self._parse_dicom_dt(getattr(ds, dt_attr, None))
            if dt_obj is not None:
                return dt_obj

        date_candidates = [
            getattr(ds, "AcquisitionDate", None),
            getattr(ds, "SeriesDate", None),
            getattr(ds, "StudyDate", None),
            getattr(ds, "ContentDate", None),
        ]
        tm_candidates = [
            getattr(ds, "AcquisitionTime", None),
            getattr(ds, "SeriesTime", None),
            getattr(ds, "StudyTime", None),
            getattr(ds, "ContentTime", None),
        ]

        d_obj = None
        for d in date_candidates:
            ds_str = str(d or "").strip()
            digits = "".join(ch for ch in ds_str if ch.isdigit())
            if len(digits) >= 8:
                try:
                    d_obj = date(int(digits[0:4]), int(digits[4:6]), int(digits[6:8]))
                    break
                except Exception:
                    continue

        t_obj = None
        for t in tm_candidates:
            t_obj = self._parse_dicom_tm(t)
            if t_obj is not None:
                break

        if d_obj is None and t_obj is None:
            return None
        if d_obj is None:
            d_obj = datetime.today().date()
        if t_obj is None:
            t_obj = time(hour=0, minute=0, second=0)
        return datetime.combine(d_obj, t_obj)

    def _extract_planar_record(self, path: str) -> dict:
        """Carga DICOM planar y devuelve registro estándar para AP/OAI/LAT."""
        import pydicom

        ds = pydicom.dcmread(path, force=True)
        img = np.asarray(ds.pixel_array, dtype=np.float64)
        while img.ndim > 2:
            img = img[img.shape[0] // 2]
        if img.ndim != 2:
            raise ValueError(f"Imagen no planar: shape={img.shape}")

        description = " ".join(filter(None, [
            str(getattr(ds, "SeriesDescription", "") or ""),
            str(getattr(ds, "ViewPosition", "") or ""),
        ])).upper()
        dt = self._dicom_acquisition_datetime(ds)
        scatter_image = None
        scatter_path = ""
        try:
            from core.raw_projections import find_scatter_sibling
            sc_path = find_scatter_sibling(path)
            if sc_path and os.path.isfile(sc_path):
                ds_sc = pydicom.dcmread(sc_path, force=True)
                sc = np.asarray(ds_sc.pixel_array, dtype=np.float64)
                while sc.ndim > 2:
                    sc = sc[sc.shape[0] // 2]
                if sc.shape == img.shape:
                    scatter_image = sc
                    scatter_path = sc_path
        except Exception:
            scatter_image = None
            scatter_path = ""
        return {
            "image": img,
            "label": description or os.path.basename(path),
            "path": path,
            "view": self._classify_planar_view(description),
            "ds": ds,
            "duration_s": self._dicom_static_duration_s(ds),
            "acq_dt": dt,
            "scatter_image": scatter_image,
            "scatter_path": scatter_path,
        }

    def _on_exclude_bone_changed(self, _=None):
        self._exclude_bone_enabled = bool(self._exclude_bone_chk.isChecked())
        self._exclude_bone_method = str(self._exclude_bone_method_combo.currentData() or "mean_subtract")
        self._exclude_bone_asym_thresh = float(self._exclude_bone_asym_spin.value())
        self._exclude_sternum_enabled = bool(self._exclude_sternum_chk.isChecked())
        self._exclude_sternum_asym_thresh = float(self._exclude_sternum_asym_spin.value())
        self._use_scatter_planar = bool(self._scatter_planar_chk.isChecked())
        self._scatter_planar_k = float(self._scatter_k_spin.value())
        show_preview = bool(getattr(self, "_show_mirror_roi_chk", None) and self._show_mirror_roi_chk.isChecked())
        mode_manual = str(self._qbone_mode_combo.currentData() or "auto") == "manual"
        self._roi_widget.set_mirror_roi_visible(show_preview and mode_manual and self._exclude_bone_enabled)
        self._update_roi_display_image()
        self._update_filter_summary()
        self._persist_user_state()
        self._update_hmr(0, 0, 0, 0)

    def _on_result_view_mode_changed(self, _idx: int):
        self._result_view_mode = str(self._result_view_combo.currentData() or "original")
        self._update_roi_display_image()
        self._persist_user_state()

    def _update_filter_summary(self):
        if not hasattr(self, "_lbl_filter_summary"):
            return
        costal_on = bool(getattr(self, "_exclude_bone_enabled", False))
        sternum_on = bool(getattr(self, "_exclude_sternum_enabled", False))
        scatter_on = bool(getattr(self, "_use_scatter_planar", False))
        order_txt = "Orden: SCATTER → Esternón → Costal"

        if self._original_image is None:
            self._lbl_filter_summary.setText(
                f"Filtro costal: {'ON' if costal_on else 'OFF'} · "
                f"Filtro esternón: {'ON' if sternum_on else 'OFF'} · "
                f"SCATTER: {'ON' if scatter_on else 'OFF'}\n{order_txt}"
            )
            return

        try:
            roi_h = ROICircle(
                cy=self._roi_widget._rois[0]["cy"],
                cx=self._roi_widget._rois[0]["cx"],
                radius=self._roi_widget._rois[0]["radius"],
            )
            roi_m = ROICircle(
                cy=self._roi_widget._rois[1]["cy"],
                cx=self._roi_widget._rois[1]["cx"],
                radius=self._roi_widget._rois[1]["radius"],
            )

            img_raw = np.asarray(self._original_image, dtype=np.float64)
            hmr_raw = compute_hmr(img_raw, roi_h, roi_m).hmr

            prev_c, prev_s = self._exclude_bone_enabled, self._exclude_sternum_enabled
            try:
                # Solo esternón
                self._exclude_bone_enabled, self._exclude_sternum_enabled = False, True
                img_st, _ = self._build_ap_quant_image(self._active_time if self._active_time in ("1h", "3h") else "1h", roi_h)
                hmr_st = compute_hmr(np.asarray(img_st, dtype=np.float64), roi_h, roi_m).hmr

                # Solo costal
                self._exclude_bone_enabled, self._exclude_sternum_enabled = True, False
                img_co, _ = self._build_ap_quant_image(self._active_time if self._active_time in ("1h", "3h") else "1h", roi_h)
                hmr_co = compute_hmr(np.asarray(img_co, dtype=np.float64), roi_h, roi_m).hmr

                # Ambos según estado actual
                self._exclude_bone_enabled, self._exclude_sternum_enabled = prev_c, prev_s
                img_both, _ = self._build_ap_quant_image(self._active_time if self._active_time in ("1h", "3h") else "1h", roi_h)
                hmr_both = compute_hmr(np.asarray(img_both, dtype=np.float64), roi_h, roi_m).hmr
            finally:
                self._exclude_bone_enabled, self._exclude_sternum_enabled = prev_c, prev_s

            self._lbl_filter_summary.setText(
                f"Filtro costal: {'ON' if costal_on else 'OFF'} · Filtro esternón: {'ON' if sternum_on else 'OFF'} · SCATTER: {'ON' if scatter_on else 'OFF'}\n"
                f"{order_txt}\n"
                f"ΔHMR esternón = {hmr_st - hmr_raw:+.2f} · ΔHMR costal = {hmr_co - hmr_raw:+.2f} · ΔHMR ambos = {hmr_both - hmr_raw:+.2f}"
            )
        except Exception:
            self._lbl_filter_summary.setText(
                f"Filtro costal: {'ON' if costal_on else 'OFF'} · "
                f"Filtro esternón: {'ON' if sternum_on else 'OFF'} · "
                f"SCATTER: {'ON' if scatter_on else 'OFF'}\n{order_txt}"
            )

    def _build_current_display_image(self) -> np.ndarray | None:
        if self._original_image is None:
            return None
        base = np.asarray(self._original_image, dtype=np.float64)
        mode = str(getattr(self, "_result_view_mode", "original") or "original")
        if mode == "original":
            return base
        if self._active_time not in ("1h", "3h"):
            return base
        try:
            roi_h = ROICircle(
                cy=self._roi_widget._rois[0]["cy"],
                cx=self._roi_widget._rois[0]["cx"],
                radius=self._roi_widget._rois[0]["radius"],
            )
            corr, _meta = self._build_ap_quant_image(self._active_time, roi_h)
            corr = np.asarray(corr, dtype=np.float64)
            if mode == "corrected":
                return corr
            if mode == "difference":
                diff = np.clip(base - corr, 0.0, None)
                return diff
        except Exception:
            return base
        return base

    def _update_roi_display_image(self):
        img = self._build_current_display_image()
        if img is None:
            return
        filter_key = self._filter_combo.currentData() if hasattr(self, "_filter_combo") else None
        if filter_key is None or filter_key == "none":
            self._roi_widget._image = np.asarray(img, dtype=np.float64)
            self._roi_widget.update()
            return
        self._on_visual_filter_changed(self._filter_combo.currentIndex())

    def _build_ap_quant_image(self, time_label: str, roi_h: ROICircle) -> tuple[np.ndarray, dict]:
        """Construye imagen para cuantificación AP con correcciones experimentales opcionales."""
        img = np.asarray(self._original_image, dtype=np.float64)
        source_ap = self._time_images.get(time_label, {}).get("ap") or {}
        meta = {
            "scatter_used": False,
            "scatter_k": 0.0,
            "rib_filter_used": False,
            "rib_filter_method": self._exclude_bone_method,
            "rib_filter_level": 0.0,
            "rib_filter_asym": None,
            "rib_filter_note": "",
            "sternum_filter_used": False,
            "sternum_filter_level": 0.0,
            "sternum_filter_asym": None,
            "sternum_filter_note": "",
        }

        # 1) Scatter planar (si está disponible y habilitado)
        if self._use_scatter_planar:
            sc = source_ap.get("scatter_image")
            if isinstance(sc, np.ndarray) and sc.shape == img.shape:
                k = float(self._scatter_planar_k)
                img = np.clip(img - k * np.asarray(sc, dtype=np.float64), 0.0, None)
                meta["scatter_used"] = True
                meta["scatter_k"] = k
            else:
                meta["rib_filter_note"] = "SC no disponible o geometría distinta"

        # 2) Filtro esternal (independiente del costal)
        if self._exclude_sternum_enabled:
            if len(self._roi_widget._rois) < 6:
                meta["sternum_filter_note"] = "Faltan ROIs Fondo est. 1/2 para filtro esternal"
            else:
                sternum_roi = ROICircle(
                    cy=self._roi_widget._rois[2]["cy"],
                    cx=self._roi_widget._rois[2]["cx"],
                    radius=self._roi_widget._rois[2]["radius"],
                )
                bg1_roi = ROICircle(
                    cy=self._roi_widget._rois[4]["cy"],
                    cx=self._roi_widget._rois[4]["cx"],
                    radius=self._roi_widget._rois[4]["radius"],
                )
                bg2_roi = ROICircle(
                    cy=self._roi_widget._rois[5]["cy"],
                    cx=self._roi_widget._rois[5]["cx"],
                    radius=self._roi_widget._rois[5]["radius"],
                )
                m_st = sternum_roi.mask(img.shape)
                m_bg1 = bg1_roi.mask(img.shape)
                m_bg2 = bg2_roi.mask(img.shape)
                if not np.any(m_st) or not np.any(m_bg1) or not np.any(m_bg2):
                    meta["sternum_filter_note"] = "ROI esternón/fondos inválidas"
                else:
                    st_mean = float(np.mean(img[m_st]))
                    bg1_mean = float(np.mean(img[m_bg1]))
                    bg2_mean = float(np.mean(img[m_bg2]))
                    bg_mean = 0.5 * (bg1_mean + bg2_mean)
                    asym_bg = abs(bg1_mean - bg2_mean) / max(bg_mean, 1e-8)
                    meta["sternum_filter_asym"] = asym_bg
                    if asym_bg > float(self._exclude_sternum_asym_thresh):
                        meta["sternum_filter_note"] = (
                            f"Asimetría fondos esternales alta ({asym_bg:.2f}) > umbral {self._exclude_sternum_asym_thresh:.2f}; corrección descartada"
                        )
                    else:
                        level_st = max(0.0, min(st_mean - bg_mean, float(np.percentile(img, 95))))
                        img = np.clip(img - (m_st.astype(np.float64) * level_st), 0.0, None)
                        meta["sternum_filter_used"] = True
                        meta["sternum_filter_level"] = level_st
                        meta["sternum_filter_note"] = (
                            f"Filtro esternón aplicado (nivel={level_st:.2f}, bg={bg_mean:.2f}, asim={asym_bg:.2f})"
                        )

        # 3) Filtro costal (independiente del esternal)
        if not self._exclude_bone_enabled or len(self._roi_widget._rois) < 4:
            return img, meta

        rib_r = ROICircle(
            cy=self._roi_widget._rois[3]["cy"],
            cx=self._roi_widget._rois[3]["cx"],
            radius=self._roi_widget._rois[3]["radius"],
        )
        rib_l = ROICircle(
            cy=rib_r.cy,
            cx=(img.shape[1] - 1) - rib_r.cx,
            radius=rib_r.radius,
        )
        m_r = rib_r.mask(img.shape)
        m_l = rib_l.mask(img.shape)
        if not np.any(m_r) or not np.any(m_l):
            meta["rib_filter_note"] = "ROI costal inválida"
            return img, meta

        right_mean = float(np.mean(img[m_r]))
        left_mean = float(np.mean(img[m_l]))
        asym = abs(left_mean - right_mean) / max(right_mean, 1e-8)
        meta["rib_filter_asym"] = asym
        if asym > float(self._exclude_bone_asym_thresh):
            meta["rib_filter_note"] = (
                f"Asimetría costal alta ({asym:.2f}) > umbral {self._exclude_bone_asym_thresh:.2f}; corrección descartada"
            )
            return img, meta

        if self._exclude_bone_method == "scaled_hotspot":
            heart_mask = roi_h.mask(img.shape)
            h_vals = img[heart_mask]
            if h_vals.size == 0:
                return img, meta
            hot_ref = float(np.percentile(h_vals, 95))
            ratio = right_mean / max(hot_ref, 1e-8)
            level = float(left_mean * ratio)
        else:
            level = right_mean

        level = max(0.0, min(level, float(np.percentile(img, 95))))
        img_corr = np.clip(img - level, 0.0, None)
        meta["rib_filter_used"] = True
        meta["rib_filter_level"] = level
        meta["rib_filter_note"] = (
            f"Filtro costal aplicado (nivel={level:.2f}, asimetría={asym:.2f}, método={self._exclude_bone_method})"
        )
        return img_corr, meta

    @staticmethod
    def _relative_hours(first: datetime, current: datetime) -> float:
        """Horas relativas desde la primera adquisición, ajustando rollover de día."""
        cur = current
        if cur < first:
            cur = cur + timedelta(days=1)
        return max(0.0, (cur - first).total_seconds() / 3600.0)

    @staticmethod
    def _closest_label_from_hours(hours: float) -> str:
        """Asigna etiqueta canónica 1h/3h por cercanía temporal."""
        return "1h" if abs(hours - 1.0) <= abs(hours - 3.0) else "3h"

    @staticmethod
    def _fmt_dt(dt: datetime | None) -> str:
        if dt is None:
            return "N/D"
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _assign_views_for_time(self, records: list[dict]) -> dict[str, dict]:
        """Asigna AP/OAI/LAT con heurística por metadatos o fallback por orden."""
        assigned: dict[str, dict | None] = {"ap": None, "oai": None, "lat": None}
        used: set[int] = set()

        # Prioridad 1: clasificación explícita por metadatos DICOM.
        for idx, record in enumerate(records):
            view = record.get("view")
            if view in assigned and assigned[view] is None:
                assigned[view] = record
                used.add(idx)

        # Prioridad 2: completar AP/OAI con remanentes si faltan.
        for role in ("ap", "oai"):
            if assigned[role] is None:
                idx = next((i for i in range(len(records)) if i not in used), None)
                if idx is not None:
                    assigned[role] = records[idx]
                    used.add(idx)

        # LAT es opcional: solo completar si quedó algún remanente.
        if assigned["lat"] is None:
            idx = next((i for i in range(len(records)) if i not in used), None)
            if idx is not None:
                assigned["lat"] = records[idx]
                used.add(idx)

        # Si AP faltó (caso extremo), tomar el primero disponible.
        if assigned["ap"] is None and records:
            assigned["ap"] = records[0]

        return {
            "ap": assigned["ap"],
            "oai": assigned["oai"],
            "lat": assigned["lat"],
        }

    @staticmethod
    def _build_label_for_role(record: dict | None, role: str, time_label: str) -> str:
        base_map = {
            "ap": "AP",
            "oai": "OAI",
            "lat": "LAT. IZQ.",
        }
        if record is None:
            return f"{base_map.get(role, role.upper())} ({time_label}) · N/D"
        raw = str(record.get("label") or "").strip()
        if raw:
            return f"{raw} ({time_label})"
        return f"{base_map.get(role, role.upper())} ({time_label})"

    def _load_early_dynamic(self):
        """Carga dinámico 0–5 min. No realiza QC de inyección."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar dinámico temprano PYP (0–5 min)",
            os.path.expanduser("~"),
            "DICOM (*.dcm *.DCM);;Todos (*)",
        )
        if not path:
            return
        try:
            import pydicom
            from core.amyloid_kinetic import normalize_dynamic_frames

            ds = pydicom.dcmread(path, force=True)
            frames = np.asarray(ds.pixel_array, dtype=np.float64)
            if frames.ndim != 3 or frames.shape[0] < 3:
                raise ValueError(f"Se esperaba dinámico [frames, rows, cols], recibido {frames.shape}")
            durations = self._dicom_frame_durations_s(ds, frames.shape[0])
            dynamic = normalize_dynamic_frames(frames, durations, decay_correct=True)
            self._early_dynamic = {
                "path": path,
                "dataset": ds,
                "dynamic": dynamic,
                "summed_cps": dynamic.frames_cps.sum(axis=0),
            }
            total_min = float(np.sum(durations) / 60.0)
            self._lbl_washout_status.setText(
                f"Dinámico temprano cargado: {frames.shape[0]} frames, {total_min:.1f} min. "
                "Método experimental; falta cuantificar 1h y 3h para análisis temporal."
            )
            self._lbl_washout_status.setStyleSheet(
                "font-size:10px; color:#000000; background:#bfdbfe; padding:6px 10px; border:1px solid #93c5fd; border-radius:999px;"
            )
            self._update_kinetic_analysis()
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO — Cinética experimental", f"No se pudo cargar el dinámico:\n{exc}")

    def _show_kinetic_help(self):
        """Muestra fundamento, utilidad y limitaciones del método experimental."""
        from core.amyloid_kinetic import EXPERIMENTAL_EXPLANATION

        dlg = QDialog(self)
        dlg.setWindowTitle("Análisis temporal PYP — Método experimental")
        dlg.resize(720, 560)
        layout = QVBoxLayout(dlg)
        warning = QLabel("MÉTODO EXPERIMENTAL — NO DIAGNÓSTICO / NO TERAPÉUTICO")
        warning.setStyleSheet("color:#fbbf24; font-weight:bold; font-size:14px; padding:8px;")
        warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(warning)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml(
            f"<h2>{EXPERIMENTAL_EXPLANATION['title']}</h2>"
            f"<h3>Qué hace</h3><p>{EXPERIMENTAL_EXPLANATION['summary']}</p>"
            f"<h3>Fundamento físico</h3><p>{EXPERIMENTAL_EXPLANATION['physics']}</p>"
            f"<h3>Posible utilidad diferencial</h3><p>{EXPERIMENTAL_EXPLANATION['differential']}</p>"
            f"<h3>Limitaciones</h3><p>{EXPERIMENTAL_EXPLANATION['limitations']}</p>"
            f"<h3>Advertencia clínica</h3><p><b>{EXPERIMENTAL_EXPLANATION['clinical_warning']}</b></p>"
        )
        layout.addWidget(text)
        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec()

    def _open_anatomical_3d(self):
        """Abre el visor anatómico 3D como prueba de concepto."""
        uptake_values = None
        image_ap = None
        try:
            if self._active_time in ("1h", "3h"):
                ap_entry = self._time_images[self._active_time]["ap"]
            else:
                ap_entry = self._time_images["1h"]["ap"]
            if ap_entry is not None:
                img = np.asarray(ap_entry["image"], dtype=np.float64)
                image_ap = img
                uptake_values = img.ravel()
        except Exception:
            uptake_values = None
            image_ap = None

        dlg = Anatomical3DPanel(
            self,
            uptake_values=uptake_values,
            image_ap=image_ap,
            title="Corazón 3D anatómico",
        )
        dlg.exec()

    def _open_amyloid_spect_3d(self):
        """Abre panel AMYLO SPECT 3D (fase 2, experimental)."""
        dlg = AmyloidSpectPanel(self)
        dlg.exec()

    def _load_time_images(self, time_label: str):
        """Carga 1 a 3 planares y asigna AP/OAI/LAT de forma flexible."""
        from ui.dicom_browser import DicomBrowserDialog
        # Determinar directorio inicial.
        start_dir = ""
        if self._study:
            # Intentar obtener la ruta del estudio actual.
            start_dir = getattr(self._study, "_source_path", "") or ""
            if start_dir and os.path.isfile(start_dir):
                start_dir = os.path.dirname(start_dir)
        if not start_dir:
            start_dir = os.path.expanduser("~")
        browser = DicomBrowserDialog(self, start_dir=start_dir)
        if browser.exec() == QDialog.DialogCode.Accepted:
            paths = browser.selected_paths()
            if len(paths) < 1 or len(paths) > 3:
                QMessageBox.information(
                    self, "SINCRO — Amyloidosis",
                    "Seleccioná entre 1 y 3 DICOM planares para este tiempo (AP/OAI/LAT opcional)."
                )
                return
            records = []
            for p in paths:
                try:
                    records.append(self._extract_planar_record(p))
                except Exception as exc:
                    QMessageBox.warning(self, "SINCRO", f"Error cargando {os.path.basename(p)}:\n{exc}")
                    return

            self._time_images[time_label] = self._assign_views_for_time(records)
            self._processed_images[time_label].clear()
            # Mantener ROIs persistidas si existen; inicializar solo cuando no hay estado previo.
            if not self._roi_state.get(time_label):
                self._roi_state[time_label] = None
            self._washout_data.pop(time_label, None)
            self._time_hours_by_label[time_label] = 1.0 if time_label == "1h" else 3.0
            first_ds = records[0]["ds"]
            self._metadata = {
                "patient": str(getattr(first_ds, "PatientName", "") or "N/D"),
                "date": str(getattr(first_ds, "StudyDate", "") or "N/D"),
                "series": str(getattr(first_ds, "SeriesDescription", "") or "N/D"),
            }
            self._info_lbl.setText(
                f"Paciente: {self._metadata['patient']}  ·  Fecha: {self._metadata['date']}  ·  Serie: {self._metadata['series']}"
            )
            target_layout = 8 if self._time_images["3h"]["ap"] is not None else 4
            combo_idx = self._layout_combo.findData(target_layout)
            self._layout_combo.blockSignals(True)
            self._layout_combo.setCurrentIndex(combo_idx)
            self._layout_combo.blockSignals(False)
            self._current_layout_n = target_layout
            self._page_offset = 0
            self._rebuild_layout(force_layout=target_layout)
            self._persist_user_state()
            self._update_washout_preview()
            self._update_kinetic_analysis()

            loaded_roles = [r for r in ("ap", "oai", "lat") if self._time_images[time_label].get(r) is not None]
            missing_roles = [r for r in ("ap", "oai", "lat") if self._time_images[time_label].get(r) is None]
            QMessageBox.information(
                self,
                "SINCRO — Amyloidosis",
                "Carga flexible completada.\n"
                f"Tiempo {time_label}: cargados {', '.join(loaded_roles) if loaded_roles else 'ninguno'}.\n"
                f"Faltantes: {', '.join(missing_roles) if missing_roles else 'ninguno'}."
            )

    def _load_washout_auto(self):
        """Carga planares (4 a 6) y asigna 1h/3h automáticamente por metadata temporal."""
        from ui.dicom_browser import DicomBrowserDialog

        start_dir = ""
        if self._study:
            start_dir = getattr(self._study, "_source_path", "") or ""
            if start_dir and os.path.isfile(start_dir):
                start_dir = os.path.dirname(start_dir)
        if not start_dir:
            start_dir = os.path.expanduser("~")

        browser = DicomBrowserDialog(self, start_dir=start_dir)
        if browser.exec() != QDialog.DialogCode.Accepted:
            return
        paths = browser.selected_paths()
        if len(paths) < 4 or len(paths) > 6:
            QMessageBox.information(
                self,
                "SINCRO — Washout automático",
                "Seleccioná entre 4 y 6 DICOM planares (ideal: 3 de ~1h y 3 de ~3h).",
            )
            return

        try:
            records = [self._extract_planar_record(p) for p in paths]
        except Exception as exc:
            QMessageBox.warning(self, "SINCRO", f"Error cargando selección:\n{exc}")
            return

        dts = [r.get("acq_dt") for r in records if r.get("acq_dt") is not None]
        if len(dts) < 2:
            QMessageBox.warning(
                self,
                "SINCRO — Washout automático",
                "Metadata temporal insuficiente. Usá la carga manual 1h/3h.",
            )
            return

        t0 = min(dts)
        for r in records:
            dt = r.get("acq_dt")
            r["rel_h"] = self._relative_hours(t0, dt) if dt is not None else 0.0
            r["time_label_guess"] = self._closest_label_from_hours(r["rel_h"])

        g1 = [r for r in records if r.get("time_label_guess") == "1h"]
        g3 = [r for r in records if r.get("time_label_guess") == "3h"]
        if len(g1) < 2 or len(g3) < 2:
            # fallback robusto: split temporal 50/50 por orden relativo
            ordered = sorted(records, key=lambda x: x.get("rel_h", 0.0))
            half = max(2, len(ordered) // 2)
            g1 = ordered[:half]
            g3 = ordered[half:]

        if len(g1) < 2 or len(g3) < 2:
            QMessageBox.warning(
                self,
                "SINCRO — Washout automático",
                "No se pudo separar en 2 tiempos con suficiente cobertura (mínimo 2 por grupo). Usá carga manual.",
            )
            return

        self._time_images["1h"] = self._assign_views_for_time(g1)
        self._time_images["3h"] = self._assign_views_for_time(g3)
        self._processed_images["1h"].clear()
        self._processed_images["3h"].clear()
        self._washout_data.pop("1h", None)
        self._washout_data.pop("3h", None)

        # Horas reales para curvas e informe.
        self._time_hours_by_label["1h"] = float(np.mean([r.get("rel_h", 1.0) for r in g1]))
        self._time_hours_by_label["3h"] = float(np.mean([r.get("rel_h", 3.0) for r in g3]))

        first_ds = g1[0]["ds"]
        self._metadata = {
            "patient": str(getattr(first_ds, "PatientName", "") or "N/D"),
            "date": str(getattr(first_ds, "StudyDate", "") or "N/D"),
            "series": str(getattr(first_ds, "SeriesDescription", "") or "N/D"),
        }
        self._info_lbl.setText(
            f"Paciente: {self._metadata['patient']}  ·  Fecha: {self._metadata['date']}  ·  Serie: {self._metadata['series']}"
        )

        target_layout = 8
        combo_idx = self._layout_combo.findData(target_layout)
        self._layout_combo.blockSignals(True)
        self._layout_combo.setCurrentIndex(combo_idx)
        self._layout_combo.blockSignals(False)
        self._current_layout_n = target_layout
        self._page_offset = 0
        self._rebuild_layout(force_layout=target_layout)
        self._persist_user_state()
        self._update_washout_preview()
        self._update_kinetic_analysis()

        QMessageBox.information(
            self,
            "SINCRO — Washout automático",
            (
                "Asignación automática completada.\n"
                f"Tiempo temprano: ~{self._time_hours_by_label['1h']:.2f} h\n"
                f"Tiempo tardío: ~{self._time_hours_by_label['3h']:.2f} h\n"
                f"Cobertura detectada: 1h={len(g1)} archivos, 3h={len(g3)} archivos.\n"
                "Podés seguir usando el modo manual 1h/3h cuando quieras."
            ),
        )

        choice = QMessageBox.question(
            self,
            "SINCRO — Washout automático",
            "¿Querés ver el detalle temporal detectado por archivo?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice == QMessageBox.StandardButton.Yes:
            def _block(title: str, group: list[dict]) -> str:
                lines = [f"{title}:"]
                for r in sorted(group, key=lambda x: x.get("rel_h", 0.0)):
                    lines.append(
                        f"- {os.path.basename(str(r.get('path', '')))} | "
                        f"{self._fmt_dt(r.get('acq_dt'))} | "
                        f"t={float(r.get('rel_h', 0.0)):.2f} h"
                    )
                return "\n".join(lines)

            detail = (
                _block("Grupo temprano (1h)", g1)
                + "\n\n"
                + _block("Grupo tardío (3h)", g3)
            )
            QMessageBox.information(self, "SINCRO — Detalle temporal", detail)

    @staticmethod
    def _classify_planar_view(text: str) -> str | None:
        """Clasifica ViewPosition/SeriesDescription; None activa fallback por orden."""
        normalized = " ".join(text.upper().replace("_", " ").replace("-", " ").split())
        words = set(normalized.split())
        if "LAT" in words or any(token in normalized for token in ("LATERAL", "LAT IZQ", "LEFT LAT", "LLAT")):
            return "lat"
        if any(token in normalized for token in ("OAI", "LAO", "OBLICUA ANTERIOR IZQ", "LEFT ANTERIOR OBLIQUE")):
            return "oai"
        if "AP" in words or "ANTERIOR" in words or "ANT" in words:
            return "ap"
        return None

    # ── Controles del cuadrante ─────────────────────────────────────

    def _on_quadrant_selected(self, idx: int):
        # Si estamos en modo swap, ejecutar el intercambio.
        if self._swap_mode and self._swap_first >= 0 and idx != self._swap_first:
            self._do_swap(self._swap_first, idx)
            self._swap_mode = False
            self._swap_first = -1
            self._btn_swap.setText("Swap")
            self._btn_swap.setStyleSheet("background: #d97706; color: white; font-weight: bold; padding: 4px;")
            return
        elif self._swap_mode:
            self._swap_first = idx
            self._btn_swap.setText(f"Swap ← #{idx+1} (click destino)")
            return
        q = self._quadrant_viewer.selected_quadrant()
        if q is None:
            return
        self._lbl_sel_quad.setText(f"#{idx+1}: {q.label}")
        # Actualizar controles sin disparar señales.
        self._cmap_combo.blockSignals(True)
        self._cmap_combo.setCurrentText(q.cmap)
        self._cmap_combo.blockSignals(False)
        self._win_low_slider.blockSignals(True)
        self._win_low_slider.setValue(int(q.win_low))
        self._win_low_slider.blockSignals(False)
        self._win_high_slider.blockSignals(True)
        self._win_high_slider.setValue(int(q.win_high))
        self._win_high_slider.blockSignals(False)
        self._update_quadrant_filter_label(q)

    def _edit_quadrant_label(self, idx: int):
        """Permite editar rótulo del cuadrante con click directo en el label."""
        layout = self._quadrant_viewer._layout
        if layout is None or idx < 0 or idx >= len(layout.quadrants):
            return
        q = layout.quadrants[idx]
        current = str(q.label or f"#{idx+1}")
        new_label, ok = QInputDialog.getText(self, "Editar rótulo", "Rótulo del cuadrante:", text=current)
        if not ok:
            return
        new_label = str(new_label).strip()
        if not new_label:
            return
        q.label = new_label
        self._quadrant_viewer.update()
        self._on_quadrant_selected(idx)
        self._persist_user_state()

    def _update_quadrant_filter_label(self, q):
        active = list(getattr(q, "filters", []) or [])
        txt = ", ".join(active) if active else "ninguno"
        self._lbl_filters_active.setText(f"Activos: {txt}")

    def _save_selected_quadrant_state(self):
        idx = self._quadrant_viewer._selected
        q = self._quadrant_viewer.selected_quadrant()
        if q is None or idx < 0:
            return
        self._quadrant_state[int(idx)] = {
            "cmap": str(q.cmap),
            "win_low": float(q.win_low),
            "win_high": float(q.win_high),
            "filters": list(q.filters),
        }

    def _restore_quadrant_states(self):
        layout = self._quadrant_viewer._layout
        if layout is None:
            return
        for idx, q in enumerate(layout.quadrants):
            state = self._quadrant_state.get(idx) or self._quadrant_state.get(str(idx))
            if not state:
                continue
            try:
                q.cmap = str(state.get("cmap", q.cmap))
                q.win_low = float(state.get("win_low", q.win_low))
                q.win_high = float(state.get("win_high", q.win_high))
                q.filters = list(state.get("filters", q.filters))
            except Exception:
                continue

    def _on_cmap_changed(self, cmap: str):
        q = self._quadrant_viewer.selected_quadrant()
        if q is None:
            return
        q.cmap = cmap
        self._save_selected_quadrant_state()
        self._quadrant_viewer._rebuild_pixmaps()
        self._quadrant_viewer.update()

    def _on_window_changed(self):
        q = self._quadrant_viewer.selected_quadrant()
        if q is None:
            return
        q.win_low = float(self._win_low_slider.value())
        q.win_high = float(self._win_high_slider.value())
        self._save_selected_quadrant_state()
        self._quadrant_viewer._rebuild_pixmaps()
        self._quadrant_viewer.update()

    def _toggle_filter(self, filt: str):
        q = self._quadrant_viewer.selected_quadrant()
        if q is None:
            return
        if filt in q.filters:
            q.filters.remove(filt)
        else:
            q.filters.append(filt)
        self._update_quadrant_filter_label(q)
        self._save_selected_quadrant_state()
        self._quadrant_viewer._rebuild_pixmaps()
        self._quadrant_viewer.update()

    # ── Swap entre cuadrantes ───────────────────────────────────────

    def _delete_selected_quadrant(self):
        """Borra el cuadrante seleccionado (limpia su imagen)."""
        q = self._quadrant_viewer.selected_quadrant()
        if q is None:
            return
        q.image = None
        q.label = "(vacío)"
        q.hmr = None
        q.roi_overlay = False
        self._quadrant_viewer._rebuild_pixmaps()
        self._quadrant_viewer.update()

    def _toggle_swap_mode(self):
        """Activa/desactiva el modo swap entre cuadrantes."""
        self._swap_mode = not self._swap_mode
        self._swap_first = -1
        if self._swap_mode:
            self._btn_swap.setText("Swap (click origen)")
            self._btn_swap.setStyleSheet("background: #ef4444; color: white; font-weight: bold; padding: 4px;")
        else:
            self._btn_swap.setText("Swap")
            self._btn_swap.setStyleSheet("background: #d97706; color: white; font-weight: bold; padding: 4px;")

    def _do_swap(self, idx_a: int, idx_b: int):
        """Intercambia las imágenes entre dos cuadrantes."""
        layout = self._quadrant_viewer._layout
        if layout is None:
            return
        qa = layout.quadrants[idx_a]
        qb = layout.quadrants[idx_b]
        # Intercambiar imagen, label, cmap, filtros, ventana, hmr.
        qa.image, qb.image = qb.image, qa.image
        qa.label, qb.label = qb.label, qa.label
        qa.cmap, qb.cmap = qb.cmap, qa.cmap
        qa.filters, qb.filters = list(qb.filters), list(qa.filters)
        qa.win_low, qb.win_low = qb.win_low, qa.win_low
        qa.win_high, qb.win_high = qb.win_high, qa.win_high
        qa.hmr, qb.hmr = qb.hmr, qa.hmr
        qa.roi_overlay, qb.roi_overlay = qb.roi_overlay, qa.roi_overlay
        self._quadrant_state[idx_a], self._quadrant_state[idx_b] = self._quadrant_state.get(idx_b, {}), self._quadrant_state.get(idx_a, {})
        self._quadrant_viewer._rebuild_pixmaps()
        self._quadrant_viewer.update()

    # ── Análisis ROI del cuadrante seleccionado ─────────────────────

    def _analyze_selected(self):
        """Abre la imagen del cuadrante seleccionado en modo análisis ROI."""
        idx = self._quadrant_viewer._selected
        q = self._quadrant_viewer.selected_quadrant()
        if q is None or q.image is None:
            QMessageBox.information(self, "SINCRO", "Selecciona un cuadrante con imagen primero.")
            return
        slot_to_meta = {}
        if self._current_layout_n == 6:
            slot_to_meta = {
                0: ("1h", "ap"), 1: ("1h", "ap"), 2: ("1h", "oai"),
                3: ("3h", "ap"), 4: ("3h", "ap"), 5: ("3h", "oai"),
            }
        elif self._current_layout_n == 1234:
            if self._layout_12q3x4_lat_hidden:
                slot_to_meta = {
                    0: ("1h", "ap"), 1: ("1h", "ap"), 2: ("1h", "oai"),
                    3: ("3h", "ap"), 4: ("3h", "ap"), 5: ("3h", "oai"),
                }
            else:
                slot_to_meta = {
                    0: ("1h", "ap"), 1: ("1h", "ap"), 2: ("1h", "oai"), 3: ("1h", "lat"),
                    4: ("3h", "ap"), 5: ("3h", "ap"), 6: ("3h", "oai"), 7: ("3h", "lat"),
                }
        elif self._current_layout_n >= 8:
            slot_to_meta = {
                0: ("1h", "ap"), 1: ("1h", "ap"), 2: ("1h", "oai"), 3: ("1h", "lat"),
                4: ("3h", "ap"), 5: ("3h", "ap"), 6: ("3h", "oai"), 7: ("3h", "lat"),
            }
        else:
            slot_to_meta = {
                0: ("1h", "ap"), 1: ("1h", "ap"), 2: ("1h", "oai"), 3: ("1h", "lat"),
            }

        meta = slot_to_meta.get(idx)
        time_label = meta[0] if meta else None
        view_role = meta[1] if meta else "ap"
        if time_label is None:
            QMessageBox.information(
                self, "SINCRO — Amyloidosis",
                "Seleccioná un cuadrante válido para análisis (AP/OAI/LAT del tiempo correspondiente)."
            )
            return
        entry = self._time_images[time_label].get(view_role)
        if entry is None:
            QMessageBox.information(self, "SINCRO — Amyloidosis", f"No hay vista {view_role.upper()} cargada para {time_label}.")
            return
        img = np.asarray(entry["image"], dtype=np.float64)
        self._image = img  # imagen actual para display/render
        self._original_image = img.copy()  # original 2D para análisis
        self._active_time = time_label
        self._active_view_role = view_role
        self._time_combo.setCurrentText(time_label)
        idx_role = self._view_role_combo.findData(view_role)
        if idx_role >= 0:
            self._view_role_combo.blockSignals(True)
            self._view_role_combo.setCurrentIndex(idx_role)
            self._view_role_combo.blockSignals(False)
        self._qbone_mode_by_time[time_label] = self._qbone_mode_by_time.get(time_label, "auto")
        self._roi_widget = ROIDragWidget(img)
        self._ensure_aux_rois()
        saved_rois = self._roi_state_oai.get(time_label) if view_role == "oai" else self._roi_state.get(time_label)
        if saved_rois:
            self._roi_widget._rois = [dict(roi) for roi in saved_rois]
        self._roi_widget.roiChanged.connect(self._update_hmr)
        # Reemplazar el widget de ROI en la página de análisis.
        old = self._stack.widget(1)
        if old is None:
            return
        old_layout = old.layout()
        if old_layout:
            # Quitar el widget viejo y poner el nuevo.
            for i in range(old_layout.count()):
                item = old_layout.itemAt(i)
                if item is None:
                    continue
                w = item.widget()
                if isinstance(w, ROIDragWidget):
                    old_layout.removeWidget(w)
                    w.deleteLater()
                    if isinstance(old_layout, QVBoxLayout):
                        old_layout.insertWidget(0, self._roi_widget, 1)
                    break
        # Cargar score guardado por tiempo o sugerir uno nuevo por HMR actual.
        if view_role == "ap" and time_label in self._perugini_by_time:
            self._set_perugini_score(int(self._perugini_by_time[time_label]))
        elif view_role == "ap":
            try:
                roi_h = ROICircle(
                    cy=self._roi_widget._rois[0]["cy"],
                    cx=self._roi_widget._rois[0]["cx"],
                    radius=self._roi_widget._rois[0]["radius"],
                )
                roi_m = ROICircle(
                    cy=self._roi_widget._rois[1]["cy"],
                    cx=self._roi_widget._rois[1]["cx"],
                    radius=self._roi_widget._rois[1]["radius"],
                )
                hmr_now = compute_hmr(self._original_image, roi_h, roi_m).hmr
                self._set_perugini_score(self._suggest_perugini_from_hmr(hmr_now))
            except Exception:
                pass
            self._perugini_confirm_chk.setChecked(bool(self._perugini_confirmed_by_time.get(time_label, False)))

        # En OAI la lectura principal no es Perugini/HMR.
        self._perugini_combo.setEnabled(view_role == "ap")
        self._perugini_confirm_chk.setEnabled(view_role == "ap")
        self._sync_qbone_mode_ui()
        self._update_roi_display_image()
        self._update_filter_summary()
        self._toggle_mode()
        self._update_hmr(0, 0, 0, 0)

    def _roi_slot_index(self, time_label: str) -> int:
        """Devuelve índice de cuadrante AP+ROIs para el tiempo dado según layout activo."""
        if time_label == "1h":
            return 0
        if self._current_layout_n == 1234 and self._layout_12q3x4_lat_hidden:
            return 3
        if self._current_layout_n == 1234:
            return 4
        if self._current_layout_n == 6:
            return 3
        if self._current_layout_n >= 8:
            return 4
        return 0

    def _sync_qbone_mode_ui(self):
        time_label = self._active_time if self._active_time in ("1h", "3h") else "1h"
        mode = self._qbone_mode_by_time.get(time_label, "auto")
        idx = self._qbone_mode_combo.findData(mode)
        self._qbone_mode_combo.blockSignals(True)
        if idx >= 0:
            self._qbone_mode_combo.setCurrentIndex(idx)
        self._qbone_mode_combo.blockSignals(False)
        self._roi_widget.set_aux_rois_visible(mode == "manual")
        show_preview = bool(getattr(self, "_show_mirror_roi_chk", None) and self._show_mirror_roi_chk.isChecked())
        self._roi_widget.set_mirror_roi_visible(show_preview and mode == "manual" and self._exclude_bone_enabled)
        self._qbone_hint_lbl.setText(
            "Q_bone manual: ajustar ROI Esternón y ROI Costilla." if mode == "manual"
            else "Q_bone automático: esternón/costilla se estiman desde ROI cardíaco."
        )

    def _on_qbone_mode_changed(self, idx: int):
        mode = str(self._qbone_mode_combo.currentData() or "auto")
        time_label = self._active_time if self._active_time in ("1h", "3h") else "1h"
        self._qbone_mode_by_time[time_label] = mode
        if self._active_time in self._roi_state:
            self._roi_state[self._active_time] = [dict(roi) for roi in self._roi_widget._rois]
        self._roi_widget.set_aux_rois_visible(mode == "manual")
        show_preview = bool(getattr(self, "_show_mirror_roi_chk", None) and self._show_mirror_roi_chk.isChecked())
        self._roi_widget.set_mirror_roi_visible(show_preview and mode == "manual" and self._exclude_bone_enabled)
        self._qbone_hint_lbl.setText(
            "Q_bone manual: ajustar ROI Esternón y ROI Costilla." if mode == "manual"
            else "Q_bone automático: esternón/costilla se estiman desde ROI cardíaco."
        )
        self._persist_user_state()

    def _on_view_role_changed(self, idx: int):
        """Cambia entre análisis AP (HMR) y OAI (washout opcional)."""
        role = str(self._view_role_combo.currentData() or "ap")
        self._active_view_role = role
        if self._active_time not in ("1h", "3h"):
            return
        source = self._time_images.get(self._active_time, {})
        entry = source.get(role)
        if entry is None:
            QMessageBox.information(
                self,
                "SINCRO — Amyloidosis",
                f"No hay vista {role.upper()} cargada para {self._active_time}.",
            )
            return
        try:
            img = np.asarray(entry["image"], dtype=np.float64)
            self._image = img
            self._original_image = img.copy()
            self._roi_widget = ROIDragWidget(img)
            self._ensure_aux_rois()
            if role == "oai":
                saved_rois = self._roi_state_oai.get(self._active_time)
            else:
                saved_rois = self._roi_state.get(self._active_time)
            if saved_rois:
                self._roi_widget._rois = [dict(roi) for roi in saved_rois]
            self._roi_widget.roiChanged.connect(self._update_hmr)
            page = self._stack.widget(1)
            lay = page.layout()
            old = lay.itemAt(0).widget()
            lay.replaceWidget(old, self._roi_widget)
            old.deleteLater()
            self._sync_qbone_mode_ui()
            self._update_roi_display_image()
            self._update_filter_summary()
            self._update_hmr(0, 0, 0, 0)
            # En OAI ocultar mediastino/HMR como métrica principal, queda opcional.
            if role == "oai":
                self._lbl_hmr.setText("OAI ROI: métrica opcional washout")
                self._lbl_class.setText("Comparación 1h vs 3h opcional (no reemplaza HMR AP)")
        except Exception as exc:
            QMessageBox.warning(self, "SINCRO", f"No se pudo cambiar a vista {role.upper()}:\n{exc}")

    def _apply_rois_to_quadrant(self):
        """Aplica el análisis al par AP+ROI/AP limpio del tiempo activo."""
        time_label = self._active_time
        if self._original_image is None or time_label not in ("1h", "3h"):
            return
        view_role = self._active_view_role if self._active_view_role in ("ap", "oai") else "ap"

        # Rama OAI opcional: comparar 1h vs 3h de ROI cardíaco (sin HMR/mediastino).
        if view_role == "oai":
            try:
                from core.amyloid_kinetic import normalize_static_image
                roi_h = ROICircle(
                    cy=self._roi_widget._rois[0]["cy"],
                    cx=self._roi_widget._rois[0]["cx"],
                    radius=self._roi_widget._rois[0]["radius"],
                )
                mask = roi_h.mask(self._original_image.shape)
                if not np.any(mask):
                    raise ValueError("ROI cardíaco OAI vacío")

                entry = self._time_images.get(time_label, {}).get("oai")
                if entry is None:
                    raise ValueError(f"No hay OAI cargada para {time_label}")

                dt = entry.get("acq_dt")
                if dt is None:
                    acq_min = 60.0 if time_label == "1h" else 180.0
                else:
                    acq_min = float(self._time_hours_by_label.get(time_label, 1.0 if time_label == "1h" else 3.0)) * 60.0

                duration_s = float(entry.get("duration_s") or 1.0)
                norm = normalize_static_image(
                    np.asarray(self._original_image, dtype=np.float64),
                    max(duration_s, 1e-6),
                    acq_min,
                    decay_correct=True,
                    reference_time_min=0.0,
                )
                oai_heart_counts = float(np.asarray(norm)[mask].mean())
                self._oai_washout_data[time_label] = {
                    "heart_counts": oai_heart_counts,
                    "heart_area_px": int(np.count_nonzero(mask)),
                }
                self._roi_state_oai[time_label] = [dict(roi) for roi in self._roi_widget._rois]

                msg = f"OAI {time_label}: cuentas corazón (norm.) = {oai_heart_counts:.4f}"
                if "1h" in self._oai_washout_data and "3h" in self._oai_washout_data:
                    h1 = float(self._oai_washout_data["1h"].get("heart_counts", 0.0))
                    h3 = float(self._oai_washout_data["3h"].get("heart_counts", 0.0))
                    if h1 > 0:
                        ret = h3 / h1
                        wo = (1.0 - ret) * 100.0
                        msg += f"\nRetención OAI 3h/1h = {ret:.3f} · Washout OAI = {wo:+.1f}%"

                self._lbl_class.setText("OAI opcional guardado (no reemplaza HMR AP)")
                self._persist_user_state()
                self._update_washout_preview()
                QMessageBox.information(self, "SINCRO — OAI opcional", msg)
            except Exception as exc:
                QMessageBox.warning(self, "SINCRO", f"Error guardando OAI opcional:\n{exc}")
            return

        try:
            roi_h = ROICircle(
                cy=self._roi_widget._rois[0]["cy"],
                cx=self._roi_widget._rois[0]["cx"],
                radius=self._roi_widget._rois[0]["radius"],
            )
            roi_m = ROICircle(
                cy=self._roi_widget._rois[1]["cy"],
                cx=self._roi_widget._rois[1]["cx"],
                radius=self._roi_widget._rois[1]["radius"],
            )
            raw_result = compute_hmr(self._original_image, roi_h, roi_m)
            quant_img, corr_meta = self._build_ap_quant_image(time_label, roi_h)
            result = compute_hmr(quant_img, roi_h, roi_m)
        except Exception as exc:
            QMessageBox.warning(self, "SINCRO", f"Error calculando HMR:\n{exc}")
            return

        # Guardar datos para curva de washout + Q_bone.
        from core.amyloid_planar import compute_q_bone
        q_bone_val = None
        try:
            mode = self._qbone_mode_by_time.get(time_label, "auto")
            if mode == "manual" and len(self._roi_widget._rois) >= 4:
                roi_sternum = ROICircle(
                    cy=self._roi_widget._rois[2]["cy"],
                    cx=self._roi_widget._rois[2]["cx"],
                    radius=self._roi_widget._rois[2]["radius"],
                )
                roi_rib = ROICircle(
                    cy=self._roi_widget._rois[3]["cy"],
                    cx=self._roi_widget._rois[3]["cx"],
                    radius=self._roi_widget._rois[3]["radius"],
                )
            else:
                # Estimación automática relativa al corazón.
                roi_sternum = ROICircle(
                    cy=max(roi_h.cy - roi_h.radius * 1.8, roi_h.radius),
                    cx=roi_h.cx,
                    radius=roi_h.radius * 0.6,
                )
                roi_rib = ROICircle(
                    cy=roi_h.cy + roi_h.radius * 0.5,
                    cx=max(roi_h.cx - roi_h.radius * 2.0, roi_h.radius),
                    radius=roi_h.radius * 0.5,
                )
            q_result = compute_q_bone(self._original_image, roi_sternum, roi_rib)
            q_bone_val = q_result.q_bone
        except Exception:
            pass

        self._washout_data[time_label] = {
            "hmr": result.hmr,
            "hmr_raw": raw_result.hmr,
            "heart_counts": result.heart_counts,
            "mediastinum_counts": result.mediastinum_counts,
            "classification": result.classification,
            "q_bone": q_bone_val,
            "q_bone_mode": self._qbone_mode_by_time.get(time_label, "auto"),
            "exclude_bone": dict(corr_meta),
        }
        self._perugini_by_time[time_label] = int(self._perugini_combo.currentData())
        self._perugini_confirmed_by_time[time_label] = bool(self._perugini_confirm_chk.isChecked())
        self._roi_state[time_label] = [dict(roi) for roi in self._roi_widget._rois]
        self._update_kinetic_analysis()
        self._persist_user_state()

        # Renderizar imagen con ROIs dibujados + leyenda al pie.
        report_img = self.get_report_image()
        from PIL import Image as PILImage, ImageDraw, ImageFont
        pil = PILImage.fromarray(report_img).convert("RGB")
        draw = ImageDraw.Draw(pil)
        w_img, h_img = pil.size

        # Fuentes.
        try:
            font8 = ImageFont.truetype("arial.ttf", 8)
            font12 = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font8 = ImageFont.load_default()
            font12 = font8

        # ── Leyenda al pie: ROI labels + HMR ───────────────────────
        # Fondo negro en la franja inferior (30px).
        band_h = 30
        draw.rectangle([0, h_img - band_h, w_img, h_img], fill=(15, 23, 42))

        # ROI labels a la izquierda, font 8.
        roi_labels = [
            ("Corazón", (255, 102, 102)),   # rojo
            ("Mediastino", (56, 189, 248)),  # azul
        ]
        x_pos = 8
        for label, color in roi_labels:
            # Cuadradito de color + nombre.
            draw.rectangle([x_pos, h_img - band_h + 6, x_pos + 8, h_img - band_h + 14], fill=color)
            draw.text((x_pos + 12, h_img - band_h + 4), label, fill=color, font=font8)
            x_pos += len(label) * 6 + 30

        # HMR a la derecha, font 12, color según resultado.
        hmr_color = (248, 113, 113) if result.hmr >= 1.5 else ((251, 191, 36) if result.hmr >= 1.0 else (74, 222, 128))
        hmr_text = f"HMR = {result.hmr:.2f}"
        bbox = draw.textbbox((0, 0), hmr_text, font=font12)
        tw = bbox[2] - bbox[0]
        draw.text((w_img - tw - 10, h_img - band_h + 6), hmr_text, fill=hmr_color, font=font12)

        img_rgb = np.array(pil, dtype=np.float64)
        self._processed_images[time_label]["roi"] = img_rgb
        self._processed_images[time_label]["clean"] = self._original_image.copy()
        self._processed_images[time_label]["corr"] = np.asarray(quant_img, dtype=np.float64).copy()
        self._processed_images[time_label]["corr_meta"] = dict(corr_meta)
        self._rebuild_layout(force_layout=self._current_layout_n)
        layout = self._quadrant_viewer._layout
        roi_idx = self._roi_slot_index(time_label)
        if layout is not None and roi_idx < len(layout.quadrants):
            layout.quadrants[roi_idx].label = f"AP + ROIs ({time_label}, HMR={result.hmr:.2f})"
            layout.quadrants[roi_idx].hmr = result.hmr
            layout.quadrants[roi_idx].roi_overlay = True
            self._quadrant_viewer._rebuild_pixmaps()
            self._quadrant_viewer.update()
        self._update_washout_preview()
        corr_used = bool(
            corr_meta.get("rib_filter_used")
            or corr_meta.get("sternum_filter_used")
            or corr_meta.get("scatter_used")
        )
        notes = []
        if corr_meta.get("rib_filter_note"):
            notes.append(str(corr_meta.get("rib_filter_note")))
        if corr_meta.get("sternum_filter_note"):
            notes.append(str(corr_meta.get("sternum_filter_note")))
        if corr_used:
            QMessageBox.information(
                self,
                "SINCRO — Corrección experimental AP",
                f"HMR raw={raw_result.hmr:.2f} · HMR corregido={result.hmr:.2f}\n"
                f"{' | '.join(notes)}",
            )

        # Volver al modo visor.
        if self._current_mode == "analisis":
            self._toggle_mode()

    def get_report_image(self) -> np.ndarray:
        """Renderiza la imagen con los ROIs como array RGB para el informe."""
        if self._original_image is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        img = self._original_image.copy()
        h, w = img.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        norm = img / max(float(img.max()), 1e-8) if img.size else img
        rgb[..., 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
        rgb[..., 1] = rgb[..., 0]
        rgb[..., 2] = rgb[..., 0]
        from PIL import Image, ImageDraw
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        mode = self._qbone_mode_by_time.get(self._active_time if self._active_time in ("1h", "3h") else "1h", "auto")
        for idx, roi in enumerate(self._roi_widget._rois):
            if idx >= 2 and mode != "manual":
                continue
            color = roi["color"]
            x0 = int(roi["cx"] - roi["radius"])
            y0 = int(roi["cy"] - roi["radius"])
            x1 = int(roi["cx"] + roi["radius"])
            y1 = int(roi["cy"] + roi["radius"])
            draw.ellipse([x0, y0, x1, y1], outline=color, width=2)
        return np.asarray(pil)

    def _update_hmr(self, roi_id: int, cy: float, cx: float, radius: float):
        if self._original_image is None:
            self._lbl_hmr.setText("HMR = N/D")
            self._lbl_class.setText("Seleccioná una imagen AP para analizar.")
            return
        try:
            roi_h = ROICircle(
                cy=self._roi_widget._rois[0]["cy"],
                cx=self._roi_widget._rois[0]["cx"],
                radius=self._roi_widget._rois[0]["radius"],
            )
            roi_m = ROICircle(
                cy=self._roi_widget._rois[1]["cy"],
                cx=self._roi_widget._rois[1]["cx"],
                radius=self._roi_widget._rois[1]["radius"],
            )
            raw_result = compute_hmr(self._original_image, roi_h, roi_m)
            if self._active_time in ("1h", "3h"):
                qimg, corr_meta = self._build_ap_quant_image(self._active_time, roi_h)
                result = compute_hmr(qimg, roi_h, roi_m)
            else:
                corr_meta = {
                    "rib_filter_used": False,
                    "sternum_filter_used": False,
                    "scatter_used": False,
                    "rib_filter_note": "",
                    "sternum_filter_note": "",
                }
                result = raw_result
            corr_used = bool(
                corr_meta.get("rib_filter_used")
                or corr_meta.get("sternum_filter_used")
                or corr_meta.get("scatter_used")
            )
            notes = []
            if corr_meta.get("rib_filter_note"):
                notes.append(str(corr_meta.get("rib_filter_note")))
            if corr_meta.get("sternum_filter_note"):
                notes.append(str(corr_meta.get("sternum_filter_note")))
            if corr_used:
                self._lbl_hmr.setText(f"HMR = {result.hmr:.2f} (raw {raw_result.hmr:.2f})")
                self._lbl_class.setText(result.classification + (f" · {' | '.join(notes)}" if notes else ""))
            else:
                self._lbl_hmr.setText(f"HMR = {result.hmr:.2f}")
                self._lbl_class.setText(result.classification)
            cls_txt = str(result.classification or "").upper()
            if "POSITIVO" in cls_txt:
                hmr_color = "#fca5a5"
                cls_color = "#fca5a5"
            elif "EQU" in cls_txt:
                hmr_color = "#fde68a"
                cls_color = "#fde68a"
            else:
                hmr_color = "#86efac"
                cls_color = "#86efac"
            self._lbl_hmr.setStyleSheet(
                f"font-size:16px; font-weight:700; color:{hmr_color}; background:#000000; padding:4px 8px;"
            )
            self._lbl_class.setStyleSheet(
                f"font-size:12px; font-weight:600; color:{cls_color}; background:#000000; padding:3px 8px;"
            )
            self._update_filter_summary()
        except Exception as exc:
            self._lbl_hmr.setText("HMR = N/D")
            self._lbl_class.setText(f"Error: {exc}")
            self._lbl_hmr.setStyleSheet(
                "font-size:16px; font-weight:700; color:#ffffff; background:#000000; padding:4px 8px;"
            )
            self._lbl_class.setStyleSheet(
                "font-size:12px; font-weight:600; color:#fca5a5; background:#000000; padding:3px 8px;"
            )
            self._update_filter_summary()

    def _reset_rois(self):
        if self._original_image is None:
            return
        h, w = self._original_image.shape
        self._roi_widget._rois[0]["cy"] = 0.4 * h
        self._roi_widget._rois[0]["cx"] = 0.4 * w
        self._roi_widget._rois[0]["radius"] = 12.0
        self._roi_widget._rois[1]["cy"] = 0.6 * h
        self._roi_widget._rois[1]["cx"] = 0.6 * w
        self._roi_widget._rois[1]["radius"] = 12.0
        self._ensure_aux_rois()
        self._roi_widget._rois[2]["cy"] = 0.25 * h
        self._roi_widget._rois[2]["cx"] = 0.50 * w
        self._roi_widget._rois[2]["radius"] = 7.2
        self._roi_widget._rois[3]["cy"] = 0.60 * h
        self._roi_widget._rois[3]["cx"] = 0.20 * w
        self._roi_widget._rois[3]["radius"] = 6.0
        self._roi_widget._rois[4]["cy"] = 0.25 * h
        self._roi_widget._rois[4]["cx"] = 0.42 * w
        self._roi_widget._rois[4]["radius"] = 5.4
        self._roi_widget._rois[5]["cy"] = 0.25 * h
        self._roi_widget._rois[5]["cx"] = 0.58 * w
        self._roi_widget._rois[5]["radius"] = 5.4
        self._roi_widget.update()
        if self._active_time in self._roi_state:
            self._roi_state[self._active_time] = [dict(roi) for roi in self._roi_widget._rois]
        self._persist_user_state()
        self._update_hmr(0, 0, 0, 0)

    def _on_visual_filter_changed(self, idx: int):
        """Aplica/quita filtro visual al widget de ROIs (solo display, no raw)."""
        from core.amyloid_planar import apply_visual_filter, VISUAL_FILTERS
        filter_key = self._filter_combo.currentData()
        display_base = self._build_current_display_image()
        if filter_key == "early_dynamic":
            if self._early_dynamic is None:
                QMessageBox.information(
                    self,
                    "SINCRO — Cinética experimental",
                    "Cargá primero un dinámico temprano. El dinámico se usa solo como guía visual; HMR permanece raw.",
                )
                self._filter_combo.setCurrentIndex(0)
                return
            guide = np.asarray(self._early_dynamic["summed_cps"], dtype=np.float64)
            if self._original_image is not None and guide.shape != self._original_image.shape:
                QMessageBox.warning(
                    self,
                    "SINCRO — Cinética experimental",
                    "La matriz del dinámico no coincide con la planar tardía. Se requiere registro espacial antes de usarla como guía.",
                )
                self._filter_combo.setCurrentIndex(0)
                return
            self._roi_widget._image = guide
            self._roi_widget.update()
            return
        if filter_key is None or filter_key == "none":
            # Restaurar imagen raw.
            if display_base is not None:
                self._roi_widget._image = np.asarray(display_base, dtype=np.float64)
                self._roi_widget.update()
            return
        if display_base is None:
            return
        _, kwargs = VISUAL_FILTERS.get(filter_key, ("", {}))
        try:
            filtered = apply_visual_filter(np.asarray(display_base, dtype=np.float64), filter_key, **kwargs)
            self._roi_widget._image = np.asarray(filtered, dtype=np.float64)
            self._roi_widget.update()
        except Exception:
            pass  # Si falla el filtro, dejar la imagen raw.

    def _export_png_jpg(self):
        """Exporta ROI actual o layout completo a PNG/JPG."""
        from PIL import Image

        choice, ok = QInputDialog.getItem(
            self,
            "SINCRO — Exportar",
            "¿Qué deseas exportar?",
            ["Layout completo", "ROI actual"],
            0,
            False,
        )
        if not ok:
            return

        options = "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Exportar imagen",
            os.path.expanduser("~"),
            options,
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if not ext:
            ext = ".png" if "PNG" in selected.upper() else ".jpg"
            path = path + ext

        if choice == "ROI actual":
            img = self.get_report_image()
        else:
            img = self.get_layout_composite_image()
        if img is None or img.size == 0:
            if choice == "Layout completo":
                QMessageBox.warning(self, "SINCRO — Exportar", "No hay layout para exportar. Cargá imágenes primero.")
                return
            QMessageBox.warning(self, "SINCRO — Exportar", "No hay imagen para exportar.")
            return

        try:
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            if ext in (".jpg", ".jpeg"):
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                pil.save(path, "JPEG", quality=95)
            else:
                pil.save(path, "PNG")
            QMessageBox.information(self, "SINCRO — Exportar", f"Imagen exportada:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO — Exportar", f"No se pudo exportar:\n{exc}")

    def _update_kinetic_analysis(self):
        """Calcula métricas temporales cuando están disponibles los tres tiempos."""
        if self._early_dynamic is None:
            self._kinetic_status.setText("Cinética experimental: dinámico temprano no cargado.")
            return
        ap_1h = self._time_images["1h"]["ap"]
        ap_3h = self._time_images["3h"]["ap"]
        if ap_1h is None or ap_3h is None:
            self._kinetic_status.setText("Dinámico cargado. Faltan AP de 1h y/o 3h para análisis temporal.")
            return
        duration_1h = ap_1h.get("duration_s")
        duration_3h = ap_3h.get("duration_s")
        if not duration_1h or not duration_3h:
            self._kinetic_status.setText(
                "Cinética pendiente: los DICOM tardíos no informan duración. "
                "No se comparan cuentas brutas de adquisiciones con distinta duración."
            )
            return
        roi_state = self._roi_state.get("1h") or self._roi_state.get("3h")
        if not roi_state:
            self._kinetic_status.setText("Cinética pendiente: cuantificá una AP para definir ROIs cardíaco y mediastinal.")
            return
        try:
            from core.amyloid_kinetic import normalize_static_image, temporal_metrics

            heart = ROICircle(**{k: roi_state[0][k] for k in ("cy", "cx", "radius")})
            medi = ROICircle(**{k: roi_state[1][k] for k in ("cy", "cx", "radius")})
            image_1h = normalize_static_image(
                np.asarray(ap_1h["image"], dtype=np.float64), duration_1h, 60.0, decay_correct=True
            )
            image_3h = normalize_static_image(
                np.asarray(ap_3h["image"], dtype=np.float64), duration_3h, 180.0, decay_correct=True
            )
            metrics = temporal_metrics(self._early_dynamic["dynamic"], image_1h, image_3h, heart, medi)
            self._kinetic_result = metrics
            self._kinetic_status.setText(
                "EXPERIMENTAL — "
                f"Retención cardíaca 3h/1h: {metrics.heart_retention_3h_over_1h:.2f}; "
                f"cambio: {metrics.heart_change_pct:+.1f}%; ΔHMR: {metrics.hmr_change:+.2f}. "
                "No subtipifica ATTR/AL ni guía tratamiento."
            )
            self._kinetic_status.setStyleSheet(
                "color:#38bdf8; border:1px solid #334155; padding:4px; font-size:10px;"
            )
        except Exception as exc:
            self._kinetic_result = None
            self._kinetic_status.setText(f"Cinética experimental no disponible: {exc}")

    def _update_washout_preview(self):
        """Actualiza estado y curva en vivo solo con 1 h y 3 h cuantificadas."""
        done = [time for time in ("1h", "3h") if time in self._washout_data]
        missing = [time for time in ("1h", "3h") if time not in self._washout_data]
        if missing:
            done_text = ", ".join(done) if done else "ninguno"
            missing_text = " y ".join(missing)
            status = f"Cuantificado: {done_text}. Falta cuantificar {missing_text}."
            self._lbl_washout_status.setText(status)
            self._lbl_washout_status.setStyleSheet(
                "font-size:10px; color:#000000; background:#e5e7eb; padding:6px 10px; border:1px solid #d1d5db; border-radius:999px;"
            )
            self._washout_preview.clear()
            self._washout_preview.setText(status)
            self._washout_preview.setVisible(True)
            return
        curve_b64 = self._generate_washout_curve_b64()
        pixmap = QPixmap()
        if curve_b64 and pixmap.loadFromData(base64.b64decode(curve_b64), "PNG"):
            self._washout_preview.setPixmap(pixmap.scaledToWidth(145, Qt.TransformationMode.SmoothTransformation))
            self._washout_preview.setVisible(True)
        oai_txt = ""
        corr_txt = ""
        for t in ("1h", "3h"):
            data = self._washout_data.get(t) or {}
            eb = dict(data.get("exclude_bone") or {})
            if eb.get("rib_filter_used") or eb.get("sternum_filter_used") or eb.get("scatter_used"):
                raw = data.get("hmr_raw")
                cur = data.get("hmr")
                if raw is not None and cur is not None:
                    corr_txt += f" · {t}: raw {raw:.2f}→corr {cur:.2f}"
        if "1h" in self._oai_washout_data and "3h" in self._oai_washout_data:
            h1 = float(self._oai_washout_data["1h"].get("heart_counts", 0.0))
            h3 = float(self._oai_washout_data["3h"].get("heart_counts", 0.0))
            if h1 > 0:
                ret = h3 / h1
                wo = (1.0 - ret) * 100.0
                oai_txt = f" · OAI opcional 3h/1h={ret:.3f} (washout {wo:+.1f}%)"
        self._lbl_washout_status.setText("Washout 1h/3h cuantificado: curva lista para el informe." + oai_txt + corr_txt)
        self._lbl_washout_status.setStyleSheet(
            "font-size:10px; color:#000000; background:#bfdbfe; padding:6px 10px; border:1px solid #93c5fd; border-radius:999px;"
        )

    def _open_report_template_preview(self):
        """Abre vista previa visual de plantillas de informe y permite selección."""
        ctx = self._build_report_context()
        dlg = QDialog(self)
        dlg.setWindowTitle("SINCRO — Preview de plantillas AMILO")
        dlg.resize(980, 680)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        info = QLabel(
            "Elegí la plantilla visual. 'Auto' selecciona según cobertura del estudio "
            f"(actual: <b>{ctx.get('template_name')}</b>)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#cbd5e1;")
        lay.addWidget(info)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Plantilla:"))
        combo = QComboBox()
        combo.addItem("Auto", "auto")
        combo.addItem("AMILO Clínico Completo", "AMILO Clínico Completo")
        combo.addItem("AMILO Planar", "AMILO Planar")
        combo.addItem("AMILO SPECT", "AMILO SPECT")
        combo.addItem("AMILO Básico", "AMILO Básico")
        current_tpl = str(self._report_template_combo.currentData() or "auto")
        idx = combo.findData(current_tpl)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        selector_row.addWidget(combo)
        selector_row.addStretch(1)
        lay.addLayout(selector_row)

        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setStyleSheet("background:#0f172a; border:1px solid #334155;")
        preview_label.setMinimumHeight(500)
        lay.addWidget(preview_label, 1)

        def _template_display_name(val: str) -> str:
            if val == "auto":
                return str(ctx.get("template_name") or "AMILO Básico")
            return val

        def _draw_preview(template_name: str):
            w, h = 900, 520
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[:, :, :] = np.array([15, 23, 42], dtype=np.uint8)

            def fill(x0, y0, x1, y1, rgb):
                x0 = max(0, min(w - 1, int(x0)))
                x1 = max(0, min(w, int(x1)))
                y0 = max(0, min(h - 1, int(y0)))
                y1 = max(0, min(h, int(y1)))
                canvas[y0:y1, x0:x1, :] = np.array(rgb, dtype=np.uint8)

            fill(20, 20, w - 20, 80, (26, 58, 92))  # header

            if template_name == "AMILO Clínico Completo":
                fill(20, 95, 430, 320, (30, 41, 59))
                fill(450, 95, w - 20, 320, (30, 41, 59))
                fill(20, 335, w - 20, 430, (30, 41, 59))
                fill(20, 445, w - 20, h - 20, (22, 78, 99))
            elif template_name == "AMILO Planar":
                fill(20, 95, 510, 360, (30, 41, 59))
                fill(530, 95, w - 20, 360, (30, 41, 59))
                fill(20, 375, w - 20, h - 20, (22, 78, 99))
            elif template_name == "AMILO SPECT":
                fill(20, 95, w - 20, 320, (30, 41, 59))
                fill(20, 335, w - 20, 430, (30, 41, 59))
                fill(20, 445, w - 20, h - 20, (22, 78, 99))
            else:  # AMILO Básico
                fill(20, 95, w - 20, 300, (30, 41, 59))
                fill(20, 315, w - 20, 415, (30, 41, 59))
                fill(20, 430, w - 20, h - 20, (22, 78, 99))

            qimg = QImage(canvas.data, w, h, canvas.strides[0], QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg.copy())
            preview_label.setPixmap(
                pix.scaled(preview_label.width(), preview_label.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )

        def _refresh_preview():
            display_tpl = _template_display_name(str(combo.currentData() or "auto"))
            _draw_preview(display_tpl)

        combo.currentIndexChanged.connect(lambda _=0: _refresh_preview())
        _refresh_preview()

        btns = QHBoxLayout()
        btns.addStretch(1)
        btn_apply = QPushButton("Aplicar selección")
        btn_cancel = QPushButton("Cancelar")
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_apply)
        lay.addLayout(btns)

        btn_cancel.clicked.connect(dlg.reject)

        def _apply_and_close():
            selected = str(combo.currentData() or "auto")
            idx_local = self._report_template_combo.findData(selected)
            if idx_local >= 0:
                self._report_template_combo.setCurrentIndex(idx_local)
                self._persist_user_state()
            dlg.accept()

        btn_apply.clicked.connect(_apply_and_close)
        dlg.exec()

    def _open_report_template_matrix(self):
        """Muestra matriz comparativa de escenarios vs plantilla automática y contenido."""
        dlg = QDialog(self)
        dlg.setWindowTitle("SINCRO — Matriz de escenarios AMILO")
        dlg.resize(980, 620)
        lay = QVBoxLayout(dlg)

        info = QLabel(
            "Matriz de referencia para entender qué plantilla usa AUTO y qué secciones se incluyen/omiten "
            "según cobertura del estudio."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#cbd5e1; font-size:12px;")
        lay.addWidget(info)

        scenarios = [
            ("Solo planar", True, False, False),
            ("Planar + SPECT", True, True, False),
            ("Planar + SPECT/CT", True, True, True),
            ("Solo SPECT", False, True, False),
            ("Solo SPECT/CT", False, True, True),
        ]

        rows_html = []
        for name, has_planar, has_spect, has_ct in scenarios:
            if has_planar and has_spect:
                tpl = "AMILO Clínico Completo"
            elif has_planar:
                tpl = "AMILO Planar"
            elif has_spect:
                tpl = "AMILO SPECT"
            else:
                tpl = "AMILO Básico"
            profile = self._template_profile(tpl, has_planar, has_spect, has_ct)
            includes = "<br>• " + "<br>• ".join(str(x) for x in profile.get("includes", [])) if profile.get("includes") else "—"
            omits = "<br>• " + "<br>• ".join(str(x) for x in profile.get("omits", [])) if profile.get("omits") else "—"
            rows_html.append(
                f"""
                <tr>
                  <td><b>{name}</b></td>
                  <td>{'Sí' if has_planar else 'No'}</td>
                  <td>{'Sí' if has_spect else 'No'}</td>
                  <td>{'Sí' if has_ct else 'No'}</td>
                  <td><b>{tpl}</b><br><span style='color:#94a3b8'>{profile.get('focus','')}</span></td>
                  <td>{includes}</td>
                  <td>{omits}</td>
                </tr>
                """
            )

        html = f"""
        <html><body style='background:#0f172a; color:#e2e8f0; font-family:Segoe UI, Arial;'>
        <table style='width:100%; border-collapse:collapse; font-size:12px;'>
          <thead>
            <tr style='background:#1e293b;'>
              <th style='border:1px solid #334155; padding:8px;'>Escenario</th>
              <th style='border:1px solid #334155; padding:8px;'>Planar</th>
              <th style='border:1px solid #334155; padding:8px;'>SPECT</th>
              <th style='border:1px solid #334155; padding:8px;'>CT</th>
              <th style='border:1px solid #334155; padding:8px;'>AUTO</th>
              <th style='border:1px solid #334155; padding:8px;'>Incluye</th>
              <th style='border:1px solid #334155; padding:8px;'>Omite</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
        <p style='margin-top:10px; color:#94a3b8;'>
          Nota: si el usuario selecciona una plantilla manual (no AUTO), esa selección fuerza el perfil.
        </p>
        </body></html>
        """

        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml(html)
        text.setStyleSheet("background:#0b1220; border:1px solid #334155; color:#e2e8f0;")
        lay.addWidget(text, 1)

        row_btn = QHBoxLayout()
        row_btn.addStretch(1)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(dlg.accept)
        row_btn.addWidget(btn_close)
        lay.addLayout(row_btn)

        dlg.exec()

    def _build_report_context(self) -> dict:
        """Resume la cobertura de datos para elegir plantilla de informe."""
        has_planar_loaded = any(
            self._time_images.get(t, {}).get("ap") is not None for t in ("1h", "3h")
        ) or (self._original_image is not None)
        has_planar_metrics = bool(self._washout_data)
        bridge = dict(self._linked_spect_ct or {})
        workflow = str(bridge.get("workflow_tag") or "").strip().lower()
        has_spect = bool(bridge) and workflow in ("perf_spect_ct", "amylo")
        has_ct = bool(str(bridge.get("ct_path") or "").strip())
        
        # Intentar obtener HMR-SPECT del bridge (si se calculó en AmyloidSpectPanel)
        hmr_spect = None
        hmr_spect_raw = None
        if has_spect:
            hmr_spect_data = bridge.get("hmr_spect") or {}
            hmr_spect = hmr_spect_data.get("hmr")
            hmr_spect_raw = hmr_spect_data.get("hmr_raw")

        auto_template = ""
        if has_planar_loaded and has_spect:
            template_name = "AMILO Clínico Completo"
        elif has_planar_loaded:
            template_name = "AMILO Planar"
        elif has_spect:
            template_name = "AMILO SPECT"
        else:
            template_name = "AMILO Básico"

        auto_template = template_name
        selected_template = str(self._report_template_combo.currentData() or "auto") if hasattr(self, "_report_template_combo") else "auto"
        if selected_template != "auto":
            template_name = selected_template

        al_status_code = str(self._al_status_combo.currentData() or "pending") if hasattr(self, "_al_status_combo") else "pending"
        flc_code = str(self._free_light_chain_combo.currentData() or "unknown") if hasattr(self, "_free_light_chain_combo") else "unknown"
        ifx_code = str(self._immunofix_combo.currentData() or "unknown") if hasattr(self, "_immunofix_combo") else "unknown"

        return {
            "template_name": template_name,
            "auto_template": auto_template,
            "selected_template": selected_template,
            "has_planar_loaded": has_planar_loaded,
            "has_planar_metrics": has_planar_metrics,
            "has_spect": has_spect,
            "has_ct": has_ct,
            "bridge": bridge,
            "hmr_spect": hmr_spect,
            "hmr_spect_raw": hmr_spect_raw,
            "al_status": self._al_status_text_from_code(al_status_code),
            "free_light_chain": self._flc_text_from_code(flc_code),
            "immunofixation": self._immunofix_text_from_code(ifx_code),
            "template_profile": self._template_profile(template_name, has_planar_loaded, has_spect, has_ct),
        }

    @staticmethod
    def _template_profile(template_name: str, has_planar_loaded: bool, has_spect: bool, has_ct: bool) -> dict:
        """Describe perfil visual/funcional de plantilla para hacerla explícita en el informe."""
        if template_name == "AMILO Clínico Completo":
            return {
                "slug": "completo",
                "accent": "#2563eb",
                "focus": "Integración multimodal planar + tomográfica",
                "includes": [
                    "Cobertura de estudio (Planar/SPECT/CT)",
                    "Métricas planares (HMR/Perugini/Q_bone)",
                    "Gráficos avanzados (histograma/washout)",
                    "Métrica OAI opcional y cinética experimental (si disponible)",
                    "Layout compuesto de imágenes",
                ],
                "omits": [],
            }
        if template_name == "AMILO Planar":
            return {
                "slug": "planar",
                "accent": "#0ea5e9",
                "focus": "Cuantificación planar y evolución temporal",
                "includes": [
                    "Métricas planares (HMR/Perugini/Q_bone)",
                    "Gráficos avanzados (histograma/washout)",
                    "Interpretación orientada a planar",
                ],
                "omits": [
                    "Cobertura tomográfica detallada",
                    "Layout compuesto tomográfico",
                    "Bloques OAI/cinética experimental",
                ],
            }
        if template_name == "AMILO SPECT":
            modality = "SPECT/CT" if has_ct else ("SPECT" if has_spect else "Tomografía")
            return {
                "slug": "spect",
                "accent": "#7c3aed",
                "focus": f"Correlación topográfica {modality}",
                "includes": [
                    "Cobertura de estudio (Planar/SPECT/CT)",
                    "Layout compuesto de imágenes",
                    "Interpretación centrada en topografía miocardio vs blood pool",
                ],
                "omits": [
                    "Métricas planares detalladas HMR/Perugini",
                    "Gráficos de washout planar",
                    "Bloques OAI/cinética experimental",
                ],
            }
        return {
            "slug": "basico",
            "accent": "#64748b",
            "focus": "Resumen clínico mínimo y trazabilidad",
            "includes": [
                "Datos de estudio y bloque AL",
                "Cobertura resumida",
                "Interpretación breve",
            ],
            "omits": [
                "Detalle avanzado de métricas y gráficos",
                "Bloques experimentales",
            ],
        }

    def _select_report_output_dir(self):
        """Selecciona carpeta de salida para informes PDF/HTML."""
        start_dir = str(getattr(self, "_report_output_dir", "") or "")
        if not start_dir or not os.path.isdir(start_dir):
            start_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_demo")
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de salida para informe", start_dir)
        if not folder:
            return
        self._report_output_dir = str(folder)
        self._persist_user_state()

    def _generate_report(self):
        """Genera el informe PDF + HTML de amiloidosis."""
        report_ctx = self._build_report_context()
        has_planar_metrics = bool(report_ctx.get("has_planar_metrics"))
        has_spect = bool(report_ctx.get("has_spect"))

        if not has_planar_metrics and not has_spect:
            QMessageBox.warning(
                self, "SINCRO — Amyloidosis",
                "No hay datos suficientes para informe.\n"
                "Necesitás al menos:\n"
                "- Planar cuantificado (HMR), o\n"
                "- Sesión SPECT/CT vinculada."
            )
            return
        try:
            result = None
            perugini = None
            report_img = None
            if self._original_image is not None and has_planar_metrics:
                roi_h = ROICircle(
                    cy=self._roi_widget._rois[0]["cy"],
                    cx=self._roi_widget._rois[0]["cx"],
                    radius=self._roi_widget._rois[0]["radius"],
                )
                roi_m = ROICircle(
                    cy=self._roi_widget._rois[1]["cy"],
                    cx=self._roi_widget._rois[1]["cx"],
                    radius=self._roi_widget._rois[1]["radius"],
                )
                result = compute_hmr(self._original_image, roi_h, roi_m)
                perugini = int(self._perugini_combo.currentData())
                report_img = self.get_report_image()

            # Obtener imagen compuesta del layout.
            composite_img = self.get_layout_composite_image()
            output_dir = str(getattr(self, "_report_output_dir", "") or "")
            if not output_dir:
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_demo")
            os.makedirs(output_dir, exist_ok=True)
            # Guardar imagen con ROIs.
            img_path = ""
            if report_img is not None:
                img_path = os.path.join(output_dir, "amyloid_planar.png")
                from PIL import Image
                Image.fromarray(report_img).save(img_path, "PNG")
            # Guardar imagen compuesta del layout.
            composite_path = ""
            if composite_img is not None:
                composite_path = os.path.join(output_dir, "amyloid_layout_composite.png")
                Image.fromarray(composite_img.astype(np.uint8)).save(composite_path, "PNG")
            output_mode = str(self._report_output_combo.currentData() or "both")
            paths_out = []
            profile = dict(report_ctx.get("template_profile") or {})
            suffix = str(profile.get("slug") or "basico")

            # PDF
            pdf_path = os.path.join(output_dir, f"informe_amyloid_{suffix}.pdf")
            if output_mode in ("both", "pdf"):
                self._generate_pdf(pdf_path, img_path, composite_path, result, perugini, report_ctx)
                paths_out.append(f"PDF: {pdf_path}")

            # HTML
            html_path = os.path.join(output_dir, f"informe_amyloid_{suffix}.html")
            if output_mode in ("both", "html"):
                self._generate_html(html_path, img_path, composite_path, result, perugini, report_ctx)
                paths_out.append(f"HTML: {html_path}")

            QMessageBox.information(
                self, "SINCRO — Amyloidosis",
                "Informe generado:\n" + "\n".join(paths_out)
            )
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO — Amyloidosis", f"Error al generar informe:\n{exc}")

    def _generate_pdf(self, pdf_path, img_path, composite_path, result, perugini, report_ctx):
        """Genera el informe PDF de amiloidosis con bloques 1h/3h y gráficos."""
        import base64, io
        from PIL import Image as PILImage
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor, white, black
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.utils import ImageReader
        from datetime import datetime

        DARK_BLUE = HexColor("#1a3a5c")
        LIGHT_BLUE = HexColor("#e8f0f8")
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=20, textColor=DARK_BLUE)
        section_style = ParagraphStyle("SectionCustom", parent=styles["Heading2"], fontSize=12, textColor=DARK_BLUE)
        body_style = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=9.5, leading=13)
        small_style = ParagraphStyle("SmallCustom", parent=styles["Normal"], fontSize=8, textColor=HexColor("#666666"))

        doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
        story = []

        template_name = str(report_ctx.get("template_name") or "AMILO Básico")
        profile = dict(report_ctx.get("template_profile") or {})
        show_coverage = template_name in ("AMILO Clínico Completo", "AMILO SPECT", "AMILO Básico")
        show_planar_blocks = template_name in ("AMILO Clínico Completo", "AMILO Planar", "AMILO Básico")
        show_advanced_graphs = template_name in ("AMILO Clínico Completo", "AMILO Planar")
        show_oai_and_kinetic = template_name == "AMILO Clínico Completo"
        show_layout = template_name in ("AMILO Clínico Completo", "AMILO SPECT")
        story.append(Paragraph("SINCRO — Informe de Amiloidosis Cardíaca", title_style))
        story.append(Paragraph(f"Plantilla: {template_name}", small_style))
        story.append(Paragraph("Análisis de captación miocárdica con Tc-99m PYP/DPD/HMDP", small_style))
        focus = str(profile.get("focus") or "")
        includes = list(profile.get("includes") or [])
        omits = list(profile.get("omits") or [])
        if focus:
            story.append(Spacer(1, 1.2*mm))
            story.append(Paragraph(f"<b>Perfil:</b> {focus}", body_style))
        if includes:
            story.append(Paragraph("<b>Incluye:</b> " + " · ".join(includes), small_style))
        if omits:
            story.append(Paragraph("<b>Omite:</b> " + " · ".join(omits), small_style))
        story.append(Spacer(1, 2*mm))
        story.append(HRFlowable(width="100%", thickness=1.2, color=DARK_BLUE))
        story.append(Spacer(1, 4*mm))

        # Datos del paciente
        patient = self._metadata["patient"]
        date = self._metadata["date"]
        series = self._metadata["series"]
        info_data = [
            ["Paciente", patient],
            ["Fecha de estudio", date],
            ["Serie", series],
            ["Fecha de informe", datetime.now().strftime("%d/%m/%Y %H:%M")],
        ]
        info_table = Table(info_data, colWidths=[50*mm, 116*mm])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
            ("LEFTPADDING", (0, 0), (-1, -1), 3*mm),
        ]))
        story.append(Paragraph("1. Datos del estudio", section_style))
        story.append(info_table)
        story.append(Spacer(1, 4*mm))

        # Cobertura del estudio + bloque AL obligatorio.
        bridge = dict(report_ctx.get("bridge") or {})
        bridge_profile = dict(bridge.get("profile") or {})
        modality_rows = [
            ["Planar cargado", "Sí" if report_ctx.get("has_planar_loaded") else "No"],
            ["Planar cuantificado (HMR)", "Sí" if report_ctx.get("has_planar_metrics") else "No"],
            ["SPECT vinculado", "Sí" if report_ctx.get("has_spect") else "No"],
            ["CT vinculado", "Sí" if report_ctx.get("has_ct") else "No"],
            ["Workflow puente", str(bridge.get("workflow_tag") or "N/D")],
            ["Serie/Protocolo puente", str(bridge_profile.get("series_description") or bridge_profile.get("protocol") or "N/D")],
        ]
        if show_coverage:
            story.append(Paragraph("2. Cobertura de estudio", section_style))
            mod_table = Table(modality_rows, colWidths=[60*mm, 106*mm])
            mod_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 3*mm),
            ]))
            story.append(mod_table)
            story.append(Spacer(1, 3*mm))

        story.append(Paragraph("3. Exclusión de AL (obligatorio para confirmar ATTR-CM)", section_style))
        al_table = Table([
            ["Cadenas livianas libres", str(report_ctx.get("free_light_chain") or "No informado")],
            ["Inmunofijación sérica/urinaria", str(report_ctx.get("immunofixation") or "No informada")],
            ["Estado AL", str(report_ctx.get("al_status") or "PENDIENTE / NO INFORMADO")],
        ], colWidths=[60*mm, 106*mm])
        al_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
            ("LEFTPADDING", (0, 0), (-1, -1), 3*mm),
        ]))
        story.append(al_table)
        story.append(Paragraph(
            "La gammagrafía no debe usarse de forma aislada para confirmar ATTR-CM si AL no fue excluida.",
            small_style,
        ))
        story.append(Spacer(1, 4*mm))

        # Imagen con ROIs (solo si no hay bloques temporales)
        if show_planar_blocks and img_path and os.path.isfile(img_path) and not self._washout_data:
            story.append(Paragraph("2. Imagen planar con ROIs", section_style))
            img = ImageReader(img_path)
            iw, ih = img.getSize()
            scale = min(160*mm / iw, 120*mm / ih)
            story.append(RLImage(img_path, width=iw*scale, height=ih*scale))
            story.append(Spacer(1, 3*mm))

        # Bloques por tiempo (1h, 3h)
        sec_num = 2
        for time_label in ("1h", "3h"):
            if not show_planar_blocks:
                continue
            data = self._washout_data.get(time_label)
            if data is None:
                continue
            hmr = data["hmr"]
            heart = data["heart_counts"]
            medi = data["mediastinum_counts"]
            cls = data.get("classification", "")
            perugini_time = self._perugini_by_time.get(time_label, perugini)
            perugini_confirmed = bool(self._perugini_confirmed_by_time.get(time_label, False))

            story.append(Paragraph(f"{sec_num}. Resultados {time_label}", section_style))

            # Imagen ROI de este tiempo
            roi_img = self._processed_images.get(time_label, {}).get("roi")
            if roi_img is not None:
                roi_pil = PILImage.fromarray(np.asarray(roi_img, dtype=np.uint8))
                roi_buf = io.BytesIO()
                roi_pil.save(roi_buf, format="PNG")
                roi_buf.seek(0)
                roi_reader = ImageReader(roi_buf)
                riw, rih = roi_reader.getSize()
                roi_scale = min(120*mm / riw, 90*mm / rih)
                story.append(RLImage(roi_buf, width=riw*roi_scale, height=rih*roi_scale))
                story.append(Spacer(1, 2*mm))

            q_bone_val = data.get("q_bone")
            q_bone_text = f"{q_bone_val:.2f}" if q_bone_val is not None else "N/D"
            q_bone_mode = str(data.get("q_bone_mode", "auto"))
            q_bone_ref = "manual (ROI esternón/costilla)" if q_bone_mode == "manual" else "auto (estimado)"
            hmr_raw = data.get("hmr_raw")
            eb = dict(data.get("exclude_bone") or {})
            rib_used = bool(eb.get("rib_filter_used", False))
            sternum_used = bool(eb.get("sternum_filter_used", False))
            sc_used = bool(eb.get("scatter_used", False))
            rib_method = str(eb.get("rib_filter_method", "N/D"))
            rib_asym = eb.get("rib_filter_asym", None)
            sternum_asym = eb.get("sternum_filter_asym", None)
            rib_note = str(eb.get("rib_filter_note", "") or "")
            sternum_note = str(eb.get("sternum_filter_note", "") or "")
            hmr_data = [
                ["Métrica", "Valor", "Referencia"],
                [f"HMR ({time_label})", f"{hmr:.2f}", "≥1.5 sugiere ATTR"],
                [f"HMR raw ({time_label})", f"{float(hmr_raw):.2f}" if hmr_raw is not None else "N/D", "Sin correcciones experimentales"],
                ["Cuentas cardíacas", f"{heart:,.0f}", ""],
                ["Cuentas mediastinales", f"{medi:,.0f}", ""],
                ["Clasificación", cls, ""],
                ["Perugini", str(perugini_time), "0–3"],
                ["Perugini estado", "Final" if perugini_confirmed else "Sugerido", "Confirmación manual"],
                ["Q_bone (calidad ósea)", q_bone_text, f"≈1 homogéneo · {q_bone_ref}"],
                ["Filtro costal", "Sí" if rib_used else "No", rib_method if rib_used else "No aplicado"],
                ["Filtro esternón", "Sí" if sternum_used else "No", "2 fondos" if sternum_used else "No aplicado"],
                ["SCATTER planar", "Sí" if sc_used else "No", f"k={float(eb.get('scatter_k', 0.0)):.2f}" if sc_used else "No aplicado"],
                ["Asimetría costal", f"{float(rib_asym):.2f}" if rib_asym is not None else "N/D", "Se descarta corrección si supera umbral"],
                ["Asimetría fondos esternón", f"{float(sternum_asym):.2f}" if sternum_asym is not None else "N/D", "Se descarta corrección si supera umbral"],
            ]
            hmr_table = Table(hmr_data, colWidths=[50*mm, 40*mm, 76*mm])
            hmr_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 3*mm),
            ]))
            story.append(hmr_table)
            notes_txt = " | ".join([x for x in [rib_note, sternum_note] if x])
            if notes_txt:
                story.append(Paragraph(f"Nota corrección experimental: {notes_txt}", small_style))
            story.append(Spacer(1, 4*mm))
            sec_num += 1

        # Si no hubo bloques temporales, mostrar resultado único
        if show_planar_blocks and not self._washout_data and result is not None:
            story.append(Paragraph("3. Métrica principal: HMR", section_style))
            hmr_data = [
                ["Métrica", "Valor", "Referencia"],
                ["HMR", f"{result.hmr:.2f}", "≥1.5 sugiere ATTR"],
                ["Cuentas cardíacas", f"{result.heart_counts:,.0f}", ""],
                ["Cuentas mediastinales", f"{result.mediastinum_counts:,.0f}", ""],
                ["Clasificación", result.classification, ""],
                ["Perugini", str(perugini), "0–3"],
            ]
            hmr_table = Table(hmr_data, colWidths=[50*mm, 40*mm, 76*mm])
            hmr_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 3*mm),
            ]))
            story.append(hmr_table)
            story.append(Spacer(1, 3*mm))
            sec_num += 1

        # --- Comparación HMR Planar vs SPECT (si ambos disponibles) ---
        hmr_planar = None
        hmr_planar_raw = None
        if self._washout_data:
            # Tomar el último tiempo disponible como referencia planar
            for t in ("3h", "1h"):
                if t in self._washout_data:
                    hmr_planar = self._washout_data[t].get("hmr")
                    hmr_planar_raw = self._washout_data[t].get("hmr_raw")
                    break
        elif result is not None:
            hmr_planar = result.hmr
            hmr_planar_raw = getattr(result, "hmr_raw", None)

        hmr_spect = report_ctx.get("hmr_spect")
        hmr_spect_raw = report_ctx.get("hmr_spect_raw")

        if hmr_planar is not None and hmr_spect is not None:
            story.append(Paragraph(f"{sec_num}. Comparación HMR Planar vs SPECT", section_style))
            
            # Determinar clasificación para cada método
            def classify_hmr(hmr_val):
                if hmr_val >= 1.6:
                    return "POSITIVO", "#ef4444"
                elif hmr_val >= 1.5:
                    return "EQUIVOCO", "#f59e0b"
                else:
                    return "NEGATIVO", "#22c55e"
            
            planar_cls, planar_color = classify_hmr(hmr_planar)
            spect_cls, spect_color = classify_hmr(hmr_spect)
            
            comparison_data = [
                ["Método", "HMR", "HMR raw", "Clasificación"],
                ["Planar", f"{hmr_planar:.2f}", f"{hmr_planar_raw:.2f}" if hmr_planar_raw else "N/D", planar_cls],
                ["SPECT", f"{hmr_spect:.2f}", f"{hmr_spect_raw:.2f}" if hmr_spect_raw else "N/D", spect_cls],
            ]
            
            comparison_table = Table(comparison_data, colWidths=[40*mm, 35*mm, 35*mm, 56*mm])
            comparison_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 3*mm),
                ("TEXTCOLOR", (3, 1), (3, 1), HexColor(planar_color)),
                ("TEXTCOLOR", (3, 2), (3, 2), HexColor(spect_color)),
                ("FONTNAME", (3, 1), (3, -1), "Helvetica-Bold"),
            ]))
            story.append(comparison_table)
            
            # Nota explicativa
            diff = abs(hmr_spect - hmr_planar)
            story.append(Paragraph(
                f"Diferencia: {diff:.2f}. "
                f"El HMR-SPECT suele ser ligeramente mayor que el planar debido a mejor contraste "
                f"(eliminación de superposición de estructuras). "
                f"Ambos métodos son complementarios para confirmar ATTR-CM.",
                small_style,
            ))
            story.append(Spacer(1, 4*mm))
            sec_num += 1

        # Perugini visual strip
        perugini_strip_b64 = self._generate_perugini_strip_b64(perugini) if perugini is not None else ""
        if perugini_strip_b64:
            strip_buf = io.BytesIO(base64.b64decode(perugini_strip_b64))
            strip_reader = ImageReader(strip_buf)
            sw, sh = strip_reader.getSize()
            strip_scale = min(170*mm / sw, 25*mm / sh)
            story.append(Paragraph(f"{sec_num}. Escala Perugini visual", section_style))
            story.append(RLImage(strip_buf, width=sw*strip_scale, height=sh*strip_scale))
            story.append(Spacer(1, 3*mm))
            sec_num += 1

        # Histograma
        hist_b64 = self._generate_roi_histogram_b64() if show_advanced_graphs else ""
        if hist_b64:
            hist_buf = io.BytesIO(base64.b64decode(hist_b64))
            hist_reader = ImageReader(hist_buf)
            hw, hh = hist_reader.getSize()
            hist_scale = min(170*mm / hw, 60*mm / hh)
            story.append(Paragraph(f"{sec_num}. Distribución de cuentas ROI cardíaco", section_style))
            story.append(RLImage(hist_buf, width=hw*hist_scale, height=hh*hist_scale))
            story.append(Paragraph("La distribución de intensidades permite evaluar homogeneidad de captación. Cola derecha sugiere pool sanguíneo residual.", small_style))
            story.append(Spacer(1, 3*mm))
            sec_num += 1

        # Bar chart corazón vs mediastino
        comparison_b64 = self._generate_comparison_bar_b64() if show_advanced_graphs else None
        if comparison_b64:
            comp_buf = io.BytesIO(base64.b64decode(comparison_b64))
            comp_reader = ImageReader(comp_buf)
            compw, comph = comp_reader.getSize()
            comp_scale = min(170*mm / compw, 60*mm / comph)
            story.append(Paragraph(f"{sec_num}. Cuentas corazón vs mediastino", section_style))
            story.append(RLImage(comp_buf, width=compw*comp_scale, height=comph*comp_scale))
            story.append(Spacer(1, 3*mm))
            sec_num += 1

        # Curva de washout
        washout_b64 = self._generate_washout_curve_b64() if show_advanced_graphs else None
        if washout_b64:
            washout_buf = io.BytesIO(base64.b64decode(washout_b64))
            washout_reader = ImageReader(washout_buf)
            ww, wh = washout_reader.getSize()
            washout_scale = min(170*mm / ww, 75*mm / wh)
            story.append(Paragraph(f"{sec_num}. Curva de washout 1h vs 3h", section_style))
            story.append(RLImage(washout_buf, width=ww*washout_scale, height=wh*washout_scale))
            story.append(Spacer(1, 3*mm))
            sec_num += 1

        # Métrica opcional OAI (si ambos tiempos fueron cuantificados).
        if show_oai_and_kinetic and "1h" in self._oai_washout_data and "3h" in self._oai_washout_data:
            h1 = float(self._oai_washout_data["1h"].get("heart_counts", 0.0))
            h3 = float(self._oai_washout_data["3h"].get("heart_counts", 0.0))
            if h1 > 0:
                ret = h3 / h1
                wo = (1.0 - ret) * 100.0
                story.append(Paragraph(f"{sec_num}. Washout OAI opcional (ROI corazón)", section_style))
                oai_data = [
                    ["Métrica", "Valor", "Nota"],
                    ["Cuentas OAI 1h (norm.)", f"{h1:.4f}", "cps corregidas por decaimiento"],
                    ["Cuentas OAI 3h (norm.)", f"{h3:.4f}", "cps corregidas por decaimiento"],
                    ["Retención OAI 3h/1h", f"{ret:.3f}", "Opcional"],
                    ["Washout OAI", f"{wo:+.1f}%", "Opcional · no reemplaza HMR AP"],
                ]
                oai_table = Table(oai_data, colWidths=[55*mm, 35*mm, 76*mm])
                oai_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
                    ("TEXTCOLOR", (0, 0), (-1, 0), white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
                ]))
                story.append(oai_table)
                story.append(Paragraph(
                    "Métrica auxiliar. Interpretar junto con AP/HMR y SPECT. "
                    "No sustituye la confirmación topográfica miocardio vs blood pool.",
                    small_style,
                ))
                story.append(Spacer(1, 3*mm))
                sec_num += 1

        # Análisis temporal experimental
        if show_oai_and_kinetic and self._kinetic_result is not None:
            km = self._kinetic_result
            story.append(Paragraph(f"{sec_num}. Análisis temporal PYP — EXPERIMENTAL", section_style))
            kinetic_data = [
                ["Métrica observable", "Valor"],
                ["Retención cardíaca 3h/1h (cps corregidas)", f"{km.heart_retention_3h_over_1h:.2f}"],
                ["Cambio cardíaco 1h→3h", f"{km.heart_change_pct:+.1f}%"],
                ["HMR temporal 1h", f"{km.hmr_1h:.2f}"],
                ["HMR temporal 3h", f"{km.hmr_3h:.2f}"],
                ["ΔHMR", f"{km.hmr_change:+.2f}"],
            ]
            kinetic_table = Table(kinetic_data, colWidths=[110*mm, 55*mm])
            kinetic_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
            ]))
            story.append(kinetic_table)
            story.append(Paragraph(
                "MÉTODO EXPERIMENTAL, NO DIAGNÓSTICO Y NO TERAPÉUTICO. "
                "Describe retención/lavado en cps corregidas por duración y decaimiento; "
                "no subtipifica ATTR/AL ni reemplaza laboratorio, SPECT/CT o biopsia.",
                small_style,
            ))
            story.append(Spacer(1, 3*mm))
            sec_num += 1

        # Layout compuesto
        if show_layout and composite_path and os.path.isfile(composite_path):
            story.append(Paragraph(f"{sec_num}. Layout completo", section_style))
            composite = ImageReader(composite_path)
            cw, ch = composite.getSize()
            composite_scale = min(170*mm / cw, 110*mm / ch)
            story.append(RLImage(composite_path, width=cw*composite_scale, height=ch*composite_scale))
            story.append(Spacer(1, 3*mm))
            sec_num += 1

        # Interpretación
        story.append(Paragraph(f"{sec_num}. Interpretación clínica", section_style))
        if show_planar_blocks and "1h" in self._washout_data and "3h" in self._washout_data:
            h1 = self._washout_data["1h"]["hmr"]
            h3 = self._washout_data["3h"]["hmr"]
            interp = f"""
            HMR 1h: {h1:.2f} ({self._washout_data["1h"].get("classification", "")}). 
            HMR 3h: {h3:.2f} ({self._washout_data["3h"].get("classification", "")}).<br/><br/>
            Si el resultado es equívoco (HMR 1.0–1.5), considerar imagen SPECT/CT o repetir planar a 3 horas
            para descartar pool sanguíneo residual.<br/><br/>
            La interpretación debe integrarse con laboratorio (cadenas livianas libres, proteínas monoclonales)
            y contexto clínico. El Perugini score ≥2 en presencia de gammapatía monoclonal ausente confirma ATTR.
            """
        elif show_planar_blocks and result is not None:
            interp = f"""
            El estudio muestra HMR de {result.hmr:.2f}. <b>{result.classification}</b><br/><br/>
            Si el resultado es equívoco (HMR 1.0–1.5), considerar imagen SPECT/CT o repetir planar a 3 horas
            para descartar pool sanguíneo residual.<br/><br/>
            La interpretación debe integrarse con laboratorio (cadenas livianas libres, proteínas monoclonales)
            y contexto clínico. El Perugini score ≥2 en presencia de gammapatía monoclonal ausente confirma ATTR.
            """
        else:
            interp = """
            No se dispone de cuantificación planar (HMR) en este estudio.
            La interpretación se sustenta en disponibilidad tomográfica (SPECT/SPECT-CT) y contexto clínico.
            Es obligatorio completar exclusión de AL (cadenas livianas libres e inmunofijación) antes de confirmar ATTR-CM.
            """
        story.append(Paragraph(interp, body_style))
        story.append(Spacer(1, 4*mm))

        story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#9aa7b5")))
        story.append(Paragraph(
            "Informe generado por SINCRO — Análisis de amiloidosis cardíaca. Resultados orientativos para apoyo clínico.",
            ParagraphStyle("Disc", parent=small_style, alignment=1),
        ))
        doc.build(story)

    def _generate_washout_curve_b64(self) -> str | None:
        """Genera curva de washout (1h vs 3h) como imagen base64.
        Devuelve None si no hay datos de ambos tiempos."""
        if len(self._washout_data) < 2:
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io, base64

        def _time_hours(label: str) -> float:
            if label in self._time_hours_by_label:
                try:
                    return float(self._time_hours_by_label[label])
                except Exception:
                    pass
            try:
                return float(label.replace("h", "").strip())
            except Exception:
                return 0.0

        sorted_items = sorted(self._washout_data.items(), key=lambda kv: _time_hours(kv[0]))
        times = [_time_hours(k) for k, _ in sorted_items]
        hmr_vals = [v["hmr"] for _, v in sorted_items]
        heart_vals = [v["heart_counts"] for _, v in sorted_items]
        medi_vals = [v["mediastinum_counts"] for _, v in sorted_items]

        h0 = heart_vals[0] if heart_vals[0] > 0 else 1.0
        m0 = medi_vals[0] if medi_vals[0] > 0 else 1.0
        heart_pct = [v / h0 * 100 for v in heart_vals]
        medi_pct = [v / m0 * 100 for v in medi_vals]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5), dpi=100)
        fig.patch.set_facecolor("#0f172a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#e2e8f0", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#475569")

        ax1.plot(times, hmr_vals, "o-", color="#38bdf8", linewidth=2, markersize=8)
        ax1.axhline(y=1.5, color="#f87171", linestyle="--", linewidth=1, alpha=0.7, label="Corte ATTR (1.5)")
        ax1.axhline(y=1.0, color="#4ade80", linestyle="--", linewidth=1, alpha=0.7, label="Corte negativo (1.0)")
        ax1.set_xlabel("Tiempo (horas)", color="#94a3b8", fontsize=8)
        ax1.set_ylabel("HMR", color="#94a3b8", fontsize=8)
        ax1.set_title("HMR vs Tiempo", color="#e2e8f0", fontsize=10, fontweight="bold")
        ax1.legend(fontsize=6, facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0")
        for t, h in zip(times, hmr_vals):
            ax1.annotate(f"{h:.2f}", (t, h), textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8, color="#e2e8f0")

        ax2.plot(times, heart_pct, "o-", color="#f87171", linewidth=2, markersize=8, label="Corazon")
        ax2.plot(times, medi_pct, "o-", color="#38bdf8", linewidth=2, markersize=8, label="Mediastino")
        ax2.set_xlabel("Tiempo (horas)", color="#94a3b8", fontsize=8)
        ax2.set_ylabel("Cuentas (% del inicial)", color="#94a3b8", fontsize=8)
        ax2.set_title("Washout de cuentas", color="#e2e8f0", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=7, facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0")

        if len(hmr_vals) >= 2:
            delta_hmr = hmr_vals[-1] - hmr_vals[0]
            delta_pct = heart_pct[-1] - heart_pct[0]
            washout_class = "ATTR (washout rapido)" if delta_hmr > 0.2 else ("AL (washout lento)" if delta_hmr < -0.1 else "Indeterminado")
            fig.suptitle(f"Washout: {delta_pct:+.1f}% cuentas corazon - {washout_class}",
                         color="#fbbf24", fontsize=9, y=0.02)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _generate_hmr_bar_b64(self, hmr_value: float) -> str:
        """Genera barra de referencia HMR como imagen base64.

        Zonas: verde (<1.0), amarillo (1.0-1.5), rojo (>=1.5).
        Un marcador vertical indica el valor del paciente.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.transforms import blended_transform_factory
        import io, base64

        fig, ax = plt.subplots(figsize=(6, 1.2), dpi=100)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')

        # Rango total de la barra.
        x_min, x_max = 0.0, 3.0
        bar_height = 0.55
        y_center = 0.5

        # Zonas coloreadas.
        zones = [
            (x_min, 1.0, '#4ade80', 'NEGATIVO\n(<1.0)'),
            (1.0, 1.5, '#fbbf24', 'EQUÍVOCO\n(1.0–1.5)'),
            (1.5, x_max, '#f87171', 'POSITIVO\n(≥1.5)'),
        ]
        for zx0, zx1, color, label in zones:
            ax.barh(
                y_center, zx1 - zx0, left=zx0, height=bar_height,
                color=color, edgecolor='none', alpha=0.85,
            )
            # Etiqueta centrada en cada zona.
            ax.text(
                (zx0 + zx1) / 2, y_center, label,
                ha='center', va='center', fontsize=7, fontweight='bold',
                color='#0f172a',
            )

        # Marcador del paciente.
        hmr_clamped = max(x_min, min(x_max, hmr_value))
        ax.annotate(
            '',
            xy=(hmr_clamped, y_center + bar_height / 2 + 0.02),
            xytext=(hmr_clamped, y_center + bar_height / 2 + 0.32),
            arrowprops=dict(arrowstyle='->', color='white', lw=2.5),
        )
        ax.text(
            hmr_clamped, y_center + bar_height / 2 + 0.36,
            f'{hmr_value:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color='white',
        )

        # Líneas de corte en 1.0 y 1.5.
        for cut in (1.0, 1.5):
            ax.axvline(cut, color='white', lw=0.8, ls='--', alpha=0.5)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1.15)
        ax.axis('off')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.05)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('ascii')

    def _generate_roi_histogram_b64(self) -> str:
        """Genera histograma de intensidades dentro del ROI cardíaco como base64."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io, base64
        from scipy.stats import skew

        image = self._original_image
        if image is None:
            return ""

        roi_data = self._roi_widget._rois[0]
        roi = ROICircle(cy=roi_data["cy"], cx=roi_data["cx"], radius=roi_data["radius"])
        mask = roi.mask(image.shape)
        pixels = image[mask].astype(np.float64)

        if pixels.size == 0:
            return ""

        mean_val = float(np.mean(pixels))
        median_val = float(np.median(pixels))
        std_val = float(np.std(pixels))
        skew_val = float(skew(pixels))

        fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')

        ax.hist(pixels, bins=60, color='#f87171', alpha=0.75, edgecolor='#fca5a5', linewidth=0.5)

        ax.axvline(mean_val, color='white', lw=1.5, ls='--', label=f'Media: {mean_val:.1f}')
        ax.axvline(median_val, color='#fbbf24', lw=1.5, ls='--', label=f'Mediana: {median_val:.1f}')

        tail_text = f"σ = {std_val:.1f}  |  Skew = {skew_val:.2f}"
        if skew_val > 1.0:
            tail_text += "\n⚠ Cola derecha → pool sanguíneo posible"
        else:
            tail_text += "\n✓ Distribución razonablemente simétrica"

        ax.text(
            0.98, 0.95, tail_text,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='#e2e8f0',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e293b', edgecolor='#475569', alpha=0.9),
        )

        ax.set_xlabel('Intensidad (cuentas)', color='#e2e8f0', fontsize=9)
        ax.set_ylabel('Frecuencia', color='#e2e8f0', fontsize=9)
        ax.set_title('Distribución de cuentas — ROI cardíaco', color='#e2e8f0', fontsize=10, fontweight='bold')
        ax.tick_params(colors='#94a3b8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#475569')
        ax.legend(fontsize=8, facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('ascii')

    def _generate_comparison_bar_b64(self) -> str | None:
        """Genera bar chart corazón vs mediastino como base64."""
        if not self._washout_data:
            return None
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io, base64

        times = sorted(self._washout_data.keys(), key=lambda t: t.replace('h', ''))
        heart_vals = [self._washout_data[t]["heart_counts"] for t in times]
        medi_vals = [self._washout_data[t]["mediastinum_counts"] for t in times]

        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')

        x = range(len(times))
        bar_w = 0.35
        bars_h = ax.bar([i - bar_w/2 for i in x], heart_vals, bar_w, color='#f87171', label='Corazón')
        bars_m = ax.bar([i + bar_w/2 for i in x], medi_vals, bar_w, color='#38bdf8', label='Mediastino')

        for bar in bars_h:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(heart_vals)*0.02,
                    f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8, color='#f87171')
        for bar in bars_m:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(medi_vals)*0.02,
                    f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8, color='#38bdf8')

        ax.set_xticks(list(x))
        ax.set_xticklabels(times, color='#e2e8f0', fontsize=9)
        ax.set_ylabel('Cuentas promedio', color='#94a3b8', fontsize=9)
        ax.set_title('Cuentas corazón vs mediastino', color='#e2e8f0', fontsize=10, fontweight='bold')
        ax.tick_params(colors='#94a3b8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#475569')
        ax.legend(fontsize=8, facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('ascii')

    def _generate_perugini_strip_b64(self, perugini_score: int) -> str:
        """Genera escala Perugini visual como base64."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import io, base64

        labels = ["0 — Sin captación", "1 — Leve (< hueso)", "2 — Moderado (= hueso)", "3 — Intenso (> hueso)"]
        colors = ['#4ade80', '#fbbf24', '#f97316', '#f87171']

        fig, ax = plt.subplots(figsize=(6, 0.8), dpi=100)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')

        for i, (lbl, col) in enumerate(zip(labels, colors)):
            is_active = (i == perugini_score)
            alpha = 1.0 if is_active else 0.35
            edgecolor = 'white' if is_active else '#475569'
            lw = 3 if is_active else 1
            rect = mpatches.FancyBboxPatch((i * 1.5 + 0.05, 0.1), 1.4, 0.6,
                                           boxstyle="round,pad=0.05",
                                           facecolor=col, edgecolor=edgecolor,
                                           linewidth=lw, alpha=alpha)
            ax.add_patch(rect)
            ax.text(i * 1.5 + 0.75, 0.4, str(i), ha='center', va='center',
                    fontsize=16 if is_active else 12, fontweight='bold',
                    color='#0f172a' if is_active else '#1e293b')

        ax.set_xlim(-0.1, 6.1)
        ax.set_ylim(0, 1)
        ax.set_xticks([i * 1.5 + 0.75 for i in range(4)])
        ax.set_xticklabels(labels, fontsize=7, color='#94a3b8')
        ax.set_yticks([])
        ax.set_title('Perugini visual score', color='#e2e8f0', fontsize=9, fontweight='bold', pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('ascii')

    def _generate_html(self, html_path, img_path, composite_path, result, perugini, report_ctx):
        """Genera el informe HTML con bloques para cada tiempo cuantificado."""
        import base64
        import io
        from html import escape
        from PIL import Image as PILImage

        img_b64 = ""
        if img_path and os.path.isfile(img_path):
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("ascii")

        hist_b64 = self._generate_roi_histogram_b64()
        washout_b64 = self._generate_washout_curve_b64()
        comparison_b64 = self._generate_comparison_bar_b64()
        perugini_strip_b64 = self._generate_perugini_strip_b64(perugini) if perugini is not None else ""
        patient = escape(str(self._metadata["patient"]))
        date = escape(str(self._metadata["date"]))
        series = escape(str(self._metadata["series"]))
        template_name_raw = str(report_ctx.get("template_name") or "AMILO Básico")
        template_name = escape(template_name_raw)
        profile = dict(report_ctx.get("template_profile") or {})
        profile_focus = escape(str(profile.get("focus") or ""))
        profile_includes = "".join(f"<li>{escape(str(x))}</li>" for x in list(profile.get("includes") or []))
        profile_omits = "".join(f"<li>{escape(str(x))}</li>" for x in list(profile.get("omits") or []))
        profile_accent = str(profile.get("accent") or "#38bdf8")
        show_coverage = template_name_raw in ("AMILO Clínico Completo", "AMILO SPECT", "AMILO Básico")
        show_planar_blocks = template_name_raw in ("AMILO Clínico Completo", "AMILO Planar", "AMILO Básico")
        show_advanced_graphs = template_name_raw in ("AMILO Clínico Completo", "AMILO Planar")
        show_oai_and_kinetic = template_name_raw == "AMILO Clínico Completo"
        show_layout = template_name_raw in ("AMILO Clínico Completo", "AMILO SPECT")
        bridge = dict(report_ctx.get("bridge") or {})
        bridge_profile = dict(bridge.get("profile") or {})

        temporal_blocks = []
        temporal_hmr = {}
        for time_label in ("1h", "3h"):
            if not show_planar_blocks:
                continue
            data = self._washout_data.get(time_label)
            has_ap = self._time_images.get(time_label, {}).get("ap") is not None
            if data is None:
                if has_ap:
                    temporal_blocks.append(f"""
<section class="card">
    <h2>Resultados {time_label}</h2>
    <p class="pending">Imagen AP cargada; cuantificación HMR pendiente.</p>
</section>""")
                continue

            hmr = float(data["hmr"])
            heart_counts = float(data["heart_counts"])
            mediastinum_counts = float(data["mediastinum_counts"])
            classification = data.get("classification") or (
                "POSITIVO" if hmr >= 1.5 else ("EQUÍVOCO" if hmr >= 1.0 else "NEGATIVO")
            )
            color_class = "positive" if hmr >= 1.5 else ("equivocal" if hmr >= 1.0 else "negative")
            perugini_time = self._perugini_by_time.get(time_label, perugini)
            perugini_confirmed = bool(self._perugini_confirmed_by_time.get(time_label, False))
            perugini_state_text = "Final (confirmado manualmente)" if perugini_confirmed else "Sugerido por HMR"
            q_bone_val = data.get("q_bone")
            q_bone_text = f"{q_bone_val:.2f}" if q_bone_val is not None else "N/D"
            q_bone_mode = str(data.get("q_bone_mode", "auto"))
            q_bone_mode_text = "manual (ROI esternón/costilla)" if q_bone_mode == "manual" else "auto (estimado)"
            hmr_raw = data.get("hmr_raw")
            eb = dict(data.get("exclude_bone") or {})
            rib_used = bool(eb.get("rib_filter_used", False))
            sternum_used = bool(eb.get("sternum_filter_used", False))
            sc_used = bool(eb.get("scatter_used", False))
            rib_method = str(eb.get("rib_filter_method", "N/D"))
            rib_asym = eb.get("rib_filter_asym", None)
            sternum_asym = eb.get("sternum_filter_asym", None)
            rib_note = str(eb.get("rib_filter_note", "") or "")
            sternum_note = str(eb.get("sternum_filter_note", "") or "")
            temporal_hmr[time_label] = hmr

            roi_html = '<div class="roi-placeholder">Imagen ROI no disponible</div>'
            roi_image = self._processed_images.get(time_label, {}).get("roi")
            if roi_image is not None:
                roi_array = np.asarray(roi_image)
                if roi_array.dtype != np.uint8:
                    roi_array = np.clip(roi_array, 0, 255).astype(np.uint8)
                roi_buffer = io.BytesIO()
                PILImage.fromarray(roi_array).save(roi_buffer, format="PNG")
                roi_b64 = base64.b64encode(roi_buffer.getvalue()).decode("ascii")
                roi_html = (
                    f'<img class="report-image" src="data:image/png;base64,{roi_b64}" '
                    f'alt="AP cuantificada {time_label}">'
                )

            temporal_blocks.append(f"""
<section class="card">
    <h2>Resultados {time_label}</h2>
    <div class="result-grid">
        <div>{roi_html}</div>
        <div>
            <div class="metric {color_class}">{hmr:.2f}</div>
            <table>
                <tr><th>Métrica</th><th>Valor</th><th>Referencia</th></tr>
                <tr><td>HMR ({time_label})</td><td>{hmr:.2f}</td><td>≥1.5 sugiere ATTR</td></tr>
                <tr><td>Cuentas cardíacas</td><td>{heart_counts:,.0f}</td><td></td></tr>
                <tr><td>Cuentas mediastinales</td><td>{mediastinum_counts:,.0f}</td><td></td></tr>
                <tr><td>Clasificación</td><td>{escape(str(classification))}</td><td></td></tr>
                <tr><td>HMR raw</td><td>{float(hmr_raw):.2f}</td><td>Sin correcciones experimentales</td></tr>
                <tr><td>Perugini</td><td>{escape(str(perugini_time))}</td><td>0–3</td></tr>
                <tr><td>Perugini estado</td><td>{escape(perugini_state_text)}</td><td>Control clínico</td></tr>
                <tr><td>Q_bone (calidad ósea)</td><td>{escape(q_bone_text)}</td><td>{escape(q_bone_mode_text)}</td></tr>
                <tr><td>Filtro costal</td><td>{'Sí' if rib_used else 'No'}</td><td>{escape(rib_method if rib_used else 'No aplicado')}</td></tr>
                <tr><td>Filtro esternón</td><td>{'Sí' if sternum_used else 'No'}</td><td>{'2 fondos' if sternum_used else 'No aplicado'}</td></tr>
                <tr><td>SCATTER planar</td><td>{'Sí' if sc_used else 'No'}</td><td>{escape(f"k={float(eb.get('scatter_k', 0.0)):.2f}" if sc_used else 'No aplicado')}</td></tr>
                <tr><td>Asimetría costal</td><td>{escape(f"{float(rib_asym):.2f}" if rib_asym is not None else 'N/D')}</td><td>Se descarta corrección si supera umbral</td></tr>
                <tr><td>Asimetría fondos esternón</td><td>{escape(f"{float(sternum_asym):.2f}" if sternum_asym is not None else 'N/D')}</td><td>Se descarta corrección si supera umbral</td></tr>
            </table>
            {f"<p style='font-size:.82rem;color:#94a3b8;'>Nota corrección experimental: {escape(' | '.join([x for x in [rib_note, sternum_note] if x]))}</p>" if (rib_note or sternum_note) else ""}
        </div>
    </div>
</section>""")

        if temporal_blocks:
            results_html = "".join(temporal_blocks)
            roi_html = ""
        elif show_planar_blocks and result is not None:
            current_hmr = float(result.hmr)
            color_class = "positive" if current_hmr >= 1.5 else ("equivocal" if current_hmr >= 1.0 else "negative")
            results_html = f"""
<section class="card">
    <h2>Resultados</h2>
    <div class="metric {color_class}">{current_hmr:.2f}</div>
    <table>
        <tr><th>Métrica</th><th>Valor</th><th>Referencia</th></tr>
        <tr><td>HMR</td><td>{current_hmr:.2f}</td><td>≥1.5 sugiere ATTR</td></tr>
        <tr><td>Cuentas cardíacas</td><td>{float(result.heart_counts):,.0f}</td><td></td></tr>
        <tr><td>Cuentas mediastinales</td><td>{float(result.mediastinum_counts):,.0f}</td><td></td></tr>
        <tr><td>Clasificación</td><td>{escape(str(result.classification))}</td><td></td></tr>
        <tr><td>Perugini</td><td>{escape(str(perugini))}</td><td>0–3</td></tr>
    </table>
</section>"""
            roi_html = f"""
<section class="card">
    <h2>Imagen planar con ROIs</h2>
    <img class="report-image" src="data:image/png;base64,{img_b64}" alt="Imagen planar con ROIs">
</section>"""
        else:
            results_html = """
<section class="card">
    <h2>Resultados</h2>
    <p class="pending">Sin cuantificación planar (HMR) en este estudio.</p>
    <p>Se prioriza correlación tomográfica SPECT/SPECT-CT y contexto clínico.</p>
</section>"""
            roi_html = ""

        layout_html = ""
        if show_layout and composite_path and os.path.isfile(composite_path):
            with open(composite_path, "rb") as f:
                composite_b64 = base64.b64encode(f.read()).decode("ascii")
            layout_html = f"""
<section class="card">
    <h2>Layout completo</h2>
    <img class="report-image" src="data:image/png;base64,{composite_b64}" alt="Layout completo">
</section>"""

        histogram_html = ""
        if show_advanced_graphs and hist_b64:
            histogram_html = f"""
<section class="card">
    <h2>Distribución de cuentas ROI cardíaco</h2>
    <img class="report-image" src="data:image/png;base64,{hist_b64}" alt="Histograma ROI cardíaco">
</section>"""

        comparison_html = ""
        if show_advanced_graphs and comparison_b64:
            comparison_html = f"""
<section class="card">
    <h2>Cuentas corazón vs mediastino</h2>
    <img class="report-image" src="data:image/png;base64,{comparison_b64}" alt="Comparación cuentas">
</section>"""

        perugini_strip_html = ""
        if perugini_strip_b64:
            perugini_strip_html = f"""
<section class="card">
    <h2>Escala Perugini visual</h2>
    <img class="report-image" src="data:image/png;base64,{perugini_strip_b64}" alt="Perugini visual">
</section>"""

        washout_html = ""
        if show_advanced_graphs and washout_b64:
            washout_html = f"""
<section class="card">
    <h2>Curva de washout (1h vs 3h)</h2>
    <img class="report-image" src="data:image/png;base64,{washout_b64}" alt="Curva de washout">
</section>"""

        oai_optional_html = ""
        if show_oai_and_kinetic and "1h" in self._oai_washout_data and "3h" in self._oai_washout_data:
            h1 = float(self._oai_washout_data["1h"].get("heart_counts", 0.0))
            h3 = float(self._oai_washout_data["3h"].get("heart_counts", 0.0))
            if h1 > 0:
                ret = h3 / h1
                wo = (1.0 - ret) * 100.0
                oai_optional_html = f"""
<section class="card">
    <h2>Washout OAI opcional (ROI corazón)</h2>
    <table>
        <tr><th>Métrica</th><th>Valor</th><th>Nota</th></tr>
        <tr><td>Cuentas OAI 1h (norm.)</td><td>{h1:.4f}</td><td>cps corregidas por decaimiento</td></tr>
        <tr><td>Cuentas OAI 3h (norm.)</td><td>{h3:.4f}</td><td>cps corregidas por decaimiento</td></tr>
        <tr><td>Retención OAI 3h/1h</td><td>{ret:.3f}</td><td>Opcional</td></tr>
        <tr><td>Washout OAI</td><td>{wo:+.1f}%</td><td>Opcional · no reemplaza HMR AP</td></tr>
    </table>
    <p style="font-size:.85rem;color:#94a3b8;">Métrica auxiliar. Interpretar junto con AP/HMR y SPECT para evitar sesgo por blood pool/proyección.</p>
</section>"""

        kinetic_html = ""
        if show_oai_and_kinetic and self._kinetic_result is not None:
            km = self._kinetic_result
            kinetic_html = f"""
<section class="card">
    <h2>Análisis temporal PYP — EXPERIMENTAL</h2>
    <p class="pending"><strong>No diagnóstico / no terapéutico.</strong> No subtipifica ATTR/AL.</p>
    <table>
        <tr><th>Métrica observable</th><th>Valor</th></tr>
        <tr><td>Retención cardíaca 3h/1h (cps corregidas)</td><td>{km.heart_retention_3h_over_1h:.2f}</td></tr>
        <tr><td>Cambio cardíaco 1h→3h</td><td>{km.heart_change_pct:+.1f}%</td></tr>
        <tr><td>HMR temporal 1h</td><td>{km.hmr_1h:.2f}</td></tr>
        <tr><td>HMR temporal 3h</td><td>{km.hmr_3h:.2f}</td></tr>
        <tr><td>ΔHMR</td><td>{km.hmr_change:+.2f}</td></tr>
    </table>
</section>"""

        if show_planar_blocks and "1h" in temporal_hmr and "3h" in temporal_hmr:
            interpretation_summary = f'HMR 1h = {temporal_hmr["1h"]:.2f} y HMR 3h = {temporal_hmr["3h"]:.2f}.'
        elif show_planar_blocks and result is not None:
            interpretation_summary = f'HMR = {float(result.hmr):.2f}. <strong>{escape(str(result.classification))}</strong>'
        else:
            interpretation_summary = (
                'Sin HMR disponible. Confirmar miocardio vs blood pool en SPECT/SPECT-CT '
                'y completar exclusión de AL antes de concluir ATTR-CM.'
            )

        study_coverage_html = f"""
<section class="card">
    <h2>Cobertura de estudio</h2>
    <table>
        <tr><th>Ítem</th><th>Estado</th></tr>
        <tr><td>Planar cargado</td><td>{'Sí' if report_ctx.get('has_planar_loaded') else 'No'}</td></tr>
        <tr><td>Planar cuantificado (HMR)</td><td>{'Sí' if report_ctx.get('has_planar_metrics') else 'No'}</td></tr>
        <tr><td>SPECT vinculado</td><td>{'Sí' if report_ctx.get('has_spect') else 'No'}</td></tr>
        <tr><td>CT vinculado</td><td>{'Sí' if report_ctx.get('has_ct') else 'No'}</td></tr>
        <tr><td>Workflow puente</td><td>{escape(str(bridge.get('workflow_tag') or 'N/D'))}</td></tr>
        <tr><td>Serie/Protocolo puente</td><td>{escape(str(bridge_profile.get('series_description') or bridge_profile.get('protocol') or 'N/D'))}</td></tr>
    </table>
</section>"""
        if not show_coverage:
            study_coverage_html = ""

        al_block_html = f"""
<section class="card">
    <h2>Exclusión de AL (obligatorio para confirmar ATTR-CM)</h2>
    <table>
        <tr><th>Ítem</th><th>Estado</th></tr>
        <tr><td>Cadenas livianas libres</td><td>{escape(str(report_ctx.get('free_light_chain') or 'No informado'))}</td></tr>
        <tr><td>Inmunofijación sérica/urinaria</td><td>{escape(str(report_ctx.get('immunofixation') or 'No informada'))}</td></tr>
        <tr><td>Estado AL</td><td>{escape(str(report_ctx.get('al_status') or 'PENDIENTE / NO INFORMADO'))}</td></tr>
    </table>
    <p style="font-size:.85rem;color:#94a3b8;">La gammagrafía no debe usarse de forma aislada para confirmar ATTR-CM si AL no fue excluida.</p>
</section>"""

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SINCRO — Amiloidosis</title>
<style>
:root {{ color-scheme:dark; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#0f172a; color:#e2e8f0; font-family:"Segoe UI",Arial,sans-serif; }}
.report {{ width:min(100% - 32px,980px); margin:0 auto; padding:24px 0; }}
.header {{ padding:24px; margin-bottom:20px; text-align:center; background:linear-gradient(135deg,#1a3a5c,#111827); border:1px solid #334155; border-bottom:3px solid #38bdf8; border-radius:14px; }}
.header h1 {{ margin:0; color:#38bdf8; font-size:1.9rem; letter-spacing:.08em; }}
.subtitle {{ color:#cbd5e1; }}
.patient-data {{ margin-top:12px; color:#94a3b8; font-size:.88rem; }}
.card {{ margin:16px 0; padding:20px; overflow:hidden; background:#1e293b; border:1px solid #475569; border-radius:12px; }}
.card h2 {{ margin:0 0 16px; color:#f8fafc; font-size:1.15rem; }}
.result-grid {{ display:grid; grid-template-columns:minmax(240px,.85fr) minmax(320px,1.15fr); gap:20px; align-items:start; }}
.report-image {{ display:block; width:100%; height:auto; border:1px solid #475569; border-radius:8px; }}
.roi-placeholder {{ display:grid; min-height:180px; place-items:center; color:#94a3b8; border:1px dashed #475569; border-radius:8px; }}
.metric {{ margin-bottom:12px; color:#38bdf8; font-size:2.6rem; font-weight:800; }}
.metric.positive {{ color:#f87171; }}
.metric.equivocal,.pending {{ color:#fbbf24; }}
.metric.negative {{ color:#4ade80; }}
table {{ width:100%; border-collapse:collapse; }}
th {{ padding:9px; background:#1a3a5c; color:#f8fafc; text-align:left; }}
td {{ padding:9px; border-bottom:1px solid #475569; }}
.footer {{ padding:16px; color:#94a3b8; font-size:.8rem; text-align:center; border-top:1px solid #475569; }}
.template-card {{ margin:16px 0; padding:14px; background:#0f172a; border:1px solid {profile_accent}; border-radius:10px; }}
.template-card h3 {{ margin:0 0 8px; color:{profile_accent}; font-size:1rem; }}
.template-card ul {{ margin:6px 0 0 18px; color:#cbd5e1; font-size:.9rem; }}
@media (max-width:700px) {{
    .report {{ width:min(100% - 20px,980px); padding:10px 0; }}
    .header,.card {{ padding:16px; }}
    .result-grid {{ grid-template-columns:1fr; }}
    table {{ font-size:.88rem; }}
}}
</style>
</head>
<body>
<main class="report">
<header class="header">
    <h1>SINCRO</h1>
    <div class="subtitle">Informe de Amiloidosis Cardíaca</div>
    <div class="subtitle">Plantilla: {template_name}</div>
    <div class="patient-data">Paciente: {patient} · Fecha: {date} · Serie: {series}</div>
</header>
<section class="template-card">
    <h3>Perfil de plantilla</h3>
    <p>{profile_focus}</p>
    {f"<p><b>Incluye</b></p><ul>{profile_includes}</ul>" if profile_includes else ""}
    {f"<p style='margin-top:8px;'><b>Omite</b></p><ul>{profile_omits}</ul>" if profile_omits else ""}
</section>
{study_coverage_html}
{al_block_html}
{roi_html}
{results_html}
<section class="card">
    <h2>Interpretación clínica</h2>
    <p>{interpretation_summary}</p>
    <p>Si EQUÍVOCO (HMR 1.0–1.5), considerar SPECT/CT o planar a 3h para descartar pool sanguíneo residual.</p>
    <p>Perugini ≥ 2 + ausencia de gammapatía monoclonal confirma ATTR.</p>
</section>
{histogram_html}
{comparison_html}
{perugini_strip_html}
{washout_html}
{oai_optional_html}
{kinetic_html}
{layout_html}
<footer class="footer">Informe SINCRO — Amiloidosis. Resultados orientativos.</footer>
</main>
</body>
</html>"""
        with open(html_path, "wb") as f:
            f.write(html.encode("utf-8"))

    # ── Paginación de layouts ───────────────────────────────────────

    def _prev_page(self):
        """El flujo 1 h/3 h usa posiciones fijas y no requiere paginación."""
        self._page_offset = 0

    def _next_page(self):
        """El flujo 1 h/3 h usa posiciones fijas y no requiere paginación."""
        self._page_offset = 0

