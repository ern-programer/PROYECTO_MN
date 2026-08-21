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

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPainter, QPen, QColor, QBrush, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QWidget, QSizePolicy, QMessageBox, QStackedWidget,
    QSlider, QFrame,
)
import os

from core.amyloid_planar import ROICircle, compute_hmr, PERUGINI_SCORES
from core.amyloid_layouts import (
    LAYOUT_NAMES,
    layout_4q, layout_8q, layout_9q, layout_12q, layout_16q,
)
from ui.quadrant_viewer import QuadrantViewer


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
        self._drag_roi = -1

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
                painter.setPen(QPen(color, 2.0, Qt.PenStyle.DashLine))
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 60)))
                painter.drawEllipse(QPointF(rcx, rcy), rr, rr)
            # Etiqueta.
            painter.setPen(QPen(color, 1.5))
            painter.drawText(int(rcx + rr + 5), int(rcy), roi["name"])

    def mousePressEvent(self, event: QMouseEvent):
        for i, roi in enumerate(self._rois):
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
            rcx = event.position().x()
            rcy = event.position().y()
            dist = np.sqrt((rcx - ox - roi["cx"] * scale) ** 2 + (rcy - oy - roi["cy"] * scale) ** 2)
            if dist < roi["radius"] * 1.5 * scale:
                new_radius = max(3.0, min(64.0, roi["radius"] + delta * 1.0))
                # Actualizar AMBOS ROIs con el mismo radio (tienen que ser igual).
                for r in self._rois:
                    r["radius"] = new_radius
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
        4: layout_4q,
        8: layout_8q,
        9: layout_9q,
        12: layout_12q,
        16: layout_16q,
    }

    def __init__(self, parent=None, image=None, study=None):
        super().__init__(parent)
        self.setWindowTitle("SINCRO — Amiloidosis")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinMaxButtonsHint)
        self.resize(1100, 700)
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
        self._perugini_by_time: dict[str, int] = {}
        self._active_time: str | None = None
        self._washout_data: dict[str, dict] = {}  # tiempo_label → {hmr, heart_counts, mediastinum_counts}

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
            n = int(key.replace("q", ""))
            self._layout_combo.addItem(name, n)
        self._layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        toolbar.addWidget(self._layout_combo)

        btn_load_1h = QPushButton("Cargar imágenes 1h")
        btn_load_1h.clicked.connect(lambda: self._load_time_images("1h"))
        toolbar.addWidget(btn_load_1h)
        btn_load_3h = QPushButton("Cargar imágenes 3h")
        btn_load_3h.clicked.connect(lambda: self._load_time_images("3h"))
        toolbar.addWidget(btn_load_3h)

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

        self._washout_preview = QLabel()
        self._washout_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._washout_preview.setWordWrap(True)
        self._washout_preview.setMinimumHeight(90)
        self._washout_preview.setStyleSheet("color:#94a3b8; border:1px solid #334155; padding:4px;")
        sidebar.addWidget(self._washout_preview)

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
        sidebar_frame.setFixedWidth(160)
        sidebar_frame.setStyleSheet("QFrame { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 8px; }")
        visor_layout.addWidget(sidebar_frame)

        self._stack.addWidget(page_visor)

        # ── Página 1: Análisis ROI (modo clásico) ──────────────────
        page_analysis = QWidget()
        analysis_layout = QVBoxLayout(page_analysis)
        analysis_layout.setContentsMargins(0, 0, 0, 0)

        self._roi_widget = ROIDragWidget(image if image is not None else np.zeros((64, 64)))
        self._roi_widget.roiChanged.connect(self._update_hmr)
        analysis_layout.addWidget(self._roi_widget, 1)

        self._lbl_hmr = QLabel("HMR = N/D")
        self._lbl_hmr.setStyleSheet("font-size: 16px; font-weight: bold; color: #e2e8f0;")
        analysis_layout.addWidget(self._lbl_hmr)

        self._lbl_class = QLabel("")
        self._lbl_class.setStyleSheet("font-size: 12px; color: #94a3b8;")
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
                suggested = 3 if result.hmr >= 1.5 else (2 if result.hmr >= 1.0 else 0)
                self._perugini_combo.setCurrentIndex(suggested)
        except Exception:
            self._perugini_combo.setCurrentIndex(0)
        analysis_layout.addWidget(self._perugini_combo)

        # Selector de tiempo para washout.
        time_row = QHBoxLayout()
        lbl_time = QLabel("Tiempo:")
        lbl_time.setStyleSheet(self._lbl_css)
        time_row.addWidget(lbl_time)
        self._time_combo = QComboBox()
        self._time_combo.addItems(["1h", "3h"])
        self._time_combo.setEnabled(False)
        self._time_combo.setStyleSheet("QComboBox { background: #1e293b; color: #e2e8f0; border: 1px solid #475569; padding: 4px; border-radius: 4px; } QComboBox QAbstractItemView { background: #1e293b; color: #e2e8f0; selection-background-color: #2563eb; }")
        time_row.addWidget(self._time_combo)
        analysis_layout.addLayout(time_row)

        # Selector de filtro visual (solo para posicionamiento de ROIs).
        from core.amyloid_planar import VISUAL_FILTERS
        filter_row = QHBoxLayout()
        lbl_filt = QLabel("Filtro visual:")
        lbl_filt.setStyleSheet(self._lbl_css)
        filter_row.addWidget(lbl_filt)
        self._filter_combo = QComboBox()
        for key, (name, _) in VISUAL_FILTERS.items():
            self._filter_combo.addItem(name, key)
        self._filter_combo.setStyleSheet("QComboBox { background: #1e293b; color: #e2e8f0; border: 1px solid #475569; padding: 4px; border-radius: 4px; } QComboBox QAbstractItemView { background: #1e293b; color: #e2e8f0; selection-background-color: #2563eb; }")
        self._filter_combo.currentIndexChanged.connect(self._on_visual_filter_changed)
        filter_row.addWidget(self._filter_combo)
        analysis_layout.addLayout(filter_row)

        # Botón Aplicar: renderiza ROIs + HMR y asigna al cuadrante AP+ROIs.
        btn_apply = QPushButton("Aplicar ROIs al cuadrante")
        btn_apply.setStyleSheet("font-size: 13px; font-weight: bold; padding: 8px; background: #2563eb; color: white; border-radius: 6px;")
        btn_apply.clicked.connect(self._apply_rois_to_quadrant)
        analysis_layout.addWidget(btn_apply)

        # Label de estado de washout.
        self._lbl_washout_status = QLabel("")
        self._lbl_washout_status.setStyleSheet("font-size: 10px; color: #94a3b8; padding: 4px;")
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
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_close)
        root.addLayout(btns)

        # ── Inicializar ────────────────────────────────────────────
        if image is not None:
            self._time_images["1h"]["ap"] = {"image": np.asarray(image, dtype=np.float64), "label": "AP", "path": ""}
            self._active_time = "1h"
        self._rebuild_layout()
        self._update_washout_preview()
        if image is not None:
            self._update_hmr(0, 0, 0, 0)

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

        def _slots(time_label: str) -> tuple[list[np.ndarray | None], list[str]]:
            source = self._time_images[time_label]
            ap_entry = source["ap"]
            ap = ap_entry["image"] if ap_entry else None
            processed = self._processed_images[time_label]
            images = [
                processed.get("roi", ap), processed.get("clean", ap),
                source["oai"]["image"] if source["oai"] else None,
                source["lat"]["image"] if source["lat"] else None,
            ]
            labels = [
                f"AP + ROIs ({time_label})" if "roi" in processed else f"AP cuantificación ({time_label})",
                f"AP limpio ({time_label})", f"OAI ({time_label})", f"LAT. IZQ. ({time_label})",
            ]
            return images, labels

        imgs_1h, labels_1h = _slots("1h")
        imgs_3h, labels_3h = _slots("3h")
        imgs = imgs_1h + imgs_3h
        labels = labels_1h + labels_3h

        if n == 4:
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
        elif n == 16:
            layout = layout_16q(images=imgs, labels=labels[:4] if len(labels) >= 4 else None)
        else:
            return
        for idx, label in enumerate(labels):
            if idx < len(layout.quadrants):
                layout.quadrants[idx].label = label
        self._quadrant_viewer.set_layout(layout)
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

    def _load_time_images(self, time_label: str):
        """Carga exactamente AP, OAI y lateral para un tiempo canónico."""
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
            if len(paths) != 3:
                QMessageBox.information(
                    self, "SINCRO — Amyloidosis",
                    "Seleccioná exactamente tres DICOM: AP, OAI y lateral izquierda."
                )
                return
            records = []
            for p in paths:
                try:
                    import pydicom
                    ds = pydicom.dcmread(p, force=True)
                    img = np.asarray(ds.pixel_array, dtype=np.float64)
                    while img.ndim > 2:
                        img = img[img.shape[0] // 2]
                    if img.ndim != 2:
                        raise ValueError(f"Imagen no planar: shape={img.shape}")
                    description = " ".join(filter(None, [
                        str(getattr(ds, "SeriesDescription", "") or ""),
                        str(getattr(ds, "ViewPosition", "") or ""),
                    ])).upper()
                    records.append({"image": img, "label": description or os.path.basename(p),
                                    "path": p, "view": self._classify_planar_view(description), "ds": ds})
                except Exception as exc:
                    QMessageBox.warning(self, "SINCRO", f"Error cargando {os.path.basename(p)}:\n{exc}")
                    return

            assigned: dict[str, dict] = {}
            used: set[int] = set()
            for idx, record in enumerate(records):
                view = record["view"]
                if view and view not in assigned:
                    assigned[view] = record
                    used.add(idx)
            for role in ("ap", "oai", "lat"):
                if role not in assigned:
                    idx = next(i for i in range(len(records)) if i not in used)
                    assigned[role] = records[idx]
                    used.add(idx)

            self._time_images[time_label] = {role: assigned[role] for role in ("ap", "oai", "lat")}
            self._processed_images[time_label].clear()
            self._roi_state[time_label] = None
            self._washout_data.pop(time_label, None)
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
            self._update_washout_preview()

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

    def _on_cmap_changed(self, cmap: str):
        q = self._quadrant_viewer.selected_quadrant()
        if q is None:
            return
        q.cmap = cmap
        self._quadrant_viewer._rebuild_pixmaps()
        self._quadrant_viewer.update()

    def _on_window_changed(self):
        q = self._quadrant_viewer.selected_quadrant()
        if q is None:
            return
        q.win_low = float(self._win_low_slider.value())
        q.win_high = float(self._win_high_slider.value())
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
        if self._current_layout_n >= 8:
            slot_to_time = {0: "1h", 1: "1h", 4: "3h", 5: "3h"}
        else:
            slot_to_time = {0: "1h", 1: "1h"}
        time_label = slot_to_time.get(idx)
        if time_label is None:
            QMessageBox.information(
                self, "SINCRO — Amyloidosis",
                "La cuantificación HMR solo se realiza sobre una celda AP.\n"
                "Seleccioná AP cuantificación o AP limpio del tiempo correspondiente."
            )
            return
        ap_entry = self._time_images[time_label]["ap"]
        if ap_entry is None:
            QMessageBox.information(self, "SINCRO — Amyloidosis", f"No hay AP cargada para {time_label}.")
            return
        img = np.asarray(ap_entry["image"], dtype=np.float64)
        self._image = img  # imagen actual para display/render
        self._original_image = img.copy()  # original 2D para análisis
        self._active_time = time_label
        self._time_combo.setCurrentText(time_label)
        self._roi_widget = ROIDragWidget(img)
        saved_rois = self._roi_state.get(time_label)
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
        self._toggle_mode()
        self._update_hmr(0, 0, 0, 0)

    def _apply_rois_to_quadrant(self):
        """Aplica el análisis al par AP+ROI/AP limpio del tiempo activo."""
        time_label = self._active_time
        if self._original_image is None or time_label not in ("1h", "3h"):
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
            result = compute_hmr(self._original_image, roi_h, roi_m)
        except Exception as exc:
            QMessageBox.warning(self, "SINCRO", f"Error calculando HMR:\n{exc}")
            return

        # Guardar datos para curva de washout + Q_bone.
        from core.amyloid_planar import compute_q_bone
        q_bone_val = None
        try:
            # Estimar posiciones de esternón y costilla relativas al corazón.
            h_img, w_img = self._original_image.shape
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
            "heart_counts": result.heart_counts,
            "mediastinum_counts": result.mediastinum_counts,
            "classification": result.classification,
            "q_bone": q_bone_val,
        }
        self._perugini_by_time[time_label] = int(self._perugini_combo.currentData())
        self._roi_state[time_label] = [dict(roi) for roi in self._roi_widget._rois]

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
        self._rebuild_layout(force_layout=self._current_layout_n)
        layout = self._quadrant_viewer._layout
        roi_idx = 0 if time_label == "1h" else 4
        if layout is not None and roi_idx < len(layout.quadrants):
            layout.quadrants[roi_idx].label = f"AP + ROIs ({time_label}, HMR={result.hmr:.2f})"
            layout.quadrants[roi_idx].hmr = result.hmr
            layout.quadrants[roi_idx].roi_overlay = True
            self._quadrant_viewer._rebuild_pixmaps()
            self._quadrant_viewer.update()
        self._update_washout_preview()

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
        for roi in self._roi_widget._rois:
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
            result = compute_hmr(self._original_image, roi_h, roi_m)
            self._lbl_hmr.setText(f"HMR = {result.hmr:.2f}")
            self._lbl_class.setText(result.classification)
            color = "#f87171" if result.hmr >= 1.5 else ("#fbbf24" if result.hmr >= 1.0 else "#4ade80")
            self._lbl_hmr.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        except Exception as exc:
            self._lbl_hmr.setText("HMR = N/D")
            self._lbl_class.setText(f"Error: {exc}")

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
        self._roi_widget.update()
        if self._active_time in self._roi_state:
            self._roi_state[self._active_time] = [dict(roi) for roi in self._roi_widget._rois]
        self._update_hmr(0, 0, 0, 0)

    def _on_visual_filter_changed(self, idx: int):
        """Aplica/quita filtro visual al widget de ROIs (solo display, no raw)."""
        from core.amyloid_planar import apply_visual_filter, VISUAL_FILTERS
        filter_key = self._filter_combo.currentData()
        if filter_key is None or filter_key == "none":
            # Restaurar imagen raw.
            if self._original_image is not None:
                self._roi_widget._image = np.asarray(self._original_image, dtype=np.float64)
                self._roi_widget.update()
            return
        if self._original_image is None:
            return
        _, kwargs = VISUAL_FILTERS.get(filter_key, ("", {}))
        try:
            filtered = apply_visual_filter(self._original_image, filter_key, **kwargs)
            self._roi_widget._image = np.asarray(filtered, dtype=np.float64)
            self._roi_widget.update()
        except Exception:
            pass  # Si falla el filtro, dejar la imagen raw.

    def _update_washout_preview(self):
        """Actualiza estado y curva en vivo solo con 1 h y 3 h cuantificadas."""
        done = [time for time in ("1h", "3h") if time in self._washout_data]
        missing = [time for time in ("1h", "3h") if time not in self._washout_data]
        if missing:
            done_text = ", ".join(done) if done else "ninguno"
            missing_text = " y ".join(missing)
            status = f"Cuantificado: {done_text}. Falta cuantificar {missing_text}."
            self._lbl_washout_status.setText(status)
            self._lbl_washout_status.setStyleSheet("font-size:10px; color:#fbbf24; padding:4px;")
            self._washout_preview.clear()
            self._washout_preview.setText(status)
            self._washout_preview.setVisible(True)
            return
        curve_b64 = self._generate_washout_curve_b64()
        pixmap = QPixmap()
        if curve_b64 and pixmap.loadFromData(base64.b64decode(curve_b64), "PNG"):
            self._washout_preview.setPixmap(pixmap.scaledToWidth(145, Qt.TransformationMode.SmoothTransformation))
            self._washout_preview.setVisible(True)
        self._lbl_washout_status.setText("Washout 1h/3h cuantificado: curva lista para el informe.")
        self._lbl_washout_status.setStyleSheet("font-size:10px; color:#4ade80; padding:4px;")

    def _generate_report(self):
        """Genera el informe PDF + HTML de amiloidosis."""
        if self._original_image is None or self._active_time not in self._washout_data:
            QMessageBox.warning(
                self, "SINCRO — Amyloidosis",
                "Seleccioná y cuantificá una imagen AP antes de generar el informe."
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
            result = compute_hmr(self._original_image, roi_h, roi_m)
            perugini = int(self._perugini_combo.currentData())
            report_img = self.get_report_image()
            # Obtener imagen compuesta del layout.
            composite_img = self.get_layout_composite_image()
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_demo")
            os.makedirs(output_dir, exist_ok=True)
            # Guardar imagen con ROIs.
            img_path = os.path.join(output_dir, "amyloid_planar.png")
            from PIL import Image
            Image.fromarray(report_img).save(img_path, "PNG")
            # Guardar imagen compuesta del layout.
            composite_path = ""
            if composite_img is not None:
                composite_path = os.path.join(output_dir, "amyloid_layout_composite.png")
                Image.fromarray(composite_img.astype(np.uint8)).save(composite_path, "PNG")
            # PDF
            pdf_path = os.path.join(output_dir, "informe_amyloid.pdf")
            self._generate_pdf(pdf_path, img_path, composite_path, result, perugini)
            # HTML
            html_path = os.path.join(output_dir, "informe_amyloid.html")
            self._generate_html(html_path, img_path, composite_path, result, perugini)
            QMessageBox.information(
                self, "SINCRO — Amyloidosis",
                f"Informe generado:\nPDF: {pdf_path}\nHTML: {html_path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO — Amyloidosis", f"Error al generar informe:\n{exc}")

    def _generate_pdf(self, pdf_path, img_path, composite_path, result, perugini):
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

        story.append(Paragraph("SINCRO — Informe de Amiloidosis Cardíaca", title_style))
        story.append(Paragraph("Análisis de captación miocárdica con Tc-99m PYP/DPD/HMDP", small_style))
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

        # Imagen con ROIs (solo si no hay bloques temporales)
        if not self._washout_data:
            story.append(Paragraph("2. Imagen planar con ROIs", section_style))
            img = ImageReader(img_path)
            iw, ih = img.getSize()
            scale = min(160*mm / iw, 120*mm / ih)
            story.append(RLImage(img_path, width=iw*scale, height=ih*scale))
            story.append(Spacer(1, 3*mm))

        # Bloques por tiempo (1h, 3h)
        sec_num = 2
        for time_label in ("1h", "3h"):
            data = self._washout_data.get(time_label)
            if data is None:
                continue
            hmr = data["hmr"]
            heart = data["heart_counts"]
            medi = data["mediastinum_counts"]
            cls = data.get("classification", "")
            perugini_time = self._perugini_by_time.get(time_label, perugini)

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

            hmr_data = [
                ["Métrica", "Valor", "Referencia"],
                [f"HMR ({time_label})", f"{hmr:.2f}", "≥1.5 sugiere ATTR"],
                ["Cuentas cardíacas", f"{heart:,.0f}", ""],
                ["Cuentas mediastinales", f"{medi:,.0f}", ""],
                ["Clasificación", cls, ""],
                ["Perugini", str(perugini_time), "0–3"],
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
            story.append(Spacer(1, 4*mm))
            sec_num += 1

        # Si no hubo bloques temporales, mostrar resultado único
        if not self._washout_data:
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

        # Perugini visual strip
        perugini_strip_b64 = self._generate_perugini_strip_b64(perugini)
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
        hist_b64 = self._generate_roi_histogram_b64()
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
        comparison_b64 = self._generate_comparison_bar_b64()
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
        washout_b64 = self._generate_washout_curve_b64()
        if washout_b64:
            washout_buf = io.BytesIO(base64.b64decode(washout_b64))
            washout_reader = ImageReader(washout_buf)
            ww, wh = washout_reader.getSize()
            washout_scale = min(170*mm / ww, 75*mm / wh)
            story.append(Paragraph(f"{sec_num}. Curva de washout 1h vs 3h", section_style))
            story.append(RLImage(washout_buf, width=ww*washout_scale, height=wh*washout_scale))
            story.append(Spacer(1, 3*mm))
            sec_num += 1

        # Layout compuesto
        if composite_path and os.path.isfile(composite_path):
            story.append(Paragraph(f"{sec_num}. Layout completo", section_style))
            composite = ImageReader(composite_path)
            cw, ch = composite.getSize()
            composite_scale = min(170*mm / cw, 110*mm / ch)
            story.append(RLImage(composite_path, width=cw*composite_scale, height=ch*composite_scale))
            story.append(Spacer(1, 3*mm))
            sec_num += 1

        # Interpretación
        story.append(Paragraph(f"{sec_num}. Interpretación clínica", section_style))
        if "1h" in self._washout_data and "3h" in self._washout_data:
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
        else:
            interp = f"""
            El estudio muestra HMR de {result.hmr:.2f}. <b>{result.classification}</b><br/><br/>
            Si el resultado es equívoco (HMR 1.0–1.5), considerar imagen SPECT/CT o repetir planar a 3 horas
            para descartar pool sanguíneo residual.<br/><br/>
            La interpretación debe integrarse con laboratorio (cadenas livianas libres, proteínas monoclonales)
            y contexto clínico. El Perugini score ≥2 en presencia de gammapatía monoclonal ausente confirma ATTR.
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

    def _generate_html(self, html_path, img_path, composite_path, result, perugini):
                """Genera el informe HTML con bloques para cada tiempo cuantificado."""
                import base64
                import io
                from html import escape
                from PIL import Image as PILImage

                with open(img_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("ascii")

                hist_b64 = self._generate_roi_histogram_b64()
                washout_b64 = self._generate_washout_curve_b64()
                comparison_b64 = self._generate_comparison_bar_b64()
                perugini_strip_b64 = self._generate_perugini_strip_b64(perugini)
                patient = escape(str(self._metadata["patient"]))
                date = escape(str(self._metadata["date"]))
                series = escape(str(self._metadata["series"]))

                temporal_blocks = []
                temporal_hmr = {}
                for time_label in ("1h", "3h"):
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
                <tr><td>Perugini</td><td>{escape(str(perugini_time))}</td><td>0–3</td></tr>
            </table>
        </div>
    </div>
</section>""")

                if temporal_blocks:
                        results_html = "".join(temporal_blocks)
                        roi_html = ""
                else:
                        current_hmr = float(result.hmr)
                        color_class = (
                                "positive" if current_hmr >= 1.5
                                else ("equivocal" if current_hmr >= 1.0 else "negative")
                        )
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

                layout_html = ""
                if composite_path and os.path.isfile(composite_path):
                        with open(composite_path, "rb") as f:
                                composite_b64 = base64.b64encode(f.read()).decode("ascii")
                        layout_html = f"""
<section class="card">
    <h2>Layout completo</h2>
    <img class="report-image" src="data:image/png;base64,{composite_b64}" alt="Layout completo">
</section>"""

                histogram_html = ""
                if hist_b64:
                        histogram_html = f"""
<section class="card">
    <h2>Distribución de cuentas ROI cardíaco</h2>
    <img class="report-image" src="data:image/png;base64,{hist_b64}" alt="Histograma ROI cardíaco">
    <p style="font-size:0.85rem; color:#94a3b8; margin-top:8px;">La distribución de intensidades dentro del ROI cardíaco permite evaluar la homogeneidad de la captación. Una cola derecha (asimetría positiva) sugiere posible pool sanguíneo residual. Una distribución simétrica indica captación miocárdica homogénea.</p>
</section>"""

                comparison_html = ""
                if comparison_b64:
                        comparison_html = f"""
<section class="card">
    <h2>Cuentas corazón vs mediastino</h2>
    <img class="report-image" src="data:image/png;base64,{comparison_b64}" alt="Comparación cuentas">
    <p style="font-size:0.85rem; color:#94a3b8; margin-top:8px;">Comparación directa de cuentas promedio entre el ROI cardíaco y el mediastinal. La diferencia justifica el valor de HMR calculado.</p>
</section>"""

                perugini_strip_html = ""
                if perugini_strip_b64:
                        perugini_strip_html = f"""
<section class="card">
    <h2>Escala Perugini visual</h2>
    <img class="report-image" src="data:image/png;base64,{perugini_strip_b64}" alt="Perugini visual">
</section>"""

                washout_html = ""
                if washout_b64:
                        washout_html = f"""
<section class="card">
    <h2>Curva de washout (1h vs 3h)</h2>
    <img class="report-image" src="data:image/png;base64,{washout_b64}" alt="Curva de washout">
</section>"""

                if "1h" in temporal_hmr and "3h" in temporal_hmr:
                        interpretation_summary = (
                                f'HMR 1h = {temporal_hmr["1h"]:.2f} y '
                                f'HMR 3h = {temporal_hmr["3h"]:.2f}.'
                        )
                else:
                        interpretation_summary = (
                                f'HMR = {float(result.hmr):.2f}. '
                                f'<strong>{escape(str(result.classification))}</strong>'
                        )

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
    <div class="patient-data">Paciente: {patient} · Fecha: {date} · Serie: {series}</div>
</header>
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

