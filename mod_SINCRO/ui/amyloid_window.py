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

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPainter, QPen, QColor, QBrush, QPolygonF
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QWidget, QSizePolicy, QMessageBox, QStackedWidget,
    QSlider, QFileDialog, QFrame,
)
import os

from core.amyloid_planar import ROICircle, compute_hmr, PERUGINI_SCORES
from core.amyloid_layouts import (
    Quadrant, Layout, LAYOUT_CATALOG, LAYOUT_NAMES,
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
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
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
        self._loaded_images: list[tuple[str, np.ndarray]] = []  # (label, img) para cuadrantes
        self._current_layout_n = 4
        self._current_mode = "visor"  # "visor" | "analisis"
        self._page_offset = 0  # índice de inicio de la página actual
        self._processed_images: dict[int, np.ndarray] = {}  # índice → imagen procesada (ROIs/limpia)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Info del paciente ──────────────────────────────────────
        patient = getattr(study, "patient_name", "") or "N/D"
        date = getattr(study, "study_date", "") or "N/D"
        series = getattr(study, "series_description", "") or "N/D"
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

        # Botón cargar imágenes.
        btn_load = QPushButton("Cargar imágenes...")
        btn_load.clicked.connect(self._load_images)
        toolbar.addWidget(btn_load)

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

        # Botón Aplicar: renderiza ROIs + HMR y asigna al cuadrante AP+ROIs.
        btn_apply = QPushButton("Aplicar ROIs al cuadrante")
        btn_apply.setStyleSheet("font-size: 13px; font-weight: bold; padding: 8px; background: #2563eb; color: white; border-radius: 6px;")
        btn_apply.clicked.connect(self._apply_rois_to_quadrant)
        analysis_layout.addWidget(btn_apply)

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
            self._loaded_images.append(("Imagen 1", image))
        self._rebuild_layout()
        self._update_hmr(0, 0, 0, 0)

    # ── Modo ────────────────────────────────────────────────────────

    def _toggle_mode(self):
        """Alterna entre visor de cuadrantes y análisis ROI."""
        if self._current_mode == "visor":
            self._current_mode = "analisis"
            self._stack.setCurrentIndex(1)
            self._btn_mode.setText("← Visor cuadrantes")
        else:
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

    def _rebuild_layout(self, force_layout: int = None):
        """Reconstruye el layout con las imágenes cargadas. Preserva posiciones
        por índice y maneja paginación si hay más imágenes que cuadrantes."""
        all_imgs = [img for _, img in self._loaded_images]
        all_labels = [lbl for lbl, _ in self._loaded_images]
        n_imgs = len(all_imgs)

        if force_layout is not None:
            n = force_layout
        else:
            if n_imgs <= 4:
                n = 4
            elif n_imgs <= 8:
                n = 8
            elif n_imgs <= 9:
                n = 9
            elif n_imgs <= 12:
                n = 12
            else:
                n = 16
        self._current_layout_n = n

        # Paginación: mostrar solo la página actual.
        start = self._page_offset
        end = start + n
        imgs = all_imgs[start:end]
        labels = all_labels[start:end]

        if n == 4:
            layout = layout_4q(
                ap_roi=imgs[0] if len(imgs) > 0 else None,
                ap_clean=imgs[1] if len(imgs) > 1 else None,
                oai=imgs[2] if len(imgs) > 2 else None,
                lat=imgs[3] if len(imgs) > 3 else None,
                ap_label=labels[0] if len(labels) > 0 else "AP",
                oai_label=labels[2] if len(labels) > 2 else "OAI 45°",
                lat_label=labels[3] if len(labels) > 3 else "LAT. IZQ.",
            )
        elif n == 8:
            half = max(1, len(imgs) // 2)
            layout = layout_8q(
                images_1h=imgs[:half],
                images_3h=imgs[half:half*2],
                labels=labels[:4] if len(labels) >= 4 else None,
            )
        elif n == 9:
            layout = layout_9q(images=imgs, labels=labels[:3] if len(labels) >= 3 else None)
        elif n == 12:
            layout = layout_12q(images=imgs, labels=labels[:3] if len(labels) >= 3 else None)
        elif n == 16:
            layout = layout_16q(images=imgs, labels=labels[:4] if len(labels) >= 4 else None)
        else:
            return
        self._quadrant_viewer.set_layout(layout)
        # Restaurar imágenes procesadas (ROIs + limpia) si existen.
        for idx, img in self._processed_images.items():
            if idx < len(layout.quadrants):
                layout.quadrants[idx].image = img
                if idx == 0:
                    layout.quadrants[idx].label = f"AP + ROIs (HMR={layout.quadrants[idx].hmr:.2f})" if layout.quadrants[idx].hmr else "AP + ROIs"
                    layout.quadrants[idx].roi_overlay = True
                elif idx == 1:
                    layout.quadrants[idx].label = "AP (limpio)"
                    layout.quadrants[idx].roi_overlay = False
        self._quadrant_viewer._rebuild_pixmaps()
        self._quadrant_viewer.update()
        self._on_quadrant_selected(0)
        # Actualizar info de paginación.
        total_pages = (len(all_imgs) + n - 1) // n if n > 0 else 1
        self._lbl_pagination.setText(f"Página {self._page_offset // n + 1}/{total_pages}")

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

    # ── Cargar imágenes ─────────────────────────────────────────────

    def _load_images(self):
        """Abre el navegador DICOM con thumbnails para seleccionar imágenes."""
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
            for p in paths:
                try:
                    if p.lower().endswith((".dcm",)):
                        import pydicom
                        ds = pydicom.dcmread(p)
                        img = ds.pixel_array.astype(np.float64)
                        label = getattr(ds, "SeriesDescription", "") or os.path.basename(p)
                    else:
                        from PIL import Image as PILImage
                        img = np.array(PILImage.open(p)).astype(np.float64)
                        if img.ndim == 3:
                            img = img.mean(axis=2)
                        label = os.path.basename(p)
                    self._loaded_images.append((label, img))
                except Exception as exc:
                    QMessageBox.warning(self, "SINCRO", f"Error cargando {os.path.basename(p)}:\n{exc}")
            self._rebuild_layout()

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
        q = self._quadrant_viewer.selected_quadrant()
        if q is None or q.image is None:
            QMessageBox.information(self, "SINCRO", "Selecciona un cuadrante con imagen primero.")
            return
        # Usar la imagen original 2D si existe; si no, usar la del cuadrante.
        img = self._original_image if self._original_image is not None else q.image
        # Si la imagen es RGB (3D), convertir a gris para el análisis ROI.
        if img.ndim == 3:
            img = img.mean(axis=2)
        self._image = img  # imagen actual para display/render
        self._original_image = img.copy()  # original 2D para análisis
        self._roi_widget = ROIDragWidget(img)
        self._roi_widget.roiChanged.connect(self._update_hmr)
        # Reemplazar el widget de ROI en la página de análisis.
        old = self._stack.widget(1)
        old_layout = old.layout()
        if old_layout:
            # Quitar el widget viejo y poner el nuevo.
            for i in range(old_layout.count()):
                w = old_layout.itemAt(i).widget()
                if isinstance(w, ROIDragWidget):
                    old_layout.removeWidget(w)
                    w.deleteLater()
                    old_layout.insertWidget(0, self._roi_widget, 1)
                    break
        self._toggle_mode()
        self._update_hmr(0, 0, 0, 0)

    def _apply_rois_to_quadrant(self):
        """Renderiza la imagen con ROIs + HMR y la asigna al cuadrante 0 (AP+ROIs)."""
        if self._original_image is None:
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

        # Asignar al cuadrante 0 (AP+ROIs) — imagen RGB.
        img_rgb = np.array(pil, dtype=np.float64)
        layout = self._quadrant_viewer._layout
        if layout is not None and len(layout.quadrants) > 0:
            layout.quadrants[0].image = img_rgb
            layout.quadrants[0].label = f"AP + ROIs (HMR={result.hmr:.2f})"
            layout.quadrants[0].roi_overlay = True
            layout.quadrants[0].hmr = result.hmr
            # Guardar imagen procesada para restaurarla al cambiar de layout.
            self._processed_images[0] = img_rgb
            # Copia limpia al cuadrante 1 (reservado para AP limpia).
            if len(layout.quadrants) > 1:
                clean_img = self._original_image.copy()
                layout.quadrants[1].image = clean_img
                layout.quadrants[1].label = "AP (limpio)"
                layout.quadrants[1].roi_overlay = False
                # Guardar imagen limpia también.
                self._processed_images[1] = clean_img
            self._quadrant_viewer._rebuild_pixmaps()
            self._quadrant_viewer.update()

        # Volver al modo visor.
        if self._current_mode == "analisis":
            self._toggle_mode()

    def get_report_image(self) -> np.ndarray:
        """Renderiza la imagen con los ROIs como array RGB para el informe."""
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
        h, w = self._original_image.shape
        self._roi_widget._rois[0]["cy"] = 0.4 * h
        self._roi_widget._rois[0]["cx"] = 0.4 * w
        self._roi_widget._rois[0]["radius"] = 12.0
        self._roi_widget._rois[1]["cy"] = 0.6 * h
        self._roi_widget._rois[1]["cx"] = 0.6 * w
        self._roi_widget._rois[1]["radius"] = 12.0
        self._roi_widget.update()
        self._update_hmr(0, 0, 0, 0)

    def _generate_report(self):
        """Genera el informe PDF + HTML de amiloidosis."""
        if self._image is None or self._study is None:
            QMessageBox.warning(self, "SINCRO — Amyloidosis", "No hay imagen cargada.")
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
            # Obtener las imágenes del layout tal cual las organizó el usuario.
            layout_imgs = self.get_layout_images()
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_demo")
            os.makedirs(output_dir, exist_ok=True)
            # Guardar imagen con ROIs.
            img_path = os.path.join(output_dir, "amyloid_planar.png")
            from PIL import Image
            Image.fromarray(report_img).save(img_path, "PNG")
            # Guardar imágenes del layout.
            layout_paths = []
            for i, img in enumerate(layout_imgs):
                img_p = os.path.join(output_dir, f"amyloid_layout_{i}.png")
                Image.fromarray(img.astype(np.uint8)).save(img_p, "PNG")
                layout_paths.append(img_p)
            # PDF
            pdf_path = os.path.join(output_dir, "informe_amyloid.pdf")
            self._generate_pdf(pdf_path, img_path, layout_paths, result, perugini)
            # HTML
            html_path = os.path.join(output_dir, "informe_amyloid.html")
            self._generate_html(html_path, img_path, layout_paths, result, perugini)
            QMessageBox.information(
                self, "SINCRO — Amyloidosis",
                f"Informe generado:\nPDF: {pdf_path}\nHTML: {html_path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO — Amyloidosis", f"Error al generar informe:\n{exc}")

    def _generate_pdf(self, pdf_path, img_path, layout_paths, result, perugini):
        """Genera el informe PDF de amiloidosis."""
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
        patient = getattr(self._study, "patient_name", "") or "N/D"
        date = getattr(self._study, "study_date", "") or "N/D"
        series = getattr(self._study, "series_description", "") or "N/D"
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

        # Imagen con ROIs
        story.append(Paragraph("2. Imagen planar con ROIs", section_style))
        img = ImageReader(img_path)
        iw, ih = img.getSize()
        scale = min(160*mm / iw, 120*mm / ih)
        story.append(RLImage(img_path, width=iw*scale, height=ih*scale))
        story.append(Paragraph("Imagen planar con ROI cardíaco (rojo) y ROI mediastinal (azul).", small_style))
        story.append(Spacer(1, 3*mm))

        # HMR
        story.append(Paragraph("3. Métrica principal: HMR", section_style))
        hmr_data = [
            ["Métrica", "Valor", "Referencia"],
            ["HMR (Heart-to-Mediastinum)", f"{result.hmr:.2f}", "≥1.5 sugiere ATTR"],
            ["Cuentas cardíacas", f"{result.heart_counts:,.0f}", ""],
            ["Cuentas mediastinales", f"{result.mediastinum_counts:,.0f}", ""],
            ["Clasificación", result.classification, ""],
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

        # Perugini
        story.append(Paragraph("4. Perugini visual score", section_style))
        story.append(Paragraph(f"Score: {perugini} — {PERUGINI_SCORES.get(perugini, 'N/D')}", body_style))
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Referencia: 0 = sin captación; 1 = leve (< hueso); 2 = moderado (= hueso); 3 = intenso (> hueso).", small_style))
        story.append(Spacer(1, 4*mm))

        # Interpretación
        story.append(Paragraph("5. Interpretación clínica", section_style))
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

    def _generate_html(self, html_path, img_path, layout_paths, result, perugini):
        """Genera el informe HTML de amiloidosis."""
        import base64
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")
        patient = getattr(self._study, "patient_name", "") or "N/D"
        date = getattr(self._study, "study_date", "") or "N/D"
        series = getattr(self._study, "series_description", "") or "N/D"

        # Construir secciones de layout como texto.
        layout_html = ""
        for i, layout_path in enumerate(layout_paths):
            with open(layout_path, "rb") as f:
                layout_b64 = base64.b64encode(f.read()).decode("ascii")
            layout_html += (
                f'<div class="card"><h3>Layout — Imagen {i+1}</h3>'
                f'<img src="data:image/png;base64,{layout_b64}" '
                f'style="max-width:100%; border-radius:8px; border:1px solid #475569;" '
                f'alt="Layout {i+1}"></div>'
            )

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>SINCRO — Informe de Amiloidosis</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; max-width: 900px; margin: 0 auto; padding: 24px; }}
.header {{ background: linear-gradient(135deg, #1a3a5c, #0f172a); border-bottom: 3px solid #38bdf8; padding: 24px; text-align: center; border-radius: 12px; margin-bottom: 24px; }}
.header h1 {{ color: #38bdf8; font-size: 1.8rem; margin: 0; }}
.header .subtitle {{ color: #94a3b8; font-size: 0.95rem; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 20px; margin: 16px 0; border: 1px solid #475569; }}
.metric {{ font-size: 2.5rem; font-weight: 800; color: #38bdf8; }}
.metric.positive {{ color: #f87171; }}
.metric.equivocal {{ color: #fbbf24; }}
.metric.negative {{ color: #4ade80; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ background: #1a3a5c; color: white; padding: 8px; text-align: left; }}
td {{ padding: 8px; border-bottom: 1px solid #475569; }}
.footer {{ text-align: center; padding: 16px; border-top: 1px solid #475569; color: #94a3b8; font-size: 0.8rem; }}
</style>
</head>
<body>
<div class="header">
  <h1>SINCRO</h1>
  <div class="subtitle">Informe de Amiloidosis Cardíaca — Análisis planar</div>
  <div style="margin-top: 12px; font-size: 0.85rem; color: #94a3b8;">Paciente: {patient} · Fecha: {date} · Serie: {series}</div>
</div>
<div class="card">
  <h3>1. Imagen planar con ROIs</h3>
  <img src="data:image/png;base64,{img_b64}" style="max-width:100%; border-radius:8px; border:1px solid #475569;" alt="Imagen planar">
</div>
{layout_html}
<div class="card">
  <h3>2. Métrica principal: HMR</h3>
  <div class="metric {"positive" if result.hmr >= 1.5 else "equivocal" if result.hmr >= 1.0 else "negative"}">{result.hmr:.2f}</div>
  <table>
    <tr><th>Métrica</th><th>Valor</th><th>Referencia</th></tr>
    <tr><td>HMR (Heart-to-Mediastinum)</td><td>{result.hmr:.2f}</td><td>≥1.5 sugiere ATTR</td></tr>
    <tr><td>Cuentas cardíacas</td><td>{result.heart_counts:,.0f}</td><td></td></tr>
    <tr><td>Cuentas mediastinales</td><td>{result.mediastinum_counts:,.0f}</td><td></td></tr>
    <tr><td>Clasificación</td><td>{result.classification}</td><td></td></tr>
  </table>
</div>
<div class="card">
  <h3>3. Perugini visual score</h3>
  <p><b>Score {perugini}</b> — {PERUGINI_SCORES.get(perugini, 'N/D')}</p>
  <p style="font-size:0.85rem; color:#94a3b8;">Referencia: 0 = sin captación; 1 = leve; 2 = moderado (= hueso); 3 = intenso (> hueso).</p>
</div>
<div class="card">
  <h3>4. Interpretación clínica</h3>
  <p>El estudio muestra HMR de {result.hmr:.2f}. <b>{result.classification}</b></p>
  <p>Si el resultado es equívoco (HMR 1.0–1.5), considerar imagen SPECT/CT o repetir planar a 3 horas para descartar pool sanguíneo residual.</p>
  <p>La interpretación debe integrarse con laboratorio (cadenas livianas libres, proteínas monoclonales) y contexto clínico. El Perugini score ≥2 en presencia de gammapatía monoclonal ausente confirma ATTR.</p>
</div>
<div class="footer">
  Informe generado por SINCRO — Análisis de amiloidosis cardíaca.<br>
  Resultados orientativos para apoyo clínico.
</div>
</body>
</html>"""
        with open(html_path, "wb") as f:
            f.write(html.encode("utf-8"))

    # ── Paginación de layouts ───────────────────────────────────────

    def _prev_page(self):
        """Ir a la página anterior."""
        if self._page_offset > 0:
            self._page_offset -= self._current_layout_n
            self._rebuild_layout(force_layout=self._current_layout_n)

    def _next_page(self):
        """Ir a la página siguiente."""
        n = self._current_layout_n
        if self._page_offset + n < len(self._loaded_images):
            self._page_offset += n
            self._rebuild_layout(force_layout=n)

