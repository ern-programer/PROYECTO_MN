# -*- coding: utf-8 -*-
"""Ventana de amiloidosis cardíaca: imagen planar + ROI draggable + HMR + Perugini.

Permite dibujar dos ROIs circulares (corazón y mediastino contralateral).
Calcula HMR (Heart-to-Mediastinum Ratio) y muestra la clasificación.

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
    QComboBox, QWidget, QSizePolicy,
)

from core.amyloid_planar import ROICircle, compute_hmr, PERUGINI_SCORES


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
        # Doble clic = ajustar radio del ROI bajo el cursor.
        scale = self._scale()
        ox = (self.width() - self._image.shape[1] * scale) // 2
        oy = (self.height() - self._image.shape[0] * scale) // 2
        for i, roi in enumerate(self._rois):
            rcx = event.position().x()
            rcy = event.position().y()
            dist = np.sqrt((rcx - ox - roi["cx"] * scale) ** 2 + (rcy - oy - roi["cy"] * scale) ** 2)
            if dist < roi["radius"] * 1.5 * scale:
                new_radius = (dist / scale) if event.position() else roi["radius"]
                self._rois[i]["radius"] = max(3.0, min(64.0, new_radius))
                self.roiChanged.emit(i, self._rois[i]["cy"], self._rois[i]["cx"], self._rois[i]["radius"])
                break
        self.update()

    def _scale(self) -> float:
        h, w = self._image.shape
        ww, wh = self.width(), self.height()
        return min(ww / max(1, w), wh / max(1, h)) * self._zoom


class AmyloidWindow(QDialog):
    """Ventana de amiloidosis: imagen planar + ROIs + HMR + Perugini."""

    def __init__(self, parent=None, image=None, study=None):
        super().__init__(parent)
        self.setWindowTitle("SINCRO — Amiloidosis")
        self.resize(900, 640)
        self._image = image
        self._study = study

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Info del paciente.
        patient = getattr(study, "patient_name", "") or "N/D"
        date = getattr(study, "study_date", "") or "N/D"
        series = getattr(study, "series_description", "") or "N/D"
        self._info_lbl = QLabel(f"Paciente: {patient} · Fecha: {date} · Serie: {series}")
        root.addWidget(self._info_lbl)

        # Widget de ROI.
        self._roi_widget = ROIDragWidget(image)
        self._roi_widget.roiChanged.connect(self._update_hmr)
        root.addWidget(self._roi_widget, 1)

        # Resultado.
        self._lbl_hmr = QLabel("HMR = N/D")
        self._lbl_hmr.setStyleSheet("font-size: 16px; font-weight: bold; color: #e2e8f0;")
        root.addWidget(self._lbl_hmr)

        self._lbl_class = QLabel("")
        self._lbl_class.setStyleSheet("font-size: 12px; color: #94a3b8;")
        root.addWidget(self._lbl_class)

        # Perugini.
        self._perugini_combo = QComboBox()
        for score, desc in PERUGINI_SCORES.items():
            self._perugini_combo.addItem(f"{score} — {desc}", score)
        self._perugini_combo.setCurrentIndex(0)
        root.addWidget(self._perugini_combo)

        # Botones.
        btns = QHBoxLayout()
        btn_reset = QPushButton("Reset ROIs")
        btn_reset.clicked.connect(self._reset_rois)
        btns.addWidget(btn_reset)
        btns.addStretch(1)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_close)
        root.addLayout(btns)

        self._update_hmr(0, 0, 0, 0)

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
            result = compute_hmr(self._image, roi_h, roi_m)
            self._lbl_hmr.setText(f"HMR = {result.hmr:.2f}")
            self._lbl_class.setText(result.classification)
            color = "#f87171" if result.hmr >= 1.5 else ("#fbbf24" if result.hmr >= 1.0 else "#4ade80")
            self._lbl_hmr.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        except Exception as exc:
            self._lbl_hmr.setText("HMR = N/D")
            self._lbl_class.setText(f"Error: {exc}")

    def _reset_rois(self):
        h, w = self._image.shape
        self._roi_widget._rois[0]["cy"] = 0.4 * h
        self._roi_widget._rois[0]["cx"] = 0.4 * w
        self._roi_widget._rois[0]["radius"] = 12.0
        self._roi_widget._rois[1]["cy"] = 0.6 * h
        self._roi_widget._rois[1]["cx"] = 0.6 * w
        self._roi_widget._rois[1]["radius"] = 12.0
        self._roi_widget.update()
        self._update_hmr(0, 0, 0, 0)
