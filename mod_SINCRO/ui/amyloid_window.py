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
    QComboBox, QWidget, QSizePolicy, QMessageBox,
)
import os

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

        # Perugini pre-cargado: sugiere score basado en HMR automático.
        self._perugini_combo = QComboBox()
        for score, desc in PERUGINI_SCORES.items():
            self._perugini_combo.addItem(f"{score} — {desc}", score)
        # Pre-cargar con score sugerido.
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
            suggested = 3 if result.hmr >= 1.5 else (2 if result.hmr >= 1.0 else 0)
            self._perugini_combo.setCurrentIndex(suggested)
        except Exception:
            self._perugini_combo.setCurrentIndex(0)
        root.addWidget(self._perugini_combo)

        # Botones.
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

        self._update_hmr(0, 0, 0, 0)

    def get_report_image(self) -> np.ndarray:
        """Renderiza la imagen con los ROIs como array RGB para el informe."""
        img = self._image.copy()
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
            draw.text((x1 + 4, int(roi["cy"])), roi["name"], fill=color)
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
            result = compute_hmr(self._image, roi_h, roi_m)
            perugini = int(self._perugini_combo.currentData())
            report_img = self.get_report_image()
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_demo")
            os.makedirs(output_dir, exist_ok=True)
            # Guardar imagen con ROIs.
            img_path = os.path.join(output_dir, "amyloid_planar.png")
            from PIL import Image
            Image.fromarray(report_img).save(img_path, "PNG")
            # PDF
            pdf_path = os.path.join(output_dir, "informe_amyloid.pdf")
            self._generate_pdf(pdf_path, img_path, result, perugini)
            # HTML
            html_path = os.path.join(output_dir, "informe_amyloid.html")
            self._generate_html(html_path, img_path, result, perugini)
            QMessageBox.information(
                self, "SINCRO — Amyloidosis",
                f"Informe generado:\nPDF: {pdf_path}\nHTML: {html_path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO — Amyloidosis", f"Error al generar informe:\n{exc}")

    def _generate_pdf(self, pdf_path, img_path, result, perugini):
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

    def _generate_html(self, html_path, img_path, result, perugini):
        """Genera el informe HTML de amiloidosis."""
        import base64
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")
        patient = getattr(self._study, "patient_name", "") or "N/D"
        date = getattr(self._study, "study_date", "") or "N/D"
        series = getattr(self._study, "series_description", "") or "N/D"
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

