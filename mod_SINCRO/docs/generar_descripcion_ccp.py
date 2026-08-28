"""Genera el documento institucional de capacidades de CCP en cinco páginas."""
from __future__ import annotations

import math
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(ROOT, "CCP_Descripcion_Capacidades_PI_ANMAT.pdf")
PAGE_W, PAGE_H = A4

NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#1F6E8C")
CYAN = colors.HexColor("#4FA3B8")
PALE = colors.HexColor("#EAF4F6")
INK = colors.HexColor("#1E2933")
MUTED = colors.HexColor("#5D6B78")
GREEN = colors.HexColor("#398564")
AMBER = colors.HexColor("#D69A2D")
RED = colors.HexColor("#C9504D")
WHITE = colors.white

styles = getSampleStyleSheet()
BODY = ParagraphStyle(
    "BodyCCP", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.15,
    leading=12.2, textColor=INK, alignment=TA_LEFT, spaceAfter=2.2 * mm,
)
SMALL = ParagraphStyle(
    "SmallCCP", parent=BODY, fontSize=7.8, leading=10.1, textColor=MUTED,
)
HEADING = ParagraphStyle(
    "HeadingCCP", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=15, leading=18, textColor=NAVY, spaceAfter=2.4 * mm,
)
SUBHEADING = ParagraphStyle(
    "SubheadingCCP", parent=styles["Heading3"], fontName="Helvetica-Bold",
    fontSize=10.2, leading=12.5, textColor=BLUE, spaceBefore=1.5 * mm,
    spaceAfter=1.2 * mm,
)
CALLOUT = ParagraphStyle(
    "CalloutCCP", parent=BODY, fontSize=8.7, leading=11.4, textColor=NAVY,
)
CENTER = ParagraphStyle(
    "CenterCCP", parent=BODY, alignment=TA_CENTER, fontSize=8.1, leading=10,
)


def paragraph(c: canvas.Canvas, text: str, x: float, y_top: float, width: float, style=BODY) -> float:
    item = Paragraph(text, style)
    _, height = item.wrap(width, PAGE_H)
    item.drawOn(c, x, y_top - height)
    return y_top - height - style.spaceAfter


def header(c: canvas.Canvas, page: int, title: str) -> float:
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 23 * mm, PAGE_W, 23 * mm, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(17 * mm, PAGE_H - 14 * mm, "CCP")
    c.setFont("Helvetica", 8.5)
    c.drawString(39 * mm, PAGE_H - 14 * mm, "Cardiac Control Panel")
    c.setFont("Helvetica", 8)
    c.drawRightString(PAGE_W - 17 * mm, PAGE_H - 14 * mm, title)
    c.setStrokeColor(CYAN)
    c.setLineWidth(2)
    c.line(17 * mm, PAGE_H - 25 * mm, PAGE_W - 17 * mm, PAGE_H - 25 * mm)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7.5)
    c.drawString(17 * mm, 10 * mm, "Descripción general y capacidades del software")
    c.drawRightString(PAGE_W - 17 * mm, 10 * mm, f"Página {page} de 5")
    return PAGE_H - 31 * mm


def section(c: canvas.Canvas, text: str, x: float, y: float, width: float) -> float:
    return paragraph(c, text, x, y, width, HEADING)


def sub(c: canvas.Canvas, text: str, x: float, y: float, width: float) -> float:
    return paragraph(c, text, x, y, width, SUBHEADING)


def bullet(c: canvas.Canvas, text: str, x: float, y: float, width: float) -> float:
    return paragraph(c, f"<font color='#1F6E8C'>●</font>&nbsp;&nbsp;{text}", x, y, width, BODY)


def rounded_box(c: canvas.Canvas, x: float, y: float, width: float, height: float, fill=PALE, stroke=CYAN) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.8)
    c.roundRect(x, y, width, height, 3 * mm, fill=1, stroke=1)


def flow_diagram(c: canvas.Canvas, x: float, y: float, width: float) -> None:
    labels = ["DICOM", "Control\nde calidad", "Procesamiento", "Análisis", "Informe y\nexportación"]
    gap = 5 * mm
    box_w = (width - gap * 4) / 5
    box_h = 19 * mm
    fills = [NAVY, BLUE, CYAN, GREEN, AMBER]
    for index, (label, fill) in enumerate(zip(labels, fills)):
        bx = x + index * (box_w + gap)
        c.setFillColor(fill)
        c.setStrokeColor(fill)
        c.roundRect(bx, y, box_w, box_h, 2.2 * mm, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 7.7)
        lines = label.split("\n")
        for line_index, line in enumerate(lines):
            c.drawCentredString(bx + box_w / 2, y + box_h / 2 + (3 - line_index * 8), line)
        if index < 4:
            ax = bx + box_w + 1 * mm
            ay = y + box_h / 2
            c.setStrokeColor(MUTED)
            c.setFillColor(MUTED)
            c.setLineWidth(1.2)
            c.line(ax, ay, ax + 3 * mm, ay)
            c.line(ax + 3 * mm, ay, ax + 1.6 * mm, ay + 1.2 * mm)
            c.line(ax + 3 * mm, ay, ax + 1.6 * mm, ay - 1.2 * mm)


def bullseye(c: canvas.Canvas, cx: float, cy: float, radius: float) -> None:
    palette = [RED, AMBER, colors.HexColor("#E6D64A"), GREEN, CYAN, BLUE]
    rings = [(0.75, 1.0, 6), (0.50, 0.75, 6), (0.25, 0.50, 4)]
    segment = 1
    for inner, outer, count in rings:
        for index in range(count):
            start = 90 - index * 360 / count
            extent = -360 / count
            c.setFillColor(palette[(segment - 1) % len(palette)])
            c.setStrokeColor(WHITE)
            c.setLineWidth(1)
            path = c.beginPath()
            points = []
            for step in range(13):
                angle = math.radians(start + extent * step / 12)
                points.append((cx + radius * outer * math.cos(angle), cy + radius * outer * math.sin(angle)))
            for step in range(12, -1, -1):
                angle = math.radians(start + extent * step / 12)
                points.append((cx + radius * inner * math.cos(angle), cy + radius * inner * math.sin(angle)))
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            path.close()
            c.drawPath(path, fill=1, stroke=1)
            mid = math.radians(start + extent / 2)
            rr = radius * (inner + outer) / 2
            c.setFillColor(NAVY)
            c.setFont("Helvetica-Bold", 6.8)
            c.drawCentredString(cx + rr * math.cos(mid), cy + rr * math.sin(mid) - 2, str(segment))
            segment += 1
    c.setFillColor(BLUE)
    c.setStrokeColor(WHITE)
    c.circle(cx, cy, radius * 0.25, fill=1, stroke=1)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 7)
    c.drawCentredString(cx, cy - 2, "17")


def page_one(c: canvas.Canvas) -> None:
    y = header(c, 1, "Identidad y finalidad")
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 27)
    c.drawString(17 * mm, y - 3 * mm, "CCP")
    c.setFont("Helvetica", 15)
    c.setFillColor(BLUE)
    c.drawString(17 * mm, y - 11 * mm, "Cardiac Control Panel")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8.5)
    c.drawString(17 * mm, y - 18 * mm, "Documento descriptivo para registro de propiedad intelectual y evaluación regulatoria")
    y -= 28 * mm

    rounded_box(c, 17 * mm, y - 31 * mm, PAGE_W - 34 * mm, 31 * mm)
    paragraph(
        c,
        "<b>CCP es un software de escritorio para visualizar, procesar y analizar estudios de medicina nuclear cardíaca.</b> Reúne en una misma interfaz la lectura DICOM, el control de calidad, la reconstrucción SPECT, la revisión del ciclo cardíaco, la segmentación, el análisis de sincronía mecánica, la perfusión y herramientas para amiloidosis cardíaca.",
        22 * mm, y - 5 * mm, PAGE_W - 44 * mm, CALLOUT,
    )
    y -= 39 * mm
    y = section(c, "1. Finalidad prevista", 17 * mm, y, PAGE_W - 34 * mm)
    y = paragraph(
        c,
        "El programa está pensado como una <b>herramienta de apoyo para profesionales capacitados</b> en medicina nuclear e imágenes médicas. Organiza la información, aplica cálculos reproducibles y presenta resultados cuantitativos y gráficos para facilitar la revisión de un estudio cardíaco.",
        17 * mm, y, PAGE_W - 34 * mm,
    )
    y = paragraph(
        c,
        "CCP no reemplaza el criterio médico y no emite por sí solo un diagnóstico definitivo. Los resultados deben revisarse junto con las imágenes originales, los controles de calidad, los antecedentes del paciente y el contexto clínico.",
        17 * mm, y, PAGE_W - 34 * mm,
    )
    y = sub(c, "Una plataforma, varios espacios de trabajo", 17 * mm, y, PAGE_W - 34 * mm)
    col_w = (PAGE_W - 39 * mm) / 2
    left_x = 17 * mm
    right_x = left_x + col_w + 5 * mm
    box_y = y - 47 * mm
    rounded_box(c, left_x, box_y, col_w, 45 * mm, colors.HexColor("#F4F8FA"), colors.HexColor("#B9D5DD"))
    rounded_box(c, right_x, box_y, col_w, 45 * mm, colors.HexColor("#F4F8FA"), colors.HexColor("#B9D5DD"))
    paragraph(c, "<b>Procesamiento cardíaco</b><br/>Reconstrucción, cine, ejes cardíacos, perfusión, función ventricular y análisis de sincronía.", left_x + 4 * mm, y - 6 * mm, col_w - 8 * mm, BODY)
    paragraph(c, "<b>Amiloidosis cardíaca</b><br/>Herramientas planares y SPECT/CT para cuantificación, fusión anatómica, segmentación y documentación.", right_x + 4 * mm, y - 6 * mm, col_w - 8 * mm, BODY)
    y = box_y - 8 * mm
    y = sub(c, "Usuarios previstos", 17 * mm, y, PAGE_W - 34 * mm)
    y = paragraph(c, "Médicos especialistas, físicos médicos, técnicos y otros profesionales entrenados en medicina nuclear. No está destinado al uso directo por pacientes ni al análisis autónomo sin supervisión.", 17 * mm, y, PAGE_W - 34 * mm)
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 7.8)
    c.drawString(17 * mm, 17 * mm, "Estado del producto: software en desarrollo y validación técnica/clínica. Fecha: 27/08/2026.")


def page_two(c: canvas.Canvas) -> None:
    y = header(c, 2, "Flujo de trabajo e ingreso de datos")
    y = section(c, "2. Del estudio al resultado", 17 * mm, y, PAGE_W - 34 * mm)
    y = paragraph(c, "El recorrido fue diseñado para que el operador pueda ver qué sucede en cada etapa, corregir lo necesario y conservar los parámetros importantes del procesamiento.", 17 * mm, y, PAGE_W - 34 * mm)
    flow_diagram(c, 17 * mm, y - 25 * mm, PAGE_W - 34 * mm)
    y -= 34 * mm
    col_w = (PAGE_W - 39 * mm) / 2
    left_x = 17 * mm
    right_x = left_x + col_w + 5 * mm
    left_y = y
    right_y = y
    left_y = sub(c, "Ingreso y normalización DICOM", left_x, left_y, col_w)
    for text in [
        "Lectura de estudios de medicina nuclear y CT utilizados por los módulos cardíacos.",
        "Reconocimiento de series multiframe, cantidad de gates, cortes, spacing y geometría.",
        "Desempaquetado de montajes y organización en volúmenes 3D o secuencias 4D.",
        "Verificación básica de coherencia temporal del latido y de la información disponible.",
    ]:
        left_y = bullet(c, text, left_x, left_y, col_w)
    left_y = sub(c, "Visualización", left_x, left_y, col_w)
    for text in [
        "Cine cardíaco por gate y navegación por cortes.",
        "Ventana, nivel, color, zoom, orientación e interpolación.",
        "Vistas en eje corto y ejes largos; montajes y comparación lado a lado.",
    ]:
        left_y = bullet(c, text, left_x, left_y, col_w)

    right_y = sub(c, "Control de calidad", right_x, right_y, col_w)
    for text in [
        "Revisión de proyecciones crudas, curvas de cuentas y sinogramas.",
        "Detección y corrección asistida de movimiento.",
        "Control del gatillado y advertencias ante datos incompletos o inconsistentes.",
        "Revisión de segmentaciones, orientación y correspondencia entre etapas.",
    ]:
        right_y = bullet(c, text, right_x, right_y, col_w)
    right_y = sub(c, "Reconstrucción SPECT", right_x, right_y, col_w)
    for text in [
        "Métodos analíticos e iterativos configurables.",
        "Filtros de pre y posprocesamiento, suavizado y reducción de ruido.",
        "Corrección de dispersión cuando la adquisición aporta la información necesaria.",
        "Generación de volúmenes gatillados y no gatillados.",
    ]:
        right_y = bullet(c, text, right_x, right_y, col_w)

    y = min(left_y, right_y) - 4 * mm
    rounded_box(c, 17 * mm, y - 25 * mm, PAGE_W - 34 * mm, 25 * mm, colors.HexColor("#FFF7E5"), AMBER)
    paragraph(c, "<b>Trazabilidad:</b> CCP permite utilizar presets y conserva parámetros relevantes del procesamiento. La finalidad es favorecer la repetibilidad, facilitar la auditoría y hacer visible cómo se obtuvo cada resultado.", 22 * mm, y - 5 * mm, PAGE_W - 44 * mm, CALLOUT)


def page_three(c: canvas.Canvas) -> None:
    y = header(c, 3, "Sincronía, función y perfusión")
    y = section(c, "3. Análisis cardíaco cuantitativo", 17 * mm, y, PAGE_W - 34 * mm)
    y = paragraph(c, "CCP analiza cómo cambia la actividad del miocardio a lo largo del ciclo cardíaco. Sobre la máscara del ventrículo izquierdo estudia la curva temporal de cada voxel y calcula su fase y amplitud mediante análisis armónico.", 17 * mm, y, PAGE_W - 34 * mm)

    chart_x = 17 * mm
    chart_y = y - 66 * mm
    chart_w = 88 * mm
    chart_h = 58 * mm
    rounded_box(c, chart_x, chart_y, chart_w, chart_h, colors.HexColor("#F7FAFB"), colors.HexColor("#C4DCE2"))
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(chart_x + 5 * mm, chart_y + chart_h - 7 * mm, "Distribución de fase ilustrativa")
    axis_x = chart_x + 9 * mm
    axis_y = chart_y + 11 * mm
    axis_w = chart_w - 16 * mm
    axis_h = chart_h - 24 * mm
    c.setStrokeColor(MUTED)
    c.line(axis_x, axis_y, axis_x + axis_w, axis_y)
    c.line(axis_x, axis_y, axis_x, axis_y + axis_h)
    values = [1, 2, 4, 8, 15, 25, 34, 40, 36, 27, 17, 9, 5, 3, 2, 1]
    bar_w = axis_w / len(values)
    for index, value in enumerate(values):
        height = axis_h * value / max(values)
        c.setFillColor(CYAN if index < 11 else AMBER)
        c.rect(axis_x + index * bar_w + 0.5, axis_y, bar_w - 1, height, fill=1, stroke=0)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 6.5)
    c.drawString(axis_x, axis_y - 4 * mm, "0°")
    c.drawCentredString(axis_x + axis_w / 2, axis_y - 4 * mm, "180°")
    c.drawRightString(axis_x + axis_w, axis_y - 4 * mm, "360°")

    bullseye(c, 151 * mm, chart_y + 29 * mm, 27 * mm)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7.2)
    c.drawCentredString(151 * mm, chart_y - 1 * mm, "Mapa polar ilustrativo de 17 segmentos")
    y = chart_y - 8 * mm

    col_w = (PAGE_W - 39 * mm) / 2
    left_x = 17 * mm
    right_x = left_x + col_w + 5 * mm
    left_y = sub(c, "Métricas de sincronía", left_x, y, col_w)
    for text in [
        "Desviación estándar de fase y ancho de banda.",
        "Entropía, fase media, fase pico e índice de asincronía.",
        "Histograma de fase, mapas sobre cortes y localización de activación tardía.",
        "Resumen por segmento y por territorio cardíaco.",
    ]:
        left_y = bullet(c, text, left_x, left_y, col_w)
    right_y = sub(c, "Perfusión y función", right_x, y, col_w)
    for text in [
        "Perfusión en cortes, mapas polares y cine polar.",
        "Integración visual entre captación y momento de contracción.",
        "Comparación de dos etapas compatibles, como reposo y esfuerzo.",
        "Estimación de volúmenes y fracción de eyección, identificada como función en validación.",
    ]:
        right_y = bullet(c, text, right_x, right_y, col_w)
    y = min(left_y, right_y) - 3 * mm
    rounded_box(c, 17 * mm, y - 25 * mm, PAGE_W - 34 * mm, 25 * mm)
    paragraph(c, "<b>Segmentación revisable:</b> el miocardio puede segmentarse automáticamente, por umbral o con intervención manual. El operador puede revisar cortes, modificar regiones de interés y volver a procesar antes de aceptar el resultado.", 22 * mm, y - 5 * mm, PAGE_W - 44 * mm, CALLOUT)


def page_four(c: canvas.Canvas) -> None:
    y = header(c, 4, "Amiloidosis, resultados e interoperabilidad")
    y = section(c, "4. Herramientas para amiloidosis cardíaca", 17 * mm, y, PAGE_W - 34 * mm)
    col_w = (PAGE_W - 39 * mm) / 2
    left_x = 17 * mm
    right_x = left_x + col_w + 5 * mm
    box_y = y - 67 * mm
    rounded_box(c, left_x, box_y, col_w, 65 * mm, colors.HexColor("#F7FAFB"), colors.HexColor("#B9D5DD"))
    rounded_box(c, right_x, box_y, col_w, 65 * mm, colors.HexColor("#F7FAFB"), colors.HexColor("#B9D5DD"))
    ly = sub(c, "Análisis planar", left_x + 5 * mm, y - 5 * mm, col_w - 10 * mm)
    for text in [
        "Ubicación y edición de regiones de interés.",
        "Relaciones cuantitativas de captación.",
        "Evaluación temporal del lavado cuando existen estudios compatibles.",
        "Escalas visuales y preparación de reporte.",
    ]:
        ly = bullet(c, text, left_x + 5 * mm, ly, col_w - 10 * mm)
    ry = sub(c, "Análisis SPECT/CT", right_x + 5 * mm, y - 5 * mm, col_w - 10 * mm)
    for text in [
        "Carga o reconstrucción del volumen SPECT.",
        "Carga de CT y registro espacial entre ambos estudios.",
        "Fusión multiplanar, orientación y ajuste manual.",
        "Segmentación y edición de máscara CT para apoyar la cuantificación.",
    ]:
        ry = bullet(c, text, right_x + 5 * mm, ry, col_w - 10 * mm)
    y = box_y - 8 * mm
    y = section(c, "5. Resultados y exportación", 17 * mm, y, PAGE_W - 34 * mm)
    y = paragraph(c, "CCP prepara resultados para lectura profesional, documentación y auditoría. Según el módulo y la información disponible, puede generar:", 17 * mm, y, PAGE_W - 34 * mm)
    col_y = y
    left_y = col_y
    right_y = col_y
    for text in [
        "Informes PDF con datos del estudio, parámetros, métricas, tablas, gráficos y notas de calidad.",
        "Imágenes de cortes, montajes, histogramas, mapas polares y paneles comparativos.",
        "Datos estructurados en JSON, CSV y planillas para revisión o análisis posterior.",
    ]:
        left_y = bullet(c, text, left_x, left_y, col_w)
    for text in [
        "Series DICOM derivadas en las funciones que admiten exportación.",
        "Presets y parámetros de procesamiento para repetir configuraciones.",
        "Registros técnicos de eventos, advertencias y errores para soporte y trazabilidad.",
    ]:
        right_y = bullet(c, text, right_x, right_y, col_w)
    y = min(left_y, right_y) - 4 * mm
    rounded_box(c, 17 * mm, y - 28 * mm, PAGE_W - 34 * mm, 28 * mm, colors.HexColor("#FFF7E5"), AMBER)
    paragraph(c, "<b>Importante:</b> las relaciones cuantitativas y clasificaciones son información de apoyo. La interpretación final requiere comprobar que la adquisición, la orientación, el registro, las regiones de interés y los parámetros sean adecuados para ese paciente y protocolo.", 22 * mm, y - 5 * mm, PAGE_W - 44 * mm, CALLOUT)


def page_five(c: canvas.Canvas) -> None:
    y = header(c, 5, "Seguridad, límites y características técnicas")
    y = section(c, "6. Controles de seguridad y uso responsable", 17 * mm, y, PAGE_W - 34 * mm)
    y = paragraph(c, "Antes de aceptar un resultado, el usuario debe verificar identidad y estudio, calidad del gatillado, movimiento, orientación, segmentación, correspondencia anatómica y coherencia de los parámetros. CCP muestra estados y advertencias y permite revisar o corregir varias etapas.", 17 * mm, y, PAGE_W - 34 * mm)
    col_w = (PAGE_W - 39 * mm) / 2
    left_x = 17 * mm
    right_x = left_x + col_w + 5 * mm
    left_y = sub(c, "Factores que pueden afectar el resultado", left_x, y, col_w)
    for text in [
        "Movimiento intenso o gatillado defectuoso.",
        "Baja estadística, artefactos o estudio incompleto.",
        "Orientación no reconocida o geometría inconsistente.",
        "Segmentación incorrecta o parámetros inadecuados.",
        "Diferencias de equipo, protocolo y reconstrucción.",
    ]:
        left_y = bullet(c, text, left_x, left_y, col_w)
    right_y = sub(c, "Límites actuales", right_x, y, col_w)
    for text in [
        "Las clasificaciones automáticas son orientativas y no constituyen diagnóstico.",
        "La exactitud depende de la calidad de los datos y de la revisión del operador.",
        "Las funciones avanzadas requieren validación clínica formal para su uso previsto definitivo.",
        "El producto no debe utilizarse como única base para una decisión asistencial.",
    ]:
        right_y = bullet(c, text, right_x, right_y, col_w)
    y = min(left_y, right_y) - 3 * mm
    y = section(c, "7. Arquitectura y aporte original", 17 * mm, y, PAGE_W - 34 * mm)
    y = paragraph(c, "CCP es una aplicación modular con interfaz gráfica de escritorio. Separa lectura DICOM, reconstrucción, segmentación, cálculos, visualización, persistencia y generación de informes. Esta arquitectura permite probar componentes, controlar versiones, mantener trazabilidad y ampliar funciones sin modificar todo el sistema.", 17 * mm, y, PAGE_W - 34 * mm)
    y = paragraph(c, "Su aporte original es la integración de un recorrido cardíaco completo dentro de un único panel: desde la adquisición y su control de calidad hasta la reconstrucción, la revisión dinámica, el análisis de fase, la perfusión, la comparación de estudios, la amiloidosis y la documentación final. Combina automatización con controles manuales para que el profesional pueda revisar qué hizo el sistema.", 17 * mm, y, PAGE_W - 34 * mm)
    y = sub(c, "Documentación necesaria para la evaluación regulatoria", 17 * mm, y, PAGE_W - 34 * mm)
    y = paragraph(c, "La versión que se presente deberá acompañarse con indicación de uso definitiva, especificación de requisitos, gestión de riesgos, verificación y validación, control de versiones, evaluación de usabilidad, ciberseguridad y evidencia de desempeño acorde con las prestaciones declaradas.", 17 * mm, y, PAGE_W - 34 * mm)
    rounded_box(c, 17 * mm, 22 * mm, PAGE_W - 34 * mm, 27 * mm, colors.HexColor("#FDEEEE"), RED)
    paragraph(c, "<b>Nota regulatoria:</b> este documento describe las capacidades técnicas del software en su estado actual. No implica autorización sanitaria, certificación de desempeño ni aprobación regulatoria. La indicación de uso y las prestaciones declaradas deberán coincidir con la versión sometida a evaluación y con su evidencia de validación.", 22 * mm, 44 * mm, PAGE_W - 44 * mm, CALLOUT)


def build() -> str:
    c = canvas.Canvas(OUTPUT, pagesize=A4)
    c.setTitle("CCP - Descripción general y capacidades del software")
    c.setAuthor("CCP - Cardiac Control Panel")
    c.setSubject("Documento descriptivo para propiedad intelectual y evaluación regulatoria")
    for page_fn in (page_one, page_two, page_three, page_four, page_five):
        page_fn(c)
        c.showPage()
    c.save()
    return OUTPUT


if __name__ == "__main__":
    print(build())