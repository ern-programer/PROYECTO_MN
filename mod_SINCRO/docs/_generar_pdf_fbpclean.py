# -*- coding: utf-8 -*-
"""Genera el PDF técnico-científico de FBP_CLEAN en docs/."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.lib import colors

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "FBP_CLEAN_fundamento.pdf")

styles = getSampleStyleSheet()
title = ParagraphStyle("t", parent=styles["Title"], fontSize=18, alignment=TA_CENTER, spaceAfter=6)
sub = ParagraphStyle("s", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=14)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1a3a5c"))
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#2c5f8a"))
body = ParagraphStyle("b", parent=styles["Normal"], fontSize=9.5, leading=13, alignment=TA_JUSTIFY, spaceAfter=5)
eq = ParagraphStyle("eq", parent=styles["Normal"], fontSize=9.5, leading=14, alignment=TA_CENTER, fontName="Courier", spaceAfter=5, backColor=colors.HexColor("#f4f4f4"))
ref = ParagraphStyle("r", parent=styles["Normal"], fontSize=8.5, leading=11, leftIndent=14, spaceAfter=2)

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=18*mm, bottomMargin=18*mm,
                        title="FBP_CLEAN - Fundamento fisico-matematico")
S = []

S.append(Paragraph("FBP_CLEAN", title))
S.append(Paragraph("Denoising de proyecciones SPECT y realce de cavidad por resta ponderada "
                   "para perfusión miocárdica de mitad de tiempo / mitad de dosis", sub))
S.append(Paragraph("Documento técnico-científico · Proyecto SINCRO · 2026", sub))

S.append(Paragraph("1. Motivación clínica y física", h1))
S.append(Paragraph(
    "La perfusión miocárdica SPECT con protocolos de tiempo o dosis reducidos sufre una degradación "
    "característica: al reconstruir por retroproyección filtrada (FBP), el ruido de Poisson presente en "
    "las proyecciones (sinograma) se transforma, tras la retroproyección, en <b>estrías radiales</b> "
    "(<i>streak artifacts</i>) que ensucian el fondo y reducen el contraste cavidad/miocardio. El objetivo "
    "de FBP_CLEAN es recuperar calidad diagnóstica atacando el ruido <b>en su dominio físico de origen</b> "
    "(las proyecciones, donde la estadística es Poisson pura) en lugar de hacerlo sobre la imagen ya "
    "reconstruida, donde el ruido ya se convirtió en artefacto estructurado.", body))

S.append(Paragraph("2. Por qué el denoising debe ser pre-reconstrucción", h1))
S.append(Paragraph(
    "El error experimental clave (banco de pruebas 022) fue constatar que un denoiser espacial aplicado "
    "<b>después</b> del FBP no elimina las estrías: una vez formadas, las estrías son una estructura "
    "coherente de baja frecuencia que cualquier suavizado local confunde con señal anatómica, por lo que "
    "eliminarlas implica difuminar también la cavidad y engordar aparentemente la pared miocárdica. "
    "La física del problema impone actuar antes: en el sinograma el ruido es granular e incorrelado "
    "(Poisson), y un filtrado que preserve los bordes del contorno cardíaco puede reducirlo sin generar "
    "estrías tras la retroproyección.", body))
S.append(Paragraph(
    "Este principio coincide con la literatura de reconstrucción de bajo conteo: los métodos comerciales "
    "de mitad de tiempo (p.ej. Wide Beam Reconstruction / UltraSPECT; DePuey 2008–2011) controlan el ruido "
    "<i>durante</i> la reconstrucción y no con un post-filtro, y los trabajos de Ali (2009) y Armstrong "
    "(2012) muestran que la recuperación de resolución sólo es útil cuando se la acompaña de control de "
    "ruido (regularización MAP).", body))

S.append(Paragraph("3. Etapa 1 — Denoising bilateral del sinograma", h1))
S.append(Paragraph(
    "Cada proyección p(θ) (imagen 2D H×W del detector) se filtra con un <b>filtro bilateral</b>, que "
    "promedia vecinos ponderando simultáneamente por cercanía espacial y por similitud de intensidad:", body))
S.append(Paragraph(
    "p&#770;(x) = (1/W) · Σ<sub>y</sub> p(y) · exp(−|x−y|²/2σ<sub>s</sub>²) · exp(−(p(x)−p(y))²/2σ<sub>c</sub>²)", eq))
S.append(Paragraph(
    "El término de rango exp(−(Δp)²/2σ<sub>c</sub>²) anula la contribución de vecinos cuya intensidad difiere "
    "(bordes del corazón), de modo que el filtro suaviza el ruido granular del fondo y del miocardio "
    "homogéneo <b>sin desplazar ni difuminar los bordes</b> del contorno cardíaco. Calibración experimental "
    "(banco 025): σ<sub>c</sub> = 0.04 (sobre proyección normalizada a su máximo), σ<sub>s</sub> = 1.5 píxeles. "
    "Valores σ<sub>c</sub> &gt; 0.08 difuminan en exceso; el punto óptimo se sitúa en 0.04–0.08.", body))

S.append(Paragraph("4. Etapa 2 — Realce de cavidad por resta ponderada", h1))
S.append(Paragraph(
    "La segunda etapa parte de una observación empírica del equipo clínico: restar a la imagen nítida una "
    "fracción de la imagen fuertemente suavizada <b>abre la cavidad</b> y afila los bordes. Formalmente es un "
    "<b>unsharp masking</b> (realce de alta frecuencia) clásico, aquí aplicado al volumen reconstruido:", body))
S.append(Paragraph("V<sub>out</sub> = max( V<sub>nítido</sub> − k · V<sub>difuso</sub>, 0 )", eq))
S.append(Paragraph(
    "donde V<sub>nítido</sub> = FBP(p&#770;<sub>σc=0.04</sub>) y V<sub>difuso</sub> = FBP(p&#770;<sub>σc=0.24</sub>). "
    "Como ambos volúmenes provienen de las mismas proyecciones, comparten escala de cuentas y geometría; su "
    "diferencia aísla la componente de alta frecuencia (bordes, pared, cavidad) que, restada con peso k, "
    "realza la estructura. El factor k (default 0.5, rango útil 0.3–0.7) equilibra realce y amplificación "
    "de ruido de fondo: a mayor k, mayor apertura de la cavidad y mejor definición de defectos de perfusión, "
    "a costa de más ruido granular.", body))
S.append(Paragraph(
    "<b>Relevancia diagnóstica:</b> el mecanismo que abre la cavidad realza por igual los <b>defectos de "
    "perfusión</b> (zonas de menor captación dentro del miocardio), que es precisamente el objetivo clínico "
    "de la perfusión miocárdica. No se inventa información: se amplifica la señal de borde ya presente.", body))

S.append(Paragraph("5. Estimación del nivel de ruido por sustracción (calibración)", h1))
S.append(Paragraph(
    "El nivel de ruido de un estudio de bajo conteo puede estimarse sin supuestos de modelo usando un par "
    "de adquisiciones del mismo paciente (p.ej. 5 s y 10 s por proyección): tras escalar ambas a igual "
    "actividad total, la diferencia entre ellas es, en buena aproximación, ruido puro sin estructura "
    "anatómica (verificado experimentalmente, banco 024). Su desviación estándar relativa a la señal "
    "cuantifica el ratio ruido/señal del estudio (≈0.48 en el caso de prueba) y sirve para fijar la fuerza "
    "del filtrado de forma objetiva, análogamente a la estimación de potencia de ruido en el filtro de Wiener.", body))

S.append(Paragraph("6. Métricas y validación", h1))
S.append(Paragraph(
    "La evaluación usa: (i) <b>limpieza de fondo</b> = desviación estándar en regiones sin actividad cardíaca "
    "(mide estrías); (ii) <b>CNR cavidad/pared</b>; y (iii) juicio visual experto sobre el par 5 s/10 s. Se "
    "constató que el RMSE píxel-a-píxel contra la referencia de alto conteo <b>no</b> es una métrica válida: "
    "premia la imagen ruidosa sin procesar porque penaliza el suavizado aunque éste mejore la legibilidad "
    "diagnóstica. La validación debe ser por contraste de estructuras y criterio clínico, no por fidelidad "
    "de píxel.", body))

S.append(Paragraph("7. Originalidad y relación con el estado del arte", h1))
S.append(Paragraph(
    "FBP_CLEAN no reproduce ningún algoritmo comercial. Combina: (a) denoising bilateral del sinograma con "
    "parámetros calibrados por sustracción de un par de conteos; (b) reconstrucción FBP estándar; y "
    "(c) realce por resta ponderada de dos reconstrucciones del mismo sinograma filtrado a distinta fuerza. "
    "La secuencia concreta, la calibración por sustracción y la aplicación del realce sobre el volumen FBP "
    "para perfusión de mitad de tiempo constituyen una contribución propia construida sobre matemática "
    "publicada (filtro bilateral, unsharp masking, estimación de ruido tipo Wiener). No utiliza código ni "
    "parámetros de Evolution (Philips), Astonish (GE) ni WBR (UltraSPECT), que se citan sólo como referencia "
    "conceptual del estado del arte.", body))

S.append(Paragraph("8. Resultados experimentales (banco de pruebas)", h1))
data = [
    ["Prueba", "Configuración", "Hallazgo"],
    ["022", "Denoise espacial post-FBP", "No quita estrías; descartado"],
    ["023", "Bilateral en sinograma (pre-FBP)", "Bajan las estrías, cavidad preservada"],
    ["024", "Resta 10s−5s", "Aísla ruido puro; RMSE no es métrica válida"],
    ["025", "Barrido σc bilateral", "Óptimo 0.04–0.08; >0.08 difumina"],
    ["026/027", "Realce nítido − k·difuso", "Cavidad y defectos realzados; k=0.5 default"],
]
t = Table(data, colWidths=[18*mm, 62*mm, 90*mm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3f8")]),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
S.append(t)

S.append(Paragraph("9. Referencias", h1))
for r in [
    "Ali I, Ruddy TD, et al. Half-time SPECT myocardial perfusion imaging with attenuation correction. J Nucl Med 2009;50:554-62. PMID 19289436.",
    "DePuey EG, et al. OSEM and wide beam reconstruction half-time gated myocardial perfusion SPECT. J Nucl Cardiol 2008;15:547-63. PMID 18674723.",
    "DePuey EG, et al. Full-time myocardial perfusion SPECT vs wide beam reconstruction half-time and half-dose. J Nucl Cardiol 2011;18:273-80. PMID 21287370.",
    "Armstrong IS, et al. Reduced-count myocardial perfusion SPECT with resolution recovery. Nucl Med Commun 2012;33:121-9. PMID 22107994.",
    "Marcassa C, et al. Wide beam reconstruction for half-dose or half-time cardiac gated SPECT. Eur J Nucl Med Mol Imaging 2011;38:499-508. PMID 21069317.",
    "Tomasi C, Manduchi R. Bilateral filtering for gray and color images. ICCV 1998.",
    "Green PJ. Bayesian reconstructions from emission tomography data using a modified EM algorithm. IEEE Trans Med Imaging 1990;9:84-93.",
]:
    S.append(Paragraph(r, ref))

S.append(Spacer(1, 8))
S.append(Paragraph(
    "Documento de trabajo — base para discusión de originalidad. La registración de propiedad intelectual "
    "requiere búsqueda de anterioridad formal por un agente de PI.", sub))

doc.build(S)
print("OK ->", OUT)
