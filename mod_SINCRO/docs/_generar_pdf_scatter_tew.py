# -*- coding: utf-8 -*-
"""Genera el PDF de la nota sobre corrección de scatter TEW en docs/.

Nota técnica de una carilla A4: por qué el scatter entra en la ventana del
fotopico y cómo lo estima/resta la ventana inferior (TEW). Lenguaje científico
con un punto de informalidad. Sin nombrar software comercial de terceros.
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer)
from reportlab.lib import colors

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "SCATTER_TEW_nota.pdf")

styles = getSampleStyleSheet()
title = ParagraphStyle("t", parent=styles["Title"], fontSize=16, alignment=TA_CENTER, spaceAfter=4)
sub = ParagraphStyle("s", parent=styles["Normal"], fontSize=9.5, alignment=TA_CENTER,
                     textColor=colors.HexColor("#555555"), spaceAfter=12)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=11, spaceBefore=7, spaceAfter=3,
                    textColor=colors.HexColor("#1a3a5c"))
body = ParagraphStyle("b", parent=styles["Normal"], fontSize=9, leading=11.5,
                      alignment=TA_JUSTIFY, spaceAfter=4)
eq = ParagraphStyle("eq", parent=styles["Normal"], fontSize=9, leading=12, alignment=TA_CENTER,
                    fontName="Courier", spaceBefore=2, spaceAfter=4,
                    backColor=colors.HexColor("#f4f4f4"))
ref = ParagraphStyle("r", parent=styles["Normal"], fontSize=8, leading=10,
                     leftIndent=14, spaceAfter=1)


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                        topMargin=12*mm, bottomMargin=12*mm,
                        title="Por qué el scatter entra en el fotopico (nota TEW)")
S = []

S.append(Paragraph("El fotón no sabe de ventanas", title))
S.append(Paragraph("Por qué la dispersión Compton se cuela en la ventana del fotopico y cómo "
                   "estimarla con una ventana inferior (método TEW) en SPECT cardíaco", sub))
S.append(Paragraph("Nota técnica · Proyecto SINCRO · 2026", sub))

# ---------------------------------------------------------------- la pregunta
S.append(Paragraph("La pregunta razonable", h1))
S.append(Paragraph(
    "Si el fotopico del <sup>99m</sup>Tc está en 140 keV y adquirimos con una ventana de ±10% "
    "(126–154 keV), los fotones que perdieron energía por dispersión Compton deberían quedar afuera. "
    "La intuición es correcta: el scatter de <b>gran ángulo</b> queda excluido. El problema es que en la "
    "práctica entre el 20% y el 40% de las cuentas del fotopico son fotones dispersados. ¿Cómo entran, "
    "si la ventana los debería rechazar? Por tres puertas distintas, y ninguna es mala calibración.", body))

# ---------------------------------------------------------------- puertas
S.append(Paragraph("Puerta 1: el detector no mide la energía, la estima", h1))
S.append(Paragraph(
    "El cristal de NaI(Tl) convierte la energía del fotón en luz, y la cantidad de luz fluctúa "
    "estadísticamente: la <b>resolución energética</b> es del orden del 10% (FWHM) a 140 keV. El espectro "
    "medido no es una raya, es una campana con σ<sub>E</sub> ≈ 6–8 keV. Sus colas cruzan los bordes de la "
    "ventana en ambos sentidos: un fotón dispersado que en verdad tiene 118 keV puede medirse como 128 keV "
    "y entrar. La ventana no rechaza fotones por su historia; solo por el número que el detector reportó, "
    "y ese número es ruidoso.", body))

S.append(Paragraph("Puerta 2: el Compton de ángulo chico casi no pierde energía", h1))
S.append(Paragraph(
    "La energía que conserva un fotón tras dispersarse un ángulo θ es", body))
S.append(Paragraph("E' = E / (1 + (E/511 keV)·(1 − cos θ))", eq))
S.append(Paragraph(
    "Para E = 140 keV, una dispersión de 30° deja al fotón en ~133 keV: <b>dentro de la ventana</b>; una de "
    "45° lo deja en ~127 keV: todavía dentro. Recién cerca de 50°–60° el fotón cae por debajo de 126 keV. "
    "O sea, la ventana de ±10% acepta todo el scatter de ángulo menor a ~50°, que es justamente el más "
    "probable en tejido (la sección eficaz de Klein–Nishina favorece ángulos chicos) y el más dañino: su "
    "distribución espacial casi coincide con la de los primarios, así que no se lo puede separar ni por "
    "energía ni por posición.", body))

S.append(Paragraph("Puerta 3: el propio detector dispersa", h1))
S.append(Paragraph(
    "Un fotón puede hacer Compton <b>dentro del cristal</b> y escapar depositando solo una fracción de su "
    "energía; una parte de esos eventos cae dentro de la ventana. Lo mismo con el backscatter desde los "
    "fotomultiplicadores y el blindaje.", body))

# ---------------------------------------------------------------- TEW
S.append(Paragraph("La solución pragmática: medir el scatter donde no hay otra cosa", h1))
S.append(Paragraph(
    "Como no podemos impedir que el scatter entre, lo estimamos. El método de la <b>ventana energética "
    "triple (TEW)</b> adquiere simultáneamente una ventana angosta justo debajo del fotopico, donde "
    "prácticamente no llegan fotones primarios:", body))
S.append(Paragraph("EM: 126–154 keV (fotopico, ±10%)&nbsp;&nbsp;&nbsp;·&nbsp;&nbsp;&nbsp;SC: 114–126 keV (scatter)", eq))
S.append(Paragraph(
    "El contenido de la ventana SC es scatter casi puro, y se asume que el scatter que contaminó el "
    "fotopico es proporcional a él. La corrección es una resta ponderada píxel a píxel, aplicada sobre las "
    "proyecciones antes de reconstruir:", body))
S.append(Paragraph("P<sub>corr</sub> = P<sub>EM</sub> − k · P<sub>SC</sub>,&nbsp;&nbsp;&nbsp;"
                   "k ≈ W<sub>EM</sub> / (2·W<sub>SC</sub>)", eq))
S.append(Paragraph(
    "Para las ventanas del ejemplo (W<sub>EM</sub> = 28 keV, W<sub>SC</sub> = 12 keV) resulta k ≈ 1.17. "
    "En la práctica el k teórico es un punto de partida: el valor fino se calibra con fantoma o con un "
    "estudio real, porque la proporción exacta depende de la geometría del paciente y del colimador. La "
    "hipótesis de proporcionalidad no es exacta —el scatter de ángulo chico, que es el que entra al "
    "fotopico, está algo sub-representado en la ventana inferior— pero alcanza para quitar el pedestal "
    "suave que rellena la cavidad ventricular, que es lo que clínicamente importa.", body))

# ---------------------------------------------------------------- moraleja
S.append(Paragraph("Moraleja", h1))
S.append(Paragraph(
    "La ventana del fotopico no es una puerta con guardia: es un tamiz con agujeros del tamaño justo para "
    "que pase el scatter de ángulo chico, más el ruido energético del detector, más las dispersiones "
    "internas del cristal. Aceptarlo y medirlo en una ventana vecina es más barato y más honesto que "
    "suponer que no existe.", body))

S.append(Spacer(1, 4))
S.append(Paragraph("Referencias", h1))
S.append(Paragraph("Klein O, Nishina Y. <i>Z Phys</i> 1929;52:853 — sección eficaz Compton.", ref))
S.append(Paragraph("Ogawa K et al. <i>IEEE Trans Med Imaging</i> 1991;10:408 — método TEW para SPECT.", ref))
S.append(Paragraph("Zaidi H, Koral KF. <i>Eur J Nucl Med Mol Imaging</i> 2004 — revisión de corrección de scatter en SPECT.", ref))

doc.build(S)
print(f"PDF generado: {OUT}")
