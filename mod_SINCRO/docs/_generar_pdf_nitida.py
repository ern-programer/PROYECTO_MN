r"""Generador del documento tecnico-cientifico de NITIDA (OmniRes).

Produce un PDF autocontenido que justifica fisica y matematicamente el motor de
recuperacion de resolucion NITIDA (OmniRes): modelo colimador-detector, PSF
dependiente de profundidad, modelado dentro de OSEM/MLEM, compromiso
resolucion-ruido, extension fan-beam, auto-configuracion multi-fabricante desde
DICOM, tabla de colimadores, extension a planar multi-isotopo, y un ANEXO con la
comparativa frente a competidores. Software PROPIETARIO (no codigo abierto).

Ejecutar con el venv de mod_SINCRO (tiene reportlab + matplotlib):
    & .\.venv\Scripts\python.exe docs\_generar_pdf_nitida.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
    HRFlowable,
)
from reportlab.lib.utils import ImageReader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ASSETS = os.path.join(HERE, "_nitida_assets")
os.makedirs(ASSETS, exist_ok=True)
OUT_PDF = os.path.join(HERE, "NITIDA_OmniRes_fundamento_tecnico.pdf")

FWHM_TO_SIGMA = 1.0 / 2.354820045

# ---------------------------------------------------------------- colimadores
# Espejo de core/collimator_specs.py (representativos, datasheets publicos).
COLLIMATORS = [
    # (fabricante, nombre, geometria, d, L, septa, R_int)
    ("GE", "LEHR", "paralelo", 1.5, 35.0, 0.20, 3.8),
    ("GE", "LEGP/LEAP", "paralelo", 1.9, 35.0, 0.20, 3.8),
    ("GE", "STARCAM-GP ('99')", "paralelo", 1.9, 32.0, 0.20, 4.5),
    ("Siemens", "LEHR", "paralelo", 1.11, 24.05, 0.16, 3.8),
    ("Philips", "VXGP", "paralelo", 1.4, 27.0, 0.20, 3.6),
    ("GVI", "NGSPECT (OnePass)", "fan-beam", 1.5, 35.0, 0.20, 3.5),
]


def r_geom(d, L, b):
    return d * (L + b) / L


def r_sys(d, L, r_int, b):
    return np.sqrt(r_geom(d, L, b) ** 2 + r_int ** 2)


# ============================================================ ecuaciones (mathtext)

def render_eq(tex: str, name: str, fontsize: int = 20) -> str:
    path = os.path.join(ASSETS, f"eq_{name}.png")
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0.0, 0.0, tex, fontsize=fontsize, color="#111827")
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return path


def eq_flowable(path: str, max_w_mm: float = 150.0):
    ir = ImageReader(path)
    iw, ih = ir.getSize()
    # px @ 200 dpi -> puntos (72 dpi)
    w_pt = iw / 200.0 * 72.0
    h_pt = ih / 200.0 * 72.0
    max_w_pt = max_w_mm * mm
    if w_pt > max_w_pt:
        scale = max_w_pt / w_pt
        w_pt *= scale
        h_pt *= scale
    return Image(path, width=w_pt, height=h_pt, hAlign="CENTER")


# ============================================================ figuras

def fig_rsys_vs_distance() -> str:
    path = os.path.join(ASSETS, "fig_rsys.png")
    b = np.linspace(0, 300, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for man, name, geom, d, L, septa, r_int in COLLIMATORS:
        if geom != "paralelo":
            continue
        ax.plot(b, r_sys(d, L, r_int, b), lw=2.0, label=f"{man} {name}")
    ax.set_xlabel("Distancia fuente-colimador  b  [mm]")
    ax.set_ylabel("Resolucion del sistema  R_sys  [mm FWHM]")
    ax.set_title("PSF crece linealmente con la distancia (colimadores paralelos)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_sigma_by_depth() -> str:
    path = os.path.join(ASSETS, "fig_sigma_depth.png")
    # Estudio representativo: GE LEHR, radio 268 mm, pixel 6.78 mm, matriz 64.
    d, L, r_int = 1.5, 35.0, 3.8
    radius_mm, pixel_mm, n = 268.0, 6.78, 64
    rows = np.arange(n)
    center = (n - 1) / 2.0
    b = np.clip(radius_mm - (rows - center) * pixel_mm, 0.0, None)
    sigma_px = r_sys(d, L, r_int, b) * FWHM_TO_SIGMA / pixel_mm
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(rows, sigma_px, lw=2.2, color="#2563eb")
    ax.fill_between(rows, sigma_px, alpha=0.15, color="#2563eb")
    ax.set_xlabel("Fila del corte rotado  (mayor = lado detector)")
    ax.set_ylabel("sigma de la PSF  [px]")
    ax.set_title("Sigma por profundidad  (LEHR, orbita 268 mm, pixel 6.78 mm)", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_planar_filters() -> str:
    path = os.path.join(ASSETS, "fig_planar_filters.png")
    f = np.linspace(0.0, 0.5, 400)  # frecuencia (ciclos/px, Nyquist=0.5)
    # MTF gaussiana de un colimador: MTF(f)=exp(-2 pi^2 sigma^2 f^2)
    sigma = 2.2  # px, ilustrativo (whole-body a distancia tabla-detector)
    mtf = np.exp(-2 * np.pi ** 2 * sigma ** 2 * f ** 2)
    n_metz = 8
    metz = (1 - (1 - mtf ** 2) ** n_metz) / np.clip(mtf, 1e-6, None)
    nsr = 0.02
    wiener = mtf / (mtf ** 2 + nsr)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(f, mtf, lw=2.0, label="MTF colimador (borrosidad)", color="#6b7280")
    ax.plot(f, metz, lw=2.2, label=f"Realce Metz (n={n_metz})", color="#dc2626")
    ax.plot(f, wiener, lw=2.2, label="Restauracion Wiener", color="#059669")
    ax.axhline(1.0, color="#9ca3af", lw=0.8, ls="--")
    ax.set_xlabel("Frecuencia espacial  [ciclos/px]  (Nyquist = 0.5)")
    ax.set_ylabel("Ganancia")
    ax.set_title("Filtros de restauracion 2D para PLANAR (recuperan la MTF del colimador)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 3.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================ documento

def build():
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.2,
                          leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15,
                        textColor=colors.HexColor("#1e3a8a"), spaceBefore=12, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5,
                        textColor=colors.HexColor("#1d4ed8"), spaceBefore=8, spaceAfter=4)
    small = ParagraphStyle("small", parent=body, fontSize=8.6, leading=11,
                           textColor=colors.HexColor("#4b5563"))
    caption = ParagraphStyle("caption", parent=small, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=24,
                           textColor=colors.HexColor("#111827"), leading=28)
    subtitle = ParagraphStyle("subtitle", parent=styles["Title"], fontSize=13,
                              textColor=colors.HexColor("#374151"), leading=17, spaceAfter=4)
    proprietary = ParagraphStyle("prop", parent=small, alignment=TA_CENTER,
                                 textColor=colors.HexColor("#991b1b"))
    cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=10.5, spaceAfter=0)
    cellh = ParagraphStyle("cellh", parent=cell, textColor=colors.white,
                           fontName="Helvetica-Bold")

    def P(txt, header=False):
        return Paragraph(str(txt), cellh if header else cell)

    def wrap_rows(rows):
        out = []
        for ri, row in enumerate(rows):
            out.append([P(c, header=(ri == 0)) for c in row])
        return out

    # figuras / ecuaciones
    p_rsys = fig_rsys_vs_distance()
    p_sigma = fig_sigma_by_depth()
    p_planar = fig_planar_filters()

    eq_rgeom = render_eq(r"$R_{geom}(b)\;=\;d\,\dfrac{L_{eff}+b}{L_{eff}}$", "rgeom")
    eq_rsys = render_eq(r"$R_{sys}(b)\;=\;\sqrt{\,R_{geom}(b)^2 + R_{int}^2\,}$", "rsys")
    eq_sigma = render_eq(r"$\sigma\;=\;\dfrac{FWHM}{2\sqrt{2\ln 2}}\;=\;\dfrac{FWHM}{2.3548}$", "sigma")
    eq_brow = render_eq(r"$b(r)\;=\;R_{orbita}\;-\;(r - c)\,\Delta_{px}\,,\qquad c=\dfrac{N-1}{2}$", "brow")
    eq_mlem = render_eq(
        r"$f_j^{(n+1)}\;=\;\dfrac{f_j^{(n)}}{\sum_i a_{ij}}\;\sum_i a_{ij}\,\dfrac{p_i}{\sum_k a_{ik}\,f_k^{(n)}}$",
        "mlem")
    eq_aij = render_eq(
        r"$a_{ij}\;=\;(\,\mathcal{R}_{\theta}\;\circ\;G_{\sigma(b)}\,)_{ij}\quad\Rightarrow\quad A^{\!\top}=G_{\sigma(b)}^{\!\top}\circ\mathcal{R}_{\theta}^{\!\top}$",
        "aij")
    eq_metz = render_eq(r"$A(f)\;=\;\dfrac{1-\left(1-MTF(f)^2\right)^{n}}{MTF(f)}$", "metz")
    eq_wiener = render_eq(r"$W(f)\;=\;\dfrac{MTF(f)}{MTF(f)^2 + NSR(f)}$", "wiener")
    eq_mtf = render_eq(r"$MTF(f)\;=\;\exp\!\left(-2\pi^2\sigma^2 f^2\right)$", "mtf")

    story = []

    # ------------------------------------------------ portada
    story.append(Spacer(1, 40 * mm))
    story.append(Paragraph("NITIDA (OmniRes)", title))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Recuperacion de resolucion dependiente de profundidad, "
                           "multi-fabricante, para SPECT y planar", subtitle))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Fundamento fisico-matematico y especificacion tecnica", subtitle))
    story.append(Spacer(1, 30 * mm))
    story.append(HRFlowable(width="60%", color=colors.HexColor("#9ca3af"), thickness=0.8))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("Suite GammaSys &mdash; Modulo SINCRO", ParagraphStyle(
        "org", parent=subtitle, alignment=TA_CENTER, fontSize=12)))
    story.append(Spacer(1, 40 * mm))
    story.append(Paragraph(
        "DOCUMENTO CONFIDENCIAL &mdash; SOFTWARE PROPIETARIO. "
        "NITIDA (OmniRes) NO es codigo abierto. El presente documento describe el fundamento "
        "propio del metodo y no divulga codigo fuente. Prohibida su reproduccion o distribucion "
        "sin autorizacion.", proprietary))
    story.append(PageBreak())

    # ------------------------------------------------ 1. Resumen
    story.append(Paragraph("1. Resumen ejecutivo", h1))
    story.append(Paragraph(
        "NITIDA (OmniRes) es el motor de <b>recuperacion de resolucion</b> (RR) de la suite. "
        "Modela la respuesta espacial del sistema colimador-detector &mdash;que se degrada con la "
        "distancia entre la fuente y el detector&mdash; e incorpora ese modelo dentro de la "
        "reconstruccion iterativa (OSEM/MLEM) en SPECT, o como filtro de restauracion 2D en "
        "adquisiciones planares. A diferencia de las soluciones atadas a una consola de un unico "
        "fabricante, NITIDA se <b>auto-configura desde el DICOM estandar</b> de cualquier gamma "
        "camara, resolviendo la geometria del colimador, el radio de orbita y el muestreo espacial "
        "a partir de los propios metadatos del estudio. El resultado es una recuperacion de "
        "nitidez que permite operar a <b>media dosis o medio tiempo</b> conservando calidad "
        "diagnostica, sobre el parque de equipos ya instalado.", body))
    story.append(Paragraph(
        "Este documento justifica el metodo <b>por si mismo</b>, a partir de la fisica de la "
        "formacion de imagen en medicina nuclear. La comparacion con productos de terceros se "
        "reserva al Anexo A.", body))

    # ------------------------------------------------ 2. Fisica
    story.append(Paragraph("2. Fundamento fisico: la respuesta colimador-detector", h1))
    story.append(Paragraph(
        "En una gamma camara tipo Anger, el colimador de huecos paralelos define la resolucion "
        "espacial. Un hueco de diametro efectivo <i>d</i> y largo efectivo <i>L_eff</i> acepta "
        "fotones dentro de un cono de aceptancia angular; la proyeccion de ese cono sobre el "
        "detector crece linealmente con la distancia <i>b</i> entre la fuente y la cara del "
        "colimador. La componente geometrica de la resolucion es:", body))
    story.append(eq_flowable(eq_rgeom, 120))
    story.append(Paragraph(
        "A ella se suma en cuadratura la resolucion intrinseca del detector <i>R_int</i> "
        "(centelleo + logica de posicionamiento), dando la resolucion del sistema:", body))
    story.append(eq_flowable(eq_rsys, 120))
    story.append(Paragraph(
        "Ambas se expresan como ancho a media altura (FWHM). Como la PSF del sistema es, en muy "
        "buena aproximacion, gaussiana, se convierte a la desviacion estandar mediante:", body))
    story.append(eq_flowable(eq_sigma, 110))
    story.append(Image(p_rsys, width=150 * mm, height=93.75 * mm, hAlign="CENTER"))
    story.append(Paragraph(
        "Figura 1. Resolucion del sistema R_sys(b) en funcion de la distancia fuente-colimador "
        "para las familias de colimador de la base multi-fabricante. El crecimiento lineal es el "
        "fenomeno que NITIDA modela y revierte.", caption))

    # ------------------------------------------------ 3. PSF por profundidad
    story.append(Paragraph("3. PSF dependiente de profundidad", h1))
    story.append(Paragraph(
        "El punto clave del metodo es que, dentro de un mismo corte transaxial, distintos voxeles "
        "estan a distinta distancia del detector segun su posicion respecto del eje de rotacion. "
        "Tras rotar el corte al angulo de proyeccion, la distancia fuente-colimador de cada fila "
        "<i>r</i> del corte rotado es:", body))
    story.append(eq_flowable(eq_brow, 130))
    story.append(Paragraph(
        "donde <i>R_orbita</i> es el radio de orbita del detector (leido del DICOM), "
        "<i>&Delta;_px</i> el tamano de pixel y <i>c</i> la fila central. La sigma de la PSF de "
        "cada fila se obtiene evaluando R_sys(b(r)) y convirtiendo a pixeles. Asi, la borrosidad "
        "no es uniforme: es minima del lado del detector y maxima en el fondo del campo.", body))
    story.append(Image(p_sigma, width=150 * mm, height=84.4 * mm, hAlign="CENTER"))
    story.append(Paragraph(
        "Figura 2. Sigma de la PSF por fila para un estudio cardiaco representativo (GE LEHR, "
        "orbita 268 mm, pixel 6.78 mm). La PSF varia con la profundidad dentro de un unico corte.",
        caption))
    story.append(Paragraph(
        "Por eficiencia, NITIDA cuantiza el continuo de sigmas en un numero reducido de bandas "
        "(por defecto 8) y aplica una convolucion gaussiana 1D por banda a lo largo del eje del "
        "detector, en lugar de una convolucion distinta por fila. El error de cuantizacion es "
        "despreciable frente al presupuesto de resolucion del sistema.", body))

    # ------------------------------------------------ 4. Modelado en OSEM/MLEM
    story.append(Paragraph("4. Incorporacion en la reconstruccion iterativa (SPECT)", h1))
    story.append(Paragraph(
        "En SPECT la recuperacion de resolucion <b>no es un pre-filtro de las proyecciones "
        "crudas</b>: es parte del <b>modelo del sistema</b> de la reconstruccion iterativa. El "
        "algoritmo MLEM actualiza la estimacion de la imagen segun:", body))
    story.append(eq_flowable(eq_mlem, 135))
    story.append(Paragraph(
        "donde <i>p_i</i> son las cuentas medidas, <i>f_j</i> los voxeles y <i>a_ij</i> la matriz "
        "del sistema (probabilidad de que una emision en el voxel <i>j</i> se detecte en el bin "
        "<i>i</i>). NITIDA introduce la PSF dependiente de profundidad <i>dentro</i> de esa matriz: "
        "el proyector directo compone la rotacion con el difuminado gaussiano por profundidad, y el "
        "retroproyector aplica la operacion transpuesta:", body))
    story.append(eq_flowable(eq_aij, 150))
    story.append(Paragraph(
        "De este modo, cada iteracion 'des-difumina' la imagen de forma consistente con la fisica "
        "de adquisicion, sin amplificar el ruido de manera incontrolada (a diferencia de una "
        "deconvolucion directa). El orden operacional correcto es: <b>(1) correccion de movimiento "
        "sobre las proyecciones, (2) reconstruccion con NITIDA activado</b>. Por su naturaleza "
        "iterativa, NITIDA requiere OSEM/MLEM; si se solicita sobre FBP, el sistema conmuta "
        "automaticamente a OSEM.", body))

    # ------------------------------------------------ 5. Resolucion-ruido / dosis
    story.append(Paragraph("5. Compromiso resolucion-ruido y fundamento del ahorro de dosis/tiempo", h1))
    story.append(Paragraph(
        "La reconstruccion iterativa con modelado de PSF concentra las cuentas detectadas en su "
        "posicion de origen mas probable, mejorando la relacion contraste-ruido para un mismo "
        "numero de cuentas. Equivalentemente, permite <b>reducir las cuentas</b> &mdash;bajando la "
        "actividad administrada (dosis) o el tiempo de adquisicion&mdash; manteniendo una relacion "
        "senal-ruido y una resolucion efectiva comparables a un protocolo estandar reconstruido "
        "sin RR. En validacion sobre un estudio a media dosis (GE Millennium, LEHR), el ruido "
        "relativo del fondo se redujo de 0.11% a 0.04% al activar OSEM+NITIDA, con una PSF "
        "auto-estimada de FWHM &asymp; 13.5 mm a 268 mm de orbita.", body))

    # ------------------------------------------------ 6. Fan-beam
    story.append(Paragraph("6. Extension a geometria fan-beam", h1))
    story.append(Paragraph(
        "Los colimadores fan-beam / de pinholes verticales (p.ej. camaras cardiacas dedicadas) "
        "magnifican el eje axial por un factor <i>M = distancia_imagen / distancia_objeto</i>. "
        "NITIDA detecta la geometria fan-beam desde el DICOM y, cuando el datasheet aporta la "
        "magnificacion real, re-muestrea el eje de filas por 1/M para llevar la proyeccion a una "
        "geometria cuasi-paralela antes del modelado de PSF. Si la magnificacion no esta "
        "disponible, el sistema lo informa y no inventa geometria (comportamiento honesto).", body))

    # ------------------------------------------------ 7. Auto-config DICOM
    story.append(Paragraph("7. Auto-configuracion multi-fabricante desde DICOM", h1))
    story.append(Paragraph(
        "El diferenciador central de NITIDA es que toda la parametrizacion fisica se deriva del "
        "DICOM estandar, sin depender de la consola del fabricante. Los campos utilizados son:", body))
    dicom_rows = [
        ["Parametro", "Tag DICOM", "Uso en NITIDA"],
        ["Fabricante", "(0008,0070)", "Seleccion de familia de colimador"],
        ["Modelo del equipo", "(0008,1090)", "Contexto / trazabilidad"],
        ["Nombre de colimador", "(0018,1180)", "Identificacion del colimador"],
        ["Tipo de colimador", "(0018,1181)", "paralelo / fan-beam"],
        ["Pixel spacing", "(0028,0030)", "Conversion mm <-> px de la PSF"],
        ["Radio de orbita", "(0018,1142)*", "Distancia fuente-detector b(r)"],
        ["Distancia focal", "(0018,1182)**", "Correccion axial fan-beam"],
    ]
    t = Table(wrap_rows(dicom_rows), colWidths=[42 * mm, 34 * mm, 74 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2ff")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Paragraph(
        "* Radio de orbita en la Rotation Information Sequence (0054,0052); se promedia si es un "
        "arreglo por vista. ** Distancia focal en la Detector Information Sequence (0054,0022).",
        small))

    # ------------------------------------------------ 8. Tabla de colimadores
    story.append(Paragraph("8. Base de colimadores (multi-fabricante)", h1))
    story.append(Paragraph(
        "La geometria fisica del colimador (que el DICOM no transporta) se resuelve contra una "
        "base editable indexada por (fabricante, nombre). Los valores son representativos de cada "
        "familia segun datasheets publicos; para uso clinico cuantitativo se sustituyen por las "
        "especificaciones exactas del modelo instalado. La columna R_sys@150mm ilustra la "
        "resolucion del sistema a 150 mm de distancia.", body))
    col_rows = [["Fabricante", "Colimador", "Geom.", "d [mm]", "L [mm]", "septa [mm]",
                 "R_int [mm]", "R_sys@150 [mm]"]]
    for man, name, geom, d, L, septa, r_int in COLLIMATORS:
        rs = r_sys(d, L, r_int, 150.0)
        col_rows.append([man, name, geom, f"{d:.2f}", f"{L:.2f}", f"{septa:.2f}",
                         f"{r_int:.1f}", f"{rs:.1f}"])
    ct = Table(wrap_rows(col_rows), colWidths=[20*mm, 33*mm, 18*mm, 15*mm, 15*mm, 18*mm, 17*mm, 22*mm])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2ff")]),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(ct)
    story.append(Paragraph(
        "Cuando el colimador no figura en la base, NITIDA recurre a un fallback LEHR generico "
        "(d=1.5, L=35.0, R_int=4.0) y, si el DICOM indica fan-beam, activa la correccion axial.",
        small))

    # ------------------------------------------------ 9. Planar
    story.append(Paragraph("9. Extension a planar whole-body multi-isotopo", h1))
    story.append(Paragraph(
        "En adquisiciones planares no hay reconstruccion; por lo tanto NITIDA se realiza como un "
        "<b>filtro de restauracion 2D</b> aplicado directamente a la imagen, usando la funcion de "
        "transferencia de modulacion (MTF) del colimador &mdash;la transformada de Fourier de la "
        "PSF gaussiana&mdash;:", body))
    story.append(eq_flowable(eq_mtf, 105))
    story.append(Paragraph(
        "La restauracion recupera las altas frecuencias atenuadas por el colimador. Dos formulaciones "
        "estandar, ambas derivables de la MTF propia del sistema:", body))
    story.append(eq_flowable(eq_metz, 120))
    story.append(eq_flowable(eq_wiener, 115))
    story.append(Paragraph(
        "El orden de Metz <i>n</i> (o la relacion ruido-senal NSR de Wiener) regula el compromiso "
        "entre realce y amplificacion de ruido.", body))
    story.append(Image(p_planar, width=150 * mm, height=89 * mm, hAlign="CENTER"))
    story.append(Paragraph(
        "Figura 3. Respuesta en frecuencia de los filtros de restauracion planar. Ambos elevan la "
        "ganancia donde la MTF del colimador cae, revirtiendo la borrosidad de forma controlada.",
        caption))
    story.append(Paragraph(
        "<b>Colimadores dependientes de energia.</b> El whole-body multi-isotopo exige el colimador "
        "correcto por energia de foton, lo que modifica d, L, septa y R_int:", body))
    iso_rows = [
        ["Isotopo", "Energia principal", "Colimador", "Nota fisica"],
        ["Tc-99m", "140 keV", "LEHR / LEGP (baja energia)", "Base actual de la tabla"],
        ["Ga-67", "93 / 185 / 300 / 393 keV", "Media energia (ME)", "Septa mas gruesa; menor resolucion"],
        ["In-111", "171 / 245 keV", "Media energia (ME)", "Septa mas gruesa; menor resolucion"],
        ["I-131", "364 keV (y 637)", "Alta energia (HE)", "Penetracion septal: modelar cola de PSF"],
    ]
    it = Table(wrap_rows(iso_rows), colWidths=[20*mm, 40*mm, 44*mm, 46*mm])
    it.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#065f46")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecfdf5")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(it)
    story.append(Paragraph(
        "En planar, la distancia relevante es la separacion tabla-colimador (no el radio de "
        "orbita) y es parametrizable por region corporal. Para I-131 (alta energia) la penetracion "
        "septal produce colas no gaussianas en la PSF que se modelan mas alla de la aproximacion "
        "gaussiana simple; alli la recuperacion de resolucion aporta el mayor beneficio relativo.",
        body))

    # ------------------------------------------------ 10. Conclusion
    story.append(Paragraph("10. Conclusion", h1))
    story.append(Paragraph(
        "NITIDA (OmniRes) se sostiene sobre la fisica establecida de la formacion de imagen en "
        "medicina nuclear: una PSF dependiente de profundidad derivada de la geometria del "
        "colimador, incorporada de forma consistente en la reconstruccion iterativa (SPECT) o como "
        "restauracion 2D (planar). Su caracter multi-fabricante &mdash;auto-configurado desde el "
        "DICOM estandar&mdash; y su cobertura multi-isotopo lo posicionan como motor de resolucion "
        "unico y transversal de la suite, habilitando protocolos de menor dosis y menor tiempo "
        "sobre equipamiento heterogeneo ya instalado.", body))

    # ------------------------------------------------ Anexo A
    story.append(PageBreak())
    story.append(Paragraph("Anexo A. Comparativa con sistemas de terceros", h1))
    story.append(Paragraph(
        "Se incluye a titulo informativo. NITIDA (OmniRes) se justifica por su propia fisica "
        "(secciones 2-9); esta comparativa no forma parte de dicha justificacion.", small))
    comp_rows = [
        ["Sistema", "Enfoque", "Acoplamiento", "Nota"],
        ["GE Evolution", "OSEM + RR (modelo CDR), + AC + scatter", "Consola GE (Xeleris)", "Un fabricante"],
        ["Philips Astonish", "OSEM + RR + regularizacion de ruido", "Consola Philips", "Un fabricante"],
        ["Siemens Flash3D", "3D-OSEM + RR + scatter + AC", "Consola Siemens", "Un fabricante"],
        ["UltraSPECT WBR", "Iterativo, solo-software (wide beam)", "Add-on (cobertura limitada)", "Software"],
        ["CZT (D-SPECT, Alcyone)", "Mejora por HARDWARE (detector CZT)", "Camara dedicada", "Otra categoria"],
        ["NITIDA (OmniRes)", "OSEM + RR (SPECT) / Wiener-Metz (planar)", "Vendor-neutral desde DICOM", "Multi-fabricante + planar"],
    ]
    cmp = Table(wrap_rows(comp_rows), colWidths=[32*mm, 54*mm, 38*mm, 26*mm])
    cmp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#dbeafe")),
        ("FONTNAME", (0, 6), (-1, 6), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(cmp)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "La diferencia estructural es el <b>acoplamiento</b>: los sistemas comerciales operan "
        "ligados al hardware/consola de su fabricante, mientras que NITIDA se configura desde el "
        "DICOM estandar y opera de forma transversal sobre equipos de distintas marcas, extendiendose "
        "ademas al dominio planar multi-isotopo.", body))
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#9ca3af"), thickness=0.6))
    story.append(Paragraph(
        "NITIDA (OmniRes) &mdash; Software propietario. No es codigo abierto. Documento confidencial.",
        proprietary))

    doc = SimpleDocTemplate(OUT_PDF, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title="NITIDA (OmniRes) - Fundamento tecnico",
                            author="Suite GammaSys")
    doc.build(story)
    print("PDF generado:", OUT_PDF)


if __name__ == "__main__":
    build()
