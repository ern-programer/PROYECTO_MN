"""SINCRO — Informe unificado AMYLO SPECT/CT (+planar si hay).

HTML autocontenido (imágenes/GIFs embebidos base64) y PDF A4 (reportlab,
estilo heredado del informe de perfusión). Plantillas builtin + personalizadas.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Plantillas
# ---------------------------------------------------------------------------

ALL_SECTIONS = [
    ("resumen", "Resumen ejecutivo"),
    ("metricas", "Métricas SPECT"),
    ("imagenes", "Imágenes (3D / cortes / fusión)"),
    ("gifs", "Animaciones (GIF)"),
    ("planar", "Bloque planar"),
    ("comparativa", "Comparativa planar vs SPECT"),
    ("limitaciones", "Limitaciones de cada método"),
    ("parametros", "Parámetros técnicos"),
]

BUILTIN_TEMPLATES: dict[str, dict] = {
    "SPECT/CT dinámico (oscuro)": {
        "accent": "#38bdf8",
        "dark": True,
        "sections": [k for k, _ in ALL_SECTIONS],
    },
    "Clínico claro A4": {
        "accent": "#1a3a5c",
        "dark": False,
        "sections": [k for k, _ in ALL_SECTIONS],
    },
    "Compacto (solo métricas + cortes)": {
        "accent": "#7c3aed",
        "dark": True,
        "sections": ["resumen", "metricas", "imagenes", "parametros"],
    },
}


def load_custom_templates(settings) -> dict[str, dict]:
    raw = str(settings.value("amylo_report/custom_templates_json", "") or "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        out = {}
        for name, tpl in dict(data).items():
            if isinstance(tpl, dict) and tpl.get("sections"):
                out[str(name)] = {
                    "accent": str(tpl.get("accent", "#38bdf8")),
                    "dark": bool(tpl.get("dark", True)),
                    "sections": [str(s) for s in tpl.get("sections", [])],
                }
        return out
    except Exception:
        return {}


def save_custom_templates(settings, templates: dict[str, dict]) -> None:
    settings.setValue("amylo_report/custom_templates_json", json.dumps(templates, ensure_ascii=False))
    settings.sync()


# ---------------------------------------------------------------------------
# Datos del informe
# ---------------------------------------------------------------------------

@dataclass
class AmyloReportData:
    patient: dict = field(default_factory=dict)       # name, id, sex, study_date, description, camera
    spect_metrics: dict = field(default_factory=dict)  # hmr, classification, volume_ml, pve{}, svd{}
    planar: dict | None = None                         # hmr, hmr_raw, perugini, washout_pct, source
    images: list = field(default_factory=list)         # [(titulo, png_path)]
    gifs: list = field(default_factory=list)           # [(titulo, gif_path)]
    warnings: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    template: dict = field(default_factory=lambda: dict(BUILTIN_TEMPLATES["SPECT/CT dinámico (oscuro)"]))
    template_name: str = "SPECT/CT dinámico (oscuro)"


LIMITACIONES_SPECT = (
    "HMR-SPECT usa VOIs 3D sobre volumen reconstruido: menor contaminación de estructuras "
    "vecinas que el planar, pero depende de la calidad de reconstrucción, del registro CT↔SPECT "
    "y del posicionamiento de los puntos A/B. Los cutoffs clínicos publicados (≥1.6 / 1.5-1.6 / <1.5) "
    "fueron validados mayormente en planar: la extrapolación a SPECT es orientativa. La corrección "
    "PVE es un modelo analítico experimental y siempre aumenta el valor: no usar el corregido "
    "contra cutoffs planares."
)
LIMITACIONES_PLANAR = (
    "HMR planar usa ROIs 2D sobre proyección AP: método validado (Perugini) con cutoffs "
    "establecidos, pero sufre superposición de estructuras (pulmón, esternón, pool vascular) "
    "y depende del posicionamiento manual de las ROIs. El grado Perugini es cualitativo y "
    "operador-dependiente."
)
NOTA_COMPARATIVA = (
    "Ambos métodos cuantifican la misma captación con geometrías distintas: valores absolutos "
    "NO intercambiables. La concordancia de clasificación entre ambos refuerza el hallazgo; "
    "la discordancia obliga a revisar técnica (ROIs, registro, reconstrucción) antes de interpretar."
)


def _fmt(value, nd: int = 2, suffix: str = "") -> str:
    try:
        f = float(value)
        if not np.isfinite(f):
            return "N/D"
        return f"{f:.{nd}f}{suffix}"
    except Exception:
        return "N/D"


def _class_color(classification: str) -> str:
    c = str(classification or "").upper()
    if "POSITIVO" in c:
        return "#ef4444"
    if "EQUIVOCO" in c or "EQUÍVOCO" in c:
        return "#f59e0b"
    if "NEGATIVO" in c:
        return "#22c55e"
    return "#94a3b8"


def _b64_file(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _gif_poster_b64(path: str) -> str:
    """Primer frame del GIF como PNG base64 (para pausar el cine en HTML)."""
    try:
        import io
        from PIL import Image
        with Image.open(path) as im:
            im.seek(0)
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def generate_amylo_html(data: AmyloReportData, output_html: str) -> str:
    tpl = data.template
    accent = str(tpl.get("accent", "#38bdf8"))
    dark = bool(tpl.get("dark", True))
    sections = set(tpl.get("sections", []))

    if dark:
        bg, card, fg, muted, border = "#0b1220", "#111a2e", "#e2e8f0", "#94a3b8", "#1e293b"
    else:
        bg, card, fg, muted, border = "#f8fafc", "#ffffff", "#1e293b", "#64748b", "#cbd5e1"

    m = data.spect_metrics or {}
    pve = m.get("pve") or {}
    svd = m.get("svd") or {}
    planar = data.planar or {}
    pat = data.patient or {}

    def sec(key: str) -> bool:
        return key in sections

    html: list[str] = []
    html.append(f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SINCRO — Informe AMYLO {pat.get('name', '')}</title>
<style>
:root {{ --accent:{accent}; --bg:{bg}; --card:{card}; --fg:{fg}; --muted:{muted}; --border:{border}; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:'Segoe UI',system-ui,sans-serif; line-height:1.5; }}
.wrap {{ max-width:1100px; margin:0 auto; padding:24px 20px 60px; }}
header {{ border-bottom:3px solid var(--accent); padding-bottom:14px; margin-bottom:22px;
  display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:8px; }}
header h1 {{ margin:0; font-size:26px; color:var(--accent); letter-spacing:0.5px; }}
header .sub {{ color:var(--muted); font-size:13px; }}
.badge {{ display:inline-block; padding:3px 12px; border-radius:999px; font-size:12px; font-weight:700;
  background:var(--accent); color:#fff; }}
.grid-cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:16px 0; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px 16px;
  transition:transform .15s ease, box-shadow .15s ease; }}
.card:hover {{ transform:translateY(-2px); box-shadow:0 6px 18px rgba(0,0,0,.25); }}
.card .k {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }}
.card .v {{ font-size:26px; font-weight:800; margin-top:4px; }}
.card .c {{ font-size:12px; font-weight:700; margin-top:2px; }}
section {{ margin:26px 0; }}
h2 {{ font-size:17px; color:var(--accent); border-left:4px solid var(--accent); padding-left:10px;
  margin:0 0 12px; text-transform:uppercase; letter-spacing:1px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th, td {{ border:1px solid var(--border); padding:7px 10px; text-align:left; }}
th {{ background:var(--accent); color:#fff; font-weight:700; }}
tr:nth-child(even) td {{ background:rgba(128,128,128,0.06); }}
.imgs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }}
.imgs-cascade {{ display:grid; grid-template-columns:1fr; gap:14px; }}
.imgs-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; margin-bottom:14px; }}
figure {{ margin:0; background:var(--card); border:1px solid var(--border); border-radius:12px;
  padding:10px; text-align:center; }}
figure img {{ max-width:100%; border-radius:8px; }}
figcaption {{ font-size:12px; color:var(--muted); margin-top:6px; }}
.warn {{ background:rgba(245,158,11,.12); border:1px solid #f59e0b; border-radius:10px;
  padding:10px 14px; font-size:13px; margin:8px 0; }}
.note {{ font-size:12px; color:var(--muted); }}
details {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
  padding:10px 14px; margin:8px 0; }}
summary {{ cursor:pointer; font-weight:700; color:var(--accent); }}
footer {{ margin-top:36px; border-top:1px solid var(--border); padding-top:12px;
  font-size:11px; color:var(--muted); }}
#cinebtn {{ position:fixed; bottom:18px; right:18px; z-index:99; padding:10px 16px;
  border:none; border-radius:999px; background:var(--accent); color:#fff; font-weight:700;
  font-size:13px; cursor:pointer; box-shadow:0 4px 14px rgba(0,0,0,.35); }}
#cinebtn:hover {{ filter:brightness(1.12); }}
@media print {{ .card:hover {{ transform:none; box-shadow:none; }} body {{ background:#fff; color:#000; }} #cinebtn {{ display:none; }} }}
</style></head><body><div class="wrap">
<header>
  <div>
    <h1>SINCRO — AMILOIDOSIS CARDÍACA</h1>
    <div class="sub">Informe {('integrado planar + SPECT/CT' if planar else 'SPECT/CT')} ·
      plantilla: {data.template_name} · generado {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
  </div>
  <span class="badge">EXPERIMENTAL — no diagnóstico automático</span>
</header>""")

    # --- Paciente ---
    html.append(f"""<section><h2>Datos del estudio</h2><table>
<tr><th>Paciente</th><td>{pat.get('name', 'N/D')}</td><th>ID</th><td>{pat.get('id', 'N/D')}</td></tr>
<tr><th>Sexo</th><td>{pat.get('sex', 'N/D')}</td><th>Fecha estudio</th><td>{pat.get('study_date', 'N/D')}</td></tr>
<tr><th>Descripción</th><td>{pat.get('description', 'N/D')}</td><th>Equipo</th><td>{pat.get('camera', 'N/D')}</td></tr>
</table></section>""")

    # --- Resumen: cards ---
    if sec("resumen"):
        cards = []
        hmr = m.get("hmr")
        if hmr is not None:
            cl = str(m.get("classification", ""))
            cards.append(("HMR-SPECT", _fmt(hmr), cl, _class_color(cl)))
        if pve.get("hmr_pve_corrected") is not None:
            cards.append(("HMR corregido PVE", _fmt(pve.get("hmr_pve_corrected")),
                          f"RC {_fmt(pve.get('rc_heart'))}", accent))
        if svd.get("ratio") is not None:
            cl = str(svd.get("classification", ""))
            cards.append(("Ratio S/√(V·D)", _fmt(svd.get("ratio")), cl, _class_color(cl)))
        if m.get("volume_ml") is not None:
            cards.append(("Volumen VOI corazón", _fmt(m.get("volume_ml"), 1, " mL"), "", accent))
        if planar.get("hmr") is not None:
            cards.append(("HMR planar", _fmt(planar.get("hmr")),
                          f"Perugini {planar.get('perugini', 'N/D')}", accent))
        card_html = "".join(
            f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
            f'<div class="c" style="color:{color};">{c}</div></div>'
            for k, v, c, color in cards
        ) or '<div class="note">Sin métricas calculadas.</div>'
        html.append(f'<section><h2>Resumen</h2><div class="grid-cards">{card_html}</div>')
        html.append(
            '<div class="note">Escala HMR: ≥1.6 POSITIVO · 1.5-1.6 EQUÍVOCO · &lt;1.5 NEGATIVO '
            '(validada en planar; en SPECT es orientativa). S/VD: ≥2.2 POSITIVO · 1.8-2.2 EQUÍVOCO · &lt;1.8 NEGATIVO.</div></section>'
        )

    # --- Métricas SPECT detalladas ---
    if sec("metricas"):
        rows = []
        for label, val in (
            ("HMR-SPECT", _fmt(m.get("hmr"))),
            ("Clasificación", str(m.get("classification", "N/D"))),
            ("Cuentas corazón (media/voxel)", _fmt(m.get("heart_counts"), 1)),
            ("Cuentas mediastino (media/voxel)", _fmt(m.get("mediastinum_counts"), 1)),
            ("Volumen VOI corazón", _fmt(m.get("volume_ml"), 1, " mL")),
            ("Método VOI", str(m.get("method", "N/D"))),
        ):
            rows.append(f"<tr><th>{label}</th><td>{val}</td></tr>")
        if pve:
            rows.append(f"<tr><th>HMR corregido PVE</th><td>{_fmt(pve.get('hmr_pve_corrected'))} "
                        f"(factor {_fmt(pve.get('pve_factor'))}, RC {_fmt(pve.get('rc_heart'))}, "
                        f"pared {_fmt(pve.get('wall_thickness_mm'), 1)} mm, FWHM {_fmt(pve.get('fwhm_mm'), 1)} mm)</td></tr>")
        if svd:
            rows.append(f"<tr><th>S/√(V·D)</th><td>{_fmt(svd.get('ratio'))} — {svd.get('classification', 'N/D')}</td></tr>")
        html.append(f'<section><h2>Métricas SPECT</h2><table>{"".join(rows)}</table></section>')

    # --- Bloque planar ---
    if sec("planar") and planar:
        rows = [
            f"<tr><th>HMR planar</th><td>{_fmt(planar.get('hmr'))}</td></tr>",
            f"<tr><th>Grado Perugini</th><td>{planar.get('perugini', 'N/D')}</td></tr>",
        ]
        if planar.get("washout_pct") is not None:
            rows.append(f"<tr><th>Washout 1h→3h</th><td>{_fmt(planar.get('washout_pct'), 1, ' %')}</td></tr>")
        if planar.get("source"):
            rows.append(f"<tr><th>Origen</th><td>{planar.get('source')}</td></tr>")
        html.append(f'<section><h2>Cuantificación planar</h2><table>{"".join(rows)}</table></section>')

    # --- Comparativa ---
    if sec("comparativa") and planar and m.get("hmr") is not None:
        hmr_s = m.get("hmr")
        hmr_p = planar.get("hmr")
        delta_txt = "N/D"
        concord = "N/D"
        if hmr_p is not None:
            try:
                delta_txt = f"{float(hmr_s) - float(hmr_p):+.2f}"
                cs, cp = str(m.get("classification", "")).upper(), ""
                hp = float(hmr_p)
                cp = "POSITIVO" if hp >= 1.6 else ("EQUIVOCO" if hp >= 1.5 else "NEGATIVO")
                concord = "CONCORDANTES ✓" if (cp in cs or cs in cp) and cp else "DISCORDANTES ⚠ revisar técnica"
            except Exception:
                pass
        html.append(f"""<section><h2>Comparativa planar vs SPECT</h2><table>
<tr><th></th><th>Planar 2D</th><th>SPECT 3D</th></tr>
<tr><th>HMR</th><td>{_fmt(hmr_p)}</td><td>{_fmt(hmr_s)}</td></tr>
<tr><th>Geometría</th><td>ROIs circulares s/ proyección AP</td><td>VOIs esféricas/anatómicas 3D</td></tr>
<tr><th>Validación cutoffs</th><td>Establecida (Perugini)</td><td>Orientativa (extrapolada)</td></tr>
<tr><th>Δ HMR (SPECT − planar)</th><td colspan="2">{delta_txt}</td></tr>
<tr><th>Clasificaciones</th><td colspan="2">{concord}</td></tr>
</table>
<div class="warn">{NOTA_COMPARATIVA}</div></section>""")

    # --- Limitaciones ---
    if sec("limitaciones"):
        blocks = [f'<details open><summary>Límites del método SPECT/CT</summary><p>{LIMITACIONES_SPECT}</p></details>']
        if planar:
            blocks.append(f'<details open><summary>Límites del método planar</summary><p>{LIMITACIONES_PLANAR}</p></details>')
        for w in data.warnings:
            blocks.append(f'<div class="warn">{w}</div>')
        html.append(f'<section><h2>Limitaciones y advertencias</h2>{"".join(blocks)}</section>')

    # --- Orden visual: VRT 3D → cortes por modalidad → MIPs rotatorios ---
    def _gif_fig(title: str, path: str) -> str:
        poster = _gif_poster_b64(path)
        poster_attr = f' data-poster="data:image/png;base64,{poster}"' if poster else ""
        return (f'<figure><img class="cine" src="data:image/gif;base64,{_b64_file(path)}"{poster_attr} '
                f'alt="{title}"><figcaption>▶ {title}</figcaption></figure>')

    mip_figs, other_figs = [], []
    if sec("gifs"):
        for title, path in data.gifs:
            if not (path and os.path.isfile(path)):
                continue
            (mip_figs if str(title).upper().startswith("MIP") else other_figs).append(_gif_fig(title, path))

    if other_figs:
        html.append(f'<section><h2>Volumen 3D (VRT)</h2><div class="imgs-cascade">{"".join(other_figs)}</div>'
                    '<div class="note">Animación visible solo en la versión HTML.</div></section>')

    if sec("imagenes") and data.images:
        figs = []
        for title, path in data.images:
            if path and os.path.isfile(path):
                figs.append(f'<figure><img src="data:image/png;base64,{_b64_file(path)}" alt="{title}">'
                            f'<figcaption>{title}</figcaption></figure>')
        if figs:
            html.append(f'<section><h2>Cortes por modalidad (a nivel del corazón)</h2>'
                        f'<div class="imgs-cascade">{"".join(figs)}</div></section>')

    if mip_figs:
        html.append(f'<section><h2>MIP rotatorio 360°</h2><div class="imgs-row">{"".join(mip_figs)}</div>'
                    '<div class="note">Animaciones visibles solo en la versión HTML.</div></section>')

    # --- Parámetros ---
    if sec("parametros") and data.params:
        rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in data.params.items())
        html.append(f'<section><h2>Parámetros técnicos</h2><details><summary>Ver parámetros del procesamiento</summary>'
                    f'<table>{rows}</table></details></section>')

    html.append(f"""<footer>SINCRO — módulo AMYLO SPECT/CT experimental. Este informe es de apoyo técnico y
no constituye interpretación diagnóstica automática. Plantilla «{data.template_name}».</footer>
</div>""")

    if data.gifs:
        html.append("""<button id="cinebtn" onclick="toggleCine()">⏸ Pausar cine</button>
<script>
var cineOn = true;
document.querySelectorAll('img.cine').forEach(function (im) { im.dataset.gif = im.src; });
function toggleCine() {
  cineOn = !cineOn;
  document.querySelectorAll('img.cine').forEach(function (im) {
    if (cineOn) { im.src = im.dataset.gif; }
    else if (im.dataset.poster) { im.src = im.dataset.poster; }
  });
  document.getElementById('cinebtn').textContent = cineOn ? '⏸ Pausar cine' : '▶ Reproducir cine';
}
</script>""")

    html.append("</body></html>")

    text = "".join(html)
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    with open(output_html, "wb") as fh:
        fh.write(text.encode("utf-8"))
    return output_html


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def generate_amylo_pdf(data: AmyloReportData, output_pdf: str) -> str:
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        HRFlowable, Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    tpl = data.template
    ACCENT = HexColor(str(tpl.get("accent", "#1a3a5c")))
    LIGHT = HexColor("#e8f0f8")
    GREY = HexColor("#f5f5f5")
    sections = set(tpl.get("sections", []))

    m = data.spect_metrics or {}
    pve = m.get("pve") or {}
    svd = m.get("svd") or {}
    planar = data.planar or {}
    pat = data.patient or {}

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=19, textColor=ACCENT)
    sub_style = ParagraphStyle("S", parent=styles["Normal"], fontSize=9.5, textColor=HexColor("#666666"))
    sect_style = ParagraphStyle("H", parent=styles["Heading2"], fontSize=12.5, textColor=ACCENT)
    body_style = ParagraphStyle("B", parent=styles["Normal"], fontSize=9.5, leading=13)
    small_style = ParagraphStyle("Sm", parent=styles["Normal"], fontSize=8, textColor=HexColor("#666666"))
    warn_style = ParagraphStyle(
        "W", parent=body_style, backColor=HexColor("#fff7ed"),
        borderColor=HexColor("#f59e0b"), borderWidth=0.8, borderPadding=4,
    )

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
        title="SINCRO — Informe AMYLO", author="SINCRO",
    )
    story: list = []

    def table_style(header_row: bool = False) -> TableStyle:
        cmds = [
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2.5 * mm),
            ("ROWBACKGROUNDS", (0, 1 if header_row else 0), (-1, -1), [white, GREY]),
        ]
        if header_row:
            cmds += [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("TEXTCOLOR", (0, 0), (-1, 0), white),
                     ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        else:
            cmds += [("BACKGROUND", (0, 0), (0, -1), LIGHT), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]
        return TableStyle(cmds)

    story.append(Paragraph("SINCRO — AMILOIDOSIS CARDÍACA", title_style))
    story.append(Paragraph(
        f"Informe {'integrado planar + SPECT/CT' if planar else 'SPECT/CT'} · "
        f"plantilla «{data.template_name}» · {datetime.now().strftime('%d/%m/%Y %H:%M')} · "
        "EXPERIMENTAL — no constituye interpretación diagnóstica automática",
        sub_style,
    ))
    story.append(Spacer(1, 1.5 * mm))
    story.append(HRFlowable(width="100%", thickness=1.4, color=ACCENT))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("1. Datos del estudio", sect_style))
    story.append(Table([
        ["Paciente", str(pat.get("name", "N/D")), "ID", str(pat.get("id", "N/D"))],
        ["Sexo", str(pat.get("sex", "N/D")), "Fecha", str(pat.get("study_date", "N/D"))],
        ["Descripción", str(pat.get("description", "N/D")), "Equipo", str(pat.get("camera", "N/D"))],
    ], colWidths=[24 * mm, 66 * mm, 22 * mm, 62 * mm], style=table_style()))
    story.append(Spacer(1, 4 * mm))

    if "metricas" in sections or "resumen" in sections:
        story.append(Paragraph("2. Métricas", sect_style))
        rows = [["Métrica", "Valor", "Interpretación"]]
        if m.get("hmr") is not None:
            rows.append(["HMR-SPECT", _fmt(m.get("hmr")), str(m.get("classification", "N/D"))])
        if pve.get("hmr_pve_corrected") is not None:
            rows.append(["HMR corregido PVE", _fmt(pve.get("hmr_pve_corrected")),
                         f"RC {_fmt(pve.get('rc_heart'))} · pared {_fmt(pve.get('wall_thickness_mm'), 1)} mm"])
        if svd.get("ratio") is not None:
            rows.append(["S/√(V·D)", _fmt(svd.get("ratio")), str(svd.get("classification", "N/D"))])
        if m.get("volume_ml") is not None:
            rows.append(["Volumen VOI corazón", _fmt(m.get("volume_ml"), 1, " mL"), ""])
        if planar.get("hmr") is not None:
            rows.append(["HMR planar", _fmt(planar.get("hmr")), f"Perugini {planar.get('perugini', 'N/D')}"])
        if planar.get("washout_pct") is not None:
            rows.append(["Washout planar 1h→3h", _fmt(planar.get("washout_pct"), 1, " %"), ""])
        story.append(Table(rows, colWidths=[54 * mm, 40 * mm, 80 * mm], style=table_style(header_row=True)))
        story.append(Spacer(1, 1.5 * mm))
        story.append(Paragraph(
            "Escala HMR: ≥1.6 POSITIVO · 1.5-1.6 EQUÍVOCO · <1.5 NEGATIVO (validada en planar; en SPECT orientativa). "
            "S/VD: ≥2.2 POSITIVO · 1.8-2.2 EQUÍVOCO · <1.8 NEGATIVO.", small_style))
        story.append(Spacer(1, 4 * mm))

    if "comparativa" in sections and planar and m.get("hmr") is not None:
        story.append(Paragraph("3. Comparativa planar vs SPECT", sect_style))
        try:
            delta = f"{float(m.get('hmr')) - float(planar.get('hmr')):+.2f}"
        except Exception:
            delta = "N/D"
        story.append(Table([
            ["", "Planar 2D", "SPECT 3D"],
            ["HMR", _fmt(planar.get("hmr")), _fmt(m.get("hmr"))],
            ["Geometría", "ROIs circulares (AP)", "VOIs 3D"],
            ["Cutoffs", "Validados (Perugini)", "Orientativos"],
            ["Δ HMR (SPECT − planar)", delta, ""],
        ], colWidths=[52 * mm, 61 * mm, 61 * mm], style=table_style(header_row=True)))
        story.append(Spacer(1, 1.5 * mm))
        story.append(Paragraph(NOTA_COMPARATIVA, warn_style))
        story.append(Spacer(1, 4 * mm))

    if "imagenes" in sections and (data.images or data.gifs):
        story.append(Paragraph("4. Imágenes", sect_style))
        img_paths = [(t, p) for t, p in data.images if p and os.path.isfile(p)]
        # GIFs → frame representativo en PDF
        for title, gpath in data.gifs:
            if gpath and os.path.isfile(gpath):
                img_paths.append((f"{title} (frame — animación disponible en HTML)", gpath))
        for title, path in img_paths:
            try:
                iw, ih = ImageReader(path).getSize()
                scale = min(160.0 * mm / iw, 110.0 * mm / ih, 1.0)
                story.append(RLImage(path, width=iw * scale, height=ih * scale))
                story.append(Paragraph(title, small_style))
                story.append(Spacer(1, 3 * mm))
            except Exception:
                continue

    if "limitaciones" in sections:
        story.append(Paragraph("5. Limitaciones y advertencias", sect_style))
        story.append(Paragraph(f"<b>SPECT/CT.</b> {LIMITACIONES_SPECT}", body_style))
        story.append(Spacer(1, 1.5 * mm))
        if planar:
            story.append(Paragraph(f"<b>Planar.</b> {LIMITACIONES_PLANAR}", body_style))
            story.append(Spacer(1, 1.5 * mm))
        for w in data.warnings:
            story.append(Paragraph(str(w), warn_style))
            story.append(Spacer(1, 1.5 * mm))
        story.append(Spacer(1, 2.5 * mm))

    if "parametros" in sections and data.params:
        story.append(Paragraph("6. Parámetros técnicos", sect_style))
        rows = [[str(k), str(v)] for k, v in data.params.items()]
        story.append(Table(rows, colWidths=[58 * mm, 116 * mm], style=table_style()))
        story.append(Spacer(1, 3 * mm))

    story.append(HRFlowable(width="100%", thickness=0.6, color=HexColor("#cccccc")))
    story.append(Paragraph(
        "SINCRO — módulo AMYLO SPECT/CT experimental. Informe de apoyo técnico; "
        "no constituye interpretación diagnóstica automática.", small_style))

    doc.build(story)
    return output_pdf
