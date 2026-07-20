"""Genera informe PDF clínico con los resultados de SINCRO."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.console_utf8 import enable_utf8
enable_utf8()

import numpy as np
from datetime import datetime

from core import dicom_loader
from core.segmentation import segment_myocardium
from core.phase_analysis import phase_analysis
from core.aha_segments import map_to_17_segments, phase_by_segment, territory_analysis
from core.metrics import calculate_phase_metrics, circular_mean_deg

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)
from reportlab.lib.utils import ImageReader

# ============================================================
# Paths
# ============================================================
SA_GATED_PATH = (
    r"C:\Users\Ernesto\Desktop\INTERCAMBIO\varios stress cardiacos de xeleris"
    r"\estudio uno\MYOMETRIX\myometrix results\REST_IRNCG_SA001_DS.dcm"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_demo")
os.makedirs(OUT_DIR, exist_ok=True)
PDF_PATH = os.path.join(OUT_DIR, "informe_sincro.pdf")

# ============================================================
# Procesamiento
# ============================================================
print("Procesando estudio...")
study = dicom_loader.load(SA_GATED_PATH, verbose=False)
def _find_axis_companion_path(sa_path: str, axis_code: str) -> str | None:
    base = os.path.basename(sa_path)
    axis_code = str(axis_code).upper()
    if "_SA" not in base.upper():
        return None
    candidate = base.upper().replace("_SA", f"_{axis_code}")
    dir_path = os.path.dirname(sa_path)
    for name in os.listdir(dir_path):
        if name.upper() == candidate:
            return os.path.join(dir_path, name)
    return None


AXIS_COMPANIONS = {}
for _axis_code in ("HLA", "VLA"):
    _axis_path = _find_axis_companion_path(SA_GATED_PATH, _axis_code)
    if _axis_path and os.path.exists(_axis_path):
        try:
            AXIS_COMPANIONS[_axis_code] = dicom_loader.load(_axis_path, verbose=False)
        except Exception as exc:
            print(f"[WARN] no se pudo cargar serie {_axis_code}: {exc}")

cube = study.cube
seg = segment_myocardium(cube, method="auto")
mask = seg.mask
res = phase_analysis(cube, mask, harmonics=1, amplitude_threshold_frac=0.10)
phases = res.phases_deg
metrics = calculate_phase_metrics(phases)
aha = map_to_17_segments(seg)
pbs = phase_by_segment(res.phase_map, aha)
terr = territory_analysis(pbs)
print(f"  {phases.size} voxels, Phase SD={metrics['phase_sd']:.1f}°, Técnica PSD={metrics['technical_classification']}")


def _slice_list_text(indices: list[int], one_based: bool = True, max_show: int = 12) -> str:
    if not indices:
        return "ninguno"
    values = [int(i + 1) if one_based else int(i) for i in indices]
    if len(values) <= max_show:
        return ", ".join(str(v) for v in values)
    head = ", ".join(str(v) for v in values[:max_show])
    return f"{head}, ... (+{len(values) - max_show})"


def _build_audit_snapshot(seg_obj, mask_arr: np.ndarray, metrics_dict: dict) -> dict[str, object]:
    n_slices = int(mask_arr.shape[0])
    slice_has_mask = mask_arr.reshape(n_slices, -1).any(axis=1)
    valid_slices = np.where(slice_has_mask)[0].astype(int).tolist()
    inner = np.asarray(getattr(seg_obj, "inner_radius", np.full((n_slices,), np.nan)), dtype=np.float64)

    no_inner_slices = [
        int(s)
        for s in valid_slices
        if s < int(inner.shape[0]) and (not np.isfinite(inner[s]) or float(inner[s]) <= 0.0)
    ]

    apex_base_candidates: list[int] = []
    if valid_slices:
        edge = max(1, int(round(0.18 * len(valid_slices))))
        apex_base_candidates = sorted(set(valid_slices[:edge] + valid_slices[-edge:]))
    no_inner_apex_base = sorted(set(no_inner_slices).intersection(apex_base_candidates))

    return {
        "method": str(getattr(seg_obj, "method", "N/D")),
        "technical_classification": str(metrics_dict.get("technical_classification", metrics_dict.get("classification", "N/D"))),
        "n_total_slices": n_slices,
        "n_valid_slices": len(valid_slices),
        "n_no_inner": len(no_inner_slices),
        "no_inner_slices": no_inner_slices,
        "n_no_inner_apex_base": len(no_inner_apex_base),
        "no_inner_apex_base_slices": no_inner_apex_base,
    }


AUDIT = _build_audit_snapshot(seg, mask, metrics)


def _build_convention_panels(study_obj, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cube4d = np.asarray(study_obj.cube, dtype=np.float64)
    n_gates, n_slices = cube4d.shape[0], cube4d.shape[1]
    mid_slice = n_slices // 2
    mid_gate = n_gates // 2

    def _norm(img):
        arr = np.asarray(img, dtype=np.float64)
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        return arr / (mx + 1e-8)

    def _oriented_axes_views(gate_index: int):
        vol_gate = cube4d[int(gate_index)]
        sa_local = _norm(vol_gate[mid_slice])
        if AXIS_COMPANIONS.get("HLA") is not None:
            hla_study = AXIS_COMPANIONS["HLA"]
            hla_local = _norm(hla_study.cube[int(gate_index), min(int(hla_study.cube.shape[1] // 2), int(hla_study.cube.shape[1] - 1))])
        else:
            hla_local = _norm(vol_gate[:, vol_gate.shape[1] // 2, :])
        if AXIS_COMPANIONS.get("VLA") is not None:
            vla_study = AXIS_COMPANIONS["VLA"]
            vla_local = _norm(vla_study.cube[int(gate_index), min(int(vla_study.cube.shape[1] // 2), int(vla_study.cube.shape[1] - 1))])
        else:
            vla_local = _norm(vol_gate[:, :, vol_gate.shape[2] // 2])
        hla_view = np.fliplr(np.rot90(hla_local, k=1))
        vla_view = np.flipud(np.rot90(vla_local, k=-1))
        return sa_local, hla_view, vla_view

    def _annotate_axis(ax, top: str, bottom: str, left: str, right: str):
        label_style = dict(
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            color="#d7f0ff",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="black", edgecolor="#8ad0ff", alpha=0.65),
        )
        ax.text(0.50, 0.98, top, ha="center", va="top", **label_style)
        ax.text(0.50, 0.02, bottom, ha="center", va="bottom", **label_style)
        ax.text(0.02, 0.50, left, ha="left", va="center", rotation=90, **label_style)
        ax.text(0.98, 0.50, right, ha="right", va="center", rotation=270, **label_style)

    # Ejes ortogonales (gate medio)
    sa_mid, hla_mid, vla_mid = _oriented_axes_views(mid_gate)
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax in axes2:
        ax.set_xticks([])
        ax.set_yticks([])
    axes2[0].imshow(sa_mid, cmap="hot")
    axes2[0].set_title(f"SA (slice {mid_slice + 1})")
    _annotate_axis(axes2[0], "ANT", "INF", "SEP", "LAT")
    axes2[1].imshow(hla_mid, cmap="hot", aspect="auto")
    axes2[1].set_title("HLA (horizontal long axis)")
    _annotate_axis(axes2[1], "BASE", "APEX", "ANT", "INF")
    axes2[2].imshow(vla_mid, cmap="hot", aspect="auto")
    axes2[2].set_title("VLA (vertical long axis)")
    _annotate_axis(axes2[2], "BASE", "APEX", "SEP", "LAT")
    if AXIS_COMPANIONS.get("HLA") is not None:
        axes2[1].text(0.03, 0.05, "ORIGINAL", transform=axes2[1].transAxes, fontsize=8, color="#ffe082", fontweight="bold")
    if AXIS_COMPANIONS.get("VLA") is not None:
        axes2[2].text(0.03, 0.05, "ORIGINAL", transform=axes2[2].transAxes, fontsize=8, color="#ffe082", fontweight="bold")
    fig2.suptitle(f"Ejes cardíacos ortogonales — Gate {mid_gate + 1}", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "ejes_ortogonales.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    if AXIS_COMPANIONS:
        hla_recon = np.fliplr(np.rot90(_norm(cube4d[mid_gate][:, cube4d[mid_gate].shape[1] // 2, :]), k=1))
        vla_recon = np.flipud(np.rot90(_norm(cube4d[mid_gate][:, :, cube4d[mid_gate].shape[2] // 2]), k=-1))
        fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(10, 8))
        for ax in axes_cmp.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        axes_cmp[0, 0].imshow(hla_mid, cmap="hot", aspect="auto")
        axes_cmp[0, 0].set_title("HLA original")
        _annotate_axis(axes_cmp[0, 0], "BASE", "APEX", "ANT", "INF")
        axes_cmp[0, 1].imshow(hla_recon, cmap="hot", aspect="auto")
        axes_cmp[0, 1].set_title("HLA reconstruido desde SA")
        _annotate_axis(axes_cmp[0, 1], "BASE", "APEX", "ANT", "INF")
        axes_cmp[1, 0].imshow(vla_mid, cmap="hot", aspect="auto")
        axes_cmp[1, 0].set_title("VLA original")
        _annotate_axis(axes_cmp[1, 0], "BASE", "APEX", "SEP", "LAT")
        axes_cmp[1, 1].imshow(vla_recon, cmap="hot", aspect="auto")
        axes_cmp[1, 1].set_title("VLA reconstruido desde SA")
        _annotate_axis(axes_cmp[1, 1], "BASE", "APEX", "SEP", "LAT")
        fig_cmp.suptitle(f"Comparación original vs reconstruido — Gate {mid_gate + 1}", fontsize=13, fontweight="bold")
        fig_cmp.tight_layout()
        fig_cmp.savefig(os.path.join(out_dir, "comparacion_ejes.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_cmp)

    # Panel clínico A/B aproximando diástole/sístole desde la intensidad global de cavidad.
    gate_signal = np.asarray([float(np.mean(cube4d[g, mid_slice])) for g in range(n_gates)], dtype=np.float64)
    ed_gate = int(np.argmax(gate_signal))
    es_gate = int(np.argmin(gate_signal))

    sa_ed, hla_ed, vla_ed = _oriented_axes_views(ed_gate)
    sa_es, hla_es, vla_es = _oriented_axes_views(es_gate)

    fig4, axes4 = plt.subplots(2, 3, figsize=(14, 8.2))
    for ax in axes4.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes4[0, 0].imshow(sa_ed, cmap="hot")
    axes4[0, 0].set_title(f"A) ED - SHORT AXIS (Gate {ed_gate + 1})", fontsize=10)
    _annotate_axis(axes4[0, 0], "ANT", "INF", "SEP", "LAT")
    axes4[0, 1].imshow(hla_ed, cmap="hot", aspect="auto")
    axes4[0, 1].set_title("A) ED - HORIZONTAL AXIS (HLA)", fontsize=10)
    _annotate_axis(axes4[0, 1], "BASE", "APEX", "ANT", "INF")
    axes4[0, 2].imshow(vla_ed, cmap="hot", aspect="auto")
    axes4[0, 2].set_title("A) ED - VERTICAL AXIS (VLA)", fontsize=10)
    _annotate_axis(axes4[0, 2], "BASE", "APEX", "SEP", "LAT")
    axes4[1, 0].imshow(sa_es, cmap="hot")
    axes4[1, 0].set_title(f"B) ES - SHORT AXIS (Gate {es_gate + 1})", fontsize=10)
    _annotate_axis(axes4[1, 0], "ANT", "INF", "SEP", "LAT")
    axes4[1, 1].imshow(hla_es, cmap="hot", aspect="auto")
    axes4[1, 1].set_title("B) ES - HORIZONTAL AXIS (HLA)", fontsize=10)
    _annotate_axis(axes4[1, 1], "BASE", "APEX", "ANT", "INF")
    axes4[1, 2].imshow(vla_es, cmap="hot", aspect="auto")
    axes4[1, 2].set_title("B) ES - VERTICAL AXIS (VLA)", fontsize=10)
    _annotate_axis(axes4[1, 2], "BASE", "APEX", "SEP", "LAT")
    fig4.suptitle(
        "Panel clínico por convención (A=diástole, B=sístole) — SA/HLA/VLA",
        fontsize=13,
        fontweight="bold",
    )
    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, "panel_clinico_convencion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)


_build_convention_panels(study, OUT_DIR)

# ============================================================
# Estilos
# ============================================================
DARK_BLUE = HexColor("#1a3a5c")
MED_BLUE = HexColor("#2c7fb8")
LIGHT_BLUE = HexColor("#e8f0f8")
ACCENT_RED = HexColor("#d7191c")
LIGHT_GREY = HexColor("#f5f5f5")

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleCustom", parent=styles["Title"],
    fontSize=22, textColor=DARK_BLUE, spaceAfter=2*mm,
    fontName="Helvetica-Bold",
)
subtitle_style = ParagraphStyle(
    "SubtitleCustom", parent=styles["Normal"],
    fontSize=11, textColor=grey, spaceAfter=8*mm,
    alignment=TA_CENTER,
)
section_style = ParagraphStyle(
    "SectionCustom", parent=styles["Heading2"],
    fontSize=14, textColor=DARK_BLUE, spaceBefore=8*mm, spaceAfter=4*mm,
    fontName="Helvetica-Bold",
    borderPadding=(0, 0, 2, 0),
)
body_style = ParagraphStyle(
    "BodyCustom", parent=styles["Normal"],
    fontSize=10, leading=14, spaceAfter=3*mm,
)
small_style = ParagraphStyle(
    "SmallCustom", parent=styles["Normal"],
    fontSize=8, textColor=grey,
)
metric_label = ParagraphStyle(
    "MetricLabel", parent=styles["Normal"],
    fontSize=10, fontName="Helvetica-Bold", textColor=DARK_BLUE,
)
metric_value = ParagraphStyle(
    "MetricValue", parent=styles["Normal"],
    fontSize=10, alignment=TA_RIGHT,
)

# ============================================================
# Construir PDF
# ============================================================
doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=A4,
    leftMargin=20*mm, rightMargin=20*mm,
    topMargin=20*mm, bottomMargin=20*mm,
    title="SINCRO - Informe de Fase Cardíaca",
    author="SINCRO Module",
)

story = []

# --- Header ---
story.append(Paragraph("SINCRO", title_style))
story.append(Paragraph("Análisis de Sincronía Cardíaca — Gated SPECT", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=DARK_BLUE))
story.append(Spacer(1, 5*mm))

# --- Info del estudio ---
story.append(Paragraph("1. Datos del Estudio", section_style))

study_name = getattr(study, 'patient_name', 'N/D') if hasattr(study, 'patient_name') else "REST_IRNCG"
study_desc = getattr(study, 'description', study_name) if hasattr(study, 'description') else study_name

info_data = [
    ["Estudio", study_desc],
    ["Archivo", "REST_IRNCG_SA001_DS.dcm"],
    ["Fecha informe", datetime.now().strftime("%d/%m/%Y %H:%M")],
    ["Dimensiones", f"{cube.shape[0]} gates × {cube.shape[1]} slices × {cube.shape[2]}×{cube.shape[3]}"],
    ["Voxels miocardio", f"{int(mask.sum()):,}"],
    ["Voxels fase válida", f"{phases.size:,}"],
    ["Método", "FFT 1er armónico, filtro amplitud 10%"],
]

info_table = Table(info_data, colWidths=[50*mm, 110*mm])
info_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("LEFTPADDING", (0, 0), (-1, -1), 4*mm),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4*mm),
    ("TOPPADDING", (0, 0), (-1, -1), 2*mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2*mm),
]))
story.append(info_table)
story.append(Spacer(1, 6*mm))

# --- Métricas principales ---
story.append(Paragraph("2. Métricas de Disincronía", section_style))

# Clasificación técnica con color
cls = metrics.get("technical_classification", metrics.get("classification", "N/D"))
cls_colors = {
    "NORMAL": "#27ae60", "MILD": "#f39c12",
    "MODERATE": "#e67e22", "SEVERE": "#c0392b",
}
cls_color = cls_colors.get(cls, "#333333")

metrics_data = [
    ["Métrica", "Valor", "Referencia", "Estado"],
    ["Phase SD", f"{metrics['phase_sd']:.1f}°", "Clasificación técnica PSD; confirmar vs DB software-específica",
     f'<font color="{cls_color}"><b>{cls}</b></font>'],
    ["Bandwidth", f"{metrics['bandwidth']:.1f}°", "Comparar vs DB software-específica", ""],
    ["Entropy Shannon", f"{metrics.get('entropy_shannon_bits', metrics['entropy']):.3f} bits", "No comparar con entropy %", ""],
    ["Entropy normalizada", f"{metrics.get('entropy_normalized_pct', float('nan')):.1f}%", "Usar para literatura con entropy 0-100%", ""],
    ["Peak Phase", f"{metrics['peak_phase']:.1f}°", "—", ""],
    ["Peak Width", f"{metrics['peak_width']:.1f}°", "—", ""],
    ["Asynchrony Index", f"{metrics['asynchrony_index']:.1f}%", "—", ""],
    ["Fase media", f"{metrics['mean_phase']:.1f}°", "—", ""],
    ["Última activación", f"{metrics['latest_activation_phase']:.1f}°", "—", ""],
]

# Convertir strings con HTML a Paragraphs
for i in range(1, len(metrics_data)):
    for j in range(len(metrics_data[i])):
        cell = metrics_data[i][j]
        if "<font" in str(cell):
            metrics_data[i][j] = Paragraph(cell, body_style)
        else:
            metrics_data[i][j] = Paragraph(str(cell), body_style)

# Header row
for j in range(len(metrics_data[0])):
    metrics_data[0][j] = Paragraph(f"<b>{metrics_data[0][j]}</b>", 
                                    ParagraphStyle("TH", parent=body_style, textColor=white, fontSize=9))

m_table = Table(metrics_data, colWidths=[45*mm, 30*mm, 55*mm, 30*mm])
m_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GREY]),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 3*mm),
    ("RIGHTPADDING", (0, 0), (-1, -1), 3*mm),
    ("TOPPADDING", (0, 0), (-1, -1), 2*mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2*mm),
]))
story.append(m_table)
story.append(Spacer(1, 4*mm))

# Interpretación automática técnica
interp_text = ""
if cls == "NORMAL":
    interp_text = "La distribución de fase es homogénea por clasificación técnica PSD; confirmar con DB normal seleccionada."
elif cls == "MILD":
    interp_text = "Se observa leve heterogeneidad por clasificación técnica PSD; confirmar con DB normal seleccionada."
elif cls == "MODERATE":
    interp_text = "Heterogeneidad moderada por clasificación técnica PSD. No usar como indicación aislada de CRT/TRC."
else:
    interp_text = "Alta heterogeneidad por clasificación técnica PSD. Requiere correlación con QRS/BRI, FEVI, perfusión, viabilidad y clínica."

story.append(Paragraph(f"<b>Interpretación:</b> {interp_text}", body_style))
story.append(Spacer(1, 4*mm))

# --- Auditoría y validación ---
story.append(Paragraph("2.1 Criterios Usados en Este Estudio (Auditoría)", section_style))

audit_data = [
    ["Campo", "Valor"],
    ["Método de segmentación", AUDIT["method"]],
    ["Clasificación técnica PSD", AUDIT["technical_classification"]],
    ["Slices totales / válidos", f"{AUDIT['n_total_slices']} / {AUDIT['n_valid_slices']}"],
    ["Slices con ROI sin interno", f"{AUDIT['n_no_inner']} ({_slice_list_text(AUDIT['no_inner_slices'])})"],
    [
        "Sin interno en extremos apex/base",
        f"{AUDIT['n_no_inner_apex_base']} ({_slice_list_text(AUDIT['no_inner_apex_base_slices'])})",
    ],
    [
        "Criterio aplicado",
        "Cuando no hay cavidad visible en apex/base, se admite ROI sin interno (r_inner='-').",
    ],
]

audit_table = Table(audit_data, colWidths=[60*mm, 100*mm])
audit_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("BACKGROUND", (0, 1), (0, -1), LIGHT_BLUE),
    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 3*mm),
    ("RIGHTPADDING", (0, 0), (-1, -1), 3*mm),
    ("TOPPADDING", (0, 0), (-1, -1), 2*mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2*mm),
]))
story.append(audit_table)
story.append(Spacer(1, 2*mm))
story.append(Paragraph(
    "<b>Impacto esperado:</b> usar ROI sin interno en apex/base puede aumentar volumen miocárdico "
    "y reducir volumen de cavidad en esos slices.",
    body_style,
))
story.append(Paragraph(
    "<b>FEVI preliminar:</b> interpretar con cautela y validar con software clínico validado "
    "(estimación orientativa de investigación).",
    body_style,
))
story.append(Spacer(1, 4*mm))

# --- Segmentos AHA ---
story.append(Paragraph("3. Fase por Segmento AHA (17 segmentos)", section_style))

seg_header = ["Seg", "Nivel", "Territorio", "Fase (°)"]
seg_rows = [seg_header]

# Mapa segmento → territorio y nivel
seg_territory = {}
seg_level = {}
for t_name, t_data in [("LAD", range(1, 8)), ("LCx", [8, 9, 12, 13, 16]), ("RCA", [10, 11, 14, 15])]:
    for s in t_data:
        seg_territory[s] = t_name
for level, segs in [("Basal", range(1, 7)), ("Medio", range(7, 13)), ("Apical", range(13, 17))]:
    for s in segs:
        seg_level[s] = level
seg_level[17] = "Apex"

for sid in sorted(pbs.keys()):
    seg_rows.append([
        str(sid),
        seg_level.get(sid, "—"),
        seg_territory.get(sid, "—"),
        f"{pbs[sid]:.1f}°",
    ])

seg_table = Table(seg_rows, colWidths=[15*mm, 30*mm, 30*mm, 30*mm])
seg_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GREY]),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("ALIGN", (0, 0), (0, -1), "CENTER"),
    ("ALIGN", (3, 0), (3, -1), "RIGHT"),
    ("LEFTPADDING", (0, 0), (-1, -1), 3*mm),
    ("RIGHTPADDING", (0, 0), (-1, -1), 3*mm),
    ("TOPPADDING", (0, 0), (-1, -1), 1.5*mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5*mm),
]))
story.append(seg_table)
story.append(Spacer(1, 6*mm))

# --- Territorios ---
story.append(Paragraph("4. Análisis por Territorio Coronario", section_style))

terr_header = ["Territorio", "Fase media (°)", "SD (°)", "N° segmentos"]
terr_rows = [terr_header]
for t_name in ["LAD", "LCx", "RCA"]:
    d = terr[t_name]
    terr_rows.append([t_name, f"{d['mean']:.1f}", f"{d['std']:.1f}", str(d['n'])])

terr_table = Table(terr_rows, colWidths=[35*mm, 40*mm, 35*mm, 35*mm])
terr_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GREY]),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("LEFTPADDING", (0, 0), (-1, -1), 3*mm),
    ("TOPPADDING", (0, 0), (-1, -1), 2*mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2*mm),
]))
story.append(terr_table)
story.append(Spacer(1, 6*mm))

# --- Imágenes (segunda página) ---
story.append(PageBreak())
story.append(Paragraph("SINCRO", title_style))
story.append(Paragraph("Visualizaciones", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=DARK_BLUE))
story.append(Spacer(1, 5*mm))


def _scaled_image(path: str, max_width: float, max_height: float) -> RLImage:
    """Crea una imagen de ReportLab ajustada sin perder relación de aspecto."""
    iw, ih = ImageReader(path).getSize()
    if iw <= 0 or ih <= 0:
        return RLImage(path, width=max_width, height=max_height)
    scale = min(float(max_width) / float(iw), float(max_height) / float(ih))
    width = float(iw) * scale
    height = float(ih) * scale
    return RLImage(path, width=width, height=height)

img_files = [
    ("slices_fase.png", "Figura 1 — Slice medio con máscara miocárdica y fase superpuesta (colormap HSV)."),
    ("polar_map.png", "Figura 2 — Bullseye AHA 17 segmentos. Cada cuña muestra el ID y fase en grados."),
    ("histograma.png", "Figura 3 — Histograma de fase. Línea roja: media circular. Naranja: P5/P95."),
    ("ejes_ortogonales.png", "Figura 4 — Ejes SA/HLA/VLA en gate medio. HLA y VLA reconstruidos desde el stack SA cuando no hay series originales."),
    ("curva_tac.png", "Figura 5 — Curva de actividad miocárdica por gate (izq.) y radar de fase por segmento AHA (der.)."),
    ("panel_clinico_convencion.png", "Figura 6 — Panel clínico por convención con ejes ortogonales en diástole/sístole (A/B)."),
    ("comparacion_ejes.png", "Figura 7 — Comparación HLA/VLA original vs reconstruido desde SA en el mismo gate."),
]

for fname, caption in img_files:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        img = _scaled_image(fpath, max_width=155*mm, max_height=100*mm)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Paragraph(caption, ParagraphStyle(
            "Caption", parent=small_style, alignment=TA_CENTER, spaceAfter=6*mm,
            fontSize=9, textColor=DARK_BLUE,
        )))

# --- Disclaimer ---
story.append(Spacer(1, 8*mm))
story.append(HRFlowable(width="100%", thickness=0.5, color=grey))
story.append(Paragraph(
    "<i>Informe generado automáticamente por SINCRO (módulo de análisis de sincronía cardíaca). "
    "Los resultados son orientativos y deben ser interpretados por un profesional médico. "
    "No constituyen diagnóstico.</i>",
    ParagraphStyle("Disclaimer", parent=small_style, alignment=TA_CENTER, spaceAfter=3*mm),
))
story.append(Paragraph(
    f"<i>Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} — SINCRO v0.4</i>",
    ParagraphStyle("Footer", parent=small_style, alignment=TA_CENTER),
))

# ============================================================
# Build
# ============================================================
doc.build(story)
print(f"\n[OK] Informe PDF generado: {PDF_PATH}")
print(f"     Tamaño: {os.path.getsize(PDF_PATH)/1024:.0f} KB")
