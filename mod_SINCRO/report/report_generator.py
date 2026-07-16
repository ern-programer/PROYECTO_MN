"""SINCRO - report.report_generator — Generación integrada de informe PDF."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import HRFlowable, Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _safe_float(value, ndigits: int = 2) -> str:
	try:
		f = float(value)
	except Exception:
		return "N/D"
	if not np.isfinite(f):
		return "N/D"
	return f"{f:.{int(ndigits)}f}"


def _slice_list_text(indices: list[int], max_show: int = 12) -> str:
	if not indices:
		return "ninguno"
	vals = [int(i) + 1 for i in sorted(set(indices))]
	if len(vals) <= max_show:
		return ", ".join(str(v) for v in vals)
	head = ", ".join(str(v) for v in vals[:max_show])
	return f"{head}, ... (+{len(vals) - max_show})"


def _scaled_image(path: str, max_width: float, max_height: float) -> RLImage:
	iw, ih = ImageReader(path).getSize()
	if iw <= 0 or ih <= 0:
		return RLImage(path, width=max_width, height=max_height)
	scale = min(float(max_width) / float(iw), float(max_height) / float(ih))
	return RLImage(path, width=float(iw) * scale, height=float(ih) * scale)


def _audit_snapshot(seg, mask: np.ndarray) -> dict[str, object]:
	n_slices = int(mask.shape[0])
	valid_slices = np.where(mask.reshape(n_slices, -1).any(axis=1))[0].astype(int).tolist()
	inner = np.asarray(getattr(seg, "inner_radius", np.full((n_slices,), np.nan)), dtype=np.float64)

	no_inner_slices = [
		int(s)
		for s in valid_slices
		if s < int(inner.shape[0]) and (not np.isfinite(inner[s]) or float(inner[s]) <= 0.0)
	]

	edge = max(1, int(round(0.18 * len(valid_slices)))) if valid_slices else 0
	apex_base_candidates = sorted(set(valid_slices[:edge] + valid_slices[-edge:])) if edge > 0 else []
	no_inner_apex_base = sorted(set(no_inner_slices).intersection(apex_base_candidates))

	return {
		"method": str(getattr(seg, "method", "N/D")),
		"n_total_slices": n_slices,
		"n_valid_slices": len(valid_slices),
		"n_no_inner": len(no_inner_slices),
		"no_inner_slices": no_inner_slices,
		"n_no_inner_apex_base": len(no_inner_apex_base),
		"no_inner_apex_base_slices": no_inner_apex_base,
	}


def generate_report(
	*,
	output_pdf: str,
	output_dir: str,
	study,
	seg,
	metrics: dict,
	territory: dict,
	processing_params: dict,
	volumes: dict,
	ef: dict,
) -> str:
	"""Genera informe PDF clínico con bloque de auditoría y retorna la ruta final."""

	os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

	styles = getSampleStyleSheet()
	DARK_BLUE = HexColor("#1a3a5c")
	LIGHT_BLUE = HexColor("#e8f0f8")
	LIGHT_GREY = HexColor("#f5f5f5")

	title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=21, textColor=DARK_BLUE)
	subtitle_style = ParagraphStyle("SubtitleCustom", parent=styles["Normal"], fontSize=10, textColor=HexColor("#666666"))
	section_style = ParagraphStyle("SectionCustom", parent=styles["Heading2"], fontSize=13, textColor=DARK_BLUE)
	body_style = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=9.5, leading=13)
	small_style = ParagraphStyle("SmallCustom", parent=styles["Normal"], fontSize=8, textColor=HexColor("#666666"))

	doc = SimpleDocTemplate(
		output_pdf,
		pagesize=A4,
		leftMargin=18 * mm,
		rightMargin=18 * mm,
		topMargin=16 * mm,
		bottomMargin=16 * mm,
		title="SINCRO - Informe clínico",
		author="SINCRO",
	)

	cube = np.asarray(study.cube)
	audit = _audit_snapshot(seg, seg.mask.astype(bool))
	story: list = []

	story.append(Paragraph("SINCRO", title_style))
	story.append(Paragraph("Análisis de sincronía cardíaca — Informe automático", subtitle_style))
	story.append(Spacer(1, 1.5 * mm))
	story.append(HRFlowable(width="100%", thickness=1.4, color=DARK_BLUE))
	story.append(Spacer(1, 4 * mm))

	story.append(Paragraph("1. Datos del estudio", section_style))
	info_data = [
		["Fecha informe", datetime.now().strftime("%d/%m/%Y %H:%M")],
		["Descripción", str(getattr(study, "study_description", "N/D") or "N/D")],
		["Serie", str(getattr(study, "series_description", "N/D") or "N/D")],
		["Dimensiones", f"{cube.shape[0]} gates x {cube.shape[1]} slices x {cube.shape[2]}x{cube.shape[3]}"],
		["Segmentación", str(audit["method"])],
		["Voxels miocardio", f"{int(np.count_nonzero(seg.mask)):,}"],
	]
	info_table = Table(info_data, colWidths=[48 * mm, 118 * mm])
	info_table.setStyle(TableStyle([
		("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
		("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
		("FONTSIZE", (0, 0), (-1, -1), 9),
		("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
		("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
		("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
	]))
	story.append(info_table)
	story.append(Spacer(1, 4 * mm))

	story.append(Paragraph("2. Métricas principales", section_style))
	metrics_rows = [
		["Phase SD", f"{_safe_float(metrics.get('phase_sd'), 1)}°"],
		["Bandwidth", f"{_safe_float(metrics.get('bandwidth'), 1)}°"],
		["Entropy", _safe_float(metrics.get("entropy"), 3)],
		["Asynchrony Index", f"{_safe_float(metrics.get('asynchrony_index'), 1)}%"],
		["Clasificación", str(metrics.get("classification", "N/D"))],
		["Volumen miocardio", f"{_safe_float(volumes.get('myocardial_ml'), 2)} mL"],
		["Volumen cavidad", f"{_safe_float(volumes.get('cavity_ml'), 2)} mL"],
	]
	if ef.get("available"):
		metrics_rows.extend([
			["EDV preliminar", f"{_safe_float(ef.get('edv_ml'), 2)} mL"],
			["ESV preliminar", f"{_safe_float(ef.get('esv_ml'), 2)} mL"],
			["FEVI preliminar", f"{_safe_float(ef.get('ef_pct'), 1)}%"],
		])
	else:
		metrics_rows.append(["FEVI preliminar", "No disponible"]) 
	met_table = Table(metrics_rows, colWidths=[62 * mm, 104 * mm])
	met_table.setStyle(TableStyle([
		("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
		("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
		("FONTSIZE", (0, 0), (-1, -1), 9),
		("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, LIGHT_GREY]),
		("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
		("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
	]))
	story.append(met_table)
	story.append(Spacer(1, 4 * mm))

	story.append(Paragraph("3. Criterios usados (auditoría y validación)", section_style))
	audit_data = [
		["Campo", "Valor"],
		["Slices totales / válidos", f"{audit['n_total_slices']} / {audit['n_valid_slices']}"],
		["Slices con ROI sin interno", f"{audit['n_no_inner']} ({_slice_list_text(audit['no_inner_slices'])})"],
		[
			"Sin interno en extremos apex/base",
			f"{audit['n_no_inner_apex_base']} ({_slice_list_text(audit['no_inner_apex_base_slices'])})",
		],
		[
			"Criterio clínico aplicado",
			"Sin cavidad visible en apex/base: permitido r_inner='-' (sin interno).",
		],
		[
			"Impacto esperado",
			"Puede aumentar volumen miocárdico y reducir volumen de cavidad en esos slices.",
		],
		[
			"Advertencia FEVI",
			"FEVI es preliminar; interpretar con cautela y validar con paquete clínico validado.",
		],
	]
	audit_table = Table(audit_data, colWidths=[60 * mm, 106 * mm])
	audit_table.setStyle(TableStyle([
		("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
		("TEXTCOLOR", (0, 0), (-1, 0), white),
		("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
		("BACKGROUND", (0, 1), (0, -1), LIGHT_BLUE),
		("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
		("FONTSIZE", (0, 0), (-1, -1), 9),
		("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
		("VALIGN", (0, 0), (-1, -1), "TOP"),
		("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
	]))
	story.append(audit_table)
	story.append(Spacer(1, 3 * mm))

	proc_txt = (
		f"Threshold={_safe_float(processing_params.get('threshold'), 2)} | "
		f"Sigma={_safe_float(processing_params.get('smooth_sigma'), 1)} | "
		f"Harmonics={processing_params.get('harmonics', 'N/D')} | "
		f"Amp filter={_safe_float(processing_params.get('amp_filter'), 2)} | "
		f"Estilo visual={processing_params.get('visual_style', 'N/D')} | "
		f"Rotación polar={processing_params.get('polar_rotation_deg', 'N/D')}°"
	)
	story.append(Paragraph(f"<b>Parámetros usados:</b> {proc_txt}", body_style))

	story.append(PageBreak())
	story.append(Paragraph("4. Visualizaciones", section_style))
	img_files = [
		("slices_fase.png", "Slice medio con máscara y fase superpuesta."),
		("polar_map.png", "Mapa polar AHA (17 segmentos)."),
		("polar_perfusion_directa.png", "Mapa polar de perfusión continua (apex-centro, base-borde)."),
		("bullseye_directo.png", "Bull's eye de perfusión directa (colores de intensidad)."),
		("histograma.png", "Histograma de fase."),
		("ejes_ortogonales.png", "Ejes SA/HLA/VLA."),
		("panel_clinico_convencion.png", "Panel clínico A/B (ED/ES)."),
		("ventriculograma.png", "Panel funcional gated SPECT (ED/ES + curvas de volumen y fase)."),
		("comparacion_ejes.png", "Comparación original vs reconstruido."),
		("curva_tac.png", "Curva de actividad por gate."),
		("curva_fevi.png", "Curva FEVI preliminar con volumen y derivada."),
	]
	for fname, caption in img_files:
		path = os.path.join(output_dir, fname)
		if not os.path.exists(path):
			continue
		img = _scaled_image(path, max_width=165 * mm, max_height=95 * mm)
		img.hAlign = "CENTER"
		story.append(img)
		story.append(Paragraph(caption, ParagraphStyle("Cap", parent=small_style, alignment=1, spaceAfter=4 * mm)))

	story.append(Spacer(1, 4 * mm))
	story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#9aa7b5")))
	story.append(Paragraph(
		"Informe generado automáticamente por SINCRO. Resultados orientativos para apoyo clínico y auditoría técnica.",
		ParagraphStyle("Disc", parent=small_style, alignment=1),
	))

	doc.build(story)
	return output_pdf
