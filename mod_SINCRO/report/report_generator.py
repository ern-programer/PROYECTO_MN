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


def _phase_label_from_source(path_text: str) -> str:
	u = os.path.basename(str(path_text or "")).upper()
	if "STRESS" in u:
		return "Esfuerzo"
	if "REST" in u:
		return "Reposo"
	return "Estudio"


def _format_dicom_date(raw: str) -> str:
	val = str(raw or "").strip()
	if len(val) == 8 and val.isdigit():
		return f"{val[6:8]}/{val[4:6]}/{val[0:4]}"
	return val or "N/D"


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
	phase_label = _phase_label_from_source(str(getattr(study, "source_path", "") or ""))
	patient_name = str(getattr(study, "patient_name", "") or "").strip() or "N/D"
	patient_id = str(getattr(study, "patient_id", "") or "").strip() or "N/D"
	patient_sex = str(getattr(study, "patient_sex", "") or "").strip() or "N/D"
	study_date = _format_dicom_date(str(getattr(study, "study_date", "") or ""))
	study_time = str(getattr(study, "study_time", "") or "").strip() or "N/D"
	accession_number = str(getattr(study, "accession_number", "") or "").strip() or "N/D"
	series_uid = str(getattr(study, "study_instance_uid", "") or "").strip() or "N/D"
	story: list = []

	story.append(Paragraph("SINCRO", title_style))
	story.append(Paragraph(f"Análisis de sincronía cardíaca — Informe automático ({phase_label})", subtitle_style))
	story.append(Spacer(1, 1.5 * mm))
	story.append(HRFlowable(width="100%", thickness=1.4, color=DARK_BLUE))
	story.append(Spacer(1, 4 * mm))

	story.append(Paragraph("1. Datos del estudio", section_style))
	info_data = [
		["Fecha informe", datetime.now().strftime("%d/%m/%Y %H:%M")],
		["Fase procesada", phase_label],
		["Paciente", patient_name],
		["Patient ID", patient_id],
		["Sexo", patient_sex],
		["Fecha/Hora estudio", f"{study_date} {study_time}".strip()],
		["Accession", accession_number],
		["Study UID", series_uid],
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
	nd = metrics.get("normal_db_eval") or {}
	nd_metrics = nd.get("metrics") or {}
	nd_label = "N/D"
	if nd:
		nd_label = "fuera de referencia" if nd.get("dyssynchrony") else "dentro de referencia"
	technical_class = str(metrics.get("technical_classification", metrics.get("classification", "N/D")))
	metrics_rows = [
		["Phase SD", f"{_safe_float(metrics.get('phase_sd'), 1)}°"],
		["Bandwidth", f"{_safe_float(metrics.get('bandwidth'), 1)}°"],
		["Entropy Shannon", f"{_safe_float(metrics.get('entropy_shannon_bits', metrics.get('entropy')), 3)} bits"],
		["Entropy normalizada", f"{_safe_float(metrics.get('entropy_normalized_pct'), 1)}%"],
		["Asynchrony Index", f"{_safe_float(metrics.get('asynchrony_index'), 1)}%"],
		["Clase PSD técnica", f"{technical_class} (orientativa, no diagnóstica)"],
		["Interpretación vs DB", nd_label],
		["Skewness / Kurtosis", f"{_safe_float(metrics.get('skewness'), 3)} / {_safe_float(metrics.get('kurtosis'), 3)}"],
		["Peak phase / width", f"{_safe_float(metrics.get('peak_phase'), 1)}° / {_safe_float(metrics.get('peak_width'), 1)}°"],
		["Latest activation", f"{_safe_float(metrics.get('latest_activation_phase'), 1)}°"],
		["Volumen miocardio", f"{_safe_float(volumes.get('myocardial_ml'), 2)} mL"],
		["Volumen cavidad", f"{_safe_float(volumes.get('cavity_ml'), 2)} mL"],
	]
	segmental = metrics.get("segmental_aha") or {}
	if segmental.get("available"):
		metrics_rows.extend([
			["Modo segmentario AHA", f"PSD {_safe_float(segmental.get('phase_sd'), 1)}° | BW {_safe_float(segmental.get('bandwidth'), 1)}° | n={segmental.get('n_segments')}"],
			["Clase segmentaria AHA", f"{segmental.get('technical_classification', 'N/D')} (robusta, menor resolución)"],
		])
	bootstrap = metrics.get("bootstrap") or {}
	if bootstrap.get("available"):
		psd_boot = bootstrap.get("phase_sd", {})
		bw_boot = bootstrap.get("bandwidth", {})
		metrics_rows.extend([
			["Bootstrap PSD IC95", f"{_safe_float(psd_boot.get('ci95_low'), 1)}–{_safe_float(psd_boot.get('ci95_high'), 1)}° | media {_safe_float(psd_boot.get('mean'), 1)}°"],
			["Bootstrap BW IC95", f"{_safe_float(bw_boot.get('ci95_low'), 1)}–{_safe_float(bw_boot.get('ci95_high'), 1)}° | media {_safe_float(bw_boot.get('mean'), 1)}°"],
		])
	roi_sens = metrics.get("roi_sensitivity") or {}
	if roi_sens.get("available"):
		qc = "sensible a ROI" if roi_sens.get("warn") else "estable ante ROI ±1 px"
		metrics_rows.append([
			"Sensibilidad ROI",
			f"PSD {_safe_float(roi_sens.get('phase_sd_min'), 1)}–{_safe_float(roi_sens.get('phase_sd_max'), 1)}° | BW {_safe_float(roi_sens.get('bandwidth_min'), 1)}–{_safe_float(roi_sens.get('bandwidth_max'), 1)}° | {qc}",
		])
	if nd:
		for key, label, unit in (("phase_sd", "DB PSD", "°"), ("bandwidth", "DB BW", "°"), ("entropy_normalized_pct", "DB Entropy norm.", "%")):
			m = nd_metrics.get(key) or {}
			if not m.get("available"):
				continue
			flag = "fuera ref." if m.get("abnormal") else "dentro ref."
			cutoff = m.get("cutoff")
			z_value = f"{m['z']:+.1f}" if m.get("z") is not None else "n/d"
			metrics_rows.append([label, f"{_safe_float(m.get('value'), 1)}{unit} | cutoff {_safe_float(cutoff, 1)}{unit} | z={z_value} | {flag}"])
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
	roi_lines = [line.strip() for line in str(processing_params.get("manual_rois_text", "") or "").splitlines() if line.strip() and not line.strip().startswith("#")]
	roi_preview = "; ".join(roi_lines[:4]) if roi_lines else "N/D"
	if len(roi_lines) > 4:
		roi_preview += f"; ... (+{len(roi_lines) - 4})"
	audit_data = [
		["Campo", "Valor"],
		["Segmentación solicitada", str(processing_params.get("seg_method", "N/D"))],
		["Segmentación efectiva", str(audit["method"])],
		["ROI reproducible", f"{len(roi_lines)} slices en formato slice,cy,cx,r_inner,r_outer" if roi_lines else "No disponible"],
		["ROI ejemplo", roi_preview],
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
		[
			"Alcance LVMD",
			"GammaSync informa asincronía mecánica intraventricular del VI; no reemplaza ECG, eco/CMR ni evaluación clínica integral.",
		],
		[
			"Contexto CRT",
			"Los cutoffs para terapia de resincronización cardíaca son heterogéneos y software-dependientes; usar solo como contexto, no como indicación aislada.",
		],
		[
			"Robustez estadística",
			"Se reporta modo voxel, modo segmentario AHA, bootstrap e impacto de mover/contraer/expandir ROI ±1 px cuando están disponibles.",
		],
	]
	if roi_sens.get("available"):
		for row in roi_sens.get("variants", [])[:7]:
			if "error" in row:
				continue
			audit_data.append([
				f"ROI {row.get('label')}",
				f"PSD {_safe_float(row.get('phase_sd'), 1)}° | BW {_safe_float(row.get('bandwidth'), 1)}° | voxels fase {row.get('phase_voxels', 'N/D')}",
			])
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
		f"Rotación polar={processing_params.get('polar_rotation_deg', 'N/D')}° | "
		f"Suavizado polar={processing_params.get('polar_perf_smooth_method', 'N/D')} {_safe_float(processing_params.get('polar_perf_smooth_strength'), 2)} | "
		f"Polar cine={processing_params.get('polar_cine_speed_ms', 'N/D')} ms/frame | "
		f"MP4 polar cine={'sí' if processing_params.get('export_polar_mp4', False) else 'no'}"
	)
	story.append(Paragraph(f"<b>Parámetros usados:</b> {proc_txt}", body_style))

	# Sección ECG si hay datos
	ecg_data = []
	if processing_params.get("ecg_ritmo"):
		ecg_data.append(["Ritmo", str(processing_params.get("ecg_ritmo", "N/D"))])
	if processing_params.get("ecg_fc"):
		ecg_data.append(["FC", f"{processing_params.get('ecg_fc', 'N/D')} lpm"])
	if processing_params.get("ecg_qrs"):
		ecg_data.append(["QRS", f"{processing_params.get('ecg_qrs', 'N/D')} ms"])
	if processing_params.get("ecg_qt"):
		ecg_data.append(["QT", f"{processing_params.get('ecg_qt', 'N/D')} ms"])
	ecg_flags = []
	if processing_params.get("ecg_bri"):
		ecg_flags.append("BRI")
	if processing_params.get("ecg_brd"):
		ecg_flags.append("BRD")
	if processing_params.get("ecg_marcapasos"):
		ecg_flags.append("Marcapasos/CRT")
	if ecg_flags:
		ecg_data.append(["Conducción", ", ".join(ecg_flags)])
	if processing_params.get("ecg_observaciones"):
		ecg_data.append(["Observaciones", str(processing_params.get("ecg_observaciones", ""))])

	if ecg_data:
		story.append(Spacer(1, 3 * mm))
		story.append(Paragraph("Contexto electrocardiográfico", section_style))
		ecg_table = Table(ecg_data, colWidths=[48 * mm, 118 * mm])
		ecg_table.setStyle(TableStyle([
			("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
			("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
			("FONTSIZE", (0, 0), (-1, -1), 9),
			("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
			("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
			("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
		]))
		story.append(ecg_table)
		story.append(Spacer(1, 2 * mm))
		# Score electro-mecánico simple
		qrs_ms = processing_params.get("ecg_qrs", 0)
		has_bri = processing_params.get("ecg_bri", False)
		psd_val = metrics.get("phase_sd", 0)
		if qrs_ms >= 120 or has_bri:
			em_txt = "Concordante: QRS ancho/BRI con posible disincronía mecánica."
		elif qrs_ms < 100 and psd_val > 40:
			em_txt = "Discordante: QRS estrecho pero PSD elevado; revisar artefacto/ROI."
		else:
			em_txt = "Sin discordancia clara electro-mecánica."
		story.append(Paragraph(f"<b>Evaluación electro-mecánica:</b> {em_txt}", body_style))
		story.append(Spacer(1, 2 * mm))

	if nd:
		db_txt = (
			f"<b>DB normal:</b> {metrics.get('normal_db_dataset', 'N/D')} | "
			f"sexo={metrics.get('normal_db_sex', 'N/D')} | protocolo={metrics.get('normal_db_protocol', 'N/D')}. "
			"Las métricas no son intercambiables entre QGS, ECTb, 4DM, cREPO, HFV o GammaSync sin validación cruzada/local."
		)
		story.append(Spacer(1, 2 * mm))
		story.append(Paragraph(db_txt, body_style))

	story.append(PageBreak())
	story.append(Paragraph("4. Visualizaciones", section_style))
	img_files = [
		("slices_fase.png", "Slice medio con máscara y fase superpuesta."),
		("polar_map.png", "Mapa polar de fase AHA (17): muestra distribución regional de activación mecánica. Uso: patrón/extensión de disincronía."),
		("polar_clinico.png", "Panel polar clínico (histograma + bullseye) con PSD/PHB para lectura rápida estilo estación clínica."),
		("polar_map_delta_signed.png", "Delta con signo (esfuerzo - reposo), circular: conserva dirección del cambio (adelanto/atraso relativo)."),
		("polar_map_absdiff.png", "Delta absoluto |esfuerzo - reposo|: magnitud del cambio regional sin dirección (hotspots dinámicos)."),
		("polar_perfusion_directa.png", "Mapa polar continuo de perfusión (apex-centro, base-borde). Uso: heterogeneidad perfusional regional continua."),
		("polar_cine_montaje.png", "Polar cine gatillado (muestra de gates). Uso: dinámica temporal del patrón polar. Animados: polar_cine.gif / polar_cine.mp4."),
		("bullseye_directo.png", "Bull's-eye segmentario AHA (17) de perfusión directa. Uso: resumen rápido de intensidad regional."),
		("ejes_ortogonales.png", "Ejes SA/HLA/VLA."),
		("panel_clinico_convencion.png", "Panel clínico A/B (ED/ES)."),
		("panel_funcional_gated.png", "Panel funcional gated (ED/ES + curvas de volumen y fase)."),
		("ventriculograma.png", "Panel funcional gated (ED/ES + curvas de volumen y fase)."),
		("comparacion_ejes.png", "Comparación original vs reconstruido."),
		("comparacion_stress_rest.png", "Comparación de disincronía entre estudios (stress vs rest): PSD, BW, Kurtosis, Entropy con Δ e interpretación de stunning."),
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

	story.append(PageBreak())
	story.append(Paragraph("5. Llamadas clínicas de interpretación", section_style))
	story.append(Paragraph(
		"<b>polar_map:</b> describe la distribución de fase por segmentos AHA."
		" Es el mapa basal para inferir patrón de disincronía intraventricular (regional y global).",
		body_style,
	))
	story.append(Spacer(1, 1.5 * mm))
	story.append(Paragraph(
		"<b>polar_clinico:</b> integra histograma de fase + bullseye en una sola vista rápida."
		" Útil para lectura inicial y comunicación clínica, sin reemplazar la revisión completa de mapas y cine.",
		body_style,
	))
	story.append(Spacer(1, 1.5 * mm))
	story.append(Paragraph(
		"<b>polar_map_Δsigned:</b> diferencia circular con signo entre esfuerzo y reposo."
		" Un valor positivo indica adelanto relativo en esfuerzo; negativo indica atraso relativo.",
		body_style,
	))
	story.append(Spacer(1, 1.5 * mm))
	story.append(Paragraph(
		"<b>polar_map_Δabs:</b> valor absoluto del cambio stress-rest."
		" Sirve para cuantificar magnitud regional del cambio sin depender de la dirección.",
		body_style,
	))
	story.append(Spacer(1, 1.5 * mm))
	story.append(Paragraph(
		"<b>polar_perfusion_directa:</b> mapa continuo de intensidad perfusional (apex en centro, base en borde)."
		" Complementa fase para diferenciar alteración temporal vs alteración de captación.",
		body_style,
	))
	story.append(Spacer(1, 1.5 * mm))
	story.append(Paragraph(
		"<b>bullseye_directo:</b> resumen segmentario AHA de perfusión."
		" Lectura compacta para identificar regiones de hipocaptación y comunicar hallazgos en reporte.",
		body_style,
	))
	story.append(Spacer(1, 1.5 * mm))
	story.append(Paragraph(
		"<b>polar_cine_montaje:</b> añade dimensión temporal gate-a-gate."
		" Útil cuando la foto estática no refleja la dinámica mecánica completa del ciclo.",
		body_style,
	))
	story.append(Spacer(1, 2.2 * mm))
	story.append(Paragraph(
		"<b>Pie de uso recomendado:</b> interpretar siempre en conjunto fase + perfusión + cine + métricas (PSD/BW/Entropy),"
		" comparadas contra referencias del mismo software o contra validación local. FEVI en este informe es preliminar.",
		small_style,
	))

	story.append(Spacer(1, 4 * mm))
	story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#9aa7b5")))
	story.append(Paragraph(
		"Informe generado automáticamente por SINCRO. Resultados orientativos para apoyo clínico y auditoría técnica.",
		ParagraphStyle("Disc", parent=small_style, alignment=1),
	))

	doc.build(story)
	return output_pdf


def generate_polar_reference_pdf(*, output_pdf: str) -> str:
	"""Genera un PDF técnico separado con explicación clínica y fórmulas de mapas polares y sincronía."""

	os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

	styles = getSampleStyleSheet()
	DARK_BLUE = HexColor("#1a3a5c")

	title_style = ParagraphStyle("RefTitle", parent=styles["Title"], fontSize=20, textColor=DARK_BLUE)
	section_style = ParagraphStyle("RefSection", parent=styles["Heading2"], fontSize=12.5, textColor=DARK_BLUE)
	body_style = ParagraphStyle("RefBody", parent=styles["Normal"], fontSize=9.6, leading=13.5)
	small_style = ParagraphStyle("RefSmall", parent=styles["Normal"], fontSize=8.2, textColor=HexColor("#555555"))

	doc = SimpleDocTemplate(
		output_pdf,
		pagesize=A4,
		leftMargin=18 * mm,
		rightMargin=18 * mm,
		topMargin=16 * mm,
		bottomMargin=16 * mm,
		title="SINCRO - Mapas polares y fórmulas",
		author="SINCRO",
	)

	story: list = []
	story.append(Paragraph("SINCRO — Guía técnica de mapas polares y sincronía", title_style))
	story.append(Paragraph("Resumen de uso clínico, fórmulas y rangos de referencia", small_style))
	story.append(Spacer(1, 2 * mm))
	story.append(HRFlowable(width="100%", thickness=1.2, color=DARK_BLUE))
	story.append(Spacer(1, 4 * mm))

	story.append(Paragraph("1. Qué representa cada mapa polar", section_style))
	story.append(Paragraph("<b>polar_map:</b> mapa polar de fase en 17 segmentos AHA. Muestra la distribución regional del tiempo de activación mecánica.", body_style))
	story.append(Paragraph("<b>polar_map_Δsigned:</b> diferencia circular con signo entre esfuerzo y reposo. Conserva dirección (adelanto/atraso relativo).", body_style))
	story.append(Paragraph("<b>polar_map_Δabs:</b> valor absoluto del cambio stress-rest. Mide magnitud regional del cambio sin dirección.", body_style))
	story.append(Paragraph("<b>polar_perfusion_directa:</b> mapa polar continuo de intensidad perfusional (apex-centro, base-borde).", body_style))
	story.append(Paragraph("<b>bullseye_directo:</b> resumen segmentario AHA de perfusión para comunicación rápida de hallazgos.", body_style))
	story.append(Paragraph("<b>polar_cine_montaje:</b> evolución temporal gate-a-gate del patrón polar, útil cuando la foto estática no alcanza.", body_style))
	story.append(Spacer(1, 3 * mm))

	story.append(Paragraph("2. Fórmulas clave", section_style))
	story.append(Paragraph("Las fases son angulares (0°–360°), por lo que las diferencias deben calcularse en espacio circular.", body_style))
	story.append(Paragraph("<b>Delta circular con signo:</b> Δsigned = ((φ_esfuerzo − φ_reposo + 180) mod 360) − 180", body_style))
	story.append(Paragraph("<b>Delta absoluto:</b> Δabs = |Δsigned|", body_style))
	story.append(Paragraph("<b>Phase SD (°):</b> desviación estándar de fase segmentaria/global. Mayor valor implica mayor dispersión temporal.", body_style))
	story.append(Paragraph("<b>Bandwidth (°):</b> ancho del histograma de fase (habitualmente percentil 95%). Mayor valor implica peor sincronía.", body_style))
	story.append(Paragraph("<b>Entropy:</b> mide desorganización del histograma de fase. GammaSync reporta Shannon en bits y entropy normalizada en % para comparación con literatura.", body_style))
	story.append(Spacer(1, 3 * mm))

	story.append(Paragraph("3. Interpretación clínica práctica", section_style))
	story.append(Paragraph("Regla base: cuanto más dispersa está la fase (mayor PSD/BW/Entropy), mayor probabilidad de asincronía patológica, siempre contrastada con una referencia software-específica.", body_style))
	story.append(Paragraph("En comparación stress-rest, un incremento relevante de PSD/BW en esfuerzo frente a reposo puede sugerir disincronía transitoria post-stress (stunning isquémico).", body_style))
	story.append(Paragraph("Siempre correlacionar con perfusión regional, QRS/BRI, FEVI, contexto clínico y evolución del paciente. No usar las métricas aisladas como indicación de CRT/TRC.", body_style))
	story.append(Spacer(1, 3 * mm))

	story.append(Paragraph("4. Rangos de referencia publicados", section_style))
	ref_rows = [
		["Software", "Métrica", "Límite superior normal publicado"],
		["QGS JSNM 2023", "PSD / BW / Entropy", "H: 12° / 43° / 43% · M: 10° / 32° / 33%"],
		["ECTb JSNM 2023", "PSD / BW", "H: 25° / 50° · M: 19° / 45°"],
		["cREPO JSNM 2023", "PSD / BW / Entropy", "H: 19° / 69° / 57% · M: 13° / 54° / 52%"],
		["HFV JSNM 2023", "PSD / BW", "H: 12° / 42° · M: 8° / 31°"],
	]
	ref_table = Table(ref_rows, colWidths=[42 * mm, 40 * mm, 84 * mm])
	ref_table.setStyle(TableStyle([
		("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
		("TEXTCOLOR", (0, 0), (-1, 0), white),
		("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
		("GRID", (0, 0), (-1, -1), 0.4, HexColor("#c8d0d8")),
		("FONTSIZE", (0, 0), (-1, -1), 9),
		("VALIGN", (0, 0), (-1, -1), "TOP"),
		("LEFTPADDING", (0, 0), (-1, -1), 2.5 * mm),
	]))
	story.append(ref_table)
	story.append(Spacer(1, 2.5 * mm))
	story.append(Paragraph("Nota: los rangos son software-dependientes y poblacionales; no reemplazan validación local ni juicio clínico individual.", small_style))
	story.append(Spacer(1, 3 * mm))

	story.append(Paragraph("5. Datos adicionales útiles para diagnóstico", section_style))
	story.append(Paragraph("• Topografía del cambio: identificar si el delta se concentra en territorios coronarios específicos.", body_style))
	story.append(Paragraph("• Coherencia fase-perfusión: discordancia relevante (alta asincronía con perfusión casi normal, o viceversa) puede requerir revisión adicional.", body_style))
	story.append(Paragraph("• Dinámica en cine: observar si la alteración es persistente en todos los gates o puntual en fases del ciclo.", body_style))
	story.append(Paragraph("• Integración con FEVI/volúmenes: interpretar asincronía junto a función global para priorizar impacto clínico.", body_style))
	story.append(Spacer(1, 2 * mm))

	story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#9aa7b5")))
	story.append(Paragraph(
		"Referencias orientativas: literatura Emory/ASNC en análisis de fase gated SPECT y trabajos sobre comparación stress-rest (incluyendo series clínicas argentinas).",
		small_style,
	))

	doc.build(story)
	return output_pdf
