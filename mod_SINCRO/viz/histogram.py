"""SINCRO - viz.histogram — Histograma de fase (0-360°) + métricas."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from core.metrics import calculate_phase_metrics, circular_mean_deg


def build_phase_histogram(
	phases_deg: np.ndarray,
	metrics: dict | None = None,
	bins: int = 72,
	title: str | None = None,
	comparison_phases_deg: np.ndarray | None = None,
	comparison_metrics: dict | None = None,
	comparison_label: str = "Clínico robusto",
	primary_label: str = "Crudo ROI",
	qc_note: str | None = None,
):
	"""Devuelve la Figure de matplotlib. Si metrics es None, calcula métricas."""
	phases = np.asarray(phases_deg, dtype=np.float64)
	phases = phases[np.isfinite(phases)]
	if phases.size == 0:
		raise ValueError("phases_deg está vacío o todo NaN.")

	if metrics is None:
		metrics = calculate_phase_metrics(phases)

	comparison_phases = None
	if comparison_phases_deg is not None:
		comparison_phases = np.asarray(comparison_phases_deg, dtype=np.float64)
		comparison_phases = comparison_phases[np.isfinite(comparison_phases)]
		if comparison_phases.size == 0:
			comparison_phases = None
		elif comparison_metrics is None:
			comparison_metrics = calculate_phase_metrics(comparison_phases)

	reference_phases = comparison_phases if comparison_phases is not None else phases
	mean_phase = float(circular_mean_deg(reference_phases))
	centered = (reference_phases - mean_phase + 180.0) % 360.0 - 180.0
	p5 = float(np.percentile(centered, 5))
	p95 = float(np.percentile(centered, 95))
	p5_abs = (mean_phase + p5) % 360.0
	p95_abs = (mean_phase + p95) % 360.0

	fig, ax = plt.subplots(figsize=(9.0, 4.8))
	if comparison_phases is not None:
		ax.hist(
			phases,
			bins=bins,
			range=(0.0, 360.0),
			color="#9ca3af",
			alpha=0.42,
			edgecolor="#f8fafc",
			label=primary_label,
		)
		ax.hist(
			comparison_phases,
			bins=bins,
			range=(0.0, 360.0),
			histtype="stepfilled",
			color="#2c7fb8",
			alpha=0.78,
			edgecolor="#0f172a",
			linewidth=0.7,
			label=comparison_label,
		)
	else:
		ax.hist(phases, bins=bins, range=(0.0, 360.0), color="#2c7fb8", alpha=0.85, edgecolor="white")

	ax.axvline(mean_phase, color="#d7191c", linestyle="-", linewidth=2, label=f"Mean {mean_phase:.1f}°")
	ax.axvline(p5_abs, color="#fdae61", linestyle="--", linewidth=1.8, label=f"P5 {p5_abs:.1f}°")
	ax.axvline(p95_abs, color="#fdae61", linestyle="--", linewidth=1.8, label=f"P95 {p95_abs:.1f}°")

	cls = metrics.get("classification", "N/A")
	ax.set_title(title or f"Phase Histogram — {cls}", fontsize=13, fontweight="bold", pad=12)
	ax.set_xlabel("Fase (°)")
	ax.set_ylabel("Frecuencia")
	ax.set_xlim(0.0, 360.0)
	ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
	ax.legend(loc="upper right", fontsize=8)

	if comparison_metrics is not None:
		clinical_cls = comparison_metrics.get("classification", "N/A")
		txt = (
			f"Resultado clínico ({comparison_metrics.get('amp_filter', 'robusto')}): {clinical_cls}\n"
			f"PSD: {comparison_metrics.get('phase_sd', np.nan)}° | BW: {comparison_metrics.get('bandwidth', np.nan)}°\n"
			f"Resultado crudo ({metrics.get('amp_filter', 'ROI')}): {cls}\n"
			f"PSD crudo: {metrics.get('phase_sd', np.nan)}° | BW crudo: {metrics.get('bandwidth', np.nan)}°"
		)
		if qc_note:
			txt += f"\nQC: {qc_note}"
	else:
		txt = (
		f"Phase SD: {metrics.get('phase_sd', np.nan)}°\n"
		f"Bandwidth: {metrics.get('bandwidth', np.nan)}°\n"
		f"Entropy: {metrics.get('entropy', np.nan)}\n"
		f"Peak Phase: {metrics.get('peak_phase', np.nan)}°\n"
		f"Class: {cls}"
		)
	fig.subplots_adjust(left=0.15, right=0.86, top=0.86, bottom=0.16)
	fig.text(
		0.015,
		0.94,
		txt,
		ha="left",
		va="top",
		fontsize=8,
		bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#999999"),
	)
	return fig


def save_histogram(fig, path: str, dpi: int = 150) -> str:
	fig.savefig(path, dpi=dpi, bbox_inches="tight")
	return path
