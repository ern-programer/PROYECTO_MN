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
):
	"""Devuelve la Figure de matplotlib. Si metrics es None, calcula métricas."""
	phases = np.asarray(phases_deg, dtype=np.float64)
	phases = phases[np.isfinite(phases)]
	if phases.size == 0:
		raise ValueError("phases_deg está vacío o todo NaN.")

	if metrics is None:
		metrics = calculate_phase_metrics(phases)

	mean_phase = float(circular_mean_deg(phases))
	centered = (phases - mean_phase + 180.0) % 360.0 - 180.0
	p5 = float(np.percentile(centered, 5))
	p95 = float(np.percentile(centered, 95))
	p5_abs = (mean_phase + p5) % 360.0
	p95_abs = (mean_phase + p95) % 360.0

	fig, ax = plt.subplots(figsize=(9.0, 4.8))
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
