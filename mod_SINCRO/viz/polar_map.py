"""SINCRO - viz.polar_map — Bullseye 17 segmentos AHA."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge

from core.aha_segments import (
	SECTOR_TO_SEGMENT_APICAL,
	SECTOR_TO_SEGMENT_BASAL,
	SECTOR_TO_SEGMENT_MEDIO,
)
from viz.colormaps import get_phase_cmap, phase_to_rgb


@dataclass
class PolarMapFigure:
	fig: "matplotlib.figure.Figure"
	segment_values: dict[int, float]
	cmap_name: str


def _wedge_midpoint(r_in: float, r_out: float, t1: float, t2: float) -> tuple[float, float]:
	r = (r_in + r_out) / 2.0
	tm = np.deg2rad((t1 + t2) / 2.0)
	return float(r * np.cos(tm)), float(r * np.sin(tm))


def _segment_color(seg_id: int, phase_by_seg: dict[int, float], cmap_name: str):
	if seg_id not in phase_by_seg:
		return (0.35, 0.35, 0.35)
	rgb = phase_to_rgb(np.array([phase_by_seg[seg_id]], dtype=np.float64), cmap_name=cmap_name)[0]
	return tuple(float(v) for v in rgb)


def _draw_bullseye(ax, phase_by_seg: dict[int, float], cmap_name: str, show_values: bool, angle_offset_deg: float = 0.0):
	rings = [
		(0.75, 1.00, 60.0, SECTOR_TO_SEGMENT_BASAL),
		(0.50, 0.75, 60.0, SECTOR_TO_SEGMENT_MEDIO),
		(0.25, 0.50, 90.0, SECTOR_TO_SEGMENT_APICAL),
	]

	for r_in, r_out, step, lut in rings:
		n = len(lut)
		for k in range(n):
			t1 = float(k * step + angle_offset_deg)
			t2 = float((k + 1) * step + angle_offset_deg)
			seg_id = int(lut[k])
			wedge = Wedge(
				(0.0, 0.0),
				r_out,
				t1,
				t2,
				width=(r_out - r_in),
				facecolor=_segment_color(seg_id, phase_by_seg, cmap_name),
				edgecolor="white",
				linewidth=1.2,
			)
			ax.add_patch(wedge)

			if show_values:
				x, y = _wedge_midpoint(r_in, r_out, t1, t2)
				val = phase_by_seg.get(seg_id, np.nan)
				if np.isfinite(val):
					label = f"{seg_id}\n{val:.0f}°"
				else:
					label = f"{seg_id}\n--"
				ax.text(x, y, label, ha="center", va="center", fontsize=8, color="black")

	apex_color = _segment_color(17, phase_by_seg, cmap_name)
	apex = Circle((0.0, 0.0), 0.25, facecolor=apex_color, edgecolor="white", linewidth=1.2)
	ax.add_patch(apex)
	if show_values:
		val = phase_by_seg.get(17, np.nan)
		label = "17\n--" if not np.isfinite(val) else f"17\n{val:.0f}°"
		ax.text(0.0, 0.0, label, ha="center", va="center", fontsize=8, color="black")


def build_polar_map(
	phase_by_seg: dict[int, float],
	cmap_name: str = "hsv",
	angle_offset_deg: float = 0.0,
	show_values: bool = True,
	title: str | None = None,
) -> PolarMapFigure:
	fig, ax = plt.subplots(figsize=(7.5, 7.0))
	ax.set_aspect("equal")
	ax.axis("off")

	_draw_bullseye(ax, phase_by_seg, cmap_name=cmap_name, show_values=show_values, angle_offset_deg=angle_offset_deg)

	# TODO calibrar orientación del bullseye vs MyoVation/GE.

	ax.set_xlim(-1.15, 1.35)
	ax.set_ylim(-1.15, 1.15)
	ax.set_title(title or "Phase Polar Map (AHA 17)")

	cmap = get_phase_cmap(cmap_name)
	sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=360.0), cmap=cmap)
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.04)
	cbar.set_label("Fase (°)")
	cbar.set_ticks([0, 60, 120, 180, 240, 300, 360])

	fig.subplots_adjust(left=0.05, right=0.90, top=0.93, bottom=0.05)
	return PolarMapFigure(fig=fig, segment_values=dict(phase_by_seg), cmap_name=cmap_name)


def build_clinical_phase_panel(
	phase_by_seg: dict[int, float],
	phases_deg: np.ndarray,
	metrics: dict | None = None,
	*,
	cmap_name: str = "hsv",
	title: str | None = None,
):
	"""Panel clínico estilo estación: histograma + polar map con PSD/PHB."""
	phases = np.asarray(phases_deg, dtype=np.float64)
	phases = phases[np.isfinite(phases)]
	if phases.size == 0:
		raise ValueError("phases_deg está vacío o todo NaN.")

	if metrics is None:
		from core.metrics import calculate_phase_metrics

		metrics = calculate_phase_metrics(phases)

	fig = plt.figure(figsize=(12.2, 4.2), facecolor="#d8d8da")
	gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.08)
	ax_hist = fig.add_subplot(gs[0, 0])
	ax_polar = fig.add_subplot(gs[0, 1])

	ax_hist.set_facecolor("black")
	hist_bins = np.linspace(0.0, 360.0, 73)
	hist_vals, hist_edges = np.histogram(phases, bins=hist_bins, range=(0.0, 360.0))
	hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
	hist_widths = np.diff(hist_edges)
	cmap_hist = get_phase_cmap(cmap_name)
	for v, c, w in zip(hist_vals, hist_centers, hist_widths):
		bar_color = cmap_hist(((float(c) % 360.0) / 360.0))
		ax_hist.bar(
			float(c),
			float(v),
			width=float(w) * 0.96,
			align="center",
			color=bar_color,
			edgecolor="#e5e7eb",
			linewidth=0.12,
			alpha=0.96,
		)
	ax_hist.set_xlim(0.0, 360.0)
	ax_hist.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
	ax_hist.tick_params(axis="x", colors="white", labelsize=8)
	ax_hist.tick_params(axis="y", colors="white", labelsize=8)
	ax_hist.set_xlabel("Onset of contraction (degrees)", color="white", fontsize=8)
	ax_hist.set_ylabel("Frequency (%)", color="white", fontsize=8)
	ax_hist.grid(axis="y", color="#3a3a3a", linestyle="-", linewidth=0.4, alpha=0.45)
	for spine in ax_hist.spines.values():
		spine.set_color("white")

	phase_sd = float(metrics.get("phase_sd", np.nan))
	bw = float(metrics.get("bandwidth", np.nan))
	ax_hist.text(
		0.53,
		0.83,
		f"PSD-{phase_sd:.2f}°\nPHB-{bw:.0f}°",
		transform=ax_hist.transAxes,
		color="white",
		fontsize=16,
		fontweight="bold",
		ha="left",
		va="top",
	)

	ax_polar.set_facecolor("black")
	ax_polar.set_aspect("equal")
	ax_polar.axis("off")
	_draw_bullseye(ax_polar, phase_by_seg, cmap_name=cmap_name, show_values=False, angle_offset_deg=0.0)
	ax_polar.set_xlim(-1.08, 1.16)
	ax_polar.set_ylim(-1.08, 1.08)

	cmap = get_phase_cmap(cmap_name)
	sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=360.0), cmap=cmap)
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax_polar, fraction=0.065, pad=0.02)
	cbar.set_ticks([])
	cbar.ax.set_facecolor("black")
	cbar.outline.set_edgecolor("white")

	fig.suptitle(title or "SINCRO — Panel polar clínico", fontsize=11.5, fontweight="bold", color="#111827")
	fig.subplots_adjust(left=0.035, right=0.955, top=0.88, bottom=0.12, wspace=0.12)
	return fig


def save_clinical_phase_panel(fig, path: str, dpi: int = 150) -> str:
	fig.savefig(path, dpi=dpi, bbox_inches="tight")
	return path


def save_polar_map(pmfig: "PolarMapFigure", path: str, dpi: int = 150) -> str:
	pmfig.fig.savefig(path, dpi=dpi, bbox_inches="tight")
	return path
