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

	fig.tight_layout()
	return PolarMapFigure(fig=fig, segment_values=dict(phase_by_seg), cmap_name=cmap_name)


def save_polar_map(pmfig: "PolarMapFigure", path: str, dpi: int = 150) -> str:
	pmfig.fig.savefig(path, dpi=dpi, bbox_inches="tight")
	return path
