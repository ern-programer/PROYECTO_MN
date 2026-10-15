"""Lámina de score de perfusión (estilo LCARS) para informes: mapas polares y bull's-eye SSS/SRS/SDS."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge

from core.aha_segments import BULLSEYE_RING_RADII, TERRITORY_MAP, segment_display_angles
from core.perfusion_quant import score_summary_text, territory_scores

BG = "#0a0c12"
PANEL = "#121722"
AMBER = "#cfaa67"
BLUE = "#7898a1"
TEXT = "#c9d4d8"
SCORE_COLORS = {0: "#1f4a52", 1: "#cfaa67", 2: "#d9824a", 3: "#a7615e", 4: "#6b4f8f", None: "#2a2e36"}
SCORE_LABELS = {0: "NORMAL", 1: "EQUÍVOCO", 2: "MODERADO", 3: "SEVERO", 4: "AUSENTE"}
TERRITORY_COLORS = {"LAD": "#e8c27a", "LCx": "#c79bc4", "RCA": "#86c3d6"}
TERRITORY_LABELS = {"LAD": "DA", "LCx": "Cx", "RCA": "CD"}
DEFECT_THRESHOLD = 0.70
REVERSIBLE_DELTA = 0.10


def score_text_color(score) -> str:
    return "#172028" if score in (1, 2) else "#f2f2f2"


def perfusion_rgba(image: np.ndarray, cmap, show_defect: bool = False) -> np.ndarray:
    rgba = np.asarray(cmap(np.nan_to_num(image, nan=0.0)), dtype=np.float64)
    rgba[..., 3] = np.where(np.isnan(image), 0.0, 1.0)
    if show_defect:
        rgba[np.nan_to_num(image, nan=1.0) < DEFECT_THRESHOLD] = (0.0, 0.0, 0.0, 1.0)
    return rgba


def reversibility_rgba(stress: np.ndarray, rest: np.ndarray, show_defect: bool = False) -> np.ndarray:
    diff = np.nan_to_num(rest - stress, nan=0.0)
    level = np.clip(diff / 0.5, 0.0, 1.0)[..., None]
    rgb = (1.0 - level) * np.array([0.35, 0.65, 0.62]) + level * np.array([0.78, 0.91, 0.75])
    alpha = np.where(diff >= 0.05, 0.35 + 0.65 * level[..., 0], 0.0)
    rgba = np.dstack([rgb, alpha])
    if show_defect:
        reversible = (np.nan_to_num(stress, nan=1.0) < DEFECT_THRESHOLD) & (diff >= REVERSIBLE_DELTA)
        rgba[reversible] = (1.0, 1.0, 1.0, 1.0)
    rgba[np.isnan(stress)] = 0.0
    return rgba


def _segment_patch(segment: int, **style):
    if segment == 17:
        return Circle((0.0, 0.0), BULLSEYE_RING_RADII[0], **style)
    rings = {range(1, 7): (BULLSEYE_RING_RADII[2], 1.0), range(7, 13): BULLSEYE_RING_RADII[1:3],
             range(13, 17): BULLSEYE_RING_RADII[0:2]}
    r0, r1 = next(radii for span, radii in rings.items() if segment in span)
    t1, t2 = segment_display_angles(segment)
    return Wedge((0.0, 0.0), r1, t1, t2, width=r1 - r0, **style)


def _label_xy(segment: int) -> tuple[float, float]:
    if segment == 17:
        return 0.0, 0.0
    patch = _segment_patch(segment)
    t1, t2 = segment_display_angles(segment)
    r = patch.r - patch.width / 2.0
    angle = np.radians((t1 + t2) / 2.0)
    return r * np.cos(angle), r * np.sin(angle)


def _setup_axis(ax, title: str, footer: tuple[str, str]):
    ax.set_facecolor(BG)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.18, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title.upper(), color=AMBER, fontsize=11, fontweight="bold", pad=4)
    ax.text(-1.05, -1.15, footer[0], color=AMBER, fontsize=9.5, fontweight="bold", ha="left", va="bottom")
    ax.text(1.05, -1.15, footer[1], color="#81c7ce", fontsize=9.5, fontweight="bold", ha="right", va="bottom")


def _draw_grid(ax, *, territories: bool = True):
    for seg in range(1, 18):
        ax.add_patch(_segment_patch(seg, fill=False, edgecolor=BLUE, linewidth=0.8))
    if territories:
        for terr, segs in TERRITORY_MAP.items():
            for seg in segs:
                ax.add_patch(_segment_patch(seg, fill=False, edgecolor=TERRITORY_COLORS[terr], linewidth=2.0, alpha=0.9))


def _draw_image(ax, rgba: np.ndarray | None):
    ax.add_patch(Circle((0.0, 0.0), 1.0, facecolor="#05070b", edgecolor="none"))
    if rgba is not None:
        image = ax.imshow(rgba, extent=(-1, 1, -1, 1), origin="upper", interpolation="bilinear", zorder=1)
        image.set_clip_path(Circle((0.0, 0.0), 1.0, transform=ax.transData))


def _draw_scores(ax, values: dict[int, int | None], edited: set[int]):
    for seg in range(1, 18):
        value = values.get(seg)
        clipped = None if value is None else int(np.clip(value, 0, 4))
        ax.add_patch(_segment_patch(seg, facecolor=SCORE_COLORS[clipped], edgecolor="none"))
    _draw_grid(ax)
    for seg in range(1, 18):
        value = values.get(seg)
        x, y = _label_xy(seg)
        if seg in edited:
            ax.add_patch(Circle((x, y), 0.085, fill=False, edgecolor="white", linewidth=1.3, zorder=5))
        ax.text(x, y, "N/D" if value is None else str(value), ha="center", va="center", zorder=6,
                fontsize=10 if value is not None else 7.5, fontweight="bold",
                color=score_text_color(None if value is None else int(np.clip(value, 0, 4))))


def build_perfusion_score_figure(scores: dict, stress_img=None, rest_img=None, extent=None, cmap=None):
    """Figura matplotlib equivalente al panel de score (sin dependencias de Qt)."""
    has_rest = bool(scores.get("has_rest"))
    rows = {row["segment"]: row for row in scores.get("rows", [])}
    cols = 3 if has_rest else 1
    fig = plt.figure(figsize=(4.2 * cols + 3.6, 9.4), facecolor=BG)
    grid = fig.add_gridspec(2, cols + 1, width_ratios=[1] * cols + [0.95], hspace=0.12, wspace=0.05,
                            left=0.02, right=0.98, top=0.90, bottom=0.03)
    total = (extent or {}).get("TOTAL", {})

    def pct(value):
        return "" if value is None else f"{value / 68 * 100:.0f}%"

    ax = fig.add_subplot(grid[0, 0])
    _draw_image(ax, perfusion_rgba(stress_img, cmap) if stress_img is not None and cmap is not None else None)
    _draw_grid(ax)
    ext = total.get("extent_pct")
    _setup_axis(ax, "Esfuerzo", (f"EXT {ext:.0f}%" if ext is not None else "", ""))
    ax = fig.add_subplot(grid[1, 0])
    _draw_scores(ax, {s: r["stress_score"] for s, r in rows.items()}, {s for s, r in rows.items() if r["stress_edited"]})
    _setup_axis(ax, "SSS · esfuerzo", (pct(scores.get("sss")) or "INCOMPLETO", f"Σ {scores.get('sss', '—')}"))
    if has_rest:
        ax = fig.add_subplot(grid[0, 1])
        _draw_image(ax, perfusion_rgba(rest_img, cmap) if rest_img is not None and cmap is not None else None)
        _draw_grid(ax)
        _setup_axis(ax, "Reposo", ("", ""))
        ax = fig.add_subplot(grid[0, 2])
        _draw_image(ax, reversibility_rgba(stress_img, rest_img) if stress_img is not None and rest_img is not None else None)
        _draw_grid(ax)
        rev = total.get("reversible_pct")
        _setup_axis(ax, "Reversibilidad", (f"REV {rev:.0f}%" if rev is not None else "", ""))
        ax = fig.add_subplot(grid[1, 1])
        _draw_scores(ax, {s: r["rest_score"] for s, r in rows.items()}, {s for s, r in rows.items() if r["rest_edited"]})
        _setup_axis(ax, "SRS · reposo", (pct(scores.get("srs")) or "INCOMPLETO", f"Σ {scores.get('srs', '—')}"))
        diffs = {s: (None if None in (r["stress_score"], r["rest_score"]) else r["stress_score"] - r["rest_score"])
                 for s, r in rows.items()}
        ax = fig.add_subplot(grid[1, 2])
        _draw_scores(ax, diffs, set())
        _setup_axis(ax, "SDS · diferencia", (pct(scores.get("sds")), f"Σ {scores.get('sds', '—')}"))

    side = fig.add_subplot(grid[:, cols])
    side.set_facecolor(PANEL)
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.set_xticks([])
    side.set_yticks([])
    for spine in side.spines.values():
        spine.set_color("#343945")
    y = 0.96
    keys = ("SSS", "SRS", "SDS") if has_rest else ("SSS",)
    for i, key in enumerate(keys):
        x = 0.08 + i * 0.31
        value = scores.get(key.lower())
        side.text(x, y, key, color=BLUE, fontsize=10, fontweight="bold", va="top")
        side.text(x, y - 0.04, "—" if value is None else str(value), color=AMBER, fontsize=24, fontweight="bold", va="top")
        side.text(x, y - 0.115, pct(value), color="#8a979c", fontsize=8, va="top")
    y = 0.75
    side.text(0.08, y, "TERRITORIOS", color=AMBER, fontsize=10, fontweight="bold")
    headers = ("", "SSS", "SRS", "SDS", "EXT%", "REV%") if has_rest else ("", "SSS", "EXT%")
    xs = np.linspace(0.08, 0.92, len(headers))
    y -= 0.045
    for x, h in zip(xs, headers):
        side.text(x, y, h, color=BLUE, fontsize=8.5, fontweight="bold", ha="center")
    terr = territory_scores(scores)
    terr["TOTAL"] = {"sss": scores.get("sss"), "srs": scores.get("srs"), "sds": scores.get("sds")}
    for name, entry in terr.items():
        y -= 0.045
        is_total = name == "TOTAL"
        if is_total:
            side.plot([0.05, 0.95], [y + 0.03, y + 0.03], color=AMBER, linewidth=0.9)
        ext = (extent or {}).get(name, {})
        values = [entry.get("sss"), entry.get("srs"), entry.get("sds"), ext.get("extent_pct"), ext.get("reversible_pct")]
        if not has_rest:
            values = [entry.get("sss"), ext.get("extent_pct")]
        cells = [TERRITORY_LABELS.get(name, name)] + ["—" if v is None else f"{v:.0f}" for v in values]
        for x, cell in zip(xs, cells):
            color = AMBER if is_total else (TERRITORY_COLORS.get(name, TEXT) if cell == cells[0] else TEXT)
            side.text(x, y, cell, color=color, fontsize=10 if is_total else 9, ha="center",
                      fontweight="bold" if (is_total or cell == cells[0]) else "normal")
    y -= 0.08
    side.text(0.08, y, "ESCALA", color=AMBER, fontsize=10, fontweight="bold")
    for score, label in SCORE_LABELS.items():
        y -= 0.04
        side.add_patch(plt.Rectangle((0.08, y - 0.012), 0.08, 0.03, color=SCORE_COLORS[score]))
        side.text(0.12, y + 0.003, str(score), color=score_text_color(score), fontsize=8.5, fontweight="bold", ha="center", va="center")
        side.text(0.2, y + 0.003, label, color=TEXT, fontsize=8.5, va="center")
    y -= 0.07
    notes = ("Score automático por %máx relativo,\nsin base de normales.\n"
             "○ = corregido manualmente.\n"
             "EXT/REV: % del área del mapa polar.\n"
             "Bordes de color: territorios DA / Cx / CD.")
    side.text(0.08, y, notes, color="#8a979c", fontsize=7.5, va="top", linespacing=1.5)
    fig.suptitle(score_summary_text(scores), color=TEXT, fontsize=11, fontweight="bold", y=0.975)
    return fig


def save_perfusion_score_figure(path: str, scores: dict, **kwargs) -> str:
    fig = build_perfusion_score_figure(scores, **kwargs)
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)
    return path
