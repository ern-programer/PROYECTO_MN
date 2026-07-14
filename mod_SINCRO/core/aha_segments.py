"""
SINCRO - core.aha_segments
===========================

Mapeo voxel → 17 segmentos AHA y territorios coronarios.

TERRITORIOS:
  LAD = [1, 2, 7, 8, 13, 14, 17]
  LCx = [5, 6, 11, 12, 16]
  RCA = [3, 4, 9, 10, 15]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.metrics import circular_mean_deg, circular_std_deg

TERRITORY_MAP = {
    "LAD": [1, 2, 7, 8, 13, 14, 17],
    "LCx": [5, 6, 11, 12, 16],
    "RCA": [3, 4, 9, 10, 15],
}

SECTOR_TO_SEGMENT_BASAL = [1, 2, 3, 4, 5, 6]
SECTOR_TO_SEGMENT_MEDIO = [7, 8, 9, 10, 11, 12]
SECTOR_TO_SEGMENT_APICAL = [13, 14, 15, 16]


@dataclass
class AHAResult:
    segment_map: np.ndarray
    apex_to_base_order: list[int]
    n_per_segment: dict[int, int]


def _valid_slices(mask: np.ndarray) -> np.ndarray:
    return np.where(mask.reshape(mask.shape[0], -1).any(axis=1))[0]


def _detect_apex_base_order(seg) -> list[int]:
    valid = _valid_slices(seg.mask)
    if valid.size == 0:
        return []

    first = int(valid[0])
    last = int(valid[-1])

    def score(s: int) -> float:
        area = float(seg.mask[s].sum())
        rin = float(seg.inner_radius[s]) if np.isfinite(seg.inner_radius[s]) else np.inf
        return area + rin * 10.0

    apex_is_first = score(first) <= score(last)
    order_apex_to_base = valid.tolist() if apex_is_first else valid[::-1].tolist()
    return [int(s) for s in order_apex_to_base]


def _slice_level_by_u(u: float) -> str:
    if u < 0.35:
        return "basal"
    if u < 0.70:
        return "medio"
    if u < 0.90:
        return "apical"
    return "apex"


def map_to_17_segments(
    seg: "SegmentationResult",
    angle_offset_deg: float = 0.0,
    clockwise: bool = False,
) -> AHAResult:
    mask = seg.mask.astype(bool)
    n_slices, H, W = mask.shape
    segment_map = np.zeros((n_slices, H, W), dtype=np.int32)

    apex_to_base_order = _detect_apex_base_order(seg)
    if not apex_to_base_order:
        return AHAResult(
            segment_map=segment_map,
            apex_to_base_order=[],
            n_per_segment={i: 0 for i in range(1, 18)},
        )

    base_to_apex = list(reversed(apex_to_base_order))
    L = len(base_to_apex)

    for i, s in enumerate(base_to_apex):
        u = i / max(L - 1, 1)
        level = _slice_level_by_u(float(u))

        cy, cx = seg.center_per_slice[s]
        if not (np.isfinite(cy) and np.isfinite(cx)):
            continue

        ys, xs = np.nonzero(mask[s])
        if ys.size == 0:
            continue

        ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360.0) % 360.0
        if clockwise:
            ang = (-ang) % 360.0
        ang = (ang + float(angle_offset_deg)) % 360.0

        if level == "apex":
            segment_map[s, ys, xs] = 17
            continue

        if level in ("basal", "medio"):
            sectors = (ang // 60.0).astype(int)
            lut = SECTOR_TO_SEGMENT_BASAL if level == "basal" else SECTOR_TO_SEGMENT_MEDIO
            seg_ids = np.array([lut[int(k) % 6] for k in sectors], dtype=np.int32)
            segment_map[s, ys, xs] = seg_ids
            continue

        sectors = (ang // 90.0).astype(int)
        # TODO calibrar orientación exacta sector->segmento vs MyoVation/GE.
        seg_ids = np.array([SECTOR_TO_SEGMENT_APICAL[int(k) % 4] for k in sectors], dtype=np.int32)
        segment_map[s, ys, xs] = seg_ids

    n_per_segment = {i: int(np.sum(segment_map == i)) for i in range(1, 18)}
    return AHAResult(
        segment_map=segment_map,
        apex_to_base_order=apex_to_base_order,
        n_per_segment=n_per_segment,
    )


def phase_by_segment(phase_map: np.ndarray, aha: "AHAResult") -> dict[int, float]:
    """Media CIRCULAR de fase por segmento (usar core.metrics.circular_mean_deg)."""
    out: dict[int, float] = {}
    for seg_id in range(1, 18):
        vals = phase_map[aha.segment_map == seg_id]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[seg_id] = float(circular_mean_deg(vals))
    return out


def territory_analysis(phase_by_seg: dict[int, float]) -> dict[str, dict]:
    """
    Para cada territorio (LAD/LCx/RCA): mean (circular), std (circular), min, max
    de las fases de sus segmentos. Usar core.metrics.circular_mean_deg / circular_std_deg.
    """
    out: dict[str, dict] = {}
    for terr, segments in TERRITORY_MAP.items():
        vals = np.array([phase_by_seg[s] for s in segments if s in phase_by_seg], dtype=np.float64)
        if vals.size == 0:
            out[terr] = {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "n": 0,
            }
            continue
        out[terr] = {
            "mean": float(circular_mean_deg(vals)),
            "std": float(circular_std_deg(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": int(vals.size),
        }
    return out
