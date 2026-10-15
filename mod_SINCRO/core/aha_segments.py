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

# SA estándar en imagen (y hacia abajo): ANT 270°, SEP 180°, INF 90°, LAT 0°.
# Sector 0 empieza en el borde anterolateral y avanza hacia septal (antihorario en pantalla).
_RING_SECTOR_START_DEG = 300.0
_APICAL_SECTOR_START_DEG = 315.0


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
    """Asigna AHA 17 sobre SA en orientación estándar: anterior arriba, septum a la izquierda."""
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
            sectors = (((_RING_SECTOR_START_DEG - ang) % 360.0) // 60.0).astype(int)
            lut = SECTOR_TO_SEGMENT_BASAL if level == "basal" else SECTOR_TO_SEGMENT_MEDIO
            seg_ids = np.array([lut[int(k) % 6] for k in sectors], dtype=np.int32)
            segment_map[s, ys, xs] = seg_ids
            continue

        sectors = (((_APICAL_SECTOR_START_DEG - ang) % 360.0) // 90.0).astype(int)
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


# Bull's-eye: radios normalizados externos de ápex, apical, medio y basal.
BULLSEYE_RING_RADII = (0.20, 0.45, 0.72, 1.0)
# r del bull's-eye -> u base(0)->ápex(1); coincide con los cortes de _slice_level_by_u.
_BULLSEYE_R_KNOTS = (0.0, 0.20, 0.45, 0.72, 1.0)
_BULLSEYE_U_KNOTS = (1.0, 0.90, 0.70, 0.35, 0.0)
# Despliegue estándar (grados antihorario desde 3 h): segmento 1 arriba, 2 arriba-izquierda.
_APICAL_ANGLE_OFFSET = 45.0
_RING_ANGLE_OFFSET = 60.0


def segment_display_angles(segment: int) -> tuple[float, float]:
    """(theta1, theta2) de pantalla, antihorario desde 3 h, para dibujar un segmento 1..16."""
    if 13 <= segment <= 16:
        start = _APICAL_ANGLE_OFFSET + 90.0 * (segment - 13)
        return start, start + 90.0
    start = _RING_ANGLE_OFFSET + 60.0 * ((segment - 1) % 6)
    return start, start + 60.0


def bullseye_segment_layout() -> list[dict]:
    """Sectores del bull's-eye: radios y ángulos de pantalla (grados, antihorario desde 3 h)."""
    apex_r, apical_r, mid_r, basal_r = BULLSEYE_RING_RADII
    layout = [{"segment": 17, "r0": 0.0, "r1": apex_r, "theta0": 0.0, "theta1": 360.0}]
    for k in range(4):
        theta0 = _APICAL_ANGLE_OFFSET + 90.0 * k
        layout.append({"segment": 13 + k, "r0": apex_r, "r1": apical_r, "theta0": theta0, "theta1": theta0 + 90.0})
    for first, r0, r1 in ((7, apical_r, mid_r), (1, mid_r, basal_r)):
        for k in range(6):
            theta0 = _RING_ANGLE_OFFSET + 60.0 * k
            layout.append({"segment": first + k, "r0": r0, "r1": r1, "theta0": theta0, "theta1": theta0 + 60.0})
    return layout


def bullseye_segment_at(r, theta_deg) -> np.ndarray:
    """Segmento AHA bajo (r normalizado, ángulo de pantalla); 0 fuera del disco."""
    r = np.asarray(r, dtype=np.float64)
    theta = np.mod(np.asarray(theta_deg, dtype=np.float64), 360.0)
    apex_r, apical_r, mid_r, basal_r = BULLSEYE_RING_RADII
    apical = 13 + (np.mod(theta - _APICAL_ANGLE_OFFSET, 360.0) // 90.0).astype(np.int32)
    sector = (np.mod(theta - _RING_ANGLE_OFFSET, 360.0) // 60.0).astype(np.int32)
    return np.select(
        [r < apex_r, r < apical_r, r < mid_r, r <= basal_r],
        [17, apical, 7 + sector, 1 + sector],
        default=0,
    ).astype(np.int32)


def _bullseye_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    coords = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    x, y = np.meshgrid(coords, -coords)
    return np.hypot(x, y), np.mod(np.degrees(np.arctan2(y, x)), 360.0)


def bullseye_segment_image(size: int = 241) -> np.ndarray:
    """Segmento AHA por píxel del bull's-eye (0 fuera del disco)."""
    return bullseye_segment_at(*_bullseye_grid(size))


def bullseye_image(volume: np.ndarray, seg, aha: AHAResult, size: int = 241) -> np.ndarray | None:
    """Mapa polar de perfusión normalizado al máximo (NaN fuera del disco).

    Muestrea la misma máscara, centros y orden base→ápex que map_to_17_segments,
    así los bordes de segmento dibujados coinciden con los usados para el score.
    """
    vol = np.asarray(volume, dtype=np.float64)
    mask = np.asarray(seg.mask, dtype=bool)
    base_to_apex = list(reversed(aha.apex_to_base_order))
    if vol.shape != mask.shape or len(base_to_apex) < 2:
        return None
    profiles = []
    bins = np.arange(360)
    for s in base_to_apex:
        cy, cx = seg.center_per_slice[s]
        ys, xs = np.nonzero(mask[s])
        profile = np.full(360, np.nan)
        if ys.size and np.isfinite(cy) and np.isfinite(cx):
            angle = np.floor(np.mod(np.degrees(np.arctan2(ys - cy, xs - cx)), 360.0)).astype(np.int32) % 360
            values = vol[s, ys, xs]
            for b in np.unique(angle):
                profile[b] = float(np.percentile(values[angle == b], 70))
        finite = np.isfinite(profile)
        if finite.any() and not finite.all():
            xv = bins[finite]
            profile[~finite] = np.interp(bins[~finite], np.r_[xv - 360, xv, xv + 360], np.tile(profile[finite], 3))
        profiles.append(profile)
    profiles = np.asarray(profiles)
    if not np.isfinite(profiles).any():
        return None
    r, theta = _bullseye_grid(size)
    pos = np.interp(r, _BULLSEYE_R_KNOTS, _BULLSEYE_U_KNOTS) * (len(base_to_apex) - 1)
    i0 = np.clip(np.floor(pos).astype(np.int32), 0, len(base_to_apex) - 1)
    i1 = np.clip(i0 + 1, 0, len(base_to_apex) - 1)
    weight = pos - i0
    # Pantalla (y arriba) espeja la imagen SA (y abajo): ángulo de imagen = -theta.
    angle = np.floor(np.mod(-theta, 360.0)).astype(np.int32) % 360
    image = (1.0 - weight) * profiles[i0, angle] + weight * profiles[i1, angle]
    image[r > BULLSEYE_RING_RADII[-1]] = np.nan
    peak = np.nanmax(image)
    if not np.isfinite(peak) or peak <= 0:
        return None
    return image / peak


def bullseye_territory_extent(
    stress_img: np.ndarray,
    rest_img: np.ndarray | None = None,
    defect_threshold: float = 0.70,
    reversible_delta: float = 0.10,
) -> dict[str, dict[str, float | None]]:
    """% del área del mapa polar con defecto (y reversible) por territorio y total."""
    seg_img = bullseye_segment_image(stress_img.shape[0])
    defect = np.nan_to_num(stress_img, nan=1.0) < defect_threshold
    reversible = None
    if rest_img is not None:
        reversible = defect & (np.nan_to_num(rest_img - stress_img, nan=0.0) >= reversible_delta)
    regions = {terr: np.isin(seg_img, segs) for terr, segs in TERRITORY_MAP.items()}
    regions["TOTAL"] = seg_img > 0
    return {
        name: {
            "extent_pct": float(defect[region].mean() * 100.0),
            "reversible_pct": float(reversible[region].mean() * 100.0) if reversible is not None else None,
        }
        for name, region in regions.items()
    }
