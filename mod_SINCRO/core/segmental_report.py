"""
SINCRO - core.segmental_report
===============================

Tabla segmentaria AHA que cruza FASE (cuándo se contrae) con PERFUSIÓN (cuánto
capta) y una clasificación de viabilidad derivada de la perfusión. Es el insumo
del panel "Guía para fase VI": una fila por segmento (1..17) + resumen por
territorio coronario (LAD/LCx/RCA).

Diseño: motor de UNA etapa (reposo o esfuerzo). El panel llama a esta función
una o dos veces según haya estudio de comparación, y arma la vista de 1 o 2
columnas. No hay dependencia de Qt: se puede testear en aislamiento.

Perfusión: se reporta como % del máximo segmentario (convención polar habitual en
SPECT: el segmento más captante = 100%). La viabilidad se deriva de ese %:
    ≥ viable_pct           → "viable"      (perfusión conservada)
    dudosa_pct .. viable   → "dudosa"      (hipoperfusión, viabilidad incierta)
    < dudosa_pct           → "no viable"   (probable escara)
Los umbrales son parametrizables; los defaults (50/70%) siguen la práctica MPI.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.aha_segments import TERRITORY_MAP, territory_analysis

#: Nombre AHA estándar por segmento (para la tabla).
SEGMENT_NAMES: dict[int, str] = {
    1: "Basal anterior",
    2: "Basal anteroseptal",
    3: "Basal inferoseptal",
    4: "Basal inferior",
    5: "Basal inferolateral",
    6: "Basal anterolateral",
    7: "Medio anterior",
    8: "Medio anteroseptal",
    9: "Medio inferoseptal",
    10: "Medio inferior",
    11: "Medio inferolateral",
    12: "Medio anterolateral",
    13: "Apical anterior",
    14: "Apical septal",
    15: "Apical inferior",
    16: "Apical lateral",
    17: "Ápice",
}

#: Segmento → territorio coronario (inverso de TERRITORY_MAP).
_SEG_TO_TERRITORY: dict[int, str] = {
    seg: terr for terr, segs in TERRITORY_MAP.items() for seg in segs
}

# Umbrales de viabilidad por perfusión (% del máximo segmentario).
DEFAULT_VIABLE_PCT = 70.0
DEFAULT_DUDOSA_PCT = 50.0


def _viability_class(perf_pct: float, viable_pct: float, dudosa_pct: float) -> str:
    if not np.isfinite(perf_pct):
        return "N/D"
    if perf_pct >= viable_pct:
        return "viable"
    if perf_pct >= dudosa_pct:
        return "dudosa"
    return "no viable"


def perfusion_percent_by_segment(
    perfusion: np.ndarray,
    segment_map: np.ndarray,
) -> dict[int, float]:
    """Perfusión media por segmento AHA, normalizada al máximo segmentario (%).

    El segmento con mayor captación media = 100%. Segmentos ausentes/vacíos
    quedan fuera del dict.
    """
    perfusion = np.asarray(perfusion, dtype=np.float64)
    segment_map = np.asarray(segment_map)
    if perfusion.shape != segment_map.shape:
        raise ValueError(
            f"perfusion {perfusion.shape} y segment_map {segment_map.shape} deben coincidir"
        )
    means: dict[int, float] = {}
    for seg_id in range(1, 18):
        vals = perfusion[segment_map == seg_id]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        means[seg_id] = float(np.mean(vals))
    if not means:
        return {}
    ref = max(means.values())
    if ref <= 0.0:
        return {}
    return {seg: (v / ref * 100.0) for seg, v in means.items()}


def build_segmental_rows(
    phase_by_seg: dict[int, float],
    perfusion: Optional[np.ndarray] = None,
    segment_map: Optional[np.ndarray] = None,
    *,
    perfusion_pct_by_seg: Optional[dict[int, float]] = None,
    viable_pct: float = DEFAULT_VIABLE_PCT,
    dudosa_pct: float = DEFAULT_DUDOSA_PCT,
) -> list[dict[str, Any]]:
    """Arma las 17 filas segmentarias (fase + perfusión % + viabilidad).

    La perfusión se puede pasar ya calculada (``perfusion_pct_by_seg``) o dejar
    que la función la derive de ``perfusion`` + ``segment_map``. Si no hay
    perfusión, las filas traen solo fase (perfusión/viabilidad = N/D).
    """
    if perfusion_pct_by_seg is None and perfusion is not None and segment_map is not None:
        perfusion_pct_by_seg = perfusion_percent_by_segment(perfusion, segment_map)
    perf = perfusion_pct_by_seg or {}
    phase = phase_by_seg or {}

    rows: list[dict[str, Any]] = []
    for seg_id in range(1, 18):
        phase_deg = float(phase.get(seg_id, np.nan))
        perf_pct = float(perf.get(seg_id, np.nan))
        rows.append({
            "segment": seg_id,
            "name": SEGMENT_NAMES.get(seg_id, str(seg_id)),
            "territory": _SEG_TO_TERRITORY.get(seg_id, "N/D"),
            "phase_deg": phase_deg,
            "perfusion_pct": perf_pct,
            "viability": _viability_class(perf_pct, viable_pct, dudosa_pct),
        })
    return rows


def latest_activation_segment(phase_by_seg: dict[int, float]) -> Optional[int]:
    """Segmento con la fase más tardía (mayor grado). None si no hay datos."""
    if not phase_by_seg:
        return None
    finite = {s: v for s, v in phase_by_seg.items() if np.isfinite(v)}
    if not finite:
        return None
    return int(max(finite, key=lambda s: finite[s]))


def territory_summary(
    phase_by_seg: dict[int, float],
    perfusion_pct_by_seg: Optional[dict[int, float]] = None,
) -> dict[str, dict[str, float]]:
    """Resumen por territorio coronario: fase (circular) + perfusión media (%).

    La parte de fase reusa ``territory_analysis``; se le agrega la perfusión
    media del territorio cuando hay datos.
    """
    phase_summary = territory_analysis(phase_by_seg or {})
    perf = perfusion_pct_by_seg or {}
    out: dict[str, dict[str, float]] = {}
    for terr, segs in TERRITORY_MAP.items():
        row = dict(phase_summary.get(terr, {}))
        perf_vals = [perf[s] for s in segs if s in perf and np.isfinite(perf[s])]
        row["perfusion_pct"] = float(np.mean(perf_vals)) if perf_vals else float("nan")
        out[terr] = row
    return out


def build_segmental_report(
    phase_by_seg: dict[int, float],
    perfusion: Optional[np.ndarray] = None,
    segment_map: Optional[np.ndarray] = None,
    *,
    viable_pct: float = DEFAULT_VIABLE_PCT,
    dudosa_pct: float = DEFAULT_DUDOSA_PCT,
) -> dict[str, Any]:
    """Reporte segmentario completo de una etapa, listo para la figura/tabla.

    Returns dict con:
        rows            : list de 17 filas (build_segmental_rows)
        territories     : resumen por territorio (territory_summary)
        perfusion_pct   : dict seg→% (o vacío si no hay perfusión)
        latest_segment  : segmento de activación más tardía
        has_perfusion   : bool
    """
    perfusion_pct_by_seg: dict[int, float] = {}
    if perfusion is not None and segment_map is not None:
        perfusion_pct_by_seg = perfusion_percent_by_segment(perfusion, segment_map)
    rows = build_segmental_rows(
        phase_by_seg,
        perfusion_pct_by_seg=perfusion_pct_by_seg,
        viable_pct=viable_pct,
        dudosa_pct=dudosa_pct,
    )
    return {
        "rows": rows,
        "territories": territory_summary(phase_by_seg, perfusion_pct_by_seg),
        "perfusion_pct": perfusion_pct_by_seg,
        "latest_segment": latest_activation_segment(phase_by_seg),
        "has_perfusion": bool(perfusion_pct_by_seg),
    }
