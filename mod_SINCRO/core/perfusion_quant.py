"""Cuantificación relativa de perfusión miocárdica por segmento AHA.

Método clásico de la literatura (mapa polar / muestreo por segmento):
la captación media de cada segmento se normaliza al segmento de máxima
captación (=100%). Los segmentos por debajo de umbrales relativos se
clasifican como hipoperfusión leve o severa, y la comparación
esfuerzo/reposo (reposo normalizado a esfuerzo por diseño, ambos en % del
propio máximo) permite estimar reversibilidad.

NO usa bases de datos de normales: es cuantificación RELATIVA intra-estudio,
orientativa y no diagnóstica. Umbrales por defecto tomados de convenciones
publicadas ampliamente (defecto <70% del máximo; severo <50%; mejora
significativa en reposo >=10 puntos porcentuales).
"""

from __future__ import annotations

import math

import numpy as np

from core.aha_segments import TERRITORY_MAP

# Nomenclatura AHA 17 segmentos (dominio público, AHA Scientific Statement 2002).
SEGMENT_NAMES = {
    1: "Basal anterior", 2: "Basal anteroseptal", 3: "Basal inferoseptal",
    4: "Basal inferior", 5: "Basal inferolateral", 6: "Basal anterolateral",
    7: "Medio anterior", 8: "Medio anteroseptal", 9: "Medio inferoseptal",
    10: "Medio inferior", 11: "Medio inferolateral", 12: "Medio anterolateral",
    13: "Apical anterior", 14: "Apical septal", 15: "Apical inferior",
    16: "Apical lateral", 17: "Ápex",
}

DEFECT_THRESHOLD_PCT = 70.0
SEVERE_THRESHOLD_PCT = 50.0
REVERSIBILITY_DELTA_PCT = 10.0
MIN_PIXELS_PER_SEGMENT = 8


def perfusion_by_segment(perfusion: np.ndarray, segment_map: np.ndarray) -> dict[int, float]:
    """Captación media por segmento AHA normalizada al segmento máximo (=100).

    Parameters
    ----------
    perfusion : (n_slices, H, W) float — cubo de perfusión (media de gates).
    segment_map : (n_slices, H, W) int — mapa AHA 1..17 (0 = fuera).

    Returns
    -------
    dict segmento -> % del máximo (NaN si el segmento no tiene píxeles).
    """
    perfusion = np.asarray(perfusion, dtype=np.float64)
    segment_map = np.asarray(segment_map)
    if perfusion.shape != segment_map.shape:
        raise ValueError("perfusion y segment_map deben tener la misma forma")
    means: dict[int, float] = {}
    for seg_id in range(1, 18):
        vals = perfusion[segment_map == seg_id]
        if vals.size < MIN_PIXELS_PER_SEGMENT:
            means[seg_id] = float("nan")
        else:
            means[seg_id] = float(np.mean(vals))
    finite = [v for v in means.values() if math.isfinite(v) and v > 0]
    if not finite:
        return {k: float("nan") for k in means}
    peak = max(finite)
    return {k: (v / peak * 100.0 if math.isfinite(v) else float("nan")) for k, v in means.items()}


def classify_segment(pct: float) -> str:
    """normal / leve / severo / N/D según % del máximo."""
    if not math.isfinite(pct):
        return "N/D"
    if pct < SEVERE_THRESHOLD_PCT:
        return "severo"
    if pct < DEFECT_THRESHOLD_PCT:
        return "leve"
    return "normal"


def _territory_of(seg_id: int) -> str:
    for terr, segs in TERRITORY_MAP.items():
        if seg_id in segs:
            return terr
    return "N/D"


def perfusion_quant_summary(
    stress_pct: dict[int, float] | None,
    rest_pct: dict[int, float] | None = None,
) -> dict:
    """Tabla por segmento + resumen de extensión/severidad/reversibilidad.

    Returns
    -------
    dict con:
      available, rows (lista por segmento con stress/rest/clasificación),
      stress_defect_segments, stress_severe_segments, extent_pct,
      reversible_segments, fixed_segments, notes.
    """
    if not stress_pct:
        return {"available": False, "reason": "sin datos de perfusión de esfuerzo"}

    rows: list[dict] = []
    defect_ids: list[int] = []
    severe_ids: list[int] = []
    reversible_ids: list[int] = []
    fixed_ids: list[int] = []
    n_valid = 0

    for seg_id in range(1, 18):
        s = float(stress_pct.get(seg_id, float("nan")))
        r = float(rest_pct.get(seg_id, float("nan"))) if rest_pct else float("nan")
        s_class = classify_segment(s)
        reversib = ""
        if math.isfinite(s):
            n_valid += 1
        if s_class in ("leve", "severo"):
            defect_ids.append(seg_id)
            if s_class == "severo":
                severe_ids.append(seg_id)
            if math.isfinite(r):
                if (r - s) >= REVERSIBILITY_DELTA_PCT:
                    reversible_ids.append(seg_id)
                    reversib = "reversible"
                else:
                    fixed_ids.append(seg_id)
                    reversib = "fijo"
        rows.append({
            "segment": seg_id,
            "name": SEGMENT_NAMES[seg_id],
            "territory": _territory_of(seg_id),
            "stress_pct": s,
            "rest_pct": r,
            "stress_class": s_class,
            "reversibility": reversib,
        })

    extent_pct = (len(defect_ids) / n_valid * 100.0) if n_valid else float("nan")
    notes = [
        "Cuantificación relativa intra-estudio (captación por segmento AHA en % del segmento máximo).",
        f"Defecto: <{DEFECT_THRESHOLD_PCT:.0f}% del máximo; severo: <{SEVERE_THRESHOLD_PCT:.0f}%; "
        f"reversible: mejora >={REVERSIBILITY_DELTA_PCT:.0f} puntos en reposo.",
        "Sin base de datos de normales: valores orientativos, no diagnósticos.",
    ]
    return {
        "available": True,
        "rows": rows,
        "n_valid_segments": n_valid,
        "stress_defect_segments": defect_ids,
        "stress_severe_segments": severe_ids,
        "extent_pct": extent_pct,
        "reversible_segments": reversible_ids,
        "fixed_segments": fixed_ids,
        "has_rest": bool(rest_pct),
        "notes": notes,
    }
