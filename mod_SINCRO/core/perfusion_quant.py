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

# Escala 0-4 (0 normal ... 4 ausente) por %máx relativo, sin base de normales.
SCORE_THRESHOLDS = ((70.0, 0), (60.0, 1), (50.0, 2), (40.0, 3))
MAX_SEGMENT_SCORE = 4
N_SEGMENTS = 17


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


def score_summary_text(scores: dict) -> str:
    """Línea de informe con SSS/SRS/SDS y estado de revisión médica."""
    def fmt(value):
        return "incompleto" if value is None else str(value)

    parts = [f"SSS {fmt(scores.get('sss'))}"]
    if scores.get("has_rest"):
        parts += [f"SRS {fmt(scores.get('srs'))}", f"SDS {fmt(scores.get('sds'))}"]
    if scores.get("sss_pct") is not None:
        parts.append(f"SSS {scores['sss_pct']:.0f}% del miocardio")
    review = (
        f"revisado por el médico ({scores.get('n_edited', 0)} corrección(es))"
        if scores.get("reviewed") else "automático SIN revisión médica"
    )
    return "Score de perfusión: " + " · ".join(parts) + f" — {review}"


def territory_scores(scores: dict) -> dict[str, dict[str, float | None]]:
    """SSS/SRS/SDS por territorio y % del máximo posible (4 x segmentos)."""
    rows = {row["segment"]: row for row in scores.get("rows", [])}
    out = {}
    for terr, segs in TERRITORY_MAP.items():
        entry = {}
        for stage, key in (("stress", "sss"), ("rest", "srs")):
            values = [rows.get(seg, {}).get(f"{stage}_score") for seg in segs]
            complete = scores.get("has_rest") or stage == "stress"
            entry[key] = int(sum(values)) if complete and None not in values else None
        entry["sds"] = entry["sss"] - entry["srs"] if None not in (entry["sss"], entry["srs"]) else None
        entry["sss_pct"] = entry["sss"] / (MAX_SEGMENT_SCORE * len(segs)) * 100.0 if entry["sss"] is not None else None
        out[terr] = entry
    return out


def auto_segment_score(pct: float) -> int | None:
    """Score 0-4 de un segmento desde su %máx; None si no hay dato."""
    if pct is None or not math.isfinite(float(pct)):
        return None
    for lower, score in SCORE_THRESHOLDS:
        if float(pct) >= lower:
            return score
    return MAX_SEGMENT_SCORE


def perfusion_scores(
    stress_pct: dict[int, float] | None,
    rest_pct: dict[int, float] | None = None,
    overrides: dict[str, dict[int, int | None]] | None = None,
) -> dict:
    """Scores por segmento (auto + corrección manual) y SSS/SRS/SDS.

    ``overrides[stage][segmento]`` reemplaza el score automático; ``None``
    excluye el segmento. Los sumarios se informan solo con los 17 segmentos
    puntuados, para no presentar un SSS parcial como si fuera completo.
    """
    if not stress_pct:
        return {"available": False, "reason": "sin datos de perfusión de esfuerzo"}
    overrides = overrides or {}
    stages = {"stress": stress_pct}
    if rest_pct:
        stages["rest"] = rest_pct
    rows = [{"segment": seg_id, "name": SEGMENT_NAMES[seg_id]} for seg_id in range(1, N_SEGMENTS + 1)]
    totals: dict[str, int | None] = {}
    n_scored: dict[str, int] = {}
    n_edited = 0
    for stage, pct in stages.items():
        edits = overrides.get(stage) or {}
        scores = []
        for row in rows:
            seg_id = row["segment"]
            auto = auto_segment_score(pct.get(seg_id, float("nan")))
            edited = seg_id in edits
            final = edits[seg_id] if edited else auto
            if final is not None:
                final = int(min(MAX_SEGMENT_SCORE, max(0, int(final))))
            edited = edited and final != auto
            n_edited += int(edited)
            row[f"{stage}_auto"] = auto
            row[f"{stage}_score"] = final
            row[f"{stage}_edited"] = edited
            if final is not None:
                scores.append(final)
        n_scored[stage] = len(scores)
        totals[stage] = int(sum(scores)) if len(scores) == N_SEGMENTS else None

    sss = totals.get("stress")
    srs = totals.get("rest")
    sds = sss - srs if sss is not None and srs is not None else None
    return {
        "available": True,
        "has_rest": "rest" in stages,
        "rows": rows,
        "sss": sss,
        "srs": srs,
        "sds": sds,
        "sss_pct": (sss / (MAX_SEGMENT_SCORE * N_SEGMENTS) * 100.0) if sss is not None else None,
        "n_scored": n_scored,
        "n_edited": n_edited,
        "notes": [
            "Score automático 0-4 por %máx relativo: >=70% 0, 60-69% 1, 50-59% 2, 40-49% 3, <40% 4.",
            "Sin base de datos de normales: orientativo; requiere revisión médica.",
            "SSS/SRS solo con los 17 segmentos puntuados; SDS = SSS - SRS. * = corregido manualmente.",
        ],
    }
