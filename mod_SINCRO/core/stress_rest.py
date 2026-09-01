"""
SINCRO - core.stress_rest
=========================

Comparación cuantitativa de fase entre estudios de ESFUERZO y REPOSO.

Ningún software comercial de fase (QGS, SyncTool, ECTb, MyoVation) compara las
métricas de disincronía de fase entre stress y rest. GammaSync ya carga los dos
estudios (bundle primario = esfuerzo, ``compare_bundle`` = reposo), así que tiene
todos los ingredientes para calcular el delta stress-rest, que es la señal con
mayor respaldo predictivo de la literatura reciente:

- **Phase ENTROPY en esfuerzo** predice eventos cardíacos mayores.
  Fukumoto et al. 2025, PMID 40021521.
- **Phase BANDWIDTH** predice remodelado reverso post-CRT.
  Tanaka et al. 2025, PMID 39948439.

Este módulo NO diagnostica ni fija cutoffs: expone los deltas y una interpretación
técnica orientativa citando la fuente. La decisión clínica es del médico.

Convención de signo: ``delta = esfuerzo - reposo``. Un delta POSITIVO de
disincronía (más SD/BW/entropy en esfuerzo) sugiere disincronía INDUCIBLE por
estrés, que es el hallazgo con valor pronóstico en los papers citados.

Todas las métricas angulares (mean_phase, peak_phase, latest_activation_phase)
se comparan con delta CIRCULAR en (-180, 180]; las escalares (phase_sd,
bandwidth, entropy, asynchrony_index) con resta lineal.
"""
from __future__ import annotations

from typing import Any

import numpy as np

#: Métricas escalares (no angulares) que se comparan con resta lineal.
_SCALAR_METRICS = (
    "phase_sd",
    "bandwidth",
    "entropy_shannon_bits",
    "entropy_normalized_pct",
    "asynchrony_index",
    "skewness",
    "kurtosis",
    "peak_width",
)

#: Métricas angulares (0-360) que se comparan con delta circular.
_ANGULAR_METRICS = (
    "mean_phase",
    "peak_phase",
    "latest_activation_phase",
)

#: Umbral orientativo (grados) a partir del cual un aumento de disincronía en
#: esfuerzo se marca como "inducible relevante". NO es un cutoff diagnóstico:
#: es solo una ayuda de lectura. La magnitud del cambio significativo depende de
#: la población y del software de referencia (validar contra DB propia).
INDUCIBLE_SD_DELTA_DEG = 5.0
INDUCIBLE_BW_DELTA_DEG = 15.0


def circular_delta_deg(a: float, b: float) -> float:
    """Delta angular ``a - b`` envuelto a (-180, 180]. NaN si alguno es NaN."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float((a - b + 180.0) % 360.0 - 180.0)


def _scalar_delta(a: Any, b: Any) -> float:
    """Resta lineal segura; NaN si alguno no es finito o no numérico."""
    try:
        av, bv = float(a), float(b)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(av) and np.isfinite(bv)):
        return float("nan")
    return float(av - bv)


def compare_territories(
    stress_territory: dict[str, dict] | None,
    rest_territory: dict[str, dict] | None,
) -> dict[str, dict]:
    """Delta de fase por territorio coronario (LAD/LCx/RCA).

    Para cada territorio devuelve el delta circular de la media de fase y el
    delta lineal del SD. Territorio ausente o vacío → NaN.
    """
    out: dict[str, dict] = {}
    stress_territory = stress_territory or {}
    rest_territory = rest_territory or {}
    for terr in ("LAD", "LCx", "RCA"):
        s = stress_territory.get(terr, {}) or {}
        r = rest_territory.get(terr, {}) or {}
        out[terr] = {
            "stress_mean": float(s.get("mean", np.nan)),
            "rest_mean": float(r.get("mean", np.nan)),
            "delta_mean_circular": circular_delta_deg(
                float(s.get("mean", np.nan)), float(r.get("mean", np.nan))
            ),
            "stress_std": float(s.get("std", np.nan)),
            "rest_std": float(r.get("std", np.nan)),
            "delta_std": _scalar_delta(s.get("std"), r.get("std")),
        }
    return out


def _interpret(deltas: dict[str, float]) -> list[str]:
    """Genera notas técnicas orientativas (no diagnósticas) para el delta."""
    notes: list[str] = []

    d_entropy = deltas.get("entropy_normalized_pct", float("nan"))
    d_bw = deltas.get("bandwidth", float("nan"))
    d_sd = deltas.get("phase_sd", float("nan"))

    if np.isfinite(d_sd):
        if d_sd >= INDUCIBLE_SD_DELTA_DEG:
            notes.append(
                f"Phase SD aumenta {d_sd:+.1f}° en esfuerzo respecto de reposo: "
                "patrón compatible con disincronía INDUCIBLE por estrés."
            )
        elif d_sd <= -INDUCIBLE_SD_DELTA_DEG:
            notes.append(
                f"Phase SD disminuye {d_sd:+.1f}° en esfuerzo: la disincronía es "
                "mayor en reposo (posible mejoría de sincronía con la demanda)."
            )
        else:
            notes.append(
                f"Phase SD estable entre esfuerzo y reposo ({d_sd:+.1f}°): sin "
                "disincronía inducible relevante por este parámetro."
            )

    if np.isfinite(d_entropy):
        notes.append(
            f"Delta de entropy de fase (esfuerzo - reposo) = {d_entropy:+.1f}%. "
            "La entropy de fase en ESFUERZO se asoció a eventos cardíacos mayores "
            "(Fukumoto 2025, PMID 40021521); interpretar en contexto, no como cutoff."
        )

    if np.isfinite(d_bw):
        notes.append(
            f"Delta de bandwidth (esfuerzo - reposo) = {d_bw:+.1f}°. "
            "El bandwidth de fase se asoció a remodelado reverso post-CRT "
            "(Tanaka 2025, PMID 39948439)."
        )

    notes.append(
        "Delta stress-rest de fase: hallazgo EXPLORATORIO. GammaSync no reemplaza "
        "ECG, ecocardiografía/CMR ni la evaluación clínica integral."
    )
    return notes


def compare_stress_rest(
    stress_metrics: dict,
    rest_metrics: dict,
    stress_territory: dict[str, dict] | None = None,
    rest_territory: dict[str, dict] | None = None,
    stress_ef: dict | None = None,
    rest_ef: dict | None = None,
) -> dict:
    """Compara las métricas de fase de esfuerzo vs reposo.

    Parameters
    ----------
    stress_metrics, rest_metrics : dict
        Salidas de ``core.metrics.calculate_phase_metrics`` para el estudio de
        esfuerzo (bundle primario) y de reposo (compare_bundle), respectivamente.
    stress_territory, rest_territory : dict, optional
        Salidas de ``core.aha_segments.territory_analysis`` para cada estudio.

    Returns
    -------
    dict
        ``available`` (bool), ``deltas`` (dict métrica→delta), ``stress`` y
        ``rest`` (valores usados), ``territory`` (delta por territorio) y
        ``notes`` (interpretación técnica orientativa).
    """
    if not stress_metrics or not rest_metrics:
        return {"available": False, "reason": "faltan métricas de esfuerzo o reposo"}

    deltas: dict[str, float] = {}
    stress_used: dict[str, float] = {}
    rest_used: dict[str, float] = {}

    for key in _SCALAR_METRICS:
        s = stress_metrics.get(key)
        r = rest_metrics.get(key)
        deltas[key] = _scalar_delta(s, r)
        stress_used[key] = float(s) if isinstance(s, (int, float)) else float("nan")
        rest_used[key] = float(r) if isinstance(r, (int, float)) else float("nan")

    for key in _ANGULAR_METRICS:
        s = stress_metrics.get(key)
        r = rest_metrics.get(key)
        try:
            deltas[key] = circular_delta_deg(float(s), float(r))
        except (TypeError, ValueError):
            deltas[key] = float("nan")
        stress_used[key] = float(s) if isinstance(s, (int, float)) else float("nan")
        rest_used[key] = float(r) if isinstance(r, (int, float)) else float("nan")

    territory = compare_territories(stress_territory, rest_territory)
    # Función ventricular: no pertenece a métricas de fase, pero forma parte de
    # la comparación clínica stress/rest y el informe debe mostrar ambas FEVI.
    def _ef_snapshot(ef: dict | None) -> dict[str, float]:
        ef = ef or {}
        out: dict[str, float] = {}
        for key in ("ef_pct", "edv_ml", "esv_ml", "pfr_edv_per_s", "tpfr_ms"):
            out[key] = _scalar_delta(ef.get(key), 0.0) if ef.get(key) is not None else float("nan")
        return out
    stress_function = _ef_snapshot(stress_ef)
    rest_function = _ef_snapshot(rest_ef)
    function_deltas = {
        key: _scalar_delta(stress_function.get(key), rest_function.get(key))
        for key in stress_function
    }

    return {
        "available": True,
        "convention": "delta = esfuerzo - reposo",
        "deltas": deltas,
        "stress": stress_used,
        "rest": rest_used,
        "territory": territory,
        "stress_function": stress_function,
        "rest_function": rest_function,
        "function_deltas": function_deltas,
        "notes": _interpret(deltas),
        "references": [
            "Fukumoto 2025 (PMID 40021521): phase entropy en esfuerzo predice eventos cardíacos mayores.",
            "Tanaka 2025 (PMID 39948439): phase bandwidth predice remodelado reverso post-CRT.",
        ],
    }
