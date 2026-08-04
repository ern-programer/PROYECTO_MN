"""
SINCRO - core.filling_metrics
==============================

Métricas de llenado ventricular (función diastólica) derivadas de la curva de
volumen por gate, siguiendo la convención del Emory Cardiac Toolbox (ECTb).

- PFR (Peak Filling Rate): máxima tasa de llenado durante la diástole.
  El ECTb lo reporta con DOBLE normalización:
    * VTD/s  → fracción del volumen de fin de diástole por segundo.
    * VTD/RR → fracción del volumen de fin de diástole por intervalo RR.
- TVmáx (Time to Peak Filling): tiempo desde fin de sístole (ES) hasta el
  instante del PFR, en ms y como % del intervalo RR.

Base geométrica (independiente de la duración del RR):
  Con N gates que cubren un intervalo RR completo, la derivada por gate es
  dV/dgate. La normalización por RR NO necesita la duración real del RR:
    PFR[VTD/RR]  = max(dV/dgate en llenado) * N / EDV
    TVmáx[%RR]   = (gate_pico − gate_ES) / N * 100
  Solo la conversión a unidades absolutas (VTD/s y ms) requiere el RR real:
    PFR[VTD/s]   = PFR[VTD/RR] / (RR_s)
    TVmáx[ms]    = TVmáx[%RR]/100 * RR_ms

Referencias: Emory Cardiac Toolbox, curva tiempo-volumen gated SPECT (pantallas
a_07/b_06). PFR y TVmáx son marcadores establecidos de disfunción diastólica.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def compute_filling_metrics(
    gate_volumes_ml: Sequence[float],
    edv_ml: float,
    es_gate: int,
    rr_ms: Optional[float] = None,
) -> dict[str, Any]:
    """Calcula PFR y TVmáx a partir de la curva de volumen por gate.

    Args:
        gate_volumes_ml: volúmenes de cavidad por gate (un ciclo RR completo).
        edv_ml: volumen de fin de diástole (mL), usado para normalizar.
        es_gate: gate de fin de sístole (1-based, como lo reporta el estimador FEVI).
        rr_ms: intervalo RR medio en ms. Si es None/no confiable, se omiten las
            unidades absolutas (VTD/s y ms) pero SÍ se calculan VTD/RR y %RR.

    Returns:
        dict con claves:
            available (bool)
            pfr_edv_per_rr (float | None)  — PFR en VTD/RR (siempre si hay curva)
            pfr_edv_per_s  (float | None)  — PFR en VTD/s (requiere rr_ms)
            tpfr_pct_rr    (float | None)  — TVmáx como % del RR (siempre si hay curva)
            tpfr_ms        (float | None)  — TVmáx en ms (requiere rr_ms)
            pfr_gate       (int | None)    — gate del pico de llenado (1-based)
            rr_ms_used     (float | None)  — RR usado, o None
    """
    unavailable = {
        "available": False,
        "pfr_edv_per_rr": None,
        "pfr_edv_per_s": None,
        "tpfr_pct_rr": None,
        "tpfr_ms": None,
        "pfr_gate": None,
        "rr_ms_used": None,
    }

    v = np.asarray(gate_volumes_ml, dtype=np.float64).ravel()
    n = int(v.size)
    if n < 3 or not np.isfinite(v).all():
        return unavailable
    try:
        edv = float(edv_ml)
    except (TypeError, ValueError):
        return unavailable
    if edv <= 0.0:
        return unavailable

    # Derivada cíclica por gate (curva periódica: gate N conecta con gate 1).
    prev = np.roll(v, 1)
    nxt = np.roll(v, -1)
    dv_dgate = (nxt - prev) / 2.0  # mL por gate

    es_idx = int(es_gate) - 1
    if es_idx < 0 or es_idx >= n:
        es_idx = int(np.argmin(v))

    # Fase de llenado: desde ES hacia adelante hasta el siguiente ED (cíclico).
    # Se recorre todo el ciclo empezando en ES y se toma el máximo dV/dgate
    # positivo, que corresponde al llenado rápido diastólico.
    order = [(es_idx + k) % n for k in range(n)]
    fill_rates = np.array([dv_dgate[i] for i in order], dtype=np.float64)
    k_peak = int(np.argmax(fill_rates))
    peak_rate = float(fill_rates[k_peak])
    if peak_rate <= 0.0:
        return unavailable
    peak_idx = order[k_peak]

    # VTD/RR y %RR no dependen de la duración real del RR (solo de N gates).
    pfr_edv_per_rr = peak_rate * float(n) / edv
    tpfr_pct_rr = float(k_peak) / float(n) * 100.0

    pfr_edv_per_s: Optional[float] = None
    tpfr_ms: Optional[float] = None
    rr_used: Optional[float] = None
    if rr_ms is not None:
        try:
            rr_val = float(rr_ms)
        except (TypeError, ValueError):
            rr_val = 0.0
        if 250.0 <= rr_val <= 2500.0:  # RR fisiológico (24–240 lpm)
            rr_used = rr_val
            rr_s = rr_val / 1000.0
            pfr_edv_per_s = pfr_edv_per_rr / rr_s
            tpfr_ms = tpfr_pct_rr / 100.0 * rr_val

    return {
        "available": True,
        "pfr_edv_per_rr": float(pfr_edv_per_rr),
        "pfr_edv_per_s": pfr_edv_per_s,
        "tpfr_pct_rr": float(tpfr_pct_rr),
        "tpfr_ms": tpfr_ms,
        "pfr_gate": int(peak_idx) + 1,
        "rr_ms_used": rr_used,
    }


def format_pfr(fm: dict[str, Any]) -> str:
    """Formato ECTb del PFR: 'X.XX VTD/s [Y.YY VTD/RR]'.

    Si no hay RR confiable, muestra solo la normalización por RR."""
    if not fm or not fm.get("available"):
        return "N/D"
    per_rr = fm.get("pfr_edv_per_rr")
    per_s = fm.get("pfr_edv_per_s")
    if per_rr is None:
        return "N/D"
    if per_s is None:
        return f"{per_rr:.2f} VTD/RR"
    return f"{per_s:.2f} VTD/s [{per_rr:.2f} VTD/RR]"


def format_tvmax(fm: dict[str, Any]) -> str:
    """Formato ECTb del TVmáx: 'X ms [Y %RR]'.

    Si no hay RR confiable, muestra solo el % del RR."""
    if not fm or not fm.get("available"):
        return "N/D"
    pct = fm.get("tpfr_pct_rr")
    ms = fm.get("tpfr_ms")
    if pct is None:
        return "N/D"
    if ms is None:
        return f"{pct:.0f} %RR"
    return f"{ms:.0f} ms [{pct:.0f} %RR]"
