"""
SINCRO - core.metrics
======================

Métricas de asincronía a partir de la distribución de fase.

PENDIENTE (Fase 1): Phase SD, Bandwidth (P95-P5), Skewness, Kurtosis, Entropy (Shannon),
Asynchrony Index, latest activation site, peak phase + peak width, clasificación.

Valores de referencia (literatura): Phase SD Normal <20° / Severo >60°;
Bandwidth Normal <60° / Severo >120°; Entropy Normal <4.0 / Severo >6.0.
"""
from __future__ import annotations

import numpy as np


def calculate_phase_metrics(phases_deg: np.ndarray) -> dict:
    """STUB — implementar en Fase 1."""
    raise NotImplementedError("calculate_phase_metrics: implementar en Fase 1.")
