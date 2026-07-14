"""
SINCRO - core.phase_analysis
=============================

Análisis de fase por voxel (algoritmo de Emory, Chen 2005).

PENDIENTE (Fase 1 - especificar con Opus, implementar con Codex):
- FFT primer armónico de la curva de actividad por voxel.
- Interpolación temporal (zero-padding a >=64) antes del FFT — OBLIGATORIA con 8 gates.
- Ponderación/filtrado por amplitud |F1| (descartar voxels de baja amplitud = ruido).
- Normalización de referencia de fase (0° = end-diastole global).
- Opción multi-armónico (k=1..3).

Entrada esperada: GatedStudy.cube (n_gates, n_slices, H, W) + máscara miocárdica.
Salida: phases (grados 0-360), amplitudes, por voxel dentro de la máscara.
"""
from __future__ import annotations

import numpy as np


def phase_analysis(cube: np.ndarray, mask: np.ndarray, interpolate_to: int = 64):
    """
    STUB — implementar en Fase 1.

    Parameters
    ----------
    cube : (n_gates, n_slices, H, W)
    mask : (n_slices, H, W) booleana — miocardio.
    interpolate_to : nº de puntos temporales tras interpolación (>= n_gates).

    Returns
    -------
    phases_deg, amplitudes : arrays por voxel miocárdico.
    """
    raise NotImplementedError("phase_analysis: implementar en Fase 1 (Opus especifica → Codex implementa).")
