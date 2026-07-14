"""
SINCRO - core.segmentation
===========================

Segmentación del miocardio del VI.

PENDIENTE (Fase 3):
- Opción A: ROI manual (elipse/contorno dibujado). Rápido y confiable para validar el motor.
- Opción B: automática por thresholding adaptativo.
- Opción C (futuro): LVSD estilo GE (6 etapas).
Ground-truth de segmentación: contornos LVSD de MyoVation (Xeleris).
"""
from __future__ import annotations

import numpy as np


def segment_myocardium(cube: np.ndarray, method: str = "manual"):
    """STUB — implementar en Fase 3."""
    raise NotImplementedError("segment_myocardium: implementar en Fase 3.")
