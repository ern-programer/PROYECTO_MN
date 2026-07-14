"""
SINCRO - core.aha_segments
===========================

Mapeo voxel → 17 segmentos AHA y territorios coronarios.

TERRITORIOS:
  LAD = [1, 2, 7, 8, 13, 14, 17]
  LCx = [5, 6, 11, 12, 16]
  RCA = [3, 4, 9, 10, 15]

PENDIENTE (Fase 3): detección apex/base + mapeo a 17 segmentos.
"""
from __future__ import annotations

TERRITORY_MAP = {
    "LAD": [1, 2, 7, 8, 13, 14, 17],
    "LCx": [5, 6, 11, 12, 16],
    "RCA": [3, 4, 9, 10, 15],
}


def map_to_17_segments(*args, **kwargs):
    """STUB — implementar en Fase 3."""
    raise NotImplementedError("map_to_17_segments: implementar en Fase 3.")
