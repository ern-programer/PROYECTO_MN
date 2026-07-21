"""Desgatillado (ungating) de estudios SPECT gated.

Convierte un cubo gated 4D (n_gates, n_slices, H, W) en un volumen estático
de perfusión total (n_slices, H, W) sumando todos los gates.

Base física: cada gate guarda la fracción de cuentas de su fase del R-R;
la suma de todos los gates recupera el estudio de perfusión completo.

Matiz: la suma simple es correcta si los gates tienen duración uniforme
(lo habitual en Xeleris/Odyssey). Si los gates fueran de duración variable,
habría que ponderar por FrameTime/RRIntervalVector de cada gate.
"""
from __future__ import annotations

import numpy as np


def ungate(cube_gated: np.ndarray, gate_durations: np.ndarray | None = None) -> np.ndarray:
    """
    Desgatilla un cubo gated 4D.

    Parameters
    ----------
    cube_gated : ndarray (n_gates, n_slices, H, W)
        Cubo gated de entrada.
    gate_durations : ndarray (n_gates,), optional
        Duración relativa de cada gate (ms o fracción). Si se provee y no es
        uniforme, se pondera la suma. Si es None, se asume duración uniforme
        y se hace suma simple (equivalente al UngRaw de Odyssey).

    Returns
    -------
    ndarray (n_slices, H, W)
        Volumen de perfusión total (desgatillado).
    """
    cube = np.asarray(cube_gated, dtype=np.float64)
    if cube.ndim != 4:
        raise ValueError(f"cube_gated debe ser 4D (n_gates,n_slices,H,W); recibió {cube.shape}")

    if gate_durations is None:
        return cube.sum(axis=0)

    dur = np.asarray(gate_durations, dtype=np.float64)
    if dur.shape[0] != cube.shape[0]:
        raise ValueError("gate_durations debe tener n_gates elementos")
    if np.allclose(dur, dur[0]):
        return cube.sum(axis=0)
    # Ponderación por duración (gates no uniformes)
    w = dur / dur.sum()
    return np.tensordot(w, cube, axes=(0, 0))


def ungate_stats(cube_gated: np.ndarray) -> dict:
    """Estadísticas del desgatillado para QC."""
    cube = np.asarray(cube_gated, dtype=np.float64)
    if cube.ndim != 4:
        return {}
    per_gate = cube.sum(axis=(1, 2, 3))
    total = float(per_gate.sum())
    return {
        "n_gates": int(cube.shape[0]),
        "total_counts": total,
        "counts_per_gate_mean": float(per_gate.mean()),
        "counts_per_gate_cv_pct": float(100.0 * per_gate.std() / per_gate.mean()) if per_gate.mean() > 0 else 0.0,
        "gate_counts": per_gate.tolist(),
    }
