"""
SINCRO - core.phase_analysis
=============================

Análisis de fase por voxel (algoritmo de Emory, Chen 2005, PMID 16344229).

Para cada voxel dentro de la máscara miocárdica se toma su curva de actividad a lo
largo del ciclo cardíaco (n_gates puntos) y se calcula, vía FFT:
- Amplitud del primer armónico |F(1)|  → magnitud del cambio de actividad.
- Fase del primer armónico  φ = atan2(Im, Re)  → momento de contracción (0-360°).

Correcciones respecto del plan original (definidas en el repaso):
1. La fase del primer armónico se calcula EXACTA del bin k=1 de la FFT (no depende de
   interpolación). Con pocos gates la DISTRIBUCIÓN de fases queda cuantizada; para
   suavizar histograma/curvas se ofrece interpolación trigonométrica (zero-padding)
   opcional, pero la fase por voxel se obtiene directo y sin sesgo del bin k=1.
2. PONDERACIÓN/FILTRADO POR AMPLITUD: voxels con |F1| baja (ruido/fondo) tienen fase
   poco confiable → se filtran por umbral relativo de amplitud.
3. REFERENCIA DE FASE NORMALIZADA (opcional): anclar 0° al pico de contracción global
   (media circular ponderada por amplitud) para comparabilidad entre estudios.
4. MULTI-ARMÓNICO (opcional): combinar k=1..K para reducir ruido (SyncTool usa multi).

Entrada:  cube (n_gates, n_slices, H, W) + mask (n_slices, H, W) booleana.
Salida:   PhaseResult con phases_deg, amplitudes y mapas 3D.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhaseResult:
    """Resultado del análisis de fase."""

    phases_deg: np.ndarray        # (n_voxels,) fase 0-360 de cada voxel miocárdico
    amplitudes: np.ndarray        # (n_voxels,) amplitud |F1| de cada voxel
    voxel_coords: np.ndarray      # (n_voxels, 3) coords (slice, y, x) de cada voxel
    phase_map: np.ndarray         # (n_slices, H, W) mapa de fase (NaN fuera de máscara)
    amplitude_map: np.ndarray     # (n_slices, H, W) mapa de amplitud (0 fuera de máscara)
    n_gates: int
    harmonics: int
    amplitude_threshold_frac: float
    n_voxels_kept: int
    n_voxels_total: int


def _fourier_phase_amplitude(curves: np.ndarray, harmonics: int = 1):
    """
    FFT por fila y extracción de fase/amplitud del (o los) armónico(s).

    curves : (n_voxels, n_gates)
    harmonics : 1 = solo primer armónico (estándar). >1 = suma vectorial de k=1..K.

    Devuelve (phases_rad, amplitudes) con phases en (-pi, pi].
    """
    n_gates = curves.shape[1]
    # Quitar la componente DC restando la media (mejora estabilidad numérica)
    curves = curves - curves.mean(axis=1, keepdims=True)
    fft = np.fft.fft(curves, axis=1)

    if harmonics <= 1:
        comp = fft[:, 1]
    else:
        kmax = min(harmonics, n_gates // 2)
        comp = fft[:, 1:kmax + 1].sum(axis=1)

    amplitudes = np.abs(comp)
    # Convención de fase de contracción (Emory/MUGA): para una curva
    # A(t) = DC + amp·cos(2π t/N - φ), la fase física φ = -angle(FFT[1]).
    # (numpy define FFT con e^{-j2πkt/N}, por lo que angle(FFT[1]) = -φ.)
    phases_rad = -np.angle(comp)
    return phases_rad, amplitudes


def _weighted_circular_mean_deg(phases_deg: np.ndarray, weights: np.ndarray) -> float:
    """Media circular ponderada (en grados). Correcta para ángulos (evita el wrap 0/360)."""
    if phases_deg.size == 0:
        return 0.0
    rad = np.radians(phases_deg)
    w = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
    c = np.sum(w * np.cos(rad))
    s = np.sum(w * np.sin(rad))
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def phase_analysis(
    cube: np.ndarray,
    mask: np.ndarray,
    harmonics: int = 1,
    amplitude_threshold_frac: float = 0.10,
    normalize_reference: bool = False,
) -> PhaseResult:
    """
    Análisis de fase de un estudio gated ya reconstruido.

    Parameters
    ----------
    cube : ndarray (n_gates, n_slices, H, W)
        Volumen gated (salida del dicom_loader; frame sumado ya descartado).
    mask : ndarray (n_slices, H, W) bool
        Máscara del miocardio. Solo se analizan los voxels True.
    harmonics : int
        1 = primer armónico (Emory estándar). >1 = multi-armónico (reduce ruido).
    amplitude_threshold_frac : float
        Fracción del máximo de amplitud por debajo de la cual el voxel se descarta
        (ruido/fondo). 0.10 = descartar voxels con |F1| < 10% del máximo.
    normalize_reference : bool
        Si True, resta la fase global de referencia (media circular ponderada por
        amplitud) para que 0° sea comparable entre estudios.

    Returns
    -------
    PhaseResult
    """
    if cube.ndim != 4:
        raise ValueError(f"cube debe ser 4D (n_gates, n_slices, H, W); recibió {cube.shape}")
    n_gates, n_slices, H, W = cube.shape
    if mask.shape != (n_slices, H, W):
        raise ValueError(f"mask debe ser {(n_slices, H, W)}; recibió {mask.shape}")
    if n_gates < 3:
        raise ValueError(f"Se necesitan >=3 gates para el análisis de fase; hay {n_gates}.")

    mask_bool = mask.astype(bool)
    coords = np.argwhere(mask_bool)                 # (n_voxels, 3) → (slice, y, x)
    if coords.size == 0:
        raise ValueError("La máscara no contiene ningún voxel (miocardio vacío).")

    # Curvas de actividad por voxel: (n_voxels, n_gates)
    s_idx, y_idx, x_idx = coords[:, 0], coords[:, 1], coords[:, 2]
    curves = cube[:, s_idx, y_idx, x_idx].T.astype(np.float64)  # (n_voxels, n_gates)

    phases_rad, amplitudes = _fourier_phase_amplitude(curves, harmonics=harmonics)

    # Filtrado por amplitud (voxels de baja amplitud = ruido)
    amp_max = amplitudes.max() if amplitudes.size else 0.0
    if amp_max > 0:
        keep = amplitudes >= (amplitude_threshold_frac * amp_max)
    else:
        keep = np.ones_like(amplitudes, dtype=bool)

    coords_k = coords[keep]
    phases_rad_k = phases_rad[keep]
    amplitudes_k = amplitudes[keep]

    # Fase a grados 0-360
    phases_deg = np.degrees(phases_rad_k) % 360.0

    # Referencia normalizada (opcional)
    if normalize_reference and phases_deg.size:
        ref = _weighted_circular_mean_deg(phases_deg, amplitudes_k)
        phases_deg = (phases_deg - ref) % 360.0

    # Mapas 3D
    phase_map = np.full((n_slices, H, W), np.nan, dtype=np.float64)
    amplitude_map = np.zeros((n_slices, H, W), dtype=np.float64)
    for i, (s, y, x) in enumerate(coords_k):
        phase_map[s, y, x] = phases_deg[i]
        amplitude_map[s, y, x] = amplitudes_k[i]

    return PhaseResult(
        phases_deg=phases_deg,
        amplitudes=amplitudes_k,
        voxel_coords=coords_k,
        phase_map=phase_map,
        amplitude_map=amplitude_map,
        n_gates=n_gates,
        harmonics=harmonics,
        amplitude_threshold_frac=amplitude_threshold_frac,
        n_voxels_kept=int(keep.sum()),
        n_voxels_total=int(coords.shape[0]),
    )
