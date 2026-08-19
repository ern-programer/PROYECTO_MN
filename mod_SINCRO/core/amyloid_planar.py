# -*- coding: utf-8 -*-
"""Procesamiento de imágenes planar estática para amiloidosis cardíaca.

Carga una imagen planar estática (no gated) y calcula:
- HMR (Heart-to-Mediastinum Ratio) con ROIs circulares.
- Perugini visual score (0-3) con referencia.

ROIs:
- Corazón: círculo draggable sobre el miocardio.
- Mediastino: círculo draggable espejo sobre el hemitórax contralateral
  (técnica de Bokhari, misma área).

Cutoffs usados (Bokhari, ASNC practice points):
- HMR ≥1.5: sugiere ATTR positivo.
- HMR 1.0–1.5: equívoco (complementar con SPECT o repeat a 3h).
- HMR <1.0: negativo.

Uso:
    from core.amyloid_planar import AmyloidPlanarResult, compute_hmr
    result = compute_hmr(image, roi_heart, roi_mediastinum)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ROICircle:
    """ROI circular en coordenadas de imagen (y, x)."""

    cy: float
    cx: float
    radius: float

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        ys, xs = np.arange(h), np.arange(w)
        dist = np.sqrt((ys[:, None] - self.cy) ** 2 + (xs[None, :] - self.cx) ** 2)
        return dist <= self.radius

    def area_px(self, shape: tuple[int, int]) -> int:
        return int(np.count_nonzero(self.mask(shape)))


@dataclass
class AmyloidPlanarResult:
    """Resultado del cálculo de HMR con referencia diagnóstica."""

    hmr: float
    heart_counts: float
    mediastinum_counts: float
    heart_area_px: int
    mediastinum_area_px: int
    roi_heart: ROICircle
    roi_mediastinum: ROICircle

    @property
    def classification(self) -> str:
        if self.hmr >= 1.5:
            return "POSITIVO (sugiere ATTR)"
        if self.hmr >= 1.0:
            return "EQUÍVOCO (complementar con SPECT o repeat a 3h)"
        return "NEGATIVO"

    @property
    def hmr_text(self) -> str:
        return f"HMR = {self.hmr:.2f} ({self.classification})"


def compute_hmr(
    image: np.ndarray,
    roi_heart: ROICircle,
    roi_mediastinum: ROICircle,
) -> AmyloidPlanarResult:
    """Calcula HMR = cuentas ROI corazón / cuentas ROI mediastino.

    Las ROIs son circulares en coordenadas (y, x). La mediastinal
    espejo usa la misma área que la cardíaca.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.size == 0:
        raise ValueError("Imagen vacía")

    mask_h = roi_heart.mask(img.shape)
    mask_m = roi_mediastinum.mask(img.shape)

    heart_counts = float(img[mask_h].sum())
    mediastinum_counts = float(img[mask_m].sum())

    hmr = heart_counts / max(mediastinum_counts, 1e-8)

    return AmyloidPlanarResult(
        hmr=hmr,
        heart_counts=heart_counts,
        mediastinum_counts=mediastinum_counts,
        heart_area_px=roi_heart.area_px(img.shape),
        mediastinum_area_px=roi_mediastinum.area_px(img.shape),
        roi_heart=roi_heart,
        roi_mediastinum=roi_mediastinum,
    )


# Perugini visual score (referencia para lectura)
PERUGINI_SCORES = {
    0: "Negativo: sin captación cardíaca significativa",
    1: "Leve: captación menor que hueso (ésternón/costillas)",
    2: "Moderado: captación igual al hueso con atenuación ósea",
    3: "Intenso: captación mayor que hueso con hueso casi ausente",
}

# Cutoffs de HMR (Bokhari, ASNC)
HMR_CUTOFF_POSITIVE = 1.5
HMR_CUTOFF_NEGATIVE = 1.0
