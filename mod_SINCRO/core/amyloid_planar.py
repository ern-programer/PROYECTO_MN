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


@dataclass
class BoneQualityResult:
    """Índice de calidad ósea Q_bone para evaluar contaminación esquelética."""
    q_bone: float          # C_sternum / C_rib_near_heart
    sternum_counts: float
    rib_counts: float
    sternum_area_px: int
    rib_area_px: int
    roi_sternum: ROICircle
    roi_rib: ROICircle

    @property
    def interpretation(self) -> str:
        if self.q_bone > 1.3:
            return "Hueso homogéneo, menor contaminación costal"
        if self.q_bone > 0.7:
            return "Captación ósea homogénea"
        return "Posible captación extraósea en costilla (verificar)"


def compute_q_bone(
    image: np.ndarray,
    roi_sternum: ROICircle,
    roi_rib: ROICircle,
) -> BoneQualityResult:
    """Calcula Q_bone = cuentas_esternón / cuentas_costilla_cerca_corazón.

    Índice de homogeneidad ósea. No modifica el HMR.
    Q_bone ≈ 1 → hueso homogéneo, contaminación predecible.
    Q_bone >> 1 → esternón más caliente que costilla, menor contaminación.
    Q_bone << 1 → alerta: posible captación extraósea en costilla.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.size == 0:
        raise ValueError("Imagen vacía")

    mask_s = roi_sternum.mask(img.shape)
    mask_r = roi_rib.mask(img.shape)

    sternum_counts = float(img[mask_s].mean()) if mask_s.any() else 0.0
    rib_counts = float(img[mask_r].mean()) if mask_r.any() else 0.0

    q_bone = sternum_counts / max(rib_counts, 1e-8)

    return BoneQualityResult(
        q_bone=q_bone,
        sternum_counts=sternum_counts,
        rib_counts=rib_counts,
        sternum_area_px=roi_sternum.area_px(img.shape),
        rib_area_px=roi_rib.area_px(img.shape),
        roi_sternum=roi_sternum,
        roi_rib=roi_rib,
    )


def apply_visual_filter(image: np.ndarray, filter_name: str, **kwargs) -> np.ndarray:
    """Aplica un filtro visual para facilitar el posicionamiento de ROIs.

    IMPORTANTE: estos filtros son SOLO para visualización. El HMR siempre
    se calcula sobre la imagen original sin filtrar.

    Filtros disponibles:
    - "bone_subtract": resta estimación de contribución ósea.
    - "denoise_gauss": suavizado gaussiano.
    - "denoise_median": suavizado mediano (mejor para ruido Poisson).
    - "clahe": ecualización adaptativa de histograma.
    - "high_contrast": estiramiento de contraste percentílico.
    - "invert": inversión de intensidad.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.size == 0:
        return img

    if filter_name == "bone_subtract":
        # Estimación de hueso: umbral alto (percentil 92) + dilatación.
        from scipy.ndimage import binary_dilation, gaussian_filter
        threshold = np.percentile(img, kwargs.get("bone_percentile", 92))
        bone_mask = img > threshold
        # Dilatar para cubrir bordes del hueso.
        iterations = kwargs.get("dilate_iterations", 3)
        bone_mask = binary_dilation(bone_mask, iterations=iterations)
        # Estimar contribución ósea como mediana de los píxeles óseos.
        bone_level = float(np.median(img[bone_mask])) if bone_mask.any() else 0.0
        # Factor de sustracción (0-1, default 0.5 para no sobre-restar).
        alpha = kwargs.get("alpha", 0.5)
        result = img.copy()
        result[bone_mask] = np.clip(result[bone_mask] - alpha * bone_level, 0, None)
        return result

    elif filter_name == "denoise_gauss":
        from scipy.ndimage import gaussian_filter
        sigma = kwargs.get("sigma", 2.0)
        return gaussian_filter(img, sigma=sigma)

    elif filter_name == "denoise_median":
        from scipy.ndimage import median_filter
        size = kwargs.get("size", 3)
        return median_filter(img, size=size)

    elif filter_name == "clahe":
        # CLAHE (Contrast Limited Adaptive Histogram Equalization).
        from skimage import exposure
        # Normalizar a 0-1.
        p_low, p_high = np.percentile(img, [1, 99])
        norm = np.clip((img - p_low) / max(p_high - p_low, 1e-8), 0, 1)
        # CLAHE con clip limit.
        clip_limit = kwargs.get("clip_limit", 0.02)
        result = exposure.equalize_adapthisttt(norm, clip_limit=clip_limit)
        return result * (p_high - p_low) + p_low

    elif filter_name == "high_contrast":
        p_low = kwargs.get("p_low", 2)
        p_high = kwargs.get("p_high", 98)
        lo, hi = np.percentile(img, [p_low, p_high])
        return np.clip((img - lo) / max(hi - lo, 1e-8), 0, 1) * (hi - lo) + lo

    elif filter_name == "invert":
        return img.max() - img

    return img


# Filtros visuales disponibles para el selector de la UI.
VISUAL_FILTERS = {
    "none": ("Sin filtro", {}),
    "bone_subtract": ("Sustracción ósea", {"bone_percentile": 92, "dilate_iterations": 3, "alpha": 0.5}),
    "denoise_gauss": ("Denoise gaussiano", {"sigma": 2.0}),
    "denoise_median": ("Denoise mediano", {"size": 3}),
    "clahe": ("CLAHE (contraste adaptativo)", {"clip_limit": 0.02}),
    "high_contrast": ("Alto contraste", {"p_low": 2, "p_high": 98}),
    "invert": ("Invertir", {}),
}


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
