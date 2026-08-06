"""Recuperación de resolución (RR) dependiente de profundidad para SPECT.

Modela la respuesta del colimador-detector (CDR): la borrosidad de la PSF crece
con la distancia fuente→detector. Es el núcleo físico del enfoque tipo GE
Evolution / Philips Astonish / Siemens Flash3D / UltraSPECT WBR, pero
**agnóstico al fabricante**: la sigma por profundidad se deriva de la
`CollimatorSpec` (tabla) + el radio de órbita y el pixel spacing del DICOM.

Uso típico (dentro del proyector iterativo):
    psf = PsfModel.from_collimator(spec, radius_mm=270, pixel_mm=6.78)
    blurred = variable_depth_gaussian(rotated_slice, psf)

Para fan-beam (p.ej. GVI OnePass) la geometría estira el eje axial (Y); ver
`correct_axial_magnification`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, zoom

from core.collimator_specs import FWHM_TO_SIGMA, CollimatorSpec


@dataclass(frozen=True)
class PsfModel:
    """Modelo de PSF por profundidad para un corte transaxial reconstruido.

    La distancia fuente→colimador de un vóxel depende de su fila dentro del
    corte rotado: la fila central está a ``radius_mm`` del detector; filas hacia
    el detector están más cerca (menos borrosas) y hacia el fondo, más lejos.
    """

    intrinsic_fwhm_mm: float
    hole_diameter_mm: float
    hole_length_mm: float
    radius_mm: float
    pixel_mm: float
    n_bins: int = 8

    @classmethod
    def from_collimator(cls, spec: CollimatorSpec, *, radius_mm: float, pixel_mm: float,
                        n_bins: int = 8) -> "PsfModel":
        return cls(
            intrinsic_fwhm_mm=float(spec.intrinsic_fwhm_mm),
            hole_diameter_mm=float(spec.hole_diameter_mm),
            hole_length_mm=float(spec.hole_length_mm),
            radius_mm=float(radius_mm),
            pixel_mm=float(pixel_mm),
            n_bins=int(n_bins),
        )

    def sigma_px_for_rows(self, n_rows: int) -> np.ndarray:
        """Sigma (px) de cada fila del corte rotado (fila n-1 = lado detector)."""
        rows = np.arange(int(n_rows), dtype=np.float64)
        center = (n_rows - 1) / 2.0
        # Distancia fuente→colimador: filas hacia el detector (row grande) más cerca.
        b_mm = self.radius_mm - (rows - center) * self.pixel_mm
        b_mm = np.clip(b_mm, 0.0, None)
        r_geom = self.hole_diameter_mm * (self.hole_length_mm + b_mm) / self.hole_length_mm
        r_sys = np.sqrt(r_geom ** 2 + self.intrinsic_fwhm_mm ** 2)
        return (r_sys * FWHM_TO_SIGMA / self.pixel_mm).astype(np.float64)


def variable_depth_gaussian(rot: np.ndarray, psf: PsfModel) -> np.ndarray:
    """Difumina cada fila del corte rotado (eje detector = axis 1) con la sigma
    que le corresponde por profundidad. Cuantiza en ``psf.n_bins`` para hacer
    pocas convoluciones globales (una por bin) en vez de una por fila."""
    rot = np.asarray(rot, dtype=np.float64)
    s = rot.shape[0]
    sigmas = psf.sigma_px_for_rows(s)
    out = np.empty_like(rot)
    lo_all, hi_all = float(sigmas.min()), float(sigmas.max())
    if hi_all - lo_all < 1e-6:
        sig = float(np.mean(sigmas))
        return gaussian_filter1d(rot, sig, axis=1, mode="constant") if sig > 0.05 else rot.copy()
    edges = np.linspace(lo_all, hi_all, int(psf.n_bins) + 1)
    for b in range(int(psf.n_bins)):
        lo, hi = edges[b], edges[b + 1]
        sel = (sigmas >= lo) & (sigmas <= hi if b == psf.n_bins - 1 else sigmas < hi)
        if not np.any(sel):
            continue
        sig = float(np.mean(sigmas[sel]))
        out[sel] = gaussian_filter1d(rot[sel], sig, axis=1, mode="constant") if sig > 0.05 else rot[sel]
    return out


def correct_axial_magnification(projections: np.ndarray, magnification: float) -> np.ndarray:
    """Corrige el estiramiento axial (Y) de un colimador fan-beam de pinholes.

    Un colimador de pinholes verticales magnifica el eje axial por un factor
    ``magnification`` (M = distancia_imagen/distancia_objeto). Para llevar la
    proyección a geometría cuasi-paralela se re-muestrea el eje de filas (H) por
    1/M. Entrada (ang,H,W) o (gates,ang,H,W); devuelve la misma forma.

    NOTA: requiere la ``magnification`` real del datasheet del colimador. Con
    M≈1 (o None) es una identidad; NO inventa la geometría.
    """
    arr = np.asarray(projections, dtype=np.float64)
    m = float(magnification or 1.0)
    if abs(m - 1.0) < 1e-3:
        return arr.copy()
    h_axis = arr.ndim - 2  # eje de filas (H)
    factors = [1.0] * arr.ndim
    factors[h_axis] = 1.0 / m
    resized = zoom(arr, factors, order=1)
    # Recorta/rellena de vuelta al alto original para no romper el resto del pipeline.
    target_h = arr.shape[h_axis]
    cur_h = resized.shape[h_axis]
    if cur_h == target_h:
        return resized
    out = np.zeros(arr.shape, dtype=np.float64)
    n = min(cur_h, target_h)
    src0 = (cur_h - n) // 2
    dst0 = (target_h - n) // 2
    src = [slice(None)] * arr.ndim
    dst = [slice(None)] * arr.ndim
    src[h_axis] = slice(src0, src0 + n)
    dst[h_axis] = slice(dst0, dst0 + n)
    out[tuple(dst)] = resized[tuple(src)]
    return out
