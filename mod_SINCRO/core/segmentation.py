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

from dataclasses import dataclass
import sys

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    center_of_mass,
    gaussian_filter,
    label,
)

from core import dicom_loader
from core.console_utf8 import enable_utf8


@dataclass
class SegmentationResult:
    mask: np.ndarray
    center_per_slice: np.ndarray
    inner_radius: np.ndarray
    outer_radius: np.ndarray
    method: str
    n_voxels: int


def _largest_component(bin_mask: np.ndarray) -> np.ndarray:
    lbl, n = label(bin_mask)
    if n <= 0:
        return np.zeros_like(bin_mask, dtype=bool)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return lbl == largest


def _slice_center_and_radii(slice_mask: np.ndarray) -> tuple[float, float, float, float]:
    if not np.any(slice_mask):
        return np.nan, np.nan, np.nan, np.nan

    cy, cx = center_of_mass(slice_mask)
    ys, xs = np.nonzero(slice_mask)
    d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    if d.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    r_inner = float(np.percentile(d, 20))
    r_outer = float(np.percentile(d, 80))
    return float(cy), float(cx), r_inner, r_outer


def _segment_auto_or_threshold(
    mean_img: np.ndarray,
    threshold_frac: float,
    smooth_sigma: float,
    with_cleanup: bool,
) -> SegmentationResult:
    n_slices, H, W = mean_img.shape
    mask = np.zeros((n_slices, H, W), dtype=bool)
    centers = np.full((n_slices, 2), np.nan, dtype=np.float64)
    inner = np.full((n_slices,), np.nan, dtype=np.float64)
    outer = np.full((n_slices,), np.nan, dtype=np.float64)

    for s in range(n_slices):
        img = mean_img[s].astype(np.float64)
        if img.size == 0 or float(img.max()) <= 0.0:
            continue

        img_s = gaussian_filter(img, sigma=float(smooth_sigma))
        thr = float(threshold_frac) * float(img_s.max())
        bin_mask = img_s > thr

        if with_cleanup:
            bin_mask = binary_opening(bin_mask, structure=np.ones((3, 3), dtype=bool))
            bin_mask = binary_closing(bin_mask, structure=np.ones((3, 3), dtype=bool))
            bin_mask = _largest_component(bin_mask)

        if int(bin_mask.sum()) < 8:
            continue

        mask[s] = bin_mask
        cy, cx, rin, rout = _slice_center_and_radii(bin_mask)
        centers[s] = [cy, cx]
        inner[s] = rin
        outer[s] = rout

    return SegmentationResult(
        mask=mask,
        center_per_slice=centers,
        inner_radius=inner,
        outer_radius=outer,
        method="auto" if with_cleanup else "threshold",
        n_voxels=int(mask.sum()),
    )


def segment_myocardium(
    cube: np.ndarray,
    method: str = "auto",
    threshold_frac: float = 0.35,
    smooth_sigma: float = 1.0,
    manual_rois: dict | None = None,
) -> SegmentationResult:
    if cube.ndim != 4:
        raise ValueError(f"cube debe ser 4D (n_gates,n_slices,H,W); recibió {cube.shape}")

    mean_img = cube.mean(axis=0)
    n_slices, H, W = mean_img.shape

    if method == "auto":
        return _segment_auto_or_threshold(
            mean_img=mean_img,
            threshold_frac=threshold_frac,
            smooth_sigma=smooth_sigma,
            with_cleanup=True,
        )

    if method == "threshold":
        return _segment_auto_or_threshold(
            mean_img=mean_img,
            threshold_frac=threshold_frac,
            smooth_sigma=smooth_sigma,
            with_cleanup=False,
        )

    if method == "manual":
        rois = manual_rois or {}
        mask = np.zeros((n_slices, H, W), dtype=bool)
        centers = np.full((n_slices, 2), np.nan, dtype=np.float64)
        inner = np.full((n_slices,), np.nan, dtype=np.float64)
        outer = np.full((n_slices,), np.nan, dtype=np.float64)

        ys, xs = np.ogrid[:H, :W]
        for s, roi in rois.items():
            if s < 0 or s >= n_slices:
                continue
            if roi is None or len(roi) != 4:
                continue
            cy, cx, r_inner, r_outer = roi
            d = np.sqrt((ys - float(cy)) ** 2 + (xs - float(cx)) ** 2)
            ring = (d >= float(r_inner)) & (d <= float(r_outer))
            if int(ring.sum()) < 1:
                continue
            mask[s] = ring
            centers[s] = [float(cy), float(cx)]
            inner[s] = float(r_inner)
            outer[s] = float(r_outer)

        return SegmentationResult(
            mask=mask,
            center_per_slice=centers,
            inner_radius=inner,
            outer_radius=outer,
            method="manual",
            n_voxels=int(mask.sum()),
        )

    raise ValueError("method debe ser 'auto', 'manual' o 'threshold'.")


def _cli() -> int:
    enable_utf8()
    if len(sys.argv) < 2:
        print("Uso: python -m core.segmentation <archivo_SA_gated.dcm>")
        return 1

    path = sys.argv[1]
    try:
        study = dicom_loader.load(path, verbose=False)
        seg = segment_myocardium(study.cube, method="auto")
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] {e}")
        return 2

    n_slices_with_ring = int(np.sum(seg.mask.reshape(seg.mask.shape[0], -1).any(axis=1)))
    print(f"Archivo: {path}")
    print(f"Método: {seg.method}")
    print(f"Máscara voxels: {seg.n_voxels}")
    print(f"Slices con anillo: {n_slices_with_ring}/{seg.mask.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
