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
    binary_erosion,
    binary_fill_holes,
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


def _lv_candidate_component(bin_mask: np.ndarray) -> np.ndarray:
    """Elige componente compatible con VI usando prior espacial central/anular.

    Evita capturar focos extracardíacos calientes (p.ej. intestino) que suelen
    aparecer fuera de la región central en estudios de baja resolución.
    """
    bin_mask = np.asarray(bin_mask, dtype=bool)
    lbl, n = label(bin_mask)
    if n <= 0:
        return np.zeros_like(bin_mask, dtype=bool)

    h, w = bin_mask.shape
    min_dim = float(min(h, w))
    cy0 = (h - 1) * 0.5
    cx0 = (w - 1) * 0.5
    ys, xs = np.ogrid[:h, :w]
    rr = np.sqrt((ys - cy0) ** 2 + (xs - cx0) ** 2)
    # Prior anular amplio para cubrir variabilidad de cámara en 22x22 y superiores.
    prior_inner = 0.12 * min_dim
    prior_outer = 0.50 * min_dim
    prior_ring = (rr >= prior_inner) & (rr <= prior_outer)

    best_score = -1e9
    best_mask = np.zeros_like(bin_mask, dtype=bool)
    for comp_id in range(1, n + 1):
        comp = lbl == comp_id
        area = int(np.count_nonzero(comp))
        if area < 4:
            continue

        cy, cx = center_of_mass(comp)
        if not (np.isfinite(cy) and np.isfinite(cx)):
            continue

        dist_center = float(np.sqrt((cy - cy0) ** 2 + (cx - cx0) ** 2))
        dist_norm = dist_center / max(1e-6, 0.5 * min_dim)

        overlap = float(np.count_nonzero(comp & prior_ring)) / float(area)
        filled = binary_fill_holes(comp)
        cavity = filled & (~comp)
        hole_frac = float(np.count_nonzero(cavity)) / max(1.0, float(np.count_nonzero(filled)))
        area_frac = float(area) / max(1.0, float(h * w))

        # Favorecer regiones centrales/anulares con cavidad interna plausible.
        center_score = max(0.0, 1.0 - dist_norm)
        hole_score = min(1.0, hole_frac / 0.18) if hole_frac > 0 else 0.0
        area_score = 1.0 - min(1.0, abs(area_frac - 0.22) / 0.22)
        score = 2.7 * overlap + 1.9 * center_score + 0.7 * hole_score + 0.5 * area_score

        # Penalización fuerte si está muy periférico (típico hot bowel inferior/lateral).
        if dist_norm > 0.95:
            score -= 2.5

        if score > best_score:
            best_score = score
            best_mask = comp

    if np.any(best_mask):
        return best_mask
    return _largest_component(bin_mask)


def _slice_center_and_radii(slice_mask: np.ndarray) -> tuple[float, float, float, float]:
    if not np.any(slice_mask):
        return np.nan, np.nan, np.nan, np.nan

    ring = _largest_component(np.asarray(slice_mask, dtype=bool))
    if not np.any(ring):
        return np.nan, np.nan, np.nan, np.nan

    filled = binary_fill_holes(ring)
    cavity = filled & (~ring)
    if int(cavity.sum()) >= 4:
        cy, cx = center_of_mass(cavity)
    else:
        cy, cx = center_of_mass(ring)

    if not (np.isfinite(cy) and np.isfinite(cx)):
        return np.nan, np.nan, np.nan, np.nan

    outer_edge = ring & (~binary_erosion(ring, structure=np.ones((3, 3), dtype=bool)))
    inner_edge = cavity & (~binary_erosion(cavity, structure=np.ones((3, 3), dtype=bool)))

    ory, orx = np.nonzero(outer_edge)
    iry, irx = np.nonzero(inner_edge)
    outer_d = np.sqrt((ory - cy) ** 2 + (orx - cx) ** 2)
    inner_d = np.sqrt((iry - cy) ** 2 + (irx - cx) ** 2)

    if outer_d.size == 0:
        ys, xs = np.nonzero(ring)
        outer_d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        if outer_d.size == 0:
            return np.nan, np.nan, np.nan, np.nan

    h, w = ring.shape
    low_res = min(h, w) <= 28
    r_outer_pct = 52 if low_res else 60
    r_outer = float(np.percentile(outer_d, r_outer_pct))
    if inner_d.size > 0:
        # En baja resolución, usar percentil algo menor para evitar sobredimensionar el ROI.
        r_inner = float(np.percentile(inner_d, 62 if low_res else 68))
    else:
        ys, xs = np.nonzero(ring)
        d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        r_inner = float(np.percentile(d, 24 if low_res else 20))

    if not np.isfinite(r_inner):
        r_inner = np.nan
    if not np.isfinite(r_outer):
        r_outer = np.nan
    if np.isfinite(r_inner) and np.isfinite(r_outer) and r_inner >= r_outer:
        r_inner = max(0.0, 0.45 * float(r_outer))
    if np.isfinite(r_inner) and np.isfinite(r_outer):
        r_inner = max(r_inner, 0.33 * float(r_outer))

    # Tope de radio externo para evitar ROIs demasiado grandes en matrices pequeñas.
    max_outer = (0.42 if low_res else 0.48) * float(min(h, w))
    if np.isfinite(r_outer):
        r_outer = min(float(r_outer), float(max_outer))
    if np.isfinite(r_inner) and np.isfinite(r_outer):
        r_inner = min(float(r_inner), 0.84 * float(r_outer))

    return float(cy), float(cx), r_inner, r_outer


def _stabilize_auto_roi_series(
    centers: np.ndarray,
    inner: np.ndarray,
    outer: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(mask, dtype=bool).reshape(mask.shape[0], -1).any(axis=1)
    valid &= np.isfinite(centers[:, 0]) & np.isfinite(centers[:, 1])
    valid &= np.isfinite(outer)

    idx = np.where(valid)[0]
    if idx.size < 3:
        return centers, inner, outer

    centers_out = centers.copy()
    inner_out = inner.copy()
    outer_out = outer.copy()

    cy = gaussian_filter(centers[idx, 0].astype(np.float64), sigma=0.85, mode="nearest")
    cx = gaussian_filter(centers[idx, 1].astype(np.float64), sigma=0.85, mode="nearest")
    ro = gaussian_filter(outer[idx].astype(np.float64), sigma=0.9, mode="nearest")

    rin_src = inner[idx].astype(np.float64)
    in_valid = np.isfinite(rin_src)
    if int(np.sum(in_valid)) >= 2:
        range_idx = np.arange(idx.size, dtype=np.float64)
        rin_interp = np.interp(range_idx, range_idx[in_valid], rin_src[in_valid])
        ri = gaussian_filter(rin_interp, sigma=0.9, mode="nearest")
    else:
        ri = 0.42 * ro

    centers_out[idx, 0] = cy
    centers_out[idx, 1] = cx
    outer_out[idx] = np.clip(ro, 1.0, None)
    inner_out[idx] = np.clip(ri, 0.0, outer_out[idx] - 0.8)
    inner_out[idx] = np.minimum(inner_out[idx], 0.9 * outer_out[idx])

    return centers_out, inner_out, outer_out


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

    low_res = min(H, W) <= 28
    for s in range(n_slices):
        img = mean_img[s].astype(np.float64)
        if img.size == 0 or float(img.max()) <= 0.0:
            continue

        eff_sigma = float(smooth_sigma)
        if low_res:
            # Suavizado levemente mayor en 22x22 para estabilizar bordes pixelados.
            eff_sigma = max(1.15, eff_sigma)
        img_s = gaussian_filter(img, sigma=eff_sigma)

        p99 = float(np.percentile(img_s, 99.0))
        robust_peak = p99 if np.isfinite(p99) and p99 > 0 else float(img_s.max())
        thr_base = float(threshold_frac) * float(robust_peak)
        thr_floor = float(np.percentile(img_s, 70.0 if low_res else 65.0))
        thr = max(thr_base, thr_floor)
        bin_mask = img_s > thr

        if with_cleanup:
            k = 2 if low_res else 3
            st = np.ones((k, k), dtype=bool)
            bin_mask = binary_opening(bin_mask, structure=st)
            bin_mask = binary_closing(bin_mask, structure=st)
            bin_mask = _lv_candidate_component(bin_mask)
            if int(np.count_nonzero(bin_mask)) < 8:
                # Reintento más permisivo cuando la resolución es baja o el umbral quedó alto.
                thr_retry = max(float(np.percentile(img_s, 58.0 if low_res else 55.0)), thr * 0.82)
                bin_retry = img_s > thr_retry
                bin_retry = binary_opening(bin_retry, structure=st)
                bin_retry = binary_closing(bin_retry, structure=st)
                bin_retry = _lv_candidate_component(bin_retry)
                if np.count_nonzero(bin_retry) > np.count_nonzero(bin_mask):
                    bin_mask = bin_retry

        if int(bin_mask.sum()) < 8:
            continue

        mask[s] = bin_mask
        cy, cx, rin, rout = _slice_center_and_radii(bin_mask)
        centers[s] = [cy, cx]
        inner[s] = rin
        outer[s] = rout

    if with_cleanup:
        centers, inner, outer = _stabilize_auto_roi_series(centers, inner, outer, mask)

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
            cy = float(cy)
            cx = float(cx)
            ri = float(r_inner)
            ro = float(r_outer)
            if not np.isfinite(cy) or not np.isfinite(cx) or not np.isfinite(ro) or ro <= 0.0:
                continue
            has_inner = np.isfinite(ri) and ri > 0.0
            if has_inner and ro <= ri:
                continue
            d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
            ring = ((d >= ri) & (d <= ro)) if has_inner else (d <= ro)
            if int(ring.sum()) < 1:
                continue
            mask[s] = ring
            centers[s] = [cy, cx]
            inner[s] = ri if has_inner else np.nan
            outer[s] = ro

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
