"""Mirada directa: NÍTIDA actual vs recon más convergida vs ground truth Xeleris.

Reconstruye el volumen UNGATED (alto conteo) del estudio Xeleris 3 con varias
recetas y renderiza un corte transaxial cardíaco de cada una, junto al IRNC /
IRNCRR de Xeleris, para juzgar con el ojo si compactamos la pared.

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _nitida_look.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.raw_projections import load_raw_projections, ungate_projections
from core.raw_reconstruction import reconstruct_projection_volume, ProjectionFilterConfig
from core.collimator_specs import lookup_collimator
from core.resolution_recovery import PsfModel

GT_DIR = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
GATED = os.path.join(GT_DIR, "Stress-10sec-1_G_EM_1001_DS.dcm")
OUT = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\_xel3_out\nitida_look.png"

FWHM_TO_SIGMA = 1.0 / 2.354820045


def cardiac_slice(vol: np.ndarray, disk_frac: float = 0.33) -> int:
    nz, ny, nx = vol.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r = min(ny, nx) * disk_frac
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    scores = [float(np.percentile(vol[z][disk], 99.0)) if disk.any() else 0.0 for z in range(nz)]
    return int(np.argmax(scores))


def crop_center(img: np.ndarray, half: int = 22) -> np.ndarray:
    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2
    y0, x0 = max(0, cy - half), max(0, cx - half)
    return img[y0:cy + half, x0:cx + half]


def recon(ung, angles, *, iters, subsets, psf, post_fwhm_mm, pixel_mm):
    vol = reconstruct_projection_volume(
        ung, angles, method="osem",
        projection_filter=ProjectionFilterConfig("none", 0.5, 1),
        iterations=iters, subsets=subsets, psf=psf,
    )
    if post_fwhm_mm > 0:
        sig = (post_fwhm_mm * FWHM_TO_SIGMA) / pixel_mm
        vol = gaussian_filter(vol, sigma=(0.0, sig, sig), mode="constant")
    return vol


def load_gt_slice(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    z = cardiac_slice(arr)
    return arr[z]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    raw = load_raw_projections(GATED)
    ung = ungate_projections(raw.projections)  # (ang,H,W) alto conteo
    angles = np.asarray(raw.angles_deg, dtype=np.float64)
    pixel_mm = float(raw.pixel_mm)
    spec = lookup_collimator(raw.manufacturer, raw.collimator_name, raw.collimator_type)
    radius_mm = float(raw.radius_mm) if raw.radius_mm else 250.0
    psf = PsfModel.from_collimator(spec, radius_mm=radius_mm, pixel_mm=pixel_mm)
    print(f"col={spec.manufacturer} {spec.name} L={spec.hole_length_mm} radius={radius_mm:.1f} pixel={pixel_mm:.3f}")

    recipes = [
        ("NÍTIDA actual\nOSEM 5x4 +RR +8mm", dict(iters=5, subsets=4, psf=psf, post_fwhm_mm=8.0)),
        ("OSEM 4x10 +RR +6mm", dict(iters=4, subsets=10, psf=psf, post_fwhm_mm=6.0)),
        ("OSEM 4x10 sin RR +6mm", dict(iters=4, subsets=10, psf=None, post_fwhm_mm=6.0)),
        ("OSEM 8x8 +RR +5mm", dict(iters=8, subsets=8, psf=psf, post_fwhm_mm=5.0)),
    ]
    imgs, titles = [], []
    for title, kw in recipes:
        vol = recon(ung, angles, pixel_mm=pixel_mm, **kw)
        z = cardiac_slice(vol)
        imgs.append(crop_center(vol[z]))
        titles.append(title)
        print(f"  {title!r}: slice={z} max={vol.max():.1f}")

    # Ground truth Xeleris (transaxial): IRNC (iter, sin RR) e IRNCRR (iter+RR).
    for name, label in (("STRESS_IRNC001_DS.dcm", "Xeleris IRNC"),
                        ("STRESS_IRNCRR001_DS.dcm", "Xeleris IRNCRR")):
        p = os.path.join(GT_DIR, name)
        if os.path.isfile(p):
            imgs.append(crop_center(load_gt_slice(p)))
            titles.append(label)

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4))
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im, cmap="hot", interpolation="nearest",
                  vmin=0, vmax=float(np.percentile(im, 99.5)))
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.suptitle("NÍTIDA vs recon convergida vs Xeleris — corte transaxial cardíaco", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
