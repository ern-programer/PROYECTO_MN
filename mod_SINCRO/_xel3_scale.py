"""Mide escala/tamaño del corazón para explicar por qué mi recon sale ~2x más
grande que Xeleris. Compara pixel spacing y extensión del objeto en:
  - proyección cruda
  - mi FBP
  - Xeleris FBP
Además muestra una proyección para inspección visual.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.raw_projections import load_raw_projections
from core.raw_reconstruction import ProjectionFilterConfig, reconstruct_fbp_volume

BASE = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
SUMMED = os.path.join(BASE, "Stress-10sec-1_T_EM001_DS.dcm")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_xel3_out")


def extent_px(img2d, frac=0.3):
    m = float(img2d.max())
    if m <= 0:
        return 0, 0, 0.0
    mask = img2d >= frac * m
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0, 0, 0.0
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), float(mask.sum())


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ds = pydicom.dcmread(SUMMED)
    print("Summed emission PixelSpacing:", getattr(ds, "PixelSpacing", None),
          "Rows/Cols:", ds.Rows, ds.Columns)

    raw = load_raw_projections(SUMMED)
    proj = raw.projections
    if proj.ndim == 4:
        proj = proj.sum(axis=0)
    print("proj:", proj.shape, "pixel_mm:", getattr(raw, "pixel_mm", "?"))

    # Proyección donde el corazón se ve de perfil (elijo la de más cuentas).
    fr_energy = proj.sum(axis=(1, 2))
    fr = int(np.argmax(fr_energy))
    prj = proj[fr]
    w, h, area = extent_px(prj)
    print(f"Proyección frame {fr}: extensión corazón ~ {w}x{h} px (área>{0.3:.0%}max={area:.0f})")

    # Mi FBP (60v/180)
    angles60 = (225.0 - 3.0 * np.arange(60)) % 360.0
    mine = reconstruct_fbp_volume(proj[:60], angles60,
                                  projection_filter=ProjectionFilterConfig("butterworth", 0.40, 10),
                                  fbp_filter_name="ramp")
    zc = int(np.argmax(mine.sum(axis=(1, 2))))
    mw, mh, ma = extent_px(mine[zc])
    print(f"MI FBP corte z={zc}: extensión ~ {mw}x{mh} px (área={ma:.0f})")

    # Xeleris FBP
    xf = np.asarray(pydicom.dcmread(os.path.join(BASE, "STRESS_FBP001_DS.dcm")).pixel_array, dtype=np.float64)
    xds = pydicom.dcmread(os.path.join(BASE, "STRESS_FBP001_DS.dcm"))
    print("Xeleris FBP PixelSpacing:", getattr(xds, "PixelSpacing", None))
    zx = int(np.argmax(xf.sum(axis=(1, 2))))
    xw, xh, xa = extent_px(xf[zx])
    print(f"Xeleris FBP corte z={zx}: extensión ~ {xw}x{xh} px (área={xa:.0f})")

    print(f"\nRatio tamaño (mi/xeleris): {mw/max(xw,1):.2f}x ancho, {mh/max(xh,1):.2f}x alto")

    # Figura: proyección + perfiles
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].imshow(prj, cmap="hot"); axes[0].set_title(f"Proyección frame {fr}\n{w}x{h}px", fontsize=8)
    axes[1].imshow(mine[zc], cmap="hot", vmax=np.percentile(mine[zc], 99.5))
    axes[1].set_title(f"MI FBP z={zc}\n{mw}x{mh}px", fontsize=8)
    axes[2].imshow(xf[zx], cmap="hot", vmax=np.percentile(xf[zx], 99.5))
    axes[2].set_title(f"Xeleris FBP z={zx}\n{xw}x{xh}px", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "SCALE_cmp.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("Guardado:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
