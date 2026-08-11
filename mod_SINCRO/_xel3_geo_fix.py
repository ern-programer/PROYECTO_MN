"""Prueba de corrección de geometría angular (doble cabezal / 60 vistas 180°).

El sumado tiene 120 frames = 60 vistas angulares (225->48, arc 180) medidas 2x.
Mi loader las trataba como 120 vistas sobre 360° con ángulos inventados en la
2da mitad => estrías + objeto agrandado.

Comparamos, contra el FBP de Xeleris (STRESS_FBP001), tres estrategias:
  A) 120 vistas 360° (loader actual, el que estría).
  B) solo primeras 60 vistas (225->48, arc 180).
  C) sumar pares (frame i + i+60) como MISMA vista => 60 proyecciones 180°.
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

pf = ProjectionFilterConfig("butterworth", 0.40, 10)


def recon(proj, angles):
    return reconstruct_fbp_volume(proj, angles, projection_filter=pf, fbp_filter_name="ramp")


def grid_row(axes_row, vol, label, idxs):
    for c, z in enumerate(idxs):
        ax = axes_row[c]
        ax.axis("off")
        sl = vol[z]
        vmax = float(np.percentile(sl, 99.5)) or float(vol.max())
        ax.imshow(sl, cmap="hot", vmin=0, vmax=max(vmax, 1e-6))
        ax.set_title(f"z={z}", fontsize=6)
    axes_row[0].set_ylabel(label, fontsize=7)
    axes_row[0].axis("on"); axes_row[0].set_xticks([]); axes_row[0].set_yticks([])


def central(vol, n=8):
    e = vol.sum(axis=(1, 2))
    zc = int(np.argmax(e))
    return np.clip(np.linspace(zc - 8, zc + 8, n).astype(int), 0, vol.shape[0] - 1)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    raw = load_raw_projections(SUMMED)
    proj = raw.projections
    if proj.ndim == 4:
        proj = proj.sum(axis=0)  # (120,64,64)
    print("proj:", proj.shape)

    # Ángulos reales de la 1ra rotación: 225 - 3*i, i=0..59 (arc 180, CW).
    angles60 = (225.0 - 3.0 * np.arange(60)) % 360.0

    # A) actual 120/360
    volA = recon(proj, raw.angles_deg)
    # B) solo primeras 60 vistas
    volB = recon(proj[:60], angles60)
    # C) sumar pares i + i+60 (misma vista angular) => 60 proyecciones
    proj_pairs = proj[:60] + proj[60:120]
    volC = recon(proj_pairs, angles60)

    # Xeleris FBP ground truth
    xf = np.asarray(pydicom.dcmread(os.path.join(BASE, "STRESS_FBP001_DS.dcm")).pixel_array, dtype=np.float64)

    fig, axes = plt.subplots(4, 8, figsize=(8 * 1.7, 4 * 1.7))
    grid_row(axes[0], xf, "Xeleris FBP (GT)", np.linspace(2, xf.shape[0]-3, 8).astype(int))
    grid_row(axes[1], volA, "A) 120v/360 (actual)", central(volA))
    grid_row(axes[2], volB, "B) 60v/180 (1er cabezal)", central(volB))
    grid_row(axes[3], volC, "C) pares sumados 60v/180", central(volC))
    fig.suptitle("Corrección geometría angular vs Xeleris FBP", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "GEO_fix_cmp.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("Guardado:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
