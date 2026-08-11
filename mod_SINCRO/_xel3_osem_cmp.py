"""Compara OSEM (sin y con Resolution Recovery/PSF) contra las reconstrucciones
iterativas de Xeleris: IRNC (OSEM sin corrección) e IRNCRR (OSEM + RR = Evolution).

Hipótesis: el FBP casero estría y agranda; OSEM (ya en SINCRO) da corazón
compacto = IRNC, y OSEM+PSF = IRNCRR/Evolution. Geometría real: 60 vistas/180°.
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
from core.collimator_specs import lookup_collimator
from core.raw_projections import load_raw_projections
from core.resolution_recovery import PsfModel
from core.raw_reconstruction import reconstruct_projection_volume

BASE = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
SUMMED = os.path.join(BASE, "Stress-10sec-1_T_EM001_DS.dcm")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_xel3_out")


def load_vol(name):
    return np.asarray(pydicom.dcmread(os.path.join(BASE, name)).pixel_array, dtype=np.float64)


def row(axes_row, vol, label, idxs):
    for c, z in enumerate(idxs):
        ax = axes_row[c]; ax.axis("off")
        sl = vol[z]
        vmax = float(np.percentile(sl, 99.5)) or float(vol.max())
        ax.imshow(sl, cmap="hot", vmin=0, vmax=max(vmax, 1e-6))
        ax.set_title(f"z={z}", fontsize=6)
    axes_row[0].set_ylabel(label, fontsize=7)
    axes_row[0].axis("on"); axes_row[0].set_xticks([]); axes_row[0].set_yticks([])


def central(vol, n=8):
    zc = int(np.argmax(vol.sum(axis=(1, 2))))
    return np.clip(np.linspace(zc - 8, zc + 8, n).astype(int), 0, vol.shape[0] - 1)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    raw = load_raw_projections(SUMMED)
    proj = raw.projections
    if proj.ndim == 4:
        proj = proj.sum(axis=0)
    proj60 = proj[:60]
    angles60 = (225.0 - 3.0 * np.arange(60)) % 360.0
    pixel_mm = float(getattr(raw, "pixel_mm", 6.797) or 6.797)

    spec = lookup_collimator("GE", "LEHR")
    radius_mm = 250.0  # órbita cardíaca típica; ajustable si el DICOM la trae
    psf = PsfModel.from_collimator(spec, radius_mm=radius_mm, pixel_mm=pixel_mm)

    print("OSEM+PSF 6x10...")
    osem6 = reconstruct_projection_volume(proj60, angles60, method="osem",
                                          iterations=6, subsets=10, psf=psf)
    print("OSEM+PSF 12x10...")
    osem12 = reconstruct_projection_volume(proj60, angles60, method="osem",
                                           iterations=12, subsets=10, psf=psf)

    irnc = load_vol("STRESS_IRNC001_DS.dcm")
    rr = load_vol("STRESS_IRNCRR001_DS.dcm")

    fig, axes = plt.subplots(4, 8, figsize=(8 * 1.7, 4 * 1.7))
    row(axes[0], rr, "Xeleris IRNCRR (GT)", np.linspace(2, rr.shape[0]-3, 8).astype(int))
    row(axes[1], osem6, "MI OSEM+PSF 6x10", central(osem6))
    row(axes[2], osem12, "MI OSEM+PSF 12x10", central(osem12))
    row(axes[3], irnc, "Xeleris IRNC (GT)", np.linspace(2, irnc.shape[0]-3, 8).astype(int))
    fig.suptitle("OSEM+PSF más iteraciones vs Xeleris IRNCRR (ventaneo por corte)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "OSEM_cmp.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("Guardado:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
