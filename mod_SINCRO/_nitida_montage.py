"""Montaje de cortes transaxiales: mi recon vs Xeleris IRNC (frame completo).

Quita los confounds de crop y slice-picker: muestra una tira de cortes
consecutivos de cada volumen, frame 64x64 completo, para ubicar y medir el
corazón real en cada uno.
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
OUT = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\_xel3_out\nitida_montage.png"
FWHM_TO_SIGMA = 1.0 / 2.354820045


def montage_row(fig, gs_row, vol, n, label, ncols):
    nz = vol.shape[0]
    idx = np.linspace(0, nz - 1, ncols).astype(int)
    vmax = float(np.percentile(vol, 99.5))
    for j, z in enumerate(idx):
        ax = fig.add_subplot(gs_row[0, j])
        ax.imshow(vol[z], cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"z={z}", fontsize=7)
        ax.axis("off")
        if j == 0:
            ax.set_ylabel(label, fontsize=9)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    raw = load_raw_projections(GATED)
    ung = ungate_projections(raw.projections)
    angles = np.asarray(raw.angles_deg, dtype=np.float64)
    pixel_mm = float(raw.pixel_mm)
    spec = lookup_collimator(raw.manufacturer, raw.collimator_name, raw.collimator_type)
    psf = PsfModel.from_collimator(spec, radius_mm=float(raw.radius_mm or 250.0), pixel_mm=pixel_mm)

    mine = reconstruct_projection_volume(
        ung, angles, method="osem",
        projection_filter=ProjectionFilterConfig("none", 0.5, 1),
        iterations=4, subsets=10, psf=psf,
    )
    sig = (6.0 * FWHM_TO_SIGMA) / pixel_mm
    mine = gaussian_filter(mine, sigma=(0.0, sig, sig), mode="constant")

    irnc = pydicom.dcmread(os.path.join(GT_DIR, "STRESS_IRNC001_DS.dcm")).pixel_array.astype(np.float64)
    irncrr = pydicom.dcmread(os.path.join(GT_DIR, "STRESS_IRNCRR001_DS.dcm")).pixel_array.astype(np.float64)

    print("mine:", mine.shape, "irnc:", irnc.shape, "irncrr:", irncrr.shape)
    # Localiza el máximo de cada volumen (dónde está el corazón/hígado).
    for nm, v in (("mine", mine), ("irnc", irnc), ("irncrr", irncrr)):
        z, y, x = np.unravel_index(int(np.argmax(v)), v.shape)
        print(f"  {nm}: argmax en z={z} y={y} x={x}  (nz={v.shape[0]})")

    ncols = 10
    fig = plt.figure(figsize=(1.5 * ncols, 5.2))
    import matplotlib.gridspec as gridspec
    outer = gridspec.GridSpec(3, 1, hspace=0.35)
    for i, (v, lab) in enumerate(((mine, "MÍA OSEM 4x10 noRR? RR +6mm"),
                                  (irnc, "Xeleris IRNC"),
                                  (irncrr, "Xeleris IRNCRR"))):
        sub = gridspec.GridSpecFromSubplotSpec(1, ncols, subplot_spec=outer[i], wspace=0.05)
        montage_row(fig, sub, v, v.shape[0], lab, ncols)
        fig.text(0.01, 0.83 - i * 0.32, lab, fontsize=9, rotation=90, va="center")

    fig.suptitle("Montaje transaxial: mi recon vs Xeleris (frame 64x64 completo)", fontsize=11)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
