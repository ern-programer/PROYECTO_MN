"""Montaje completo 8x8: todos los cortes axiales de mi recon (FBP Butterworth).

Objetivo: localizar visualmente el corazón dentro del volumen de tórax completo
y confirmar que se reconstruye como un anillo válido (recon OK; falta aislar).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.raw_projections import load_raw_projections, ungate_projections
from core.raw_reconstruction import reconstruct_projection_volume, ProjectionFilterConfig

GT_DIR = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
GATED = os.path.join(GT_DIR, "Stress-10sec-1_G_EM_1001_DS.dcm")
OUT = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\_xel3_out\nitida_full64.png"


def main():
    raw = load_raw_projections(GATED)
    ung = ungate_projections(raw.projections)
    angles = np.asarray(raw.angles_deg, dtype=np.float64)
    vol = reconstruct_projection_volume(
        ung, angles, method="fbp",
        projection_filter=ProjectionFilterConfig("butterworth", 0.52, 5),
        fbp_filter_name="ramp",
    )
    nz = vol.shape[0]
    vmax = float(np.percentile(vol, 99.6))
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    for z, ax in enumerate(axes.ravel()):
        if z < nz:
            ax.imshow(vol[z], cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"z={z}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Mi recon FBP-Butterworth — 64 cortes axiales (tórax completo)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=95)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
