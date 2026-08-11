"""¿Está el corazón bien reconstruido pero sin aislar? Recorte central por cortes.

Reconstruye mi volumen y muestra un recorte central (ventana tamaño-corazón)
a lo largo de los cortes axiales, para ver si aparece un anillo miocárdico
compacto en alguna parte (prueba de que la recon está bien y falta AISLAR).
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
OUT = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\_xel3_out\nitida_heart_search.png"
FWHM_TO_SIGMA = 1.0 / 2.354820045


def main():
    raw = load_raw_projections(GATED)
    ung = ungate_projections(raw.projections)
    angles = np.asarray(raw.angles_deg, dtype=np.float64)
    pixel_mm = float(raw.pixel_mm)
    spec = lookup_collimator(raw.manufacturer, raw.collimator_name, raw.collimator_type)
    psf = PsfModel.from_collimator(spec, radius_mm=float(raw.radius_mm or 250.0), pixel_mm=pixel_mm)

    vol = reconstruct_projection_volume(
        ung, angles, method="osem",
        projection_filter=ProjectionFilterConfig("none", 0.5, 1),
        iterations=4, subsets=10, psf=psf,
    )
    sig = (6.0 * FWHM_TO_SIGMA) / pixel_mm
    vol = gaussian_filter(vol, sigma=(0.0, sig, sig), mode="constant")

    # El corazón está cerca del eje de rotación (centro transversal). Busco el
    # corte con más "estructura de anillo" dentro de una ventana central: alto
    # contraste dentro de un disco central (excluye hígado/GI periféricos).
    nz, ny, nx = vol.shape
    cy, cx = ny // 2, nx // 2
    win = 14  # radio de ventana ~ 95 mm (corazón cabe)
    sub = vol[:, cy - win:cy + win, cx - win:cx + win]
    # Puntaje: percentil 99 dentro de la ventana central por corte.
    scores = np.array([np.percentile(sub[z], 99.0) for z in range(nz)])
    zc = int(np.argmax(scores))
    print(f"vol {vol.shape}  centro=({cy},{cx})  mejor z central={zc}")

    # Montaje: 12 cortes alrededor del mejor central, recorte central 28x28.
    zs = np.clip(np.arange(zc - 6, zc + 6), 0, nz - 1)
    vmax = float(np.percentile(sub, 99.7))
    fig, axes = plt.subplots(2, 6, figsize=(13, 4.6))
    for ax, z in zip(axes.ravel(), zs):
        ax.imshow(sub[z], cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"z={z}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Mi recon — recorte CENTRAL 28x28 (¿aparece el anillo del VI?)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
