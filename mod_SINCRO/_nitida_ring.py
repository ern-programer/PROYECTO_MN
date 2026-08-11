"""Detector de anillo (dona): aísla el miocardio del tórax completo.

En vez del pico de intensidad (que cae en hígado/GI), puntúa cada corte por
'ring-ness' = rim brillante con centro oscuro dentro de una ventana central.
Muestra el mejor rango de cortes recortado al corazón, junto al IRNCRR Xeleris.
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
OUT = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\_xel3_out\nitida_ring.png"
FWHM_TO_SIGMA = 1.0 / 2.354820045


def ring_score_map(vol, r_in=5, r_out=9):
    """Por cada (z,y,x) candidato central: media(anillo) - media(disco). Devuelve
    el mejor (z,y,x) y el mapa por z evaluado en el centro geométrico."""
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    rr = np.sqrt(yy ** 2 + xx ** 2)
    disk = rr <= r_in
    annulus = (rr > r_in) & (rr <= r_out)
    nz, ny, nx = vol.shape
    best = (-1e9, 0, 0, 0)
    # Busca centro en una región central (excluye bordes/periferia).
    for z in range(nz):
        sl = vol[z]
        for cy in range(20, 45, 2):
            for cx in range(20, 45, 2):
                patch = sl[cy - r_out:cy + r_out + 1, cx - r_out:cx + r_out + 1]
                if patch.shape != disk.shape:
                    continue
                s = float(patch[annulus].mean() - patch[disk].mean())
                if s > best[0]:
                    best = (s, z, cy, cx)
    return best


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
        iterations=3, subsets=8, psf=psf,
    )
    vol = gaussian_filter(vol, sigma=(0.6, 1.0, 1.0), mode="constant")

    s, z, cy, cx = ring_score_map(vol)
    print(f"mejor anillo: score={s:.2f} en z={z} centro=({cy},{cx})")

    w = 13
    zs = np.clip(np.arange(z - 5, z + 5), 0, vol.shape[0] - 1)
    sub = vol[:, cy - w:cy + w, cx - w:cx + w]
    vmax = float(np.percentile(sub[zs], 99.5))

    irncrr = pydicom.dcmread(os.path.join(GT_DIR, "STRESS_IRNCRR001_DS.dcm")).pixel_array.astype(np.float64)
    # centro del corazón Xeleris ~ (21,37); recorta igual
    gy, gx = 21, 37
    gsub = irncrr[:, gy - w:gy + w, gx - w:gx + w]
    gz = np.clip(np.arange(2, 12), 0, irncrr.shape[0] - 1)
    gvmax = float(np.percentile(gsub[gz], 99.5))

    fig, axes = plt.subplots(2, 10, figsize=(18, 4.2))
    for j, zz in enumerate(zs):
        axes[0, j].imshow(sub[zz], cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        axes[0, j].set_title(f"z={zz}", fontsize=7)
        axes[0, j].axis("off")
    for j, zz in enumerate(gz):
        axes[1, j].imshow(gsub[zz], cmap="hot", vmin=0, vmax=gvmax, interpolation="nearest")
        axes[1, j].set_title(f"z={zz}", fontsize=7)
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("MÍA (aislada)", fontsize=9)
    fig.text(0.005, 0.72, "MÍA OSEM+RR\n(aislada al VI)", fontsize=9, rotation=90, va="center")
    fig.text(0.005, 0.28, "Xeleris IRNCRR", fontsize=9, rotation=90, va="center")
    fig.suptitle("El miocardio SÍ está: recorte al VI vs Xeleris (mismo tamaño de ventana)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
