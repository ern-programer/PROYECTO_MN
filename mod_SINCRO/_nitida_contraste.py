"""Barrido de CONTRASTE: prior median vs TV vs bilateral en la cadena reo.

Objetivo: ver cuál regularización VACÍA LA CAVIDAD sin meter ruido ni lavar la
pared. Misma cadena que _nitida_reo (latido→centro, PCA-cáscara→eje largo,
heart-crop, reslice a eje corto), variando solo el prior del MRP-OSEM.

Salida:  _xel3_out/contraste_priors.png
Filas (montaje SA eje corto, 8 cortes centrales):
  1) MRP median   2) MRP TV   3) MRP bilateral   4) Xeleris IRNCRR SA
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
from core.cardiac_reorientation import reslice_from_vector, sa_stack
from _nitida_mrp import mrp_osem_slab, suppress_background
from _nitida_reo import beat_center, long_axis_from_shell

GT_DIR = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
GATED = os.path.join(GT_DIR, "Stress-10sec-1_G_EM_1001_DS.dcm")
OUT = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\_xel3_out\contraste_priors.png"


def montage(ax_row, stack, ncols, vmax, label):
    nz = stack.shape[0]
    idx = np.linspace(nz * 0.32, nz * 0.68, ncols).astype(int)
    for j, k in enumerate(idx):
        ax = ax_row[j]
        ax.imshow(stack[k], cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        ax.axis("off")
    ax_row[0].set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", labelpad=40)


def reo_from_recon(vol_mrp, center, u, sphere_r=15):
    disp = suppress_background(vol_mrp, frac=0.12)
    zz, yy, xx = np.mgrid[0:disp.shape[0], 0:disp.shape[1], 0:disp.shape[2]]
    cz, cy, cx = center
    sphere = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= sphere_r ** 2
    disp = np.where(sphere, disp, 0.0)
    reo = reslice_from_vector(disp, center, u, out_size=32, order=1)
    return sa_stack(reo)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    raw = load_raw_projections(GATED)
    ung = ungate_projections(raw.projections)
    angles = np.asarray(raw.angles_deg, dtype=np.float64)
    pixel_mm = float(raw.pixel_mm)
    spec = lookup_collimator(raw.manufacturer, raw.collimator_name, raw.collimator_type)
    psf = PsfModel.from_collimator(spec, radius_mm=float(raw.radius_mm or 250.0), pixel_mm=pixel_mm)

    ung_quick = reconstruct_projection_volume(
        ung, angles, method="fbp",
        projection_filter=ProjectionFilterConfig("butterworth", 0.5, 5), fbp_filter_name="ramp",
    )
    center = beat_center(raw.projections, angles)

    priors = [("median", 0.30), ("tv", 0.30), ("bilateral", 0.30)]
    stacks = []
    axis_u = None
    for name, beta in priors:
        vol = mrp_osem_slab(ung, angles, iterations=6, subsets=8, beta=beta, psf=psf, prior=name)
        if axis_u is None:
            axis_u = long_axis_from_shell(vol, center, ung_quick, radius=14)
            print(f"centro VI={tuple(round(c,1) for c in center)}  eje largo={np.round(axis_u,3)}")
        stacks.append((name, reo_from_recon(vol, center, axis_u)))
        print(f"prior={name:9s} listo")

    xsa = pydicom.dcmread(os.path.join(GT_DIR, "STRESS_IRNCRR_SA001_DS.dcm")).pixel_array.astype(np.float64)

    ncols = 8
    fig, axes = plt.subplots(4, ncols, figsize=(2.0 * ncols, 9.0))
    for r, (name, sa) in enumerate(stacks):
        montage(axes[r], sa, ncols, float(np.percentile(sa, 99.5)), f"MRP {name}")
    montage(axes[3], xsa, ncols, float(np.percentile(xsa, 99.5)), "Xeleris SA")
    fig.suptitle("CONTRASTE: prior median vs TV vs bilateral (¿vacía la cavidad?)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
