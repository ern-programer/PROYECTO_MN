"""Recentrado sobre el MIOCARDIO + control de ZOOM en la reorientación SA.

Problema detectado en contraste_priors.png: el VI queda gigante y descentrado
(llena la esfera de 32x32), mientras Xeleris lo muestra pequeño y centrado con
margen negro. Causa: (1) el centroide del LATIDO se corre al pool sanguíneo;
(2) reslice a paso 1 vóxel sobre esfera r=15 => sobre-ampliación.

Solución:
  1. Recentrar sobre el MIOCARDIO: centro de masa de la cáscara (p80) dentro de
     la esfera del latido — no del pool.
  2. reslice con SPACING (paso de muestreo) para elegir cuánto FOV entra por
     cuadro. spacing>1 aleja (VI más chico + margen), spacing<1 acerca.

Salida:  _xel3_out/centrado_zoom.png
Filas (SA eje corto, 8 cortes centrales):
  1) spacing 1.0   2) spacing 1.4   3) spacing 1.8   4) Xeleris IRNCRR SA
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.raw_projections import load_raw_projections, ungate_projections
from core.raw_reconstruction import reconstruct_projection_volume, ProjectionFilterConfig
from core.collimator_specs import lookup_collimator
from core.resolution_recovery import PsfModel
from core.cardiac_reorientation import _basis_from_long_axis, sa_stack
from _nitida_mrp import mrp_osem_slab, suppress_background
from _nitida_reo import beat_center, long_axis_from_shell

GT_DIR = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
GATED = os.path.join(GT_DIR, "Stress-10sec-1_G_EM_1001_DS.dcm")
OUT = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\_xel3_out\centrado_zoom.png"


def myocardium_center(vol, seed_center, radius=15, pct=80.0):
    """Centro de masa del MIOCARDIO (vóxeles brillantes de la cáscara), no del pool."""
    v = np.asarray(vol, dtype=np.float64)
    cz, cy, cx = seed_center
    zz, yy, xx = np.mgrid[0:v.shape[0], 0:v.shape[1], 0:v.shape[2]]
    within = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
    thr = np.percentile(v[within], pct)
    shell = within & (v >= thr)
    w = v[shell]
    sw = float(w.sum())
    pz, py, px = np.nonzero(shell)
    return (float((pz * w).sum() / sw), float((py * w).sum() / sw), float((px * w).sum() / sw))


def reslice_zoom(volume, center, long_axis, out_size, spacing=1.0, order=1):
    """reslice_from_vector con SPACING: M escalada => controla el zoom.

    spacing>1 => cada paso avanza >1 vóxel => más FOV => VI más chico (margen negro).
    spacing<1 => acercamiento.
    """
    vol = np.asarray(volume, dtype=np.float64)
    u = np.asarray(long_axis, dtype=np.float64)
    if np.linalg.norm(u) <= 0:
        u = np.array([1.0, 0.0, 0.0])
    u, e_j, e_i = _basis_from_long_axis(u)
    M = np.stack([u, e_j, e_i], axis=1) * float(spacing)
    n = int(out_size)
    out_shape = (n, n, n)
    oc = (np.array(out_shape, dtype=np.float64) - 1.0) / 2.0
    c = np.asarray(center, dtype=np.float64)
    offset = c - M @ oc
    return affine_transform(vol, M, offset=offset, output_shape=out_shape,
                            order=order, mode="constant", cval=0.0)


def montage(ax_row, stack, ncols, vmax, label):
    nz = stack.shape[0]
    idx = np.linspace(nz * 0.32, nz * 0.68, ncols).astype(int)
    for j, k in enumerate(idx):
        ax = ax_row[j]
        ax.imshow(stack[k], cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        ax.axis("off")
    ax_row[0].set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", labelpad=40)


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
    seed = beat_center(raw.projections, angles)
    vol = mrp_osem_slab(ung, angles, iterations=8, subsets=8, beta=0.30, psf=psf, prior="median")

    # EJE LARGO: estimar con el centro del LATIDO (estable). NO recalcular tras
    # recentrar (el centro descentrado deforma la PCA -> eje falso casi axial).
    u = long_axis_from_shell(vol, seed, ung_quick, radius=14)
    # CENTRO para crop/display: recentrar sobre el miocardio, pero SOLO en el
    # plano perpendicular al eje (no arrastrar el centro a lo largo del eje).
    myo = np.asarray(myocardium_center(vol, seed, radius=15, pct=80.0), dtype=np.float64)
    s = np.asarray(seed, dtype=np.float64)
    d = myo - s
    center = tuple(s + (d - np.dot(d, u) * u))  # quita la componente a lo largo de u
    print(f"seed(latido)={tuple(round(c,1) for c in seed)}  centro (perp)={tuple(round(float(c),1) for c in center)}")
    print(f"eje largo={np.round(u,3)}")

    disp = suppress_background(vol, frac=0.12)
    # Heart-crop centrado en el miocardio.
    zz, yy, xx = np.mgrid[0:disp.shape[0], 0:disp.shape[1], 0:disp.shape[2]]
    cz, cy, cx = center
    sphere = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= 15 ** 2
    disp = np.where(sphere, disp, 0.0)

    xsa = pydicom.dcmread(os.path.join(GT_DIR, "STRESS_IRNCRR_SA001_DS.dcm")).pixel_array.astype(np.float64)

    spacings = [1.0, 1.4, 1.8]
    ncols = 8
    fig, axes = plt.subplots(4, ncols, figsize=(2.0 * ncols, 9.0))
    for r, sp in enumerate(spacings):
        reo = reslice_zoom(disp, center, u, out_size=32, spacing=sp, order=1)
        sa = sa_stack(reo)
        montage(axes[r], sa, ncols, float(np.percentile(sa, 99.5)), f"spacing {sp:.1f}")
    montage(axes[3], xsa, ncols, float(np.percentile(xsa, 99.5)), "Xeleris SA")
    fig.suptitle("CENTRADO+ZOOM: recentrado en miocardio, spacing 1.0/1.4/1.8 vs Xeleris", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
