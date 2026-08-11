"""Serie nítida v2: reorientación a eje corto guiada por el LATIDO.

Cadena:
  1. Localiza el VI por latido (varianza entre gates reconstruida) → centro Y
     eje largo (PCA ponderada del mapa de latido). Robusto: solo el corazón late.
  2. Reconstruye el gate ED con MRP-OSEM (feta implícita por el reslice).
  3. Reslice oblicuo al eje corto (reslice_from_vector) → cortes SA = DONAS.
  4. Supresión de fondo + montaje SA vs Xeleris STRESS_IRNCRR_SA001.
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

GT_DIR = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
GATED = os.path.join(GT_DIR, "Stress-10sec-1_G_EM_1001_DS.dcm")
OUT = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\_xel3_out\nitida_reo.png"


def beat_center(gated_proj, angles):
    """Centro (z,y,x) del VI desde el mapa de latido reconstruido (robusto)."""
    g = np.clip(np.asarray(gated_proj, dtype=np.float64), 0.0, None)
    var_sino = g.var(axis=0)  # (ang,H,W)
    heat = reconstruct_projection_volume(
        var_sino, angles, method="fbp",
        projection_filter=ProjectionFilterConfig("butterworth", 0.45, 6),
        fbp_filter_name="ramp",
    )
    heat = gaussian_filter(np.clip(heat, 0.0, None), sigma=1.2, mode="constant")
    thr = np.percentile(heat, 96.0)
    mask = heat >= thr
    zz, yy, xx = np.nonzero(mask)
    w = heat[mask]
    sw = float(w.sum())
    return (float((zz * w).sum() / sw), float((yy * w).sum() / sw), float((xx * w).sum() / sw))


def long_axis_from_shell(vol, center, ungated_vol, radius=14):
    """Eje largo del VI = eje MAYOR del elipsoide prolato de la cáscara miocárdica.

    Toma los vóxeles brillantes (miocardio) dentro de una esfera alrededor del
    centro y hace PCA: el miocardio es un cascarón prolato, su eigenvector de
    mayor autovalor apunta base→ápex. Mucho más fiable que la PCA del latido.
    """
    v = np.asarray(vol, dtype=np.float64)
    cz, cy, cx = center
    zz, yy, xx = np.mgrid[0:v.shape[0], 0:v.shape[1], 0:v.shape[2]]
    within = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
    local = np.where(within, v, 0.0)
    thr = np.percentile(local[within], 80.0)
    shell = within & (v >= thr)
    pz, py, px = np.nonzero(shell)
    w = v[shell]
    sw = float(w.sum())
    mz = float((pz * w).sum() / sw); my = float((py * w).sum() / sw); mx = float((px * w).sum() / sw)
    pts = np.stack([pz - mz, py - my, px - mx], axis=1)
    cov = (pts.T * w) @ pts / sw
    _, evecs = np.linalg.eigh(cov)
    u = np.asarray(evecs[:, -1], dtype=np.float64)
    u /= (np.linalg.norm(u) or 1.0)
    # Signo base→ápex: ápex apunta hacia afuera del centro del cuerpo.
    uv = np.asarray(ungated_vol, dtype=np.float64)
    tot = float(uv.sum())
    zc = float((uv.sum(axis=(1, 2)) * np.arange(uv.shape[0])).sum() / tot)
    yc = float((uv.sum(axis=(0, 2)) * np.arange(uv.shape[1])).sum() / tot)
    xc = float((uv.sum(axis=(0, 1)) * np.arange(uv.shape[2])).sum() / tot)
    if float(np.dot(u, [cz - zc, cy - yc, cx - xc])) < 0:
        u = -u
    return u


def montage(ax_row, stack, ncols, vmax, label):
    nz = stack.shape[0]
    idx = np.linspace(nz * 0.32, nz * 0.68, ncols).astype(int)
    for j, k in enumerate(idx):
        ax = ax_row[j]
        ax.imshow(stack[k], cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        ax.axis("off")
    ax_row[0].set_ylabel(label, fontsize=9)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    raw = load_raw_projections(GATED)
    ung = ungate_projections(raw.projections)
    angles = np.asarray(raw.angles_deg, dtype=np.float64)
    pixel_mm = float(raw.pixel_mm)
    spec = lookup_collimator(raw.manufacturer, raw.collimator_name, raw.collimator_type)
    psf = PsfModel.from_collimator(spec, radius_mm=float(raw.radius_mm or 250.0), pixel_mm=pixel_mm)

    ung_vol_quick = reconstruct_projection_volume(
        ung, angles, method="fbp",
        projection_filter=ProjectionFilterConfig("butterworth", 0.5, 5), fbp_filter_name="ramp",
    )
    center = beat_center(raw.projections, angles)

    # Recon MRP del UNGATED (perfusión = suma, como Xeleris IRNCRR): más cuentas, más limpio.
    ung_mrp = mrp_osem_slab(ung, angles, iterations=8, subsets=8, beta=0.15, psf=psf)
    u = long_axis_from_shell(ung_mrp, center, ung_vol_quick, radius=14)
    print(f"centro VI={tuple(round(c,1) for c in center)}  eje largo (z,y,x)={np.round(u,3)}")

    disp = suppress_background(ung_mrp, frac=0.12)
    # Heart-crop esférico (imita el recorte de corazón de Xeleris): fuera de la
    # esfera alrededor del centro por latido -> negro. Aísla el VI del hígado/torso.
    zz, yy, xx = np.mgrid[0:disp.shape[0], 0:disp.shape[1], 0:disp.shape[2]]
    cz, cy, cx = center
    sphere = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= 15 ** 2
    disp = np.where(sphere, disp, 0.0)
    reo = reslice_from_vector(disp, center, u, out_size=32, order=1)
    sa = sa_stack(reo)
    print("reo:", reo.shape, "sa:", sa.shape)

    # Xeleris SA ground truth
    xsa = pydicom.dcmread(os.path.join(GT_DIR, "STRESS_IRNCRR_SA001_DS.dcm")).pixel_array.astype(np.float64)

    ncols = 8
    fig, axes = plt.subplots(2, ncols, figsize=(2.0 * ncols, 5.0))
    montage(axes[0], sa, ncols, float(np.percentile(sa, 99.5)), "MÍA SA (nítida)")
    montage(axes[1], xsa, ncols, float(np.percentile(xsa, 99.5)), "Xeleris SA")
    fig.suptitle("Eje corto: mi recon reorientada por latido vs Xeleris IRNCRR SA", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
