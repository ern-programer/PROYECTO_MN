"""Compara los transverse reconstruidos por Xeleris (FBP / IRNC / IRNCRR)
contra mi reconstrucción FBP, para ver el gap real de calidad.

Xeleris exporta cada transverse como 19 cortes 64x64 (pixel 6.8mm), centrados
en el corazón. Mi recon es 64^3 full-FOV. Comparamos la apariencia in-plane.
"""
from __future__ import annotations

import glob
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
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_xel3_out")


def load_vol(name):
    ds = pydicom.dcmread(os.path.join(BASE, name))
    return np.asarray(ds.pixel_array, dtype=np.float64)


def row(ax_list, vol, label, idxs, cmap="hot"):
    for c, z in enumerate(idxs):
        ax = ax_list[c]
        ax.axis("off")
        sl = vol[z]
        vmax = float(np.percentile(sl, 99.5)) or float(vol.max())
        ax.imshow(sl, cmap=cmap, vmin=0, vmax=max(vmax, 1e-6))
        ax.set_title(f"z={z}", fontsize=6)
    ax_list[0].set_ylabel(label, fontsize=8)
    ax_list[0].axis("on")
    ax_list[0].set_xticks([]); ax_list[0].set_yticks([])


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    fbp = load_vol("STRESS_FBP001_DS.dcm")
    irnc = load_vol("STRESS_IRNC001_DS.dcm")
    rr = load_vol("STRESS_IRNCRR001_DS.dcm")
    print(f"Xeleris FBP={fbp.shape} IRNC={irnc.shape} IRNCRR={rr.shape}")

    # Mi FBP del sumado (120 vistas), Butterworth 0.4/10 + ramp.
    raw = load_raw_projections(os.path.join(BASE, "Stress-10sec-1_T_EM001_DS.dcm"))
    proj = raw.projections
    if proj.ndim == 4:
        proj = proj.sum(axis=0)
    mine = reconstruct_fbp_volume(proj, raw.angles_deg,
                                  projection_filter=ProjectionFilterConfig("butterworth", 0.40, 10),
                                  fbp_filter_name="ramp")
    print(f"Mi FBP={mine.shape}")

    # Xeleris: 19 cortes centrados en corazón. Tomo 8 centrales.
    zx = np.linspace(2, fbp.shape[0] - 3, 8).astype(int)
    # Mío: 64 cortes; ubico el corazón por máxima energía y tomo 8 alrededor.
    energy = mine.sum(axis=(1, 2))
    zc = int(np.argmax(energy))
    zm = np.clip(np.linspace(zc - 10, zc + 10, 8).astype(int), 0, mine.shape[0] - 1)

    rows = 4
    cols = 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.7))
    row(axes[0], fbp, "Xeleris FBP", zx)
    row(axes[1], irnc, "Xeleris IRNC (OSEM)", zx)
    row(axes[2], rr, "Xeleris IRNCRR (OSEM+RR)=EVOLUTION", zx)
    row(axes[3], mine, "MI FBP Bw0.4/10", zm)
    fig.suptitle("Transverse: Xeleris vs mi reconstrucción (ventaneo por corte)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "GT_transverse_cmp.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("Guardado:", out)

    # También: SA limpio de Xeleris (FBP vs IRNCRR) como referencia de orientación.
    fbp_sa = load_vol("STRESS_FBP_SA001_DS.dcm")
    rr_sa = load_vol("STRESS_IRNCRR_SA001_DS.dcm")
    zs = np.linspace(2, fbp_sa.shape[0] - 3, 8).astype(int)
    fig2, axes2 = plt.subplots(2, 8, figsize=(8 * 1.7, 2 * 1.7))
    row(axes2[0], fbp_sa, "Xeleris SA FBP", zs)
    row(axes2[1], rr_sa, "Xeleris SA IRNCRR", zs)
    fig2.suptitle("Short Axis reconstruido por Xeleris (target de orientación)", fontsize=11)
    fig2.tight_layout()
    out2 = os.path.join(OUTDIR, "GT_sa_xeleris.png")
    fig2.savefig(out2, dpi=100)
    plt.close(fig2)
    print("Guardado:", out2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
