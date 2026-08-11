"""Test definitivo del 2x de tamaño: round-trip con MI proyector.

Tomo el volumen IRNC de Xeleris (corazón compacto 14px), lo forward-projecto
con MI proyector y lo reconstruyo con MI OSEM. Si vuelve 14px => mi proyector
conserva escala y el 2x viene de las proyecciones reales (corrección geométrica
que Xeleris aplica). Si vuelve 28px => mi proyector magnifica (bug).
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
from core.raw_reconstruction import (
    _forward_project_slice,
    reconstruct_projection_volume,
)

BASE = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_xel3_out")


def extent_px(img2d, frac=0.4):
    m = float(img2d.max())
    if m <= 0:
        return 0
    ys, xs = np.nonzero(img2d >= frac * m)
    return int(xs.max() - xs.min() + 1) if xs.size else 0


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    irnc = np.asarray(pydicom.dcmread(os.path.join(BASE, "STRESS_IRNC001_DS.dcm")).pixel_array, dtype=np.float64)
    # slice más brillante de Xeleris
    zx = int(np.argmax(irnc.sum(axis=(1, 2))))
    xsl = irnc[zx]
    wx = extent_px(xsl)
    print(f"Xeleris IRNC z={zx}: heart width ~{wx}px")

    angles60 = (225.0 - 3.0 * np.arange(60)) % 360.0

    # forward-project el volumen 3D de Xeleris con mi proyector, luego OSEM.
    # armo proyecciones (angles, H, W): para cada fila axial z, proyecto ese corte.
    H, W = irnc.shape[1], irnc.shape[2]
    sino_vol = np.zeros((len(angles60), irnc.shape[0], W), dtype=np.float64)
    for z in range(irnc.shape[0]):
        s = _forward_project_slice(irnc[z], angles60, detector_size=W)  # (W, angles)
        sino_vol[:, z, :] = s.T
    print("proyecciones sintéticas:", sino_vol.shape)

    rt = reconstruct_projection_volume(sino_vol, angles60, method="osem",
                                       iterations=6, subsets=10)
    zr = int(np.argmax(rt.sum(axis=(1, 2))))
    wr = extent_px(rt[zr])
    print(f"Round-trip mi OSEM z={zr}: heart width ~{wr}px")
    print(f"\n=> ratio roundtrip/xeleris = {wr/max(wx,1):.2f}x")
    if wr <= wx * 1.3:
        print("   Mi proyector CONSERVA escala. El 2x viene de las proyecciones reales.")
    else:
        print("   Mi proyector MAGNIFICA. Bug de escala en el proyector.")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(xsl, cmap="hot", vmax=np.percentile(xsl, 99.5)); axes[0].set_title(f"Xeleris IRNC\n{wx}px", fontsize=8)
    axes[1].imshow(sino_vol[:, zx, :], cmap="hot"); axes[1].set_title("sinograma sintético\n(mi forward)", fontsize=8)
    axes[2].imshow(rt[zr], cmap="hot", vmax=np.percentile(rt[zr], 99.5)); axes[2].set_title(f"round-trip mi OSEM\n{wr}px", fontsize=8)
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "ROUNDTRIP.png")
    fig.savefig(out, dpi=100); plt.close(fig)
    print("Guardado:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
