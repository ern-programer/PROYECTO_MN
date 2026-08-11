"""Barrido de orientación del VI sobre el Xeleris 3 reconstruido en SINCRO.

Objetivo: separar el problema de RECONSTRUCCIÓN del de ORIENTACIÓN.
El transaxial por-corte ya muestra el anillo miocárdico, así que la recon
sirve. Pero el SA reorientado sale amorfo => el eje largo de auto_orient_lv
está mal (dio casi puro eje z: (0.955, -0.077, -0.287)).

Este script prueba VARIOS ejes largos / centros y guarda el SA de cada uno,
para ver cuál da la dona limpia:
  A) auto_orient_lv sobre el volumen COMPLETO (referencia, la que falla).
  B) auto_orient_lv sobre un RECORTE al corazón (excluye intestino/hígado).
  C) PCA manual sobre los vóxeles del anillo en el recorte.
  D) ejes largos oblicuos fijos típicos (para calibrar visualmente).

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _xel3_orient_sweep.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cardiac_reorientation import (
    auto_orient_lv,
    reslice_from_vector_gated,
    sa_stack,
)
from core.raw_projections import load_raw_projections
from core.raw_reconstruction import ProjectionFilterConfig, reconstruct_gated_fbp_volume

BASE = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
STRESS = os.path.join(BASE, "Stress-10sec-1_G_EM_1001_DS.dcm")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_xel3_out")


def sa_grid(sa, title, path, cmap="hot"):
    """Grilla de la pila SA con ventaneo POR CORTE (como Evolution)."""
    n = sa.shape[0]
    idxs = np.linspace(0, n - 1, min(n, 16)).astype(int)
    cols = 4
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for k, i in enumerate(idxs):
        ax = np.atleast_1d(axes).ravel()[k]
        vmax = float(np.percentile(sa[i], 99.5)) or float(sa.max())
        ax.imshow(sa[i], cmap=cmap, vmin=0, vmax=max(vmax, 1e-6))
        ax.set_title(f"k={i}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print("  guardado:", path)


def motion_map(cube):
    """|FFT[1]| por vóxel (amplitud del 1er armónico) = mapa de movimiento."""
    c = cube - cube.mean(axis=0, keepdims=True)
    m = np.abs(np.fft.fft(c, axis=0)[1])
    try:
        from scipy.ndimage import gaussian_filter
        m = gaussian_filter(m, sigma=1.0)
    except Exception:
        pass
    return m


def pca_long_axis(vol3d, center, radius, thr_frac=0.5):
    """PCA ponderado sobre vóxeles brillantes dentro de una esfera alrededor
    de ``center``. Devuelve (long_axis unit (z,y,x), refined_center)."""
    zc, yc, xc = center
    zz, yy, xx = np.mgrid[0:vol3d.shape[0], 0:vol3d.shape[1], 0:vol3d.shape[2]]
    dist2 = (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2
    inside = dist2 <= radius ** 2
    sub = np.where(inside, vol3d, 0.0)
    thr = thr_frac * float(sub.max())
    mask = sub >= thr
    if mask.sum() < 10:
        return None, center
    pz, py, px = np.nonzero(mask)
    w = sub[pz, py, px]
    sw = w.sum()
    rcz = float((pz * w).sum() / sw)
    rcy = float((py * w).sum() / sw)
    rcx = float((px * w).sum() / sw)
    pts = np.stack([pz - rcz, py - rcy, px - rcx], axis=1)
    cov = (pts.T * w) @ pts / sw
    evals, evecs = np.linalg.eigh(cov)
    u = np.asarray(evecs[:, -1], dtype=np.float64)
    u /= (np.linalg.norm(u) or 1.0)
    return u, (rcz, rcy, rcx)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Cargando STRESS gated...")
    raw = load_raw_projections(STRESS)
    pf = ProjectionFilterConfig("butterworth", 0.40, 10)
    print("Reconstruyendo FBP gated (Butterworth 0.4/10)...")
    vol = reconstruct_gated_fbp_volume(raw.projections, raw.angles_deg, projection_filter=pf)
    ung = vol.mean(axis=0)
    print(f"  volumen gated={vol.shape}")

    # --- A) AUTO sobre volumen completo (referencia que falla) ---
    a = auto_orient_lv(vol, ung)
    print("\nA) AUTO completo:")
    print(f"   center={tuple(round(v,1) for v in a['center'])}  la={tuple(round(float(v),3) for v in a['long_axis'])}")
    reo = reslice_from_vector_gated(vol, a["center"], a["long_axis"], 40, order=1)
    sa_grid(sa_stack(reo[0]), "A) SA auto-completo (FALLA)", os.path.join(OUTDIR, "SW_A_auto_full.png"))

    # --- Recorte al corazón alrededor del pico del mapa de movimiento ---
    mm = motion_map(vol)
    pk = np.unravel_index(int(np.argmax(mm)), mm.shape)
    print(f"\nPico de movimiento (z,y,x)={pk}")
    R = 14  # radio del recorte esférico del corazón (vóxeles)

    # --- B) AUTO sobre recorte cúbico alrededor del pico ---
    z0, y0, x0 = [max(0, p - R) for p in pk]
    z1, y1, x1 = [min(s, p + R + 1) for p, s in zip(pk, vol.shape[1:])]
    crop = vol[:, z0:z1, y0:y1, x0:x1]
    ungc = crop.mean(axis=0)
    b = auto_orient_lv(crop, ungc)
    if b is not None:
        # el centro/eje están en coords del recorte; el reslice se hace sobre el
        # volumen COMPLETO, así que trasladamos el centro a coords globales.
        gc = (b["center"][0] + z0, b["center"][1] + y0, b["center"][2] + x0)
        print("B) AUTO recorte:")
        print(f"   center_global={tuple(round(v,1) for v in gc)}  la={tuple(round(float(v),3) for v in b['long_axis'])}")
        reo = reslice_from_vector_gated(vol, gc, b["long_axis"], 40, order=1)
        sa_grid(sa_stack(reo[0]), "B) SA auto-recorte", os.path.join(OUTDIR, "SW_B_auto_crop.png"))

    # --- C) PCA manual sobre el anillo brillante alrededor del pico ---
    u_pca, c_pca = pca_long_axis(ung, pk, radius=R, thr_frac=0.5)
    if u_pca is not None:
        print("C) PCA manual anillo:")
        print(f"   center={tuple(round(v,1) for v in c_pca)}  la={tuple(round(float(v),3) for v in u_pca)}")
        reo = reslice_from_vector_gated(vol, c_pca, u_pca, 40, order=1)
        sa_grid(sa_stack(reo[0]), "C) SA PCA-manual", os.path.join(OUTDIR, "SW_C_pca.png"))

    # --- D) ejes largos oblicuos fijos típicos, centrados en el pico ---
    # apex antero-inferior-izquierdo: probamos un abanico en (z,y,x).
    fixed = {
        "D1_45yx": np.array([0.0, -0.707, -0.707]),
        "D2_zyx": np.array([0.5, -0.6, -0.6]),
        "D3_ap": np.array([0.0, -1.0, 0.0]),
        "D4_lat": np.array([0.0, 0.0, -1.0]),
    }
    for name, u in fixed.items():
        reo = reslice_from_vector_gated(vol, tuple(float(p) for p in pk), u, 40, order=1)
        sa_grid(sa_stack(reo[0]), f"D) SA fijo {name} la={tuple(round(float(v),2) for v in u)}",
                os.path.join(OUTDIR, f"SW_{name}.png"))

    print("\nListo. Imágenes SW_*.png en:", OUTDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
