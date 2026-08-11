"""Reconstruye el Xeleris 3 como Evolution (FBP + Butterworth 0.4/10, sin AC)
y guarda PNGs de cortes transaxiales y del SA reorientado, para comparar
lado a lado con la captura de Evolution.

Objetivo: ver si el volumen TRANSAXIAL (antes de reorientar) ya sale como anillo
o como masa amorfa. Si el transaxial está bien pero el SA no, el problema es la
reorientación; si el transaxial ya está mal, el problema es la reconstrucción.

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _xel3_recon_img.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cardiac_reorientation import auto_orient_lv, reslice_from_vector_gated, sa_stack
from core.raw_projections import load_raw_projections
from core.raw_reconstruction import ProjectionFilterConfig, reconstruct_gated_fbp_volume

BASE = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
STRESS = os.path.join(BASE, "Stress-10sec-1_G_EM_1001_DS.dcm")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_xel3_out")


def grid(vol, title, path, cmap="hot", win="global"):
    """Monta cortes de un volumen 3D (z,y,x) en una grilla.

    win='global' -> vmax = percentil global (puede diluir el corazón si hay
    intestino hiperbrillante). win='perslice' -> vmax al máximo de CADA corte
    (como Evolution, que ventanea local) => el anillo destaca aunque haya un
    píxel extracardíaco muy brillante en otro corte.
    """
    n = vol.shape[0]
    idxs = np.linspace(0, n - 1, min(n, 16)).astype(int)
    cols = 4
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    gvmax = float(np.percentile(vol, 99.5))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for k, i in enumerate(idxs):
        ax = np.atleast_1d(axes).ravel()[k]
        if win == "perslice":
            vmax = float(np.percentile(vol[i], 99.5)) or gvmax
        else:
            vmax = gvmax
        ax.imshow(vol[i], cmap=cmap, vmin=0, vmax=max(vmax, 1e-6))
        ax.set_title(f"z={i}", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print("  guardado:", path)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Cargando STRESS gated...")
    raw = load_raw_projections(STRESS)
    print(f"  proyecciones={raw.projections.shape}  angles: start={raw.start_angle} "
          f"step={raw.angular_step} dir={raw.rotation_direction} arc={raw.scan_arc}")
    print(f"  angles_deg[:5]={np.round(raw.angles_deg[:5],1)}  counts={raw.projections.sum():,.0f}")

    pf = ProjectionFilterConfig("butterworth", 0.40, 10)
    print("\nReconstruyendo FBP gated como Evolution (Butterworth 0.4/10)...")
    vol = reconstruct_gated_fbp_volume(raw.projections, raw.angles_deg, projection_filter=pf)
    print(f"  volumen gated={vol.shape}")

    # ED = gate 0. Transaxial crudo (lo que sale del FBP, antes de reorientar).
    ed = vol[0]
    grid(ed, "TRANSAXIAL FBP Bw0.4/10 - gate ED (z=cortes axiales)",
         os.path.join(OUTDIR, "1_transaxial_ed.png"))

    # Ungated (media de gates) para orientación.
    ung = vol.mean(axis=0)
    grid(ung, "TRANSAXIAL FBP - ungated (media gates)",
         os.path.join(OUTDIR, "2_transaxial_ungated.png"))

    # Mismo volumen, ventaneo POR CORTE (como Evolution). Si el anillo aparece
    # acá, el problema era el ventaneo de display, no la reconstrucción.
    grid(ed, "TRANSAXIAL gate ED - ventaneo POR CORTE (estilo Evolution)",
         os.path.join(OUTDIR, "1b_transaxial_ed_perslice.png"), win="perslice")
    grid(ung, "TRANSAXIAL ungated - ventaneo POR CORTE",
         os.path.join(OUTDIR, "2b_transaxial_ungated_perslice.png"), win="perslice")

    # Reorientación automática y SA.
    print("\nReorientación automática (auto_orient_lv)...")
    orient = auto_orient_lv(vol, ung)
    if orient is not None:
        center, la = orient["center"], orient["long_axis"]
        print(f"  center(z,y,x)={tuple(round(v,1) for v in center)}  long_axis={tuple(round(float(v),3) for v in la)}")
        reo = reslice_from_vector_gated(vol, center, la, 40, order=1)
        sa = sa_stack(reo[0])
        grid(sa, "SA reorientado AUTO - gate ED (anillo esperado)",
             os.path.join(OUTDIR, "3_sa_auto_ed.png"))
    else:
        print("  auto_orient_lv devolvió None")

    print("\nListo. Imágenes en:", OUTDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
