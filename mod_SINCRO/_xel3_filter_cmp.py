"""Comparación de FILTROS de FBP sobre el Xeleris 3 (datos sumados no-gated).

Hipótesis: las estrías radiales del transaxial vienen del FBP con rampa cruda
(Ram-Lak) sobre 60 vistas (sub-Nyquist para 64px). Evolution las suaviza con
apodización fuerte. Comparamos, sobre el MISMO sinograma:

  1) actual: Butterworth 0.4/10 pre + iradon rampa (Ram-Lak)
  2) Butterworth 0.4/10 pre + iradon 'hann'
  3) sin pre + iradon 'hann'
  4) sin pre + iradon 'cosine'
  5) Butterworth 0.3/8 (más agresivo) pre + iradon 'shepp-logan'

Mostramos 6 cortes centrales de cada variante, ventaneados por corte, para ver
cuál da un miocardio compacto / anillo limpio como Evolution.

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _xel3_filter_cmp.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.raw_projections import load_raw_projections
from core.raw_reconstruction import (
    ProjectionFilterConfig,
    filter_projections,
    reconstruct_fbp_volume,
)

BASE = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
# Tomográfico SUMADO (no-gated): mejor SNR que promediar gates.
SUMMED = os.path.join(BASE, "Stress-10sec-1_T_EM001_DS.dcm")
GATED = os.path.join(BASE, "Stress-10sec-1_G_EM_1001_DS.dcm")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_xel3_out")


def recon(proj, angles, pre, fbp_filter):
    p = proj
    if pre is not None:
        p = filter_projections(proj[None], pre)[0] if proj.ndim == 3 else proj
        p = filter_projections(proj, pre)
    return reconstruct_fbp_volume(p, angles, projection_filter=None, fbp_filter_name=fbp_filter)


def central_row(vol, title, cmap="hot"):
    """Devuelve (fig-listo) 6 cortes centrales ventaneados por corte."""
    n = vol.shape[0]
    idxs = np.linspace(n * 0.30, n * 0.70, 6).astype(int)
    return idxs


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    src = SUMMED if os.path.exists(SUMMED) else GATED
    print("Cargando proyecciones:", os.path.basename(src))
    raw = load_raw_projections(src)
    proj = raw.projections
    if proj.ndim == 4:  # gated -> sumar gates
        proj = proj.sum(axis=0)
    print(f"  proj={proj.shape}  counts={proj.sum():,.0f}  "
          f"angles start={raw.start_angle} step={raw.angular_step} dir={raw.rotation_direction} arc={raw.scan_arc}")

    variants = [
        ("1_actual_Bw0.4_ramp", ProjectionFilterConfig("butterworth", 0.40, 10), "ramp"),
        ("2_Bw0.4_hann", ProjectionFilterConfig("butterworth", 0.40, 10), "hann"),
        ("3_nopre_hann", None, "hann"),
        ("4_nopre_cosine", None, "cosine"),
        ("5_Bw0.3_shepp", ProjectionFilterConfig("butterworth", 0.30, 8), "shepp-logan"),
    ]

    recons = {}
    for name, pre, filt in variants:
        print(f"Reconstruyendo {name} (pre={pre.kind if pre else 'none'} filt={filt})...")
        recons[name] = recon(proj, raw.angles_deg, pre, filt)

    idxs = central_row(next(iter(recons.values())), "")
    rows = len(variants)
    cols = len(idxs)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 1.9))
    for r, (name, _, _) in enumerate(variants):
        vol = recons[name]
        for c, z in enumerate(idxs):
            ax = axes[r, c]
            ax.axis("off")
            sl = vol[z]
            vmax = float(np.percentile(sl, 99.5)) or float(vol.max())
            ax.imshow(sl, cmap="hot", vmin=0, vmax=max(vmax, 1e-6))
            if c == 0:
                ax.set_ylabel(name, fontsize=7)
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"z={z}", fontsize=6)
    fig.suptitle("Comparación filtros FBP - transaxial sumado (ventaneo por corte)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "FCMP_filters.png")
    fig.savefig(out, dpi=95)
    plt.close(fig)
    print("\nGuardado:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
