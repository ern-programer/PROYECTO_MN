"""Test end-to-end: pipeline de reconstrucción con NITIDA II activado (5s real).

Corre reconstruct_raw_gated_pipeline con nitida2_mode none/temporal/spatiotemporal
sobre el 5s real y compara ruido/granulado/movimiento en el volumen gated final.

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _nitida2_pipeline.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import laplace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.dicom_loader import load
from core.raw_reconstruction import RawReconConfig, reconstruct_raw_gated_pipeline

BASE = r"D:\- GAMMASYS\estudios evolution\New Folder\1.2.840.113619.2.265.1.2.0.9092025113215781.32146"
P5 = os.path.join(BASE, "SGATE5seg_G_1001_DS.dcm")


def metrics(gated: np.ndarray):
    ung = gated.sum(axis=0)
    sig = ung >= np.percentile(ung, 96)
    bg = ung <= np.percentile(ung, 50)
    ruido = float(np.mean([np.std(gated[g][bg]) for g in range(gated.shape[0])]))
    gran = float(np.mean([np.std(laplace(gated[g])[sig]) for g in range(gated.shape[0])]))
    ts = np.stack([gated[g][sig] for g in range(gated.shape[0])], axis=1)
    p = np.mean(np.abs(np.fft.rfft(ts, axis=1)) ** 2, axis=0)
    p = p / (p[0] + 1e-12) * 100.0
    return ruido, gran, p[1] + p[2], p[3] + p[4]


def run(mode: str):
    s5 = load(P5)
    proj = np.asarray(s5.cube, dtype=np.float64)
    cfg = RawReconConfig(reconstruction_method="fbp", nitida2_mode=mode, nitida2_band_sigma=0.7)
    res = reconstruct_raw_gated_pipeline(proj, s5.angles_deg, config=cfg)
    return res.gated_volume


def main() -> None:
    print("Pipeline end-to-end (FBP + motion correction interna), 5s real.\n")
    print(f"  {'modo':16s}  {'ruido_bg':>9s}  {'granulado':>9s}  {'H1-2(mov)':>10s}  {'H3-4':>7s}")
    for mode in ("none", "temporal", "spatiotemporal"):
        r, g, mov, noi = metrics(run(mode))
        print(f"  {mode:16s}  {r:9.4f}  {g:9.4f}  {mov:9.2f}%  {noi:6.3f}%")
    print("\n[LEER] spatiotemporal debe bajar granulado y H3-4 manteniendo H1-2 razonable.")


if __name__ == "__main__":
    main()
