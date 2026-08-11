"""Validación NITIDA II sobre estudio Xeleris 3 (16 gates, SPECT/CT Evolution).

NO es producto: arnés de auditoría. Reconstruye el gated de STRESS (bajo conteo
relativo) por FBP, aplica NITIDA II (temporal / espaciotemporal) y mide:
  - ruido de fondo (std en background, ↓ mejor),
  - granulado espacial en miocardio (std del Laplaciano, ↓ mejor),
  - potencia armónica temporal H1..Hn relativa a DC (el MOVIMIENTO vive en H1-2;
    NITIDA II debe CONSERVAR H1-2 y ANULAR las bandas altas de ruido).

Con 16 gates la frecuencia de Nyquist temporal es alta: el movimiento cardíaco
sigue concentrado en H1-2 (1-2 ciclos), y H3..H8 son casi todo ruido de Poisson.

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _nitida2_xel3.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.nitida2 import denoise_spatiotemporal, temporal_harmonic_filter
from core.raw_projections import load_raw_projections
from core.raw_reconstruction import ProjectionFilterConfig, reconstruct_gated_fbp_volume

BASE = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
STRESS = os.path.join(BASE, "Stress-10sec-1_G_EM_1001_DS.dcm")  # gated bajo conteo (oro relativo)
REST = os.path.join(BASE, "RGATE-10sec_G_EM_1001_DS.dcm")       # gated alto conteo

N_HARM = 2
BAND_SIGMA = 0.7


def masks(ungated_vol: np.ndarray):
    sig = ungated_vol >= np.percentile(ungated_vol, 96)
    bg = ungated_vol <= np.percentile(ungated_vol, 50)
    return sig, bg


def noise_std(cube: np.ndarray, bg: np.ndarray) -> float:
    return float(np.mean([np.std(cube[g][bg]) for g in range(cube.shape[0])]))


def harmonic_power(cube: np.ndarray, sig: np.ndarray) -> np.ndarray:
    ng = cube.shape[0]
    ts = np.stack([cube[g][sig] for g in range(ng)], axis=1)
    spec = np.fft.rfft(ts, axis=1)
    power = np.mean(np.abs(spec) ** 2, axis=0)
    return power / (power[0] + 1e-12) * 100.0


def roughness(cube: np.ndarray, sig: np.ndarray) -> float:
    from scipy.ndimage import laplace
    vals = []
    for g in range(cube.shape[0]):
        lap = laplace(cube[g])
        vals.append(np.std(lap[sig]))
    return float(np.mean(vals))


def report(tag, cube, sig, bg, ref_hp=None):
    hp = harmonic_power(cube, sig)
    n = noise_std(cube, bg)
    r = roughness(cube, sig)
    h12 = hp[1] + (hp[2] if len(hp) > 2 else 0.0)
    hhi = float(np.sum(hp[3:])) if len(hp) > 3 else 0.0
    d = ""
    if ref_hp is not None:
        ref12 = ref_hp[1] + (ref_hp[2] if len(ref_hp) > 2 else 0.0)
        d = f"  (H1-2 vs crudo: {h12/ref12*100:5.1f}%)"
    print(f"  {tag:<28} ruido_bg={n:7.4f}  granulado={r:7.4f}  "
          f"H1-2={h12:6.3f}%  H3+={hhi:7.4f}%{d}")
    return hp


def main():
    print("Cargando Xeleris 3 (16 gates)...")
    raw_s = load_raw_projections(STRESS)
    raw_r = load_raw_projections(REST)
    print(f"  STRESS gated: {raw_s.projections.shape}  counts={raw_s.projections.sum():,.0f}")
    print(f"  REST   gated: {raw_r.projections.shape}  counts={raw_r.projections.sum():,.0f}")

    pf = ProjectionFilterConfig("butterworth", 0.40, 10)
    t0 = time.time()
    print("\nReconstruyendo FBP gated (STRESS)...")
    vs = reconstruct_gated_fbp_volume(raw_s.projections, raw_s.angles_deg, projection_filter=pf)
    print(f"  -> {vs.shape} en {time.time()-t0:.0f}s")

    ung_s = vs.mean(axis=0)
    sig, bg = masks(ung_s)

    print("\n=== STRESS gated (16 gates) — crudo vs NITIDA II ===")
    ref = report("crudo (baseline)", vs, sig, bg)
    vs_temp = temporal_harmonic_filter(vs, n_harmonics=N_HARM, axis=0)
    report(f"NITIDA II temporal keep={N_HARM}", vs_temp, sig, bg, ref)
    vs_st = denoise_spatiotemporal(vs, n_harmonics=N_HARM, band_sigma=BAND_SIGMA,
                                   guide_volume=ung_s / vs.shape[0])
    report("NITIDA II espaciotemporal", vs_st, sig, bg, ref)

    print("\nInterpretación:")
    print("  - H1-2 vs crudo ≈100% => el MOVIMIENTO se conserva (bueno).")
    print("  - H3+ debe caer a ~0 (temporal) => se quita ruido de banda alta.")
    print("  - granulado ↓ (espaciotemporal) sin tocar H1-2 => limpia sin dañar función.")
    print("\nListo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
