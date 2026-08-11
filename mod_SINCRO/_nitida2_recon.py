"""NITIDA II · validación en espacio RECONSTRUIDO (FBP) del par real 5s/10s.

En proyección el movimiento cardíaco es sub-ruido por-píxel (espectro plano). La
retroproyección lo vuelve coherente: en el miocardio reconstruido la 1ª armónica
(contracción) supera al ruido. Aquí reconstruimos 5s y 10s con FBP (lineal) y
medimos, dentro del miocardio, la potencia por armónico y el ruido.

Como el filtro temporal es lineal, filtrar proyecciones == filtrar el volumen;
aplicamos temporal_harmonic_filter directamente al volumen reconstruido.

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _nitida2_recon.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.dicom_loader import load
from core.nitida2 import (
    denoise_gates_with_guide,
    denoise_spatiotemporal,
    temporal_harmonic_filter,
)
from core.raw_reconstruction import reconstruct_gated_projection_volume

BASE = r"D:\- GAMMASYS\estudios evolution\New Folder\1.2.840.113619.2.265.1.2.0.9092025113215781.32146"
P5 = os.path.join(BASE, "SGATE5seg_G_1001_DS.dcm")
P10 = os.path.join(BASE, "SGATE10seg_G_1001_DS.dcm")


def masks(ungated_vol: np.ndarray):
    sig = ungated_vol >= np.percentile(ungated_vol, 96)   # miocardio (compacto)
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
    """Granulado espacial en el miocardio: std del Laplaciano por gate (↓ mejor)."""
    from scipy.ndimage import laplace
    vals = []
    for g in range(cube.shape[0]):
        lap = laplace(cube[g])
        vals.append(float(np.std(lap[sig])))
    return float(np.mean(vals))


def report(name: str, cube: np.ndarray, sig: np.ndarray, bg: np.ndarray) -> None:
    p = harmonic_power(cube, sig)
    mov = p[1] + p[2]
    noi = p[3] + p[4]
    print(f"  {name:20s}: ruido_bg={noise_std(cube, bg):7.3f}  granulado={roughness(cube, sig):7.3f}  "
          f"H1-2(mov)={mov:7.2f}%  H3-4={noi:6.3f}%  "
          f"[H1={p[1]:.2f} H2={p[2]:.2f}]")


def recon(cube_proj: np.ndarray, angles) -> np.ndarray:
    return reconstruct_gated_projection_volume(cube_proj, angles, method="fbp",
                                               fbp_filter_name="hann")


def main() -> None:
    s5 = load(P5)
    s10 = load(P10)
    ang = s5.angles_deg
    print(f"Reconstruyendo FBP (hann)... 5s y 10s, {s5.cube.shape} proyecciones")
    v5 = recon(np.asarray(s5.cube, dtype=np.float64), ang)
    v10 = recon(np.asarray(s10.cube, dtype=np.float64), ang)
    print(f"volúmenes: 5s {v5.shape}  10s {v10.shape}")

    ung5 = v5.sum(axis=0)
    ung10 = v10.sum(axis=0)
    sig5, bg5 = masks(ung5)
    sig10, bg10 = masks(ung10)

    print("\nH1-2 = movimiento cardíaco coherente (conservar); H3-4 = ruido (bajar).")
    print("=== REFERENCIAS ===")
    report("5s crudo", v5, sig5, bg5)
    report("10s oro", v10, sig10, bg10)

    print("\n=== NITIDA II · temporal armónico (5s) ===")
    report("temporal keep=2", temporal_harmonic_filter(v5, n_harmonics=2), sig5, bg5)
    report("temporal keep=1", temporal_harmonic_filter(v5, n_harmonics=1), sig5, bg5)

    print("\n=== NITIDA II · espaciotemporal por bandas (Ingrediente 2, 5s) ===")
    report("spatiotemp s0.7", denoise_spatiotemporal(v5, n_harmonics=2, guide_volume=ung5 / v5.shape[0],
                                                      dc_radius=2, dc_eps=0.01, band_sigma=0.7), sig5, bg5)
    report("spatiotemp s1.0", denoise_spatiotemporal(v5, n_harmonics=2, guide_volume=ung5 / v5.shape[0],
                                                      dc_radius=2, dc_eps=0.005, band_sigma=1.0), sig5, bg5)

    print("\n=== comparación: guided-ungated (5s) ===")
    report("guided r1 eps.02", denoise_gates_with_guide(v5, guide_volume=ung5, radius=1, eps=0.02), sig5, bg5)
    report("guided r2 eps.01", denoise_gates_with_guide(v5, guide_volume=ung5, radius=2, eps=0.01), sig5, bg5)

    print(
        "\n[LEER] Si en el miocardio H1 >> H3,H4 (crudo), hay movimiento coherente. "
        "temporal keep=2 debe: ruido_bg ↓, H3-4→0, H1-2 casi igual al crudo. "
        "El guided baja H1-2 (come movimiento) si su H1 cae respecto del crudo."
    )


if __name__ == "__main__":
    main()
