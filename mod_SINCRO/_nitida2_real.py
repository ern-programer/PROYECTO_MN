"""NITIDA II sobre el par REAL 5s/10s en ESPACIO DE PROYECCIÓN (raw gated).

Los DICOM SGATE*seg_G son proyecciones crudas (8 gates × 60 ángulos × 64×64).
El filtrado temporal por armónicos es LINEAL a lo largo de los gates, así que
conmuta con la retroproyección: medirlo en proyecciones equivale a medirlo en el
volumen reconstruido.

Descompone la señal temporal (8 gates) por bandas de armónicos:
  - armónicos 1-2  -> MOVIMIENTO cardíaco (baja frecuencia). Debe conservarse.
  - armónicos 3-4  -> RUIDO (banda alta). Debe caer.
Y mide el ruido como std en fondo (aire).

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _nitida2_real.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.dicom_loader import load
from core.nitida2 import denoise_gates_with_guide, temporal_harmonic_filter

BASE = r"D:\- GAMMASYS\estudios evolution\New Folder\1.2.840.113619.2.265.1.2.0.9092025113215781.32146"
P5 = os.path.join(BASE, "SGATE5seg_G_1001_DS.dcm")
P10 = os.path.join(BASE, "SGATE10seg_G_1001_DS.dcm")


def masks(ungated: np.ndarray):
    """Máscara de señal (alto conteo) y de fondo (aire) desde el ungated."""
    sig = ungated >= np.percentile(ungated, 90)
    bg = ungated <= np.percentile(ungated, 40)
    return sig, bg


def noise_std(cube: np.ndarray, bg: np.ndarray) -> float:
    return float(np.mean([np.std(cube[g][bg]) for g in range(cube.shape[0])]))


def harmonic_power(cube: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Potencia media por armónico (bins rfft) dentro de la máscara de señal.

    Devuelve vector [P0(DC), P1, P2, P3, P4] normalizado por el DC (%).
    """
    ng = cube.shape[0]
    ts = np.stack([cube[g][sig] for g in range(ng)], axis=1)  # (n_sig, n_gates)
    spec = np.fft.rfft(ts, axis=1)                            # (n_sig, n_bins)
    power = np.mean(np.abs(spec) ** 2, axis=0)                # media sobre voxels
    dc = power[0] + 1e-12
    return power / dc * 100.0


def report(name: str, cube: np.ndarray, sig: np.ndarray, bg: np.ndarray) -> None:
    p = harmonic_power(cube, sig)
    mov = p[1] + (p[2] if len(p) > 2 else 0.0)
    noi = (p[3] if len(p) > 3 else 0.0) + (p[4] if len(p) > 4 else 0.0)
    print(f"  {name:20s}: ruido_bg={noise_std(cube, bg):7.3f}  "
          f"H1-2(mov)={mov:8.2f}%  H3-4(ruido)={noi:7.3f}%  "
          f"[H1={p[1]:.2f} H2={p[2]:.2f} H3={p[3]:.3f} H4={p[4]:.3f}]")


def main() -> None:
    s5 = load(P5)
    s10 = load(P10)
    c5 = np.asarray(s5.cube, dtype=np.float64)
    c10 = np.asarray(s10.cube, dtype=np.float64)
    ung5 = c5.sum(axis=0)
    ung10 = c10.sum(axis=0)
    sig5, bg5 = masks(ung5)
    sig10, bg10 = masks(ung10)

    print(f"5s/10s: {c5.shape} (gates,ang,H,W)  total 5s={c5.sum():.0f}  10s={c10.sum():.0f}")
    print("Potencia por armónico normalizada al DC. H1-2 = movimiento (conservar), "
          "H3-4 = ruido (bajar).\n")

    print("=== REFERENCIAS ===")
    report("5s crudo", c5, sig5, bg5)
    report("10s oro", c10, sig10, bg10)

    print("\n=== NITIDA II · filtro temporal armónico (5s) ===")
    report("temporal keep=2", temporal_harmonic_filter(c5, n_harmonics=2), sig5, bg5)
    report("temporal keep=1", temporal_harmonic_filter(c5, n_harmonics=1), sig5, bg5)

    print("\n=== comparación: guided-ungated (5s) ===")
    report("guided r1 eps.02", denoise_gates_with_guide(c5, guide_volume=ung5, radius=1, eps=0.02), sig5, bg5)

    print(
        "\n[LEER] El temporal keep=2 debe: ruido_bg ↓, H3-4 ≈ 0 (por construcción), "
        "H1-2 IGUAL al crudo (movimiento intacto). Si el guided baja H1-2 respecto "
        "del crudo, está comiendo movimiento."
    )


if __name__ == "__main__":
    main()
