"""Validación NITIDA II sobre fantoma sintético (rama NITIDA_II).

NO es producto. Prueba la afirmación central del denoiser guiado:
  1. BAJA el ruido de los gates de bajo conteo (5s).
  2. NO aplasta el movimiento (la contracción ED->ES se conserva).

Fantoma: anillo miocárdico de eje corto que se CONTRAE (cavidad grande en ED,
chica en ES) y engrosa. Se simula ruido Poisson a dos niveles (10s alto conteo,
5s bajo conteo). Se denoisa el 5s con el ungated (suma de gates) como guía y se
mide ruido y contracción antes/después, contra el 10s "oro".

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _nitida2_eval.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.nitida2 import denoise_gates_with_guide

RNG = np.random.default_rng(42)

N_GATES = 8
N_SLICES = 16
H = W = 48
CY = CX = H / 2.0
WALL_COUNTS = 100.0          # nivel de actividad de la pared (unidades arbitrarias)
BG_FLOOR = 8.0               # piso de fondo (scatter/tejido blando) para ruido realista


def _ring(cavity_r: float, wall_thick: float) -> np.ndarray:
    """Un corte: anillo de pared entre cavity_r y cavity_r+wall_thick."""
    ys, xs = np.ogrid[:H, :W]
    d = np.sqrt((ys - CY) ** 2 + (xs - CX) ** 2)
    inner = cavity_r
    outer = cavity_r + wall_thick
    return ((d >= inner) & (d <= outer)).astype(np.float64)


def build_phantom() -> tuple[np.ndarray, np.ndarray]:
    """Fantoma limpio 4D (gates,slices,H,W) y su curva de área de cavidad ideal."""
    cube = np.full((N_GATES, N_SLICES, H, W), BG_FLOOR, dtype=np.float64)
    # Contracción sinusoidal: ED (gate 0) cavidad grande; ES (~gate 4) chica.
    for g in range(N_GATES):
        phase = 2.0 * np.pi * g / N_GATES
        contract = 0.5 * (1.0 + np.cos(phase))   # 1 en ED, 0 en ES
        cavity_r = 6.0 + 5.0 * contract          # 11 (ED) -> 6 (ES)
        wall_thick = 4.0 + 2.0 * (1.0 - contract)  # engrosa en ES
        # Apex->base: la cavidad se cierra hacia el apex (slices extremos).
        for s in range(N_SLICES):
            axial = np.sin(np.pi * (s + 0.5) / N_SLICES)  # 0 en extremos, 1 al medio
            r_s = cavity_r * axial
            if r_s < 1.0:
                continue
            cube[g, s] += _ring(r_s, wall_thick) * WALL_COUNTS * axial
    return cube


def poisson(cube: np.ndarray, dose_scale: float) -> np.ndarray:
    """Ruido Poisson: dose_scale escala las cuentas (10s=1.0, 5s=0.5)."""
    lam = np.clip(cube * dose_scale, 0.0, None)
    return RNG.poisson(lam).astype(np.float64)


def noise_std(cube: np.ndarray) -> float:
    """Ruido: desvío en una zona de FONDO uniforme (esquinas, sin señal)."""
    corner = cube[:, :, :8, :8]
    return float(np.std(corner))


def wall_cnr(cube: np.ndarray) -> float:
    """CNR pared vs fondo: (media_pared - media_fondo)/std_fondo, gate ED."""
    ed = cube[0]
    wall = ed[ed > 0.3 * ed.max()]
    bg = ed[:, :8, :8] if ed.ndim == 3 else ed[:8, :8]
    mu_w = float(wall.mean()) if wall.size else 0.0
    mu_b = float(bg.mean())
    sd_b = float(bg.std()) + 1e-9
    return (mu_w - mu_b) / sd_b


def cavity_area_curve(cube: np.ndarray) -> np.ndarray:
    """Área de cavidad por gate (px) por umbral, en el corte central."""
    s = N_SLICES // 2
    areas = np.zeros(cube.shape[0])
    for g in range(cube.shape[0]):
        img = cube[g, s]
        thr = 0.4 * img.max() if img.max() > 0 else 1.0
        ys, xs = np.ogrid[:H, :W]
        d = np.sqrt((ys - CY) ** 2 + (xs - CX) ** 2)
        # Cavidad = zona central por debajo del umbral rodeada de pared.
        cav = (d <= 12) & (img < thr)
        areas[g] = float(cav.sum())
    return areas


def ef_from_curve(areas: np.ndarray) -> float:
    edv, esv = float(areas.max()), float(areas.min())
    return (edv - esv) / edv * 100.0 if edv > 0 else 0.0


def main() -> None:
    clean = build_phantom()
    gold10 = poisson(clean, dose_scale=1.0)     # 10s alto conteo
    low5 = poisson(clean, dose_scale=0.5)       # 5s bajo conteo

    # NITIDA II: denoisar el 5s con el ungated (suma de gates) como guía.
    guide = low5.sum(axis=0)
    den5 = denoise_gates_with_guide(low5, guide_volume=guide, radius=2, eps=0.01)

    print("=== RUIDO (std en fondo uniforme; más bajo = mejor) ===")
    print(f"  10s (oro)      : {noise_std(gold10):.2f}")
    print(f"  5s crudo       : {noise_std(low5):.2f}")
    print(f"  5s NITIDA II   : {noise_std(den5):.2f}")

    print("\n=== CNR pared/fondo (gate ED; más alto = mejor) ===")
    print(f"  10s (oro)      : {wall_cnr(gold10):.2f}")
    print(f"  5s crudo       : {wall_cnr(low5):.2f}")
    print(f"  5s NITIDA II   : {wall_cnr(den5):.2f}")

    print("\n=== MOVIMIENTO: FEVI por área de cavidad (debe CONSERVARSE) ===")
    ef_clean = ef_from_curve(cavity_area_curve(clean))
    print(f"  Fantoma limpio : EF={ef_clean:.1f}%  (referencia real)")
    print(f"  10s (oro)      : EF={ef_from_curve(cavity_area_curve(gold10)):.1f}%")
    print(f"  5s crudo       : EF={ef_from_curve(cavity_area_curve(low5)):.1f}%")
    print(f"  5s NITIDA II   : EF={ef_from_curve(cavity_area_curve(den5)):.1f}%")
    print(
        "\n[LEER] NITIDA II debe: (a) bajar el std de fondo hacia el 10s, "
        "(b) subir el CNR, (c) mantener la EF cerca del fantoma limpio "
        "(si la EF se DESPLOMA, el filtro estaría borrando el movimiento)."
    )


if __name__ == "__main__":
    main()
