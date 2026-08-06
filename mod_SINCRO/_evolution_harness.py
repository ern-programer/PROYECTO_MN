"""Banco de validación half-dose / half-time (prototipo estilo GE Evolution).

Objetivo: demostrar, sobre un estudio SPECT real, el problema del bajo conteo y
cómo la reconstrucción iterativa con recuperación de resolución (RR) lo compensa.

Flujo:
  1. Carga proyecciones crudas gated reales (o sintéticas si no hay estudio).
  2. Las "ungatea" (suma de gates) -> proyecciones 3D (ang, H, W) en CUENTAS.
  3. Simula MITAD (y CUARTO) de dosis/tiempo por *thinning binomial* de Poisson:
     half = Binomial(n=full_counts, p=0.5). Esto es EXACTO: adelgazar un Poisson
     con probabilidad p da otro Poisson con tasa p·λ (no una aproximación).
  4. Sobre una banda de cortes al nivel del corazón reconstruye:
       - ref      = FBP con conteo COMPLETO   (referencia "verdad práctica")
       - fbp_half  = FBP con MITAD de conteo
       - osem_half = OSEM con MITAD de conteo (sin RR)
       - rr_half   = OSEM + Resolution Recovery con MITAD de conteo (Evolution-like)
  5. Métricas vs. referencia: NRMSE, ruido de fondo (CoV), contraste
     miocardio/cavidad. Guarda un PNG comparativo.

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _evolution_harness.py

NO es código de producción: es un prototipo de banco de pruebas.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.raw_reconstruction import reconstruct_fbp_volume, reconstruct_projection_volume
from core.collimator_specs import lookup_collimator
from core.resolution_recovery import PsfModel, correct_axial_magnification

# --- Estudios crudos reales candidatos (primero que cargue, gana) ----------
CANDIDATES = [
    r"C:\Users\Ernesto\Desktop\INTERCAMBIO\varios stress cardiacos de xeleris\cardiac myosight\SGATE_01E1001_DS.dcm",
    r"C:\Users\Ernesto\Desktop\INTERCAMBIO\varios stress cardiacos de xeleris\motion correction\STR_EM_1001_DS.dcm",
]

OUT_PNG = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\output_demo\evolution_half_dose.png"
RNG = np.random.default_rng(20260421)

DEFAULT_RADIUS_MM = 250.0  # fallback cuando el DICOM no guarda el radio de órbita
DEFAULT_PIXEL_MM = 6.4     # fallback cuando falta pixel spacing


# =========================================================================
# Carga de datos + geometría física
# =========================================================================
def load_study() -> dict:
    """Carga el estudio y arma la config física (colimador, radio, pixel)."""
    from core.raw_projections import load_raw_projections, ungate_projections

    cli = [sys.argv[1]] if len(sys.argv) > 1 else []
    for p in cli + CANDIDATES:
        if os.path.isfile(p):
            try:
                raw = load_raw_projections(p)
                ung = ungate_projections(raw.projections)  # (ang,H,W)
                counts = np.rint(np.clip(ung, 0, None)).astype(np.int64)
                spec = lookup_collimator(raw.manufacturer, raw.collimator_name, raw.collimator_type)
                radius_mm = float(raw.radius_mm) if raw.radius_mm else DEFAULT_RADIUS_MM
                pixel_mm = float(raw.pixel_mm) if raw.pixel_mm else DEFAULT_PIXEL_MM
                label = f"REAL: {os.path.basename(p)} ({raw.n_gates}g x {raw.n_angles}a)"
                return {
                    "counts": counts,
                    "angles": np.asarray(raw.angles_deg, dtype=np.float64),
                    "label": label,
                    "spec": spec,
                    "radius_mm": radius_mm,
                    "radius_from_dicom": bool(raw.radius_mm),
                    "pixel_mm": pixel_mm,
                }
            except Exception as exc:  # noqa: BLE001
                print(f"  (no cargó {os.path.basename(p)}: {exc})")

    print("  Sin estudio real disponible -> proyecciones sintéticas.")
    from core.raw_projections import make_synthetic_raw_motion_projections

    syn = make_synthetic_raw_motion_projections()
    counts = np.rint(np.clip(syn.projections.sum(axis=0), 0, None)).astype(np.int64)
    return {
        "counts": counts,
        "angles": np.asarray(syn.angles_deg, dtype=np.float64),
        "label": "SINTÉTICO",
        "spec": lookup_collimator("GE", "LEHR"),
        "radius_mm": DEFAULT_RADIUS_MM,
        "radius_from_dicom": False,
        "pixel_mm": DEFAULT_PIXEL_MM,
    }


def pick_cardiac_slice(volume: np.ndarray, disk_frac: float = 0.33) -> int:
    """Elige el corte cardíaco: máxima actividad dentro de un disco central.

    El corazón se reconstruye cerca del eje de rotación (centro del FOV); el
    hígado/intestino quedan en la periferia. Un disco central de ``disk_frac``
    del FOV excluye esas fuentes extracardíacas y deja ganar al miocardio.
    """
    vol = np.asarray(volume, dtype=np.float64)
    nz, ny, nx = vol.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r = min(ny, nx) * float(disk_frac)
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    scores = np.zeros(nz, dtype=np.float64)
    for z in range(nz):
        v = vol[z][disk]
        scores[z] = float(np.percentile(v, 99.0)) if v.size else 0.0
    return int(np.argmax(scores))


# =========================================================================
# Métricas
# =========================================================================
def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    m = v.max()
    return v / m if m > 0 else v


def nrmse(a: np.ndarray, ref: np.ndarray) -> float:
    a, ref = normalize(a), normalize(ref)
    denom = ref.max() - ref.min() or 1.0
    return float(np.sqrt(np.mean((a - ref) ** 2)) / denom)


def background_noise(img: np.ndarray) -> float:
    """Ruido = desvío estándar del fondo (esquinas) normalizado por el pico.

    Robusto a valores negativos del FBP (undershoot de la rampa): el desvío es
    siempre positivo y crece con el ruido. Mayor = más ruidoso. En %.
    """
    g = normalize(img)
    n = g.shape[0]
    k = max(4, n // 6)
    corners = np.concatenate([
        g[:k, :k].ravel(), g[:k, -k:].ravel(),
        g[-k:, :k].ravel(), g[-k:, -k:].ravel(),
    ])
    return float(corners.std() * 100.0)


def myo_cavity_contrast(img: np.ndarray) -> float:
    """Contraste = pico miocárdico / centro (cavidad). Mayor = mejor definido."""
    g = normalize(img)
    peak = float(np.percentile(g, 99.5))
    n = g.shape[0]
    c = max(2, n // 12)
    cav = float(g[n // 2 - c:n // 2 + c, n // 2 - c:n // 2 + c].mean()) + 1e-6
    return peak / cav


# =========================================================================
# Main
# =========================================================================
def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    print("Cargando estudio...")
    study = load_study()
    counts = study["counts"]
    angles = study["angles"]
    spec = study["spec"]
    pixel_mm = study["pixel_mm"]
    radius_mm = study["radius_mm"]
    print(f"  {study['label']} | proj3d {counts.shape} | total cuentas {counts.sum():,}")
    print(f"  Colimador: {spec.manufacturer} {spec.name} [{spec.geometry}] | "
          f"pixel {pixel_mm:.2f} mm | radio {radius_mm:.0f} mm"
          f"{'' if study['radius_from_dicom'] else ' (fallback)'}")

    # --- Fan-beam: corrección axial si hay magnificación conocida ----------
    if spec.geometry == "fanbeam":
        if spec.axial_magnification:
            counts = np.rint(correct_axial_magnification(counts.astype(np.float64),
                                                         spec.axial_magnification)).astype(np.int64)
            print(f"  Fan-beam: corrección axial aplicada (M={spec.axial_magnification}).")
        else:
            print("  AVISO fan-beam: sin magnificación de datasheet -> se procesa como "
                  "paralelo (la geometría Y NO está corregida; resultado aproximado).")

    # --- Selección de corte por segmentación central ----------------------
    print("Buscando el corte cardíaco (FBP full + disco central)...")
    vol_full = reconstruct_fbp_volume(counts.astype(np.float64), angles)  # (H,W,W)
    best = pick_cardiac_slice(vol_full)
    H = counts.shape[1]
    r0 = max(0, min(best - 3, H - 6))
    r1 = min(H, r0 + 6)
    r0 = max(0, r1 - 6)
    mid = best - r0
    band_full = counts[:, r0:r1, :].astype(np.float64)
    print(f"  Corte cardíaco: {best} | banda filas {r0}:{r1} (cuentas banda {int(band_full.sum()):,})")

    # --- Dosis reducida por thinning binomial -----------------------------
    band_int = counts[:, r0:r1, :]
    band_half = RNG.binomial(band_int, 0.5).astype(np.float64)
    band_quarter = RNG.binomial(band_int, 0.25).astype(np.float64)
    print(f"  Mitad de dosis: {int(band_half.sum()):,} cuentas | Cuarto: {int(band_quarter.sum()):,}")

    # --- PSF dependiente de profundidad (auto desde colimador + DICOM) -----
    psf = PsfModel.from_collimator(spec, radius_mm=radius_mm, pixel_mm=pixel_mm)
    sig = psf.sigma_px_for_rows(counts.shape[2])
    print(f"  PSF: sigma {sig.min():.2f}–{sig.max():.2f} px "
          f"(FWHM sistema {spec.system_fwhm_mm(radius_mm):.1f} mm @ {radius_mm:.0f} mm)")

    print("Reconstruyendo (FBP ref, FBP half, OSEM half, OSEM+RR half)...")
    ref = reconstruct_fbp_volume(band_full, angles)[mid]
    fbp_half = reconstruct_fbp_volume(band_half, angles)[mid]
    osem_half = reconstruct_projection_volume(band_half, angles, method="osem",
                                              iterations=6, subsets=6)[mid]
    rr_half = reconstruct_projection_volume(band_half, angles, method="osem",
                                            iterations=6, subsets=6, psf=psf)[mid]
    fbp_quarter = reconstruct_fbp_volume(band_quarter, angles)[mid]

    panels = [
        ("FBP · dosis COMPLETA (ref)", ref),
        ("FBP · MITAD de dosis", fbp_half),
        ("OSEM · MITAD (sin RR)", osem_half),
        ("OSEM+RR · MITAD (Evolution-like)", rr_half),
        ("FBP · CUARTO de dosis", fbp_quarter),
    ]

    print("\n=== MÉTRICAS (NRMSE vs FBP dosis completa) ===")
    print(f"{'imagen':<34} {'NRMSE↓':>8} {'ruido%↓':>9} {'contraste↑':>11}")
    for name, img in panels:
        print(f"{name:<34} {nrmse(img, ref):8.3f} {background_noise(img):9.2f} {myo_cavity_contrast(img):11.2f}")

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.6))
    fig.patch.set_facecolor("#0b1220")
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(normalize(img), cmap="hot", vmin=0, vmax=1)
        ax.set_title(name, color="white", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Banco half-dose · {study['label']} · {spec.manufacturer} {spec.name}",
                 color="white", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, dpi=130, facecolor=fig.get_facecolor())
    print(f"\nGuardado: {OUT_PNG}")


if __name__ == "__main__":
    main()
