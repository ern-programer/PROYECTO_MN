"""Prototipo 'serie nítida' v1: feta axial + MRP-OSEM + supresión de fondo.

Reglas propias (nada licenciado):
  1. FETA (cilindro del corazón): recorta las proyecciones a la banda axial del
     VI, igual que Xeleris/Odyssey (el operador ubica la franja sobre el crudo
     ya corregido). En haz paralelo, fila de detector = corte axial, así que la
     feta = recortar filas H. Tira afuera hígado/GI de otros niveles → fondo negro.
  2. MRP-OSEM (Median Root Prior, one-step-late de Green): regularización que
     preserva bordes → vacía la cavidad del VI sin amplificar ruido. La mediana
     local no cruza bordes, así que el miocardio queda nítido y el fondo liso.
  3. Supresión de fondo: umbral suave sobre el volumen de la feta → el fondo
     residual (pool/scatter bajo) se va a negro como en Xeleris.

Compara: OSEM plano vs MRP-OSEM (ambos sobre la feta) vs Xeleris IRNCRR.

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _nitida_mrp.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.raw_projections import load_raw_projections, ungate_projections
from core.raw_reconstruction import (
    reconstruct_projection_volume, ProjectionFilterConfig,
    _forward_project_slice, _backproject_slice, _build_sensitivity_cache,
)
from core.collimator_specs import lookup_collimator
from core.resolution_recovery import PsfModel

GT_DIR = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
GATED = os.path.join(GT_DIR, "Stress-10sec-1_G_EM_1001_DS.dcm")
OUT = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\_xel3_out\nitida_mrp.png"


# =========================================================================
# 1. Localización del corazón por LATIDO (regla propia)
# =========================================================================
def find_heart_by_beat(gated_proj, angles, half=10):
    """Localiza el VI reconstruyendo la VARIANZA entre gates.

    El corazón es la única estructura que se mueve con el ciclo cardíaco; hígado,
    GI y fondo no laten. La varianza voxel-a-voxel entre gates, reconstruida,
    enciende solo el miocardio/pool → su centroide da (z,y,x) del VI y su
    extensión axial da la feta. Robusto frente a focos extracardíacos calientes.
    """
    g = np.clip(np.asarray(gated_proj, dtype=np.float64), 0.0, None)  # (gates,ang,H,W)
    var_sino = g.var(axis=0)  # (ang,H,W): "sinograma" de latido
    heat = reconstruct_projection_volume(
        var_sino, angles, method="fbp",
        projection_filter=ProjectionFilterConfig("butterworth", 0.45, 6),
        fbp_filter_name="ramp",
    )
    heat = gaussian_filter(np.clip(heat, 0.0, None), sigma=1.2, mode="constant")
    nz, ny, nx = heat.shape
    # Centroide ponderado por el mapa de latido, sobre el 15% más caliente.
    thr = np.percentile(heat, 96.0)
    mask = heat >= thr
    zz, yy, xx = np.nonzero(mask)
    w = heat[mask]
    z = int(round(np.average(zz, weights=w)))
    cy = int(round(np.average(yy, weights=w)))
    cx = int(round(np.average(xx, weights=w)))
    z0, z1 = max(0, z - half), min(nz, z + half)
    return z0, z1, cy, cx, heat


# =========================================================================
# 2. MRP-OSEM propio (one-step-late, median root prior)
# =========================================================================
def _prior_reference(img, prior, med_size, eps):
    """Referencia local M para el OSL: la penalización empuja img hacia M.

    - 'median': mediana local (v1). Suaviza pero puede lavar contraste pared/cavidad.
    - 'tv': variación total (Chambolle). Aplana zonas planas (cavidad/fondo) SIN
      cruzar bordes → vacía la cavidad y deja la pared nítida. Preferido.
    - 'bilateral': suaviza respetando bordes por rango de intensidad.
    """
    if prior == "median":
        return median_filter(img, size=med_size, mode="nearest")
    scale = float(img.max()) or 1.0
    norm = img / scale
    if prior == "tv":
        ref = denoise_tv_chambolle(norm, weight=0.06)
    elif prior == "bilateral":
        ref = denoise_bilateral(norm, sigma_color=0.08, sigma_spatial=1.5)
    else:
        return median_filter(img, size=med_size, mode="nearest")
    return np.clip(ref * scale, 0.0, None)


def mrp_osem_slab(slab_proj, theta, *, iterations, subsets, beta, psf, med_size=3, prior="median"):
    """OSEM con prior one-step-late (OSL de Green) sobre una feta (ang,H,W).

    Actualización por corte:
        x <- x * BP(y/FP(x)) / ( S * (1 + beta*(x - M)/M) )
    con M = referencia local edge-preserving (ver _prior_reference). beta=0 => OSEM.
    """
    proj = np.clip(np.asarray(slab_proj, dtype=np.float64), 0.0, None)
    n_ang, H, W = proj.shape
    theta = np.asarray(theta, dtype=np.float64)
    subset_count = max(1, min(int(subsets), n_ang))
    sens = _build_sensitivity_cache(theta, subsets=subset_count, detector_size=W, output_size=W, psf=psf)
    ang_idx = np.arange(n_ang)
    eps = 1e-6
    out = np.zeros((H, W, W), dtype=np.float64)
    for z in range(H):
        measured = proj[:, z, :].T  # (W, n_ang)
        if not np.any(measured > 0):
            continue
        img = np.full((W, W), max(float(measured.mean()), 1.0))
        for _it in range(iterations):
            for sid in range(subset_count):
                idx = ang_idx[sid::subset_count]
                th = theta[idx]
                est = _forward_project_slice(img, th, detector_size=W, psf=psf)
                ratio = measured[:, idx] / np.maximum(est, eps)
                corr = _backproject_slice(ratio, th, output_size=W, psf=psf)
                S = sens[sid]
                if beta > 0:
                    M = _prior_reference(img, prior, med_size, eps)
                    penal = 1.0 + beta * (img - M) / np.maximum(M, eps)
                    S = S * np.maximum(penal, 0.05)
                img *= corr / np.maximum(S, eps)
                img = np.clip(img, 0.0, None)
        out[z] = img
    return out


# =========================================================================
# 3. Supresión de fondo (umbral suave)
# =========================================================================
def suppress_background(vol, frac=0.12, soft=0.04):
    """Lleva a negro el fondo bajo un umbral (frac del max), con transición
    suave para no crear bordes duros artificiales."""
    v = np.asarray(vol, dtype=np.float64)
    vmax = float(np.percentile(v, 99.9)) or 1.0
    t = frac * vmax
    w = soft * vmax
    gate = np.clip((v - (t - w)) / (2 * w + 1e-9), 0.0, 1.0)
    return v * gate


def crop(v, cy, cx, w=13):
    return v[:, cy - w:cy + w, cx - w:cx + w]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    raw = load_raw_projections(GATED)
    ung = ungate_projections(raw.projections)
    angles = np.asarray(raw.angles_deg, dtype=np.float64)
    pixel_mm = float(raw.pixel_mm)
    spec = lookup_collimator(raw.manufacturer, raw.collimator_name, raw.collimator_type)
    psf = PsfModel.from_collimator(spec, radius_mm=float(raw.radius_mm or 250.0), pixel_mm=pixel_mm)

    z0, z1, cy, cx, heat = find_heart_by_beat(raw.projections, angles, half=10)
    print(f"feta axial (latido): z0={z0}..z1={z1} (={z1-z0} cortes)  centro VI=({cy},{cx})")
    slab = ung[:, z0:z1, :]  # LA FETA (ungated): suma de gates → pared latiente promediada
    slab_ed = raw.projections[0, :, z0:z1, :]  # LA FETA de UN gate (ED, gate 0)

    # (a) UNGATED (suma) — pared engrosada por promediar el latido
    mrp_ung = mrp_osem_slab(slab, angles, iterations=6, subsets=8, beta=0.3, psf=psf)
    mrp_ung_bg = suppress_background(mrp_ung, frac=0.14)
    # (b) GATE ÚNICO ED — sin promediar el latido (1/8 de cuentas → más ruido)
    mrp_ed = mrp_osem_slab(slab_ed, angles, iterations=6, subsets=8, beta=0.4, psf=psf)
    mrp_ed_bg = suppress_background(mrp_ed, frac=0.14)

    irncrr = pydicom.dcmread(os.path.join(GT_DIR, "STRESS_IRNCRR001_DS.dcm")).pixel_array.astype(np.float64)

    # OJO: recortamos filas (eje H=axial); el centro transaxial (cy,cx) no cambia con z0.
    imgs = [
        (crop(mrp_ung_bg, cy, cx), "UNGATED (suma)\nfeta+MRP+fondo"),
        (crop(mrp_ed_bg, cy, cx), "GATE ED (gate 0)\nfeta+MRP+fondo"),
        (irncrr[:, 21 - 13:21 + 13, 37 - 13:37 + 13], "Xeleris IRNCRR"),
    ]
    # Corte central de cada uno.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (vol, title) in zip(axes, imgs):
        zc = vol.shape[0] // 2
        im = vol[zc]
        ax.imshow(im, cmap="hot", vmin=0, vmax=float(np.percentile(im, 99.5)), interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle("Serie nítida — UNGATED vs GATE ED vs Xeleris (¿se vacía la cavidad?)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=135)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
