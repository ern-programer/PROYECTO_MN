"""Validación clínica NITIDA II: ¿preserva FEVI y engrosamiento en espacio SA?

NO es producto: script de auditoría para probar, sobre datos reales 5s/10s, que
el denoiser NITIDA II (temporal / espaciotemporal) aplicado post-recon al gated
de bajo conteo NO altera la función ventricular medida (FEVI, volúmenes,
engrosamiento), que es el criterio clínico duro. El movimiento cardíaco vive en
las bandas de baja frecuencia (DC..H2); NITIDA II conserva esas bandas y limpia
el ruido de banda alta. Si la física es correcta, la FEVI del 5s+NITIDA debe
quedar igual a la del 5s crudo (y ambas cerca del 10s oro).

Método (calca la geometría del pipeline real):
  1. Reconstruye el GATED completo (8 gates) por FBP con el mismo filtro de
     proyección gated del pipeline (Butterworth 0.40/orden 10) para 10s y 5s.
  2. Aplica NITIDA II sobre el volumen transaxial gated ANTES de reorientar,
     exactamente donde lo hace reconstruct_raw_gated_pipeline (post-recon, solo
     gated). Modos: temporal (keep DC..H2) y espaciotemporal (además suaviza).
  3. Deriva UNA reorientación (centro + eje largo) del 10s FBP de alta
     estadística con auto_orient_lv y la REUSA en todas las columnas, así el
     único cambio entre columnas es el denoiser, no la geometría.
  4. Reslicea a eje corto, segmenta (auto) y corre ECTb -> EDV/ESV/FEVI +
     engrosamiento de pared.

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _nitida2_fevi.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cardiac_reorientation import auto_orient_lv, reslice_from_vector_gated
from core.ectb_lv import ECTbLVConfig, analyze_lv_ectb
from core.nitida2 import denoise_spatiotemporal, temporal_harmonic_filter
from core.raw_projections import load_raw_projections
from core.raw_reconstruction import ProjectionFilterConfig, reconstruct_gated_fbp_volume
from core.segmentation import segment_myocardium

P5 = r"D:\- GAMMASYS\estudios evolution\New Folder\1.2.840.113619.2.265.1.2.0.9092025113215781.32146\SGATE5seg_G_1001_DS.dcm"
P10 = r"D:\- GAMMASYS\estudios evolution\New Folder\1.2.840.113619.2.265.1.2.0.9092025113215781.32146\SGATE10seg_G_1001_DS.dcm"

OUT_SIZE = 40  # lado del cubo SA reorientado

# Parámetros NITIDA II (defaults validados: keep=2 conserva el movimiento exacto,
# band_sigma=0.7 no sobre-suaviza).
N_HARM = 2
BAND_SIGMA = 0.7


def recon_gated_fbp(raw, pf, label):
    t0 = time.time()
    cube = reconstruct_gated_fbp_volume(raw.projections, raw.angles_deg, projection_filter=pf)
    print(f"  [{label}] FBP gated -> {cube.shape} en {time.time()-t0:.0f}s")
    return cube


def fevi_from_gated(cube_gated, center, long_axis, pixel_mm):
    """Reorienta a SA, segmenta auto y corre ECTb. Devuelve el ECTbLVResult."""
    sa = reslice_from_vector_gated(cube_gated, center, long_axis, OUT_SIZE, order=1)
    seg = segment_myocardium(sa, method="auto")
    return analyze_lv_ectb(sa, seg, (pixel_mm, pixel_mm), pixel_mm, ECTbLVConfig())


def show(tag, res, ref=None):
    if not getattr(res, "available", False):
        print(f"  {tag:<42} NO disponible: {res.reason}")
        return None
    d = ""
    if ref is not None and getattr(ref, "available", False):
        d = (f"  (ΔFEVI={res.ef_pct-ref.ef_pct:+.1f}  "
             f"ΔThk={res.thickening_pct-ref.thickening_pct:+.1f})")
    print(f"  {tag:<42} FEVI={res.ef_pct:5.1f}%  EDV={res.edv_ml:6.1f}  ESV={res.esv_ml:6.1f}  "
          f"SV={res.sv_ml:6.1f}  Thk={res.thickening_pct:5.1f}%  "
          f"(ED={res.wall_thickness_ed_mm:.1f}→ES={res.wall_thickness_es_mm:.1f}mm){d}")
    return res


def main():
    print("Cargando estudios...")
    raw10 = load_raw_projections(P10)
    raw5 = load_raw_projections(P5)
    c10, c5 = float(raw10.projections.sum()), float(raw5.projections.sum())
    print(f"  10s counts={c10:,.0f}  5s counts={c5:,.0f}  ratio={c5/c10:.3f}")
    px = float(raw5.pixel_spacing[0]) if getattr(raw5, "pixel_spacing", None) else 6.4
    print(f"  pixel_mm={px:.2f}\n")

    pf_gated = ProjectionFilterConfig("butterworth", 0.40, 10)

    print("Reconstruyendo volúmenes gated (pesado)...")
    cube10 = recon_gated_fbp(raw10, pf_gated, "10s oro")
    cube5 = recon_gated_fbp(raw5, pf_gated, "5s crudo")

    # NITIDA II sobre el transaxial gated, igual que el pipeline (post-recon).
    print("Aplicando NITIDA II al 5s (transaxial, post-recon)...")
    cube5_temp = temporal_harmonic_filter(cube5, n_harmonics=N_HARM, axis=0)
    cube5_st = denoise_spatiotemporal(cube5, n_harmonics=N_HARM, band_sigma=BAND_SIGMA)

    # Reorientación única derivada del 10s oro (alta estadística), reusada.
    print("\nReorientación automática (derivada del 10s oro, reusada en todas)...")
    ung10 = cube10.mean(axis=0)
    orient = auto_orient_lv(cube10, ung10)
    if orient is None:
        print("  [ERROR] auto_orient_lv no pudo detectar el VI. Abortando.")
        return 1
    center, long_axis = orient["center"], orient["long_axis"]
    print(f"  center(z,y,x)={tuple(round(v,1) for v in center)}  "
          f"long_axis={tuple(round(float(v),3) for v in long_axis)}")

    print("\n=== FEVI / volúmenes / engrosamiento (ECTb, misma reorientación) ===")
    ref = show("10s FBP (ORO/referencia)", fevi_from_gated(cube10, center, long_axis, px))
    print("  " + "-" * 78)
    crudo5 = fevi_from_gated(cube5, center, long_axis, px)
    show("5s crudo (baseline sin denoiser)", crudo5, ref)
    show("5s + NITIDA II temporal", fevi_from_gated(cube5_temp, center, long_axis, px), crudo5)
    show("5s + NITIDA II espaciotemporal", fevi_from_gated(cube5_st, center, long_axis, px), crudo5)

    print("\nInterpretación:")
    print("  - ΔFEVI/ΔThk vs '5s crudo' cercanos a 0 => el denoiser NO altera la función.")
    print("  - Si el 5s crudo ya difiere del 10s oro, es ruido de bajo conteo, no NITIDA II.")
    print("\nListo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
