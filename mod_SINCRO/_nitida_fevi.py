"""Validación FEVI: 10s FBP (oro) vs 5s NÍTIDA con receta iter2 + post-filtro.

NO es producto: script de auditoría para decidir si el post-filtro sesga la
FEVI/volúmenes antes de cambiar los defaults de NÍTIDA.

Método:
  1. Reconstruye el volumen GATED completo (8 gates) para cada config.
  2. Deriva una reorientación única (centro + eje largo) del 10s de alta
     estadística con auto_orient_lv y la REUSA en todas las configs (mismo
     corazón, misma adquisición 5s/10s), así el único cambio entre columnas es
     la reconstrucción/suavizado, no la geometría.
  3. Reslicea a eje corto, segmenta (auto) y corre ECTb -> EDV/ESV/FEVI.
  4. El post-filtro (gaussiano por gate) se aplica al volumen transaxial ANTES
     de reorientar, igual que en reconstruct_raw_gated_pipeline.

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _nitida_fevi.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.raw_projections import load_raw_projections
from core.raw_reconstruction import (
    ProjectionFilterConfig,
    reconstruct_gated_fbp_volume,
    reconstruct_gated_projection_volume,
)
from core.cardiac_reorientation import auto_orient_lv, reslice_from_vector_gated
from core.segmentation import segment_myocardium
from core.ectb_lv import analyze_lv_ectb, ECTbLVConfig
from core.collimator_specs import lookup_collimator, CollimatorSpec
from core.resolution_recovery import PsfModel

P5 = r"D:\- GAMMASYS\estudios evolution\New Folder\1.2.840.113619.2.265.1.2.0.9092025113215781.32146\SGATE5seg_G_1001_DS.dcm"
P10 = r"D:\- GAMMASYS\estudios evolution\New Folder\1.2.840.113619.2.265.1.2.0.9092025113215781.32146\SGATE10seg_G_1001_DS.dcm"

OUT_SIZE = 40          # lado del cubo SA reorientado
FWHM_TO_SIGMA = 2.354820045


def build_psf(raw):
    spec = lookup_collimator(raw.manufacturer or "", raw.collimator_name or "", raw.collimator_type or "")
    if spec is None:
        spec = CollimatorSpec(name="LEHR-fallback", manufacturer="generic", geometry="parallel",
                              hole_diameter_mm=1.5, hole_length_mm=35.0, septal_mm=0.2,
                              intrinsic_fwhm_mm=4.0, focal_length_mm=None, axial_magnification=1.0)
    pixel_mm = float(raw.pixel_spacing[0]) if getattr(raw, "pixel_spacing", None) else 6.4
    radius_mm = float(raw.radius_mm) if getattr(raw, "radius_mm", None) else 250.0
    psf = PsfModel.from_collimator(spec, radius_mm=radius_mm, pixel_mm=pixel_mm)
    return psf, pixel_mm


def recon_gated(raw, method, pf, psf, iters, subs, label):
    t0 = time.time()
    cube = reconstruct_gated_projection_volume(
        raw.projections, raw.angles_deg, method=method, projection_filter=pf,
        iterations=iters, subsets=subs, psf=psf,
    )
    print(f"  [{label}] {method} iter={iters} sub={subs} psf={'Y' if psf is not None else 'N'} "
          f"-> {cube.shape} en {time.time()-t0:.0f}s")
    return cube


def recon_gated_fbp(raw, pf, label):
    t0 = time.time()
    cube = reconstruct_gated_fbp_volume(raw.projections, raw.angles_deg, projection_filter=pf)
    print(f"  [{label}] FBP -> {cube.shape} en {time.time()-t0:.0f}s")
    return cube


def post_filter(cube_gated, fwhm_mm, pixel_mm):
    """Gaussiano 3D por gate (no mezcla gates). Igual criterio que el pipeline."""
    if fwhm_mm <= 0.0:
        return cube_gated
    sigma = (fwhm_mm / FWHM_TO_SIGMA) / max(pixel_mm, 1e-6)
    return np.stack([gaussian_filter(cube_gated[g], sigma) for g in range(cube_gated.shape[0])], axis=0)


def fevi_from_gated(cube_gated, center, long_axis, pixel_mm):
    """Reorienta a SA, segmenta auto y corre ECTb. Devuelve el ECTbLVResult."""
    sa = reslice_from_vector_gated(cube_gated, center, long_axis, OUT_SIZE, order=1)
    seg = segment_myocardium(sa, method="auto")
    res = analyze_lv_ectb(sa, seg, (pixel_mm, pixel_mm), pixel_mm, ECTbLVConfig())
    return res


def show(tag, res, ref_ef=None):
    if not getattr(res, "available", False):
        print(f"  {tag:<40} NO disponible: {res.reason}")
        return None
    d = f"  (Δ={res.ef_pct-ref_ef:+.1f})" if ref_ef is not None else ""
    print(f"  {tag:<40} FEVI={res.ef_pct:5.1f}%  EDV={res.edv_ml:6.1f}  "
          f"ESV={res.esv_ml:6.1f}  SV={res.sv_ml:6.1f}  ED_g={res.ed_gate} ES_g={res.es_gate}{d}")
    return res.ef_pct


def main():
    print("Cargando estudios...")
    raw10 = load_raw_projections(P10)
    raw5 = load_raw_projections(P5)
    c10, c5 = float(raw10.projections.sum()), float(raw5.projections.sum())
    print(f"  10s counts={c10:,.0f}  5s counts={c5:,.0f}  ratio={c5/c10:.3f}")

    psf5, px5 = build_psf(raw5)
    psf10, px10 = build_psf(raw10)
    print(f"  pixel_mm 10s={px10:.2f}  5s={px5:.2f}\n")

    # Filtros de proyección igual que el pipeline gated (Butterworth 0.40/10)
    pf_gated = ProjectionFilterConfig("butterworth", 0.40, 10)

    print("Reconstruyendo volúmenes gated (pesado)...")
    cube10_fbp = recon_gated_fbp(raw10, pf_gated, "10s FBP oro")
    cube10_nit2 = recon_gated(raw10, "osem", None, psf10, 2, 4, "10s NIT iter2")
    cube5_nit2 = recon_gated(raw5, "osem", None, psf5, 2, 4, "5s NIT iter2")
    cube5_nit5 = recon_gated(raw5, "osem", None, psf5, 5, 4, "5s NIT iter5 (default actual)")

    # Reorientación única derivada del 10s FBP (alta estadística) -> se reusa.
    print("\nReorientación automática (derivada del 10s FBP, reusada en todas)...")
    ung10 = cube10_fbp.mean(axis=0)
    orient = auto_orient_lv(cube10_fbp, ung10)
    if orient is None:
        print("  [ERROR] auto_orient_lv no pudo detectar el VI. Abortando.")
        return 1
    center = orient["center"]
    long_axis = orient["long_axis"]
    print(f"  center(z,y,x)={tuple(round(v,1) for v in center)}  "
          f"long_axis={tuple(round(float(v),3) for v in long_axis)}")

    print("\n=== FEVI / volúmenes (ECTb) ===")
    ref = fevi_from_gated(cube10_fbp, center, long_axis, px10)
    ref_ef = show("10s FBP (ORO/referencia)", ref)
    show("10s NITIDA iter2 sin post", fevi_from_gated(cube10_nit2, center, long_axis, px10), ref_ef)
    show("10s NITIDA iter2 + post 8mm", fevi_from_gated(post_filter(cube10_nit2, 8.0, px10), center, long_axis, px10), ref_ef)
    print("  " + "-" * 70)
    show("5s NITIDA iter5 sin post (DEFAULT ACTUAL)", fevi_from_gated(cube5_nit5, center, long_axis, px5), ref_ef)
    show("5s NITIDA iter2 sin post", fevi_from_gated(cube5_nit2, center, long_axis, px5), ref_ef)
    show("5s NITIDA iter2 + post 6mm", fevi_from_gated(post_filter(cube5_nit2, 6.0, px5), center, long_axis, px5), ref_ef)
    show("5s NITIDA iter2 + post 8mm (PROPUESTO)", fevi_from_gated(post_filter(cube5_nit2, 8.0, px5), center, long_axis, px5), ref_ef)
    show("5s NITIDA iter2 + post 10mm", fevi_from_gated(post_filter(cube5_nit2, 10.0, px5), center, long_axis, px5), ref_ef)
    print("\nListo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
