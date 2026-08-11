"""Diagnóstico NÍTIDA: 5s vs 10s. NO es parte del producto (script de auditoría)."""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.raw_projections import load_raw_projections, ungate_projections
from core.raw_reconstruction import (
    ProjectionFilterConfig,
    reconstruct_projection_volume,
)
from core.collimator_specs import lookup_collimator
from core.resolution_recovery import PsfModel

P5 = r"D:\- GAMMASYS\estudios evolution\New Folder\1.2.840.113619.2.265.1.2.0.9092025113215781.32146\SGATE5seg_G_1001_DS.dcm"
P10 = r"D:\- GAMMASYS\estudios evolution\New Folder\1.2.840.113619.2.265.1.2.0.9092025113215781.32146\SGATE10seg_G_1001_DS.dcm"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_nitida_audit_out")
os.makedirs(OUT, exist_ok=True)


def build_psf(raw):
    spec = lookup_collimator(raw.manufacturer or "", raw.collimator_name or "", raw.collimator_type or "")
    if spec is None:
        print(f"  [PSF] colimador NO identificado (manuf={raw.manufacturer!r} col={raw.collimator_name!r}); fallback LEHR")
        from core.collimator_specs import CollimatorSpec
        spec = CollimatorSpec(name="LEHR-fallback", manufacturer="generic", geometry="parallel",
                              hole_diameter_mm=1.5, hole_length_mm=35.0, septal_mm=0.2,
                              intrinsic_fwhm_mm=4.0, focal_length_mm=None, axial_magnification=1.0)
    pixel_mm = float(raw.pixel_spacing[0]) if getattr(raw, "pixel_spacing", None) else 6.4
    radius_mm = float(raw.radius_mm) if getattr(raw, "radius_mm", None) else 250.0
    psf = PsfModel.from_collimator(spec, radius_mm=radius_mm, pixel_mm=pixel_mm)
    print(f"  [PSF] {spec.manufacturer} {spec.name} radio={radius_mm:.0f}mm pixel={pixel_mm:.2f}mm FWHMint={spec.intrinsic_fwhm_mm}")
    return psf, pixel_mm


def recon_ung(raw, method, pf, psf, iters, subs):
    proj3d = ungate_projections(raw.projections)  # (angles,H,W)
    t0 = time.time()
    vol = reconstruct_projection_volume(
        proj3d, raw.angles_deg, method=method, projection_filter=pf,
        iterations=iters, subsets=subs, psf=psf,
    )
    print(f"  recon {method} iter={iters} sub={subs} psf={'Y' if psf is not None else 'N'} -> {vol.shape} en {time.time()-t0:.1f}s")
    return vol


def norm(img):
    v = np.asarray(img, dtype=np.float64)
    hi = np.percentile(v, 99.5)
    return np.clip(v / hi, 0, 1) if hi > 0 else v


def main():
    for label, p in (("10s", P10), ("5s", P5)):
        raw = load_raw_projections(p)
        tot = float(raw.projections.sum())
        print(f"[{label}] shape={raw.projections.shape} counts_totales={tot:,.0f} "
              f"manuf={raw.manufacturer!r} col={raw.collimator_name!r} radio={getattr(raw,'radius_mm',None)} "
              f"pixel={getattr(raw,'pixel_spacing',None)}")

    raw10 = load_raw_projections(P10)
    raw5 = load_raw_projections(P5)
    c10, c5 = raw10.projections.sum(), raw5.projections.sum()
    print(f"\nRatio cuentas 5s/10s = {c5/c10:.3f}\n")

    psf5, px5 = build_psf(raw5)
    fbp_ung = ProjectionFilterConfig("butterworth", 0.52, 5)

    print("Reconstruyendo...")
    vol10_fbp = recon_ung(raw10, "fbp", fbp_ung, None, 4, 4)
    vol5_fbp = recon_ung(raw5, "fbp", fbp_ung, None, 4, 4)
    vol5_nit5 = recon_ung(raw5, "osem", None, psf5, 5, 4)   # NÍTIDA actual (iter5 x sub4, sin post)
    vol5_nit2 = recon_ung(raw5, "osem", None, psf5, 2, 4)   # NÍTIDA menos iteraciones
    psf10, px10 = build_psf(raw10)
    vol10_nit2 = recon_ung(raw10, "osem", None, psf10, 2, 4)  # 10s NÍTIDA (objetivo)
    vol10_nit5 = recon_ung(raw10, "osem", None, psf10, 5, 4)

    def sig(fwhm_mm):
        return (fwhm_mm / 2.354820045) / max(px5, 1e-6)

    # Corte axial representativo (medio)
    z = vol10_fbp.shape[0] // 2

    panels = [
        ("10s FBP (ref)", vol10_fbp[z]),
        ("5s FBP", vol5_fbp[z]),
        ("5s NITIDA iter5 (actual, sin post)", vol5_nit5[z]),
        ("5s NITIDA iter5 + post 8mm", gaussian_filter(vol5_nit5, sig(8.0))[z]),
        ("5s NITIDA iter5 + post 10mm", gaussian_filter(vol5_nit5, sig(10.0))[z]),
        ("5s NITIDA iter5 + post 12mm", gaussian_filter(vol5_nit5, sig(12.0))[z]),
        ("5s NITIDA iter2 + post 8mm", gaussian_filter(vol5_nit2, sig(8.0))[z]),
        ("5s NITIDA iter2 + post 10mm", gaussian_filter(vol5_nit2, sig(10.0))[z]),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (title, img) in zip(axs.ravel(), panels):
        ax.imshow(norm(img), cmap="hot", interpolation="bilinear")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"NITIDA audit - UngGat corte z={z} | cuentas 5s/10s={c5/c10:.2f}", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT, "compare_slice.png")
    fig.savefig(out, dpi=110)
    print(f"\nGuardado: {out}")

    # --- Figura 2: objetivo (10s NÍTIDA) vs 5s actual vs 5s propuesto ---
    def sig10(fwhm_mm):
        return (fwhm_mm / 2.354820045) / max(px10, 1e-6)
    panels2 = [
        ("10s NITIDA iter5 (sin post)", vol10_nit5[z]),
        ("10s NITIDA iter2 + post 8mm (objetivo)", gaussian_filter(vol10_nit2, sig10(8.0))[z]),
        ("5s NITIDA iter5 sin post (ACTUAL)", vol5_nit5[z]),
        ("5s NITIDA iter2 + post 8mm (propuesto)", gaussian_filter(vol5_nit2, sig(8.0))[z]),
    ]
    fig2, axs2 = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (title, img) in zip(axs2.ravel(), panels2):
        ax.imshow(norm(img), cmap="hot", interpolation="bilinear")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig2.suptitle("NITIDA: objetivo 10s vs 5s actual vs 5s propuesto", fontsize=12)
    fig2.tight_layout()
    out2 = os.path.join(OUT, "compare_target.png")
    fig2.savefig(out2, dpi=120)
    print(f"Guardado: {out2}")

    # Métricas de ruido (coef. variación en un ROI de fondo homogéneo) por panel
    print("\nRuido (CoV) en ventana central 20x20:")
    h = vol10_fbp.shape[1]
    a, b = h//2 - 10, h//2 + 10
    for title, img in panels:
        roi = np.asarray(img)[a:b, a:b]
        m = roi.mean()
        cov = roi.std() / m if m > 0 else 0
        print(f"  {title:42s} CoV={cov:.3f}")


if __name__ == "__main__":
    main()
