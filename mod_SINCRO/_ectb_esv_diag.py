"""Diagnóstico ECTb: por qué el ESV colapsa a ~2 mL / FEVI ~96% en el 5s.

NO es producto. Corre ECTb sobre el CUBO SA ya reorientado (el que se exporta
desde la app con "Guardar ejes DICOM") y vuelca la geometría interna del método
para ubicar el colapso de la cavidad en sístole:

  - EDV/ESV/FEVI y curva de volumen por gate.
  - center_r (radio del centro de pared por máximo de cuentas) en ED vs ES.
  - espesor de pared por gate y ratio ES/ED (efecto del engrosamiento).
  - endocardio: mínimo, media y FRACCIÓN de radios que quedaron en 0
    (clip de `center - 5mm`), que es el driver sospechado del colapso.
  - barridos de `ed_wall_thickness_mm` y `use_thickening` para ver cuánto
    mueven el ESV/EF (elegir el fix con datos, no a ciegas).

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _ectb_esv_diag.py "RUTA\\AL\\SA_CUTS.dcm"
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import dicom_loader
from core.segmentation import segment_myocardium
from core.ectb_lv import analyze_lv_ectb, ECTbLVConfig


def _fmt(v: float, n: int = 1) -> str:
    return f"{v:.{n}f}"


def run(path: str) -> None:
    print(f"[LOAD] {path}")
    study = dicom_loader.load(path)
    cube = np.asarray(study.cube, dtype=np.float64)
    print(f"[CUBE] shape={cube.shape} (gates,slices,H,W)")
    px = getattr(study, "pixel_spacing", None)
    z_mm = getattr(study, "z_spacing_mm", None)
    print(f"[SPACING] pixel_spacing={px} z_spacing_mm={z_mm}")
    if not px or z_mm is None:
        print("[WARN] Sin spacing DICOM -> ECTb usaría fallback; verificá el estudio.")
        return

    n_gates = int(cube.shape[0])

    # Segmentación auto sobre el cubo SA (ya recortado al corazón; el intestino
    # no domina como en el volumen transaxial completo).
    seg = segment_myocardium(cube, method="auto")
    print(f"[SEG] method={getattr(seg, 'method', '?')} n_voxels={int(np.count_nonzero(seg.mask))}")

    pixel_mm = (float(px[0]), float(px[1]))
    slice_mm = float(z_mm)

    def analyze(cfg: ECTbLVConfig, label: str):
        res = analyze_lv_ectb(cube, seg, pixel_mm, slice_mm, cfg)
        if not res.available:
            print(f"[{label}] NO DISPONIBLE: {res.reason}")
            return None
        print(
            f"[{label}] EDV={_fmt(res.edv_ml)} ESV={_fmt(res.esv_ml)} SV={_fmt(res.sv_ml)} mL "
            f"EF={_fmt(res.ef_pct)}% | ED g{res.ed_gate} ES g{res.es_gate} | "
            f"thk_ED={_fmt(res.wall_thickness_ed_mm)} thk_ES={_fmt(res.wall_thickness_es_mm)} mm "
            f"thickening={_fmt(res.thickening_pct)}%"
        )
        vols = ", ".join(_fmt(v) for v in np.asarray(res.gate_volumes_ml))
        print(f"        gate_volumes_ml=[{vols}]")
        return res

    print("\n=== BASELINE (config default de la app) ===")
    base = analyze(ECTbLVConfig(), "DEFAULT")
    if base is None:
        return

    # Introspección de la geometría en ED vs ES.
    endo = np.asarray(base.endo_radii_mm)      # (n_gates, n_valid, n_ang)
    center = np.asarray(base.center_radii_mm)
    ed = base.ed_gate - 1
    es = base.es_gate - 1
    print("\n=== GEOMETRÍA ED vs ES ===")
    for name, g in (("ED", ed), ("ES", es)):
        e = endo[g]
        c = center[g]
        zero_frac = float(np.mean(e <= 1e-6)) * 100.0
        print(
            f"[{name} g{g+1}] center_r mm: min={_fmt(c.min())} mean={_fmt(c.mean())} max={_fmt(c.max())} | "
            f"endo mm: min={_fmt(e.min())} mean={_fmt(e.mean())} max={_fmt(e.max())} | "
            f"endo==0: {_fmt(zero_frac)}% de los radios"
        )
    print(
        "\n[INTERP] Si en ES 'endo==0' es alto y center_r_ES ~ mitad del espesor "
        "(~5 mm), el colapso viene de restar la pared fija a un center_r chico."
    )

    print("\n=== BARRIDO ed_wall_thickness_mm (con thickening ON) ===")
    for thk in (10.0, 8.0, 6.0, 4.0):
        analyze(ECTbLVConfig(ed_wall_thickness_mm=thk), f"thk={thk:.0f}mm")

    print("\n=== EFECTO DEL ENGROSAMIENTO (wall fijo 10mm) ===")
    analyze(ECTbLVConfig(use_thickening=False), "no_thickening")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Uso: python _ectb_esv_diag.py "RUTA\\AL\\SA_CUTS.dcm"')
        print("Exportá el cubo SA desde la app con el botón \"Guardar ejes DICOM\".")
        sys.exit(2)
    run(sys.argv[1])
