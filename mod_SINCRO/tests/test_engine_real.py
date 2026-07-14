"""
Validación end-to-end del motor de fase sobre un estudio REAL (Xeleris SA gated).

Como la segmentación fina es Fase 3, aquí se usa una máscara PROVISIONAL por umbral
(voxels con actividad media > fracción del máximo) solo para verificar que el motor
corre de punta a punta y produce métricas plausibles sobre datos reales.

NO es validación clínica (eso requiere SyncTool). Es un smoke-test de integración.

Correr:  python tests/test_engine_real.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dicom_loader
from core.phase_analysis import phase_analysis
from core.metrics import calculate_phase_metrics
from core.segmentation import segment_myocardium
from core.console_utf8 import enable_utf8

enable_utf8()

SA_GATED_PATH = (
    r"C:\Users\Ernesto\Desktop\INTERCAMBIO\varios stress cardiacos de xeleris"
    r"\estudio uno\MYOMETRIX\myometrix results\REST_IRNCG_SA001_DS.dcm"
)


def provisional_mask(cube, frac=0.35):
    """Máscara provisional: voxels cuya actividad media supera frac*max (Fase 3 hará LVSD)."""
    mean_img = cube.mean(axis=0)                 # (n_slices, H, W)
    thr = frac * mean_img.max()
    return mean_img > thr


def main():
    if not os.path.exists(SA_GATED_PATH):
        print(f"[SKIP] no existe: {SA_GATED_PATH}")
        return 0
    study = dicom_loader.load(SA_GATED_PATH, verbose=True)
    seg = segment_myocardium(study.cube, method="auto", threshold_frac=0.35)
    mask = seg.mask
    if int(mask.sum()) == 0:
        print("\n[WARN] segmentación auto vacía. Fallback a máscara provisional por umbral.")
        mask = provisional_mask(study.cube, frac=0.35)
    print(f"\nMáscara usada: {int(mask.sum())} voxels (de {mask.size}).")

    res = phase_analysis(study.cube, mask, harmonics=1, amplitude_threshold_frac=0.10)
    print(f"Voxels analizados: {res.n_voxels_kept}/{res.n_voxels_total} "
          f"(tras filtro de amplitud).")

    metrics = calculate_phase_metrics(res.phases_deg)
    print("\n=== MÉTRICAS DE FASE (estudio real, máscara provisional) ===")
    for k, v in metrics.items():
        print(f"  {k:24s}: {v}")

    print("\nNOTA: máscara provisional por umbral (no LVSD). Números orientativos, "
          "NO validación clínica. Comparar vs SyncTool cuando haya acceso (Nivel 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
