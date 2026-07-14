"""
Tests Fase 3: segmentación miocárdica + mapeo AHA 17 segmentos.

Correr:
    ./.venv/Scripts/python.exe tests/test_segmentation.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dicom_loader  # noqa: E402
from core.aha_segments import map_to_17_segments, phase_by_segment, territory_analysis  # noqa: E402
from core.console_utf8 import enable_utf8  # noqa: E402
from core.phase_analysis import phase_analysis  # noqa: E402
from core.segmentation import segment_myocardium  # noqa: E402

enable_utf8()

SA_GATED_PATH = (
    r"C:\Users\Ernesto\Desktop\INTERCAMBIO\varios stress cardiacos de xeleris"
    r"\estudio uno\MYOMETRIX\myometrix results\REST_IRNCG_SA001_DS.dcm"
)


def _make_ring_cube(
    n_gates: int = 8,
    n_slices: int = 10,
    H: int = 32,
    W: int = 32,
    cy: float = 16.0,
    cx: float = 16.0,
    r_inner: float = 4.0,
    r_outer: float = 8.0,
    amp: float = 80.0,
    dc: float = 120.0,
) -> np.ndarray:
    ys, xs = np.ogrid[:H, :W]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    ring = ((dist >= r_inner) & (dist <= r_outer)).astype(np.float64)

    cube = np.zeros((n_gates, n_slices, H, W), dtype=np.float64)
    for g in range(n_gates):
        val = dc + amp * np.cos(2.0 * np.pi * g / n_gates)
        cube[g] = ring * val
    return cube


def test_synthetic_segmentation_and_aha():
    cube = _make_ring_cube()

    seg = segment_myocardium(cube, method="auto")
    n_slices_non_empty = int(np.sum(seg.mask.reshape(seg.mask.shape[0], -1).any(axis=1)))
    assert n_slices_non_empty == seg.mask.shape[0], "todos los slices sintéticos deben tener anillo"

    centers = seg.center_per_slice
    assert np.all(np.isfinite(centers)), "centros deben ser finitos en sintético"
    assert np.all(np.abs(centers[:, 0] - 16.0) <= 1.5), f"cy fuera de tolerancia: {centers[:, 0]}"
    assert np.all(np.abs(centers[:, 1] - 16.0) <= 1.5), f"cx fuera de tolerancia: {centers[:, 1]}"

    aha = map_to_17_segments(seg)
    sm = aha.segment_map
    vals = sm[sm > 0]
    assert vals.size > 0, "segment_map debe tener voxels segmentados"
    assert int(vals.min()) >= 1 and int(vals.max()) <= 17, "segmentos fuera de rango 1..17"

    present = {i for i in range(1, 18) if aha.n_per_segment.get(i, 0) > 0}
    assert present == set(range(1, 18)), f"faltan segmentos en sintético: {sorted(set(range(1,18)) - present)}"

    assert sum(aha.n_per_segment.values()) == int(seg.mask.sum()), "n_per_segment debe sumar voxels de máscara"
    print("[OK] sintético: segmentación y mapeo AHA 1..17")


def test_real_smoke_if_available():
    if not os.path.exists(SA_GATED_PATH):
        print(f"[SKIP] no existe: {SA_GATED_PATH}")
        return

    study = dicom_loader.load(SA_GATED_PATH, verbose=False)
    seg = segment_myocardium(study.cube, method="auto")
    aha = map_to_17_segments(seg)

    res = phase_analysis(study.cube, seg.mask, harmonics=1, amplitude_threshold_frac=0.10)
    pbs = phase_by_segment(res.phase_map, aha)
    terr = territory_analysis(pbs)

    base_to_apex = list(reversed(aha.apex_to_base_order))
    L = len(base_to_apex)
    by_level = {"basal": 0, "medio": 0, "apical": 0, "apex": 0}
    for i, s in enumerate(base_to_apex):
        u = i / max(L - 1, 1)
        if u < 0.35:
            lvl = "basal"
        elif u < 0.70:
            lvl = "medio"
        elif u < 0.90:
            lvl = "apical"
        else:
            lvl = "apex"
        by_level[lvl] += int(np.sum((aha.segment_map[s] > 0) & seg.mask[s]))

    non_empty_segments = sum(1 for i in range(1, 18) if aha.n_per_segment.get(i, 0) > 0)
    print(f"[REAL] voxels por nivel: {by_level}")
    print(f"[REAL] segmentos no vacíos: {non_empty_segments}/17")
    print(f"[REAL] phase_by_segment: {pbs}")
    print(f"[REAL] territory_analysis: {terr}")

    assert non_empty_segments >= 15, f"esperaba >=15 segmentos no vacíos, hay {non_empty_segments}"
    for t in ("LAD", "LCx", "RCA"):
        assert int(terr[t]["n"]) > 0, f"territorio {t} sin datos"


def _run_all():
    test_synthetic_segmentation_and_aha()
    test_real_smoke_if_available()
    print("\n[TODOS LOS TESTS DE SEGMENTACIÓN PASARON]")


if __name__ == "__main__":
    _run_all()
