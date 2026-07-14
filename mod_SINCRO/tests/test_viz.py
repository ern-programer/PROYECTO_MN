"""Tests Fase 4: visualización (colormap, bullseye, histograma)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dicom_loader  # noqa: E402
from core.aha_segments import map_to_17_segments, phase_by_segment  # noqa: E402
from core.console_utf8 import enable_utf8  # noqa: E402
from core.phase_analysis import phase_analysis  # noqa: E402
from core.segmentation import segment_myocardium  # noqa: E402
from viz.colormaps import phase_to_rgb  # noqa: E402
from viz.histogram import build_phase_histogram, save_histogram  # noqa: E402
from viz.polar_map import build_polar_map, save_polar_map  # noqa: E402

enable_utf8()

SA_GATED_PATH = (
    r"C:\Users\Ernesto\Desktop\INTERCAMBIO\varios stress cardiacos de xeleris"
    r"\estudio uno\MYOMETRIX\myometrix results\REST_IRNCG_SA001_DS.dcm"
)


def test_phase_colormap_cyclic_and_nan():
    rgb = phase_to_rgb(np.array([0.0, 360.0]), cmap_name="hsv")
    assert np.allclose(rgb[0], rgb[1], atol=0.05), f"0° y 360° deberían verse igual: {rgb}"

    nan_color = (0.1, 0.1, 0.1)
    rgb_nan = phase_to_rgb(np.array([np.nan]), cmap_name="hsv", nan_color=nan_color)
    assert np.allclose(rgb_nan[0], np.array(nan_color), atol=1e-8)
    print("[OK] colormap cíclico y NaN")


def test_polar_map_synthetic_png():
    phase_by_seg = {i: float((i * 20) % 360) for i in range(1, 18)}
    pm = build_polar_map(phase_by_seg)
    assert getattr(pm, "fig", None) is not None
    assert len(pm.fig.axes) >= 1

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "polar_synth.png")
        save_polar_map(pm, out)
        assert os.path.exists(out), "No se creó PNG de polar map"
        assert os.path.getsize(out) > 0, "PNG de polar map vacío"
    print("[OK] polar map sintético")


def test_histogram_synthetic_png():
    phases = np.random.default_rng(0).normal(120, 30, 2000) % 360
    fig = build_phase_histogram(phases)
    assert fig is not None
    assert len(fig.axes) >= 1

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "hist_synth.png")
        save_histogram(fig, out)
        assert os.path.exists(out), "No se creó PNG de histograma"
        assert os.path.getsize(out) > 0, "PNG de histograma vacío"
    print("[OK] histograma sintético")


def test_real_integration_skip_safe():
    if not os.path.exists(SA_GATED_PATH):
        print(f"[SKIP] no existe: {SA_GATED_PATH}")
        return

    study = dicom_loader.load(SA_GATED_PATH, verbose=False)
    seg = segment_myocardium(study.cube, method="auto")
    res = phase_analysis(study.cube, seg.mask, harmonics=1, amplitude_threshold_frac=0.10)
    aha = map_to_17_segments(seg)
    pbs = phase_by_segment(res.phase_map, aha)

    pm = build_polar_map(pbs)
    hf = build_phase_histogram(res.phases_deg)

    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "polar_real.png")
        p2 = os.path.join(td, "hist_real.png")
        save_polar_map(pm, p1)
        save_histogram(hf, p2)
        assert os.path.exists(p1) and os.path.getsize(p1) > 0
        assert os.path.exists(p2) and os.path.getsize(p2) > 0

    print("[OK] integración real viz")


def _run_all():
    test_phase_colormap_cyclic_and_nan()
    test_polar_map_synthetic_png()
    test_histogram_synthetic_png()
    test_real_integration_skip_safe()
    print("\n[TODOS LOS TESTS DE VIZ PASARON]")


if __name__ == "__main__":
    _run_all()
