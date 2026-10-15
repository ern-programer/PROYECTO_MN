import numpy as np
import pytest

from core.perfusion_quant import auto_segment_score, perfusion_scores, score_summary_text


@pytest.mark.parametrize("pct,expected", [
    (100.0, 0), (70.0, 0), (69.9, 1), (60.0, 1), (59.9, 2), (50.0, 2),
    (49.9, 3), (40.0, 3), (39.9, 4), (0.0, 4), (float("nan"), None), (None, None),
])
def test_auto_score_thresholds(pct, expected):
    assert auto_segment_score(pct) == expected


def _pct(value=100.0, **segments):
    out = {seg: value for seg in range(1, 18)}
    out.update({int(key[1:]): val for key, val in segments.items()})
    return out


def test_scores_sum_stress_rest_and_difference():
    res = perfusion_scores(_pct(s1=35.0, s7=55.0), _pct(s1=65.0))
    assert res["sss"] == 6 and res["srs"] == 1 and res["sds"] == 5
    assert res["sss_pct"] == pytest.approx(6 / 68 * 100)
    assert res["n_edited"] == 0
    assert res["rows"][0]["stress_auto"] == 4 and res["rows"][0]["rest_auto"] == 1


def test_manual_override_replaces_auto_and_marks_only_real_changes():
    overrides = {"stress": {1: 2, 2: 0}, "rest": {1: 0}}
    res = perfusion_scores(_pct(s1=35.0), _pct(s1=65.0), overrides)
    row = res["rows"][0]
    assert row["stress_auto"] == 4 and row["stress_score"] == 2 and row["stress_edited"]
    assert res["rows"][1]["stress_score"] == 0 and not res["rows"][1]["stress_edited"]
    assert row["rest_score"] == 0 and row["rest_edited"]
    assert (res["sss"], res["srs"], res["sds"], res["n_edited"]) == (2, 0, 2, 2)


def test_missing_or_excluded_segments_do_not_produce_partial_totals():
    res = perfusion_scores(_pct(s17=float("nan")))
    assert res["sss"] is None and res["sss_pct"] is None
    assert res["n_scored"]["stress"] == 16
    manual = perfusion_scores(_pct(s17=float("nan")), overrides={"stress": {17: 1}})
    assert manual["sss"] == 1
    excluded = perfusion_scores(_pct(), overrides={"stress": {3: None}})
    assert excluded["sss"] is None and excluded["rows"][2]["stress_edited"]


def test_stress_only_and_summary_reports_review_state():
    res = perfusion_scores(_pct(s4=45.0))
    assert not res["has_rest"] and res["srs"] is None and res["sds"] is None
    assert "SRS" not in score_summary_text(res)
    assert "SIN revisión" in score_summary_text(res)
    res["reviewed"] = True
    assert "revisado por el médico (0 corrección(es))" in score_summary_text(res)
    assert perfusion_scores(None)["available"] is False


def test_out_of_range_override_is_clamped():
    res = perfusion_scores(_pct(), overrides={"stress": {1: 9}})
    assert res["rows"][0]["stress_score"] == 4


def _synthetic_aha(n_slices=12, size=33):
    from types import SimpleNamespace
    from core.aha_segments import map_to_17_segments
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(yy - size // 2, xx - size // 2)
    mask = np.repeat(((radius >= 6) & (radius <= 10))[None], n_slices, axis=0)
    inner = np.linspace(2.0, 5.0, n_slices)
    seg = SimpleNamespace(mask=mask, center_per_slice=np.full((n_slices, 2), size // 2, dtype=float), inner_radius=inner)
    return seg, map_to_17_segments(seg)


@pytest.mark.parametrize("segment", [1, 2, 4, 6, 7, 9, 12, 13, 14, 15, 16, 17])
def test_bullseye_drawing_matches_segments_used_for_scores(segment):
    from core.aha_segments import bullseye_image, bullseye_segment_image
    seg, aha = _synthetic_aha()
    volume = np.ones(seg.mask.shape)
    volume[aha.segment_map == segment] = 5.0
    image = bullseye_image(volume, seg, aha)
    display = bullseye_segment_image(image.shape[0])
    inside = np.nanmean(image[display == segment])
    outside = np.nanmean(image[(display > 0) & (display != segment)])
    assert inside > 0.75 and outside < 0.45


@pytest.mark.parametrize("position,basal,mid,apical", [
    ((-1, 0), 1, 7, 13), ((-1, -1), 2, 8, None), ((1, -1), 3, 9, None), ((1, 0), 4, 10, 15),
    ((1, 1), 5, 11, None), ((-1, 1), 6, 12, None), ((0, -1), None, None, 14), ((0, 1), None, None, 16),
])
def test_aha_assignment_follows_standard_short_axis(position, basal, mid, apical):
    """SA estándar: anterior arriba (fila menor), septum a la izquierda (columna menor)."""
    seg, aha = _synthetic_aha()
    order = list(reversed(aha.apex_to_base_order))
    dy, dx = position
    y, x = 16 + int(round(8 * dy / np.hypot(dy, dx))), 16 + int(round(8 * dx / np.hypot(dy, dx)))
    levels = {"basal": order[1], "mid": order[len(order) // 2], "apical": order[int(len(order) * 0.8)]}
    for name, expected in (("basal", basal), ("mid", mid), ("apical", apical)):
        if expected is not None:
            assert aha.segment_map[levels[name], y, x] == expected, (name, position)


def test_display_angles_match_bullseye_layout():
    from core.aha_segments import bullseye_segment_layout, segment_display_angles
    for item in bullseye_segment_layout():
        if item["segment"] != 17:
            assert segment_display_angles(item["segment"]) == (item["theta0"], item["theta1"])
    t1, t2 = segment_display_angles(1)
    assert t1 < 90 < t2
    t1, t2 = segment_display_angles(2)
    assert t1 < 150 < t2


def test_bullseye_layout_matches_hit_testing_and_standard_positions():
    from core.aha_segments import bullseye_segment_at, bullseye_segment_layout
    for item in bullseye_segment_layout():
        r = (item["r0"] + item["r1"]) / 2
        theta = (item["theta0"] + item["theta1"]) / 2
        assert int(bullseye_segment_at(r, theta)) == item["segment"]
    assert int(bullseye_segment_at(0.9, 90)) == 1 and int(bullseye_segment_at(0.9, 270)) == 4
    assert int(bullseye_segment_at(0.3, 180)) == 14 and int(bullseye_segment_at(1.2, 0)) == 0


def test_territory_extent_and_scores_follow_territory_map():
    from core.aha_segments import TERRITORY_MAP, bullseye_segment_image, bullseye_territory_extent
    from core.perfusion_quant import territory_scores
    display = bullseye_segment_image(121)
    stress = np.where(display > 0, 1.0, np.nan)
    rest = stress.copy()
    lad = np.isin(display, TERRITORY_MAP["LAD"])
    stress[lad] = 0.5
    rest[lad] = 0.9
    extent = bullseye_territory_extent(stress, rest)
    assert extent["LAD"]["extent_pct"] == pytest.approx(100) and extent["LAD"]["reversible_pct"] == pytest.approx(100)
    assert extent["RCA"]["extent_pct"] == 0 and 0 < extent["TOTAL"]["extent_pct"] < 100
    scores = perfusion_scores(_pct(**{f"s{s}": 45.0 for s in TERRITORY_MAP["LAD"]}), _pct())
    terr = territory_scores(scores)
    assert terr["LAD"]["sss"] == 21 and terr["LAD"]["sds"] == 21 and terr["RCA"]["sss"] == 0
    assert terr["LAD"]["sss_pct"] == pytest.approx(75)
    assert territory_scores(perfusion_scores(_pct()))["LAD"]["srs"] is None


def test_lcars_window_renders_and_edits_in_offscreen_subprocess(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    script = tmp_path / "render_score.py"
    script.write_text(f"""
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
import numpy as np
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from tests.test_perfusion_score import _synthetic_aha, _pct
from core.aha_segments import bullseye_image
from ui.perfusion_score_window import PerfusionScoreWindow
from viz.colormaps import get_phase_cmap
seg, aha = _synthetic_aha()
stress = np.ones(seg.mask.shape); stress[np.isin(aha.segment_map, [1, 7, 13])] = 0.45
rest = np.ones(seg.mask.shape); rest[aha.segment_map == 1] = 0.8
win = PerfusionScoreWindow(None, stress_pct=_pct(s1=45.0, s7=45.0, s13=45.0), rest_pct=_pct(s1=80.0),
    stress_img=bullseye_image(stress, seg, aha), rest_img=bullseye_image(rest, seg, aha),
    overrides={{"stress": {{2: 1}}}}, cmap=get_phase_cmap("odyssey_cool"))
win.resize(1500, 860)
win.view_sss.segmentActivated.emit(7, 1)
win.view_srs.segmentActivated.emit(4, 0)
assert win.edits["stress"] == {{2: 1, 7: 4}}, win.edits
assert win.edits["rest"] == {{4: None}}, win.edits
assert win.result["srs"] is None and win.cells[("LAD", "SSS")].text() == "10"
win.grab().save(sys.argv[1])
print("OK", win.result["sss"])
""", encoding="utf-8")
    out = tmp_path / "score_window.png"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    proc = subprocess.run([sys.executable, str(script), str(out)], capture_output=True, text=True, env=env, timeout=120)
    if proc.returncode in (3221225501, -1073741795):
        pytest.skip("Qt offscreen nativo no disponible en este entorno")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK 10" in proc.stdout and out.stat().st_size > 10000


def test_html_report_shows_final_scores_and_review(tmp_path):
    from types import SimpleNamespace
    from core.perfusion_quant import perfusion_quant_summary
    from report.html_report import generate_html_report
    quant = perfusion_quant_summary(_pct(s1=35.0), _pct(s1=65.0))
    quant["scores"] = perfusion_scores(_pct(s1=35.0), _pct(s1=65.0), {"stress": {1: 3}})
    quant["scores"]["reviewed"] = True
    quant["score_image"] = _score_figure(tmp_path, quant["scores"])
    study = SimpleNamespace(cube=np.zeros((8, 4, 4, 4)), patient_name="Test", patient_id="1",
                            study_date="20260101", pixel_spacing=(6.8, 6.8), z_spacing_mm=6.8)
    seg = SimpleNamespace(mask=np.ones((4, 4, 4), dtype=bool), method="auto", n_voxels=64,
                          center_per_slice=np.full((4, 2), 2.0), inner_radius=np.ones(4), outer_radius=np.full(4, 2.0))
    out = tmp_path / "informe.html"
    generate_html_report(output_html=str(out), output_dir=str(tmp_path), study=study, seg=seg,
                         metrics={}, territory={}, processing_params={}, volumes={}, ef={},
                         perfusion_quant=quant)
    html = out.read_text(encoding="utf-8")
    assert "Score esf." in html and "3*" in html
    assert "SSS 3 · SRS 1 · SDS 2" in html and "revisado por el médico (1 corrección(es))" in html
    assert 'alt="Lámina de score de perfusión"' in html and 'src="data:image/' in html.split('alt="Lámina de score')[0][-200000:]


def _score_figure(tmp_path, scores):
    from core.aha_segments import bullseye_image, bullseye_territory_extent
    from viz.colormaps import get_phase_cmap
    from viz.perfusion_score_figure import save_perfusion_score_figure
    seg, aha = _synthetic_aha()
    stress = np.ones(seg.mask.shape)
    stress[aha.segment_map == 1] = 0.35
    s_img, r_img = bullseye_image(stress, seg, aha), bullseye_image(np.ones(seg.mask.shape), seg, aha)
    return save_perfusion_score_figure(str(tmp_path / "perfusion_score.png"), scores, stress_img=s_img, rest_img=r_img,
                                       extent=bullseye_territory_extent(s_img, r_img), cmap=get_phase_cmap("odyssey_cool"))


@pytest.mark.parametrize("with_rest", [True, False])
def test_score_figure_renders_with_and_without_rest(tmp_path, with_rest):
    scores = perfusion_scores(_pct(s1=35.0), _pct() if with_rest else None, {"stress": {2: 1}})
    path = _score_figure(tmp_path, scores)
    from PIL import Image
    with Image.open(path) as image:
        assert image.width > 1000 and image.height > 900


def test_pdf_report_shows_final_scores_and_review(tmp_path):
    from types import SimpleNamespace
    from core.perfusion_quant import perfusion_quant_summary
    from report.report_generator import generate_report
    pypdf = pytest.importorskip("pypdf")
    quant = perfusion_quant_summary(_pct(s1=35.0), _pct(s1=65.0))
    quant["scores"] = perfusion_scores(_pct(s1=35.0), _pct(s1=65.0), {"stress": {1: 3}})
    quant["scores"]["reviewed"] = True
    quant["score_image"] = _score_figure(tmp_path, quant["scores"])
    study = SimpleNamespace(cube=np.zeros((8, 4, 4, 4)), patient_name="Test", patient_id="1",
                            study_date="20260101", pixel_spacing=(6.8, 6.8), z_spacing_mm=6.8)
    seg = SimpleNamespace(mask=np.ones((4, 4, 4), dtype=bool), method="auto", n_voxels=64,
                          center_per_slice=np.full((4, 2), 2.0), inner_radius=np.ones(4), outer_radius=np.full(4, 2.0))
    out = tmp_path / "informe.pdf"
    generate_report(output_pdf=str(out), output_dir=str(tmp_path), study=study, seg=seg, metrics={},
                    territory={"LAD": {}}, processing_params={}, volumes={}, ef={}, perfusion_quant=quant)
    text = " ".join(page.extract_text() for page in pypdf.PdfReader(str(out)).pages)
    assert "Score E" in text and "Score R" in text and "3*" in text
    assert "SDS 2" in text and "revisado por el médico" in text
    assert sum(len(page.images) for page in pypdf.PdfReader(str(out)).pages) >= 1
