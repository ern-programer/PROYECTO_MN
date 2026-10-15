import numpy as np
import pytest

from core import axis_normals


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(axis_normals, "_DB_DIR", str(tmp_path))
    monkeypatch.setattr(axis_normals, "_DB_PATH", str(tmp_path / "axis_normals.json"))
    return tmp_path


def test_same_study_and_stage_keeps_only_latest_correction(db):
    assert axis_normals.record_case([0.4, 0.8, 0.5], stage="stress", study_key="A")
    assert axis_normals.record_case([0.5, 0.7, 0.5], stage="stress", study_key="A")
    assert axis_normals.record_case([0.5, 0.7, 0.5], stage="rest", study_key="A")
    cases = axis_normals._load()["cases"]
    assert len(cases) == 2
    stress = next(c for c in cases if c["stage"] == "stress")
    assert stress["axis"] == pytest.approx(list(np.array([0.5, 0.7, 0.5]) / np.linalg.norm([0.5, 0.7, 0.5])), abs=1e-4)


def test_degenerate_default_axis_is_not_recorded(db):
    assert not axis_normals.record_case([1.0, 0.0, 0.0], stage="stress", study_key="A")
    assert not axis_normals.record_case([0.0, 0.0, 0.0])
    assert axis_normals.count_cases() == 0


def test_recalibration_rejects_outliers_and_default_axes(db):
    rng = np.random.default_rng(0)
    base = np.array([0.45, 0.75, 0.48])
    for k in range(10):
        axis_normals.record_case(base + rng.normal(0, 0.03, 3), stage="stress", study_key=f"S{k}")
    axis_normals.record_case([0.0, 0.2, 1.0], stage="stress", study_key="outlier1")
    axis_normals.record_case([0.95, 0.1, 0.0], stage="stress", study_key="outlier2")
    data = axis_normals._load()
    data["cases"].append({"axis": [1.0, 0.0, 0.0], "stage": "", "study_key": ""})
    axis_normals._save(data)
    prior, used = axis_normals.recalibrate()
    assert used == 10
    angle = np.degrees(np.arccos(min(1.0, abs(np.dot(prior, base / np.linalg.norm(base))))))
    assert angle < 3.0


def test_trust_reorientation_does_not_feed_database():
    import ast
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "ui" / "main_window.py"
    text = source.read_text(encoding="utf-8")
    assert "dlg._accept(record_normal=False)" in text
    tree = ast.parse((source.parent / "reorientation_dialog.py").read_text(encoding="utf-8"))
    accept = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_accept")
    body = ast.unparse(accept)
    assert "record_normal and" in body and "study_key=self._study_key" in body
