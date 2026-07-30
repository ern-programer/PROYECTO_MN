"""Tests para el export segmental (fase por segmento + territorio + textura + stress-rest)."""
from __future__ import annotations

import os

from core.export_manager import export_segmental_csv


def _meta():
    return {
        "patient_name": "TEST^PACIENTE",
        "patient_id": "P123",
        "study_description": "GATED SPECT",
    }


def test_export_segmental_fase_y_territorio(tmp_path):
    phase_by_seg = {i: float(i * 10 % 360) for i in range(1, 18)}
    territory = {
        "LAD": {"mean": 30.0, "std": 12.0, "min": 10.0, "max": 50.0, "n": 7},
        "LCx": {"mean": 120.0, "std": 15.0, "min": 100.0, "max": 140.0, "n": 5},
        "RCA": {"mean": 210.0, "std": 18.0, "min": 190.0, "max": 230.0, "n": 5},
    }
    n_per_seg = {i: 100 + i for i in range(1, 18)}
    out = os.path.join(str(tmp_path), "seg.csv")
    export_segmental_csv(out, _meta(), phase_by_seg, territory, n_per_seg)
    text = open(out, encoding="utf-8").read()
    assert "Fase media por segmento" in text
    assert "Territorio coronario" in text
    assert "LAD,30.000,12.000" in text
    # segmento 17 = territorio LAD (ápex)
    assert "17,LAD," in text


def test_export_segmental_con_stress_rest(tmp_path):
    from core.stress_rest import compare_stress_rest

    def _m(sd, bw, ent):
        return {
            "phase_sd": sd, "bandwidth": bw, "entropy_shannon_bits": ent,
            "entropy_normalized_pct": ent * 15, "asynchrony_index": 5.0,
            "skewness": 0.0, "kurtosis": 0.0, "peak_width": 10.0,
            "mean_phase": 100.0, "peak_phase": 100.0, "latest_activation_phase": 130.0,
        }

    sr = compare_stress_rest(_m(45, 90, 4.6), _m(30, 60, 4.0))
    out = os.path.join(str(tmp_path), "seg_sr.csv")
    export_segmental_csv(out, _meta(), {1: 10.0}, stress_rest=sr)
    text = open(out, encoding="utf-8").read()
    assert "Delta stress-rest de fase" in text
    assert "phase_sd,45.000,30.000,15.000" in text
    assert "40021521" in text  # nota con PMID


def test_export_segmental_con_textura(tmp_path):
    texture = {
        1: {"available": True, "contrast": 12.3, "dissimilarity": 2.1,
            "homogeneity": 0.4, "energy": 0.2, "correlation": 0.8,
            "glcm_entropy": 3.5, "n_pixels": 120},
    }
    from core.perfusion_texture import combine_perfusion_phase

    rows = combine_perfusion_phase(texture, {1: 45.0})
    out = os.path.join(str(tmp_path), "seg_tex.csv")
    export_segmental_csv(out, _meta(), {1: 45.0}, texture_by_seg=texture, perfusion_phase_rows=rows)
    text = open(out, encoding="utf-8").read()
    assert "Textura GLCM de perfusión por segmento" in text
    assert "40391672" in text  # Jiang PMID en el header
    assert "Perfusión (GLCM) + fase por segmento" in text
