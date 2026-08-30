# -*- coding: utf-8 -*-
"""Tests puros del modelo dual de perfusión (core/dual_stage.py)."""
from __future__ import annotations

import pytest

from core.dual_stage import (
    StageState,
    DualSession,
    STAGE_STRESS,
    STAGE_REST,
    STAGES,
    PIPELINE_STEPS,
    normalize_stage,
)


def test_normalize_stage_aliases():
    assert normalize_stage("rest") == STAGE_REST
    assert normalize_stage("Reposo") == STAGE_REST
    assert normalize_stage("BASAL") == STAGE_REST
    assert normalize_stage("stress") == STAGE_STRESS
    assert normalize_stage("esfuerzo") == STAGE_STRESS
    assert normalize_stage("") == STAGE_STRESS  # default
    assert normalize_stage(None) == STAGE_STRESS


def test_stage_default_label_and_stage():
    s = StageState(stage="rest")
    assert s.stage == STAGE_REST
    assert s.label == "Reposo"
    s2 = StageState(stage="esfuerzo")
    assert s2.stage == STAGE_STRESS
    assert s2.label == "Esfuerzo"


def test_stage_custom_label_preserved():
    s = StageState(stage="stress", label="Post-dipiridamol")
    assert s.label == "Post-dipiridamol"


def test_progress_empty_all_false():
    s = StageState()
    prog = s.progress()
    assert list(prog.keys()) == list(PIPELINE_STEPS)
    assert all(v is False for v in prog.values())
    assert s.last_completed_step() is None
    assert not s.is_complete()


def test_progress_contiguous_and_gaps():
    s = StageState()
    s.raw_study = object()
    s.motion_result = {"ok": True}
    s.recon_result = object()
    # salteo reorient (gap): cuts presente pero no contiguo
    s.cut_study = object()
    assert s.step_done("raw")
    assert s.step_done("motion")
    assert s.step_done("recon")
    assert not s.step_done("reorient")
    assert s.step_done("cuts")
    # last_completed_step corta en el primer hueco
    assert s.last_completed_step() == "recon"


def test_step_done_unknown_raises():
    with pytest.raises(ValueError):
        StageState().step_done("inexistente")


def test_reset_from_invalidates_downstream_only():
    s = StageState()
    s.raw_study = object()
    s.motion_result = {"a": 1}
    s.recon_result = object()
    s.reoriented_ungated = object()
    s.cut_study = object()
    s.axes = {"SA": object()}
    s.metrics = {"phase_sd": 10.0}
    # invalidar desde recon: recon/reorient/cuts/analysis se limpian; raw/motion quedan
    s.reset_from("recon")
    assert s.has_raw()
    assert s.motion_result is not None
    assert s.recon_result is None
    assert not s.has_reorient()
    assert not s.has_cuts()
    assert s.metrics is None


def test_reset_from_raw_clears_everything():
    s = StageState()
    s.raw_study = object()
    s.source_path = "x.dcm"
    s.recon_result = object()
    s.metrics = {"phase_sd": 1.0}
    s.reset_from("raw")
    assert not s.has_raw()
    assert s.source_path == ""
    assert s.recon_result is None
    assert s.metrics is None


def test_reset_from_analysis_keeps_cuts():
    s = StageState()
    s.raw_study = object()
    s.cut_study = object()
    s.metrics = {"phase_sd": 1.0}
    s.reset_from("analysis")
    assert s.has_cuts()
    assert s.metrics is None


def test_dualsession_creates_canonical_stages():
    sess = DualSession()
    assert set(sess.stages.keys()) >= set(STAGES)
    assert sess.active == STAGE_STRESS
    assert isinstance(sess.stage("stress"), StageState)
    assert isinstance(sess.stage("rest"), StageState)


def test_dualsession_active_and_other():
    sess = DualSession()
    sess.set_active("rest")
    assert sess.active == STAGE_REST
    assert sess.active_stage().stage == STAGE_REST
    assert sess.other_stage().stage == STAGE_STRESS
    sess.set_active("stress")
    assert sess.other_stage().stage == STAGE_REST


def test_dualsession_has_both_and_loaded():
    sess = DualSession()
    assert not sess.has_both()
    assert sess.loaded_stages() == []
    sess.stage("stress").raw_study = object()
    assert sess.loaded_stages() == [STAGE_STRESS]
    assert not sess.has_both()
    sess.stage("rest").raw_study = object()
    assert sess.has_both()
    assert set(sess.loaded_stages()) == set(STAGES)


def test_dualsession_both_complete():
    sess = DualSession()
    sess.stage("stress").metrics = {"phase_sd": 1.0}
    assert not sess.both_complete()
    sess.stage("rest").metrics = {"phase_sd": 2.0}
    assert sess.both_complete()


def test_lock_voi_only_first_wins():
    sess = DualSession()
    sess.lock_voi({"rz": 10, "ry": 8, "rx": 8}, "stress")
    assert sess.locked_voi_stage == STAGE_STRESS
    assert sess.locked_voi["rz"] == 10
    # segunda etapa NO puede pisar el lock
    sess.lock_voi({"rz": 20, "ry": 4, "rx": 4}, "rest")
    assert sess.locked_voi_stage == STAGE_STRESS
    assert sess.locked_voi["rz"] == 10


def test_clear_resets_session():
    sess = DualSession()
    sess.stage("stress").raw_study = object()
    sess.stage("rest").metrics = {"phase_sd": 1.0}
    sess.lock_voi({"rz": 1, "ry": 1, "rx": 1}, "stress")
    sess.comparison = {"delta": 5.0}
    sess.clear()
    assert not sess.has_both()
    assert sess.locked_voi is None
    assert sess.comparison is None
    assert sess.active == STAGE_STRESS


def test_snapshot_progress_shape():
    sess = DualSession()
    sess.stage("stress").raw_study = object()
    snap = sess.snapshot_progress()
    assert set(snap.keys()) >= set(STAGES)
    assert snap[STAGE_STRESS]["raw"] is True
    assert snap[STAGE_REST]["raw"] is False
