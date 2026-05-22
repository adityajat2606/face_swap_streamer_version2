from __future__ import annotations

from face_swap.config import Config
from face_swap.errors import ExitCode
from face_swap.observability import Observatory
from face_swap.pipeline import (
    RunDirs,
    _exit_code_from_kpis,
    _load_baseline_kpis,
    _register_health_checks,
    build_manifest_skeleton,
)


def _cfg(tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"fake")
    return Config(project={"name": "t", "mode": "quality"},
                  input={"video_path": vid, "hero_reference": vid, "heroine_reference": vid})


def test_run_dirs_make(tmp_path):
    dirs = RunDirs(tmp_path, "run1").make()
    for d in (dirs.root, dirs.checkpoint, dirs.frames, dirs.debug, dirs.reports, dirs.config):
        assert d.is_dir()


def test_manifest_skeleton_has_hashes(tmp_path):
    cfg = _cfg(tmp_path)
    dirs = RunDirs(tmp_path, "run1").make()
    m = build_manifest_skeleton(cfg, "run1", dirs)
    assert m["run_id"] == "run1"
    assert len(m["config_hash"]) == 64
    assert m["source_video"]["sha256"] is not None
    assert m["random_seed"] == 42


def test_manifest_redacts_paths_when_enabled(tmp_path):
    cfg = _cfg(tmp_path).model_copy(update={
        "output": _cfg(tmp_path).output.model_copy(update={"redact_paths": True})})
    dirs = RunDirs(tmp_path, "r").make()
    m = build_manifest_skeleton(cfg, "r", dirs)
    assert m["source_video"]["path"].startswith("sha256:")


def test_exit_code_kpi_met():
    assert _exit_code_from_kpis({"flicker_score_median": 0.05}, False) == ExitCode.OK


def test_exit_code_kpi_not_met():
    assert _exit_code_from_kpis({"flicker_score_median": 0.20}, False) == ExitCode.KPI_NOT_MET


def test_exit_code_manual_review_wins():
    assert _exit_code_from_kpis({"flicker_score_median": 0.05}, True) == \
        ExitCode.HUMAN_REVIEW_REQUIRED


def test_health_checks_registered(tmp_path):
    cfg = _cfg(tmp_path)
    obs = Observatory()
    _register_health_checks(obs, cfg)
    report = obs.health.run()
    assert "config" in report["checks"]
    assert "input" in report["checks"]
    assert report["checks"]["input"]["status"] == "UP"  # video exists


def test_load_baseline_missing_returns_none(tmp_path):
    assert _load_baseline_kpis(None) is None
    assert _load_baseline_kpis(tmp_path / "absent.json") is None


def test_load_baseline_parses(tmp_path):
    p = tmp_path / "b.json"
    p.write_text('{"flat_kpis": {"flicker_score_p95": 0.2}}')
    assert _load_baseline_kpis(p)["flat_kpis"]["flicker_score_p95"] == 0.2
