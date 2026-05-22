"""End-to-end orchestration of pipeline.run with a stubbed StageRunner.

Validates that run() creates the run dir, persists config, writes the manifest
and reports, and maps KPIs to the right exit code — all without a GPU.
"""

from __future__ import annotations

import json

import face_swap.runner as runner_mod
from face_swap.config import Config
from face_swap.errors import ExitCode
from face_swap.pipeline import run


class _FakeRunner:
    def __init__(self, cfg, dirs, obs, manifest):
        self.cfg, self.dirs, self.obs, self.manifest = cfg, dirs, obs, manifest
        self.had_manual_review = False

    def execute(self, start_frame=0):
        # emit a couple of quality lines so the reporter has data
        with self.dirs.quality_jsonl.open("w", encoding="utf-8") as fh:
            for i in range(3):
                fh.write(json.dumps({"frame": i, "verdict": "PASS",
                                     "flicker_score": 0.05, "components": {},
                                     "reasons": []}) + "\n")
        return {"n_frames": 3, "flicker_score_median": 0.05, "flicker_score_p95": 0.07,
                "manual_review_rate": 0.0}


def _cfg(tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"fake-video")
    return Config(project={"name": "t", "mode": "quality"},
                  input={"video_path": vid, "hero_reference": vid, "heroine_reference": vid})


def test_run_writes_manifest_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "StageRunner", _FakeRunner)
    out = tmp_path / "output"
    code = run(_cfg(tmp_path), out)
    assert code == ExitCode.OK

    run_dirs = list(out.iterdir())
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    assert (rd / "processing_log.json").is_file()
    assert (rd / "config" / "run_config.yaml").is_file()
    assert (rd / "reports" / "summary.md").is_file()

    manifest = json.loads((rd / "processing_log.json").read_text())
    assert manifest["exit_code"] == int(ExitCode.OK)
    assert manifest["kpi_results"]["n_frames"] == 3
    assert "observability" in manifest
    assert manifest["source_video"]["sha256"] is not None


def test_run_kpi_not_met_exit_code(tmp_path, monkeypatch):
    class _HighFlicker(_FakeRunner):
        def execute(self, start_frame=0):
            super().execute(start_frame)
            return {"n_frames": 3, "flicker_score_median": 0.20, "manual_review_rate": 0.0}

    monkeypatch.setattr(runner_mod, "StageRunner", _HighFlicker)
    code = run(_cfg(tmp_path), tmp_path / "out2")
    assert code == ExitCode.KPI_NOT_MET


def test_run_manual_review_exit_code(tmp_path, monkeypatch):
    class _Review(_FakeRunner):
        def __init__(self, *a):
            super().__init__(*a)
            self.had_manual_review = True

    monkeypatch.setattr(runner_mod, "StageRunner", _Review)
    code = run(_cfg(tmp_path), tmp_path / "out3")
    assert code == ExitCode.HUMAN_REVIEW_REQUIRED
