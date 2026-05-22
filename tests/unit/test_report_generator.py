from __future__ import annotations

import json

from face_swap.report_generator import (
    Reporter,
    compute_kpis,
    read_quality_jsonl,
    write_failed_frames,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _row(frame, verdict, flicker, reasons=()):
    return {"frame": frame, "verdict": verdict, "flicker_score": flicker,
            "components": {"embedding": 0.1}, "retry_count": 0, "reasons": list(reasons)}


def test_compute_kpis(tmp_path):
    rows = [_row(i, "PASS", 0.05) for i in range(9)] + [_row(9, "FAIL", 0.40)]
    p = tmp_path / "quality.jsonl"
    _write_jsonl(p, rows)
    kpis = compute_kpis(read_quality_jsonl(p))
    assert kpis["n_frames"] == 10
    assert kpis["fail_rate"] == 0.1
    assert kpis["frames_over_0_25_pct"] == 0.1
    assert kpis["flicker_score_median"] == 0.05


def test_failed_frames_includes_budget_exhausted(tmp_path):
    rows = [_row(0, "PASS", 0.05),
            _row(1, "WARNING", 0.2, reasons=["crop_larger", "budget_exhausted"]),
            _row(2, "FAIL", 0.5)]
    p = tmp_path / "quality.jsonl"
    _write_jsonl(p, rows)
    failed = write_failed_frames(read_quality_jsonl(p), tmp_path / "failed.json")
    assert set(failed) == {1, 2}


def test_reporter_writes_all_files(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    _write_jsonl(run / "quality.jsonl", [_row(i, "PASS", 0.04) for i in range(5)])
    kpis = Reporter().write_all(run, run_id="run1",
                                baseline_kpis={"flicker_score_p95": 0.2},
                                meta={"mode": "quality"})
    assert (run / "reports" / "summary.md").is_file()
    assert (run / "reports" / "quality_report.csv").is_file()
    assert (run / "reports" / "failed_frames.json").is_file()
    assert kpis["n_frames"] == 5
    summary = (run / "reports" / "summary.md").read_text()
    assert "KPI compliance" in summary


def test_empty_quality_jsonl(tmp_path):
    kpis = compute_kpis(read_quality_jsonl(tmp_path / "nope.jsonl"))
    assert kpis["n_frames"] == 0
    assert kpis["flicker_score_median"] == 0.0
