"""Quality reporting (CLAUDE.md §10.3, §17.4).

Read-only over ``quality.jsonl``. Produces quality_report.csv, summary.md, and
failed_frames.json, with an optional delta-vs-baseline KPI table (§35A).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import metrics as M
from .logging_setup import get_logger

_log = get_logger("face_swap.report")


def read_quality_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compute_kpis(rows: list[dict]) -> dict[str, float]:
    flickers = [r["flicker_score"] for r in rows if "flicker_score" in r]
    n = len(rows)
    n_fail = sum(1 for r in rows if r.get("verdict") == "FAIL")
    n_review = sum(1 for r in rows if "budget_exhausted" in r.get("reasons", []))
    over_025 = sum(1 for f in flickers if f > 0.25)
    return {
        "n_frames": n,
        "flicker_score_median": M.median(flickers),
        "flicker_score_p95": M.percentile(flickers, 95),
        "flicker_score_mean": (sum(flickers) / len(flickers)) if flickers else 0.0,
        "frames_over_0_25_pct": (over_025 / n) if n else 0.0,
        "fail_rate": (n_fail / n) if n else 0.0,
        "manual_review_rate": (n_review / n) if n else 0.0,
        "mean_retry_count": (sum(r.get("retry_count", 0) for r in rows) / n) if n else 0.0,
    }


def write_quality_csv(rows: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp_keys = ["embedding", "color", "landmark", "mask", "sharpness"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "verdict", "flicker_score", *comp_keys, "retry_count",
                    "retry_strategies"])
        for r in rows:
            comps = r.get("components", {})
            w.writerow([
                r.get("frame"), r.get("verdict"), r.get("flicker_score"),
                *[comps.get(k, "") for k in comp_keys],
                r.get("retry_count", 0), "|".join(r.get("retry_strategies", [])),
            ])


def write_failed_frames(rows: list[dict], out_path: Path) -> list[int]:
    failed = [r["frame"] for r in rows
              if r.get("verdict") == "FAIL" or "budget_exhausted" in r.get("reasons", [])]
    Path(out_path).write_text(json.dumps({"failed_frames": failed}, indent=2), encoding="utf-8")
    return failed


def _delta_table(kpis: dict[str, float], baseline: dict[str, float] | None) -> list[str]:
    rows = [
        "| Metric | Target | Actual | Baseline (v0) | Delta vs v0 |",
        "| --- | --- | --- | --- | --- |",
    ]
    spec = [
        ("Median Flicker Score", "< 0.08", "flicker_score_median", "flicker_score_median"),
        ("p95 Flicker Score", "↓ ≥ 50% vs v0", "flicker_score_p95", "flicker_score_p95"),
        ("% frames > 0.25", "< 1%", "frames_over_0_25_pct", None),
        ("Manual review rate", "< 2%", "manual_review_rate", None),
    ]
    for label, target, key, base_key in spec:
        actual = kpis.get(key, 0.0)
        base_val = baseline.get(base_key) if (baseline and base_key) else None
        if base_val is not None and base_val:
            delta = f"{(actual - base_val) / base_val * 100:+.0f}%"
            base_str = f"{base_val:.4f}"
        else:
            delta, base_str = "n/a", "n/a"
        rows.append(f"| {label} | {target} | {actual:.4f} | {base_str} | {delta} |")
    return rows


def write_summary_md(
    out_path: Path, *, run_id: str, kpis: dict[str, float],
    baseline: dict[str, float] | None = None, meta: dict | None = None,
) -> None:
    meta = meta or {}
    lines = [
        f"# Run {run_id} — Quality Summary",
        "",
        f"**Source:** {meta.get('source', 'n/a')}",
        f"**Frames:** {kpis.get('n_frames', 0)}",
        f"**Pipeline:** {meta.get('git_sha', 'n/a')} at {meta.get('captured_at', 'n/a')}",
        f"**Config:** {meta.get('config_name', 'n/a')}  (mode={meta.get('mode', 'n/a')})",
        "",
        "## §4A KPI compliance",
        "",
        *_delta_table(kpis, baseline),
        "",
        "## Failures",
        f"Manual review queue: {int(round(kpis.get('manual_review_rate', 0) * kpis.get('n_frames', 0)))} frames "
        f"({kpis.get('manual_review_rate', 0) * 100:.2f}%).",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class Reporter:
    """Write all reports for a run directory."""

    def write_all(self, run_dir: Path, *, run_id: str, baseline_kpis: dict | None = None,
                  meta: dict | None = None) -> dict[str, float]:
        run_dir = Path(run_dir)
        reports = run_dir / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        rows = read_quality_jsonl(run_dir / "quality.jsonl")
        kpis = compute_kpis(rows)
        write_quality_csv(rows, reports / "quality_report.csv")
        write_failed_frames(rows, reports / "failed_frames.json")
        write_summary_md(reports / "summary.md", run_id=run_id, kpis=kpis,
                         baseline=baseline_kpis, meta=meta)
        _log.info("reports_written", run_dir=str(run_dir), **{k: kpis[k] for k in
                  ("n_frames", "flicker_score_median", "flicker_score_p95")})
        return kpis
