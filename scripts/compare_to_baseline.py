"""Compare a run's KPIs against baseline/v0_metrics.json (CLAUDE.md §17.4, §35A).

Asserts the §35A gate: p95 Flicker Score reduced >= 50% vs v0. Exit 0 if met,
3 if not (KPI gap), 10 on input error.

Usage: python scripts/compare_to_baseline.py <run_dir> [baseline.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from face_swap.report_generator import compute_kpis, read_quality_jsonl  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: compare_to_baseline.py <run_dir> [baseline.json]", file=sys.stderr)
        return 10
    run_dir = Path(sys.argv[1])
    baseline_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("baseline/v0_metrics.json")
    kpis = compute_kpis(read_quality_jsonl(run_dir / "quality.jsonl"))
    base = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else {}
    base_p95 = (base.get("flat_kpis") or {}).get("flicker_score_p95")
    cur_p95 = kpis["flicker_score_p95"]
    print(f"current p95 flicker = {cur_p95:.4f}")
    if base_p95:
        reduction = (base_p95 - cur_p95) / base_p95
        print(f"baseline p95 = {base_p95:.4f}  reduction = {reduction * 100:.1f}%")
        if reduction < 0.50:
            print("FAIL: <50% reduction in p95 Flicker Score (§35A gate)")
            return 3
        print("PASS: >=50% reduction (§35A gate)")
    else:
        print("no baseline p95 available; skipping gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
