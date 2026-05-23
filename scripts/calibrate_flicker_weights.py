"""Calibrate the FR-9 Flicker Score weights against human rubric scores.

Usage:
  python scripts/calibrate_flicker_weights.py <quality.jsonl> <manual_scores.csv>

`manual_scores.csv` has columns ``frame,rubric`` (rubric 1-5). The script joins
on frame index, runs NNLS, prints calibrated weights + Spearman ρ, and writes
``calibrated_weights.json``. Per PRD §15.4 acceptance is Spearman ρ >= 0.5.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from face_swap.calibration import calibrate_weights  # noqa: E402
from face_swap.report_generator import read_quality_jsonl  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: calibrate_flicker_weights.py <quality.jsonl> <manual_scores.csv>",
              file=sys.stderr)
        return 10
    rows = read_quality_jsonl(Path(sys.argv[1]))
    comps_by_frame = {r["frame"]: r.get("components", {}) for r in rows}
    with open(sys.argv[2], newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rubric_pairs = [(int(r["frame"]), float(r["rubric"])) for r in reader]
    components, rubric = [], []
    for frame, score in rubric_pairs:
        if frame in comps_by_frame:
            components.append(comps_by_frame[frame])
            rubric.append(score)
    if not components:
        print("no overlap between quality.jsonl and manual_scores.csv frames", file=sys.stderr)
        return 10
    weights, rho = calibrate_weights(components, rubric)
    print(f"weights: {weights}")
    print(f"spearman rho: {rho:.3f}  (PRD §15.4 accept >= 0.5)")
    out = Path("calibrated_weights.json")
    out.write_text(json.dumps({"weights": weights, "spearman_rho": rho}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0 if rho >= 0.5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
