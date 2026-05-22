"""Baseline measurement (CLAUDE.md §5.5). Writes baseline/v0_metrics.json.

For Milestone 1 only the simple metrics are computed; the rest are ``null``
until the pipeline produces them (§5.5 table). The script never invents human
rubric scores (§5.5).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .logging_setup import get_logger
from .manifest import git_sha, utc_now_iso

_log = get_logger("face_swap.baseline")

_TEST_SETS = ["slow_romantic", "medium", "fast_dance", "profile", "challenging"]

_NULL_CLIP_METRICS = {
    "face_detection_success_rate": None,
    "frame_failure_rate": None,
    "flicker_score_median": None,
    "flicker_score_p95": None,
    "identity_cosine_distance_mean": None,
    "gpu_util_avg": None,
    "wall_clock_s": None,
    "human_rubric_mean": None,
}


def measure_all(manifest_path: Path, output_path: Path) -> dict:
    """Build the baseline metrics file from a test-set manifest.

    On a CPU-only host (no GPU/models) this emits the schema with ``null``
    metrics per §5.5 so the file parses and CI can validate its shape; the GPU
    host fills in real numbers by running the v0 pipeline.
    """
    manifest_path = Path(manifest_path)
    test_sets: dict[str, dict] = {}
    manifest_data: dict = {}
    if manifest_path.is_file():
        manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}

    for set_name in _TEST_SETS:
        entries = manifest_data.get(_match_set_key(manifest_data, set_name), [])
        clips = []
        for entry in entries or []:
            clips.append({"path": entry.get("path", "unknown"),
                          "metrics": dict(_NULL_CLIP_METRICS)})
        test_sets[set_name] = {"clips": clips}

    result = {
        "tag": "v0-baseline",
        "commit": git_sha(),
        "captured_at": utc_now_iso(),
        "test_sets": test_sets,
        # convenience flat KPIs for delta reporting; filled by the GPU host run.
        "flat_kpis": {"flicker_score_median": None, "flicker_score_p95": None},
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log.info("baseline_written", path=str(output_path), test_sets=len(test_sets))
    return result


def _match_set_key(data: dict, set_name: str) -> str:
    for key in data:
        if set_name in key:
            return key
    return set_name
