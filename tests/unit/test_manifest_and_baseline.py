from __future__ import annotations

import json
import re

from face_swap import manifest as MF
from face_swap.baseline import measure_all


def test_run_id_format():
    rid = MF.new_run_id()
    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", rid)


def test_sha256_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    h = MF.sha256_file(p)
    assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_file_or_none_missing():
    assert MF.sha256_file_or_none(None) is None
    assert MF.sha256_file_or_none("nope.bin") is None


def test_model_versions(tmp_path):
    m = tmp_path / "model.onnx"
    m.write_bytes(b"weights")
    out = MF.model_versions({"swap": str(m), "detector": None})
    assert out["swap"].startswith("model.onnx@")
    assert out["detector"] == "absent"


def test_redact():
    assert MF.redact("/secret/path.mp4", True).startswith("sha256:")
    assert MF.redact("/secret/path.mp4", False) == "/secret/path.mp4"
    assert MF.redact(None, True) is None


def test_write_manifest_atomic(tmp_path):
    MF.write_manifest(tmp_path, {"run_id": "x"})
    assert (tmp_path / "processing_log.json").is_file()
    assert not (tmp_path / "processing_log.json.tmp").exists()
    data = json.loads((tmp_path / "processing_log.json").read_text())
    assert data["run_id"] == "x"


def test_baseline_schema_with_missing_manifest(tmp_path):
    out = tmp_path / "v0_metrics.json"
    result = measure_all(tmp_path / "absent.yaml", out)
    assert out.is_file()
    assert result["tag"] == "v0-baseline"
    # all five test sets present (§5.6 gate)
    assert set(result["test_sets"]) == {
        "slow_romantic", "medium", "fast_dance", "profile", "challenging"}


def test_baseline_reads_manifest_clips(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "test_set_1_slow_romantic:\n  - path: clips/slow_01.mp4\n", encoding="utf-8")
    result = measure_all(manifest, tmp_path / "out.json")
    assert result["test_sets"]["slow_romantic"]["clips"][0]["path"] == "clips/slow_01.mp4"
    assert result["test_sets"]["slow_romantic"]["clips"][0]["metrics"][
        "flicker_score_median"] is None
