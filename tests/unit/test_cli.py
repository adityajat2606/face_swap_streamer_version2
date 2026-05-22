from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from face_swap.cli import main

CONFIGS = Path(__file__).resolve().parent.parent.parent / "configs"


def test_version():
    res = CliRunner().invoke(main, ["--version"])
    assert res.exit_code == 0
    assert "face_swap" in res.output


def test_help_lists_commands():
    res = CliRunner().invoke(main, ["--help"])
    assert res.exit_code == 0
    for cmd in ("run", "resume", "baseline", "health", "assign-identities"):
        assert cmd in res.output


def test_health_command(tmp_path):
    res = CliRunner().invoke(main, ["health", "--config", str(CONFIGS / "draft.yaml")])
    # input video not configured in the preset → DOWN → exit 1, but JSON prints
    assert "status" in res.output
    assert "checks" in res.output


def test_run_missing_video_errors():
    res = CliRunner().invoke(main, [
        "run", "--video", "nope.mp4", "--hero", "h.png", "--heroine", "hr.png",
        "--config", str(CONFIGS / "draft.yaml"), "--output", "out"])
    assert res.exit_code != 0  # click validates exists=True


def test_set_override_bad_format(tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"x")
    res = CliRunner().invoke(main, [
        "run", "--video", str(vid), "--hero", str(vid), "--heroine", str(vid),
        "--config", str(CONFIGS / "draft.yaml"), "--output", str(tmp_path / "o"),
        "--set", "no_equals_sign"])
    assert res.exit_code != 0
