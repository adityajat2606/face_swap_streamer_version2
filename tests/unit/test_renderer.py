from __future__ import annotations

from pathlib import Path

from face_swap import renderer


def test_encode_argv_framerate_before_input(monkeypatch):
    captured = {}
    monkeypatch.setattr(renderer.subprocess, "run", lambda cmd, **k: captured.setdefault("cmd", cmd))
    cmd = renderer.encode_from_png_sequence(Path("frames"), 24.0, Path("out.mp4"),
                                            codec="h264", crf=16, preset="slow", ffmpeg="ffmpeg")
    # -framerate must precede -i (§6.4)
    assert cmd.index("-framerate") < cmd.index("-i")
    assert "libx264" in cmd
    assert "-crf" in cmd and "16" in cmd


def test_reattach_audio_copies_not_reencodes(monkeypatch):
    class _R:
        returncode = 0

    monkeypatch.setattr(renderer.subprocess, "run", lambda cmd, **k: _R())
    cmd = renderer.reattach_audio(Path("v.mp4"), Path("orig.mp4"), Path("final.mp4"),
                                  ffmpeg="ffmpeg")
    assert "-c:a" in cmd and "copy" in cmd
    assert "1:a:0?" in cmd  # optional audio map


def test_reattach_audio_falls_back_to_aac(monkeypatch):
    calls = []

    class _Fail:
        returncode = 1

    def fake_run(cmd, **k):
        calls.append(cmd)
        return _Fail()

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    renderer.reattach_audio(Path("v.mp4"), Path("o.mp4"), Path("f.mp4"), ffmpeg="ffmpeg")
    # second call uses aac
    assert any("aac" in c for c in calls[-1])


def test_side_by_side_uses_hstack(monkeypatch):
    monkeypatch.setattr(renderer.subprocess, "run", lambda cmd, **k: None)
    cmd = renderer.side_by_side_preview(Path("a.mp4"), Path("b.mp4"), Path("p.mp4"),
                                        ffmpeg="ffmpeg")
    assert any("hstack" in part for part in cmd)
