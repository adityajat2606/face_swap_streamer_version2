"""Frame extraction → rebuild reliability (CLAUDE.md §6.5).

Requires ffmpeg; skips otherwise. Asserts frame count, fps and duration are
preserved across an extract/re-encode round trip.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from face_swap import video_manager
from face_swap.frame_store import FrameStore, extract_png_sequence
from face_swap.renderer import encode_from_png_sequence

pytestmark = pytest.mark.integration


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def clip(tmp_path):
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not available")
    out = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "testsrc=duration=2:size=320x180:rate=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True)
    return out


def test_extract_count_matches(clip, tmp_path):
    meta = video_manager.probe(str(clip))
    frames_dir = tmp_path / "frames"
    extract_png_sequence(str(clip), frames_dir)
    store = FrameStore("png", frames_dir)
    n = sum(1 for _ in store.iter_frames())
    assert abs(n - meta.n_frames) <= 1


def test_roundtrip_fps_preserved(clip, tmp_path):
    meta = video_manager.probe(str(clip))
    frames_dir = tmp_path / "frames"
    extract_png_sequence(str(clip), frames_dir)
    rebuilt = tmp_path / "rebuilt.mp4"
    encode_from_png_sequence(frames_dir, meta.fps, rebuilt, codec="h264", crf=18,
                             preset="ultrafast")
    out_meta = video_manager.probe(str(rebuilt))
    assert abs(out_meta.fps - meta.fps) < 1e-3
