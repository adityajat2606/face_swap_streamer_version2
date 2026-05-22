"""Shared fixtures (CLAUDE.md §16.2)."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def synthetic_face(rng):
    """A noise patch shaped like a 256x256 BGR face crop."""
    return rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture(scope="session")
def tiny_clip_path(tmp_path_factory):
    """A 24-frame 320x180 clip via ffmpeg; skips if ffmpeg is absent."""
    if not _have_ffmpeg():
        pytest.skip("ffmpeg not available")
    out = tmp_path_factory.mktemp("clips") / "tiny.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "testsrc=duration=1:size=320x180:rate=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    return out
