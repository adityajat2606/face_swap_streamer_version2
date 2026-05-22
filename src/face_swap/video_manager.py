"""Video ingest + metadata via ffprobe (CLAUDE.md §6.1, §11).

Pure I/O; no model loads. All ffmpeg/ffprobe calls use argument lists, never
``shell=True`` (§18.8).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import InputError
from .logging_setup import get_logger

_log = get_logger("face_swap.video")


def _ffprobe_bin() -> str:
    return shutil.which("ffprobe") or "ffprobe"


def _ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


@dataclass(slots=True, frozen=True)
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    codec: str
    has_audio: bool
    audio_codec: str | None
    pix_fmt: str


def _parse_rate(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/")
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(rate)


def probe(path: str | Path, *, ffprobe: str | None = None) -> VideoMeta:
    """Probe a video's metadata. Computes ``n_frames`` by stream scan if the
    container does not report it (common for VFR)."""
    path = str(path)
    if not Path(path).is_file():
        raise InputError(f"video not found: {path}")
    probe_bin = ffprobe or _ffprobe_bin()
    try:
        out = subprocess.check_output(
            [probe_bin, "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", path],
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise InputError(f"ffprobe not found ({probe_bin}); install FFmpeg") from exc
    except subprocess.CalledProcessError as exc:
        raise InputError(f"ffprobe failed on {path}: {exc.stderr.decode(errors='replace')}") from exc

    return parse_probe_json(out.decode("utf-8"), path, probe_bin=probe_bin)


def parse_probe_json(text: str, path: str, *, probe_bin: str | None = None) -> VideoMeta:
    """Parse ffprobe JSON into :class:`VideoMeta` (separated for testability)."""
    data = json.loads(text)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise InputError(f"no video stream in {path}")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fps = _parse_rate(v.get("r_frame_rate", "0/1"))
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)

    n_frames = int(v.get("nb_frames") or 0)
    if n_frames <= 0:
        # container didn't report it — derive from fps*duration as a fallback.
        n_frames = int(round(fps * duration)) if fps and duration else 0

    return VideoMeta(
        path=path,
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        n_frames=n_frames,
        duration_s=duration,
        codec=v.get("codec_name", "unknown"),
        has_audio=a is not None,
        audio_codec=a.get("codec_name") if a else None,
        pix_fmt=v.get("pix_fmt", "yuv420p"),
    )


def count_frames_by_scan(path: str | Path, *, ffprobe: str | None = None) -> int:
    """Count frames by scanning the stream (authoritative for VFR)."""
    probe_bin = ffprobe or _ffprobe_bin()
    out = subprocess.check_output(
        [probe_bin, "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-print_format", "json", str(path)],
        stderr=subprocess.PIPE,
    )
    data = json.loads(out)
    return int(data["streams"][0]["nb_read_frames"])
