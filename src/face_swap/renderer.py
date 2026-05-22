"""Final assembly: encode frames, reattach original audio (CLAUDE.md §6.3, §6.4).

Audio is copied, never re-encoded (§18.4). FPS is matched exactly (§6.4). All
ffmpeg calls use argument lists (§18.8).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .logging_setup import get_logger

_log = get_logger("face_swap.render")

_CODEC_TO_FFMPEG = {"h264": "libx264", "h265": "libx265", "ffv1": "ffv1", "prores": "prores_ks"}


def _ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def encode_from_png_sequence(
    png_dir: Path, fps: float, out_no_audio: Path, *,
    codec: str = "h264", crf: int = 16, preset: str = "slow", ffmpeg: str | None = None,
) -> list[str]:
    """Build (and run) the encode command. Returns the argv (for testing)."""
    vcodec = _CODEC_TO_FFMPEG.get(codec, "libx264")
    cmd = [
        ffmpeg or _ffmpeg_bin(), "-y",
        "-framerate", f"{fps:.6f}",  # input rate BEFORE -i (§6.4)
        "-i", str(Path(png_dir) / "frame_%06d.png"),
        "-c:v", vcodec, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]
    if codec in ("h264", "h265"):
        cmd += ["-crf", str(crf), "-preset", preset]
    cmd.append(str(out_no_audio))
    subprocess.run(cmd, check=True)
    return cmd


def reattach_audio(
    video_no_audio: Path, original_video: Path, final_path: Path, *,
    ffmpeg: str | None = None,
) -> list[str]:
    """Mux rebuilt video with the ORIGINAL audio (copy, no re-encode)."""
    bin_ = ffmpeg or _ffmpeg_bin()
    cmd = [
        bin_, "-y",
        "-i", str(video_no_audio),
        "-i", str(original_video),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "copy", "-shortest",
        str(final_path),
    ]
    rc = subprocess.run(cmd, capture_output=True).returncode
    if rc != 0:
        # rare codec/container mismatch — fall back to AAC (§6.3) and warn.
        _log.warning("audio_copy_failed_falling_back_to_aac", rc=rc)
        cmd = [
            bin_, "-y", "-i", str(video_no_audio), "-i", str(original_video),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final_path),
        ]
        subprocess.run(cmd, check=True)
    return cmd


def side_by_side_preview(
    original: Path, swapped: Path, out_path: Path, *,
    crf: int = 18, preset: str = "medium", ffmpeg: str | None = None,
) -> list[str]:
    """2-up preview: original | swapped, half-width each (§10.2)."""
    cmd = [
        ffmpeg or _ffmpeg_bin(), "-y", "-i", str(original), "-i", str(swapped),
        "-filter_complex",
        "[0:v]scale=iw/2:ih[a];[1:v]scale=iw/2:ih[b];[a][b]hstack=inputs=2[out]",
        "-map", "[out]", "-map", "1:a:0?",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return cmd
