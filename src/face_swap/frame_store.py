"""Frame extraction + uniform iteration (CLAUDE.md §6.2, §11).

Two backends: PNG sequence (debuggable) and FFV1-in-MKV (lossless, ~3x smaller,
default per §45A Q3). The iterator presents one API regardless of backend and is
resumable from an arbitrary start frame.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .errors import InputError
from .logging_setup import get_logger

_log = get_logger("face_swap.frames")


def _ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def extract_ffv1(src: str, dst_mkv: Path, *, ffmpeg: str | None = None) -> None:
    """Lossless FFV1/MKV extraction (audio stripped; re-attached at render)."""
    dst_mkv = Path(dst_mkv)
    dst_mkv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg or _ffmpeg_bin(), "-y", "-i", src,
         "-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1",
         "-g", "1", "-slices", "16", "-slicecrc", "1", "-an", str(dst_mkv)],
        check=True,
    )


def extract_png_sequence(src: str, dst_dir: Path, *, ffmpeg: str | None = None) -> None:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg or _ffmpeg_bin(), "-y", "-i", src,
         "-pix_fmt", "rgb24", "-start_number", "0",
         str(dst_dir / "frame_%06d.png")],
        check=True,
    )


class FrameStore:
    """Frame access over a backend. ``write_debug`` dumps debug images."""

    def __init__(self, backend: str, location: Path, debug_dir: Path | None = None):
        if backend not in ("png", "ffv1"):
            raise InputError(f"unknown frame backend: {backend}")
        self.backend = backend
        self.location = Path(location)
        self.debug_dir = Path(debug_dir) if debug_dir else None

    def iter_frames(self, start: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_idx, BGR uint8)`` in order, resumable from ``start``."""
        if self.backend == "png":
            yield from self._iter_png(start)
        else:
            yield from self._iter_ffv1(start)

    def _iter_png(self, start: int) -> Iterator[tuple[int, np.ndarray]]:
        import cv2

        files = sorted(self.location.glob("frame_*.png"))
        for idx, fp in enumerate(files):
            if idx < start:
                continue
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                raise InputError(f"failed to read frame {fp}")
            yield idx, img

    def _iter_ffv1(self, start: int) -> Iterator[tuple[int, np.ndarray]]:
        import cv2

        cap = cv2.VideoCapture(str(self.location))
        if not cap.isOpened():
            raise InputError(f"failed to open frame store {self.location}")
        try:
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            idx = start
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
                idx += 1
        finally:
            cap.release()

    def get(self, idx: int) -> np.ndarray:
        for fidx, frame in self.iter_frames(idx):
            if fidx == idx:
                return frame
        raise IndexError(f"frame {idx} not in store")

    def write_debug(self, idx: int, name: str, img: np.ndarray) -> Path:
        import cv2

        if self.debug_dir is None:
            raise InputError("no debug_dir configured")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        out = self.debug_dir / f"frame_{idx:06d}_{name}.png"
        cv2.imwrite(str(out), img)
        return out
