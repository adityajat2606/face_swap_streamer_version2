"""Face restoration with adaptive, temporally-stable strength (CLAUDE.md §8.4).

Restoration is the single largest flicker source (§41). Mitigations here:
adaptive strength scaled by scene blur, a rate-limited strength schedule, and
post-restoration sharpness matching. ``match_sharpness`` is pure cv2 and is unit
tested; the GFPGAN model load is lazy (only on a GPU host with weights).
Never use ``strength: high`` in final_cinematic (§18.7).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .logging_setup import get_logger

_log = get_logger("face_swap.restore")


def match_sharpness(restored_face: np.ndarray, source_ring: np.ndarray) -> np.ndarray:
    """Blur the restored face to match the surrounding ring's sharpness (§8.4)."""
    g_ring = (cv2.cvtColor(source_ring, cv2.COLOR_BGR2GRAY)
              if source_ring.ndim == 3 else source_ring)
    g_face = (cv2.cvtColor(restored_face, cv2.COLOR_BGR2GRAY)
              if restored_face.ndim == 3 else restored_face)
    target_var = float(cv2.Laplacian(g_ring, cv2.CV_64F).var())
    current_var = float(cv2.Laplacian(g_face, cv2.CV_64F).var())
    if current_var <= target_var * 1.2 or target_var <= 1e-6:
        return restored_face
    sigma = float(np.sqrt(np.log(current_var / target_var)))
    sigma = float(np.clip(sigma, 0.3, 2.0))
    return cv2.GaussianBlur(restored_face, (0, 0), sigmaX=sigma)


def adaptive_strength(frame_bgr: np.ndarray, *, base: float = 0.5, max_strength: float = 0.7) -> float:
    """Scale restoration by ``(1 - blur)`` so soft scenes aren't over-sharpened."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = float(np.clip(1.0 - lap_var / 500.0, 0.0, 1.0))  # heuristic normalization
    return float(np.clip(base * (1.0 - blur), 0.0, max_strength))


def rate_limit_strength(prev: float, target: float, max_delta: float = 0.05) -> float:
    """Never let restoration strength jump more than ``max_delta`` per frame."""
    delta = float(np.clip(target - prev, -max_delta, max_delta))
    return float(np.clip(prev + delta, 0.0, 1.0))


class Restorer:
    """Lazy GFPGAN wrapper. ``restore`` is a no-op passthrough until ``load``."""

    def __init__(self, model_path: str | None = None, max_strength: float = 0.7):
        self.model_path = model_path
        self.max_strength = max_strength
        self._model: Any = None

    def load(self) -> None:  # pragma: no cover - GPU host only
        from ._dll import register_cuda_dlls

        register_cuda_dlls()
        try:
            from gfpgan import GFPGANer
        except ImportError as exc:
            from .errors import ModelLoadError

            raise ModelLoadError(f"gfpgan not installed: {exc}") from exc
        self._model = GFPGANer(model_path=self.model_path, upscale=1, arch="clean",
                               channel_multiplier=2)

    def restore(self, face_bgr: np.ndarray, strength: float) -> np.ndarray:
        """Restore a face crop at the given strength (0 = passthrough)."""
        strength = float(np.clip(strength, 0.0, self.max_strength))
        if self._model is None or strength <= 0.0:
            return face_bgr
        _, _, restored = self._model.enhance(  # pragma: no cover - GPU host only
            face_bgr, has_aligned=False, only_center_face=True, paste_back=True, weight=strength
        )
        return restored if restored is not None else face_bgr
