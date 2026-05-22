"""Face swap engine — model-agnostic interface over inswapper_128 (§11).

Wraps the InsightFace inswapper, matching the proven v0 call
``swapper.get(frame, target_face, source_face, paste_back=True)``. Returns a
:class:`SwapResult` carrying the pasted frame and the soft mask. onnxruntime/
insightface are imported lazily; after load the active provider is verified
(CLAUDE.md issue #8).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .errors import ModelLoadError, SwapInferenceError
from .logging_setup import get_logger
from .types import BBox, Landmarks, SwapResult

_log = get_logger("face_swap.swap")


class Swapper:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("FACESWAP_INSWAPPER_PATH", "")
        self._swapper: Any = None

    def load(self) -> None:
        from ._dll import register_cuda_dlls

        register_cuda_dlls()
        try:
            import insightface
        except ImportError as exc:  # pragma: no cover - GPU host only
            raise ModelLoadError(f"insightface not installed: {exc}") from exc
        if not self.model_path or not os.path.isfile(self.model_path):
            raise ModelLoadError(f"inswapper model not found: {self.model_path!r}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._swapper = insightface.model_zoo.get_model(self.model_path, providers=providers)
        active = self._swapper.session.get_providers()
        _log.info("inswapper_loaded", providers=active)
        if active == ["CPUExecutionProvider"]:
            raise ModelLoadError("inswapper loaded on CPU only — CUDA failed (issue #8)")

    def swap(
        self, frame_bgr: np.ndarray, target_face, source_face, *, paste_back: bool = True
    ) -> SwapResult:
        """Swap ``source_face`` identity onto ``target_face`` in ``frame_bgr``.

        ``target_face``/``source_face`` are native insightface Face objects (the
        detector's raw outputs), as in the v0 pipeline.
        """
        if self._swapper is None:
            raise ModelLoadError("Swapper.load() not called")
        try:
            swapped = self._swapper.get(
                frame_bgr, target_face, source_face, paste_back=paste_back
            )
        except Exception as exc:  # narrow at call-site; surface as typed error
            raise SwapInferenceError(f"inswapper failed: {exc}") from exc

        bbox = target_face.bbox
        face_bbox = BBox(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                         float(getattr(target_face, "det_score", 1.0)))
        mask = self._derive_mask(frame_bgr.shape[:2], face_bbox)
        lm = getattr(target_face, "kps", None)
        landmarks = Landmarks(np.asarray(lm, np.float32), 1.0) if lm is not None else None
        return SwapResult(
            frame_idx=-1, swapped_frame=swapped, mask=mask, face_bbox=face_bbox,
            landmarks=landmarks,
        )

    @staticmethod
    def _derive_mask(shape: tuple[int, int], bbox: BBox) -> np.ndarray:
        """Soft elliptical mask over the face bbox (re-derivable from landmarks
        downstream by the stabilizer, §8.3)."""
        import cv2

        h, w = shape
        mask = np.zeros((h, w), np.float32)
        cx, cy = bbox.center
        ax = max(int(bbox.width / 2), 1)
        ay = max(int(bbox.height / 2), 1)
        cv2.ellipse(mask, (int(cx), int(cy)), (ax, ay), 0, 0, 360, 1.0, -1)
        k = max(int(min(ax, ay) / 5) * 2 + 1, 5)
        return cv2.GaussianBlur(mask, (k, k), 0)
