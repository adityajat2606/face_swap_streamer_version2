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
        self,
        frame_bgr: np.ndarray,
        target_face,
        source_face,
        *,
        natural: bool = True,
        color_strength: float = 0.6,
        sharpen: float = 0.4,
        restorer: Any = None,
        restoration_strength: float = 0.0,
    ) -> SwapResult:
        """Swap ``source_face`` identity onto ``target_face`` in ``frame_bgr``.

        ``natural=True`` colour-matches the swapped crop to the target's own
        lighting/skin tone and pastes it through a feathered mask (PRD §22/§23).
        If a ``restorer`` is supplied with ``restoration_strength > 0`` it is
        applied to the inswapper's raw 128px output and then sharpness-matched to
        the surrounding frame (PRD §FR-7, §41, Risk 4) so the restored face
        doesn't end up sharper than the rest of the video.
        """
        if self._swapper is None:
            raise ModelLoadError("Swapper.load() not called")
        bbox = target_face.bbox
        face_bbox = BBox(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                         float(getattr(target_face, "det_score", 1.0)))
        lm = getattr(target_face, "kps", None)
        landmarks = Landmarks(np.asarray(lm, np.float32), 1.0) if lm is not None else None

        try:
            if natural:
                from insightface.utils import face_align

                from . import blend
                from .restoration_engine import match_sharpness

                bgr_fake, M = self._swapper.get(
                    frame_bgr, target_face, source_face, paste_back=False
                )
                swap_size = int(self._swapper.input_size[0])
                aligned, _ = face_align.norm_crop2(frame_bgr, target_face.kps, swap_size)
                if restorer is not None and restoration_strength > 0:
                    try:
                        bgr_fake = restorer.restore(bgr_fake, restoration_strength)
                        bgr_fake = match_sharpness(bgr_fake, aligned)
                    except Exception as exc:  # noqa: BLE001 - restoration is optional
                        _log.warning("restoration_skipped", error=str(exc))
                merged, mask = blend.paste_natural(
                    frame_bgr, aligned, bgr_fake, M,
                    color=True, color_strength=color_strength, sharpen=sharpen,
                )
                if mask is None:
                    merged = self._swapper.get(
                        frame_bgr, target_face, source_face, paste_back=True
                    )
                    mask = self._derive_mask(frame_bgr.shape[:2], face_bbox)
                else:
                    mask = mask[:, :, 0]
            else:
                merged = self._swapper.get(
                    frame_bgr, target_face, source_face, paste_back=True
                )
                mask = self._derive_mask(frame_bgr.shape[:2], face_bbox)
        except SwapInferenceError:
            raise
        except Exception as exc:  # narrow at call-site; surface as typed error
            raise SwapInferenceError(f"inswapper failed: {exc}") from exc

        return SwapResult(
            frame_idx=-1, swapped_frame=merged, mask=mask, face_bbox=face_bbox,
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
