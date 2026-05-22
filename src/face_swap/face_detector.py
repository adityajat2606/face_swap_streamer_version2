"""Face detection + embedding via InsightFace (CLAUDE.md §7.1, §11).

Wraps ``FaceAnalysis`` (RetinaFace + ArcFace + landmarks). Implements the §FR-3
fallback ladder. insightface/onnxruntime are imported lazily so importing this
module on a CPU-only host does not fail; only :meth:`Detector.load` touches the
GPU. After load, the active provider is verified — a silent CPU fallback raises
:class:`ModelLoadError` (CLAUDE.md issue #8).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .errors import ModelLoadError
from .logging_setup import get_logger
from .types import BBox, FaceDetection, Landmarks

_log = get_logger("face_swap.detector")


class Detector:
    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (1280, 1280),
        min_confidence: float = 0.70,
    ):
        self.model_pack = model_pack
        self.det_size = det_size
        self.min_confidence = min_confidence
        self.app: Any = None  # insightface FaceAnalysis; set in load()

    def load(self, ctx_id: int = 0) -> None:
        from ._dll import register_cuda_dlls

        register_cuda_dlls()
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - GPU host only
            raise ModelLoadError(f"insightface not installed: {exc}") from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(name=self.model_pack, providers=providers)
        det_thresh = float(os.getenv("FACESWAP_DET_THRESH", str(self.min_confidence)))
        app.prepare(ctx_id=ctx_id, det_size=self.det_size, det_thresh=det_thresh)
        self.app = app
        self._verify_providers()

    def _verify_providers(self) -> None:
        # Inspect a sub-model session to confirm CUDA actually loaded (issue #8).
        try:
            models = getattr(self.app, "models", {})
            for m in models.values():
                sess = getattr(m, "session", None)
                if sess is not None and sess.get_providers() == ["CPUExecutionProvider"]:
                    raise ModelLoadError(
                        "detector loaded on CPU only — CUDA failed (see §19)"
                    )
        except AttributeError:
            pass

    def _to_detection(self, f, frame_idx: int) -> FaceDetection:
        x1, y1, x2, y2 = (float(v) for v in f.bbox)
        emb = getattr(f, "normed_embedding", None)
        return FaceDetection(
            frame_idx=frame_idx,
            bbox=BBox(x1, y1, x2, y2, float(f.det_score)),
            landmarks=Landmarks(points=np.asarray(f.kps, np.float32), score=float(f.det_score)),
            embedding=np.asarray(emb, np.float32) if emb is not None else None,
            pose=tuple(f.pose) if getattr(f, "pose", None) is not None else None,
            occlusion=None,
        )

    def detect(
        self, frame_bgr: np.ndarray, frame_idx: int, *, last_bbox: BBox | None = None
    ) -> list[FaceDetection]:
        """Detect faces with the §FR-3 fallback ladder."""
        if self.app is None:
            raise ModelLoadError("Detector.load() not called")

        faces = self.app.get(frame_bgr)
        if faces:
            return [self._to_detection(f, frame_idx) for f in faces]

        # Fallback 2: bigger det_size.
        try:
            self.app.det_model.input_size = (1920, 1920)
            faces = self.app.get(frame_bgr)
        except Exception:  # noqa: BLE001 - fallback is best-effort
            faces = []
        finally:
            self.app.det_model.input_size = self.det_size
        if faces:
            _log.info("detect_fallback_hires", frame_idx=frame_idx)
            return [self._to_detection(f, frame_idx) for f in faces]

        # Fallback 3: search inside last known bbox dilated by 50%.
        if last_bbox is not None:
            crop, off = _dilated_crop(frame_bgr, last_bbox, 0.5)
            faces = self.app.get(crop)
            if faces:
                _log.info("detect_fallback_roi", frame_idx=frame_idx)
                dets = []
                for f in faces:
                    f.bbox = f.bbox + np.array([off[0], off[1], off[0], off[1]])
                    f.kps = f.kps + np.array(off)
                    dets.append(self._to_detection(f, frame_idx))
                return dets

        # Fallback 4: reuse prior bbox, flag occluded.
        if last_bbox is not None:
            _log.warning("detect_fallback_carryforward", frame_idx=frame_idx)
            return [
                FaceDetection(
                    frame_idx=frame_idx,
                    bbox=BBox(last_bbox.x1, last_bbox.y1, last_bbox.x2, last_bbox.y2, 0.0),
                    landmarks=Landmarks(points=np.zeros((5, 2), np.float32), score=0.0),
                    embedding=None,
                    occlusion=1.0,
                )
            ]
        _log.warning("detect_no_faces", frame_idx=frame_idx)
        return []

    def embed_reference(self, path: str) -> list[tuple[np.ndarray, float]]:
        """Return ``(embedding, area)`` for every face in a reference image."""
        import cv2

        if self.app is None:
            raise ModelLoadError("Detector.load() not called")
        img = cv2.imread(path)
        if img is None:
            raise ModelLoadError(f"cannot read reference image: {path}")
        out = []
        for f in self.app.get(img):
            area = float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            out.append((np.asarray(f.normed_embedding, np.float32), area))
        return out


def _dilated_crop(frame: np.ndarray, bbox: BBox, factor: float) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = frame.shape[:2]
    dx = bbox.width * factor / 2
    dy = bbox.height * factor / 2
    x1 = max(int(bbox.x1 - dx), 0)
    y1 = max(int(bbox.y1 - dy), 0)
    x2 = min(int(bbox.x2 + dx), w)
    y2 = min(int(bbox.y2 + dy), h)
    return frame[y1:y2, x1:x2].copy(), (x1, y1)
