"""Temporal stabilizer — per-track smoothing of bboxes, landmarks, masks (§8).

Smooths landmarks/bboxes with per-track One-Euro filters, then re-derives the
mask from the smoothed landmarks rather than smoothing mask pixels (§8.3 — pixel
smoothing causes ghost edges). Pulls measurements from flicker.py (§11).
"""

from __future__ import annotations

import numpy as np

from .stabilizer.one_euro import VectorOneEuro
from .types import BBox, Landmarks


class Stabilizer:
    def __init__(
        self,
        *,
        one_euro_landmarks: dict[str, float] | None = None,
        one_euro_bbox: dict[str, float] | None = None,
    ):
        self._lm_params = one_euro_landmarks or {"min_cutoff": 1.0, "beta": 0.007}
        self._bbox_params = one_euro_bbox or {"min_cutoff": 1.0, "beta": 0.02}
        self._lm_filters: dict[int, VectorOneEuro] = {}
        self._bbox_filters: dict[int, VectorOneEuro] = {}

    def smooth_bbox(self, track_id: int, bbox: BBox, t: float) -> BBox:
        filt = self._bbox_filters.get(track_id)
        if filt is None:
            filt = VectorOneEuro(4, **self._bbox_params)
            self._bbox_filters[track_id] = filt
        raw = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], np.float32)
        s = filt(raw, t)
        return BBox(float(s[0]), float(s[1]), float(s[2]), float(s[3]), bbox.score)

    def smooth_landmarks(self, track_id: int, lm: Landmarks, t: float) -> Landmarks:
        pts = np.asarray(lm.points, np.float32)
        filt = self._lm_filters.get(track_id)
        if filt is None or filt.size != pts.size:
            filt = VectorOneEuro(pts.size, **self._lm_params)
            self._lm_filters[track_id] = filt
        smoothed = filt(pts.reshape(-1), t).reshape(pts.shape)
        return Landmarks(points=smoothed.astype(np.float32), score=lm.score)

    def mask_from_landmarks(self, shape: tuple[int, int], lm: Landmarks) -> np.ndarray:
        """Re-derive a soft mask from (smoothed) landmarks via convex hull (§8.3)."""
        import cv2

        h, w = shape
        mask = np.zeros((h, w), np.float32)
        pts = np.asarray(lm.points, np.float32)
        if pts.shape[0] < 3:
            return mask
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
        # dilate slightly so the hull covers the full face, then feather.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(mask, kernel)
        return cv2.GaussianBlur(dilated, (0, 0), sigmaX=5.0)

    def reset(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._lm_filters.clear()
            self._bbox_filters.clear()
        else:
            self._lm_filters.pop(track_id, None)
            self._bbox_filters.pop(track_id, None)
