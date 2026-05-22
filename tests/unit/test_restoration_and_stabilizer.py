from __future__ import annotations

import cv2
import numpy as np

from face_swap.restoration_engine import (
    adaptive_strength,
    match_sharpness,
    rate_limit_strength,
)
from face_swap.temporal_stabilizer import Stabilizer
from face_swap.types import BBox, Landmarks


def test_match_sharpness_blurs_oversharp_face(rng):
    ring = cv2.GaussianBlur(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), (0, 0), 5)
    sharp = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)  # very sharp noise
    out = match_sharpness(sharp, ring)
    v_in = cv2.Laplacian(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    v_out = cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert v_out < v_in


def test_match_sharpness_noop_when_already_soft():
    ring = np.full((32, 32, 3), 100, np.uint8)
    face = np.full((32, 32, 3), 100, np.uint8)
    out = match_sharpness(face, ring)
    assert np.array_equal(out, face)


def test_rate_limit_strength():
    assert rate_limit_strength(0.5, 1.0, max_delta=0.05) == 0.55
    assert rate_limit_strength(0.5, 0.0, max_delta=0.05) == 0.45
    assert rate_limit_strength(0.5, 0.52, max_delta=0.05) == 0.52


def test_adaptive_strength_bounds(rng):
    img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    s = adaptive_strength(img, base=0.5, max_strength=0.7)
    assert 0.0 <= s <= 0.7


def test_stabilizer_smooths_bbox_jitter():
    st = Stabilizer()
    out = []
    for i in range(20):
        jitter = (-1) ** i * 3.0  # alternating jitter
        b = BBox(100 + jitter, 100, 200 + jitter, 200, 0.9)
        out.append(st.smooth_bbox(0, b, t=i / 24.0))
    # later smoothed x1 should vary less than the raw ±3 jitter
    raw_span = 6.0
    smoothed_span = max(o.x1 for o in out[5:]) - min(o.x1 for o in out[5:])
    assert smoothed_span < raw_span


def test_stabilizer_landmarks_shape_preserved():
    st = Stabilizer()
    lm = Landmarks(points=np.random.rand(5, 2).astype(np.float32))
    out = st.smooth_landmarks(0, lm, t=0.0)
    assert out.points.shape == (5, 2)


def test_mask_from_landmarks_nonempty():
    st = Stabilizer()
    pts = np.array([[20, 20], [80, 20], [50, 80], [30, 60], [70, 60]], np.float32)
    mask = st.mask_from_landmarks((100, 100), Landmarks(points=pts))
    assert mask.max() > 0.0
    assert mask.shape == (100, 100)
