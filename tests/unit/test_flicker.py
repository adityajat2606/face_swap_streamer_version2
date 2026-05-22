from __future__ import annotations

import cv2
import numpy as np

from face_swap import flicker

W = (0.30, 0.20, 0.20, 0.15, 0.15)


def test_identical_crops_zero_flicker(synthetic_face):
    f = synthetic_face
    comps = flicker.compute_components(
        face_a=f, face_b=f.copy(), emb_a=np.ones(8), emb_b=np.ones(8),
        lm_a=np.zeros((5, 2)), lm_b=np.zeros((5, 2)),
        mask_a=np.ones((256, 256), np.uint8), mask_b=np.ones((256, 256), np.uint8),
        bbox_diag=100.0, frame_mean_lap=50.0,
    )
    assert all(abs(v) < 1e-6 for v in comps.values())
    assert flicker.flicker_score(comps, W) < 1e-6


def test_blur_raises_sharpness(synthetic_face):
    f = synthetic_face
    blurred = cv2.GaussianBlur(f, (0, 0), sigmaX=5)
    s = flicker.sharpness_delta(f, blurred, frame_mean_lap=100.0)
    assert s > 0.3


def test_different_people_raise_embedding():
    a = np.array([1.0, 0, 0, 0], np.float32)
    b = np.array([0, 1.0, 0, 0], np.float32)
    assert flicker.cosine_distance(a, b) > 0.4


def test_cosine_distance_bounds():
    a = np.array([1.0, 0.0])
    assert flicker.cosine_distance(a, a) < 1e-6
    assert abs(flicker.cosine_distance(a, -a) - 1.0) < 1e-6


def test_landmark_residual_normalized():
    lm_a = np.zeros((5, 2))
    lm_b = np.ones((5, 2)) * 10
    r = flicker.landmark_residual(lm_a, lm_b, bbox_diag=100.0)
    assert 0.0 < r < 1.0


def test_mask_boundary_delta_disjoint():
    a = np.zeros((50, 50), np.uint8)
    b = np.zeros((50, 50), np.uint8)
    a[:10, :10] = 1
    b[40:, 40:] = 1
    assert flicker.mask_boundary_delta(a, b) == 1.0


def test_mask_boundary_delta_identical():
    m = np.zeros((50, 50), np.uint8)
    m[10:20, 10:20] = 1
    assert flicker.mask_boundary_delta(m, m) < 1e-6


def test_color_hist_delta_range(synthetic_face):
    other = np.zeros_like(synthetic_face)
    d = flicker.color_hist_delta_lab(synthetic_face, other)
    assert 0.0 <= d <= 1.0


def test_flicker_score_weighted_sum():
    comps = {"embedding": 1.0, "color": 0.0, "landmark": 0.0, "mask": 0.0, "sharpness": 0.0}
    assert abs(flicker.flicker_score(comps, W) - 0.30) < 1e-9


def test_warp_identity_flow_is_noop(synthetic_face):
    flow = np.zeros((2, 256, 256), np.float32)
    out = flicker.warp_face_to_prev(synthetic_face, flow)
    assert np.array_equal(out, synthetic_face)
