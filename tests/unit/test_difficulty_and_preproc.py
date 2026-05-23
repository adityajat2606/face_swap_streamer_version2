from __future__ import annotations

import numpy as np

from face_swap.difficulty import frame_difficulty, retry_budget_for
from face_swap.reference_preproc import normalize_lighting, preprocess_reference, white_balance


def test_easy_frame_low_difficulty():
    d = frame_difficulty(detection_score=0.99, landmark_score=0.95,
                         yaw_deg=5, pitch_deg=2, motion_px=2, bbox_diag=200,
                         laplacian_var=250, occlusion=0.0)
    assert d < 0.15


def test_hard_frame_high_difficulty():
    d = frame_difficulty(detection_score=0.55, landmark_score=0.50,
                         yaw_deg=45, pitch_deg=25, motion_px=120, bbox_diag=200,
                         laplacian_var=20, occlusion=0.4)
    assert d > 0.6


def test_bounded_to_01():
    d = frame_difficulty(detection_score=-1, landmark_score=2,
                         yaw_deg=999, pitch_deg=999, motion_px=1e9, bbox_diag=1,
                         laplacian_var=-100, occlusion=10)
    assert 0.0 <= d <= 1.0


def test_retry_budget_scales_with_difficulty():
    assert retry_budget_for(0.0, 4) == 4
    assert retry_budget_for(1.0, 4) == 6   # 4 * 1.5
    assert retry_budget_for(0.5, 4) == 5


def test_white_balance_neutralizes_tint():
    # warm-tinted neutral image (blue lower)
    img = np.full((32, 32, 3), (60, 130, 200), np.uint8)  # B,G,R
    wb = white_balance(img)
    means = wb.reshape(-1, 3).mean(0)
    spread_before = float(np.std(img.reshape(-1, 3).mean(0)))
    spread_after = float(np.std(means))
    assert spread_after < spread_before


def test_normalize_lighting_preserves_shape():
    img = np.full((48, 48, 3), 100, np.uint8)
    out = normalize_lighting(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_preprocess_reference_no_op_flags():
    img = np.full((32, 32, 3), 100, np.uint8)
    out = preprocess_reference(img, balance=False, lighting=False)
    assert np.array_equal(out, img)
