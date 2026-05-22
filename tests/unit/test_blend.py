from __future__ import annotations

import numpy as np

from face_swap import blend


def test_color_transfer_moves_toward_reference():
    pale = np.full((64, 64, 3), (180, 170, 200), np.uint8)
    warm = np.full((64, 64, 3), (90, 120, 170), np.uint8)
    out = blend.color_transfer(pale, warm, 0.6)
    # each channel mean should move from the pale value toward the warm target
    before = pale.reshape(-1, 3).mean(0)
    after = out.reshape(-1, 3).mean(0)
    target = warm.reshape(-1, 3).mean(0)
    assert np.all(np.abs(after - target) < np.abs(before - target) + 1e-6)


def test_color_transfer_strength_zero_is_noop(rng):
    a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    out = blend.color_transfer(a, b, 0.0)
    assert np.array_equal(out, a)


def test_unsharp_noop_when_amount_zero(rng):
    a = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert np.array_equal(blend.unsharp(a, 0.0), a)


def test_unsharp_increases_local_contrast():
    img = np.zeros((32, 32, 3), np.uint8)
    img[16:, :, :] = 200  # an edge
    out = blend.unsharp(img, 1.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_feathered_mask_identity_affine():
    # identity-ish affine mapping a 64x64 crop into a 128x128 frame at origin
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    mask, IM = blend.feathered_mask(M, 64, 64, 128, 128)
    assert mask is not None
    assert mask.shape == (128, 128, 1)
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0


def test_paste_natural_changes_only_face_region():
    frame = np.full((128, 128, 3), 50, np.uint8)
    aligned = np.full((64, 64, 3), 50, np.uint8)
    swapped = np.full((64, 64, 3), 200, np.uint8)
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    out, mask = blend.paste_natural(frame, aligned, swapped, M, color=False, sharpen=0.0)
    assert out.shape == frame.shape
    # bottom-right (outside the 64x64 face region) is untouched
    assert np.allclose(out[100:, 100:], 50, atol=2)
    # top-left (inside the face region) moved toward the swapped value
    assert out[10:40, 10:40].mean() > 60
