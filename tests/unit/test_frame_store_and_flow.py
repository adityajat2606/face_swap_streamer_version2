from __future__ import annotations

import cv2
import numpy as np
import pytest

from face_swap.errors import InputError
from face_swap.frame_store import FrameStore
from face_swap.stabilizer.flow import affine_flow_from_landmarks


def _write_png_frames(d, n, rng):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(d / f"frame_{i:06d}.png"), img)


def test_png_iter_count(tmp_path, rng):
    _write_png_frames(tmp_path / "f", 5, rng)
    store = FrameStore("png", tmp_path / "f")
    assert sum(1 for _ in store.iter_frames()) == 5


def test_png_iter_resumable(tmp_path, rng):
    _write_png_frames(tmp_path / "f", 5, rng)
    store = FrameStore("png", tmp_path / "f")
    idxs = [i for i, _ in store.iter_frames(start=2)]
    assert idxs == [2, 3, 4]


def test_get_specific_frame(tmp_path, rng):
    _write_png_frames(tmp_path / "f", 3, rng)
    store = FrameStore("png", tmp_path / "f")
    frame = store.get(1)
    assert frame.shape == (32, 32, 3)


def test_unknown_backend_raises(tmp_path):
    with pytest.raises(InputError):
        FrameStore("webm", tmp_path)


def test_write_debug(tmp_path, rng):
    store = FrameStore("png", tmp_path, debug_dir=tmp_path / "dbg")
    img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    out = store.write_debug(7, "mask", img)
    assert out.exists() and "frame_000007_mask" in out.name


def test_write_debug_without_dir_raises(tmp_path, rng):
    store = FrameStore("png", tmp_path)
    with pytest.raises(InputError):
        store.write_debug(0, "x", np.zeros((4, 4, 3), np.uint8))


def test_affine_flow_identity_is_near_zero():
    lm = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], np.float32)
    flow = affine_flow_from_landmarks(lm, lm, (20, 20))
    assert flow.shape == (2, 20, 20)
    assert np.abs(flow).max() < 1e-3


def test_affine_flow_translation():
    lm_a = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], np.float32)
    lm_b = lm_a + np.array([5.0, 0.0], np.float32)
    flow = affine_flow_from_landmarks(lm_a, lm_b, (20, 20))
    # x-component of flow ≈ +5 everywhere
    assert abs(flow[0].mean() - 5.0) < 0.5
