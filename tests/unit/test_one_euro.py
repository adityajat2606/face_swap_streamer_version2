from __future__ import annotations

import numpy as np
import pytest

from face_swap.stabilizer.one_euro import OneEuroFilter, VectorOneEuro


def test_first_sample_passthrough():
    f = OneEuroFilter()
    assert f(5.0, 0.0) == 5.0


def test_rejects_noise_tracks_step():
    """Step input + sinusoidal noise: filter rejects noise, tracks the step
    within 5 frames (CLAUDE.md §8.5)."""
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    out = []
    for i in range(60):
        t = i / 30.0
        step = 0.0 if i < 30 else 10.0
        noise = 0.5 * np.sin(i * 3.0) + rng.normal(0, 0.1)
        out.append(f(step + noise, t))
    # before step: stays near 0 (noise rejected)
    assert abs(np.mean(out[20:30])) < 1.0
    # within 5 frames after step: reaches most of the way to 10
    assert out[34] > 7.0


def test_positive_cutoff_required():
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0.0)


def test_reset_clears_state():
    f = OneEuroFilter()
    f(1.0, 0.0)
    f(2.0, 0.1)
    f.reset()
    assert f(9.0, 0.0) == 9.0


def test_vector_filter_shape_preserved():
    v = VectorOneEuro(4)
    out = v(np.array([1, 2, 3, 4], np.float32), 0.0)
    assert out.shape == (4,)
    assert np.allclose(out, [1, 2, 3, 4])


def test_vector_filter_wrong_size_raises():
    v = VectorOneEuro(4)
    with pytest.raises(ValueError):
        v(np.array([1, 2, 3]), 0.0)


def test_vector_preserves_2d_shape():
    v = VectorOneEuro(6)
    pts = np.arange(6, dtype=np.float32).reshape(3, 2)
    out = v(pts, 0.0)
    assert out.shape == (3, 2)
