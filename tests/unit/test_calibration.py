from __future__ import annotations

import numpy as np
import pytest

from face_swap.calibration import COMPONENT_ORDER, calibrate_weights


def _synthetic(rng, n=200, true_w=(0.30, 0.20, 0.20, 0.15, 0.15)):
    C = rng.random((n, 5))
    y = (C * np.asarray(true_w)).sum(axis=1) + 0.02 * rng.standard_normal(n)
    rubric = np.clip(5 - 4 * y, 1, 5)
    components = [{k: float(C[i, j]) for j, k in enumerate(COMPONENT_ORDER)} for i in range(n)]
    return components, rubric.tolist()


def test_calibrate_recovers_known_weights():
    rng = np.random.default_rng(0)
    components, rubric = _synthetic(rng)
    weights, rho = calibrate_weights(components, rubric)
    # weights normalised
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    # Spearman comfortably clears the PRD §15.4 acceptance gate
    assert rho >= 0.5
    # rough fit to the truth (allow generous slack for noise + NNLS scale)
    truth = {"embedding": 0.30, "color": 0.20, "landmark": 0.20, "mask": 0.15, "sharpness": 0.15}
    err = sum(abs(weights[k] - truth[k]) for k in COMPONENT_ORDER)
    assert err < 0.25


def test_calibrate_input_validation():
    with pytest.raises(ValueError):
        calibrate_weights([], [])
    with pytest.raises(ValueError):
        calibrate_weights([{"color": 0.1}], [1, 2])
