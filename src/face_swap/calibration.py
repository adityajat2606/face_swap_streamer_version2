"""Flicker-Score weight calibration (PRD §15.4).

Fits non-negative weights of the FR-9 component vector against a human rubric:

  * y = (5 - rubric) / 4         # rubric 5 -> y=0 (no flicker), rubric 1 -> y=1
  * solve  C @ w = y  with w >= 0  (NNLS)
  * normalise w to sum to 1.
  * report Spearman ρ between predicted y and observed y.

Acceptance per PRD §15.4: Spearman ≥ 0.5. Pure scipy/numpy, CPU-testable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

COMPONENT_ORDER = ("embedding", "color", "landmark", "mask", "sharpness")


def _rubric_to_y(rubric: Sequence[float]) -> np.ndarray:
    return np.clip((5.0 - np.asarray(rubric, np.float64)) / 4.0, 0.0, 1.0)


def calibrate_weights(
    components: Sequence[dict], rubric_scores: Sequence[float]
) -> tuple[dict[str, float], float]:
    """Fit non-negative weights for the FR-9 components against a rubric.

    Returns ``(weights_dict, spearman_rho)`` where ``weights_dict`` keys are
    ``COMPONENT_ORDER`` and sum to 1.0 (or all-zero if the system is degenerate).
    """
    if len(components) != len(rubric_scores) or not components:
        raise ValueError("components and rubric_scores must have equal, non-zero length")
    C = np.array([[c.get(k, 0.0) for k in COMPONENT_ORDER] for c in components], np.float64)
    y = _rubric_to_y(rubric_scores)
    w, _ = nnls(C, y)
    s = float(w.sum())
    if s <= 0:
        weights = dict.fromkeys(COMPONENT_ORDER, 0.0)
    else:
        weights = dict(zip(COMPONENT_ORDER, (w / s).tolist(), strict=True))
    pred = C @ w
    rho, _ = spearmanr(pred, y)
    rho_f = float(rho) if not np.isnan(rho) else 0.0
    return weights, rho_f
