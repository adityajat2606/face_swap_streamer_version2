"""Per-frame difficulty score (PRD §18).

Composite ``[0, 1]`` score combining detection confidence, pose extremity, motion
magnitude, blur (Laplacian variance), and occlusion proxy. Higher = harder.
Drives the retry budget routing — hard frames should get more aggressive
strategies. Pure numpy, fully unit-tested.
"""

from __future__ import annotations

import numpy as np

# Weighted contributions (PRD §18 list, normalized). Tune on the 200-frame
# calibration set alongside the Flicker Score weights (§15.4).
W_DETECTION = 0.25
W_LANDMARK = 0.15
W_POSE = 0.20
W_MOTION = 0.15
W_BLUR = 0.15
W_OCCLUSION = 0.10


def _norm(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def frame_difficulty(
    *,
    detection_score: float = 1.0,
    landmark_score: float = 1.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    motion_px: float = 0.0,
    bbox_diag: float = 1.0,
    laplacian_var: float = 200.0,
    occlusion: float = 0.0,
) -> float:
    """Compute a frame's difficulty in ``[0, 1]``.

    * ``detection_score`` / ``landmark_score`` in ``[0, 1]`` — lower = harder.
    * ``yaw_deg`` / ``pitch_deg`` — pose magnitudes; |yaw|+|pitch| ≥ 60° is hard.
    * ``motion_px`` / ``bbox_diag`` — face displacement normalized by face size;
      > 0.5 (half-face shift) is hard.
    * ``laplacian_var`` — sharper frame = higher variance; < 50 is blurry.
    * ``occlusion`` in ``[0, 1]`` — directly contributes.
    """
    det_d = 1.0 - float(np.clip(detection_score, 0.0, 1.0))
    lm_d = 1.0 - float(np.clip(landmark_score, 0.0, 1.0))
    pose_d = _norm(abs(yaw_deg) + abs(pitch_deg), 15.0, 60.0)
    motion_d = _norm(motion_px / max(bbox_diag, 1.0), 0.05, 0.5)
    blur_d = _norm(150.0 - laplacian_var, 0.0, 150.0)  # var<50 -> 1.0
    occ_d = float(np.clip(occlusion, 0.0, 1.0))
    score = (
        W_DETECTION * det_d
        + W_LANDMARK * lm_d
        + W_POSE * pose_d
        + W_MOTION * motion_d
        + W_BLUR * blur_d
        + W_OCCLUSION * occ_d
    )
    return float(np.clip(score, 0.0, 1.0))


def retry_budget_for(difficulty: float, base_budget: int, *, scale: float = 1.5) -> int:
    """Scale the retry budget up to ``scale x base_budget`` as difficulty -> 1."""
    return int(round(base_budget * (1.0 + (scale - 1.0) * float(np.clip(difficulty, 0, 1)))))
