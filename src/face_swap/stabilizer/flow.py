"""Optical flow (RAFT-large) for motion-compensated flicker measurement (§8.2).

torch/torchvision are imported lazily; this module imports fine on a CPU host.
RAFT-large at 1080p uses ~3-4.5 GB VRAM — run as a separate pass and unload
before the swap stage if models cannot co-reside (§10A, §11.2).
"""

from __future__ import annotations

import numpy as np


def make_raft(device: str = "cuda"):  # pragma: no cover - GPU host only
    import torch  # noqa: F401
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.DEFAULT
    return raft_large(weights=weights, progress=False).to(device).eval()


def flow(model, frame_a, frame_b):  # pragma: no cover - GPU host only
    import torch

    with torch.inference_mode():
        flows = model(frame_a.unsqueeze(0), frame_b.unsqueeze(0))
    return flows[-1].squeeze(0)  # [2, H, W]


def affine_flow_from_landmarks(
    lm_a: np.ndarray, lm_b: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Fallback dense flow (§15.3): fit an affine from landmark correspondences
    and expand it to a per-pixel flow field. Used when RAFT is unavailable."""
    import cv2

    h, w = shape
    src = np.asarray(lm_a, np.float32)
    dst = np.asarray(lm_b, np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst)
    if matrix is None:
        return np.zeros((2, h, w), np.float32)
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    ones = np.ones_like(grid_x)
    coords = np.stack([grid_x, grid_y, ones], axis=0).reshape(3, -1)
    mapped = matrix @ coords  # (2, H*W)
    mapped = mapped.reshape(2, h, w)
    flow_field = np.stack([mapped[0] - grid_x, mapped[1] - grid_y], axis=0)
    return flow_field.astype(np.float32)
