"""Flicker Score — CLAUDE.md §15 / PRD §FR-9.

``FlickerScore(N) = w1·EmbeddingDelta + w2·ColorHistDelta + w3·LandmarkResidual
                    + w4·MaskBoundaryDelta + w5·SharpnessDelta``

All terms are normalized to ``[0, 1]`` and computed on the swapped face region
of frame N vs. frame N-1 *after optical-flow motion compensation* (§15.3).
This module knows nothing about the swap engine — it takes crops and returns
numbers (CLAUDE.md §4.2).

Calibration: weights are calibrated against the human rubric (§15.4); the
default (0.30, 0.20, 0.20, 0.15, 0.15) is the pre-calibration prior. Locked
weights live in configs/final_cinematic.yaml. (calibration commit: TBD)
"""

from __future__ import annotations

import cv2
import numpy as np

COMPONENT_KEYS = ("embedding", "color", "landmark", "mask", "sharpness")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cosine distance in ``[0, 1]`` (0 = identical direction)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.clip(1.0 - a @ b, 0.0, 2.0)) / 2.0


def color_hist_delta_lab(face_a: np.ndarray, face_b: np.ndarray, bins: int = 32) -> float:
    """Chi-square distance between LAB joint histograms, normalized to ``[0,1]``."""
    lab_a = cv2.cvtColor(face_a, cv2.COLOR_BGR2LAB)
    lab_b = cv2.cvtColor(face_b, cv2.COLOR_BGR2LAB)
    h_a = cv2.calcHist(
        [lab_a], [0, 1, 2], None, [bins] * 3, [0, 256, 0, 256, 0, 256]
    ).flatten()
    h_b = cv2.calcHist(
        [lab_b], [0, 1, 2], None, [bins] * 3, [0, 256, 0, 256, 0, 256]
    ).flatten()
    h_a = h_a / (h_a.sum() + 1e-9)
    h_b = h_b / (h_b.sum() + 1e-9)
    chi2 = 0.5 * np.sum((h_a - h_b) ** 2 / (h_a + h_b + 1e-9))
    return float(np.clip(chi2, 0.0, 1.0))


def landmark_residual(lm_a: np.ndarray, lm_b: np.ndarray, bbox_diag: float) -> float:
    """Mean L2 over corresponding landmarks, normalized by bbox diagonal."""
    lm_a = np.asarray(lm_a, dtype=np.float64)
    lm_b = np.asarray(lm_b, dtype=np.float64)
    d = np.linalg.norm(lm_a - lm_b, axis=1)
    return float(np.clip(d.mean() / max(bbox_diag, 1e-3), 0.0, 1.0))


def mask_boundary_delta(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """``1 - IoU`` of dilated binary masks, in ``[0, 1]``."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    a = cv2.dilate((mask_a > 0).astype(np.uint8), kernel)
    b = cv2.dilate((mask_b > 0).astype(np.uint8), kernel)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum()) + 1e-9
    return float(np.clip(1.0 - inter / union, 0.0, 1.0))


def sharpness_delta(face_a: np.ndarray, face_b: np.ndarray, frame_mean_lap: float) -> float:
    """``|Δ Laplacian variance| / frame_mean``, normalized to ``[0, 1]``."""
    g_a = cv2.cvtColor(face_a, cv2.COLOR_BGR2GRAY)
    g_b = cv2.cvtColor(face_b, cv2.COLOR_BGR2GRAY)
    v_a = float(cv2.Laplacian(g_a, cv2.CV_64F).var())
    v_b = float(cv2.Laplacian(g_b, cv2.CV_64F).var())
    return float(np.clip(abs(v_a - v_b) / max(frame_mean_lap, 1e-3), 0.0, 1.0))


def warp_face_to_prev(face_b: np.ndarray, flow_a_to_b: np.ndarray) -> np.ndarray:
    """Warp ``face_b`` into frame A's coordinate system using optical flow.

    ``flow_a_to_b`` has shape ``(2, H, W)`` (x, y components). §15.3.
    """
    h, w = face_b.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = grid_x - flow_a_to_b[0]
    map_y = grid_y - flow_a_to_b[1]
    return cv2.remap(
        face_b, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def flicker_score(
    components: dict[str, float],
    weights: tuple[float, float, float, float, float],
) -> float:
    """Combine pre-computed components with weights. Missing components are 0."""
    w1, w2, w3, w4, w5 = weights
    return float(
        w1 * components.get("embedding", 0.0)
        + w2 * components.get("color", 0.0)
        + w3 * components.get("landmark", 0.0)
        + w4 * components.get("mask", 0.0)
        + w5 * components.get("sharpness", 0.0)
    )


def compute_components(
    *,
    face_a: np.ndarray,
    face_b: np.ndarray,
    emb_a: np.ndarray | None,
    emb_b: np.ndarray | None,
    lm_a: np.ndarray | None,
    lm_b: np.ndarray | None,
    mask_a: np.ndarray | None,
    mask_b: np.ndarray | None,
    bbox_diag: float,
    frame_mean_lap: float,
) -> dict[str, float]:
    """Compute all five Flicker components for adjacent (motion-compensated)
    face crops. ``face_a``/``face_b`` must already be size-aligned."""
    components: dict[str, float] = dict.fromkeys(COMPONENT_KEYS, 0.0)
    if emb_a is not None and emb_b is not None:
        components["embedding"] = cosine_distance(emb_a, emb_b)
    components["color"] = color_hist_delta_lab(face_a, face_b)
    if lm_a is not None and lm_b is not None:
        components["landmark"] = landmark_residual(lm_a, lm_b, bbox_diag)
    if mask_a is not None and mask_b is not None:
        components["mask"] = mask_boundary_delta(mask_a, mask_b)
    components["sharpness"] = sharpness_delta(face_a, face_b, frame_mean_lap)
    return components
