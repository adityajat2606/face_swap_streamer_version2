"""Aggregate metric helpers for baseline measurement and reporting (§5.5, §17).

Pure numpy/cv2 — no model loads. The per-frame Flicker components live in
flicker.py; this module aggregates run-level KPIs (PRD §4A / §35A).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

import numpy as np


def detection_success_rate(n_detections: int, n_frames: int) -> float:
    """Fraction of frames with ≥1 detected face."""
    if n_frames <= 0:
        return 0.0
    return float(np.clip(n_detections / n_frames, 0.0, 1.0))


def frame_failure_rate(n_failed: int, n_frames: int) -> float:
    if n_frames <= 0:
        return 0.0
    return float(np.clip(n_failed / n_frames, 0.0, 1.0))


def identity_cosine_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine distance in ``[0, 2]`` between two embeddings (lower = closer)."""
    a = np.asarray(emb_a, np.float64)
    b = np.asarray(emb_b, np.float64)
    a /= np.linalg.norm(a) + 1e-9
    b /= np.linalg.norm(b) + 1e-9
    return float(np.clip(1.0 - a @ b, 0.0, 2.0))


def identity_drift_max_window(
    embeddings: Sequence[np.ndarray], reference: np.ndarray, window: int = 100
) -> float:
    """Max change in identity distance over any sliding window (PRD §4A:
    identity drift < 0.05 over any 100-frame window)."""
    if len(embeddings) < 2:
        return 0.0
    dists = [identity_cosine_distance(e, reference) for e in embeddings]
    worst = 0.0
    for i in range(len(dists)):
        j = min(i + window, len(dists))
        seg = dists[i:j]
        worst = max(worst, max(seg) - min(seg))
    return float(worst)


def percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    rank = pct / 100.0 * (len(vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def parse_nvidia_smi_util(log_text: str) -> float:
    """Average GPU utilization (0..1) from an ``nvidia-smi`` util log.

    Accepts lines containing a percentage like ``42 %`` or CSV ``42``.
    """
    utils: list[float] = []
    for line in log_text.splitlines():
        line = line.strip().rstrip("%").strip()
        if not line:
            continue
        try:
            val = float(line.split(",")[0].strip().rstrip("%").strip())
        except ValueError:
            continue
        utils.append(val)
    if not utils:
        return 0.0
    return float(np.clip(statistics.mean(utils) / 100.0, 0.0, 1.0))


def color_shift_delta_e(face_a: np.ndarray, face_b: np.ndarray) -> float:
    """Mean CIE76 ΔE between two BGR face crops (proxy for color drift)."""
    import cv2

    lab_a = cv2.cvtColor(face_a, cv2.COLOR_BGR2LAB).astype(np.float64)
    lab_b = cv2.cvtColor(face_b, cv2.COLOR_BGR2LAB).astype(np.float64)
    if lab_a.shape != lab_b.shape:
        lab_b = cv2.resize(lab_b, (lab_a.shape[1], lab_a.shape[0])).astype(np.float64)
    de = np.sqrt(((lab_a - lab_b) ** 2).sum(axis=2))
    return float(de.mean())
