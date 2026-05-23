"""Reference-image preprocessing (PRD §20).

Apply to the new-identity hero/heroine photos *before* detection / embedding,
so AI-generated or oddly-lit references don't poison the swap:

* :func:`white_balance` — grey-world auto white-balance to neutralize tint.
* :func:`normalize_lighting` — CLAHE on the LAB L channel to flatten harsh
  shadows / blown highlights without changing colour.
* :func:`preprocess_reference` — convenience wrapper applying both.

Pure cv2 / numpy. CPU-testable.
"""

from __future__ import annotations

import cv2
import numpy as np


def white_balance(bgr: np.ndarray) -> np.ndarray:
    """Grey-world white balance: rescale each channel so its mean matches the
    overall luminance mean. Removes warm/cool cast from AI-generated refs."""
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = float(means.mean())
    scale = gray / means
    out = img * scale
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_lighting(bgr: np.ndarray, *, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE on the LAB L-channel to flatten extreme highlights/shadows without
    shifting hue/saturation."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_reference(bgr: np.ndarray, *, balance: bool = True, lighting: bool = True) -> np.ndarray:
    out = bgr
    if balance:
        out = white_balance(out)
    if lighting:
        out = normalize_lighting(out)
    return out
