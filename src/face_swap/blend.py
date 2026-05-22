"""Natural face-paste blending — colour match + feathered edge (PRD §22, §23).

Pure cv2/numpy (no GPU, no model) so it is fully unit-tested on CPU. The swap
engine calls these after running the inswapper, so the swapped face matches the
target's own lighting / skin tone / colour-grade and blends seamlessly into the
neck — instead of looking like a differently-coloured cut-out (the "visibly AI"
failure mode from PRD §3 / §22). Math is the proven recipe shared with the
production webapp.
"""

from __future__ import annotations

import cv2
import numpy as np


def color_transfer(src: np.ndarray, ref: np.ndarray, strength: float) -> np.ndarray:
    """Match ``src``'s colour statistics to ``ref`` in LAB (Reinhard transfer).

    Both BGR uint8, same HxW. ``strength`` in [0,1] blends the matched result
    back toward the raw swap so lighting is corrected without washing out the
    swapped identity (PRD §22). Returns BGR uint8.
    """
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = s.copy()
    for i in range(3):
        smean, sstd = s[..., i].mean(), s[..., i].std() + 1e-6
        rmean, rstd = r[..., i].mean(), r[..., i].std() + 1e-6
        out[..., i] = (s[..., i] - smean) * (rstd / sstd) + rmean
    matched = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if strength >= 0.999:
        return matched
    return cv2.addWeighted(matched, float(strength), src, 1.0 - float(strength), 0.0)


def unsharp(img: np.ndarray, amount: float, radius: float = 1.0) -> np.ndarray:
    """Mild unsharp mask to recover detail lost by the 128px swap. ``amount<=0``
    is a no-op. Kept gentle to avoid the "sharper than the video" look (Risk 4)."""
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0.0)


def feathered_mask(
    M: np.ndarray, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[np.ndarray | None, np.ndarray]:
    """Soft paste mask: a white aligned crop warped back into the frame, eroded
    inward and Gaussian-feathered (inswapper's own recipe). Avoids hard edges /
    halos (PRD §23). Returns ``((dst_h,dst_w,1) float32 in [0,1], inverse_affine)``
    or ``(None, IM)`` if the warped region is empty.
    """
    IM = cv2.invertAffineTransform(M)
    white = np.full((src_h, src_w), 255.0, dtype=np.float32)
    mask = cv2.warpAffine(white, IM, (dst_w, dst_h), borderValue=0.0)
    mask[mask > 20] = 255
    ys, xs = np.where(mask == 255)
    if len(ys) == 0:
        return None, IM
    msize = int(np.sqrt(max((ys.max() - ys.min()) * (xs.max() - xs.min()), 1)))
    k = max(msize // 10, 10)
    mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    k = max(msize // 20, 5)
    blur = (2 * k + 1, 2 * k + 1)
    feathered = cv2.GaussianBlur(mask, blur, 0) / 255.0
    return feathered[:, :, None], IM


def paste_natural(
    frame: np.ndarray,
    aligned_target: np.ndarray,
    swapped_crop: np.ndarray,
    M: np.ndarray,
    *,
    color: bool = True,
    color_strength: float = 0.6,
    sharpen: float = 0.4,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Colour-match + sharpen ``swapped_crop`` to ``aligned_target`` and paste it
    back into ``frame`` through a feathered mask. Geometry mirrors inswapper's
    paste exactly (same ``M`` / mask), so only the *pixels* change. Returns
    ``(merged_frame_uint8, mask)``; falls back to ``(frame, None)`` if the mask
    is empty.
    """
    crop = swapped_crop
    if color:
        crop = color_transfer(crop, aligned_target, color_strength)
    crop = unsharp(crop, sharpen)
    h, w = frame.shape[:2]
    mask, IM = feathered_mask(M, aligned_target.shape[0], aligned_target.shape[1], h, w)
    if mask is None:
        return frame, None
    warped = cv2.warpAffine(crop, IM, (w, h), borderValue=0.0)
    merged = mask * warped + (1.0 - mask) * frame.astype(np.float32)
    return merged.astype(np.uint8), mask
