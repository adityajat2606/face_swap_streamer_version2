"""Source→face routing: gender-aware reference assignment + sticky temporal
matching (PRD §FR-4, §FR-5, §FR-8).

Pure numpy logic — no GPU, no face objects (works on bboxes + a similarity
matrix), so it is fully unit-tested on CPU. This is the proven routing that
keeps the female source bound to the female identity and stops the swap from
blinking on/off as the face turns.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np


def bbox_iou(a, b) -> float:
    """IoU of two ``[x1,y1,x2,y2]`` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    aa = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    bb = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    u = aa + bb - inter
    return float(inter / u) if u > 0 else 0.0


def assign_sources_to_clusters(
    source_genders: list[str], clusters: list[dict], *, pool_extra: int = 4
) -> list[int]:
    """Bind each source to a distinct video identity cluster (PRD §FR-4).

    ``clusters`` is sorted by size desc; each is a dict with ``size``, ``m_frac``,
    ``f_frac``. Maximises ``sum(source_gender_fraction(cluster)**2 * size)`` over
    an expanded candidate pool, using the SOFT gender fraction (not a noisy hard
    label). Squaring makes gender fit dominate so the (often smaller, sometimes
    majority-mislabeled) actress cluster still binds to the female source; the
    size factor avoids tiny spurious clusters. Returns a source-ordered list of
    cluster indices (into ``clusters``).
    """
    K = len(source_genders)
    if not clusters:
        return [0] * K  # caller pads
    pool_n = min(len(clusters), max(2 * K, K + pool_extra))
    pool = list(range(pool_n))

    def fit(si: int, ci: int) -> float:
        c = clusters[ci]
        frac = c["m_frac"] if source_genders[si] == "M" else c["f_frac"]
        return frac * frac * c["size"]

    best_score, best = -1.0, tuple(range(min(K, pool_n)))
    for perm in permutations(pool, min(K, pool_n)):
        score = sum(fit(si, ci) for si, ci in enumerate(perm))
        if score > best_score:
            best_score, best = score, perm
    out = list(best)
    if len(out) < K:  # fewer clusters than sources: pad with the largest
        out = (out + [0] * K)[:K]
    return out


class SourceTracker:
    """Per-source temporal matcher with hysteresis + carry-through (PRD §FR-5,
    §FR-8). Sequential-frame use.

    ``match(sims, bboxes)`` returns ``[(face_index, source_index)]`` and updates
    state. ``sims`` is ``(T_faces, S_sources)`` similarity (NN-over-members);
    ``bboxes`` are ``[x1,y1,x2,y2]`` per detected face.
    """

    def __init__(
        self,
        n_sources: int,
        ref_thresh: float,
        *,
        sticky_factor: float = 0.5,
        carry_frames: int = 5,
        iou_gate: float = 0.3,
    ):
        self.S = n_sources
        self.ref_thresh = ref_thresh
        self.sticky = max(ref_thresh * sticky_factor, 0.05)
        self.carry_frames = carry_frames
        self.iou_gate = iou_gate
        self.last_bbox: list = [None] * n_sources
        self.misses: list = [10**9] * n_sources

    def _warm(self, si: int) -> bool:
        return self.misses[si] <= self.carry_frames and self.last_bbox[si] is not None

    def match(self, sims: np.ndarray, bboxes: list[np.ndarray]) -> list[tuple[int, int]]:
        T, S = sims.shape
        face_taken = [False] * T
        src_used = [False] * S
        picks: list[tuple[int, int]] = []
        primary: dict[int, np.ndarray] = {}
        force = (T == S)  # duet: every source must be applied

        def acceptable(ti, si):
            if force:
                return True
            v = float(sims[ti, si])
            if v >= self.ref_thresh:
                return True
            if self._warm(si) and v >= self.sticky \
                    and bbox_iou(bboxes[ti], self.last_bbox[si]) >= self.iou_gate:
                return True
            return False

        # phase 1: greedy one-to-one by descending similarity, with hysteresis
        order = np.dstack(
            np.unravel_index(np.argsort(sims, axis=None)[::-1], sims.shape)
        )[0]
        for pair in order:
            ti, si = int(pair[0]), int(pair[1])
            if face_taken[ti] or src_used[si] or not acceptable(ti, si):
                continue
            picks.append((ti, si))
            face_taken[ti] = True
            src_used[si] = True
            primary[si] = bboxes[ti]
            if all(src_used) or all(face_taken):
                break

        # phase 2: spatial carry-through for a warm source the embedding lost
        for si in range(S):
            if src_used[si] or not self._warm(si):
                continue
            best_ti, best_iou = -1, self.iou_gate
            for ti in range(T):
                if face_taken[ti]:
                    continue
                iou = bbox_iou(bboxes[ti], self.last_bbox[si])
                if iou >= best_iou:
                    best_iou, best_ti = iou, ti
            if best_ti >= 0:
                picks.append((best_ti, si))
                face_taken[best_ti] = True
                src_used[si] = True
                primary[si] = bboxes[best_ti]

        # phase 3: extra faces (repeated identity / crowd) -> best source
        for ti in range(T):
            if face_taken[ti]:
                continue
            si = int(np.argmax(sims[ti]))
            if float(sims[ti, si]) >= self.ref_thresh:
                picks.append((ti, si))
                face_taken[ti] = True

        # update temporal state from the primary (one-to-one / carry) matches
        for si in range(S):
            if si in primary:
                self.last_bbox[si] = primary[si]
                self.misses[si] = 0
            else:
                self.misses[si] += 1
        return picks
