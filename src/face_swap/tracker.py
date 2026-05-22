"""Embedding-based tracker with bbox-IoU fallback (CLAUDE.md §7.3, §45A Q5).

Stateful: owns the track table. Per frame it scores each (track, detection)
pair as ``S = w_emb * cos_sim + (1 - w_emb) * iou``, solves assignment with the
Hungarian algorithm, rejects low-embedding matches, and ages out stale tracks.
Wrong-face prevention (§7.5): embedding term dominates; a detection whose
embedding drifts too far from the track's rolling median does not update the
median and lowers assignment confidence.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import BBox, FaceDetection, Identity, TrackState

_EMB_HISTORY_CAP = 30


class _Track:
    __slots__ = (
        "track_id",
        "identity",
        "embeddings",
        "last_bbox",
        "last_frame_seen",
        "velocity",
        "confidence",
        "active",
    )

    def __init__(self, track_id: int, det: FaceDetection):
        self.track_id = track_id
        self.identity: Identity = "unknown"
        self.embeddings: deque[np.ndarray] = deque(maxlen=_EMB_HISTORY_CAP)
        if det.embedding is not None:
            self.embeddings.append(det.embedding.astype(np.float32))
        self.last_bbox: BBox = det.bbox
        self.last_frame_seen: int = det.frame_idx
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.confidence: float = float(det.bbox.score)
        self.active: bool = True

    def median_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        stack = np.stack(self.embeddings)
        med = np.median(stack, axis=0)
        norm = np.linalg.norm(med)
        return (med / norm).astype(np.float32) if norm > 0 else med.astype(np.float32)

    def predicted_bbox(self, frame_idx: int) -> BBox:
        dt = max(frame_idx - self.last_frame_seen, 1)
        vx, vy = self.velocity
        return BBox(
            self.last_bbox.x1 + vx * dt,
            self.last_bbox.y1 + vy * dt,
            self.last_bbox.x2 + vx * dt,
            self.last_bbox.y2 + vy * dt,
            self.last_bbox.score,
        )

    def to_state(self) -> TrackState:
        med = self.median_embedding()
        return TrackState(
            track_id=self.track_id,
            identity=self.identity,
            last_embedding=med if med is not None else np.zeros(512, np.float32),
            last_bbox=self.last_bbox,
            last_frame_seen=self.last_frame_seen,
            embedding_history=tuple(self.embeddings),
            velocity=self.velocity,
            confidence=self.confidence,
            active=self.active,
        )


class Tracker:
    """Multi-face tracker producing persistent track IDs."""

    def __init__(
        self,
        *,
        embedding_weight: float = 0.70,
        embedding_match_threshold: float = 0.40,
        max_track_gap_frames: int = 30,
        consistency_drift_threshold: float = 0.25,
    ):
        self.w_emb = float(embedding_weight)
        self.match_thresh = float(embedding_match_threshold)
        self.max_gap = int(max_track_gap_frames)
        self.drift_thresh = float(consistency_drift_threshold)
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _new_track(self, det: FaceDetection) -> _Track:
        tr = _Track(self._next_id, det)
        self._tracks[self._next_id] = tr
        self._next_id += 1
        return tr

    def update(self, detections: list[FaceDetection], frame_idx: int) -> list[TrackState]:
        active = [t for t in self._tracks.values() if t.active]

        if detections and active:
            cost = np.ones((len(active), len(detections)), dtype=np.float64)
            emb_sim = np.zeros_like(cost)
            for ti, tr in enumerate(active):
                med = tr.median_embedding()
                pred = tr.predicted_bbox(frame_idx)
                for di, det in enumerate(detections):
                    if med is not None and det.embedding is not None:
                        cs = self._cos(med, det.embedding)
                    else:
                        cs = 0.0
                    iou = pred.iou(det.bbox)
                    emb_sim[ti, di] = cs
                    score = self.w_emb * cs + (1.0 - self.w_emb) * iou
                    cost[ti, di] = 1.0 - score
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_dets: set[int] = set()
            for r, c in zip(row_ind, col_ind, strict=True):
                # Reject likely different-person matches (§7.3 step 5).
                if emb_sim[r, c] < self.match_thresh and emb_sim[r, c] != 0.0:
                    continue
                self._absorb(active[r], detections[c], frame_idx, emb_sim[r, c])
                matched_dets.add(c)
            for di, det in enumerate(detections):
                if di not in matched_dets:
                    self._new_track(det)
        else:
            for det in detections:
                self._new_track(det)

        # Age out stale tracks.
        for tr in self._tracks.values():
            if frame_idx - tr.last_frame_seen > self.max_gap:
                tr.active = False

        return [t.to_state() for t in self._tracks.values()]

    def _absorb(self, tr: _Track, det: FaceDetection, frame_idx: int, emb_sim: float) -> None:
        # velocity from bbox-center motion
        old_cx, old_cy = tr.last_bbox.center
        new_cx, new_cy = det.bbox.center
        dt = max(frame_idx - tr.last_frame_seen, 1)
        tr.velocity = ((new_cx - old_cx) / dt, (new_cy - old_cy) / dt)

        # Embedding consistency: only update rolling median if not drifting (§7.5).
        med = tr.median_embedding()
        drifting = (
            med is not None
            and det.embedding is not None
            and (1.0 - self._cos(med, det.embedding)) > self.drift_thresh
        )
        if det.embedding is not None and not drifting:
            tr.embeddings.append(det.embedding.astype(np.float32))
        if drifting:
            tr.confidence = max(0.0, tr.confidence * 0.5)
        else:
            tr.confidence = float(det.bbox.score)
        tr.last_bbox = det.bbox
        tr.last_frame_seen = frame_idx
        tr.active = True

    # ---- identity wiring ----------------------------------------------
    def apply_identity_map(self, mapping: dict[int, Identity]) -> None:
        for tid, ident in mapping.items():
            if tid in self._tracks:
                self._tracks[tid].identity = ident

    def active_tracks(self) -> list[TrackState]:
        return [t.to_state() for t in self._tracks.values() if t.active]

    def all_tracks(self) -> list[TrackState]:
        return [t.to_state() for t in self._tracks.values()]

    def restore(self, states: list[TrackState]) -> None:
        """Rebuild track table from checkpointed states (§14)."""
        self._tracks.clear()
        max_id = -1
        for st in states:
            tr = _Track.__new__(_Track)
            tr.track_id = st.track_id
            tr.identity = st.identity
            tr.embeddings = deque(st.embedding_history, maxlen=_EMB_HISTORY_CAP)
            if not tr.embeddings and st.last_embedding is not None:
                tr.embeddings.append(st.last_embedding)
            tr.last_bbox = st.last_bbox
            tr.last_frame_seen = st.last_frame_seen
            tr.velocity = st.velocity
            tr.confidence = st.confidence
            tr.active = st.active
            self._tracks[st.track_id] = tr
            max_id = max(max_id, st.track_id)
        self._next_id = max_id + 1


def detect_and_fix_id_swaps(per_frame_tracks: list[dict[int, BBox]]) -> int:
    """Post-hoc pass (§7.5): count adjacent frames where two track IDs appear to
    swap spatial positions. Returns the number of suspected swaps detected.

    This is a lightweight detector used by the report generator; the actual
    flip-back is applied during rendering when a swap is confirmed.
    """
    swaps = 0
    for a, b in zip(per_frame_tracks, per_frame_tracks[1:], strict=False):
        common = set(a) & set(b)
        ids = list(common)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                # did id1 move to where id2 was and vice versa?
                d_straight = _center_dist(a[id1], b[id1]) + _center_dist(a[id2], b[id2])
                d_crossed = _center_dist(a[id1], b[id2]) + _center_dist(a[id2], b[id1])
                if d_crossed + 1e-6 < d_straight * 0.5:
                    swaps += 1
    return swaps


def _center_dist(a: BBox, b: BBox) -> float:
    ax, ay = a.center
    bx, by = b.center
    return float(np.hypot(ax - bx, ay - by))
