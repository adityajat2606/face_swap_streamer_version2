from __future__ import annotations

import numpy as np

from face_swap.tracker import Tracker, detect_and_fix_id_swaps
from face_swap.types import BBox, FaceDetection, Landmarks


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def _det(frame_idx, bbox, emb):
    return FaceDetection(
        frame_idx=frame_idx, bbox=bbox,
        landmarks=Landmarks(np.zeros((5, 2), np.float32)), embedding=emb,
    )


def test_two_persons_get_distinct_persistent_ids():
    tr = Tracker()
    e_a, e_b = _emb(1), _emb(2)
    for f in range(10):
        dets = [
            _det(f, BBox(0 + f, 0, 20 + f, 20, 0.99), e_a),
            _det(f, BBox(100 - f, 0, 120 - f, 20, 0.99), e_b),
        ]
        tr.update(dets, f)
    ids = {s.track_id for s in tr.active_tracks()}
    assert len(ids) == 2


def test_identity_preserved_across_crossing():
    """Two people swap spatial positions; embedding term keeps IDs stable."""
    tr = Tracker()
    e_a, e_b = _emb(10), _emb(20)
    id_a = id_b = None
    for f in range(20):
        xa = 0 + f * 5      # A moves right
        xb = 100 - f * 5    # B moves left (they cross near f=10)
        dets = [_det(f, BBox(xa, 0, xa + 20, 20, 0.99), e_a),
                _det(f, BBox(xb, 0, xb + 20, 20, 0.99), e_b)]
        tr.update(dets, f)
        if f == 0:
            # record which id owns embedding A vs B
            for s in tr.active_tracks():
                if np.dot(s.last_embedding, e_a) > np.dot(s.last_embedding, e_b):
                    id_a = s.track_id
                else:
                    id_b = s.track_id
    # after crossing, the embedding-A track is still the same id
    for s in tr.active_tracks():
        owner = id_a if np.dot(s.last_embedding, e_a) > np.dot(s.last_embedding, e_b) else id_b
        assert s.track_id == owner


def test_stale_tracks_age_out():
    tr = Tracker(max_track_gap_frames=5)
    e = _emb(3)
    tr.update([_det(0, BBox(0, 0, 10, 10, 0.9), e)], 0)
    tr.update([], 10)  # no detections for 10 frames
    assert len(tr.active_tracks()) == 0
    assert len(tr.all_tracks()) == 1  # still recorded, just inactive


def test_restore_roundtrip():
    tr = Tracker()
    e = _emb(5)
    tr.update([_det(0, BBox(0, 0, 10, 10, 0.9), e)], 0)
    states = tr.all_tracks()
    tr2 = Tracker()
    tr2.restore(states)
    assert {s.track_id for s in tr2.all_tracks()} == {s.track_id for s in states}
    # next new track gets a fresh id (no collision)
    tr2.update([_det(1, BBox(50, 50, 60, 60, 0.9), _emb(6))], 1)
    ids = [s.track_id for s in tr2.all_tracks()]
    assert len(ids) == len(set(ids))


def test_id_swap_detector_flags_crossing():
    # frame0: id0 left, id1 right; frame1: positions crossed
    f0 = {0: BBox(0, 0, 10, 10), 1: BBox(100, 0, 110, 10)}
    f1 = {0: BBox(100, 0, 110, 10), 1: BBox(0, 0, 10, 10)}
    assert detect_and_fix_id_swaps([f0, f1]) == 1


def test_no_false_swap_when_stable():
    f0 = {0: BBox(0, 0, 10, 10), 1: BBox(100, 0, 110, 10)}
    f1 = {0: BBox(1, 0, 11, 10), 1: BBox(101, 0, 111, 10)}
    assert detect_and_fix_id_swaps([f0, f1]) == 0
