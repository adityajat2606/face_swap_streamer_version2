from __future__ import annotations

import numpy as np
import pytest

from face_swap.errors import AmbiguousIdentityError, InputError
from face_swap.identity_manager import assign_identities, load_identity
from face_swap.types import BBox, TrackState


def _track(tid, emb):
    return TrackState(track_id=tid, identity="unknown", last_embedding=emb,
                      last_bbox=BBox(0, 0, 1, 1), last_frame_seen=0)


def test_correct_assignment():
    hero = np.array([1.0, 0, 0, 0], np.float32)
    heroine = np.array([0, 1.0, 0, 0], np.float32)
    tracks = [_track(0, np.array([0.9, 0.1, 0, 0], np.float32)),
              _track(1, np.array([0.1, 0.9, 0, 0], np.float32))]
    mapping = assign_identities(tracks, hero, heroine)
    assert mapping[0] == "hero"
    assert mapping[1] == "heroine"


def test_ambiguous_raises():
    hero = np.array([1.0, 0], np.float32)
    heroine = np.array([0.99, 0.14], np.float32)  # nearly identical references
    tracks = [_track(0, np.array([1.0, 0.0], np.float32)),
              _track(1, np.array([0.98, 0.2], np.float32))]
    with pytest.raises(AmbiguousIdentityError):
        assign_identities(tracks, hero, heroine, ambiguity_margin=0.2)


def test_empty_tracks_returns_empty():
    assert assign_identities([], np.ones(4), np.ones(4)) == {}


def test_load_identity_largest_face():
    def embed_fn(_path):
        small = (np.array([0, 1, 0], np.float32), 10.0)
        big = (np.array([1, 0, 0], np.float32), 100.0)
        return [small, big]

    emb = load_identity(embed_fn, "ref.png")
    assert np.argmax(emb) == 0  # picked the bigger face
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-6


def test_load_identity_no_face_raises():
    with pytest.raises(InputError):
        load_identity(lambda _p: [], "ref.png")
