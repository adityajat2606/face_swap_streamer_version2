"""Resume continuity (CLAUDE.md §9.5, §14).

These exercise the checkpoint/resume contract without a GPU: a checkpoint written
mid-run restores tracker state and the next-frame pointer, and resume is refused
on config/model/source mismatch.
"""

from __future__ import annotations

import numpy as np
import pytest

from face_swap.checkpoint import (
    CheckpointState,
    load_checkpoint,
    verify_resumable,
    write_checkpoint,
)
from face_swap.errors import ResumeError
from face_swap.tracker import Tracker
from face_swap.types import BBox, FaceDetection, Landmarks

pytestmark = pytest.mark.integration


def _det(frame_idx, bbox, emb):
    return FaceDetection(frame_idx=frame_idx, bbox=bbox,
                         landmarks=Landmarks(np.zeros((5, 2), np.float32)), embedding=emb)


def _emb(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_tracker_state_survives_checkpoint(tmp_path):
    """Track table after resume equals the table at checkpoint time."""
    tr = Tracker()
    ea, eb = _emb(1), _emb(2)
    for f in range(100):
        tr.update([_det(f, BBox(f, 0, f + 20, 20, 0.9), ea),
                   _det(f, BBox(200 - f, 0, 220 - f, 20, 0.9), eb)], f)

    state = CheckpointState(
        run_id="r", config_hash="c", source_video_hash="v",
        hero_ref_hash="h", heroine_ref_hash="hr", model_versions={"swap": "m@1"},
        last_completed_frame=99, next_frame_to_process=100, random_seed=42,
        tracks=tr.all_tracks(), cumulative_metrics={}, elapsed_seconds=5.0)
    write_checkpoint(state, tmp_path)

    loaded = load_checkpoint(tmp_path)
    tr2 = Tracker()
    tr2.restore(loaded.tracks)

    assert loaded.next_frame_to_process == 100
    before = {s.track_id: s.identity for s in tr.all_tracks()}
    after = {s.track_id: s.identity for s in tr2.all_tracks()}
    assert before == after
    # tracking continues with no id collisions
    tr2.update([_det(100, BBox(100, 0, 120, 20, 0.9), _emb(3))], 100)
    ids = [s.track_id for s in tr2.all_tracks()]
    assert len(ids) == len(set(ids))


def test_resume_refused_on_config_change(tmp_path):
    state = CheckpointState(
        run_id="r", config_hash="ORIGINAL", source_video_hash="v",
        hero_ref_hash="h", heroine_ref_hash="hr", model_versions={"swap": "m@1"},
        last_completed_frame=99, next_frame_to_process=100, random_seed=42,
        tracks=[], cumulative_metrics={}, elapsed_seconds=0.0)
    write_checkpoint(state, tmp_path)
    loaded = load_checkpoint(tmp_path)
    with pytest.raises(ResumeError):
        verify_resumable(loaded, config_hash="CHANGED", source_video_hash="v",
                         model_versions={"swap": "m@1"})
