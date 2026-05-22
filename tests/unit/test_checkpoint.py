from __future__ import annotations

import numpy as np
import pytest

from face_swap.checkpoint import (
    CheckpointState,
    clear_checkpoint,
    load_checkpoint,
    verify_resumable,
    write_checkpoint,
)
from face_swap.errors import ResumeError
from face_swap.types import BBox, TrackState


def _state(**over):
    base = {
        "run_id": "r1", "config_hash": "cfg", "source_video_hash": "vid",
        "hero_ref_hash": "h", "heroine_ref_hash": "hr",
        "model_versions": {"swap": "m@abc"}, "last_completed_frame": 99,
        "next_frame_to_process": 100, "random_seed": 42,
        "tracks": [TrackState(0, "hero", np.ones(512, np.float32), BBox(0, 0, 1, 1), 99,
                              embedding_history=(np.ones(512, np.float32),))],
        "cumulative_metrics": {"flicker": 0.05}, "elapsed_seconds": 12.5,
    }
    base.update(over)
    return CheckpointState(**base)


def test_write_load_roundtrip(tmp_path):
    s = _state()
    write_checkpoint(s, tmp_path)
    loaded = load_checkpoint(tmp_path)
    assert loaded.run_id == s.run_id
    assert loaded.next_frame_to_process == 100
    assert loaded.tracks[0].identity == "hero"
    assert np.allclose(loaded.tracks[0].last_embedding, 1.0)
    assert loaded.tracks[0].embedding_history[0].shape == (512,)


def test_atomic_no_tmp_left(tmp_path):
    write_checkpoint(_state(), tmp_path)
    assert not (tmp_path / "state.json.tmp").exists()
    assert not (tmp_path / "state.npz.tmp").exists()


def test_verify_resumable_passes_on_match():
    s = _state()
    verify_resumable(s, config_hash="cfg", source_video_hash="vid",
                     model_versions={"swap": "m@abc"})  # no raise


def test_verify_refuses_config_change():
    s = _state()
    with pytest.raises(ResumeError):
        verify_resumable(s, config_hash="DIFFERENT", source_video_hash="vid",
                         model_versions={"swap": "m@abc"})


def test_verify_refuses_model_change():
    s = _state()
    with pytest.raises(ResumeError):
        verify_resumable(s, config_hash="cfg", source_video_hash="vid",
                         model_versions={"swap": "m@OTHER"})


def test_load_missing_raises(tmp_path):
    with pytest.raises(ResumeError):
        load_checkpoint(tmp_path)


def test_clear_checkpoint(tmp_path):
    write_checkpoint(_state(), tmp_path)
    clear_checkpoint(tmp_path)
    assert not (tmp_path / "state.json").exists()
    assert not (tmp_path / "state.npz").exists()


def test_track_without_history(tmp_path):
    s = _state(tracks=[TrackState(0, "unknown", np.zeros(512, np.float32),
                                  BBox(0, 0, 1, 1), 5)])
    write_checkpoint(s, tmp_path)
    loaded = load_checkpoint(tmp_path)
    assert loaded.tracks[0].embedding_history == ()
