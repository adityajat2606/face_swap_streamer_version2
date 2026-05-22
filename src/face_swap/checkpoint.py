"""Checkpoint & resume (CLAUDE.md §14 / PRD §FR-11).

Atomic writes: every file is written to a ``.tmp`` sibling then ``os.replace``-d
(atomic on POSIX and Windows ≥ Vista). A power loss mid-write must never leave a
corrupt checkpoint. Scalars live in ``state.json``; embeddings/bboxes live in
``state.npz``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import ResumeError
from .types import BBox, Identity, TrackState

_SCHEMA_VERSION = 1


@dataclass
class CheckpointState:
    run_id: str
    config_hash: str
    source_video_hash: str
    hero_ref_hash: str
    heroine_ref_hash: str
    model_versions: dict[str, str]
    last_completed_frame: int
    next_frame_to_process: int
    random_seed: int
    tracks: list[TrackState] = field(default_factory=list)
    cumulative_metrics: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    # ---- serialization split ------------------------------------------
    def serializable(self) -> dict:
        track_meta = []
        for t in self.tracks:
            track_meta.append(
                {
                    "track_id": t.track_id,
                    "identity": t.identity,
                    "last_bbox": [t.last_bbox.x1, t.last_bbox.y1, t.last_bbox.x2,
                                  t.last_bbox.y2, t.last_bbox.score],
                    "last_frame_seen": t.last_frame_seen,
                    "velocity": list(t.velocity),
                    "confidence": t.confidence,
                    "active": t.active,
                    "n_embeddings": len(t.embedding_history),
                }
            )
        return {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "source_video_hash": self.source_video_hash,
            "hero_ref_hash": self.hero_ref_hash,
            "heroine_ref_hash": self.heroine_ref_hash,
            "model_versions": self.model_versions,
            "last_completed_frame": self.last_completed_frame,
            "next_frame_to_process": self.next_frame_to_process,
            "random_seed": self.random_seed,
            "cumulative_metrics": self.cumulative_metrics,
            "elapsed_seconds": self.elapsed_seconds,
            "tracks": track_meta,
        }

    def arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for t in self.tracks:
            out[f"track_{t.track_id}_last"] = np.asarray(t.last_embedding, np.float32)
            if t.embedding_history:
                out[f"track_{t.track_id}_hist"] = np.stack(
                    [np.asarray(e, np.float32) for e in t.embedding_history]
                )
        return out


def write_checkpoint(state: CheckpointState, checkpoint_dir: Path) -> None:
    """Atomically write ``state.json`` and ``state.npz``."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tmp_json = checkpoint_dir / "state.json.tmp"
    tmp_npz = checkpoint_dir / "state.npz.tmp"

    with tmp_json.open("w", encoding="utf-8") as fh:
        json.dump(state.serializable(), fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    arrays = state.arrays()
    with tmp_npz.open("wb") as fh:
        np.savez(fh, **arrays)  # type: ignore[arg-type]  # stub types kwds as bool
        fh.flush()
        os.fsync(fh.fileno())

    os.replace(tmp_json, checkpoint_dir / "state.json")
    os.replace(tmp_npz, checkpoint_dir / "state.npz")


def load_checkpoint(checkpoint_dir: Path) -> CheckpointState:
    checkpoint_dir = Path(checkpoint_dir)
    json_path = checkpoint_dir / "state.json"
    npz_path = checkpoint_dir / "state.npz"
    if not json_path.is_file() or not npz_path.is_file():
        raise ResumeError(f"no checkpoint found in {checkpoint_dir}")
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResumeError(f"corrupt checkpoint state.json: {exc}") from exc
    if meta.get("schema_version") != _SCHEMA_VERSION:
        raise ResumeError(
            f"checkpoint schema {meta.get('schema_version')} != {_SCHEMA_VERSION}"
        )
    arrays = np.load(npz_path)

    tracks: list[TrackState] = []
    for tm in meta["tracks"]:
        tid = tm["track_id"]
        last = arrays.get(f"track_{tid}_last", np.zeros(512, np.float32))
        hist_key = f"track_{tid}_hist"
        hist = tuple(arrays[hist_key]) if hist_key in arrays.files else ()
        bx = tm["last_bbox"]
        tracks.append(
            TrackState(
                track_id=tid,
                identity=_identity(tm["identity"]),
                last_embedding=np.asarray(last, np.float32),
                last_bbox=BBox(*bx),
                last_frame_seen=tm["last_frame_seen"],
                embedding_history=hist,
                velocity=(float(tm["velocity"][0]), float(tm["velocity"][1])),
                confidence=tm["confidence"],
                active=tm["active"],
            )
        )
    return CheckpointState(
        run_id=meta["run_id"],
        config_hash=meta["config_hash"],
        source_video_hash=meta["source_video_hash"],
        hero_ref_hash=meta["hero_ref_hash"],
        heroine_ref_hash=meta["heroine_ref_hash"],
        model_versions=meta["model_versions"],
        last_completed_frame=meta["last_completed_frame"],
        next_frame_to_process=meta["next_frame_to_process"],
        random_seed=meta["random_seed"],
        tracks=tracks,
        cumulative_metrics=meta["cumulative_metrics"],
        elapsed_seconds=meta["elapsed_seconds"],
    )


def verify_resumable(
    state: CheckpointState,
    *,
    config_hash: str,
    source_video_hash: str,
    model_versions: dict[str, str],
) -> None:
    """Refuse to resume on any mismatch (§14.2). Raises :class:`ResumeError`."""
    if state.config_hash != config_hash:
        raise ResumeError(
            "config changed since checkpoint; refusing to resume "
            f"(checkpoint={state.config_hash[:12]}, current={config_hash[:12]})"
        )
    if state.source_video_hash != source_video_hash:
        raise ResumeError("source video changed since checkpoint; refusing to resume")
    if state.model_versions != model_versions:
        raise ResumeError(
            "model versions changed since checkpoint; refusing to resume "
            f"(checkpoint={state.model_versions}, current={model_versions})"
        )


def clear_checkpoint(checkpoint_dir: Path) -> None:
    """Remove checkpoint files on clean success (§14.5). Keeps the run dir."""
    checkpoint_dir = Path(checkpoint_dir)
    for name in ("state.json", "state.npz"):
        p = checkpoint_dir / name
        if p.exists():
            p.unlink()


def _identity(value: str) -> Identity:
    return value if value in ("hero", "heroine", "unknown") else "unknown"  # type: ignore[return-value]
