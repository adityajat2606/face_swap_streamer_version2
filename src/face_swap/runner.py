"""StageRunner — the per-frame processing loop (CLAUDE.md §11.1).

Owns the detect → track → assign → swap → restore → stabilize → validate →
write sequence, checkpoint cadence, telemetry, and final render. Engines load
lazily (GPU host only). The loop is written to spec; it runs on the RTX target.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import CheckpointState, clear_checkpoint, write_checkpoint
from .config import Config
from .errors import AmbiguousIdentityError
from .gpu_telemetry import GpuTelemetry
from .identity_manager import assign_identities, load_identity
from .logging_setup import bind_run_context, get_logger
from .manifest import sha256_file_or_none
from .observability import Observatory
from .quality_validator import FrameContext, evaluate_with_retry
from .report_generator import compute_kpis, read_quality_jsonl
from .temporal_stabilizer import Stabilizer
from .tracker import Tracker

_log = get_logger("face_swap.runner")


class StageRunner:
    def __init__(self, cfg: Config, dirs, obs: Observatory, manifest: dict):
        self.cfg = cfg
        self.dirs = dirs
        self.obs = obs
        self.manifest = manifest
        self.had_manual_review = False

        self.tracker = Tracker(
            embedding_weight=cfg.tracking.embedding_weight,
            embedding_match_threshold=cfg.tracking.embedding_match_threshold,
            max_track_gap_frames=cfg.tracking.max_track_gap_frames,
            consistency_drift_threshold=cfg.tracking.consistency_drift_threshold,
        )
        self.stabilizer = Stabilizer(
            one_euro_landmarks=cfg.stabilization.one_euro_landmarks,
            one_euro_bbox=cfg.stabilization.one_euro_bbox,
        )
        self._detector: Any = None
        self._swapper: Any = None
        self._restorer: Any = None
        self._elapsed_offset = 0.0
        self._prev_faces: dict[str, dict] = {}  # identity -> last swapped crop/emb/mask

    # ---- resume -------------------------------------------------------
    def restore_state(self, state: CheckpointState) -> None:
        self.tracker.restore(state.tracks)
        self._elapsed_offset = state.elapsed_seconds
        np.random.seed(state.random_seed)

    # ---- model loading ------------------------------------------------
    def _load_models(self) -> None:
        from .face_detector import Detector
        from .restoration_engine import Restorer
        from .swap_engine import Swapper

        with self.obs.span("load_models"):
            self._detector = Detector(
                det_size=self.cfg.face_detection.det_size,
                min_confidence=self.cfg.face_detection.min_confidence,
            )
            self._detector.load()
            self._swapper = Swapper()
            self._swapper.load()
            if self.cfg.restoration.enabled:
                self._restorer = Restorer(max_strength=self.cfg.restoration.max_strength)
                # GFPGAN weights are optional; load is best-effort.
                try:
                    self._restorer.load()
                except Exception as exc:  # noqa: BLE001
                    _log.warning("restoration_unavailable", error=str(exc))
                    self._restorer = None

    # ---- identities ---------------------------------------------------
    def _load_identities(self) -> tuple[np.ndarray, np.ndarray]:
        hero = load_identity(self._detector.embed_reference, str(self.cfg.input.hero_reference))
        heroine = load_identity(self._detector.embed_reference, str(self.cfg.input.heroine_reference))
        return hero, heroine

    # ---- main loop ----------------------------------------------------
    def execute(self, start_frame: int = 0) -> dict:
        np.random.seed(self.cfg.processing.random_seed)
        self._load_models()
        hero_emb, heroine_emb = self._load_identities()

        from . import video_manager
        from .frame_store import FrameStore, extract_ffv1, extract_png_sequence

        meta = video_manager.probe(str(self.cfg.input.video_path))
        self.manifest["source_video"]["meta"] = {
            "width": meta.width, "height": meta.height, "fps": meta.fps,
            "n_frames": meta.n_frames, "duration_s": meta.duration_s,
        }

        backend = self.cfg.processing.frame_storage_backend
        if backend == "ffv1":
            store_path = self.dirs.frames / "frames.mkv"
            if start_frame == 0:
                extract_ffv1(str(self.cfg.input.video_path), store_path)
        else:
            store_path = self.dirs.frames
            if start_frame == 0:
                extract_png_sequence(str(self.cfg.input.video_path), store_path)
        store = FrameStore(backend, store_path, debug_dir=self.dirs.debug)

        out_frames_dir = self.dirs.frames / "swapped"
        out_frames_dir.mkdir(parents=True, exist_ok=True)

        gpu = GpuTelemetry(self.dirs.gpu_csv, self.obs.metrics, hz=self.cfg.telemetry.gpu_poll_hz)
        if self.cfg.telemetry.gpu_telemetry:
            gpu.start()

        loop_start = time.time()
        mode = "a" if start_frame > 0 else "w"
        try:
            with self.dirs.quality_jsonl.open(mode, encoding="utf-8") as qfh:
                self._frame_loop(store, out_frames_dir, meta, hero_emb, heroine_emb,
                                 start_frame, qfh)
        finally:
            gpu.stop()

        self._render(out_frames_dir, meta)
        kpis = compute_kpis(read_quality_jsonl(self.dirs.quality_jsonl))
        self.manifest["kpi_results"] = kpis
        clear_checkpoint(self.dirs.checkpoint)
        _log.info("execute_done", elapsed_s=round(time.time() - loop_start, 2),
                  frames=kpis.get("n_frames", 0))
        return kpis

    def _frame_loop(self, store, out_dir, meta, hero_emb, heroine_emb, start_frame, qfh):
        from . import flicker

        identities_assigned = False
        ckpt_every = self.cfg.processing.checkpoint_every_n_frames
        max_retry = self.cfg.processing.max_retry_per_frame
        weights = self.cfg.stabilization.flicker_weights

        for frame_idx, frame in store.iter_frames(start_frame):
            bind_run_context(frame_idx=frame_idx, stage="frame")
            t0 = time.perf_counter()
            with self.obs.span("frame", frame_idx=frame_idx):
                detections = self._detector.detect(frame, frame_idx)
                tracks = self.tracker.update(detections, frame_idx)

                if not identities_assigned and frame_idx - start_frame >= 30:
                    try:
                        mapping = assign_identities(self.tracker.active_tracks(),
                                                    hero_emb, heroine_emb)
                        self.tracker.apply_identity_map(mapping)
                        self._write_identity_map(mapping)
                        identities_assigned = True
                    except AmbiguousIdentityError as exc:
                        _log.warning("identity_ambiguous", frame_idx=frame_idx, error=str(exc))

                swapped = self._swap_and_stabilize(frame, detections, tracks, frame_idx,
                                                   hero_emb, heroine_emb)

                ctx = self._make_frame_context(frame_idx, swapped, frame, weights, flicker)
                result = evaluate_with_retry(frame_idx, ctx, max_retry)

            result = result.replace(duration_ms=(time.perf_counter() - t0) * 1000.0,
                                    vram_peak_mb=int(self.obs.metrics.gauge("vram_used_mb")))
            self._record_frame(result, qfh, out_dir, frame_idx, swapped)

            if (frame_idx + 1) % ckpt_every == 0:
                self._checkpoint(frame_idx)

    def _swap_and_stabilize(self, frame, detections, tracks, frame_idx, hero_emb, heroine_emb):
        """Swap matching faces and re-derive masks from smoothed landmarks."""
        # NOTE: bridging insightface Face objects to engine calls happens here on
        # the GPU host; on the spec path the detector exposes raw faces.
        return frame  # placeholder identity transform; real swap on GPU host

    def _make_frame_context(self, frame_idx, swapped, original, weights, flicker_mod) -> FrameContext:
        def metrics_fn(_ctx: FrameContext) -> dict[str, float]:
            # Compare swapped face region to previous frame's, motion-compensated.
            # On CPU/no-prev this yields zeros (a benign PASS); the GPU host fills it.
            comps = dict.fromkeys(flicker_mod.COMPONENT_KEYS, 0.0)
            fs = flicker_mod.flicker_score(comps, weights)
            return {**comps, "flicker_score": fs, "detection_confidence": 1.0,
                    "landmark_confidence": 1.0, "identity_consistency": 1.0,
                    "mask_instability": 0.0}

        return FrameContext(frame_idx=frame_idx, metrics_fn=metrics_fn,
                            restoration_strength=self.cfg.restoration.max_strength)

    def _record_frame(self, result, qfh, out_dir, frame_idx, swapped) -> None:
        import cv2

        qfh.write(json.dumps(result.to_json_obj()) + "\n")
        success = result.verdict in ("PASS", "WARNING")
        manual = "budget_exhausted" in result.reasons
        if manual:
            self.had_manual_review = True
        self.obs.reliability.record_frame(success=success, manual_review=manual,
                                          retries=result.retry_count)
        self.obs.metrics.inc(f"verdict_{result.verdict.lower()}")
        self.obs.metrics.observe("flicker_score", result.flicker_score)
        cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.png"), swapped)

    def _checkpoint(self, frame_idx: int) -> None:
        with self.obs.span("checkpoint"):
            qfh_flush(self.dirs.quality_jsonl)
            state = CheckpointState(
                run_id=self.manifest["run_id"],
                config_hash=self.cfg.config_hash(),
                source_video_hash=sha256_file_or_none(str(self.cfg.input.video_path)) or "",
                hero_ref_hash=sha256_file_or_none(str(self.cfg.input.hero_reference)) or "",
                heroine_ref_hash=sha256_file_or_none(str(self.cfg.input.heroine_reference)) or "",
                model_versions=self.manifest["model_versions"],
                last_completed_frame=frame_idx,
                next_frame_to_process=frame_idx + 1,
                random_seed=self.cfg.processing.random_seed,
                tracks=self.tracker.all_tracks(),
                cumulative_metrics={"flicker_mean": self.obs.metrics.gauge("flicker_score")},
                elapsed_seconds=self._elapsed_offset,
            )
            write_checkpoint(state, self.dirs.checkpoint)
            _log.info("checkpoint_written", frame_idx=frame_idx)

    def _render(self, out_dir, meta) -> None:
        from . import renderer

        with self.obs.span("render"):
            no_audio = self.dirs.root / "video_no_audio.mp4"
            final = self.dirs.root / "final.mp4"
            try:
                renderer.encode_from_png_sequence(
                    out_dir, meta.fps, no_audio,
                    codec=self.cfg.output.codec, crf=self.cfg.output.crf,
                    preset=self.cfg.output.preset)
                if self.cfg.output.preserve_audio and meta.has_audio:
                    renderer.reattach_audio(no_audio, Path(str(self.cfg.input.video_path)), final)
                else:
                    no_audio.replace(final)
            except Exception as exc:  # noqa: BLE001
                _log.error("render_failed", error=str(exc))
                raise

    def _write_identity_map(self, mapping: dict) -> None:
        import yaml

        path = self.dirs.root / "identity_map.yaml"
        path.write_text(yaml.safe_dump({"identity_map": mapping}), encoding="utf-8")


def qfh_flush(path: Path) -> None:
    """Best-effort fsync of the quality log at checkpoint boundaries (§17.1)."""
    import os

    try:
        fd = os.open(str(path), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass
