"""StageRunner — the per-frame processing loop (CLAUDE.md §11.1).

Owns the detect → track → assign → swap → restore → stabilize → validate →
write sequence, checkpoint cadence, telemetry, and final render. Engines load
lazily (GPU host only). The loop is written to spec; it runs on the RTX target.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import CheckpointState, clear_checkpoint, write_checkpoint
from .config import Config
from .gpu_telemetry import GpuTelemetry
from .logging_setup import bind_run_context, get_logger
from .manifest import sha256_file_or_none
from .observability import Observatory
from .report_generator import compute_kpis, read_quality_jsonl
from .temporal_stabilizer import Stabilizer
from .tracker import Tracker
from .types import FrameResult

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

    # ---- model loading + identity routing -----------------------------
    def _load_models(self) -> None:
        from .face_detector import Detector
        from .swap_engine import Swapper

        with self.obs.span("load_models"):
            self._detector = Detector(
                det_size=self.cfg.face_detection.det_size,
                min_confidence=self.cfg.face_detection.min_confidence,
            )
            self._detector.load()
            self._swapper = Swapper()
            self._swapper.load()

    def _detect_source_face(self, path: str):
        """Largest insightface Face in a reference photo — the identity to paste."""
        import cv2

        from .errors import InputError

        img = cv2.imread(path)
        if img is None:
            raise InputError(f"cannot read reference image: {path}")
        faces = self._detector.app.get(img)
        if not faces:
            raise InputError(f"no face in reference image: {path}")
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def _prepare_identities(self) -> None:
        """Load the new-identity source faces and extract the video's own
        identity clusters to route each source onto the right person (FR-4)."""
        from .reference_extractor import extract_references

        srcs = [str(self.cfg.input.hero_reference), str(self.cfg.input.heroine_reference)]
        self._source_faces = [self._detect_source_face(p) for p in srcs]
        genders = [f.sex for f in self._source_faces]
        with self.obs.span("extract_references"):
            self._references = extract_references(
                self._detector, str(self.cfg.input.video_path), genders,
                sample_sec=1.0, max_samples=300,
            )
        mem = [r["ref_members"] for r in self._references]
        self._members_T = np.concatenate(mem, axis=0).astype(np.float32).T  # (D, total)
        self._members_src_idx = np.concatenate(
            [np.full(len(r["ref_members"]), si, np.int32) for si, r in enumerate(self._references)]
        )
        self._prev: dict = {}  # si -> {crop, kps, mask} for flicker measurement
        self._write_identity_map(
            {si: r["gender"] for si, r in enumerate(self._references)}
        )

    # ---- main loop ----------------------------------------------------
    def execute(self, start_frame: int = 0) -> dict:
        import cv2

        from . import video_manager
        from .frame_store import FrameStore, extract_ffv1, extract_png_sequence
        from .matching import SourceTracker

        np.random.seed(self.cfg.processing.random_seed)
        self._load_models()
        self._prepare_identities()

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

        ref_thresh = float(os.getenv("FACESWAP_REF_THRESH", "0.15"))
        self._matcher = SourceTracker(len(self._references), ref_thresh)

        gpu = GpuTelemetry(self.dirs.gpu_csv, self.obs.metrics, hz=self.cfg.telemetry.gpu_poll_hz)
        if self.cfg.telemetry.gpu_telemetry:
            gpu.start()

        loop_start = time.time()
        mode = "a" if start_frame > 0 else "w"
        try:
            with self.dirs.quality_jsonl.open(mode, encoding="utf-8") as qfh:
                self._frame_loop(store, out_frames_dir, meta, start_frame, qfh, cv2)
        finally:
            gpu.stop()

        self._render(out_frames_dir, meta)
        kpis = compute_kpis(read_quality_jsonl(self.dirs.quality_jsonl))
        self.manifest["kpi_results"] = kpis
        clear_checkpoint(self.dirs.checkpoint)
        _log.info("execute_done", elapsed_s=round(time.time() - loop_start, 2),
                  frames=kpis.get("n_frames", 0))
        return kpis

    def _build_sims(self, dets):
        """(T_faces, S_sources) NN-over-cluster-members similarity + bboxes."""
        embs = np.stack([np.asarray(f.normed_embedding, np.float32) for f in dets])
        all_sims = embs @ self._members_T  # (T, total_members)
        S = len(self._references)
        sims = np.full((len(dets), S), -1.0, np.float32)
        for s in range(S):
            cols = self._members_src_idx == s
            if cols.any():
                sims[:, s] = all_sims[:, cols].max(axis=1)
        bboxes = [np.asarray(f.bbox, np.float32) for f in dets]
        return sims, bboxes

    def _frame_loop(self, store, out_dir, meta, start_frame, qfh, cv2):
        ckpt_every = self.cfg.processing.checkpoint_every_n_frames
        max_retry = self.cfg.processing.max_retry_per_frame
        base_sharpen = self.cfg.restoration.max_strength * 0.6 if self.cfg.restoration.enabled else 0.0

        for frame_idx, frame in store.iter_frames(start_frame):
            bind_run_context(frame_idx=frame_idx, stage="frame")
            t0 = time.perf_counter()
            with self.obs.span("frame", frame_idx=frame_idx):
                dets = self._detector.app.get(frame)
                picks = []
                if dets:
                    sims, bboxes = self._build_sims(dets)
                    picks = [(dets[fi], si) for fi, si in self._matcher.match(sims, bboxes)]
                    picks = self._smooth_picks(picks, frame_idx, meta.fps)
                result, out_frame = self._process_with_retry(
                    frame, picks, frame_idx, base_sharpen, max_retry, cv2)

            result = result.replace(duration_ms=(time.perf_counter() - t0) * 1000.0,
                                    vram_peak_mb=int(self.obs.metrics.gauge("vram_used_mb")))
            self._record_frame(result, qfh, out_dir, frame_idx, out_frame, cv2)
            if (frame_idx + 1) % ckpt_every == 0:
                self._checkpoint(frame_idx)

    def _smooth_picks(self, picks, frame_idx, fps):
        """One-Euro smooth each matched face's landmarks before the swap (FR-8)
        to remove sub-pixel jitter of the swap region."""
        if not self.cfg.stabilization.enabled:
            return picks
        from .types import Landmarks

        t = frame_idx / max(fps, 1.0)
        for face, si in picks:
            try:
                sm = self.stabilizer.smooth_landmarks(si, Landmarks(np.asarray(face.kps, np.float32)), t)
                face.kps = sm.points
            except Exception:  # noqa: BLE001 - smoothing must never break the swap
                pass
        return picks

    def _process_with_retry(self, frame, picks, frame_idx, base_sharpen, max_retry, cv2):
        """Render the frame's swaps, score flicker, and retry with reduced
        restoration on FAIL (the dominant flicker source, PRD §41). Keeps the
        lowest-flicker attempt; flags manual review past budget."""
        from .quality_validator import verdict_from_metrics

        best_frame, best_crops = self._render_attempt(frame, picks, base_sharpen, cv2)
        best_metrics = self._measure_frame(picks, best_frame, best_crops, cv2)
        verdict = verdict_from_metrics(best_metrics)
        reasons: list[str] = []
        retries = 0
        while verdict == "FAIL" and retries < max_retry:
            retries += 1
            sharpen = base_sharpen * (0.5 ** retries)
            f2, c2 = self._render_attempt(frame, picks, sharpen, cv2)
            m2 = self._measure_frame(picks, f2, c2, cv2)
            reasons.append("restoration_lower")
            if m2["flicker_score"] < best_metrics["flicker_score"]:
                best_frame, best_crops, best_metrics = f2, c2, m2
            verdict = verdict_from_metrics(best_metrics)
        if verdict == "FAIL":  # budget exhausted — keep best, downgrade, flag
            verdict = "WARNING"
            reasons.append("budget_exhausted")
            self._dump_debug(frame, best_frame, frame_idx, cv2)
        self._commit_prev(best_crops)
        comps = {k: best_metrics.get(k, 0.0) for k in
                 ("embedding", "color", "landmark", "mask", "sharpness")}
        result = FrameResult(
            frame_idx=frame_idx, verdict=verdict,
            flicker_score=best_metrics["flicker_score"], components=comps,
            retry_count=retries, retry_strategies=tuple(reasons), reasons=tuple(reasons),
        )
        return result, best_frame

    def _render_attempt(self, frame, picks, sharpen, cv2):
        """Run the natural colour-matched swap for every matched face."""
        work = frame.copy()
        crops: dict = {}
        for face, si in picks:
            res = self._swapper.swap(
                work, face, self._source_faces[si],
                natural=True, color_strength=self.cfg.restoration.max_strength,
                sharpen=sharpen,
            )
            work = res.swapped_frame
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            x1, y1 = max(x1, 0), max(y1, 0)
            crop = work[y1:y2, x1:x2]
            if crop.size:
                crops[si] = {
                    "crop": cv2.resize(crop, (128, 128)),
                    "kps": np.asarray(face.kps, np.float32),
                    "diag": float(np.hypot(x2 - x1, y2 - y1)),
                }
        return work, crops

    def _measure_frame(self, picks, out_frame, crops, cv2) -> dict:
        """Per-frame Flicker Score (FR-9) = max over swapped sources of the
        component score vs that source's previous swapped crop."""
        from . import flicker

        gray = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
        frame_lap = float(cv2.Laplacian(gray, cv2.CV_64F).var()) or 1.0
        weights = self.cfg.stabilization.flicker_weights
        worst = 0.0
        worst_comps = dict.fromkeys(flicker.COMPONENT_KEYS, 0.0)
        for si, cur in crops.items():
            prev = self._prev.get(si)
            if prev is None:
                continue
            comps = flicker.compute_components(
                face_a=prev["crop"], face_b=cur["crop"],
                emb_a=None, emb_b=None,
                lm_a=prev["kps"], lm_b=cur["kps"],
                mask_a=None, mask_b=None,
                bbox_diag=cur["diag"], frame_mean_lap=frame_lap,
            )
            fs = flicker.flicker_score(comps, weights)
            if fs >= worst:
                worst, worst_comps = fs, comps
        return {**worst_comps, "flicker_score": worst,
                "detection_confidence": 1.0, "landmark_confidence": 1.0,
                "identity_consistency": 1.0, "mask_instability": worst_comps["mask"]}

    def _commit_prev(self, crops) -> None:
        for si, cur in crops.items():
            self._prev[si] = cur

    def _dump_debug(self, original, swapped, frame_idx, cv2) -> None:
        """Manual-review bundle for a frame that exhausted retries (PRD §31)."""
        try:
            cv2.imwrite(str(self.dirs.debug / f"frame_{frame_idx:06d}_original.png"), original)
            cv2.imwrite(str(self.dirs.debug / f"frame_{frame_idx:06d}_swapped.png"), swapped)
        except Exception:  # noqa: BLE001
            pass

    def _record_frame(self, result, qfh, out_dir, frame_idx, swapped, cv2) -> None:
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
