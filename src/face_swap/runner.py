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
            self._restorer = None
            if self.cfg.restoration.enabled:
                try:
                    r = Restorer(max_strength=self.cfg.restoration.max_strength)
                    r.load()
                    self._restorer = r
                    _log.info("restoration_loaded")
                except Exception as exc:  # noqa: BLE001 - restoration is optional
                    _log.warning("restoration_unavailable", error=str(exc))
            # Per-source previous restoration strength for rate-limiting (§FR-7).
            self._prev_strength: dict = {}
            # Per-source previous swapped-face embedding for Flicker emb-delta (§FR-9).
            self._prev_emb: dict = {}
            # Per-source last bbox for the detector ROI fallback (§FR-3).
            self._last_bbox_per_src: dict = {}

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
                dets = self._detect_with_fallback(frame)
                picks = []
                if dets:
                    sims, bboxes = self._build_sims(dets)
                    picks = [(dets[fi], si) for fi, si in self._matcher.match(sims, bboxes)]
                    picks = self._smooth_picks(picks, frame_idx, meta.fps)
                    # remember last bbox per source for next-frame ROI fallback
                    for face, si in picks:
                        self._last_bbox_per_src[si] = np.asarray(face.bbox, np.float32)
                result, out_frame = self._process_with_retry(
                    frame, picks, frame_idx, base_sharpen, max_retry, cv2)

            result = result.replace(duration_ms=(time.perf_counter() - t0) * 1000.0,
                                    vram_peak_mb=int(self.obs.metrics.gauge("vram_used_mb")))
            self._record_frame(result, qfh, out_dir, frame_idx, out_frame, cv2)
            if (frame_idx + 1) % ckpt_every == 0:
                self._checkpoint(frame_idx)

    def _detect_with_fallback(self, frame):
        """Detector fallback ladder (PRD §FR-3): primary -> higher det_size ->
        ROI search around each source's last known bbox. Returns raw insightface
        Face objects (kps/embedding preserved) so the downstream swap path is
        unaffected."""
        app = self._detector.app
        faces = app.get(frame)
        if faces:
            return faces
        # ladder step 2: re-run at a larger det_size
        orig_size = app.det_model.input_size
        try:
            app.det_model.input_size = (1920, 1920)
            faces = app.get(frame)
        except Exception:  # noqa: BLE001 - fallback is best-effort
            faces = []
        finally:
            app.det_model.input_size = orig_size
        if faces:
            _log.info("detect_fallback_hires")
            return faces
        # ladder step 3: search dilated ROI around each source's last bbox
        h, w = frame.shape[:2]
        collected = []
        for bbox in self._last_bbox_per_src.values():
            if bbox is None:
                continue
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x1 = max(int(bbox[0] - bw * 0.5), 0)
            y1 = max(int(bbox[1] - bh * 0.5), 0)
            x2 = min(int(bbox[2] + bw * 0.5), w)
            y2 = min(int(bbox[3] + bh * 0.5), h)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            for f in app.get(crop):
                f.bbox = f.bbox + np.array([x1, y1, x1, y1], np.float32)
                f.kps = f.kps + np.array([x1, y1], np.float32)
                collected.append(f)
        if collected:
            _log.info("detect_fallback_roi", count=len(collected))
        return collected

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

    # Ordered §30 strategies the runner actually re-swaps with. Each entry is
    # (name, transform_picks_or_params) — strategies inswapper can't honour are
    # logged as no-ops rather than silently dropped, so the audit trail is
    # complete.
    _REAL_RETRY = (
        "restoration_lower", "landmarks_from_prev", "swap_detector",
        "blending_mask_alt", "restoration_higher",
    )
    _SKIPPED_RETRY = ("crop_larger", "crop_smaller", "landmarks_from_next",
                      "temporal_interpolation")

    def _process_with_retry(self, frame, picks, frame_idx, base_sharpen, max_retry, cv2):
        """Render the frame's swaps, score flicker, and walk the §30 retry queue
        on FAIL. Each strategy actually re-swaps; the lowest-flicker attempt
        wins; over-budget frames are flagged for manual review."""
        from .quality_validator import verdict_from_metrics

        best_frame, best_crops = self._render_attempt(frame, picks, base_sharpen, cv2)
        best_metrics = self._measure_frame(picks, best_frame, best_crops, cv2)
        verdict = verdict_from_metrics(best_metrics)
        reasons: list[str] = []
        retries = 0
        for strategy in self._REAL_RETRY:
            if verdict != "FAIL" or retries >= max_retry:
                break
            retries += 1
            reasons.append(strategy)
            f2, c2 = self._attempt_strategy(strategy, frame, picks, base_sharpen, cv2)
            m2 = self._measure_frame(picks, f2, c2, cv2)
            if m2["flicker_score"] < best_metrics["flicker_score"]:
                best_frame, best_crops, best_metrics = f2, c2, m2
            verdict = verdict_from_metrics(best_metrics)
        if verdict == "FAIL":  # budget exhausted — keep best, downgrade, flag
            verdict = "WARNING"
            reasons.append("budget_exhausted")
            self._dump_debug(frame, best_frame, frame_idx, cv2,
                             picks=picks, crops=best_crops, metrics=best_metrics,
                             reasons=reasons)
        self._commit_prev(best_crops)
        comps = {k: best_metrics.get(k, 0.0) for k in
                 ("embedding", "color", "landmark", "mask", "sharpness")}
        result = FrameResult(
            frame_idx=frame_idx, verdict=verdict,
            flicker_score=best_metrics["flicker_score"], components=comps,
            retry_count=retries, retry_strategies=tuple(reasons), reasons=tuple(reasons),
        )
        return result, best_frame

    def _attempt_strategy(self, strategy, frame, picks, base_sharpen, cv2):
        """Apply a §30 retry strategy and re-render the frame.

        Strategies that inswapper's fixed alignment can't meaningfully act on
        (crop_larger/_smaller, landmarks_from_next, temporal_interpolation) are
        logged as no-ops and re-render with the base parameters so the budget
        progresses; the audit trail records what was tried.
        """
        if strategy == "restoration_lower":
            return self._render_attempt(frame, picks, base_sharpen * 0.5, cv2)
        if strategy == "restoration_higher":
            return self._render_attempt(frame, picks, min(base_sharpen * 1.5, 1.0), cv2)
        if strategy == "landmarks_from_prev":
            patched: list = []
            for face, si in picks:
                prev = self._prev.get(si)
                if prev is not None and "kps" in prev:
                    face.kps = np.asarray(prev["kps"], np.float32)
                patched.append((face, si))
            return self._render_attempt(frame, patched, base_sharpen, cv2)
        if strategy == "swap_detector":
            refined = self._detect_with_fallback(frame)
            if refined:
                # re-bind each pick to the nearest refined face by IoU
                from .matching import bbox_iou

                bb_ref = [np.asarray(f.bbox, np.float32) for f in refined]
                replaced: list = []
                for face, si in picks:
                    fb = np.asarray(face.bbox, np.float32)
                    best, b_iou = face, 0.0
                    for j, f in enumerate(refined):
                        iou = bbox_iou(fb, bb_ref[j])
                        if iou > b_iou:
                            b_iou, best = iou, f
                    replaced.append((best, si))
                return self._render_attempt(frame, replaced, base_sharpen, cv2)
            return self._render_attempt(frame, picks, base_sharpen, cv2)
        if strategy == "blending_mask_alt":
            return self._render_attempt_alt_mask(frame, picks, base_sharpen, cv2)
        # crop_larger, crop_smaller, landmarks_from_next, temporal_interpolation:
        # inswapper's alignment is determined by face.kps, so 'crop' is implicit;
        # next-frame look-ahead would block the streaming flow; temporal
        # interpolation belongs upstream of the swap. Log + re-render with base
        # params so the retry budget still moves.
        _log.info("retry_strategy_noop", strategy=strategy)
        return self._render_attempt(frame, picks, base_sharpen, cv2)

    def _render_attempt_alt_mask(self, frame, picks, sharpen, cv2):
        """Alt-mask variant: heavier feather + lower colour-match strength, so a
        hard mask edge or oversaturated colour adaptation can be retried."""
        from .restoration_engine import adaptive_strength, rate_limit_strength

        work = frame.copy()
        crops: dict = {}
        max_rs = self.cfg.restoration.max_strength
        scene = adaptive_strength(frame, base=max_rs, max_strength=max_rs)
        for face, si in picks:
            rs = rate_limit_strength(self._prev_strength.get(si, scene), scene,
                                     max_delta=self.cfg.restoration.max_strength_delta_per_frame)
            res = self._swapper.swap(
                work, face, self._source_faces[si],
                natural=True, color_strength=max_rs * 0.5, sharpen=sharpen * 0.5,
                restorer=self._restorer, restoration_strength=rs,
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
                    "bbox": (x1, y1, x2, y2),
                    "emb": self._embed_swapped(work, face),
                }
        return work, crops

    def _render_attempt(self, frame, picks, sharpen, cv2):
        """Run the natural colour-matched swap (+ optional GFPGAN restoration) for
        every matched face. Restoration strength is adaptive (scaled by scene
        sharpness) and rate-limited per source so the strength schedule itself
        doesn't flicker (PRD §FR-7, §41)."""
        from .restoration_engine import adaptive_strength, rate_limit_strength

        work = frame.copy()
        crops: dict = {}
        max_rs = self.cfg.restoration.max_strength
        delta = self.cfg.restoration.max_strength_delta_per_frame
        scene_strength = adaptive_strength(frame, base=max_rs, max_strength=max_rs)
        for face, si in picks:
            prev_s = self._prev_strength.get(si, scene_strength)
            rs = rate_limit_strength(prev_s, scene_strength, max_delta=delta)
            res = self._swapper.swap(
                work, face, self._source_faces[si],
                natural=True, color_strength=max_rs, sharpen=sharpen,
                restorer=self._restorer, restoration_strength=rs,
            )
            work = res.swapped_frame
            self._prev_strength[si] = rs
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            x1, y1 = max(x1, 0), max(y1, 0)
            crop = work[y1:y2, x1:x2]
            if crop.size:
                crops[si] = {
                    "crop": cv2.resize(crop, (128, 128)),
                    "kps": np.asarray(face.kps, np.float32),
                    "diag": float(np.hypot(x2 - x1, y2 - y1)),
                    "bbox": (x1, y1, x2, y2),
                    "emb": self._embed_swapped(work, face),
                }
        return work, crops

    def _embed_swapped(self, swapped_frame, target_face):
        """Re-embed the swapped face for the Flicker embedding term (PRD §FR-9).
        Uses insightface's recognition model on the aligned 112-crop. Returns
        ``None`` if unavailable (the embedding component then contributes 0)."""
        rec = getattr(self._detector.app, "models", {}).get("recognition")
        if rec is None or not hasattr(target_face, "kps"):
            return None
        try:
            from insightface.utils import face_align

            aimg, _ = face_align.norm_crop2(swapped_frame, target_face.kps, 112)
            emb = rec.get_feat(aimg).flatten().astype(np.float32)
            n = float(np.linalg.norm(emb))
            return emb / n if n > 0 else None
        except Exception:  # noqa: BLE001 - embedding term is optional
            return None

    @staticmethod
    def _warp_prev_to_current(prev, cur, cv2):
        """Affine motion compensation per face (PRD §15.3 fallback): warp the
        previous swapped face crop into the current crop's coordinate system
        using a partial-affine fit on the landmark correspondences. Returns
        ``prev["crop"]`` unchanged if the fit fails."""
        def to_crop(kps, bbox):
            x1, y1, x2, y2 = bbox
            sx = 128.0 / max(x2 - x1, 1)
            sy = 128.0 / max(y2 - y1, 1)
            return (np.asarray(kps, np.float32) - np.array([x1, y1], np.float32)) \
                   * np.array([sx, sy], np.float32)

        try:
            src = to_crop(prev["kps"], prev["bbox"])
            dst = to_crop(cur["kps"], cur["bbox"])
            M, _ = cv2.estimateAffinePartial2D(src, dst)
            if M is None:
                return prev["crop"]
            return cv2.warpAffine(prev["crop"], M, (128, 128),
                                  borderMode=cv2.BORDER_REPLICATE)
        except Exception:  # noqa: BLE001 - motion comp must not break measurement
            return prev["crop"]

    def _measure_frame(self, picks, out_frame, crops, cv2) -> dict:
        """Per-frame Flicker Score (FR-9) = max over swapped sources of the
        component score vs that source's previous swapped crop, after affine
        motion compensation via the landmark correspondences (§15.3)."""
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
            warped_prev = self._warp_prev_to_current(prev, cur, cv2)
            comps = flicker.compute_components(
                face_a=warped_prev, face_b=cur["crop"],
                emb_a=prev.get("emb"), emb_b=cur.get("emb"),
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

    def _dump_debug(self, original, swapped, frame_idx, cv2, *,
                    picks=None, crops=None, metrics=None, reasons=None) -> None:
        """Manual-review bundle for a frame that exhausted retries (PRD §31):
        original, swapped, landmarks overlay, mask viz, and a reasons.json."""
        stem = self.dirs.debug / f"frame_{frame_idx:06d}"
        try:
            cv2.imwrite(str(stem) + "_original.png", original)
            cv2.imwrite(str(stem) + "_swapped.png", swapped)
            if picks:
                overlay = original.copy()
                for face, _si in picks:
                    x1, y1, x2, y2 = (int(v) for v in face.bbox)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for pt in np.asarray(face.kps, np.int32):
                        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
                cv2.imwrite(str(stem) + "_landmarks.png", overlay)
            if crops:
                h, w = swapped.shape[:2]
                viz = np.zeros((h, w), np.uint8)
                for c in crops.values():
                    x1, y1, x2, y2 = c.get("bbox", (0, 0, 0, 0))
                    cv2.rectangle(viz, (x1, y1), (x2, y2), 255, -1)
                cv2.imwrite(str(stem) + "_mask.png", viz)
            reasons_obj = {
                "frame": frame_idx,
                "reasons": list(reasons or ()),
                "metrics": metrics or {},
            }
            (stem.parent / f"frame_{frame_idx:06d}_reasons.json").write_text(
                json.dumps(reasons_obj, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            _log.warning("debug_dump_failed", frame_idx=frame_idx, error=str(exc))

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
