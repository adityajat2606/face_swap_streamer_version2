# File inventory — Cinematic Face Swap Stabilization Engine

Milestone 1 (§5.1) inventory of the new `src/face_swap/` package. Every file is
under the 600-line modularity limit (§21.4 — largest is `observability.py`).

| File | Lines |
| --- | --- |
| `src/face_swap/__init__.py` | 18 |
| `src/face_swap/__main__.py` | 8 |
| `src/face_swap/_dll.py` | 32 |
| `src/face_swap/baseline.py` | 75 |
| `src/face_swap/checkpoint.py` | 188 |
| `src/face_swap/cli.py` | 119 |
| `src/face_swap/config.py` | 185 |
| `src/face_swap/errors.py` | 76 |
| `src/face_swap/face_detector.py` | 152 |
| `src/face_swap/flicker.py` | 129 |
| `src/face_swap/frame_store.py` | 112 |
| `src/face_swap/gpu_telemetry.py` | 130 |
| `src/face_swap/identity_manager.py` | 81 |
| `src/face_swap/logging_setup.py` | 53 |
| `src/face_swap/manifest.py` | 84 |
| `src/face_swap/metrics.py` | 99 |
| `src/face_swap/observability.py` | 362 |
| `src/face_swap/pipeline.py` | 252 |
| `src/face_swap/quality_validator.py` | 165 |
| `src/face_swap/renderer.py` | 83 |
| `src/face_swap/report_generator.py` | 136 |
| `src/face_swap/restoration_engine.py` | 80 |
| `src/face_swap/runner.py` | 268 |
| `src/face_swap/stabilizer/__init__.py` | 1 |
| `src/face_swap/stabilizer/flow.py` | 48 |
| `src/face_swap/stabilizer/one_euro.py` | 69 |
| `src/face_swap/swap_engine.py` | 87 |
| `src/face_swap/temporal_stabilizer.py` | 68 |
| `src/face_swap/tracker.py` | 233 |
| `src/face_swap/types.py` | 137 |
| `src/face_swap/video_manager.py` | 111 |
| **TOTAL (31 files)** | **3641** |

## Pre-existing v0 pipeline (baseline — unmodified)

The legacy streamer is the v0 baseline this engine measures against. It was
**not** modified in this work (§5.6 gate). Key files: `webapp.py` (~1.9k LOC
Flask streamer), `webapp_mp.py` (multiprocessing variant), `server/` (FastAPI
rewrite), `cli/` (C++ port), `stream-swap.py`, `extract-ref.py`, `probe.py`.

### v0 model loads (§5.1.4)

| Role | File | Path | Loader |
| --- | --- | --- | --- |
| Detector + ArcFace | `webapp.py:147` | `buffalo_l` (downloaded to `~/.insightface`) | `insightface.app.FaceAnalysis` |
| Swap | `webapp.py:155` | `deep-live-cam/models/inswapper_128_fp16.onnx` | `insightface.model_zoo.get_model` |

The new engine reuses these exact models (`face_detector.py`, `swap_engine.py`)
so v0 and v1 are comparable on identical weights.
