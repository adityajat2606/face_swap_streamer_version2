# Module audit — `src/face_swap/`

One paragraph per module: responsibility, key imports, and who calls it. This is
the Milestone 1 (§5.1) architectural map for the Cinematic Face Swap
Stabilization Engine. The engine is an **additive** package alongside the
existing v0 streamer (`webapp.py`, `webapp_mp.py`, `cli/`), which it treats as
the baseline to measure against — no v0 files were modified (§5.6 gate).

## I/O + data

- **`types.py`** — frozen value objects (`BBox`, `Landmarks`, `FaceDetection`,
  `TrackState`, `SwapResult`, `FrameResult`). Imported by nearly every module;
  the only cross-module vocabulary (§4.2). No heavy deps.
- **`config.py`** — Pydantic schema + YAML loading + dotted-key overrides +
  config hashing (§12). Called by `cli`, `pipeline`, `runner`.
- **`video_manager.py`** — ffprobe metadata (`VideoMeta`, `probe`). Pure I/O.
  Called by `runner`.
- **`frame_store.py`** — PNG / FFV1 frame extraction + uniform iterator (§6.2).
  Called by `runner`.
- **`renderer.py`** — ffmpeg encode + audio reattach + side-by-side preview
  (§6.3, §10.2). Called by `runner`.
- **`manifest.py`** — run id, file hashing, model versions, `processing_log.json`
  (§17.2). Called by `pipeline`, `runner`.
- **`checkpoint.py`** — atomic checkpoint write/load + resume guard (§14). Called
  by `runner`, `pipeline`.

## Models (lazy GPU imports)

- **`_dll.py`** — Windows CUDA/cuDNN/TensorRT DLL registration (issues #1/#9).
- **`face_detector.py`** — InsightFace `FaceAnalysis` wrapper + the §FR-3
  fallback ladder. Lazy insightface import; verifies CUDA after load (#8).
- **`swap_engine.py`** — inswapper_128 wrapper (`swap()` → `SwapResult`),
  matching the proven v0 call. Lazy import; verifies provider.
- **`restoration_engine.py`** — adaptive-strength GFPGAN wrapper + sharpness
  matching (§8.4). `match_sharpness`/`adaptive_strength` are pure cv2.
- **`stabilizer/flow.py`** — RAFT-large optical flow (§8.2) + a landmark-affine
  flow fallback. Lazy torch import.

## Logic (CPU, fully unit-tested)

- **`stabilizer/one_euro.py`** — One-Euro filter (§8.1).
- **`temporal_stabilizer.py`** — per-track bbox/landmark smoothing + mask
  re-derivation (§8.3). Uses `stabilizer.one_euro`.
- **`flicker.py`** — Flicker Score components + combiner (§15 / §FR-9). Takes
  crops, returns numbers; knows nothing of the swap engine (§4.2).
- **`tracker.py`** — embedding + IoU Hungarian tracker, id-swap detector (§7.3,
  §7.5). Uses `scipy`.
- **`identity_manager.py`** — reference embedding + hero/heroine Hungarian
  assignment with ambiguity guard (§7.2, §7.4). Pure functions.
- **`quality_validator.py`** — verdict table + strategy-queue retry (§9). Pure
  orchestration with an injected metrics function; no detect/swap imports.
- **`metrics.py`** — run-level KPI aggregation (§4A, §35A). Pure numpy/cv2.

## Observability / SRE (CPU, unit-tested)

- **`logging_setup.py`** — structlog config + run-context binding (§4.5).
- **`observability.py`** — metrics registry (counters/gauges/histograms +
  Prometheus render), span tracer, health-check registry, and
  `ReliabilityTracker` (SLO + error budget). The `Observatory` bundle threads
  through the pipeline. *(Beyond the PRD — the SRE layer.)*
- **`gpu_telemetry.py`** — background pynvml/nvidia-smi poller →
  `gpu_telemetry.csv` + gauges (§17.3). No-op on a CPU host.

## Orchestration / entry

- **`pipeline.py`** — run-dir layout, manifest, health-check wiring, exit-code
  mapping, report invocation (§11.1, §13.2). CPU-testable scaffolding.
- **`runner.py`** — `StageRunner`: the per-frame detect→track→assign→swap→
  restore→stabilize→validate→write loop, checkpoint cadence, render (§11.1).
  GPU host only.
- **`baseline.py`** — `measure_all` → `baseline/v0_metrics.json` (§5.5).
- **`cli.py`** — `click` CLI: `run`, `resume`, `assign-identities`, `baseline`,
  `health` (§13). The only module allowed to `print` / own exit codes.
