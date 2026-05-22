# Known bugs / gaps — Milestone 1

Filed during the baseline audit (§5.2). Items here are not fixed in Milestone 1
(the audit milestone changes no pipeline code); they are tracked for later
milestones.

## Environment gaps (this dev host)

- **GPU + model weights absent on the audit host.** This machine has Python 3.13
  with only CPU deps; no torch/insightface/onnxruntime/CUDA. The engine is
  therefore built and **fully unit-tested on CPU**; the GPU-only paths
  (`runner` frame loop, `face_detector`/`swap_engine`/`restoration_engine` model
  inference, RAFT flow) are written to spec and must be exercised on the RTX
  5080 target machine. Baseline KPIs that require a real run
  (`flicker_score_*`, `gpu_util_avg`, `wall_clock_s`) are emitted as `null` in
  `baseline/v0_metrics.json` per §5.5 and filled in by the GPU-host run.
- **No test clips committed.** `tests/fixtures/clips/*` are gitignored and not
  present. `tests/fixtures/test_set_manifest.yaml` documents the five required
  sets (§36) as placeholders; the product owner must supply consented, licensed
  footage before Milestone 1's clip gate fully passes.

## Open product/eng questions (escalate per §22)

Defaults implemented; owner confirmation pending at the noted milestone:

| # | Question | Default implemented |
| --- | --- | --- |
| Q1 | Optical flow algorithm | RAFT-large (+ landmark-affine fallback) |
| Q2 | Swap model standardization | current inswapper_128 behind `Swapper` interface |
| Q3 | Intermediate storage | FFV1 (PNG selectable) |
| Q5 | Tracking algorithm | embedding + IoU Hungarian |
| Q6 | 3+ faces | top-2 by area matched, rest pass through |
| Q8 | Final codec | H.264 CRF 16 |

## Deferred to later milestones

- **Optical-flow motion compensation in the live Flicker measurement** — the
  `flicker.warp_face_to_prev` plumbing exists; wiring RAFT output into the
  per-frame metric on the GPU host is Milestone 4 work.
- **Flicker weight calibration (§15.4)** — needs 200 human-rubric-scored frames.
  Default weights `(0.30, 0.20, 0.20, 0.15, 0.15)` are the pre-calibration prior.
- **Interactive `assign-identities` resolver** — raises `NotImplementedError`;
  the GPU host renders keyframes and writes `identity_map.yaml` (Milestone 3).
- **Real swap bridging in `runner._swap_and_stabilize`** — currently a
  passthrough placeholder; the GPU host bridges insightface Face objects into
  `Swapper.swap` (Milestone 3/6).
