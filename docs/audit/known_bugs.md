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

## Engine swap pipeline — now WIRED (was placeholder)

The engine's runner is no longer a passthrough. It now does, end to end:
detect → gender-aware reference routing (`reference_extractor` +
`matching.assign_sources_to_clusters`) → sticky temporal matching
(`matching.SourceTracker`) → colour-matched feathered swap (`swap_engine` +
`blend`) → One-Euro landmark smoothing → per-frame Flicker Score (`flicker`) →
PASS/WARN/FAIL → retry with reduced restoration on FAIL → checkpoint/resume →
reports → debug dumps. The pure parts (blend, matching, clustering) are CPU
unit-tested; the GPU path is validated on the RTX box via `docs/RUNBOOK.md`.

## Still approximate / follow-up

- **Flicker `embedding` component = 0** in the live metric — computing a
  swapped-face ArcFace embedding every frame ~doubles detector cost; the other
  four components (color/landmark/mask/sharpness) are computed for real. Turn on
  if §4A identity-stability KPIs require it.
- **Retry exercises the restoration-strength strategy** (the dominant flicker
  source, §41) rather than all nine §30 strategies. The strategy table exists in
  `quality_validator`; wiring the remaining strategies to real re-swaps is a
  bounded follow-up.
- **Optical-flow motion compensation** before the Flicker metric (§15.3) —
  `flicker.warp_face_to_prev` + `stabilizer/flow` plumbing exists; RAFT wiring
  into the live metric is optional P1 work.
- **Flicker weight calibration (§15.4)** — needs 200 human-rubric-scored frames.
- **Interactive `assign-identities` resolver** — raises `NotImplementedError`;
  routing currently auto-assigns by gender + embedding clusters.

## Can only be done on the GPU box (not code — see docs/RUNBOOK.md)

- §35A baseline numbers, §4A KPI measurement, §37 human evaluation, and the
  byte-identical-resume verification all require the RTX box + real clips +
  reviewers.
