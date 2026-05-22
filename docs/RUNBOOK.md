# GPU-box runbook — running the engine & closing the PRD KPIs

Everything in this file must run on the **RTX 5080 box** with CUDA + model
weights. It can't be done on a CPU-only dev machine. The engine code is complete
and CPU-unit-tested; this runbook produces the real output and the §4A KPIs /
§35A baseline / §37 human-eval that the PRD's Definition of Done requires.

## 1. Install (one time)
```bash
uv venv --python 3.11
uv pip install -e ".[gpu,restoration]"
uv run python scripts/smoke_gpu.py            # expect capability (12, 0)
# point the engine at the inswapper weights it shares with the v0 app:
export FACESWAP_INSWAPPER_PATH=/path/to/inswapper_128_fp16.onnx
```
`buffalo_l` (detector+ArcFace) auto-downloads to `~/.insightface` on first use.

## 2. Smoke run (one short clip)
```bash
uv run face_swap run \
  --video input/videos/clip.mp4 \
  --hero  input/references/hero.png \
  --heroine input/references/heroine.png \
  --config configs/quality.yaml \
  --output output/
```
Outputs under `output/<run_id>/`:
- `final.mp4` (swapped + original audio), `quality.jsonl` (per-frame verdict + Flicker Score),
- `processing_log.json` (manifest + observability snapshot), `reports/` (summary.md, CSV, failed_frames.json),
- `gpu_telemetry.csv`, `checkpoint/`, `debug_frames/`.

Verify: faces swapped, colour matches the body, no flicker; `reports/summary.md`
shows median/p95 Flicker Score.

## 3. Resume test (FR-11 / §51)
```bash
# Ctrl+C the run after ~100 frames, then:
uv run face_swap resume output/<run_id>
```
Confirm it continues from the next frame and the run completes.

## 4. Build the v0 baseline (§35A) — required for the DoD
1. Put 1 clip per test set in `tests/fixtures/clips/` and fill paths in
   `tests/fixtures/test_set_manifest.yaml`.
2. Run the **current/older** pipeline on each clip and record its metrics into
   `baseline/v0_metrics.json` (`flat_kpis.flicker_score_p95` is what the gate
   compares against). `uv run face_swap baseline` writes the schema; populate the
   numbers from real v0 runs.
3. Commit `baseline/v0_metrics.json` (it's frozen after this).

## 5. Candidate run + delta vs baseline (§35A gate)
```bash
uv run face_swap run ... --config configs/final_cinematic.yaml \
  --output output/ --report-delta baseline/v0_metrics.json
uv run python scripts/compare_to_baseline.py output/<run_id> baseline/v0_metrics.json
# PASS requires >= 50% reduction in p95 Flicker Score (exit 0)
```

## 6. KPIs (§4A) and human eval (§37) — the remaining DoD items
- Automated KPIs come from `reports/` + `processing_log.json` across the 5 test sets:
  median Flicker < 0.08, identity drift < 0.05/100f, detection > 99% slow / > 95% fast.
- **Human rubric (§37):** 3 reviewers, blind A/B (v0 vs engine), score 1-5. Targets:
  slow ≥ 4.0, medium ≥ 3.5, fast ≥ 3.0. Record in a CSV alongside the run.

## 7. Tuning knobs (env)
| Var | Default | Use |
|---|---|---|
| `FACESWAP_DET_SIZE` | from config | 1280 for max face coverage |
| `FACESWAP_REF_THRESH` | 0.15 | lower if a face is missed |
| `restoration.max_strength` (config) | 0.7 | colour-match/sharpen strength |
| `processing.max_retry_per_frame` (config) | 5/8 | retries on FAIL frames |

## What's real vs. still approximate (be honest in the report)
- **Real now:** detect → gender-aware routing → sticky temporal matching →
  colour-matched feathered swap → One-Euro landmark smoothing → per-frame Flicker
  Score (color/landmark/sharpness/mask) → PASS/WARN/FAIL → retry with reduced
  restoration on FAIL → checkpoint/resume → reports → debug dumps.
- **Approximate / follow-up:** the Flicker `embedding` component is set to 0
  (computing a swapped-face ArcFace embedding per frame ~doubles detector cost);
  retry currently exercises the restoration-strength strategy (the dominant
  flicker source, §41) rather than all 9 §30 strategies. Both are noted in
  `docs/audit/known_bugs.md` and can be turned on if KPIs require.
