# Cinematic Face Swap Stabilization Engine

A production-grade, modular face-swap pipeline focused on **temporal stability**
(flicker reduction), identity-correct tracking, per-frame quality validation with
retry, checkpoint/resume, and a full SRE/observability layer. Built as an
**additive** package (`src/face_swap/`) alongside the existing v0 streamer, which
it treats as the measurable baseline.

> This document covers the new engine. The repo's top-level `README.md` /
> `CLAUDE.md` describe the original v0 streaming app. The build/run rules for the
> engine itself live in `Claude_AI_Cinematic_Face_Swap_Stabilization_Instructions.md`
> (the engine's CLAUDE.md) and the PRD.

## What it does

Upload a song video plus one hero (male) and one heroine (female) reference
image. The engine swaps both faces onto the matching subjects and renders a
cinematic-quality MP4 with the original audio and FPS preserved, while measuring
and minimizing frame-to-frame flicker.

## Architecture (11 modules + SRE)

```
video → video_manager → frame_store
                              │
                       face_detector ──► identity_manager
                              │              │
                              ▼              ▼
                          tracker ◄──────────┘
                              ▼
                        swap_engine → restoration_engine → temporal_stabilizer
                              ▼
                     quality_validator ──(retry)──► swap_engine
                              ▼
                          renderer → final.mp4
                              ▼
                      report_generator
```

Cross-cutting: `observability` (metrics/tracing/health/SLO), `gpu_telemetry`,
`logging_setup`, `checkpoint`, `manifest`, `config`, `flicker`, `metrics`. See
[audit/modules.md](audit/modules.md) for the per-module map and
[OBSERVABILITY.md](OBSERVABILITY.md) for the SRE layer.

Module boundaries are **enforced** by `tests/test_imports.py` (no module imports
the CLI; `flicker` is swap-agnostic; the package root is import-light).

## Install

CPU tooling / tests (any host):

```bash
uv venv --python 3.11 && uv sync           # or: pip install -e ".[dev]"
uv run pytest -q -m "not gpu and not e2e"
```

GPU runtime (RTX 5080 / Blackwell target, §19):

```bash
uv pip install -e ".[gpu,restoration]"
uv run python scripts/smoke_gpu.py          # expect capability (12, 0)
export FACESWAP_INSWAPPER_PATH=/path/to/inswapper_128_fp16.onnx
```

## Run

```bash
face_swap run \
  --video input/videos/song.mp4 \
  --hero  input/references/hero.png \
  --heroine input/references/heroine.png \
  --config configs/final_cinematic.yaml \
  --output output/ \
  --report-delta baseline/v0_metrics.json

face_swap resume output/<run_id>          # after an interruption
face_swap health --config configs/quality.yaml   # SRE health probe
```

Exit codes (§13.2): `0` ok · `2` KPI gap · `3` manual review needed · `10`
config/input · `11` model load · `12` VRAM · `20` interrupted (resumable).

### Configs (§12.3)

`draft.yaml` (fast, crf22, 2 retries) · `quality.yaml` (adaptive, crf18, 5
retries) · `final_cinematic.yaml` (RAFT flow, crf16, 8 retries). Override any
field: `--set stabilization.flicker_detection=false`.

## Outputs (per run, under `output/<run_id>/`)

- `final.mp4` — swapped video + original audio.
- `quality.jsonl` — one verdict line per frame (§17.1).
- `processing_log.json` — manifest: hashes, model versions, KPIs, observability
  snapshot (§17.2).
- `reports/{summary.md, quality_report.csv, failed_frames.json}` (§10.3).
- `gpu_telemetry.csv` (§17.3), `checkpoint/` (§14), `debug_frames/`.

## Testing

```bash
uv run pytest -q -m "not integration and not e2e and not gpu"  # unit (CPU, <3s)
uv run pytest -q -m integration                                 # needs ffmpeg
uv run ruff check src/face_swap tests
uv run mypy src/face_swap
```

Core-logic coverage (§16.5): flicker 100%, observability 99%, quality_validator
96%, tracker 95%, checkpoint 95%, one_euro 96%.

## Milestone status

| Milestone | Status |
| --- | --- |
| M1 Baseline audit | done — `docs/audit/*`, `baseline/v0_metrics.json`, manifest |
| M2 Frame extract/rebuild | code + tests done; clip round-trip runs on a host with ffmpeg |
| M3 Tracking layer | tracker + identity + ambiguity guard + id-swap detector done & tested |
| M4 Temporal stabilization | One-Euro + Flicker Score + mask re-derivation done & tested; RAFT wiring on GPU host |
| M5 Quality validation & retry | verdict table + strategy-queue retry + checkpoint/resume done & tested |
| M6 Final cinematic | preset, preview, reporting, delta-vs-baseline done; KPI/human-eval gates run on GPU host |

GPU-only items (real swap inference, RAFT flow, baseline KPI numbers, blind
human eval) are written to spec and must be executed on the RTX target; see
[audit/known_bugs.md](audit/known_bugs.md).
