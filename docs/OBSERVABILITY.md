# Observability & SRE

The engine ships a self-contained observability layer (`observability.py`,
`gpu_telemetry.py`, `logging_setup.py`). It is dependency-free (stdlib +
structlog), runs on any host, and is unit-tested (99% coverage). Everything is
bundled per run in an `Observatory` that the pipeline threads through.

## 1. Structured logging

`structlog`, JSON or console. Every line carries run-scoped context bound at
startup (`run_id`, `stage`, and `frame_idx` inside the frame loop). No `print()`
outside the CLI.

```bash
face_swap --log-json run ...        # machine-readable logs to stderr
```

## 2. Metrics

In-process Prometheus-style families:

- **Counters** — `verdict_pass/warning/fail`, frame counts.
- **Gauges** — `gpu_util_pct`, `vram_used_mb`, `gpu_temp_c`.
- **Histograms** — `flicker_score`, per-stage `span.<name>` durations,
  `vram_used_mb_hist`. Each exposes count/sum/mean/min/max/p50/p95/p99.

`MetricsRegistry.render_prometheus()` emits text-exposition format, so the
snapshot can be scraped or pushed to a gateway without code changes. The full
snapshot is also embedded in `processing_log.json` under `observability`.

## 3. Tracing

`Tracer.span(name, **attrs)` is a nestable context manager. Each span records a
duration histogram and emits a `span_end` log; nested spans form a per-frame
trace tree (e.g. `frame → {detect, swap, validate}`). Completed root spans are in
the run snapshot.

## 4. Health checks

`HealthRegistry` of named checks → aggregate verdict = worst child
(`UP > DEGRADED > DOWN`); a check that raises is itself `DOWN`. Built-in checks:
`config`, `input` (video present), `slo` (success rate vs target / error
budget). Probe without a full run:

```bash
face_swap health --config configs/quality.yaml      # prints JSON, exit 1 if DOWN
```

## 5. Reliability / SLO

`ReliabilityTracker` records each frame's success (verdict PASS/WARNING and no
manual review). It reports `success_rate`, `manual_review_rate`, whether the SLO
(default 0.98) is met, and the **error budget remaining** — the fraction of the
allowed-failure budget still unspent (0 = exhausted). The `slo` health check and
exit codes derive from this; manual-review frames over the §4A 2% bar push the
run to exit code 3.

## 6. GPU/VRAM telemetry

`GpuTelemetry` polls pynvml (preferred) or `nvidia-smi` (fallback) at a
configurable rate on a daemon thread, writing `gpu_telemetry.csv`
(`timestamp,gpu_util_pct,vram_used_mb,vram_total_mb,temperature_c,power_w`) and
feeding gauges. It degrades to a no-op on a host without a GPU, so it is safe to
start unconditionally. Tune via `telemetry.gpu_poll_hz`.

## Where it surfaces

| Signal | Location |
| --- | --- |
| Logs | stderr (console or JSON) |
| Per-frame verdicts | `output/<run_id>/quality.jsonl` |
| Metrics + traces + SLO snapshot | `processing_log.json → observability` |
| GPU time series | `output/<run_id>/gpu_telemetry.csv` |
| KPI / delta-vs-baseline | `reports/summary.md` |
| Live health | `face_swap health` |
