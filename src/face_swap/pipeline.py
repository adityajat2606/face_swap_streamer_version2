"""End-to-end pipeline orchestration (CLAUDE.md §11.1 dataflow).

This module wires the stages together and owns run-directory layout, manifest,
checkpoint cadence, telemetry, and reporting. GPU work (detect/swap/restore) is
delegated to the engine modules, which load lazily — so this module imports on a
CPU host and its scaffolding (run-dir setup, manifest, resume guard, reporting)
is unit-testable without a GPU.

The heavy per-frame loop (``_process_frames``) requires a GPU + model weights and
runs on the RTX target machine.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import Config
from .errors import ExitCode, FaceSwapError, InputError, ModelLoadError
from .logging_setup import bind_run_context, clear_run_context, get_logger
from .manifest import (
    git_sha,
    model_versions,
    new_run_id,
    redact,
    sha256_file_or_none,
    utc_now_iso,
    write_manifest,
)
from .observability import Health, HealthStatus, Observatory
from .report_generator import Reporter

_log = get_logger("face_swap.pipeline")

# Frame-success SLO: PRD §4A allows < 2% manual review, i.e. >= 98% success.
_SLO_TARGET = 0.98


def _inswapper_path() -> str:
    return os.getenv("FACESWAP_INSWAPPER_PATH", "")


class RunDirs:
    """Canonical per-run directory layout (CLAUDE.md §4.1)."""

    def __init__(self, output_root: Path, run_id: str):
        self.root = Path(output_root) / run_id
        self.checkpoint = self.root / "checkpoint"
        self.frames = self.root / "frames"
        self.debug = self.root / "debug_frames"
        self.reports = self.root / "reports"
        self.config = self.root / "config"
        self.quality_jsonl = self.root / "quality.jsonl"
        self.gpu_csv = self.root / "gpu_telemetry.csv"

    def make(self) -> RunDirs:
        for d in (self.root, self.checkpoint, self.frames, self.debug,
                  self.reports, self.config):
            d.mkdir(parents=True, exist_ok=True)
        return self


def build_manifest_skeleton(cfg: Config, run_id: str, dirs: RunDirs) -> dict:
    """Build the processing_log.json skeleton (everything knowable up front)."""
    redact_on = cfg.output.redact_paths
    vp = str(cfg.input.video_path) if cfg.input.video_path else None
    hero = str(cfg.input.hero_reference) if cfg.input.hero_reference else None
    heroine = str(cfg.input.heroine_reference) if cfg.input.heroine_reference else None
    models = model_versions({
        "swap": _inswapper_path() or None,
        "detector": None,  # buffalo_l resolves at load; recorded after load
        "restoration": None,
    })
    return {
        "run_id": run_id,
        "started_at": utc_now_iso(),
        "ended_at": None,
        "exit_code": None,
        "git_sha": git_sha(),
        "config_hash": cfg.config_hash(),
        "config_mode": cfg.project.mode,
        "random_seed": cfg.processing.random_seed,
        "source_video": {"path": redact(vp, redact_on), "sha256": sha256_file_or_none(vp)},
        "references": {
            "hero": {"path": redact(hero, redact_on), "sha256": sha256_file_or_none(hero)},
            "heroine": {"path": redact(heroine, redact_on), "sha256": sha256_file_or_none(heroine)},
        },
        "model_versions": models,
        "kpi_results": {},
        "delta_vs_baseline": {},
    }


def _register_health_checks(obs: Observatory, cfg: Config) -> None:
    """Wire baseline health checks (SRE)."""

    def config_check() -> HealthStatus:
        return HealthStatus(Health.UP, f"mode={cfg.project.mode}")

    def input_check() -> HealthStatus:
        vp = cfg.input.video_path
        if vp and Path(vp).is_file():
            return HealthStatus(Health.UP, "video present")
        return HealthStatus(Health.DOWN, "video missing")

    def slo_check() -> HealthStatus:
        r = obs.reliability
        if r.total == 0:
            return HealthStatus(Health.UP, "no frames yet")
        if r.slo_met:
            return HealthStatus(Health.UP, f"success={r.success_rate:.4f}")
        if r.error_budget_remaining > 0:
            return HealthStatus(Health.DEGRADED, f"budget={r.error_budget_remaining:.3f}")
        return HealthStatus(Health.DOWN, "error budget exhausted")

    obs.health.register("config", config_check)
    obs.health.register("input", input_check)
    obs.health.register("slo", slo_check)


def _load_baseline_kpis(baseline_path: Path | None) -> dict | None:
    if not baseline_path:
        return None
    p = Path(baseline_path)
    if not p.is_file():
        _log.warning("baseline_not_found", path=str(p))
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _log.warning("baseline_unparseable", error=str(exc))
        return None


def run(cfg: Config, output: Path, *, baseline_path: Path | None = None) -> ExitCode:
    """Run the pipeline end-to-end. Returns the process exit code (§13.2)."""
    if cfg.input.video_path is None:
        raise InputError("no input video configured")

    run_id = new_run_id()
    output_root = Path(output)
    if output_root.suffix:  # a file path was passed; use its parent as the root
        output_root = output_root.parent
    dirs = RunDirs(output_root, run_id).make()

    obs = Observatory(slo_target=_SLO_TARGET)
    _register_health_checks(obs, cfg)
    bind_run_context(run_id=run_id, stage="init")

    # persist the resolved config for resume verification (§14.2).
    (dirs.config / "run_config.yaml").write_text(_dump_config(cfg), encoding="utf-8")
    manifest = build_manifest_skeleton(cfg, run_id, dirs)
    write_manifest(dirs.root, manifest)

    _log.info("run_started", run_id=run_id, mode=cfg.project.mode,
              config_hash=manifest["config_hash"][:12])

    start = time.time()
    exit_code = ExitCode.OK
    kpis: dict = {}
    try:
        from .runner import StageRunner  # lazy: pulls GPU engines

        runner = StageRunner(cfg, dirs, obs, manifest)
        kpis = runner.execute(start_frame=0)
        exit_code = _exit_code_from_kpis(kpis, runner.had_manual_review)
    except ModelLoadError:
        exit_code = ExitCode.MODEL_LOAD_ERROR
        raise
    except FaceSwapError as exc:
        exit_code = exc.exit_code
        raise
    finally:
        manifest["ended_at"] = utc_now_iso()
        manifest["exit_code"] = int(exit_code)
        manifest["elapsed_seconds"] = round(time.time() - start, 3)
        manifest["kpi_results"] = kpis  # pipeline owns the manifest (§17.2)
        manifest["observability"] = obs.snapshot()
        write_manifest(dirs.root, manifest)

        baseline = _load_baseline_kpis(baseline_path)
        try:
            Reporter().write_all(
                dirs.root, run_id=run_id,
                baseline_kpis=baseline.get("flat_kpis") if baseline else None,
                meta={"git_sha": manifest["git_sha"], "captured_at": manifest["started_at"],
                      "config_name": cfg.project.name, "mode": cfg.project.mode,
                      "source": manifest["source_video"]["path"]},
            )
        except Exception as exc:  # noqa: BLE001 - reporting must not mask the run result
            _log.warning("report_generation_failed", error=str(exc))
        clear_run_context()

    _log.info("run_finished", run_id=run_id, exit_code=int(exit_code))
    return exit_code


def resume(run_dir: Path) -> ExitCode:
    """Resume a checkpointed run (§14.2)."""
    from .checkpoint import load_checkpoint, verify_resumable
    from .config import load_config

    run_dir = Path(run_dir)
    cfg_path = run_dir / "config" / "run_config.yaml"
    if not cfg_path.is_file():
        raise InputError(f"no run_config.yaml in {run_dir}")
    cfg = load_config(cfg_path)
    state = load_checkpoint(run_dir / "checkpoint")

    vp = str(cfg.input.video_path) if cfg.input.video_path else ""
    verify_resumable(
        state,
        config_hash=cfg.config_hash(),
        source_video_hash=sha256_file_or_none(vp) or "",
        model_versions=model_versions({"swap": _inswapper_path() or None,
                                        "detector": None, "restoration": None}),
    )

    obs = Observatory(slo_target=_SLO_TARGET)
    _register_health_checks(obs, cfg)
    bind_run_context(run_id=state.run_id, stage="resume")
    _log.info("run_resumed", run_id=state.run_id, from_frame=state.next_frame_to_process)

    dirs = RunDirs(run_dir.parent, state.run_id)
    manifest = json.loads((run_dir / "processing_log.json").read_text(encoding="utf-8"))
    from .runner import StageRunner

    runner = StageRunner(cfg, dirs, obs, manifest)
    runner.restore_state(state)
    kpis = runner.execute(start_frame=state.next_frame_to_process)
    clear_run_context()
    return _exit_code_from_kpis(kpis, runner.had_manual_review)


def resolve_identity_mapping(run_id: str) -> None:  # pragma: no cover - interactive
    """Interactive identity confirmation entry point (§7.4)."""
    raise NotImplementedError(
        "interactive identity resolution runs on the GPU host; "
        "see docs and output/<run_id>/identity_map.yaml"
    )


def _exit_code_from_kpis(kpis: dict, had_manual_review: bool) -> ExitCode:
    if had_manual_review:
        return ExitCode.HUMAN_REVIEW_REQUIRED
    median = kpis.get("flicker_score_median", 0.0)
    if median is not None and median >= 0.08:
        return ExitCode.KPI_NOT_MET
    return ExitCode.OK


def _dump_config(cfg: Config) -> str:
    import yaml

    return yaml.safe_dump(json.loads(cfg.model_dump_json()), sort_keys=False)
