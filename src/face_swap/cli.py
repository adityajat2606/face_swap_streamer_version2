"""Command-line interface (CLAUDE.md §13). Thin shell over library functions.

Parses args, configures logging, dispatches. Exit codes follow §13.2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .errors import ExitCode, FaceSwapError


@click.group()
@click.version_option(__version__, prog_name="face_swap")
@click.option("--log-level", default="INFO", show_default=True)
@click.option("--log-json", is_flag=True, help="Emit logs as JSON.")
def main(log_level: str, log_json: bool) -> None:
    """AI Cinematic Face Swap Stabilization Engine."""
    from .logging_setup import configure

    configure(level=log_level, json=log_json)


def _load_cfg(config_path: Path, overrides: tuple[str, ...]):
    from .config import load_config

    override_map: dict[str, str] = {}
    for o in overrides:
        if "=" not in o:
            raise click.BadParameter(f"--set expects key=value, got {o!r}")
        k, v = o.split("=", 1)
        override_map[k] = v
    return load_config(config_path, override_map)


@main.command("run")
@click.option("--video", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--hero", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--heroine", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--set", "overrides", multiple=True, help="Override config: --set a.b=c")
@click.option("--report-delta", type=click.Path(exists=True, path_type=Path), default=None,
              help="Baseline metrics JSON for delta reporting.")
def run_cmd(video, hero, heroine, config_path, output, overrides, report_delta):
    """Run the face swap pipeline end-to-end."""
    from .pipeline import run

    try:
        cfg = _load_cfg(config_path, overrides)
        cfg = cfg.model_copy(update={"input": cfg.input.model_copy(update={
            "video_path": video, "hero_reference": hero, "heroine_reference": heroine})})
        code = run(cfg, output, baseline_path=report_delta)
    except FaceSwapError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(int(exc.exit_code))
    except KeyboardInterrupt:
        click.echo("interrupted (resumable with `face_swap resume`)", err=True)
        sys.exit(int(ExitCode.INTERRUPTED))
    sys.exit(int(code))


@main.command("resume")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
def resume_cmd(run_dir):
    """Resume a previously checkpointed run."""
    from .pipeline import resume

    try:
        code = resume(run_dir)
    except FaceSwapError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(int(exc.exit_code))
    sys.exit(int(code))


@main.command("assign-identities")
@click.option("--run-id", required=True)
def assign_cmd(run_id):
    """Interactive identity-mapping resolver."""
    from .pipeline import resolve_identity_mapping

    resolve_identity_mapping(run_id)


@main.command("baseline")
@click.option("--manifest", default="tests/fixtures/test_set_manifest.yaml",
              type=click.Path(path_type=Path))
@click.option("--output", default="baseline/v0_metrics.json", type=click.Path(path_type=Path))
def baseline_cmd(manifest, output):
    """Compute baseline metrics on the test-set manifest (Milestone 1)."""
    from .baseline import measure_all

    measure_all(Path(manifest), Path(output))


@main.command("health")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def health_cmd(config_path):
    """Run health checks and print a JSON report (SRE)."""
    import json

    from .observability import Observatory
    from .pipeline import _register_health_checks

    cfg = _load_cfg(config_path, ())
    obs = Observatory()
    _register_health_checks(obs, cfg)
    report = obs.health.run()
    click.echo(json.dumps(report, indent=2))
    sys.exit(0 if report["status"] != "DOWN" else 1)


if __name__ == "__main__":
    main()
