from __future__ import annotations

from pathlib import Path

import pytest

from face_swap.config import Config, load_config
from face_swap.errors import ConfigError

CONFIGS = Path(__file__).resolve().parent.parent.parent / "configs"


@pytest.mark.parametrize("name", ["draft", "quality", "final_cinematic"])
def test_preset_configs_load(name):
    cfg = load_config(CONFIGS / f"{name}.yaml")
    assert cfg.project.mode == name
    assert isinstance(cfg.config_hash(), str) and len(cfg.config_hash()) == 64


def test_preset_retry_budgets():
    assert load_config(CONFIGS / "draft.yaml").processing.max_retry_per_frame == 2
    assert load_config(CONFIGS / "quality.yaml").processing.max_retry_per_frame == 5
    assert load_config(CONFIGS / "final_cinematic.yaml").processing.max_retry_per_frame == 8


def test_final_cinematic_never_high_restoration():
    cfg = load_config(CONFIGS / "final_cinematic.yaml")
    assert cfg.restoration.strength != "high"  # §18.7


def test_override_scalar_coercion(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("project: {name: t, mode: quality}\n")
    cfg = load_config(p, {"stabilization.flicker_detection": "false",
                          "processing.max_retry_per_frame": "7"})
    assert cfg.stabilization.flicker_detection is False
    assert cfg.processing.max_retry_per_frame == 7


def test_override_creates_nested_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("project: {name: t, mode: draft}\n")
    cfg = load_config(p, {"output.crf": "20"})
    assert cfg.output.crf == 20


def test_config_hash_ignores_paths_and_telemetry(tmp_path):
    base = Config(project={"name": "a", "mode": "quality"})
    h1 = base.config_hash()
    h2 = base.model_copy(update={"telemetry": base.telemetry.model_copy(
        update={"gpu_poll_hz": 5.0})}).config_hash()
    assert h1 == h2  # telemetry excluded from fingerprint


def test_config_hash_changes_with_processing(tmp_path):
    base = Config(project={"name": "a", "mode": "quality"})
    changed = base.model_copy(update={"processing": base.processing.model_copy(
        update={"random_seed": 7})})
    assert base.config_hash() != changed.config_hash()


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config(Path("does/not/exist.yaml"))


def test_invalid_value_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("processing: {max_retry_per_frame: 999}\n")  # > le=20
    with pytest.raises(ConfigError):
        load_config(p)


def test_non_mapping_root_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        load_config(p)


@pytest.mark.parametrize("name", ["draft", "quality", "final_cinematic"])
def test_dump_reload_preserves_config_hash(name, tmp_path):
    """Resume reads persisted run_config.yaml and compares config_hash; the
    dump→reload round trip must be hash-stable or every resume is refused."""
    from face_swap.pipeline import _dump_config

    cfg = load_config(CONFIGS / f"{name}.yaml")
    p = tmp_path / "run_config.yaml"
    p.write_text(_dump_config(cfg), encoding="utf-8")
    assert load_config(p).config_hash() == cfg.config_hash()
