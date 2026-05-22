"""Configuration schema, loading, dotted-key overrides, and config hashing.

Configs are validated Pydantic models loaded from YAML (CLAUDE.md §12). The
canonical-JSON SHA-256 of the *processing-relevant* fields is the config hash
that gates resume (§12.4, §14.2).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .errors import ConfigError


class ProcessingCfg(BaseModel):
    preserve_original_fps: bool = True
    process_all_frames: bool = True
    enable_resume: bool = True
    quality_first: bool = True
    max_retry_per_frame: int = Field(5, ge=0, le=20)
    checkpoint_every_n_frames: int = Field(100, ge=1)
    random_seed: int = 42
    frame_storage_backend: Literal["png", "ffv1"] = "ffv1"


class FaceDetectionCfg(BaseModel):
    primary_detector: Literal["auto", "insightface_buffalo_l", "retinaface"] = "auto"
    fallback_detector: bool = True
    min_confidence: float = Field(0.70, ge=0, le=1)
    det_size: tuple[int, int] = (1280, 1280)


class TrackingCfg(BaseModel):
    enable_identity_tracks: bool = True
    enable_landmark_smoothing: bool = True
    enable_optical_flow_guidance: bool = True
    max_track_gap_frames: int = Field(30, ge=1)
    embedding_match_threshold: float = Field(0.40, ge=0, le=1)
    embedding_weight: float = Field(0.70, ge=0, le=1)
    consistency_drift_threshold: float = Field(0.25, ge=0, le=1)


class SwapCfg(BaseModel):
    model: Literal["auto", "inswapper_128", "simswap"] = "auto"
    preserve_expression: bool = True
    preserve_pose: bool = True


class RestorationCfg(BaseModel):
    enabled: bool = True
    strength: Literal["low", "medium", "high", "adaptive"] = "adaptive"
    avoid_plastic_skin: bool = True
    max_strength: float = Field(0.7, ge=0, le=1)
    max_strength_delta_per_frame: float = Field(0.05, ge=0, le=1)


class StabilizationCfg(BaseModel):
    enabled: bool = True
    temporal_window_frames: int = Field(5, ge=3, le=11)
    flicker_detection: bool = True
    reprocess_unstable_frames: bool = True
    one_euro_landmarks: dict[str, float] = Field(
        default_factory=lambda: {"min_cutoff": 1.0, "beta": 0.007}
    )
    one_euro_bbox: dict[str, float] = Field(
        default_factory=lambda: {"min_cutoff": 1.0, "beta": 0.02}
    )
    # (w_embedding, w_color, w_landmark, w_mask, w_sharpness) — §FR-9
    flicker_weights: tuple[float, float, float, float, float] = (0.30, 0.20, 0.20, 0.15, 0.15)


class OutputCfg(BaseModel):
    format: Literal["mp4", "mkv"] = "mp4"
    codec: Literal["h264", "h265", "ffv1", "prores"] = "h264"
    crf: int = Field(16, ge=0, le=51)
    preset: str = "slow"
    preserve_audio: bool = True
    generate_side_by_side_preview: bool = True
    generate_quality_report: bool = True
    redact_paths: bool = False


class ProjectCfg(BaseModel):
    name: str = "untitled"
    mode: Literal["draft", "quality", "final_cinematic"] = "quality"


class InputCfg(BaseModel):
    video_path: Path | None = None
    hero_reference: Path | None = None
    heroine_reference: Path | None = None


class TelemetryCfg(BaseModel):
    """SRE / observability knobs (added beyond the PRD)."""

    enabled: bool = True
    gpu_telemetry: bool = True
    gpu_poll_hz: float = Field(1.0, gt=0, le=10)
    log_json: bool = False
    metrics_flush_every_n_frames: int = Field(100, ge=1)
    health_check_enabled: bool = True


class Config(BaseModel):
    project: ProjectCfg = ProjectCfg()
    input: InputCfg = InputCfg()
    processing: ProcessingCfg = ProcessingCfg()
    face_detection: FaceDetectionCfg = FaceDetectionCfg()
    tracking: TrackingCfg = TrackingCfg()
    swap: SwapCfg = SwapCfg()
    restoration: RestorationCfg = RestorationCfg()
    stabilization: StabilizationCfg = StabilizationCfg()
    output: OutputCfg = OutputCfg()
    telemetry: TelemetryCfg = TelemetryCfg()

    # ---- hashing -------------------------------------------------------
    def processing_fingerprint(self) -> dict[str, Any]:
        """The config subset that determines output bytes (excludes paths,
        telemetry, and reporting toggles)."""
        data = self.model_dump(mode="json")
        for volatile in ("input", "telemetry"):
            data.pop(volatile, None)
        # Output presentation toggles don't change the swapped frames.
        out = data.get("output", {})
        for k in ("generate_side_by_side_preview", "generate_quality_report", "redact_paths"):
            out.pop(k, None)
        return data

    def config_hash(self) -> str:
        canonical = json.dumps(self.processing_fingerprint(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _coerce_scalar(value: str) -> Any:
    """Best-effort scalar coercion for CLI ``--set key=value`` strings."""
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _set_nested(raw: dict[str, Any], keys: list[str], value: Any) -> None:
    node = raw
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            node[k] = nxt
        node = nxt
    node[keys[-1]] = value


def load_config(path: Path, overrides: dict[str, str] | None = None) -> Config:
    """Load a YAML config and apply dotted-key overrides (``--set a.b=c``)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # narrow: don't swallow everything
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    for key, value in (overrides or {}).items():
        _set_nested(raw, key.split("."), _coerce_scalar(value))
    try:
        return Config(**raw)
    except Exception as exc:  # pydantic ValidationError → ConfigError
        raise ConfigError(f"config validation failed: {exc}") from exc
