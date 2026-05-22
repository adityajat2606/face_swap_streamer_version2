"""Typed exceptions and process exit codes.

Catch narrow exception types, never bare ``Exception`` (CLAUDE.md §18.12).
Exit codes are the CLI contract from §13.2 and are asserted in tests.
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes (CLAUDE.md §13.2)."""

    OK = 0
    UNEXPECTED = 1
    KPI_NOT_MET = 2
    HUMAN_REVIEW_REQUIRED = 3
    CONFIG_OR_INPUT_ERROR = 10
    MODEL_LOAD_ERROR = 11
    VRAM_EXHAUSTION = 12
    INTERRUPTED = 20


class FaceSwapError(Exception):
    """Base class for all pipeline errors."""

    exit_code: ExitCode = ExitCode.UNEXPECTED


class ConfigError(FaceSwapError):
    """Invalid configuration or input arguments."""

    exit_code = ExitCode.CONFIG_OR_INPUT_ERROR


class InputError(FaceSwapError):
    """Missing or unreadable input video / reference image."""

    exit_code = ExitCode.CONFIG_OR_INPUT_ERROR


class ModelLoadError(FaceSwapError):
    """A model failed to load (missing weights, CPU-only fallback, etc.)."""

    exit_code = ExitCode.MODEL_LOAD_ERROR


class VramExhaustionError(FaceSwapError):
    """CUDA out-of-memory; see CLAUDE.md §10A / §19 mitigations."""

    exit_code = ExitCode.VRAM_EXHAUSTION


class DetectionError(FaceSwapError):
    """Face detection failed irrecoverably for a frame."""


class SwapInferenceError(FaceSwapError):
    """The swap model failed on a given face crop."""


class AmbiguousIdentityError(FaceSwapError):
    """Hero/heroine assignment is ambiguous; needs human confirmation (§7.4)."""


class ResumeError(FaceSwapError):
    """A checkpoint cannot be resumed (config/model/source mismatch, §14.2)."""

    exit_code = ExitCode.CONFIG_OR_INPUT_ERROR


class ResourceConflictError(FaceSwapError):
    """Co-resident models would exceed the VRAM budget (§11.2)."""

    exit_code = ExitCode.VRAM_EXHAUSTION
