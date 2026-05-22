"""Quality validation & retry orchestration (CLAUDE.md §9 / PRD §FR-10, §FR-11).

Every frame gets a verdict. ``FAIL`` triggers a strategy-queue retry; the budget
is respected; nothing is silently accepted (§18.3). This module orchestrates by
calling other modules — it contains no detection or swap logic (§4.2). To keep
it unit-testable without a GPU, the per-frame metric computation is injected as
``FrameContext.metrics_fn``; the pipeline supplies the real GPU implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .logging_setup import get_logger
from .types import FrameResult, QualityVerdict

_log = get_logger("face_swap.quality")

# (metric_name, direction, pass_threshold, warning_threshold) — PRD §29.
# direction "above": higher is better. "below": lower is better.
THRESHOLDS: list[tuple[str, str, float, float]] = [
    ("detection_confidence", "above", 0.90, 0.70),
    ("landmark_confidence", "above", 0.85, 0.65),
    ("identity_consistency", "above", 0.80, 0.65),
    ("flicker_score", "below", 0.10, 0.25),
    ("color_shift_de", "below", 3.0, 8.0),
    ("mask_instability", "below", 0.05, 0.15),
]

# Ordered retry tactics (PRD §30).
RETRY_STRATEGIES: list[str] = [
    "crop_larger",
    "crop_smaller",
    "swap_detector",
    "landmarks_from_prev",
    "landmarks_from_next",
    "restoration_lower",
    "restoration_higher",
    "blending_mask_alt",
    "temporal_interpolation",
]


@dataclass
class FrameContext:
    """Everything computed so far for one frame, plus tunable retry params.

    ``metrics_fn`` maps a context to a metrics dict (must include
    ``flicker_score`` and may include any THRESHOLDS key). The pipeline injects
    the real GPU pipeline; tests inject a synthetic function.
    """

    frame_idx: int
    metrics_fn: Callable[[FrameContext], dict[str, float]]
    # tunable params mutated by retry strategies:
    crop_scale: float = 1.0
    detector: str = "primary"
    landmark_source: str = "current"  # current | prev | next
    restoration_strength: float = 0.5
    blending_mask: str = "default"  # default | alt
    temporal_interpolate: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def verdict_from_metrics(m: dict[str, float]) -> QualityVerdict:
    """Worst-of verdict over the threshold table (§29). Missing metrics skipped."""
    worst: QualityVerdict = "PASS"
    for name, direction, pass_t, warn_t in THRESHOLDS:
        v = m.get(name)
        if v is None:
            continue
        if direction == "above":
            if v < warn_t:
                return "FAIL"
            if v < pass_t:
                worst = "WARNING"
        else:
            if v > warn_t:
                return "FAIL"
            if v > pass_t:
                worst = "WARNING"
    return worst


def evaluate(frame_idx: int, ctx: FrameContext) -> FrameResult:
    """Compute metrics for the frame and return a verdict-bearing result."""
    metrics = ctx.metrics_fn(ctx)
    verdict = verdict_from_metrics(metrics)
    components = {k: v for k, v in metrics.items() if k in
                 ("embedding", "color", "landmark", "mask", "sharpness")}
    return FrameResult(
        frame_idx=frame_idx,
        verdict=verdict,
        flicker_score=float(metrics.get("flicker_score", 0.0)),
        components=components,
    )


def apply_strategy(strategy: str, ctx: FrameContext) -> FrameContext:
    """Return a new context with the named retry tactic applied (§9.2)."""
    handlers: dict[str, Callable[[FrameContext], FrameContext]] = {
        "crop_larger": lambda c: replace(c, crop_scale=min(c.crop_scale * 1.25, 2.0)),
        "crop_smaller": lambda c: replace(c, crop_scale=max(c.crop_scale * 0.8, 0.5)),
        "swap_detector": lambda c: replace(
            c, detector="fallback" if c.detector == "primary" else "primary"
        ),
        "landmarks_from_prev": lambda c: replace(c, landmark_source="prev"),
        "landmarks_from_next": lambda c: replace(c, landmark_source="next"),
        "restoration_lower": lambda c: replace(
            c, restoration_strength=max(c.restoration_strength - 0.2, 0.0)
        ),
        "restoration_higher": lambda c: replace(
            c, restoration_strength=min(c.restoration_strength + 0.2, 1.0)
        ),
        "blending_mask_alt": lambda c: replace(c, blending_mask="alt"),
        "temporal_interpolation": lambda c: replace(c, temporal_interpolate=True),
    }
    handler = handlers.get(strategy)
    if handler is None:
        raise ValueError(f"unknown retry strategy: {strategy}")
    return handler(ctx)


def retry(frame_idx: int, ctx: FrameContext, budget: int = 5) -> FrameResult:
    """Run the strategy queue until PASS or the budget is exhausted (§9.2)."""
    budget = max(0, budget)
    attempts: list[tuple[str, FrameResult]] = []
    for strategy in RETRY_STRATEGIES[:budget]:
        new_ctx = apply_strategy(strategy, ctx)
        result = evaluate(frame_idx, new_ctx)
        attempts.append((strategy, result))
        if result.verdict == "PASS":
            _log.info("retry_succeeded", frame_idx=frame_idx, strategy=strategy,
                      attempts=len(attempts))
            return result.replace(
                retry_count=len(attempts),
                retry_strategies=tuple(s for s, _ in attempts),
                reasons=tuple(s for s, _ in attempts),
            )
    if not attempts:
        # budget == 0: re-evaluate once so a verdict is always produced.
        return evaluate(frame_idx, ctx)
    # Exhausted budget — keep the lowest-flicker attempt, downgrade to WARNING.
    best_strategy, best = min(attempts, key=lambda sr: sr[1].flicker_score)
    _log.warning("retry_budget_exhausted", frame_idx=frame_idx,
                 best_strategy=best_strategy, best_flicker=best.flicker_score)
    return best.replace(
        verdict="WARNING",
        retry_count=len(attempts),
        retry_strategies=tuple(s for s, _ in attempts),
        reasons=tuple(s for s, _ in attempts) + ("budget_exhausted",),
    )


def evaluate_with_retry(
    frame_idx: int, ctx: FrameContext, max_retry: int
) -> FrameResult:
    """Evaluate once; if FAIL, run the retry queue. The single entry point used
    by the pipeline."""
    result = evaluate(frame_idx, ctx)
    if result.verdict != "FAIL":
        return result
    return retry(frame_idx, ctx, budget=max_retry)
