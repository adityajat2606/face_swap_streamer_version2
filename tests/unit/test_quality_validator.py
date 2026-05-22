from __future__ import annotations

from face_swap.quality_validator import (
    RETRY_STRATEGIES,
    FrameContext,
    apply_strategy,
    evaluate,
    evaluate_with_retry,
    retry,
    verdict_from_metrics,
)


def test_verdict_pass():
    m = {"detection_confidence": 0.95, "flicker_score": 0.05, "mask_instability": 0.01}
    assert verdict_from_metrics(m) == "PASS"


def test_verdict_warning():
    m = {"flicker_score": 0.15}  # between pass(0.10) and warn(0.25)
    assert verdict_from_metrics(m) == "WARNING"


def test_verdict_fail():
    m = {"flicker_score": 0.40}  # above warn threshold
    assert verdict_from_metrics(m) == "FAIL"


def test_verdict_skips_missing_metrics():
    assert verdict_from_metrics({}) == "PASS"


def test_above_direction_fail():
    assert verdict_from_metrics({"detection_confidence": 0.5}) == "FAIL"


def test_apply_strategy_crop_larger():
    ctx = FrameContext(0, metrics_fn=lambda c: {"flicker_score": 0.0})
    out = apply_strategy("crop_larger", ctx)
    assert out.crop_scale > ctx.crop_scale


def test_all_strategies_apply():
    ctx = FrameContext(0, metrics_fn=lambda c: {"flicker_score": 0.0})
    for s in RETRY_STRATEGIES:
        assert apply_strategy(s, ctx) is not None


def test_retry_succeeds_after_n_attempts():
    """Fail until crop_scale grows past 1.2, then PASS."""
    def mfn(c: FrameContext):
        return {"flicker_score": 0.05 if c.crop_scale > 1.2 else 0.5}

    ctx = FrameContext(0, metrics_fn=mfn)
    result = retry(0, ctx, budget=5)
    assert result.verdict == "PASS"
    assert result.retry_count >= 1
    assert result.retry_strategies[0] == "crop_larger"


def test_retry_budget_exhausted_becomes_warning():
    ctx = FrameContext(0, metrics_fn=lambda c: {"flicker_score": 0.9})
    result = retry(0, ctx, budget=3)
    assert result.verdict == "WARNING"
    assert "budget_exhausted" in result.reasons
    assert result.retry_count == 3


def test_retry_budget_zero():
    ctx = FrameContext(0, metrics_fn=lambda c: {"flicker_score": 0.9})
    result = retry(0, ctx, budget=0)
    assert result.frame_idx == 0  # still produces a verdict


def test_evaluate_with_retry_no_retry_on_pass():
    calls = {"n": 0}

    def mfn(c):
        calls["n"] += 1
        return {"flicker_score": 0.01}

    evaluate_with_retry(0, FrameContext(0, metrics_fn=mfn), max_retry=5)
    assert calls["n"] == 1  # evaluated once, no retries


def test_evaluate_components_extracted():
    def mfn(c):
        return {"flicker_score": 0.05, "embedding": 0.1, "color": 0.2,
                "detection_confidence": 0.99}

    r = evaluate(0, FrameContext(0, metrics_fn=mfn))
    assert r.components == {"embedding": 0.1, "color": 0.2}
