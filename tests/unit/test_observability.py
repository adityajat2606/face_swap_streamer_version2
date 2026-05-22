from __future__ import annotations

from face_swap.observability import (
    Health,
    HealthStatus,
    MetricsRegistry,
    Observatory,
    ReliabilityTracker,
    Tracer,
)


def test_counter_and_gauge():
    m = MetricsRegistry()
    m.inc("frames")
    m.inc("frames", 4)
    m.set_gauge("vram", 1024)
    assert m.counter("frames") == 5
    assert m.gauge("vram") == 1024


def test_histogram_quantiles():
    m = MetricsRegistry()
    for v in range(1, 101):
        m.observe("lat", v)
    h = m.histogram("lat")
    assert h.count == 100
    q = h.quantiles()
    assert 49 <= q["p50"] <= 52
    assert 94 <= q["p95"] <= 96


def test_prometheus_render():
    m = MetricsRegistry()
    m.inc("frames", 3)
    m.set_gauge("vram_mb", 2048)
    m.observe("dur", 10)
    text = m.render_prometheus()
    assert "face_swap_frames 3" in text
    assert "face_swap_vram_mb 2048" in text
    assert "quantile=" in text


def test_tracer_nesting_and_histogram():
    m = MetricsRegistry()
    t = Tracer(m)
    with t.span("outer"):
        with t.span("inner"):
            pass
    assert len(t.completed_roots) == 1
    assert t.completed_roots[0].name == "outer"
    assert len(t.completed_roots[0].children) == 1
    assert m.histogram("span.inner").count == 1


def test_health_aggregate_worst():
    o = Observatory()
    o.health.register("a", lambda: HealthStatus(Health.UP))
    o.health.register("b", lambda: HealthStatus(Health.DEGRADED, "slow"))
    report = o.health.run()
    assert report["status"] == "DEGRADED"


def test_health_check_that_raises_is_down():
    o = Observatory()

    def boom():
        raise RuntimeError("x")

    o.health.register("boom", boom)
    assert o.health.run()["status"] == "DOWN"


def test_reliability_slo_and_budget():
    r = ReliabilityTracker(slo_target=0.9)
    for _ in range(95):
        r.record_frame(success=True)
    for _ in range(5):
        r.record_frame(success=False)
    assert r.success_rate == 0.95
    assert r.slo_met is True
    assert 0.0 <= r.error_budget_remaining <= 1.0


def test_reliability_budget_exhausted():
    r = ReliabilityTracker(slo_target=0.98)
    for _ in range(90):
        r.record_frame(success=True)
    for _ in range(10):
        r.record_frame(success=False, manual_review=True)
    assert r.slo_met is False
    assert r.error_budget_remaining == 0.0
    assert r.manual_review_rate == 0.10


def test_observatory_snapshot_shape():
    o = Observatory()
    o.metrics.inc("x")
    o.reliability.record_frame(success=True)
    snap = o.snapshot()
    assert {"metrics", "reliability", "health", "traces"} <= set(snap)
