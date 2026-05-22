"""Observability primitives: metrics, tracing spans, health checks, and SLO/
reliability telemetry.

This is the SRE layer. It is intentionally dependency-free (stdlib + structlog)
so it can run on any host and be unit-tested without a GPU. The pipeline wires
a single :class:`Observatory` per run; modules pull it from the run context.

Design notes
------------
* Metrics are in-process (Prometheus-style families) and can be flushed to a
  JSON snapshot for the run manifest. No network exporter is assumed, but the
  snapshot is exporter-friendly (it renders to Prometheus text via
  :meth:`MetricsRegistry.render_prometheus`).
* Tracing is span-based with parent/child nesting tracked through a contextvar,
  so a stage timed inside another stage produces a tree, and every span close
  emits a structured log line and a histogram observation.
* Health checks are named callables returning :class:`HealthStatus`; the
  aggregate verdict is the worst child (UP > DEGRADED > DOWN).
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum

from .logging_setup import get_logger

_log = get_logger("face_swap.obs")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass
class Histogram:
    """A bounded-memory histogram keeping summary stats and recent samples."""

    name: str
    _samples: list[float] = field(default_factory=list)
    _max_samples: int = 100_000
    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def observe(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        if len(self._samples) < self._max_samples:
            self._samples.append(value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def quantiles(self) -> dict[str, float]:
        s = sorted(self._samples)
        return {
            "p50": _percentile(s, 50),
            "p95": _percentile(s, 95),
            "p99": _percentile(s, 99),
        }

    def snapshot(self) -> dict[str, float]:
        q = self.quantiles()
        return {
            "count": self.count,
            "sum": round(self.total, 6),
            "mean": round(self.mean, 6),
            "min": round(self.min, 6) if self.count else 0.0,
            "max": round(self.max, 6) if self.count else 0.0,
            **{k: round(v, 6) for k, v in q.items()},
        }


class MetricsRegistry:
    """Thread-safe registry of counters, gauges and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, Histogram] = {}

    def inc(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + amount

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            hist = self._histograms.get(name)
            if hist is None:
                hist = Histogram(name)
                self._histograms[name] = hist
            hist.observe(value)

    def counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def gauge(self, name: str) -> float:
        with self._lock:
            return self._gauges.get(name, 0.0)

    def histogram(self, name: str) -> Histogram | None:
        with self._lock:
            return self._histograms.get(name)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: h.snapshot() for k, h in self._histograms.items()},
            }

    def render_prometheus(self) -> str:
        """Render the current state as Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {k: h.snapshot() for k, h in self._histograms.items()}
        for name, val in sorted(counters.items()):
            metric = _prom_name(name)
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"{metric} {val}")
        for name, val in sorted(gauges.items()):
            metric = _prom_name(name)
            lines.append(f"# TYPE {metric} gauge")
            lines.append(f"{metric} {val}")
        for name, stats in sorted(histograms.items()):
            metric = _prom_name(name)
            lines.append(f"# TYPE {metric} summary")
            for q in ("p50", "p95", "p99"):
                quant = {"p50": "0.5", "p95": "0.95", "p99": "0.99"}[q]
                lines.append(f'{metric}{{quantile="{quant}"}} {stats[q]}')
            lines.append(f"{metric}_sum {stats['sum']}")
            lines.append(f"{metric}_count {stats['count']}")
        return "\n".join(lines) + "\n"


def _prom_name(name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in name)
    return f"face_swap_{safe}"


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "current_span", default=None
)


@dataclass
class Span:
    name: str
    start: float
    parent: Span | None = None
    duration_ms: float | None = None
    attributes: dict[str, object] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms or 0.0, 3),
            "attributes": dict(self.attributes),
            "children": [c.to_dict() for c in self.children],
        }


class Tracer:
    """Span-based tracer. Each span records a histogram observation named
    ``span.<name>`` on close and logs a structured ``span_end`` event."""

    def __init__(self, metrics: MetricsRegistry) -> None:
        self._metrics = metrics
        self.completed_roots: list[Span] = []

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[Span]:
        parent = _current_span.get()
        span = Span(name=name, start=time.perf_counter(), parent=parent, attributes=dict(attributes))
        token = _current_span.set(span)
        try:
            yield span
        finally:
            span.duration_ms = (time.perf_counter() - span.start) * 1000.0
            _current_span.reset(token)
            self._metrics.observe(f"span.{name}", span.duration_ms)
            if parent is not None:
                parent.children.append(span)
            else:
                self.completed_roots.append(span)
            _log.debug(
                "span_end", span=name, duration_ms=round(span.duration_ms, 3), **span.attributes
            )


# --------------------------------------------------------------------------- #
# Health checks
# --------------------------------------------------------------------------- #
class Health(StrEnum):
    UP = "UP"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"


_HEALTH_ORDER = {Health.UP: 0, Health.DEGRADED: 1, Health.DOWN: 2}


@dataclass
class HealthStatus:
    status: Health
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status.value, "detail": self.detail}


class HealthRegistry:
    """Registry of named health checks. Aggregate = worst child."""

    def __init__(self) -> None:
        self._checks: dict[str, Callable[[], HealthStatus]] = {}

    def register(self, name: str, check: Callable[[], HealthStatus]) -> None:
        self._checks[name] = check

    def run(self) -> dict[str, object]:
        results: dict[str, HealthStatus] = {}
        worst = Health.UP
        for name, check in self._checks.items():
            try:
                status = check()
            except Exception as exc:  # a check that throws is itself DOWN
                status = HealthStatus(Health.DOWN, f"check raised: {exc}")
            results[name] = status
            if _HEALTH_ORDER[status.status] > _HEALTH_ORDER[worst]:
                worst = status.status
        return {
            "status": worst.value,
            "checks": {k: v.to_dict() for k, v in results.items()},
        }


# --------------------------------------------------------------------------- #
# Reliability / SLO
# --------------------------------------------------------------------------- #
@dataclass
class ReliabilityTracker:
    """Tracks frame-level success/failure for SLO and error-budget reporting.

    A frame "succeeds" if its verdict is PASS or WARNING and it did not require
    manual review. The SLO target is configurable; the error budget is the
    allowed failure fraction.
    """

    slo_target: float = 0.98  # fraction of frames that must succeed
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    manual_review: int = 0
    retries: int = 0

    def record_frame(self, *, success: bool, manual_review: bool = False, retries: int = 0) -> None:
        self.total += 1
        self.retries += retries
        if manual_review:
            self.manual_review += 1
        if success:
            self.succeeded += 1
        else:
            self.failed += 1

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 1.0

    @property
    def manual_review_rate(self) -> float:
        return self.manual_review / self.total if self.total else 0.0

    @property
    def error_budget_remaining(self) -> float:
        """Fraction of the allowed-failure budget still unspent, in ``[0, 1]``.

        budget = (1 - slo_target). spent = observed failure rate. Returns
        ``1 - spent/budget`` clamped to ``[0, 1]`` (0 = budget exhausted).
        """
        budget = max(1.0 - self.slo_target, 1e-9)
        failure_rate = self.failed / self.total if self.total else 0.0
        remaining = 1.0 - failure_rate / budget
        return max(0.0, min(1.0, remaining))

    @property
    def slo_met(self) -> bool:
        return self.success_rate >= self.slo_target

    def snapshot(self) -> dict[str, float | int | bool]:
        return {
            "total_frames": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "manual_review": self.manual_review,
            "total_retries": self.retries,
            "success_rate": round(self.success_rate, 6),
            "manual_review_rate": round(self.manual_review_rate, 6),
            "slo_target": self.slo_target,
            "slo_met": self.slo_met,
            "error_budget_remaining": round(self.error_budget_remaining, 6),
        }


# --------------------------------------------------------------------------- #
# Observatory — the one object the pipeline threads through
# --------------------------------------------------------------------------- #
class Observatory:
    """Bundles metrics, tracer, health and reliability for one run."""

    def __init__(self, slo_target: float = 0.98) -> None:
        self.metrics = MetricsRegistry()
        self.tracer = Tracer(self.metrics)
        self.health = HealthRegistry()
        self.reliability = ReliabilityTracker(slo_target=slo_target)

    def span(self, name: str, **attrs: object):
        return self.tracer.span(name, **attrs)

    def snapshot(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.snapshot(),
            "reliability": self.reliability.snapshot(),
            "health": self.health.run(),
            "traces": [s.to_dict() for s in self.tracer.completed_roots],
        }
