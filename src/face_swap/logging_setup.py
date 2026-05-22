"""Structured logging via structlog (CLAUDE.md §4.5).

Every log line carries ``run_id``, ``stage`` and (when applicable) ``frame_idx``
via context vars bound by the pipeline. No ``print()`` outside cli.py.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure(level: str = "INFO", json: bool = False) -> None:
    """Configure structlog + stdlib logging. Idempotent."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stderr,
        format="%(message)s",
        force=True,
    )
    renderer = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "face_swap") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind_run_context(**kwargs: object) -> None:
    """Bind run-scoped fields (run_id, stage, ...) onto every subsequent log."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_run_context() -> None:
    structlog.contextvars.clear_contextvars()
