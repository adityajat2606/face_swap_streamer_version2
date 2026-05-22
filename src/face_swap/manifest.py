"""Run identity, file hashing, and the processing_log.json manifest (§17.2).

Determinism (§2.4): a manifest records model hashes, config hash, input hashes,
and the random seed so a run is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    """``YYYYMMDD_HHMMSS_<6-hex>`` (CLAUDE.md §20)."""
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{secrets.token_hex(3)}"


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_file_or_none(path: str | Path | None) -> str | None:
    if path and Path(path).is_file():
        return sha256_file(path)
    return None


def git_sha(repo_dir: str | Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def model_versions(model_paths: dict[str, str | None]) -> dict[str, str]:
    """Map model role -> ``name@sha12`` for the manifest and resume guard."""
    out: dict[str, str] = {}
    for role, path in model_paths.items():
        if path and Path(path).is_file():
            out[role] = f"{Path(path).name}@{sha256_file(path)[:12]}"
        else:
            out[role] = "absent"
    return out


def redact(path: str | None, redact_paths: bool) -> str | None:
    """Hash a path when redaction is on (§18.9)."""
    if path is None:
        return None
    if redact_paths:
        return "sha256:" + hashlib.sha256(path.encode()).hexdigest()[:16]
    return path


def write_manifest(run_dir: Path, manifest: dict) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "processing_log.json"
    tmp = run_dir / "processing_log.json.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    return out
