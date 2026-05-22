"""Compute baseline metrics (CLAUDE.md §5.5). Thin wrapper over face_swap.baseline.

Usage:
  python scripts/measure_baseline.py [manifest.yaml] [out.json]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from face_swap.baseline import measure_all  # noqa: E402
from face_swap.logging_setup import configure  # noqa: E402


def main() -> int:
    configure()
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "tests/fixtures/test_set_manifest.yaml")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("baseline/v0_metrics.json")
    measure_all(manifest, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
