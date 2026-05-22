"""AI Cinematic Face Swap Stabilization Engine (Phase 1).

The package root stays import-light on purpose: importing ``face_swap`` must
not pull in torch / cv2 / insightface. Submodules import heavy deps lazily so
that CPU-only tooling (config validation, the test suite, report generation)
works on a host without a GPU. See CLAUDE.md §11.3 and §18.13.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("face_swap")
except PackageNotFoundError:  # editable / source checkout without install
    __version__ = "1.0.0.dev"

__all__ = ["__version__"]
