"""Value objects exchanged between modules.

Modules talk through these dataclasses, never raw dicts/tuples (CLAUDE.md §4.2).
Array-bearing types use ``eq=False`` so a frozen instance stays hashable-by-id
and never triggers numpy's ambiguous-truth ``__eq__`` when compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Identity = Literal["hero", "heroine", "unknown"]
QualityVerdict = Literal["PASS", "WARNING", "FAIL"]


@dataclass(slots=True, frozen=True)
class BBox:
    """Axis-aligned face box in pixel coordinates with a detector score."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    @property
    def diag(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: BBox) -> float:
        """Intersection-over-union with another box, in ``[0, 1]``."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
        inter = iw * ih
        union = self.area + other.area - inter
        return float(inter / union) if union > 0 else 0.0


@dataclass(slots=True, frozen=True, eq=False)
class Landmarks:
    """Facial landmark points, shape ``(N, 2)`` float32; N in {5, 68, 106}."""

    points: np.ndarray
    score: float = 1.0


@dataclass(slots=True, frozen=True, eq=False)
class FaceDetection:
    """One detected face in one frame."""

    frame_idx: int
    bbox: BBox
    landmarks: Landmarks
    embedding: np.ndarray | None = None  # (512,) unit-norm ArcFace, if computed
    pose: tuple[float, float, float] | None = None  # yaw, pitch, roll (deg)
    occlusion: float | None = None  # [0, 1]; 1 = fully occluded


@dataclass(slots=True, frozen=True, eq=False)
class TrackState:
    """The tracker's per-identity record."""

    track_id: int
    identity: Identity
    last_embedding: np.ndarray  # (512,) float32, rolling-median
    last_bbox: BBox
    last_frame_seen: int
    embedding_history: tuple[np.ndarray, ...] = ()  # capped at 30
    velocity: tuple[float, float] = (0.0, 0.0)  # bbox-center px/frame
    confidence: float = 1.0
    active: bool = True


@dataclass(slots=True, frozen=True, eq=False)
class SwapResult:
    """Output of the swap engine for one face."""

    frame_idx: int
    swapped_frame: np.ndarray  # full BGR frame with the face pasted back
    mask: np.ndarray  # float32 [0,1], same H,W as frame
    face_bbox: BBox
    landmarks: Landmarks | None = None


@dataclass(slots=True, frozen=True)
class FrameResult:
    """The verdict + metrics for one processed frame (one line of quality.jsonl)."""

    frame_idx: int
    verdict: QualityVerdict
    flicker_score: float
    components: dict[str, float] = field(default_factory=dict)
    retry_count: int = 0
    retry_strategies: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    duration_ms: float = 0.0
    vram_peak_mb: int = 0

    def replace(self, **changes: Any) -> FrameResult:
        """Frozen-friendly update (``dataclasses.replace`` wrapper)."""
        from dataclasses import replace as _replace

        return _replace(self, **changes)

    def to_json_obj(self) -> dict:
        """Serialize for a quality.jsonl line."""
        return {
            "frame": self.frame_idx,
            "verdict": self.verdict,
            "flicker_score": round(self.flicker_score, 6),
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "retry_count": self.retry_count,
            "retry_strategies": list(self.retry_strategies),
            "reasons": list(self.reasons),
            "duration_ms": round(self.duration_ms, 3),
            "vram_peak_mb": self.vram_peak_mb,
        }
