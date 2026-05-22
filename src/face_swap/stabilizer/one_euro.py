"""One-Euro filter (Casiez et al., 2012) — CLAUDE.md §8.1.

Responsive to true motion, smooths jitter at low motion, two tunable params.
Used to smooth bbox corners, landmark coordinates, and mask boundaries.
"""

from __future__ import annotations

import numpy as np


class OneEuroFilter:
    """Scalar One-Euro filter. Call once per timestep with ``(value, t)``."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoffs must be positive")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0
        self.t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        x = float(x)
        if self.t_prev is None:
            self.t_prev, self.x_prev = t, x
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt  # type: ignore[operator]
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev  # type: ignore[operator]
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class VectorOneEuro:
    """Independent One-Euro filters over a fixed-length vector (e.g. landmark
    coords flattened, or the 4 bbox corners)."""

    def __init__(self, size: int, min_cutoff: float = 1.0, beta: float = 0.007):
        self._filters = [OneEuroFilter(min_cutoff, beta) for _ in range(size)]
        self.size = size

    def __call__(self, values: np.ndarray, t: float) -> np.ndarray:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size != self.size:
            raise ValueError(f"expected {self.size} values, got {flat.size}")
        out = np.array([f(v, t) for f, v in zip(self._filters, flat, strict=True)],
                       dtype=np.float32)
        return out.reshape(np.asarray(values).shape)

    def reset(self) -> None:
        for f in self._filters:
            f.reset()
