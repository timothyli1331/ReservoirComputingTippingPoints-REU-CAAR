"""Cached numerical basin boundaries used as a global reward oracle."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np


LabelFunction = Callable[[tuple[float, float]], int]


class GlobalBoundary:
    """Represent a curved basin boundary by points extracted from a label grid.

    The costly basin classification is performed only by :meth:`rebuild`.
    Distance queries made at every environment step are simple vectorized
    nearest-point calculations.
    """

    def __init__(
        self,
        x_limits: tuple[float, float],
        v_limits: tuple[float, float],
        resolution: int = 41,
    ) -> None:
        if resolution < 10:
            raise ValueError("boundary resolution must be at least 10")
        self.x_limits = tuple(map(float, x_limits))
        self.v_limits = tuple(map(float, v_limits))
        self.resolution = int(resolution)
        self.points = np.empty((0, 2), dtype=float)
        self.labels = np.empty((0, 0), dtype=np.int8)

    def rebuild(self, label_function: LabelFunction) -> None:
        """Classify a global grid and extract cells separating the two basins."""
        xs = np.linspace(*self.x_limits, self.resolution)
        vs = np.linspace(*self.v_limits, self.resolution)
        labels = np.empty((self.resolution, self.resolution), dtype=np.int8)
        for row, velocity in enumerate(vs):
            for column, position in enumerate(xs):
                labels[row, column] = 1 if label_function((position, velocity)) >= 0 else -1

        points: list[tuple[float, float]] = []
        # Add midpoints of every horizontal or vertical grid edge whose basin
        # label changes. These points follow the full curved separatrix.
        horizontal = labels[:, 1:] != labels[:, :-1]
        for row, column in np.argwhere(horizontal):
            points.append(((xs[column] + xs[column + 1]) / 2.0, vs[row]))
        vertical = labels[1:, :] != labels[:-1, :]
        for row, column in np.argwhere(vertical):
            points.append((xs[column], (vs[row] + vs[row + 1]) / 2.0))
        if not points:
            raise RuntimeError(
                "no basin boundary was found; enlarge the boundary domain or horizon"
            )
        self.labels = labels
        self.points = np.asarray(points, dtype=float)

    def distance(self, state: np.ndarray, scales: tuple[float, float] = (1.0, 1.0)) -> float:
        """Return normalized Euclidean distance to the nearest boundary point."""
        if not len(self.points):
            raise RuntimeError("global boundary has not been built")
        scale = np.asarray(scales, dtype=float)
        if np.any(scale <= 0):
            raise ValueError("boundary distance scales must be positive")
        offsets = (self.points - np.asarray(state, dtype=float)) / scale
        return float(np.sqrt(np.min(np.sum(offsets * offsets, axis=1))))

    def snapshot(self) -> Mapping[str, np.ndarray]:
        """Return copies suitable for plotting without exposing mutable state."""
        return {"points": self.points.copy(), "labels": self.labels.copy()}
