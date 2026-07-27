"""Linear and periodic population decoders."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def linear_center(values: NDArray[np.float64], positions: NDArray[np.float64]) -> float:
    """Return the nonnegative-weighted center of mass on a line."""

    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    positions = np.asarray(positions, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return float("nan")
    return float(np.dot(values, positions) / total)


def circular_center(values: NDArray[np.float64], positions: NDArray[np.float64], length: float) -> float:
    """Return the population-vector center on a periodic segment."""

    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(values.sum())
    if total <= 0:
        return float("nan")
    phase = 2.0 * np.pi * np.asarray(positions, dtype=float) / length
    vector = np.dot(values, np.exp(1j * phase)) / total
    return float(np.angle(vector) % (2.0 * np.pi) * length / (2.0 * np.pi))


def vector_strength(values: NDArray[np.float64], positions: NDArray[np.float64], length: float) -> float:
    """Return circular population-vector magnitude as a bump-validity measure."""

    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    phase = 2.0 * np.pi * np.asarray(positions, dtype=float) / length
    return float(abs(np.dot(values, np.exp(1j * phase)) / total))


def learned_competitive_positions(weights: NDArray[np.float64], input_positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Assign each competitive neuron the peak position of its learned row."""

    weights = np.asarray(weights, dtype=float)
    input_positions = np.asarray(input_positions, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != input_positions.size:
        raise ValueError("weights/input position mismatch")
    return input_positions[np.argmax(weights, axis=1)]

