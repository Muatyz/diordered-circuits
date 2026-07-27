"""Map saved simulation samples onto honest neural and learning timescales."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from prospective.config.schema import ExperimentConfig


def neural_frame_indices(times: NDArray[np.float64], config: ExperimentConfig) -> NDArray[np.int64]:
    """Select a short continuous neural-dynamics window after transient."""

    times = np.asarray(times, dtype=float)
    start = max(config.analysis.transient_duration, max(0.0, float(times[-1]) - config.animation.neural_window_seconds))
    mask = (times >= start) & (times <= start + config.animation.neural_window_seconds)
    indices = np.flatnonzero(mask)
    if indices.size < 2:
        raise ValueError("not enough saved state samples for neural animation window")
    return indices.astype(np.int64)


def learning_frame_indices(times: NDArray[np.float64], config: ExperimentConfig) -> NDArray[np.int64]:
    """Select nearest samples at fixed tutor-cycle intervals for a montage."""

    times = np.asarray(times, dtype=float)
    speed = abs(config.tutor.speed)
    if speed <= 1e-12:
        targets = np.linspace(times[0], times[-1], min(20, len(times)))
    else:
        period = config.geometry.length / speed * config.animation.learning_frame_interval_cycles
        targets = np.arange(times[0], times[-1] + 0.5 * period, period)
    indices = np.unique([int(np.argmin(np.abs(times - target))) for target in targets])
    if indices.size < 2:
        indices = np.asarray([0, len(times) - 1], dtype=int)
    hold = max(1, int(round(config.animation.fps * config.animation.learning_hold_seconds)))
    return np.repeat(indices, hold).astype(np.int64)


def global_training_frame_indices(
    times: NDArray[np.float64], config: ExperimentConfig
) -> NDArray[np.int64]:
    """Uniformly sample the complete saved training interval, including both ends."""

    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2 or not np.all(np.diff(times) >= 0):
        raise ValueError("global animation requires at least two ordered time samples")
    count = min(config.animation.global_frame_count, times.size)
    targets = np.linspace(float(times[0]), float(times[-1]), count)
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 0, times.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(times[left] - targets) <= np.abs(times[right] - targets)
    indices = np.where(choose_left, left, right)
    indices = np.unique(indices)
    if indices[0] != 0:
        indices = np.insert(indices, 0, 0)
    if indices[-1] != times.size - 1:
        indices = np.append(indices, times.size - 1)
    return indices.astype(np.int64)


def top_update_connections(delta_weights: NDArray[np.float64], top_k: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return post/pre indices of the largest absolute instantaneous updates."""

    delta = np.asarray(delta_weights, dtype=float)
    if delta.ndim != 2 or top_k < 1:
        raise ValueError("delta_weights must be a matrix and top_k positive")
    count = min(top_k, delta.size)
    flat = np.argpartition(np.abs(delta).ravel(), -count)[-count:]
    post, pre = np.unravel_index(flat, delta.shape)
    order = np.argsort(np.abs(delta[post, pre]))[::-1]
    return post[order].astype(np.int64), pre[order].astype(np.int64)
