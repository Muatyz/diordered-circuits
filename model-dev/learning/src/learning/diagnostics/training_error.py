"""Online Vafidis Eq. (19) learning-error recording during training."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def learning_error_window_start_steps(
    *,
    total_steps: int,
    interval_fraction: float,
) -> np.ndarray:
    """Return unique window starts at 0%, interval, ..., below 100%."""

    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0.0 < interval_fraction <= 1.0:
        raise ValueError("interval_fraction must be in (0, 1]")
    fractions = np.arange(0.0, 1.0, interval_fraction, dtype=float)
    start_steps = np.rint(fractions * total_steps).astype(np.int64)
    return np.unique(np.clip(start_steps, 0, total_steps - 1))


@dataclass
class TrainingAbsoluteLearningErrorRecorder:
    """Stream 10 s mean absolute local prediction errors at fixed fractions.

    The recorded statistic is the discrete form of Vafidis et al. Eq. (19):
    the absolute firing-rate prediction error is averaged over HD neurons and
    over a short forward window beginning at each requested training fraction.
    Only active windows accumulate data, so the full per-step error history is
    never retained.
    """

    total_steps: int
    n_hd: int
    dt: float
    window_duration: float = 10.0
    interval_fraction: float = 0.01
    rate_scale_to_spikes_per_second: float = 1000.0
    window_start_steps: np.ndarray = field(init=False)
    window_end_steps: np.ndarray = field(init=False)
    sample_count: np.ndarray = field(init=False)
    absolute_error_sum_per_neuron: np.ndarray = field(init=False)
    _next_window_index: int = field(init=False, default=0)
    _active_window_indices: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.n_hd < 1:
            raise ValueError("n_hd must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.window_duration <= 0.0:
            raise ValueError("window_duration must be positive")
        if self.rate_scale_to_spikes_per_second <= 0.0:
            raise ValueError("rate_scale_to_spikes_per_second must be positive")
        self.window_start_steps = learning_error_window_start_steps(
            total_steps=self.total_steps,
            interval_fraction=self.interval_fraction,
        )
        window_steps = max(1, int(round(self.window_duration / self.dt)))
        self.window_end_steps = np.minimum(
            self.window_start_steps + window_steps,
            self.total_steps,
        )
        window_count = self.window_start_steps.size
        self.sample_count = np.zeros(window_count, dtype=np.int64)
        self.absolute_error_sum_per_neuron = np.zeros(
            (window_count, self.n_hd),
            dtype=float,
        )

    def update(self, *, completed_step: int, e_hd: np.ndarray) -> None:
        """Accumulate one completed model step into every active window."""

        if not 1 <= completed_step <= self.total_steps:
            raise ValueError("completed_step must be in [1, total_steps]")
        e_hd = np.asarray(e_hd, dtype=float)
        if e_hd.shape != (self.n_hd,):
            raise ValueError(f"e_hd must have shape ({self.n_hd},)")

        while (
            self._next_window_index < self.window_start_steps.size
            and self.window_start_steps[self._next_window_index] < completed_step
        ):
            self._active_window_indices.append(self._next_window_index)
            self._next_window_index += 1

        if not self._active_window_indices:
            return
        absolute_error = np.abs(e_hd)
        still_active: list[int] = []
        for window_index in self._active_window_indices:
            if completed_step <= self.window_end_steps[window_index]:
                self.absolute_error_sum_per_neuron[window_index] += absolute_error
                self.sample_count[window_index] += 1
            if completed_step < self.window_end_steps[window_index]:
                still_active.append(window_index)
        self._active_window_indices = still_active

    def to_history(self) -> dict[str, np.ndarray]:
        """Return completed or early-truncated windows in plotting units."""

        recorded = self.sample_count > 0
        counts = self.sample_count[recorded]
        start_steps = self.window_start_steps[recorded]
        mean_per_neuron = (
            self.absolute_error_sum_per_neuron[recorded]
            / counts[:, None]
            * self.rate_scale_to_spikes_per_second
        )
        return {
            "absolute_learning_error_time": start_steps.astype(float) * self.dt,
            "absolute_learning_error_window_start_time": (
                start_steps.astype(float) * self.dt
            ),
            "absolute_learning_error_window_end_time": (
                (start_steps + counts).astype(float) * self.dt
            ),
            "absolute_learning_error_mean_spikes_per_s": np.mean(
                mean_per_neuron,
                axis=1,
            ),
            "absolute_learning_error_per_neuron_spikes_per_s": mean_per_neuron,
            "absolute_learning_error_window_sample_count": counts,
            "absolute_learning_error_window_duration": np.asarray(
                self.window_duration,
                dtype=float,
            ),
            "absolute_learning_error_interval_fraction": np.asarray(
                self.interval_fraction,
                dtype=float,
            ),
            "absolute_learning_error_rate_scale_to_spikes_per_second": np.asarray(
                self.rate_scale_to_spikes_per_second,
                dtype=float,
            ),
        }
