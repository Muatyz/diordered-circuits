from __future__ import annotations

import numpy as np
import pytest

from learning.diagnostics.training_error import (
    TrainingAbsoluteLearningErrorRecorder,
    learning_error_window_start_steps,
)


def test_learning_error_recorder_matches_population_and_time_absolute_mean() -> None:
    recorder = TrainingAbsoluteLearningErrorRecorder(
        total_steps=20,
        n_hd=2,
        dt=0.5,
        window_duration=2.0,
        interval_fraction=0.5,
        rate_scale_to_spikes_per_second=10.0,
    )
    for completed_step in range(1, 21):
        recorder.update(
            completed_step=completed_step,
            e_hd=np.asarray([completed_step, -2.0 * completed_step]),
        )

    history = recorder.to_history()
    np.testing.assert_allclose(history["absolute_learning_error_time"], [0.0, 5.0])
    np.testing.assert_allclose(
        history["absolute_learning_error_window_end_time"],
        [2.0, 7.0],
    )
    np.testing.assert_array_equal(
        history["absolute_learning_error_window_sample_count"],
        [4, 4],
    )
    np.testing.assert_allclose(
        history["absolute_learning_error_per_neuron_spikes_per_s"],
        [[25.0, 50.0], [125.0, 250.0]],
    )
    np.testing.assert_allclose(
        history["absolute_learning_error_mean_spikes_per_s"],
        [37.5, 187.5],
    )


def test_learning_error_recorder_supports_overlapping_and_truncated_windows() -> None:
    recorder = TrainingAbsoluteLearningErrorRecorder(
        total_steps=10,
        n_hd=3,
        dt=1.0,
        window_duration=4.0,
        interval_fraction=0.2,
        rate_scale_to_spikes_per_second=1.0,
    )
    for completed_step in range(1, 11):
        recorder.update(completed_step=completed_step, e_hd=np.asarray([1.0, -2.0, 3.0]))

    history = recorder.to_history()
    np.testing.assert_array_equal(
        history["absolute_learning_error_window_sample_count"],
        [4, 4, 4, 4, 2],
    )
    np.testing.assert_allclose(
        history["absolute_learning_error_mean_spikes_per_s"],
        np.full(5, 2.0),
    )


def test_learning_error_window_start_steps_are_unique_for_short_runs() -> None:
    np.testing.assert_array_equal(
        learning_error_window_start_steps(total_steps=4, interval_fraction=0.01),
        [0, 1, 2, 3],
    )


@pytest.mark.parametrize("interval_fraction", [0.0, -0.1, 1.1])
def test_learning_error_window_start_steps_reject_invalid_fraction(
    interval_fraction: float,
) -> None:
    with pytest.raises(ValueError, match="interval_fraction"):
        learning_error_window_start_steps(
            total_steps=10,
            interval_fraction=interval_fraction,
        )
