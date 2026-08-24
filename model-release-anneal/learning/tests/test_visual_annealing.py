"""Tests for the visual-annealing schedule and pretrained-weight resume."""

import numpy as np

from learning.common.random import make_rng
from learning.config.schema import ExperimentConfig
from learning.models.vafidis_toy import (
    VafidisToyParams,
    initialize_vafidis_toy_state,
    step_vafidis_toy,
)
from learning.stimuli.visual import ScheduledVisualAmplitude


def _make_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.n_theta = 8
    config.model.n_hr = 8
    config.simulation.dt = 0.001
    config.simulation.proximal_integration_method = "exact_linear"
    config.simulation.progress = False
    config.visual.amplitude = 4.0
    config.visual.training_amplitude_schedule = [
        {"end_fraction": 0.5, "amplitude": 4.0},
        {"end_fraction": 0.75, "amplitude": 1.5},
        {"end_fraction": 1.0, "amplitude": 0.5},
    ]
    return config


def test_scheduled_visual_amplitude_piecewise_values() -> None:
    schedule = ScheduledVisualAmplitude(
        total_duration=100.0,
        amplitude_schedule=[
            {"end_fraction": 0.5, "amplitude": 4.0},
            {"end_fraction": 0.75, "amplitude": 1.5},
            {"end_fraction": 1.0, "amplitude": 0.5},
        ],
    )
    # Early phase: strong cue.
    assert np.isclose(schedule.current_amplitude(0.0), 4.0)
    schedule.elapsed_time = 40.0
    assert np.isclose(schedule.current_amplitude(0.0), 4.0)
    # Mid phase: annealed.
    schedule.elapsed_time = 55.0
    assert np.isclose(schedule.current_amplitude(0.0), 1.5)
    # Final phase: weak cue (night vision).
    schedule.elapsed_time = 90.0
    assert np.isclose(schedule.current_amplitude(0.0), 0.5)
    # Past the end stays at the final value.
    schedule.elapsed_time = 500.0
    assert np.isclose(schedule.current_amplitude(0.0), 0.5)


def test_scheduled_visual_amplitude_step_advances_clock() -> None:
    schedule = ScheduledVisualAmplitude(
        total_duration=100.0,
        amplitude_schedule=[
            {"end_fraction": 0.5, "amplitude": 4.0},
            {"end_fraction": 1.0, "amplitude": 0.5},
        ],
    )
    amplitude = schedule.step(1.0)
    assert np.isclose(amplitude, 4.0)
    assert np.isclose(schedule.elapsed_time, 1.0)
    schedule.elapsed_time = 60.0
    assert np.isclose(schedule.step(1.0), 0.5)


def test_scheduled_visual_amplitude_validates_schedule() -> None:
    import pytest

    with pytest.raises(ValueError, match="end_fraction"):
        ScheduledVisualAmplitude(
            total_duration=100.0,
            amplitude_schedule=[
                {"end_fraction": 0.8, "amplitude": 1.0},
                {"end_fraction": 0.5, "amplitude": 0.5},
            ],
        )
    with pytest.raises(ValueError, match="must end at fraction 1.0"):
        ScheduledVisualAmplitude(
            total_duration=100.0,
            amplitude_schedule=[{"end_fraction": 0.8, "amplitude": 1.0}],
        )
    with pytest.raises(ValueError, match="non-negative"):
        ScheduledVisualAmplitude(
            total_duration=100.0,
            amplitude_schedule=[{"end_fraction": 1.0, "amplitude": -1.0}],
        )


def test_step_uses_scheduled_visual_amplitude_override() -> None:
    """The override must change the proximal visual current."""
    config = _make_config()
    params = VafidisToyParams.from_config(config)
    state = initialize_vafidis_toy_state(config=config, rng=make_rng(7))
    # Steady visual current at amplitude 4 vs 0.5 must differ markedly.
    strong = step_vafidis_toy(
        state=state,
        params=params,
        angular_velocity=0.0,
        visual_teacher=True,
        training=False,
        visual_amplitude=4.0,
    )
    weak = step_vafidis_toy(
        state=state,
        params=params,
        angular_velocity=0.0,
        visual_teacher=True,
        training=False,
        visual_amplitude=0.5,
    )
    assert np.mean(np.abs(strong.i_vis_to_hd - weak.i_vis_to_hd)) > 0.1
    # Without an override the configured amplitude is used.
    default_step = step_vafidis_toy(
        state=state,
        params=params,
        angular_velocity=0.0,
        visual_teacher=True,
        training=False,
    )
    np.testing.assert_allclose(default_step.i_vis_to_hd, strong.i_vis_to_hd)
