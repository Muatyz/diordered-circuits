from __future__ import annotations

import numpy as np
import pytest

from learning.common.random import make_rng
from learning.config.schema import ExperimentConfig
from learning.diagnostics.weight_development import (
    run_weight_snapshot_pi_development_diagnostic,
    select_weight_snapshot_indices,
)
from learning.models.vafidis_toy import initialize_vafidis_toy_state


def make_diagnostic_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.n_theta = 12
    config.model.n_hr = 12
    config.simulation.seed = 7
    config.simulation.dt = 0.0005
    config.simulation.progress = False
    config.tests.weight_snapshot_pi_velocities = [-0.4, 0.0, 0.4]
    config.tests.weight_snapshot_pi_initial_heading = 0.25
    config.tests.weight_snapshot_pi_interval_fraction = 0.01
    config.tests.weight_snapshot_pi_cue_duration = 0.003
    config.tests.weight_snapshot_pi_duration = 0.006
    config.tests.weight_snapshot_pi_average_start_time = 0.002
    return config


def test_snapshot_pi_diagnostic_is_frozen_and_records_each_condition() -> None:
    config = make_diagnostic_config()
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )
    w_hd_before = trained_state.w_hd_to_hd.copy()
    w_hr_before = trained_state.w_hr_to_hd.copy()
    weight_history = {
        "time": np.asarray([0.0, 10.0]),
        "w_hd_to_hd": np.stack(
            [trained_state.w_hd_to_hd, 1.1 * trained_state.w_hd_to_hd]
        ),
        "w_hr_to_hd": np.stack(
            [trained_state.w_hr_to_hd, 1.1 * trained_state.w_hr_to_hd]
        ),
    }

    history, metrics = run_weight_snapshot_pi_development_diagnostic(
        config=config,
        trained_state=trained_state,
        weight_history=weight_history,
    )

    assert history["time_averaged_abs_pi_error"].shape == (2, 3)
    assert history["rms_pi_error"].shape == (2, 3)
    assert history["minimum_pva_strength"].shape == (2, 3)
    assert history["velocity_bias"].shape == (2, 3)
    assert history["aggregate_rms_velocity_bias"].shape == (2,)
    assert history["effective_weight_growth_hd_to_hd"].shape == (2,)
    assert history["effective_weight_growth_hr_to_hd"].shape == (2,)
    assert history["selection_metric"] == "mean_abs_unwrapped_error"
    assert np.isfinite(history["time_averaged_abs_pi_error"]).all()
    assert np.isfinite(history["aggregate_time_averaged_abs_pi_error"]).all()
    assert metrics["weight_snapshot_pi_snapshot_count"] == 2.0
    assert metrics["weight_snapshot_pi_velocity_count"] == 3.0
    np.testing.assert_allclose(trained_state.w_hd_to_hd, w_hd_before)
    np.testing.assert_allclose(trained_state.w_hr_to_hd, w_hr_before)


def test_snapshot_pi_diagnostic_uses_final_state_when_history_is_missing() -> None:
    config = make_diagnostic_config()
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )
    trained_state.time = 12.5

    history, metrics = run_weight_snapshot_pi_development_diagnostic(
        config=config,
        trained_state=trained_state,
        weight_history={},
    )

    np.testing.assert_allclose(history["snapshot_time"], [12.5])
    assert history["time_averaged_abs_pi_error"].shape == (1, 3)
    assert metrics["weight_snapshot_pi_snapshot_count"] == 1.0


def test_snapshot_pi_selection_aggregates_multiple_initial_headings() -> None:
    config = make_diagnostic_config()
    config.tests.weight_snapshot_pi_initial_headings = [-1.0, 0.25, 1.5]
    config.tests.weight_snapshot_pi_selection_metric = "rms_velocity_bias"
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    history, metrics = run_weight_snapshot_pi_development_diagnostic(
        config=config,
        trained_state=trained_state,
        weight_history={},
    )

    assert history["time_averaged_abs_pi_error"].shape == (1, 3)
    assert history[
        "time_averaged_abs_pi_error_by_initial_heading"
    ].shape == (1, 3, 3)
    assert history["velocity_bias_by_initial_heading"].shape == (1, 3, 3)
    np.testing.assert_allclose(history["initial_heading"], [-1.0, 0.25, 1.5])
    assert history["selection_metric"] == "rms_velocity_bias"
    assert metrics["weight_snapshot_pi_initial_heading_count"] == 3.0


def test_snapshot_pi_selection_rejects_duplicate_wrapped_headings() -> None:
    config = make_diagnostic_config()
    config.tests.weight_snapshot_pi_initial_headings = [0.0, 2.0 * np.pi]
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    with pytest.raises(ValueError, match="unique modulo"):
        run_weight_snapshot_pi_development_diagnostic(
            config=config,
            trained_state=trained_state,
            weight_history={},
        )


def test_snapshot_pi_diagnostic_rejects_empty_velocity_grid() -> None:
    config = make_diagnostic_config()
    config.tests.weight_snapshot_pi_velocities = []
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    with pytest.raises(ValueError, match="non-empty"):
        run_weight_snapshot_pi_development_diagnostic(
            config=config,
            trained_state=trained_state,
            weight_history={},
        )


def test_snapshot_pi_selection_metric_is_explicitly_validated() -> None:
    config = make_diagnostic_config()
    config.tests.weight_snapshot_pi_selection_metric = "mystery"
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    with pytest.raises(ValueError, match="weight_snapshot_pi_selection_metric"):
        run_weight_snapshot_pi_development_diagnostic(
            config=config,
            trained_state=trained_state,
            weight_history={},
        )


def test_snapshot_fraction_selects_one_point_per_percent() -> None:
    snapshot_time = np.linspace(0.0, 80_000.0, 1001)

    selected = select_weight_snapshot_indices(
        snapshot_time=snapshot_time,
        interval_fraction=0.01,
    )

    assert selected.size == 101
    np.testing.assert_allclose(snapshot_time[selected], np.linspace(0.0, 80_000.0, 101))
