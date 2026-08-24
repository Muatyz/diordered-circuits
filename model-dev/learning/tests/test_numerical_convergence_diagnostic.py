from __future__ import annotations

import numpy as np

from learning.common.random import make_rng
from learning.config.schema import ExperimentConfig
from learning.diagnostics.numerical_convergence import (
    run_numerical_convergence_diagnostic,
)
from learning.models.vafidis_toy import initialize_vafidis_toy_state


def _short_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.simulation.progress = False
    config.tests.numerical_convergence_duration = 0.04
    config.tests.numerical_convergence_sample_interval = 0.01
    return config


def test_whole_step_convergence_marks_unstable_euler_and_refines() -> None:
    config = _short_config()
    state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    history, metrics = run_numerical_convergence_diagnostic(
        config=config,
        trained_state=state,
    )

    dt = history["dt"]
    method = history["integration_method"].astype(str)
    euler_1ms = (method == "forward_euler") & np.isclose(dt, 0.001)
    exact_1ms = (method == "exact_linear") & np.isclose(dt, 0.001)
    assert not bool(history["valid_configuration"][euler_1ms][0])
    assert bool(history["valid_configuration"][exact_1ms][0])
    assert metrics["numerical_convergence_valid_fraction"] == 7.0 / 8.0

    for integration_method in ("forward_euler", "exact_linear"):
        mask = (
            (method == integration_method)
            & history["valid_configuration"]
            & history["finite_trajectory"]
        )
        order = np.argsort(dt[mask])[::-1]
        error = history["max_abs_heading_error"][mask][order]
        assert np.all(np.diff(error) < 0.0)


def test_release_literal_lowpass_parity_is_reported_not_applied() -> None:
    config = _short_config()
    state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    history, _metrics = run_numerical_convergence_diagnostic(
        config=config,
        trained_state=state,
    )

    np.testing.assert_allclose(
        history["paper_hd_to_hr_time_constant"],
        config.model.tau_s,
    )
    np.testing.assert_allclose(
        history["release_literal_hd_to_hr_effective_time_constant"],
        config.model.tau_s * history["dt"] * 1000.0,
    )
    half_ms = np.isclose(history["dt"], 0.0005)
    np.testing.assert_allclose(
        history["release_literal_hd_to_hr_effective_time_constant"][half_ms],
        0.0325,
    )
