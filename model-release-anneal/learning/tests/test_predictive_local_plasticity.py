from __future__ import annotations

import numpy as np

from learning.connectivity.constraints import constrain_w_hd_to_hd, constrain_w_hr_to_hd
from learning.plasticity.predictive_local import (
    update_predictive_local_weights,
    update_predictive_local_weights_block,
)


def test_predictive_local_update_respects_hd_diagonal_and_bounds() -> None:
    n_theta = 6
    w_hd_to_hd = np.zeros((n_theta, n_theta))
    w_hr_to_hd = np.zeros((n_theta, n_theta))
    delta_w_hd_to_hd = np.zeros_like(w_hd_to_hd)
    delta_w_hr_to_hd = np.zeros_like(w_hr_to_hd)
    e_hd = np.ones(n_theta)
    p_hd = np.ones(n_theta)
    p_hr = np.ones(n_theta)
    (
        next_w_hd_to_hd,
        next_w_hr_to_hd,
        next_delta_w_hd_to_hd,
        next_delta_w_hr_to_hd,
    ) = update_predictive_local_weights(
        w_hd_to_hd=w_hd_to_hd,
        w_hr_to_hd=w_hr_to_hd,
        delta_w_hd_to_hd=delta_w_hd_to_hd,
        delta_w_hr_to_hd=delta_w_hr_to_hd,
        e_hd=e_hd,
        p_hd=p_hd,
        p_hr=p_hr,
        dt=0.1,
        tau_delta=0.1,
        eta_hd_to_hd=1.0,
        eta_hr_to_hd=1.0,
        w_hd_to_hd_min=0.0,
        w_hd_to_hd_max=0.2,
        w_hr_to_hd_min=-0.2,
        w_hr_to_hd_max=0.2,
        zero_hd_to_hd_diagonal=True,
    )
    assert np.allclose(np.diag(next_w_hd_to_hd), 0.0)
    assert np.max(next_w_hd_to_hd) <= 0.2
    assert np.min(next_w_hr_to_hd) >= -0.2
    assert np.all(next_w_hr_to_hd >= 0.0)
    assert np.allclose(next_delta_w_hd_to_hd, 1.0)
    assert np.allclose(next_delta_w_hr_to_hd, 1.0)


def test_release_rate_units_need_only_ms_to_seconds_eta_conversion() -> None:
    """One update equals release ``w += 0.05 * Delta * dt_ms`` exactly."""
    dt_seconds = 0.0005
    dt_milliseconds = 0.5
    tau_delta_seconds = 0.1
    release_eta_per_millisecond_step = 0.05
    eta_for_seconds_engine = 1000.0 * release_eta_per_millisecond_step
    e_hd_khz = np.array([0.03])
    p_pre_khz = np.array([0.04])
    old_delta_khz_squared = np.array([[0.001]])

    (
        next_w_hd_to_hd,
        _next_w_hr_to_hd,
        next_delta_w_hd_to_hd,
        _next_delta_w_hr_to_hd,
    ) = update_predictive_local_weights(
        w_hd_to_hd=np.array([[0.2]]),
        w_hr_to_hd=np.array([[0.2]]),
        delta_w_hd_to_hd=old_delta_khz_squared,
        delta_w_hr_to_hd=old_delta_khz_squared,
        e_hd=e_hd_khz,
        p_hd=p_pre_khz,
        p_hr=p_pre_khz,
        dt=dt_seconds,
        tau_delta=tau_delta_seconds,
        eta_hd_to_hd=eta_for_seconds_engine,
        eta_hr_to_hd=eta_for_seconds_engine,
        w_hd_to_hd_min=None,
        w_hd_to_hd_max=None,
        w_hr_to_hd_min=None,
        w_hr_to_hd_max=None,
    )

    expected_delta = old_delta_khz_squared + (
        dt_milliseconds / 100.0
    ) * (np.outer(e_hd_khz, p_pre_khz) - old_delta_khz_squared)
    expected_weight = 0.2 + (
        release_eta_per_millisecond_step * expected_delta * dt_milliseconds
    )
    np.testing.assert_allclose(next_delta_w_hd_to_hd, expected_delta)
    np.testing.assert_allclose(next_w_hd_to_hd, expected_weight)


def test_block_update_matches_repeated_single_clock_plasticity_algebra() -> None:
    rng = np.random.default_rng(17)
    n_steps = 7
    n_hd = 5
    n_hr = 6
    dt = 0.0005
    tau_delta = 0.1
    e_hd_history = rng.normal(size=(n_steps, n_hd))
    p_hd_history = rng.normal(size=(n_steps, n_hd))
    p_hr_history = rng.normal(size=(n_steps, n_hr))
    initial_w_hd_to_hd = rng.normal(size=(n_hd, n_hd))
    initial_w_hr_to_hd = rng.normal(size=(n_hd, n_hr))
    initial_delta_w_hd_to_hd = rng.normal(size=(n_hd, n_hd))
    initial_delta_w_hr_to_hd = rng.normal(size=(n_hd, n_hr))

    repeated = (
        initial_w_hd_to_hd.copy(),
        initial_w_hr_to_hd.copy(),
        initial_delta_w_hd_to_hd.copy(),
        initial_delta_w_hr_to_hd.copy(),
    )
    for step_index in range(n_steps):
        repeated = update_predictive_local_weights(
            w_hd_to_hd=repeated[0],
            w_hr_to_hd=repeated[1],
            delta_w_hd_to_hd=repeated[2],
            delta_w_hr_to_hd=repeated[3],
            e_hd=e_hd_history[step_index],
            p_hd=p_hd_history[step_index],
            p_hr=p_hr_history[step_index],
            dt=dt,
            tau_delta=tau_delta,
            eta_hd_to_hd=3.0,
            eta_hr_to_hd=5.0,
            w_hd_to_hd_min=None,
            w_hd_to_hd_max=None,
            w_hr_to_hd_min=None,
            w_hr_to_hd_max=None,
        )

    blocked = update_predictive_local_weights_block(
        w_hd_to_hd=initial_w_hd_to_hd,
        w_hr_to_hd=initial_w_hr_to_hd,
        delta_w_hd_to_hd=initial_delta_w_hd_to_hd,
        delta_w_hr_to_hd=initial_delta_w_hr_to_hd,
        e_hd_history=e_hd_history,
        p_hd_history=p_hd_history,
        p_hr_history=p_hr_history,
        dt=dt,
        tau_delta=tau_delta,
        eta_hd_to_hd=3.0,
        eta_hr_to_hd=5.0,
        w_hd_to_hd_min=None,
        w_hd_to_hd_max=None,
        w_hr_to_hd_min=None,
        w_hr_to_hd_max=None,
    )

    for blocked_array, repeated_array in zip(blocked, repeated, strict=True):
        np.testing.assert_allclose(blocked_array, repeated_array, rtol=2e-14, atol=2e-14)


def test_hr_to_hd_antisymmetric_balance_removes_common_mode() -> None:
    w_hr_to_hd = np.array(
        [
            [1.0, 2.0, 0.5, -0.5],
            [-0.5, 0.25, 1.5, -1.0],
        ]
    )
    balanced_w_hr_to_hd = constrain_w_hr_to_hd(
        w_hr_to_hd,
        lower_bound=-2.0,
        upper_bound=2.0,
        balance_mode="antisymmetric_wings",
    )
    n_hr_per_wing = w_hr_to_hd.shape[1] // 2
    w_lhr_to_hd = balanced_w_hr_to_hd[:, :n_hr_per_wing]
    w_rhr_to_hd = balanced_w_hr_to_hd[:, n_hr_per_wing:]

    assert np.allclose(w_lhr_to_hd + w_rhr_to_hd, 0.0)
    assert np.allclose(
        w_rhr_to_hd - w_lhr_to_hd,
        w_hr_to_hd[:, n_hr_per_wing:] - w_hr_to_hd[:, :n_hr_per_wing],
    )


def test_null_weight_bounds_and_vafidis_default_keep_full_hd_matrix() -> None:
    w_hd_to_hd = np.array(
        [
            [9.0, 3.0, -4.0],
            [5.0, -8.0, 6.0],
            [-7.0, 8.0, 2.0],
        ]
    )
    w_hr_to_hd = np.array([[3.0, -4.0], [5.0, -6.0]])

    unconstrained_hd = constrain_w_hd_to_hd(
        w_hd_to_hd,
        lower_bound=None,
        upper_bound=None,
    )
    unconstrained_hr = constrain_w_hr_to_hd(
        w_hr_to_hd,
        lower_bound=None,
        upper_bound=None,
    )

    assert np.allclose(unconstrained_hd, w_hd_to_hd)
    assert np.allclose(unconstrained_hr, w_hr_to_hd)


def test_hd_to_hd_symmetric_constraint_removes_antisymmetric_component() -> None:
    w_hd_to_hd = np.array(
        [
            [0.0, 1.0, -0.5],
            [0.25, 0.0, 2.0],
            [0.75, -1.0, 0.0],
        ]
    )
    constrained_w_hd_to_hd = constrain_w_hd_to_hd(
        w_hd_to_hd,
        lower_bound=-2.0,
        upper_bound=2.0,
        symmetry_mode="symmetric",
        zero_diagonal=True,
    )

    assert np.allclose(constrained_w_hd_to_hd, constrained_w_hd_to_hd.T)
    assert np.allclose(np.diag(constrained_w_hd_to_hd), 0.0)


def test_hd_to_hd_zero_sum_constraint_removes_common_mode() -> None:
    w_hd_to_hd = np.ones((5, 5))
    np.fill_diagonal(w_hd_to_hd, 0.0)
    w_hd_to_hd[0, 1] += 2.0
    w_hd_to_hd[1, 0] += 2.0

    constrained_w_hd_to_hd = constrain_w_hd_to_hd(
        w_hd_to_hd,
        lower_bound=-5.0,
        upper_bound=5.0,
        symmetry_mode="symmetric",
        balance_mode="zero_sum",
    )

    assert np.allclose(constrained_w_hd_to_hd, constrained_w_hd_to_hd.T)
    assert np.allclose(np.diag(constrained_w_hd_to_hd), 0.0)
    assert abs(np.mean(constrained_w_hd_to_hd)) < 1e-6
    assert np.max(constrained_w_hd_to_hd) > 0.0
    assert np.min(constrained_w_hd_to_hd) < 0.0
