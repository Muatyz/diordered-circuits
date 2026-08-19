from __future__ import annotations

import numpy as np

from learning.analysis.phase_flow import (
    actual_stable_basins,
    estimate_discrete_phase_flow,
)


def _integrate_scalar_field(
    initial: np.ndarray,
    time: np.ndarray,
    *,
    offset: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    theta = np.empty((initial.size, time.size), dtype=float)
    theta[:, 0] = initial
    for time_index in range(1, time.size):
        dt = float(time[time_index] - time[time_index - 1])
        theta[:, time_index] = theta[:, time_index - 1] + dt * (
            offset - amplitude * np.sin(theta[:, time_index - 1])
        )
    return theta


def test_discrete_phase_flow_recovers_actual_roots_and_basin() -> None:
    time = np.linspace(0.0, 0.5, 101)
    initial = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    theta = _integrate_scalar_field(initial, time, offset=0.0)

    estimate = estimate_discrete_phase_flow(
        time=time,
        theta=theta,
        commanded_velocity=0.0,
        fit_start_time=0.0,
        fit_duration=0.5,
        angular_bin_count=360,
        smoothing_bin_count=5,
        empirical_lambda_speed_floor=0.02,
    )

    stability = np.asarray(estimate["fixed_point_stability"])
    angle = np.asarray(estimate["fixed_point_angle"])
    stable = angle[stability == -1]
    unstable = angle[stability == 1]
    assert stable.size == 1
    assert unstable.size == 1
    assert np.min(np.abs(stable)) < np.deg2rad(2.0)
    assert np.min(np.abs(np.abs(unstable) - np.pi)) < np.deg2rad(2.0)
    basin = actual_stable_basins(
        fixed_point_angle=angle,
        fixed_point_stability=stability,
    )
    assert basin["stable_angle"].size == 1


def test_empirical_acceleration_ratio_matches_discrete_field_derivative() -> None:
    time = np.linspace(0.0, 0.4, 81)
    initial = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    theta = _integrate_scalar_field(initial, time, offset=0.3)
    estimate = estimate_discrete_phase_flow(
        time=time,
        theta=theta,
        commanded_velocity=0.3,
        fit_start_time=0.0,
        fit_duration=0.4,
        angular_bin_count=180,
        smoothing_bin_count=5,
        empirical_lambda_speed_floor=0.03,
    )

    derivative = np.asarray(estimate["phase_flow_derivative"])
    empirical_lambda = np.asarray(estimate["empirical_lambda"])
    assert np.all(np.isfinite(empirical_lambda))
    assert np.median(np.abs(derivative - empirical_lambda)) < 0.2


def test_de_pinned_discrete_field_has_no_fixed_points() -> None:
    time = np.linspace(0.0, 0.4, 81)
    initial = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    theta = _integrate_scalar_field(
        initial,
        time,
        offset=0.6,
        amplitude=0.2,
    )
    estimate = estimate_discrete_phase_flow(
        time=time,
        theta=theta,
        commanded_velocity=0.6,
        fit_start_time=0.0,
        fit_duration=0.4,
        angular_bin_count=180,
        smoothing_bin_count=5,
        empirical_lambda_speed_floor=0.03,
    )
    assert np.asarray(estimate["fixed_point_angle"]).size == 0
