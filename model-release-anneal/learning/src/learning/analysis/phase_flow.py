"""Direct discrete phase-flow analysis for frozen HD-network probes."""

from __future__ import annotations

import numpy as np

from learning.common.angles import wrap_angle


def _circular_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("circular smoothing window must be odd")
    half_window = window // 2
    padded = np.concatenate([values[-half_window:], values, values[:half_window]])
    kernel = np.full(window, 1.0 / float(window), dtype=float)
    return np.convolve(padded, kernel, mode="valid")


def _periodic_interpolate(
    angle: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Fill empty circular bins by linear interpolation between observed bins."""
    angle = np.asarray(angle, dtype=float)
    values = np.asarray(values, dtype=float)
    observed = np.isfinite(values)
    if np.count_nonzero(observed) < 2:
        raise ValueError("discrete phase flow needs at least two observed bins")
    observed_angle = angle[observed]
    observed_value = values[observed]
    return np.interp(
        angle,
        np.concatenate(
            [
                observed_angle - 2.0 * np.pi,
                observed_angle,
                observed_angle + 2.0 * np.pi,
            ]
        ),
        np.tile(observed_value, 3),
    )


def _periodic_roots(
    *,
    angle: np.ndarray,
    value: np.ndarray,
    derivative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find sign-changing roots of a sampled periodic scalar field."""
    angle = np.asarray(angle, dtype=float)
    value = np.asarray(value, dtype=float)
    derivative = np.asarray(derivative, dtype=float)
    angular_resolution = 2.0 * np.pi / angle.size
    zero_tolerance = max(1e-12, 1e-10 * float(np.max(np.abs(value))))
    root_angle: list[float] = []
    root_slope: list[float] = []
    root_stability: list[int] = []
    for index in range(angle.size):
        next_index = (index + 1) % angle.size
        previous_index = (index - 1) % angle.size
        left_value = float(value[index])
        right_value = float(value[next_index])
        if abs(left_value) <= zero_tolerance:
            previous_value = float(value[previous_index])
            if previous_value * right_value >= 0.0:
                continue
            root = float(angle[index])
            slope = float(derivative[index])
            stability = -1 if previous_value > right_value else 1
        elif left_value * right_value < 0.0:
            left_angle = float(angle[index])
            right_angle = float(angle[next_index])
            if next_index == 0:
                right_angle += 2.0 * np.pi
            fraction = -left_value / (right_value - left_value)
            root = float(wrap_angle(left_angle + fraction * (right_angle - left_angle)))
            slope = float(
                derivative[index]
                + fraction * (derivative[next_index] - derivative[index])
            )
            stability = -1 if left_value > right_value else 1
        else:
            continue
        if any(
            abs(float(wrap_angle(root - existing))) < 0.5 * angular_resolution
            for existing in root_angle
        ):
            continue
        root_angle.append(root)
        root_slope.append(slope)
        root_stability.append(stability)
    return (
        np.asarray(root_angle, dtype=float),
        np.asarray(root_slope, dtype=float),
        np.asarray(root_stability, dtype=np.int8),
    )


def actual_stable_basins(
    *,
    fixed_point_angle: np.ndarray,
    fixed_point_stability: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return one-dimensional basins bounded by adjacent unstable roots."""
    angle = np.asarray(fixed_point_angle, dtype=float)
    stability = np.asarray(fixed_point_stability, dtype=np.int8)
    if angle.size == 0:
        return {
            "stable_angle": np.empty(0),
            "left_boundary": np.empty(0),
            "right_boundary": np.empty(0),
        }
    order = np.argsort(angle)
    angle = angle[order]
    stability = stability[order]
    stable_angle: list[float] = []
    left_boundary: list[float] = []
    right_boundary: list[float] = []
    for index, candidate_stability in enumerate(stability):
        if candidate_stability != -1:
            continue
        previous_index = (index - 1) % angle.size
        next_index = (index + 1) % angle.size
        if stability[previous_index] != 1 or stability[next_index] != 1:
            continue
        stable_angle.append(float(angle[index]))
        left_boundary.append(float(angle[previous_index]))
        right_boundary.append(float(angle[next_index]))
    return {
        "stable_angle": np.asarray(stable_angle, dtype=float),
        "left_boundary": np.asarray(left_boundary, dtype=float),
        "right_boundary": np.asarray(right_boundary, dtype=float),
    }


def estimate_discrete_phase_flow(
    *,
    time: np.ndarray,
    theta: np.ndarray,
    commanded_velocity: float,
    fit_start_time: float,
    fit_duration: float,
    angular_bin_count: int,
    smoothing_bin_count: int,
    empirical_lambda_speed_floor: float,
) -> dict[str, np.ndarray | float]:
    """Estimate F_v(theta) directly by binning dense probe trajectories."""
    time = np.asarray(time, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if time.ndim != 1 or time.size < 3:
        raise ValueError("phase-flow time must contain at least three samples")
    if theta.ndim != 2 or theta.shape[1] != time.size or theta.shape[0] < 3:
        raise ValueError("phase-flow theta must have shape (probe, time)")
    if angular_bin_count < 8:
        raise ValueError("phase-flow angular bin count must be at least eight")
    if smoothing_bin_count <= 0 or smoothing_bin_count % 2 == 0:
        raise ValueError("phase-flow smoothing bin count must be positive and odd")
    if smoothing_bin_count > angular_bin_count:
        raise ValueError("phase-flow smoothing window exceeds angular grid")
    if fit_start_time < 0.0 or fit_duration <= 0.0:
        raise ValueError("phase-flow fit window is invalid")
    if empirical_lambda_speed_floor <= 0.0:
        raise ValueError("empirical-lambda speed floor must be positive")

    fit_end_time = min(float(time[-1]), fit_start_time + fit_duration)
    fit_mask = (time >= fit_start_time) & (time <= fit_end_time)
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError("phase-flow fit window must contain at least three samples")

    unwrapped_theta = np.unwrap(theta, axis=1)
    phase_velocity = np.gradient(unwrapped_theta, time, axis=1, edge_order=2)
    phase_acceleration = np.gradient(phase_velocity, time, axis=1, edge_order=2)
    sample_angle = np.asarray(wrap_angle(theta[:, fit_mask]), dtype=float).ravel()
    sample_velocity = phase_velocity[:, fit_mask].ravel()
    sample_acceleration = phase_acceleration[:, fit_mask].ravel()
    finite = (
        np.isfinite(sample_angle)
        & np.isfinite(sample_velocity)
        & np.isfinite(sample_acceleration)
    )
    sample_angle = sample_angle[finite]
    sample_velocity = sample_velocity[finite]
    sample_acceleration = sample_acceleration[finite]

    angular_edges = np.linspace(-np.pi, np.pi, angular_bin_count + 1)
    grid_angle = 0.5 * (angular_edges[:-1] + angular_edges[1:])
    angular_step = 2.0 * np.pi / angular_bin_count
    bin_index = np.floor(
        (sample_angle + np.pi) / (2.0 * np.pi) * angular_bin_count
    ).astype(int) % angular_bin_count
    bin_sample_count = np.bincount(bin_index, minlength=angular_bin_count)
    phase_flow_raw = np.full(angular_bin_count, np.nan, dtype=float)
    empirical_lambda_raw = np.full(angular_bin_count, np.nan, dtype=float)
    lambda_speed_floor = max(
        empirical_lambda_speed_floor,
        0.05 * abs(float(commanded_velocity)),
    )
    reliable_lambda_sample = np.abs(sample_velocity) >= lambda_speed_floor
    empirical_lambda_sample = np.divide(
        sample_acceleration,
        sample_velocity,
        out=np.full_like(sample_velocity, np.nan),
        where=reliable_lambda_sample,
    )
    for current_bin in range(angular_bin_count):
        current = bin_index == current_bin
        if np.any(current):
            phase_flow_raw[current_bin] = float(np.median(sample_velocity[current]))
        if not np.isclose(commanded_velocity, 0.0):
            current_lambda = current & np.isfinite(empirical_lambda_sample)
            if np.any(current_lambda):
                empirical_lambda_raw[current_bin] = float(
                    np.median(empirical_lambda_sample[current_lambda])
                )

    observed = np.isfinite(phase_flow_raw)
    phase_flow_interpolated = _periodic_interpolate(grid_angle, phase_flow_raw)
    phase_flow = _circular_moving_average(
        phase_flow_interpolated,
        smoothing_bin_count,
    )
    phase_flow_derivative = (
        np.roll(phase_flow, -1) - np.roll(phase_flow, 1)
    ) / (2.0 * angular_step)
    fixed_angle, fixed_slope, fixed_stability = _periodic_roots(
        angle=grid_angle,
        value=phase_flow,
        derivative=phase_flow_derivative,
    )

    empirical_lambda = np.full(angular_bin_count, np.nan, dtype=float)
    if not np.isclose(commanded_velocity, 0.0) and np.count_nonzero(
        np.isfinite(empirical_lambda_raw)
    ) >= 2:
        empirical_lambda = _circular_moving_average(
            _periodic_interpolate(grid_angle, empirical_lambda_raw),
            smoothing_bin_count,
        )

    predicted_sample_velocity = phase_flow[bin_index]
    residual_rms = float(
        np.sqrt(np.mean(np.square(sample_velocity - predicted_sample_velocity)))
    )
    return {
        "grid_angle": grid_angle,
        "phase_flow_raw": phase_flow_raw,
        "phase_flow": phase_flow,
        "phase_flow_derivative": phase_flow_derivative,
        "empirical_lambda_raw": empirical_lambda_raw,
        "empirical_lambda": empirical_lambda,
        "bin_sample_count": bin_sample_count,
        "fixed_point_angle": fixed_angle,
        "fixed_point_slope": fixed_slope,
        "fixed_point_stability": fixed_stability,
        "sample_support_fraction": float(np.mean(observed)),
        "residual_rms": residual_rms,
        "lambda_speed_floor": lambda_speed_floor,
    }


def summarize_velocity_phase_flows(
    *,
    velocity_history,
    angular_bin_count: int,
    smoothing_bin_count: int,
    empirical_lambda_speed_floor: float,
) -> dict[str, np.ndarray]:
    """Summarize direct discrete fields for every stored dense-probe speed."""
    velocity = np.asarray(velocity_history["commanded_velocity"], dtype=float)
    probe_velocity_index = np.asarray(
        velocity_history.get("phase_flow_velocity_index", np.empty(0)),
        dtype=int,
    )
    probe_time = np.asarray(
        velocity_history.get("phase_flow_time", np.empty(0)),
        dtype=float,
    )
    probe_theta = np.asarray(
        velocity_history.get("phase_flow_theta_pva", np.empty((0, 0, 0))),
        dtype=float,
    )
    if (
        probe_velocity_index.size == 0
        or probe_time.size < 3
        or probe_theta.ndim != 3
        or probe_theta.shape[0] != probe_velocity_index.size
    ):
        raise ValueError(
            "direct phase-flow analysis requires stored dense probe trajectories"
        )
    fit_start = float(
        np.asarray(velocity_history.get("phase_flow_fit_start_time", 0.0))
    )
    fit_duration = float(
        np.asarray(
            velocity_history.get(
                "phase_flow_fit_duration",
                float(probe_time[-1]) - fit_start,
            )
        )
    )
    selected_velocity_index = np.asarray(probe_velocity_index, dtype=int)
    selected_velocity = velocity[selected_velocity_index]
    slot_count = selected_velocity.size
    grid_angle = np.linspace(
        -np.pi + np.pi / angular_bin_count,
        np.pi - np.pi / angular_bin_count,
        angular_bin_count,
    )
    phase_flow_raw = np.full((slot_count, angular_bin_count), np.nan)
    phase_flow = np.full_like(phase_flow_raw, np.nan)
    phase_flow_derivative = np.full_like(phase_flow_raw, np.nan)
    empirical_lambda_raw = np.full_like(phase_flow_raw, np.nan)
    empirical_lambda = np.full_like(phase_flow_raw, np.nan)
    bin_sample_count = np.zeros((slot_count, angular_bin_count), dtype=int)
    support_fraction = np.zeros(slot_count, dtype=float)
    residual_rms = np.full(slot_count, np.nan)
    lambda_speed_floor = np.full(slot_count, np.nan)
    fixed_velocity_slot: list[np.ndarray] = []
    fixed_angle: list[np.ndarray] = []
    fixed_slope: list[np.ndarray] = []
    fixed_stability: list[np.ndarray] = []
    basin_velocity_slot: list[np.ndarray] = []
    basin_stable_angle: list[np.ndarray] = []
    basin_left_boundary: list[np.ndarray] = []
    basin_right_boundary: list[np.ndarray] = []

    for slot, current_velocity in enumerate(selected_velocity):
        estimate = estimate_discrete_phase_flow(
            time=probe_time,
            theta=probe_theta[slot],
            commanded_velocity=float(current_velocity),
            fit_start_time=fit_start,
            fit_duration=fit_duration,
            angular_bin_count=angular_bin_count,
            smoothing_bin_count=smoothing_bin_count,
            empirical_lambda_speed_floor=empirical_lambda_speed_floor,
        )
        grid_angle = np.asarray(estimate["grid_angle"], dtype=float)
        phase_flow_raw[slot] = np.asarray(estimate["phase_flow_raw"])
        phase_flow[slot] = np.asarray(estimate["phase_flow"])
        phase_flow_derivative[slot] = np.asarray(
            estimate["phase_flow_derivative"]
        )
        empirical_lambda_raw[slot] = np.asarray(
            estimate["empirical_lambda_raw"]
        )
        empirical_lambda[slot] = np.asarray(estimate["empirical_lambda"])
        bin_sample_count[slot] = np.asarray(estimate["bin_sample_count"])
        support_fraction[slot] = float(estimate["sample_support_fraction"])
        residual_rms[slot] = float(estimate["residual_rms"])
        lambda_speed_floor[slot] = float(estimate["lambda_speed_floor"])
        current_fixed_angle = np.asarray(estimate["fixed_point_angle"])
        current_fixed_slope = np.asarray(estimate["fixed_point_slope"])
        current_fixed_stability = np.asarray(estimate["fixed_point_stability"])
        fixed_velocity_slot.append(
            np.full(current_fixed_angle.size, slot, dtype=int)
        )
        fixed_angle.append(current_fixed_angle)
        fixed_slope.append(current_fixed_slope)
        fixed_stability.append(current_fixed_stability)
        basin = actual_stable_basins(
            fixed_point_angle=current_fixed_angle,
            fixed_point_stability=current_fixed_stability,
        )
        basin_velocity_slot.append(
            np.full(basin["stable_angle"].size, slot, dtype=int)
        )
        basin_stable_angle.append(basin["stable_angle"])
        basin_left_boundary.append(basin["left_boundary"])
        basin_right_boundary.append(basin["right_boundary"])

    def concatenate(parts: list[np.ndarray], dtype=float) -> np.ndarray:
        if not parts:
            return np.empty(0, dtype=dtype)
        return np.concatenate(parts).astype(dtype, copy=False)

    return {
        "selected_velocity_index": selected_velocity_index,
        "commanded_velocity": selected_velocity,
        "grid_angle": grid_angle,
        "phase_flow_raw": phase_flow_raw,
        "phase_flow": phase_flow,
        "phase_flow_derivative": phase_flow_derivative,
        "empirical_lambda_raw": empirical_lambda_raw,
        "empirical_lambda": empirical_lambda,
        "bin_sample_count": bin_sample_count,
        "sample_support_fraction": support_fraction,
        "residual_rms": residual_rms,
        "lambda_speed_floor": lambda_speed_floor,
        "probe_time": probe_time,
        "probe_theta_pva": probe_theta,
        "phase_flow_fit_start_time": np.asarray(fit_start),
        "phase_flow_fit_duration": np.asarray(fit_duration),
        "angular_bin_count": np.asarray(angular_bin_count),
        "smoothing_bin_count": np.asarray(smoothing_bin_count),
        "fixed_point_velocity_slot": concatenate(fixed_velocity_slot, int),
        "fixed_point_angle": concatenate(fixed_angle),
        "fixed_point_slope": concatenate(fixed_slope),
        "fixed_point_stability": concatenate(fixed_stability, np.int8),
        "basin_velocity_slot": concatenate(basin_velocity_slot, int),
        "basin_stable_angle": concatenate(basin_stable_angle),
        "basin_left_boundary": concatenate(basin_left_boundary),
        "basin_right_boundary": concatenate(basin_right_boundary),
    }
