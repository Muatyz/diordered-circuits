"""Behavioral metrics for Vafidis toy-model tests."""

from __future__ import annotations

import numpy as np

from learning.common.angles import circular_difference, unwrap_heading_trace


def linear_fit_slope_intercept(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Return least-squares slope and intercept for a 1D line fit."""
    x_mean = float(np.mean(x_values))
    y_mean = float(np.mean(y_values))
    centered_x = x_values - x_mean
    denominator = float(np.sum(centered_x**2))
    if denominator <= 1e-12:
        return float("nan"), float("nan")
    slope = float(np.sum(centered_x * (y_values - y_mean)) / denominator)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def circular_error_trace(theta_decoded: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    return circular_difference(theta_decoded, theta_true)


def empirical_two_point_correlation(tuning_curves: np.ndarray) -> np.ndarray:
    """Return Clark et al.'s uncentered empirical two-point function.

    Rows are neurons and columns are heading samples.  Following Eq. 4, the
    result is ``tuning_curves.T @ tuning_curves / n_neurons``; neither a
    neuron-wise nor a heading-wise mean is subtracted.
    """
    tuning_curves = np.asarray(tuning_curves, dtype=float)
    if tuning_curves.ndim != 2 or tuning_curves.shape[0] == 0:
        raise ValueError("tuning_curves must contain at least one neuron")
    if not np.isfinite(tuning_curves).all():
        raise ValueError("tuning_curves must be finite")
    return np.einsum(
        "ia,ib->ab",
        tuning_curves,
        tuning_curves,
        optimize=False,
    ) / float(tuning_curves.shape[0])


def clark_overlap_order_parameter(
    target_rate: np.ndarray,
    population_activity: np.ndarray,
) -> np.ndarray:
    r"""Return Clark et al.'s uncentered overlap order parameter (Eq. 6).

    ``target_rate`` has shape ``(neuron, heading)`` and represents the target
    activity manifold :math:`\phi^*(\theta)`. ``population_activity`` may be a
    single state ``(neuron,)`` or a batch/time series ``(sample, neuron)``.
    The returned array always has shape ``(heading, sample)`` and implements
    ``target_rate.T @ population_activity.T / N`` without mean centering or
    cosine normalization.
    """
    target_rate = np.asarray(target_rate, dtype=float)
    population_activity = np.asarray(population_activity, dtype=float)
    if target_rate.ndim != 2 or target_rate.shape[0] == 0 or target_rate.shape[1] == 0:
        raise ValueError("target_rate must have shape (neuron, heading)")
    if population_activity.ndim == 1:
        population_activity = population_activity[None, :]
    if (
        population_activity.ndim != 2
        or population_activity.shape[1] != target_rate.shape[0]
    ):
        raise ValueError("population_activity must contain one value per neuron")
    if not np.isfinite(target_rate).all() or not np.isfinite(population_activity).all():
        raise ValueError("target_rate and population_activity must be finite")
    return (
        target_rate.T @ population_activity.T
        / float(target_rate.shape[0])
    )


def decode_heading_by_clark_overlap(
    *,
    theta_template: np.ndarray,
    target_rate: np.ndarray,
    population_activity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode heading at the maximum of Clark's overlap order parameter.

    Returns one decoded angle and one maximum-overlap value per activity
    sample. Ties follow NumPy's deterministic first-maximum convention.
    """
    theta_template = np.asarray(theta_template, dtype=float)
    if theta_template.ndim != 1 or theta_template.size == 0:
        raise ValueError("theta_template must be a non-empty one-dimensional array")
    if not np.isfinite(theta_template).all():
        raise ValueError("theta_template must be finite")
    target_rate = np.asarray(target_rate, dtype=float)
    if target_rate.ndim != 2 or target_rate.shape[1] != theta_template.size:
        raise ValueError("target_rate headings must match theta_template")
    overlap = clark_overlap_order_parameter(target_rate, population_activity)
    maximum_indices = np.argmax(overlap, axis=0)
    return theta_template[maximum_indices].astype(float), overlap[
        maximum_indices,
        np.arange(maximum_indices.size),
    ].astype(float)


def nearest_closed_manifold_distance(
    states: np.ndarray,
    manifold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure distance to a closed piecewise-linear target manifold.

    ``manifold[k]`` and ``manifold[(k + 1) % n_points]`` define one segment.
    Allowing the nearest point to lie between sampled target headings prevents
    motion tangent to the ring from being counted as a normal displacement.
    Distances are full-state L2 distances divided by ``sqrt(n_features)``, as
    in Clark et al. Figure 3F.  The second return value is a fractional segment
    coordinate ``k + alpha`` on the sampled ring.
    """
    states = np.asarray(states, dtype=float)
    manifold = np.asarray(manifold, dtype=float)
    if states.ndim == 1:
        states = states[None, :]
    if states.ndim != 2 or manifold.ndim != 2:
        raise ValueError("states and manifold must be one- or two-dimensional arrays")
    if states.shape[1] != manifold.shape[1]:
        raise ValueError("states and manifold must have the same feature count")
    if manifold.shape[0] < 2 or manifold.shape[1] == 0:
        raise ValueError("a closed manifold requires at least two non-empty points")

    nearest_coordinate = np.full(states.shape[0], np.nan, dtype=float)
    nearest_l2 = np.full(states.shape[0], np.nan, dtype=float)
    finite_states = np.all(np.isfinite(states), axis=1)
    next_manifold = np.roll(manifold, -1, axis=0)
    finite_segments = np.all(np.isfinite(manifold), axis=1) & np.all(
        np.isfinite(next_manifold),
        axis=1,
    )
    segment_indices = np.flatnonzero(finite_segments)
    if segment_indices.size == 0:
        return nearest_l2, nearest_coordinate

    starts = manifold[segment_indices]
    vectors = next_manifold[segment_indices] - starts
    start_norm = np.sum(starts * starts, axis=1)
    vector_norm = np.sum(vectors * vectors, axis=1)
    good_rows = np.flatnonzero(finite_states)
    state_chunk_size = 256
    segment_chunk_size = 256

    for state_start in range(0, good_rows.size, state_chunk_size):
        row_indices = good_rows[state_start : state_start + state_chunk_size]
        state_chunk = states[row_indices]
        state_norm = np.sum(state_chunk * state_chunk, axis=1, keepdims=True)
        best_squared = np.full(row_indices.size, np.inf, dtype=float)
        best_segment = np.full(row_indices.size, -1, dtype=int)
        best_alpha = np.zeros(row_indices.size, dtype=float)

        for segment_start in range(0, segment_indices.size, segment_chunk_size):
            segment_stop = min(
                segment_start + segment_chunk_size,
                segment_indices.size,
            )
            starts_part = starts[segment_start:segment_stop]
            vectors_part = vectors[segment_start:segment_stop]
            vector_norm_part = vector_norm[segment_start:segment_stop]
            state_dot_start = np.einsum(
                "ij,kj->ik",
                state_chunk,
                starts_part,
                optimize=False,
            )
            state_dot_vector = np.einsum(
                "ij,kj->ik",
                state_chunk,
                vectors_part,
                optimize=False,
            )
            start_dot_vector = np.sum(
                starts_part * vectors_part,
                axis=1,
            )[None, :]
            projection_numerator = state_dot_vector - start_dot_vector
            alpha = np.divide(
                projection_numerator,
                vector_norm_part[None, :],
                out=np.zeros_like(projection_numerator),
                where=vector_norm_part[None, :] > 0.0,
            )
            alpha = np.clip(alpha, 0.0, 1.0)
            squared = (
                state_norm
                + start_norm[None, segment_start:segment_stop]
                - 2.0 * state_dot_start
                - 2.0 * alpha * projection_numerator
                + alpha * alpha * vector_norm_part[None, :]
            )
            squared = np.maximum(squared, 0.0)
            local_segment = np.argmin(squared, axis=1)
            local_squared = squared[np.arange(row_indices.size), local_segment]
            improved = local_squared < best_squared
            if np.any(improved):
                best_squared[improved] = local_squared[improved]
                best_segment[improved] = segment_start + local_segment[improved]
                best_alpha[improved] = alpha[
                    np.arange(row_indices.size),
                    local_segment,
                ][improved]

        valid_best = best_segment >= 0
        nearest_l2[row_indices[valid_best]] = np.sqrt(best_squared[valid_best])
        nearest_coordinate[row_indices[valid_best]] = (
            segment_indices[best_segment[valid_best]] + best_alpha[valid_best]
        ) % manifold.shape[0]

    return nearest_l2 / np.sqrt(states.shape[1]), nearest_coordinate


def estimate_relaxation_e_folding_time(
    *,
    time: np.ndarray,
    distance: np.ndarray,
    tail_fraction: float = 0.2,
    peak_window: float = 0.25,
) -> dict[str, np.ndarray]:
    """Estimate decay time from the early peak to one e-fold above the floor.

    ``distance`` may have any leading dimensions and must use time on its last
    axis.  The late median is treated as a residual floor.  A short early peak
    search makes the estimate robust to one-filter-step latency after a state
    perturbation.  Non-recovered traces are right-censored at the observation
    duration and marked by ``event_observed=False``.
    """
    time = np.asarray(time, dtype=float)
    distance = np.asarray(distance, dtype=float)
    if time.ndim != 1 or time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be a strictly increasing 1D array")
    if distance.ndim < 1 or distance.shape[-1] != time.size:
        raise ValueError("distance must use time on its last axis")
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must lie between zero and one")
    if peak_window < 0.0:
        raise ValueError("peak_window must be non-negative")

    leading_shape = distance.shape[:-1]
    flat_distance = distance.reshape(-1, time.size)
    tail_count = max(1, int(np.ceil(tail_fraction * time.size)))
    floor = np.nanmedian(flat_distance[:, -tail_count:], axis=1)
    peak_candidates = np.flatnonzero(time <= time[0] + peak_window + 1e-12)
    if peak_candidates.size == 0:
        peak_candidates = np.asarray([0], dtype=int)

    e_folding_time = np.full(flat_distance.shape[0], np.nan, dtype=float)
    event_observed = np.zeros(flat_distance.shape[0], dtype=bool)
    peak_time = np.full(flat_distance.shape[0], np.nan, dtype=float)
    peak_value = np.full(flat_distance.shape[0], np.nan, dtype=float)
    for trace_index, trace in enumerate(flat_distance):
        early = trace[peak_candidates]
        finite_early = np.isfinite(early)
        if not np.any(finite_early) or not np.isfinite(floor[trace_index]):
            continue
        finite_candidate_indices = peak_candidates[finite_early]
        local_peak_index = int(
            finite_candidate_indices[
                np.argmax(trace[finite_candidate_indices])
            ]
        )
        peak_time[trace_index] = time[local_peak_index]
        peak_value[trace_index] = trace[local_peak_index]
        amplitude = peak_value[trace_index] - floor[trace_index]
        numerical_scale = max(abs(peak_value[trace_index]), 1.0)
        if not np.isfinite(amplitude) or amplitude <= 1e-12 * numerical_scale:
            continue
        target = floor[trace_index] + amplitude / np.e
        after_peak = np.arange(local_peak_index + 1, time.size)
        crossings = after_peak[
            np.isfinite(trace[after_peak]) & (trace[after_peak] <= target)
        ]
        if crossings.size == 0:
            e_folding_time[trace_index] = time[-1] - time[local_peak_index]
            continue
        crossing_index = int(crossings[0])
        previous_index = crossing_index - 1
        previous_value = trace[previous_index]
        crossing_value = trace[crossing_index]
        if (
            np.isfinite(previous_value)
            and previous_value > target
            and crossing_value < previous_value
        ):
            interpolation_fraction = (previous_value - target) / (
                previous_value - crossing_value
            )
            crossing_time = time[previous_index] + interpolation_fraction * (
                time[crossing_index] - time[previous_index]
            )
        else:
            crossing_time = time[crossing_index]
        e_folding_time[trace_index] = crossing_time - time[local_peak_index]
        event_observed[trace_index] = True

    return {
        "e_folding_time": e_folding_time.reshape(leading_shape),
        "event_observed": event_observed.reshape(leading_shape),
        "floor": floor.reshape(leading_shape),
        "peak_time": peak_time.reshape(leading_shape),
        "peak_value": peak_value.reshape(leading_shape),
    }


def angular_first_passage_time(
    *,
    time: np.ndarray,
    angular_displacement: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first absolute angular-threshold passage and event indicators.

    Traces that never cross are right-censored at the final observation time.
    Linear interpolation between recorded samples reduces sample-grid bias.
    """
    time = np.asarray(time, dtype=float)
    angular_displacement = np.asarray(angular_displacement, dtype=float)
    if time.ndim != 1 or time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be a strictly increasing 1D array")
    if (
        angular_displacement.ndim < 1
        or angular_displacement.shape[-1] != time.size
    ):
        raise ValueError("angular_displacement must use time on its last axis")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    leading_shape = angular_displacement.shape[:-1]
    flat_displacement = np.abs(angular_displacement.reshape(-1, time.size))
    passage_time = np.full(
        flat_displacement.shape[0],
        time[-1] - time[0],
        dtype=float,
    )
    event_observed = np.zeros(flat_displacement.shape[0], dtype=bool)
    for trace_index, trace in enumerate(flat_displacement):
        crossings = np.flatnonzero(np.isfinite(trace) & (trace >= threshold))
        if crossings.size == 0:
            if not np.any(np.isfinite(trace)):
                passage_time[trace_index] = np.nan
            continue
        crossing_index = int(crossings[0])
        if crossing_index == 0:
            passage_time[trace_index] = 0.0
        else:
            previous_value = trace[crossing_index - 1]
            crossing_value = trace[crossing_index]
            if (
                np.isfinite(previous_value)
                and crossing_value > previous_value
                and previous_value < threshold
            ):
                interpolation_fraction = (threshold - previous_value) / (
                    crossing_value - previous_value
                )
                passage_time[trace_index] = (
                    time[crossing_index - 1]
                    + interpolation_fraction
                    * (time[crossing_index] - time[crossing_index - 1])
                    - time[0]
                )
            else:
                passage_time[trace_index] = time[crossing_index] - time[0]
        event_observed[trace_index] = True
    return passage_time.reshape(leading_shape), event_observed.reshape(leading_shape)


def relative_circulant_error(matrix: np.ndarray) -> float:
    """Measure relative Frobenius distance from the circulant projection."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite")
    matrix_size = matrix.shape[0]
    row_indices, column_indices = np.indices(matrix.shape)
    periodic_offset = (row_indices - column_indices) % matrix_size
    diagonal_means = np.asarray(
        [np.mean(matrix[periodic_offset == offset]) for offset in range(matrix_size)],
        dtype=float,
    )
    circulant_projection = diagonal_means[periodic_offset]
    denominator = max(float(np.linalg.norm(matrix)), 1e-12)
    return float(np.linalg.norm(matrix - circulant_projection) / denominator)


def kuiper_uniformity_test_asymptotic(
    angles_rad: np.ndarray,
    *,
    max_terms: int = 100,
) -> tuple[float, float]:
    """Apply a one-sample Kuiper test to circular angles.

    Angles are wrapped onto ``[0, 2*pi)``. The returned statistic is
    ``V = D_plus + D_minus`` and the p-value uses the finite-sample-corrected
    asymptotic Kuiper series used by the Clark Figure 4 reproduction.
    """
    angles_rad = np.asarray(angles_rad, dtype=float)
    angles_rad = angles_rad[np.isfinite(angles_rad)]
    if angles_rad.size < 2:
        raise ValueError("Kuiper test requires at least two finite angles")
    if max_terms < 1:
        raise ValueError("max_terms must be positive")

    uniform_values = np.sort(np.mod(angles_rad, 2.0 * np.pi) / (2.0 * np.pi))
    sample_size = uniform_values.size
    ranks = np.arange(1, sample_size + 1, dtype=float)
    d_plus = float(np.max(ranks / sample_size - uniform_values))
    d_minus = float(np.max(uniform_values - (ranks - 1.0) / sample_size))
    statistic = d_plus + d_minus

    root_sample_size = np.sqrt(float(sample_size))
    scaled_statistic = (
        root_sample_size + 0.155 + 0.24 / root_sample_size
    ) * statistic
    terms = np.arange(1, int(max_terms) + 1, dtype=float)
    squared_terms = terms * terms
    p_value = 2.0 * np.sum(
        (4.0 * squared_terms * scaled_statistic**2 - 1.0)
        * np.exp(-2.0 * squared_terms * scaled_statistic**2)
    )
    return statistic, float(np.clip(p_value, 0.0, 1.0))


def benjamini_hochberg_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or p_values.size == 0 or not np.isfinite(p_values).all():
        raise ValueError("p_values must be a non-empty finite one-dimensional array")
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")

    order = np.argsort(p_values, kind="stable")
    ordered_p_values = p_values[order]
    ranks = np.arange(1, p_values.size + 1, dtype=float)
    adjusted_ordered = ordered_p_values * float(p_values.size) / ranks
    adjusted_ordered = np.minimum.accumulate(adjusted_ordered[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = np.clip(adjusted_ordered, 0.0, 1.0)
    return adjusted


def rms_circular_error(theta_decoded: np.ndarray, theta_true: np.ndarray) -> float:
    error_trace = circular_error_trace(theta_decoded, theta_true)
    finite_error = error_trace[np.isfinite(error_trace)]
    if finite_error.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite_error**2)))


def final_abs_circular_error(theta_decoded: np.ndarray, theta_reference: float) -> float:
    finite_decoded = theta_decoded[np.isfinite(theta_decoded)]
    if finite_decoded.size == 0:
        return float("nan")
    return float(abs(circular_difference(finite_decoded[-1], theta_reference)))


def estimate_decoded_velocity(
    *,
    time: np.ndarray,
    theta_decoded: np.ndarray,
    start_fraction: float = 0.25,
) -> float:
    finite_mask = np.isfinite(time) & np.isfinite(theta_decoded)
    if np.count_nonzero(finite_mask) < 3:
        return float("nan")
    finite_time = time[finite_mask]
    finite_theta_decoded = unwrap_heading_trace(theta_decoded[finite_mask])
    start_index = int(np.floor(start_fraction * finite_time.size))
    selected_time = finite_time[start_index:]
    selected_theta_decoded = finite_theta_decoded[start_index:]
    if selected_time.size < 3:
        return float("nan")
    slope, _intercept = linear_fit_slope_intercept(selected_time, selected_theta_decoded)
    return float(slope)


def summarize_velocity_gain(
    *,
    commanded_velocity: np.ndarray,
    decoded_velocity: np.ndarray,
) -> dict[str, float]:
    finite_mask = np.isfinite(commanded_velocity) & np.isfinite(decoded_velocity)
    if np.count_nonzero(finite_mask) < 2:
        return {
            "gain": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
            "linear_fit_rmse": float("nan"),
        }
    finite_commanded = np.asarray(commanded_velocity[finite_mask], dtype=float)
    finite_decoded = np.asarray(decoded_velocity[finite_mask], dtype=float)
    slope, intercept = linear_fit_slope_intercept(
        finite_commanded,
        finite_decoded,
    )
    fitted_decoded = slope * finite_commanded + intercept
    residual = finite_decoded - fitted_decoded
    residual_sum_squares = float(np.sum(residual**2))
    total_sum_squares = float(np.sum((finite_decoded - np.mean(finite_decoded)) ** 2))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 1e-12
        else float("nan")
    )
    return {
        "gain": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "linear_fit_rmse": float(np.sqrt(np.mean(residual**2))),
    }


def estimate_velocity_tracking_operating_range(
    *,
    commanded_velocity: np.ndarray,
    decoded_velocity: np.ndarray,
    relative_error_tolerance: float = 0.25,
    absolute_error_tolerance: float = 0.25,
) -> float:
    """Return the largest contiguous-from-zero velocity tracked within tolerance.

    For each sampled absolute command magnitude, every available sign must pass
    ``abs(decoded-commanded) <= absolute + relative*abs(commanded)``.  Scanning
    stops at the first failed magnitude, so isolated high-velocity successes do
    not hide an intervening lock-up regime.
    """
    commanded_velocity = np.asarray(commanded_velocity, dtype=float)
    decoded_velocity = np.asarray(decoded_velocity, dtype=float)
    if commanded_velocity.shape != decoded_velocity.shape or commanded_velocity.ndim != 1:
        raise ValueError("commanded_velocity and decoded_velocity must be matching 1D arrays")
    if relative_error_tolerance < 0.0 or absolute_error_tolerance < 0.0:
        raise ValueError("velocity tracking tolerances must be non-negative")
    finite_mask = np.isfinite(commanded_velocity) & np.isfinite(decoded_velocity)
    if not np.any(finite_mask):
        return float("nan")
    finite_commanded = commanded_velocity[finite_mask]
    finite_decoded = decoded_velocity[finite_mask]
    magnitudes = np.unique(np.abs(finite_commanded))
    operating_range = 0.0
    for magnitude in np.sort(magnitudes):
        magnitude_mask = np.isclose(np.abs(finite_commanded), magnitude)
        error = np.abs(finite_decoded[magnitude_mask] - finite_commanded[magnitude_mask])
        tolerance = absolute_error_tolerance + relative_error_tolerance * magnitude
        if not np.all(error <= tolerance):
            break
        operating_range = float(magnitude)
    return operating_range


def summarize_velocity_tracking(
    *,
    commanded_velocity: np.ndarray,
    decoded_velocity: np.ndarray,
    ideal_gain: float = 1.0,
) -> dict[str, float]:
    """Summarize decoded velocity against the ideal PI relation.

    Unlike the fitted gain, these errors keep the absolute mismatch scale and
    therefore catch offset, saturation, and nonlinear failures across the tested
    velocity range.
    """
    finite_mask = np.isfinite(commanded_velocity) & np.isfinite(decoded_velocity)
    if np.count_nonzero(finite_mask) == 0:
        return {
            "velocity_tracking_rmse": float("nan"),
            "velocity_tracking_mae": float("nan"),
            "velocity_tracking_max_abs_error": float("nan"),
            "velocity_tracking_bias": float("nan"),
            "velocity_tracking_rmse_fraction_of_max_command": float("nan"),
            "velocity_direction_match_fraction": float("nan"),
        }
    finite_commanded = np.asarray(commanded_velocity[finite_mask], dtype=float)
    finite_decoded = np.asarray(decoded_velocity[finite_mask], dtype=float)
    tracking_error = finite_decoded - ideal_gain * finite_commanded
    max_abs_command = float(np.max(np.abs(finite_commanded)))
    nonzero_mask = np.abs(finite_commanded) > 1e-12
    if np.any(nonzero_mask):
        direction_match_fraction = float(
            np.mean(
                np.sign(finite_decoded[nonzero_mask])
                == np.sign(finite_commanded[nonzero_mask])
            )
        )
    else:
        direction_match_fraction = float("nan")
    rmse = float(np.sqrt(np.mean(tracking_error**2)))
    return {
        "velocity_tracking_rmse": rmse,
        "velocity_tracking_mae": float(np.mean(np.abs(tracking_error))),
        "velocity_tracking_max_abs_error": float(np.max(np.abs(tracking_error))),
        "velocity_tracking_bias": float(np.mean(tracking_error)),
        "velocity_tracking_rmse_fraction_of_max_command": (
            rmse / max_abs_command if max_abs_command > 1e-12 else float("nan")
        ),
        "velocity_direction_match_fraction": direction_match_fraction,
    }


def empirical_tuning_preferred_directions(
    *,
    theta_true: np.ndarray,
    r_hd_by_heading: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each neuron's circular COM preference and resultant strength."""
    theta_true = np.asarray(theta_true, dtype=float)
    rates = np.asarray(r_hd_by_heading, dtype=float)
    if theta_true.ndim != 1 or rates.ndim != 2 or rates.shape[0] != theta_true.size:
        raise ValueError("r_hd_by_heading must have shape (heading, neuron)")
    finite_rates = np.where(np.isfinite(rates), np.maximum(rates, 0.0), 0.0)
    complex_moment = np.sum(
        finite_rates * np.exp(1j * theta_true[:, None]),
        axis=0,
    )
    rate_sum = np.sum(finite_rates, axis=0)
    preferred_direction = np.angle(complex_moment)
    strength = np.divide(
        np.abs(complex_moment),
        rate_sum,
        out=np.zeros_like(rate_sum),
        where=rate_sum > 1e-12,
    )
    preferred_direction[rate_sum <= 1e-12] = np.nan
    return preferred_direction.astype(float), strength.astype(float)


def summarize_com_aligned_tuning_curves(
    *,
    theta_true: np.ndarray,
    r_hd_by_heading: np.ndarray,
) -> dict[str, np.ndarray]:
    """Normalize and circular-COM align every HD tuning curve.

    The primary output divides each neuron by its own maximum firing rate; it
    does not use a population-wide maximum.  Clark et al.'s original unit-mean
    normalization is retained as a separate reference output.  Both versions
    are shifted by the same integer number of heading bins so each circular
    center of mass is closest to zero.  Integer-bin shifts preserve the
    measured samples and their full periodic range.  Standard deviations use
    the population definition across all ``N`` HD neurons (``ddof=0``).

    A neuron whose angular mean or peak is at most ``1e-12`` is treated as
    effectively silent.  Its all-zero curve remains in the primary all-``N``
    peak-normalized population as a zero row, while its mathematically
    undefined unit-mean reference row is stored as NaN.  Validity masks and
    counts make this failure mode explicit instead of aborting figure creation.
    """
    theta_true = np.asarray(theta_true, dtype=float)
    rates = np.asarray(r_hd_by_heading, dtype=float)
    if theta_true.ndim != 1 or rates.ndim != 2 or rates.shape[0] != theta_true.size:
        raise ValueError("r_hd_by_heading must have shape (heading, neuron)")
    if theta_true.size < 2 or rates.shape[1] == 0:
        raise ValueError("COM-aligned tuning analysis requires headings and HD neurons")
    if not np.isfinite(theta_true).all() or not np.isfinite(rates).all():
        raise ValueError("theta_true and r_hd_by_heading must be finite")
    if np.any(rates < 0.0):
        raise ValueError("HD firing-rate tuning curves must be non-negative")

    heading_order = np.argsort(theta_true, kind="stable")
    theta_ordered = theta_true[heading_order]
    circular_gaps = np.diff(
        np.concatenate([theta_ordered, [theta_ordered[0] + 2.0 * np.pi]])
    )
    expected_gap = 2.0 * np.pi / float(theta_true.size)
    if not np.allclose(circular_gaps, expected_gap, rtol=1e-6, atol=1e-9):
        raise ValueError("theta_true must be a uniform grid spanning one circular period")

    tuning_by_neuron = rates[heading_order].T
    angular_mean_rate = np.mean(tuning_by_neuron, axis=1)
    peak_rate = np.max(tuning_by_neuron, axis=1)
    activity_floor = 1e-12
    tuning_valid_mask = (
        (angular_mean_rate > activity_floor)
        & (peak_rate > activity_floor)
    )
    peak_normalized_tuning = np.zeros_like(tuning_by_neuron)
    np.divide(
        tuning_by_neuron,
        peak_rate[:, None],
        out=peak_normalized_tuning,
        where=tuning_valid_mask[:, None],
    )
    unit_mean_tuning = np.full_like(tuning_by_neuron, np.nan)
    np.divide(
        tuning_by_neuron,
        angular_mean_rate[:, None],
        out=unit_mean_tuning,
        where=tuning_valid_mask[:, None],
    )
    preferred_direction, _strength = empirical_tuning_preferred_directions(
        theta_true=theta_ordered,
        r_hd_by_heading=tuning_by_neuron.T,
    )
    tuning_valid_mask &= np.isfinite(preferred_direction)
    preferred_direction[~tuning_valid_mask] = np.nan
    preferred_heading_bin = np.full(rates.shape[1], -1, dtype=int)
    if np.any(tuning_valid_mask):
        circular_bin_distance = np.abs(
            circular_difference(
                theta_ordered[:, None],
                preferred_direction[tuning_valid_mask][None, :],
            )
        )
        preferred_heading_bin[tuning_valid_mask] = np.argmin(
            circular_bin_distance,
            axis=0,
        )
    zero_heading_bin = theta_true.size // 2
    alignment_shift_bins = np.zeros(rates.shape[1], dtype=int)
    alignment_shift_bins[tuning_valid_mask] = (
        zero_heading_bin - preferred_heading_bin[tuning_valid_mask]
    )
    peak_normalized_aligned_tuning = np.vstack(
        [
            np.roll(neuron_tuning, int(shift_bins))
            for neuron_tuning, shift_bins in zip(
                peak_normalized_tuning,
                alignment_shift_bins,
            )
        ]
    )
    unit_mean_aligned_tuning = np.vstack(
        [
            np.roll(neuron_tuning, int(shift_bins))
            for neuron_tuning, shift_bins in zip(
                unit_mean_tuning,
                alignment_shift_bins,
            )
        ]
    )
    theta_aligned = (
        np.arange(theta_true.size, dtype=float) - float(zero_heading_bin)
    ) * expected_gap
    if np.any(tuning_valid_mask):
        unit_mean_population_mean = np.nanmean(unit_mean_aligned_tuning, axis=0)
        unit_mean_population_std = np.nanstd(
            unit_mean_aligned_tuning,
            axis=0,
            ddof=0,
        )
    else:
        unit_mean_population_mean = np.full(theta_true.size, np.nan)
        unit_mean_population_std = np.full(theta_true.size, np.nan)
    return {
        "theta_aligned": theta_aligned,
        "r_hd_peak_normalized_com_aligned": peak_normalized_aligned_tuning,
        "r_hd_peak_normalized_com_aligned_mean": np.mean(
            peak_normalized_aligned_tuning,
            axis=0,
        ),
        "r_hd_peak_normalized_com_aligned_std": np.std(
            peak_normalized_aligned_tuning,
            axis=0,
            ddof=0,
        ),
        "r_hd_unit_mean_com_aligned": unit_mean_aligned_tuning,
        "r_hd_unit_mean_com_aligned_mean": unit_mean_population_mean,
        "r_hd_unit_mean_com_aligned_std": unit_mean_population_std,
        "r_hd_peak_rate": peak_rate,
        "r_hd_angular_mean": angular_mean_rate,
        "r_hd_tuning_valid_mask": tuning_valid_mask,
        "r_hd_tuning_silent_mask": ~tuning_valid_mask,
        "r_hd_tuning_valid_neuron_count": np.asarray(
            np.count_nonzero(tuning_valid_mask),
            dtype=int,
        ),
        "r_hd_tuning_silent_neuron_count": np.asarray(
            np.count_nonzero(~tuning_valid_mask),
            dtype=int,
        ),
        "r_hd_tuning_silent_neuron_fraction": np.asarray(
            np.mean(~tuning_valid_mask),
            dtype=float,
        ),
        "r_hd_tuning_activity_floor": np.asarray(activity_floor, dtype=float),
        "empirical_preferred_direction": preferred_direction,
        "preferred_heading_bin": preferred_heading_bin.astype(int),
        "com_alignment_shift_bins": alignment_shift_bins.astype(int),
        "plot_normalization": np.asarray("per_neuron_peak"),
        "simulated_mouse_count": np.asarray(1, dtype=int),
        "n_hd_neurons": np.asarray(rates.shape[1], dtype=int),
    }


def summarize_pi_error_ensemble(
    *,
    time: np.ndarray,
    pi_error: np.ndarray,
    fit_start_time: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Summarize PVA-only PI error across independent trajectories."""
    time = np.asarray(time, dtype=float)
    errors = np.asarray(pi_error, dtype=float)
    if time.ndim != 1 or errors.ndim != 2 or errors.shape[1] != time.size:
        raise ValueError("pi_error must have shape (trial, time)")
    if errors.shape[0] == 0 or time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("PI ensemble requires trials and strictly increasing time")
    elapsed_time = time - float(time[0])
    mean_error = np.nanmean(errors, axis=0)
    error_std = np.nanstd(errors, axis=0, ddof=0)
    finite_count = np.sum(np.isfinite(errors), axis=0)
    error_sem = np.divide(
        error_std,
        np.sqrt(finite_count),
        out=np.full_like(error_std, np.nan),
        where=finite_count > 0,
    )
    fit_mask = (
        np.isfinite(elapsed_time)
        & np.isfinite(mean_error)
        & (elapsed_time >= float(fit_start_time))
    )
    if np.count_nonzero(fit_mask) >= 2:
        drift_velocity, drift_intercept = linear_fit_slope_intercept(
            elapsed_time[fit_mask], mean_error[fit_mask]
        )
    else:
        drift_velocity, drift_intercept = float("nan"), float("nan")
    return {
        "time": elapsed_time,
        "pi_error_mean": mean_error,
        "pi_error_std": error_std,
        "pi_error_sem": error_sem,
        "systematic_drift_velocity": float(drift_velocity),
        "drift_intercept": float(drift_intercept),
        "rms_mean_pi_error": float(np.sqrt(np.nanmean(mean_error**2))),
        "final_mean_pi_error": float(mean_error[-1]),
        "final_pi_error_std": float(error_std[-1]),
        "n_trials": float(errors.shape[0]),
    }


def estimate_effective_diffusion_coefficient(
    *,
    time: np.ndarray,
    theta_decoded: np.ndarray,
    theta_reference: float,
    start_fraction: float = 0.0,
) -> float:
    """Estimate an effective angular diffusion coefficient from one trace.

    This diagnostic fits mean-square circular displacement as
    ``E[Delta theta^2] ~= 2 D t``.  With one trajectory it should be interpreted
    as a protocol-level effective coefficient, not a full ensemble estimate.
    """
    finite_mask = np.isfinite(time) & np.isfinite(theta_decoded)
    if np.count_nonzero(finite_mask) < 3:
        return float("nan")
    finite_time = np.asarray(time[finite_mask], dtype=float)
    finite_time = finite_time - float(finite_time[0])
    displacement = circular_difference(theta_decoded[finite_mask], theta_reference)
    squared_displacement = displacement**2
    start_index = int(np.floor(start_fraction * finite_time.size))
    selected_time = finite_time[start_index:]
    selected_msd = squared_displacement[start_index:]
    if selected_time.size < 3:
        return float("nan")
    slope, _intercept = linear_fit_slope_intercept(selected_time, selected_msd)
    return float(max(0.0, slope / 2.0))


def summarize_ensemble_diffusion_coefficient(
    *,
    angular_displacement: np.ndarray,
    duration: float,
) -> dict[str, float]:
    """Estimate D from an ensemble displacement distribution.

    This follows the convention used by Vafidis et al.: ``D = Var(Delta theta) / t``.
    The returned value is in ``rad^2/s`` when displacements are in radians.
    """
    finite_displacement = np.asarray(angular_displacement, dtype=float)
    finite_displacement = finite_displacement[np.isfinite(finite_displacement)]
    if finite_displacement.size == 0 or duration <= 0.0:
        return {
            "diffusion_coefficient": float("nan"),
            "displacement_mean": float("nan"),
            "displacement_std": float("nan"),
            "n_trials": 0.0,
        }
    displacement_variance = float(np.var(finite_displacement, ddof=0))
    return {
        "diffusion_coefficient": displacement_variance / float(duration),
        "displacement_mean": float(np.mean(finite_displacement)),
        "displacement_std": float(np.sqrt(displacement_variance)),
        "n_trials": float(finite_displacement.size),
    }


def fit_anomalous_diffusion_power_law(
    *,
    time: np.ndarray,
    displacement_variance: np.ndarray,
    fit_start_time: float = 0.0,
    fit_end_time: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Fit ``Var[Delta theta(t)] = D_alpha * t**alpha`` in log-log space.

    The centered ensemble variance is used rather than the raw second moment,
    because systematic drift is reported separately. Zero time and zero
    variance cannot enter a logarithmic fit and are excluded explicitly.
    """
    time = np.asarray(time, dtype=float)
    variance = np.asarray(displacement_variance, dtype=float)
    if time.ndim != 1 or variance.ndim != 1 or time.shape != variance.shape:
        raise ValueError("time and displacement_variance must be matching 1D arrays")
    if fit_start_time < 0.0:
        raise ValueError("fit_start_time must be non-negative")
    if fit_end_time is not None and fit_end_time <= fit_start_time:
        raise ValueError("fit_end_time must be greater than fit_start_time")

    fit_mask = (
        np.isfinite(time)
        & np.isfinite(variance)
        & (time > 0.0)
        & (variance > 0.0)
        & (time >= float(fit_start_time))
    )
    if fit_end_time is not None:
        fit_mask &= time <= float(fit_end_time)

    fit_trace = np.full(time.shape, np.nan, dtype=float)
    if np.count_nonzero(fit_mask) < 3:
        return {
            "anomalous_diffusion_exponent": float("nan"),
            "generalized_diffusion_coefficient": float("nan"),
            "anomalous_diffusion_log_r_squared": float("nan"),
            "anomalous_diffusion_fit_n_points": float(np.count_nonzero(fit_mask)),
            "anomalous_diffusion_fit_start_time": float("nan"),
            "anomalous_diffusion_fit_end_time": float("nan"),
            "anomalous_diffusion_fit_trace": fit_trace,
        }

    log_time = np.log(time[fit_mask])
    log_variance = np.log(variance[fit_mask])
    exponent, log_coefficient = linear_fit_slope_intercept(log_time, log_variance)
    generalized_coefficient = float(np.exp(log_coefficient))
    predicted_log_variance = exponent * log_time + log_coefficient
    residual_sum_squares = float(np.sum((log_variance - predicted_log_variance) ** 2))
    total_sum_squares = float(np.sum((log_variance - np.mean(log_variance)) ** 2))
    log_r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 1e-12
        else float("nan")
    )
    positive_time = np.isfinite(time) & (time > 0.0)
    fit_trace[positive_time] = generalized_coefficient * time[positive_time] ** exponent
    return {
        "anomalous_diffusion_exponent": float(exponent),
        "generalized_diffusion_coefficient": generalized_coefficient,
        "anomalous_diffusion_log_r_squared": float(log_r_squared),
        "anomalous_diffusion_fit_n_points": float(np.count_nonzero(fit_mask)),
        "anomalous_diffusion_fit_start_time": float(np.min(time[fit_mask])),
        "anomalous_diffusion_fit_end_time": float(np.max(time[fit_mask])),
        "anomalous_diffusion_fit_trace": fit_trace,
    }


def summarize_ensemble_diffusion_trajectories(
    *,
    time: np.ndarray,
    angular_displacement: np.ndarray,
    fit_start_time: float = 0.0,
    fit_end_time: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Separate systematic bump drift from Vafidis-style diffusion.

    ``angular_displacement`` has shape ``(trial, time)`` and must already be
    unwrapped relative to each trial's first darkness sample. The paper defines
    ``D(t) = Var[Delta theta(t)] / t`` (without the Brownian ``1/2`` factor).
    """
    time = np.asarray(time, dtype=float)
    displacement = np.asarray(angular_displacement, dtype=float)
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if displacement.ndim != 2 or displacement.shape[1] != time.size:
        raise ValueError("angular_displacement must have shape (trial, time)")
    if time.size < 2 or displacement.shape[0] == 0:
        raise ValueError("at least one trial and two time samples are required")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing")

    elapsed_time = time - float(time[0])
    displacement_mean = np.nanmean(displacement, axis=0)
    displacement_variance = np.nanvar(displacement, axis=0, ddof=0)
    diffusion_coefficient_trace = np.full(time.shape, np.nan, dtype=float)
    positive_time = elapsed_time > 0.0
    diffusion_coefficient_trace[positive_time] = (
        displacement_variance[positive_time] / elapsed_time[positive_time]
    )
    duration = float(elapsed_time[-1])
    final_displacement = displacement[:, -1]
    endpoint_summary = summarize_ensemble_diffusion_coefficient(
        angular_displacement=final_displacement,
        duration=duration,
    )
    anomalous_summary = fit_anomalous_diffusion_power_law(
        time=elapsed_time,
        displacement_variance=displacement_variance,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
    )
    return {
        **endpoint_summary,
        **anomalous_summary,
        "systematic_drift_velocity": float(displacement_mean[-1] / duration),
        "time": elapsed_time,
        "displacement_mean_trace": displacement_mean,
        "displacement_variance_trace": displacement_variance,
        "diffusion_coefficient_trace": diffusion_coefficient_trace,
    }
