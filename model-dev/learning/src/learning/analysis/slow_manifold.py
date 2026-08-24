"""Ságodi-style slow-ring identification and local stability diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import ArpackNoConvergence, eigs

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class _NoOpProgress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback

        def update(self, count):
            del count

    def tqdm(iterable=None, **kwargs):
        del kwargs
        return _NoOpProgress() if iterable is None else iterable

from learning.common.angles import sum_activity_by_theta, wrap_angle
from learning.dynamics.autonomous import FrozenAutonomousDynamics


def _pca_variance_summary(sample_by_feature: np.ndarray) -> dict[str, np.ndarray | float]:
    """Return the complete linear-PCA variance spectrum for centered samples."""
    sample_by_feature = np.asarray(sample_by_feature, dtype=float)
    if sample_by_feature.ndim != 2 or sample_by_feature.shape[0] < 2:
        raise ValueError("PCA samples must be a 2D array with at least two rows")
    if not np.all(np.isfinite(sample_by_feature)):
        raise ValueError("PCA samples must be finite")
    centered = sample_by_feature - np.mean(sample_by_feature, axis=0)
    _left, singular_value, right = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular_value)
    total_variance = float(np.sum(variance))
    explained_fraction = (
        variance / total_variance if total_variance > 0.0 else np.zeros_like(variance)
    )
    cumulative_fraction = np.cumsum(explained_fraction)
    participation_ratio = (
        float(np.square(total_variance) / np.sum(np.square(variance)))
        if np.sum(np.square(variance)) > 0.0
        else 0.0
    )
    return {
        "center": np.mean(sample_by_feature, axis=0),
        "components": right,
        "singular_value": singular_value,
        "explained_fraction": explained_fraction,
        "cumulative_fraction": cumulative_fraction,
        "participation_ratio": participation_ratio,
    }


def _canonical_state_block_scale(
    *,
    dynamics: FrozenAutonomousDynamics,
    sample_by_state: np.ndarray,
) -> np.ndarray:
    """Return one empirical scale per canonical Markov-state block.

    The five canonical blocks mix rates, currents, and voltages.  A raw-state
    PCA would therefore depend on their numerical units.  Scaling every feature
    in a block by the same value preserves within-block population geometry
    while preventing one physical variable type from dominating solely because
    of its units.
    """
    sample_by_state = np.asarray(sample_by_state, dtype=float)
    if sample_by_state.ndim != 2 or sample_by_state.shape[1] != dynamics.state_dimension:
        raise ValueError(
            "canonical state samples must have shape "
            "(sample, autonomous_state_dimension)"
        )
    if not np.all(np.isfinite(sample_by_state)):
        raise ValueError("canonical state samples must be finite")

    state_scale = np.ones(dynamics.state_dimension, dtype=float)
    for current_slice in dynamics.component_slices.values():
        block = sample_by_state[:, current_slice]
        block_scale = float(np.std(block))
        if not np.isfinite(block_scale) or block_scale < 1e-6:
            block_scale = max(float(np.sqrt(np.mean(np.square(block)))), 1e-3)
        state_scale[current_slice] = block_scale
    return state_scale


def _minimum_rank_for_fraction(
    cumulative_fraction: np.ndarray,
    target_fraction: float,
) -> float:
    """Return the one-based PCA rank reaching a target, or NaN if unavailable."""
    cumulative_fraction = np.asarray(cumulative_fraction, dtype=float)
    reached = np.flatnonzero(cumulative_fraction >= target_fraction)
    return float(reached[0] + 1) if reached.size else float("nan")


def _sum_hd_rates_by_preferred_direction(
    *,
    theta_hd_pref: np.ndarray,
    hd_rate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact angular-rate statistic entering the PVA complex sum."""
    theta_hd_pref = np.asarray(theta_hd_pref, dtype=float)
    hd_rate = np.asarray(hd_rate, dtype=float)
    if hd_rate.ndim != 2 or hd_rate.shape[1] != theta_hd_pref.size:
        raise ValueError("hd_rate must have shape (sample, theta_hd_pref.size)")
    return sum_activity_by_theta(theta_hd_pref, hd_rate)


def candidate_angular_bin_counts(
    *,
    candidate_theta: np.ndarray,
    angular_bin_count: int,
) -> np.ndarray:
    """Count candidate support on a fixed circular grid."""
    candidate_theta = np.asarray(wrap_angle(candidate_theta), dtype=float)
    if candidate_theta.ndim != 1:
        raise ValueError("candidate_theta must be one-dimensional")
    if angular_bin_count < 8:
        raise ValueError("angular_bin_count must be at least eight")
    angular_edges = np.linspace(-np.pi, np.pi, angular_bin_count + 1)
    bin_index = np.searchsorted(
        angular_edges,
        candidate_theta,
        side="right",
    ) - 1
    bin_index = np.clip(bin_index, 0, angular_bin_count - 1)
    return np.bincount(bin_index, minlength=angular_bin_count)


def summarize_candidate_angle_clusters(
    *,
    bin_sample_count: np.ndarray,
) -> dict[str, np.ndarray]:
    """Summarize connected occupied regions on a periodic angular grid.

    These are low-speed angle clusters, not fixed points: a cluster only says
    that the strict speed criterion retained states in a connected set of
    bins.  Direct flow reversal remains necessary for fixed-point claims.
    """
    bin_sample_count = np.asarray(bin_sample_count, dtype=int)
    if bin_sample_count.ndim != 1 or bin_sample_count.size < 2:
        raise ValueError("bin_sample_count must be a one-dimensional circular grid")
    occupied = bin_sample_count > 0
    if not np.any(occupied):
        return {
            "cluster_theta": np.empty(0),
            "cluster_sample_count": np.empty(0, dtype=int),
            "cluster_bin_count": np.empty(0, dtype=int),
        }
    if np.all(occupied):
        starts = np.asarray([0], dtype=int)
    else:
        starts = np.flatnonzero(occupied & ~np.roll(occupied, 1))
    angular_center = np.linspace(
        -np.pi + np.pi / bin_sample_count.size,
        np.pi - np.pi / bin_sample_count.size,
        bin_sample_count.size,
    )
    cluster_theta: list[float] = []
    cluster_sample_count: list[int] = []
    cluster_bin_count: list[int] = []
    for start in starts:
        indices: list[int] = []
        current = int(start)
        while occupied[current] and (
            not indices or current != int(start)
        ):
            indices.append(current)
            current = (current + 1) % occupied.size
        index_array = np.asarray(indices, dtype=int)
        weights = bin_sample_count[index_array].astype(float)
        circular_vector = np.sum(weights * np.exp(1j * angular_center[index_array]))
        cluster_theta.append(float(wrap_angle(np.angle(circular_vector))))
        cluster_sample_count.append(int(np.sum(weights)))
        cluster_bin_count.append(int(index_array.size))
    return {
        "cluster_theta": np.asarray(cluster_theta, dtype=float),
        "cluster_sample_count": np.asarray(cluster_sample_count, dtype=int),
        "cluster_bin_count": np.asarray(cluster_bin_count, dtype=int),
    }


def select_slow_candidate_indices(
    *,
    speed: np.ndarray,
    speed_fraction: float,
    maximum_points: int,
    speed_floor: float | None = None,
    time: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Select points below a trajectory-relative (and optionally physical) speed threshold.

    Ságodi et al. use one thousandth of the maximum speed along each
    trajectory.  For a network whose bump collapses onto discrete attractors
    the trajectory maximum is set by the initial relaxation transient
    (``|F| ~ 1e3 /s``), so ``speed_fraction * max`` alone is too permissive:
    it admits mid-relaxation points and biases the slow set toward the
    late-time basin.  When ``speed_floor`` is provided the effective threshold
    is ``min(speed_fraction * max_speed, speed_floor)``, i.e. a physical speed
    below the pinning barrier.  Points are then genuinely close to the
    attracting set, and the angular support of the candidates directly probes
    whether that set covers the whole ring (continuous-like) or only isolated
    basins (discrete).

    When more points qualify than the storage budget and ``time`` is provided,
    candidates are re-sampled uniformly in *time* (not index), so early
    relaxation and late fixed-point samples contribute in proportion to their
    duration instead of crowding the final basin.
    """
    speed = np.asarray(speed, dtype=float)
    if speed.ndim != 1 or speed.size == 0:
        raise ValueError("speed must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(speed)) or np.any(speed < 0.0):
        raise ValueError("speed must be finite and non-negative")
    if not 0.0 < speed_fraction < 1.0:
        raise ValueError("speed_fraction must lie strictly between zero and one")
    if maximum_points <= 0:
        raise ValueError("maximum_points must be positive")
    if speed_floor is not None:
        speed_floor = float(speed_floor)
        if not np.isfinite(speed_floor) or speed_floor <= 0.0:
            raise ValueError("speed_floor must be finite and positive")
    if time is not None:
        time = np.asarray(time, dtype=float)
        if time.shape != speed.shape or not np.all(np.isfinite(time)):
            raise ValueError("time must be finite and match speed in shape")

    relative_threshold = speed_fraction * float(np.max(speed))
    threshold = (
        min(relative_threshold, speed_floor)
        if speed_floor is not None
        else relative_threshold
    )
    candidate_index = np.flatnonzero(speed <= threshold)
    if candidate_index.size > maximum_points:
        if time is not None and time.size > 1:
            # Uniform time-quantile re-sampling: split the candidate time span
            # into equal durations and pick the first candidate in each slot,
            # so early and late portions of the trajectory contribute fairly.
            candidate_time = time[candidate_index]
            duration = float(candidate_time[-1] - candidate_time[0])
            if duration <= 1e-12:
                selection = np.rint(
                    np.linspace(0, candidate_index.size - 1, maximum_points)
                ).astype(int)
            else:
                slot_edges = np.linspace(
                    float(candidate_time[0]),
                    float(candidate_time[-1]),
                    maximum_points + 1,
                )
                # Store *positions into candidate_index* for every selected
                # candidate, then map once at the end.  Mixing values with
                # positions caused an out-of-bounds index on short candidates.
                selection = np.empty(maximum_points, dtype=int)
                filled = 0
                for slot in range(maximum_points):
                    left = slot_edges[slot]
                    right = slot_edges[slot + 1]
                    slot_indices = np.flatnonzero(
                        (candidate_time >= left) & (candidate_time < right)
                    )
                    if slot_indices.size:
                        selection[filled] = int(slot_indices[0])
                        filled += 1
                if filled < maximum_points:
                    # Fill the remaining slots with evenly spaced candidate
                    # positions (not candidate values), then map once below.
                    remainder = np.rint(
                        np.linspace(0, candidate_index.size - 1, maximum_points - filled)
                    ).astype(int)
                    selection[filled:] = remainder
            # Map candidate positions -> actual trajectory indices once.
            candidate_index = candidate_index[selection]
        else:
            selection = np.rint(
                np.linspace(0, candidate_index.size - 1, maximum_points)
            ).astype(int)
            candidate_index = candidate_index[selection]
    return candidate_index, threshold


def fit_periodic_state_curve(
    *,
    candidate_theta: np.ndarray,
    candidate_state: np.ndarray,
    angular_bin_count: int,
) -> dict[str, np.ndarray | float]:
    """Fit ``state = c(theta)`` with a periodic cubic spline.

    Candidate states are first averaged inside circular angular bins.  This
    removes duplicate output angles and prevents dense late-time samples from
    dominating the spline.  Missing bins are bridged by the periodic spline
    and are exposed through ``angular_support_fraction`` for QC.
    """
    candidate_theta = np.asarray(wrap_angle(candidate_theta), dtype=float)
    candidate_state = np.asarray(candidate_state, dtype=float)
    if candidate_theta.ndim != 1 or candidate_theta.size < 4:
        raise ValueError("at least four slow candidate angles are required")
    if candidate_state.ndim != 2 or candidate_state.shape[0] != candidate_theta.size:
        raise ValueError("candidate_state must have shape (candidate, state_dimension)")
    if not np.all(np.isfinite(candidate_theta)) or not np.all(np.isfinite(candidate_state)):
        raise ValueError("slow manifold candidates must be finite")
    if angular_bin_count < 8:
        raise ValueError("angular_bin_count must be at least eight")

    angular_edges = np.linspace(-np.pi, np.pi, angular_bin_count + 1)
    angular_centers = 0.5 * (angular_edges[:-1] + angular_edges[1:])
    bin_index = np.searchsorted(
        angular_edges,
        candidate_theta,
        side="right",
    ) - 1
    bin_index = np.clip(bin_index, 0, angular_bin_count - 1)
    bin_sample_count = np.bincount(bin_index, minlength=angular_bin_count)
    observed_mask = bin_sample_count > 0
    if np.count_nonzero(observed_mask) < 4:
        raise ValueError("slow candidates cover fewer than four angular bins")
    observed_theta = angular_centers[observed_mask]
    observed_state = np.stack(
        [
            np.mean(candidate_state[bin_index == current_bin], axis=0)
            for current_bin in np.flatnonzero(observed_mask)
        ],
        axis=0,
    )

    spline_theta = np.concatenate([observed_theta, [observed_theta[0] + 2.0 * np.pi]])
    spline_state = np.concatenate([observed_state, observed_state[:1]], axis=0)
    spline = CubicSpline(
        spline_theta,
        spline_state,
        axis=0,
        bc_type="periodic",
    )
    theta_grid = angular_centers
    evaluation_theta = (
        (theta_grid - observed_theta[0]) % (2.0 * np.pi) + observed_theta[0]
    )
    manifold_state = np.asarray(spline(evaluation_theta), dtype=float)
    manifold_tangent = np.asarray(spline(evaluation_theta, 1), dtype=float)
    tangent_norm = np.linalg.norm(manifold_tangent, axis=1)
    if np.any(tangent_norm <= 1e-12):
        raise ValueError("periodic slow-manifold spline has a degenerate tangent")
    return {
        "theta": theta_grid,
        "state": manifold_state,
        "tangent": manifold_tangent,
        "tangent_norm": tangent_norm,
        "angular_bin_sample_count": bin_sample_count,
        "angular_support_fraction": float(np.mean(observed_mask)),
        "observed_theta": observed_theta,
        "observed_state": observed_state,
    }


def _periodic_scalar_roots(
    *,
    theta: np.ndarray,
    value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    value = np.asarray(value, dtype=float)
    angular_step = 2.0 * np.pi / theta.size
    derivative = (np.roll(value, -1) - np.roll(value, 1)) / (2.0 * angular_step)
    tolerance = max(1e-12, 1e-10 * float(np.max(np.abs(value))))
    roots: list[float] = []
    slopes: list[float] = []
    stability: list[int] = []
    for index in range(theta.size):
        next_index = (index + 1) % theta.size
        left_value = float(value[index])
        right_value = float(value[next_index])
        if abs(left_value) <= tolerance:
            previous_value = float(value[(index - 1) % theta.size])
            if previous_value * right_value >= 0.0:
                continue
            root = float(theta[index])
            slope = float(derivative[index])
        elif left_value * right_value < 0.0:
            right_theta = float(theta[next_index])
            if next_index == 0:
                right_theta += 2.0 * np.pi
            fraction = -left_value / (right_value - left_value)
            root = float(
                wrap_angle(theta[index] + fraction * (right_theta - theta[index]))
            )
            slope = float(
                derivative[index]
                + fraction * (derivative[next_index] - derivative[index])
            )
        else:
            continue
        if any(
            abs(float(wrap_angle(root - existing))) < 0.5 * angular_step
            for existing in roots
        ):
            continue
        roots.append(root)
        slopes.append(slope)
        stability.append(-1 if slope < 0.0 else 1)
    return (
        np.asarray(roots, dtype=float),
        np.asarray(slopes, dtype=float),
        np.asarray(stability, dtype=np.int8),
    )


def _basin_summary(
    *,
    fixed_point_theta: np.ndarray,
    fixed_point_stability: np.ndarray,
) -> dict[str, np.ndarray | float]:
    order = np.argsort(fixed_point_theta)
    theta = np.asarray(fixed_point_theta, dtype=float)[order]
    stability = np.asarray(fixed_point_stability, dtype=np.int8)[order]
    stable_theta: list[float] = []
    left_boundary: list[float] = []
    right_boundary: list[float] = []
    basin_width: list[float] = []
    for index, current_stability in enumerate(stability):
        if current_stability != -1 or theta.size < 3:
            continue
        previous_index = (index - 1) % theta.size
        next_index = (index + 1) % theta.size
        if stability[previous_index] != 1 or stability[next_index] != 1:
            continue
        left = float(theta[previous_index])
        right = float(theta[next_index])
        width = float((right - left) % (2.0 * np.pi))
        stable_theta.append(float(theta[index]))
        left_boundary.append(left)
        right_boundary.append(right)
        basin_width.append(width)
    widths = np.asarray(basin_width, dtype=float)
    if widths.size and float(np.sum(widths)) > 0.0:
        basin_fraction = widths / float(np.sum(widths))
        basin_entropy = float(-np.sum(basin_fraction * np.log(basin_fraction)))
    else:
        basin_fraction = np.empty(0, dtype=float)
        basin_entropy = float("nan")
    return {
        "basin_stable_theta": np.asarray(stable_theta, dtype=float),
        "basin_left_boundary": np.asarray(left_boundary, dtype=float),
        "basin_right_boundary": np.asarray(right_boundary, dtype=float),
        "basin_width": widths,
        "basin_fraction": basin_fraction,
        "basin_entropy": basin_entropy,
    }


def _leading_eigensystem(
    *,
    jacobian: np.ndarray,
    eigenvalue_count: int,
    dense_dimension_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    state_dimension = jacobian.shape[0]
    requested_count = min(eigenvalue_count, state_dimension)
    if state_dimension <= dense_dimension_limit or requested_count >= state_dimension - 1:
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    else:
        try:
            eigenvalues, eigenvectors = eigs(
                jacobian,
                k=min(requested_count, state_dimension - 2),
                which="LR",
                tol=1e-7,
                maxiter=max(2000, 20 * state_dimension),
            )
        except ArpackNoConvergence as error:
            raise RuntimeError(
                "leading Jacobian eigensolver did not converge; reduce the "
                "state dimension or increase the dense-dimension limit"
            ) from error
    order = np.argsort(eigenvalues.real)[::-1][:requested_count]
    return eigenvalues[order], eigenvectors[:, order]


def analyze_ramesan_firing_rate_geometry(
    *,
    dynamics: FrozenAutonomousDynamics,
    probe_phase: np.ndarray,
    probe_state: np.ndarray,
    candidate_theta: np.ndarray,
    candidate_state: np.ndarray,
    candidate_speed: np.ndarray,
    q_threshold: float,
    jacobian_anchor_count: int,
    jacobian_eigenvalue_count: int,
    jacobian_dense_dimension_limit: int,
    show_progress: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Diagnose a cue-defined ring using Ramesan-style ``q`` and Jacobians.

    ``probe_state`` contains zero-input canonical states obtained after visual
    cue settling.  It defines a closed *candidate* ring parameterized by the
    uniformly sampled cue phase.  The primary visualization PCA is fitted to
    the block-standardized full canonical Markov state.  Separate spectra are
    retained for the HD+HR firing-rate observable and the paired-HD angular-rate
    statistic entering PVA.  Flow, q, tangent projection, and the Jacobian are
    all evaluated in the unprojected full canonical Markov state.
    """
    probe_phase = np.asarray(wrap_angle(probe_phase), dtype=float)
    probe_state = np.asarray(probe_state, dtype=float)
    candidate_theta = np.asarray(wrap_angle(candidate_theta), dtype=float)
    candidate_state = np.asarray(candidate_state, dtype=float)
    candidate_speed = np.asarray(candidate_speed, dtype=float)
    if probe_phase.ndim != 1 or probe_phase.size < 4:
        raise ValueError("Ramesan geometry requires at least four ring probes")
    if probe_state.shape != (probe_phase.size, dynamics.state_dimension):
        raise ValueError(
            "probe_state must have shape (probe, autonomous_state_dimension)"
        )
    if not np.all(np.isfinite(probe_phase)) or not np.all(np.isfinite(probe_state)):
        raise ValueError("Ramesan ring probes must be finite")
    if candidate_state.ndim != 2 or candidate_state.shape[1:] != (
        dynamics.state_dimension,
    ):
        raise ValueError(
            "candidate_state must have shape (candidate, autonomous_state_dimension)"
        )
    if candidate_theta.shape != (candidate_state.shape[0],):
        raise ValueError("candidate_theta must match candidate_state rows")
    if candidate_speed.shape != (candidate_state.shape[0],):
        raise ValueError("candidate_speed must match candidate_state rows")
    if not np.all(np.isfinite(candidate_state)) or not np.all(
        np.isfinite(candidate_speed)
    ):
        raise ValueError("Ramesan slow candidates must be finite")
    if np.any(candidate_speed < 0.0):
        raise ValueError("candidate_speed must be non-negative")
    if q_threshold <= 0.0 or not np.isfinite(q_threshold):
        raise ValueError("q_threshold must be finite and positive")
    if jacobian_anchor_count < 1:
        raise ValueError("jacobian_anchor_count must be positive")
    if jacobian_eigenvalue_count < 1:
        raise ValueError("jacobian_eigenvalue_count must be positive")

    state_center = np.mean(probe_state, axis=0)
    state_feature_scale = _canonical_state_block_scale(
        dynamics=dynamics,
        sample_by_state=probe_state,
    )
    standardized_probe_state = (
        probe_state - state_center
    ) / state_feature_scale
    state_pca = _pca_variance_summary(standardized_probe_state)
    standardized_state_components = np.asarray(
        state_pca["components"], dtype=float
    )
    state_singular_value = np.asarray(
        state_pca["singular_value"], dtype=float
    )
    explained_variance_spectrum = np.asarray(
        state_pca["explained_fraction"], dtype=float
    )
    cumulative_explained_variance = np.asarray(
        state_pca["cumulative_fraction"], dtype=float
    )
    retained_component_count = min(3, standardized_state_components.shape[0])
    pca_components = np.zeros((3, dynamics.state_dimension), dtype=float)
    pca_components[:retained_component_count] = (
        standardized_state_components[:retained_component_count]
        / state_feature_scale[None, :]
    )
    probe_pc = (probe_state - state_center) @ pca_components.T
    explained_variance = np.zeros(3, dtype=float)
    explained_variance[:retained_component_count] = explained_variance_spectrum[
        :retained_component_count
    ]

    probe_firing_rate = np.stack(
        [dynamics.firing_rate_state(state_vector) for state_vector in probe_state],
        axis=0,
    )
    firing_rate_pca = _pca_variance_summary(probe_firing_rate)
    firing_rate_explained_variance_spectrum = np.asarray(
        firing_rate_pca["explained_fraction"], dtype=float
    )
    firing_rate_cumulative_explained_variance = np.asarray(
        firing_rate_pca["cumulative_fraction"], dtype=float
    )

    pva_observable = [
        dynamics.pva_angular_rate_state(state_vector)
        for state_vector in probe_state
    ]
    pva_theta_pref = np.asarray(pva_observable[0][0], dtype=float)
    if not all(
        np.array_equal(theta_pref, pva_theta_pref)
        for theta_pref, _angular_rate in pva_observable
    ):
        raise RuntimeError("PVA preferred-direction grid changed across probe states")
    pva_angular_rate = np.stack(
        [angular_rate for _theta_pref, angular_rate in pva_observable],
        axis=0,
    )
    pva_rate_pca = _pca_variance_summary(pva_angular_rate)
    pva_explained_variance_spectrum = np.asarray(
        pva_rate_pca["explained_fraction"], dtype=float
    )
    pva_cumulative_explained_variance = np.asarray(
        pva_rate_pca["cumulative_fraction"], dtype=float
    )

    if candidate_state.shape[0]:
        candidate_firing_rate = np.stack(
            [
                dynamics.firing_rate_state(state_vector)
                for state_vector in candidate_state
            ],
            axis=0,
        )
        candidate_pc = (candidate_state - state_center) @ pca_components.T
    else:
        candidate_firing_rate = np.empty(
            (0, probe_firing_rate.shape[1]), dtype=float
        )
        candidate_pc = np.empty((0, 3), dtype=float)

    circular_phase_step = np.diff(
        np.unwrap(np.concatenate([probe_phase, [probe_phase[0]]]))
    )
    mean_phase_step = float(2.0 * np.pi / probe_phase.size)
    if not np.allclose(
        circular_phase_step,
        mean_phase_step,
        rtol=1e-4,
        atol=1e-8,
    ):
        raise ValueError("probe_phase must be a uniformly ordered circular grid")
    probe_tangent = (
        np.roll(probe_state, -1, axis=0) - np.roll(probe_state, 1, axis=0)
    ) / (2.0 * mean_phase_step)
    tangent_norm = np.linalg.norm(probe_tangent, axis=1)
    tangent_defined = tangent_norm > 1e-12
    tangent_unit = np.zeros_like(probe_tangent)
    tangent_unit[tangent_defined] = (
        probe_tangent[tangent_defined] / tangent_norm[tangent_defined, None]
    )
    probe_flow = np.stack(
        [dynamics.flow(state_vector) for state_vector in probe_state],
        axis=0,
    )
    tangent_state_speed = np.full(probe_phase.size, np.nan, dtype=float)
    tangent_state_speed[tangent_defined] = np.sum(
        tangent_unit[tangent_defined] * probe_flow[tangent_defined], axis=1
    )
    coordinate_phase_velocity = np.full(probe_phase.size, np.nan, dtype=float)
    coordinate_phase_velocity[tangent_defined] = np.sum(
        probe_tangent[tangent_defined] * probe_flow[tangent_defined], axis=1
    ) / np.square(tangent_norm[tangent_defined])
    tangent_flow = tangent_unit * tangent_state_speed[:, None]
    tangent_flow[~tangent_defined] = 0.0
    normal_flow_norm = np.linalg.norm(probe_flow - tangent_flow, axis=1)
    probe_flow_norm = np.linalg.norm(probe_flow, axis=1)
    probe_q = 0.5 * np.square(probe_flow_norm)
    candidate_q = 0.5 * np.square(candidate_speed)
    candidate_below_q_threshold = candidate_q <= q_threshold

    eligible_candidate_index = np.flatnonzero(candidate_below_q_threshold)
    if eligible_candidate_index.size:
        eligible_candidate_index = eligible_candidate_index[
            np.argsort(candidate_theta[eligible_candidate_index])
        ]
        selected_position = np.unique(
            np.rint(
                np.linspace(
                    0,
                    eligible_candidate_index.size - 1,
                    min(jacobian_anchor_count, eligible_candidate_index.size),
                )
            ).astype(int)
        )
        anchor_index = eligible_candidate_index[selected_position]
    else:
        anchor_index = np.empty(0, dtype=int)
    lambda_max_real = np.empty(anchor_index.size, dtype=float)
    anchor_iterable = tqdm(
        enumerate(anchor_index),
        total=anchor_index.size,
        disable=not show_progress,
        desc="slow-point Jacobian lambda_max",
        unit="anchor",
        dynamic_ncols=True,
    )
    for slot, current_index in anchor_iterable:
        jacobian = dynamics.flow_jacobian(candidate_state[current_index])
        try:
            eigenvalues, _eigenvectors = _leading_eigensystem(
                jacobian=jacobian,
                eigenvalue_count=jacobian_eigenvalue_count,
                dense_dimension_limit=jacobian_dense_dimension_limit,
            )
            lambda_max_real[slot] = float(np.max(eigenvalues.real))
        except RuntimeError:
            lambda_max_real[slot] = np.nan

    adjacent_rate_distance = np.linalg.norm(
        np.roll(probe_firing_rate, -1, axis=0) - probe_firing_rate,
        axis=1,
    )
    median_adjacent_distance = float(np.median(adjacent_rate_distance))
    closure_ratio = (
        float(adjacent_rate_distance[-1] / median_adjacent_distance)
        if median_adjacent_distance > 0.0
        else float("nan")
    )
    state_dimension = float(dynamics.state_dimension)
    finite_lambda_max = lambda_max_real[np.isfinite(lambda_max_real)]
    history = {
        "ramesan_probe_phase": probe_phase,
        "ramesan_probe_decoded_theta": np.asarray(
            [dynamics.decoded_heading(state_vector) for state_vector in probe_state],
            dtype=float,
        ),
        "ramesan_probe_state": probe_state,
        "ramesan_probe_firing_rate": probe_firing_rate,
        "ramesan_pca_center": state_center,
        "ramesan_pca_feature_scale": state_feature_scale,
        "ramesan_pca_components": pca_components,
        "ramesan_pca_standardized_components": standardized_state_components,
        "ramesan_pca_explained_variance": explained_variance,
        "ramesan_pca_singular_value": state_singular_value,
        "ramesan_pca_explained_variance_spectrum": explained_variance_spectrum,
        "ramesan_pca_cumulative_explained_variance": cumulative_explained_variance,
        "ramesan_firing_rate_pca_center": np.asarray(
            firing_rate_pca["center"], dtype=float
        ),
        "ramesan_firing_rate_pca_components": np.asarray(
            firing_rate_pca["components"], dtype=float
        ),
        "ramesan_firing_rate_pca_singular_value": np.asarray(
            firing_rate_pca["singular_value"], dtype=float
        ),
        "ramesan_firing_rate_pca_explained_variance_spectrum": (
            firing_rate_explained_variance_spectrum
        ),
        "ramesan_firing_rate_pca_cumulative_explained_variance": (
            firing_rate_cumulative_explained_variance
        ),
        "ramesan_pva_theta_pref": pva_theta_pref,
        "ramesan_pva_angular_rate": pva_angular_rate,
        "ramesan_pva_rate_pca_singular_value": np.asarray(
            pva_rate_pca["singular_value"], dtype=float
        ),
        "ramesan_pva_rate_pca_explained_variance_spectrum": (
            pva_explained_variance_spectrum
        ),
        "ramesan_pva_rate_pca_cumulative_explained_variance": (
            pva_cumulative_explained_variance
        ),
        "ramesan_probe_pc": probe_pc,
        "ramesan_probe_flow": probe_flow,
        "ramesan_probe_q": probe_q,
        "ramesan_probe_tangent": probe_tangent,
        "ramesan_probe_tangent_defined": tangent_defined,
        "ramesan_probe_tangent_state_speed": tangent_state_speed,
        "ramesan_probe_coordinate_phase_velocity": coordinate_phase_velocity,
        "ramesan_probe_normal_flow_norm": normal_flow_norm,
        "ramesan_candidate_theta": candidate_theta,
        "ramesan_candidate_firing_rate": candidate_firing_rate,
        "ramesan_candidate_pc": candidate_pc,
        "ramesan_candidate_q": candidate_q,
        "ramesan_candidate_below_q_threshold": candidate_below_q_threshold,
        "ramesan_q_threshold": np.asarray(q_threshold),
        "ramesan_jacobian_anchor_index": anchor_index,
        "ramesan_jacobian_anchor_theta": candidate_theta[anchor_index],
        "ramesan_jacobian_anchor_pc": candidate_pc[anchor_index],
        "ramesan_jacobian_anchor_q": candidate_q[anchor_index],
        "ramesan_jacobian_lambda_max_real": lambda_max_real,
    }
    metrics = {
        "ramesan_diagnostic_succeeded": 1.0,
        "ramesan_probe_count": float(probe_phase.size),
        "ramesan_pca_pc1_explained_fraction": float(explained_variance[0]),
        "ramesan_pca_pc2_explained_fraction": float(explained_variance[1]),
        "ramesan_pca_pc3_explained_fraction": float(explained_variance[2]),
        "ramesan_pca_pc123_explained_fraction": float(
            np.sum(explained_variance)
        ),
        "ramesan_pca_effective_dimension_participation_ratio": float(
            state_pca["participation_ratio"]
        ),
        "ramesan_pca_rank_80pct": _minimum_rank_for_fraction(
            cumulative_explained_variance, 0.80
        ),
        "ramesan_pca_rank_90pct": _minimum_rank_for_fraction(
            cumulative_explained_variance, 0.90
        ),
        "ramesan_pca_rank_95pct": _minimum_rank_for_fraction(
            cumulative_explained_variance, 0.95
        ),
        "ramesan_firing_rate_pca_pc123_explained_fraction": float(
            np.sum(firing_rate_explained_variance_spectrum[:3])
        ),
        "ramesan_firing_rate_pca_effective_dimension_participation_ratio": float(
            firing_rate_pca["participation_ratio"]
        ),
        "ramesan_firing_rate_pca_rank_80pct": _minimum_rank_for_fraction(
            firing_rate_cumulative_explained_variance, 0.80
        ),
        "ramesan_firing_rate_pca_rank_90pct": _minimum_rank_for_fraction(
            firing_rate_cumulative_explained_variance, 0.90
        ),
        "ramesan_firing_rate_pca_rank_95pct": _minimum_rank_for_fraction(
            firing_rate_cumulative_explained_variance, 0.95
        ),
        "ramesan_pva_rate_pca_pc123_explained_fraction": float(
            np.sum(pva_explained_variance_spectrum[:3])
        ),
        "ramesan_pva_rate_pca_effective_dimension_participation_ratio": float(
            pva_rate_pca["participation_ratio"]
        ),
        "ramesan_pva_rate_pca_rank_80pct": _minimum_rank_for_fraction(
            pva_cumulative_explained_variance, 0.80
        ),
        "ramesan_pva_rate_pca_rank_90pct": _minimum_rank_for_fraction(
            pva_cumulative_explained_variance, 0.90
        ),
        "ramesan_pva_rate_pca_rank_95pct": _minimum_rank_for_fraction(
            pva_cumulative_explained_variance, 0.95
        ),
        "ramesan_firing_rate_ring_closure_ratio": closure_ratio,
        "ramesan_probe_q_min": float(np.min(probe_q)),
        "ramesan_probe_q_median": float(np.median(probe_q)),
        "ramesan_probe_q_max": float(np.max(probe_q)),
        "ramesan_probe_tangent_defined_fraction": float(
            np.mean(tangent_defined)
        ),
        "ramesan_probe_tangent_speed_abs_max_per_sqrt_dim": float(
            np.nanmax(np.abs(tangent_state_speed)) / np.sqrt(state_dimension)
            if np.any(tangent_defined)
            else float("nan")
        ),
        "ramesan_probe_normal_speed_median_per_sqrt_dim": float(
            np.median(normal_flow_norm) / np.sqrt(state_dimension)
        ),
        "ramesan_q_threshold": float(q_threshold),
        "ramesan_candidate_count": float(candidate_q.size),
        "ramesan_candidate_below_q_threshold_count": float(
            np.count_nonzero(candidate_below_q_threshold)
        ),
        "ramesan_candidate_below_q_threshold_fraction": (
            float(np.mean(candidate_below_q_threshold))
            if candidate_q.size
            else float("nan")
        ),
        "ramesan_candidate_q_min": (
            float(np.min(candidate_q)) if candidate_q.size else float("nan")
        ),
        "ramesan_jacobian_anchor_count": float(anchor_index.size),
        "ramesan_jacobian_converged_fraction": float(
            finite_lambda_max.size / lambda_max_real.size
        ) if lambda_max_real.size else float("nan"),
        "ramesan_lambda_max_real_min": (
            float(np.min(finite_lambda_max))
            if finite_lambda_max.size
            else float("nan")
        ),
        "ramesan_lambda_max_real_max": (
            float(np.max(finite_lambda_max))
            if finite_lambda_max.size
            else float("nan")
        ),
        "ramesan_lambda_max_abs_min": (
            float(np.min(np.abs(finite_lambda_max)))
            if finite_lambda_max.size
            else float("nan")
        ),
    }
    return history, metrics


def _fill_and_smooth_periodic_scalar(
    value: np.ndarray,
    *,
    smoothing_bins: int,
) -> np.ndarray:
    """Periodically interpolate missing bins and apply a circular box filter."""
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or value.size < 4:
        raise ValueError("periodic scalar field must have at least four bins")
    observed = np.isfinite(value)
    if np.count_nonzero(observed) < 4:
        return np.full(value.shape, np.nan, dtype=float)
    index = np.arange(value.size, dtype=float)
    observed_index = index[observed]
    observed_value = value[observed]
    extended_index = np.concatenate(
        [observed_index - value.size, observed_index, observed_index + value.size]
    )
    extended_value = np.tile(observed_value, 3)
    filled = np.interp(index, extended_index, extended_value)
    window = max(1, int(smoothing_bins))
    if window > value.size:
        window = value.size
    if window % 2 == 0 and window > 1:
        window -= 1
    if window <= 1:
        return filled
    half_width = window // 2
    padded = np.pad(filled, (half_width, half_width), mode="wrap")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _periodic_effective_potential(phase_velocity: np.ndarray) -> tuple[np.ndarray, float]:
    """Integrate the zero-mean part of a periodic one-dimensional flow.

    The returned field obeys ``v(theta) = mean_v - dU/dtheta`` on the sampled
    periodic grid.  A non-zero ``mean_v`` is a non-conservative circulation
    that no single-valued periodic potential can represent.
    """
    phase_velocity = np.asarray(phase_velocity, dtype=float)
    if phase_velocity.ndim != 1 or phase_velocity.size < 4:
        raise ValueError("phase velocity must be a one-dimensional periodic grid")
    if not np.all(np.isfinite(phase_velocity)):
        return np.full(phase_velocity.shape, np.nan), float("nan")
    mean_velocity = float(np.mean(phase_velocity))
    periodic_velocity = phase_velocity - mean_velocity
    mode = np.fft.fftfreq(phase_velocity.size, d=1.0 / phase_velocity.size)
    velocity_fourier = np.fft.fft(periodic_velocity)
    potential_fourier = np.zeros_like(velocity_fourier, dtype=complex)
    nonzero = mode != 0.0
    potential_fourier[nonzero] = -velocity_fourier[nonzero] / (
        1j * mode[nonzero]
    )
    potential = np.fft.ifft(potential_fourier).real
    potential -= float(np.min(potential))
    return potential, mean_velocity


def _sample_ambient_q_tube(
    *,
    dynamics: FrozenAutonomousDynamics,
    probe_phase: np.ndarray,
    probe_state: np.ndarray,
    base_theta: np.ndarray,
    base_state: np.ndarray,
    sample_count: int,
    perturbation_scales: np.ndarray,
    seed: int,
    pca_center: np.ndarray,
    pca_components: np.ndarray,
    show_progress: bool,
) -> dict[str, np.ndarray]:
    """Evaluate ``q`` in a standardized normal tube around trajectory states.

    This is an ambient-state probe, not slow-point optimization.  Each random
    direction is made orthogonal to the nearest cue-ring tangent after the
    five canonical state blocks are standardized by their empirical scale.
    """
    base_theta = np.asarray(wrap_angle(base_theta), dtype=float)
    base_state = np.asarray(base_state, dtype=float)
    perturbation_scales = np.asarray(perturbation_scales, dtype=float)
    if sample_count <= 0 or base_theta.size == 0:
        return {
            "theta": np.empty(0),
            "base_theta": np.empty(0),
            "q": np.empty(0),
            "requested_scale": np.empty(0),
            "standardized_distance": np.empty(0),
            "pc": np.empty((0, 3)),
        }
    if base_state.shape != (base_theta.size, dynamics.state_dimension):
        raise ValueError("ambient base states must match base_theta and state dimension")
    if perturbation_scales.ndim != 1 or perturbation_scales.size == 0:
        raise ValueError("ambient perturbation scales must be non-empty")
    if not np.all(np.isfinite(perturbation_scales)) or np.any(
        perturbation_scales <= 0.0
    ):
        raise ValueError("ambient perturbation scales must be finite and positive")

    probe_phase = np.asarray(wrap_angle(probe_phase), dtype=float)
    probe_state = np.asarray(probe_state, dtype=float)
    phase_step = 2.0 * np.pi / probe_phase.size
    probe_tangent = (
        np.roll(probe_state, -1, axis=0) - np.roll(probe_state, 1, axis=0)
    ) / (2.0 * phase_step)

    state_scale = _canonical_state_block_scale(
        dynamics=dynamics,
        sample_by_state=base_state,
    )

    rng = np.random.default_rng(int(seed))
    sorted_base_index = np.argsort(base_theta)
    samples_per_scale = np.full(
        perturbation_scales.size,
        sample_count // perturbation_scales.size,
        dtype=int,
    )
    samples_per_scale[: sample_count % perturbation_scales.size] += 1
    ambient_theta = np.empty(sample_count, dtype=float)
    ambient_base_theta = np.empty(sample_count, dtype=float)
    ambient_q = np.empty(sample_count, dtype=float)
    requested_scale = np.empty(sample_count, dtype=float)
    standardized_distance = np.empty(sample_count, dtype=float)
    ambient_pc = np.empty((sample_count, 3), dtype=float)
    output_index = 0
    progress = tqdm(
        total=sample_count,
        disable=not show_progress,
        desc="ambient Ramesan q tube",
        unit="sample",
        dynamic_ncols=True,
    )
    with progress:
        for scale, current_count in zip(perturbation_scales, samples_per_scale):
            if current_count == 0:
                continue
            base_positions = np.rint(
                np.linspace(0, sorted_base_index.size - 1, current_count)
            ).astype(int)
            for base_position in base_positions:
                base_index = int(sorted_base_index[base_position])
                current_base = base_state[base_index]
                current_theta = float(base_theta[base_index])
                phase_distance = np.abs(wrap_angle(probe_phase - current_theta))
                tangent = probe_tangent[int(np.argmin(phase_distance))] / state_scale
                direction = rng.normal(size=dynamics.state_dimension)
                tangent_squared_norm = float(np.dot(tangent, tangent))
                if tangent_squared_norm > 1e-12:
                    direction -= (
                        float(np.dot(direction, tangent)) / tangent_squared_norm
                    ) * tangent
                direction_rms = float(np.sqrt(np.mean(np.square(direction))))
                if direction_rms <= 1e-12:
                    continue
                direction /= direction_rms
                perturbed_state = current_base + float(scale) * state_scale * direction
                for rate_name in ("r_hd_to_hr_lp", "r_hr"):
                    rate_slice = dynamics.component_slices[rate_name]
                    perturbed_state[rate_slice] = np.clip(
                        perturbed_state[rate_slice], 0.0, 1.0
                    )
                standardized_delta = (perturbed_state - current_base) / state_scale
                flow = dynamics.flow(perturbed_state)
                ambient_base_theta[output_index] = current_theta
                ambient_theta[output_index] = dynamics.decoded_heading(perturbed_state)
                ambient_q[output_index] = 0.5 * float(np.dot(flow, flow))
                requested_scale[output_index] = float(scale)
                standardized_distance[output_index] = float(
                    np.sqrt(np.mean(np.square(standardized_delta)))
                )
                ambient_pc[output_index] = (
                    perturbed_state - pca_center
                ) @ pca_components.T
                output_index += 1
                progress.update(1)

    return {
        "theta": np.asarray(wrap_angle(ambient_theta[:output_index]), dtype=float),
        "base_theta": np.asarray(
            wrap_angle(ambient_base_theta[:output_index]), dtype=float
        ),
        "q": ambient_q[:output_index],
        "requested_scale": requested_scale[:output_index],
        "standardized_distance": standardized_distance[:output_index],
        "pc": ambient_pc[:output_index],
    }


def analyze_ramesan_phase_landscape(
    *,
    dynamics: FrozenAutonomousDynamics,
    probe_phase: np.ndarray,
    probe_state: np.ndarray,
    trajectory_theta: np.ndarray,
    trajectory_state: np.ndarray,
    trajectory_speed: np.ndarray,
    q_threshold: float,
    angular_bin_count: int,
    smoothing_bins: int,
    ambient_enabled: bool,
    ambient_sample_count: int,
    ambient_perturbation_scales: np.ndarray,
    ambient_seed: int,
    pca_center: np.ndarray,
    pca_components: np.ndarray,
    phase_velocity_floor: float = 1e-3,
    show_progress: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Map slow phase regions, empirical phase flow, and an ambient q tube.

    The decoded-PVA phase velocity is sampled on the trajectory grid (default
    0.1 s).  A bump that sits at a fixed point contributes zeros for many
    consecutive frames and crosses one angular bin in a single frame when it
    moves, so the raw within-bin median mixes the two regimes.  The phase
    field is therefore split into a *settled* fraction (frames with
    ``|v| < phase_velocity_floor``, i.e. time spent at an attracting phase)
    and a *moving* velocity (median over frames with ``|v| >= floor``, i.e.
    the actual drift rate while crossing).  This separates the two regimes
    instead of reporting a sampling-biased average.
    """
    trajectory_theta = np.asarray(wrap_angle(trajectory_theta), dtype=float)
    trajectory_state = np.asarray(trajectory_state, dtype=float)
    trajectory_speed = np.asarray(trajectory_speed, dtype=float)
    if trajectory_theta.size < 4:
        return {}, {"ramesan_phase_landscape_succeeded": 0.0}
    if trajectory_state.shape != (
        trajectory_theta.size,
        dynamics.state_dimension,
    ):
        raise ValueError("trajectory states must match phase and state dimension")
    if trajectory_speed.shape != trajectory_theta.shape:
        raise ValueError("trajectory speed must match trajectory phase")
    if angular_bin_count < 8:
        raise ValueError("ramesan phase angular bins must be at least eight")
    if smoothing_bins < 1:
        raise ValueError("ramesan phase smoothing bins must be positive")
    if ambient_sample_count < 0:
        raise ValueError("ramesan ambient sample count must be non-negative")
    phase_velocity_floor = float(phase_velocity_floor)
    if not np.isfinite(phase_velocity_floor) or phase_velocity_floor <= 0.0:
        raise ValueError("phase_velocity_floor must be finite and positive")

    phase_edges = np.linspace(-np.pi, np.pi, angular_bin_count + 1)
    phase_center = 0.5 * (phase_edges[:-1] + phase_edges[1:])
    bin_index = np.searchsorted(phase_edges, trajectory_theta, side="right") - 1
    bin_index = np.clip(bin_index, 0, angular_bin_count - 1)
    sample_count = np.bincount(bin_index, minlength=angular_bin_count)
    trajectory_q = 0.5 * np.square(trajectory_speed)
    phase_q_min = np.full(angular_bin_count, np.nan)
    phase_q_median = np.full(angular_bin_count, np.nan)
    phase_velocity_median = np.full(angular_bin_count, np.nan)
    phase_velocity_q25 = np.full(angular_bin_count, np.nan)
    phase_velocity_q75 = np.full(angular_bin_count, np.nan)
    phase_settled_fraction = np.full(angular_bin_count, np.nan)
    phase_moving_velocity_median = np.full(angular_bin_count, np.nan)
    phase_moving_velocity_q25 = np.full(angular_bin_count, np.nan)
    phase_moving_velocity_q75 = np.full(angular_bin_count, np.nan)

    instantaneous_velocity = np.empty(trajectory_theta.size, dtype=float)
    velocity_progress = tqdm(
        range(trajectory_theta.size),
        disable=not show_progress,
        desc="trajectory phase field",
        unit="state",
        dynamic_ncols=True,
    )
    for state_index in velocity_progress:
        next_state = dynamics.step(trajectory_state[state_index])
        next_theta = dynamics.decoded_heading(next_state)
        instantaneous_velocity[state_index] = float(
            wrap_angle(next_theta - trajectory_theta[state_index]) / dynamics.params.dt
        )

    for current_bin in np.flatnonzero(sample_count):
        current_mask = bin_index == current_bin
        phase_q_min[current_bin] = float(np.min(trajectory_q[current_mask]))
        phase_q_median[current_bin] = float(np.median(trajectory_q[current_mask]))
        bin_velocity = instantaneous_velocity[current_mask]
        phase_velocity_median[current_bin] = float(np.median(bin_velocity))
        phase_velocity_q25[current_bin], phase_velocity_q75[current_bin] = (
            np.quantile(bin_velocity, [0.25, 0.75])
        )
        settled = np.abs(bin_velocity) < phase_velocity_floor
        phase_settled_fraction[current_bin] = float(np.mean(settled))
        moving = ~settled
        if np.any(moving):
            moving_velocity = bin_velocity[moving]
            phase_moving_velocity_median[current_bin] = float(
                np.median(moving_velocity)
            )
            phase_moving_velocity_q25[current_bin], phase_moving_velocity_q75[
                current_bin
            ] = np.quantile(moving_velocity, [0.25, 0.75])

    # The moving velocity field is the meaningful phase flow: it is the drift
    # rate of the bump while it actually crosses a phase interval.  Smoothing
    # and root finding therefore use it (settled bins keep their zero floor
    # implicitly through the settled fraction).  Bins with no moving frames
    # are left as NaN and bridged periodically.
    smoothed_velocity = _fill_and_smooth_periodic_scalar(
        phase_moving_velocity_median,
        smoothing_bins=smoothing_bins,
    )
    effective_potential, mean_drift = _periodic_effective_potential(
        smoothed_velocity
    )
    tilted_potential = effective_potential - mean_drift * (
        phase_center - phase_center[0]
    )
    if np.all(np.isfinite(tilted_potential)):
        tilted_potential -= float(np.min(tilted_potential))
    if np.all(np.isfinite(smoothed_velocity)):
        fixed_theta, fixed_slope, fixed_stability = _periodic_scalar_roots(
            theta=phase_center,
            value=smoothed_velocity,
        )
    else:
        fixed_theta = np.empty(0)
        fixed_slope = np.empty(0)
        fixed_stability = np.empty(0, dtype=np.int8)
    slow_mask = np.isfinite(phase_q_min) & (phase_q_min <= q_threshold)

    ambient = _sample_ambient_q_tube(
        dynamics=dynamics,
        probe_phase=probe_phase,
        probe_state=probe_state,
        base_theta=trajectory_theta,
        base_state=trajectory_state,
        sample_count=ambient_sample_count if ambient_enabled else 0,
        perturbation_scales=np.asarray(ambient_perturbation_scales, dtype=float),
        seed=ambient_seed,
        pca_center=np.asarray(pca_center, dtype=float),
        pca_components=np.asarray(pca_components, dtype=float),
        show_progress=show_progress,
    )
    ambient_q = ambient["q"]
    velocity_iqr = phase_velocity_q75 - phase_velocity_q25
    observed_velocity_iqr = velocity_iqr[np.isfinite(velocity_iqr)]
    observed_settled = phase_settled_fraction[np.isfinite(phase_settled_fraction)]
    observed_moving_velocity = phase_moving_velocity_median[
        np.isfinite(phase_moving_velocity_median)
    ]
    history = {
        "ramesan_phase_bin_center": phase_center,
        "ramesan_phase_sample_count": sample_count,
        "ramesan_phase_q_min": phase_q_min,
        "ramesan_phase_q_median": phase_q_median,
        "ramesan_phase_slow_mask": slow_mask,
        "ramesan_phase_velocity_median": phase_velocity_median,
        "ramesan_phase_velocity_q25": phase_velocity_q25,
        "ramesan_phase_velocity_q75": phase_velocity_q75,
        "ramesan_phase_settled_fraction": phase_settled_fraction,
        "ramesan_phase_moving_velocity_median": phase_moving_velocity_median,
        "ramesan_phase_moving_velocity_q25": phase_moving_velocity_q25,
        "ramesan_phase_moving_velocity_q75": phase_moving_velocity_q75,
        "ramesan_phase_velocity_floor": np.asarray(phase_velocity_floor),
        "ramesan_phase_velocity_smoothed": smoothed_velocity,
        "ramesan_phase_effective_potential": effective_potential,
        "ramesan_phase_tilted_potential": tilted_potential,
        "ramesan_phase_mean_drift": np.asarray(mean_drift),
        "ramesan_phase_fixed_point_theta": fixed_theta,
        "ramesan_phase_fixed_point_slope": fixed_slope,
        "ramesan_phase_fixed_point_stability": fixed_stability,
        "ramesan_trajectory_theta": trajectory_theta,
        "ramesan_trajectory_q": trajectory_q,
        "ramesan_trajectory_phase_velocity": instantaneous_velocity,
        "ramesan_ambient_theta": ambient["theta"],
        "ramesan_ambient_base_theta": ambient["base_theta"],
        "ramesan_ambient_q": ambient_q,
        "ramesan_ambient_requested_scale": ambient["requested_scale"],
        "ramesan_ambient_standardized_distance": ambient[
            "standardized_distance"
        ],
        "ramesan_ambient_pc": ambient["pc"],
    }
    metrics = {
        "ramesan_phase_landscape_succeeded": float(
            np.all(np.isfinite(smoothed_velocity))
        ),
        "ramesan_phase_observed_fraction": float(np.mean(sample_count > 0)),
        "ramesan_phase_slow_fraction": float(np.mean(slow_mask)),
        "ramesan_phase_mean_drift": mean_drift,
        "ramesan_phase_velocity_iqr_median": (
            float(np.median(observed_velocity_iqr))
            if observed_velocity_iqr.size
            else float("nan")
        ),
        "ramesan_phase_settled_fraction_median": (
            float(np.median(observed_settled))
            if observed_settled.size
            else float("nan")
        ),
        "ramesan_phase_settled_fraction_min": (
            float(np.min(observed_settled))
            if observed_settled.size
            else float("nan")
        ),
        "ramesan_phase_moving_velocity_median_deg_s": (
            float(np.rad2deg(np.median(observed_moving_velocity)))
            if observed_moving_velocity.size
            else float("nan")
        ),
        "ramesan_phase_moving_velocity_max_deg_s": (
            float(np.rad2deg(np.max(np.abs(observed_moving_velocity))))
            if observed_moving_velocity.size
            else float("nan")
        ),
        "ramesan_phase_effective_potential_barrier": (
            float(np.ptp(tilted_potential))
            if np.all(np.isfinite(tilted_potential))
            else float("nan")
        ),
        "ramesan_phase_fixed_point_count": float(fixed_theta.size),
        "ramesan_phase_stable_fixed_point_count": float(
            np.count_nonzero(fixed_stability == -1)
        ),
        "ramesan_phase_unstable_fixed_point_count": float(
            np.count_nonzero(fixed_stability == 1)
        ),
        "ramesan_ambient_sample_count": float(ambient_q.size),
        "ramesan_ambient_below_q_threshold_fraction": (
            float(np.mean(ambient_q <= q_threshold))
            if ambient_q.size
            else float("nan")
        ),
    }
    return history, metrics


def analyze_slow_manifold_candidates(
    *,
    dynamics: FrozenAutonomousDynamics,
    candidate_theta: np.ndarray,
    candidate_state: np.ndarray,
    candidate_speed: np.ndarray,
    angular_bin_count: int,
    jacobian_anchor_count: int,
    jacobian_eigenvalue_count: int,
    jacobian_dense_dimension_limit: int,
    minimum_angular_support_fraction: float = 0.5,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fit a slow ring, find phase-flow reversals, and analyze Jacobians."""
    candidate_speed = np.asarray(candidate_speed, dtype=float)
    curve = fit_periodic_state_curve(
        candidate_theta=candidate_theta,
        candidate_state=candidate_state,
        angular_bin_count=angular_bin_count,
    )
    if not 0.0 < minimum_angular_support_fraction <= 1.0:
        raise ValueError("minimum_angular_support_fraction must lie in (0, 1]")
    if float(curve["angular_support_fraction"]) < minimum_angular_support_fraction:
        raise ValueError("slow candidates have insufficient angular support")
    manifold_theta = np.asarray(curve["theta"], dtype=float)
    manifold_state = np.asarray(curve["state"], dtype=float)
    manifold_tangent = np.asarray(curve["tangent"], dtype=float)
    tangent_squared_norm = np.sum(np.square(manifold_tangent), axis=1)
    manifold_flow = np.stack(
        [dynamics.flow(state_vector) for state_vector in manifold_state], axis=0
    )
    angular_flow = np.sum(manifold_tangent * manifold_flow, axis=1) / tangent_squared_norm
    tangent_flow = manifold_tangent * angular_flow[:, None]
    tangent_state_speed = np.sum(
        manifold_flow * manifold_tangent / np.sqrt(tangent_squared_norm)[:, None],
        axis=1,
    )
    normal_flow = manifold_flow - tangent_flow
    normal_flow_norm = np.linalg.norm(normal_flow, axis=1)
    fixed_theta, fixed_slope, fixed_stability = _periodic_scalar_roots(
        theta=manifold_theta,
        value=angular_flow,
    )
    basin = _basin_summary(
        fixed_point_theta=fixed_theta,
        fixed_point_stability=fixed_stability,
    )

    if jacobian_anchor_count < 3:
        raise ValueError("jacobian_anchor_count must be at least three")
    if jacobian_eigenvalue_count < 2:
        raise ValueError("jacobian_eigenvalue_count must be at least two")
    anchor_index = np.unique(
        np.floor(
            np.arange(jacobian_anchor_count) * manifold_theta.size / jacobian_anchor_count
        ).astype(int)
    )
    eigenvalue_real = np.empty((anchor_index.size, jacobian_eigenvalue_count))
    eigenvalue_imag = np.empty_like(eigenvalue_real)
    slow_mode_tangent_alignment = np.empty(anchor_index.size)
    for slot, current_index in enumerate(anchor_index):
        jacobian = dynamics.flow_jacobian(manifold_state[current_index])
        eigenvalues, eigenvectors = _leading_eigensystem(
            jacobian=jacobian,
            eigenvalue_count=jacobian_eigenvalue_count,
            dense_dimension_limit=jacobian_dense_dimension_limit,
        )
        retained_count = eigenvalues.size
        eigenvalue_real[slot] = np.nan
        eigenvalue_imag[slot] = np.nan
        eigenvalue_real[slot, :retained_count] = eigenvalues.real
        eigenvalue_imag[slot, :retained_count] = eigenvalues.imag
        tangent = manifold_tangent[current_index]
        slow_vector = eigenvectors[:, 0]
        slow_mode_tangent_alignment[slot] = float(
            np.abs(np.vdot(tangent, slow_vector))
            / (np.linalg.norm(tangent) * np.linalg.norm(slow_vector))
        )

    leading_real = eigenvalue_real[:, 0]
    second_real = eigenvalue_real[:, 1]
    spectral_gap = leading_real - second_real
    basin_width = np.asarray(basin["basin_width"], dtype=float)
    history = {
        "candidate_theta": np.asarray(wrap_angle(candidate_theta), dtype=float),
        "candidate_state": np.asarray(candidate_state, dtype=float),
        "candidate_speed": candidate_speed,
        "manifold_theta": manifold_theta,
        "manifold_state": manifold_state,
        "manifold_tangent": manifold_tangent,
        "manifold_flow": manifold_flow,
        "angular_flow": angular_flow,
        "tangent_state_speed": tangent_state_speed,
        "normal_flow_norm": normal_flow_norm,
        "angular_bin_sample_count": np.asarray(curve["angular_bin_sample_count"]),
        "fixed_point_theta": fixed_theta,
        "fixed_point_slope": fixed_slope,
        "fixed_point_stability": fixed_stability,
        "basin_stable_theta": np.asarray(basin["basin_stable_theta"]),
        "basin_left_boundary": np.asarray(basin["basin_left_boundary"]),
        "basin_right_boundary": np.asarray(basin["basin_right_boundary"]),
        "basin_width": basin_width,
        "basin_fraction": np.asarray(basin["basin_fraction"]),
        "jacobian_anchor_index": anchor_index,
        "jacobian_anchor_theta": manifold_theta[anchor_index],
        "jacobian_eigenvalue_real": eigenvalue_real,
        "jacobian_eigenvalue_imag": eigenvalue_imag,
        "jacobian_spectral_gap": spectral_gap,
        "slow_mode_tangent_alignment": slow_mode_tangent_alignment,
    }
    metrics = {
        "slow_manifold_enabled": 1.0,
        "slow_manifold_candidate_count": float(candidate_speed.size),
        "slow_manifold_candidate_speed_median": float(np.median(candidate_speed)),
        "slow_manifold_candidate_speed_max": float(np.max(candidate_speed)),
        "slow_manifold_angular_support_fraction": float(
            curve["angular_support_fraction"]
        ),
        "slow_manifold_eta_theta_rad_s": float(np.max(np.abs(angular_flow))),
        "slow_manifold_eta_theta_deg_s": float(
            np.rad2deg(np.max(np.abs(angular_flow)))
        ),
        "slow_manifold_tangent_state_speed_max": float(
            np.max(np.abs(tangent_state_speed))
        ),
        "slow_manifold_tangent_state_speed_max_deg_s": float(
            np.rad2deg(np.max(np.abs(tangent_state_speed)))
        ),
        "slow_manifold_normal_flow_norm_median": float(np.median(normal_flow_norm)),
        "slow_manifold_normal_flow_norm_max": float(np.max(normal_flow_norm)),
        "slow_manifold_fixed_point_count": float(fixed_theta.size),
        "slow_manifold_stable_fixed_point_count": float(
            np.count_nonzero(fixed_stability == -1)
        ),
        "slow_manifold_saddle_fixed_point_count": float(
            np.count_nonzero(fixed_stability == 1)
        ),
        "slow_manifold_basin_entropy": float(basin["basin_entropy"]),
        "slow_manifold_max_basin_width_rad": (
            float(np.max(basin_width)) if basin_width.size else float("nan")
        ),
        "slow_manifold_jacobian_anchor_count": float(anchor_index.size),
        "slow_manifold_leading_real_max": float(np.max(leading_real)),
        "slow_manifold_leading_real_min": float(np.min(leading_real)),
        "slow_manifold_second_real_max": float(np.max(second_real)),
        "slow_manifold_normal_spectral_margin_min": float(-np.max(second_real)),
        "slow_manifold_spectral_gap_min": float(np.min(spectral_gap)),
        "slow_manifold_slow_mode_tangent_alignment_median": float(
            np.median(slow_mode_tangent_alignment)
        ),
        "slow_manifold_slow_mode_tangent_alignment_min": float(
            np.min(slow_mode_tangent_alignment)
        ),
        "slow_manifold_normal_modes_stable_fraction": float(np.mean(second_real < 0.0)),
    }
    return history, metrics


def empty_slow_manifold_result() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    return {
        "candidate_theta": np.empty(0),
        "candidate_state": np.empty((0, 0)),
        "candidate_speed": np.empty(0),
        "manifold_theta": np.empty(0),
        "manifold_state": np.empty((0, 0)),
        "manifold_tangent": np.empty((0, 0)),
        "manifold_flow": np.empty((0, 0)),
        "angular_flow": np.empty(0),
        "fixed_point_theta": np.empty(0),
        "fixed_point_stability": np.empty(0, dtype=np.int8),
        "jacobian_anchor_theta": np.empty(0),
        "jacobian_eigenvalue_real": np.empty((0, 0)),
        "jacobian_eigenvalue_imag": np.empty((0, 0)),
        "angular_bin_sample_count": np.empty(0, dtype=int),
    }, {"slow_manifold_enabled": 0.0}
