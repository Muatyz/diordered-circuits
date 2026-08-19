"""Frozen-weight basin diagnostic for the original Vafidis LearnPI network.

This module is intentionally an external caller of the released implementation:
it loads a trained ``.npz`` file and repeatedly calls ``fly_rec.simulate`` with
``train=False``, ``day=False``, and ``stab=True``.  None of the original model,
dynamics, or plasticity functions are modified.
"""

from __future__ import print_function

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np


DEFAULT_NETWORK_NAME = (
    "fly_rec2Enoughv02inh1rot15NoClipOUsigma225tau05NoBound"
    "x1k1b25s015exc4N60InitNoAnneal05.npz"
)

# The public GIN archive was produced by an earlier revision whose saved
# parameter dictionary predates several keys now read by fly_rec.simulate.
# These are the released code's own defaults, not alternative dynamics.
RELEASE_PARAMETER_DEFAULTS = {
    "M": 4,
    "vary_w_rot": False,
    "adj": False,
    "rand_w_rot": False,
    "filt": True,
    "tau_d": 100,
    "x0": 1,
    "beta": 2.5,
    "gD": 2,
    "gL": 1,
    "fmax": 0.15,
    "eta": 5e-2,
}


def circular_difference_deg(angle_a, angle_b):
    """Return ``angle_a - angle_b`` wrapped to [-180, 180)."""
    return (np.asarray(angle_a) - np.asarray(angle_b) + 180.0) % 360.0 - 180.0


def circular_mean_deg(angle_deg):
    """Return the circular mean in [0, 360), or NaN for empty input."""
    angle_deg = np.asarray(angle_deg, dtype=float)
    if angle_deg.size == 0:
        return float("nan")
    mean_vector = np.mean(np.exp(1j * np.deg2rad(angle_deg)))
    if np.abs(mean_vector) <= 1e-12:
        return float("nan")
    mean_deg = float(np.rad2deg(np.angle(mean_vector)) % 360.0)
    if np.isclose(mean_deg, 360.0, atol=1e-12):
        mean_deg = 0.0
    return mean_deg


def decode_pva_history(firing_rate, preferred_direction_deg):
    """Decode PVA angle and normalized vector strength for an HD history.

    Parameters
    ----------
    firing_rate : ndarray, shape (n_hd, n_time)
        HD rates returned by the original ``fly_rec.simulate`` function.
    preferred_direction_deg : ndarray, shape (n_hd,)
        Original paired HD preferred directions in degrees.
    """
    firing_rate = np.asarray(firing_rate, dtype=float)
    preferred_direction_deg = np.asarray(preferred_direction_deg, dtype=float)
    if firing_rate.ndim != 2:
        raise ValueError("firing_rate must have shape (n_hd, n_time)")
    if preferred_direction_deg.shape != (firing_rate.shape[0],):
        raise ValueError("preferred_direction_deg must have shape (n_hd,)")
    if not np.all(np.isfinite(firing_rate)):
        raise ValueError("firing_rate contains NaN or Inf")

    preferred_vector = np.exp(1j * np.deg2rad(preferred_direction_deg))
    population_vector = np.dot(firing_rate.T, preferred_vector)
    rate_mass = np.sum(firing_rate, axis=0)
    pva_strength = np.divide(
        np.abs(population_vector),
        rate_mass,
        out=np.full(rate_mass.shape, np.nan, dtype=float),
        where=rate_mass > 1e-12,
    )
    pva_angle_deg = np.rad2deg(np.angle(population_vector)) % 360.0
    pva_angle_deg[np.abs(population_vector) <= 1e-12] = np.nan
    return pva_angle_deg, pva_strength


def decode_peak_history(firing_rate):
    """Decode the strongest paired HD angular bin in [0, 360)."""
    firing_rate = np.asarray(firing_rate, dtype=float)
    if firing_rate.ndim != 2 or firing_rate.shape[0] % 2 != 0:
        raise ValueError("paired peak decoding requires even (n_hd, n_time) rates")
    n_position = firing_rate.shape[0] // 2
    collapsed_rate = firing_rate.reshape(n_position, 2, firing_rate.shape[1]).mean(axis=1)
    peak_index = np.argmax(collapsed_rate, axis=0)
    peak_angle_deg = peak_index.astype(float) * 360.0 / n_position
    contrast = np.max(collapsed_rate, axis=0) - np.min(collapsed_rate, axis=0)
    peak_angle_deg[contrast <= 1e-12] = np.nan
    return peak_angle_deg, contrast


def estimate_late_drift_deg_s(theta_deg, time_s, late_fraction=0.25):
    """Fit each unwrapped trajectory over its final time fraction."""
    theta_deg = np.asarray(theta_deg, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if theta_deg.ndim != 2 or time_s.ndim != 1:
        raise ValueError("theta_deg and time_s must have shapes (trial, time) and (time,)")
    if theta_deg.shape[1] != time_s.size or time_s.size < 2:
        raise ValueError("theta_deg must contain at least two samples matching time_s")
    if late_fraction <= 0.0 or late_fraction > 1.0:
        raise ValueError("late_fraction must lie in (0, 1]")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time_s must be strictly increasing")

    late_start = time_s[-1] - late_fraction * (time_s[-1] - time_s[0])
    late_mask = time_s >= late_start
    drift_deg_s = np.full(theta_deg.shape[0], np.nan, dtype=float)
    for trial_index, wrapped_trajectory_deg in enumerate(theta_deg):
        finite_mask = np.isfinite(wrapped_trajectory_deg)
        trajectory_deg = np.full(wrapped_trajectory_deg.shape, np.nan, dtype=float)
        trajectory_deg[finite_mask] = np.rad2deg(
            np.unwrap(np.deg2rad(wrapped_trajectory_deg[finite_mask]))
        )
        fit_mask = late_mask & finite_mask
        if np.count_nonzero(fit_mask) >= 2:
            drift_deg_s[trial_index] = np.polyfit(
                time_s[fit_mask], trajectory_deg[fit_mask], 1
            )[0]
    return drift_deg_s


def _linear_complete_link_clusters(unwrapped_angle, diameter_tolerance_deg):
    clusters = []
    cluster_start = 0
    for index in range(1, unwrapped_angle.size):
        if unwrapped_angle[index] - unwrapped_angle[cluster_start] > diameter_tolerance_deg:
            clusters.append(unwrapped_angle[cluster_start:index])
            cluster_start = index
    clusters.append(unwrapped_angle[cluster_start:])
    return clusters


def cluster_circular_endpoints(endpoint_deg, diameter_tolerance_deg=5.0):
    """Group nearby circular endpoints without chain-merging a continuous ring.

    ``diameter_tolerance_deg`` is the maximum angular diameter of one candidate
    basin, not a nearest-neighbour linkage distance.  Trying every possible
    circular cut prevents a basin around 0/360 degrees from being split.
    """
    endpoint_deg = np.asarray(endpoint_deg, dtype=float)
    endpoint_deg = endpoint_deg[np.isfinite(endpoint_deg)] % 360.0
    if diameter_tolerance_deg <= 0.0 or diameter_tolerance_deg >= 180.0:
        raise ValueError("diameter_tolerance_deg must lie in (0, 180)")
    if endpoint_deg.size == 0:
        return {
            "center_deg": np.empty(0, dtype=float),
            "occupancy": np.empty(0, dtype=int),
            "label": np.empty(0, dtype=int),
        }

    sorted_angle = np.sort(endpoint_deg)
    best_clusters = None
    best_score = None
    for cut_index in range(sorted_angle.size):
        rotated = np.concatenate(
            [
                sorted_angle[cut_index:],
                sorted_angle[:cut_index] + 360.0,
            ]
        )
        clusters = _linear_complete_link_clusters(rotated, diameter_tolerance_deg)
        squared_error = 0.0
        for cluster in clusters:
            center = circular_mean_deg(cluster)
            squared_error += float(
                np.sum(np.square(circular_difference_deg(cluster, center)))
            )
        score = (len(clusters), squared_error)
        if best_score is None or score < best_score:
            best_score = score
            best_clusters = clusters

    center_deg = np.asarray(
        [circular_mean_deg(cluster) for cluster in best_clusters],
        dtype=float,
    )
    occupancy = np.asarray([cluster.size for cluster in best_clusters], dtype=int)
    order = np.argsort(center_deg)
    center_deg = center_deg[order]
    occupancy = occupancy[order]

    distance = np.abs(
        circular_difference_deg(endpoint_deg[:, np.newaxis], center_deg[np.newaxis, :])
    )
    label = np.argmin(distance, axis=1).astype(int)
    return {
        "center_deg": center_deg,
        "occupancy": occupancy,
        "label": label,
    }


def normalize_release_params(stored_params):
    """Fill keys missing from old public archives with release-code defaults."""
    params = dict(RELEASE_PARAMETER_DEFAULTS)
    params.update(dict(stored_params))
    if "M" not in stored_params and "A" in stored_params:
        params["M"] = stored_params["A"]
    required = [
        "dt",
        "n_neu",
        "v0",
        "v_max",
        "M",
        "sigma",
        "inh",
        "inh_rot",
        "every_perc",
        "avg_err",
        "n_sigma",
        "exc",
        "tau_s",
        "gain",
    ]
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError("network params are missing required keys: {}".format(missing))
    return params


def _load_trained_network(network_path):
    network_path = Path(network_path).expanduser().resolve()
    if not network_path.is_file():
        raise FileNotFoundError("trained network not found: {}".format(network_path))
    with np.load(str(network_path), allow_pickle=True) as network:
        if "w" not in network or "params" not in network:
            raise ValueError("network archive must contain 'w' and 'params'")
        weight_history = np.asarray(network["w"], dtype=float)
        if weight_history.ndim == 3:
            weight = weight_history[:, :, -1].copy()
        elif weight_history.ndim == 2:
            weight = weight_history.copy()
        else:
            raise ValueError("network['w'] must be a 2D weight or 3D weight history")
        params_object = network["params"]
        if params_object.shape == ():
            params = dict(params_object.item())
        else:
            params = dict(params_object.reshape(-1)[0])
        weight_rot = None
        if "w_rot" in network:
            candidate = np.asarray(network["w_rot"], dtype=float)
            if candidate.size:
                weight_rot = candidate.copy()

    params = normalize_release_params(params)
    n_hd = int(params["n_neu"])
    if n_hd <= 0 or n_hd % 2 != 0:
        raise ValueError("the original paired-HD geometry requires even params['n_neu']")
    if weight.shape != (n_hd, 2 * n_hd):
        raise ValueError("trained w must have shape (n_neu, 2*n_neu)")
    if weight_rot is not None and weight_rot.shape != (n_hd, n_hd):
        raise ValueError("w_rot must have shape (n_neu, n_neu)")
    return network_path, params, weight, weight_rot


def _sample_indices(n_time, dt, sample_interval):
    if sample_interval <= 0.0:
        raise ValueError("sample_interval must be positive")
    stride = max(1, int(round(sample_interval / dt)))
    # In the released simulate(), f[:, 0] is written before its explicit bump
    # initializer replaces f_old.  The first returned bump sample is f[:, 1].
    indices = np.arange(1, n_time, stride, dtype=int)
    if indices[-1] != n_time - 1:
        indices = np.append(indices, n_time - 1)
    return indices


def run_uniform_bump_basin_test(
    network_path,
    initial_condition_count=36,
    duration=30.0,
    sample_interval=0.25,
    basin_tolerance_deg=None,
    minimum_pva_strength=0.05,
    test_noise_std=0.0,
    random_seed=0,
    show_original_progress=False,
):
    """Run uniformly initialized, frozen-weight, zero-input bump trajectories."""
    if initial_condition_count < 2:
        raise ValueError("initial_condition_count must be at least two")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if minimum_pva_strength < 0.0 or minimum_pva_strength > 1.0:
        raise ValueError("minimum_pva_strength must lie in [0, 1]")
    if test_noise_std < 0.0:
        raise ValueError("test_noise_std must be non-negative")

    network_path, stored_params, weight, weight_rot = _load_trained_network(network_path)
    params = dict(stored_params)
    dt = float(params["dt"])
    # At least two post-initializer samples are needed for the late-drift fit.
    n_time = max(3, int(round(duration / dt)))
    actual_duration = n_time * dt
    params["t_run"] = actual_duration
    params["n_sigma"] = float(test_noise_std)
    # With darkness and stab=True, gain only rescales theta0 before the released
    # bump initializer.  Fixing it to one makes requested initial headings
    # uniform even when testing a network trained for a non-unit velocity gain.
    params["gain"] = 1.0
    # This only reduces the released function's progress/history bookkeeping;
    # it does not enter the network dynamics when train=False and store_f=True.
    params["every_perc"] = 100

    sample_index = _sample_indices(n_time, dt, sample_interval)
    time = sample_index.astype(float) * dt
    initial_spacing_deg = 360.0 / int(initial_condition_count)
    # Use circular-bin midpoints.  The released rectangular initializer has a
    # special slicing edge case at exactly theta0 == 0; a half-spacing phase
    # shift preserves uniform 360-degree coverage without changing that code.
    theta_initial_deg = (
        np.arange(int(initial_condition_count), dtype=float) + 0.5
    ) * initial_spacing_deg
    if basin_tolerance_deg is None:
        # Keep a continuous endpoint map close to one candidate per start even
        # when the user increases the angular sampling resolution.
        basin_tolerance_deg = 0.45 * initial_spacing_deg
    if basin_tolerance_deg <= 0.0 or basin_tolerance_deg >= 180.0:
        raise ValueError("basin_tolerance_deg must lie in (0, 180)")
    theta_pva_deg = np.empty((theta_initial_deg.size, time.size), dtype=float)
    theta_peak_deg = np.empty_like(theta_pva_deg)
    pva_strength = np.empty_like(theta_pva_deg)
    bump_contrast = np.empty_like(theta_pva_deg)
    final_firing_rate = np.empty((theta_initial_deg.size, int(params["n_neu"])), dtype=float)

    n_position = int(params["n_neu"]) // 2
    preferred_direction_deg = np.repeat(
        np.arange(n_position, dtype=float) * 360.0 / n_position,
        2,
    )

    import fly_rec as rec

    np.random.seed(int(random_seed))
    for trial_index, initial_heading_deg in enumerate(theta_initial_deg):
        print(
            "Basin trial {}/{}: initial bump {:.1f} deg".format(
                trial_index + 1,
                theta_initial_deg.size,
                initial_heading_deg,
            )
        )
        # fly_rec uses landmark angle theta0 and plots heading as 360-theta0.
        landmark_angle_deg = (-initial_heading_deg) % 360.0
        theta0 = np.full(n_time, landmark_angle_deg, dtype=float)
        original_output = contextlib.nullcontext()
        if not show_original_progress:
            original_output = contextlib.redirect_stdout(io.StringIO())
        with original_output:
            _, simulated_weight_rot, firing_rate, _, _ = rec.simulate(
                actual_duration,
                theta0,
                params,
                store_f=True,
                train=False,
                w=weight,
                w_rot=weight_rot,
                day=False,
                stab=True,
            )
        # Some public archives predate saving w_rot.  Reuse the first trial's
        # released-code reconstruction so every start probes the same network.
        if weight_rot is None:
            weight_rot = np.asarray(simulated_weight_rot, dtype=float).copy()
        if firing_rate.shape != (int(params["n_neu"]), n_time):
            raise RuntimeError(
                "unexpected firing-rate shape {}; expected ({}, {})".format(
                    firing_rate.shape,
                    int(params["n_neu"]),
                    n_time,
                )
            )
        sampled_rate = firing_rate[:, sample_index]
        pva_angle, strength = decode_pva_history(
            sampled_rate,
            preferred_direction_deg,
        )
        peak_angle, contrast = decode_peak_history(sampled_rate)
        theta_pva_deg[trial_index] = pva_angle
        theta_peak_deg[trial_index] = peak_angle
        pva_strength[trial_index] = strength
        bump_contrast[trial_index] = contrast
        final_firing_rate[trial_index] = firing_rate[:, -1]

    final_pva_deg = theta_pva_deg[:, -1]
    late_drift_deg_s = estimate_late_drift_deg_s(theta_pva_deg, time)
    endpoint_valid = np.isfinite(final_pva_deg) & (
        pva_strength[:, -1] >= minimum_pva_strength
    )
    basin = cluster_circular_endpoints(
        final_pva_deg[endpoint_valid],
        diameter_tolerance_deg=basin_tolerance_deg,
    )
    occupancy = basin["occupancy"]
    if occupancy.size:
        occupancy_probability = occupancy.astype(float) / np.sum(occupancy)
        effective_basin_count = float(1.0 / np.sum(np.square(occupancy_probability)))
    else:
        effective_basin_count = 0.0

    final_error_deg = circular_difference_deg(final_pva_deg, theta_initial_deg)
    final_peak_error_deg = circular_difference_deg(
        theta_peak_deg[:, -1],
        theta_initial_deg,
    )
    valid_count = int(np.count_nonzero(endpoint_valid))
    valid_late_drift = np.abs(late_drift_deg_s[endpoint_valid])
    valid_late_drift = valid_late_drift[np.isfinite(valid_late_drift)]
    summary = {
        "network": str(network_path),
        "initialization": "direct released-code bump; no visual cue",
        "frozen_weights": True,
        "visual_input": False,
        "velocity_input": False,
        "initial_condition_count": int(theta_initial_deg.size),
        "duration_s": float(time[-1]),
        "sample_interval_s": float(sample_interval),
        "test_noise_std": float(test_noise_std),
        "minimum_pva_strength": float(minimum_pva_strength),
        "valid_endpoint_count": valid_count,
        "invalid_endpoint_count": int(theta_initial_deg.size - valid_count),
        "candidate_basin_count": int(basin["center_deg"].size),
        "candidate_basin_center_deg": basin["center_deg"].tolist(),
        "candidate_basin_occupancy": occupancy.tolist(),
        "effective_basin_count": effective_basin_count,
        "endpoint_compression_ratio": (
            float(basin["center_deg"].size / valid_count) if valid_count else None
        ),
        "basin_maximum_diameter_deg": float(basin_tolerance_deg),
        "final_pva_error_rms_deg": (
            float(np.sqrt(np.mean(np.square(final_error_deg[endpoint_valid]))))
            if valid_count
            else None
        ),
        "final_peak_error_rms_deg": (
            float(
                np.sqrt(
                    np.mean(
                        np.square(final_peak_error_deg[np.isfinite(final_peak_error_deg)])
                    )
                )
            )
            if np.any(np.isfinite(final_peak_error_deg))
            else None
        ),
        "final_pva_strength_median": float(np.nanmedian(pva_strength[:, -1])),
        "final_pva_strength_min": float(np.nanmin(pva_strength[:, -1])),
        "final_bump_contrast_median_khz": float(np.nanmedian(bump_contrast[:, -1])),
        "late_abs_pva_drift_median_deg_s": (
            float(np.median(valid_late_drift)) if valid_late_drift.size else None
        ),
        "late_abs_pva_drift_p95_deg_s": (
            float(np.percentile(valid_late_drift, 95.0))
            if valid_late_drift.size
            else None
        ),
    }
    history = {
        "time": time,
        "theta_initial_deg": theta_initial_deg,
        "theta_pva_deg": theta_pva_deg,
        "theta_peak_deg": theta_peak_deg,
        "pva_strength": pva_strength,
        "bump_contrast": bump_contrast,
        "final_firing_rate": final_firing_rate,
        "endpoint_valid": endpoint_valid,
        "late_pva_drift_deg_s": late_drift_deg_s,
        "basin_center_deg": basin["center_deg"],
        "basin_occupancy": occupancy,
    }
    return history, summary


def plot_basin_test(history, summary, output_path):
    """Write trajectory, endpoint, occupancy, and bump-quality diagnostics."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(history["time"], dtype=float)
    theta_initial_deg = np.asarray(history["theta_initial_deg"], dtype=float)
    theta_pva_deg = np.asarray(history["theta_pva_deg"], dtype=float)
    final_pva_deg = theta_pva_deg[:, -1]
    endpoint_valid = np.asarray(history["endpoint_valid"], dtype=bool)
    basin_center_deg = np.asarray(history["basin_center_deg"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    color = plt.cm.hsv(theta_initial_deg / 360.0)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(theta_pva_deg), axis=1))
    initial_decoded = theta_initial_deg + circular_difference_deg(
        theta_pva_deg[:, 0], theta_initial_deg
    )
    aligned_trajectory = initial_decoded[:, np.newaxis] + unwrapped - unwrapped[:, :1]
    for trial_index in range(theta_initial_deg.size):
        axes[0, 0].plot(
            time,
            aligned_trajectory[trial_index],
            color=color[trial_index],
            linewidth=0.8,
            alpha=0.7,
        )
    axes[0, 0].set_title("Frozen-weight zero-input PVA trajectories")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("unwrapped decoded angle [deg]")
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].plot(
        [0.0, 360.0],
        [0.0, 360.0],
        linestyle="--",
        color="black",
        linewidth=0.8,
        alpha=0.55,
        label="ideal continuous ring",
    )
    axes[0, 1].scatter(
        theta_initial_deg[endpoint_valid],
        final_pva_deg[endpoint_valid],
        c=color[endpoint_valid],
        s=24.0,
        edgecolors="none",
    )
    if np.any(~endpoint_valid):
        axes[0, 1].scatter(
            theta_initial_deg[~endpoint_valid],
            np.zeros(np.count_nonzero(~endpoint_valid)),
            marker="x",
            color="black",
            label="invalid / weak final PVA",
        )
    for center_deg in basin_center_deg:
        axes[0, 1].axhline(center_deg, color="#9c4f45", alpha=0.25, linewidth=0.8)
    axes[0, 1].set_xlim(0.0, 360.0)
    axes[0, 1].set_ylim(0.0, 360.0)
    axes[0, 1].set_title("Endpoint map (horizontal bands = pinned basins)")
    axes[0, 1].set_xlabel("requested initial bump [deg]")
    axes[0, 1].set_ylabel("final PVA [deg]")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(frameon=False, fontsize=8)

    histogram_bins = np.linspace(0.0, 360.0, 73)
    axes[1, 0].hist(
        final_pva_deg[endpoint_valid],
        bins=histogram_bins,
        color="#365f8d",
        alpha=0.75,
    )
    for center_deg in basin_center_deg:
        axes[1, 0].axvline(center_deg, color="#9c4f45", linewidth=1.2)
    axes[1, 0].set_xlim(0.0, 360.0)
    axes[1, 0].set_title(
        "Final endpoint occupancy: {} candidate basins".format(
            summary["candidate_basin_count"]
        )
    )
    axes[1, 0].set_xlabel("final PVA [deg]")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].grid(alpha=0.2)

    final_strength = np.asarray(history["pva_strength"], dtype=float)[:, -1]
    axes[1, 1].plot(
        theta_initial_deg,
        final_strength,
        marker="o",
        markersize=3.0,
        linewidth=0.8,
        color="#427a5b",
    )
    axes[1, 1].axhline(
        summary["minimum_pva_strength"],
        linestyle="--",
        color="black",
        linewidth=0.8,
        label="validity threshold",
    )
    axes[1, 1].set_xlim(0.0, 360.0)
    axes[1, 1].set_ylim(bottom=0.0)
    axes[1, 1].set_title("Final bump confidence")
    axes[1, 1].set_xlabel("requested initial bump [deg]")
    axes[1, 1].set_ylabel("normalized PVA strength")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Original LearnPI basin diagnostic: n={}, T={:g} s, valid={}/{}".format(
            summary["initial_condition_count"],
            summary["duration_s"],
            summary["valid_endpoint_count"],
            summary["initial_condition_count"],
        )
    )
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def save_basin_test(history, summary, output_dir):
    """Save the reproducible numeric history, JSON summary, and main figure."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "bump_basin_history.npz"
    summary_path = output_dir / "bump_basin_summary.json"
    figure_path = output_dir / "bump_basin_diagnostics.png"
    np.savez(str(history_path), **history)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)
        summary_file.write("\n")
    plot_basin_test(history, summary, figure_path)
    return history_path, summary_path, figure_path


def build_argument_parser():
    script_dir = Path(__file__).resolve().parent
    default_network = script_dir.parent / "savefiles" / "trained_networks" / DEFAULT_NETWORK_NAME
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether uniformly initialized bumps in the original LearnPI "
            "network converge to a small set of discrete zero-input basins."
        )
    )
    parser.add_argument("--network", type=Path, default=default_network)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--initial-conditions", type=int, default=36)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument(
        "--basin-tolerance-deg",
        type=float,
        default=None,
        help=(
            "maximum endpoint diameter of one candidate basin; default is "
            "0.45 times the initial angular spacing"
        ),
    )
    parser.add_argument("--minimum-pva-strength", type=float, default=0.05)
    parser.add_argument("--test-noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--show-original-progress",
        action="store_true",
        help="show fly_rec.simulate's per-trial percentage output",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    network_path = args.network.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent
            / "savefiles"
            / "basin_tests"
            / network_path.stem
        )
    history, summary = run_uniform_bump_basin_test(
        network_path=network_path,
        initial_condition_count=args.initial_conditions,
        duration=args.duration,
        sample_interval=args.sample_interval,
        basin_tolerance_deg=args.basin_tolerance_deg,
        minimum_pva_strength=args.minimum_pva_strength,
        test_noise_std=args.test_noise_std,
        random_seed=args.seed,
        show_original_progress=args.show_original_progress,
    )
    history_path, summary_path, figure_path = save_basin_test(
        history,
        summary,
        output_dir,
    )
    print("Candidate basins: {}".format(summary["candidate_basin_count"]))
    print("Basin centers [deg]: {}".format(summary["candidate_basin_center_deg"]))
    print("Basin occupancy: {}".format(summary["candidate_basin_occupancy"]))
    print("Valid endpoints: {}/{}".format(
        summary["valid_endpoint_count"], summary["initial_condition_count"]
    ))
    print(
        "Late absolute PVA drift median [deg/s]: {}".format(
            summary["late_abs_pva_drift_median_deg_s"]
        )
    )
    print("Saved history: {}".format(history_path))
    print("Saved summary: {}".format(summary_path))
    print("Saved figure: {}".format(figure_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
