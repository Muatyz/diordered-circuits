import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reproduction_config import load_figure3_config
from plot_figure3_abcd import (
    PROCESSED,
    REPRODUCTION_ROOT,
    STABLE_REGULARIZATION,
    build_figure3_matrices,
    existing_data_dir,
)
from utils import (
    optimized_recurrent_velocity_factors,
    overlap_order_parameter,
    simulate_velocity_modulated_rate_network,
    softplus,
    softplus_inverse,
)


INTERIM = existing_data_dir("interim")
FIGURES = REPRODUCTION_ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)
DEFAULT_MATRIX_PATH = PROCESSED / "figure3_abcd_weight_matrices.npz"
DEFAULT_OUTPUT_PATH = PROCESSED / "figure3_jkl_overlap.npz"
FIGURE3_CONFIG = load_figure3_config()
NETWORK_CONFIG = FIGURE3_CONFIG["network"]
JKL_CONFIG = FIGURE3_CONFIG["panels_jkl"]


def circular_error(angle_a, angle_b):
    """
    Return signed shortest angular differences in radians.
    """
    return np.angle(np.exp(1j * (np.asarray(angle_a) - np.asarray(angle_b))))


def interpolate_circular_manifold(manifold, angle_rad):
    """
    Linearly interpolate a periodically sampled manifold at one angle.

    `manifold` is shaped `(n_angles, n_neurons)` and is assumed to cover
    `[0, 2pi)` uniformly. Interpolation across the final bin wraps to bin zero.
    """
    manifold = np.asarray(manifold, dtype=np.float64)
    if manifold.ndim != 2 or len(manifold) < 2:
        raise ValueError("manifold must be shaped (n_angles, n_neurons)")
    coordinate = (float(angle_rad) % (2.0 * np.pi)) / (2.0 * np.pi) * len(manifold)
    left = int(np.floor(coordinate)) % len(manifold)
    fraction = coordinate - np.floor(coordinate)
    right = (left + 1) % len(manifold)
    return (1.0 - fraction) * manifold[left] + fraction * manifold[right]


def load_behavior_window(subject_id, duration_s, display_start_angle_rad, offset_s=0.0):
    """
    Load one contiguous behavior window and derive its angular velocity.

    Head direction is unwrapped before interpolation and differentiation.
    `display_phase_shift_rad` is a circular relabeling used for panels K/L;
    it does not alter the angular velocity supplied to the network.
    """
    path = INTERIM / f"{subject_id}_behavior_wake_square.parquet"
    behavior = pd.read_parquet(path).sort_values("time_s").reset_index(drop=True)
    time_s = behavior["time_s"].to_numpy(float)
    angle_rad = behavior["head_direction_rad"].to_numpy(float)
    if len(time_s) < 2:
        raise ValueError(f"{subject_id} has too few behavior samples")

    start_s = float(time_s[0] + float(offset_s))
    stop_s = start_s + float(duration_s)
    if stop_s > float(time_s[-1]):
        raise ValueError(
            f"requested window ends at {stop_s:.3f}s, after behavior ends at {time_s[-1]:.3f}s"
        )
    initial_angle = float(np.interp(start_s, time_s, np.unwrap(angle_rad)))
    phase_shift = (float(display_start_angle_rad) - initial_angle) % (2.0 * np.pi)
    return {
        "path": path,
        "time_s": time_s,
        "angle_unwrapped_rad": np.unwrap(angle_rad),
        "start_s": start_s,
        "stop_s": stop_s,
        "display_phase_shift_rad": phase_shift,
    }


def angular_velocity_for_steps(behavior_window, dt_s, n_steps, velocity_bin_s=0.1):
    """
    Convert the mouse trajectory to piecewise-constant angular velocity.

    Velocity is computed from unwrapped head direction over 100 ms intervals,
    matching the temporal binning used for the experimental panel L. Each
    interval value is then held across the 1 ms Euler steps inside that bin.
    """
    steps_per_velocity_bin = int(np.round(float(velocity_bin_s) / float(dt_s)))
    if steps_per_velocity_bin <= 0 or not np.isclose(
        steps_per_velocity_bin * float(dt_s),
        float(velocity_bin_s),
    ):
        raise ValueError("velocity_bin_s must be an integer multiple of dt_s")
    if int(n_steps) % steps_per_velocity_bin != 0:
        raise ValueError("simulation duration must contain whole velocity bins")

    relative_time = np.arange(int(n_steps), dtype=float) * float(dt_s)
    n_velocity_bins = int(n_steps) // steps_per_velocity_bin
    velocity_edges = np.arange(n_velocity_bins + 1, dtype=float) * float(velocity_bin_s)
    edge_unwrapped = np.interp(
        behavior_window["start_s"] + velocity_edges,
        behavior_window["time_s"],
        behavior_window["angle_unwrapped_rad"],
    )
    interval_omega = np.diff(edge_unwrapped) / float(velocity_bin_s)
    omega = np.repeat(interval_omega, steps_per_velocity_bin)
    unwrapped = np.interp(relative_time, velocity_edges, edge_unwrapped)
    displayed_angle = np.mod(
        unwrapped + behavior_window["display_phase_shift_rad"],
        2.0 * np.pi,
    )
    return relative_time, omega, displayed_angle


def bin_recorded_spikes(subject_id, unit_ids, start_s, duration_s, bin_s=0.1):
    """
    Count spikes for selected neurons in fixed-width bins.

    The returned activity is shaped `(n_time_bins, n_neurons)` and contains
    counts, not rates, matching the Figure 3L description in the paper.
    """
    n_bins = int(np.round(float(duration_s) / float(bin_s)))
    edges = float(start_s) + np.arange(n_bins + 1, dtype=float) * float(bin_s)
    spikes = np.load(INTERIM / f"{subject_id}_spikes_wake_square.npz")
    counts = np.empty((n_bins, len(unit_ids)), dtype=np.float64)
    for neuron_idx, unit_id in enumerate(unit_ids):
        counts[:, neuron_idx] = np.histogram(
            spikes[f"unit_{int(unit_id)}"],
            bins=edges,
        )[0]
    centers = 0.5 * (edges[:-1] + edges[1:]) - float(start_s)
    return centers, counts


def load_single_mouse_tuning(subject_id):
    """
    Load the single-neuron-normalized tuning curves used for neural projection.
    """
    data = np.load(PROCESSED / f"{subject_id}_hd_tuning_100bins.npz")
    included = data["included_qc"].astype(bool)
    tuning = data["normalized_rate"][included].astype(np.float64)
    unit_ids = data["unit_ids"][included].astype(int)
    angles = data["bin_centers_rad"].astype(np.float64)
    if not np.all(np.isfinite(tuning)):
        raise ValueError(f"{subject_id} included tuning curves contain non-finite values")
    return tuning, unit_ids, angles


def rotate_overlap_rows(overlap, phase_shift_rad):
    """
    Circularly relabel an overlap heatmap's theta axis by a phase shift.
    """
    overlap = np.asarray(overlap)
    shift_bins = int(np.rint(float(phase_shift_rad) / (2.0 * np.pi) * overlap.shape[0]))
    return np.roll(overlap, shift_bins, axis=0), shift_bins


def cache_matches(
    path,
    subject_id,
    duration_s,
    tau_s,
    dt_s,
    record_every_s,
    regularization,
    inhibition_c,
    constant_omega,
    behavior_offset_s,
    display_start_angle_rad,
    velocity_bin_s,
    neural_spike_bin_s,
):
    """
    Check whether a saved J-K-L calculation matches the requested settings.
    """
    path = Path(path)
    if not path.exists():
        return False
    data = np.load(path)
    required = {"overlap_j", "overlap_k", "overlap_l", "omega_mouse"}
    return required.issubset(data.files) and (
        str(data["subject_id"]) == str(subject_id)
        and float(data["duration_s"]) == float(duration_s)
        and float(data["tau_s"]) == float(tau_s)
        and float(data["dt_s"]) == float(dt_s)
        and float(data["record_every_s"]) == float(record_every_s)
        and float(data["regularization"]) == float(regularization)
        and float(data["inhibition_c"]) == float(inhibition_c)
        and float(data["constant_omega"]) == float(constant_omega)
        and float(data["behavior_offset_s"]) == float(behavior_offset_s)
        and float(data["display_start_angle_rad"]) == float(display_start_angle_rad)
        and float(data["velocity_bin_s"]) == float(velocity_bin_s)
        and float(data["neural_spike_bin_s"]) == float(neural_spike_bin_s)
    )


def compute_figure3_jkl(
    matrix_path=DEFAULT_MATRIX_PATH,
    out_path=DEFAULT_OUTPUT_PATH,
    subject_id=str(JKL_CONFIG["subject_id"]),
    duration_s=float(JKL_CONFIG["duration_s"]),
    tau_s=float(NETWORK_CONFIG["tau_s"]),
    dt_s=float(NETWORK_CONFIG["dt_s"]),
    record_every_s=float(JKL_CONFIG["record_every_s"]),
    regularization=STABLE_REGULARIZATION,
    inhibition_c=float(NETWORK_CONFIG["inhibition_c"]),
    constant_omega=float(JKL_CONFIG["constant_omega_rad_s"]),
    behavior_offset_s=float(JKL_CONFIG["behavior_offset_s"]),
    display_start_angle_rad=float(JKL_CONFIG["display_start_angle_pi"]) * np.pi,
    velocity_bin_s=float(JKL_CONFIG["velocity_bin_s"]),
    neural_spike_bin_s=float(JKL_CONFIG["neural_spike_bin_s"]),
    force=False,
):
    """
    Compute Figure 3J-K network overlaps and Figure 3L neural-data overlap.

    Panels J/K use the pooled 1,533-neuron target manifold and Eq. 5
    velocity-modulated weights. Panel L instead uses only the selected mouse's
    measured tuning curves and 100 ms spike counts.
    """
    out_path = Path(out_path)
    if not force and cache_matches(
        out_path,
        subject_id=subject_id,
        duration_s=duration_s,
        tau_s=tau_s,
        dt_s=dt_s,
        record_every_s=record_every_s,
        regularization=regularization,
        inhibition_c=inhibition_c,
        constant_omega=constant_omega,
        behavior_offset_s=behavior_offset_s,
        display_start_angle_rad=display_start_angle_rad,
        velocity_bin_s=velocity_bin_s,
        neural_spike_bin_s=neural_spike_bin_s,
    ):
        print(f"Using cached Figure 3J-L overlaps: {out_path}", flush=True)
        return out_path

    matrix_path = Path(matrix_path)
    if not matrix_path.exists():
        matrix_path = build_figure3_matrices(out_path=matrix_path)
    matrix_data = np.load(matrix_path)
    target_rate = matrix_data["phi_star_safe"].astype(np.float64)
    target_angles = matrix_data["bin_centers_rad"].astype(np.float64)
    activation_beta = float(matrix_data["activation_beta"])
    matrix_regularization = float(matrix_data["regularization"])
    if not np.isclose(matrix_regularization, float(regularization), rtol=0.0, atol=0.0):
        raise ValueError(
            "Figure 3J-K regularization must match Figure 3A-F: "
            f"matrix cache uses {matrix_regularization:g}, requested {float(regularization):g}"
        )
    target_current = softplus_inverse(target_rate, beta=activation_beta).T

    behavior_window = load_behavior_window(
        subject_id,
        duration_s=duration_s,
        display_start_angle_rad=display_start_angle_rad,
        offset_s=behavior_offset_s,
    )
    n_steps = int(np.round(float(duration_s) / float(dt_s)))
    step_time, omega_mouse, displayed_mouse_angle_steps = angular_velocity_for_steps(
        behavior_window,
        dt_s=dt_s,
        n_steps=n_steps,
        velocity_bin_s=velocity_bin_s,
    )
    omega = np.column_stack(
        [
            np.full(n_steps, float(constant_omega)),
            omega_mouse,
        ]
    )

    initial_j = interpolate_circular_manifold(target_current, 0.0)
    initial_k = interpolate_circular_manifold(target_current, display_start_angle_rad)
    initial_states = np.vstack([initial_j, initial_k])
    factors = optimized_recurrent_velocity_factors(
        target_rate,
        tau_s=tau_s,
        regularization=regularization,
        activation_beta=activation_beta,
        enforce_zero_diagonal=True,
    )

    started = time.perf_counter()
    record_time, current_trajectory = simulate_velocity_modulated_rate_network(
        *factors,
        initial_states=initial_states,
        angular_velocity=omega,
        tau_s=tau_s,
        dt_s=dt_s,
        inhibition_c=inhibition_c,
        activation_beta=activation_beta,
        record_every_s=record_every_s,
        progress_label="Figure 3J-K",
    )
    rate_trajectory = softplus(current_trajectory, beta=activation_beta)
    overlap_j = overlap_order_parameter(target_rate, rate_trajectory[:, 0, :])
    overlap_k = overlap_order_parameter(target_rate, rate_trajectory[:, 1, :])

    expected_j_angle = np.mod(float(constant_omega) * record_time, 2.0 * np.pi)
    expected_k_unwrapped = np.interp(
        record_time,
        step_time,
        np.unwrap(displayed_mouse_angle_steps),
    )
    expected_k_angle = np.mod(expected_k_unwrapped, 2.0 * np.pi)
    decoded_j_angle = target_angles[np.argmax(overlap_j, axis=0)]
    decoded_k_angle = target_angles[np.argmax(overlap_k, axis=0)]

    mouse_tuning, mouse_unit_ids, mouse_angles = load_single_mouse_tuning(subject_id)
    neural_time, spike_counts = bin_recorded_spikes(
        subject_id,
        mouse_unit_ids,
        start_s=behavior_window["start_s"],
        duration_s=duration_s,
        bin_s=neural_spike_bin_s,
    )
    overlap_l_raw = overlap_order_parameter(mouse_tuning, spike_counts)
    overlap_l, neural_phase_shift_bins = rotate_overlap_rows(
        overlap_l_raw,
        behavior_window["display_phase_shift_rad"],
    )
    decoded_l_angle = mouse_angles[np.argmax(overlap_l, axis=0)]
    expected_l_unwrapped = np.interp(
        neural_time,
        step_time,
        np.unwrap(displayed_mouse_angle_steps),
    )
    expected_l_angle = np.mod(expected_l_unwrapped, 2.0 * np.pi)

    diagnostics = {
        "subject_id": subject_id,
        "n_network_neurons": int(target_rate.shape[0]),
        "n_mouse_neurons": int(mouse_tuning.shape[0]),
        "behavior_start_s": float(behavior_window["start_s"]),
        "behavior_stop_s": float(behavior_window["stop_s"]),
        "display_phase_shift_rad": float(behavior_window["display_phase_shift_rad"]),
        "omega_mouse_min": float(np.min(omega_mouse)),
        "omega_mouse_max": float(np.max(omega_mouse)),
        "omega_mouse_rms": float(np.sqrt(np.mean(omega_mouse * omega_mouse))),
        "velocity_bin_s": float(velocity_bin_s),
        "neural_spike_bin_s": float(neural_spike_bin_s),
        "panel_j_mean_abs_decode_error_rad": float(
            np.mean(np.abs(circular_error(decoded_j_angle, expected_j_angle)))
        ),
        "panel_k_mean_abs_decode_error_rad": float(
            np.mean(np.abs(circular_error(decoded_k_angle, expected_k_angle)))
        ),
        "panel_l_mean_abs_decode_error_rad": float(
            np.mean(np.abs(circular_error(decoded_l_angle, expected_l_angle)))
        ),
        "elapsed_s": float(time.perf_counter() - started),
    }

    np.savez_compressed(
        out_path,
        overlap_j=overlap_j.astype(np.float32),
        overlap_k=overlap_k.astype(np.float32),
        overlap_l=overlap_l.astype(np.float32),
        record_time_s=record_time,
        neural_time_s=neural_time,
        target_angles_rad=target_angles,
        mouse_angles_rad=mouse_angles,
        omega_mouse=omega_mouse.astype(np.float32),
        omega_step_time_s=step_time,
        expected_j_angle_rad=expected_j_angle,
        expected_k_angle_rad=expected_k_angle,
        expected_l_angle_rad=expected_l_angle,
        decoded_j_angle_rad=decoded_j_angle,
        decoded_k_angle_rad=decoded_k_angle,
        decoded_l_angle_rad=decoded_l_angle,
        spike_counts=spike_counts.astype(np.float32),
        mouse_unit_ids=mouse_unit_ids,
        subject_id=subject_id,
        duration_s=float(duration_s),
        tau_s=float(tau_s),
        dt_s=float(dt_s),
        record_every_s=float(record_every_s),
        regularization=float(regularization),
        inhibition_c=float(inhibition_c),
        activation_beta=float(activation_beta),
        constant_omega=float(constant_omega),
        behavior_offset_s=float(behavior_offset_s),
        behavior_start_s=float(behavior_window["start_s"]),
        display_start_angle_rad=float(display_start_angle_rad),
        velocity_bin_s=float(velocity_bin_s),
        neural_spike_bin_s=float(neural_spike_bin_s),
        display_phase_shift_rad=float(behavior_window["display_phase_shift_rad"]),
        neural_phase_shift_bins=int(neural_phase_shift_bins),
        figure3_config_schema_version=int(FIGURE3_CONFIG["schema_version"]),
        diagnostics_json=json.dumps(diagnostics, indent=2),
    )
    out_path.with_name(out_path.stem + "_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    print(f"Saved Figure 3J-L overlaps: {out_path}", flush=True)
    print(json.dumps(diagnostics, indent=2), flush=True)
    return out_path


def robust_color_limits(overlap):
    """
    Choose stable heatmap limits without changing the overlap calculation.
    """
    overlap = np.asarray(overlap, dtype=float)
    return (
        float(np.nanpercentile(overlap, 1.0)),
        float(np.nanpercentile(overlap, 99.5)),
    )


def plot_figure3_jkl(
    data_path=DEFAULT_OUTPUT_PATH,
    out_path=FIGURES / "figure3_jkl_reproduction.png",
):
    """
    Plot the three overlap heatmaps using the Figure 3 theta/time layout.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        data_path = compute_figure3_jkl(out_path=data_path)
    data = np.load(data_path)
    panels = [
        (
            data["overlap_j"],
            data["record_time_s"],
            r"J  $m(\theta,t)$, network simulation, $\omega=0.8$ rad/s",
        ),
        (
            data["overlap_k"],
            data["record_time_s"],
            r"K  $m(\theta,t)$, network simulation, $\omega(t)$ from A3707",
        ),
        (
            data["overlap_l"],
            data["neural_time_s"],
            r"L  $m(\theta,t)$, A3707 neural-data projection (100 ms counts)",
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.5), constrained_layout=True)
    for ax, (overlap, times, title) in zip(axes, panels):
        vmin, vmax = robust_color_limits(overlap)
        image = ax.imshow(
            overlap,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[float(times[0]), float(times[-1]), 0.0, 2.0 * np.pi],
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        ax.set_ylabel(r"$\theta$")
        ax.set_title(title, loc="left")
        fig.colorbar(image, ax=ax, pad=0.01, label=r"$m(\theta,t)$")
    axes[-1].set_xlabel("Time (s)")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    """
    Command-line entry point for Figure 3J-K-L reproduction.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--duration-s", type=float, default=float(JKL_CONFIG["duration_s"]))
    parser.add_argument("--dt-s", type=float, default=float(NETWORK_CONFIG["dt_s"]))
    parser.add_argument(
        "--record-every-s",
        type=float,
        default=float(JKL_CONFIG["record_every_s"]),
    )
    parser.add_argument(
        "--behavior-offset-s",
        type=float,
        default=float(JKL_CONFIG["behavior_offset_s"]),
    )
    parser.add_argument("--subject-id", default=str(JKL_CONFIG["subject_id"]))
    args = parser.parse_args()

    data_path = compute_figure3_jkl(
        subject_id=args.subject_id,
        duration_s=args.duration_s,
        dt_s=args.dt_s,
        record_every_s=args.record_every_s,
        behavior_offset_s=args.behavior_offset_s,
        force=args.force,
    )
    figure_path = plot_figure3_jkl(data_path)
    print("Saved:", figure_path)


if __name__ == "__main__":
    main()
