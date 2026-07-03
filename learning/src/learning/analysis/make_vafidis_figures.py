"""Generate standard figures for a saved Vafidis toy-model run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from learning.analysis.metrics import circular_error_trace
from learning.analysis.weights import compute_weight_eigenvalues, summarize_eigenvalue_pair_degeneracy
from learning.io.save_load import load_npz, save_json, save_npz
from learning.plotting.activity import plot_activity_heatmap, plot_activity_tuning_slices
from learning.plotting.heading import (
    plot_decoded_vs_true_heading_panels,
    plot_pi_error,
    plot_true_vs_decoded_heading,
    plot_velocity_gain_curve,
)
from learning.plotting.weights import plot_weight_eigen_spectrum, plot_weight_matrices_side_by_side, plot_weight_matrix


def _history_time_window(history, *, max_duration: float) -> dict[str, np.ndarray]:
    time = history.get("time", np.empty(0))
    history_keys = history.files if hasattr(history, "files") else history.keys()
    if time.size == 0 or float(time[-1] - time[0]) <= max_duration:
        return {key: history[key] for key in history_keys}
    mask = time <= float(time[0] + max_duration)
    windowed_history: dict[str, np.ndarray] = {}
    for key in history_keys:
        value = history[key]
        if value.shape[:1] == time.shape:
            windowed_history[key] = value[mask]
        else:
            windowed_history[key] = value
    return windowed_history


def _history_movement_window(
    history,
    *,
    duration: float,
    angular_velocity_threshold: float = 1e-6,
) -> dict[str, np.ndarray]:
    time = history.get("time", np.empty(0))
    history_keys = history.files if hasattr(history, "files") else history.keys()
    if time.size == 0:
        return {key: history[key] for key in history_keys}
    theta_decoded = history.get("theta_hd_decoded", np.empty(0))
    start_time: float | None = None
    if theta_decoded.size == time.size and time.size >= 3:
        median_dt = float(np.median(np.diff(time)))
        window_size = max(2, int(round(duration / median_dt))) if median_dt > 0.0 else 2
        if window_size < time.size:
            theta_unwrapped = np.unwrap(theta_decoded)
            window_ranges = np.array(
                [
                    np.nanmax(theta_unwrapped[start_index : start_index + window_size])
                    - np.nanmin(theta_unwrapped[start_index : start_index + window_size])
                    for start_index in range(time.size - window_size + 1)
                ]
            )
            if np.any(np.isfinite(window_ranges)):
                start_time = float(time[int(np.nanargmax(window_ranges))])
    if start_time is None:
        angular_velocity = history.get("angular_velocity", np.zeros_like(time))
        moving_indices = np.flatnonzero(np.abs(angular_velocity) > angular_velocity_threshold)
        start_time = float(time[moving_indices[0]]) if moving_indices.size else float(time[0])
    end_time = start_time + duration
    mask = (time >= start_time) & (time <= end_time)
    if np.count_nonzero(mask) < 2:
        mask = time <= float(time[0] + duration)
    windowed_history: dict[str, np.ndarray] = {}
    for key in history_keys:
        value = history[key]
        if value.shape[:1] == time.shape:
            windowed_history[key] = value[mask]
        else:
            windowed_history[key] = value
    return windowed_history


def make_vafidis_figures_for_run(*, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    trained_weights = load_npz(run_dir / "trained_weights.npz")
    training_history = load_npz(run_dir / "training_history.npz")
    bump_history = load_npz(run_dir / "bump_history.npz")
    darkness_history = load_npz(run_dir / "darkness_history.npz")
    ou_darkness_history = (
        load_npz(run_dir / "ou_darkness_history.npz")
        if (run_dir / "ou_darkness_history.npz").exists()
        else None
    )
    velocity_gain_history = load_npz(run_dir / "velocity_gain_history.npz")
    theta_hd_pref = trained_weights.get("theta_hd_pref", np.empty(0))
    w_hd_to_hd = trained_weights["w_hd_to_hd"]
    w_hr_to_hd = trained_weights["w_hr_to_hd"]
    hd_to_hd_eigenvalues = compute_weight_eigenvalues(w_hd_to_hd)
    hr_to_hd_eigenvalues = compute_weight_eigenvalues(w_hr_to_hd)
    spectrum_diagnostics = {
        "hd_to_hd": summarize_eigenvalue_pair_degeneracy(weight_matrix=w_hd_to_hd),
        "hr_to_hd": summarize_eigenvalue_pair_degeneracy(weight_matrix=w_hr_to_hd),
    }
    save_npz(
        run_dir / "weight_eigenvalues.npz",
        hd_to_hd=hd_to_hd_eigenvalues,
        hr_to_hd=hr_to_hd_eigenvalues,
    )
    save_json(run_dir / "weight_spectrum_diagnostics.json", spectrum_diagnostics)
    training_activity_history = _history_time_window(training_history, max_duration=120.0)
    training_movement_history = _history_movement_window(training_history, duration=30.0)
    training_history_truncated = (
        training_activity_history.get("time", np.empty(0)).size
        != training_history.get("time", np.empty(0)).size
    )

    plot_activity_heatmap(
        r_hd_history=training_activity_history.get("r_hd", np.empty((0, 0))),
        time=training_activity_history.get("time", np.empty(0)),
        path=figures_dir / "training_hd_activity_heatmap.png",
        title="Training HD activity (first 120 s)" if training_history_truncated else "Training HD activity",
        theta_hd_pref=theta_hd_pref,
        theta_hd_decoded=training_activity_history.get("theta_hd_decoded", None),
        theta_hd_decoded_peak=training_activity_history.get("theta_hd_decoded_peak", None),
    )
    plot_decoded_vs_true_heading_panels(
        time=training_movement_history.get("time", np.empty(0)),
        theta_true=training_movement_history.get("theta_true", np.empty(0)),
        theta_hd_decoded=training_movement_history.get("theta_hd_decoded", np.empty(0)),
        theta_hd_decoded_peak=training_movement_history.get("theta_hd_decoded_peak", None),
        path=figures_dir / "training_heading_short_window.png",
        title="Training heading comparison (30 s movement window)",
    )
    if training_history_truncated:
        plot_activity_heatmap(
            r_hd_history=training_history.get("r_hd", np.empty((0, 0))),
            time=training_history.get("time", np.empty(0)),
            path=figures_dir / "training_hd_activity_heatmap_full.png",
            title="Training HD activity (full)",
            theta_hd_pref=theta_hd_pref,
        )
    plot_activity_tuning_slices(
        r_hd_history=training_history.get("r_hd", np.empty((0, 0))),
        time=training_history.get("time", np.empty(0)),
        theta_hd_pref=theta_hd_pref,
        path=figures_dir / "training_hd_activity_slices.png",
        title="Training HD tuning slices",
    )
    plot_activity_heatmap(
        r_hd_history=bump_history.get("r_hd", np.empty((0, 0))),
        time=bump_history.get("time", np.empty(0)),
        path=figures_dir / "bump_maintenance_hd_activity_heatmap.png",
        title="Bump maintenance HD activity (cue, then visual off, omega=0)",
        theta_hd_pref=theta_hd_pref,
        theta_hd_decoded=bump_history.get("theta_hd_decoded", None),
        theta_hd_decoded_peak=bump_history.get("theta_hd_decoded_peak", None),
    )
    plot_activity_tuning_slices(
        r_hd_history=bump_history.get("r_hd", np.empty((0, 0))),
        time=bump_history.get("time", np.empty(0)),
        theta_hd_pref=theta_hd_pref,
        path=figures_dir / "bump_maintenance_hd_activity_slices.png",
        title="Bump maintenance HD tuning slices",
    )
    plot_activity_heatmap(
        r_hd_history=darkness_history.get("r_hd", np.empty((0, 0))),
        time=darkness_history.get("time", np.empty(0)),
        path=figures_dir / "darkness_hd_activity_heatmap.png",
        title="Constant-velocity PI HD activity (visual, dark, visual)",
        theta_hd_pref=theta_hd_pref,
        theta_hd_decoded=darkness_history.get("theta_hd_decoded", None),
        theta_hd_decoded_peak=darkness_history.get("theta_hd_decoded_peak", None),
    )
    plot_activity_tuning_slices(
        r_hd_history=darkness_history.get("r_hd", np.empty((0, 0))),
        time=darkness_history.get("time", np.empty(0)),
        theta_hd_pref=theta_hd_pref,
        path=figures_dir / "darkness_hd_activity_slices.png",
        title="Constant-velocity PI HD tuning slices",
    )
    if ou_darkness_history is not None:
        plot_activity_heatmap(
            r_hd_history=ou_darkness_history.get("r_hd", np.empty((0, 0))),
            time=ou_darkness_history.get("time", np.empty(0)),
            path=figures_dir / "ou_darkness_hd_activity_heatmap.png",
            title="OU PI HD activity (visual, dark, visual)",
            theta_hd_pref=theta_hd_pref,
            theta_hd_decoded=ou_darkness_history.get("theta_hd_decoded", None),
            theta_hd_decoded_peak=ou_darkness_history.get("theta_hd_decoded_peak", None),
        )
    plot_weight_matrix(
        weight_matrix=w_hd_to_hd,
        path=figures_dir / "training_weight_hd_to_hd.png",
        title="Trained HD-to-HD",
        cmap="coolwarm",
        x_label="source HD preferred direction theta_source [rad]",
        y_label="target HD preferred direction theta_target [rad]",
        extent=(-np.pi, np.pi, -np.pi, np.pi),
    )
    plot_weight_matrix(
        weight_matrix=w_hr_to_hd,
        path=figures_dir / "training_weight_hr_to_hd.png",
        title="Trained HR-to-HD",
        cmap="coolwarm",
        x_label="source HR cell index [unitless]",
        y_label="target HD preferred direction (theta_target) [deg]",
        extent=(0.0, float(w_hr_to_hd.shape[1]), 0.0, 360.0),
    )
    plot_weight_matrices_side_by_side(
        w_hd_to_hd=w_hd_to_hd,
        w_hr_to_hd=w_hr_to_hd,
        path=figures_dir / "training_weight_matrices_side_by_side.png",
        title="Trained HD-to-HD and HR-to-HD weights",
    )
    plot_weight_eigen_spectrum(
        hd_to_hd_eigenvalues=hd_to_hd_eigenvalues,
        hr_to_hd_eigenvalues=hr_to_hd_eigenvalues,
        path=figures_dir / "training_weight_eigen_spectrum.png",
        title="Weight eigenvalue spectrum",
        diagnostics=spectrum_diagnostics,
    )
    plot_true_vs_decoded_heading(
        time=bump_history["time"],
        theta_true=bump_history["theta_true"],
        theta_hd_decoded=bump_history["theta_hd_decoded"],
        path=figures_dir / "bump_maintenance_decoded_heading.png",
        title="Bump maintenance (cue, then visual off, omega=0)",
        theta_hd_decoded_peak=bump_history.get("theta_hd_decoded_peak", None),
    )
    plot_true_vs_decoded_heading(
        time=darkness_history["time"],
        theta_true=darkness_history["theta_true"],
        theta_hd_decoded=darkness_history["theta_hd_decoded"],
        path=figures_dir / "darkness_true_vs_decoded_heading.png",
        title="Constant-velocity path integration (visual, dark, visual)",
        theta_hd_decoded_peak=darkness_history.get("theta_hd_decoded_peak", None),
        phase_id=darkness_history.get("phase_id", None),
        angle_unit="pi",
    )
    darkness_pi_error = circular_error_trace(
        darkness_history["theta_hd_decoded"],
        darkness_history["theta_true"],
    )
    plot_pi_error(
        time=darkness_history["time"],
        pi_error=darkness_pi_error,
        path=figures_dir / "darkness_pi_error.png",
        title="Constant-velocity PI error (visual, dark, visual)",
        phase_id=darkness_history.get("phase_id", None),
    )
    if ou_darkness_history is not None:
        plot_true_vs_decoded_heading(
            time=ou_darkness_history["time"],
            theta_true=ou_darkness_history["theta_true"],
            theta_hd_decoded=ou_darkness_history["theta_hd_decoded"],
            path=figures_dir / "ou_darkness_true_vs_decoded_heading.png",
            title="OU path integration (visual, dark, visual)",
            theta_hd_decoded_peak=ou_darkness_history.get("theta_hd_decoded_peak", None),
            phase_id=ou_darkness_history.get("phase_id", None),
            angle_unit="pi",
        )
        ou_darkness_pi_error = circular_error_trace(
            ou_darkness_history["theta_hd_decoded"],
            ou_darkness_history["theta_true"],
        )
        plot_pi_error(
            time=ou_darkness_history["time"],
            pi_error=ou_darkness_pi_error,
            path=figures_dir / "ou_darkness_pi_error.png",
            title="OU PI error (visual, dark, visual)",
            phase_id=ou_darkness_history.get("phase_id", None),
        )
    plot_velocity_gain_curve(
        commanded_velocity=velocity_gain_history["commanded_velocity"],
        decoded_velocity=velocity_gain_history["decoded_velocity"],
        path=figures_dir / "velocity_gain_curve.png",
        title="Velocity gain curve",
        decoded_velocity_peak=velocity_gain_history.get("decoded_velocity_peak", None),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Saved run directory.")
    args = parser.parse_args()
    make_vafidis_figures_for_run(run_dir=args.run_dir)
    print(f"Saved figures to {Path(args.run_dir) / 'figures'}")


if __name__ == "__main__":
    main()
