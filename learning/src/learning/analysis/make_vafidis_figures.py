"""Generate standard figures for a saved Vafidis toy-model run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from learning.analysis.metrics import circular_error_trace
from learning.io.save_load import load_npz
from learning.plotting.activity import plot_activity_heatmap, plot_activity_tuning_slices
from learning.plotting.heading import (
    plot_pi_error,
    plot_true_vs_decoded_heading,
    plot_velocity_gain_curve,
)
from learning.plotting.weights import plot_weight_matrix


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


def make_vafidis_figures_for_run(*, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    trained_weights = load_npz(run_dir / "trained_weights.npz")
    training_history = load_npz(run_dir / "training_history.npz")
    bump_history = load_npz(run_dir / "bump_history.npz")
    darkness_history = load_npz(run_dir / "darkness_history.npz")
    velocity_gain_history = load_npz(run_dir / "velocity_gain_history.npz")
    theta_hd_pref = trained_weights.get("theta_hd_pref", np.empty(0))
    training_activity_history = _history_time_window(training_history, max_duration=120.0)
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
        title="Bump maintenance HD activity",
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
        title="Darkness HD activity",
        theta_hd_pref=theta_hd_pref,
        theta_hd_decoded=darkness_history.get("theta_hd_decoded", None),
        theta_hd_decoded_peak=darkness_history.get("theta_hd_decoded_peak", None),
    )
    plot_activity_tuning_slices(
        r_hd_history=darkness_history.get("r_hd", np.empty((0, 0))),
        time=darkness_history.get("time", np.empty(0)),
        theta_hd_pref=theta_hd_pref,
        path=figures_dir / "darkness_hd_activity_slices.png",
        title="Darkness HD tuning slices",
    )
    plot_weight_matrix(
        weight_matrix=trained_weights["w_hd_to_hd"],
        path=figures_dir / "training_weight_hd_to_hd.png",
        title="Trained HD-to-HD",
        cmap="coolwarm",
        x_label="source HD preferred direction theta_source [rad]",
        y_label="target HD preferred direction theta_target [rad]",
        extent=(-np.pi, np.pi, -np.pi, np.pi),
    )
    plot_weight_matrix(
        weight_matrix=trained_weights["w_hr_to_hd"],
        path=figures_dir / "training_weight_hr_to_hd.png",
        title="Trained HR-to-HD",
        cmap="coolwarm",
        x_label="source HR cell index [unitless]",
        y_label="target HD preferred direction (theta_target) [deg]",
        extent=(0.0, float(trained_weights["w_hr_to_hd"].shape[1]), 0.0, 360.0),
    )
    plot_true_vs_decoded_heading(
        time=bump_history["time"],
        theta_true=bump_history["theta_true"],
        theta_hd_decoded=bump_history["theta_hd_decoded"],
        path=figures_dir / "bump_maintenance_decoded_heading.png",
        title="Bump maintenance",
        theta_hd_decoded_peak=bump_history.get("theta_hd_decoded_peak", None),
    )
    plot_true_vs_decoded_heading(
        time=darkness_history["time"],
        theta_true=darkness_history["theta_true"],
        theta_hd_decoded=darkness_history["theta_hd_decoded"],
        path=figures_dir / "darkness_true_vs_decoded_heading.png",
        title="Darkness path integration",
        theta_hd_decoded_peak=darkness_history.get("theta_hd_decoded_peak", None),
    )
    darkness_pi_error = circular_error_trace(
        darkness_history["theta_hd_decoded"],
        darkness_history["theta_true"],
    )
    plot_pi_error(
        time=darkness_history["time"],
        pi_error=darkness_pi_error,
        path=figures_dir / "darkness_pi_error.png",
        title="Darkness PI error",
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
