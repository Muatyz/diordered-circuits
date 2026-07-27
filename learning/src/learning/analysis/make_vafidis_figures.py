"""Generate standard figures for a saved Vafidis toy-model run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from learning.analysis.metrics import (
    circular_error_trace,
    fit_anomalous_diffusion_power_law,
    summarize_com_aligned_tuning_curves,
)
from learning.analysis.phase_flow import summarize_velocity_phase_flows
from learning.config.load_config import load_experiment_config
from learning.dynamics.hd_dynamics import (
    HD_DISTAL_NORMALIZATION_RAW_SUM,
    effective_hd_distal_weight_matrices,
)
from learning.analysis.weights import (
    compute_weight_eigenvalues,
    sort_weight_matrices_by_hd_preference,
    summarize_eigenvalue_pair_degeneracy,
)
from learning.io.save_load import load_json, load_npz, save_json, save_npz
from learning.plotting.activity import (
    plot_activity_heatmap,
    plot_activity_tuning_slices,
    plot_com_aligned_hd_tuning_population,
    plot_hd_current_heatmap,
    plot_hd_tuning_stage_comparison,
    plot_hd_tuning_settling_diagnostics,
    plot_heterogeneous_visual_input_profiles,
    plot_single_neuron_hd_tuning_curves,
)
from learning.plotting.heading import (
    plot_actual_fp_basin_rings,
    plot_bump_attractor_decoder_trajectories,
    plot_ensemble_diffusion_variance,
    plot_effective_diffusion_msd,
    plot_decoded_vs_true_heading_panels,
    plot_heading_and_pi_error_panels,
    plot_pi_error,
    plot_pi_error_ensemble,
    plot_true_vs_decoded_heading,
    plot_timescale_separation_diagnostics,
    plot_velocity_dense_probe_trajectories,
    plot_velocity_phase_flow_diagnostics,
    plot_velocity_trajectory_sweep,
    plot_velocity_gain_curve,
)
from learning.plotting.weights import (
    plot_hd_to_hd_weight_profile_history,
    plot_hr_to_hd_weight_profile_history,
    plot_weight_eigen_spectrum,
    plot_weight_matrices_side_by_side,
    plot_weight_matrix,
    plot_weight_norm_trace,
    plot_weight_snapshot_grid,
)


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


def _select_bump_maintenance_slice_times(history) -> np.ndarray | None:
    """Select post-cue landmarks and omit the freshly initialized t=0 state."""
    time = np.asarray(history.get("time", np.empty(0)), dtype=float)
    phase_id = np.asarray(history.get("phase_id", np.empty(0)), dtype=float)
    if time.ndim != 1 or phase_id.shape != time.shape or time.size == 0:
        return None
    cue_indices = np.flatnonzero(np.isclose(phase_id, 0.0))
    dark_indices = np.flatnonzero(np.isclose(phase_id, 1.0))
    if cue_indices.size == 0 or dark_indices.size == 0:
        return None
    dark_landmark_offsets = np.rint(
        np.linspace(0, dark_indices.size - 1, 4)
    ).astype(int)
    selected_indices = np.concatenate(
        ([cue_indices[-1]], dark_indices[dark_landmark_offsets])
    )
    return time[np.unique(selected_indices)]


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


def make_velocity_phase_flow_figures_for_run(*, run_dir: str | Path) -> None:
    """Regenerate direct discrete phase-flow diagnostics."""
    run_dir = Path(run_dir)
    velocity_path = run_dir / "velocity_trajectory_sweep_history.npz"
    if not velocity_path.exists():
        raise FileNotFoundError(
            f"phase-flow figures require {velocity_path.name}"
        )
    test_config_path = run_dir / "test_config_resolved.yaml"
    base_config_path = run_dir / "config_resolved.yaml"
    resolved_config = load_experiment_config(
        test_config_path if test_config_path.exists() else base_config_path
    )
    velocity_history = load_npz(velocity_path)
    summary = summarize_velocity_phase_flows(
        velocity_history=velocity_history,
        angular_bin_count=int(
            resolved_config.tests.velocity_phase_flow_angular_bins
        ),
        smoothing_bin_count=int(
            resolved_config.tests.velocity_phase_flow_smoothing_bins
        ),
        empirical_lambda_speed_floor=float(
            resolved_config.tests.velocity_phase_flow_empirical_lambda_speed_floor
        ),
    )
    save_npz(run_dir / "velocity_phase_flow_summary.npz", **summary)
    heading_dir = run_dir / "figures" / "heading"
    heading_dir.mkdir(parents=True, exist_ok=True)
    plot_actual_fp_basin_rings(
        summary=summary,
        path=heading_dir / "velocity_actual_fp_basin_rings.png",
    )
    plot_velocity_dense_probe_trajectories(
        summary=summary,
        path=heading_dir / "velocity_dense_probe_trajectories.png",
    )
    plot_velocity_phase_flow_diagnostics(
        summary=summary,
        path=heading_dir / "velocity_phase_flow_diagnostics.png",
    )


def make_vafidis_figures_for_run(*, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    activity_dir = figures_dir / "activity"
    heading_dir = figures_dir / "heading"
    weights_dir = figures_dir / "weights"
    gain_dir = figures_dir / "gain"
    diagnostics_dir = figures_dir / "diagnostics"
    for figure_subdir in [activity_dir, heading_dir, weights_dir, gain_dir, diagnostics_dir]:
        figure_subdir.mkdir(parents=True, exist_ok=True)

    test_config_path = run_dir / "test_config_resolved.yaml"
    resolved_config_path = (
        test_config_path
        if test_config_path.exists()
        else run_dir / "config_resolved.yaml"
    )
    resolved_config = (
        load_experiment_config(resolved_config_path)
        if resolved_config_path.exists()
        else None
    )
    trained_weights = load_npz(run_dir / "trained_weights.npz")
    test_metrics = load_json(run_dir / "test_metrics.json") if (run_dir / "test_metrics.json").exists() else {}
    training_history = load_npz(run_dir / "training_history.npz")
    weight_history = (
        load_npz(run_dir / "weight_history.npz")
        if (run_dir / "weight_history.npz").exists()
        else {}
    )
    bump_history = load_npz(run_dir / "bump_history.npz")
    hd_tuning_history = (
        load_npz(run_dir / "hd_tuning_history.npz")
        if (run_dir / "hd_tuning_history.npz").exists()
        else None
    )
    bump_diffusion_history = (
        load_npz(run_dir / "bump_diffusion_history.npz")
        if (run_dir / "bump_diffusion_history.npz").exists()
        else None
    )
    bump_attractor_trajectory_history = (
        load_npz(run_dir / "bump_attractor_trajectory_history.npz")
        if (run_dir / "bump_attractor_trajectory_history.npz").exists()
        else None
    )
    timescale_separation_history = (
        load_npz(run_dir / "timescale_separation_history.npz")
        if (run_dir / "timescale_separation_history.npz").exists()
        else None
    )
    velocity_trajectory_sweep_history = (
        load_npz(run_dir / "velocity_trajectory_sweep_history.npz")
        if (run_dir / "velocity_trajectory_sweep_history.npz").exists()
        else None
    )
    anomalous_diffusion_summary: dict[str, np.ndarray | float] | None = None
    if bump_diffusion_history is not None:
        if resolved_config is not None:
            fit_start_time = float(resolved_config.tests.bump_diffusion_fit_start_time)
            fit_end_time = resolved_config.tests.bump_diffusion_fit_end_time
        else:
            fit_start_time = 1.0
            fit_end_time = None
        anomalous_diffusion_summary = fit_anomalous_diffusion_power_law(
            time=bump_diffusion_history["time"],
            displacement_variance=bump_diffusion_history["pva_displacement_variance"],
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
        )
        anomalous_metric_names = {
            "anomalous_diffusion_exponent": "bump_ensemble_anomalous_diffusion_exponent",
            "generalized_diffusion_coefficient": (
                "bump_ensemble_generalized_diffusion_coefficient"
            ),
            "anomalous_diffusion_log_r_squared": (
                "bump_ensemble_anomalous_diffusion_log_r_squared"
            ),
            "anomalous_diffusion_fit_n_points": (
                "bump_ensemble_anomalous_diffusion_fit_n_points"
            ),
            "anomalous_diffusion_fit_start_time": (
                "bump_ensemble_anomalous_diffusion_fit_start_time"
            ),
            "anomalous_diffusion_fit_end_time": (
                "bump_ensemble_anomalous_diffusion_fit_end_time"
            ),
        }
        for summary_name, metric_name in anomalous_metric_names.items():
            test_metrics[metric_name] = float(anomalous_diffusion_summary[summary_name])
        rad2_to_deg2 = float(np.rad2deg(1.0) ** 2)
        if "bump_ensemble_diffusion_coefficient" in test_metrics:
            test_metrics["bump_ensemble_diffusion_coefficient_deg2_s"] = float(
                rad2_to_deg2 * test_metrics["bump_ensemble_diffusion_coefficient"]
            )
        test_metrics[
            "bump_ensemble_generalized_diffusion_coefficient_deg2_s_alpha"
        ] = float(
            rad2_to_deg2
            * test_metrics["bump_ensemble_generalized_diffusion_coefficient"]
        )
        save_json(run_dir / "test_metrics.json", test_metrics)
    darkness_history = load_npz(run_dir / "darkness_history.npz")
    ou_darkness_history = (
        load_npz(run_dir / "ou_darkness_history.npz")
        if (run_dir / "ou_darkness_history.npz").exists()
        else None
    )
    velocity_gain_history = load_npz(run_dir / "velocity_gain_history.npz")
    theta_hd_pref = trained_weights.get("theta_hd_pref", np.empty(0))
    visual_tuning_profiles = trained_weights.get(
        "visual_tuning_profiles",
        np.empty((0, 0)),
    )
    ou_pi_ensemble_history = (
        load_npz(run_dir / "ou_pi_ensemble_history.npz")
        if (run_dir / "ou_pi_ensemble_history.npz").exists()
        else None
    )
    w_hd_to_hd = trained_weights["w_hd_to_hd"]
    w_hr_to_hd = trained_weights["w_hr_to_hd"]
    hd_distal_normalization = (
        resolved_config.model.hd_distal_normalization
        if resolved_config is not None
        else HD_DISTAL_NORMALIZATION_RAW_SUM
    )
    effective_w_hd_to_hd, effective_w_hr_to_hd = effective_hd_distal_weight_matrices(
        w_hd_to_hd=w_hd_to_hd,
        w_hr_to_hd=w_hr_to_hd,
        normalization=hd_distal_normalization,
    )
    empirical_hd_preference_raw = (
        np.asarray(hd_tuning_history["empirical_preferred_direction"], dtype=float)
        if hd_tuning_history is not None
        and "empirical_preferred_direction" in hd_tuning_history
        else np.asarray(theta_hd_pref, dtype=float)
    )
    empirical_hd_preference = np.where(
        np.isfinite(empirical_hd_preference_raw),
        empirical_hd_preference_raw,
        np.asarray(theta_hd_pref, dtype=float),
    )
    activity_sort_label = (
        "empirical-COM" if hd_tuning_history is not None else "model-preference"
    )
    theta_hd_plot_pref = empirical_hd_preference
    sorted_weights = sort_weight_matrices_by_hd_preference(
        w_hd_to_hd=w_hd_to_hd,
        w_hr_to_hd=w_hr_to_hd,
        theta_hd_preference=empirical_hd_preference,
    )
    save_npz(
        run_dir / "empirical_com_sort_order.npz",
        hd_order=np.asarray(sorted_weights["hd_order"], dtype=int),
        hr_order=np.asarray(sorted_weights["hr_order"], dtype=int),
        theta_hd_preference=np.asarray(
            sorted_weights["theta_hd_preference"], dtype=float
        ),
    )
    hd_to_hd_eigenvalues = compute_weight_eigenvalues(effective_w_hd_to_hd)
    hr_to_hd_eigenvalues = compute_weight_eigenvalues(effective_w_hr_to_hd)
    spectrum_diagnostics = {
        "normalization": hd_distal_normalization,
        "matrix_semantics": "effective_connectivity",
        "hd_to_hd": summarize_eigenvalue_pair_degeneracy(
            weight_matrix=effective_w_hd_to_hd
        ),
        "hr_to_hd": summarize_eigenvalue_pair_degeneracy(
            weight_matrix=effective_w_hr_to_hd
        ),
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

    if visual_tuning_profiles.size > 0 and resolved_config is not None:
        plot_heterogeneous_visual_input_profiles(
            tuning_profiles=visual_tuning_profiles,
            path=activity_dir / "heterogeneous_visual_input_profiles.png",
            sample_count=resolved_config.visual.heterogeneous_plot_sample_count,
            seed=(
                resolved_config.simulation.seed
                + resolved_config.visual.heterogeneous_plot_seed_offset
            ),
            amplitude=resolved_config.visual.amplitude,
            baseline=resolved_config.visual.baseline,
            light_excitation=resolved_config.visual.light_excitation,
            proximal_scale=resolved_config.visual.proximal_scale,
            theta_hd_pref=theta_hd_pref,
        )

    if hd_tuning_history is not None and resolved_config is not None:
        plot_single_neuron_hd_tuning_curves(
            theta_true=hd_tuning_history["theta_true"],
            r_hd_by_heading=hd_tuning_history["r_hd"],
            preferred_direction=empirical_hd_preference,
            path=activity_dir / "single_neuron_hd_tuning_curves.png",
            sample_count=resolved_config.tests.hd_tuning_curve_sample_count,
            seed=resolved_config.simulation.seed + 70_000,
        )

    if hd_tuning_history is not None:
        com_aligned_tuning_summary = summarize_com_aligned_tuning_curves(
            theta_true=hd_tuning_history["theta_true"],
            r_hd_by_heading=hd_tuning_history["r_hd"],
        )
        test_metrics["post_training_hd_tuning_valid_neuron_count"] = int(
            com_aligned_tuning_summary["r_hd_tuning_valid_neuron_count"]
        )
        test_metrics["post_training_hd_tuning_silent_neuron_count"] = int(
            com_aligned_tuning_summary["r_hd_tuning_silent_neuron_count"]
        )
        test_metrics["post_training_hd_tuning_silent_neuron_fraction"] = float(
            com_aligned_tuning_summary["r_hd_tuning_silent_neuron_fraction"]
        )
        save_npz(
            run_dir / "hd_tuning_com_aligned.npz",
            **com_aligned_tuning_summary,
        )
        plot_com_aligned_hd_tuning_population(
            theta_aligned=com_aligned_tuning_summary["theta_aligned"],
            r_hd_peak_normalized_com_aligned=com_aligned_tuning_summary[
                "r_hd_peak_normalized_com_aligned"
            ],
            population_mean=com_aligned_tuning_summary[
                "r_hd_peak_normalized_com_aligned_mean"
            ],
            population_std=com_aligned_tuning_summary[
                "r_hd_peak_normalized_com_aligned_std"
            ],
            path=activity_dir / "com_aligned_hd_tuning_population.png",
        )
        if "r_hd_visual_only" in hd_tuning_history:
            visual_only_tuning_summary = summarize_com_aligned_tuning_curves(
                theta_true=hd_tuning_history["theta_true"],
                r_hd_by_heading=hd_tuning_history["r_hd_visual_only"],
            )
            test_metrics["visual_only_hd_tuning_valid_neuron_count"] = int(
                visual_only_tuning_summary["r_hd_tuning_valid_neuron_count"]
            )
            test_metrics["visual_only_hd_tuning_silent_neuron_count"] = int(
                visual_only_tuning_summary["r_hd_tuning_silent_neuron_count"]
            )
            test_metrics["visual_only_hd_tuning_silent_neuron_fraction"] = float(
                visual_only_tuning_summary["r_hd_tuning_silent_neuron_fraction"]
            )
            save_npz(
                run_dir / "hd_tuning_com_aligned.npz",
                **com_aligned_tuning_summary,
                **{
                    f"visual_only_{key}": value
                    for key, value in visual_only_tuning_summary.items()
                    if key != "theta_aligned"
                },
            )
            plot_hd_tuning_stage_comparison(
                theta_aligned=com_aligned_tuning_summary["theta_aligned"],
                visual_only_mean=visual_only_tuning_summary[
                    "r_hd_peak_normalized_com_aligned_mean"
                ],
                visual_only_std=visual_only_tuning_summary[
                    "r_hd_peak_normalized_com_aligned_std"
                ],
                post_training_mean=com_aligned_tuning_summary[
                    "r_hd_peak_normalized_com_aligned_mean"
                ],
                post_training_std=com_aligned_tuning_summary[
                    "r_hd_peak_normalized_com_aligned_std"
                ],
                path=activity_dir / "hd_tuning_visual_only_vs_post_training.png",
            )
        save_json(run_dir / "test_metrics.json", test_metrics)
        if (
            resolved_config is not None
            and resolved_config.tests.hd_tuning_curve_convergence_tolerance is not None
            and all(
                key in hd_tuning_history
                for key in [
                    "actual_settle_duration",
                    "final_window_max_rate_change",
                    "settle_converged",
                ]
            )
        ):
            plot_hd_tuning_settling_diagnostics(
                theta_true=hd_tuning_history["theta_true"],
                actual_settle_duration=hd_tuning_history["actual_settle_duration"],
                final_window_max_rate_change=hd_tuning_history[
                    "final_window_max_rate_change"
                ],
                settle_converged=hd_tuning_history["settle_converged"],
                convergence_tolerance=float(
                    resolved_config.tests.hd_tuning_curve_convergence_tolerance
                ),
                path=diagnostics_dir / "hd_tuning_settling_diagnostics.png",
            )

    plot_activity_heatmap(
        r_hd_history=training_activity_history.get("r_hd", np.empty((0, 0))),
        time=training_activity_history.get("time", np.empty(0)),
        path=activity_dir / "training_hd_activity_heatmap.png",
        title=f"Training HD activity ({activity_sort_label} sorted; first 120 s)" if training_history_truncated else f"Training HD activity ({activity_sort_label} sorted)",
        theta_hd_pref=theta_hd_plot_pref,
        theta_hd_decoded=training_activity_history.get("theta_hd_decoded", None),
        theta_hd_decoded_peak=training_activity_history.get("theta_hd_decoded_peak", None),
        decode_theta_hd_pref=theta_hd_plot_pref,
    )
    plot_hd_current_heatmap(
        current_history=training_activity_history.get("i_vis_to_hd", np.empty((0, 0))),
        time=training_activity_history.get("time", np.empty(0)),
        path=activity_dir / "training_visual_input_heatmap.png",
        title=f"Training visual input to HD ({activity_sort_label} sorted; first 120 s)" if training_history_truncated else f"Training visual input to HD ({activity_sort_label} sorted)",
        theta_hd_pref=theta_hd_plot_pref,
        theta_true=training_activity_history.get("theta_true", None),
        colorbar_label="I_vis to HD [current]",
    )
    plot_decoded_vs_true_heading_panels(
        time=training_movement_history.get("time", np.empty(0)),
        theta_true=training_movement_history.get("theta_true", np.empty(0)),
        theta_hd_decoded=training_movement_history.get("theta_hd_decoded", np.empty(0)),
        theta_hd_decoded_peak=training_movement_history.get("theta_hd_decoded_peak", None),
        path=heading_dir / "training_heading_short_window.png",
        title="Training heading comparison (30 s movement window)",
    )
    if training_history_truncated:
        plot_activity_heatmap(
            r_hd_history=training_history.get("r_hd", np.empty((0, 0))),
            time=training_history.get("time", np.empty(0)),
            path=activity_dir / "training_hd_activity_heatmap_full.png",
            title=f"Training HD activity ({activity_sort_label} sorted; full)",
            theta_hd_pref=theta_hd_plot_pref,
        )
    plot_activity_tuning_slices(
        r_hd_history=training_history.get("r_hd", np.empty((0, 0))),
        time=training_history.get("time", np.empty(0)),
        theta_hd_pref=theta_hd_plot_pref,
        path=activity_dir / "training_hd_activity_slices.png",
        title=f"Training HD activity slices ({activity_sort_label} sorted)",
    )
    plot_activity_heatmap(
        r_hd_history=bump_history.get("r_hd", np.empty((0, 0))),
        time=bump_history.get("time", np.empty(0)),
        path=activity_dir / "bump_maintenance_hd_activity_heatmap.png",
        title=f"Bump maintenance HD activity ({activity_sort_label} sorted; cue, then visual off, omega=0)",
        theta_hd_pref=theta_hd_plot_pref,
        theta_hd_decoded=bump_history.get("theta_hd_decoded", None),
        theta_hd_decoded_peak=bump_history.get("theta_hd_decoded_peak", None),
        decode_theta_hd_pref=theta_hd_plot_pref,
        phase_id=bump_history.get("phase_id", None),
    )
    plot_hd_current_heatmap(
        current_history=bump_history.get("i_vis_to_hd", np.empty((0, 0))),
        time=bump_history.get("time", np.empty(0)),
        path=activity_dir / "bump_maintenance_visual_input_heatmap.png",
        title="Bump maintenance visual input (cue, then visual off)",
        theta_hd_pref=theta_hd_plot_pref,
        theta_true=bump_history.get("theta_true", None),
        phase_id=bump_history.get("phase_id", None),
        colorbar_label="I_vis to HD [current]",
    )
    plot_activity_tuning_slices(
        r_hd_history=bump_history.get("r_hd", np.empty((0, 0))),
        time=bump_history.get("time", np.empty(0)),
        theta_hd_pref=theta_hd_plot_pref,
        path=activity_dir / "bump_maintenance_hd_activity_slices.png",
        title="Post-training bump maintenance HD activity slices",
        slice_times=_select_bump_maintenance_slice_times(bump_history),
        time_context="frozen-weight protocol",
    )
    plot_activity_heatmap(
        r_hd_history=darkness_history.get("r_hd", np.empty((0, 0))),
        time=darkness_history.get("time", np.empty(0)),
        path=activity_dir / "darkness_hd_activity_heatmap.png",
        title=f"Constant-velocity PI HD activity ({activity_sort_label} sorted; visual, dark, visual)",
        theta_hd_pref=theta_hd_plot_pref,
        theta_hd_decoded=darkness_history.get("theta_hd_decoded", None),
        theta_hd_decoded_peak=darkness_history.get("theta_hd_decoded_peak", None),
        decode_theta_hd_pref=theta_hd_plot_pref,
        phase_id=darkness_history.get("phase_id", None),
    )
    plot_hd_current_heatmap(
        current_history=darkness_history.get("i_vis_to_hd", np.empty((0, 0))),
        time=darkness_history.get("time", np.empty(0)),
        path=activity_dir / "darkness_visual_input_heatmap.png",
        title="Constant-velocity PI visual input (visual, dark, visual)",
        theta_hd_pref=theta_hd_plot_pref,
        theta_true=darkness_history.get("theta_true", None),
        phase_id=darkness_history.get("phase_id", None),
        colorbar_label="I_vis to HD [current]",
    )
    plot_activity_tuning_slices(
        r_hd_history=darkness_history.get("r_hd", np.empty((0, 0))),
        time=darkness_history.get("time", np.empty(0)),
        theta_hd_pref=theta_hd_plot_pref,
        path=activity_dir / "darkness_hd_activity_slices.png",
        title="Constant-velocity PI HD tuning slices",
    )
    if ou_darkness_history is not None:
        plot_activity_heatmap(
            r_hd_history=ou_darkness_history.get("r_hd", np.empty((0, 0))),
            time=ou_darkness_history.get("time", np.empty(0)),
            path=activity_dir / "ou_darkness_hd_activity_heatmap.png",
            title=f"OU PI HD activity ({activity_sort_label} sorted; visual, dark, visual)",
            theta_hd_pref=theta_hd_plot_pref,
            theta_hd_decoded=ou_darkness_history.get("theta_hd_decoded", None),
            theta_hd_decoded_peak=ou_darkness_history.get("theta_hd_decoded_peak", None),
            decode_theta_hd_pref=theta_hd_plot_pref,
            phase_id=ou_darkness_history.get("phase_id", None),
        )
        plot_hd_current_heatmap(
            current_history=ou_darkness_history.get("i_vis_to_hd", np.empty((0, 0))),
            time=ou_darkness_history.get("time", np.empty(0)),
            path=activity_dir / "ou_darkness_visual_input_heatmap.png",
            title="OU PI visual input (visual, dark, visual)",
            theta_hd_pref=theta_hd_plot_pref,
            theta_true=ou_darkness_history.get("theta_true", None),
            phase_id=ou_darkness_history.get("phase_id", None),
            colorbar_label="I_vis to HD [current]",
        )
    plot_weight_matrix(
        weight_matrix=sorted_weights["w_hd_to_hd"],
        path=weights_dir / "training_weight_hd_to_hd.png",
        title=f"Trained HD-to-HD ({activity_sort_label} sorted)",
        cmap="coolwarm",
        x_label="source HD neuron ID (COM-sorted)",
        y_label="target HD neuron ID (COM-sorted)",
    )
    plot_weight_matrix(
        weight_matrix=sorted_weights["w_hr_to_hd"],
        path=weights_dir / "training_weight_hr_to_hd.png",
        title=f"Trained HR-to-HD (target {activity_sort_label} sorted; L/R sources sorted within wing)",
        cmap="coolwarm",
        x_label="source HR neuron ID (L/R, COM-sorted)",
        y_label="target HD neuron ID (COM-sorted)",
    )
    plot_weight_matrices_side_by_side(
        w_hd_to_hd=sorted_weights["w_hd_to_hd"],
        w_hr_to_hd=sorted_weights["w_hr_to_hd"],
        path=weights_dir / "training_weight_matrices_side_by_side.png",
        title=f"Trained weights ({activity_sort_label} sorted)",
    )
    plot_weight_matrix(
        weight_matrix=w_hd_to_hd,
        path=weights_dir / "training_weight_hd_to_hd_raw_index_order.png",
        title="Trained HD-to-HD (raw model index order)",
        cmap="coolwarm",
    )
    plot_weight_matrix(
        weight_matrix=w_hr_to_hd,
        path=weights_dir / "training_weight_hr_to_hd_raw_index_order.png",
        title="Trained HR-to-HD (raw model index order)",
        cmap="coolwarm",
    )
    if weight_history:
        hd_order = np.asarray(sorted_weights["hd_order"], dtype=int)
        hr_order = np.asarray(sorted_weights["hr_order"], dtype=int)
        w_hd_history_raw = weight_history.get("w_hd_to_hd", np.empty((0, 0, 0)))
        w_hr_history_raw = weight_history.get("w_hr_to_hd", np.empty((0, 0, 0)))
        w_hd_history_sorted = (
            w_hd_history_raw[:, hd_order][:, :, hd_order]
            if w_hd_history_raw.ndim == 3 and w_hd_history_raw.shape[1:] == w_hd_to_hd.shape
            else np.empty((0, 0, 0))
        )
        w_hr_history_sorted = (
            w_hr_history_raw[:, hd_order][:, :, hr_order]
            if w_hr_history_raw.ndim == 3 and w_hr_history_raw.shape[1:] == w_hr_to_hd.shape
            else np.empty((0, 0, 0))
        )
        plot_hd_to_hd_weight_profile_history(
            weight_history=weight_history.get("w_hd_to_hd", np.empty((0, 0, 0))),
            time=weight_history.get("time", np.empty(0)),
            theta_hd_pref=theta_hd_pref,
            path=weights_dir / "training_weight_hd_to_hd_profile_history.png",
            title="HD-to-HD weight profile development",
        )
        plot_hr_to_hd_weight_profile_history(
            weight_history=weight_history.get("w_hr_to_hd", np.empty((0, 0, 0))),
            time=weight_history.get("time", np.empty(0)),
            theta_hd_pref=theta_hd_pref,
            path=weights_dir / "training_weight_hr_to_hd_profile_history.png",
            title="HR-to-HD weight profile development",
        )
        plot_weight_snapshot_grid(
            weight_history=w_hd_history_sorted,
            time=weight_history.get("time", np.empty(0)),
            path=weights_dir / "training_weight_hd_to_hd_over_time.png",
            title=f"HD-to-HD weights across training (final {activity_sort_label} order)",
            x_label="source HD neuron ID (COM-sorted)",
            y_label="target HD neuron ID (COM-sorted)",
        )
        plot_weight_snapshot_grid(
            weight_history=w_hr_history_sorted,
            time=weight_history.get("time", np.empty(0)),
            path=weights_dir / "training_weight_hr_to_hd_over_time.png",
            title=f"HR-to-HD weights across training (final {activity_sort_label} order)",
            x_label="source HR neuron ID (L/R, COM-sorted)",
            y_label="target HD neuron ID (COM-sorted)",
        )
        plot_weight_norm_trace(
            time=weight_history.get("time", np.empty(0)),
            weight_norm_hd_to_hd=weight_history.get(
                "effective_weight_norm_hd_to_hd",
                weight_history.get("weight_norm_hd_to_hd", np.empty(0)),
            ),
            weight_norm_hr_to_hd=weight_history.get(
                "effective_weight_norm_hr_to_hd",
                weight_history.get("weight_norm_hr_to_hd", np.empty(0)),
            ),
            path=weights_dir / "training_weight_norms_over_time.png",
            title="Effective connectivity norms across training",
        )
    plot_weight_eigen_spectrum(
        hd_to_hd_eigenvalues=hd_to_hd_eigenvalues,
        hr_to_hd_eigenvalues=hr_to_hd_eigenvalues,
        path=diagnostics_dir / "training_weight_eigen_spectrum.png",
        title="Effective connectivity eigenvalue spectrum",
        diagnostics=spectrum_diagnostics,
    )
    plot_true_vs_decoded_heading(
        time=bump_history["time"],
        theta_true=bump_history["theta_true"],
        theta_hd_decoded=bump_history["theta_hd_decoded"],
        path=heading_dir / "bump_maintenance_decoded_heading.png",
        title="Bump maintenance (cue, then visual off, omega=0)",
        theta_hd_decoded_peak=bump_history.get("theta_hd_decoded_peak", None),
    )
    if (
        bump_attractor_trajectory_history is not None
        and np.asarray(
            bump_attractor_trajectory_history.get("time", np.empty(0)),
            dtype=float,
        ).size
        > 0
    ):
        plot_bump_attractor_decoder_trajectories(
            time=bump_attractor_trajectory_history["time"],
            theta_initial=bump_attractor_trajectory_history["theta_initial"],
            theta_pva=bump_attractor_trajectory_history["theta_pva"],
            theta_peak=bump_attractor_trajectory_history["theta_peak"],
            theta_overlap=bump_attractor_trajectory_history["theta_overlap"],
            path=heading_dir / "bump_attractor_decoder_trajectories.png",
        )
    if (
        timescale_separation_history is not None
        and np.asarray(
            timescale_separation_history.get("normal_time", np.empty(0)),
            dtype=float,
        ).size
        > 0
    ):
        plot_timescale_separation_diagnostics(
            normal_time=timescale_separation_history["normal_time"],
            perturbation_scale=timescale_separation_history["perturbation_scale"],
            normal_distance_to_manifold=timescale_separation_history[
                "normal_distance_to_manifold"
            ],
            normal_control_distance_to_manifold=timescale_separation_history[
                "normal_control_distance_to_manifold"
            ],
            normal_e_folding_time=timescale_separation_history[
                "normal_e_folding_time"
            ],
            normal_recovery_observed=timescale_separation_history[
                "normal_recovery_observed"
            ],
            tangential_time=timescale_separation_history["tangential_time"],
            tangential_overlap_displacement=timescale_separation_history[
                "tangential_overlap_displacement"
            ],
            tangential_first_passage_time=timescale_separation_history[
                "tangential_first_passage_time"
            ],
            tangential_first_passage_observed=timescale_separation_history[
                "tangential_first_passage_observed"
            ],
            tangential_threshold_rad=float(
                timescale_separation_history["tangential_threshold_rad"]
            ),
            normal_time_p90=float(
                timescale_separation_history["normal_time_p90"]
            ),
            tangential_time_p10=float(
                timescale_separation_history["tangential_time_p10"]
            ),
            conservative_timescale_ratio=float(
                timescale_separation_history["conservative_timescale_ratio"]
            ),
            criterion_ratio_threshold=float(
                timescale_separation_history["criterion_ratio_threshold"]
            ),
            criterion_passed=bool(
                float(timescale_separation_history["criterion_passed"])
            ),
            criterion_ratio_is_lower_bound=bool(
                float(
                    timescale_separation_history[
                        "criterion_ratio_is_lower_bound"
                    ]
                )
            ),
            path=heading_dir / "timescale_separation_diagnostics.png",
        )
    if (
        velocity_trajectory_sweep_history is not None
        and np.asarray(
            velocity_trajectory_sweep_history.get("time", np.empty(0)),
            dtype=float,
        ).size
        > 0
    ):
        plot_velocity_trajectory_sweep(
            time=velocity_trajectory_sweep_history["time"],
            commanded_velocity=velocity_trajectory_sweep_history[
                "commanded_velocity"
            ],
            theta_initial=velocity_trajectory_sweep_history["theta_initial"],
            pva_angular_displacement=velocity_trajectory_sweep_history[
                "pva_angular_displacement"
            ],
            overlap_angular_displacement=velocity_trajectory_sweep_history[
                "overlap_angular_displacement"
            ],
            decoded_velocity_pva=velocity_trajectory_sweep_history[
                "decoded_velocity_pva"
            ],
            decoded_velocity_overlap=velocity_trajectory_sweep_history[
                "decoded_velocity_overlap"
            ],
            path=heading_dir / "velocity_trajectory_sweep.png",
        )
        phase_flow_theta = np.asarray(
            velocity_trajectory_sweep_history.get(
                "phase_flow_theta_pva", np.empty(0)
            ),
            dtype=float,
        )
        if phase_flow_theta.size > 0:
            phase_flow_summary = summarize_velocity_phase_flows(
                velocity_history=velocity_trajectory_sweep_history,
                angular_bin_count=(
                    360
                    if resolved_config is None
                    else int(
                        resolved_config.tests.velocity_phase_flow_angular_bins
                    )
                ),
                smoothing_bin_count=(
                    5
                    if resolved_config is None
                    else int(
                        resolved_config.tests.velocity_phase_flow_smoothing_bins
                    )
                ),
                empirical_lambda_speed_floor=(
                    0.02
                    if resolved_config is None
                    else float(
                        resolved_config.tests.velocity_phase_flow_empirical_lambda_speed_floor
                    )
                ),
            )
            save_npz(
                run_dir / "velocity_phase_flow_summary.npz",
                **phase_flow_summary,
            )
            plot_actual_fp_basin_rings(
                summary=phase_flow_summary,
                path=heading_dir / "velocity_actual_fp_basin_rings.png",
            )
            plot_velocity_dense_probe_trajectories(
                summary=phase_flow_summary,
                path=heading_dir / "velocity_dense_probe_trajectories.png",
            )
            plot_velocity_phase_flow_diagnostics(
                summary=phase_flow_summary,
                path=heading_dir / "velocity_phase_flow_diagnostics.png",
            )
    plot_true_vs_decoded_heading(
        time=darkness_history["time"],
        theta_true=darkness_history["theta_true"],
        theta_hd_decoded=darkness_history["theta_hd_decoded"],
        path=heading_dir / "darkness_true_vs_decoded_heading.png",
        title="Constant-velocity path integration (visual, dark, visual)",
        theta_hd_decoded_peak=darkness_history.get("theta_hd_decoded_peak", None),
        phase_id=darkness_history.get("phase_id", None),
    )
    darkness_pi_error = circular_error_trace(
        darkness_history["theta_hd_decoded"],
        darkness_history["theta_true"],
    )
    plot_heading_and_pi_error_panels(
        time=darkness_history["time"],
        theta_true=darkness_history["theta_true"],
        theta_hd_decoded=darkness_history["theta_hd_decoded"],
        path=heading_dir / "darkness_heading_and_pi_error.png",
        title="Constant-velocity path integration (visual, dark, visual)",
        theta_hd_decoded_peak=darkness_history.get("theta_hd_decoded_peak", None),
        phase_id=darkness_history.get("phase_id", None),
    )
    plot_pi_error(
        time=darkness_history["time"],
        pi_error=darkness_pi_error,
        path=heading_dir / "darkness_pi_error.png",
        title="Constant-velocity PI error (visual, dark, visual)",
        phase_id=darkness_history.get("phase_id", None),
    )
    if bump_diffusion_history is not None:
        pva_displacement_trials = np.asarray(
            bump_diffusion_history.get("pva_angular_displacement", np.empty((0, 0))),
            dtype=float,
        )
        ensemble_trial_count = (
            int(pva_displacement_trials.shape[0])
            if pva_displacement_trials.ndim == 2
            else None
        )
        anomalous_fit = bump_diffusion_history.get("pva_anomalous_diffusion_fit", None)
        if anomalous_diffusion_summary is not None:
            anomalous_fit = np.asarray(
                anomalous_diffusion_summary["anomalous_diffusion_fit_trace"],
                dtype=float,
            )
        plot_ensemble_diffusion_variance(
            time=bump_diffusion_history["time"],
            displacement_mean=bump_diffusion_history["pva_displacement_mean"],
            displacement_variance=bump_diffusion_history["pva_displacement_variance"],
            diffusion_coefficient=float(
                test_metrics.get("bump_ensemble_diffusion_coefficient", np.nan)
            ),
            systematic_drift_velocity=float(
                test_metrics.get("bump_ensemble_systematic_drift_velocity", np.nan)
            ),
            anomalous_diffusion_fit=anomalous_fit,
            anomalous_diffusion_exponent=float(
                test_metrics.get("bump_ensemble_anomalous_diffusion_exponent", np.nan)
            ),
            generalized_diffusion_coefficient=float(
                test_metrics.get(
                    "bump_ensemble_generalized_diffusion_coefficient", np.nan
                )
            ),
            anomalous_diffusion_log_r_squared=float(
                test_metrics.get(
                    "bump_ensemble_anomalous_diffusion_log_r_squared", np.nan
                )
            ),
            anomalous_diffusion_fit_start_time=float(
                test_metrics.get(
                    "bump_ensemble_anomalous_diffusion_fit_start_time", np.nan
                )
            ),
            anomalous_diffusion_fit_end_time=float(
                test_metrics.get(
                    "bump_ensemble_anomalous_diffusion_fit_end_time", np.nan
                )
            ),
            n_trials=ensemble_trial_count,
            path=diagnostics_dir / "bump_ensemble_diffusion_variance.png",
        )
    else:
        plot_effective_diffusion_msd(
            time=bump_history["time"],
            theta_decoded=bump_history["theta_hd_decoded"],
            theta_reference=(
                float(bump_history["theta_true"][0])
                if bump_history["theta_true"].size
                else 0.0
            ),
            diffusion_coefficient=float(
                test_metrics.get("bump_effective_diffusion_coefficient", np.nan)
            ),
            path=diagnostics_dir / "bump_effective_diffusion_msd.png",
            title="Legacy single-trace displacement diagnostic",
        )
    if ou_darkness_history is not None:
        plot_true_vs_decoded_heading(
            time=ou_darkness_history["time"],
            theta_true=ou_darkness_history["theta_true"],
            theta_hd_decoded=ou_darkness_history["theta_hd_decoded"],
            path=heading_dir / "ou_darkness_true_vs_decoded_heading.png",
            title="OU path integration (visual, dark, visual)",
            theta_hd_decoded_peak=ou_darkness_history.get("theta_hd_decoded_peak", None),
            phase_id=ou_darkness_history.get("phase_id", None),
        )
        ou_darkness_pi_error = circular_error_trace(
            ou_darkness_history["theta_hd_decoded"],
            ou_darkness_history["theta_true"],
        )
        plot_heading_and_pi_error_panels(
            time=ou_darkness_history["time"],
            theta_true=ou_darkness_history["theta_true"],
            theta_hd_decoded=ou_darkness_history["theta_hd_decoded"],
            path=heading_dir / "ou_darkness_heading_and_pi_error.png",
            title="OU path integration (visual, dark, visual)",
            theta_hd_decoded_peak=ou_darkness_history.get("theta_hd_decoded_peak", None),
            phase_id=ou_darkness_history.get("phase_id", None),
        )
        plot_pi_error(
            time=ou_darkness_history["time"],
            pi_error=ou_darkness_pi_error,
            path=heading_dir / "ou_darkness_pi_error.png",
            title="OU PI error (visual, dark, visual)",
            phase_id=ou_darkness_history.get("phase_id", None),
        )
    if ou_pi_ensemble_history is not None:
        plot_pi_error_ensemble(
            time=ou_pi_ensemble_history["time"],
            pi_error_mean=ou_pi_ensemble_history["pva_pi_error_mean"],
            pi_error_sem=ou_pi_ensemble_history["pva_pi_error_sem"],
            systematic_drift_velocity=float(
                test_metrics.get("ou_pi_ensemble_systematic_drift_velocity", np.nan)
            ),
            drift_intercept=float(
                test_metrics.get("ou_pi_ensemble_drift_intercept", np.nan)
            ),
            n_trials=int(test_metrics.get("ou_pi_ensemble_n_trials", 0)),
            path=heading_dir / "ou_pi_error_ensemble.png",
        )
    plot_velocity_gain_curve(
        commanded_velocity=velocity_gain_history["commanded_velocity"],
        decoded_velocity=velocity_gain_history["decoded_velocity"],
        path=gain_dir / "velocity_gain_curve.png",
        title="Velocity gain curve (visual cue vs darkness)",
        decoded_velocity_peak=velocity_gain_history.get("decoded_velocity_peak", None),
        decoded_velocity_visual=velocity_gain_history.get("decoded_velocity_visual", None),
        decoded_velocity_visual_peak=velocity_gain_history.get("decoded_velocity_visual_peak", None),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Saved run directory.")
    parser.add_argument(
        "--phase-flow-only",
        action="store_true",
        help="Regenerate only direct phase-flow, fixed-point, and basin figures.",
    )
    args = parser.parse_args()
    if args.phase_flow_only:
        make_velocity_phase_flow_figures_for_run(run_dir=args.run_dir)
    else:
        make_vafidis_figures_for_run(run_dir=args.run_dir)
    print(f"Saved figures to {Path(args.run_dir) / 'figures'}")


if __name__ == "__main__":
    main()
