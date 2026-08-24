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
from learning.config.diagnostics import (
    DIAGNOSTIC_GROUPS,
    selected_diagnostic_groups,
)
from learning.io.save_load import load_json, load_npz, save_json, save_npz
from learning.plotting.activity import (
    plot_activity_heatmap,
    plot_com_aligned_hd_tuning_population,
    plot_hd_current_heatmap,
    plot_hd_tuning_stage_comparison,
    plot_single_neuron_hd_tuning_curves,
)
from learning.plotting.heading import (
    plot_actual_fp_basin_rings,
    plot_bump_attractor_cue_transfer,
    plot_bump_attractor_decoder_trajectories,
    plot_bump_attractor_pva_trajectories,
    plot_diffusion_noise_sweep,
    plot_ensemble_diffusion_variance,
    plot_numerical_convergence_diagnostics,
    plot_heading_and_pi_error_panels,
    plot_constant_velocity_pi_error_grid,
    plot_pi_error,
    plot_pi_error_ensemble,
    plot_true_vs_decoded_heading,
    plot_timescale_separation_diagnostics,
    plot_velocity_dense_probe_trajectories,
    plot_velocity_phase_flow_diagnostics,
    plot_velocity_trajectory_sweep,
    plot_velocity_gain_curve,
)
from learning.plotting.slow_manifold import (
    plot_ramesan_firing_rate_diagnostics,
    plot_ramesan_pca_variance_rank,
    plot_ramesan_phase_landscape,
    plot_slow_manifold_diagnostics,
)
from learning.plotting.weights import (
    plot_weight_matrices_side_by_side,
    plot_weight_norm_trace,
    plot_training_absolute_learning_error,
    plot_weight_snapshot_pi_development,
    plot_weight_snapshot_grid,
)


def _plot_pi_error_trace(history: dict[str, np.ndarray]) -> np.ndarray:
    """Return a full-protocol PI error trace with no NaN gaps.

    Preferred source is the full-protocol accumulated error
    (``pi_error_full_protocol``), which is referenced to darkness onset and
    unwrapped continuously across the whole cue-darkness-recue window: the
    cue and recue segments are real integrator errors (visual anchor holds /
    re-anchors the accumulated error), not NaN gaps.

    Fallback: ``pi_error_release_relative`` is only defined over the darkness
    phase (its reference zero is darkness onset); the visual cue and recue
    phases are stored as NaN.  Plotting that array directly erases the
    cue/recue segments from the error panel.  This helper fills every NaN
    sample with the absolute circular decode error (``decoded - true``), so
    the whole visual-dark-visual window is drawn: darkness shows the
    accumulated release-relative integrator error, while the anchored
    cue/recue phases show the small instantaneous decode error.
    """
    if "pi_error_full_protocol" in history:
        return np.asarray(
            history["pi_error_full_protocol"], dtype=float
        ).copy()
    if "pi_error_release_relative" not in history:
        return circular_error_trace(
            history["theta_hd_decoded"],
            history["theta_true"],
        )
    trace = np.asarray(history["pi_error_release_relative"], dtype=float).copy()
    missing = np.isnan(trace)
    if np.any(missing):
        decode_error = circular_error_trace(
            history["theta_hd_decoded"],
            history["theta_true"],
        )
        trace[missing] = decode_error[missing]
    return trace


def _plot_bump_attractor_pva_figure_set(*, history, figure_dir: Path) -> None:
    """Write the three coordinate-separated PVA diagnostic figures."""

    endpoint_arguments = {
        "time": history["time"],
        "theta_initial": history["theta_initial"],
        "theta_pva": history["theta_pva"],
        "cue_time": history.get("cue_time"),
        "cue_theta_pva": history.get("cue_theta_pva"),
        "endpoint_probe_theta_initial": history.get(
            "endpoint_probe_theta_initial"
        ),
        "endpoint_probe_refinement_level": history.get(
            "endpoint_probe_refinement_level"
        ),
        "endpoint_probe_theta_release_pva": history.get(
            "endpoint_probe_theta_release_pva"
        ),
        "endpoint_probe_theta_final_pva": history.get(
            "endpoint_probe_theta_final_pva"
        ),
        "endpoint_probe_theta_pva_trajectory": history.get(
            "endpoint_probe_theta_pva_trajectory"
        ),
    }
    plot_bump_attractor_pva_trajectories(
        **endpoint_arguments,
        endpoint_map_coordinate="initial_cue",
        path=figure_dir / "bump_attractor_pva_initial_cue_endpoint_map.png",
    )
    plot_bump_attractor_pva_trajectories(
        **endpoint_arguments,
        endpoint_map_coordinate="release",
        path=figure_dir / "bump_attractor_pva_release_angle_endpoint_map.png",
    )

    # Cue transfer is defined on the uniform coarse cue grid.  Bisection
    # probes refine autonomous basin boundaries and would create repeated,
    # nonuniform x coordinates in this presentation-only figure.
    transfer_initial = history["theta_initial"]
    cue_theta_pva = history.get("cue_theta_pva")
    transfer_release = (
        np.asarray(cue_theta_pva, dtype=float)[:, -1]
        if cue_theta_pva is not None
        else np.asarray(history["theta_pva"], dtype=float)[:, 0]
    )
    plot_bump_attractor_cue_transfer(
        theta_initial=transfer_initial,
        theta_release=transfer_release,
        path=figure_dir / "bump_attractor_pva_cue_transfer.png",
    )


def _diagnostic_figure_dir(
    *, figures_dir: Path, group_name: str
) -> Path:
    """Return the canonical output directory for one diagnostics-config group."""
    if group_name not in DIAGNOSTIC_GROUPS:
        raise ValueError(f"unknown diagnostic figure group: {group_name!r}")
    figure_dir = figures_dir / group_name
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


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
    figure_dir = _diagnostic_figure_dir(
        figures_dir=run_dir / "figures",
        group_name="velocity_dynamics_and_phase_flow",
    )
    plot_actual_fp_basin_rings(
        summary=summary,
        path=figure_dir / "velocity_actual_fp_basin_rings.png",
    )
    plot_velocity_dense_probe_trajectories(
        summary=summary,
        path=figure_dir / "velocity_dense_probe_trajectories.png",
    )
    plot_velocity_phase_flow_diagnostics(
        summary=summary,
        path=figure_dir / "velocity_phase_flow_diagnostics.png",
    )


def _load_group_history(
    *,
    run_dir: Path,
    filename: str,
    group_name: str,
):
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"diagnostic group {group_name!r} requires {filename}"
        )
    return load_npz(path)


def _make_grouped_diagnostic_figures(
    *,
    run_dir: Path,
    resolved_config,
    figure_dir: Path,
    groups: frozenset[str] | None = None,
) -> None:
    """Plot only the diagnostic groups selected by the hyper config."""

    # The plotting sections historically used five presentation-based output
    # names.  They deliberately alias the one config-group directory here so
    # every artifact from a job remains colocated with its diagnostics switch.
    activity_dir = figure_dir
    heading_dir = figure_dir
    weights_dir = figure_dir
    gain_dir = figure_dir
    diagnostics_dir = figure_dir

    groups = (
        selected_diagnostic_groups(resolved_config)
        if groups is None
        else groups
    )
    if not groups:
        return
    final_metrics_path = run_dir / "test_metrics.json"
    partial_metrics_path = run_dir / "test_metrics.partial.json"
    test_metrics = (
        load_json(final_metrics_path)
        if final_metrics_path.exists()
        else (
            load_json(partial_metrics_path)
            if partial_metrics_path.exists()
            else {}
        )
    )
    diagnostic_weights_path = run_dir / "diagnostic_weights.npz"
    trained_weights = load_npz(
        diagnostic_weights_path
        if diagnostic_weights_path.exists()
        else run_dir / "trained_weights.npz"
    )
    theta_hd_pref = np.asarray(
        trained_weights.get("theta_hd_pref", np.empty(0)),
        dtype=float,
    )
    firing_rate_label = (
        "HD firing rate [kHz]"
        if np.isclose(resolved_config.model.activation.max_rate, 0.15)
        else "normalized HD firing rate [a.u.]"
    )

    if "training_convergence" in groups:
        training_history = _load_group_history(
            run_dir=run_dir,
            filename="training_history.npz",
            group_name="training_convergence",
        )
        if (
            "absolute_learning_error_time" in training_history
            and "absolute_learning_error_mean_spikes_per_s" in training_history
        ):
            plot_training_absolute_learning_error(
                time=training_history["absolute_learning_error_time"],
                mean_absolute_error_spikes_per_s=training_history[
                    "absolute_learning_error_mean_spikes_per_s"
                ],
                window_duration=float(
                    training_history["absolute_learning_error_window_duration"]
                ),
                path=(
                    diagnostics_dir
                    / "training_absolute_learning_error.png"
                ),
            )

    if "bump_maintenance" in groups:
        history = _load_group_history(
            run_dir=run_dir,
            filename="bump_history.npz",
            group_name="bump_maintenance",
        )
        plot_activity_heatmap(
            r_hd_history=history["r_hd"],
            time=history["time"],
            path=activity_dir / "bump_maintenance_hd_activity_heatmap.png",
            title="Bump maintenance (cue, then darkness)",
            theta_hd_pref=theta_hd_pref,
            theta_hd_decoded=history.get("theta_hd_decoded"),
            theta_hd_decoded_peak=history.get("theta_hd_decoded_peak"),
            decode_theta_hd_pref=theta_hd_pref,
            phase_id=history.get("phase_id"),
            firing_rate_label=firing_rate_label,
        )
        plot_hd_current_heatmap(
            current_history=history.get("i_vis_to_hd", np.empty((0, 0))),
            time=history["time"],
            path=activity_dir / "bump_maintenance_visual_input_heatmap.png",
            title="Bump maintenance visual input",
            theta_hd_pref=theta_hd_pref,
            theta_true=history.get("theta_true"),
            phase_id=history.get("phase_id"),
            colorbar_label="I_vis to HD [current]",
        )
        plot_true_vs_decoded_heading(
            time=history["time"],
            theta_true=history["theta_true"],
            theta_hd_decoded=history["theta_hd_decoded"],
            theta_hd_decoded_peak=history.get("theta_hd_decoded_peak"),
            path=heading_dir / "bump_maintenance_decoded_heading.png",
            title="Bump maintenance (cue, then darkness)",
        )

    if groups & {
        "path_integration_and_pi_error",
        "path_integration_constant",
    }:
        darkness_history = _load_group_history(
            run_dir=run_dir,
            filename="darkness_history.npz",
            group_name="path_integration_and_pi_error",
        )
        darkness_error = _plot_pi_error_trace(darkness_history)
        plot_activity_heatmap(
            r_hd_history=darkness_history["r_hd"],
            time=darkness_history["time"],
            path=activity_dir / "darkness_hd_activity_heatmap.png",
            title="Constant-velocity PI HD activity",
            theta_hd_pref=theta_hd_pref,
            theta_hd_decoded=darkness_history.get("theta_hd_decoded"),
            theta_hd_decoded_peak=darkness_history.get(
                "theta_hd_decoded_peak"
            ),
            decode_theta_hd_pref=theta_hd_pref,
            phase_id=darkness_history.get("phase_id"),
            firing_rate_label=firing_rate_label,
        )
        plot_hd_current_heatmap(
            current_history=darkness_history.get(
                "i_vis_to_hd", np.empty((0, 0))
            ),
            time=darkness_history["time"],
            path=activity_dir / "darkness_visual_input_heatmap.png",
            title="Constant-velocity PI visual input",
            theta_hd_pref=theta_hd_pref,
            theta_true=darkness_history.get("theta_true"),
            phase_id=darkness_history.get("phase_id"),
            colorbar_label="I_vis to HD [current]",
        )
        plot_heading_and_pi_error_panels(
            time=darkness_history["time"],
            theta_true=darkness_history["theta_true"],
            theta_hd_decoded=darkness_history["theta_hd_decoded"],
            theta_hd_decoded_peak=darkness_history.get("theta_hd_decoded_peak"),
            pi_error=darkness_error,
            peak_pi_error=darkness_history.get(
                "peak_pi_error_release_relative"
            ),
            circular_error_axis=(
                "pi_error_release_relative" not in darkness_history
            ),
            phase_id=darkness_history.get("phase_id"),
            path=heading_dir / "darkness_heading_and_pi_error.png",
            title="Constant-velocity path integration",
        )
        plot_pi_error(
            time=darkness_history["time"],
            pi_error=darkness_error,
            phase_id=darkness_history.get("phase_id"),
            path=heading_dir / "darkness_pi_error.png",
            title="Constant-velocity PI error",
            circular_axis=(
                "pi_error_release_relative" not in darkness_history
            ),
        )
        constant_pi_keys = {
            "constant_pi_time",
            "constant_pi_error",
            "constant_pi_commanded_velocity",
            "constant_pi_decoded_velocity",
        }
        if constant_pi_keys.issubset(darkness_history):
            plot_constant_velocity_pi_error_grid(
                time=darkness_history["constant_pi_time"],
                pi_error=darkness_history["constant_pi_error"],
                commanded_velocity=darkness_history[
                    "constant_pi_commanded_velocity"
                ],
                decoded_velocity=darkness_history[
                    "constant_pi_decoded_velocity"
                ],
                phase_id=darkness_history.get("constant_pi_phase_id"),
                path=heading_dir / "constant_velocity_pi_error_grid.png",
            )
    if groups & {
        "path_integration_and_pi_error",
        "path_integration_ou",
    }:
        ou_history = _load_group_history(
            run_dir=run_dir,
            filename="ou_darkness_history.npz",
            group_name="path_integration_and_pi_error",
        )
        ou_error = _plot_pi_error_trace(ou_history)
        plot_activity_heatmap(
            r_hd_history=ou_history["r_hd"],
            time=ou_history["time"],
            path=activity_dir / "ou_darkness_hd_activity_heatmap.png",
            title="OU PI HD activity",
            theta_hd_pref=theta_hd_pref,
            theta_hd_decoded=ou_history.get("theta_hd_decoded"),
            theta_hd_decoded_peak=ou_history.get("theta_hd_decoded_peak"),
            decode_theta_hd_pref=theta_hd_pref,
            phase_id=ou_history.get("phase_id"),
            firing_rate_label=firing_rate_label,
        )
        plot_hd_current_heatmap(
            current_history=ou_history.get("i_vis_to_hd", np.empty((0, 0))),
            time=ou_history["time"],
            path=activity_dir / "ou_darkness_visual_input_heatmap.png",
            title="OU PI visual input",
            theta_hd_pref=theta_hd_pref,
            theta_true=ou_history.get("theta_true"),
            phase_id=ou_history.get("phase_id"),
            colorbar_label="I_vis to HD [current]",
        )
        plot_heading_and_pi_error_panels(
            time=ou_history["time"],
            theta_true=ou_history["theta_true"],
            theta_hd_decoded=ou_history["theta_hd_decoded"],
            theta_hd_decoded_peak=ou_history.get("theta_hd_decoded_peak"),
            pi_error=ou_error,
            peak_pi_error=ou_history.get("peak_pi_error_release_relative"),
            circular_error_axis=(
                "pi_error_release_relative" not in ou_history
            ),
            phase_id=ou_history.get("phase_id"),
            path=heading_dir / "ou_darkness_heading_and_pi_error.png",
            title="OU path integration",
        )
        plot_pi_error(
            time=ou_history["time"],
            pi_error=ou_error,
            phase_id=ou_history.get("phase_id"),
            path=heading_dir / "ou_darkness_pi_error.png",
            title="OU PI error",
            circular_axis=("pi_error_release_relative" not in ou_history),
        )
    if groups & {
        "path_integration_and_pi_error",
        "path_integration_ensemble",
    }:
        ensemble = _load_group_history(
            run_dir=run_dir,
            filename="ou_pi_ensemble_history.npz",
            group_name="path_integration_and_pi_error",
        )
        plot_pi_error_ensemble(
            time=ensemble["time"],
            pi_error_mean=ensemble["pva_pi_error_mean"],
            pi_error_sem=ensemble["pva_pi_error_sem"],
            systematic_drift_velocity=float(
                test_metrics.get("ou_pi_ensemble_systematic_drift_velocity", np.nan)
            ),
            drift_intercept=float(
                test_metrics.get("ou_pi_ensemble_drift_intercept", np.nan)
            ),
            n_trials=int(test_metrics.get("ou_pi_ensemble_n_trials", 0)),
            path=heading_dir / "ou_pi_error_ensemble.png",
        )

    if "pva_spectrum_and_visualization" in groups:
        tuning = _load_group_history(
            run_dir=run_dir,
            filename="hd_tuning_history.npz",
            group_name="pva_spectrum_and_visualization",
        )
        preferred_direction = np.asarray(
            tuning.get("empirical_preferred_direction", theta_hd_pref),
            dtype=float,
        )
        preferred_direction = np.where(
            np.isfinite(preferred_direction),
            preferred_direction,
            theta_hd_pref,
        )
        plot_single_neuron_hd_tuning_curves(
            theta_true=tuning["theta_true"],
            r_hd_by_heading=tuning["r_hd"],
            preferred_direction=preferred_direction,
            path=activity_dir / "single_neuron_hd_tuning_curves.png",
            sample_count=resolved_config.tests.hd_tuning_curve_sample_count,
            seed=resolved_config.simulation.seed + 70_000,
        )
        tuning_summary = summarize_com_aligned_tuning_curves(
            theta_true=tuning["theta_true"],
            r_hd_by_heading=tuning["r_hd"],
        )
        save_npz(run_dir / "hd_tuning_com_aligned.npz", **tuning_summary)
        plot_com_aligned_hd_tuning_population(
            theta_aligned=tuning_summary["theta_aligned"],
            r_hd_peak_normalized_com_aligned=tuning_summary[
                "r_hd_peak_normalized_com_aligned"
            ],
            population_mean=tuning_summary[
                "r_hd_peak_normalized_com_aligned_mean"
            ],
            population_std=tuning_summary[
                "r_hd_peak_normalized_com_aligned_std"
            ],
            path=activity_dir / "com_aligned_hd_tuning_population.png",
        )
        if "r_hd_visual_only" in tuning:
            visual_only_summary = summarize_com_aligned_tuning_curves(
                theta_true=tuning["theta_true"],
                r_hd_by_heading=tuning["r_hd_visual_only"],
            )
            plot_hd_tuning_stage_comparison(
                theta_aligned=tuning_summary["theta_aligned"],
                visual_only_mean=visual_only_summary[
                    "r_hd_peak_normalized_com_aligned_mean"
                ],
                visual_only_std=visual_only_summary[
                    "r_hd_peak_normalized_com_aligned_std"
                ],
                post_training_mean=tuning_summary[
                    "r_hd_peak_normalized_com_aligned_mean"
                ],
                post_training_std=tuning_summary[
                    "r_hd_peak_normalized_com_aligned_std"
                ],
                path=activity_dir / "hd_tuning_visual_only_vs_post_training.png",
            )
        training_history = _load_group_history(
            run_dir=run_dir,
            filename="training_history.npz",
            group_name="pva_spectrum_and_visualization",
        )
        plot_hd_current_heatmap(
            current_history=training_history.get(
                "i_vis_to_hd", np.empty((0, 0))
            ),
            time=training_history.get("time", np.empty(0)),
            path=activity_dir / "training_visual_input_heatmap.png",
            title="Training visual input to HD",
            theta_hd_pref=preferred_direction,
            theta_true=training_history.get("theta_true"),
            colorbar_label="I_vis to HD [current]",
        )
        slow = _load_group_history(
            run_dir=run_dir,
            filename="slow_manifold_diagnostics.npz",
            group_name="pva_spectrum_and_visualization",
        )
        if np.asarray(slow.get("candidate_theta", np.empty(0))).size:
            plot_slow_manifold_diagnostics(
                history=slow,
                path=diagnostics_dir / "slow_manifold_diagnostics.png",
            )
        if np.asarray(slow.get("ramesan_probe_phase", np.empty(0))).size:
            plot_ramesan_firing_rate_diagnostics(
                history=slow,
                path=diagnostics_dir / "ramesan_firing_rate_diagnostics.png",
            )
        if np.asarray(
            slow.get("ramesan_pca_explained_variance_spectrum", np.empty(0))
        ).size:
            plot_ramesan_pca_variance_rank(
                history=slow,
                path=diagnostics_dir / "ramesan_pca_variance_rank.png",
            )
        if np.asarray(slow.get("ramesan_phase_bin_center", np.empty(0))).size:
            plot_ramesan_phase_landscape(
                history=slow,
                path=(
                    diagnostics_dir
                    / "ramesan_phase_slow_regions_and_effective_potential.png"
                ),
            )

    if "velocity_gain" in groups:
        history = _load_group_history(
            run_dir=run_dir,
            filename="velocity_gain_history.npz",
            group_name="velocity_gain",
        )
        plot_velocity_gain_curve(
            commanded_velocity=history["commanded_velocity"],
            decoded_velocity=history["decoded_velocity"],
            decoded_velocity_peak=history.get("decoded_velocity_peak"),
            decoded_velocity_visual=history.get("decoded_velocity_visual"),
            decoded_velocity_visual_peak=history.get(
                "decoded_velocity_visual_peak"
            ),
            path=gain_dir / "velocity_gain_curve.png",
            title="Velocity gain curve (visual cue vs darkness)",
        )

    if "trajectory_and_fixed_points" in groups:
        history = _load_group_history(
            run_dir=run_dir,
            filename="bump_attractor_trajectory_history.npz",
            group_name="trajectory_and_fixed_points",
        )
        plot_bump_attractor_decoder_trajectories(
            time=history["time"],
            theta_initial=history["theta_initial"],
            theta_pva=history["theta_pva"],
            theta_peak=history["theta_peak"],
            theta_overlap=history["theta_overlap"],
            cue_time=history.get("cue_time"),
            cue_theta_pva=history.get("cue_theta_pva"),
            cue_theta_peak=history.get("cue_theta_peak"),
            cue_theta_overlap=history.get("cue_theta_overlap"),
            endpoint_probe_theta_initial=history.get(
                "endpoint_probe_theta_initial"
            ),
            endpoint_probe_refinement_level=history.get(
                "endpoint_probe_refinement_level"
            ),
            endpoint_probe_theta_release_pva=history.get(
                "endpoint_probe_theta_release_pva"
            ),
            endpoint_probe_theta_release_peak=history.get(
                "endpoint_probe_theta_release_peak"
            ),
            endpoint_probe_theta_release_overlap=history.get(
                "endpoint_probe_theta_release_overlap"
            ),
            endpoint_probe_theta_final_pva=history.get(
                "endpoint_probe_theta_final_pva"
            ),
            endpoint_probe_theta_final_peak=history.get(
                "endpoint_probe_theta_final_peak"
            ),
            endpoint_probe_theta_final_overlap=history.get(
                "endpoint_probe_theta_final_overlap"
            ),
            path=heading_dir / "bump_attractor_decoder_trajectories.png",
        )
        _plot_bump_attractor_pva_figure_set(
            history=history,
            figure_dir=figure_dir,
        )

    if "weight_snapshots_and_development" in groups:
        w_hd_to_hd = trained_weights["w_hd_to_hd"]
        w_hr_to_hd = trained_weights["w_hr_to_hd"]
        plot_weight_matrices_side_by_side(
            w_hd_to_hd=w_hd_to_hd,
            w_hr_to_hd=w_hr_to_hd,
            path=weights_dir / "training_weight_matrices_side_by_side.png",
            title="Trained weights (model index order)",
        )
        weight_history = _load_group_history(
            run_dir=run_dir,
            filename="weight_history.npz",
            group_name="weight_snapshots_and_development",
        )
        plot_weight_snapshot_grid(
            weight_history=weight_history.get(
                "w_hd_to_hd", np.empty((0, 0, 0))
            ),
            time=weight_history.get("time", np.empty(0)),
            path=weights_dir / "training_weight_hd_to_hd_over_time.png",
            title="HD-to-HD weights across training",
            x_label="source HD neuron ID",
            y_label="target HD neuron ID",
        )
        plot_weight_snapshot_grid(
            weight_history=weight_history.get(
                "w_hr_to_hd", np.empty((0, 0, 0))
            ),
            time=weight_history.get("time", np.empty(0)),
            path=weights_dir / "training_weight_hr_to_hd_over_time.png",
            title="HR-to-HD weights across training",
            x_label="source HR neuron ID",
            y_label="target HD neuron ID",
        )
        training_history = _load_group_history(
            run_dir=run_dir,
            filename="training_history.npz",
            group_name="weight_snapshots_and_development",
        )
        pathway_keys = {
            "time",
            "rms_i_hd_from_hd",
            "rms_i_hd_from_lhr",
            "rms_i_hd_from_rhr",
        }
        pathway_plot_kwargs = (
            {
                "pathway_time": training_history["time"],
                "rms_i_hd_from_hd": training_history["rms_i_hd_from_hd"],
                "rms_i_hd_from_lhr": training_history["rms_i_hd_from_lhr"],
                "rms_i_hd_from_rhr": training_history["rms_i_hd_from_rhr"],
            }
            if pathway_keys.issubset(training_history)
            else {}
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
            **pathway_plot_kwargs,
        )
        snapshot_pi_history = _load_group_history(
            run_dir=run_dir,
            filename="weight_snapshot_pi_development.npz",
            group_name="weight_snapshots_and_development",
        )
        plot_weight_snapshot_pi_development(
            snapshot_time=snapshot_pi_history["snapshot_time"],
            commanded_velocity=snapshot_pi_history["commanded_velocity"],
            time_averaged_abs_pi_error=snapshot_pi_history[
                "time_averaged_abs_pi_error"
            ],
            aggregate_time_averaged_abs_pi_error=snapshot_pi_history[
                "aggregate_time_averaged_abs_pi_error"
            ],
            aggregate_rms_pi_error=snapshot_pi_history[
                "aggregate_rms_pi_error"
            ],
            minimum_pva_strength=snapshot_pi_history["minimum_pva_strength"],
            best_snapshot_time=float(snapshot_pi_history["best_snapshot_time"]),
            aggregate_rms_velocity_bias=snapshot_pi_history.get(
                "aggregate_rms_velocity_bias"
            ),
            maximum_abs_velocity_bias=snapshot_pi_history.get(
                "maximum_abs_velocity_bias"
            ),
            depinning_velocity=snapshot_pi_history.get("depinning_velocity"),
            snapshot_acceptance_passed=snapshot_pi_history.get(
                "snapshot_acceptance_passed"
            ),
            selection_was_fallback=(
                bool(float(snapshot_pi_history["selection_was_fallback"]))
                if "selection_was_fallback" in snapshot_pi_history
                else None
            ),
            effective_weight_growth_hd_to_hd=snapshot_pi_history.get(
                "effective_weight_growth_hd_to_hd"
            ),
            effective_weight_growth_hr_to_hd=snapshot_pi_history.get(
                "effective_weight_growth_hr_to_hd"
            ),
            effective_weight_norm_hr_to_hd_over_hd_to_hd=(
                snapshot_pi_history.get(
                    "effective_weight_norm_hr_to_hd_over_hd_to_hd"
                )
            ),
            selection_metric=str(
                snapshot_pi_history.get(
                    "selection_metric",
                    np.asarray("mean_abs_unwrapped_error"),
                )
            ),
            path=weights_dir / "training_snapshot_frozen_pi_error.png",
        )

    if "numerical_convergence" in groups:
        history = _load_group_history(
            run_dir=run_dir,
            filename="numerical_convergence_history.npz",
            group_name="numerical_convergence",
        )
        plot_numerical_convergence_diagnostics(
            time=history["time"],
            dt=history["dt"],
            integration_method=history["integration_method"],
            heading_error=history["heading_error"],
            rate_rms_error=history["rate_rms_error"],
            max_abs_heading_error=history["max_abs_heading_error"],
            max_rate_rms_error=history["max_rate_rms_error"],
            convergence_passed=history["convergence_passed"],
            path=diagnostics_dir / "numerical_convergence.png",
        )

    if "bump_diffusion" in groups:
        history = _load_group_history(
            run_dir=run_dir,
            filename="bump_diffusion_history.npz",
            group_name="bump_diffusion",
        )
        anomalous = fit_anomalous_diffusion_power_law(
            time=history["time"],
            displacement_variance=history["pva_displacement_variance"],
            fit_start_time=float(resolved_config.tests.bump_diffusion_fit_start_time),
            fit_end_time=resolved_config.tests.bump_diffusion_fit_end_time,
        )
        for summary_name, metric_name in {
            "anomalous_diffusion_exponent": (
                "bump_ensemble_anomalous_diffusion_exponent"
            ),
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
        }.items():
            test_metrics[metric_name] = float(anomalous[summary_name])
        rad2_to_deg2 = float(np.rad2deg(1.0) ** 2)
        if "bump_ensemble_diffusion_coefficient" in test_metrics:
            test_metrics["bump_ensemble_diffusion_coefficient_deg2_s"] = float(
                rad2_to_deg2
                * test_metrics["bump_ensemble_diffusion_coefficient"]
            )
        test_metrics[
            "bump_ensemble_generalized_diffusion_coefficient_deg2_s_alpha"
        ] = float(
            rad2_to_deg2
            * test_metrics["bump_ensemble_generalized_diffusion_coefficient"]
        )
        save_json(run_dir / "test_metrics.json", test_metrics)
        plot_ensemble_diffusion_variance(
            time=history["time"],
            displacement_mean=history["pva_displacement_mean"],
            displacement_variance=history["pva_displacement_variance"],
            diffusion_coefficient=float(
                test_metrics.get("bump_ensemble_diffusion_coefficient", np.nan)
            ),
            systematic_drift_velocity=float(
                test_metrics.get("bump_ensemble_systematic_drift_velocity", np.nan)
            ),
            anomalous_diffusion_fit=np.asarray(
                anomalous["anomalous_diffusion_fit_trace"]
            ),
            anomalous_diffusion_exponent=float(
                anomalous["anomalous_diffusion_exponent"]
            ),
            generalized_diffusion_coefficient=float(
                anomalous["generalized_diffusion_coefficient"]
            ),
            anomalous_diffusion_log_r_squared=float(
                anomalous["anomalous_diffusion_log_r_squared"]
            ),
            anomalous_diffusion_fit_start_time=float(
                anomalous["anomalous_diffusion_fit_start_time"]
            ),
            anomalous_diffusion_fit_end_time=float(
                anomalous["anomalous_diffusion_fit_end_time"]
            ),
            n_trials=(
                int(test_metrics["bump_ensemble_diffusion_n_trials"])
                if "bump_ensemble_diffusion_n_trials" in test_metrics
                else None
            ),
            path=diagnostics_dir / "bump_ensemble_diffusion_variance.png",
        )
        if (
            "test_noise_std" in history
            and "pva_endpoint_diffusion_coefficient" in history
        ):
            plot_diffusion_noise_sweep(
                noise_std=history["test_noise_std"],
                diffusion_coefficient=history[
                    "pva_endpoint_diffusion_coefficient"
                ],
                path=diagnostics_dir / "bump_diffusion_vs_test_noise.png",
            )

    if "timescale_separation" in groups:
        history = _load_group_history(
            run_dir=run_dir,
            filename="timescale_separation_history.npz",
            group_name="timescale_separation",
        )
        plot_timescale_separation_diagnostics(
            normal_time=history["normal_time"],
            perturbation_scale=history["perturbation_scale"],
            normal_distance_to_manifold=history["normal_distance_to_manifold"],
            normal_control_distance_to_manifold=history[
                "normal_control_distance_to_manifold"
            ],
            normal_e_folding_time=history["normal_e_folding_time"],
            normal_recovery_observed=history["normal_recovery_observed"],
            tangential_time=history["tangential_time"],
            tangential_overlap_displacement=history[
                "tangential_overlap_displacement"
            ],
            tangential_first_passage_time=history[
                "tangential_first_passage_time"
            ],
            tangential_first_passage_observed=history[
                "tangential_first_passage_observed"
            ],
            tangential_threshold_rad=float(history["tangential_threshold_rad"]),
            normal_time_p90=float(history["normal_time_p90"]),
            tangential_time_p10=float(history["tangential_time_p10"]),
            conservative_timescale_ratio=float(
                history["conservative_timescale_ratio"]
            ),
            criterion_ratio_threshold=float(history["criterion_ratio_threshold"]),
            criterion_passed=bool(float(history["criterion_passed"])),
            criterion_ratio_is_lower_bound=bool(
                float(history["criterion_ratio_is_lower_bound"])
            ),
            path=heading_dir / "timescale_separation_diagnostics.png",
        )

    if "velocity_dynamics_and_phase_flow" in groups:
        history = _load_group_history(
            run_dir=run_dir,
            filename="velocity_trajectory_sweep_history.npz",
            group_name="velocity_dynamics_and_phase_flow",
        )
        plot_velocity_trajectory_sweep(
            time=history["time"],
            commanded_velocity=history["commanded_velocity"],
            theta_initial=history["theta_initial"],
            pva_angular_displacement=history["pva_angular_displacement"],
            overlap_angular_displacement=history["overlap_angular_displacement"],
            decoded_velocity_pva=history["decoded_velocity_pva"],
            decoded_velocity_overlap=history["decoded_velocity_overlap"],
            path=heading_dir / "velocity_trajectory_sweep.png",
        )
        summary = summarize_velocity_phase_flows(
            velocity_history=history,
            angular_bin_count=int(resolved_config.tests.velocity_phase_flow_angular_bins),
            smoothing_bin_count=int(
                resolved_config.tests.velocity_phase_flow_smoothing_bins
            ),
            empirical_lambda_speed_floor=float(
                resolved_config.tests.velocity_phase_flow_empirical_lambda_speed_floor
            ),
        )
        save_npz(run_dir / "velocity_phase_flow_summary.npz", **summary)
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


def make_vafidis_figures_for_run(*, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"

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
    if resolved_config is None:
        raise FileNotFoundError("run is missing a resolved config")
    selected_groups = selected_diagnostic_groups(resolved_config)
    figure_jobs: list[tuple[str, str, str]] = []
    for group_name in sorted(selected_groups):
        if group_name == "path_integration_and_pi_error":
            figure_jobs.extend(
                [
                    (
                        "path_integration.constant",
                        "path_integration_constant",
                        group_name,
                    ),
                    (
                        "path_integration.ou",
                        "path_integration_ou",
                        group_name,
                    ),
                    (
                        "path_integration.ensemble",
                        "path_integration_ensemble",
                        group_name,
                    ),
                ]
            )
        else:
            figure_jobs.append((group_name, group_name, group_name))

    figure_status: dict[str, dict] = {"figures": {}}

    def flush_figure_status() -> None:
        entries = figure_status["figures"]
        figure_status["completed_count"] = sum(
            entry["status"] == "completed" for entry in entries.values()
        )
        figure_status["failed_count"] = sum(
            entry["status"] == "failed" for entry in entries.values()
        )
        save_json(run_dir / "figure_status.json", figure_status)

    flush_figure_status()
    for status_name, internal_group_name, output_group_name in figure_jobs:
        output_dir = f"figures/{output_group_name}"
        figure_status["figures"][status_name] = {
            "status": "running",
            "output_dir": output_dir,
        }
        flush_figure_status()
        try:
            figure_dir = _diagnostic_figure_dir(
                figures_dir=figures_dir,
                group_name=output_group_name,
            )
            _make_grouped_diagnostic_figures(
                run_dir=run_dir,
                resolved_config=resolved_config,
                figure_dir=figure_dir,
                groups=frozenset({internal_group_name}),
            )
        except Exception as error:
            figure_status["figures"][status_name] = {
                "status": "failed",
                "output_dir": output_dir,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        else:
            figure_status["figures"][status_name] = {
                "status": "completed",
                "output_dir": output_dir,
            }
        flush_figure_status()
    return figure_status


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
