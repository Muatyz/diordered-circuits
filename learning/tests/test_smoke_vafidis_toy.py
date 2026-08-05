from __future__ import annotations

import numpy as np

from learning.common.random import make_rng
from learning.config.diagnostics import DIAGNOSTIC_GROUPS
from learning.config.load_config import load_experiment_config, save_yaml
from learning.config.schema import ExperimentConfig
from learning.experiments.run_vafidis_toy import (
    count_local_hd_peaks,
    run_bump_attractor_trajectory_test,
    run_bump_diffusion_ensemble_test,
    run_all_tests,
    run_hd_tuning_curve_test,
    run_experiment,
    run_training,
)
from learning.io.save_load import load_json, load_npz
from learning.experiments.test_vafidis_toy import run_tests_for_existing_run
from learning.models.vafidis_toy import VafidisToyParams, initialize_vafidis_toy_state, step_vafidis_toy


def test_count_local_hd_peaks_uses_circular_neighbors() -> None:
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    rates = np.asarray([1.0, 0.1, 0.8, 0.1, 0.1, 0.7, 0.1, 0.2])
    assert count_local_hd_peaks(theta_hd_pref=theta, r_hd=rates) == 3


def make_short_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.n_theta = 12
    config.model.n_hr = 12
    config.simulation.seed = 3
    config.simulation.dt = 0.0005
    config.simulation.train_duration = 0.05
    config.simulation.bump_test_duration = 0.03
    config.simulation.darkness_test_duration = 0.03
    config.simulation.cue_duration = 0.02
    config.simulation.pi_cue_duration = 0.04
    config.simulation.recue_duration = 0.02
    # Preserve the old 10 ms record and 20 ms weight-snapshot cadence after
    # adopting the paper's stable 0.5 ms integration step.
    config.simulation.save_interval_steps = 20
    config.simulation.weight_snapshot_interval_steps = 40
    config.simulation.progress = False
    config.tests.bump_diffusion_duration = 0.03
    config.tests.bump_diffusion_trials = 4
    config.tests.bump_diffusion_test_noise_std = 0.1
    config.diagnostics.bump_maintenance = True
    config.diagnostics.path_integration_and_pi_error = True
    config.diagnostics.pva_spectrum_and_visualization = True
    config.diagnostics.velocity_gain = True
    config.diagnostics.trajectory_and_fixed_points = True
    config.diagnostics.weight_snapshots_and_development = True
    config.diagnostics.bump_diffusion = True
    config.diagnostics.timescale_separation = True
    config.diagnostics.velocity_dynamics_and_phase_flow = True
    config.tests.bump_attractor_initial_conditions = 8
    config.tests.bump_attractor_duration = 0.02
    config.tests.bump_attractor_cue_duration = 0.01
    config.tests.bump_attractor_sample_interval = 0.01
    config.tests.timescale_separation_initial_conditions = 2
    config.tests.timescale_separation_normal_duration = 0.02
    config.tests.timescale_separation_sample_interval = 0.01
    config.tests.timescale_separation_perturbation_scales = [0.05]
    config.tests.timescale_separation_perturbations_per_condition = 1
    config.tests.velocity_trajectory_sweep_velocities = [0.0, 0.05, 0.2]
    config.tests.velocity_trajectory_sweep_max_velocity = 0.2
    config.tests.velocity_trajectory_sweep_count = 3
    config.tests.velocity_trajectory_sweep_initial_conditions = 2
    config.tests.velocity_trajectory_sweep_duration = 0.02
    config.tests.velocity_trajectory_sweep_cue_duration = 0.01
    config.tests.velocity_trajectory_sweep_sample_interval = 0.01
    config.tests.velocity_trajectory_sweep_fit_start_time = 0.0
    config.tests.velocity_phase_flow_initial_conditions = 4
    config.tests.velocity_phase_flow_duration = 0.02
    config.tests.velocity_phase_flow_sample_interval = 0.01
    config.tests.velocity_phase_flow_fit_start_time = 0.0
    config.tests.velocity_phase_flow_fit_duration = 0.02
    config.tests.velocity_phase_flow_angular_bins = 8
    config.tests.velocity_phase_flow_smoothing_bins = 3
    config.tests.velocity_phase_flow_empirical_lambda_speed_floor = 0.02
    config.tests.velocity_phase_flow_probe_velocities = [0.0, 0.05, 0.2]
    config.tests.hd_tuning_curve_angles = 8
    config.tests.hd_tuning_curve_settle_duration = 0.01
    config.tests.hd_tuning_curve_sample_count = 2
    config.tests.ou_pi_ensemble_trials = 2
    config.tests.ou_pi_ensemble_fit_start_time = 0.0
    config.tests.gain_velocities = [-0.5, 0.5]
    return config


def test_explicit_trajectory_selection_skips_unrelated_diagnostics() -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    config.diagnostics.trajectory_and_fixed_points = True
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    outputs = run_all_tests(config=config, trained_state=trained_state)
    (
        hd_tuning_history,
        bump_history,
        trajectory_history,
        _slow_manifold_history,
        _timescale_history,
        _velocity_sweep_history,
        bump_diffusion_history,
        darkness_history,
        ou_darkness_history,
        ou_ensemble_history,
        velocity_gain_history,
        metrics,
    ) = outputs

    assert hd_tuning_history["theta_true"].size == config.tests.hd_tuning_curve_angles
    assert trajectory_history["theta_initial"].size == (
        config.tests.bump_attractor_initial_conditions
    )
    assert not bump_history
    assert not bump_diffusion_history
    assert not darkness_history
    assert not ou_darkness_history
    assert not ou_ensemble_history
    assert not velocity_gain_history
    assert metrics["hd_tuning_dependency_computed"] == 1.0
    assert metrics["bump_diffusion_diagnostic_enabled"] == 0.0


def test_testing_phase_freezes_weights() -> None:
    config = make_short_config()
    params = VafidisToyParams.from_config(config)
    state = initialize_vafidis_toy_state(config=config, rng=make_rng(config.simulation.seed))
    trained_state = step_vafidis_toy(
        state=state,
        params=params,
        angular_velocity=0.4,
        visual_teacher=True,
        training=True,
    )
    frozen_state = step_vafidis_toy(
        state=trained_state,
        params=params,
        angular_velocity=0.4,
        visual_teacher=False,
        training=False,
    )
    assert np.allclose(frozen_state.w_hd_to_hd, trained_state.w_hd_to_hd)
    assert np.allclose(frozen_state.w_hr_to_hd, trained_state.w_hr_to_hd)


def test_weight_snapshot_grid_includes_initial_periodic_and_final_states() -> None:
    config = make_short_config()
    _state, _training_history, weight_history = run_training(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    assert np.allclose(weight_history["time"], [0.0, 0.02, 0.04, 0.05])
    assert weight_history["w_hd_to_hd"].shape[0] == 4
    assert weight_history["w_hr_to_hd"].shape[0] == 4


def test_training_early_stopping_keeps_duration_as_a_hard_cap() -> None:
    config = make_short_config()
    config.simulation.train_duration = 0.08
    config.simulation.early_stopping.enabled = True
    config.simulation.early_stopping.min_duration = 0.02
    config.simulation.early_stopping.check_interval = 0.01
    config.simulation.early_stopping.window_duration = 0.01
    config.simulation.early_stopping.patience_checks = 1
    config.simulation.early_stopping.relative_weight_change_tolerance = 1e6
    config.simulation.early_stopping.relative_error_change_tolerance = 1e6

    _state, training_history, weight_history = run_training(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    assert np.isclose(training_history["training_requested_duration"], 0.08)
    assert np.isclose(training_history["training_actual_duration"], 0.02)
    assert np.isclose(training_history["training_early_stopped"], 1.0)
    assert np.isclose(training_history["time"][-1], 0.02)
    assert np.isclose(weight_history["time"][-1], 0.02)
    assert training_history["early_stopping_time"].size == 2
    assert np.isclose(
        training_history["early_stopping_consecutive_converged_checks"][-1],
        1.0,
    )


def test_population_mean_training_records_effective_pathway_diagnostics() -> None:
    config = make_short_config()
    config.model.hd_distal_normalization = "presynaptic_population_mean"
    config.simulation.random_stream_mode = "component_split"
    _state, training_history, weight_history = run_training(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    for pathway_name in ["hd", "lhr", "rhr"]:
        assert f"mean_i_hd_from_{pathway_name}" in training_history
        assert f"rms_i_hd_from_{pathway_name}" in training_history
    assert "rms_e_hd" in training_history
    assert "mean_r_hd" in training_history
    assert "effective_weight_norm_hd_to_hd" in weight_history
    assert "effective_weight_norm_hr_to_hd" in weight_history


def test_component_split_velocity_stream_is_independent_of_neuron_count() -> None:
    config_small = make_short_config()
    config_small.simulation.random_stream_mode = "component_split"
    config_small.model.init.w_hd_to_hd_mode = "random_normal"
    config_small.model.init.w_hr_to_hd_mode = "random_normal"
    config_large = make_short_config()
    config_large.simulation.random_stream_mode = "component_split"
    config_large.model.n_theta = 24
    config_large.model.n_hr = 24
    config_large.model.init.w_hd_to_hd_mode = "random_normal"
    config_large.model.init.w_hr_to_hd_mode = "random_normal"

    _small_state, small_history, _small_weights = run_training(
        config=config_small,
        rng=make_rng(config_small.simulation.seed),
    )
    _large_state, large_history, _large_weights = run_training(
        config=config_large,
        rng=make_rng(config_large.simulation.seed),
    )

    np.testing.assert_allclose(
        small_history["angular_velocity"],
        large_history["angular_velocity"],
    )


def test_training_protocol_can_freeze_weights_for_phase_1a_control() -> None:
    config = make_short_config()
    config.simulation.plasticity_enabled = False
    config.model.init.w_hd_to_hd_mode = "local_kernel"
    config.model.init.w_hr_to_hd_mode = "local_kernel"
    _state, _history, weight_history = run_training(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    np.testing.assert_allclose(
        weight_history["w_hd_to_hd"],
        np.broadcast_to(
            weight_history["w_hd_to_hd"][0],
            weight_history["w_hd_to_hd"].shape,
        ),
    )
    np.testing.assert_allclose(
        weight_history["w_hr_to_hd"],
        np.broadcast_to(
            weight_history["w_hr_to_hd"][0],
            weight_history["w_hr_to_hd"].shape,
        ),
    )


def test_hd_tuning_sweep_records_adaptive_settling_and_visual_only_stage() -> None:
    config = make_short_config()
    config.tests.hd_tuning_curve_settle_duration = 0.02
    config.tests.hd_tuning_curve_max_settle_duration = 0.06
    config.tests.hd_tuning_curve_convergence_window = 0.01
    config.tests.hd_tuning_curve_convergence_tolerance = 1.0
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    history, metrics = run_hd_tuning_curve_test(
        config=config,
        trained_state=trained_state,
    )

    assert history["r_hd_visual_only"].shape == history["r_hd"].shape
    np.testing.assert_allclose(history["actual_settle_duration"], 0.02)
    np.testing.assert_allclose(history["settle_converged"], 1.0)
    assert np.isclose(metrics["hd_tuning_curve_converged_fraction"], 1.0)
    assert "hd_tuning_post_vs_visual_only_median_correlation" in metrics


def test_bump_diffusion_ensemble_uses_progress_bar(monkeypatch) -> None:
    config = make_short_config()
    captured: dict[str, object] = {}

    def fake_trange(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return range(*args)

    monkeypatch.setattr("learning.experiments.run_vafidis_toy.trange", fake_trange)
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )
    _history, diffusion_metrics = run_bump_diffusion_ensemble_test(
        config=config,
        trained_state=trained_state,
    )

    assert captured["args"] == (config.tests.bump_diffusion_trials,)
    assert captured["kwargs"] == {
        "disable": True,
        "desc": "bump diffusion ensemble",
        "unit": "trial",
    }
    assert np.isclose(
        diffusion_metrics["bump_ensemble_diffusion_coefficient_deg2_s"],
        np.rad2deg(1.0) ** 2
        * diffusion_metrics["bump_ensemble_diffusion_coefficient"],
    )


def test_bump_attractor_progress_tracks_all_dynamics_steps(monkeypatch) -> None:
    config = make_short_config()
    config.simulation.progress = True
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )
    hd_tuning_history, _metrics = run_hd_tuning_curve_test(
        config=config,
        trained_state=trained_state,
    )
    captured: dict[str, object] = {"updated": 0, "postfix": []}

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback

        def update(self, count):
            captured["updated"] = int(captured["updated"]) + int(count)

        def set_postfix(self, *args, **kwargs):
            del args
            captured["postfix"].append(kwargs)

    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.tqdm",
        FakeProgress,
    )
    run_bump_attractor_trajectory_test(
        config=config,
        trained_state=trained_state,
        hd_tuning_history=hd_tuning_history,
    )

    cue_steps = round(config.tests.bump_attractor_cue_duration / config.simulation.dt)
    darkness_steps = round(config.tests.bump_attractor_duration / config.simulation.dt)
    expected_steps = config.tests.bump_attractor_initial_conditions * (
        cue_steps + darkness_steps
    )
    assert captured["kwargs"]["total"] == expected_steps
    assert captured["kwargs"]["disable"] is False
    assert captured["kwargs"]["unit"] == "step"
    assert captured["updated"] == expected_steps
    assert {entry["phase"] for entry in captured["postfix"]} == {
        "cue",
        "darkness",
    }


def test_short_experiment_writes_required_outputs(tmp_path) -> None:
    config = make_short_config()
    run_dir = run_experiment(config=config, project_root=tmp_path, make_figures=True)
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "params.json").exists()
    assert (run_dir / "trained_weights.npz").exists()
    assert (run_dir / "training_history.npz").exists()
    assert (run_dir / "weight_history.npz").exists()
    assert (run_dir / "bump_diffusion_history.npz").exists()
    assert (run_dir / "bump_attractor_trajectory_history.npz").exists()
    assert (run_dir / "timescale_separation_history.npz").exists()
    assert (run_dir / "velocity_trajectory_sweep_history.npz").exists()
    assert (run_dir / "hd_tuning_history.npz").exists()
    assert (run_dir / "hd_tuning_com_aligned.npz").exists()
    assert (run_dir / "ou_darkness_history.npz").exists()
    assert (run_dir / "ou_pi_ensemble_history.npz").exists()
    assert (run_dir / "test_metrics.json").exists()
    assert (run_dir / "figures" / "activity" / "training_visual_input_heatmap.png").exists()
    assert (run_dir / "figures" / "activity" / "bump_maintenance_visual_input_heatmap.png").exists()
    assert (
        run_dir
        / "figures"
        / "heading"
        / "bump_attractor_decoder_trajectories.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "heading"
        / "timescale_separation_diagnostics.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "heading"
        / "velocity_trajectory_sweep.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "heading"
        / "velocity_actual_fp_basin_rings.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "heading"
        / "velocity_dense_probe_trajectories.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "heading"
        / "velocity_phase_flow_diagnostics.png"
    ).exists()
    assert (run_dir / "velocity_phase_flow_summary.npz").exists()
    assert (run_dir / "figures" / "activity" / "darkness_visual_input_heatmap.png").exists()
    assert (run_dir / "figures" / "activity" / "ou_darkness_visual_input_heatmap.png").exists()
    assert (
        run_dir
        / "figures"
        / "activity"
        / "com_aligned_hd_tuning_population.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "activity"
        / "hd_tuning_visual_only_vs_post_training.png"
    ).exists()
    training_history = load_npz(run_dir / "training_history.npz")
    weight_history = load_npz(run_dir / "weight_history.npz")
    test_metrics = load_json(run_dir / "test_metrics.json")
    assert "pva_strength_hd" in training_history
    assert np.allclose(weight_history["time"], [0.0, 0.02, 0.04, 0.05])
    assert "bump_contrast_hd" in training_history
    assert "theta_hd_decoded_peak" in training_history
    assert "i_vis_to_hd" in training_history
    assert training_history["i_vis_to_hd"].shape == (
        training_history["time"].size,
        config.model.n_theta,
    )
    assert "contrast_r_lhr" in training_history
    assert "contrast_r_rhr" in training_history
    darkness_history = load_npz(run_dir / "darkness_history.npz")
    ou_darkness_history = load_npz(run_dir / "ou_darkness_history.npz")
    bump_history = load_npz(run_dir / "bump_history.npz")
    bump_diffusion_history = load_npz(run_dir / "bump_diffusion_history.npz")
    bump_attractor_history = load_npz(
        run_dir / "bump_attractor_trajectory_history.npz"
    )
    timescale_history = load_npz(run_dir / "timescale_separation_history.npz")
    velocity_trajectory_history = load_npz(
        run_dir / "velocity_trajectory_sweep_history.npz"
    )
    hd_tuning_history = load_npz(run_dir / "hd_tuning_history.npz")
    hd_tuning_com_aligned = load_npz(run_dir / "hd_tuning_com_aligned.npz")
    ou_pi_ensemble_history = load_npz(run_dir / "ou_pi_ensemble_history.npz")
    velocity_gain_history = load_npz(run_dir / "velocity_gain_history.npz")
    assert "phase_id" in darkness_history
    assert "visual_teacher" in darkness_history
    assert "i_vis_to_hd" in darkness_history
    assert "phase_id" in ou_darkness_history
    assert "i_vis_to_hd" in ou_darkness_history
    assert "i_vis_to_hd" in bump_history
    assert bump_diffusion_history["pva_angular_displacement"].shape[0] == 4
    assert "pva_displacement_variance" in bump_diffusion_history
    assert bump_attractor_history["theta_pva"].shape == (8, 3)
    assert bump_attractor_history["theta_peak"].shape == (8, 3)
    assert bump_attractor_history["theta_overlap"].shape == (8, 3)
    assert "pva_angular_displacement" in bump_attractor_history
    assert "peak_angular_displacement" in bump_attractor_history
    assert "overlap_angular_displacement" in bump_attractor_history
    assert "bump_attractor_overlap_trajectory_flatness_rmse_deg" in test_metrics
    assert timescale_history["normal_distance_to_manifold"].shape == (1, 2, 1, 3)
    assert "timescale_separation_conservative_ratio" in test_metrics
    assert velocity_trajectory_history["theta_pva"].shape == (3, 2, 3)
    np.testing.assert_allclose(
        velocity_trajectory_history["commanded_velocity"],
        [0.0, 0.05, 0.2],
    )
    assert np.isclose(velocity_trajectory_history["theta_initial"][0], 0.0)
    assert velocity_trajectory_history["phase_flow_theta_pva"].shape == (
        3,
        4,
        3,
    )
    np.testing.assert_allclose(
        velocity_trajectory_history["phase_flow_commanded_velocity"],
        [0.0, 0.05, 0.2],
    )
    assert "velocity_trajectory_sweep_estimated_depinning_threshold_rad_s" in test_metrics
    assert hd_tuning_history["r_hd"].shape == (8, config.model.n_theta)
    assert "empirical_preferred_direction" in hd_tuning_history
    assert "r_hd_visual_only" in hd_tuning_history
    assert hd_tuning_com_aligned["r_hd_peak_normalized_com_aligned"].shape == (
        config.model.n_theta,
        8,
    )
    np.testing.assert_allclose(
        np.max(
            hd_tuning_com_aligned["r_hd_peak_normalized_com_aligned"],
            axis=1,
        ),
        1.0,
    )
    assert str(hd_tuning_com_aligned["plot_normalization"]) == "per_neuron_peak"
    assert int(hd_tuning_com_aligned["simulated_mouse_count"]) == 1
    assert ou_pi_ensemble_history["pva_pi_error"].shape[0] == 2
    assert "ou_pi_ensemble_systematic_drift_velocity" in test_metrics
    assert np.count_nonzero(bump_history["phase_id"] == 0.0) == (
        int(round(config.simulation.cue_duration / config.simulation.dt)) + 1
    )
    assert np.count_nonzero(darkness_history["phase_id"] == 0.0) == (
        int(round(config.simulation.pi_cue_duration / config.simulation.dt)) + 1
    )
    assert np.count_nonzero(darkness_history["phase_id"] == 1.0) == int(
        round(config.simulation.darkness_test_duration / config.simulation.dt)
    )
    assert np.count_nonzero(darkness_history["phase_id"] == 2.0) == int(
        round(config.simulation.recue_duration / config.simulation.dt)
    )
    assert np.allclose(darkness_history["i_vis_to_hd"][darkness_history["phase_id"] == 1.0], 0.0)
    assert np.allclose(bump_history["i_vis_to_hd"][bump_history["phase_id"] == 1.0], 0.0)
    assert np.any(np.abs(darkness_history["i_vis_to_hd"][darkness_history["phase_id"] == 0.0]) > 0.0)
    assert "darkness_final_pva_strength" in test_metrics
    assert "darkness_final_bump_contrast" in test_metrics
    assert "visual_cue_hd_decode_rms_error" in test_metrics
    assert "darkness_hd_decode_rms_error" in test_metrics
    assert "darkness_hd_decode_final_abs_error" in test_metrics
    assert "darkness_mean_saturated_hd_bins" in test_metrics
    assert "bump_final_saturated_hd_bins" in test_metrics
    assert "darkness_mean_near_peak_hd_bins" in test_metrics
    assert "bump_final_near_peak_hd_bins" in test_metrics
    assert "darkness_recue_final_abs_pi_error" in test_metrics
    assert "ou_darkness_rms_pi_error" in test_metrics
    assert "ou_darkness_recue_final_abs_pi_error" in test_metrics
    assert "bump_intrinsic_drift_velocity_deg_s" in test_metrics
    assert "bump_effective_diffusion_coefficient" in test_metrics
    assert "bump_ensemble_diffusion_coefficient" in test_metrics
    assert "bump_ensemble_systematic_drift_velocity" in test_metrics
    assert "bump_ensemble_anomalous_diffusion_exponent" in test_metrics
    assert "bump_ensemble_generalized_diffusion_coefficient" in test_metrics
    assert "bump_ensemble_diffusion_coefficient_deg2_s" in test_metrics
    assert (
        "bump_ensemble_generalized_diffusion_coefficient_deg2_s_alpha"
        in test_metrics
    )
    assert np.isclose(
        test_metrics["bump_ensemble_diffusion_coefficient_deg2_s"],
        np.rad2deg(1.0) ** 2 * test_metrics["bump_ensemble_diffusion_coefficient"],
        equal_nan=True,
    )
    assert "bump_peak_transition_time_after_dark_onset" in test_metrics
    assert "bump_dark_final_top1_angle_deg" in test_metrics
    assert "bump_dark_final_top1_minus_top2_rate" in test_metrics
    assert "darkness_peak_decoded_velocity" in test_metrics
    assert "decoded_velocity_visual" in velocity_gain_history
    assert "decoded_velocity_darkness" in velocity_gain_history
    assert "decoded_velocity_visual_peak" in velocity_gain_history
    assert "decoded_velocity_darkness_peak" in velocity_gain_history
    assert "visual_velocity_gain" in test_metrics
    assert "darkness_velocity_gain" in test_metrics
    assert "darkness_minus_visual_velocity_gain" in test_metrics
    assert test_metrics["velocity_gain"] == test_metrics["darkness_velocity_gain"]


def test_short_experiment_with_ou_visual_noise(tmp_path) -> None:
    config = make_short_config()
    config.visual.noise_std = 0.15
    config.visual.noise_process = "ou_additive"
    config.visual.noise_correlation_time = 0.02
    config.visual.apply_noise_during_training = True
    config.visual.apply_noise_during_visual_test = True

    run_dir = run_experiment(config=config, project_root=tmp_path, make_figures=False)
    test_metrics = load_json(run_dir / "test_metrics.json")
    training_history = load_npz(run_dir / "training_history.npz")
    darkness_history = load_npz(run_dir / "darkness_history.npz")
    velocity_gain_history = load_npz(run_dir / "velocity_gain_history.npz")

    assert "visual_cue_hd_decode_rms_error" in test_metrics
    assert "darkness_hd_decode_rms_error" in test_metrics
    assert "decoded_velocity_visual" in velocity_gain_history
    assert "i_vis_to_hd" in training_history
    assert "i_vis_to_hd" in darkness_history
    assert np.any(
        np.abs(
            training_history["i_vis_to_hd"][:, 0::2]
            - training_history["i_vis_to_hd"][:, 1::2]
        )
        > 1e-6
    )
    assert np.allclose(darkness_history["i_vis_to_hd"][darkness_history["phase_id"] == 1.0], 0.0)


def test_existing_run_uses_group_switches_from_one_hyper_config(tmp_path) -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    run_dir = run_experiment(config=config, project_root=tmp_path, make_figures=False)
    diagnostics_path = tmp_path / "diagnostics.yaml"
    switches = {group_name: False for group_name in DIAGNOSTIC_GROUPS}
    switches["timescale_separation"] = True
    switches["velocity_dynamics_and_phase_flow"] = True
    save_yaml(
        diagnostics_path,
        {
            "diagnostics": switches,
            "tests": {
                "bump_attractor_initial_conditions": 12,
                "bump_attractor_duration": 0.04,
            },
        },
    )

    run_tests_for_existing_run(
        run_dir=run_dir,
        make_figures=False,
        diagnostics_config_path=diagnostics_path,
    )

    resolved_test_config = load_experiment_config(
        run_dir / "test_config_resolved.yaml"
    )
    assert resolved_test_config.tests.bump_attractor_initial_conditions == 12
    assert np.isclose(resolved_test_config.tests.bump_attractor_duration, 0.04)
    assert resolved_test_config.diagnostics.timescale_separation is True
    assert resolved_test_config.diagnostics.velocity_dynamics_and_phase_flow is True
    timescale_history = load_npz(run_dir / "timescale_separation_history.npz")
    assert timescale_history["normal_time"].size > 0
    velocity_trajectory_history = load_npz(
        run_dir / "velocity_trajectory_sweep_history.npz"
    )
    assert velocity_trajectory_history["time"].size > 0


def test_existing_run_can_apply_reusable_diagnostics_config(tmp_path) -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    run_dir = run_experiment(config=config, project_root=tmp_path, make_figures=False)
    diagnostics_path = tmp_path / "short_diagnostics.yaml"
    switches = {group_name: False for group_name in DIAGNOSTIC_GROUPS}
    switches["trajectory_and_fixed_points"] = True
    save_yaml(
        diagnostics_path,
        {
            "diagnostics": switches,
            "simulation": {"darkness_test_duration": 0.04},
            "tests": {"bump_attractor_initial_conditions": 10},
        },
    )

    run_tests_for_existing_run(
        run_dir=run_dir,
        make_figures=True,
        diagnostics_config_path=diagnostics_path,
    )

    resolved_test_config = load_experiment_config(
        run_dir / "test_config_resolved.yaml"
    )
    assert resolved_test_config.model.n_theta == config.model.n_theta
    assert np.isclose(
        resolved_test_config.simulation.train_duration,
        config.simulation.train_duration,
    )
    assert np.isclose(resolved_test_config.simulation.darkness_test_duration, 0.04)
    assert resolved_test_config.tests.bump_attractor_initial_conditions == 10
    assert resolved_test_config.diagnostics.trajectory_and_fixed_points is True
    assert resolved_test_config.diagnostics.bump_diffusion is False
    assert (
        run_dir
        / "figures"
        / "heading"
        / "bump_attractor_decoder_trajectories.png"
    ).exists()
    assert not (run_dir / "bump_diffusion_history.npz").exists()
    assert not (run_dir / "darkness_history.npz").exists()
