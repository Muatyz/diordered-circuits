from __future__ import annotations

import numpy as np
import pytest

import learning.analysis.make_vafidis_figures as figures_module
from learning.common.random import make_rng
from learning.config.diagnostics import DIAGNOSTIC_GROUPS
from learning.config.load_config import load_experiment_config, save_yaml
from learning.config.schema import ExperimentConfig
from learning.experiments.run_vafidis_toy import (
    IncrementalDiagnosticRecorder,
    count_local_hd_peaks,
    evaluate_training_checkpoint_behavior,
    run_constant_velocity_pi_error_sweep_test,
    run_bump_attractor_trajectory_test,
    run_bump_diffusion_ensemble_test,
    run_all_tests,
    run_hd_tuning_curve_test,
    run_experiment,
    run_training,
    resolve_weight_states,
)
from learning.io.save_load import load_json, load_npz, save_npz
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
    config.simulation.save_interval_duration = 0.01
    config.simulation.weight_snapshot_interval_duration = 0.02
    config.simulation.progress = False
    config.tests.bump_diffusion_duration = 0.03
    config.tests.bump_diffusion_cue_duration = 0.01
    config.tests.bump_diffusion_release_skip_steps = 1
    config.tests.bump_diffusion_integration_method = "forward_euler"
    config.tests.bump_diffusion_trials = 4
    config.tests.bump_diffusion_test_noise_stds = [0.1]
    config.tests.bump_diffusion_test_noise_std = 0.1
    config.diagnostics.bump_maintenance = True
    config.diagnostics.path_integration_and_pi_error = True
    config.diagnostics.pva_spectrum_and_visualization = True
    config.diagnostics.velocity_gain = True
    config.diagnostics.training_convergence = True
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
    config.tests.constant_pi_velocities = [-0.5, -0.2, 0.2, 0.5]
    config.tests.weight_snapshot_pi_velocities = [-0.5, 0.5]
    config.tests.weight_snapshot_pi_initial_heading = 0.0
    config.tests.weight_snapshot_pi_interval_fraction = 0.01
    config.tests.weight_snapshot_pi_cue_duration = 0.001
    config.tests.weight_snapshot_pi_duration = 0.003
    config.tests.weight_snapshot_pi_average_start_time = 0.0
    config.tests.learning_error_window_duration = 0.005
    config.tests.learning_error_interval_fraction = 0.2
    config.tests.gain_velocities = [-0.5, 0.5]
    return config


def test_diagnostic_weight_source_selects_best_final_or_explicit_snapshot() -> None:
    config = make_short_config()
    template_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )
    w_hd_history = np.stack(
        [template_state.w_hd_to_hd + offset for offset in (1.0, 2.0, 3.0)]
    )
    w_hr_history = np.stack(
        [template_state.w_hr_to_hd + offset for offset in (4.0, 5.0, 6.0)]
    )
    weight_history = {
        "time": np.asarray([0.0, 10.0, 20.0]),
        "w_hd_to_hd": w_hd_history,
        "w_hr_to_hd": w_hr_history,
    }
    selection_history = {
        "snapshot_source_index": np.asarray([0, 2], dtype=int),
        "best_snapshot_index": np.asarray(0, dtype=int),
    }

    expected_index_by_source = {"best": 0, "final": 2, "snapshot": 1}
    for source, expected_index in expected_index_by_source.items():
        config.diagnostics.weight_source = source
        config.diagnostics.weight_snapshot_index = 1 if source == "snapshot" else None
        final_state, best_state, diagnostic_state, metadata = resolve_weight_states(
            config=config,
            training_selected_state=template_state,
            weight_history=weight_history,
            weight_selection_history=selection_history,
        )
        np.testing.assert_allclose(final_state.w_hd_to_hd, w_hd_history[2])
        np.testing.assert_allclose(best_state.w_hd_to_hd, w_hd_history[0])
        np.testing.assert_allclose(
            diagnostic_state.w_hd_to_hd,
            w_hd_history[expected_index],
        )
        assert int(metadata["diagnostic_snapshot_source_index"]) == expected_index


def test_explicit_trajectory_selection_skips_unrelated_diagnostics() -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    config.diagnostics.trajectory_and_fixed_points = True
    config.tests.bump_attractor_cue_amplitude = 8.0
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
        weight_snapshot_pi_history,
        _numerical_convergence_history,
        metrics,
    ) = outputs

    assert hd_tuning_history["theta_true"].size == config.tests.hd_tuning_curve_angles
    assert trajectory_history["theta_initial"].size == (
        config.tests.bump_attractor_initial_conditions
    )
    assert trajectory_history["endpoint_probe_theta_initial"].size >= (
        config.tests.bump_attractor_initial_conditions
    )
    assert trajectory_history["endpoint_probe_refinement_level"].shape == (
        trajectory_history["endpoint_probe_theta_initial"].shape
    )
    assert np.max(
        trajectory_history["endpoint_probe_refinement_level"], initial=0
    ) <= config.tests.bump_attractor_boundary_bisection_depth
    for decoder_name in ("pva", "peak", "overlap"):
        assert trajectory_history[
            f"endpoint_probe_theta_release_{decoder_name}"
        ].shape == trajectory_history["endpoint_probe_theta_initial"].shape
        assert trajectory_history[
            f"endpoint_probe_theta_final_{decoder_name}"
        ].shape == trajectory_history["endpoint_probe_theta_initial"].shape
    assert "bump_attractor_endpoint_probe_count" in metrics
    assert "bump_attractor_boundary_nominal_resolution_deg" in metrics
    assert metrics["bump_attractor_cue_amplitude"] == 8.0
    assert (
        "bump_attractor_cue_release_visual_to_distal_modulation_ratio_median"
        in metrics
    )
    np.testing.assert_allclose(trajectory_history["cue_time"], [-0.01, 0.0])
    for decoder_name in ("pva", "peak", "overlap"):
        cue_trace = trajectory_history[f"cue_theta_{decoder_name}"]
        darkness_trace = trajectory_history[f"theta_{decoder_name}"]
        assert cue_trace.shape == (config.tests.bump_attractor_initial_conditions, 2)
        np.testing.assert_allclose(cue_trace[:, -1], darkness_trace[:, 0])
        np.testing.assert_allclose(
            trajectory_history[f"theta_release_{decoder_name}"],
            darkness_trace[:, 0],
        )
        assert (
            f"bump_attractor_{decoder_name}_cue_release_linearity_slope"
            in metrics
        )
        assert (
            f"bump_attractor_{decoder_name}_cue_release_linearity_r_squared"
            in metrics
        )
        assert (
            f"bump_attractor_{decoder_name}_endpoint_attractor_count"
            in metrics
        )
        assert (
            f"bump_attractor_{decoder_name}_endpoint_"
            "basin_boundary_bracket_count"
            in metrics
        )
        assert (
            f"bump_attractor_{decoder_name}_endpoint_"
            "trajectory_inferred_unstable_count"
            in metrics
        )
        assert (
            f"bump_attractor_{decoder_name}_endpoint_subbin_boundary_count"
            in metrics
        )
        assert (
            f"bump_attractor_{decoder_name}_endpoint_"
            "nonmonotonic_crossing_count"
            in metrics
        )
    assert not bump_history
    assert not bump_diffusion_history
    assert not darkness_history
    assert not ou_darkness_history
    assert not ou_ensemble_history
    assert not velocity_gain_history
    assert not weight_snapshot_pi_history
    assert metrics["hd_tuning_dependency_computed"] == 1.0
    assert metrics["bump_diffusion_diagnostic_enabled"] == 0.0


def test_numerical_convergence_group_runs_through_common_runner() -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    config.diagnostics.numerical_convergence = True
    config.tests.numerical_convergence_duration = 0.02
    config.tests.numerical_convergence_sample_interval = 0.01
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    outputs = run_all_tests(config=config, trained_state=trained_state)
    numerical_history = outputs[-2]
    metrics = outputs[-1]

    assert numerical_history["dt"].size == 8
    assert numerical_history["heading_error"].shape[0] == 8
    assert metrics["numerical_convergence_diagnostic_enabled"] == 1.0


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


def test_behavioral_checkpoint_selection_restores_best_before_patience_stop(
    monkeypatch,
) -> None:
    config = make_short_config()
    config.simulation.train_duration = 0.08
    selection = config.simulation.checkpoint_selection
    selection.enabled = True
    selection.min_duration = 0.02
    selection.check_interval = 0.01
    selection.patience_checks = 2
    selection.success_checks = 99
    selection.minimum_improvement = 0.0
    selection.cue_duration = 0.001
    selection.probe_duration = 0.002
    selection.fit_start_time = 0.0
    selection.velocities = [-0.2, 0.0, 0.2]
    selection.initial_headings = [0.0]
    scores = iter([0.5, 0.4, 0.45, 0.46])

    def fake_behavioral_evaluation(*, config, trained_state):
        del config, trained_state
        score = next(scores)
        return {
            "score": score,
            "rms_velocity_bias": score,
            "maximum_abs_velocity_bias": score,
            "maximum_abs_zero_velocity_drift": score,
            "minimum_pva_strength": 1.0,
            "minimum_bump_contrast": 1.0,
            "fully_defined": 1.0,
            "acceptance_passed": 0.0,
        }

    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.evaluate_training_checkpoint_behavior",
        fake_behavioral_evaluation,
    )
    selected_state, training_history, weight_history = run_training(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    assert np.isclose(training_history["training_actual_duration"], 0.05)
    assert np.isclose(training_history["training_selected_checkpoint_time"], 0.03)
    assert np.isclose(selected_state.time, 0.03)
    assert np.isclose(weight_history["time"][-1], 0.05)
    assert training_history["training_stop_reason"] == "behavior_patience"
    assert np.isclose(
        training_history["training_checkpoint_selection_restored_best"],
        1.0,
    )
    assert np.isclose(
        training_history["training_checkpoint_selection_was_fallback"],
        1.0,
    )
    np.testing.assert_allclose(
        training_history["checkpoint_selection_score"],
        [0.5, 0.4, 0.45, 0.46],
    )


def test_behavioral_checkpoint_selection_prefers_an_accepted_checkpoint(
    monkeypatch,
) -> None:
    config = make_short_config()
    config.simulation.train_duration = 0.05
    selection = config.simulation.checkpoint_selection
    selection.enabled = True
    selection.min_duration = 0.02
    selection.check_interval = 0.01
    selection.patience_checks = 10
    selection.success_checks = 1
    selection.minimum_improvement = 0.0
    selection.cue_duration = 0.001
    selection.probe_duration = 0.002
    selection.fit_start_time = 0.0
    selection.velocities = [-0.2, 0.0, 0.2]
    selection.initial_headings = [0.0]
    evaluations = iter([(0.1, 0.0), (0.2, 1.0)])

    def fake_behavioral_evaluation(*, config, trained_state):
        del config, trained_state
        score, acceptance_passed = next(evaluations)
        return {
            "score": score,
            "rms_velocity_bias": score,
            "maximum_abs_velocity_bias": score,
            "maximum_abs_zero_velocity_drift": score,
            "minimum_pva_strength": 1.0,
            "minimum_bump_contrast": 1.0,
            "fully_defined": 1.0,
            "acceptance_passed": acceptance_passed,
        }

    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.evaluate_training_checkpoint_behavior",
        fake_behavioral_evaluation,
    )
    selected_state, training_history, _weight_history = run_training(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    assert training_history["training_stop_reason"] == "behavior_success"
    assert np.isclose(training_history["training_actual_duration"], 0.03)
    assert np.isclose(training_history["training_selected_checkpoint_time"], 0.03)
    assert np.isclose(training_history["training_selected_checkpoint_score"], 0.2)
    assert np.isclose(training_history["training_behavioral_acceptance_passed"], 1.0)
    assert np.isclose(
        training_history["training_checkpoint_selection_was_fallback"],
        0.0,
    )
    assert np.isclose(selected_state.time, 0.03)


def test_behavioral_checkpoint_evaluation_is_frozen_and_finite() -> None:
    config = make_short_config()
    selection = config.simulation.checkpoint_selection
    selection.enabled = True
    selection.min_duration = 0.01
    selection.check_interval = 0.01
    selection.cue_duration = 0.001
    selection.probe_duration = 0.003
    selection.fit_start_time = 0.0005
    selection.velocities = [0.0, 0.2]
    selection.initial_headings = [0.0, np.pi / 2.0]
    state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )
    w_hd_before = state.w_hd_to_hd.copy()
    w_hr_before = state.w_hr_to_hd.copy()

    metrics = evaluate_training_checkpoint_behavior(
        config=config,
        trained_state=state,
    )

    for metric_name in (
        "score",
        "rms_velocity_bias",
        "maximum_abs_velocity_bias",
        "maximum_abs_zero_velocity_drift",
        "minimum_pva_strength",
        "minimum_bump_contrast",
    ):
        assert np.isfinite(metrics[metric_name])
    np.testing.assert_allclose(state.w_hd_to_hd, w_hd_before)
    np.testing.assert_allclose(state.w_hr_to_hd, w_hr_before)


def test_constant_velocity_pi_error_sweep_records_each_configured_speed() -> None:
    config = make_short_config()
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    history, metrics = run_constant_velocity_pi_error_sweep_test(
        config=config,
        trained_state=trained_state,
    )

    np.testing.assert_allclose(
        history["constant_pi_commanded_velocity"],
        config.tests.constant_pi_velocities,
    )
    assert history["constant_pi_error"].shape[0] == 4
    assert history["constant_pi_velocity_gain"].shape == (4,)
    assert np.isfinite(history["constant_pi_decoded_velocity"]).all()
    assert metrics["constant_pi_velocity_count"] == 4.0
    assert np.isfinite(metrics["constant_pi_rms_velocity_bias"])


def test_constant_velocity_pi_error_sweep_keeps_zero_speed_as_drift_probe() -> None:
    config = make_short_config()
    config.tests.constant_pi_velocities = [-0.5, 0.0, 0.5]
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    history, metrics = run_constant_velocity_pi_error_sweep_test(
        config=config,
        trained_state=trained_state,
    )

    zero_index = int(
        np.flatnonzero(
            np.isclose(history["constant_pi_commanded_velocity"], 0.0)
        )[0]
    )
    moving_mask = np.arange(3) != zero_index
    assert np.isnan(history["constant_pi_velocity_gain"][zero_index])
    assert np.isnan(history["constant_pi_peak_velocity_gain"][zero_index])
    assert np.isfinite(history["constant_pi_velocity_gain"][moving_mask]).all()
    assert metrics["constant_pi_zero_velocity_probe_count"] == 1.0
    assert np.isclose(
        metrics["constant_pi_zero_velocity_drift"],
        history["constant_pi_decoded_velocity"][zero_index],
    )
    assert np.isfinite(metrics["constant_pi_rms_gain_error"])


def test_diagnostic_failure_preserves_completed_independent_results(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    config.diagnostics.bump_maintenance = True
    config.diagnostics.velocity_gain = True
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    def fail_bump(**kwargs):
        del kwargs
        raise RuntimeError("synthetic bump failure")

    def finish_velocity_gain(**kwargs):
        del kwargs
        return (
            {
                "commanded_velocity": np.asarray([-0.5, 0.5]),
                "decoded_velocity": np.asarray([-0.4, 0.4]),
            },
            {"synthetic_velocity_gain_completed": 1.0},
        )

    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.run_bump_maintenance_test",
        fail_bump,
    )
    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.run_velocity_gain_test",
        finish_velocity_gain,
    )
    recorder = IncrementalDiagnosticRecorder(run_dir=tmp_path)

    outputs = run_all_tests(
        config=config,
        trained_state=trained_state,
        diagnostic_recorder=recorder,
        continue_on_error=True,
    )

    assert not outputs[1]
    assert outputs[10]["commanded_velocity"].size == 2
    assert outputs[-1]["diagnostic_failure_count"] == 1.0
    assert outputs[-1]["diagnostic_blocked_count"] == 1.0
    assert (tmp_path / "velocity_gain_history.npz").exists()
    assert not (tmp_path / "bump_history.npz").exists()
    status = load_json(tmp_path / "diagnostic_status.json")
    assert status["diagnostics"]["bump_maintenance"]["status"] == "failed"
    assert status["diagnostics"]["velocity_gain"]["status"] == "completed"
    assert status["diagnostics"]["zero_velocity_drive"]["status"] == "blocked"
    partial_metrics = load_json(tmp_path / "test_metrics.partial.json")
    assert partial_metrics["bump_maintenance_diagnostic_failed"] == 1.0
    assert partial_metrics["synthetic_velocity_gain_completed"] == 1.0


def test_pi_subdiagnostic_failure_does_not_discard_other_pi_histories(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    config.diagnostics.path_integration_and_pi_error = True
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.run_darkness_path_integration_test",
        lambda **kwargs: (
            {"time": np.asarray([0.0]), "primary_marker": np.asarray([1.0])},
            {"primary_pi_completed": 1.0},
        ),
    )

    def fail_constant_sweep(**kwargs):
        del kwargs
        raise ValueError("synthetic constant sweep failure")

    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.run_constant_velocity_pi_error_sweep_test",
        fail_constant_sweep,
    )
    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.run_ou_path_integration_test",
        lambda **kwargs: (
            {"time": np.asarray([0.0]), "ou_marker": np.asarray([1.0])},
            {"ou_pi_completed": 1.0},
        ),
    )
    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.run_ou_path_integration_ensemble_test",
        lambda **kwargs: (
            {
                "time": np.asarray([0.0]),
                "ensemble_marker": np.asarray([1.0]),
            },
            {"ou_ensemble_completed": 1.0},
        ),
    )
    recorder = IncrementalDiagnosticRecorder(run_dir=tmp_path)

    outputs = run_all_tests(
        config=config,
        trained_state=trained_state,
        diagnostic_recorder=recorder,
        continue_on_error=True,
    )

    assert outputs[7]["primary_marker"].item() == 1.0
    assert outputs[8]["ou_marker"].item() == 1.0
    assert outputs[9]["ensemble_marker"].item() == 1.0
    for filename in (
        "darkness_history.npz",
        "ou_darkness_history.npz",
        "ou_pi_ensemble_history.npz",
    ):
        assert (tmp_path / filename).exists()
    status = load_json(tmp_path / "diagnostic_status.json")
    assert status["diagnostics"]["constant_velocity_pi_sweep"]["status"] == "failed"
    assert status["diagnostics"]["darkness_path_integration"]["status"] == "completed"
    assert status["diagnostics"]["ou_path_integration"]["status"] == "completed"
    assert status["diagnostics"]["ou_pi_ensemble"]["status"] == "completed"


def test_figure_groups_continue_after_an_independent_group_fails(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    config.diagnostics.bump_maintenance = True
    config.diagnostics.velocity_gain = True
    save_yaml(tmp_path / "config_resolved.yaml", config.to_dict())
    calls: list[tuple[str, str]] = []

    def make_one_group(**kwargs):
        group_name = next(iter(kwargs["groups"]))
        calls.append((group_name, kwargs["figure_dir"].name))
        if group_name == "bump_maintenance":
            raise KeyError("synthetic missing bump field")

    monkeypatch.setattr(
        figures_module,
        "_make_grouped_diagnostic_figures",
        make_one_group,
    )

    status = figures_module.make_vafidis_figures_for_run(run_dir=tmp_path)

    assert calls == [
        ("bump_maintenance", "bump_maintenance"),
        ("velocity_gain", "velocity_gain"),
    ]
    assert status["figures"]["bump_maintenance"]["status"] == "failed"
    assert status["figures"]["bump_maintenance"]["output_dir"] == (
        "figures/bump_maintenance"
    )
    assert status["figures"]["velocity_gain"]["status"] == "completed"
    assert status["figures"]["velocity_gain"]["output_dir"] == (
        "figures/velocity_gain"
    )
    assert (tmp_path / "figures" / "bump_maintenance").is_dir()
    assert (tmp_path / "figures" / "velocity_gain").is_dir()
    for legacy_dir_name in ("activity", "diagnostics", "gain", "heading", "weights"):
        assert not (tmp_path / "figures" / legacy_dir_name).exists()
    saved_status = load_json(tmp_path / "figure_status.json")
    assert saved_status["completed_count"] == 1
    assert saved_status["failed_count"] == 1


def test_pi_figure_jobs_share_the_canonical_group_directory(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    config.diagnostics.path_integration_and_pi_error = True
    save_yaml(tmp_path / "config_resolved.yaml", config.to_dict())
    calls: list[tuple[str, str]] = []

    def make_one_group(**kwargs):
        calls.append(
            (next(iter(kwargs["groups"])), kwargs["figure_dir"].name)
        )

    monkeypatch.setattr(
        figures_module,
        "_make_grouped_diagnostic_figures",
        make_one_group,
    )

    status = figures_module.make_vafidis_figures_for_run(run_dir=tmp_path)

    assert calls == [
        ("path_integration_constant", "path_integration_and_pi_error"),
        ("path_integration_ou", "path_integration_and_pi_error"),
        ("path_integration_ensemble", "path_integration_and_pi_error"),
    ]
    assert {
        entry["output_dir"] for entry in status["figures"].values()
    } == {"figures/path_integration_and_pi_error"}


def test_phase_flow_only_uses_the_canonical_group_directory(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_short_config()
    save_yaml(tmp_path / "config_resolved.yaml", config.to_dict())
    save_npz(
        tmp_path / "velocity_trajectory_sweep_history.npz",
        marker=np.asarray([1.0]),
    )
    monkeypatch.setattr(
        figures_module,
        "summarize_velocity_phase_flows",
        lambda **kwargs: {"marker": np.asarray([1.0])},
    )
    figure_paths = []

    def record_figure(*, summary, path):
        figure_paths.append(path)

    for plot_name in (
        "plot_actual_fp_basin_rings",
        "plot_velocity_dense_probe_trajectories",
        "plot_velocity_phase_flow_diagnostics",
    ):
        monkeypatch.setattr(figures_module, plot_name, record_figure)

    figures_module.make_velocity_phase_flow_figures_for_run(run_dir=tmp_path)

    assert {path.name for path in figure_paths} == {
        "velocity_actual_fp_basin_rings.png",
        "velocity_dense_probe_trajectories.png",
        "velocity_phase_flow_diagnostics.png",
    }
    assert {
        path.parent.name for path in figure_paths
    } == {"velocity_dynamics_and_phase_flow"}


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
    history, diffusion_metrics = run_bump_diffusion_ensemble_test(
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
    endpoint_displacement = history["pva_endpoint_angular_displacement"][0]
    expected_darkness_duration = (
        config.tests.bump_diffusion_duration
        - config.tests.bump_diffusion_cue_duration
    )
    assert np.isclose(
        diffusion_metrics["bump_ensemble_diffusion_coefficient"],
        np.var(endpoint_displacement) / expected_darkness_duration,
    )
    assert np.isclose(
        diffusion_metrics["bump_ensemble_diffusion_duration"],
        expected_darkness_duration,
    )
    assert diffusion_metrics["bump_ensemble_diffusion_forward_euler"] == 1.0
    assert history["theta_initial"].shape == (1, config.tests.bump_diffusion_trials)


def test_path_integration_group_progress_tracks_all_protocol_steps(
    monkeypatch,
) -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    config.diagnostics.path_integration_and_pi_error = True
    config.simulation.progress = True
    trained_state = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
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
    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.trange",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(f"unexpected nested progress: {args}, {kwargs}")
        ),
    )

    outputs = run_all_tests(config=config, trained_state=trained_state)

    full_protocol_steps = sum(
        round(duration / config.simulation.dt)
        for duration in (
            config.simulation.pi_cue_duration,
            config.simulation.darkness_test_duration,
            config.simulation.recue_duration,
        )
    )
    ensemble_protocol_steps = sum(
        round(duration / config.simulation.dt)
        for duration in (
            config.simulation.pi_cue_duration,
            config.simulation.darkness_test_duration,
        )
    )
    primary_is_in_sweep = any(
        np.isclose(config.tests.darkness_angular_velocity, velocity)
        for velocity in config.tests.constant_pi_velocities
    )
    constant_protocol_count = (
        1 + len(config.tests.constant_pi_velocities) - int(primary_is_in_sweep)
    )
    expected_steps = (
        constant_protocol_count * full_protocol_steps
        + full_protocol_steps
        + config.tests.ou_pi_ensemble_trials * ensemble_protocol_steps
    )
    progress_kwargs = captured["kwargs"]
    assert progress_kwargs["total"] == expected_steps
    assert progress_kwargs["disable"] is False
    assert progress_kwargs["desc"] == "path integration diagnostics"
    assert progress_kwargs["unit"] == "step"
    assert captured["updated"] == expected_steps
    jobs = {entry["job"] for entry in captured["postfix"]}
    assert "OU PI single trial" in jobs
    assert f"OU PI ensemble {config.tests.ou_pi_ensemble_trials}/{config.tests.ou_pi_ensemble_trials}" in jobs
    assert any(job.startswith("constant PI primary") for job in jobs)
    assert not outputs[6]
    assert outputs[7]
    assert outputs[8]
    assert outputs[9]


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


def test_mid_training_failure_keeps_latest_atomic_weight_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_short_config()
    config.simulation.train_duration = 0.01
    config.simulation.weight_snapshot_interval_duration = 0.001
    call_count = 0

    def fail_after_first_periodic_checkpoint(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("synthetic training failure")
        return step_vafidis_toy(**kwargs)

    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.step_vafidis_toy",
        fail_after_first_periodic_checkpoint,
    )
    run_id = "mid_training_failure"
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_experiment(
            config=config,
            project_root=tmp_path,
            run_id=run_id,
            make_figures=False,
        )

    run_dir = tmp_path / config.paths.runs_root / run_id
    checkpoint_path = run_dir / "training_checkpoint_latest.npz"
    assert checkpoint_path.exists()
    checkpoint = load_npz(checkpoint_path)
    assert str(checkpoint["weight_source"]) == "training_checkpoint"
    assert np.isclose(float(checkpoint["weight_snapshot_time"]), 0.001)
    assert checkpoint["w_hd_to_hd"].shape == (
        config.model.n_theta,
        config.model.n_theta,
    )
    assert checkpoint["w_hr_to_hd"].shape == (
        config.model.n_theta,
        config.model.n_hr,
    )


def test_post_training_diagnostic_failure_keeps_selected_weight_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_short_config()

    def fail_diagnostics(**kwargs):
        del kwargs
        raise RuntimeError("synthetic diagnostic failure")

    monkeypatch.setattr(
        "learning.experiments.run_vafidis_toy.run_all_tests",
        fail_diagnostics,
    )
    run_id = "post_training_diagnostic_failure"
    with pytest.raises(RuntimeError, match="synthetic diagnostic failure"):
        run_experiment(
            config=config,
            project_root=tmp_path,
            run_id=run_id,
            make_figures=False,
        )

    run_dir = tmp_path / config.paths.runs_root / run_id
    for filename in (
        "training_checkpoint_latest.npz",
        "training_selected_weights.npz",
        "final_weights.npz",
        "best_weights.npz",
        "diagnostic_weights.npz",
        "trained_weights.npz",
        "training_history.npz",
        "weight_history.npz",
        "weight_selection_history.npz",
    ):
        assert (run_dir / filename).exists(), filename
    assert not (run_dir / "test_metrics.json").exists()


def test_short_experiment_writes_required_outputs(tmp_path) -> None:
    config = make_short_config()
    run_dir = run_experiment(config=config, project_root=tmp_path, make_figures=True)
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "params.json").exists()
    assert (run_dir / "trained_weights.npz").exists()
    assert (run_dir / "final_weights.npz").exists()
    assert (run_dir / "best_weights.npz").exists()
    assert (run_dir / "training_selected_weights.npz").exists()
    assert (run_dir / "diagnostic_weights.npz").exists()
    assert (run_dir / "weight_selection_history.npz").exists()
    assert (run_dir / "training_history.npz").exists()
    assert (run_dir / "weight_history.npz").exists()
    assert (run_dir / "weight_snapshot_pi_development.npz").exists()
    assert (run_dir / "bump_diffusion_history.npz").exists()
    assert (run_dir / "bump_attractor_trajectory_history.npz").exists()
    assert (run_dir / "timescale_separation_history.npz").exists()
    assert (run_dir / "velocity_trajectory_sweep_history.npz").exists()
    assert (run_dir / "hd_tuning_history.npz").exists()
    assert (run_dir / "hd_tuning_com_aligned.npz").exists()
    assert (run_dir / "ou_darkness_history.npz").exists()
    assert (run_dir / "ou_pi_ensemble_history.npz").exists()
    assert (run_dir / "test_metrics.json").exists()
    assert (
        run_dir
        / "figures"
        / "weight_snapshots_and_development"
        / "training_snapshot_frozen_pi_error.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "pva_spectrum_and_visualization"
        / "training_visual_input_heatmap.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "training_convergence"
        / "training_absolute_learning_error.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "bump_maintenance"
        / "bump_maintenance_visual_input_heatmap.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "trajectory_and_fixed_points"
        / "bump_attractor_decoder_trajectories.png"
    ).exists()
    for pva_figure_name in (
        "bump_attractor_pva_initial_cue_endpoint_map.png",
        "bump_attractor_pva_release_angle_endpoint_map.png",
        "bump_attractor_pva_cue_transfer.png",
    ):
        assert (
            run_dir
            / "figures"
            / "trajectory_and_fixed_points"
            / pva_figure_name
        ).exists()
    assert (
        run_dir
        / "figures"
        / "timescale_separation"
        / "timescale_separation_diagnostics.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "velocity_dynamics_and_phase_flow"
        / "velocity_trajectory_sweep.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "velocity_dynamics_and_phase_flow"
        / "velocity_actual_fp_basin_rings.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "velocity_dynamics_and_phase_flow"
        / "velocity_dense_probe_trajectories.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "velocity_dynamics_and_phase_flow"
        / "velocity_phase_flow_diagnostics.png"
    ).exists()
    assert (run_dir / "velocity_phase_flow_summary.npz").exists()
    assert (
        run_dir
        / "figures"
        / "path_integration_and_pi_error"
        / "darkness_visual_input_heatmap.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "path_integration_and_pi_error"
        / "ou_darkness_visual_input_heatmap.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "path_integration_and_pi_error"
        / "constant_velocity_pi_error_grid.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "pva_spectrum_and_visualization"
        / "com_aligned_hd_tuning_population.png"
    ).exists()
    assert (
        run_dir
        / "figures"
        / "pva_spectrum_and_visualization"
        / "hd_tuning_visual_only_vs_post_training.png"
    ).exists()
    training_history = load_npz(run_dir / "training_history.npz")
    weight_history = load_npz(run_dir / "weight_history.npz")
    final_weights = load_npz(run_dir / "final_weights.npz")
    best_weights = load_npz(run_dir / "best_weights.npz")
    diagnostic_weights = load_npz(run_dir / "diagnostic_weights.npz")
    trained_weights = load_npz(run_dir / "trained_weights.npz")
    weight_selection_history = load_npz(
        run_dir / "weight_selection_history.npz"
    )
    test_metrics = load_json(run_dir / "test_metrics.json")
    assert "pva_strength_hd" in training_history
    assert "absolute_learning_error_mean_spikes_per_s" in training_history
    assert training_history["absolute_learning_error_time"].size == 5
    assert np.allclose(weight_history["time"], [0.0, 0.02, 0.04, 0.05])
    np.testing.assert_allclose(
        final_weights["w_hd_to_hd"], weight_history["w_hd_to_hd"][-1]
    )
    best_source_index = int(
        weight_selection_history["best_snapshot_source_index"]
    )
    np.testing.assert_allclose(
        best_weights["w_hd_to_hd"],
        weight_history["w_hd_to_hd"][best_source_index],
    )
    np.testing.assert_allclose(
        diagnostic_weights["w_hd_to_hd"], best_weights["w_hd_to_hd"]
    )
    np.testing.assert_allclose(
        trained_weights["w_hd_to_hd"], diagnostic_weights["w_hd_to_hd"]
    )
    assert str(diagnostic_weights["weight_source"]) == "best"
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
    weight_snapshot_pi_history = load_npz(
        run_dir / "weight_snapshot_pi_development.npz"
    )
    assert "phase_id" in darkness_history
    assert "visual_teacher" in darkness_history
    assert "i_vis_to_hd" in darkness_history
    assert "phase_id" in ou_darkness_history
    assert "i_vis_to_hd" in ou_darkness_history
    assert "i_vis_to_hd" in bump_history
    assert bump_diffusion_history["pva_endpoint_angular_displacement"].shape == (1, 4)
    np.testing.assert_allclose(bump_diffusion_history["test_noise_std"], [0.1])
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
    assert weight_snapshot_pi_history["time_averaged_abs_pi_error"].shape == (4, 2)
    np.testing.assert_allclose(
        weight_snapshot_pi_history["snapshot_time"],
        weight_history["time"],
    )
    assert "weight_snapshot_pi_best_snapshot_time" in test_metrics
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
    assert darkness_history["constant_pi_error"].shape == (
        4,
        darkness_history["constant_pi_time"].size,
    )
    np.testing.assert_allclose(
        darkness_history["constant_pi_commanded_velocity"],
        config.tests.constant_pi_velocities,
    )
    assert darkness_history["constant_pi_velocity_gain"].shape == (4,)
    assert test_metrics["constant_pi_velocity_count"] == 4.0
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
        / "trajectory_and_fixed_points"
        / "bump_attractor_decoder_trajectories.png"
    ).exists()
    for pva_figure_name in (
        "bump_attractor_pva_initial_cue_endpoint_map.png",
        "bump_attractor_pva_release_angle_endpoint_map.png",
        "bump_attractor_pva_cue_transfer.png",
    ):
        assert (
            run_dir
            / "figures"
            / "trajectory_and_fixed_points"
            / pva_figure_name
        ).exists()
    assert not (run_dir / "bump_diffusion_history.npz").exists()
    assert not (run_dir / "darkness_history.npz").exists()


def test_existing_run_can_evaluate_saved_weight_snapshots(tmp_path) -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    run_dir = run_experiment(config=config, project_root=tmp_path, make_figures=False)
    diagnostics_path = tmp_path / "snapshot_diagnostics.yaml"
    switches = {group_name: False for group_name in DIAGNOSTIC_GROUPS}
    switches["weight_snapshots_and_development"] = True
    save_yaml(
        diagnostics_path,
        {
            "diagnostics": switches,
            "tests": {
                "weight_snapshot_pi_velocities": [-0.5, 0.5],
                "weight_snapshot_pi_interval_fraction": 0.01,
                "weight_snapshot_pi_cue_duration": 0.001,
                "weight_snapshot_pi_duration": 0.003,
                "weight_snapshot_pi_average_start_time": 0.0,
            },
        },
    )

    run_tests_for_existing_run(
        run_dir=run_dir,
        make_figures=True,
        diagnostics_config_path=diagnostics_path,
    )

    snapshot_history = load_npz(run_dir / "weight_snapshot_pi_development.npz")
    weight_history = load_npz(run_dir / "weight_history.npz")
    np.testing.assert_allclose(snapshot_history["snapshot_time"], weight_history["time"])
    assert snapshot_history["time_averaged_abs_pi_error"].shape == (4, 2)
    assert (
        run_dir
        / "figures"
        / "weight_snapshots_and_development"
        / "training_snapshot_frozen_pi_error.png"
    ).exists()


def test_existing_run_diagnostics_can_explicitly_use_final_weights(tmp_path) -> None:
    config = make_short_config()
    for group_name in DIAGNOSTIC_GROUPS:
        setattr(config.diagnostics, group_name, False)
    run_dir = run_experiment(config=config, project_root=tmp_path, make_figures=False)
    diagnostics_path = tmp_path / "final_weight_diagnostics.yaml"
    save_yaml(
        diagnostics_path,
        {
            "diagnostics": {
                **{group_name: False for group_name in DIAGNOSTIC_GROUPS},
                "weight_source": "final",
                "weight_snapshot_index": None,
            }
        },
    )

    run_tests_for_existing_run(
        run_dir=run_dir,
        make_figures=False,
        diagnostics_config_path=diagnostics_path,
    )

    final_weights = load_npz(run_dir / "final_weights.npz")
    diagnostic_weights = load_npz(run_dir / "diagnostic_weights.npz")
    np.testing.assert_allclose(
        diagnostic_weights["w_hd_to_hd"],
        final_weights["w_hd_to_hd"],
    )
    assert str(diagnostic_weights["weight_source"]) == "final"
