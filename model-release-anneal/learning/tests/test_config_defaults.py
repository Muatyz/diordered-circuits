from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learning.config.diagnostics import DIAGNOSTIC_GROUPS, selected_diagnostics
from learning.config.load_config import load_experiment_config, load_yaml, save_yaml
from learning.config.schema import RETIRED_TEST_CONFIG_FIELDS
from learning.dynamics.hd_dynamics import (
    PROXIMAL_INTEGRATION_EXACT_LINEAR,
    PROXIMAL_INTEGRATION_FORWARD_EULER,
)
from learning.experiments.run_attractor_robustness import (
    _retune_visual_width_for_neuron_count,
    _visual_sigma_from_kappa,
)


def test_default_training_protocol_has_positive_duration() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "vafidis_toy.yaml"
    config = load_experiment_config(config_path)
    assert config.simulation.train_duration > 0.0


def test_sampling_intervals_are_physical_durations_in_seconds() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "vafidis_toy.yaml"
    )
    config = load_experiment_config(config_path)

    assert np.isclose(config.simulation.save_interval_duration, 500.0)
    assert np.isclose(config.simulation.weight_snapshot_interval_duration, 500.0)


def test_vafidis_toy_direct_run_enables_rank_only_target_diagnostics() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "vafidis_toy.yaml"
    )
    config = load_experiment_config(config_path)

    assert config.simulation.checkpoint_selection.enabled is False
    assert config.diagnostics.weight_source == "best"
    assert config.diagnostics.path_integration_and_pi_error is True
    assert config.diagnostics.trajectory_and_fixed_points is True
    assert config.diagnostics.weight_snapshots_and_development is True
    assert config.diagnostics.numerical_convergence is True
    assert config.tests.weight_snapshot_pi_selection_metric == "rms_velocity_bias"
    np.testing.assert_allclose(
        np.rad2deg(config.tests.weight_snapshot_pi_velocities),
        [-75.0, -30.0, -15.0, 0.0, 15.0, 30.0, 75.0],
    )
    np.testing.assert_allclose(
        np.rad2deg(config.tests.weight_snapshot_pi_initial_headings),
        [-135.0, -45.0, 45.0, 135.0],
    )
    assert config.tests.bump_attractor_boundary_bisection_depth == 6


def test_all_experiment_configs_share_target_diagnostics_and_snapshot_ranking() -> None:
    experiment_dir = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    configs = [
        load_experiment_config(experiment_dir / filename)
        for filename in (
            "vafidis_toy.yaml",
            "vafidis_von_mises.yaml",
            "vafidis_heterogeneous.yaml",
        )
    ]

    reference = configs[0]
    for config in configs:
        assert config.simulation.checkpoint_selection.enabled is False
        assert config.simulation.train_duration == reference.simulation.train_duration
        assert config.simulation.darkness_test_duration == 60.0
        assert config.diagnostics == reference.diagnostics
        assert config.tests == reference.tests
        assert config.tests.weight_snapshot_pi_selection_metric == "rms_velocity_bias"


def test_removed_step_interval_fields_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy_steps.yaml"
    save_yaml(config_path, {"simulation": {"save_interval_steps": 40}})

    with pytest.raises(ValueError, match="Unknown config field: save_interval_steps"):
        load_experiment_config(config_path)


def test_behavioral_checkpoint_selection_profile_is_explicit_opt_in() -> None:
    project_root = Path(__file__).resolve().parents[1]
    experiment_path = project_root / "configs" / "experiments" / "vafidis_toy.yaml"
    baseline = load_experiment_config(experiment_path)
    selected = load_experiment_config(
        experiment_path,
        profile_paths=(
            project_root
            / "configs"
            / "profiles"
            / "behavioral_checkpoint_selection.yaml",
        ),
    )

    assert baseline.simulation.checkpoint_selection.enabled is False
    assert selected.simulation.checkpoint_selection.enabled is True
    assert selected.simulation.checkpoint_selection.restore_best is True


def test_vafidis_baseline_matches_release_firing_rate_and_weight_units() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "vafidis_toy.yaml"
    config = load_experiment_config(config_path)
    release_fmax_khz = 0.15

    assert config.experiment_name == "vafidis_release_parameter_baseline"
    assert config.model.n_theta == 60
    assert config.model.n_hr == 60
    assert np.isclose(config.simulation.dt, 5e-4)
    assert (
        config.simulation.proximal_integration_method
        == PROXIMAL_INTEGRATION_EXACT_LINEAR
    )
    assert np.isclose(config.simulation.train_duration, 8e4)
    assert config.simulation.random_stream_mode == "component_split"
    assert config.simulation.early_stopping.enabled is False
    assert config.simulation.checkpoint_selection.enabled is False
    assert config.simulation.checkpoint_selection.restore_best is True
    assert 0.0 in config.simulation.checkpoint_selection.velocities
    np.testing.assert_allclose(
        np.rad2deg(config.tests.constant_pi_velocities),
        [-75.0, -30.0, 30.0, 75.0],
    )
    assert np.isclose(config.model.tau_s, 0.065)
    assert np.isclose(config.model.tau_hd_to_hr, 0.065)
    assert np.isclose(config.model.tau_l_hd, 0.01)
    assert np.isclose(config.model.c_hd_proximal, 0.001)
    assert np.isclose(config.model.g_l_hd_proximal, 1.0)
    assert np.isclose(config.model.g_d_hd_to_proximal, 2.0)
    assert np.isclose(config.model.p_distal_to_proximal, 2.0 / 3.0)
    assert np.isclose(config.model.b_hd, 1.0)
    assert np.isclose(config.model.b_hr, 1.5)
    assert config.model.hd_distal_normalization == "raw_sum"
    assert np.isclose(config.model.activation.max_rate, release_fmax_khz)
    assert np.isclose(
        config.model.w_hd_to_hr_strength,
        2.0 / release_fmax_khz,
    )
    assert config.model.init.w_hd_to_hd_mode == "random_normal"
    assert config.model.init.w_hr_to_hd_mode == "random_normal"
    expected_weight_std = 1.0 / np.sqrt(120.0)
    assert np.isclose(config.model.init.w_hd_to_hd_scale, expected_weight_std)
    assert np.isclose(config.model.init.w_hr_to_hd_scale, expected_weight_std)
    # The engine keeps seconds as its time unit, while release uses ms.
    # Firing rates and weights themselves are no longer rescaled.
    expected_eta = 0.05 * 1000.0
    assert np.isclose(config.learning_rule.eta_hd_to_hd, expected_eta)
    assert np.isclose(config.learning_rule.eta_hr_to_hd, expected_eta)
    assert config.learning_rule.w_hd_to_hd_min is None
    assert config.learning_rule.w_hd_to_hd_max is None
    assert config.learning_rule.w_hr_to_hd_min is None
    assert config.learning_rule.w_hr_to_hd_max is None
    assert config.learning_rule.zero_hd_to_hd_diagonal is False
    assert np.isclose(config.visual.amplitude, 4.0)
    assert np.isclose(config.visual.baseline, 5.0)
    assert np.isclose(config.visual.light_excitation, 4.0)
    assert np.isclose(config.visual.proximal_scale, 1.0)
    assert np.isclose(config.velocity.std, 225.0 * np.pi / 180.0)
    assert config.velocity.clip is None


def test_release_pilot_changes_only_training_budget_from_release_baseline() -> None:
    experiment_dir = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    baseline = load_experiment_config(experiment_dir / "vafidis_toy.yaml")
    pilot = load_experiment_config(experiment_dir / "vafidis_release_dt1ms_pilot.yaml")

    assert pilot.experiment_name == "vafidis_release_dt1ms_pilot"
    assert np.isclose(pilot.simulation.dt, 5e-4)
    assert (
        pilot.simulation.proximal_integration_method
        == PROXIMAL_INTEGRATION_FORWARD_EULER
    )
    assert np.isclose(pilot.simulation.train_duration, 1e4)
    assert int(round(pilot.simulation.train_duration / pilot.simulation.dt)) == 20_000_000
    assert pilot.model == baseline.model
    assert pilot.learning_rule == baseline.learning_rule
    assert pilot.visual == baseline.visual
    assert pilot.velocity == baseline.velocity
    assert pilot.tests == baseline.tests
    assert pilot.diagnostics == baseline.diagnostics


def test_reusable_profile_and_cli_overrides_compose_in_order() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(
        project_root / "configs" / "experiments" / "vafidis_toy.yaml",
        profile_paths=[project_root / "configs" / "profiles" / "code_smoke.yaml"],
        overrides=[
            "simulation.train_duration=0.125",
            "simulation.progress=true",
            "tests.gain_velocities=[-1.0, 0.0, 1.0]",
            "experiment_name=one_off_smoke",
        ],
    )

    assert config.model.n_theta == 12
    assert config.model.n_hr == 12
    assert np.isclose(config.simulation.dt, 5e-4)
    assert np.isclose(config.simulation.train_duration, 0.125)
    assert config.simulation.progress is True
    assert config.diagnostics.training_convergence is True
    assert not any(
        getattr(config.diagnostics, group_name)
        for group_name in DIAGNOSTIC_GROUPS
        if group_name != "training_convergence"
    )
    assert config.tests.gain_velocities == [-1.0, 0.0, 1.0]
    assert config.experiment_name == "one_off_smoke"
    assert config.paths.runs_root == "runs/vafidis_code_smoke"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("simulation.missing_field=1", "Unknown config override path"),
        ("simulation.train_duration=fast", "must be numeric"),
        ("simulation.progress=1", "must be a boolean"),
        ("simulation.train_duration", "dotted.path=value"),
    ],
)
def test_cli_config_overrides_reject_typos_and_wrong_types(
    override: str,
    message: str,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "vafidis_toy.yaml"
    )

    with pytest.raises(ValueError, match=message):
        load_experiment_config(config_path, overrides=[override])


def test_multiple_profiles_are_ordered_and_reject_unknown_fields(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "experiments" / "vafidis_toy.yaml"
    first_profile = tmp_path / "first.yaml"
    second_profile = tmp_path / "second.yaml"
    invalid_profile = tmp_path / "invalid.yaml"
    save_yaml(first_profile, {"simulation": {"train_duration": 1.0}})
    save_yaml(second_profile, {"simulation": {"train_duration": 2.0}})
    save_yaml(invalid_profile, {"simulation": {"trian_duration": 3.0}})

    config = load_experiment_config(
        config_path,
        profile_paths=[first_profile, second_profile],
    )
    assert np.isclose(config.simulation.train_duration, 2.0)

    with pytest.raises(ValueError, match="simulation.trian_duration"):
        load_experiment_config(config_path, profile_paths=[invalid_profile])


def test_experiment_configs_enable_paper_learning_error_recording() -> None:
    experiment_config_dir = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    config_paths = sorted(experiment_config_dir.glob("*.yaml"))
    assert config_paths

    for config_path in config_paths:
        authored_config = load_yaml(config_path)
        assert "diagnostics_config" not in authored_config
        assert authored_config["diagnostics"]["training_convergence"] is True
        assert np.isclose(
            authored_config["tests"]["learning_error_window_duration"],
            10.0,
        )
        assert np.isclose(
            authored_config["tests"]["learning_error_interval_fraction"],
            0.01,
        )
        assert np.isclose(authored_config["model"]["activation"]["max_rate"], 0.15)
        config = load_experiment_config(config_path)
        assert np.isclose(config.model.activation.max_rate, 0.15)
        assert "learning_error_development" in selected_diagnostics(config)


def test_historical_resolved_config_without_max_rate_keeps_unit_peak_semantics(
    tmp_path: Path,
) -> None:
    old_resolved_path = tmp_path / "config_resolved.yaml"
    save_yaml(
        old_resolved_path,
        {"model": {"activation": {"name": "sigmoid", "gain": 2.5, "bias": 1.0}}},
    )

    config = load_experiment_config(old_resolved_path)

    assert np.isclose(config.model.activation.max_rate, 1.0)
    assert (
        config.simulation.proximal_integration_method
        == PROXIMAL_INTEGRATION_FORWARD_EULER
    )


def test_diagnostics_hyper_config_requires_every_boolean_group(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    diagnostics_path = tmp_path / "diagnostics.yaml"
    save_yaml(experiment_path, {"experiment_name": "composed"})
    save_yaml(
        diagnostics_path,
        {
            "diagnostics": {
                group_name: group_name == "trajectory_and_fixed_points"
                for group_name in DIAGNOSTIC_GROUPS
            },
            "simulation": {"darkness_test_duration": 12.0},
            "tests": {"bump_attractor_initial_conditions": 360},
        },
    )

    config = load_experiment_config(
        experiment_path,
        diagnostics_path=diagnostics_path,
    )

    assert np.isclose(config.simulation.darkness_test_duration, 12.0)
    assert config.diagnostics.trajectory_and_fixed_points is True
    assert config.diagnostics.weight_source == "best"
    assert config.diagnostics.weight_snapshot_index is None
    # trajectory_and_fixed_points carries the trajectory job (which includes
    # the adaptive bisection fixed-point search).
    assert selected_diagnostics(config) == {"bump_attractor_trajectories"}

    invalid_path = tmp_path / "missing_group.yaml"
    save_yaml(
        invalid_path,
        {"diagnostics": {"trajectory_and_fixed_points": True}},
    )
    with pytest.raises(ValueError, match="missing group switches"):
        load_experiment_config(experiment_path, diagnostics_path=invalid_path)

    invalid_source_path = tmp_path / "invalid_weight_source.yaml"
    save_yaml(
        invalid_source_path,
        {
            "diagnostics": {
                **{group_name: False for group_name in DIAGNOSTIC_GROUPS},
                "weight_source": "unknown",
            }
        },
    )
    with pytest.raises(ValueError, match="diagnostics.weight_source"):
        load_experiment_config(
            experiment_path,
            diagnostics_path=invalid_source_path,
        )

    missing_snapshot_path = tmp_path / "missing_snapshot_index.yaml"
    save_yaml(
        missing_snapshot_path,
        {
            "diagnostics": {
                **{group_name: False for group_name in DIAGNOSTIC_GROUPS},
                "weight_source": "snapshot",
            }
        },
    )
    with pytest.raises(ValueError, match="weight_snapshot_index is required"):
        load_experiment_config(
            experiment_path,
            diagnostics_path=missing_snapshot_path,
        )


def test_single_hyper_config_loads_trajectory_parameters() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(
        project_root / "configs" / "experiments" / "vafidis_toy.yaml",
        diagnostics_path=(
            project_root
            / "configs"
            / "diagnostics"
            / "vafidis_diagnostics.yaml"
        ),
    )

    assert config.tests.bump_attractor_initial_conditions == 360
    assert config.tests.bump_attractor_boundary_bisection_depth == 6
    assert np.isclose(config.tests.bump_attractor_duration, 5.0)
    assert np.isclose(config.tests.bump_attractor_cue_duration, 1.0)
    assert np.isclose(config.tests.bump_attractor_cue_amplitude, 24.0)
    assert np.isclose(config.tests.bump_attractor_sample_interval, 0.1)
    np.testing.assert_allclose(
        config.tests.weight_snapshot_pi_velocities,
        np.deg2rad([-75.0, -30.0, -15.0, 0.0, 15.0, 30.0, 75.0]),
    )
    np.testing.assert_allclose(
        config.tests.weight_snapshot_pi_initial_headings,
        np.deg2rad([-135.0, -45.0, 45.0, 135.0]),
    )
    assert np.isclose(config.tests.weight_snapshot_pi_cue_duration, 1.0)
    assert np.isclose(config.tests.weight_snapshot_pi_interval_fraction, 0.02)
    assert np.isclose(config.tests.weight_snapshot_pi_duration, 5.0)
    assert np.isclose(config.tests.weight_snapshot_pi_average_start_time, 0.5)
    assert (
        config.tests.weight_snapshot_pi_selection_metric
        == "rms_velocity_bias"
    )
    assert config.diagnostics.weight_snapshots_and_development is True
    assert config.diagnostics.numerical_convergence is True
    np.testing.assert_allclose(
        config.tests.numerical_convergence_dt_values,
        [0.001, 0.0005, 0.00025, 0.000125],
    )
    assert config.tests.numerical_convergence_methods == [
        "forward_euler",
        "exact_linear",
    ]


def test_diagnostics_config_cannot_change_training_or_model_fields(
    tmp_path: Path,
) -> None:
    diagnostics_path = tmp_path / "invalid_diagnostics.yaml"
    save_yaml(
        diagnostics_path,
        {"model": {"n_theta": 999}},
    )
    experiment_path = tmp_path / "experiment.yaml"
    save_yaml(experiment_path, {})

    with pytest.raises(ValueError, match="simulation/tests"):
        load_experiment_config(experiment_path, diagnostics_path=diagnostics_path)


def test_retired_diagnostics_fields_are_ignored_for_old_resolved_configs(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "old_config_resolved.yaml"
    save_yaml(
        config_path,
        {"tests": {field_name: 1.0 for field_name in RETIRED_TEST_CONFIG_FIELDS}},
    )

    config = load_experiment_config(config_path)

    for field_name in RETIRED_TEST_CONFIG_FIELDS:
        assert not hasattr(config.tests, field_name)


def test_single_hyper_config_contains_shared_extended_parameters() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "experiments" / "vafidis_toy.yaml"
    diagnostics_path = (
        project_root / "configs" / "diagnostics" / "vafidis_diagnostics.yaml"
    )
    config = load_experiment_config(config_path, diagnostics_path=diagnostics_path)
    assert config.tests.velocity_trajectory_sweep_initial_conditions == 1
    assert config.tests.velocity_trajectory_sweep_ring_velocity_count == 4
    assert config.tests.velocity_trajectory_sweep_velocities[:5] == [
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
    ]
    assert config.tests.velocity_phase_flow_initial_conditions == 360
    assert config.tests.velocity_phase_flow_angular_bins == 360
    assert config.tests.velocity_phase_flow_smoothing_bins == 5
    assert config.tests.velocity_phase_flow_probe_velocities[-1] == 0.6
    assert np.isclose(config.tests.velocity_trajectory_sweep_duration, 30.0)
    assert np.isclose(config.simulation.recue_duration, 10.0)


def test_default_visual_noise_uses_ou_current_process() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "vafidis_toy.yaml"
    config = load_experiment_config(config_path)
    assert config.visual.noise_process == "ou_additive"
    assert np.isclose(config.visual.noise_correlation_time, config.model.tau_s)
    assert config.visual.apply_noise_during_training is True
    assert config.visual.apply_noise_during_visual_test is True


def test_section_73_sampling_and_diffusion_defaults() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "vafidis_toy.yaml"
    config = load_experiment_config(config_path)

    assert config.simulation.weight_snapshot_interval_duration is not None
    assert config.simulation.weight_snapshot_interval_duration > 0.0
    assert config.tests.bump_diffusion_trials >= 100
    assert np.isclose(config.tests.bump_diffusion_duration, 10.0)
    assert np.isclose(config.tests.bump_diffusion_cue_duration, 2.0)
    assert np.isclose(config.tests.bump_diffusion_cue_amplitude, 16.0)
    assert np.isclose(config.tests.bump_diffusion_cue_sigma, 0.25)
    assert config.tests.bump_diffusion_release_skip_steps == 5
    assert config.tests.bump_diffusion_integration_method == "forward_euler"
    np.testing.assert_allclose(
        config.tests.bump_diffusion_test_noise_stds,
        np.linspace(0.0, 1.0, 11),
    )
    assert config.tests.bump_diffusion_test_noise_std > 0.0
    assert np.isclose(config.tests.bump_diffusion_fit_start_time, 1.0)
    assert config.tests.bump_diffusion_fit_end_time is None


def test_low_neuron_visual_width_retuning_preserves_resolvable_teacher() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "vafidis_toy.yaml"
    config = load_experiment_config(config_path)
    original_sigma = _visual_sigma_from_kappa(config.visual.kappa)

    _retune_visual_width_for_neuron_count(
        config=config,
        n_theta=16,
        min_visual_sigma_bins=0.70,
    )

    tuned_sigma = _visual_sigma_from_kappa(config.visual.kappa)
    theta_step = 2.0 * np.pi / 8.0
    assert tuned_sigma > original_sigma
    assert np.isclose(tuned_sigma, 0.70 * theta_step)


def test_heterogeneous_experiment_config() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "vafidis_heterogeneous.yaml"
    )
    config = load_experiment_config(config_path)

    assert config.experiment_name == "vafidis_heterogeneous"
    assert config.paths.runs_root == "runs/vafidis_heterogeneous"
    assert config.model.n_theta > 60
    assert config.model.n_hr == config.model.n_theta
    assert config.model.n_theta % 2 == 0
    assert config.visual.profile == "heterogeneous_gaussian_process"
    assert config.visual.heterogeneous_n_angles >= config.model.n_theta
    assert config.visual.heterogeneous_alignment == "center_of_mass"
    assert config.visual.heterogeneous_normalization == "unit_angular_mean"
    reference_mean = np.exp(-config.visual.kappa) * np.i0(config.visual.kappa)
    assert np.isclose(config.visual.amplitude, 4.0 * reference_mean)
    assert config.visual.heterogeneous_plot_sample_count == 16
    assert config.visual.heterogeneous_plot_seed_offset == 50_000
    assert {
        "darkness_path_integration",
        "ou_path_integration",
        "ou_pi_ensemble",
        "bump_attractor_trajectories",
        "weight_snapshot_pi_development",
        "numerical_convergence",
    }.issubset(selected_diagnostics(config))


def test_heterogeneous_experiment_is_matched_to_von_mises_control() -> None:
    experiment_dir = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    control = load_experiment_config(experiment_dir / "vafidis_von_mises.yaml")
    heterogeneous = load_experiment_config(
        experiment_dir / "vafidis_heterogeneous.yaml"
    )

    for section_name in (
        "model",
        "simulation",
        "learning_rule",
        "velocity",
        "diagnostics",
        "tests",
    ):
        assert getattr(heterogeneous, section_name) == getattr(control, section_name)

    control_visual = control.to_dict()["visual"]
    heterogeneous_visual = heterogeneous.to_dict()["visual"]
    differing_visual_fields = {
        field_name
        for field_name in control_visual
        if control_visual[field_name] != heterogeneous_visual[field_name]
    }
    assert differing_visual_fields == {
        "profile",
        "amplitude",
        "heterogeneous_n_angles",
    }
    assert heterogeneous.paths.reports_root == control.paths.reports_root


def test_population_mean_heterogeneous_experiment_config() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "vafidis_population_mean_heterogeneous.yaml"
    )
    config = load_experiment_config(config_path)

    assert config.model.hd_distal_normalization == "presynaptic_population_mean"
    assert config.experiment_name == "vafidis_population_mean_heterogeneous"
    assert config.model.n_theta == config.model.n_hr
    assert config.model.n_theta % 2 == 0
    assert config.simulation.random_stream_mode == "component_split"
    assert config.simulation.plasticity_enabled is True
    assert config.simulation.early_stopping.enabled is True
    assert (
        config.simulation.early_stopping.min_duration
        < config.simulation.train_duration
    )
    assert config.simulation.early_stopping.check_interval > 0.0
    assert (
        config.simulation.early_stopping.window_duration
        >= config.simulation.early_stopping.check_interval
    )
    assert config.visual.heterogeneous_population_sampling == "nested_master"
    assert config.visual.heterogeneous_master_n_hd_cells >= config.model.n_theta
    assert config.visual.heterogeneous_master_n_hd_cells % config.model.n_theta == 0
    assert config.tests.bump_attractor_initial_conditions > 0
    assert config.tests.bump_attractor_duration > 0.0
    assert config.tests.bump_attractor_sample_interval > 0.0
    assert "ref" not in config.model.hd_distal_normalization
