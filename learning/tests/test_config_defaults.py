from __future__ import annotations

from pathlib import Path

import numpy as np

from learning.config.load_config import load_experiment_config
from learning.experiments.run_attractor_robustness import (
    _retune_visual_width_for_neuron_count,
    _visual_sigma_from_kappa,
)


def test_default_training_protocol_has_positive_duration() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "vafidis_toy.yaml"
    config = load_experiment_config(config_path)
    assert config.simulation.train_duration > 0.0


def test_all_single_mouse_configs_expose_configurable_attractor_landscape_protocol() -> None:
    experiment_config_dir = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    config_paths = sorted(experiment_config_dir.glob("*.yaml"))
    assert {config_path.name for config_path in config_paths} == {
        "vafidis_toy.yaml",
        "vafidis_mammalian_heterogeneous.yaml",
        "vafidis_population_mean_heterogeneous.yaml",
        "vafidis_population_mean_von_mises.yaml",
    }

    for config_path in config_paths:
        config = load_experiment_config(config_path)
        assert config.tests.bump_attractor_trajectory_enabled is True
        assert config.tests.bump_attractor_initial_conditions > 0
        assert config.tests.bump_attractor_duration > 0.0
        assert config.tests.bump_attractor_cue_duration >= 0.0
        assert config.tests.bump_attractor_sample_interval > 0.0

    von_mises_config = load_experiment_config(
        experiment_config_dir / "vafidis_population_mean_von_mises.yaml"
    )
    assert von_mises_config.tests.timescale_separation_enabled is True
    assert von_mises_config.tests.timescale_separation_initial_conditions == 12
    assert np.isclose(
        von_mises_config.tests.timescale_separation_ratio_threshold,
        10.0,
    )
    assert von_mises_config.tests.velocity_trajectory_sweep_enabled is True
    assert np.isclose(
        von_mises_config.tests.velocity_trajectory_sweep_max_velocity,
        2.0,
    )
    assert von_mises_config.tests.velocity_trajectory_sweep_count == 11
    assert von_mises_config.tests.velocity_trajectory_sweep_ring_velocity_count == 4
    assert np.isclose(
        von_mises_config.tests.velocity_trajectory_sweep_zero_input_flow_window,
        5.0,
    )
    assert von_mises_config.tests.velocity_phase_flow_probe_enabled is True
    assert von_mises_config.tests.velocity_phase_flow_initial_conditions == 360
    assert np.isclose(
        von_mises_config.tests.velocity_phase_flow_fit_start_time,
        0.0,
    )
    assert np.isclose(
        von_mises_config.tests.velocity_phase_flow_fit_duration,
        2.0,
    )
    assert np.isclose(
        von_mises_config.tests.velocity_trajectory_sweep_initial_heading,
        0.0,
    )
    assert von_mises_config.tests.velocity_trajectory_sweep_initial_conditions == 1
    assert np.isclose(
        von_mises_config.tests.velocity_trajectory_sweep_duration,
        30.0,
    )


def test_default_pi_recue_is_long_enough_to_check_relocking() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "vafidis_toy.yaml"
    config = load_experiment_config(config_path)
    assert config.tests.velocity_trajectory_sweep_enabled is True
    assert config.tests.velocity_trajectory_sweep_initial_conditions == 1
    assert config.tests.velocity_trajectory_sweep_ring_velocity_count == 4
    assert config.tests.velocity_trajectory_sweep_velocities[:5] == [
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
    ]
    assert np.isclose(
        config.tests.velocity_trajectory_sweep_zero_input_flow_window,
        5.0,
    )
    assert config.tests.velocity_phase_flow_probe_enabled is True
    assert config.tests.velocity_phase_flow_initial_conditions == 360
    assert config.tests.velocity_phase_flow_angular_bins == 360
    assert config.tests.velocity_phase_flow_smoothing_bins == 5
    assert config.tests.velocity_phase_flow_probe_velocities[-1] == 0.6
    assert np.isclose(config.tests.velocity_trajectory_sweep_duration, 30.0)
    assert np.isclose(config.simulation.recue_duration, 8.0)


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

    assert config.simulation.weight_snapshot_interval_steps is not None
    assert config.simulation.weight_snapshot_interval_steps > 0
    assert config.tests.bump_diffusion_trials >= 100
    assert np.isclose(config.tests.bump_diffusion_duration, 10.0)
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


def test_mammalian_heterogeneous_experiment_config() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "vafidis_mammalian_heterogeneous.yaml"
    )
    config = load_experiment_config(config_path)

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
    assert config.tests.bump_diffusion_trials == 300
    assert config.tests.ou_pi_ensemble_trials > 1
    assert config.tests.hd_tuning_curve_angles >= 60
    assert config.tests.hd_tuning_curve_settle_duration >= 1.4
    assert (
        config.tests.hd_tuning_curve_max_settle_duration
        > config.tests.hd_tuning_curve_settle_duration
    )
    assert config.tests.hd_tuning_curve_convergence_tolerance is not None


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
    assert config.tests.bump_attractor_trajectory_enabled is True
    assert config.tests.bump_attractor_initial_conditions > 0
    assert config.tests.bump_attractor_duration > 0.0
    assert config.tests.bump_attractor_sample_interval > 0.0
    assert "ref" not in config.model.hd_distal_normalization
