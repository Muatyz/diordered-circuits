import numpy as np

from learning.common.random import make_rng
from learning.config.schema import ExperimentConfig
from learning.dynamics.autonomous import FrozenAutonomousDynamics
from learning.experiments.run_vafidis_toy import (
    run_bump_attractor_trajectory_test,
    run_hd_tuning_curve_test,
    run_slow_manifold_diagnostic,
)
from learning.models.vafidis_toy import VafidisToyParams, initialize_vafidis_toy_state
from learning.plotting.slow_manifold import (
    plot_ramesan_firing_rate_diagnostics,
    plot_ramesan_pca_variance_rank,
    plot_ramesan_phase_landscape,
    plot_slow_manifold_diagnostics,
)


def test_short_sagodi_pipeline_reports_coverage_and_runs_ramesan(tmp_path) -> None:
    config = ExperimentConfig()
    config.model.n_theta = 8
    config.model.n_hr = 8
    config.simulation.dt = 0.0005
    config.simulation.progress = False
    config.tests.hd_tuning_curve_angles = 8
    config.tests.hd_tuning_curve_settle_duration = 0.01
    config.tests.hd_tuning_curve_sample_count = 1
    config.diagnostics.trajectory_and_fixed_points = True
    config.diagnostics.pva_spectrum_and_visualization = True
    config.tests.bump_attractor_initial_conditions = 8
    config.tests.bump_attractor_duration = 0.05
    config.tests.bump_attractor_cue_duration = 0.01
    config.tests.bump_attractor_sample_interval = 0.01
    # A permissive threshold keeps this tiny smoke test independent of whether
    # an untrained random network has already entered a genuinely slow regime.
    config.tests.slow_manifold_speed_fraction = 0.99
    config.tests.slow_manifold_candidate_count = 64
    config.tests.slow_manifold_angular_bins = 8
    config.tests.slow_manifold_jacobian_anchors = 3
    config.tests.slow_manifold_jacobian_eigenvalues = 4
    config.tests.slow_manifold_jacobian_dense_dimension_limit = 64
    config.tests.ramesan_q_threshold = 1e12
    config.tests.ramesan_trajectory_sample_count = 32
    config.tests.ramesan_phase_angular_bins = 8
    config.tests.ramesan_phase_smoothing_bins = 3
    config.tests.ramesan_ambient_enabled = True
    config.tests.ramesan_ambient_sample_count = 16
    config.tests.ramesan_ambient_perturbation_scales = [0.02, 0.05]

    state = initialize_vafidis_toy_state(config=config, rng=make_rng(13))
    tuning_history, _ = run_hd_tuning_curve_test(
        config=config,
        trained_state=state,
    )
    trajectory_history, _ = run_bump_attractor_trajectory_test(
        config=config,
        trained_state=state,
        hd_tuning_history=tuning_history,
    )
    assert trajectory_history["autonomous_probe_phase"].shape == (8,)
    assert trajectory_history["autonomous_probe_state"].shape == (8, 40)
    assert trajectory_history["ramesan_trajectory_state"].shape == (32, 40)
    slow_history, slow_metrics = run_slow_manifold_diagnostic(
        config=config,
        trained_state=state,
        bump_attractor_trajectory_history=trajectory_history,
    )

    # This untrained random network collapses its slow candidates into one
    # angular cluster. The pipeline must report insufficient ring coverage,
    # while the trajectory-conditioned Ramesan diagnostics still run.
    assert slow_metrics["slow_manifold_fit_succeeded"] == 0.0
    assert slow_metrics["slow_manifold_fit_failure_is_insufficient_coverage"] == 1.0
    assert np.isclose(slow_metrics["slow_manifold_angular_support_fraction"], 0.25)
    assert slow_metrics["ramesan_diagnostic_succeeded"] == 1.0
    assert slow_history["manifold_state"].shape == (0, 0)
    assert slow_history["jacobian_eigenvalue_real"].shape == (0, 0)
    assert slow_history["ramesan_probe_pc"].shape == (8, 3)
    assert slow_history["ramesan_pca_center"].shape == (40,)
    assert slow_history["ramesan_pca_feature_scale"].shape == (40,)
    assert slow_history["ramesan_pca_components"].shape == (3, 40)
    assert slow_history["ramesan_pca_standardized_components"].shape == (8, 40)
    assert slow_history["ramesan_pca_explained_variance_spectrum"].shape == (8,)
    assert slow_history["ramesan_pca_cumulative_explained_variance"].shape == (8,)
    assert slow_history[
        "ramesan_firing_rate_pca_explained_variance_spectrum"
    ].shape == (8,)
    assert slow_history[
        "ramesan_firing_rate_pca_cumulative_explained_variance"
    ].shape == (8,)
    assert slow_history[
        "ramesan_pva_rate_pca_explained_variance_spectrum"
    ].shape == (4,)
    assert slow_history[
        "ramesan_pva_rate_pca_cumulative_explained_variance"
    ].shape == (4,)
    np.testing.assert_allclose(
        slow_history["ramesan_pca_cumulative_explained_variance"][-1],
        1.0,
    )
    np.testing.assert_allclose(
        slow_history[
            "ramesan_firing_rate_pca_cumulative_explained_variance"
        ][-1],
        1.0,
    )
    np.testing.assert_allclose(
        slow_history["ramesan_pva_rate_pca_cumulative_explained_variance"][-1],
        1.0,
    )
    pva_vector = slow_history["ramesan_pva_angular_rate"] @ np.exp(
        1j * slow_history["ramesan_pva_theta_pref"]
    )
    np.testing.assert_allclose(
        np.exp(1j * np.angle(pva_vector)),
        np.exp(1j * slow_history["ramesan_probe_decoded_theta"]),
    )
    reconstructed_pc = (
        slow_history["ramesan_probe_state"]
        - slow_history["ramesan_pca_center"]
    ) @ slow_history["ramesan_pca_components"].T
    np.testing.assert_allclose(reconstructed_pc, slow_history["ramesan_probe_pc"])
    dynamics = FrozenAutonomousDynamics.from_state(
        params=VafidisToyParams.from_config(config),
        state=state,
    )
    expected_hd_rate = np.stack(
        [
            dynamics.hd_rate(state_vector)
            for state_vector in slow_history["ramesan_probe_state"]
        ]
    )
    np.testing.assert_allclose(
        slow_history["ramesan_probe_firing_rate"][:, : config.model.n_theta],
        expected_hd_rate,
    )
    assert slow_history["ramesan_jacobian_lambda_max_real"].shape == (3,)
    assert slow_metrics["ramesan_phase_landscape_succeeded"] == 1.0
    assert slow_history["ramesan_phase_bin_center"].shape == (8,)
    assert slow_history["ramesan_ambient_q"].shape == (16,)
    np.testing.assert_allclose(
        slow_history["ramesan_candidate_q"],
        0.5 * np.square(trajectory_history["slow_candidate_speed"]),
    )
    figure_path = tmp_path / "slow_manifold.png"
    plot_slow_manifold_diagnostics(history=slow_history, path=figure_path)
    assert figure_path.exists()
    ramesan_figure_path = tmp_path / "ramesan_firing_rate.png"
    plot_ramesan_firing_rate_diagnostics(
        history=slow_history,
        path=ramesan_figure_path,
    )
    assert ramesan_figure_path.exists()
    variance_rank_path = tmp_path / "ramesan_pca_variance_rank.png"
    plot_ramesan_pca_variance_rank(
        history=slow_history,
        path=variance_rank_path,
    )
    assert variance_rank_path.exists()
    phase_figure_path = tmp_path / "ramesan_phase_landscape.png"
    plot_ramesan_phase_landscape(
        history=slow_history,
        path=phase_figure_path,
    )
    assert phase_figure_path.exists()


def test_sparse_low_speed_angles_reject_ring_fit() -> None:
    config = ExperimentConfig()
    config.model.n_theta = 8
    config.model.n_hr = 8
    config.diagnostics.pva_spectrum_and_visualization = True
    config.tests.slow_manifold_angular_bins = 16
    config.tests.slow_manifold_min_angular_support_fraction = 0.5
    state = initialize_vafidis_toy_state(config=config, rng=make_rng(17))
    packed_dimension = 40
    candidate_count = 12
    history = {
        "slow_candidate_theta": np.zeros(candidate_count),
        "slow_candidate_state": np.zeros((candidate_count, packed_dimension)),
        "slow_candidate_speed": np.linspace(0.0, 0.01, candidate_count),
    }
    slow_history, metrics = run_slow_manifold_diagnostic(
        config=config,
        trained_state=state,
        bump_attractor_trajectory_history=history,
    )
    assert metrics["slow_manifold_fit_succeeded"] == 0.0
    assert metrics["slow_manifold_fit_failure_is_insufficient_coverage"] == 1.0
    assert metrics["slow_manifold_low_speed_angle_cluster_count"] == 1.0
    assert slow_history["manifold_theta"].size == 0
