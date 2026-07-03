from __future__ import annotations

import numpy as np

from learning.common.random import make_rng
from learning.config.schema import ExperimentConfig
from learning.experiments.run_vafidis_toy import run_experiment
from learning.io.save_load import load_json, load_npz
from learning.models.vafidis_toy import VafidisToyParams, initialize_vafidis_toy_state, step_vafidis_toy


def make_short_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.n_theta = 12
    config.model.n_hr = 12
    config.simulation.seed = 3
    config.simulation.dt = 0.01
    config.simulation.train_duration = 0.05
    config.simulation.bump_test_duration = 0.03
    config.simulation.darkness_test_duration = 0.03
    config.simulation.cue_duration = 0.02
    config.simulation.pi_cue_duration = 0.04
    config.simulation.recue_duration = 0.02
    config.simulation.save_interval_steps = 1
    config.simulation.progress = False
    config.tests.gain_velocities = [-0.5, 0.5]
    return config


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


def test_short_experiment_writes_required_outputs(tmp_path) -> None:
    config = make_short_config()
    run_dir = run_experiment(config=config, project_root=tmp_path, make_figures=False)
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "params.json").exists()
    assert (run_dir / "trained_weights.npz").exists()
    assert (run_dir / "training_history.npz").exists()
    assert (run_dir / "ou_darkness_history.npz").exists()
    assert (run_dir / "test_metrics.json").exists()
    training_history = load_npz(run_dir / "training_history.npz")
    test_metrics = load_json(run_dir / "test_metrics.json")
    assert "pva_strength_hd" in training_history
    assert "bump_contrast_hd" in training_history
    assert "theta_hd_decoded_peak" in training_history
    assert "contrast_r_lhr" in training_history
    assert "contrast_r_rhr" in training_history
    darkness_history = load_npz(run_dir / "darkness_history.npz")
    ou_darkness_history = load_npz(run_dir / "ou_darkness_history.npz")
    bump_history = load_npz(run_dir / "bump_history.npz")
    assert "phase_id" in darkness_history
    assert "visual_teacher" in darkness_history
    assert "phase_id" in ou_darkness_history
    assert np.count_nonzero(bump_history["phase_id"] == 0.0) == 3
    assert np.count_nonzero(darkness_history["phase_id"] == 0.0) == 5
    assert np.count_nonzero(darkness_history["phase_id"] == 1.0) == 3
    assert np.count_nonzero(darkness_history["phase_id"] == 2.0) == 2
    assert "darkness_final_pva_strength" in test_metrics
    assert "darkness_final_bump_contrast" in test_metrics
    assert "darkness_mean_saturated_hd_bins" in test_metrics
    assert "bump_final_saturated_hd_bins" in test_metrics
    assert "darkness_mean_near_peak_hd_bins" in test_metrics
    assert "bump_final_near_peak_hd_bins" in test_metrics
    assert "darkness_recue_final_abs_pi_error" in test_metrics
    assert "ou_darkness_rms_pi_error" in test_metrics
    assert "ou_darkness_recue_final_abs_pi_error" in test_metrics
    assert "bump_intrinsic_drift_velocity_deg_s" in test_metrics
    assert "darkness_peak_decoded_velocity" in test_metrics
