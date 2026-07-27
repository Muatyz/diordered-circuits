from __future__ import annotations

from pathlib import Path

import numpy as np

from learning.common.angles import make_vafidis_paired_theta_hd_pref
from learning.config.load_config import load_yaml
from learning.experiments.run_hr_parameter_sweep import (
    aggregate_trial_rows,
    make_antisymmetric_shifted_hr_kernel,
    run_sweep,
)


def test_hr_phase1b_config_scans_two_hd_backgrounds_and_both_orientations() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / "configs" / "analysis" / "population_mean_hr_velocity_phase1b.yaml")

    assert len(config["hd_kernels"]) == 2
    assert config["hr_kernel"]["orientations"] == [-1.0, 1.0]
    assert len(config["protocol"]["cue_conditions"]) == 2
    assert 0.0 in config["protocol"]["gain_velocities"]
    assert config["acceptance"]["maximum_velocity_gain_abs_error"] == 0.25


def test_antisymmetric_hr_kernel_has_exact_wing_cancellation() -> None:
    theta = make_vafidis_paired_theta_hd_pref(120)
    weights = make_antisymmetric_shifted_hr_kernel(
        theta_hd_preference=theta,
        scale=40.0,
        sigma_radians=0.35,
        shift_radians=0.25,
        orientation=1.0,
    )
    left, right = np.split(weights, 2, axis=1)

    assert weights.shape == (120, 120)
    np.testing.assert_allclose(right, -left)
    assert np.min(left) < 0.0 < np.max(left)


def test_hr_aggregate_prefers_successful_low_error_candidate() -> None:
    base = {
        "hd_kernel_name": "hd",
        "cue_condition": "clean",
        "hr_scale": 40.0,
        "hr_sigma_radians": 0.35,
        "hr_shift_radians": 0.25,
        "hr_orientation": 1.0,
        "velocity_gain": 1.02,
        "velocity_gain_abs_error": 0.02,
        "velocity_gain_intercept_abs": 0.01,
        "velocity_gain_r_squared": 0.99,
        "velocity_tracking_rmse": 0.05,
        "bump_final_pva_strength": 0.9,
        "bump_final_contrast": 0.8,
        "bump_abs_release_shift_degrees": 1.0,
        "passes_all": True,
    }
    aggregate = aggregate_trial_rows([base, dict(base)])

    assert aggregate[0]["success_fraction"] == 1.0
    assert aggregate[0]["mean_velocity_gain"] == 1.02


def test_hr_phase1b_dry_run_counts_candidates(capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "analysis" / "population_mean_hr_velocity_phase1b.yaml"

    assert run_sweep(sweep_config_path=path, dry_run=True) is None
    output = capsys.readouterr().out
    assert "trials: 504" in output
    assert "HR candidates per HD kernel and cue: 126" in output
