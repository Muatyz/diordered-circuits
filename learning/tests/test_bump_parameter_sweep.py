from __future__ import annotations

from pathlib import Path

import numpy as np

from learning.common.angles import make_vafidis_paired_theta_hd_pref
from learning.config.load_config import load_yaml
from learning.experiments.run_bump_parameter_sweep import (
    aggregate_trial_rows,
    make_gaussian_minus_uniform_kernel,
    run_sweep,
)


def test_phase1a_config_is_frozen_n120_factorial_search() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "analysis" / "population_mean_bump_phase1a.yaml"
    config = load_yaml(config_path)

    assert config["protocol"]["n_hd"] == 120
    assert len(config["protocol"]["cue_conditions"]) == 2
    assert config["kernel"]["form"] == "gaussian_minus_uniform"
    assert max(config["kernel"]["excitatory_scales"]) == 120.0
    assert config["kernel"]["include_self_connections"] is False
    assert config["acceptance"]["minimum_final_pva_strength"] == 0.5
    assert config["acceptance"]["minimum_dark_pva_strength"] == 0.4


def test_gaussian_minus_uniform_kernel_is_circular_and_has_zero_diagonal() -> None:
    theta = make_vafidis_paired_theta_hd_pref(12)
    kernel = make_gaussian_minus_uniform_kernel(
        theta_preference=theta,
        excitatory_scale=20.0,
        sigma_radians=0.45,
        inhibitory_ratio=0.25,
        include_self_connections=False,
    )

    np.testing.assert_allclose(kernel, kernel.T)
    np.testing.assert_allclose(np.diag(kernel), 0.0)
    # Paired HD cells have identical preferred directions and therefore
    # identical kernel rows apart from their excluded self entries.
    assert kernel.shape == (12, 12)
    assert np.min(kernel) < 0.0 < np.max(kernel)


def test_aggregate_trial_rows_rewards_robust_success() -> None:
    common = {
        "cue_condition": "clean",
        "excitatory_scale": 20.0,
        "sigma_radians": 0.45,
        "inhibitory_ratio": 0.1,
        "dark_final_pva_strength": 0.8,
        "dark_minimum_pva_strength": 0.75,
        "dark_final_contrast": 0.7,
        "dark_minimum_contrast": 0.65,
        "contrast_retention": 0.9,
        "abs_release_shift_degrees": 2.0,
        "dark_max_saturated_fraction": 0.0,
        "dark_final_mean_rate": 0.1,
        "dark_final_max_rate": 0.8,
        "dark_final_heading_error_degrees": 1.0,
        "dark_final_half_max_width_degrees": 24.0,
    }
    rows = [
        {**common, "theta0": -1.0, "passes_all": True},
        {**common, "theta0": 1.0, "passes_all": True},
    ]

    aggregate = aggregate_trial_rows(rows)

    assert len(aggregate) == 1
    assert aggregate[0]["success_fraction"] == 1.0
    assert aggregate[0]["minimum_final_pva_strength"] == 0.8


def test_phase1a_dry_run_validates_trial_count(capsys) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "analysis" / "population_mean_bump_phase1a.yaml"

    result = run_sweep(sweep_config_path=config_path, dry_run=True)

    assert result is None
    assert "trials: 1200" in capsys.readouterr().out


def test_refined_phase1a_uses_condition_specific_seeds_and_half_bin_headings(
    capsys,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = (
        project_root
        / "configs"
        / "analysis"
        / "population_mean_bump_phase1a_refined.yaml"
    )
    config = load_yaml(config_path)

    assert config["protocol"]["cue_conditions"][0]["seed_offsets"] == [0]
    assert config["protocol"]["cue_conditions"][1]["seed_offsets"] == [0, 1, 2, 3, 4]
    assert config["acceptance"]["maximum_final_max_rate"] == 0.98
    assert run_sweep(sweep_config_path=config_path, dry_run=True) is None
    assert "trials: 5400" in capsys.readouterr().out
