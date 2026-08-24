from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learning.config.load_config import load_experiment_config
from learning.diagnostics.protocols import summarize_frozen_velocity_probe_grid
from learning.diagnostics.weight_development import (
    run_weight_snapshot_pi_development_diagnostic,
)
from learning.models.vafidis_toy import initialize_vafidis_toy_state
from learning.stimuli.velocity import ScheduledOUAngularVelocity


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_scheduled_ou_uses_configured_training_fractions() -> None:
    process = ScheduledOUAngularVelocity(
        mean=0.0,
        tau=0.5,
        clip=None,
        rng=np.random.default_rng(1),
        total_duration=10.0,
        std_schedule=[
            {"end_fraction": 0.5, "std": 3.0},
            {"end_fraction": 0.75, "std": 2.0},
            {"end_fraction": 1.0, "std": 1.0},
        ],
    )
    assert process.current_std() == 3.0
    process.elapsed_time = 6.0
    assert process.current_std() == 2.0
    process.elapsed_time = 9.0
    assert process.current_std() == 1.0


def test_scheduled_ou_rejects_incomplete_schedule() -> None:
    with pytest.raises(ValueError, match="end at fraction 1.0"):
        ScheduledOUAngularVelocity(
            mean=0.0,
            tau=0.5,
            clip=None,
            rng=np.random.default_rng(1),
            total_duration=10.0,
            std_schedule=[{"end_fraction": 0.5, "std": 1.0}],
        )


def test_depinning_requires_all_headings_and_both_turning_directions() -> None:
    commanded = np.asarray([-0.4, -0.2, 0.0, 0.2, 0.4])
    decoded = np.asarray(
        [
            [-0.38, -0.16, 0.01, 0.15, 0.38],
            [-0.36, -0.05, -0.01, 0.14, 0.36],
        ]
    )
    summary = summarize_frozen_velocity_probe_grid(
        commanded_velocity=commanded,
        decoded_velocity=decoded,
        minimum_pva_strength=np.full(decoded.shape, 0.9),
        minimum_bump_contrast=np.full(decoded.shape, 0.2),
        minimum_moving_gain=0.5,
    )
    assert np.isclose(summary["negative_depinning_velocity"], 0.4)
    assert np.isclose(summary["positive_depinning_velocity"], 0.2)
    assert np.isclose(summary["depinning_velocity"], 0.4)
    assert np.isclose(summary["maximum_abs_zero_velocity_drift"], 0.01)


def test_offline_selection_prefers_accepted_snapshot_before_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config(
        PROJECT_ROOT / "configs" / "experiments" / "vafidis_toy.yaml",
        profile_paths=[
            PROJECT_ROOT / "configs" / "profiles" / "code_smoke.yaml"
        ],
    )
    config.tests.weight_snapshot_pi_velocities = [-0.2, 0.0, 0.2]
    config.tests.weight_snapshot_pi_initial_headings = [0.0]
    config.tests.weight_snapshot_pi_selection_metric = (
        "maximum_abs_velocity_bias"
    )
    config.tests.weight_snapshot_pi_maximum_depinning_velocity = 0.1
    state = initialize_vafidis_toy_state(
        config=config,
        rng=np.random.default_rng(2),
    )
    weights_hd = np.stack(
        [state.w_hd_to_hd.copy(), state.w_hd_to_hd.copy() + 1.0]
    )
    weights_hr = np.stack(
        [state.w_hr_to_hd.copy(), state.w_hr_to_hd.copy() + 1.0]
    )

    def fake_grid(*, trained_state, initial_headings, velocities, **_kwargs):
        accepted_candidate = bool(np.mean(trained_state.w_hd_to_hd) > 0.5)
        shape = (len(initial_headings), len(velocities))
        bias = 0.02 if accepted_candidate else 0.01
        details = {
            name: np.full(shape, value, dtype=float)
            for name, value in {
                "time_averaged_abs_error": bias,
                "rms_error": bias,
                "final_abs_error": bias,
                "wrapped_time_averaged_abs_error": bias,
                "wrapped_rms_error": bias,
                "wrapped_final_abs_error": bias,
                "decoded_velocity": 0.0,
                "velocity_bias": bias,
                "velocity_gain": 1.0,
                "stalled": 0.0,
                "mean_pva_strength": 0.9,
                "minimum_pva_strength": 0.9,
                "mean_bump_contrast": 0.2,
                "minimum_bump_contrast": 0.2,
            }.items()
        }
        depinning = 0.05 if accepted_candidate else 0.2
        summary = {
            "rms_velocity_bias": bias,
            "maximum_abs_velocity_bias": bias,
            "maximum_abs_zero_velocity_drift": 0.0,
            "negative_depinning_velocity": depinning,
            "positive_depinning_velocity": depinning,
            "depinning_velocity": depinning,
            "stall_fraction": 0.0,
            "minimum_pva_strength": 0.9,
            "minimum_bump_contrast": 0.2,
            "fully_defined": 1.0,
        }
        return details, summary

    monkeypatch.setattr(
        "learning.diagnostics.weight_development.run_frozen_velocity_probe_grid",
        fake_grid,
    )
    history, metrics = run_weight_snapshot_pi_development_diagnostic(
        config=config,
        trained_state=state,
        weight_history={
            "time": np.asarray([0.0, 1.0]),
            "w_hd_to_hd": weights_hd,
            "w_hr_to_hd": weights_hr,
        },
    )
    assert int(history["best_snapshot_index"]) == 1
    assert metrics["weight_snapshot_pi_best_acceptance_passed"] == 1.0
    assert metrics["weight_snapshot_pi_selection_was_fallback"] == 0.0


def test_pi_robust_profiles_compose_without_changing_learning_rule() -> None:
    base_path = PROJECT_ROOT / "configs" / "experiments" / "vafidis_toy.yaml"
    robust_path = PROJECT_ROOT / "configs" / "profiles" / "pi_robust_vafidis.yaml"
    n120_path = PROJECT_ROOT / "configs" / "profiles" / "pi_robust_n120.yaml"
    base = load_experiment_config(base_path)
    robust = load_experiment_config(base_path, profile_paths=[robust_path, n120_path])
    assert robust.model.n_theta == robust.model.n_hr == 120
    assert robust.velocity.training_ou_std_schedule[-1]["std"] < base.velocity.std
    assert robust.simulation.checkpoint_selection.velocity_during_cue
    assert robust.learning_rule == base.learning_rule
