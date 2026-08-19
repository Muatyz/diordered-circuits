from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learning.common.random import make_rng
from learning.config.load_config import load_experiment_config
from learning.config.schema import (
    TRAINING_INTEGRATION_BLOCK_MULTIRATE,
    TRAINING_INTEGRATION_SINGLE_CLOCK,
    ExperimentConfig,
)
from learning.experiments.run_vafidis_toy import run_training
from learning.models.vafidis_toy import VafidisToyParams


def make_training_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.n_theta = 6
    config.model.n_hr = 6
    config.simulation.dt = 0.0005
    config.simulation.train_duration = 0.01
    config.simulation.save_interval_duration = 0.005
    config.simulation.weight_snapshot_interval_duration = 0.005
    config.simulation.plasticity_update_interval_duration = 0.001
    config.simulation.progress = False
    config.simulation.random_stream_mode = "component_split"
    config.visual.noise_std = 0.0
    return config


def test_block_size_one_matches_single_clock_training() -> None:
    single_clock_config = make_training_config()
    single_clock_config.simulation.training_integration_method = (
        TRAINING_INTEGRATION_SINGLE_CLOCK
    )
    single_clock_config.simulation.plasticity_update_interval_duration = (
        single_clock_config.simulation.dt
    )
    block_config = make_training_config()
    block_config.simulation.training_integration_method = (
        TRAINING_INTEGRATION_BLOCK_MULTIRATE
    )
    block_config.simulation.plasticity_update_interval_duration = (
        block_config.simulation.dt
    )

    single_state, single_history, single_weights = run_training(
        config=single_clock_config,
        rng=make_rng(single_clock_config.simulation.seed),
    )
    block_state, block_history, block_weights = run_training(
        config=block_config,
        rng=make_rng(block_config.simulation.seed),
    )

    for field_name in (
        "r_hd",
        "r_hr",
        "v_hd_distal",
        "v_hd_proximal",
        "delta_w_hd_to_hd",
        "delta_w_hr_to_hd",
        "w_hd_to_hd",
        "w_hr_to_hd",
    ):
        np.testing.assert_allclose(
            getattr(block_state, field_name),
            getattr(single_state, field_name),
            rtol=1e-13,
            atol=1e-13,
        )
    np.testing.assert_allclose(block_history["rms_e_hd"], single_history["rms_e_hd"])
    np.testing.assert_allclose(block_weights["w_hd_to_hd"], single_weights["w_hd_to_hd"])
    assert str(block_history["training_integration_method"]) == "block_multirate"
    assert int(block_history["plasticity_update_interval_steps"]) == 1


def test_multirate_training_flushes_partial_final_block() -> None:
    config = make_training_config()
    config.simulation.training_integration_method = (
        TRAINING_INTEGRATION_BLOCK_MULTIRATE
    )
    config.simulation.train_duration = 0.0115
    config.simulation.plasticity_update_interval_duration = 0.002

    state, history, _weight_history = run_training(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    assert np.isclose(state.time, config.simulation.train_duration)
    assert np.any(np.abs(state.delta_w_hd_to_hd) > 0.0)
    assert np.all(np.isfinite(state.w_hd_to_hd))
    assert int(history["plasticity_update_interval_steps"]) == 4


def test_matched_noise_short_training_stays_close_to_single_clock() -> None:
    single_clock_config = make_training_config()
    single_clock_config.simulation.train_duration = 0.05
    single_clock_config.simulation.save_interval_duration = 0.05
    single_clock_config.simulation.weight_snapshot_interval_duration = 0.05
    single_clock_config.simulation.training_integration_method = (
        TRAINING_INTEGRATION_SINGLE_CLOCK
    )
    block_config = make_training_config()
    block_config.simulation.train_duration = 0.05
    block_config.simulation.save_interval_duration = 0.05
    block_config.simulation.weight_snapshot_interval_duration = 0.05
    block_config.simulation.training_integration_method = (
        TRAINING_INTEGRATION_BLOCK_MULTIRATE
    )
    block_config.simulation.plasticity_update_interval_duration = 0.002

    single_state, _single_history, _single_weights = run_training(
        config=single_clock_config,
        rng=make_rng(single_clock_config.simulation.seed),
    )
    block_state, _block_history, _block_weights = run_training(
        config=block_config,
        rng=make_rng(block_config.simulation.seed),
    )

    for field_name in ("w_hd_to_hd", "w_hr_to_hd"):
        single_weight = getattr(single_state, field_name)
        relative_error = np.linalg.norm(
            getattr(block_state, field_name) - single_weight
        ) / max(np.linalg.norm(single_weight), 1e-15)
        assert relative_error < 1e-4
    assert np.sqrt(np.mean(np.square(block_state.r_hd - single_state.r_hd))) < 1e-6


def test_block_multirate_profile_is_selectable_from_config() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(
        project_root / "configs" / "experiments" / "vafidis_toy.yaml",
        profile_paths=[
            project_root / "configs" / "profiles" / "block_multirate.yaml"
        ],
    )

    assert (
        config.simulation.training_integration_method
        == TRAINING_INTEGRATION_BLOCK_MULTIRATE
    )
    assert np.isclose(config.simulation.plasticity_update_interval_duration, 0.01)


def test_multirate_config_requires_known_method_and_integer_block() -> None:
    config = make_training_config()
    config.simulation.training_integration_method = "unknown_solver"
    with pytest.raises(ValueError, match="training_integration_method"):
        VafidisToyParams.from_config(config)

    config.simulation.training_integration_method = (
        TRAINING_INTEGRATION_BLOCK_MULTIRATE
    )
    config.simulation.plasticity_update_interval_duration = 0.0007
    with pytest.raises(ValueError, match="integer multiple"):
        VafidisToyParams.from_config(config)
