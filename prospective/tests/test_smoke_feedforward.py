from dataclasses import replace

import numpy as np

from prospective.config.schema import ExperimentConfig
from prospective.experiments.run_feedforward import train_feedforward
from prospective.experiments.run_prospective_probe import run_single_probe
from prospective.theory.equilibrium import theoretical_weights
from prospective.common.geometry import uniform_positions


def _smoke_config(tmp_path):
    base = ExperimentConfig()
    return replace(
        base,
        experiment=replace(base.experiment, name="smoke", output_root=str(tmp_path), seed=7),
        geometry=replace(base.geometry, n_input=8, n_competitive=8, length=40.0),
        tutor=replace(base.tutor, sigma=3.0, speed=10.0),
        neural=replace(base.neural, inhibition_strength=2.0),
        simulation=replace(base.simulation, duration=0.2, state_sample_interval_steps=2, weight_snapshot_interval_steps=10, progress=False),
        analysis=replace(base.analysis, transient_duration=0.0),
        animation=replace(base.animation, enabled=True),
    )


def test_short_training_saves_reloadable_arrays(tmp_path):
    config = _smoke_config(tmp_path)
    result = train_feedforward(config, output_cwd=tmp_path, make_figures=False)
    assert result.state.step == round(config.simulation.duration / config.neural.dt)
    assert (result.run_dir / "config_resolved.yaml").exists()
    assert (result.run_dir / "training_history.npz").exists()
    assert "weights" in result.history
    assert result.history["time"][0] == 0.0
    assert result.history["step"][0] == 0
    assert np.all(result.history["delta_weights"][0] == 0.0)
    assert np.all(np.isfinite(result.state.weights))


def test_theory_oracle_keeps_weights_fixed(tmp_path):
    config = _smoke_config(tmp_path)
    result = train_feedforward(config, output_cwd=tmp_path, weight_mode="theory", learning_enabled=False, make_figures=False)
    with np.load(result.run_dir / "final_weights.npz") as values:
        assert np.array_equal(values["initial_weights"], values["weights"])


def test_zero_velocity_probe_starts_away_from_reset_boundary(tmp_path):
    config = _smoke_config(tmp_path)
    x_input = uniform_positions(config.geometry.n_input, config.geometry.length)
    x_comp = uniform_positions(config.geometry.n_competitive, config.geometry.length)
    weights = theoretical_weights(
        x_input,
        x_comp,
        beta=config.feedforward_learning.beta,
        sigma_r=config.tutor.sigma,
        integrated_drive=config.tutor.integrated_drive,
        alpha=config.feedforward_learning.alpha,
        length=config.geometry.length,
        periodic=False,
    )
    metrics, arrays = run_single_probe(
        config,
        weights,
        adaptation_strength=0.0,
        speed=0.0,
        duration=0.2,
        transient_duration=0.0,
        sample_interval_steps=2,
    )
    assert np.allclose(arrays["tutor_position"], config.geometry.length / 2.0)
    assert metrics["valid_samples"] > 0
