from dataclasses import replace

import pytest

from prospective.config.load import load_config
from prospective.config.schema import ExperimentConfig


def test_default_config_is_valid():
    ExperimentConfig().validate()


@pytest.mark.parametrize("beta", [0.0, 2.0, -1.0, 2.5])
def test_invalid_beta_is_rejected(beta):
    base = ExperimentConfig()
    config = replace(base, feedforward_learning=replace(base.feedforward_learning, beta=beta))
    with pytest.raises(ValueError, match="0 < beta < 2"):
        config.validate()


def test_unknown_boundary_mode_is_rejected():
    base = ExperimentConfig()
    config = replace(base, geometry=replace(base.geometry, boundary_mode="silent_guess"))
    with pytest.raises(ValueError, match="boundary_mode"):
        config.validate()


def test_shipped_toy_yaml_loads_with_numeric_threshold():
    config = load_config("configs/experiments/feedforward_toy.yaml")
    assert isinstance(config.simulation.divergence_threshold, float)
    assert config.simulation.divergence_threshold == 1_000_000.0


@pytest.mark.parametrize("name", ["fixed_theory_dynamics", "animation_demo", "paper_reference_feedforward"])
def test_other_shipped_experiment_yamls_load(name):
    load_config(f"configs/experiments/{name}.yaml")
