import numpy as np
import pytest

from prospective.common.random import make_rng
from prospective.config.schema import ExperimentConfig
from prospective.models.feedforward_toy import initialize_state, step_feedforward


def test_state_and_weight_matrix_convention():
    config = ExperimentConfig()
    state = initialize_state(config, make_rng(3))
    assert state.tutor_rate.shape == (config.geometry.n_input,)
    assert state.membrane.shape == (config.geometry.n_competitive,)
    assert state.weights.shape == (config.geometry.n_competitive, config.geometry.n_input)
    next_state = step_feedforward(state, config)
    assert next_state.weights.shape == state.weights.shape


def test_fixed_weight_shape_is_checked():
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="fixed_weights"):
        initialize_state(config, make_rng(3), fixed_weights=np.zeros((2, 2)))

