from __future__ import annotations

import numpy as np

from learning.common.random import make_rng
from learning.config.schema import ExperimentConfig
from learning.models.vafidis_toy import initialize_vafidis_toy_state


def test_state_initialization_is_seed_reproducible() -> None:
    config = ExperimentConfig()
    config.model.n_theta = 10
    config.model.n_hr = 10
    config.model.init.random_jitter = 0.01
    first_state = initialize_vafidis_toy_state(config=config, rng=make_rng(123))
    second_state = initialize_vafidis_toy_state(config=config, rng=make_rng(123))
    assert np.allclose(first_state.w_hd_to_hd, second_state.w_hd_to_hd)
    assert np.allclose(first_state.w_hr_to_hd, second_state.w_hr_to_hd)
    assert np.allclose(np.diag(first_state.w_hd_to_hd), 0.0)
