from __future__ import annotations

from learning.common.random import make_rng
from learning.config.schema import ExperimentConfig
from learning.models.vafidis_toy import VafidisToyParams, initialize_vafidis_toy_state, validate_vafidis_toy_state


def test_initialized_state_shapes_are_valid() -> None:
    config = ExperimentConfig()
    config.model.n_theta = 12
    config.model.n_hr = 12
    rng = make_rng(config.simulation.seed)
    state = initialize_vafidis_toy_state(config=config, rng=rng)
    validate_vafidis_toy_state(state, VafidisToyParams.from_config(config))
    assert state.r_hd.shape == (12,)
    assert state.r_hr.shape == (12,)
    assert state.r_hd_to_hr_lp.shape == (12,)
    assert state.p_hd_synaptic.shape == (12,)
    assert state.p_hd.shape == (12,)
    assert state.p_hr_synaptic.shape == (12,)
    assert state.p_hr.shape == (12,)
    assert state.w_hd_to_hd.shape == (12, 12)
    assert state.w_hr_to_hd.shape == (12, 12)
    assert state.w_hd_to_hr.shape == (12, 12)
    assert state.delta_w_hd_to_hd.shape == (12, 12)
    assert state.delta_w_hr_to_hd.shape == (12, 12)
