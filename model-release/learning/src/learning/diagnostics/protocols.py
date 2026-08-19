"""Shared state construction for frozen-weight intervention protocols."""

from __future__ import annotations

from learning.common.random import make_rng
from learning.config.schema import ExperimentConfig
from learning.dynamics.hd_dynamics import compute_hd_distal_pathway_drives
from learning.models.vafidis_toy import (
    VafidisToyParams,
    VafidisToyState,
    initialize_vafidis_toy_state,
    validate_vafidis_toy_state,
)


def initialize_frozen_protocol_state(
    *,
    config: ExperimentConfig,
    trained_state: VafidisToyState,
    theta_true: float,
) -> VafidisToyState:
    """Create a reproducible fresh state carrying only frozen learned parameters."""

    params = VafidisToyParams.from_config(config)
    protocol_rng = make_rng(config.simulation.seed)
    state = initialize_vafidis_toy_state(
        config=config,
        rng=protocol_rng,
        theta_true=theta_true,
    )
    state.w_hd_to_hd = trained_state.w_hd_to_hd.copy()
    state.w_hr_to_hd = trained_state.w_hr_to_hd.copy()
    state.w_hd_to_hr = trained_state.w_hd_to_hr.copy()
    state.visual_tuning_profiles = (
        None
        if trained_state.visual_tuning_profiles is None
        else trained_state.visual_tuning_profiles.copy()
    )
    (
        state.i_hd_from_hd,
        state.i_hd_from_lhr,
        state.i_hd_from_rhr,
    ) = compute_hd_distal_pathway_drives(
        w_hd_to_hd=state.w_hd_to_hd,
        r_hd=state.r_hd,
        w_hr_to_hd=state.w_hr_to_hd,
        r_hr=state.r_hr,
        normalization=params.hd_distal_normalization,
    )
    validate_vafidis_toy_state(state, params)
    return state
