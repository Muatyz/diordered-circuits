from __future__ import annotations

import numpy as np

from learning.common.angles import make_theta_hd_pref
from learning.config.schema import ExperimentConfig
from learning.common.random import make_rng
from learning.models.vafidis_toy import initialize_vafidis_toy_state
from learning.stimuli.visual import make_i_vis_to_hd


def test_default_visual_teacher_has_inhibitory_surround() -> None:
    config = ExperimentConfig()
    theta_hd_pref = make_theta_hd_pref(48)
    i_vis_to_hd = make_i_vis_to_hd(
        theta_hd_pref=theta_hd_pref,
        theta_true=0.0,
        amplitude=config.visual.amplitude,
        kappa=config.visual.kappa,
        baseline=config.visual.baseline,
        normalize_peak=config.visual.normalize_peak,
    )
    assert np.max(i_vis_to_hd) > 0.0
    assert np.min(i_vis_to_hd) < 0.0


def test_paper_like_visual_teacher_is_scaled_at_proximal_compartment() -> None:
    config = ExperimentConfig()
    config.model.n_theta = 48
    config.model.n_hr = 48
    config.model.init.random_jitter = 0.0
    config.visual.amplitude = 4.0
    config.visual.baseline = 5.0
    config.visual.light_excitation = 4.0
    config.visual.proximal_scale = 1.0 / 3.0

    state = initialize_vafidis_toy_state(config=config, rng=make_rng(config.simulation.seed))

    assert np.isclose(np.max(state.v_hd_proximal), 1.0)
    assert np.min(state.v_hd_proximal) < 0.0
    assert np.min(state.v_hd_proximal) > -0.35
