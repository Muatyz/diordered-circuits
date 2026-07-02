from __future__ import annotations

import numpy as np

from learning.config.schema import ExperimentConfig
from learning.dynamics.activation import apply_activation
from learning.dynamics.hd_dynamics import euler_update_v_hd_distal
from learning.dynamics.hr_dynamics import compute_i_hr, euler_update_r_hd_to_hr_lp
from learning.models.vafidis_toy import VafidisToyParams, initialize_vafidis_toy_state, step_vafidis_toy
from learning.common.random import make_rng
from learning.connectivity.initialize import initialize_w_hd_to_hr
from learning.stimuli.velocity import make_i_vel_to_hr
from learning.plasticity.traces import euler_update_psp_trace


def test_hd_distal_voltage_is_independent_leaky_state() -> None:
    v_hd_distal = np.array([0.0, 2.0])
    i_hd_distal = np.array([1.0, 0.0])
    next_v_hd_distal = euler_update_v_hd_distal(
        v_hd_distal=v_hd_distal,
        i_hd_distal=i_hd_distal,
        dt=0.1,
        tau_l_hd=0.2,
    )
    assert np.allclose(next_v_hd_distal, np.array([0.5, 1.0]))


def test_hr_velocity_input_is_not_low_pass_filtered_with_hd_input() -> None:
    r_hd_to_hr_lp = np.array([0.0, 0.0])
    r_hd = np.array([1.0, 0.0])
    next_r_hd_to_hr_lp = euler_update_r_hd_to_hr_lp(
        r_hd_to_hr_lp=r_hd_to_hr_lp,
        r_hd=r_hd,
        dt=0.1,
        tau_s=0.2,
    )
    i_hr = compute_i_hr(
        w_hd_to_hr=np.eye(2),
        r_hd_to_hr_lp=next_r_hd_to_hr_lp,
        i_vel_to_hr=np.array([3.0, -3.0]),
        b_hr=0.5,
    )
    assert np.allclose(next_r_hd_to_hr_lp, np.array([0.5, 0.0]))
    assert np.allclose(i_hr, np.array([3.0, -3.5]))


def test_hr_velocity_input_sign_matches_release_code_wing_order() -> None:
    i_vel_to_hr = make_i_vel_to_hr(n_hr=6, angular_velocity=2.0, k_vel=0.5)

    assert np.allclose(i_vel_to_hr[:3], 1.0)
    assert np.allclose(i_vel_to_hr[3:], -1.0)


def test_additive_hr_velocity_input_preserves_hd_spatial_drive() -> None:
    config = ExperimentConfig()
    config.model.n_theta = 24
    config.model.n_hr = 24
    config.velocity.k_vel = 1.0
    params = VafidisToyParams.from_config(config)
    state = initialize_vafidis_toy_state(config=config, rng=make_rng(config.simulation.seed))
    state.r_hd_to_hr_lp = np.zeros(params.n_theta)
    state.r_hd_to_hr_lp[0:2] = 1.0
    i_vel_to_hr = make_i_vel_to_hr(
        n_hr=params.n_hr,
        angular_velocity=0.0,
        k_vel=params.k_vel,
    )
    i_hr = compute_i_hr(
        w_hd_to_hr=state.w_hd_to_hr,
        r_hd_to_hr_lp=state.r_hd_to_hr_lp,
        i_vel_to_hr=i_vel_to_hr,
        b_hr=params.b_hr,
    )
    r_hr = apply_activation(
        i_hr,
        activation_name=params.activation_name,
        gain=params.activation_gain,
        bias=params.activation_bias,
    )
    n_hr_per_wing = params.n_hr // 2
    r_lhr = r_hr[:n_hr_per_wing]
    r_rhr = r_hr[n_hr_per_wing:]
    assert np.max(r_lhr) - np.min(r_lhr) > 0.10
    assert np.max(r_rhr) - np.min(r_rhr) > 0.10
    assert int(np.argmax(r_lhr)) == 0
    assert int(np.argmax(r_rhr)) == 0


def test_mature_hd_bump_drives_sparse_conjunctive_hr_population() -> None:
    config = ExperimentConfig()
    config.model.n_theta = 48
    config.model.n_hr = 48
    config.model.init.random_jitter = 0.0
    state = initialize_vafidis_toy_state(config=config, rng=make_rng(config.simulation.seed))
    state.r_hd_to_hr_lp = np.zeros(config.model.n_theta)
    state.r_hd_to_hr_lp[0:2] = 1.0
    i_hr = compute_i_hr(
        w_hd_to_hr=state.w_hd_to_hr,
        r_hd_to_hr_lp=state.r_hd_to_hr_lp,
        i_vel_to_hr=np.zeros(config.model.n_hr),
        b_hr=config.model.b_hr,
    )
    r_hr = apply_activation(
        i_hr,
        activation_name=config.model.activation.name,
        gain=config.model.activation.gain,
        bias=config.model.activation.bias,
    )

    n_hr_per_wing = r_hr.size // 2
    r_lhr = r_hr[:n_hr_per_wing]
    r_rhr = r_hr[n_hr_per_wing:]
    for r_hr_wing in [r_lhr, r_rhr]:
        assert np.max(r_hr_wing) > 0.20
        assert np.min(r_hr_wing) < 0.01
        assert np.max(r_hr_wing) - np.min(r_hr_wing) > 0.20
        assert np.mean(r_hr_wing) < 0.02
    assert int(np.argmax(r_lhr)) == 0
    assert int(np.argmax(r_rhr)) == 0


def test_hd_to_hr_uses_vafidis_odd_even_wing_mapping() -> None:
    w_hd_to_hr = initialize_w_hd_to_hr(n_theta=6, n_hr=6, strength=2.0)

    expected_w_hd_to_hr = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        ]
    )
    assert np.allclose(w_hd_to_hr, expected_w_hd_to_hr)


def test_psp_trace_uses_synaptic_and_distal_filter_stages() -> None:
    p_synaptic, p_trace = euler_update_psp_trace(
        p_synaptic=np.array([0.0]),
        p_trace=np.array([0.0]),
        r_pre=np.array([1.0]),
        dt=0.1,
        tau_s=0.2,
        tau_l=0.5,
    )
    assert np.allclose(p_synaptic, np.array([0.5]))
    assert np.allclose(p_trace, np.array([0.1]))

    next_p_synaptic, next_p_trace = euler_update_psp_trace(
        p_synaptic=p_synaptic,
        p_trace=p_trace,
        r_pre=np.array([1.0]),
        dt=0.1,
        tau_s=0.2,
        tau_l=0.5,
    )
    assert np.allclose(next_p_synaptic, np.array([0.75]))
    assert np.allclose(next_p_trace, np.array([0.23]))


def test_step_uses_previous_presynaptic_rates_for_hd_current_and_psp() -> None:
    config = ExperimentConfig()
    config.model.n_theta = 4
    config.model.n_hr = 4
    config.model.init.w_hd_to_hd_mode = "zeros"
    config.model.init.w_hr_to_hd_mode = "zeros"
    config.model.init.random_jitter = 0.0
    config.model.b_hd = 0.0
    config.velocity.k_vel = 4.0
    params = VafidisToyParams.from_config(config)
    state = initialize_vafidis_toy_state(config=config, rng=make_rng(config.simulation.seed))
    state.r_hr = np.zeros(params.n_hr)
    state.p_hr_synaptic = np.zeros(params.n_hr)
    state.p_hr = np.zeros(params.n_hr)
    state.i_hd_distal = np.zeros(params.n_theta)
    state.w_hd_to_hd = np.zeros((params.n_theta, params.n_theta))
    state.w_hr_to_hd = np.ones((params.n_theta, params.n_hr))

    next_state = step_vafidis_toy(
        state=state,
        params=params,
        angular_velocity=1.0,
        visual_teacher=True,
        training=False,
    )

    assert np.max(next_state.r_hr) > 0.0
    assert np.allclose(next_state.i_hd_distal, 0.0)
    assert np.allclose(next_state.p_hr_synaptic, 0.0)
