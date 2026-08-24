from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from learning.config.schema import ExperimentConfig
from learning.dynamics.activation import apply_activation
from learning.dynamics.hd_dynamics import (
    PROXIMAL_INTEGRATION_EXACT_LINEAR,
    PROXIMAL_INTEGRATION_FORWARD_EULER,
    compute_hd_distal_pathway_drives,
    euler_update_i_hd_distal,
    euler_update_v_hd_distal,
    euler_update_v_hd_proximal,
    exact_linear_update_v_hd_proximal,
    update_v_hd_proximal,
)
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


def test_sigmoid_returns_release_firing_rate_in_khz() -> None:
    rate = apply_activation(
        np.array([1.0, 100.0]),
        activation_name="sigmoid",
        gain=2.5,
        bias=1.0,
        max_rate=0.15,
    )

    np.testing.assert_allclose(rate, np.array([0.075, 0.15]), atol=1e-12)


def test_sigmoid_rejects_nonpositive_firing_rate_ceiling() -> None:
    with pytest.raises(ValueError, match="max_rate must be positive"):
        apply_activation(
            np.array([0.0]),
            activation_name="sigmoid",
            max_rate=0.0,
        )


def test_release_rate_migration_preserves_currents_but_restores_physical_error() -> None:
    """The migrated state stores kHz rates/errors and release-scale weights."""
    f_max_khz = 0.15
    normalized_config = ExperimentConfig()
    normalized_config.model.n_theta = 8
    normalized_config.model.n_hr = 8
    normalized_config.model.init.w_hd_to_hd_mode = "random_normal"
    normalized_config.model.init.w_hr_to_hd_mode = "random_normal"
    normalized_config.model.init.w_hd_to_hd_scale = 0.02
    normalized_config.model.init.w_hr_to_hd_scale = 0.02
    normalized_config.model.init.random_jitter = 0.0
    normalized_config.model.activation.max_rate = 1.0
    normalized_config.model.w_hd_to_hr_strength = 2.0
    normalized_config.learning_rule.eta_hd_to_hd = 0.16875
    normalized_config.learning_rule.eta_hr_to_hd = 0.16875
    normalized_config.learning_rule.w_hd_to_hd_min = None
    normalized_config.learning_rule.w_hd_to_hd_max = None
    normalized_config.learning_rule.w_hr_to_hd_min = None
    normalized_config.learning_rule.w_hr_to_hd_max = None

    release_config = deepcopy(normalized_config)
    release_config.model.activation.max_rate = f_max_khz
    release_config.model.w_hd_to_hr_strength /= f_max_khz
    release_config.model.init.w_hd_to_hd_scale /= f_max_khz
    release_config.model.init.w_hr_to_hd_scale /= f_max_khz
    release_config.learning_rule.eta_hd_to_hd /= f_max_khz**3
    release_config.learning_rule.eta_hr_to_hd /= f_max_khz**3

    normalized_state = initialize_vafidis_toy_state(
        config=normalized_config,
        rng=make_rng(19),
    )
    release_state = initialize_vafidis_toy_state(
        config=release_config,
        rng=make_rng(19),
    )
    normalized_next = step_vafidis_toy(
        state=normalized_state,
        params=VafidisToyParams.from_config(normalized_config),
        angular_velocity=0.7,
        visual_teacher=True,
        training=True,
    )
    release_next = step_vafidis_toy(
        state=release_state,
        params=VafidisToyParams.from_config(release_config),
        angular_velocity=0.7,
        visual_teacher=True,
        training=True,
    )

    for current_name in [
        "i_hr",
        "i_hd_distal",
        "v_hd_distal",
        "v_hd_ss",
        "v_hd_proximal",
    ]:
        np.testing.assert_allclose(
            getattr(release_next, current_name),
            getattr(normalized_next, current_name),
            atol=1e-12,
        )
    for rate_name in ["r_hd", "r_hr", "p_hd", "p_hr", "e_hd"]:
        np.testing.assert_allclose(
            getattr(release_next, rate_name),
            f_max_khz * getattr(normalized_next, rate_name),
            atol=1e-12,
        )
    for weight_name in ["w_hd_to_hd", "w_hr_to_hd", "w_hd_to_hr"]:
        np.testing.assert_allclose(
            getattr(release_next, weight_name),
            getattr(normalized_next, weight_name) / f_max_khz,
            atol=1e-12,
        )


def test_hd_proximal_voltage_follows_vafidis_equation_4() -> None:
    next_v_hd_proximal = euler_update_v_hd_proximal(
        v_hd_proximal=np.array([1.0, -1.0]),
        v_hd_distal=np.array([2.0, 0.5]),
        i_vis_to_hd=np.array([0.5, -0.5]),
        dt=0.0005,
        c_hd_proximal=0.001,
        g_l_hd_proximal=1.0,
        g_d_hd_to_proximal=2.0,
    )
    # Va + dt/C * [-gL*Va - gD*(Va - Vd) + Iprox]
    np.testing.assert_allclose(next_v_hd_proximal, np.array([1.75, 0.75]))


def test_exact_linear_proximal_voltage_matches_analytic_solution() -> None:
    v_hd_proximal = np.array([1.0, -1.0])
    v_hd_distal = np.array([2.0, 0.5])
    i_vis_to_hd = np.array([0.5, -0.5])
    dt = 0.0005
    c_hd_proximal = 0.001
    g_l_hd_proximal = 1.0
    g_d_hd_to_proximal = 2.0

    actual = exact_linear_update_v_hd_proximal(
        v_hd_proximal=v_hd_proximal,
        v_hd_distal=v_hd_distal,
        i_vis_to_hd=i_vis_to_hd,
        dt=dt,
        c_hd_proximal=c_hd_proximal,
        g_l_hd_proximal=g_l_hd_proximal,
        g_d_hd_to_proximal=g_d_hd_to_proximal,
    )

    total_conductance = g_l_hd_proximal + g_d_hd_to_proximal
    steady_state = (
        g_d_hd_to_proximal * v_hd_distal + i_vis_to_hd
    ) / total_conductance
    retention = np.exp(-dt * total_conductance / c_hd_proximal)
    expected = steady_state + retention * (v_hd_proximal - steady_state)
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_proximal_update_dispatches_both_configured_methods() -> None:
    arguments = {
        "v_hd_proximal": np.array([0.4]),
        "v_hd_distal": np.array([1.2]),
        "i_vis_to_hd": np.array([-0.3]),
        "dt": 0.0005,
        "c_hd_proximal": 0.001,
        "g_l_hd_proximal": 1.0,
        "g_d_hd_to_proximal": 2.0,
    }

    euler_value = update_v_hd_proximal(
        **arguments,
        integration_method=PROXIMAL_INTEGRATION_FORWARD_EULER,
    )
    exact_value = update_v_hd_proximal(
        **arguments,
        integration_method=PROXIMAL_INTEGRATION_EXACT_LINEAR,
    )

    np.testing.assert_allclose(
        euler_value,
        euler_update_v_hd_proximal(**arguments),
    )
    np.testing.assert_allclose(
        exact_value,
        exact_linear_update_v_hd_proximal(**arguments),
    )
    assert not np.allclose(euler_value, exact_value)


def test_vafidis_equation_4_rejects_unstable_euler_step() -> None:
    config = ExperimentConfig()
    config.simulation.dt = 0.001

    with pytest.raises(ValueError, match="unstable.*Eq. 4"):
        VafidisToyParams.from_config(config)


def test_exact_linear_proximal_step_accepts_dt_outside_euler_stability_interval() -> None:
    config = ExperimentConfig()
    config.simulation.dt = 0.001
    config.simulation.proximal_integration_method = PROXIMAL_INTEGRATION_EXACT_LINEAR

    params = VafidisToyParams.from_config(config)

    assert params.proximal_integration_method == PROXIMAL_INTEGRATION_EXACT_LINEAR


def test_vafidis_rejects_unknown_proximal_integration_method() -> None:
    config = ExperimentConfig()
    config.simulation.proximal_integration_method = "mystery_solver"

    with pytest.raises(ValueError, match="proximal_integration_method"):
        VafidisToyParams.from_config(config)


def test_vafidis_step_uses_configured_proximal_integration_method() -> None:
    euler_config = ExperimentConfig()
    euler_config.model.n_theta = 8
    euler_config.model.n_hr = 8
    exact_config = deepcopy(euler_config)
    exact_config.simulation.proximal_integration_method = (
        PROXIMAL_INTEGRATION_EXACT_LINEAR
    )
    state = initialize_vafidis_toy_state(
        config=euler_config,
        rng=make_rng(23),
    )

    euler_next = step_vafidis_toy(
        state=state.copy(),
        params=VafidisToyParams.from_config(euler_config),
        angular_velocity=0.7,
        visual_teacher=True,
        training=False,
    )
    exact_next = step_vafidis_toy(
        state=state.copy(),
        params=VafidisToyParams.from_config(exact_config),
        angular_velocity=0.7,
        visual_teacher=True,
        training=False,
    )

    expected_exact = exact_linear_update_v_hd_proximal(
        v_hd_proximal=state.v_hd_proximal,
        v_hd_distal=exact_next.v_hd_distal,
        i_vis_to_hd=exact_next.i_vis_to_hd,
        dt=exact_config.simulation.dt,
        c_hd_proximal=exact_config.model.c_hd_proximal,
        g_l_hd_proximal=exact_config.model.g_l_hd_proximal,
        g_d_hd_to_proximal=exact_config.model.g_d_hd_to_proximal,
    )
    np.testing.assert_allclose(exact_next.v_hd_proximal, expected_exact)
    np.testing.assert_allclose(exact_next.v_hd_distal, euler_next.v_hd_distal)
    assert not np.allclose(exact_next.v_hd_proximal, euler_next.v_hd_proximal)


def test_raw_sum_hd_distal_update_remains_the_default() -> None:
    next_current = euler_update_i_hd_distal(
        i_hd_distal=np.zeros(2),
        w_hd_to_hd=np.ones((2, 2)),
        r_hd=np.array([1.0, 2.0]),
        w_hr_to_hd=np.ones((2, 2)),
        r_hr=np.array([3.0, 4.0]),
        b_hd=1.0,
        dt=0.1,
        tau_s=0.2,
    )
    np.testing.assert_allclose(next_current, 0.5 * (3.0 + 7.0 - 1.0))


def test_population_mean_normalizes_hd_and_each_hr_wing_separately() -> None:
    i_hd_from_hd, i_hd_from_lhr, i_hd_from_rhr = compute_hd_distal_pathway_drives(
        w_hd_to_hd=np.ones((4, 4)),
        r_hd=np.array([1.0, 2.0, 3.0, 4.0]),
        w_hr_to_hd=np.ones((4, 4)),
        r_hr=np.array([2.0, 4.0, 6.0, 8.0]),
        normalization="presynaptic_population_mean",
    )
    np.testing.assert_allclose(i_hd_from_hd, 2.5)
    np.testing.assert_allclose(i_hd_from_lhr, 3.0)
    np.testing.assert_allclose(i_hd_from_rhr, 7.0)


def test_population_mean_is_invariant_to_replicating_all_population_samples() -> None:
    replication = 3
    base_w_hd = np.array([[1.0, 2.0], [3.0, 4.0]])
    base_w_lhr = np.array([[0.5], [1.5]])
    base_w_rhr = np.array([[-1.0], [2.0]])
    base_w_hr = np.concatenate([base_w_lhr, base_w_rhr], axis=1)
    base_r_hd = np.array([0.25, 0.75])
    base_r_hr = np.array([0.4, 0.8])
    base_drives = compute_hd_distal_pathway_drives(
        w_hd_to_hd=base_w_hd,
        r_hd=base_r_hd,
        w_hr_to_hd=base_w_hr,
        r_hr=base_r_hr,
        normalization="presynaptic_population_mean",
    )

    replicated_w_hd = np.tile(base_w_hd, (replication, replication))
    replicated_w_hr = np.concatenate(
        [
            np.tile(base_w_lhr, (replication, replication)),
            np.tile(base_w_rhr, (replication, replication)),
        ],
        axis=1,
    )
    replicated_drives = compute_hd_distal_pathway_drives(
        w_hd_to_hd=replicated_w_hd,
        r_hd=np.tile(base_r_hd, replication),
        w_hr_to_hd=replicated_w_hr,
        r_hr=np.concatenate(
            [
                np.tile(base_r_hr[:1], replication),
                np.tile(base_r_hr[1:], replication),
            ]
        ),
        normalization="presynaptic_population_mean",
    )
    for base_drive, replicated_drive in zip(base_drives, replicated_drives):
        np.testing.assert_allclose(replicated_drive, np.tile(base_drive, replication))


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


def test_step_accepts_explicit_noise_at_all_synaptic_inputs() -> None:
    config = ExperimentConfig()
    config.model.n_theta = 4
    config.model.n_hr = 4
    params = VafidisToyParams.from_config(config)
    state = initialize_vafidis_toy_state(config=config, rng=make_rng(config.simulation.seed))

    quiet_state = step_vafidis_toy(
        state=state,
        params=params,
        angular_velocity=0.0,
        visual_teacher=False,
        training=False,
    )
    noisy_state = step_vafidis_toy(
        state=state,
        params=params,
        angular_velocity=0.0,
        visual_teacher=False,
        training=False,
        i_hd_distal_noise=np.full(params.n_theta, 0.2),
        i_hd_proximal_noise=np.full(params.n_theta, 0.3),
        i_hr_noise=np.full(params.n_hr, 0.4),
    )

    assert np.allclose(
        noisy_state.i_hd_distal - quiet_state.i_hd_distal,
        params.dt / params.tau_s * 0.2,
    )
    assert np.allclose(noisy_state.i_hr - quiet_state.i_hr, 0.4)
    expected_distal_voltage_noise = (
        params.dt / params.tau_l_hd * params.dt / params.tau_s * 0.2
    )
    expected_proximal_voltage_noise = params.dt / params.c_hd_proximal * (
        0.3 + params.g_d_hd_to_proximal * expected_distal_voltage_noise
    )
    assert np.allclose(
        noisy_state.v_hd_proximal - quiet_state.v_hd_proximal,
        expected_proximal_voltage_noise,
    )
    assert not np.allclose(noisy_state.r_hd, quiet_state.r_hd)
