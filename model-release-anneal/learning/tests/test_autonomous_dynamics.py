import numpy as np
import pytest

from learning.common.random import make_rng
from learning.config.schema import ExperimentConfig
from learning.dynamics.autonomous import FrozenAutonomousDynamics
from learning.dynamics.hd_dynamics import (
    PROXIMAL_INTEGRATION_EXACT_LINEAR,
    PROXIMAL_INTEGRATION_FORWARD_EULER,
)
from learning.models.vafidis_toy import (
    VafidisToyParams,
    initialize_vafidis_toy_state,
    step_vafidis_toy,
)


def make_small_system(
    proximal_integration_method: str = PROXIMAL_INTEGRATION_FORWARD_EULER,
):
    config = ExperimentConfig()
    config.model.n_theta = 8
    config.model.n_hr = 8
    config.simulation.dt = 0.0005
    config.simulation.proximal_integration_method = proximal_integration_method
    state = initialize_vafidis_toy_state(config=config, rng=make_rng(7))
    params = VafidisToyParams.from_config(config)
    dynamics = FrozenAutonomousDynamics.from_state(params=params, state=state)
    return state, params, dynamics


@pytest.mark.parametrize(
    "proximal_integration_method",
    [PROXIMAL_INTEGRATION_FORWARD_EULER, PROXIMAL_INTEGRATION_EXACT_LINEAR],
)
def test_autonomous_map_matches_frozen_model_step(
    proximal_integration_method: str,
) -> None:
    state, params, dynamics = make_small_system(proximal_integration_method)
    # The initialized state contains a visual cue. One darkness step removes
    # its current while preserving the dynamic Eq. (4) proximal state.
    state = step_vafidis_toy(
        state=state,
        params=params,
        angular_velocity=0.0,
        visual_teacher=False,
        training=False,
    )
    expected = step_vafidis_toy(
        state=state,
        params=params,
        angular_velocity=0.0,
        visual_teacher=False,
        training=False,
    )
    actual_vector = dynamics.step(dynamics.pack_state(state))
    assert np.allclose(actual_vector, dynamics.pack_state(expected))


@pytest.mark.parametrize(
    "proximal_integration_method",
    [PROXIMAL_INTEGRATION_FORWARD_EULER, PROXIMAL_INTEGRATION_EXACT_LINEAR],
)
def test_autonomous_flow_jacobian_matches_central_difference(
    proximal_integration_method: str,
) -> None:
    state, _params, dynamics = make_small_system(proximal_integration_method)
    state_vector = dynamics.pack_state(state)
    analytic = dynamics.flow_jacobian(state_vector)
    numerical = np.empty_like(analytic)
    step_size = 1e-6
    for column in range(state_vector.size):
        perturbation = np.zeros_like(state_vector)
        perturbation[column] = step_size
        numerical[:, column] = (
            dynamics.flow(state_vector + perturbation)
            - dynamics.flow(state_vector - perturbation)
        ) / (2.0 * step_size)
    assert np.allclose(analytic, numerical, atol=2e-6, rtol=2e-5)


def test_firing_rate_state_concatenates_hd_and_hr_observables() -> None:
    state, _params, dynamics = make_small_system()
    state_vector = dynamics.pack_state(state)
    firing_rate_state = dynamics.firing_rate_state(state_vector)
    blocks = dynamics.unpack_state(state_vector)
    assert firing_rate_state.shape == (16,)
    np.testing.assert_allclose(firing_rate_state[:8], dynamics.hd_rate(state_vector))
    np.testing.assert_allclose(firing_rate_state[8:], blocks["r_hr"])


def test_pva_phase_reads_dynamic_proximal_voltage_not_other_state_blocks() -> None:
    state, _params, dynamics = make_small_system()
    state_vector = dynamics.pack_state(state)
    distal_changed = state_vector.copy()
    distal_slice = dynamics.component_slices["v_hd_distal"]
    distal_changed[distal_slice] += np.linspace(-20.0, 20.0, 8)
    assert np.isclose(
        np.exp(1j * dynamics.decoded_heading(distal_changed)),
        np.exp(1j * dynamics.decoded_heading(state_vector)),
    )

    proximal_slice = dynamics.component_slices["v_hd_proximal"]
    first_heading_state = state_vector.copy()
    first_heading_state[proximal_slice] = -20.0
    first_heading_state[proximal_slice.start : proximal_slice.start + 2] = 20.0
    opposite_heading_state = state_vector.copy()
    opposite_heading_state[proximal_slice] = -20.0
    opposite_heading_state[proximal_slice.start + 4 : proximal_slice.start + 6] = 20.0
    assert np.isclose(
        np.exp(1j * dynamics.decoded_heading(first_heading_state)),
        np.exp(-1j * np.pi),
    )
    assert np.isclose(
        np.exp(1j * dynamics.decoded_heading(opposite_heading_state)),
        1.0 + 0.0j,
    )
