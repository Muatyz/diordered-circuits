import numpy as np

from prospective.dynamics.activation import divisive_quadratic_rate
from prospective.dynamics.competitive import competitive_euler_step


def test_divisive_quadratic_rate_matches_hand_calculation():
    membrane = np.array([-1.0, 1.0, 2.0])
    expected = np.array([0.0, 1.0, 4.0]) / (1.0 + 0.5 * 5.0)
    assert np.allclose(divisive_quadratic_rate(membrane, 0.5), expected)


def test_euler_step_uses_old_state_for_both_derivatives():
    u = np.array([1.0])
    v = np.array([0.5])
    current = np.array([2.0])
    next_u, next_v, rate = competitive_euler_step(
        u, v, current, dt=0.1, tau_u=1.0, tau_v=2.0,
        adaptation_strength=0.2, inhibition_strength=0.0,
    )
    assert np.allclose(next_u, [1.05])
    assert np.allclose(next_v, [0.485])
    assert np.allclose(rate, next_u**2)


def test_no_adaptation_drive_decays_existing_adaptation():
    _, next_v, _ = competitive_euler_step(
        np.array([2.0]), np.array([1.0]), np.array([0.0]),
        dt=0.1, tau_u=1.0, tau_v=1.0, adaptation_strength=0.0, inhibition_strength=0.0,
    )
    assert next_v[0] < 1.0

