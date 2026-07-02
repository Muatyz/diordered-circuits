from __future__ import annotations

import numpy as np

from learning.analysis.metrics import estimate_decoded_velocity, summarize_velocity_gain


def test_estimate_decoded_velocity_uses_simple_linear_fit() -> None:
    time = np.linspace(0.0, 1.0, 11)
    theta_decoded = 0.25 + 0.75 * time
    assert np.isclose(
        estimate_decoded_velocity(time=time, theta_decoded=theta_decoded, start_fraction=0.0),
        0.75,
    )


def test_summarize_velocity_gain_uses_simple_linear_fit() -> None:
    commanded_velocity = np.array([-1.0, -0.5, 0.5, 1.0])
    decoded_velocity = 0.2 + 0.8 * commanded_velocity
    summary = summarize_velocity_gain(
        commanded_velocity=commanded_velocity,
        decoded_velocity=decoded_velocity,
    )
    assert np.isclose(summary["gain"], 0.8)
    assert np.isclose(summary["intercept"], 0.2)
