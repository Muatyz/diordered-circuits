"""Euler integration of membrane potential and firing-rate adaptation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from prospective.dynamics.activation import divisive_quadratic_rate


def competitive_euler_step(
    membrane: NDArray[np.float64],
    adaptation: NDArray[np.float64],
    input_current: NDArray[np.float64],
    *,
    dt: float,
    tau_u: float,
    tau_v: float,
    adaptation_strength: float,
    inhibition_strength: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Advance Eqs. 2--3 once using simultaneous explicit Euler updates.

    Both derivatives use the old `membrane` and `adaptation` state. Firing
    rates are evaluated from the newly advanced membrane potential.
    """

    membrane = np.asarray(membrane, dtype=float)
    adaptation = np.asarray(adaptation, dtype=float)
    input_current = np.asarray(input_current, dtype=float)
    if membrane.shape != adaptation.shape or membrane.shape != input_current.shape:
        raise ValueError("membrane, adaptation, and input_current shapes must match")
    next_membrane = membrane + (dt / tau_u) * (-membrane - adaptation + input_current)
    next_adaptation = adaptation + (dt / tau_v) * (-adaptation + adaptation_strength * membrane)
    rate = divisive_quadratic_rate(next_membrane, inhibition_strength)
    return next_membrane, next_adaptation, rate

