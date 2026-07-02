"""Activation functions used by the toy model."""

from __future__ import annotations

import numpy as np


def activation_sigmoid(voltage: np.ndarray, gain: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """Stable logistic activation with nonnegative firing rates."""
    scaled_voltage = np.clip(gain * (voltage - bias), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scaled_voltage))


def apply_activation(
    voltage: np.ndarray,
    activation_name: str,
    gain: float = 1.0,
    bias: float = 0.0,
) -> np.ndarray:
    if activation_name == "sigmoid":
        return activation_sigmoid(voltage, gain=gain, bias=bias)
    raise ValueError(f"Unknown activation_name: {activation_name}")
