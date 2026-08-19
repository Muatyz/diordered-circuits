"""Angular velocity inputs for HR cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_i_vel_to_hr(
    *,
    n_hr: int,
    angular_velocity: float,
    k_vel: float,
) -> np.ndarray:
    """Return release-code ordered left/right HR velocity currents."""
    if n_hr % 2 != 0:
        raise ValueError("n_hr must be even so left/right HR wings have equal size")
    n_hr_per_wing = n_hr // 2
    i_vel_to_lhr = k_vel * angular_velocity * np.ones(n_hr_per_wing, dtype=float)
    i_vel_to_rhr = -k_vel * angular_velocity * np.ones(n_hr_per_wing, dtype=float)
    return np.concatenate([i_vel_to_lhr, i_vel_to_rhr])


@dataclass
class OUAngularVelocity:
    """Euler-Maruyama OU process for angular velocity."""

    mean: float
    std: float
    tau: float
    clip: float | None
    rng: np.random.Generator
    angular_velocity: float = 0.0

    def step(self, dt: float) -> float:
        noise_scale = self.std * np.sqrt(max(2.0 * dt / self.tau, 0.0))
        drift = (self.mean - self.angular_velocity) * dt / self.tau
        self.angular_velocity = self.angular_velocity + drift + noise_scale * self.rng.normal()
        if self.clip is not None:
            self.angular_velocity = float(np.clip(self.angular_velocity, -self.clip, self.clip))
        return float(self.angular_velocity)

