"""Angular velocity inputs for HR cells."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
        self.angular_velocity = _step_ou_angular_velocity(
            angular_velocity=self.angular_velocity,
            mean=self.mean,
            std=self.std,
            tau=self.tau,
            dt=dt,
            rng=self.rng,
        )
        if self.clip is not None:
            self.angular_velocity = float(np.clip(self.angular_velocity, -self.clip, self.clip))
        return float(self.angular_velocity)


def _step_ou_angular_velocity(
    *,
    angular_velocity: float,
    mean: float,
    std: float,
    tau: float,
    dt: float,
    rng: np.random.Generator,
) -> float:
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("OU velocity tau must be finite and positive")
    if not np.isfinite(std) or std < 0.0:
        raise ValueError("OU velocity std must be finite and non-negative")
    noise_scale = std * np.sqrt(max(2.0 * dt / tau, 0.0))
    drift = (mean - angular_velocity) * dt / tau
    return float(angular_velocity + drift + noise_scale * rng.normal())


@dataclass
class ScheduledOUAngularVelocity:
    """OU velocity process with a piecewise-constant training-noise scale."""

    mean: float
    tau: float
    clip: float | None
    rng: np.random.Generator
    total_duration: float
    std_schedule: list[dict[str, Any]]
    angular_velocity: float = 0.0
    elapsed_time: float = 0.0
    _schedule: tuple[tuple[float, float], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.total_duration) or self.total_duration <= 0.0:
            raise ValueError("scheduled OU total_duration must be finite and positive")
        if not isinstance(self.std_schedule, list) or not self.std_schedule:
            raise ValueError("velocity.training_ou_std_schedule must be a non-empty list")
        parsed: list[tuple[float, float]] = []
        previous_end = 0.0
        for index, entry in enumerate(self.std_schedule):
            if not isinstance(entry, dict) or set(entry) != {"end_fraction", "std"}:
                raise ValueError(
                    "each velocity.training_ou_std_schedule entry must contain "
                    "only end_fraction and std"
                )
            end_fraction = float(entry["end_fraction"])
            std = float(entry["std"])
            if (
                not np.isfinite(end_fraction)
                or end_fraction <= previous_end
                or end_fraction > 1.0
            ):
                raise ValueError(
                    "velocity.training_ou_std_schedule end_fraction values must "
                    "be strictly increasing in (0, 1]"
                )
            if not np.isfinite(std) or std < 0.0:
                raise ValueError(
                    "velocity.training_ou_std_schedule std values must be "
                    "finite and non-negative"
                )
            parsed.append((end_fraction, std))
            previous_end = end_fraction
        if not np.isclose(parsed[-1][0], 1.0, atol=1e-12, rtol=0.0):
            raise ValueError(
                "velocity.training_ou_std_schedule must end at fraction 1.0"
            )
        self._schedule = tuple(parsed)

    def current_std(self, dt: float = 0.0) -> float:
        midpoint_time = min(
            self.elapsed_time + 0.5 * max(float(dt), 0.0),
            self.total_duration,
        )
        fraction = midpoint_time / self.total_duration
        for end_fraction, std in self._schedule:
            if fraction <= end_fraction:
                return std
        return self._schedule[-1][1]

    def step(self, dt: float) -> float:
        dt = float(dt)
        self.angular_velocity = _step_ou_angular_velocity(
            angular_velocity=self.angular_velocity,
            mean=self.mean,
            std=self.current_std(dt),
            tau=self.tau,
            dt=dt,
            rng=self.rng,
        )
        if self.clip is not None:
            self.angular_velocity = float(
                np.clip(self.angular_velocity, -self.clip, self.clip)
            )
        self.elapsed_time += dt
        return float(self.angular_velocity)
