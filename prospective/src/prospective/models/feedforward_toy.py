"""Feedforward Emina-Kropff toy model with explicit state transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from prospective.common.geometry import uniform_positions
from prospective.config.schema import ExperimentConfig
from prospective.dynamics.competitive import competitive_euler_step
from prospective.plasticity.feedforward import feedforward_update
from prospective.stimuli.moving_tutor import moving_tutor_activity
from prospective.theory.equilibrium import theoretical_weights


@dataclass
class FeedforwardState:
    """Complete mutable state of the feedforward competitive network."""

    time: float
    step: int
    input_positions: NDArray[np.float64]
    competitive_positions: NDArray[np.float64]
    tutor_position: float
    tutor_rate: NDArray[np.float64]
    membrane: NDArray[np.float64]
    adaptation: NDArray[np.float64]
    rate: NDArray[np.float64]
    weights: NDArray[np.float64]
    last_delta_weights: NDArray[np.float64]
    last_clipped: NDArray[np.bool_]

    def copy(self) -> "FeedforwardState":
        """Return a deep numerical copy suitable for independent probes."""

        return FeedforwardState(
            time=self.time,
            step=self.step,
            input_positions=self.input_positions.copy(),
            competitive_positions=self.competitive_positions.copy(),
            tutor_position=self.tutor_position,
            tutor_rate=self.tutor_rate.copy(),
            membrane=self.membrane.copy(),
            adaptation=self.adaptation.copy(),
            rate=self.rate.copy(),
            weights=self.weights.copy(),
            last_delta_weights=self.last_delta_weights.copy(),
            last_clipped=self.last_clipped.copy(),
        )


def initialize_state(
    config: ExperimentConfig,
    rng: np.random.Generator,
    *,
    weight_mode: str = "random",
    fixed_weights: NDArray[np.float64] | None = None,
) -> FeedforwardState:
    """Initialize random, analytic-theory, or explicitly supplied weights.

    `theory` is the M1 oracle. The M2 learning experiment must use `random`.
    Competitive coordinates are labels before learning; their learned spatial
    preference is inferred later from each weight row rather than imposed.
    """

    g = config.geometry
    x_input = uniform_positions(g.n_input, g.length)
    x_competitive = uniform_positions(g.n_competitive, g.length)
    if fixed_weights is not None:
        weights = np.asarray(fixed_weights, dtype=float).copy()
    elif weight_mode == "random":
        weights = rng.uniform(
            0.0,
            config.feedforward_learning.initial_weight_scale,
            size=(g.n_competitive, g.n_input),
        )
    elif weight_mode == "theory":
        weights = theoretical_weights(
            x_input,
            x_competitive,
            beta=config.feedforward_learning.beta,
            sigma_r=config.tutor.sigma,
            integrated_drive=config.tutor.integrated_drive,
            alpha=config.feedforward_learning.alpha,
            length=g.length,
            periodic=g.boundary_mode == "periodic_ring",
        )
    else:
        raise ValueError("weight_mode must be random or theory")
    expected_shape = (g.n_competitive, g.n_input)
    if weights.shape != expected_shape:
        raise ValueError(f"fixed_weights must have shape {expected_shape}")
    center, tutor_rate = moving_tutor_activity(x_input, 0.0, config)
    zeros = np.zeros(g.n_competitive, dtype=float)
    return FeedforwardState(
        time=0.0,
        step=0,
        input_positions=x_input,
        competitive_positions=x_competitive,
        tutor_position=center,
        tutor_rate=tutor_rate,
        membrane=zeros.copy(),
        adaptation=zeros.copy(),
        rate=zeros.copy(),
        weights=weights,
        last_delta_weights=np.zeros_like(weights),
        last_clipped=np.zeros_like(weights, dtype=bool),
    )


def step_feedforward(
    state: FeedforwardState,
    config: ExperimentConfig,
    *,
    learning_enabled: bool | None = None,
) -> FeedforwardState:
    """Advance tutor, neural state, rate, then local weights by one Euler step.

    This fixed ordering is recorded in README and tested. The current is
    `J @ R`; no recurrent or hidden teacher correction is present in M0--M3.
    """

    dt = config.neural.dt
    next_time = state.time + dt
    center, tutor_rate = moving_tutor_activity(state.input_positions, next_time, config)
    input_current = state.weights @ tutor_rate
    membrane, adaptation, rate = competitive_euler_step(
        state.membrane,
        state.adaptation,
        input_current,
        dt=dt,
        tau_u=config.neural.tau_u,
        tau_v=config.neural.tau_v,
        adaptation_strength=config.neural.adaptation_strength,
        inhibition_strength=config.neural.inhibition_strength,
    )
    should_learn = config.feedforward_learning.enabled if learning_enabled is None else learning_enabled
    if should_learn:
        weights, delta, clipped = feedforward_update(
            state.weights,
            tutor_rate,
            rate,
            dt=dt,
            eta=config.feedforward_learning.eta,
            alpha=config.feedforward_learning.alpha,
            beta=config.feedforward_learning.beta,
            nonnegative_clip=config.feedforward_learning.nonnegative_clip,
        )
    else:
        weights = state.weights
        delta = np.zeros_like(state.weights)
        clipped = np.zeros_like(state.weights, dtype=bool)
    return FeedforwardState(
        time=next_time,
        step=state.step + 1,
        input_positions=state.input_positions,
        competitive_positions=state.competitive_positions,
        tutor_position=center,
        tutor_rate=tutor_rate,
        membrane=membrane,
        adaptation=adaptation,
        rate=rate,
        weights=weights,
        last_delta_weights=delta,
        last_clipped=clipped,
    )


def validate_state(state: FeedforwardState, config: ExperimentConfig) -> None:
    """Raise if shapes, finiteness, nonnegativity, or safety bounds fail."""

    g = config.geometry
    if state.weights.shape != (g.n_competitive, g.n_input):
        raise ValueError("invalid feedforward weight shape")
    for name, value in {
        "tutor_rate": state.tutor_rate,
        "membrane": state.membrane,
        "adaptation": state.adaptation,
        "rate": state.rate,
        "weights": state.weights,
    }.items():
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"non-finite {name} at step {state.step}")
        if np.max(np.abs(value), initial=0.0) > config.simulation.divergence_threshold:
            raise FloatingPointError(f"{name} exceeded divergence threshold at step {state.step}")
    if np.any(state.rate < 0):
        raise ValueError("firing rates must remain nonnegative")
    if config.feedforward_learning.nonnegative_clip and np.any(state.weights < 0):
        raise ValueError("weights violate the enabled nonnegative constraint")

