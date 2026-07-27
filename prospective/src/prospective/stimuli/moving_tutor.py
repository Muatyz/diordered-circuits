"""Moving Gaussian tutor used during self-organization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from prospective.common.geometry import normalized_gaussian, tutor_position
from prospective.config.schema import ExperimentConfig


def moving_tutor_activity(
    positions: NDArray[np.float64], time: float, config: ExperimentConfig
) -> tuple[float, NDArray[np.float64]]:
    """Return current tutor center and discrete input-layer firing rates.

    `paper_reset` evaluates the real-line Gaussian on `[0, L)` and resets its
    center at the boundary. `periodic_ring` uses shortest periodic distance and
    is deliberately labeled as a numerical control rather than paper-faithful.
    """

    center = tutor_position(
        config.tutor.initial_position,
        config.tutor.speed,
        time,
        config.geometry.length,
    )
    profile = normalized_gaussian(
        positions,
        center,
        config.tutor.sigma,
        length=config.geometry.length,
        periodic=config.geometry.boundary_mode == "periodic_ring",
    )
    return center, config.tutor.integrated_drive * profile

