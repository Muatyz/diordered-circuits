"""Explicit random-number construction."""

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a local NumPy generator without mutating global RNG state."""

    return np.random.default_rng(seed)

