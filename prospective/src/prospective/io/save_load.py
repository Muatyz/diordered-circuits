"""Array persistence for training and frozen-weight probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_npz(path: str | Path, **arrays: Any) -> None:
    """Save named numerical arrays using compressed NPZ storage."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ into ordinary arrays and close the file immediately."""

    with np.load(Path(path), allow_pickle=False) as values:
        return {key: values[key] for key in values.files}

