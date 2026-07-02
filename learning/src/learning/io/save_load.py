"""Small serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(value, file_handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        loaded_value = json.load(file_handle)
    if not isinstance(loaded_value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return loaded_value


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded_npz:
        return {array_name: loaded_npz[array_name] for array_name in loaded_npz.files}

