"""Small serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    with temporary_path.open("w", encoding="utf-8") as file_handle:
        json.dump(value, file_handle, indent=2, sort_keys=True)
    temporary_path.replace(path)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        loaded_value = json.load(file_handle)
    if not isinstance(loaded_value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return loaded_value


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    with temporary_path.open("wb") as file_handle:
        np.savez_compressed(file_handle, **arrays)
    # os.replace semantics preserve the previous complete archive until the
    # replacement archive has been closed successfully.  This matters for the
    # rolling training checkpoint, which is intentionally overwritten.
    temporary_path.replace(path)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded_npz:
        return {array_name: loaded_npz[array_name] for array_name in loaded_npz.files}
