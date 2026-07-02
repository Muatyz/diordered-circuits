"""YAML config loading and project-root resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from learning.config.schema import ExperimentConfig, experiment_config_from_dict


def find_project_root(start_path: str | Path | None = None) -> Path:
    """Find the learning subproject root by walking upward."""
    current_path = Path(start_path or Path.cwd()).resolve()
    if current_path.is_file():
        current_path = current_path.parent
    for candidate_path in [current_path, *current_path.parents]:
        if (candidate_path / "pyproject.toml").exists() and (candidate_path / "src" / "learning").exists():
            return candidate_path
        if (candidate_path / ".SKILL.md").exists() and (candidate_path / "notebooks").exists():
            return candidate_path
    raise FileNotFoundError(f"Could not locate learning project root from {current_path}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        loaded_value = yaml.safe_load(file_handle) or {}
    if not isinstance(loaded_value, dict):
        raise ValueError(f"Expected mapping at {path}")
    return loaded_value


def save_yaml(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(value, file_handle, sort_keys=False, allow_unicode=True)


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    raw_config = load_yaml(config_path)
    return experiment_config_from_dict(raw_config)

