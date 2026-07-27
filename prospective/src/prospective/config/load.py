"""YAML configuration I/O."""

from __future__ import annotations

from pathlib import Path

import yaml

from prospective.config.schema import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return ExperimentConfig.from_dict(values)


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    """Write every resolved configuration value to a YAML file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
