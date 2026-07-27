"""Create traceable, non-overwriting experiment run directories."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from prospective import __version__
from prospective.config.load import save_resolved_config
from prospective.config.schema import ExperimentConfig


def _git_value(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: str | Path, values: dict[str, Any]) -> None:
    """Write readable JSON, converting NumPy scalar values safely."""

    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"cannot serialize {type(value)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False, default=default), encoding="utf-8")


def create_run_dir(config: ExperimentConfig, *, base_cwd: str | Path | None = None) -> Path:
    """Create a timestamped run and record resolved config plus provenance."""

    cwd = Path(base_cwd or Path.cwd()).resolve()
    output_root = Path(config.experiment.output_root)
    if not output_root.is_absolute():
        output_root = cwd / output_root
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / config.experiment.name / f"{stamp}_seed{config.experiment.seed}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / config.experiment.name / f"{stamp}_seed{config.experiment.seed}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    save_resolved_config(config, run_dir / "config_resolved.yaml")
    repository = cwd
    while repository.parent != repository and not (repository / ".git").exists():
        repository = repository.parent
    dirty = _git_value(["status", "--porcelain"], repository)
    metadata = {
        "experiment": config.experiment.name,
        "seed": config.experiment.seed,
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "package_version": __version__,
        "git_commit": _git_value(["rev-parse", "HEAD"], repository),
        "dirty_worktree": bool(dirty) if dirty is not None else None,
        "matrix_convention": "J[post, pre]",
        "boundary_mode": config.geometry.boundary_mode,
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "status.json", {"status": "running"})
    return run_dir


def mark_run(run_dir: str | Path, status: str, **details: Any) -> None:
    """Mark a run completed or failed without hiding partial results."""

    if status not in {"running", "completed", "failed"}:
        raise ValueError("invalid run status")
    write_json(Path(run_dir) / "status.json", {"status": status, **details})

