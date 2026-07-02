"""Run directory helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_id(*, experiment_name: str, seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}_{experiment_name}_{seed}"


def create_run_dir(
    *,
    project_root: Path,
    runs_root: str,
    experiment_name: str,
    seed: int,
    run_id: str | None = None,
) -> Path:
    selected_run_id = run_id or make_run_id(experiment_name=experiment_name, seed=seed)
    run_dir = project_root / runs_root / selected_run_id
    if run_dir.exists():
        counter = 1
        while True:
            candidate_run_dir = project_root / runs_root / f"{selected_run_id}_{counter:02d}"
            if not candidate_run_dir.exists():
                run_dir = candidate_run_dir
                break
            counter += 1
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir

