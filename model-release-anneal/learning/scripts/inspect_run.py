"""Print a compact summary of a saved Vafidis toy-model run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory to inspect.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open("r", encoding="utf-8") as file_handle:
        metrics = json.load(file_handle)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

