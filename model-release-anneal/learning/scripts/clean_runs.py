"""Delete generated run directories after explicit confirmation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/vafidis_toy")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    runs_root = Path(args.runs_root).resolve()
    if not runs_root.exists():
        print(f"No runs root at {runs_root}")
        return
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    if not run_dirs:
        print(f"No run directories under {runs_root}")
        return
    if not args.yes:
        print(f"Would delete {len(run_dirs)} run directories under {runs_root}. Re-run with --yes.")
        return
    for run_dir in run_dirs:
        shutil.rmtree(run_dir)
    print(f"Deleted {len(run_dirs)} run directories under {runs_root}")


if __name__ == "__main__":
    main()

