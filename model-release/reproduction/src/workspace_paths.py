"""Stable repository paths for the nested development workspace."""

from pathlib import Path


REPRODUCTION_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPRODUCTION_ROOT.parent
REPOSITORY_ROOT = MODEL_ROOT.parent
DATA_ROOT = REPOSITORY_ROOT / "data"
REFERENCES_ROOT = REPOSITORY_ROOT / "references"

