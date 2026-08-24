from pathlib import Path

from learning.config.load_config import find_project_root
from learning.experiments.run_vafidis_toy import resolve_config_path


LEARNING_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = LEARNING_ROOT.parents[1]


def test_project_root_falls_back_to_active_installed_tree(tmp_path):
    assert find_project_root(tmp_path) == LEARNING_ROOT


def test_shared_notebooks_are_not_treated_as_learning_project_root():
    shared_notebooks_root = REPOSITORY_ROOT / "notebooks"

    assert shared_notebooks_root.is_dir()
    assert find_project_root(shared_notebooks_root) == LEARNING_ROOT


def test_config_resolves_from_repository_root_without_crossing_release(monkeypatch):
    monkeypatch.chdir(REPOSITORY_ROOT)

    resolved = resolve_config_path("configs/experiments/vafidis_toy.yaml")

    assert resolved == LEARNING_ROOT / "configs/experiments/vafidis_toy.yaml"
