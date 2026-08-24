from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from workspace_paths import DATA_ROOT, MODEL_ROOT, REPOSITORY_ROOT, REPRODUCTION_ROOT


def test_nested_development_workspace_paths():
    repository_root = Path(__file__).resolve().parents[3]

    assert REPOSITORY_ROOT == repository_root
    assert MODEL_ROOT == repository_root / "model-dev"
    assert REPRODUCTION_ROOT == MODEL_ROOT / "reproduction"
    assert DATA_ROOT == repository_root / "data"
