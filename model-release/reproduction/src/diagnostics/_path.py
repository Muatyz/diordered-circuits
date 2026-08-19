from pathlib import Path
import sys


def ensure_src_on_path():
    """
    Add `reproduction/src` to sys.path for standalone diagnostic scripts.
    """
    src_dir = Path(__file__).resolve().parents[1]
    src_text = str(src_dir)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
