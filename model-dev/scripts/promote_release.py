"""Safely promote a tested ``model-dev`` snapshot into a frozen release tree.

The default release tree is ``model-release``.  ``--release-root`` selects an
alternate frozen snapshot directory (e.g. ``model-release-anneal``) so an
experiment series can run long training on a frozen copy of the latest dev
code without disturbing the default release tree or the dev tree.  Each
snapshot keeps its own manifest and training lock.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "model-dev"
DEFAULT_RELEASE_ROOT = REPOSITORY_ROOT / "model-release"
MANIFEST_NAME = ".release-manifest.json"
LOCK_NAME = ".training-active"

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".todo",
    "__pycache__",
}
EXCLUDED_PREFIXES = {
    ("learning", "reports"),
    ("learning", "runs"),
    ("reproduction", "data"),
    ("reproduction", "reports"),
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_managed(relative_path: Path) -> bool:
    """Return whether a source file belongs to the reproducible release snapshot."""
    parts = relative_path.parts
    if relative_path.name in {MANIFEST_NAME, LOCK_NAME}:
        return False
    if relative_path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in parts[:-1]):
        return False
    return not any(parts[: len(prefix)] == prefix for prefix in EXCLUDED_PREFIXES)


def collect_managed_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    files = {}
    for path in root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(root)
            if is_managed(relative):
                files[relative.as_posix()] = path
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "model-dev"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def current_release_files(release_root: Path) -> dict[str, Path]:
    manifest_path = release_root / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            names = manifest["managed_files"].keys()
            return {
                name: release_root / Path(name)
                for name in names
                if (release_root / Path(name)).is_file()
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            raise RuntimeError(f"Invalid release manifest: {manifest_path}") from None
    return collect_managed_files(release_root)


def calculate_changes(
    source_files: dict[str, Path], release_files: dict[str, Path]
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    source_hashes = {name: sha256(path) for name, path in source_files.items()}
    added = sorted(source_files.keys() - release_files.keys())
    removed = sorted(release_files.keys() - source_files.keys())
    changed = sorted(
        name
        for name in source_files.keys() & release_files.keys()
        if source_hashes[name] != sha256(release_files[name])
    )
    return added, changed, removed, source_hashes


def print_changes(added: list[str], changed: list[str], removed: list[str]) -> None:
    print(
        f"release preview: {len(added)} added, "
        f"{len(changed)} changed, {len(removed)} removed"
    )
    for marker, names in (("+", added), ("~", changed), ("-", removed)):
        for name in names:
            print(f"{marker} {name}")


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def remove_empty_parents(path: Path, release_root: Path) -> None:
    parent = path.parent
    while parent != release_root:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def apply_release(
    source_files: dict[str, Path],
    copy_names: list[str],
    removed: list[str],
    source_hashes: dict[str, str],
    *,
    allow_dirty: bool,
    release_root: Path,
) -> None:
    lock_path = release_root / LOCK_NAME
    if lock_path.exists():
        raise RuntimeError(
            f"Release is marked as training-active; refusing to update: {lock_path}"
        )

    commit, dirty = git_state()
    if dirty and not allow_dirty:
        raise RuntimeError(
            "model-dev has uncommitted changes; commit them first or explicitly use "
            "--allow-dirty"
        )

    release_root.mkdir(parents=True, exist_ok=True)
    for name in removed:
        destination = release_root / Path(name)
        destination.unlink(missing_ok=True)
        remove_empty_parents(destination, release_root)
    for name in copy_names:
        atomic_copy(source_files[name], release_root / Path(name))

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "source": "model-dev",
        "release_root": release_root.name,
        "git_commit": commit,
        "git_dirty": dirty,
        "managed_files": dict(sorted(source_hashes.items())),
    }
    manifest_path = release_root / MANIFEST_NAME
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(f"release updated: {manifest_path}")


def set_training_lock(release_root: Path) -> None:
    release_root.mkdir(parents=True, exist_ok=True)
    lock_path = release_root / LOCK_NAME
    payload = {
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
    }
    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    except FileExistsError:
        raise RuntimeError(f"Training lock already exists: {lock_path}") from None
    print(f"training lock created: {lock_path}")


def clear_training_lock(release_root: Path) -> None:
    lock_path = release_root / LOCK_NAME
    if not lock_path.is_file():
        raise RuntimeError(f"Training lock does not exist: {lock_path}")
    lock_path.unlink()
    print(f"training lock removed: {lock_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        default=None,
        metavar="DIR",
        help=(
            "Frozen snapshot directory name under the repository root.  "
            f"Defaults to {DEFAULT_RELEASE_ROOT.name}.  Each snapshot has its "
            "own manifest and training lock, so an alternate root can run long "
            "training on the latest dev code without touching the default "
            "release tree."
        ),
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--apply", action="store_true", help="apply the previewed promotion")
    actions.add_argument(
        "--lock-training", action="store_true", help="block promotions during a release run"
    )
    actions.add_argument(
        "--unlock-training", action="store_true", help="remove the release training lock"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow promotion from an uncommitted model-dev tree",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_root = (
        Path(args.release_root).resolve()
        if args.release_root
        else DEFAULT_RELEASE_ROOT
    )
    if args.lock_training:
        set_training_lock(release_root)
        return
    if args.unlock_training:
        clear_training_lock(release_root)
        return
    if args.allow_dirty and not args.apply:
        raise SystemExit("--allow-dirty is only valid with --apply")

    source_files = collect_managed_files(SOURCE_ROOT)
    if not source_files:
        raise RuntimeError(f"No development files found under {SOURCE_ROOT}")
    release_files = current_release_files(release_root)
    added, changed, removed, hashes = calculate_changes(source_files, release_files)
    print_changes(added, changed, removed)
    if args.apply:
        apply_release(
            source_files,
            sorted([*added, *changed]),
            removed,
            hashes,
            allow_dirty=args.allow_dirty,
            release_root=release_root,
        )
    else:
        print("dry run only; rerun with --apply after tests pass")


if __name__ == "__main__":
    main()
