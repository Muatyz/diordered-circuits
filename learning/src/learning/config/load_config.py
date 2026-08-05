"""YAML config loading and project-root resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import yaml

from learning.config.diagnostics import DIAGNOSTIC_GROUPS
from learning.config.schema import ExperimentConfig, experiment_config_from_dict


DIAGNOSTIC_SIMULATION_FIELDS = frozenset(
    {
        "bump_test_duration",
        "darkness_test_duration",
        "cue_duration",
        "pi_cue_duration",
        "recue_duration",
    }
)


def find_project_root(start_path: str | Path | None = None) -> Path:
    """Find the learning subproject root by walking upward."""
    current_path = Path(start_path or Path.cwd()).resolve()
    if current_path.is_file():
        current_path = current_path.parent
    for candidate_path in [current_path, *current_path.parents]:
        if (candidate_path / "pyproject.toml").exists() and (candidate_path / "src" / "learning").exists():
            return candidate_path
        if (candidate_path / ".SKILL.md").exists() and (candidate_path / "notebooks").exists():
            return candidate_path
    raise FileNotFoundError(f"Could not locate learning project root from {current_path}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        loaded_value = yaml.safe_load(file_handle) or {}
    if not isinstance(loaded_value, dict):
        raise ValueError(f"Expected mapping at {path}")
    return loaded_value


def save_yaml(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(value, file_handle, sort_keys=False, allow_unicode=True)


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    merged = deepcopy(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge(base_value, override_value)
        else:
            merged[key] = deepcopy(override_value)
    return merged


def _coerce_override_value(
    *,
    current_value: Any,
    override_value: Any,
    dotted_path: str,
) -> Any:
    """Validate a profile/CLI value against the resolved config value type."""

    if override_value is None or current_value is None:
        return deepcopy(override_value)
    if isinstance(current_value, bool):
        if not isinstance(override_value, bool):
            raise ValueError(f"Override {dotted_path} must be a boolean")
        return override_value
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        if not isinstance(override_value, int) or isinstance(override_value, bool):
            raise ValueError(f"Override {dotted_path} must be an integer")
        return int(override_value)
    if isinstance(current_value, float):
        if not isinstance(override_value, (int, float)) or isinstance(
            override_value, bool
        ):
            raise ValueError(f"Override {dotted_path} must be numeric")
        return float(override_value)
    if isinstance(current_value, str):
        if not isinstance(override_value, str):
            raise ValueError(f"Override {dotted_path} must be a string")
        return override_value
    if isinstance(current_value, list):
        if not isinstance(override_value, list):
            raise ValueError(f"Override {dotted_path} must be a list")
        return deepcopy(override_value)
    if isinstance(current_value, dict):
        if not isinstance(override_value, dict):
            raise ValueError(f"Override {dotted_path} must be a mapping")
        return deepcopy(override_value)
    return deepcopy(override_value)


def _merge_profile_mapping(
    base: dict[str, Any],
    profile: dict[str, Any],
    *,
    path_prefix: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Merge a partial reusable profile while rejecting unknown/type-wrong fields."""

    merged = deepcopy(base)
    for key, profile_value in profile.items():
        current_path = (*path_prefix, key)
        dotted_path = ".".join(current_path)
        if key not in merged:
            raise ValueError(f"Unknown config override path: {dotted_path}")
        current_value = merged[key]
        if isinstance(current_value, dict) and isinstance(profile_value, dict):
            merged[key] = _merge_profile_mapping(
                current_value,
                profile_value,
                path_prefix=current_path,
            )
        else:
            merged[key] = _coerce_override_value(
                current_value=current_value,
                override_value=profile_value,
                dotted_path=dotted_path,
            )
    return merged


def apply_config_overrides(
    config_mapping: dict[str, Any],
    assignments: Sequence[str],
) -> dict[str, Any]:
    """Apply ordered ``dotted.path=YAML_VALUE`` assignments to a config copy."""

    overridden = deepcopy(config_mapping)
    for assignment in assignments:
        dotted_path, separator, raw_value = assignment.partition("=")
        dotted_path = dotted_path.strip()
        if not separator or not dotted_path:
            raise ValueError(
                "Config overrides must use dotted.path=value syntax; "
                f"got {assignment!r}"
            )
        path_components = dotted_path.split(".")
        if any(not component for component in path_components):
            raise ValueError(f"Invalid config override path: {dotted_path!r}")
        target: dict[str, Any] = overridden
        for component in path_components[:-1]:
            if component not in target or not isinstance(target[component], dict):
                raise ValueError(f"Unknown config override path: {dotted_path}")
            target = target[component]
        leaf_name = path_components[-1]
        if leaf_name not in target:
            raise ValueError(f"Unknown config override path: {dotted_path}")
        try:
            parsed_value = yaml.safe_load(raw_value)
        except yaml.YAMLError as error:
            raise ValueError(
                f"Could not parse YAML value for override {dotted_path}: {raw_value!r}"
            ) from error
        target[leaf_name] = _coerce_override_value(
            current_value=target[leaf_name],
            override_value=parsed_value,
            dotted_path=dotted_path,
        )
    return overridden


def _validate_diagnostics_mapping(
    diagnostics: dict[str, Any],
    *,
    source_path: Path,
) -> None:
    unknown_sections = set(diagnostics) - {"simulation", "diagnostics", "tests"}
    if unknown_sections:
        raise ValueError(
            f"Diagnostics config {source_path} may only contain simulation/tests "
            "plus the diagnostics selection section; "
            f"found {sorted(unknown_sections)}"
        )

    simulation = diagnostics.get("simulation", {})
    if not isinstance(simulation, dict):
        raise ValueError(f"simulation must be a mapping in {source_path}")
    unknown_simulation_fields = set(simulation) - DIAGNOSTIC_SIMULATION_FIELDS
    if unknown_simulation_fields:
        raise ValueError(
            f"Diagnostics config {source_path} cannot override training simulation "
            f"fields {sorted(unknown_simulation_fields)}"
        )

    tests = diagnostics.get("tests", {})
    if not isinstance(tests, dict):
        raise ValueError(f"tests must be a mapping in {source_path}")

    group_switches = diagnostics.get("diagnostics")
    if not isinstance(group_switches, dict):
        raise ValueError(
            f"Diagnostics config {source_path} must define a diagnostics mapping"
        )
    missing_groups = DIAGNOSTIC_GROUPS - set(group_switches)
    unknown_fields = set(group_switches) - (
        DIAGNOSTIC_GROUPS | {"reuse_cached_dependencies"}
    )
    if missing_groups:
        raise ValueError(
            f"Diagnostics config {source_path} is missing group switches "
            f"{sorted(missing_groups)}"
        )
    if unknown_fields:
        raise ValueError(
            f"Diagnostics config {source_path} has unknown diagnostics fields "
            f"{sorted(unknown_fields)}"
        )
    non_boolean_groups = sorted(
        group_name
        for group_name in DIAGNOSTIC_GROUPS
        if not isinstance(group_switches[group_name], bool)
    )
    if non_boolean_groups:
        raise ValueError(
            f"Diagnostics group switches must be booleans: {non_boolean_groups}"
        )
    if (
        "reuse_cached_dependencies" in group_switches
        and not isinstance(group_switches["reuse_cached_dependencies"], bool)
    ):
        raise ValueError("diagnostics.reuse_cached_dependencies must be a boolean")


def _load_diagnostics_mapping(
    diagnostics_path: Path,
) -> dict[str, Any]:
    diagnostics_path = diagnostics_path.resolve()
    raw_diagnostics = load_yaml(diagnostics_path)
    _validate_diagnostics_mapping(raw_diagnostics, source_path=diagnostics_path)
    return raw_diagnostics


def load_experiment_config(
    config_path: str | Path,
    *,
    diagnostics_path: str | Path | None = None,
    profile_paths: Sequence[str | Path] = (),
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """Load an experiment plus one optional diagnostics hyper config.

    Generic profiles are partial ExperimentConfig mappings; ``overrides`` use
    ``dotted.path=YAML_VALUE`` syntax and have the highest precedence.
    """

    resolved_config_path = Path(config_path).resolve()
    raw_config = load_yaml(resolved_config_path)
    if diagnostics_path is not None:
        explicit_diagnostics_path = Path(diagnostics_path).resolve()
        raw_config = _deep_merge(
            raw_config,
            _load_diagnostics_mapping(explicit_diagnostics_path),
        )
    # Materialize defaults before applying generic profiles so every legal
    # dotted field exists and typos can be rejected immediately.
    resolved_mapping = experiment_config_from_dict(raw_config).to_dict()
    for profile_path in profile_paths:
        resolved_profile_path = Path(profile_path).resolve()
        profile_mapping = load_yaml(resolved_profile_path)
        resolved_mapping = _merge_profile_mapping(
            resolved_mapping,
            profile_mapping,
        )
    resolved_mapping = apply_config_overrides(resolved_mapping, overrides)
    return experiment_config_from_dict(resolved_mapping)
