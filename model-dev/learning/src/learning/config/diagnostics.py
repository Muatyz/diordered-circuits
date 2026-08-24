"""Group selection for training-time and frozen-weight diagnostics."""

from __future__ import annotations

from typing import Any


DIAGNOSTIC_NAMES = frozenset(
    {
        "hd_tuning",
        "bump_maintenance",
        "bump_attractor_trajectories",
        "slow_manifold",
        "timescale_separation",
        "velocity_trajectory_sweep",
        "bump_diffusion",
        "darkness_path_integration",
        "ou_path_integration",
        "ou_pi_ensemble",
        "velocity_gain",
        "learning_error_development",
        "weight_structure",
        "weight_snapshot_pi_development",
        "numerical_convergence",
    }
)

DIAGNOSTIC_GROUPS = frozenset(
    {
        "bump_maintenance",
        "path_integration_and_pi_error",
        "pva_spectrum_and_visualization",
        "velocity_gain",
        "training_convergence",
        "trajectory_and_fixed_points",
        "weight_snapshots_and_development",
        "bump_diffusion",
        "timescale_separation",
        "velocity_dynamics_and_phase_flow",
        "numerical_convergence",
    }
)

DIAGNOSTICS_BY_GROUP = {
    "bump_maintenance": frozenset({"bump_maintenance"}),
    "path_integration_and_pi_error": frozenset(
        {
            "darkness_path_integration",
            "ou_path_integration",
            "ou_pi_ensemble",
        }
    ),
    # The PVA/Ramesan block includes the heading-response templates used for
    # visualization and the PCA/slow-manifold spectrum derived from activity.
    "pva_spectrum_and_visualization": frozenset({"hd_tuning", "slow_manifold"}),
    "velocity_gain": frozenset({"velocity_gain"}),
    "training_convergence": frozenset({"learning_error_development"}),
    "trajectory_and_fixed_points": frozenset(
        {"bump_attractor_trajectories"}
    ),
    "weight_snapshots_and_development": frozenset(
        {"weight_structure", "weight_snapshot_pi_development"}
    ),
    "bump_diffusion": frozenset({"bump_diffusion"}),
    "timescale_separation": frozenset({"timescale_separation"}),
    "velocity_dynamics_and_phase_flow": frozenset(
        {"velocity_trajectory_sweep"}
    ),
    "numerical_convergence": frozenset({"numerical_convergence"}),
}


def selected_diagnostics(config: Any) -> frozenset[str]:
    """Expand the enabled high-level groups into internal diagnostic jobs."""

    enabled = set()
    for group_name in selected_diagnostic_groups(config):
        enabled.update(DIAGNOSTICS_BY_GROUP[group_name])
    return frozenset(enabled)


def selected_diagnostic_groups(config: Any) -> frozenset[str]:
    """Return the high-level groups enabled by their boolean switches."""

    return frozenset(
        group_name
        for group_name in DIAGNOSTIC_GROUPS
        if getattr(config.diagnostics, group_name) is True
    )


def diagnostic_is_enabled(config: Any, diagnostic_name: str) -> bool:
    if diagnostic_name not in DIAGNOSTIC_NAMES:
        raise ValueError(f"Unknown diagnostic name: {diagnostic_name}")
    return diagnostic_name in selected_diagnostics(config)
