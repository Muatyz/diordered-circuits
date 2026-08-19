try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()
import json
import time

import numpy as np

from diagnose_inverse_floor import factors_from_target_current, softplus_inverse_with_floor
from plot_figure3_abcd import PROCESSED, REPRODUCTION_ROOT
from utils import (
    lowrank_recurrent_drive,
    nearest_manifold_distance,
    simulate_rate_network_with_drive,
)


REPORTS = REPRODUCTION_ROOT / "reports"
OUT_PATH = REPORTS / "figure3_lambda_c_full_dynamics.json"


def elapsed_text(seconds):
    """
    Format elapsed wall time for progress messages.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


def l2_rows(values):
    """
    Compute per-row L2 norms without numpy.linalg.
    """
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.sum(values * values, axis=1))


def make_initial_states(target_current, n_initial, noise_std, seed):
    """
    Sample regularly spaced target-manifold points plus iid current noise.
    """
    rng = np.random.default_rng(seed)
    angle_idx = np.linspace(0, target_current.shape[0], n_initial, endpoint=False, dtype=int)
    initial = target_current[angle_idx].copy()
    initial += rng.normal(0.0, noise_std, size=initial.shape)
    return angle_idx, initial


def summarize_case(phi, beta, regularization, inhibition_c, eps, n_initial, duration_s, seed):
    """
    Run one full 30 s perturbation simulation and summarize convergence.
    """
    start = time.perf_counter()
    x_star = softplus_inverse_with_floor(phi, beta=beta, eps=eps)
    target_current = x_star.T
    factor_a, factor_b, diagonal = factors_from_target_current(
        phi,
        x_star,
        regularization=regularization,
        enforce_zero_diagonal=True,
    )
    drive = lowrank_recurrent_drive(factor_a, factor_b, diagonal)
    residual = l2_rows(drive(phi.T) - target_current) / np.sqrt(phi.shape[0])
    angle_idx, initial = make_initial_states(target_current, n_initial=n_initial, noise_std=0.05, seed=seed)

    times, trajectory = simulate_rate_network_with_drive(
        drive,
        initial,
        tau_s=0.05,
        dt_s=0.001,
        duration_s=duration_s,
        inhibition_c=inhibition_c,
        activation_beta=beta,
        record_every_s=0.5,
        current_clip=None,
        stop_abs=1e12,
        progress_label=f"lambda={regularization:g}, c={inhibition_c:g}, eps={eps:g}",
        progress_interval_wall_s=30.0,
    )

    flat_distance, _, _ = nearest_manifold_distance(
        trajectory.reshape(-1, trajectory.shape[-1]),
        target_current,
        return_l2=True,
    )
    distance = flat_distance.reshape(trajectory.shape[:2])
    finite_distance = distance[np.isfinite(distance)]
    final = distance[-1]
    final_finite = final[np.isfinite(final)]
    finite_values = trajectory[np.isfinite(trajectory)]
    max_abs_state = float(np.max(np.abs(finite_values))) if finite_values.size else float("inf")
    actual_duration_s = float(times[-1]) if len(times) else 0.0
    stopped_early = bool(actual_duration_s < duration_s - 0.25)

    return {
        "regularization": float(regularization),
        "inhibition_c": float(inhibition_c),
        "inverse_eps": float(eps),
        "n_initial": int(n_initial),
        "seed": int(seed),
        "planned_duration_s": float(duration_s),
        "actual_duration_s": actual_duration_s,
        "stopped_early": stopped_early,
        "target_flow_residual_median": float(np.median(residual)),
        "target_flow_residual_max": float(np.max(residual)),
        "initial_distance_median": float(np.nanmedian(distance[0])),
        "initial_distance_max": float(np.nanmax(distance[0])),
        "final_distance_median": float(np.nanmedian(final_finite)) if final_finite.size else float("nan"),
        "final_distance_max": float(np.nanmax(final_finite)) if final_finite.size else float("nan"),
        "all_distance_max": float(np.nanmax(finite_distance)) if finite_distance.size else float("nan"),
        "max_abs_state": max_abs_state,
        "finite_trajectory": bool(np.isfinite(trajectory).all()),
        "elapsed_wall_s": float(time.perf_counter() - start),
    }


def main():
    """
    Run full-duration lambda/c diagnostics for the current Figure 3 target.
    """
    data = np.load(PROCESSED / "figure3_abcd_weight_matrices.npz")
    phi = data["doubly_normalized_tuning"].astype(float)
    beta = float(data["activation_beta"])
    n_neurons, n_angles = phi.shape

    cases = []
    lambda_cases = [
        ("paper_lambda", 1e-6),
        ("eq27_kernel_plus_1e-6_I_equivalent", 1e-6 * n_neurons / n_angles),
        ("lambda_3e-5", 3e-5),
        ("lambda_1e-4", 1e-4),
        ("lambda_3e-4", 3e-4),
        ("lambda_1e-3", 1e-3),
    ]
    c_cases = [0.5, 1.0, 2.0]
    eps_cases = [1e-12]

    for label, regularization in lambda_cases:
        for inhibition_c in c_cases:
            for eps in eps_cases:
                print(
                    f"Running {label}: lambda={regularization:g}, c={inhibition_c:g}, eps={eps:g}",
                    flush=True,
                )
                case = summarize_case(
                    phi,
                    beta=beta,
                    regularization=regularization,
                    inhibition_c=inhibition_c,
                    eps=eps,
                    n_initial=24,
                    duration_s=30.0,
                    seed=20260531,
                )
                case["label"] = label
                cases.append(case)
                print(
                    "  actual="
                    f"{case['actual_duration_s']:.3f}s final_med={case['final_distance_median']:.3g} "
                    f"final_max={case['final_distance_max']:.3g} stopped={case['stopped_early']} "
                    f"wall={elapsed_text(case['elapsed_wall_s'])}",
                    flush=True,
                )

    report = {
        "status": "diagnostic_only_no_pipeline_change",
        "target_source": str(PROCESSED / "figure3_abcd_weight_matrices.npz"),
        "n_neurons": int(n_neurons),
        "n_angles": int(n_angles),
        "activation_beta": beta,
        "cases": cases,
        "interpretation_hint": (
            "Only cases with lambda=1e-6 and c=1 match the paper's stated "
            "Figure 3 setting; other cases test whether instability is a "
            "regularization or mean-mode stabilization scale issue."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
