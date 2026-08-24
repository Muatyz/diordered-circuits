# python ./reproduction/src/plot_attractor_stability_summary.py
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from plot_figure3_abcd import REPRODUCTION_ROOT


REPORTS = REPRODUCTION_ROOT / "reports"
SOURCE_PATH = REPORTS / "figure3_lambda_perturbation_grid.json"
OUTPUT_PATH = REPORTS / "figures/attractor_stability_parameter_summary.png"


def load_cases(source_path):
    """
    Load and index the completed lambda-by-perturbation scan.
    """
    report = json.loads(Path(source_path).read_text(encoding="utf-8"))
    cases = {
        (float(case["regularization"]), float(case["noise_std"])): case
        for case in report["cases"]
    }
    lambdas = sorted({key[0] for key in cases})
    noise_stds = sorted({key[1] for key in cases})
    return report, cases, lambdas, noise_stds


def plot_summary(source_path=SOURCE_PATH, output_path=OUTPUT_PATH):
    """
    Render the parameter-scan summary with standard Matplotlib styling.
    """
    _, cases, lambdas, noise_stds = load_cases(source_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_text = axes.ravel()

    distance_grid = np.full((len(noise_stds), len(lambdas)), np.nan)
    for row, noise_std in enumerate(noise_stds):
        for col, regularization in enumerate(lambdas):
            case = cases[(regularization, noise_std)]
            if not case["stopped_early"] and case["finite_trajectory"]:
                distance_grid[row, col] = case["final_distance_median"]

    finite_distances = distance_grid[np.isfinite(distance_grid)]
    image = ax_a.imshow(
        distance_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=np.min(finite_distances), vmax=np.max(finite_distances)),
    )
    ax_a.set_xticks(np.arange(len(lambdas)), [f"{value:.0e}" for value in lambdas])
    ax_a.set_yticks(np.arange(len(noise_stds)), [f"{value:g}" for value in noise_stds])
    ax_a.set_xlabel(r"Regularization $\lambda$")
    ax_a.set_ylabel(r"Initial perturbation $\sigma$")
    ax_a.set_title("A  Final distance across parameter space")
    fig.colorbar(image, ax=ax_a, label="30 s median distance")
    for row, noise_std in enumerate(noise_stds):
        for col, regularization in enumerate(lambdas):
            case = cases[(regularization, noise_std)]
            if case["stopped_early"] or not case["finite_trajectory"]:
                ax_a.text(col, row, "×", ha="center", va="center")

    for noise_std in noise_stds:
        stable_lambda = []
        stable_distance = []
        for regularization in lambdas:
            case = cases[(regularization, noise_std)]
            if not case["stopped_early"] and case["finite_trajectory"]:
                stable_lambda.append(regularization)
                stable_distance.append(case["final_distance_median"])
        ax_b.loglog(stable_lambda, stable_distance, marker="o", label=fr"$\sigma={noise_std:g}$")
    ax_b.axhline(1e-3, color="black", linestyle="--", label="Paper ~1e-3")
    ax_b.set_xlabel(r"Regularization $\lambda$")
    ax_b.set_ylabel("Final distance")
    ax_b.set_title("B  Regularization sets the residual error floor")
    ax_b.legend()

    best_lambda = 7e-5
    selected = [cases[(best_lambda, noise_std)] for noise_std in noise_stds]
    initial = [case["initial_distance_median"] for case in selected]
    final = [case["final_distance_median"] for case in selected]
    ax_c.loglog(noise_stds, initial, marker="o", label="Initial")
    ax_c.loglog(noise_stds, final, marker="o", label="30 s")
    ax_c.set_xlabel(r"Initial perturbation $\sigma$")
    ax_c.set_ylabel("Distance")
    ax_c.set_title(r"C  Basin test at $\lambda=7\times10^{-5}$")
    ax_c.legend()

    ax_text.axis("off")
    ax_text.set_title("Interpretation")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "• Paper λ=1e-6 is unstable for the reconstructed target.",
                "• Global stability emerges near λ=7e-5.",
                "• Stable runs plateau near 1.2e-2.",
                "• Perturbation size changes relaxation time,",
                "  but not the final distance floor.",
                "• The remaining discrepancy points upstream to",
                "  tuning-curve or target-manifold reconstruction.",
            ]
        ),
        va="top",
    )

    fig.suptitle("Attractor stability parameter summary")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    """
    Command-line entry point for the standalone stability summary.
    """
    output_path = plot_summary()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
