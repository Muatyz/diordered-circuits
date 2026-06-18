import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils import (
    finite_row_mask,
    fourier_correlation_error,
    generated_tuning_fourier_coefficients,
    sample_circular_gaussian_process,
    tuning_fourier_power_coefficients,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPRODUCTION_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPRODUCTION_ROOT.parent
CONFIG_PATH = REPRODUCTION_ROOT / "config/figure5.json"
PROCESSED = WORKSPACE_ROOT / "data/processed"
FIGURES = REPRODUCTION_ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_figure5_de_config(path=CONFIG_PATH):
    """
    Load and validate the Figure 5D-E fitting and plotting parameters.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    panels = config["panels_de"]
    sigma_values = np.asarray(panels["sigma_values"], dtype=float)
    if sigma_values.ndim != 1 or len(sigma_values) < 3 or np.any(sigma_values <= 0.0):
        raise ValueError("Figure 5D sigma_values must contain positive values")
    if int(panels["n_fourier_modes"]) <= 0:
        raise ValueError("n_fourier_modes must be positive")
    if int(panels["n_monte_carlo_samples"]) <= 0:
        raise ValueError("n_monte_carlo_samples must be positive")
    if not 0.0 < float(panels["beta_min"]) < float(panels["beta_max"]):
        raise ValueError("invalid beta bounds")
    if not 0.0 <= float(panels["bias_min"]) < float(panels["bias_max"]):
        raise ValueError("invalid bias bounds")
    return config


def resolve_tuning_path(path):
    """
    Resolve a processed tuning path saved with Windows or POSIX separators.
    """
    path = Path(str(path).replace("\\", "/"))
    for candidate in (path, WORKSPACE_ROOT / path, PROCESSED / path.name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find processed tuning file: {path}")


def load_pooled_experimental_tuning(index_path=PROCESSED / "hd_tuning_index.csv"):
    """
    Pool all finite, QC-passing, unit-mean tuning curves across mice.

    The paper fits one shared parameter set to the experimental correlation
    function pooled across recordings, rather than fitting each mouse
    separately.
    """
    if not index_path.exists():
        raise FileNotFoundError("Run python src\\compute_hd_tuning.py first")

    curves = []
    subject_ids = []
    n_angles = None
    for _, row in pd.read_csv(index_path).iterrows():
        data = np.load(resolve_tuning_path(row["tuning_path"]))
        tuning = np.asarray(data["normalized_rate"], dtype=float)
        included = np.asarray(data["included_qc"], dtype=bool)
        valid = included & finite_row_mask(tuning, min_fraction=1.0)
        tuning = tuning[valid]
        if tuning.shape[0] == 0:
            continue
        if n_angles is None:
            n_angles = tuning.shape[1]
        elif tuning.shape[1] != n_angles:
            raise ValueError("all experimental tuning curves must use the same angle grid")
        curves.append(tuning)
        subject_ids.extend([str(row["subject_id"])] * tuning.shape[0])

    if not curves:
        raise ValueError("No finite QC-passing experimental tuning curves were found")
    return np.vstack(curves), np.asarray(subject_ids)


def _generated_coefficients(currents, beta, bias, n_modes):
    """
    Evaluate generated correlation coefficients for one parameter pair.
    """
    return generated_tuning_fourier_coefficients(
        currents,
        beta=float(beta),
        bias=float(bias),
        n_modes=n_modes,
    )


def fit_beta_bias_at_sigma(
    currents,
    target_coefficients,
    n_modes,
    beta_bounds,
    bias_bounds,
    start,
):
    """
    Minimize the Figure 5 Fourier-correlation error at one fixed sigma.

    Fixed Gaussian white noise is reused for every objective evaluation. This
    common-random-number construction removes Monte Carlo jitter from the
    parameter landscape without changing the generative process.
    """
    def objective(parameters):
        beta, bias = parameters
        generated = _generated_coefficients(currents, beta, bias, n_modes)
        return fourier_correlation_error(target_coefficients, generated)

    result = minimize(
        objective,
        x0=np.asarray(start, dtype=float),
        method="Nelder-Mead",
        bounds=(beta_bounds, bias_bounds),
        options={
            "xatol": 1e-4,
            "fatol": 1e-10,
            "maxiter": 200,
        },
    )
    if not result.success:
        raise RuntimeError(f"beta/bias optimization failed: {result.message}")
    return {
        "beta": float(result.x[0]),
        "bias": float(result.x[1]),
        "error": float(result.fun),
        "n_function_evaluations": int(result.nfev),
    }


def evaluate_error_landscape(
    currents,
    target_coefficients,
    n_modes,
    beta_values,
    bias_values,
):
    """
    Evaluate E(sigma, beta, b) on the panel-E rectangular parameter grid.
    """
    error = np.empty((len(beta_values), len(bias_values)), dtype=float)
    for beta_index, beta in enumerate(beta_values):
        for bias_index, bias in enumerate(bias_values):
            generated = _generated_coefficients(currents, beta, bias, n_modes)
            error[beta_index, bias_index] = fourier_correlation_error(
                target_coefficients,
                generated,
            )
    return error


def build_figure5_de_data(config):
    """
    Fit the Gaussian generative process and construct Figure 5D-E source data.
    """
    panels = config["panels_de"]
    tuning, subject_ids = load_pooled_experimental_tuning()
    n_angles = tuning.shape[1]
    n_modes = int(panels["n_fourier_modes"])
    target = tuning_fourier_power_coefficients(tuning, n_modes=n_modes)

    rng = np.random.default_rng(int(panels["seed"]))
    white_noise = rng.normal(
        size=(int(panels["n_monte_carlo_samples"]), n_angles)
    )
    sigma_values = np.asarray(panels["sigma_values"], dtype=float)
    beta_bounds = (float(panels["beta_min"]), float(panels["beta_max"]))
    bias_bounds = (float(panels["bias_min"]), float(panels["bias_max"]))
    start = np.asarray(panels["optimizer_start"], dtype=float)

    fit_rows = []
    for sigma in sigma_values:
        _, currents = sample_circular_gaussian_process(
            n_samples=white_noise.shape[0],
            n_angles=n_angles,
            sigma=sigma,
            white_noise=white_noise,
        )
        fit = fit_beta_bias_at_sigma(
            currents,
            target,
            n_modes,
            beta_bounds,
            bias_bounds,
            start,
        )
        fit_rows.append({"sigma": float(sigma), **fit})
        start = np.asarray([fit["beta"], fit["bias"]])

    fit_table = pd.DataFrame(fit_rows)
    best_index = int(fit_table["error"].idxmin())
    best_row = fit_table.loc[best_index]
    best_sigma = float(best_row["sigma"])
    _, best_currents = sample_circular_gaussian_process(
        n_samples=white_noise.shape[0],
        n_angles=n_angles,
        sigma=best_sigma,
        white_noise=white_noise,
    )
    beta_values = np.linspace(
        beta_bounds[0],
        beta_bounds[1],
        int(panels["beta_points"]),
    )
    bias_values = np.linspace(
        bias_bounds[0],
        bias_bounds[1],
        int(panels["bias_points"]),
    )
    landscape = evaluate_error_landscape(
        best_currents,
        target,
        n_modes,
        beta_values,
        bias_values,
    )
    grid_minimum = np.unravel_index(int(np.argmin(landscape)), landscape.shape)

    paper = panels["paper_fit"]
    _, paper_currents = sample_circular_gaussian_process(
        n_samples=white_noise.shape[0],
        n_angles=n_angles,
        sigma=float(paper["sigma"]),
        white_noise=white_noise,
    )
    paper_coefficients = _generated_coefficients(
        paper_currents,
        float(paper["beta"]),
        float(paper["bias"]),
        n_modes,
    )

    return {
        "fit_table": fit_table,
        "target_coefficients": target,
        "beta_values": beta_values,
        "bias_values": bias_values,
        "error_landscape": landscape,
        "best_sigma": best_sigma,
        "best_beta": float(best_row["beta"]),
        "best_bias": float(best_row["bias"]),
        "best_error": float(best_row["error"]),
        "grid_min_beta": float(beta_values[grid_minimum[0]]),
        "grid_min_bias": float(bias_values[grid_minimum[1]]),
        "paper_coefficients": paper_coefficients,
        "paper_error": fourier_correlation_error(target, paper_coefficients),
        "n_fourier_modes": int(n_modes),
        "n_neurons": int(tuning.shape[0]),
        "n_mice": int(len(np.unique(subject_ids))),
        "n_angles": int(n_angles),
    }


def save_figure5_de_data(data):
    """
    Save the fitted curve, error landscape, coefficients, and diagnostics.
    """
    fit_path = PROCESSED / "figure5_d_sigma_profile.csv"
    arrays_path = PROCESSED / "figure5_de_parameter_fit.npz"
    diagnostics_path = PROCESSED / "figure5_de_parameter_fit.json"
    data["fit_table"].to_csv(fit_path, index=False)
    np.savez_compressed(
        arrays_path,
        target_coefficients=data["target_coefficients"],
        beta_values=data["beta_values"],
        bias_values=data["bias_values"],
        error_landscape=data["error_landscape"],
        paper_coefficients=data["paper_coefficients"],
    )
    diagnostics = {
        key: value
        for key, value in data.items()
        if key
        in {
            "best_sigma",
            "best_beta",
            "best_bias",
            "best_error",
            "grid_min_beta",
            "grid_min_bias",
            "paper_error",
            "n_fourier_modes",
            "n_neurons",
            "n_mice",
            "n_angles",
        }
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    return fit_path, arrays_path, diagnostics_path


def load_saved_figure5_de_data():
    """
    Reload a completed Figure 5D-E scan without repeating the Monte Carlo fit.
    """
    fit_path = PROCESSED / "figure5_d_sigma_profile.csv"
    arrays_path = PROCESSED / "figure5_de_parameter_fit.npz"
    diagnostics_path = PROCESSED / "figure5_de_parameter_fit.json"
    missing = [
        path
        for path in (fit_path, arrays_path, diagnostics_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing saved Figure 5D-E data: {missing}")

    arrays = np.load(arrays_path)
    data = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    data.update(
        {
            "fit_table": pd.read_csv(fit_path),
            "target_coefficients": arrays["target_coefficients"],
            "beta_values": arrays["beta_values"],
            "bias_values": arrays["bias_values"],
            "error_landscape": arrays["error_landscape"],
            "paper_coefficients": arrays["paper_coefficients"],
        }
    )
    return data


def plot_figure5_de(data, config):
    """
    Render the sigma profile and beta-bias error landscape of Figure 5D-E.
    """
    panels = config["panels_de"]
    paper = panels["paper_fit"]
    fit_table = data["fit_table"]

    fig, (ax_d, ax_e) = plt.subplots(1, 2, figsize=(7.1, 3.25))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.2, top=0.82, wspace=0.38)

    ax_d.plot(
        fit_table["sigma"],
        fit_table["error"],
        color="black",
        marker="o",
        markersize=4.2,
        linewidth=1.25,
    )
    y_max = 1.06 * float(fit_table["error"].max())
    ax_d.plot(
        [float(paper["sigma"]), float(paper["sigma"])],
        [0.0, y_max],
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
    ax_d.set(
        xlabel=r"$\sigma$ [Fourier decay scale]",
        ylabel=r"min. error at $\sigma$",
        xlim=(fit_table["sigma"].min() - 0.03, fit_table["sigma"].max() + 0.03),
        ylim=(0.0, y_max),
    )
    ax_d.spines[["top", "right"]].set_visible(False)

    error = data["error_landscape"]
    error_min = float(np.min(error))
    error_span = max(float(np.max(error) - error_min), 1e-15)
    normalized_error = (error - error_min) / error_span
    image = ax_e.imshow(
        normalized_error,
        origin="lower",
        extent=(
            data["bias_values"][0],
            data["bias_values"][-1],
            data["beta_values"][0],
            data["beta_values"][-1],
        ),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax_e.scatter(
        data["best_bias"],
        data["best_beta"],
        s=34,
        facecolor="white",
        edgecolor="none",
        zorder=3,
    )
    ax_e.set(
        xlabel=r"$b$ [nonlin. bias]",
        ylabel=r"$\beta$ [nonlin. steepness]",
    )
    colorbar = fig.colorbar(
        image,
        ax=ax_e,
        orientation="horizontal",
        location="top",
        fraction=0.08,
        pad=0.04,
        ticks=[0.0, 0.5, 1.0],
    )
    colorbar.set_label("error", labelpad=2)

    for label, ax in zip("DE", (ax_d, ax_e)):
        ax.text(
            -0.14,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
        )

    output = FIGURES / "figure5_de_reproduction.png"
    fig.savefig(output, dpi=260)
    plt.close(fig)
    return output


def main(argv=None):
    """
    Fit the Figure 5 generative process and save panels D-E and source data.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw from saved fit data without repeating the parameter scan",
    )
    args = parser.parse_args(argv)
    config = load_figure5_de_config()
    if args.plot_only:
        data = load_saved_figure5_de_data()
        data_paths = ()
    else:
        data = build_figure5_de_data(config)
        data_paths = save_figure5_de_data(data)
    figure_path = plot_figure5_de(data, config)
    print("Saved:", figure_path)
    for path in data_paths:
        print("Saved:", path)
    print(
        "Best fit:",
        f"sigma={data['best_sigma']:.4g},",
        f"beta={data['best_beta']:.4g},",
        f"b={data['best_bias']:.4g},",
        f"error={data['best_error']:.6g}",
    )
    print("Paper-parameter error:", f"{data['paper_error']:.6g}")
    print("Experimental curves:", data["n_neurons"], "from", data["n_mice"], "mice")


if __name__ == "__main__":
    main()
