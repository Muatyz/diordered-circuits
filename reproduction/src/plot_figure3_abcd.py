import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from render_utils import diverging_rgb, draw_centered_text, load_font, matrix_heatmap
from utils import (
    circular_center_of_mass_angles,
    circulant_from_diagonal_means,
    finite_row_mask,
    materialize_lowrank_weights,
    optimized_recurrent_factors,
    prepare_phi_star_for_inverse,
    scramble_residuals,
    fixed_point_residual_from_weights,
    current_space_jacobian,
    invert_spd_cholesky,
    softplus_inverse,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPRODUCTION_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPRODUCTION_ROOT.parent


def existing_data_dir(name):
    """
    Return the preferred data directory while tolerating the repository move.

    Older cached files live under the workspace-level `data/` directory,
    whereas the refactored reproduction project may keep data under
    `reproduction/data/`. The first existing location wins; if neither exists,
    the reproduction-local path is returned so new outputs stay with the
    subproject.
    """
    candidates = [REPRODUCTION_ROOT / "data" / name, WORKSPACE_ROOT / "data" / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_existing_path(path):
    """
    Resolve a path saved before or after the reproduction directory move.
    """
    path = Path(path)
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        Path.cwd() / path,
        WORKSPACE_ROOT / path,
        REPRODUCTION_ROOT / path,
        PROCESSED / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


PROCESSED = existing_data_dir("processed")
FIGURES = REPRODUCTION_ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)
WEIGHT_FORMULA_VERSION = "paper_a2_discrete_kernel_b2_per_neuron_cv_v15_phi_safe"
PAPER_REGULARIZATION = 1e-6
STABLE_REGULARIZATION = 1e-4


def compact_preprocess_info(info):
    """
    Keep preprocessing diagnostics readable while saving masks separately.
    """
    compact = dict(info)
    if "valid_neuron_mask" in compact:
        mask = np.asarray(compact.pop("valid_neuron_mask"), dtype=bool)
        compact["valid_neuron_mask_true_count"] = int(np.sum(mask))
        compact["valid_neuron_mask_false_count"] = int(mask.size - np.sum(mask))
    return compact


def load_population_tuning(index_path=PROCESSED / "hd_tuning_index.csv"):
    """
    Load all QC-passing HD tuning curves into one population matrix.

    Returns the pooled unit-mean-normalized tuning curves, angular bin centers,
    and metadata for tracing each row back to its subject and unit id.
    """
    if not index_path.exists():
        raise FileNotFoundError("Run python src\\compute_hd_tuning.py first")

    index = pd.read_csv(index_path)
    curves = []
    metadata = []
    bin_centers_rad = None

    for _, row in index.iterrows():
        data = np.load(resolve_existing_path(row["tuning_path"]))
        matrix = data["normalized_rate"]
        included = data["included_qc"].astype(bool)
        unit_ids = data["unit_ids"]

        valid = included & finite_row_mask(matrix, min_fraction=1.0)
        matrix = matrix[valid]
        unit_ids = unit_ids[valid]
        if len(matrix) == 0:
            continue

        curves.append(matrix)
        for unit_id in unit_ids:
            metadata.append({"subject_id": row["subject_id"], "unit_id": int(unit_id)})

        if bin_centers_rad is None:
            bin_centers_rad = data["bin_centers_rad"]

    if not curves:
        raise ValueError("No finite included HD tuning curves were found")

    return np.vstack(curves), np.asarray(bin_centers_rad), pd.DataFrame(metadata)


def build_figure3_matrices(
    regularization=STABLE_REGULARIZATION,
    activation_beta=2.0,
    alpha_floor=1e-4,
    do_double_normalize=True,
    circulant_gain=1.275,
    seed=20260126,
    out_path=PROCESSED / "figure3_abcd_weight_matrices.npz",
):
    """
    Construct the four Figure 3A-D weight matrices from processed tuning data.

    Panel A uses a fixed random neuron order. Panel B uses circular
    center-of-mass order. Panel C is the diagonal-average circulant matrix.
    Panel D adds randomly permuted B-C residuals back to the circulant matrix.
    The paper reports lambda=1e-6 for Figure 3. With the locally extracted
    finite-N target, lambda=1e-6 leaves rare unstable directions in long
    simulations; the default below is the smallest tested stable scale for the
    Figure 3E/F distance diagnostic.
    """
    tuning_raw, angles_rad, metadata = load_population_tuning()
    tuning, preprocess_info = prepare_phi_star_for_inverse(
        tuning_raw,
        theta_axis=-1,
        neuron_axis=0,
        alpha_floor=alpha_floor,
        do_double_normalize=do_double_normalize,
    )
    valid_mask = np.asarray(preprocess_info["valid_neuron_mask"], dtype=bool)
    metadata = metadata.loc[valid_mask].reset_index(drop=True)
    com_angles = circular_center_of_mass_angles(tuning, angles_rad)

    factor_a, factor_b, factor_diagonal = optimized_recurrent_factors(
        tuning,
        regularization=regularization,
        activation_beta=activation_beta,
        enforce_zero_diagonal=True,
    )
    weights = materialize_lowrank_weights(factor_a, factor_b, factor_diagonal, dtype=np.float32)
    diagnostics = figure3_weight_diagnostics(
        phi_raw=tuning_raw,
        phi_safe=tuning,
        weights=weights,
        regularization=regularization,
        activation_beta=activation_beta,
        alpha_floor=alpha_floor,
        preprocess_info=preprocess_info,
    )
    sorted_order = np.argsort(com_angles)

    rng = np.random.default_rng(seed)
    random_order = rng.permutation(len(weights))

    weights_random = weights[np.ix_(random_order, random_order)]
    weights_sorted = weights[np.ix_(sorted_order, sorted_order)]
    weights_circulant, diagonal_means = circulant_from_diagonal_means(weights_sorted)
    weights_noisy, scrambled_residual = scramble_residuals(weights_sorted, weights_circulant, rng)

    # The reference scales the circulant variants to ensure bump formation.
    weights_circulant_scaled = circulant_gain * weights_circulant
    weights_noisy_scaled = circulant_gain * weights_noisy

    metadata_sorted = metadata.iloc[sorted_order].reset_index(drop=True)
    metadata_random = metadata.iloc[random_order].reset_index(drop=True)
    metadata_sorted.to_csv(PROCESSED / "figure3_sorted_units.csv", index=False)
    metadata_random.to_csv(PROCESSED / "figure3_random_units.csv", index=False)

    np.savez_compressed(
        out_path,
        weights_random=weights_random.astype(np.float32),
        weights_sorted=weights_sorted.astype(np.float32),
        weights_circulant=weights_circulant_scaled.astype(np.float32),
        weights_noisy_circulant=weights_noisy_scaled.astype(np.float32),
        sorted_order=sorted_order,
        random_order=random_order,
        center_of_mass_rad=com_angles,
        bin_centers_rad=angles_rad,
        doubly_normalized_tuning=tuning.astype(np.float64),
        phi_star_safe=tuning.astype(np.float64),
        valid_neuron_mask=valid_mask,
        factor_a=factor_a.astype(np.float64),
        factor_b=factor_b.astype(np.float64),
        factor_diagonal=factor_diagonal.astype(np.float64),
        diagonal_means=diagonal_means.astype(np.float32),
        scrambled_residual=scrambled_residual.astype(np.float32),
        regularization=regularization,
        paper_regularization_reference=PAPER_REGULARIZATION,
        activation_beta=activation_beta,
        alpha_floor=alpha_floor,
        do_double_normalize=bool(do_double_normalize),
        phi_star_preprocess_info=json.dumps(compact_preprocess_info(preprocess_info), indent=2),
        figure3_weight_diagnostics=json.dumps(diagnostics, indent=2),
        circulant_gain=circulant_gain,
        weight_formula_version=WEIGHT_FORMULA_VERSION,
        seed=seed,
    )
    diagnostics_path = out_path.with_name(out_path.stem + "_diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return out_path


def regression_kernel_spectrum(phi, regularization):
    """
    Summarize the A2 dual regression kernel used to fit recurrent weights.
    """
    phi = np.asarray(phi, dtype=float)
    n_neurons, n_angles = phi.shape
    kernel = np.einsum("ia,ib->ab", phi, phi, optimize=False) / n_neurons
    kernel += (float(regularization) * n_angles / n_neurons) * np.eye(n_angles)
    kernel = 0.5 * (kernel + kernel.T)
    if not np.isfinite(kernel).all():
        raise ValueError("Regression kernel contains non-finite values")

    inverse = invert_spd_cholesky(kernel)
    kernel_inf_norm = float(np.max(np.sum(np.abs(kernel), axis=1)))
    inverse_inf_norm = float(np.max(np.sum(np.abs(inverse), axis=1)))
    diag = np.diag(kernel)
    row_abs_sum = np.sum(np.abs(kernel), axis=1)
    radius = row_abs_sum - np.abs(diag)
    return {
        "condition_number": float(kernel_inf_norm * inverse_inf_norm),
        "condition_number_kind": "infinity_norm_estimate",
        "spectrum_skipped": True,
        "spectrum_skip_reason": "Avoid NumPy/SciPy LAPACK calls that can hard-crash in this Windows environment",
        "singular_values_min": None,
        "singular_values_max": None,
        "singular_values": [],
        "trace": float(np.trace(kernel)),
        "frobenius_norm": float(np.sqrt(np.sum(kernel * kernel))),
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
        "gershgorin_lower_bound": float(np.min(diag - radius)),
        "gershgorin_upper_bound": float(np.max(diag + radius)),
    }


def leading_tangent_overlap(jacobian, tangent):
    """
    Return the leading eigenvalue and its normalized overlap with the tangent.
    """
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    leading_idx = int(np.argmax(eigenvalues.real))
    vector = eigenvectors[:, leading_idx]
    tangent_norm = float(np.linalg.norm(tangent))
    denom = float(np.linalg.norm(vector)) * tangent_norm
    overlap = float(np.abs(np.vdot(vector, tangent)) / denom) if denom > 0.0 else np.nan
    return {
        "max_real_eigenvalue": float(eigenvalues[leading_idx].real),
        "leading_eigenvalue_imag": float(eigenvalues[leading_idx].imag),
        "leading_tangent_overlap": overlap,
    }


def figure3_weight_diagnostics(
    phi_raw,
    phi_safe,
    weights,
    regularization,
    activation_beta,
    alpha_floor,
    preprocess_info,
    sample_bins=(0, 25, 50, 75),
    c_values=(0.0, 0.1, 0.3, 0.5, 1.0, 2.0),
    max_exact_eig_neurons=300,
):
    """
    Collect numerical diagnostics for the Figure 3 data-derived weight fit.
    """
    phi_raw = np.asarray(phi_raw, dtype=float)
    phi_safe = np.asarray(phi_safe, dtype=float)
    weights = np.asarray(weights, dtype=float)
    x_star = softplus_inverse(phi_safe, beta=activation_beta)
    tangent = 0.5 * (np.roll(x_star, -1, axis=1) - np.roll(x_star, 1, axis=1))

    sample_bins = [int(bin_idx % phi_safe.shape[1]) for bin_idx in sample_bins]
    c_scan = []
    for c_value in c_values:
        residual = fixed_point_residual_from_weights(weights, x_star, phi_safe, inhibition_c=c_value)
        jacobian_rows = []
        if phi_safe.shape[0] <= int(max_exact_eig_neurons):
            for bin_idx in sample_bins:
                jacobian = current_space_jacobian(
                    weights,
                    phi_safe[:, bin_idx],
                    beta=activation_beta,
                    inhibition_c=c_value,
                )
                row = leading_tangent_overlap(jacobian, tangent[:, bin_idx])
                row["theta_bin"] = int(bin_idx)
                jacobian_rows.append(row)
        max_real = max((row["max_real_eigenvalue"] for row in jacobian_rows), default=None)
        max_real_idx = int(np.argmax([row["max_real_eigenvalue"] for row in jacobian_rows])) if jacobian_rows else None
        c_scan.append(
            {
                "c": float(c_value),
                "fixed_point_residual_rms": float(np.sqrt(np.mean(residual * residual))),
                "fixed_point_residual_max_abs": float(np.max(np.abs(residual))),
                "jacobian_sample": jacobian_rows,
                "jacobian_exact_eig_skipped": bool(not jacobian_rows),
                "jacobian_exact_eig_skip_reason": (
                    f"n_neurons={phi_safe.shape[0]} exceeds max_exact_eig_neurons={int(max_exact_eig_neurons)}"
                    if not jacobian_rows
                    else None
                ),
                "jacobian_max_real_sampled": float(max_real) if max_real is not None else None,
                "tangent_overlap_at_max_real": float(jacobian_rows[max_real_idx]["leading_tangent_overlap"])
                if max_real_idx is not None
                else None,
            }
        )

    return {
        "phi_raw_shape": list(phi_raw.shape),
        "phi_safe_shape": list(phi_safe.shape),
        "raw_min": float(np.nanmin(phi_raw)),
        "raw_max": float(np.nanmax(phi_raw)),
        "raw_mean": float(np.nanmean(phi_raw)),
        "raw_exact_zero_count": int(np.sum(phi_raw == 0.0)),
        "raw_below_1e-12_count": int(np.sum(phi_raw < 1e-12)),
        "raw_below_1e-8_count": int(np.sum(phi_raw < 1e-8)),
        "raw_below_1e-6_count": int(np.sum(phi_raw < 1e-6)),
        "raw_below_1e-4_count": int(np.sum(phi_raw < 1e-4)),
        "alpha_floor": float(alpha_floor),
        "phi_safe_min": float(np.min(phi_safe)),
        "phi_safe_max": float(np.max(phi_safe)),
        "phi_safe_mean": float(np.mean(phi_safe)),
        "x_star_min": float(np.min(x_star)),
        "x_star_max": float(np.max(x_star)),
        "x_star_mean": float(np.mean(x_star)),
        "x_star_std": float(np.std(x_star)),
        "raw_regression_kernel": regression_kernel_spectrum(
            np.maximum(phi_raw[np.asarray(preprocess_info["valid_neuron_mask"], dtype=bool)], 0.0),
            regularization,
        ),
        "safe_regression_kernel": regression_kernel_spectrum(phi_safe, regularization),
        "preprocess": compact_preprocess_info(preprocess_info),
        "c_scan": c_scan,
    }


def run_alpha_floor_scan(
    alpha_floors=(1e-6, 1e-5, 1e-4, 1e-3),
    c_values=(0.0, 0.1, 0.3, 0.5, 1.0, 2.0),
    regularization=1e-6,
    activation_beta=2.0,
    out_path=PROCESSED / "figure3_alpha_floor_c_scan.json",
):
    """
    Sweep `alpha_floor` values and save compact stability diagnostics.

    The full population has 1533 neurons, so this scan records conditioning,
    fixed-point residuals, and current-tail statistics for every `(alpha, c)`.
    Exact dense Jacobian spectra and short simulations are intentionally left to
    heavier diagnostics; their status is recorded in each row.
    """
    tuning_raw, _, _ = load_population_tuning()
    rows = []
    for alpha_floor in alpha_floors:
        phi_safe, info = prepare_phi_star_for_inverse(
            tuning_raw,
            theta_axis=-1,
            neuron_axis=0,
            alpha_floor=alpha_floor,
            do_double_normalize=True,
        )
        x_star = softplus_inverse(phi_safe, beta=activation_beta)
        factor_a, factor_b, factor_diagonal = optimized_recurrent_factors(
            phi_safe,
            regularization=regularization,
            activation_beta=activation_beta,
            enforce_zero_diagonal=True,
        )
        weights = materialize_lowrank_weights(factor_a, factor_b, factor_diagonal, dtype=np.float32)
        spectrum = regression_kernel_spectrum(phi_safe, regularization)
        diagnostics = figure3_weight_diagnostics(
            phi_raw=tuning_raw,
            phi_safe=phi_safe,
            weights=weights,
            regularization=regularization,
            activation_beta=activation_beta,
            alpha_floor=alpha_floor,
            preprocess_info=info,
            c_values=c_values,
        )
        for c_row in diagnostics["c_scan"]:
            rows.append(
                {
                    "alpha_floor": float(alpha_floor),
                    "c": float(c_row["c"]),
                    "min_phi": float(np.min(phi_safe)),
                    "min_x": float(np.min(x_star)),
                    "max_abs_x": float(np.max(np.abs(x_star))),
                    "cond": float(spectrum["condition_number"]),
                    "residual_rms": float(c_row["fixed_point_residual_rms"]),
                    "residual_max_abs": float(c_row["fixed_point_residual_max_abs"]),
                    "max_real_eig": c_row["jacobian_max_real_sampled"],
                    "tangent_overlap": c_row["tangent_overlap_at_max_real"],
                    "simulation_status": "not_run_in_lightweight_scan",
                    "jacobian_status": "skipped_full_dense_exact_eig_for_large_n"
                    if c_row["jacobian_exact_eig_skipped"]
                    else "computed",
                    "n_valid_neurons": int(info["n_valid_neurons"]),
                    "n_removed_neurons": int(info["n_removed_neurons"]),
                }
            )

    out_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    return out_path, rows


def figure3_matrix_cache_matches(
    matrix_path,
    regularization=STABLE_REGULARIZATION,
    activation_beta=2.0,
    alpha_floor=1e-4,
    do_double_normalize=True,
    circulant_gain=1.275,
):
    """
    Check whether a cached Figure 3A-D matrix file matches current settings.

    This protects downstream dynamics from silently reusing old weights built
    with a different softplus beta or optimization regularization.
    """
    if not matrix_path.exists():
        return False

    data = np.load(matrix_path)
    return (
        float(data.get("regularization", np.nan)) == float(regularization)
        and float(data.get("activation_beta", np.nan)) == float(activation_beta)
        and float(data.get("alpha_floor", np.nan)) == float(alpha_floor)
        and bool(data.get("do_double_normalize", True)) == bool(do_double_normalize)
        and float(data.get("circulant_gain", np.nan)) == float(circulant_gain)
        and str(data.get("weight_formula_version", "")) == WEIGHT_FORMULA_VERSION
    )


def symmetric_color_limits(matrix, percentile=95.0):
    """
    Choose color limits from robust percentiles of matrix magnitudes.

    The full matrices are saved without clipping. For display, each panel uses
    its own symmetric clipped scale so the weak circulant component is visible
    alongside the much larger disordered residuals.
    """
    vmax = float(np.nanpercentile(np.abs(matrix), percentile))
    return -vmax, vmax


def draw_colorbar(canvas, draw, rect, vmax, font):
    """
    Draw a small vertical diverging colorbar.
    """
    x0, y0, x1, y1 = map(int, rect)
    values = np.linspace(vmax, -vmax, max(2, y1 - y0))[:, None]
    bar = Image.fromarray(diverging_rgb(values, -vmax, vmax), mode="RGB").resize((x1 - x0, y1 - y0))
    canvas.paste(bar, (x0, y0))
    draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=1)
    draw.text((x1 + 5, y0 - 8), f"{vmax:.2g}", font=font, fill=(45, 45, 45))
    draw.text((x1 + 5, (y0 + y1) / 2 - 8), "0", font=font, fill=(45, 45, 45))
    draw.text((x1 + 5, y1 - 10), f"{-vmax:.2g}", font=font, fill=(45, 45, 45))


def plot_figure3_abcd(matrix_path=PROCESSED / "figure3_abcd_weight_matrices.npz"):
    """
    Generate the Figure 3A-D weight-matrix reproduction.

    If the processed matrix file is absent, it is constructed first from the
    pooled DANDI 000939 HD tuning curves.
    """
    if not figure3_matrix_cache_matches(matrix_path):
        matrix_path = build_figure3_matrices(out_path=matrix_path)

    data = np.load(matrix_path)
    matrices = [
        data["weights_random"],
        data["weights_sorted"],
        data["weights_circulant"],
        data["weights_noisy_circulant"],
    ]

    titles = [
        "A  optimized weights, random order",
        "B  optimized weights, COM order",
        "C  diagonal-averaged circulant",
        "D  circulant plus shuffled residuals",
    ]

    panel_w = 660
    panel_h = 650
    heat_size = 520
    canvas = Image.new("RGB", (panel_w * 2, panel_h * 2), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22, bold=True)
    label_font = load_font(16)
    tick_font = load_font(13)

    for idx, (matrix, title) in enumerate(zip(matrices, titles)):
        col = idx % 2
        row = idx // 2
        x = col * panel_w
        y = row * panel_h
        heat, vmax = matrix_heatmap(matrix, size=heat_size, percentile=95.0)
        draw.text((x + 34, y + 22), title, font=title_font, fill=(20, 20, 20))
        heat_x = x + 44
        heat_y = y + 72
        canvas.paste(heat, (heat_x, heat_y))
        draw.rectangle((heat_x, heat_y, heat_x + heat_size, heat_y + heat_size), outline=(30, 30, 30), width=1)
        draw.text((heat_x + 155, heat_y + heat_size + 18), "Presynaptic neuron", font=label_font, fill=(45, 45, 45))
        draw.text((heat_x - 6, heat_y - 24), "Postsynaptic neuron", font=label_font, fill=(45, 45, 45))
        draw_colorbar(
            canvas,
            draw,
            (heat_x + heat_size + 22, heat_y, heat_x + heat_size + 42, heat_y + heat_size),
            vmax,
            tick_font,
        )

    draw_centered_text(
        draw,
        (canvas.width / 2, canvas.height - 20),
        f"N = {matrices[0].shape[0]}, lambda = {float(data['regularization']):g}, beta = {float(data['activation_beta']):g}",
        label_font,
        fill=(45, 45, 45),
    )
    out = FIGURES / "figure3_abcd_reproduction.png"
    canvas.save(out)
    return out, matrix_path


def main():
    """
    Command-line entry point for Figure 3A-D reproduction.
    """
    matrix_path = build_figure3_matrices()
    figure_path, _ = plot_figure3_abcd(matrix_path)
    data = np.load(matrix_path)
    print("Saved:", figure_path)
    print("Saved:", matrix_path)
    print("N:", data["weights_sorted"].shape[0])
    print("regularization:", float(data["regularization"]))
    print("activation_beta:", float(data["activation_beta"]))
    print("alpha_floor:", float(data["alpha_floor"]))
    print("do_double_normalize:", bool(data["do_double_normalize"]))
    print("circulant_gain:", float(data["circulant_gain"]))


if __name__ == "__main__":
    main()
