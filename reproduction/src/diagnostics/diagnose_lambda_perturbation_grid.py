try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()

import json
import time

import numpy as np

from plot_figure3_abcd import PROCESSED, REPRODUCTION_ROOT
from utils import (
    circular_resample,
    lowrank_recurrent_drive,
    nearest_circular_manifold_distance,
    optimized_recurrent_factors,
    simulate_rate_network_with_drive,
    softplus_inverse,
)


REPORTS = REPRODUCTION_ROOT / "reports"
FIGURES = REPORTS / "figures"
OUT_JSON = REPORTS / "figure3_lambda_perturbation_grid.json"
OUT_CSV = REPORTS / "figure3_lambda_perturbation_grid.csv"
OUT_FIGURE = FIGURES / "figure3_lambda_perturbation_grid.png"

REGULARIZATIONS = (1e-6, 3e-6, 1e-5, 3e-5, 4e-5, 5e-5, 7e-5, 1e-4, 3e-4, 1e-3)
NOISE_STDS = (0.05, 0.1, 0.2, 0.5, 1.0)
SUMMARY_TIMES_S = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)


def elapsed_text(seconds):
    """
    Format a wall-clock duration for progress messages.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


def load_figure3_target(matrix_path):
    """
    Load the same sorted firing-rate and current manifolds used by Figure 3E/F.
    """
    data = np.load(matrix_path)
    sorted_order = data["sorted_order"]
    rate_key = "phi_star_safe" if "phi_star_safe" in data.files else "doubly_normalized_tuning"
    target_rate = data[rate_key][sorted_order].astype(float)
    beta = float(data["activation_beta"])
    target_current = softplus_inverse(target_rate, beta=beta).T
    return target_rate, target_current, beta


def selected_time_indices(times, requested_times):
    """
    Return unique trajectory indices nearest to requested biological times.
    """
    times = np.asarray(times, dtype=float)
    indices = [int(np.argmin(np.abs(times - requested))) for requested in requested_times]
    return np.asarray(sorted(set(indices)), dtype=int)


def summarize_distance_times(times, distance):
    """
    Summarize nearest-manifold distances at recorded checkpoints.
    """
    summaries = []
    for index, time_s in enumerate(times):
        values = distance[index]
        finite = values[np.isfinite(values)]
        summaries.append(
            {
                "time_s": float(time_s),
                "median": float(np.median(finite)) if finite.size else float("nan"),
                "max": float(np.max(finite)) if finite.size else float("nan"),
                "finite_count": int(finite.size),
            }
        )
    return summaries


def summary_value_near(summaries, requested_time_s, tolerance_s=0.051):
    """
    Return a median distance near a requested time, or NaN after early stops.
    """
    candidates = [
        entry for entry in summaries if abs(entry["time_s"] - requested_time_s) <= tolerance_s
    ]
    if not candidates:
        return float("nan")
    nearest = min(candidates, key=lambda entry: abs(entry["time_s"] - requested_time_s))
    return nearest["median"]


def run_case(
    target_current,
    target_current_dense,
    drive,
    beta,
    regularization,
    noise_std,
    base_noise,
    duration_s=30.0,
):
    """
    Simulate one lambda/noise pair and measure distance to the target manifold.
    """
    start = time.perf_counter()
    angle_idx = np.linspace(0, len(target_current), len(base_noise), endpoint=False, dtype=int)
    initial = target_current[angle_idx] + float(noise_std) * base_noise
    times, trajectory = simulate_rate_network_with_drive(
        drive,
        initial,
        tau_s=0.05,
        dt_s=0.001,
        duration_s=duration_s,
        inhibition_c=1.0,
        activation_beta=beta,
        record_every_s=0.1,
        current_clip=None,
        stop_abs=1e12,
        progress_label=f"lambda={regularization:g}, noise={noise_std:g}",
        progress_interval_wall_s=30.0,
    )

    time_indices = selected_time_indices(times, SUMMARY_TIMES_S)
    selected_times = times[time_indices]
    selected_states = trajectory[time_indices].reshape(-1, trajectory.shape[-1])
    flat_distance, flat_coordinate = nearest_circular_manifold_distance(
        selected_states,
        target_current_dense,
    )
    distance = flat_distance.reshape(len(time_indices), len(base_noise))
    coordinate = flat_coordinate.reshape(len(time_indices), len(base_noise))
    summaries = summarize_distance_times(selected_times, distance)
    median_trace = np.asarray([entry["median"] for entry in summaries], dtype=float)
    minimum_index = int(np.nanargmin(median_trace)) if np.any(np.isfinite(median_trace)) else 0

    finite_values = trajectory[np.isfinite(trajectory)]
    actual_duration_s = float(times[-1])
    stopped_early = actual_duration_s < duration_s - 0.05
    final_coordinate = coordinate[-1]
    initial_coordinate = coordinate[0]
    n_manifold = len(target_current_dense)
    angular_shift = np.minimum(
        np.mod(final_coordinate - initial_coordinate, n_manifold),
        np.mod(initial_coordinate - final_coordinate, n_manifold),
    )
    angular_shift_deg = angular_shift * 360.0 / n_manifold

    return {
        "regularization": float(regularization),
        "noise_std": float(noise_std),
        "n_initial": int(len(base_noise)),
        "planned_duration_s": float(duration_s),
        "actual_duration_s": actual_duration_s,
        "stopped_early": bool(stopped_early),
        "finite_trajectory": bool(np.isfinite(trajectory).all()),
        "max_abs_state": float(np.max(np.abs(finite_values))) if finite_values.size else float("inf"),
        "distance_times": summaries,
        "initial_distance_median": summaries[0]["median"],
        "initial_distance_max": summaries[0]["max"],
        "minimum_distance_median": summaries[minimum_index]["median"],
        "minimum_distance_time_s": summaries[minimum_index]["time_s"],
        "distance_1s_median": summary_value_near(summaries, 1.0),
        "final_distance_median": summaries[-1]["median"],
        "final_distance_max": summaries[-1]["max"],
        "final_angular_drift_median_deg": float(np.nanmedian(angular_shift_deg)),
        "final_angular_drift_max_deg": float(np.nanmax(angular_shift_deg)),
        "elapsed_wall_s": float(time.perf_counter() - start),
    }


def write_csv(cases):
    """
    Write one compact row per lambda/noise pair for spreadsheet inspection.
    """
    columns = [
        "regularization",
        "noise_std",
        "actual_duration_s",
        "stopped_early",
        "finite_trajectory",
        "initial_distance_median",
        "minimum_distance_median",
        "minimum_distance_time_s",
        "distance_1s_median",
        "final_distance_median",
        "final_distance_max",
        "final_angular_drift_median_deg",
        "max_abs_state",
    ]
    lines = [",".join(columns)]
    for case in cases:
        lines.append(",".join(str(case[column]) for column in columns))
    OUT_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(cases):
    """
    Plot final distance and representative time courses across the parameter grid.
    """
    from PIL import Image, ImageDraw

    from render_utils import draw_polyline, hsv_colors, load_font

    canvas = Image.new("RGB", (1500, 650), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22, bold=True)
    label_font = load_font(16)
    small_font = load_font(13)
    colors = [tuple(color) for color in hsv_colors(np.linspace(0.0, 0.9, len(REGULARIZATIONS)))]
    left = (90, 75, 700, 540)
    right = (830, 75, 1440, 540)

    def map_log_panel(x, y, rect, x_limits, y_limits, log_x=False):
        """
        Map numerical coordinates into a rectangular plotting panel.
        """
        x0, y0, x1, y1 = rect
        x_value = np.log10(x) if log_x else x
        x_min = np.log10(x_limits[0]) if log_x else x_limits[0]
        x_max = np.log10(x_limits[1]) if log_x else x_limits[1]
        y_value = np.log10(y)
        y_min, y_max = np.log10(y_limits[0]), np.log10(y_limits[1])
        px = x0 + (x_value - x_min) / (x_max - x_min) * (x1 - x0)
        py = y1 - (y_value - y_min) / (y_max - y_min) * (y1 - y0)
        return px, py

    def draw_panel_frame(rect, title, x_label):
        """
        Draw a shared logarithmic-distance panel frame.
        """
        draw.rectangle(rect, outline=(50, 50, 50), width=1)
        draw.text((rect[0], 25), title, font=title_font, fill=(20, 20, 20))
        draw.text(((rect[0] + rect[2]) / 2 - 60, rect[3] + 55), x_label, font=label_font, fill=(40, 40, 40))
        draw.text((rect[0] - 78, rect[1] - 28), "distance / sqrt(N)", font=small_font, fill=(40, 40, 40))
        for exponent in range(-3, 1):
            value = 10.0 ** exponent
            if y_limits[0] <= value <= y_limits[1]:
                _, py = map_log_panel(1.0, value, rect, (1.0, 10.0), y_limits)
                draw.line((rect[0], py, rect[2], py), fill=(225, 225, 225), width=1)
                draw.text((rect[0] - 50, py - 7), f"1e{exponent}", font=small_font, fill=(70, 70, 70))

    y_limits = (1e-3, 1.0)
    draw_panel_frame(left, "Final distance; failed runs clipped at 1", "regularization lambda")
    draw_panel_frame(right, "Time course at initial noise std = 0.2; clipped at 1", "time (s)")

    for index, noise_std in enumerate(NOISE_STDS):
        selected = [case for case in cases if case["noise_std"] == noise_std]
        points = np.asarray(
            [
                map_log_panel(
                    case["regularization"],
                    np.clip(case["final_distance_median"], *y_limits),
                    left,
                    (min(REGULARIZATIONS), max(REGULARIZATIONS)),
                    y_limits,
                    log_x=True,
                )
                for case in selected
            ]
        )
        draw_polyline(draw, points, fill=colors[index], width=3)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=colors[index])
        draw.text(
            (left[0] + 15 + index * 112, left[3] + 15),
            f"noise={noise_std:g}",
            font=small_font,
            fill=colors[index],
        )

    representative_noise = 0.2
    selected_cases = [case for case in cases if case["noise_std"] == representative_noise]
    for index, case in enumerate(selected_cases):
        points = np.asarray(
            [
                map_log_panel(
                    entry["time_s"],
                    np.clip(entry["median"], *y_limits),
                    right,
                    (0.0, 30.0),
                    y_limits,
                    log_x=False,
                )
                for entry in case["distance_times"]
            ]
        )
        draw_polyline(draw, points, fill=colors[index], width=3)
        draw.text(
            (right[0] + 15 + (index % 4) * 140, right[3] + 15 + (index // 4) * 18),
            f"lambda={case['regularization']:g}",
            font=small_font,
            fill=colors[index],
        )

    for rect, x_ticks, log_x in [
        (left, REGULARIZATIONS, True),
        (right, (0.0, 5.0, 10.0, 20.0, 30.0), False),
    ]:
        for value in x_ticks:
            px, _ = map_log_panel(
                value,
                y_limits[0],
                rect,
                (min(REGULARIZATIONS), max(REGULARIZATIONS)) if log_x else (0.0, 30.0),
                y_limits,
                log_x=log_x,
            )
            draw.line((px, rect[3], px, rect[3] + 5), fill=(50, 50, 50), width=1)
            label = f"{value:g}"
            draw.text((px - 18, rect[3] + 4), label, font=small_font, fill=(60, 60, 60))

    canvas.save(OUT_FIGURE)


def main():
    """
    Run a full lambda-by-initial-perturbation scan for Figure 3 dynamics.
    """
    total_start = time.perf_counter()
    matrix_path = PROCESSED / "figure3_abcd_weight_matrices.npz"
    target_rate, target_current, beta = load_figure3_target(matrix_path)
    target_current_dense = circular_resample(target_current, 500, axis=0)
    rng = np.random.default_rng(20260618)
    n_initial = 8
    base_noise = rng.normal(size=(n_initial, target_current.shape[1]))

    cases = []
    if OUT_JSON.exists():
        previous = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        cases = previous.get("cases", [])
    completed_keys = {
        (float(case["regularization"]), float(case["noise_std"])) for case in cases
    }

    for regularization in REGULARIZATIONS:
        factor_a, factor_b, diagonal = optimized_recurrent_factors(
            target_rate,
            regularization=regularization,
            activation_beta=beta,
            enforce_zero_diagonal=True,
        )
        drive = lowrank_recurrent_drive(factor_a, factor_b, diagonal)
        for noise_std in NOISE_STDS:
            key = (float(regularization), float(noise_std))
            if key in completed_keys:
                print(f"Skipping completed lambda={regularization:g}, noise={noise_std:g}", flush=True)
                continue
            case = run_case(
                target_current,
                target_current_dense,
                drive,
                beta,
                regularization,
                noise_std,
                base_noise,
            )
            cases.append(case)
            completed_keys.add(key)
            print(
                f"  initial={case['initial_distance_median']:.3g} "
                f"min={case['minimum_distance_median']:.3g} "
                f"final={case['final_distance_median']:.3g} "
                f"stopped={case['stopped_early']} wall={elapsed_text(case['elapsed_wall_s'])}",
                flush=True,
            )

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    cases.sort(key=lambda case: (case["regularization"], case["noise_std"]))
    report = {
        "status": "diagnostic_only_no_default_parameter_change",
        "target_source": str(matrix_path),
        "distance_definition": "nearest point on closed piecewise-linear target-current manifold",
        "distance_normalization": "l2_div_sqrt_n_neurons",
        "activation_beta": beta,
        "uniform_inhibition_c": 1.0,
        "tau_s": 0.05,
        "dt_s": 0.001,
        "regularizations": list(REGULARIZATIONS),
        "noise_stds": list(NOISE_STDS),
        "n_initial_per_case": n_initial,
        "seed": 20260618,
        "cases": cases,
        "elapsed_wall_s": float(time.perf_counter() - total_start),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(cases)
    plot_summary(cases)
    print(f"Saved: {OUT_JSON}")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_FIGURE}")
    print(f"Total wall time: {elapsed_text(time.perf_counter() - total_start)}")


if __name__ == "__main__":
    main()
