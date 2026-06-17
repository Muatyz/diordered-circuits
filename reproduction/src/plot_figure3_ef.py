# python ./reproduction/src/plot_figure3_ef.py
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from plot_figure3_abcd import (
    PROCESSED,
    REPRODUCTION_ROOT,
    STABLE_REGULARIZATION,
    WEIGHT_FORMULA_VERSION,
    build_figure3_matrices,
    figure3_matrix_cache_matches,
)
from render_utils import (
    draw_axes_box,
    draw_centered_text,
    draw_polyline,
    fit_points_to_rect,
    hsv_colors,
    load_font,
    project_3d_to_2d,
)
from utils import (
    circular_resample,
    lowrank_recurrent_drive,
    nearest_manifold_distance,
    optimized_recurrent_factors,
    pca_basis,
    project_onto_basis,
    simulate_rate_network_with_drive,
    softplus,
    softplus_inverse,
)


DEFAULT_MATRIX_PATH = PROCESSED / "figure3_abcd_weight_matrices.npz"
FIGURES = REPRODUCTION_ROOT / "reports/figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def elapsed_text(seconds):
    """
    Format a wall-clock duration for terminal progress messages.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


def load_sorted_target_manifold(matrix_path, activation_beta=2.0, alpha_floor=1e-4):
    """
    Load Figure 3A-D matrices and return the COM-ordered target manifold.

    The optimized weight matrix in panel B is sorted by each neuron's circular
    center of mass, so the target firing-rate manifold must be sorted in the
    same order before it is used as a fixed-point target for panels E and F.
    """
    matrix_path = Path(matrix_path)
    should_validate_default_cache = matrix_path.resolve() == DEFAULT_MATRIX_PATH.resolve()
    if (not matrix_path.exists()) or (
        should_validate_default_cache
            and not figure3_matrix_cache_matches(
                matrix_path,
                activation_beta=activation_beta,
                alpha_floor=alpha_floor,
            )
    ):
        print(f"Missing {matrix_path}; building Figure 3A-D matrices first.", flush=True)
        matrix_path = build_figure3_matrices(
            out_path=matrix_path,
            activation_beta=activation_beta,
            alpha_floor=alpha_floor,
        )

    data = np.load(matrix_path)
    sorted_order = data["sorted_order"]
    target_rate_key = "phi_star_safe" if "phi_star_safe" in data.files else "doubly_normalized_tuning"
    target_rate = data[target_rate_key][sorted_order].T.astype(float)
    activation_beta = float(data["activation_beta"])
    target_current = softplus_inverse(target_rate, beta=activation_beta).astype(float)
    angles_rad = data["bin_centers_rad"].astype(float)
    regularization = float(data["regularization"])
    alpha_floor = float(data["alpha_floor"]) if "alpha_floor" in data.files else float(alpha_floor)
    return target_current, target_rate, angles_rad, regularization, activation_beta, alpha_floor


def make_initial_states(target_current, angle_idx, noise_std=0.05, seed=20260531):
    """
    Create initial conditions near selected points on the target manifold.

    Each trial starts at `x*(theta)` plus small independent Gaussian current
    noise. These are simulated initial conditions, not recorded neural states.
    """
    rng = np.random.default_rng(seed)
    initial = target_current[angle_idx].copy()
    initial += rng.normal(0.0, noise_std, size=initial.shape).astype(np.float32)
    return initial


def set_equal_3d_limits(ax, points, pad_fraction=0.08):
    """
    Give a 3D axis equal data scaling so ring and trajectories are readable.
    """
    points = np.asarray(points, dtype=float)
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.nanmax(maxs - mins)
    radius *= 1.0 + pad_fraction
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def normalized_flow_residual(recurrent_drive, states, inhibition_c=1.0, activation_beta=2.0):
    """
    Measure autonomous flow at candidate manifold points.

    The output is the current-space L2 norm divided by `sqrt(n_neurons)`, the
    same normalization used for the Figure 3F distance axis.
    """
    states = np.asarray(states, dtype=float)
    rates = softplus(states, beta=activation_beta)
    recurrent = np.asarray(recurrent_drive(rates), dtype=float)
    mean_error = np.mean(rates, axis=1, keepdims=True) - 1.0
    dx = -states + recurrent - inhibition_c * mean_error
    return np.sqrt(np.sum(dx * dx, axis=1)) / np.sqrt(states.shape[1])


def compute_figure3_ef_dynamics(
    matrix_path=DEFAULT_MATRIX_PATH,
    out_path=PROCESSED / "figure3_ef_dynamics.npz",
    n_initial=24,
    duration_s=30.0,
    tau_s=0.05,
    dt_s=0.001,
    record_every_s=0.1,
    simulation_manifold_bins=500,
    noise_std=0.05,
    inhibition_c=1.0,
    activation_beta=2.0,
    alpha_floor=1e-4,
    current_clip=None,
    state_stop_abs=1e12,
    seed=20260531,
    force=False,
):
    """
    Simulate optimized-network convergence to the target attractor manifold.

    The saved arrays support panel E's PCA visualization and panel F's nearest
    distance-to-manifold traces. Existing results are reused unless `force` is
    true because this simulation can take noticeably longer than plotting.
    """
    if out_path.exists() and not force:
        cached = np.load(out_path)
        cache_matches = (
            float(cached.get("duration_s", np.nan)) == float(duration_s)
            and float(cached.get("tau_s", np.nan)) == float(tau_s)
            and float(cached.get("dt_s", np.nan)) == float(dt_s)
            and float(cached.get("record_every_s", np.nan)) == float(record_every_s)
            and int(cached.get("simulation_manifold_bins", -1)) == int(simulation_manifold_bins)
            and float(cached.get("noise_std", np.nan)) == float(noise_std)
            and float(cached.get("inhibition_c", np.nan)) == float(inhibition_c)
            and float(cached.get("activation_beta", np.nan)) == float(activation_beta)
            and float(cached.get("alpha_floor", np.nan)) == float(alpha_floor)
            and float(cached.get("regularization", np.nan)) == float(STABLE_REGULARIZATION)
            and str(cached.get("current_clip", "none")) == ("none" if current_clip is None else str(float(current_clip)))
            and str(cached.get("state_stop_abs", "none")) == ("none" if state_stop_abs is None else str(float(state_stop_abs)))
            and str(cached.get("distance_space", "")) == "input_current_full_state"
            and str(cached.get("distance_target", "")) == "target_manifold"
            and str(cached.get("distance_normalization", "")) == "l2_div_sqrt_n_neurons"
            and str(cached.get("weight_formula_version", "")) == WEIGHT_FORMULA_VERSION
            and "stopped_early" in cached.files
            and "actual_duration_s" in cached.files
            and "optimized_l2_distance_to_manifold" in cached.files
            and int(cached.get("seed", -1)) == int(seed)
            and len(cached.get("angle_idx", [])) == int(n_initial)
        )
        if cache_matches:
            print(f"Using cached dynamics: {out_path}", flush=True)
            return out_path
        print(f"Cached dynamics parameters differ; recomputing {out_path}.", flush=True)

    start = time.perf_counter()
    print("Loading Figure 3 target manifold...", flush=True)
    target_current, target_rate, angles_rad, regularization, activation_beta, alpha_floor = load_sorted_target_manifold(
        matrix_path,
        activation_beta=activation_beta,
        alpha_floor=alpha_floor,
    )
    angles_dense = np.linspace(0.0, 2.0 * np.pi, simulation_manifold_bins, endpoint=False)
    angle_idx = np.linspace(0, len(angles_rad), n_initial, endpoint=False, dtype=int)

    print("Building low-rank optimized recurrent drive...", flush=True)
    factors_start = time.perf_counter()
    factor_a, factor_b, diagonal = optimized_recurrent_factors(
        target_rate.T,
        regularization=regularization,
        activation_beta=activation_beta,
        enforce_zero_diagonal=True,
    )
    recurrent_drive = lowrank_recurrent_drive(factor_a, factor_b, diagonal)
    print(f"Built recurrent drive in {elapsed_text(time.perf_counter() - factors_start)}.", flush=True)

    target_current_dense = circular_resample(target_current, simulation_manifold_bins, axis=0).astype(np.float32)
    target_flow_residual = normalized_flow_residual(
        recurrent_drive,
        target_current,
        inhibition_c=inhibition_c,
        activation_beta=activation_beta,
    )
    initial_states = make_initial_states(target_current, angle_idx, noise_std=noise_std, seed=seed)
    print(
        "Autonomous residual flow on target manifold: "
        f"median={np.median(target_flow_residual):.3e}, max={np.max(target_flow_residual):.3e} / sqrt(N).",
        flush=True,
    )

    print(
        "Simulating optimized recurrent dynamics "
        f"({n_initial} initial states, {duration_s:g}s biological time, tau={tau_s:g}s, dt={dt_s:g}s)...",
        flush=True,
    )
    times, trajectory = simulate_rate_network_with_drive(
        recurrent_drive,
        initial_states,
        tau_s=tau_s,
        dt_s=dt_s,
        duration_s=duration_s,
        inhibition_c=inhibition_c,
        activation_beta=activation_beta,
        record_every_s=record_every_s,
        current_clip=current_clip,
        stop_abs=state_stop_abs,
        progress_label="Figure 3E/F dynamics",
        progress_interval_wall_s=10.0,
    )
    finite_trajectory = bool(np.isfinite(trajectory).all())
    finite_values = trajectory[np.isfinite(trajectory)]
    max_abs_trajectory = float(np.max(np.abs(finite_values))) if finite_values.size else np.inf
    actual_duration_s = float(times[-1]) if len(times) else 0.0
    stopped_early = actual_duration_s < float(duration_s) - 0.5 * float(record_every_s)
    if not finite_trajectory:
        stop_reason = "non_finite_state"
    elif stopped_early and state_stop_abs is not None and max_abs_trajectory >= 0.99 * float(state_stop_abs):
        stop_reason = "state_stop_abs_exceeded"
    elif stopped_early:
        stop_reason = "early_stop"
    else:
        stop_reason = "completed"
    if not finite_trajectory:
        print(
            "Figure 3E/F optimized dynamics produced non-finite states; "
            "saving the partial trajectory for diagnostic plotting.",
            flush=True,
        )
    if stopped_early:
        print(
            f"Figure 3E/F dynamics stopped early at t={actual_duration_s:.3f}s "
            f"of planned {duration_s:g}s ({stop_reason}).",
            flush=True,
        )

    print("Projecting target manifold and trajectories onto three PCs...", flush=True)
    pca_mean, basis = pca_basis(target_current_dense, n_components=3)
    manifold_pc = project_onto_basis(target_current_dense, pca_mean, basis).astype(np.float32)
    trajectory_pc = np.empty((*trajectory.shape[:2], 3), dtype=np.float32)
    distance = np.empty(trajectory.shape[:2], dtype=np.float32)
    l2_distance = np.empty(trajectory.shape[:2], dtype=np.float32)
    nearest_idx = np.empty(trajectory.shape[:2], dtype=np.int64)

    for t_idx in range(len(times)):
        trajectory_pc[t_idx] = project_onto_basis(trajectory[t_idx], pca_mean, basis)

    print("Computing nearest-manifold distances in one batch...", flush=True)
    distance_start = time.perf_counter()
    flat_distance, flat_nearest_idx, flat_l2_distance = nearest_manifold_distance(
        trajectory.reshape(-1, trajectory.shape[-1]),
        target_current_dense,
        return_l2=True,
    )
    distance[:] = flat_distance.reshape(trajectory.shape[:2])
    nearest_idx[:] = flat_nearest_idx.reshape(trajectory.shape[:2])
    l2_distance[:] = flat_l2_distance.reshape(trajectory.shape[:2])

    print(f"Computed manifold distances in {elapsed_text(time.perf_counter() - distance_start)}.", flush=True)

    np.savez_compressed(
        out_path,
        times=times,
        angle_idx=angle_idx,
        bin_centers_rad=angles_rad,
        target_current=target_current,
        target_current_dense=target_current_dense,
        target_rate=target_rate,
        dense_angles_rad=angles_dense,
        initial_states=initial_states,
        optimized_trajectory=trajectory,
        pca_mean=pca_mean,
        pca_basis=basis,
        manifold_pc=manifold_pc,
        optimized_pc=trajectory_pc,
        optimized_distance_to_manifold=distance,
        optimized_l2_distance_to_manifold=l2_distance,
        optimized_nearest_angle_idx=nearest_idx,
        target_flow_residual=target_flow_residual.astype(np.float32),
        dynamics_finite=finite_trajectory,
        max_abs_trajectory=max_abs_trajectory,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        actual_duration_s=actual_duration_s,
        distance_space="input_current_full_state",
        distance_target="target_manifold",
        distance_normalization="l2_div_sqrt_n_neurons",
        weight_formula_version=WEIGHT_FORMULA_VERSION,
        initial_condition_source="target_manifold_points_plus_iid_gaussian_current_noise",
        simulation_manifold_bins=simulation_manifold_bins,
        duration_s=duration_s,
        tau_s=tau_s,
        dt_s=dt_s,
        record_every_s=record_every_s,
        noise_std=noise_std,
        inhibition_c=inhibition_c,
        activation_beta=activation_beta,
        alpha_floor=alpha_floor,
        current_clip="none" if current_clip is None else float(current_clip),
        state_stop_abs="none" if state_stop_abs is None else float(state_stop_abs),
        planned_duration_s=duration_s,
        regularization=regularization,
        seed=seed,
    )
    print(f"Saved dynamics: {out_path} ({elapsed_text(time.perf_counter() - start)})", flush=True)
    return out_path


def plot_figure3_ef(dynamics_path=PROCESSED / "figure3_ef_dynamics.npz"):
    """
    Plot Figure 3E-F from the cached or newly computed dynamics.
    """
    if not dynamics_path.exists():
        dynamics_path = compute_figure3_ef_dynamics(out_path=dynamics_path)

    data = np.load(dynamics_path)
    angles = data["bin_centers_rad"]
    angle_idx = data["angle_idx"]
    colors = hsv_colors((angles[angle_idx] % (2 * np.pi)) / (2 * np.pi))
    initial_pc = project_onto_basis(data["initial_states"], data["pca_mean"], data["pca_basis"])
    final_pc = data["optimized_pc"][-1]

    canvas = Image.new("RGB", (1420, 680), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24, bold=True)
    label_font = load_font(16)
    small_font = load_font(13)

    left_rect = (70, 88, 665, 575)
    right_rect = (790, 95, 1345, 555)
    left_size = (left_rect[2] - left_rect[0], left_rect[3] - left_rect[1])
    right_size = (right_rect[2] - right_rect[0], right_rect[3] - right_rect[1])
    left_panel = Image.new("RGB", left_size, "white")
    left_draw = ImageDraw.Draw(left_panel)
    right_panel = Image.new("RGB", right_size, "white")
    right_draw = ImageDraw.Draw(right_panel)
    manifold_pc = data["manifold_pc"]
    fit_source = np.vstack([manifold_pc, initial_pc])
    fit_finite = fit_source[np.all(np.isfinite(fit_source), axis=1)]
    pc_min = np.min(fit_finite, axis=0)
    pc_max = np.max(fit_finite, axis=0)
    pc_center = 0.5 * (pc_min + pc_max)
    pc_span = np.maximum(pc_max - pc_min, 1e-9)
    pc_span[:] = np.max(pc_span) * 1.12
    pc_min = pc_center - 0.5 * pc_span
    pc_max = pc_center + 0.5 * pc_span
    box_corners = np.array(
        [
            [pc_min[0], pc_min[1], pc_min[2]],
            [pc_max[0], pc_min[1], pc_min[2]],
            [pc_min[0], pc_max[1], pc_min[2]],
            [pc_max[0], pc_max[1], pc_min[2]],
            [pc_min[0], pc_min[1], pc_max[2]],
            [pc_max[0], pc_min[1], pc_max[2]],
            [pc_min[0], pc_max[1], pc_max[2]],
            [pc_max[0], pc_max[1], pc_max[2]],
        ],
        dtype=float,
    )
    axis_points = np.array(
        [
            [pc_min[0], pc_min[1], pc_min[2]],
            [pc_max[0], pc_min[1], pc_min[2]],
            [pc_min[0], pc_max[1], pc_min[2]],
            [pc_min[0], pc_min[1], pc_max[2]],
        ],
        dtype=float,
    )
    all_pc = np.vstack([manifold_pc, initial_pc, data["optimized_pc"].reshape(-1, 3), box_corners, axis_points])
    projected_all, projected_depth = project_3d_to_2d(all_pc)
    projected_fit, _ = project_3d_to_2d(np.vstack([fit_source, box_corners]))
    mapped_all = fit_points_to_rect(
        projected_all,
        (0, 0, left_size[0] - 1, left_size[1] - 1),
        equal_scale=True,
        reference_points=projected_fit,
    )

    offset = 0
    manifold_2d = mapped_all[offset : offset + len(manifold_pc)]
    offset += len(manifold_pc)
    initial_2d = mapped_all[offset : offset + len(initial_pc)]
    offset += len(initial_pc)
    trajectory_count = data["optimized_pc"].shape[0] * data["optimized_pc"].shape[1]
    trajectory_2d = mapped_all[offset : offset + trajectory_count].reshape(
        data["optimized_pc"].shape[0],
        data["optimized_pc"].shape[1],
        2,
    )
    offset += trajectory_count
    box_2d = mapped_all[offset : offset + len(box_corners)]
    box_depth = projected_depth[offset : offset + len(box_corners)]
    offset += len(box_corners)
    axis_2d = mapped_all[offset : offset + len(axis_points)]

    def keep_near_panel(points, panel_size, margin_fraction=0.5):
        """
        Drop projected points far outside the panel before drawing.

        Divergent trajectories can map to extremely large pixel coordinates.
        Clipping them to NaN preserves the target-manifold scale and prevents
        Pillow from rasterizing huge off-canvas line segments across the panel.
        """
        points = np.asarray(points, dtype=float).copy()
        margin_x = panel_size[0] * float(margin_fraction)
        margin_y = panel_size[1] * float(margin_fraction)
        near = (
            np.isfinite(points[:, 0])
            & np.isfinite(points[:, 1])
            & (points[:, 0] >= -margin_x)
            & (points[:, 0] <= panel_size[0] + margin_x)
            & (points[:, 1] >= -margin_y)
            & (points[:, 1] <= panel_size[1] + margin_y)
        )
        points[~near] = np.nan
        return points

    def draw_3d_frame(panel_draw):
        """
        Draw a wireframe PC1-PC2-PC3 box behind the target manifold.
        """
        edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        median_depth = float(np.median(box_depth))
        for a, b in edges:
            if np.all(np.isfinite(box_2d[[a, b]])):
                color = (218, 218, 218) if 0.5 * (box_depth[a] + box_depth[b]) < median_depth else (176, 176, 176)
                panel_draw.line((*box_2d[a], *box_2d[b]), fill=color, width=1)

        origin = axis_2d[0]
        labels = [("PC1", axis_2d[1]), ("PC2", axis_2d[2]), ("PC3", axis_2d[3])]
        if np.all(np.isfinite(origin)):
            for label, endpoint in labels:
                if np.all(np.isfinite(endpoint)):
                    panel_draw.line((*origin, *endpoint), fill=(95, 95, 95), width=2)
                    direction = endpoint - origin
                    norm = max(float(np.sqrt(np.sum(direction * direction))), 1e-12)
                    label_xy = endpoint + 12.0 * direction / norm
                    panel_draw.text((label_xy[0] - 10, label_xy[1] - 8), label, font=small_font, fill=(45, 45, 45))

    draw.text((left_rect[0], 30), "E  optimized dynamics in target-current PC1-PC2-PC3 space", font=title_font, fill=(20, 20, 20))
    left_draw.rectangle((0, 0, left_size[0] - 1, left_size[1] - 1), outline=(60, 60, 60), width=1)
    draw_3d_frame(left_draw)
    draw_polyline(left_draw, manifold_2d, fill=(35, 35, 35), width=2)
    for point in manifold_2d[:: max(1, len(manifold_2d) // 100)]:
        if np.all(np.isfinite(point)):
            x, y = point
            left_draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(70, 70, 70))

    for trial in range(len(angle_idx)):
        draw_polyline(left_draw, keep_near_panel(trajectory_2d[:, trial], left_size), fill=colors[trial], width=2)
        point = initial_2d[trial]
        if np.all(np.isfinite(point)):
            x, y = point
            left_draw.line((x - 5, y - 5, x + 5, y + 5), fill=(0, 0, 0), width=2)
            left_draw.line((x - 5, y + 5, x + 5, y - 5), fill=(0, 0, 0), width=2)
        point = trajectory_2d[-1, trial]
        if np.all(np.isfinite(point)):
            x, y = point
            left_draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(0, 0, 0))

    canvas.paste(left_panel, (left_rect[0], left_rect[1]))
    draw.text((left_rect[0], left_rect[3] + 14), "x: initial states    dot: final finite states", font=small_font, fill=(50, 50, 50))

    colorbar_x0 = left_rect[0] + 260
    colorbar_y0 = left_rect[3] + 15
    colorbar_w = 180
    colorbar_h = 10
    gradient = Image.new("RGB", (colorbar_w, colorbar_h), "white")
    gradient_pixels = gradient.load()
    gradient_colors = hsv_colors(np.linspace(0.0, 1.0, colorbar_w, endpoint=True))
    for gx, color in enumerate(gradient_colors):
        for gy in range(colorbar_h):
            gradient_pixels[gx, gy] = tuple(color)
    canvas.paste(gradient, (colorbar_x0, colorbar_y0))
    draw.rectangle(
        (colorbar_x0, colorbar_y0, colorbar_x0 + colorbar_w, colorbar_y0 + colorbar_h),
        outline=(80, 80, 80),
        width=1,
    )
    draw.text((colorbar_x0 - 18, colorbar_y0 - 3), "0", font=small_font, fill=(50, 50, 50))
    draw.text((colorbar_x0 + colorbar_w + 6, colorbar_y0 - 3), "2pi", font=small_font, fill=(50, 50, 50))

    draw.text((right_rect[0], 30), "F  distance to nearest target manifold point", font=title_font, fill=(20, 20, 20))
    draw_axes_box(right_draw, (0, 0, right_size[0] - 1, right_size[1] - 1), label_x=None, label_y=None, font=label_font)
    times = data["times"].astype(float)
    distance = data["optimized_distance_to_manifold"].astype(float)
    finite_dist = distance[np.isfinite(distance) & (distance > 0)]
    if finite_dist.size:
        y_min = min(1e-3, float(np.nanpercentile(finite_dist, 1)))
        y_max = max(1e-1, float(np.nanpercentile(finite_dist, 99)))
    else:
        y_min, y_max = 1e-4, 1.0
    y_min = max(y_min / 1.5, 1e-8)
    y_max = max(y_max * 1.5, y_min * 10)
    log_min = np.log10(y_min)
    log_max = np.log10(y_max)
    x0, y0, x1, y1 = (0, 0, right_size[0] - 1, right_size[1] - 1)

    def map_xy(t, y):
        x = x0 + (t - times[0]) / max(times[-1] - times[0], 1e-12) * (x1 - x0)
        yy = y1 - (np.log10(y) - log_min) / max(log_max - log_min, 1e-12) * (y1 - y0)
        return x, yy

    exponent_min = int(np.floor(log_min))
    exponent_max = int(np.ceil(log_max))
    exponent_step = max(1, int(np.ceil((exponent_max - exponent_min + 1) / 9)))
    exponents = list(range(exponent_min, exponent_max + 1, exponent_step))
    if -3 not in exponents and exponent_min <= -3 <= exponent_max:
        exponents.append(-3)
    for exponent in sorted(set(exponents)):
        y_value = 10.0 ** exponent
        if y_min <= y_value <= y_max:
            _, yy = map_xy(times[0], y_value)
            right_draw.line((x0, yy, x1, yy), fill=(225, 225, 225), width=1)
            draw.text((right_rect[0] - 54, right_rect[1] + yy - 8), f"1e{exponent}", font=small_font, fill=(70, 70, 70))
    for t in np.linspace(times[0], times[-1], 4):
        xx, _ = map_xy(t, y_min)
        right_draw.line((xx, y1, xx, y1 + 5), fill=(50, 50, 50), width=1)
        tick_text = f"{t:.2f}".rstrip("0").rstrip(".")
        draw_centered_text(draw, (right_rect[0] + xx, right_rect[1] + y1 + 18), tick_text, small_font, fill=(70, 70, 70))

    if y_min <= 1e-3 <= y_max:
        p0 = map_xy(times[0], 1e-3)
        p1 = map_xy(times[-1], 1e-3)
        right_draw.line((*p0, *p1), fill=(60, 60, 60), width=1)
        right_draw.text((x1 - 52, p0[1] - 18), "1e-3", font=small_font, fill=(60, 60, 60))

    for trial, color in enumerate(colors):
        y = distance[:, trial]
        valid = np.isfinite(y) & (y > 0)
        points = np.full((len(times), 2), np.nan, dtype=float)
        points[valid] = [map_xy(t, yy) for t, yy in zip(times[valid], y[valid])]
        draw_polyline(right_draw, points, fill=color, width=2)

    canvas.paste(right_panel, (right_rect[0], right_rect[1]))
    draw.text((right_rect[0] - 42, right_rect[1] - 4), "distance / sqrt(N)", font=label_font, fill=(40, 40, 40))
    draw_centered_text(draw, ((right_rect[0] + right_rect[2]) / 2, right_rect[3] + 48), "time (s)", label_font, fill=(40, 40, 40))

    finite_flag = bool(data["dynamics_finite"]) if "dynamics_finite" in data.files else bool(np.isfinite(data["optimized_trajectory"]).all())
    planned_duration = float(data["planned_duration_s"]) if "planned_duration_s" in data.files else float(data["duration_s"])
    actual_duration = float(data["actual_duration_s"]) if "actual_duration_s" in data.files else float(times[-1])
    stopped_early = bool(data["stopped_early"]) if "stopped_early" in data.files else actual_duration < planned_duration
    stop_reason = str(data["stop_reason"]) if "stop_reason" in data.files else ("early_stop" if stopped_early else "completed")
    final = distance[-1]
    final_finite = final[np.isfinite(final)]
    final_text = "final max finite distance: n/a" if final_finite.size == 0 else f"final max finite distance: {np.max(final_finite):.3e}"
    if stopped_early:
        status_text = f"stopped early at {actual_duration:.3g}s / {planned_duration:.3g}s ({stop_reason})"
    elif finite_flag:
        status_text = "completed planned simulation"
    else:
        status_text = "non-finite trajectory detected"
    status_fill = (150, 30, 30) if stopped_early or not finite_flag else (40, 40, 40)
    draw.text((right_rect[0], right_rect[3] + 76), f"{status_text}; {final_text}", font=small_font, fill=status_fill)

    out = FIGURES / "figure3_ef_reproduction.png"
    canvas.save(out)
    return out, dynamics_path


def main():
    """
    Command-line entry point for Figure 3E-F reproduction.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute dynamics even if the cache exists")
    args = parser.parse_args()

    dynamics_path = compute_figure3_ef_dynamics(force=args.force)
    figure_path, _ = plot_figure3_ef(dynamics_path)
    data = np.load(dynamics_path)
    final_max = float(np.max(data["optimized_distance_to_manifold"][-1]))
    print("Saved:", figure_path)
    print("Saved:", dynamics_path)
    if "stopped_early" in data.files and bool(data["stopped_early"]):
        print(
            "Stopped early:",
            f"t={float(data['actual_duration_s']):.3f}s",
            "reason=" + str(data["stop_reason"]),
        )
    print("Final max distance / sqrt(N):", f"{final_max:.3e}")


if __name__ == "__main__":
    main()
