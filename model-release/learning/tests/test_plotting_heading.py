from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

from learning.plotting.heading import (
    _dark_phase_intervals,
    _set_pi_y_axis,
    _shade_dark_phase,
    _wrapped_trace_for_axis,
    plot_bump_attractor_pva_trajectories,
    plot_numerical_convergence_diagnostics,
    plot_constant_velocity_pi_error_grid,
    plot_diffusion_noise_sweep,
    plot_ensemble_diffusion_variance,
    plot_velocity_gain_curve,
    plot_pi_error_ensemble,
)


def test_bump_trajectory_plot_separates_cue_transfer_and_endpoint_map(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, matplotlib.figure.Figure] = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda figure, *args, **kwargs: captured.update(figure=figure),
    )
    theta_initial = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    cue_time = np.asarray([-1.0, -0.5, 0.0])
    time = np.asarray([0.0, 0.5, 1.0])
    cue_trace = np.column_stack(
        [theta_initial + 0.1, theta_initial + 0.05, theta_initial]
    )
    final_theta = theta_initial - 0.2 * np.sin(2.0 * theta_initial)
    darkness_trace = np.column_stack(
        [theta_initial, 0.5 * (theta_initial + final_theta), final_theta]
    )

    plot_bump_attractor_pva_trajectories(
        time=time,
        theta_initial=theta_initial,
        theta_pva=darkness_trace,
        cue_time=cue_time,
        cue_theta_pva=cue_trace,
        path=tmp_path / "trajectory.png",
    )

    figure = captured["figure"]
    trajectory_axis = figure.axes[0]
    endpoint_axis = figure.axes[2]
    trajectory_lines = [
        line
        for line in trajectory_axis.lines
        if np.asarray(line.get_xdata()).size == time.size
    ]
    assert trajectory_lines
    np.testing.assert_allclose(
        trajectory_lines[0].get_xdata(),
        time,
    )
    assert "darkness" in trajectory_axis.get_title()
    np.testing.assert_allclose(trajectory_axis.get_ylim(), [-180.0, 180.0])
    np.testing.assert_allclose(endpoint_axis.get_ylim(), [-180.0, 180.0])
    cue_axis = figure.axes[1]
    assert "initial cue angle" in cue_axis.get_xlabel()
    assert "relaxed angle" in cue_axis.get_ylabel()
    legend_labels = [text.get_text() for text in endpoint_axis.get_legend().texts]
    assert "endpoint map" in legend_labels
    assert "stable FP" in legend_labels
    assert "unstable FP" in legend_labels
    assert "initial cue angle" in endpoint_axis.get_xlabel()
    assert not any("cue onset" in label for label in legend_labels)


def test_pva_trajectory_plot_uses_three_large_high_resolution_panels(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def capture_figure(figure, *args, **kwargs) -> None:
        captured["figure"] = figure
        captured["dpi"] = kwargs.get("dpi")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)
    theta_initial = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    cue_time = np.asarray([-1.0, 0.0])
    time = np.asarray([0.0, 0.5, 1.0])
    cue_trace = np.column_stack([theta_initial + 0.05, theta_initial])
    darkness_trace = np.column_stack(
        [theta_initial, theta_initial + 0.01, theta_initial + 0.02]
    )

    plot_bump_attractor_pva_trajectories(
        time=time,
        theta_initial=theta_initial,
        theta_pva=darkness_trace,
        cue_time=cue_time,
        cue_theta_pva=cue_trace,
        path=tmp_path / "pva_trajectory.png",
    )

    figure = captured["figure"]
    assert isinstance(figure, matplotlib.figure.Figure)
    assert len(figure.axes) == 3
    assert figure.get_figwidth() == 15.6
    assert figure.get_figheight() == 4.8
    assert captured["dpi"] == 300
    assert figure.axes[0].get_ylabel().startswith("PVA")
    assert "initial cue angle" in figure.axes[2].get_xlabel()


def test_numerical_convergence_plot_has_three_scientific_panels(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, matplotlib.figure.Figure] = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda figure, *args, **kwargs: captured.update(figure=figure),
    )
    time = np.asarray([0.0, 0.01, 0.02])
    dt = np.asarray([0.0005, 0.00025, 0.0005, 0.00025])
    method = np.asarray(
        ["forward_euler", "forward_euler", "exact_linear", "exact_linear"]
    )
    trace = np.outer(np.asarray([2.0, 1.0, 1.5, 0.7]), time)

    plot_numerical_convergence_diagnostics(
        time=time,
        dt=dt,
        integration_method=method,
        heading_error=trace,
        rate_rms_error=0.01 * trace,
        max_abs_heading_error=np.max(np.abs(trace), axis=1),
        max_rate_rms_error=np.max(np.abs(0.01 * trace), axis=1),
        convergence_passed=np.asarray([False, True, False, True]),
        path=tmp_path / "numerics.png",
    )

    assert len(captured["figure"].axes) == 3
    assert "phase error" in captured["figure"].axes[0].get_title()
    assert "step-size convergence" in captured["figure"].axes[2].get_title()


def test_dark_phase_intervals_use_midpoints_between_phase_samples() -> None:
    time = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    phase_id = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 2.0])

    assert _dark_phase_intervals(time, phase_id) == [(0.15000000000000002, 0.45)]


def test_shade_dark_phase_adds_data_coordinate_image() -> None:
    time = np.asarray([0.0, 0.1, 0.2, 0.3])
    phase_id = np.asarray([0.0, 1.0, 1.0, 2.0])
    fig, axis = plt.subplots()
    try:
        axis.plot(time, [0.0, 1.0, 2.0, 3.0])
        axis.set_ylim(-1.0, 4.0)

        _shade_dark_phase(axis, time=time, phase_id=phase_id)

        assert len(axis.images) == 1
        image = axis.images[0]
        assert np.allclose(image.get_extent(), (0.05, 0.25, -1.0, 4.0))
    finally:
        plt.close(fig)


def test_wrapped_trace_stays_on_pi_interval() -> None:
    trace = np.asarray([0.0, 0.2, 2.0 * np.pi + 0.3])

    wrapped_trace = _wrapped_trace_for_axis(trace)

    assert np.allclose(wrapped_trace, [0.0, 0.2, 0.3])
    assert np.nanmin(wrapped_trace) >= -np.pi
    assert np.nanmax(wrapped_trace) <= np.pi


def test_wrapped_trace_breaks_at_pi_boundary() -> None:
    wrapped_trace = _wrapped_trace_for_axis(np.asarray([3.0, 3.2, 3.3]))

    assert np.isclose(wrapped_trace[0], 3.0)
    assert np.isnan(wrapped_trace[1])
    assert -np.pi <= wrapped_trace[2] <= np.pi


def test_set_pi_y_axis_uses_radian_ticks() -> None:
    fig, axis = plt.subplots()
    try:
        _set_pi_y_axis(axis)

        assert np.allclose(axis.get_ylim(), (-np.pi, np.pi))
        assert [tick.get_text() for tick in axis.get_yticklabels()] == [
            "-pi",
            "-pi/2",
            "0",
            "pi/2",
            "pi",
        ]
    finally:
        plt.close(fig)


def test_plot_ensemble_diffusion_variance_smoke(monkeypatch, tmp_path) -> None:
    captured: dict[str, matplotlib.figure.Figure] = {}

    def capture_figure(figure, *args, **kwargs) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)
    time = np.array([0.0, 0.5, 1.0])

    plot_ensemble_diffusion_variance(
        time=time,
        displacement_mean=0.1 * time,
        displacement_variance=0.2 * time,
        diffusion_coefficient=0.2,
        systematic_drift_velocity=0.1,
        anomalous_diffusion_fit=0.2 * time**1.2,
        anomalous_diffusion_exponent=1.2,
        generalized_diffusion_coefficient=0.2,
        anomalous_diffusion_log_r_squared=0.98,
        anomalous_diffusion_fit_start_time=0.5,
        anomalous_diffusion_fit_end_time=1.0,
        n_trials=120,
        path=tmp_path / "ensemble.png",
    )

    assert captured["figure"]._suptitle.get_text().endswith("(n=120 trials)")


def test_plot_diffusion_noise_sweep_uses_deg2_per_second(monkeypatch, tmp_path) -> None:
    captured: dict[str, matplotlib.figure.Figure] = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda figure, *args, **kwargs: captured.update(figure=figure),
    )
    diffusion = np.asarray([0.0, 0.01, 0.04])
    plot_diffusion_noise_sweep(
        noise_std=np.asarray([0.0, 0.1, 0.2]),
        diffusion_coefficient=diffusion,
        path=tmp_path / "diffusion_sweep.png",
    )

    axis = captured["figure"].axes[0]
    np.testing.assert_allclose(
        axis.lines[0].get_ydata(),
        np.rad2deg(1.0) ** 2 * diffusion,
    )


def test_nonlinear_velocity_gain_connects_samples_and_hides_fit(monkeypatch, tmp_path) -> None:
    captured: dict[str, matplotlib.figure.Figure] = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda figure, *args, **kwargs: captured.update(figure=figure),
    )

    plot_velocity_gain_curve(
        commanded_velocity=np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
        decoded_velocity=np.asarray([-2.0, -1.0, 0.0, 0.0, 0.0]),
        path=tmp_path / "gain.png",
    )

    axis = captured["figure"].axes[0]
    raw_line = next(line for line in axis.lines if line.get_label() == "darkness PVA")
    assert np.allclose(raw_line.get_xdata(), [-2.0, -1.0, 0.0, 1.0, 2.0])
    assert np.isclose(raw_line.get_markersize(), 3.5)
    assert not any(line.get_label() == "darkness PVA linear fit" for line in axis.lines)
    assert "nonlinear: fit hidden" in axis.texts[0].get_text()


def test_pi_error_ensemble_plot_labels_pva_trial_count(monkeypatch, tmp_path) -> None:
    captured: dict[str, matplotlib.figure.Figure] = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda figure, *args, **kwargs: captured.update(figure=figure),
    )
    time = np.linspace(0.0, 1.0, 5)
    plot_pi_error_ensemble(
        time=time,
        pi_error_mean=0.1 * time,
        pi_error_sem=np.full(time.shape, 0.02),
        systematic_drift_velocity=0.1,
        drift_intercept=0.0,
        n_trials=24,
        path=tmp_path / "ensemble.png",
    )
    axis = captured["figure"].axes[0]
    assert "PVA/COM" in axis.get_title()
    assert "n=24" in axis.get_title()


def test_constant_velocity_pi_error_grid_uses_four_panels(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, matplotlib.figure.Figure] = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda figure, *args, **kwargs: captured.update(figure=figure),
    )
    time = np.linspace(0.0, 2.0, 5)
    velocities = np.deg2rad(np.asarray([-75.0, -30.0, 30.0, 75.0]))
    decoded = 1.1 * velocities
    pi_error = np.outer(decoded - velocities, time)

    plot_constant_velocity_pi_error_grid(
        time=time,
        pi_error=pi_error,
        commanded_velocity=velocities,
        decoded_velocity=decoded,
        phase_id=np.asarray([0.0, 1.0, 1.0, 1.0, 2.0]),
        path=tmp_path / "constant_pi.png",
    )

    figure = captured["figure"]
    assert len(figure.axes) == 4
    assert [len(axis.lines) for axis in figure.axes] == [2, 2, 2, 2]
    assert all("gain 1.100" in axis.get_title() for axis in figure.axes)
    assert "-75 deg/s" in figure.axes[0].get_title()
    assert "+75 deg/s" in figure.axes[3].get_title()
