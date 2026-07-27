from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

from learning.plotting.heading import (
    _dark_phase_intervals,
    _set_pi_y_axis,
    _shade_dark_phase,
    _wrapped_trace_for_axis,
    plot_ensemble_diffusion_variance,
    plot_velocity_gain_curve,
    plot_pi_error_ensemble,
)


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
