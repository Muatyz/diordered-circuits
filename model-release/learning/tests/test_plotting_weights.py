from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from learning.common.angles import circular_difference, make_theta_hd_pref
from learning.plotting.weights import (
    _linear_time_edges,
    _set_pi_ticks_for_extent,
    _training_time_scale,
    compute_receptive_field_offset_profile_history,
    plot_weight_matrices_side_by_side,
    plot_training_absolute_learning_error,
    plot_weight_snapshot_pi_development,
)


def test_set_pi_ticks_for_extent_marks_angular_axes() -> None:
    fig, axis = plt.subplots()
    try:
        _set_pi_ticks_for_extent(axis, (-np.pi, np.pi, 0.0, 3.0))

        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "-pi",
            "-pi/2",
            "0",
            "pi/2",
            "pi",
        ]
    finally:
        plt.close(fig)


def test_receptive_field_profile_history_averages_by_target_source_offset() -> None:
    theta_base = make_theta_hd_pref(6)
    theta_paired = np.repeat(theta_base, 2)
    offset_matrix = circular_difference(theta_paired[:, None], theta_paired[None, :])
    base_weight = np.cos(offset_matrix) + 0.25 * np.cos(2.0 * offset_matrix)
    weight_history = np.stack([base_weight, 2.0 * base_weight], axis=0)

    offset_grid, profile_history = compute_receptive_field_offset_profile_history(
        weight_history=weight_history,
        theta_target_pref=theta_paired,
        theta_source_pref=theta_paired,
    )

    expected_profile = np.cos(offset_grid) + 0.25 * np.cos(2.0 * offset_grid)
    assert profile_history.shape == (theta_base.size, 2)
    np.testing.assert_allclose(profile_history[:, 0], expected_profile, atol=1e-12)
    np.testing.assert_allclose(profile_history[:, 1], 2.0 * expected_profile, atol=1e-12)


def test_profile_time_edges_do_not_log_expand_initial_snapshot() -> None:
    time = np.asarray([0.01, 400.0, 800.0, 1200.0, 4000.0], dtype=float)

    edges = _linear_time_edges(time)
    scale, unit = _training_time_scale(time)

    np.testing.assert_allclose(edges[:3], [0.0, 200.005, 600.0])
    assert edges[-1] == time[-1]
    assert unit == "h"
    assert scale == 1.0 / 3600.0


def test_sorted_weight_panels_use_neuron_id_axes(tmp_path, monkeypatch) -> None:
    saved_figure: dict[str, plt.Figure] = {}

    def capture_figure(figure: plt.Figure, *_args, **_kwargs) -> None:
        saved_figure["figure"] = figure

    monkeypatch.setattr(plt.Figure, "savefig", capture_figure)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    plot_weight_matrices_side_by_side(
        w_hd_to_hd=np.arange(16, dtype=float).reshape(4, 4),
        w_hr_to_hd=np.arange(24, dtype=float).reshape(4, 6),
        path=tmp_path / "weights.png",
    )

    figure = saved_figure["figure"]
    hd_axis, hr_axis = figure.axes[:2]
    assert hd_axis.get_xlabel() == "source HD neuron ID (COM-sorted)"
    assert hd_axis.get_ylabel() == "target HD neuron ID (COM-sorted)"
    assert hr_axis.get_xlabel() == "source HR neuron ID (L/R, COM-sorted)"
    assert hr_axis.get_ylabel() == "target HD neuron ID (COM-sorted)"
    np.testing.assert_allclose(hd_axis.images[0].get_extent(), (-0.5, 3.5, -0.5, 3.5))
    np.testing.assert_allclose(hr_axis.images[0].get_extent(), (-0.5, 5.5, -0.5, 3.5))


def test_snapshot_pi_development_plot_accepts_zero_and_log_times(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, plt.Figure] = {}
    monkeypatch.setattr(
        plt.Figure,
        "savefig",
        lambda figure, *args, **kwargs: captured.update(figure=figure),
    )
    figure_path = tmp_path / "snapshot_pi.png"
    plot_weight_snapshot_pi_development(
        snapshot_time=np.asarray([0.0, 10.0, 100.0]),
        commanded_velocity=np.asarray([-0.5, 0.5]),
        time_averaged_abs_pi_error=np.asarray(
            [[1.0, 1.1], [0.4, 0.5], [0.6, 0.7]]
        ),
        aggregate_time_averaged_abs_pi_error=np.asarray([1.05, 0.45, 0.65]),
        aggregate_rms_pi_error=np.asarray([1.1, 0.5, 0.7]),
        minimum_pva_strength=np.asarray(
            [[0.1, 0.2], [0.7, 0.8], [0.6, 0.7]]
        ),
        best_snapshot_time=10.0,
        aggregate_rms_velocity_bias=np.asarray([0.08, 0.02, 0.04]),
        effective_weight_growth_hd_to_hd=np.asarray([1.0, 2.0, 2.5]),
        effective_weight_growth_hr_to_hd=np.asarray([1.0, 2.5, 3.2]),
        effective_weight_norm_hr_to_hd_over_hd_to_hd=np.asarray(
            [1.0, 1.25, 1.28]
        ),
        path=figure_path,
    )
    assert len(captured["figure"].axes) == 4


def test_training_absolute_learning_error_plot_writes_diagnostics_figure(
    tmp_path,
) -> None:
    figure_path = tmp_path / "diagnostics" / "training_error.png"
    plot_training_absolute_learning_error(
        time=np.asarray([0.0, 800.0, 1600.0]),
        mean_absolute_error_spikes_per_s=np.asarray([20.0, 8.0, 4.0]),
        window_duration=10.0,
        path=figure_path,
    )
    assert figure_path.exists()
