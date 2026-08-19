from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
import numpy as np

from learning.plotting.activity import (
    plot_activity_heatmap,
    plot_activity_tuning_slices,
    plot_com_aligned_hd_tuning_population,
    plot_hd_tuning_stage_comparison,
    plot_hd_tuning_settling_diagnostics,
    plot_heterogeneous_visual_input_profiles,
    plot_single_neuron_hd_tuning_curves,
)
from learning.analysis.make_vafidis_figures import _select_bump_maintenance_slice_times


def test_heterogeneous_profile_subplots_are_seeded_and_use_minus_pi_to_pi(
    monkeypatch,
    tmp_path: Path,
) -> None:
    n_neurons = 10
    n_angles = 16
    theta_grid = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    tuning_profiles = np.asarray(
        [
            1.0 + 0.1 * neuron_index + 0.5 * np.cos(theta_grid - neuron_index)
            for neuron_index in range(n_neurons)
        ]
    )
    saved_figures: list[Figure] = []

    def capture_figure(figure: Figure, path, *args, **kwargs) -> None:
        saved_figures.append(figure)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    keyword_arguments = {
        "tuning_profiles": tuning_profiles,
        "path": tmp_path / "heterogeneous_visual_input_profiles.png",
        "sample_count": 5,
        "seed": 123,
        "amplitude": 4.0,
        "baseline": 5.0,
        "light_excitation": 4.0,
        "proximal_scale": 1.0 / 3.0,
        "theta_hd_pref": np.linspace(-np.pi, np.pi, n_neurons, endpoint=False),
    }

    first_indices = plot_heterogeneous_visual_input_profiles(**keyword_arguments)
    second_indices = plot_heterogeneous_visual_input_profiles(**keyword_arguments)

    np.testing.assert_array_equal(first_indices, second_indices)
    assert first_indices.size == 5
    assert np.unique(first_indices).size == first_indices.size
    assert len(saved_figures) == 2
    first_profile_line = saved_figures[0].axes[0].lines[0]
    theta_plot = np.asarray(first_profile_line.get_xdata())
    current_plot = np.asarray(first_profile_line.get_ydata())
    assert np.isclose(theta_plot[0], -np.pi)
    assert np.isclose(theta_plot[-1], np.pi)
    assert np.isclose(current_plot[0], current_plot[-1])
    preferred_line = saved_figures[0].axes[0].lines[1]
    sampled_preference = keyword_arguments["theta_hd_pref"][first_indices[0]]
    assert preferred_line.get_linestyle() == "--"
    assert np.allclose(preferred_line.get_xdata(), sampled_preference)


def test_activity_heatmap_marks_dark_phase(monkeypatch, tmp_path: Path) -> None:
    saved_figures: list[Figure] = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved_figures.append(figure))

    plot_activity_heatmap(
        r_hd_history=np.ones((4, 4)),
        time=np.asarray([0.0, 0.1, 0.2, 0.3]),
        path=tmp_path / "activity.png",
        theta_hd_pref=np.linspace(-np.pi, np.pi, 4, endpoint=False),
        phase_id=np.asarray([0.0, 1.0, 1.0, 2.0]),
    )

    axis = saved_figures[0].axes[0]
    assert len(axis.images) == 2
    assert np.allclose(axis.images[1].get_extent()[:2], (0.05, 0.25))


def test_activity_heatmap_preserves_irregular_empirical_theta_and_redecodes_overlay(
    monkeypatch,
    tmp_path: Path,
) -> None:
    saved_figures: list[Figure] = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved_figures.append(figure))
    theta = np.asarray([-2.8, -0.2, 0.05, 2.4])
    rates = np.asarray([[0.0, 1.0, 3.0, 0.0], [0.0, 1.0, 3.0, 0.0]])

    plot_activity_heatmap(
        r_hd_history=rates,
        time=np.asarray([0.0, 1.0]),
        path=tmp_path / "irregular.png",
        theta_hd_pref=theta,
        theta_hd_decoded=np.asarray([2.0, 2.0]),
        decode_theta_hd_pref=theta,
    )

    axis = saved_figures[0].axes[0]
    assert len(axis.collections) >= 1
    pva_line = next(line for line in axis.lines if line.get_label() == "PVA decode")
    expected = np.angle(np.sum(rates[0] * np.exp(1j * theta)))
    np.testing.assert_allclose(pva_line.get_ydata(), expected)
    assert not np.allclose(pva_line.get_ydata(), 2.0)


def test_bump_slice_times_omit_freshly_initialized_state() -> None:
    history = {
        "time": np.arange(8, dtype=float) * 0.1,
        "phase_id": np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    }

    selected = _select_bump_maintenance_slice_times(history)

    assert selected is not None
    assert np.isclose(selected[0], 0.2)
    assert 0.0 not in selected
    assert np.isclose(selected[-1], 0.7)


def test_tuning_slice_title_identifies_frozen_weight_protocol(monkeypatch, tmp_path: Path) -> None:
    saved_figures: list[Figure] = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved_figures.append(figure))

    plot_activity_tuning_slices(
        r_hd_history=np.ones((3, 4)),
        time=np.asarray([0.2, 0.3, 0.4]),
        theta_hd_pref=np.linspace(-np.pi, np.pi, 4, endpoint=False),
        path=tmp_path / "slices.png",
        time_context="frozen-weight protocol",
    )

    assert "frozen-weight protocol t=0.20-0.40 s" in saved_figures[0].axes[0].get_title()


def test_heterogeneous_profile_plot_caps_sample_count_at_population_size(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    selected_indices = plot_heterogeneous_visual_input_profiles(
        tuning_profiles=np.ones((3, 8)),
        path=tmp_path / "profiles.png",
        sample_count=20,
        seed=8,
        amplitude=1.0,
        baseline=0.0,
        light_excitation=0.0,
        proximal_scale=1.0,
    )

    np.testing.assert_array_equal(selected_indices, np.arange(3))


def test_heterogeneous_profile_standard_sample_uses_four_by_four_grid(
    monkeypatch,
    tmp_path: Path,
) -> None:
    saved_figures: list[Figure] = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, *args, **kwargs: saved_figures.append(figure),
    )

    selected_indices = plot_heterogeneous_visual_input_profiles(
        tuning_profiles=np.ones((20, 32)),
        path=tmp_path / "profiles_4x4.png",
        sample_count=16,
        seed=8,
        amplitude=1.0,
        baseline=0.0,
        light_excitation=0.0,
        proximal_scale=1.0,
    )

    assert selected_indices.size == 16
    assert len(saved_figures[0].axes) == 16
    subplot_spec = saved_figures[0].axes[-1].get_subplotspec()
    assert subplot_spec.rowspan.start == 3
    assert subplot_spec.colspan.start == 3


def test_single_neuron_tuning_curve_marks_empirical_com(monkeypatch, tmp_path: Path) -> None:
    saved_figures: list[Figure] = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved_figures.append(figure))
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    rates = np.column_stack([1.0 + np.cos(theta - offset) for offset in theta[:4]])

    selected = plot_single_neuron_hd_tuning_curves(
        theta_true=theta,
        r_hd_by_heading=rates,
        preferred_direction=theta[:4],
        path=tmp_path / "single_neuron.png",
        sample_count=2,
        seed=4,
    )

    assert selected.size == 2
    preference_line = saved_figures[0].axes[0].lines[1]
    assert preference_line.get_linestyle() == "--"
    assert np.allclose(preference_line.get_xdata(), theta[selected[0]])


def test_com_aligned_population_plot_draws_every_neuron_and_thick_black_mean(
    monkeypatch,
    tmp_path: Path,
) -> None:
    saved_figures: list[Figure] = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved_figures.append(figure))
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    aligned_rates = np.asarray(
        [1.0 + 0.2 * np.cos(theta) + 0.03 * neuron_index for neuron_index in range(5)]
    )

    plot_com_aligned_hd_tuning_population(
        theta_aligned=theta,
        r_hd_peak_normalized_com_aligned=aligned_rates,
        population_mean=np.mean(aligned_rates, axis=0),
        population_std=np.std(aligned_rates, axis=0),
        path=tmp_path / "com_aligned.png",
    )

    tuning_axis, std_axis = saved_figures[0].axes
    assert len(tuning_axis.lines) == aligned_rates.shape[0] + 1
    mean_line = tuning_axis.lines[-1]
    assert mean_line.get_color() == "#111111"
    assert mean_line.get_linewidth() >= 2.5
    assert "N=5" in mean_line.get_label()
    assert "one simulated mouse" in tuning_axis.get_title()
    assert len(std_axis.lines) == 1
    assert std_axis.lines[0].get_color() == "#111111"


def test_tuning_stage_comparison_labels_visual_only_and_post_training(
    monkeypatch,
    tmp_path: Path,
) -> None:
    saved_figures: list[Figure] = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved_figures.append(figure))
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)

    plot_hd_tuning_stage_comparison(
        theta_aligned=theta,
        visual_only_mean=0.5 + 0.2 * np.cos(theta),
        visual_only_std=0.1 + 0.01 * np.cos(theta),
        post_training_mean=0.4 + 0.4 * np.cos(theta),
        post_training_std=0.2 + 0.02 * np.cos(theta),
        path=tmp_path / "stages.png",
    )

    mean_axis, std_axis = saved_figures[0].axes
    assert {line.get_label() for line in mean_axis.lines} == {
        "visual-only mean",
        "post-training steady mean",
    }
    assert {line.get_label() for line in std_axis.lines} == {
        "visual-only std",
        "post-training steady std",
    }


def test_tuning_settling_diagnostics_marks_nonconverged_headings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    saved_figures: list[Figure] = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved_figures.append(figure))
    theta = np.linspace(-np.pi, np.pi, 4, endpoint=False)

    plot_hd_tuning_settling_diagnostics(
        theta_true=theta,
        actual_settle_duration=np.asarray([1.4, 1.6, 3.0, 1.4]),
        final_window_max_rate_change=np.asarray([0.001, 0.0015, 0.02, 0.0008]),
        settle_converged=np.asarray([1.0, 1.0, 0.0, 1.0]),
        convergence_tolerance=0.002,
        path=tmp_path / "settling.png",
    )

    time_axis, residual_axis = saved_figures[0].axes
    assert "3/4" in time_axis.get_title()
    assert residual_axis.get_yscale() == "log"
    point_colors = time_axis.collections[0].get_facecolors()
    assert np.unique(point_colors, axis=0).shape[0] == 2
