from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import numpy as np

from learning.config.load_config import load_yaml
from learning.experiments.run_attractor_robustness import (
    _aggregate_sweep_grid_rows,
    _create_timestamped_report_dir,
    _metric_performance_score_grid,
    _plot_cross_mouse_tuning_moments,
    _plot_sweep_metric_meshgrid,
    _write_clark_figure4_summary,
    _write_cross_mouse_tuning_summaries,
)
from learning.io.save_load import load_npz, save_npz


def test_visual_current_noise_std_config_enables_noise_only() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "analysis"
        / "visual_current_noise_std_comparison.yaml"
    )
    config = load_yaml(config_path)

    assert config["sweeps"]["visual_noise"]["enabled"] is True
    assert config["sweeps"]["neuron_count"]["enabled"] is False
    assert config["sweeps"]["noise_by_neuron_count"]["enabled"] is False
    assert config["runtime"]["seed_offsets"] == [0]
    assert "visual_current_noise_std_comparison" in config["output"]["directory"]
    assert max(config["sweeps"]["visual_noise"]["stds"]) >= 2.0


def test_heterogeneous_30_mouse_hyper_preset_has_requested_hierarchy() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = (
        project_root
        / "configs"
        / "analysis"
        / "heterogeneous_30_mouse_hyper.yaml"
    )
    config = load_yaml(config_path)
    base_experiment = load_yaml(project_root / config["experiment"]["config"])

    assert base_experiment["visual"]["profile"] == "heterogeneous_gaussian_process"
    assert config["sweeps"]["neuron_count"]["counts"] == [360]
    assert config["runtime"]["seed_offsets"] == list(range(30))
    assert config["plots"]["neuron_count_metrics"]["enabled"] is False
    assert config["plots"]["cross_mouse_tuning"] == {
        "enabled": True,
        "minimum_converged_fraction": None,
        "normalizations": ["per_neuron_peak"],
    }
    assert config["plots"]["clark_figure4"] == {
        "enabled": True,
        "minimum_converged_fraction": None,
        "uniformity_alpha": 0.05,
        "example_mouse_indices": [1, 10, 20, 30],
        "subset_sizes": [5, 10, 20, 40, 80, 120, 240, 360],
        "random_seed": 20251026,
    }


def test_cross_mouse_tuning_summary_preserves_mouse_hierarchy(tmp_path: Path) -> None:
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    rows = []
    for mouse_index, seed in enumerate([42, 43]):
        run_dir = tmp_path / f"mouse_{seed}"
        peak_mean = 0.4 + 0.1 * mouse_index + 0.3 * np.cos(theta)
        peak_std = 0.1 + 0.02 * mouse_index + 0.02 * np.cos(theta)
        save_npz(
            run_dir / "hd_tuning_com_aligned.npz",
            theta_aligned=theta,
            r_hd_peak_normalized_com_aligned_mean=peak_mean,
            r_hd_peak_normalized_com_aligned_std=peak_std,
            r_hd_unit_mean_com_aligned_mean=2.0 * peak_mean,
            r_hd_unit_mean_com_aligned_std=2.0 * peak_std,
        )
        rows.append({"run_dir": str(run_dir), "n_theta": 240, "seed": seed})

    summary = _write_cross_mouse_tuning_summaries(rows=rows, output_dir=tmp_path)

    assert {row["normalization"] for row in summary} == {
        "per_neuron_peak",
        "unit_mean_clark",
    }
    peak_archive = load_npz(
        tmp_path / "heterogeneous_cross_mouse_tuning_n240_per_neuron_peak.npz"
    )
    assert peak_archive["mouse_mean_curves"].shape == (2, theta.size)
    np.testing.assert_allclose(
        peak_archive["cross_mouse_mean_curve"],
        np.mean(peak_archive["mouse_mean_curves"], axis=0),
    )

    peak_only_dir = tmp_path / "peak_only"
    rows[0]["hd_tuning_curve_converged_fraction"] = 0.5
    peak_only_summary = _write_cross_mouse_tuning_summaries(
        rows=rows,
        output_dir=peak_only_dir,
        normalizations=["per_neuron_peak"],
        minimum_converged_fraction=None,
    )
    assert [row["normalization"] for row in peak_only_summary] == ["per_neuron_peak"]
    peak_only_archive = load_npz(
        peak_only_dir / "heterogeneous_cross_mouse_tuning_n240_per_neuron_peak.npz"
    )
    assert peak_only_archive["mouse_mean_curves"].shape[0] == 2
    assert not (
        peak_only_dir / "heterogeneous_cross_mouse_tuning_n240_unit_mean_clark.npz"
    ).exists()


def test_cross_mouse_tuning_plot_uses_gray_mice_and_thick_black_average(
    tmp_path: Path,
    monkeypatch,
) -> None:
    saved_figures: list[matplotlib.figure.Figure] = []

    def capture_figure(figure, *_args, **_kwargs) -> None:
        saved_figures.append(figure)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    mouse_mean_curves = np.vstack([0.5 + 0.2 * np.cos(theta), 0.6 + 0.1 * np.cos(theta)])
    mouse_std_curves = np.vstack([0.1 + 0.02 * np.cos(theta), 0.12 + 0.01 * np.cos(theta)])

    _plot_cross_mouse_tuning_moments(
        theta_aligned=theta,
        mouse_mean_curves=mouse_mean_curves,
        mouse_std_curves=mouse_std_curves,
        path=tmp_path / "cross_mouse.png",
        normalization_label="per_neuron_peak",
        n_theta=120,
    )

    assert len(saved_figures) == 1
    for axis in saved_figures[0].axes:
        mouse_lines = axis.lines[:-1]
        average_line = axis.lines[-1]
        assert len(mouse_lines) == 2
        assert all(line.get_color() == "#c7c7c7" for line in mouse_lines)
        assert all(line.get_linewidth() == 0.9 for line in mouse_lines)
        assert average_line.get_color() == "#111111"
        assert average_line.get_linewidth() == 3.0


def test_clark_figure4_summary_uses_unaligned_nested_tuning_subsets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def save_figure_without_rendering(figure, path, **_kwargs) -> None:
        del figure
        Path(path).touch()

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        save_figure_without_rendering,
    )
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    rows = []
    for mouse_index, seed in enumerate([42, 43, 44, 45], start=1):
        run_dir = tmp_path / f"mouse_{seed}"
        preferred_bins = np.arange(theta.size)
        preferred_directions = theta[preferred_bins]
        raw_unit_mean_tuning = np.vstack(
            [
                1.0 + 0.5 * np.cos(theta - preferred_direction)
                for preferred_direction in preferred_directions
            ]
        )
        alignment_shifts = theta.size // 2 - preferred_bins
        aligned_tuning = np.vstack(
            [
                np.roll(curve, int(shift))
                for curve, shift in zip(
                    raw_unit_mean_tuning,
                    alignment_shifts,
                    strict=True,
                )
            ]
        )
        save_npz(
            run_dir / "hd_tuning_com_aligned.npz",
            theta_aligned=theta,
            r_hd_unit_mean_com_aligned=aligned_tuning,
            com_alignment_shift_bins=alignment_shifts,
            r_hd_tuning_valid_mask=np.ones(theta.size, dtype=bool),
            empirical_preferred_direction=preferred_directions,
        )
        rows.append(
            {
                "run_dir": str(run_dir),
                "n_theta": 8,
                "seed": seed,
                "hd_tuning_curve_converged_fraction": 1.0,
            }
        )

    summary = _write_clark_figure4_summary(
        rows=rows,
        output_dir=tmp_path,
        example_mouse_indices=[1, 4],
        subset_sizes=[2, 8],
        random_seed=20251026,
        minimum_converged_fraction=None,
    )

    statistics = load_npz(tmp_path / "heterogeneous_clark_figure4_statistics.npz")
    assert statistics["empirical_com_by_mouse"].shape == (4, 8)
    assert statistics["correlation_matrices"].shape == (2, 2, 8, 8)
    assert np.all(statistics["empirical_com_by_mouse"] >= -np.pi)
    assert np.all(statistics["empirical_com_by_mouse"] < np.pi)
    assert np.all(statistics["relative_circulant_error"][:, -1] < 1e-12)
    assert np.all(statistics["kuiper_p_values"] > 0.5)
    assert np.count_nonzero(statistics["kuiper_bh_adjusted_p_values"] < 0.05) == 0
    assert Path(str(summary["figure_path"])).exists()
    assert Path(str(summary["circulant_error_table_path"])).exists()
    assert Path(str(summary["com_uniformity_table_path"])).exists()


def test_noise_by_neuron_count_config_enables_joint_grid_only() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "analysis"
        / "noise_by_neuron_count_comparison.yaml"
    )
    config = load_yaml(config_path)

    assert config["sweeps"]["visual_noise"]["enabled"] is False
    assert config["sweeps"]["neuron_count"]["enabled"] is False
    assert config["sweeps"]["noise_by_neuron_count"]["enabled"] is True
    assert config["runtime"]["seed_offsets"] == [0]
    assert 32 in config["sweeps"]["noise_by_neuron_count"]["counts"]
    assert max(config["sweeps"]["noise_by_neuron_count"]["stds"]) >= 2.0


def test_timestamped_report_dir_preserves_repeated_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "reports" / "attractor_robustness"

    first_report_dir = _create_timestamped_report_dir(
        output_root=output_root,
        label="noise/std comparison",
        timestamp="20260707-010203",
    )
    second_report_dir = _create_timestamped_report_dir(
        output_root=output_root,
        label="noise/std comparison",
        timestamp="20260707-010203",
    )

    assert first_report_dir.name == "20260707-010203_noise_std_comparison"
    assert second_report_dir.name == "20260707-010203_noise_std_comparison_01"
    assert (first_report_dir / "figures").exists()
    assert (second_report_dir / "figures").exists()


def test_aggregate_sweep_grid_rows_groups_by_noise_and_neuron_count() -> None:
    rows = [
        {"n_theta": 16, "visual_noise_std": 0.0, "visual_velocity_gain": 1.0},
        {"n_theta": 16, "visual_noise_std": 0.0, "visual_velocity_gain": 1.2},
        {"n_theta": 32, "visual_noise_std": 0.0, "visual_velocity_gain": 0.8},
        {"n_theta": 16, "visual_noise_std": 0.1, "visual_velocity_gain": 0.7},
    ]

    aggregate_rows = _aggregate_sweep_grid_rows(
        rows,
        x_key="n_theta",
        y_key="visual_noise_std",
    )
    first_cell = aggregate_rows[0]

    assert first_cell["n_theta"] == 16
    assert first_cell["visual_noise_std"] == 0.0
    assert first_cell["repeat_count"] == 2
    assert np.isclose(first_cell["visual_velocity_gain_mean"], 1.1)
    assert np.isclose(first_cell["visual_velocity_gain_sem"], 0.1)


def test_metric_performance_score_grid_uses_metric_direction() -> None:
    gain_scores = _metric_performance_score_grid(
        "visual_velocity_gain",
        np.asarray([[1.0, 0.8], [1.2, 0.5]], dtype=float),
    )
    assert gain_scores[0, 0] > gain_scores[0, 1]
    assert np.isclose(gain_scores[0, 1], gain_scores[1, 0])
    assert gain_scores[0, 0] > gain_scores[1, 1]

    rmse_scores = _metric_performance_score_grid(
        "darkness_velocity_tracking_rmse",
        np.asarray([[0.05, 0.4]], dtype=float),
    )
    assert rmse_scores[0, 0] > rmse_scores[0, 1]

    hd_error_scores = _metric_performance_score_grid(
        "darkness_hd_decode_rms_error",
        np.asarray([[0.1, 2.0]], dtype=float),
    )
    assert hd_error_scores[0, 0] > hd_error_scores[0, 1]

    visual_cue_error_scores = _metric_performance_score_grid(
        "visual_cue_hd_decode_rms_error",
        np.asarray([[0.05, 1.5]], dtype=float),
    )
    assert visual_cue_error_scores[0, 0] > visual_cue_error_scores[0, 1]

    pva_scores = _metric_performance_score_grid(
        "darkness_final_pva_strength",
        np.asarray([[0.2, 0.9]], dtype=float),
    )
    assert pva_scores[0, 1] > pva_scores[0, 0]

    ideal_tie_scores = _metric_performance_score_grid(
        "darkness_minus_visual_velocity_gain",
        np.zeros((2, 2), dtype=float),
    )
    assert np.allclose(ideal_tie_scores, 1.0)


def test_sweep_metric_meshgrid_plot_smoke(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, *args, **kwargs: None)
    aggregate_rows = _aggregate_sweep_grid_rows(
        [
            {"n_theta": 16, "visual_noise_std": 0.0, "visual_velocity_gain": 1.0},
            {"n_theta": 32, "visual_noise_std": 0.0, "visual_velocity_gain": 0.9},
            {"n_theta": 16, "visual_noise_std": 0.1, "visual_velocity_gain": 0.8},
            {"n_theta": 32, "visual_noise_std": 0.1, "visual_velocity_gain": 0.7},
        ],
        x_key="n_theta",
        y_key="visual_noise_std",
    )

    _plot_sweep_metric_meshgrid(
        aggregate_rows=aggregate_rows,
        x_key="n_theta",
        y_key="visual_noise_std",
        path=tmp_path / "mesh.png",
        title="test mesh",
    )
