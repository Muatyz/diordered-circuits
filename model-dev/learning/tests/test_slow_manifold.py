import numpy as np

from learning.analysis.slow_manifold import (
    _pca_variance_summary,
    _sum_hd_rates_by_preferred_direction,
    fit_periodic_state_curve,
    select_slow_candidate_indices,
    summarize_candidate_angle_clusters,
)
from learning.plotting.slow_manifold import plot_ramesan_pca_variance_rank


def test_slow_candidate_selection_uses_per_trajectory_maximum() -> None:
    speed = np.asarray([10.0, 1.0, 0.009, 0.005, 0.02])
    selected, threshold = select_slow_candidate_indices(
        speed=speed,
        speed_fraction=1e-3,
        maximum_points=10,
    )
    assert np.isclose(threshold, 0.01)
    assert np.array_equal(selected, np.asarray([2, 3]))


def test_slow_candidate_selection_speed_floor_caps_threshold() -> None:
    # The trajectory maximum is set by the relaxation transient (10.0), so the
    # relative threshold alone (0.01) admits mid-relaxation points.  A
    # physical floor must further restrict the candidates.
    speed = np.asarray([10.0, 1.0, 0.009, 0.005, 0.02, 0.0])
    selected, threshold = select_slow_candidate_indices(
        speed=speed,
        speed_fraction=1e-3,
        maximum_points=10,
        speed_floor=0.008,
    )
    assert np.isclose(threshold, 0.008)
    # 0.009 exceeds the floor, so only indices 3 (0.005) and 5 (0.0) qualify.
    assert np.array_equal(selected, np.asarray([3, 5]))


def test_slow_candidate_selection_time_uniform_resampling() -> None:
    # Many late-time points qualify; with time provided, the budget is spread
    # uniformly in time instead of clumping in the final basin.
    time = np.linspace(0.0, 5.0, 21)
    speed = np.empty_like(time)
    speed[:] = 1.0
    speed[0] = 100.0  # transient maximum
    speed[1:] = 0.001  # all slow after the first sample
    selected, threshold = select_slow_candidate_indices(
        speed=speed,
        speed_fraction=1e-3,
        maximum_points=5,
        speed_floor=0.01,
        time=time,
    )
    assert np.isclose(threshold, 0.01)
    assert selected.size == 5
    # Candidates must span the whole time range, not just the tail.
    assert time[selected].min() < 1.0
    assert time[selected].max() > 4.0
    assert len(np.unique(selected)) == 5


def test_slow_candidate_time_resampling_fills_sparse_slots() -> None:
    # Regression: when more candidates qualify than the budget, the time-slot
    # fill must store candidate positions (not values) before mapping once.
    # The old code mixed the two and indexed past the candidate array,
    # raising IndexError on N=60 runs whose darkness trajectory has 51
    # samples and ~30 slow candidates below the physical speed floor.
    time = np.linspace(0.0, 5.0, 51)
    speed = np.full_like(time, 1.0)
    speed[0] = 100.0
    # 30 candidates qualify (a dense late-time run), budget 12.
    speed[20:] = 0.0001
    selected, threshold = select_slow_candidate_indices(
        speed=speed,
        speed_fraction=1e-3,
        maximum_points=12,
        speed_floor=0.001,
        time=time,
    )
    assert np.isclose(threshold, 0.001)
    assert selected.size == 12
    # Every returned index must be a valid trajectory index AND a candidate.
    assert np.all(selected >= 0)
    assert np.all(selected < speed.size)
    assert np.all(speed[selected] <= threshold)
    assert len(np.unique(selected)) == 12


def test_periodic_spline_recovers_a_closed_ring() -> None:
    theta = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    state = np.column_stack([np.cos(theta), np.sin(theta), 0.25 * np.cos(2.0 * theta)])
    result = fit_periodic_state_curve(
        candidate_theta=theta,
        candidate_state=state,
        angular_bin_count=64,
    )
    fitted = np.asarray(result["state"])
    tangent = np.asarray(result["tangent"])
    fitted_theta = np.asarray(result["theta"])
    expected = np.column_stack(
        [
            np.cos(fitted_theta),
            np.sin(fitted_theta),
            0.25 * np.cos(2.0 * fitted_theta),
        ]
    )
    assert fitted.shape == state.shape
    assert np.max(np.linalg.norm(fitted - expected, axis=1)) < 0.06
    assert np.all(np.linalg.norm(tangent, axis=1) > 0.0)
    assert result["angular_support_fraction"] > 0.9


def test_candidate_angle_clusters_merge_across_periodic_boundary() -> None:
    counts = np.asarray([2, 1, 0, 0, 3, 0, 0, 4])
    summary = summarize_candidate_angle_clusters(bin_sample_count=counts)
    assert np.array_equal(
        np.sort(summary["cluster_sample_count"]), np.asarray([3, 7])
    )
    assert np.array_equal(np.sort(summary["cluster_bin_count"]), np.asarray([1, 3]))


def test_pca_variance_summary_keeps_the_complete_rank_spectrum() -> None:
    phase = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    samples = np.column_stack(
        [
            np.cos(phase),
            np.sin(phase),
            0.25 * np.cos(2.0 * phase),
            0.1 * np.sin(3.0 * phase),
        ]
    )
    summary = _pca_variance_summary(samples)
    explained = np.asarray(summary["explained_fraction"])
    cumulative = np.asarray(summary["cumulative_fraction"])
    assert explained.shape == (4,)
    assert cumulative.shape == (4,)
    assert np.all(np.diff(cumulative) >= -1e-15)
    assert np.isclose(cumulative[-1], 1.0)
    assert 1.0 <= float(summary["participation_ratio"]) <= 4.0


def test_pair_summed_hd_rates_are_the_exact_pva_sufficient_statistic() -> None:
    theta_hd_pref = np.repeat(
        np.asarray([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0]), 2
    )
    hd_rate = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.8, 0.7, 0.2, 0.1],
            [0.5, 0.4, 0.1, 0.2, 0.3, 0.2, 0.9, 0.8],
        ]
    )
    unique_theta, angular_rate = _sum_hd_rates_by_preferred_direction(
        theta_hd_pref=theta_hd_pref,
        hd_rate=hd_rate,
    )
    direct_vector = hd_rate @ np.exp(1j * theta_hd_pref)
    grouped_vector = angular_rate @ np.exp(1j * unique_theta)
    np.testing.assert_allclose(grouped_vector, direct_vector)


def test_variance_rank_plot_accepts_state_rate_and_pva_spectra(tmp_path) -> None:
    explained = np.asarray([0.40, 0.30, 0.20, 0.10])
    firing_rate_explained = np.asarray([0.45, 0.30, 0.15, 0.10])
    pva_explained = np.asarray([0.55, 0.30, 0.10, 0.05])
    output_path = tmp_path / "variance_rank.png"
    plot_ramesan_pca_variance_rank(
        history={
            "ramesan_pca_explained_variance_spectrum": explained,
            "ramesan_pca_cumulative_explained_variance": np.cumsum(explained),
            "ramesan_pca_feature_scale": np.ones(12),
            "ramesan_firing_rate_pca_explained_variance_spectrum": (
                firing_rate_explained
            ),
            "ramesan_firing_rate_pca_cumulative_explained_variance": np.cumsum(
                firing_rate_explained
            ),
            "ramesan_pva_rate_pca_explained_variance_spectrum": pva_explained,
            "ramesan_pva_rate_pca_cumulative_explained_variance": np.cumsum(
                pva_explained
            ),
        },
        path=output_path,
    )
    assert output_path.exists()


def test_variance_rank_plot_accepts_legacy_two_spectrum_history(tmp_path) -> None:
    explained = np.asarray([0.6, 0.3, 0.1])
    output_path = tmp_path / "legacy_variance_rank.png"
    plot_ramesan_pca_variance_rank(
        history={
            "ramesan_pca_explained_variance_spectrum": explained,
            "ramesan_pca_cumulative_explained_variance": np.cumsum(explained),
            "ramesan_pva_rate_pca_explained_variance_spectrum": explained,
            "ramesan_pva_rate_pca_cumulative_explained_variance": np.cumsum(
                explained
            ),
        },
        path=output_path,
    )
    assert output_path.exists()
