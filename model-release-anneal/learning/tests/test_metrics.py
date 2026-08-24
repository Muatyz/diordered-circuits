from __future__ import annotations

import numpy as np

from learning.analysis.metrics import (
    angular_first_passage_time,
    benjamini_hochberg_adjusted_p_values,
    classify_endpoint_map_fixed_points,
    clark_overlap_order_parameter,
    decode_heading_by_clark_overlap,
    empirical_two_point_correlation,
    estimate_decoded_velocity,
    estimate_relaxation_e_folding_time,
    estimate_velocity_tracking_operating_range,
    empirical_tuning_preferred_directions,
    fit_anomalous_diffusion_power_law,
    kuiper_uniformity_test_asymptotic,
    nearest_closed_manifold_distance,
    relative_circulant_error,
    release_relative_pi_error_trace,
    summarize_ensemble_diffusion_coefficient,
    summarize_ensemble_diffusion_trajectories,
    summarize_com_aligned_tuning_curves,
    summarize_velocity_gain,
    summarize_velocity_tracking,
    summarize_pi_error_ensemble,
)


def test_endpoint_map_classifies_attractors_and_bracketed_boundaries() -> None:
    theta_initial = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    theta_final = np.where(
        np.abs(theta_initial) < 0.5 * np.pi,
        0.0,
        -np.pi,
    )

    fixed_points = classify_endpoint_map_fixed_points(
        theta_initial=theta_initial,
        theta_final=theta_final,
    )

    theta = fixed_points["fixed_point_theta"]
    stability = fixed_points["fixed_point_stability"]
    assert np.count_nonzero(stability == -1) == 2
    assert np.count_nonzero(stability == 1) == 2
    attracting = np.sort(theta[stability == -1])
    boundaries = np.sort(fixed_points["basin_boundary_theta"])
    np.testing.assert_allclose(attracting, [-np.pi, 0.0], atol=np.deg2rad(1.1))
    np.testing.assert_allclose(
        boundaries,
        [-0.5 * np.pi, 0.5 * np.pi],
        atol=np.deg2rad(1.1),
    )
    assert fixed_points["unresolved_boundary_theta"].size == 0


def test_endpoint_map_identity_has_no_discrete_fixed_point_markers() -> None:
    theta_initial = np.linspace(-np.pi, np.pi, 120, endpoint=False)

    fixed_points = classify_endpoint_map_fixed_points(
        theta_initial=theta_initial,
        theta_final=theta_initial,
    )

    assert fixed_points["fixed_point_theta"].size == 0
    assert fixed_points["fixed_point_stability"].size == 0
    assert fixed_points["basin_boundary_theta"].size == 0
    assert fixed_points["unresolved_boundary_theta"].size == 0


def test_endpoint_map_keeps_unbracketed_cluster_transitions_unresolved() -> None:
    theta_initial = np.linspace(-np.pi, np.pi, 60, endpoint=False)
    theta_final = np.repeat(theta_initial[::2], 2)

    fixed_points = classify_endpoint_map_fixed_points(
        theta_initial=theta_initial,
        theta_final=theta_final,
    )

    # A one-sided staircase contains repeated endpoints but no signed
    # displacement crossing, so it does not provide enough evidence to label
    # either side of a fixed point.
    assert fixed_points["fixed_point_stability"].size == 0
    assert fixed_points["basin_boundary_theta"].size == 0
    assert fixed_points["repeated_endpoint_support_fraction"] == 1.0


def test_endpoint_map_classifies_periodic_root_at_wrap_seam() -> None:
    theta_initial = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    displacement = -0.2 * np.sin(theta_initial)
    theta_final = theta_initial + displacement

    fixed_points = classify_endpoint_map_fixed_points(
        theta_initial=theta_initial,
        theta_final=theta_final,
    )

    stability = fixed_points["fixed_point_stability"]
    assert np.count_nonzero(stability == -1) == 1
    assert np.count_nonzero(stability == 1) == 1
    np.testing.assert_allclose(
        fixed_points["fixed_point_theta"][stability == -1],
        0.0,
        atol=np.deg2rad(1.1),
    )
    assert np.min(
        np.abs(
            np.angle(
                np.exp(
                    1j
                    * (
                        fixed_points["fixed_point_theta"][stability == 1]
                        + np.pi
                    )
                )
            )
        )
    ) <= np.deg2rad(1.1)


def test_endpoint_map_infers_subbin_boundaries_for_quantized_peak_decoder() -> None:
    theta_probe = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    probe_phase = np.mod(theta_probe + np.pi, 2.0 * np.pi)
    peak_bin_width = 2.0 * np.pi / 30.0
    theta_release = (
        np.floor(probe_phase / peak_bin_width) * peak_bin_width - np.pi
    )
    theta_final = theta_release + 0.2 * np.sin(
        theta_probe - np.deg2rad(3.0)
    )

    landscape = classify_endpoint_map_fixed_points(
        theta_initial=theta_probe,
        theta_release=theta_release,
        theta_final=theta_final,
    )

    stability = landscape["fixed_point_stability"]
    assert bool(landscape["cue_transfer_valid"])
    assert np.isclose(
        landscape["cue_transfer_orientation_preserving_fraction"],
        1.0,
    )
    assert landscape["cue_transfer_plateau_fraction"] > 0.9
    assert landscape["nonmonotonic_transition_theta"].size == 0
    assert np.count_nonzero(stability == -1) == 1
    assert np.count_nonzero(stability == 1) == 1
    assert np.all(landscape["fixed_point_release_resolution_limited"])
    np.testing.assert_allclose(
        landscape["subbin_boundary_initial_theta"],
        np.deg2rad(3.0),
        atol=np.deg2rad(1.1),
    )
    assert landscape["unresolved_boundary_theta"].size == 0


def test_endpoint_map_uses_release_phase_for_autonomous_displacement() -> None:
    theta_probe = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    theta_release = theta_probe + np.deg2rad(8.0)
    theta_final = np.where(
        np.abs(theta_probe) < 0.5 * np.pi,
        np.deg2rad(8.0),
        -np.pi + np.deg2rad(8.0),
    )

    landscape = classify_endpoint_map_fixed_points(
        theta_initial=theta_probe,
        theta_release=theta_release,
        theta_final=theta_final,
    )

    np.testing.assert_allclose(
        np.sort(landscape["basin_boundary_theta"]),
        np.deg2rad([-82.0, 98.0]),
        atol=np.deg2rad(1.1),
    )


def test_endpoint_map_keeps_one_unstable_per_attractor_interval() -> None:
    theta_probe = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    theta_final = theta_probe - 0.15 * np.sin(3.0 * theta_probe)

    landscape = classify_endpoint_map_fixed_points(
        theta_initial=theta_probe,
        theta_release=theta_probe,
        theta_final=theta_final,
    )

    stability = landscape["fixed_point_stability"]
    assert np.count_nonzero(stability == -1) == 3
    assert np.count_nonzero(stability == 1) == 3
    assert landscape["basin_boundary_theta"].size == 3
    assert landscape["nonmonotonic_transition_theta"].size == 0
    assert landscape["unresolved_boundary_theta"].size == 0
    assert landscape["missing_boundary_interval_theta"].size == 0
    stable = np.sort(
        np.mod(
            landscape["fixed_point_theta"][stability == -1] + np.pi,
            2.0 * np.pi,
        )
    )
    unstable = np.mod(
        landscape["fixed_point_theta"][stability == 1] + np.pi,
        2.0 * np.pi,
    )
    for stable_index, left_stable in enumerate(stable):
        stable_gap = (
            stable[(stable_index + 1) % stable.size] - left_stable
        ) % (2.0 * np.pi)
        unstable_offset = (unstable - left_stable) % (2.0 * np.pi)
        assert np.count_nonzero(unstable_offset < stable_gap) == 1


def test_endpoint_map_reports_distorted_cue_transfer_separately() -> None:
    theta_probe = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    theta_release = -theta_probe
    theta_final = theta_release - 0.1 * np.sin(2.0 * theta_release)

    landscape = classify_endpoint_map_fixed_points(
        theta_initial=theta_probe,
        theta_release=theta_release,
        theta_final=theta_final,
    )

    assert not bool(landscape["cue_transfer_valid"])
    assert landscape["nonmonotonic_transition_theta"].size > 0
    assert landscape["fixed_point_theta"].size == 4


def test_release_relative_pi_error_removes_alignment_and_preserves_turns() -> None:
    true = np.asarray([3.0, -2.5, -1.7, -0.9])
    decoded_unwrapped = np.unwrap(true) + 0.4 + np.asarray([0.0, 0.1, 0.2, 0.3])
    decoded = (decoded_unwrapped + np.pi) % (2.0 * np.pi) - np.pi

    error = release_relative_pi_error_trace(decoded, true)

    np.testing.assert_allclose(error, [0.0, 0.1, 0.2, 0.3], atol=1e-12)


def test_closed_manifold_distance_interpolates_between_heading_bins() -> None:
    manifold = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    distance, coordinate = nearest_closed_manifold_distance(
        np.asarray([[0.5, 0.5], [0.0, 0.0]]),
        manifold,
    )

    assert np.isclose(distance[0], 0.0)
    assert np.isclose(coordinate[0], 0.5)
    assert np.isclose(distance[1], 0.5)


def test_relaxation_and_first_passage_estimators_report_censoring() -> None:
    time = np.linspace(0.0, 5.0, 501)
    distance = 0.1 + np.exp(-time / 0.5)
    relaxation = estimate_relaxation_e_folding_time(
        time=time,
        distance=distance[None, :],
        peak_window=0.0,
    )
    assert bool(relaxation["event_observed"][0])
    assert np.isclose(relaxation["e_folding_time"][0], 0.5, atol=0.02)

    passage_time, observed = angular_first_passage_time(
        time=np.asarray([0.0, 1.0]),
        angular_displacement=np.asarray([[0.0, 0.4], [0.0, 0.1]]),
        threshold=0.2,
    )
    np.testing.assert_allclose(passage_time, [0.5, 1.0])
    np.testing.assert_array_equal(observed, [True, False])


def test_clark_overlap_order_parameter_is_uncentered_and_decodes_argmax() -> None:
    theta_template = np.asarray([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi])
    target_rate = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    activity = np.asarray([[2.0, 0.1], [0.1, 3.0]])

    overlap = clark_overlap_order_parameter(target_rate, activity)
    np.testing.assert_allclose(overlap, target_rate.T @ activity.T / 2.0)
    decoded, maximum_overlap = decode_heading_by_clark_overlap(
        theta_template=theta_template,
        target_rate=target_rate,
        population_activity=activity,
    )
    np.testing.assert_allclose(decoded, [-np.pi, -0.5 * np.pi])
    np.testing.assert_allclose(maximum_overlap, [2.0, 1.5])


def test_clark_two_point_function_is_uncentered() -> None:
    tuning = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(
        empirical_two_point_correlation(tuning),
        tuning.T @ tuning / 2.0,
    )


def test_relative_circulant_error_detects_translation_symmetry() -> None:
    first_column = np.array([1.0, 0.5, -0.2, 0.5])
    circulant = np.column_stack(
        [np.roll(first_column, shift) for shift in range(first_column.size)]
    )
    assert relative_circulant_error(circulant) < 1e-12

    perturbed = circulant.copy()
    perturbed[0, 1] += 0.7
    assert relative_circulant_error(perturbed) > 0.01


def test_kuiper_uniformity_and_bh_correction() -> None:
    uniform_angles = np.linspace(-np.pi, np.pi, 40, endpoint=False)
    statistic, p_value = kuiper_uniformity_test_asymptotic(uniform_angles)
    assert statistic > 0.0
    assert p_value > 0.5

    adjusted = benjamini_hochberg_adjusted_p_values(np.array([0.01, 0.04, 0.5]))
    np.testing.assert_allclose(adjusted, np.array([0.03, 0.06, 0.5]))


def test_estimate_decoded_velocity_uses_simple_linear_fit() -> None:
    time = np.linspace(0.0, 1.0, 11)
    theta_decoded = 0.25 + 0.75 * time
    assert np.isclose(
        estimate_decoded_velocity(time=time, theta_decoded=theta_decoded, start_fraction=0.0),
        0.75,
    )


def test_summarize_velocity_gain_uses_simple_linear_fit() -> None:
    commanded_velocity = np.array([-1.0, -0.5, 0.5, 1.0])
    decoded_velocity = 0.2 + 0.8 * commanded_velocity
    summary = summarize_velocity_gain(
        commanded_velocity=commanded_velocity,
        decoded_velocity=decoded_velocity,
    )
    assert np.isclose(summary["gain"], 0.8)
    assert np.isclose(summary["intercept"], 0.2)
    assert np.isclose(summary["r_squared"], 1.0)
    assert np.isclose(summary["linear_fit_rmse"], 0.0)


def test_velocity_tracking_operating_range_stops_at_first_failed_magnitude() -> None:
    commanded = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    decoded = np.asarray([-3.1, -2.1, -1.0, 0.0, 1.0, 0.0, 3.1])

    assert np.isclose(
        estimate_velocity_tracking_operating_range(
            commanded_velocity=commanded,
            decoded_velocity=decoded,
        ),
        1.0,
    )


def test_empirical_tuning_preference_uses_circular_center_of_mass() -> None:
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    rates = np.column_stack(
        [
            1.0 + np.cos(theta - 0.5 * np.pi),
            1.0 + np.cos(theta + 0.5 * np.pi),
        ]
    )
    preference, strength = empirical_tuning_preferred_directions(
        theta_true=theta,
        r_hd_by_heading=rates,
    )
    np.testing.assert_allclose(preference, [0.5 * np.pi, -0.5 * np.pi])
    assert np.all(strength > 0.0)


def test_com_aligned_tuning_summary_peak_normalizes_every_hd_neuron() -> None:
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    preferred_directions = theta[[1, 3, 5, 7]]
    amplitudes = np.asarray([0.5, 1.0, 2.0, 4.0])
    rates = np.column_stack(
        [
            amplitude * (2.0 + np.cos(theta - preferred_direction))
            for amplitude, preferred_direction in zip(amplitudes, preferred_directions)
        ]
    )

    summary = summarize_com_aligned_tuning_curves(
        theta_true=theta,
        r_hd_by_heading=rates,
    )

    aligned = summary["r_hd_peak_normalized_com_aligned"]
    assert aligned.shape == (preferred_directions.size, theta.size)
    np.testing.assert_allclose(np.max(aligned, axis=1), 1.0)
    np.testing.assert_allclose(aligned, np.repeat(aligned[:1], aligned.shape[0], axis=0))
    np.testing.assert_allclose(summary["r_hd_peak_normalized_com_aligned_mean"], aligned[0])
    np.testing.assert_allclose(
        summary["r_hd_peak_normalized_com_aligned_std"],
        0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(summary["r_hd_peak_rate"], 3.0 * amplitudes)
    np.testing.assert_allclose(summary["r_hd_angular_mean"], 2.0 * amplitudes)
    np.testing.assert_array_equal(summary["plot_normalization"], "per_neuron_peak")
    assert int(summary["simulated_mouse_count"]) == 1
    assert int(summary["n_hd_neurons"]) == preferred_directions.size
    np.testing.assert_allclose(
        summary["empirical_preferred_direction"],
        preferred_directions,
    )


def test_com_aligned_tuning_summary_records_effectively_silent_hd_neurons() -> None:
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    active_curve = 2.0 + np.cos(theta - theta[2])
    rates = np.column_stack(
        [
            active_curve,
            np.zeros_like(theta),
            np.full_like(theta, 1e-20),
        ]
    )

    summary = summarize_com_aligned_tuning_curves(
        theta_true=theta,
        r_hd_by_heading=rates,
    )

    aligned = summary["r_hd_peak_normalized_com_aligned"]
    np.testing.assert_allclose(np.max(aligned[0]), 1.0)
    np.testing.assert_array_equal(aligned[1:], 0.0)
    np.testing.assert_array_equal(
        summary["r_hd_tuning_valid_mask"],
        [True, False, False],
    )
    assert int(summary["r_hd_tuning_valid_neuron_count"]) == 1
    assert int(summary["r_hd_tuning_silent_neuron_count"]) == 2
    assert np.isclose(summary["r_hd_tuning_silent_neuron_fraction"], 2.0 / 3.0)
    assert np.isnan(summary["r_hd_unit_mean_com_aligned"][1:]).all()
    assert np.isnan(summary["empirical_preferred_direction"][1:]).all()
    np.testing.assert_array_equal(summary["preferred_heading_bin"][1:], -1)
    np.testing.assert_array_equal(summary["com_alignment_shift_bins"][1:], 0)
    np.testing.assert_allclose(
        summary["r_hd_peak_normalized_com_aligned_mean"],
        aligned[0] / 3.0,
    )


def test_pi_error_ensemble_fits_drift_of_mean_pva_error() -> None:
    time = np.linspace(0.0, 2.0, 5)
    errors = np.vstack([0.2 * time - 0.1, 0.2 * time + 0.1])
    summary = summarize_pi_error_ensemble(time=time, pi_error=errors)
    np.testing.assert_allclose(summary["pi_error_mean"], 0.2 * time)
    assert np.isclose(summary["systematic_drift_velocity"], 0.2)
    assert np.isclose(summary["final_pi_error_std"], 0.1)


def test_summarize_velocity_tracking_measures_error_to_ideal_pi() -> None:
    commanded_velocity = np.array([-1.0, 0.0, 1.0])
    decoded_velocity = np.array([-0.5, 0.1, 1.2])
    summary = summarize_velocity_tracking(
        commanded_velocity=commanded_velocity,
        decoded_velocity=decoded_velocity,
    )

    expected_error = decoded_velocity - commanded_velocity
    assert np.isclose(summary["velocity_tracking_rmse"], np.sqrt(np.mean(expected_error**2)))
    assert np.isclose(summary["velocity_tracking_mae"], np.mean(np.abs(expected_error)))
    assert np.isclose(summary["velocity_tracking_max_abs_error"], np.max(np.abs(expected_error)))
    assert np.isclose(summary["velocity_tracking_bias"], np.mean(expected_error))
    assert np.isclose(summary["velocity_direction_match_fraction"], 1.0)


def test_summarize_ensemble_diffusion_uses_variance_over_duration() -> None:
    displacement = np.array([-0.2, 0.0, 0.2])
    summary = summarize_ensemble_diffusion_coefficient(
        angular_displacement=displacement,
        duration=2.0,
    )

    assert np.isclose(summary["diffusion_coefficient"], np.var(displacement) / 2.0)
    assert np.isclose(summary["displacement_mean"], 0.0)
    assert np.isclose(summary["displacement_std"], np.sqrt(np.var(displacement)))
    assert summary["n_trials"] == 3.0


def test_ensemble_trajectory_summary_separates_drift_from_diffusion() -> None:
    time = np.array([0.0, 0.5, 1.0])
    displacement = np.array(
        [
            [-0.0, -0.5, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 1.0],
        ]
    )
    summary = summarize_ensemble_diffusion_trajectories(
        time=time,
        angular_displacement=displacement,
    )

    assert np.isclose(summary["diffusion_coefficient"], 2.0 / 3.0)
    assert np.isclose(summary["systematic_drift_velocity"], 0.0)
    assert np.allclose(summary["displacement_mean_trace"], 0.0)
    assert np.isclose(summary["diffusion_coefficient_trace"][-1], 2.0 / 3.0)


def test_anomalous_diffusion_power_law_recovers_exponent_and_coefficient() -> None:
    time = np.linspace(0.0, 4.0, 17)
    expected_exponent = 1.4
    expected_coefficient = 0.3
    variance = np.zeros_like(time)
    variance[1:] = expected_coefficient * time[1:] ** expected_exponent

    summary = fit_anomalous_diffusion_power_law(
        time=time,
        displacement_variance=variance,
        fit_start_time=0.5,
        fit_end_time=3.5,
    )

    assert np.isclose(summary["anomalous_diffusion_exponent"], expected_exponent)
    assert np.isclose(summary["generalized_diffusion_coefficient"], expected_coefficient)
    assert np.isclose(summary["anomalous_diffusion_log_r_squared"], 1.0)
    assert np.isclose(summary["anomalous_diffusion_fit_start_time"], 0.5)
    assert np.isclose(summary["anomalous_diffusion_fit_end_time"], 3.5)
