from __future__ import annotations

import numpy as np
import pytest

from learning.common.angles import make_theta_hd_pref, make_vafidis_paired_theta_hd_pref
from learning.config.schema import ExperimentConfig
from learning.common.random import make_rng
from learning.models.vafidis_toy import initialize_vafidis_toy_state
from learning.stimuli.visual import (
    VisualCurrentNoiseProcess,
    add_visual_noise,
    generate_heterogeneous_visual_profiles,
    make_heterogeneous_i_vis_to_hd,
    make_i_vis_to_hd,
)


def test_nested_master_profiles_preserve_coarse_grid_neurons() -> None:
    common_kwargs = {
        "n_angles": 64,
        "sigma": 1.4,
        "beta": 2.6,
        "bias": 2.1,
        "seed": 17,
        "alignment": "center_of_mass",
        "normalization": "unit_angular_mean",
        "population_sampling": "nested_master",
        "master_n_hd_cells": 120,
    }
    profiles_60 = generate_heterogeneous_visual_profiles(
        theta_hd_pref=make_vafidis_paired_theta_hd_pref(60),
        **common_kwargs,
    )
    profiles_120 = generate_heterogeneous_visual_profiles(
        theta_hd_pref=make_vafidis_paired_theta_hd_pref(120),
        **common_kwargs,
    )
    coarse_indices_in_120 = np.column_stack(
        [2 * np.arange(0, 60, 2), 2 * np.arange(0, 60, 2) + 1]
    ).reshape(-1)
    np.testing.assert_allclose(profiles_60, profiles_120[coarse_indices_in_120])


def test_default_visual_teacher_has_inhibitory_surround() -> None:
    config = ExperimentConfig()
    theta_hd_pref = make_theta_hd_pref(48)
    i_vis_to_hd = make_i_vis_to_hd(
        theta_hd_pref=theta_hd_pref,
        theta_true=0.0,
        amplitude=config.visual.amplitude,
        kappa=config.visual.kappa,
        baseline=config.visual.baseline,
        normalize_peak=config.visual.normalize_peak,
    )
    assert np.max(i_vis_to_hd) > 0.0
    assert np.min(i_vis_to_hd) < 0.0


def test_paper_like_visual_teacher_is_scaled_at_proximal_compartment() -> None:
    config = ExperimentConfig()
    config.model.n_theta = 48
    config.model.n_hr = 48
    config.model.init.random_jitter = 0.0
    config.visual.amplitude = 4.0
    config.visual.baseline = 5.0
    config.visual.light_excitation = 4.0
    config.visual.proximal_scale = 1.0 / 3.0

    state = initialize_vafidis_toy_state(config=config, rng=make_rng(config.simulation.seed))

    assert np.isclose(np.max(state.v_hd_proximal), 1.0)
    assert np.min(state.v_hd_proximal) < 0.0
    assert np.min(state.v_hd_proximal) > -0.35


def test_add_visual_noise_is_seed_reproducible() -> None:
    i_vis_to_hd = np.zeros(8, dtype=float)

    noisy_a = add_visual_noise(i_vis_to_hd, noise_std=0.2, rng=np.random.default_rng(123))
    noisy_b = add_visual_noise(i_vis_to_hd, noise_std=0.2, rng=np.random.default_rng(123))

    assert np.allclose(noisy_a, noisy_b)
    assert not np.allclose(noisy_a, i_vis_to_hd)


def test_zero_visual_noise_returns_copy_without_change() -> None:
    i_vis_to_hd = np.arange(5, dtype=float)

    unchanged = add_visual_noise(i_vis_to_hd, noise_std=0.0, rng=np.random.default_rng(123))

    assert np.allclose(unchanged, i_vis_to_hd)
    assert unchanged is not i_vis_to_hd


def test_ou_visual_noise_is_seed_reproducible() -> None:
    process_a = VisualCurrentNoiseProcess(
        mode="ou_additive",
        std=0.2,
        shape=(6,),
        rng=np.random.default_rng(123),
        correlation_time=0.05,
    )
    process_b = VisualCurrentNoiseProcess(
        mode="ou_additive",
        std=0.2,
        shape=(6,),
        rng=np.random.default_rng(123),
        correlation_time=0.05,
    )

    samples_a = np.asarray([process_a.step(0.01) for _ in range(8)])
    samples_b = np.asarray([process_b.step(0.01) for _ in range(8)])

    assert np.allclose(samples_a, samples_b)


def test_ou_visual_noise_has_stationary_std_and_autocorrelation() -> None:
    dt = 0.01
    tau = 0.05
    std = 0.3
    process = VisualCurrentNoiseProcess(
        mode="ou_additive",
        std=std,
        shape=(512,),
        rng=np.random.default_rng(321),
        correlation_time=tau,
    )

    samples = np.asarray([process.step(dt) for _ in range(1200)])
    assert np.isclose(np.std(samples), std, rtol=0.08)

    previous_samples = samples[:-1].reshape(-1)
    next_samples = samples[1:].reshape(-1)
    lag_one_correlation = np.corrcoef(previous_samples, next_samples)[0, 1]
    assert np.isclose(lag_one_correlation, np.exp(-dt / tau), atol=0.03)


def test_heterogeneous_visual_profiles_are_reproducible_and_com_aligned() -> None:
    theta_hd_pref = make_theta_hd_pref(24)
    keyword_arguments = {
        "theta_hd_pref": theta_hd_pref,
        "n_angles": 128,
        "sigma": 1.4,
        "beta": 2.6057585657926885,
        "bias": 2.0814271322479385,
        "seed": 77,
        "alignment": "center_of_mass",
        "normalization": "unit_angular_mean",
    }
    first = generate_heterogeneous_visual_profiles(**keyword_arguments)
    second = generate_heterogeneous_visual_profiles(**keyword_arguments)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (24, 128)
    np.testing.assert_allclose(np.mean(first, axis=1), 1.0, atol=1e-12)
    # Distinct rows retain heterogeneous shapes after preference alignment.
    assert not np.allclose(first[0], first[1])

    theta_grid = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    centers = np.angle(first @ np.exp(1j * theta_grid))
    center_errors = np.angle(np.exp(1j * (centers - theta_hd_pref)))
    assert np.max(np.abs(center_errors)) < 2.0 * np.pi / 128


def test_per_neuron_peak_calibration_gives_shared_visual_amplitude() -> None:
    profiles = generate_heterogeneous_visual_profiles(
        theta_hd_pref=make_theta_hd_pref(24),
        n_angles=128,
        sigma=1.4,
        beta=2.6057585657926885,
        bias=2.0814271322479385,
        seed=77,
        alignment="center_of_mass",
        normalization="per_neuron_peak",
    )

    np.testing.assert_allclose(np.max(profiles, axis=1), 1.0, atol=1e-12)
    assert not np.allclose(profiles[0], profiles[1])

    effective_current = (4.0 * profiles - 5.0 + 4.0) / 3.0
    np.testing.assert_allclose(np.max(effective_current, axis=1), 1.0, atol=1e-12)


def test_population_global_peak_normalization_is_not_supported() -> None:
    with pytest.raises(ValueError, match="Unknown heterogeneous_normalization"):
        generate_heterogeneous_visual_profiles(
            theta_hd_pref=make_theta_hd_pref(12),
            n_angles=64,
            sigma=1.4,
            beta=2.6,
            bias=2.08,
            seed=91,
            alignment="center_of_mass",
            normalization="global_peak",
        )


def test_unit_mean_heterogeneous_gain_matches_vafidis_mean_excitation() -> None:
    kappa = 11.11111111111111
    reference_amplitude = 4.0
    shared_heterogeneous_amplitude = (
        reference_amplitude * np.exp(-kappa) * np.i0(kappa)
    )
    profiles = generate_heterogeneous_visual_profiles(
        theta_hd_pref=make_theta_hd_pref(24),
        n_angles=256,
        sigma=1.4,
        beta=2.6057585657926885,
        bias=2.0814271322479385,
        seed=77,
        alignment="center_of_mass",
        normalization="unit_angular_mean",
    )

    np.testing.assert_allclose(np.mean(profiles, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        np.mean(shared_heterogeneous_amplitude * profiles, axis=1),
        shared_heterogeneous_amplitude,
        atol=1e-12,
    )
    theta = np.linspace(-np.pi, np.pi, 256, endpoint=False)
    reference_profile = np.exp(kappa * (np.cos(theta) - 1.0))
    assert np.isclose(
        shared_heterogeneous_amplitude,
        np.mean(reference_amplitude * reference_profile),
        atol=1e-12,
    )
    assert np.ptp(np.max(profiles, axis=1)) > 1.0


def test_heterogeneous_visual_input_is_periodic() -> None:
    theta_hd_pref = make_theta_hd_pref(12)
    profiles = generate_heterogeneous_visual_profiles(
        theta_hd_pref=theta_hd_pref,
        n_angles=64,
        sigma=1.4,
        beta=2.6,
        bias=2.08,
        seed=91,
        alignment="center_of_mass",
        normalization="unit_angular_mean",
    )
    before_wrap = make_heterogeneous_i_vis_to_hd(
        tuning_profiles=profiles,
        theta_true=-1e-8,
        amplitude=4.0,
        baseline=5.0,
    )
    after_wrap = make_heterogeneous_i_vis_to_hd(
        tuning_profiles=profiles,
        theta_true=2.0 * np.pi - 1e-8,
        amplitude=4.0,
        baseline=5.0,
    )
    np.testing.assert_allclose(before_wrap, after_wrap, atol=1e-12)


def test_model_initialization_builds_configured_heterogeneous_teacher() -> None:
    config = ExperimentConfig()
    config.model.n_theta = 24
    config.model.n_hr = 24
    config.visual.profile = "heterogeneous_gaussian_process"
    config.visual.heterogeneous_n_angles = 64
    config.visual.heterogeneous_seed_offset = 700

    first = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )
    second = initialize_vafidis_toy_state(
        config=config,
        rng=make_rng(config.simulation.seed),
    )

    assert first.visual_tuning_profiles is not None
    assert first.visual_tuning_profiles.shape == (24, 64)
    np.testing.assert_array_equal(
        first.visual_tuning_profiles,
        second.visual_tuning_profiles,
    )
    assert first.i_vis_to_hd.shape == (24,)
