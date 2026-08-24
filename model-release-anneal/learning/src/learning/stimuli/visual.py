"""Visual teacher input for HD cells."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from learning.common.angles import circular_difference


@dataclass
class ScheduledVisualAmplitude:
    """Visual teacher amplitude with a piecewise-constant training schedule.

    Mirrors ``ScheduledOUAngularVelocity``: the teacher amplitude is constant
    within each interval of training progress (``end_fraction`` of
    ``total_duration``) and steps down/up at the configured boundaries.  This
    implements the "night-vision" curriculum: expose the network to a strong
    cue early (stable learning signal), then anneal toward a weak cue so late
    training must rely on the learned recurrent weights.
    """

    total_duration: float
    amplitude_schedule: list[dict[str, Any]]
    elapsed_time: float = 0.0
    _schedule: tuple[tuple[float, float], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.total_duration) or self.total_duration <= 0.0:
            raise ValueError(
                "scheduled visual amplitude total_duration must be finite and positive"
            )
        if not isinstance(self.amplitude_schedule, list) or not self.amplitude_schedule:
            raise ValueError(
                "visual.training_amplitude_schedule must be a non-empty list"
            )
        parsed: list[tuple[float, float]] = []
        previous_end = 0.0
        for index, entry in enumerate(self.amplitude_schedule):
            if not isinstance(entry, dict) or set(entry) != {
                "end_fraction",
                "amplitude",
            }:
                raise ValueError(
                    "each visual.training_amplitude_schedule entry must contain "
                    "only end_fraction and amplitude"
                )
            end_fraction = float(entry["end_fraction"])
            amplitude = float(entry["amplitude"])
            if (
                not np.isfinite(end_fraction)
                or end_fraction <= previous_end
                or end_fraction > 1.0
            ):
                raise ValueError(
                    "visual.training_amplitude_schedule end_fraction values must "
                    "be strictly increasing in (0, 1]"
                )
            if not np.isfinite(amplitude) or amplitude < 0.0:
                raise ValueError(
                    "visual.training_amplitude_schedule amplitude values must "
                    "be finite and non-negative"
                )
            parsed.append((end_fraction, amplitude))
            previous_end = end_fraction
        if not np.isclose(parsed[-1][0], 1.0, atol=1e-12, rtol=0.0):
            raise ValueError(
                "visual.training_amplitude_schedule must end at fraction 1.0"
            )
        self._schedule = tuple(parsed)

    def current_amplitude(self, dt: float = 0.0) -> float:
        """Return the teacher amplitude at the midpoint of the current step."""
        midpoint_time = min(
            self.elapsed_time + 0.5 * max(float(dt), 0.0),
            self.total_duration,
        )
        fraction = midpoint_time / self.total_duration
        for end_fraction, amplitude in self._schedule:
            if fraction <= end_fraction:
                return amplitude
        return self._schedule[-1][1]

    def step(self, dt: float) -> float:
        """Advance the schedule clock and return the amplitude for this step."""
        amplitude = self.current_amplitude(dt)
        self.elapsed_time += float(dt)
        return amplitude


def _wrapped_gaussian_correlation(delta_theta: np.ndarray, sigma: float) -> np.ndarray:
    """Clark et al. Eq. 9, matching ``reproduction/src/utils.py``.

    This small local implementation keeps the learning package independently
    installable while using the same tested generative process as the
    reproduction subproject.
    """
    delta_theta = np.asarray(delta_theta, dtype=float)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("heterogeneous_sigma must be positive")
    wrapped_delta = (delta_theta + np.pi) % (2.0 * np.pi) - np.pi
    tail_distance = np.sqrt(-2.0 * np.log(1e-14)) / sigma
    image_radius = max(1, int(np.ceil((tail_distance + np.pi) / (2.0 * np.pi))))
    images = np.arange(-image_radius, image_radius + 1, dtype=float)
    shifted = wrapped_delta[..., None] + 2.0 * np.pi * images
    return np.sum(np.exp(-0.5 * sigma * sigma * shifted * shifted), axis=-1)


def _sample_circular_gaussian_process(
    *,
    n_samples: int,
    n_angles: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fourier-sample the discretized wrapped-Gaussian process."""
    if n_samples <= 0 or n_angles <= 1:
        raise ValueError("n_samples must be positive and n_angles must exceed one")
    theta_grid = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    covariance_row = _wrapped_gaussian_correlation(theta_grid, sigma)
    eigenvalues = np.fft.rfft(covariance_row).real
    roundoff_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-10 * roundoff_scale:
        raise ValueError("heterogeneous covariance is not positive semidefinite")
    white_noise = rng.normal(size=(n_samples, n_angles))
    white_spectrum = np.fft.rfft(white_noise, axis=-1)
    return np.fft.irfft(
        white_spectrum * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :],
        n=n_angles,
        axis=-1,
    )


def _normalized_softplus_tuning(
    input_currents: np.ndarray,
    *,
    beta: float,
    bias: float,
) -> np.ndarray:
    """Clark et al. Eq. 10/B3 normalized-softplus tuning curves."""
    if beta <= 0.0:
        raise ValueError("heterogeneous_beta must be positive")
    if not np.isfinite(input_currents).all():
        raise ValueError("heterogeneous input currents must be finite")
    unnormalized = np.logaddexp(beta * (input_currents - bias), 0.0)
    normalization = np.mean(unnormalized, axis=1, keepdims=True)
    if np.any(normalization <= 0.0) or not np.isfinite(normalization).all():
        raise ValueError("heterogeneous softplus normalization must be positive")
    return unnormalized / normalization


def _periodic_row_interpolation(
    tuning_profiles: np.ndarray,
    query_angles: np.ndarray,
) -> np.ndarray:
    """Evaluate one periodic query angle for every row of a tuning matrix."""
    n_angles = tuning_profiles.shape[1]
    fractional_indices = np.mod(query_angles, 2.0 * np.pi) * n_angles / (2.0 * np.pi)
    lower_indices = np.floor(fractional_indices).astype(int) % n_angles
    upper_indices = (lower_indices + 1) % n_angles
    upper_weight = fractional_indices - np.floor(fractional_indices)
    row_indices = np.arange(tuning_profiles.shape[0])
    return (
        (1.0 - upper_weight) * tuning_profiles[row_indices, lower_indices]
        + upper_weight * tuning_profiles[row_indices, upper_indices]
    )


def _nested_paired_hd_indices(*, n_hd_cells: int, master_n_hd_cells: int) -> np.ndarray:
    """Select a coarse paired-HD angular grid from a maximum-size population."""
    n_hd_cells = int(n_hd_cells)
    master_n_hd_cells = int(master_n_hd_cells)
    if n_hd_cells <= 0 or master_n_hd_cells <= 0:
        raise ValueError("HD population sizes must be positive")
    if n_hd_cells % 2 != 0 or master_n_hd_cells % 2 != 0:
        raise ValueError("nested paired-HD sampling requires even population sizes")
    n_heading = n_hd_cells // 2
    master_n_heading = master_n_hd_cells // 2
    if master_n_heading < n_heading or master_n_heading % n_heading != 0:
        raise ValueError(
            "heterogeneous_master_n_hd_cells must define an integer refinement "
            "of the requested paired-HD heading grid"
        )
    heading_stride = master_n_heading // n_heading
    master_heading_indices = np.arange(n_heading, dtype=int) * heading_stride
    return np.column_stack(
        [2 * master_heading_indices, 2 * master_heading_indices + 1]
    ).reshape(-1)


def generate_heterogeneous_visual_profiles(
    *,
    theta_hd_pref: np.ndarray,
    n_angles: int,
    sigma: float,
    beta: float,
    bias: float,
    seed: int,
    alignment: str,
    normalization: str,
    population_sampling: str = "independent",
    master_n_hd_cells: int | None = None,
) -> np.ndarray:
    """Generate a reproducible heterogeneous visual-teacher manifold.

    Each GP sample is phase-shifted so its circular center of mass coincides
    with the corresponding model HD preference.  This retains the model's
    paired HD/HR geometry while replacing the identical von-Mises profiles by
    heterogeneous, potentially asymmetric or multi-peaked profiles.

    ``per_neuron_peak`` makes every shape peak at one before the common visual
    amplitude is applied. ``unit_angular_mean`` retains Clark's generator
    normalization and therefore retains heterogeneous peaks. Both choices are
    independent of the largest sampled cell in the population.
    """
    theta_hd_pref = np.asarray(theta_hd_pref, dtype=float)
    population_sampling = str(population_sampling).lower()
    rng = np.random.default_rng(int(seed))
    if population_sampling == "independent":
        currents = _sample_circular_gaussian_process(
            n_samples=theta_hd_pref.size,
            n_angles=int(n_angles),
            sigma=float(sigma),
            rng=rng,
        )
    elif population_sampling == "nested_master":
        if master_n_hd_cells is None:
            raise ValueError(
                "nested_master sampling requires heterogeneous_master_n_hd_cells"
            )
        nested_indices = _nested_paired_hd_indices(
            n_hd_cells=theta_hd_pref.size,
            master_n_hd_cells=int(master_n_hd_cells),
        )
        master_currents = _sample_circular_gaussian_process(
            n_samples=int(master_n_hd_cells),
            n_angles=int(n_angles),
            sigma=float(sigma),
            rng=rng,
        )
        currents = master_currents[nested_indices]
    else:
        raise ValueError(
            f"Unknown heterogeneous_population_sampling: {population_sampling}"
        )
    tuning_profiles = _normalized_softplus_tuning(
        currents,
        beta=float(beta),
        bias=float(bias),
    )
    alignment = alignment.lower()
    if alignment != "center_of_mass":
        raise ValueError(f"Unknown heterogeneous_alignment: {alignment}")
    theta_grid = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    circular_moment = tuning_profiles @ np.exp(1j * theta_grid)
    source_centers = np.angle(circular_moment)
    weak_moment = np.abs(circular_moment) <= 1e-12
    if np.any(weak_moment):
        source_centers[weak_moment] = theta_grid[
            np.argmax(tuning_profiles[weak_moment], axis=1)
        ]
    source_query_angles = (
        theta_grid[None, :] - theta_hd_pref[:, None] + source_centers[:, None]
    )
    fractional_indices = np.mod(source_query_angles, 2.0 * np.pi) * int(n_angles) / (
        2.0 * np.pi
    )
    lower_indices = np.floor(fractional_indices).astype(int) % int(n_angles)
    upper_indices = (lower_indices + 1) % int(n_angles)
    upper_weight = fractional_indices - np.floor(fractional_indices)
    aligned_profiles = (
        (1.0 - upper_weight) * np.take_along_axis(tuning_profiles, lower_indices, axis=1)
        + upper_weight * np.take_along_axis(tuning_profiles, upper_indices, axis=1)
    )
    normalization = normalization.lower()
    if normalization == "per_neuron_peak":
        aligned_profiles = aligned_profiles / np.max(aligned_profiles, axis=1, keepdims=True)
    elif normalization != "unit_angular_mean":
        raise ValueError(f"Unknown heterogeneous_normalization: {normalization}")
    if not np.isfinite(aligned_profiles).all() or np.any(aligned_profiles < 0.0):
        raise ValueError("heterogeneous visual profiles must be finite and non-negative")
    return aligned_profiles


def make_heterogeneous_i_vis_to_hd(
    *,
    tuning_profiles: np.ndarray,
    theta_true: float,
    amplitude: float,
    baseline: float,
) -> np.ndarray:
    """Evaluate the heterogeneous teacher manifold at ``theta_true``."""
    tuning_profiles = np.asarray(tuning_profiles, dtype=float)
    if tuning_profiles.ndim != 2:
        raise ValueError("tuning_profiles must have shape (n_theta, n_angles)")
    query_angles = np.full(tuning_profiles.shape[0], float(theta_true))
    return amplitude * _periodic_row_interpolation(tuning_profiles, query_angles) - baseline


def make_i_vis_to_hd(
    *,
    theta_hd_pref: np.ndarray,
    theta_true: float,
    amplitude: float,
    kappa: float,
    baseline: float,
    normalize_peak: bool,
) -> np.ndarray:
    """Return a von-Mises-like visual bump centered on theta_true.

    Vafidis' release code uses ``exp(-sin(delta/2)^2 / (2*sigma^2))`` with
    ``sigma=0.15``.  Near the peak, the matching normalized von-Mises width is
    approximately ``kappa = 1 / (4*sigma^2)``.
    """
    angle_error = circular_difference(theta_hd_pref, theta_true)
    if normalize_peak:
        i_vis_to_hd = amplitude * np.exp(kappa * (np.cos(angle_error) - 1.0))
    else:
        i_vis_to_hd = amplitude * np.exp(kappa * np.cos(angle_error))
    return i_vis_to_hd - baseline


def add_visual_noise(
    i_vis_to_hd: np.ndarray,
    *,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add independent Gaussian current noise to the visual HD drive."""
    i_vis_to_hd = np.asarray(i_vis_to_hd, dtype=float)
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if noise_std == 0.0:
        return i_vis_to_hd.copy()
    return i_vis_to_hd + rng.normal(loc=0.0, scale=noise_std, size=i_vis_to_hd.shape)


@dataclass
class VisualCurrentNoiseProcess:
    """Generate additive current noise for the visual HD drive.

    ``ou_additive`` uses the exact discrete-time OU update with unit stationary
    variance before multiplication by ``std``:

    eta(t + dt) = rho eta(t) + sqrt(1 - rho^2) Normal(0, 1).
    """

    mode: str
    std: float
    shape: tuple[int, ...]
    rng: np.random.Generator
    correlation_time: float
    state: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.mode = self.mode.lower()
        self.std = float(self.std)
        self.correlation_time = float(self.correlation_time)
        if self.std < 0.0:
            raise ValueError("std must be non-negative")
        if self.mode in {"none", "off"}:
            return
        if self.mode not in {"iid_additive", "ou_additive"}:
            raise ValueError(f"Unknown visual noise mode: {self.mode}")
        if self.mode == "ou_additive":
            if self.correlation_time <= 0.0:
                raise ValueError("OU visual noise requires positive correlation_time")
            if self.state is None:
                self.state = self.rng.normal(loc=0.0, scale=1.0, size=self.shape)
            else:
                self.state = np.asarray(self.state, dtype=float).copy()
                if self.state.shape != self.shape:
                    raise ValueError("OU visual noise state shape does not match shape")

    def step(self, dt: float) -> np.ndarray | None:
        """Return the next additive current-noise sample."""
        if self.std == 0.0 or self.mode in {"none", "off"}:
            return None
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.mode == "iid_additive":
            return self.rng.normal(loc=0.0, scale=self.std, size=self.shape)
        if self.state is None:
            raise RuntimeError("OU visual noise state was not initialized")
        rho = float(np.exp(-float(dt) / self.correlation_time))
        innovation_scale = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
        self.state = rho * self.state + innovation_scale * self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=self.shape,
        )
        return self.std * self.state.copy()


def make_zero_i_vis_to_hd(n_theta: int) -> np.ndarray:
    return np.zeros(n_theta, dtype=float)
