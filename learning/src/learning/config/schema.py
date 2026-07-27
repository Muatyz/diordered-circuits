"""Dataclass schema for Vafidis toy-model configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import pi
from typing import Any


VAFIDIS_TAU_S = 0.065
VAFIDIS_HR_INHIBITION = 1.5
VAFIDIS_VELOCITY_GAIN = 1.0 / (2.0 * pi)
VAFIDIS_OU_STD_RAD = 225.0 * pi / 180.0
VAFIDIS_TEST_VELOCITY_RAD = 500.0 * pi / 180.0
VAFIDIS_MODEL_CELLS = 60
VAFIDIS_VISUAL_SIGMA = 0.15
VAFIDIS_VISUAL_KAPPA = 1.0 / (4.0 * VAFIDIS_VISUAL_SIGMA * VAFIDIS_VISUAL_SIGMA)

# Vafidis uses wHD = Aactive / fmax = 2 / 0.15 = 13.33 because rates are in
# kHz with fmax = 0.15. The toy activation is normalized to max 1, so the
# equivalent fixed HD-to-HR voltage scale is Aactive itself.
VAFIDIS_ACTIVE_INPUT_RANGE = 2.0


@dataclass
class ActivationConfig:
    name: str = "sigmoid"
    gain: float = 2.5
    bias: float = 1.0


@dataclass
class ModelInitConfig:
    w_hd_to_hd_mode: str = "zeros"
    w_hr_to_hd_mode: str = "zeros"
    w_hd_to_hd_scale: float = 0.03
    w_hr_to_hd_scale: float = 0.02
    local_sigma: float = 0.45
    random_jitter: float = 0.001


@dataclass
class ModelConfig:
    n_theta: int = VAFIDIS_MODEL_CELLS
    n_hr: int = VAFIDIS_MODEL_CELLS
    tau_s: float = VAFIDIS_TAU_S
    tau_hd_to_hr: float | None = None
    tau_l_hd: float = 0.01
    p_distal_to_proximal: float = 2.0 / 3.0
    b_hd: float = 0.6
    b_hr: float = VAFIDIS_HR_INHIBITION
    w_hd_to_hr_strength: float = VAFIDIS_ACTIVE_INPUT_RANGE
    # ``raw_sum`` preserves legacy runs. ``presynaptic_population_mean``
    # treats dense plastic matrices as intensive continuum kernels and divides
    # each HD, LHR, and RHR pathway by its own presynaptic population size.
    hd_distal_normalization: str = "raw_sum"
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    init: ModelInitConfig = field(default_factory=ModelInitConfig)


@dataclass
class TrainingEarlyStoppingConfig:
    """Optional convergence stop for long online-learning runs.

    ``train_duration`` remains the hard upper bound.  At each check, the
    current intensive HD/HR matrices are compared with a checkpoint one
    ``window_duration`` earlier, while the interval-mean RMS prediction error
    is compared with the corresponding earlier interval.  Requiring several
    consecutive checks prevents a single quiet OU segment from stopping a run.
    """

    enabled: bool = False
    min_duration: float = 8_000.0
    check_interval: float = 500.0
    window_duration: float = 1_000.0
    patience_checks: int = 3
    relative_weight_change_tolerance: float = 0.02
    relative_error_change_tolerance: float = 0.05


@dataclass
class SimulationConfig:
    seed: int = 11
    dt: float = 0.01
    train_duration: float = 160.0
    bump_test_duration: float = 4.0
    darkness_test_duration: float = 6.0
    cue_duration: float = 0.35
    pi_cue_duration: float | None = None
    recue_duration: float = 8.0
    save_interval_steps: int = 40
    weight_snapshot_interval_steps: int | None = None
    theta0: float = 0.0
    progress: bool = True
    plasticity_enabled: bool = True
    # Component-split streams keep the velocity trajectory invariant when a
    # different neuron count consumes a different number of initialization
    # random variates. Legacy configs retain their shared stream by default.
    random_stream_mode: str = "legacy_shared"
    initialization_seed_offset: int = 0
    velocity_seed_offset: int = 10_000
    early_stopping: TrainingEarlyStoppingConfig = field(
        default_factory=TrainingEarlyStoppingConfig
    )


@dataclass
class LearningRuleConfig:
    eta_hd_to_hd: float = 3.0
    eta_hr_to_hd: float = 20.0
    tau_delta: float = 0.10
    w_hd_to_hd_min: float = -1.5
    w_hd_to_hd_max: float = 2.0
    w_hr_to_hd_min: float = -1.5
    w_hr_to_hd_max: float = 1.5
    hd_to_hd_symmetry_mode: str = "none"
    hd_to_hd_balance_mode: str = "none"
    hr_to_hd_balance_mode: str = "none"


@dataclass
class VisualConfig:
    # ``von_mises`` is the release-code teacher.  ``heterogeneous_gaussian_process``
    # uses the Clark et al. wrapped-Gaussian process reproduced in /reproduction.
    profile: str = "von_mises"
    amplitude: float = 2.0
    kappa: float = VAFIDIS_VISUAL_KAPPA
    baseline: float = 1.5
    normalize_peak: bool = True
    light_excitation: float = 0.0
    proximal_scale: float = 1.0
    noise_std: float = 0.0
    noise_process: str = "ou_additive"
    noise_correlation_time: float = VAFIDIS_TAU_S
    apply_noise_during_training: bool = True
    apply_noise_during_visual_test: bool = True
    noise_seed_offset: int = 20_000
    heterogeneous_sigma: float = 1.4
    heterogeneous_beta: float = 2.6057585657926885
    heterogeneous_bias: float = 2.0814271322479385
    heterogeneous_n_angles: int = 256
    heterogeneous_seed_offset: int = 40_000
    heterogeneous_alignment: str = "center_of_mass"
    # Preserve Clark's per-neuron unit-angular-mean generative normalization.
    # Heterogeneous experiments then apply one shared visual amplitude to every
    # generated curve; the experiment YAML calibrates that gain to its control.
    heterogeneous_normalization: str = "unit_angular_mean"
    # ``nested_master`` samples a maximum-size paired-HD population once and
    # selects angularly nested subsets for smaller networks.
    heterogeneous_population_sampling: str = "independent"
    heterogeneous_master_n_hd_cells: int | None = None
    heterogeneous_plot_sample_count: int = 16
    heterogeneous_plot_seed_offset: int = 50_000


@dataclass
class VelocityConfig:
    process: str = "ou"
    mean: float = 0.0
    std: float = VAFIDIS_OU_STD_RAD
    tau: float = 0.50
    clip: float | None = 4.0 * pi
    k_vel: float = VAFIDIS_VELOCITY_GAIN


@dataclass
class TestsConfig:
    darkness_angular_velocity: float = VAFIDIS_TEST_VELOCITY_RAD
    hd_tuning_curve_angles: int = 120
    # Minimum visual-on settling time at each frozen heading.  Optional
    # convergence settings can extend individual headings up to the maximum.
    hd_tuning_curve_settle_duration: float = 0.35
    hd_tuning_curve_max_settle_duration: float | None = None
    hd_tuning_curve_convergence_window: float = 0.20
    hd_tuning_curve_convergence_tolerance: float | None = None
    hd_tuning_curve_sample_count: int = 8
    ou_pi_ensemble_trials: int = 24
    ou_pi_ensemble_seed_offset: int = 60_000
    ou_pi_ensemble_fit_start_time: float = 0.5
    bump_diffusion_duration: float = 10.0
    bump_diffusion_trials: int = 120
    bump_diffusion_test_noise_std: float = 0.1
    bump_diffusion_seed_offset: int = 30_000
    bump_diffusion_fit_start_time: float = 1.0
    bump_diffusion_fit_end_time: float | None = None
    # Deterministic attractor-landscape probe. Uniform cue locations initialize
    # bumps before a long zero-velocity, zero-visual-input darkness interval.
    # It is disabled by default because its cost scales with the number of
    # initial conditions times the darkness duration.
    bump_attractor_trajectory_enabled: bool = False
    bump_attractor_initial_conditions: int = 360
    bump_attractor_duration: float = 60.0
    bump_attractor_cue_duration: float = 4.0
    bump_attractor_sample_interval: float = 0.1
    # Operational Clark-style separation-of-timescales assay.  Normal
    # relaxation is measured after current-space perturbations away from the
    # frozen visual target manifold; tangential motion is measured from the
    # long zero-input attractor trajectories above.
    timescale_separation_enabled: bool = False
    timescale_separation_initial_conditions: int = 12
    timescale_separation_normal_duration: float = 5.0
    timescale_separation_sample_interval: float = 0.02
    timescale_separation_perturbation_scales: list[float] = field(
        default_factory=lambda: [0.025, 0.05, 0.1]
    )
    timescale_separation_perturbations_per_condition: int = 3
    timescale_separation_seed_offset: int = 90_000
    timescale_separation_tangential_threshold_deg: float = 10.0
    timescale_separation_ratio_threshold: float = 10.0
    # Noorman et al. Figure-2-style constant-velocity trajectory sweep.  This
    # resolves low-velocity pinning and the transition to continuous sliding,
    # rather than retaining only one fitted velocity per input as the standard
    # gain curve does.
    velocity_trajectory_sweep_enabled: bool = False
    # Explicit non-negative velocities take precedence over max/count.  The
    # fallback remains for loading older resolved configs.
    velocity_trajectory_sweep_velocities: list[float] = field(default_factory=list)
    velocity_trajectory_sweep_max_velocity: float = 2.0
    velocity_trajectory_sweep_count: int = 11
    velocity_trajectory_sweep_ring_velocity_count: int = 4
    # Deprecated compatibility field retained for older resolved configs.
    velocity_trajectory_sweep_zero_input_flow_window: float = 5.0
    # Dense, short frozen-weight rollouts used to reconstruct F_v(theta) for
    # the velocities shown as rings.  This is separate from the long shared-
    # origin trajectory sweep so phase-space coverage does not depend on
    # whether that one trajectory is pinned.
    velocity_phase_flow_probe_enabled: bool = False
    velocity_phase_flow_initial_conditions: int = 360
    velocity_phase_flow_duration: float = 2.0
    velocity_phase_flow_sample_interval: float = 0.02
    velocity_phase_flow_fit_start_time: float = 0.0
    velocity_phase_flow_fit_duration: float = 2.0
    velocity_phase_flow_angular_bins: int = 360
    velocity_phase_flow_smoothing_bins: int = 5
    velocity_phase_flow_empirical_lambda_speed_floor: float = 0.02
    # Deprecated compatibility fields for older resolved configs.  The direct
    # discrete estimator no longer uses Fourier regression or ridge fitting.
    velocity_phase_flow_fourier_harmonics: int = 12
    velocity_phase_flow_ridge_strength: float = 0.01
    # Requested probe velocities must also occur in the trajectory sweep.
    # An empty list retains the legacy evenly spaced ring selection.
    velocity_phase_flow_probe_velocities: list[float] = field(default_factory=list)
    # Deprecated compatibility field.
    velocity_phase_flow_minimum_regime_r_squared: float = 0.8
    velocity_trajectory_sweep_initial_heading: float = 0.0
    velocity_trajectory_sweep_initial_conditions: int = 1
    velocity_trajectory_sweep_duration: float = 30.0
    velocity_trajectory_sweep_cue_duration: float = 4.0
    velocity_trajectory_sweep_sample_interval: float = 0.05
    velocity_trajectory_sweep_fit_start_time: float = 2.0
    # Deprecated compatibility fields from the removed trajectory-local
    # acceleration classifier.
    velocity_trajectory_sweep_regime_smoothing_time: float = 0.5
    velocity_trajectory_sweep_regime_lambda_threshold: float = 0.02
    velocity_trajectory_sweep_depinning_gain_threshold: float = 0.5
    velocity_trajectory_sweep_depinning_max_stall_fraction: float = 0.2
    velocity_trajectory_sweep_depinning_success_fraction: float = 0.9
    gain_velocities: list[float] = field(
        default_factory=lambda: [
            -VAFIDIS_TEST_VELOCITY_RAD,
            -0.8 * VAFIDIS_TEST_VELOCITY_RAD,
            -0.6 * VAFIDIS_TEST_VELOCITY_RAD,
            -0.4 * VAFIDIS_TEST_VELOCITY_RAD,
            -0.2 * VAFIDIS_TEST_VELOCITY_RAD,
            0.0,
            0.2 * VAFIDIS_TEST_VELOCITY_RAD,
            0.4 * VAFIDIS_TEST_VELOCITY_RAD,
            0.6 * VAFIDIS_TEST_VELOCITY_RAD,
            0.8 * VAFIDIS_TEST_VELOCITY_RAD,
            VAFIDIS_TEST_VELOCITY_RAD,
        ]
    )


@dataclass
class PathsConfig:
    runs_root: str = "runs/vafidis_toy"
    reports_root: str = "reports"


@dataclass
class ExperimentConfig:
    experiment_name: str = "vafidis_toy"
    model: ModelConfig = field(default_factory=ModelConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    learning_rule: LearningRuleConfig = field(default_factory=LearningRuleConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    tests: TestsConfig = field(default_factory=TestsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for field_name, field_value in values.items():
        if not hasattr(instance, field_name):
            raise ValueError(f"Unknown config field: {field_name}")
        current_value = getattr(instance, field_name)
        if hasattr(current_value, "__dataclass_fields__") and isinstance(field_value, dict):
            setattr(instance, field_name, _merge_dataclass(current_value, field_value))
        else:
            setattr(instance, field_name, field_value)
    return instance


def experiment_config_from_dict(config_dict: dict[str, Any]) -> ExperimentConfig:
    model_config = config_dict.get("model")
    if isinstance(model_config, dict) and "n_theta" in model_config and "n_hr" not in model_config:
        model_config["n_hr"] = model_config["n_theta"]
    config = ExperimentConfig()
    return _merge_dataclass(config, config_dict)
