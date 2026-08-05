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
VAFIDIS_MAX_FIRING_RATE_KHZ = 0.15
VAFIDIS_VISUAL_SIGMA = 0.15
VAFIDIS_VISUAL_KAPPA = 1.0 / (4.0 * VAFIDIS_VISUAL_SIGMA * VAFIDIS_VISUAL_SIGMA)
VAFIDIS_PROXIMAL_CAPACITANCE = 0.001
VAFIDIS_PROXIMAL_LEAK_CONDUCTANCE = 1.0
VAFIDIS_DISTAL_TO_PROXIMAL_CONDUCTANCE = 2.0

# Voltage/current amplitude used before multiplying by the inverse firing-rate
# unit. Authored Vafidis configs explicitly use
# wHD = VAFIDIS_ACTIVE_INPUT_RANGE / VAFIDIS_MAX_FIRING_RATE_KHZ.
VAFIDIS_ACTIVE_INPUT_RANGE = 2.0

# Old resolved configs may contain these fields from diagnostics that have
# since been removed.  They are ignored on load rather than kept in every new
# config and resolved output indefinitely.
RETIRED_TEST_CONFIG_FIELDS = frozenset(
    {
        "bump_attractor_trajectory_enabled",
        "slow_manifold_enabled",
        "timescale_separation_enabled",
        "velocity_trajectory_sweep_enabled",
        "velocity_phase_flow_probe_enabled",
        "velocity_trajectory_sweep_zero_input_flow_window",
        "velocity_phase_flow_fourier_harmonics",
        "velocity_phase_flow_ridge_strength",
        "velocity_phase_flow_minimum_regime_r_squared",
        "velocity_trajectory_sweep_regime_smoothing_time",
        "velocity_trajectory_sweep_regime_lambda_threshold",
    }
)


@dataclass
class ActivationConfig:
    name: str = "sigmoid"
    gain: float = 2.5
    bias: float = 1.0
    # Kept at one in the bare schema so historical resolved configs that did
    # not record this newly explicit field retain their normalized-rate
    # semantics. Every authored scientific experiment sets the release value
    # 0.15 kHz explicitly.
    max_rate: float = 1.0


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
    # Vafidis et al. Table 1 and Eq. 4. Time is represented in seconds, so
    # the paper's proximal capacitance C=1 ms is stored as 0.001 s.
    c_hd_proximal: float = VAFIDIS_PROXIMAL_CAPACITANCE
    g_l_hd_proximal: float = VAFIDIS_PROXIMAL_LEAK_CONDUCTANCE
    g_d_hd_to_proximal: float = VAFIDIS_DISTAL_TO_PROXIMAL_CONDUCTANCE
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
    # The release implementation uses 0.5 ms forward-Euler steps. This is also
    # safely inside the Eq. 4 stability interval for C=1 ms, gL=1, gD=2.
    dt: float = 0.0005
    # Historical/resolved configs without this field retain the released
    # forward-Euler semantics. Active training configs may opt into the exact
    # linear Eq. (4) substep explicitly.
    proximal_integration_method: str = "forward_euler"
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
    # ``None`` disables the corresponding bound.  This is needed for the
    # release-parameter baseline because the published LearnPI code does not
    # clip plastic weights during online learning.
    w_hd_to_hd_min: float | None = -1.5
    w_hd_to_hd_max: float | None = 2.0
    w_hr_to_hd_min: float | None = -1.5
    w_hr_to_hd_max: float | None = 1.5
    hd_to_hd_symmetry_mode: str = "none"
    hd_to_hd_balance_mode: str = "none"
    hr_to_hd_balance_mode: str = "none"
    # The released model learns the full recurrent matrix, including its
    # diagonal. Setting this true is an explicitly non-paper control.
    zero_hd_to_hd_diagonal: bool = False


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
    # Optional multiplier on the current entering Eq. 4. The Vafidis baseline
    # uses 1.0; the old quasi-steady toy used 1/(gD+gL), which must not be used
    # together with the now-explicit proximal voltage dynamics.
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
class DiagnosticsConfig:
    bump_maintenance: bool = False
    path_integration_and_pi_error: bool = False
    pva_spectrum_and_visualization: bool = False
    velocity_gain: bool = False
    trajectory_and_fixed_points: bool = False
    weight_snapshots_and_development: bool = False
    bump_diffusion: bool = False
    timescale_separation: bool = False
    velocity_dynamics_and_phase_flow: bool = False
    reuse_cached_dependencies: bool = True


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
    bump_attractor_initial_conditions: int = 360
    bump_attractor_duration: float = 60.0
    bump_attractor_cue_duration: float = 4.0
    bump_attractor_sample_interval: float = 0.1
    # Ságodi-style autonomous slow-ring diagnostics reuse the sampled
    # zero-input bump trajectories. Candidate points satisfy
    # ||f(x)|| <= speed_fraction * max_trajectory ||f||.
    slow_manifold_speed_fraction: float = 1e-3
    slow_manifold_candidate_count: int = 1024
    slow_manifold_angular_bins: int = 180
    slow_manifold_min_angular_support_fraction: float = 0.5
    slow_manifold_jacobian_anchors: int = 24
    slow_manifold_jacobian_eigenvalues: int = 6
    slow_manifold_jacobian_dense_dimension_limit: int = 256
    # Ramesan-style slow-point criterion q(x)=0.5*||F(x)||^2.  Its absolute
    # scale depends on state units, so the figure also reports continuous q.
    ramesan_q_threshold: float = 1e-4
    # Retain a bounded, uniformly subsampled collection of full autonomous
    # states along the zero-input trajectories.  These states support a
    # trajectory-conditioned phase-flow landscape without treating the
    # visually defined probe ring as an invariant manifold.
    ramesan_trajectory_sample_count: int = 0
    ramesan_phase_angular_bins: int = 180
    ramesan_phase_smoothing_bins: int = 5
    # Optional tube probe around the retained trajectory states.  Perturbation
    # scales are RMS distances after blockwise state standardization; random
    # directions are projected normal to the cue-ring tangent before mapping
    # back to the physical canonical state.
    ramesan_ambient_enabled: bool = False
    ramesan_ambient_sample_count: int = 0
    ramesan_ambient_perturbation_scales: list[float] = field(
        default_factory=lambda: [0.01, 0.03, 0.1]
    )
    ramesan_ambient_seed_offset: int = 110_000
    # Operational Clark-style separation-of-timescales assay.  Normal
    # relaxation is measured after current-space perturbations away from the
    # frozen visual target manifold; tangential motion is measured from the
    # long zero-input attractor trajectories above.
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
    # Explicit non-negative velocities take precedence over max/count.  An
    # empty list uses the max/count grid below.
    velocity_trajectory_sweep_velocities: list[float] = field(default_factory=list)
    velocity_trajectory_sweep_max_velocity: float = 2.0
    velocity_trajectory_sweep_count: int = 11
    velocity_trajectory_sweep_ring_velocity_count: int = 4
    # Dense, short frozen-weight rollouts used to reconstruct F_v(theta) for
    # the velocities shown as rings.  This is separate from the long shared-
    # origin trajectory sweep so phase-space coverage does not depend on
    # whether that one trajectory is pinned.
    velocity_phase_flow_initial_conditions: int = 360
    velocity_phase_flow_duration: float = 2.0
    velocity_phase_flow_sample_interval: float = 0.02
    velocity_phase_flow_fit_start_time: float = 0.0
    velocity_phase_flow_fit_duration: float = 2.0
    velocity_phase_flow_angular_bins: int = 360
    velocity_phase_flow_smoothing_bins: int = 5
    velocity_phase_flow_empirical_lambda_speed_floor: float = 0.02
    # Requested probe velocities must also occur in the trajectory sweep.
    # An empty list selects evenly spaced rings from that sweep.
    velocity_phase_flow_probe_velocities: list[float] = field(default_factory=list)
    velocity_trajectory_sweep_initial_heading: float = 0.0
    velocity_trajectory_sweep_initial_conditions: int = 1
    velocity_trajectory_sweep_duration: float = 30.0
    velocity_trajectory_sweep_cue_duration: float = 4.0
    velocity_trajectory_sweep_sample_interval: float = 0.05
    velocity_trajectory_sweep_fit_start_time: float = 2.0
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
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
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
    config_dict = dict(config_dict)
    model_config = config_dict.get("model")
    if isinstance(model_config, dict):
        model_config = dict(model_config)
        config_dict["model"] = model_config
        if "n_theta" in model_config and "n_hr" not in model_config:
            model_config["n_hr"] = model_config["n_theta"]
    tests_config = config_dict.get("tests")
    if isinstance(tests_config, dict):
        tests_config = dict(tests_config)
        for retired_field in RETIRED_TEST_CONFIG_FIELDS:
            tests_config.pop(retired_field, None)
        config_dict["tests"] = tests_config
    config = ExperimentConfig()
    return _merge_dataclass(config, config_dict)
