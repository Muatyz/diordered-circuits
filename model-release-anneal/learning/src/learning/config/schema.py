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

TRAINING_INTEGRATION_SINGLE_CLOCK = "single_clock"
TRAINING_INTEGRATION_BLOCK_MULTIRATE = "block_multirate"
TRAINING_INTEGRATION_METHODS = frozenset(
    {
        TRAINING_INTEGRATION_SINGLE_CLOCK,
        TRAINING_INTEGRATION_BLOCK_MULTIRATE,
    }
)

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
class TrainingCheckpointSelectionConfig:
    """Optional frozen-weight behavioral model selection during training.

    This is deliberately separate from :class:`TrainingEarlyStoppingConfig`:
    prediction-error/weight convergence and useful path-integration behavior
    are different claims.  Validation never updates weights.  A run may stop
    for lack of validation improvement without satisfying the behavioral
    acceptance thresholds, and that outcome remains explicitly recorded.
    """

    enabled: bool = False
    min_duration: float = 8_000.0
    check_interval: float = 800.0
    patience_checks: int = 8
    success_checks: int = 3
    restore_best: bool = True
    minimum_improvement: float = 1e-4
    cue_duration: float = 1.0
    probe_duration: float = 2.0
    fit_start_time: float = 0.5
    # If true, the visual cue moves at the probe velocity and is positioned so
    # darkness still begins at each configured initial heading.  This matches
    # the moving-cue state distribution used by the final PI protocol.
    velocity_during_cue: bool = False
    selection_metric: str = "rms_velocity_bias"
    minimum_moving_gain: float = 0.5
    velocities: list[float] = field(
        default_factory=lambda: [
            -1.3089969389957472,
            -0.5235987755982988,
            0.0,
            0.5235987755982988,
            1.3089969389957472,
        ]
    )
    initial_headings: list[float] = field(
        default_factory=lambda: [
            -3.141592653589793,
            -1.5707963267948966,
            0.0,
            1.5707963267948966,
        ]
    )
    maximum_rms_velocity_bias: float = 0.01
    maximum_abs_velocity_bias: float = 0.02
    maximum_abs_zero_velocity_drift: float = 0.01
    maximum_depinning_velocity: float = float("inf")
    minimum_pva_strength: float = 0.5
    minimum_bump_contrast: float = 0.05


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
    # ``single_clock`` preserves the release-aligned ordered plasticity update
    # at every neural timestep. ``block_multirate`` keeps the neural timestep
    # unchanged, freezes plastic weights within a short physical-time block,
    # and integrates the accumulated local E P^T signal at the block boundary.
    training_integration_method: str = TRAINING_INTEGRATION_SINGLE_CLOCK
    plasticity_update_interval_duration: float = 0.01
    train_duration: float = 160.0
    bump_test_duration: float = 4.0
    darkness_test_duration: float = 6.0
    cue_duration: float = 0.35
    pi_cue_duration: float | None = None
    recue_duration: float = 8.0
    # User-facing sampling intervals are physical durations in seconds.  The
    # training loop converts them to integer steps using ``dt``.
    save_interval_duration: float = 0.02
    weight_snapshot_interval_duration: float | None = None
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
    checkpoint_selection: TrainingCheckpointSelectionConfig = field(
        default_factory=TrainingCheckpointSelectionConfig
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
    # Optional training-time amplitude curriculum for the same visual teacher.
    # Entries are mappings with increasing ``end_fraction`` (ending at 1.0)
    # and a non-negative ``amplitude`` (same units as ``visual.amplitude``).
    # During training the effective teacher amplitude follows this piecewise-
    # constant schedule; frozen-weight diagnostics keep using the configured
    # ``amplitude`` / ``bump_attractor_cue_amplitude`` unchanged.  This is the
    # "night-vision" protocol: annealing the cue from strong to weak exposes
    # the network to near-darkness during late training so the learned
    # recurrent weights must sustain the bump autonomously.
    training_amplitude_schedule: list[dict[str, float]] = field(default_factory=list)
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
    # Optional curriculum for the same OU process and Vafidis learning rule.
    # Entries are mappings with increasing ``end_fraction`` (ending at 1.0)
    # and a non-negative ``std`` in rad/s.
    training_ou_std_schedule: list[dict[str, float]] = field(default_factory=list)


@dataclass
class DiagnosticsConfig:
    # Weight matrix used by all frozen-weight diagnostics. ``best`` minimizes
    # the explicitly configured snapshot PI selection metric.
    # ``snapshot`` selects a zero-based entry from weight_history.npz.
    weight_source: str = "best"
    weight_snapshot_index: int | None = None
    bump_maintenance: bool = False
    path_integration_and_pi_error: bool = False
    pva_spectrum_and_visualization: bool = False
    velocity_gain: bool = False
    training_convergence: bool = False
    trajectory_and_fixed_points: bool = False
    weight_snapshots_and_development: bool = False
    bump_diffusion: bool = False
    timescale_separation: bool = False
    velocity_dynamics_and_phase_flow: bool = False
    numerical_convergence: bool = False
    reuse_cached_dependencies: bool = True


@dataclass
class TestsConfig:
    darkness_angular_velocity: float = VAFIDIS_TEST_VELOCITY_RAD
    # A compact balanced grid for comparing constant-velocity PI error across
    # both speed magnitude and turning direction.  The scalar above remains
    # the backward-compatible primary trace and summary.
    constant_pi_velocities: list[float] = field(
        default_factory=lambda: [
            -75.0 * pi / 180.0,
            -30.0 * pi / 180.0,
            30.0 * pi / 180.0,
            75.0 * pi / 180.0,
        ]
    )
    # Vafidis Eq. (19): population/time mean absolute firing-rate prediction
    # error, sampled in forward windows beginning at fixed training fractions.
    learning_error_window_duration: float = 10.0
    learning_error_interval_fraction: float = 0.01
    # Authored Vafidis experiments store rates in kHz (ms^-1); multiplying by
    # 1000 reports the paper's spikes/s plotting unit without normalizing fmax.
    learning_error_rate_scale_to_spikes_per_second: float = 1000.0
    # Frozen-weight performance across saved training snapshots.  These probes
    # intentionally use deterministic constant velocities: OU trajectories are
    # retained for the final PI ensemble, while matched commands make changes
    # across training time directly attributable to the weights.
    weight_snapshot_pi_velocities: list[float] = field(
        default_factory=lambda: [
            -75.0 * pi / 180.0,
            -30.0 * pi / 180.0,
            30.0 * pi / 180.0,
            75.0 * pi / 180.0,
        ]
    )
    # Empty preserves the historical single-heading probe below.  Supplying a
    # list makes checkpoint ranking average over several cue-release phases,
    # reducing the chance of selecting a snapshot that happens to perform well
    # in only one autonomous basin.
    weight_snapshot_pi_initial_headings: list[float] = field(default_factory=list)
    weight_snapshot_pi_initial_heading: float = 0.0
    # Evaluate weights nearest to 0%, 1%, ..., 100% of the saved training span.
    # The training snapshot cadence must be at least this dense; diagnostics
    # cannot reconstruct weight states that were never saved.
    weight_snapshot_pi_interval_fraction: float = 0.01
    weight_snapshot_pi_cue_duration: float = 1.0
    weight_snapshot_pi_duration: float = 5.0
    weight_snapshot_pi_average_start_time: float = 0.5
    weight_snapshot_pi_velocity_during_cue: bool = False
    weight_snapshot_pi_minimum_moving_gain: float = 0.5
    # Keep checkpoint selection explicit: accumulated phase error preserves
    # current behavior, while rms_velocity_bias targets horizon-independent
    # integrator gain without silently changing existing runs.
    weight_snapshot_pi_selection_metric: str = "mean_abs_unwrapped_error"
    # Acceptance gates are applied before minimizing the selection metric.
    # Infinite maxima and zero minima preserve historical ranking unless an
    # authored profile opts into stricter behavioral acceptance.
    weight_snapshot_pi_maximum_rms_velocity_bias: float = float("inf")
    weight_snapshot_pi_maximum_abs_velocity_bias: float = float("inf")
    weight_snapshot_pi_maximum_abs_zero_velocity_drift: float = float("inf")
    weight_snapshot_pi_maximum_depinning_velocity: float = float("inf")
    weight_snapshot_pi_minimum_pva_strength: float = 0.0
    weight_snapshot_pi_minimum_bump_contrast: float = 0.0
    # Whole-step deterministic convergence audit. The high-resolution
    # exact-linear row is used only as a numerical reference; it does not
    # redefine release-code parity or the configured training integrator.
    numerical_convergence_dt_values: list[float] = field(
        default_factory=lambda: [0.001, 0.0005, 0.00025, 0.000125]
    )
    numerical_convergence_methods: list[str] = field(
        default_factory=lambda: ["forward_euler", "exact_linear"]
    )
    numerical_convergence_reference_dt: float = 0.0000625
    numerical_convergence_cue_duration: float = 0.5
    numerical_convergence_duration: float = 2.0
    numerical_convergence_sample_interval: float = 0.01
    numerical_convergence_angular_velocity: float = VAFIDIS_TEST_VELOCITY_RAD
    numerical_convergence_max_heading_error_deg: float = 1.0
    numerical_convergence_max_rate_rms_error: float = 0.002
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
    # Optional angular-velocity clip (rad/s) applied only to the frozen-weight
    # OU darkness tests, independent of the training OU clip.  The original
    # LearnPI PI-error test calls ``gen_theta0_OU(..., bound_vel=True,
    # v_max=500)`` (i.e. 500 deg/s), while the toy default inherits
    # ``velocity.clip`` (720 deg/s).  Setting this to ``8.726646259971648``
    # (500 deg/s) reproduces the paper's test-time velocity range without
    # changing the training OU process.
    ou_test_clip: float | None = None
    # Original LearnPI ``stability.py`` protocol: 10 s total simulation with
    # a 2 s strong visual cue followed by 8 s of zero-velocity darkness.
    bump_diffusion_duration: float = 10.0
    bump_diffusion_cue_duration: float = 2.0
    bump_diffusion_cue_amplitude: float = 16.0
    bump_diffusion_cue_sigma: float = 0.25
    bump_diffusion_release_skip_steps: int = 5
    bump_diffusion_integration_method: str = "forward_euler"
    bump_diffusion_trials: int = 1000
    bump_diffusion_test_noise_stds: list[float] = field(
        default_factory=lambda: [0.1 * index for index in range(11)]
    )
    # Primary noise level used for backward-compatible scalar metrics and the
    # detailed variance-versus-time figure. It must occur in the sweep above.
    bump_diffusion_test_noise_std: float = 0.1
    bump_diffusion_seed_offset: int = 30_000
    bump_diffusion_fit_start_time: float = 1.0
    bump_diffusion_fit_end_time: float | None = None
    # Deterministic attractor-landscape probe. Uniform cue locations initialize
    # bumps before a long zero-velocity, zero-visual-input darkness interval.
    bump_attractor_initial_conditions: int = 360
    # Refine every adjacent endpoint-map transition by repeatedly probing the
    # midpoint on the visual-cue initialization curve.  Three levels turn the
    # default one-degree coarse spacing into a nominal 0.125-degree bracket.
    bump_attractor_boundary_bisection_depth: int = 3
    bump_attractor_duration: float = 60.0
    bump_attractor_cue_duration: float = 1.0
    # Optional diagnostic-only cue magnitude. None reuses visual.amplitude so
    # frozen-weight basin initialization can be tuned without changing training.
    bump_attractor_cue_amplitude: float | None = None
    bump_attractor_sample_interval: float = 0.1
    # Ságodi-style autonomous slow-ring diagnostics reuse the sampled
    # zero-input bump trajectories. Candidate points satisfy
    # ||f(x)|| <= min(speed_fraction * max_trajectory ||f||, speed_floor)
    # when speed_floor is provided.  speed_floor is a physical speed (rad/s)
    # below the pinning barrier; for networks whose bump relaxes onto
    # discrete attractors it keeps the slow set close to the attracting set
    # instead of admitting mid-relaxation points (the trajectory maximum is
    # set by the initial transient and makes the relative threshold alone too
    # permissive).
    slow_manifold_speed_fraction: float = 1e-3
    slow_manifold_speed_floor: float | None = None
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
    # Frames with decoded-PVA phase speed below this physical floor (rad/s)
    # are counted as *settled* (time spent at an attracting phase).  The
    # remaining moving frames define the empirical phase flow.  The raw
    # within-bin median mixes the two regimes because the 0.1 s trajectory
    # grid samples a fixed point as many zeros and a moving bump as single
    # frame jumps.
    ramesan_phase_velocity_floor: float = 1e-3
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
