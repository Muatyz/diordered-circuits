# Diagnostics architecture

Diagnostics are selected by the boolean groups in
`configs/diagnostics/*.yaml`. `learning.config.diagnostics` is the registry
that expands a group into reusable diagnostic jobs. Experiment YAML owns
training, model and weight-snapshot storage; diagnostics YAML owns test
interventions and their sampling density. The lightweight
`training_convergence` group is also enabled explicitly in every authored
experiment because its online windows cannot be reconstructed after training.

## Shared execution path

Frozen diagnostic jobs for both a newly trained model and
`test_vafidis_toy --run-dir ...` pass through `run_all_tests`. They consume the
same resolved `ExperimentConfig`, the same frozen `VafidisToyState`, and, when
required, saved run artifacts. They do not update weights. The online training
error is instead accumulated inside the one shared training loop. Figure
generation only reads saved artifacts and does not rerun dynamics.

## Figure output layout

Figure directories mirror the canonical boolean group names in the resolved
diagnostics config:

```text
figures/
  bump_maintenance/
  path_integration_and_pi_error/
  pva_spectrum_and_visualization/
  velocity_gain/
  training_convergence/
  trajectory_and_fixed_points/
  weight_snapshots_and_development/
  bump_diffusion/
  timescale_separation/
  velocity_dynamics_and_phase_flow/
  numerical_convergence/
```

Only enabled groups are created. The constant-velocity, single-OU and
OU-ensemble figure jobs all write to `path_integration_and_pi_error/`, even
though they retain separate entries in `figure_status.json`. Each status entry
records its canonical `output_dir`. The standalone `--phase-flow-only` path
writes to `velocity_dynamics_and_phase_flow/` as well.

This layout is non-destructive for historical runs. Regenerating figures
writes the canonical group directories but does not move or remove legacy
`activity/`, `diagnostics/`, `gain/`, `heading/` or `weights/` directories.

Fresh frozen protocols use
`learning.diagnostics.protocols.initialize_frozen_protocol_state`. It resets
dynamic state reproducibly while copying the learned plastic matrices, fixed
HD-to-HR projection and visual tuning profiles from the tested state.

Online behavioral checkpoint selection and offline weight-snapshot ranking
also share `run_frozen_velocity_probe_grid`. The grid can keep velocity active
during the visual cue while arranging every trial to enter darkness at the
configured release heading. It reports RMS and maximum velocity bias,
zero-input drift, bump quality, stall fraction, and the smallest command that
de-pins every sampled heading in each turning direction. Thus training-time
selection and the final darkness diagnostics no longer use different rollout
semantics.

## Training convergence

The `training_convergence` group expands to `learning_error_development`.
During the shared training loop, `TrainingAbsoluteLearningErrorRecorder`
computes the discrete Vafidis Equation (19) statistic

```text
mean_{neuron,time in window} |f(V_a) - f(V_ss)|.
```

Defaults match the released analysis: a 10 s forward window begins at every
1% of the requested training duration. Only active windows accumulate
per-neuron sums, so no full-resolution error trajectory is retained. Authored
Vafidis experiments store rates in kHz and configure a factor of 1000 for the
paper's spikes/s plotting unit.

Outputs are fields prefixed `absolute_learning_error_` in
`training_history.npz` and
`figures/training_convergence/training_absolute_learning_error.png`. Historical runs
without these online fields remain readable, but the exact windowed statistic
cannot be reconstructed from their sparse RMS samples.

## Weight snapshots and development

The `weight_snapshots_and_development` group expands to:

- `weight_structure`: final learned-weight summaries;
- `weight_snapshot_pi_development`: constant-velocity darkness performance
  for frozen weights selected from `weight_history.npz`.

`simulation.weight_snapshot_interval_duration` (seconds) controls which weights are saved
during training. `tests.weight_snapshot_pi_interval_fraction` controls which
of those saved states are evaluated. A value of `0.01` requests the state
nearest every 1% of the saved training span, including initial and final
states. A diagnostic cannot reconstruct a weight state that was not saved, so
the training cadence must be at least as dense as the requested evaluation
cadence.

For snapshot `s` and commanded velocity `v`, the primary statistic is

```text
mean_t |(unwrap(decoded_heading(t)) - decoded_heading(0))
        - (unwrap(true_heading(t)) - true_heading(0))|.
```

The initial offset is removed so the statistic measures accumulated path-
integration error rather than visual-cue alignment. The unwrapped primary does
not fold a multi-turn error back into `[-pi, pi)`; wrapped error is saved as an
explicit companion view. Velocity bias, PVA strength and bump contrast are
also retained. All probe steps use `training=False` and no visual input after
cue removal.

`tests.weight_snapshot_pi_selection_metric` makes checkpoint semantics
explicit. `mean_abs_unwrapped_error` preserves phase-error selection;
`rms_velocity_bias` selects a horizon-independent average gain criterion;
`maximum_abs_velocity_bias` selects the worst sampled heading/velocity; and
`depinning_velocity` directly targets low-speed pinning. Acceptance thresholds
are applied before score minimization. If no snapshot passes, the lowest-score
fallback is still saved but `selection_was_fallback=1`, so a failed model is
never silently relabeled as behaviorally satisfactory. Weight-norm growth,
the HR/HD norm ratio and realized training-time pathway-current RMS are plotted
beside frozen PI performance. These are diagnostics and do not silently
regularize either plastic pathway.

The opt-in `configs/profiles/pi_robust_vafidis.yaml` uses the same Vafidis
learning rule with a broad-to-low-speed OU standard-deviation schedule and a
dense moving-cue selection grid. `pi_robust_n120.yaml` is a composable
finite-size control that doubles both populations and rescales only the random
initialization. Neither profile imposes circulant weights, symmetry, decay, or
an additional objective.

Outputs:

- `weight_snapshot_pi_development.npz`;
- `test_metrics.json` fields prefixed `weight_snapshot_pi_`;
- `figures/weight_snapshots_and_development/training_snapshot_frozen_pi_error.png`.

The velocity grid, initial heading, 1% interval, cue duration, darkness
duration and averaging-window start are diagnostics parameters. Constant
commands are used for matched low-variance comparisons across training time;
the OU single-trial and ensemble jobs remain the naturalistic final PI check.

## Path integration and PI error

Constant-velocity, single-OU and OU-ensemble jobs share the release-relative
error definition above. Absolute circular decoder error remains available for
cue-alignment questions, but is not used as accumulated PI error. Constant
commands report decoded velocity and drift prediction as well as phase error.
OU ensembles separately save the systematic mean drift, across-trial error
variance and the Vafidis-style effective diffusion trace.

## Trajectories and fixed points

The canonical PVA output is split into three PNG files so cue settling and
autonomous dynamics never share ambiguous coordinates:

- `figures/trajectory_and_fixed_points/bump_attractor_pva_initial_cue_endpoint_map.png`
  contains the bump
  trajectories and darkness endpoint versus nominal initial cue;
- `figures/trajectory_and_fixed_points/bump_attractor_pva_release_angle_endpoint_map.png`
  contains the same bump
  trajectories and the autonomous release-to-endpoint map;
- `figures/trajectory_and_fixed_points/bump_attractor_pva_cue_transfer.png`
  contains release angle versus initial
  cue alongside the circular residual `wrap(release-initial)` (ideal zero).

Peak and Clark-overlap decoders remain in the artifact for quality checks but
are not mixed into these PVA figures.

Fixed points are roots of the periodic endpoint displacement
`D(phi)=wrap(E_T(phi)-phi)` in the actual cue-release coordinate. A
positive-to-negative crossing is stable and a negative-to-positive crossing
is unstable. Periodic seam handling, unresolved large gaps, cue-transfer
orientation/coverage and stable/unstable alternation mismatch are all saved;
the implementation never forces equal root counts. The release-angle endpoint
figure shows roots in their autonomous coordinate. The initial-cue endpoint
figure shows the corresponding preimages under the measured cue-transfer map.

## Slow manifold and timescale separation (audit and rework 2026-08-23)

The `pva_spectrum_and_visualization` (Ramesan/slow-ring), `slow_manifold` and
`timescale_separation` groups were disabled after an audit because their
first implementation conflated several distinct quantities.  They are still
off by default in the experiment/diagnostics YAMLs; the code below is the
reworked version to re-enable after a full-training check.

### What is actually measured

- `q(x) = 0.5 ||F_dt(x)||^2` and the Jacobians are evaluated on the full
  canonical Markov state (`4*N_HD + N_HR`), never on a PCA projection.
  PC1-3 are a visualization only.
- A numeric check on the N=120 trained network showed the leading Jacobian
  spectrum at cue-release states is NOT polluted by the algebraic HR block:
  `lambda_1 ~ -1.8..-10 /s` and `lambda_3 ~ -12 /s`; the HR `-1/dt` mode does
  not enter the leading spectrum.  The absolute `q` scale (`~1e7` at cue
  release) is dominated by the fast HD->HR low-pass difference, which makes
  any fixed absolute `ramesan_q_threshold` meaningless.  Slow points are
  selected per trajectory (see below); the threshold is reported but never
  treated as a universal constant.

### Rework 1: physical slow-set floor and time-uniform candidates

`select_slow_candidate_indices` now takes an optional `speed_floor` (rad/s)
and `time`.  The effective threshold is
`min(speed_fraction * max_trajectory_speed, speed_floor)` and, when more
points qualify than the budget, candidates are re-sampled uniformly in time
instead of by index.  This fixes two artifacts:

1. the trajectory maximum is set by the initial relaxation transient
   (`|F| ~ 7e3 /s`), so `1e-3 * max` alone admits mid-relaxation points and
   biases the slow set toward the late-time basin;
2. with a physical floor below the pinning barrier, the candidate set probes
   the attracting set directly: full angular coverage means the attractors
   tile the ring (continuous-like), isolated clusters mean discrete basins.

Configure with `tests.slow_manifold_speed_floor` (default `null` = old
relative-only behavior).  The dev diagnostics YAML uses `0.0005` rad/s
(~0.03 deg/s).

### Rework 2: settled vs moving phase decomposition

`analyze_ramesan_phase_landscape` splits the decoded-PVA phase velocity into
a *settled* fraction (frames with `|v| < tests.ramesan_phase_velocity_floor`,
default 1e-3 rad/s) and a *moving* velocity (median over the remaining
frames).  The raw within-bin median mixes many zeros at a fixed point with
single-frame jumps when the bump moves across the 0.1 s trajectory grid;
the decomposition reports the time spent at each phase (attracting phases
have settled fraction near one) and the actual drift while crossing.  The
smoothed flow, effective potential, and fixed-point roots now use the moving
velocity field.

### Rework 3: timescale separation

`run_timescale_separation_test` keeps the Clark-style normal relaxation
assay (HD distal current perturbed away from the closed visual-target
manifold) and the tangential Clark-overlap first-passage time.  The
dev diagnostics YAML reduces `timescale_separation_tangential_threshold_deg`
from 10 to 3: the N=120 heterogeneous network has median stable-basin width
~10 deg, so 10 deg is crossed almost immediately and inflates the
tangential-to-normal ratio.  Report `tangential_passage_fraction` alongside
the conservative ratio.

### Recommended re-enable checklist

1. run the two N=120 experiments to completion (von-Mises and
   heterogeneous), then compare `slow_manifold_eta_theta_deg_s` (max tangent
   phase flow), `slow_manifold_spectral_gap_min`, `slow_mode_tangent_alignment_median`,
   basin entropy, FP count and the settled-fraction map across the pair;
2. keep `ramesan_phase_angular_bins = 360` and
   `ramesan_phase_smoothing_bins = 1` so the 12 deg alternation is resolved;
3. only then re-enable `pva_spectrum_and_visualization`,
   `slow_manifold` and `timescale_separation` in a diagnostics YAML.

## Numerical convergence

The `numerical_convergence` group expands to one reusable whole-step audit in
`learning.diagnostics.numerical_convergence`. A high-resolution
`exact_linear` cue first constructs one shared, well-defined release Markov
state. Every configured `(dt, proximal method)` then evolves a copy under the
same deterministic darkness intervention and is compared at common physical
sample times.

The artifact `numerical_convergence_history.npz` contains phase, HD-rate,
proximal/distal-voltage and HD-to-HR-low-pass errors, Eq. (4) homogeneous
amplification factors and per-row pass flags. The diagnostic also reports the
effective physical time constant implied by the released HD-to-HR line that
omitted `dt`; that value is parity metadata only. Production dynamics continue
to implement the paper's `dt/tau_s` Equation (9) update. `exact_linear` refers
only to the proximal Equation (4) substep, never to the full coupled solver.
