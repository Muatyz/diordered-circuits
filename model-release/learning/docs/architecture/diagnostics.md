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

Fresh frozen protocols use
`learning.diagnostics.protocols.initialize_frozen_protocol_state`. It resets
dynamic state reproducibly while copying the learned plastic matrices, fixed
HD-to-HR projection and visual tuning profiles from the tested state.

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
`figures/diagnostics/training_absolute_learning_error.png`. Historical runs
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
`rms_velocity_bias` selects a horizon-independent gain criterion. Weight-norm
growth, the HR/HD norm ratio and realized training-time pathway-current RMS
are plotted beside frozen PI performance. These are diagnostics and do not
silently regularize either plastic pathway.

Outputs:

- `weight_snapshot_pi_development.npz`;
- `test_metrics.json` fields prefixed `weight_snapshot_pi_`;
- `figures/weights/training_snapshot_frozen_pi_error.png`.

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

The canonical PVA figure contains exactly three panels: autonomous darkness
trajectories, initial cue angle versus relaxed cue-release angle, and initial
cue angle versus darkness endpoint. Peak and Clark-overlap decoders remain in
the artifact for quality checks but are not mixed into the main figure.

Fixed points are roots of the periodic endpoint displacement
`D(phi)=wrap(E_T(phi)-phi)` in the actual cue-release coordinate. A
positive-to-negative crossing is stable and a negative-to-positive crossing
is unstable. Periodic seam handling, unresolved large gaps, cue-transfer
orientation/coverage and stable/unstable alternation mismatch are all saved;
the implementation never forces equal root counts.

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
