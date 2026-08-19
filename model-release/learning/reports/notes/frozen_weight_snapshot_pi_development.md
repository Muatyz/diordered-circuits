# Frozen-weight PI performance across training snapshots

## Literature mapping

Vafidis et al. train with an Ornstein-Uhlenbeck angular-velocity process
(Methods Eq. 17). Their naturalistic darkness PI example and the 1000-trial,
60 s error distribution use variable trajectories. The velocity-gain assay in
Figure 2C instead holds each commanded velocity constant for 5 s.

Equation 19 averages the absolute *local learning error* over neurons and a
10 s time window. It is not a frozen-weight PI metric. The present diagnostic
borrows only its time-window averaging idea and names the new quantity
explicitly: time-averaged absolute circular accumulated PI error.

## Diagnostic definition

For each selected training snapshot, visual input first initializes the bump
at a stationary heading. We then remove visual input, freeze both plastic
matrices, apply one constant angular velocity and measure

```text
e_PI(t) = wrap(Delta theta_decoded(t) - Delta theta_true(t)).
```

The primary score is `mean(abs(e_PI))` over the configured darkness window.
The stored arrays also contain RMS and final absolute error, mean/minimum PVA
strength and mean bump contrast. Results are recorded separately for every
velocity and averaged across the configured balanced velocity set.

## Snapshot cadence

The original Figure 3 records development at every 1% of the simulation. The
default diagnostic therefore uses
`tests.weight_snapshot_pi_interval_fraction: 0.01`. Actual weights must first
have been saved by the training-owned
`simulation.weight_snapshot_interval_duration` (seconds). The 80,000 s baseline saves every
800 s, so it supplies 101 states including `t=0` and `t=80,000 s`.

## Why constant velocity is primary here

Matched constant commands reduce validation variance and reveal direction
bias, gain error and low-speed pinning directly. This is preferable when the
question is whether performance improves and then degrades as training
continues. A fixed-seed OU ensemble remains the better final test of
naturalistic accumulated error and diffusion, but repeating it for every
snapshot would be substantially more expensive and would mix weight changes
with finite-ensemble variability.

## Code and outputs

- Registry: `src/learning/config/diagnostics.py`
- Diagnostic: `src/learning/diagnostics/weight_development.py`
- Shared frozen initialization: `src/learning/diagnostics/protocols.py`
- Plot: `src/learning/plotting/weights.py`
- Data: `weight_snapshot_pi_development.npz`
- Figure: `figures/weights/training_snapshot_frozen_pi_error.png`

No Vafidis-specific `references/<paper_id>/code_map.md` currently exists in
this repository; this task-level note provides the literature-to-code mapping
without creating an incomplete parallel reference package.

## Equation 19 online record

The separate `training_convergence` diagnostic now reproduces the paper's
local-error statistic rather than the frozen PI quantity above. Its online
recorder computes `mean_{neuron,time} |f(V_a)-f(V_ss)|` in 10 s forward
windows beginning every 1% of training. It stores population and per-neuron
values in spikes/s under `absolute_learning_error_*` fields in
`training_history.npz` and plots
`figures/diagnostics/training_absolute_learning_error.png`.

- Recorder: `src/learning/diagnostics/training_error.py`
- Training integration: `src/learning/experiments/run_vafidis_toy.py`
- Plot: `src/learning/plotting/weights.py`

The statistic must be collected online. Historical sparse `rms_e_hd` samples
use a different aggregation and cannot reconstruct Equation 19 exactly.

## 80,000 s baseline demo (2026-08-06)

Run:
`runs/vafidis_release_parameter_baseline/20260806-123005_vafidis_release_parameter_baseline_42`

Protocol: 101 snapshots at 1% intervals; constant velocities
`[-75, -30, 30, 75] deg/s`; 1 s stationary cue; 5 s darkness; statistics
start 0.5 s after cue removal.

- Best single snapshot: `t = 22,400 s`, aggregate mean absolute PI error
  `2.58 deg` and aggregate RMS error `3.22 deg`.
- Robust trend minimum: rolling windows of 3--9 snapshots center between
  `21,600 s` and `24,000 s`; the broad useful region is approximately
  `20,800--27,200 s`.
- Final snapshot at `80,000 s`: aggregate mean absolute PI error `36.24 deg`,
  or `33.66 deg` worse than the best snapshot.
- Every tested direction worsens by the end. Individual speed optima occur
  between `21,600 s` and `35,200 s`, which is why the balanced aggregate is
  preferable to selecting on one speed.
- Minimum PVA strength is `0.949` at the best snapshot and `0.955` at the
  final snapshot. The worsening PI error is therefore not explained by bump
  disappearance.
- Effective weight norms continue growing: HD-to-HD `173.75 -> 257.23` and
  HR-to-HD `401.19 -> 619.81` from the best to final snapshot.

Conclusion: this run clearly passes through a high-performance PI phase and
then overtrains. For this seed and protocol, a practical checkpoint target is
about `22--24 ks`, not the 80 ks hard cap. The exact stop should still be
validated across seeds and with a fixed-seed OU ensemble before being treated
as a universal training duration.
