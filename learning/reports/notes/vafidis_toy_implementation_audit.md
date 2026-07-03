# Vafidis Toy Implementation Audit

Date: 2026-06-29

## Source

- Paper: Vafidis et al. 2022, "Learning accurate path integration in ring attractor models of the head direction system"
- Local PDF: `references/Learning accurate path integration in ring attractor models of the head direction system.pdf`
- Extracted text: `learning/reports/notes/vafidis2022_paper.txt`

## Corrected mappings

| Paper equation | Meaning | Code target |
| --- | --- | --- |
| Eq. 2 | HD distal input current dynamics | `learning/src/learning/dynamics/hd_dynamics.py::euler_update_i_hd_distal` |
| Eq. 3 | Independent HD axon-distal voltage leak | `learning/src/learning/dynamics/hd_dynamics.py::euler_update_v_hd_distal` |
| Eq. 4, 31 | Light-only proximal drive uses `Ivis + Iexc`, scaled by `1/(gD+gL)` in the steady-state reduction | `learning/src/learning/models/vafidis_toy.py::_make_i_vis_to_hd_proximal` |
| Eq. 8-10 | HR firing rate uses low-pass HD input plus instantaneous velocity input | `learning/src/learning/dynamics/hr_dynamics.py` |
| Eq. 11 | Distal-to-proximal steady-state attenuation `gD/(gD+gL)` | `p_distal_to_proximal`, default `2/3` |
| Eq. 12-14 | Predictive error times double-filtered presynaptic PSP | `learning/src/learning/plasticity/traces.py` and `predictive_local.py` |
| Eq. 15-16 | Low-pass plasticity-induction variable before weight update | `delta_w_hd_to_hd`, `delta_w_hr_to_hd` in `learning/src/learning/models/vafidis_toy.py` |

## 2026-06-29 current correction

The toy model now separates raw visual current from the effective proximal
teacher drive.  Configs use the paper-like raw visual values `M = 4`,
`Ivis_o = -5`, light-only `Iexc = 4`, and the reduced-model proximal scale
`1 / (gD + gL) = 1/3`.  This avoids treating the visual current as an
unscaled direct voltage term.

The default short diagnostic also applies `hd_to_hd_balance_mode: zero_sum`.
This is a toy structural constraint, not a new learning objective: it removes
the recurrent all-HD common mode after local plasticity updates so that the
learned `W_HD->HD` keeps local excitatory and inhibitory structure instead of
becoming dense global excitation.  The local rule remains Eq. 12-16.

Fresh checks after this change:

- `runs/vafidis_toy/codex_visual_exact_centered_probe`: the 40 s short
  diagnostic recovers a stable bump (`bump_final_pva_strength = 0.7673`) but
  still has near-zero velocity gain (`-0.0179`).
- `runs/vafidis_toy/codex_paper_like_after_eq31_centering`: the paper-like
  6000 s OU run recovers same-direction path integration
  (`velocity_gain = 0.3213`, `darkness_decoded_velocity = 3.0684 rad/s` for
  an `8.7266 rad/s` command), but it is still not a quantitative reproduction
  of the paper's gain-1 result.

## 2026-06-29 paired-HD / scale correction

The toy model now matches two structural details of the released LearnPI code:

- 60 HD cells represent 30 angular positions, with odd/even partners sharing
  the same preferred direction.
- With normalized toy firing rates, the fixed HD-to-HR strength is `Aactive =
  2.0` rather than `Aactive / fmax ~= 13.3`.

The default diagnostic protocol now uses a `160 s` OU trajectory and fly-scale
velocity tests.  A fresh run after these changes,
`runs/vafidis_toy/codex_paired_ou_160s_probe`, maintained a localized bump
(`bump_final_pva_strength = 0.8332`, `bump_final_contrast = 0.9997`) and
recovered a substantial velocity response (`darkness_decoded_velocity =
7.3096 rad/s` for an `8.7266 rad/s` command, `velocity_gain = 0.8416`).

This is still below the paper's quantitative gain-1 result, but it removes the
main toy-level failure mode where low-speed diagnostics and mismatched HR
scaling made the learned HR-to-HD pathway appear unusable.

## 2026-06-29 signed-angle / drift correction

The toy now represents angles in `[-pi, pi)` rather than `[0, 2*pi)`.  This
keeps the default `theta0 = 0` visual bump in the middle of the activity
array instead of splitting it across both ends.

A drift audit showed that the large zero-velocity displacement in the paired
model was not caused by the HR-to-HD pathway directly.  A recurrent-only run
with the same signed-angle representation had almost no drift
(`bump_final_abs_drift = 5.7e-7`), whereas joint OU training made the learned
`W_HD->HD` acquire a sizeable tangent drive.  This means the short toy
protocol was letting movement-related prediction errors reshape the static
recurrent backbone.

The default diagnostic now uses a local training schedule:

- `80 s` zero-velocity recurrent warmup with `eta_hr_to_hd = 0`;
- `160 s` OU training where `W_HD->HD` is updated only for
  `|angular_velocity| <= 4 rad/s`, while faster samples primarily train
  `W_HR->HD`.

Fresh check: `runs/vafidis_toy/codex_final_signed_warmup_velgate4` keeps the
bump nearly stationary after the initial cue (`bump_final_abs_drift = 0.0252`,
`bump_intrinsic_drift_velocity = -5.8e-5`) while preserving velocity drive
(`velocity_gain = 0.8467`, `darkness_decoded_velocity = 7.2639 rad/s` for an
`8.7266 rad/s` command).

## 2026-07-02 release-code alignment

A fresh comparison against `learning/original/fly_rec.py` exposed three
remaining toy-level mismatches:

- The released code updates `I_PSP` before using it to update `PSP` in the same
  timestep. The toy was using the previous synaptic trace in the second PSP
  filter stage, adding one extra timestep of learning-rule delay.
- The released code's HR velocity sign is first HR wing `+k*v`, second HR wing
  `-k*v` after the even/odd HD-to-HR split. The toy had the opposite wing
  convention.
- The released code keeps the HD-to-HR delayed copy `x` separate from HD distal
  current and PSP state. With the toy's coarser `dt = 10 ms`, using the same
  `tau_s = 65 ms` for this pathway under-drives velocity translation. The toy
  now exposes `model.tau_hd_to_hr`, defaulting to `tau_s` unless explicitly set;
  the fast diagnostic config uses `0.04 s`.

Fresh verification run:

```text
runs/vafidis_toy/codex_after_release_alignment
```

Compared with `codex_current_baseline`, the aligned diagnostic keeps a clear
bump and improves the velocity response:

| Metric | Before | After |
| --- | ---: | ---: |
| `velocity_gain` | 0.8467 | 0.8751 |
| `darkness_decoded_velocity` | 7.2639 | 7.3873 |
| `darkness_final_abs_pi_error` | 2.3223 | 1.8571 |
| `bump_final_abs_drift` | 0.0252 | 0.0677 |

The remaining limitation is still quantitative gain/error rather than bump
collapse: the darkness bump remains high-contrast, but decoded velocity is
still below the `8.7266 rad/s` command.

## 2026-07-02 decode / visual-width diagnostic update

A new comparison against the released `fly_rec.py` visual input found that the
toy's von-Mises visual teacher was too broad.  The release code uses
`sigma = 0.15` in
`exp(-sin(delta/2)^2 / (2*sigma^2))`; the matching local von-Mises width is
approximately `kappa = 1 / (4*sigma^2) = 11.1111`.  The active config now uses
that value instead of `kappa = 3.0`.

The previous bump-maintenance offset of about half a 30-position bin was not an
ongoing dark drift.  It came from a saturated flat-topped bump: PVA/COM decoded
the center of a broad plateau near `-6 deg`, while a simple max-bin decoder
could jump between tied plateau bins.  The code now records both:

- `theta_hd_decoded`: PVA/COM decode, kept as the main continuous readout.
- `theta_hd_decoded_peak`: highest-peak/plateau-center decode, after collapsing
  paired HD cells with identical preferred directions.

Fresh run:

```text
runs/vafidis_toy/codex_kappa11_peak_decode
```

Key diagnostics:

| Metric | Previous `20260702-195200` | New run |
| --- | ---: | ---: |
| `bump_final_abs_drift` | `0.1117 rad` | `0.0066 rad` |
| `bump_final_abs_peak_drift` | not recorded | `0.0 rad` |
| `bump_intrinsic_drift_velocity_deg_s` | not recorded | `2.2e-4 deg/s` |
| final bump cells with `r_HD > 0.99` | `16` | `6` |
| `velocity_gain` | `1.0212` | `0.9438` |

This fixes the main bump-maintenance decode error and the misleading
flat-topped tuning slices.  The remaining tradeoff is that the paper-like
narrower visual teacher learns a slightly under-gained darkness velocity
response (`darkness_velocity_bias = -1.00 rad/s` for the 500 deg/s test), so
future work should tune the HR-to-HD training protocol without re-broadening
the visual teacher.

Plotting changes:

- HD activity heatmaps and tuning slices use `[-pi, pi]` axes, so the default
  `theta0 = 0` bump is centered instead of split at a `0..360 deg` boundary.
- Training heatmaps plot PVA and peak decode as labeled traces, not unlabeled
  COM scatter points.
- Long training runs save a readable first-120-s training heatmap plus a full
  heatmap without decode overlay.

## 2026-07-02 darkness semantics / PI-error plotting update

The figure named `darkness_hd_activity_heatmap.png` was ambiguous.  In the
released code, `day=False` means visual input is zero, but velocity-gain tests
still inject vestibular input into HR cells.  The toy now labels this condition
explicitly as `visual off, velocity input on`.  The zero-velocity dark drift
condition remains the bump-maintenance protocol, labeled as `cue, then visual
off, omega=0`.

Plotting updates:

- `darkness_pi_error.png` now uses radians on a fixed `[-pi, pi]` axis.
- `training_heading_short_window.png` shows a 30 s window with network decode
  in the top panel and true heading in the bottom panel.  The window is chosen
  from the part of training where decoded heading moves most, so it directly
  checks whether the bump trajectory is continuous.
- `velocity_gain_curve.png` now uses 11 test velocities from `-500` to
  `500 deg/s` in `100 deg/s` increments, following the release code's principle
  of sampling a velocity range rather than fitting four symmetric points.

Fresh run after the plotting and gain-sampling update:

```text
runs/vafidis_toy/codex_gain11_semantic_figures
```

Key metrics:

| Metric | Value |
| --- | ---: |
| `velocity_gain` | `0.9721` |
| `velocity_gain_peak` | `0.9720` |
| `darkness_decoded_velocity` at 500 deg/s command | `7.7228 rad/s` |
| `darkness_velocity_bias` at 500 deg/s command | `-1.0038 rad/s` |
| `bump_intrinsic_drift_velocity_deg_s` | `2.20e-4 deg/s` |

The remaining PI-error issue is therefore not zero-input drift.  It is a
high-speed darkness under-gain of the current toy HR-to-HD pathway.  A
paper-compatible next step is to investigate Vafidis-style gain adaptation or
longer/finer-step training, rather than broadening the visual teacher again or
using post-hoc supervised calibration.

## 2026-07-02 PI-error interpretation / spectrum outputs

The remaining large `darkness_pi_error.png` should not be interpreted as a
faithful reproduction of an inherent failure of the Vafidis learning rule.  The
paper's explicit functional claim is stronger: after training, the network
should perform almost gain-1 path integration in darkness across the fly-scale
range `|v| < 500 deg/s`, with failure expected mainly near synaptic-delay
velocity limits.  The released code's `utilities.py::vel_gain` also tests
`day=False` while still injecting velocity input into HR cells, matching the
toy's current darkness PI protocol.

For `runs/vafidis_toy/codex_gain11_semantic_figures`, the zero-input drift is
negligible (`2.20e-4 deg/s`), but the 500 deg/s darkness test is under-gained:

```text
commanded velocity =  8.7266 rad/s
decoded velocity   =  7.7228 rad/s
bias               = -1.0038 rad/s
```

Over a 6 s test this accumulated velocity bias wraps around the circular
`[-pi, pi]` error axis, producing a large RMS PI error even though the final
wrapped error can be small by coincidence.  This is therefore best classified
as an incomplete toy/protocol reproduction of the paper's quantitative gain-1
result, not as evidence that the original local learning rule is intrinsically
unable to solve darkness PI.

The standard figure generator now also writes:

- `figures/training_weight_matrices_side_by_side.png`
- `figures/training_weight_eigen_spectrum.png`
- `weight_eigenvalues.npz`
- `weight_spectrum_diagnostics.json`

For the latest run, the HD recurrent matrix shows approximate nonconstant-mode
double degeneracy in the usual ring-attractor sense: after removing the
constant mode, `86.2%` of adjacent sorted-real eigenvalue pairs fall within a
2% normalized gap threshold, with first nonconstant pair gap `0.00386`.  This
supports the view that `W_HD->HD` has learned a mostly ring-like symmetric
operator.  The `W_HR->HD` spectrum is plotted too, but because that matrix is a
feedforward HR-to-HD map rather than a recurrent operator on one population,
its complex eigenvalues should be treated as a structural diagnostic rather
than direct Clark-style attractor modes.

## 2026-07-03 V-D-V protocol / peak-readout diagnostic update

The test protocol now separates two visually similar but conceptually
different conditions:

- bump maintenance uses the short `cue_duration` to initialize a bump, then
  tests zero-velocity darkness with frozen weights;
- PI tests use `pi_cue_duration` followed by darkness and re-cue, matching the
  visual-dark-visual layout in the released Figure 2A / Appendix 1 example.

For the current 6 s darkness diagnostic, the default timing is 4 s visual,
6 s dark, and 2 s visual re-cue.  This preserves the original example's
20:30:10 ratio while keeping the toy run short.  During all test phases,
`training=False`; visual input during cue and re-cue is only a teacher/correction
signal, not ongoing plasticity.

The paper's main heading estimate is the PVA/COM of HD activity.  The toy keeps
that as `theta_hd_decoded` and uses it for PI-error metrics.  The added
`theta_hd_decoded_peak` remains a diagnostic readout for the user's concern
that a small HD population should look more like a polygonal tuning curve than
a smooth dense curve.  When several adjacent paired-HD angular bins saturate
near the sigmoid ceiling, peak decode can become worse than PVA because there
is no unique biological "highest bin" to select.  The code now records
`*_saturated_hd_bins` metrics after collapsing odd/even HD partners so this
failure mode is visible in `test_metrics.json`.

The large visually uniform or "pure color" regions in exported weight matrices
are therefore not automatically normal.  A constant-color component can reflect
common-mode learned drive or color-scale compression, and it should be read
together with bump contrast, PVA strength, saturated-bin counts, row/column
balance, and eigenvalue diagnostics.  The released model relies on global
inhibition and long local training rather than an online mirroring operation or
post-hoc symmetry projection of HR weights; such nonlocal cleanup should remain
outside the biological toy unless it is explicitly labeled as a control.

Plotting changes:

- `training_heading_short_window.png` now overlays true heading, PVA decode,
  and peak decode in one panel, with wrapped circular error in the second.
- Constant-velocity and OU path-integration heading plots use `pi rad` y-units
  rather than thousands of degrees.

## 2026-07-03 activation / peak-sharpness audit

The latest peak-readout failure was not fixed by a simple activation-parameter
change.  Short retraining probes showed:

- lowering sigmoid gain from `2.5` toward `1.5` reduced saturation, but damaged
  bump maintenance and velocity gain;
- raising sigmoid bias reduced saturated dark bins only when it also strongly
  reduced PI gain;
- narrowing the visual teacher produced single-bin peaks, but departed from the
  released-code `sigma = 0.15` mapping and caused poor velocity gain.
- lowering plastic weight upper bounds sharpened the peak in short probes, but
  worsened constant-velocity darkness PI and reduced velocity gain.

The current default therefore keeps the paper-like logistic shape
(`beta = 2.5`, `x0 = 1`) and visual width.  The code-level change is limited to
making peak decoding plateau-aware: bins within 0.5% of the same collapsed-HD
peak top are treated as one saturated peak.  This makes the diagnostic readout
less sensitive to arbitrary tiny differences inside a biologically
indistinguishable plateau, while the new `*_near_peak_hd_bins` metrics expose
whether the model has actually learned a single-bin peak.  A real dynamics fix
should retune the reduced toy's voltage scale or local learning-rate scale
rather than adding winner-take-all, HR-weight mirroring, or supervised decoder
calibration.

## 2026-06-29 diagnostic update

The poor learned-weight performance is not caused by the velocity-gain
readout alone.  Re-testing the stable run
`runs/vafidis_toy/20260629-132937_vafidis_toy_11` at both the old toy
velocities (`[-1, -0.5, 0.5, 1]` rad/s) and paper-scale velocities
(up to 500 deg/s, `8.7266` rad/s) still gave velocity gain near zero.

Hand-shifted HR-to-HD probes showed the same thing at the old velocity
scale: even deliberately shifted HR kernels did not move the bump unless
the velocity input was made very large, so the learned matrix is being
tested in a regime where the local attractor barrier dominates the HR
drive.

The implementation also contained non-paper learning branches:

- a multiplicative `hd_gated` HR velocity mode, absent from Eq. 8-10;
- `hr_to_hd_update_sign`, which can flip Eq. 12 for HR weights;
- explicit weight decay inside Eq. 16.

These have been removed from the active model.  Old YAML fields are still
ignored by the config loader so previous run directories remain readable.

## Protocol mismatch found

The current `vafidis_toy.yaml` remains a fast diagnostic protocol, not a
paper-faithful training protocol.  Vafidis' released code trains with:

- OU angular velocity, `sigma_v = 225 deg / sqrt(s)` and `tau_v = 0.5 s`;
- velocity tests across the fly-scale range, roughly up to `500 deg/s`;
- Gaussian initialization of the plastic weights;
- `sim_run = "2Enough"`, i.e. `80,000 s` of simulated training at
  `dt = 0.5 ms`.

The toy default now uses a short `160 s` zero-init OU run and fly-scale
velocity tests.  It is intended to expose both static bump maintenance and
velocity drive in seconds, but it still omits the paper's long convergence
time, Gaussian initialization protocol, and smaller `0.5 ms` integration step.

`configs/experiments/vafidis_paper_like.yaml` records the paper-scale
velocity and initialization choices for further work.  It is not yet a
drop-in reproduction because the toy model still compresses or omits
some dynamics listed below.

## Previous mismatches

- The toy previously treated `NHD = 60` as 60 distinct angular positions.  The
  released code uses 60 HD cells but 30 angular positions, with odd/even HD
  partners sharing the same preferred direction.
- The normalized-rate toy previously kept the paper/code value
  `wHD = Aactive / fmax ~= 13.3`.  Because the toy sigmoid has maximum 1 rather
  than `fmax = 0.15 kHz`, the equivalent HD-to-HR voltage scale is now
  `Aactive = 2.0`.
- The toy short config previously used low-speed tests and `k_vel = 1/pi`.
  The paper's `k = 1/360 s/deg` corresponds to `1/(2*pi)` for rad/s inputs.
- `v_hd_distal` was set equal to `i_hd_distal`, omitting the paper's separate leak equation for distal voltage.
- Visual input was previously added directly to `v_hd_proximal`; Eq. 4/31 instead uses `(Ivis + Iexc)/(gD+gL)` in the steady-state reduced model.
- HR dynamics low-pass filtered the sum of HD input and velocity input; the paper low-pass filters HD firing rates and adds velocity input instantaneously.
- PSP traces were single-time-constant rate traces; the paper defines a double-exponential filter from `tau_s` and `tau_l`.
- Weight updates used the instantaneous plasticity induction term directly; the paper filters this term with `tau_delta`.
- Default `p_distal_to_proximal` was `1.0`; the paper's default conductances imply `gD/(gD+gL)=2/3`.

## Remaining toy simplifications

- `NHD` and `NHR` now follow the paper-scale primary model in the experiment configs: `n_theta = 60`, `n_hr = 60`, with 30 angular HD positions represented by odd/even HD pairs and 30 LHR / 30 RHR cells. The HD-to-HR projection uses the paper's odd/even HD split rather than duplicating every HD cell into both HR wings.
- The visual input now follows the paper-like disinhibitory raw current and
  steady-state proximal scaling.
- The default `dt` remains larger than the paper's `0.5 ms` to keep toy experiments fast; high-fidelity runs should lower `simulation.dt`.
- The current toy model omits proximal voltage membrane dynamics and uses the steady-state form from the paper's reduced derivation.
- `hd_to_hd_balance_mode: zero_sum` is an explicit toy structural constraint
  used to keep the short diagnostic out of dense common-mode recurrent drive;
  the released full model instead relies on longer training, global
  inhibition, random initialization, and the full dynamics.
