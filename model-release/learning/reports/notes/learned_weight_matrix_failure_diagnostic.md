# Learned Weight Matrix Failure Diagnostic

Date: 2026-06-29

## Question

Why does the currently learned weight matrix perform poorly for the Vafidis-style predictive local plasticity toy model?

The target criteria from `learning/.todo/TODO.md` and `learning/.SKILL.md` are:

1. `w_hd_to_hd` should form a local symmetric recurrent structure.
2. `w_lhr_to_hd` and `w_rhr_to_hd` should form opposite shifted structures.
3. In darkness, decoded HD should move with angular velocity input.

## Short Conclusion

The current runs fail mainly because the HD population saturates into an almost spatially uniform high-rate state during darkness. Once this happens, PVA decoding has almost zero vector length, so the decoded heading is not a real bump position and velocity input cannot produce meaningful path integration.

This is not just a failure of the learned HR-to-HD weights. The hand-shifted HR baseline also gives near-zero velocity gain, which indicates that the test-time network dynamics cannot use HR input to move a stable bump under the current activation/input/inhibition regime.

## Evidence From Existing Runs

Existing metrics show velocity gain is approximately zero across learned and control runs:

| Run | `bump_final_abs_drift` | `darkness_rms_pi_error` | `velocity_gain` |
| --- | ---: | ---: | ---: |
| `default_final` | 0.6061 | 2.1898 | ~0 |
| `default_tuned` | 0.2905 | 2.1237 | ~0 |
| `hr_only_warm_probe` | 0.4215 | 2.2950 | ~0 |
| `recurrent_only_probe` | 0.0001 | 0.0001 with zero test velocity | ~0 |

Darkness activity diagnostics show HD rates are nearly saturated everywhere:

| Run | darkness `r_hd` mean | max | min | PVA abs start | PVA abs end |
| --- | ---: | ---: | ---: | ---: | ---: |
| `default_final` | 0.9999997 | 1.0000000 | 0.9999983 | 8.6e-08 | 1.4e-05 |
| `default_tuned` | 0.9999999 | 1.0000000 | 0.9999994 | 1.9e-08 | 4.6e-06 |
| `hr_only_warm_probe` | 0.9999607 | 0.9999996 | 0.9999536 | 2.6e-04 | 1.3e-04 |
| `recurrent_only_probe` | 0.9981597 | 0.9999984 | 0.9927072 | 6.9e-02 | 6.9e-02 |

So the decoder is operating on an almost uniform ring, not a localized bump.

## Likely Causes

### 1. Visual teacher is too positive and lacks inhibitory surround

Current implementation:

- `learning/src/learning/stimuli/visual.py` creates a positive von-Mises-like bump and subtracts `baseline`.
- Default config sets `visual.baseline: 0.0`.
- With sigmoid bias `0.0`, zero or weak input already gives firing rate around 0.5.

Relevant code/config:

- `learning/src/learning/stimuli/visual.py`
- `learning/configs/experiments/vafidis_toy.yaml`

Paper mismatch:

- Vafidis uses a visual input baseline `Ivis_o < 0`; Table 1 gives visual baseline `-5`.
- The paper states the visual input is inhibitory in the surround and acts by disinhibition.
- The existing audit already records this as a remaining toy simplification.

Consequence:

The training teacher does not strongly suppress off-bump HD cells. The presynaptic PSP traces therefore stay broad/nonzero, and the local outer-product update fills large parts of the matrix.

### 2. HD recurrent weights are constrained to be nonnegative

Default config:

- `learning_rule.w_hd_to_hd_min: 0.0`

Relevant code:

- `learning/src/learning/connectivity/constraints.py`
- `learning/src/learning/plasticity/predictive_local.py`

Paper mismatch:

- Vafidis Fig. 3 / text describes negative sidelobes in recurrent weights.
- It also describes excitation in one direction accompanied by inhibition in the reverse direction for HR-to-HD profiles.

Consequence:

The model can learn local excitation, but it cannot learn recurrent inhibitory sidelobes. Global inhibition alone is currently not enough to prevent all-HD saturation after positive recurrent mass accumulates.

### 3. Learned weights are dense and strongly positive

Saved weights show this directly:

| Run | matrix | min | max | mean | positive fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| `default_final` | `w_hd_to_hd` | 0.0000 | 0.0783 | 0.0417 | 0.9792 |
| `default_final` | `w_lhr_to_hd` | 0.0466 | 0.0767 | 0.0619 | 1.0000 |
| `default_final` | `w_rhr_to_hd` | 0.1062 | 0.1892 | 0.1476 | 1.0000 |
| `default_tuned` | `w_rhr_to_hd` | 0.1239 | 0.2756 | 0.2042 | 1.0000 |

This is not a useful local ring-attractor structure; it is close to a dense excitatory drive that pushes all HD cells high.

### 4. HR pathway is not the first bottleneck

The hand-shifted HR diagnostic gives near-zero velocity gain even when HR-to-HD weights are manually shifted. That points to a dynamics/testing bottleneck: the recurrent HD state is already saturated or too uniform for the HR pathway to translate a bump.

Therefore, fixing only the HR learning rule is unlikely to solve performance until the HD bump regime is restored.

## Recommended Next Checks

1. Add a diagnostic metric for PVA vector length / bump contrast during bump and darkness tests.
2. Try visual input with negative baseline/inhibitory surround, closer to Eq. 5 and Table 1 in Vafidis.
3. Allow negative recurrent weights or add an explicit centered/Mexican-hat constraint for `w_hd_to_hd`.
4. Tune global inhibition and sigmoid bias together; current `b_hd: 0.6`, `b_hr: 0.6`, and sigmoid bias `0.0` leave high tonic activity.
5. Re-run the hand-shifted HR diagnostic only after a non-saturated recurrent bump is confirmed.

## 2026-06-29 Code Update

Implemented the first diagnostic-driven corrections:

1. Added `pva_strength_hd` and `bump_contrast_hd` to run histories.
2. Added bump-quality metrics to `test_metrics.json`.
3. Changed default visual input to include inhibitory surround via `visual.baseline: 1.5`.
4. Allowed negative recurrent HD weights via `w_hd_to_hd_min: -1.5`.
5. Replaced simple diagnostic uses of `np.linalg.norm` and `np.polyfit` with direct scalar formulas to avoid fragile NumPy linalg calls in the available Windows conda environment.

Verification run:

```text
runs/vafidis_toy/post_diagnostic_fix_01
```

This run no longer collapses into an almost uniform saturated HD state:

| Metric | Value |
| --- | ---: |
| `bump_final_pva_strength` | 0.9681 |
| `bump_final_contrast` | 0.9998 |
| `darkness_final_pva_strength` | 0.9829 |
| `darkness_final_bump_contrast` | 1.0000 |

The remaining failure mode is now more specific: velocity-driven path integration is still absent.

```text
velocity_gain: 0.0
darkness_final_abs_pi_error: 2.8141
```

The post-fix hand-shifted HR diagnostic was saved to:

```text
learning/reports/notes/hand_shifted_hr_diagnostic_post_fix.csv
```

It still shows near-zero velocity gain even for manually shifted HR-to-HD kernels. Therefore the next bottleneck is likely in the toy model's velocity-to-HR-to-HD dynamics or protocol sensitivity, not merely the learned HR-to-HD weight structure.

## 2026-06-29 Further HR-Pathway Update

Implemented additional diagnostics for the second bottleneck:

1. Added HR wing activity diagnostics to every run history:
   - `mean_r_lhr`
   - `mean_r_rhr`
   - `contrast_r_lhr`
   - `contrast_r_rhr`
2. Added optional HR velocity input mode:
   - `additive`: Vafidis-style additive velocity current, kept as the default for backward compatibility and the main config.
   - `hd_gated`: experimental toy-mode where velocity multiplicatively gates the HD-to-HR drive, preventing off-bump HR cells from being uniformly lit by velocity alone.
3. Updated `diagnose_hand_shifted_hr.py`:
   - Removed the previous forced `k_vel >= 4.0` override.
   - Added `--k-vel` for explicit velocity-gain sweeps.
   - Added `--kernel-mode signed`, where HR-to-HD kernels have excitation in one direction and inhibition in the reverse direction.

Verification:

```text
pytest -q learning/tests
18 passed
```

Default additive verification run:

```text
runs/vafidis_toy/post_hr_diagnostics_default
```

The default run still maintains a clear HD bump:

| Metric | Value |
| --- | ---: |
| `bump_final_pva_strength` | 0.9681 |
| `darkness_final_pva_strength` | 0.9829 |
| `velocity_gain` | 0.0 |

New HR diagnostics expose the next failure signature during darkness with positive velocity:

| Quantity | End value |
| --- | ---: |
| `mean_r_lhr` | 0.0146 |
| `mean_r_rhr` | 0.9761 |
| `contrast_r_lhr` | 0.1167 |
| `contrast_r_rhr` | 0.0261 |

So the active RHR wing is almost uniformly high, leaving little spatial structure for HR-to-HD weights to translate into sustained bump motion.

Signed hand-shifted diagnostics were saved to:

```text
learning/reports/notes/hand_shifted_hr_diagnostic_signed_post_hr.csv
learning/reports/notes/hand_shifted_hr_diagnostic_signed_k1_post_hr.csv
```

Even signed excitation/inhibition HR kernels produce near-zero sustained `velocity_gain`. Lowering `k_vel` to 1.0 improves neither sustained gain nor PI. This suggests the remaining issue is not just learned weight sign or hand baseline shape; the current toy recurrent dynamics convert HR input mostly into fixed displacement / locking rather than continuous bump velocity.

## Test Note

Attempted to run:

```bash
PYTHONPATH=learning/src python -m pytest -q learning/tests
```

using the available `random` conda environment. Six tests printed as passed, then the process crashed inside NumPy `polyfit` during the smoke test velocity-gain calculation. This appears to be an environment/runtime issue in that conda environment, not a clean pytest failure from project assertions.

## 2026-06-29 Eq.31 / Common-Mode Update

The current implementation now corrects two additional bottlenecks:

1. Visual teacher input is treated as the paper's proximal drive:
   `(Ivis + Iexc) / (gD + gL)`, using `M = 4`, `Ivis_o = -5`,
   `Iexc = 4`, and proximal scale `1/3`.
2. `W_HD->HD` can use `hd_to_hd_balance_mode: zero_sum`, which removes the
   recurrent all-HD common mode while preserving the local Eq. 12-16 update.

Fresh verification:

| Run | `bump_final_pva_strength` | `darkness_decoded_velocity` | `velocity_gain` |
| --- | ---: | ---: | ---: |
| `codex_visual_exact_centered_probe` | 0.7673 | 2.59e-05 | -0.0179 |
| `codex_paper_like_after_eq31_centering` | 0.7543 | 3.0684 | 0.3213 |

The short diagnostic now learns a stable non-uniform HD bump, so the old
"uniform HD state" failure is fixed for the default config.  Hand-shifted
signed HR kernels can now move the bump, confirming that the remaining
near-zero gain in the short run is no longer a hard dynamics/readout failure.
It is mainly a learned HR-to-HD/training-protocol issue: paper-scale OU
velocity statistics and much longer training produce same-direction movement,
though still below the original paper's gain-1 result.

## 2026-06-29 Paired-HD / Normalized-Scale Update

A direct comparison with the released LearnPI code exposed two additional
toy-level mismatches:

1. The release code uses `NHD = 60` cells but only 30 angular positions; odd
   and even HD partners share the same preferred direction.  The toy had been
   treating all 60 HD cells as distinct angular positions, introducing an
   artificial half-bin offset between LHR and RHR pathways.
2. The release code sets `wHD = Aactive / fmax ~= 13.3` because firing rates are
   measured in kHz with `fmax = 0.15`.  The toy sigmoid is normalized to max 1,
   so the corresponding fixed HD-to-HR voltage scale is `Aactive = 2.0`.

The default diagnostic now uses paired HD preferences, `w_hd_to_hr_strength =
2.0`, `k_vel = 1/(2*pi)`, and a 160 s paper-scale OU velocity protocol.

Fresh verification:

| Run | `bump_final_pva_strength` | `darkness_decoded_velocity` | `velocity_gain` |
| --- | ---: | ---: | ---: |
| `codex_final_paired_scale_default` | 0.8332 | 7.3096 | 0.8416 |

The model is still a toy and not a full quantitative reproduction, but the
previous near-zero gain was largely caused by a combination of mismatched HD
topography, normalized-rate scaling, and low-speed testing.

## 2026-06-29 Drift Audit After Paired-HD Fix

The paired-HD fix exposed a new failure mode: joint OU training could make the
static bump drift even at zero velocity.

The decisive comparison was:

| Run | `bump_final_abs_drift` | `bump_intrinsic_drift_velocity` | `velocity_gain` |
| --- | ---: | ---: | ---: |
| `codex_signed_recurrent_only_probe` | 5.7e-7 | ~0 | 0 |
| `codex_signed_angle_default` | 0.0849 | -2.2e-4 | 0.8416 |

So the drift was not caused by the signed angle representation or by a direct
zero-velocity HR push.  It came from letting movement-related errors reshape
`W_HD->HD` during joint OU training.

The default protocol now uses:

- signed angles in `[-pi, pi)`;
- `80 s` zero-velocity recurrent warmup;
- main OU training with `W_HD->HD` updates gated to
  `|angular_velocity| <= 4 rad/s`.

Fresh verification:

| Run | `bump_final_abs_drift` | `bump_intrinsic_drift_velocity` | `velocity_gain` |
| --- | ---: | ---: | ---: |
| `codex_final_signed_warmup_velgate4` | 0.0252 | -5.8e-5 | 0.8467 |

This does not fully solve quantitative PI error, but it removes the obvious
ongoing zero-velocity drift while preserving learned same-direction velocity
drive.

## 2026-07-02 Decode Error / Flat-Top Diagnostic

The latest failure signature was a `~6 deg` bump-maintenance decode offset even
though the bump did not appear to drift continuously in darkness.  Direct
inspection of `runs/vafidis_toy/20260702-195200_vafidis_toy_11` showed:

- PVA final bump error: `-6.4 deg`.
- Highest raw max bin: often `-12 deg`, because multiple plateau bins were tied
  near saturation.
- Zero-velocity fitted drift velocity: about `-1.55e-5 rad/s`, so the large
  final offset was mostly a fixed attractor/plateau alignment error rather than
  a sustained dark angular velocity.
- Final bump activity had `16` cells above `0.99`, far broader and flatter than
  the release-code visual cue should produce.

The root cause was the visual-teacher width.  The toy used `kappa = 3.0`, while
the released Vafidis code's `sigma = 0.15` maps locally to
`kappa ~= 11.1111` in the normalized von-Mises teacher.  The broad teacher made
the local learning rule train a saturated plateau; PVA then decoded the
plateau's center rather than a sharp bump peak.

Implemented fixes:

1. Added a highest-peak/plateau-center decoder, stored as
   `theta_hd_decoded_peak`, while keeping PVA as `theta_hd_decoded`.
2. Added peak-based bump and darkness metrics, plus drift velocities in
   `deg/s`.
3. Collapsed paired HD cells before plotting tuning slices, preventing duplicate
   preferred directions from producing misleading vertical segments.
4. Changed activity plots to `[-pi, pi]` axes and labeled PVA vs peak decode in
   legends.
5. Updated the default config to `visual.kappa = 11.11111111111111`.

Verification run:

```text
runs/vafidis_toy/codex_kappa11_peak_decode
```

Key results:

| Quantity | Value |
| --- | ---: |
| `bump_final_abs_drift` | `0.00661 rad` (`0.379 deg`) |
| `bump_final_abs_peak_drift` | `0.0 rad` |
| `bump_intrinsic_drift_velocity_deg_s` | `2.20e-4` |
| final bump cells with `r_HD > 0.99` | `6` |
| `velocity_gain` / `velocity_gain_peak` | `0.9438` / `0.9437` |

Remaining issue:

The narrower, release-code-aligned visual teacher reduces the bump offset and
flat top, but the darkness velocity is now under-gained by about `1.00 rad/s`
at the 500 deg/s test velocity.  This points to HR-to-HD protocol/calibration
rather than zero-velocity drift: the zero-velocity tangent drive is dominated by
the learned recurrent component, but the fitted dark drift speed is essentially
zero during bump maintenance.

## 2026-07-02 Darkness Semantics and Remaining PI Error

The dark HD heatmap that appears to rotate many times in 6 s is not a
zero-input drift test.  It is the darkness PI condition: visual input is off,
but a 500 deg/s vestibular input is injected into HR cells, following
`utilities.py::vel_gain` and `fly_rec.py::simulate(day=False)`.  The zero-input
drift test is the bump-maintenance protocol after the cue, where
`angular_velocity = 0`.

The current zero-input drift is small:

```text
bump_intrinsic_drift_velocity = 3.84e-6 rad/s
bump_intrinsic_drift_velocity_deg_s = 2.20e-4 deg/s
```

The visually large PI error in the 500 deg/s darkness test comes from
under-gain:

```text
commanded velocity = 8.7266 rad/s
decoded velocity   = 7.7228 rad/s
bias               = -1.0038 rad/s
```

Over 6 s this accumulated speed error wraps around the circular error axis,
which is why the radian PI-error plot shows a sawtooth.  This is a real
remaining model limitation of the current toy HR-to-HD pathway, not a plotting
artifact.

A small parameter audit found a tradeoff:

- `kappa = 11.1111`, matching the release-code visual width, gives a cleaner
  bump and low zero-input drift but under-gains high-speed PI.
- Broader visual teachers such as `kappa = 5` improve high-speed gain but bring
  back a wider saturated plateau, so they are not a clean paper-based fix.
- Raising `b_hd` toward the release-code inhibition scale (`inh = -1`) did not
  remove the high-speed under-gain.

Therefore the next principled fix should be a protocol-level one, such as
Vafidis-style gain adaptation, longer training, or smaller `dt`, rather than a
post-hoc decoder or supervised rescaling.

## 2026-07-02 Remaining Flat-Top and Eigenvalue Diagnostics

The remaining "flat-top mountain" shape in some tuning slices is now a real
activity-saturation issue rather than the earlier plotting artifact.  The
paired-HD duplicate-angle artifact has been fixed by collapsing cells with the
same preferred direction before plotting.  However, the latest run still
saturates several central bump cells after learning:

```text
mean fraction r_HD >= 0.99 over training history = 0.127
widest training slice with r_HD >= 0.99          = 16 cells
late representative slices with r_HD >= 0.99     = 6 cells
```

This remaining plateau is caused by the normalized toy sigmoid and learned
local recurrent drive pushing the bump center close to the activation ceiling.
It is therefore not Clark-style heterogeneous tuning.  It also should not be
fixed by reshaping plotted curves or by post-hoc decoder calibration; the
paper-compatible route is to retune the reduced toy dynamics or training
protocol so that the bump remains localized without saturating the central
cells.

The new spectral diagnostic gives a more encouraging view of the recurrent
operator itself.  In `codex_gain11_semantic_figures`, `W_HD->HD` is symmetric
and its nonconstant eigenvalues are mostly paired:

```text
constant-mode eigenvalue real part        = -8.90e-17
spectral radius                           = 8.1809
first nonconstant pair gap / scale        = 0.00386
median nonconstant pair gap / scale       = 0.00149
pair fraction within 2% normalized gap    = 0.862
```

So the current failure is not that the recurrent weight matrix lacks the usual
ring-attractor spectral signature.  The main remaining problems are the
saturated activity nonlinearity and the high-speed HR-to-HD under-gain.

## 2026-07-03 Peak Decode and PI Protocol Clarification

The latest user-facing failure of peak decode is best interpreted as a
flat-top activity problem, not as evidence that the peak readout formula alone
is wrong.  In the current paired-HD geometry, several adjacent angular bins can
sit near the sigmoid ceiling.  A peak decoder then has to choose the center of
an extended plateau, whereas PVA/COM, the paper's primary readout, is still
partly stabilized by the full activity profile.

The code now records saturated-bin counts (`*_saturated_hd_bins`) after
collapsing odd/even HD partners.  These counts should be checked before using
peak-vs-PVA disagreements as a performance conclusion.

PI tests now use a visual-dark-visual protocol with separate `pi_cue_duration`.
For the default short diagnostic this gives 4 s visual, 6 s darkness, and 2 s
visual re-cue, matching the 20:30:10 proportions of the released Figure 2A /
Appendix 1 example.  All PI and bump tests keep weights frozen; visual input in
the first and last segments corrects/anchors activity but does not continue
learning.

## 2026-07-03 Activation / Single-Peak Audit

Parameter probes support the user's concern that decode quality must be fixed
before interpreting PI, but they did not support a one-parameter activation
fix:

- Lower sigmoid gain made the activity less saturated but degraded bump drift
  and velocity gain.
- Higher sigmoid bias could make the dark peak nearly single-bin, but only by
  suppressing the velocity-driven travelling bump.
- Narrower visual tuning made peaks unique, but violated the current
  release-code `sigma = 0.15` alignment and damaged PI gain.
- Lowering weight upper bounds made short-probe peaks sharper, but it worsened
  constant-velocity darkness PI and reduced gain.

The safer immediate change is to record `*_near_peak_hd_bins` and make the peak
decoder group the same saturated peak top with a 0.5% tolerance.  This improves
the diagnostic readout without pretending that the underlying tuning curve has
become truly single-peaked.  A true fix should focus next on rescaling local
learning rates and voltage/current magnitudes so long training does not push
the bump center to the sigmoid ceiling.
