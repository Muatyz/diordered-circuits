# 2026-08-16 PI, trajectory and numerical audit

## Scope

This note records the evidence behind the 08.16 code changes. It separates
three questions that were previously mixed together: long-horizon path-
integration error, endpoint-map fixed points, and timestep convergence.

## Baseline evidence

Run:
`runs/vafidis_release_parameter_baseline/20260807-130716_vafidis_toy_42`
using the saved best weights.

- The 5 s frozen-snapshot aggregate error is about 3.36 deg at 24,500 s,
  versus 14.12 deg for the final 50,000 s weights. This is functional
  overtraining evidence; it is not evidence that the bump disappeared.
- The darkness velocity fit has gain 0.9465 and R-squared 0.9987. The network
  follows velocity approximately linearly, but its roughly 5% gain mismatch
  accumulates over 60 s. The constant-probe final RMS circular error is about
  96.4 deg.
- The OU single-trial RMS circular error is about 70.6 deg. The OU ensemble
  additionally contains systematic mean drift and across-trial spreading, so
  those moments must be reported separately.
- The effective HR-to-HD/HD-to-HD norm ratio grows from about 1.01 initially
  to 2.28 at the best snapshot and 2.38 finally. Both pathway norms continue
  growing after the functional optimum. The ratio alone is not causal: the
  heterogeneous run has a ratio near 1.5 and still performs poorly.

Consequently the implementation now saves release-relative unwrapped PI
error, velocity bias, wrapped companion error, both effective norm-growth
traces and their ratio. No learning rate, clipping rule or normalization was
changed from this correlation alone.

## Endpoint map

The previous baseline output contained roughly 30 attracting endpoints, 29
strictly bracketed repelling boundaries and one unresolved interval. This is
close to the expected stable/unstable alternation, but the old visualization
mixed cue onset, cue release, three decoders and two endpoint coordinates.

The canonical figure is now PVA-only:

1. darkness trajectories after cue release;
2. initial cue angle versus relaxed cue-release angle;
3. initial cue angle versus darkness endpoint.

Fixed points are roots of
`D(phi)=wrap(endpoint(phi)-phi)` in the actual release coordinate. A
positive-to-negative crossing is stable; a negative-to-positive crossing is
unstable. Periodic seam handling, cue-map validity, unresolved gaps and any
alternation mismatch are explicit outputs. Equal counts are checked but never
forced.

## Whole-step numerical audit

The fastest proximal mode has time constant
`C/(gL+gD)=0.333... ms`. A 1 ms forward-Euler step is rejected because its
homogeneous amplification reaches -2. `exact_linear` removes that one Eq. (4)
stability restriction but does not analytically integrate distal currents,
rates, the HD-to-HR low-pass or the rest of the ordered map.

The new diagnostic first forms one shared release state using a 0.5 s visual
cue at 0.0625 ms with the exact-linear proximal substep. It then compares all
methods from that same state for 2 s at common physical sample times. On the
baseline best weights, maximum PVA phase error / maximum HD-rate RMS error
relative to the reference were:

| method | dt [ms] | max phase error [deg] | max rate RMS error |
| --- | ---: | ---: | ---: |
| forward Euler | 1.0 | invalid | invalid |
| forward Euler | 0.5 | 2.645 | 0.00699 |
| forward Euler | 0.25 | 1.193 | 0.00315 |
| forward Euler | 0.125 | 0.461 | 0.00122 |
| exact-linear Eq. (4) | 1.0 | 3.423 | 0.00900 |
| exact-linear Eq. (4) | 0.5 | 1.734 | 0.00457 |
| exact-linear Eq. (4) | 0.25 | 0.774 | 0.00205 |
| exact-linear Eq. (4) | 0.125 | 0.264 | 0.00070 |

With the authored strict thresholds (1 deg phase and 0.002 rate RMS), only
0.125 ms passes both on this single saved state; exact-linear 0.25 ms is
borderline and misses the rate threshold by about 2.5%. This rules out the
claim that exact-linear makes 0.5 or 1 ms automatically trustworthy. It does
not by itself justify changing long stochastic training to 0.125 ms: full
training convergence also needs matched noise paths and multiple seeds.

The release HD-to-HR line omitted `dt`. At release dt=0.5 ms its literal
physical effective time constant is 32.5 ms, whereas paper Eq. (9) and the
production model use 65 ms. The numerical artifact records both semantics;
the release-literal discrepancy is not enabled as a production solver mode.

## Decisions

- Keep the current 0.25 ms exact-linear experiment setting pending matched-
  noise, multi-seed training convergence; do not revert to 1 ms.
- Keep 50,000 s as a hard cap rather than a convergence claim. Use explicit
  frozen-snapshot selection (phase error or RMS velocity bias) and the
  existing checkpoint/early-stopping mechanism for research profiles.
- Do not modify pathway learning rates from norm growth alone. Compare norms,
  realized pathway currents, local learning error and frozen behavior on the
  same training-time axis.
