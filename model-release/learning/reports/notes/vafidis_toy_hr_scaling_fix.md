# Vafidis toy HR scaling fix

Date: 2026-06-29

## Paper constraints used

- Eq. 8 drives HR cells additively from fixed HD input, angular velocity input, and constant HR inhibition.
- Eq. 9 low-pass filters HD rates into HR with tau_s = 65 ms.
- Table 1 sets the HR inhibitory current magnitude to 1.5.
- The Methods state that the fixed HD->HR projection strength is chosen so the HD firing-rate range maps onto the HR active range.
- Eq. 12 uses raw presynaptic PSP traces in the local predictive update, so this fix does not baseline-subtract PSPs or introduce a global loss.

## Failure signature before this fix

With `b_hr = 0.6` and `w_hd_to_hr_strength = 2.0`, the default HR population had weak bump contrast and broad positive background. Because `E_HD = f(Va) - f(Vss)` was mean-negative for much of training, the positive HR PSP background drove most HR->HD updates inhibitory. The resulting matrix could keep a weak HD bump but produced almost no decoded velocity in darkness.

## Code change

- Set default `tau_s` to 0.065.
- Set default `b_hr` to 1.5.
- Set default `w_hd_to_hr_strength` to 14.0 for this toy activation scale.
- Kept `velocity.input_mode = additive` and `k_vel = 1 / (2 pi)`, matching the paper's HR velocity-current principle.
- Updated experiment YAMLs and diagnostic sweeps so they do not silently reintroduce the old HR scale.

## Current status

The default initial HR population is now a sparse conjunctive bump: off-bump rates are near zero and the bump peak is in the active sigmoid range. This removes the immediate all-background HR failure mode. It does not by itself solve full path integration gain: higher-velocity training can produce more shifted excitatory HR->HD structure, but it still tends to destabilize bump maintenance. The next paper-aligned target is staged or better-conditioned training of HR->HD against an already stable HD ring, without changing Eq. 12 into a supervised/global objective.

Verification with `learning/configs/experiments/vafidis_toy.yaml` after the fix:

| Quantity | Value |
| --- | ---: |
| initial L/R HR min | 0.0021 |
| initial L/R HR max | 0.8241 |
| initial L/R HR contrast | 0.8221 |
| final training LHR contrast | 0.1383 |
| final training RHR contrast | 0.0785 |
| bump final absolute drift | 0.1203 |
| darkness RMS PI error | 1.9242 |
| velocity gain | -9.5e-08 |

An in-memory hand-shifted HR probe still produced only transient displacement, locking, or bump collapse rather than sustained velocity. That points to the next implementation target: the simplified HD compartment/test-time dynamics must be made able to convert a signed, shifted HR current into continuous bump translation before learned HR weights can be expected to score well.
