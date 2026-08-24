# Bump-attractor cue-amplitude sweep

Date: 2026-08-06

## Question

The frozen-weight trajectory diagnostic showed that uniformly spaced visual
cues converged toward recurrent basins before cue-off. This sweep tested
whether the training amplitude `visual.amplitude = 4` was also strong enough
to initialize the autonomous basin diagnostic.

The trained network was held fixed:

```text
runs/vafidis_release_parameter_baseline/
20260805-123005_vafidis_release_parameter_baseline_42
```

Only the diagnostic cue amplitude changed. The visual width, integration
method, learned weights, and all other model parameters remained fixed.

## Controlled 360-angle boundary sweep

The boundary sweep used 360 uniformly spaced cue angles and a 1 s cue. The
darkness interval was reduced to one integration step because this stage only
measured the cue-off transfer map.

| Cue amplitude | PVA RMSE (deg) | Slope | R2 | Near-zero local-gain fraction | Visual/distal modulation ratio | Median saturated bins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 2.448 | 1.0012 | 0.99945 | 0.2869 | 0.930 | 5 |
| 22 | 1.854 | 1.0011 | 0.99968 | 0.0390 | 1.019 | 5 |
| 24 | 1.292 | 1.0006 | 0.99985 | 0.0000 | 1.106 | 5 |
| 32 | 1.532 | 1.0003 | 0.99978 | 0.0000 | 1.387 | 5 |

`M = 24` was the smallest tested value that removed near-zero local-gain
intervals. Increasing to 32 did not improve the global cue-release error and
reduced median PVA strength from about 0.953 to 0.939.

## Default-duration confirmation

The selected amplitude was rerun with the configured 5 s cue and 360 angles:

| Metric | Value |
| --- | ---: |
| PVA cue-release RMSE | 1.338 deg |
| Linear slope | 1.00069 |
| R2 | 0.99984 |
| Near-zero local-gain fraction | 0.0000 |
| Local-gain median | 0.516 |
| Local-gain 5th--95th percentile | 0.144--3.778 |
| Median PVA strength | 0.953 |
| Visual/distal modulation ratio | 1.107 |
| Median saturated angular bins | 5 |

The cue-release map is therefore globally close to the identity and no longer
contains flat local intervals, but it retains a periodic local-gain ripple.
That ripple is consistent with the 30 unique preferred-direction positions
and should not be interpreted as a perfectly continuous microscopic map.

## Release-referenced endpoint interpretation

The endpoint map is now ordered by the uniform cue probes but computes its
autonomous displacement from the actual decoder phase at cue-off. Replaying
the saved 360-angle history gives:

| Decoder | Stable FPs | Trajectory-inferred unstable FPs | Unresolved intervals | Nonmonotonic crossings |
| --- | ---: | ---: | ---: | ---: |
| PVA | 30 | 30 | 0 | 12 |
| peak neuron | 30 | 0 | 30 | 12 |
| Clark overlap | 30 | 1 | 29 | 12 |

The PVA endpoint labels contain six duplicate forward crossings and six
reverse crossings. They arise from local folds of the cue-conditioned initial
state curve. The topology-aware trajectory criterion retains one strict
negative-to-positive bracket in each adjacent stable-FP interval and reports
the other 12 transitions as nonmonotonic quality diagnostics. Quantized peak
and overlap decoders rarely provide a strict displacement sign bracket, so
their intervals remain unresolved rather than receiving midpoint FPs.

## Decision

- Keep the paper-matched training value `visual.amplitude = 4` unchanged.
- Use `tests.bump_attractor_cue_amplitude = 24` for the frozen-weight
  trajectory/basin diagnostic.
- Use a 1 s diagnostic cue. The cue-conditioned state and its PVA transfer map
  have already settled by about 0.5 s; shortening the previous 5 s cue saves
  computation without changing the autonomous basin contraction. Near-zero
  cue duration is invalid because hidden recurrent state has not equilibrated.
- Continue to report saturation and local-gain metrics. A stronger cue is an
  initialization intervention; it does not repair autonomous discrete basins.
