# Code Map: sagodi2024_back_to_continuous_attractor

## Concept-to-code map

| Paper concept | Current code | Current status | Required change |
|---|---|---|---|
| Frozen autonomous dynamics | `dynamics/autonomous.py::FrozenAutonomousDynamics` | Implemented; exact map and analytic flow Jacobian match the existing darkness step | Keep `r_hr` as explicit lagged state unless model dynamics are deliberately changed |
| Slow ring candidate | `run_bump_attractor_trajectory_test` | Implemented per-trajectory `10^-3 max speed` capture without storing the full all-trajectory tensor | Always report angular support before fitting |
| Reference manifold | `run_timescale_separation_test` | Uses visual-teacher `v_hd_distal` curve | Keep as teacher-manifold assay; add separately identified autonomous manifold |
| Normal recovery | `run_timescale_separation_test` | Operational trajectory e-folding test exists | Perturb/project in full Markov state around autonomous manifold |
| Tangential flow | `analysis/slow_manifold.py`, `analysis/phase_flow.py` | Spline-coordinate zero-input flow and existing rollout phase flow implemented | Add finite-horizon bound validation |
| Fixed points and basins | `analysis/slow_manifold.py`, `phase_flow.py::actual_stable_basins` | Reversal roots and basin entropy implemented for coverage-qualified ring | Add independent long-run root/basin validation |
| Normal hyperbolicity | `dynamics/autonomous.py::flow_jacobian`, `analysis/slow_manifold.py` | Leading full-dynamics spectrum and tangent alignment implemented conditionally | Do not compute/interpret a ring spectrum when angular coverage fails |
| PCA visualization | `plotting/slow_manifold.py` | Full-state ring PCA implemented only for accepted fit; rejected fit gets candidate coverage plot | PCA remains display-only |
| S-type robustness | input/state-noise and perturbation tests | Several noise paths exist but semantics are mixed | Add explicit state perturbation/noise result group |
| D-type robustness | none | Training-noise sweeps are not D-type | Add frozen weight-perturbation sweep with paired seeds/directions |
| Finite-time bound | angular errors + phase flow primitives | Not assembled | Compare empirical circular error with `min(t eta, pi)` |
| Asymptotic memory capacity | basin boundaries | Not implemented | Save basin entropy and max basin width |

## Proposed ownership

- `learning/src/learning/dynamics/autonomous.py`: canonical frozen state and vector field/step map.
- `learning/src/learning/analysis/slow_manifold.py`: tail-state selection, periodic manifold, full-state projection and invariance diagnostics.
- `learning/src/learning/dynamics/autonomous.py`: exact analytic Jacobian; central finite difference is its test oracle.
- `learning/src/learning/analysis/phase_flow.py`: decoded tangent field, fixed points, basin metrics and finite-time bound inputs.
- `learning/src/learning/experiments/test_vafidis_toy.py`: CLI orchestration only.
- `learning/src/learning/experiments/analyze_slow_manifold.py`: focused saved-run entry point.
- `learning/src/learning/analysis/make_vafidis_figures.py`: plots saved diagnostics without recomputing dynamics.

## Dependency order

```text
canonical frozen state/map
  -> autonomous trajectory cloud
  -> periodic full-state slow manifold
  -> full-state projection + invariance
  -> Jacobian spectrum + tangent alignment
  -> phase-flow eta + finite-time bound
  -> basin entropy / asymptotic tests
  -> S-type and D-type robustness sweeps
  -> PCA presentation and N/noise comparisons
```

## Existing results that must not be over-interpreted

- `compute_weight_eigenvalues` diagnoses connectivity organization, not nonlinear local stability.
- a high `timescale_separation_conservative_ratio` supports fast/slow behavior relative to a chosen target curve, but does not alone establish invariance or normal hyperbolicity.
- near-unit velocity gain tests driven integration, not zero-input memory drift.
- low diffusion under stochastic noise does not replace deterministic phase-flow/fixed-point analysis.

## First saved-run audit

`learning/runs/vafidis_toy/20260727-144656_vafidis_toy_42` retained 1024 strict slow points but covered only 21 of 180 angular bins (11.7%). The 21 occupied regions are disconnected low-speed angle clusters. The configured 50% support gate therefore rejected the periodic-ring fit and suppressed Jacobian/root claims. This is evidence for discrete/pinned slow states, not yet a direct flow-reversal count of fixed points.
