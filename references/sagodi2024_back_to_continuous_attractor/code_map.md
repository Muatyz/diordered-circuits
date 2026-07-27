# Code Map: sagodi2024_back_to_continuous_attractor

## Concept-to-code map

| Paper concept | Current code | Current status | Required change |
|---|---|---|---|
| Frozen autonomous dynamics | `models/vafidis_toy.py::step_vafidis_toy` | Dynamics, stimulus and plasticity share one step API | Add a pure frozen autonomous state/map wrapper before Jacobian work |
| Slow ring candidate | `run_bump_attractor_trajectory_test` | Long zero-input trajectories exist | Save full dynamic state, not decoded angle/HD voltage alone |
| Reference manifold | `run_timescale_separation_test` | Uses visual-teacher `v_hd_distal` curve | Keep as teacher-manifold assay; add separately identified autonomous manifold |
| Normal recovery | `run_timescale_separation_test` | Operational trajectory e-folding test exists | Perturb/project in full Markov state around autonomous manifold |
| Tangential flow | `analysis/phase_flow.py` | Direct PVA angular flow is implemented | Add zero-input `eta`, coverage/QC and bound validation |
| Fixed points and basins | `phase_flow.py::_periodic_roots`, `actual_stable_basins` | Implemented geometrically | Add independent long-run validation, basin fractions and entropy |
| Normal hyperbolicity | none | Weight spectrum and trajectory ratio are only proxies | Compute full dynamics Jacobian along manifold and tangent alignment |
| PCA visualization | pending TODO | Not implemented | Plot identified manifold/perturbations only after full-state analysis; PCA is display-only |
| S-type robustness | input/state-noise and perturbation tests | Several noise paths exist but semantics are mixed | Add explicit state perturbation/noise result group |
| D-type robustness | none | Training-noise sweeps are not D-type | Add frozen weight-perturbation sweep with paired seeds/directions |
| Finite-time bound | angular errors + phase flow primitives | Not assembled | Compare empirical circular error with `min(t eta, pi)` |
| Asymptotic memory capacity | basin boundaries | Not implemented | Save basin entropy and max basin width |

## Proposed ownership

- `learning/src/learning/dynamics/autonomous.py`: canonical frozen state and vector field/step map.
- `learning/src/learning/analysis/slow_manifold.py`: tail-state selection, periodic manifold, full-state projection and invariance diagnostics.
- `learning/src/learning/analysis/jacobian.py`: finite-difference Jacobian, eigenspectrum and tangent alignment.
- `learning/src/learning/analysis/phase_flow.py`: decoded tangent field, fixed points, basin metrics and finite-time bound inputs.
- `learning/src/learning/experiments/test_vafidis_toy.py`: CLI orchestration only.
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

