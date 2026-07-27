# Open Questions: sagodi2024_back_to_continuous_attractor

1. What is the minimal Markov state of the current Euler implementation? In particular, should the one-step-lagged `r_hr` be retained as an explicit state, or should the dynamics be refactored to a simultaneous continuous-time RHS first?
2. Does the autonomous darkness state cloud form one closed ring in full state space, or only a ring after PVA projection?
3. Is the visual-teacher steady curve close to the autonomous slow manifold, and how does their Hausdorff/full-state distance depend on heading?
4. Does the full Jacobian have exactly one slow mode everywhere along the ring, and is its eigenvector aligned with the geometric tangent?
5. Is the zero-input `eta = sup |theta_dot|` sufficient to bound measured finite-time PVA error at all headings, or do decoder singularities/multi-peak activity violate the smooth near-bijection assumption C1?
6. Are zero-input fixed points and saddles alternating and stable across angular binning/smoothing choices?
7. How much asymptotic information is retained according to basin entropy, and does it predict long-horizon error better than velocity gain?
8. Do state perturbations (S-type) and frozen weight perturbations (D-type) produce different failure transitions across neuron count and visual heterogeneity?
9. Does training visual noise reduce tangential flow or improve normal spectral margin across paired seeds, rather than merely changing one finite-time test metric?
10. What behaviorally relevant horizon should define success for fly-scale versus mammalian-scale presets?

