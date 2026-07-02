# Equation Map: clark2025_symmetries

## Eq. 1: rate RNN dynamics

### Paper form

`tau dx_i/dt = -x_i + sum_j J_ij phi_j(x_j) + b_i`

### Code target

- `reproduction/src/utils.py`: `simulate_rate_network`, `simulate_rate_network_with_drive`, `relax_rate_network_states`
- Shapes: state `(N,)` or `(n_init, N)`；trajectory `(T, N)` 或 `(n_init, T, N)`

### Validation

- manifold fixed-point residual
- perturbation distance to nearest manifold point
- NaN/overflow and time-step sensitivity

## Eq. 2–3: flow-matching loss and optimized weights

### Discrete implementation

- `theta` is discretized into `n_theta` periodic bins.
- Integral over the ring becomes a weighted mean/sum over bins.
- Row-wise ridge regression maps `Phi` to `X` while enforcing `J_ii = 0`.
- Minimum-norm structure is essential to off-manifold stability.

### Code target

- `reproduction/src/utils.py`: `optimized_recurrent_weights`, `optimized_recurrent_factors`
- `reproduction/src/plot_figure3_abcd.py`: `build_figure3_matrices`
- Output: `data/processed/figure3_abcd_weight_matrices.npz`

### Numerical risks

- inverse-softplus tail
- ill-conditioned `Phi Phi^T`
- leave-one-neuron-out diagonal constraint
- normalized versus unnormalized rates

## Eq. 4: firing-rate correlation

`C_phi(theta, theta') = (1/N) sum_i phi_i(theta) phi_i(theta')`

- Code: `utils.py::empirical_two_point_correlation`
- Validation: convergence toward circulant form and Fourier modes as sample size grows

## Eq. 5: velocity-modulated weights

- Static and tangent components are built from `x_star` and `d_theta x_star`.
- Code: `utils.py::optimized_recurrent_velocity_factors`, `simulate_velocity_modulated_rate_network`
- Validation: decoded overlap bump follows constant and recorded angular velocity.

## Eq. 6: overlap order parameter

`m(theta, t) = (1/N) sum_i phi_star_i(theta) phi_i(t)`

- Code: `utils.py::overlap_order_parameter`
- Script: `plot_figure3_jkl.py`
- Output: `data/processed/figure3_jkl_overlap.npz`

## Eq. 9–10 / Appendix B3: generative process

- Sample `x_star(theta)` from a wrapped-Gaussian process.
- Apply biased normalized softplus and per-neuron mean normalization.
- Code: `utils.py::sample_circular_gaussian_process`, `normalized_softplus_tuning`
- Pipeline: `gaussian_generative_process.py`

## Appendix B4: double normalization and inhibition

- Neuron-wise normalization gives mean-one tuning curves.
- Angle-wise population normalization makes the population mean exactly one at every angle.
- Weight shift `J_ij -> J_ij - c/N` with bias compensation stabilizes mean activity.

## Jacobian on the target manifold

`A_ij(theta) = -delta_ij + J_ij phi'(x_star_j(theta))`，再加入 uniform inhibition 对应项。

- Code: `utils.py::current_space_jacobian`
- Diagnostics: `reproduction/src/diagnostics/diagnose_jacobian*.py`
- Expected signature: one near-neutral tangent mode; normal modes have negative real parts.
