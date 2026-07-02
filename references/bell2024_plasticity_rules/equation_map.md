# Equation Map: bell2024_plasticity_rules

## Eq. 1: parameterized local rule

`dw_ij/dt = Theta(|w_ij|) sum_k c_k F_k(x_i, x_j, w_ij, tau_k)`

- Candidate basis uses pre/post activity, filtered traces and current synapse.
- Proposed project target: `learning/src/learning/local_rules.py`
- Meta-parameters: coefficients `c_k` and trace constants `tau_k`

## Eq. 4: reduced unperturbed rule

`dw_ij/dt = c0 * x_pre_trace * x_post - c1 * x_pre * x_post_trace - c2 * w_ij * x_post_trace * x_post`

Interpretation: temporally asymmetric Hebbian terms plus Oja-like postsynaptic normalization.

## Eq. 6: reduced turnover-robust rule

`dw_ij/dt = c0 + c1 * x_pre_trace * x_post - c2 * x_post_trace * x_post - c3 * w_ij * x_post_trace * x_post`

Interpretation: growth/replacement pressure plus synapse-independent and synapse-scaled homeostasis.

## Project translation

- Replace sequence fitness with ring-manifold fitness.
- Keep update locality audit separate from outer-loop objective.
- Compare against offline `J_opt`, global-gradient and nonlocal covariance baselines.
