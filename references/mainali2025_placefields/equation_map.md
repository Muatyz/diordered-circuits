# Equation Map: mainali2025_placefields

## Generative field

`g(r) ~ GP(0, C(r-r'))`，`rate(r) = max(g(r) - threshold, 0)`。

- Current code: `reproduction/src/utils.py::gaussian_process_place_fields_1d`
- Shape: 1D rate maps `(N, n_position)`
- Risk: current helper is a qualitative toy and does not yet expose the paper's full `(s, q)` fitting contract.

## Eq. 1: universal field-size distribution

The high-threshold form depends on spatial dimension `D`; in 1D it reduces to a Rayleigh-type distribution, while 2D gives an exponential area distribution.

- Not implemented analytically.
- Validation target: simulated excursion-set histograms across `D=1,2,3`.

## Eq. 3–4: normalized threshold and correlation length

- `q`: threshold divided by GP standard deviation.
- `s`: correlation length derived from the covariance derivatives.
- Required future API: explicit covariance, threshold and dimension rather than only filter bins.

## Bridge to optimized recurrent weights

- Current experiment: `compute_place_cell_optimized_weights.py`
- This is a project extension, not a direct result claimed by Mainali et al.
