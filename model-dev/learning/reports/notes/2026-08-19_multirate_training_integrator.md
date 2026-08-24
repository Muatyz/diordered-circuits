# 2026-08-19 block-multirate training integrator

## Decision

The advisor's slow-fast premise is useful, but a monolithic adaptive ODE is
not the production implementation. The current Vafidis model has no
`-lambda W` term: it evolves the filtered local induction variable and weight
as

```text
tau_delta d(delta)/dt = -delta + E P^T
dW/dt = eta delta.
```

For `N_HD=N_HR=60`, including both plastic matrices and both induction
matrices makes the current Markov state roughly 14,881 dimensional. The
fastest proximal mode remains about 1/3 ms, while OU input is stochastic.
An unstructured BDF/Radau solve would therefore add a large Jacobian problem
without guaranteeing fewer useful neural-dynamics evaluations.

## Implemented method

`simulation.training_integration_method` now selects:

- `single_clock`: the release-aligned ordered plasticity update at every
  neural timestep;
- `block_multirate`: neural/compartment/PSP states retain the same `dt`, while
  plastic weights are frozen within a configurable short block.

Every microstep's local `e_hd`, `p_hd`, and `p_hr` is retained. At the block
boundary, weighted matrix products evaluate the closed-form composition of
the existing Euler `delta` and `W` recurrences. Thus no local samples are
discarded or replaced by `outer(mean(E), mean(P))`. The only splitting error
is delayed feedback of the slowly changing weights within the block.

Optional clipping, symmetry, diagonal, and balance constraints are applied at
the block boundary. Such constrained experiments require their own
single-clock convergence check because constraint projection is nonlinear.
State validation also runs at each block/checkpoint boundary rather than each
microstep; any non-finite state is still rejected within at most one block.

## Selection

The authored baseline remains `single_clock`. Opt in with:

```text
python -m learning.experiments.run_vafidis_toy ^
  --config configs\experiments\vafidis_toy.yaml ^
  --profile configs\profiles\block_multirate.yaml
```

The profile uses a 10 ms plasticity interval. The interval must be a positive
integer multiple of `simulation.dt`. Partial final blocks and checkpoint
boundaries are flushed before state is recorded.

## Verification and benchmark

Tests cover:

1. algebraic block update versus repeated single-clock plasticity updates;
2. exact training parity when block size is one;
3. partial-final-block flushing and config validation;
4. a matched-noise short-training error bound;
5. loading the reusable profile through the normal config composition path.

On this workstation, 50,000 training steps with one BLAS thread gave:

| Network | single_clock | block_multirate | speedup | relative HD-weight difference | HD-rate RMS difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=60 | 7,750 step/s | 18,531 step/s | 2.39x | 3.77e-7 | 3.16e-8 |
| N=120 | 5,739 step/s | 15,641 step/s | 2.73x | 1.11e-6 | 6.70e-8 |

These are implementation-scale matched-stream results, not evidence of
long-horizon scientific equivalence. The remaining acceptance experiment is
matched-noise, multi-seed full training comparing absolute learning error,
weight development, selected snapshots, and frozen path integration.
