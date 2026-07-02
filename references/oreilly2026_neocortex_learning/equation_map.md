# Equation Map: oreilly2026_neocortex_learning

## Temporal-state error proxy

`delta activity = activity_plus - activity_minus`

- minus phase: prediction state
- plus phase: outcome-driven state
- candidate synaptic signal: pre activity/traces multiplied by a postsynaptic temporal difference

## Fast-minus-slow implementation

`temporal_derivative_proxy = fast_integral(signal) - slow_integral(signal)`

- Proposed project location: future `learning/src/learning/local_rules.py`
- Required state: local fast/slow pre and post traces
- Required audit: whether outcome/modulatory input is locally available

## Candidate project baseline

`Delta J_ij ∝ pre_trace_j * (post_plus_i - post_minus_i)`

This is a conceptual translation, not a verbatim equation from the paper and must be labeled as a project hypothesis when implemented.
