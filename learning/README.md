# Vafidis-style predictive local plasticity toy model

`learning/` currently keeps a single main experiment for testing whether a
local predictive plasticity rule can learn:

- HD bump maintenance;
- velocity-driven path integration;
- interpretable `W_HD->HD` and `W_HR->HD` structure.

The implementation is intentionally compact.  Historical sweeps, hand-designed
HR probes, split YAML fragments, and alternate experiment configs were removed
so the active path is easy to inspect.

## Setup

Run commands from this directory:

```bash
cd learning
conda env create -f environment.yml
conda activate learning
python -m pip install -e .
```

An existing environment is fine if it already has the dependencies:

```bash
conda activate random
python -m pip install -e .
```

## Main Commands

Train and test the current toy model:

```bash
python -m learning.experiments.run_vafidis_toy --config configs/experiments/vafidis_toy.yaml
```

Skip figure generation during quick checks:

```bash
python -m learning.experiments.run_vafidis_toy --config configs/experiments/vafidis_toy.yaml --no-figures
```

Retest saved weights:

```bash
python -m learning.experiments.test_vafidis_toy --run-dir runs/vafidis_toy/<run_id>
```

Regenerate figures from an existing run:

```bash
python -m learning.analysis.make_vafidis_figures --run-dir runs/vafidis_toy/<run_id>
```

Inspect metrics:

```bash
python scripts/inspect_run.py runs/vafidis_toy/<run_id>
```

Run tests:

```bash
python -m pytest -q
```

On Windows, if pytest has trouble cleaning the system temp directory, use:

```bash
python -m pytest -q --basetemp runs/.pytest_tmp
```

## Current Config

The only active config is:

```text
configs/experiments/vafidis_toy.yaml
```

It is a complete YAML file, not a Hydra-style composition.  The config contains
model, simulation, learning-rule, visual-input, velocity-input, test, and path
settings in one place.

## Outputs

Each run writes to:

```text
runs/vafidis_toy/<run_id>/
```

Key files:

```text
config_resolved.yaml
params.json
trained_weights.npz
training_history.npz
bump_history.npz
darkness_history.npz
ou_darkness_history.npz
velocity_gain_history.npz
test_metrics.json
figures/
```

Figure naming note:

- `bump_maintenance_*` uses a short visual cue followed by darkness with
  `angular_velocity = 0`.
- `darkness_*` uses a constant-velocity visual -> dark -> visual protocol.
  The default timing follows the Figure 2A / Appendix 1 example proportions:
  `pi_cue_duration = 4 s`, `darkness_test_duration = 6 s`, and
  `recue_duration = 2 s`.
- `ou_darkness_*` uses the same visual -> dark -> visual protocol with an OU
  angular-velocity trajectory.
- Constant-velocity and OU heading figures plot unwrapped heading in units of
  `pi rad`; PI-error figures remain wrapped to `[-pi, pi]`.

Useful metrics in `test_metrics.json` include:

- `bump_final_abs_drift`
- `bump_final_abs_peak_drift`
- `bump_intrinsic_drift_velocity_deg_s`
- `bump_final_pva_strength`
- `darkness_final_abs_pi_error`
- `darkness_recue_final_abs_pi_error`
- `ou_darkness_rms_pi_error`
- `ou_darkness_recue_final_abs_pi_error`
- `darkness_decoded_velocity`
- `darkness_peak_decoded_velocity`
- `darkness_mean_saturated_hd_bins`
- `bump_final_saturated_hd_bins`
- `darkness_mean_near_peak_hd_bins`
- `bump_final_near_peak_hd_bins`
- `velocity_gain`
- `velocity_gain_peak`
- `hd_to_hd_local_symmetry_score`
- `lhr_to_hd_excitatory_source_offset`
- `rhr_to_hd_excitatory_source_offset`

## Code Layout

```text
src/learning/config/          dataclass schema and YAML loading
src/learning/common/          angles, array checks, random seed helpers
src/learning/stimuli/         visual and OU angular-velocity inputs
src/learning/dynamics/        HD/HR dynamics and sigmoid activation
src/learning/connectivity/    weight initialization and constraints
src/learning/plasticity/      predictive local plasticity and PSP traces
src/learning/models/          VafidisToyState, params, and step function
src/learning/experiments/     train/test command entry points
src/learning/analysis/        metrics and figure generation
src/learning/plotting/        plotting helpers
src/learning/io/              save/load and run-directory helpers
tests/                        focused pytest checks
scripts/                      run inspection and cleanup helpers
```

The most important implementation file is:

```text
src/learning/models/vafidis_toy.py
```

The core timestep order is: update true heading and angular velocity, generate
visual/velocity inputs, update HR, update HD distal current and voltage,
compute proximal firing and local prediction error, update PSP traces, then
update weights only during training.

## Notes

This toy model is a diagnostic implementation, not a quantitative reproduction
of Vafidis et al.  The local learning rule remains restricted to local
variables; no backpropagation, global loss, or supervised RNN trainer is used.

Implementation notes and historical audit details live in:

```text
reports/notes/vafidis_toy_implementation_audit.md
```
