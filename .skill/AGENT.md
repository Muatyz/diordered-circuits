本文档用于规范 codex 在 AGENT 模式下的行为. 

# Repository Architecture Contract

## Core principle

This repository is a reusable scientific experiment framework, not a
collection of task-specific reproduction scripts.

A concrete analysis, ablation, paper figure, or diagnostic request must be
implemented as a composition of reusable components and configuration
overrides.

## Non-negotiable invariants

1. Do not create task-specific Python entrypoints such as:

   - figure_*.py
   - reproduce_*.py
   - *_only.py
   - run_*_analysis.py

   unless the user explicitly approves a new permanent entrypoint.

2. Paper-figure reproduction must use:

   - existing experiment runners;
   - reusable diagnostic modules;
   - thin configuration recipes;
   - figure-layout specifications.

3. Do not duplicate model, training, rollout, or hyperparameter definitions
inside diagnostic or figure-specific configs.

4. All diagnostics must consume a shared RunContext produced by the common
experiment pipeline.

5. A diagnostic must not independently rerun training or simulation unless
its scientific definition explicitly requires a new intervention.

6. New diagnostics must be:

   - implemented under learning/diagnostics/;
   - registered in the diagnostic registry;
   - configurable through the common diagnostics schema;
   - covered by tests;
   - documented in docs/architecture/diagnostics.md.

7. Before creating a new file, inspect the existing registry, configuration
schema, runners, and neighboring modules. Prefer extending an existing
abstraction over adding a parallel code path.

8. If the requested feature does not fit the current abstractions, first
propose a refactor of the abstraction. Do not bypass the architecture with
temporary scripts.

## Configuration ownership

- Experiment configs own model, training, data, seeds, and rollout settings.
- Diagnostic configs own only diagnostic enablement and diagnostic-local
  parameters.
- Recipe configs may compose or override existing configs but must not
  redefine implementation logic.
- Figure specifications own panel selection and presentation only.

## Required workflow

For changes that add a diagnostic category, alter configuration schemas,
touch multiple modules, or introduce a new execution path:

1. Read AGENTS.md and docs/architecture/diagnostics.md.
2. Inspect the relevant code.
3. Produce an implementation plan.
4. Identify the reusable abstraction being extended.
5. List files to add, modify, and delete.
6. State explicitly whether any new runner or task-specific script is needed.
7. Do not implement until the plan is accepted.

## Definition of done

A feature is incomplete unless:

- it is accessible through the common runner;
- it is controlled by the common configuration system;
- it reuses the same experiment hyperparameters;
- it introduces no duplicated simulation path;
- tests pass;
- documentation is updated;
- obsolete temporary files are removed.