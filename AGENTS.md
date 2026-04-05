# AGENTS.md

This file gives repo-specific guidance for coding agents and contributors working in `ReconDreamer-RL`.

## Project Overview

This repository contains an actor-learner reinforcement learning stack for autonomous driving policies. The current main training path is file-buffer based:

`actor -> rollout -> shard buffer -> learner -> checkpoint/version publish -> actor reload`

The main entrypoint is:

- `script/train_actor_learner_v2.py`

The learner path currently combines PyTorch Lightning with a custom actor-learner runtime. When making changes, preserve the end-to-end training contract unless the task explicitly asks for architectural refactoring.

## Workflow Expectations

### Plan first, then execute

For non-trivial tasks, do not jump straight into code changes.

Expected workflow:

1. Enter plan mode first.
2. Produce a concrete step-by-step plan.
3. Wait for plan confirmation when the task is complex or architectural.
4. Only then start implementation.

Use this especially for:

- multi-file changes
- architecture refactors
- actor-learner protocol changes
- Lightning/DDP/optimizer ownership changes
- config schema changes

For very small and local edits, a lightweight plan in plain text is still preferred before editing.

### Review after each round of changes

Every meaningful implementation round must be followed by a review pass.

Expected workflow:

1. Make a bounded set of changes.
2. Run `/review` or perform an equivalent review pass.
3. Check for regressions, risky assumptions, and boundary violations.
4. Only then continue to the next round or finalize.

Do not treat review as optional at the end. The default here is iterative change -> review -> refine.

## Start Here

If you are new to the repo, read files in this order:

1. `script/train_actor_learner_v2.py`
2. `framework/runner/actor_learner.py`
3. `framework/runner/factories.py`
4. `framework/rollout/collector.py`
5. `framework/lightning/actor_learner_datamodule.py`
6. `framework/lightning/actor_learner_module.py`
7. `framework/algorithms/trajectory_batch.py`
8. `framework/algorithms/trajectory_policy_core.py`
9. The relevant implementation under `framework/agent/`

Also read:

- `framework/README.md`
- the README inside the subpackage you are editing

## Directory Responsibilities

Treat the current package boundaries as follows.

### `framework/runner/`

- Runtime orchestration
- Entry points for actor, learner, orchestrator
- Config normalization and object assembly
- Process launch concerns

Do not add unrelated math or model logic here unless the task is specifically about runtime wiring.

### `framework/agent/`

- Policy/model adapters
- Action sampling
- Replay encoding for training
- Log-prob recomputation from replay
- Checkpoint load/save

Keep model-specific logic here. Avoid leaking model internals into runner or algorithms.

### `framework/rollout/`

- Actor-side environment interaction
- Building shard payloads from sampled trajectories

### `framework/io/`

- Buffer paths
- Version files
- STOP markers
- Shard lifecycle helpers

Changes here can easily break actor-learner coordination. Be conservative.

### `framework/lightning/`

- Lightning modules and datamodules
- Learner-side training lifecycle hooks
- Training-step integration with RL objectives

### `framework/algorithms/`

- Objective math
- Advantage/return preparation
- PPO / Reinforce-related training logic

### `framework/env_wrapper/`

- RL environment wrappers around the simulator
- Scene sampling and vector-env support
- Reward and termination integration points

### `framework/rewards/`

- Reward shaping and terminal penalties

### `framework/utils/`

- Shared utilities
- Repo path resolution
- Observation helpers
- gsplat and extension support

## Working Rules For This Repo

### Prefer minimal, local changes

- Make the smallest change that solves the task.
- Avoid broad file moves or cleanup unless the task is explicitly architectural.
- Do not introduce backup files, duplicate modules, or `v2/final/fixed` variants.

### Respect the actor-learner protocol

The following are protocol-level pieces. Do not change them casually:

- shard naming/version assumptions
- `weights/latest.ckpt`
- `weights/version.txt`
- `STOP`
- `TRAINING_LOCK`
- shard fields such as `obs`, `old_logp`, `reward`, `done`, `replay`, and PPO-related value fields

If you change shard schema or checkpoint/version behavior, review both actor and learner sides together.

### Keep one source of truth

This codebase already has some thin wrappers and compatibility layers. When modifying architecture:

- prefer consolidating logic instead of adding another wrapper
- prefer canonical import paths over new aliases
- remove dead abstractions only when you can verify the main path still works

### Be careful with config changes

If you add or rename config fields, check all of these places:

- `script/train_actor_learner_v2.py`
- `framework/runner/factories.py`
- `framework/runner/actor_learner.py`
- any YAML config under `script/configs/`

Do not assume a config key is unused until you confirm all readers.

### Keep model-specific assumptions scoped

This repo supports multiple policy backends, including DiffusionDriveV2 and SparseDrive variants. Avoid hardcoding assumptions for one model in shared code paths unless the task explicitly narrows scope.

## Lightning Guidance

The current code uses Lightning, but not all training ownership is fully inside Lightning yet.

When refactoring learner code, prefer this direction:

- `LightningModule` owns training-step behavior
- `configure_optimizers()` is the authoritative optimizer definition point
- `Trainer` should own device placement and distributed strategy where practical
- runner should assemble and launch, not micromanage training internals

Do not half-migrate a training path. If you move optimizer, DDP, or device ownership into Lightning, update the surrounding code so responsibilities stay clear.

## Common Hotspots

Be extra careful when editing these files:

- `framework/runner/actor_learner.py`
- `framework/runner/factories.py`
- `framework/lightning/actor_learner_datamodule.py`
- `framework/lightning/actor_learner_module.py`
- `framework/algorithms/trajectory_batch.py`
- `framework/agent/policy_diffusiondrivev2.py`
- `framework/agent/policy_sparsedrive_v2.py`

These files sit on the main runtime path and are easy places to accidentally create duplicated responsibilities.

## Verification Expectations

Before claiming a change is complete, run the narrowest meaningful verification you can.

Prefer targeted checks first:

- `pytest tests -q`
- targeted `pytest` for the area you changed
- a small smoke run of `script/train_actor_learner_v2.py` when touching actor-learner wiring

If you change any of the following, mention what you did and what you could not verify:

- shard schema
- Lightning module/datamodule contracts
- optimizer construction
- distributed behavior
- checkpoint/version publishing

## When Doing Architecture Work

If the task is architectural cleanup, optimize for clearer ownership:

- runtime code in `runner/`
- protocol and buffer logic in `io/`
- batch preparation in `batch/` or the canonical batch module
- objective math in `algorithms/`
- model adapters in `agent/`
- learner lifecycle in `lightning/`

A good change in this repo usually reduces cross-directory jumping for one feature, not increases it.

## Notes For Future Agents

- Check whether an abstraction is actually used on the main path before extending it.
- Some interfaces look more generic than they really are.
- Read imports and call sites, not just class names and README claims.
- If you find duplicate logic in runner and Lightning, prefer documenting it first, then consolidating with tests.
