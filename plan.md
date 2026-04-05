# SparseDriveV2 Frozen-Backbone Value Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SparseDriveV2 PPO critic training materially more stable by reusing the agent's frozen visual backbone and frozen status encoder, while training only a lightweight value head on top.

**Architecture:** Use a staged plan focused only on SparseDriveV2. First verify the current failure mode with better critic diagnostics and a reproducible baseline. Then replace the image-only critic path with a frozen-backbone feature extractor that pools SparseDriveV2 visual features, concatenates frozen status encodings, and feeds them into a trainable value head. Keep the actor-learner shard protocol stable and avoid joint backbone training in this phase.

**Tech Stack:** PyTorch, PyTorch Lightning, actor-learner file buffer, PPO, SparseDriveV2 replay features.

---

## Problem Statement

Current SparseDriveV2 PPO instability is most likely driven by a mismatch between what the policy sees and what the value net sees:

- SparseDriveV2 policy log-prob recomputation uses replay-derived `camera_feature` and `status_feature`.
- The current learner-side `ValueNet` in `framework/runner/learner_factory.py` only sees a `(B, 18, 64, 64)` image tensor.
- SparseDriveV2 already has a trained visual backbone and a learned `_status_encoding` layer, but the critic ignores both.
- `status_feature` contains route/command and ego dynamics information that can change return substantially even when image appearance is similar.
- Returns are noisy because the environment includes early termination, collision/failure conditions, and terminal penalties.

This means the current critic is being asked to fit a hard target with incomplete state information. Negative `explained_variance` is therefore plausible even when the metric code itself is correct.

## Recommended Strategy

Recommended path:

1. Add evidence first. Do not change the critic architecture blindly.
2. Keep the actor-learner protocol stable. Derive critic inputs from SparseDriveV2 replay instead of changing shard schema.
3. Reuse SparseDriveV2 `_backbone` and `_status_encoding` in frozen mode, then train only a new value head.
4. Do not let value loss update the policy backbone in this phase.

## Success Criteria

- `explained_variance` becomes consistently positive after warmup instead of staying heavily negative.
- `loss_v` stops diverging or oscillating violently across adjacent updates.
- PPO policy metrics remain stable: no new `approx_kl` spikes or `clip_frac` blow-ups caused by the critic change.
- The actor-learner shard contract remains backward-compatible during phase 1.
- A small smoke training run completes with the SparseDriveV2 frozen-backbone critic path enabled.

## Non-Goals

- No broad refactor of actor-learner runtime ownership.
- No immediate redesign of replay schema.
- No policy architecture rewrite.
- No DiffusionDriveV2 work in this phase.

## File Map

Likely files for phase 1:

- Modify: `framework/runner/learner_factory.py`
- Modify: `framework/lightning/trajectory_module.py`
- Modify: `framework/lightning/actor_learner_module.py`
- Modify: `framework/agent/base.py`
- Modify: `script/configs/sparsedrive_v2/ppo_closed_loop_sparsedrive_v2.yaml`
- Modify: `framework/agent/policy_sparsedrive_v2.py`

Likely tests:

- Modify or add: `tests/framework/test_trajectory_objectives.py`
- Add: `tests/framework/test_value_net_inputs.py`

## Phase 0 Decisions

Before implementation, keep these guardrails:

- Prefer optional new batch fields over changing required shard fields.
- Reuse replay data that already exists; do not add new actor-side writes in phase 1.
- Keep the current PPO loss contract intact while changing value inputs.
- Make the SparseDriveV2 frozen-backbone critic gated by config so rollback is easy.

### Task 1: Capture a Reproducible Critic Baseline

**Files:**
- Modify: `script/configs/sparsedrive_v2/ppo_closed_loop_sparsedrive_v2.yaml`
- Add: `tests/framework/test_value_net_inputs.py`

- [ ] **Step 1: Define a short, repeatable PPO baseline run**

Use a tiny debug profile derived from the existing PPO config:
- low `max_updates`
- fixed seed
- stable `shards_per_update`
- no unrelated config changes

- [ ] **Step 2: Record baseline critic metrics to compare against later**

Track at minimum:
- `explained_variance`
- `loss_v`
- `ret_mean`
- `ret_std`
- `done_rate`
- `reward_mean`

- [ ] **Step 3: Add a regression test for value input shape assumptions**

Test cases should verify:
- image-only critic input shape stays valid
- optional future status input shape can be added without breaking existing batch code

- [ ] **Step 4: Run targeted tests**

Run:
```bash
pytest tests/framework/test_actor_learner_batch_contract.py -q
pytest tests/framework/test_trajectory_objectives.py -q
```

- [ ] **Step 5: Save baseline notes**

Write down:
- config used
- average sign/range of `explained_variance`
- whether instability is immediate or appears after a few updates

### Task 2: Add Critic Diagnostics Before Changing Architecture

**Files:**
- Modify: `framework/lightning/actor_learner_module.py`
- Modify: `framework/lightning/trajectory_module.py`
- Modify: `framework/batch/actor_learner.py`
- Modify: `tests/framework/test_trajectory_objectives.py`

- [ ] **Step 1: Extend logged critic statistics**

Add W&B/stage metrics for:
- `value_pred_mean`
- `value_pred_std`
- `old_value_mean`
- `old_value_std`
- `ret_abs_mean`
- `value_error_mean`
- `value_error_std`

- [ ] **Step 2: Expose enough batch stats to explain value failure**

Extend `LoadedShardBatch` with optional aggregate stats needed for debugging, while keeping existing fields compatible.

- [ ] **Step 3: Add tests for new diagnostics**

Verify:
- metrics are numeric
- empty batches still behave safely
- PPO path still logs existing metrics

- [ ] **Step 4: Re-run baseline PPO and compare**

Hypothesis to test:
- if `value_pred_*` stays low-variance while `ret_std` is large, the critic is under-informed
- if `value_pred_mean` moves wildly relative to `ret_mean`, critic optimization may be unstable

- [ ] **Step 5: Commit diagnostics only**

```bash
git add framework/lightning/actor_learner_module.py framework/lightning/trajectory_module.py framework/batch/actor_learner.py tests/framework/test_trajectory_objectives.py
git commit -m "chore: add critic diagnostics for PPO debugging"
```

### Task 3: Add a SparseDriveV2 Value Feature Interface Without Changing Shard Schema

**Files:**
- Modify: `framework/agent/base.py`
- Modify: `framework/agent/policy_sparsedrive_v2.py`
- Modify: `framework/lightning/trajectory_module.py`
- Add: `tests/framework/test_value_net_inputs.py`

- [ ] **Step 1: Define an optional agent interface for critic features**

Add an optional agent method such as `value_features_from_replay_batch(replays)` and a feature-dimension hook. Do not make it mandatory for non-SparseDrive agents.

- [ ] **Step 2: Implement SparseDriveV2 frozen feature extraction**

For SparseDriveV2:
- build batched replay features from `camera_feature` and `status_feature`
- run `_backbone` in eval/no-grad mode
- global-pool the last feature map to a fixed vector
- run `_status_encoding` in eval/no-grad mode
- concatenate pooled vision features and status encoding
- return detached critic features

- [ ] **Step 3: Keep compatibility checks strict**

If replay is malformed or missing required SparseDriveV2 fields, fail loudly instead of silently using bad value features.

- [ ] **Step 4: Add tests**

Cover:
- extracted feature shape is stable
- returned features do not require gradients
- replay shape mismatches fail loudly

- [ ] **Step 5: Verify existing trajectory tests still pass**

Run:
```bash
pytest tests/framework/test_trajectory_objectives.py -q
pytest tests/framework/test_value_net_inputs.py -q
```

### Task 4: Replace the Current Image-Only Critic with a Frozen-Backbone Value Head

**Files:**
- Modify: `framework/runner/learner_factory.py`
- Modify: `framework/lightning/trajectory_module.py`
- Modify: `framework/agent/base.py`
- Modify: `framework/agent/policy_sparsedrive_v2.py`
- Add: `tests/framework/test_value_net_inputs.py`

- [ ] **Step 1: Replace the current CNN critic with a value head**

The new module should take precomputed SparseDriveV2 value features as input. It should not own a vision backbone.

- [ ] **Step 2: Keep the value head lightweight**

Recommended first version:
- input dim = `pooled_backbone_dim + status_encoding_dim`
- 1-2 linear layers
- ReLU
- scalar value output

- [ ] **Step 3: Update PPO training_step to prefer agent value features**

Behavior:
- if agent exposes `value_features_from_replay_batch`, use that path
- otherwise keep the existing obs-based fallback for compatibility

- [ ] **Step 4: Gate the new critic by config**

Suggested config flags:
- `critic_use_agent_features: true/false`
- `critic_hidden_dim`

Default recommendation:
- enable for `sparsedrive_v2` PPO configs under active training

- [ ] **Step 5: Add shape and forward-pass tests**

Verify:
- obs-based fallback still works
- SparseDriveV2 feature-head mode works
- batch size 0/1/N cases are safe

- [ ] **Step 6: Commit the frozen-backbone value head**

```bash
git add framework/runner/learner_factory.py framework/lightning/trajectory_module.py framework/agent/base.py framework/agent/policy_sparsedrive_v2.py tests/framework/test_value_net_inputs.py
git commit -m "feat: add SparseDriveV2 frozen-backbone value head"
```

### Task 5: Tune Critic Optimization Separately from Policy

**Files:**
- Modify: `framework/runner/config_normalization.py`
- Modify: `script/configs/sparsedrive_v2/ppo_closed_loop_sparsedrive_v2.yaml`

- [ ] **Step 1: Expose critic-specific knobs clearly**

Make sure the config surface is explicit for:
- `lr_value`
- `vf_coef`
- `critic_use_agent_features`
- optional critic hidden dim if needed

- [ ] **Step 2: Run the narrowest useful sweep**

Recommended order:
1. keep architecture fixed, lower `lr_value` if value overshoots
2. if `loss_v` dominates, reduce `vf_coef`
3. only then widen the critic hidden size

- [ ] **Step 3: Compare against phase 0 baseline**

Require improvement on:
- median `explained_variance`
- volatility of `loss_v`
- no regression in `reward_mean`

- [ ] **Step 4: Keep one winning config only**

Do not land multiple speculative PPO configs in parallel. Preserve one clear default.

### Task 6: Verify SparseDriveV2 End-to-End

**Files:**
- Use: `tests/framework/fixtures/tiny_actor_learner_smoke.yaml`

- [ ] **Step 1: Run targeted tests**

```bash
pytest tests/framework/test_trajectory_objectives.py -q
pytest tests/framework/test_value_net_inputs.py -q
```

- [ ] **Step 2: Run a tiny smoke actor-learner PPO job**

Use the smallest practical config derived from:
- `tests/framework/fixtures/tiny_actor_learner_smoke.yaml`

Goal:
- verify the learner can consume shards
- verify the SparseDriveV2 frozen-backbone critic path runs
- verify no protocol breakage

- [ ] **Step 3: Compare baseline vs upgraded critic**

Required summary:
- before/after `explained_variance`
- before/after `loss_v`
- before/after `reward_mean`
- whether instability shifted from critic to policy metrics

## Recommendation Summary

Implement in this order:

1. diagnostics
2. SparseDriveV2 frozen feature interface
3. frozen-backbone value head
4. critic-only hyperparameter sweep

This ordering is recommended because it attacks the most likely root cause first: critic information mismatch. It also keeps risk low by preserving the actor-learner protocol and by preventing value loss from perturbing the policy backbone.

## Verification Checklist

- `explained_variance` no longer stays heavily negative for most updates
- `loss_v` magnitude and variance are both lower than baseline
- PPO policy metrics remain within normal range
- tests covering the objective path and SparseDriveV2 critic feature path still pass
- tiny smoke training run completes without shard/schema regression

## Rollback Plan

If the new critic path destabilizes learner execution:

- disable `critic_use_agent_features`
- fall back to the original image-only critic
- keep diagnostics logging enabled
- investigate whether the failure is feature shape mismatch, optimization instability, or SparseDriveV2 replay incompatibility
