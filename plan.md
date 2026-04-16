# Closed-Loop Reward And GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade closed-loop reward to a path-based smooth reward with progress, then add learner-side counterfactual GRPO on top of the existing ReinforcePP path.

**Architecture:** Keep reward shaping on the actor/env side inside `framework/rewards/` and `framework/env_wrapper/`, while keeping counterfactual scoring and GRPO loss on the learner side inside `framework/agent/`, `framework/algorithms/`, and `framework/lightning/`. Preserve the current actor-learner shard protocol and add only backward-compatible replay/loss hooks.

**Tech Stack:** Python, PyTorch, PyTorch Lightning, pytest, existing actor-learner runtime

---

## Reward Design To Implement

Target step reward:

\[
r_t
=
w_{prog} r^{prog}_t
- w_{lat} r^{lat}_t
- w_{yaw} r^{yaw}_t
- w_{comfort} r^{comfort}_t
- r^{col}_t
+ r^{terminal}_t
\]

Where:

- `r_prog`: longitudinal progress along densified expert path
- `r_lat`: lateral deviation penalty with deadzone
- `r_yaw`: path-heading penalty with deadzone
- `r_comfort`: longitudinal jerk + yaw jerk penalty with deadzone
- `r_col`: static/dynamic collision penalties
- `r_terminal`: existing terminal penalty logic

Definitions:

- Project ego pose onto expert path and compute projected arc length `s_t`
- `delta_s_t = s_t - s_{t-1}`
- `r_prog_t = clip(delta_s_t, -progress_backward_cap_m, progress_forward_cap_m)`
- `r_lat_t = huber(max(0, |e_lat_t| - lateral_free_m), lateral_huber_delta_m)`
- `r_yaw_t = huber(max(0, |e_yaw_t| - yaw_free_deg), yaw_huber_delta_deg)`
- `r_comfort_t = huber(max(0, |jerk_t| - longitudinal_jerk_free), longitudinal_jerk_delta) + huber(max(0, |yaw_jerk_t| - yaw_jerk_free), yaw_jerk_delta)`

Optional auxiliary first-step anchor:

\[
r^{anchor}_t
=
w_{anchor\_progress} \cdot clip(s^{plan0}_t - s_t, 0, anchor_cap_m)
- w_{anchor\_lateral} \cdot huber(max(0, |e^{plan0}_{lat,t}| - anchor_free_m), anchor_free_m)
\]

First implementation round will add the config and plumbing for anchor terms but may leave the anchor reward disabled by default until the base path reward is verified.

## Task 1: Add Failing Reward Tests

**Files:**
- Create: `tests/test_tracking_reward_path_mode.py`
- Modify: `/root/clone/ReconDreamer-RL/plan.md`

- [x] **Step 1: Write failing tests for path progress reward**

Test cases:
- Progress increases when ego moves forward along a straight expert path
- Lateral penalty stays zero inside deadzone and increases outside it
- Yaw penalty uses wrapped angle and stays smooth around +/-180 degrees
- Terminal penalty behavior remains unchanged

- [x] **Step 2: Run the targeted reward tests to verify they fail**

Run: `pytest tests/test_tracking_reward_path_mode.py -q`
Expected: FAIL because path-based reward helpers/config are not implemented yet.

## Task 2: Implement Path-Based Smooth Reward

**Files:**
- Modify: `framework/rewards/tracking.py`
- Modify: `framework/rewards/README.md`

- [x] **Step 1: Add path helper utilities**

Helpers to add:
- densify expert polyline
- cumulative arc-length cache
- point-to-polyline projection
- path tangent heading lookup
- scalar Huber penalty helper

- [x] **Step 2: Extend `TrackingRewardComputer` state**

Add cached state for:
- reference path points
- cumulative path lengths
- previous projected progress `s_{t-1}`

- [x] **Step 3: Replace old threshold-only position reward**

Implement path-based:
- progress reward
- lateral penalty
- yaw penalty against path heading
- comfort penalty with deadzones
- collision penalty passthrough

- [x] **Step 4: Preserve and enrich logging fields**

Expose metrics such as:
- `progress_s`
- `progress_delta_s`
- `lateral_error_m`
- `path_heading_deg`
- `yaw_path_err_deg`
- per-term weighted contributions

- [x] **Step 5: Run targeted tests to verify they pass**

Run: `pytest tests/test_tracking_reward_path_mode.py -q`
Expected: PASS

## Task 3: Wire Reward Config

**Files:**
- Modify: `script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2.yaml`
- Modify: `script/configs/sparsedrive_v2/ppo_closed_loop_sparsedrive_v2.yaml`
- Modify: `script/configs/sparsedrive/ppo_closed_loop_sparsedrive.yaml`
- Modify: `script/configs/diffusiondrive_v2/ppo_closed_loop.yaml`

- [x] **Step 1: Add backward-compatible nested config structure**

Add sections:
- `reward.path`
- `reward.comfort`
- `reward.collision`
- retain existing `reward.terminal`

- [x] **Step 2: Keep defaults backward compatible**

Existing flat keys should still work when nested keys are absent.

- [x] **Step 3: Smoke-check config loading**

Run a narrow import/config normalization smoke check after config edits.

## Task 4: Review And Visual Logging Pass

**Files:**
- Modify: `framework/env_wrapper/rl_wrapper.py` (only if extra info propagation is needed)

- [x] **Step 1: Review reward outputs**

Verify logs now include enough per-step decomposition to inspect whether the reward matches human preferences.

- [x] **Step 2: Keep actor-learner protocol unchanged**

No shard schema change in this round.

## Task 5: Add Learner-Side Counterfactual Candidate Hook

**Files:**
- Modify: `framework/agent/policy_sparsedrive_v2.py`
- Add: `tests/test_sparsedrive_v2_counterfactual_candidates.py`

- [x] **Step 1: Write failing tests for counterfactual candidate extraction**
- [x] **Step 2: Add learner-side batch API returning candidate trajectories and candidate log-probs**
- [x] **Step 3: Verify targeted tests pass**

## Task 6: Add PDM/GRPO Objective

**Files:**
- Add: `framework/algorithms/pdm_scorer.py`
- Modify: `framework/algorithms/trajectory_policy_core.py`
- Modify: `framework/algorithms/reinforcepp.py`
- Modify: `framework/runner/learner_factory.py`
- Modify: `framework/lightning/config.py`
- Modify: `framework/lightning/trajectory_module.py`
- Add: `tests/test_reinforce_grpo_objective.py`

- [x] **Step 1: Write failing tests for GRPO objective math**
- [x] **Step 2: Implement group-relative normalized advantages from PDM scores**
- [x] **Step 3: Add `loss_total = loss_base + grpo_coef * loss_grpo` on ReinforcePP path only**
- [x] **Step 4: Add metrics/logging for group score stats and GRPO loss**
- [x] **Step 5: Verify targeted tests pass**

## Task 6b: Add NuScenes Direct Counterfactual Scorer

**Files:**
- Add: `framework/algorithms/nuscenes_token_scorer.py`
- Modify: `framework/agent/policy_sparsedrive_v2.py`
- Modify: `reconsimulator/envs/nus0331.py`
- Add: `tests/test_nuscenes_token_scorer.py`
- Modify: `tests/test_sparsedrive_v2_counterfactual_candidates.py`

- [x] **Step 1: Write failing tests for NuScenes token scorer and replay metadata**
- [x] **Step 2: Expose `sample_token` in observation/replay metadata**
- [x] **Step 3: Implement direct NuScenes scorer backed by `assets/nus/information/token2vad.pkl`**
- [x] **Step 4: Verify targeted tests pass**

## Task 6c: Add NuScenes GRPO Debug Visualization

**Files:**
- Modify: `framework/algorithms/nuscenes_token_scorer.py`
- Modify: `framework/agent/policy_sparsedrive_v2.py`
- Modify: `framework/algorithms/reinforcepp.py`
- Modify: `framework/runner/learner_factory.py`
- Modify: `framework/lightning/config.py`
- Modify: `framework/lightning/trajectory_module.py`
- Modify: `script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2.yaml`
- Modify: `tests/test_nuscenes_token_scorer.py`
- Modify: `tests/test_trajectory_module_grpo.py`

- [x] **Step 1: Write failing tests for debug artifact dumping and learner hook triggering**
- [x] **Step 2: Add scorer-side score breakdown export and top-k trajectory visualization dumping**
- [x] **Step 3: Add learner config plumbing and optional debug dump hook call**
- [x] **Step 4: Verify targeted tests pass and run a real token2vad smoke check**

## Task 7: Final Verification

**Files:**
- Modify: `plan.md`

- [x] **Step 1: Run targeted reward and GRPO tests**

Run:
- `pytest tests/test_tracking_reward_path_mode.py -q`
- `pytest tests/test_sparsedrive_v2_counterfactual_candidates.py -q`
- `pytest tests/test_reinforce_grpo_objective.py -q`

- [x] **Step 2: Run any additional narrow smoke tests needed for touched code paths**

- [x] **Step 3: Perform a review pass before declaring completion**
