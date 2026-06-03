from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from framework.rollout.timing import build_rollout_timing, extract_env_timing
from framework.utils.obs import obs_to_tensor


def _reward_summary_defaults() -> Dict[str, float]:
    return {
        "reward_sum": 0.0,
        "positive_reward_sum": 0.0,
        "gated_positive_reward_sum": 0.0,
        "cost_reward_sum": 0.0,
        "safety_gate_active_count": 0.0,
        "collision_gate_count": 0.0,
        "severe_tracking_lateral_gate_count": 0.0,
        "severe_tracking_yaw_gate_count": 0.0,
        "terminal_failure_count": 0.0,
        "terminal_timeout_count": 0.0,
        "terminal_env_done_count": 0.0,
        "ea_available_count": 0.0,
        "ea_evaluated_pairs_sum": 0.0,
        "ea_cost_sum": 0.0,
        "ea_risk_sum": 0.0,
        "ea_max_value": 0.0,
        "ea_min_value": 0.0,
        "ea_mean_sum": 0.0,
        "step_count": 0.0,
    }


def _accumulate_reward_summary(summary: Dict[str, float], info: Any, *, reward: float) -> None:
    summary["reward_sum"] += float(reward)
    summary["step_count"] += 1.0
    if not isinstance(info, dict):
        return
    summary["positive_reward_sum"] += float(info.get("positive_reward", 0.0) or 0.0)
    summary["gated_positive_reward_sum"] += float(info.get("gated_positive_reward", 0.0) or 0.0)
    summary["cost_reward_sum"] += float(info.get("cost_reward", 0.0) or 0.0)
    if bool(info.get("safety_gate_active", False)):
        summary["safety_gate_active_count"] += 1.0
    gate_sources = info.get("safety_gate_sources", [])
    if isinstance(gate_sources, (list, tuple)):
        gate_source_set = {str(item) for item in gate_sources}
    elif gate_sources:
        gate_source_set = {str(gate_sources)}
    else:
        gate_source_set = set()
    if "collision_constraint" in gate_source_set:
        summary["collision_gate_count"] += 1.0
    if "severe_tracking_lateral" in gate_source_set:
        summary["severe_tracking_lateral_gate_count"] += 1.0
    if "severe_tracking_yaw" in gate_source_set:
        summary["severe_tracking_yaw_gate_count"] += 1.0
    terminal_kind = str(info.get("terminal_kind", "") or "")
    if terminal_kind == "failure":
        summary["terminal_failure_count"] += 1.0
    elif terminal_kind == "timeout":
        summary["terminal_timeout_count"] += 1.0
    elif terminal_kind == "env_done":
        summary["terminal_env_done_count"] += 1.0
    if bool(info.get("ea_available", False)):
        summary["ea_available_count"] += 1.0
        summary["ea_evaluated_pairs_sum"] += float(info.get("ea_evaluated_pairs", 0.0) or 0.0)
        summary["ea_cost_sum"] += float(info.get("ea_cost", 0.0) or 0.0)
        summary["ea_risk_sum"] += float(info.get("ea_risk", 0.0) or 0.0)
        ea_max = float(info.get("ea_max", 0.0) or 0.0)
        ea_min = float(info.get("ea_min", ea_max) or 0.0)
        ea_mean = float(info.get("ea_mean", ea_max) or 0.0)
        if summary["ea_available_count"] <= 1.0:
            summary["ea_max_value"] = ea_max
            summary["ea_min_value"] = ea_min
        else:
            summary["ea_max_value"] = max(float(summary["ea_max_value"]), ea_max)
            summary["ea_min_value"] = min(float(summary["ea_min_value"]), ea_min)
        summary["ea_mean_sum"] += ea_mean


def _default_obs_tensor(obs: Any) -> torch.Tensor:
    try:
        return obs_to_tensor(obs, device=torch.device("cpu")).squeeze(0).detach().cpu()
    except Exception:
        return torch.zeros((18, 64, 64), dtype=torch.float32)


def _next_value_feature(agent: Any, next_observation: Any) -> Optional[torch.Tensor]:
    supports_fn = getattr(agent, "supports_value_features", None)
    if callable(supports_fn) and not bool(supports_fn()):
        return None
    batch_fn = getattr(agent, "value_features_from_observation_batch", None)
    if callable(batch_fn):
        features = batch_fn([next_observation])
        if torch.is_tensor(features) and int(features.shape[0]) > 0:
            return features[0].detach().cpu().to(dtype=torch.float32).clone()
    single_fn = getattr(agent, "value_features_from_observation", None)
    if callable(single_fn):
        feature = single_fn(next_observation)
        if torch.is_tensor(feature):
            return feature.detach().cpu().to(dtype=torch.float32).view(-1).clone()
    return None


def _extract_external_plan_local_xyyaw(replay: Any) -> Optional[np.ndarray]:
    if not isinstance(replay, dict):
        return None

    plan = replay.get("traj_xyyaw", None)
    if plan is None:
        return None

    if torch.is_tensor(plan):
        plan_arr = plan.detach().cpu().numpy()
    else:
        try:
            plan_arr = np.asarray(plan)
        except Exception:
            return None

    if plan_arr.ndim != 2 or plan_arr.shape[0] <= 0 or plan_arr.shape[1] < 3:
        return None

    return np.asarray(plan_arr[:, :3], dtype=np.float64).copy()


def _inject_external_plan_single_env(env: Any, replay: Any) -> None:
    plan_arr = _extract_external_plan_local_xyyaw(replay)
    if plan_arr is None:
        return

    setter = getattr(env, "set_external_plan_local_xyyaw", None)
    if callable(setter):
        setter(plan_arr)


def _inject_external_plan_vec_env(vec_env: Any, env_idx: int, replay: Any) -> None:
    plan_arr = _extract_external_plan_local_xyyaw(replay)
    if plan_arr is None:
        return

    call_one = getattr(vec_env, "call_one", None)
    if callable(call_one):
        call_one(int(env_idx), "set_external_plan_local_xyyaw", plan_arr)


def _inject_gt_reference_from_info(replay: Any, info: Any) -> None:
    if not isinstance(replay, dict) or not isinstance(info, dict):
        return
    gt_sample_token = info.get("grpo_gt_sample_token", info.get("recon_cache_sample_token", None))
    if gt_sample_token is not None and str(gt_sample_token):
        replay["gt_sample_token_override"] = str(gt_sample_token)
    gt_frame_idx = info.get("grpo_gt_frame_idx", info.get("recon_cache_frame_idx", None))
    if gt_frame_idx is not None:
        try:
            replay["gt_frame_idx_override"] = int(gt_frame_idx)
        except Exception:
            pass


def _emit_heartbeat(heartbeat_fn: Any | None, phase: str, step: int | None = None, *, force: bool = False) -> None:
    if not callable(heartbeat_fn):
        return
    try:
        heartbeat_fn(phase, step, force=bool(force))
    except TypeError:
        heartbeat_fn(phase, step)


def collect_single_env_shard(
    *,
    env: Any,
    agent: Any,
    obs: Any,
    horizon: int,
    eta: float,
    mode_idx: int,
    mode_select: str,
    actor_id: int,
    local_ver: int,
    shard_idx: int,
    store_obs: bool = True,
    info: Any = None,
    return_info: bool = False,
    heartbeat_fn: Any | None = None,
    end_shard_on_done: bool = False,
) -> tuple[Dict[str, Any], Any]:
    obs_buf: List[torch.Tensor] = []
    old_logp_buf: List[torch.Tensor] = []
    rew_buf: List[float] = []
    done_buf: List[float] = []
    terminated_buf: List[float] = []
    truncated_buf: List[float] = []
    replay_buf: List[Dict[str, Any]] = []

    last_next_obs_t: Optional[torch.Tensor] = None
    last_done = 1.0
    last_terminated = 1.0
    next_obs_after = obs
    current_info: Any = info
    step_records: List[Dict[str, float]] = []
    counters: Dict[str, int] = {"done_count": 0, "reset_count": 0}
    reward_summary = _reward_summary_defaults()
    needs_reset_after = False

    step_count = 0
    collect_t0 = time.perf_counter()
    while step_count < int(horizon):
        _emit_heartbeat(heartbeat_fn, "collect_single_step", step_count)
        obs_decision = obs
        step_timing: Dict[str, float] = {}
        obs_t: torch.Tensor | None = None
        if bool(store_obs):
            t0 = time.perf_counter()
            obs_t = _default_obs_tensor(obs_decision)
            step_timing["obs_tensor_s"] = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _emit_heartbeat(heartbeat_fn, "act_start", step_count, force=True)
        action0, logp, replay = agent.act(obs_decision, eta=eta, mode_idx=mode_idx, mode_select=mode_select)
        _emit_heartbeat(heartbeat_fn, "act_done", step_count, force=True)
        step_timing["act_s"] = float(time.perf_counter() - t0)
        _inject_gt_reference_from_info(replay, current_info)
        _inject_external_plan_single_env(env, replay)
        t0 = time.perf_counter()
        _emit_heartbeat(heartbeat_fn, "env_step_start", step_count, force=True)
        obs, reward, terminated, truncated, _info = env.step(action0)
        _emit_heartbeat(heartbeat_fn, "env_step_done", step_count, force=True)
        step_timing["env_step_s"] = float(time.perf_counter() - t0)
        step_timing.update(extract_env_timing(_info))
        done = bool(terminated or truncated)
        next_obs_after = obs
        _accumulate_reward_summary(reward_summary, _info, reward=float(reward))

        if obs_t is not None:
            obs_buf.append(obs_t)
        old_logp_buf.append(logp.detach().cpu().float())
        replay_buf.append(replay)
        rew_buf.append(float(reward))
        done_buf.append(1.0 if done else 0.0)
        terminated_buf.append(1.0 if bool(terminated) else 0.0)
        truncated_buf.append(1.0 if bool(truncated) else 0.0)
        step_count += 1

        if bool(store_obs):
            last_next_obs_t = _default_obs_tensor(next_obs_after)
        last_done = 1.0 if done else 0.0
        last_terminated = 1.0 if bool(terminated) else 0.0

        if done:
            counters["done_count"] += 1
            if bool(end_shard_on_done):
                needs_reset_after = True
                current_info = _info
                step_records.append(step_timing)
                break
            else:
                t0 = time.perf_counter()
                _emit_heartbeat(heartbeat_fn, "env_reset_start", step_count, force=True)
                obs, _info = env.reset()
                _emit_heartbeat(heartbeat_fn, "env_reset_done", step_count, force=True)
                step_timing["env_reset_s"] = float(time.perf_counter() - t0)
                counters["reset_count"] += 1
        current_info = _info
        step_records.append(step_timing)

    next_obs_t = None
    if bool(store_obs):
        next_obs_t = last_next_obs_t if last_next_obs_t is not None else _default_obs_tensor(obs)
    next_value_feature_t0 = time.perf_counter()
    next_value_feature = _next_value_feature(agent, next_obs_after if step_count > 0 else obs)
    next_value_feature_s = float(time.perf_counter() - next_value_feature_t0)
    timing = build_rollout_timing(
        horizon=int(horizon),
        step_records=step_records,
        collect_shard_s=float(time.perf_counter() - collect_t0),
        next_value_feature_s=float(next_value_feature_s),
        counters=counters,
    )
    shard = {
        "old_logp": torch.stack(old_logp_buf, dim=0).view(-1),
        "reward": torch.tensor(rew_buf, dtype=torch.float32),
        "done": torch.tensor(done_buf, dtype=torch.float32),
        "terminated": torch.tensor(terminated_buf, dtype=torch.float32),
        "truncated": torch.tensor(truncated_buf, dtype=torch.float32),
        "done_last": torch.tensor(float(last_done), dtype=torch.float32),
        "terminated_last": torch.tensor(float(last_terminated), dtype=torch.float32),
        "replay": replay_buf,
        "meta": {
            "actor_id": int(actor_id),
            "env_id": 0,
            "horizon": int(horizon),
            "num_steps": int(step_count),
            "needs_reset_after": bool(needs_reset_after),
            "weights_version": int(local_ver),
            "time": float(time.time()),
            "shard_idx": int(shard_idx),
            "timing": timing,
            "reward_summary": reward_summary,
        },
    }
    if bool(store_obs):
        shard["obs"] = torch.stack(obs_buf, dim=0)
        shard["next_obs"] = next_obs_t
    if next_value_feature is not None:
        shard["next_value_feature"] = next_value_feature
    obs_after = None if bool(needs_reset_after) else obs
    if bool(return_info):
        return shard, obs_after, current_info
    return shard, obs_after


def collect_vector_env_shards(
    *,
    vec_env: Any,
    agent: Any,
    obs_list: List[Any],
    num_envs_per_actor: int,
    horizon: int,
    eta: float,
    mode_idx: int,
    mode_select: str,
    actor_id: int,
    local_ver: int,
    shard_idx_per_env: List[int],
    store_obs: bool = True,
    info_list: List[Any] | None = None,
    return_info: bool = False,
    heartbeat_fn: Any | None = None,
) -> tuple[List[Dict[str, Any]], List[Any]]:
    obs_bufs: List[List[torch.Tensor]] = [[] for _ in range(int(num_envs_per_actor))]
    old_logp_bufs: List[List[torch.Tensor]] = [[] for _ in range(int(num_envs_per_actor))]
    rew_bufs: List[List[float]] = [[] for _ in range(int(num_envs_per_actor))]
    done_bufs: List[List[float]] = [[] for _ in range(int(num_envs_per_actor))]
    terminated_bufs: List[List[float]] = [[] for _ in range(int(num_envs_per_actor))]
    truncated_bufs: List[List[float]] = [[] for _ in range(int(num_envs_per_actor))]
    replay_bufs: List[List[Dict[str, Any]]] = [[] for _ in range(int(num_envs_per_actor))]
    last_next_obs_ts: List[Optional[torch.Tensor]] = [None for _ in range(int(num_envs_per_actor))]
    last_next_obs_raw: List[Any] = [None for _ in range(int(num_envs_per_actor))]
    last_dones: List[float] = [1.0 for _ in range(int(num_envs_per_actor))]
    last_terminateds: List[float] = [1.0 for _ in range(int(num_envs_per_actor))]
    step_records_by_env: List[List[Dict[str, float]]] = [[] for _ in range(int(num_envs_per_actor))]
    counters_by_env: List[Dict[str, int]] = [{"done_count": 0, "reset_count": 0} for _ in range(int(num_envs_per_actor))]
    reward_summaries_by_env: List[Dict[str, float]] = [_reward_summary_defaults() for _ in range(int(num_envs_per_actor))]
    if isinstance(info_list, (list, tuple)):
        current_info_list: List[Any] = list(info_list)
    elif info_list is not None:
        current_info_list = [info_list for _ in range(int(num_envs_per_actor))]
    else:
        current_info_list = [None for _ in range(int(num_envs_per_actor))]
    if len(current_info_list) < int(num_envs_per_actor):
        current_info_list.extend([None for _ in range(int(num_envs_per_actor) - len(current_info_list))])
    if len(current_info_list) > int(num_envs_per_actor):
        current_info_list = current_info_list[: int(num_envs_per_actor)]

    step_count = 0
    collect_t0 = time.perf_counter()
    while step_count < int(horizon):
        if callable(heartbeat_fn):
            heartbeat_fn("collect_vector_step", step_count)
        act_t0 = time.perf_counter()
        obs_t_list = [_default_obs_tensor(obs) for obs in obs_list] if bool(store_obs) else []
        actions0, logps, replays = agent.act_batch(
            obs_list,
            eta=eta,
            mode_idx=mode_idx,
            mode_select=mode_select,
        )
        act_s = float(time.perf_counter() - act_t0) / float(max(1, int(num_envs_per_actor)))
        for i, replay in enumerate(replays):
            _inject_gt_reference_from_info(replay, current_info_list[i] if i < len(current_info_list) else None)
            _inject_external_plan_vec_env(vec_env, i, replay)

        env_step_t0 = time.perf_counter()
        next_obs_list, reward_list, term_list, trunc_list, _info_list = vec_env.step(actions0)
        env_step_s = float(time.perf_counter() - env_step_t0) / float(max(1, int(num_envs_per_actor)))
        step_done = [False for _ in range(int(num_envs_per_actor))]
        step_next_obs: List[Any] = list(next_obs_list)
        next_current_info_list: List[Any] = list(_info_list)
        reset_s_list = [0.0 for _ in range(int(num_envs_per_actor))]
        for i in range(int(num_envs_per_actor)):
            done = bool(term_list[i] or trunc_list[i])
            step_done[i] = done
            if done:
                reset_t0 = time.perf_counter()
                o2, _info2 = vec_env.reset_one(i)
                reset_s_list[i] = float(time.perf_counter() - reset_t0)
                next_obs_list[i] = o2
                next_current_info_list[i] = _info2
                counters_by_env[i]["done_count"] += 1
                counters_by_env[i]["reset_count"] += 1
        obs_list = next_obs_list
        current_info_list = next_current_info_list

        for i in range(int(num_envs_per_actor)):
            if bool(store_obs):
                obs_bufs[i].append(obs_t_list[i])
            old_logp_bufs[i].append(logps[i].detach().cpu().float())
            replay_bufs[i].append(replays[i])
            rew_bufs[i].append(float(reward_list[i]))
            _accumulate_reward_summary(reward_summaries_by_env[i], _info_list[i], reward=float(reward_list[i]))
            done_bufs[i].append(1.0 if step_done[i] else 0.0)
            terminated_bufs[i].append(1.0 if bool(term_list[i]) else 0.0)
            truncated_bufs[i].append(1.0 if bool(trunc_list[i]) else 0.0)
            if bool(store_obs):
                last_next_obs_ts[i] = _default_obs_tensor(step_next_obs[i])
            last_next_obs_raw[i] = step_next_obs[i]
            last_dones[i] = 1.0 if step_done[i] else 0.0
            last_terminateds[i] = 1.0 if bool(term_list[i]) else 0.0
            step_timing: Dict[str, float] = {
                "act_s": float(act_s),
                "env_step_s": float(env_step_s),
            }
            step_timing.update(extract_env_timing(_info_list[i]))
            if step_done[i]:
                step_timing["env_reset_s"] = float(reset_s_list[i])
            step_records_by_env[i].append(step_timing)

        step_count += 1

    shards: List[Dict[str, Any]] = []
    for i in range(int(num_envs_per_actor)):
        next_obs_t = None
        if bool(store_obs):
            next_obs_t = last_next_obs_ts[i] if last_next_obs_ts[i] is not None else _default_obs_tensor(obs_list[i])
        next_value_feature_t0 = time.perf_counter()
        next_value_feature = _next_value_feature(
            agent,
            last_next_obs_raw[i] if last_next_obs_raw[i] is not None else obs_list[i],
        )
        next_value_feature_s = float(time.perf_counter() - next_value_feature_t0)
        timing = build_rollout_timing(
            horizon=int(horizon),
            step_records=step_records_by_env[i],
            collect_shard_s=float(time.perf_counter() - collect_t0),
            next_value_feature_s=float(next_value_feature_s),
            counters=counters_by_env[i],
        )
        shard_i = {
                "old_logp": torch.stack(old_logp_bufs[i], dim=0).view(-1),
                "reward": torch.tensor(rew_bufs[i], dtype=torch.float32),
                "done": torch.tensor(done_bufs[i], dtype=torch.float32),
                "terminated": torch.tensor(terminated_bufs[i], dtype=torch.float32),
                "truncated": torch.tensor(truncated_bufs[i], dtype=torch.float32),
                "done_last": torch.tensor(float(last_dones[i]), dtype=torch.float32),
                "terminated_last": torch.tensor(float(last_terminateds[i]), dtype=torch.float32),
                "replay": replay_bufs[i],
                "meta": {
                    "actor_id": int(actor_id),
                    "env_id": int(i),
                    "horizon": int(horizon),
                    "num_steps": int(len(rew_bufs[i])),
                    "weights_version": int(local_ver),
                    "time": float(time.time()),
                    "shard_idx": int(shard_idx_per_env[i]),
                    "timing": timing,
                    "reward_summary": reward_summaries_by_env[i],
                },
            }
        if bool(store_obs):
            shard_i["obs"] = torch.stack(obs_bufs[i], dim=0)
            shard_i["next_obs"] = next_obs_t
        shards.append(shard_i)
        if next_value_feature is not None:
            shards[-1]["next_value_feature"] = next_value_feature
    if bool(return_info):
        return shards, obs_list, current_info_list
    return shards, obs_list
