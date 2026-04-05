from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from framework.utils.obs import obs_to_tensor


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

    step_count = 0
    while step_count < int(horizon):
        obs_decision = obs
        obs_t = _default_obs_tensor(obs_decision)
        action0, logp, replay = agent.act(obs_decision, eta=eta, mode_idx=mode_idx, mode_select=mode_select)
        obs, reward, terminated, truncated, _info = env.step(action0)
        done = bool(terminated or truncated)
        next_obs_after = obs

        obs_buf.append(obs_t)
        old_logp_buf.append(logp.detach().cpu().float())
        replay_buf.append(replay)
        rew_buf.append(float(reward))
        done_buf.append(1.0 if done else 0.0)
        terminated_buf.append(1.0 if bool(terminated) else 0.0)
        truncated_buf.append(1.0 if bool(truncated) else 0.0)
        step_count += 1

        last_next_obs_t = _default_obs_tensor(next_obs_after)
        last_done = 1.0 if done else 0.0
        last_terminated = 1.0 if bool(terminated) else 0.0

        if done:
            obs, _info = env.reset()

    next_obs_t = last_next_obs_t if last_next_obs_t is not None else _default_obs_tensor(obs)
    next_value_feature = _next_value_feature(agent, next_obs_after if step_count > 0 else obs)
    shard = {
        "obs": torch.stack(obs_buf, dim=0),
        "old_logp": torch.stack(old_logp_buf, dim=0).view(-1),
        "reward": torch.tensor(rew_buf, dtype=torch.float32),
        "done": torch.tensor(done_buf, dtype=torch.float32),
        "terminated": torch.tensor(terminated_buf, dtype=torch.float32),
        "truncated": torch.tensor(truncated_buf, dtype=torch.float32),
        "next_obs": next_obs_t,
        "done_last": torch.tensor(float(last_done), dtype=torch.float32),
        "terminated_last": torch.tensor(float(last_terminated), dtype=torch.float32),
        "replay": replay_buf,
        "meta": {
            "actor_id": int(actor_id),
            "env_id": 0,
            "horizon": int(horizon),
            "weights_version": int(local_ver),
            "time": float(time.time()),
            "shard_idx": int(shard_idx),
        },
    }
    if next_value_feature is not None:
        shard["next_value_feature"] = next_value_feature
    return shard, obs


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

    step_count = 0
    while step_count < int(horizon):
        obs_t_list = [_default_obs_tensor(obs) for obs in obs_list]
        actions0, logps, replays = agent.act_batch(
            obs_list,
            eta=eta,
            mode_idx=mode_idx,
            mode_select=mode_select,
        )

        next_obs_list, reward_list, term_list, trunc_list, _info_list = vec_env.step(actions0)
        step_done = [False for _ in range(int(num_envs_per_actor))]
        step_next_obs: List[Any] = list(next_obs_list)
        for i in range(int(num_envs_per_actor)):
            done = bool(term_list[i] or trunc_list[i])
            step_done[i] = done
            if done:
                o2, _info2 = vec_env.reset_one(i)
                next_obs_list[i] = o2
        obs_list = next_obs_list

        for i in range(int(num_envs_per_actor)):
            obs_bufs[i].append(obs_t_list[i])
            old_logp_bufs[i].append(logps[i].detach().cpu().float())
            replay_bufs[i].append(replays[i])
            rew_bufs[i].append(float(reward_list[i]))
            done_bufs[i].append(1.0 if step_done[i] else 0.0)
            terminated_bufs[i].append(1.0 if bool(term_list[i]) else 0.0)
            truncated_bufs[i].append(1.0 if bool(trunc_list[i]) else 0.0)
            last_next_obs_ts[i] = _default_obs_tensor(step_next_obs[i])
            last_next_obs_raw[i] = step_next_obs[i]
            last_dones[i] = 1.0 if step_done[i] else 0.0
            last_terminateds[i] = 1.0 if bool(term_list[i]) else 0.0

        step_count += 1

    shards: List[Dict[str, Any]] = []
    for i in range(int(num_envs_per_actor)):
        next_obs_t = last_next_obs_ts[i] if last_next_obs_ts[i] is not None else _default_obs_tensor(obs_list[i])
        shards.append(
            {
                "obs": torch.stack(obs_bufs[i], dim=0),
                "old_logp": torch.stack(old_logp_bufs[i], dim=0).view(-1),
                "reward": torch.tensor(rew_bufs[i], dtype=torch.float32),
                "done": torch.tensor(done_bufs[i], dtype=torch.float32),
                "terminated": torch.tensor(terminated_bufs[i], dtype=torch.float32),
                "truncated": torch.tensor(truncated_bufs[i], dtype=torch.float32),
                "next_obs": next_obs_t,
                "done_last": torch.tensor(float(last_dones[i]), dtype=torch.float32),
                "terminated_last": torch.tensor(float(last_terminateds[i]), dtype=torch.float32),
                "replay": replay_bufs[i],
                "meta": {
                    "actor_id": int(actor_id),
                    "env_id": int(i),
                    "horizon": int(horizon),
                    "weights_version": int(local_ver),
                    "time": float(time.time()),
                    "shard_idx": int(shard_idx_per_env[i]),
                },
            }
        )
        next_value_feature = _next_value_feature(
            agent,
            last_next_obs_raw[i] if last_next_obs_raw[i] is not None else obs_list[i],
        )
        if next_value_feature is not None:
            shards[-1]["next_value_feature"] = next_value_feature
    return shards, obs_list
