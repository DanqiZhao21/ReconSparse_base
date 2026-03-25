from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class LoadedShardBatch:
    batch: Dict[str, Any]
    num_samples: int
    reward_sum: float
    reward_count: int
    done_sum: float
    done_count: int


def compute_gae(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    terminated: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(lambda) with explicit terminated vs done semantics.

    `dones` stops the recursive carry across episode boundaries.
    `terminated` controls whether the TD target may bootstrap on the current step.
    When `terminated` is omitted, it defaults to `dones` for backward compatibility.
    """
    if rewards.ndim != 1 or dones.ndim != 1 or values.ndim != 1:
        raise ValueError("compute_gae expects 1D tensors (T,)")
    if rewards.shape[0] != dones.shape[0] or rewards.shape[0] != values.shape[0]:
        raise ValueError("compute_gae expects matching lengths")
    if terminated is None:
        terminated = dones
    if terminated.ndim != 1 or terminated.shape[0] != rewards.shape[0]:
        raise ValueError("terminated must be a 1D tensor matching rewards")

    rewards = rewards.to(dtype=torch.float32)
    dones = dones.to(device=rewards.device, dtype=torch.float32)
    values = values.to(device=rewards.device, dtype=torch.float32)
    terminated = terminated.to(device=rewards.device, dtype=torch.float32)
    last_value = last_value.to(device=rewards.device, dtype=torch.float32).view(())

    horizon = int(rewards.shape[0])
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(horizon)):
        not_done = 1.0 - dones[t]
        not_terminated = 1.0 - terminated[t]
        next_value = last_value if t == (horizon - 1) else values[t + 1]
        delta = rewards[t] + float(gamma) * next_value * not_terminated - values[t]
        last_gae = delta + float(gamma) * float(gae_lambda) * not_done * last_gae
        adv[t] = last_gae

    ret = adv + values
    return adv, ret


def compute_returns(*, rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    if rewards.ndim != 1 or dones.ndim != 1:
        raise ValueError("compute_returns expects 1D tensors (T,)")
    if rewards.shape[0] != dones.shape[0]:
        raise ValueError("compute_returns expects matching lengths")

    rewards = rewards.to(dtype=torch.float32)
    dones = dones.to(device=rewards.device, dtype=torch.float32)
    ret = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(int(rewards.shape[0]))):
        running = rewards[t] + float(gamma) * running * (1.0 - dones[t])
        ret[t] = running
    return ret


def normalize_advantages(
    adv: torch.Tensor,
    *,
    ddp_enabled: bool,
    dist_module: Any,
    device: torch.device,
    eps: float = 1e-8,
) -> torch.Tensor:
    if adv.numel() == 0:
        return adv

    if ddp_enabled and getattr(dist_module, "is_initialized", lambda: False)():
        stats = torch.stack(
            [
                adv.sum(),
                (adv * adv).sum(),
                torch.tensor(float(adv.numel()), device=device, dtype=adv.dtype),
            ],
            dim=0,
        )
        dist_module.all_reduce(stats, op=dist_module.ReduceOp.SUM)
        mean = stats[0] / stats[2]
        var = (stats[1] / stats[2]) - (mean * mean)
        std = torch.sqrt(torch.clamp(var, min=0.0) + float(eps))
        return (adv - mean) / std

    return (adv - adv.mean()) / (adv.std(unbiased=False) + float(eps))


def build_training_batch(
    *,
    selected: List[str],
    algo_key: str,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
    value_net: Optional[torch.nn.Module],
    value_optim: Optional[torch.optim.Optimizer],
    ddp_enabled: bool,
    dist_module: Any,
    norm_eps: float = 1e-8,
) -> LoadedShardBatch:
    obs_all: List[torch.Tensor] = []
    old_logp_all: List[torch.Tensor] = []
    old_value_all: List[torch.Tensor] = []
    adv_all: List[torch.Tensor] = []
    ret_all: List[torch.Tensor] = []
    replay_all: List[Dict[str, Any]] = []

    reward_sum = 0.0
    reward_count = 0
    done_sum = 0.0
    done_count = 0

    is_ppo_family = str(algo_key).startswith("ppo")

    if is_ppo_family:
        if value_net is None:
            raise RuntimeError("PPO batch build requires value_net")
        value_net.eval()
        with torch.inference_mode():
            for fp in selected:
                shard = torch.load(fp, map_location="cpu")
                obs_i = shard["obs"].to(device=device, dtype=torch.float32)
                old_logp_i = shard["old_logp"].to(device=device, dtype=torch.float32).view(-1)
                rewards_i = shard["reward"].to(device=device, dtype=torch.float32).view(-1)
                dones_i = shard["done"].to(device=device, dtype=torch.float32).view(-1)
                terminated_i = shard.get("terminated", dones_i)
                if torch.is_tensor(terminated_i):
                    terminated_i = terminated_i.to(device=device, dtype=torch.float32).view(-1)
                else:
                    terminated_i = dones_i
                replay_i = list(shard.get("replay", []))
                
                #统计整个 batch 的 reward/done 信息
                reward_sum += float(rewards_i.detach().sum().cpu().item())
                reward_count += int(rewards_i.numel())
                done_sum += float(dones_i.detach().sum().cpu().item())
                done_count += int(dones_i.numel())

                done_last = float(shard.get("done_last", float(dones_i[-1].item() if dones_i.numel() else 1.0)))
                terminated_last = float(
                    shard.get(
                        "terminated_last",
                        float(terminated_i[-1].item() if terminated_i.numel() else done_last),
                    )
                )
                next_obs = shard.get("next_obs", None)
                next_obs_t = obs_i[-1] if next_obs is None else next_obs.to(device=device, dtype=torch.float32)

                values_i = value_net(obs_i).detach().view(-1)
                bootstrap_allowed = terminated_last < 0.5
                last_value = (
                    value_net(next_obs_t.unsqueeze(0)).detach().view(-1)[0]
                    if bootstrap_allowed
                    else torch.zeros((), device=device, dtype=values_i.dtype)
                )
                adv_i, ret_i = compute_gae(
                    rewards=rewards_i,
                    dones=dones_i,
                    terminated=terminated_i,
                    values=values_i,
                    last_value=last_value,
                    gamma=float(gamma),
                    gae_lambda=float(gae_lambda),
                )

                obs_all.append(obs_i)
                old_logp_all.append(old_logp_i)
                old_value_all.append(values_i)
                adv_all.append(adv_i)
                ret_all.append(ret_i)
                replay_all.extend(replay_i)
    else:
        with torch.inference_mode():
            for fp in selected:
                shard = torch.load(fp, map_location="cpu")
                rewards_i = shard["reward"].to(device=device, dtype=torch.float32).view(-1)
                dones_i = shard["done"].to(device=device, dtype=torch.float32).view(-1)
                old_logp_i = shard.get("old_logp", torch.empty((0,), dtype=torch.float32))
                replay_i = list(shard.get("replay", []))

                reward_sum += float(rewards_i.detach().sum().cpu().item())
                reward_count += int(rewards_i.numel())
                done_sum += float(dones_i.detach().sum().cpu().item())
                done_count += int(dones_i.numel())

                ret_i = compute_returns(rewards=rewards_i, dones=dones_i, gamma=float(gamma))
                adv_i = ret_i
                if torch.is_tensor(old_logp_i) and int(old_logp_i.numel()) > 0:
                    old_logp_all.append(old_logp_i.to(device=device, dtype=torch.float32).view(-1))
                adv_all.append(adv_i)
                ret_all.append(ret_i)
                replay_all.extend(replay_i)

    obs_batch = torch.cat(obs_all, dim=0) if len(obs_all) else torch.empty((0, 18, 64, 64), device=device)
    old_logp = torch.cat(old_logp_all, dim=0) if len(old_logp_all) else torch.empty((0,), device=device)
    old_value = torch.cat(old_value_all, dim=0) if len(old_value_all) else torch.empty((0,), device=device)
    adv = torch.cat(adv_all, dim=0) if len(adv_all) else torch.empty((0,), device=device)
    ret = torch.cat(ret_all, dim=0) if len(ret_all) else torch.empty((0,), device=device)
    n = int(adv.shape[0])

    if is_ppo_family:
        if int(obs_batch.shape[0]) != n:
            raise RuntimeError(f"obs_batch length mismatch: obs={int(obs_batch.shape[0])} adv={n}")
        if int(old_logp.shape[0]) != n:
            raise RuntimeError(f"old_logp length mismatch: old_logp={int(old_logp.shape[0])} adv={n}")
        adv = normalize_advantages(adv, ddp_enabled=ddp_enabled, dist_module=dist_module, device=device, eps=float(norm_eps))
    else:
        if int(old_logp.numel()) not in {0, n}:
            raise RuntimeError(f"old_logp length mismatch: old_logp={int(old_logp.shape[0])} adv={n}")
        adv = normalize_advantages(adv, ddp_enabled=ddp_enabled, dist_module=dist_module, device=device, eps=float(norm_eps))

    if len(replay_all) != n:
        raise RuntimeError(f"replay_all length mismatch: len={len(replay_all)} n={n}")

    batch = {
        "obs_batch": obs_batch,
        "old_logp": old_logp,
        "old_value": old_value,
        "adv": adv,
        "ret": ret,
        "replay": replay_all,
        "value_net": value_net,
        "value_optim": value_optim,
    }
    return LoadedShardBatch(
        batch=batch,
        num_samples=n,
        reward_sum=float(reward_sum),
        reward_count=int(reward_count),
        done_sum=float(done_sum),
        done_count=int(done_count),
    )


__all__ = [
    "LoadedShardBatch",
    "build_training_batch",
    "compute_gae",
    "compute_returns",
    "normalize_advantages",
]