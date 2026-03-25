from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from .trajectory_batch import compute_gae, normalize_advantages
from .trajectory_policy_core import agent_logp_from_replay_batch, compute_ppo_objective


@dataclass
class PPOUpdateResult:
    loss_pi: float
    loss_v: float
    approx_kl: float
    ratio_mean: float
    adv_mean: float


def _optimizer_grad_params(optimizer: torch.optim.Optimizer) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                params.append(param)
    return params


def ppo_update(
    *,
    agent: Any,
    value_net: torch.nn.Module,
    value_optim: torch.optim.Optimizer,
    obs_batch: torch.Tensor,
    old_logp: torch.Tensor,
    old_value: Optional[torch.Tensor],
    adv: torch.Tensor,
    ret: torch.Tensor,
    replay: List[Dict[str, Any]],
    device: torch.device,
    eta: float,
    clip_eps: float,
    vf_coef: float,
    ppo_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    grad_accum_steps: int,
    ddp_enabled: bool,
    world_size: int,
    rank: int,
    ddp_seed: int,
    update_seed: int,
    value_clip_eps: float = 0.0,
    kl_coef: float = 0.0,
    dual_clip: float | None = None,
    use_distributed_sampler: bool = True,
) -> PPOUpdateResult:
    optimizer = getattr(agent, "optimizer", None)
    if optimizer is None:
        raise RuntimeError("ppo_update requires agent.optimizer")

    policy_module = getattr(agent, "trainable_module", None)
    if policy_module is None:
        raise RuntimeError("ppo_update requires agent.trainable_module")

    sample_count = int(obs_batch.shape[0])
    if sample_count == 0:
        return PPOUpdateResult(0.0, 0.0, 0.0, 0.0, 0.0)
    if len(replay) != sample_count:
        raise RuntimeError(f"Replay length mismatch: len(replay)={len(replay)} n={sample_count}")

    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)
    value_optim.zero_grad(set_to_none=True)

    shuffle_indices = np.arange(sample_count)
    sampler: Optional[DistributedSampler] = None
    if ddp_enabled and use_distributed_sampler:
        sampler = DistributedSampler(
            list(range(sample_count)),
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=True,
            drop_last=False,
        )

    last_loss_pi = 0.0
    last_loss_v = 0.0
    last_approx_kl = 0.0
    last_ratio_mean = 0.0
    last_adv_mean = 0.0
    accum_i = 0

    for epoch in range(int(ppo_epochs)):
        if sampler is not None:
            sampler.set_epoch(int(ddp_seed) + int(update_seed) * 1000 + int(epoch))
            minibatch_indices = list(iter(sampler))
        else:
            np.random.shuffle(shuffle_indices)
            minibatch_indices = shuffle_indices.tolist()

        for start in range(0, len(minibatch_indices), int(minibatch_size)):
            mb_idx = minibatch_indices[start : start + int(minibatch_size)]
            if len(mb_idx) == 0:
                continue

            mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=device)
            replay_mb = [replay[i] for i in mb_idx]
            new_logp = agent_logp_from_replay_batch(agent, replay_mb, device=device, eta=float(eta))
            value_pred = value_net(obs_batch[mb_idx_t]).view(-1)
            old_value_mb = None
            if torch.is_tensor(old_value) and int(old_value.numel()) == int(obs_batch.shape[0]):
                old_value_mb = old_value[mb_idx_t]

            objective = compute_ppo_objective(
                new_logp=new_logp,
                old_logp=old_logp[mb_idx_t],
                adv=adv[mb_idx_t],
                ret=ret[mb_idx_t],
                value_pred=value_pred,
                old_value=old_value_mb,
                clip_eps=float(clip_eps),
                vf_coef=float(vf_coef),
                value_clip_eps=float(value_clip_eps),
                kl_coef=float(kl_coef),
                dual_clip=dual_clip,
            )

            loss = objective.loss / float(grad_accum_steps)
            sync_now = ((accum_i + 1) % grad_accum_steps) == 0
            policy_cm = nullcontext()
            value_cm = nullcontext()
            if ddp_enabled and hasattr(policy_module, "no_sync") and not sync_now:
                policy_cm = policy_module.no_sync()
            if ddp_enabled and hasattr(value_net, "no_sync") and not sync_now:
                value_cm = value_net.no_sync()

            with policy_cm, value_cm:
                loss.backward()

            accum_i += 1
            if sync_now:
                policy_grad_params = _optimizer_grad_params(optimizer)
                if len(policy_grad_params) > 0:
                    torch.nn.utils.clip_grad_norm_(policy_grad_params, float(max_grad_norm))
                value_grad_params = _optimizer_grad_params(value_optim)
                if len(value_grad_params) > 0:
                    torch.nn.utils.clip_grad_norm_(value_grad_params, float(max_grad_norm))
                optimizer.step()
                value_optim.step()
                optimizer.zero_grad(set_to_none=True)
                value_optim.zero_grad(set_to_none=True)

            last_loss_pi = float(objective.loss_pi.detach().cpu().item())
            last_loss_v = float(objective.loss_v.detach().cpu().item())
            last_approx_kl = float(objective.approx_kl.detach().cpu().item())
            last_ratio_mean = float(objective.ratio_mean.detach().cpu().item())
            last_adv_mean = float(objective.adv_mean.detach().cpu().item())

    if (accum_i % grad_accum_steps) != 0:
        policy_grad_params = _optimizer_grad_params(optimizer)
        if len(policy_grad_params) > 0:
            torch.nn.utils.clip_grad_norm_(policy_grad_params, float(max_grad_norm))
        value_grad_params = _optimizer_grad_params(value_optim)
        if len(value_grad_params) > 0:
            torch.nn.utils.clip_grad_norm_(value_grad_params, float(max_grad_norm))
        optimizer.step()
        value_optim.step()
        optimizer.zero_grad(set_to_none=True)
        value_optim.zero_grad(set_to_none=True)

    return PPOUpdateResult(
        loss_pi=float(last_loss_pi),
        loss_v=float(last_loss_v),
        approx_kl=float(last_approx_kl),
        ratio_mean=float(last_ratio_mean),
        adv_mean=float(last_adv_mean),
    )


__all__ = [
    "PPOUpdateResult",
    "compute_gae",
    "normalize_advantages",
    "ppo_update",
]