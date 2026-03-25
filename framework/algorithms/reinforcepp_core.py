from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .trajectory_batch import compute_returns as generic_compute_returns
from .trajectory_batch import normalize_advantages as generic_normalize_advantages
from .trajectory_policy_core import agent_logp_from_replay_batch


@dataclass
class ReinforcePPUpdateResult:
    loss_pi: float
    approx_kl: float
    adv_mean: float


def compute_returns(*, rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    return generic_compute_returns(rewards=rewards, dones=dones, gamma=float(gamma))


def _apply_group_mean_baseline_inplace(adv: torch.Tensor, group_ids: Sequence[Optional[int]]) -> torch.Tensor:
    """Subtract per-group mean from advantage.

    Only applies to entries with a non-None group id.
    """
    if adv.numel() == 0:
        return adv
    if len(group_ids) != int(adv.shape[0]):
        raise ValueError("group_ids length mismatch")

    idx = [i for i, g in enumerate(group_ids) if g is not None]
    if len(idx) == 0:
        return adv

    idx_t = torch.tensor(idx, dtype=torch.long, device=adv.device)
    g_t = torch.tensor([int(group_ids[i]) for i in idx], dtype=torch.long, device=adv.device)

    uniq = torch.unique(g_t)
    for g in uniq.tolist():
        mask = g_t == int(g)
        if int(mask.sum().item()) <= 1:
            continue
        sel = idx_t[mask]
        adv_mean = adv[sel].mean()
        adv[sel] = adv[sel] - adv_mean

    return adv


def normalize_advantages(
    adv: torch.Tensor,
    *,
    ddp_enabled: bool,
    dist_module: Any,
    device: torch.device,
    eps: float = 1e-8,
) -> torch.Tensor:
    return generic_normalize_advantages(
        adv,
        ddp_enabled=ddp_enabled,
        dist_module=dist_module,
        device=device,
        eps=float(eps),
    )


def reinforcepp_update(
    *,
    agent: Any,
    ref_agent: Any | None,
    adv: torch.Tensor,
    replay: List[Dict[str, Any]],
    device: torch.device,
    eta: float,
    kl_coef: float,
    epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    grad_accum_steps: int,
    ddp_enabled: bool,
    world_size: int,
    rank: int,
    ddp_seed: int,
    update_seed: int,
    use_distributed_sampler: bool = True,
) -> ReinforcePPUpdateResult:
    """Reinforce++ update using the agent replay-logp interface.

    The algorithm is generic; agent-specific replay decoding stays behind
    `agent.logp_from_replay_batch()`.
    """
    optimizer = getattr(agent, "optimizer", None)
    if optimizer is None:
        raise RuntimeError("reinforcepp_update requires agent.optimizer")
    policy_module = getattr(agent, "trainable_module", None)
    if policy_module is None:
        raise RuntimeError("reinforcepp_update requires agent.trainable_module")

    n = int(adv.shape[0])
    if n == 0:
        return ReinforcePPUpdateResult(0.0, 0.0, 0.0)
    if len(replay) != n:
        raise RuntimeError(f"Replay length mismatch: len(replay)={len(replay)} n={n}")

    grad_accum_steps = max(1, int(grad_accum_steps))

    optimizer.zero_grad(set_to_none=True)

    idxs = np.arange(n)

    last_loss_pi = 0.0
    last_approx_kl = 0.0
    last_adv_mean = 0.0

    ds = list(range(n))
    sampler = None
    if ddp_enabled and use_distributed_sampler:
        try:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(ds, num_replicas=int(world_size), rank=int(rank), shuffle=True, drop_last=False)
        except Exception:
            sampler = None

    accum_i = 0
    for ep in range(int(epochs)):
        if sampler is not None:
            sampler.set_epoch(int(ddp_seed) + int(update_seed) * 1000 + int(ep))
            mb_indices_iter = list(iter(sampler))
        else:
            np.random.shuffle(idxs)
            mb_indices_iter = idxs.tolist()

        for start in range(0, len(mb_indices_iter), int(minibatch_size)):
            mb_idx = mb_indices_iter[start : start + int(minibatch_size)]
            if len(mb_idx) == 0:
                continue

            mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=device)
            adv_mb = adv[mb_idx_t].detach()
            replay_mb = [replay[i] for i in mb_idx]
            new_logp_vec = agent_logp_from_replay_batch(
                agent,
                replay_mb,
                device=device,
                eta=float(eta),
            )

            loss_pi = -(adv_mb * new_logp_vec).mean()

            approx_kl = torch.zeros((), device=device, dtype=torch.float32)
            if float(kl_coef) > 0.0 and ref_agent is not None:
                with torch.inference_mode():
                    ref_logp_vec = agent_logp_from_replay_batch(
                        ref_agent,
                        replay_mb,
                        device=device,
                        eta=float(eta),
                    )

                approx_kl = (new_logp_vec - ref_logp_vec).mean().detach()
                loss_pi = loss_pi + float(kl_coef) * (new_logp_vec - ref_logp_vec).mean()

            loss = loss_pi / float(grad_accum_steps)

            sync_now = ((accum_i + 1) % grad_accum_steps) == 0
            cm = nullcontext()
            if ddp_enabled and hasattr(policy_module, "no_sync") and not sync_now:
                cm = policy_module.no_sync()
            with cm:
                loss.backward()

            accum_i += 1
            if sync_now:
                grad_params = []
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            grad_params.append(param)
                if len(grad_params) > 0:
                    torch.nn.utils.clip_grad_norm_(grad_params, float(max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            last_loss_pi = float(loss_pi.detach().cpu().item())
            last_approx_kl = float(approx_kl.detach().cpu().item())
            last_adv_mean = float(adv_mb.detach().mean().cpu().item())

    if (accum_i % grad_accum_steps) != 0:
        grad_params = []
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    grad_params.append(param)
        if len(grad_params) > 0:
            torch.nn.utils.clip_grad_norm_(grad_params, float(max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return ReinforcePPUpdateResult(
        loss_pi=float(last_loss_pi),
        approx_kl=float(last_approx_kl),
        adv_mean=float(last_adv_mean),
    )


__all__ = [
    "ReinforcePPUpdateResult",
    "compute_returns",
    "_apply_group_mean_baseline_inplace",
    "normalize_advantages",
    "reinforcepp_update",
]
