from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


@dataclass
class ReinforcePPUpdateResult:
    loss_pi: float
    approx_kl: float
    adv_mean: float


def compute_returns(*, rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute reward-to-go returns for a single rollout.

    Args:
        rewards: (T,) float
        dones: (T,) float/bool, 1.0 when episode ended after this step.
        gamma: discount factor

    Returns:
        ret: (T,) discounted reward-to-go

    Notes:
        For truncated rollouts where the episode continues past T, this implicitly
        assumes a bootstrap value of 0 beyond the truncation.
    """
    if rewards.ndim != 1 or dones.ndim != 1:
        raise ValueError("compute_returns expects 1D tensors (T,)")
    if rewards.shape[0] != dones.shape[0]:
        raise ValueError("compute_returns expects matching lengths")

    T = int(rewards.shape[0])
    ret = torch.zeros_like(rewards)
    g = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        g = rewards[t] + float(gamma) * g * mask
        ret[t] = g
    return ret


def _apply_group_mean_baseline_inplace(adv: torch.Tensor, group_ids: Sequence[Optional[int]]) -> torch.Tensor:
    """Subtract per-group mean from advantage.

    Only applies to entries with a non-None group id.
    """
    if adv.numel() == 0:
        return adv
    if len(group_ids) != int(adv.shape[0]):
        raise ValueError("group_ids length mismatch")

    # Build tensors for entries that have a group id.
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
    """Global advantage normalization (Reinforce++ core idea).

    If DDP is enabled, compute global mean/std via all_reduce without all_gather.
    """
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


def ddv2_reinforcepp_update(
    *,
    agent: Any,
    ref_agent: Any | None,
    adv: torch.Tensor,
    replay: List[Dict[str, Any]],
    device: torch.device,
    ddv2_eta: float,
    ddv2_mode_idx_default: int,
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
    replay_compute_camera_dtype: torch.dtype = torch.float32,
    replay_compute_chain_dtype: torch.dtype = torch.float32,
    use_distributed_sampler: bool = True,
) -> ReinforcePPUpdateResult:
    """Reinforce++ update for DDV2 policy using diffusion replay.

    Implements a critic-free policy-gradient update with Global Advantage Normalization.

    Policy loss:
        L_pi = -E[ A_norm * logpi_theta(a|s) ] + kl_coef * E[ logpi_theta(a|s) - logpi_ref(a|s) ]

    where the KL term uses a single-sample estimator under the current policy.
    """
    if agent is None or getattr(agent, "_ddv2_optimizer", None) is None:
        raise RuntimeError("ddv2_reinforcepp_update requires agent._ddv2_optimizer")

    n = int(adv.shape[0])
    if n == 0:
        return ReinforcePPUpdateResult(0.0, 0.0, 0.0)
    if len(replay) != n:
        raise RuntimeError(f"Replay length mismatch: len(replay)={len(replay)} n={n}")

    ddp_model = agent._agent._transfuser_model
    core = ddp_model.module if hasattr(ddp_model, "module") else ddp_model

    grad_accum_steps = max(1, int(grad_accum_steps))

    agent._ddv2_optimizer.zero_grad(set_to_none=True)

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

            cam = torch.cat([replay[i]["camera_feature"] for i in mb_idx], dim=0)
            chain = torch.cat([replay[i]["diffusion_chain"] for i in mb_idx], dim=0)
            mb_mode_idx = torch.as_tensor(
                [int(replay[i].get("mode_idx", ddv2_mode_idx_default)) for i in mb_idx],
                dtype=torch.long,
                device=device,
            )

            features = {
                "camera_feature": cam.to(device=device, dtype=replay_compute_camera_dtype),
                "lidar_feature": torch.zeros((cam.shape[0], 1, 256, 256), dtype=torch.float32, device=device),
                "status_feature": torch.zeros((cam.shape[0], 8), dtype=torch.float32, device=device),
            }

            all_logps = core.compute_log_probs_from_diffusion_chain(
                features,
                chain.to(device=device, dtype=replay_compute_chain_dtype),
                eta=float(ddv2_eta),
            )
            bsz = int(cam.shape[0])
            sel = all_logps[torch.arange(bsz, device=device), mb_mode_idx, :]
            new_logp_vec = sel.sum(dim=-1).to(dtype=torch.float32)

            mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=device)
            adv_mb = adv[mb_idx_t].detach()  # do not backprop through advantage

            loss_pi = -(adv_mb * new_logp_vec).mean()

            approx_kl = torch.zeros((), device=device, dtype=torch.float32)
            if float(kl_coef) > 0.0 and ref_agent is not None:
                ref_ddp = ref_agent._agent._transfuser_model
                ref_core = ref_ddp.module if hasattr(ref_ddp, "module") else ref_ddp
                with torch.inference_mode():
                    ref_all_logps = ref_core.compute_log_probs_from_diffusion_chain(
                        features,
                        chain.to(device=device, dtype=replay_compute_chain_dtype),
                        eta=float(ddv2_eta),
                    )
                    ref_sel = ref_all_logps[torch.arange(bsz, device=device), mb_mode_idx, :]
                    ref_logp_vec = ref_sel.sum(dim=-1).to(dtype=torch.float32)

                # single-sample KL estimator: E_{a~pi}[logpi(a)-logpref(a)]
                approx_kl = (new_logp_vec - ref_logp_vec).mean().detach()
                loss_pi = loss_pi + float(kl_coef) * (new_logp_vec - ref_logp_vec).mean()

            loss = loss_pi / float(grad_accum_steps)

            sync_now = ((accum_i + 1) % grad_accum_steps) == 0
            cm = nullcontext()
            if ddp_enabled and hasattr(ddp_model, "no_sync") and not sync_now:
                cm = ddp_model.no_sync()
            with cm:
                loss.backward()

            accum_i += 1
            if sync_now:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in agent._ddv2_optimizer.param_groups[0]["params"] if p.grad is not None],
                    float(max_grad_norm),
                )
                agent._ddv2_optimizer.step()
                agent._ddv2_optimizer.zero_grad(set_to_none=True)

            last_loss_pi = float(loss_pi.detach().cpu().item())
            last_approx_kl = float(approx_kl.detach().cpu().item())
            last_adv_mean = float(adv_mb.detach().mean().cpu().item())

    if (accum_i % grad_accum_steps) != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in agent._ddv2_optimizer.param_groups[0]["params"] if p.grad is not None],
            float(max_grad_norm),
        )
        agent._ddv2_optimizer.step()
        agent._ddv2_optimizer.zero_grad(set_to_none=True)

    return ReinforcePPUpdateResult(
        loss_pi=float(last_loss_pi),
        approx_kl=float(last_approx_kl),
        adv_mean=float(last_adv_mean),
    )
