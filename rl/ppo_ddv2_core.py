from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DDV2PPOUpdateResult:
    loss_pi: float
    loss_v: float
    approx_kl: float
    ratio_mean: float
    adv_mean: float


def compute_gae(
	*,
	rewards: torch.Tensor,
	dones: torch.Tensor,
	values: torch.Tensor,
	last_value: torch.Tensor,
	gamma: float,
	gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""GAE(λ) with bootstrap for truncated rollouts.

	Matches the exact update math used in train_closed_loop.py, except that we allow
	providing a bootstrap value for the final transition.

	Args:
		rewards: (T,)
		dones: (T,) float/bool, 1.0 when episode ended after this step.
		values: (T,)
		last_value: scalar tensor, V(s_T) used to bootstrap the final transition.
			If the final transition ended an episode, caller should pass 0.

	Returns:
		adv: (T,)
		ret: (T,)
	"""
	if rewards.ndim != 1 or dones.ndim != 1 or values.ndim != 1:
		raise ValueError("compute_gae expects 1D tensors (T,)")
	if rewards.shape[0] != dones.shape[0] or rewards.shape[0] != values.shape[0]:
		raise ValueError("compute_gae expects matching lengths")

	T = int(rewards.shape[0])
	adv = torch.zeros_like(rewards)
	last_gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)

	for t in reversed(range(T)):
		mask = 1.0 - dones[t]
		v_next = last_value if t == (T - 1) else values[t + 1]
		delta = rewards[t] + float(gamma) * v_next * mask - values[t]
		last_gae = delta + float(gamma) * float(gae_lambda) * mask * last_gae
		adv[t] = last_gae

	ret = adv + values
	return adv, ret


def normalize_advantages(
    adv: torch.Tensor,
    *,
    ddp_enabled: bool,
    dist_module: Any,
    device: torch.device,
) -> torch.Tensor:
    """Normalize advantages globally across DDP ranks (no all_gather needed)."""
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
        std = torch.sqrt(torch.clamp(var, min=0.0) + 1e-8)
        return (adv - mean) / std

    return (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)


def ddv2_ppo_update(
    *,
    agent: Any,
    value_net: torch.nn.Module,
    value_optim: torch.optim.Optimizer,
    obs_batch: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    ret: torch.Tensor,
    replay: List[Dict[str, Any]],
    device: torch.device,
    ddv2_eta: float,
    ddv2_mode_idx_default: int,
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
    replay_compute_camera_dtype: torch.dtype = torch.float32,
    replay_compute_chain_dtype: torch.dtype = torch.float32,
    use_distributed_sampler: bool = True,
) -> DDV2PPOUpdateResult:
    """PPO update for DDV2 policy using diffusion replay.

    Extracted from train_closed_loop.py so actor-learner and closed-loop share identical math.
    """
    if agent is None or getattr(agent, "_ddv2_optimizer", None) is None:
        raise RuntimeError("ddv2_ppo_update requires agent._ddv2_optimizer")

    n = int(obs_batch.shape[0])
    if n == 0:
        return DDV2PPOUpdateResult(0.0, 0.0, 0.0, 0.0, 0.0)
    if len(replay) != n:
        raise RuntimeError(f"Replay length mismatch: len(replay)={len(replay)} n={n}")

    ddp_model = agent._agent._transfuser_model
    value_ddp = value_net

    grad_accum_steps = max(1, int(grad_accum_steps))

    agent._ddv2_optimizer.zero_grad(set_to_none=True)
    value_optim.zero_grad(set_to_none=True)

    idxs = np.arange(n)

    last_loss_pi = 0.0
    last_loss_v = 0.0
    last_approx_kl = 0.0
    last_ratio_mean = 0.0
    last_adv_mean = 0.0

    ds = list(range(n))
    sampler: Optional[DistributedSampler] = None
    if ddp_enabled and use_distributed_sampler:
        sampler = DistributedSampler(ds, num_replicas=int(world_size), rank=int(rank), shuffle=True, drop_last=False)

    accum_i = 0
    for ep in range(int(ppo_epochs)):
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

            core = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
            all_logps = core.compute_log_probs_from_diffusion_chain(
                features,
                chain.to(device=device, dtype=replay_compute_chain_dtype),
                eta=float(ddv2_eta),
            )
            bsz = int(cam.shape[0])
            sel = all_logps[torch.arange(bsz, device=device), mb_mode_idx, :]
            new_logp_vec = sel.sum(dim=-1).to(dtype=torch.float32)

            mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=device)
            old_logp_mb = old_logp[mb_idx_t]
            adv_mb = adv[mb_idx_t]
            ret_mb = ret[mb_idx_t]

            v_pred = value_ddp(obs_batch[mb_idx_t])
            loss_v = F.mse_loss(v_pred, ret_mb)

            ratio = torch.exp(new_logp_vec - old_logp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv_mb
            loss_pi = -(torch.min(surr1, surr2)).mean()

            loss = loss_pi + float(vf_coef) * loss_v
            approx_kl = (old_logp_mb - new_logp_vec).mean().detach()

            loss = loss / float(grad_accum_steps)

            sync_now = ((accum_i + 1) % grad_accum_steps) == 0
            cm1 = nullcontext()
            cm2 = nullcontext()
            if ddp_enabled and hasattr(ddp_model, "no_sync") and not sync_now:
                cm1 = ddp_model.no_sync()
            if ddp_enabled and hasattr(value_ddp, "no_sync") and not sync_now:
                cm2 = value_ddp.no_sync()
            with cm1, cm2:
                loss.backward()

            accum_i += 1
            if sync_now:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in agent._ddv2_optimizer.param_groups[0]["params"] if p.grad is not None],
                    float(max_grad_norm),
                )
                agent._ddv2_optimizer.step()
                value_optim.step()
                agent._ddv2_optimizer.zero_grad(set_to_none=True)
                value_optim.zero_grad(set_to_none=True)

            last_loss_pi = float(loss_pi.detach().cpu().item())
            last_loss_v = float(loss_v.detach().cpu().item())
            last_approx_kl = float(approx_kl.detach().cpu().item())
            last_ratio_mean = float(ratio.detach().mean().cpu().item())
            last_adv_mean = float(adv_mb.detach().mean().cpu().item())

    if (accum_i % grad_accum_steps) != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in agent._ddv2_optimizer.param_groups[0]["params"] if p.grad is not None],
            float(max_grad_norm),
        )
        agent._ddv2_optimizer.step()
        value_optim.step()
        agent._ddv2_optimizer.zero_grad(set_to_none=True)
        value_optim.zero_grad(set_to_none=True)

    return DDV2PPOUpdateResult(
        loss_pi=float(last_loss_pi),
        loss_v=float(last_loss_v),
        approx_kl=float(last_approx_kl),
        ratio_mean=float(last_ratio_mean),
        adv_mean=float(last_adv_mean),
    )
