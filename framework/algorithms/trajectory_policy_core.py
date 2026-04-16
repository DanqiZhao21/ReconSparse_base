from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TrajectoryPPOObjective:#Python dataclass = 自动生成构造函数
    loss: torch.Tensor
    loss_pi: torch.Tensor
    loss_v: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    value_clip_frac: torch.Tensor
    ratio_mean: torch.Tensor
    adv_mean: torch.Tensor


@dataclass
class TrajectoryReinforceObjective:
    loss: torch.Tensor
    loss_pi: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    ratio_mean: torch.Tensor
    adv_mean: torch.Tensor


@dataclass
class TrajectoryGRPOObjective:
    loss: torch.Tensor
    advantages: torch.Tensor
    score_mean: torch.Tensor
    score_std: torch.Tensor
    score_min: torch.Tensor
    score_max: torch.Tensor


def _as_logp_tensor(logp_out: Any, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(logp_out):
        return logp_out.to(device=device, dtype=torch.float32).view(-1)
    if isinstance(logp_out, (list, tuple)):
        vals: List[torch.Tensor] = []
        for item in logp_out:
            if not torch.is_tensor(item):
                raise TypeError("logp_from_replay_batch must return tensors or a tensor")
            vals.append(item.to(device=device, dtype=torch.float32).view(()))
        if len(vals) == 0:
            return torch.empty((0,), device=device, dtype=torch.float32)
        return torch.stack(vals, dim=0)
    raise TypeError(f"Unsupported logp output type: {type(logp_out)!r}")


def agent_logp_from_replay_batch(
    agent: Any,
    replays: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    eta: float = 1.0,
) -> torch.Tensor:
    if len(replays) == 0:
        return torch.empty((0,), device=device, dtype=torch.float32)

    batch_fn = getattr(agent, "logp_from_replay_batch", None)
    if callable(batch_fn):
        return _as_logp_tensor(batch_fn(list(replays), eta=float(eta)), device=device)

    vals = [agent.logp_from_replay(rep, eta=float(eta)) for rep in replays]
    return _as_logp_tensor(vals, device=device)


'''
输入: logp / adv / value / return
输出: loss + metrics
'''

def compute_ppo_objective(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    ret: torch.Tensor,
    value_pred: torch.Tensor,
    old_value: Optional[torch.Tensor],
    clip_eps: float,
    vf_coef: float,
    value_clip_eps: float = 0.0,
    kl_coef: float = 0.0,
    dual_clip: float | None = None,
) -> TrajectoryPPOObjective:
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv
    policy_term = torch.min(surr1, surr2)
    
    clip_frac = ((ratio - 1.0).abs() > float(clip_eps)).to(dtype=torch.float32).mean()
    if dual_clip is not None:
        clipped = torch.max(policy_term, torch.as_tensor(float(dual_clip), device=adv.device, dtype=adv.dtype) * adv)
        policy_term = torch.where(adv < 0.0, clipped, policy_term)
    loss_pi = -policy_term.mean()

    if float(value_clip_eps) > 0.0 and old_value is not None and int(old_value.numel()) == int(ret.numel()):
        value_delta = value_pred - old_value
        value_pred_clipped = old_value + value_delta.clamp(-float(value_clip_eps), float(value_clip_eps))
        loss_v_unclipped = F.mse_loss(value_pred, ret, reduction="none")
        loss_v_clipped = F.mse_loss(value_pred_clipped, ret, reduction="none")
        loss_v = torch.max(loss_v_unclipped, loss_v_clipped).mean()
        value_clip_frac = (value_delta.abs() > float(value_clip_eps)).to(dtype=torch.float32).mean()
    else:
        loss_v = F.mse_loss(value_pred, ret)
        value_clip_frac = torch.zeros((), device=ret.device, dtype=torch.float32)

    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    loss = loss_pi + float(vf_coef) * loss_v + float(kl_coef) * approx_kl
    return TrajectoryPPOObjective(
        loss=loss,
        loss_pi=loss_pi,
        loss_v=loss_v,
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        value_clip_frac=value_clip_frac,
        ratio_mean=ratio.mean(),
        adv_mean=adv.mean(),
    )

#adv是在 framework/batch/actor_learner.py (line 101) 的 build_training_batch(...) 里生成的。

def compute_reinforce_objective(
    *,
    new_logp: torch.Tensor,
    old_logp: Optional[torch.Tensor],
    adv: torch.Tensor,
    clip_eps: float,
    kl_coef: float = 0.0,
) -> TrajectoryReinforceObjective:
    if old_logp is not None and int(old_logp.numel()) == int(adv.numel()):
        log_ratio = new_logp - old_logp
        ratio = torch.exp(log_ratio)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv
        loss_pi = -torch.min(surr1, surr2).mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_frac = ((ratio - 1.0).abs() > float(clip_eps)).to(dtype=torch.float32).mean()
        ratio_mean = ratio.mean()
    else:
        loss_pi = -(adv * new_logp).mean()
        approx_kl = torch.zeros((), device=adv.device, dtype=torch.float32)
        clip_frac = torch.zeros((), device=adv.device, dtype=torch.float32)
        ratio_mean = torch.ones((), device=adv.device, dtype=torch.float32)

    loss = loss_pi + float(kl_coef) * approx_kl
    return TrajectoryReinforceObjective(
        loss=loss,
        loss_pi=loss_pi,
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        ratio_mean=ratio_mean,
        adv_mean=adv.mean(),
    )


def compute_grpo_objective(
    *,
    candidate_log_probs: torch.Tensor,
    candidate_scores: torch.Tensor,
    score_norm_eps: float = 1e-6,
    use_rank_adv: bool = False,
    score_clip: float | None = None,
) -> TrajectoryGRPOObjective:
    if candidate_log_probs.ndim != 2 or candidate_scores.ndim != 2:
        raise ValueError("candidate_log_probs and candidate_scores must both be 2D tensors (batch, candidates)")
    if tuple(candidate_log_probs.shape) != tuple(candidate_scores.shape):
        raise ValueError(
            "candidate_log_probs and candidate_scores must have identical shapes; "
            f"got log_probs={tuple(candidate_log_probs.shape)} scores={tuple(candidate_scores.shape)}"
        )

    scores = candidate_scores.to(device=candidate_log_probs.device, dtype=torch.float32)
    if score_clip is not None:
        scores = scores.clamp(min=-float(score_clip), max=float(score_clip))

    if use_rank_adv:
        order = torch.argsort(scores, dim=1, descending=False)
        ranks = torch.argsort(order, dim=1, descending=False).to(dtype=torch.float32)
        denom = torch.clamp(torch.as_tensor(scores.shape[1] - 1, device=scores.device, dtype=torch.float32), min=1.0)
        advantages = (ranks / denom) - 0.5
    else:
        score_mean = scores.mean(dim=1, keepdim=True)
        score_std = scores.std(dim=1, keepdim=True, unbiased=False)
        advantages = (scores - score_mean) / (score_std + float(score_norm_eps))

    loss = -(advantages.detach() * candidate_log_probs).mean()
    score_mean = scores.mean(dim=1)
    score_std = scores.std(dim=1, unbiased=False)
    score_min = scores.min(dim=1).values
    score_max = scores.max(dim=1).values
    return TrajectoryGRPOObjective(
        loss=loss,
        advantages=advantages,
        score_mean=score_mean.mean(),
        score_std=score_std.mean(),
        score_min=score_min.mean(),
        score_max=score_max.mean(),
    )


def compute_ppo_metrics(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    ret: torch.Tensor,
    value_pred: torch.Tensor,
    loss: TrajectoryPPOObjective,
) -> Dict[str, torch.Tensor]:
    var_ret = torch.var(ret, unbiased=False)
    explained_variance = 1.0 - (torch.var(ret - value_pred, unbiased=False) / (var_ret + 1e-8))
    return {
        "loss_pi": loss.loss_pi.detach(),
        "loss_v": loss.loss_v.detach(),
        "approx_kl": loss.approx_kl.detach(),
        "clip_frac": loss.clip_frac.detach(),
        "value_clip_frac": loss.value_clip_frac.detach(),
        "ratio_mean": loss.ratio_mean.detach(),
        "adv_mean": loss.adv_mean.detach(),
        "explained_variance": explained_variance.detach(),
    }


def compute_reinforce_metrics(
    *,
    new_logp: torch.Tensor,
    old_logp: Optional[torch.Tensor],
    adv: torch.Tensor,
    loss: TrajectoryReinforceObjective,
) -> Dict[str, torch.Tensor]:
    del new_logp, old_logp
    return {
        "loss_pi": loss.loss_pi.detach(),
        "approx_kl": loss.approx_kl.detach(),
        "clip_frac": loss.clip_frac.detach(),
        "ratio_mean": loss.ratio_mean.detach(),
        "adv_mean": loss.adv_mean.detach(),
    }


__all__ = [
    "TrajectoryPPOObjective",
    "TrajectoryReinforceObjective",
    "TrajectoryGRPOObjective",
    "agent_logp_from_replay_batch",
    "compute_grpo_objective",
    "compute_ppo_objective",
    "compute_reinforce_objective",
    "compute_ppo_metrics",
    "compute_reinforce_metrics",
]
