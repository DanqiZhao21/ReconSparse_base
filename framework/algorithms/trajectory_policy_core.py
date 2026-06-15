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
class TrajectorySACObjective:
    loss: torch.Tensor
    loss_pi: torch.Tensor
    loss_pg: torch.Tensor
    loss_entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    ratio_mean: torch.Tensor
    adv_mean: torch.Tensor
    logp_mean: torch.Tensor
    entropy_coef: torch.Tensor


@dataclass
class TrajectoryGRPOObjective:
    loss: torch.Tensor
    advantages: torch.Tensor
    score_mean: torch.Tensor
    score_std: torch.Tensor
    score_min: torch.Tensor
    score_max: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    ratio_mean: torch.Tensor


@dataclass
class TrajectoryRiskDecelAuxiliaryObjective:
    loss: torch.Tensor
    active_count: torch.Tensor
    decel_prob_mean: torch.Tensor
    accel_prob_mean: torch.Tensor


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


def _as_score_tensor(scores: Any, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(scores):
        return scores.to(device=device, dtype=torch.float32)
    if isinstance(scores, np.ndarray):
        return torch.from_numpy(scores).to(device=device, dtype=torch.float32)
    if isinstance(scores, (list, tuple)):
        return torch.as_tensor(scores, device=device, dtype=torch.float32)
    raise TypeError(f"Unsupported counterfactual score type: {type(scores)!r}")


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


def score_counterfactual_trajectories(
    agent: Any,
    replays: Sequence[dict[str, Any]],
    traj_xyyaw: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    scorer_fn = getattr(agent, "pdm_score_counterfactuals_from_replay_batch", None)
    if not callable(scorer_fn):
        scorer_fn = getattr(agent, "score_counterfactuals_from_replay_batch", None)
    if not callable(scorer_fn):
        raise RuntimeError(
            "GRPO is enabled but the agent does not expose a counterfactual scorer hook. "
            "Expected `pdm_score_counterfactuals_from_replay_batch(...)`."
        )

    scores = scorer_fn(replays, traj_xyyaw)
    return _as_score_tensor(scores, device=device)


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


def compute_sac_objective(
    *,
    new_logp: torch.Tensor,
    old_logp: Optional[torch.Tensor],
    adv: torch.Tensor,
    entropy_coef: float,
    kl_coef: float = 0.0,
    clip_eps: float = 0.2,
) -> TrajectorySACObjective:
    if old_logp is not None and int(old_logp.numel()) == int(adv.numel()):
        log_ratio = new_logp - old_logp
        ratio = torch.exp(log_ratio)
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_frac = ((ratio - 1.0).abs() > float(clip_eps)).to(dtype=torch.float32).mean()
        ratio_mean = ratio.mean()
    else:
        approx_kl = torch.zeros((), device=adv.device, dtype=torch.float32)
        clip_frac = torch.zeros((), device=adv.device, dtype=torch.float32)
        ratio_mean = torch.ones((), device=adv.device, dtype=torch.float32)

    loss_pg = -(adv.detach() * new_logp).mean()
    loss_entropy = torch.as_tensor(float(entropy_coef), device=adv.device, dtype=torch.float32) * new_logp.mean()
    loss = loss_pg + loss_entropy + float(kl_coef) * approx_kl
    return TrajectorySACObjective(
        loss=loss,
        loss_pi=loss,
        loss_pg=loss_pg,
        loss_entropy=loss_entropy,
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        ratio_mean=ratio_mean,
        adv_mean=adv.mean(),
        logp_mean=new_logp.mean(),
        entropy_coef=torch.as_tensor(float(entropy_coef), device=adv.device, dtype=torch.float32),
    )


def compute_grpo_objective(
    *,
    candidate_log_probs: torch.Tensor,
    old_candidate_log_probs: torch.Tensor | None = None,
    candidate_scores: torch.Tensor,
    candidate_score_logits: torch.Tensor | None = None,
    score_norm_eps: float = 1e-6,
    use_rank_adv: bool = False,
    score_clip: float | None = None,
    objective: str = "logprob",
    temperature: float = 1.0,
    clip_eps: float = 0.2,
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

    objective_key = str(objective).strip().lower()
    if objective_key in {"logprob", "reinforce", "grpo"}:
        loss = -(advantages.detach() * candidate_log_probs).mean()
        approx_kl = torch.zeros((), device=candidate_log_probs.device, dtype=torch.float32)
        clip_frac = torch.zeros((), device=candidate_log_probs.device, dtype=torch.float32)
        ratio_mean = torch.ones((), device=candidate_log_probs.device, dtype=torch.float32)
    elif objective_key == "expected_prob":
        logits = candidate_score_logits
        if logits is None:
            logits = candidate_log_probs
        logits = logits.to(device=candidate_log_probs.device, dtype=torch.float32)
        if tuple(logits.shape) != tuple(candidate_log_probs.shape):
            raise ValueError(
                "candidate_score_logits must match candidate_log_probs shape for expected_prob objective; "
                f"got logits={tuple(logits.shape)} log_probs={tuple(candidate_log_probs.shape)}"
            )
        temp = max(1.0e-6, float(temperature))
        probs = torch.softmax(logits / temp, dim=1)
        loss = -(probs * advantages.detach()).sum(dim=1).mean()
        approx_kl = torch.zeros((), device=candidate_log_probs.device, dtype=torch.float32)
        clip_frac = torch.zeros((), device=candidate_log_probs.device, dtype=torch.float32)
        ratio_mean = torch.ones((), device=candidate_log_probs.device, dtype=torch.float32)
    elif objective_key in {"clipped_ratio", "ppo_ratio", "strict_grpo"}:
        if old_candidate_log_probs is None:
            raise ValueError("objective='clipped_ratio' requires old_candidate_log_probs")
        old_log_probs = old_candidate_log_probs.to(device=candidate_log_probs.device, dtype=torch.float32)
        if tuple(old_log_probs.shape) != tuple(candidate_log_probs.shape):
            raise ValueError(
                "old_candidate_log_probs must match candidate_log_probs shape for clipped_ratio objective; "
                f"got old={tuple(old_log_probs.shape)} new={tuple(candidate_log_probs.shape)}"
            )
        log_ratio = candidate_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        adv_detached = advantages.detach()
        unclipped = ratio * adv_detached
        clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
        clipped = clipped_ratio * adv_detached
        loss = -torch.min(unclipped, clipped).mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_frac = ((ratio - 1.0).abs() > float(clip_eps)).to(dtype=torch.float32).mean()
        ratio_mean = ratio.mean()
    else:
        raise ValueError(
            f"Unsupported GRPO objective={objective!r}; expected 'logprob', 'expected_prob', or 'clipped_ratio'"
        )
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
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        ratio_mean=ratio_mean,
    )


def compute_risk_decel_auxiliary_objective(
    *,
    candidate_score_logits: torch.Tensor,
    candidate_traj_xyyaw: torch.Tensor,
    high_risk_mask: torch.Tensor,
    ego_speed_mps: torch.Tensor | None = None,
    dt_s: float = 0.5,
    speed_margin_mps: float = 0.1,
    eps: float = 1.0e-6,
) -> TrajectoryRiskDecelAuxiliaryObjective:
    if candidate_score_logits.ndim != 2:
        raise ValueError(
            "candidate_score_logits must be a 2D tensor (batch, candidates); "
            f"got {tuple(candidate_score_logits.shape)}"
        )
    if candidate_traj_xyyaw.ndim != 4 or int(candidate_traj_xyyaw.shape[-1]) < 2:
        raise ValueError(
            "candidate_traj_xyyaw must have shape (batch, candidates, horizon, dims>=2); "
            f"got {tuple(candidate_traj_xyyaw.shape)}"
        )
    if tuple(candidate_score_logits.shape[:2]) != tuple(candidate_traj_xyyaw.shape[:2]):
        raise ValueError(
            "candidate_score_logits and candidate_traj_xyyaw must agree on batch/candidate dimensions; "
            f"got logits={tuple(candidate_score_logits.shape)} traj={tuple(candidate_traj_xyyaw.shape)}"
        )

    device = candidate_score_logits.device
    dtype = torch.float32
    logits = candidate_score_logits.to(device=device, dtype=dtype)
    traj = candidate_traj_xyyaw.to(device=device, dtype=dtype)
    high_risk = high_risk_mask.to(device=device, dtype=torch.bool).view(-1)
    if int(high_risk.numel()) != int(logits.shape[0]):
        raise ValueError(
            "high_risk_mask must match batch size; "
            f"got mask={int(high_risk.numel())} batch={int(logits.shape[0])}"
        )

    if int(logits.numel()) == 0 or int(traj.shape[2]) <= 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return TrajectoryRiskDecelAuxiliaryObjective(
            loss=zero,
            active_count=zero,
            decel_prob_mean=zero,
            accel_prob_mean=zero,
        )

    step0_xy = traj[:, :, 0, :2]
    first_window_speed = torch.linalg.norm(step0_xy, dim=-1) / max(1.0e-6, float(dt_s))
    if ego_speed_mps is None:
        if int(traj.shape[2]) > 1:
            step1_speed = torch.linalg.norm(traj[:, :, 1, :2] - step0_xy, dim=-1) / max(1.0e-6, float(dt_s))
            reference_speed = step1_speed
        else:
            reference_speed = first_window_speed.mean(dim=1, keepdim=True).expand_as(first_window_speed)
    else:
        ego_speed = ego_speed_mps.to(device=device, dtype=dtype).view(-1)
        if int(ego_speed.numel()) != int(logits.shape[0]):
            raise ValueError(
                "ego_speed_mps must match batch size; "
                f"got speed={int(ego_speed.numel())} batch={int(logits.shape[0])}"
            )
        reference_speed = ego_speed[:, None].expand_as(first_window_speed)

    margin = max(0.0, float(speed_margin_mps))
    decel_mask = first_window_speed <= (reference_speed - margin)
    accel_mask = first_window_speed >= (reference_speed + margin)
    active = high_risk & decel_mask.any(dim=1)

    probs = torch.softmax(logits, dim=1)
    decel_prob = (probs * decel_mask.to(dtype=dtype)).sum(dim=1)
    accel_prob = (probs * accel_mask.to(dtype=dtype)).sum(dim=1)

    if not bool(active.any()):
        zero = logits.sum() * 0.0
        return TrajectoryRiskDecelAuxiliaryObjective(
            loss=zero,
            active_count=torch.zeros((), device=device, dtype=dtype),
            decel_prob_mean=zero.detach(),
            accel_prob_mean=zero.detach(),
        )

    active_decel_prob = decel_prob[active].clamp(min=float(eps), max=1.0)
    active_accel_prob = accel_prob[active].clamp(min=0.0, max=1.0 - float(eps))
    loss_terms = -torch.log(active_decel_prob) - torch.log1p(-active_accel_prob)
    loss = loss_terms.mean()
    return TrajectoryRiskDecelAuxiliaryObjective(
        loss=loss,
        active_count=active.to(dtype=dtype).sum(),
        decel_prob_mean=active_decel_prob.detach().mean(),
        accel_prob_mean=active_accel_prob.detach().mean(),
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
    "TrajectorySACObjective",
    "TrajectoryGRPOObjective",
    "TrajectoryRiskDecelAuxiliaryObjective",
    "agent_logp_from_replay_batch",
    "compute_grpo_objective",
    "compute_risk_decel_auxiliary_objective",
    "compute_ppo_objective",
    "compute_reinforce_objective",
    "compute_sac_objective",
    "compute_ppo_metrics",
    "compute_reinforce_metrics",
]
