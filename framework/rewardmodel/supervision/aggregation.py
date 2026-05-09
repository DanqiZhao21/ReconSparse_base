from __future__ import annotations

import torch

from framework.rewardmodel.config import RewardAggregationConfig
from framework.rewardmodel.constants import SAFETY_METRIC_INDICES, TASK_METRIC_INDICES
from framework.rewardmodel.types import RewardAggregationOutput


def _resolve_metric_weights(metric_scores: torch.Tensor, cfg: RewardAggregationConfig) -> torch.Tensor:
    weights = torch.as_tensor(
        cfg.metric_weights[: int(metric_scores.shape[-1])],
        dtype=metric_scores.dtype,
        device=metric_scores.device,
    )
    if int(weights.numel()) < int(metric_scores.shape[-1]):
        pad = torch.ones(
            int(metric_scores.shape[-1]) - int(weights.numel()),
            dtype=metric_scores.dtype,
            device=metric_scores.device,
        )
        weights = torch.cat([weights, pad], dim=0)
    return weights


def _resolve_horizon_weights(metric_scores: torch.Tensor, cfg: RewardAggregationConfig) -> torch.Tensor:
    weights = torch.as_tensor(
        cfg.horizon_weights[: int(metric_scores.shape[-2])],
        dtype=metric_scores.dtype,
        device=metric_scores.device,
    )
    if int(weights.numel()) < int(metric_scores.shape[-2]):
        pad = torch.ones(
            int(metric_scores.shape[-2]) - int(weights.numel()),
            dtype=metric_scores.dtype,
            device=metric_scores.device,
        )
        weights = torch.cat([weights, pad], dim=0)
    return weights


def aggregate_reward_metrics(
    metric_scores: torch.Tensor,
    cfg: RewardAggregationConfig | None = None,
) -> RewardAggregationOutput:
    if cfg is None:
        cfg = RewardAggregationConfig()
    scores = metric_scores.clamp(min=cfg.epsilon, max=1.0)
    metric_weights = _resolve_metric_weights(scores, cfg)
    safety_weights = metric_weights[list(SAFETY_METRIC_INDICES)]
    task_weights = metric_weights[list(TASK_METRIC_INDICES)]

    safe_metrics = scores[..., list(SAFETY_METRIC_INDICES)]
    task_metrics = scores[..., list(TASK_METRIC_INDICES)]

    safe_score = torch.sum(torch.log(safe_metrics) * safety_weights, dim=-1)
    task_score = torch.log(torch.sum(task_metrics * task_weights, dim=-1).clamp(min=cfg.epsilon))
    horizon_score = safe_score + task_score

    horizon_weights = _resolve_horizon_weights(scores, cfg)
    final_score = torch.sum(horizon_score * horizon_weights, dim=-1)
    return RewardAggregationOutput(
        metric_scores=scores,
        safe_score=safe_score,
        task_score=task_score,
        horizon_score=horizon_score,
        final_score=final_score,
    )

