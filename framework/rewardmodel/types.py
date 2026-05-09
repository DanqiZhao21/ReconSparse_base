from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RewardAggregationOutput:
    metric_scores: torch.Tensor
    safe_score: torch.Tensor
    task_score: torch.Tensor
    horizon_score: torch.Tensor
    final_score: torch.Tensor


@dataclass
class ObservationRewardModelOutput:
    metric_logits: torch.Tensor
    metric_scores: torch.Tensor
    safe_score: torch.Tensor
    task_score: torch.Tensor
    horizon_score: torch.Tensor
    final_score: torch.Tensor


@dataclass
class TeacherTargetBatch:
    targets: torch.Tensor
    valid_mask: torch.Tensor

