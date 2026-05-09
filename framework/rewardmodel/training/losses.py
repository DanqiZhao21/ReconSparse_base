from __future__ import annotations

import torch
import torch.nn.functional as F

from framework.rewardmodel.config import RewardLossConfig


def reward_model_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None,
    config: RewardLossConfig,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    metric_weights = torch.as_tensor(
        config.metric_weights[: int(logits.shape[-1])],
        dtype=logits.dtype,
        device=logits.device,
    ).view(1, 1, 1, -1)
    horizon_weights = torch.as_tensor(
        config.horizon_weights[: int(logits.shape[-2])],
        dtype=logits.dtype,
        device=logits.device,
    ).view(1, 1, -1, 1)
    weighted = loss * metric_weights * horizon_weights
    if valid_mask is not None:
        weighted = weighted * valid_mask.to(dtype=weighted.dtype)
        denom = valid_mask.to(dtype=weighted.dtype).mul(metric_weights).mul(horizon_weights).sum().clamp(min=1.0)
        return weighted.sum() / denom
    return weighted.mean()

