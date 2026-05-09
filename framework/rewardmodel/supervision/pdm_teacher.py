from __future__ import annotations

from typing import Any

import torch

from framework.rewardmodel.config import RewardSupervisionConfig
from framework.rewardmodel.constants import REWARD_METRIC_TO_INDEX
from framework.rewardmodel.types import TeacherTargetBatch


def _clamp_scores(tensor: torch.Tensor, cfg: RewardSupervisionConfig) -> torch.Tensor:
    return tensor.to(dtype=torch.float32).clamp(min=cfg.clamp_min, max=cfg.clamp_max)


def normalize_teacher_scores(
    scores: Any,
    cfg: RewardSupervisionConfig,
) -> TeacherTargetBatch:
    if isinstance(scores, dict):
        targets = torch.zeros((1, 1, cfg.num_horizons, cfg.num_metrics), dtype=torch.float32)
        valid_mask = torch.zeros_like(targets, dtype=torch.bool)
        inferred_shape: tuple[int, int] | None = None
        for name, value in scores.items():
            idx = REWARD_METRIC_TO_INDEX.get(str(name))
            if idx is None:
                continue
            tensor = _clamp_scores(torch.as_tensor(value), cfg)
            if tensor.ndim != 3:
                raise ValueError(f"Metric teacher value for {name} must have shape [B,G,H], got {tuple(tensor.shape)}")
            if inferred_shape is None:
                inferred_shape = (int(tensor.shape[0]), int(tensor.shape[1]))
                targets = torch.zeros((tensor.shape[0], tensor.shape[1], cfg.num_horizons, cfg.num_metrics), dtype=torch.float32)
                valid_mask = torch.zeros_like(targets, dtype=torch.bool)
            targets[..., idx] = tensor[..., : cfg.num_horizons]
            valid_mask[..., idx] = True
        return TeacherTargetBatch(targets=targets, valid_mask=valid_mask)

    tensor = _clamp_scores(torch.as_tensor(scores), cfg)
    if tensor.ndim == 2:
        tensor = tensor[:, :, None, None].expand(tensor.shape[0], tensor.shape[1], cfg.num_horizons, cfg.num_metrics)
    elif tensor.ndim == 3:
        tensor = tensor[:, :, :, None].expand(tensor.shape[0], tensor.shape[1], tensor.shape[2], cfg.num_metrics)
        if int(tensor.shape[2]) < cfg.num_horizons:
            raise ValueError("Teacher horizon dimension shorter than config.num_horizons")
        tensor = tensor[:, :, : cfg.num_horizons, :]
    elif tensor.ndim == 4:
        tensor = tensor[:, :, : cfg.num_horizons, : cfg.num_metrics]
    else:
        raise ValueError(f"Unsupported teacher score shape: {tuple(tensor.shape)}")

    valid_mask = torch.ones_like(tensor, dtype=torch.bool)
    return TeacherTargetBatch(targets=tensor, valid_mask=valid_mask)

