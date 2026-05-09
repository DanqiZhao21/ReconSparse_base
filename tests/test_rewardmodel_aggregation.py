from __future__ import annotations

import torch

from framework.rewardmodel.config import RewardAggregationConfig
from framework.rewardmodel.constants import (
    REWARD_METRIC_NAMES,
    SAFETY_REWARD_METRICS,
    TASK_REWARD_METRICS,
)
from framework.rewardmodel.supervision.aggregation import aggregate_reward_metrics


def test_reward_metric_schema_matches_dreamerad_dimensions() -> None:
    assert REWARD_METRIC_NAMES == (
        "rnc",
        "rdac",
        "rddc",
        "rtlc",
        "rep",
        "rttc",
        "rlk",
        "rhc",
    )
    assert SAFETY_REWARD_METRICS == ("rnc", "rdac", "rddc", "rtlc")
    assert TASK_REWARD_METRICS == ("rep", "rttc", "rlk", "rhc")


def test_safety_first_aggregation_penalizes_safety_failures() -> None:
    cfg = RewardAggregationConfig(epsilon=1.0e-4)
    safe_scores = torch.full((1, 1, 2, 8), 0.9, dtype=torch.float32)
    unsafe_scores = safe_scores.clone()
    unsafe_scores[..., 0] = 0.05

    safe = aggregate_reward_metrics(safe_scores, cfg)
    unsafe = aggregate_reward_metrics(unsafe_scores, cfg)

    assert safe.final_score.shape == (1, 1)
    assert safe.horizon_score.shape == (1, 1, 2)
    assert torch.all(safe.final_score > unsafe.final_score)
    assert torch.all(safe.safe_score > unsafe.safe_score)

