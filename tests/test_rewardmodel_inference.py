from __future__ import annotations

import torch

from framework.rewardmodel.config import ObservationRewardModelConfig, RewardLossConfig
from framework.rewardmodel.inference.scorer import FrozenRewardModelScorer
from framework.rewardmodel.models.reward_model import ObservationTrajectoryRewardModel
from framework.rewardmodel.training.losses import reward_model_bce_loss


def test_reward_model_bce_loss_respects_valid_mask() -> None:
    logits = torch.zeros((1, 2, 3, 8), dtype=torch.float32)
    targets = torch.ones_like(logits)
    valid_mask = torch.zeros_like(logits, dtype=torch.bool)
    valid_mask[..., 0] = True

    loss = reward_model_bce_loss(
        logits,
        targets,
        valid_mask=valid_mask,
        config=RewardLossConfig(),
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_frozen_reward_model_scorer_loads_checkpoint_and_scores(tmp_path) -> None:
    cfg = ObservationRewardModelConfig(
        observation_channels=18,
        ego_state_dim=5,
        hidden_dim=32,
        query_dim=16,
        num_horizons=4,
    )
    model = ObservationTrajectoryRewardModel(cfg)
    checkpoint_path = tmp_path / "reward_model.pt"
    torch.save(
        {
            "model_config": cfg.to_dict(),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    scorer = FrozenRewardModelScorer.from_checkpoint(checkpoint_path, device="cpu")
    output = scorer.score(
        observations=torch.rand(1, 18, 64, 64),
        ego_states=torch.rand(1, 5),
        candidate_trajectories=torch.rand(1, 2, 8, 3),
    )

    assert output.final_score.shape == (1, 2)
    assert output.metric_scores.shape == (1, 2, 4, 8)
    assert not scorer.model.training
    assert all(not param.requires_grad for param in scorer.model.parameters())
