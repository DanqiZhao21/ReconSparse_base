from __future__ import annotations

import torch
import pytest
from PIL import Image

from framework.rewardmodel.config import ObservationRewardModelConfig
from framework.rewardmodel.models.observation_encoder import ObservationEncoder
from framework.rewardmodel.models.reward_model import ObservationTrajectoryRewardModel
from framework.rewardmodel.data.cached_dataset import CachedRewardModelDataset


def test_observation_reward_model_config_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError):
        ObservationRewardModelConfig.from_dict({"observation_channels": 18, "legacy_field": True})


def test_observation_encoder_requires_config_object() -> None:
    with pytest.raises(TypeError):
        ObservationEncoder(18, 32)


def test_observation_encoder_returns_query_compressed_tokens() -> None:
    cfg = ObservationRewardModelConfig(
        observation_channels=18,
        hidden_dim=32,
        num_observation_queries=6,
        num_attention_heads=4,
    )
    encoder = ObservationEncoder(cfg)

    tokens = encoder(torch.rand(2, 18, 64, 64))

    assert tokens.shape == (2, 6, 32)


def test_observation_trajectory_reward_model_forward_shapes() -> None:
    cfg = ObservationRewardModelConfig(
        observation_channels=18,
        ego_state_dim=7,
        hidden_dim=32,
        query_dim=16,
        num_horizons=4,
    )
    model = ObservationTrajectoryRewardModel(cfg)

    output = model(
        observations=torch.rand(2, 18, 64, 64),
        ego_states=torch.rand(2, 7),
        candidate_trajectories=torch.rand(2, 3, 8, 3),
    )

    assert output.metric_logits.shape == (2, 3, 4, 8)
    assert output.metric_scores.shape == (2, 3, 4, 8)
    assert output.horizon_score.shape == (2, 3, 4)
    assert output.final_score.shape == (2, 3)
    assert torch.all(output.metric_scores >= 0.0)
    assert torch.all(output.metric_scores <= 1.0)


def test_observation_trajectory_reward_model_backward() -> None:
    cfg = ObservationRewardModelConfig(
        observation_channels=18,
        ego_state_dim=7,
        hidden_dim=32,
        query_dim=16,
        num_horizons=4,
        num_observation_queries=6,
        num_attention_heads=4,
    )
    model = ObservationTrajectoryRewardModel(cfg)

    output = model(
        observations=torch.rand(2, 18, 64, 64),
        ego_states=torch.rand(2, 7),
        candidate_trajectories=torch.rand(2, 3, 8, 3),
    )
    loss = output.metric_logits.mean()
    loss.backward()

    assert any(param.grad is not None for param in model.parameters())


def test_cached_reward_model_dataset_loads_images_from_paths(tmp_path) -> None:
    img0 = tmp_path / "cam0.jpg"
    img1 = tmp_path / "cam1.jpg"
    Image.new("RGB", (8, 6), color=(255, 0, 0)).save(img0)
    Image.new("RGB", (8, 6), color=(0, 255, 0)).save(img1)

    sample_path = tmp_path / "sample.pt"
    torch.save(
        {
            "token": "tok",
            "image_paths": [str(img0), str(img1)],
            "ego_states": torch.zeros(3),
            "candidate_trajectories": torch.zeros(2, 4, 3),
            "targets": torch.zeros(2, 1, 8),
        },
        sample_path,
    )

    dataset = CachedRewardModelDataset(tmp_path, image_size=(4, 4))
    sample = dataset[0]

    assert sample["observations"].shape == (6, 4, 4)
    assert sample["image_paths"] == [str(img0), str(img1)]


def test_model_accepts_observation_history_channels() -> None:
    cfg = ObservationRewardModelConfig(
        observation_channels=36,
        ego_state_dim=4,
        hidden_dim=32,
        query_dim=16,
        num_horizons=8,
    )
    model = ObservationTrajectoryRewardModel(cfg)

    output = model(
        observations=torch.rand(1, 36, 32, 32),
        ego_states=torch.rand(1, 4),
        candidate_trajectories=torch.rand(1, 2, 8, 3),
    )

    assert output.metric_logits.shape == (1, 2, 8, 8)
