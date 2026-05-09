from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from framework.rewardmodel.config import ObservationRewardModelConfig
from framework.rewardmodel.models.reward_model import ObservationTrajectoryRewardModel
from framework.rewardmodel.types import ObservationRewardModelOutput


@dataclass
class FrozenRewardModelScorer:
    model: ObservationTrajectoryRewardModel
    device: torch.device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "FrozenRewardModelScorer":
        ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
        config = ObservationRewardModelConfig.from_dict(dict(ckpt["model_config"]))
        model = ObservationTrajectoryRewardModel(config)
        model.load_state_dict(dict(ckpt["state_dict"]))
        dev = torch.device(device)
        model.to(dev)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return cls(model=model, device=dev)

    @torch.no_grad()
    def score(
        self,
        *,
        observations: torch.Tensor,
        ego_states: torch.Tensor,
        candidate_trajectories: torch.Tensor,
    ) -> ObservationRewardModelOutput:
        return self.model(
            observations=observations.to(self.device),
            ego_states=ego_states.to(self.device),
            candidate_trajectories=candidate_trajectories.to(self.device),
        )
