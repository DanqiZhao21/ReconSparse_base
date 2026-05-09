from __future__ import annotations

import torch
from torch import nn

from framework.rewardmodel.config import ObservationRewardModelConfig, RewardAggregationConfig
from framework.rewardmodel.models.observation_encoder import ObservationEncoder
from framework.rewardmodel.supervision.aggregation import aggregate_reward_metrics
from framework.rewardmodel.types import ObservationRewardModelOutput


class ObservationTrajectoryRewardModel(nn.Module):
    def __init__(self, config: ObservationRewardModelConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_encoder = ObservationEncoder(config)
        self.ego_encoder = nn.Sequential(
            nn.Linear(config.ego_state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.query_dim),
        )
        self.trajectory_encoder = nn.GRU(
            input_size=3,
            hidden_size=config.trajectory_hidden_dim,
            batch_first=True,
        )
        self.trajectory_projection = nn.Linear(config.trajectory_hidden_dim, config.query_dim)
        self.step_embedding = nn.Embedding(config.num_horizons, config.query_dim)
        self.metric_queries = nn.Parameter(torch.randn(config.num_metrics, config.query_dim) * 0.02)
        self.reward_attention = nn.MultiheadAttention(
            embed_dim=config.query_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.observation_key_projection = nn.Linear(config.hidden_dim, config.query_dim)
        self.query_norm = nn.LayerNorm(config.query_dim)
        self.attention_norm = nn.LayerNorm(config.query_dim)
        self.metric_head = nn.Sequential(
            nn.Linear(config.query_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        *,
        observations: torch.Tensor,
        ego_states: torch.Tensor,
        candidate_trajectories: torch.Tensor,
    ) -> ObservationRewardModelOutput:
        batch_size, num_candidates = int(candidate_trajectories.shape[0]), int(candidate_trajectories.shape[1])
        obs_tokens = self.observation_encoder(observations)
        obs_tokens = self.observation_key_projection(obs_tokens)
        ego_context = self.ego_encoder(ego_states)

        traj_flat = candidate_trajectories.reshape(batch_size * num_candidates, int(candidate_trajectories.shape[2]), 3)
        traj_outputs, _ = self.trajectory_encoder(traj_flat)
        horizon_indices = torch.linspace(
            0,
            int(candidate_trajectories.shape[2]) - 1,
            steps=self.config.num_horizons,
            device=candidate_trajectories.device,
        ).round().to(dtype=torch.long)
        traj_context = traj_outputs.index_select(1, horizon_indices)
        traj_context = traj_context.reshape(
            batch_size,
            num_candidates,
            self.config.num_horizons,
            self.config.trajectory_hidden_dim,
        )
        traj_context = self.trajectory_projection(traj_context)

        ego_context = ego_context[:, None, None, :].expand(
            batch_size,
            num_candidates,
            self.config.num_horizons,
            ego_context.shape[-1],
        )

        step_ids = torch.arange(self.config.num_horizons, device=observations.device)
        step_embed = self.step_embedding(step_ids)[None, None, :, :]
        metric_queries = self.metric_queries[None, None, None, :, :]
        reward_queries = traj_context[:, :, :, None, :] + ego_context[:, :, :, None, :] + step_embed[:, :, :, None, :] + metric_queries
        reward_queries = self.query_norm(reward_queries)

        flat_queries = reward_queries.reshape(batch_size, num_candidates * self.config.num_horizons * self.config.num_metrics, self.config.query_dim)
        attended, _ = self.reward_attention(flat_queries, obs_tokens, obs_tokens, need_weights=False)
        attended = self.attention_norm(attended + flat_queries)
        metric_logits = self.metric_head(attended).reshape(
            batch_size,
            num_candidates,
            self.config.num_horizons,
            self.config.num_metrics,
        )
        metric_scores = torch.sigmoid(metric_logits)
        aggregated = aggregate_reward_metrics(metric_scores, RewardAggregationConfig())
        return ObservationRewardModelOutput(
            metric_logits=metric_logits,
            metric_scores=metric_scores,
            safe_score=aggregated.safe_score,
            task_score=aggregated.task_score,
            horizon_score=aggregated.horizon_score,
            final_score=aggregated.final_score,
        )
