from __future__ import annotations

import torch
from torch import nn

from framework.rewardmodel.config import ObservationRewardModelConfig


class ObservationEncoder(nn.Module):
    def __init__(self, config: ObservationRewardModelConfig) -> None:
        super().__init__()
        if not isinstance(config, ObservationRewardModelConfig):
            raise TypeError("ObservationEncoder requires an ObservationRewardModelConfig")
        in_channels = int(config.observation_channels)
        model_dim = int(config.hidden_dim)
        num_queries = int(config.num_observation_queries)
        num_heads = int(config.num_attention_heads)
        dropout = float(config.attention_dropout)

        mid_dim = max(16, model_dim // 2)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(mid_dim, model_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(model_dim, model_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.query_tokens = nn.Parameter(torch.randn(num_queries, model_dim) * 0.02)
        self.query_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.backbone(observations)
        tokens = features.flatten(2).transpose(1, 2)
        queries = self.query_tokens[None, :, :].expand(observations.shape[0], -1, -1)
        compressed, _ = self.query_attention(queries, tokens, tokens, need_weights=False)
        return self.output_norm(compressed)
