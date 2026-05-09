from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .constants import DEFAULT_HORIZON_WEIGHTS, DEFAULT_METRIC_WEIGHTS


@dataclass
class ObservationRewardModelConfig:
    observation_channels: int = 18
    ego_state_dim: int = 8
    hidden_dim: int = 128
    query_dim: int = 64
    trajectory_hidden_dim: int = 128
    num_horizons: int = 8
    num_metrics: int = 8
    architecture: str = "attention"
    num_observation_queries: int = 32
    num_attention_heads: int = 4
    attention_dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ObservationRewardModelConfig":
        return cls(**dict(data))


@dataclass
class RewardAggregationConfig:
    metric_weights: tuple[float, ...] = field(default_factory=lambda: tuple(DEFAULT_METRIC_WEIGHTS))
    horizon_weights: tuple[float, ...] = field(default_factory=lambda: tuple(DEFAULT_HORIZON_WEIGHTS))
    epsilon: float = 1.0e-6


@dataclass
class RewardSupervisionConfig:
    num_horizons: int = 8
    num_metrics: int = 8
    clamp_min: float = 0.0
    clamp_max: float = 1.0


@dataclass
class RewardLossConfig:
    metric_weights: tuple[float, ...] = field(default_factory=lambda: tuple(DEFAULT_METRIC_WEIGHTS))
    horizon_weights: tuple[float, ...] = field(default_factory=lambda: tuple(DEFAULT_HORIZON_WEIGHTS))
