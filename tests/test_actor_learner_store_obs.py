from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from framework.batch import build_training_batch
from framework.lightning.config import actor_learner_lightning_config_from_algorithm
from framework.rollout.collector import collect_single_env_shard
from framework.runner.actor_runtime import resolve_store_obs
from framework.runner.learner_factory import ValueHead, ValueNet


class _TinyEnv:
    def __init__(self) -> None:
        self.step_count = 0
        self.reset_count = 0

    def step(self, action: Any):
        del action
        self.step_count += 1
        return _obs(self.step_count), 1.0, False, False, {}

    def reset(self):
        self.reset_count += 1
        return _obs(0), {}


class _DoneAfterFirstStepEnv:
    def __init__(self) -> None:
        self.step_count = 0
        self.reset_count = 0

    def step(self, action: Any):
        del action
        self.step_count += 1
        return _obs(self.step_count), 1.0, True, False, {"terminal_kind": "failure"}

    def reset(self):
        self.reset_count += 1
        return _obs(100 + self.reset_count), {}


class _TinyAgent:
    def act(self, obs, *, eta: float, mode_idx: int, mode_select: str):
        del obs, eta, mode_idx, mode_select
        return (0.0, 0.0, 0.0, 2), torch.tensor(0.0), {"step": 0}


class _ReplayFeatureAgent:
    @property
    def value_feature_dim(self) -> int:
        return 2

    def supports_value_features(self) -> bool:
        return True

    def value_features_from_replay_batch(self, replay):
        return torch.ones((len(replay), 2), dtype=torch.float32)


def _obs(seed: int) -> dict[str, np.ndarray]:
    image = np.full((8, 8, 3), int(seed), dtype=np.uint8)
    return {
        "front_left": image,
        "front": image,
        "front_right": image,
        "back_left": image,
        "back": image,
        "back_right": image,
    }


def _write_shard(path: Path, shard: dict[str, Any]) -> None:
    torch.save(shard, path)


def test_resolve_store_obs_auto_skips_replay_feature_paths() -> None:
    assert resolve_store_obs({"algo": "reinforcepp"}, {"store_obs": "auto"}, _ReplayFeatureAgent()) is False
    assert (
        resolve_store_obs(
            {"algo": "ppo", "critic_use_agent_features": True},
            {"store_obs": "auto"},
            _ReplayFeatureAgent(),
        )
        is False
    )


def test_resolve_store_obs_auto_keeps_ppo_fallback_obs() -> None:
    assert (
        resolve_store_obs(
            {"algo": "ppo", "critic_use_agent_features": False},
            {"store_obs": "auto"},
            _ReplayFeatureAgent(),
        )
        is True
    )
    assert resolve_store_obs({"algo": "ppo"}, {"store_obs": "auto"}, object()) is True


def test_collect_single_env_shard_can_skip_obs_storage() -> None:
    shard, _next_obs = collect_single_env_shard(
        env=_TinyEnv(),
        agent=_TinyAgent(),
        obs=_obs(0),
        horizon=2,
        eta=1.0,
        mode_idx=-1,
        mode_select="sample",
        actor_id=0,
        local_ver=1,
        shard_idx=0,
        store_obs=False,
    )

    assert "obs" not in shard
    assert "next_obs" not in shard
    assert shard["reward"].shape == (2,)
    assert len(shard["replay"]) == 2


def test_collect_single_env_shard_can_finish_early_on_done_for_expensive_envs() -> None:
    env = _DoneAfterFirstStepEnv()
    shard, next_obs = collect_single_env_shard(
        env=env,
        agent=_TinyAgent(),
        obs=_obs(0),
        horizon=4,
        eta=1.0,
        mode_idx=-1,
        mode_select="sample",
        actor_id=0,
        local_ver=1,
        shard_idx=0,
        store_obs=False,
        end_shard_on_done=True,
    )

    assert shard["reward"].shape == (1,)
    assert int(shard["meta"]["timing"]["done_count"]) == 1
    assert int(shard["meta"]["timing"]["reset_count"]) == 1
    assert env.step_count == 1
    assert env.reset_count == 1
    assert int(next_obs["front"][0, 0, 0]) == 101


def test_replay_feature_ppo_batch_accepts_shard_without_obs(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.pt"
    _write_shard(
        shard_path,
        {
            "old_logp": torch.zeros(2),
            "reward": torch.ones(2),
            "done": torch.zeros(2),
            "terminated": torch.zeros(2),
            "done_last": torch.tensor(0.0),
            "terminated_last": torch.tensor(0.0),
            "replay": [{"step": 0}, {"step": 1}],
            "next_value_feature": torch.ones(2),
        },
    )

    loaded = build_training_batch(
        selected=[str(shard_path)],
        agent=_ReplayFeatureAgent(),
        algo_key="ppo",
        device=torch.device("cpu"),
        gamma=0.99,
        gae_lambda=0.95,
        value_net=ValueHead(input_dim=2),
        ddp_enabled=False,
        dist_module=None,
    )

    assert loaded.num_samples == 2
    assert loaded.batch["obs_batch"].shape == (0, 18, 64, 64)


def test_ppo_fallback_batch_requires_obs(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.pt"
    _write_shard(
        shard_path,
        {
            "old_logp": torch.zeros(2),
            "reward": torch.ones(2),
            "done": torch.zeros(2),
            "terminated": torch.zeros(2),
            "replay": [{"step": 0}, {"step": 1}],
        },
    )

    with pytest.raises(RuntimeError, match="requires shard\\['obs'\\]"):
        build_training_batch(
            selected=[str(shard_path)],
            agent=object(),
            algo_key="ppo",
            device=torch.device("cpu"),
            gamma=0.99,
            gae_lambda=0.95,
            value_net=ValueNet(),
            ddp_enabled=False,
            dist_module=None,
        )


def test_ppo_replay_feature_critic_learner_config_does_not_include_obs() -> None:
    algo = type(
        "Algo",
        (),
        {
            "variant": "ppo",
            "policy_lr": 1.0e-4,
            "value_lr": 1.0e-4,
            "weight_decay": 0.0,
            "eta": 1.0,
            "clip_eps": 0.2,
            "vf_coef": 0.5,
            "value_clip_eps": 0.0,
            "kl_coef": 0.0,
            "ppo_epochs": 1,
            "grad_accum_steps": 1,
            "use_distributed_sampler": True,
            "ddp_seed": 0,
        },
    )()

    learner_cfg = actor_learner_lightning_config_from_algorithm(
        algo,
        train_cfg={"gamma": 0.99, "gae_lambda": 0.95},
        actor_learner_cfg={"mode": "async", "num_actors": 1, "shards_per_update": 1},
        algo_meta={
            "algo_key": "ppo",
            "eta": 1.0,
            "clip_eps": 0.2,
            "value_clip_eps": 0.0,
            "critic_use_agent_features": True,
        },
    )

    assert learner_cfg.include_obs is False
