from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from framework.batch import build_training_batch
from framework.lightning.config import actor_learner_lightning_config_from_algorithm
from framework.rollout.collector import collect_single_env_shard, collect_vector_env_shards
from framework.runner.actor_runtime import resolve_store_obs
from framework.runner.learner_factory import ValueHead, ValueNet


class _TinyEnv:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self, action: Any):
        del action
        self.step_count += 1
        return _obs(self.step_count), 1.0, False, False, {}

    def reset(self):
        return _obs(0), {}


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


class _ZeroValueFeatureNet(torch.nn.Module):
    expects_value_features = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros((features.shape[0],), dtype=torch.float32, device=features.device)


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


def test_collect_single_env_shard_injects_current_gt_reference_info() -> None:
    class EnvWithGtInfo(_TinyEnv):
        def reset(self):
            return _obs(0), {
                "grpo_gt_sample_token": "tok-reset",
                "grpo_gt_frame_idx": 10,
            }

        def step(self, action: Any):
            del action
            self.step_count += 1
            return _obs(self.step_count), 1.0, False, False, {
                "grpo_gt_sample_token": f"tok-step-{self.step_count}",
                "grpo_gt_frame_idx": 10 + self.step_count,
            }

    env = EnvWithGtInfo()
    obs, info = env.reset()
    shard, _next_obs = collect_single_env_shard(
        env=env,
        agent=_TinyAgent(),
        obs=obs,
        info=info,
        horizon=2,
        eta=1.0,
        mode_idx=-1,
        mode_select="sample",
        actor_id=0,
        local_ver=1,
        shard_idx=0,
        store_obs=False,
    )

    assert shard["replay"][0]["gt_sample_token_override"] == "tok-reset"
    assert shard["replay"][0]["gt_frame_idx_override"] == 10
    assert shard["replay"][1]["gt_sample_token_override"] == "tok-step-1"
    assert shard["replay"][1]["gt_frame_idx_override"] == 11


def test_collect_single_env_shard_injects_front_obstacle_safety_context() -> None:
    class EnvWithSafetyInfo(_TinyEnv):
        def reset(self):
            return _obs(0), {
                "front_obstacle_available": True,
                "front_obstacle_gap_m": 6.5,
                "front_obstacle_lateral_m": 0.4,
                "front_obstacle_closing_speed_mps": 3.0,
                "front_obstacle_ttc_s": 1.2,
                "front_obstacle_category": "vehicle.car",
            }

    shard, _next_obs = collect_single_env_shard(
        env=EnvWithSafetyInfo(),
        agent=_TinyAgent(),
        obs=_obs(0),
        horizon=1,
        eta=1.0,
        mode_idx=-1,
        mode_select="sample",
        actor_id=0,
        local_ver=1,
        shard_idx=0,
        info=EnvWithSafetyInfo().reset()[1],
        store_obs=False,
    )

    replay = shard["replay"][0]
    assert replay["front_obstacle_available"] is True
    assert replay["front_obstacle_gap_m"] == 6.5
    assert replay["front_obstacle_ttc_s"] == 1.2
    assert replay["front_obstacle_category"] == "vehicle.car"


def test_collect_single_env_shard_can_end_on_done_without_resetting() -> None:
    class DoneEnv:
        def __init__(self) -> None:
            self.step_count = 0
            self.reset_count = 0

        def step(self, action: Any):
            del action
            self.step_count += 1
            done = self.step_count == 2
            return _obs(self.step_count), 1.0, done, False, {"step": self.step_count}

        def reset(self):
            self.reset_count += 1
            return _obs(100 + self.reset_count), {"reset": self.reset_count}

    env = DoneEnv()

    shard, next_obs, info = collect_single_env_shard(
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
        return_info=True,
        end_shard_on_done=True,
    )

    assert env.reset_count == 0
    assert shard["reward"].shape == (2,)
    assert shard["done"].tolist() == [0.0, 1.0]
    assert shard["meta"]["num_steps"] == 2
    assert shard["meta"]["needs_reset_after"] is True
    assert next_obs is None
    assert info == {"step": 2}


def test_collect_vector_env_shards_injects_current_gt_reference_info() -> None:
    class VecEnvWithGtInfo:
        def __init__(self) -> None:
            self.step_count = 0

        def step(self, actions):
            del actions
            self.step_count += 1
            return (
                [_obs(self.step_count), _obs(self.step_count + 10)],
                [1.0, 2.0],
                [False, False],
                [False, False],
                [
                    {"grpo_gt_sample_token": f"tok-a-step-{self.step_count}", "grpo_gt_frame_idx": 20 + self.step_count},
                    {"grpo_gt_sample_token": f"tok-b-step-{self.step_count}", "grpo_gt_frame_idx": 30 + self.step_count},
                ],
            )

        def reset_one(self, env_idx: int):
            return _obs(100 + int(env_idx)), {"grpo_gt_sample_token": f"tok-reset-{env_idx}"}

        def call_one(self, env_idx: int, method_name: str, *args):
            del env_idx, method_name, args

    class BatchAgent:
        def act_batch(self, obs_list, *, eta: float, mode_idx: int, mode_select: str):
            del eta, mode_idx, mode_select
            actions = [(0.0, 0.0, 0.0, 2) for _ in obs_list]
            logps = [torch.tensor(0.0) for _ in obs_list]
            replays = [{"env": idx} for idx, _obs_item in enumerate(obs_list)]
            return actions, logps, replays

    shards, _obs_list = collect_vector_env_shards(
        vec_env=VecEnvWithGtInfo(),
        agent=BatchAgent(),
        obs_list=[_obs(0), _obs(1)],
        info_list=[
            {"grpo_gt_sample_token": "tok-a-reset", "grpo_gt_frame_idx": 20},
            {"grpo_gt_sample_token": "tok-b-reset", "grpo_gt_frame_idx": 30},
        ],
        num_envs_per_actor=2,
        horizon=2,
        eta=1.0,
        mode_idx=-1,
        mode_select="sample",
        actor_id=0,
        local_ver=1,
        shard_idx_per_env=[0, 0],
        store_obs=False,
    )

    assert shards[0]["replay"][0]["gt_sample_token_override"] == "tok-a-reset"
    assert shards[0]["replay"][1]["gt_sample_token_override"] == "tok-a-step-1"
    assert shards[1]["replay"][0]["gt_sample_token_override"] == "tok-b-reset"
    assert shards[1]["replay"][1]["gt_sample_token_override"] == "tok-b-step-1"


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


def test_reinforcepp_batch_keeps_raw_returns_as_advantage(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.pt"
    _write_shard(
        shard_path,
        {
            "old_logp": torch.zeros(2),
            "reward": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "done": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "replay": [{"step": 0}, {"step": 1}],
        },
    )

    loaded = build_training_batch(
        selected=[str(shard_path)],
        agent=object(),
        algo_key="reinforcepp",
        device=torch.device("cpu"),
        gamma=0.5,
        gae_lambda=0.95,
        value_net=None,
        ddp_enabled=False,
        dist_module=None,
    )

    assert torch.allclose(loaded.batch["ret"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(loaded.batch["adv"], torch.tensor([1.0, 0.0]))


def test_ppo_batch_keeps_raw_gae_as_advantage(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.pt"
    _write_shard(
        shard_path,
        {
            "old_logp": torch.zeros(2),
            "reward": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "done": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "terminated": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "done_last": torch.tensor(1.0),
            "terminated_last": torch.tensor(1.0),
            "replay": [{"step": 0}, {"step": 1}],
            "next_value_feature": torch.ones(2),
        },
    )

    loaded = build_training_batch(
        selected=[str(shard_path)],
        agent=_ReplayFeatureAgent(),
        algo_key="ppo",
        device=torch.device("cpu"),
        gamma=0.5,
        gae_lambda=0.95,
        value_net=_ZeroValueFeatureNet(),
        ddp_enabled=False,
        dist_module=None,
    )

    assert torch.allclose(loaded.batch["ret"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(loaded.batch["adv"], torch.tensor([1.0, 0.0]))


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
