import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from framework.batch.actor_learner import build_training_batch, compute_gae
from framework.rollout.collector import collect_single_env_shard


def test_compute_gae_bootstraps_truncated_final_step():
    rewards = torch.tensor([1.0, 2.0], dtype=torch.float32)
    dones = torch.tensor([0.0, 1.0], dtype=torch.float32)
    terminated = torch.tensor([0.0, 0.0], dtype=torch.float32)
    values = torch.tensor([0.5, 0.7], dtype=torch.float32)
    last_value = torch.tensor(1.2, dtype=torch.float32)

    adv, ret = compute_gae(
        rewards=rewards,
        dones=dones,
        terminated=terminated,
        values=values,
        last_value=last_value,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.allclose(adv, torch.tensor([3.7, 2.5], dtype=torch.float32))
    assert torch.allclose(ret, torch.tensor([4.2, 3.2], dtype=torch.float32))


class _DummyValueNet(torch.nn.Module):
    def forward(self, obs_t: torch.Tensor) -> torch.Tensor:
        return obs_t[:, 0, 0, 0]


class _FeatureValueNet(torch.nn.Module):
    expects_value_features = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0]


class _FeatureAgent:
    def value_features_from_replay_batch(self, replays):
        vals = [float(rep["value_feature"]) for rep in replays]
        return torch.tensor(vals, dtype=torch.float32).view(-1, 1)


def test_build_training_batch_uses_terminated_for_ppo_bootstrap(tmp_path):
    obs = torch.zeros((2, 18, 64, 64), dtype=torch.float32)
    obs[0, 0, 0, 0] = 1.0
    obs[1, 0, 0, 0] = 2.0
    next_obs = torch.zeros((18, 64, 64), dtype=torch.float32)
    next_obs[0, 0, 0] = 3.0
    shard_path = tmp_path / "shard.pt"

    torch.save(
        {
            "obs": obs,
            "old_logp": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "reward": torch.tensor([10.0, 20.0], dtype=torch.float32),
            "done": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "terminated": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "truncated": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "next_obs": next_obs,
            "done_last": torch.tensor(1.0, dtype=torch.float32),
            "terminated_last": torch.tensor(0.0, dtype=torch.float32),
            "replay": [{}, {}],
        },
        shard_path,
    )

    loaded = build_training_batch(
        selected=[str(shard_path)],
        agent=object(),
        algo_key="ppo",
        device=torch.device("cpu"),
        gamma=1.0,
        gae_lambda=1.0,
        value_net=_DummyValueNet(),
        ddp_enabled=False,
        dist_module=None,
    )

    batch = loaded.batch
    assert loaded.num_samples == 2
    assert torch.allclose(batch["old_value"], torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert torch.allclose(batch["ret"], torch.tensor([33.0, 23.0], dtype=torch.float32))
    assert torch.allclose(batch["adv"], torch.tensor([1.0, -1.0], dtype=torch.float32))
    assert "value_net" not in batch
    assert "value_optim" not in batch


def test_build_training_batch_uses_next_value_feature_for_truncated_feature_critic(tmp_path):
    shard_path = tmp_path / "feature_shard.pt"
    torch.save(
        {
            "obs": torch.zeros((2, 18, 64, 64), dtype=torch.float32),
            "old_logp": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "reward": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "done": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "terminated": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "truncated": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "done_last": torch.tensor(1.0, dtype=torch.float32),
            "terminated_last": torch.tensor(0.0, dtype=torch.float32),
            "replay": [{"value_feature": 5.0}, {"value_feature": 7.0}],
            "next_value_feature": torch.tensor([11.0], dtype=torch.float32),
        },
        shard_path,
    )

    loaded = build_training_batch(
        selected=[str(shard_path)],
        agent=_FeatureAgent(),
        algo_key="ppo",
        device=torch.device("cpu"),
        gamma=1.0,
        gae_lambda=1.0,
        value_net=_FeatureValueNet(),
        ddp_enabled=False,
        dist_module=None,
    )

    batch = loaded.batch
    assert torch.allclose(batch["old_value"], torch.tensor([5.0, 7.0], dtype=torch.float32))
    assert torch.allclose(batch["ret"], torch.tensor([14.0, 13.0], dtype=torch.float32))
    assert torch.allclose(batch["adv"], torch.tensor([1.0, -1.0], dtype=torch.float32))


class _SingleStepDoneEnv:
    def __init__(self) -> None:
        self._step_calls = 0

    def step(self, action):
        del action
        self._step_calls += 1
        next_obs = {
            "front_left": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 9,
            "front": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 9,
            "front_right": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 9,
            "back_left": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 9,
            "back": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 9,
            "back_right": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 9,
        }
        return next_obs, 1.0, False, True, {}

    def reset(self):
        reset_obs = {
            "front_left": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 1,
            "front": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 1,
            "front_right": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 1,
            "back_left": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 1,
            "back": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 1,
            "back_right": torch.ones((4, 4, 3), dtype=torch.uint8).numpy() * 1,
        }
        return reset_obs, {}


class _CollectorAgent:
    def act(self, observation, *, eta=1.0, mode_idx=-1, mode_select="sample"):
        del observation, eta, mode_idx, mode_select
        return (0.0, 0.0, 0.0, 2), torch.tensor(0.0, dtype=torch.float32), {}

    def value_features_from_observation_batch(self, observations):
        vals = [float(obs["front"][0, 0, 0]) for obs in observations]
        return torch.tensor(vals, dtype=torch.float32).view(-1, 1)


def test_collect_single_env_shard_stores_pre_reset_next_value_feature():
    initial_obs = {
        "front_left": torch.zeros((4, 4, 3), dtype=torch.uint8).numpy(),
        "front": torch.zeros((4, 4, 3), dtype=torch.uint8).numpy(),
        "front_right": torch.zeros((4, 4, 3), dtype=torch.uint8).numpy(),
        "back_left": torch.zeros((4, 4, 3), dtype=torch.uint8).numpy(),
        "back": torch.zeros((4, 4, 3), dtype=torch.uint8).numpy(),
        "back_right": torch.zeros((4, 4, 3), dtype=torch.uint8).numpy(),
    }

    shard, _next_obs = collect_single_env_shard(
        env=_SingleStepDoneEnv(),
        agent=_CollectorAgent(),
        obs=initial_obs,
        horizon=1,
        eta=1.0,
        mode_idx=-1,
        mode_select="sample",
        actor_id=0,
        local_ver=1,
        shard_idx=0,
    )

    assert "next_value_feature" in shard
    assert torch.allclose(shard["next_value_feature"], torch.tensor([9.0], dtype=torch.float32))
