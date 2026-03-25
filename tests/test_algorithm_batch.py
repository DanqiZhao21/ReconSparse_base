import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from framework.algorithms.trajectory_batch import build_training_batch, compute_gae


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
        algo_key="ppo",
        device=torch.device("cpu"),
        gamma=1.0,
        gae_lambda=1.0,
        value_net=_DummyValueNet(),
        value_optim=None,
        ddp_enabled=False,
        dist_module=None,
    )

    batch = loaded.batch
    assert loaded.num_samples == 2
    assert torch.allclose(batch["old_value"], torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert torch.allclose(batch["ret"], torch.tensor([33.0, 23.0], dtype=torch.float32))
    assert torch.allclose(batch["adv"], torch.tensor([1.0, -1.0], dtype=torch.float32))
