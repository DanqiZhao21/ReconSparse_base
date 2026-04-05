import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy
from framework.lightning.config import ActorLearnerLightningConfig, LearnerOptimizerConfig
from framework.lightning.trajectory_module import TrajectoryLightningModule


class _FakePPOAgent:
    def logp_from_replay_batch(self, replays, *, eta: float = 1.0) -> torch.Tensor:
        del replays, eta
        return torch.zeros((2,), dtype=torch.float32)

    def value_features_from_replay_batch(self, replays) -> torch.Tensor:
        del replays
        return torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )


class _FakeBackbone(torch.nn.Module):
    def forward(self, imgs: torch.Tensor):
        batch_size = int(imgs.shape[0])
        return [torch.arange(batch_size * 2 * 1 * 1 * 3, dtype=torch.float32).view(batch_size, 2, 3, 1, 1)]


class _FakeSparseDriveModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._backbone = _FakeBackbone()
        self._status_encoding = torch.nn.Linear(8, 5, bias=False)


def _ppo_config() -> ActorLearnerLightningConfig:
    return ActorLearnerLightningConfig(
        algo_kind="ppo",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1e-3, value_lr=1e-3),
        eta=1.0,
        clip_eps=0.2,
        vf_coef=0.5,
        value_clip_eps=0.0,
        kl_coef=0.0,
        forward_kl_coef=0.0,
        reverse_kl_coef=0.0,
        distill_temperature=1.0,
        teacher_ckpt=None,
        dual_clip=None,
        gamma=0.99,
        gae_lambda=0.95,
        ddp_seed=0,
        minibatch_size=2,
        include_obs=True,
        use_distributed_sampler=False,
        mode="async",
        num_actors=1,
        shards_per_update=1,
        poll_s=0.2,
        max_shard_version_gap=2,
        norm_eps=1e-8,
        inner_epochs=1,
        accumulate_grad_batches=1,
        gradient_clip_val=0.0,
        max_updates=1,
    )


def test_training_step_prefers_agent_value_features_for_ppo():
    module = TrajectoryLightningModule(
        agent=_FakePPOAgent(),
        learner_config=_ppo_config(),
        value_net=torch.nn.Linear(4, 1),
    )

    batch = {
        "replay": [{"id": 0}, {"id": 1}],
        "adv": torch.tensor([1.0, -1.0], dtype=torch.float32),
        "ret": torch.tensor([0.5, -0.25], dtype=torch.float32),
        "old_logp": torch.zeros((2,), dtype=torch.float32),
    }

    loss = module.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_sparsedrive_v2_value_features_from_replay_batch_returns_fixed_width_features():
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device_override = "cpu"
    policy._model = _FakeSparseDriveModel()

    replays = [
        {
            "camera_feature": {"imgs": torch.zeros((1, 2, 3, 4, 4), dtype=torch.float32)},
            "status_feature": torch.zeros((1, 8), dtype=torch.float32),
        },
        {
            "camera_feature": {"imgs": torch.ones((1, 2, 3, 4, 4), dtype=torch.float32)},
            "status_feature": torch.ones((1, 8), dtype=torch.float32),
        },
    ]

    features = policy.value_features_from_replay_batch(replays)

    assert features.shape == (2, 8)
    assert features.requires_grad is False
    assert not features.is_inference()


def test_sparsedrive_v2_value_features_require_replay_status_feature():
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device_override = "cpu"
    policy._model = _FakeSparseDriveModel()

    with pytest.raises(RuntimeError):
        policy.value_features_from_replay_batch(
            [
                {
                    "camera_feature": {"imgs": torch.zeros((1, 2, 3, 4, 4), dtype=torch.float32)},
                }
            ]
        )


class _FakeReinforceDistillAgent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor([[0.0]], dtype=torch.float32))

    def logp_from_replay_batch(self, replays, *, eta: float = 1.0) -> torch.Tensor:
        del replays, eta
        return self.param.view(1).expand(2)

    def distill_student_log_probs_from_replay_batch(
        self,
        replays,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        del replays, temperature
        return torch.log(
            torch.tensor(
                [
                    [0.7, 0.3],
                    [0.4, 0.6],
                ],
                dtype=torch.float32,
            )
        )

    def distill_teacher_log_probs_from_replay_batch(
        self,
        replays,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        del replays, temperature
        return torch.log(
            torch.tensor(
                [
                    [0.6, 0.4],
                    [0.5, 0.5],
                ],
                dtype=torch.float32,
            )
        )

    @property
    def trainable_module(self):
        return self


def _reinforce_config() -> ActorLearnerLightningConfig:
    return ActorLearnerLightningConfig(
        algo_kind="reinforcepp",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1e-3, value_lr=None),
        eta=1.0,
        clip_eps=0.2,
        vf_coef=0.0,
        value_clip_eps=0.0,
        kl_coef=0.0,
        forward_kl_coef=0.2,
        reverse_kl_coef=0.5,
        distill_temperature=1.0,
        teacher_ckpt="teacher_sparse.ckpt",
        dual_clip=None,
        gamma=0.99,
        gae_lambda=0.95,
        ddp_seed=0,
        minibatch_size=2,
        include_obs=False,
        use_distributed_sampler=False,
        mode="async",
        num_actors=1,
        shards_per_update=1,
        poll_s=0.2,
        max_shard_version_gap=2,
        norm_eps=1e-8,
        inner_epochs=1,
        accumulate_grad_batches=1,
        gradient_clip_val=0.0,
        max_updates=1,
    )


def test_training_step_adds_forward_and_reverse_kl_when_agent_exposes_distill_hooks():
    agent = _FakeReinforceDistillAgent()
    module = TrajectoryLightningModule(
        agent=agent,
        learner_config=_reinforce_config(),
        value_net=None,
    )

    batch = {
        "replay": [{"id": 0}, {"id": 1}],
        "adv": torch.tensor([1.0, -1.0], dtype=torch.float32),
        "ret": torch.zeros((2,), dtype=torch.float32),
        "old_logp": torch.zeros((2,), dtype=torch.float32),
    }

    loss = module.training_step(batch, batch_idx=0)

    student_log_probs = agent.distill_student_log_probs_from_replay_batch(batch["replay"])
    teacher_log_probs = agent.distill_teacher_log_probs_from_replay_batch(batch["replay"])
    teacher_probs = teacher_log_probs.exp()
    student_probs = student_log_probs.exp()
    forward_kl = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=1).mean()
    reverse_kl = torch.sum(student_probs * (student_log_probs - teacher_log_probs), dim=1).mean()
    expected = 0.2 * forward_kl + 0.5 * reverse_kl

    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    assert module.latest_metrics["forward_kl"] == pytest.approx(forward_kl.item(), abs=1e-6)
    assert module.latest_metrics["reverse_kl"] == pytest.approx(reverse_kl.item(), abs=1e-6)
    assert module.latest_metrics["loss_forward_kl"] == pytest.approx((0.2 * forward_kl).item(), abs=1e-6)
    assert module.latest_metrics["loss_reverse_kl"] == pytest.approx((0.5 * reverse_kl).item(), abs=1e-6)
