import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from framework.agent.policy_dummy import DummyPolicy
from framework.algorithms.ppo import PPO
from framework.algorithms.reinforcepp import ReinforcePP
from framework.lightning.config import (
    ActorLearnerLightningConfig,
    LearnerOptimizerConfig,
    actor_learner_lightning_config_from_algorithm,
    optimizer_config_from_algorithm,
    trainer_kwargs_from_learner_config,
)
from framework.lightning.trajectory_module import TrajectoryLightningModule
from framework.runner import learner_factory


def test_ppo_spec_exposes_value_components():
    value_net = torch.nn.Linear(1, 1)
    algo = PPO(
        value_net=value_net,
        policy_lr=2e-4,
        value_lr=3e-4,
        weight_decay=4e-5,
        forward_kl_coef=0.2,
        reverse_kl_coef=0.5,
        distill_temperature=1.3,
        teacher_ckpt="/tmp/teacher.ckpt",
    )

    config = optimizer_config_from_algorithm(algo, {})

    assert algo.value_net is value_net
    assert config == LearnerOptimizerConfig(policy_lr=2e-4, value_lr=3e-4, weight_decay=4e-5)
    assert algo.forward_kl_coef == pytest.approx(0.2)
    assert algo.reverse_kl_coef == pytest.approx(0.5)
    assert algo.distill_temperature == pytest.approx(1.3)
    assert algo.teacher_ckpt == "/tmp/teacher.ckpt"


def test_reinforce_spec_is_runtime_owned_only_by_lightning():
    algo = ReinforcePP(
        clip_eps=0.1,
        policy_lr=1e-4,
    )

    with pytest.raises(RuntimeError, match="no longer owned by framework.algorithms"):
        algo.update(agent=None, batch={}, device=torch.device("cpu"))


def test_runtime_helper_builds_single_lightning_handoff_config():
    algo = PPO(
        value_net=torch.nn.Linear(1, 1),
        clip_eps=0.15,
        vf_coef=0.7,
        ppo_epochs=4,
        policy_lr=2e-4,
        value_lr=3e-4,
        weight_decay=4e-5,
        minibatch_size=8,
        grad_accum_steps=3,
        max_grad_norm=0.8,
        ddp_seed=17,
        eta=0.6,
        value_clip_eps=0.12,
        forward_kl_coef=0.2,
        reverse_kl_coef=0.5,
        distill_temperature=1.4,
        teacher_ckpt="teacher_sparse.ckpt",
    )

    learner_config = actor_learner_lightning_config_from_algorithm(
        algo,
        train_cfg={
            "gamma": 0.97,
            "gae_lambda": 0.91,
            "minibatch_size": 8,
        },
        actor_learner_cfg={
            "mode": "async",
            "num_actors": 4,
            "shards_per_update": 3,
            "poll_interval_s": 0.05,
            "max_shard_version_gap": 5,
            "max_updates": 6,
        },
        algo_meta={
            "algo_key": "ppo",
            "rpp_norm_eps": 1e-6,
        },
    )

    assert learner_config == ActorLearnerLightningConfig(
        algo_kind="ppo",
        optimizer_config=LearnerOptimizerConfig(policy_lr=2e-4, value_lr=3e-4, weight_decay=4e-5),
        eta=0.6,
        clip_eps=0.15,
        vf_coef=0.7,
        value_clip_eps=0.12,
        kl_coef=0.0,
        forward_kl_coef=0.2,
        reverse_kl_coef=0.5,
        distill_temperature=1.4,
        teacher_ckpt="teacher_sparse.ckpt",
        dual_clip=None,
        gamma=0.97,
        gae_lambda=0.91,
        ddp_seed=17,
        minibatch_size=8,
        include_obs=True,
        use_distributed_sampler=True,
        mode="async",
        num_actors=4,
        shards_per_update=3,
        poll_s=0.05,
        max_shard_version_gap=5,
        norm_eps=1e-6,
        inner_epochs=4,
        accumulate_grad_batches=3,
        gradient_clip_val=0.8,
        max_updates=6,
    )

    trainer_kwargs = trainer_kwargs_from_learner_config(learner_config, accelerator="cpu")
    assert trainer_kwargs["max_epochs"] == 24
    assert trainer_kwargs["accumulate_grad_batches"] == 3
    assert trainer_kwargs["gradient_clip_val"] == 0.8
    assert trainer_kwargs["accelerator"] == "cpu"


def test_configure_optimizers_uses_runtime_handoff_config_for_ppo():
    agent = DummyPolicy(ckpt_path=None, device="cpu")
    value_net = torch.nn.Linear(1, 1)
    learner_config = ActorLearnerLightningConfig(
        algo_kind="ppo",
        optimizer_config=LearnerOptimizerConfig(policy_lr=5e-4, value_lr=7e-4, weight_decay=1e-5),
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
        use_distributed_sampler=True,
        mode="async",
        num_actors=1,
        shards_per_update=1,
        poll_s=0.01,
        max_shard_version_gap=2,
        norm_eps=1e-8,
        accumulate_grad_batches=1,
        gradient_clip_val=0.5,
        max_updates=1,
    )

    module = TrajectoryLightningModule(
        agent=agent,
        learner_config=learner_config,
        value_net=value_net,
    )
    optimizer = module.configure_optimizers()

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(7e-4)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.0)


def test_ppo_bundle_keeps_value_net_ddp_wrapping_contract(monkeypatch):
    class _FakeDDP(torch.nn.Module):
        def __init__(self, module, **kwargs):
            super().__init__()
            self.module = module
            self.kwargs = kwargs

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(learner_factory, "DDP", _FakeDDP)
    monkeypatch.setattr(learner_factory.torch.cuda, "is_available", lambda: True)

    algo, value_net, _meta = learner_factory.build_algorithm_bundle(
        {
            "train": {
                "algo": "ppo",
                "policy_lr": 1e-4,
                "lr_value": 2e-4,
            }
        },
        agent=DummyPolicy(ckpt_path=None, device="cpu"),
        device=torch.device("cpu"),
        ddp_enabled=True,
        world_size=2,
        rank=1,
        process_group=object(),
    )

    assert isinstance(value_net, _FakeDDP)
    assert algo.value_net is value_net
    assert value_net.kwargs["process_group"] is not None


def test_reinforce_spec_exposes_distillation_components():
    algo = ReinforcePP(
        clip_eps=0.1,
        policy_lr=1e-4,
        forward_kl_coef=0.2,
        reverse_kl_coef=0.5,
        distill_temperature=0.9,
        teacher_ckpt="teacher_sparse.ckpt",
    )

    assert algo.forward_kl_coef == pytest.approx(0.2)
    assert algo.reverse_kl_coef == pytest.approx(0.5)
    assert algo.distill_temperature == pytest.approx(0.9)
    assert algo.teacher_ckpt == "teacher_sparse.ckpt"
