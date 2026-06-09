from __future__ import annotations

from types import SimpleNamespace

import torch

from framework.algorithms.trajectory_policy_core import TrajectoryGRPOObjective, TrajectoryPPOObjective, TrajectorySACObjective
from framework.lightning.config import ActorLearnerLightningConfig, LearnerOptimizerConfig
from framework.lightning.trajectory_module import TrajectoryLightningModule


class _DummyAgent:
    def __init__(self) -> None:
        self.trainable_module = torch.nn.Linear(1, 1)

    def sample_counterfactual_trajectories_from_replay_batch(
        self,
        replay,
        *,
        num_candidates: int,
        candidate_select: str = "topk",
    ):
        del replay, num_candidates, candidate_select
        return {
            "traj_xyyaw": torch.zeros((2, 3, 4, 3), dtype=torch.float32),
            "log_probs": torch.full((2, 3), -0.5, dtype=torch.float32),
        }


class _FusedDummyAgent:
    def __init__(self) -> None:
        self.trainable_module = torch.nn.Linear(1, 1)
        self.fused_calls = 0

    def logp_from_replay_batch(self, replay, *, eta: float = 1.0):
        del replay, eta
        raise AssertionError("training_step should use fused replay policy outputs")

    def sample_counterfactual_trajectories_from_replay_batch(
        self,
        replay,
        *,
        num_candidates: int,
        candidate_select: str = "topk",
    ):
        del replay, num_candidates, candidate_select
        raise AssertionError("training_step should use fused replay policy outputs")

    def replay_policy_outputs_from_replay_batch(
        self,
        replay,
        *,
        eta: float = 1.0,
        num_candidates: int,
        candidate_select: str = "topk",
    ):
        del replay, eta, num_candidates, candidate_select
        self.fused_calls += 1
        return {
            "new_logp": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "counterfactual": {
                "traj_xyyaw": torch.zeros((2, 3, 4, 3), dtype=torch.float32),
                "log_probs": torch.full((2, 3), -0.5, dtype=torch.float32),
            },
        }


def test_ppo_shared_grpo_uses_fused_replay_policy_outputs(monkeypatch) -> None:
    agent = _FusedDummyAgent()
    learner_config = ActorLearnerLightningConfig(
        algo_kind="ppo",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4, value_lr=5.0e-5, weight_decay=0.0),
        eta=1.0,
        clip_eps=0.2,
        vf_coef=0.5,
        value_clip_eps=0.0,
        grpo_coef=0.4,
        grpo_num_candidates=3,
        grpo_candidate_select="topk",
        grpo_norm_eps=1.0e-6,
        grpo_use_rank_adv=False,
        grpo_score_clip=None,
    )
    module = TrajectoryLightningModule(
        agent=agent,
        learner_config=learner_config,
        value_net=torch.nn.Linear(1, 1),
    )

    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_ppo_objective",
        lambda **kwargs: TrajectoryPPOObjective(
            loss=torch.tensor(2.0, dtype=torch.float32),
            loss_pi=torch.tensor(1.0, dtype=torch.float32),
            loss_v=torch.tensor(0.5, dtype=torch.float32),
            approx_kl=torch.tensor(0.1, dtype=torch.float32),
            clip_frac=torch.tensor(0.0, dtype=torch.float32),
            value_clip_frac=torch.tensor(0.0, dtype=torch.float32),
            ratio_mean=torch.tensor(1.0, dtype=torch.float32),
            adv_mean=torch.tensor(0.0, dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_ppo_metrics",
        lambda **kwargs: {"loss_pi": torch.tensor(1.0, dtype=torch.float32)},
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.score_counterfactual_trajectories",
        lambda *args, **kwargs: torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_grpo_objective",
        lambda **kwargs: TrajectoryGRPOObjective(
            loss=torch.tensor(3.0, dtype=torch.float32),
            advantages=torch.zeros((2, 3), dtype=torch.float32),
            score_mean=torch.tensor(0.0, dtype=torch.float32),
            score_std=torch.tensor(1.0, dtype=torch.float32),
            score_min=torch.tensor(-1.0, dtype=torch.float32),
            score_max=torch.tensor(1.0, dtype=torch.float32),
        ),
    )

    batch = {
        "replay": [{"step": 0}, {"step": 1}],
        "adv": torch.tensor([0.2, 0.4], dtype=torch.float32),
        "ret": torch.tensor([1.0, 1.5], dtype=torch.float32),
        "old_logp": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "old_value": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "obs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
    }

    loss = module.training_step(batch, batch_idx=0)

    assert torch.is_tensor(loss)
    assert torch.allclose(loss.detach(), torch.tensor(3.2, dtype=torch.float32))
    assert agent.fused_calls == 1


def test_ppo_training_step_adds_shared_grpo_auxiliary_loss(monkeypatch) -> None:
    agent = _DummyAgent()
    learner_config = ActorLearnerLightningConfig(
        algo_kind="ppo",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4, value_lr=5.0e-5, weight_decay=0.0),
        eta=1.0,
        clip_eps=0.2,
        vf_coef=0.5,
        value_clip_eps=0.0,
        grpo_coef=0.4,
        grpo_num_candidates=3,
        grpo_candidate_select="topk",
        grpo_norm_eps=1.0e-6,
        grpo_use_rank_adv=False,
        grpo_score_clip=None,
    )
    module = TrajectoryLightningModule(
        agent=agent,
        learner_config=learner_config,
        value_net=torch.nn.Linear(1, 1),
    )

    monkeypatch.setattr(
        "framework.lightning.trajectory_module.agent_logp_from_replay_batch",
        lambda *args, **kwargs: torch.tensor([0.1, 0.2], dtype=torch.float32),
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_ppo_objective",
        lambda **kwargs: TrajectoryPPOObjective(
            loss=torch.tensor(2.0, dtype=torch.float32),
            loss_pi=torch.tensor(1.0, dtype=torch.float32),
            loss_v=torch.tensor(0.5, dtype=torch.float32),
            approx_kl=torch.tensor(0.1, dtype=torch.float32),
            clip_frac=torch.tensor(0.0, dtype=torch.float32),
            value_clip_frac=torch.tensor(0.0, dtype=torch.float32),
            ratio_mean=torch.tensor(1.0, dtype=torch.float32),
            adv_mean=torch.tensor(0.0, dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_ppo_metrics",
        lambda **kwargs: {"loss_pi": torch.tensor(1.0, dtype=torch.float32)},
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.score_counterfactual_trajectories",
        lambda *args, **kwargs: torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_grpo_objective",
        lambda **kwargs: TrajectoryGRPOObjective(
            loss=torch.tensor(3.0, dtype=torch.float32),
            advantages=torch.zeros((2, 3), dtype=torch.float32),
            score_mean=torch.tensor(0.0, dtype=torch.float32),
            score_std=torch.tensor(1.0, dtype=torch.float32),
            score_min=torch.tensor(-1.0, dtype=torch.float32),
            score_max=torch.tensor(1.0, dtype=torch.float32),
        ),
    )

    batch = {
        "replay": [{"step": 0}, {"step": 1}],
        "adv": torch.tensor([0.2, 0.4], dtype=torch.float32),
        "ret": torch.tensor([1.0, 1.5], dtype=torch.float32),
        "old_logp": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "old_value": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "obs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
    }

    loss = module.training_step(batch, batch_idx=0)

    assert torch.is_tensor(loss)
    assert torch.allclose(loss.detach(), torch.tensor(3.2, dtype=torch.float32))


def test_training_step_applies_closed_loop_loss_coef_before_grpo(monkeypatch) -> None:
    agent = _DummyAgent()
    learner_config = ActorLearnerLightningConfig(
        algo_kind="reinforcepp",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4, value_lr=None, weight_decay=0.0),
        eta=1.0,
        clip_eps=0.2,
        closed_loop_loss_coef=0.5,
        grpo_enabled=True,
        grpo_coef=1.0,
        grpo_num_candidates=3,
        grpo_candidate_select="topk",
        grpo_norm_eps=1.0e-6,
        grpo_use_rank_adv=False,
        grpo_score_clip=None,
    )
    module = TrajectoryLightningModule(agent=agent, learner_config=learner_config)

    monkeypatch.setattr(
        "framework.lightning.trajectory_module.agent_logp_from_replay_batch",
        lambda *args, **kwargs: torch.tensor([0.1, 0.2], dtype=torch.float32),
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_reinforce_objective",
        lambda **kwargs: SimpleNamespace(
            loss=torch.tensor(2.0, dtype=torch.float32),
            loss_pi=torch.tensor(2.0, dtype=torch.float32),
            approx_kl=torch.tensor(0.0, dtype=torch.float32),
            clip_frac=torch.tensor(0.0, dtype=torch.float32),
            ratio_mean=torch.tensor(1.0, dtype=torch.float32),
            adv_mean=torch.tensor(0.0, dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_reinforce_metrics",
        lambda **kwargs: {"loss_pi": torch.tensor(2.0, dtype=torch.float32)},
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.score_counterfactual_trajectories",
        lambda *args, **kwargs: torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_grpo_objective",
        lambda **kwargs: TrajectoryGRPOObjective(
            loss=torch.tensor(3.0, dtype=torch.float32),
            advantages=torch.zeros((2, 3), dtype=torch.float32),
            score_mean=torch.tensor(0.0, dtype=torch.float32),
            score_std=torch.tensor(1.0, dtype=torch.float32),
            score_min=torch.tensor(-1.0, dtype=torch.float32),
            score_max=torch.tensor(1.0, dtype=torch.float32),
        ),
    )

    batch = {
        "replay": [{"step": 0}, {"step": 1}],
        "adv": torch.tensor([0.2, 0.4], dtype=torch.float32),
        "ret": torch.tensor([1.0, 1.5], dtype=torch.float32),
        "old_logp": torch.tensor([0.0, 0.0], dtype=torch.float32),
    }

    loss = module.training_step(batch, batch_idx=0)

    assert torch.is_tensor(loss)
    assert torch.allclose(loss.detach(), torch.tensor(4.0, dtype=torch.float32))
    assert module.latest_metrics["closed_loop_loss_coef"] == 0.5


def test_sac_training_step_uses_entropy_regularized_objective(monkeypatch) -> None:
    agent = _DummyAgent()
    learner_config = ActorLearnerLightningConfig(
        algo_kind="sac",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4, value_lr=None, weight_decay=0.0),
        eta=1.0,
        clip_eps=0.2,
        kl_coef=0.03,
        sac_entropy_coef=0.02,
        closed_loop_loss_coef=0.5,
    )
    module = TrajectoryLightningModule(agent=agent, learner_config=learner_config)

    monkeypatch.setattr(
        "framework.lightning.trajectory_module.agent_logp_from_replay_batch",
        lambda *args, **kwargs: torch.tensor([-0.1, -0.2], dtype=torch.float32),
    )
    seen = {}

    def _fake_sac_objective(**kwargs):
        seen.update(kwargs)
        return TrajectorySACObjective(
            loss=torch.tensor(2.0, dtype=torch.float32),
            loss_pi=torch.tensor(2.0, dtype=torch.float32),
            loss_pg=torch.tensor(1.8, dtype=torch.float32),
            loss_entropy=torch.tensor(0.2, dtype=torch.float32),
            approx_kl=torch.tensor(0.1, dtype=torch.float32),
            clip_frac=torch.tensor(0.25, dtype=torch.float32),
            ratio_mean=torch.tensor(1.1, dtype=torch.float32),
            adv_mean=torch.tensor(0.0, dtype=torch.float32),
            logp_mean=torch.tensor(-0.15, dtype=torch.float32),
            entropy_coef=torch.tensor(0.02, dtype=torch.float32),
        )

    monkeypatch.setattr("framework.lightning.trajectory_module.compute_sac_objective", _fake_sac_objective)

    batch = {
        "replay": [{"step": 0}, {"step": 1}],
        "adv": torch.tensor([0.2, 0.4], dtype=torch.float32),
        "ret": torch.tensor([1.0, 1.5], dtype=torch.float32),
        "old_logp": torch.tensor([-0.3, -0.4], dtype=torch.float32),
    }

    loss = module.training_step(batch, batch_idx=0)

    assert torch.is_tensor(loss)
    assert torch.allclose(loss.detach(), torch.tensor(1.0, dtype=torch.float32))
    assert seen["entropy_coef"] == 0.02
    assert seen["kl_coef"] == 0.03
    assert torch.equal(seen["old_logp"], batch["old_logp"])
    assert module.latest_metrics["sac_pg_loss"] == torch.tensor(1.8, dtype=torch.float32).item()
    assert module.latest_metrics["sac_entropy_loss"] == torch.tensor(0.2, dtype=torch.float32).item()
    assert module.latest_metrics["sac_logp_mean"] == torch.tensor(-0.15, dtype=torch.float32).item()
    assert module.latest_metrics["closed_loop_loss_coef"] == 0.5


def test_training_step_records_update_timing_parts(monkeypatch) -> None:
    agent = _DummyAgent()
    learner_config = ActorLearnerLightningConfig(
        algo_kind="ppo",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4, value_lr=5.0e-5, weight_decay=0.0),
        eta=1.0,
        clip_eps=0.2,
        vf_coef=0.5,
        value_clip_eps=0.0,
        grpo_coef=0.0,
        grpo_enabled=False,
    )
    module = TrajectoryLightningModule(
        agent=agent,
        learner_config=learner_config,
        value_net=torch.nn.Linear(1, 1),
    )

    monkeypatch.setattr(
        "framework.lightning.trajectory_module.agent_logp_from_replay_batch",
        lambda *args, **kwargs: torch.tensor([0.1, 0.2], dtype=torch.float32),
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_ppo_objective",
        lambda **kwargs: TrajectoryPPOObjective(
            loss=torch.tensor(2.0, dtype=torch.float32),
            loss_pi=torch.tensor(1.0, dtype=torch.float32),
            loss_v=torch.tensor(0.5, dtype=torch.float32),
            approx_kl=torch.tensor(0.1, dtype=torch.float32),
            clip_frac=torch.tensor(0.0, dtype=torch.float32),
            value_clip_frac=torch.tensor(0.0, dtype=torch.float32),
            ratio_mean=torch.tensor(1.0, dtype=torch.float32),
            adv_mean=torch.tensor(0.0, dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_ppo_metrics",
        lambda **kwargs: {"loss_pi": torch.tensor(1.0, dtype=torch.float32)},
    )

    batch = {
        "replay": [{"step": 0}, {"step": 1}],
        "adv": torch.tensor([0.2, 0.4], dtype=torch.float32),
        "ret": torch.tensor([1.0, 1.5], dtype=torch.float32),
        "old_logp": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "old_value": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "obs": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
    }

    module.training_step(batch, batch_idx=0)

    timing_parts = module.aggregated_update_timing()
    assert timing_parts["timed_minibatches"] == 1.0
    for key in [
        "new_logp_s",
        "value_s",
        "objective_s",
        "metrics_compute_s",
        "distill_s",
        "metrics_log_s",
        "training_step_total_s",
        "new_logp_s_avg",
        "training_step_total_s_max",
    ]:
        assert key in timing_parts
        assert timing_parts[key] >= 0.0


def test_grpo_only_training_step_uses_only_counterfactual_objective(monkeypatch) -> None:
    agent = _DummyAgent()
    learner_config = ActorLearnerLightningConfig(
        algo_kind="grpo_only",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4, value_lr=None, weight_decay=0.0),
        eta=1.0,
        clip_eps=0.2,
        grpo_enabled=True,
        grpo_coef=1.0,
        grpo_num_candidates=3,
        grpo_candidate_select="topk",
        grpo_norm_eps=1.0e-6,
        grpo_use_rank_adv=False,
        grpo_score_clip=None,
    )
    module = TrajectoryLightningModule(agent=agent, learner_config=learner_config)

    def _unexpected_base_objective(**kwargs):
        del kwargs
        raise AssertionError("grpo_only must not compute PPO/Reinforce objective")

    monkeypatch.setattr("framework.lightning.trajectory_module.agent_logp_from_replay_batch", _unexpected_base_objective)
    monkeypatch.setattr("framework.lightning.trajectory_module.compute_ppo_objective", _unexpected_base_objective)
    monkeypatch.setattr("framework.lightning.trajectory_module.compute_reinforce_objective", _unexpected_base_objective)
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.score_counterfactual_trajectories",
        lambda *args, **kwargs: torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        "framework.lightning.trajectory_module.compute_grpo_objective",
        lambda **kwargs: TrajectoryGRPOObjective(
            loss=torch.tensor(3.0, dtype=torch.float32),
            advantages=torch.zeros((2, 3), dtype=torch.float32),
            score_mean=torch.tensor(0.0, dtype=torch.float32),
            score_std=torch.tensor(1.0, dtype=torch.float32),
            score_min=torch.tensor(-1.0, dtype=torch.float32),
            score_max=torch.tensor(1.0, dtype=torch.float32),
        ),
    )

    batch = {
        "replay": [{"step": 0}, {"step": 1}],
        "adv": torch.tensor([0.2, 0.4], dtype=torch.float32),
        "ret": torch.tensor([1.0, 1.5], dtype=torch.float32),
        "old_logp": torch.tensor([0.0, 0.0], dtype=torch.float32),
    }

    loss = module.training_step(batch, batch_idx=0)

    assert torch.is_tensor(loss)
    assert torch.allclose(loss.detach(), torch.tensor(3.0, dtype=torch.float32))
    assert module.latest_metrics["grpo_loss"] == 3.0
