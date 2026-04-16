from __future__ import annotations

from pathlib import Path

import torch

from framework.lightning.config import ActorLearnerLightningConfig, LearnerOptimizerConfig
from framework.lightning.trajectory_module import TrajectoryLightningModule


class _DebugAgent:
    def __init__(self) -> None:
        self.trainable_module = torch.nn.Linear(1, 1)
        self.dump_calls: list[tuple[str, int]] = []

    def sample_counterfactual_trajectories_from_replay_batch(
        self,
        replay,
        *,
        num_candidates: int,
        candidate_select: str = "topk",
    ):
        del replay, candidate_select
        return {
            "traj_xyyaw": torch.zeros((1, num_candidates, 4, 3), dtype=torch.float32),
            "log_probs": torch.full((1, num_candidates), -0.25, dtype=torch.float32),
        }

    def dump_counterfactual_debug_from_replay_batch(
        self,
        replays,
        traj_xyyaw,
        candidate_scores,
        *,
        out_dir: str,
        step_tag: str,
        top_k: int,
    ) -> None:
        del replays, traj_xyyaw, candidate_scores
        self.dump_calls.append((step_tag, top_k))
        Path(out_dir, f"{step_tag}.txt").write_text("debug", encoding="utf-8")


def test_grpo_debug_dump_runs_without_aux_loss_and_zero_max_batches_is_unlimited(
    monkeypatch,
    tmp_path: Path,
) -> None:
    agent = _DebugAgent()
    learner_config = ActorLearnerLightningConfig(
        algo_kind="ppo",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4),
        eta=1.0,
        clip_eps=0.2,
        grpo_enabled=False,
        grpo_coef=0.0,
        grpo_num_candidates=3,
        grpo_candidate_select="topk",
        grpo_debug_visualize=True,
        grpo_debug_dir=str(tmp_path),
        grpo_debug_max_batches=0,
        grpo_debug_top_k=2,
    )
    module = TrajectoryLightningModule(
        agent=agent,
        learner_config=learner_config,
        value_net=torch.nn.Linear(1, 1),
    )

    monkeypatch.setattr(
        "framework.lightning.trajectory_module.score_counterfactual_trajectories",
        lambda *args, **kwargs: torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32),
    )

    base_loss = torch.tensor(2.0, dtype=torch.float32)
    base_metrics = {"loss_pi": torch.tensor(1.0, dtype=torch.float32)}
    replay = [{"scene_id": 146, "frame_idx": 0, "sample_token": "tok"}]

    out_loss_0, out_metrics_0 = module._maybe_apply_grpo_loss(
        replay=replay,
        device=torch.device("cpu"),
        batch_idx=0,
        loss=base_loss,
        metrics=base_metrics,
    )
    out_loss_1, out_metrics_1 = module._maybe_apply_grpo_loss(
        replay=replay,
        device=torch.device("cpu"),
        batch_idx=1,
        loss=base_loss,
        metrics=base_metrics,
    )

    assert torch.equal(out_loss_0, base_loss)
    assert torch.equal(out_loss_1, base_loss)
    assert out_metrics_0 == base_metrics
    assert out_metrics_1 == base_metrics
    assert len(agent.dump_calls) == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == ["step000000_batch0000.txt", "step000000_batch0001.txt"]
