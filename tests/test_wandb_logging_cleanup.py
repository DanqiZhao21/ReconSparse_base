from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from framework.io.buffer import BufferPaths, ensure_buffer_layout
from framework.lightning import actor_learner_module, trajectory_module
from framework.lightning.actor_learner_module import ActorLearnerLightningModule
from framework.lightning.config import ActorLearnerLightningConfig, LearnerOptimizerConfig
from framework.lightning.trajectory_module import TrajectoryLightningModule


class _SavingAgent:
    trainable_module = torch.nn.Linear(1, 1)

    def save_checkpoint(self, path: str) -> None:
        torch.save({"ok": True}, path)


class _FakeWandb:
    def __init__(self) -> None:
        self.logged: list[dict] = []

    def log(self, payload: dict) -> None:
        self.logged.append(dict(payload))


def _learner_config(**overrides: object) -> ActorLearnerLightningConfig:
    values = {
        "algo_kind": "reinforcepp",
        "optimizer_config": LearnerOptimizerConfig(policy_lr=1.0e-4),
        "eta": 1.0,
        "clip_eps": 0.2,
        "inner_epochs": 1,
    }
    values.update(overrides)
    return ActorLearnerLightningConfig(**values)


def test_actor_learner_wandb_logs_clean_update_namespaces_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    shard_path = Path(paths.shards_dir) / "actor0_e0_v7_t1000_deadbeef.pt"
    torch.save({"replay": [{"ok": True}]}, shard_path)
    Path(paths.version_file).write_text("7", encoding="utf-8")

    loaded = SimpleNamespace(
        num_samples=3,
        reward_sum=6.0,
        reward_count=3,
        done_sum=1.0,
        done_count=3,
        reward_summary={
            "step_count": 3,
            "positive_reward_sum": 9.0,
            "gated_positive_reward_sum": 4.5,
            "cost_reward_sum": -1.5,
            "safety_gate_active_count": 1,
            "collision_gate_count": 0,
            "severe_tracking_lateral_gate_count": 1,
            "severe_tracking_yaw_gate_count": 0,
            "terminal_failure_count": 1,
            "terminal_timeout_count": 0,
            "terminal_env_done_count": 1,
        },
        batch={
            "ret": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "adv": torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32),
        },
    )
    datamodule = SimpleNamespace(
        current_selected=[str(shard_path)],
        current_loaded=loaded,
        current_wait_shards_s=11.0,
        current_load_shards_s=2.0,
        current_prepare_batch_s=3.0,
    )
    module = ActorLearnerLightningModule(
        agent=_SavingAgent(),
        learner_config=_learner_config(),
        value_net=None,
        paths=paths,
        stage_fn=lambda *_args, **_kwargs: None,
        ddp_enabled=False,
        dist_module=None,
        rank=0,
        wandb_enabled=True,
    )
    module._trainer = SimpleNamespace(datamodule=datamodule, should_stop=False)
    module._latest_epoch_had_data = True
    module._update_train_t0 = 100.0
    module._is_update_end = lambda: True  # type: ignore[method-assign]
    module._update_index = lambda: 4  # type: ignore[method-assign]
    module.latest_metrics = {
        "loss_pi": 0.25,
        "approx_kl": 0.01,
        "clip_frac": 0.2,
    }
    monkeypatch.setattr(actor_learner_module.time, "time", lambda: 150.0)
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(actor_learner_module, "wandb", fake_wandb)

    module.on_train_epoch_end()

    assert len(fake_wandb.logged) == 1
    payload = fake_wandb.logged[0]
    assert payload["progress/update"] == 4
    assert payload["progress/weights_version"] == 8
    assert payload["progress/global_sample_step"] == 3
    assert payload["data/samples"] == 3
    assert payload["data/shards"] == 1
    assert payload["data/done_rate"] == 1.0 / 3.0
    assert payload["time/collect_s"] == 11.0
    assert payload["time/train_s"] == 50.0
    assert payload["time/update_s"] == 61.0
    assert payload["optim/loss_pi"] == 0.25
    assert payload["optim/approx_kl"] == 0.01
    assert payload["reward/mean"] == 2.0
    assert payload["reward/positive_mean"] == 3.0
    assert payload["reward_gate/safety_rate"] == 1.0 / 3.0
    assert payload["terminal/failure_rate"] == 1.0 / 3.0
    assert payload["batch/ret_mean"] == 2.0
    assert "train_update/reward_mean" not in payload
    assert "reward_mean" not in payload
    assert "global_step" not in payload


def test_minibatch_wandb_logging_is_disabled_by_default(monkeypatch) -> None:
    module = TrajectoryLightningModule(
        agent=_SavingAgent(),
        learner_config=_learner_config(),
        value_net=None,
    )
    module.wandb_enabled = True
    module._trainer = SimpleNamespace(global_step=0)
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(trajectory_module, "wandb", fake_wandb)

    module._maybe_log_train_seen_samples(
        metrics={"loss_pi": torch.tensor(0.25)},
        adv=torch.tensor([1.0, 2.0], dtype=torch.float32),
        ret=torch.tensor([3.0, 4.0], dtype=torch.float32),
        batch_size=2,
    )

    assert fake_wandb.logged == []
