from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from framework.io.buffer import BufferPaths, ensure_buffer_layout, stop_requested
from framework.lightning.actor_learner_module import ActorLearnerLightningModule
from framework.lightning.config import ActorLearnerLightningConfig, LearnerOptimizerConfig


class _FailingPublishAgent:
    trainable_module = torch.nn.Linear(1, 1)

    def save_checkpoint(self, _path: str) -> None:
        raise RuntimeError("simulated publish failure")


class _RecordingPublishAgent:
    trainable_module = torch.nn.Linear(1, 1)

    def __init__(self) -> None:
        self.saved_paths: list[str] = []

    def save_checkpoint(self, path: str) -> None:
        self.saved_paths.append(str(path))
        torch.save({"ok": True}, path)


def _build_module(paths: BufferPaths) -> ActorLearnerLightningModule:
    learner_config = ActorLearnerLightningConfig(
        algo_kind="reinforcepp",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4),
        eta=1.0,
        clip_eps=0.2,
        inner_epochs=1,
    )
    return ActorLearnerLightningModule(
        agent=_FailingPublishAgent(),
        learner_config=learner_config,
        value_net=None,
        paths=paths,
        stage_fn=lambda *_args, **_kwargs: None,
        ddp_enabled=False,
        dist_module=None,
        rank=0,
        wandb_enabled=False,
    )


def _build_success_module(paths: BufferPaths, *, max_updates: int) -> tuple[ActorLearnerLightningModule, _RecordingPublishAgent]:
    learner_config = ActorLearnerLightningConfig(
        algo_kind="reinforcepp",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4),
        eta=1.0,
        clip_eps=0.2,
        inner_epochs=1,
        max_updates=int(max_updates),
    )
    agent = _RecordingPublishAgent()
    module = ActorLearnerLightningModule(
        agent=agent,
        learner_config=learner_config,
        value_net=None,
        paths=paths,
        stage_fn=lambda *_args, **_kwargs: None,
        ddp_enabled=False,
        dist_module=None,
        rank=0,
        wandb_enabled=False,
    )
    module.aggregated_update_metrics = lambda: {}  # type: ignore[method-assign]
    module.aggregated_update_timing = lambda: {}  # type: ignore[method-assign]
    return module, agent


def test_publish_failure_does_not_consume_shards_or_bump_version(tmp_path: Path) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)

    shard_path = Path(paths.shards_dir) / "actor0_e0_v7_t1000_deadbeef.pt"
    torch.save({"replay": [{"ok": True}]}, shard_path)
    Path(paths.version_file).write_text("7", encoding="utf-8")

    loaded = SimpleNamespace(
        num_samples=1,
        reward_sum=1.0,
        reward_count=1,
        done_sum=0.0,
        done_count=1,
        reward_summary={},
        batch={
            "ret": torch.tensor([1.0], dtype=torch.float32),
            "adv": torch.tensor([0.5], dtype=torch.float32),
        },
    )
    datamodule = SimpleNamespace(
        current_selected=[str(shard_path)],
        current_loaded=loaded,
    )

    module = _build_module(paths)
    module._trainer = SimpleNamespace(datamodule=datamodule, should_stop=False)
    module._latest_epoch_had_data = True
    module._update_train_t0 = 0.0
    module._is_update_end = lambda: True  # type: ignore[method-assign]
    module._update_index = lambda: 0  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated publish failure"):
        module.on_train_epoch_end()

    assert shard_path.exists()
    assert not (Path(paths.consumed_dir) / shard_path.name).exists()
    assert Path(paths.version_file).read_text(encoding="utf-8").strip() == "7"


def test_final_update_requests_actor_stop_before_trainer_exit(tmp_path: Path) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)

    shard_path = Path(paths.shards_dir) / "actor0_e0_v2_t1000_deadbeef.pt"
    torch.save({"replay": [{"ok": True}]}, shard_path)
    Path(paths.version_file).write_text("2", encoding="utf-8")
    training_lock = Path(paths.root) / "TRAINING_LOCK"
    training_lock.write_text("training\n", encoding="utf-8")

    loaded = SimpleNamespace(
        num_samples=1,
        reward_sum=1.0,
        reward_count=1,
        done_sum=0.0,
        done_count=1,
        reward_summary={},
        batch={
            "ret": torch.tensor([1.0], dtype=torch.float32),
            "adv": torch.tensor([0.5], dtype=torch.float32),
        },
    )
    datamodule = SimpleNamespace(
        current_selected=[str(shard_path)],
        current_loaded=loaded,
    )

    module, agent = _build_success_module(paths, max_updates=2)
    module._trainer = SimpleNamespace(datamodule=datamodule, should_stop=False)
    module._latest_epoch_had_data = True
    module._update_train_t0 = 0.0
    module._is_update_end = lambda: True  # type: ignore[method-assign]
    module._update_index = lambda: 1  # type: ignore[method-assign]

    module.on_train_epoch_end()

    assert agent.saved_paths == [paths.latest_ckpt]
    assert stop_requested(paths) is True
    assert not training_lock.exists()
    assert Path(paths.version_file).read_text(encoding="utf-8").strip() == "3"
