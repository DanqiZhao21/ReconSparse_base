from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from framework.io.buffer import BufferPaths, ensure_buffer_layout
from framework.lightning.actor_learner_module import ActorLearnerLightningModule
from framework.lightning.config import ActorLearnerLightningConfig, LearnerOptimizerConfig


class _FailingPublishAgent:
    trainable_module = torch.nn.Linear(1, 1)

    def save_checkpoint(self, _path: str) -> None:
        raise RuntimeError("simulated publish failure")


class _SuccessfulPublishAgent:
    trainable_module = torch.nn.Linear(1, 1)

    def save_checkpoint(self, path: str) -> None:
        torch.save({"state_dict": {"ok": torch.tensor(1.0)}}, path)


def _build_module(paths: BufferPaths, *, agent=None, debug_retain_versions: int = 0) -> ActorLearnerLightningModule:
    learner_config = ActorLearnerLightningConfig(
        algo_kind="reinforcepp",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4),
        eta=1.0,
        clip_eps=0.2,
        inner_epochs=1,
        debug_retain_versions=int(debug_retain_versions),
        debug_retain_ckpts=bool(debug_retain_versions > 0),
        debug_retain_shards=bool(debug_retain_versions > 0),
    )
    return ActorLearnerLightningModule(
        agent=agent if agent is not None else _FailingPublishAgent(),
        learner_config=learner_config,
        value_net=None,
        paths=paths,
        stage_fn=lambda *_args, **_kwargs: None,
        ddp_enabled=False,
        dist_module=None,
        rank=0,
        wandb_enabled=False,
    )


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


def test_publish_retains_debug_checkpoint_and_selected_shards(tmp_path: Path) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)

    shard_path = Path(paths.shards_dir) / "actor0_e0_v1_t1000_deadbeef.pt"
    torch.save(
        {
            "old_logp": torch.zeros(1),
            "reward": torch.ones(1),
            "done": torch.ones(1),
            "replay": [{"ok": True}],
            "meta": {"weights_version": 1, "actor_id": 0, "num_steps": 1},
        },
        shard_path,
    )
    Path(paths.version_file).write_text("1", encoding="utf-8")

    loaded = SimpleNamespace(
        num_samples=1,
        reward_sum=1.0,
        reward_count=1,
        done_sum=1.0,
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

    module = _build_module(paths, agent=_SuccessfulPublishAgent(), debug_retain_versions=5)
    module._trainer = SimpleNamespace(datamodule=datamodule, should_stop=False)
    module._latest_epoch_had_data = True
    module._update_train_t0 = 0.0
    module._is_update_end = lambda: True  # type: ignore[method-assign]
    module._update_index = lambda: 0  # type: ignore[method-assign]

    module.on_train_epoch_end()

    assert Path(paths.latest_ckpt).exists()
    assert Path(paths.version_file).read_text(encoding="utf-8").strip() == "2"
    assert (Path(paths.weights_dir) / "history" / "version_000002.ckpt").exists()
    manifest_path = (
        Path(paths.root)
        / "debug_retention"
        / "update_000000_from_v000001_to_v000002"
        / "manifest.json"
    )
    assert manifest_path.exists()
    assert (
        manifest_path.parent
        / "shards"
        / shard_path.name
    ).exists()
    assert not shard_path.exists()
    assert (Path(paths.consumed_dir) / shard_path.name).exists()
