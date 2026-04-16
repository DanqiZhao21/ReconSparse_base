from __future__ import annotations

from pathlib import Path

import pytest
import torch

from framework.io.buffer import BufferPaths, ensure_buffer_layout, write_actor_failure
from framework.lightning.actor_learner_datamodule import ActorLearnerUpdateDataModule
from framework.lightning.config import ActorLearnerLightningConfig, LearnerOptimizerConfig
from framework.utils.nuscenes_token import resolve_sample_token


class _DummyAgent:
    trainable_module = torch.nn.Identity()


def _write_dummy_shard(path: Path) -> None:
    torch.save({"replay": [{"ok": True}]}, path)


def _build_learner_config() -> ActorLearnerLightningConfig:
    return ActorLearnerLightningConfig(
        algo_kind="ppo",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4),
        eta=1.0,
        clip_eps=0.2,
        mode="async",
        num_actors=6,
        shards_per_update=24,
        max_inflight_per_actor=4,
        poll_s=0.0,
    )


def test_async_collection_target_shrinks_after_actor_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    for actor_id in [0, 1, 2, 3, 5]:
        for shard_idx in range(4):
            name = f"actor{actor_id}_e{shard_idx % 2}_v34_t{1000 + actor_id * 10 + shard_idx}_abcd{shard_idx}.pt"
            _write_dummy_shard(Path(paths.shards_dir) / name)

    write_actor_failure(paths, actor_id=4, message="actor crashed during rollout")
    Path(paths.version_file).write_text("34", encoding="utf-8")

    learner = ActorLearnerUpdateDataModule(
        paths=paths,
        agent=_DummyAgent(),
        learner_config=_build_learner_config(),
        device=torch.device("cpu"),
        value_net=None,
        ddp_enabled=False,
        dist_module=None,
        world_size=1,
        rank=0,
        stage_fn=lambda *_args, **_kwargs: None,
        start_version=34,
    )

    def _fail_if_waiting(_seconds: float) -> None:
        raise AssertionError("collection loop kept waiting instead of shrinking the async shard target")

    monkeypatch.setattr("framework.lightning.actor_learner_datamodule.time.sleep", _fail_if_waiting)

    selected = learner._select_shards()

    assert len(selected) == 20


def test_resolve_sample_token_uses_scene_and_frame_assets() -> None:
    token = resolve_sample_token(scene_id=146, frame_idx=0)

    assert isinstance(token, str)
    assert token
