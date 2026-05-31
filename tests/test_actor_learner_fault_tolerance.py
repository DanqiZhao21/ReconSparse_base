from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from framework.io import buffer as buffer_io
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


def test_async_collection_marks_stale_actor_heartbeat_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    for actor_id in [0, 1, 2, 3, 5]:
        for shard_idx in range(4):
            name = f"actor{actor_id}_e0_v34_t{1500 + actor_id * 10 + shard_idx}_stale{shard_idx}.pt"
            _write_dummy_shard(Path(paths.shards_dir) / name)

    Path(paths.version_file).write_text("34", encoding="utf-8")
    heartbeat_path = buffer_io.write_actor_heartbeat(paths, actor_id=4, message="collecting")
    old_ts = 100.0
    Path(heartbeat_path).touch()

    os.utime(heartbeat_path, (old_ts, old_ts))

    learner_config = ActorLearnerLightningConfig(
        **{
            **_build_learner_config().__dict__,
            "actor_heartbeat_timeout_s": 10.0,
        }
    )
    learner = ActorLearnerUpdateDataModule(
        paths=paths,
        agent=_DummyAgent(),
        learner_config=learner_config,
        device=torch.device("cpu"),
        value_net=None,
        ddp_enabled=False,
        dist_module=None,
        world_size=1,
        rank=0,
        stage_fn=lambda *_args, **_kwargs: None,
        start_version=34,
    )

    monkeypatch.setattr("framework.lightning.actor_learner_datamodule.time.time", lambda: 200.0)
    monkeypatch.setattr(
        "framework.lightning.actor_learner_datamodule.time.sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("learner should not wait after stale actor is failed")),
    )

    selected = learner._select_shards()

    assert len(selected) == 20
    assert (Path(paths.actors_dir) / "actor4.failed").exists()


def test_async_collection_waits_for_full_target_after_timeout_without_permanent_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    for actor_id in [0, 1, 3, 4, 5]:
        for shard_idx in range(4):
            name = f"actor{actor_id}_e0_v34_t{2000 + actor_id * 10 + shard_idx}_efgh{shard_idx}.pt"
            _write_dummy_shard(Path(paths.shards_dir) / name)

    learner_config = ActorLearnerLightningConfig(
        **{
            **_build_learner_config().__dict__,
            "shard_collect_timeout_s": 1.0,
        }
    )
    Path(paths.version_file).write_text("34", encoding="utf-8")

    learner = ActorLearnerUpdateDataModule(
        paths=paths,
        agent=_DummyAgent(),
        learner_config=learner_config,
        device=torch.device("cpu"),
        value_net=None,
        ddp_enabled=False,
        dist_module=None,
        world_size=1,
        rank=0,
        stage_fn=lambda *_args, **_kwargs: None,
        start_version=34,
    )

    fake_time = {"value": 0.0}

    def _fake_time() -> float:
        return float(fake_time["value"])

    sleep_count = {"value": 0}

    def _fake_sleep(seconds: float) -> None:
        sleep_count["value"] += 1
        fake_time["value"] += max(0.51, float(seconds) if seconds > 0 else 0.51)
        if int(sleep_count["value"]) >= 3 and not any(Path(paths.shards_dir).glob("actor2_*.pt")):
            for shard_idx in range(4):
                name = f"actor2_e0_v34_t{2500 + shard_idx}_late{shard_idx}.pt"
                _write_dummy_shard(Path(paths.shards_dir) / name)

    monkeypatch.setattr("framework.lightning.actor_learner_datamodule.time.time", _fake_time)
    monkeypatch.setattr("framework.lightning.actor_learner_datamodule.time.sleep", _fake_sleep)

    selected = learner._select_shards()

    assert len(selected) == 24
    assert sum("actor2_" in item for item in selected) == 4
    assert list(Path(paths.actors_dir).glob("*.failed")) == []


def test_async_collection_can_temporarily_shrink_after_timeout_when_partial_updates_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    for actor_id in [0, 1, 3, 4, 5]:
        for shard_idx in range(4):
            name = f"actor{actor_id}_e0_v34_t{3000 + actor_id * 10 + shard_idx}_ijkl{shard_idx}.pt"
            _write_dummy_shard(Path(paths.shards_dir) / name)

    learner_config = ActorLearnerLightningConfig(
        **{
            **_build_learner_config().__dict__,
            "shard_collect_timeout_s": 1.0,
            "allow_partial_updates_after_timeout": True,
        }
    )
    Path(paths.version_file).write_text("34", encoding="utf-8")

    learner = ActorLearnerUpdateDataModule(
        paths=paths,
        agent=_DummyAgent(),
        learner_config=learner_config,
        device=torch.device("cpu"),
        value_net=None,
        ddp_enabled=False,
        dist_module=None,
        world_size=1,
        rank=0,
        stage_fn=lambda *_args, **_kwargs: None,
        start_version=34,
    )

    fake_time = {"value": 0.0}

    def _fake_time() -> float:
        return float(fake_time["value"])

    def _fake_sleep(seconds: float) -> None:
        fake_time["value"] += max(0.51, float(seconds) if seconds > 0 else 0.51)

    monkeypatch.setattr("framework.lightning.actor_learner_datamodule.time.time", _fake_time)
    monkeypatch.setattr("framework.lightning.actor_learner_datamodule.time.sleep", _fake_sleep)

    selected_first = learner._select_shards()
    assert len(selected_first) == 20

    for shard_idx in range(4):
        name = f"actor2_e0_v34_t{4000 + shard_idx}_mnop{shard_idx}.pt"
        _write_dummy_shard(Path(paths.shards_dir) / name)

    fake_time["value"] = 0.0
    selected_second = learner._select_shards()

    assert len(selected_second) == 24
    assert sum("actor2_" in item for item in selected_second) == 4


def test_resolve_sample_token_uses_scene_and_frame_assets() -> None:
    token = resolve_sample_token(scene_id=146, frame_idx=0)

    assert isinstance(token, str)
    assert token
