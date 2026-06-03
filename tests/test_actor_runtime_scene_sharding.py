from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from framework.io.buffer import BufferPaths, ensure_buffer_layout
from framework.runner.actor_runtime import _actor_main_impl, _shard_num_steps


class _StopVectorActor(RuntimeError):
    pass


class _StopAfterShard(RuntimeError):
    pass


class _DummyAgent:
    pass


class _ClosableEnv:
    def __init__(self) -> None:
        self.close_count = 0
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        return {"obs": self.reset_count}, {"reset_count": self.reset_count}

    def close(self) -> None:
        self.close_count += 1


def _single_env_cfg(paths: BufferPaths, *, backend: str) -> dict:
    return {
        "env": {
            "backend": backend,
        },
        "train": {
            "eta": 1.0,
            "mode_idx": -1,
            "policy_mode_select": "sample",
            "actor_learner": {
                "mode": "async",
                "buffer_dir": str(paths.root),
                "actor_horizon": 4,
                "poll_interval_s": 0.0,
                "max_inflight_per_actor": 4,
                "num_envs_per_actor": 1,
                "vec_env_mode": "serial",
                "num_actors": 1,
            },
        },
        "agent": {
            "type": "dummy",
        },
    }


def _fake_shard() -> dict:
    return {
        "old_logp": torch.zeros(1),
        "reward": torch.zeros(1),
        "done": torch.zeros(1),
        "terminated": torch.zeros(1),
        "truncated": torch.zeros(1),
        "done_last": torch.tensor(0.0),
        "terminated_last": torch.tensor(0.0),
        "replay": [{}],
        "meta": {"timing": {}},
    }


def test_shard_num_steps_prefers_actual_meta_count() -> None:
    shard = {
        "reward": torch.zeros(32),
        "replay": [{} for _ in range(32)],
        "meta": {"num_steps": 8},
    }

    assert _shard_num_steps(shard, default=32) == 8


def test_vector_actor_env_builds_receive_total_actors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)

    cfg = {
        "train": {
            "eta": 1.0,
            "mode_idx": -1,
            "policy_mode_select": "sample",
            "actor_learner": {
                "mode": "async",
                "buffer_dir": str(paths.root),
                "actor_horizon": 4,
                "poll_interval_s": 0.0,
                "max_inflight_per_actor": 2,
                "num_envs_per_actor": 2,
                "vec_env_mode": "serial",
                "num_actors": 6,
            },
        },
        "agent": {
            "type": "dummy",
        },
    }

    calls: list[dict[str, int | None]] = []

    def _fake_build_actor_env(_cfg, *, cuda, actor_id, worker_id=None, total_actors=1):
        calls.append(
            {
                "cuda": int(cuda),
                "actor_id": int(actor_id),
                "worker_id": None if worker_id is None else int(worker_id),
                "total_actors": int(total_actors),
            }
        )
        return object()

    class _FakeVecEnv:
        def __init__(self, env_fns):
            self._envs = [fn() for fn in env_fns]

        def reset(self):
            raise _StopVectorActor("stop after env construction")

    fake_env_wrapper = types.ModuleType("framework.env_wrapper")
    fake_env_wrapper.SerialVecEnv = _FakeVecEnv
    fake_env_wrapper.SubprocVecEnv = _FakeVecEnv

    monkeypatch.setitem(sys.modules, "framework.env_wrapper", fake_env_wrapper)
    monkeypatch.setattr("framework.runner.actor_runtime.build_agent", lambda *_args, **_kwargs: _DummyAgent())
    monkeypatch.setattr("framework.runner.actor_runtime.build_actor_env", _fake_build_actor_env)
    monkeypatch.setattr("framework.runner.actor_runtime.torch.cuda.is_available", lambda: False)

    with pytest.raises(_StopVectorActor):
        _actor_main_impl(
            cfg,
            actor_id=2,
            gpu_id=None,
            total_actors=6,
            paths=paths,
        )

    assert len(calls) == 2
    assert {int(call["total_actors"]) for call in calls} == {6}
    assert {int(call["worker_id"]) for call in calls} == {2000, 2001}


def test_actor_heartbeat_reports_initial_weight_loading_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    Path(paths.version_file).write_text("3", encoding="utf-8")
    Path(paths.latest_ckpt).write_bytes(b"checkpoint")
    cfg = _single_env_cfg(paths, backend="recon")
    beats: list[tuple[str, int | None, bool]] = []

    class _FakeHeartbeat:
        def __init__(self, **_kwargs):
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def beat(self, phase=None, step=None, *, force=False) -> None:
            beats.append(("" if phase is None else str(phase), None if step is None else int(step), bool(force)))

    class _Agent:
        def load_checkpoint(self, path: str) -> None:
            assert path == paths.latest_ckpt

    def _stop_after_weight_load(*_args, **_kwargs):
        raise _StopAfterShard("stop after initial weight load")

    monkeypatch.setattr("framework.runner.actor_runtime.PeriodicActorHeartbeat", _FakeHeartbeat)
    monkeypatch.setattr("framework.runner.actor_runtime.build_agent", lambda *_args, **_kwargs: _Agent())
    monkeypatch.setattr("framework.runner.actor_runtime.build_actor_env", _stop_after_weight_load)
    monkeypatch.setattr("framework.runner.actor_runtime.torch.cuda.is_available", lambda: False)

    with pytest.raises(_StopAfterShard):
        _actor_main_impl(
            cfg,
            actor_id=0,
            gpu_id=None,
            total_actors=1,
            paths=paths,
        )

    assert ("loading_weights", 3, True) in beats
    assert ("loaded_weights", 3, True) in beats


def test_hugsim_single_env_actor_closes_session_between_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    env = _ClosableEnv()
    cfg = _single_env_cfg(paths, backend="hugsim_ori")

    def _fake_collect_single_env_shard(**_kwargs):
        return _fake_shard(), {"obs": "next"}, {"info": "next"}

    def _stop_after_save(*_args, **_kwargs):
        raise _StopAfterShard("stop after close/save boundary")

    monkeypatch.setattr("framework.runner.actor_runtime.build_agent", lambda *_args, **_kwargs: _DummyAgent())
    monkeypatch.setattr("framework.runner.actor_runtime.build_actor_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr("framework.runner.actor_runtime.collect_single_env_shard", _fake_collect_single_env_shard)
    monkeypatch.setattr("framework.runner.actor_runtime.atomic_torch_save", _stop_after_save)
    monkeypatch.setattr("framework.runner.actor_runtime.torch.cuda.is_available", lambda: False)

    with pytest.raises(_StopAfterShard):
        _actor_main_impl(
            cfg,
            actor_id=0,
            gpu_id=None,
            total_actors=1,
            paths=paths,
        )

    assert env.reset_count == 1
    assert env.close_count == 1


def test_recon_single_env_actor_keeps_session_between_shards_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    env = _ClosableEnv()
    cfg = _single_env_cfg(paths, backend="recon")

    def _fake_collect_single_env_shard(**_kwargs):
        return _fake_shard(), {"obs": "next"}, {"info": "next"}

    def _stop_after_save(*_args, **_kwargs):
        raise _StopAfterShard("stop after save boundary")

    monkeypatch.setattr("framework.runner.actor_runtime.build_agent", lambda *_args, **_kwargs: _DummyAgent())
    monkeypatch.setattr("framework.runner.actor_runtime.build_actor_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr("framework.runner.actor_runtime.collect_single_env_shard", _fake_collect_single_env_shard)
    monkeypatch.setattr("framework.runner.actor_runtime.atomic_torch_save", _stop_after_save)
    monkeypatch.setattr("framework.runner.actor_runtime.torch.cuda.is_available", lambda: False)

    with pytest.raises(_StopAfterShard):
        _actor_main_impl(
            cfg,
            actor_id=0,
            gpu_id=None,
            total_actors=1,
            paths=paths,
        )

    assert env.reset_count == 1
    assert env.close_count == 0


def test_hugsim_single_env_actor_closes_session_when_collection_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    env = _ClosableEnv()
    cfg = _single_env_cfg(paths, backend="hugsim_ori")

    def _fail_collect_single_env_shard(**_kwargs):
        raise _StopAfterShard("stop during collection")

    monkeypatch.setattr("framework.runner.actor_runtime.build_agent", lambda *_args, **_kwargs: _DummyAgent())
    monkeypatch.setattr("framework.runner.actor_runtime.build_actor_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr("framework.runner.actor_runtime.collect_single_env_shard", _fail_collect_single_env_shard)
    monkeypatch.setattr("framework.runner.actor_runtime.torch.cuda.is_available", lambda: False)

    with pytest.raises(_StopAfterShard):
        _actor_main_impl(
            cfg,
            actor_id=0,
            gpu_id=None,
            total_actors=1,
            paths=paths,
        )

    assert env.reset_count == 1
    assert env.close_count == 1
