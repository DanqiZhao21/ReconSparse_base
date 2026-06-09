from __future__ import annotations

from pathlib import Path
from typing import Any

from framework.io.buffer import BufferPaths, list_failed_actor_ids, write_actor_failure
from framework.runner.orchestrator import _launch_worker, _terminate_process, orchestrator_main


class _FakeProc:
    sleep_calls = 0
    actor_launches: dict[int, int] = {}
    terminated_actor_ids: list[int] = []

    def __init__(self, cmd: list[str], env: dict[str, str] | None = None, **_kwargs: Any) -> None:
        self.cmd = list(cmd)
        self.env = dict(env or {})
        self.role = self.cmd[self.cmd.index("--role") + 1]
        self.actor_id: int | None = None
        self.terminated = False
        if self.role == "actor":
            self.actor_id = int(self.cmd[self.cmd.index("--actor-id") + 1])
            self.actor_launches[self.actor_id] = self.actor_launches.get(self.actor_id, 0) + 1

    def poll(self) -> int | None:
        if self.role == "learner" and self.sleep_calls >= 2:
            return 0
        return None

    def terminate(self) -> None:
        self.terminated = True
        if self.actor_id is not None:
            self.terminated_actor_ids.append(int(self.actor_id))

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        return 0


def test_orchestrator_restarts_actor_marked_failed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train: {}\n", encoding="utf-8")
    _FakeProc.sleep_calls = 0
    _FakeProc.actor_launches = {}
    _FakeProc.terminated_actor_ids = []

    def _fake_sleep(_seconds: float) -> None:
        if _FakeProc.sleep_calls == 0:
            write_actor_failure(paths, 0, message="learner observed shard stall")
        _FakeProc.sleep_calls += 1

    monkeypatch.setattr("framework.runner.orchestrator.subprocess.Popen", _FakeProc)
    monkeypatch.setattr("framework.runner.orchestrator.time.sleep", _fake_sleep)
    monkeypatch.setattr("framework.runner.orchestrator.warmup_gsplat_cuda", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("framework.runner.orchestrator.build_launch_env", lambda **_kwargs: {})
    monkeypatch.setattr("framework.runner.orchestrator.stage", lambda *_args, **_kwargs: None)

    orchestrator_main(
        {
            "agent": {"type": "ddv2"},
            "train": {
                "actor_learner": {
                    "buffer_dir": str(paths.root),
                    "num_actors": 2,
                    "actor_gpu_ids": [0, 1],
                    "learner_gpu_ids": [0],
                    "restart_failed_actors": True,
                    "max_actor_restarts": 1,
                }
            },
        },
        config_path=str(config_path),
    )

    assert _FakeProc.actor_launches[0] == 2
    assert _FakeProc.actor_launches[1] == 1
    assert _FakeProc.terminated_actor_ids[0] == 0
    assert list_failed_actor_ids(paths) == []


def test_launch_worker_installs_parent_death_preexec_on_posix(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakePopen:
        def __init__(self, cmd: list[str], **kwargs: Any) -> None:
            captured["cmd"] = list(cmd)
            captured["kwargs"] = dict(kwargs)

    monkeypatch.setattr("framework.runner.orchestrator.subprocess.Popen", FakePopen)
    monkeypatch.setattr("framework.runner.orchestrator.os.name", "posix")

    _launch_worker(["python", "entry.py"], env={"A": "1"})

    assert captured["cmd"] == ["python", "entry.py"]
    assert captured["kwargs"]["env"] == {"A": "1"}
    assert callable(captured["kwargs"]["preexec_fn"])


def test_terminate_process_signals_process_group_before_kill(monkeypatch: Any) -> None:
    signals: list[tuple[int, int]] = []

    class FakeProc:
        pid = 1234

        def __init__(self) -> None:
            self.wait_calls = 0
            self.terminated = False
            self.killed = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise TimeoutError("still running")
            return 0

    def fake_killpg(pid: int, sig: int) -> None:
        signals.append((int(pid), int(sig)))

    monkeypatch.setattr("framework.runner.orchestrator.os.killpg", fake_killpg)

    proc = FakeProc()
    _terminate_process(proc, timeout_s=1.0)

    assert signals == [(1234, 15), (1234, 9)]
    assert not proc.terminated
    assert not proc.killed
