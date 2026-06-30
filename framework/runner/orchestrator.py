from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from framework.io.buffer import (
    BufferPaths,
    actor_failure_flag_path,
    clear_actor_failure,
    ensure_buffer_layout,
    list_failed_actor_ids,
    write_actor_failure,
)
from framework.runner.config_normalization import resolve_actor_gpu_ids, resolve_learner_gpu_ids
from framework.runner.launch_env import build_launch_env
from framework.runner.logging import stage
from framework.utils.gsplat_warmup import warmup_gsplat_cuda
from framework.utils.repo_paths import REPO_ROOT


def _write_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _set_parent_death_signal(*, parent_pid: int, sig: int = signal.SIGTERM) -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = getattr(libc, "prctl")
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        if int(prctl(1, int(sig), 0, 0, 0)) != 0:
            return
        if os.getppid() != int(parent_pid):
            os.kill(os.getpid(), int(sig))
    except Exception:
        return


def _make_worker_preexec(parent_pid: int) -> Any:
    def _worker_preexec() -> None:
        try:
            os.setpgrp()
        except Exception:
            pass
        _set_parent_death_signal(parent_pid=int(parent_pid), sig=signal.SIGTERM)

    return _worker_preexec


def _launch_worker(cmd: List[str], *, env: Dict[str, str]) -> subprocess.Popen:
    kwargs: Dict[str, Any] = {"env": env}
    if os.name == "posix":
        kwargs["preexec_fn"] = _make_worker_preexec(os.getpid())
    return subprocess.Popen(cmd, **kwargs)


@dataclass(frozen=True)
class LearnerLaunchSpec:
    rank: int
    local_rank: int
    cmd: List[str]
    env: Dict[str, str]


def build_learner_launch_specs(
    *,
    learner_gpu_ids: List[int],
    base_env: Dict[str, str],
    entry: str,
    config_path: str,
    python_executable: str = "python",
) -> List[LearnerLaunchSpec]:
    gpu_ids = [int(gpu_id) for gpu_id in learner_gpu_ids]
    if len(gpu_ids) == 0:
        gpu_ids = [0]
    world_size = len(gpu_ids)
    specs: List[LearnerLaunchSpec] = []
    for rank, gpu_id in enumerate(gpu_ids):
        env = dict(base_env)
        env["RANK"] = str(int(rank))
        env["WORLD_SIZE"] = str(int(world_size))
        env["LOCAL_RANK"] = str(int(gpu_id))
        if world_size > 1:
            env.setdefault("MASTER_ADDR", "127.0.0.1")
            env.setdefault("MASTER_PORT", "29500")
        cmd = [python_executable, entry, "--config", str(config_path), "--role", "learner"]
        specs.append(
            LearnerLaunchSpec(
                rank=int(rank),
                local_rank=int(gpu_id),
                cmd=cmd,
                env=env,
            )
        )
    return specs


def _build_actor_launch(
    *,
    python_executable: str,
    entry: str,
    config_path: str,
    actor_id: int,
    gpu_id: int,
    num_actors: int,
    base_env: Dict[str, str],
) -> tuple[List[str], Dict[str, str]]:
    actor_cmd = [
        python_executable,
        entry,
        "--config",
        str(config_path),
        "--role",
        "actor",
        "--actor-id",
        str(int(actor_id)),
        "--gpu-id",
        str(int(gpu_id)),
        "--num-actors",
        str(int(num_actors)),
    ]
    actor_env = dict(base_env)
    actor_env.setdefault("LOCAL_RANK", str(int(gpu_id if gpu_id >= 0 else 0)))
    return actor_cmd, actor_env


def _terminate_process(proc: subprocess.Popen, *, timeout_s: float) -> None:
    try:
        if proc.poll() is not None:
            return
    except Exception:
        pass

    pid = getattr(proc, "pid", None)
    signaled_group = False
    try:
        if os.name == "posix" and pid is not None:
            try:
                os.killpg(int(pid), signal.SIGTERM)
                signaled_group = True
            except Exception:
                proc.terminate()
        else:
            proc.terminate()
        proc.wait(timeout=float(timeout_s))
        return
    except Exception:
        pass
    try:
        if os.name == "posix" and pid is not None:
            try:
                os.killpg(int(pid), signal.SIGKILL)
            except Exception:
                proc.kill()
        elif not signaled_group:
            proc.kill()
        proc.wait(timeout=max(1.0, min(5.0, float(timeout_s))))
    except Exception:
        pass


def orchestrator_main(cfg: Dict[str, Any], *, config_path: str | None = None) -> None:
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}
    agent_cfg = cfg.get("agent", {}) or {}
    if config_path is None:
        raise ValueError("orchestrator_main requires config_path for subprocess launch")
    num_actors = int(al_cfg.get("num_actors", 4))
    actor_gpu_plan = resolve_actor_gpu_ids(al_cfg, num_actors=num_actors)
    learner_gpu_ids = resolve_learner_gpu_ids(al_cfg)
    learner_gpu_id = int(learner_gpu_ids[0])
    paths = BufferPaths(root=str(al_cfg.get("buffer_dir", "outputs/actor_learner")))
    restart_failed_actors = bool(al_cfg.get("restart_failed_actors", False))
    max_actor_restarts = int(al_cfg.get("max_actor_restarts", 1 if restart_failed_actors else 0) or 0)
    ensure_buffer_layout(paths)
    training_lock_file = os.path.join(paths.root, "TRAINING_LOCK")
    for path in [paths.stop_file, training_lock_file]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    py = sys.executable
    entry = os.path.join(REPO_ROOT, "script", "train_actor_learner_v2.py")
    launch_env = build_launch_env(agent_type=agent_cfg.get("type", "ddv2"))
    stage(f"[orchestrator] launch learner_gpus={learner_gpu_ids} num_actors={num_actors} actor_gpu_plan={actor_gpu_plan}")
    stage(
        f"[orchestrator] env CUDA_HOME={launch_env.get('CUDA_HOME', '')} "
        f"TORCH_EXTENSIONS_DIR={launch_env.get('TORCH_EXTENSIONS_DIR', '')}"
    )
    stage("[orchestrator] warmup gsplat CUDA extension before launching worker fan-out")
    warmup_gsplat_cuda(py, env=launch_env)

    learner_specs = build_learner_launch_specs(
        learner_gpu_ids=learner_gpu_ids,
        base_env=launch_env,
        entry=entry,
        config_path=str(config_path),
        python_executable=py,
    )
    learner_procs = [_launch_worker(spec.cmd, env=spec.env) for spec in learner_specs]
    actor_procs: List[subprocess.Popen] = []
    reported_actor_exits: set[int] = set()
    actor_restart_counts: Dict[int, int] = {}
    learner_exit: tuple[int, int] | None = None
    try:
        for aid in range(num_actors):
            gpu_id = int(actor_gpu_plan[aid]) if aid < len(actor_gpu_plan) else -1
            actor_cmd, actor_env = _build_actor_launch(
                python_executable=py,
                entry=entry,
                config_path=str(config_path),
                actor_id=int(aid),
                gpu_id=int(gpu_id),
                num_actors=int(num_actors),
                base_env=launch_env,
            )
            actor_procs.append(_launch_worker(actor_cmd, env=actor_env))
        while True:
            learner_exit: tuple[int, int] | None = None
            for spec, proc in zip(learner_specs, learner_procs):
                lret = proc.poll()
                if lret is not None:
                    learner_exit = (int(spec.rank), int(lret))
                    break
            if learner_exit is not None:
                stage(f"[orchestrator] learner rank={learner_exit[0]} exited code={learner_exit[1]}")
                break
            for i, proc in enumerate(actor_procs):
                pret = proc.poll()
                if pret is not None and pret != 0:
                    if int(i) in reported_actor_exits:
                        continue
                    reported_actor_exits.add(int(i))
                    if not os.path.exists(actor_failure_flag_path(paths, int(i))):
                        write_actor_failure(
                            paths,
                            int(i),
                            message=f"orchestrator observed actor exit code={int(pret)}",
                        )
                    stage(f"[orchestrator] actor{i} exited early code={pret}")
            if bool(restart_failed_actors) and int(max_actor_restarts) > 0:
                for actor_id in list_failed_actor_ids(paths):
                    aid = int(actor_id)
                    if aid < 0 or aid >= len(actor_procs):
                        continue
                    restart_count = int(actor_restart_counts.get(aid, 0))
                    if restart_count >= int(max_actor_restarts):
                        continue
                    gpu_id = int(actor_gpu_plan[aid]) if aid < len(actor_gpu_plan) else -1
                    stage(
                        f"[orchestrator] restarting actor{aid} after failure marker "
                        f"restart={restart_count + 1}/{int(max_actor_restarts)}"
                    )
                    _terminate_process(actor_procs[aid], timeout_s=10.0)
                    clear_actor_failure(paths, aid)
                    actor_cmd, actor_env = _build_actor_launch(
                        python_executable=py,
                        entry=entry,
                        config_path=str(config_path),
                        actor_id=aid,
                        gpu_id=int(gpu_id),
                        num_actors=int(num_actors),
                        base_env=launch_env,
                    )
                    actor_procs[aid] = _launch_worker(actor_cmd, env=actor_env)
                    actor_restart_counts[aid] = restart_count + 1
                    reported_actor_exits.discard(aid)
            time.sleep(2.0)
    finally:
        try:
            _write_text(paths.stop_file, "stop requested by orchestrator\n")
        except Exception:
            pass
        for proc in actor_procs:
            _terminate_process(proc, timeout_s=15.0)
        for proc in learner_procs:
            _terminate_process(proc, timeout_s=15.0)
    if learner_exit is not None and int(learner_exit[1]) != 0:
        raise RuntimeError(f"learner rank={learner_exit[0]} exited code={learner_exit[1]}")


__all__ = ["LearnerLaunchSpec", "build_learner_launch_specs", "orchestrator_main"]
