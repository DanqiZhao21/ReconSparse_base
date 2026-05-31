from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Any, List

import torch


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def atomic_torch_save(obj: Any, path: str) -> None:
    """Atomically write a torch checkpoint (best-effort on POSIX)."""
    directory = os.path.dirname(os.path.abspath(path))
    _mkdir(directory)
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def read_int(path: str, default: int = 0) -> int:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except Exception:
        return int(default)


def write_int(path: str, value: int) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    _mkdir(directory)
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(str(int(value)))
    os.replace(tmp, path)


@dataclass
class BufferPaths:
    root: str

    @property
    def shards_dir(self) -> str:
        return os.path.join(self.root, "buffer", "shards")

    @property
    def consumed_dir(self) -> str:
        return os.path.join(self.root, "buffer", "consumed")

    @property
    def weights_dir(self) -> str:
        return os.path.join(self.root, "weights")

    @property
    def actors_dir(self) -> str:
        return os.path.join(self.root, "actors")

    @property
    def latest_ckpt(self) -> str:
        return os.path.join(self.weights_dir, "latest.ckpt")

    @property
    def version_file(self) -> str:
        return os.path.join(self.weights_dir, "version.txt")

    @property
    def stop_file(self) -> str:
        return os.path.join(self.root, "STOP")


def stop_requested(paths: BufferPaths) -> bool:
    try:
        return os.path.exists(paths.stop_file)
    except Exception:
        return False


def ensure_buffer_layout(paths: BufferPaths) -> None:
    _mkdir(paths.shards_dir)
    _mkdir(paths.consumed_dir)
    _mkdir(paths.weights_dir)
    _mkdir(paths.actors_dir)


def actor_failure_flag_path(paths: BufferPaths, actor_id: int) -> str:
    ensure_buffer_layout(paths)
    return os.path.join(paths.actors_dir, f"actor{int(actor_id)}.failed")


def actor_failure_log_path(paths: BufferPaths, actor_id: int) -> str:
    ensure_buffer_layout(paths)
    return os.path.join(paths.actors_dir, f"actor{int(actor_id)}.log")


def actor_heartbeat_path(paths: BufferPaths, actor_id: int) -> str:
    ensure_buffer_layout(paths)
    return os.path.join(paths.actors_dir, f"actor{int(actor_id)}.heartbeat")


def clear_actor_failure(paths: BufferPaths, actor_id: int) -> None:
    for path in [actor_failure_flag_path(paths, actor_id), actor_failure_log_path(paths, actor_id)]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def write_actor_failure(
    paths: BufferPaths,
    actor_id: int,
    *,
    message: str,
    traceback_text: str | None = None,
) -> str:
    ensure_buffer_layout(paths)
    flag_path = actor_failure_flag_path(paths, actor_id)
    log_path = actor_failure_log_path(paths, actor_id)
    write_ts = time.time()
    text = f"time={write_ts:.6f}\nactor_id={int(actor_id)}\nmessage={str(message).strip()}\n"
    if traceback_text is not None and str(traceback_text).strip():
        text += "\ntraceback:\n"
        text += str(traceback_text).rstrip() + "\n"
    _mkdir(os.path.dirname(os.path.abspath(flag_path)))
    with open(flag_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return flag_path


def write_actor_heartbeat(paths: BufferPaths, actor_id: int, *, message: str = "") -> str:
    ensure_buffer_layout(paths)
    path = actor_heartbeat_path(paths, actor_id)
    now = time.time()
    text = f"time={now:.6f}\nactor_id={int(actor_id)}\nmessage={str(message).strip()}\n"
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)
    return path


def mark_stale_actor_heartbeats(
    paths: BufferPaths,
    actor_ids: List[int],
    *,
    timeout_s: float,
    now: float | None = None,
    stage_fn: Any | None = None,
) -> List[int]:
    timeout = float(timeout_s)
    if timeout <= 0.0:
        return []
    ensure_buffer_layout(paths)
    failed = set(list_failed_actor_ids(paths))
    marked: List[int] = []
    cur_t = float(time.time() if now is None else now)
    for actor_id in actor_ids:
        aid = int(actor_id)
        if aid in failed:
            continue
        heartbeat = actor_heartbeat_path(paths, aid)
        try:
            last_t = float(os.path.getmtime(heartbeat))
        except FileNotFoundError:
            continue
        except Exception:
            continue
        age_s = cur_t - float(last_t)
        if age_s < timeout:
            continue
        write_actor_failure(
            paths,
            aid,
            message=f"actor heartbeat stale age_s={age_s:.2f} timeout_s={timeout:.2f}",
        )
        marked.append(aid)

    if marked and callable(stage_fn):
        stage_fn(f"[learner] marked stale heartbeat actor(s) failed: {marked}")
    return marked


def list_failed_actor_ids(paths: BufferPaths) -> List[int]:
    ensure_buffer_layout(paths)
    failed: List[int] = []
    try:
        names = os.listdir(paths.actors_dir)
    except Exception:
        return failed
    for name in names:
        if not name.startswith("actor") or not name.endswith(".failed"):
            continue
        middle = name[len("actor") : -len(".failed")]
        try:
            failed.append(int(middle))
        except Exception:
            continue
    failed.sort()
    return failed


def list_shards(paths: BufferPaths, *, suffix: str = ".pt") -> List[str]:
    ensure_buffer_layout(paths)
    out: List[str] = []
    try:
        for name in os.listdir(paths.shards_dir):
            if name.endswith(str(suffix)):
                out.append(os.path.join(paths.shards_dir, name))
    except Exception:
        return []
    out.sort()
    return out


def move_to_consumed(paths: BufferPaths, shard_path: str) -> None:
    ensure_buffer_layout(paths)
    base = os.path.basename(shard_path)
    dst = os.path.join(paths.consumed_dir, base)
    try:
        os.replace(shard_path, dst)
    except Exception:
        try:
            shutil.copy2(shard_path, dst)
            os.remove(shard_path)
        except Exception:
            pass


def prune_consumed(
    paths: BufferPaths,
    *,
    keep_basenames: set[str] | None = None,
    keep_last: int | None = None,
    suffix: str = ".pt",
) -> int:
    ensure_buffer_layout(paths)
    deleted = 0

    try:
        names = [name for name in os.listdir(paths.consumed_dir) if name.endswith(str(suffix))]
    except Exception:
        return 0

    if keep_basenames is not None:
        keep = {str(name) for name in keep_basenames}
        for name in names:
            if name in keep:
                continue
            path = os.path.join(paths.consumed_dir, name)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                    deleted += 1
            except Exception:
                pass
        return int(deleted)

    if keep_last is None:
        return 0

    keep_n = max(0, int(keep_last))
    if keep_n <= 0:
        for name in names:
            path = os.path.join(paths.consumed_dir, name)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                    deleted += 1
            except Exception:
                pass
        return int(deleted)

    files: List[tuple[float, str]] = []
    for name in names:
        path = os.path.join(paths.consumed_dir, name)
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0.0
        files.append((float(mtime), name))
    files.sort(key=lambda item: item[0], reverse=True)

    for _, name in files[keep_n:]:
        path = os.path.join(paths.consumed_dir, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                deleted += 1
        except Exception:
            pass
    return int(deleted)


def count_inflight(paths: BufferPaths, *, actor_id: str) -> int:
    return sum(1 for path in list_shards(paths) if f"actor{actor_id}_" in os.path.basename(path))


def wait_for_version(
    paths: BufferPaths,
    *,
    min_version: int,
    poll_s: float = 0.2,
    timeout_s: float | None = None,
    stop_file: str | None = None,
) -> int:
    t0 = time.time()
    while True:
        if stop_file is not None:
            try:
                if os.path.exists(stop_file):
                    return read_int(paths.version_file, default=0)
            except Exception:
                pass
        version = read_int(paths.version_file, default=0)
        if int(version) >= int(min_version):
            return int(version)
        if timeout_s is not None and (time.time() - t0) > float(timeout_s):
            return int(version)
        time.sleep(float(poll_s))


__all__ = [
    "BufferPaths",
    "atomic_torch_save",
    "count_inflight",
    "ensure_buffer_layout",
    "actor_failure_flag_path",
    "actor_failure_log_path",
    "actor_heartbeat_path",
    "clear_actor_failure",
    "list_shards",
    "list_failed_actor_ids",
    "mark_stale_actor_heartbeats",
    "move_to_consumed",
    "prune_consumed",
    "read_int",
    "stop_requested",
    "wait_for_version",
    "write_int",
    "write_actor_failure",
    "write_actor_heartbeat",
]
