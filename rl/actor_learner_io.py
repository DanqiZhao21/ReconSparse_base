from __future__ import annotations

import os
import time
import uuid
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def atomic_torch_save(obj: Any, path: str) -> None:
    """Atomically write a torch checkpoint (best-effort on POSIX)."""
    d = os.path.dirname(os.path.abspath(path))
    _mkdir(d)
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def read_int(path: str, default: int = 0) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        return int(s)
    except Exception:
        return int(default)


def write_int(path: str, value: int) -> None:
    d = os.path.dirname(os.path.abspath(path))
    _mkdir(d)
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(int(value)))
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
    def latest_ckpt(self) -> str:
        return os.path.join(self.weights_dir, "latest.ckpt")

    @property
    def version_file(self) -> str:
        return os.path.join(self.weights_dir, "version.txt")

    @property
    def stop_file(self) -> str:
        # Presence of this file indicates all processes should terminate.
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


def list_shards(paths: BufferPaths, *, suffix: str = ".pt") -> List[str]:
    ensure_buffer_layout(paths)
    out: List[str] = []
    try:
        for name in os.listdir(paths.shards_dir):
            if not name.endswith(suffix):
                continue
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
        # Fallback: copy then remove
        try:
            shutil.copy2(shard_path, dst)
            os.remove(shard_path)
        except Exception:
            pass


def count_inflight(paths: BufferPaths, *, actor_id: str) -> int:
    n = 0
    for p in list_shards(paths):
        if f"actor{actor_id}_" in os.path.basename(p):
            n += 1
    return int(n)


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
        v = read_int(paths.version_file, default=0)
        if int(v) >= int(min_version):
            return int(v)
        if timeout_s is not None and (time.time() - t0) > float(timeout_s):
            return int(v)
        time.sleep(float(poll_s))
