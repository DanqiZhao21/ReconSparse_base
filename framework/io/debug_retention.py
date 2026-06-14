from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from framework.io.buffer import BufferPaths


def history_checkpoint_path(paths: BufferPaths, *, version: int) -> str:
    return os.path.join(paths.weights_dir, "history", f"version_{int(version):06d}.ckpt")


def should_retain_version(*, version: int, retain_versions: int) -> bool:
    return int(retain_versions) > 0 and 1 <= int(version) <= int(retain_versions)


def copy_latest_to_history(paths: BufferPaths, *, version: int) -> str:
    src = paths.latest_ckpt
    dst = history_checkpoint_path(paths, version=int(version))
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _load_shard_meta(path: str) -> Dict[str, Any]:
    try:
        shard = torch.load(path, map_location="cpu")
    except Exception as exc:
        return {"load_error": str(exc)}
    if not isinstance(shard, dict):
        return {"load_error": f"expected mapping, got {type(shard)!r}"}
    meta = shard.get("meta", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    out: Dict[str, Any] = {
        "weights_version": meta.get("weights_version"),
        "actor_id": meta.get("actor_id"),
        "env_id": meta.get("env_id"),
        "num_steps": meta.get("num_steps"),
    }
    for key in ("old_logp", "reward", "done", "replay"):
        value = shard.get(key, None)
        if torch.is_tensor(value):
            out[f"{key}_len"] = int(value.view(-1).shape[0])
        elif isinstance(value, list):
            out[f"{key}_len"] = int(len(value))
    return out


def archive_selected_shards_for_debug(
    paths: BufferPaths,
    *,
    selected: Sequence[str],
    update_index: int,
    cur_version: int,
    new_version: int,
) -> str:
    archive_root = Path(paths.root) / "debug_retention" / (
        f"update_{int(update_index):06d}_from_v{int(cur_version):06d}_to_v{int(new_version):06d}"
    )
    shard_dir = archive_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest_shards: List[Dict[str, Any]] = []
    for source in selected:
        src = Path(source)
        dst = shard_dir / src.name
        shutil.copy2(src, dst)
        shard_info = {
            "source_path": str(src),
            "archive_path": str(dst),
            "basename": src.name,
        }
        shard_info.update(_load_shard_meta(str(dst)))
        manifest_shards.append(shard_info)

    manifest = {
        "created_time": float(time.time()),
        "update_index": int(update_index),
        "cur_version": int(cur_version),
        "new_version": int(new_version),
        "shards": manifest_shards,
    }
    manifest_path = archive_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return str(manifest_path)
