from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from framework.io.buffer import BufferPaths, move_to_consumed


def parse_shard_weights_version(filename: str) -> Optional[int]:
    try:
        text = str(filename)
        start = text.find("_v")
        if start < 0:
            return None
        start += 2
        end = start
        while end < len(text) and text[end].isdigit():
            end += 1
        if end == start:
            return None
        return int(text[start:end])
    except Exception:
        return None


def discard_stale_shards(
    paths: BufferPaths,
    shard_files: List[str],
    *,
    cur_weights_version: int,
    max_version_gap: int,
) -> List[str]:
    max_gap = max(0, min(2, int(max_version_gap)))
    upcoming = int(cur_weights_version) + 1
    min_ok = int(upcoming - max_gap)
    kept: List[str] = []
    stale: List[str] = []
    for fp in shard_files:
        version = parse_shard_weights_version(os.path.basename(fp))
        if version is None:
            stale.append(fp)
            continue
        if int(version) < int(min_ok):
            stale.append(fp)
            continue
        if int(version) > int(cur_weights_version):
            continue
        kept.append(fp)
    for fp in stale:
        move_to_consumed(paths, fp)
    return kept


def discard_incompatible_shards(
    paths: BufferPaths,
    shard_files: List[str],
    *,
    agent: Any,
    stage_fn: Any | None = None,
) -> List[str]:
    validator = getattr(agent, "replay_is_compatible", None)
    if not callable(validator):
        return list(shard_files)

    kept: List[str] = []
    dropped = 0
    for fp in shard_files:
        try:
            shard = torch.load(fp, map_location="cpu")
            replay = list(shard.get("replay", []))
            if len(replay) == 0:
                kept.append(fp)
                continue
            if all(bool(validator(rep)) for rep in replay):
                kept.append(fp)
                continue
        except Exception as exc:
            if callable(stage_fn):
                stage_fn(f"[learner] dropping incompatible shard {os.path.basename(fp)}: {exc}")
            move_to_consumed(paths, fp)
            dropped += 1
            continue

        if callable(stage_fn):
            stage_fn(f"[learner] dropping incompatible shard {os.path.basename(fp)} due to replay schema mismatch")
        move_to_consumed(paths, fp)
        dropped += 1

    if dropped > 0 and callable(stage_fn):
        stage_fn(f"[learner] discarded {dropped} incompatible shard(s) from {paths.shards_dir}")
    return kept


def select_shards_for_update(
    shard_files: List[str],
    *,
    mode: str,
    num_actors: int,
    shards_per_update: int,
) -> List[str]:
    files = list(shard_files)
    if str(mode).strip().lower().startswith("sync"):
        have = set()
        for fp in files:
            name = os.path.basename(fp)
            for actor_idx in range(int(num_actors)):
                if name.startswith(f"actor{actor_idx}_"):
                    have.add(actor_idx)
        if len(have) < int(num_actors):
            return []

        per: Dict[int, str] = {}
        for fp in files:
            name = os.path.basename(fp)
            for actor_idx in range(int(num_actors)):
                if name.startswith(f"actor{actor_idx}_") and actor_idx not in per:
                    per[actor_idx] = fp
        return [per[a] for a in sorted(per.keys())][: int(num_actors)]

    need = max(1, int(shards_per_update))
    if len(files) < need:
        return []
    return files[:need]


__all__ = [
    "discard_incompatible_shards",
    "discard_stale_shards",
    "parse_shard_weights_version",
    "select_shards_for_update",
]
