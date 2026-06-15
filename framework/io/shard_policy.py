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
    max_version_lag: int,
) -> List[str]:
    max_lag = max(0, int(max_version_lag))
    min_ok = int(cur_weights_version) - int(max_lag)
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


def shard_sample_count(shard_path: str) -> int:
    try:
        shard = torch.load(shard_path, map_location="cpu")
    except Exception:
        return 0

    reward = shard.get("reward", None) if isinstance(shard, dict) else None
    if torch.is_tensor(reward):
        return max(0, int(reward.view(-1).shape[0]))

    replay = shard.get("replay", None) if isinstance(shard, dict) else None
    if isinstance(replay, list):
        return max(0, int(len(replay)))

    meta = shard.get("meta", {}) if isinstance(shard, dict) else {}
    if isinstance(meta, dict):
        try:
            return max(0, int(meta.get("num_steps", 0)))
        except Exception:
            return 0
    return 0


def select_shards_for_update(
    shard_files: List[str],
    *,
    mode: str,
    num_actors: int,
    shards_per_update: int,
    samples_per_update: int = 0,
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
    sample_target = max(0, int(samples_per_update))
    if sample_target > 0:
        selected: List[str] = []
        sample_count = 0
        for fp in files:
            selected.append(fp)
            sample_count += int(shard_sample_count(fp))
            if sample_count >= int(sample_target):
                return selected
        return []

    if len(files) < need:
        return []
    return files[:need]


def resolve_async_shards_per_update(
    *,
    requested_shards_per_update: int,
    num_actors: int,
    max_inflight_per_actor: int,
    failed_actor_ids: List[int] | None = None,
) -> int:
    failed = {int(actor_id) for actor_id in (failed_actor_ids or [])}
    alive_actors = max(0, int(num_actors) - len(failed))
    if alive_actors <= 0:
        return 0
    live_capacity = max(1, int(alive_actors)) * max(1, int(max_inflight_per_actor))
    return min(max(1, int(requested_shards_per_update)), int(live_capacity))


__all__ = [
    "discard_incompatible_shards",
    "discard_stale_shards",
    "parse_shard_weights_version",
    "resolve_async_shards_per_update",
    "select_shards_for_update",
    "shard_sample_count",
]
