from __future__ import annotations

import math
from typing import Any, Dict, List

import torch


def _list_int(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: List[int] = []
        for value in values:
            try:
                out.append(int(value))
            except Exception:
                continue
        return out
    if isinstance(values, str):
        out: List[int] = []
        for text in values.split(","):
            text = text.strip()
            if not text:
                continue
            try:
                out.append(int(text))
            except Exception:
                continue
        return out
    return []


def resolve_actor_gpu_ids(al_cfg: Dict[str, Any], *, num_actors: int) -> List[int]:
    n = max(1, int(num_actors))
    explicit = _list_int(al_cfg.get("actor_gpu_ids", None))
    if len(explicit) > 0:
        if len(explicit) >= n:
            return explicit[:n]
        return [int(explicit[i % len(explicit)]) for i in range(n)]

    if not torch.cuda.is_available():
        return [-1 for _ in range(n)]

    visible = list(range(int(torch.cuda.device_count())))
    learner_gpu = int(al_cfg.get("learner_gpu_id", 0))
    actor_per_gpu = max(1, int(al_cfg.get("actor_per_gpu", 1)))
    ordered = [learner_gpu] + [gid for gid in visible if gid != learner_gpu]
    if len(ordered) == 0:
        ordered = [0]

    plan: List[int] = []
    idx = 0
    while len(plan) < n:
        gid = int(ordered[idx % len(ordered)])
        for _ in range(actor_per_gpu):
            if len(plan) >= n:
                break
            plan.append(gid)
        idx += 1
    return plan


def normalize_actor_learner_cfg(cfg: Dict[str, Any]) -> None:
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}
    if not isinstance(al_cfg, dict) or len(al_cfg) == 0:
        return

    explicit_ids = _list_int(al_cfg.get("actor_gpu_ids", None))
    actor_gpu_pool = _list_int(al_cfg.get("actor_gpu_pool", None) or al_cfg.get("gpu_ids", None) or al_cfg.get("gpus", None))
    actors_per_gpu = al_cfg.get("actors_per_gpu", None)
    if actors_per_gpu is None:
        actors_per_gpu = al_cfg.get("actor_per_gpu", None)
    actors_per_gpu_i = int(actors_per_gpu) if actors_per_gpu is not None else 0

    if len(explicit_ids) == 0 and len(actor_gpu_pool) > 0 and actors_per_gpu_i > 0:
        plan: List[int] = []
        for gid in actor_gpu_pool:
            for _ in range(int(actors_per_gpu_i)):
                plan.append(int(gid))
        al_cfg["actor_gpu_ids"] = plan
        al_cfg["num_actors"] = int(len(plan))

    auto_inflight = al_cfg.get("auto_max_inflight_per_actor", None)
    if auto_inflight is None:
        auto_inflight = bool(len(actor_gpu_pool) > 0 and actors_per_gpu_i > 0)

    if bool(auto_inflight):
        shards_per_update = int(al_cfg.get("shards_per_update", al_cfg.get("num_actors", 1)))
        num_actors = int(al_cfg.get("num_actors", 0))
        if num_actors <= 0:
            ids = _list_int(al_cfg.get("actor_gpu_ids", None))
            num_actors = int(len(ids)) if len(ids) > 0 else 1
            al_cfg["num_actors"] = int(num_actors)
        required = max(1, int(math.ceil(float(shards_per_update) / float(max(1, int(num_actors))))))
        cur = al_cfg.get("max_inflight_per_actor", None)
        if cur is None or int(cur) < int(required):
            al_cfg["max_inflight_per_actor"] = int(required)

    train_cfg["actor_learner"] = al_cfg
    cfg["train"] = train_cfg
