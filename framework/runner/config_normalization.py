from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List

import torch

_RUN_TIMESTAMP_PREFIX_RE = re.compile(r"^\d{8}_\d{6}(?:_|$)")

_CLOSED_LOOP_ALGO_ALIASES = {
    "ppo": "ppo",
    "reinforce": "reinforce",
    "reinforcepp": "reinforcepp",
    "reinforce++": "reinforcepp",
    "reinforce_pp": "reinforcepp",
    "sac": "sac",
    "soft_actor_critic": "sac",
    "noclose": "grpo_only",
    "noclosereward": "grpo_only",
    "no_close_reward": "grpo_only",
    "grpoonly": "grpo_only",
    "grpo_only": "grpo_only",
}
_GRPO_TOKENS = {"grpo", "nogrpo", "no_grpo"}
_AUX_TOKENS = {"aux", "auxi", "auxiliary", "noaux", "noauxi", "no_aux", "no_auxiliary"}


def timestamp_actor_learner_buffer_dir(cfg: Dict[str, Any], *, timestamp: str) -> str | None:
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}
    if not isinstance(al_cfg, dict) or len(al_cfg) == 0:
        return None
    if bool(al_cfg.get("timestamp_buffer_dir", True)) is False:
        return None

    buffer_dir = str(al_cfg.get("buffer_dir", "outputs/actor_learner"))
    path = Path(buffer_dir)
    name = path.name
    if _RUN_TIMESTAMP_PREFIX_RE.match(name):
        return None

    resolved = str(path.with_name(f"{str(timestamp)}_{name}"))
    al_cfg["buffer_dir"] = resolved
    train_cfg["actor_learner"] = al_cfg
    cfg["train"] = train_cfg
    return resolved


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


def resolve_learner_gpu_ids(al_cfg: Dict[str, Any]) -> List[int]:
    explicit = _list_int(al_cfg.get("learner_gpu_ids", None))
    if len(explicit) > 0:
        return explicit
    return [int(al_cfg.get("learner_gpu_id", 0))]


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
    learner_gpu_ids = resolve_learner_gpu_ids(al_cfg)
    learner_gpu = int(learner_gpu_ids[0])
    actor_per_gpu = max(1, int(al_cfg.get("actor_per_gpu", 1)))
    if len(learner_gpu_ids) > 1:
        learner_gpu_set = {int(gid) for gid in learner_gpu_ids}
        ordered = [gid for gid in visible if gid not in learner_gpu_set]
        if len(ordered) == 0:
            ordered = [learner_gpu]
    else:
        ordered = [learner_gpu] + [gid for gid in visible if gid != learner_gpu]

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


def _normalize_algo_token(token: str) -> str:
    return str(token).strip().lower().replace("-", "_")


def closed_loop_algorithm_configs(train_cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    closed_loop_cfg = train_cfg.get("closed_loop", {}) or {}
    if not isinstance(closed_loop_cfg, dict):
        closed_loop_cfg = {}

    reinforcepp_cfg = closed_loop_cfg.get("reinforcepp", {}) or {}
    ppo_cfg = closed_loop_cfg.get("ppo", {}) or {}
    sac_cfg = closed_loop_cfg.get("sac", {}) or {}
    if not isinstance(reinforcepp_cfg, dict):
        reinforcepp_cfg = {}
    if not isinstance(ppo_cfg, dict):
        ppo_cfg = {}
    if not isinstance(sac_cfg, dict):
        sac_cfg = {}
    return reinforcepp_cfg, ppo_cfg, sac_cfg


def normalize_train_algorithm_cfg(cfg: Dict[str, Any]) -> None:
    train_cfg = cfg.get("train", {}) or {}
    if not isinstance(train_cfg, dict):
        raise ValueError("train must be a mapping")

    raw_spec = str(train_cfg.get("algo", "ppo")).strip()
    if raw_spec == "":
        raw_spec = "ppo"
    spec_key = raw_spec.lower()
    parts = [_normalize_algo_token(part) for part in raw_spec.split("-") if str(part).strip()]

    closed_loop: str | None = None
    grpo_enabled: bool | None = None
    auxiliary_enabled: bool | None = None
    unknown: list[str] = []
    for part in parts:
        algo = _CLOSED_LOOP_ALGO_ALIASES.get(part)
        if algo is not None:
            if closed_loop is not None and closed_loop != algo:
                raise ValueError(f"train.algo={raw_spec!r} contains more than one closed-loop algorithm")
            closed_loop = algo
            continue
        if part in _GRPO_TOKENS:
            value = part == "grpo"
            if grpo_enabled is not None and grpo_enabled != value:
                raise ValueError(f"train.algo={raw_spec!r} contains conflicting GRPO switches")
            grpo_enabled = value
            continue
        if part in _AUX_TOKENS:
            value = part in {"aux", "auxi", "auxiliary"}
            if auxiliary_enabled is not None and auxiliary_enabled != value:
                raise ValueError(f"train.algo={raw_spec!r} contains conflicting auxiliary switches")
            auxiliary_enabled = value
            continue
        unknown.append(part)

    if unknown:
        raise ValueError(f"Unsupported train.algo token(s) in {raw_spec!r}: {', '.join(unknown)}")
    if closed_loop is None:
        raise ValueError(f"train.algo={raw_spec!r} must include one closed-loop algorithm")

    if grpo_enabled is None:
        grpo_cfg_existing = train_cfg.get("grpo", {}) or {}
        grpo_enabled = bool(grpo_cfg_existing.get("enable", False)) if isinstance(grpo_cfg_existing, dict) else False
    if auxiliary_enabled is None:
        aux_cfg_existing = train_cfg.get("auxiliary", {}) or {}
        auxiliary_enabled = bool(aux_cfg_existing.get("enable", False)) if isinstance(aux_cfg_existing, dict) else False

    if closed_loop == "grpo_only" and not grpo_enabled:
        raise ValueError("train.algo noCloseReward/grpo_only requires GRPO to be enabled")

    train_cfg["algo_spec"] = spec_key
    train_cfg["algo"] = str(closed_loop)
    closed_loop_cfg = train_cfg.get("closed_loop", {}) or {}
    if not isinstance(closed_loop_cfg, dict):
        closed_loop_cfg = {}
    closed_loop_cfg["kind"] = str(closed_loop)
    train_cfg["closed_loop"] = closed_loop_cfg

    grpo_cfg = train_cfg.get("grpo", {}) or {}
    if not isinstance(grpo_cfg, dict):
        grpo_cfg = {}
    grpo_cfg["enable"] = bool(grpo_enabled)
    if not bool(grpo_enabled):
        grpo_cfg["coef"] = 0.0
        grpo_cfg["objective"] = "grpo"
        grpo_cfg["num_candidates"] = 0
    train_cfg["grpo"] = grpo_cfg

    aux_cfg = train_cfg.get("auxiliary", None)
    if aux_cfg is None:
        aux_cfg = {}
    if not isinstance(aux_cfg, dict):
        aux_cfg = {}
    aux_cfg["enable"] = bool(auxiliary_enabled)
    risk_cfg = aux_cfg.get("risk_decel", {}) or {}
    if not isinstance(risk_cfg, dict):
        risk_cfg = {}
    risk_coef = float(risk_cfg.get("coef", 0.0) or 0.0)
    risk_cfg["enable"] = bool(auxiliary_enabled and risk_coef > 0.0)
    if not bool(auxiliary_enabled):
        risk_cfg["coef"] = 0.0
    aux_cfg["risk_decel"] = risk_cfg
    train_cfg["auxiliary"] = aux_cfg

    cfg["train"] = train_cfg


def normalize_actor_learner_cfg(cfg: Dict[str, Any]) -> None:
    normalize_train_algorithm_cfg(cfg)

    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}
    if not isinstance(al_cfg, dict) or len(al_cfg) == 0:
        return

    explicit_ids = _list_int(al_cfg.get("actor_gpu_ids", None))
    learner_gpu_ids = resolve_learner_gpu_ids(al_cfg)
    al_cfg["learner_gpu_ids"] = list(learner_gpu_ids)
    al_cfg["learner_gpu_id"] = int(learner_gpu_ids[0])

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

    num_envs_per_actor = int(al_cfg.get("num_envs_per_actor", 1))
    if num_envs_per_actor <= 0:
        raise ValueError(f"num_envs_per_actor must be >= 1, got {num_envs_per_actor}")
    al_cfg["num_envs_per_actor"] = int(num_envs_per_actor)

    vec_env_mode = str(al_cfg.get("vec_env_mode", "serial")).strip().lower()
    if vec_env_mode not in {"serial", "subproc"}:
        raise ValueError(f"vec_env_mode must be 'serial' or 'subproc', got {vec_env_mode!r}")
    al_cfg["vec_env_mode"] = str(vec_env_mode)

    cur_inflight = al_cfg.get("max_inflight_per_actor", None)
    if cur_inflight is not None and int(cur_inflight) < int(num_envs_per_actor):
        al_cfg["max_inflight_per_actor"] = int(num_envs_per_actor)

    if bool(auto_inflight):
        shards_per_update = int(al_cfg.get("shards_per_update", al_cfg.get("num_actors", 1)))
        num_actors = int(al_cfg.get("num_actors", 0))
        if num_actors <= 0:
            ids = _list_int(al_cfg.get("actor_gpu_ids", None))
            num_actors = int(len(ids)) if len(ids) > 0 else 1
            al_cfg["num_actors"] = int(num_actors)
        per_actor_target = max(1, int(math.ceil(float(shards_per_update)*2 / float(max(1, int(num_actors))))))
        env_batch = max(1, int(num_envs_per_actor))
        required = int(env_batch * math.ceil(float(per_actor_target) / float(env_batch)))
        cur = al_cfg.get("max_inflight_per_actor", None)
        if cur is None or int(cur) < int(required):
            al_cfg["max_inflight_per_actor"] = int(required)

    train_cfg["actor_learner"] = al_cfg
    cfg["train"] = train_cfg
