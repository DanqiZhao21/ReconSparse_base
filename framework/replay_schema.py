from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence


REPLAY_SCHEMA_VERSION = 2


def ensure_replay_schema(replay: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if not isinstance(replay, MutableMapping):
        raise RuntimeError(f"replay must be a mutable mapping, got {type(replay)!r}")
    replay["schema_version"] = REPLAY_SCHEMA_VERSION
    return replay


def ensure_section(replay: MutableMapping[str, Any], name: str) -> MutableMapping[str, Any]:
    ensure_replay_schema(replay)
    section = replay.get(name, None)
    if section is None:
        section = {}
        replay[name] = section
    if not isinstance(section, MutableMapping):
        raise RuntimeError(f"replay[{name!r}] must be a mapping")
    return section


def require_section(replay: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(replay, Mapping):
        raise RuntimeError(f"replay must be a mapping, got {type(replay)!r}")
    section = replay.get(name, None)
    if not isinstance(section, Mapping):
        raise RuntimeError(f"replay missing required section {name!r}")
    return section


def make_policy_replay(
    *,
    backend: str,
    model_inputs: Mapping[str, Any],
    action_id: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
    schema_version: int = 1,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "backend": str(backend),
        "schema_version": int(schema_version),
        "model_inputs": dict(model_inputs),
        "action_id": dict(action_id),
    }
    if extra:
        out["extra"] = dict(extra)
    return out


def get_policy_replay(replay: Mapping[str, Any]) -> Mapping[str, Any]:
    return require_section(replay, "policy")


def get_policy_model_inputs(replay: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = get_policy_replay(replay)
    model_inputs = policy.get("model_inputs", None)
    if not isinstance(model_inputs, Mapping):
        raise RuntimeError("replay['policy'] missing required model_inputs")
    return model_inputs


def get_policy_action_id(replay: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = get_policy_replay(replay)
    action_id = policy.get("action_id", None)
    if not isinstance(action_id, Mapping):
        raise RuntimeError("replay['policy'] missing required action_id")
    return action_id


def set_env_plan(
    replay: MutableMapping[str, Any],
    *,
    plan_xyyaw: Any,
    plan_frame: str = "ego",
    dt_s: float | None = None,
    action_flag: int | None = None,
) -> None:
    env = ensure_section(replay, "env")
    env["plan_xyyaw"] = plan_xyyaw
    env["plan_frame"] = str(plan_frame)
    if dt_s is not None:
        env["dt_s"] = float(dt_s)
    if action_flag is not None:
        env["action_flag"] = int(action_flag)


def get_env_plan_xyyaw(replay: Mapping[str, Any]) -> Any:
    env = require_section(replay, "env")
    if "plan_xyyaw" not in env:
        raise RuntimeError("replay['env'] missing required plan_xyyaw")
    return env["plan_xyyaw"]


def ensure_grpo(replay: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    return ensure_section(replay, "grpo")


def ensure_grpo_scorer(replay: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    grpo = ensure_grpo(replay)
    scorer = grpo.get("scorer", None)
    if scorer is None:
        scorer = {}
        grpo["scorer"] = scorer
    if not isinstance(scorer, MutableMapping):
        raise RuntimeError("replay['grpo']['scorer'] must be a mapping")
    return scorer


def get_grpo_scorer(replay: Mapping[str, Any]) -> Mapping[str, Any]:
    grpo = require_section(replay, "grpo")
    scorer = grpo.get("scorer", None)
    if not isinstance(scorer, Mapping):
        raise RuntimeError("replay['grpo'] missing required scorer section")
    return scorer


def set_grpo_scorer_fields(replay: MutableMapping[str, Any], **fields: Any) -> None:
    scorer = ensure_grpo_scorer(replay)
    for key, value in fields.items():
        if value is not None:
            scorer[str(key)] = value


def set_grpo_candidates(
    replay: MutableMapping[str, Any],
    *,
    mode_indices: Any,
    old_log_probs: Any,
    traj_xyyaw: Any,
    score_logits: Any | None = None,
) -> None:
    grpo = ensure_grpo(replay)
    candidates: dict[str, Any] = {
        "mode_indices": mode_indices,
        "old_log_probs": old_log_probs,
        "traj_xyyaw": traj_xyyaw,
    }
    if score_logits is not None:
        candidates["score_logits"] = score_logits
    grpo["candidates"] = candidates


def get_grpo_candidates(replay: Mapping[str, Any]) -> Mapping[str, Any]:
    grpo = require_section(replay, "grpo")
    candidates = grpo.get("candidates", None)
    if not isinstance(candidates, Mapping):
        raise RuntimeError("replay['grpo'] missing required candidates section")
    return candidates


def require_grpo_candidate_field(replay: Mapping[str, Any], key: str, *, index: int) -> Any:
    candidates = get_grpo_candidates(replay)
    if key not in candidates:
        raise RuntimeError(f"GRPO replay missing candidates[{key!r}] at index {int(index)}")
    return candidates[key]


def set_front_obstacle_aux(replay: MutableMapping[str, Any], *, info: Mapping[str, Any]) -> None:
    front: dict[str, Any] = {}
    key_map = {
        "front_obstacle_available": "available",
        "front_obstacle_gap_m": "gap_m",
        "front_obstacle_lateral_m": "lateral_m",
        "front_obstacle_closing_speed_mps": "closing_speed_mps",
        "front_obstacle_ttc_s": "ttc_s",
        "front_obstacle_category": "category",
    }
    for source_key, target_key in key_map.items():
        if source_key in info:
            front[target_key] = info[source_key]
    if not front:
        return
    aux = ensure_section(replay, "aux")
    aux["front_obstacle"] = front


def get_front_obstacle_aux(replay: Mapping[str, Any]) -> Mapping[str, Any] | None:
    aux = replay.get("aux", None)
    if not isinstance(aux, Mapping):
        return None
    front = aux.get("front_obstacle", None)
    if not isinstance(front, Mapping):
        return None
    return front


def nested_grpo_scorer_replays(replays: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(get_grpo_scorer(replay)) for replay in replays]
