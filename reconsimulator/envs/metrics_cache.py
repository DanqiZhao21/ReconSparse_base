import json
import os
from typing import Any, Dict, Optional

_scene_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}
_scene_env_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}


def _scene_dir(scene_id: int) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "nus", "data", f"{scene_id:03d}")


def _cache_path(scene_id: int) -> str:
    return os.path.join(_scene_dir(scene_id), "metrics_cache.json")


def _env_cache_path(scene_id: int) -> str:
    return os.path.join(_scene_dir(scene_id), "env_cache.json")


def load_scene_cache(scene_id: int) -> Optional[Dict[int, Dict[str, Any]]]:
    if scene_id in _scene_cache:
        return _scene_cache[scene_id]
    path = _cache_path(scene_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # keys are strings of frame indices → convert to int
    cache: Dict[int, Dict[str, Any]] = {int(k): v for k, v in data.items()}
    _scene_cache[scene_id] = cache
    return cache


def get_precomputed_step_metrics(scene_id: int, step_idx: int) -> Optional[Dict[str, Any]]:
    cache = load_scene_cache(scene_id)
    if cache is None:
        return None
    return cache.get(int(step_idx))


def save_scene_cache(scene_id: int, cache: Dict[int, Dict[str, Any]]) -> str:
    os.makedirs(_scene_dir(scene_id), exist_ok=True)
    path = _cache_path(scene_id)
    serializable = {str(k): v for k, v in cache.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    # store in memory as well
    _scene_cache[scene_id] = cache
    return path


# -------- Environment snapshot cache (for online ego-conditioned metrics) -------- #

def load_scene_env_cache(scene_id: int) -> Optional[Dict[int, Dict[str, Any]]]:
    if scene_id in _scene_env_cache:
        return _scene_env_cache[scene_id]
    path = _env_cache_path(scene_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # allow optional meta section
    if isinstance(data, dict) and "meta" in data:
        entries = {k: v for k, v in data.items() if k != "meta"}
    else:
        entries = data
    cache: Dict[int, Dict[str, Any]] = {int(k): v for k, v in entries.items()}
    _scene_env_cache[scene_id] = cache
    return cache


def get_env_snapshot(scene_id: int, step_idx: int) -> Optional[Dict[str, Any]]:
    cache = load_scene_env_cache(scene_id)
    if cache is None:
        return None
    return cache.get(int(step_idx))


def save_scene_env_cache(scene_id: int, cache: Dict[int, Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(_scene_dir(scene_id), exist_ok=True)
    path = _env_cache_path(scene_id)
    payload: Dict[str, Any]
    if meta is not None:
        payload = {"meta": meta}
        payload.update({str(k): v for k, v in cache.items()})
    else:
        payload = {str(k): v for k, v in cache.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _scene_env_cache[scene_id] = cache
    return path
