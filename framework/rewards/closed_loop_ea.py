from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


_DEFAULT_EA_PROJECT_SRC = Path("/root/clone/EA/src")
_DEFAULT_AGENT_STATE_CACHE_ROOT = Path(__file__).resolve().parents[2] / "assets" / "nus" / "data"


def _wrap_angle(angle: float) -> float:
    return float(math.atan2(math.sin(float(angle)), math.cos(float(angle))))


def _pose_yaw_xy(pose: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(pose, dtype=np.float64)
    return (
        float(arr[0, 3]),
        float(arr[2, 3]),
        float(math.atan2(float(arr[2, 0]), float(arr[0, 0]))),
    )


def _load_compute_fn(project_src: str | Path | None) -> Callable[..., float] | None:
    try:
        src = Path(project_src) if project_src is not None and str(project_src).strip() else _DEFAULT_EA_PROJECT_SRC
        if src.exists():
            src_text = str(src)
            if src_text not in sys.path:
                sys.path.insert(0, src_text)
        from ea_project.core_ea import compute_final_ea

        return compute_final_ea
    except Exception:
        return None


class ClosedLoopEAScorer:
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg if isinstance(cfg, dict) else {}
        self.enabled = bool(self.cfg.get("enable", False))
        self.project_src = self.cfg.get("project_src", self.cfg.get("ea_project_src", str(_DEFAULT_EA_PROJECT_SRC)))
        self.agent_state_cache_root = Path(
            self.cfg.get("agent_state_cache_root", self.cfg.get("cache_root", str(_DEFAULT_AGENT_STATE_CACHE_ROOT)))
        )
        self.max_agents = max(1, int(self.cfg.get("max_agents", 4)))
        self.max_distance_m = float(self.cfg.get("max_distance_m", 40.0))
        self.good_threshold = float(self.cfg.get("good_threshold", 0.0))
        self.bad_threshold = float(self.cfg.get("bad_threshold", 8.0))
        self.horizon_s = float(self.cfg.get("horizon_s", self.cfg.get("T_total", 4.0)))
        self.dt_coarse_s = float(self.cfg.get("dt_coarse_s", self.cfg.get("dt_coarse", 0.1)))
        self.dt_fine_s = float(self.cfg.get("dt_fine_s", self.cfg.get("dt_fine", 0.02)))
        self.ego_length_m = float(self.cfg.get("ego_length_m", 4.9))
        self.ego_width_m = float(self.cfg.get("ego_width_m", 2.1))
        self._compute_fn: Callable[..., float] | None | bool = None
        self._agent_cache: dict[int, dict[str, Any]] = {}

    def _ensure_compute_fn(self) -> Callable[..., float] | None:
        if self._compute_fn is False:
            return None
        if callable(self._compute_fn):
            return self._compute_fn
        fn = _load_compute_fn(self.project_src)
        self._compute_fn = fn if fn is not None else False
        return fn

    def _load_agent_cache(self, scene_id: int) -> dict[str, Any]:
        key = int(scene_id)
        cached = self._agent_cache.get(key)
        if cached is not None:
            return cached
        path = self.agent_state_cache_root / f"{key:03d}" / "agent_state_cache.json"
        if not path.exists():
            payload = {"meta": {}, "frames": {}}
            self._agent_cache[key] = payload
            return payload
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        meta = dict(loaded.get("meta", {})) if isinstance(loaded, dict) else {}
        frames = {int(k): v for k, v in dict(loaded).items() if k != "meta"}
        payload = {"meta": meta, "frames": frames}
        self._agent_cache[key] = payload
        return payload

    def _lookup_agent_snapshot(self, *, scene_id: int, frame_idx: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        payload = self._load_agent_cache(scene_id)
        frames = dict(payload.get("frames", {}))
        if not frames:
            return None, dict(payload.get("meta", {}))
        idx = int(frame_idx)
        if idx in frames:
            return frames[idx], dict(payload.get("meta", {}))
        keys = sorted(int(k) for k in frames.keys())
        for key in reversed(keys):
            if key <= idx:
                return frames[key], dict(payload.get("meta", {}))
        return None, dict(payload.get("meta", {}))

    @staticmethod
    def _world_agent_to_local(agent: dict[str, Any], *, camera_front_start: np.ndarray) -> dict[str, Any]:
        cfs = np.asarray(camera_front_start, dtype=np.float64)
        world_to_local = np.linalg.inv(cfs)
        center = np.asarray(agent.get("center_xy", [0.0, 0.0]), dtype=np.float64).reshape(-1)
        world_pt = np.asarray([float(center[0]), float(center[1]), 0.0, 1.0], dtype=np.float64)
        local_pt = world_to_local @ world_pt

        rot = world_to_local[:3, :3]
        velocity = np.asarray(agent.get("velocity_xy", [0.0, 0.0]), dtype=np.float64).reshape(-1)
        world_vec = np.asarray([float(velocity[0]), float(velocity[1]), 0.0], dtype=np.float64)
        local_vec = rot @ world_vec

        camera_yaw = float(math.atan2(float(cfs[2, 0]), float(cfs[0, 0])))
        yaw_local = _wrap_angle(float(agent.get("yaw_rad", 0.0)) - camera_yaw)
        return {
            **agent,
            "center_xy": [float(local_pt[0]), float(local_pt[2])],
            "velocity_xy": [float(local_vec[0]), float(local_vec[2])],
            "yaw_rad": float(yaw_local),
            "speed_mps": float(np.linalg.norm(local_vec[[0, 2]])),
        }

    def _current_ego_state(self, *, ego_pose: np.ndarray, ego_velocity_xy: Any, dt_s: float, previous_ego_pose: np.ndarray | None) -> dict[str, float]:
        x, y, yaw = _pose_yaw_xy(np.asarray(ego_pose, dtype=np.float64))
        vel = np.asarray(ego_velocity_xy if ego_velocity_xy is not None else [0.0, 0.0], dtype=np.float64).reshape(-1)
        speed = float(np.linalg.norm(vel[:2])) if vel.size >= 2 else 0.0
        yaw_rate = 0.0
        if previous_ego_pose is not None:
            _px, _py, prev_yaw = _pose_yaw_xy(np.asarray(previous_ego_pose, dtype=np.float64))
            yaw_rate = _wrap_angle(yaw - prev_yaw) / max(1.0e-6, float(dt_s))
        return {
            "x": float(x),
            "y": float(y),
            "speed_mps": float(speed),
            "yaw_rad": float(yaw),
            "yaw_rate_rps": float(yaw_rate),
            "length_m": float(self.ego_length_m),
            "width_m": float(self.ego_width_m),
        }

    @staticmethod
    def _agent_to_ea_state(agent: dict[str, Any]) -> dict[str, float]:
        center = np.asarray(agent.get("center_xy", [0.0, 0.0]), dtype=np.float64).reshape(-1)
        velocity = np.asarray(agent.get("velocity_xy", [0.0, 0.0]), dtype=np.float64).reshape(-1)
        speed = float(agent.get("speed_mps", np.linalg.norm(velocity[:2]) if velocity.size >= 2 else 0.0))
        return {
            "x": float(center[0]),
            "y": float(center[1]),
            "speed_mps": float(speed),
            "yaw_rad": float(agent.get("yaw_rad", 0.0)),
            "yaw_rate_rps": float(agent.get("yaw_rate_rps", 0.0)),
            "length_m": float(agent.get("length_m", 1.0)),
            "width_m": float(agent.get("width_m", 1.0)),
        }

    def _risk_from_ea(self, ea_value: float) -> float:
        if not math.isfinite(float(ea_value)):
            return 1.0
        value = max(0.0, float(ea_value))
        if value <= self.good_threshold:
            return 0.0
        if value >= self.bad_threshold:
            return 1.0
        span = max(1.0e-6, self.bad_threshold - self.good_threshold)
        return float(np.clip((value - self.good_threshold) / span, 0.0, 1.0))

    def score_current_step(
        self,
        *,
        scene_id: int,
        frame_idx: int,
        ego_pose: np.ndarray,
        ego_velocity_xy: Any = None,
        previous_ego_pose: np.ndarray | None = None,
        camera_front_start: np.ndarray | None = None,
        dt_s: float = 0.5,
    ) -> dict[str, Any]:
        base = {
            "ea_enabled": bool(self.enabled),
            "ea_available": False,
            "ea_max": 0.0,
            "ea_min": 0.0,
            "ea_mean": 0.0,
            "ea_risk": 0.0,
            "ea_evaluated_pairs": 0.0,
        }
        if not self.enabled:
            return base
        compute_fn = self._ensure_compute_fn()
        if compute_fn is None:
            return {**base, "ea_error": "compute_fn_unavailable"}
        snapshot, meta = self._lookup_agent_snapshot(scene_id=int(scene_id), frame_idx=int(frame_idx))
        if not isinstance(snapshot, dict):
            return {**base, "ea_error": "agent_snapshot_unavailable"}

        agents = list(snapshot.get("agents", []) or [])
        coord_frame = str(meta.get("coordinate_frame", "")).strip().lower()
        if coord_frame == "world":
            if camera_front_start is None:
                return {**base, "ea_error": "camera_front_start_unavailable"}
            agents = [self._world_agent_to_local(dict(agent), camera_front_start=np.asarray(camera_front_start, dtype=np.float64)) for agent in agents]

        ego_state = self._current_ego_state(
            ego_pose=np.asarray(ego_pose, dtype=np.float64),
            ego_velocity_xy=ego_velocity_xy,
            dt_s=float(dt_s),
            previous_ego_pose=previous_ego_pose,
        )
        ego_xy = np.asarray([ego_state["x"], ego_state["y"]], dtype=np.float64)
        vehicle_agents = []
        for agent in agents:
            if "vehicle" not in str(agent.get("category", "")).strip().lower():
                continue
            state = self._agent_to_ea_state(dict(agent))
            dist = float(np.linalg.norm(np.asarray([state["x"], state["y"]], dtype=np.float64) - ego_xy))
            if dist <= float(self.max_distance_m):
                vehicle_agents.append((dist, state))
        vehicle_agents.sort(key=lambda item: item[0])
        selected = vehicle_agents[: self.max_agents]
        if not selected:
            return {**base, "ea_available": True}

        values: list[float] = []
        for _dist, agent_state in selected:
            try:
                values.append(
                    float(
                        compute_fn(
                            xA=ego_state["x"],
                            yA=ego_state["y"],
                            vA=ego_state["speed_mps"],
                            hA=ego_state["yaw_rad"],
                            lA=ego_state["length_m"],
                            wA=ego_state["width_m"],
                            yawA=ego_state["yaw_rate_rps"],
                            xB=agent_state["x"],
                            yB=agent_state["y"],
                            vB=agent_state["speed_mps"],
                            hB=agent_state["yaw_rad"],
                            lB=agent_state["length_m"],
                            wB=agent_state["width_m"],
                            yawB=agent_state["yaw_rate_rps"],
                            T_total=float(self.horizon_s),
                            dt_coarse=float(self.dt_coarse_s),
                            dt_fine=float(self.dt_fine_s),
                        )
                    )
                )
            except Exception:
                continue
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not values:
            return {**base, "ea_available": True}
        if finite:
            ea_max = max(finite)
            ea_min = min(finite)
            ea_mean = float(np.mean(np.asarray(finite, dtype=np.float64)))
        else:
            ea_max = float("inf")
            ea_min = float("inf")
            ea_mean = float("inf")
        return {
            **base,
            "ea_available": True,
            "ea_max": float(ea_max),
            "ea_min": float(ea_min),
            "ea_mean": float(ea_mean),
            "ea_risk": float(self._risk_from_ea(float(ea_max))),
            "ea_evaluated_pairs": float(len(values)),
        }
