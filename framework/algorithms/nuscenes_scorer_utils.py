from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

try:
    from reconsimulator.envs import nus_config as nus_cfg
except Exception:
    nus_cfg = None

try:
    from shapely import affinity as shapely_affinity
    from shapely.geometry import Point as ShapelyPoint
    from shapely.geometry import Polygon as ShapelyPolygon
except Exception:
    shapely_affinity = None
    ShapelyPoint = None
    ShapelyPolygon = None


_DEFAULT_NUSCENES_VERSION = str(getattr(nus_cfg, "NUSCENES_VERSION", "v1.0-trainval"))
_DEFAULT_NUSCENES_ROOTS = tuple(
    str(path)
    for path in (
        os.environ.get("NUSCENES_DATAROOT", ""),
        getattr(nus_cfg, "NUSCENES_DATA_ROOT", ""),
        "/OpenDataset/nuscenes/nuscenes",
        "/OpenDataset/nuscenes",
        "/OpenDataset/nuscenes/nuscenes/v1.0-trainval",
    )
    if str(path).strip() != ""
)
_MAP_POLYGON_LAYERS = (
    "drivable_area",
    "road_segment",
    "road_block",
    "lane",
    "lane_connector",
    "ped_crossing",
    "walkway",
)
_MAP_LINE_LAYERS = (
    "lane_divider",
    "road_divider",
)
_DEFAULT_PATCH_RADIUS_M = 20.0
_DEFAULT_SCENE_CACHE_ROOT = Path(__file__).resolve().parents[2] / "assets" / "nus" / "data"
_DEFAULT_TTC_FUTURE_OFFSETS_S = (0.0, 0.3, 0.6, 0.9)
_DEFAULT_TTC_STOPPED_SPEED_THRESHOLD_MPS = 5.0e-3
_DEFAULT_DIRECTION_COMPLIANCE_THRESHOLD_M = 2.0
_DEFAULT_DIRECTION_VIOLATION_THRESHOLD_M = 6.0
_DEFAULT_DIRECTION_REVERSE_ALIGNMENT_THRESHOLD = 0.8
_DEFAULT_DIRECTION_MIN_CONTINUOUS_REVERSE_S = 0.5
_DEFAULT_EA_PROJECT_SRC = Path("/root/clone/EA/src")
_DEFAULT_EA_GOOD_THRESHOLD = 0.0
_DEFAULT_EA_BAD_THRESHOLD = 8.0
_DEFAULT_EA_MAX_AGENTS = 4
_DEFAULT_EA_HORIZON_S = 4.0
_DEFAULT_EA_DT_COARSE_S = 0.1
_DEFAULT_EA_DT_FINE_S = 0.02


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))

#从轨迹点 (x, y) 计算每一帧的朝向 yaw
#TODO: 第一个点prev只是一个占位；
def _path_yaw_from_xy(points_xy: np.ndarray) -> np.ndarray:
    if int(points_xy.shape[0]) <= 0:
        return np.zeros((0,), dtype=np.float32)
    prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), points_xy[:-1]], axis=0)
    # points_xy:   [p0, p1, p2, p3, ...]
    # prev:        [0 , p0, p1, p2, ...]
    delta = points_xy - prev
    yaw = np.arctan2(delta[:, 1], delta[:, 0]).astype(np.float32)
    if int(yaw.shape[0]) > 1:
        yaw[0] = yaw[1]
    return yaw

#计算轨迹的累计路径长度（s）;;把一串轨迹点，变成“沿路径的累计距离（arclength）”
def _polyline_arclength(points_xy: np.ndarray) -> np.ndarray:
    if int(points_xy.shape[0]) <= 0:
        return np.zeros((0,), dtype=np.float32)  #np.zeros(shape, dtype=...)   array([], dtype=float32) ; NumPy 的 shape 必须是“元组（tuple）”，而 (0) 不是元组
    if int(points_xy.shape[0]) == 1:
        return np.zeros((1,), dtype=np.float32)
    seg = np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg, dtype=np.float32)], axis=0)

#_project_progress(cand_xy[-1], gt_xy, gt_s)
def _project_progress(point_xy: np.ndarray, path_xy: np.ndarray, path_s: np.ndarray) -> float:
    if int(path_xy.shape[0]) <= 1:
        return 0.0
    best_dist = float("inf")
    best_s = 0.0
    for idx in range(int(path_xy.shape[0]) - 1):
        p0 = path_xy[idx]
        p1 = path_xy[idx + 1]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq <= 1e-12:
            continue
        alpha = float(np.dot(point_xy - p0, seg) / seg_len_sq)
        alpha = max(0.0, min(1.0, alpha))
        proj = p0 + alpha * seg
        dist = float(np.linalg.norm(point_xy - proj))
        if dist < best_dist:
            best_dist = dist
            best_s = float(path_s[idx] + alpha * math.sqrt(seg_len_sq))
    return best_s


def _linear_decay_score(value: float, *, good_threshold: float, bad_threshold: float) -> float:
    if value <= good_threshold:
        return 1.0
    if value >= bad_threshold:
        return 0.0
    span = max(1.0e-6, bad_threshold - good_threshold)
    return float(np.clip((bad_threshold - value) / span, 0.0, 1.0))


class NuScenesScorerUtils:
    def __init__(
        self,
        *,
        token2vad_path: str | Path,
        nuscenes_dataroot: str | Path | None = None,
        nuscenes_version: str = _DEFAULT_NUSCENES_VERSION,
        scene_cache_root: str | Path | None = None,
        agent_state_cache_root: str | Path | None = None,
        #ea指标相关
        ea_project_src: str | Path | None = None,
        ea_gate_enabled: bool = False,
        ea_gate_good_threshold: float = _DEFAULT_EA_GOOD_THRESHOLD,
        ea_gate_bad_threshold: float = _DEFAULT_EA_BAD_THRESHOLD,
        ea_gate_max_agents: int = _DEFAULT_EA_MAX_AGENTS,
        ea_gate_horizon_s: float = _DEFAULT_EA_HORIZON_S,
        ea_gate_dt_coarse_s: float = _DEFAULT_EA_DT_COARSE_S,
        ea_gate_dt_fine_s: float = _DEFAULT_EA_DT_FINE_S,
        driving_direction_gate_enabled: bool = True,
        dac_gate_enabled: bool = True,
        #weight评分权重
        dac_weight: float = 0.0,
        progress_weight: float = 8.0,
        ttc_weight: float = 5.0,
        lane_keeping_weight: float = 2.0,
        history_comfort_weight: float = 2.0,
    ) -> None:
        self.token2vad_path = Path(token2vad_path)
        self._token2vad: dict[str, dict[str, Any]] | None = None
        self.nuscenes_version = str(nuscenes_version)
        self.nuscenes_dataroot = self._resolve_nuscenes_dataroot(nuscenes_dataroot)
        self.scene_cache_root = Path(scene_cache_root) if scene_cache_root is not None else _DEFAULT_SCENE_CACHE_ROOT
        self.agent_state_cache_root = (
            Path(agent_state_cache_root) if agent_state_cache_root is not None else self.scene_cache_root
        )
        self.sample_context_cache_root = self.scene_cache_root / "_sample_score_context"
        self.ea_project_src = (
            Path(ea_project_src)
            if ea_project_src is not None
            else Path(os.environ.get("EA_PROJECT_SRC", str(_DEFAULT_EA_PROJECT_SRC)))
        )
        self.ea_gate_enabled = bool(ea_gate_enabled)
        self.ea_gate_good_threshold = float(ea_gate_good_threshold)
        self.ea_gate_bad_threshold = float(ea_gate_bad_threshold)
        self.ea_gate_max_agents = max(1, int(ea_gate_max_agents))
        self.ea_gate_horizon_s = float(ea_gate_horizon_s)
        self.ea_gate_dt_coarse_s = float(ea_gate_dt_coarse_s)
        self.ea_gate_dt_fine_s = float(ea_gate_dt_fine_s)
        self.driving_direction_gate_enabled = bool(driving_direction_gate_enabled)
        self.dac_gate_enabled = bool(dac_gate_enabled)
        self.dac_weight = float(dac_weight)
        self.progress_weight = float(progress_weight)
        self.ttc_weight = float(ttc_weight)
        self.lane_keeping_weight = float(lane_keeping_weight)
        self.history_comfort_weight = float(history_comfort_weight)
        self._nusc: Any | None = None
        self._map_cache: dict[str, Any] = {}
        self._scene_env_cache: dict[int, dict[int, dict[str, Any]]] = {}
        self._scene_agent_state_cache: dict[int, dict[str, Any]] = {}
        self._sample_static_context_cache: dict[str, dict[str, Any]] = {}
        self._ea_compute_fn: Callable[..., float] | None | bool = None

    @staticmethod
    def _resolve_nuscenes_dataroot(explicit_root: str | Path | None) -> Path | None:
        candidates: list[Path] = []
        if explicit_root is not None and str(explicit_root).strip() != "":
            candidates.append(Path(str(explicit_root)))
        for item in _DEFAULT_NUSCENES_ROOTS:
            candidates.append(Path(item))
        for candidate in candidates:
            if not candidate.exists():
                continue
            if (candidate / "maps").exists():
                return candidate
            version_dir = candidate / _DEFAULT_NUSCENES_VERSION
            if version_dir.exists() and (version_dir / "maps").exists():
                return version_dir
        return None

    def _ensure_loaded(self) -> dict[str, dict[str, Any]]:
        if self._token2vad is None:
            with self.token2vad_path.open("rb") as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, dict):
                raise RuntimeError(f"token2vad file must contain a dict, got {type(loaded)!r}")
            self._token2vad = loaded
        return self._token2vad

    @staticmethod
    def _gt_to_env_xy(gt_ego_fut_trajs: np.ndarray, *, cumulative: bool = True) -> np.ndarray:
        gt = np.asarray(gt_ego_fut_trajs, dtype=np.float32)
        if gt.ndim != 2 or gt.shape[1] < 2:
            raise RuntimeError(f"Expected gt_ego_fut_trajs with shape (T, 2+), got {gt.shape}")
        # token2vad stores local future as (lateral, forward) step deltas.
        # Policy candidates are cumulative future waypoints in (forward, left), so convert here.
        out = np.zeros((gt.shape[0], 2), dtype=np.float32)
        out[:, 0] = gt[:, 1]
        out[:, 1] = gt[:, 0]
        if bool(cumulative) and int(out.shape[0]) > 0:
            out = np.cumsum(out, axis=0, dtype=np.float32)
        return out

    @staticmethod
    def _normalize_traj_valid_mask(mask_like: Any, expected_len: int) -> np.ndarray | None:
        if mask_like is None:
            return None
        mask_arr = np.asarray(mask_like).reshape(-1)
        if mask_arr.size <= 0:
            return None
        mask_bool = (mask_arr[:expected_len] > 0).astype(bool, copy=False)
        if mask_bool.size <= 0 or int(np.count_nonzero(mask_bool)) <= 0:
            return None
        return mask_bool

    @classmethod
    def _sanitize_local_traj_xy(
        cls,
        traj_any: Any,
        *,
        valid_mask_any: Any = None,
        min_trailing_zero_pad: int = 2,
    ) -> np.ndarray:
        traj = np.asarray(traj_any, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float32)
        xy = traj[:, :2]
        finite = np.isfinite(xy).all(axis=1)
        if int(np.count_nonzero(finite)) <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        valid_mask = cls._normalize_traj_valid_mask(valid_mask_any, expected_len=int(xy.shape[0]))
        if valid_mask is not None:
            keep = finite & valid_mask
            if int(np.count_nonzero(keep)) <= 0:
                return np.zeros((0, 2), dtype=np.float32)
            return xy[keep].astype(np.float32, copy=False)

        finite_idx = np.where(finite)[0]
        xy_finite = xy[: int(finite_idx[-1]) + 1]
        nonzero = np.linalg.norm(xy_finite, axis=1) > 1.0e-6
        if int(np.count_nonzero(nonzero)) <= 0:
            return xy_finite.astype(np.float32, copy=False)

        last_nonzero = int(np.where(nonzero)[0][-1])
        trailing = int(xy_finite.shape[0]) - (last_nonzero + 1)
        if trailing >= int(min_trailing_zero_pad):
            return xy_finite[: last_nonzero + 1].astype(np.float32, copy=False)
        return xy_finite.astype(np.float32, copy=False)

    def _lookup_row(self, sample_token: str) -> dict[str, Any]:
        token2vad = self._ensure_loaded()
        row = token2vad.get(str(sample_token), None)
        if not isinstance(row, dict):
            raise RuntimeError(f"sample_token={sample_token!r} not found in token2vad index")
        return row

    def _lookup_gt(self, sample_token: str) -> np.ndarray:
        row = self._lookup_row(sample_token)
        gt = row.get("gt_ego_fut_trajs", None)
        if gt is None:
            raise RuntimeError(f"sample_token={sample_token!r} missing gt_ego_fut_trajs")
        gt_local = self._sanitize_local_traj_xy(
            gt,
            valid_mask_any=row.get("gt_ego_fut_masks", row.get("gt_ego_fut_mask", row.get("gt_ego_fut_valid", None))),
        )
        return self._gt_to_env_xy(gt_local, cumulative=True)

    @staticmethod
    def _gt_sample_token_for_replay(replay: Mapping[str, Any], sample_token: str) -> str:
        for key in ("gt_sample_token_override", "grpo_gt_sample_token"):
            value = replay.get(key, None)
            if value is not None and str(value):
                return str(value)
        return str(sample_token)

    @staticmethod
    def _rebase_xy(points_xy: np.ndarray, origin_xy: np.ndarray | None) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float32)
        if origin_xy is None:
            return pts.copy()
        origin = np.asarray(origin_xy, dtype=np.float32).reshape(1, 2)
        return pts - origin

    @staticmethod
    def _quat_to_rotmat_xy(rotation_wxyz: Sequence[float]) -> np.ndarray:
        quat = np.asarray(list(rotation_wxyz), dtype=np.float32).reshape(-1)
        if quat.shape[0] < 4:
            return np.eye(2, dtype=np.float32)
        w, x, y, z = [float(val) for val in quat[:4]]
        rot = np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z)],
            ],
            dtype=np.float32,
        )
        return rot

    def _global_to_local_xy(self, points_xy: np.ndarray, row: dict[str, Any]) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float32)
        translation = np.asarray(row.get("ego2global_translation", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        rot = self._quat_to_rotmat_xy(row.get("ego2global_rotation", [1.0, 0.0, 0.0, 0.0]))
        delta = pts[:, :2] - translation[:2].reshape(1, 2)
        return delta @ rot

    def _lookup_gt_history(self, row: dict[str, Any], *, origin_xy: np.ndarray | None) -> np.ndarray:
        history = row.get("gt_ego_his_trajs", None)
        if history is None:
            return np.zeros((0, 2), dtype=np.float32)
        history_local = self._sanitize_local_traj_xy(
            history,
            valid_mask_any=row.get("gt_ego_his_masks", row.get("gt_ego_his_mask", row.get("gt_ego_his_valid", None))),
        )
        history_xy = self._gt_to_env_xy(history_local, cumulative=False)
        return self._rebase_xy(history_xy, origin_xy)

    @staticmethod
    def _normalize_agent_future_traj(agent_traj_any: Any) -> np.ndarray:
        arr = np.asarray(agent_traj_any, dtype=np.float32)
        if arr.ndim == 1:
            if int(arr.size) < 2 or int(arr.size) % 2 != 0:
                return np.zeros((0, 2), dtype=np.float32)
            arr = arr.reshape(-1, 2)
        if arr.ndim != 2 or int(arr.shape[1]) < 2:
            return np.zeros((0, 2), dtype=np.float32)
        return arr[:, :2].astype(np.float32, copy=False)

    @staticmethod
    def _interp_angle(time_s: float, times_s: np.ndarray, yaw_rad: np.ndarray) -> float:
        if int(times_s.shape[0]) <= 0 or int(yaw_rad.shape[0]) <= 0:
            return 0.0
        yaw_unwrapped = np.unwrap(np.asarray(yaw_rad, dtype=np.float64))
        return float(np.interp(float(time_s), np.asarray(times_s, dtype=np.float64), yaw_unwrapped))

    @staticmethod
    def _sample_state_at_time(
        *,
        current_state: dict[str, Any],
        future_xy: np.ndarray,
        future_yaw: np.ndarray | None,
        time_s: float,
        dt_s: float,
    ) -> dict[str, Any] | None:
        query_t = float(time_s)
        if query_t <= 0.0:
            return dict(current_state)

        future_xy_arr = np.asarray(future_xy, dtype=np.float32)
        if future_xy_arr.ndim != 2 or int(future_xy_arr.shape[0]) <= 0 or int(future_xy_arr.shape[1]) < 2:
            return None

        sample_dt = max(1.0e-6, float(dt_s))
        times = np.arange(1, int(future_xy_arr.shape[0]) + 1, dtype=np.float64) * sample_dt
        if query_t > float(times[-1]) + 1.0e-6:
            return None

        current_xy = np.asarray(
            [float(current_state.get("x", 0.0)), float(current_state.get("y", 0.0))],
            dtype=np.float32,
        ).reshape(1, 2)
        series_xy = np.concatenate([current_xy, future_xy_arr[:, :2]], axis=0).astype(np.float64, copy=False)
        series_times = np.concatenate([np.asarray([0.0], dtype=np.float64), times], axis=0)
        right_idx = int(np.searchsorted(series_times, query_t, side="right"))
        seg_hi = min(max(1, right_idx), int(series_times.shape[0]) - 1)
        seg_lo = max(0, seg_hi - 1)
        t0 = float(series_times[seg_lo])
        t1 = float(series_times[seg_hi])
        alpha = 0.0 if t1 <= t0 + 1.0e-9 else float((query_t - t0) / (t1 - t0))
        p0 = series_xy[seg_lo]
        p1 = series_xy[seg_hi]
        interp_xy = (1.0 - alpha) * p0 + alpha * p1
        x = float(interp_xy[0])
        y = float(interp_xy[1])
        seg_dt = max(1.0e-6, t1 - t0)
        speed = float(np.linalg.norm(p1 - p0) / seg_dt)

        yaw_series = np.asarray(future_yaw if future_yaw is not None else [], dtype=np.float32).reshape(-1)
        if int(yaw_series.shape[0]) == int(future_xy_arr.shape[0]):
            series_yaw = np.concatenate(
                [np.asarray([float(current_state.get("yaw_rad", 0.0))], dtype=np.float64), yaw_series.astype(np.float64)],
                axis=0,
            )
        else:
            derived_yaw = _path_yaw_from_xy(series_xy.astype(np.float32))
            series_yaw = np.asarray(derived_yaw, dtype=np.float64)
            if int(series_yaw.shape[0]) > 0:
                series_yaw[0] = float(current_state.get("yaw_rad", series_yaw[0]))
        yaw_lo = float(series_yaw[seg_lo]) if seg_lo < int(series_yaw.shape[0]) else float(current_state.get("yaw_rad", 0.0))
        yaw_hi = float(series_yaw[seg_hi]) if seg_hi < int(series_yaw.shape[0]) else yaw_lo
        yaw_delta = float(math.atan2(math.sin(yaw_hi - yaw_lo), math.cos(yaw_hi - yaw_lo)))
        yaw = float(yaw_lo + alpha * yaw_delta)
        sampled_yaw_rate = float(yaw_delta / seg_dt)

        return {
            **current_state,
            "x": float(x),
            "y": float(y),
            "speed_mps": float(speed),
            "yaw_rad": float(math.atan2(math.sin(yaw), math.cos(yaw))),
            "yaw_rate_rps": float(sampled_yaw_rate),
        }

    def _lookup_ea_agent_future_truth(
        self,
        row: dict[str, Any],
        *,
        patch_radius: float,
    ) -> list[dict[str, Any]]:
        boxes = np.asarray(row.get("gt_boxes", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32)
        if boxes.ndim != 2 or int(boxes.shape[0]) <= 0:
            return []
        velocities = np.asarray(row.get("gt_velocity", np.zeros((boxes.shape[0], 2), dtype=np.float32)), dtype=np.float32)
        names = np.asarray(row.get("gt_names", np.asarray([], dtype=object)))
        valid = np.asarray(row.get("valid_flag", np.ones((boxes.shape[0],), dtype=bool)))
        fut_trajs = np.asarray(row.get("gt_agent_fut_trajs", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        fut_masks = np.asarray(row.get("gt_agent_fut_masks", np.zeros((0,), dtype=np.float32)))
        fut_yaw = np.asarray(row.get("gt_agent_fut_yaw", np.zeros((0,), dtype=np.float32)), dtype=np.float32)

        out: list[dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            if idx < valid.shape[0] and not bool(valid[idx]):
                continue
            center = np.asarray(box[:2], dtype=np.float32).reshape(-1)
            if center.size < 2 or float(np.linalg.norm(center[:2])) > float(patch_radius) * 1.8:
                continue
            velocity_xy = velocities[idx, :2].astype(np.float32) if idx < velocities.shape[0] else np.zeros((2,), dtype=np.float32)
            item: dict[str, Any] = {
                "category": str(names[idx]) if idx < names.shape[0] else "unknown",
                "center_xy": center[:2].astype(np.float32),
                "velocity_xy": velocity_xy.reshape(2),
                "yaw_rad": float(box[6]) if box.shape[0] > 6 else 0.0,
                "speed_mps": float(np.linalg.norm(velocity_xy)),
                "length_m": float(abs(box[3])) if box.shape[0] > 3 else 1.0,
                "width_m": float(abs(box[4])) if box.shape[0] > 4 else 1.0,
            }
            if fut_trajs.ndim >= 2 and idx < fut_trajs.shape[0]:
                traj_local = self._sanitize_local_traj_xy(
                    self._normalize_agent_future_traj(fut_trajs[idx]),
                    valid_mask_any=fut_masks[idx] if fut_masks.ndim >= 2 and idx < fut_masks.shape[0] else None,
                )
                if int(traj_local.shape[0]) > 0:
                    item["future_xy"] = (self._gt_to_env_xy(traj_local, cumulative=True) + center[:2].reshape(1, 2)).astype(np.float32, copy=False)
                    item["future_dt_s"] = 0.5
                    if fut_yaw.ndim >= 2 and idx < fut_yaw.shape[0]:
                        yaw_series = np.asarray(fut_yaw[idx], dtype=np.float32).reshape(-1)
                        if int(yaw_series.shape[0]) >= int(traj_local.shape[0]):
                            item["future_yaw"] = yaw_series[: int(traj_local.shape[0])].astype(np.float32, copy=False)
            out.append(item)
        return out

    @staticmethod
    def _box_corners_xy(center_x: float, center_y: float, length: float, width: float, yaw: float) -> np.ndarray:
        dx = float(length) * 0.5
        dy = float(width) * 0.5
        corners = np.asarray(
            [
                [dx, dy],
                [dx, -dy],
                [-dx, -dy],
                [-dx, dy],
            ],
            dtype=np.float32,
        )
        c = math.cos(float(yaw))
        s = math.sin(float(yaw))
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        return corners @ rot.T + np.asarray([[float(center_x), float(center_y)]], dtype=np.float32)

    def _extract_scene_objects(self, row: dict[str, Any], *, patch_radius: float) -> list[dict[str, Any]]:
        boxes = np.asarray(row.get("gt_boxes", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[0] <= 0:
            return []
        names = np.asarray(row.get("gt_names", np.asarray([], dtype=object)))
        velocities = np.asarray(row.get("gt_velocity", np.zeros((boxes.shape[0], 2), dtype=np.float32)), dtype=np.float32)
        valid = np.asarray(row.get("valid_flag", np.ones((boxes.shape[0],), dtype=bool)))
        lidar_pts = np.asarray(row.get("num_lidar_pts", np.zeros((boxes.shape[0],), dtype=np.int64)))
        radar_pts = np.asarray(row.get("num_radar_pts", np.zeros((boxes.shape[0],), dtype=np.int64)))
        objects: list[dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            if idx < valid.shape[0] and not bool(valid[idx]):
                continue
            x = float(box[0])
            y = float(box[1])
            if max(abs(x), abs(y)) > float(patch_radius) * 1.2 and math.hypot(x, y) > float(patch_radius) * 1.4:
                continue
            length = float(abs(box[3])) if box.shape[0] > 3 else 1.0
            width = float(abs(box[4])) if box.shape[0] > 4 else 1.0
            yaw = float(box[6]) if box.shape[0] > 6 else 0.0
            speed = float(np.linalg.norm(velocities[idx])) if idx < velocities.shape[0] else 0.0
            category = str(names[idx]) if idx < names.shape[0] else "unknown"
            corners = self._box_corners_xy(x, y, length, width, yaw)
            objects.append(
                {
                    "category": category,
                    "center_xy": [x, y],
                    "length_m": length,
                    "width_m": width,
                    "yaw_rad": yaw,
                    "speed_mps": speed,
                    "num_lidar_pts": int(lidar_pts[idx]) if idx < lidar_pts.shape[0] else 0,
                    "num_radar_pts": int(radar_pts[idx]) if idx < radar_pts.shape[0] else 0,
                    "corners_xy": corners.astype(np.float32).tolist(),
                }
            )
        return objects

    def _ensure_nusc(self) -> Any | None:
        if self._nusc is not None:
            return self._nusc
        if self.nuscenes_dataroot is None:
            return None
        try:
            from nuscenes.nuscenes import NuScenes

            self._nusc = NuScenes(version=self.nuscenes_version, dataroot=str(self.nuscenes_dataroot), verbose=False)
        except Exception:
            self._nusc = None
        return self._nusc

    def _lookup_map_layers(self, row: dict[str, Any], *, patch_radius: float = _DEFAULT_PATCH_RADIUS_M) -> dict[str, Any]:
        nusc = self._ensure_nusc()
        if nusc is None:
            return {"patch_radius": float(patch_radius), "layers": {}}
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
        except Exception:
            return {"patch_radius": float(patch_radius), "layers": {}}

        location = row.get("map_location", None)
        if location is None:
            sample_token = str(row.get("token", ""))
            if sample_token == "":
                return {"patch_radius": float(patch_radius), "layers": {}}
            try:
                sample_record = nusc.get("sample", sample_token)
                scene_record = nusc.get("scene", sample_record["scene_token"])
                log_record = nusc.get("log", scene_record["log_token"])
                location = log_record["location"]
            except Exception:
                return {"patch_radius": float(patch_radius), "layers": {}}
        location = str(location)
        if location not in self._map_cache:
            try:
                self._map_cache[location] = NuScenesMap(dataroot=str(self.nuscenes_dataroot), map_name=location)
            except Exception:
                self._map_cache[location] = None
        map_api = self._map_cache.get(location, None)
        if map_api is None:
            return {"patch_radius": float(patch_radius), "layers": {}}

        ego_translation = np.asarray(row.get("ego2global_translation", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        x = float(ego_translation[0]) if ego_translation.shape[0] > 0 else 0.0
        y = float(ego_translation[1]) if ego_translation.shape[0] > 1 else 0.0
        layers: dict[str, list[list[list[float]]]] = {}
        try:
            records = map_api.get_records_in_radius(
                x,
                y,
                float(patch_radius),
                list(_MAP_POLYGON_LAYERS) + list(_MAP_LINE_LAYERS),
                mode="intersect",
            )
        except Exception:
            return {"patch_radius": float(patch_radius), "layers": {}}

        for layer in _MAP_POLYGON_LAYERS:
            geometries: list[list[list[float]]] = []
            for token in records.get(layer, []):
                try:
                    record = map_api.get(layer, token)
                    polygon_tokens = record.get("polygon_tokens", None) if layer == "drivable_area" else [record.get("polygon_token", None)]
                    for polygon_token in polygon_tokens or []:
                        if polygon_token is None:
                            continue
                        polygon = map_api.extract_polygon(polygon_token)
                        if polygon is None:
                            continue
                        coords = np.asarray(polygon.exterior.coords, dtype=np.float32)
                        local = self._global_to_local_xy(coords[:, :2], row)
                        if local.shape[0] >= 3:
                            geometries.append(local.astype(np.float32).tolist())
                except Exception:
                    continue
            if len(geometries) > 0:
                layers[layer] = geometries

        for layer in _MAP_LINE_LAYERS:
            geometries = []
            for token in records.get(layer, []):
                try:
                    record = map_api.get(layer, token)
                    line_token = record.get("line_token", None)
                    if line_token is None:
                        continue
                    line = map_api.extract_line(line_token)
                    if line is None:
                        continue
                    coords = np.asarray(line.coords, dtype=np.float32)
                    local = self._global_to_local_xy(coords[:, :2], row)
                    if local.shape[0] >= 2:
                        geometries.append(local.astype(np.float32).tolist())
                except Exception:
                    continue
            if len(geometries) > 0:
                layers[layer] = geometries

        centerlines: list[list[list[float]]] = []
        lane_tokens = list(records.get("lane", [])) + list(records.get("lane_connector", []))
        if len(lane_tokens) > 0:
            try:
                raw_centerlines = map_api.discretize_lanes(lane_tokens, 1.0)
                if isinstance(raw_centerlines, dict):
                    iter_centerlines = raw_centerlines.values()
                else:
                    iter_centerlines = raw_centerlines
                for centerline in iter_centerlines:
                    coords = np.asarray(centerline, dtype=np.float32)
                    if coords.ndim == 2 and coords.shape[0] >= 2:
                        local = self._global_to_local_xy(coords[:, :2], row)
                        centerlines.append(local.astype(np.float32).tolist())
            except Exception:
                pass
        if len(centerlines) > 0:
            layers["lane_centerline"] = centerlines

        return {"patch_radius": float(patch_radius), "layers": layers}

    def _scene_env_cache_path(self, scene_id: int) -> Path:
        return self.scene_cache_root / f"{int(scene_id):03d}" / "env_cache.json"

    def _load_scene_env_cache(self, scene_id: int) -> dict[int, dict[str, Any]]:
        scene_key = int(scene_id)
        cached = self._scene_env_cache.get(scene_key, None)
        if cached is not None:
            return cached
        path = self._scene_env_cache_path(scene_key)
        if not path.exists():
            self._scene_env_cache[scene_key] = {}
            return self._scene_env_cache[scene_key]
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and "meta" in loaded:
            loaded = {key: value for key, value in loaded.items() if key != "meta"}
        cache = {int(key): value for key, value in dict(loaded).items()}
        self._scene_env_cache[scene_key] = cache
        return cache

    def _scene_agent_state_cache_path(self, scene_id: int) -> Path:
        return self.agent_state_cache_root / f"{int(scene_id):03d}" / "agent_state_cache.json"

    def _load_scene_agent_state_cache(self, scene_id: int) -> dict[str, Any]:
        scene_key = int(scene_id)
        cached = self._scene_agent_state_cache.get(scene_key, None)
        if cached is not None:
            return cached
        path = self._scene_agent_state_cache_path(scene_key)
        if not path.exists():
            empty = {"meta": {}, "frames": {}}
            self._scene_agent_state_cache[scene_key] = empty
            return empty
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        meta = dict(loaded.get("meta", {})) if isinstance(loaded, dict) else {}
        frames = {
            int(key): value
            for key, value in dict(loaded).items()
            if key != "meta"
        }
        payload = {"meta": meta, "frames": frames}
        self._scene_agent_state_cache[scene_key] = payload
        return payload

    def _lookup_scene_snapshot(self, replay: dict[str, Any]) -> dict[str, Any] | None:
        scene_id = replay.get("scene_id", None)
        frame_idx = replay.get("frame_idx", None)
        if scene_id is None or frame_idx is None:
            return None
        try:
            cache = self._load_scene_env_cache(int(scene_id))
        except Exception:
            return None
        return cache.get(int(frame_idx), None)

    def _lookup_agent_state_snapshot(self, replay: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        scene_id = replay.get("scene_id", None)
        frame_idx = replay.get("frame_idx", None)
        if scene_id is None or frame_idx is None:
            return None, {}
        try:
            payload = self._load_scene_agent_state_cache(int(scene_id))
        except Exception:
            return None, {}
        frames = dict(payload.get("frames", {}))
        if not frames:
            return None, dict(payload.get("meta", {}))
        fidx = int(frame_idx)
        if fidx in frames:
            return frames.get(fidx), dict(payload.get("meta", {}))
        keys = sorted(frames.keys())
        for key in reversed(keys):
            if key <= fidx:
                return frames.get(key), dict(payload.get("meta", {}))
        return None, dict(payload.get("meta", {}))

    @staticmethod
    def _global_xy_to_snapshot_local(points_xy: np.ndarray, snapshot: dict[str, Any]) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float32)
        ego_pose = dict(snapshot.get("ego_pose", {}))
        center = np.asarray([ego_pose.get("x", 0.0), ego_pose.get("y", 0.0)], dtype=np.float32)
        yaw = float(ego_pose.get("yaw", 0.0))
        delta = pts[:, :2] - center.reshape(1, 2)
        c = math.cos(-yaw)
        s = math.sin(-yaw)
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        return delta @ rot.T

    @staticmethod
    def _rotate_world_vec_to_snapshot_local(vec_xy: np.ndarray, snapshot: dict[str, Any]) -> np.ndarray:
        vec = np.asarray(vec_xy, dtype=np.float32).reshape(-1)
        if vec.size < 2:
            return np.zeros((2,), dtype=np.float32)
        ego_pose = dict(snapshot.get("ego_pose", {}))
        yaw = float(ego_pose.get("yaw", 0.0))
        c = math.cos(-yaw)
        s = math.sin(-yaw)
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        return (np.asarray([[float(vec[0]), float(vec[1])]], dtype=np.float32) @ rot.T).reshape(2)

    @classmethod
    def _agent_state_to_snapshot_local(cls, agent_state: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
        center_local = cls._global_xy_to_snapshot_local(
            np.asarray([agent_state.get("center_xy", [0.0, 0.0])], dtype=np.float32),
            snapshot,
        )
        velocity_local = cls._rotate_world_vec_to_snapshot_local(
            np.asarray(agent_state.get("velocity_xy", [0.0, 0.0]), dtype=np.float32),
            snapshot,
        )
        ego_pose = dict(snapshot.get("ego_pose", {}))
        ego_yaw = float(ego_pose.get("yaw", 0.0))
        yaw_local = float(np.arctan2(np.sin(float(agent_state.get("yaw_rad", 0.0)) - ego_yaw), np.cos(float(agent_state.get("yaw_rad", 0.0)) - ego_yaw)))
        return {
            **agent_state,
            "center_xy": center_local.reshape(2).astype(np.float32),
            "velocity_xy": velocity_local.astype(np.float32),
            "yaw_rad": yaw_local,
        }

    def _collect_ea_agent_states(
        self,
        replay: dict[str, Any],
        *,
        patch_radius: float,
        row: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.ea_gate_enabled:
            return []
        if isinstance(row, dict):
            truth_states = self._lookup_ea_agent_future_truth(row, patch_radius=float(patch_radius))
            if len(truth_states) > 0:
                return truth_states
        snapshot, meta = self._lookup_agent_state_snapshot(replay)
        if not isinstance(snapshot, dict):
            return []
        agents = list(snapshot.get("agents", []) or [])
        if len(agents) <= 0:
            return []
        coordinate_frame = str(meta.get("coordinate_frame", "")).strip().lower()
        scene_snapshot = self._lookup_scene_snapshot(replay)
        if coordinate_frame == "world" and not isinstance(scene_snapshot, dict):
            return []
        out: list[dict[str, Any]] = []
        for item in agents:
            agent_state = dict(item)
            if coordinate_frame == "world" and isinstance(scene_snapshot, dict):
                agent_state = self._agent_state_to_snapshot_local(agent_state, scene_snapshot)
            center = np.asarray(agent_state.get("center_xy", [0.0, 0.0]), dtype=np.float32).reshape(-1)
            if center.size < 2:
                continue
            if float(np.linalg.norm(center[:2])) > float(patch_radius) * 1.8:
                continue
            agent_state["center_xy"] = center[:2].astype(np.float32)
            agent_state["velocity_xy"] = np.asarray(agent_state.get("velocity_xy", [0.0, 0.0]), dtype=np.float32).reshape(2)
            out.append(agent_state)
        return out

    def _lookup_cached_map_layers(
        self,
        replay: dict[str, Any],
        *,
        patch_radius: float = _DEFAULT_PATCH_RADIUS_M,
    ) -> dict[str, Any] | None:
        snapshot = self._lookup_scene_snapshot(replay)
        if snapshot is None:
            return None

        layers: dict[str, list[list[list[float]]]] = {}
        drivable = []
        for polygon in snapshot.get("drivable_polygons", []) or []:
            local = self._global_xy_to_snapshot_local(np.asarray(polygon, dtype=np.float32), snapshot)
            if local.shape[0] >= 3:
                drivable.append(local.astype(np.float32).tolist())
        if drivable:
            layers["drivable_area"] = drivable

        centerlines = []
        for line in snapshot.get("lanes_centerlines", []) or []:
            local = self._global_xy_to_snapshot_local(np.asarray(line, dtype=np.float32), snapshot)
            if local.shape[0] >= 2:
                centerlines.append(local.astype(np.float32).tolist())
        if centerlines:
            layers["lane_centerline"] = centerlines

        return {"patch_radius": float(patch_radius), "layers": layers}

    @staticmethod
    def _safe_float_list(values: Any) -> list[float]:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        return [float(item) for item in arr.tolist()]

    @staticmethod
    def _sample_context_cache_filename(sample_token: str, *, cache_variant: str = "default") -> str:
        token = str(sample_token).strip()
        safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", token)[:64] or "sample"
        digest = hashlib.sha1(f"{token}|{str(cache_variant).strip()}".encode("utf-8")).hexdigest()[:16]
        return f"{safe_prefix}_{digest}.pkl"

    def _sample_context_cache_path(self, sample_token: str, *, cache_variant: str = "default") -> Path:
        return self.sample_context_cache_root / self._sample_context_cache_filename(sample_token, cache_variant=cache_variant)

    def _sample_context_cache_variant(self) -> str:
        return f"box-lw-v2-ea-{int(bool(self.ea_gate_enabled))}"

    def _load_persisted_static_sample_context(self, sample_token: str) -> dict[str, Any] | None:
        path = self._sample_context_cache_path(sample_token, cache_variant=self._sample_context_cache_variant())
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _save_persisted_static_sample_context(self, sample_token: str, payload: dict[str, Any]) -> None:
        path = self._sample_context_cache_path(sample_token, cache_variant=self._sample_context_cache_variant())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=f"{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(tmp_fd, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path_str, path)
        except Exception:
            try:
                os.unlink(tmp_path_str)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _point_in_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
        point = np.asarray(point_xy, dtype=np.float32).reshape(-1)
        polygon = np.asarray(polygon_xy, dtype=np.float32)
        if point.shape[0] < 2 or polygon.ndim != 2 or polygon.shape[0] < 3:
            return False
        if ShapelyPolygon is not None and ShapelyPoint is not None:
            try:
                poly = ShapelyPolygon(polygon[:, :2])
                return bool(poly.buffer(1.0e-6).contains(ShapelyPoint(float(point[0]), float(point[1]))))
            except Exception:
                pass

        inside = False
        x = float(point[0])
        y = float(point[1])
        for idx in range(int(polygon.shape[0])):
            x0, y0 = [float(item) for item in polygon[idx - 1, :2]]
            x1, y1 = [float(item) for item in polygon[idx, :2]]
            intersects = ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / max(1.0e-8, (y1 - y0)) + x0)
            if intersects:
                inside = not inside
        return bool(inside)

    @classmethod
    def _point_in_polygons(cls, point_xy: np.ndarray, polygons: Sequence[Any]) -> bool:
        for polygon in polygons or []:
            if cls._point_in_polygon(point_xy, np.asarray(polygon, dtype=np.float32)):
                return True
        return False

    @staticmethod
    def _nearest_centerline_stats(
        point_xy: np.ndarray,
        centerlines: Sequence[Any],
        *,
        direction_xy: np.ndarray | None = None,
        same_dir_dot_threshold: float = 0.2,
        same_dir_distance_margin_m: float = 0.75,
    ) -> tuple[float, np.ndarray]:
        point = np.asarray(point_xy, dtype=np.float32).reshape(2)
        best_dist = float("inf")
        best_tangent = np.asarray([1.0, 0.0], dtype=np.float32)
        same_dir_dist = float("inf")
        same_dir_tangent = np.asarray([1.0, 0.0], dtype=np.float32)
        direction = None
        if direction_xy is not None:
            direction = np.asarray(direction_xy, dtype=np.float32).reshape(2)
            direction_norm = float(np.linalg.norm(direction))
            direction = direction / direction_norm if direction_norm > 1.0e-6 else None
        for line in centerlines or []:
            pts = np.asarray(line, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            for idx in range(int(pts.shape[0]) - 1):
                p0 = pts[idx, :2]
                p1 = pts[idx + 1, :2]
                seg = p1 - p0
                seg_len_sq = float(np.dot(seg, seg))
                if seg_len_sq <= 1.0e-8:
                    continue
                alpha = float(np.dot(point - p0, seg) / seg_len_sq)
                alpha = max(0.0, min(1.0, alpha))
                proj = p0 + alpha * seg
                dist = float(np.linalg.norm(point - proj))
                norm = float(np.linalg.norm(seg))
                tangent = (seg / norm).astype(np.float32) if norm > 1.0e-8 else best_tangent
                if dist < best_dist:
                    best_dist = dist
                    best_tangent = tangent
                if direction is not None and float(np.dot(direction, tangent)) >= float(same_dir_dot_threshold) and dist < same_dir_dist:
                    same_dir_dist = dist
                    same_dir_tangent = tangent
        if same_dir_dist < float("inf") and same_dir_dist <= best_dist + max(0.0, float(same_dir_distance_margin_m)):
            return same_dir_dist, same_dir_tangent
        return best_dist, best_tangent

    @classmethod
    def _collect_scene_objects(
        cls,
        row: dict[str, Any],
        *,
        patch_radius: float,
    ) -> list[dict[str, Any]]:
        boxes = np.asarray(row.get("gt_boxes", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[0] <= 0:
            return []
        velocities = np.asarray(row.get("gt_velocity", np.zeros((boxes.shape[0], 2), dtype=np.float32)), dtype=np.float32)
        names = np.asarray(row.get("gt_names", np.asarray([], dtype=object)))
        valid = np.asarray(row.get("valid_flag", np.ones((boxes.shape[0],), dtype=bool)))
        lidar_pts = np.asarray(row.get("num_lidar_pts", np.zeros((boxes.shape[0],), dtype=np.int64)))
        radar_pts = np.asarray(row.get("num_radar_pts", np.zeros((boxes.shape[0],), dtype=np.int64)))
        objects: list[dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            if idx < valid.shape[0] and not bool(valid[idx]):
                continue
            center = np.asarray(box[:2], dtype=np.float32)
            if float(np.linalg.norm(center)) > float(patch_radius) * 1.5:
                continue
            length = float(abs(box[3])) if box.shape[0] > 3 else 1.0
            width = float(abs(box[4])) if box.shape[0] > 4 else 1.0
            yaw = float(box[6]) if box.shape[0] > 6 else 0.0
            category = str(names[idx]) if idx < names.shape[0] else "unknown"
            velocity_xy = velocities[idx, :2].astype(np.float32) if idx < velocities.shape[0] else np.zeros((2,), dtype=np.float32)
            corners = cls._box_corners_xy(center[0], center[1], length, width, yaw).astype(np.float32)
            objects.append(
                {
                    "category": category,
                    "center_xy": center.astype(np.float32),
                    "velocity_xy": velocity_xy,
                    "length_m": length,
                    "width_m": width,
                    "yaw_rad": yaw,
                    "speed_mps": float(np.linalg.norm(velocity_xy)),
                    "num_lidar_pts": int(lidar_pts[idx]) if idx < lidar_pts.shape[0] else 0,
                    "num_radar_pts": int(radar_pts[idx]) if idx < radar_pts.shape[0] else 0,
                    "corners_xy": corners,
                }
            )
        return objects

    @staticmethod
    def _has_replay_object_context_override(replay: dict[str, Any]) -> bool:
        return any(
            key in replay
            for key in (
                "scene_objects_override",
                "ea_agent_states_override",
                "ttc_agent_states_override",
            )
        )

    @staticmethod
    def _poly_heading_length_width(poly_xy: np.ndarray) -> tuple[float, float, float]:
        poly = np.asarray(poly_xy, dtype=np.float32)
        if poly.ndim != 2 or poly.shape[0] < 2 or poly.shape[1] < 2:
            return 0.0, 1.0, 1.0
        if poly.shape[0] >= 4:
            front_mid = 0.5 * (poly[0, :2] + poly[1, :2])
            rear_mid = 0.5 * (poly[2, :2] + poly[3, :2])
            heading = front_mid - rear_mid
            yaw = float(math.atan2(float(heading[1]), float(heading[0]))) if float(np.linalg.norm(heading)) > 1.0e-9 else 0.0
            length = float(max(1.0e-6, np.linalg.norm(front_mid - rear_mid)))
            width = float(max(1.0e-6, np.linalg.norm(poly[0, :2] - poly[1, :2])))
            return yaw, length, width
        delta = poly[1, :2] - poly[0, :2]
        yaw = float(math.atan2(float(delta[1]), float(delta[0]))) if float(np.linalg.norm(delta)) > 1.0e-9 else 0.0
        return yaw, float(max(1.0e-6, np.linalg.norm(delta))), 1.0

    @classmethod
    def _normalize_replay_object_context(
        cls,
        objects: Any,
        *,
        patch_radius: float,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, raw in enumerate([] if objects is None else objects):
            if not isinstance(raw, dict):
                continue
            corners_raw = raw.get("corners_xy", raw.get("poly", None))
            try:
                corners = np.asarray(corners_raw, dtype=np.float32)
            except Exception:
                corners = np.zeros((0, 2), dtype=np.float32)
            center_raw = raw.get("center_xy", None)
            if corners.ndim == 2 and corners.shape[0] >= 3 and corners.shape[1] >= 2:
                corners = corners[:, :2].astype(np.float32, copy=False)
                center = np.mean(corners, axis=0).astype(np.float32)
                yaw, length, width = cls._poly_heading_length_width(corners)
            else:
                try:
                    center = np.asarray(center_raw, dtype=np.float32).reshape(-1)[:2]
                except Exception:
                    center = np.zeros((0,), dtype=np.float32)
                if center.size < 2:
                    continue
                length = float(abs(raw.get("length_m", 1.0)))
                width = float(abs(raw.get("width_m", 1.0)))
                yaw = float(raw.get("yaw_rad", 0.0))
                corners = cls._box_corners_xy(float(center[0]), float(center[1]), length, width, yaw).astype(np.float32)
            if float(np.linalg.norm(center[:2])) > float(patch_radius) * 1.8:
                continue
            velocity_xy = np.asarray(raw.get("velocity_xy", [0.0, 0.0]), dtype=np.float32).reshape(-1)
            if velocity_xy.size < 2:
                velocity_xy = np.zeros((2,), dtype=np.float32)
            obj = {
                "category": str(raw.get("category", "vehicle.car")),
                "center_xy": center[:2].astype(np.float32),
                "velocity_xy": velocity_xy[:2].astype(np.float32),
                "length_m": float(abs(raw.get("length_m", length))),
                "width_m": float(abs(raw.get("width_m", width))),
                "yaw_rad": float(raw.get("yaw_rad", yaw)),
                "speed_mps": float(raw.get("speed_mps", np.linalg.norm(velocity_xy[:2]))),
                "corners_xy": corners[:, :2].astype(np.float32),
                "token": str(raw.get("token", raw.get("id", f"override_obj_{idx}"))),
                "source": str(raw.get("source", "replay_override")),
            }
            out.append(obj)
        return out

    def _build_static_sample_context(
        self,
        replay: dict[str, Any],
        *,
        patch_radius: float,
    ) -> dict[str, Any]:
        sample_token = replay.get("sample_token", None)
        if sample_token is None:
            raise RuntimeError("NuScenesScorerUtils requires replay['sample_token']")
        sample_token_str = str(sample_token)
        gt_sample_token_str = self._gt_sample_token_for_replay(replay, sample_token_str)
        cache_key = (
            sample_token_str
            if gt_sample_token_str == sample_token_str
            else f"{sample_token_str}|gt={gt_sample_token_str}"
        )
        has_object_override = self._has_replay_object_context_override(replay)
        if has_object_override:
            row = self._lookup_row(sample_token_str)
            gt_xy = self._lookup_gt(gt_sample_token_str).copy()
            gt_yaw = _path_yaw_from_xy(gt_xy)
            gt_s = _polyline_arclength(gt_xy)
            gt_total_len = float(max(1.0e-6, gt_s[-1] if int(gt_s.shape[0]) > 0 else 1.0))
            cached_map_context = self._lookup_cached_map_layers(replay, patch_radius=float(patch_radius))
            map_context = (
                cached_map_context
                if cached_map_context is not None
                else self._lookup_map_layers(row, patch_radius=float(patch_radius))
            )
            patch_radius_resolved = float(map_context.get("patch_radius", patch_radius))
            scene_objects = self._normalize_replay_object_context(
                replay.get("scene_objects_override", []),
                patch_radius=patch_radius_resolved,
            )
            if "ea_agent_states_override" in replay:
                ea_agent_states = self._normalize_replay_object_context(
                    replay.get("ea_agent_states_override", []),
                    patch_radius=patch_radius_resolved,
                )
            else:
                ea_agent_states = [dict(item) for item in scene_objects]
            payload = {
                "row": row,
                "gt_xy": gt_xy,
                "gt_yaw": gt_yaw,
                "gt_s": gt_s,
                "gt_total_len": gt_total_len,
                "gt_sample_token": gt_sample_token_str,
                "map_context": map_context,
                "scene_objects": scene_objects,
                "ea_agent_states": ea_agent_states,
                "replay_scoped_object_context": True,
            }
            if "ttc_agent_states_override" in replay:
                payload["ttc_agent_states"] = self._normalize_replay_object_context(
                    replay.get("ttc_agent_states_override", []),
                    patch_radius=patch_radius_resolved,
                )
            return payload

        cached = self._sample_static_context_cache.get(cache_key, None)
        if cached is not None:
            return cached

        persisted = None
        if gt_sample_token_str == sample_token_str:
            persisted = self._load_persisted_static_sample_context(sample_token_str)
        if persisted is not None:
            self._sample_static_context_cache[cache_key] = persisted
            return persisted

        row = self._lookup_row(sample_token_str)
        gt_xy = self._lookup_gt(gt_sample_token_str).copy()
        gt_yaw = _path_yaw_from_xy(gt_xy)
        gt_s = _polyline_arclength(gt_xy)
        gt_total_len = float(max(1.0e-6, gt_s[-1] if int(gt_s.shape[0]) > 0 else 1.0))

        cached_map_context = self._lookup_cached_map_layers(replay, patch_radius=float(patch_radius))
        map_context = (
            cached_map_context
            if cached_map_context is not None
            else self._lookup_map_layers(row, patch_radius=float(patch_radius))
        )
        patch_radius_resolved = float(map_context.get("patch_radius", patch_radius))
        scene_objects = self._collect_scene_objects(row, patch_radius=patch_radius_resolved)
        ea_agent_states = self._collect_ea_agent_states(
            replay,
            patch_radius=patch_radius_resolved,
            row=row,
        )

        payload = {
            "row": row,
            "gt_xy": gt_xy,
            "gt_yaw": gt_yaw,
            "gt_s": gt_s,
            "gt_total_len": gt_total_len,
            "gt_sample_token": gt_sample_token_str,
            "map_context": map_context,
            "scene_objects": scene_objects,
            "ea_agent_states": ea_agent_states,
        }
        self._sample_static_context_cache[cache_key] = payload
        if gt_sample_token_str == sample_token_str:
            self._save_persisted_static_sample_context(sample_token_str, payload)
        return payload

    def _ensure_ea_compute_fn(self) -> Callable[..., float] | None:
        if self._ea_compute_fn is False:
            return None
        if callable(self._ea_compute_fn):
            return self._ea_compute_fn
        try:
            src = Path(self.ea_project_src)
            if src.exists():
                src_text = str(src)
                if src_text not in sys.path:
                    sys.path.insert(0, src_text)
            from ea_project.core_ea import compute_final_ea

            self._ea_compute_fn = compute_final_ea
            return compute_final_ea
        except Exception:
            self._ea_compute_fn = False
            return None

    def _compute_ea_value_for_pair(self, ego_state: dict[str, Any], agent_state: dict[str, Any]) -> float:
        compute_fn = self._ensure_ea_compute_fn()
        if compute_fn is None:
            raise RuntimeError("EA project is unavailable")
        return float(
            compute_fn(
                xA=float(ego_state["x"]),
                yA=float(ego_state["y"]),
                vA=float(ego_state["speed_mps"]),
                hA=float(ego_state["yaw_rad"]),
                lA=float(ego_state["length_m"]),
                wA=float(ego_state["width_m"]),
                yawA=float(ego_state["yaw_rate_rps"]),
                xB=float(agent_state["x"]),
                yB=float(agent_state["y"]),
                vB=float(agent_state["speed_mps"]),
                hB=float(agent_state["yaw_rad"]),
                lB=float(agent_state["length_m"]),
                wB=float(agent_state["width_m"]),
                yawB=float(agent_state["yaw_rate_rps"]),
                T_total=float(self.ea_gate_horizon_s),
                dt_coarse=float(self.ea_gate_dt_coarse_s),
                dt_fine=float(self.ea_gate_dt_fine_s),
            )
        )

    @staticmethod
    def _propagate_ctrv_state(state: dict[str, Any], *, time_s: float) -> dict[str, Any]:
        t = max(0.0, float(time_s))
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        speed = float(state.get("speed_mps", 0.0))
        yaw = float(state.get("yaw_rad", 0.0))
        yaw_rate = float(state.get("yaw_rate_rps", 0.0))
        if abs(yaw_rate) <= 1.0e-6:
            x = x + speed * math.cos(yaw) * t
            y = y + speed * math.sin(yaw) * t
        else:
            radius = speed / yaw_rate
            delta_yaw = yaw_rate * t
            x = x + radius * (math.sin(yaw + delta_yaw) - math.sin(yaw))
            y = y - radius * (math.cos(yaw + delta_yaw) - math.cos(yaw))
            yaw = yaw + delta_yaw
        return {**state, "x": float(x), "y": float(y), "yaw_rad": float(yaw)}

    @staticmethod
    def _candidate_speed_and_yaw_rate(cand_xy: np.ndarray, cand_yaw: np.ndarray, *, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        if int(cand_xy.shape[0]) <= 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        prev_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), cand_xy[:-1]], axis=0)
        vel_xy = (cand_xy - prev_xy) / max(1.0e-6, float(dt_s))
        speed = np.linalg.norm(vel_xy, axis=1).astype(np.float32, copy=False)
        prev_yaw = np.concatenate([cand_yaw[:1], cand_yaw[:-1]], axis=0)
        yaw_rate = (_wrap_angle(cand_yaw - prev_yaw) / max(1.0e-6, float(dt_s))).astype(np.float32, copy=False)
        return speed, yaw_rate

    def _score_ea_safety_gate(
        self,
        *,
        cand_xy: np.ndarray,
        cand_yaw: np.ndarray,
        agent_states: Sequence[dict[str, Any]],
        dt_s: float,
    ) -> dict[str, float]:
        if not self.ea_gate_enabled:
            return {"gate": 1.0, "max_ea": 0.0, "evaluated_pairs": 0.0}
        if len(agent_states) <= 0:
            return {"gate": 1.0, "max_ea": 0.0, "evaluated_pairs": 0.0}
        if not callable(getattr(self, "_compute_ea_value_for_pair", None)):
            return {"gate": 1.0, "max_ea": 0.0, "evaluated_pairs": 0.0}

        horizon = int(cand_xy.shape[0])
        if horizon <= 0:
            return {"gate": 0.0, "max_ea": 0.0, "evaluated_pairs": 0.0}

        candidate_path = np.asarray(cand_xy, dtype=np.float32)
        ranked_agents = sorted(
            (
                dict(item)
                for item in agent_states
                if "vehicle" in str(item.get("category", "")).strip().lower()
            ),
            key=lambda item: float(
                np.min(np.linalg.norm(candidate_path - np.asarray(item.get("center_xy", [0.0, 0.0]), dtype=np.float32).reshape(1, 2), axis=1))
            ),
        )
        selected_agents = ranked_agents[: self.ea_gate_max_agents]
        if len(selected_agents) <= 0:
            return {"gate": 1.0, "max_ea": 0.0, "evaluated_pairs": 0.0}

        step_indices = sorted({0, max(0, horizon // 2), horizon - 1})
        min_gate = 1.0
        max_ea = 0.0
        pair_count = 0
        for step_idx in step_indices:
            time_offset_s = float(step_idx + 1) * float(dt_s)
            ego_state = self._sample_state_at_time(
                current_state={
                    "x": 0.0,
                    "y": 0.0,
                    "speed_mps": 0.0,
                    "yaw_rad": 0.0,
                    "yaw_rate_rps": 0.0,
                    "length_m": 4.9,
                    "width_m": 2.1,
                },
                future_xy=candidate_path,
                future_yaw=np.asarray(cand_yaw, dtype=np.float32),
                time_s=time_offset_s,
                dt_s=float(dt_s),
            )
            if ego_state is None:
                continue
            for agent in selected_agents:
                agent_velocity = np.asarray(agent.get("velocity_xy", [0.0, 0.0]), dtype=np.float32).reshape(2)
                agent_current_state = {
                    "x": float(np.asarray(agent.get("center_xy", [0.0, 0.0]), dtype=np.float32).reshape(2)[0]),
                    "y": float(np.asarray(agent.get("center_xy", [0.0, 0.0]), dtype=np.float32).reshape(2)[1]),
                    "speed_mps": float(agent.get("speed_mps", np.linalg.norm(agent_velocity))),
                    "yaw_rad": float(agent.get("yaw_rad", 0.0)),
                    "yaw_rate_rps": float(agent.get("yaw_rate_rps", 0.0)),
                    "length_m": float(agent.get("length_m", 1.0)),
                    "width_m": float(agent.get("width_m", 1.0)),
                }
                agent_state = self._sample_state_at_time(
                    current_state=agent_current_state,
                    future_xy=np.asarray(agent.get("future_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32),
                    future_yaw=np.asarray(agent.get("future_yaw", np.zeros((0,), dtype=np.float32)), dtype=np.float32),
                    time_s=time_offset_s,
                    dt_s=float(agent.get("future_dt_s", dt_s)),
                )
                if agent_state is None:
                    agent_state = self._propagate_ctrv_state(
                        agent_current_state,
                        time_s=time_offset_s,
                    )
                try:
                    ea_value = float(self._compute_ea_value_for_pair(ego_state, agent_state))
                except Exception:
                    continue
                pair_count += 1
                if not math.isfinite(ea_value):
                    pair_gate = 0.0
                    max_ea = float("inf")
                else:
                    max_ea = max(max_ea, max(0.0, ea_value))
                    pair_gate = _linear_decay_score(
                        max(0.0, ea_value),
                        good_threshold=float(self.ea_gate_good_threshold),
                        bad_threshold=float(self.ea_gate_bad_threshold),
                    )
                min_gate = min(min_gate, float(pair_gate))
        return {
            "gate": float(min_gate),
            "max_ea": float(max_ea),
            "evaluated_pairs": float(pair_count),
        }

    @staticmethod
    def _category_color(category: str) -> str:
        key = str(category).strip().lower()
        if "pedestrian" in key:
            return "#8e44ad"
        if "bicycle" in key or "motorcycle" in key:
            return "#27ae60"
        if "bus" in key or "truck" in key or "trailer" in key:
            return "#d35400"
        return "#e74c3c"

    @staticmethod
    def _compute_plot_limits(
        *,
        gt_xy: np.ndarray,
        gt_history_xy: np.ndarray,
        candidates: Sequence[dict[str, Any]],
        scene_objects: Sequence[dict[str, Any]],
        patch_radius: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        del gt_xy, gt_history_xy, candidates, scene_objects
        return (-float(patch_radius), float(patch_radius)), (-float(patch_radius), float(patch_radius))

    @staticmethod
    def _candidate_line_style(*, rank: int, total: int) -> dict[str, float]:
        total_count = max(1, int(total))
        rank_idx = max(0, int(rank))
        if rank_idx <= 0:
            alpha = 0.62
            linewidth = 2.3
            marker_size = 26.0
        elif rank_idx == 1:
            alpha = 0.54
            linewidth = 2.1
            marker_size = 23.0
        elif rank_idx == 2:
            alpha = 0.46
            linewidth = 1.9
            marker_size = 20.0
        else:
            tail_total = max(1, total_count - 4)
            tail_rank = min(tail_total, rank_idx - 3)
            tail_fade = float(tail_rank) / float(tail_total)
            alpha = max(0.14, 0.30 - 0.12 * tail_fade)
            linewidth = max(1.05, 1.55 - 0.35 * tail_fade)
            marker_size = max(9.0, 15.0 - 4.5 * tail_fade)
        return {
            "alpha": float(alpha),
            "linewidth": float(linewidth),
            "marker_size": float(marker_size),
        }

    @staticmethod
    def _normalize_polygons(geometries: Sequence[Any]) -> list[list[list[float]]]:
        out: list[list[list[float]]] = []
        for geom in geometries or []:
            arr = np.asarray(geom, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 3:
                out.append(arr[:, :2].astype(np.float32).tolist())
        return out

    @staticmethod
    def _normalize_lines(geometries: Sequence[Any]) -> list[list[list[float]]]:
        out: list[list[list[float]]] = []
        for geom in geometries or []:
            arr = np.asarray(geom, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                out.append(arr[:, :2].astype(np.float32).tolist())
        return out

    @staticmethod
    def _polygon_to_edges(polygons: Sequence[Any]) -> list[list[list[float]]]:
        edges: list[list[list[float]]] = []
        for polygon in polygons or []:
            arr = np.asarray(polygon, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            pts = arr[:, :2]
            closed = np.concatenate([pts, pts[:1]], axis=0)
            edges.append(closed.astype(np.float32).tolist())
        return edges

    @staticmethod
    def _centerline_arrows(lines: Sequence[Any], *, arrow_len: float = 1.8) -> list[list[list[float]]]:
        arrows: list[list[list[float]]] = []
        for line in lines or []:
            pts = np.asarray(line, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            idx = max(0, (pts.shape[0] // 2) - 1)
            start = pts[idx]
            end = pts[idx + 1]
            vec = end - start
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-6:
                continue
            direction = vec / norm
            center = 0.5 * (start + end)
            arrow_start = center - direction * (float(arrow_len) * 0.5)
            arrow_end = center + direction * (float(arrow_len) * 0.5)
            arrows.append([arrow_start.astype(np.float32).tolist(), arrow_end.astype(np.float32).tolist()])
        return arrows

    @classmethod
    def _crossing_stripes(cls, polygons: Sequence[Any]) -> list[list[list[float]]]:
        if ShapelyPolygon is None or shapely_affinity is None:
            return []
        stripes: list[list[list[float]]] = []
        for polygon in polygons or []:
            arr = np.asarray(polygon, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            try:
                poly = ShapelyPolygon(arr[:, :2])
            except Exception:
                continue
            if poly.is_empty or not poly.is_valid or float(poly.area) <= 1e-3:
                continue
            rect = poly.minimum_rotated_rectangle
            coords = np.asarray(rect.exterior.coords[:-1], dtype=np.float32)
            if coords.shape[0] != 4:
                continue
            edges = []
            for idx in range(4):
                p0 = coords[idx]
                p1 = coords[(idx + 1) % 4]
                vec = p1 - p0
                length = float(np.linalg.norm(vec))
                if length > 1e-4:
                    edges.append((length, vec / length))
            if len(edges) < 2:
                continue
            edges.sort(key=lambda item: item[0], reverse=True)
            long_vec = np.asarray(edges[0][1], dtype=np.float32)
            short_vec = np.asarray([-long_vec[1], long_vec[0]], dtype=np.float32)
            centroid = np.asarray(poly.centroid.coords[0], dtype=np.float32)
            centered = arr[:, :2] - centroid.reshape(1, 2)
            long_proj = centered @ long_vec
            short_proj = centered @ short_vec
            long_min = float(np.min(long_proj))
            long_max = float(np.max(long_proj))
            short_min = float(np.min(short_proj))
            short_max = float(np.max(short_proj))
            short_span = short_max - short_min
            if short_span <= 0.2:
                continue
            stripe_width = min(0.6, max(0.25, (long_max - long_min) / 12.0))
            stripe_gap = stripe_width * 0.75
            usable_half = max(0.1, (short_span * 0.5) - 0.08)
            cursor = long_min + stripe_gap * 0.5
            while cursor < long_max:
                center = centroid + long_vec * cursor
                rect = np.asarray(
                    [
                        center + long_vec * (stripe_width * 0.5) + short_vec * usable_half,
                        center + long_vec * (stripe_width * 0.5) - short_vec * usable_half,
                        center - long_vec * (stripe_width * 0.5) - short_vec * usable_half,
                        center - long_vec * (stripe_width * 0.5) + short_vec * usable_half,
                    ],
                    dtype=np.float32,
                )
                try:
                    stripe_poly = ShapelyPolygon(rect)
                    clipped = poly.intersection(stripe_poly)
                except Exception:
                    cursor += stripe_width + stripe_gap
                    continue
                if clipped.is_empty:
                    cursor += stripe_width + stripe_gap
                    continue
                geoms = [clipped] if clipped.geom_type == "Polygon" else list(getattr(clipped, "geoms", []))
                for geom in geoms:
                    try:
                        pts = np.asarray(geom.exterior.coords[:-1], dtype=np.float32)
                    except Exception:
                        continue
                    if pts.ndim == 2 and pts.shape[0] >= 3:
                        stripes.append(pts[:, :2].astype(np.float32).tolist())
                cursor += stripe_width + stripe_gap
        return stripes

    @classmethod
    def _build_render_layers(cls, map_layers: dict[str, Any]) -> dict[str, list[list[list[float]]]]:
        drivable_polygons = cls._normalize_polygons(map_layers.get("drivable_area", []))
        road_surface_polygons = cls._normalize_polygons(
            list(map_layers.get("road_segment", []))
            + list(map_layers.get("road_block", []))
            + list(map_layers.get("lane", []))
            + list(map_layers.get("lane_connector", []))
        )
        if len(road_surface_polygons) == 0:
            road_surface_polygons = list(drivable_polygons)
        lane_fill_polygons = cls._normalize_polygons(
            list(map_layers.get("lane", [])) + list(map_layers.get("lane_connector", []))
        )
        walkway_polygons = cls._normalize_polygons(map_layers.get("walkway", []))
        crossing_polygons = cls._normalize_polygons(map_layers.get("ped_crossing", []))

        road_edge_lines = cls._polygon_to_edges(road_surface_polygons)
        if len(road_edge_lines) == 0:
            road_edge_lines = cls._polygon_to_edges(drivable_polygons)

        lane_marking_lines = cls._normalize_lines(
            list(map_layers.get("lane_divider", [])) + list(map_layers.get("road_divider", []))
        )
        if len(lane_marking_lines) == 0:
            lane_marking_lines = cls._polygon_to_edges(lane_fill_polygons)

        lane_centerlines = cls._normalize_lines(map_layers.get("lane_centerline", []))
        lane_boundary_lines = cls._polygon_to_edges(lane_fill_polygons)
        crossing_stripe_polygons = cls._crossing_stripes(crossing_polygons)

        return {
            "drivable_polygons": drivable_polygons,
            "road_surface_polygons": road_surface_polygons,
            "lane_fill_polygons": lane_fill_polygons,
            "walkway_polygons": walkway_polygons,
            "crossing_polygons": crossing_polygons,
            "crossing_stripe_polygons": crossing_stripe_polygons,
            "road_edge_lines": road_edge_lines,
            "lane_boundary_lines": lane_boundary_lines,
            "lane_marking_lines": lane_marking_lines,
            "lane_centerlines": lane_centerlines,
            "lane_direction_arrows": cls._centerline_arrows(lane_centerlines),
        }

    @staticmethod
    def _ego_corners_from_state(xy: np.ndarray, yaw: float) -> np.ndarray:
        return NuScenesScorerUtils._box_corners_xy(
            float(xy[0]),
            float(xy[1]),
            length=4.9,
            width=2.1,
            yaw=float(yaw),
        )

    @staticmethod
    def _polygon_intersects(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
        arr_a = np.asarray(poly_a, dtype=np.float32)
        arr_b = np.asarray(poly_b, dtype=np.float32)
        if ShapelyPolygon is not None:
            try:
                return bool(ShapelyPolygon(arr_a[:, :2]).intersects(ShapelyPolygon(arr_b[:, :2])))
            except Exception:
                pass
        min_a = arr_a[:, :2].min(axis=0)
        max_a = arr_a[:, :2].max(axis=0)
        min_b = arr_b[:, :2].min(axis=0)
        max_b = arr_b[:, :2].max(axis=0)
        overlap = np.logical_and(max_a >= min_b, max_b >= min_a)
        return bool(np.all(overlap))

    def _score_candidate_pdm_like(
        self,
        *,
        cand_xy: np.ndarray,
        cand_yaw: np.ndarray,
        gt_xy_cmp: np.ndarray,
        gt_yaw_cmp: np.ndarray,
        gt_xy_full: np.ndarray,
        gt_s_full: np.ndarray,
        gt_total_len: float,
        centerlines: Sequence[Any],
        drivable_polygons: Sequence[Any],
        scene_objects: Sequence[dict[str, Any]],
        agent_states_for_ea: Sequence[dict[str, Any]] = (),
        dt_s: float,
    ) -> dict[str, Any]:
        horizon = int(cand_xy.shape[0])
        if horizon <= 0:
            multiplicative_metrics = {
                "no_collision": 0.0,
                "drivable_area": 0.0,
                "driving_direction": 1.0,
            }
            if self.ea_gate_enabled:
                multiplicative_metrics["ea_safety"] = 0.0#若ea打开就是一个新的multiplicative_merics gate项
            return {
                "score": 0.0,
                "weighted_score": 0.0,
                "multiplicative_product": 0.0,
                "multiplicative_metrics": multiplicative_metrics,
                "weighted_metrics": {
                    "progress": 0.0,
                    "ttc": 0.0,
                    "lane_keeping": 0.0,
                    "history_comfort": 0.0,
                },
            }

        pos_err = np.linalg.norm(cand_xy - gt_xy_cmp, axis=1) if gt_xy_cmp.size else np.zeros((horizon,), dtype=np.float32)
        first_err = float(pos_err[0]) if pos_err.size else 0.0
        final_err = float(pos_err[-1]) if pos_err.size else 0.0
        mean_err = float(pos_err.mean()) if pos_err.size else 0.0
        yaw_err = np.abs(_wrap_angle(cand_yaw - gt_yaw_cmp)) if gt_yaw_cmp.size else np.zeros((horizon,), dtype=np.float32)
        mean_yaw_err = float(yaw_err.mean()) if yaw_err.size else 0.0
        #仅仅作为分析没有进入最终的score
        #关于自车进度，有一个想法是将拉伸为直线，然后两者比较， 
        cand_progress = _project_progress(cand_xy[-1], gt_xy_full, gt_s_full) if gt_xy_full.shape[0] > 0 else 0.0
        progress_ratio = float(np.clip(cand_progress / max(1.0e-6, gt_total_len), 0.0, 1.0))
        #动力学（舒适性）
        cand_prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), cand_xy[:-1]], axis=0)
        cand_vel = cand_xy - cand_prev
        cand_acc = cand_vel[1:] - cand_vel[:-1] if cand_vel.shape[0] > 1 else np.zeros((0, 2), dtype=np.float32)
        cand_speed = np.linalg.norm(cand_vel, axis=1) / max(1.0e-6, float(dt_s))
        cand_yaw_prev = np.concatenate([cand_yaw[:1], cand_yaw[:-1]], axis=0)
        cand_yaw_rate = np.abs(_wrap_angle(cand_yaw - cand_yaw_prev)) / max(1.0e-6, float(dt_s))
        smooth_pen = float(np.linalg.norm(cand_acc, axis=1).mean()) if cand_acc.size else 0.0
        #舒适性 = 纵向平滑 + 横向平滑”，而且必须做尺度平衡
        comfort_cost = smooth_pen + 0.25 * float(cand_yaw_rate.mean()) if cand_yaw_rate.size else smooth_pen
        history_comfort = float(np.clip(1.0 - 0.08 * comfort_cost, 0.0, 1.0))

        lateral_errors: list[float] = []
        oncoming_progress_m = 0.0
        continuous_reverse_time_s = 0.0
        for step_idx in range(horizon):
            move_vec = cand_vel[step_idx]
            lateral_dist, tangent = self._nearest_centerline_stats(cand_xy[step_idx], centerlines, direction_xy=move_vec)
            lateral_errors.append(lateral_dist)
            move_norm = float(np.linalg.norm(move_vec))
            if move_norm > 1.0e-6:
                move_dir = (move_vec / move_norm).astype(np.float32)
                #逆行检测（driving direction）
                reverse_alignment = max(0.0, -float(np.dot(move_dir, tangent)))
                if reverse_alignment > _DEFAULT_DIRECTION_REVERSE_ALIGNMENT_THRESHOLD:
                    continuous_reverse_time_s += float(dt_s)
                    if continuous_reverse_time_s > _DEFAULT_DIRECTION_MIN_CONTINUOUS_REVERSE_S:
                        oncoming_progress_m += move_norm * reverse_alignment
                else:
                    continuous_reverse_time_s = 0.0
        mean_lateral = float(np.mean(lateral_errors)) if lateral_errors else 0.0
        #横向偏移（lane keeping）
        lane_keeping = float(np.clip(1.0 - (mean_lateral / 2.0), 0.0, 1.0)) if centerlines else float(
            np.clip(1.0 - 0.25 * mean_err, 0.0, 1.0)
        )
        driving_direction = _linear_decay_score(
            oncoming_progress_m,
            good_threshold=_DEFAULT_DIRECTION_COMPLIANCE_THRESHOLD_M,
            bad_threshold=_DEFAULT_DIRECTION_VIOLATION_THRESHOLD_M,
        )
        #可行使区域
        offroad = (
            any(
                not all(
                    self._point_in_polygons(corner_xy, drivable_polygons)
                    for corner_xy in self._ego_corners_from_state(cand_xy[step_idx], float(cand_yaw[step_idx]))
                )
                for step_idx in range(horizon)
            )
            if drivable_polygons
            else False
        )
        drivable_area = 0.0 if offroad else 1.0

        collision = False
        earliest_ttc_risk_s = float("inf")
        for obj in scene_objects:
            obj_center = np.asarray(obj["center_xy"], dtype=np.float32).reshape(2)
            obj_velocity = np.asarray(obj.get("velocity_xy", [0.0, 0.0]), dtype=np.float32).reshape(2)
            obj_length = float(obj.get("length_m", 1.0))
            obj_width = float(obj.get("width_m", 1.0))
            obj_yaw = float(obj.get("yaw_rad", 0.0))
            for step_idx in range(horizon):
                cand_box = self._ego_corners_from_state(cand_xy[step_idx], float(cand_yaw[step_idx]))
                proj_center = obj_center + obj_velocity * (float(step_idx) * float(dt_s))
                obj_box = self._box_corners_xy(proj_center[0], proj_center[1], obj_length, obj_width, obj_yaw)
                if self._polygon_intersects(cand_box, obj_box):
                    collision = True
                    break
                if float(cand_speed[step_idx]) < _DEFAULT_TTC_STOPPED_SPEED_THRESHOLD_MPS:
                    continue

                heading_vec = np.asarray(
                    [math.cos(float(cand_yaw[step_idx])), math.sin(float(cand_yaw[step_idx]))],
                    dtype=np.float32,
                )
                rel_center = proj_center - cand_xy[step_idx]
                longitudinal = float(np.dot(rel_center, heading_vec))
                if longitudinal <= 0.0:
                    continue

                for future_offset_s in _DEFAULT_TTC_FUTURE_OFFSETS_S:
                    proj_ego_xy = cand_xy[step_idx] + heading_vec * float(cand_speed[step_idx]) * float(future_offset_s)
                    proj_ego_box = self._ego_corners_from_state(proj_ego_xy, float(cand_yaw[step_idx]))
                    proj_obj_center = proj_center + obj_velocity * float(future_offset_s)
                    proj_obj_box = self._box_corners_xy(proj_obj_center[0], proj_obj_center[1], obj_length, obj_width, obj_yaw)
                    if self._polygon_intersects(proj_ego_box, proj_obj_box):
                        earliest_ttc_risk_s = min(earliest_ttc_risk_s, float(future_offset_s))
                        break
            if collision:
                break
        no_collision = 0.0 if collision else 1.0
        ttc_horizon_s = float(_DEFAULT_TTC_FUTURE_OFFSETS_S[-1]) if _DEFAULT_TTC_FUTURE_OFFSETS_S else 1.0
        if math.isfinite(earliest_ttc_risk_s):#有碰撞风险
            ttc_score = float(np.clip(earliest_ttc_risk_s / max(1.0e-6, ttc_horizon_s), 0.0, 1.0))
        else:#无碰撞风险，则直接满分1分
            ttc_score = 1.0

        ea_gate_result = self._score_ea_safety_gate(
            cand_xy=cand_xy,
            cand_yaw=cand_yaw,
            agent_states=agent_states_for_ea,
            dt_s=float(dt_s),
        ) if self.ea_gate_enabled else {"gate": 1.0, "max_ea": 0.0, "evaluated_pairs": 0.0}

        weighted_metrics = {
            "progress": progress_ratio,
            "ttc": float(ttc_score),
            "lane_keeping": float(lane_keeping),
            "history_comfort": float(history_comfort),
        }
        weighted_weights = {
            "progress": float(self.progress_weight),
            "ttc": float(self.ttc_weight),
            "lane_keeping": float(self.lane_keeping_weight),
            "history_comfort": float(self.history_comfort_weight),
        }
        weighted_score = float(
            sum(weighted_metrics[key] * weighted_weights[key] for key in weighted_metrics)
            / max(1.0, sum(weighted_weights.values()))
        )
        multiplicative_metrics = {
            "no_collision": float(no_collision),
            "drivable_area": float(drivable_area),
            "driving_direction": float(driving_direction) if self.driving_direction_gate_enabled else 1.0,
        }
        if self.ea_gate_enabled:
            multiplicative_metrics["ea_safety"] = float(ea_gate_result["gate"])
        multiplicative_product = float(np.prod(np.asarray(list(multiplicative_metrics.values()), dtype=np.float32)))
        score = float(weighted_score * multiplicative_product)

        score_terms = {
            "progress_reward": 2.0 * progress_ratio,
            "mean_error_penalty": -0.35 * mean_err,
            "final_error_penalty": -0.50 * final_err,
            "first_error_penalty": -0.35 * first_err,
            "yaw_error_penalty": -0.20 * mean_yaw_err,
            "smoothness_penalty": -0.05 * smooth_pen,
            "weighted_score": weighted_score,
            "multiplicative_product": multiplicative_product,
        }
        return {
            "score": score,
            "weighted_score": weighted_score,
            "multiplicative_product": multiplicative_product,
            "multiplicative_metrics": multiplicative_metrics,
            "weighted_metrics": weighted_metrics,
            "progress_ratio": progress_ratio,
            "mean_error_m": mean_err,
            "final_error_m": final_err,
            "first_error_m": first_err,
            "mean_yaw_error_rad": mean_yaw_err,
            "smoothness_penalty_raw": smooth_pen,
            "ttc_earliest_risk_time_s": float(earliest_ttc_risk_s),
            "driving_direction_oncoming_progress_m": float(oncoming_progress_m),
            "ea_gate_max_ea": float(ea_gate_result.get("max_ea", 0.0)),
            "ea_gate_evaluated_pairs": float(ea_gate_result.get("evaluated_pairs", 0.0)),
            "score_terms": score_terms,
        }

    def _score_batch_common(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
        *,
        include_debug_context: bool,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if traj_xyyaw.ndim != 4 or traj_xyyaw.shape[-1] < 2:
            raise RuntimeError(
                "NuScenesScorerUtils expects traj_xyyaw with shape (batch, candidates, horizon, 3); "
                f"got {tuple(traj_xyyaw.shape)}"
            )
        if len(replays) != int(traj_xyyaw.shape[0]):
            raise RuntimeError(f"Replay batch length mismatch: replays={len(replays)} traj_batch={int(traj_xyyaw.shape[0])}")

        traj_np = traj_xyyaw.detach().cpu().numpy().astype(np.float32)
        scores = np.zeros((traj_np.shape[0], traj_np.shape[1]), dtype=np.float32)
        details: list[dict[str, Any]] = []
        for batch_idx, replay in enumerate(replays):
            sample_token = replay.get("sample_token", None)
            if sample_token is None:
                raise RuntimeError("NuScenesScorerUtils requires replay['sample_token']")
            sample_token_str = str(sample_token)
            static_ctx = self._build_static_sample_context(
                replay,
                patch_radius=_DEFAULT_PATCH_RADIUS_M,
            )
            row = dict(static_ctx["row"])
            gt_xy = np.asarray(static_ctx["gt_xy"], dtype=np.float32).copy()
            gt_yaw = np.asarray(static_ctx["gt_yaw"], dtype=np.float32).copy()
            gt_s = np.asarray(static_ctx["gt_s"], dtype=np.float32).copy()
            gt_total_len = float(static_ctx["gt_total_len"])
            gt_sample_token_str = str(static_ctx.get("gt_sample_token", sample_token_str))
            map_context = dict(static_ctx["map_context"])
            gt_history_xy = self._lookup_gt_history(row, origin_xy=None) if include_debug_context else np.zeros((0, 2), dtype=np.float32)
            scene_objects = list(static_ctx["scene_objects"])
            ea_agent_states = list(static_ctx["ea_agent_states"])

            horizon = min(int(gt_xy.shape[0]), int(traj_np.shape[2]))
            gt_xy_cmp = gt_xy[:horizon]
            gt_yaw_cmp = gt_yaw[:horizon]
            lane_centerlines = list(map_context.get("layers", {}).get("lane_centerline", []))
            drivable_polygons = list(map_context.get("layers", {}).get("drivable_area", []))

            sample_detail: dict[str, Any] = {
                "batch_index": int(batch_idx),
                "sample_token": sample_token_str,
                "gt_sample_token": gt_sample_token_str,
                "gt_origin_shift_xy": [0.0, 0.0],
                "gt_xy": gt_xy_cmp.copy(),
                "gt_yaw": gt_yaw_cmp.copy(),
                "gt_history_xy": gt_history_xy.copy(),
                "scene_objects": scene_objects if include_debug_context else [],
                "map_layers": dict(map_context.get("layers", {})) if include_debug_context else dict(map_context.get("layers", {})),
                "map_patch_radius_m": float(map_context.get("patch_radius", _DEFAULT_PATCH_RADIUS_M)),
                "candidates": [],
            }
            for cand_idx in range(int(traj_np.shape[1])):
                cand = traj_np[batch_idx, cand_idx, :horizon, :]
                cand_xy = cand[:, :2]
                cand_yaw = cand[:, 2] if cand.shape[1] >= 3 else _path_yaw_from_xy(cand_xy)
                candidate_result = self._score_candidate_pdm_like(
                    cand_xy=cand_xy,
                    cand_yaw=cand_yaw,
                    gt_xy_cmp=gt_xy_cmp,
                    gt_yaw_cmp=gt_yaw_cmp,
                    gt_xy_full=gt_xy,
                    gt_s_full=gt_s,
                    gt_total_len=gt_total_len,
                    centerlines=lane_centerlines,
                    drivable_polygons=drivable_polygons,
                    scene_objects=scene_objects,
                    agent_states_for_ea=ea_agent_states,
                    dt_s=0.5,
                )
                score = float(candidate_result["score"])
                scores[batch_idx, cand_idx] = np.float32(score)
                sample_detail["candidates"].append(
                    {
                        "candidate_index": int(cand_idx),
                        "traj_xyyaw": cand.copy(),
                        **candidate_result,
                    }
                )
            details.append(sample_detail)
        return scores, details

    def _score_batch(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
        *,
        include_debug_context: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        return self._score_batch_common(replays, traj_xyyaw, include_debug_context=bool(include_debug_context))

    @staticmethod
    def _path_yaw_from_xy_torch(points_xy: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if points_xy.ndim != 3:
            raise RuntimeError(f"Expected points_xy with shape (batch, horizon, 2), got {tuple(points_xy.shape)}")
        if valid_mask.shape != points_xy.shape[:2]:
            raise RuntimeError(
                "valid_mask must match points_xy batch/horizon dimensions: "
                f"mask={tuple(valid_mask.shape)} points={tuple(points_xy.shape)}"
            )
        if int(points_xy.shape[1]) <= 0:
            return torch.zeros(points_xy.shape[:2], device=points_xy.device, dtype=torch.float32)

        prev = torch.cat(
            [
                torch.zeros((points_xy.shape[0], 1, points_xy.shape[2]), device=points_xy.device, dtype=points_xy.dtype),
                points_xy[:, :-1, :],
            ],
            dim=1,
        )
        delta = points_xy - prev
        yaw = torch.atan2(delta[..., 1], delta[..., 0]).to(dtype=torch.float32)
        if int(points_xy.shape[1]) > 1:
            multi_step = valid_mask.sum(dim=1) > 1
            yaw[:, 0] = torch.where(multi_step, yaw[:, 1], yaw[:, 0])
        return torch.where(valid_mask, yaw, torch.zeros_like(yaw))

    @staticmethod
    def _polyline_arclength_torch(points_xy: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if points_xy.ndim != 3:
            raise RuntimeError(f"Expected points_xy with shape (batch, horizon, 2), got {tuple(points_xy.shape)}")
        if valid_mask.shape != points_xy.shape[:2]:
            raise RuntimeError(
                "valid_mask must match points_xy batch/horizon dimensions: "
                f"mask={tuple(valid_mask.shape)} points={tuple(points_xy.shape)}"
            )
        if int(points_xy.shape[1]) <= 0:
            return torch.zeros(points_xy.shape[:2], device=points_xy.device, dtype=torch.float32)

        if int(points_xy.shape[1]) == 1:
            return torch.zeros(points_xy.shape[:2], device=points_xy.device, dtype=torch.float32)

        seg = torch.linalg.norm(points_xy[:, 1:, :] - points_xy[:, :-1, :], dim=-1).to(dtype=torch.float32)
        seg_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        seg = torch.where(seg_valid, seg, torch.zeros_like(seg))
        prefix = torch.zeros((points_xy.shape[0], 1), device=points_xy.device, dtype=torch.float32)
        return torch.cat([prefix, torch.cumsum(seg, dim=1)], dim=1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
        mask_f = mask.to(device=values.device, dtype=values.dtype)
        denom = mask_f.sum(dim=dim).clamp_min(1.0)
        mean = (values * mask_f).sum(dim=dim) / denom
        empty = mask_f.sum(dim=dim) <= 0
        return torch.where(empty, torch.zeros_like(mean), mean)

    @staticmethod
    def _gather_last_valid(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise RuntimeError(f"Expected values with shape (batch, candidates, horizon), got {tuple(values.shape)}")
        if lengths.ndim != 1 or lengths.shape[0] != values.shape[0]:
            raise RuntimeError(
                "lengths must have shape (batch,) matching values batch dimension: "
                f"lengths={tuple(lengths.shape)} values={tuple(values.shape)}"
            )

        last_idx = lengths.clamp_min(1) - 1
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, values.shape[1], 1)
        gathered = torch.gather(values, dim=2, index=gather_idx).squeeze(2)
        return torch.where(
            lengths.view(-1, 1) > 0,
            gathered,
            torch.zeros_like(gathered),
        )

    @staticmethod
    def _gather_last_valid_xy(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise RuntimeError(f"Expected values with shape (batch, candidates, horizon, 2), got {tuple(values.shape)}")
        if lengths.ndim != 1 or lengths.shape[0] != values.shape[0]:
            raise RuntimeError(
                "lengths must have shape (batch,) matching values batch dimension: "
                f"lengths={tuple(lengths.shape)} values={tuple(values.shape)}"
            )

        last_idx = lengths.clamp_min(1) - 1
        gather_idx = last_idx.view(-1, 1, 1, 1).expand(-1, values.shape[1], 1, values.shape[3])
        gathered = torch.gather(values, dim=2, index=gather_idx).squeeze(2)
        return torch.where(
            (lengths.view(-1, 1, 1) > 0),
            gathered,
            torch.zeros_like(gathered),
        )

    def _prepare_torch_gt_batch(
        self,
        replays: Sequence[dict[str, Any]],
        *,
        traj_horizon: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        gt_arrays: list[np.ndarray] = []
        for replay in replays:
            sample_token = replay.get("sample_token", None)
            if sample_token is None:
                raise RuntimeError("NuScenesScorerUtils requires replay['sample_token']")
            sample_token_str = str(sample_token)
            gt_xy_raw = self._lookup_gt(self._gt_sample_token_for_replay(replay, sample_token_str))
            gt_arrays.append(gt_xy_raw)

        batch_size = len(gt_arrays)
        max_gt_horizon = max([1, *[int(arr.shape[0]) for arr in gt_arrays]])
        gt_full_xy = torch.zeros((batch_size, max_gt_horizon, 2), device=device, dtype=torch.float32)
        gt_full_mask = torch.zeros((batch_size, max_gt_horizon), device=device, dtype=torch.bool)
        gt_cmp_xy = torch.zeros((batch_size, traj_horizon, 2), device=device, dtype=torch.float32)

        cmp_lengths_list: list[int] = []
        full_lengths_list: list[int] = []
        for batch_idx, gt_xy in enumerate(gt_arrays):
            full_len = int(gt_xy.shape[0])
            cmp_len = min(full_len, int(traj_horizon))
            full_lengths_list.append(full_len)
            cmp_lengths_list.append(cmp_len)
            if full_len > 0:
                gt_tensor = torch.as_tensor(gt_xy, device=device, dtype=torch.float32)
                gt_full_xy[batch_idx, :full_len, :] = gt_tensor
                gt_full_mask[batch_idx, :full_len] = True
                if cmp_len > 0:
                    gt_cmp_xy[batch_idx, :cmp_len, :] = gt_tensor[:cmp_len]

        cmp_lengths = torch.as_tensor(cmp_lengths_list, device=device, dtype=torch.long)
        full_lengths = torch.as_tensor(full_lengths_list, device=device, dtype=torch.long)
        cmp_valid = torch.arange(traj_horizon, device=device).unsqueeze(0) < cmp_lengths.unsqueeze(1)
        gt_full_yaw = self._path_yaw_from_xy_torch(gt_full_xy, gt_full_mask)
        gt_full_s = self._polyline_arclength_torch(gt_full_xy, gt_full_mask)
        gt_cmp_yaw = torch.zeros((batch_size, traj_horizon), device=device, dtype=torch.float32)
        if traj_horizon > 0:
            copy_horizon = min(int(traj_horizon), int(gt_full_yaw.shape[1]))
            gt_cmp_yaw[:, :copy_horizon] = gt_full_yaw[:, :copy_horizon]
        last_full_idx = full_lengths.clamp_min(1) - 1
        gt_total_len = torch.gather(gt_full_s, dim=1, index=last_full_idx.view(-1, 1)).squeeze(1)
        gt_total_len = torch.where(full_lengths > 0, gt_total_len, torch.ones_like(gt_total_len)).clamp_min(1.0e-6)
        return {
            "gt_full_xy": gt_full_xy,
            "gt_full_mask": gt_full_mask,
            "gt_full_s": gt_full_s,
            "gt_cmp_xy": gt_cmp_xy,
            "gt_cmp_yaw": gt_cmp_yaw,
            "cmp_valid": cmp_valid,
            "cmp_lengths": cmp_lengths,
            "gt_total_len": gt_total_len,
        }

    @staticmethod
    def _project_progress_torch(
        point_xy: torch.Tensor,
        path_xy: torch.Tensor,
        path_s: torch.Tensor,
        path_mask: torch.Tensor,
    ) -> torch.Tensor:
        if point_xy.ndim != 3:
            raise RuntimeError(f"Expected point_xy with shape (batch, candidates, 2), got {tuple(point_xy.shape)}")
        if path_xy.ndim != 3 or path_s.ndim != 2 or path_mask.ndim != 2:
            raise RuntimeError(
                "Expected path tensors with shapes (batch, horizon, 2), (batch, horizon), (batch, horizon); "
                f"got {tuple(path_xy.shape)}, {tuple(path_s.shape)}, {tuple(path_mask.shape)}"
            )

        if int(path_xy.shape[1]) <= 1:
            return torch.zeros(point_xy.shape[:2], device=point_xy.device, dtype=torch.float32)

        p0 = path_xy[:, :-1, :]
        p1 = path_xy[:, 1:, :]
        seg = p1 - p0
        seg_valid = path_mask[:, :-1] & path_mask[:, 1:]
        seg_len_sq = (seg * seg).sum(dim=-1).to(dtype=torch.float32)
        safe_seg_len_sq = torch.where(seg_valid, seg_len_sq.clamp_min(1.0e-12), torch.ones_like(seg_len_sq))

        delta = point_xy.unsqueeze(2) - p0.unsqueeze(1)
        alpha = (delta * seg.unsqueeze(1)).sum(dim=-1) / safe_seg_len_sq.unsqueeze(1)
        alpha = alpha.clamp(0.0, 1.0)
        proj = p0.unsqueeze(1) + alpha.unsqueeze(-1) * seg.unsqueeze(1)
        dist = torch.linalg.norm(point_xy.unsqueeze(2) - proj, dim=-1).to(dtype=torch.float32)
        inf = torch.full_like(dist, float("inf"))
        dist = torch.where(seg_valid.unsqueeze(1), dist, inf)

        seg_len = torch.sqrt(safe_seg_len_sq)
        best_s_all = path_s[:, :-1].unsqueeze(1) + alpha * seg_len.unsqueeze(1)
        best_s_all = torch.where(seg_valid.unsqueeze(1), best_s_all, torch.zeros_like(best_s_all))
        best_idx = dist.argmin(dim=2)
        best_s = torch.gather(best_s_all, dim=2, index=best_idx.unsqueeze(-1)).squeeze(-1)
        has_valid_seg = seg_valid.any(dim=1, keepdim=True)
        return torch.where(has_valid_seg, best_s, torch.zeros_like(best_s))

    def _score_batch_torch(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> torch.Tensor:
        scores, _ = self._score_batch_common(replays, traj_xyyaw, include_debug_context=False)
        return torch.from_numpy(scores).to(device=traj_xyyaw.device, dtype=torch.float32)

    def score_with_details(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        scores, details = self._score_batch(replays, traj_xyyaw, include_debug_context=True)
        return torch.from_numpy(scores), details

    def score(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            return self._score_batch_torch(replays, traj_xyyaw)

    def dump_debug_artifacts(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
        *,
        out_dir: str | Path,
        step_tag: str,
        top_k: int = 4,
    ) -> list[dict[str, str]]:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon as MplPolygon

        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        _, details = self.score_with_details(replays, traj_xyyaw)
        artifacts: list[dict[str, str]] = []
        for sample_detail in details:
            sample_token = str(sample_detail["sample_token"])
            token_slug = sample_token.replace("/", "_")[:24]
            prefix = f"{step_tag}_b{int(sample_detail['batch_index']):03d}_{token_slug}"
            json_path = out_root / f"{prefix}.json"
            png_path = out_root / f"{prefix}.png"

            ranked = sorted(sample_detail["candidates"], key=lambda item: float(item["score"]), reverse=True)
            kept = ranked[: max(1, int(top_k))]
            payload = {
                "sample_token": sample_token,
                "step_tag": str(step_tag),
                "batch_index": int(sample_detail["batch_index"]),
                "gt_origin_shift_xy": np.asarray(sample_detail["gt_origin_shift_xy"], dtype=np.float32).tolist(),
                "gt_xy": np.asarray(sample_detail["gt_xy"], dtype=np.float32).tolist(),
                "gt_yaw": np.asarray(sample_detail["gt_yaw"], dtype=np.float32).tolist(),
                "gt_history_xy": np.asarray(sample_detail.get("gt_history_xy", []), dtype=np.float32).tolist(),
                "scene_objects": [
                    {
                        "category": str(item["category"]),
                        "center_xy": np.asarray(item["center_xy"], dtype=np.float32).tolist(),
                        "length_m": float(item["length_m"]),
                        "width_m": float(item["width_m"]),
                        "yaw_rad": float(item["yaw_rad"]),
                        "speed_mps": float(item["speed_mps"]),
                        "num_lidar_pts": int(item["num_lidar_pts"]),
                        "num_radar_pts": int(item["num_radar_pts"]),
                        "corners_xy": np.asarray(item["corners_xy"], dtype=np.float32).tolist(),
                    }
                    for item in sample_detail.get("scene_objects", [])
                ],
                "map_patch_radius_m": float(sample_detail.get("map_patch_radius_m", _DEFAULT_PATCH_RADIUS_M)),
                "map_layers": {
                    str(layer): [
                        np.asarray(geom, dtype=np.float32).tolist()
                        for geom in geometries
                    ]
                    for layer, geometries in (sample_detail.get("map_layers", {}) or {}).items()
                },
                "render_layers": {},
                "candidates": [
                    {
                        "candidate_index": int(item["candidate_index"]),
                        "score": float(item["score"]),
                        "progress_ratio": float(item["progress_ratio"]),
                        "mean_error_m": float(item["mean_error_m"]),
                        "final_error_m": float(item["final_error_m"]),
                        "first_error_m": float(item["first_error_m"]),
                        "mean_yaw_error_rad": float(item["mean_yaw_error_rad"]),
                        "smoothness_penalty_raw": float(item["smoothness_penalty_raw"]),
                        "ttc_earliest_risk_time_s": (
                            float(item["ttc_earliest_risk_time_s"])
                            if math.isfinite(float(item["ttc_earliest_risk_time_s"]))
                            else None
                        ),
                        "driving_direction_oncoming_progress_m": float(item["driving_direction_oncoming_progress_m"]),
                        "score_terms": {key: float(val) for key, val in item["score_terms"].items()},
                        "traj_xyyaw": np.asarray(item["traj_xyyaw"], dtype=np.float32).tolist(),
                    }
                    for item in kept
                ],
            }
            payload["render_layers"] = self._build_render_layers(payload["map_layers"])
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=160)
            fig.patch.set_facecolor("#f4f2ed")
            ax.set_facecolor("#fbfbf8")

            render_layers = payload["render_layers"]
            for polygon in render_layers.get("drivable_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(MplPolygon(patch[:, :2], closed=True, facecolor="#e6e4df", edgecolor="none", alpha=0.95, zorder=0))
            for polygon in render_layers.get("road_surface_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(MplPolygon(patch[:, :2], closed=True, facecolor="#d8d5cf", edgecolor="none", alpha=0.95, zorder=0.5))
            for polygon in render_layers.get("lane_fill_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(MplPolygon(patch[:, :2], closed=True, facecolor="#d2d0ca", edgecolor="none", alpha=0.45, zorder=0.7))
            for polygon in render_layers.get("walkway_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(MplPolygon(patch[:, :2], closed=True, facecolor="#b8d8b0", edgecolor="none", alpha=0.55, zorder=0.8))
            for polygon in render_layers.get("crossing_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(
                        MplPolygon(
                            patch[:, :2],
                            closed=True,
                            facecolor="#f7f7f4",
                            edgecolor="#b1b1ab",
                            linewidth=0.9,
                            hatch="///",
                            alpha=0.85,
                            zorder=0.9,
                        )
                    )
            for polygon in render_layers.get("crossing_stripe_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(
                        MplPolygon(
                            patch[:, :2],
                            closed=True,
                            facecolor="#fefefe",
                            edgecolor="none",
                            alpha=0.95,
                            zorder=1.0,
                        )
                    )

            for line in render_layers.get("road_edge_lines", []):
                pts = np.asarray(line, dtype=np.float32)
                if pts.shape[0] >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], color="#8b8881", linewidth=1.2, alpha=0.85, zorder=1.2)
            for line in render_layers.get("lane_boundary_lines", []):
                pts = np.asarray(line, dtype=np.float32)
                if pts.shape[0] >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], color="#9f9b92", linewidth=0.8, alpha=0.55, zorder=1.25)
            for line in render_layers.get("lane_marking_lines", []):
                pts = np.asarray(line, dtype=np.float32)
                if pts.shape[0] >= 2:
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        color="#6c6c6a",
                        linewidth=1.0,
                        alpha=0.9,
                        linestyle=(0, (4.0, 3.0)),
                        zorder=1.3,
                    )
            for line in render_layers.get("lane_centerlines", []):
                pts = np.asarray(line, dtype=np.float32)
                if pts.shape[0] >= 2:
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        color="#4e97d1",
                        linewidth=1.2,
                        alpha=0.95,
                        linestyle=(0, (8.0, 4.0)),
                        zorder=1.4,
                    )
            for arrow in render_layers.get("lane_direction_arrows", []):
                pts = np.asarray(arrow, dtype=np.float32)
                if pts.shape[0] >= 2:
                    start = pts[0]
                    end = pts[1]
                    delta = end - start
                    ax.arrow(
                        float(start[0]),
                        float(start[1]),
                        float(delta[0]),
                        float(delta[1]),
                        width=0.03,
                        head_width=0.35,
                        head_length=0.45,
                        length_includes_head=True,
                        color="#5f5b54",
                        alpha=0.7,
                        zorder=1.45,
                    )

            gt_xy = np.asarray(sample_detail["gt_xy"], dtype=np.float32)
            gt_history_xy = np.asarray(sample_detail.get("gt_history_xy", []), dtype=np.float32)
            for obj in payload["scene_objects"]:
                corners = np.asarray(obj["corners_xy"], dtype=np.float32)
                if corners.shape[0] >= 3:
                    color = self._category_color(str(obj["category"]))
                    ax.add_patch(MplPolygon(corners[:, :2], closed=True, facecolor=color, edgecolor="#2c3e50", alpha=0.45, linewidth=1.0, zorder=2))
                    center = np.asarray(obj["center_xy"], dtype=np.float32)
                    ax.text(center[0], center[1], str(obj["category"]), fontsize=6.5, color="#2c3e50", ha="center", va="center", zorder=3)

            ego_box = self._box_corners_xy(0.0, 0.0, length=4.6, width=1.9, yaw=0.0)
            ax.add_patch(MplPolygon(ego_box, closed=True, facecolor="#111111", edgecolor="#111111", alpha=0.9, linewidth=1.2, zorder=4))
            ax.scatter([0.0], [0.0], color="#e63946", marker="x", s=70, label="ego_origin", zorder=5)
            if gt_history_xy.ndim == 2 and gt_history_xy.shape[0] > 0:
                ax.plot(gt_history_xy[:, 0], gt_history_xy[:, 1], color="#7f8c8d", linestyle="--", linewidth=1.8, label="ego_history", zorder=5)
            cmap = plt.get_cmap("viridis", max(2, len(kept)))
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linewidth=2.7, label="gt_future", zorder=6)
            for rank, item in enumerate(kept):
                cand_xyyaw = np.asarray(item["traj_xyyaw"], dtype=np.float32)
                cand_xy = cand_xyyaw[:, :2]
                label = f"rank{rank + 1} idx{int(item['candidate_index'])} score={float(item['score']):.3f}"
                candidate_style = self._candidate_line_style(rank=rank, total=len(kept))
                ax.plot(
                    cand_xy[:, 0],
                    cand_xy[:, 1],
                    color=cmap(rank),
                    linewidth=float(candidate_style["linewidth"]),
                    alpha=float(candidate_style["alpha"]),
                    label=label,
                    zorder=7,
                )
                ax.scatter(
                    [cand_xy[-1, 0]],
                    [cand_xy[-1, 1]],
                    color=cmap(rank),
                    s=float(candidate_style["marker_size"]),
                    alpha=float(candidate_style["alpha"]),
                    zorder=8,
                )
            xlim, ylim = self._compute_plot_limits(
                gt_xy=gt_xy,
                gt_history_xy=gt_history_xy,
                candidates=kept,
                scene_objects=payload["scene_objects"],
                patch_radius=float(payload["map_patch_radius_m"]),
            )
            ax.set_title(f"NuScenes GRPO debug\n{sample_token}")
            ax.set_xlabel("forward x (m)")
            ax.set_ylabel("left y (m)")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.15, color="#737373")
            semantic_handles = [
                Line2D([0], [0], color="black", linewidth=2.7, label="gt_future"),
                Line2D([0], [0], color="#7f8c8d", linewidth=1.8, linestyle="--", label="ego_history"),
                Line2D([0], [0], color="#6c6c6a", linewidth=1.0, linestyle=(0, (4.0, 3.0)), label="lane_marking"),
                Line2D([0], [0], color="#4e97d1", linewidth=1.2, linestyle=(0, (8.0, 4.0)), label="lane_centerline"),
                Line2D([0], [0], marker="s", markersize=8, linestyle="None", markerfacecolor="#e74c3c", markeredgecolor="#2c3e50", label="vehicles/agents"),
            ]
            candidate_handles = [
                Line2D(
                    [0],
                    [0],
                    color=cmap(rank),
                    linewidth=float(self._candidate_line_style(rank=rank, total=len(kept))["linewidth"]),
                    alpha=float(self._candidate_line_style(rank=rank, total=len(kept))["alpha"]),
                    label=f"rank{rank + 1} score={float(item['score']):.3f}",
                )
                for rank, item in enumerate(kept)
            ]
            ax.legend(handles=semantic_handles + candidate_handles, loc="upper right", fontsize=8, framealpha=0.95)
            fig.tight_layout()
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)

            artifacts.append(
                {
                    "sample_token": sample_token,
                    "json_path": str(json_path),
                    "png_path": str(png_path),
                }
            )
        return artifacts


__all__ = [
    "NuScenesScorerUtils",
]
