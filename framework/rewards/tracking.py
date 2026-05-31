from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from framework.algorithms.craft_reward import (
    CRAFT_CORRECTIVE_DEFAULTS,
    compute_corrective_reward_scalar,
)


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _huber(value: float, delta: float) -> float:
    x = abs(float(value))
    d = max(1e-6, float(delta))
    if x <= d:
        return 0.5 * x * x / d
    return x - (0.5 * d)


def _angle_to_deg(angle_rad: float) -> float:
    return math.degrees(float(angle_rad))


@dataclass
class TrackingRewardResult:
    reward: float
    info: Dict[str, Any]


@dataclass(frozen=True)
class ReferencePathCandidate:
    points: np.ndarray
    source: str
    source_legacy: bool = False


_REWARD_MODE_ALIASES = {
    "craft_close_loop": "craft_close_loop",
    "craft_closed_loop": "craft_close_loop",
    "closed_loop": "craft_close_loop",
    "closed loop": "craft_close_loop",
    "close_loop": "craft_close_loop",
    "close loop": "craft_close_loop",
    "craft_sparse_loop": "craft_sparse_loop",
    "craft_corrective": "craft_sparse_loop",
    "sparse_corrective": "craft_sparse_loop",
    "corrective": "craft_sparse_loop",
    "sparse": "craft_sparse_loop",
    "step_path": "step_path",
}


def _reward_mode_key(mode: Any) -> str:
    raw = str(mode or "").strip().lower()
    normalized = raw.replace("-", "_")
    return _REWARD_MODE_ALIASES.get(normalized, normalized)


def _craft_reward_mode_key(craft_cfg: Dict[str, Any]) -> str:
    raw = str(craft_cfg.get("real_reward_model", craft_cfg.get("reward_mode", "corrective"))).strip().lower()
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in {"sparse", "sparse_corrective", "corrective", "craft_corrective", "craft_sparse_loop"}:
        return "craft_sparse_loop"
    if normalized in {
        "close_loop",
        "closed_loop",
        "craft_close_loop",
        "craft_closed_loop",
        "dense_carl",
        "carl",
    }:
        return "craft_close_loop"
    return "craft_close_loop"


def select_reward_mode_cfg(reward_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(reward_cfg, dict):
        return {}

    mode_key = _reward_mode_key(reward_cfg.get("mode", ""))
    mode_cfg = reward_cfg.get(mode_key, None)
    if isinstance(mode_cfg, dict):
        selected = dict(mode_cfg)
        selected["selected_reward_mode"] = mode_key
        return selected

    return reward_cfg


class TrackingRewardComputer:
    def __init__(self, reward_cfg: Dict[str, Any] | None = None) -> None:
        self.reward_cfg = select_reward_mode_cfg(reward_cfg or {})
        self._last_xz: np.ndarray | None = None
        self._last_yaw: float | None = None
        self._last_v: float | None = None
        self._last_yaw_rate: float | None = None
        self._last_a: float | None = None
        self._last_yaw_acc: float | None = None

        # Cached reference path in reward x-z coordinates.
        self._path_xy: np.ndarray | None = None
        self._path_s: np.ndarray | None = None #Frenet  s
        self._path_source: str | None = None
        self._path_source_legacy: bool = False
        self._path_raw_points: int = 0
        self._last_progress_s: float | None = None #progress_delta_s = current progress_s - last frame progress_s

        #global route/center line/ heading dev: deviation correction reward
        self._last_craft_global_dev_ratio: float | None = None #clipe to [0, 1]
        self._last_craft_center_dev_ratio: float | None = None
        self._last_heading_dev_ratio: float | None = None

    def reset(self) -> None:
        self._last_xz = None
        self._last_yaw = None
        self._last_v = None
        self._last_yaw_rate = None
        self._last_a = None
        self._last_yaw_acc = None

        self._path_xy = None
        self._path_s = None
        self._path_source = None
        self._path_source_legacy = False
        self._path_raw_points = 0
        self._last_progress_s = None

        self._last_craft_global_dev_ratio = None
        self._last_craft_center_dev_ratio = None
        self._last_heading_dev_ratio = None

    #from self.reward_cfg fetch different type of reward cfg
    def _path_cfg(self) -> Dict[str, Any]:
        cfg = self.reward_cfg or {}
        path_cfg = cfg.get("path", {}) or {}
        return path_cfg if isinstance(path_cfg, dict) else {}

    def _comfort_cfg(self) -> Dict[str, Any]:
        cfg = self.reward_cfg or {}
        comfort_cfg = cfg.get("comfort", {}) or {}
        return comfort_cfg if isinstance(comfort_cfg, dict) else {}

    def _collision_cfg(self) -> Dict[str, Any]:
        cfg = self.reward_cfg or {}
        collision_cfg = cfg.get("collision", {}) or {}
        return collision_cfg if isinstance(collision_cfg, dict) else {}

    def _ea_cfg(self) -> Dict[str, Any]:
        cfg = self.reward_cfg or {}
        ea_cfg = cfg.get("ea", {}) or {}
        return ea_cfg if isinstance(ea_cfg, dict) else {}

    def _safety_cfg(self) -> Dict[str, Any]:
        cfg = self.reward_cfg or {}
        safety_cfg = cfg.get("safety", {}) or {}
        return safety_cfg if isinstance(safety_cfg, dict) else {}

    def _collision_mode(self) -> str:
        collision_cfg = self._collision_cfg()
        mode = collision_cfg.get("mode", "constraint_gate")
        return str(mode).strip().lower() or "constraint_gate"

    def _terminal_cfg(self) -> Dict[str, Any]:
        cfg = self.reward_cfg or {}
        terminal_cfg = cfg.get("terminal", {}) or {}
        return terminal_cfg if isinstance(terminal_cfg, dict) else {}

    def _craft_cfg(self) -> Dict[str, Any]:
        cfg = self.reward_cfg or {}
        craft_cfg = cfg.get("CRAFT", {}) or {}
        return craft_cfg if isinstance(craft_cfg, dict) else {}

    def _craft_enabled(self) -> bool:
        craft_cfg = self._craft_cfg()
        return bool(craft_cfg.get("enable", False))

    def _craft_corrective_progress_cfg(self) -> Dict[str, Any]:
        craft_cfg = self._craft_cfg()
        progress_cfg = craft_cfg.get("corrective_progress", {}) or {}
        return progress_cfg if isinstance(progress_cfg, dict) else {}

    @staticmethod
    def _craft_map_cfg_from(craft_cfg: Dict[str, Any]) -> Dict[str, Any]:
        map_cfg = craft_cfg.get("map", {}) or {}
        return map_cfg if isinstance(map_cfg, dict) else {}

    def _craft_off_global_route_threshold(self, craft_cfg: Dict[str, Any]) -> float | None:
        map_cfg = self._craft_map_cfg_from(craft_cfg)
        threshold = craft_cfg.get("off_global_route_threshold_m", map_cfg.get("off_global_route_threshold_m", None))
        if threshold is None:
            return None
        return max(0.0, float(threshold))

    def _craft_event_flags(
        self,
        *,
        craft_cfg: Dict[str, Any],
        info_in: Dict[str, Any],
        lateral_error_m: float,
    ) -> Dict[str, Any]:
        static_collision = bool(info_in.get("static_collision", False))
        dynamic_collision = bool(info_in.get("dynamic_collision", False))
        done_reason = str(info_in.get("done_reason", ""))

        off_global_route_threshold_m = self._craft_off_global_route_threshold(craft_cfg)
        provided_off_global_route = bool(info_in.get("off_global_route", False))
        computed_off_global_route = bool(
            off_global_route_threshold_m is not None
            and float(lateral_error_m) >= float(off_global_route_threshold_m)
        )
        terminal_off_global_route = bool(info_in.get("terminated_by_threshold", False)) and (
            "xz_err" in done_reason or "yaw_err" in done_reason
        )
        off_global_route = bool(provided_off_global_route or computed_off_global_route or terminal_off_global_route)
        off_global_route_source = ""
        if provided_off_global_route:
            off_global_route_source = "provided"
        elif computed_off_global_route:
            off_global_route_source = "route_lateral_threshold"
        elif terminal_off_global_route:
            off_global_route_source = "terminal_threshold"

        run_red_light = float(info_in.get("red_light_violation", info_in.get("run_red_light", 0.0)))
        run_stop_sign = float(info_in.get("stop_sign_violation", info_in.get("run_stop_sign", 0.0)))
        return {
            "static_collision": bool(static_collision),
            "dynamic_collision": bool(dynamic_collision),
            "collision": bool(static_collision or dynamic_collision),
            "off_road": bool(info_in.get("off_road", info_in.get("craft_off_road", False))),
            "off_global_route": bool(off_global_route),
            "off_global_route_source": str(off_global_route_source),
            "off_global_route_threshold_m": off_global_route_threshold_m,
            "emergency_lane": bool(info_in.get("emergency_lane", info_in.get("in_emergency_lane", False))),
            "opposite_lane": bool(
                info_in.get(
                    "opposite_lane",
                    info_in.get("driving_direction_violation", info_in.get("craft_opposite_lane", False)),
                )
            ),
            "red_light_signal_present": "red_light_violation" in info_in or "run_red_light" in info_in,
            "stop_sign_signal_present": "stop_sign_violation" in info_in or "run_stop_sign" in info_in,
            "run_red_light": float(run_red_light),
            "run_stop_sign": float(run_stop_sign),
            "red_light": bool(run_red_light),
            "stop_sign": bool(run_stop_sign),
            "route_completed": bool(info_in.get("route_completed", False) or info_in.get("terminal_kind", None) == "env_done"),
            "route_deviation": bool(info_in.get("route_deviation", False)),
        }

    @staticmethod
    def _heading_dev_ratio_from_info(
        *,
        info_in: Dict[str, Any],
        route_heading_dev_ratio: float,
    ) -> float:
        value = info_in.get(
            "heading_dev_ratio",
            info_in.get("map_heading_dev_ratio", route_heading_dev_ratio),
        )
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _ego_yaw_from_pose(pose: np.ndarray) -> float:
        rot = np.asarray(pose[:3, :3], dtype=np.float64)
        return float(math.atan2(float(rot[2, 0]), float(rot[0, 0])))

    def _reference_path_candidate_from_env(self, env: Any) -> ReferencePathCandidate:
        expert_poses = getattr(env, "all_expert_ego", None)
        if isinstance(expert_poses, list) and len(expert_poses) > 0:
            pts = [np.asarray(pose[:3, 3][[0, 2]], dtype=np.float64) for pose in expert_poses]
            return ReferencePathCandidate(
                points=np.asarray(pts, dtype=np.float64),
                source="all_expert_ego",
                source_legacy=False,
            )

        expert_pairs = getattr(env, "expert_pair", None)
        if isinstance(expert_pairs, list) and len(expert_pairs) > 0:
            return ReferencePathCandidate(
                points=np.asarray(expert_pairs, dtype=np.float64),
                source="expert_pair",
                source_legacy=True,
            )

        ego = np.asarray(env.start_ego[:3, 3][[0, 2]], dtype=np.float64)
        return ReferencePathCandidate(
            points=ego.reshape(1, 2),
            source="start_ego",
            source_legacy=False,
        )

    def _reference_points_from_env(self, env: Any) -> np.ndarray:
        return self._reference_path_candidate_from_env(env).points

    def _reference_path_info(self) -> Dict[str, Any]:
        return {
            "reference_path_source": str(self._path_source or "unknown"),
            "reference_path_source_legacy": bool(self._path_source_legacy),
            "reference_path_raw_points": int(self._path_raw_points),
            "reference_path_points": int(self._path_xy.shape[0]) if self._path_xy is not None else 0,
        }

    @staticmethod
    #Deduplicate waypoints points first, then interpolate, and finally deduplicate again.
    def _dedupe_polyline(points: np.ndarray) -> np.ndarray:
        if int(points.shape[0]) <= 1:
            return points
        kept = [points[0]]
        for idx in range(1, int(points.shape[0])):
            if float(np.linalg.norm(points[idx] - kept[-1])) > 1e-6:
                kept.append(points[idx])
        return np.asarray(kept, dtype=np.float64)

    @staticmethod
    def _densify_polyline(points: np.ndarray, *, ds: float) -> np.ndarray:
        if int(points.shape[0]) <= 1:
            return points
        step = max(1e-3, float(ds))
        out = [points[0]]
        for idx in range(int(points.shape[0]) - 1):
            p0 = points[idx]
            p1 = points[idx + 1]
            delta = p1 - p0
            seg_len = float(np.linalg.norm(delta))
            if seg_len <= 1e-9:
                continue
            n_steps = max(1, int(math.ceil(seg_len / step)))
            for sub_idx in range(1, n_steps + 1):
                alpha = min(1.0, float(sub_idx) / float(n_steps))
                out.append((1.0 - alpha) * p0 + alpha * p1)
        return np.asarray(out, dtype=np.float64)

    @staticmethod
    #Calculate the cumulative arc length along the path.
    def _path_arclength(points: np.ndarray) -> np.ndarray:
        if int(points.shape[0]) <= 0:
            return np.zeros((0,), dtype=np.float64)
        if int(points.shape[0]) == 1:
            return np.zeros((1,), dtype=np.float64)
        seg = np.linalg.norm(points[1:] - points[:-1], axis=1)
        return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg, dtype=np.float64)], axis=0)

    def _ensure_reference_path(self, env: Any) -> tuple[np.ndarray, np.ndarray]:
        if self._path_xy is not None and self._path_s is not None and int(self._path_xy.shape[0]) > 0:
            return self._path_xy, self._path_s

        path_cfg = self._path_cfg()
        interp_ds = float(path_cfg.get("interp_ds_m", 0.25))
        candidate = self._reference_path_candidate_from_env(env)
        raw = candidate.points
        raw = self._dedupe_polyline(raw)
        dense = self._densify_polyline(raw, ds=interp_ds)
        dense = self._dedupe_polyline(dense)
        dense_s = self._path_arclength(dense)
        self._path_xy = dense
        self._path_s = dense_s
        self._path_source = candidate.source
        self._path_source_legacy = bool(candidate.source_legacy)
        self._path_raw_points = int(raw.shape[0])
        return dense, dense_s

    @staticmethod
    def _project_point_to_path(point: np.ndarray, path_xy: np.ndarray, path_s: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        if int(path_xy.shape[0]) <= 1:
            tangent = np.array([1.0, 0.0], dtype=np.float64)
            return 0.0, float(np.linalg.norm(point - path_xy[0])), path_xy[0], tangent

        best_dist = float("inf")
        best_s = 0.0
        best_proj = path_xy[0]
        best_tangent = np.array([1.0, 0.0], dtype=np.float64)

        for idx in range(int(path_xy.shape[0]) - 1):
            p0 = path_xy[idx]
            p1 = path_xy[idx + 1]
            seg = p1 - p0
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq <= 1e-12:
                continue
            alpha = float(np.dot(point - p0, seg) / seg_len_sq)
            alpha = max(0.0, min(1.0, alpha))
            proj = p0 + alpha * seg
            dist = float(np.linalg.norm(point - proj))
            if dist < best_dist:
                best_dist = dist
                best_proj = proj
                seg_len = math.sqrt(seg_len_sq)
                best_tangent = seg / max(1e-12, seg_len)
                best_s = float(path_s[idx] + alpha * seg_len)

        return best_s, best_dist, best_proj, best_tangent

    def compute(
        self,
        *,
        env: Any,
        info: Dict[str, Any] | None,
        step_idx: int,
        done: bool,
    ) -> TrackingRewardResult:
        cfg = self.reward_cfg or {}
        path_cfg = self._path_cfg()
        comfort_cfg = self._comfort_cfg()
        collision_cfg = self._collision_cfg()
        ea_cfg = self._ea_cfg()
        safety_cfg = self._safety_cfg()

        dt = max(1e-6, float(cfg.get("dt", 0.5)))
        ego_xz = np.asarray(env.start_ego[:3, 3][[0, 2]], dtype=np.float64)
        ego_yaw = self._ego_yaw_from_pose(np.asarray(env.start_ego, dtype=np.float64))

        path_xy, path_s = self._ensure_reference_path(env)#Frenet 坐标
        progress_s, lateral_error_m, _proj, tangent = self._project_point_to_path(ego_xz, path_xy, path_s)
        total_path_len = float(path_s[-1]) if int(path_s.shape[0]) > 0 else 0.0
        completion_ratio = float(np.clip(progress_s / max(1.0e-6, total_path_len), 0.0, 1.0))
        path_heading = float(math.atan2(float(tangent[1]), float(tangent[0])))
        yaw_path_err_rad = _wrap_angle(float(ego_yaw) - float(path_heading))#航向误差
        yaw_path_err_deg = abs(_angle_to_deg(yaw_path_err_rad))

        if self._last_progress_s is None:
            progress_delta_s = 0.0
        else:
            progress_delta_s = float(progress_s - self._last_progress_s)
        self._last_progress_s = float(progress_s)
        
        selected_reward_mode = _reward_mode_key(cfg.get("selected_reward_mode", ""))
        craft_cfg = self._craft_cfg()
        if selected_reward_mode == "craft_sparse_loop" or (
            selected_reward_mode == "" and self._craft_enabled() and _craft_reward_mode_key(craft_cfg) == "craft_sparse_loop"
        ):
            info_in = info if isinstance(info, dict) else {}
            return self._compute_craft_corrective_reward(
                craft_cfg=craft_cfg,
                info=info,
                info_in=info_in,
                step_idx=step_idx,
                done=done,
                progress_s=progress_s,
                total_path_len=total_path_len,
                completion_ratio=completion_ratio,
                progress_delta_s=progress_delta_s,
                lateral_error_m=lateral_error_m,
                path_heading=path_heading,
                ego_yaw=ego_yaw,
                yaw_path_err_deg=yaw_path_err_deg,
            )

        if selected_reward_mode == "craft_close_loop" or (
            selected_reward_mode == "" and self._craft_enabled() and _craft_reward_mode_key(craft_cfg) == "craft_close_loop"
        ):
            return self._compute_craft_closed_loop_reward(
                cfg=cfg,
                craft_cfg=craft_cfg,
                info=info,
                step_idx=step_idx,
                done=done,
                progress_s=progress_s,
                total_path_len=total_path_len,
                completion_ratio=completion_ratio,
                progress_delta_s=progress_delta_s,
                lateral_error_m=lateral_error_m,
                path_heading=path_heading,
                ego_yaw=ego_yaw,
                yaw_path_err_deg=yaw_path_err_deg,
            )

        return self._compute_step_path_reward(
            cfg=cfg,
            path_cfg=path_cfg,
            comfort_cfg=comfort_cfg,
            collision_cfg=collision_cfg,
            ea_cfg=ea_cfg,
            safety_cfg=safety_cfg,
            dt=dt,
            info=info,
            step_idx=step_idx,
            done=done,
            ego_xz=ego_xz,
            ego_yaw=ego_yaw,
            progress_s=progress_s,
            total_path_len=total_path_len,
            completion_ratio=completion_ratio,
            progress_delta_s=progress_delta_s,
            lateral_error_m=lateral_error_m,
            path_heading=path_heading,
            yaw_path_err_deg=yaw_path_err_deg,
        )

    def _compute_step_path_reward(
        self,
        *,
        cfg: Dict[str, Any],
        path_cfg: Dict[str, Any],
        comfort_cfg: Dict[str, Any],
        collision_cfg: Dict[str, Any],
        ea_cfg: Dict[str, Any],
        safety_cfg: Dict[str, Any],
        dt: float,
        info: Dict[str, Any] | None,
        step_idx: int,
        done: bool,
        ego_xz: np.ndarray,
        ego_yaw: float,
        progress_s: float,
        total_path_len: float,
        completion_ratio: float,
        progress_delta_s: float,
        lateral_error_m: float,
        path_heading: float,
        yaw_path_err_deg: float,
    ) -> TrackingRewardResult:
        """Compute the legacy non-CRAFT step_path reward."""
    
        ###进度奖励;      这一帧比上一帧，在路径上多走了多少距离
        progress_forward_cap = float(path_cfg.get("progress_forward_cap_m", 2.0))   
        progress_backward_cap = float(path_cfg.get("progress_backward_cap_m", 0.5))
        w_progress = float(path_cfg.get("w_progress", 1.0))
        progress_reward = float(np.clip(progress_delta_s, -progress_backward_cap, progress_forward_cap))
        progress_term = w_progress * progress_reward #1*【2.-0.5】
        
        ###safety相关的奖励
        #只有safety_cfg["enable"] == True 才会计算
        front_obstacle_active = False
        front_obstacle_gap_m = math.inf
        front_obstacle_lateral_m = math.inf                                                             
        front_obstacle_closing_speed_mps = 0.0
        front_obstacle_ttc_s = math.inf
        front_obstacle_overlap = 0.0
        front_obstacle_clearance_risk = 0.0
        front_obstacle_ttc_risk = 0.0
        front_obstacle_cost = 0.0
        safe_progress_gate = 1.0
        if bool(safety_cfg.get("enable", False)):
            # ##进行safety计算：
            # lookahead_m = safety_cfg["lookahead_m"] -> 15.0
            # corridor_half_width_m = safety_cfg["corridor_half_width_m"] -> 2.5
            # safe_gap_m = safety_cfg["safe_gap_m"] -> 8.0
            # safe_ttc_s = safety_cfg["safe_ttc_s"] -> 3.0
            # w_clearance = safety_cfg["w_clearance"] -> 0.0
            # w_ttc = safety_cfg["w_ttc"] -> 0.0
            # progress_gate_strength = safety_cfg["progress_gate_strength"] -> 1.0
            # min_progress_gate = safety_cfg["min_progress_gate"] -> 0.0
            lookahead_m = max(1.0e-6, float(safety_cfg.get("lookahead_m", 15.0)))
            corridor_half_width_m = max(1.0e-6, float(safety_cfg.get("corridor_half_width_m", 2.5)))
            safe_gap_m = max(1.0e-6, float(safety_cfg.get("safe_gap_m", 8.0)))
            safe_ttc_s = max(1.0e-6, float(safety_cfg.get("safe_ttc_s", 3.0)))
            w_clearance = float(safety_cfg.get("w_clearance", 0.0))
            w_ttc = float(safety_cfg.get("w_ttc", 0.0))
            progress_gate_strength = float(safety_cfg.get("progress_gate_strength", 1.0)) #progress_gate_strength=1
            min_progress_gate = float(np.clip(float(safety_cfg.get("min_progress_gate", 0.0)), 0.0, 1.0))
            if isinstance(info, dict):
                front_obstacle_gap_m = float(info.get("front_obstacle_gap_m", math.inf))
                front_obstacle_lateral_m = float(info.get("front_obstacle_lateral_m", math.inf))
                front_obstacle_closing_speed_mps = float(info.get("front_obstacle_closing_speed_mps", 0.0))
                ttc_value = info.get("front_obstacle_ttc_s", None)
                if ttc_value is not None:
                    front_obstacle_ttc_s = float(ttc_value)
                elif front_obstacle_closing_speed_mps > 1.0e-6 and math.isfinite(front_obstacle_gap_m):
                    front_obstacle_ttc_s = float(front_obstacle_gap_m / max(1.0e-6, front_obstacle_closing_speed_mps))
            #确定obstacle是在前方一定范围内
            in_front = math.isfinite(front_obstacle_gap_m) and 0.0 < front_obstacle_gap_m < lookahead_m
            lateral_abs = abs(float(front_obstacle_lateral_m))
            #lateral越小 约在正前方这个overlap越接近1；
            front_obstacle_overlap = float(np.clip((corridor_half_width_m - lateral_abs) / corridor_half_width_m, 0.0, 1.0))
            front_obstacle_active = bool(in_front and front_obstacle_overlap > 0.0)
            #削弱前进奖励 + 产生安全惩罚项
            if front_obstacle_active:
                front_obstacle_clearance_risk = float(np.clip((safe_gap_m - front_obstacle_gap_m) / safe_gap_m, 0.0, 1.0))
                if math.isfinite(front_obstacle_ttc_s):
                    front_obstacle_ttc_risk = float(np.clip((safe_ttc_s - front_obstacle_ttc_s) / safe_ttc_s, 0.0, 1.0))
                #距离风险和 TTC 风险只要有一个高，就认为危险
                risk = max(front_obstacle_clearance_risk, front_obstacle_ttc_risk) * front_obstacle_overlap
                #当前方有障碍物风险的时候，削弱正向progress奖励
                safe_progress_gate = float(np.clip(1.0 - progress_gate_strength * risk, min_progress_gate, 1.0))
                # gated_positive_reward = positive_reward * float(safety_gate_scale) * float(safe_progress_gate)
                # reward = gated_positive_reward - cost_reward
                front_obstacle_cost = float(
                    front_obstacle_overlap#lateral<2.5m且longitude< 12m 才会启动safety计算
                    * (
                        w_clearance * front_obstacle_clearance_risk * front_obstacle_clearance_risk
                        + w_ttc * front_obstacle_ttc_risk * front_obstacle_ttc_risk
                    )
                )
                
                
        ###横向惩罚
        lateral_free = float(path_cfg.get("lateral_free_m", 0.3))
        lateral_delta = float(path_cfg.get("lateral_huber_delta_m", 0.5))
        lateral_excess = max(0.0, float(lateral_error_m) - lateral_free)
        lateral_penalty = _huber(lateral_excess, lateral_delta)
        lateral_term = float(path_cfg.get("w_lateral", cfg.get("w_pos", 0.0))) * lateral_penalty
        ###航向角误差惩罚yaw
        #ego yaw 和参考路径切线方向的夹角
        yaw_free_deg = float(path_cfg.get("yaw_free_deg", 5.0))
        yaw_delta_deg = float(path_cfg.get("yaw_huber_delta_deg", 10.0))
        yaw_excess_deg = max(0.0, float(yaw_path_err_deg) - yaw_free_deg)
        yaw_penalty = _huber(yaw_excess_deg, yaw_delta_deg)
        yaw_term = float(path_cfg.get("w_yaw", cfg.get("w_heading", 0.0))) * yaw_penalty
        ###碰撞约束
        static_collision = bool(info.get("static_collision", False)) if isinstance(info, dict) else False
        dynamic_collision = bool(info.get("dynamic_collision", False)) if isinstance(info, dict) else False
        collision_mode = self._collision_mode()
        static_collision_penalty = 0.0
        dynamic_collision_penalty = 0.0
        safety_gate_active = False
        safety_gate_scale = 1.0
        safety_gate_sources: list[str] = []
        if collision_mode == "dense_penalty":
            static_collision_penalty = float(collision_cfg.get("w_static", cfg.get("w_static", 5.0))) if static_collision else 0.0
            dynamic_collision_penalty = float(collision_cfg.get("w_dynamic", cfg.get("w_dynamic", 5.0))) if dynamic_collision else 0.0
        elif static_collision or dynamic_collision:
            #非 dense 模式下，碰撞不是直接加 collision_penalty，而是把 progress term 乘上 gate_scale。默认 gate_scale=0.0
            safety_gate_active = True
            safety_gate_scale = min(float(safety_gate_scale), float(collision_cfg.get("gate_scale", 0.0)))
            safety_gate_sources.append("collision_constraint")
        #严重横向 / 航向偏离
        severe_gate_scale = float(path_cfg.get("severe_gate_scale", cfg.get("severe_gate_scale", 0.0)))
        severe_lateral_error_m = path_cfg.get("severe_lateral_error_m", cfg.get("severe_lateral_error_m", None))
        severe_yaw_error_deg = path_cfg.get("severe_yaw_error_deg", cfg.get("severe_yaw_error_deg", None))
        severe_lateral_cost = 0.0
        severe_yaw_cost = 0.0
        if severe_lateral_error_m is not None and float(lateral_error_m) > float(severe_lateral_error_m):
            safety_gate_active = True
            safety_gate_scale = min(float(safety_gate_scale), float(severe_gate_scale))
            safety_gate_sources.append("severe_tracking_lateral")
            severe_lateral_cost = float(path_cfg.get("severe_lateral_cost", cfg.get("severe_lateral_cost", 0.0)))
        if severe_yaw_error_deg is not None and float(yaw_path_err_deg) > float(severe_yaw_error_deg):
            safety_gate_active = True
            safety_gate_scale = min(float(safety_gate_scale), float(severe_gate_scale))
            safety_gate_sources.append("severe_tracking_yaw")
            severe_yaw_cost = float(path_cfg.get("severe_yaw_cost", path_cfg.get("severe_heading_cost", cfg.get("severe_yaw_cost", 0.0))))

        ###Jerk舒适度约束
        jerk_clip = float(comfort_cfg.get("jerk_clip", cfg.get("jerk_clip", 50.0)))
        jerk = 0.0
        yaw_jerk = 0.0
        if (
            float(comfort_cfg.get("w_longitudinal_jerk", cfg.get("w_longitudinal_jerk", 0.0))) != 0.0
            or float(comfort_cfg.get("w_yaw_jerk", cfg.get("w_yaw_jerk", 0.0))) != 0.0
        ):
            cur_xz = np.asarray(ego_xz, dtype=np.float64)
            if self._last_xz is None:
                self._last_xz = cur_xz
                self._last_yaw = float(ego_yaw)
            dxz = cur_xz - self._last_xz
            heading = np.array([math.cos(float(ego_yaw)), math.sin(float(ego_yaw))], dtype=np.float64)
            v = float(np.dot(dxz, heading) / dt)
            yaw_rate = float(_wrap_angle(float(ego_yaw) - float(self._last_yaw if self._last_yaw is not None else ego_yaw)) / dt)
            if self._last_v is None:
                self._last_v = v
                self._last_yaw_rate = yaw_rate
            a = float((v - float(self._last_v)) / dt)
            yaw_acc = float((yaw_rate - float(self._last_yaw_rate)) / dt)
            if self._last_a is None:
                self._last_a = a
                self._last_yaw_acc = yaw_acc
            jerk = float(np.clip((a - float(self._last_a)) / dt, -jerk_clip, jerk_clip))
            yaw_jerk = float(np.clip((yaw_acc - float(self._last_yaw_acc)) / dt, -jerk_clip, jerk_clip))
            self._last_xz = cur_xz
            self._last_yaw = float(ego_yaw)
            self._last_v = v
            self._last_yaw_rate = yaw_rate
            self._last_a = a
            self._last_yaw_acc = yaw_acc

        longitudinal_jerk_free = float(comfort_cfg.get("longitudinal_jerk_free", 0.0))
        longitudinal_jerk_delta = float(comfort_cfg.get("longitudinal_jerk_delta", 1.0))
        yaw_jerk_free = float(comfort_cfg.get("yaw_jerk_free", 0.0))
        yaw_jerk_delta = float(comfort_cfg.get("yaw_jerk_delta", 1.0))
        jerk_penalty = _huber(max(0.0, abs(float(jerk)) - longitudinal_jerk_free), longitudinal_jerk_delta)
        yaw_jerk_penalty = _huber(max(0.0, abs(float(yaw_jerk)) - yaw_jerk_free), yaw_jerk_delta)
        jerk_term = float(comfort_cfg.get("w_longitudinal_jerk", cfg.get("w_longitudinal_jerk", 0.0))) * jerk_penalty
        yaw_jerk_term = float(comfort_cfg.get("w_yaw_jerk", cfg.get("w_yaw_jerk", 0.0))) * yaw_jerk_penalty
        ###TODO:确定一下ea指标的大致范围
        ea_enabled = bool(ea_cfg.get("enable", False))
        ea_available = bool(info.get("ea_available", False)) if isinstance(info, dict) else False
        ea_max = float(info.get("ea_max", 0.0)) if isinstance(info, dict) else 0.0
        ea_min = float(info.get("ea_min", ea_max)) if isinstance(info, dict) else 0.0
        ea_mean = float(info.get("ea_mean", ea_max)) if isinstance(info, dict) else 0.0
        ea_risk = float(np.clip(float(info.get("ea_risk", 0.0)) if isinstance(info, dict) else 0.0, 0.0, 1.0))
        ea_weight = float(ea_cfg.get("weight", ea_cfg.get("w_ea", 0.0)))
        ea_cost = float(ea_weight * ea_risk) if ea_enabled and ea_available else 0.0

        positive_reward = progress_term
        cost_reward = (
            lateral_term
            + yaw_term
            + static_collision_penalty
            + dynamic_collision_penalty
            + severe_lateral_cost
            + severe_yaw_cost
            + jerk_term
            + yaw_jerk_term
            + ea_cost
            + front_obstacle_cost
        )
        gated_positive_reward = positive_reward * float(safety_gate_scale) * float(safe_progress_gate)
        reward = gated_positive_reward - cost_reward

        out_info = dict(info or {})
        out_info.update(
            {
                # step_path reward 总公式:
                # reward = positive_reward * safety_gate_scale * safe_progress_gate - cost_reward.
                "reward_mode": "step_path",
                "step_idx": int(step_idx),
                "reward": float(reward),
                "done": bool(done),

                # 路径进度项。progress_term 是主要正奖励。
                "progress_s": float(progress_s),
                "total_path_len_m": float(total_path_len),
                "completion_ratio": float(completion_ratio),
                "progress_delta_s": float(progress_delta_s),
                "progress_reward": float(progress_reward),
                "positive_reward": float(positive_reward),
                "gated_positive_reward": float(gated_positive_reward),
                "cost_reward": float(cost_reward),
                "progress_term": float(progress_term),

                # 前方障碍物安全项。gate 会压低前进奖励，cost 会进入 cost_reward。
                "safe_progress_gate": float(safe_progress_gate),
                "front_obstacle_active": bool(front_obstacle_active),
                "front_obstacle_gap_m": float(front_obstacle_gap_m),
                "front_obstacle_lateral_m": float(front_obstacle_lateral_m),
                "front_obstacle_closing_speed_mps": float(front_obstacle_closing_speed_mps),
                "front_obstacle_ttc_s": float(front_obstacle_ttc_s),
                "front_obstacle_overlap": float(front_obstacle_overlap),
                "front_obstacle_clearance_risk": float(front_obstacle_clearance_risk),
                "front_obstacle_ttc_risk": float(front_obstacle_ttc_risk),
                "front_obstacle_cost": float(front_obstacle_cost),

                # 路径跟踪误差项。lateral_term 和 yaw_term 都进入 cost_reward。
                "lateral_error_m": float(lateral_error_m),
                "lateral_penalty": float(lateral_penalty),
                "lateral_term": float(lateral_term),
                "path_heading_deg": float(_angle_to_deg(path_heading)),
                "ego_yaw_deg": float(_angle_to_deg(ego_yaw)),
                "yaw_path_err_deg": float(yaw_path_err_deg),
                "yaw_penalty": float(yaw_penalty),
                "yaw_term": float(yaw_term),

                # 舒适度项。jerk_term 和 yaw_jerk_term 都进入 cost_reward。
                "longitudinal_jerk": float(jerk),
                "yaw_jerk": float(yaw_jerk),
                "jerk_penalty": float(jerk_penalty),
                "yaw_jerk_penalty": float(yaw_jerk_penalty),
                "jerk_term": float(jerk_term),
                "yaw_jerk_term": float(yaw_jerk_term),

                # EA 风险项。EA 开启且可用时，ea_cost 进入 cost_reward。
                "ea_enabled": bool(ea_enabled),
                "ea_available": bool(ea_available),
                "ea_max": float(ea_max),
                "ea_min": float(ea_min),
                "ea_mean": float(ea_mean),
                "ea_risk": float(ea_risk),
                "ea_weight": float(ea_weight),
                "ea_cost": float(ea_cost),
                "ea_evaluated_pairs": float(info.get("ea_evaluated_pairs", 0.0)) if isinstance(info, dict) else 0.0,

                # 碰撞和严重偏离项。dense penalty 直接进入 cost_reward；
                # constraint gate 只负责缩小正奖励。
                "static_collision": bool(static_collision),
                "dynamic_collision": bool(dynamic_collision),
                "collision_mode": str(collision_mode),
                "static_collision_penalty": float(static_collision_penalty),
                "dynamic_collision_penalty": float(dynamic_collision_penalty),
                "safety_gate_active": bool(safety_gate_active),
                "safety_gate_scale": float(safety_gate_scale),
                "safety_gate_source": "+".join(safety_gate_sources),
                "safety_gate_sources": list(safety_gate_sources),
                "severe_lateral_error_m": None if severe_lateral_error_m is None else float(severe_lateral_error_m),
                "severe_yaw_error_deg": None if severe_yaw_error_deg is None else float(severe_yaw_error_deg),
                "severe_lateral_cost": float(severe_lateral_cost),
                "severe_yaw_cost": float(severe_yaw_cost),

                # 旧日志/可视化兼容字段。
                "pos_dev": float(lateral_error_m),
                "pos_dev_source": "path_projection",
                "yaw_err_deg": float(yaw_path_err_deg),
                "rpd": float(lateral_term),
                "rhd": float(yaw_term),
                "rsc": float(static_collision_penalty),
                "rdc": float(dynamic_collision_penalty),
                "jerk_pen": float(jerk_term),
                "yaw_jerk_pen": float(yaw_jerk_term),
            }
        )
        out_info.update(self._reference_path_info())
        return TrackingRewardResult(reward=float(reward), info=out_info)

    def _compute_craft_corrective_reward(
        self,
        *,
        craft_cfg: Dict[str, Any],
        info: Dict[str, Any] | None,
        info_in: Dict[str, Any],
        step_idx: int,
        done: bool,
        progress_s: float,
        total_path_len: float,
        completion_ratio: float,
        progress_delta_s: float,
        lateral_error_m: float,
        path_heading: float,
        ego_yaw: float,
        yaw_path_err_deg: float,
        ) -> TrackingRewardResult:
        
        # "reward = corrective_progress_bonus - rule/safety_event_cost - ea_cost"
        ea_cfg = self._ea_cfg()
        corrective_progress_cfg = self._craft_corrective_progress_cfg()#控制是否给 progress bonus。
        corrective_params = craft_cfg.get("corrective", {}) or {}#CRAFT.corrective 下的规则/安全扣分权重
        if not isinstance(corrective_params, dict):
            corrective_params = {}
        corrective_progress_enabled = bool(corrective_progress_cfg.get("enable", False))
        heading_max_deg = max(1.0e-6, float(craft_cfg.get("heading_max_deg", 60.0)))
        route_heading_dev_ratio = float(np.clip(float(yaw_path_err_deg) / heading_max_deg, 0.0, 1.0))
        heading_dev_ratio = self._heading_dev_ratio_from_info(
            info_in=info_in,
            route_heading_dev_ratio=route_heading_dev_ratio,
        )

        clipped_progress = 0.0
        progress_bonus = 0.0
        progress_efficiency = 0.0
        progress_lateral_ratio = 0.0
        progress_heading_ratio = heading_dev_ratio
        #只有当进度开关打开的时候 才会有轻微的进度奖励
        if corrective_progress_enabled:
            progress_weight = float(corrective_progress_cfg.get("weight", corrective_progress_cfg.get("w_progress", 0.0)))
            progress_max_m = max(1.0e-6, float(corrective_progress_cfg.get("max_m", corrective_progress_cfg.get("progress_max_m", 1.2))))
            progress_min_m = float(corrective_progress_cfg.get("min_m", corrective_progress_cfg.get("progress_min_m", 0.0)))
            clipped_progress = float(np.clip(float(progress_delta_s), progress_min_m, progress_max_m))
            progress_bonus = float(progress_weight * (clipped_progress / progress_max_m))

            progress_lateral_safe_m = float(corrective_progress_cfg.get("lateral_safe_m", craft_cfg.get("lateral_safe_m", 0.2)))
            progress_lateral_max_default = float(
                craft_cfg.get("corrective_progress_lateral_max_m", craft_cfg.get("lateral_max_m", 3.0))
            )
            progress_lateral_max_m = max(
                progress_lateral_safe_m + 1.0e-6,
                float(corrective_progress_cfg.get("lateral_max_m", progress_lateral_max_default)),
            )
            progress_lateral_ratio = float(
                np.clip(
                    (float(lateral_error_m) - progress_lateral_safe_m)
                    / max(1.0e-6, progress_lateral_max_m - progress_lateral_safe_m),
                    0.0,
                    1.0,
                )
            )
            w_lateral_efficiency = float(
                corrective_progress_cfg.get(
                    "w_lateral_efficiency",
                    corrective_progress_cfg.get("w_global_efficiency", corrective_progress_cfg.get("w_g", 0.0)),
                )
            )
            w_heading_efficiency = float(
                corrective_progress_cfg.get("w_heading_efficiency", corrective_progress_cfg.get("w_h", 0.0))
            )
            efficiency_floor = float(corrective_progress_cfg.get("efficiency_floor", 0.0))
            progress_efficiency = float(
                max(
                    efficiency_floor,#设置最低效率,避免进度奖励被压到0；
                    math.exp(-w_lateral_efficiency * progress_lateral_ratio)
                    * math.exp(-w_heading_efficiency * progress_heading_ratio),
                )
            )
            progress_bonus = float(progress_bonus * progress_efficiency)
        self._last_heading_dev_ratio = float(heading_dev_ratio)

        events = self._craft_event_flags(
            craft_cfg=craft_cfg,
            info_in=info_in,
            lateral_error_m=lateral_error_m,
        )
        reward, corrective_info = compute_corrective_reward_scalar(
            params={**CRAFT_CORRECTIVE_DEFAULTS, **corrective_params},
            off_road=events["off_road"],
            emergency_lane=events["emergency_lane"],
            off_global_route=events["off_global_route"],
            run_red_light=events["run_red_light"],
            run_stop_sign=events["run_stop_sign"],
            collision=events["collision"],
        )
        ea_enabled = bool(ea_cfg.get("enable", False))
        ea_available = bool(info_in.get("ea_available", False))
        ea_risk = float(np.clip(float(info_in.get("ea_risk", 0.0)), 0.0, 1.0))
        ea_weight = float(ea_cfg.get("weight", ea_cfg.get("w_ea", 0.0)))
        ea_cost = float(ea_weight * ea_risk) if ea_enabled and ea_available else 0.0
        safety_cost = float(corrective_info["craft_corrective_total_cost"]) + float(ea_cost)
        reward = float(reward) + float(progress_bonus) - float(ea_cost)

        out_info = dict(info or {})
        out_info.update(
            {
                "reward_mode": "craft_corrective",
                "step_idx": int(step_idx),
                "reward": float(reward),
                "done": bool(done),
                #道路进度相关
                "progress_s": float(progress_s),
                "total_path_len_m": float(total_path_len),
                "completion_ratio": float(completion_ratio),
                "progress_delta_s": float(progress_delta_s),
                "progress_reward": float(clipped_progress),
                #是否开启progress bonus
                "craft_corrective_progress_enabled": bool(corrective_progress_enabled),
                "craft_corrective_progress_reward": float(progress_bonus),
                "craft_corrective_progress_efficiency": float(progress_efficiency),
                "craft_corrective_progress_lateral_ratio": float(progress_lateral_ratio),
                "craft_corrective_progress_heading_ratio": float(progress_heading_ratio),
                #横向/朝向误差
                "lateral_error_m": float(lateral_error_m),
                "path_heading_deg": float(_angle_to_deg(path_heading)),
                "ego_yaw_deg": float(_angle_to_deg(ego_yaw)),
                "yaw_path_err_deg": float(yaw_path_err_deg),
                "heading_dev_ratio": float(heading_dev_ratio),
                #安全/规则事件
                "off_road": bool(events["off_road"]),
                "emergency_lane": bool(events["emergency_lane"]),
                "off_global_route": bool(events["off_global_route"]),
                "run_red_light": float(events["run_red_light"]),
                "run_stop_sign": float(events["run_stop_sign"]),
                "collision": bool(events["collision"]),
                "static_collision": bool(events["static_collision"]),
                "dynamic_collision": bool(events["dynamic_collision"]),
                "cost_reward": float(safety_cost),
                #交通灯/停牌信号存在性
                "craft_red_light_signal_present": bool(events["red_light_signal_present"]),
                "craft_stop_sign_signal_present": bool(events["stop_sign_signal_present"]),
                #EA 风险
                "ea_enabled": bool(ea_enabled),
                "ea_available": bool(ea_available),
                "ea_risk": float(ea_risk),
                "ea_weight": float(ea_weight),
                "ea_cost": float(ea_cost),
                "ea_max": float(info_in.get("ea_max", 0.0)),
                "ea_min": float(info_in.get("ea_min", info_in.get("ea_max", 0.0))),
                "ea_mean": float(info_in.get("ea_mean", info_in.get("ea_max", 0.0))),
                "ea_evaluated_pairs": float(info_in.get("ea_evaluated_pairs", 0.0)),
                #off global route 诊断
                "off_global_route_source": str(events["off_global_route_source"]),
                "off_global_route_threshold_m": events["off_global_route_threshold_m"],
                
            }
        )
        out_info.update(self._reference_path_info())
        out_info.update(corrective_info)
        return TrackingRewardResult(reward=float(reward), info=out_info)


    def _compute_craft_closed_loop_reward(
        self,
        *,
        cfg: Dict[str, Any],
        craft_cfg: Dict[str, Any],
        info: Dict[str, Any] | None,
        step_idx: int,
        done: bool,
        progress_s: float,
        total_path_len: float,
        completion_ratio: float,
        progress_delta_s: float,
        lateral_error_m: float,
        path_heading: float,
        ego_yaw: float,
        yaw_path_err_deg: float,
    ) -> TrackingRewardResult:
        del cfg
        #reward = effective_progress + correction_reward + route_completed_reward - safety_cost
        info_in = info if isinstance(info, dict) else {}
        ea_cfg = self._ea_cfg()

        #进度效率奖励
        progress_max_m = max(1.0e-6, float(craft_cfg.get("progress_max_m", 1.2)))
        progress_min_m = float(craft_cfg.get("progress_min_m", 0.0))
        progress_weight = float(craft_cfg.get("progress_weight", craft_cfg.get("w_prog", 5.0)))
        clipped_progress = float(np.clip(float(progress_delta_s), progress_min_m, progress_max_m))
        progress_reward = progress_weight * (clipped_progress / progress_max_m)

        lateral_safe_m = float(craft_cfg.get("lateral_safe_m", 0.2))
        lateral_max_m = max(lateral_safe_m + 1.0e-6, float(craft_cfg.get("lateral_max_m", 3.0)))
        route_lateral_dev_ratio = float(
            np.clip((float(lateral_error_m) - lateral_safe_m) / max(1.0e-6, lateral_max_m - lateral_safe_m), 0.0, 1.0)
        )
        global_dev_ratio = float(info_in.get("global_dev_ratio", info_in.get("craft_global_dev_ratio", route_lateral_dev_ratio)))
        global_dev_ratio = float(np.clip(global_dev_ratio, 0.0, 1.0))
        center_dev_ratio = float(info_in.get("center_dev_ratio", info_in.get("craft_center_dev_ratio", route_lateral_dev_ratio)))
        center_dev_ratio = float(np.clip(center_dev_ratio, 0.0, 1.0))
        heading_max_deg = max(1.0e-6, float(craft_cfg.get("heading_max_deg", 60.0)))
        route_heading_dev_ratio = float(np.clip(float(yaw_path_err_deg) / heading_max_deg, 0.0, 1.0))
        heading_dev_ratio = self._heading_dev_ratio_from_info(
            info_in=info_in,
            route_heading_dev_ratio=route_heading_dev_ratio,
        )

        w_global_eff = float(craft_cfg.get("w_g", craft_cfg.get("w_lateral_efficiency", 3.0)))
        w_center_eff = float(craft_cfg.get("w_c", craft_cfg.get("w_center_efficiency", 0.0)))
        w_heading_eff = float(craft_cfg.get("w_h", craft_cfg.get("w_heading_efficiency", 2.0)))
        efficiency_floor = float(craft_cfg.get("efficiency_floor", 0.1))
        efficiency = (
            math.exp(-w_global_eff * global_dev_ratio)
            * math.exp(-w_center_eff * center_dev_ratio)
            * math.exp(-w_heading_eff * heading_dev_ratio)
        )
        efficiency = max(float(efficiency_floor), float(efficiency))
        effective_progress = progress_reward * efficiency

        #纠偏奖励
        correction_reward = 0.0
        correction_clip = float(craft_cfg.get("correction_clip", 0.5))
        ddev_clip = max(0.0, float(craft_cfg.get("ddev_clip", craft_cfg.get("correction_delta_clip", 1.0e6))))
        if self._last_craft_global_dev_ratio is not None:
            prev_global = float(self._last_craft_global_dev_ratio)
            prev_center = float(self._last_craft_center_dev_ratio if self._last_craft_center_dev_ratio is not None else prev_global)
            prev_heading = float(self._last_heading_dev_ratio if self._last_heading_dev_ratio is not None else 0.0)
            delta_global = float(np.clip(global_dev_ratio - prev_global, -ddev_clip, ddev_clip))
            delta_center = float(np.clip(center_dev_ratio - prev_center, -ddev_clip, ddev_clip))
            delta_heading = float(np.clip(heading_dev_ratio - prev_heading, -ddev_clip, ddev_clip))
            thresh_global = float(craft_cfg.get("correction_apply_thresh_global", 0.0))
            thresh_center = float(craft_cfg.get("correction_apply_thresh_center", craft_cfg.get("correction_apply_thresh_lateral", 0.0)))
            thresh_heading = float(craft_cfg.get("correction_apply_thresh_heading", 0.0))
            #只有当前偏离超过一定程度，才给纠偏奖励
            if global_dev_ratio > thresh_global:
                correction_reward += float(craft_cfg.get("k_g", craft_cfg.get("correction_lateral_weight", 0.4))) * (-delta_global)
            if center_dev_ratio > thresh_center:
                correction_reward += float(craft_cfg.get("k_c", craft_cfg.get("correction_center_weight", 0.0))) * (-delta_center)
            if heading_dev_ratio > thresh_heading:
                correction_reward += float(craft_cfg.get("k_h", craft_cfg.get("correction_heading_weight", 0.3))) * (-delta_heading)
            correction_reward = float(np.clip(correction_reward, -correction_clip, correction_clip))
            #车有正常往前走时，纠偏奖励完整；如果几乎没前进，纠偏奖励会被压到最多一半，避免 policy 靠原地调整方向/位置刷 correction。
            correction_reward *= 0.5 + 0.5 * float(np.clip(clipped_progress / progress_max_m, 0.0, 1.0))
        self._last_craft_global_dev_ratio = float(global_dev_ratio)
        self._last_craft_center_dev_ratio = float(center_dev_ratio)
        self._last_heading_dev_ratio = float(heading_dev_ratio)


        #Route completion 以及 safety惩罚
        #碰撞检测
        events = self._craft_event_flags(
            craft_cfg=craft_cfg,
            info_in=info_in,
            lateral_error_m=lateral_error_m,
        )
        static_collision = bool(events["static_collision"])
        dynamic_collision = bool(events["dynamic_collision"])
        has_static_collision_cost = craft_cfg.get("collision_cost_static", None) is not None
        has_dynamic_collision_cost = craft_cfg.get("collision_cost_dynamic", None) is not None
        collision_terminal_cost = 0.0
        if (not has_static_collision_cost) and (not has_dynamic_collision_cost) and (static_collision or dynamic_collision):
            collision_terminal_cost = float(craft_cfg.get("term_collision", 30.0))
        static_collision_cost = float(craft_cfg.get("collision_cost_static", 0.0)) if (static_collision and has_static_collision_cost) else 0.0
        dynamic_collision_cost = float(craft_cfg.get("collision_cost_dynamic", 0.0)) if (dynamic_collision and has_dynamic_collision_cost) else 0.0
        #严重的横向 航向偏离
        severe_lateral_cost = 0.0
        severe_heading_cost = 0.0
        severe_lateral_error_m = craft_cfg.get("severe_lateral_error_m", None)
        severe_heading_error_deg = craft_cfg.get("severe_heading_error_deg", None)
        if severe_lateral_error_m is not None and float(lateral_error_m) > float(severe_lateral_error_m):
            severe_lateral_cost = float(craft_cfg.get("severe_lateral_cost", 0.0))
        if severe_heading_error_deg is not None and float(yaw_path_err_deg) > float(severe_heading_error_deg):
            severe_heading_cost = float(craft_cfg.get("severe_heading_cost", 0.0))
        # 地图/交通规则类违规
        off_road_cost = float(craft_cfg.get("cost_off_road", 5.0)) if bool(events["off_road"]) else 0.0
        opposite_lane_cost = float(craft_cfg.get("cost_opposite_lane", 1.0)) if bool(events["opposite_lane"]) else 0.0
        off_global_route_cost = float(craft_cfg.get("cost_off_global_route", 4.0)) if bool(events["off_global_route"]) else 0.0
        emergency_lane_cost = float(craft_cfg.get("cost_emergency_lane", 3.0)) if bool(events["emergency_lane"]) else 0.0
        red_light_cost = float(craft_cfg.get("cost_red_light", 6.0)) if bool(events["red_light"]) else 0.0
        stop_sign_cost = float(craft_cfg.get("cost_stop_sign", 6.0)) if bool(events["stop_sign"]) else 0.0
        route_deviation_cost = float(craft_cfg.get("term_route_dev", 30.0)) if bool(events["route_deviation"]) else 0.0
        route_completed_reward = float(craft_cfg.get("reward_completed", 0.0)) if bool(events["route_completed"]) else 0.0
        ea_enabled = bool(ea_cfg.get("enable", False))
        ea_available = bool(info_in.get("ea_available", False))
        ea_risk = float(np.clip(float(info_in.get("ea_risk", 0.0)), 0.0, 1.0))
        ea_weight = float(ea_cfg.get("weight", ea_cfg.get("w_ea", 0.0)))
        ea_cost = float(ea_weight * ea_risk) if ea_enabled and ea_available else 0.0

        safety_cost = (
            static_collision_cost
            + dynamic_collision_cost
            + collision_terminal_cost
            + severe_lateral_cost
            + severe_heading_cost
            + off_road_cost
            + opposite_lane_cost
            + off_global_route_cost
            + emergency_lane_cost
            + red_light_cost
            + stop_sign_cost
            + route_deviation_cost
            + ea_cost
        )
        reward = effective_progress + correction_reward + route_completed_reward - safety_cost
        positive_reward = float(effective_progress + max(0.0, correction_reward) + max(0.0, route_completed_reward))

        out_info = dict(info or {})
        out_info.update(
            {
                # 基础 step 信息
                "reward_mode": "craft_closed_loop",
                "step_idx": int(step_idx),
                "reward": float(reward),
                "done": bool(done),

                # 路径进度信息
                "progress_s": float(progress_s),
                "total_path_len_m": float(total_path_len),
                "completion_ratio": float(completion_ratio),
                "progress_delta_s": float(progress_delta_s),
                # 历史字段名：这里记录的是裁剪后的前进距离，不是加权 reward
                "progress_reward": float(clipped_progress),

                # reward 正负项汇总
                "positive_reward": float(positive_reward),
                "gated_positive_reward": float(positive_reward),
                "cost_reward": float(safety_cost + max(0.0, -correction_reward)),

                # 路径跟踪误差
                "lateral_error_m": float(lateral_error_m),
                "path_heading_deg": float(_angle_to_deg(path_heading)),
                "ego_yaw_deg": float(_angle_to_deg(ego_yaw)),
                "yaw_path_err_deg": float(yaw_path_err_deg),

                # 碰撞标记
                "static_collision": bool(static_collision),
                "dynamic_collision": bool(dynamic_collision),

                # craft 前进、效率、纠偏分量
                "craft_progress_reward": float(progress_reward),
                "craft_raw_progress_reward": float(progress_reward),
                "craft_effective_progress": float(effective_progress),
                "craft_efficiency": float(efficiency),
                "craft_correction_reward": float(correction_reward),
                "craft_safety_cost": float(safety_cost),

                # efficiency 和 correction 使用的偏离比例
                "craft_global_dev_ratio": float(global_dev_ratio),
                "craft_center_dev_ratio": float(center_dev_ratio),
                "craft_lateral_dev_ratio": float(route_lateral_dev_ratio),
                "heading_dev_ratio": float(heading_dev_ratio),

                # 碰撞和严重偏离成本
                "craft_static_collision_cost": float(static_collision_cost),
                "craft_dynamic_collision_cost": float(dynamic_collision_cost),
                "craft_collision_terminal_cost": float(collision_terminal_cost),
                "craft_severe_lateral_cost": float(severe_lateral_cost),
                "craft_severe_heading_cost": float(severe_heading_cost),

                # 地图、交通规则、路线相关成本
                "craft_off_road_cost": float(off_road_cost),
                "craft_opposite_lane_cost": float(opposite_lane_cost),
                "craft_off_global_route_cost": float(off_global_route_cost),
                "craft_emergency_lane_cost": float(emergency_lane_cost),
                "craft_red_light_cost": float(red_light_cost),
                "craft_stop_sign_cost": float(stop_sign_cost),
                "craft_red_light_signal_present": bool(events["red_light_signal_present"]),
                "craft_stop_sign_signal_present": bool(events["stop_sign_signal_present"]),
                "craft_route_deviation_cost": float(route_deviation_cost),
                "craft_route_completed_reward": float(route_completed_reward),

                # EA 风险项
                "ea_enabled": bool(ea_enabled),
                "ea_available": bool(ea_available),
                "ea_max": float(info_in.get("ea_max", 0.0)),
                "ea_min": float(info_in.get("ea_min", info_in.get("ea_max", 0.0))),
                "ea_mean": float(info_in.get("ea_mean", info_in.get("ea_max", 0.0))),
                "ea_risk": float(ea_risk),
                "ea_weight": float(ea_weight),
                "ea_cost": float(ea_cost),
                "ea_evaluated_pairs": float(info_in.get("ea_evaluated_pairs", 0.0)),

                # 原始事件标记和来源
                "off_road": bool(events["off_road"]),
                "opposite_lane": bool(events["opposite_lane"]),
                "off_global_route": bool(events["off_global_route"]),
                "off_global_route_source": str(events["off_global_route_source"]),
                "off_global_route_threshold_m": events["off_global_route_threshold_m"],

                # 兼容旧日志字段
                "pos_dev": float(lateral_error_m),
                "pos_dev_source": "path_projection",
                "yaw_err_deg": float(yaw_path_err_deg),
            }
        )
        out_info.update(self._reference_path_info())
        return TrackingRewardResult(reward=float(reward), info=out_info)

#Unified post-processing after all reward modes have been computed
#In this program only effective for step path reward mode
    def apply_terminal_penalty(
        self,
        *,
        reward: float,
        info: Dict[str, Any],
        term_cfg: Dict[str, Any],
        terminal_kind: str | None,
    ) -> TrackingRewardResult:
        success_bonus = float(term_cfg.get("success_bonus", 0.0))
        out_info = dict(info)
        reward_out = float(reward)
        if terminal_kind == "env_done" and success_bonus != 0.0:
            reward_out += success_bonus
            out_info["terminal_success_bonus"] = float(success_bonus)
            out_info["terminal_success_bonus_applied"] = True

        penalty = float(term_cfg.get("penalty", 0.0))
        if penalty == 0.0:
            out_info["reward"] = float(reward_out)
            return TrackingRewardResult(reward=float(reward_out), info=out_info)

        apply_on_failure = bool(term_cfg.get("apply_on_failure", True))
        apply_on_timeout = bool(term_cfg.get("apply_on_timeout", False))
        apply_on_env_done = bool(term_cfg.get("apply_on_env_done", False))
        should_apply = False
        if terminal_kind == "failure" and apply_on_failure:
            should_apply = True
        elif terminal_kind == "timeout" and apply_on_timeout:
            should_apply = True
        elif terminal_kind == "env_done" and apply_on_env_done:
            should_apply = True

        if not should_apply:
            out_info["reward"] = float(reward_out)
            return TrackingRewardResult(reward=float(reward_out), info=out_info)

        out_info["terminal_kind"] = terminal_kind
        out_info["terminal_penalty"] = float(penalty)
        out_info["terminal_penalty_applied"] = True
        out_info["reward"] = float(reward_out + penalty)
        return TrackingRewardResult(reward=float(reward_out + penalty), info=out_info)
