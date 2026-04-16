from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


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


class TrackingRewardComputer:
    def __init__(self, reward_cfg: Dict[str, Any] | None = None) -> None:
        self.reward_cfg = reward_cfg or {}
        self._last_xz: np.ndarray | None = None
        self._last_yaw: float | None = None
        self._last_v: float | None = None
        self._last_yaw_rate: float | None = None
        self._last_a: float | None = None
        self._last_yaw_acc: float | None = None
        self._path_xy: np.ndarray | None = None
        self._path_s: np.ndarray | None = None
        self._last_progress_s: float | None = None

    def reset(self) -> None:
        self._last_xz = None
        self._last_yaw = None
        self._last_v = None
        self._last_yaw_rate = None
        self._last_a = None
        self._last_yaw_acc = None
        self._path_xy = None
        self._path_s = None
        self._last_progress_s = None

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

    @staticmethod
    def _ego_yaw_from_pose(pose: np.ndarray) -> float:
        rot = np.asarray(pose[:3, :3], dtype=np.float64)
        return float(math.atan2(float(rot[2, 0]), float(rot[0, 0])))

    def _reference_points_from_env(self, env: Any) -> np.ndarray:
        expert_poses = getattr(env, "all_expert_ego", None)
        if isinstance(expert_poses, list) and len(expert_poses) > 0:
            pts = [np.asarray(pose[:3, 3][[0, 2]], dtype=np.float64) for pose in expert_poses]
            return np.asarray(pts, dtype=np.float64)

        expert_pairs = getattr(env, "expert_pair", None)
        if isinstance(expert_pairs, list) and len(expert_pairs) > 0:
            return np.asarray(expert_pairs, dtype=np.float64)

        ego = np.asarray(env.start_ego[:3, 3][[0, 2]], dtype=np.float64)
        return ego.reshape(1, 2)

    @staticmethod
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
        raw = self._reference_points_from_env(env)
        raw = self._dedupe_polyline(raw)
        dense = self._densify_polyline(raw, ds=interp_ds)
        dense = self._dedupe_polyline(dense)
        dense_s = self._path_arclength(dense)
        self._path_xy = dense
        self._path_s = dense_s
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

        dt = max(1e-6, float(cfg.get("dt", 0.5)))
        ego_xz = np.asarray(env.start_ego[:3, 3][[0, 2]], dtype=np.float64)
        ego_yaw = self._ego_yaw_from_pose(np.asarray(env.start_ego, dtype=np.float64))

        path_xy, path_s = self._ensure_reference_path(env)#Frenet 坐标
        progress_s, lateral_error_m, _proj, tangent = self._project_point_to_path(ego_xz, path_xy, path_s)
        path_heading = float(math.atan2(float(tangent[1]), float(tangent[0])))
        yaw_path_err_rad = _wrap_angle(float(ego_yaw) - float(path_heading))#航向误差
        yaw_path_err_deg = abs(_angle_to_deg(yaw_path_err_rad))

        if self._last_progress_s is None:
            progress_delta_s = 0.0
        else:
            progress_delta_s = float(progress_s - self._last_progress_s)
        self._last_progress_s = float(progress_s)
        #进度奖励;这一帧比上一帧，在路径上多走了多少距离
        progress_forward_cap = float(path_cfg.get("progress_forward_cap_m", 2.0))
        progress_backward_cap = float(path_cfg.get("progress_backward_cap_m", 0.5))
        w_progress = float(path_cfg.get("w_progress", 0.0))
        progress_reward = float(np.clip(progress_delta_s, -progress_backward_cap, progress_forward_cap))
        progress_term = w_progress * progress_reward
        #横向惩罚
        lateral_free = float(path_cfg.get("lateral_free_m", 0.3))
        lateral_delta = float(path_cfg.get("lateral_huber_delta_m", 0.5))
        lateral_excess = max(0.0, float(lateral_error_m) - lateral_free)
        lateral_penalty = _huber(lateral_excess, lateral_delta)
        lateral_term = float(path_cfg.get("w_lateral", cfg.get("w_pos", 0.0))) * lateral_penalty
        #航向角误差惩罚yaw
        yaw_free_deg = float(path_cfg.get("yaw_free_deg", 5.0))
        yaw_delta_deg = float(path_cfg.get("yaw_huber_delta_deg", 10.0))
        yaw_excess_deg = max(0.0, float(yaw_path_err_deg) - yaw_free_deg)
        yaw_penalty = _huber(yaw_excess_deg, yaw_delta_deg)
        yaw_term = float(path_cfg.get("w_yaw", cfg.get("w_heading", 0.0))) * yaw_penalty
        #碰撞惩罚
        static_collision = bool(info.get("static_collision", False)) if isinstance(info, dict) else False
        dynamic_collision = bool(info.get("dynamic_collision", False)) if isinstance(info, dict) else False
        static_collision_penalty = float(collision_cfg.get("w_static", cfg.get("w_static", 5.0))) if static_collision else 0.0
        dynamic_collision_penalty = float(collision_cfg.get("w_dynamic", cfg.get("w_dynamic", 5.0))) if dynamic_collision else 0.0

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

#TODO:
        anchor_progress_term = 0.0
        anchor_lateral_term = 0.0
        reward = (
            progress_term
            - lateral_term
            - yaw_term
            - static_collision_penalty
            - dynamic_collision_penalty
            - jerk_term
            - yaw_jerk_term
            + anchor_progress_term
            - anchor_lateral_term
        )

        out_info = dict(info or {})
        out_info.update(
            {
                "reward_mode": "step_path",
                "step_idx": int(step_idx),
                "reward": float(reward),
                "done": bool(done),
                "progress_s": float(progress_s),
                "progress_delta_s": float(progress_delta_s),
                "progress_reward": float(progress_reward),
                "progress_term": float(progress_term),#####
                "lateral_error_m": float(lateral_error_m),
                "lateral_penalty": float(lateral_penalty),
                "lateral_term": float(lateral_term),#####
                "path_heading_deg": float(_angle_to_deg(path_heading)),
                "ego_yaw_deg": float(_angle_to_deg(ego_yaw)),
                "yaw_path_err_deg": float(yaw_path_err_deg),
                "yaw_penalty": float(yaw_penalty),
                "yaw_term": float(yaw_term),
                "longitudinal_jerk": float(jerk),
                "yaw_jerk": float(yaw_jerk),
                "jerk_penalty": float(jerk_penalty),
                "yaw_jerk_penalty": float(yaw_jerk_penalty),
                "jerk_term": float(jerk_term),#####
                "yaw_jerk_term": float(yaw_jerk_term),#####
                "static_collision": bool(static_collision),
                "dynamic_collision": bool(dynamic_collision),
                "static_collision_penalty": float(static_collision_penalty),
                "dynamic_collision_penalty": float(dynamic_collision_penalty),
                "anchor_progress_term": float(anchor_progress_term),#####
                "anchor_lateral_term": float(anchor_lateral_term),#####
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
        return TrackingRewardResult(reward=float(reward), info=out_info)

    def apply_terminal_penalty(
        self,
        *,
        reward: float,
        info: Dict[str, Any],
        term_cfg: Dict[str, Any],
        terminal_kind: str | None,
    ) -> TrackingRewardResult:
        penalty = float(term_cfg.get("penalty", 0.0))
        if penalty == 0.0:
            return TrackingRewardResult(reward=float(reward), info=info)

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
            return TrackingRewardResult(reward=float(reward), info=info)

        out_info = dict(info)
        out_info["terminal_kind"] = terminal_kind
        out_info["terminal_penalty"] = float(penalty)
        out_info["terminal_penalty_applied"] = True
        out_info["reward"] = float(reward + penalty)
        return TrackingRewardResult(reward=float(reward + penalty), info=out_info)
