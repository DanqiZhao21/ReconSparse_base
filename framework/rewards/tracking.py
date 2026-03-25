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

    def reset(self) -> None:
        self._last_xz = None
        self._last_yaw = None
        self._last_v = None
        self._last_yaw_rate = None
        self._last_a = None
        self._last_yaw_acc = None

    def compute(
        self,
        *,
        env: Any,
        info: Dict[str, Any] | None,
        step_idx: int,
        done: bool,
    ) -> TrackingRewardResult:
        cfg = self.reward_cfg or {}
        dt = max(1e-6, float(cfg.get("dt", 0.5)))
        dmax = float(cfg.get("dmax", 2.0))
        psi_max_deg = float(cfg.get("psi_max_deg", 30.0))
        w_pos = float(cfg.get("w_pos", 2.0))
        w_heading = float(cfg.get("w_heading", 1.0))
        w_static = float(cfg.get("w_static", 5.0))
        w_dynamic = float(cfg.get("w_dynamic", 5.0))
        w_longitudinal_jerk = float(cfg.get("w_longitudinal_jerk", 0.0))
        w_yaw_jerk = float(cfg.get("w_yaw_jerk", 0.0))
        jerk_clip = float(cfg.get("jerk_clip", 50.0))

        ego_xz = env.start_ego[:3, 3][[0, 2]]
        pos_dev = 0.0
        try:
            #TODO:这里是否换成不参照traj 而是使用Path来
            step_frames = int(getattr(env, "step_frames", 1))
            now_frame = int(getattr(env, "now_frame", 0))
            exp_list = getattr(env, "all_expert_ego", None)
            if isinstance(exp_list, list) and len(exp_list) > 0 and step_frames > 0:
                idx = max(0, min(int(now_frame // step_frames), len(exp_list) - 1))
                exp_pose = exp_list[idx]
                exp_xz = np.asarray(exp_pose[:3, 3][[0, 2]], dtype=np.float32)
                pos_dev = float(np.linalg.norm(exp_xz - ego_xz))
            else:
                raise ValueError("expert list unavailable")
        except Exception:
            expert_xz_list = getattr(env, "expert_pair", [])
            if len(expert_xz_list) > 0:
                expert_arr = np.asarray(expert_xz_list, dtype=np.float32)
                pos_dev = float(np.linalg.norm(expert_arr - ego_xz, axis=1).min())

        rot = env.start_ego[:3, :3]
        yaw = math.atan2(float(rot[0, 0]), float(rot[2, 0]))

        yaw_err_deg = 0.0
        try:
            if isinstance(info, dict) and info.get("yaw_err_deg") is not None:
                yaw_err_deg = float(info.get("yaw_err_deg"))
        except Exception:
            yaw_err_deg = 0.0

        static_collision = bool(info.get("static_collision", False)) if isinstance(info, dict) else False
        dynamic_collision = bool(info.get("dynamic_collision", False)) if isinstance(info, dict) else False

        jerk = 0.0
        yaw_jerk = 0.0
        if (w_longitudinal_jerk != 0.0 or w_yaw_jerk != 0.0):
            cur_xz = np.asarray(ego_xz, dtype=np.float32)
            if self._last_xz is None:
                self._last_xz = cur_xz
                self._last_yaw = float(yaw)
            dxz = cur_xz - self._last_xz
            heading = np.array([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float32)
            v = float(np.dot(dxz, heading) / dt)
            yaw_rate = float(_wrap_angle(float(yaw) - float(self._last_yaw if self._last_yaw is not None else yaw)) / dt)
            if self._last_v is None:
                self._last_v = v
                self._last_yaw_rate = yaw_rate
            a = (v - float(self._last_v)) / dt
            yaw_acc = (yaw_rate - float(self._last_yaw_rate)) / dt
            if self._last_a is None:
                self._last_a = a
                self._last_yaw_acc = yaw_acc
            jerk = float(np.clip((a - float(self._last_a)) / dt, -jerk_clip, jerk_clip))
            yaw_jerk = float(np.clip((yaw_acc - float(self._last_yaw_acc)) / dt, -jerk_clip, jerk_clip))
            self._last_xz = cur_xz
            self._last_yaw = float(yaw)
            self._last_v = v
            self._last_yaw_rate = yaw_rate
            self._last_a = a
            self._last_yaw_acc = yaw_acc

        rpd = w_pos * max(0.0, float(pos_dev) - dmax)
        rhd = w_heading * max(0.0, float(yaw_err_deg) - psi_max_deg)
        rsc = w_static if static_collision else 0.0
        rdc = w_dynamic if dynamic_collision else 0.0
        jerk_pen = w_longitudinal_jerk * abs(float(jerk))
        yaw_jerk_pen = w_yaw_jerk * abs(float(yaw_jerk))
        reward = -(rpd + rhd + rsc + rdc + jerk_pen + yaw_jerk_pen)

        out_info = dict(info or {})
        out_info.update(
            {
                "reward_mode": "step",
                "step_idx": int(step_idx),
                "reward": float(reward),
                "pos_dev": float(pos_dev),
                "yaw_err_deg": float(yaw_err_deg),
                "longitudinal_jerk": float(jerk),
                "yaw_jerk": float(yaw_jerk),
                "static_collision": bool(static_collision),
                "dynamic_collision": bool(dynamic_collision),
                "rpd": float(rpd),
                "rhd": float(rhd),
                "rsc": float(rsc),
                "rdc": float(rdc),
                "jerk_pen": float(jerk_pen),
                "yaw_jerk_pen": float(yaw_jerk_pen),
                "done": bool(done),
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