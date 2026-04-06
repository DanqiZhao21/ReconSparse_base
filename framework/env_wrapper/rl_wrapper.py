import math
import bisect
from typing import Any, Dict, Tuple

import numpy as np

from reconsimulator.envs.nus import ReconSimulator
from reconsimulator.envs.metrics import oriented_box, EGO_LENGTH, EGO_WIDTH
from reconsimulator.envs.metrics_cache import load_scene_env_cache
from shapely.geometry import Polygon

from framework.rewards import TrackingRewardComputer


class RLReconEnv:
    """
    Minimal RL wrapper around `ReconSimulator` to provide a Gymnasium-style
    interface: (obs, reward, terminated, truncated, info).

    - Computes a simple reward based on deviation from expert trajectory.
    - Adapts action to the env's expected continuous pose action format.
    - Keeps env unmodified; all compatibility handled here.
    """

    def __init__(
        self,
        cuda: int = 0,
        scene: int = 0,
        reward_cfg: Dict[str, Any] | None = None,
        debug: bool = False,
        *,
        render_w: int | None = None,
        render_h: int | None = None,
    ):
        # debug=False → 使用候选锚点规划；debug=True/flag=True → 使用专家轨迹
        if render_w is None or render_h is None:
            self.env = ReconSimulator(cuda=cuda, scene=scene, debug=bool(debug))
        else:
            self.env = ReconSimulator(cuda=cuda, scene=scene, debug=bool(debug), render_w=int(render_w), render_h=int(render_h))
        self.reward_cfg = reward_cfg or {}
        self._reward_computer = TrackingRewardComputer(self.reward_cfg)

        # Step index for aligning with NuScenes keyframes (2Hz → every 5 steps at 10Hz)
        self._step_idx: int = 0

        # Jerk-related running state (for smoothness rewards)
        self._last_xz: np.ndarray | None = None
        self._last_yaw: float | None = None
        self._last_v: float | None = None
        self._last_yaw_rate: float | None = None
        self._last_a: float | None = None
        self._last_yaw_acc: float | None = None

        # Debounce counters for threshold-based termination
        self._yaw_exceed_count: int = 0
        self._xz_exceed_count: int = 0

        # Episode termination reason (kept for compatibility)
        self._ep_done_reason: str | None = None

        # Env cache for map-based objects (static/dynamic) per scene
        self._env_cache_scene_id: int | None = None
        self._env_cache: Dict[int, Dict[str, Any]] = {}
        self._env_cache_keys: list[int] = []

    def set_external_plan_local_xyyaw(self, plan: Any) -> None:
        if plan is None:
            self.env._external_plan_local_xyyaw = None
            return

        arr = np.asarray(plan, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
            self.env._external_plan_local_xyyaw = None
            return

        self.env._external_plan_local_xyyaw = np.asarray(arr[:, :3], dtype=np.float64).copy()

    def reset(
        self,
        scene: int | None = None,
        *,
        start_frame: int | None = None,
        step_frames: int | None = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        # `ReconSimulator.reset(seed)` internally uses `update(seed)` → seed acts as scene id.
        seed = scene if scene is not None else self.env.scene
        options: Dict[str, Any] = {}
        if start_frame is not None:
            options["start_frame"] = int(start_frame)
        if step_frames is not None:
            options["step_frames"] = int(step_frames)
        obs, info = self.env.reset(seed=seed, options=options if len(options) else None)

        # Reset reward history
        self._last_xz = None
        self._last_yaw = None
        self._last_v = None
        self._last_yaw_rate = None
        self._last_a = None
        self._last_yaw_acc = None
        self._reward_computer.reset()

        # Reset debounce counters
        self._yaw_exceed_count = 0
        self._xz_exceed_count = 0
        self._ep_done_reason = None
        # Reset step counter: track the underlying simulator's frame index.
        # This makes metrics-cache alignment correct for both step_frames=1 (10Hz)
        # and step_frames=5 (2Hz keyframes).
        try:
            self._step_idx = int(getattr(self.env, "now_frame"))
        except Exception:
            self._step_idx = int(start_frame) if start_frame is not None else 0
        return obs, info

    def _get_env_cache_snapshot(self, *, scene_id: int, step_idx: int) -> Dict[str, Any] | None:
        if self._env_cache_scene_id != int(scene_id):
            cache = load_scene_env_cache(int(scene_id)) or {}
            self._env_cache_scene_id = int(scene_id)
            self._env_cache = dict(cache)
            self._env_cache_keys = sorted(self._env_cache.keys())

        if not self._env_cache:
            return None

        sidx = int(step_idx)
        if sidx in self._env_cache:
            return self._env_cache.get(sidx)

        # Fallback: use nearest previous key (aligned to step_frames)
        if self._env_cache_keys:
            pos = bisect.bisect_right(self._env_cache_keys, sidx) - 1
            if pos >= 0:
                return self._env_cache.get(self._env_cache_keys[pos])
        return None

    def _compute_collision_flags(self) -> tuple[bool, bool]:
        """Compute static/dynamic collision using env_cache polygons."""
        try:
            scene_id = int(getattr(self.env, "scene", 0))
            step_idx = int(self._step_idx)
            snap = self._get_env_cache_snapshot(scene_id=scene_id, step_idx=step_idx)
            if not isinstance(snap, dict):
                return False, False

            # Ego pose in map coordinates (x,z -> map x,y)
            ego_x = float(self.env.start_ego[:3, 3][0])
            ego_y = float(self.env.start_ego[:3, 3][2])
            Rm = self.env.start_ego[:3, :3]
            ego_yaw = float(math.atan2(float(Rm[2, 0]), float(Rm[0, 0])))
            ego_poly = oriented_box(ego_x, ego_y, float(EGO_LENGTH), float(EGO_WIDTH), float(ego_yaw))

            static_collision = False
            dynamic_collision = False

            for obj in snap.get("static_objects", []) or []:
                poly = obj.get("poly", None)
                if isinstance(poly, list) and len(poly) >= 3:
                    if ego_poly.intersects(Polygon(poly)):
                        static_collision = True
                        break

            for obj in snap.get("dynamic_objects", []) or []:
                poly = obj.get("poly", None)
                if isinstance(poly, list) and len(poly) >= 3:
                    if ego_poly.intersects(Polygon(poly)):
                        dynamic_collision = True
                        break

            return bool(static_collision), bool(dynamic_collision)
        except Exception:
            print("⚠️ Warning: failed to compute collision flags from env_cache.")
            return False, False
    
#ADD
    def step(self, action: Tuple[float, float, float, int]):
        # Normalize action to env expected format.
        # Supported runtime actions:
        # - Expert mode: (x, y, yaw, flag=1)
        # - Continuous policy mode: (x, y, yaw, flag=2)
        if len(action) == 4:
            x, y, yaw, flag = float(action[0]), float(action[1]), float(action[2]), int(action[3])
            if int(flag) not in {1, 2}:
                raise ValueError(f"Unsupported action flag={flag}; expected expert flag=1 or continuous flag=2")
            env_action = (x, y, yaw, flag)
        else:
            raise ValueError(f"Unsupported action format (len={len(action)}): {action}")

        obs, terminated, truncated, info = self.env.step(env_action)

        # Normalize simulator-provided yaw error to a wrapped, minimal angular difference.
        # This avoids false positives near the wrap boundary (e.g., 179° vs -179°).
        #把角度差强行拉回[−180°,180°]
        #TODO:处理yaw误差
        #NOTE 处理YAW误差
        try: 
            if isinstance(info, dict):
                exp_yaw_deg = info.get("exp_yaw_deg", None)
                act_yaw_deg = info.get("act_yaw_deg", None)
                if (exp_yaw_deg is not None) and (act_yaw_deg is not None):
                    dy = float(act_yaw_deg) - float(exp_yaw_deg)
                    dy_wrap = ((dy + 180.0) % 360.0) - 180.0
                    yaw_err_wrapped = abs(float(dy_wrap))
                    if yaw_err_wrapped > 120.0:
                        yaw_err_wrapped = abs(abs(act_yaw_deg+180)-abs(exp_yaw_deg))
                    if info.get("yaw_err_deg") is not None and info.get("yaw_err_deg_raw") is None:
                        info["yaw_err_deg_raw"] = info.get("yaw_err_deg")
                    info["yaw_err_deg"] = float(yaw_err_wrapped)
                    info["yaw_err_deg_signed"] = float(dy_wrap)#可用于分析左右偏转
                    # print("🎯[StepDebug] exp_yaw_deg={:.2f}, act_yaw_deg={:.2f}, yaw_err_deg_raw={:.2f} -> yaw_err_deg_wrapped={:.2f}".format(exp_yaw_deg, act_yaw_deg, info.get("yaw_err_deg_raw", float('nan')), yaw_err_wrapped))
        except Exception:
            pass
        # Update step counter: always follow simulator frame index if available.
        try:
            self._step_idx = int(getattr(self.env, "now_frame"))#使用frame的真实帧号 +5
        except Exception:
            try:
                self._step_idx += 1
            except Exception:
                self._step_idx = 1

        # Compute collision flags from env_cache and expose in info
        static_collision, dynamic_collision = self._compute_collision_flags()
        if info is None:
            info = {}
        if isinstance(info, dict):
            info["static_collision"] = bool(static_collision)
            info["dynamic_collision"] = bool(dynamic_collision)

        done = bool(terminated or truncated)

        #NOTE 进行终止判断
        # Optional threshold-based early termination (gameover) using tracking error/collisions.
        # This stays compatible with episode-level rewards: we still compute a single
        # scalar reward at the (possibly early) termination boundary.
        term_cfg = {}
        try:
            if isinstance(self.reward_cfg, dict):
                term_cfg = self.reward_cfg.get("terminal", {}) or {}
        except Exception:
            term_cfg = {}

        if not done and bool(term_cfg.get("enable", False)):
            xz_thr = term_cfg.get("xz_err_m_max", None)
            yaw_thr = term_cfg.get("yaw_err_deg_max", None)
            terminate_on_xz = bool(term_cfg.get("terminate_on_xz_err", True))
            terminate_on_yaw = bool(term_cfg.get("terminate_on_yaw_err", True))
            terminate_on_static_col = bool(term_cfg.get("terminate_on_static_collision", True))
            terminate_on_dynamic_col = bool(term_cfg.get("terminate_on_dynamic_collision", True))

            xz_err = None
            yaw_err = None
            if isinstance(info, dict):
                xz_err = info.get("xz_err_m", None)
                yaw_err = info.get("yaw_err_deg", None)

            # Debounce / robustness:
            # - require N consecutive steps above threshold before terminating
            # - default N=1 keeps the previous behavior
            try:
                yaw_patience = int(term_cfg.get("yaw_err_patience_steps", 1))
            except Exception:
                yaw_patience = 1
            try:
                xz_patience = int(term_cfg.get("xz_err_patience_steps", term_cfg.get("xy_err_patience_steps", 1)))
            except Exception:
                xz_patience = 1
            yaw_patience = max(1, yaw_patience)
            xz_patience = max(1, xz_patience)

            try:
                if terminate_on_yaw and yaw_thr is not None and yaw_err is not None and float(yaw_err) > float(yaw_thr):
                    self._yaw_exceed_count += 1
                else:
                    self._yaw_exceed_count = 0
            except Exception:
                self._yaw_exceed_count = 0
            try:
                if terminate_on_xz and xz_thr is not None and xz_err is not None and float(xz_err) > float(xz_thr):
                    self._xz_exceed_count += 1
                else:
                    self._xz_exceed_count = 0
            except Exception:
                self._xz_exceed_count = 0

            reasons: list[str] = []
            try:
                if terminate_on_xz and xz_thr is not None and xz_err is not None and self._xz_exceed_count >= int(xz_patience):
                    reasons.append("xz_err")
            except Exception:
                pass
            try:
                if terminate_on_yaw and yaw_thr is not None and yaw_err is not None and self._yaw_exceed_count >= int(yaw_patience):
                    reasons.append("yaw_err")
            except Exception:
                pass
            # Collision-based termination using last computed map metrics (updated every 5 steps)
            try:
                static_col = False
                dynamic_col = False
                if isinstance(info, dict):
                    static_col = bool(info.get("static_collision", info.get("metrics_static_collision", False)))
                    dynamic_col = bool(info.get("dynamic_collision", info.get("metrics_dynamic_collision", False)))
                if terminate_on_static_col and static_col:
                    reasons.append("static_collision")
                if terminate_on_dynamic_col and dynamic_col:
                    reasons.append("dynamic_collision")
            except Exception:
                pass
            #如果有任何阈值被触发 → 强制结束 episode
            if len(reasons) > 0:
                terminated = True
                truncated = False
                done = True
                self._ep_done_reason = "+".join(reasons)
                if info is None:
                    info = {}
                if isinstance(info, dict):
                    info["done_reason"] = self._ep_done_reason
                    info["terminated_by_threshold"] = True
                    if xz_thr is not None:
                        info["xz_err_m_max"] = float(xz_thr)
                    if yaw_thr is not None:
                        info["yaw_err_deg_max"] = float(yaw_thr)

            # Always expose current debounce counters for debugging.
            if isinstance(info, dict):
                info["yaw_err_exceed_count"] = int(self._yaw_exceed_count)
                info["xz_err_exceed_count"] = int(self._xz_exceed_count)
                info["yaw_err_patience_steps"] = int(yaw_patience)
                info["xz_err_patience_steps"] = int(xz_patience)
                
        #NOTE 计算奖励
        terminal_kind = None
        if done and isinstance(info, dict):
            failure_done = bool(info.get("terminated_by_threshold", False)) or bool(
                info.get("static_collision", False) or info.get("dynamic_collision", False)
            )
            if failure_done:
                terminal_kind = "failure"
                info.setdefault("done_reason", self._ep_done_reason or "failure")
            elif bool(truncated):
                terminal_kind = "timeout"
                info.setdefault("done_reason", "timeout")
            elif bool(terminated):
                terminal_kind = "env_done"
                info.setdefault("done_reason", self._ep_done_reason or "env_done")
            info["terminal_kind"] = terminal_kind

        reward, info = self._compute_reward(info, done=done)
        if done and isinstance(info, dict) and bool(term_cfg.get("enable", False)):
            reward_result = self._reward_computer.apply_terminal_penalty(
                reward=float(reward),
                info=info,
                term_cfg=term_cfg,
                terminal_kind=terminal_kind,
            )
            reward = float(reward_result.reward)
            info = reward_result.info
        return obs, reward, terminated, truncated, info


 #   # -------------------- Reward -------------------- #（from frame/reward）
    #NOTE reward = -(rpd + rhd + rsc + rdc + jerk_pen + yaw_jerk_pen)
    def _compute_reward(self, info: Dict[str, Any] | None = None, *, done: bool = False) -> tuple[float, Dict[str, Any]]:
        reward_result = self._reward_computer.compute(
            env=self.env,
            info=info,
            step_idx=int(self._step_idx),
            done=bool(done),
        )
        log_data = dict(reward_result.info)
        try:
            logging_cfg = {}
            if isinstance(self.reward_cfg, dict):
                logging_cfg = self.reward_cfg.get("logging", {}) or {}
            wandb_cfg = logging_cfg.get("wandb", {}) or {}
            if bool(wandb_cfg.get("enable", False)):
                import wandb
                prefix = str(wandb_cfg.get("prefix", "rl"))
                wandb.log({f"{prefix}/" + k: v for k, v in log_data.items()})
        except Exception:
            pass
        return float(reward_result.reward), reward_result.info
