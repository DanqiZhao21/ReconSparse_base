import math
import bisect
from typing import Any, Dict, Tuple

import numpy as np

from .nus import ReconSimulator
from .metrics import oriented_box, EGO_LENGTH, EGO_WIDTH
from .metrics_cache import load_scene_env_cache
from shapely.geometry import Polygon


class RLReconEnv:
    """
    Minimal RL wrapper around `ReconSimulator` to provide a Gymnasium-style
    interface: (obs, reward, terminated, truncated, info).

    - Computes a simple reward based on deviation from expert trajectory.
    - Adapts action to the env's expected triplet (ax, ay, flag).
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

        # Reset debounce counters
        self._yaw_exceed_count = 0
        self._xz_exceed_count = 0
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
    def step(self, action: Tuple[int, int] | Tuple[int, int, int]):
        # Normalize action to env expected format.
        # - Anchor mode: (ax, ay) or (ax, ay, flag)
        # - Continuous mode: (x, y, yaw, flag=2)
        if len(action) == 2:
            ax, ay = int(action[0]), int(action[1])
            flag = 0
            env_action = (ax, ay, flag)
        elif len(action) == 3:
            ax, ay, flag = int(action[0]), int(action[1]), int(action[2])
            env_action = (ax, ay, flag)
        elif len(action) == 4:
            x, y, yaw, flag = float(action[0]), float(action[1]), float(action[2]), int(action[3])
            env_action = (x, y, yaw, flag)
        else:
            raise ValueError(f"Unsupported action format (len={len(action)}): {action}")

        obs, terminated, truncated, info = self.env.step(env_action)

        # Normalize simulator-provided yaw error to a wrapped, minimal angular difference.
        # This avoids false positives near the wrap boundary (e.g., 179° vs -179°).
        #把角度差强行拉回[−180°,180°]
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
                xz_patience = int(term_cfg.get("xz_err_patience_steps", 1))
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
        reward, info = self._compute_reward(info, done=done)
        return obs, reward, terminated, truncated, info


 #ADD   # -------------------- Reward -------------------- #
    #NOTE reward = -(rpd + rhd + rsc + rdc + jerk_pen + yaw_jerk_pen)
    def _compute_reward(self, info: Dict[str, Any] | None = None, *, done: bool = False) -> tuple[float, Dict[str, Any]]:
        """
        Step-only reward with components {rdc, rsc, rpd, rhd} + jerk penalties.
        """
        cfg = self.reward_cfg or {}
        dt = float(cfg.get("dt", 0.5))
        dt = max(1e-6, dt)
        
        #THRESHOLDS AND WEIGHTS
        dmax = float(cfg.get("dmax", 2.0))
        psi_max_deg = float(cfg.get("psi_max_deg", 30.0))
        
        w_pos = float(cfg.get("w_pos", 2.0))
        w_heading = float(cfg.get("w_heading", 1.0))
        w_static = float(cfg.get("w_static", 5.0))
        w_dynamic = float(cfg.get("w_dynamic", 5.0))

        w_longitudinal_jerk = float(cfg.get("w_longitudinal_jerk", 0.0))
        w_yaw_jerk = float(cfg.get("w_yaw_jerk", 0.0))
        jerk_clip = float(cfg.get("jerk_clip", 50.0))

        #NOTE位置偏差 --- Position deviation vs expert ---
        
        '''
        expert_xz_list 不是单一点，而是整条参考轨迹的若干离散点;
        车辆当前可能不正好落在某个轨迹点上，所以计算 ego 与所有轨迹点的欧式距离;
        如果没有专家轨迹则 既定pos-dev=0
        '''
        # Current ego position (x,z)
        ego_xz = self.env.start_ego[:3, 3][[0, 2]]
        # Prefer expert pose at the current frame index (aligned by step_frames).
        pos_dev = 0.0
        try:
            step_frames = int(getattr(self.env, "step_frames", 1))
            now_frame = int(getattr(self.env, "now_frame", 0))
            exp_list = getattr(self.env, "all_expert_ego", None)
            if isinstance(exp_list, list) and len(exp_list) > 0 and step_frames > 0:
                idx = int(now_frame // step_frames)
                idx = max(0, min(idx, len(exp_list) - 1))
                exp_pose = exp_list[idx]
                exp_xz = np.asarray(exp_pose[:3, 3][[0, 2]], dtype=np.float32)
                pos_dev = float(np.linalg.norm(exp_xz - ego_xz))
                
                # print(
                #     f"🎯[RewardDebug] ego_xz.shape={ego_xz.shape}, "
                #     f"expert_arr.shape={exp_xz.shape}, "
                #     f"pos_dev={pos_dev}"
                # )
            else:
                raise ValueError("expert list unavailable")
        except Exception:
            # Fallback: nearest point over dense expert trajectory
            expert_xz_list = getattr(self.env, "expert_pair", [])
            if len(expert_xz_list) == 0:
                pos_dev = 0.0
            else:
                expert_arr = np.asarray(expert_xz_list, dtype=np.float32)
                dists = np.linalg.norm(expert_arr - ego_xz, axis=1)
                pos_dev = float(dists.min())

        #NOTE方向偏差--- Heading deviation vs expert yaw ---
        # Extract yaw from current 3x3 rotation (x-z plane)
        # r_hd​=wheading​⋅max(0,∣ψerr​∣−ψmax​)
        R = self.env.start_ego[:3, :3]
        yaw = math.atan2(float(R[0, 0]), float(R[2, 0]))

        # Use simulator-provided tracking yaw error (act_yaw vs exp_yaw), in degrees.
        # If missing, best-effort compute from exp_yaw_deg/act_yaw_deg.
        def _wrap_angle_deg(a: float) -> float:
            return float(math.degrees(math.atan2(math.sin(math.radians(float(a))), math.cos(math.radians(float(a))))))

        env_yaw_err_deg = None
        try:
            if isinstance(info, dict) and info.get("yaw_err_deg") is not None:
                env_yaw_err_deg = float(info.get("yaw_err_deg"))
        except Exception:
            env_yaw_err_deg = None
        if env_yaw_err_deg is None:
            try:
                if isinstance(info, dict) and (info.get("exp_yaw_deg") is not None) and (info.get("act_yaw_deg") is not None):
                    env_yaw_err_deg = abs(_wrap_angle_deg(float(info.get("act_yaw_deg")) - float(info.get("exp_yaw_deg"))))
            except Exception:
                env_yaw_err_deg = None
        if env_yaw_err_deg is None:
            env_yaw_err_deg = 0.0
        #NOTE:碰撞损失
        # --- Collisions (computed in step; otherwise treated as False) ---
        static_collision_flag = bool(info.get("static_collision", False)) if isinstance(info, dict) else False
        dynamic_collision_flag = bool(info.get("dynamic_collision", False)) if isinstance(info, dict) else False

        #NOTE平滑性（jerk）
        # Approximate longitudinal velocity from pose delta projected onto heading.
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
            jerk = (a - float(self._last_a)) / dt
            yaw_jerk = (yaw_acc - float(self._last_yaw_acc)) / dt

            # Update history
            self._last_xz = cur_xz
            self._last_yaw = float(yaw)
            self._last_v = v
            self._last_yaw_rate = yaw_rate
            self._last_a = a
            self._last_yaw_acc = yaw_acc

            jerk = float(np.clip(jerk, -jerk_clip, jerk_clip))
            yaw_jerk = float(np.clip(yaw_jerk, -jerk_clip, jerk_clip))
        # Penalize only when exceeding thresholds (paper's four components)
        rpd = w_pos * max(0.0, float(pos_dev) - dmax)
        rhd = w_heading * max(0.0, float(env_yaw_err_deg) - psi_max_deg)
        rsc = w_static if static_collision_flag else 0.0
        rdc = w_dynamic if dynamic_collision_flag else 0.0
        # Optional smoothness
        jerk_pen = w_longitudinal_jerk * abs(float(jerk))
        yaw_jerk_pen = w_yaw_jerk * abs(float(yaw_jerk))

        reward = -(rpd + rhd + rsc + rdc + jerk_pen + yaw_jerk_pen)
        # Prepare wandb logging payload for step-level metrics
        log_data = {
            "step_idx": int(self._step_idx),
            "reward": float(reward),
            "rpd": float(rpd),#无
            "rhd": float(rhd),
            "rsc": float(rsc),#无
            "rdc": float(rdc),#无
            "jerk_pen": float(jerk_pen),
            "yaw_jerk_pen": float(yaw_jerk_pen),
            "pos_dev": float(pos_dev),#无  这个指标不应该为0 ！！因为肯定不会完全一样
            "yaw_err_deg": float(env_yaw_err_deg),
            "static_collision": bool(static_collision_flag),
            "dynamic_collision": bool(dynamic_collision_flag),
        }
        if info is None:
            info = {}

        info.update(
            {
                "reward_mode": "step",
                "pos_dev": float(pos_dev),
                # Keep/propagate env yaw_err_deg (act vs exp). This is the only yaw error we use.
                "yaw_err_deg": float(env_yaw_err_deg),
                "longitudinal_jerk": float(jerk),
                "yaw_jerk": float(yaw_jerk),
                "reward": float(reward),
                # expose collision flags used for reward
                "static_collision": bool(static_collision_flag),
                "dynamic_collision": bool(dynamic_collision_flag),
                # four-component breakdown
                "rpd": float(rpd),
                "rhd": float(rhd),
                "rsc": float(rsc),
                "rdc": float(rdc),
            }
        )
        # Optional wandb logging
        try:
            logging_cfg = {}
            if isinstance(self.reward_cfg, dict):
                logging_cfg = self.reward_cfg.get("logging", {}) or {}
            wandb_cfg = logging_cfg.get("wandb", {}) or {}
            if bool(wandb_cfg.get("enable", False)):
                import wandb
                # Optional prefix for namespacing
                prefix = str(wandb_cfg.get("prefix", "rl"))
                wandb.log({f"{prefix}/" + k: v for k, v in log_data.items()})
        except Exception:
            pass
        return float(reward), info

    def _finalize_episode_reward(self, *, dt: float, done_reason: str) -> tuple[float, Dict[str, Any]]:
        """Episode reward is disabled; return neutral values for compatibility."""
        metrics = {
            "episode_len": 0,
            "done_reason": str(done_reason),
        }
        return 0.0, metrics

    def finalize_episode_reward(self, *, done_reason: str = "timeout") -> tuple[float, Dict[str, Any]]:
        """Public helper; episode reward disabled, returns zero reward and minimal metrics."""
        cfg = self.reward_cfg or {}
        dt = float(cfg.get("dt", 0.1))
        dt = max(1e-6, dt)
        return self._finalize_episode_reward(dt=dt, done_reason=str(done_reason))


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
