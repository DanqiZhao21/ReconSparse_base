import math
from typing import Any, Dict, Tuple

import numpy as np

from .nus import ReconSimulator


class RLReconEnv:
    """
    Minimal RL wrapper around `ReconSimulator` to provide a Gymnasium-style
    interface: (obs, reward, terminated, truncated, info).

    - Computes a simple reward based on deviation from expert trajectory.
    - Adapts action to the env's expected triplet (ax, ay, flag).
    - Keeps env unmodified; all compatibility handled here.
    """

    def __init__(self, cuda: int = 0, scene: int = 0, reward_cfg: Dict[str, Any] | None = None, debug: bool = False):
        # debug=False → 使用候选锚点规划；debug=True/flag=True → 使用专家轨迹
        self.env = ReconSimulator(cuda=cuda, scene=scene, debug=bool(debug))
        self.reward_cfg = reward_cfg or {}

        # Step index for aligning with NuScenes keyframes (2Hz → every 5 steps at 10Hz)
        self._step_idx: int = 0

        # Cache of last computed map-based metrics (updated every 5 steps)
        self._last_metrics: Dict[str, Any] = {
            "drivable_compliance": 1.0,
            "driving_direction_compliance": 1.0,
            "static_collision": False,
            "dynamic_collision": False,
        }

        # Jerk-related running state (for smoothness rewards)
        self._last_xz: np.ndarray | None = None
        self._last_yaw: float | None = None
        self._last_v: float | None = None
        self._last_yaw_rate: float | None = None
        self._last_a: float | None = None
        self._last_yaw_acc: float | None = None

        # Episode buffers (for episode-level reward modes)
        self._ep_xz: list[np.ndarray] = []
        self._ep_yaw: list[float] = []
        self._ep_xz_err_m: list[float] = []
        self._ep_yaw_err_deg: list[float] = []
        # Episode-level aggregates for compliance & collisions
        self._ep_drivable_sum: float = 0.0
        self._ep_direction_sum: float = 0.0
        self._ep_metrics_count: int = 0
        self._ep_static_collision_any: bool = False
        self._ep_dynamic_collision_any: bool = False

        # Episode termination reason (for sparse terminal penalties)
        self._ep_done_reason: str | None = None

    def reset(self, scene: int | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        # `ReconSimulator.reset(seed)` internally uses `update(seed)` → seed acts as scene id.
        seed = scene if scene is not None else self.env.scene
        obs, info = self.env.reset(seed=seed)

        # Reset reward history
        self._last_xz = None
        self._last_yaw = None
        self._last_v = None
        self._last_yaw_rate = None
        self._last_a = None
        self._last_yaw_acc = None

        # Reset episode buffers
        self._ep_xz = []
        self._ep_yaw = []
        self._ep_xz_err_m = []
        self._ep_yaw_err_deg = []
        self._ep_done_reason = None
        # Reset episode compliance/collision aggregates
        self._ep_drivable_sum = 0.0
        self._ep_direction_sum = 0.0
        self._ep_metrics_count = 0
        self._ep_static_collision_any = False
        self._ep_dynamic_collision_any = False
        # Reset step counter
        self._step_idx = 0
        # Reset metrics cache
        self._last_metrics = {
            "drivable_compliance": 1.0,
            "driving_direction_compliance": 1.0,
            "static_collision": False,
            "dynamic_collision": False,
        }
        return obs, info

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
        # Update step counter
        try:
            self._step_idx += 1
        except Exception:
            self._step_idx = 1

        # Record episode kinematics for episode-level rewards
        try:#NOTE 记录 episode 运动状态
            ego_xz = self.env.start_ego[:3, 3][[0, 2]].astype(np.float32)
            R = self.env.start_ego[:3, :3]
            yaw = float(math.atan2(R[1, 0], R[0, 0]))
            self._ep_xz.append(ego_xz)
            self._ep_yaw.append(yaw)
        except Exception:
            pass

        # Record episode tracking errors if provided by Recondreamer simulator info
        try:#NOTE 记录误差信息
            if isinstance(info, dict):
                if info.get("xz_err_m") is not None:
                    self._ep_xz_err_m.append(float(info["xz_err_m"]))
                if info.get("yaw_err_deg") is not None:
                    self._ep_yaw_err_deg.append(float(info["yaw_err_deg"]))
        except Exception:
            pass

        done = bool(terminated or truncated)

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

            reasons: list[str] = []
            try:
                if terminate_on_xz and xz_thr is not None and xz_err is not None and float(xz_err) > float(xz_thr):
                    reasons.append("xz_err")
            except Exception:
                pass
            try:
                if terminate_on_yaw and yaw_thr is not None and yaw_err is not None and float(yaw_err) > float(yaw_thr):
                    reasons.append("yaw_err")
            except Exception:
                pass
            # Collision-based termination using last computed map metrics (updated every 5 steps)
            try:
                lm = self._last_metrics or {}
                if terminate_on_static_col and bool(lm.get("static_collision", False)):
                    reasons.append("static_collision")
                if terminate_on_dynamic_col and bool(lm.get("dynamic_collision", False)):
                    reasons.append("dynamic_collision")
            except Exception:
                pass
            #NOTE 如果有任何阈值被触发 → 强制结束 episode
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

        # Record a best-effort termination reason for metrics/logging.
        if done and self._ep_done_reason is None:
            if bool(terminated) and not bool(truncated):
                self._ep_done_reason = "env_terminated"
            elif bool(truncated):
                self._ep_done_reason = "env_truncated"
            else:
                self._ep_done_reason = "done"
            if isinstance(info, dict):
                info.setdefault("done_reason", self._ep_done_reason)
        reward, info = self._compute_reward(info, done=done)

        # ----- Per-step metrics (map-based) every 5 steps aligned to keyframes -----
        try:
            metrics_cfg = {}
            if isinstance(self.reward_cfg, dict):
                metrics_cfg = self.reward_cfg.get("metrics", {}) or {}
            enable_metrics = bool(metrics_cfg.get("enable", True))
            if enable_metrics and (self._step_idx % 5 == 0):
                scene_id = int(getattr(self.env, "scene", 0))
                m = None
                # 1) Prefer environment snapshot cache → online computation with current ego pose
                try:
                    from .metrics_cache import get_env_snapshot
                    snap = get_env_snapshot(scene_id=scene_id, step_idx=self._step_idx)
                except Exception:
                    snap = None
                if isinstance(snap, dict) and len(snap) > 0:
                    # Build ego state from current pose (x,z → map x,y)
                    try:
                        ego_x = float(self.env.start_ego[:3, 3][0])
                        ego_y = float(self.env.start_ego[:3, 3][2])
                        Rm = self.env.start_ego[:3, :3]
                        ego_yaw = float(math.atan2(Rm[1, 0], Rm[0, 0]))
                    except Exception:
                        ego_x = 0.0; ego_y = 0.0; ego_yaw = 0.0
                    ego_state = {"x": ego_x, "y": ego_y, "yaw": ego_yaw}
                    from .metrics import compute_metrics_from_snapshot
                    m = compute_metrics_from_snapshot(snap, ego_state)

                # 2) Backward compatible: try precomputed metrics if no env snapshot
                if not m:
                    try:
                        from .metrics_cache import get_precomputed_step_metrics
                        m = get_precomputed_step_metrics(scene_id=scene_id, step_idx=self._step_idx)
                    except Exception:
                        m = None

                # 3) Fallback to on-the-fly NuScenes compute if nothing cached
                if not m:
                    dataroot = str(metrics_cfg.get("dataroot", "/OpenDataset/nuscenes/nuscenes/v1.0-trainval"))
                    version = str(metrics_cfg.get("version", "v1.0-trainval"))
                    from .metrics import compute_step_metrics
                    m = compute_step_metrics(scene_id=scene_id, step_idx=self._step_idx, dataroot=dataroot, version=version)

                if isinstance(m, dict) and len(m) > 0:
                    # Cache for reward computation on all steps until next update
                    self._last_metrics = {
                        "drivable_compliance": float(m.get("drivable_compliance", 1.0)),
                        "driving_direction_compliance": float(m.get("driving_direction_compliance", 1.0)),
                        "static_collision": bool(m.get("static_collision", False)),
                        "dynamic_collision": bool(m.get("dynamic_collision", False)),
                    }
                    # Episode-level aggregates: accumulate compliance and collisions
                    try:
                        self._ep_drivable_sum += float(m.get("drivable_compliance", 1.0))
                        self._ep_direction_sum += float(m.get("driving_direction_compliance", 1.0))
                        self._ep_metrics_count += 1
                        if bool(m.get("static_collision", False)):
                            self._ep_static_collision_any = True
                        if bool(m.get("dynamic_collision", False)):
                            self._ep_dynamic_collision_any = True
                    except Exception:
                        pass
                if isinstance(info, dict) and isinstance(m, dict):
                    # Expose minimal metrics in info
                    info.update({
                        "metrics_drivable_compliance": float(m.get("drivable_compliance", 0.0)),
                        "metrics_driving_direction_compliance": float(m.get("driving_direction_compliance", 0.0)),
                        "metrics_collision": bool(m.get("collision", False)),
                        "metrics_collision_status": str(m.get("collision_status", "No-Collision")),
                        "metrics_static_collision": bool(m.get("static_collision", False)),
                        "metrics_dynamic_collision": bool(m.get("dynamic_collision", False)),
                        "metrics_step_idx": int(self._step_idx),
                    })
        except Exception:
            # Best-effort; do not break training if metrics fail
            pass
        return obs, reward, terminated, truncated, info

    # -------------------- Reward -------------------- #
    def _compute_reward(self, info: Dict[str, Any] | None = None, *, done: bool = False) -> tuple[float, Dict[str, Any]]:
        """
        Reward modes (set by env.reward.mode in YAML):
        - step: per-step reward with components {rdc, rsc, rpd, rhd} + optional jerk terms.
        - episode: per-step reward = 0; on termination, compute episode-level jerk reward + optional terminal penalty.

        Components inspired by the paper:
        - rdc: dynamic collision penalty (if ego bbox intersects dynamic agents).
        - rsc: static collision penalty (if ego bbox intersects static obstacles).
        - rpd: positional deviation beyond dmax from expert trajectory (Euclidean distance to nearest expert point).
        - rhd: heading deviation beyond psi_max from expert tangent.

        Jerk terms use finite differences of ego pose over dt.
        """
        cfg = self.reward_cfg or {}
        mode_raw = str(cfg.get("mode", "step")).lower()
        episode_mode = mode_raw.startswith("episode") or (mode_raw == "episode")
        step_mode = not episode_mode
        dt = float(cfg.get("dt", 0.1))
        dt = max(1e-6, dt)

        dmax = float(cfg.get("dmax", 2.0))
        psi_max_deg = float(cfg.get("psi_max_deg", 30.0))
        w_pos = float(cfg.get("w_pos", 2.0))
        w_heading = float(cfg.get("w_heading", 1.0))
        w_static = float(cfg.get("w_static", 5.0))
        w_dynamic = float(cfg.get("w_dynamic", 5.0))

        w_longitudinal_jerk = float(cfg.get("w_longitudinal_jerk", 0.0))
        w_yaw_jerk = float(cfg.get("w_yaw_jerk", 0.0))
        jerk_clip = float(cfg.get("jerk_clip", 50.0))

        # Episode-level: return 0 until done; finalize in _finalize_episode_reward
        if episode_mode:
            if info is None:
                info = {}
            info.update({"reward_mode": "episode", "reward": 0.0})
            if done:
                done_reason = self._ep_done_reason or "done"
                ep_reward, metrics = self._finalize_episode_reward(dt=dt, done_reason=done_reason)
                # Provide per-episode scalar reward and metrics for trainer to backfill.
                info["episode_reward"] = float(ep_reward)
                info["episode_metrics"] = metrics
                info["episode_len"] = int(metrics.get("episode_len", 0))
                info["done_reason"] = str(metrics.get("done_reason", done_reason))
            return 0.0, info

        # --- Position deviation vs expert ---
        #NOTE
        '''
        expert_xz_list 不是单一点，而是整条参考轨迹的若干离散点;
        车辆当前可能不正好落在某个轨迹点上，所以计算 ego 与所有轨迹点的欧式距离;
        如果没有专家轨迹则 既定pos-dev=0
        '''
        # Current ego position (x,z)
        ego_xz = self.env.start_ego[:3, 3][[0, 2]]
        # Precomputed expert points in x,z
        expert_xz_list = self.env.expert_pair  # list of shape (..., 2)
        if len(expert_xz_list) == 0:
            pos_dev = 0.0
            print("⚠️ Warning: no expert trajectory points available for reward computation.")
        else:
            # Find nearest reference
            distances = [float(np.linalg.norm(ego_xz - ref)) for ref in expert_xz_list]
            pos_dev = min(distances)

        # --- Heading deviation vs expert yaw ---
        # Extract yaw from current 3x3 rotation
        R = self.env.start_ego[:3, :3]
        yaw = math.atan2(R[1, 0], R[0, 0])  # z-yaw assuming standard frame

        # Approximate expert yaw by local tangent between nearest two points
        if len(expert_xz_list) >= 2:
            # Find nearest index
            nearest_idx = int(np.argmin([float(np.linalg.norm(ego_xz - ref)) for ref in expert_xz_list]))
            j = max(0, min(nearest_idx + 1, len(expert_xz_list) - 1))
            p0 = expert_xz_list[nearest_idx]
            p1 = expert_xz_list[j]
            vec = p1 - p0
            ref_yaw = math.atan2(float(vec[1]), float(vec[0]))
            yaw_err = abs(_wrap_angle(yaw - ref_yaw))
        else:
            yaw_err = 0.0
#FIXME:
        # --- Collisions (from last computed map metrics) ---
        lm = self._last_metrics or {}
        static_collision_flag = bool(lm.get("static_collision", False))
        dynamic_collision_flag = bool(lm.get("dynamic_collision", False))

        # --- Smoothness (jerk) ---
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
        yaw_err_deg = float(math.degrees(yaw_err))
        rpd = w_pos * max(0.0, float(pos_dev) - dmax)
        rhd = w_heading * max(0.0, float(yaw_err_deg) - psi_max_deg)
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
            "rpd": float(rpd),
            "rhd": float(rhd),
            "rsc": float(rsc),
            "rdc": float(rdc),
            "jerk_pen": float(jerk_pen),
            "yaw_jerk_pen": float(yaw_jerk_pen),
            "pos_dev": float(pos_dev),
            "yaw_err_deg": float(yaw_err_deg),
            "static_collision": bool(static_collision_flag),
            "dynamic_collision": bool(dynamic_collision_flag),
        }
        if info is None:
            info = {}
        info.update(
            {
                "reward_mode": "step",
                "pos_dev": float(pos_dev),
                "yaw_err_deg": float(yaw_err_deg),
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

    def _finalize_episode_jerk_reward(self, *, dt: float) -> tuple[float, Dict[str, Any]]:
        """Compute episode-level jerk metrics using numpy gradient.

        Returns:
            ep_reward: scalar (negative penalty)
            metrics: dict with longitudinal_jerk_mean / yaw_jerk_mean / episode_len
        """
        cfg = self.reward_cfg or {}
        w_longitudinal_jerk = float(cfg.get("w_longitudinal_jerk", 0.05))
        w_yaw_jerk = float(cfg.get("w_yaw_jerk", 0.05))
        jerk_clip = float(cfg.get("jerk_clip", 50.0))

        n = min(len(self._ep_xz), len(self._ep_yaw))
        if n < 2:
            metrics = {
                "episode_len": int(n),
                "longitudinal_jerk_mean": 0.0,
                "yaw_jerk_mean": 0.0,
                "xz_err_mean_m": float(np.mean(self._ep_xz_err_m)) if len(self._ep_xz_err_m) > 0 else 0.0,
                "xz_err_max_m": float(np.max(self._ep_xz_err_m)) if len(self._ep_xz_err_m) > 0 else 0.0,
                "yaw_err_mean_deg": float(np.mean(self._ep_yaw_err_deg)) if len(self._ep_yaw_err_deg) > 0 else 0.0,
                "yaw_err_max_deg": float(np.max(self._ep_yaw_err_deg)) if len(self._ep_yaw_err_deg) > 0 else 0.0,
            }
            return 0.0, metrics

        xz = np.stack(self._ep_xz[:n], axis=0).astype(np.float32)  # (T,2)
        yaw = np.asarray(self._ep_yaw[:n], dtype=np.float32)  # (T,)

        # Velocity (projected along heading) and yaw rate
        heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1).astype(np.float32)  # (T,2)
        dxz = np.diff(xz, axis=0, prepend=xz[:1])
        v = (dxz * heading).sum(axis=-1) / float(dt)  # (T,)

        # unwrap yaw then diff
        yaw_unwrapped = np.unwrap(yaw)
        yaw_rate = np.diff(yaw_unwrapped, axis=0, prepend=yaw_unwrapped[:1]) / float(dt)

        # time sequence
        t = np.arange(n, dtype=np.float32) * float(dt)

        # jerk sequences by gradient
        a = np.gradient(v, t)
        jerk = np.gradient(a, t)
        yaw_acc = np.gradient(yaw_rate, t)
        yaw_jerk = np.gradient(yaw_acc, t)
        #NOTE 平均绝对值作为惩罚
        jerk = np.clip(jerk, -jerk_clip, jerk_clip)
        yaw_jerk = np.clip(yaw_jerk, -jerk_clip, jerk_clip)

        longitudinal_jerk_mean = float(np.mean(np.abs(jerk)))
        yaw_jerk_mean = float(np.mean(np.abs(yaw_jerk)))

        ep_reward = -(w_longitudinal_jerk * longitudinal_jerk_mean + w_yaw_jerk * yaw_jerk_mean)
        metrics = {
            "episode_len": int(n),
            "longitudinal_jerk_mean": float(longitudinal_jerk_mean),
            "yaw_jerk_mean": float(yaw_jerk_mean),
            "xz_err_mean_m": float(np.mean(self._ep_xz_err_m)) if len(self._ep_xz_err_m) > 0 else 0.0,
            "xz_err_max_m": float(np.max(self._ep_xz_err_m)) if len(self._ep_xz_err_m) > 0 else 0.0,
            "yaw_err_mean_deg": float(np.mean(self._ep_yaw_err_deg)) if len(self._ep_yaw_err_deg) > 0 else 0.0,
            "yaw_err_max_deg": float(np.max(self._ep_yaw_err_deg)) if len(self._ep_yaw_err_deg) > 0 else 0.0,
        }
        return float(ep_reward), metrics
#NOTE 最终 episode reward = jerk reward + 稀疏 terminal penalty

    def _finalize_episode_reward(self, *, dt: float, done_reason: str) -> tuple[float, Dict[str, Any]]:
        """Finalize episode reward as a sum of episode-level penalties:

        - Trajectory tracking: mean exceedance over thresholds for position/yaw (rpd_ep/rhd_ep)
        - Compliance: mean non-compliance for drivable/direction
        - Collisions: static/dynamic collision penalty if occurred during episode
        - Smoothness: mean absolute jerk penalties (longitudinal & yaw)
        - Plus optional sparse terminal penalty
        """
        jerk_reward, metrics = self._finalize_episode_jerk_reward(dt=dt)
        cfg = self.reward_cfg or {}

        term_cfg = {}
        try:
            if isinstance(self.reward_cfg, dict):
                term_cfg = self.reward_cfg.get("terminal", {}) or {}
        except Exception:
            term_cfg = {}

        # -------- Episode-level tracking exceedance (beyond thresholds) --------
        try:
            dmax = float(cfg.get("dmax", 2.0))
            psi_max_deg = float(cfg.get("psi_max_deg", 30.0))
            w_pos = float(cfg.get("w_pos", 2.0))
            w_heading = float(cfg.get("w_heading", 1.0))
        except Exception:
            dmax, psi_max_deg, w_pos, w_heading = 2.0, 30.0, 2.0, 1.0

        if len(self._ep_xz_err_m) > 0:
            import numpy as _np
            rpd_ep = w_pos * float(_np.mean(_np.maximum(_np.asarray(self._ep_xz_err_m, dtype=_np.float32) - float(dmax), 0.0)))
        else:
            rpd_ep = 0.0
        if len(self._ep_yaw_err_deg) > 0:
            import numpy as _np
            rhd_ep = w_heading * float(_np.mean(_np.maximum(_np.asarray(self._ep_yaw_err_deg, dtype=_np.float32) - float(psi_max_deg), 0.0)))
        else:
            rhd_ep = 0.0

        # -------- Episode-level compliance (drivable/direction) --------
        try:
            w_drivable = float(cfg.get("w_drivable", 2.0))
            w_direction = float(cfg.get("w_direction", 1.0))
        except Exception:
            w_drivable, w_direction = 2.0, 1.0
        if self._ep_metrics_count > 0:
            drv_mean = float(self._ep_drivable_sum) / float(self._ep_metrics_count)
            dir_mean = float(self._ep_direction_sum) / float(self._ep_metrics_count)
        else:
            drv_mean = 1.0
            dir_mean = 1.0
        drivable_noncomp = max(0.0, 1.0 - drv_mean)
        direction_noncomp = max(0.0, 1.0 - dir_mean)
        r_drv_ep = w_drivable * drivable_noncomp
        r_dir_ep = w_direction * direction_noncomp

        # -------- Episode-level collisions (occurred at least once) --------
        try:
            w_static = float(cfg.get("w_static", 5.0))
            w_dynamic = float(cfg.get("w_dynamic", 5.0))
        except Exception:
            w_static, w_dynamic = 5.0, 5.0
        rsc_ep = w_static if bool(self._ep_static_collision_any) else 0.0
        rdc_ep = w_dynamic if bool(self._ep_dynamic_collision_any) else 0.0

        # -------- Episode-level smoothness (already negative in jerk_reward) --------
        # Convert jerk_reward(negative) back to positive penalty to unify sum
        jerk_pen_ep = -float(jerk_reward)

        # Total episode penalty (positive), reward is negative of this plus terminal penalty
        total_episode_penalty = float(rpd_ep + rhd_ep + rsc_ep + rdc_ep + r_drv_ep + r_dir_ep + jerk_pen_ep)

        terminal_penalty = 0.0
        if bool(term_cfg.get("enable", False)):
            penalty = float(term_cfg.get("penalty", 0.0))
            apply_on_failure = bool(term_cfg.get("apply_on_failure", True))
            apply_on_timeout = bool(term_cfg.get("apply_on_timeout", False))
            apply_on_env_done = bool(term_cfg.get("apply_on_env_done", False))#环境强制终止 env_terminated / env_truncated 不加
            #NOTE 根据 done_reason 决定是否应用终止惩罚
            if done_reason == "timeout":
                if apply_on_timeout:
                    terminal_penalty = penalty
            elif done_reason.startswith("env_"):
                if apply_on_env_done:
                    terminal_penalty = penalty
            else:
                if apply_on_failure:
                    terminal_penalty = penalty

        # Episode reward: negative total penalty + terminal penalty (penalty usually positive)
        total_reward = float(-total_episode_penalty + terminal_penalty)
        metrics = dict(metrics)
        metrics.update(
            {
                "done_reason": str(done_reason),
                "terminal_penalty": float(terminal_penalty),
                # components (episode-level)
                "rpd_ep": float(rpd_ep),
                "rhd_ep": float(rhd_ep),
                "rsc_ep": float(rsc_ep),
                "rdc_ep": float(rdc_ep),
                "drivable_noncomp_mean": float(drivable_noncomp),
                "direction_noncomp_mean": float(direction_noncomp),
                "jerk_pen_ep": float(jerk_pen_ep),
                # rewards
                "episode_reward_jerk": float(jerk_reward),
                "episode_reward_total": float(total_reward),
            }
        )
        # Optional wandb logging for episode-level metrics
        try:
            logging_cfg = {}
            if isinstance(cfg, dict):
                logging_cfg = cfg.get("logging", {}) or {}
            wandb_cfg = logging_cfg.get("wandb", {}) or {}
            if bool(wandb_cfg.get("enable", False)):
                import wandb
                prefix = str(wandb_cfg.get("prefix", "rl"))
                wandb.log({f"{prefix}/episode/" + k: v for k, v in metrics.items()})
        except Exception:
            pass
        return total_reward, metrics

    def finalize_episode_reward(self, *, done_reason: str = "timeout") -> tuple[float, Dict[str, Any]]:
        """Public helper to finalize episode reward/metrics.

        Useful when the training loop forces an episode boundary (e.g. max_steps)
        without the underlying simulator reporting terminated/truncated.
        """
        if self._ep_done_reason is None:
            self._ep_done_reason = str(done_reason)
        cfg = self.reward_cfg or {}
        dt = float(cfg.get("dt", 0.1))
        dt = max(1e-6, dt)
        return self._finalize_episode_reward(dt=dt, done_reason=str(self._ep_done_reason or done_reason))


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
