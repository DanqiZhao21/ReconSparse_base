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

    def reset(self, scene: int | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        # `ReconSimulator.reset(seed)` internally uses `update(seed)` → seed acts as scene id.
        seed = scene if scene is not None else self.env.scene
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action: Tuple[int, int] | Tuple[int, int, int]):
        # Normalize action to (ax, ay, flag)
        if len(action) == 2:
            ax, ay = int(action[0]), int(action[1])
            flag = 0
        else:
            ax, ay, flag = int(action[0]), int(action[1]), int(action[2])

        obs, terminated, truncated, info = self.env.step((ax, ay, flag))
        reward = self._compute_reward()
        return obs, reward, terminated, truncated, info

    # -------------------- Reward -------------------- #
    def _compute_reward(self) -> float:
        """
        Four-component deviation reward using nearest expert reference:
        - Position deviation: distance in meters (x,z plane) vs threshold `dmax`
        - Heading deviation: yaw error vs threshold `psi_max_deg`
        - Static and dynamic collision penalties (placeholders: 0 without detectors)

        Final reward is negative weighted sum of exceeded deviations.
        """
        cfg = self.reward_cfg or {}
        dmax = float(cfg.get("dmax", 2.0))
        psi_max_deg = float(cfg.get("psi_max_deg", 30.0))
        w_pos = float(cfg.get("w_pos", 2.0))
        w_heading = float(cfg.get("w_heading", 1.0))
        w_static = float(cfg.get("w_static", 5.0))
        w_dynamic = float(cfg.get("w_dynamic", 5.0))

        # --- Position deviation vs expert ---
        # Current ego position (x,z)
        ego_xz = self.env.start_ego[:3, 3][[0, 2]]
        # Precomputed expert points in x,z
        expert_xz_list = self.env.expert_pair  # list of shape (..., 2)
        if len(expert_xz_list) == 0:
            pos_dev = 0.0
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

        # --- Collision placeholders ---
        static_collision = 0.0
        dynamic_collision = 0.0

        # Penalize only when exceeding thresholds
        pos_pen = w_pos * max(0.0, pos_dev - dmax)
        heading_pen = w_heading * max(0.0, math.degrees(yaw_err) - psi_max_deg)
        static_pen = w_static * static_collision
        dynamic_pen = w_dynamic * dynamic_collision

        reward = -(pos_pen + heading_pen + static_pen + dynamic_pen)
        return float(reward)


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
