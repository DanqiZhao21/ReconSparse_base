from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


CRAFT_CORRECTIVE_DEFAULTS: Dict[str, float] = {
    "cost_off_road": 0.5,
    "cost_emergency_lane": 0.2,
    "cost_off_global_route": 0.5,
    "cost_red_light": 2.0,
    "cost_stop_sign": 2.0,
    "cost_collision": 5.0,
}


CRAFT_CARL_FORWARD_SIM_DEFAULTS: Dict[str, float] = {
    "dp_min": 0.0,
    "dp_max": 1.2,
    "w_prog": 8.0,
    "w_g": 3.0,
    "w_c": 0.8,
    "w_h": 2.0,
    "efficiency_floor": 0.0,
    "ddev_clip": 0.10,
    "k_g": 0.4,
    "k_c": 0.2,
    "k_h": 0.4,
    "correction_apply_thresh_global": 0.08,
    "correction_apply_thresh_center": 0.08,
    "corr_clip": 0.5,
    "cost_off_road": 1.5,
    "cost_opposite_lane": 0.1,
    "cost_off_global_route": 1.5,
    "cost_emergency_lane": 1.0,
    "term_collision": 40.0,
    "term_route_dev": 40.0,
    "term_red_light": 40.0,
    "term_stop_sign": 40.0,
}


@dataclass(frozen=True)
class CraftCarlRewardResult:
    reward_steps: np.ndarray
    candidate_scores: np.ndarray
    terms: Dict[str, np.ndarray]


def _merged_params(defaults: Dict[str, float], params: Dict[str, Any] | None) -> Dict[str, float]:
    out = dict(defaults)
    for key, value in (params or {}).items():
        if key in out:
            out[key] = float(value)
    return out


def compute_corrective_reward_scalar( #只扣安全/规则成本。
    *,
    params: Dict[str, Any] | None = None,
    off_road: bool | float,
    emergency_lane: bool | float,
    off_global_route: bool | float,
    run_red_light: bool | float,
    run_stop_sign: bool | float,
    collision: bool | float,
) -> tuple[float, Dict[str, float]]:
    p = _merged_params(CRAFT_CORRECTIVE_DEFAULTS, params)
    off_road_cost = p["cost_off_road"] * float(off_road)
    emergency_lane_cost = p["cost_emergency_lane"] * float(emergency_lane)
    off_global_route_cost = p["cost_off_global_route"] * float(off_global_route)
    red_light_cost = p["cost_red_light"] * float(run_red_light)
    stop_sign_cost = p["cost_stop_sign"] * float(run_stop_sign)
    collision_cost = p["cost_collision"] * float(collision)
    total_cost = (
        off_road_cost
        + emergency_lane_cost
        + off_global_route_cost
        + red_light_cost
        + stop_sign_cost
        + collision_cost
    )
    info = {
        "craft_corrective_cost_off_road": float(off_road_cost),
        "craft_corrective_cost_emergency_lane": float(emergency_lane_cost),
        "craft_corrective_cost_off_global_route": float(off_global_route_cost),
        "craft_corrective_cost_red_light": float(red_light_cost),
        "craft_corrective_cost_stop_sign": float(stop_sign_cost),
        "craft_corrective_cost_collision": float(collision_cost),
        "craft_corrective_total_cost": float(total_cost),
    }
    return -float(total_cost), info


def _as_step_array(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (candidates,horizon), got {tuple(arr.shape)}")
    return arr


def _as_candidate_bool(name: str, value: np.ndarray, num_candidates: int) -> np.ndarray:
    arr = np.asarray(value, dtype=bool).reshape(-1)
    if int(arr.shape[0]) != int(num_candidates):
        raise ValueError(f"{name} must have shape (candidates,), got {tuple(arr.shape)}")
    return arr

#候选轨迹级 dense CaRL reward，包含 progress、efficiency、correction、
# off-road、collision、red light、stop sign 等。
def compute_carl_reward_numpy(
    *,
    params: Dict[str, Any] | None = None,
    delta_progress: np.ndarray,
    global_dev_ratio: np.ndarray,
    center_dev_ratio: np.ndarray,
    heading_dev_ratio: np.ndarray,
    delta_global_dev_ratio: np.ndarray,
    delta_center_dev_ratio: np.ndarray,
    delta_heading_dev_ratio: np.ndarray,
    off_road: np.ndarray,
    opposite_lane: np.ndarray,
    emergency_lane: np.ndarray,
    off_global_route: np.ndarray,
    collision: np.ndarray,
    red_light_dev: np.ndarray,
    stop_sign_dev: np.ndarray,
    route_deviation: np.ndarray | None = None,
) -> CraftCarlRewardResult:
    p = _merged_params(CRAFT_CARL_FORWARD_SIM_DEFAULTS, params)
    dp = _as_step_array("delta_progress", delta_progress)
    shape = tuple(dp.shape)
    arrays = {
        "global_dev_ratio": _as_step_array("global_dev_ratio", global_dev_ratio),
        "center_dev_ratio": _as_step_array("center_dev_ratio", center_dev_ratio),
        "heading_dev_ratio": _as_step_array("heading_dev_ratio", heading_dev_ratio),
        "delta_global_dev_ratio": _as_step_array("delta_global_dev_ratio", delta_global_dev_ratio),
        "delta_center_dev_ratio": _as_step_array("delta_center_dev_ratio", delta_center_dev_ratio),
        "delta_heading_dev_ratio": _as_step_array("delta_heading_dev_ratio", delta_heading_dev_ratio),
        "off_road": _as_step_array("off_road", off_road),
        "opposite_lane": _as_step_array("opposite_lane", opposite_lane),
        "emergency_lane": _as_step_array("emergency_lane", emergency_lane),
        "off_global_route": _as_step_array("off_global_route", off_global_route),
        "collision": _as_step_array("collision", collision),
    }
    for name, arr in arrays.items():
        if tuple(arr.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(arr.shape)}")

    num_candidates, horizon = shape
    red_light = _as_candidate_bool("red_light_dev", red_light_dev, num_candidates)
    stop_sign = _as_candidate_bool("stop_sign_dev", stop_sign_dev, num_candidates)
    route_dev = (
        np.zeros((num_candidates,), dtype=bool)
        if route_deviation is None
        else _as_candidate_bool("route_deviation", route_deviation, num_candidates)
    )

    clipped_dp = np.clip(dp, p["dp_min"], p["dp_max"]).astype(np.float32, copy=False)
    dp_norm = (clipped_dp / max(1.0e-6, p["dp_max"])).astype(np.float32, copy=False)
    progress_reward = (p["w_prog"] * dp_norm).astype(np.float32, copy=False)

    eff_g = np.exp(-p["w_g"] * arrays["global_dev_ratio"])
    eff_c = np.exp(-p["w_c"] * arrays["center_dev_ratio"])
    eff_h = np.exp(-p["w_h"] * arrays["heading_dev_ratio"])
    efficiency = np.maximum(eff_g * eff_c * eff_h, p["efficiency_floor"]).astype(np.float32, copy=False)
    effective_progress = (progress_reward * efficiency).astype(np.float32, copy=False)

    ddev_clip = max(0.0, p["ddev_clip"])
    delta_global = np.clip(arrays["delta_global_dev_ratio"], -ddev_clip, ddev_clip)
    delta_center = np.clip(arrays["delta_center_dev_ratio"], -ddev_clip, ddev_clip)
    delta_heading = np.clip(arrays["delta_heading_dev_ratio"], -ddev_clip, ddev_clip)
    correction = np.zeros(shape, dtype=np.float32)
    correction += (
        arrays["global_dev_ratio"] > p["correction_apply_thresh_global"]
    ).astype(np.float32) * (p["k_g"] * (-delta_global))
    correction += (
        arrays["center_dev_ratio"] > p["correction_apply_thresh_center"]
    ).astype(np.float32) * (p["k_c"] * (-delta_center))
    correction += (
        arrays["heading_dev_ratio"] > p["correction_apply_thresh_center"]
    ).astype(np.float32) * (p["k_h"] * (-delta_heading))
    correction = np.clip(correction, -p["corr_clip"], p["corr_clip"]).astype(np.float32, copy=False)
    correction = (correction * (0.5 + 0.5 * dp_norm)).astype(np.float32, copy=False)

    route_safety_cost = (
        p["cost_off_road"] * arrays["off_road"]
        + p["cost_opposite_lane"] * arrays["opposite_lane"]
        + p["cost_off_global_route"] * arrays["off_global_route"]
        + p["cost_emergency_lane"] * arrays["emergency_lane"]
    ).astype(np.float32, copy=False)
    collision_cost = (p["term_collision"] * arrays["collision"]).astype(np.float32, copy=False)

    reward_steps = (effective_progress + correction - route_safety_cost - collision_cost).astype(np.float32, copy=False)
    if horizon > 0:
        reward_steps[:, 0] = reward_steps[:, 0] - (
            p["term_red_light"] * red_light.astype(np.float32)
            + p["term_stop_sign"] * stop_sign.astype(np.float32)
            + p["term_route_dev"] * route_dev.astype(np.float32)
        )

    terms = {
        "clipped_delta_progress": clipped_dp,
        "dp_norm": dp_norm,
        "progress_reward": progress_reward,
        "efficiency": efficiency,
        "effective_progress": effective_progress,
        "correction_reward": correction,
        "route_safety_cost": route_safety_cost,
        "collision_cost": collision_cost,
        "reward_steps": reward_steps,
    }
    return CraftCarlRewardResult(
        reward_steps=reward_steps,
        candidate_scores=reward_steps.sum(axis=1).astype(np.float32, copy=False),
        terms=terms,
    )


__all__ = [
    "CRAFT_CARL_FORWARD_SIM_DEFAULTS",
    "CRAFT_CORRECTIVE_DEFAULTS",
    "CraftCarlRewardResult",
    "compute_carl_reward_numpy",
    "compute_corrective_reward_scalar",
]
