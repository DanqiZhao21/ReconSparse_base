from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Callable

import numpy as np

from framework.utils.tracker_execution import TrackerExecutionResult, build_execution_result


DEFAULT_HUGSIM_REPO = "/root/clone/HUGSIM"
DEFAULT_WHEELBASE = 2.7
_HUGSIM_SOLVER_CACHE: dict[tuple[float, float], Any] = {}


def plan_local_xyyaw_to_hugsim_plan_xy(plan_local_xyyaw: np.ndarray) -> np.ndarray:
    plan = np.asarray(plan_local_xyyaw, dtype=np.float64)
    if plan.ndim != 2 or plan.shape[1] < 2:
        raise ValueError(f"Expected (N, >=2) local plan, got {plan.shape}")
    out = np.zeros((plan.shape[0], 2), dtype=np.float64)
    out[:, 0] = -plan[:, 1]
    out[:, 1] = plan[:, 0]
    return out


def controller_info_from_status(
    *,
    velocity_xy: np.ndarray,
    steering_angle: float,
    dt: float,
    wheelbase: float,
) -> dict[str, float]:
    vel = np.asarray(velocity_xy, dtype=np.float64).reshape(-1)
    speed = float(np.linalg.norm(vel[:2], ord=2)) if vel.size > 0 else 0.0
    return {
        "ego_velo": float(speed),
        "ego_steer": float(steering_angle),
        "dt": float(dt),
        "wheelbase": float(wheelbase),
    }


def resolve_wheelbase(
    *,
    sparse_repo_path: str | None,
    explicit_wheelbase: float | None,
    load_wheelbase_fn: Callable[[str], float | None] | None = None,
) -> float:
    if explicit_wheelbase is not None:
        return float(explicit_wheelbase)
    if callable(load_wheelbase_fn) and sparse_repo_path:
        try:
            wheelbase = load_wheelbase_fn(str(sparse_repo_path))
            if wheelbase is not None:
                return float(wheelbase)
        except Exception:
            pass
    return float(DEFAULT_WHEELBASE)


def _resample_hugsim_plan_traj(plan_traj: np.ndarray, *, source_dt: float, target_dt: float) -> np.ndarray:
    plan = np.asarray(plan_traj, dtype=np.float64)
    if plan.ndim != 2 or plan.shape[1] != 2:
        raise ValueError(f"Expected HUGSIM plan with shape (N, 2), got {plan.shape}")
    if plan.shape[0] <= 0:
        return plan.copy()
    src = float(source_dt)
    dst = float(target_dt)
    if np.isclose(src, dst):
        return plan.copy()
    horizon = float(plan.shape[0]) * src
    source_times = np.arange(plan.shape[0] + 1, dtype=np.float64) * src
    target_count = int(np.round(horizon / dst))
    target_times = np.arange(1, target_count + 1, dtype=np.float64) * dst
    points = np.vstack((np.zeros((1, 2), dtype=np.float64), plan))
    x = np.interp(target_times, source_times, points[:, 0])
    y = np.interp(target_times, source_times, points[:, 1])
    return np.stack((x, y), axis=1)


def _build_hugsim_reference_trajectory(plan_traj: np.ndarray, *, plan_dt: float, control_dt: float) -> np.ndarray:
    plan = _resample_hugsim_plan_traj(np.asarray(plan_traj, dtype=np.float64), source_dt=float(plan_dt), target_dt=float(control_dt))
    plan_traj_stats = np.zeros((plan.shape[0] + 1, 5), dtype=np.float64)
    ego_plan_traj = np.asarray(plan, dtype=np.float64)[:, [1, 0]]
    ego_plan_traj[:, 1] *= -1.0
    plan_traj_stats[1:, :2] = ego_plan_traj
    prev_x = 0.0
    prev_y = 0.0
    for i, (x, y) in enumerate(ego_plan_traj):
        rot = np.arctan2(y - prev_y, x - prev_x)
        rot = np.where(rot > np.pi / 2.0, rot - np.pi, rot)
        rot = np.where(rot < -np.pi / 2.0, rot + np.pi, rot)
        plan_traj_stats[i + 1, 2] = rot
        prev_x = float(x)
        prev_y = float(y)
    return plan_traj_stats


def _solve_hugsim_control_sequence(
    plan_traj: np.ndarray,
    info: dict[str, float],
    *,
    wheelbase: float | None,
    plan_dt: float,
    control_dt: float | None,
    build_solver_fn: Any,
) -> tuple[np.ndarray, np.ndarray]:
    dt = float(info.get("dt", plan_dt)) if control_dt is None else float(control_dt)
    reference = _build_hugsim_reference_trajectory(plan_traj, plan_dt=float(plan_dt), control_dt=float(dt))
    current_state = np.array([0.0, 0.0, 0.0, float(info["ego_velo"]), float(info["ego_steer"])], dtype=np.float64)
    wb = float(DEFAULT_WHEELBASE if wheelbase is None else wheelbase)
    cache_key = (round(wb, 6), round(float(dt), 6))
    solver = _HUGSIM_SOLVER_CACHE.get(cache_key)
    if solver is None:
        solver = build_solver_fn(wb, float(dt))
        _HUGSIM_SOLVER_CACHE[cache_key] = solver
    solutions = solver.solve(current_state, reference)
    final = solutions[-1]
    return np.asarray(final.state_trajectory, dtype=np.float64), np.asarray(final.input_trajectory, dtype=np.float64)


def build_hugsim_execution_from_solution(
    *,
    prev_pose: np.ndarray,
    state_trajectory: np.ndarray,
    input_trajectory: np.ndarray,
) -> TrackerExecutionResult:
    states = np.asarray(state_trajectory, dtype=np.float64)
    inputs = np.asarray(input_trajectory, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 5:
        raise ValueError(f"Invalid HUGSIM state trajectory shape: {states.shape}")
    if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] < 2:
        raise ValueError(f"Invalid HUGSIM input trajectory shape: {inputs.shape}")
    rollout = np.asarray(states[1:, :3], dtype=np.float64)
    first = np.asarray(rollout[0], dtype=np.float64)
    return build_execution_result(
        prev_pose=np.asarray(prev_pose, dtype=np.float64),
        tracked_rollout_local_xyyaw=np.asarray(rollout, dtype=np.float64),
        tracked_first_local_xyyaw=np.asarray(first, dtype=np.float64),
        velocity_xy=np.asarray([float(states[1, 3]), 0.0], dtype=np.float32),
        acceleration_xy=np.asarray([float(inputs[0, 0]), 0.0], dtype=np.float32),
        steering_angle=float(states[1, 4]),
        steering_rate=float(inputs[0, 1]),
        command_state=np.asarray(inputs[0, :2], dtype=np.float64),
    )


def solve_hugsim_execution(
    *,
    prev_pose: np.ndarray,
    plan_local_xyyaw: np.ndarray,
    velocity_xy: np.ndarray,
    steering_angle: float,
    dt: float,
    wheelbase: float,
    solve_sequence_fn: Callable[..., tuple[np.ndarray, np.ndarray]] = _solve_hugsim_control_sequence,
    build_solver_fn: Any,
) -> tuple[TrackerExecutionResult, dict[str, float]]:
    controller_info = controller_info_from_status(
        velocity_xy=np.asarray(velocity_xy, dtype=np.float64),
        steering_angle=float(steering_angle),
        dt=float(dt),
        wheelbase=float(wheelbase),
    )
    hugsim_plan_xy = plan_local_xyyaw_to_hugsim_plan_xy(np.asarray(plan_local_xyyaw, dtype=np.float64))
    state_trajectory, input_trajectory = solve_sequence_fn(
        hugsim_plan_xy,
        controller_info,
        wheelbase=float(wheelbase),
        plan_dt=float(dt),
        control_dt=float(dt),
        build_solver_fn=build_solver_fn,
    )
    execution = build_hugsim_execution_from_solution(
        prev_pose=np.asarray(prev_pose, dtype=np.float64),
        state_trajectory=np.asarray(state_trajectory, dtype=np.float64),
        input_trajectory=np.asarray(input_trajectory, dtype=np.float64),
    )
    return execution, {
        "acc": float(input_trajectory[0, 0]),
        "steer_rate": float(input_trajectory[0, 1]),
        "speed_next": float(state_trajectory[1, 3]),
        "steer_next": float(state_trajectory[1, 4]),
    }


def load_hugsim_runtime(repo_path: str = DEFAULT_HUGSIM_REPO) -> tuple[Any, Any, Any]:
    resolved = os.path.abspath(str(repo_path))
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    try:
        import sim.ilqr.lqr as hugsim_lqr  # type: ignore
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "Missing HUGSIM runtime dependency for controller rollout. "
            f"Import failed on module: {missing}. Check HUGSIM env and repo path."
        ) from e
    vehicle_module_path = os.path.join(resolved, "sim", "utils", "sparsedrive_vehicle.py")
    spec = importlib.util.spec_from_file_location("hugsim_sparsedrive_vehicle", vehicle_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load HUGSIM vehicle helper: {vehicle_module_path}")
    vehicle_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vehicle_module)
    return _solve_hugsim_control_sequence, hugsim_lqr._build_solver, vehicle_module.load_sparsedrive_wheelbase
