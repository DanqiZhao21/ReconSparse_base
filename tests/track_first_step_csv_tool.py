from __future__ import annotations

import argparse
import dataclasses
import math
import os
import re
import sys
import types
from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _infer_scene_id_from_csv_path(path: str) -> int:
    text = os.path.abspath(str(path))
    match = re.search(r"scene(\d+)", text)
    if match is None:
        raise ValueError(f"Failed to infer scene id from path: {path}")
    return int(match.group(1))


def _pose_from_xyyaw(x: float, y: float, yaw: float) -> np.ndarray:
    c = float(math.cos(float(yaw)))
    s = float(math.sin(float(yaw)))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pose[0, 3] = float(x)
    pose[1, 3] = float(y)
    return pose


def _xyyaw_from_pose(pose: np.ndarray) -> np.ndarray:
    yaw = float(math.atan2(float(pose[1, 0]), float(pose[0, 0])))
    return np.asarray([float(pose[0, 3]), float(pose[1, 3]), float(yaw)], dtype=np.float64)


def _load_recon_simulator_class():
    if "gymnasium" not in sys.modules:
        try:
            import gymnasium  # type: ignore  # noqa: F401
        except Exception:
            gym_stub = types.ModuleType("gymnasium")

            class _Env:
                pass

            class _Space:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    del args, kwargs

            spaces_stub = types.SimpleNamespace(
                Box=_Space,
                Dict=_Space,
                MultiDiscrete=_Space,
            )
            gym_stub.Env = _Env
            gym_stub.spaces = spaces_stub
            sys.modules["gymnasium"] = gym_stub
    if "nuplan" not in sys.modules:
        try:
            import nuplan  # type: ignore  # noqa: F401
        except Exception:
            nuplan_mod = types.ModuleType("nuplan")
            common_mod = types.ModuleType("nuplan.common")
            actor_state_mod = types.ModuleType("nuplan.common.actor_state")
            planning_mod = types.ModuleType("nuplan.planning")
            sim_mod = types.ModuleType("nuplan.planning.simulation")
            time_ctrl_mod = types.ModuleType("nuplan.planning.simulation.simulation_time_controller")

            @dataclasses.dataclass(frozen=True)
            class TimeDuration:
                time_s: float

                @classmethod
                def from_s(cls, seconds: float) -> "TimeDuration":
                    return cls(float(seconds))

            @dataclasses.dataclass(frozen=True)
            class TimePoint:
                time_s: float

                def __add__(self, other: TimeDuration) -> "TimePoint":
                    return TimePoint(float(self.time_s) + float(other.time_s))

                def __sub__(self, other: "TimePoint") -> TimeDuration:
                    return TimeDuration(float(self.time_s) - float(other.time_s))

            @dataclasses.dataclass(frozen=True)
            class VehicleParameters:
                wheel_base: float = 3.089

            def get_pacifica_parameters() -> VehicleParameters:
                return VehicleParameters()

            def principal_value(angle: Any) -> Any:
                return np.arctan2(np.sin(angle), np.cos(angle))

            @dataclasses.dataclass(frozen=True)
            class StateSE2:
                x: float
                y: float
                heading: float

            @dataclasses.dataclass(frozen=True)
            class SimulationIteration:
                time_point: TimePoint
                index: int

            ego_state_mod = types.ModuleType("nuplan.common.actor_state.ego_state")
            ego_state_mod.EgoState = object

            state_repr_mod = types.ModuleType("nuplan.common.actor_state.state_representation")
            state_repr_mod.TimePoint = TimePoint
            state_repr_mod.TimeDuration = TimeDuration
            state_repr_mod.StateSE2 = StateSE2

            vehicle_params_mod = types.ModuleType("nuplan.common.actor_state.vehicle_parameters")
            vehicle_params_mod.VehicleParameters = VehicleParameters
            vehicle_params_mod.get_pacifica_parameters = get_pacifica_parameters

            geometry_mod = types.ModuleType("nuplan.common.geometry")
            compute_mod = types.ModuleType("nuplan.common.geometry.compute")
            compute_mod.principal_value = principal_value

            sim_iteration_mod = types.ModuleType(
                "nuplan.planning.simulation.simulation_time_controller.simulation_iteration"
            )
            sim_iteration_mod.SimulationIteration = SimulationIteration

            sys.modules["nuplan"] = nuplan_mod
            sys.modules["nuplan.common"] = common_mod
            sys.modules["nuplan.common.actor_state"] = actor_state_mod
            sys.modules["nuplan.common.actor_state.ego_state"] = ego_state_mod
            sys.modules["nuplan.common.actor_state.state_representation"] = state_repr_mod
            sys.modules["nuplan.common.actor_state.vehicle_parameters"] = vehicle_params_mod
            sys.modules["nuplan.common.geometry"] = geometry_mod
            sys.modules["nuplan.common.geometry.compute"] = compute_mod
            sys.modules["nuplan.planning"] = planning_mod
            sys.modules["nuplan.planning.simulation"] = sim_mod
            sys.modules["nuplan.planning.simulation.simulation_time_controller"] = time_ctrl_mod
            sys.modules[
                "nuplan.planning.simulation.simulation_time_controller.simulation_iteration"
            ] = sim_iteration_mod

            nuplan_mod.common = common_mod
            nuplan_mod.planning = planning_mod
            common_mod.actor_state = actor_state_mod
            common_mod.geometry = geometry_mod
            actor_state_mod.ego_state = ego_state_mod
            actor_state_mod.state_representation = state_repr_mod
            actor_state_mod.vehicle_parameters = vehicle_params_mod
            geometry_mod.compute = compute_mod
            planning_mod.simulation = sim_mod
            sim_mod.simulation_time_controller = time_ctrl_mod
            time_ctrl_mod.simulation_iteration = sim_iteration_mod
    navsim_root = "/root/clone/ReconDreamer-RL/egoADs/DiffusionDriveV2"
    if os.path.isdir(navsim_root) and navsim_root not in sys.path:
        sys.path.insert(0, navsim_root)
    if "framework.env_wrapper" not in sys.modules:
        env_wrapper_stub = types.ModuleType("framework.env_wrapper")
        env_wrapper_stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules["framework.env_wrapper"] = env_wrapper_stub
    if "framework.env_wrapper.tool" not in sys.modules:
        tool_stub = types.ModuleType("framework.env_wrapper.tool")

        def _unsupported(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("framework.env_wrapper.tool stub should not be used in tracking tool")

        tool_stub.get_splat = _unsupported
        tool_stub.get_sky_view = _unsupported
        tool_stub.move_to_device = _unsupported
        tool_stub.slerp = _unsupported
        sys.modules["framework.env_wrapper.tool"] = tool_stub
        setattr(sys.modules["framework.env_wrapper"], "tool", tool_stub)
    from reconsimulator.envs.nus import ReconSimulator

    return ReconSimulator


def _build_absolute_poses_from_csv(df: pd.DataFrame) -> np.ndarray:
    required = {"x", "y", "yaw_xy_rad_signed"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    poses = [
        _pose_from_xyyaw(float(row["x"]), float(row["y"]), float(row["yaw_xy_rad_signed"]))
        for _, row in df.iterrows()
    ]
    return np.stack(poses, axis=0)


def _build_local_plan_from_absolute_poses(
    *,
    absolute_poses: np.ndarray,
    start_index: int,
    step_stride: int,
    horizon_points: int,
) -> np.ndarray:
    if absolute_poses.ndim != 3 or absolute_poses.shape[1:] != (4, 4):
        raise ValueError("absolute_poses must have shape (N, 4, 4)")
    if int(start_index) < 0 or int(start_index) >= int(absolute_poses.shape[0]):
        raise IndexError(start_index)

    count = int(absolute_poses.shape[0])
    stride = max(1, int(step_stride))
    horizon = max(1, int(horizon_points))
    base_inv = np.linalg.inv(np.asarray(absolute_poses[int(start_index)], dtype=np.float64))
    out = np.zeros((horizon, 3), dtype=np.float64)

    for i in range(horizon):
        target_index = min(count - 1, int(start_index) + stride * (i + 1))
        rel = base_inv @ np.asarray(absolute_poses[target_index], dtype=np.float64)
        out[i] = _xyyaw_from_pose(rel)

    return out


def _make_tracking_probe(*, scene: int) -> Any:
    ReconSimulator = _load_recon_simulator_class()
    env = ReconSimulator.__new__(ReconSimulator)
    env.scene = int(scene)
    env._status_prev_vel_xy = None
    env._status_vel_xy = np.zeros((2,), dtype=np.float32)
    env._status_acc_xy = np.zeros((2,), dtype=np.float32)
    env._status_cmd = np.zeros((4,), dtype=np.float32)
    env._tracked_first_step_xyyaw = np.zeros((3,), dtype=np.float64)
    env._external_plan_local_xyyaw = None
    env._nusc = None
    env._nusc_can_bus = None
    env._nusc_can_bus_cache = {}
    env._nusc_sample_by_token = {}
    env._nusc_scene_name_by_token = {}
    env._nusc_meta_loaded = False
    env._pdm_tracker = None
    env._pdm_motion_model = None
    env._pdm_state_index = None
    env._load_token_mappings()
    return env


@dataclass
class TrackingRow:
    frame: int
    src_index: int
    target_index: int
    desired_local_xyyaw: np.ndarray
    tracked_local_xyyaw: np.ndarray
    desired_abs_xyyaw: np.ndarray
    tracked_abs_xyyaw: np.ndarray
    dataset_vel_xy: np.ndarray
    dataset_acc_xy: np.ndarray
    tracked_vel_xy: np.ndarray
    tracked_acc_xy: np.ndarray
    xy_error_m: float
    yaw_error_deg: float


def _downsample_frame_indices(*, total_count: int, stride: int) -> list[int]:
    total = max(0, int(total_count))
    step = max(1, int(stride))
    return list(range(0, total, step))


def _valid_start_indices(
    *,
    total_count: int,
    step_stride: int,
    horizon_points: int,
    require_full_horizon: bool,
) -> list[int]:
    total = max(0, int(total_count))
    stride = max(1, int(step_stride))
    horizon = max(1, int(horizon_points))
    if total <= 1:
        return []
    if not bool(require_full_horizon):
        return list(range(max(0, total - 1)))

    max_start = total - 1 - stride * horizon
    if max_start < 0:
        return []
    return list(range(max_start + 1))


def _run_exact_tracking(
    *,
    csv_path: str,
    dt: float,
    csv_downsample_stride: int,
    step_stride: int,
    horizon_points: int,
    require_full_horizon: bool,
) -> tuple[list[TrackingRow], np.ndarray, pd.DataFrame]:
    scene = _infer_scene_id_from_csv_path(csv_path)
    env = _make_tracking_probe(scene=scene)
    if not bool(env._load_pdm_tracking_modules()):
        raise RuntimeError(
            "Exact PDM tracking modules are unavailable. "
            "The tool refuses to fall back because this run requested exact tracking."
        )
    raw_df = pd.read_csv(csv_path)
    downsample_indices = _downsample_frame_indices(total_count=len(raw_df), stride=csv_downsample_stride)
    if len(downsample_indices) == 0:
        raise RuntimeError("CSV downsampling produced no frames")
    df = raw_df.iloc[downsample_indices].reset_index(drop=True)
    absolute_poses = _build_absolute_poses_from_csv(df)

    rows: list[TrackingRow] = []
    ReconSimulator = _load_recon_simulator_class()
    start_indices = _valid_start_indices(
        total_count=len(df),
        step_stride=step_stride,
        horizon_points=horizon_points,
        require_full_horizon=require_full_horizon,
    )
    for src_index in start_indices:
        frame = int(df.iloc[src_index]["frame"])
        dataset_vel, dataset_acc, _cmd = env._status_from_dataset(frame)
        env._status_vel_xy = np.asarray(dataset_vel, dtype=np.float32)
        env._status_acc_xy = np.asarray(dataset_acc, dtype=np.float32)
        env._status_prev_vel_xy = np.asarray(dataset_vel, dtype=np.float32).copy()

        plan_local_xyyaw = _build_local_plan_from_absolute_poses(
            absolute_poses=absolute_poses,
            start_index=src_index,
            step_stride=step_stride,
            horizon_points=horizon_points,
        )
        tracked_vel, tracked_acc = env._track_first_step_vel_acc(
            prev_pose=np.asarray(absolute_poses[src_index], dtype=np.float64),
            plan_local_xyyaw=np.asarray(plan_local_xyyaw, dtype=np.float64),
            dt=float(dt),
        )

        tracked_local = np.asarray(env._tracked_first_step_xyyaw, dtype=np.float64)
        target_index = min(int(len(df)) - 1, int(src_index) + int(step_stride))
        base_pose = np.asarray(absolute_poses[src_index], dtype=np.float64)
        desired_abs_pose = base_pose @ ReconSimulator._pose_from_local_xyyaw(*plan_local_xyyaw[0])
        tracked_abs_pose = base_pose @ ReconSimulator._pose_from_local_xyyaw(*tracked_local)

        desired_abs = _xyyaw_from_pose(desired_abs_pose)
        tracked_abs = _xyyaw_from_pose(tracked_abs_pose)
        xy_error = float(np.linalg.norm(tracked_abs[:2] - desired_abs[:2]))
        yaw_error_deg = float(np.rad2deg(ReconSimulator._wrap_angle(float(tracked_abs[2] - desired_abs[2]))))

        rows.append(
            TrackingRow(
                frame=frame,
                src_index=int(src_index),
                target_index=int(target_index),
                desired_local_xyyaw=np.asarray(plan_local_xyyaw[0], dtype=np.float64),
                tracked_local_xyyaw=np.asarray(tracked_local, dtype=np.float64),
                desired_abs_xyyaw=np.asarray(desired_abs, dtype=np.float64),
                tracked_abs_xyyaw=np.asarray(tracked_abs, dtype=np.float64),
                dataset_vel_xy=np.asarray(dataset_vel, dtype=np.float32),
                dataset_acc_xy=np.asarray(dataset_acc, dtype=np.float32),
                tracked_vel_xy=np.asarray(tracked_vel, dtype=np.float32),
                tracked_acc_xy=np.asarray(tracked_acc, dtype=np.float32),
                xy_error_m=float(xy_error),
                yaw_error_deg=float(yaw_error_deg),
            )
        )

    return rows, absolute_poses, df


def _norm_rows(values: Iterable[np.ndarray]) -> np.ndarray:
    arr = np.stack([np.asarray(v, dtype=np.float64) for v in values], axis=0)
    return np.linalg.norm(arr, axis=1)


def _default_output_path(csv_path: str) -> str:
    directory = os.path.dirname(os.path.abspath(csv_path))
    return os.path.join(directory, "tracked_first_step_exact.svg")


def _default_output_paths(csv_path: str) -> dict[str, str]:
    directory = os.path.dirname(os.path.abspath(csv_path))
    base = os.path.join(directory, "tracked_first_step_exact")
    return {
        "absolute_xy": f"{base}_absolute_xy.svg",
        "absolute_xy_sampled18": f"{base}_absolute_xy_sampled18.svg",
        "local_first_step": f"{base}_local_first_step.svg",
        "tracking_error": f"{base}_tracking_error.svg",
        "velocity_accel": f"{base}_velocity_accel.svg",
    }


def _resolve_output_paths(csv_path: str, out: str | None) -> dict[str, str]:
    if out is None or str(out).strip() == "":
        return _default_output_paths(csv_path)

    out_abs = os.path.abspath(str(out))
    if out_abs.lower().endswith(".svg"):
        stem = out_abs[:-4]
        return {
            "absolute_xy": f"{stem}_absolute_xy.svg",
            "absolute_xy_sampled18": f"{stem}_absolute_xy_sampled18.svg",
            "local_first_step": f"{stem}_local_first_step.svg",
            "tracking_error": f"{stem}_tracking_error.svg",
            "velocity_accel": f"{stem}_velocity_accel.svg",
        }

    os.makedirs(out_abs, exist_ok=True)
    base = os.path.join(out_abs, "tracked_first_step_exact")
    return {
        "absolute_xy": f"{base}_absolute_xy.svg",
        "absolute_xy_sampled18": f"{base}_absolute_xy_sampled18.svg",
        "local_first_step": f"{base}_local_first_step.svg",
        "tracking_error": f"{base}_tracking_error.svg",
        "velocity_accel": f"{base}_velocity_accel.svg",
    }


def _compute_axis_limits_with_padding(
    values: np.ndarray,
    *,
    min_span: float,
    pad_ratio: float,
) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        half = float(min_span) * 0.5
        return -half, half

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = float(max(hi - lo, float(min_span)))
    center = 0.5 * (lo + hi)
    half = 0.5 * span
    lo = center - half
    hi = center + half
    pad = max(float(min_span) * 0.05, float(span) * float(pad_ratio))
    return lo - pad, hi + pad


def _uniform_sample_indices(*, total_count: int, sample_count: int) -> list[int]:
    total = max(0, int(total_count))
    want = max(0, int(sample_count))
    if total <= 0 or want <= 0:
        return []
    if want >= total:
        return list(range(total))
    raw = np.linspace(0, total - 1, num=want)
    idx = np.asarray(np.round(raw), dtype=np.int64)
    idx[0] = 0
    idx[-1] = total - 1
    uniq = []
    seen = set()
    for value in idx.tolist():
        value = max(0, min(total - 1, int(value)))
        if value not in seen:
            seen.add(value)
            uniq.append(value)
    if len(uniq) == want:
        return uniq
    # Fill any rare duplicate gaps with remaining evenly ordered indices.
    for value in range(total):
        if value not in seen:
            uniq.append(value)
            seen.add(value)
        if len(uniq) >= want:
            break
    uniq.sort()
    return uniq[:want]


def _render_svg(
    *,
    rows: list[TrackingRow],
    absolute_poses: np.ndarray,
    out_paths: dict[str, str],
    csv_path: str,
    dt: float,
    csv_downsample_stride: int,
    step_stride: int,
    horizon_points: int,
    require_full_horizon: bool,
    absolute_sample_count: int,
) -> None:
    if len(rows) == 0:
        raise RuntimeError("No tracking rows were generated")

    expert_xy = absolute_poses[:, :2, 3]
    desired_abs_xy = np.stack([row.desired_abs_xyyaw[:2] for row in rows], axis=0)
    tracked_abs_xy = np.stack([row.tracked_abs_xyyaw[:2] for row in rows], axis=0)
    desired_local_xy = np.stack([row.desired_local_xyyaw[:2] for row in rows], axis=0)
    tracked_local_xy = np.stack([row.tracked_local_xyyaw[:2] for row in rows], axis=0)
    frames = np.asarray([row.frame for row in rows], dtype=np.int32)
    xy_errors = np.asarray([row.xy_error_m for row in rows], dtype=np.float64)
    yaw_errors = np.asarray([row.yaw_error_deg for row in rows], dtype=np.float64)
    dataset_speed = _norm_rows(row.dataset_vel_xy for row in rows)
    tracked_speed = _norm_rows(row.tracked_vel_xy for row in rows)
    dataset_acc = _norm_rows(row.dataset_acc_xy for row in rows)
    tracked_acc = _norm_rows(row.tracked_acc_xy for row in rows)

    mean_xy = float(np.mean(xy_errors))
    max_xy = float(np.max(xy_errors))
    mean_yaw = float(np.mean(np.abs(yaw_errors)))
    title_suffix = (
        f"csv={os.path.basename(csv_path)}  dt={dt:.2f}s  csv_stride={csv_downsample_stride}  "
        f"step_stride={step_stride}  horizon={horizon_points}  strict_full_horizon={bool(require_full_horizon)}  "
        f"mean_xy={mean_xy:.3f}m  max_xy={max_xy:.3f}m  mean|yaw|={mean_yaw:.3f}deg"
    )
    sampled_indices = _uniform_sample_indices(total_count=len(rows), sample_count=absolute_sample_count)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(expert_xy[:, 0], expert_xy[:, 1], color="#9aa0a6", linewidth=1.5, label="expert absolute path")
    ax.scatter(
        desired_abs_xy[:, 0],
        desired_abs_xy[:, 1],
        s=28,
        color="#1f77b4",
        label="desired next step",
    )
    ax.scatter(
        tracked_abs_xy[:, 0],
        tracked_abs_xy[:, 1],
        s=28,
        color="#d62728",
        label="tracked next step",
    )
    for idx in range(len(rows)):
        ax.plot(
            [desired_abs_xy[idx, 0], tracked_abs_xy[idx, 0]],
            [desired_abs_xy[idx, 1], tracked_abs_xy[idx, 1]],
            color="#ff9896",
            linewidth=1.2,
            alpha=0.7,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Absolute XY: Expert vs Tracked First Step")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Absolute XY: Expert vs Tracked First Step (All Valid Starts)\n{title_suffix}", fontsize=11)
    os.makedirs(os.path.dirname(os.path.abspath(out_paths["absolute_xy"])), exist_ok=True)
    fig.savefig(out_paths["absolute_xy"], format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(expert_xy[:, 0], expert_xy[:, 1], color="#9aa0a6", linewidth=1.5, label="expert absolute path")
    ax.scatter(
        desired_abs_xy[sampled_indices, 0],
        desired_abs_xy[sampled_indices, 1],
        s=28,
        color="#1f77b4",
        label="desired next step",
    )
    ax.scatter(
        tracked_abs_xy[sampled_indices, 0],
        tracked_abs_xy[sampled_indices, 1],
        s=28,
        color="#d62728",
        label="tracked next step",
    )
    for idx in sampled_indices:
        ax.plot(
            [desired_abs_xy[idx, 0], tracked_abs_xy[idx, 0]],
            [desired_abs_xy[idx, 1], tracked_abs_xy[idx, 1]],
            color="#ff9896",
            linewidth=1.2,
            alpha=0.7,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Absolute XY: Expert vs Tracked First Step")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Absolute XY: Expert vs Tracked First Step (18 Uniform Samples)\n{title_suffix}", fontsize=11)
    fig.savefig(out_paths["absolute_xy_sampled18"], format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 12), constrained_layout=True)
    ax.scatter(
        desired_local_xy[:, 0],
        desired_local_xy[:, 1],
        s=28,
        color="#1f77b4",
        label="desired local first step",
    )
    ax.scatter(
        tracked_local_xy[:, 0],
        tracked_local_xy[:, 1],
        s=28,
        color="#d62728",
        label="tracked local first step",
    )
    for idx in range(len(rows)):
        ax.plot(
            [desired_local_xy[idx, 0], tracked_local_xy[idx, 0]],
            [desired_local_xy[idx, 1], tracked_local_xy[idx, 1]],
            color="#ff9896",
            linewidth=1.2,
            alpha=0.7,
        )
    ax.set_aspect("equal", adjustable="box")
    all_local_y = np.concatenate([desired_local_xy[:, 1], tracked_local_xy[:, 1]], axis=0)
    y_lo, y_hi = _compute_axis_limits_with_padding(
        all_local_y,
        min_span=0.3,
        pad_ratio=0.15,
    )
    ax.set_ylim(y_lo, y_hi)
    ax.set_title("Local First Step Comparison")
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Local First Step Comparison (All Valid Starts)\n{title_suffix}", fontsize=11)
    fig.savefig(out_paths["local_first_step"], format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(frames, xy_errors, color="#d62728", linewidth=1.5, label="xy error (m)")
    ax.plot(frames, np.abs(yaw_errors), color="#9467bd", linewidth=1.2, label="|yaw error| (deg)")
    ax.set_title("Tracking Error by Frame")
    ax.set_xlabel("frame")
    ax.set_ylabel("error")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Tracking Error by Frame\n{title_suffix}", fontsize=11)
    fig.savefig(out_paths["tracking_error"], format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(frames, dataset_speed, color="#1f77b4", linewidth=1.2, label="dataset speed")
    ax.plot(frames, tracked_speed, color="#d62728", linewidth=1.2, label="tracked speed")
    ax.plot(frames, dataset_acc, color="#17becf", linewidth=1.0, linestyle="--", label="dataset acc norm")
    ax.plot(frames, tracked_acc, color="#ff7f0e", linewidth=1.0, linestyle="--", label="tracked acc norm")
    ax.set_title("Velocity / Acceleration")
    ax.set_xlabel("frame")
    ax.set_ylabel("norm")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Velocity / Acceleration\n{title_suffix}", fontsize=11)
    fig.savefig(out_paths["velocity_accel"], format="svg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay exact _track_first_step_vel_acc on expert CSV and export SVG")
    parser.add_argument("--csv", required=True, help="Path to expert_ego_local_frame.csv")
    parser.add_argument("--out", default=None, help="Output SVG path")
    parser.add_argument("--dt", type=float, default=0.5, help="Tracking dt in seconds for each tracker step")
    parser.add_argument(
        "--csv-downsample-stride",
        type=int,
        default=5,
        help="Keep every Nth CSV row before tracking; 5 turns a 10Hz CSV into a 0.5s view",
    )
    parser.add_argument("--step-stride", type=int, default=1, help="Future-point stride on the downsampled sequence")
    parser.add_argument("--horizon-points", type=int, default=8, help="Number of future points passed into tracker")
    parser.add_argument(
        "--allow-tail-pad",
        action="store_true",
        help="Allow tail padding instead of requiring a full future horizon for each start point",
    )
    parser.add_argument(
        "--absolute-sample-count",
        type=int,
        default=18,
        help="Uniform sample count for the extra absolute-XY sampled SVG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_paths = _resolve_output_paths(str(args.csv), args.out)
    rows, absolute_poses, df = _run_exact_tracking(
        csv_path=str(args.csv),
        dt=float(args.dt),
        csv_downsample_stride=int(args.csv_downsample_stride),
        step_stride=int(args.step_stride),
        horizon_points=int(args.horizon_points),
        require_full_horizon=not bool(args.allow_tail_pad),
    )
    _render_svg(
        rows=rows,
        absolute_poses=absolute_poses,
        out_paths=out_paths,
        csv_path=str(args.csv),
        dt=float(args.dt),
        csv_downsample_stride=int(args.csv_downsample_stride),
        step_stride=int(args.step_stride),
        horizon_points=int(args.horizon_points),
        require_full_horizon=not bool(args.allow_tail_pad),
        absolute_sample_count=int(args.absolute_sample_count),
    )
    xy_errors = np.asarray([row.xy_error_m for row in rows], dtype=np.float64)
    yaw_errors = np.asarray([abs(row.yaw_error_deg) for row in rows], dtype=np.float64)
    print(f"CSV rows after downsampling: {len(df)}")
    print(f"Tracking steps: {len(rows)}")
    print(f"Mean XY error: {float(np.mean(xy_errors)):.6f} m")
    print(f"Max  XY error: {float(np.max(xy_errors)):.6f} m")
    print(f"Mean |yaw| error: {float(np.mean(yaw_errors)):.6f} deg")
    for key, path in out_paths.items():
        print(f"SVG written ({key}): {path}")


if __name__ == "__main__":
    main()
