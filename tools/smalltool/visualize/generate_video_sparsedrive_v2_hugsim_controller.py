#!/usr/bin/env python3
"""Generate SparseDriveV2 closed-loop video using HUGSIM controller semantics.

Pipeline per step:
1) Use current observation to run SparseDriveV2 planner.
2) Convert planner trajectory to HUGSIM trajectory coordinates.
3) Use HUGSIM `traj2control` to get `(acc, steer_rate)`.
4) Propagate one control interval with HUGSIM bicycle dynamics.
5) Inject the executed local pose/state into ReconSimulator for rendering.
"""
# ''''''
# python tools/smalltool/visualize/generate_video_sparsedrive_v2_hugsim_controller.py   --scene 56
# '''
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from typing import Any, Dict, List

import imageio
import numpy as np
import torch


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.utils.tracker_execution import build_execution_result

_BASE_MODULE_PATH = os.path.join(_REPO_ROOT, "tools", "smalltool", "visualize", "generate_video_sparsedrive_v2.py")
_DEFAULT_HUGSIM_REPO = "/root/clone/HUGSIM"

_BASE_MODULE: Any | None = None
_HUGSIM_SOLVER_CACHE: dict[tuple[float, float], Any] = {}


def _load_base_module() -> Any:
    global _BASE_MODULE
    if _BASE_MODULE is not None:
        return _BASE_MODULE
    spec = importlib.util.spec_from_file_location("generate_video_sparsedrive_v2_base", _BASE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base rollout module: {_BASE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _BASE_MODULE = module
    return module


def _lazy_import_runtime() -> tuple[Any, Any]:
    return _load_base_module()._lazy_import_runtime()


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
    info: Dict[str, float],
    *,
    wheelbase: float | None,
    plan_dt: float,
    control_dt: float | None,
    build_solver_fn: Any,
) -> tuple[np.ndarray, np.ndarray]:
    dt = float(info.get("dt", plan_dt)) if control_dt is None else float(control_dt)
    reference = _build_hugsim_reference_trajectory(plan_traj, plan_dt=float(plan_dt), control_dt=float(dt))
    current_state = np.array([0.0, 0.0, 0.0, float(info["ego_velo"]), float(info["ego_steer"])], dtype=np.float64)
    wb = 2.7 if wheelbase is None else float(wheelbase)
    cache_key = (round(wb, 6), round(float(dt), 6))
    solver = _HUGSIM_SOLVER_CACHE.get(cache_key)
    if solver is None:
        solver = build_solver_fn(wb, float(dt))
        _HUGSIM_SOLVER_CACHE[cache_key] = solver
    solutions = solver.solve(current_state, reference)
    final = solutions[-1]
    return np.asarray(final.state_trajectory, dtype=np.float64), np.asarray(final.input_trajectory, dtype=np.float64)


def _build_execution_from_hugsim_solution(
    *,
    prev_pose: np.ndarray,
    plan_local_xyyaw: np.ndarray,
    state_trajectory: np.ndarray,
    input_trajectory: np.ndarray,
) -> Any:
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


def _load_hugsim_helpers(repo_path: str) -> tuple[Any, Any, Any]:
    resolved = os.path.abspath(str(repo_path))
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    try:
        import sim.ilqr.lqr as hugsim_lqr  # type: ignore
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "Missing HUGSIM runtime dependency for controller rollout. "
            f"Import failed on module: {missing}. Check --hugsim-repo and env."
        ) from e
    vehicle_module_path = os.path.join(resolved, "sim", "utils", "sparsedrive_vehicle.py")
    spec = importlib.util.spec_from_file_location("hugsim_sparsedrive_vehicle", vehicle_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load HUGSIM vehicle helper: {vehicle_module_path}")
    vehicle_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vehicle_module)
    return _solve_hugsim_control_sequence, hugsim_lqr._build_solver, vehicle_module.load_sparsedrive_wheelbase


def _plan_local_xyyaw_to_hugsim_plan_xy(plan_local_xyyaw: np.ndarray) -> np.ndarray:
    plan = np.asarray(plan_local_xyyaw, dtype=np.float64)
    if plan.ndim != 2 or plan.shape[1] < 2:
        raise ValueError(f"Expected (N, >=2) local plan, got {plan.shape}")
    out = np.zeros((plan.shape[0], 2), dtype=np.float64)
    out[:, 0] = -plan[:, 1]
    out[:, 1] = plan[:, 0]
    return out


def _build_hugsim_controller_info(*, obs: Dict[str, Any], sim: Any, dt: float, wheelbase: float) -> Dict[str, float]:
    vel = np.asarray(obs.get("ego_velocity", np.zeros((2,), dtype=np.float32)), dtype=np.float64).reshape(-1)
    speed = float(np.linalg.norm(vel[:2], ord=2)) if vel.size > 0 else 0.0
    steer = float(getattr(sim, "_status_steering_angle", 0.0))
    return {
        "ego_velo": float(speed),
        "ego_steer": float(steer),
        "dt": float(dt),
        "wheelbase": float(wheelbase),
    }


def _hugsim_one_step_local_xyyaw(
    *,
    speed: float,
    steer: float,
    acc: float,
    steer_rate: float,
    dt: float,
    wheelbase: float,
) -> Dict[str, float | np.ndarray]:
    dt_s = float(dt)
    L = max(1e-6, float(wheelbase))
    speed_next = float(speed) + float(acc) * dt_s
    steer_next = float(steer) + float(steer_rate) * dt_s
    theta_prev = 0.0
    x_right = float(speed_next) * math.sin(theta_prev) * dt_s
    y_forward = float(speed_next) * math.cos(theta_prev) * dt_s
    theta_next = theta_prev + float(speed_next) * math.tan(float(steer_next)) / L * dt_s
    return {
        "local_xyyaw": np.asarray([y_forward, -x_right, theta_next], dtype=np.float64),
        "speed_next": float(speed_next),
        "steer_next": float(steer_next),
    }


def _build_hugsim_execution_override(
    *,
    prev_pose: np.ndarray,
    obs: Dict[str, Any],
    sim: Any,
    traj_xyyaw: np.ndarray,
    dt: float,
    wheelbase: float,
    solve_sequence_fn: Any,
    build_solver_fn: Any,
) -> tuple[Any, Dict[str, float]]:
    controller_info = _build_hugsim_controller_info(obs=obs, sim=sim, dt=float(dt), wheelbase=float(wheelbase))
    hugsim_plan_xy = _plan_local_xyyaw_to_hugsim_plan_xy(traj_xyyaw)
    state_trajectory, input_trajectory = solve_sequence_fn(
        hugsim_plan_xy,
        controller_info,
        wheelbase=float(wheelbase),
        plan_dt=float(dt),
        control_dt=float(dt),
        build_solver_fn=build_solver_fn,
    )
    execution = _build_execution_from_hugsim_solution(
        prev_pose=np.asarray(prev_pose, dtype=np.float64),
        plan_local_xyyaw=np.asarray(traj_xyyaw, dtype=np.float64),
        state_trajectory=np.asarray(state_trajectory, dtype=np.float64),
        input_trajectory=np.asarray(input_trajectory, dtype=np.float64),
    )
    return execution, {
        "acc": float(input_trajectory[0, 0]),
        "steer_rate": float(input_trajectory[0, 1]),
        "speed_next": float(state_trajectory[1, 3]),
        "steer_next": float(state_trajectory[1, 4]),
    }


def main() -> None:
    base = _load_base_module()

    ap = argparse.ArgumentParser(description="Generate SparseDriveV2 rollout video using HUGSIM controller semantics")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--ckpt", type=str, default=base._DEFAULT_CKPT)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--traj-csv", type=str, default=None)
    ap.add_argument("--traj-plot", type=str, default=None)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--step-frames", type=int, default=5)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--render-w", type=int, default=None)
    ap.add_argument("--render-h", type=int, default=None)
    ap.add_argument("--mode-select", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--expert-high", dest="expert_high", action="store_true", default=True)
    ap.add_argument("--no-expert-high", dest="expert_high", action="store_false")
    ap.add_argument("--hugsim-repo", type=str, default=_DEFAULT_HUGSIM_REPO)
    ap.add_argument("--wheelbase", type=float, default=None)
    args = ap.parse_args()

    scene = int(args.scene)
    RLReconEnv, SparseDriveV2Policy = _lazy_import_runtime()
    solve_sequence_fn, build_solver_fn, load_sparsedrive_wheelbase = _load_hugsim_helpers(str(args.hugsim_repo))
    ckpt_path = base._resolve_repo_path(str(args.ckpt))

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"SparseDriveV2 ckpt not found: {ckpt_path}")

    sparse_repo = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    wheelbase = float(args.wheelbase) if args.wheelbase is not None else load_sparsedrive_wheelbase(sparse_repo, None)
    if wheelbase is None:
        wheelbase = 2.7
    wheelbase = float(wheelbase)

    ts = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"scene{scene:03d}_{ts}_sparsedrivev2_hugsimctrl"
    out_path = args.out or os.path.join(base._DEFAULT_OUT_DIR, f"{suffix}_rollout.mp4")
    traj_csv = args.traj_csv or os.path.join(base._DEFAULT_OUT_DIR, f"{suffix}_plan_frontframe.csv")
    traj_plot = args.traj_plot or os.path.join(base._DEFAULT_OUT_DIR, f"{suffix}_expert_vs_ego_traj.svg")
    base._ensure_parent(out_path)
    base._ensure_parent(traj_csv)
    base._ensure_parent(traj_plot)

    env = RLReconEnv(
        cuda=int(args.cuda),
        scene=scene,
        reward_cfg={},
        debug=bool(args.debug),
        render_w=(int(args.render_w) if args.render_w is not None else None),
        render_h=(int(args.render_h) if args.render_h is not None else None),
    )

    obs, _info = env.reset(scene=scene, start_frame=int(args.start_frame), step_frames=int(args.step_frames))
    sim = getattr(env, "env")
    setattr(sim, "use_expert_height", bool(args.expert_high))

    policy = SparseDriveV2Policy(
        ckpt_path=str(ckpt_path),
        device=(f"cuda:{int(args.cuda)}" if torch.cuda.is_available() else "cpu"),
        execute_mode="first_step",
        rl_lr=1e-5,
    )

    step_dt_s = float(getattr(sim, "step_frames", int(args.step_frames))) * 0.1
    if step_dt_s <= 0:
        raise RuntimeError("Invalid step dt")
    max_steps = None if args.duration_s is None else max(1, int(round(float(args.duration_s) / step_dt_s)))
    fps = float(args.fps) if args.fps is not None else (1.0 / step_dt_s)

    print("==== generate_video_sparsedrive_v2_hugsim_controller ====")
    print(f"scene={scene} start_frame={int(args.start_frame)} step_frames={int(args.step_frames)}")
    print(f"ckpt={ckpt_path}")
    print(f"hugsim_repo={os.path.abspath(str(args.hugsim_repo))}")
    print(f"wheelbase={wheelbase:.6f}")
    print(f"mode_select={args.mode_select}")
    print(f"use_expert_height={bool(args.expert_high)}")
    if args.duration_s is None:
        print(f"duration_s=until_done max_steps=none step_dt_s={step_dt_s:.3f} fps={fps:.3f}")
    else:
        print(f"duration_s={float(args.duration_s):.3f} max_steps={max_steps} step_dt_s={step_dt_s:.3f} fps={fps:.3f}")
    print(f"out_video={out_path}")
    print(f"out_traj_csv={traj_csv}")
    print(f"out_traj_plot={traj_plot}")

    writer = imageio.get_writer(
        out_path,
        mode="I",
        fps=float(fps),
        macro_block_size=1,
        codec="libx264",
        ffmpeg_log_level="error",
        input_params=["-framerate", str(float(fps))],
        output_params=["-pix_fmt", "yuv420p"],
    )

    rows: List[Dict[str, float | int | str]] = []
    ego_xz: List[List[float]] = []
    expert_xz_online: List[List[float]] = []
    online_summary_rows: List[Dict[str, float | int]] = []
    online_rollout_rows: List[Dict[str, float | int]] = []

    start_pose = np.asarray(getattr(sim, "start_ego"), dtype=np.float64)
    ego_xz.append([float(start_pose[0, 3]), float(start_pose[2, 3])])
    try:
        start_frame_expert_xz = base._load_expert_front_xz_for_frame(
            scene=scene,
            start_frame=int(args.start_frame),
            frame_idx=int(args.start_frame),
        )
        base._append_online_expert_xz(expert_xz_online, start_frame_expert_xz)
    except Exception as e:
        print(f"[traj-online-hugsim] failed to load expert start pose: {e}")

    done = False
    steps = 0
    frames = 0
    writer.append_data(base._grid_frame(obs))
    frames += 1

    while (max_steps is None or steps < max_steps) and not done:
        obs_in = base._ensure_obs_for_sparsedrive_v2(obs, sim)
        start_ego = np.asarray(getattr(sim, "start_ego"), dtype=np.float64).copy()
        now_frame = int(getattr(sim, "now_frame", -1))

        action, logp, replay = policy.sample_sparsedrivev2_with_replay(
            obs_in,
            mode_idx=-1,
            mode_select=str(args.mode_select),
        )
        traj_xyyaw = base._traj_xyyaw_from_replay(replay)
        traj_front = base._local_plan_to_front_frame(start_ego, traj_xyyaw)
        execution_override, control_meta = _build_hugsim_execution_override(
            prev_pose=np.asarray(start_ego, dtype=np.float64),
            obs=obs_in,
            sim=sim,
            traj_xyyaw=np.asarray(traj_xyyaw, dtype=np.float64),
            dt=float(step_dt_s),
            wheelbase=float(wheelbase),
            solve_sequence_fn=solve_sequence_fn,
            build_solver_fn=build_solver_fn,
        )

        logp_v = float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp)
        print(f"[plan-v2-hugsim] step={steps} frame={now_frame} shape={traj_xyyaw.shape}")
        print(np.array2string(traj_xyyaw, precision=6, suppress_small=False))

        pred_plan_front_xz = np.asarray([float(traj_front[0, 0]), float(traj_front[0, 2])], dtype=np.float64)

        rows.append(
            {
                "step": int(steps),
                "frame": int(now_frame),
                "plan_idx": 0,
                "cmd_idx": int(replay.get("cmd_idx", -1)),
                "mode_idx": int(replay.get("mode_idx", -1)),
                "logp": float(logp_v),
                "controller": "hugsim",
                "acc": float(control_meta["acc"]),
                "steer_rate": float(control_meta["steer_rate"]),
                "local_x": float(traj_xyyaw[0, 0]),
                "local_y": float(traj_xyyaw[0, 1]),
                "local_yaw": float(traj_xyyaw[0, 2]),
                "front_x": float(traj_front[0, 0]),
                "front_y": float(traj_front[0, 1]),
                "front_z": float(traj_front[0, 2]),
                "front_yaw": float(traj_front[0, 3]),
            }
        )

        setattr(sim, "_external_plan_local_xyyaw", np.asarray(traj_xyyaw, dtype=np.float64).copy())
        setattr(sim, "_external_execution_override", execution_override)

        obs, _reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)

        pose_after = np.asarray(obs.get("ego_pose", getattr(sim, "start_ego")), dtype=np.float64)
        ego_xz.append([float(pose_after[0, 3]), float(pose_after[2, 3])])

        pred_xz = pred_plan_front_xz
        real_xz = np.asarray([float(pose_after[0, 3]), float(pose_after[2, 3])], dtype=np.float64)
        err_xz = float(np.linalg.norm(pred_xz - real_xz, ord=2))

        frame_after = int(getattr(sim, "now_frame", -1))
        tracked_first_local = np.asarray(getattr(sim, "_tracked_first_step_xyyaw", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(3)
        executed_first_local = np.asarray(getattr(sim, "_executed_first_step_xyyaw", tracked_first_local), dtype=np.float64).reshape(3)
        tracked_rollout = np.asarray(getattr(sim, "_tracked_rollout_local_xyyaw", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
        actual_local = base._relative_local_xyyaw(start_ego, pose_after)
        tracked_first_front = base._local_plan_to_front_frame(start_ego, tracked_first_local.reshape(1, 3))[0]

        try:
            expert_after_xz = base._load_expert_front_xz_for_frame(
                scene=scene,
                start_frame=int(args.start_frame),
                frame_idx=int(frame_after),
            )
            base._append_online_expert_xz(expert_xz_online, expert_after_xz)
        except Exception as e:
            print(f"[traj-online-hugsim] failed to load expert pose for frame={frame_after}: {e}")
            expert_after_xz = np.asarray([np.nan, np.nan], dtype=np.float64)

        online_summary_rows.append(
            {
                "step": int(steps),
                "frame_before": int(now_frame),
                "frame_after": int(frame_after),
                "plan_tracked_xy_err": float(np.linalg.norm(traj_xyyaw[0, :2] - tracked_first_local[:2])),
                "tracked_executed_xy_err": float(np.linalg.norm(tracked_first_local[:2] - executed_first_local[:2])),
                "plan_actual_front_xz_err": float(np.linalg.norm(pred_xz - real_xz, ord=2)),
                "tracked_actual_front_xz_err": float(np.linalg.norm(tracked_first_front[[0, 2]] - np.asarray([real_xz[0], real_xz[1]], dtype=np.float64), ord=2)),
                "expert_actual_front_xz_err": float(np.linalg.norm(expert_after_xz - real_xz, ord=2)) if np.isfinite(expert_after_xz).all() else float("nan"),
                "tracked_local_x": float(tracked_first_local[0]),
                "tracked_local_y": float(tracked_first_local[1]),
                "tracked_local_yaw": float(tracked_first_local[2]),
                "executed_local_x": float(executed_first_local[0]),
                "executed_local_y": float(executed_first_local[1]),
                "executed_local_yaw": float(executed_first_local[2]),
                "actual_local_x": float(actual_local[0]),
                "actual_local_y": float(actual_local[1]),
                "actual_local_yaw": float(actual_local[2]),
            }
        )
        for pt_idx in range(int(traj_xyyaw.shape[0])):
            tracked_pt = tracked_rollout[pt_idx] if tracked_rollout.ndim == 2 and pt_idx < tracked_rollout.shape[0] else np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
            online_rollout_rows.append(
                {
                    "step": int(steps),
                    "point_idx": int(pt_idx),
                    "plan_local_x": float(traj_xyyaw[pt_idx, 0]),
                    "plan_local_y": float(traj_xyyaw[pt_idx, 1]),
                    "plan_local_yaw": float(traj_xyyaw[pt_idx, 2]),
                    "tracked_local_x": float(tracked_pt[0]),
                    "tracked_local_y": float(tracked_pt[1]),
                    "tracked_local_yaw": float(tracked_pt[2]),
                }
            )

        print(
            "[pose-check-v2-hugsim] "
            f"step={steps} frame={now_frame} "
            f"acc={float(control_meta['acc']):.6f} steer_rate={float(control_meta['steer_rate']):.6f} "
            f"tracked_first_local=({tracked_first_local[0]:.6f},{tracked_first_local[1]:.6f},{tracked_first_local[2]:.6f}) "
            f"executed_first_local=({executed_first_local[0]:.6f},{executed_first_local[1]:.6f},{executed_first_local[2]:.6f}) "
            f"actual_local=({actual_local[0]:.6f},{actual_local[1]:.6f},{actual_local[2]:.6f}) "
            f"pred_next_xz=({pred_xz[0]:.6f},{pred_xz[1]:.6f}) "
            f"real_next_xz=({real_xz[0]:.6f},{real_xz[1]:.6f}) "
            f"l2_err={err_xz:.9f}"
        )

        writer.append_data(base._grid_frame(obs))
        frames += 1
        steps += 1

    writer.close()

    fieldnames = list(rows[0].keys()) if rows else []
    with open(traj_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    ego_xz_np = np.asarray(ego_xz, dtype=np.float64)
    expert_xz_np = np.asarray(expert_xz_online, dtype=np.float64)
    print(f"[traj-v2-hugsim] ego_xz shape={ego_xz_np.shape}")
    print(np.array2string(ego_xz_np, precision=6, suppress_small=False))
    print(f"[traj-v2-hugsim-online] expert_xz shape={expert_xz_np.shape}")
    print(np.array2string(expert_xz_np, precision=6, suppress_small=False))

    if expert_xz_np.shape[0] >= 2 and ego_xz_np.shape[0] >= 2:
        saved = base._save_traj_plot_xz(scene=scene, expert_xz=expert_xz_np, ego_xz=ego_xz_np, out_path=traj_plot)
        if saved:
            print(f"traj_plot_saved={traj_plot}")
    else:
        print("[traj-plot-hugsim] skip export due to insufficient online trajectory points")

    try:
        stats_paths = base._build_online_step_stats_paths(traj_plot)
        stats_module = base._load_scene99_step_summary_module()
        rollout_by_step = stats_module.build_step_rollout_arrays(online_rollout_rows)
        per_step_rows, aggregate = stats_module.summarize_step_tracking(online_summary_rows, rollout_by_step)
        stats_module._save_csv_rows(per_step_rows, stats_paths["per_step_csv"])
        stats_module._save_csv_row(aggregate, stats_paths["aggregate_csv"])
        base._save_online_rollout_points_csv(online_rollout_rows, stats_paths["rollout_csv"])
        stats_module._save_overlay_plot(rollout_by_step, stats_paths["overlay_svg"])
        stats_module._save_error_hist_plot(per_step_rows, stats_paths["error_hist_svg"])
        stats_module._save_worst_cases_plot(rollout_by_step, per_step_rows, stats_paths["worst_svg"])
        print(f"[online-step-stats-hugsim] num_steps={int(aggregate['num_steps'])}")
        print(f"[online-step-stats-hugsim] mean_first_point_plan_tracked_xy_err_m={float(aggregate['mean_first_point_plan_tracked_xy_err_m']):.9f}")
        print(f"[online-step-stats-hugsim] mean_rollout_mean_xy_err_m={float(aggregate['mean_rollout_mean_xy_err_m']):.9f}")
        print(f"[online-step-stats-hugsim] mean_expert_actual_front_xz_err_m={float(aggregate['mean_expert_actual_front_xz_err_m']):.9f}")
        print(f"online_step_summary_saved={stats_paths['per_step_csv']}")
        print(f"online_step_aggregate_saved={stats_paths['aggregate_csv']}")
        print(f"online_rollout_points_saved={stats_paths['rollout_csv']}")
        print(f"online_rollout_overlay_saved={stats_paths['overlay_svg']}")
        print(f"online_error_hist_saved={stats_paths['error_hist_svg']}")
        print(f"online_worst_steps_saved={stats_paths['worst_svg']}")
    except Exception as e:
        print(f"[online-step-stats-hugsim] failed to export online stats: {e}")

    print(f"done={done} steps={steps} frames={frames}")
    print(f"video_saved={out_path}")
    print(f"traj_saved={traj_csv}")
    print("==== all done ====")


if __name__ == "__main__":
    main()
