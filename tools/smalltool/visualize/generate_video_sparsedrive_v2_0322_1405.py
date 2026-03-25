#!/usr/bin/env python3
"""Generate one closed-loop rollout video with SparseDriveV2 in ReconSimulator.

Pipeline per step:
1) Use current observation to run SparseDriveV2 planner.
2) Execute only the first point as env continuous action (flag=2).
3) Re-plan at next step.

Unavailable parts are kept as placeholders (e.g., exact mode log-prob from V2).
"""

from __future__ import annotations

import argparse
import csv
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

_DEFAULT_CKPT = os.path.join(_REPO_ROOT, "SparseDriveV2", "ckpt", "sparsedrive_navsimv2.ckpt")
_DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "outputs", "visualize", "sparsedriveV2")


def _lazy_import_runtime() -> tuple[Any, Any]:
    try:
        from framework.env_wrapper import RLReconEnv  # type: ignore
        from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy  # type: ignore

        return RLReconEnv, SparseDriveV2Policy
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "Missing runtime dependency for SparseDriveV2 rollout. "
            f"Import failed on module: {missing}. Activate project env and retry."
        ) from e


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _grid_frame(observation: Dict[str, np.ndarray]) -> np.ndarray:
    keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
    imgs = [observation[k] for k in keys]
    row1 = np.concatenate(imgs[:3], axis=1)
    row2 = np.concatenate(imgs[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)


def _pose_matrix_from_xyyaw(x: float, y: float, yaw: float) -> np.ndarray:
    c = float(math.cos(yaw))
    s = float(math.sin(yaw))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    return T


def _predict_next_pose_from_action(start_ego: np.ndarray, action: tuple[Any, ...]) -> np.ndarray:
    if not (isinstance(action, (tuple, list)) and len(action) == 4 and int(action[3]) == 2):
        raise RuntimeError(f"Unsupported continuous action for prediction: {action}")
    dx = float(action[0])
    dy = float(action[1])
    dyaw = float(action[2])
    return np.asarray(start_ego, dtype=np.float64) @ _pose_matrix_from_xyyaw(dx, dy, dyaw)


def _yaw_from_R_xy(Rm: np.ndarray) -> float:
    return float(np.arctan2(float(Rm[1, 0]), float(Rm[0, 0])))


def _local_plan_to_front_frame(start_ego: np.ndarray, traj_xyyaw: np.ndarray) -> np.ndarray:
    out = np.zeros((traj_xyyaw.shape[0], 4), dtype=np.float64)
    for i in range(traj_xyyaw.shape[0]):
        lx, ly, lyaw = float(traj_xyyaw[i, 0]), float(traj_xyyaw[i, 1]), float(traj_xyyaw[i, 2])
        tpt = _pose_matrix_from_xyyaw(lx, ly, lyaw)
        T_front = np.asarray(start_ego, dtype=np.float64) @ tpt
        out[i, 0] = float(T_front[0, 3])
        out[i, 1] = float(T_front[1, 3])
        out[i, 2] = float(T_front[2, 3])
        out[i, 3] = _yaw_from_R_xy(T_front[:3, :3])
    return out


def _ensure_obs_for_sparsedrive_v2(obs: Dict[str, Any], sim: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(obs)
    out.setdefault("timestamp", np.float32(float(getattr(sim, "now_frame", 0)) * 0.1))
    if "ego_pose" not in out:
        out["ego_pose"] = np.asarray(getattr(sim, "start_ego", np.eye(4)), dtype=np.float32)
    if "cam2ego" not in out:
        cam2ego = getattr(sim, "cam2ego", None)
        if isinstance(cam2ego, list) and len(cam2ego) >= 3:
            out["cam2ego"] = np.asarray(np.stack(cam2ego, axis=0), dtype=np.float32)
    if "cam_intrinsics" not in out:
        all_cams = getattr(sim, "all_cams", None)
        if isinstance(all_cams, list) and len(all_cams) >= 3:
            intr = []
            hw = []
            for cam in all_cams:
                intr.append(np.asarray(cam.get("intrinsics"), dtype=np.float32))
                hw.append([float(cam.get("height", sim.h)), float(cam.get("width", sim.w))])
            out["cam_intrinsics"] = np.asarray(np.stack(intr, axis=0), dtype=np.float32)
            out.setdefault("cam_hw", np.asarray(hw, dtype=np.float32))
    if "driving_command" not in out:
        out["driving_command"] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if "ego_status" not in out:
        vel = np.asarray(out.get("ego_velocity", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        acc = np.asarray(out.get("ego_acceleration", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        cmd = np.asarray(out.get("driving_command", np.zeros((4,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        cmd4 = np.zeros((4,), dtype=np.float32)
        vel2 = np.zeros((2,), dtype=np.float32)
        acc2 = np.zeros((2,), dtype=np.float32)
        cmd4[: min(4, cmd.shape[0])] = cmd[: min(4, cmd.shape[0])]
        vel2[: min(2, vel.shape[0])] = vel[: min(2, vel.shape[0])]
        acc2[: min(2, acc.shape[0])] = acc[: min(2, acc.shape[0])]
        out["ego_status"] = np.concatenate([cmd4, vel2, acc2], axis=0).astype(np.float32)
    return out


def _traj_xyyaw_from_replay(replay: Dict[str, Any]) -> np.ndarray:
    traj = replay.get("traj_xyyaw", None)
    if traj is None:
        raise RuntimeError("Replay missing traj_xyyaw")
    if torch.is_tensor(traj):
        arr = traj.detach().cpu().numpy()
    else:
        arr = np.asarray(traj)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RuntimeError(f"Invalid traj_xyyaw shape: {arr.shape}")
    return arr[:, :3]


def _load_expert_traj_front_xz(scene: int, start_frame: int, step_frames: int) -> np.ndarray:
    from reconsimulator.envs import nus_config as cfg  # type: ignore

    scene_dir = os.path.join(cfg.BASE_DATA_DIR, f"{int(scene):03d}")
    ego_pose_dir = os.path.join(scene_dir, "ego_pose")
    cam2ego0_path = os.path.join(scene_dir, "cam2ego", "0.txt")

    if not os.path.isdir(ego_pose_dir):
        raise FileNotFoundError(f"missing dir: {ego_pose_dir}")
    if not os.path.isfile(cam2ego0_path):
        raise FileNotFoundError(f"missing file: {cam2ego0_path}")

    pose_files = [n for n in os.listdir(ego_pose_dir) if n.endswith(".txt")]
    all_frames = sorted(int(os.path.splitext(n)[0]) for n in pose_files)
    frames = [f for f in all_frames if f >= int(start_frame) and ((f - int(start_frame)) % int(step_frames) == 0)]
    if not frames:
        return np.zeros((0, 2), dtype=np.float64)

    ego0_world = np.asarray(np.loadtxt(os.path.join(ego_pose_dir, f"{int(start_frame):03d}.txt")), dtype=np.float64)
    cam2ego0 = np.asarray(np.loadtxt(cam2ego0_path), dtype=np.float64)
    camera_front_start = ego0_world @ cam2ego0
    inv_front = np.linalg.inv(camera_front_start)

    rows = []
    for f in frames:
        T_ego_world = np.asarray(np.loadtxt(os.path.join(ego_pose_dir, f"{int(f):03d}.txt")), dtype=np.float64)
        T_front = inv_front @ T_ego_world
        rows.append([float(T_front[0, 3]), float(T_front[2, 3])])
    return np.asarray(rows, dtype=np.float64)


def _save_traj_plot_xz(scene: int, expert_xz: np.ndarray, ego_xz: np.ndarray, out_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[traj-plot] matplotlib not installed, skip export")
        return False

    if expert_xz.ndim != 2 or expert_xz.shape[1] != 2:
        raise RuntimeError(f"Invalid expert_xz shape: {expert_xz.shape}")
    if ego_xz.ndim != 2 or ego_xz.shape[1] != 2:
        raise RuntimeError(f"Invalid ego_xz shape: {ego_xz.shape}")

    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=140)
    ax.plot(expert_xz[:, 0], expert_xz[:, 1], color="#1f77b4", linewidth=2.0, label="expert")
    ax.plot(ego_xz[:, 0], ego_xz[:, 1], color="#d62728", linewidth=2.0, label="ego")
    ax.scatter(expert_xz[0, 0], expert_xz[0, 1], color="#1f77b4", s=28)
    ax.scatter(ego_xz[0, 0], ego_xz[0, 1], color="#d62728", s=28)
    ax.set_title(f"Scene {scene:03d}: SparseDriveV2 Expert vs Ego (front-frame XZ)")
    ax.set_xlabel("x (right +)")
    ax.set_ylabel("z (forward/north +)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SparseDriveV2 rollout video in 3DGS env")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--ckpt", type=str, default=_DEFAULT_CKPT)
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
    args = ap.parse_args()

    scene = int(args.scene)
    RLReconEnv, SparseDriveV2Policy = _lazy_import_runtime()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"SparseDriveV2 ckpt not found: {args.ckpt}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.out or os.path.join(_DEFAULT_OUT_DIR, f"scene{scene:03d}_{ts}_sparsedrivev2_rollout.mp4")
    traj_csv = args.traj_csv or os.path.join(_DEFAULT_OUT_DIR, f"scene{scene:03d}_{ts}_sparsedrivev2_plan_frontframe.csv")
    traj_plot = args.traj_plot or os.path.join(_DEFAULT_OUT_DIR, f"scene{scene:03d}_{ts}_sparsedrivev2_expert_vs_ego_traj.svg")
    _ensure_parent(out_path)
    _ensure_parent(traj_csv)
    _ensure_parent(traj_plot)

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
        ckpt_path=str(args.ckpt),
        device=(f"cuda:{int(args.cuda)}" if torch.cuda.is_available() else "cpu"),
        execute_mode="first_step",
        rl_lr=1e-5,
    )

    step_dt_s = float(getattr(sim, "step_frames", int(args.step_frames))) * 0.1
    if step_dt_s <= 0:
        raise RuntimeError("Invalid step dt")
    max_steps = None if args.duration_s is None else max(1, int(round(float(args.duration_s) / step_dt_s)))
    fps = float(args.fps) if args.fps is not None else (1.0 / step_dt_s)

    print("==== generate_video_sparsedrive_v2 ====")
    print(f"scene={scene} start_frame={int(args.start_frame)} step_frames={int(args.step_frames)}")
    print(f"ckpt={args.ckpt}")
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

    rows: List[Dict[str, float | int]] = []
    ego_xz: List[List[float]] = []

    start_pose = np.asarray(getattr(sim, "start_ego"), dtype=np.float64)
    ego_xz.append([float(start_pose[0, 3]), float(start_pose[2, 3])])

    done = False
    steps = 0
    frames = 0
    writer.append_data(_grid_frame(obs))
    frames += 1

    while (max_steps is None or steps < max_steps) and not done:
        obs_in = _ensure_obs_for_sparsedrive_v2(obs, sim)
        start_ego = np.asarray(getattr(sim, "start_ego"), dtype=np.float64).copy()
        now_frame = int(getattr(sim, "now_frame", -1))

        action, logp, replay = policy.sample_sparsedrivev2_with_replay(
            obs_in,
            mode_idx=-1,
            mode_select=str(args.mode_select),
        )

        traj_xyyaw = _traj_xyyaw_from_replay(replay)
        traj_front = _local_plan_to_front_frame(start_ego, traj_xyyaw)

        logp_v = float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp)
        print(f"[plan-v2] step={steps} frame={now_frame} shape={traj_xyyaw.shape}")
        print(np.array2string(traj_xyyaw, precision=6, suppress_small=False))

        pred_next = _predict_next_pose_from_action(start_ego, action)

        rows.append(
            {
                "step": int(steps),
                "frame": int(now_frame),
                "plan_idx": 0,
                "cmd_idx": int(replay.get("cmd_idx", -1)),
                "mode_idx": int(replay.get("mode_idx", -1)),
                "logp": float(logp_v),
                "local_x": float(traj_xyyaw[0, 0]),
                "local_y": float(traj_xyyaw[0, 1]),
                "local_yaw": float(traj_xyyaw[0, 2]),
                "front_x": float(traj_front[0, 0]),
                "front_y": float(traj_front[0, 1]),
                "front_z": float(traj_front[0, 2]),
                "front_yaw": float(traj_front[0, 3]),
            }
        )

        obs, _reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)

        pose_after = np.asarray(getattr(sim, "start_ego"), dtype=np.float64)
        ego_xz.append([float(pose_after[0, 3]), float(pose_after[2, 3])])

        pred_xz = np.asarray([float(pred_next[0, 3]), float(pred_next[2, 3])], dtype=np.float64)
        real_xz = np.asarray([float(pose_after[0, 3]), float(pose_after[2, 3])], dtype=np.float64)
        err_xz = float(np.linalg.norm(pred_xz - real_xz, ord=2))
        print(
            "[pose-check-v2] "
            f"step={steps} frame={now_frame} "
            f"action(dx,dy,dyaw)=({float(action[0]):.6f},{float(action[1]):.6f},{float(action[2]):.6f}) "
            f"pred_next_xz=({pred_xz[0]:.6f},{pred_xz[1]:.6f}) "
            f"real_next_xz=({real_xz[0]:.6f},{real_xz[1]:.6f}) "
            f"l2_err={err_xz:.9f}"
        )

        writer.append_data(_grid_frame(obs))
        frames += 1
        steps += 1

    writer.close()

    fieldnames = [
        "step",
        "frame",
        "plan_idx",
        "cmd_idx",
        "mode_idx",
        "logp",
        "local_x",
        "local_y",
        "local_yaw",
        "front_x",
        "front_y",
        "front_z",
        "front_yaw",
    ]
    with open(traj_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    try:
        expert_xz = _load_expert_traj_front_xz(
            scene=scene,
            start_frame=int(args.start_frame),
            step_frames=int(args.step_frames),
        )
    except Exception as e:
        print(f"[traj-plot] failed to load expert ego2world trajectory: {e}")
        expert_xz = np.zeros((0, 2), dtype=np.float64)

    ego_xz_np = np.asarray(ego_xz, dtype=np.float64)
    print(f"[traj-v2] ego_xz shape={ego_xz_np.shape}")
    print(np.array2string(ego_xz_np, precision=6, suppress_small=False))
    print(f"[traj-v2] expert_xz shape={expert_xz.shape}")
    print(np.array2string(expert_xz, precision=6, suppress_small=False))

    if expert_xz.shape[0] >= 2 and ego_xz_np.shape[0] >= 2:
        saved = _save_traj_plot_xz(scene=scene, expert_xz=expert_xz, ego_xz=ego_xz_np, out_path=traj_plot)
        if saved:
            print(f"traj_plot_saved={traj_plot}")
    else:
        print("[traj-plot] skip export due to insufficient trajectory points")

    print(f"done={done} steps={steps} frames={frames}")
    print(f"video_saved={out_path}")
    print(f"traj_saved={traj_csv}")
    print("==== all done ====")


if __name__ == "__main__":
    main()
