#!/usr/bin/env python3
"""Generate one closed-loop rollout video with SparseDrive in ReconSimulator.

Pipeline per step:
1) Use current observation to run SparseDrive planner (6 points / 3s by default).
2) Execute only the first point as env continuous action (flag=2).
3) Next observation arrives, then re-plan again.

For debugging, this script also exports planned trajectories transformed from
current ego-local frame to simulator front-start frame using:
    T_world_like = start_ego @ T_tpt(local_x, local_y, local_yaw)
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

_DEFAULT_CONFIG = os.path.join(
    _REPO_ROOT,
    "SparseDrive",
    "projects",
    "configs",
    "sparsedrive_small_stage2.py",
)
_DEFAULT_CKPT = os.path.join(_REPO_ROOT, "SparseDrive", "ckpt", "sparsedrive_stage2.pth")


def _lazy_import_runtime() -> tuple[Any, Any]:
    try:
        from framework.env_wrapper import RLReconEnv  # type: ignore
        from framework.agent.policy_sparsedrive import SparseDrivePolicy  # type: ignore

        return RLReconEnv, SparseDrivePolicy
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "Missing runtime dependency for SparseDrive rollout. "
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
    T[:3, :3] = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    return T


def _yaw_from_R_xy(Rm: np.ndarray) -> float:
    return float(np.arctan2(float(Rm[1, 0]), float(Rm[0, 0])))


def _ensure_obs_for_sparsedrive(obs: Dict[str, Any], sim: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(obs)

    if "ego_pose" not in out:
        print("💣[attn in genVideo] Warning: obs missing **ego_pose**, attempting to fill from sim.start_ego or identity")
        out["ego_pose"] = np.asarray(getattr(sim, "start_ego", np.eye(4)), dtype=np.float32)

    if "cam2ego" not in out:
        print("💣[attn in genVideo] Warning: obs missing **cam2ego**, attempting to fill from sim.cam2ego if available")
        cam2ego = getattr(sim, "cam2ego", None)
        if isinstance(cam2ego, list) and len(cam2ego) == 6:
            out["cam2ego"] = np.asarray(np.stack(cam2ego, axis=0), dtype=np.float32)

    if "cam_intrinsics" not in out:
        print("💣[attn in genVideo] Warning: obs missing **cam_intrinsics**, attempting to fill from sim.all_cams if available")
        all_cams = getattr(sim, "all_cams", None)
        if isinstance(all_cams, list) and len(all_cams) == 6:
            intr = []
            hw = []
            for cam in all_cams:
                intr.append(np.asarray(cam.get("intrinsics"), dtype=np.float32))
                hw.append([float(cam.get("height", sim.h)), float(cam.get("width", sim.w))])
            out["cam_intrinsics"] = np.asarray(np.stack(intr, axis=0), dtype=np.float32)
            out.setdefault("cam_hw", np.asarray(hw, dtype=np.float32))

    out.setdefault("timestamp", np.float32(float(getattr(sim, "now_frame", 0)) * 0.1))

    if "driving_command" not in out:
        print("💣[attn in genVideo] Warning: obs missing **driving_command**, attempting to fill with default")
        out["driving_command"] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    if "ego_status" not in out:
        print("💣[attn in genVideo] Warning: obs missing **ego_status**, attempting to fill")
        vel = np.asarray(out.get("ego_velocity", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        acc = np.asarray(out.get("ego_acceleration", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        cmd = np.asarray(out.get("driving_command", np.zeros((4,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if cmd.shape[0] < 4:
            cmd_pad = np.zeros((4,), dtype=np.float32)
            cmd_pad[: cmd.shape[0]] = cmd
            cmd = cmd_pad
        if vel.shape[0] < 2:
            vel_pad = np.zeros((2,), dtype=np.float32)
            vel_pad[: vel.shape[0]] = vel
            vel = vel_pad
        if acc.shape[0] < 2:
            acc_pad = np.zeros((2,), dtype=np.float32)
            acc_pad[: acc.shape[0]] = acc
            acc = acc_pad
        out["ego_status"] = np.concatenate([cmd[:4], vel[:2], acc[:2]], axis=0).astype(np.float32)

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

#这个是从自车ego-local的规划转化成tpt矩阵；然后再左乘start_ego得到在front-start坐标系下 走完第一个规划点的位置（froan camera坐标）
def _local_plan_to_front_frame(start_ego: np.ndarray, traj_xyyaw: np.ndarray) -> np.ndarray:
    """Convert local planned poses to front-start frame via left-multiplication.

    Returns array with shape (N, 4): [x, y, z, yaw_xy].
    """
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

##以下是新增的绘图函数##################################################################################
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
    if not pose_files:
        raise RuntimeError(f"no ego pose files under {ego_pose_dir}")

    all_frames = sorted(int(os.path.splitext(n)[0]) for n in pose_files)
    frames = [f for f in all_frames if f >= int(start_frame) and ((f - int(start_frame)) % int(step_frames) == 0)]
    if not frames:
        raise RuntimeError("no frames selected for expert trajectory")

    ego0_world = np.asarray(np.loadtxt(os.path.join(ego_pose_dir, f"{int(start_frame):03d}.txt")), dtype=np.float64)
    cam2ego0 = np.asarray(np.loadtxt(cam2ego0_path), dtype=np.float64)
    camera_front_start = ego0_world @ cam2ego0
    inv_camera_front_start = np.linalg.inv(camera_front_start)

    rows = []
    for f in frames:
        T_ego_world = np.asarray(np.loadtxt(os.path.join(ego_pose_dir, f"{int(f):03d}.txt")), dtype=np.float64)
        T_front = inv_camera_front_start @ T_ego_world
        rows.append([float(T_front[0, 3]), float(T_front[2, 3])])
    return np.asarray(rows, dtype=np.float64)


def _save_traj_plot_xz(scene: int, expert_xz: np.ndarray, ego_xz: np.ndarray, out_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[traj-plot] matplotlib not installed, skip trajectory plot export")
        return False

    if expert_xz.ndim != 2 or expert_xz.shape[1] != 2:
        raise RuntimeError(f"Invalid expert_xz shape: {expert_xz.shape}")
    if ego_xz.ndim != 2 or ego_xz.shape[1] != 2:
        raise RuntimeError(f"Invalid ego_xz shape: {ego_xz.shape}")

    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=140)

    ax.plot(expert_xz[:, 0], expert_xz[:, 1], color="#1f77b4", linewidth=2.0, label="expert (from ego2world)")
    ax.plot(ego_xz[:, 0], ego_xz[:, 1], color="#d62728", linewidth=2.0, label="ego (rollout)")

    ax.scatter(expert_xz[0, 0], expert_xz[0, 1], color="#1f77b4", s=28)
    ax.scatter(ego_xz[0, 0], ego_xz[0, 1], color="#d62728", s=28)

    ax.set_title(f"Scene {scene:03d}: Expert vs Ego Trajectory (front-frame XZ)")
    ax.set_xlabel("x (right +)")
    ax.set_ylabel("z (forward/north +)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True

##以上是新增的绘图函数##################################################################

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SparseDrive rollout video in 3DGS env")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--config", type=str, default=_DEFAULT_CONFIG)
    ap.add_argument("--ckpt", type=str, default=_DEFAULT_CKPT)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--traj-csv", type=str, default=None, help="Export per-step planned 6-point trajectories")
    ap.add_argument("--traj-plot", type=str, default=None, help="Export expert-vs-ego trajectory plot (.png)")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--step-frames", type=int, default=5)
    ap.add_argument("--duration-s", type=float, default=None, help="If unset, run until env done.")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--render-w", type=int, default=None)
    ap.add_argument("--render-h", type=int, default=None)
    ap.add_argument("--mode-select", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--execute-mode", type=str, default="first_step", choices=["first_step"])
    ap.add_argument("--expert-high", dest="expert_high", action="store_true", default=True)
    ap.add_argument("--no-expert-high", dest="expert_high", action="store_false")
    args = ap.parse_args()

    scene = int(args.scene)
    RLReconEnv, SparseDrivePolicy = _lazy_import_runtime()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"SparseDrive config not found: {args.config}")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"SparseDrive ckpt not found: {args.ckpt}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.out or os.path.join(
        _REPO_ROOT,
        "outputs",
        "visualize",
        f"scene{scene:03d}_{ts}_sparsedrive_rollout.mp4",
    )
    traj_csv = args.traj_csv or os.path.join(
        _REPO_ROOT,
        "outputs",
        "visualize",
        f"scene{scene:03d}_{ts}_sparsedrive_plan_frontframe.csv",
    )
    traj_plot = args.traj_plot or os.path.join(
        _REPO_ROOT,
        "outputs",
        "visualize",
        f"scene{scene:03d}_{ts}_expert_vs_ego_traj.png",
    )
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

    policy = SparseDrivePolicy(
        config_path=str(args.config),
        ckpt_path=str(args.ckpt),
        device=(f"cuda:{int(args.cuda)}" if torch.cuda.is_available() else "cpu"),
        execute_mode=str(args.execute_mode),
        rl_lr=1e-5,
        x_anchor=int(getattr(sim, "x_anchor", 61)),
        y_anchor=int(getattr(sim, "y_anchor", 61)),
    )

    step_dt_s = float(getattr(sim, "step_frames", int(args.step_frames))) * 0.1
    if step_dt_s <= 0:
        raise RuntimeError("Invalid step dt")
    if args.duration_s is None:
        max_steps = None
    else:
        max_steps = max(1, int(round(float(args.duration_s) / step_dt_s)))
    fps = float(args.fps) if args.fps is not None else (1.0 / step_dt_s)
    if fps <= 0:
        raise RuntimeError("fps must be > 0")

    print("==== generate_video_sparsedrive ====")
    print(f"scene={scene} start_frame={int(args.start_frame)} step_frames={int(args.step_frames)}")
    print(f"config={args.config}")
    print(f"ckpt={args.ckpt}")
    print(f"execute_mode={args.execute_mode} mode_select={args.mode_select}")
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
        obs_in = _ensure_obs_for_sparsedrive(obs, sim)
        start_ego = np.asarray(getattr(sim, "start_ego"), dtype=np.float64).copy()
        now_frame = int(getattr(sim, "now_frame", -1))

        #这里的action形状就是(dx, dy, yaw, 2) 
        action, logp, replay = policy.sample_sparsedrive_with_replay(
            obs_in,
            mode_idx=-1,
            mode_select=str(args.mode_select),
        )

        traj_xyyaw = _traj_xyyaw_from_replay(replay)##这里replay存的就是traj-xyyaw，shape是（T,3），每行是（x,y,yaw）
        traj_front = _local_plan_to_front_frame(start_ego, traj_xyyaw) #这里已经是得到的走的第一个点的front camera下面的位置了

        cmd_idx = int(replay.get("cmd_idx", -1))
        mode_idx = int(replay.get("mode_idx", -1))
        logp_v = float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp)

        # Print planned local trajectory for current step: shape (6, 3) -> (x, y, yaw).
        print(f"[plan] step={steps} frame={now_frame} cmd_idx={cmd_idx} mode_idx={mode_idx} shape={traj_xyyaw.shape}")
        print(np.array2string(traj_xyyaw, precision=6, suppress_small=False))
        print(f"[action] step={steps} frame={now_frame} action_x={action[0]} action_y={action[1]} action_yaw={action[2]} ")

        # Log only the actually executed point each env step: plan_idx is always 0.
        rows.append(
            {
                "step": int(steps),
                "frame": int(now_frame),
                "plan_idx": 0,
                "cmd_idx": int(cmd_idx),
                "mode_idx": int(mode_idx),
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
        #在ego pose更新之前自己算一下：
        pose_before = np.asarray(getattr(sim, "start_ego"), dtype=np.float64)
        print("[front camera pose] pose_before =\n", pose_before)
        


        obs, _reward, terminated, truncated, _info = env.step(action)#action的形状是(dx, dy, yaw, 2)
        done = bool(terminated or truncated)
        pose_after = np.asarray(getattr(sim, "start_ego"), dtype=np.float64)
        print("[front camera pose] pose_after =\n", pose_after)
        ego_xz.append([float(pose_after[0, 3]), float(pose_after[2, 3])])

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
    #这里是将自车在ego-local坐标系下xy的转化成了xz平面 然后和专家的轨迹使用同一个plot_xz函数

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
    print(f"[traj] ego_xz shape={ego_xz_np.shape}")
    print(np.array2string(ego_xz_np, precision=6, suppress_small=False))
    print(f"[traj] expert_xz shape={expert_xz.shape}")
    print(np.array2string(expert_xz, precision=6, suppress_small=False))

    if expert_xz.shape[0] >= 2 and len(ego_xz) >= 2:
        saved = _save_traj_plot_xz(
            scene=scene,
            expert_xz=np.asarray(expert_xz, dtype=np.float64),
            ego_xz=ego_xz_np,
            out_path=traj_plot,
        )
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
