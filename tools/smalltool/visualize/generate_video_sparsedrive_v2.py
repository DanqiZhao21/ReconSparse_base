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

def _resolve_ego_ads_subdir(name: str) -> str:
    preferred = os.path.join(_REPO_ROOT, "egoADs", str(name))
    if os.path.isdir(preferred):
        return preferred
    return os.path.join(_REPO_ROOT, str(name))


def _resolve_repo_path(path: str) -> str:
    text = str(path)
    if os.path.isabs(text):
        return text
    direct = os.path.join(_REPO_ROOT, text)
    if os.path.exists(direct):
        return direct
    egoads_candidate = os.path.join(_REPO_ROOT, "egoADs", text)
    if os.path.exists(egoads_candidate):
        return egoads_candidate
    return direct


_DEFAULT_CKPT = os.path.join(_resolve_ego_ads_subdir("SparseDriveV2"), "ckpt", "sparsedrive_navsimv2.ckpt")
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


def _relative_local_xyyaw(prev_pose: np.ndarray, next_pose: np.ndarray) -> np.ndarray:
    rel = np.linalg.inv(np.asarray(prev_pose, dtype=np.float64)) @ np.asarray(next_pose, dtype=np.float64)
    return np.asarray(
        [
            float(rel[0, 3]),
            float(rel[1, 3]),
            float(_yaw_from_R_xy(rel[:3, :3])),
        ],
        dtype=np.float64,
    )


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


def _extract_status_from_obs(obs: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cmd = np.asarray(obs.get("driving_command", np.zeros((4,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    vel = np.asarray(obs.get("ego_velocity", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    acc = np.asarray(obs.get("ego_acceleration", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)

    cmd4 = np.zeros((4,), dtype=np.float32)
    vel2 = np.zeros((2,), dtype=np.float32)
    acc2 = np.zeros((2,), dtype=np.float32)
    cmd4[: min(4, cmd.shape[0])] = cmd[: min(4, cmd.shape[0])]
    vel2[: min(2, vel.shape[0])] = vel[: min(2, vel.shape[0])]
    acc2[: min(2, acc.shape[0])] = acc[: min(2, acc.shape[0])]
    return cmd4, vel2, acc2


def _dataset_status_from_sim(sim: Any, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Prefer simulator's internal dataset mapping when available.
    fn = getattr(sim, "_status_from_dataset", None)
    if callable(fn):
        try:
            vel, acc, cmd = fn(int(frame_idx))
            vel2 = np.asarray(vel, dtype=np.float32).reshape(-1)[:2]
            acc2 = np.asarray(acc, dtype=np.float32).reshape(-1)[:2]
            cmd4 = np.asarray(cmd, dtype=np.float32).reshape(-1)[:4]

            out_vel = np.zeros((2,), dtype=np.float32)
            out_acc = np.zeros((2,), dtype=np.float32)
            out_cmd = np.zeros((4,), dtype=np.float32)
            out_vel[: vel2.shape[0]] = vel2
            out_acc[: acc2.shape[0]] = acc2
            out_cmd[: cmd4.shape[0]] = cmd4
            return out_cmd, out_vel, out_acc
        except Exception:
            pass
    return (
        np.zeros((4,), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
    )


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


def _load_expert_front_xz_for_frame(
    scene: int,
    start_frame: int,
    frame_idx: int,
    *,
    base_data_dir: str | None = None,
) -> np.ndarray:
    if base_data_dir is None:
        from reconsimulator.envs import nus_config as cfg  # type: ignore

        base_data_dir = str(cfg.BASE_DATA_DIR)

    scene_dir = os.path.join(str(base_data_dir), f"{int(scene):03d}")
    ego_pose_dir = os.path.join(scene_dir, "ego_pose")
    cam2ego0_path = os.path.join(scene_dir, "cam2ego", "0.txt")

    ego0_world = np.asarray(np.loadtxt(os.path.join(ego_pose_dir, f"{int(start_frame):03d}.txt")), dtype=np.float64)
    cam2ego0 = np.asarray(np.loadtxt(cam2ego0_path), dtype=np.float64)
    camera_front_start = ego0_world @ cam2ego0
    inv_front = np.linalg.inv(camera_front_start)

    T_ego_world = np.asarray(np.loadtxt(os.path.join(ego_pose_dir, f"{int(frame_idx):03d}.txt")), dtype=np.float64)
    T_front = inv_front @ T_ego_world
    return np.asarray([float(T_front[0, 3]), float(T_front[2, 3])], dtype=np.float64)


def _append_online_expert_xz(target: List[List[float]], expert_front_xz: np.ndarray) -> None:
    arr = np.asarray(expert_front_xz, dtype=np.float64).reshape(2)
    target.append([float(arr[0]), float(arr[1])])


def _build_online_step_stats_paths(traj_plot_path: str) -> Dict[str, str]:
    traj_plot_abs = os.path.abspath(str(traj_plot_path))
    root, _ext = os.path.splitext(traj_plot_abs)
    suffix = "_expert_vs_ego_traj"
    if root.endswith(suffix):
        prefix = root[: -len(suffix)]
    else:
        prefix = root
    return {
        "per_step_csv": f"{prefix}_online_step_summary.csv",
        "aggregate_csv": f"{prefix}_online_step_aggregate.csv",
        "rollout_csv": f"{prefix}_online_rollout_points.csv",
        "overlay_svg": f"{prefix}_online_rollout_overlay.svg",
        "error_hist_svg": f"{prefix}_online_error_hist.svg",
        "worst_svg": f"{prefix}_online_worst_steps.svg",
    }


def _load_scene99_step_summary_module() -> Any:
    module_path = os.path.join(_REPO_ROOT, "outputs", "visualize", "debug_tracker_scene099", "summarize_scene99_tracker_steps.py")
    spec = importlib.util.spec_from_file_location("summarize_scene99_tracker_steps", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load scene99 summary module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_online_rollout_points_csv(rows: List[Dict[str, float | int]], out_path: str) -> None:
    _ensure_parent(out_path)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _step_marker_indices(num_points: int, every: int = 5) -> List[int]:
    if num_points <= 0:
        return []
    every_n = max(1, int(every))
    indices = list(range(0, int(num_points), every_n))
    last_idx = int(num_points) - 1
    if indices[-1] != last_idx:
        indices.append(last_idx)
    return indices


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
    for step_idx in _step_marker_indices(expert_xz.shape[0], every=5):
        x_val = float(expert_xz[step_idx, 0])
        z_val = float(expert_xz[step_idx, 1])
        ax.scatter([x_val], [z_val], color="#1f77b4", s=34, marker="o")
        ax.annotate(
            f"step {step_idx}",
            (x_val, z_val),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#1f77b4",
            alpha=0.95,
        )
    for step_idx in _step_marker_indices(ego_xz.shape[0], every=5):
        x_val = float(ego_xz[step_idx, 0])
        z_val = float(ego_xz[step_idx, 1])
        ax.scatter([x_val], [z_val], color="#d62728", s=38, marker="^")
        ax.annotate(
            f"step {step_idx}",
            (x_val, z_val),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=7,
            color="#d62728",
            alpha=0.95,
        )
    ax.set_title(f"Scene {scene:03d}: SparseDriveV2 Expert vs Ego (front-frame XZ, markers every 5 steps)")
    ax.set_xlabel("x (right +)")
    ax.set_ylabel("z (forward/north +)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    # fig.savefig(out_path)
    fig.savefig(out_path, format='svg')  
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
    ckpt_path = _resolve_repo_path(str(args.ckpt))

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"SparseDriveV2 ckpt not found: {ckpt_path}")

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

    print("==== generate_video_sparsedrive_v2 ====")
    print(f"scene={scene} start_frame={int(args.start_frame)} step_frames={int(args.step_frames)}")
    print(f"ckpt={ckpt_path}")
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
    expert_xz_online: List[List[float]] = []
    online_summary_rows: List[Dict[str, float | int]] = []
    online_rollout_rows: List[Dict[str, float | int]] = []

    start_pose = np.asarray(getattr(sim, "start_ego"), dtype=np.float64)
    ego_xz.append([float(start_pose[0, 3]), float(start_pose[2, 3])])
    try:
        start_frame_expert_xz = _load_expert_front_xz_for_frame(
            scene=scene,
            start_frame=int(args.start_frame),
            frame_idx=int(args.start_frame),
        )
        _append_online_expert_xz(expert_xz_online, start_frame_expert_xz)
    except Exception as e:
        print(f"[traj-online] failed to load expert start pose: {e}")

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

        # Prediction based on the plan's first point in front-start frame.
        pred_plan_front_xz = np.asarray([float(traj_front[0, 0]), float(traj_front[0, 2])], dtype=np.float64)

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

        # Feed the full planned trajectory to simulator so PDM tracks actual planner output.
        setattr(sim, "_external_plan_local_xyyaw", np.asarray(traj_xyyaw, dtype=np.float64).copy())

        obs, _reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)

        pose_after = np.asarray(obs.get("ego_pose", getattr(sim, "start_ego")), dtype=np.float64)
        ego_xz.append([float(pose_after[0, 3]), float(pose_after[2, 3])])

        pred_xz = pred_plan_front_xz
        real_xz = np.asarray([float(pose_after[0, 3]), float(pose_after[2, 3])], dtype=np.float64)
        err_xz = float(np.linalg.norm(pred_xz - real_xz, ord=2))

        frame_after = int(getattr(sim, "now_frame", -1))
        tracked_first_local = np.asarray(getattr(sim, "_tracked_first_step_xyyaw", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(3)
        executed_first_local = np.asarray(
            getattr(sim, "_executed_first_step_xyyaw", tracked_first_local),
            dtype=np.float64,
        ).reshape(3)
        tracked_rollout = np.asarray(getattr(sim, "_tracked_rollout_local_xyyaw", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
        actual_local = _relative_local_xyyaw(start_ego, pose_after)
        tracked_first_front = _local_plan_to_front_frame(start_ego, tracked_first_local.reshape(1, 3))[0]
        try:
            expert_after_xz = _load_expert_front_xz_for_frame(
                scene=scene,
                start_frame=int(args.start_frame),
                frame_idx=int(frame_after),
            )
            _append_online_expert_xz(expert_xz_online, expert_after_xz)
        except Exception as e:
            print(f"[traj-online] failed to load expert pose for frame={frame_after}: {e}")
            expert_after_xz = np.asarray([np.nan, np.nan], dtype=np.float64)
        cmd_obs, vel_obs, acc_obs = _extract_status_from_obs(obs)
        cmd_ds, vel_ds, acc_ds = _dataset_status_from_sim(sim, frame_after)

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
            "[pose-check-v2] "
            f"step={steps} frame={now_frame} "
            f"action(dx,dy,dyaw)=({float(action[0]):.6f},{float(action[1]):.6f},{float(action[2]):.6f}) "
            f"tracked_first_local=({tracked_first_local[0]:.6f},{tracked_first_local[1]:.6f},{tracked_first_local[2]:.6f}) "
            f"executed_first_local=({executed_first_local[0]:.6f},{executed_first_local[1]:.6f},{executed_first_local[2]:.6f}) "
            f"actual_local=({actual_local[0]:.6f},{actual_local[1]:.6f},{actual_local[2]:.6f}) "
            f"pred_src=plan_first_point "
            f"pred_next_xz=({pred_xz[0]:.6f},{pred_xz[1]:.6f}) "
            f"real_next_xz=({real_xz[0]:.6f},{real_xz[1]:.6f}) "
            f"l2_err={err_xz:.9f}"
        )
        print(
            "[status-check-v2] "
            f"step={steps} frame_after={frame_after} "
            f"command_obs={np.array2string(cmd_obs, precision=6, suppress_small=False)} "
            f"vel_obs={np.array2string(vel_obs, precision=6, suppress_small=False)} "
            f"acc_obs={np.array2string(acc_obs, precision=6, suppress_small=False)} "
            # f"command_dataset={np.array2string(cmd_ds, precision=6, suppress_small=False)} "
            f"vel_dataset={np.array2string(vel_ds, precision=6, suppress_small=False)} "
            f"acc_dataset={np.array2string(acc_ds, precision=6, suppress_small=False)}"
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

    ego_xz_np = np.asarray(ego_xz, dtype=np.float64)
    expert_xz_np = np.asarray(expert_xz_online, dtype=np.float64)
    print(f"[traj-v2] ego_xz shape={ego_xz_np.shape}")
    print(np.array2string(ego_xz_np, precision=6, suppress_small=False))
    print(f"[traj-v2-online] expert_xz shape={expert_xz_np.shape}")
    print(np.array2string(expert_xz_np, precision=6, suppress_small=False))

    if expert_xz_np.shape[0] >= 2 and ego_xz_np.shape[0] >= 2:
        saved = _save_traj_plot_xz(scene=scene, expert_xz=expert_xz_np, ego_xz=ego_xz_np, out_path=traj_plot)
        if saved:
            print(f"traj_plot_saved={traj_plot}")
    else:
        print("[traj-plot] skip export due to insufficient online trajectory points")

    try:
        stats_paths = _build_online_step_stats_paths(traj_plot)
        stats_module = _load_scene99_step_summary_module()
        rollout_by_step = stats_module.build_step_rollout_arrays(online_rollout_rows)
        per_step_rows, aggregate = stats_module.summarize_step_tracking(online_summary_rows, rollout_by_step)
        stats_module._save_csv_rows(per_step_rows, stats_paths["per_step_csv"])
        stats_module._save_csv_row(aggregate, stats_paths["aggregate_csv"])
        _save_online_rollout_points_csv(online_rollout_rows, stats_paths["rollout_csv"])
        stats_module._save_overlay_plot(rollout_by_step, stats_paths["overlay_svg"])
        stats_module._save_error_hist_plot(per_step_rows, stats_paths["error_hist_svg"])
        stats_module._save_worst_cases_plot(rollout_by_step, per_step_rows, stats_paths["worst_svg"])
        print(f"[online-step-stats] num_steps={int(aggregate['num_steps'])}")
        print(f"[online-step-stats] mean_first_point_plan_tracked_xy_err_m={float(aggregate['mean_first_point_plan_tracked_xy_err_m']):.9f}")
        print(f"[online-step-stats] mean_rollout_mean_xy_err_m={float(aggregate['mean_rollout_mean_xy_err_m']):.9f}")
        print(f"[online-step-stats] mean_expert_actual_front_xz_err_m={float(aggregate['mean_expert_actual_front_xz_err_m']):.9f}")
        print(f"online_step_summary_saved={stats_paths['per_step_csv']}")
        print(f"online_step_aggregate_saved={stats_paths['aggregate_csv']}")
        print(f"online_rollout_points_saved={stats_paths['rollout_csv']}")
        print(f"online_rollout_overlay_saved={stats_paths['overlay_svg']}")
        print(f"online_error_hist_saved={stats_paths['error_hist_svg']}")
        print(f"online_worst_steps_saved={stats_paths['worst_svg']}")
    except Exception as e:
        print(f"[online-step-stats] failed to export online stats: {e}")

    print(f"done={done} steps={steps} frames={frames}")
    print(f"video_saved={out_path}")
    print(f"traj_saved={traj_csv}")
    print("==== all done ====")


if __name__ == "__main__":
    main()
