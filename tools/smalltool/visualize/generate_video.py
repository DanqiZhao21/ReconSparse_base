#!/usr/bin/env python3
"""Minimal closed-loop video generator for one scene.

Features kept intentionally small:
- Input a scene id and generate one MP4.
- Print DDV2 full plan (8x3) at every step (when not expert mode).
- Print final executed ego trajectory points and expert trajectory points.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

import imageio
import numpy as np
import torch


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_CKPT = os.path.join(_REPO_ROOT, "outputs", "weight", "20260129_ppo_ver27_latest.ckpt")


def _default_traj_plot_path(video_path: str) -> str:
    root, _ = os.path.splitext(video_path)
    return f"{root}_traj.png"


def _default_height_plot_path(video_path: str) -> str:
    root, _ = os.path.splitext(video_path)
    return f"{root}_height.png"


def _default_ground3d_plot_path(video_path: str) -> str:
    root, _ = os.path.splitext(video_path)
    return f"{root}_ground3d.png"


def _lazy_import_runtime() -> tuple[Any, Any]:
    try:
        from framework.env_wrapper import RLReconEnv  # type: ignore
        from framework.agent.policy_diffusiondrivev2 import DiffusionDriveV2Policy  # type: ignore

        return RLReconEnv, DiffusionDriveV2Policy
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "Missing runtime dependency for video rollout. "
            f"Import failed on module: {missing}. "
            "Activate project env and retry."
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


def _yaw_from_R_xz(R: np.ndarray) -> float | None:
    if R.shape[0] < 3 or R.shape[1] < 3:
        return None
    return float(np.arctan2(float(R[2, 0]), float(R[0, 0])))#R[0,0] = x_axis 的 world-x 分量//R[2,0] = x_axis 的 world-z 分量  atan2(z, x);;yaw 是从 x 轴转到当前方向测量的角度


#
def _pose_xzyaw_from_matrix(T: Any) -> tuple[float, float, float] | None:
    if T is None:
        return None
    arr = np.asarray(T, dtype=np.float32)
    if arr.shape[0] < 4 or arr.shape[1] < 4:
        return None
    yaw = _yaw_from_R_xz(arr[:3, :3])
    if yaw is None:
        return None
    return float(arr[0, 3]), float(arr[2, 3]), float(yaw)


def _pose_xzyaw_from_info(info: Dict[str, Any], prefix: str) -> tuple[float, float, float] | None:
    pos = info.get(f"{prefix}_pos", None)
    yaw_deg = info.get(f"{prefix}_yaw_deg", None)
    if pos is None or yaw_deg is None:
        return None
    p = np.asarray(pos, dtype=np.float32).reshape(-1)
    if p.shape[0] < 3:
        return None
    return float(p[0]), float(p[2]), float(np.deg2rad(float(yaw_deg)))


def _save_trajectory_plot(
    out_path: str,
    act_traj: np.ndarray,
    exp_traj: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing matplotlib for trajectory plot. "
            f"Import failed on module: {getattr(e, 'name', None) or str(e)}"
        ) from e

    _ensure_parent(out_path)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    if exp_traj.size > 0:
        ax.plot(exp_traj[:, 0], exp_traj[:, 1], color="#1f77b4", linewidth=2.2, label="expert")
    if act_traj.size > 0:
        ax.plot(act_traj[:, 0], act_traj[:, 1], color="#d62728", linewidth=2.2, label="ego")

    ax.set_title("Trajectory Comparison (forward-x vs left-z)")
    ax.set_xlabel("forward x")
    ax.set_ylabel("left z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_height_plot(out_path: str, t_s: list[float], act_y: list[float], exp_y: list[float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing matplotlib for height plot. "
            f"Import failed on module: {getattr(e, 'name', None) or str(e)}"
        ) from e

    if not t_s or len(t_s) != len(act_y) or len(t_s) != len(exp_y):
        raise RuntimeError("Height series length mismatch")

    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    ax.plot(t_s, exp_y, color="#1f77b4", linewidth=2.2, label="expert_y")
    ax.plot(t_s, act_y, color="#d62728", linewidth=2.2, label="ego_y")
    ax.set_title("Height Comparison (y)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_ground_match_3d_plot(
    out_path: str,
    expert_xyz_all: np.ndarray,
    ego_xyz: np.ndarray,
    match_xyz: np.ndarray,
    match_dist_m: np.ndarray,
    frame_idx: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing matplotlib for 3D ground-match plot. "
            f"Import failed on module: {getattr(e, 'name', None) or str(e)}"
        ) from e

    _ensure_parent(out_path)
    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    if expert_xyz_all.size > 0:
        ax.plot(
            expert_xyz_all[:, 0],
            expert_xyz_all[:, 2],
            expert_xyz_all[:, 1],
            color="#1f77b4",
            linewidth=1.6,
            alpha=0.85,
            label="expert_full_xyz",
        )

    if ego_xyz.size > 0:
        ax.plot(
            ego_xyz[:, 0],
            ego_xyz[:, 2],
            ego_xyz[:, 1],
            color="#d62728",
            linewidth=2.0,
            alpha=0.95,
            label="ego_executed_xyz",
        )

        # Label each ego point with its frame index for debugging alignment.
        if frame_idx.size == ego_xyz.shape[0]:
            for i in range(ego_xyz.shape[0]):
                ex, ey, ez = float(ego_xyz[i, 0]), float(ego_xyz[i, 1]), float(ego_xyz[i, 2])
                ax.text(ex, ez, ey, str(int(frame_idx[i])), fontsize=5, color="#222222", alpha=0.8)

    if ego_xyz.shape[0] > 0 and match_xyz.shape[0] == ego_xyz.shape[0]:
        valid = np.all(np.isfinite(match_xyz), axis=1)
        added_vertical_label = False
        added_match_label = False
        for i in np.where(valid)[0].tolist():
            ex, ey, ez = float(ego_xyz[i, 0]), float(ego_xyz[i, 1]), float(ego_xyz[i, 2])
            mx, my, mz = float(match_xyz[i, 0]), float(match_xyz[i, 1]), float(match_xyz[i, 2])

            # Vertical line at ego (x,z): actual applied height update on ego pose.
            vertical_label = "applied_y_at_ego_xz" if not added_vertical_label else None
            ax.plot(
                [ex, ex],
                [ez, ez],
                [ey, my],
                linestyle="-",
                color="#555555",
                linewidth=1.0,
                alpha=0.55,
                label=vertical_label,
            )
            added_vertical_label = True

            # Dashed line to the nearest expert reference point used to fetch y.
            # This can be slanted because nearest-point matching allows x-z offset.
            match_label = "nearest_expert_ref(xz,y)" if not added_match_label else None
            ax.plot(
                [ex, mx],
                [ez, mz],
                [my, my],
                linestyle="--",
                color="#9a9a9a",
                linewidth=0.8,
                alpha=0.45,
                label=match_label,
            )
            added_match_label = True

    mean_dist = float(np.nanmean(match_dist_m)) if match_dist_m.size > 0 else float("nan")
    ax.set_title(f"Ground-Y Match Debug (axes=x,z,y; mean nearest-xz dist={mean_dist:.3f} m)")
    ax.set_xlabel("forward x")
    ax.set_ylabel("left z")
    ax.set_zlabel("up y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate one scene rollout MP4 (minimal).")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--traj-plot-out", type=str, default=None)
    ap.add_argument("--height-plot-out", type=str, default=None)
    ap.add_argument("--ground3d-plot-out", type=str, default=None)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--expert", action="store_true")
    ap.add_argument("--expert-high", action="store_true", help="Use expert current-frame y directly for camera height.")
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--execute-mode", type=str, default="nearest_anchor", choices=["first_step", "continuous", "nearest_anchor"])
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--step-frames", type=int, default=None)
    ap.add_argument("--frame-dt-s", type=float, default=0.1)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--render-w", type=int, default=None)
    ap.add_argument("--render-h", type=int, default=None)
    args = ap.parse_args()

    RLReconEnv, DiffusionDriveV2Policy = _lazy_import_runtime()

    scene_id = int(args.scene)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = str(args.out) if args.out else os.path.join(_REPO_ROOT, "outputs", "visualize", f"scene{scene_id:03d}_{ts}.mp4")
    traj_plot_out = str(args.traj_plot_out) if args.traj_plot_out else _default_traj_plot_path(out_path)
    height_plot_out = str(args.height_plot_out) if args.height_plot_out else _default_height_plot_path(out_path)
    ground3d_plot_out = str(args.ground3d_plot_out) if args.ground3d_plot_out else _default_ground3d_plot_path(out_path)
    _ensure_parent(out_path)
    _ensure_parent(traj_plot_out)
    _ensure_parent(height_plot_out)
    _ensure_parent(ground3d_plot_out)

    cuda = int(args.cuda)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    env = RLReconEnv(
        cuda=cuda,
        scene=scene_id,
        reward_cfg={},
        debug=bool(args.debug),
        render_w=(int(args.render_w) if args.render_w is not None else None),
        render_h=(int(args.render_h) if args.render_h is not None else None),
    )
    obs, _ = env.reset(scene=scene_id, start_frame=0, step_frames=(int(args.step_frames) if args.step_frames is not None else None))
    sim = getattr(env, "env", None)
    if sim is not None:
        setattr(sim, "use_expert_height", bool(args.expert_high))

    step_frames = int(getattr(sim, "step_frames", 1)) if sim is not None else (int(args.step_frames) if args.step_frames is not None else 1)
    step_dt_s = float(step_frames) * float(args.frame_dt_s)
    if step_dt_s <= 0:
        raise RuntimeError("Invalid step dt")

    if args.duration_s is None:
        max_steps = None
    else:
        dur = float(args.duration_s)
        if dur <= 0:
            raise RuntimeError("--duration-s must be > 0")
        max_steps = max(1, int(round(dur / step_dt_s)))

    fps_used = float(args.fps) if args.fps is not None else (1.0 / step_dt_s)
    if fps_used <= 0:
        raise RuntimeError("--fps must be > 0")

    policy = None
    if not bool(args.expert):
        ckpt = str(args.ckpt)
        if not ckpt:
            raise RuntimeError("Empty --ckpt")
        policy = DiffusionDriveV2Policy(
            x_anchor=int(getattr(env.env, "x_anchor", 61)),
            y_anchor=int(getattr(env.env, "y_anchor", 61)),
            ckpt_path=ckpt,
            device=str(device),
            rl_lr=1e-5,
            reinforce_baseline_beta=0.98,
            execute_mode=str(args.execute_mode),
        )

    print("==== generate_video minimal ====")
    print(f"scene={scene_id} out={out_path}")
    print(f"traj_plot_out={traj_plot_out}")
    print(f"height_plot_out={height_plot_out}")
    print(f"ground3d_plot_out={ground3d_plot_out}")
    print(f"expert_high={bool(args.expert_high)}")
    print(f"mode={'expert' if args.expert else 'ddv2'} execute_mode={args.execute_mode} step_dt={step_dt_s:.3f}s fps={fps_used:.3f}")

    writer = imageio.get_writer(
        out_path,
        mode="I",
        fps=float(fps_used),
        macro_block_size=1,
        codec="libx264",
        ffmpeg_log_level="error",
        input_params=["-framerate", str(float(fps_used))],
        output_params=["-pix_fmt", "yuv420p"],
    )

    steps = 0
    done = False
    frames = 0

    act_traj: list[tuple[float, float, float]] = []
    exp_traj: list[tuple[float, float, float]] = []
    act_xyz_series: list[tuple[float, float, float]] = []
    act_frame_idx_series: list[int] = []
    matched_xyz_series: list[tuple[float, float, float]] = []
    matched_dist_series: list[float] = []
    t_series_s: list[float] = []
    act_y_series: list[float] = []
    exp_y_series: list[float] = []

    start_pose = _pose_xzyaw_from_matrix(getattr(sim, "start_ego", None)) if sim is not None else None
    if start_pose is not None:
        act_traj.append(start_pose)
        exp_traj.append(start_pose)
    start_T = getattr(sim, "start_ego", None) if sim is not None else None
    if start_T is not None:
        start_arr = np.asarray(start_T, dtype=np.float32)
        if start_arr.shape[0] >= 4 and start_arr.shape[1] >= 4:
            y0 = float(start_arr[1, 3])
            x0 = float(start_arr[0, 3])
            z0 = float(start_arr[2, 3])
            act_xyz_series.append((x0, y0, z0))
            if sim is not None:
                act_frame_idx_series.append(int(getattr(sim, "now_frame", 0)))
            else:
                act_frame_idx_series.append(0)
            matched_xyz_series.append((np.nan, np.nan, np.nan))
            matched_dist_series.append(float("nan"))
            t_series_s.append(0.0)
            act_y_series.append(y0)
            exp_y_series.append(y0)

    writer.append_data(_grid_frame(obs))
    frames += 1

    while True:
        if max_steps is not None and steps >= int(max_steps):
            break

        if bool(args.expert):
            action = (0, 0, 1)
        else:
            assert policy is not None
            action, _logp, replay = policy.sample_ddv2rl_with_replay(
                obs,
                eta=1.0,
                mode_idx=-1,
                mode_select="greedy",
            )
            full_plan = replay.get("traj_xyyaw", None) if isinstance(replay, dict) else None
            if torch.is_tensor(full_plan):
                plan_np = full_plan.detach().cpu().numpy().astype(np.float32, copy=False)
                print(f"[step {steps:03d}] ddv2_plan_8x3=\n{plan_np}")

        obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        act_pose = _pose_xzyaw_from_info(info, "act") if isinstance(info, dict) else None
        exp_pose = _pose_xzyaw_from_info(info, "exp") if isinstance(info, dict) else None

        if act_pose is None and sim is not None:
            act_pose = _pose_xzyaw_from_matrix(getattr(sim, "start_ego", None))
        if act_pose is not None:
            act_traj.append(act_pose)
        if exp_pose is not None:
            exp_traj.append(exp_pose)

        if isinstance(info, dict):
            act_pos = info.get("act_pos", None)
            exp_pos = info.get("exp_pos", None)
            match_pos = info.get("ground_ref_pos", None)
            match_dist = info.get("ground_ref_dist_m", None)
            t_series_s.append(float(steps + 1) * float(step_dt_s))
            if act_pos is not None and exp_pos is not None:
                ap3 = np.asarray(act_pos, dtype=np.float32).reshape(-1)
                ep3 = np.asarray(exp_pos, dtype=np.float32).reshape(-1)
                if ap3.shape[0] >= 3 and ep3.shape[0] >= 3:
                    act_xyz_series.append((float(ap3[0]), float(ap3[1]), float(ap3[2])))
                    if sim is not None:
                        act_frame_idx_series.append(int(getattr(sim, "now_frame", steps + 1)))
                    else:
                        act_frame_idx_series.append(int(steps + 1) * int(step_frames))

                    if match_pos is not None:
                        mp3 = np.asarray(match_pos, dtype=np.float32).reshape(-1)
                        if mp3.shape[0] >= 3:
                            matched_xyz_series.append((float(mp3[0]), float(mp3[1]), float(mp3[2])))
                        else:
                            matched_xyz_series.append((np.nan, np.nan, np.nan))
                    else:
                        matched_xyz_series.append((np.nan, np.nan, np.nan))

                    if match_dist is not None:
                        matched_dist_series.append(float(match_dist))
                    else:
                        matched_dist_series.append(float("nan"))

                    act_y_series.append(float(ap3[1]))
                    exp_y_series.append(float(ep3[1]))
                else:
                    t_series_s.pop()
            else:
                t_series_s.pop()

        writer.append_data(_grid_frame(obs))
        frames += 1
        steps += 1

        if done:
            break
        if max_steps is None and steps >= 5000:
            print("warning: safety stop at 5000 steps")
            break

    writer.close()

    act_arr = np.asarray(act_traj, dtype=np.float32)
    exp_arr = np.asarray(exp_traj, dtype=np.float32)
    act_xyz_arr = np.asarray(act_xyz_series, dtype=np.float32)
    act_frame_idx_arr = np.asarray(act_frame_idx_series, dtype=np.int32)
    matched_xyz_arr = np.asarray(matched_xyz_series, dtype=np.float32)
    matched_dist_arr = np.asarray(matched_dist_series, dtype=np.float32)

    expert_full_xyz_arr = np.zeros((0, 3), dtype=np.float32)
    if sim is not None and hasattr(sim, "expert_world_all"):
        try:
            mats = getattr(sim, "expert_world_all")
            pts = [np.asarray(m[:3, 3], dtype=np.float32).reshape(3) for m in mats]
            if len(pts) > 0:
                expert_full_xyz_arr = np.asarray(pts, dtype=np.float32)
        except Exception:
            expert_full_xyz_arr = np.zeros((0, 3), dtype=np.float32)

    if act_arr.size > 0 or exp_arr.size > 0:
        _save_trajectory_plot(
            traj_plot_out,
            act_arr[:, :2] if act_arr.size > 0 else np.zeros((0, 2), dtype=np.float32),
            exp_arr[:, :2] if exp_arr.size > 0 else np.zeros((0, 2), dtype=np.float32),
        )
    if t_series_s and len(t_series_s) == len(act_y_series) == len(exp_y_series):
        _save_height_plot(height_plot_out, t_series_s, act_y_series, exp_y_series)
    if act_xyz_arr.size > 0:
        _save_ground_match_3d_plot(
            ground3d_plot_out,
            expert_full_xyz_arr,
            act_xyz_arr,
            matched_xyz_arr,
            matched_dist_arr,
            act_frame_idx_arr,
        )

    print(f"done={done} steps={steps} frames={frames}")
    print(f"video_saved={out_path}")
    print(f"trajectory_plot_saved={traj_plot_out}")
    if t_series_s and len(t_series_s) == len(act_y_series) == len(exp_y_series):
        print(f"height_plot_saved={height_plot_out}")
    if act_xyz_arr.size > 0:
        print(f"ground3d_plot_saved={ground3d_plot_out}")
    print(f"act_traj_shape={act_arr.shape}")
    print(f"exp_traj_shape={exp_arr.shape}")
    print("act_traj_points:")
    print(act_arr)
    print("exp_traj_points:")
    print(exp_arr)
    print("==== all done ====")


if __name__ == "__main__":
    main()
