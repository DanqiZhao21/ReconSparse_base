#!/usr/bin/env python3
"""Generate two videos for one scene:
1) expert rollout video (env expert action)
2) csv-driven rollout video (relative actions derived from ego-local CSV)

The CSV is expected to be ego0-local absolute poses (like expert_ego_local_frame.csv).
We convert consecutive absolute poses to relative SE(2) actions in x-z plane and
feed them to ReconSimulator through RLReconEnv with action format (dx, dz, dyaw, 2).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import imageio
import numpy as np
import pandas as pd
import torch


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _lazy_import_runtime() -> Any:
    try:
        from framework.env_wrapper import RLReconEnv  # type: ignore

        return RLReconEnv
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "Missing runtime dependency for rollout. "
            f"Import failed on module: {missing}. "
            "Activate project env and retry."
        ) from e


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _grid_frame(observation: dict[str, np.ndarray]) -> np.ndarray:
    keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
    imgs = [observation[k] for k in keys]
    row1 = np.concatenate(imgs[:3], axis=1)
    row2 = np.concatenate(imgs[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)


def _yaw_from_R_xy(Rm: np.ndarray) -> float:
    return float(np.arctan2(float(Rm[1, 0]), float(Rm[0, 0])))


def _pose_matrix_from_xyyaw(x: float, y: float, yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [[c,-s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    return T


def _infer_step_frames_from_csv(df: pd.DataFrame, fallback: int) -> int:
    '''
    从 CSV 的 frame 列推断每一步动作的帧间隔
    '''
    if "frame" not in df.columns:
        return int(fallback)
    frames = np.asarray(df["frame"].values, dtype=np.int64)
    if frames.size < 2:
        return int(fallback)
    diffs = np.diff(frames)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return int(fallback)
    return int(np.median(diffs))


def _get_signed_yaw_rad_series(df: pd.DataFrame) -> np.ndarray:
    if "yaw_xz_rad_signed" in df.columns:
        return np.asarray(df["yaw_xz_rad_signed"].values, dtype=np.float64)
    if "yaw_xy_rad_signed" in df.columns:
        return np.asarray(df["yaw_xy_rad_signed"].values, dtype=np.float64)
    if "yaw_xz_rad" in df.columns:
        vals = np.asarray(df["yaw_xz_rad"].values, dtype=np.float64)
        return np.arctan2(np.sin(vals), np.cos(vals))
    if "yaw_xz_deg" in df.columns:
        vals = np.deg2rad(np.asarray(df["yaw_xz_deg"].values, dtype=np.float64))
        return np.arctan2(np.sin(vals), np.cos(vals))
    raise RuntimeError("CSV missing yaw column: need yaw_xz_deg or yaw_xz_rad(_signed)")

# def _build_relative_actions_from_ego_local_csv(
#     csv_path: str
# ) -> tuple[list[tuple[float, float, float, int]], int, int]:
#     """
#     从 ego-local CSV 构建连续动作序列 (dx, dy, dyaw, 2)，完全与 step 对齐。
#     CSV 必须包含 x, y 列，可选 frame 列。
#     """
#     df = pd.read_csv(csv_path)
#     required = {"x", "y"}
#     if not required.issubset(df.columns):
#         raise RuntimeError(f"CSV missing columns {sorted(required)}: {csv_path}")

#     # 按 frame 排序
#     if "frame" in df.columns:
#         df = df.sort_values("frame", ascending=True).reset_index(drop=True)

#     xs = np.asarray(df["x"].values, dtype=np.float64)
#     ys = np.asarray(df["y"].values, dtype=np.float64)
#     yaw_rad = _get_signed_yaw_rad_series(df)

#     actions: list[tuple[float, float, float, int]] = []
#     for i in range(1, len(xs)):
#         dx = xs[i] - xs[i - 1]
#         dy = ys[i] - ys[i - 1]
#         dyaw = np.arctan2(np.sin(yaw_rad[i] - yaw_rad[i - 1]),
#                           np.cos(yaw_rad[i] - yaw_rad[i - 1]))
#         actions.append((dx, dy, dyaw, 2))  # flag=2 连续动作模式

#     if len(xs) < 2:
#         raise RuntimeError(f"CSV rows not enough for rollout: {csv_path}")

#     start_frame = int(df["frame"].iloc[0]) if "frame" in df.columns else 0
#     step_frames = _infer_step_frames_from_csv(df, fallback=1)

#     return actions, start_frame, step_frames

def _build_relative_actions_from_ego_local_csv(csv_path: str) -> tuple[list[tuple[float, float, float, int]], int, int]:
    df = pd.read_csv(csv_path)
    required = {"x", "y"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"CSV missing columns {sorted(required)}: {csv_path}")

    if "frame" in df.columns:
        df = df.sort_values("frame", ascending=True).reset_index(drop=True)

    yaw_rad = _get_signed_yaw_rad_series(df)
    xs = np.asarray(df["x"].values, dtype=np.float64)
    ys = np.asarray(df["y"].values, dtype=np.float64)

    Ts: list[np.ndarray] = []
    for x, y, yaw in zip(xs, ys, yaw_rad):
        Ts.append(_pose_matrix_from_xyyaw(float(x), float(y), float(yaw)))

    actions: list[tuple[float, float, float, int]] = []
    for i in range(1, len(Ts)):
        rel = np.linalg.inv(Ts[i - 1]) @ Ts[i]
        dx = float(rel[0, 3])
        dy = float(rel[1, 3])
        dyaw = _yaw_from_R_xy(rel[:3, :3])
        actions.append((dx, dy, dyaw, 2))

    if len(Ts) < 2:
        raise RuntimeError(f"CSV rows not enough for rollout: {csv_path}")

    start_frame = int(df["frame"].iloc[0]) if "frame" in df.columns else 0
    step_frames = _infer_step_frames_from_csv(df, fallback=1)
    return actions, start_frame, step_frames


def _run_rollout_video(
    *,
    env: Any,
    obs0: dict[str, np.ndarray],
    out_path: str,
    fps: float,
    actions: list[tuple[float, float, float, int]] | None,
    expert: bool,
    max_steps: int | None,
) -> tuple[int, int, bool]:
    _ensure_parent(out_path)

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

    steps = 0
    frames = 0
    done = False

    obs = obs0
    writer.append_data(_grid_frame(obs))
    frames += 1

    if expert:
        # Expert branch runs same number of steps as csv actions for fair comparison.
        target_steps = len(actions) if actions is not None else 0
        if max_steps is not None:
            target_steps = min(target_steps, int(max_steps))
        for _ in range(target_steps):
            obs, _reward, terminated, truncated, _info = env.step((0, 0, 1))
            done = bool(terminated or truncated)
            writer.append_data(_grid_frame(obs))
            frames += 1
            steps += 1
            if done:
                break
    else:
        if actions is None:
            raise RuntimeError("Non-expert rollout requires action list")
        action_seq = actions if max_steps is None else actions[: max(0, int(max_steps))]
        for a in action_seq:
            print("🐷 roll out step")
            obs, _reward, terminated, truncated, _info = env.step(a)
            done = bool(terminated or truncated)
            writer.append_data(_grid_frame(obs))
            frames += 1
            steps += 1
            if done:
                break

    writer.close()
    return steps, frames, done


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate expert and csv-driven rollout videos for one scene")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument(
        "--csv",
        type=str,
        default=os.path.join(_REPO_ROOT, "outputs", "visualize", "trajTransition-scene{scene:03d}", "expert_ego_local_frame.csv"),
        help="Ego-local absolute trajectory CSV (x,z,yaw columns).",
    )
    ap.add_argument("--out-dir", type=str, default=os.path.join(_REPO_ROOT, "outputs", "visualize", "trajTransition-scene{scene:03d}"))
    ap.add_argument("--start-frame", type=int, default=None, help="Override start frame (default uses CSV first frame)")
    ap.add_argument("--step-frames", type=int, default=None, help="Override step_frames (default inferred from CSV frame column)")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--frame-dt-s", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--expert-high", dest="expert_high", action="store_true", default=True)
    ap.add_argument("--no-expert-high", dest="expert_high", action="store_false")
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--render-w", type=int, default=None)
    ap.add_argument("--render-h", type=int, default=None)
    # args = ap.parse_args()

    # RLReconEnv = _lazy_import_runtime()

    # csv_path = str(args.csv)
    # if not os.path.isfile(csv_path):
    #     raise FileNotFoundError(f"CSV not found: {csv_path}")

    # actions, csv_start_frame, csv_step_frames = _build_relative_actions_from_ego_local_csv(csv_path)
    
    args = ap.parse_args()
    scene = int(args.scene)
    RLReconEnv = _lazy_import_runtime()
    csv_path = str(args.csv).format(scene=scene)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    print(f"Loading CSV from {csv_path}")
    # Loading CSV from /root/clone/ReconDreamer-RL/outputs/visualize/trajTransition-scene099/expert_ego_local_frame.csv
    actions, csv_start_frame, csv_step_frames = _build_relative_actions_from_ego_local_csv(csv_path)

    scene = int(args.scene)
    start_frame = int(args.start_frame) if args.start_frame is not None else int(csv_start_frame)
    step_frames = int(args.step_frames) if args.step_frames is not None else int(csv_step_frames)
    if step_frames <= 0:
        raise RuntimeError("step_frames must be > 0")

    step_dt_s = float(step_frames) * float(args.frame_dt_s)
    if step_dt_s <= 0:
        raise RuntimeError("invalid dt from step_frames/frame-dt-s")
    fps = float(args.fps) if args.fps is not None else (1.0 / step_dt_s)
    if fps <= 0:
        raise RuntimeError("fps must be > 0")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    expert_video = os.path.join(out_dir, f"scene{scene:03d}_{ts}_expert.mp4")
    csv_video = os.path.join(out_dir, f"scene{scene:03d}_{ts}_csv_rollout.mp4")

    print("==== replay_expert_vs_csv_video ====")
    print(f"scene={scene}")
    print(f"csv={csv_path}")
    print(f"start_frame={start_frame} step_frames={step_frames}")
    print(f"actions_from_csv={len(actions)}")
    print(f"max_steps={args.max_steps}")
    print(f"expert_video={expert_video}")
    print(f"csv_video={csv_video}")

    cuda = int(args.cuda)
    _device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    # Run expert video.
    env_expert = RLReconEnv(
        cuda=cuda,
        scene=scene,
        reward_cfg={},
        debug=bool(args.debug),
        render_w=(int(args.render_w) if args.render_w is not None else None),
        render_h=(int(args.render_h) if args.render_h is not None else None),
    )
    obs0, _info0 = env_expert.reset(scene=scene, start_frame=start_frame, step_frames=step_frames)
    sim0 = getattr(env_expert, "env", None)
    if sim0 is not None:
        setattr(sim0, "use_expert_height", bool(args.expert_high))
    exp_steps, exp_frames, exp_done = _run_rollout_video(
        env=env_expert,
        obs0=obs0,
        out_path=expert_video,
        fps=fps,
        actions=actions,
        expert=True,
        max_steps=(int(args.max_steps) if args.max_steps is not None else None),
    )

    # Run CSV-driven video.
    env_csv = RLReconEnv(
        cuda=cuda,
        scene=scene,
        reward_cfg={},
        debug=bool(args.debug),
        render_w=(int(args.render_w) if args.render_w is not None else None),
        render_h=(int(args.render_h) if args.render_h is not None else None),
    )
    obs1, _info1 = env_csv.reset(scene=scene, start_frame=start_frame, step_frames=step_frames)
    sim1 = getattr(env_csv, "env", None)
    if sim1 is not None:
        setattr(sim1, "use_expert_height", bool(args.expert_high))
    csv_steps, csv_frames, csv_done = _run_rollout_video(
        env=env_csv,
        obs0=obs1,
        out_path=csv_video,
        fps=fps,
        actions=actions,
        expert=False,
        max_steps=(int(args.max_steps) if args.max_steps is not None else None),
    )

    print(f"expert_done={exp_done} expert_steps={exp_steps} expert_frames={exp_frames}")
    print(f"csv_done={csv_done} csv_steps={csv_steps} csv_frames={csv_frames}")
    print(f"video_saved={expert_video}")
    print(f"video_saved={csv_video}")
    print("==== all done ====")


if __name__ == "__main__":
    main()
