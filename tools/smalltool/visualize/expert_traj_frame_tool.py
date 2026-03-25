#!/usr/bin/env python3
"""Extract and validate expert trajectory transforms across frames.

This tool helps verify matrix conversions between:
- world frame (ego_pose files)
- front-camera-start frame
- ego-start-local frame

It exports pose tables with xyz + euler angles, validates conversion identities,
and can compare a user-provided second trajectory in ego-local frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R

# Ensure local repo modules are importable even without external PYTHONPATH.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from reconsimulator.envs import nus_config as cfg


@dataclass
class PoseRow_xz_y:
    frame: int
    x: float
    y: float
    z: float
    yaw_xz_deg: float #simulator使用的yaw atan2(z, x)
    yaw_xz_rad_signed: float
    yaw_zyx_deg: float #标准欧拉角
    pitch_zyx_deg: float
    roll_zyx_deg: float


def _as_matrix(path: str) -> np.ndarray:
    '''
    读取4x4矩阵文本文件，返回numpy数组。
    ego_pose/xxx.txt
    cam2ego/0.txt
    '''
    
    arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"invalid 4x4 matrix in {path}: shape={arr.shape}")
    return arr


def _frame_from_path(path: str) -> int:
    #从文件名取frame
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    return int(stem)


def _yaw_xz_deg(T: np.ndarray) -> float:
    # Keep consistent with simulator internal heading extraction.
    yaw = float(np.arctan2(float(T[2, 0]), float(T[0, 0])))
    return float(np.rad2deg(yaw))


def _ypr_from_rot_zyx_deg(T: np.ndarray) -> tuple[float, float, float]:
    # zyx returns [yaw(z), pitch(y), roll(x)] in degrees.
    y, p, r = R.from_matrix(T[:3, :3]).as_euler("zyx", degrees=True)
    return float(y), float(p), float(r)


def _to_row_xz_y(frame: int, T: np.ndarray) -> PoseRow_xz_y:
    """把 SE3 矩阵转换成 PoseRow_xz_y，主要 yaw 用 XZ 平面"""
    y, p, r = _ypr_from_rot_zyx_deg(T)
    yaw_xz_rad = float(np.arctan2(T[2, 0], T[0, 0]))
    return PoseRow_xz_y(
        frame=int(frame),
        x=float(T[0, 3]),
        y=float(T[1, 3]),
        z=float(T[2, 3]),
        yaw_xz_deg=float(np.rad2deg(yaw_xz_rad)),
        yaw_xz_rad_signed=yaw_xz_rad,
        yaw_zyx_deg=float(y),
        pitch_zyx_deg=float(p),
        roll_zyx_deg=float(r),
    )
    

            
def _write_rows_csv_xz_y(path: str, rows: Iterable[PoseRow_xz_y]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame",
            "x",
            "y",
            "z",
            "yaw_xz_deg",
            "yaw_xz_rad_signed",
            "yaw_zyx_deg",
            "pitch_zyx_deg",
            "roll_zyx_deg",
        ])
        for r in rows:
            w.writerow([
                r.frame,
                f"{r.x:.9f}",
                f"{r.y:.9f}",
                f"{r.z:.9f}",
                f"{r.yaw_xz_deg:.9f}",
                f"{r.yaw_xz_rad_signed:.9f}",
                f"{r.yaw_zyx_deg:.9f}",
                f"{r.pitch_zyx_deg:.9f}",
                f"{r.roll_zyx_deg:.9f}",
            ])


#########################xy-z平面
@dataclass
class PoseRow_xy_z:
    frame: int
    x: float
    y: float
    z: float
    yaw_xy_deg: float          # XY 平面 yaw (绕 Z 轴)
    yaw_xy_rad_signed: float   # XY 平面 yaw 弧度
    yaw_zyx_deg: float         # 原来的 zyx yaw
    pitch_zyx_deg: float
    roll_zyx_deg: float

# --- 提取 XY 平面 yaw ---
def _yaw_xy_deg(T: np.ndarray) -> float:
    """提取 XY 平面 yaw (绕 Z 轴)"""
    yaw = float(np.arctan2(T[1, 0], T[0, 0]))  # 注意 XY 平面：arctan2(r21, r11)
    return float(np.rad2deg(yaw))

def _ypr_from_rot_zyx_deg(T: np.ndarray) -> tuple[float, float, float]:
    """ZYX 欧拉角, 保留原始"""
    y, p, r = R.from_matrix(T[:3, :3]).as_euler("zyx", degrees=True)
    return float(y), float(p), float(r)

def _to_row_xy_z(frame: int, T: np.ndarray) -> PoseRow_xy_z:
    """把 SE3 矩阵转换成 PoseRow，主要 yaw 用 XY 平面"""
    y, p, r = _ypr_from_rot_zyx_deg(T)
    yaw_xy_rad = float(np.arctan2(T[1, 0], T[0, 0]))
    return PoseRow_xy_z(
        frame=int(frame),
        x=float(T[0, 3]),
        y=float(T[1, 3]),
        z=float(T[2, 3]),
        yaw_xy_deg=float(np.rad2deg(yaw_xy_rad)),
        yaw_xy_rad_signed=yaw_xy_rad,
        yaw_zyx_deg=float(y),
        pitch_zyx_deg=float(p),
        roll_zyx_deg=float(r),
    )

def _write_rows_csv_xy_z(path: str, rows: Iterable[PoseRow_xy_z]) -> None:
    """写入 CSV, 使用 XY 平面 yaw"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame",
            "x",
            "y",
            "z",
            "yaw_xy_deg",
            "yaw_xy_rad_signed",
            "yaw_zyx_deg",
            "pitch_zyx_deg",
            "roll_zyx_deg",
        ])
        for r in rows:
            w.writerow([
                r.frame,
                f"{r.x:.9f}",
                f"{r.y:.9f}",
                f"{r.z:.9f}",
                f"{r.yaw_xy_deg:.9f}",
                f"{r.yaw_xy_rad_signed:.9f}",
                f"{r.yaw_zyx_deg:.9f}",
                f"{r.pitch_zyx_deg:.9f}",
                f"{r.roll_zyx_deg:.9f}",
            ])



def _load_second_traj_csv(path: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            frame = int(row["frame"])
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            T = np.eye(4, dtype=np.float64)
            T[0, 3] = x
            T[1, 3] = y
            T[2, 3] = z
            # Optional yaw columns for orientation check/export.
            yaw = None
            if "yaw_xz_rad_signed" in row and row["yaw_xz_rad_signed"] not in ("", None):
                yaw = float(row["yaw_xz_rad_signed"])
            elif "yaw_xz_rad" in row and row["yaw_xz_rad"] not in ("", None):
                yaw = float(row["yaw_xz_rad"])
            elif "yaw_xz_deg" in row and row["yaw_xz_deg"] not in ("", None):
                yaw = float(np.deg2rad(float(row["yaw_xz_deg"])))

            if yaw is not None:
                c, s = float(np.cos(yaw)), float(np.sin(yaw))
                T[:3, :3] = np.array(
                    [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64
                )
            out[frame] = T
    return out

#TODO: 这个是在自车坐标系下面么
def _rel_actions_from_pose_list(frames: list[int], Ts: list[np.ndarray]) -> list[dict[str, float]]:
    actions: list[dict[str, float]] = []
    for i in range(1, len(Ts)):
        rel = np.linalg.inv(Ts[i - 1]) @ Ts[i]
        dyaw = float(np.arctan2(float(rel[2, 0]), float(rel[0, 0])))
        actions.append(
            {
                "from_frame": int(frames[i - 1]),
                "to_frame": int(frames[i]),
                "dx": float(rel[0, 3]),
                "dz": float(rel[2, 3]),
                "dyaw_rad": dyaw,
            }
        )
    return actions


def main() -> None:
    ap = argparse.ArgumentParser(description="Expert trajectory frame conversion/validation tool")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--step-frames", type=int, default=1)
    ap.add_argument(
            "--out-dir",
            type=str,
            default="/root/clone/ReconDreamer-RL/outputs/visualize/trajTransition-scene{scene:03d}",
        )
    ap.add_argument(
        "--second-traj-csv",
        type=str,
        default=None,
        help="Optional second trajectory in ego-local frame. Required columns: frame,x,y,z",
    )
    args = ap.parse_args()
    scene = int(args.scene)
    start_frame = int(args.start_frame)
    step_frames = max(1, int(args.step_frames))
    if args.out_dir:
        out_dir = args.out_dir.format(scene=scene)
    else:
        out_dir = os.path.join(
            "outputs",
            "smalltool",
            f"scene{scene:03d}_frame_check",
        )
    os.makedirs(out_dir, exist_ok=True)

    scene_dir = os.path.join(cfg.BASE_DATA_DIR, f"{scene:03d}")
    ego_pose_dir = os.path.join(scene_dir, "ego_pose")
    cam2ego0_path = os.path.join(scene_dir, "cam2ego", "0.txt")

    if not os.path.isfile(cam2ego0_path):
        raise FileNotFoundError(f"missing file: {cam2ego0_path}")
    if not os.path.isdir(ego_pose_dir):
        raise FileNotFoundError(f"missing dir: {ego_pose_dir}")

    all_pose_files = [
        os.path.join(ego_pose_dir, n)
        for n in os.listdir(ego_pose_dir)
        if n.endswith(".txt")
    ]
    if not all_pose_files:
        raise RuntimeError(f"no ego pose files under {ego_pose_dir}")

    all_frames = sorted(_frame_from_path(p) for p in all_pose_files)
    frames = [f for f in all_frames if f >= start_frame and ((f - start_frame) % step_frames == 0)]
    if not frames:
        raise RuntimeError("no frames selected by --start-frame/--step-frames")

#这一部分是可以直接从Info抓取。每一个场景下自车在世界坐标系下面的pose；用于计算起始帧 前摄像头在world下面的位姿
    ego0_world = _as_matrix(os.path.join(ego_pose_dir, f"{start_frame:03d}.txt"))
    cam2ego0 = _as_matrix(cam2ego0_path)
    camera_front_start = ego0_world @ cam2ego0

    T_front_list: list[np.ndarray] = []
    T_ego_local_list: list[np.ndarray] = []
    T_ego_local_from_front_list: list[np.ndarray] = []

    for f in frames:
        #每一个场景下自车在世界坐标系下面的pose。可以直接从Info获得  ego 2 world
        T_ego_world = _as_matrix(os.path.join(ego_pose_dir, f"{f:03d}.txt"))
        
        #起始帧前摄像头坐标系下的位姿  world 2 front caera0 @ ego 2 world   == ego 2 front camera0
        T_front = np.linalg.inv(camera_front_start) @ T_ego_world
        
        #下面是从世界坐标系将ego位姿转化成自车坐标系下局部位姿的两条路径（一条是借助世界坐标下的位姿作为中转；一条是借助前视摄像头）
        #自车局部坐标系(ego0）下的位姿  world 2 ego0 @ego 2 world   == ego 2 ego0
        T_ego_local = np.linalg.inv(ego0_world) @ T_ego_world
        # 自车在局部坐标系下的位姿=前视摄像头（cam0)相对自车起始帧ego0的坐标转化@自车在起始帧前视摄像头下的位姿                     cam 2 ego0  @ ego 2 front camera0 ==ego 2 ego0? 对 两者相等
        T_ego_local_from_front = cam2ego0 @ T_front

        T_front_list.append(T_front)
        T_ego_local_list.append(T_ego_local)
        T_ego_local_from_front_list.append(T_ego_local_from_front)

    # Identity check: cam2ego0 @ T_front == T_ego_local
    trans_err = [
        float(np.linalg.norm((a[:3, 3] - b[:3, 3]), ord=2))
        for a, b in zip(T_ego_local_list, T_ego_local_from_front_list)
    ]
    rot_err = [
        float(np.linalg.norm((a[:3, :3] - b[:3, :3]), ord="fro"))
        for a, b in zip(T_ego_local_list, T_ego_local_from_front_list)
    ]

    front_rows = [_to_row_xz_y(f, T) for f, T in zip(frames, T_front_list)]
    ego_local_rows = [_to_row_xy_z(f, T) for f, T in zip(frames, T_ego_local_list)]
    world_rows = [_to_row_xy_z(f, _as_matrix(os.path.join(ego_pose_dir, f"{f:03d}.txt"))) for f in frames]
    
    print("===== DEBUG T_front_list first 5 =====")
    for i, (f, T) in enumerate(zip(frames, T_front_list)):
        if i >= 5:
            break
        print(f"[frame {f}] T_front =\n{T}")

    print("===== DEBUG T_ego_local_list first 5 =====")
    for i, (f, T) in enumerate(zip(frames, T_ego_local_list)):
        if i >= 5:
            break
        print(f"[frame {f}] T_ego_local =\n{T}")
    
    

    _write_rows_csv_xz_y(os.path.join(out_dir, "expert_front_camera_frame.csv"), front_rows) #自车在起始帧前视摄像头下的位姿
    _write_rows_csv_xy_z(os.path.join(out_dir, "expert_ego_local_frame.csv"), ego_local_rows) #自车在起始帧自车坐标系也就是ego0下面的位姿
    _write_rows_csv_xy_z(os.path.join(out_dir, "expert_world_frame.csv"), world_rows)#自车在世界坐标系下的位姿

    actions = _rel_actions_from_pose_list(frames, T_front_list)
    with open(os.path.join(out_dir, "expert_front_rel_actions.json"), "w", encoding="utf-8") as f:
        json.dump(actions, f, indent=2)

    summary = {
        "scene": scene,
        "start_frame": start_frame,
        "step_frames": step_frames,
        "num_frames": len(frames),
        "conversion_identity": "T_ego_local == cam2ego0 @ T_front",
        "translation_err_l2_max": float(np.max(trans_err)),
        "translation_err_l2_mean": float(np.mean(trans_err)),
        "rotation_err_fro_max": float(np.max(rot_err)),
        "rotation_err_fro_mean": float(np.mean(rot_err)),
        "files": {
            "expert_front_camera_frame_csv": os.path.join(out_dir, "expert_front_camera_frame.csv"),
            "expert_ego_local_frame_csv": os.path.join(out_dir, "expert_ego_local_frame.csv"),
            "expert_world_frame_csv": os.path.join(out_dir, "expert_world_frame.csv"),
            "expert_front_rel_actions_json": os.path.join(out_dir, "expert_front_rel_actions.json"),
        },
    }

    if args.second_traj_csv:
        second = _load_second_traj_csv(str(args.second_traj_csv))
        common_frames = [f for f in frames if f in second]
        if not common_frames:
            raise RuntimeError("second traj has no frame intersection with selected expert frames")

        exp_map = {f: T for f, T in zip(frames, T_ego_local_list)}
        trans2 = []
        rot2 = []
        for f in common_frames:
            A = exp_map[f]
            B = second[f]
            trans2.append(float(np.linalg.norm((A[:3, 3] - B[:3, 3]), ord=2)))
            rot2.append(float(np.linalg.norm((A[:3, :3] - B[:3, :3]), ord="fro")))

        second_front_rows = []
        second_front_Ts = []
        for f in common_frames:
            T_front_from_second = np.linalg.inv(cam2ego0) @ second[f]
            second_front_rows.append(_to_row_xz_y(f, T_front_from_second))
            second_front_Ts.append(T_front_from_second)
        _write_rows_csv_xz_y(os.path.join(out_dir, "second_traj_mapped_to_front_frame.csv"), second_front_rows)

        second_actions = _rel_actions_from_pose_list(common_frames, second_front_Ts)
        second_actions_path = os.path.join(out_dir, "second_front_rel_actions.json")
        with open(second_actions_path, "w", encoding="utf-8") as f:
            json.dump(second_actions, f, indent=2)

        summary["second_traj_compare"] = {
            "input_csv": str(args.second_traj_csv),
            "common_frames": len(common_frames),
            "translation_err_l2_max": float(np.max(trans2)),
            "translation_err_l2_mean": float(np.mean(trans2)),
            "rotation_err_fro_max": float(np.max(rot2)),
            "rotation_err_fro_mean": float(np.mean(rot2)),
            "second_traj_mapped_to_front_frame_csv": os.path.join(out_dir, "second_traj_mapped_to_front_frame.csv"),
            "second_front_rel_actions_json": second_actions_path,
        }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("==== expert frame tool done ====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

'''
python tools/smalltool/expert_traj_frame_tool.py --scene 33

python tools/smalltool/expert_traj_frame_tool.py \
  --scene 33 \
  --start-frame 0 \
  --step-frames 1 \
  --second-traj-csv your_second_traj_ego_local.csv
'''
