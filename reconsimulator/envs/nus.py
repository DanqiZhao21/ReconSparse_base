import os
import csv
import copy
import math
import json
import pickle
import time
import torch
import numpy as np
import gymnasium as gym
import bisect
from typing import Any
# NOTE: Import helpers from the concrete module to avoid a circular import:
# - framework.env_wrapper.__init__ imports RLReconEnv -> imports ReconSimulator (this file)
# - importing from framework.env_wrapper here would re-enter __init__ while it's initializing
from framework.env_wrapper.tool import get_splat, get_sky_view, move_to_device, slerp
from framework.utils.hugsim_execution import DEFAULT_HUGSIM_REPO, load_hugsim_runtime, resolve_wheelbase, solve_hugsim_execution
from framework.utils.tracker_execution import TrackerExecutionResult, build_execution_result
from reconsimulator.envs import nus_config as cfg
from scipy.spatial.transform import Slerp, Rotation as R
from scipy.spatial.distance import cdist

# Coordinate conventions used in this simulator:
# - Frame: front-start local frame (origin at the selected start frame's front camera).
# - Planar motion is on x-y; yaw is rotation about +z.
# - Yaw extraction uses atan2(R[1,0], R[0,0]).

# NOTE: keep on CPU by default; move to the env's target device at runtime.
TRANSFORM_MATRIX = torch.eye(4, dtype=torch.float32)


class ReconSimulator(gym.Env):
    def __init__(self, cuda=0, scene=0, debug=True, *, render_w: int = 800, render_h: int = 450):
        self.device = f"cuda:{cuda}"
        self.debug = debug
        self.scene = scene
        self.w, self.h = int(render_w), int(render_h)
        self._transform_matrix = TRANSFORM_MATRIX.to(self.device)

        # Observation space: 6 camera RGB views + ego status vectors.
        obs_dict = {
            name: gym.spaces.Box(low=0, high=255, shape=(self.h, self.w, 3), dtype=np.uint8)
            for name in ["front", "front_left", "front_right", "back_left", "back_right", "back"]
        }
        obs_dict.update({
            "ego_velocity": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "ego_acceleration": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "driving_command": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            "ego_status": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
        })
        self.observation_space = gym.spaces.Dict(obs_dict)

        # Action space: discrete anchor indices
        self.action_space = gym.spaces.MultiDiscrete([61, 61])

        # Load trainer
        self.trainer, self.num_timesteps = get_splat(self.device, self.scene)
        self.trainer.eval()

        # Frame control
        #NOTE 更新一下环境的步长
        self.step_frames = 5
        self.final_frame = 186
        self.now_frame = 0

        # Load all data
        self._load_camera_and_images()
        self._load_ego_and_cam_matrices()
        self._load_expert_ego_frames()
        self._load_plan_anchors()
        self._load_token_mappings()

        self.all_camera_now = []
        self.get_all_point_for_expert()
        
        #FIXME:查看在哪里用到了  ndarray | None = None
        self._status_prev_vel_xy: np.ndarray | None = None
        self._status_vel_xy = np.zeros((2,), dtype=np.float32)
        self._status_acc_xy = np.zeros((2,), dtype=np.float32)
        self._status_cmd = np.zeros((4,), dtype=np.float32)
        self._status_steering_angle = 0.0
        self._status_steering_rate = 0.0
        self._last_obs_timing: dict[str, float] = {}
        self._last_info_timing: dict[str, float] = {}
        self._tracked_first_step_xyyaw = np.zeros((3,), dtype=np.float64)
        self._executed_first_step_xyyaw = np.zeros((3,), dtype=np.float64)
        self._tracked_rollout_local_xyyaw = np.zeros((0, 3), dtype=np.float64)
        self._last_tracker_execution: TrackerExecutionResult | None = None
        self._external_execution_override: TrackerExecutionResult | None = None
        self._external_plan_local_xyyaw: np.ndarray | None = None
        self._tracker_debug_pending = False
        self._tracker_debug_cleaned = False
        self._tracker_debug_image_index = 0
        self._tracker_debug_last_path: str | None = None
        self._tracker_debug_csv_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Optional: nuScenes DB + can_bus access (lazy-init).
        self._nusc = None
        self._nusc_can_bus = None
        self._nusc_can_bus_cache: dict[str, dict[str, object]] = {}
        self._nusc_sample_by_token: dict[str, dict[str, Any]] = {}
        self._nusc_scene_name_by_token: dict[str, str] = {}
        self._nusc_meta_loaded: bool = False
        self._hugsim_repo = DEFAULT_HUGSIM_REPO
        self._hugsim_solve_sequence_fn = None
        self._hugsim_build_solver_fn = None
        self._hugsim_load_wheelbase_fn = None
        self._hugsim_wheelbase: float | None = None

        # Height alignment switch: when enabled, snap ego y to nearest expert y each step.
        self.use_expert_height = False

    def _scene_ego_pose_dir(self) -> str:
        return os.path.join(cfg.BASE_DATA_DIR, f"{int(self.scene):03d}", "ego_pose")

    def _frame_to_token(self, frame_idx: int) -> str | None:
        fidx = int(frame_idx)
        try:
            tok = self.frame2token.get(fidx, None)
            if tok is not None:
                return str(tok)
        except Exception:
            pass
        try:
            tok = self.frame2token.get(str(fidx), None)
            if tok is not None:
                return str(tok)
        except Exception:
            pass
        return None

    def _ensure_nusc_meta_loaded(self) -> None:
        if bool(self._nusc_meta_loaded):
            return
        self._nusc_sample_by_token = {}
        self._nusc_scene_name_by_token = {}

        sample_path = os.path.join(cfg.NUSCENES_DATA_ROOT, cfg.NUSCENES_VERSION, "sample.json")
        scene_path = os.path.join(cfg.NUSCENES_DATA_ROOT, cfg.NUSCENES_VERSION, "scene.json")

        try:
            if os.path.isfile(scene_path):
                with open(scene_path, "r", encoding="utf-8") as f:
                    scene_rows = json.load(f)
                if isinstance(scene_rows, list):
                    for row in scene_rows:
                        if isinstance(row, dict):
                            tok = row.get("token", None)
                            name = row.get("name", None)
                            if tok is not None and name is not None:
                                self._nusc_scene_name_by_token[str(tok)] = str(name)
        except Exception:
            self._nusc_scene_name_by_token = {}

        try:
            if os.path.isfile(sample_path):
                with open(sample_path, "r", encoding="utf-8") as f:
                    sample_rows = json.load(f)
                if isinstance(sample_rows, list):
                    for row in sample_rows:
                        if isinstance(row, dict):
                            tok = row.get("token", None)
                            if tok is not None:
                                self._nusc_sample_by_token[str(tok)] = row
        except Exception:
            self._nusc_sample_by_token = {}

        self._nusc_meta_loaded = True

    def _scene_name_and_timestamp_from_frame(self, frame_idx: int) -> tuple[str | None, int | None, str | None]:
        token = self._frame_to_token(int(frame_idx))
        if token is None:
            return None, None, None

        self._ensure_nusc_meta_loaded()
        row = self._nusc_sample_by_token.get(str(token), None)
        if not isinstance(row, dict):
            return None, None, str(token)

        scene_tok = row.get("scene_token", None)
        ts = row.get("timestamp", None)
        scene_name = self._nusc_scene_name_by_token.get(str(scene_tok), None) if scene_tok is not None else None
        ts_us = None
        try:
            if ts is not None:
                ts_us = int(ts)
        except Exception:
            ts_us = None
        return scene_name, ts_us, str(token)

    def _load_can_bus_messages(self, scene_name: str, msg_key: str) -> list[dict[str, Any]]:
        cache_key = f"{scene_name}:{msg_key}"
        cached = self._nusc_can_bus_cache.get(cache_key, None)
        if isinstance(cached, dict) and isinstance(cached.get("rows", None), list):
            return cached.get("rows", [])  # type: ignore[return-value]

        fp = os.path.join(cfg.NUSCENES_DATA_ROOT, "can_bus", f"{scene_name}_{msg_key}.json")
        rows: list[dict[str, Any]] = []
        try:
            if os.path.isfile(fp):
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    rows = [x for x in data if isinstance(x, dict)]
        except Exception:
            rows = []

        self._nusc_can_bus_cache[cache_key] = {"rows": rows}
        return rows

    @staticmethod
    def _locate_message_index(utimes: list[int], target_utime: int) -> int:
        if len(utimes) <= 0:
            return -1
        i = int(np.searchsorted(np.asarray(utimes, dtype=np.int64), int(target_utime)))
        if i >= len(utimes):
            i = len(utimes) - 1
        if i > 0:
            prev_i = i - 1
            if abs(int(target_utime) - int(utimes[prev_i])) <= abs(int(utimes[i]) - int(target_utime)):
                i = prev_i
        return int(i)

    def _driving_command_from_expert_traj(self, start_frame: int, horizon_s: float = 3.0) -> np.ndarray:
        # Command mapping requested by user:
        # left  -> [1,0,0,0] when y > 2
        # straight -> [0,1,0,0] when -2 <= y <= 2
        # right -> [0,0,1,0] when y < -2
        cmd_left = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cmd_straight = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        cmd_right = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        pose_dir = self._scene_ego_pose_dir()
        try:
            frames = self._available_pose_frames()
            if len(frames) <= 0:
                return cmd_straight
            sidx = int(start_frame)
            if sidx not in frames:
                pos0 = bisect.bisect_left(frames, sidx)
                if pos0 >= len(frames):
                    pos0 = len(frames) - 1
                sidx = int(frames[pos0])

            target_frame = int(round(float(sidx) + float(horizon_s) * 10.0))
            pos_t = bisect.bisect_left(frames, target_frame)
            if pos_t >= len(frames):
                pos_t = len(frames) - 1
            tidx = int(frames[pos_t])

            start_fp = os.path.join(pose_dir, f"{int(sidx):03d}.txt")
            fut_fp = os.path.join(pose_dir, f"{int(tidx):03d}.txt")
            if (not os.path.isfile(start_fp)) or (not os.path.isfile(fut_fp)):
                return cmd_straight

            start_world = np.asarray(np.loadtxt(start_fp), dtype=np.float64)
            fut_world = np.asarray(np.loadtxt(fut_fp), dtype=np.float64)
            rel = np.linalg.inv(start_world) @ fut_world
            y_off = float(rel[1, 3])
            if y_off > 2.0:
                return cmd_left
            if y_off < -2.0:
                return cmd_right
            return cmd_straight
        except Exception:
            return cmd_straight

    def _available_pose_frames(self) -> list[int]:
        pose_dir = self._scene_ego_pose_dir()
        if not os.path.isdir(pose_dir):
            return []
        out: list[int] = []
        try:
            for n in os.listdir(pose_dir):
                if not n.endswith(".txt"):
                    continue
                try:
                    out.append(int(os.path.splitext(n)[0]))
                except Exception:
                    continue
        except Exception:
            return []
        out.sort()
        return out

    def _load_world_xy(self, frame_idx: int) -> np.ndarray | None:
        try:
            fp = os.path.join(self._scene_ego_pose_dir(), f"{int(frame_idx):03d}.txt")
            if not os.path.isfile(fp):
                return None
            T = np.asarray(np.loadtxt(fp), dtype=np.float64)
            return np.asarray([float(T[0, 3]), float(T[1, 3])], dtype=np.float32)
        except Exception:
            return None

    def _status_from_dataset(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fidx = int(frame_idx)
        cmd = self._driving_command_from_expert_traj(start_frame=fidx, horizon_s=3.0)
        vel = np.zeros((2,), dtype=np.float32)
        acc = np.zeros((2,), dtype=np.float32)

        scene_name, ts_us, _token = self._scene_name_and_timestamp_from_frame(fidx)
        if scene_name is None or ts_us is None:
            return vel, acc, cmd

        pose_msgs = self._load_can_bus_messages(scene_name, "pose")
        if len(pose_msgs) <= 0:
            return vel, acc, cmd

        utimes: list[int] = []
        for m in pose_msgs:
            try:
                utimes.append(int(m.get("utime", 0)))
            except Exception:
                utimes.append(0)

        idx = self._locate_message_index(utimes, int(ts_us))
        if idx < 0 or idx >= len(pose_msgs):
            return vel, acc, cmd

        msg = pose_msgs[idx]
        try:
            vel_arr = np.asarray(msg.get("vel", [0.0, 0.0]), dtype=np.float32).reshape(-1)
            if vel_arr.shape[0] >= 2:
                vel = vel_arr[:2].astype(np.float32)
        except Exception:
            pass
        try:
            acc_arr = np.asarray(msg.get("accel", [0.0, 0.0]), dtype=np.float32).reshape(-1)
            if acc_arr.shape[0] >= 2:
                acc = acc_arr[:2].astype(np.float32)
        except Exception:
            pass
        return vel, acc, cmd
    def _repo_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _default_sparsedrive_repo(self) -> str | None:
        preferred = os.path.join(self._repo_root(), "egoADs", "SparseDriveV2")
        if os.path.isdir(preferred):
            return preferred
        fallback = os.path.join(self._repo_root(), "SparseDriveV2")
        if os.path.isdir(fallback):
            return fallback
        return None

    def _ensure_hugsim_runtime(self) -> bool:
        if self._hugsim_solve_sequence_fn is not None and self._hugsim_build_solver_fn is not None:
            return True
        try:
            solve_sequence_fn, build_solver_fn, load_wheelbase_fn = load_hugsim_runtime(str(self._hugsim_repo))
        except Exception:
            self._hugsim_solve_sequence_fn = None
            self._hugsim_build_solver_fn = None
            self._hugsim_load_wheelbase_fn = None
            return False

        self._hugsim_solve_sequence_fn = solve_sequence_fn
        self._hugsim_build_solver_fn = build_solver_fn
        self._hugsim_load_wheelbase_fn = load_wheelbase_fn
        if self._hugsim_wheelbase is None:
            self._hugsim_wheelbase = resolve_wheelbase(
                sparse_repo_path=self._default_sparsedrive_repo(),
                explicit_wheelbase=None,
                load_wheelbase_fn=load_wheelbase_fn,
            )
        return True

    @staticmethod
    def _yaw_from_pose_xy(T: np.ndarray) -> float:
        return float(math.atan2(float(T[1, 0]), float(T[0, 0])))
    
    @staticmethod
    def _yaw_from_pose_xz(T: np.ndarray) -> float:
        return float(math.atan2(float(T[2, 0]), float(T[0, 0])))

    @staticmethod
    def _pose_from_local_xyyaw(x: float, y: float, yaw: float) -> np.ndarray:
        c = float(math.cos(float(yaw)))
        s = float(math.sin(float(yaw)))
        tpt = np.eye(4, dtype=np.float64)
        tpt[:3, :3] = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        tpt[0, 3] = float(x)
        tpt[1, 3] = float(y)
        return tpt

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(math.atan2(math.sin(float(angle)), math.cos(float(angle))))

    def _store_tracker_execution_result(self, execution: TrackerExecutionResult) -> None:
        self._tracked_first_step_xyyaw = np.asarray(execution.tracked_first_local_xyyaw, dtype=np.float64).reshape(3)
        self._executed_first_step_xyyaw = np.asarray(execution.executed_local_xyyaw, dtype=np.float64).reshape(3)
        self._tracked_rollout_local_xyyaw = np.asarray(execution.tracked_rollout_local_xyyaw, dtype=np.float64)
        self._last_tracker_execution = execution

    def _tracker_debug_output_dir(self) -> str:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(
            repo_root,
            "outputs",
            "visualize",
            f"trajTransition-scene{int(self.scene):03d}",
            "trackerdebug",
        )

    def _tracker_debug_csv_path(self) -> str:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(
            repo_root,
            "outputs",
            "visualize",
            f"trajTransition-scene{int(self.scene):03d}",
            "expert_ego_local_frame.csv",
        )

    def _load_tracker_debug_csv_poses(self) -> tuple[np.ndarray, np.ndarray] | None:
        cached = self._tracker_debug_csv_cache.get(int(self.scene), None)
        if cached is not None:
            return cached

        csv_path = self._tracker_debug_csv_path()
        if not os.path.isfile(csv_path):
            return None

        try:
            frames: list[int] = []
            poses: list[np.ndarray] = []
            with open(csv_path, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    frame = int(row["frame"])
                    x = float(row["x"])
                    y = float(row["y"])
                    yaw = float(row["yaw_xy_rad_signed"])
                    frames.append(frame)
                    poses.append(self._pose_from_local_xyyaw(x, y, yaw))
            if len(frames) <= 0 or len(poses) <= 0:
                return None
            cached_value = (
                np.asarray(frames, dtype=np.int64),
                np.stack(poses, axis=0).astype(np.float64, copy=False),
            )
            self._tracker_debug_csv_cache[int(self.scene)] = cached_value
            return cached_value
        except Exception:
            return None

    def _build_tracker_debug_gt_local_xyyaw(
        self,
        *,
        start_frame_idx: int,
        horizon_points: int,
        step_stride: int,
    ) -> np.ndarray | None:
        cached = self._load_tracker_debug_csv_poses()
        if cached is None:
            return None

        frames, poses = cached
        if frames.ndim != 1 or poses.ndim != 3 or poses.shape[0] != frames.shape[0] or poses.shape[1:] != (4, 4):
            return None
        if frames.shape[0] <= 0:
            return None

        start_frame = int(start_frame_idx)
        stride = max(1, int(step_stride))
        horizon = max(1, int(horizon_points))

        pos = int(np.searchsorted(frames, start_frame))
        if pos >= int(frames.shape[0]):
            pos = int(frames.shape[0]) - 1
        if pos > 0:
            prev_pos = pos - 1
            if abs(int(frames[prev_pos]) - start_frame) <= abs(int(frames[pos]) - start_frame):
                pos = prev_pos

        base_pose = np.asarray(poses[pos], dtype=np.float64)
        base_inv = np.linalg.inv(base_pose)
        out = np.zeros((horizon, 3), dtype=np.float64)
        for i in range(horizon):
            target_frame = start_frame + stride * (i + 1)
            target_pos = int(np.searchsorted(frames, int(target_frame)))
            if target_pos >= int(frames.shape[0]):
                target_pos = int(frames.shape[0]) - 1
            if target_pos > 0 and target_pos < int(frames.shape[0]):
                prev_target = target_pos - 1
                if abs(int(frames[prev_target]) - int(target_frame)) <= abs(int(frames[target_pos]) - int(target_frame)):
                    target_pos = prev_target
            rel = base_inv @ np.asarray(poses[target_pos], dtype=np.float64)
            out[i, 0] = float(rel[0, 3])
            out[i, 1] = float(rel[1, 3])
            out[i, 2] = self._yaw_from_pose_xy(rel)
        return out

    def _save_tracker_debug_plot(
        self,
        *,
        frame_idx: int,
        plan_local_xyyaw: np.ndarray,
        tracked_rollout_local_xyyaw: np.ndarray,
        gt_local_xyyaw: np.ndarray | None = None,
    ) -> None:
        if not bool(getattr(self, "_tracker_debug_pending", False)):
            return

        try:
            import matplotlib.pyplot as plt
        except Exception:
            self._tracker_debug_pending = False
            return

        try:
            out_dir = self._tracker_debug_output_dir()
            os.makedirs(out_dir, exist_ok=True)
            if not bool(getattr(self, "_tracker_debug_cleaned", False)):
                for name in os.listdir(out_dir):
                    if not name.lower().endswith(".png"):
                        continue
                    fp = os.path.join(out_dir, name)
                    if os.path.isfile(fp):
                        try:
                            os.remove(fp)
                        except Exception:
                            pass
                self._tracker_debug_cleaned = True

            plan = np.asarray(plan_local_xyyaw, dtype=np.float64)
            tracked = np.asarray(tracked_rollout_local_xyyaw, dtype=np.float64)
            gt = None if gt_local_xyyaw is None else np.asarray(gt_local_xyyaw, dtype=np.float64)

            fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=160)
            ax.scatter([0.0], [0.0], c="black", s=45, marker="x", label="start")
            if plan.shape[0] > 0:
                ax.plot(plan[:, 0], plan[:, 1], color="#1f77b4", marker="o", linewidth=2.0, label="proposal")
            if tracked.shape[0] > 0:
                ax.plot(tracked[:, 0], tracked[:, 1], color="#d62728", marker="x", linewidth=2.0, label="tracked")
            if gt is not None and gt.shape[0] > 0:
                ax.plot(gt[:, 0], gt[:, 1], color="#2ca02c", marker="s", linewidth=2.0, label="gt_4s_csv")

            count = int(min(plan.shape[0], tracked.shape[0]))
            for i in range(count):
                ax.text(float(plan[i, 0]), float(plan[i, 1]), f"p{i + 1}", color="#1f77b4", fontsize=8)
                ax.text(float(tracked[i, 0]), float(tracked[i, 1]), f"t{i + 1}", color="#d62728", fontsize=8)
            if gt is not None:
                for i in range(int(gt.shape[0])):
                    ax.text(float(gt[i, 0]), float(gt[i, 1]), f"g{i + 1}", color="#2ca02c", fontsize=8)

            if plan.shape[0] > 0 and tracked.shape[0] > 0:
                first_err = float(np.linalg.norm(plan[0, :2] - tracked[0, :2], ord=2))
            else:
                first_err = float("nan")
            if plan.shape[0] > 0 and gt is not None and gt.shape[0] > 0:
                gt_first_err = float(np.linalg.norm(plan[0, :2] - gt[0, :2], ord=2))
            else:
                gt_first_err = float("nan")

            ax.set_title(
                f"scene={int(self.scene):03d} frame={int(frame_idx):03d}\n"
                f"proposal vs tracker rollout (track err={first_err:.4f} m, gt err={gt_first_err:.4f} m)"
            )
            ax.set_xlabel("local x (m)")
            ax.set_ylabel("local y (m)")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.axis("equal")
            ax.legend(loc="best")
            fig.tight_layout()

            out_path = os.path.join(
                out_dir,
                f"trackerdebug_{int(self._tracker_debug_image_index):03d}_frame{int(frame_idx):03d}.png",
            )
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            self._tracker_debug_image_index += 1
            self._tracker_debug_last_path = out_path
        except Exception:
            pass
        finally:
            self._tracker_debug_pending = False

    def _build_plan_local_xyyaw(self, *, action: Any, flag: int, ax_index: int | None, ay_index: int | None) -> np.ndarray:
        # Target sampling is 4s / 0.5s = 8 points.
        n = 8

        # Prefer full planned trajectory injected by caller (e.g., policy replay traj_xyyaw).
        ext_plan = getattr(self, "_external_plan_local_xyyaw", None)
        if isinstance(ext_plan, np.ndarray) and ext_plan.ndim == 2 and ext_plan.shape[0] > 0 and ext_plan.shape[1] >= 3:
            # print("💗使用plan local xyyaw")
            m = int(min(n, ext_plan.shape[0]))
            out = np.zeros((n, 3), dtype=np.float64)
            out[:m, :3] = np.asarray(ext_plan[:m, :3], dtype=np.float64)
            if m < n:
                out[m:, :3] = out[m - 1, :3]
            
            # print(f"💗💗[IN plan local xyyaw] out is {out}")
            return out

        if int(flag) == 2 and isinstance(action, (tuple, list)) and len(action) == 4:
            dx = float(action[0])
            dy = float(action[1])
            dyaw = float(action[2])
            out = np.zeros((n, 3), dtype=np.float64)
            for k in range(1, n + 1):
                out[k - 1, 0] = float(dx) * float(k)
                out[k - 1, 1] = float(dy) * float(k)
                out[k - 1, 2] = float(dyaw) * float(k)
            return out

        if int(flag) != 1 and (ax_index is not None) and (ay_index is not None):
            try:
                selected_idx = int(ax_index) * int(self.y_anchor) + int(ay_index)
                xy = np.asarray(self.plan_anchors[selected_idx], dtype=np.float64)
                if xy.ndim == 2 and xy.shape[1] >= 2 and xy.shape[0] > 0:
                    m = int(min(n, xy.shape[0]))
                    out = np.zeros((n, 3), dtype=np.float64)
                    out[:m, :2] = xy[:m, :2]
                    if m < n:
                        out[m:, :2] = out[m - 1, :2]

                    if m >= 2:
                        dxy = out[1:m, :2] - out[0 : m - 1, :2]
                        yaws = np.zeros((m,), dtype=np.float64)
                        yaws[1:m] = np.arctan2(dxy[:, 1], dxy[:, 0])
                        yaws[0] = yaws[1] if m > 1 else 0.0
                        out[:m, 2] = yaws
                    if m < n:
                        out[m:, 2] = out[m - 1, 2]
                    return out
            except Exception:
                pass

        # Expert / unknown fallback: zero local displacement.
        return np.zeros((n, 3), dtype=np.float64)

    def _track_plan_execution(self, *, prev_pose: np.ndarray, plan_local_xyyaw: np.ndarray, dt: float = 0.5) -> TrackerExecutionResult:
        dt = max(1e-3, float(dt))

        def _fallback() -> TrackerExecutionResult:
            try:
                plan_arr = np.asarray(plan_local_xyyaw, dtype=np.float64)
            except Exception:
                plan_arr = np.zeros((0, 3), dtype=np.float64)
            if plan_arr.ndim != 2 or plan_arr.shape[0] <= 0 or plan_arr.shape[1] < 3:
                plan_arr = np.zeros((0, 3), dtype=np.float64)
                first = np.zeros((3,), dtype=np.float64)
            else:
                first = np.asarray(plan_arr[0, :3], dtype=np.float64)
            ref_frame_idx = int(max(0, int(getattr(self, "now_frame", 0)) - int(getattr(self, "step_frames", 1))))
            gt_local = self._build_tracker_debug_gt_local_xyyaw(
                start_frame_idx=ref_frame_idx,
                horizon_points=int(plan_arr.shape[0]) if plan_arr.ndim == 2 else 0,
                step_stride=int(getattr(self, "step_frames", 1)),
            )
            v = np.asarray([first[0] / dt, first[1] / dt], dtype=np.float32)
            prev_vel = np.asarray(
                self._status_prev_vel_xy if self._status_prev_vel_xy is not None else np.zeros((2,), dtype=np.float32),
                dtype=np.float32,
            )
            a = ((v - prev_vel) / dt).astype(np.float32)
            execution = build_execution_result(
                prev_pose=np.asarray(prev_pose, dtype=np.float64),
                tracked_rollout_local_xyyaw=np.asarray(plan_arr[:, :3], dtype=np.float64).copy(),
                tracked_first_local_xyyaw=np.asarray(first, dtype=np.float64),
                velocity_xy=np.asarray(v, dtype=np.float32),
                acceleration_xy=np.asarray(a, dtype=np.float32),
                steering_angle=float(getattr(self, "_status_steering_angle", 0.0)),
                steering_rate=0.0,
            )
            self._store_tracker_execution_result(execution)
            self._status_steering_angle = float(execution.steering_angle)
            self._status_steering_rate = float(execution.steering_rate)
            self._save_tracker_debug_plot(
                frame_idx=ref_frame_idx,
                plan_local_xyyaw=np.asarray(plan_arr, dtype=np.float64),
                tracked_rollout_local_xyyaw=np.asarray(execution.tracked_rollout_local_xyyaw, dtype=np.float64),
                gt_local_xyyaw=gt_local,
            )
            return execution

        if (not isinstance(plan_local_xyyaw, np.ndarray)) or plan_local_xyyaw.ndim != 2 or plan_local_xyyaw.shape[0] <= 0:
            return _fallback()
        if not self._ensure_hugsim_runtime():
            return _fallback()

        try:
            execution, _control_meta = solve_hugsim_execution(
                prev_pose=np.asarray(prev_pose, dtype=np.float64),
                plan_local_xyyaw=np.asarray(plan_local_xyyaw, dtype=np.float64),
                velocity_xy=np.asarray(self._status_vel_xy, dtype=np.float32),
                steering_angle=float(getattr(self, "_status_steering_angle", 0.0)),
                dt=float(dt),
                wheelbase=float(self._hugsim_wheelbase if self._hugsim_wheelbase is not None else 2.7),
                solve_sequence_fn=self._hugsim_solve_sequence_fn,
                build_solver_fn=self._hugsim_build_solver_fn,
            )
            ref_frame_idx = int(max(0, int(getattr(self, "now_frame", 0)) - int(getattr(self, "step_frames", 1))))
            gt_local = self._build_tracker_debug_gt_local_xyyaw(
                start_frame_idx=ref_frame_idx,
                horizon_points=int(plan_local_xyyaw.shape[0]),
                step_stride=int(getattr(self, "step_frames", 1)),
            )
            self._store_tracker_execution_result(execution)
            self._status_steering_angle = float(execution.steering_angle)
            self._status_steering_rate = float(execution.steering_rate)
            self._save_tracker_debug_plot(
                frame_idx=ref_frame_idx,
                plan_local_xyyaw=np.asarray(plan_local_xyyaw, dtype=np.float64),
                tracked_rollout_local_xyyaw=np.asarray(self._tracked_rollout_local_xyyaw, dtype=np.float64),
                gt_local_xyyaw=gt_local,
            )
            return execution
        except Exception:
            return _fallback()

    def _refresh_status_from_plan(self, *, frame_idx: int, prev_pose: np.ndarray, plan_local_xyyaw: np.ndarray) -> None:
        # Command is always from dataset (current frame), vel/acc from plan tracking.
        try:
            _vel_ds, _acc_ds, cmd = self._status_from_dataset(int(frame_idx))
        except Exception:
            cmd = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        execution = self._track_plan_execution(
            prev_pose=np.asarray(prev_pose, dtype=np.float64),
            plan_local_xyyaw=np.asarray(plan_local_xyyaw, dtype=np.float64),
            dt=float(getattr(self, "step_frames", 5)) * 0.1,
        )

        self._status_vel_xy = np.asarray(execution.velocity_xy, dtype=np.float32)
        self._status_acc_xy = np.asarray(execution.acceleration_xy, dtype=np.float32)
        self._status_cmd = np.asarray(cmd, dtype=np.float32)
        self._status_prev_vel_xy = self._status_vel_xy.copy()

    def _apply_external_execution_override(self) -> None:
        execution = getattr(self, "_external_execution_override", None)
        self._external_execution_override = None
        if not isinstance(execution, TrackerExecutionResult):
            return
        self._store_tracker_execution_result(execution)
        self._status_vel_xy = np.asarray(execution.velocity_xy, dtype=np.float32)
        self._status_acc_xy = np.asarray(execution.acceleration_xy, dtype=np.float32)
        self._status_prev_vel_xy = self._status_vel_xy.copy()
        self._status_steering_angle = float(execution.steering_angle)
        self._status_steering_rate = float(execution.steering_rate)

    def _refresh_status_from_dataset(self) -> None:
        try:
            vel, acc, cmd = self._status_from_dataset(int(self.now_frame))
            self._status_vel_xy = np.asarray(vel, dtype=np.float32)
            self._status_acc_xy = np.asarray(acc, dtype=np.float32)
            self._status_cmd = np.asarray(cmd, dtype=np.float32)
            self._status_prev_vel_xy = self._status_vel_xy.copy()
        except Exception:
            # Keep previous defaults when dataset mapping is unavailable.
            pass

    def _refresh_status_from_derivation(self, *, frame_idx: int, plan_local_xyyaw: np.ndarray, dt: float = 0.5) -> None:
        """
        Compatibility-only status refresh under a perfect one-step tracking assumption:
        ego reaches the first planned local pose exactly in dt.
        """
        dt = max(1e-3, float(dt))

        try:
            _vel_ds, _acc_ds, cmd = self._status_from_dataset(int(frame_idx))
        except Exception:
            cmd = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        if (not isinstance(plan_local_xyyaw, np.ndarray)) or plan_local_xyyaw.ndim != 2 or plan_local_xyyaw.shape[0] <= 0:
            first = np.zeros((3,), dtype=np.float64)
        else:
            first = np.asarray(plan_local_xyyaw[0, :3], dtype=np.float64)

        # Perfect one-step tracking: first point is executed within dt.
        x1 = float(first[0])
        y1 = float(first[1])
        dyaw = self._wrap_angle(float(first[2]))

        vel = np.asarray([x1 / dt, y1 / dt], dtype=np.float32)
        prev_vel = np.asarray(
            self._status_prev_vel_xy if self._status_prev_vel_xy is not None else np.zeros((2,), dtype=np.float32),
            dtype=np.float32,
        )
        acc = ((vel - prev_vel) / dt).astype(np.float32)

        self._tracked_first_step_xyyaw = np.asarray([x1, y1, float(dyaw)], dtype=np.float64)
        self._status_vel_xy = np.asarray(vel, dtype=np.float32)
        self._status_acc_xy = np.asarray(acc, dtype=np.float32)
        self._status_cmd = np.asarray(cmd, dtype=np.float32)
        self._status_prev_vel_xy = self._status_vel_xy.copy()
        self._status_steering_rate = 0.0

    # Backward-compatible alias with the user's original spelling.
    def _refresh_status_form_divation(self, *, frame_idx: int, plan_local_xyyaw: np.ndarray, dt: float = 0.5) -> None:
        self._refresh_status_from_derivation(
            frame_idx=int(frame_idx),
            plan_local_xyyaw=np.asarray(plan_local_xyyaw, dtype=np.float64),
            dt=float(dt),
        )
 
 #ADD: COMMAND     end   

    # ------------------------- Private loading functions ------------------------ #
    # ALL_CAMS_FILE   = os.path.join(DATA_ROOT, "others", "all_cams.pkl") 6 个相机的「静态相机参数模板」
    # ALL_IMAGES_FILE = os.path.join(DATA_ROOT, "others", "all_images.pkl")
    def _load_camera_and_images(self):
        with open(cfg.ALL_CAMS_FILE, "rb") as f:
            self.all_cams = pickle.load(f)
        with open(cfg.ALL_IMAGES_FILE, "rb") as f:
            self.all_images = pickle.load(f)

    def _load_ego_and_cam_matrices(self):
        cam2ego = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/0.txt"))
        ego2world = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/000.txt"))
        self.camera_front_start = ego2world @ cam2ego
        self._world_to_front_start = np.linalg.inv(self.camera_front_start)
        self.start_ego = np.linalg.inv(self.camera_front_start) @ ego2world

        # Load all camera-to-ego matrices
        self.cam2ego = [
            np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/{i}.txt"))
            for i in range(6)
            if os.path.exists(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/{i}.txt"))
        ]

    def _load_expert_ego_frames(self):#Note:专家车辆轨迹（ground-truth trajectory）:世界坐标到前置相机起始坐标的相对变换
        self.all_expert_ego = []
        for i in range(0, self.final_frame + self.step_frames, self.step_frames):
            expert_world = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/{i:03d}.txt"))
            expert_world = np.linalg.inv(self.camera_front_start) @ expert_world
            self.all_expert_ego.append(expert_world)

    def _load_plan_anchors(self):
        self.plan_anchors = torch.from_numpy(np.load(cfg.PLAN_ANCHORS_FILE).astype(np.float32))
        self.plan_anchors_yaw = torch.from_numpy(np.load(cfg.PLAN_ANCHORS_YAW_FILE).astype(np.float32)) * 5
        self.plan_anchors_mask = torch.from_numpy(np.load(cfg.PLAN_ANCHORS_MASK_FILE).reshape(-1))
        self.x_anchor = 61
        self.y_anchor = 61
        self.anchor_exec_index = max(0, int(self.plan_anchors.shape[1]) - 1)

    def _load_token_mappings(self):
        frame2token_path = os.path.join(cfg.FRAME2TOKEN_DIR, f"{self.scene:03d}.json")
        with open(frame2token_path, 'r') as f:
            data = json.load(f)
            self.frame2token = {v: k for k, v in data.items()}
        with open(cfg.TOKEN2VAD_FILE, 'rb') as f:
            self.token2vad = pickle.load(f)

    # ------------------------- Observation & Info ------------------------ #
    def _get_obs(self):
        """
        Compute observation images from all active cameras using the trainer.根据当前相机信息生成可观察的 RGB 图像
        """
        self.now_observe_image = []
        render_t0 = time.perf_counter()
        camera_render_s: list[float] = []
        with torch.no_grad():
            for cam in self.all_camera_now:
                cam_info, img_info = cam
                camera_t0 = time.perf_counter()
                results = self.trainer(img_info, cam_info)# self.trainer(img_info, cam_info)：
                rgb = results['rgb'].clamp(0, 1).cpu().numpy()
                scaled_rgb = (rgb * 255).astype(np.uint8)
                self.now_observe_image.append(scaled_rgb)
                camera_render_s.append(float(time.perf_counter() - camera_t0))
        self.all_camera_now = []
        render_total_s = float(time.perf_counter() - render_t0)
        self._last_obs_timing = {
            "render_s": float(render_total_s),
            "render_camera_total_s": float(sum(camera_render_s)),
            "render_camera_avg_s": float(sum(camera_render_s) / float(max(1, len(camera_render_s)))),
            "render_camera_max_s": float(max(camera_render_s) if len(camera_render_s) > 0 else 0.0),
        }
        out = {
            "front": self.now_observe_image[0],
            "front_left": self.now_observe_image[1],
            "front_right": self.now_observe_image[2],
            "back_left": self.now_observe_image[3],
            "back_right": self.now_observe_image[4],
            "back": self.now_observe_image[5],
        }
        out["ego_velocity"] = self._status_vel_xy.astype(np.float32, copy=True)
        out["ego_acceleration"] = self._status_acc_xy.astype(np.float32, copy=True)
        out["driving_command"] = self._status_cmd.astype(np.float32, copy=True)
        out["ego_status"] = np.concatenate(
            # Align with DDV2/Transfuser `status_feature` ordering:
            # [driving_command(4), ego_velocity(2), ego_acceleration(2)].
            [out["driving_command"], out["ego_velocity"], out["ego_acceleration"]],
            axis=0,
        ).astype(np.float32, copy=False)

        # Extra metadata for downstream model-based agents (e.g., SparseDrive).
        # Keep these as lightweight CPU numpy arrays.
        try:
            out["scene_id"] = np.int32(int(getattr(self, "scene", 0)))
        except Exception:
            out["scene_id"] = np.int32(0)
        try:
            out["frame_idx"] = np.int32(int(getattr(self, "now_frame", 0)))
        except Exception:
            out["frame_idx"] = np.int32(0)
        try:
            out["step_frames"] = np.int32(int(getattr(self, "step_frames", 1)))
        except Exception:
            out["step_frames"] = np.int32(1)
        # A simple monotonic timestamp in seconds (10Hz base * step_frames).
        try:
            out["timestamp"] = np.float32(float(out["frame_idx"]) * 0.1)
        except Exception:
            out["timestamp"] = np.float32(0.0)
        # Ego pose in the simulator's local (front-start) frame.
        try:
            out["ego_pose"] = np.asarray(self.start_ego, dtype=np.float32)
        except Exception:
            out["ego_pose"] = np.eye(4, dtype=np.float32)
        # Camera calibration (constant per-scene).
        try:
            if hasattr(self, "cam2ego") and isinstance(self.cam2ego, list) and len(self.cam2ego) == 6:
                out["cam2ego"] = np.asarray(np.stack(self.cam2ego, axis=0), dtype=np.float32)
        except Exception:
            pass
        try:
            if hasattr(self, "all_cams") and isinstance(self.all_cams, list) and len(self.all_cams) == 6:
                intr = []
                hw = []
                for cam in self.all_cams:
                    intr.append(np.asarray(cam.get("intrinsics"), dtype=np.float32))
                    hw.append([float(cam.get("height", self.h)), float(cam.get("width", self.w))])
                out["cam_intrinsics"] = np.stack(intr, axis=0).astype(np.float32, copy=False)
                out["cam_hw"] = np.asarray(hw, dtype=np.float32)
        except Exception:
            pass
        return out

    def _nearest_expert_tracking_info(self) -> dict[str, Any]:
        try:
            act_pose = np.asarray(getattr(self, "start_ego"), dtype=np.float64)
        except Exception:
            return {}

        try:
            expert_poses = list(getattr(self, "expert_world_all"))
            expert_pair = np.asarray(getattr(self, "expert_pair"), dtype=np.float64)
        except Exception:
            return {}

        if act_pose.shape != (4, 4):
            return {}
        if len(expert_poses) <= 0 or expert_pair.ndim != 2 or expert_pair.shape[1] != 2:
            return {}

        count = min(int(len(expert_poses)), int(expert_pair.shape[0]))
        if count <= 0:
            return {}

        act_pos = np.asarray(act_pose[:3, 3], dtype=np.float32)
        act_xz = np.asarray(act_pose[:3, 3][[0, 2]], dtype=np.float64)
        candidate_xz = np.asarray(expert_pair[:count], dtype=np.float64)
        distances = np.linalg.norm(candidate_xz - act_xz.reshape(1, 2), axis=1)
        nearest_idx = int(np.argmin(distances))

        exp_pose = np.asarray(expert_poses[nearest_idx], dtype=np.float64)
        if exp_pose.shape != (4, 4):
            return {}
        exp_pos = np.asarray(exp_pose[:3, 3], dtype=np.float32)

        exp_yaw_deg = float(np.rad2deg(self._yaw_from_pose_xz(exp_pose)))
        act_yaw_deg = float(np.rad2deg(self._yaw_from_pose_xz(act_pose)))
        yaw_err_deg = abs(float(np.rad2deg(self._wrap_angle(np.deg2rad(act_yaw_deg - exp_yaw_deg)))))
        xz_err_m = float(distances[nearest_idx])

        info = {
            "scene_id": int(getattr(self, "scene", 0)),
            "now_frame": int(getattr(self, "now_frame", 0)),
            "nearest_expert_idx": int(nearest_idx),
            "exp_pos": exp_pos,
            "act_pos": act_pos,
            "exp_yaw_deg": float(exp_yaw_deg),
            "act_yaw_deg": float(act_yaw_deg),
            "xz_err_m": float(xz_err_m),
            "xy_err_m": float(xz_err_m),
            "yaw_err_deg": float(yaw_err_deg),
            "ground_ref_idx": int(nearest_idx),
            "ground_ref_pos": exp_pos,
            "ground_ref_dist_m": float(xz_err_m),
        }

        self.last_exp_pos = exp_pos
        self.last_act_pos = act_pos
        self.last_exp_yaw_deg = float(exp_yaw_deg)
        self.last_act_yaw_deg = float(act_yaw_deg)
        self.last_xy_err_m = float(xz_err_m)
        self.last_yaw_err_deg = float(yaw_err_deg)
        self.last_ground_ref_idx = int(nearest_idx)
        self.last_ground_ref_pos = exp_pos
        self.last_ground_ref_dist_m = float(xz_err_m)
        return info

    def _get_info(self):
        info = self._nearest_expert_tracking_info()
        if isinstance(info, dict) and len(self._last_info_timing) > 0:
            info["timing"] = dict(self._last_info_timing)
        return info

#     # ------------------------- Gym API ------------------------ #
# ------------------------- Gym API ------------------------ #
    def reset(self, seed=None, options=None):#NOTE 重置环境，重新开始一个新场景。
        start_frame = None
        step_frames = None
        try:
            if isinstance(options, dict):
                if options.get("start_frame") is not None:
                    start_frame = int(options.get("start_frame"))
                if options.get("step_frames") is not None:
                    step_frames = int(options.get("step_frames"))
        except Exception:
            start_frame = None
            step_frames = None

        self.update(seed, step_frames=step_frames, start_frame=start_frame)
        self._tracked_first_step_xyyaw = np.zeros((3,), dtype=np.float64)
        self._executed_first_step_xyyaw = np.zeros((3,), dtype=np.float64)
        self._tracked_rollout_local_xyyaw = np.zeros((0, 3), dtype=np.float64)
        self._last_tracker_execution = None
        self._external_execution_override = None
        self._status_steering_angle = 0.0
        self._status_steering_rate = 0.0
        self._tracker_debug_pending = False
        self._tracker_debug_cleaned = False
        self._tracker_debug_image_index = 0
        self._tracker_debug_last_path = None
        self._tracker_debug_csv_cache = {}
        # Initialize status vectors from dataset at the selected start frame.
        self._refresh_status_from_dataset()
        start_pose = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/{self.now_frame:03d}.txt"))
        self.start_ego = np.linalg.inv(self.camera_front_start) @ start_pose

        self.all_camera_now = []
        camera_setup_t0 = time.perf_counter()
        get_sky_view_s = 0.0
        for i in range(6):
            cam_info = copy.deepcopy(self.all_cams[i])
            cam_info = move_to_device(cam_info, self.device)
            cam_info['camera_to_world'] = torch.tensor(self.start_ego @ self.cam2ego[i], device=self.device, dtype=torch.float32)
            cam_info['camera_to_world'] = cam_info['camera_to_world'] @ self._transform_matrix

            img_info = copy.deepcopy(self.all_images[i])
            img_info = move_to_device(img_info, self.device)
            sky_t0 = time.perf_counter()
            img_info['origins'], img_info['viewdirs'], img_info['direction_norm'] = get_sky_view(
                cam_info['camera_to_world'], cam_info['intrinsics'], self.device, self.h, self.w
            )
            get_sky_view_s += float(time.perf_counter() - sky_t0)
            img_info['normed_time'] = torch.tensor(
                self.trainer.normalized_timestamps[self.now_frame].item(),
                device=self.device,
                dtype=torch.float32,
            )
            self.all_camera_now.append((cam_info, img_info))

        camera_setup_s = float(time.perf_counter() - camera_setup_t0)
        obs = self._get_obs()
        self._last_info_timing = dict(self._last_obs_timing)
        self._last_info_timing["camera_setup_s"] = float(camera_setup_s)
        self._last_info_timing["get_sky_view_s"] = float(get_sky_view_s)
        self._last_info_timing["obs_pipeline_s"] = float(camera_setup_s + float(self._last_obs_timing.get("render_s", 0.0)))
        return obs, self._get_info()
    
    def step(self, action):#NOT 根据动作 action 更新车辆状态（ego pose
        self.now_frame = min(int(self.final_frame - 1), int(self.now_frame + int(self.step_frames)))
        # Action parsing (backward-compatible but clearer):
        # - Expert: action is None or "expert" or (0,0,1)
        # - First-step pose: (x:float, y:float, yaw:float, 2)
        # - Anchor index: (ax:int, ay:int, 0)  (3-tuple where last!=1)
        if action is None or action == "expert":
            ax_index = 0
            ay_index = 0
            flag = 1
            x_cmd = y_cmd = yaw_cmd = 0.0
        elif isinstance(action, (tuple, list)) and len(action) == 4:#走连续模式 flag=2
            x_cmd, y_cmd, yaw_cmd, flag = action
            ax_index = ay_index = None
            x_cmd = float(x_cmd)
            y_cmd = float(y_cmd)
            yaw_cmd = float(yaw_cmd)
            flag = int(flag)
        else:
            ax_index, ay_index, flag = action #走anchor模式
            ax_index = int(ax_index)
            ay_index = int(ay_index)
            flag = int(flag)
            x_cmd = y_cmd = yaw_cmd = 0.0

        # Build planned trajectory in local frame for status tracking update.
        self._tracker_debug_pending = bool(
            int(flag) == 2
            and isinstance(getattr(self, "_external_plan_local_xyyaw", None), np.ndarray)
        )
        plan_local_xyyaw = self._build_plan_local_xyyaw(
            action=action,
            flag=int(flag),
            ax_index=ax_index,
            ay_index=ay_index,
        )
        # Consume one-shot external plan to avoid stale reuse.
        self._external_plan_local_xyyaw = None

        # --- 计算专家下一帧位姿（world→front-start 相对变换） ---
        #NOTE 所有 motion 都在 front_start 局部坐标系
        expert_next_ego = np.linalg.inv(self.camera_front_start) @ np.loadtxt(
            os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/{self.now_frame:03d}.txt")
        )
        prev_pose = self.start_ego.copy()#前视摄像头front-camera坐标系
        
        
        #TODO:使用原来的未经过跟踪的轨迹点
        #试用一下完全的数据集
        # self._refresh_status_from_dataset()
        # print(f"💗[In step] dataset vel = {self._status_vel_xy}, acc = {self._status_acc_xy}, cmd = {self._status_cmd}")
        
        # # 使用控制器跟踪：Update status from tracked plan (vel/acc) + dataset command.
        self._refresh_status_from_plan(
            frame_idx=int(self.now_frame),
            prev_pose=np.asarray(prev_pose, dtype=np.float64),
            plan_local_xyyaw=np.asarray(plan_local_xyyaw, dtype=np.float64),
        )
        self._apply_external_execution_override()
        # print(f"💗[In step] plan vel = {self._status_vel_xy}, acc = {self._status_acc_xy}, cmd = {self._status_cmd}")
        # self._refresh_status_from_dataset()
        # print(f"💗[In step] dataset vel = {self._status_vel_xy}, acc = {self._status_acc_xy}, cmd = {self._status_cmd}")
    
        
        # --- 计算 action 推进的“假设下一帧”位姿（用于对比或真实推进） ---
        # front-start 局部坐标约定：x-y 为平面，yaw 绕 z 轴。
        if flag == 1:
            self._last_tracker_execution = None
            self.start_ego = expert_next_ego
        elif flag == 2:
            execution = getattr(self, "_last_tracker_execution", None)
            if isinstance(execution, TrackerExecutionResult):
                self.start_ego = np.asarray(execution.executed_pose, dtype=np.float64).copy()
            else:
                dx_fwd, dy_left, dyaw = (
                    float(self._tracked_first_step_xyyaw[0]),
                    float(self._tracked_first_step_xyyaw[1]),
                    float(self._tracked_first_step_xyyaw[2]),
                )
                c = math.cos(dyaw)
                s = math.sin(dyaw)
                tpt = np.array(
                    [
                        [c, -s, 0.0, dx_fwd],
                        [s, c, 0.0, dy_left],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                self.start_ego = self.start_ego @ tpt
        else:
            self._last_tracker_execution = None
            selected_idx = ax_index * self.y_anchor + ay_index
            exec_idx = min(int(getattr(self, "anchor_exec_index", 7)), int(self.plan_anchors[selected_idx].shape[0]) - 1)#anchor上的轨迹点全部执行
            future_xy = self.plan_anchors[selected_idx][exec_idx, :]
            if torch.is_tensor(future_xy):
                dx_fwd = float(future_xy[0].item())
                dy_left = float(future_xy[1].item())
            else:
                dx_fwd = float(future_xy[0])
                dy_left = float(future_xy[1])

            future_yaw_v = self.plan_anchors_yaw[selected_idx]
            if torch.is_tensor(future_yaw_v):
                future_yaw = float(future_yaw_v.item())
            else:
                future_yaw = float(future_yaw_v)
            # Apply anchor step as relative SE(2) in x-y plane, yaw about +z.
            c = math.cos(future_yaw)
            s = math.sin(future_yaw)
            tpt = np.array(
                [
                    [c, -s, 0.0, dx_fwd],
                    [s, c, 0.0, dy_left],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            self.start_ego = self.start_ego @ tpt

        if bool(getattr(self, "use_expert_height", False)):
            try:
                y_ref = float(self.updateGroundDistance())
                self.start_ego[1, 3] = y_ref
            except Exception:
                pass
        w, h = int(self.w), int(self.h)
        camera_setup_t0 = time.perf_counter()
        get_sky_view_s = 0.0
        for i in range(6):#NOTE 更新相机信息
            loaded_cam_infos = copy.deepcopy(self.all_cams[i])
            loaded_cam_infos = move_to_device(loaded_cam_infos,self.device)
            loaded_cam_infos['camera_to_world'] = torch.tensor(self.start_ego @ self.cam2ego[i]).to(self.device).to(torch.float32)
            loaded_cam_infos['camera_to_world'] = loaded_cam_infos['camera_to_world'] @ self._transform_matrix
            loaded_img_infos = copy.deepcopy(self.all_images[i])
            loaded_img_infos = move_to_device(loaded_img_infos,self.device)
            sky_t0 = time.perf_counter()
            loaded_img_infos['origins'],\
            loaded_img_infos['viewdirs'], \
            loaded_img_infos['direction_norm'] = get_sky_view(loaded_cam_infos['camera_to_world'],\
                                                                  loaded_cam_infos['intrinsics'],\
                                                                    self.device,h,w)
            get_sky_view_s += float(time.perf_counter() - sky_t0)
            loaded_img_infos['normed_time'] = torch.tensor(
                self.trainer.normalized_timestamps[self.now_frame].item(),
                device=self.device,
                dtype=torch.float32,
            )
            self.all_camera_now.append((loaded_cam_infos,loaded_img_infos))
        observation = self._get_obs()#自动传入了self.all_camera_now
        camera_setup_s = float(time.perf_counter() - camera_setup_t0)
        self._last_info_timing = dict(self._last_obs_timing)
        self._last_info_timing["camera_setup_s"] = float(camera_setup_s)
        self._last_info_timing["get_sky_view_s"] = float(get_sky_view_s)
        self._last_info_timing["obs_pipeline_s"] = float(camera_setup_s + float(self._last_obs_timing.get("render_s", 0.0)))

        terminated, truncated = False, False
        if self.now_frame == self.final_frame - 1:
            terminated = True
        else:
            terminated = False
        
        if bool(self.check_coliision()):
            truncated = True
        else:
            truncated = False
        
        return observation, terminated, truncated, self._get_info()#self._get_info()用于收集车辆上一时刻的状态和误差信息
    
    def check_coliision(self):
        return False
    
    def update(self, scene: int, *, step_frames: int = None, start_frame: int = None):
        self.scene = int(scene)
        if step_frames is not None:
            self.step_frames = int(step_frames)        
        sf = 0
        try:
            if start_frame is not None:
                sf = int(start_frame)
        except Exception:
            sf = 0
        try:
            sf = max(0, min(int(sf), int(self.final_frame) - 1))
        except Exception:
            sf = max(0, int(sf))
        try:
            if int(self.step_frames) > 1:
                sf = (sf // int(self.step_frames)) * int(self.step_frames)
        except Exception:
            pass

        self.now_frame = int(sf)
        self.all_camera_now = []
        self.save = None

        self.trainer, self.num_timesteps = get_splat(self.device, self.scene)
        self.trainer.eval()

        with open(cfg.ALL_CAMS_FILE, "rb") as f:
            self.all_cams = pickle.load(f)
        with open(cfg.ALL_IMAGES_FILE, "rb") as f:
            self.all_images = pickle.load(f)

        cam2ego_0 = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/0.txt"))
        ego2world_sf = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/{self.now_frame:03d}.txt"))
        self.camera_front_start = ego2world_sf @ cam2ego_0

        self.start_ego = np.linalg.inv(self.camera_front_start) @ ego2world_sf
        
        self.cam2ego = []
        for i in range(6):
            cam_path = os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/{i}.txt")
            if os.path.exists(cam_path):
                self.cam2ego.append(np.loadtxt(cam_path))

        self.all_expert_ego = []
        for i in range(0, self.final_frame + self.step_frames, self.step_frames):
            expert_world = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/{i:03d}.txt"))
            expert_world = np.linalg.inv(self.camera_front_start) @ expert_world
            self.all_expert_ego.append(expert_world)
        self.get_all_point_for_expert()

    def get_all_point_for_expert(self):
        self.expert_world_all = []
        for i in range(len(self.all_expert_ego) - 1):
            start_matrix = self.all_expert_ego[i]
            end_matrix = self.all_expert_ego[i + 1]
            for alpha in np.linspace(0, 1, 40): 
                translation = (1 - alpha) * start_matrix[:3, 3] + alpha * end_matrix[:3, 3]
                start_rot = R.from_matrix(start_matrix[:3, :3])
                end_rot = R.from_matrix(end_matrix[:3, :3])
                interp_rot = slerp(start_rot, end_rot, alpha)
                new_matrix = np.eye(4)
                new_matrix[:3, :3] = interp_rot.as_matrix()
                new_matrix[:3, 3] = translation
                self.expert_world_all.append(new_matrix)

        self.expert_pair = [matrix[:3, 3][[0, 2]] for matrix in self.expert_world_all]
        self.expert_altitude  = [matrix[:3, 3][[1]] for matrix in self.expert_world_all]


    def updateGroundDistance(self):#NOT 用当前 x,z 找到离自己最近的 expert 点;取这个 expert 点的 y 作为地面高度
        start_ego_position = self.start_ego[:3, 3][[0, 2]]
        distances = cdist([start_ego_position], self.expert_pair, 'euclidean')[0]
        nearest_indices = np.argsort(distances)[:1] 
        return float(self.expert_altitude[nearest_indices[0]][0])
