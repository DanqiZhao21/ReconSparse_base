from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation, Slerp

from framework.env_wrapper.hugsim_recon_alignment import (
    HUGSIMReconAlignment,
    Sim2Transform,
    build_hugsim_recon_alignment,
    build_local_hugsim_recon_alignment,
    transform_hugsim_box_to_recon_poly,
    transform_hugsim_boxes_to_recon_objects,
    transform_hugsim_ego_box_to_reward_pose,
)
from framework.env_wrapper.hugsim_scene_index import HUGSIMFrameMapping
from framework.rewards import TrackingRewardComputer
from framework.utils.repo_paths import resolve_hugsim_path, resolve_hugsim_root


HUGSIM_REPO_DEFAULT = resolve_hugsim_root()
HUGSIM_CAMERA_MAP = {
    "CAM_FRONT_LEFT": ("front_left", 1),
    "CAM_FRONT": ("front", 0),
    "CAM_FRONT_RIGHT": ("front_right", 2),
    "CAM_BACK_LEFT": ("back_left", 3),
    "CAM_BACK": ("back", 4),
    "CAM_BACK_RIGHT": ("back_right", 5),
}
ALL_CAMERA_INDEX = {
    "front": 0,
    "front_left": 1,
    "front_right": 2,
    "back_left": 3,
    "back": 4,
    "back_right": 5,
}

RECON_DATA_ROOT_DEFAULT = Path(__file__).resolve().parents[2] / "assets" / "nus" / "data"
FIFO_RUNNER_DEFAULT = Path(__file__).resolve().parent / "hugsim_fifo_runner.py"


def _fov2focal(fov: float, pixels: int) -> float:
    return float(pixels) / (2.0 * math.tan(float(fov) / 2.0))


def _intrinsics_3x3(intrinsic_cfg: dict[str, Any]) -> np.ndarray:
    h = int(intrinsic_cfg["H"])
    w = int(intrinsic_cfg["W"])
    k = np.eye(3, dtype=np.float32)
    k[0, 0] = _fov2focal(float(intrinsic_cfg["fovx"]), w)
    k[1, 1] = _fov2focal(float(intrinsic_cfg["fovy"]), h)
    k[0, 2] = float(intrinsic_cfg["cx"])
    k[1, 2] = float(intrinsic_cfg["cy"])
    return k


def _command4(hugsim_info: dict[str, Any]) -> np.ndarray:
    mapping = {0: 2, 1: 0, 2: 1, 3: 3}
    out = np.zeros((4,), dtype=np.float32)
    try:
        slot = mapping.get(int(hugsim_info.get("command", 2)), 1)
    except Exception:
        slot = 1
    out[int(slot)] = 1.0
    return out


def _ego_status(hugsim_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steer = float(hugsim_info.get("ego_steer", 0.0))
    speed = float(hugsim_info.get("ego_velo", 0.0))
    acc = float(hugsim_info.get("accelerate", 0.0))
    yaw = -steer
    velocity = np.asarray([speed * math.cos(yaw), speed * math.sin(yaw)], dtype=np.float32)
    acceleration = np.asarray([acc * math.cos(yaw), acc * math.sin(yaw)], dtype=np.float32)
    command = _command4(hugsim_info)
    return command, velocity, acceleration


def _ego_pose_matrix(hugsim_info: dict[str, Any]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    try:
        rot = np.asarray(hugsim_info.get("ego_rot", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
        trans = np.asarray(hugsim_info.get("ego_pos", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
        pose[:3, :3] = Rotation.from_euler("XYZ", rot).as_matrix().astype(np.float32)
        pose[:3, 3] = trans
    except Exception:
        pass
    return pose


def _recon_xy_pose_to_reward_xz_pose(pose_xy: Any) -> np.ndarray:
    src = np.asarray(pose_xy, dtype=np.float64).reshape(4, 4)
    out = np.asarray(src, dtype=np.float64).copy()
    out[0, 3] = float(src[0, 3])
    out[1, 3] = 0.0
    out[2, 3] = float(src[1, 3])
    yaw = float(math.atan2(float(src[1, 0]), float(src[0, 0])))
    c, s = float(math.cos(yaw)), float(math.sin(yaw))
    out[:3, :3] = np.asarray(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=np.float64,
    )
    return out


class HUGSIMRewardProxy:
    """State adapter consumed by ReconDreamer-RL TrackingRewardComputer."""

    def __init__(self, *, recon_data_root: str | Path = RECON_DATA_ROOT_DEFAULT) -> None:
        self.recon_data_root = Path(recon_data_root)
        self.scene = 0
        self.now_frame = 0
        self.start_ego = np.eye(4, dtype=np.float64)
        self.all_expert_ego: list[np.ndarray] = []
        self.expert_world_all: list[np.ndarray] = []
        self.expert_pair: list[np.ndarray] = []
        self._expert_scene: int | None = None
        self._status_vel_xy = np.zeros((2,), dtype=np.float64)
        self._status_acc_xy = np.zeros((2,), dtype=np.float64)
        self._status_cmd = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    def update_from_hugsim_info(
        self,
        *,
        recon_scene_id: int,
        frame_idx: int,
        hugsim_info: dict[str, Any],
        reward_pose: np.ndarray | None = None,
    ) -> None:
        self.scene = int(recon_scene_id)
        self.now_frame = int(frame_idx)
        self._ensure_expert_loaded(int(recon_scene_id))
        if reward_pose is not None:
            self.start_ego = np.asarray(reward_pose, dtype=np.float64).reshape(4, 4)
        else:
            self.start_ego = _ego_pose_matrix(hugsim_info).astype(np.float64, copy=False)
        command, velocity, acceleration = _ego_status(hugsim_info)
        self._status_cmd = command.astype(np.float64, copy=False)
        self._status_vel_xy = velocity.astype(np.float64, copy=False)
        self._status_acc_xy = acceleration.astype(np.float64, copy=False)

    def _ensure_expert_loaded(self, recon_scene_id: int) -> None:
        if self._expert_scene == int(recon_scene_id) and len(self.all_expert_ego) > 0:
            return
        scene_dir = self.recon_data_root / f"{int(recon_scene_id):03d}" / "ego_pose"
        poses: list[np.ndarray] = []
        for path in sorted(scene_dir.glob("*.txt")):
            try:
                poses.append(_recon_xy_pose_to_reward_xz_pose(np.loadtxt(path)))
            except Exception:
                continue
        if not poses:
            poses = [np.eye(4, dtype=np.float64)]
        self._expert_scene = int(recon_scene_id)
        self.all_expert_ego = poses
        self._build_dense_expert_path()

    def _build_dense_expert_path(self) -> None:
        dense: list[np.ndarray] = []
        if len(self.all_expert_ego) <= 1:
            dense = [np.asarray(p, dtype=np.float64).copy() for p in self.all_expert_ego]
        else:
            for idx in range(len(self.all_expert_ego) - 1):
                start = np.asarray(self.all_expert_ego[idx], dtype=np.float64)
                end = np.asarray(self.all_expert_ego[idx + 1], dtype=np.float64)
                start_rot = Rotation.from_matrix(start[:3, :3])
                end_rot = Rotation.from_matrix(end[:3, :3])
                slerp = Slerp([0.0, 1.0], Rotation.concatenate([start_rot, end_rot]))
                for alpha in np.linspace(0.0, 1.0, 40):
                    pose = np.eye(4, dtype=np.float64)
                    pose[:3, 3] = (1.0 - float(alpha)) * start[:3, 3] + float(alpha) * end[:3, 3]
                    pose[:3, :3] = slerp([float(alpha)])[0].as_matrix()
                    dense.append(pose)
        self.expert_world_all = dense
        self.expert_pair = [np.asarray(p[:3, 3][[0, 2]], dtype=np.float64) for p in dense]


def build_recondreamer_obs_from_hugsim(
    *,
    hugsim_obs: dict[str, Any],
    hugsim_info: dict[str, Any],
    mapping: HUGSIMFrameMapping,
    hugsim_step_idx: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rgb = hugsim_obs.get("rgb", {})
    cam_params = hugsim_info.get("cam_params", {})

    cam2ego = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 6, axis=0)
    cam_intr = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 6, axis=0)
    cam_hw = np.zeros((6, 2), dtype=np.float32)
    cam_dist = np.zeros((6, 5), dtype=np.float32)

    for hugsim_cam, (obs_key, cam_idx) in HUGSIM_CAMERA_MAP.items():
        if hugsim_cam not in rgb:
            raise KeyError(f"Missing HUGSIM RGB camera: {hugsim_cam}")
        image = np.asarray(rgb[hugsim_cam], dtype=np.uint8)
        out[obs_key] = image
        params = cam_params.get(hugsim_cam, {})
        if "v2c" in params:
            try:
                cam2ego[cam_idx] = np.linalg.inv(np.asarray(params["v2c"], dtype=np.float32))
            except Exception:
                cam2ego[cam_idx] = np.eye(4, dtype=np.float32)
        if "intrinsic" in params:
            cam_intr[cam_idx] = _intrinsics_3x3(dict(params["intrinsic"]))
            cam_hw[cam_idx] = np.asarray(
                [
                    float(params["intrinsic"].get("H", image.shape[0])),
                    float(params["intrinsic"].get("W", image.shape[1])),
                ],
                dtype=np.float32,
            )
        else:
            cam_hw[cam_idx] = np.asarray([float(image.shape[0]), float(image.shape[1])], dtype=np.float32)

    command, velocity, acceleration = _ego_status(hugsim_info)
    ego_status = np.concatenate([command, velocity, acceleration], axis=0).astype(np.float32)
    out["driving_command"] = command
    out["ego_velocity"] = velocity
    out["ego_acceleration"] = acceleration
    out["ego_status"] = ego_status
    out["scene_id"] = np.int32(int(mapping.recon_scene_id))
    out["frame_idx"] = np.int32(int(mapping.frame_idx))
    out["sample_token"] = str(mapping.sample_token)
    out["token"] = str(mapping.sample_token)
    out["timestamp"] = np.float32(float(mapping.sample_relative_time_s))
    out["hugsim_timestamp"] = np.float32(float(mapping.hugsim_relative_time_s))
    out["hugsim_step_idx"] = np.int32(int(hugsim_step_idx))
    out["official_scene_name"] = str(mapping.official_scene_name)
    out["ego_pose"] = _ego_pose_matrix(hugsim_info)
    out["cam2ego"] = cam2ego
    out["cam_intrinsics"] = cam_intr
    out["cam_hw"] = cam_hw
    out["cam_distortions"] = cam_dist
    return out



# Standardize the termination and collision information in HUGSIM info.
def _augment_hugsim_reward_info(
    *,
    info: dict[str, Any],
    hugsim_info: dict[str, Any],
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    out = dict(info)
    collision = bool(hugsim_info.get("collision", False))
    route_completed = False
    try:
        route_completed = float(hugsim_info.get("rc", 0.0)) >= 1.0
    except Exception:
        route_completed = False

    out["collision"] = bool(collision)
    out["static_collision"] = False
    out["dynamic_collision"] = bool(collision)
    out["route_completed"] = bool(route_completed)
    if "rc" in hugsim_info:
        try:
            out["hugsim_route_completion"] = float(hugsim_info["rc"])
        except Exception:
            pass
    if "ego_box" in hugsim_info:
        out["ego_box"] = np.asarray(hugsim_info["ego_box"], dtype=np.float64).reshape(-1).tolist()
    if "obj_boxes" in hugsim_info:
        obj_boxes = []
        for box in hugsim_info.get("obj_boxes", []) or []:
            try:
                obj_boxes.append(np.asarray(box, dtype=np.float64).reshape(-1).tolist())
            except Exception:
                continue
        out["obj_boxes"] = obj_boxes

    if bool(terminated or truncated):
        if collision:
            out["terminal_kind"] = "failure"
            out["done_reason"] = "hugsim_collision"
        elif bool(truncated):
            out["terminal_kind"] = "timeout"
            out["done_reason"] = "timeout"
        elif route_completed:
            out["terminal_kind"] = "env_done"
            out["done_reason"] = "route_completed"
        else:
            out["terminal_kind"] = "env_done"
            out["done_reason"] = "hugsim_terminated"
    return out

# Recon cache and obstacle info.
def _load_recon_env_cache(recon_data_root: str | Path, scene_id: int) -> dict[int, dict[str, Any]]:
    path = Path(recon_data_root) / f"{int(scene_id):03d}" / "env_cache.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    if isinstance(payload, dict) and "frames" in payload and isinstance(payload["frames"], dict):
        payload = payload["frames"]
    if isinstance(payload, dict) and "meta" in payload:
        payload = {k: v for k, v in payload.items() if k != "meta"}
    out: dict[int, dict[str, Any]] = {}
    for key, value in (payload.items() if isinstance(payload, dict) else []):
        try:
            out[int(key)] = dict(value)
        except Exception:
            continue
    return out


def _load_recon_ego_pose_xy_by_frame(recon_data_root: str | Path, scene_id: int) -> dict[int, np.ndarray]:
    scene_dir = Path(recon_data_root) / f"{int(scene_id):03d}" / "ego_pose"
    out: dict[int, np.ndarray] = {}
    for path in sorted(scene_dir.glob("*.txt")):
        try:
            pose = np.asarray(np.loadtxt(path), dtype=np.float64).reshape(4, 4)
            out[int(path.stem)] = np.asarray([pose[0, 3], pose[1, 3]], dtype=np.float64)
        except Exception:
            continue
    return out


def _snapshot_for_frame(cache: dict[int, dict[str, Any]], frame_idx: int) -> tuple[int | None, dict[str, Any]]:
    if not cache:
        return None, {}
    frame = int(frame_idx)
    if frame in cache:
        return frame, dict(cache[frame])
    keys = sorted(int(k) for k in cache.keys())
    lower = [k for k in keys if k <= frame]
    chosen = lower[-1] if lower else keys[0]
    return int(chosen), dict(cache[chosen])


def _nearest_cache_frame_by_ego_pose(
    *,
    cache: dict[int, dict[str, Any]],
    ego_pose_xy_by_frame: dict[int, np.ndarray],
    ego_xy: Any,
) -> tuple[int | None, float | None]:
    if not cache or not ego_pose_xy_by_frame:
        return None, None
    try:
        target = np.asarray(ego_xy, dtype=np.float64).reshape(-1)[:2]
        if target.shape[0] != 2:
            return None, None
    except Exception:
        return None, None
    candidates = [int(frame) for frame in cache.keys() if int(frame) in ego_pose_xy_by_frame]
    if not candidates:
        return None, None
    best_frame: int | None = None
    best_dist = float("inf")
    for frame in sorted(candidates):
        dist = float(np.linalg.norm(np.asarray(ego_pose_xy_by_frame[frame], dtype=np.float64) - target))
        if dist < best_dist:
            best_frame = int(frame)
            best_dist = dist
    if best_frame is None:
        return None, None
    return best_frame, best_dist


def _object_poly_xy(obj: dict[str, Any]) -> np.ndarray | None:
    poly = obj.get("poly", None)
    if isinstance(poly, list) and len(poly) >= 3:
        try:
            return np.asarray(poly, dtype=np.float64)
        except Exception:
            return None
    if "translation" in obj and "size" in obj:
        try:
            x, y = np.asarray(obj["translation"], dtype=np.float64).reshape(-1)[:2]
            size = np.asarray(obj["size"], dtype=np.float64).reshape(-1)
            width = float(size[0])
            length = float(size[1] if size.shape[0] > 1 else size[0])
            yaw = float(obj.get("yaw", obj.get("heading", 0.0)))
            c, s = math.cos(yaw), math.sin(yaw)
            offsets = np.asarray(
                [[length * 0.5, width * 0.5], [length * 0.5, -width * 0.5], [-length * 0.5, -width * 0.5], [-length * 0.5, width * 0.5]],
                dtype=np.float64,
            )
            rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
            return (offsets @ rot.T) + np.asarray([[x, y]], dtype=np.float64)
        except Exception:
            return None
    return None


def _compute_front_obstacle_metrics(
    *,
    ego_xy: np.ndarray,
    ego_yaw: float,
    ego_speed_mps: float,
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    heading = np.asarray([math.cos(float(ego_yaw)), math.sin(float(ego_yaw))], dtype=np.float64)
    lateral_axis = np.asarray([-heading[1], heading[0]], dtype=np.float64)
    best: dict[str, Any] | None = None
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        poly = _object_poly_xy(obj)
        if poly is None or int(poly.shape[0]) < 3:
            continue
        center = np.mean(poly, axis=0)
        rel_center = center - np.asarray(ego_xy, dtype=np.float64)
        gap = float(np.dot(rel_center, heading))
        if gap <= 0.0:
            continue
        lateral = float(np.dot(rel_center, lateral_axis))
        velocity = np.asarray(obj.get("velocity", obj.get("velocity_xy", [0.0, 0.0])), dtype=np.float64).reshape(-1)
        obj_forward_v = float(np.dot(velocity[:2], heading)) if velocity.shape[0] >= 2 else 0.0
        closing = max(0.0, float(ego_speed_mps) - obj_forward_v)
        ttc = float(gap / closing) if closing > 1.0e-6 else math.inf
        item = {
            "front_obstacle_available": True,
            "front_obstacle_gap_m": gap,
            "front_obstacle_lateral_m": lateral,
            "front_obstacle_closing_speed_mps": closing,
            "front_obstacle_ttc_s": ttc,
            "front_obstacle_category": str(obj.get("category", obj.get("type", ""))),
        }
        if best is None or gap < float(best["front_obstacle_gap_m"]):
            best = item
    if best is not None:
        return best
    return {
        "front_obstacle_available": False,
        "front_obstacle_gap_m": math.inf,
        "front_obstacle_lateral_m": math.inf,
        "front_obstacle_closing_speed_mps": 0.0,
        "front_obstacle_ttc_s": math.inf,
        "front_obstacle_category": "",
    }


def _detect_polygon_collision_tokens(ego_poly: Any, objects: list[dict[str, Any]]) -> list[str]:
    try:
        ego_arr = np.asarray(ego_poly, dtype=np.float64)
        if ego_arr.ndim != 2 or ego_arr.shape[0] < 3 or ego_arr.shape[1] != 2:
            return []
        ego_shape = Polygon(ego_arr.tolist())
    except Exception:
        return []
    tokens: list[str] = []
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        poly = obj.get("poly", None)
        if not (isinstance(poly, list) and len(poly) >= 3):
            continue
        try:
            obj_shape = Polygon(np.asarray(poly, dtype=np.float64).tolist())
        except Exception:
            continue
        if bool(ego_shape.intersects(obj_shape)):
            tokens.append(str(obj.get("token", obj.get("id", f"obj_{idx}"))))
    return tokens



class HUGSIMFifoClient:
    def __init__(
        self,
        *,
        hugsim_repo: str | Path,
        scenario_path: str | Path,
        base_path: str | Path | None,
        camera_path: str | Path | None,
        kinematic_path: str | Path | None,
        output_dir: str | Path,
        pixi_cmd: str = "pixi",
        runner_path: str | Path = FIFO_RUNNER_DEFAULT,
        ad: str = "sparsedrive_v2",
        fifo_timeout_s: float = 300.0,
        fifo_poll_interval_s: float = 0.2,
        cuda: int | None = None,
    ) -> None:
        self.hugsim_repo = str(resolve_hugsim_path(str(hugsim_repo)))
        self.scenario_path = str(resolve_hugsim_path(str(scenario_path)))
        repo = Path(self.hugsim_repo)
        self.base_path = str(
            resolve_hugsim_path(str(base_path))
            if base_path is not None
            else repo / "configs" / "sim" / "nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml"
        )
        self.camera_path = str(
            resolve_hugsim_path(str(camera_path))
            if camera_path is not None
            else repo / "configs" / "sim" / "nuscenes_camera.yaml"
        )
        self.kinematic_path = str(
            resolve_hugsim_path(str(kinematic_path))
            if kinematic_path is not None
            else repo / "configs" / "sim" / "kinematic.yaml"
        )
        self.output_dir = Path(output_dir)
        self.pixi_cmd = str(pixi_cmd)
        self.runner_path = str(runner_path)
        self.ad = str(ad)
        self.fifo_timeout_s = float(fifo_timeout_s)
        self.fifo_poll_interval_s = float(fifo_poll_interval_s)
        self.cuda = None if cuda is None else int(cuda)
        self.process: subprocess.Popen[Any] | None = None
        self._episode_done = False
        self.obs_pipe = self.output_dir / "obs_pipe"
        self.plan_pipe = self.output_dir / "plan_pipe"

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._clear_stale_session_files()
        cmd = [
            self.pixi_cmd,
            "run",
            "python",
            self.runner_path,
            "--scenario_path",
            self.scenario_path,
            "--base_path",
            self.base_path,
            "--camera_path",
            self.camera_path,
            "--kinematic_path",
            self.kinematic_path,
            "--output_dir",
            str(self.output_dir),
            "--ad",
            self.ad,
            "--fifo_timeout_s",
            str(self.fifo_timeout_s),
            "--fifo_poll_interval_s",
            str(self.fifo_poll_interval_s),
        ]
        env = os.environ.copy()
        if self.cuda is not None and self.cuda >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(int(self.cuda))
        log_handle = (self.output_dir / "hugsim_fifo_runner.log").open("ab")
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=self.hugsim_repo,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        finally:
            log_handle.close()

    def _clear_stale_session_files(self) -> None:
        for path in (self.obs_pipe, self.plan_pipe, self.output_dir / "status.json"):
            if not (path.exists() or path.is_symlink()):
                continue
            if path.is_dir():
                raise RuntimeError(f"Cannot replace HUGSIM FIFO session path because it is a directory: {path}")
            path.unlink()

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._episode_done:
            self.close()
        self.start()
        obs, info = self._read_obs_info()
        self._episode_done = False
        return dict(obs), dict(info)

    def step(self, plan_traj: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.process is None:
            raise RuntimeError("HUGSIM FIFO client step called before start")
        from framework.env_wrapper.fifo_io import write_fifo_payload

        write_fifo_payload(
            self.plan_pipe,
            np.asarray(plan_traj, dtype=np.float32),
            process=self.process,
            timeout_s=self.fifo_timeout_s,
            poll_interval_s=self.fifo_poll_interval_s,
        )
        obs, info = self._read_obs_info()
        terminated = bool(info.get("terminated", False) or info.get("done", False))
        truncated = bool(info.get("truncated", False))
        self._episode_done = bool(terminated or truncated)
        reward = float(info.get("reward", 0.0))
        return dict(obs), reward, terminated, truncated, dict(info)

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                from framework.env_wrapper.fifo_io import write_fifo_payload

                write_fifo_payload(
                    self.plan_pipe,
                    "STOP",
                    process=process,
                    timeout_s=min(self.fifo_timeout_s, 2.0),
                    poll_interval_s=min(self.fifo_poll_interval_s, 0.05),
                )
            except Exception:
                pass
            try:
                process.terminate()
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        self.process = None
        self._episode_done = False

    def _read_obs_info(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.process is None:
            raise RuntimeError("HUGSIM FIFO client is not started")
        from framework.env_wrapper.fifo_io import read_fifo_payload

        payload = read_fifo_payload(
            self.obs_pipe,
            process=self.process,
            timeout_s=self.fifo_timeout_s,
            poll_interval_s=self.fifo_poll_interval_s,
        )
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise RuntimeError(f"Expected FIFO payload (obs, info), got {type(payload).__name__}")
        obs, info = payload
        return dict(obs), dict(info)


class HUGSIMReconEnv:
    def __init__(
        self,
        *,
        scenario_name: str,
        scenario_path: str | Path,
        scene_index: Any,
        reward_cfg: dict[str, Any] | None = None,
        output_root: str | Path = "outputs/hugsim_rl",
        hugsim_repo: str | Path = HUGSIM_REPO_DEFAULT,
        base_path: str | Path | None = None,
        camera_path: str | Path | None = None,
        kinematic_path: str | Path | None = None,
        recon_data_root: str | Path = RECON_DATA_ROOT_DEFAULT,
        hugsim_model_base: str | Path | None = None,
        alignment_enabled: bool = True,
        alignment_max_rmse_m: float = 2.0,
        use_recon_cache_objects: bool = True,
        use_hugsim_inserted_objects: bool = True,
        launch_mode: str = "fifo",
        pixi_cmd: str = "pixi",
        fifo_timeout_s: float = 300.0,
        fifo_poll_interval_s: float = 0.2,
        fifo_runner_path: str | Path = FIFO_RUNNER_DEFAULT,
        cuda: int | None = None,
        min_gt_route_points: int = 2,
    ) -> None:
        self.official_scene_name = str(scenario_name)
        self.scenario_path = str(scenario_path)
        self.scene_index = scene_index
        self.reward_cfg = reward_cfg or {}
        self.hugsim_repo = str(hugsim_repo)
        self.launch_mode = str(launch_mode).strip().lower()
        if self.launch_mode != "fifo":
            raise ValueError(f"Unsupported HUGSIM launch_mode: {launch_mode!r}; only 'fifo' is supported")
        self.recon_data_root = Path(recon_data_root)
        self.hugsim_model_base = None if hugsim_model_base is None else Path(hugsim_model_base)
        self.alignment_enabled = bool(alignment_enabled)
        self.alignment_max_rmse_m = float(alignment_max_rmse_m)
        self.use_recon_cache_objects = bool(use_recon_cache_objects)
        self.use_hugsim_inserted_objects = bool(use_hugsim_inserted_objects)
        self._reward_computer = TrackingRewardComputer(self.reward_cfg)
        self._reward_proxy = HUGSIMRewardProxy(recon_data_root=recon_data_root)
        self._alignment_cache: dict[tuple[str, int], HUGSIMReconAlignment] = {}
        self._alignment: HUGSIMReconAlignment | None = None
        self._recon_env_cache: dict[int, dict[int, dict[str, Any]]] = {}
        self._recon_ego_pose_xy_cache: dict[int, dict[int, np.ndarray]] = {}
        self._hugsim_step_idx = 0
        self._last_hugsim_obs = None
        self._last_hugsim_info = None
        self._external_plan_local_xyyaw = None
        self.min_gt_route_points = max(0, int(min_gt_route_points))
        scenario_output_name = Path(self.scenario_path).stem or self.official_scene_name
        output_dir = (Path(output_root) / scenario_output_name).resolve()
        self.env = HUGSIMFifoClient(
            hugsim_repo=self.hugsim_repo,
            scenario_path=self.scenario_path,
            base_path=base_path,
            camera_path=camera_path,
            kinematic_path=kinematic_path,
            output_dir=output_dir,
            pixi_cmd=pixi_cmd,
            runner_path=fifo_runner_path,
            fifo_timeout_s=fifo_timeout_s,
            fifo_poll_interval_s=fifo_poll_interval_s,
            cuda=cuda,
        )

    def set_external_plan_local_xyyaw(self, plan: Any) -> None:
        self._external_plan_local_xyyaw = None if plan is None else np.asarray(plan, dtype=np.float32)

    def _maybe_mark_short_gt_route_done(
        self,
        info: dict[str, Any],
        mapping: HUGSIMFrameMapping,
        *,
        terminal_metadata_allowed: bool,
    ) -> bool:
        counter = getattr(self.scene_index, "remaining_future_sample_count", None)
        if not callable(counter) or self.min_gt_route_points <= 0:
            return False
        try:
            count = int(counter(str(mapping.official_scene_name), int(mapping.sample_index)))
        except Exception:
            return False
        if count >= int(self.min_gt_route_points):
            return False
        info["remaining_future_sample_count"] = int(count)
        if bool(terminal_metadata_allowed):
            info["terminal_kind"] = "env_done"
            info["done_reason"] = "short_gt_route"
        return True

    def _get_alignment(self, mapping: HUGSIMFrameMapping) -> HUGSIMReconAlignment:
        key = (str(mapping.official_scene_name), int(mapping.recon_scene_id))
        if key in self._alignment_cache:
            self._alignment = self._alignment_cache[key]
            return self._alignment
        if not self.alignment_enabled:
            alignment = HUGSIMReconAlignment(
                official_scene_name=str(mapping.official_scene_name),
                recon_scene_id=int(mapping.recon_scene_id),
                transform=Sim2Transform(
                    scale=1.0,
                    rotation=np.eye(2, dtype=np.float64),
                    translation_xy=np.zeros((2,), dtype=np.float64),
                    rmse_m=-1.0,
                ),
                valid=False,
                reason="disabled",
            )
        else:
            alignment = build_hugsim_recon_alignment(
                official_scene_name=str(mapping.official_scene_name),
                recon_scene_id=int(mapping.recon_scene_id),
                hugsim_model_base=self.hugsim_model_base,
                recon_data_root=self.recon_data_root,
                max_rmse_m=self.alignment_max_rmse_m,
            )
        self._alignment_cache[key] = alignment
        self._alignment = alignment
        return alignment

    def _alignment_info(self, alignment: HUGSIMReconAlignment) -> dict[str, Any]:
        rmse = float(alignment.transform.rmse_m)
        return {
            "hugsim_recon_alignment_valid": bool(alignment.valid),
            "hugsim_recon_alignment_rmse_m": rmse if math.isfinite(rmse) else -1.0,
            "hugsim_recon_alignment_scale": float(alignment.transform.scale),
            "hugsim_recon_alignment_reason": str(alignment.reason),
            "hugsim_recon_alignment_mode": str(getattr(alignment, "mode", "global")),
        }

    def _alignment_for_step(
        self,
        *,
        alignment: HUGSIMReconAlignment,
        mapping: HUGSIMFrameMapping,
        hugsim_info: dict[str, Any],
    ) -> HUGSIMReconAlignment:
        if bool(alignment.valid) or not self.alignment_enabled or "ego_box" not in hugsim_info:
            return alignment
        try:
            ego_xy = np.asarray(hugsim_info["ego_box"], dtype=np.float64).reshape(-1)[:2]
        except Exception:
            return alignment
        local = build_local_hugsim_recon_alignment(
            official_scene_name=str(mapping.official_scene_name),
            recon_scene_id=int(mapping.recon_scene_id),
            hugsim_model_base=self.hugsim_model_base,
            recon_data_root=self.recon_data_root,
            hugsim_xy=ego_xy,
            recon_frame_idx=int(mapping.frame_idx),
            base_transform=alignment.transform,
        )
        if bool(local.valid):
            return local
        return alignment

    def _reward_pose_from_alignment(
        self,
        *,
        alignment: HUGSIMReconAlignment,
        hugsim_info: dict[str, Any],
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        if not bool(alignment.valid) or "ego_box" not in hugsim_info:
            return None, {"reward_pose_source": "hugsim_native_pose"}
        try:
            reward_pose = transform_hugsim_ego_box_to_reward_pose(hugsim_info["ego_box"], alignment.transform)
            xy = np.asarray(reward_pose[:3, 3][[0, 2]], dtype=np.float64)
            return reward_pose, {
                "reward_pose_source": "hugsim_recon_alignment",
                "recon_global_ego_xy": [float(xy[0]), float(xy[1])],
                "recon_global_ego_yaw": float(alignment.transform.transform_yaw(float(np.asarray(hugsim_info["ego_box"]).reshape(-1)[6]))),
            }
        except Exception as exc:
            return None, {"reward_pose_source": "hugsim_native_pose", "hugsim_recon_alignment_pose_error": str(exc)}

    def _augment_aligned_context(
        self,
        *,
        info: dict[str, Any],
        hugsim_info: dict[str, Any],
        mapping: HUGSIMFrameMapping,
        alignment: HUGSIMReconAlignment,
    ) -> dict[str, Any]:
        out = dict(info)
        out.update(self._alignment_info(alignment))
        reward_pose, pose_info = self._reward_pose_from_alignment(alignment=alignment, hugsim_info=hugsim_info)
        out.update(pose_info)
        ego_poly_recon: list[list[float]] | None = None
        hugsim_objects: list[dict[str, Any]] = []
        if bool(alignment.valid) and reward_pose is not None and "ego_box" in hugsim_info:
            try:
                ego_poly_recon = transform_hugsim_box_to_recon_poly(
                    hugsim_info["ego_box"], alignment.transform
                ).astype(float).tolist()
                out["hugsim_ego_box_recon_global_poly"] = ego_poly_recon
            except Exception:
                pass
            if self.use_hugsim_inserted_objects:
                hugsim_objects = transform_hugsim_boxes_to_recon_objects(
                    hugsim_info.get("obj_boxes", []), alignment.transform
                )
                out["hugsim_obj_boxes_recon_global"] = hugsim_objects

        objects: list[dict[str, Any]] = []
        if self.use_recon_cache_objects:
            scene_id = int(mapping.recon_scene_id)
            if scene_id not in self._recon_env_cache:
                self._recon_env_cache[scene_id] = _load_recon_env_cache(self.recon_data_root, scene_id)
            cache = self._recon_env_cache.get(scene_id, {})
            time_frame_used, _time_snap = _snapshot_for_frame(cache, int(mapping.frame_idx))
            frame_used, snap = time_frame_used, _time_snap
            frame_source = "time_mapping"
            pose_dist_m: float | None = None
            if reward_pose is not None:
                if scene_id not in self._recon_ego_pose_xy_cache:
                    self._recon_ego_pose_xy_cache[scene_id] = _load_recon_ego_pose_xy_by_frame(
                        self.recon_data_root, scene_id
                    )
                ego_xy = np.asarray(reward_pose[:3, 3][[0, 2]], dtype=np.float64)
                nearest_frame, nearest_dist = _nearest_cache_frame_by_ego_pose(
                    cache=cache,
                    ego_pose_xy_by_frame=self._recon_ego_pose_xy_cache.get(scene_id, {}),
                    ego_xy=ego_xy,
                )
                if nearest_frame is not None:
                    frame_used = int(nearest_frame)
                    snap = dict(cache[int(nearest_frame)])
                    frame_source = "nearest_pose"
                    pose_dist_m = float(nearest_dist) if nearest_dist is not None else None
            out["recon_cache_frame_idx"] = -1 if frame_used is None else int(frame_used)
            out["recon_cache_time_frame_idx"] = -1 if time_frame_used is None else int(time_frame_used)
            out["recon_cache_frame_source"] = frame_source
            token_for_frame = getattr(self.scene_index, "sample_token_for_frame", None)
            if callable(token_for_frame):
                if time_frame_used is not None:
                    time_token = token_for_frame(scene_id, int(time_frame_used))
                    if time_token is not None:
                        out["recon_cache_time_sample_token"] = str(time_token)
                if frame_used is not None:
                    frame_token = token_for_frame(scene_id, int(frame_used))
                    if frame_token is not None:
                        out["recon_cache_sample_token"] = str(frame_token)
                        out["grpo_gt_sample_token"] = str(frame_token)
                        out["grpo_gt_frame_idx"] = int(frame_used)
            if pose_dist_m is not None:
                out["recon_cache_frame_pose_dist_m"] = pose_dist_m
            raw_objects = snap.get("dynamic_objects", []) if isinstance(snap, dict) else []
            objects = [dict(obj) for obj in raw_objects if isinstance(obj, dict)]
            for obj in objects:
                obj.setdefault("source", "recon_cache")
            out["recon_cache_dynamic_objects"] = objects

        aligned_objects = [*hugsim_objects, *objects]
        collision_tokens: list[str] = []
        if ego_poly_recon is not None:
            collision_tokens = _detect_polygon_collision_tokens(ego_poly_recon, aligned_objects)
            if collision_tokens:
                out["collision"] = True
                out["dynamic_collision"] = True
                out["hugsim_aligned_collision"] = True
                out["hugsim_aligned_collision_tokens"] = collision_tokens

        if reward_pose is not None:
            ego_xy = np.asarray(reward_pose[:3, 3][[0, 2]], dtype=np.float64)
            ego_yaw = float(alignment.transform.transform_yaw(float(np.asarray(hugsim_info.get("ego_box", [0, 0, 0, 0, 0, 0, 0])).reshape(-1)[6])))
            try:
                ego_speed = float(hugsim_info.get("ego_velo", 0.0))
            except Exception:
                ego_speed = 0.0
            out.update(
                _compute_front_obstacle_metrics(
                    ego_xy=ego_xy,
                    ego_yaw=ego_yaw,
                    ego_speed_mps=ego_speed,
                    objects=aligned_objects,
                )
            )
        return out

    def reset(self, scene: int | None = None, *, start_frame: int | None = None, step_frames: int | None = None):
        del scene, start_frame, step_frames
        self._hugsim_step_idx = 0
        self._reward_computer.reset()
        hugsim_obs, hugsim_info = self.env.reset()
        self._last_hugsim_obs = hugsim_obs
        self._last_hugsim_info = hugsim_info
        mapping = self.scene_index.map_time(self.official_scene_name, float(hugsim_info.get("timestamp", 0.0)))
        obs = build_recondreamer_obs_from_hugsim(
            hugsim_obs=hugsim_obs,
            hugsim_info=hugsim_info,
            mapping=mapping,
            hugsim_step_idx=self._hugsim_step_idx,
        )
        alignment = self._get_alignment(mapping)
        alignment = self._alignment_for_step(alignment=alignment, mapping=mapping, hugsim_info=dict(hugsim_info))
        reset_info: dict[str, Any] = {
            "scene": int(mapping.recon_scene_id),
            "now_frame": int(mapping.frame_idx),
            "sample_token": mapping.sample_token,
        }
        reset_info = self._augment_aligned_context(
            info=reset_info,
            hugsim_info=dict(hugsim_info),
            mapping=mapping,
            alignment=alignment,
        )
        reward_pose, _pose_info = self._reward_pose_from_alignment(alignment=alignment, hugsim_info=dict(hugsim_info))
        self._reward_proxy.update_from_hugsim_info(
            recon_scene_id=int(mapping.recon_scene_id),
            frame_idx=int(mapping.frame_idx),
            hugsim_info=dict(hugsim_info),
            reward_pose=reward_pose,
        )
        return obs, reset_info

    def _plan_from_action(self, action: Any) -> np.ndarray:
        if isinstance(self._external_plan_local_xyyaw, np.ndarray) and self._external_plan_local_xyyaw.ndim == 2:
            plan = np.asarray(self._external_plan_local_xyyaw[:, :2], dtype=np.float32)
            hugsim_plan = plan[:, [1, 0]].copy()
            hugsim_plan[:, 0] *= -1.0
            self._external_plan_local_xyyaw = None
            return hugsim_plan
        if isinstance(action, (tuple, list)) and len(action) >= 3:
            first = np.asarray([[float(action[0]), float(action[1])]], dtype=np.float32)
            hugsim_plan = first[:, [1, 0]].copy()
            hugsim_plan[:, 0] *= -1.0
            return hugsim_plan
        raise ValueError(f"Unsupported HUGSIM action format: {action!r}")

    def step(self, action: Any):
        if self._last_hugsim_info is None:
            raise RuntimeError("HUGSIMReconEnv.step called before reset")
        plan_traj = self._plan_from_action(action)
        hugsim_obs, base_reward, terminated, truncated, hugsim_info = self.env.step(plan_traj)
        self._hugsim_step_idx += 1
        self._last_hugsim_obs = hugsim_obs
        self._last_hugsim_info = hugsim_info
        mapping = self.scene_index.map_time(self.official_scene_name, float(hugsim_info.get("timestamp", 0.0)))
        obs = build_recondreamer_obs_from_hugsim(
            hugsim_obs=hugsim_obs,
            hugsim_info=hugsim_info,
            mapping=mapping,
            hugsim_step_idx=self._hugsim_step_idx,
        )
        info = {
            "scene": int(mapping.recon_scene_id),
            "scene_id": int(mapping.recon_scene_id),
            "now_frame": int(mapping.frame_idx),
            "frame_idx": int(mapping.frame_idx),
            "sample_token": str(mapping.sample_token),
            "hugsim_timestamp": float(mapping.hugsim_relative_time_s),
            "hugsim_base_reward": float(base_reward),
            "official_scene_name": str(mapping.official_scene_name),
        }
        info = _augment_hugsim_reward_info(
            info=info,
            hugsim_info=dict(hugsim_info),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        alignment = self._get_alignment(mapping)
        alignment = self._alignment_for_step(alignment=alignment, mapping=mapping, hugsim_info=dict(hugsim_info))
        info = self._augment_aligned_context(
            info=info,
            hugsim_info=dict(hugsim_info),
            mapping=mapping,
            alignment=alignment,
        )
        collision_done = bool(info.get("collision", False) or info.get("static_collision", False) or info.get("dynamic_collision", False))
        short_gt_route_done = self._maybe_mark_short_gt_route_done(
            info,
            mapping,
            terminal_metadata_allowed=not bool(collision_done or truncated),
        )
        if collision_done and not bool(truncated):
            terminated = True
        if bool(short_gt_route_done) and not bool(truncated):
            terminated = True
        if bool(terminated or truncated) and "terminal_kind" not in info:
            if collision_done:
                info["terminal_kind"] = "failure"
                info["done_reason"] = "hugsim_collision"
            elif bool(truncated):
                info["terminal_kind"] = "timeout"
                info["done_reason"] = "timeout"
            elif bool(info.get("route_completed", False)):
                info["terminal_kind"] = "env_done"
                info["done_reason"] = "route_completed"
            else:
                info["terminal_kind"] = "env_done"
                info["done_reason"] = "hugsim_terminated"
        reward_pose, _pose_info = self._reward_pose_from_alignment(alignment=alignment, hugsim_info=dict(hugsim_info))
        self._reward_proxy.update_from_hugsim_info(
            recon_scene_id=int(mapping.recon_scene_id),
            frame_idx=int(mapping.frame_idx),
            hugsim_info=dict(hugsim_info),
            reward_pose=reward_pose,
        )
        reward_result = self._reward_computer.compute(
            env=self._reward_proxy,
            info=info,
            step_idx=int(mapping.frame_idx),
            done=bool(terminated or truncated),
        )
        return obs, float(reward_result.reward), bool(terminated), bool(truncated), reward_result.info

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
