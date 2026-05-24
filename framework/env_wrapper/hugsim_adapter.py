from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from framework.env_wrapper.hugsim_scene_index import HUGSIMFrameMapping
from framework.rewards import TrackingRewardComputer


HUGSIM_REPO_DEFAULT = "/root/clone/HUGSIM-ORI"
FRONT_CAMERA_MAP = {
    "CAM_FRONT_LEFT": ("front_left", 1),
    "CAM_FRONT": ("front", 0),
    "CAM_FRONT_RIGHT": ("front_right", 2),
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

    def update_from_hugsim_info(self, *, recon_scene_id: int, frame_idx: int, hugsim_info: dict[str, Any]) -> None:
        self.scene = int(recon_scene_id)
        self.now_frame = int(frame_idx)
        self._ensure_expert_loaded(int(recon_scene_id))
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
                poses.append(np.asarray(np.loadtxt(path), dtype=np.float64).reshape(4, 4))
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

    for hugsim_cam, (obs_key, cam_idx) in FRONT_CAMERA_MAP.items():
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


def _ensure_hugsim_repo_on_path(hugsim_repo: str | Path = HUGSIM_REPO_DEFAULT) -> None:
    repo = str(hugsim_repo)
    sim_dir = str(Path(repo) / "sim")
    for path in [repo, sim_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_traj2control(hugsim_repo: str | Path = HUGSIM_REPO_DEFAULT):
    _ensure_hugsim_repo_on_path(hugsim_repo)
    from sim.utils.sim_utils import traj2control as fn  # type: ignore

    return fn


traj2control = _load_traj2control


def execute_hugsim_control_horizon(
    *,
    env: Any,
    plan_traj: np.ndarray,
    initial_info: dict[str, Any],
    substeps_per_rl_step: int = 2,
    hugsim_repo: str | Path = HUGSIM_REPO_DEFAULT,
) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    if callable(traj2control) and getattr(traj2control, "__name__", "") == "_load_traj2control":
        control_fn = traj2control(hugsim_repo)
    else:
        control_fn = traj2control
    acc, steer_rate = control_fn(np.asarray(plan_traj, dtype=np.float32), dict(initial_info))
    action = {"acc": float(acc), "steer_rate": float(steer_rate)}

    obs: dict[str, Any] = {}
    info: dict[str, Any] = dict(initial_info)
    total_reward = 0.0
    terminated = False
    truncated = False
    for _ in range(max(1, int(substeps_per_rl_step))):
        obs, reward, terminated, truncated, info = env.step(dict(action))
        total_reward += float(reward)
        if bool(terminated or truncated):
            break
    return obs, float(total_reward), bool(terminated), bool(truncated), info


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


def create_hugsim_env(
    *,
    scenario_path: str | Path,
    output_dir: str | Path,
    hugsim_repo: str | Path = HUGSIM_REPO_DEFAULT,
    base_path: str | Path | None = None,
    camera_path: str | Path | None = None,
    kinematic_path: str | Path | None = None,
    ad: str = "sparsedrive_v2",
) -> Any:
    _ensure_hugsim_repo_on_path(hugsim_repo)
    import gymnasium  # type: ignore
    import hugsim_env  # noqa: F401
    from sim.utils.config_loader import load_closed_loop_cfg  # type: ignore

    repo = Path(hugsim_repo)
    cfg, _output = load_closed_loop_cfg(
        scenario_path=str(scenario_path),
        base_path=str(base_path or repo / "configs" / "sim" / "nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml"),
        camera_path=str(camera_path or repo / "configs" / "sim" / "nuscenes_camera.yaml"),
        kinematic_path=str(kinematic_path or repo / "configs" / "sim" / "kinematic.yaml"),
        ad=str(ad),
    )
    output = str(output_dir)
    Path(output).mkdir(parents=True, exist_ok=True)
    return gymnasium.make("hugsim_env/HUGSim-v0", cfg=cfg, output=output)


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
        self.hugsim_repo = str(hugsim_repo)
        self.scenario_path = str(scenario_path)
        repo = Path(self.hugsim_repo)
        self.base_path = str(base_path or repo / "configs" / "sim" / "nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml")
        self.camera_path = str(camera_path or repo / "configs" / "sim" / "nuscenes_camera.yaml")
        self.kinematic_path = str(kinematic_path or repo / "configs" / "sim" / "kinematic.yaml")
        self.output_dir = Path(output_dir)
        self.pixi_cmd = str(pixi_cmd)
        self.runner_path = str(runner_path)
        self.ad = str(ad)
        self.fifo_timeout_s = float(fifo_timeout_s)
        self.fifo_poll_interval_s = float(fifo_poll_interval_s)
        self.cuda = None if cuda is None else int(cuda)
        self.process: subprocess.Popen[Any] | None = None
        self.obs_pipe = self.output_dir / "obs_pipe"
        self.plan_pipe = self.output_dir / "plan_pipe"

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
        self.process = subprocess.Popen(cmd, cwd=self.hugsim_repo, env=env)

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self.start()
        obs, info = self._read_obs_info()
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
        substeps_per_rl_step: int = 2,
        recon_data_root: str | Path = RECON_DATA_ROOT_DEFAULT,
        launch_mode: str = "direct",
        pixi_cmd: str = "pixi",
        fifo_timeout_s: float = 300.0,
        fifo_poll_interval_s: float = 0.2,
        fifo_runner_path: str | Path = FIFO_RUNNER_DEFAULT,
        cuda: int | None = None,
    ) -> None:
        self.official_scene_name = str(scenario_name)
        self.scenario_path = str(scenario_path)
        self.scene_index = scene_index
        self.reward_cfg = reward_cfg or {}
        self.hugsim_repo = str(hugsim_repo)
        self.substeps_per_rl_step = int(substeps_per_rl_step)
        self.launch_mode = str(launch_mode).strip().lower()
        self._reward_computer = TrackingRewardComputer(self.reward_cfg)
        self._reward_proxy = HUGSIMRewardProxy(recon_data_root=recon_data_root)
        self._hugsim_step_idx = 0
        self._last_hugsim_obs = None
        self._last_hugsim_info = None
        self._external_plan_local_xyyaw = None
        output_dir = (Path(output_root) / self.official_scene_name).resolve()
        if self.launch_mode == "fifo":
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
        elif self.launch_mode == "direct":
            self.env = create_hugsim_env(
                scenario_path=self.scenario_path,
                output_dir=output_dir,
                hugsim_repo=self.hugsim_repo,
                base_path=base_path,
                camera_path=camera_path,
                kinematic_path=kinematic_path,
            )
        else:
            raise ValueError(f"Unsupported HUGSIM launch_mode: {launch_mode!r}")

    def set_external_plan_local_xyyaw(self, plan: Any) -> None:
        self._external_plan_local_xyyaw = None if plan is None else np.asarray(plan, dtype=np.float32)

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
        self._reward_proxy.update_from_hugsim_info(
            recon_scene_id=int(mapping.recon_scene_id),
            frame_idx=int(mapping.frame_idx),
            hugsim_info=dict(hugsim_info),
        )
        return obs, {"scene": int(mapping.recon_scene_id), "now_frame": int(mapping.frame_idx), "sample_token": mapping.sample_token}

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
        if self.launch_mode == "fifo":
            hugsim_obs, base_reward, terminated, truncated, hugsim_info = self.env.step(plan_traj)
            self._hugsim_step_idx += 1
        else:
            hugsim_obs, base_reward, terminated, truncated, hugsim_info = execute_hugsim_control_horizon(
                env=self.env,
                plan_traj=plan_traj,
                initial_info=dict(self._last_hugsim_info),
                substeps_per_rl_step=self.substeps_per_rl_step,
                hugsim_repo=self.hugsim_repo,
            )
            self._hugsim_step_idx += self.substeps_per_rl_step
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
        self._reward_proxy.update_from_hugsim_info(
            recon_scene_id=int(mapping.recon_scene_id),
            frame_idx=int(mapping.frame_idx),
            hugsim_info=dict(hugsim_info),
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
