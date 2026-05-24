from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from framework.env_wrapper.hugsim_scene_index import HUGSIMFrameMapping


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
    ) -> None:
        self.official_scene_name = str(scenario_name)
        self.scenario_path = str(scenario_path)
        self.scene_index = scene_index
        self.reward_cfg = reward_cfg or {}
        self.hugsim_repo = str(hugsim_repo)
        self.substeps_per_rl_step = int(substeps_per_rl_step)
        self._hugsim_step_idx = 0
        self._last_hugsim_obs = None
        self._last_hugsim_info = None
        self._external_plan_local_xyyaw = None
        output_dir = Path(output_root) / self.official_scene_name
        self.env = create_hugsim_env(
            scenario_path=self.scenario_path,
            output_dir=output_dir,
            hugsim_repo=self.hugsim_repo,
            base_path=base_path,
            camera_path=camera_path,
            kinematic_path=kinematic_path,
        )

    def set_external_plan_local_xyyaw(self, plan: Any) -> None:
        self._external_plan_local_xyyaw = None if plan is None else np.asarray(plan, dtype=np.float32)

    def reset(self, scene: int | None = None, *, start_frame: int | None = None, step_frames: int | None = None):
        del scene, start_frame, step_frames
        self._hugsim_step_idx = 0
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
        return obs, float(base_reward), bool(terminated), bool(truncated), info
