#!/usr/bin/env python3
"""Generate one closed-loop rollout video with DiffusionDriveV2 in ReconSimulator."""

from __future__ import annotations

import argparse
import csv
import cv2
import importlib
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

from framework.utils.repo_paths import resolve_ego_ads_subdir, resolve_repo_path


_DEFAULT_CKPT = os.path.join(resolve_ego_ads_subdir("DiffusionDriveV2"), "ckpt", "diffusiondrivev2_rl.ckpt")
_DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "outputs", "visualize", "diffusiondriveV2")
_DDV2_ROOT = resolve_ego_ads_subdir("DiffusionDriveV2")


def _ensure_torch_scheduler_compat() -> None:
    lr_scheduler = torch.optim.lr_scheduler
    if hasattr(lr_scheduler, "LRScheduler"):
        return
    fallback = getattr(lr_scheduler, "_LRScheduler", None)
    if fallback is not None:
        setattr(lr_scheduler, "LRScheduler", fallback)


def _force_import_diffusiondrive_v2_modules() -> tuple[type, type]:
    if _DDV2_ROOT not in sys.path:
        sys.path.insert(0, _DDV2_ROOT)

    _ensure_torch_scheduler_compat()

    try:
        from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_agent import Diffusiondrivev2_Rl_Agent
        from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_config import TransfuserConfig

        return Diffusiondrivev2_Rl_Agent, TransfuserConfig
    except Exception:
        pass

    stale_keys = [k for k in list(sys.modules.keys()) if k == "navsim" or k.startswith("navsim.")]
    for key in stale_keys:
        try:
            del sys.modules[key]
        except Exception:
            pass

    importlib.invalidate_caches()
    _ensure_torch_scheduler_compat()
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_agent import Diffusiondrivev2_Rl_Agent
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_config import TransfuserConfig

    return Diffusiondrivev2_Rl_Agent, TransfuserConfig


def _lazy_import_env() -> Any:
    try:
        from framework.env_wrapper import RLReconEnv  # type: ignore

        return RLReconEnv
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "Missing runtime dependency for DiffusionDriveV2 rollout. "
            f"Import failed on module: {missing}. Activate project env and retry."
        ) from e


class _DiffusionDriveV2Inferencer:
    def __init__(
        self,
        *,
        ckpt_path: str,
        x_anchor: int,
        y_anchor: int,
        device: str | None = None,
        execute_mode: str = "nearest_anchor",
        rl_lr: float = 1e-5,
        moving_projection_min_norm_m: float = 0.10,
    ) -> None:
        Diffusiondrivev2_Rl_Agent, TransfuserConfig = _force_import_diffusiondrive_v2_modules()

        self.x_anchor = int(x_anchor)
        self.y_anchor = int(y_anchor)
        self.ckpt_path = str(ckpt_path)
        self._device_override = device
        exec_mode = str(execute_mode).strip().lower().replace("-", "_")
        if exec_mode in {"anchor", "nearest", "nearest_anchor"}:
            exec_mode = "nearest_anchor"
        elif exec_mode in {"continuous", "first_step", "step1", "traj_first_step"}:
            exec_mode = "first_step"
        else:
            raise ValueError(f"Unsupported execute_mode: {execute_mode}")
        self._execute_mode = exec_mode
        self._moving_projection_min_norm_m = max(0.0, float(moving_projection_min_norm_m))

        cfg = TransfuserConfig()
        cfg.bkb_path = os.path.join(_DDV2_ROOT, "ckpt", "resnet34.a1_in1k", "pytorch_model.bin")
        cfg.plan_anchor_path = os.path.join(_DDV2_ROOT, "ckpt", "resnet34.a1_in1k", "kmeans_navsim_traj_20.npy")
        self._agent = Diffusiondrivev2_Rl_Agent(config=cfg, lr=float(rl_lr), checkpoint_path=self.ckpt_path)
        self.to(self.device)

        self._anchor_exec_xy: np.ndarray | None = None
        self._anchor_exec_yaw: np.ndarray | None = None
        self._anchor_exec_norm: np.ndarray | None = None
        self._anchor_mask: np.ndarray | None = None

    @property
    def device(self) -> torch.device:
        if self._device_override:
            try:
                return torch.device(str(self._device_override))
            except Exception:
                pass
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def to(self, device: str | torch.device) -> "_DiffusionDriveV2Inferencer":
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        self._device_override = str(dev)
        try:
            self._agent._transfuser_model.to(dev)
        except Exception:
            pass
        try:
            self._agent.to(dev)
        except Exception:
            pass
        return self

    @staticmethod
    def _build_camera_feature(observation: Dict[str, np.ndarray]) -> torch.Tensor:
        keys = ["front_left", "front", "front_right"]
        imgs: List[np.ndarray] = []
        for key in keys:
            image = observation.get(key, None)
            if image is not None:
                imgs.append(np.asarray(image))
                continue
            fallback = observation.get("front", None)
            if fallback is None and imgs:
                fallback = imgs[0]
            if fallback is None:
                raise ValueError("No camera images available in observation")
            imgs.append(np.asarray(fallback))

        def safe_crop(img: np.ndarray, mode: str) -> np.ndarray:
            h, w = img.shape[:2]
            top, bottom = 28, 28
            left_lr, right_lr = 416, 416
            if mode in {"l", "r"}:
                y0, y1 = (top, h - bottom) if (h > top + bottom) else (0, h)
                x0, x1 = (left_lr, w - right_lr) if (w > left_lr + right_lr) else (0, w)
                return img[y0:y1, x0:x1]
            y0, y1 = (top, h - bottom) if (h > top + bottom) else (0, h)
            return img[y0:y1]

        l0 = safe_crop(imgs[0], "l")
        f0 = safe_crop(imgs[1], "f")
        r0 = safe_crop(imgs[2], "r")

        target_h = min(l0.shape[0], f0.shape[0], r0.shape[0])

        def resize_to_h(img: np.ndarray, th: int) -> np.ndarray:
            if img.shape[0] == th:
                return img
            scale = th / max(1, img.shape[0])
            new_w = max(1, int(round(img.shape[1] * scale)))
            return cv2.resize(img, (new_w, th), interpolation=cv2.INTER_LINEAR)

        stitched = np.concatenate(
            [resize_to_h(l0, target_h), resize_to_h(f0, target_h), resize_to_h(r0, target_h)],
            axis=1,
        )
        stitched = cv2.resize(stitched, (1024, 256), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(stitched.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _build_status_feature(observation: Dict[str, Any]) -> torch.Tensor:
        if "ego_status" in observation:
            status = np.asarray(observation["ego_status"], dtype=np.float32).reshape(-1)
            if status.shape[0] >= 8:
                return torch.from_numpy(status[:8][None, :]).to(dtype=torch.float32)

        command = np.asarray(
            observation.get("driving_command", np.array([1, 0, 0, 0], dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        velocity = np.asarray(observation.get("ego_velocity", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        acceleration = np.asarray(
            observation.get("ego_acceleration", np.zeros((2,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)

        command4 = np.zeros((4,), dtype=np.float32)
        velocity2 = np.zeros((2,), dtype=np.float32)
        acceleration2 = np.zeros((2,), dtype=np.float32)
        command4[: min(4, command.shape[0])] = command[: min(4, command.shape[0])]
        velocity2[: min(2, velocity.shape[0])] = velocity[: min(2, velocity.shape[0])]
        acceleration2[: min(2, acceleration.shape[0])] = acceleration[: min(2, acceleration.shape[0])]
        status = np.concatenate([command4, velocity2, acceleration2], axis=0)
        return torch.from_numpy(status[None, :]).to(dtype=torch.float32)

    def _load_anchor_bank(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self._anchor_exec_xy is None or self._anchor_exec_yaw is None or self._anchor_exec_norm is None:
            from reconsimulator.envs import nus_config as nus_cfg  # type: ignore

            anchors = np.load(nus_cfg.PLAN_ANCHORS_FILE).astype(np.float32)
            anchors_yaw = np.load(nus_cfg.PLAN_ANCHORS_YAW_FILE).astype(np.float32).reshape(-1) * 5.0
            mask = np.load(nus_cfg.PLAN_ANCHORS_MASK_FILE).reshape(-1)
            exec_idx = max(0, int(anchors.shape[1]) - 1)
            exec_xy = anchors[:, exec_idx, :2].copy()
            self._anchor_exec_xy = exec_xy
            self._anchor_exec_yaw = anchors_yaw.copy()
            self._anchor_exec_norm = np.linalg.norm(exec_xy, axis=1).astype(np.float32, copy=False)
            self._anchor_mask = mask.astype(bool, copy=True)
        return self._anchor_exec_xy, self._anchor_exec_yaw, self._anchor_mask

    def _project_first_step_to_anchor(self, traj_xyyaw: torch.Tensor) -> tuple[tuple[int, int, int], tuple[float, float, float], int]:
        anchor_exec_xy, anchor_exec_yaw, anchor_mask = self._load_anchor_bank()
        target_xy = traj_xyyaw[0, :2].detach().cpu().numpy().astype(np.float32, copy=False)
        dist2 = np.sum((anchor_exec_xy - target_xy[None, :]) ** 2, axis=1)
        if anchor_mask is not None and anchor_mask.shape[0] == dist2.shape[0] and bool(np.any(anchor_mask)):
            dist2 = np.where(anchor_mask, dist2, np.inf)
        target_norm = float(np.linalg.norm(target_xy))
        anchor_exec_norm = self._anchor_exec_norm if self._anchor_exec_norm is not None else None
        if anchor_exec_norm is not None and target_norm > float(self._moving_projection_min_norm_m):
            moving_mask = anchor_exec_norm >= float(self._moving_projection_min_norm_m)
            if bool(np.any(moving_mask)):
                dist2 = np.where(moving_mask, dist2, np.inf)
        selected_idx = int(np.argmin(dist2))
        ax = int(selected_idx // self.y_anchor)
        ay = int(selected_idx % self.y_anchor)
        exec_pose = (
            float(anchor_exec_xy[selected_idx, 0]),
            float(anchor_exec_xy[selected_idx, 1]),
            float(anchor_exec_yaw[selected_idx]),
        )
        return (ax, ay, 0), exec_pose, selected_idx

    def _build_env_action_from_traj(self, traj_xyyaw: torch.Tensor) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        first_step = (
            float(traj_xyyaw[0, 0].item()),
            float(traj_xyyaw[0, 1].item()),
            float(traj_xyyaw[0, 2].item()),
        )
        debug: Dict[str, Any] = {
            "first_step_xyyaw": torch.tensor(first_step, dtype=torch.float32),
            "execute_mode": self._execute_mode,
        }
        if self._execute_mode == "nearest_anchor":
            action, anchor_pose, anchor_idx = self._project_first_step_to_anchor(traj_xyyaw)
            debug["projected_anchor_xyyaw"] = torch.tensor(anchor_pose, dtype=torch.float32)
            debug["exec_anchor_idx"] = int(anchor_idx)
            return action, debug
        return (first_step[0], first_step[1], first_step[2], 2), debug

    def sample_with_replay(
        self,
        observation: Dict[str, Any],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "greedy",
    ) -> tuple[tuple[Any, ...], torch.Tensor, Dict[str, Any]]:
        camera_feature = self._build_camera_feature(observation)
        lidar_feature = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        status_feature = self._build_status_feature(observation)

        model_device = next(self._agent.parameters()).device if hasattr(self._agent, "parameters") else torch.device("cpu")
        features = {
            "camera_feature": camera_feature.to(model_device),
            "lidar_feature": lidar_feature.to(model_device),
            "status_feature": status_feature.to(model_device),
        }

        self._agent._transfuser_model.eval()
        with torch.inference_mode():
            pred = self._agent._transfuser_model(
                features,
                targets=None,
                eta=float(eta),
                metric_cache=None,
                cal_pdm=False,
                token=None,
            )

        traj = pred.get("trajectory", None)
        log_probs = pred.get("log_probs", None)
        diffusion_chain = pred.get("all_diffusion_output", None)
        if traj is None or log_probs is None or diffusion_chain is None:
            raise RuntimeError("DDV2-RL model did not return trajectory/log_probs/all_diffusion_output")

        traj0 = traj[0]
        if log_probs.dim() == 3:
            mode_logps = log_probs[0].sum(dim=-1)
        elif log_probs.dim() == 2:
            mode_logps = log_probs.sum(dim=-1)
        else:
            mode_logps = log_probs.reshape(-1, log_probs.shape[-1]).sum(dim=-1)

        if int(mode_idx) < 0:
            selector = str(mode_select).strip().lower()
            if selector in {"greedy", "max", "argmax"}:
                mi = int(torch.argmax(mode_logps).item())
            else:
                probs = torch.softmax(mode_logps, dim=0)
                if torch.isfinite(probs).all() and float(probs.sum().item()) > 0:
                    mi = int(torch.distributions.Categorical(probs).sample().item())
                else:
                    mi = int(torch.argmax(mode_logps).item())
        else:
            mi = max(0, min(int(mode_idx), int(traj0.shape[0]) - 1))

        traj_sel = traj0[mi]
        action, exec_debug = self._build_env_action_from_traj(traj_sel)
        replay = {
            "camera_feature": camera_feature.detach().cpu().clone(),
            "status_feature": status_feature.detach().cpu().clone(),
            "diffusion_chain": diffusion_chain.detach().cpu().clone(),
            "mode_idx": int(mi),
            "traj_xyyaw": traj_sel.detach().cpu().clone(),
            "traj_xyyaw_raw": traj0[mi].detach().cpu().clone(),
            "first_step_xyyaw": exec_debug.get("first_step_xyyaw"),
            "projected_anchor_xyyaw": exec_debug.get("projected_anchor_xyyaw", None),
            "exec_anchor_idx": exec_debug.get("exec_anchor_idx", None),
            "exec_mode": exec_debug.get("execute_mode", self._execute_mode),
        }
        return action, mode_logps[mi], replay


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


def _local_pose_to_front_frame(start_ego: np.ndarray, local_xyyaw: np.ndarray) -> np.ndarray:
    T_front = np.asarray(start_ego, dtype=np.float64) @ _pose_matrix_from_xyyaw(
        float(local_xyyaw[0]),
        float(local_xyyaw[1]),
        float(local_xyyaw[2]),
    )
    return np.asarray(
        [
            float(T_front[0, 3]),
            float(T_front[1, 3]),
            float(T_front[2, 3]),
            _yaw_from_R_xy(T_front[:3, :3]),
        ],
        dtype=np.float64,
    )


def _ensure_obs_for_diffusiondrive_v2(obs: Dict[str, Any], sim: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(obs)
    out.setdefault("timestamp", np.float32(float(getattr(sim, "now_frame", 0)) * 0.1))
    out.setdefault("ego_pose", np.asarray(getattr(sim, "start_ego", np.eye(4)), dtype=np.float32))
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


def _tensor_like_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 3:
        return None
    return arr[:3].copy()


def _extract_exec_local_pose(replay: Dict[str, Any], traj_xyyaw: np.ndarray) -> np.ndarray:
    exec_mode = str(replay.get("exec_mode", "first_step"))
    if exec_mode == "nearest_anchor":
        projected = _tensor_like_to_numpy(replay.get("projected_anchor_xyyaw", None))
        if projected is not None:
            return projected
    first_step = _tensor_like_to_numpy(replay.get("first_step_xyyaw", None))
    if first_step is not None:
        return first_step
    return np.asarray(traj_xyyaw[0, :3], dtype=np.float64).copy()


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
    ax.set_title(f"Scene {scene:03d}: DiffusionDriveV2 Expert vs Ego (front-frame XZ)")
    ax.set_xlabel("x (right +)")
    ax.set_ylabel("z (forward/north +)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate DiffusionDriveV2 rollout video in 3DGS env")
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
    ap.add_argument("--execute-mode", type=str, default="nearest_anchor", choices=["first_step", "continuous", "nearest_anchor"])
    ap.add_argument("--mode-select", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--expert-high", dest="expert_high", action="store_true", default=True)
    ap.add_argument("--no-expert-high", dest="expert_high", action="store_false")
    args = ap.parse_args()

    scene = int(args.scene)
    RLReconEnv = _lazy_import_env()
    ckpt_path = resolve_repo_path(str(args.ckpt))

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"DiffusionDriveV2 ckpt not found: {ckpt_path}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.out or os.path.join(_DEFAULT_OUT_DIR, f"scene{scene:03d}_{ts}_diffusiondrivev2_rollout.mp4")
    traj_csv = args.traj_csv or os.path.join(_DEFAULT_OUT_DIR, f"scene{scene:03d}_{ts}_diffusiondrivev2_plan_frontframe.csv")
    traj_plot = args.traj_plot or os.path.join(_DEFAULT_OUT_DIR, f"scene{scene:03d}_{ts}_diffusiondrivev2_expert_vs_ego_traj.svg")
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

    policy = _DiffusionDriveV2Inferencer(
        x_anchor=int(getattr(sim, "x_anchor", 61)),
        y_anchor=int(getattr(sim, "y_anchor", 61)),
        ckpt_path=str(ckpt_path),
        device=(f"cuda:{int(args.cuda)}" if torch.cuda.is_available() else "cpu"),
        execute_mode=str(args.execute_mode),
        rl_lr=1e-5,
    )

    step_dt_s = float(getattr(sim, "step_frames", int(args.step_frames))) * 0.1
    if step_dt_s <= 0:
        raise RuntimeError("Invalid step dt")
    max_steps = None if args.duration_s is None else max(1, int(round(float(args.duration_s) / step_dt_s)))
    fps = float(args.fps) if args.fps is not None else (1.0 / step_dt_s)

    print("==== generate_video_diffusiondrive_v2 ====")
    print(f"scene={scene} start_frame={int(args.start_frame)} step_frames={int(args.step_frames)}")
    print(f"ckpt={ckpt_path}")
    print(f"execute_mode={args.execute_mode} mode_select={args.mode_select} eta={float(args.eta):.3f}")
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

    rows: List[Dict[str, float | int | str]] = []
    ego_xz: List[List[float]] = []

    start_pose = np.asarray(getattr(sim, "start_ego"), dtype=np.float64)
    ego_xz.append([float(start_pose[0, 3]), float(start_pose[2, 3])])

    done = False
    steps = 0
    frames = 0
    writer.append_data(_grid_frame(obs))
    frames += 1

    while (max_steps is None or steps < max_steps) and not done:
        obs_in = _ensure_obs_for_diffusiondrive_v2(obs, sim)
        start_ego = np.asarray(getattr(sim, "start_ego"), dtype=np.float64).copy()
        now_frame = int(getattr(sim, "now_frame", -1))

        action, logp, replay = policy.sample_with_replay(
            obs_in,
            eta=float(args.eta),
            mode_idx=-1,
            mode_select=str(args.mode_select),
        )

        traj_xyyaw = _traj_xyyaw_from_replay(replay)
        traj_front = _local_plan_to_front_frame(start_ego, traj_xyyaw)
        exec_local = _extract_exec_local_pose(replay, traj_xyyaw)
        exec_front = _local_pose_to_front_frame(start_ego, exec_local)

        logp_v = float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp)
        print(f"[plan-ddv2] step={steps} frame={now_frame} shape={traj_xyyaw.shape}")
        print(np.array2string(traj_xyyaw, precision=6, suppress_small=False))

        rows.append(
            {
                "step": int(steps),
                "frame": int(now_frame),
                "plan_idx": 0,
                "mode_idx": int(replay.get("mode_idx", -1)),
                "logp": float(logp_v),
                "exec_mode": str(replay.get("exec_mode", args.execute_mode)),
                "local_x": float(traj_xyyaw[0, 0]),
                "local_y": float(traj_xyyaw[0, 1]),
                "local_yaw": float(traj_xyyaw[0, 2]),
                "front_x": float(traj_front[0, 0]),
                "front_y": float(traj_front[0, 1]),
                "front_z": float(traj_front[0, 2]),
                "front_yaw": float(traj_front[0, 3]),
                "exec_local_x": float(exec_local[0]),
                "exec_local_y": float(exec_local[1]),
                "exec_local_yaw": float(exec_local[2]),
                "exec_front_x": float(exec_front[0]),
                "exec_front_y": float(exec_front[1]),
                "exec_front_z": float(exec_front[2]),
                "exec_front_yaw": float(exec_front[3]),
            }
        )

        setattr(sim, "_external_plan_local_xyyaw", np.asarray(traj_xyyaw, dtype=np.float64).copy())

        obs, _reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)

        pose_after = np.asarray(obs.get("ego_pose", getattr(sim, "start_ego")), dtype=np.float64)
        ego_xz.append([float(pose_after[0, 3]), float(pose_after[2, 3])])

        pred_xz = np.asarray([float(exec_front[0]), float(exec_front[2])], dtype=np.float64)
        real_xz = np.asarray([float(pose_after[0, 3]), float(pose_after[2, 3])], dtype=np.float64)
        err_xz = float(np.linalg.norm(pred_xz - real_xz, ord=2))

        frame_after = int(getattr(sim, "now_frame", -1))
        cmd_obs, vel_obs, acc_obs = _extract_status_from_obs(obs)
        _cmd_ds, vel_ds, acc_ds = _dataset_status_from_sim(sim, frame_after)

        print(
            "[pose-check-ddv2] "
            f"step={steps} frame={now_frame} "
            f"action={action} "
            f"exec_mode={replay.get('exec_mode', args.execute_mode)} "
            f"pred_next_xz=({pred_xz[0]:.6f},{pred_xz[1]:.6f}) "
            f"real_next_xz=({real_xz[0]:.6f},{real_xz[1]:.6f}) "
            f"l2_err={err_xz:.9f}"
        )
        print(
            "[status-check-ddv2] "
            f"step={steps} frame_after={frame_after} "
            f"command_obs={np.array2string(cmd_obs, precision=6, suppress_small=False)} "
            f"vel_obs={np.array2string(vel_obs, precision=6, suppress_small=False)} "
            f"acc_obs={np.array2string(acc_obs, precision=6, suppress_small=False)} "
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
        "mode_idx",
        "logp",
        "exec_mode",
        "local_x",
        "local_y",
        "local_yaw",
        "front_x",
        "front_y",
        "front_z",
        "front_yaw",
        "exec_local_x",
        "exec_local_y",
        "exec_local_yaw",
        "exec_front_x",
        "exec_front_y",
        "exec_front_z",
        "exec_front_yaw",
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
    print(f"[traj-ddv2] ego_xz shape={ego_xz_np.shape}")
    print(np.array2string(ego_xz_np, precision=6, suppress_small=False))
    print(f"[traj-ddv2] expert_xz shape={expert_xz.shape}")
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