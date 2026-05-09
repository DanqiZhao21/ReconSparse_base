from __future__ import annotations

import importlib
import os
import sys
import uuid
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .base import Agent
from reconsimulator.envs import nus_config as nus_cfg
from framework.utils.nuscenes_token import resolve_sample_token
from framework.utils.repo_paths import resolve_ego_ads_subdir


_SPARSEDRIVE_V2_ROOT = resolve_ego_ads_subdir("SparseDriveV2")
if _SPARSEDRIVE_V2_ROOT not in sys.path:
    sys.path.insert(0, _SPARSEDRIVE_V2_ROOT)


def _force_import_sparsedrive_v2_modules() -> tuple[type, type]:
    try:
        from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
        from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel

        return SparseDriveConfig, SparseDriveModel
    except Exception:
        pass

    stale_keys = [k for k in list(sys.modules.keys()) if k == "navsim" or k.startswith("navsim.")]
    for key in stale_keys:
        try:
            del sys.modules[key]
        except Exception:
            pass

    importlib.invalidate_caches()
    from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
    from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel

    return SparseDriveConfig, SparseDriveModel


def _strip_state_dict_prefix(key: str) -> str:
    prefixes = [
        "module.",
        "agent._sparsedrive_model.",
        "_sparsedrive_model.",
        "agent.",
        "model.",
    ]
    out = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if out.startswith(prefix):
                out = out[len(prefix) :]
                changed = True
    return out


def _normalize_trainable_prefixes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _apply_trainable_prefixes(module: torch.nn.Module, prefixes: Sequence[str]) -> tuple[int, int]:
    normalized = [str(prefix).strip() for prefix in prefixes if str(prefix).strip()]
    if len(normalized) == 0:
        total = sum(1 for _ in module.parameters())
        trainable = sum(1 for param in module.parameters() if getattr(param, "requires_grad", False))
        return total, trainable

    for param in module.parameters():
        param.requires_grad = False

    trainable = 0
    total = 0
    for name, param in module.named_parameters():
        total += 1
        if any(name.startswith(prefix) for prefix in normalized):
            param.requires_grad = True
            trainable += 1
    return total, trainable


def _summarize_parameter_status(module: torch.nn.Module) -> tuple[List[str], int, int, int, int]:
    trainable_names: List[str] = []
    total_tensors = 0
    trainable_tensors = 0
    total_params = 0
    trainable_params = 0
    for name, param in module.named_parameters():
        total_tensors += 1
        total_params += int(param.numel())
        if getattr(param, "requires_grad", False):
            trainable_names.append(name)
            trainable_tensors += 1
            trainable_params += int(param.numel())
    return trainable_names, total_tensors, trainable_tensors, total_params, trainable_params


class SparseDriveV2Policy(Agent):
    def __init__(
        self,
        *,
        ckpt_path: str,
        device: str | None = None,
        rl_lr: float = 1e-5,
        execute_mode: str = "first_step",
        trainable_prefixes: Sequence[str] | None = None,
        nuscenes_scorer_config: Dict[str, Any] | None = None,
    ) -> None:
        try:
            SparseDriveConfig, SparseDriveModel = _force_import_sparsedrive_v2_modules()
        except Exception as exc:
            raise ImportError(
                f"[SparseDriveV2Policy] Failed to import SparseDriveV2 modules: {exc}"
            ) from exc

        if str(execute_mode).strip().lower() not in {"first_step", "continuous", "step1"}:
            raise ValueError(f"Unsupported execute_mode for SparseDriveV2: {execute_mode}")

        self._SparseDriveConfig = SparseDriveConfig
        self._SparseDriveModel = SparseDriveModel
        self.ckpt_path = str(ckpt_path)
        self._device_override = device
        self._execute_mode = "first_step"
        self._trainable_prefixes = _normalize_trainable_prefixes(trainable_prefixes)
        self._nuscenes_scorer_config = dict(nuscenes_scorer_config or {})

        self._cfg = self._SparseDriveConfig()
        self._cfg.bkb_path = os.path.join(_SPARSEDRIVE_V2_ROOT, "ckpt", "resnet34.bin")
        self._cfg.path_anchor = os.path.join(_SPARSEDRIVE_V2_ROOT, "ckpt", "kmeans", "path_1024.npy")
        self._cfg.velocity_anchor = os.path.join(_SPARSEDRIVE_V2_ROOT, "ckpt", "kmeans", "velocity_256.npy")
        self._cfg.trajectory_anchor = os.path.join(_SPARSEDRIVE_V2_ROOT, "ckpt", "kmeans", "trajectory_1024_256.npz")

        self._model = self._SparseDriveModel(self._cfg)
        self.to(self.device)
        self._load_weights(self.ckpt_path)
        _apply_trainable_prefixes(self._model, self._trainable_prefixes)
        trainable_names, total_tensors, trainable_tensors, total_params, trainable_params = _summarize_parameter_status(self._model)
        if len(self._trainable_prefixes) > 0:
            print(
                "[SparseDriveV2Policy] trainable_prefixes="
                f"{self._trainable_prefixes} -> trainable tensors {trainable_tensors}/{total_tensors}, "
                f"trainable params {trainable_params}/{total_params}"
            )
            # for name in trainable_names:
            #     print(f"[SparseDriveV2Policy] trainable_param={name}")
            for name in trainable_names[-1:]:
                print(f"[SparseDriveV2Policy] trainable_param[-1]={name}")
        else:
            print(
                "[SparseDriveV2Policy] trainable_prefixes=[] -> training all currently-enabled parameters: "
                f"trainable tensors {trainable_tensors}/{total_tensors}, trainable params {trainable_params}/{total_params}"
            )

        params = [param for param in self._model.parameters() if getattr(param, "requires_grad", False)]
        self._optimizer: torch.optim.Optimizer | None = None
        if len(params) > 0:
            self._optimizer = torch.optim.Adam(params, lr=float(rl_lr))

        self._last_missing_feature_fields: List[str] = []
        self._teacher_model: torch.nn.Module | None = None
        self._nuscenes_token_scorer: Any | None = None
        self._nuscenes_pdm_scorer: Any | None = None
        self._nuscenes_pdm_gpu_scorer: Any | None = None

    @property
    def device(self) -> torch.device:
        if self._device_override:
            try:
                return torch.device(str(self._device_override))
            except Exception:
                pass
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def to(self, device: str | torch.device) -> "SparseDriveV2Policy":
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        self._device_override = str(dev)
        try:
            if isinstance(self._model, DDP):
                self._model.module.to(dev)
            else:
                self._model.to(dev)
        except Exception:
            pass
        return self

    def _load_state_into_model(self, model: torch.nn.Module, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SparseDriveV2 ckpt not found: {path}")
        ckpt = torch.load(path, map_location="cpu")
        sd_raw = ckpt.get("state_dict", ckpt)
        if not isinstance(sd_raw, dict):
            raise RuntimeError("Invalid checkpoint format: missing state_dict")

        state_dict: Dict[str, torch.Tensor] = {}
        for key, value in sd_raw.items():
            if torch.is_tensor(value):
                normalized_key = _strip_state_dict_prefix(str(key))
                if normalized_key.startswith("img_backbone"):
                    normalized_key = "_backbone." + normalized_key
                state_dict[normalized_key] = value

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[SparseDriveV2Policy] Loaded ckpt: {path}")
        if missing:
            print(f"[SparseDriveV2Policy] missing_keys={len(missing)}")
        if unexpected:
            print(f"[SparseDriveV2Policy] unexpected_keys={len(unexpected)} (ignored)")

    def _load_weights(self, path: str) -> None:
        self._load_state_into_model(self._model, path)

    def initialize(self) -> None:
        return

    def parameters(self):
        model = self._model.module if isinstance(self._model, DDP) else self._model
        return model.parameters()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        model = self._model.module if isinstance(self._model, DDP) else self._model
        return model.state_dict()

    @property
    def trainable_module(self):
        return self._model

    def save_checkpoint(self, path: str) -> None:
        state_dict = {f"agent.{key}": value.detach().cpu() for key, value in self.state_dict().items()}
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(path)
        tmp = os.path.join(out_dir, f".{base}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            torch.save({"state_dict": state_dict}, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    def load_checkpoint(self, path: str, *, strict: bool = False) -> None:
        ckpt = torch.load(path, map_location="cpu")
        sd_raw = ckpt.get("state_dict", ckpt)
        if not isinstance(sd_raw, dict):
            raise RuntimeError("Invalid checkpoint format")
        state_dict: Dict[str, torch.Tensor] = {}
        for key, value in sd_raw.items():
            if torch.is_tensor(value):
                state_dict[_strip_state_dict_prefix(str(key))] = value
        model = self._model.module if isinstance(self._model, DDP) else self._model
        model.load_state_dict(state_dict, strict=bool(strict))

    def wrap_ddp(
        self,
        *,
        device_id: int,
        process_group: Any | None = None,
        find_unused_parameters: bool = True,
        rl_lr: float | None = None,
    ) -> None:
        model = self._model
        if isinstance(model, DDP):
            return
        target = torch.device(f"cuda:{int(device_id)}") if torch.cuda.is_available() else torch.device("cpu")
        self.to(target)
        self._model = DDP(
            model,
            device_ids=[int(device_id)] if target.type == "cuda" else None,
            output_device=int(device_id) if target.type == "cuda" else None,
            process_group=process_group,
            find_unused_parameters=bool(find_unused_parameters),
        )

        lr = float(rl_lr) if rl_lr is not None else 1e-5
        if self._optimizer is not None:
            lr = float(self._optimizer.param_groups[0].get("lr", lr))
        core = self._model.module
        params = [param for param in core.parameters() if getattr(param, "requires_grad", False)]
        self._optimizer = torch.optim.Adam(params, lr=lr)

    @staticmethod
    def _normalize_rgb(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32)
        return (x - mean[None, None, :]) / std[None, None, :]

    @staticmethod
    def _camera_obs_key_for_cfg_cam(cam_name: str) -> str:
        mapping = {
            "cam_l0": "front_left",
            "cam_f0": "front",
            "cam_r0": "front_right",
            "cam_l1": "back_left",
            "cam_b0": "back",
            "cam_r1": "back_right",
        }
        return mapping.get(str(cam_name), str(cam_name))

    @staticmethod
    def _camera_obs_index(obs_key: str) -> int:
        index_map = {
            "front": 0,
            "front_left": 1,
            "front_right": 2,
            "back_left": 3,
            "back": 4,
            "back_right": 5,
        }
        if obs_key not in index_map:
            raise KeyError(f"Unsupported camera key: {obs_key}")
        return int(index_map[obs_key])

    @staticmethod
    def _as_intrinsics_3x3(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape == (4, 4):
            return matrix[:3, :3]
        if matrix.shape == (3, 3):
            return matrix
        out = np.eye(3, dtype=np.float32)
        h = min(3, matrix.shape[0])
        w = min(3, matrix.shape[1])
        out[:h, :w] = matrix[:h, :w]
        return out

    def _build_camera_meta(
        self,
        cam2ego: np.ndarray,
        cam_intrinsics: np.ndarray,
        cam_hw: np.ndarray | None,
        cam_distortions: np.ndarray,
        selected_idx: List[int],
        out_hw: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        n = len(selected_idx)
        lidar2img = np.zeros((n, 4, 4), dtype=np.float32)
        lidar2cam = np.zeros((n, 4, 4), dtype=np.float32)
        cam2lidar = np.zeros((n, 4, 4), dtype=np.float32)
        cam_intrinsic = np.zeros((n, 3, 3), dtype=np.float32)
        distortions = np.zeros((n, cam_distortions.shape[1]), dtype=np.float32)
        image_wh = np.zeros((n, 2), dtype=np.float32)
        image_wh[:, 0] = float(out_w)
        image_wh[:, 1] = float(out_h)

        for i, camera_id in enumerate(selected_idx):
            intrinsics = self._as_intrinsics_3x3(cam_intrinsics[camera_id])
            if cam_hw is not None and cam_hw.shape[0] > camera_id:
                h0 = float(cam_hw[camera_id, 0])
                w0 = float(cam_hw[camera_id, 1])
            else:
                h0, w0 = float(out_h), float(out_w)

            sx = float(out_w) / max(1.0, w0)
            sy = float(out_h) / max(1.0, h0)
            scaled = intrinsics.copy()
            scaled[0, 0] *= sx
            scaled[0, 2] *= sx
            scaled[1, 1] *= sy
            scaled[1, 2] *= sy

            cam_to_ego = np.asarray(cam2ego[camera_id], dtype=np.float32)
            ego_to_cam = np.linalg.inv(cam_to_ego)

            viewpad = np.eye(4, dtype=np.float32)
            viewpad[:3, :3] = scaled
            lidar2img[i] = viewpad @ ego_to_cam
            lidar2cam[i] = ego_to_cam
            cam2lidar[i] = cam_to_ego
            cam_intrinsic[i] = scaled
            distortions[i] = np.asarray(cam_distortions[camera_id], dtype=np.float32)

        return {
            "distortions": distortions,
            "lidar2img": lidar2img,
            "lidar2cam": lidar2cam,
            "cam2lidar": cam2lidar,
            "cam_intrinsic": cam_intrinsic,
            "projection_mat": lidar2img.copy(),
            "image_wh": image_wh,
        }

    def _build_status_feature(self, observation: Dict[str, Any]) -> torch.Tensor:
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

    def _build_features(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        missing_fields: List[str] = []

        cfg_cams = [str(cam) for cam in list(self._cfg.cams)]
        cam_keys = [self._camera_obs_key_for_cfg_cam(cam) for cam in cfg_cams]
        cam_idx = [self._camera_obs_index(cam) for cam in cam_keys]

        imgs: List[np.ndarray] = []
        for key in cam_keys:
            if key not in observation:
                raise KeyError(f"Missing camera key in observation: {key}")
            imgs.append(np.asarray(observation[key]))

        out_h, out_w = int(self._cfg.final_dim[0]), int(self._cfg.final_dim[1])
        imgs_rs = [cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR) for img in imgs]
        mean = np.asarray(self._cfg.img_mean, dtype=np.float32)
        std = np.asarray(self._cfg.img_std, dtype=np.float32)
        imgs_norm = [self._normalize_rgb(img, mean, std) for img in imgs_rs]
        img_tensor = torch.from_numpy(
            np.stack([img.transpose(2, 0, 1) for img in imgs_norm], axis=0)
        ).to(dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0)

        cam2ego = observation.get("cam2ego", None)
        cam_intr = observation.get("cam_intrinsics", None)
        cam_hw = observation.get("cam_hw", None)
        cam_dist = observation.get("cam_distortions", None)

        if cam2ego is None or cam_intr is None:
            missing_fields.extend(["cam2ego", "cam_intrinsics"])
            cam2ego_np = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 6, axis=0)
            cam_intr_np = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 6, axis=0)
            cam_intr_np[:, 0, 0] = float(out_w)
            cam_intr_np[:, 1, 1] = float(out_h)
            cam_intr_np[:, 0, 2] = float(out_w) * 0.5
            cam_intr_np[:, 1, 2] = float(out_h) * 0.5
            cam_hw_np = np.asarray([[out_h, out_w]] * 6, dtype=np.float32)
        else:
            cam2ego_np = np.asarray(cam2ego, dtype=np.float32)
            cam_intr_np = np.asarray(cam_intr, dtype=np.float32)
            cam_hw_np = np.asarray(cam_hw, dtype=np.float32) if cam_hw is not None else np.asarray([[out_h, out_w]] * cam2ego_np.shape[0], dtype=np.float32)

        if cam_hw is None:
            missing_fields.append("cam_hw")

        if cam_dist is None:
            missing_fields.append("cam_distortions")
            cam_dist_np = np.zeros((cam2ego_np.shape[0], 5), dtype=np.float32)
        else:
            cam_dist_np = np.asarray(cam_dist, dtype=np.float32)
            if cam_dist_np.ndim == 1:
                cam_dist_np = cam_dist_np[None, :]

        camera_meta_np = self._build_camera_meta(
            cam2ego=cam2ego_np,
            cam_intrinsics=cam_intr_np,
            cam_hw=cam_hw_np,
            cam_distortions=cam_dist_np,
            selected_idx=cam_idx,
            out_hw=(out_h, out_w),
        )

        camera_feature: Dict[str, torch.Tensor] = {"imgs": img_tensor}
        for key, value in camera_meta_np.items():
            camera_feature[key] = torch.from_numpy(value).unsqueeze(0).to(dtype=torch.float32)

        self._last_missing_feature_fields = sorted(set(missing_fields))
        return {
            "camera_feature": camera_feature,
            "status_feature": self._build_status_feature(observation),
            "feature_missing_fields": list(self._last_missing_feature_fields),
        }

    def _to_device_features(self, features: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in features.items():
            if isinstance(value, dict):
                out[key] = {
                    sub_key: sub_val.to(device=device, dtype=torch.float32)
                    for sub_key, sub_val in value.items()
                    if torch.is_tensor(sub_val)
                }
            elif torch.is_tensor(value):
                out[key] = value.to(device=device, dtype=torch.float32)
        return out

    def _batch_observation_features(
        self,
        observations: Sequence[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        features_list = [self._build_features(obs) for obs in observations]
        if len(features_list) == 0:
            return [], {"camera_feature": {}, "status_feature": torch.empty((0,), dtype=torch.float32)}

        camera_keys = list(features_list[0]["camera_feature"].keys())
        batched_camera = {
            key: torch.cat([feat["camera_feature"][key] for feat in features_list], dim=0)
            for key in camera_keys
        }
        return features_list, {
            "camera_feature": batched_camera,
            "status_feature": torch.cat([feat["status_feature"] for feat in features_list], dim=0),
        }

    @staticmethod
    def _unwrap_model(module: torch.nn.Module | DDP) -> torch.nn.Module:
        return module.module if isinstance(module, DDP) else module

    def _forward_policy_on_model(self, model: torch.nn.Module, features_dev: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out, _loss_dict = model(features_dev, targets={})
        if "trajectory" not in out:
            raise RuntimeError("SparseDriveV2 output missing 'trajectory'")

        candidate_trajectories = out.get("candidate_trajectories", None)
        candidate_scores = out.get("candidate_scores", None)
        if candidate_trajectories is None:
            trajectory = out["trajectory"]
            if trajectory.ndim == 2:
                trajectory = trajectory.unsqueeze(0)
            candidate_trajectories = trajectory.unsqueeze(1)
        if candidate_scores is None:
            candidate_scores = torch.zeros(
                (candidate_trajectories.shape[0], candidate_trajectories.shape[1]),
                dtype=torch.float32,
                device=candidate_trajectories.device,
            )

        out["candidate_trajectories"] = candidate_trajectories
        out["candidate_scores"] = candidate_scores.to(dtype=torch.float32)
        return out

    def _forward_policy(self, features_dev: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        model = self._unwrap_model(self._model)
        return self._forward_policy_on_model(model, features_dev)

    @staticmethod
    def _traj_to_xyyaw(traj: torch.Tensor) -> np.ndarray:
        arr = traj.detach().cpu().numpy().astype(np.float32)
        if arr.ndim != 2 or arr.shape[0] < 1:
            raise RuntimeError(f"Invalid trajectory output shape: {arr.shape}")
        if arr.shape[1] >= 3:
            return arr[:, :3]

        xy = arr[:, :2]
        out = np.zeros((xy.shape[0], 3), dtype=np.float32)
        out[:, :2] = xy
        prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), xy[:-1]], axis=0)
        dxy = xy - prev
        out[:, 2] = np.arctan2(dxy[:, 1], dxy[:, 0]).astype(np.float32)
        return out
    
    @staticmethod
    def _traj_batch_to_xyyaw(trajs: torch.Tensor) -> torch.Tensor:
        arr = trajs.detach().cpu().to(dtype=torch.float32)
        if arr.ndim != 3 or arr.shape[1] < 1:
            raise RuntimeError(f"Invalid batched trajectory output shape: {tuple(arr.shape)}")
        if arr.shape[2] >= 3:
            return arr[:, :, :3].clone()

        xy = arr[:, :, :2]
        out = torch.zeros((xy.shape[0], xy.shape[1], 3), dtype=torch.float32)
        out[:, :, :2] = xy
        prev = torch.cat([torch.zeros((xy.shape[0], 1, 2), dtype=torch.float32), xy[:, :-1, :]], dim=1)
        dxy = xy - prev
        out[:, :, 2] = torch.atan2(dxy[:, :, 1], dxy[:, :, 0])
        return out


    @staticmethod
    def _build_env_action(traj_xyyaw: np.ndarray | torch.Tensor) -> Tuple[Any, ...]:
        if torch.is_tensor(traj_xyyaw):
            traj_xyyaw = traj_xyyaw.detach().cpu().numpy()
        first = traj_xyyaw[0]
        return (float(first[0]), float(first[1]), float(first[2]), 2)

    @staticmethod
    def _select_mode(score_logits: torch.Tensor, *, mode_idx: int, mode_select: str) -> int:
        logits = score_logits.reshape(-1)
        if int(mode_idx) >= 0:
            return max(0, min(int(mode_idx), int(logits.shape[0]) - 1))

        select = str(mode_select).strip().lower()
        if select in {"greedy", "argmax", "max"}:
            return int(torch.argmax(logits).item())

        probs = torch.softmax(logits, dim=0)
        if torch.isfinite(probs).all() and float(probs.sum().item()) > 0.0:
            return int(torch.distributions.Categorical(probs).sample().item())
        return int(torch.argmax(logits).item())

    @staticmethod
    def _select_mode_batch(score_logits: torch.Tensor, *, mode_idx: int, mode_select: str) -> torch.Tensor:
        logits = score_logits
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.ndim != 2:
            raise RuntimeError(f"Invalid candidate score shape for batch selection: {tuple(logits.shape)}")

        batch_size, num_modes = int(logits.shape[0]), int(logits.shape[1])
        if int(mode_idx) >= 0:
            fixed_idx = max(0, min(int(mode_idx), num_modes - 1))
            return torch.full((batch_size,), fixed_idx, dtype=torch.long, device=logits.device)

        select = str(mode_select).strip().lower()
        if select in {"greedy", "argmax", "max"}:
            return torch.argmax(logits, dim=1).to(dtype=torch.long)

        probs = torch.softmax(logits, dim=1)
        if torch.isfinite(probs).all() and torch.all(probs.sum(dim=1) > 0.0):
            return torch.distributions.Categorical(probs=probs).sample().to(dtype=torch.long)
        return torch.argmax(logits, dim=1).to(dtype=torch.long)

    @staticmethod
    def _detach_replay_features(features: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in features.items():
            if isinstance(value, dict):
                out[key] = {
                    sub_key: sub_val.detach().cpu().clone()
                    for sub_key, sub_val in value.items()
                    if torch.is_tensor(sub_val)
                }
            elif torch.is_tensor(value):
                out[key] = value.detach().cpu().clone()
        return out

    @staticmethod
    def _update_replay_with_observation_metadata(replay: Dict[str, Any], observation: Dict[str, Any]) -> None:
        token = observation.get("sample_token", observation.get("token", None))
        if token is None:
            token = resolve_sample_token(
                scene_id=observation.get("scene_id", None),
                frame_idx=observation.get("frame_idx", None),
            )
        if token is not None:
            replay["sample_token"] = str(token)
        try:
            replay["scene_id"] = int(observation.get("scene_id", 0))
        except Exception:
            pass
        try:
            replay["frame_idx"] = int(observation.get("frame_idx", 0))
        except Exception:
            pass
        try:
            replay["timestamp_s"] = float(observation.get("timestamp", 0.0))
        except Exception:
            pass

    def sample_sparsedrivev2_with_replay(
        self,
        observation: Dict[str, Any],
        *,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ) -> Tuple[Tuple[Any, ...], torch.Tensor, Dict[str, Any]]:
        actions, logps, replays = self.sample_sparsedrivev2_with_replay_batch(
            [observation],
            mode_idx=int(mode_idx),
            mode_select=str(mode_select),
        )
        return actions[0], logps[0], replays[0]
#actor采样时：闭环log π(a|s)
    def sample_sparsedrivev2_with_replay_batch(
        self,
        observations: List[Dict[str, Any]],
        *,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ) -> Tuple[List[Tuple[Any, ...]], List[torch.Tensor], List[Dict[str, Any]]]:
        if len(observations) == 0:
            return [], [], []

        features_list, batched_features = self._batch_observation_features(observations)
        batched_dev = self._to_device_features(batched_features, self.device)

        model = self._unwrap_model(self._model)
        model.eval()
        with torch.inference_mode():
            out = self._forward_policy(batched_dev)

        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        selected_mode_indices = self._select_mode_batch(
            score_logits,
            mode_idx=int(mode_idx),
            mode_select=str(mode_select),
        )
        logp_all = torch.log_softmax(score_logits, dim=1)
        batch_indices = torch.arange(score_logits.shape[0], device=score_logits.device)
        selected_logps = logp_all[batch_indices, selected_mode_indices]

        all_scores = score_logits.detach().cpu()
        all_trajs = out["candidate_trajectories"].detach().cpu()
        selected_mode_indices_cpu = selected_mode_indices.detach().cpu()
        selected_trajs = all_trajs[torch.arange(all_trajs.shape[0]), selected_mode_indices_cpu]
        traj_xyyaw_batch = self._traj_batch_to_xyyaw(selected_trajs)
        actions: List[Tuple[Any, ...]] = []
        logps: List[torch.Tensor] = []
        replays: List[Dict[str, Any]] = []

        for idx, features in enumerate(features_list):
            scores = all_scores[idx]
            selected_mode_idx = int(selected_mode_indices_cpu[idx].item())
            logp = selected_logps[idx]
            traj_xyyaw = traj_xyyaw_batch[idx]
            replay = self._detach_replay_features(features)
            replay.update(
                {
                    "mode_idx": int(selected_mode_idx),
                    "traj_xyyaw": traj_xyyaw.detach().cpu().clone(),
                    "candidate_scores": scores.detach().cpu().clone(),
                    "execute_mode": self._execute_mode,
                    "feature_missing_fields": list(features.get("feature_missing_fields", [])),
                }
            )
            self._update_replay_with_observation_metadata(replay, observations[idx])
            actions.append(self._build_env_action(traj_xyyaw))
            logps.append(logp)
            replays.append(replay)

        return actions, logps, replays
# learner 训练时
    def logp_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        eta: float = 1.0,
    ) -> torch.Tensor:
        del eta
        if len(replays) == 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)

        bad_indices = [
            idx for idx, replay in enumerate(replays)
            if not self.replay_is_compatible(replay)
        ]
        if len(bad_indices) > 0:
            raise RuntimeError(
                "SparseDriveV2 replay batch contains incompatible replay entries; "
                "this usually means old-format shards are mixed into the current buffer. "
                f"Bad replay indices: {bad_indices[:8]}"
            )

        camera_keys = list(replays[0]["camera_feature"].keys())
        batched_camera = {
            key: torch.cat([rep["camera_feature"][key] for rep in replays], dim=0)
            for key in camera_keys
        }
        batched_features = {
            "camera_feature": batched_camera,
            "status_feature": torch.cat([rep["status_feature"] for rep in replays], dim=0),
        }
        batched_dev = self._to_device_features(batched_features, self.device)

        model = self._model.module if isinstance(self._model, DDP) else self._model
        model.eval()
        out = self._forward_policy(batched_dev)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        mode_indices = torch.as_tensor(
            [int(rep.get("mode_idx", 0)) for rep in replays],
            dtype=torch.long,
            device=score_logits.device,
        )
        logp_all = torch.log_softmax(score_logits, dim=1)
        return logp_all[torch.arange(score_logits.shape[0], device=score_logits.device), mode_indices]

#GRPO部分：输入已经 forward 出来的结果，然后挑候选。
    def _select_counterfactual_candidates_from_policy_outputs(
        self,
        *,
        score_logits: torch.Tensor,
        candidate_trajs: torch.Tensor,
        num_candidates: int,
        candidate_select: str,
        logp_all: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if candidate_trajs.ndim != 4:
            raise RuntimeError(
                "SparseDriveV2 counterfactual candidates require candidate trajectories with shape "
                f"(batch, modes, horizon, dims); got {tuple(candidate_trajs.shape)}"
            )

        batch_size, num_modes, horizon, traj_dim = tuple(candidate_trajs.shape)
        del batch_size, horizon, traj_dim
        k = max(1, min(int(num_candidates), int(num_modes)))
        select = str(candidate_select).strip().lower()
        if select not in {"topk", "all"}:
            raise ValueError(f"Unsupported candidate_select={candidate_select!r}; expected 'topk' or 'all'")

        if select == "all":
            selected_indices = torch.argsort(score_logits, dim=1, descending=True)[:, :k]
        else:
            selected_indices = torch.topk(score_logits, k=k, dim=1, largest=True, sorted=True).indices

        gather_idx_scores = selected_indices
        gather_idx_traj = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            candidate_trajs.shape[2],
            candidate_trajs.shape[3],
        )
        selected_score_logits = torch.gather(score_logits, dim=1, index=gather_idx_scores)
        selected_trajs = torch.gather(candidate_trajs, dim=1, index=gather_idx_traj)

        if logp_all is None:
            logp_all = torch.log_softmax(score_logits, dim=1)
        selected_log_probs = torch.gather(logp_all, dim=1, index=gather_idx_scores)

        selected_trajs_flat = selected_trajs.reshape(-1, selected_trajs.shape[2], selected_trajs.shape[3])
        traj_xyyaw = self._traj_batch_to_xyyaw(selected_trajs_flat).reshape(
            selected_trajs.shape[0],
            selected_trajs.shape[1],
            selected_trajs.shape[2],
            3,
        )

        return {
            "traj_xyyaw": traj_xyyaw.detach(),
            "log_probs": selected_log_probs,
            "mode_indices": selected_indices.to(dtype=torch.long),
            "score_logits": selected_score_logits,
        }
        
#调用上层函数
    def sample_counterfactual_trajectories_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        num_candidates: int,
        candidate_select: str = "topk",
    ) -> Dict[str, torch.Tensor]:
        if len(replays) == 0:
            return {
                "traj_xyyaw": torch.empty((0, 0, 0, 3), dtype=torch.float32),
                "log_probs": torch.empty((0, 0), device=self.device, dtype=torch.float32),
                "mode_indices": torch.empty((0, 0), device=self.device, dtype=torch.long),
                "score_logits": torch.empty((0, 0), device=self.device, dtype=torch.float32),
            }

        batched_dev = self._batched_replay_features(replays)
        model = self._unwrap_model(self._model)
        model.eval()
        out = self._forward_policy_on_model(model, batched_dev)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        candidate_trajs = out["candidate_trajectories"]
        return self._select_counterfactual_candidates_from_policy_outputs(
            score_logits=score_logits,
            candidate_trajs=candidate_trajs,
            num_candidates=int(num_candidates),
            candidate_select=str(candidate_select),
        )

    def replay_policy_outputs_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        eta: float = 1.0,
        num_candidates: int,
        candidate_select: str = "topk",
    ) -> Dict[str, Any]:
        del eta
        if len(replays) == 0:
            empty_candidates = {
                "traj_xyyaw": torch.empty((0, 0, 0, 3), dtype=torch.float32),
                "log_probs": torch.empty((0, 0), device=self.device, dtype=torch.float32),
                "mode_indices": torch.empty((0, 0), device=self.device, dtype=torch.long),
                "score_logits": torch.empty((0, 0), device=self.device, dtype=torch.float32),
            }
            return {
                "new_logp": torch.empty((0,), device=self.device, dtype=torch.float32),
                "counterfactual": empty_candidates,
            }

        batched_dev = self._batched_replay_features(replays)
        model = self._unwrap_model(self._model)
        model.eval()
        out = self._forward_policy_on_model(model, batched_dev)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        candidate_trajs = out["candidate_trajectories"]
        if candidate_trajs.ndim != 4:
            raise RuntimeError(
                "SparseDriveV2 replay policy outputs require candidate trajectories with shape "
                f"(batch, modes, horizon, dims); got {tuple(candidate_trajs.shape)}"
            )

        mode_indices = torch.as_tensor(
            [int(rep.get("mode_idx", 0)) for rep in replays],
            dtype=torch.long,
            device=score_logits.device,
        )
        logp_all = torch.log_softmax(score_logits, dim=1)
        new_logp = logp_all[torch.arange(score_logits.shape[0], device=score_logits.device), mode_indices]
        counterfactual = self._select_counterfactual_candidates_from_policy_outputs(
            score_logits=score_logits,
            candidate_trajs=candidate_trajs,
            num_candidates=int(num_candidates),
            candidate_select=str(candidate_select),
            logp_all=logp_all,
        )

        return {
            "new_logp": new_logp,
            "counterfactual": counterfactual,
        }

    def init_distillation_teacher(self, *, ckpt_path: str | None = None) -> None:
        teacher_ckpt = str(ckpt_path) if ckpt_path is not None else str(self.ckpt_path)
        teacher = self._SparseDriveModel(self._cfg)
        self._load_state_into_model(teacher, teacher_ckpt)
        teacher.to(self.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        self._teacher_model = teacher

    def _distill_log_probs_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        temperature: float = 1.0,
        model: torch.nn.Module,
        use_inference_mode: bool,
    ) -> torch.Tensor:
        if len(replays) == 0:
            return torch.empty((0, 0), device=self.device, dtype=torch.float32)

        batched_dev = self._batched_replay_features(replays)
        if use_inference_mode:
            model.eval()
            with torch.inference_mode():
                out = self._forward_policy_on_model(model, batched_dev)
        else:
            out = self._forward_policy_on_model(model, batched_dev)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        return F.log_softmax(score_logits / float(temperature), dim=1)

    def distill_student_log_probs_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return self._distill_log_probs_from_replay_batch(
            replays,
            temperature=float(temperature),
            model=self._unwrap_model(self._model),
            use_inference_mode=False,
        )

    def distill_teacher_log_probs_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if self._teacher_model is None:
            raise RuntimeError("SparseDriveV2 distillation teacher is not initialized")
        return self._distill_log_probs_from_replay_batch(
            replays,
            temperature=float(temperature),
            model=self._teacher_model,
            use_inference_mode=True,
        )

    @property
    def value_feature_dim(self) -> int | None:
        model = self._model.module if isinstance(self._model, DDP) else self._model
        backbone_dim = int(getattr(getattr(model, "_backbone", None), "embed_dims", 0))
        status_encoder = getattr(model, "_status_encoding", None)
        status_dim = int(getattr(status_encoder, "out_features", 0)) if status_encoder is not None else 0
        if backbone_dim <= 0 or status_dim <= 0:
            return None
        return int(backbone_dim + status_dim)

    def _batched_replay_features(self, replays: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if len(replays) == 0:
            raise RuntimeError("SparseDriveV2 value replay batch is empty")

        bad_indices = [
            idx for idx, replay in enumerate(replays)
            if not self.replay_is_compatible(replay)
        ]
        if len(bad_indices) > 0:
            raise RuntimeError(
                "SparseDriveV2 value replay batch contains incompatible replay entries. "
                f"Bad replay indices: {bad_indices[:8]}"
            )

        camera_feature = replays[0].get("camera_feature", None)
        if not isinstance(camera_feature, dict) or "imgs" not in camera_feature:
            raise RuntimeError("SparseDriveV2 value replay batch is missing camera_feature['imgs']")

        batched_camera = {
            key: torch.cat([rep["camera_feature"][key] for rep in replays], dim=0)
            for key in camera_feature.keys()
        }
        try:
            batched_status = torch.cat([rep["status_feature"] for rep in replays], dim=0)
        except KeyError as exc:
            raise RuntimeError("SparseDriveV2 value replay batch requires status_feature") from exc

        return self._to_device_features(
            {
                "camera_feature": batched_camera,
                "status_feature": batched_status,
            },
            self.device,
        )

    def value_features_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        model = self._model.module if isinstance(self._model, DDP) else self._model
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = self.device
        if len(replays) == 0:
            feature_dim = int(self.value_feature_dim or 0)
            return torch.empty((0, feature_dim), device=model_device, dtype=torch.float32)

        batched_dev = self._to_device_features(self._batched_replay_features(replays), model_device)
        model.eval()
        with torch.inference_mode():
            feature_maps = model._backbone(batched_dev["camera_feature"]["imgs"])
            last_feature_map = feature_maps[-1].to(dtype=torch.float32)
            pooled_vision = last_feature_map.mean(dim=(1, 3, 4))
            status_encoding = model._status_encoding(batched_dev["status_feature"]).to(dtype=torch.float32)
            features = torch.cat([pooled_vision, status_encoding], dim=1)
        return features.detach().clone()

    def value_features_from_observation(self, observation: Dict[str, Any]) -> torch.Tensor:
        return self.value_features_from_observation_batch([observation]).view(-1)

    def value_features_from_observation_batch(
        self,
        observations: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        if len(observations) == 0:
            feature_dim = int(self.value_feature_dim or 0)
            return torch.empty((0, feature_dim), device=self.device, dtype=torch.float32)

        features_list = [self._build_features(obs) for obs in observations]
        camera_keys = list(features_list[0]["camera_feature"].keys())
        batched_camera = {
            key: torch.cat([feat["camera_feature"][key] for feat in features_list], dim=0)
            for key in camera_keys
        }
        batched_status = torch.cat([feat["status_feature"] for feat in features_list], dim=0)
        batched_dev = self._to_device_features(
            {
                "camera_feature": batched_camera,
                "status_feature": batched_status,
            },
            self.device,
        )

        model = self._model.module if isinstance(self._model, DDP) else self._model
        model.eval()
        with torch.inference_mode():
            feature_maps = model._backbone(batched_dev["camera_feature"]["imgs"])
            last_feature_map = feature_maps[-1].to(dtype=torch.float32)
            pooled_vision = last_feature_map.mean(dim=(1, 3, 4))
            status_encoding = model._status_encoding(batched_dev["status_feature"]).to(dtype=torch.float32)
            features = torch.cat([pooled_vision, status_encoding], dim=1)
        return features.detach().clone()

    def _counterfactual_scorer_backend_name(self) -> str:
        backend = str(self._nuscenes_scorer_config.get("backend", "token")).strip().lower()
        if backend in {"pdm", "nuscenes_pdm"}:
            return "nuscenes_pdm"
        if backend in {"pdm_gpu", "nuscenes_pdm_gpu"}:
            return "nuscenes_pdm_gpu"
        return "token"

    def _ensure_nuscenes_token_scorer(self):
        if self._nuscenes_token_scorer is None:
            from framework.algorithms.nuscenes_token_scorer import NuScenesTokenScorer

            scorer_kwargs = {
                key: value
                for key, value in self._nuscenes_scorer_config.items()
                if key != "backend"
            }
            self._nuscenes_token_scorer = NuScenesTokenScorer(
                token2vad_path=nus_cfg.TOKEN2VAD_FILE,
                **scorer_kwargs,
            )
        return self._nuscenes_token_scorer

    def _ensure_nuscenes_pdm_scorer(self):
        if self._nuscenes_pdm_scorer is None:
            from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

            scorer_kwargs = {
                key: value
                for key, value in self._nuscenes_scorer_config.items()
                if key != "backend"
            }
            self._nuscenes_pdm_scorer = NuScenesPDMScorer(
                token2vad_path=nus_cfg.TOKEN2VAD_FILE,
                **scorer_kwargs,
            )
        return self._nuscenes_pdm_scorer

    def _ensure_nuscenes_pdm_gpu_scorer(self):
        if self._nuscenes_pdm_gpu_scorer is None:
            import importlib.util
            import sys
            from pathlib import Path

            module_path = Path(__file__).resolve().parents[1] / "algorithms" / "nuscenes_pdm_backend-GPU.py"
            module_name = "framework.algorithms.nuscenes_pdm_backend_gpu_runtime"
            module = sys.modules.get(module_name)
            if module is None:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"Failed to load GPU scorer backend from {module_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            scorer_cls = getattr(module, "NuScenesPDMScorer")

            scorer_kwargs = {
                key: value
                for key, value in self._nuscenes_scorer_config.items()
                if key != "backend"
            }
            self._nuscenes_pdm_gpu_scorer = scorer_cls(
                token2vad_path=nus_cfg.TOKEN2VAD_FILE,
                **scorer_kwargs,
            )
        return self._nuscenes_pdm_gpu_scorer

    def _ensure_counterfactual_scorer_backend(self):
        backend_name = self._counterfactual_scorer_backend_name()
        if backend_name == "nuscenes_pdm":
            return self._ensure_nuscenes_pdm_scorer()
        if backend_name == "nuscenes_pdm_gpu":
            return self._ensure_nuscenes_pdm_gpu_scorer()
        return self._ensure_nuscenes_token_scorer()

    def pdm_score_counterfactuals_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> torch.Tensor:
        from framework.algorithms.pdm_scorer import _as_score_tensor

        scorer = self._ensure_counterfactual_scorer_backend()
        return _as_score_tensor(scorer.score(replays, traj_xyyaw), device=self.device)

    def dump_counterfactual_debug_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        traj_xyyaw: torch.Tensor,
        candidate_scores: torch.Tensor,
        *,
        out_dir: str,
        step_tag: str,
        top_k: int,
    ) -> None:
        del candidate_scores
        scorer = self._ensure_counterfactual_scorer_backend()
        scorer.dump_debug_artifacts(
            replays,
            traj_xyyaw,
            out_dir=out_dir,
            step_tag=step_tag,
            top_k=top_k,
        )

    def replay_is_compatible(self, replay: Dict[str, Any]) -> bool:
        if not isinstance(replay, dict):
            return False
        camera_feature = replay.get("camera_feature", None)
        status_feature = replay.get("status_feature", None)
        if not isinstance(camera_feature, dict):
            return False
        if not torch.is_tensor(status_feature):
            return False
        return len(camera_feature) > 0

    def logp_from_replay(self, replay: Dict[str, Any], *, eta: float = 1.0) -> torch.Tensor:
        return self.logp_from_replay_batch([replay], eta=float(eta)).view(())

    def act(
        self,
        observation: Dict[str, Any],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ):
        del eta
        return self.sample_sparsedrivev2_with_replay(
            observation,
            mode_idx=mode_idx,
            mode_select=mode_select,
        )

    def act_batch(
        self,
        observations: List[Dict[str, Any]],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ):
        del eta
        return self.sample_sparsedrivev2_with_replay_batch(
            observations,
            mode_idx=mode_idx,
            mode_select=mode_select,
        )
