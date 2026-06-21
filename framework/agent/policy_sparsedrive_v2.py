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


def _apply_trainable_prefixes(
    module: torch.nn.Module,
    prefixes: Sequence[str],
    *,
    frozen_prefixes: Sequence[str] | None = None,
) -> tuple[int, int]:
    normalized = [str(prefix).strip() for prefix in prefixes if str(prefix).strip()]
    normalized_frozen = [str(prefix).strip() for prefix in (frozen_prefixes or []) if str(prefix).strip()]
    if len(normalized) == 0:
        if len(normalized_frozen) > 0:
            for name, param in module.named_parameters():
                if any(name.startswith(prefix) for prefix in normalized_frozen):
                    param.requires_grad = False
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
        if any(name.startswith(prefix) for prefix in normalized_frozen):
            param.requires_grad = False
        if getattr(param, "requires_grad", False):
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
        frozen_prefixes: Sequence[str] | None = None,
        nuscenes_scorer_config: Dict[str, Any] | None = None,
        grpo_num_candidates: int = 0,
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
        self._frozen_prefixes = _normalize_trainable_prefixes(frozen_prefixes)
        self._nuscenes_scorer_config = dict(nuscenes_scorer_config or {})
        self._grpo_num_candidates = max(0, int(grpo_num_candidates or 0))

        self._cfg = self._SparseDriveConfig()
        self._cfg.bkb_path = os.path.join(_SPARSEDRIVE_V2_ROOT, "ckpt", "resnet34.bin")
        self._cfg.path_anchor = os.path.join(_SPARSEDRIVE_V2_ROOT, "ckpt", "kmeans", "path_1024.npy")
        self._cfg.velocity_anchor = os.path.join(_SPARSEDRIVE_V2_ROOT, "ckpt", "kmeans", "velocity_256.npy")
        self._cfg.trajectory_anchor = os.path.join(_SPARSEDRIVE_V2_ROOT, "ckpt", "kmeans", "trajectory_1024_256.npz")

        self._model = self._SparseDriveModel(self._cfg)
        self.to(self.device)
        self._load_weights(self.ckpt_path)
        _apply_trainable_prefixes(
            self._model,
            self._trainable_prefixes,
            frozen_prefixes=self._frozen_prefixes,
        )
        trainable_names, total_tensors, trainable_tensors, total_params, trainable_params = _summarize_parameter_status(self._model)
        if len(self._trainable_prefixes) > 0:
            print(
                "[SparseDriveV2Policy] trainable_prefixes="
                f"{self._trainable_prefixes} frozen_prefixes={self._frozen_prefixes} "
                f"-> trainable tensors {trainable_tensors}/{total_tensors}, "
                f"trainable params {trainable_params}/{total_params}"
            )
            # for name in trainable_names:
            #     print(f"[SparseDriveV2Policy] trainable_param={name}")
            for name in trainable_names[-1:]:
                print(f"[SparseDriveV2Policy] trainable_param[-1]={name}")
        else:
            print(
                "[SparseDriveV2Policy] trainable_prefixes=[] "
                f"frozen_prefixes={self._frozen_prefixes} -> training all currently-enabled "
                "parameters except frozen prefixes: "
                f"trainable tensors {trainable_tensors}/{total_tensors}, trainable params {trainable_params}/{total_params}"
            )

        params = [param for param in self._model.parameters() if getattr(param, "requires_grad", False)]
        self._optimizer: torch.optim.Optimizer | None = None
        if len(params) > 0:
            self._optimizer = torch.optim.Adam(params, lr=float(rl_lr))

        self._last_missing_feature_fields: List[str] = []
        self._teacher_model: torch.nn.Module | None = None
        self._nuscenes_pdm_scorer: Any | None = None
        self._nuscenes_craft_scorer: Any | None = None

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

    def _forward_policy_on_model(
        self,
        model: torch.nn.Module,
        features_dev: Dict[str, Any],
        targets: Dict[str, Any] | None = None,
    ) -> Dict[str, torch.Tensor]:
        out, _loss_dict = model(features_dev, targets={} if targets is None else targets)
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

    def _forward_policy(self, features_dev: Dict[str, Any], targets: Dict[str, Any] | None = None) -> Dict[str, torch.Tensor]:
        model = self._unwrap_model(self._model)
        return self._forward_policy_on_model(model, features_dev, targets=targets)

    def _forward_policy_with_forced_global_indices(
        self,
        features_dev: Dict[str, Any],
        forced_global_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        model = self._unwrap_model(self._model)
        head = getattr(model, "_trajectory_head", None)
        decoder = getattr(head, "decoder", None)
        layers = getattr(decoder, "layers", None)
        if head is None or layers is None:
            return self._forward_policy_on_model(
                model,
                features_dev,
                targets={"forced_global_indices": forced_global_indices},
            )

        camera_feature = dict(features_dev["camera_feature"])
        status_feature = features_dev["status_feature"]
        status_encoding = model._status_encoding(status_feature)
        feature_maps = model._backbone(camera_feature["imgs"])
        camera_feature["feature_maps"] = feature_maps

        batch_size = int(status_encoding.shape[0])
        path_vocab = head.path_vocab.data[None].repeat(batch_size, 1, 1, 1)
        vel_vocab = head.vel_vocab.data[None].repeat(batch_size, 1, 1)
        traj_vocab = head.traj_vocab.data[None].repeat(batch_size, 1, 1, 1, 1)
        traj_mask = head.traj_mask.data[None].repeat(batch_size, 1, 1, 1)
        path_indices = torch.arange(path_vocab.shape[1], device=path_vocab.device, dtype=torch.long)[None].repeat(batch_size, 1)
        vel_indices = torch.arange(vel_vocab.shape[1], device=vel_vocab.device, dtype=torch.long)[None].repeat(batch_size, 1)
        path_embed = head.path_pos_embed(path_vocab.flatten(-2, -1))
        vel_embed = head.vel_pos_embed(vel_vocab)

        total_num_vel = int(vel_vocab.shape[1])
        forced_global_indices = forced_global_indices.to(device=status_feature.device, dtype=torch.long).view(-1)
        forced_path_indices = torch.div(forced_global_indices, total_num_vel, rounding_mode="floor")
        forced_vel_indices = forced_global_indices.remainder(total_num_vel)
        batch_indices = torch.arange(batch_size, device=status_feature.device, dtype=torch.long)

        from navsim.agents.sparsedrive.ops import deformable_format

        output: Dict[str, torch.Tensor] = {}
        for layer in layers:
            num_path = int(path_embed.shape[1])
            num_vel = int(vel_embed.shape[1])
            img_value = camera_feature["feature_maps"][-1].permute(0, 1, 3, 4, 2).flatten(1, 3)
            deform_value = deformable_format(camera_feature["feature_maps"])

            path_embed = path_embed + status_encoding.unsqueeze(1)
            vel_embed = vel_embed + status_encoding.unsqueeze(1)

            path_vocab_flat = path_vocab[..., :2].flatten(-2)
            path_embed = layer.p_deform_model(
                path_embed,
                path_vocab_flat,
                None,
                deform_value,
                camera_feature,
                None,
            )
            path_embed = path_embed + layer.p_dropout1(layer.p_attention(path_embed, path_embed, path_embed)[0])
            path_embed = layer.p_norm1(path_embed)
            path_embed = path_embed + layer.p_dropout2(layer.p_ffn(path_embed))
            path_embed = layer.p_norm2(path_embed)
            path_scores = layer.path_mlp(path_embed).squeeze(-1)

            vel_embed = vel_embed + layer.v_img_attention(vel_embed, img_value, img_value)[0]
            vel_embed = vel_embed + layer.v_dropout1(layer.v_attention(vel_embed, vel_embed, vel_embed)[0])
            vel_embed = layer.v_norm1(vel_embed)
            vel_embed = vel_embed + layer.v_dropout2(layer.v_ffn(vel_embed))
            vel_embed = layer.v_norm2(vel_embed)
            vel_scores = layer.vel_mlp(vel_embed).squeeze(-1)

            if num_path > layer._config.path_filter_num[layer.decoder_idx]:
                _scores, topk_path_pos = torch.topk(
                    path_scores,
                    layer._config.path_filter_num[layer.decoder_idx],
                    dim=1,
                )
                filter_path_embed = torch.gather(path_embed, 1, topk_path_pos.unsqueeze(-1).expand(-1, -1, path_embed.shape[-1]))
                filter_path_vocab = torch.gather(
                    path_vocab,
                    1,
                    topk_path_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, path_vocab.shape[-2], path_vocab.shape[-1]),
                )
                filter_traj_vocab = torch.gather(
                    traj_vocab,
                    1,
                    topk_path_pos[:, :, None, None, None].expand(
                        -1,
                        -1,
                        traj_vocab.shape[-3],
                        traj_vocab.shape[-2],
                        traj_vocab.shape[-1],
                    ),
                )
                filter_traj_mask = torch.gather(
                    traj_mask,
                    1,
                    topk_path_pos[:, :, None, None].expand(-1, -1, traj_mask.shape[-2], traj_mask.shape[-1]),
                )
                filter_path_indices = torch.gather(path_indices, 1, topk_path_pos)
            else:
                filter_path_embed = path_embed
                filter_path_vocab = path_vocab
                filter_traj_vocab = traj_vocab
                filter_traj_mask = traj_mask
                filter_path_indices = path_indices

            source_path_pos = (path_indices == forced_path_indices[:, None]).to(dtype=torch.long).argmax(dim=1)
            path_missing = ~(filter_path_indices == forced_path_indices[:, None]).any(dim=1)
            forced_path_embed = path_embed[batch_indices, source_path_pos].unsqueeze(1)
            forced_path_vocab = path_vocab[batch_indices, source_path_pos].unsqueeze(1)
            forced_path_traj_vocab = traj_vocab[batch_indices, source_path_pos].unsqueeze(1)
            forced_path_traj_mask = traj_mask[batch_indices, source_path_pos].unsqueeze(1)
            filter_path_embed = torch.cat(
                [filter_path_embed, torch.where(path_missing[:, None, None], forced_path_embed, filter_path_embed[:, :1])],
                dim=1,
            )
            filter_path_vocab = torch.cat(
                [filter_path_vocab, torch.where(path_missing[:, None, None, None], forced_path_vocab, filter_path_vocab[:, :1])],
                dim=1,
            )
            filter_traj_vocab = torch.cat(
                [
                    filter_traj_vocab,
                    torch.where(path_missing[:, None, None, None, None], forced_path_traj_vocab, filter_traj_vocab[:, :1]),
                ],
                dim=1,
            )
            filter_traj_mask = torch.cat(
                [
                    filter_traj_mask,
                    torch.where(path_missing[:, None, None, None], forced_path_traj_mask, filter_traj_mask[:, :1]),
                ],
                dim=1,
            )
            filter_path_indices = torch.cat(
                [filter_path_indices, torch.where(path_missing[:, None], forced_path_indices[:, None], filter_path_indices[:, :1])],
                dim=1,
            )

            pre_vel_traj_vocab = filter_traj_vocab
            pre_vel_traj_mask = filter_traj_mask
            if num_vel > layer._config.velocity_filter_num[layer.decoder_idx]:
                _scores, topk_vel_pos = torch.topk(
                    vel_scores,
                    layer._config.velocity_filter_num[layer.decoder_idx],
                    dim=1,
                )
                filter_vel_embed = torch.gather(vel_embed, 1, topk_vel_pos.unsqueeze(-1).expand(-1, -1, vel_embed.shape[-1]))
                filter_vel_vocab = torch.gather(vel_vocab, 1, topk_vel_pos.unsqueeze(-1).expand(-1, -1, vel_vocab.shape[-1]))
                filter_traj_vocab = torch.gather(
                    pre_vel_traj_vocab,
                    2,
                    topk_vel_pos[:, None, :, None, None].expand(
                        -1,
                        pre_vel_traj_vocab.shape[-4],
                        -1,
                        pre_vel_traj_vocab.shape[-2],
                        pre_vel_traj_vocab.shape[-1],
                    ),
                )
                filter_traj_mask = torch.gather(
                    pre_vel_traj_mask,
                    2,
                    topk_vel_pos[:, None, :, None].expand(-1, pre_vel_traj_mask.shape[-3], -1, pre_vel_traj_mask.shape[-1]),
                )
                filter_vel_indices = torch.gather(vel_indices, 1, topk_vel_pos)
            else:
                filter_vel_embed = vel_embed
                filter_vel_vocab = vel_vocab
                filter_vel_indices = vel_indices

            source_vel_pos = (vel_indices == forced_vel_indices[:, None]).to(dtype=torch.long).argmax(dim=1)
            vel_missing = ~(filter_vel_indices == forced_vel_indices[:, None]).any(dim=1)
            forced_vel_embed = vel_embed[batch_indices, source_vel_pos].unsqueeze(1)
            forced_vel_vocab = vel_vocab[batch_indices, source_vel_pos].unsqueeze(1)
            forced_vel_traj_vocab = pre_vel_traj_vocab[
                batch_indices[:, None],
                torch.arange(pre_vel_traj_vocab.shape[1], device=pre_vel_traj_vocab.device)[None, :],
                source_vel_pos[:, None],
            ].unsqueeze(2)
            forced_vel_traj_mask = pre_vel_traj_mask[
                batch_indices[:, None],
                torch.arange(pre_vel_traj_mask.shape[1], device=pre_vel_traj_mask.device)[None, :],
                source_vel_pos[:, None],
            ].unsqueeze(2)
            filter_vel_embed = torch.cat(
                [filter_vel_embed, torch.where(vel_missing[:, None, None], forced_vel_embed, filter_vel_embed[:, :1])],
                dim=1,
            )
            filter_vel_vocab = torch.cat(
                [filter_vel_vocab, torch.where(vel_missing[:, None, None], forced_vel_vocab, filter_vel_vocab[:, :1])],
                dim=1,
            )
            filter_traj_vocab = torch.cat(
                [
                    filter_traj_vocab,
                    torch.where(vel_missing[:, None, None, None, None], forced_vel_traj_vocab, filter_traj_vocab[:, :, :1]),
                ],
                dim=2,
            )
            filter_traj_mask = torch.cat(
                [
                    filter_traj_mask,
                    torch.where(vel_missing[:, None, None, None], forced_vel_traj_mask, filter_traj_mask[:, :, :1]),
                ],
                dim=2,
            )
            filter_vel_indices = torch.cat(
                [filter_vel_indices, torch.where(vel_missing[:, None], forced_vel_indices[:, None], filter_vel_indices[:, :1])],
                dim=1,
            )

            base_path_indices = filter_path_indices[:, :-1]
            base_vel_indices = filter_vel_indices[:, :-1]

            path_embed = filter_path_embed
            vel_embed = filter_vel_embed
            path_vocab = filter_path_vocab
            vel_vocab = filter_vel_vocab
            traj_vocab = filter_traj_vocab
            traj_mask = filter_traj_mask
            path_indices = filter_path_indices
            vel_indices = filter_vel_indices

            if layer.decoder_idx != layer._config.decoder_num_layers - 1:
                continue

            traj_embed = path_embed.unsqueeze(2) + vel_embed.unsqueeze(1)
            traj_embed = traj_embed.flatten(1, 2)
            filter_traj_vocab_flat = traj_vocab[..., :2].flatten(1, 2).flatten(-2)
            traj_embed = layer.t_deform_model(
                traj_embed,
                filter_traj_vocab_flat,
                None,
                deform_value,
                camera_feature,
                None,
            )
            traj_embed = traj_embed + layer.t_dropout1(layer.t_attention(traj_embed, traj_embed, traj_embed)[0])
            traj_embed = layer.t_norm1(traj_embed)
            traj_embed = traj_embed + layer.t_dropout2(layer.t_ffn(traj_embed))
            traj_embed = layer.t_norm2(traj_embed)
            traj_scores = layer.traj_mlp(traj_embed).squeeze(-1)
            metric_logit: Dict[str, torch.Tensor] = {}
            for metric in layer._config.metrics:
                metric_logit[metric] = layer.metric_heads[metric](traj_embed).squeeze(-1)

            candidate_trajectories = traj_vocab.flatten(1, 2)
            candidate_path_indices = path_indices[:, :, None].expand(-1, -1, vel_indices.shape[1]).flatten(1, 2)
            candidate_vel_indices = vel_indices[:, None, :].expand(-1, path_indices.shape[1], -1).flatten(1, 2)
            candidate_global_indices = candidate_path_indices * int(total_num_vel) + candidate_vel_indices
            if layer._config.dataset_version == "v1":
                scores = (
                    metric_logit["no_at_fault_collisions"].sigmoid()
                    * metric_logit["drivable_area_compliance"].sigmoid()
                ) * (
                    5 * metric_logit["time_to_collision_within_bound"].sigmoid()
                    + 5 * metric_logit["ego_progress"].sigmoid()
                    + 2 * metric_logit["comfort"].sigmoid()
                )
            elif layer._config.dataset_version == "v2":
                scores = (
                    metric_logit["no_at_fault_collisions"].sigmoid()
                    * metric_logit["drivable_area_compliance"].sigmoid()
                    * metric_logit["driving_direction_compliance"].sigmoid()
                    * metric_logit["traffic_light_compliance"].sigmoid()
                ) * (
                    5 * metric_logit["time_to_collision_within_bound"].sigmoid()
                    + 5 * metric_logit["ego_progress"].sigmoid()
                    + 2 * metric_logit["lane_keeping"].sigmoid()
                    + 2 * metric_logit["history_comfort"].sigmoid()
                )
            else:
                raise ValueError(f"Unsupported SparseDrive dataset_version={layer._config.dataset_version!r}")

            base_candidate_path_indices = base_path_indices[:, :, None].expand(-1, -1, base_vel_indices.shape[1]).flatten(1, 2)
            base_candidate_vel_indices = base_vel_indices[:, None, :].expand(-1, base_path_indices.shape[1], -1).flatten(1, 2)
            base_candidate_global_indices = base_candidate_path_indices * int(total_num_vel) + base_candidate_vel_indices
            base_source_pos = (
                candidate_global_indices[:, :, None] == base_candidate_global_indices[:, None, :]
            ).to(dtype=torch.long).argmax(dim=1)
            forced_source_pos = (candidate_global_indices == forced_global_indices[:, None]).to(dtype=torch.long).argmax(dim=1)

            base_candidate_trajectories = candidate_trajectories.gather(
                1,
                base_source_pos[:, :, None, None].expand(-1, -1, candidate_trajectories.shape[2], candidate_trajectories.shape[3]),
            )
            base_scores = torch.gather(scores, 1, base_source_pos)
            base_traj_scores = torch.gather(traj_scores, 1, base_source_pos)
            forced_candidate_trajectories = candidate_trajectories[batch_indices, forced_source_pos].unsqueeze(1)
            forced_scores = scores[batch_indices, forced_source_pos].unsqueeze(1)
            forced_traj_scores = traj_scores[batch_indices, forced_source_pos].unsqueeze(1)
            present_in_base = (base_candidate_global_indices == forced_global_indices[:, None]).any(dim=1)
            forced_scores = torch.where(
                present_in_base[:, None],
                torch.full_like(forced_scores, float("-inf")),
                forced_scores,
            )
            candidate_trajectories = torch.cat([base_candidate_trajectories, forced_candidate_trajectories], dim=1)
            scores = torch.cat([base_scores, forced_scores], dim=1)
            traj_scores = torch.cat([base_traj_scores, forced_traj_scores], dim=1)
            candidate_path_indices = torch.cat([base_candidate_path_indices, forced_path_indices[:, None]], dim=1)
            candidate_vel_indices = torch.cat([base_candidate_vel_indices, forced_vel_indices[:, None]], dim=1)
            candidate_global_indices = torch.cat([base_candidate_global_indices, forced_global_indices[:, None]], dim=1)
            candidate_is_forced = torch.zeros_like(candidate_global_indices, dtype=torch.bool)
            candidate_is_forced[:, -1] = True

            bs_indices = torch.arange(scores.shape[0], device=scores.device)
            mode_indices = scores.argmax(1)
            output["trajectory"] = candidate_trajectories[bs_indices, mode_indices]
            output["candidate_trajectories"] = candidate_trajectories
            output["candidate_scores"] = scores
            output["candidate_global_indices"] = candidate_global_indices
            output["candidate_path_indices"] = candidate_path_indices
            output["candidate_vel_indices"] = candidate_vel_indices
            output["candidate_is_forced"] = candidate_is_forced
            output["candidate_traj_logits"] = traj_scores
            for metric_name, metric_value in metric_logit.items():
                output[f"metric_logits/{metric_name}"] = torch.cat(
                    [
                        torch.gather(metric_value, 1, base_source_pos),
                        metric_value[batch_indices, forced_source_pos].unsqueeze(1),
                    ],
                    dim=1,
                )

        if not output:
            raise RuntimeError("SparseDriveV2 forced forward did not produce trajectory outputs")
        return output

    def _trajectory_anchor_lookup(self) -> tuple[Dict[bytes, int], int, tuple[int, ...]]:
        model = self._unwrap_model(self._model)
        head = getattr(model, "_trajectory_head", None)
        traj_vocab = getattr(head, "traj_vocab", None)
        if traj_vocab is None or not torch.is_tensor(traj_vocab):
            raise RuntimeError("SparseDriveV2 global action identity requires model._trajectory_head.traj_vocab")

        anchor = traj_vocab.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if anchor.ndim != 4:
            raise RuntimeError(f"SparseDriveV2 traj_vocab must have shape (path, vel, horizon, dims); got {tuple(anchor.shape)}")
        cache_key = tuple(int(v) for v in anchor.shape)
        cached = getattr(self, "_trajectory_anchor_lookup_cache", None)
        if isinstance(cached, tuple) and len(cached) == 3 and cached[0] == cache_key:
            return cached[1], int(cached[2]), cache_key

        num_path, num_vel = int(anchor.shape[0]), int(anchor.shape[1])
        flat = anchor.reshape(num_path * num_vel, int(anchor.shape[2]), int(anchor.shape[3]))
        lookup: Dict[bytes, int] = {}
        for global_idx in range(int(flat.shape[0])):
            lookup[flat[global_idx].numpy().tobytes()] = int(global_idx)
        self._trajectory_anchor_lookup_cache = (cache_key, lookup, int(num_vel))
        return lookup, int(num_vel), cache_key

    def _candidate_identity_from_trajectories(self, candidate_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not torch.is_tensor(candidate_trajectories) or candidate_trajectories.ndim != 4:
            raise RuntimeError(
                "SparseDriveV2 candidate identity requires candidate_trajectories with shape "
                f"(batch, candidates, horizon, dims); got {tuple(candidate_trajectories.shape) if torch.is_tensor(candidate_trajectories) else type(candidate_trajectories)!r}"
            )
        lookup, num_vel, anchor_shape = self._trajectory_anchor_lookup()
        if tuple(candidate_trajectories.shape[-2:]) != tuple(anchor_shape[-2:]):
            raise RuntimeError(
                "SparseDriveV2 candidate trajectory shape does not match traj_vocab anchors: "
                f"candidate={tuple(candidate_trajectories.shape[-2:])} anchor={tuple(anchor_shape[-2:])}"
            )

        candidates_cpu = candidate_trajectories.detach().to(device="cpu", dtype=torch.float32).contiguous()
        batch_size, num_candidates = int(candidates_cpu.shape[0]), int(candidates_cpu.shape[1])
        global_indices = torch.empty((batch_size, num_candidates), dtype=torch.long)
        for batch_idx in range(batch_size):
            for candidate_idx in range(num_candidates):
                key = candidates_cpu[batch_idx, candidate_idx].numpy().tobytes()
                global_idx = lookup.get(key, None)
                if global_idx is None:
                    raise RuntimeError(
                        "SparseDriveV2 candidate trajectory was not found in the fixed traj_vocab; "
                        f"batch={batch_idx} candidate={candidate_idx}"
                    )
                global_indices[batch_idx, candidate_idx] = int(global_idx)

        path_indices = torch.div(global_indices, int(num_vel), rounding_mode="floor")
        vel_indices = global_indices.remainder(int(num_vel))
        return {
            "candidate_global_indices": global_indices.to(device=candidate_trajectories.device),
            "candidate_path_indices": path_indices.to(device=candidate_trajectories.device),
            "candidate_vel_indices": vel_indices.to(device=candidate_trajectories.device),
        }

    def _candidate_identity_from_outputs(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        identity = self._candidate_identity_from_trajectories(out["candidate_trajectories"])
        out.update(identity)
        return identity

    @staticmethod
    def _global_mode_indices_from_replay(replays: Sequence[Dict[str, Any]], *, device: torch.device) -> torch.Tensor:
        values: List[int] = []
        for idx, replay in enumerate(replays):
            if "global_mode_idx" not in replay:
                raise RuntimeError(f"SparseDriveV2 replay missing global_mode_idx at index {idx}")
            values.append(int(replay["global_mode_idx"]))
        return torch.as_tensor(values, dtype=torch.long, device=device)

    @staticmethod
    def _logp_for_global_modes_or_missing(
        *,
        score_logits: torch.Tensor,
        candidate_global_indices: torch.Tensor,
        target_global_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_global_indices.shape[:2] != score_logits.shape[:2]:
            raise RuntimeError(
                "candidate_global_indices must align with candidate_scores; "
                f"indices={tuple(candidate_global_indices.shape)} scores={tuple(score_logits.shape)}"
            )
        matches = candidate_global_indices.to(device=score_logits.device, dtype=torch.long) == target_global_indices[:, None]
        present = matches.any(dim=1)
        local_indices = matches.to(dtype=torch.long).argmax(dim=1)
        logp_all = torch.log_softmax(score_logits, dim=1)
        batch_indices = torch.arange(score_logits.shape[0], device=score_logits.device)
        logp = logp_all[batch_indices, local_indices]
        return logp, present

    @classmethod
    def _logp_for_global_modes(
        cls,
        *,
        score_logits: torch.Tensor,
        candidate_global_indices: torch.Tensor,
        target_global_indices: torch.Tensor,
    ) -> torch.Tensor:
        logp, present = cls._logp_for_global_modes_or_missing(
            score_logits=score_logits,
            candidate_global_indices=candidate_global_indices,
            target_global_indices=target_global_indices,
        )
        if not bool(present.all().item()):
            missing = (~present).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
            raise RuntimeError(
                "SparseDriveV2 replay global_mode_idx was not present in current candidate_trajectories; "
                f"missing batch indices: {missing[:8]}"
            )
        return logp

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
        candidate_identity = self._candidate_identity_from_outputs(out)
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
        all_global_indices = candidate_identity["candidate_global_indices"].detach().cpu()
        all_path_indices = candidate_identity["candidate_path_indices"].detach().cpu()
        all_vel_indices = candidate_identity["candidate_vel_indices"].detach().cpu()
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
            selected_global_idx = int(all_global_indices[idx, selected_mode_idx].item())
            selected_path_idx = int(all_path_indices[idx, selected_mode_idx].item())
            selected_vel_idx = int(all_vel_indices[idx, selected_mode_idx].item())
            replay.update(
                {
                    "selected_path_idx": int(selected_path_idx),
                    "selected_vel_idx": int(selected_vel_idx),
                    "global_mode_idx": int(selected_global_idx),
                    "traj_xyyaw": traj_xyyaw.detach().cpu().clone(),
                    "candidate_scores": scores.detach().cpu().clone(),
                    "execute_mode": self._execute_mode,
                    "feature_missing_fields": list(features.get("feature_missing_fields", [])),
                }
            )
            if self._grpo_num_candidates > 0:
                strict_candidates = self._select_strict_grpo_old_policy_candidates(
                    score_logits=score_logits[idx : idx + 1],
                    candidate_trajs=out["candidate_trajectories"][idx : idx + 1],
                    candidate_global_indices=all_global_indices[idx : idx + 1].to(device=score_logits.device),
                    num_candidates=int(self._grpo_num_candidates),
                )
                replay.update(
                    {
                        "grpo_candidate_mode_indices": strict_candidates["mode_indices"][0].detach().cpu().clone(),
                        "grpo_candidate_old_log_probs": strict_candidates["old_log_probs"][0].detach().cpu().clone(),
                        "grpo_candidate_traj_xyyaw": strict_candidates["traj_xyyaw"][0].detach().cpu().clone(),
                        "grpo_candidate_old_score_logits": strict_candidates["score_logits"][0].detach().cpu().clone(),
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
        if len(replays) > 1:
            parts = [self.logp_from_replay_batch([replay], eta=1.0).view(1) for replay in replays]
            return torch.cat(parts, dim=0)

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
        candidate_identity = self._candidate_identity_from_outputs(out)
        target_global_indices = self._global_mode_indices_from_replay(replays, device=score_logits.device)
        logp, present = self._logp_for_global_modes_or_missing(
            score_logits=score_logits,
            candidate_global_indices=candidate_identity["candidate_global_indices"],
            target_global_indices=target_global_indices,
        )
        if bool(present.all().item()):
            return logp

        missing = (~present).nonzero(as_tuple=False).view(-1)
        if len(replays) == 1:
            return self._forced_logp_from_replay_batch(replays)

        for missing_idx_tensor in missing:
            missing_idx = int(missing_idx_tensor.item())
            logp[missing_idx] = self.logp_from_replay_batch([replays[missing_idx]], eta=1.0).view(())
        return logp

    def _forced_logp_from_replay_batch(self, replays: Sequence[Dict[str, Any]]) -> torch.Tensor:
        batched_dev = self._batched_replay_features(replays)
        target_global_indices = self._global_mode_indices_from_replay(replays, device=self.device)
        out = self._forward_policy_with_forced_global_indices(batched_dev, target_global_indices)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        candidate_identity = self._candidate_identity_from_outputs(out)
        target_global_indices = target_global_indices.to(device=score_logits.device)
        logp, present = self._logp_for_global_modes_or_missing(
            score_logits=score_logits,
            candidate_global_indices=candidate_identity["candidate_global_indices"],
            target_global_indices=target_global_indices,
        )
        if bool(present.all().item()):
            return logp
        missing = (~present).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
        raise RuntimeError(
            "SparseDriveV2 replay global_mode_idx was not present after forced scoring; "
            f"missing batch indices: {missing[:8]}"
        )

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

    @staticmethod
    def _select_strict_grpo_old_policy_candidates(
        *,
        score_logits: torch.Tensor,
        candidate_trajs: torch.Tensor,
        candidate_global_indices: torch.Tensor,
        num_candidates: int,
    ) -> Dict[str, torch.Tensor]:
        if score_logits.ndim != 2:
            raise RuntimeError(f"score_logits must be 2D (batch,candidates), got {tuple(score_logits.shape)}")
        if candidate_trajs.ndim != 4:
            raise RuntimeError(f"candidate_trajs must be 4D (batch,candidates,horizon,dims), got {tuple(candidate_trajs.shape)}")
        if tuple(candidate_global_indices.shape) != tuple(score_logits.shape):
            raise RuntimeError(
                "candidate_global_indices must match score_logits shape; "
                f"indices={tuple(candidate_global_indices.shape)} logits={tuple(score_logits.shape)}"
            )
        k = max(1, min(int(num_candidates), int(score_logits.shape[1])))
        probs = torch.softmax(score_logits, dim=1)
        local_indices = torch.multinomial(probs, num_samples=k, replacement=False)
        gather_idx_scores = local_indices
        gather_idx_traj = local_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            int(candidate_trajs.shape[2]),
            int(candidate_trajs.shape[3]),
        )
        selected_trajs = torch.gather(candidate_trajs, dim=1, index=gather_idx_traj)
        selected_logits = torch.gather(score_logits, dim=1, index=gather_idx_scores)
        selected_global_indices = torch.gather(
            candidate_global_indices.to(device=score_logits.device, dtype=torch.long),
            dim=1,
            index=gather_idx_scores,
        )
        old_log_probs = torch.gather(torch.log_softmax(score_logits, dim=1), dim=1, index=gather_idx_scores)
        selected_trajs_flat = selected_trajs.reshape(-1, selected_trajs.shape[2], selected_trajs.shape[3])
        traj_xyyaw = SparseDriveV2Policy._traj_batch_to_xyyaw(selected_trajs_flat).reshape(
            selected_trajs.shape[0],
            selected_trajs.shape[1],
            selected_trajs.shape[2],
            3,
        )
        return {
            "traj_xyyaw": traj_xyyaw.detach(),
            "old_log_probs": old_log_probs,
            "mode_indices": selected_global_indices,
            "local_indices": local_indices.to(dtype=torch.long),
            "score_logits": selected_logits,
        }

    @staticmethod
    def _stored_grpo_candidate_mode_indices_from_replay(
        replays: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        values: List[torch.Tensor] = []
        expected_k: int | None = None
        for idx, replay in enumerate(replays):
            raw = replay.get("grpo_candidate_mode_indices", None)
            if raw is None:
                raise RuntimeError(f"Strict GRPO replay missing grpo_candidate_mode_indices at index {idx}")
            tensor = torch.as_tensor(raw, dtype=torch.long, device=device).view(-1)
            if expected_k is None:
                expected_k = int(tensor.numel())
            elif int(tensor.numel()) != expected_k:
                raise RuntimeError(
                    "Strict GRPO candidate count mismatch across replay entries: "
                    f"expected={expected_k} got={int(tensor.numel())} index={idx}"
                )
            values.append(tensor)
        return torch.stack(values, dim=0) if values else torch.empty((0, 0), dtype=torch.long, device=device)

    @staticmethod
    def _stored_grpo_candidate_old_log_probs_from_replay(
        replays: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        values: List[torch.Tensor] = []
        expected_k: int | None = None
        for idx, replay in enumerate(replays):
            raw = replay.get("grpo_candidate_old_log_probs", None)
            if raw is None:
                raise RuntimeError(f"Strict GRPO replay missing grpo_candidate_old_log_probs at index {idx}")
            tensor = torch.as_tensor(raw, dtype=torch.float32, device=device).view(-1)
            if expected_k is None:
                expected_k = int(tensor.numel())
            elif int(tensor.numel()) != expected_k:
                raise RuntimeError(
                    "Strict GRPO old logp count mismatch across replay entries: "
                    f"expected={expected_k} got={int(tensor.numel())} index={idx}"
                )
            values.append(tensor)
        return torch.stack(values, dim=0) if values else torch.empty((0, 0), dtype=torch.float32, device=device)

    @staticmethod
    def _stored_grpo_candidate_traj_xyyaw_from_replay(
        replays: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        values: List[torch.Tensor] = []
        expected_shape: tuple[int, ...] | None = None
        for idx, replay in enumerate(replays):
            raw = replay.get("grpo_candidate_traj_xyyaw", None)
            if raw is None:
                raise RuntimeError(f"Strict GRPO replay missing grpo_candidate_traj_xyyaw at index {idx}")
            tensor = torch.as_tensor(raw, dtype=torch.float32, device=device)
            if tensor.ndim != 3 or int(tensor.shape[-1]) < 3:
                raise RuntimeError(
                    "Strict GRPO replay grpo_candidate_traj_xyyaw must have shape (candidates,horizon,3+); "
                    f"got {tuple(tensor.shape)} at index {idx}"
                )
            tensor = tensor[..., :3]
            if expected_shape is None:
                expected_shape = tuple(int(v) for v in tensor.shape)
            elif tuple(int(v) for v in tensor.shape) != expected_shape:
                raise RuntimeError(
                    "Strict GRPO candidate trajectory shape mismatch across replay entries: "
                    f"expected={expected_shape} got={tuple(tensor.shape)} index={idx}"
                )
            values.append(tensor)
        return torch.stack(values, dim=0) if values else torch.empty((0, 0, 0, 3), dtype=torch.float32, device=device)

    @staticmethod
    def _new_log_probs_for_stored_grpo_candidates_from_outputs(
        *,
        score_logits: torch.Tensor,
        candidate_global_indices: torch.Tensor,
        replays: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        target_indices = SparseDriveV2Policy._stored_grpo_candidate_mode_indices_from_replay(
            replays,
            device=score_logits.device,
        )
        if int(target_indices.shape[0]) != int(score_logits.shape[0]):
            raise RuntimeError(
                "Strict GRPO target batch mismatch: "
                f"targets={tuple(target_indices.shape)} logits={tuple(score_logits.shape)}"
            )
        logp_all = torch.log_softmax(score_logits, dim=1)
        rows: List[torch.Tensor] = []
        for batch_idx in range(int(score_logits.shape[0])):
            matches = candidate_global_indices[batch_idx].to(device=score_logits.device, dtype=torch.long).view(1, -1) == target_indices[
                batch_idx
            ].view(-1, 1)
            present = matches.any(dim=1)
            if not bool(present.all().item()):
                missing = target_indices[batch_idx][~present].detach().cpu().tolist()
                raise RuntimeError(
                    "Strict GRPO stored candidate indices are absent from current candidate set; "
                    f"batch={batch_idx} missing={missing[:8]}"
                )
            local_indices = matches.to(dtype=torch.long).argmax(dim=1)
            rows.append(logp_all[batch_idx].gather(0, local_indices))
        return torch.stack(rows, dim=0) if rows else torch.empty((0, 0), device=score_logits.device, dtype=torch.float32)

    @staticmethod
    def _score_logits_for_stored_grpo_candidates_from_outputs(
        *,
        score_logits: torch.Tensor,
        candidate_global_indices: torch.Tensor,
        replays: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        target_indices = SparseDriveV2Policy._stored_grpo_candidate_mode_indices_from_replay(
            replays,
            device=score_logits.device,
        )
        if int(target_indices.shape[0]) != int(score_logits.shape[0]):
            raise RuntimeError(
                "Strict GRPO target batch mismatch: "
                f"targets={tuple(target_indices.shape)} logits={tuple(score_logits.shape)}"
            )
        rows: List[torch.Tensor] = []
        for batch_idx in range(int(score_logits.shape[0])):
            matches = candidate_global_indices[batch_idx].to(device=score_logits.device, dtype=torch.long).view(1, -1) == target_indices[
                batch_idx
            ].view(-1, 1)
            present = matches.any(dim=1)
            if not bool(present.all().item()):
                missing = target_indices[batch_idx][~present].detach().cpu().tolist()
                raise RuntimeError(
                    "Strict GRPO stored candidate indices are absent from current candidate set; "
                    f"batch={batch_idx} missing={missing[:8]}"
                )
            local_indices = matches.to(dtype=torch.long).argmax(dim=1)
            rows.append(score_logits[batch_idx].gather(0, local_indices))
        return torch.stack(rows, dim=0) if rows else torch.empty((0, 0), device=score_logits.device, dtype=torch.float32)

    def _forced_grpo_logp_and_logit_for_global_index(
        self,
        replay: Dict[str, Any],
        *,
        global_index: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forced_replay = dict(replay)
        forced_replay["global_mode_idx"] = int(global_index)
        batched_dev = self._batched_replay_features([forced_replay])
        target = torch.as_tensor([int(global_index)], dtype=torch.long, device=self.device)
        out = self._forward_policy_with_forced_global_indices(batched_dev, target)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        candidate_identity = self._candidate_identity_from_outputs(out)
        logp, present = self._logp_for_global_modes_or_missing(
            score_logits=score_logits,
            candidate_global_indices=candidate_identity["candidate_global_indices"],
            target_global_indices=target.to(device=score_logits.device),
        )
        if not bool(present.all().item()):
            raise RuntimeError(f"Strict GRPO forced scoring could not recover global_mode_idx={int(global_index)}")
        matches = candidate_identity["candidate_global_indices"].to(device=score_logits.device, dtype=torch.long) == target[
            :, None
        ].to(device=score_logits.device)
        local_index = matches.to(dtype=torch.long).argmax(dim=1)
        forced_logit = score_logits[0].gather(0, local_index.view(-1))[0]
        return logp.view(()).to(device=device), forced_logit.to(device=device)

    def _strict_grpo_log_probs_and_logits_for_stored_candidates(
        self,
        replay: Dict[str, Any],
        *,
        score_logits: torch.Tensor,
        candidate_global_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_indices = self._stored_grpo_candidate_mode_indices_from_replay(
            [replay],
            device=score_logits.device,
        ).view(-1)
        if int(score_logits.shape[0]) != 1:
            raise RuntimeError(
                "Strict GRPO per-replay logp recovery expects single-sample logits; "
                f"got {tuple(score_logits.shape)}"
            )
        logp_all = torch.log_softmax(score_logits, dim=1)
        new_log_probs: List[torch.Tensor] = []
        selected_logits: List[torch.Tensor] = []
        global_indices = candidate_global_indices[0].to(device=score_logits.device, dtype=torch.long).view(-1)
        for target in target_indices:
            matches = global_indices == target.to(device=score_logits.device, dtype=torch.long)
            if bool(matches.any().item()):
                local_index = matches.to(dtype=torch.long).argmax(dim=0).view(())
                new_log_probs.append(logp_all[0].gather(0, local_index.view(1))[0])
                selected_logits.append(score_logits[0].gather(0, local_index.view(1))[0])
                continue
            forced_logp, forced_logit = self._forced_grpo_logp_and_logit_for_global_index(
                replay,
                global_index=int(target.detach().cpu().item()),
                device=score_logits.device,
            )
            new_log_probs.append(forced_logp)
            selected_logits.append(forced_logit)
        if not new_log_probs:
            empty = torch.empty((0,), device=score_logits.device, dtype=torch.float32)
            return empty, empty
        return torch.stack(new_log_probs, dim=0), torch.stack(selected_logits, dim=0)

    @staticmethod
    def _concat_counterfactual_candidate_batches(
        parts: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if len(parts) == 0:
            return {}
        keys = list(parts[0].keys())
        return {
            key: torch.cat([part[key] for part in parts], dim=0)
            for key in keys
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

        if len(replays) > 1:
            parts = [
                self.sample_counterfactual_trajectories_from_replay_batch(
                    [replay],
                    num_candidates=int(num_candidates),
                    candidate_select=str(candidate_select),
                )
                for replay in replays
            ]
            return self._concat_counterfactual_candidate_batches(parts)

        batched_dev = self._batched_replay_features(replays)
        model = self._unwrap_model(self._model)
        model.eval()
        out = self._forward_policy_on_model(model, batched_dev)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        candidate_trajs = out["candidate_trajectories"]
        candidate_identity = self._candidate_identity_from_outputs(out)
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

        if len(replays) > 1:
            parts = [
                self.replay_policy_outputs_from_replay_batch(
                    [replay],
                    eta=1.0,
                    num_candidates=int(num_candidates),
                    candidate_select=str(candidate_select),
                )
                for replay in replays
            ]
            return {
                "new_logp": torch.cat([part["new_logp"].view(-1) for part in parts], dim=0),
                "counterfactual": self._concat_counterfactual_candidate_batches(
                    [part["counterfactual"] for part in parts]
                ),
            }

        batched_dev = self._batched_replay_features(replays)
        model = self._unwrap_model(self._model)
        model.eval()
        out = self._forward_policy_on_model(model, batched_dev)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        candidate_trajs = out["candidate_trajectories"]
        candidate_identity = self._candidate_identity_from_outputs(out)
        if candidate_trajs.ndim != 4:
            raise RuntimeError(
                "SparseDriveV2 replay policy outputs require candidate trajectories with shape "
                f"(batch, modes, horizon, dims); got {tuple(candidate_trajs.shape)}"
            )

        target_global_indices = self._global_mode_indices_from_replay(replays, device=score_logits.device)
        new_logp, present = self._logp_for_global_modes_or_missing(
            score_logits=score_logits,
            candidate_global_indices=candidate_identity["candidate_global_indices"],
            target_global_indices=target_global_indices,
        )
        if not bool(present.all().item()):
            missing = (~present).nonzero(as_tuple=False).view(-1)
            if len(replays) == 1:
                new_logp = self._forced_logp_from_replay_batch(replays)
            else:
                for missing_idx_tensor in missing:
                    missing_idx = int(missing_idx_tensor.item())
                    single_outputs = self.replay_policy_outputs_from_replay_batch(
                        [replays[missing_idx]],
                        eta=1.0,
                        num_candidates=int(num_candidates),
                        candidate_select=str(candidate_select),
                    )
                    new_logp[missing_idx] = single_outputs["new_logp"].view(())
        logp_all = torch.log_softmax(score_logits, dim=1)
        if all(isinstance(replay, dict) and "grpo_candidate_mode_indices" in replay for replay in replays):
            recovered_rows = [
                self._strict_grpo_log_probs_and_logits_for_stored_candidates(
                    replay,
                    score_logits=score_logits[batch_idx : batch_idx + 1],
                    candidate_global_indices=candidate_identity["candidate_global_indices"][batch_idx : batch_idx + 1],
                )
                for batch_idx, replay in enumerate(replays)
            ]
            new_log_probs_row = torch.stack([row[0] for row in recovered_rows], dim=0)
            score_logits_row = torch.stack([row[1] for row in recovered_rows], dim=0)
            old_candidate_log_probs = self._stored_grpo_candidate_old_log_probs_from_replay(
                replays,
                device=score_logits.device,
            )
            traj_xyyaw = self._stored_grpo_candidate_traj_xyyaw_from_replay(
                replays,
                device=score_logits.device,
            )
            mode_indices = self._stored_grpo_candidate_mode_indices_from_replay(
                replays,
                device=score_logits.device,
            )
            counterfactual = {
                "traj_xyyaw": traj_xyyaw.detach(),
                "log_probs": new_log_probs_row,
                "old_log_probs": old_candidate_log_probs,
                "mode_indices": mode_indices,
                "score_logits": score_logits_row,
            }
        else:
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
        backend = str(self._nuscenes_scorer_config.get("backend", "nuscenes_pdm")).strip().lower()
        if backend in {"craft", "craft_carl", "carl", "nuscenes_craft"}:
            return "craft_carl"
        if backend in {"pdm", "nuscenes_pdm", "token", "nuscenes_token"}:
            return "nuscenes_pdm"
        return "nuscenes_pdm"

    def _ensure_nuscenes_pdm_scorer(self):
        if self._nuscenes_pdm_scorer is None:
            from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

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

    def _ensure_nuscenes_craft_scorer(self):
        if not hasattr(self, "_nuscenes_craft_scorer"):
            self._nuscenes_craft_scorer = None
        if self._nuscenes_craft_scorer is None:
            from framework.algorithms.nuscenes_craft_scorer import NuScenesCraftScorer

            scorer_kwargs = {
                key: value
                for key, value in self._nuscenes_scorer_config.items()
                if key != "backend"
            }
            self._nuscenes_craft_scorer = NuScenesCraftScorer(
                token2vad_path=nus_cfg.TOKEN2VAD_FILE,
                **scorer_kwargs,
            )
        return self._nuscenes_craft_scorer

    def _ensure_counterfactual_scorer_backend(self):
        backend_name = self._counterfactual_scorer_backend_name()
        if backend_name == "craft_carl":
            return self._ensure_nuscenes_craft_scorer()
        return self._ensure_nuscenes_pdm_scorer()

    def pdm_score_counterfactuals_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> torch.Tensor:
        from framework.algorithms.trajectory_policy_core import _as_score_tensor

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
        for key in ("global_mode_idx", "selected_path_idx", "selected_vel_idx"):
            if key not in replay:
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
