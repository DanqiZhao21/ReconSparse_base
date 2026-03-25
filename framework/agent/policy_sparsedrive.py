import os
import sys
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from torch.nn.parallel import DistributedDataParallel as DDP

from .base import Agent
from framework.utils.repo_paths import resolve_ego_ads_subdir


# Ensure SparseDrive is importable
_SPARSEDRIVE_ROOT = resolve_ego_ads_subdir("SparseDrive")
if _SPARSEDRIVE_ROOT not in sys.path:
    sys.path.append(_SPARSEDRIVE_ROOT)


def _as_float_tensor(x: Any, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _resize_rgb_uint8(img: np.ndarray, *, out_hw: Tuple[int, int]) -> np.ndarray:
    """Resize HWC RGB uint8 to (out_h,out_w).把相机图 resize 到 256*704，保持和 SparseDrive 训练时输入一致。"""
    import cv2

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if img is None:
        raise ValueError("img is None")
    if img.shape[0] == out_h and img.shape[1] == out_w:
        return img
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


class SparseDrivePolicy(Agent):
    """RL policy wrapper around SparseDrive planning head.

    This wrapper exposes the same replay/logp semantics used by PPO in this repo.

    Action execution:
    - continuous first-point only: execute (dx, dy, yaw, flag=2)

    Log-prob definition:
    - A categorical policy over planning modes (ego_fut_mode) conditioned on the
      current driving command (cmd_idx).
    - logp = log_softmax(logits)[mode_idx]

    Notes:
    - SparseDrive internally uses multi-view geometry; we construct projection
      matrices from ego_pose + cam2ego + intrinsics provided by the env.
    - PPO updates call `logp_from_replay()` which recomputes logits under current
      parameters with autograd enabled.
    """

    def __init__(
        self,
        *,
        config_path: str,
        ckpt_path: str,
        device: str | None = None,
        rl_lr: float = 1e-5,
        execute_mode: str = "first_step",
    ) -> None:
        self.config_path = str(config_path)
        self.ckpt_path = str(ckpt_path)
        self._device_override = device

        # Only one runtime mode is supported: continuous first-point execution.
        # Keep `execute_mode` argument only for backward-compatible call sites.
        self._execute_mode = "continuous"

        # Lazy imports (mmcv/mmdet are heavy)
        try:
            from mmcv import Config
            # from mmcv.utils import Config
            # from mmcv.utils import ConfigDict
            from mmcv import DictAction
            import importlib
            from mmdet.models import build_detector
            from mmcv.runner import load_checkpoint
        except Exception as e:
            raise ImportError(
                f"[SparseDrivePolicy] Missing SparseDrive deps (mmcv/mmdet). Import error: {e}"
            )

        # cfg = ConfigDict.fromfile(self.config_path)
        cfg = Config.fromfile(self.config_path)
        if getattr(cfg, "plugin", False):
            plugin_dir = getattr(cfg, "plugin_dir", None)
            if plugin_dir:
                _module_dir = os.path.dirname(str(plugin_dir))
            else:
                _module_dir = os.path.dirname(self.config_path)
            # Support both relative and absolute plugin_dir.
            # Example absolute path:
            #   /root/clone/ReconDreamer-RL/SparseDrive/projects/mmdet3d_plugin/
            # should map to import path:
            #   projects.mmdet3d_plugin
            _module_dir_abs = os.path.abspath(_module_dir)
            _sparsedrive_abs = os.path.abspath(_SPARSEDRIVE_ROOT)
            if _module_dir_abs.startswith(_sparsedrive_abs + os.sep):
                rel = os.path.relpath(_module_dir_abs, _sparsedrive_abs)
                _module_path = rel.replace(os.sep, ".")
            else:
                _module_path = _module_dir_abs.strip(os.sep).replace(os.sep, ".")

            importlib.import_module(_module_path)

        # Build model   禁用 backbone 的预训练权重
        cfg.model.pretrained = None
        self._model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        _ = load_checkpoint(self._model, self.ckpt_path, map_location="cpu", strict=False)

        # Find final input size from config (defaults to (256, 704) H,W)
        final_dim = None
        try:
            #读取输入分辨率
            data_aug = getattr(cfg, "data_aug_conf", None)
            if isinstance(data_aug, dict) and data_aug.get("final_dim") is not None:
                fd = data_aug.get("final_dim")
                if isinstance(fd, (list, tuple)) and len(fd) == 2:
                    final_dim = (int(fd[0]), int(fd[1]))
        except Exception:
            final_dim = None
        self._input_hw = final_dim if final_dim is not None else (256, 704)

        # Norm config
        self._img_mean = np.asarray(getattr(cfg, "img_norm_cfg", {}).get("mean", [123.675, 116.28, 103.53]), dtype=np.float32)
        self._img_std = np.asarray(getattr(cfg, "img_norm_cfg", {}).get("std", [58.395, 57.12, 57.375]), dtype=np.float32)

        # Planning modes
        ego_fut_mode = None
        ego_fut_ts = None
        try:
            ego_fut_mode = int(cfg.get("ego_fut_mode"))
            ego_fut_ts = int(cfg.get("ego_fut_ts"))
        except Exception:
            pass
        #TODO: 这里是否考虑了command=3呢
        self._ego_fut_mode = int(ego_fut_mode) if ego_fut_mode is not None else 6
        self._ego_fut_ts = int(ego_fut_ts) if ego_fut_ts is not None else 6

        # Optimizer (optional; PPO update can use external optimizer too, but keep parity with ddv2)
        self._optimizer: torch.optim.Optimizer | None = None
        params = [p for p in self._model.parameters() if getattr(p, "requires_grad", False)]
        if len(params) > 0:
            self._optimizer = torch.optim.Adam(params, lr=float(rl_lr))

        self._ddp_enabled = False
        self.to(self.device)
        print(f"[SparseDrivePolicy] Loaded SparseDrive ckpt: {self.ckpt_path}")

    # -------------------- device / ddp / ckpt --------------------
    @property
    def device(self) -> torch.device:
        if getattr(self, "_device_override", None):
            try:
                return torch.device(str(self._device_override))
            except Exception:
                pass
        try:
            return next(self._model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def to(self, device: str | torch.device) -> "SparseDrivePolicy":
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

    def wrap_ddp(
        self,
        *,
        device_id: int,
        process_group: Any | None = None,
        find_unused_parameters: bool = True,
        rl_lr: float | None = None,
    ) -> None:
        m = self._model
        if isinstance(m, DDP):
            self._ddp_enabled = True
            return

        target_device = torch.device(f"cuda:{int(device_id)}") if torch.cuda.is_available() else torch.device("cpu")
        self.to(target_device)

        self._model = DDP(
            m,
            device_ids=[int(device_id)],
            output_device=int(device_id),
            process_group=process_group,
            find_unused_parameters=bool(find_unused_parameters),
        )
        self._ddp_enabled = True

        if self._optimizer is not None:
            lr = float(rl_lr) if rl_lr is not None else float(self._optimizer.param_groups[0].get("lr", 1e-5))
        else:
            lr = float(rl_lr) if rl_lr is not None else 1e-5

        core = self._model.module
        params = [p for p in core.parameters() if getattr(p, "requires_grad", False)]
        if len(params) == 0:
            raise RuntimeError("SparseDrive DDP wrap found no trainable parameters")
        self._optimizer = torch.optim.Adam(params, lr=lr)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        m = self._model.module if isinstance(self._model, DDP) else self._model
        return m.state_dict()

    @property
    def trainable_module(self):
        return self._model

    def save_checkpoint(self, path: str) -> None:
        sd = {f"agent.{k}": v.detach().cpu() for k, v in self.state_dict().items()}
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(path)
        tmp_path = os.path.join(out_dir, f".{base}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            torch.save({"state_dict": sd}, tmp_path)
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def load_from_checkpoint(self, path: str, *, strict: bool = False) -> None:
        ckpt = torch.load(path, map_location=self.device)
        sd = ckpt.get("state_dict", ckpt)
        if not isinstance(sd, dict):
            raise ValueError("Checkpoint does not contain a state_dict")
        sd2: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            kk = str(k)
            if kk.startswith("agent."):
                kk = kk[len("agent.") :]
            if torch.is_tensor(v):
                sd2[kk] = v
        m = self._model.module if isinstance(self._model, DDP) else self._model
        m.load_state_dict(sd2, strict=bool(strict))

    # -------------------- core: build model inputs --------------------
    def _build_projection_mats(
        self,
        *,
        ego_pose: np.ndarray,
        cam2ego: np.ndarray,
        cam_intrinsics: np.ndarray,
        cam_hw: np.ndarray | None,
        img_hw: Tuple[int, int],
        net_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (projection_mat[6,4,4], image_wh[6,2]) in float32 for net resolution."""
        h_cur, w_cur = int(img_hw[0]), int(img_hw[1])
        h_net, w_net = int(net_hw[0]), int(net_hw[1])

        proj = np.zeros((6, 4, 4), dtype=np.float32)
        image_wh = np.zeros((6, 2), dtype=np.float32)
        image_wh[:, 0] = float(w_net)
        image_wh[:, 1] = float(h_net)

        for i in range(6):
            K0 = np.asarray(cam_intrinsics[i], dtype=np.float32)
            if K0.shape == (4, 4):
                K0 = K0[:3, :3]
            if cam_hw is not None and cam_hw.shape[0] == 6:
                h0 = float(cam_hw[i, 0])
                w0 = float(cam_hw[i, 1])
            else:
                h0 = float(h_cur)
                w0 = float(w_cur)

            # Scale intrinsics to current rendered image size
            sw1 = float(w_cur) / max(1.0, float(w0))
            sh1 = float(h_cur) / max(1.0, float(h0))
            K1 = K0.copy()
            K1[0, 0] *= sw1
            K1[0, 2] *= sw1
            K1[1, 1] *= sh1
            K1[1, 2] *= sh1

            # Scale to network input size
            sw2 = float(w_net) / max(1.0, float(w_cur))
            sh2 = float(h_net) / max(1.0, float(h_cur))
            K = K1.copy()
            K[0, 0] *= sw2
            K[0, 2] *= sw2
            K[1, 1] *= sh2
            K[1, 2] *= sh2

            c2e = np.asarray(cam2ego[i], dtype=np.float32)
            e2c = np.linalg.inv(c2e)

            # lidar == ego; lidar2cam == ego2cam
            P3 = K @ e2c[:3, :]
            P = np.eye(4, dtype=np.float32)
            P[:3, :4] = P3
            proj[i] = P

        return proj, image_wh#shape = (6, 4, 4) projection 6个相机，每个相机4*4 ； image wh:shape = (6, 2) 六个相机，每个相机的图像尺寸wh

    def _normalize_rgb(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32)
        x = (x - self._img_mean[None, None, :]) / self._img_std[None, None, :]
        return x

    def _build_batch(self, observation: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cam_keys = ["front", "front_left", "front_right", "back_left", "back_right", "back"]
        imgs = []
        for k in cam_keys:
            img = observation.get(k, None)
            if img is None:
                raise KeyError(f"Missing camera key in observation: {k}")
            imgs.append(img)

        # Resize and normalize
        h_net, w_net = int(self._input_hw[0]), int(self._input_hw[1])
        imgs_rs = [_resize_rgb_uint8(im, out_hw=(h_net, w_net)) for im in imgs]
        imgs_norm = [self._normalize_rgb(im) for im in imgs_rs]
        # (6,3,H,W)
        img_t = torch.from_numpy(np.stack([im.transpose(2, 0, 1) for im in imgs_norm], axis=0)).to(dtype=torch.float32)
        img_t = img_t.unsqueeze(0)  # (1,6,3,H,W)

        ego_pose = np.asarray(observation.get("ego_pose"), dtype=np.float32)
        cam2ego = np.asarray(observation.get("cam2ego"), dtype=np.float32)
        cam_intr = np.asarray(observation.get("cam_intrinsics"), dtype=np.float32)
        cam_hw = observation.get("cam_hw", None)
        if cam_hw is not None:
            cam_hw = np.asarray(cam_hw, dtype=np.float32)

        img0 = imgs[0]
        h_cur, w_cur = int(img0.shape[0]), int(img0.shape[1])
        proj, image_wh = self._build_projection_mats(
            ego_pose=ego_pose,
            cam2ego=cam2ego,
            cam_intrinsics=cam_intr,
            cam_hw=cam_hw,
            img_hw=(h_cur, w_cur),
            net_hw=(h_net, w_net),
        )

        #TODO:这里需要后续自己从数据集抓取真值补充
        # Command: SparseDrive planning decoder uses 3 commands.  优先用 observation["driving_command"]
        cmd4 = np.asarray(observation.get("driving_command", np.zeros((4,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if cmd4.shape[0] >= 3:
            cmd3 = cmd4[:3].copy()
        else:
            print("💣[attn in SparseDrPolicy] Warning: obs missing driving_command or has insufficient length, attempting to fill with zeros")
            cmd3 = np.zeros((3,), dtype=np.float32)
            cmd3[: cmd4.shape[0]] = cmd4
        if not np.isfinite(cmd3).all():
            print("💣[attn in SparseDrPolicy] Warning: obs has invalid driving_command, attempting to fill with zeros")
            cmd3 = np.zeros((3,), dtype=np.float32)
        if float(np.abs(cmd3).sum()) <= 1e-6:
            cmd3[0] = 1.0

        ego_status = np.asarray(observation.get("ego_status", np.zeros((8,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if ego_status.shape[0] < 8:
            tmp = np.zeros((8,), dtype=np.float32)
            tmp[: ego_status.shape[0]] = ego_status
            ego_status = tmp

        ts = observation.get("timestamp", 0.0)
        try:
            ts = float(ts)
        except Exception:
            ts = 0.0

        T_global_inv = np.linalg.inv(ego_pose)
        data = {
            "projection_mat": torch.from_numpy(proj).unsqueeze(0),  # (1,6,4,4)
            "image_wh": torch.from_numpy(image_wh).unsqueeze(0),  # (1,6,2)
            "T_global": torch.from_numpy(ego_pose).unsqueeze(0),
            "T_global_inv": torch.from_numpy(T_global_inv).unsqueeze(0),
            "timestamp": torch.tensor([ts], dtype=torch.float32),
            "ego_status": torch.from_numpy(ego_status[None, :]),
            "gt_ego_fut_cmd": torch.from_numpy(cmd3[None, :]),
            # SparseDrive heads access per-sample meta via metas["img_metas"].
            "img_metas": [
                {
                    "T_global": ego_pose.astype(np.float32, copy=True),
                    "T_global_inv": T_global_inv.astype(np.float32, copy=True),
                    "timestamp": np.float32(ts),
                }
            ],
        }

        return img_t, data

    def _to_model_metas(self, data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in data.items():
            if k == "img_metas":
                out[k] = v
            else:
                out[k] = _as_float_tensor(v, device=device)
        return out

    # -------------------- action conversion --------------------
    #输入：（T,2)
    def _build_env_action(self, *, traj_xy: np.ndarray) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        first = traj_xy[0].astype(np.float32)#只取第一步去执行
        dx = float(first[0])
        dy = float(first[1])
        yaw = float(np.arctan2(first[1], first[0])) if float(np.linalg.norm(first)) > 1e-6 else 0.0
        debug: Dict[str, Any] = {
            "first_step_xy": torch.tensor([dx, dy], dtype=torch.float32),
            "first_step_yaw": torch.tensor([yaw], dtype=torch.float32),
            "execute_mode": self._execute_mode,
        }

        return (dx, dy, yaw, 2), debug

    def _forward_raw_planning(
        self,
        *,
        img: torch.Tensor,
        data_dev: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return raw planning logits/trajs before post-process.

        Returns:
        - logits: (3, M) raw classification logits for 3 commands and M modes.
        - trajs:  (3, M, T, 2) cumulative trajectory in ego frame.
        """
        model = self._model.module if isinstance(self._model, DDP) else self._model
        feature_maps = model.extract_feat(img)
        _det_out, _map_out, _motion_out, planning_out = model.head(feature_maps, data_dev)
        if planning_out is None:
            raise RuntimeError("SparseDrive planning output is None")

        cls_raw = planning_out["classification"][-1]
        reg_raw = planning_out["prediction"][-1]
        if cls_raw.ndim == 2:
            cls_raw = cls_raw.unsqueeze(0)
        if reg_raw.ndim == 4:
            reg_raw = reg_raw.unsqueeze(0)

        bs = int(cls_raw.shape[0])
        cls = cls_raw.reshape(bs, 3, self._ego_fut_mode)
        traj = reg_raw.reshape(bs, 3, self._ego_fut_mode, self._ego_fut_ts, 2).cumsum(dim=-2)
        return cls[0], traj[0]

    # -------------------- sampling / replay / logp --------------------
    def sample_sparsedrive_with_replay(
        self,
        observation: Dict[str, Any],
        *,
        mode_idx: int = -1,
        mode_select: str = "sample",
        anchor_bank: Dict[str, Any] | None = None,
    ) -> Tuple[Tuple[Any, ...], torch.Tensor, Dict[str, Any]]:
        img_t, data = self._build_batch(observation)

        model = self._model.module if isinstance(self._model, DDP) else self._model
        model_device = self.device

        img_t = img_t.to(model_device)
        data_dev = self._to_model_metas(data, model_device)

        model.eval()
        with torch.inference_mode():
            logits_all, traj_all = self._forward_raw_planning(img=img_t, data_dev=data_dev)
        logits_all = logits_all.detach().cpu()
        traj_all = traj_all.detach().cpu()

        # select cmd index
        cmd = data["gt_ego_fut_cmd"].detach().cpu().reshape(-1)
        if cmd.numel() >= 3:
            cmd_idx = int(torch.argmax(cmd[:3]).item())
        else:
            cmd_idx = 0

        logits = logits_all[cmd_idx].reshape(-1)  # (ego_fut_mode,)
        logp_all = torch.log_softmax(logits, dim=0)

        if int(mode_idx) < 0:
            sel = str(mode_select).strip().lower()
            if sel in {"greedy", "max", "argmax"}:
                mi = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits, dim=0)
                if torch.isfinite(probs).all() and float(probs.sum().item()) > 0:
                    mi = int(torch.distributions.Categorical(probs).sample().item())
                else:
                    mi = int(torch.argmax(logits).item())
        else:
            mi = max(0, min(int(mode_idx), int(logits.shape[0]) - 1))

        lp = logp_all[mi]

        # Select trajectory with exactly the same (cmd, mode) used for logp.
        traj_xy = traj_all[cmd_idx, mi].reshape(-1, 2).numpy().astype(np.float32)

        # Convert XY trajectory to XYYaw so replay keeps the executed pose representation.
        traj_xyyaw = np.zeros((traj_xy.shape[0], 3), dtype=np.float32)
        traj_xyyaw[:, :2] = traj_xy
        prev_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), traj_xy[:-1]], axis=0)
        #从sparsedrive拿到xy后自己计算yaw值
        dxy = traj_xy - prev_xy
        yaw = np.arctan2(dxy[:, 1], dxy[:, 0]).astype(np.float32)
        traj_xyyaw[:, 2] = yaw
        #trajxy的shape是（T,2）， traj_xyyaw的shape是（T,3）
        #将prediction的xy轨迹转化成环境可执行的action格式（dx, dy, yaw, flag=2），这里只执行第一步的动作
        #在这个build_env_action里面又重新计算了一下yaw值
        action, exec_debug = self._build_env_action(traj_xy=traj_xy)

        replay = {
            "img": img_t.detach().cpu().clone(),
            "projection_mat": data["projection_mat"].detach().cpu().clone(),
            "image_wh": data["image_wh"].detach().cpu().clone(),
            "T_global": data["T_global"].detach().cpu().clone(),
            "T_global_inv": data["T_global_inv"].detach().cpu().clone(),
            "timestamp": data["timestamp"].detach().cpu().clone(),
            "ego_status": data["ego_status"].detach().cpu().clone(),
            "gt_ego_fut_cmd": data["gt_ego_fut_cmd"].detach().cpu().clone(),
            "img_metas": data["img_metas"],
            "cmd_idx": int(cmd_idx),
            "mode_idx": int(mi),
            "final_planning": torch.from_numpy(traj_xy).detach().cpu().clone(),
            "traj_xyyaw": torch.from_numpy(traj_xyyaw).detach().cpu().clone(),
            "execute_mode": exec_debug.get("execute_mode", self._execute_mode),
        }

        return action, lp, replay

    def sample_sparsedrive_with_replay_batch(
        self,
        observations: List[Dict[str, Any]],
        *,
        mode_idx: int = -1,
        mode_select: str = "sample",
        anchor_bank: Dict[str, Any] | None = None,
    ) -> Tuple[List[Tuple[Any, ...]], List[torch.Tensor], List[Dict[str, Any]]]:
        if len(observations) == 0:
            return [], [], []

        actions: List[Tuple[Any, ...]] = []
        logps: List[torch.Tensor] = []
        replays: List[Dict[str, Any]] = []
        for obs in observations:
            a, lp, rep = self.sample_sparsedrive_with_replay(
                obs,
                mode_idx=mode_idx,
                mode_select=mode_select,
                anchor_bank=anchor_bank,
            )
            actions.append(a)
            logps.append(lp)
            replays.append(rep)
        return actions, logps, replays

    def logp_from_replay(self, replay: Dict[str, Any], *, eta: float = 1.0) -> torch.Tensor:
        # eta unused for SparseDrive; keep signature parity.
        img = replay["img"]
        data = {
            "projection_mat": replay["projection_mat"],
            "image_wh": replay["image_wh"],
            "T_global": replay["T_global"],
            "T_global_inv": replay["T_global_inv"],
            "timestamp": replay["timestamp"],
            "ego_status": replay["ego_status"],
            "gt_ego_fut_cmd": replay["gt_ego_fut_cmd"],
            "img_metas": replay.get("img_metas"),
        }
        if data["img_metas"] is None:
            t_global = np.asarray(data["T_global"].detach().cpu().numpy()[0], dtype=np.float32)
            t_global_inv = np.asarray(data["T_global_inv"].detach().cpu().numpy()[0], dtype=np.float32)
            ts = float(data["timestamp"].detach().cpu().numpy().reshape(-1)[0])
            data["img_metas"] = [
                {
                    "T_global": t_global,
                    "T_global_inv": t_global_inv,
                    "timestamp": np.float32(ts),
                }
            ]
        mode_idx = int(replay.get("mode_idx", 0))
        cmd_idx = int(replay.get("cmd_idx", 0))

        model = self._model.module if isinstance(self._model, DDP) else self._model
        model_device = self.device

        img = img.to(model_device)
        data_dev = self._to_model_metas(data, model_device)

        model.eval()
        logits_all, _traj_all = self._forward_raw_planning(img=img, data_dev=data_dev)
        logits = logits_all[cmd_idx].reshape(-1)
        lp = torch.log_softmax(logits, dim=0)[mode_idx]
        return lp

    # -------------------- Agent interface adapters --------------------
    def initialize(self) -> None:
        return

    def act(
        self,
        observation: Dict[str, Any],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ):
        return self.sample_sparsedrive_with_replay(
            observation,
            mode_idx=int(mode_idx),
            mode_select=str(mode_select),
        )

    def act_batch(
        self,
        observations: List[Dict[str, Any]],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ):
        return self.sample_sparsedrive_with_replay_batch(
            observations,
            mode_idx=int(mode_idx),
            mode_select=str(mode_select),
        )

    def load_checkpoint(self, path: str, *, strict: bool = False) -> None:
        self.load_from_checkpoint(path, strict=bool(strict))

    def parameters(self):
        m = self._model.module if isinstance(self._model, DDP) else self._model
        return m.parameters()
