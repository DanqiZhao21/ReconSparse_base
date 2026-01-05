import os
import sys
from typing import Dict, Tuple, Any

import numpy as np
import torch
import cv2

# Ensure DiffusionDriveV2 is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DDV2_ROOT = os.path.join(REPO_ROOT, 'DiffusionDriveV2')
if DDV2_ROOT not in sys.path:
    sys.path.append(DDV2_ROOT)

try:
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_agent import Diffusiondrivev2_Sel_Agent
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig
except Exception as e:
    Diffusiondrivev2_Sel_Agent = None
    TransfuserConfig = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class DiffusionDriveV2Policy:
    """
    Minimal RL policy wrapper around DiffusionDriveV2 selection agent.
    - Loads provided checkpoint for future fine-tuning.
    - Exposes `act(obs)` returning (ax, ay, flag) to match `RLReconEnv.step`.
    - Currently uses a simple fallback policy for action until full feature
      mapping (camera/lidar/status → model features) is integrated.
    """

    def __init__(self, x_anchor: int = 61, y_anchor: int = 61, ckpt_path: str | None = None, device: str | None = None):
        self.x_anchor = int(x_anchor)
        self.y_anchor = int(y_anchor)
        self.ckpt_path = ckpt_path
        self.device = device

        self._agent = None
        if _IMPORT_ERROR is not None:
            print(f"[DiffusionDriveV2Policy] Import error: {_IMPORT_ERROR}. Using fallback policy.")
        else:
            try:
                cfg = TransfuserConfig()
                # lr is irrelevant for inference-only; pass a small value
                self._agent = Diffusiondrivev2_Sel_Agent(config=cfg, lr=1e-4, checkpoint_path=self.ckpt_path)
                print(f"[DiffusionDriveV2Policy] Loaded DiffusionDriveV2 SEL agent from: {self.ckpt_path}")
            except Exception as e:
                print(f"[DiffusionDriveV2Policy] Failed to init agent ({e}). Using fallback policy.")
                self._agent = None
#ADD Start
    def act(self, observation: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
        """
        使用 DiffusionDriveV2 SEL agent 推理得到未来轨迹 (8,3)，
        取第一个时间步的 (x,y)，按 61x61 离散网格（x∈[0,15.011], y∈[-0.756,0.756]）量化为 (ax, ay)。
        返回 (ax, ay, flag=0) 供 3DGS 环境使用候选锚点推进。
        """
        # 若模型未正确加载，回退为随机策略
        if self._agent is None:
            ax = np.random.randint(0, self.x_anchor)
            ay = np.random.randint(0, self.y_anchor)
            return int(ax), int(ay), 0

        # 构造 features（只使用相机，LiDAR/状态用零占位）
        try:
            camera_feature = self._build_camera_feature(observation)  # (1,3,256,1024)
            lidar_feature = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
            status_feature = torch.zeros((1, 8), dtype=torch.float32)  # 4(cmd)+2(vel)+2(acc)

            # 对齐设备
            model_device = next(self._agent.parameters()).device if hasattr(self._agent, 'parameters') else torch.device('cpu')
            camera_feature = camera_feature.to(model_device)
            lidar_feature = lidar_feature.to(model_device)
            status_feature = status_feature.to(model_device)

            features = {
                "camera_feature": camera_feature,
                "lidar_feature": lidar_feature,
                "status_feature": status_feature,
            }

            with torch.no_grad():
                pred = self._forward_sel(features, with_grad=False)
            traj = pred.get("trajectory", None)
            if traj is None:
                raise RuntimeError("No trajectory in predictions")
            traj_np = traj.squeeze(0).detach().cpu().numpy()  # (8,3)
            x, y = float(traj_np[0, 0]), float(traj_np[0, 1])

            # 将 (x,y) 量化到 61x61 网格
            ax, ay = self._quantize_to_anchor(x, y)
            return int(ax), int(ay), 0

        except Exception as e:
            # 推理失败时回退（避免中断训练循环）
            print(f"[DiffusionDriveV2Policy] Inference fallback due to: {e}")
            ax = np.random.randint(0, self.x_anchor)
            ay = np.random.randint(0, self.y_anchor)
            return int(ax), int(ay), 0

    def _build_camera_feature(self, observation: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        将三路前向相机(front_left, front, front_right)裁剪并横向拼接，后缩放至 (256,1024)，再转为 (1,3,256,1024) tensor。
        裁剪策略对齐 Transfuser：l/r 去除上下各28像素、左右各416像素；f 去除上下各28像素。
        输入为 uint8(H,W,3)，输出为 float32 [0,1]。
        """
        keys = ["front_left", "front", "front_right"]
        imgs: list[np.ndarray] = []
        for k in keys:
            if k in observation and observation[k] is not None:
                imgs.append(observation[k])
            else:
                # 若缺失某一路，用正前视图填充；若也缺失，则用已有第一张复制
                fallback = observation.get("front") or (len(imgs) and imgs[0])
                if fallback is None:
                    raise ValueError("No camera images available in observation")
                imgs.append(fallback)

        def safe_crop(img: np.ndarray, mode: str) -> np.ndarray:
            h, w = img.shape[:2]
            top, bottom = 28, 28
            left_lr, right_lr = 416, 416
            # 仅当尺寸足够时执行与 Transfuser 相同的裁剪，否则跳过以避免负切片
            if mode in ("l", "r"):
                y0, y1 = (top, h - bottom) if (h > top + bottom) else (0, h)
                x0, x1 = (left_lr, w - right_lr) if (w > left_lr + right_lr) else (0, w)
                return img[y0:y1, x0:x1]
            else:  # front
                y0, y1 = (top, h - bottom) if (h > top + bottom) else (0, h)
                return img[y0:y1]

        l0 = safe_crop(imgs[0], "l")
        f0 = safe_crop(imgs[1], "f")
        r0 = safe_crop(imgs[2], "r")

        # 为了能横向拼接，按最小高度做等比缩放到一致高度
        target_h = min(l0.shape[0], f0.shape[0], r0.shape[0])
        def resize_to_h(img: np.ndarray, th: int) -> np.ndarray:
            if img.shape[0] == th:
                return img
            scale = th / max(1, img.shape[0])
            new_w = max(1, int(round(img.shape[1] * scale)))
            return cv2.resize(img, (new_w, th), interpolation=cv2.INTER_LINEAR)

        l0 = resize_to_h(l0, target_h)
        f0 = resize_to_h(f0, target_h)
        r0 = resize_to_h(r0, target_h)

        stitched = np.concatenate([l0, f0, r0], axis=1)  # (H, W_total, 3)
        stitched = cv2.resize(stitched, (1024, 256), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(stitched.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _quantize_to_anchor(self, x: float, y: float) -> Tuple[int, int]:
        """
        将物理坐标 (x,y)（米）映射到 61x61 的离散索引。
        x ∈ [0, 15.011], y ∈ [-0.756, 0.756]。
        """
        x_min, x_max = 0.0, 15.011
        y_min, y_max = -0.756, 0.756
        # 归一化到 [0,1]
        x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.0
        y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        x_norm = float(np.clip(x_norm, 0.0, 1.0))
        y_norm = float(np.clip(y_norm, 0.0, 1.0))

        ax = int(round(x_norm * (self.x_anchor - 1)))
        ay = int(round(y_norm * (self.y_anchor - 1)))
        ax = int(np.clip(ax, 0, self.x_anchor - 1))
        ay = int(np.clip(ay, 0, self.y_anchor - 1))
        return ax, ay

    def quantize_xy_to_action(self, x: float, y: float) -> torch.Tensor:
        """
        公共量化接口：将 (x,y) 转为环境动作为 [ax, ay, flag]。
        flag 固定为 0（与 3DGS 环境候选锚点推进一致）。
        """
        ax, ay = self._quantize_to_anchor(x, y)
        return torch.tensor([int(ax), int(ay), 0])
    
    def forward_test(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        统一推理接口（测试/交互）：
        - 使用 learned scorer 选优轨迹，不计算 PDM（cal_pdm=False），不需要 metric_cache。
        - 在 no_grad 下运行，返回 {"trajectory": (1,8,3)}。
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        camera_feature = self._build_camera_feature(observation)  # (1,3,256,1024)
        lidar_feature = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        status_feature = torch.zeros((1, 8), dtype=torch.float32)

        model_device = next(self._agent.parameters()).device if hasattr(self._agent, 'parameters') else torch.device('cpu')
        features = {
            "camera_feature": camera_feature.to(model_device),
            "lidar_feature": lidar_feature.to(model_device),
            "status_feature": status_feature.to(model_device),
        }
        with torch.no_grad():
            pred = self._forward_sel(features, with_grad=False)
        traj = pred.get("trajectory", None)
        if traj is None:
            raise RuntimeError("No trajectory in predictions")
        return {"trajectory": traj}

    def forward_train(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        统一推理接口（训练）：
        - 使用 learned scorer 选优轨迹，不计算 PDM（cal_pdm=False），不需要 metric_cache。
        - 保留梯度，返回 {"trajectory": (1,8,3)} 以便参与 RL/WM 损失反传。
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        camera_feature = self._build_camera_feature(observation)  # (1,3,256,1024)
        lidar_feature = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        status_feature = torch.zeros((1, 8), dtype=torch.float32)

        model_device = next(self._agent.parameters()).device if hasattr(self._agent, 'parameters') else torch.device('cpu')
        features = {
            "camera_feature": camera_feature.to(model_device),
            "lidar_feature": lidar_feature.to(model_device),
            "status_feature": status_feature.to(model_device),
        }
        pred = self._forward_sel(features, with_grad=True)
        traj = pred.get("trajectory", None)
        if traj is None:
            raise RuntimeError("No trajectory in predictions")
        return {"trajectory": traj}
    
    def _forward_sel(self, features: Dict[str, torch.Tensor], with_grad: bool = False):
        """
        统一封装 SEL 模型前向：不计算 PDM（cal_pdm=False），走 learned scorer 路径。
        - with_grad=False：用于 act()，开启 no_grad 推理。
        - with_grad=True：用于训练阶段的前向，保留梯度（不进入 no_grad），但仍使用 forward_test_rl 分支。
        """
        if hasattr(self._agent, "_transfuser_model"):
            # 通过将模型置为 eval，触发 forward_test_rl（不依赖 metric_cache）；eval 不影响 autograd。
            self._agent._transfuser_model.eval()
            if with_grad:
                return self._agent._transfuser_model(
                    features,
                    targets=None,
                    eta=0.0,
                    metric_cache=None,
                    cal_pdm=False,
                )
            else:
                with torch.no_grad():
                    return self._agent._transfuser_model(
                        features,
                        targets=None,
                        eta=0.0,
                        metric_cache=None,
                        cal_pdm=False,
                    )
        # Fallback：调用代理封装
        if with_grad:
            return self._agent.forward(features)
        else:
            with torch.no_grad():
                return self._agent.forward(features)

    def plan_best_trajectory(self, observation: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        训练阶段使用：保留梯度的前向，返回最佳轨迹 tensor (1,8,3)。
        - 不依赖 PDM cache，使用已学习的 scorer 选择轨迹。
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        camera_feature = self._build_camera_feature(observation)  # (1,3,256,1024)
        lidar_feature = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        status_feature = torch.zeros((1, 8), dtype=torch.float32)

        model_device = next(self._agent.parameters()).device if hasattr(self._agent, 'parameters') else torch.device('cpu')
        features = {
            "camera_feature": camera_feature.to(model_device),
            "lidar_feature": lidar_feature.to(model_device),
            "status_feature": status_feature.to(model_device),
        }
        pred = self._forward_sel(features, with_grad=True)
        traj = pred.get("trajectory", None)
        if traj is None:
            raise RuntimeError("No trajectory in predictions")
        return traj  # (1,8,3)
#ADD END
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        #TODO: Placeholder: return empty metrics.
        return {"loss_pi": 0.0, "loss_v": 0.0}
