import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import cv2
from torch.nn.parallel import DistributedDataParallel as DDP

from .base import Agent

# Ensure DiffusionDriveV2 is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DDV2_ROOT = os.path.join(REPO_ROOT, 'DiffusionDriveV2')
if DDV2_ROOT not in sys.path:
    sys.path.append(DDV2_ROOT)

try:
    # RL variant only (we need diffusion log-probs for policy-gradient)
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_agent import Diffusiondrivev2_Rl_Agent
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_config import TransfuserConfig
except Exception as e_rl:
    Diffusiondrivev2_Rl_Agent = None
    TransfuserConfig = None
    _IMPORT_ERROR = e_rl
else:
    _IMPORT_ERROR = None

#NOTE RL policy 封装器;调用 DiffusionDriveV2-RL 模型 (Diffusiondrivev2_Rl_Agent) 来生成轨迹动作。
class DiffusionDriveV2Policy(Agent):
    """
        Minimal RL policy wrapper around DiffusionDriveV2-RL.
        This wrapper is intended for policy-gradient optimization via diffusion log-probabilities
        (REINFORCE style), producing continuous actions: (x, y, yaw, flag=2).
    """

    def __init__(
        self,
        x_anchor: int = 61,
        y_anchor: int = 61,
        ckpt_path: str | None = None,
        device: str | None = None,
        *,
        rl_lr: float = 1e-5,
        reinforce_baseline_beta: float = 0.98,
    ):
        self.x_anchor = int(x_anchor)
        self.y_anchor = int(y_anchor)
        self.ckpt_path = ckpt_path
        self._device_override = device

        self._agent = None
        self._ddv2_optimizer: torch.optim.Optimizer | None = None
        self._baseline_beta = float(reinforce_baseline_beta)
        self._reward_baseline: float = 0.0
        if _IMPORT_ERROR is not None or Diffusiondrivev2_Rl_Agent is None or TransfuserConfig is None:
            raise ImportError(
                f"[DiffusionDriveV2Policy] DiffusionDriveV2-RL import failed: {_IMPORT_ERROR}. "
                "This project is configured to use diffusiondrivev2-rl only (no SEL fallback)."
            )

        cfg = TransfuserConfig()
        self._agent = Diffusiondrivev2_Rl_Agent(config=cfg, lr=rl_lr, checkpoint_path=self.ckpt_path)
        print(f"[DiffusionDriveV2Policy] Loaded DiffusionDriveV2 RL agent from: {self.ckpt_path}")

        # Ensure the underlying model lives on the requested device.
        # Some upstream agent constructors keep modules on CPU by default.
        try:
            self.to(self.device)
        except Exception:
            pass

        # If we are using DDV2-RL, set up an optimizer for trainable params (mostly _trajectory_head).
        if self._agent is not None and hasattr(self._agent, "parameters"):
            params = [p for p in self._agent.parameters() if getattr(p, "requires_grad", False)]
            if len(params) > 0:
                self._ddv2_optimizer = torch.optim.Adam(params, lr=float(rl_lr))

        self._ddp_enabled: bool = False

    # -------------------- Agent interface -------------------- #
    def initialize(self) -> None:
        return
#NOTE
########################################### 
# 对外动作接口
########################################### 
# 实际调用 sample_ddv2rl_with_replay(...)  sample_ddv2rl_with_replay_batch(...)
# 返回值不是只给动作，还给：logp（或 logp 列表）replay dict（PPO 训练需要）
    def act(
        self,
        observation: Dict[str, Any],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ) -> Tuple[Tuple[float, float, float, int], torch.Tensor, Dict[str, Any]]:
        return self.sample_ddv2rl_with_replay(
            observation,
            eta=float(eta),
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
    ) -> Tuple[List[Tuple[float, float, float, int]], List[torch.Tensor], List[Dict[str, Any]]]:
        return self.sample_ddv2rl_with_replay_batch(
            observations,
            eta=float(eta),
            mode_idx=int(mode_idx),
            mode_select=str(mode_select),
        )

    def load_checkpoint(self, path: str, *, strict: bool = False) -> None:
        self.load_from_checkpoint(path, strict=bool(strict))

    def parameters(self):
        if self._agent is None or not hasattr(self._agent, "parameters"):
            return []
        return self._agent.parameters()
#NOTE 
########################################### 
# 多卡训练用 DDP 包装 Distributed Data Parallel
########################################### 
# 作用：把 self._agent._transfuser_model 用 torch.nn.parallel.DistributedDataParallel 包起来。
    def wrap_ddp(
        self,
        *,
        device_id: int,
        process_group: Any | None = None,
        find_unused_parameters: bool = True,
        rl_lr: float | None = None,
    ) -> None:
        """Wrap the internal DDV2 transfuser model with DDP and rebuild optimizer.

        This is required for multi-GPU fine-tuning (ddv2_rl_reinforce / ddv2_rl_ppo).
        Safe to call multiple times.
        """
        if self._agent is None or not hasattr(self._agent, "_transfuser_model"):
            raise RuntimeError("DDV2 agent is not initialized")
        m = self._agent._transfuser_model
        if isinstance(m, DDP):
            self._ddp_enabled = True
            return

        # DDP requires parameters to already be on the target device.
        target_device = torch.device(f"cuda:{int(device_id)}") if torch.cuda.is_available() else torch.device("cpu")
        try:
            self.to(target_device)
        except Exception:
            try:
                m.to(target_device)
            except Exception:
                pass

        self._agent._transfuser_model = DDP(
            m,
            device_ids=[int(device_id)],
            output_device=int(device_id),
            process_group=process_group,
            find_unused_parameters=bool(find_unused_parameters),
        )
        self._ddp_enabled = True

        # Rebuild optimizer to make sure it references the right params
        lr = float(rl_lr) if rl_lr is not None else float(self._ddv2_optimizer.param_groups[0].get("lr", 1e-5) if self._ddv2_optimizer else 1e-5)
        core = self._agent._transfuser_model.module
        params = [p for p in core.parameters() if getattr(p, "requires_grad", False)]
        if len(params) == 0:
            raise RuntimeError("DDV2 DDP wrap found no trainable parameters")
        self._ddv2_optimizer = torch.optim.Adam(params, lr=lr)

    def to(self, device: str | torch.device) -> "DiffusionDriveV2Policy":
        """Move the internal transfuser model to device (best-effort)."""
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        self._device_override = str(dev)
        if self._agent is None:
            return self

        # Move known internal model.
        try:
            if hasattr(self._agent, "_transfuser_model"):
                m = self._agent._transfuser_model
                if isinstance(m, DDP):
                    m.module.to(dev)
                else:
                    m.to(dev)
        except Exception:
            pass

        # Some agents implement .to() / .cuda().
        try:
            if hasattr(self._agent, "to"):
                self._agent.to(dev)
        except Exception:
            pass

        return self
#NOTE
########################################### 
#统一保存/加载格式
########################################### 
    # -------------------- Checkpoint IO (actor-learner) -------------------- #
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the underlying transfuser model state_dict."""
        if self._agent is None or not hasattr(self._agent, "_transfuser_model"):
            raise RuntimeError("DDV2 agent is not initialized")
        m = self._agent._transfuser_model
        core = m.module if isinstance(m, DDP) else m
        return core.state_dict()

    def save_checkpoint(self, path: str) -> None:
        """Save weights in the same format used by train_closed_loop.py."""
        sd = self.state_dict()
        sd_pref = {f"agent.{k}": v.detach().cpu() for k, v in sd.items()}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"state_dict": sd_pref}, path)

    def load_from_checkpoint(self, path: str, *, strict: bool = False) -> None:
        """Load a checkpoint produced by save_checkpoint/train_closed_loop.

        Accepts either:
        - {"state_dict": {"agent.xxx": tensor, ...}}
        - raw state_dict
        """
        if self._agent is None or not hasattr(self._agent, "_transfuser_model"):
            raise RuntimeError("DDV2 agent is not initialized")

        map_location = self.device
        ckpt = torch.load(path, map_location=map_location)
        sd = ckpt.get("state_dict", ckpt)
        if not isinstance(sd, dict):
            raise ValueError("Checkpoint does not contain a state_dict")

        # Strip optional prefix.
        sd2: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            kk = str(k)
            if kk.startswith("agent."):
                kk = kk[len("agent.") :]
            # Some checkpoints save the *agent* state_dict and include an extra
            # "_transfuser_model." namespace.
            if kk.startswith("_transfuser_model."):
                kk = kk[len("_transfuser_model.") :]
            if torch.is_tensor(v):
                sd2[kk] = v

        m = self._agent._transfuser_model
        core = m.module if isinstance(m, DDP) else m
        core.load_state_dict(sd2, strict=bool(strict))
    @property
    def device(self) -> torch.device:
        if getattr(self, "_device_override", None):
            try:
                return torch.device(str(self._device_override))
            except Exception:
                pass
        # Fallback: try model device
        if self._agent is not None and hasattr(self._agent, "parameters"):
            try:
                return next(self._agent.parameters()).device
            except Exception:
                pass
        return torch.device("cpu")

#NOTE 采样功能函数
########################################### 
# REINFORCE 用的单步采样（带梯度）
########################################### 

    # -------------------- DDV2-RL policy-gradient (REINFORCE) -------------------- #
    def step_ddv2rl(self, observation: Dict[str, np.ndarray], *, eta: float = 1.0):
        """Sample a continuous (x,y,yaw) action from DiffusionDriveV2-RL and return (action, logp).

        Action format: (x, y, yaw, flag=2) -> executed directly by env.
        logp is the summed diffusion log-prob over denoising steps for the chosen trajectory.
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized")
        if not hasattr(self._agent, "_transfuser_model"):
            raise RuntimeError("Unexpected DDV2 agent type")

        camera_feature = self._build_camera_feature(observation)  # (1,3,256,1024)
        lidar_feature = torch.zeros((1, 1, 256, 256), dtype=torch.float32)#占位符
        status_feature = torch.zeros((1, 8), dtype=torch.float32)#占位符

        model_device = next(self._agent.parameters()).device if hasattr(self._agent, 'parameters') else torch.device('cpu')
        features = {
            "camera_feature": camera_feature.to(model_device),
            "lidar_feature": lidar_feature.to(model_device),
            "status_feature": status_feature.to(model_device),
        }

        # Important: do NOT wrap in no_grad; we want gradients w.r.t. DDV2 params.
        # Use model eval branch which now returns trajectory/log_probs when targets/metric_cache are None.
        # We need the inference branch (no targets/metric_cache) but still want gradients.
        # .eval() toggles dropout/bn behavior; it does NOT disable autograd.
        self._agent._transfuser_model.eval()
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
        if traj is None or log_probs is None:
            raise RuntimeError("DDV2-RL model did not return trajectory/log_probs")
        print("💜traj shape:", traj.shape, "💜log_probs shape:", log_probs.shape)
        # traj: typically (B, K=20*4=80, 8, 3) and log_probs: (B, K=20*4=80, step_num)
        traj0 = traj[0]
        # 按每个模式的总 logp 进行软最大采样选择模式索引
        if log_probs.dim() == 3:
            mode_logps = log_probs[0].sum(dim=-1)  # (N,)
        elif log_probs.dim() == 2:
            mode_logps = log_probs.sum(dim=-1)     # (N,)
        else:
            mode_logps = log_probs.reshape(-1, log_probs.shape[-1]).sum(dim=-1)

        # 温度采样（默认温度 1.0）；若数值异常则退化为贪心
        temperature = 1.0
        probs = torch.softmax(mode_logps / max(1e-6, temperature), dim=0)
        if torch.isfinite(probs).all() and float(probs.sum().item()) > 0:
            mode_idx = int(torch.distributions.Categorical(probs).sample().item())
        else:
            mode_idx = int(torch.argmax(mode_logps).item())

        x = traj0[mode_idx, 0, 0].item()
        y = traj0[mode_idx, 0, 1].item()
        yaw = traj0[mode_idx, 0, 2].item()

        lp = mode_logps[mode_idx]

        action = (float(x), float(y), float(yaw), 2)
        return action, lp

########################################### 
# PPO 采样（无梯度 + 保存 replay
########################################### 
    def sample_ddv2rl_with_replay(
        self,
        observation: Dict[str, np.ndarray],
        *,
        eta: float = 1.0,
        mode_idx: int = 0,
        mode_select: str = "sample",
    ) -> Tuple[Tuple[float, float, float, int], torch.Tensor, Dict[str, Any]]:
        """Like step_ddv2rl(), but also returns replay info for PPO.

        The replay dict contains:
        - camera_feature: (1,3,256,1024) float tensor (CPU)
        - diffusion_chain: `all_diffusion_output` returned by DDV2 (CPU)
        - mode_idx: int
        - mode_select: str (default "sample")
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        camera_feature = self._build_camera_feature(observation)
        lidar_feature = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        status_feature = torch.zeros((1, 8), dtype=torch.float32)

        model_device = next(self._agent.parameters()).device if hasattr(self._agent, 'parameters') else torch.device('cpu')
        features = {
            "camera_feature": camera_feature.to(model_device),
            "lidar_feature": lidar_feature.to(model_device),
            "status_feature": status_feature.to(model_device),
        }

        self._agent._transfuser_model.eval()
        # PPO collection does not need autograd; avoid building graphs for speed/memory.
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

        traj0 = traj[0]#（B=1，batch size）是为了和ddv2对齐
        # 自动选择：若传入 mode_idx<0，则按 mode_select 执行：
        # - sample: softmax(mode_logps) 采样（带探索，默认）
        # - greedy: argmax(mode_logps)（更稳定，适合 debug）
        # 否则（mode_idx>=0）裁剪到合法范围
        if log_probs.dim() == 3:
            mode_logps = log_probs[0].sum(dim=-1)  # (N,)
        elif log_probs.dim() == 2:
            mode_logps = log_probs.sum(dim=-1)     # (N,)
        else:
            mode_logps = log_probs.reshape(-1, log_probs.shape[-1]).sum(dim=-1)

        if int(mode_idx) < 0:
            sel = str(mode_select).strip().lower()
            if sel in {"greedy", "max", "argmax"}:
                mi = int(torch.argmax(mode_logps).item())
            else:
                temperature = 1.0
                probs = torch.softmax(mode_logps / max(1e-6, temperature), dim=0)
                if torch.isfinite(probs).all() and float(probs.sum().item()) > 0:
                    mi = int(torch.distributions.Categorical(probs).sample().item())
                else:
                    mi = int(torch.argmax(mode_logps).item())
        else:
            mi = int(mode_idx)
            mi = max(0, min(mi, int(traj0.shape[0]) - 1))

        x = traj0[mi, 0, 0].item()
        y = traj0[mi, 0, 1].item()
        yaw = traj0[mi, 0, 2].item()

        lp = mode_logps[mi]

        action = (float(x), float(y), float(yaw), 2)
        replay = {
            # IMPORTANT: clone() to avoid retaining a larger underlying storage
            # (e.g. from batched inference views). Otherwise torch.save may
            # serialize the full storage and bloat shard files massively.
            "camera_feature": camera_feature.detach().cpu().clone(),
            "diffusion_chain": diffusion_chain.detach().cpu().clone(),
            "mode_idx": mi,
            # Selected trajectory points (for commit/closed-loop execution on actor side).
            # Shape: (H, 3) where H is DDV2 trajectory horizon (typically 8).
            "traj_xyyaw": traj0[mi].detach().cpu().clone(),
        }
        return action, lp, replay

    def sample_ddv2rl_with_replay_batch(
        self,
        observations: List[Dict[str, np.ndarray]],
        *,
        eta: float = 1.0,
        mode_idx: int = 0,
        mode_select: str = "sample",
    ) -> Tuple[List[Tuple[float, float, float, int]], List[torch.Tensor], List[Dict[str, Any]]]:
        """Batched variant of sample_ddv2rl_with_replay() for vectorized env rollout.

        Returns:
        - actions: list[(x,y,yaw,flag=2)]
        - logps: list[Tensor scalar] (summed over diffusion steps for chosen mode)
        - replays: list[dict] per-sample, compatible with PPO update code
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized")
        if len(observations) == 0:
            return [], [], []

        # Build camera features (CPU) then batch on device.
        cams = [self._build_camera_feature(obs) for obs in observations]  # each: (1,3,256,1024)
        camera_feature = torch.cat(cams, dim=0)
        bsz = int(camera_feature.shape[0])

        lidar_feature = torch.zeros((bsz, 1, 256, 256), dtype=torch.float32)
        status_feature = torch.zeros((bsz, 8), dtype=torch.float32)

        model_device = next(self._agent.parameters()).device if hasattr(self._agent, 'parameters') else torch.device('cpu')
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

        # mode_logps: (B, N)
        if log_probs.dim() == 3:
            mode_logps = log_probs.sum(dim=-1)
        elif log_probs.dim() == 2:
            mode_logps = log_probs
        else:
            mode_logps = log_probs.reshape(log_probs.shape[0], -1, log_probs.shape[-1]).sum(dim=-1)

        # Select per-sample mode index
        sel = str(mode_select).strip().lower()
        mode_idx_list: List[int] = []
        if int(mode_idx) < 0:
            if sel in {"greedy", "max", "argmax"}:
                mode_idx_list = [int(torch.argmax(mode_logps[b]).item()) for b in range(bsz)]
            else:
                temperature = 1.0
                for b in range(bsz):
                    probs = torch.softmax(mode_logps[b] / max(1e-6, temperature), dim=0)
                    if torch.isfinite(probs).all() and float(probs.sum().item()) > 0:
                        mode_idx_list.append(int(torch.distributions.Categorical(probs).sample().item()))
                    else:
                        mode_idx_list.append(int(torch.argmax(mode_logps[b]).item()))
        else:
            # Clamp fixed mode_idx
            n_modes = int(traj.shape[1])
            mi = max(0, min(int(mode_idx), n_modes - 1))
            mode_idx_list = [int(mi) for _ in range(bsz)]

        actions: List[Tuple[float, float, float, int]] = []
        logps: List[torch.Tensor] = []
        replays: List[Dict[str, Any]] = []

        for b in range(bsz):
            mi = int(mode_idx_list[b])
            x = traj[b, mi, 0, 0].item()
            y = traj[b, mi, 0, 1].item()
            yaw = traj[b, mi, 0, 2].item()
            lp = mode_logps[b, mi]
            actions.append((float(x), float(y), float(yaw), 2))
            logps.append(lp)
            replays.append(
                {
                    # IMPORTANT: slices like x[b:b+1] are views that can retain
                    # the full batch storage. clone() makes each item own its
                    # minimal storage to keep shard files small.
                    "camera_feature": camera_feature[b : b + 1].detach().cpu().clone(),
                    "diffusion_chain": diffusion_chain[b : b + 1].detach().cpu().clone(),
                    "mode_idx": mi,
                    # Shape: (H, 3)
                    "traj_xyyaw": traj[b, mi].detach().cpu().clone(),
                }
            )

        return actions, logps, replays

    def logp_from_replay(self, replay: Dict[str, Any], *, eta: float = 1.0) -> torch.Tensor:
        """Recompute logp for the stored diffusion chain under current params."""
        if self._agent is None:
            raise RuntimeError("Agent not initialized")
        if not hasattr(self._agent, "_transfuser_model"):
            raise RuntimeError("Unexpected DDV2 agent type")
        if not hasattr(self._agent._transfuser_model, "compute_log_probs_from_diffusion_chain"):
            raise RuntimeError("DDV2 model does not expose compute_log_probs_from_diffusion_chain")

        camera_feature = replay["camera_feature"]
        diffusion_chain = replay["diffusion_chain"]
        mode_idx = int(replay.get("mode_idx", 0))

        lidar_feature = torch.zeros((camera_feature.shape[0], 1, 256, 256), dtype=torch.float32)
        status_feature = torch.zeros((camera_feature.shape[0], 8), dtype=torch.float32)

        model_device = next(self._agent.parameters()).device if hasattr(self._agent, 'parameters') else torch.device('cpu')
        features = {
            "camera_feature": camera_feature.to(model_device),
            "lidar_feature": lidar_feature.to(model_device),
            "status_feature": status_feature.to(model_device),
        }

        chain = diffusion_chain.to(model_device)
        self._agent._transfuser_model.eval()
        all_log_probs = self._agent._transfuser_model.compute_log_probs_from_diffusion_chain(
            features,
            chain,
            eta=float(eta),
        )
        # all_log_probs: (B, N, step_num)
        lp = all_log_probs[0, mode_idx].sum()
        return lp

    def reinforce_update(self, logp: torch.Tensor, reward: float) -> Dict[str, float]:
        """One-step REINFORCE update for DDV2-RL."""
        if self._ddv2_optimizer is None:
            raise RuntimeError("DDV2 optimizer not initialized (no trainable params?)")
        r = float(reward)
        # EMA baseline to reduce variance
        self._reward_baseline = self._baseline_beta * self._reward_baseline + (1.0 - self._baseline_beta) * r
        adv = r - self._reward_baseline
        #NOTE reinforce损失计算  L=−A_t​ * log π_θ​(a_t​∣s_t​)
        loss = -(float(adv) * logp)
        self._ddv2_optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # grad_norm_sq = 0.0 #无需计算梯度范数
        # for p in self._ddv2_optimizer.param_groups[0]["params"]:
        #     if p.grad is None:
        #         continue
        #     grad_norm_sq += float(p.grad.detach().pow(2).sum().cpu().item())
        # grad_norm = float(grad_norm_sq ** 0.5)

        self._ddv2_optimizer.step()

        return {
            "loss_reinforce": float(loss.detach().cpu().item()),
            "baseline": float(self._reward_baseline),
            "adv": float(adv),
            # "grad_norm": grad_norm,
        }
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
