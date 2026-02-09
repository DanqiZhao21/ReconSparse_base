from __future__ import annotations

from typing import Any, Dict, List
import torch

from framework.algorithms.ppo_ddv2_core import ddv2_ppo_update

from .base import Algorithm

class PPO(Algorithm):
    def __init__(self, *, clip_eps: float = 0.2, vf_coef: float = 0.5,
                 ppo_epochs: int = 2, minibatch_size: int = 64,
                 max_grad_norm: float = 0.5, grad_accum_steps: int = 1,
                 ddp_enabled: bool = False, world_size: int = 1, rank: int = 0,
                 ddp_seed: int = 0, update_seed: int = 0,
                 ddv2_eta: float = 1.0, ddv2_mode_idx_default: int = -1,
                 replay_compute_camera_dtype: torch.dtype = torch.float32,
                 replay_compute_chain_dtype: torch.dtype = torch.float32,
                 use_distributed_sampler: bool = True):
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.grad_accum_steps = int(grad_accum_steps)
        self.ddp_enabled = bool(ddp_enabled)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.ddp_seed = int(ddp_seed)
        self.update_seed = int(update_seed)
        self.ddv2_eta = float(ddv2_eta)
        self.ddv2_mode_idx_default = int(ddv2_mode_idx_default)
        self.replay_compute_camera_dtype = replay_compute_camera_dtype
        self.replay_compute_chain_dtype = replay_compute_chain_dtype
        self.use_distributed_sampler = bool(use_distributed_sampler)

    def update(self, *, agent: Any, batch: Dict[str, Any], device: torch.device) -> Dict[str, float]:
        res = ddv2_ppo_update(
            agent=agent,
            value_net=batch["value_net"],
            value_optim=batch["value_optim"],
            obs_batch=batch["obs_batch"],
            old_logp=batch["old_logp"],
            adv=batch["adv"],
            ret=batch["ret"],
            replay=batch["replay"],
            device=device,
            ddv2_eta=self.ddv2_eta,
            ddv2_mode_idx_default=self.ddv2_mode_idx_default,
            clip_eps=self.clip_eps,
            vf_coef=self.vf_coef,
            ppo_epochs=self.ppo_epochs,
            minibatch_size=self.minibatch_size,
            max_grad_norm=self.max_grad_norm,
            grad_accum_steps=self.grad_accum_steps,
            ddp_enabled=self.ddp_enabled,
            world_size=self.world_size,
            rank=self.rank,
            ddp_seed=self.ddp_seed,
            update_seed=self.update_seed,
            replay_compute_camera_dtype=self.replay_compute_camera_dtype,
            replay_compute_chain_dtype=self.replay_compute_chain_dtype,
            use_distributed_sampler=self.use_distributed_sampler,
        )
        # Return a standard metric dict
        out = {
            "loss_pi": float(res.loss_pi),
            "loss_v": float(res.loss_v),
            "approx_kl": float(res.approx_kl),
            "ratio_mean": float(getattr(res, "ratio_mean", 0.0)),
            "adv_mean": float(getattr(res, "adv_mean", 0.0)),
        }
        return out
