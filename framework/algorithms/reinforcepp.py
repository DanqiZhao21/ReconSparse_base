from __future__ import annotations

from typing import Any, Dict, List
import torch

from framework.algorithms.reinforcepp_core import ddv2_reinforcepp_update

from .base import Algorithm

class ReinforcePP(Algorithm):
    def __init__(self, *, kl_coef: float = 0.0, epochs: int = 1, minibatch_size: int = 64,
                 max_grad_norm: float = 0.5, grad_accum_steps: int = 1,
                 ddp_enabled: bool = False, world_size: int = 1, rank: int = 0,
                 ddp_seed: int = 0, update_seed: int = 0,
                 ddv2_eta: float = 1.0, ddv2_mode_idx_default: int = -1,
                 replay_compute_camera_dtype: torch.dtype = torch.float32,
                 replay_compute_chain_dtype: torch.dtype = torch.float32,
                 use_distributed_sampler: bool = True,
                 ref_agent: Any | None = None):
        self.kl_coef = float(kl_coef)
        self.epochs = int(epochs)
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
        self.ref_agent = ref_agent

    def update(self, *, agent: Any, batch: Dict[str, Any], device: torch.device) -> Dict[str, float]:
        adv = batch["adv"]
        replay = batch["replay"]
        res = ddv2_reinforcepp_update(
            agent=agent,
            ref_agent=self.ref_agent,
            adv=adv,
            replay=replay,
            device=device,
            ddv2_eta=self.ddv2_eta,
            ddv2_mode_idx_default=self.ddv2_mode_idx_default,
            kl_coef=self.kl_coef,
            epochs=self.epochs,
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
        return {
            "loss_pi": float(res.loss_pi),
            "approx_kl": float(res.approx_kl),
            "adv_mean": float(res.adv_mean),
        }
