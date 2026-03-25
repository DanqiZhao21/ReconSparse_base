from __future__ import annotations

from typing import Any, Dict

import torch

from framework.lightning import TrajectoryLightningModule, TrajectoryUpdateDataModule
from framework.lightning_compat import L

from .base import Algorithm

class ReinforcePP(Algorithm):
    def __init__(self, *, optimizer: torch.optim.Optimizer,
                 clip_eps: float = 0.2, kl_coef: float = 0.0, epochs: int = 1, minibatch_size: int = 64,
                 max_grad_norm: float = 0.5, grad_accum_steps: int = 1,
                 ddp_enabled: bool = False, world_size: int = 1, rank: int = 0,
                 ddp_seed: int = 0, update_seed: int = 0,
                 eta: float = 1.0,
                 use_distributed_sampler: bool = True,
                 ref_agent: Any | None = None,
                 variant: str = "reinforcepp"):
        self.optimizer = optimizer
        self.clip_eps = float(clip_eps)
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
        self.eta = float(eta)
        self.use_distributed_sampler = bool(use_distributed_sampler)
        self.ref_agent = ref_agent
        self.variant = str(variant)

    def update(self, *, agent: Any, batch: Dict[str, Any], device: torch.device) -> Dict[str, float]:
        module = TrajectoryLightningModule(
            agent=agent,
            optimizer=self.optimizer,
            algo_kind=self.variant,
            eta=self.eta,
            clip_eps=self.clip_eps,
            kl_coef=self.kl_coef,
        )
        data = TrajectoryUpdateDataModule(
            batch=batch,
            minibatch_size=self.minibatch_size,
            ddp_enabled=self.ddp_enabled,
            world_size=self.world_size,
            rank=self.rank,
            seed=self.ddp_seed,
            update_seed=self.update_seed,
            include_obs=False,
            use_distributed_sampler=self.use_distributed_sampler,
        )
        trainer = L.Trainer(
            accelerator=("gpu" if device.type == "cuda" else "cpu"),
            devices=1,
            max_epochs=self.epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accumulate_grad_batches=self.grad_accum_steps,
            gradient_clip_val=self.max_grad_norm,
            num_sanity_val_steps=0,
            use_distributed_sampler=False,
        )
        trainer.fit(module, datamodule=data)
        return dict(module.latest_metrics)
