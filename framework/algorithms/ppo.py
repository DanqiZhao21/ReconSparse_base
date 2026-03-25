from __future__ import annotations

from typing import Any, Dict

import torch

from framework.lightning import TrajectoryLightningModule, TrajectoryUpdateDataModule
from framework.lightning_compat import L

from .base import Algorithm

class PPO(Algorithm):
    def __init__(self, *, optimizer: torch.optim.Optimizer,
                 value_net: torch.nn.Module,
                 clip_eps: float = 0.2, vf_coef: float = 0.5,
                 ppo_epochs: int = 2, minibatch_size: int = 64,
                 max_grad_norm: float = 0.5, grad_accum_steps: int = 1,
                 ddp_enabled: bool = False, world_size: int = 1, rank: int = 0,
                 ddp_seed: int = 0, update_seed: int = 0,
                 eta: float = 1.0,
                 use_distributed_sampler: bool = True,
                 variant: str = "ppo",
                 kl_coef: float = 0.0,
                 dual_clip: float | None = None,
                 value_clip_eps: float = 0.0):
        self.optimizer = optimizer
        self.value_net = value_net
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
        self.eta = float(eta)
        self.use_distributed_sampler = bool(use_distributed_sampler)
        self.variant = str(variant)
        self.kl_coef = float(kl_coef)
        self.dual_clip = None if dual_clip is None else float(dual_clip)
        self.value_clip_eps = float(value_clip_eps)

    def get_value_components(self) -> tuple[Any | None, Any | None]:
        return self.value_net, self.optimizer

    def update(self, *, agent: Any, batch: Dict[str, Any], device: torch.device) -> Dict[str, float]:
        module = TrajectoryLightningModule(
            agent=agent,
            optimizer=self.optimizer,
            algo_kind=self.variant,
            eta=self.eta,
            clip_eps=self.clip_eps,
            vf_coef=self.vf_coef,
            value_clip_eps=self.value_clip_eps,
            kl_coef=self.kl_coef,
            dual_clip=self.dual_clip,
            value_net=self.value_net,
        )
        data = TrajectoryUpdateDataModule(
            batch=batch,
            minibatch_size=self.minibatch_size,
            ddp_enabled=self.ddp_enabled,
            world_size=self.world_size,
            rank=self.rank,
            seed=self.ddp_seed,
            update_seed=self.update_seed,
            include_obs=True,
            use_distributed_sampler=self.use_distributed_sampler,
        )
        trainer = L.Trainer(
            accelerator=("gpu" if device.type == "cuda" else "cpu"),
            devices=1,
            max_epochs=self.ppo_epochs,
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
