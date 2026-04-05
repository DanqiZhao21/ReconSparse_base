from __future__ import annotations

import torch

from .base import Algorithm


class PPO(Algorithm):
    """PPO configuration/spec container for the learner runtime."""

    def __init__(
        self,
        *,
        value_net: torch.nn.Module,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ppo_epochs: int = 2,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        grad_accum_steps: int = 1,
        ddp_seed: int = 0,
        eta: float = 1.0,
        use_distributed_sampler: bool = True,
        variant: str = "ppo",
        kl_coef: float = 0.0,
        dual_clip: float | None = None,
        value_clip_eps: float = 0.0,
        policy_lr: float = 1e-5,
        value_lr: float = 1e-4,
        weight_decay: float = 0.0,
        forward_kl_coef: float = 0.0,
        reverse_kl_coef: float = 0.0,
        distill_temperature: float = 1.0,
        teacher_ckpt: str | None = None,
    ):
        self.value_net = value_net
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.grad_accum_steps = int(grad_accum_steps)
        self.ddp_seed = int(ddp_seed)
        self.eta = float(eta)
        self.use_distributed_sampler = bool(use_distributed_sampler)
        self.variant = str(variant)
        self.kl_coef = float(kl_coef)
        self.dual_clip = None if dual_clip is None else float(dual_clip)
        self.value_clip_eps = float(value_clip_eps)
        self.policy_lr = float(policy_lr)
        self.value_lr = float(value_lr)
        self.weight_decay = float(weight_decay)
        self.forward_kl_coef = float(forward_kl_coef)
        self.reverse_kl_coef = float(reverse_kl_coef)
        self.distill_temperature = float(distill_temperature)
        self.teacher_ckpt = None if teacher_ckpt is None else str(teacher_ckpt)
