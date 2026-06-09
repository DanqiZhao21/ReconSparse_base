from __future__ import annotations

from .base import Algorithm


class SAC(Algorithm):
    """SAC-style policy-gradient configuration for closed-loop replay training.

    The current actor-learner shard contract stores sampled replay, rewards, and
    log-probs, not a standard off-policy SAC replay buffer with differentiable
    actions. This spec therefore exposes an entropy-regularized actor objective
    that can run on the existing closed-loop reward path.
    """

    def __init__(
        self,
        *,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.0,
        epochs: int = 1,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        grad_accum_steps: int = 1,
        ddp_seed: int = 0,
        eta: float = 1.0,
        use_distributed_sampler: bool = True,
        variant: str = "sac",
        policy_lr: float = 1e-6,
        weight_decay: float = 0.0,
        forward_kl_coef: float = 0.0,
        reverse_kl_coef: float = 0.0,
        distill_temperature: float = 1.0,
        teacher_ckpt: str | None = None,
    ) -> None:
        self.entropy_coef = float(entropy_coef)
        self.kl_coef = float(kl_coef)
        self.epochs = int(epochs)
        self.minibatch_size = int(minibatch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.grad_accum_steps = int(grad_accum_steps)
        self.ddp_seed = int(ddp_seed)
        self.eta = float(eta)
        self.use_distributed_sampler = bool(use_distributed_sampler)
        self.variant = str(variant)
        self.policy_lr = float(policy_lr)
        self.weight_decay = float(weight_decay)
        self.forward_kl_coef = float(forward_kl_coef)
        self.reverse_kl_coef = float(reverse_kl_coef)
        self.distill_temperature = float(distill_temperature)
        self.teacher_ckpt = None if teacher_ckpt is None else str(teacher_ckpt)
