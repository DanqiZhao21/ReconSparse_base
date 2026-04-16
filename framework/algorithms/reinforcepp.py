from __future__ import annotations

from .base import Algorithm


class ReinforcePP(Algorithm):
    """ReinforcePP configuration/spec container for the learner runtime."""

    def __init__(
        self,
        *,
        clip_eps: float = 0.2,
        kl_coef: float = 0.0,
        epochs: int = 1,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        grad_accum_steps: int = 1,
        ddp_seed: int = 0,
        eta: float = 1.0,
        use_distributed_sampler: bool = True,
        variant: str = "reinforcepp",
        policy_lr: float = 1e-5,
        weight_decay: float = 0.0,
        forward_kl_coef: float = 0.0,
        reverse_kl_coef: float = 0.0,
        distill_temperature: float = 1.0,
        teacher_ckpt: str | None = None,
        grpo_coef: float = 0.0,
        grpo_num_candidates: int = 0,
        grpo_candidate_select: str = "topk",
        grpo_norm_eps: float = 1e-6,
        grpo_use_rank_adv: bool = False,
        grpo_score_clip: float | None = None,
        grpo_debug_visualize: bool = False,
        grpo_debug_dir: str | None = None,
        grpo_debug_max_batches: int = 0,
        grpo_debug_top_k: int = 4,
    ):
        self.clip_eps = float(clip_eps)
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
        self.grpo_coef = float(grpo_coef)
        self.grpo_num_candidates = int(grpo_num_candidates)
        self.grpo_candidate_select = str(grpo_candidate_select)
        self.grpo_norm_eps = float(grpo_norm_eps)
        self.grpo_use_rank_adv = bool(grpo_use_rank_adv)
        self.grpo_score_clip = None if grpo_score_clip is None else float(grpo_score_clip)
        self.grpo_debug_visualize = bool(grpo_debug_visualize)
        self.grpo_debug_dir = None if grpo_debug_dir is None else str(grpo_debug_dir)
        self.grpo_debug_max_batches = int(grpo_debug_max_batches)
        self.grpo_debug_top_k = int(grpo_debug_top_k)
