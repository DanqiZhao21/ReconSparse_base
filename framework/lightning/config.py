from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class LearnerOptimizerConfig:
    policy_lr: float
    value_lr: float | None = None
    weight_decay: float = 0.0


@dataclass(frozen=True)
class ActorLearnerLightningConfig:
    algo_kind: str
    optimizer_config: LearnerOptimizerConfig
    eta: float
    clip_eps: float
    vf_coef: float = 0.0
    value_clip_eps: float = 0.0
    kl_coef: float = 0.0
    forward_kl_coef: float = 0.0
    reverse_kl_coef: float = 0.0
    distill_temperature: float = 1.0
    teacher_ckpt: str | None = None
    dual_clip: float | None = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ddp_seed: int = 0
    minibatch_size: int = 64
    include_obs: bool = False
    use_distributed_sampler: bool = True
    mode: str = "async"
    num_actors: int = 1
    shards_per_update: int = 1
    poll_s: float = 0.2
    max_shard_version_gap: int = 2
    norm_eps: float = 1e-8
    inner_epochs: int = 1
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 0.0
    max_updates: int = 0


def optimizer_config_from_algorithm(algo: Any, train_cfg: Dict[str, Any]) -> LearnerOptimizerConfig:
    return LearnerOptimizerConfig(
        policy_lr=float(getattr(algo, "policy_lr", train_cfg.get("policy_lr", train_cfg.get("ddv2_lr", 1e-5)))),
        value_lr=(float(getattr(algo, "value_lr")) if getattr(algo, "value_lr", None) is not None else None),
        weight_decay=float(getattr(algo, "weight_decay", train_cfg.get("weight_decay", 0.0))),
    )


def actor_learner_lightning_config_from_algorithm(
    algo: Any,
    *,
    train_cfg: Dict[str, Any],
    actor_learner_cfg: Dict[str, Any],
    algo_meta: Dict[str, Any],
) -> ActorLearnerLightningConfig:
    algo_kind = str(algo_meta.get("algo_key", getattr(algo, "variant", "ppo")))
    raw_max_updates = actor_learner_cfg.get("max_updates", train_cfg.get("updates", 50))
    inner_epochs = int(
        getattr(
            algo,
            "ppo_epochs",
            getattr(algo, "epochs", 1),
        )
        or 1
    )
    return ActorLearnerLightningConfig(
        algo_kind=algo_kind,
        optimizer_config=optimizer_config_from_algorithm(algo, train_cfg),
        eta=float(getattr(algo, "eta", algo_meta.get("eta", 1.0))),
        clip_eps=float(getattr(algo, "clip_eps", algo_meta.get("clip_eps", 0.2))),
        vf_coef=float(getattr(algo, "vf_coef", 0.0)),
        value_clip_eps=float(getattr(algo, "value_clip_eps", algo_meta.get("value_clip_eps", 0.0))),
        kl_coef=float(getattr(algo, "kl_coef", 0.0)),
        forward_kl_coef=float(getattr(algo, "forward_kl_coef", 0.0)),
        reverse_kl_coef=float(getattr(algo, "reverse_kl_coef", 0.0)),
        distill_temperature=float(getattr(algo, "distill_temperature", 1.0)),
        teacher_ckpt=getattr(algo, "teacher_ckpt", None),
        dual_clip=getattr(algo, "dual_clip", None),
        gamma=float(train_cfg.get("gamma", 0.99)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        ddp_seed=int(getattr(algo, "ddp_seed", ((train_cfg.get("ddp", {}) or {}).get("seed", 0)))),
        minibatch_size=int(getattr(algo, "minibatch_size", train_cfg.get("minibatch_size", 64))),
        include_obs=bool(algo_kind.startswith("ppo")),
        use_distributed_sampler=bool(getattr(algo, "use_distributed_sampler", True)),
        mode=str(actor_learner_cfg.get("mode", "async")).strip().lower(),
        num_actors=int(actor_learner_cfg.get("num_actors", 1)),
        shards_per_update=int(actor_learner_cfg.get("shards_per_update", actor_learner_cfg.get("num_actors", 1))),
        poll_s=float(actor_learner_cfg.get("poll_interval_s", 0.2)),
        max_shard_version_gap=int(actor_learner_cfg.get("max_shard_version_gap", 2)),
        norm_eps=float(algo_meta.get("rpp_norm_eps", 1e-8)),
        inner_epochs=max(1, int(inner_epochs)),
        accumulate_grad_batches=int(
            getattr(algo, "grad_accum_steps", ((train_cfg.get("ddp", {}) or {}).get("grad_accum_steps", 1)))
        ),
        gradient_clip_val=float(getattr(algo, "max_grad_norm", train_cfg.get("max_grad_norm", 0.0))),
        max_updates=int(raw_max_updates or 0),
    )


def trainer_kwargs_from_learner_config(
    learner_config: ActorLearnerLightningConfig,
    *,
    accelerator: str,
) -> Dict[str, Any]:
    return {
        "accelerator": str(accelerator),
        "devices": 1,
        "max_epochs": (
            int(learner_config.max_updates) * max(1, int(learner_config.inner_epochs))
            if int(learner_config.max_updates) > 0
            else -1
        ),
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "accumulate_grad_batches": int(learner_config.accumulate_grad_batches),
        "gradient_clip_val": float(learner_config.gradient_clip_val),
        "num_sanity_val_steps": 0,
        "use_distributed_sampler": False,
        "reload_dataloaders_every_n_epochs": 1,
    }


__all__ = [
    "ActorLearnerLightningConfig",
    "LearnerOptimizerConfig",
    "actor_learner_lightning_config_from_algorithm",
    "optimizer_config_from_algorithm",
    "trainer_kwargs_from_learner_config",
]
