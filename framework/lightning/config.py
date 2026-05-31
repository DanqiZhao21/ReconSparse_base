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
    closed_loop_loss_coef: float = 1.0
    forward_kl_coef: float = 0.0
    reverse_kl_coef: float = 0.0
    distill_temperature: float = 1.0
    teacher_ckpt: str | None = None
    grpo_enabled: bool = False
    grpo_config_path: str | None = None
    grpo_coef: float = 0.0
    grpo_num_candidates: int = 0
    grpo_candidate_select: str = "topk"
    grpo_norm_eps: float = 1e-6
    grpo_use_rank_adv: bool = False
    grpo_score_clip: float | None = None
    grpo_objective: str = "logprob"
    grpo_temperature: float = 1.0
    grpo_debug_visualize: bool = False
    grpo_debug_dir: str | None = None
    grpo_debug_max_batches: int = 0
    grpo_debug_top_k: int = 4
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
    max_inflight_per_actor: int = 1
    poll_s: float = 0.2
    shard_collect_timeout_s: float = 0.0
    allow_partial_updates_after_timeout: bool = False
    actor_heartbeat_timeout_s: float = 0.0
    max_shard_version_lag: int = 2
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


def resolve_grpo_config(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    shared_cfg = train_cfg.get("grpo", {}) or {}
    if not isinstance(shared_cfg, dict):
        shared_cfg = {}

    config_path = shared_cfg.get("config_path", None)
    if config_path is not None and str(config_path).strip() != "":
        raise NotImplementedError(
            "train.grpo.config_path is reserved for future external-yaml merging and is not supported yet"
        )

    resolved = {
        "enabled": bool(shared_cfg.get("enable", False)),
        "config_path": None if config_path is None else str(config_path),
        "coef": float(shared_cfg.get("coef", 0.0)),
        "num_candidates": int(shared_cfg.get("num_candidates", 0)),
        "candidate_select": str(shared_cfg.get("candidate_select", "topk")),
        "norm_eps": float(shared_cfg.get("norm_eps", 1e-6)),
        "use_rank_adv": bool(shared_cfg.get("use_rank_adv", False)),
        "score_clip": shared_cfg.get("score_clip", None),
        "objective": str(shared_cfg.get("objective", "logprob")),
        "temperature": float(shared_cfg.get("temperature", 1.0)),
        "debug_visualize": bool(shared_cfg.get("debug_visualize", False)),
        "debug_dir": shared_cfg.get("debug_dir", None),
        "debug_max_batches": int(shared_cfg.get("debug_max_batches", 0)),
        "debug_top_k": int(shared_cfg.get("debug_top_k", 4)),
    }
    if not bool(resolved["enabled"]):
        resolved["coef"] = 0.0
    return resolved


def actor_learner_lightning_config_from_algorithm(
    algo: Any,
    *,
    train_cfg: Dict[str, Any],
    actor_learner_cfg: Dict[str, Any],
    algo_meta: Dict[str, Any],
) -> ActorLearnerLightningConfig:
    algo_kind = str(algo_meta.get("algo_key", getattr(algo, "variant", "ppo")))
    grpo_cfg = resolve_grpo_config(train_cfg)
    raw_max_shard_version_lag = actor_learner_cfg.get("max_shard_version_lag", 2)
    raw_max_updates = actor_learner_cfg.get("max_updates", train_cfg.get("updates", 50))
    shard_collect_timeout_s = float(actor_learner_cfg.get("shard_collect_timeout_s", 0.0) or 0.0)
    raw_actor_heartbeat_timeout_s = actor_learner_cfg.get("actor_heartbeat_timeout_s", None)
    actor_heartbeat_timeout_s = (
        float(raw_actor_heartbeat_timeout_s)
        if raw_actor_heartbeat_timeout_s is not None
        else (float(shard_collect_timeout_s) * 5.0 if float(shard_collect_timeout_s) > 0.0 else 0.0)
    )
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
        closed_loop_loss_coef=float(
            (train_cfg.get("reinforcepp", {}) or {}).get(
                "policy_grad_weight",
                1.0,
            )
        ),
        forward_kl_coef=float(getattr(algo, "forward_kl_coef", 0.0)),
        reverse_kl_coef=float(getattr(algo, "reverse_kl_coef", 0.0)),
        distill_temperature=float(getattr(algo, "distill_temperature", 1.0)),
        teacher_ckpt=getattr(algo, "teacher_ckpt", None),
        grpo_enabled=bool(grpo_cfg["enabled"]),
        grpo_config_path=grpo_cfg["config_path"],
        grpo_coef=float(grpo_cfg["coef"]),
        grpo_num_candidates=int(grpo_cfg["num_candidates"]),
        grpo_candidate_select=str(grpo_cfg["candidate_select"]),
        grpo_norm_eps=float(grpo_cfg["norm_eps"]),
        grpo_use_rank_adv=bool(grpo_cfg["use_rank_adv"]),
        grpo_score_clip=grpo_cfg["score_clip"],
        grpo_objective=str(grpo_cfg["objective"]),
        grpo_temperature=float(grpo_cfg["temperature"]),
        grpo_debug_visualize=bool(grpo_cfg["debug_visualize"]),
        grpo_debug_dir=grpo_cfg["debug_dir"],
        grpo_debug_max_batches=int(grpo_cfg["debug_max_batches"]),
        grpo_debug_top_k=int(grpo_cfg["debug_top_k"]),
        dual_clip=getattr(algo, "dual_clip", None),
        gamma=float(train_cfg.get("gamma", 0.99)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        ddp_seed=int(getattr(algo, "ddp_seed", ((train_cfg.get("ddp", {}) or {}).get("seed", 0)))),
        minibatch_size=int(getattr(algo, "minibatch_size", train_cfg.get("minibatch_size", 64))),
        include_obs=bool(algo_kind.startswith("ppo") and not bool(algo_meta.get("critic_use_agent_features", False))),
        use_distributed_sampler=bool(getattr(algo, "use_distributed_sampler", True)),
        mode=str(actor_learner_cfg.get("mode", "async")).strip().lower(),
        num_actors=int(actor_learner_cfg.get("num_actors", 1)),
        shards_per_update=int(actor_learner_cfg.get("shards_per_update", actor_learner_cfg.get("num_actors", 1))),
        max_inflight_per_actor=int(actor_learner_cfg.get("max_inflight_per_actor", 1)),
        poll_s=float(actor_learner_cfg.get("poll_interval_s", 0.2)),
        shard_collect_timeout_s=float(shard_collect_timeout_s),
        allow_partial_updates_after_timeout=bool(actor_learner_cfg.get("allow_partial_updates_after_timeout", False)),
        actor_heartbeat_timeout_s=float(actor_heartbeat_timeout_s),
        max_shard_version_lag=int(raw_max_shard_version_lag),
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
    device_id: int | None = None,
) -> Dict[str, Any]:
    devices: int | list[int]
    if str(accelerator) == "gpu" and device_id is not None:
        devices = [int(device_id)]
    else:
        devices = 1
    return {
        "accelerator": str(accelerator),
        "devices": devices,
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
    "resolve_grpo_config",
    "trainer_kwargs_from_learner_config",
]
