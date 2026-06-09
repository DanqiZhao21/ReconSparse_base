from __future__ import annotations

from types import SimpleNamespace

from framework.lightning.config import actor_learner_lightning_config_from_algorithm


def test_shared_train_grpo_overrides_algorithm_grpo_fields() -> None:
    algo = SimpleNamespace(
        variant="ppo",
        policy_lr=1.0e-4,
        value_lr=5.0e-5,
        weight_decay=0.0,
        eta=1.0,
        clip_eps=0.2,
        vf_coef=0.5,
        value_clip_eps=0.0,
        kl_coef=0.0,
        grpo_coef=9.9,
        grpo_num_candidates=99,
        grpo_candidate_select="all",
        grpo_norm_eps=9.0e-2,
        grpo_use_rank_adv=True,
        grpo_score_clip=7.0,
        grpo_debug_visualize=True,
        grpo_debug_dir="legacy/debug",
        grpo_debug_max_batches=9,
        grpo_debug_top_k=9,
        ppo_epochs=2,
        grad_accum_steps=1,
        max_grad_norm=0.5,
        use_distributed_sampler=True,
        ddp_seed=0,
    )
    train_cfg = {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "minibatch_size": 4,
        "grpo": {
            "enable": True,
            "coef": 0.3,
            "num_candidates": 8,
            "candidate_select": "topk",
            "norm_eps": 1.0e-6,
            "use_rank_adv": False,
            "score_clip": 1.5,
            "objective": "expected_prob",
            "temperature": 0.7,
            "debug_visualize": False,
            "debug_dir": "outputs/grpo",
            "debug_max_batches": 2,
            "debug_top_k": 3,
        },
    }

    learner_cfg = actor_learner_lightning_config_from_algorithm(
        algo,
        train_cfg=train_cfg,
        actor_learner_cfg={"mode": "async", "num_actors": 1, "shards_per_update": 1},
        algo_meta={"algo_key": "ppo", "eta": 1.0, "clip_eps": 0.2, "rpp_norm_eps": 1.0e-8},
    )

    assert learner_cfg.grpo_coef == 0.3
    assert learner_cfg.grpo_num_candidates == 8
    assert learner_cfg.grpo_candidate_select == "topk"
    assert learner_cfg.grpo_norm_eps == 1.0e-6
    assert learner_cfg.grpo_use_rank_adv is False
    assert learner_cfg.grpo_score_clip == 1.5
    assert learner_cfg.grpo_objective == "expected_prob"
    assert learner_cfg.grpo_temperature == 0.7
    assert learner_cfg.grpo_debug_visualize is False
    assert learner_cfg.grpo_debug_dir == "outputs/grpo"
    assert learner_cfg.grpo_debug_max_batches == 2
    assert learner_cfg.grpo_debug_top_k == 3


def test_missing_shared_grpo_disables_grpo_without_legacy_backfill() -> None:
    algo = SimpleNamespace(
        variant="reinforcepp",
        policy_lr=1.0e-4,
        value_lr=None,
        weight_decay=0.0,
        eta=1.0,
        clip_eps=0.2,
        kl_coef=0.0,
        epochs=1,
        grad_accum_steps=1,
        max_grad_norm=0.5,
        use_distributed_sampler=True,
        ddp_seed=0,
    )
    train_cfg = {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "minibatch_size": 4,
        "reinforcepp": {
            "norm_eps": 1.0e-8,
            "kl_coef": 0.0,
            "epochs": 1,
            "policy_grad_weight": 0.5,
            "forward_kl_coef": 0.0,
            "reverse_kl_coef": 0.0,
            "distill_temperature": 1.0,
        },
    }

    learner_cfg = actor_learner_lightning_config_from_algorithm(
        algo,
        train_cfg=train_cfg,
        actor_learner_cfg={"mode": "async", "num_actors": 1, "shards_per_update": 1},
        algo_meta={"algo_key": "reinforcepp", "eta": 1.0, "clip_eps": 0.2, "rpp_norm_eps": 1.0e-8},
    )

    assert learner_cfg.grpo_enabled is False
    assert learner_cfg.grpo_coef == 0.0
    assert learner_cfg.grpo_num_candidates == 0
    assert learner_cfg.grpo_candidate_select == "topk"
    assert learner_cfg.grpo_norm_eps == 1.0e-6
    assert learner_cfg.grpo_use_rank_adv is False
    assert learner_cfg.grpo_score_clip is None
    assert learner_cfg.grpo_debug_visualize is False
    assert learner_cfg.grpo_debug_dir is None
    assert learner_cfg.grpo_debug_max_batches == 0
    assert learner_cfg.grpo_debug_top_k == 4


def test_grpo_only_algorithm_builds_policy_only_learner_config() -> None:
    from framework.runner.learner_factory import build_algorithm_bundle

    class Agent:
        def __init__(self) -> None:
            self.trainable_module = __import__("torch").nn.Linear(1, 1)

    cfg = {
        "train": {
            "algo": "grpo_only",
            "policy_lr": 1.0e-5,
            "clip_eps": 0.2,
            "minibatch_size": 4,
            "actor_learner": {"mode": "async", "num_actors": 1, "shards_per_update": 1},
            "grpo": {"enable": True, "coef": 1.0, "num_candidates": 8},
        }
    }

    algo, value_net, meta = build_algorithm_bundle(
        cfg,
        agent=Agent(),
        device=__import__("torch").device("cpu"),
        ddp_enabled=False,
        world_size=1,
        rank=0,
    )
    learner_cfg = actor_learner_lightning_config_from_algorithm(
        algo,
        train_cfg=cfg["train"],
        actor_learner_cfg=cfg["train"]["actor_learner"],
        algo_meta=meta,
    )

    assert value_net is None
    assert meta["algo_key"] == "grpo_only"
    assert learner_cfg.algo_kind == "grpo_only"
    assert learner_cfg.grpo_enabled is True
    assert learner_cfg.grpo_coef == 1.0
    assert learner_cfg.include_obs is False


def test_sac_algorithm_builds_policy_only_learner_config() -> None:
    from framework.runner.learner_factory import build_algorithm_bundle

    class Agent:
        def __init__(self) -> None:
            self.trainable_module = __import__("torch").nn.Linear(1, 1)

    cfg = {
        "train": {
            "algo": "sac",
            "policy_lr": 5.0e-7,
            "clip_eps": 0.2,
            "minibatch_size": 8,
            "actor_learner": {"mode": "async", "num_actors": 1, "shards_per_update": 1},
            "sac": {
                "entropy_coef": 0.02,
                "kl_coef": 0.03,
                "epochs": 2,
                "norm_eps": 1.0e-7,
            },
        }
    }

    algo, value_net, meta = build_algorithm_bundle(
        cfg,
        agent=Agent(),
        device=__import__("torch").device("cpu"),
        ddp_enabled=False,
        world_size=1,
        rank=0,
    )
    learner_cfg = actor_learner_lightning_config_from_algorithm(
        algo,
        train_cfg=cfg["train"],
        actor_learner_cfg=cfg["train"]["actor_learner"],
        algo_meta=meta,
    )

    assert value_net is None
    assert meta["algo_key"] == "sac"
    assert meta["rpp_norm_eps"] == 1.0e-7
    assert learner_cfg.algo_kind == "sac"
    assert learner_cfg.sac_entropy_coef == 0.02
    assert learner_cfg.kl_coef == 0.03
    assert learner_cfg.inner_epochs == 2
    assert learner_cfg.include_obs is False


def test_explicit_grpo_objective_is_respected() -> None:
    algo = SimpleNamespace(
        variant="reinforcepp",
        policy_lr=1.0e-4,
        value_lr=None,
        weight_decay=0.0,
        eta=1.0,
        clip_eps=0.2,
        kl_coef=0.0,
        epochs=1,
        grad_accum_steps=1,
        max_grad_norm=0.5,
        use_distributed_sampler=True,
        ddp_seed=0,
    )
    train_cfg = {
        "grpo": {"enable": True, "coef": 0.2, "num_candidates": 4, "objective": "logprob"},
    }

    learner_cfg = actor_learner_lightning_config_from_algorithm(
        algo,
        train_cfg=train_cfg,
        actor_learner_cfg={"mode": "async", "num_actors": 1, "shards_per_update": 1},
        algo_meta={"algo_key": "reinforcepp", "eta": 1.0, "clip_eps": 0.2, "rpp_norm_eps": 1.0e-8},
    )

    assert learner_cfg.grpo_objective == "logprob"


def test_auxiliary_risk_decel_config_is_respected() -> None:
    algo = SimpleNamespace(
        variant="reinforcepp",
        policy_lr=1.0e-4,
        value_lr=None,
        weight_decay=0.0,
        eta=1.0,
        clip_eps=0.2,
        kl_coef=0.0,
        epochs=1,
        grad_accum_steps=1,
        max_grad_norm=0.5,
        use_distributed_sampler=True,
        ddp_seed=0,
    )
    train_cfg = {
        "auxiliary_objectives": {
            "risk_decel": {
                "enable": True,
                "coef": 0.7,
                "dt_s": 0.5,
                "high_risk_gap_m": 10.0,
                "high_risk_ttc_s": 2.0,
                "lateral_m": 2.25,
                "speed_margin_mps": 0.15,
            },
        },
    }

    learner_cfg = actor_learner_lightning_config_from_algorithm(
        algo,
        train_cfg=train_cfg,
        actor_learner_cfg={"mode": "async", "num_actors": 1, "shards_per_update": 1},
        algo_meta={"algo_key": "reinforcepp", "eta": 1.0, "clip_eps": 0.2, "rpp_norm_eps": 1.0e-8},
    )

    assert learner_cfg.aux_risk_decel_enabled is True
    assert learner_cfg.aux_risk_decel_coef == 0.7
    assert learner_cfg.aux_risk_decel_dt_s == 0.5
    assert learner_cfg.aux_risk_decel_high_risk_gap_m == 10.0
    assert learner_cfg.aux_risk_decel_high_risk_ttc_s == 2.0
    assert learner_cfg.aux_risk_decel_lateral_m == 2.25
    assert learner_cfg.aux_risk_decel_speed_margin_mps == 0.15


def test_reinforcepp_policy_grad_weight_sets_closed_loop_loss_coef() -> None:
    algo = SimpleNamespace(
        variant="reinforcepp",
        policy_lr=1.0e-4,
        value_lr=None,
        weight_decay=0.0,
        eta=1.0,
        clip_eps=0.2,
        kl_coef=0.0,
        epochs=1,
        grad_accum_steps=1,
        max_grad_norm=0.5,
        use_distributed_sampler=True,
        ddp_seed=0,
    )
    train_cfg = {
        "reinforcepp": {"policy_grad_weight": 0.5},
        "grpo": {"enable": True, "coef": 1.0, "num_candidates": 4},
    }

    learner_cfg = actor_learner_lightning_config_from_algorithm(
        algo,
        train_cfg=train_cfg,
        actor_learner_cfg={"mode": "async", "num_actors": 1, "shards_per_update": 1},
        algo_meta={"algo_key": "reinforcepp", "eta": 1.0, "clip_eps": 0.2, "rpp_norm_eps": 1.0e-8},
    )

    assert learner_cfg.closed_loop_loss_coef == 0.5
