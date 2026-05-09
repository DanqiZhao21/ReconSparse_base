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
    assert learner_cfg.grpo_debug_visualize is False
    assert learner_cfg.grpo_debug_dir == "outputs/grpo"
    assert learner_cfg.grpo_debug_max_batches == 2
    assert learner_cfg.grpo_debug_top_k == 3


def test_legacy_reinforcepp_grpo_fields_still_backfill_when_shared_grpo_missing() -> None:
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
            "grpo_coef": 0.7,
            "grpo_num_candidates": 6,
            "grpo_candidate_select": "all",
            "grpo_norm_eps": 1.0e-5,
            "grpo_use_rank_adv": True,
            "grpo_score_clip": 2.5,
            "grpo_debug_visualize": True,
            "grpo_debug_dir": "legacy/grpo",
            "grpo_debug_max_batches": 4,
            "grpo_debug_top_k": 2,
        },
    }

    learner_cfg = actor_learner_lightning_config_from_algorithm(
        algo,
        train_cfg=train_cfg,
        actor_learner_cfg={"mode": "async", "num_actors": 1, "shards_per_update": 1},
        algo_meta={"algo_key": "reinforcepp", "eta": 1.0, "clip_eps": 0.2, "rpp_norm_eps": 1.0e-8},
    )

    assert learner_cfg.grpo_coef == 0.7
    assert learner_cfg.grpo_num_candidates == 6
    assert learner_cfg.grpo_candidate_select == "all"
    assert learner_cfg.grpo_norm_eps == 1.0e-5
    assert learner_cfg.grpo_use_rank_adv is True
    assert learner_cfg.grpo_score_clip == 2.5
    assert learner_cfg.grpo_debug_visualize is True
    assert learner_cfg.grpo_debug_dir == "legacy/grpo"
    assert learner_cfg.grpo_debug_max_batches == 4
    assert learner_cfg.grpo_debug_top_k == 2


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
