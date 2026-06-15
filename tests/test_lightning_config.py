from __future__ import annotations

from framework.lightning.config import (
    ActorLearnerLightningConfig,
    LearnerOptimizerConfig,
    actor_learner_lightning_config_from_algorithm,
    trainer_kwargs_from_learner_config,
)


def _build_learner_config() -> ActorLearnerLightningConfig:
    return ActorLearnerLightningConfig(
        algo_kind="ppo",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4, value_lr=1.0e-4),
        eta=1.0,
        clip_eps=0.2,
        inner_epochs=1,
        max_updates=1,
    )


def test_trainer_kwargs_pin_single_gpu_to_requested_index() -> None:
    learner_config = _build_learner_config()

    kwargs = trainer_kwargs_from_learner_config(
        learner_config,
        accelerator="gpu",
        device_id=2,
    )

    assert kwargs["devices"] == [2]


def test_trainer_kwargs_keep_scalar_device_count_for_cpu() -> None:
    learner_config = _build_learner_config()

    kwargs = trainer_kwargs_from_learner_config(
        learner_config,
        accelerator="cpu",
        device_id=None,
    )

    assert kwargs["devices"] == 1


def test_actor_learner_config_reads_async_collection_timing_options() -> None:
    class _Algo:
        eta = 1.0
        clip_eps = 0.2
        minibatch_size = 4

    cfg = actor_learner_lightning_config_from_algorithm(
        _Algo(),
        train_cfg={"gamma": 0.99},
        actor_learner_cfg={
            "mode": "async",
            "num_actors": 12,
            "shards_per_update": 24,
            "samples_per_update": 768,
            "poll_interval_s": 0.1,
            "shard_collect_timeout_s": 30.0,
            "actor_shard_stall_timeout_s": 300.0,
            "allow_partial_updates_after_timeout": True,
        },
        algo_meta={"algo_key": "ppo", "eta": 1.0, "clip_eps": 0.2},
    )

    assert cfg.shard_collect_timeout_s == 30.0
    assert cfg.samples_per_update == 768
    assert cfg.allow_partial_updates_after_timeout is True
    assert cfg.actor_heartbeat_timeout_s == 150.0
    assert cfg.actor_shard_stall_timeout_s == 300.0


def test_actor_learner_config_defaults_partial_async_updates_when_timeout_is_configured() -> None:
    class _Algo:
        eta = 1.0
        clip_eps = 0.2
        minibatch_size = 4

    cfg = actor_learner_lightning_config_from_algorithm(
        _Algo(),
        train_cfg={"gamma": 0.99},
        actor_learner_cfg={
            "mode": "async",
            "num_actors": 12,
            "shards_per_update": 12,
            "shard_collect_timeout_s": 30.0,
        },
        algo_meta={"algo_key": "ppo", "eta": 1.0, "clip_eps": 0.2},
    )

    assert cfg.allow_partial_updates_after_timeout is True


def test_actor_learner_config_reads_reinforcepp_advantage_normalization_flag() -> None:
    class _Algo:
        eta = 1.0
        clip_eps = 0.2
        minibatch_size = 4

    cfg = actor_learner_lightning_config_from_algorithm(
        _Algo(),
        train_cfg={
            "gamma": 0.99,
            "reinforcepp": {
                "normalize_advantage": False,
            },
        },
        actor_learner_cfg={},
        algo_meta={"algo_key": "reinforcepp", "eta": 1.0, "clip_eps": 0.2},
    )

    assert cfg.normalize_advantage is False


def test_actor_learner_config_ignores_removed_wandb_logging_flags() -> None:
    class _Algo:
        eta = 1.0
        clip_eps = 0.2
        minibatch_size = 4

    cfg = actor_learner_lightning_config_from_algorithm(
        _Algo(),
        train_cfg={
            "gamma": 0.99,
            "wandb": {
                "log_minibatch_metrics": True,
                "log_legacy_raw_metrics": True,
            },
        },
        actor_learner_cfg={
            "mode": "async",
            "num_actors": 1,
            "shards_per_update": 1,
        },
        algo_meta={"algo_key": "ppo", "eta": 1.0, "clip_eps": 0.2},
    )

    assert not hasattr(cfg, "wandb_log_minibatch_metrics")
    assert not hasattr(cfg, "wandb_log_legacy_raw_metrics")


def test_actor_learner_config_reads_debug_retention_options() -> None:
    class _Algo:
        eta = 1.0
        clip_eps = 0.2
        minibatch_size = 4

    cfg = actor_learner_lightning_config_from_algorithm(
        _Algo(),
        train_cfg={"gamma": 0.99},
        actor_learner_cfg={
            "mode": "async",
            "num_actors": 1,
            "shards_per_update": 1,
            "debug_retain_versions": 5,
            "debug_retain_ckpts": True,
            "debug_retain_shards": True,
        },
        algo_meta={"algo_key": "reinforcepp", "eta": 1.0, "clip_eps": 0.2},
    )

    assert cfg.debug_retain_versions == 5
    assert cfg.debug_retain_ckpts is True
    assert cfg.debug_retain_shards is True
