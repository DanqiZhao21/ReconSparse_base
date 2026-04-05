import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from framework.agent.policy_dummy import DummyPolicy
from framework.lightning import actor_learner_datamodule as datamodule_mod
from framework.lightning import actor_learner_module as module_mod
from framework.lightning.actor_learner_datamodule import ActorLearnerUpdateDataModule
from framework.lightning.actor_learner_module import ActorLearnerLightningModule
from framework.lightning.config import ActorLearnerLightningConfig, LearnerOptimizerConfig


class _TrainerStub:
    pass


def _learner_config(*, inner_epochs: int, algo_kind: str = "reinforcepp") -> ActorLearnerLightningConfig:
    return ActorLearnerLightningConfig(
        algo_kind=algo_kind,
        optimizer_config=LearnerOptimizerConfig(policy_lr=1e-3, value_lr=None, weight_decay=0.0),
        eta=1.0,
        clip_eps=0.2,
        vf_coef=0.0,
        value_clip_eps=0.0,
        kl_coef=0.0,
        forward_kl_coef=0.0,
        reverse_kl_coef=0.0,
        distill_temperature=1.0,
        teacher_ckpt=None,
        dual_clip=None,
        gamma=0.99,
        gae_lambda=0.95,
        ddp_seed=0,
        minibatch_size=2,
        include_obs=bool(algo_kind.startswith("ppo")),
        use_distributed_sampler=False,
        mode="async",
        num_actors=1,
        shards_per_update=1,
        poll_s=0.01,
        max_shard_version_gap=2,
        norm_eps=1e-8,
        inner_epochs=int(inner_epochs),
        accumulate_grad_batches=1,
        gradient_clip_val=0.5,
        max_updates=2,
    )


def test_datamodule_reuses_same_shards_within_inner_epochs(monkeypatch, tmp_path):
    build_calls = []
    select_calls = []

    def fake_build_training_batch(**kwargs):
        build_calls.append(kwargs["selected"])
        return SimpleNamespace(
            batch={
                "adv": torch.tensor([1.0, -1.0], dtype=torch.float32),
                "ret": torch.tensor([1.0, 0.0], dtype=torch.float32),
                "replay": [{"feature": 1.0}, {"feature": 2.0}],
                "old_logp": torch.tensor([0.0, 0.0], dtype=torch.float32),
            },
            num_samples=2,
            reward_sum=1.0,
            reward_count=2,
            done_sum=0.0,
            done_count=2,
        )

    monkeypatch.setattr(datamodule_mod, "build_training_batch", fake_build_training_batch)

    datamodule = ActorLearnerUpdateDataModule(
        paths=SimpleNamespace(root=str(tmp_path), shards_dir=str(tmp_path / "shards"), version_file=str(tmp_path / "version.txt")),
        agent=DummyPolicy(ckpt_path=None, device="cpu"),
        learner_config=_learner_config(inner_epochs=3),
        device=torch.device("cpu"),
        value_net=None,
        ddp_enabled=False,
        dist_module=SimpleNamespace(),
        world_size=1,
        rank=0,
        stage_fn=lambda *_args, **_kwargs: None,
        start_version=1,
    )

    def fake_select_shards():
        select_calls.append(datamodule.trainer.current_epoch)
        return [f"epoch{datamodule.trainer.current_epoch}.pt"]

    monkeypatch.setattr(datamodule, "_select_shards", fake_select_shards)
    datamodule.trainer = SimpleNamespace(current_epoch=0)

    datamodule.train_dataloader()
    datamodule.trainer.current_epoch = 1
    datamodule.train_dataloader()
    datamodule.trainer.current_epoch = 2
    datamodule.train_dataloader()
    datamodule.trainer.current_epoch = 3
    datamodule.train_dataloader()

    assert select_calls == [0, 3]
    assert build_calls == [["epoch0.pt"], ["epoch3.pt"]]


def test_actor_learner_module_publishes_once_per_update(monkeypatch, tmp_path):
    moved = []
    pruned = []
    written = []
    saved = []

    monkeypatch.setattr(module_mod, "move_to_consumed", lambda _paths, fp: moved.append(fp))
    monkeypatch.setattr(module_mod, "prune_consumed", lambda _paths, keep_basenames: pruned.append(sorted(keep_basenames)))
    monkeypatch.setattr(module_mod, "read_int", lambda _path, default=0: 1)
    monkeypatch.setattr(module_mod, "write_int", lambda _path, value: written.append(value))

    agent = DummyPolicy(ckpt_path=None, device="cpu")
    monkeypatch.setattr(agent, "save_checkpoint", lambda path: saved.append(path))

    root = tmp_path / "buffer"
    root.mkdir(parents=True, exist_ok=True)

    datamodule = SimpleNamespace(
        current_selected=[str(tmp_path / "shard_a.pt")],
        current_loaded=SimpleNamespace(
            batch={
                "ret": torch.tensor([1.0, 0.5], dtype=torch.float32),
                "adv": torch.tensor([0.1, -0.1], dtype=torch.float32),
            },
            num_samples=2,
            reward_sum=1.5,
            reward_count=2,
            done_sum=0.0,
            done_count=2,
        ),
        current_wait_shards_s=0.1,
        current_load_shards_s=0.1,
        current_prepare_batch_s=0.0,
        should_stop=False,
    )

    module = ActorLearnerLightningModule(
        agent=agent,
        learner_config=_learner_config(inner_epochs=3),
        paths=SimpleNamespace(root=str(root), latest_ckpt=str(root / "latest.ckpt"), version_file=str(root / "version.txt")),
        stage_fn=lambda *_args, **_kwargs: None,
        ddp_enabled=False,
        dist_module=SimpleNamespace(is_initialized=lambda: False),
        rank=0,
        wandb_enabled=False,
    )
    module.latest_metrics = {"loss_pi": 0.1}
    trainer = _TrainerStub()
    trainer.datamodule = datamodule
    trainer.should_stop = False
    trainer.current_epoch = 0
    module.trainer = trainer

    module.on_train_epoch_start()
    module.on_train_epoch_end()

    assert moved == []
    assert pruned == []
    assert written == []
    assert saved == []

    trainer.current_epoch = 2
    module.on_train_epoch_start()
    module.on_train_epoch_end()

    assert moved == [str(tmp_path / "shard_a.pt")]
    assert pruned == [["shard_a.pt"]]
    assert written == [2]
    assert saved == [str(root / "latest.ckpt")]


def test_actor_learner_module_logs_update_and_sample_views(monkeypatch, tmp_path):
    logged = []

    monkeypatch.setattr(module_mod, "move_to_consumed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module_mod, "prune_consumed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module_mod, "read_int", lambda _path, default=0: 1)
    monkeypatch.setattr(module_mod, "write_int", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module_mod, "wandb", SimpleNamespace(log=lambda payload: logged.append(payload)))

    agent = DummyPolicy(ckpt_path=None, device="cpu")
    monkeypatch.setattr(agent, "save_checkpoint", lambda _path: None)

    root = tmp_path / "buffer"
    root.mkdir(parents=True, exist_ok=True)

    datamodule = SimpleNamespace(
        current_selected=[str(tmp_path / "shard_a.pt")],
        current_loaded=SimpleNamespace(
            batch={
                "ret": torch.tensor([1.0, 0.5], dtype=torch.float32),
                "adv": torch.tensor([0.1, -0.1], dtype=torch.float32),
            },
            num_samples=2,
            reward_sum=1.5,
            reward_count=2,
            done_sum=0.0,
            done_count=2,
        ),
        current_wait_shards_s=0.1,
        current_load_shards_s=0.1,
        current_prepare_batch_s=0.0,
        should_stop=False,
    )

    module = ActorLearnerLightningModule(
        agent=agent,
        learner_config=_learner_config(inner_epochs=1),
        paths=SimpleNamespace(root=str(root), latest_ckpt=str(root / "latest.ckpt"), version_file=str(root / "version.txt")),
        stage_fn=lambda *_args, **_kwargs: None,
        ddp_enabled=False,
        dist_module=SimpleNamespace(is_initialized=lambda: False),
        rank=0,
        wandb_enabled=True,
    )
    trainer = _TrainerStub()
    trainer.datamodule = datamodule
    trainer.should_stop = False
    trainer.current_epoch = 0
    module.trainer = trainer

    module.on_train_epoch_start()
    module._record_update_metrics(
        {
            "loss_pi": torch.tensor(0.4, dtype=torch.float32),
            "approx_kl": torch.tensor(0.01, dtype=torch.float32),
        },
        batch_size=4,
    )
    module._record_update_metrics(
        {
            "loss_pi": torch.tensor(0.2, dtype=torch.float32),
            "approx_kl": torch.tensor(0.03, dtype=torch.float32),
        },
        batch_size=2,
    )
    module.on_train_epoch_end()

    assert len(logged) == 1
    payload = logged[0]
    assert payload["update"] == 0
    assert payload["global_sample_step"] == 2
    assert payload["global_step"] == 2
    assert payload["num_minibatches"] == 2
    assert payload["loss_pi"] == pytest.approx(0.3333333333333333)
    assert payload["approx_kl_max"] == pytest.approx(0.03)
    assert payload["train_update/loss_pi"] == pytest.approx(0.3333333333333333)
    assert payload["train_update/approx_kl_max"] == pytest.approx(0.03)
    assert "train_seen_samples/loss_pi" not in payload
    assert payload["global_train_seen_sample_step"] == 0


def test_trajectory_module_logs_high_frequency_train_seen_samples(monkeypatch, tmp_path):
    logged = []

    monkeypatch.setattr(
        datamodule_mod,
        "wandb",
        SimpleNamespace(log=lambda payload: logged.append(payload)),
        raising=False,
    )
    monkeypatch.setattr(
        module_mod,
        "wandb",
        SimpleNamespace(log=lambda payload: logged.append(payload)),
        raising=False,
    )
    from framework.lightning import trajectory_module as trajectory_module_mod

    monkeypatch.setattr(
        trajectory_module_mod,
        "wandb",
        SimpleNamespace(log=lambda payload: logged.append(payload)),
    )

    agent = DummyPolicy(ckpt_path=None, device="cpu")
    root = tmp_path / "buffer"
    root.mkdir(parents=True, exist_ok=True)

    module = ActorLearnerLightningModule(
        agent=agent,
        learner_config=_learner_config(inner_epochs=1),
        paths=SimpleNamespace(root=str(root), latest_ckpt=str(root / "latest.ckpt"), version_file=str(root / "version.txt")),
        stage_fn=lambda *_args, **_kwargs: None,
        ddp_enabled=False,
        dist_module=SimpleNamespace(is_initialized=lambda: False),
        rank=0,
        wandb_enabled=True,
    )
    trainer = _TrainerStub()
    trainer.current_epoch = 0
    trainer.global_step = 7
    trainer.barebones = False
    module.trainer = trainer
    monkeypatch.setattr(module, "log", lambda *_args, **_kwargs: None)

    batch = {
        "replay": [{"feature": 1.0}, {"feature": 2.0}],
        "adv": torch.tensor([1.0, -1.0], dtype=torch.float32),
        "ret": torch.tensor([0.5, -0.25], dtype=torch.float32),
        "old_logp": torch.tensor([0.0, 0.0], dtype=torch.float32),
    }

    module.training_step(batch, batch_idx=0)
    module.training_step(batch, batch_idx=1)

    assert len(logged) == 2
    assert logged[0]["global_train_seen_sample_step"] == 2
    assert logged[0]["train_seen_samples/seen_batch_size"] == pytest.approx(2.0)
    assert "train_seen_samples/loss_pi" in logged[0]
    assert logged[1]["global_train_seen_sample_step"] == 4
    assert logged[1]["update"] == 0
