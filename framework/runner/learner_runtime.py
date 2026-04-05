from __future__ import annotations

import os
from typing import Any, Dict

import torch
import torch.distributed as dist

from framework.io.buffer import BufferPaths, ensure_buffer_layout, read_int, write_int
from framework.lightning.actor_learner_datamodule import ActorLearnerUpdateDataModule
from framework.lightning.actor_learner_module import ActorLearnerLightningModule
from framework.lightning.config import (
    actor_learner_lightning_config_from_algorithm,
    trainer_kwargs_from_learner_config,
)
from framework.lightning_compat import L
from framework.runner.agent_factory import build_agent
from framework.runner.dist import learner_init_dist
from framework.runner.learner_factory import build_algorithm_bundle
from framework.runner.logging import (
    _exception_is_cuda_oom,
    log_cuda_memory_snapshot,
    stage,
    wandb_init_if_enabled,
)


def learner_main(cfg: Dict[str, Any], *, learner_rank: int = 0) -> None:
    del learner_rank
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}
    ddp_cfg = train_cfg.get("ddp", {}) or {}
    ddp_timeout_s = ddp_cfg.get("timeout_s", None)
    rank, world_size, local_rank = learner_init_dist(
        timeout_s=(int(ddp_timeout_s) if ddp_timeout_s is not None else None)
    )
    ddp_enabled = world_size > 1

    paths = BufferPaths(root=str(al_cfg.get("buffer_dir", "outputs/actor_learner")))
    ensure_buffer_layout(paths)
    mode = str(al_cfg.get("mode", "async")).strip().lower()
    num_actors = int(al_cfg.get("num_actors", 2))
    learner_gpu_id = int(al_cfg.get("learner_gpu_id", 0))
    shards_per_update = int(al_cfg.get("shards_per_update", num_actors))
    max_inflight_per_actor = int(al_cfg.get("max_inflight_per_actor", 2))
    poll_s = float(al_cfg.get("poll_interval_s", 0.2))

    if mode.startswith("async"):
        total_capacity = max(1, int(num_actors)) * max(1, int(max_inflight_per_actor))
        if int(shards_per_update) > int(total_capacity):
            stage(f"[learner] config deadlock risk: clamping shards_per_update {shards_per_update} -> {total_capacity}")
            shards_per_update = int(total_capacity)

    if torch.cuda.is_available():
        learner_device_id = int(local_rank) if ddp_enabled else int(learner_gpu_id)
        torch.cuda.set_device(int(learner_device_id))
        device = torch.device(f"cuda:{learner_device_id}")
    else:
        device = torch.device("cpu")
    if rank == 0:
        log_cuda_memory_snapshot(label=f"learner_start device={device}", log_fn=stage)

    training_lock_file = os.path.join(paths.root, "TRAINING_LOCK")
    agent = build_agent(cfg, device=device)
    if rank == 0:
        log_cuda_memory_snapshot(label="learner_after_agent_init", log_fn=stage)
    if ddp_enabled and torch.cuda.is_available():
        agent.wrap_ddp(device_id=local_rank, process_group=dist.group.WORLD)

    algo, value_net, algo_meta = build_algorithm_bundle(
        cfg,
        agent=agent,
        device=device,
        ddp_enabled=ddp_enabled,
        world_size=world_size,
        rank=rank,
        process_group=(dist.group.WORLD if ddp_enabled else None),
    )
    algo_key = str(algo_meta["algo_key"])
    learner_actor_cfg = dict(al_cfg)
    learner_actor_cfg["shards_per_update"] = int(shards_per_update)
    learner_config = actor_learner_lightning_config_from_algorithm(
        algo,
        train_cfg=train_cfg,
        actor_learner_cfg=learner_actor_cfg,
        algo_meta=algo_meta,
    )
    init_distill_teacher = getattr(agent, "init_distillation_teacher", None)
    if callable(init_distill_teacher):
        if float(learner_config.forward_kl_coef) > 0.0 or float(learner_config.reverse_kl_coef) > 0.0:
            init_distill_teacher(ckpt_path=learner_config.teacher_ckpt)

    if rank == 0:
        version = read_int(paths.version_file, default=0)
        if version <= 0:
            write_int(paths.version_file, 1)
            try:
                agent.save_checkpoint(paths.latest_ckpt)
            except Exception as exc:
                stage(f"[learner] initial save failed: {exc}")
    if ddp_enabled:
        dist.barrier()

    start_version = read_int(paths.version_file, default=0)
    if rank == 0:
        stage(
            f"[learner] start algo={algo_key} device={device} "
            f"weights_version={start_version} max_updates={learner_config.max_updates if learner_config.max_updates > 0 else 'inf'}"
        )
    wb_enabled = False
    if rank == 0:
        wb_enabled = wandb_init_if_enabled(cfg, role="learner", ddp_enabled=ddp_enabled, rank=int(rank))
    module = ActorLearnerLightningModule(
        agent=agent,
        learner_config=learner_config,
        value_net=value_net,
        paths=paths,
        stage_fn=stage,
        ddp_enabled=ddp_enabled,
        dist_module=dist,
        rank=int(rank),
        wandb_enabled=bool(wb_enabled),
    )
    data = ActorLearnerUpdateDataModule(
        paths=paths,
        agent=agent,
        learner_config=learner_config,
        device=device,
        value_net=value_net,
        ddp_enabled=ddp_enabled,
        dist_module=dist,
        world_size=world_size,
        rank=int(rank),
        stage_fn=stage,
        start_version=int(start_version),
    )
    trainer = L.Trainer(**trainer_kwargs_from_learner_config(learner_config, accelerator=("gpu" if device.type == "cuda" else "cpu")))
    try:
        trainer.fit(module, datamodule=data)
    except Exception as exc:
        if rank == 0:
            if _exception_is_cuda_oom(exc):
                stage(f"[learner] caught CUDA OOM: {exc}")
                log_cuda_memory_snapshot(label="learner_exception_cuda_oom", log_fn=stage)
            else:
                stage(f"[learner] trainer.fit raised: {type(exc).__name__}: {exc}")
                log_cuda_memory_snapshot(label="learner_exception", log_fn=stage)
        raise
    finally:
        if rank == 0 and os.path.exists(training_lock_file):
            try:
                os.remove(training_lock_file)
            except Exception:
                pass


__all__ = ["learner_main"]
