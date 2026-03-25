from __future__ import annotations

import datetime
import os
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from framework.io.buffer import (
    BufferPaths,
    atomic_torch_save,
    count_inflight,
    ensure_buffer_layout,
    list_shards,
    move_to_consumed,
    prune_consumed,
    read_int,
    stop_requested,
    wait_for_version,
    write_int,
)
from framework.lightning.actor_learner_datamodule import ActorLearnerUpdateDataModule
from framework.lightning.actor_learner_module import ActorLearnerLightningModule
from framework.lightning_compat import L
from framework.rollout import collect_single_env_shard, collect_vector_env_shards
from framework.runner.factories import (
    REPO_ROOT,
    build_actor_env,
    build_agent,
    build_algorithm_bundle,
    resolve_actor_gpu_ids,
)
from framework.runner.launch_env import build_launch_env
from framework.utils.gsplat_warmup import warmup_gsplat_cuda

try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


def stage(msg: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    prefix = f"[{time.strftime('%H:%M:%S')}]"
    if world > 1:
        prefix = f"{prefix} [rank {rank}]"
    print(f"{prefix} {msg}", flush=True)


def _write_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _parse_shard_weights_version(filename: str) -> Optional[int]:
    try:
        text = str(filename)
        start = text.find("_v")
        if start < 0:
            return None
        start += 2
        end = start
        while end < len(text) and text[end].isdigit():
            end += 1
        if end == start:
            return None
        return int(text[start:end])
    except Exception:
        return None


def _filter_and_discard_stale_shards(
    paths: BufferPaths,
    shard_files: List[str],
    *,
    cur_weights_version: int,
    max_version_gap: int,
) -> List[str]:
    max_gap = max(0, min(2, int(max_version_gap)))
    upcoming = int(cur_weights_version) + 1
    min_ok = int(upcoming - max_gap)
    kept: List[str] = []
    stale: List[str] = []
    for fp in shard_files:
        version = _parse_shard_weights_version(os.path.basename(fp))
        if version is None:
            stale.append(fp)
            continue
        if int(version) < int(min_ok):
            stale.append(fp)
            continue
        if int(version) > int(cur_weights_version):
            continue
        kept.append(fp)
    for fp in stale:
        move_to_consumed(paths, fp)
    return kept


def _filter_and_discard_incompatible_shards(
    paths: BufferPaths,
    shard_files: List[str],
    *,
    agent: Any,
) -> List[str]:
    validator = getattr(agent, "replay_is_compatible", None)
    if not callable(validator):
        return shard_files

    kept: List[str] = []
    dropped = 0
    for fp in shard_files:
        try:
            shard = torch.load(fp, map_location="cpu")
            replay = list(shard.get("replay", []))
            if len(replay) == 0:
                kept.append(fp)
                continue
            if all(bool(validator(rep)) for rep in replay):
                kept.append(fp)
                continue
        except Exception as exc:
            stage(f"[learner] dropping incompatible shard {os.path.basename(fp)}: {exc}")
            move_to_consumed(paths, fp)
            dropped += 1
            continue

        stage(f"[learner] dropping incompatible shard {os.path.basename(fp)} due to replay schema mismatch")
        move_to_consumed(paths, fp)
        dropped += 1

    if dropped > 0:
        stage(f"[learner] discarded {dropped} incompatible shard(s) from {paths.shards_dir}")
    return kept


def _get_train_wandb_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {}) or {}
    wb_cfg = train_cfg.get("wandb", {}) or {}
    return wb_cfg if isinstance(wb_cfg, dict) else {}


def _wandb_enabled_in_cfg(cfg: Dict[str, Any]) -> bool:
    return bool(_get_train_wandb_cfg(cfg).get("enabled", False))


def _wandb_init_if_enabled(
    cfg: Dict[str, Any],
    *,
    role: str,
    ddp_enabled: bool,
    rank: int,
    actor_id: Optional[int] = None,
) -> bool:
    if not _WANDB_AVAILABLE:
        return False
    wb_cfg = _get_train_wandb_cfg(cfg)
    if not bool(wb_cfg.get("enabled", False)):
        return False
    if ddp_enabled and int(rank) != 0:
        return False

    project = str(wb_cfg.get("project", "ReconDreamer-RL"))
    entity = wb_cfg.get("entity", None)
    group = wb_cfg.get("group", None)
    mode = wb_cfg.get("mode", None)
    run_id = wb_cfg.get("id", None)
    resume = wb_cfg.get("resume", None)
    wb_dir = wb_cfg.get("dir", None)
    tags = wb_cfg.get("tags", None)
    name = wb_cfg.get("name", None)
    if name is None or str(name).strip() == "":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{role}_{stamp}" if actor_id is None else f"{role}{int(actor_id)}_{stamp}"

    init_kwargs: Dict[str, Any] = {"project": project, "name": str(name), "config": cfg}
    entity_str = None if entity is None else str(entity).strip()
    if entity_str:
        if entity_str.isdigit():
            stage("[wandb] train.wandb.entity looks like a numeric account id; set a team/workspace slug instead")
        else:
            init_kwargs["entity"] = entity_str
    if group is not None:
        init_kwargs["group"] = str(group)
    if mode is not None:
        init_kwargs["mode"] = str(mode)
    if run_id is not None:
        init_kwargs["id"] = str(run_id)
    if resume is not None:
        init_kwargs["resume"] = str(resume)
    if wb_dir is not None:
        init_kwargs["dir"] = str(wb_dir)
    if tags is not None:
        init_kwargs["tags"] = tags

    try:
        wandb.init(**init_kwargs)
        wandb.define_metric("update")
        wandb.define_metric("global_step")
        for key in [
            "loss_pi",
            "loss_v",
            "approx_kl",
            "ratio_mean",
            "adv_mean",
            "collect_time_s",
            "train_time_s",
            "update_time_s",
            "reward_mean",
            "reward_sum",
            "done_rate",
            "ret_mean",
            "ret_std",
            "adv_std",
            "samples",
            "shards",
            "weights_version",
        ]:
            wandb.define_metric(key, summary="last")
        return True
    except Exception as exc:
        stage(f"[wandb] init failed: {exc}")
        return False


def learner_init_dist(*, timeout_s: Optional[int] = None) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if timeout_s is None:
            timeout_s = 2 * 60 * 60
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(seconds=int(timeout_s)),
        )
    return rank, world_size, local_rank


def _stop_before_writing_shard(paths: BufferPaths, *, actor_id: int, shard_count: int) -> bool:
    if not stop_requested(paths):
        return False
    stage(f"[actor{actor_id}] stop requested after collecting {shard_count} shard(s); discarding unsaved shard(s)")
    return True


def actor_main(
    cfg: Dict[str, Any],
    *,
    actor_id: int,
    gpu_id: Optional[int] = None,
    total_actors: Optional[int] = None,
) -> None:
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}
    paths = BufferPaths(root=str(al_cfg.get("buffer_dir", "outputs/actor_learner")))
    ensure_buffer_layout(paths)

    mode = str(al_cfg.get("mode", "sync")).strip().lower()
    horizon = int(al_cfg.get("actor_horizon", 32))
    policy_execute_mode = str(train_cfg.get("policy_execute_mode", train_cfg.get("ddv2_execute_mode", "continuous"))).strip().lower().replace("-", "_")
    max_inflight = int(al_cfg.get("max_inflight_per_actor", 2))
    poll_s = float(al_cfg.get("poll_interval_s", 0.2))
    num_envs_per_actor = max(1, int(al_cfg.get("num_envs_per_actor", 1)))
    vec_env_mode = str(al_cfg.get("vec_env_mode", "serial")).strip().lower()
    if num_envs_per_actor != 1:
        stage(f"[actor{actor_id}] forcing num_envs_per_actor=1 (process-level parallel only)")
        num_envs_per_actor = 1

    if torch.cuda.is_available():
        if gpu_id is None:
            gpu_id = int(actor_id) % max(1, int(torch.cuda.device_count()))
        cuda = int(gpu_id)
        torch.cuda.set_device(cuda)
        device = torch.device(f"cuda:{cuda}")
    else:
        cuda = -1
        device = torch.device("cpu")

    learner_gpu_id = int(al_cfg.get("learner_gpu_id", 0))
    pause_actor_on_learner_gpu = bool(al_cfg.get("pause_actor_on_learner_gpu", True))
    training_lock_file = os.path.join(paths.root, "TRAINING_LOCK")
    agent = build_agent(cfg, device=device)
    local_ver = 0
    v0 = read_int(paths.version_file, default=0)
    if v0 > 0 and os.path.exists(paths.latest_ckpt):
        try:
            agent.load_checkpoint(paths.latest_ckpt)
            local_ver = int(v0)
            stage(f"[actor{actor_id}] loaded learner weights ver={local_ver}")
        except Exception as exc:
            stage(f"[actor{actor_id}] failed to load learner weights: {exc}")

    eta = float(train_cfg.get("eta", train_cfg.get("ddv2_eta", 1.0)))
    mode_idx = int(train_cfg.get("mode_idx", train_cfg.get("ddv2_mode_idx", -1)))
    mode_select = str(train_cfg.get("policy_mode_select", train_cfg.get("ddv2_mode_select", "sample"))).strip().lower()
    gamma = float(train_cfg.get("gamma", 0.99))
    shard_idx = 0
    shard_idx_per_env = [0 for _ in range(int(num_envs_per_actor))]

    if num_envs_per_actor == 1:
        env = build_actor_env(
            cfg,
            cuda=int(cuda if cuda >= 0 else 0),
            actor_id=int(actor_id),
            total_actors=(int(total_actors) if total_actors is not None else max(1, int(al_cfg.get("num_actors", 1)))),
        )
        obs, _info = env.reset()
        while True:
            if stop_requested(paths):
                stage(f"[actor{actor_id}] stop requested; exiting")
                break
            if pause_actor_on_learner_gpu and int(cuda) >= 0 and int(cuda) == int(learner_gpu_id):
                while os.path.exists(training_lock_file):
                    if stop_requested(paths):
                        stage(f"[actor{actor_id}] stop requested during learner-train pause; exiting")
                        return
                    time.sleep(poll_s)
            reserve = 1
            while count_inflight(paths, actor_id=str(actor_id)) >= max(1, int(max_inflight) - int(reserve) + 1):
                if stop_requested(paths):
                    stage(f"[actor{actor_id}] stop requested during backpressure; exiting")
                    return
                time.sleep(poll_s)

            cur_ver = read_int(paths.version_file, default=0)
            if cur_ver > local_ver and os.path.exists(paths.latest_ckpt):
                try:
                    agent.load_checkpoint(paths.latest_ckpt)
                    local_ver = int(cur_ver)
                    stage(f"[actor{actor_id}] updated weights ver={local_ver}")
                except Exception as exc:
                    stage(f"[actor{actor_id}] weight reload failed: {exc}")

            shard, obs = collect_single_env_shard(
                env=env,
                agent=agent,
                obs=obs,
                horizon=horizon,
                eta=eta,
                mode_idx=mode_idx,
                mode_select=mode_select,
                actor_id=actor_id,
                local_ver=local_ver,
                shard_idx=shard_idx,
            )
            if _stop_before_writing_shard(paths, actor_id=int(actor_id), shard_count=1):
                break
            name = f"actor{actor_id}_e0_v{local_ver}_t{int(time.time())}_{uuid.uuid4().hex[:8]}.pt"
            atomic_torch_save(shard, os.path.join(paths.shards_dir, name))
            stage(f"[actor{actor_id}] wrote shard {shard_idx} horizon={horizon} ver={local_ver}")
            shard_idx += 1
            if mode.startswith("sync"):
                wait_for_version(paths, min_version=int(local_ver) + 1, poll_s=poll_s, timeout_s=None, stop_file=paths.stop_file)
                if stop_requested(paths):
                    stage(f"[actor{actor_id}] stop requested after sync wait; exiting")
                    break
    else:
        from framework.env_wrapper import SerialVecEnv, SubprocVecEnv

        env_fns = []
        for i in range(int(num_envs_per_actor)):
            wid = int(actor_id) * 1000 + int(i)
            env_fns.append(lambda i=i: build_actor_env(cfg, cuda=int(cuda if cuda >= 0 else 0), actor_id=int(actor_id), worker_id=int(wid)))
        vec_env = SerialVecEnv(env_fns) if not vec_env_mode.startswith("sub") else SubprocVecEnv(env_fns)
        obs_list, _info = vec_env.reset()
        while True:
            if stop_requested(paths):
                stage(f"[actor{actor_id}] stop requested; exiting")
                break
            reserve = max(1, int(num_envs_per_actor))
            while count_inflight(paths, actor_id=str(actor_id)) >= max(1, int(max_inflight) - int(reserve) + 1):
                if stop_requested(paths):
                    stage(f"[actor{actor_id}] stop requested during backpressure; exiting")
                    return
                time.sleep(poll_s)
            cur_ver = read_int(paths.version_file, default=0)
            if cur_ver > local_ver and os.path.exists(paths.latest_ckpt):
                try:
                    agent.load_checkpoint(paths.latest_ckpt)
                    local_ver = int(cur_ver)
                    stage(f"[actor{actor_id}] updated weights ver={local_ver}")
                except Exception as exc:
                    stage(f"[actor{actor_id}] weight reload failed: {exc}")

            shards, obs_list = collect_vector_env_shards(
                vec_env=vec_env,
                agent=agent,
                obs_list=obs_list,
                num_envs_per_actor=num_envs_per_actor,
                horizon=horizon,
                eta=eta,
                mode_idx=mode_idx,
                mode_select=mode_select,
                actor_id=actor_id,
                local_ver=local_ver,
                shard_idx_per_env=shard_idx_per_env,
            )
            if _stop_before_writing_shard(paths, actor_id=int(actor_id), shard_count=len(shards)):
                break
            for i, shard in enumerate(shards):
                name = f"actor{actor_id}_e{i}_v{local_ver}_t{int(time.time())}_{uuid.uuid4().hex[:8]}.pt"
                atomic_torch_save(shard, os.path.join(paths.shards_dir, name))
                stage(f"[actor{actor_id}] wrote shard env={i} idx={shard_idx_per_env[i]} horizon={horizon} ver={local_ver}")
                shard_idx_per_env[i] += 1
            if mode.startswith("sync"):
                wait_for_version(paths, min_version=int(local_ver) + 1, poll_s=poll_s, timeout_s=None, stop_file=paths.stop_file)
                if stop_requested(paths):
                    stage(f"[actor{actor_id}] stop requested after sync wait; exiting")
                    break


def learner_main(cfg: Dict[str, Any], *, learner_rank: int = 0) -> None:
    del learner_rank
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}
    ddp_cfg = (train_cfg.get("ddp", {}) or {})
    ddp_timeout_s = ddp_cfg.get("timeout_s", None)
    rank, world_size, local_rank = learner_init_dist(timeout_s=(int(ddp_timeout_s) if ddp_timeout_s is not None else None))
    ddp_enabled = world_size > 1

    paths = BufferPaths(root=str(al_cfg.get("buffer_dir", "outputs/actor_learner")))
    ensure_buffer_layout(paths)
    mode = str(al_cfg.get("mode", "sync")).strip().lower()
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

    raw_max_updates = al_cfg.get("max_updates", train_cfg.get("updates", 0))
    max_updates = int(raw_max_updates or 0)
    gamma = float(train_cfg.get("gamma", 0.99))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    if torch.cuda.is_available():
        learner_device_id = int(local_rank) if ddp_enabled else int(learner_gpu_id)
        torch.cuda.set_device(int(learner_device_id))
        device = torch.device(f"cuda:{learner_device_id}")
    else:
        device = torch.device("cpu")

    training_lock_file = os.path.join(paths.root, "TRAINING_LOCK")
    agent = build_agent(cfg, device=device)
    if ddp_enabled and torch.cuda.is_available():
        agent.wrap_ddp(device_id=local_rank, process_group=dist.group.WORLD)
    #构建 算法包
    algo, value_net, value_optim, algo_meta = build_algorithm_bundle(
        cfg,
        agent=agent,
        device=device,
        ddp_enabled=ddp_enabled,
        world_size=world_size,
        rank=rank,
        process_group=(dist.group.WORLD if ddp_enabled else None),
    )
    algo_key = str(algo_meta["algo_key"])

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
        stage(f"[learner] start algo={algo_key} device={device} weights_version={start_version} max_updates={max_updates if max_updates > 0 else 'inf'}")
    wb_enabled = False
    if rank == 0:
        wb_enabled = _wandb_init_if_enabled(cfg, role="learner", ddp_enabled=ddp_enabled, rank=int(rank))
    module = ActorLearnerLightningModule(
        agent=agent,
        optimizer=algo.optimizer,
        algo_kind=algo_key,
        eta=float(getattr(algo, "eta", 1.0)),
        clip_eps=float(getattr(algo, "clip_eps", 0.2)),
        vf_coef=float(getattr(algo, "vf_coef", 0.0)),
        value_clip_eps=float(getattr(algo, "value_clip_eps", 0.0)),
        kl_coef=float(getattr(algo, "kl_coef", 0.0)),
        dual_clip=getattr(algo, "dual_clip", None),
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
        algo_key=algo_key,
        device=device,
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        value_net=value_net,
        value_optim=value_optim,
        ddp_enabled=ddp_enabled,
        dist_module=dist,
        world_size=world_size,
        rank=int(rank),
        seed=int(getattr(algo, "ddp_seed", 0)),
        minibatch_size=int(getattr(algo, "minibatch_size", train_cfg.get("minibatch_size", 64))),
        include_obs=bool(str(algo_key).startswith("ppo")),
        use_distributed_sampler=bool(getattr(algo, "use_distributed_sampler", True)),
        mode=mode,
        num_actors=num_actors,
        shards_per_update=shards_per_update,
        poll_s=float(poll_s),
        max_shard_version_gap=int(al_cfg.get("max_shard_version_gap", 2)),
        norm_eps=float(algo_meta.get("rpp_norm_eps", 1e-8)),
        stage_fn=stage,
        start_version=int(start_version),
    )
    trainer = L.Trainer(
        accelerator=("gpu" if device.type == "cuda" else "cpu"),
        devices=1,
        max_epochs=(int(max_updates) if int(max_updates) > 0 else -1),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accumulate_grad_batches=int(getattr(algo, "grad_accum_steps", 1)),
        gradient_clip_val=float(getattr(algo, "max_grad_norm", 0.0)),
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        reload_dataloaders_every_n_epochs=1,
    )
    try:
        trainer.fit(module, datamodule=data)
    finally:
        if rank == 0 and os.path.exists(training_lock_file):
            try:
                os.remove(training_lock_file)
            except Exception:
                pass


def orchestrator_main(cfg: Dict[str, Any], *, config_path: str | None = None) -> None:
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}
    agent_cfg = cfg.get("agent", {}) or {}
    if config_path is None:
        raise ValueError("orchestrator_main requires config_path for subprocess launch")
    num_actors = int(al_cfg.get("num_actors", 4))
    actor_gpu_plan = resolve_actor_gpu_ids(al_cfg, num_actors=num_actors)
    learner_gpu_id = int(al_cfg.get("learner_gpu_id", 0))
    paths = BufferPaths(root=str(al_cfg.get("buffer_dir", "outputs/actor_learner")))
    ensure_buffer_layout(paths)
    training_lock_file = os.path.join(paths.root, "TRAINING_LOCK")
    for path in [paths.stop_file, training_lock_file]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    py = sys.executable
    entry = os.path.join(REPO_ROOT, "script", "train_actor_learner_v2.py")
    launch_env = build_launch_env(agent_type=agent_cfg.get("type", "ddv2"))
    stage(f"[orchestrator] launch learner_gpu={learner_gpu_id} num_actors={num_actors} actor_gpu_plan={actor_gpu_plan}")
    stage(
        f"[orchestrator] env CUDA_HOME={launch_env.get('CUDA_HOME', '')} TORCH_EXTENSIONS_DIR={launch_env.get('TORCH_EXTENSIONS_DIR', '')}"
    )
    stage("[orchestrator] warmup gsplat CUDA extension before launching worker fan-out")
    warmup_gsplat_cuda(py, env=launch_env)

    learner_env = launch_env.copy()
    learner_env.setdefault("RANK", "0")
    learner_env.setdefault("WORLD_SIZE", "1")
    learner_env.setdefault("LOCAL_RANK", str(int(learner_gpu_id)))
    learner_cmd = [py, entry, "--config", str(config_path), "--role", "learner"]
    learner_proc = subprocess.Popen(learner_cmd, env=learner_env)
    actor_procs: List[subprocess.Popen] = []
    try:
        for aid in range(num_actors):
            gpu_id = int(actor_gpu_plan[aid]) if aid < len(actor_gpu_plan) else -1
            actor_cmd = [
                py,
                entry,
                "--config",
                str(config_path),
                "--role",
                "actor",
                "--actor-id",
                str(int(aid)),
                "--gpu-id",
                str(int(gpu_id)),
                "--num-actors",
                str(int(num_actors)),
            ]
            actor_env = launch_env.copy()
            actor_env.setdefault("LOCAL_RANK", str(int(gpu_id if gpu_id >= 0 else 0)))
            actor_procs.append(subprocess.Popen(actor_cmd, env=actor_env))
        while True:
            lret = learner_proc.poll()
            if lret is not None:
                stage(f"[orchestrator] learner exited code={lret}")
                break
            for i, proc in enumerate(actor_procs):
                pret = proc.poll()
                if pret is not None and pret != 0:
                    stage(f"[orchestrator] actor{i} exited early code={pret}")
            time.sleep(2.0)
    finally:
        try:
            _write_text(paths.stop_file, "stop requested by orchestrator\n")
        except Exception:
            pass
        for proc in actor_procs:
            try:
                proc.wait(timeout=15)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
        try:
            learner_proc.wait(timeout=15)
        except Exception:
            try:
                learner_proc.terminate()
            except Exception:
                pass


__all__ = [
    "actor_main",
    "learner_main",
    "learner_init_dist",
    "normalize_actor_learner_cfg",
    "orchestrator_main",
    "resolve_actor_gpu_ids",
    "stage",
]