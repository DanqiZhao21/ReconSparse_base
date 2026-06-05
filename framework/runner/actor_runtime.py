from __future__ import annotations

import os
import time
import traceback
import uuid
from typing import Any, Dict, Optional

import torch

from framework.io.buffer import (
    BufferPaths,
    atomic_torch_save,
    clear_actor_failure,
    count_inflight,
    ensure_buffer_layout,
    read_int,
    stop_requested,
    wait_for_version,
    write_actor_failure,
)
from framework.rollout import collect_single_env_shard, collect_vector_env_shards
from framework.rollout.timing import format_rollout_timing_summary
from framework.runner.agent_factory import build_agent
from framework.runner.config_normalization import resolve_learner_gpu_ids
from framework.runner.env_factory import build_actor_env
from framework.runner.logging import stage


def _stop_before_writing_shard(paths: BufferPaths, *, actor_id: int, shard_count: int) -> bool:
    if not stop_requested(paths):
        return False
    stage(f"[actor{actor_id}] stop requested after collecting {shard_count} shard(s); discarding unsaved shard(s)")
    return True


def _actor_should_pause_for_learner(al_cfg: Dict[str, Any], *, cuda: int) -> bool:
    if int(cuda) < 0:
        return False
    if not bool(al_cfg.get("pause_actor_on_learner_gpu", True)):
        return False
    learner_gpu_ids = {int(gpu_id) for gpu_id in resolve_learner_gpu_ids(al_cfg)}
    return int(cuda) in learner_gpu_ids


def resolve_actor_env_runtime(cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}
    num_envs_per_actor = int(al_cfg.get("num_envs_per_actor", 1))
    if num_envs_per_actor <= 0:
        raise ValueError(f"num_envs_per_actor must be >= 1, got {num_envs_per_actor}")
    vec_env_mode = str(al_cfg.get("vec_env_mode", "serial")).strip().lower()
    if vec_env_mode not in {"serial", "subproc"}:
        raise ValueError(f"vec_env_mode must be 'serial' or 'subproc', got {vec_env_mode!r}")
    return {
        "num_envs_per_actor": int(num_envs_per_actor),
        "vec_env_mode": str(vec_env_mode),
        "use_vector_env": bool(num_envs_per_actor > 1),
    }


def resolve_finish_shard_on_done(cfg: Dict[str, Any], actor_learner_cfg: Dict[str, Any], *, use_vector_env: bool) -> bool:
    raw = actor_learner_cfg.get("finish_shard_on_done", "auto")
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return not bool(use_vector_env)
        if text in {"false", "0", "no", "off"}:
            return False
        if text != "auto":
            raise ValueError(
                f"Unsupported actor_learner.finish_shard_on_done={raw!r}; expected true, false, or auto"
            )
    elif raw is not None:
        return bool(raw) and not bool(use_vector_env)

    env_backend = str(((cfg.get("env", {}) or {}).get("backend", "recon"))).strip().lower()
    return env_backend == "hugsim_ori" and not bool(use_vector_env)


def _normalize_algo_key(value: Any) -> str:
    text = str(value or "ppo").strip().lower()
    if text in {"reinforce++", "reinforce_pp", "reinforce_clip"}:
        return "reinforcepp"
    if text in {"reinforce_vanilla", "vanilla_reinforce"}:
        return "reinforce"
    return text


def resolve_store_obs(train_cfg: Dict[str, Any], actor_learner_cfg: Dict[str, Any], agent: Any) -> bool:
    raw = actor_learner_cfg.get("store_obs", "auto")
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
        if text != "auto":
            raise ValueError(f"Unsupported actor_learner.store_obs={raw!r}; expected true, false, or auto")
    elif raw is not None:
        return bool(raw)

    algo_key = _normalize_algo_key(train_cfg.get("algo", "ppo"))
    if not algo_key.startswith("ppo"):
        return False

    critic_use_agent_features = bool(train_cfg.get("critic_use_agent_features", True))
    if not critic_use_agent_features:
        return True

    supports_value_features = bool(getattr(agent, "supports_value_features", lambda: False)())
    has_feature_dim = getattr(agent, "value_feature_dim", None) is not None
    return not bool(supports_value_features and has_feature_dim)


def actor_main(
    cfg: Dict[str, Any],
    *,
    actor_id: int,
    gpu_id: Optional[int] = None,
    total_actors: Optional[int] = None,
) -> None:
    paths = BufferPaths(root=str(((cfg.get("train", {}) or {}).get("actor_learner", {}) or {}).get("buffer_dir", "outputs/actor_learner")))
    ensure_buffer_layout(paths)
    clear_actor_failure(paths, int(actor_id))
    try:
        _actor_main_impl(
            cfg,
            actor_id=actor_id,
            gpu_id=gpu_id,
            total_actors=total_actors,
            paths=paths,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        marker_path = write_actor_failure(
            paths,
            int(actor_id),
            message=f"{type(exc).__name__}: {exc}",
            traceback_text=tb,
        )
        stage(f"[actor{actor_id}] fatal error recorded at {marker_path}: {type(exc).__name__}: {exc}")
        raise


def _actor_main_impl(
    cfg: Dict[str, Any],
    *,
    actor_id: int,
    gpu_id: Optional[int],
    total_actors: Optional[int],
    paths: BufferPaths,
) -> None:
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}

    mode = str(al_cfg.get("mode", "sync")).strip().lower()
    horizon = int(al_cfg.get("actor_horizon", 32))
    max_inflight = int(al_cfg.get("max_inflight_per_actor", 2))
    poll_s = float(al_cfg.get("poll_interval_s", 0.2))
    runtime_cfg = resolve_actor_env_runtime(cfg)
    num_envs_per_actor = int(runtime_cfg["num_envs_per_actor"])
    vec_env_mode = str(runtime_cfg["vec_env_mode"])
    finish_shard_on_done = resolve_finish_shard_on_done(
        cfg,
        al_cfg,
        use_vector_env=bool(runtime_cfg["use_vector_env"]),
    )

    if torch.cuda.is_available():
        if gpu_id is None:
            gpu_id = int(actor_id) % max(1, int(torch.cuda.device_count()))
        cuda = int(gpu_id)
        torch.cuda.set_device(cuda)
        device = torch.device(f"cuda:{cuda}")
    else:
        cuda = -1
        device = torch.device("cpu")

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
    store_obs = resolve_store_obs(train_cfg, al_cfg, agent)
    stage(f"[actor{actor_id}] store_obs={bool(store_obs)}")
    shard_idx = 0
    shard_idx_per_env = [0 for _ in range(int(num_envs_per_actor))]
    if bool(runtime_cfg["use_vector_env"]):
        stage(f"[actor{actor_id}] batched actor mode enabled num_envs_per_actor={num_envs_per_actor} vec_env_mode={vec_env_mode}")

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
            if pause_actor_on_learner_gpu and _actor_should_pause_for_learner(al_cfg, cuda=int(cuda)):
                while os.path.exists(training_lock_file):
                    if stop_requested(paths):
                        stage(f"[actor{actor_id}] stop requested during learner-train pause; exiting")
                        return
                    time.sleep(poll_s)
            reserve = 1
            backpressure_wait_t0 = time.perf_counter()
            while count_inflight(paths, actor_id=str(actor_id)) >= max(1, int(max_inflight) - int(reserve) + 1):
                if stop_requested(paths):
                    stage(f"[actor{actor_id}] stop requested during backpressure; exiting")
                    return
                time.sleep(poll_s)
            backpressure_wait_s = float(time.perf_counter() - backpressure_wait_t0)

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
                store_obs=bool(store_obs),
                end_shard_on_done=bool(finish_shard_on_done),
                stop_checker=lambda: stop_requested(paths),
            )
            if _stop_before_writing_shard(paths, actor_id=int(actor_id), shard_count=1):
                break
            name = f"actor{actor_id}_e0_v{local_ver}_t{int(time.time())}_{uuid.uuid4().hex[:8]}.pt"
            timing = dict(((shard.get("meta", {}) or {}).get("timing", {}) or {}))
            timing["backpressure_wait_s"] = float(backpressure_wait_s)
            if isinstance(shard.get("meta", None), dict):
                shard["meta"]["timing"] = dict(timing)
            save_t0 = time.perf_counter()
            atomic_torch_save(shard, os.path.join(paths.shards_dir, name))
            timing["save_shard_s"] = float(time.perf_counter() - save_t0)
            timing_summary = format_rollout_timing_summary(timing)
            suffix = f" {timing_summary}" if timing_summary else ""
            stage(f"[actor{actor_id}] wrote shard {shard_idx} horizon={horizon} ver={local_ver}{suffix}")
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
            env_fns.append(
                lambda i=i, wid=wid: build_actor_env(
                    cfg,
                    cuda=int(cuda if cuda >= 0 else 0),
                    actor_id=int(actor_id),
                    worker_id=int(wid),
                    total_actors=(
                        int(total_actors)
                        if total_actors is not None
                        else max(1, int(al_cfg.get("num_actors", 1)))
                    ),
                )
            )
        vec_env = SerialVecEnv(env_fns) if not vec_env_mode.startswith("sub") else SubprocVecEnv(env_fns)
        obs_list, _info = vec_env.reset()
        while True:
            if stop_requested(paths):
                stage(f"[actor{actor_id}] stop requested; exiting")
                break
            if pause_actor_on_learner_gpu and _actor_should_pause_for_learner(al_cfg, cuda=int(cuda)):
                while os.path.exists(training_lock_file):
                    if stop_requested(paths):
                        stage(f"[actor{actor_id}] stop requested during learner-train pause; exiting")
                        return
                    time.sleep(poll_s)
            reserve = max(1, int(num_envs_per_actor))
            backpressure_wait_t0 = time.perf_counter()
            while count_inflight(paths, actor_id=str(actor_id)) >= max(1, int(max_inflight) - int(reserve) + 1):
                if stop_requested(paths):
                    stage(f"[actor{actor_id}] stop requested during backpressure; exiting")
                    return
                time.sleep(poll_s)
            backpressure_wait_s = float(time.perf_counter() - backpressure_wait_t0)
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
                store_obs=bool(store_obs),
                stop_checker=lambda: stop_requested(paths),
            )
            if _stop_before_writing_shard(paths, actor_id=int(actor_id), shard_count=len(shards)):
                break
            for i, shard in enumerate(shards):
                name = f"actor{actor_id}_e{i}_v{local_ver}_t{int(time.time())}_{uuid.uuid4().hex[:8]}.pt"
                timing = dict(((shard.get("meta", {}) or {}).get("timing", {}) or {}))
                timing["backpressure_wait_s"] = float(backpressure_wait_s)
                if isinstance(shard.get("meta", None), dict):
                    shard["meta"]["timing"] = dict(timing)
                save_t0 = time.perf_counter()
                atomic_torch_save(shard, os.path.join(paths.shards_dir, name))
                timing["save_shard_s"] = float(time.perf_counter() - save_t0)
                timing_summary = format_rollout_timing_summary(timing)
                suffix = f" {timing_summary}" if timing_summary else ""
                stage(f"[actor{actor_id}] wrote shard env={i} idx={shard_idx_per_env[i]} horizon={horizon} ver={local_ver}{suffix}")
                shard_idx_per_env[i] += 1
            if mode.startswith("sync"):
                wait_for_version(paths, min_version=int(local_ver) + 1, poll_s=poll_s, timeout_s=None, stop_file=paths.stop_file)
                if stop_requested(paths):
                    stage(f"[actor{actor_id}] stop requested after sync wait; exiting")
                    break

__all__ = ["actor_main", "resolve_actor_env_runtime", "resolve_store_obs"]
