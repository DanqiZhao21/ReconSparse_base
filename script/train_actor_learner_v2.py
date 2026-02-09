import os
import sys
import time
import uuid
import argparse
import random
import datetime
from typing import Any, Dict, List, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

# Project root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from framework.env_wrapper import (
    RLReconEnv,
    SceneSamplingSpec,
    SceneSamplingEnv,
    make_scene_sampling_env,
    SubprocVecEnv,
    SerialVecEnv,
)
from framework.agent.policy_diffusiondrivev2 import DiffusionDriveV2Policy
from framework.algorithms.reinforcepp import ReinforcePP
from framework.algorithms.ppo import PPO
from framework.io.buffer import (
    BufferPaths,
    atomic_torch_save,
    ensure_buffer_layout,
    list_shards,
    move_to_consumed,
    prune_consumed,
    read_int,
    write_int,
    wait_for_version,
    count_inflight,
    stop_requested,
)
from framework.utils.obs import obs_to_tensor
from framework.algorithms.ppo_ddv2_core import compute_gae, normalize_advantages as ppo_normalize_advantages
from framework.algorithms.reinforcepp_core import compute_returns, normalize_advantages as reinforcepp_normalize_advantages


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage(msg: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    prefix = f"[{time.strftime('%H:%M:%S')}]"
    if world > 1:
        prefix = f"{prefix} [rank {rank}]"
    print(f"{prefix} {msg}", flush=True)


def build_actor_env(cfg: Dict[str, Any], *, cuda: int, actor_id: int, worker_id: Optional[int] = None) -> SceneSamplingEnv:
    env_cfg = cfg.get("env", {}) or {}

    render_w = env_cfg.get("render_w", None)
    render_h = env_cfg.get("render_h", None)
    step_frames = env_cfg.get("step_frames", None)

    start_cfg = env_cfg.get("start_frame", {}) or {}
    start_mode = str(start_cfg.get("mode", "random")).strip().lower()
    allow_short_tail = bool(start_cfg.get("allow_short_tail", False))
    start_min = int(start_cfg.get("min", 0))
    start_max = start_cfg.get("max", None)
    start_stride = start_cfg.get("stride", None)

    max_steps = int(env_cfg.get("max_steps", 60))

    use_all_scenes = bool(env_cfg.get("use_all_scenes", True))
    require_ckpt = bool(env_cfg.get("require_ckpt", True))
    scene_sampling = str(env_cfg.get("scene_sampling", "random")).strip().lower()
    scene0 = int(env_cfg.get("scene", 0))

    # Discover scenes
    from script.train_actor_learner import discover_scene_ids
    from reconsimulator.envs import nus_config as nus_cfg
    scene_ids = [scene0]
    if use_all_scenes:
        scene_ids = discover_scene_ids(nus_cfg.BASE_DATA_DIR, require_ckpt=require_ckpt) or [scene0]

    ddp_seed = int(((cfg.get("train", {}) or {}).get("ddp", {}) or {}).get("seed", 0))
    rank = int(os.environ.get("RANK", "0"))
    wid = int(worker_id) if worker_id is not None else int(actor_id)

    return make_scene_sampling_env(
        cuda=int(cuda),
        reward_cfg=env_cfg.get("reward", {}) or {},
        debug=bool(env_cfg.get("debug", False)),
        scene_ids=list(scene_ids),
        scene_sampling=str(scene_sampling),
        ddp_seed=int(ddp_seed),
        rank=int(rank),
        worker_id=int(wid),
        start_mode=str(start_mode),
        allow_short_tail=bool(allow_short_tail),
        start_min=int(start_min),
        start_max=(int(start_max) if start_max is not None else None),
        start_stride=(int(start_stride) if start_stride is not None else None),
        max_steps=int(max_steps),
        render_w=(int(render_w) if render_w is not None else None),
        render_h=(int(render_h) if render_h is not None else None),
        step_frames=(int(step_frames) if step_frames is not None else None),
    )


def actor_main(cfg: Dict[str, Any], *, actor_id: int) -> None:
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}

    buffer_root = str(al_cfg.get("buffer_dir", "outputs/actor_learner"))
    paths = BufferPaths(root=buffer_root)
    ensure_buffer_layout(paths)

    mode = str(al_cfg.get("mode", "sync")).strip().lower()  # sync | async | mini_sync
    horizon = int(al_cfg.get("actor_horizon", 32))
    commit_steps = max(1, int(al_cfg.get("commit_steps", 1)))
    max_inflight = int(al_cfg.get("max_inflight_per_actor", 2))
    poll_s = float(al_cfg.get("poll_interval_s", 0.2))

    num_envs_per_actor = max(1, int(al_cfg.get("num_envs_per_actor", 1)))
    vec_env_mode = str(al_cfg.get("vec_env_mode", "serial")).strip().lower()

    cuda = 0 if torch.cuda.is_available() else -1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build agent
    agent_cfg = cfg.get("agent", {}) or {}
    ckpt_path = agent_cfg.get("ckpt", None)
    if ckpt_path is None:
        raise RuntimeError("agent.ckpt is required")
    agent = DiffusionDriveV2Policy(ckpt_path=str(ckpt_path), device=str(device), rl_lr=float(train_cfg.get("ddv2_lr", 1e-5)))

    # Warm-load learner weights if present
    local_ver = 0
    v0 = read_int(paths.version_file, default=0)
    if v0 > 0 and os.path.exists(paths.latest_ckpt):
        try:
            agent.load_checkpoint(paths.latest_ckpt)
            local_ver = int(v0)
            stage(f"💚[actor{actor_id}] loaded learner weights ver={local_ver}")
        except Exception as e:
            stage(f"[actor{actor_id}] failed to load learner weights: {e}")

    ddv2_eta = float(train_cfg.get("ddv2_eta", 1.0))
    ddv2_mode_idx = int(train_cfg.get("ddv2_mode_idx", -1))
    ddv2_mode_select = str(train_cfg.get("ddv2_mode_select", "sample")).strip().lower()

    gamma = float(train_cfg.get("gamma", 0.99))

    shard_idx = 0
    shard_idx_per_env = [0 for _ in range(int(num_envs_per_actor))]

    if num_envs_per_actor == 1:
        env = build_actor_env(cfg, cuda=int(cuda if cuda >= 0 else 0), actor_id=int(actor_id))
        obs, info = env.reset()
        vec_env = None
        while True:
            if stop_requested(paths):
                stage(f"[actor{actor_id}] stop requested; exiting")
                break

            # Backpressure
            reserve = 1
            while count_inflight(paths, actor_id=str(actor_id)) >= max(1, int(max_inflight) - int(reserve) + 1):
                if stop_requested(paths):
                    stage(f"[actor{actor_id}] stop requested during backpressure; exiting")
                    return
                time.sleep(poll_s)

            # Pull latest weights
            cur_ver = read_int(paths.version_file, default=0)
            if cur_ver > local_ver and os.path.exists(paths.latest_ckpt):
                try:
                    agent.load_checkpoint(paths.latest_ckpt)
                    local_ver = int(cur_ver)
                    stage(f"💚[actor{actor_id}] updated weights ver={local_ver}")
                except Exception as e:
                    stage(f"[actor{actor_id}] weight reload failed: {e}")

            obs_buf: List[torch.Tensor] = []
            old_logp_buf: List[torch.Tensor] = []
            rew_buf: List[float] = []
            done_buf: List[float] = []
            replay_buf: List[Dict[str, Any]] = []

            last_next_obs_t: Optional[torch.Tensor] = None
            last_done: float = 1.0

            macro_steps = 0
            while macro_steps < horizon:
                obs_decision = obs
                obs_t = obs_to_tensor(obs_decision, device=torch.device("cpu")).squeeze(0).detach().cpu()
                action0, logp, replay = agent.act(obs_decision, eta=ddv2_eta, mode_idx=ddv2_mode_idx, mode_select=ddv2_mode_select)

                traj_xyyaw = replay.get("traj_xyyaw", None)
                if torch.is_tensor(traj_xyyaw) and traj_xyyaw.ndim == 3:
                    traj_xyyaw = traj_xyyaw[0]

                macro_reward = 0.0
                done = False
                next_obs_after = obs_decision
                for k in range(int(commit_steps)):
                    if torch.is_tensor(traj_xyyaw) and int(k) < int(traj_xyyaw.shape[0]):
                        x = float(traj_xyyaw[k, 0].item())
                        y = float(traj_xyyaw[k, 1].item())
                        yaw = float(traj_xyyaw[k, 2].item())
                        action = (x, y, yaw, 2)
                    else:
                        action = action0

                    obs, reward, terminated, truncated, info = env.step(action)
                    next_obs_after = obs
                    macro_reward += (float(gamma) ** float(k)) * float(reward)
                    done = bool(terminated or truncated)
                    if done:
                        break

                obs_buf.append(obs_t)
                old_logp_buf.append(logp.detach().cpu().float())
                replay_buf.append(replay)
                rew_buf.append(float(macro_reward))
                done_buf.append(1.0 if done else 0.0)
                macro_steps += 1

                try:
                    last_next_obs_t = obs_to_tensor(next_obs_after, device=torch.device("cpu")).squeeze(0).detach().cpu()
                except Exception:
                    last_next_obs_t = torch.zeros((18, 64, 64), dtype=torch.float32)
                last_done = 1.0 if done else 0.0

                if done:
                    obs, info = env.reset()

            next_obs_t = last_next_obs_t
            if next_obs_t is None:
                try:
                    next_obs_t = obs_to_tensor(obs, device=torch.device("cpu")).squeeze(0).detach().cpu()
                except Exception:
                    next_obs_t = torch.zeros((18, 64, 64), dtype=torch.float32)
            done_last = float(last_done) if len(done_buf) > 0 else 1.0

            shard = {
                "obs": torch.stack(obs_buf, dim=0),
                "old_logp": torch.stack(old_logp_buf, dim=0).view(-1),
                "reward": torch.tensor(rew_buf, dtype=torch.float32),
                "done": torch.tensor(done_buf, dtype=torch.float32),
                "next_obs": next_obs_t,
                "done_last": torch.tensor(done_last, dtype=torch.float32),
                "replay": replay_buf,
                "meta": {
                    "actor_id": int(actor_id),
                    "env_id": 0,
                    "horizon": int(horizon),
                    "commit_steps": int(commit_steps),
                    "weights_version": int(local_ver),
                    "time": float(time.time()),
                    "shard_idx": int(shard_idx),
                },
            }
            name = f"actor{actor_id}_e0_v{local_ver}_t{int(time.time())}_{uuid.uuid4().hex[:8]}.pt"
            out_path = os.path.join(paths.shards_dir, name)
            stage(f"[actor{actor_id}] wrote shard {shard_idx} horizon={horizon} ver={local_ver}")
            atomic_torch_save(shard, out_path)
            shard_idx += 1

            if mode.startswith("sync"):
                wait_for_version(paths, min_version=int(local_ver) + 1, poll_s=poll_s, timeout_s=None, stop_file=paths.stop_file)
                if stop_requested(paths):
                    stage(f"[actor{actor_id}] stop requested after sync wait; exiting")
                    break
    else:
        # Vectorized rollout path (similar to v1 but using Agent interface)
        env_fns = []
        for i in range(int(num_envs_per_actor)):
            wid = int(actor_id) * 1000 + int(i)
            env_fns.append(lambda i=i: build_actor_env(cfg, cuda=int(cuda if cuda >= 0 else 0), actor_id=int(actor_id), worker_id=int(wid)))
        vec_env = SerialVecEnv(env_fns) if not vec_env_mode.startswith("sub") else SubprocVecEnv(env_fns)

        obs_list, _info_list = vec_env.reset()
        while True:
            if stop_requested(paths):
                stage(f"[actor{actor_id}] stop requested; exiting")
                break

            # Backpressure
            reserve = max(1, int(num_envs_per_actor))
            while count_inflight(paths, actor_id=str(actor_id)) >= max(1, int(max_inflight) - int(reserve) + 1):
                if stop_requested(paths):
                    stage(f"[actor{actor_id}] stop requested during backpressure; exiting")
                    return
                time.sleep(poll_s)

            # Pull latest weights
            cur_ver = read_int(paths.version_file, default=0)
            if cur_ver > local_ver and os.path.exists(paths.latest_ckpt):
                try:
                    agent.load_checkpoint(paths.latest_ckpt)
                    local_ver = int(cur_ver)
                    stage(f"💚[actor{actor_id}] updated weights ver={local_ver}")
                except Exception as e:
                    stage(f"[actor{actor_id}] weight reload failed: {e}")

            obs_bufs: List[List[torch.Tensor]] = [[] for _ in range(int(num_envs_per_actor))]
            old_logp_bufs: List[List[torch.Tensor]] = [[] for _ in range(int(num_envs_per_actor))]
            rew_bufs: List[List[float]] = [[] for _ in range(int(num_envs_per_actor))]
            done_bufs: List[List[float]] = [[] for _ in range(int(num_envs_per_actor))]
            replay_bufs: List[List[Dict[str, Any]]] = [[] for _ in range(int(num_envs_per_actor))]

            last_next_obs_ts: List[Optional[torch.Tensor]] = [None for _ in range(int(num_envs_per_actor))]
            last_dones: List[float] = [1.0 for _ in range(int(num_envs_per_actor))]

            macro_steps = 0
            while macro_steps < horizon:
                # Decision obs per env
                obs_t_list: List[torch.Tensor] = []
                for o in obs_list:
                    obs_t_list.append(obs_to_tensor(o, device=torch.device("cpu")).squeeze(0).detach().cpu())

                actions0, logps, replays = agent.act_batch(
                    obs_list,
                    eta=ddv2_eta,
                    mode_idx=ddv2_mode_idx,
                    mode_select=ddv2_mode_select,
                )

                macro_rewards = [0.0 for _ in range(int(num_envs_per_actor))]
                macro_done = [False for _ in range(int(num_envs_per_actor))]
                macro_next_obs: List[Any] = list(obs_list)

                for k in range(int(commit_steps)):
                    step_actions: List[Any] = []
                    for i in range(int(num_envs_per_actor)):
                        if macro_done[i]:
                            step_actions.append(None)
                            continue
                        rep = replays[i] if isinstance(replays[i], dict) else {}
                        traj_xyyaw = rep.get("traj_xyyaw", None)
                        if torch.is_tensor(traj_xyyaw) and traj_xyyaw.ndim == 3:
                            traj_xyyaw = traj_xyyaw[0]
                        if torch.is_tensor(traj_xyyaw) and int(k) < int(traj_xyyaw.shape[0]):
                            x = float(traj_xyyaw[k, 0].item())
                            y = float(traj_xyyaw[k, 1].item())
                            yaw = float(traj_xyyaw[k, 2].item())
                            step_actions.append((x, y, yaw, 2))
                        else:
                            step_actions.append(actions0[i])

                    next_obs_list, reward_list, term_list, trunc_list, info_list = vec_env.step(step_actions)
                    for i in range(int(num_envs_per_actor)):
                        if macro_done[i]:
                            continue
                        macro_rewards[i] += (float(gamma) ** float(k)) * float(reward_list[i])
                        done = bool(term_list[i] or trunc_list[i])
                        macro_next_obs[i] = next_obs_list[i]
                        if done:
                            macro_done[i] = True
                            o2, _info2 = vec_env.reset_one(i)
                            next_obs_list[i] = o2
                    obs_list = next_obs_list

                for i in range(int(num_envs_per_actor)):
                    obs_bufs[i].append(obs_t_list[i])
                    old_logp_bufs[i].append(logps[i].detach().cpu().float())
                    replay_bufs[i].append(replays[i])
                    rew_bufs[i].append(float(macro_rewards[i]))
                    done_bufs[i].append(1.0 if macro_done[i] else 0.0)

                    try:
                        last_next_obs_ts[i] = obs_to_tensor(macro_next_obs[i], device=torch.device("cpu")).squeeze(0).detach().cpu()
                    except Exception:
                        last_next_obs_ts[i] = torch.zeros((18, 64, 64), dtype=torch.float32)
                    last_dones[i] = 1.0 if macro_done[i] else 0.0

                macro_steps += 1

            for i in range(int(num_envs_per_actor)):
                try:
                    next_obs_t = last_next_obs_ts[i] if last_next_obs_ts[i] is not None else obs_to_tensor(obs_list[i], device=torch.device("cpu")).squeeze(0).detach().cpu()
                except Exception:
                    next_obs_t = torch.zeros((18, 64, 64), dtype=torch.float32)
                done_last = float(last_dones[i]) if len(done_bufs[i]) > 0 else 1.0
                shard = {
                    "obs": torch.stack(obs_bufs[i], dim=0),
                    "old_logp": torch.stack(old_logp_bufs[i], dim=0).view(-1),
                    "reward": torch.tensor(rew_bufs[i], dtype=torch.float32),
                    "done": torch.tensor(done_bufs[i], dtype=torch.float32),
                    "next_obs": next_obs_t,
                    "done_last": torch.tensor(done_last, dtype=torch.float32),
                    "replay": replay_bufs[i],
                    "meta": {
                        "actor_id": int(actor_id),
                        "env_id": int(i),
                        "horizon": int(horizon),
                        "commit_steps": int(commit_steps),
                        "weights_version": int(local_ver),
                        "time": float(time.time()),
                        "shard_idx": int(shard_idx_per_env[i]),
                    },
                }
                name = f"actor{actor_id}_e{i}_v{local_ver}_t{int(time.time())}_{uuid.uuid4().hex[:8]}.pt"
                out_path = os.path.join(paths.shards_dir, name)
                stage(f"[actor{actor_id}] wrote shard env={i} idx={shard_idx_per_env[i]} horizon={horizon} ver={local_ver}")
                atomic_torch_save(shard, out_path)
                shard_idx_per_env[i] += 1

            if mode.startswith("sync"):
                wait_for_version(paths, min_version=int(local_ver) + 1, poll_s=poll_s, timeout_s=None, stop_file=paths.stop_file)
                if stop_requested(paths):
                    stage(f"[actor{actor_id}] stop requested after sync wait; exiting")
                    break


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


def learner_main(cfg: Dict[str, Any]) -> None:
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}

    algo_key = str(train_cfg.get("algo", "ppo")).strip().lower()
    if algo_key in {"reinforce++", "reinforce_pp"}:
        algo_key = "reinforcepp"

    ddp_cfg = (train_cfg.get("ddp", {}) or {})
    ddp_timeout_s = ddp_cfg.get("timeout_s", None)
    rank, world_size, local_rank = learner_init_dist(timeout_s=(int(ddp_timeout_s) if ddp_timeout_s is not None else None))
    ddp_enabled = world_size > 1

    buffer_root = str(al_cfg.get("buffer_dir", "outputs/actor_learner"))
    paths = BufferPaths(root=buffer_root)
    ensure_buffer_layout(paths)

    mode = str(al_cfg.get("mode", "sync")).strip().lower()
    num_actors = int(al_cfg.get("num_actors", 2))
    horizon = int(al_cfg.get("actor_horizon", 32))
    commit_steps = max(1, int(al_cfg.get("commit_steps", 1)))
    shards_per_update = int(al_cfg.get("shards_per_update", num_actors))
    poll_s = float(al_cfg.get("poll_interval_s", 0.2))

    raw_max_updates = al_cfg.get("max_updates", None)
    if raw_max_updates is None:
        raw_max_updates = train_cfg.get("updates", 0)
    max_updates = int(raw_max_updates or 0)

    # Shared hyperparams
    minibatch_size = int(train_cfg.get("minibatch_size", 16))
    gamma = float(train_cfg.get("gamma", 0.99))
    gamma_eff = float(gamma) ** float(commit_steps)
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    max_grad_norm = float(train_cfg.get("ddv2_max_grad_norm", 0.5))

    # Algorithm-specific
    clip_eps = float(train_cfg.get("clip_eps", 0.2))
    ppo_epochs = int(train_cfg.get("epochs", 2))
    vf_coef = float(train_cfg.get("vf_coef", 0.5))

    rpp_cfg = (train_cfg.get("reinforcepp", {}) or {})
    rpp_epochs = int(rpp_cfg.get("epochs", 1))
    rpp_kl_coef = float(rpp_cfg.get("kl_coef", 0.0))
    rpp_norm_eps = float(rpp_cfg.get("norm_eps", 1e-8))
    rpp_group_baseline = str(rpp_cfg.get("group_baseline", "none")).strip().lower()

    ddv2_eta = float(train_cfg.get("ddv2_eta", 1.0))
    ddv2_mode_idx = int(train_cfg.get("ddv2_mode_idx", -1))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    agent_cfg = cfg.get("agent", {}) or {}
    ckpt_path = agent_cfg.get("ckpt", None)
    if ckpt_path is None:
        raise RuntimeError("agent.ckpt is required")

    agent = DiffusionDriveV2Policy(ckpt_path=str(ckpt_path), device=str(device), rl_lr=float(train_cfg.get("ddv2_lr", 1e-5)))
    if ddp_enabled and torch.cuda.is_available():
        agent.wrap_ddp(device_id=local_rank, process_group=dist.group.WORLD)

    # Algo instance
    if algo_key == "ppo":
        algo = PPO(
            clip_eps=clip_eps, vf_coef=vf_coef, ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size, max_grad_norm=max_grad_norm,
            grad_accum_steps=int(((train_cfg.get("ddp", {}) or {}).get("grad_accum_steps", 1))),
            ddp_enabled=ddp_enabled, world_size=world_size, rank=rank,
            ddp_seed=int(((train_cfg.get("ddp", {}) or {}).get("seed", 0))), update_seed=0,
            ddv2_eta=ddv2_eta, ddv2_mode_idx_default=ddv2_mode_idx,
        )
        # value_net only for PPO
        class ValueNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(18, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
                )
                with torch.no_grad():
                    dummy = torch.zeros(1, 18, 64, 64)
                    n_flat = int(self.conv(dummy).view(1, -1).shape[1])
                self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, 512), nn.ReLU(inplace=True))
                self.v = nn.Linear(512, 1)
            def forward(self, obs_t: torch.Tensor) -> torch.Tensor:
                h = self.fc(self.conv(obs_t))
                return self.v(h).squeeze(-1)
        value_net = ValueNet().to(device)
        value_optim = torch.optim.Adam(value_net.parameters(), lr=float(train_cfg.get("lr_value", 1e-4)))
    else:
        algo = ReinforcePP(
            kl_coef=rpp_kl_coef, epochs=rpp_epochs, minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm, grad_accum_steps=int(((train_cfg.get("ddp", {}) or {}).get("grad_accum_steps", 1))),
            ddp_enabled=ddp_enabled, world_size=world_size, rank=rank,
            ddp_seed=int(((train_cfg.get("ddp", {}) or {}).get("seed", 0))), update_seed=0,
            ddv2_eta=ddv2_eta, ddv2_mode_idx_default=ddv2_mode_idx,
        )
        value_net = None
        value_optim = None

    # Initialize weights version
    if rank == 0:
        v = read_int(paths.version_file, default=0)
        if v <= 0:
            write_int(paths.version_file, 1)
            try:
                agent.save_checkpoint(paths.latest_ckpt)
            except Exception as e:
                stage(f"💜[learner] initial save failed: {e}")
    if ddp_enabled:
        dist.barrier()

    start_version = read_int(paths.version_file, default=0)
    if rank == 0:
        stage(f"💜[learner] algo={algo_key} start weights_version={start_version} max_updates={max_updates if max_updates > 0 else 'inf'}")

    update_idx = 0
    while True:
        if stop_requested(paths):
            if rank == 0:
                stage("💜[learner] stop requested; exiting")
            break

        if max_updates > 0:
            cur_ver = read_int(paths.version_file, default=start_version)
            if int(cur_ver) - int(start_version) >= int(max_updates):
                if rank == 0:
                    try:
                        with open(paths.stop_file, "w", encoding="utf-8") as f:
                            f.write(f"stop: reached max_updates={int(max_updates)} at version={int(cur_ver)}\n")
                    except Exception:
                        pass
                    stage(f"💜[learner] reached max_updates={max_updates} (version {cur_ver} from start {start_version}); stopping")
                if ddp_enabled:
                    dist.barrier()
                break

        # Coordinator picks shards
        selected: List[str] = []
        if rank == 0:
            t_wait0 = time.time()
            while True:
                if stop_requested(paths):
                    selected = []
                    break
                files = list_shards(paths)
                if mode.startswith("sync"):
                    have = set()
                    for fp in files:
                        bn = os.path.basename(fp)
                        for a in range(num_actors):
                            if bn.startswith(f"actor{a}_"):
                                have.add(a)
                    if len(have) >= num_actors:
                        per: Dict[int, str] = {}
                        for fp in files:
                            bn = os.path.basename(fp)
                            for a in range(num_actors):
                                if bn.startswith(f"actor{a}_") and a not in per:
                                    per[a] = fp
                        selected = [per[a] for a in sorted(per.keys())][:num_actors]
                        break
                else:
                    if len(files) >= max(1, int(shards_per_update)):
                        selected = files[: int(shards_per_update)]
                        break
                time.sleep(poll_s)
            wait_shards_s = time.time() - t_wait0
        else:
            wait_shards_s = 0.0

        if ddp_enabled:
            obj_list: List[Any] = [selected]
            dist.broadcast_object_list(obj_list, src=0)
            selected = obj_list[0]
        if len(selected) == 0:
            if ddp_enabled:
                dist.barrier()
            break

        # Load shards
        obs_all: List[torch.Tensor] = []
        old_logp_all: List[torch.Tensor] = []
        adv_all: List[torch.Tensor] = []
        ret_all: List[torch.Tensor] = []
        replay_all: List[Dict[str, Any]] = []

        reward_sum = 0.0
        reward_cnt = 0
        done_sum = 0.0
        done_cnt = 0

        if algo_key == "ppo":
            assert value_net is not None
            value_net.eval()
            with torch.inference_mode():
                for fp in selected:
                    shard = torch.load(fp, map_location="cpu")
                    obs_i = shard["obs"].to(device=device, dtype=torch.float32)
                    old_logp_i = shard["old_logp"].to(device=device, dtype=torch.float32).view(-1)
                    rewards_i = shard["reward"].to(device=device, dtype=torch.float32).view(-1)
                    dones_i = shard["done"].to(device=device, dtype=torch.float32).view(-1)
                    replay_i = list(shard.get("replay", []))

                    reward_sum += float(rewards_i.detach().sum().cpu().item())
                    reward_cnt += int(rewards_i.numel())
                    done_sum += float(dones_i.detach().sum().cpu().item())
                    done_cnt += int(dones_i.numel())

                    done_last = float(shard.get("done_last", float(dones_i[-1].item() if dones_i.numel() else 1.0)))
                    next_obs = shard.get("next_obs", None)
                    next_obs_t = obs_i[-1] if next_obs is None else next_obs.to(device=device, dtype=torch.float32)

                    values_i = value_net(obs_i).detach().view(-1)
                    last_value = torch.tensor(0.0, device=device, dtype=values_i.dtype) if done_last >= 0.5 else value_net(next_obs_t.unsqueeze(0)).detach().view(-1)[0]
                    adv_i, ret_i = compute_gae(rewards=rewards_i, dones=dones_i, values=values_i, last_value=last_value, gamma=float(gamma_eff), gae_lambda=float(gae_lambda))

                    obs_all.append(obs_i)
                    old_logp_all.append(old_logp_i)
                    adv_all.append(adv_i)
                    ret_all.append(ret_i)
                    replay_all.extend(replay_i)
        else:
            with torch.inference_mode():
                for fp in selected:
                    shard = torch.load(fp, map_location="cpu")
                    rewards_i = shard["reward"].to(device=device, dtype=torch.float32).view(-1)
                    dones_i = shard["done"].to(device=device, dtype=torch.float32).view(-1)
                    replay_i = list(shard.get("replay", []))
                    T = int(rewards_i.shape[0])

                    reward_sum += float(rewards_i.detach().sum().cpu().item())
                    reward_cnt += int(rewards_i.numel())
                    done_sum += float(dones_i.detach().sum().cpu().item())
                    done_cnt += int(dones_i.numel())

                    ret_i = compute_returns(rewards=rewards_i, dones=dones_i, gamma=float(gamma_eff))
                    adv_i = ret_i

                    adv_all.append(adv_i)
                    ret_all.append(ret_i)
                    replay_all.extend(replay_i)

        obs_batch = torch.cat(obs_all, dim=0) if len(obs_all) else torch.empty((0, 18, 64, 64), device=device)
        old_logp = torch.cat(old_logp_all, dim=0) if len(old_logp_all) else torch.empty((0,), device=device)
        adv = torch.cat(adv_all, dim=0) if len(adv_all) else torch.empty((0,), device=device)
        ret = torch.cat(ret_all, dim=0) if len(ret_all) else torch.empty((0,), device=device)
        n = int(adv.shape[0])

        if algo_key == "ppo":
            if int(obs_batch.shape[0]) != n:
                raise RuntimeError(f"obs_batch length mismatch: obs={int(obs_batch.shape[0])} adv={n}")
            if int(old_logp.shape[0]) != n:
                raise RuntimeError(f"old_logp length mismatch: old_logp={int(old_logp.shape[0])} adv={n}")
        if len(replay_all) != n:
            raise RuntimeError(f"replay_all length mismatch: len={len(replay_all)} n={n}")

        if algo_key == "ppo":
            adv = ppo_normalize_advantages(adv, ddp_enabled=ddp_enabled, dist_module=dist, device=device)
        else:
            adv = reinforcepp_normalize_advantages(adv, ddp_enabled=ddp_enabled, dist_module=dist, device=device, eps=float(rpp_norm_eps))

        batch = {
            "obs_batch": obs_batch,
            "old_logp": old_logp,
            "adv": adv,
            "ret": ret,
            "replay": replay_all,
            "value_net": value_net,
            "value_optim": value_optim,
        }
        metrics = algo.update(agent=agent, batch=batch, device=device)

        if ddp_enabled:
            dist.barrier()
        if rank == 0:
            # Consume shards
            for fp in selected:
                move_to_consumed(paths, fp)

            # Prune consumed_dir: keep only shards consumed in this update.
            prune_consumed(paths, keep_basenames={os.path.basename(fp) for fp in selected})
            # Save weights and bump version
            cur_v = read_int(paths.version_file, default=1)
            new_v = int(cur_v) + 1
            try:
                agent.save_checkpoint(paths.latest_ckpt)
                write_int(paths.version_file, new_v)
            except Exception as e:
                stage(f"💜[learner] save/bump failed: {e}")
            stage(f"💜[learner] update={update_idx} shards={len(selected)} samples={n} ver={new_v} metrics={metrics}")
        if ddp_enabled:
            dist.barrier()

        update_idx += 1
        if max_updates > 0:
            cur_ver = read_int(paths.version_file, default=start_version)
            if int(cur_ver) - int(start_version) >= int(max_updates):
                if rank == 0:
                    stage(f"💜[learner] max_updates reached (version {cur_ver} from start {start_version}); exiting loop")
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "configs", "ppo_closed_loop.yaml"))
    ap.add_argument("--role", type=str, choices=["actor", "learner"], required=True)
    ap.add_argument("--actor-id", type=int, default=0)
    args = ap.parse_args()

    cfg = load_yaml(str(args.config))
    if str(args.role) == "actor":
        actor_main(cfg, actor_id=int(args.actor_id))
    else:
        learner_main(cfg)


if __name__ == "__main__":
    main()
