import os
import sys
import time
import uuid
import argparse
import random
import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False

# Project root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.env_wrapper import (
    RLReconEnv,
    SceneSamplingSpec,
    SceneSamplingEnv,
    make_scene_sampling_env,
    SubprocVecEnv,
    SerialVecEnv,
)
from framework.agent.policy_diffusiondrivev2 import DiffusionDriveV2Policy
from framework.utils.obs import obs_to_tensor
from framework.io.actor_learner_io import (
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
from reconsimulator.envs import nus_config as nus_cfg
from framework.algorithms.ppo_ddv2_core import compute_gae, normalize_advantages as ppo_normalize_advantages, ddv2_ppo_update
from framework.algorithms.reinforcepp_core import (
    compute_returns,
    normalize_advantages as reinforcepp_normalize_advantages,
    ddv2_reinforcepp_update,
    _apply_group_mean_baseline_inplace,
)


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


def _wandb_init_if_enabled(
    cfg: Dict[str, Any],
    *,
    role: str,
    ddp_enabled: bool,
    rank: int,
    actor_id: Optional[int] = None,
) -> bool:
    wb_cfg = (cfg.get("train", {}) or {}).get("wandb", {}) or {}
    wb_enabled = bool(wb_cfg.get("enabled", False)) and _WANDB_AVAILABLE
    if not wb_enabled:
        return False

    # In DDP, only rank0 should create a learner run.
    if role == "learner" and ddp_enabled and rank != 0:
        return False

    # For actors, allow disabling actor-side runs (default: enabled when wandb enabled).
    al_cfg = ((cfg.get("train", {}) or {}).get("actor_learner", {}) or {})
    if role == "actor" and not bool(al_cfg.get("wandb_actors", True)):
        return False

    project = str(wb_cfg.get("project", "ReconDreamerRL"))
    base_name = str(wb_cfg.get("run_name", f"actor_learner_{time.strftime('%Y%m%d-%H%M%S')}"))

    # Use group to associate learner + actors under one experiment.
    group = str(wb_cfg.get("group", base_name))

    if role == "actor":
        run_name = f"{base_name}_actor{int(actor_id or 0)}"
    else:
        run_name = f"{base_name}_learner"

    try:
        wandb.init(project=project, name=run_name, group=group, config=cfg)
    except Exception:
        return False

    try:
        wandb.define_metric("update")
        wandb.define_metric("global_step")
        if role == "learner":
            wandb.define_metric("loss_pi", summary="last")
            wandb.define_metric("loss_v", summary="last")
            wandb.define_metric("approx_kl", summary="last")
            wandb.define_metric("ratio_mean", summary="last")
            wandb.define_metric("adv_mean", summary="last")
            wandb.define_metric("collect_time_s", summary="last")
            wandb.define_metric("opt_time_s", summary="last")
            wandb.define_metric("update_time_s", summary="last")
            wandb.define_metric("reward_mean", summary="last")
            wandb.define_metric("reward_sum", summary="last")
            wandb.define_metric("done_rate", summary="last")
            wandb.define_metric("ret_mean", summary="last")
            wandb.define_metric("ret_std", summary="last")
            wandb.define_metric("adv_std", summary="last")
            wandb.define_metric("samples", summary="last")
            wandb.define_metric("shards", summary="last")
            wandb.define_metric("weights_version", summary="last")
        else:
            wandb.define_metric("actor/global_step")
            wandb.define_metric("actor/shard_idx", summary="last")
            wandb.define_metric("actor/weights_version", summary="last")
            wandb.define_metric("actor/inflight", summary="last")
    except Exception:
        pass

    return True


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 18, 64, 64)
            n_flat = int(self.conv(dummy).view(1, -1).shape[1])
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, 512), nn.ReLU(inplace=True))
        self.v = nn.Linear(512, 1)

    def forward(self, obs_t: torch.Tensor) -> torch.Tensor:
        h = self.fc(self.conv(obs_t))
        return self.v(h).squeeze(-1)


def discover_scene_ids(base_dir: str, *, require_ckpt: bool) -> List[int]:
    ids: List[int] = []
    try:
        for name in os.listdir(base_dir):
            if not name.isdigit():
                continue
            name3 = f"{int(name):03d}"
            cam0 = os.path.join(base_dir, name3, "cam2ego", "0.txt")
            ego0 = os.path.join(base_dir, name3, "ego_pose", "000.txt")
            ckpt = os.path.join(base_dir, name3, "3DGS_without_prior", "checkpoint_final.pth")
            if os.path.exists(cam0) and os.path.exists(ego0):
                if (not require_ckpt) or os.path.exists(ckpt):
                    ids.append(int(name))
    except Exception:
        pass
    ids.sort()
    return ids


def build_actor_env(cfg: Dict[str, Any], *, cuda: int, actor_id: int, worker_id: Optional[int] = None) -> SceneSamplingEnv:
    env_cfg = cfg.get("env", {}) or {}
    reward_cfg = env_cfg.get("reward", {}) or {}

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

    scene_ids = [scene0]
    if use_all_scenes:
        scene_ids = discover_scene_ids(nus_cfg.BASE_DATA_DIR, require_ckpt=require_ckpt) or [scene0]

    # DDP seed is used only for deterministic sampling here.
    ddp_seed = int(((cfg.get("train", {}) or {}).get("ddp", {}) or {}).get("seed", 0))
    rank = int(os.environ.get("RANK", "0"))

    wid = int(worker_id) if worker_id is not None else int(actor_id)

    return make_scene_sampling_env(
        cuda=int(cuda),
        reward_cfg=reward_cfg,
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
    # Commit steps per sampled trajectory: sample once, execute first K steps without re-sampling.
    commit_steps = max(1, int(al_cfg.get("commit_steps", 1)))
    max_inflight = int(al_cfg.get("max_inflight_per_actor", 2))#最多允许几个未被 learner 消费的 shard
    poll_s = float(al_cfg.get("poll_interval_s", 0.2))

    # Vectorized rollout within one actor process to improve GPU utilization.
    # - num_envs_per_actor: how many env instances to run on this GPU
    # - vec_env_mode: serial (default; safe with CUDA) or subproc (higher isolation; may be fragile)
    num_envs_per_actor = int(al_cfg.get("num_envs_per_actor", 1))
    vec_env_mode = str(al_cfg.get("vec_env_mode", "serial")).strip().lower()
    num_envs_per_actor = max(1, int(num_envs_per_actor))

    # ---- wandb init (optional) ----
    wb_enabled = _wandb_init_if_enabled(cfg, role="actor", ddp_enabled=False, rank=0, actor_id=int(actor_id))
    actor_global_step = 0
    wandb_log_every_steps = int(al_cfg.get("wandb_log_every_steps", 1))

    # Used for aggregating per-micro-step rewards into a macro-step return.
    gamma = float(train_cfg.get("gamma", 0.99))

    # Actor uses its local visible GPU as cuda:0.
    cuda = 0 if torch.cuda.is_available() else -1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Seeds
    base_seed = int(train_cfg.get("seed", 0))
    seed = int(base_seed) + 10000 + int(actor_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build env(s)
    # Actor can run multiple env instances on one GPU and batch DDV2 inference.
    if num_envs_per_actor == 1:
        env = build_actor_env(cfg, cuda=int(cuda if cuda >= 0 else 0), actor_id=int(actor_id))
        obs, info = env.reset()
        vec_env = None
        obs_list: List[Dict[str, Any]] = []
    else:
        def _make_env_fn(eidx: int):
            # Ensure distinct per-env sampling streams.
            wid = int(actor_id) * 1000 + int(eidx)
            return lambda: build_actor_env(
                cfg,
                cuda=int(cuda if cuda >= 0 else 0),
                actor_id=int(actor_id),
                worker_id=int(wid),
            )

        env_fns = [_make_env_fn(i) for i in range(int(num_envs_per_actor))]
        if vec_env_mode.startswith("sub"):
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = SerialVecEnv(env_fns)

        obs_list, _info_list = vec_env.reset()
        # for compatibility with existing codepaths
        obs = obs_list[0]
        info = _info_list[0] if len(_info_list) else {}

    # Build policy (eval only)
    agent_cfg = cfg.get("agent", {}) or {}
    ckpt_path = agent_cfg.get("ckpt", None)
    if ckpt_path is None:
        raise RuntimeError("agent.ckpt is required")

    # Anchor sizes
    try:
        x_anchor = int(getattr(env.env.env, "x_anchor", 61))
        y_anchor = int(getattr(env.env.env, "y_anchor", 61))
    except Exception:
        x_anchor, y_anchor = 61, 61

    agent = DiffusionDriveV2Policy(
        x_anchor=x_anchor,
        y_anchor=y_anchor,
        ckpt_path=str(ckpt_path),
        device=str(device),
        rl_lr=float(train_cfg.get("ddv2_lr", 1e-5)),
        reinforce_baseline_beta=float(train_cfg.get("ddv2_baseline_beta", 0.98)),
    )

    # Warm-load weights from learner if available
    local_ver = 0
    v0 = read_int(paths.version_file, default=0)
    if v0 > 0 and os.path.exists(paths.latest_ckpt):
        try:
            agent.load_from_checkpoint(paths.latest_ckpt)
            local_ver = int(v0)
            stage(f"💚[actor{actor_id}] loaded learner weights ver={local_ver}")
        except Exception as e:
            stage(f"[actor{actor_id}] failed to load learner weights: {e}")

    ddv2_eta = float(train_cfg.get("ddv2_eta", 1.0))
    ddv2_mode_idx = int(train_cfg.get("ddv2_mode_idx", -1))
    ddv2_mode_select = str(train_cfg.get("ddv2_mode_select", "sample")).strip().lower()

    shard_idx = 0
    shard_idx_per_env = [0 for _ in range(int(num_envs_per_actor))]
    while True:
        if stop_requested(paths):
            stage(f"[actor{actor_id}] stop requested; exiting")
            break

        # Backpressure: avoid unbounded disk growth
        reserve = max(1, int(num_envs_per_actor))
        while count_inflight(paths, actor_id=str(actor_id)) >= max(1, int(max_inflight) - int(reserve) + 1):
            if stop_requested(paths):
                stage(f"[actor{actor_id}] stop requested during backpressure; exiting")
                return
            time.sleep(poll_s)

        # Pull latest weights if updated
        cur_ver = read_int(paths.version_file, default=0)
        if cur_ver > local_ver and os.path.exists(paths.latest_ckpt):
            try:
                agent.load_from_checkpoint(paths.latest_ckpt)
                local_ver = int(cur_ver)
                stage(f"💚[actor{actor_id}] updated weights ver={local_ver}")
            except Exception as e:
                stage(f"[actor{actor_id}] weight reload failed: {e}")

        if num_envs_per_actor == 1:
            # ---------------- Single-env rollout (legacy path) ----------------
            obs_buf: List[torch.Tensor] = []
            old_logp_buf: List[torch.Tensor] = []
            rew_buf: List[float] = []
            done_buf: List[float] = []
            replay_buf: List[Dict[str, Any]] = []

            last_next_obs_t: Optional[torch.Tensor] = None
            last_done: float = 1.0

            macro_steps = 0
            while macro_steps < horizon:
                # Decision state (store once per sampled trajectory)
                obs_decision = obs
                obs_t = obs_to_tensor(obs_decision, device=torch.device("cpu")).squeeze(0).detach().cpu()

                action0, logp, replay = agent.sample_ddv2rl_with_replay(
                    obs_decision,
                    eta=ddv2_eta,
                    mode_idx=ddv2_mode_idx,
                    mode_select=ddv2_mode_select,
                )

                # Execute the first K points from the sampled trajectory without re-sampling.
                traj_xyyaw = replay.get("traj_xyyaw", None)
                if torch.is_tensor(traj_xyyaw):
                    # Accept (H,3) or (1,H,3)
                    if traj_xyyaw.ndim == 3:
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
                        # Fallback: repeat the immediate action if trajectory points are missing.
                        action = action0

                    obs, reward, terminated, truncated, info = env.step(action)
                    next_obs_after = obs
                    macro_reward += (float(gamma) ** float(k)) * float(reward)
                    actor_global_step += 1

                    done = bool(terminated or truncated)
                    if done:
                        break

                # Record macro transition
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

                if wb_enabled:
                    try:
                        if (int(actor_global_step) % max(1, int(wandb_log_every_steps))) == 0:
                            inflight = count_inflight(paths, actor_id=str(actor_id))
                            wandb.log(
                                {
                                    "actor/global_step": int(actor_global_step),
                                    "actor/weights_version": int(local_ver),
                                    "actor/inflight": int(inflight),
                                }
                            )
                    except Exception:
                        pass

                if done:
                    obs, info = env.reset()

                if wb_enabled:
                    try:
                        if (int(actor_global_step) % max(1, int(wandb_log_every_steps))) == 0:
                            inflight = count_inflight(paths, actor_id=str(actor_id))
                            wandb.log(
                                {
                                    "actor/global_step": int(actor_global_step),
                                    "actor/weights_version": int(local_ver),
                                    "actor/inflight": int(inflight),
                                }
                            )
                    except Exception:
                        pass
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
            stage(f"[actor{actor_id}] SingelEnv wrote shard {shard_idx} horizon={horizon} ver={local_ver}")
            atomic_torch_save(shard, out_path)

            if wb_enabled:
                try:
                    inflight = count_inflight(paths, actor_id=str(actor_id))
                    rew_t = shard["reward"].float()
                    done_t = shard["done"].float()
                    wandb.log(
                        {
                            "actor/shard_idx": int(shard_idx),
                            "actor/weights_version": int(local_ver),
                            "actor/inflight": int(inflight),
                            "actor/shard_reward_sum": float(rew_t.sum().item()),
                            "actor/shard_reward_mean": float(rew_t.mean().item()) if rew_t.numel() else 0.0,
                            "actor/shard_done_rate": float(done_t.mean().item()) if done_t.numel() else 0.0,
                        }
                    )
                except Exception:
                    pass
            shard_idx += 1
        else:
            # ---------------- Multi-env rollout + batched DDV2 inference ----------------
            assert vec_env is not None

            # Per-env buffers
            obs_bufs: List[List[torch.Tensor]] = [[] for _ in range(int(num_envs_per_actor))]
            old_logp_bufs: List[List[torch.Tensor]] = [[] for _ in range(int(num_envs_per_actor))]
            rew_bufs: List[List[float]] = [[] for _ in range(int(num_envs_per_actor))]
            done_bufs: List[List[float]] = [[] for _ in range(int(num_envs_per_actor))]
            replay_bufs: List[List[Dict[str, Any]]] = [[] for _ in range(int(num_envs_per_actor))]
            scene_per_env: List[Optional[int]] = [None for _ in range(int(num_envs_per_actor))]
            mode_idx_per_env: List[Optional[int]] = [None for _ in range(int(num_envs_per_actor))]

            last_next_obs_ts: List[Optional[torch.Tensor]] = [None for _ in range(int(num_envs_per_actor))]
            last_dones: List[float] = [1.0 for _ in range(int(num_envs_per_actor))]

            macro_steps = 0
            while macro_steps < horizon:
                # Decision obs per env (store once per sampled trajectory)
                obs_t_list: List[torch.Tensor] = []
                for o in obs_list:
                    ot = obs_to_tensor(o, device=torch.device("cpu")).squeeze(0).detach().cpu()
                    obs_t_list.append(ot)

                actions0, logps, replays = agent.sample_ddv2rl_with_replay_batch(
                    obs_list,
                    eta=ddv2_eta,
                    mode_idx=ddv2_mode_idx,
                    mode_select=ddv2_mode_select,
                )

                # Track per-env macro reward/done and terminal next_obs
                macro_rewards = [0.0 for _ in range(int(num_envs_per_actor))]
                macro_done = [False for _ in range(int(num_envs_per_actor))]
                macro_next_obs: List[Any] = list(obs_list)

                # Execute commit_steps micro steps; done envs idle afterwards.
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

                    # Update per-env state; reset+idle done envs
                    for i in range(int(num_envs_per_actor)):
                        if macro_done[i]:
                            continue
                        macro_rewards[i] += (float(gamma) ** float(k)) * float(reward_list[i])
                        done = bool(term_list[i] or trunc_list[i])
                        macro_next_obs[i] = next_obs_list[i]

                        try:
                            if isinstance(info_list[i], dict) and info_list[i].get("scene") is not None:
                                scene_per_env[i] = int(info_list[i].get("scene"))
                        except Exception:
                            pass
                        try:
                            if isinstance(replays[i], dict) and replays[i].get("mode_idx") is not None:
                                mode_idx_per_env[i] = int(replays[i].get("mode_idx"))
                        except Exception:
                            pass

                        if done:
                            macro_done[i] = True
                            # Prepare env for the next macro decision
                            o2, _info2 = vec_env.reset_one(i)
                            next_obs_list[i] = o2

                    obs_list = next_obs_list
                    obs = obs_list[0]
                    actor_global_step += int(num_envs_per_actor)

                # Record macro transitions (one per env)
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

                if wb_enabled:
                    try:
                        if (int(actor_global_step) % max(1, int(wandb_log_every_steps))) == 0:
                            inflight = count_inflight(paths, actor_id=str(actor_id))
                            wandb.log(
                                {
                                    "actor/global_step": int(actor_global_step),
                                    "actor/weights_version": int(local_ver),
                                    "actor/inflight": int(inflight),
                                }
                            )
                    except Exception:
                        pass

            # Write one shard per env instance
            for i in range(int(num_envs_per_actor)):
                next_obs_t = last_next_obs_ts[i]
                if next_obs_t is None:
                    try:
                        next_obs_t = obs_to_tensor(obs_list[i], device=torch.device("cpu")).squeeze(0).detach().cpu()
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
                        "scene": (int(scene_per_env[i]) if scene_per_env[i] is not None else None),
                        "mode_idx": (int(mode_idx_per_env[i]) if mode_idx_per_env[i] is not None else None),
                    },
                }

                name = f"actor{actor_id}_e{i}_v{local_ver}_t{int(time.time())}_{uuid.uuid4().hex[:8]}.pt"
                out_path = os.path.join(paths.shards_dir, name)
                scene_str = str(scene_per_env[i]) if scene_per_env[i] is not None else "?"
                mode_str = str(mode_idx_per_env[i]) if mode_idx_per_env[i] is not None else "?"
                stage(
                    f"[actor{actor_id}] MultiEnv wrote shard env={i} scene={scene_str} mode={mode_str} "
                    f"idx={shard_idx_per_env[i]} horizon={horizon} ver={local_ver}"
                )
                atomic_torch_save(shard, out_path)

                shard_idx_per_env[i] += 1
                shard_idx += 1

        if mode in {"sync", "mini_sync", "mini-batch", "minisync"}:
            # Wait for learner to advance version before continuing
            wait_for_version(paths, min_version=int(local_ver) + 1, poll_s=poll_s, timeout_s=None, stop_file=paths.stop_file)
            if stop_requested(paths):
                stage(f"[actor{actor_id}] stop requested after sync wait; exiting")
                break


def learner_init_dist(*, timeout_s: Optional[int] = None) -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # NOTE: rank0 may wait a long time for enough shards to arrive before
        # broadcasting the selected shard list. If timeout is too small, rank1+
        # can hit NCCL collective timeouts while blocked on that broadcast.
        if timeout_s is None:
            # Default to a conservative 2h.
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

    algo = str(train_cfg.get("algo", "ppo")).strip().lower()  # ppo | reinforcepp
    if algo not in {"ppo", "reinforcepp", "reinforce++", "reinforce_pp"}:
        raise ValueError(f"Unknown train.algo={algo!r} (expected ppo|reinforcepp)")
    if algo in {"reinforce++", "reinforce_pp"}:
        algo = "reinforcepp"

    ddp_cfg = (train_cfg.get("ddp", {}) or {})
    ddp_timeout_s = ddp_cfg.get("timeout_s", None)
    rank, world_size, local_rank = learner_init_dist(timeout_s=(int(ddp_timeout_s) if ddp_timeout_s is not None else None))
    ddp_enabled = world_size > 1

    # ---- wandb init (optional; only learner rank0 in DDP) ----
    wb_enabled = _wandb_init_if_enabled(cfg, role="learner", ddp_enabled=ddp_enabled, rank=int(rank))
    learner_samples_seen = 0

    buffer_root = str(al_cfg.get("buffer_dir", "outputs/actor_learner"))
    paths = BufferPaths(root=buffer_root)
    ensure_buffer_layout(paths)

    mode = str(al_cfg.get("mode", "sync")).strip().lower()
    num_actors = int(al_cfg.get("num_actors", 2))
    horizon = int(al_cfg.get("actor_horizon", 32))
    commit_steps = max(1, int(al_cfg.get("commit_steps", 1)))
    shards_per_update = int(al_cfg.get("shards_per_update", num_actors))
    poll_s = float(al_cfg.get("poll_interval_s", 0.2))

    # Termination control: number of learner updates (weights version bumps) to run.
    # - train.actor_learner.max_updates takes precedence
    # - otherwise fallback to train.updates (for consistency with train_closed_loop)
    raw_max_updates = al_cfg.get("max_updates", None)
    if raw_max_updates is None:
        raw_max_updates = train_cfg.get("updates", 0)
    max_updates = int(raw_max_updates or 0)

    # Replay dtype knobs (keep identical semantics to train_closed_loop)
    replay_cfg = (train_cfg.get("replay", {}) or {})
    replay_compute_dtype_cfg = (replay_cfg.get("compute_dtype", {}) or {})

    def _parse_torch_dtype(x: Any) -> torch.dtype:
        s = str(x).strip().lower()
        if s in {"fp16", "float16", "half"}:
            return torch.float16
        if s in {"bf16", "bfloat16"}:
            return torch.bfloat16
        return torch.float32

    replay_compute_camera_dtype = _parse_torch_dtype(replay_compute_dtype_cfg.get("camera_feature", "fp32"))
    replay_compute_chain_dtype = _parse_torch_dtype(replay_compute_dtype_cfg.get("diffusion_chain", "fp32"))

    # Shared hyperparams
    minibatch_size = int(train_cfg.get("minibatch_size", 16))
    gamma = float(train_cfg.get("gamma", 0.99))
    # Macro-step discount between decisions (one decision executes K micro steps).
    gamma_eff = float(gamma) ** float(commit_steps)
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    max_grad_norm = float(train_cfg.get("ddv2_max_grad_norm", 0.5))

    # PPO hyperparams
    clip_eps = float(train_cfg.get("clip_eps", 0.2))
    ppo_epochs = int(train_cfg.get("epochs", 1))
    vf_coef = float(train_cfg.get("vf_coef", 0.5))

    # Reinforce++ hyperparams
    rpp_cfg = (train_cfg.get("reinforcepp", {}) or {})
    rpp_epochs = int(rpp_cfg.get("epochs", 1))
    rpp_kl_coef = float(rpp_cfg.get("kl_coef", 0.0))
    rpp_norm_eps = float(rpp_cfg.get("norm_eps", 1e-8))
    rpp_group_baseline = str(rpp_cfg.get("group_baseline", "none")).strip().lower()  # none | scene

    ddv2_eta = float(train_cfg.get("ddv2_eta", 1.0))
    ddv2_mode_idx = int(train_cfg.get("ddv2_mode_idx", -1))

    # Learner device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Build policy (train)
    agent_cfg = cfg.get("agent", {}) or {}
    ckpt_path = agent_cfg.get("ckpt", None)
    if ckpt_path is None:
        raise RuntimeError("agent.ckpt is required")

    # We do not need env here; use default anchor sizes.
    x_anchor = 61
    y_anchor = 61

    agent = DiffusionDriveV2Policy(
        x_anchor=x_anchor,
        y_anchor=y_anchor,
        ckpt_path=str(ckpt_path),
        device=str(device),
        rl_lr=float(train_cfg.get("ddv2_lr", 1e-5)),
        reinforce_baseline_beta=float(train_cfg.get("ddv2_baseline_beta", 0.98)),
    )

    if ddp_enabled and torch.cuda.is_available():
        agent.wrap_ddp(device_id=local_rank, process_group=dist.group.WORLD)

    value_net: Optional[nn.Module] = None
    value_optim: Optional[torch.optim.Optimizer] = None
    if algo == "ppo":
        value_net = ValueNet().to(device)
        if ddp_enabled and torch.cuda.is_available():
            value_net = nn.parallel.DistributedDataParallel(value_net, device_ids=[local_rank], output_device=local_rank)

        value_lr = float(train_cfg.get("lr_value", 1e-4))
        value_optim = torch.optim.Adam(value_net.parameters(), lr=value_lr)

    # Reference policy for Reinforce++ KL regularization (optional)
    ref_agent = None
    if algo == "reinforcepp" and float(rpp_kl_coef) > 0.0:
        ref_ckpt = str(rpp_cfg.get("ref_ckpt", ckpt_path))
        ref_agent = DiffusionDriveV2Policy(
            x_anchor=x_anchor,
            y_anchor=y_anchor,
            ckpt_path=str(ref_ckpt),
            device=str(device),
            rl_lr=float(train_cfg.get("ddv2_lr", 1e-5)),
            reinforce_baseline_beta=float(train_cfg.get("ddv2_baseline_beta", 0.98)),
        )
        # Ensure reference does not accidentally get updated.
        try:
            for p in ref_agent._agent.parameters():
                p.requires_grad_(False)
        except Exception:
            pass

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
        stage(
            f"💜[learner] algo={algo} start weights_version={start_version} max_updates={max_updates if max_updates > 0 else 'inf'}"
        )

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
                    # Ensure STOP exists so actors exit too.
                    try:
                        with open(paths.stop_file, "w", encoding="utf-8") as f:
                            f.write(f"stop: reached max_updates={int(max_updates)} at version={int(cur_ver)}\n")
                    except Exception:
                        pass
                    stage(
                        f"💜[learner] reached max_updates={max_updates} (version {cur_ver} from start {start_version}); stopping"
                    )
                if ddp_enabled:
                    dist.barrier()
                break

        t_update0 = time.time()
        # Coordinator picks shards; broadcast the file list to other learner ranks.
        selected: List[str] = []
        if rank == 0:
            t_wait0 = time.time()
            pending_shards = 0
            while True:
                if stop_requested(paths):
                    selected = []
                    break
                files = list_shards(paths)
                pending_shards = int(len(files))
                if mode.startswith("sync"):
                    # wait until we have at least one shard from each actor
                    have = set()
                    for fp in files:
                        bn = os.path.basename(fp)
                        for a in range(num_actors):
                            if bn.startswith(f"actor{a}_"):
                                have.add(a)
                    if len(have) >= num_actors:
                        # pick the oldest shard per actor
                        per: Dict[int, str] = {}
                        for fp in files:
                            bn = os.path.basename(fp)
                            for a in range(num_actors):
                                if bn.startswith(f"actor{a}_") and a not in per:
                                    per[a] = fp
                        selected = [per[a] for a in sorted(per.keys())][:num_actors]
                        break
                else:
                    # async / mini_sync: wait for enough shards
                    if len(files) >= max(1, int(shards_per_update)):
                        selected = files[: int(shards_per_update)]
                        break
                time.sleep(poll_s)
            wait_shards_s = time.time() - t_wait0
        else:
            wait_shards_s = 0.0
            pending_shards = 0

        if ddp_enabled:
            obj_list: List[Any] = [selected]
            dist.broadcast_object_list(obj_list, src=0)
            selected = obj_list[0]

        if len(selected) == 0:
            # stop requested while waiting for shards
            if ddp_enabled:
                dist.barrier()
            break

        # Load shards locally (avoid dist object transfer)
        # PPO: compute per-shard GAE with bootstrap.
        # Reinforce++: compute reward-to-go returns and globally normalize advantages.
        obs_all: List[torch.Tensor] = []
        old_logp_all: List[torch.Tensor] = []
        adv_all: List[torch.Tensor] = []
        ret_all: List[torch.Tensor] = []
        replay_all: List[Dict[str, Any]] = []
        group_ids_all: List[Optional[int]] = []

        reward_sum = 0.0
        reward_cnt = 0
        done_sum = 0.0
        done_cnt = 0

        t_load0 = time.time()

        if algo == "ppo":
            assert value_net is not None
            value_net.eval()
            with torch.inference_mode():
                for fp in selected:
                    shard = torch.load(fp, map_location="cpu")
                    obs_i = shard["obs"].to(device=device, dtype=torch.float32)  # (T,18,64,64)
                    old_logp_i = shard["old_logp"].to(device=device, dtype=torch.float32).view(-1)
                    rewards_i = shard["reward"].to(device=device, dtype=torch.float32).view(-1)
                    dones_i = shard["done"].to(device=device, dtype=torch.float32).view(-1)
                    replay_i = list(shard.get("replay", []))

                    # Stats for logging (computed during load; no extra disk IO)
                    reward_sum += float(rewards_i.detach().sum().cpu().item())
                    reward_cnt += int(rewards_i.numel())
                    done_sum += float(dones_i.detach().sum().cpu().item())
                    done_cnt += int(dones_i.numel())

                    # Bootstrap value for last transition
                    done_last = float(shard.get("done_last", float(dones_i[-1].item() if dones_i.numel() else 1.0)))
                    next_obs = shard.get("next_obs", None)
                    if next_obs is None:
                        # best-effort fallback: use the last obs (not ideal but avoids crash)
                        next_obs_t = obs_i[-1]
                    else:
                        next_obs_t = next_obs.to(device=device, dtype=torch.float32)

                    values_i = value_net(obs_i).detach().view(-1)
                    if done_last >= 0.5:
                        last_value = torch.tensor(0.0, device=device, dtype=values_i.dtype)
                    else:
                        last_value = value_net(next_obs_t.unsqueeze(0)).detach().view(-1)[0]

                    adv_i, ret_i = compute_gae(
                        rewards=rewards_i,
                        dones=dones_i,
                        values=values_i,
                        last_value=last_value,
                        gamma=float(gamma_eff),
                        gae_lambda=float(gae_lambda),
                    )

                    # Accumulate
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

                    # Critic-free advantage: use discounted returns (reward-to-go)
                    ret_i = compute_returns(rewards=rewards_i, dones=dones_i, gamma=float(gamma_eff))
                    adv_i = ret_i

                    # Optional group baseline: subtract per-scene mean (adaptation of R++ w/ baseline)
                    gid: Optional[int] = None
                    if rpp_group_baseline in {"scene", "scene_id"}:
                        try:
                            meta = shard.get("meta", {}) or {}
                            if meta.get("scene") is not None:
                                gid = int(meta.get("scene"))
                        except Exception:
                            gid = None
                    group_ids_all.extend([gid for _ in range(T)])

                    adv_all.append(adv_i)
                    ret_all.append(ret_i)
                    replay_all.extend(replay_i)

        # Note:
        # - PPO needs obs_batch + old_logp.
        # - Reinforce++ does NOT use obs_batch; it uses (adv, ret, replay).
        obs_batch = torch.cat(obs_all, dim=0) if len(obs_all) else torch.empty((0, 18, 64, 64), device=device)
        old_logp = torch.cat(old_logp_all, dim=0) if len(old_logp_all) else torch.empty((0,), device=device)
        adv = torch.cat(adv_all, dim=0) if len(adv_all) else torch.empty((0,), device=device)
        ret = torch.cat(ret_all, dim=0) if len(ret_all) else torch.empty((0,), device=device)

        load_shards_s = time.time() - t_load0

        # Consistency checks
        n = int(adv.shape[0])
        if int(ret.shape[0]) != n:
            raise RuntimeError(f"ret length mismatch: ret={int(ret.shape[0])} adv={n}")

        if algo == "ppo":
            if int(obs_batch.shape[0]) != n:
                raise RuntimeError(f"obs_batch length mismatch: obs={int(obs_batch.shape[0])} adv={n}")
            if int(old_logp.shape[0]) != n:
                raise RuntimeError(f"old_logp length mismatch: old_logp={int(old_logp.shape[0])} adv={n}")

        if len(replay_all) != n:
            raise RuntimeError(f"replay_all length mismatch: len={len(replay_all)} n={n}")

        if algo == "ppo":
            adv = ppo_normalize_advantages(adv, ddp_enabled=ddp_enabled, dist_module=dist, device=device)
        else:
            if rpp_group_baseline in {"scene", "scene_id"}:
                _apply_group_mean_baseline_inplace(adv, group_ids_all)
            adv = reinforcepp_normalize_advantages(
                adv,
                ddp_enabled=ddp_enabled,
                dist_module=dist,
                device=device,
                eps=float(rpp_norm_eps),
            )

        # For PPO, n == obs_batch.shape[0]. For Reinforce++, n == adv.shape[0].
        n = int(n)

        grad_accum_steps = int(((train_cfg.get("ddp", {}) or {}).get("grad_accum_steps", 1)))

        reward_mean = float(reward_sum / max(1, reward_cnt))
        done_rate = float(done_sum / max(1, done_cnt))

        ret_mean = float(ret.detach().mean().cpu().item()) if ret.numel() else 0.0
        ret_std = float(ret.detach().std(unbiased=False).cpu().item()) if ret.numel() else 0.0
        adv_std = float(adv.detach().std(unbiased=False).cpu().item()) if adv.numel() else 0.0

        t_opt0 = time.time()

        if algo == "ppo":
            assert value_net is not None and value_optim is not None
            res = ddv2_ppo_update(
                agent=agent,
                value_net=value_net,
                value_optim=value_optim,
                obs_batch=obs_batch,
                old_logp=old_logp,
                adv=adv,
                ret=ret,
                replay=replay_all,
                device=device,
                ddv2_eta=float(ddv2_eta),
                ddv2_mode_idx_default=int(ddv2_mode_idx),
                clip_eps=float(clip_eps),
                vf_coef=float(vf_coef),
                ppo_epochs=int(ppo_epochs),
                minibatch_size=int(minibatch_size),
                max_grad_norm=float(max_grad_norm),
                grad_accum_steps=int(grad_accum_steps),
                ddp_enabled=bool(ddp_enabled),
                world_size=int(world_size),
                rank=int(rank),
                ddp_seed=int(((train_cfg.get("ddp", {}) or {}).get("seed", 0))),
                update_seed=int(update_idx),
                replay_compute_camera_dtype=replay_compute_camera_dtype,
                replay_compute_chain_dtype=replay_compute_chain_dtype,
                # Here all learner ranks see the same shards; partition indices to avoid duplicate compute.
                use_distributed_sampler=bool(ddp_enabled),
            )
            last_loss_pi = float(res.loss_pi)
            last_loss_v = float(res.loss_v)
            last_approx_kl = float(res.approx_kl)
            ratio_mean = float(getattr(res, "ratio_mean", 0.0))
            adv_mean = float(getattr(res, "adv_mean", 0.0))
        else:
            rpp_res = ddv2_reinforcepp_update(
                agent=agent,
                ref_agent=ref_agent,
                adv=adv,
                replay=replay_all,
                device=device,
                ddv2_eta=float(ddv2_eta),
                ddv2_mode_idx_default=int(ddv2_mode_idx),
                kl_coef=float(rpp_kl_coef),
                epochs=int(rpp_epochs),
                minibatch_size=int(minibatch_size),
                max_grad_norm=float(max_grad_norm),
                grad_accum_steps=int(grad_accum_steps),
                ddp_enabled=bool(ddp_enabled),
                world_size=int(world_size),
                rank=int(rank),
                ddp_seed=int(((train_cfg.get("ddp", {}) or {}).get("seed", 0))),
                update_seed=int(update_idx),
                replay_compute_camera_dtype=replay_compute_camera_dtype,
                replay_compute_chain_dtype=replay_compute_chain_dtype,
                use_distributed_sampler=bool(ddp_enabled),
            )
            last_loss_pi = float(rpp_res.loss_pi)
            last_loss_v = 0.0
            last_approx_kl = float(rpp_res.approx_kl)
            ratio_mean = 0.0
            adv_mean = float(rpp_res.adv_mean)

        opt_time_s = time.time() - t_opt0

        if ddp_enabled:
            dist.barrier()

        if rank == 0:
            # Consume shards
            for fp in selected:
                move_to_consumed(paths, fp)

            # Prune consumed_dir: keep only shards consumed in this update.
            prune_consumed(paths, keep_basenames={os.path.basename(fp) for fp in selected})

            # Advance version and save weights
            cur_v = read_int(paths.version_file, default=1)
            new_v = int(cur_v) + 1
            try:
                agent.save_checkpoint(paths.latest_ckpt)
                write_int(paths.version_file, new_v)
            except Exception as e:
                stage(f"💜[learner] save/bump failed: {e}")

            # Terminate after producing enough new versions
            if max_updates > 0 and (int(new_v) - int(start_version)) >= int(max_updates):
                try:
                    with open(paths.stop_file, "w", encoding="utf-8") as f:
                        f.write(f"stop: reached max_updates={int(max_updates)} at version={int(new_v)}\n")
                except Exception:
                    pass

            # Optional checkpoint upload to W&B
            if wb_enabled:
                try:
                    save_cfg = (cfg.get("train", {}) or {}).get("save", {}) or {}
                    if bool(save_cfg.get("upload_to_wandb", False)):
                        wandb.save(paths.latest_ckpt)
                except Exception:
                    pass

            # W&B per-update logging
            if wb_enabled:
                try:
                    learner_samples_seen += int(n)
                    cuda_mem_alloc = 0.0
                    cuda_mem_reserved = 0.0
                    if torch.cuda.is_available():
                        try:
                            cuda_mem_alloc = float(torch.cuda.max_memory_allocated(device=device) / (1024**3))
                            cuda_mem_reserved = float(torch.cuda.max_memory_reserved(device=device) / (1024**3))
                        except Exception:
                            pass
                    wandb.log(
                        {
                            "update": int(update_idx),
                            "global_step": int(learner_samples_seen),
                            "samples": int(n),
                            "shards": int(len(selected)),
                            "pending_shards": int(pending_shards),
                            "weights_version": int(new_v),
                            "loss_pi": float(last_loss_pi),
                            "loss_v": float(last_loss_v),
                            "approx_kl": float(last_approx_kl),
                            "ratio_mean": float(ratio_mean),
                            "adv_mean": float(adv_mean),
                            "reward_sum": float(reward_sum),
                            "reward_mean": float(reward_mean),
                            "done_rate": float(done_rate),
                            "ret_mean": float(ret_mean),
                            "ret_std": float(ret_std),
                            "adv_std": float(adv_std),
                            "wait_shards_s": float(wait_shards_s),
                            "load_shards_s": float(load_shards_s),
                            "opt_time_s": float(opt_time_s),
                            "update_time_s": float(time.time() - t_update0),
                            "cuda/max_mem_alloc_gb": float(cuda_mem_alloc),
                            "cuda/max_mem_reserved_gb": float(cuda_mem_reserved),
                        }
                    )
                except Exception:
                    pass

            stage(
                f"💜[learner] update={update_idx} shards={len(selected)} samples={n} "
                f"loss_pi={last_loss_pi:.4f} loss_v={last_loss_v:.4f} kl={last_approx_kl:.4f} ver={new_v}"
            )

        if ddp_enabled:
            dist.barrier()

        if max_updates > 0:
            cur_ver = read_int(paths.version_file, default=start_version)
            if int(cur_ver) - int(start_version) >= int(max_updates):
                if rank == 0:
                    stage(
                        f"💜[learner] max_updates reached (version {cur_ver} from start {start_version}); exiting loop"
                    )
                break

        update_idx += 1


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
