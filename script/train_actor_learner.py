import os
import sys
import time
import uuid
import argparse
import random
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

from reconsimulator.envs.rl_wrapper import RLReconEnv
from reconsimulator.envs.subproc_vec_env import make_scene_sampling_env, SceneSamplingEnv
from rl.policy_diffusiondrivev2 import DiffusionDriveV2Policy
from rl.ppo import _obs_to_tensor as obs_to_tensor
from rl.actor_learner_io import (
    BufferPaths,
    atomic_torch_save,
    ensure_buffer_layout,
    list_shards,
    move_to_consumed,
    read_int,
    write_int,
    wait_for_version,
    count_inflight,
    stop_requested,
)
from reconsimulator.envs import nus_config as nus_cfg
from rl.ppo_ddv2_core import compute_gae, normalize_advantages, ddv2_ppo_update


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


def build_actor_env(cfg: Dict[str, Any], *, cuda: int, actor_id: int) -> SceneSamplingEnv:
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

    return make_scene_sampling_env(
        cuda=int(cuda),
        reward_cfg=reward_cfg,
        debug=bool(env_cfg.get("debug", False)),
        scene_ids=list(scene_ids),
        scene_sampling=str(scene_sampling),
        ddp_seed=int(ddp_seed),
        rank=int(rank),
        worker_id=int(actor_id),
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
    max_inflight = int(al_cfg.get("max_inflight_per_actor", 2))#最多允许几个未被 learner 消费的 shard
    poll_s = float(al_cfg.get("poll_interval_s", 0.2))

    # ---- wandb init (optional) ----
    wb_enabled = _wandb_init_if_enabled(cfg, role="actor", ddp_enabled=False, rank=0, actor_id=int(actor_id))
    actor_global_step = 0
    wandb_log_every_steps = int(al_cfg.get("wandb_log_every_steps", 1))

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

    # Build env
    '''
    Actor使用 SceneSamplingEnv，根据配置采样场景。
    支持：
        固定 / 随机起始帧
        最大步数
        是否只使用有 checkpoint 的场景
    作用：提供与环境交互的接口 env.step(action)。
    '''
    env = build_actor_env(cfg, cuda=int(cuda if cuda >= 0 else 0), actor_id=int(actor_id))
    obs, info = env.reset()

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
    while True:
        if stop_requested(paths):
            stage(f"[actor{actor_id}] stop requested; exiting")
            break

        # Backpressure: avoid unbounded disk growth
        while count_inflight(paths, actor_id=str(actor_id)) >= max_inflight:
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

        # Collect fixed-horizon rollout (continue across episode boundaries)
        obs_buf: List[torch.Tensor] = []
        old_logp_buf: List[torch.Tensor] = []
        rew_buf: List[float] = []
        done_buf: List[float] = []
        replay_buf: List[Dict[str, Any]] = []

        steps = 0
        while steps < horizon:
            obs_t = obs_to_tensor(obs, device=torch.device("cpu"))  # store CPU; learner will move to GPU

            action, logp, replay = agent.sample_ddv2rl_with_replay(
                obs,
                eta=ddv2_eta,
                mode_idx=ddv2_mode_idx,
                mode_select=ddv2_mode_select,
            )

            obs_buf.append(obs_t.squeeze(0).detach().cpu())
            old_logp_buf.append(logp.detach().cpu().float())
            replay_buf.append(replay)

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            rew_buf.append(float(reward))
            done_buf.append(1.0 if done else 0.0)
            steps += 1

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
            actor_global_step += 1

            if done:
                obs, info = env.reset()

        # Bootstrap info for truncated horizon:
        # - next_obs is the observation after the last transition (already in `obs`)
        # - done_last indicates whether the last transition ended an episode
        try:
            next_obs_t = obs_to_tensor(obs, device=torch.device("cpu")).squeeze(0).detach().cpu()
        except Exception:
            # Fallback: use zeros if obs tensorization fails (should be rare)
            next_obs_t = torch.zeros((18, 64, 64), dtype=torch.float32)
        done_last = float(done_buf[-1]) if len(done_buf) > 0 else 1.0

        shard = {
            "obs": torch.stack(obs_buf, dim=0),  # (T,18,64,64)
            "old_logp": torch.stack(old_logp_buf, dim=0).view(-1),
            "reward": torch.tensor(rew_buf, dtype=torch.float32),
            "done": torch.tensor(done_buf, dtype=torch.float32),
            "next_obs": next_obs_t,  # (18,64,64)
            "done_last": torch.tensor(done_last, dtype=torch.float32),
            "replay": replay_buf,
            "meta": {
                "actor_id": int(actor_id),
                "horizon": int(horizon),
                "weights_version": int(local_ver),
                "time": float(time.time()),
                "shard_idx": int(shard_idx),
            },
        }

        name = f"actor{actor_id}_v{local_ver}_t{int(time.time())}_{uuid.uuid4().hex[:8]}.pt"
        out_path = os.path.join(paths.shards_dir, name)
        stage(f"[actor{actor_id}] wrote shard {shard_idx} horizon={horizon} ver={local_ver}")
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

        if mode in {"sync", "mini_sync", "mini-batch", "minisync"}:
            # Wait for learner to advance version before continuing
            wait_for_version(paths, min_version=int(local_ver) + 1, poll_s=poll_s, timeout_s=None, stop_file=paths.stop_file)
            if stop_requested(paths):
                stage(f"[actor{actor_id}] stop requested after sync wait; exiting")
                break


def learner_init_dist() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank


def learner_main(cfg: Dict[str, Any]) -> None:
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}

    rank, world_size, local_rank = learner_init_dist()
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

    # PPO hyperparams PPO的超参数
    clip_eps = float(train_cfg.get("clip_eps", 0.2))
    ppo_epochs = int(train_cfg.get("epochs", 1))
    minibatch_size = int(train_cfg.get("minibatch_size", 16))
    gamma = float(train_cfg.get("gamma", 0.99))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    vf_coef = float(train_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(train_cfg.get("ddv2_max_grad_norm", 0.5))

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

    value_net = ValueNet().to(device)
    if ddp_enabled and torch.cuda.is_available():
        value_net = nn.parallel.DistributedDataParallel(value_net, device_ids=[local_rank], output_device=local_rank)

    value_lr = float(train_cfg.get("lr_value", 1e-4))
    value_optim = torch.optim.Adam(value_net.parameters(), lr=value_lr)

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
            f"💜[learner] start weights_version={start_version} max_updates={max_updates if max_updates > 0 else 'inf'}"
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
        # Compute per-shard GAE with bootstrap to make fixed-horizon math equivalent.
        obs_all: List[torch.Tensor] = []
        old_logp_all: List[torch.Tensor] = []
        adv_all: List[torch.Tensor] = []
        ret_all: List[torch.Tensor] = []
        replay_all: List[Dict[str, Any]] = []

        reward_sum = 0.0
        reward_cnt = 0
        done_sum = 0.0
        done_cnt = 0

        t_load0 = time.time()

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
                    gamma=float(gamma),
                    gae_lambda=float(gae_lambda),
                )

                # Accumulate
                obs_all.append(obs_i)
                old_logp_all.append(old_logp_i)
                adv_all.append(adv_i)
                ret_all.append(ret_i)
                replay_all.extend(replay_i)

        obs_batch = torch.cat(obs_all, dim=0)
        old_logp = torch.cat(old_logp_all, dim=0)
        adv = torch.cat(adv_all, dim=0)
        ret = torch.cat(ret_all, dim=0)

        load_shards_s = time.time() - t_load0

        if len(replay_all) != int(obs_batch.shape[0]):
            raise RuntimeError(f"replay_all length mismatch: len={len(replay_all)} n={int(obs_batch.shape[0])}")

        adv = normalize_advantages(adv, ddp_enabled=ddp_enabled, dist_module=dist, device=device)

        n = int(obs_batch.shape[0])

        grad_accum_steps = int(((train_cfg.get("ddp", {}) or {}).get("grad_accum_steps", 1)))

        reward_mean = float(reward_sum / max(1, reward_cnt))
        done_rate = float(done_sum / max(1, done_cnt))

        ret_mean = float(ret.detach().mean().cpu().item()) if ret.numel() else 0.0
        ret_std = float(ret.detach().std(unbiased=False).cpu().item()) if ret.numel() else 0.0
        adv_std = float(adv.detach().std(unbiased=False).cpu().item()) if adv.numel() else 0.0

        t_opt0 = time.time()

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

        opt_time_s = time.time() - t_opt0

        last_loss_pi = float(res.loss_pi)
        last_loss_v = float(res.loss_v)
        last_approx_kl = float(res.approx_kl)

        if ddp_enabled:
            dist.barrier()

        if rank == 0:
            # Consume shards
            for fp in selected:
                move_to_consumed(paths, fp)

            # Prune consumed_dir: keep only the most recently consumed shards (this update).
            keep = {os.path.basename(fp) for fp in selected}
            try:
                for name in os.listdir(paths.consumed_dir):
                    if name in keep:
                        continue
                    p = os.path.join(paths.consumed_dir, name)
                    try:
                        if os.path.isfile(p) or os.path.islink(p):
                            os.remove(p)
                    except Exception:
                        pass
            except Exception:
                pass

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
                            "ratio_mean": float(getattr(res, "ratio_mean", 0.0)),
                            "adv_mean": float(getattr(res, "adv_mean", 0.0)),
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
