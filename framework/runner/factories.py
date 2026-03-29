from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from framework.algorithms.ppo import PPO
from framework.algorithms.reinforcepp import ReinforcePP
from framework.utils.repo_paths import REPO_ROOT, resolve_ego_ads_subdir, resolve_repo_path

'''
强化学习训练框架的“组装工厂”：
Actor(环境采样)
Agent(策略网络)
Algorithm(PPO / Reinforce)
ValueNet(价值网络)
处理多GPU + 分布式 + 多Actor采样。
'''

def _list_int(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: List[int] = []
        for value in values:
            try:
                out.append(int(value))
            except Exception:
                continue
        return out
    if isinstance(values, str):
        out: List[int] = []
        for text in values.split(","):
            text = text.strip()
            if not text:
                continue
            try:
                out.append(int(text))
            except Exception:
                continue
        return out
    return []

#=======================================================
#-----------------+++ Actor +++-------------------------
#=======================================================

def resolve_actor_gpu_ids(al_cfg: Dict[str, Any], *, num_actors: int) -> List[int]:
    n = max(1, int(num_actors))
    explicit = _list_int(al_cfg.get("actor_gpu_ids", None))
    if len(explicit) > 0:
        if len(explicit) >= n:
            return explicit[:n]
        return [int(explicit[i % len(explicit)]) for i in range(n)]

    if not torch.cuda.is_available():
        return [-1 for _ in range(n)]

    visible = list(range(int(torch.cuda.device_count())))
    learner_gpu = int(al_cfg.get("learner_gpu_id", 0))
    actor_per_gpu = max(1, int(al_cfg.get("actor_per_gpu", 1)))
    ordered = [learner_gpu] + [gid for gid in visible if gid != learner_gpu]
    if len(ordered) == 0:
        ordered = [0]

    plan: List[int] = []
    idx = 0
    while len(plan) < n:
        gid = int(ordered[idx % len(ordered)])
        for _ in range(actor_per_gpu):
            if len(plan) >= n:
                break
            plan.append(gid)
        idx += 1
    return plan


def normalize_actor_learner_cfg(cfg: Dict[str, Any]) -> None:
    train_cfg = cfg.get("train", {}) or {}
    al_cfg = train_cfg.get("actor_learner", {}) or {}
    if not isinstance(al_cfg, dict) or len(al_cfg) == 0:
        return

    explicit_ids = _list_int(al_cfg.get("actor_gpu_ids", None))
    actor_gpu_pool = _list_int(al_cfg.get("actor_gpu_pool", None) or al_cfg.get("gpu_ids", None) or al_cfg.get("gpus", None))
    actors_per_gpu = al_cfg.get("actors_per_gpu", None)
    if actors_per_gpu is None:
        actors_per_gpu = al_cfg.get("actor_per_gpu", None)
    actors_per_gpu_i = int(actors_per_gpu) if actors_per_gpu is not None else 0

    if len(explicit_ids) == 0 and len(actor_gpu_pool) > 0 and actors_per_gpu_i > 0:
        plan: List[int] = []
        for gid in actor_gpu_pool:
            for _ in range(int(actors_per_gpu_i)):
                plan.append(int(gid))
        al_cfg["actor_gpu_ids"] = plan
        al_cfg["num_actors"] = int(len(plan))

    auto_inflight = al_cfg.get("auto_max_inflight_per_actor", None)
    if auto_inflight is None:
        auto_inflight = bool(len(actor_gpu_pool) > 0 and actors_per_gpu_i > 0)

    if bool(auto_inflight):
        shards_per_update = int(al_cfg.get("shards_per_update", al_cfg.get("num_actors", 1)))
        num_actors = int(al_cfg.get("num_actors", 0))
        if num_actors <= 0:
            ids = _list_int(al_cfg.get("actor_gpu_ids", None))
            num_actors = int(len(ids)) if len(ids) > 0 else 1
            al_cfg["num_actors"] = int(num_actors)
        required = max(1, int(math.ceil(float(shards_per_update) / float(max(1, int(num_actors))))))
        cur = al_cfg.get("max_inflight_per_actor", None)
        if cur is None or int(cur) < int(required):
            al_cfg["max_inflight_per_actor"] = int(required)

    train_cfg["actor_learner"] = al_cfg
    cfg["train"] = train_cfg


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


def _normalize_policy_execute_mode(mode: Any) -> str:
    text = str(mode if mode is not None else "first_step").strip().lower().replace("-", "_")
    if text in {"", "continuous", "first_step", "step1", "traj_first_step"}:
        return "first_step"
    return "first_step"


def build_actor_env(
    cfg: Dict[str, Any],
    *,
    cuda: int,
    actor_id: int,
    worker_id: Optional[int] = None,
    total_actors: int = 1,
) -> Any:
    from framework.env_wrapper import make_scene_sampling_env

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
    al_cfg = ((cfg.get("train", {}) or {}).get("actor_learner", {}) or {})

    from reconsimulator.envs import nus_config as nus_cfg

    scene_ids = [scene0]
    if use_all_scenes:
        discovered = discover_scene_ids(nus_cfg.BASE_DATA_DIR, require_ckpt=require_ckpt) or [scene0]
        scene_ids = list(discovered)
        if bool(al_cfg.get("scene_shard_by_actor", True)) and int(total_actors) > 1 and len(scene_ids) > 0:
            shard_strategy = str(al_cfg.get("scene_shard_strategy", "round_robin")).strip().lower()
            aid = int(actor_id)
            tacts = max(1, int(total_actors))
            if shard_strategy.startswith("contig"):
                chunk = max(1, len(scene_ids) // tacts)
                start = min(len(scene_ids), aid * chunk)
                end = len(scene_ids) if aid == tacts - 1 else min(len(scene_ids), start + chunk)
                shard_ids = scene_ids[start:end]
            else:
                shard_ids = [sid for i, sid in enumerate(scene_ids) if (i % tacts) == (aid % tacts)]
            if len(shard_ids) == 0:
                shard_ids = [scene_ids[aid % len(scene_ids)]]
            scene_ids = shard_ids

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

#=======================================================
#-----------------+++ Agent +++-------------------------
#=======================================================

'''
不同的自车policy;
checkpoint,设置训练层、lr、执行模式;
'''

def build_agent(cfg: Dict[str, Any], *, device: torch.device) -> Any:
    train_cfg = cfg.get("train", {}) or {}
    agent_cfg = cfg.get("agent", {}) or {}
    agent_type = str(agent_cfg.get("type", "ddv2")).strip().lower().replace("-", "_")
    ckpt_path = agent_cfg.get("ckpt", None)
    if ckpt_path is None:
        raise RuntimeError("agent.ckpt is required")
    ckpt_path = resolve_repo_path(str(ckpt_path))
    policy_execute_mode = _normalize_policy_execute_mode(
        train_cfg.get(
            "policy_execute_mode",
            train_cfg.get("ddv2_execute_mode", "continuous"),
        )
    )

    if agent_type == "sparsedrive":
        from framework.agent.policy_sparsedrive import SparseDrivePolicy

        sparse_root = resolve_ego_ads_subdir("SparseDrive")
        config_path = agent_cfg.get("config", os.path.join(sparse_root, "projects", "configs", "sparsedrive_small_stage2.py"))
        config_path = resolve_repo_path(str(config_path))
        return SparseDrivePolicy(
            config_path=str(config_path),
            ckpt_path=str(ckpt_path),
            device=str(device),
            rl_lr=float(train_cfg.get("policy_lr", train_cfg.get("ddv2_lr", 1e-5))),
            execute_mode=policy_execute_mode,
        )
    if agent_type in {"sparsedrive_v2", "sparsedrivev2", "sdv2"}:
        from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy

        trainable_prefixes = agent_cfg.get("trainable_prefixes", train_cfg.get("trainable_layers", []))
        return SparseDriveV2Policy(
            ckpt_path=str(ckpt_path),
            device=str(device),
            rl_lr=float(train_cfg.get("policy_lr", train_cfg.get("ddv2_lr", 1e-5))),
            execute_mode=policy_execute_mode,
            trainable_prefixes=trainable_prefixes,
        )
    from framework.agent.policy_diffusiondrivev2 import DiffusionDriveV2Policy

    return DiffusionDriveV2Policy(
        ckpt_path=str(ckpt_path),
        device=str(device),
        rl_lr=float(train_cfg.get("ddv2_lr", 1e-5)),
        execute_mode=policy_execute_mode,
    )



class ValueNet(nn.Module):
    def __init__(self) -> None:
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
        return self.v(self.fc(self.conv(obs_t))).squeeze(-1)


def _resolve_policy_lr(train_cfg: Dict[str, Any], agent: Any) -> float:
    optimizer = getattr(agent, "optimizer", None)
    if optimizer is not None and len(getattr(optimizer, "param_groups", [])) > 0:
        try:
            return float(optimizer.param_groups[0].get("lr", 1e-5))
        except Exception:
            pass
    return float(train_cfg.get("policy_lr", train_cfg.get("ddv2_lr", 1e-5)))


def _trainable_parameters(module: Any) -> List[torch.nn.Parameter]:
    if module is None or not hasattr(module, "parameters"):
        return []
    return [param for param in module.parameters() if getattr(param, "requires_grad", False)]


#=======================================================
#-----------------+++ Algorithm +++-------------------------
#=======================================================
'''
PPO // Reinforce++;
optimizer;
value_net(PPO);
'''

def build_algorithm_bundle(
    cfg: Dict[str, Any],
    *,
    agent: Any,
    device: torch.device,
    ddp_enabled: bool,
    world_size: int,
    rank: int,
    process_group: Any | None = None,
) -> tuple[Any, Any | None, Any | None, Dict[str, Any]]:
    train_cfg = cfg.get("train", {}) or {}
    algo_key = str(train_cfg.get("algo", "ppo")).strip().lower()
    if algo_key in {"reinforce++", "reinforce_pp", "reinforce_clip"}:
        algo_key = "reinforcepp"
    if algo_key in {"reinforce_vanilla", "vanilla_reinforce"}:
        algo_key = "reinforce"

    minibatch_size = int(train_cfg.get("minibatch_size", 16))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.5))
    clip_eps = float(train_cfg.get("clip_eps", 0.2))
    ppo_epochs = int(train_cfg.get("epochs", 2))
    vf_coef = float(train_cfg.get("vf_coef", 0.5))
    value_clip_eps = float((train_cfg.get("ppo", {}) or {}).get("value_clip_eps", 0.0))
    eta = float(train_cfg.get("eta", 1.0))
    replay_mode_idx = int(train_cfg.get("mode_idx", -1))
    ddp_seed = int(((train_cfg.get("ddp", {}) or {}).get("seed", 0)))
    grad_accum_steps = int(((train_cfg.get("ddp", {}) or {}).get("grad_accum_steps", 1)))
    rpp_cfg = (train_cfg.get("reinforcepp", {}) or {})
    ppo_cfg = (train_cfg.get("ppo", {}) or {})
    reinforce_cfg = (train_cfg.get("reinforce", {}) or {})
    policy_lr = _resolve_policy_lr(train_cfg, agent)
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    if algo_key in {"ppo", "ppo_kl", "ppo_dual_clip", "ppo_value_clip"}:
        
        policy_params = _trainable_parameters(getattr(agent, "trainable_module", None) or agent)
        if len(policy_params) == 0:
            raise RuntimeError("No trainable policy parameters found for PPO")
        # 1)value net
        value_net = ValueNet().to(device)
        if ddp_enabled and torch.cuda.is_available():
            value_net = DDP(
                value_net,
                device_ids=[int(device.index)] if device.index is not None else None,
                output_device=int(device.index) if device.index is not None else None,
                process_group=process_group,
                find_unused_parameters=False,
            )
        # 2)optimizer 
        value_params = _trainable_parameters(value_net)
        value_lr = float(train_cfg.get("lr_value", 1e-4))
        optimizer = torch.optim.Adam(
            [
                {"params": policy_params, "lr": float(policy_lr), "weight_decay": float(weight_decay)},
                {"params": value_params, "lr": float(value_lr), "weight_decay": 0.0},
            ]
        )
        value_optim = optimizer
        
        algo = PPO(
            optimizer=optimizer,
            value_net=value_net,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
            grad_accum_steps=grad_accum_steps,
            ddp_enabled=ddp_enabled,
            world_size=world_size,
            rank=rank,
            ddp_seed=ddp_seed,
            update_seed=0,
            eta=eta,
            variant=algo_key,
            kl_coef=float(ppo_cfg.get("kl_coef", 0.0)) if algo_key == "ppo_kl" else 0.0,
            dual_clip=(float(ppo_cfg.get("dual_clip", 3.0)) if algo_key == "ppo_dual_clip" else None),
            value_clip_eps=(float(value_clip_eps) if algo_key in {"ppo", "ppo_value_clip"} else 0.0),
        )
        
        
    
    elif algo_key in {"reinforce++", "reinforce", "reinforcepp", "reinforce_vanilla"}:
    # else: algo_key in {"reinforcepp", "reinforce", "reinforce_vanilla", "vanilla_reinforce"}
        policy_params = _trainable_parameters(getattr(agent, "trainable_module", None) or agent)
        if len(policy_params) == 0:
            raise RuntimeError("No trainable policy parameters found for Reinforce")
        optimizer = torch.optim.Adam(
            [{"params": policy_params, "lr": float(policy_lr), "weight_decay": float(weight_decay)}]
        )
        algo = ReinforcePP(
            optimizer=optimizer,
            clip_eps=clip_eps,
            kl_coef=float(reinforce_cfg.get("kl_coef", rpp_cfg.get("kl_coef", 0.0))) if algo_key in {"reinforce_kl", "reinforcepp"} else 0.0,
            epochs=int(reinforce_cfg.get("epochs", rpp_cfg.get("epochs", 1))),
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
            grad_accum_steps=grad_accum_steps,
            ddp_enabled=ddp_enabled,
            world_size=world_size,
            rank=rank,
            ddp_seed=ddp_seed,
            update_seed=0,
            eta=eta,
            variant=algo_key,
        )
        value_net = None
        value_optim = None

    meta = {
        "algo_key": algo_key,
        "eta": eta,
        "mode_idx": replay_mode_idx,
        "clip_eps": clip_eps,
        "minibatch_size": minibatch_size,
        "max_grad_norm": max_grad_norm,
        "rpp_norm_eps": float(rpp_cfg.get("norm_eps", 1e-8)),
        "value_clip_eps": float(value_clip_eps),
    }
    return algo, value_net, value_optim, meta