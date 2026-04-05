from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


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
