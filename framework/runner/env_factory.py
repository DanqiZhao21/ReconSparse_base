from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from framework.utils.repo_paths import resolve_hugsim_path, resolve_hugsim_root


@dataclass(frozen=True)
class HUGSIMScenarioSpec:
    official_scene_name: str
    scenario_path: str


def discover_hugsim_scenarios(scenario_dir: str) -> List[HUGSIMScenarioSpec]:
    out: List[HUGSIMScenarioSpec] = []
    root = Path(scenario_dir)
    for path in sorted(root.glob("*.yaml")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        scene_name = payload.get("scene_name")
        if scene_name is None:
            continue
        out.append(HUGSIMScenarioSpec(official_scene_name=str(scene_name), scenario_path=str(path)))
    return out


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
    backend = str(env_cfg.get("backend", "recon")).strip().lower()
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

    scene_ids = [scene0]
    hugsim_scenarios: List[Dict[str, str]] | None = None
    hugsim_kwargs: Dict[str, Any] | None = None
    if backend == "hugsim_ori":
        from framework.env_wrapper.hugsim_scene_index import HUGSIMSceneIndex

        hugsim_cfg = env_cfg.get("hugsim", {}) or {}
        scenario_dir = str(
            resolve_hugsim_path(
                hugsim_cfg.get("scenario_dir", None),
                "configs",
                "scenarios",
                "nuscenes",
            )
        )
        discovered_scenarios = discover_hugsim_scenarios(scenario_dir)
        scene_filter = hugsim_cfg.get("scenes", None)
        if scene_filter:
            allowed = {str(name) for name in scene_filter}
            discovered_scenarios = [
                s
                for s in discovered_scenarios
                if s.official_scene_name in allowed
                or Path(s.scenario_path).stem in allowed
                or Path(s.scenario_path).name in allowed
            ]
        if not discovered_scenarios:
            raise RuntimeError(f"No HUGSIM scenarios discovered under {scenario_dir}")
        hugsim_scenarios = [
            {"official_scene_name": spec.official_scene_name, "scenario_path": spec.scenario_path}
            for spec in discovered_scenarios
        ]
        scene_ids = list(range(len(hugsim_scenarios)))
        scene_index = HUGSIMSceneIndex(
            nuscenes_root=hugsim_cfg.get("nuscenes_root", "assets/nuscenes/v1.0-trainval"),
            frame2token_dir=hugsim_cfg.get("frame2token_dir", "assets/nus/information/frame2token"),
        )
        hugsim_kwargs = {
            "scene_index": scene_index,
            "hugsim_repo": resolve_hugsim_path(hugsim_cfg.get("repo", None)) or resolve_hugsim_root(),
            "base_path": resolve_hugsim_path(
                hugsim_cfg.get("base_path", None),
                "configs",
                "sim",
                "nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml",
            ),
            "camera_path": resolve_hugsim_path(hugsim_cfg.get("camera_path", None), "configs", "sim", "nuscenes_camera.yaml"),
            "kinematic_path": resolve_hugsim_path(hugsim_cfg.get("kinematic_path", None), "configs", "sim", "kinematic.yaml"),
            "output_root": hugsim_cfg.get("output_root", "outputs/hugsim_rl"),
            "recon_data_root": hugsim_cfg.get("recon_data_root", "assets/nus/data"),
            "hugsim_model_base": hugsim_cfg.get("model_base", None),
            "launch_mode": hugsim_cfg.get("launch_mode", "fifo"),
            "pixi_cmd": hugsim_cfg.get("pixi_cmd", "pixi"),
            "fifo_timeout_s": float(hugsim_cfg.get("fifo_timeout_s", 300.0)),
            "fifo_poll_interval_s": float(hugsim_cfg.get("fifo_poll_interval_s", 0.2)),
            "min_gt_route_points": int(hugsim_cfg.get("min_gt_route_points", 2)),
        }
        alignment_cfg = hugsim_cfg.get("alignment", {}) or {}
        if isinstance(alignment_cfg, dict):
            hugsim_kwargs.update(
                {
                    "alignment_enabled": bool(alignment_cfg.get("enabled", True)),
                    "alignment_max_rmse_m": float(alignment_cfg.get("max_rmse_m", 2.0)),
                    "use_recon_cache_objects": bool(alignment_cfg.get("use_recon_cache_objects", True)),
                    "use_hugsim_inserted_objects": bool(alignment_cfg.get("use_hugsim_inserted_objects", True)),
                }
            )
        if hugsim_cfg.get("fifo_runner_path", None) is not None:
            hugsim_kwargs["fifo_runner_path"] = hugsim_cfg.get("fifo_runner_path")
    else:
        from reconsimulator.envs import nus_config as nus_cfg

        if use_all_scenes:
            discovered = discover_scene_ids(nus_cfg.BASE_DATA_DIR, require_ckpt=require_ckpt) or [scene0]
            scene_ids = list(discovered)

    if use_all_scenes:
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
    if hugsim_kwargs is not None:
        hugsim_kwargs["output_namespace"] = f"actor{int(actor_id)}_worker{int(wid)}"

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
        env_backend=backend,
        hugsim_scenarios=hugsim_scenarios,
        hugsim_kwargs=hugsim_kwargs,
    )
