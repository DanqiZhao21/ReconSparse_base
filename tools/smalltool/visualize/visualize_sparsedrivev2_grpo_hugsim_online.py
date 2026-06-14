#!/usr/bin/env python3
"""Visualize SparseDriveV2 GRPO rewards on HUGSIM-ORI scenarios.

Example:
CUDA_VISIBLE_DEVICES=1 python tools/smalltool/visualize/visualize_sparsedrivev2_grpo_hugsim_online.py \
  --config script/configs/sparsedrive_v2/202605280811_HUGSM_reinforcepp_closed_loop_Noclose_GRPOCrafty.yaml \
  --ckpt /root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt \
  --scene-name YOUR_HUGSIM_SCENE_NAME \
  --duration-s 8 \
  --num-candidates 8 \
  --top-k 5 \
  --mode-select sample
  
  
  /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes/scene-0062-medium-00.yaml
  /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes/scene-0411-medium-00.yaml
  
CUDA_VISIBLE_DEVICES=2 python tools/smalltool/visualize/visualize_sparsedrivev2_grpo_hugsim_online.py \
  --config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/202606130003_HUGSM_reinforcepp_closed_loop_steppath_hd_collision_only_extreme_GRPOPdm_auxRiskDecel_substeps1.yaml \
  --ckpt /root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/20260610_164918_3_ColisionOnly_NoCraft_ReinforcePP_NoGRPO_ver01_latest.ckpt \
  --scenario-path /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes/scene-0062-medium-00.yaml \
  --max-steps 42 \
  --num-candidates 64 \
  --top-k 3 \
  --mode-select greedy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.runner.env_factory import discover_hugsim_scenarios
from framework.utils.repo_paths import resolve_hugsim_path, resolve_hugsim_root
from framework.utils.repo_paths import resolve_repo_path
from tools.smalltool.visualize.visualize_sparsedrivev2_grpo_craft_online import (
    _extract_plan_and_scores,
    _build_bev_sample_detail,
    _grid_frame,
    _obs_for_policy,
    _overlay_debug_text,
    build_candidate_score_payload,
    extract_candidate_score_details,
    format_pdm_score_percent,
    overlay_top_right_inset,
    render_bev_debug_image,
    write_score_payload,
)

_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "RewardCheckandVideo" / "HUGSIM-GRPO"


@dataclass(frozen=True)
class HUGSIMSelection:
    scene_index: int
    official_scene_name: str
    scenario_path: str


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    video: Path
    frames_dir: Path
    bev_dir: Path
    candidate_grid_dir: Path
    scores_dir: Path
    steps_dir: Path
    manifest: Path
    step_csv: Path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _json_safe(value.detach().cpu().item())
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return _json_safe(arr.item())
        return [_json_safe(v) for v in arr.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _ensure_output_paths(out_dir: str | Path, *, scene_index: int) -> OutputPaths:
    root = Path(out_dir)
    video = root / f"hugsim_scene_{int(scene_index):03d}_grpo.mp4"
    paths = OutputPaths(
        root=root,
        video=video,
        frames_dir=root / "frames",
        bev_dir=root / "bev",
        candidate_grid_dir=root / "candidate_grid",
        scores_dir=root / "scores",
        steps_dir=root / "steps",
        manifest=root / "run_manifest.json",
        step_csv=root / "step_summary.csv",
    )
    for path in [paths.root, paths.frames_dir, paths.bev_dir, paths.candidate_grid_dir, paths.scores_dir, paths.steps_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _sanitize_path_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "hugsim_scene"


def build_timestamped_output_paths(
    *,
    output_root: str | Path,
    official_scene_name: str,
    timestamp: str | None = None,
) -> OutputPaths:
    scene_slug = _sanitize_path_part(str(official_scene_name))
    ts = str(timestamp or datetime.now().strftime("%Y%m%d-%H%M%S"))
    root = Path(output_root) / f"{scene_slug}_{ts}"
    paths = OutputPaths(
        root=root,
        video=root / f"{scene_slug}_grpo.mp4",
        frames_dir=root / "frames",
        bev_dir=root / "bev",
        candidate_grid_dir=root / "candidate_grid",
        scores_dir=root / "scores",
        steps_dir=root / "steps",
        manifest=root / "run_manifest.json",
        step_csv=root / "step_summary.csv",
    )
    for path in [paths.root, paths.frames_dir, paths.bev_dir, paths.candidate_grid_dir, paths.scores_dir, paths.steps_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def select_hugsim_scenario(
    *,
    config: Mapping[str, Any],
    scene: int | None,
    scene_name: str | None,
    scenario_path: str | None = None,
) -> HUGSIMSelection:
    env_cfg = dict(config.get("env", {}) or {})
    hugsim_cfg = dict(env_cfg.get("hugsim", {}) or {})
    scenario_dir = str(
        resolve_hugsim_path(
            hugsim_cfg.get("scenario_dir", None),
            "configs",
            "scenarios",
            "nuscenes",
        )
    )
    scenarios = discover_hugsim_scenarios(scenario_dir)
    if not scenarios:
        raise RuntimeError(f"No HUGSIM scenarios discovered under {scenario_dir}")

    if scenario_path:
        wanted_path = Path(resolve_repo_path(str(scenario_path))).resolve()
        for idx, spec in enumerate(scenarios):
            if Path(spec.scenario_path).resolve() == wanted_path:
                return HUGSIMSelection(
                    scene_index=int(idx),
                    official_scene_name=str(spec.official_scene_name),
                    scenario_path=str(spec.scenario_path),
                )
        try:
            import yaml

            with wanted_path.open("r", encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}
            official = str(payload.get("scene_name", wanted_path.stem))
        except Exception:
            official = wanted_path.stem
        return HUGSIMSelection(
            scene_index=-1,
            official_scene_name=official,
            scenario_path=str(wanted_path),
        )

    if scene_name:
        wanted = str(scene_name)
        for idx, spec in enumerate(scenarios):
            path = Path(spec.scenario_path)
            if wanted in {spec.official_scene_name, path.stem, path.name, spec.scenario_path}:
                return HUGSIMSelection(
                    scene_index=int(idx),
                    official_scene_name=str(spec.official_scene_name),
                    scenario_path=str(spec.scenario_path),
                )
        raise RuntimeError(f"HUGSIM scene name not found: {wanted}")

    idx = int(scene or 0)
    if idx < 0 or idx >= len(scenarios):
        raise RuntimeError(f"HUGSIM scene index out of range: {idx}; discovered={len(scenarios)}")
    spec = scenarios[idx]
    return HUGSIMSelection(
        scene_index=int(idx),
        official_scene_name=str(spec.official_scene_name),
        scenario_path=str(spec.scenario_path),
    )


def build_run_manifest_payload(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    ckpt_path: str | Path,
    out_dir: str | Path,
    scene_index: int,
    official_scene_name: str,
    scenario_path: str,
    cuda: int,
    num_candidates: int,
    candidate_select: str,
    mode_select: str,
) -> dict[str, Any]:
    env_cfg = dict(config.get("env", {}) or {})
    agent_cfg = dict(config.get("agent", {}) or {})
    train_cfg = dict(config.get("train", {}) or {})
    grpo_cfg = dict(train_cfg.get("grpo", {}) or {})
    scorer_cfg = dict(agent_cfg.get("nuscenes_scorer", {}) or {})
    return {
        "config_path": str(config_path),
        "ckpt_path": str(ckpt_path),
        "out_dir": str(out_dir),
        "cuda": int(cuda),
        "env_backend": str(env_cfg.get("backend", "recon")),
        "scene_index": int(scene_index),
        "official_scene_name": str(official_scene_name),
        "scenario_path": str(scenario_path),
        "scorer_backend": str(scorer_cfg.get("backend", "nuscenes_pdm")),
        "scorer_config": _json_safe(scorer_cfg),
        "grpo": {
            "num_candidates": int(num_candidates),
            "candidate_select": str(candidate_select),
            "config_num_candidates": int(grpo_cfg.get("num_candidates", num_candidates)),
            "config_candidate_select": str(grpo_cfg.get("candidate_select", candidate_select)),
        },
        "policy": {
            "mode_select": str(mode_select),
            "execute_mode": str(train_cfg.get("policy_execute_mode", "first_step")),
        },
    }


def extract_reward_terms(reward_info: Mapping[str, Any]) -> dict[str, float | bool | str]:
    interesting = (
        "reward",
        "positive",
        "cost",
        "progress",
        "lateral",
        "yaw",
        "collision",
        "off_road",
        "route",
        "front_obstacle",
        "terminal",
        "done_reason",
        "reward_mode",
        "hugsim_base_reward",
        "xz_err",
        "yaw_err",
    )
    out: dict[str, float | bool | str] = {}
    for key, value in reward_info.items():
        key_text = str(key)
        if not any(part in key_text for part in interesting):
            continue
        if isinstance(value, (str, bool, int, float, np.integer, np.floating, np.bool_)):
            out[key_text] = _json_safe(value)
    return out


def _score_summary(scores: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _polygon_heading_length_width(poly_xy: np.ndarray) -> tuple[float, float, float]:
    poly = np.asarray(poly_xy, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 2 or poly.shape[1] < 2:
        return 0.0, 1.0, 1.0
    if poly.shape[0] >= 4:
        front_mid = 0.5 * (poly[0, :2] + poly[1, :2])
        rear_mid = 0.5 * (poly[2, :2] + poly[3, :2])
        heading = front_mid - rear_mid
        yaw = float(math.atan2(float(heading[1]), float(heading[0]))) if float(np.linalg.norm(heading)) > 1.0e-9 else 0.0
        length = float(max(1.0e-6, np.linalg.norm(front_mid - rear_mid)))
        width = float(max(1.0e-6, np.linalg.norm(poly[0, :2] - poly[1, :2])))
        return yaw, length, width
    delta = poly[1, :2] - poly[0, :2]
    yaw = float(math.atan2(float(delta[1]), float(delta[0]))) if float(np.linalg.norm(delta)) > 1.0e-9 else 0.0
    return yaw, float(max(1.0e-6, np.linalg.norm(delta))), 1.0


def _localize_recon_global_poly(
    poly_xy: Any,
    *,
    ego_xy: np.ndarray,
    world_to_ego_rot: np.ndarray,
) -> np.ndarray | None:
    try:
        poly = np.asarray(poly_xy, dtype=np.float64)
    except Exception:
        return None
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
        return None
    rel = poly[:, :2] - np.asarray(ego_xy, dtype=np.float64).reshape(1, 2)
    return (rel @ np.asarray(world_to_ego_rot, dtype=np.float64).T).astype(np.float32)


def _plan_for_hugsim_object(token: str, index: int, scenario_plan_list: Sequence[Any] | None) -> Any | None:
    plans = list(scenario_plan_list or [])
    if index < len(plans):
        return plans[index]
    match = re.search(r"(\d+)$", str(token))
    if match is None:
        return None
    plan_idx = int(match.group(1))
    return plans[plan_idx] if plan_idx < len(plans) else None


def _constant_planner_future_from_object(
    *,
    raw: Mapping[str, Any],
    local_poly: np.ndarray,
    plan: Any,
    ego_xy: np.ndarray,
    world_to_ego_rot: np.ndarray,
    ego_yaw: float,
    future_horizon: int,
    future_dt_s: float,
) -> tuple[list[list[float]], list[float], list[float]] | None:
    plan_items = list(plan) if isinstance(plan, (list, tuple)) else []
    if len(plan_items) < 7 or str(plan_items[6]) != "ConstantPlanner":
        return None
    try:
        speed = float(plan_items[4])
    except Exception:
        speed = float(raw.get("speed_mps", 0.0))
    yaw_world = float(raw.get("yaw_rad", plan_items[3] if len(plan_items) > 3 else 0.0))
    center_local = np.mean(local_poly[:, :2], axis=0).astype(np.float64)
    center_world = np.asarray(ego_xy, dtype=np.float64).reshape(2) + center_local @ np.asarray(world_to_ego_rot, dtype=np.float64)
    a = float(center_world[0])
    b = float(center_world[1])
    dt = max(1.0e-6, float(future_dt_s))
    future_xy_world: list[list[float]] = []
    future_yaw_local: list[float] = []
    future_mask: list[float] = []
    for _ in range(1, max(1, int(future_horizon))):
        a = a - speed * math.sin(yaw_world) * dt
        b = b + speed * math.cos(yaw_world) * dt
        future_xy_world.append([float(a), float(b)])
        future_yaw_local.append(float(math.atan2(math.sin(yaw_world - float(ego_yaw)), math.cos(yaw_world - float(ego_yaw)))))
        future_mask.append(1.0)
    if not future_xy_world:
        return None
    future_local = (np.asarray(future_xy_world, dtype=np.float64) - ego_xy.reshape(1, 2)) @ world_to_ego_rot.T
    return future_local.astype(np.float32).tolist(), future_yaw_local, future_mask


def build_hugsim_grpo_object_context(
    info: Mapping[str, Any],
    *,
    scenario_plan_list: Sequence[Any] | None = None,
    future_horizon: int = 8,
    future_dt_s: float = 0.5,
) -> dict[str, Any]:
    """Build scorer-local object overrides from HUGSIM aligned Recon-global context."""
    ego_poly = info.get("hugsim_ego_box_recon_global_poly", None)
    try:
        ego_poly_arr = np.asarray(ego_poly, dtype=np.float64)
    except Exception:
        ego_poly_arr = np.zeros((0, 2), dtype=np.float64)
    if ego_poly_arr.ndim != 2 or ego_poly_arr.shape[0] < 4 or ego_poly_arr.shape[1] < 2:
        return {
            "available": False,
            "scene_objects": [],
            "ea_agent_states": [],
            "hugsim_object_count": 0,
            "recon_cache_object_count": 0,
        }

    ego_xy = np.mean(ego_poly_arr[:, :2], axis=0)
    front_mid = 0.5 * (ego_poly_arr[0, :2] + ego_poly_arr[1, :2])
    rear_mid = 0.5 * (ego_poly_arr[2, :2] + ego_poly_arr[3, :2])
    heading = front_mid - rear_mid
    ego_yaw = float(math.atan2(float(heading[1]), float(heading[0]))) if float(np.linalg.norm(heading)) > 1.0e-9 else 0.0
    c, s = float(math.cos(-ego_yaw)), float(math.sin(-ego_yaw))
    world_to_ego_rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)

    def convert_items(items: Any, *, default_source: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, raw in enumerate([] if items is None else items):
            if not isinstance(raw, Mapping):
                continue
            local_poly = _localize_recon_global_poly(
                raw.get("poly", raw.get("corners_xy", None)),
                ego_xy=ego_xy,
                world_to_ego_rot=world_to_ego_rot,
            )
            if local_poly is None:
                continue
            yaw, length, width = _polygon_heading_length_width(local_poly)
            velocity_xy = np.asarray(raw.get("velocity_xy", [0.0, 0.0]), dtype=np.float32).reshape(-1)
            if velocity_xy.size < 2:
                velocity_xy = np.zeros((2,), dtype=np.float32)
            velocity_local = (
                velocity_xy[:2].astype(np.float64).reshape(1, 2) @ world_to_ego_rot.T
            ).reshape(2).astype(np.float32)
            source = str(raw.get("source", default_source))
            token = str(raw.get("token", raw.get("id", f"{source}_{idx}")))
            obj = {
                "source": source,
                "token": token,
                "category": str(raw.get("category", raw.get("name", "vehicle.car"))),
                "center_xy": np.mean(local_poly[:, :2], axis=0).astype(np.float32).tolist(),
                "corners_xy": local_poly[:, :2].astype(np.float32).tolist(),
                "length_m": float(raw.get("length_m", length)),
                "width_m": float(raw.get("width_m", width)),
                "yaw_rad": float(raw.get("yaw_rad", yaw)),
                "velocity_xy": velocity_local.tolist(),
                "speed_mps": float(raw.get("speed_mps", np.linalg.norm(velocity_local))),
            }
            has_explicit_future = False
            for future_key in ("future_xy", "future_centers_xy", "future_trajectory_xy", "future_traj_xy"):
                if future_key not in raw:
                    continue
                try:
                    future_world = np.asarray(raw.get(future_key), dtype=np.float64)
                except Exception:
                    future_world = np.zeros((0, 2), dtype=np.float64)
                if future_world.ndim == 2 and future_world.shape[0] > 0 and future_world.shape[1] >= 2:
                    future_local = (future_world[:, :2] - ego_xy.reshape(1, 2)) @ world_to_ego_rot.T
                    obj["future_xy"] = future_local.astype(np.float32).tolist()
                    has_explicit_future = True
                break
            for yaw_key in ("future_yaw", "future_yaw_rad", "future_headings_rad", "future_heading_rad"):
                if yaw_key in raw:
                    future_yaw_world = np.asarray(raw.get(yaw_key), dtype=np.float32).reshape(-1)
                    future_yaw_local = np.arctan2(
                        np.sin(future_yaw_world - float(ego_yaw)),
                        np.cos(future_yaw_world - float(ego_yaw)),
                    ).astype(np.float32)
                    obj["future_yaw"] = future_yaw_local.tolist()
                    break
            for mask_key in ("future_mask", "future_masks", "future_valid", "future_valid_mask"):
                if mask_key in raw:
                    obj["future_mask"] = np.asarray(raw.get(mask_key), dtype=np.float32).reshape(-1).tolist()
                    break
            if "future_dt_s" in raw:
                obj["future_dt_s"] = float(raw.get("future_dt_s", 0.5))
            elif "future_dt" in raw:
                obj["future_dt_s"] = float(raw.get("future_dt", 0.5))
            elif "dt_s" in raw:
                obj["future_dt_s"] = float(raw.get("dt_s", 0.5))
            if (
                not has_explicit_future
                and source == "hugsim_inserted"
                and scenario_plan_list is not None
            ):
                rollout = _constant_planner_future_from_object(
                    raw=raw,
                    local_poly=local_poly,
                    plan=_plan_for_hugsim_object(token, idx, scenario_plan_list),
                    ego_xy=ego_xy,
                    world_to_ego_rot=world_to_ego_rot,
                    ego_yaw=ego_yaw,
                    future_horizon=future_horizon,
                    future_dt_s=future_dt_s,
                )
                if rollout is not None:
                    obj["future_xy"], obj["future_yaw"], obj["future_mask"] = rollout
                    obj["future_dt_s"] = float(future_dt_s)
            out.append(obj)
        return out

    hugsim_objects = convert_items(info.get("hugsim_obj_boxes_recon_global", []), default_source="hugsim_inserted")
    recon_objects = convert_items(info.get("recon_cache_dynamic_objects", []), default_source="recon_cache")
    scene_objects = [*hugsim_objects, *recon_objects]
    return {
        "available": True,
        "scene_objects": scene_objects,
        "ea_agent_states": [dict(item) for item in scene_objects],
        "hugsim_object_count": len(hugsim_objects),
        "recon_cache_object_count": len(recon_objects),
    }


def build_hugsim_step_payload(
    *,
    step: int,
    scene_index: int,
    official_scene_name: str,
    frame_idx: int,
    sample_token: str,
    selected_mode_index: int | None,
    selected_logp: float,
    closed_loop_reward: float,
    closed_loop_reward_sum: float,
    reward_info: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    grpo_object_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = list(candidate_payload.get("candidates", []) or [])
    scores = [float(item.get("score", 0.0)) for item in candidates]
    object_context = dict(grpo_object_context or {})
    return {
        "step": int(step),
        "scene_index": int(scene_index),
        "official_scene_name": str(official_scene_name),
        "frame_idx": int(frame_idx),
        "sample_token": str(sample_token),
        "selected": {
            "mode_index": None if selected_mode_index is None else int(selected_mode_index),
            "logp": float(selected_logp),
        },
        "grpo_score_summary": _score_summary(scores),
        "closed_loop_reward": {
            "step_reward": float(closed_loop_reward),
            "cumulative_reward": float(closed_loop_reward_sum),
        },
        "reward_terms": extract_reward_terms(reward_info),
        "reward_info": _json_safe(reward_info),
        "grpo_object_context": _json_safe(object_context),
        "candidate_scores": _json_safe(candidate_payload),
    }


def render_candidate_grid_image(
    *,
    sample_detail: Mapping[str, Any],
    traj_xyyaw: np.ndarray,
    scores: Sequence[float],
    top_k: int,
    panel_size: int = 320,
    columns: int = 4,
) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("render_candidate_grid_image requires opencv-python") from exc

    traj = np.asarray(traj_xyyaw, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if traj.ndim != 3 or traj.shape[-1] < 3:
        raise RuntimeError(f"Expected traj_xyyaw shape (candidates,horizon,3), got {tuple(traj.shape)}")
    count = min(int(traj.shape[0]), int(score_arr.shape[0]))
    if count <= 0:
        return np.full((int(panel_size), int(panel_size), 3), 248, dtype=np.uint8)

    cols = max(1, int(columns))
    rows = int(math.ceil(float(count) / float(cols)))
    panel = max(96, int(panel_size))
    canvas = np.full((rows * panel, cols * panel, 3), 245, dtype=np.uint8)
    ranked = np.argsort(-score_arr[:count], kind="stable").astype(np.int64)
    rank_by_idx = {int(idx): int(rank + 1) for rank, idx in enumerate(ranked.tolist())}
    summary = _score_summary(score_arr[:count])

    for idx in range(count):
        row = idx // cols
        col = idx % cols
        detail = dict(sample_detail)
        detail["sample_token"] = f"cand={idx} rank={rank_by_idx.get(idx, -1)} score={format_pdm_score_percent(float(score_arr[idx]))}"
        one_traj = traj[idx : idx + 1]
        one_score = np.asarray([score_arr[idx]], dtype=np.float32)
        tile = render_bev_debug_image(
            sample_detail=detail,
            traj_xyyaw=one_traj,
            scores=one_score,
            top_k=1 if rank_by_idx.get(idx, count + 1) <= int(top_k) else 0,
            width=panel,
            height=panel,
        )
        x0 = col * panel
        y0 = row * panel
        canvas[y0 : y0 + panel, x0 : x0 + panel] = tile
        title = f"#{idx} r{rank_by_idx.get(idx, -1)} s={format_pdm_score_percent(float(score_arr[idx]))}"
        cv2.rectangle(canvas, (x0, y0), (x0 + panel - 1, y0 + 25), (255, 255, 255), -1)
        cv2.putText(canvas, title, (x0 + 7, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    footer = (
        f"GRPO mean={format_pdm_score_percent(summary['mean'])} "
        f"std={format_pdm_score_percent(summary['std'])} "
        f"min={format_pdm_score_percent(summary['min'])} "
        f"max={format_pdm_score_percent(summary['max'])}"
    )
    cv2.rectangle(canvas, (0, max(0, canvas.shape[0] - 30)), (canvas.shape[1] - 1, canvas.shape[0] - 1), (255, 255, 255), -1)
    cv2.putText(canvas, footer, (10, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_step_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(str(key))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in keys})


def _selected_mode_index(replay: Mapping[str, Any], fallback: Any) -> int | None:
    for key in ["mode_idx", "mode_index", "selected_mode_index"]:
        if key in replay:
            try:
                return int(replay[key])
            except Exception:
                pass
    try:
        arr = np.asarray(fallback).reshape(-1)
        if arr.size > 0:
            return int(arr[0])
    except Exception:
        pass
    return None


def _make_hugsim_env(*, config: Mapping[str, Any], selection: HUGSIMSelection, cuda: int):
    from framework.env_wrapper.hugsim_adapter import HUGSIMReconEnv
    from framework.env_wrapper.hugsim_scene_index import HUGSIMSceneIndex

    env_cfg = dict(config.get("env", {}) or {})
    hugsim_cfg = dict(env_cfg.get("hugsim", {}) or {})
    alignment_cfg = dict(hugsim_cfg.get("alignment", {}) or {})
    scene_index = HUGSIMSceneIndex(
        nuscenes_root=hugsim_cfg.get("nuscenes_root", "assets/nuscenes/v1.0-trainval"),
        frame2token_dir=hugsim_cfg.get("frame2token_dir", "assets/nus/information/frame2token"),
    )
    kwargs = {
        "scenario_name": selection.official_scene_name,
        "scenario_path": selection.scenario_path,
        "scene_index": scene_index,
        "reward_cfg": env_cfg.get("reward", {}) or {},
        "output_root": hugsim_cfg.get("output_root", "outputs/hugsim_rl_visualize"),
        "hugsim_repo": resolve_hugsim_path(hugsim_cfg.get("repo", None)) or resolve_hugsim_root(),
        "base_path": resolve_hugsim_path(
            hugsim_cfg.get("base_path", None),
            "configs",
            "sim",
            "nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml",
        ),
        "camera_path": resolve_hugsim_path(hugsim_cfg.get("camera_path", None), "configs", "sim", "nuscenes_camera.yaml"),
        "kinematic_path": resolve_hugsim_path(hugsim_cfg.get("kinematic_path", None), "configs", "sim", "kinematic.yaml"),
        "substeps_per_rl_step": int(hugsim_cfg.get("substeps_per_rl_step", 2)),
        "recon_data_root": hugsim_cfg.get("recon_data_root", "assets/nus/data"),
        "hugsim_model_base": hugsim_cfg.get("model_base", None),
        "launch_mode": hugsim_cfg.get("launch_mode", "direct"),
        "pixi_cmd": hugsim_cfg.get("pixi_cmd", "pixi"),
        "fifo_timeout_s": float(hugsim_cfg.get("fifo_timeout_s", 300.0)),
        "fifo_poll_interval_s": float(hugsim_cfg.get("fifo_poll_interval_s", 0.2)),
        "cuda": int(cuda),
        "min_gt_route_points": int(hugsim_cfg.get("min_gt_route_points", 2)),
        "alignment_enabled": bool(alignment_cfg.get("enabled", True)),
        "alignment_max_rmse_m": float(alignment_cfg.get("max_rmse_m", 2.0)),
        "use_recon_cache_objects": bool(alignment_cfg.get("use_recon_cache_objects", True)),
        "use_hugsim_inserted_objects": bool(alignment_cfg.get("use_hugsim_inserted_objects", True)),
    }
    if hugsim_cfg.get("fifo_runner_path", None) is not None:
        kwargs["fifo_runner_path"] = hugsim_cfg.get("fifo_runner_path")
    return HUGSIMReconEnv(**kwargs)


def _make_policy(*, config: Mapping[str, Any], ckpt_path: str | Path, cuda: int):
    from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy

    agent_cfg = dict(config.get("agent", {}) or {})
    train_cfg = dict(config.get("train", {}) or {})
    scorer_cfg = dict(agent_cfg.get("nuscenes_scorer", {}) or {})
    for key in ("scene_cache_root", "agent_state_cache_root", "ea_project_src", "nuscenes_dataroot", "token2vad_path"):
        if key in scorer_cfg and scorer_cfg[key] is not None:
            scorer_cfg[key] = str(resolve_repo_path(str(scorer_cfg[key])))
    device = f"cuda:{int(cuda)}" if torch.cuda.is_available() else "cpu"
    return SparseDriveV2Policy(
        ckpt_path=str(resolve_repo_path(str(ckpt_path))),
        device=device,
        execute_mode=str(train_cfg.get("policy_execute_mode", "first_step")),
        rl_lr=float(train_cfg.get("policy_lr", 1.0e-5)),
        trainable_prefixes=agent_cfg.get("trainable_prefixes", None),
        frozen_prefixes=agent_cfg.get("frozen_prefixes", None),
        nuscenes_scorer_config=scorer_cfg,
    )


def run_online_hugsim_scene(
    *,
    config_path: str | Path,
    ckpt_path: str | Path,
    out_dir: str | Path | None = None,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    scene: int | None = None,
    scene_name: str | None = None,
    scenario_path: str | None = None,
    timestamp: str | None = None,
    cuda: int = 0,
    fps: float = 2.0,
    duration_s: float | None = None,
    max_steps: int | None = None,
    num_candidates: int | None = None,
    top_k: int = 5,
    bev_size: int = 1600,
    grid_panel_size: int = 360,
    grid_columns: int = 4,
    candidate_select: str | None = None,
    mode_select: str | None = None,
) -> dict[str, Any]:
    resolved_config = resolve_repo_path(str(config_path))
    config = _load_yaml(resolved_config)
    selection = select_hugsim_scenario(
        config=config,
        scene=scene,
        scene_name=scene_name,
        scenario_path=scenario_path,
    )
    if out_dir is None:
        paths = build_timestamped_output_paths(
            output_root=output_root,
            official_scene_name=selection.official_scene_name,
            timestamp=timestamp,
        )
    else:
        paths = _ensure_output_paths(out_dir, scene_index=selection.scene_index)

    train_cfg = dict(config.get("train", {}) or {})
    grpo_cfg = dict(train_cfg.get("grpo", {}) or {})
    n_candidates = int(num_candidates if num_candidates is not None else grpo_cfg.get("num_candidates", 8))
    cand_select = str(candidate_select if candidate_select is not None else grpo_cfg.get("candidate_select", "topk"))
    policy_mode_select = str(mode_select if mode_select is not None else train_cfg.get("policy_mode_select", "sample"))
    try:
        scenario_payload = _load_yaml(selection.scenario_path)
    except Exception:
        scenario_payload = {}
    scenario_plan_list = list(scenario_payload.get("plan_list", []) or [])

    manifest = build_run_manifest_payload(
        config=config,
        config_path=resolved_config,
        ckpt_path=resolve_repo_path(str(ckpt_path)),
        out_dir=paths.root,
        scene_index=selection.scene_index,
        official_scene_name=selection.official_scene_name,
        scenario_path=selection.scenario_path,
        cuda=int(cuda),
        num_candidates=n_candidates,
        candidate_select=cand_select,
        mode_select=policy_mode_select,
    )
    _write_json(paths.manifest, manifest)

    env = _make_hugsim_env(config=config, selection=selection, cuda=int(cuda))
    policy = _make_policy(config=config, ckpt_path=ckpt_path, cuda=int(cuda))

    obs, reset_info = env.reset()
    current_info = dict(reset_info or {})
    step_limit = max_steps
    if step_limit is None and duration_s is not None:
        step_limit = max(1, int(math.ceil(float(duration_s) * float(fps))))
    if step_limit is None:
        step_limit = int((config.get("env", {}) or {}).get("max_steps", 36))

    writer = imageio.get_writer(
        str(paths.video),
        mode="I",
        fps=float(fps),
        macro_block_size=1,
        codec="libx264",
        ffmpeg_log_level="error",
        output_params=["-pix_fmt", "yuv420p"],
    )

    score_paths: list[str] = []
    step_paths: list[str] = []
    candidate_grid_paths: list[str] = []
    csv_rows: list[dict[str, Any]] = []
    reward_sum = 0.0
    done = False
    try:
        for step in range(int(step_limit)):
            if done:
                break
            obs_in = _obs_for_policy(obs, env)
            action, logp, replay = policy.sample_sparsedrivev2_with_replay(
                obs_in,
                mode_idx=-1,
                mode_select=policy_mode_select,
            )
            grpo_object_context = build_hugsim_grpo_object_context(
                current_info,
                scenario_plan_list=scenario_plan_list,
                future_horizon=8,
                future_dt_s=0.5,
            )
            if bool(grpo_object_context.get("available", False)):
                replay["scene_objects_override"] = list(grpo_object_context.get("scene_objects", []))
                replay["ea_agent_states_override"] = list(grpo_object_context.get("ea_agent_states", []))
                replay["ttc_agent_states_override"] = list(grpo_object_context.get("scene_objects", []))
                replay["grpo_object_context"] = {
                    "source": "hugsim_aligned_recon_global",
                    "hugsim_object_count": int(grpo_object_context.get("hugsim_object_count", 0)),
                    "recon_cache_object_count": int(grpo_object_context.get("recon_cache_object_count", 0)),
                    "scene_object_count": len(grpo_object_context.get("scene_objects", []) or []),
                }
            gt_sample_token = current_info.get("grpo_gt_sample_token", current_info.get("recon_cache_sample_token", None))
            if gt_sample_token is not None and str(gt_sample_token):
                replay["gt_sample_token_override"] = str(gt_sample_token)
            gt_frame_idx = current_info.get("grpo_gt_frame_idx", current_info.get("recon_cache_frame_idx", None))
            if gt_frame_idx is not None:
                try:
                    replay["gt_frame_idx_override"] = int(gt_frame_idx)
                except Exception:
                    pass
            replay_out = _extract_plan_and_scores(
                policy,
                replay,
                num_candidates=n_candidates,
                candidate_select=cand_select,
            )
            traj_xyyaw = np.asarray(replay_out["traj_xyyaw"], dtype=np.float32)
            mode_indices = np.asarray(replay_out["mode_indices"], dtype=np.int64)
            score_logits = np.asarray(replay_out["score_logits"], dtype=np.float32)
            score_tensor = policy.pdm_score_counterfactuals_from_replay_batch(
                [replay],
                torch.as_tensor(traj_xyyaw[None, ...], dtype=torch.float32),
            )
            if torch.is_tensor(score_tensor):
                candidate_scores = np.asarray(score_tensor.detach().cpu().numpy(), dtype=np.float32)[0]
            else:
                candidate_scores = np.asarray(score_tensor, dtype=np.float32)[0]

            sample_detail = _build_bev_sample_detail(policy, replay, traj_xyyaw)
            selected_mode = _selected_mode_index(replay, np.asarray([mode_indices[0] if mode_indices.size else -1]))
            selected_candidate_idx = None
            if selected_mode is not None and mode_indices.size:
                matches = np.where(mode_indices == int(selected_mode))[0]
                if int(matches.shape[0]) > 0:
                    selected_candidate_idx = int(matches[0])
            bev_img = render_bev_debug_image(
                sample_detail=sample_detail,
                traj_xyyaw=traj_xyyaw,
                scores=candidate_scores,
                top_k=int(top_k),
                width=int(bev_size),
                height=int(bev_size),
                selected_index=selected_candidate_idx,
                max_display_candidates=max(4, min(10, int(top_k) + 3)),
            )
            imageio.imwrite(paths.bev_dir / f"step_{step:06d}_bev.png", bev_img)
            candidate_grid_img = render_candidate_grid_image(
                sample_detail=sample_detail,
                traj_xyyaw=traj_xyyaw,
                scores=candidate_scores,
                top_k=int(top_k),
                panel_size=int(grid_panel_size),
                columns=int(grid_columns),
            )
            candidate_grid_path = paths.candidate_grid_dir / f"step_{step:06d}_candidates.png"
            imageio.imwrite(candidate_grid_path, candidate_grid_img)
            candidate_grid_paths.append(str(candidate_grid_path))

            frame_idx = int(current_info.get("now_frame", current_info.get("frame_idx", -1)))
            sample_token = str(replay.get("sample_token", current_info.get("sample_token", "")))
            candidate_payload = build_candidate_score_payload(
                scene=int(selection.scene_index),
                step=int(step),
                frame_idx=int(frame_idx),
                sample_token=sample_token,
                traj_xyyaw=traj_xyyaw,
                scores=candidate_scores,
                score_logits=score_logits,
                mode_indices=mode_indices,
                top_k=int(top_k),
                candidate_details=extract_candidate_score_details(
                    policy,
                    replay,
                    traj_xyyaw,
                    candidate_scores,
                    sample_detail=sample_detail,
                ),
            )
            score_path = write_score_payload(paths.scores_dir, candidate_payload)
            score_paths.append(str(score_path))

            selected_plan = np.asarray(replay.get("traj_xyyaw", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
            if selected_plan.ndim == 2 and selected_plan.shape[0] > 0:
                env.set_external_plan_local_xyyaw(selected_plan[:, :3])
            obs_next, reward_v, terminated, truncated, reward_info = env.step(action)
            reward_info = dict(reward_info or {})
            reward_sum += float(reward_v)
            done = bool(terminated or truncated)
            next_frame_idx = int(reward_info.get("now_frame", reward_info.get("frame_idx", frame_idx)))

            logp_float = float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp)
            step_payload = build_hugsim_step_payload(
                step=int(step),
                scene_index=int(selection.scene_index),
                official_scene_name=selection.official_scene_name,
                frame_idx=int(frame_idx),
                sample_token=str(reward_info.get("sample_token", sample_token)),
                selected_mode_index=selected_mode,
                selected_logp=logp_float,
                closed_loop_reward=float(reward_v),
                closed_loop_reward_sum=float(reward_sum),
                reward_info=reward_info,
                candidate_payload=candidate_payload,
                grpo_object_context=grpo_object_context,
            )
            step_path = paths.steps_dir / f"step_{step:06d}.json"
            _write_json(step_path, step_payload)
            step_paths.append(str(step_path))

            summary = step_payload["grpo_score_summary"]
            frame = _grid_frame(obs)
            frame = _overlay_debug_text(
                frame,
                [
                    f"hugsim_scene={selection.scene_index:03d} {selection.official_scene_name}",
                    f"step={step} frame={frame_idx}->{next_frame_idx} reward={float(reward_v):.3f} sum={reward_sum:.3f}",
                    f"grpo mean={summary['mean']:.3f} min={summary['min']:.3f} max={summary['max']:.3f}",
                    f"mode={selected_mode} logp={logp_float:.4f} done={done}",
                ],
            )
            frame = overlay_top_right_inset(frame, bev_img, inset_width=390, margin=12, border_px=2)
            writer.append_data(frame)
            imageio.imwrite(paths.frames_dir / f"step_{step:06d}.png", frame)

            row = {
                "step": int(step),
                "scene_index": int(selection.scene_index),
                "official_scene_name": selection.official_scene_name,
                "frame_idx": int(frame_idx),
                "next_frame_idx": int(next_frame_idx),
                "sample_token": str(reward_info.get("sample_token", sample_token)),
                "reward": float(reward_v),
                "reward_sum": float(reward_sum),
                "grpo_score_mean": float(summary["mean"]),
                "grpo_score_std": float(summary["std"]),
                "grpo_score_min": float(summary["min"]),
                "grpo_score_max": float(summary["max"]),
                "selected_mode_index": "" if selected_mode is None else int(selected_mode),
                "selected_logp": logp_float,
                "done": bool(done),
                "done_reason": reward_info.get("done_reason", ""),
                "reward_mode": reward_info.get("reward_mode", ""),
                "grpo_context_available": bool(grpo_object_context.get("available", False)),
                "grpo_hugsim_object_count": int(grpo_object_context.get("hugsim_object_count", 0)),
                "grpo_recon_cache_object_count": int(grpo_object_context.get("recon_cache_object_count", 0)),
                "grpo_scene_object_count": len(grpo_object_context.get("scene_objects", []) or []),
            }
            for key, value in extract_reward_terms(reward_info).items():
                if key not in row:
                    row[key] = value
            csv_rows.append(row)
            obs = obs_next
            current_info = dict(reward_info)
    finally:
        writer.close()
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    _write_step_csv(paths.step_csv, csv_rows)
    return {
        "video": str(paths.video),
        "manifest": str(paths.manifest),
        "step_csv": str(paths.step_csv),
        "frames_dir": str(paths.frames_dir),
        "bev_dir": str(paths.bev_dir),
        "candidate_grid_dir": str(paths.candidate_grid_dir),
        "scores_dir": str(paths.scores_dir),
        "steps_dir": str(paths.steps_dir),
        "score_json": score_paths,
        "step_json": step_paths,
        "candidate_grid": candidate_grid_paths,
        "steps": len(csv_rows),
        "reward_sum": float(reward_sum),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="SparseDriveV2 HUGSIM GRPO reward visualizer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--out", type=str, default=None, help="Exact output run directory. Overrides timestamped output-root layout.")
    parser.add_argument("--scene-index", type=int, default=None, help="Index in HUGSIM scenario_dir discovery order.")
    parser.add_argument("--scene", type=int, default=None, help="Alias for --scene-index.")
    parser.add_argument("--scene-name", type=str, default=None, help="Official scene_name or scenario yaml stem.")
    parser.add_argument("--scenario-path", type=str, default=None, help="Exact HUGSIM scenario yaml path.")
    parser.add_argument("--timestamp", type=str, default=None, help="Override run timestamp for output folder naming.")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bev-size", type=int, default=1600)
    parser.add_argument("--grid-panel-size", type=int, default=360)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--candidate-select", type=str, default=None, choices=["topk", "all"])
    parser.add_argument("--mode-select", type=str, default=None, choices=["greedy", "sample"])
    args = parser.parse_args()

    result = run_online_hugsim_scene(
        config_path=args.config,
        ckpt_path=args.ckpt,
        out_dir=args.out,
        output_root=args.output_root,
        scene=args.scene_index if args.scene_index is not None else args.scene,
        scene_name=args.scene_name,
        scenario_path=args.scenario_path,
        timestamp=args.timestamp,
        cuda=int(args.cuda),
        fps=float(args.fps),
        duration_s=args.duration_s,
        max_steps=args.max_steps,
        num_candidates=args.num_candidates,
        top_k=int(args.top_k),
        bev_size=int(args.bev_size),
        grid_panel_size=int(args.grid_panel_size),
        grid_columns=int(args.grid_columns),
        candidate_select=args.candidate_select,
        mode_select=args.mode_select,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
