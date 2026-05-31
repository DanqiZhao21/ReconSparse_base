#!/usr/bin/env python3
"""Generate one closed-loop rollout video with SparseDriveV2 in ReconSimulator.

Pipeline per step:
1) Use current observation to run SparseDriveV2 planner.
2) Execute only the first point as env continuous action (flag=2).
3) Re-plan at next step.

Unavailable parts are kept as placeholders (e.g., exact mode log-prob from V2).


cd /root/clone/ReconDreamer-RL

python tools/smalltool/visualize/generate_video_sparsedrive_v2-HUGSIMori.py \
  --scene 0051 \
  --config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/202605251610_HUGSM_reinforcepp_closed_loop_closeCloselopop_openGRPOCraft.yaml \
  --ckpt /root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt
  
python tools/smalltool/visualize/generate_video_sparsedrive_v2-HUGSIMori.py \
  --hugsim-scene scene-0051-hard-00 \
  --config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/202605301356_HUGSM_reinforcepp_closed_loop_reward-close_loop_openGRPOCraft.yaml \
  --ckpt /root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt
  
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from typing import Any, Dict, List

import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import json
from shapely.geometry import Polygon


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _prepend_env_path(env_key: str, values: list[str]) -> None:
    existing = os.environ.get(env_key, "")
    parts: list[str] = []
    seen: set[str] = set()
    for value in values + ([existing] if existing else []):
        if not value:
            continue
        for item in str(value).split(os.pathsep):
            item = item.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            parts.append(item)
    if parts:
        os.environ[env_key] = os.pathsep.join(parts)


def _prepare_cuda_extension_env() -> None:
    cuda_home = os.environ.get("CUDA_HOME", "").strip() or "/usr/local/cuda"
    os.environ["CUDA_HOME"] = cuda_home
    include_dirs = [
        os.path.join(cuda_home, "include"),
        os.path.join(cuda_home, "targets", "x86_64-linux", "include"),
    ]
    library_dirs = [
        os.path.join(cuda_home, "lib64"),
        os.path.join(cuda_home, "targets", "x86_64-linux", "lib"),
    ]
    _prepend_env_path("CPATH", include_dirs)
    _prepend_env_path("CPLUS_INCLUDE_PATH", include_dirs)
    _prepend_env_path("LIBRARY_PATH", library_dirs)
    _prepend_env_path("LD_LIBRARY_PATH", library_dirs)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(_REPO_ROOT, ".cache", "torch_extensions"))
    os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)


_prepare_cuda_extension_env()

def _resolve_ego_ads_subdir(name: str) -> str:
    preferred = os.path.join(_REPO_ROOT, "egoADs", str(name))
    if os.path.isdir(preferred):
        return preferred
    return os.path.join(_REPO_ROOT, str(name))


def _resolve_repo_path(path: str) -> str:
    text = str(path)
    if os.path.isabs(text):
        return text
    direct = os.path.join(_REPO_ROOT, text)
    if os.path.exists(direct):
        return direct
    egoads_candidate = os.path.join(_REPO_ROOT, "egoADs", text)
    if os.path.exists(egoads_candidate):
        return egoads_candidate
    return direct


_DEFAULT_CKPT = os.path.join(_resolve_ego_ads_subdir("SparseDriveV2"), "ckpt", "sparsedrive_navsimv2.ckpt")
_DEFAULT_OUTPUT_ROOT = os.path.join(_REPO_ROOT, "outputs", "RewardCheckandVideo")
_DEFAULT_HUGSIM_OUTPUT_ROOT = os.path.join(_DEFAULT_OUTPUT_ROOT, "HUGSIM")


def _lazy_import_runtime() -> tuple[Any, Any]:
    try:
        from framework.env_wrapper import RLReconEnv  # type: ignore
        from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy  # type: ignore

        return RLReconEnv, SparseDriveV2Policy
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise RuntimeError(
            "Missing runtime dependency for SparseDriveV2 rollout. "
            f"Import failed on module: {missing}. Activate project env and retry."
        ) from e


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _build_auto_run_paths(
    *,
    scene: int,
    timestamp: str,
    output_root: str = _DEFAULT_OUTPUT_ROOT,
) -> Dict[str, str]:
    scene_dir = os.path.join(str(output_root), f"scene{int(scene):03d}-{timestamp}")
    artifacts_dir = os.path.join(scene_dir, "artifacts")
    return {
        "run_dir": scene_dir,
        "artifacts_dir": artifacts_dir,
        "run_manifest": os.path.join(scene_dir, "run_info.md"),
        "video_path": os.path.join(artifacts_dir, f"scene{int(scene):03d}_{timestamp}_sparsedrivev2_rollout.mp4"),
        "traj_csv": os.path.join(artifacts_dir, f"scene{int(scene):03d}_{timestamp}_sparsedrivev2_plan_frontframe.csv"),
        "traj_plot": os.path.join(artifacts_dir, f"scene{int(scene):03d}_{timestamp}_sparsedrivev2_expert_vs_ego_traj.svg"),
    }


def _build_hugsim_shard_run_paths(
    *,
    timestamp: str,
    output_root: str = _DEFAULT_HUGSIM_OUTPUT_ROOT,
    label: str = "hugsim_ori",
) -> Dict[str, str]:
    safe_label = str(label or "hugsim_ori").strip().replace(os.sep, "_").replace(" ", "_")
    scene_dir = os.path.join(str(output_root), f"{safe_label}-{timestamp}")
    artifacts_dir = os.path.join(scene_dir, "artifacts")
    stem = f"{safe_label}_{timestamp}"
    return {
        "run_dir": scene_dir,
        "artifacts_dir": artifacts_dir,
        "run_manifest": os.path.join(scene_dir, "run_info.md"),
        "shard_path": os.path.join(artifacts_dir, f"{stem}_actor_learner_shard.pt"),
        "video_path": os.path.join(artifacts_dir, f"{stem}_hugsim_online.mp4"),
        "traj_csv": os.path.join(artifacts_dir, f"{stem}_shard_plan.csv"),
        "traj_plot": os.path.join(artifacts_dir, f"{stem}_shard_bev.svg"),
    }


def _write_run_manifest(
    *,
    manifest_path: str,
    scene: int,
    config_path: str,
    ckpt_path: str,
    timestamp: str,
    extra_lines: List[str] | None = None,
) -> str:
    _ensure_parent(manifest_path)
    run_dir = os.path.dirname(manifest_path)
    lines = [
        f"# SparseDriveV2 rollout run\n",
        f"- run_dir: `{run_dir}`\n",
        f"- scene: {int(scene):03d}\n",
        f"- timestamp: {timestamp}\n",
        f"- config: `{config_path}`\n",
        f"- ckpt: `{ckpt_path}`\n",
    ]
    for line in extra_lines or []:
        text = str(line).rstrip("\n")
        if not text:
            continue
        if text.startswith("- "):
            lines.append(text + "\n")
        else:
            lines.append(f"- {text}\n")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return manifest_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        arr = value.detach().cpu()
        if arr.ndim == 0:
            return _json_safe(arr.item())
        return arr.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _markdown_table(rows: List[Dict[str, Any]], fieldnames: List[str]) -> str:
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = []
    for row in rows:
        vals = []
        for key in fieldnames:
            val = _json_safe(row.get(key, ""))
            if isinstance(val, (dict, list)):
                text = json.dumps(val, ensure_ascii=False, sort_keys=True)
            else:
                text = str(val)
            vals.append(text.replace("|", "\\|").replace("\n", "<br>"))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def _new_markdown_cell(text: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _new_code_cell(code: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def _save_reward_detail_notebook(
    *,
    out_path: str,
    scene: int,
    reward_rows: List[Dict[str, Any]],
    debug_shard: Dict[str, Any],
    reward_cfg: Dict[str, Any],
    video_path: str,
    reward_csv_path: str,
    debug_shard_path: str,
) -> str:
    _ensure_parent(out_path)
    meta_rows = debug_shard.get("meta", []) if isinstance(debug_shard, dict) else []
    info_by_step: Dict[int, Dict[str, Any]] = {}
    for item in meta_rows if isinstance(meta_rows, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            step = int(item.get("step", len(info_by_step)))
        except Exception:
            continue
        info = item.get("info", {})
        info_by_step[step] = dict(info) if isinstance(info, dict) else {"info": info}

    reward_fields = [
        "step",
        "frame_before",
        "frame_after",
        "reward",
        "cum_reward",
        "progress_reward",
        "cost_reward",
        "done",
        "done_reason",
        "dynamic_collision",
        "static_collision",
        "collision_tokens",
    ]
    reward_fields = [k for k in reward_fields if any(k in row for row in reward_rows)]
    if not reward_fields and reward_rows:
        reward_fields = sorted({str(k) for row in reward_rows for k in row.keys()})

    cfg_preview = json.dumps(_json_safe(reward_cfg), ensure_ascii=False, indent=2, sort_keys=True)
    cells: List[Dict[str, Any]] = [
        _new_markdown_cell(
            f"# Scene {int(scene):03d} Reward Detail\n\n"
            "This notebook is generated by `generate_video_sparsedrive_v2.py` during rollout export.\n\n"
            f"- video: `{video_path}`\n"
            f"- reward_csv: `{reward_csv_path}`\n"
            f"- debug_shard: `{debug_shard_path}`\n"
            f"- steps: `{len(reward_rows)}`\n"
        ),
        _new_markdown_cell("## Reward Config\n\n```json\n" + cfg_preview + "\n```\n"),
        _new_code_cell(
            "import pandas as pd\n"
            "from IPython.display import Video, display\n\n"
            f"reward_csv = {reward_csv_path!r}\n"
            f"video_path = {video_path!r}\n"
            "df = pd.read_csv(reward_csv)\n"
            "display(df)\n"
            "display(Video(video_path, embed=False))\n"
        ),
        _new_markdown_cell("## Per-Step Reward Summary\n\n" + _markdown_table(reward_rows, reward_fields)),
    ]

    for row in reward_rows:
        try:
            step = int(row.get("step", len(cells)))
        except Exception:
            step = len(cells)
        info = info_by_step.get(step, {})
        info_rows = [{"key": str(k), "value": _json_safe(v)} for k, v in sorted(info.items(), key=lambda kv: str(kv[0]))]
        cells.append(
            _new_markdown_cell(
                f"## Step {step}\n\n"
                "### Reward Row\n\n"
                + _markdown_table([row], sorted(str(k) for k in row.keys()))
                + "\n### Env Info Detail\n\n"
                + _markdown_table(info_rows, ["key", "value"])
            )
        )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path


def _grid_frame(observation: Dict[str, np.ndarray]) -> np.ndarray:
    keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
    imgs = [observation[k] for k in keys]
    row1 = np.concatenate(imgs[:3], axis=1)
    row2 = np.concatenate(imgs[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)


def _grid_frame_available_cameras(observation: Dict[str, np.ndarray]) -> np.ndarray:
    front_keys = ["front_left", "front", "front_right"]
    rear_keys = ["back_left", "back", "back_right"]
    if all(k in observation for k in front_keys + rear_keys):
        return _grid_frame(observation)
    if all(k in observation for k in front_keys):
        return np.concatenate([np.asarray(observation[k], dtype=np.uint8) for k in front_keys], axis=1)
    available = [k for k in front_keys + rear_keys if k in observation]
    if not available:
        raise RuntimeError("Observation has no camera images to render")
    return np.concatenate([np.asarray(observation[k], dtype=np.uint8) for k in available], axis=1)


def _obs_tensor_to_camera_observation(obs_t: Any) -> Dict[str, np.ndarray]:
    if torch.is_tensor(obs_t):
        arr = obs_t.detach().cpu().numpy()
    else:
        arr = np.asarray(obs_t)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 18:
        raise RuntimeError(f"Expected obs tensor shape (18,H,W), got {arr.shape}")
    arr = np.clip(arr.reshape(6, 3, arr.shape[1], arr.shape[2]).transpose(0, 2, 3, 1), 0.0, 1.0)
    arr_u8 = np.rint(arr * 255.0).astype(np.uint8)
    keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
    return {key: arr_u8[i] for i, key in enumerate(keys)}


def _overlay_debug_text(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    if len(lines) == 0:
        return frame
    try:
        import cv2
    except Exception:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    box_h = min(h - 10, 26 + 22 * len(lines))
    box_w = min(w - 10, 920)
    x0, y0 = 8, 8
    roi = out[y0 : y0 + box_h, x0 : x0 + box_w].copy()
    shade = roi.copy()
    cv2.rectangle(shade, (0, 0), (box_w - 1, box_h - 1), (16, 16, 16), thickness=-1)
    out[y0 : y0 + box_h, x0 : x0 + box_w] = cv2.addWeighted(shade, 0.50, roi, 0.50, 0)
    y = y0 + 22
    for line in lines:
        cv2.putText(out, str(line), (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (236, 236, 236), 1, cv2.LINE_AA)
        y += 22
    return out


def _tensor_seq_value(seq: Any, idx: int, default: Any = None) -> Any:
    if seq is None:
        return default
    try:
        value = seq[idx]
    except Exception:
        return default
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _maybe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _maybe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _build_shard_reward_rows(shard: Dict[str, Any]) -> List[Dict[str, Any]]:
    rewards = shard.get("reward", [])
    replay_rows = shard.get("replay", [])
    if torch.is_tensor(rewards):
        num_steps = int(rewards.numel())
    else:
        num_steps = len(rewards)
    reward_rows: List[Dict[str, Any]] = []
    reward_sum = 0.0
    for step in range(num_steps):
        replay = replay_rows[step] if isinstance(replay_rows, list) and step < len(replay_rows) and isinstance(replay_rows[step], dict) else {}
        reward_v = _maybe_float(_tensor_seq_value(rewards, step, 0.0))
        reward_sum += reward_v
        done_v = bool(_maybe_float(_tensor_seq_value(shard.get("done"), step, 0.0)) > 0.5)
        old_logp = _maybe_float(_tensor_seq_value(shard.get("old_logp"), step, 0.0))
        try:
            traj_points = int(_traj_xyyaw_from_replay(replay).shape[0])
        except Exception:
            traj_points = 0
        reward_rows.append(
            {
                "step": int(step),
                "scene_id": _maybe_int(replay.get("scene_id", -1)),
                "frame_idx": _maybe_int(replay.get("frame_idx", -1)),
                "timestamp_s": _maybe_float(replay.get("timestamp_s", 0.0)),
                "mode_idx": _maybe_int(replay.get("mode_idx", -1)),
                "old_logp": float(old_logp),
                "reward": float(reward_v),
                "cum_reward": float(reward_sum),
                "done": bool(done_v),
                "traj_points": int(traj_points),
            }
        )
    return reward_rows


def _draw_shard_plan_bev(
    frame: np.ndarray,
    traj_xyyaw: np.ndarray | None,
    *,
    view_m: float = 25.0,
) -> np.ndarray:
    if traj_xyyaw is None:
        return frame
    try:
        import cv2
    except Exception:
        return frame

    traj = np.asarray(traj_xyyaw, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] < 2 or traj.shape[0] == 0:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    box_w, box_h = 320, 320
    x0, y0 = w - box_w - 10, 10
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (120, 120, 120), 1)
    cx, cy = x0 + box_w // 2, y0 + int(box_h * 0.78)
    scale = float(min(box_w, box_h) * 0.42 / max(1e-6, view_m))

    def local_to_px(xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float64)
        px = np.zeros((pts.shape[0], 2), dtype=np.int32)
        px[:, 0] = np.rint(cx + pts[:, 1] * scale).astype(np.int32)
        px[:, 1] = np.rint(cy - pts[:, 0] * scale).astype(np.int32)
        return px

    grid_color = (70, 70, 70)
    for m in range(0, int(view_m) + 1, 5):
        y = int(round(cy - float(m) * scale))
        if y0 <= y <= y0 + box_h:
            cv2.line(out, (x0 + 6, y), (x0 + box_w - 6, y), grid_color, 1)
    center_line = local_to_px(np.asarray([[0.0, 0.0], [view_m, 0.0]], dtype=np.float64))
    cv2.line(out, tuple(center_line[0]), tuple(center_line[1]), (90, 90, 90), 1)

    ego = local_to_px(np.asarray([[0.0, 0.0]], dtype=np.float64))[0]
    cv2.circle(out, tuple(ego), 5, (255, 255, 255), -1)
    cv2.putText(out, "EGO", (int(ego[0]) + 7, int(ego[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    pts = local_to_px(traj[:, :2])
    if pts.shape[0] >= 2:
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (0, 220, 255), 2)
    for idx, pt in enumerate(pts):
        cv2.circle(out, tuple(pt), 4 if idx == 0 else 3, (255, 160, 0) if idx == 0 else (0, 220, 255), -1)
        if idx in {0, pts.shape[0] - 1}:
            cv2.putText(out, str(idx), (int(pt[0]) + 4, int(pt[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(out, "BEV plan view", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def _resize_frame_min_width(frame: np.ndarray, min_width: int) -> np.ndarray:
    if int(frame.shape[1]) >= int(min_width):
        return frame
    try:
        import cv2
    except Exception:
        return frame
    scale = float(min_width) / float(max(1, int(frame.shape[1])))
    out_w = int(round(float(frame.shape[1]) * scale))
    out_h = int(round(float(frame.shape[0]) * scale))
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _load_env_cache(scene: int) -> Dict[int, Dict[str, Any]]:
    p = os.path.join(_REPO_ROOT, "assets", "nus", "data", f"{int(scene):03d}", "env_cache.json")
    if not os.path.isfile(p):
        return {}
    data = json.load(open(p, "r", encoding="utf-8"))
    if isinstance(data, dict) and "meta" in data:
        data = {k: v for k, v in data.items() if k != "meta"}
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in (data.items() if isinstance(data, dict) else []):
        try:
            out[int(k)] = dict(v)
        except Exception:
            continue
    return out


def _world_pose_from_sim(sim: Any) -> np.ndarray:
    cfs = np.asarray(getattr(sim, "camera_front_start"), dtype=np.float64)
    local = np.asarray(getattr(sim, "start_ego"), dtype=np.float64)
    if cfs.shape == (4, 4) and local.shape == (4, 4):
        return cfs @ local
    return local


def _ego_poly_world(world_pose: np.ndarray, ego_len: float = 4.2, ego_w: float = 1.9) -> np.ndarray:
    x = float(world_pose[0, 3])
    y = float(world_pose[1, 3])
    yaw = float(np.arctan2(float(world_pose[1, 0]), float(world_pose[0, 0])))
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    hl, hw = float(ego_len) * 0.5, float(ego_w) * 0.5
    pts = np.asarray([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float64)
    R = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return (pts @ R.T) + np.asarray([[x, y]], dtype=np.float64)


def _draw_collision_bev(
    frame: np.ndarray,
    *,
    world_pose: np.ndarray,
    snap: Dict[str, Any] | None,
    view_m: float = 25.0,
) -> tuple[np.ndarray, List[str]]:
    if not isinstance(snap, dict):
        return frame, []
    try:
        import cv2
    except Exception:
        return frame, []

    out = frame.copy()
    h, w = out.shape[:2]
    box_w, box_h = 320, 320
    x0, y0 = w - box_w - 10, 10
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (120, 120, 120), 1)

    ego_xy = np.asarray([float(world_pose[0, 3]), float(world_pose[1, 3])], dtype=np.float64)
    yaw = float(np.arctan2(float(world_pose[1, 0]), float(world_pose[0, 0])))
    c, s = float(np.cos(-yaw)), float(np.sin(-yaw))
    R = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    scale = float(min(box_w, box_h) * 0.45 / max(1e-6, view_m))
    cx, cy = x0 + box_w // 2, y0 + box_h // 2

    def world_to_px(poly_xy: np.ndarray) -> np.ndarray:
        rel = np.asarray(poly_xy, dtype=np.float64) - ego_xy[None, :]
        loc = rel @ R.T
        px = np.zeros_like(loc)
        px[:, 0] = cx - loc[:, 1] * scale
        px[:, 1] = cy - loc[:, 0] * scale
        return px.astype(np.int32)

    ego_poly = _ego_poly_world(world_pose)
    ego_shapely = Polygon(ego_poly.tolist())
    ego_px = world_to_px(ego_poly)
    # Colors are RGB (frame is written by imageio in RGB space).
    cv2.polylines(out, [ego_px.reshape(-1, 1, 2)], True, (255, 255, 255), 2)
    cv2.putText(out, "EGO", (int(np.mean(ego_px[:, 0])) + 4, int(np.mean(ego_px[:, 1])) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    collided_tokens: List[str] = []
    for obj in snap.get("dynamic_objects", []) or []:
        poly = obj.get("poly", None)
        if not (isinstance(poly, list) and len(poly) >= 3):
            continue
        poly_xy = np.asarray(poly, dtype=np.float64)
        shp = Polygon(poly_xy.tolist())
        is_hit = bool(ego_shapely.intersects(shp))
        clr = (255, 0, 0) if is_hit else (0, 200, 255)
        if is_hit:
            tok = str(obj.get("token", obj.get("id", "unknown")))
            collided_tokens.append(tok)
        px = world_to_px(poly_xy)
        if is_hit:
            overlay = out.copy()
            cv2.fillPoly(overlay, [px.reshape(-1, 1, 2)], (255, 0, 0))
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
            cv2.polylines(out, [px.reshape(-1, 1, 2)], True, (255, 0, 0), 2)
            ccx, ccy = int(np.mean(px[:, 0])), int(np.mean(px[:, 1]))
            cv2.putText(out, "HIT", (ccx + 3, ccy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 80), 1, cv2.LINE_AA)
        else:
            cv2.polylines(out, [px.reshape(-1, 1, 2)], True, clr, 1)

    cv2.putText(out, "BEV collision view", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return out, collided_tokens


def _hugsim_box_poly_xy(box: Any) -> np.ndarray:
    """HUGSIM box format is [x, y, z, width, length, height, yaw]."""
    arr = np.asarray(box, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 7:
        raise ValueError(f"Expected HUGSIM box with 7 values, got {arr.shape[0]}")
    x, y, _z, width, length, _height, yaw = [float(v) for v in arr[:7]]
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    offsets = np.asarray(
        [
            [length * 0.5, width * 0.5],
            [length * 0.5, -width * 0.5],
            [-length * 0.5, -width * 0.5],
            [-length * 0.5, width * 0.5],
        ],
        dtype=np.float64,
    )
    R = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return (offsets @ R.T) + np.asarray([[x, y]], dtype=np.float64)


def _draw_hugsim_box_bev(
    frame: np.ndarray,
    *,
    ego_box: Any,
    obj_boxes: Any,
    collision: bool = False,
    view_m: float = 25.0,
) -> tuple[np.ndarray, List[str]]:
    try:
        import cv2
    except Exception:
        return frame, []

    try:
        ego_arr = np.asarray(ego_box, dtype=np.float64).reshape(-1)
        ego_poly = _hugsim_box_poly_xy(ego_arr)
    except Exception:
        return frame, []

    out = frame.copy()
    h, w = out.shape[:2]
    box_w, box_h = min(320, max(120, w - 20)), min(320, max(120, h - 20))
    x0, y0 = w - box_w - 10, 10
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (120, 120, 120), 1)

    ego_xy = ego_arr[:2].astype(np.float64, copy=False)
    ego_yaw = float(ego_arr[6])
    c, s = float(np.cos(-ego_yaw)), float(np.sin(-ego_yaw))
    R = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    scale = float(min(box_w, box_h) * 0.45 / max(1e-6, view_m))
    cx, cy = x0 + box_w // 2, y0 + box_h // 2

    def hugsim_to_px(poly_xy: np.ndarray) -> np.ndarray:
        rel = np.asarray(poly_xy, dtype=np.float64) - ego_xy[None, :]
        loc = rel @ R.T
        px = np.zeros_like(loc)
        px[:, 0] = cx - loc[:, 1] * scale
        px[:, 1] = cy - loc[:, 0] * scale
        return px.astype(np.int32)

    ego_shape = Polygon(ego_poly.tolist())
    ego_px = hugsim_to_px(ego_poly)
    cv2.polylines(out, [ego_px.reshape(-1, 1, 2)], True, (255, 255, 255), 2)
    cv2.putText(
        out,
        "EGO",
        (int(np.mean(ego_px[:, 0])) + 4, int(np.mean(ego_px[:, 1])) - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    collided_tokens: List[str] = []
    obj_iter = [] if obj_boxes is None else obj_boxes
    for idx, obj_box in enumerate(obj_iter):
        try:
            obj_poly = _hugsim_box_poly_xy(obj_box)
            obj_shape = Polygon(obj_poly.tolist())
        except Exception:
            continue
        is_hit = bool(ego_shape.intersects(obj_shape))
        token = f"obj{idx}"
        px = hugsim_to_px(obj_poly)
        if is_hit:
            collided_tokens.append(token)
            overlay = out.copy()
            cv2.fillPoly(overlay, [px.reshape(-1, 1, 2)], (255, 0, 0))
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
            cv2.polylines(out, [px.reshape(-1, 1, 2)], True, (255, 0, 0), 2)
            ccx, ccy = int(np.mean(px[:, 0])), int(np.mean(px[:, 1]))
            cv2.putText(out, "HIT", (ccx + 3, ccy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 80), 1, cv2.LINE_AA)
        else:
            cv2.polylines(out, [px.reshape(-1, 1, 2)], True, (0, 200, 255), 1)

    label = "HUGSIM BEV boxes"
    if bool(collision) and not collided_tokens:
        label += " collision"
    cv2.putText(out, label, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return out, collided_tokens


def _draw_aligned_recon_global_bev(
    frame: np.ndarray,
    *,
    ego_poly: Any,
    hugsim_objects: Any,
    recon_objects: Any,
    view_m: float = 25.0,
) -> tuple[np.ndarray, List[str]]:
    try:
        import cv2
    except Exception:
        return frame, []

    try:
        ego_poly_arr = np.asarray(ego_poly, dtype=np.float64)
        if ego_poly_arr.ndim != 2 or ego_poly_arr.shape[0] < 3 or ego_poly_arr.shape[1] != 2:
            return frame, []
    except Exception:
        return frame, []

    out = frame.copy()
    h, w = out.shape[:2]
    box_w, box_h = min(320, max(120, w - 20)), min(320, max(120, h - 20))
    x0, y0 = w - box_w - 10, 10
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (120, 120, 120), 1)

    ego_xy = np.mean(ego_poly_arr, axis=0)
    front_mid = 0.5 * (ego_poly_arr[0] + ego_poly_arr[1])
    rear_mid = 0.5 * (ego_poly_arr[2] + ego_poly_arr[3])
    heading = front_mid - rear_mid
    ego_yaw = float(np.arctan2(float(heading[1]), float(heading[0]))) if float(np.linalg.norm(heading)) > 1.0e-9 else 0.0
    c, s = float(np.cos(-ego_yaw)), float(np.sin(-ego_yaw))
    R = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    scale = float(min(box_w, box_h) * 0.45 / max(1e-6, view_m))
    cx, cy = x0 + box_w // 2, y0 + box_h // 2

    def world_to_px(poly_xy: np.ndarray) -> np.ndarray:
        rel = np.asarray(poly_xy, dtype=np.float64) - ego_xy[None, :]
        loc = rel @ R.T
        px = np.zeros_like(loc)
        px[:, 0] = cx - loc[:, 1] * scale
        px[:, 1] = cy - loc[:, 0] * scale
        return px.astype(np.int32)

    ego_shape = Polygon(ego_poly_arr.tolist())
    ego_px = world_to_px(ego_poly_arr)
    cv2.polylines(out, [ego_px.reshape(-1, 1, 2)], True, (255, 255, 255), 2)
    cv2.putText(out, "EGO", (int(np.mean(ego_px[:, 0])) + 4, int(np.mean(ego_px[:, 1])) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    collided_tokens: List[str] = []
    draw_items: list[tuple[dict[str, Any], tuple[int, int, int]]] = []
    for obj in [] if hugsim_objects is None else hugsim_objects:
        if isinstance(obj, dict):
            draw_items.append((obj, (0, 220, 255)))
    for obj in [] if recon_objects is None else recon_objects:
        if isinstance(obj, dict):
            draw_items.append((obj, (255, 165, 0)))

    for obj, color in draw_items:
        poly = obj.get("poly", None)
        if not (isinstance(poly, list) and len(poly) >= 3):
            continue
        try:
            poly_xy = np.asarray(poly, dtype=np.float64)
            shp = Polygon(poly_xy.tolist())
        except Exception:
            continue
        is_hit = bool(ego_shape.intersects(shp))
        token = str(obj.get("token", obj.get("id", "unknown")))
        px = world_to_px(poly_xy)
        if is_hit:
            collided_tokens.append(token)
            overlay = out.copy()
            cv2.fillPoly(overlay, [px.reshape(-1, 1, 2)], (255, 0, 0))
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
            cv2.polylines(out, [px.reshape(-1, 1, 2)], True, (255, 0, 0), 2)
        else:
            cv2.polylines(out, [px.reshape(-1, 1, 2)], True, color, 1)

    cv2.putText(out, "Aligned Recon BEV", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return out, collided_tokens


def _pose_matrix_from_xyyaw(x: float, y: float, yaw: float) -> np.ndarray:
    c = float(math.cos(yaw))
    s = float(math.sin(yaw))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    return T


def _predict_next_pose_from_action(start_ego: np.ndarray, action: tuple[Any, ...]) -> np.ndarray:
    if not (isinstance(action, (tuple, list)) and len(action) == 4 and int(action[3]) == 2):
        raise RuntimeError(f"Unsupported continuous action for prediction: {action}")
    dx = float(action[0])
    dy = float(action[1])
    dyaw = float(action[2])
    return np.asarray(start_ego, dtype=np.float64) @ _pose_matrix_from_xyyaw(dx, dy, dyaw)


def _yaw_from_R_xy(Rm: np.ndarray) -> float:
    return float(np.arctan2(float(Rm[1, 0]), float(Rm[0, 0])))


def _relative_local_xyyaw(prev_pose: np.ndarray, next_pose: np.ndarray) -> np.ndarray:
    rel = np.linalg.inv(np.asarray(prev_pose, dtype=np.float64)) @ np.asarray(next_pose, dtype=np.float64)
    return np.asarray(
        [
            float(rel[0, 3]),
            float(rel[1, 3]),
            float(_yaw_from_R_xy(rel[:3, :3])),
        ],
        dtype=np.float64,
    )


def _local_plan_to_front_frame(start_ego: np.ndarray, traj_xyyaw: np.ndarray) -> np.ndarray:
    out = np.zeros((traj_xyyaw.shape[0], 4), dtype=np.float64)
    for i in range(traj_xyyaw.shape[0]):
        lx, ly, lyaw = float(traj_xyyaw[i, 0]), float(traj_xyyaw[i, 1]), float(traj_xyyaw[i, 2])
        tpt = _pose_matrix_from_xyyaw(lx, ly, lyaw)
        T_front = np.asarray(start_ego, dtype=np.float64) @ tpt
        out[i, 0] = float(T_front[0, 3])
        out[i, 1] = float(T_front[1, 3])
        out[i, 2] = float(T_front[2, 3])
        out[i, 3] = _yaw_from_R_xy(T_front[:3, :3])
    return out


def _ensure_obs_for_sparsedrive_v2(obs: Dict[str, Any], sim: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(obs)
    out.setdefault("timestamp", np.float32(float(getattr(sim, "now_frame", 0)) * 0.1))
    if "ego_pose" not in out:
        out["ego_pose"] = np.asarray(getattr(sim, "start_ego", np.eye(4)), dtype=np.float32)
    if "cam2ego" not in out:
        cam2ego = getattr(sim, "cam2ego", None)
        if isinstance(cam2ego, list) and len(cam2ego) >= 3:
            out["cam2ego"] = np.asarray(np.stack(cam2ego, axis=0), dtype=np.float32)
    if "cam_intrinsics" not in out:
        all_cams = getattr(sim, "all_cams", None)
        if isinstance(all_cams, list) and len(all_cams) >= 3:
            intr = []
            hw = []
            for cam in all_cams:
                intr.append(np.asarray(cam.get("intrinsics"), dtype=np.float32))
                hw.append([float(cam.get("height", sim.h)), float(cam.get("width", sim.w))])
            out["cam_intrinsics"] = np.asarray(np.stack(intr, axis=0), dtype=np.float32)
            out.setdefault("cam_hw", np.asarray(hw, dtype=np.float32))
    if "driving_command" not in out:
        out["driving_command"] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if "ego_status" not in out:
        vel = np.asarray(out.get("ego_velocity", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        acc = np.asarray(out.get("ego_acceleration", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        cmd = np.asarray(out.get("driving_command", np.zeros((4,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        cmd4 = np.zeros((4,), dtype=np.float32)
        vel2 = np.zeros((2,), dtype=np.float32)
        acc2 = np.zeros((2,), dtype=np.float32)
        cmd4[: min(4, cmd.shape[0])] = cmd[: min(4, cmd.shape[0])]
        vel2[: min(2, vel.shape[0])] = vel[: min(2, vel.shape[0])]
        acc2[: min(2, acc.shape[0])] = acc[: min(2, acc.shape[0])]
        out["ego_status"] = np.concatenate([cmd4, vel2, acc2], axis=0).astype(np.float32)
    return out


def _traj_xyyaw_from_replay(replay: Dict[str, Any]) -> np.ndarray:
    traj = replay.get("traj_xyyaw", None)
    if traj is None:
        raise RuntimeError("Replay missing traj_xyyaw")
    if torch.is_tensor(traj):
        arr = traj.detach().cpu().numpy()
    else:
        arr = np.asarray(traj)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RuntimeError(f"Invalid traj_xyyaw shape: {arr.shape}")
    return arr[:, :3]


def _extract_status_from_obs(obs: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cmd = np.asarray(obs.get("driving_command", np.zeros((4,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    vel = np.asarray(obs.get("ego_velocity", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    acc = np.asarray(obs.get("ego_acceleration", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)

    cmd4 = np.zeros((4,), dtype=np.float32)
    vel2 = np.zeros((2,), dtype=np.float32)
    acc2 = np.zeros((2,), dtype=np.float32)
    cmd4[: min(4, cmd.shape[0])] = cmd[: min(4, cmd.shape[0])]
    vel2[: min(2, vel.shape[0])] = vel[: min(2, vel.shape[0])]
    acc2[: min(2, acc.shape[0])] = acc[: min(2, acc.shape[0])]
    return cmd4, vel2, acc2


def _dataset_status_from_sim(sim: Any, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Prefer simulator's internal dataset mapping when available.
    fn = getattr(sim, "_status_from_dataset", None)
    if callable(fn):
        try:
            vel, acc, cmd = fn(int(frame_idx))
            vel2 = np.asarray(vel, dtype=np.float32).reshape(-1)[:2]
            acc2 = np.asarray(acc, dtype=np.float32).reshape(-1)[:2]
            cmd4 = np.asarray(cmd, dtype=np.float32).reshape(-1)[:4]

            out_vel = np.zeros((2,), dtype=np.float32)
            out_acc = np.zeros((2,), dtype=np.float32)
            out_cmd = np.zeros((4,), dtype=np.float32)
            out_vel[: vel2.shape[0]] = vel2
            out_acc[: acc2.shape[0]] = acc2
            out_cmd[: cmd4.shape[0]] = cmd4
            return out_cmd, out_vel, out_acc
        except Exception:
            pass
    return (
        np.zeros((4,), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
    )


def _load_expert_traj_front_xz(scene: int, start_frame: int, step_frames: int) -> np.ndarray:
    from reconsimulator.envs import nus_config as cfg  # type: ignore

    scene_dir = os.path.join(cfg.BASE_DATA_DIR, f"{int(scene):03d}")
    ego_pose_dir = os.path.join(scene_dir, "ego_pose")
    cam2ego0_path = os.path.join(scene_dir, "cam2ego", "0.txt")

    if not os.path.isdir(ego_pose_dir):
        raise FileNotFoundError(f"missing dir: {ego_pose_dir}")
    if not os.path.isfile(cam2ego0_path):
        raise FileNotFoundError(f"missing file: {cam2ego0_path}")

    pose_files = [n for n in os.listdir(ego_pose_dir) if n.endswith(".txt")]
    all_frames = sorted(int(os.path.splitext(n)[0]) for n in pose_files)
    frames = [f for f in all_frames if f >= int(start_frame) and ((f - int(start_frame)) % int(step_frames) == 0)]
    if not frames:
        return np.zeros((0, 2), dtype=np.float64)

    ego0_world = np.asarray(np.loadtxt(os.path.join(ego_pose_dir, f"{int(start_frame):03d}.txt")), dtype=np.float64)
    cam2ego0 = np.asarray(np.loadtxt(cam2ego0_path), dtype=np.float64)
    camera_front_start = ego0_world @ cam2ego0
    inv_front = np.linalg.inv(camera_front_start)

    rows = []
    for f in frames:
        T_ego_world = np.asarray(np.loadtxt(os.path.join(ego_pose_dir, f"{int(f):03d}.txt")), dtype=np.float64)
        T_front = inv_front @ T_ego_world
        rows.append([float(T_front[0, 3]), float(T_front[2, 3])])
    return np.asarray(rows, dtype=np.float64)


def _load_expert_front_xz_for_frame(
    scene: int,
    start_frame: int,
    frame_idx: int,
    *,
    base_data_dir: str | None = None,
) -> np.ndarray:
    if base_data_dir is None:
        from reconsimulator.envs import nus_config as cfg  # type: ignore

        base_data_dir = str(cfg.BASE_DATA_DIR)

    scene_dir = os.path.join(str(base_data_dir), f"{int(scene):03d}")
    ego_pose_dir = os.path.join(scene_dir, "ego_pose")
    cam2ego0_path = os.path.join(scene_dir, "cam2ego", "0.txt")

    ego0_world = np.asarray(np.loadtxt(os.path.join(ego_pose_dir, f"{int(start_frame):03d}.txt")), dtype=np.float64)
    cam2ego0 = np.asarray(np.loadtxt(cam2ego0_path), dtype=np.float64)
    camera_front_start = ego0_world @ cam2ego0
    inv_front = np.linalg.inv(camera_front_start)

    T_ego_world = np.asarray(np.loadtxt(os.path.join(ego_pose_dir, f"{int(frame_idx):03d}.txt")), dtype=np.float64)
    T_front = inv_front @ T_ego_world
    return np.asarray([float(T_front[0, 3]), float(T_front[2, 3])], dtype=np.float64)


def _append_online_expert_xz(target: List[List[float]], expert_front_xz: np.ndarray) -> None:
    arr = np.asarray(expert_front_xz, dtype=np.float64).reshape(2)
    target.append([float(arr[0]), float(arr[1])])


def _build_online_step_stats_paths(traj_plot_path: str) -> Dict[str, str]:
    traj_plot_abs = os.path.abspath(str(traj_plot_path))
    root, _ext = os.path.splitext(traj_plot_abs)
    suffix = "_expert_vs_ego_traj"
    if root.endswith(suffix):
        prefix = root[: -len(suffix)]
    else:
        prefix = root
    return {
        "per_step_csv": f"{prefix}_online_step_summary.csv",
        "aggregate_csv": f"{prefix}_online_step_aggregate.csv",
        "rollout_csv": f"{prefix}_online_rollout_points.csv",
        "overlay_svg": f"{prefix}_online_rollout_overlay.svg",
        "error_hist_svg": f"{prefix}_online_error_hist.svg",
        "worst_svg": f"{prefix}_online_worst_steps.svg",
    }


def _load_scene99_step_summary_module() -> Any:
    module_path = os.path.join(_REPO_ROOT, "outputs", "visualize", "debug_tracker_scene099", "summarize_scene99_tracker_steps.py")
    spec = importlib.util.spec_from_file_location("summarize_scene99_tracker_steps", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load scene99 summary module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_online_rollout_points_csv(rows: List[Dict[str, float | int]], out_path: str) -> None:
    _ensure_parent(out_path)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _step_marker_indices(num_points: int, every: int = 5) -> List[int]:
    if num_points <= 0:
        return []
    every_n = max(1, int(every))
    indices = list(range(0, int(num_points), every_n))
    last_idx = int(num_points) - 1
    if indices[-1] != last_idx:
        indices.append(last_idx)
    return indices


def _save_traj_plot_xz(scene: int, expert_xz: np.ndarray, ego_xz: np.ndarray, out_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[traj-plot] matplotlib not installed, skip export")
        return False

    if expert_xz.ndim != 2 or expert_xz.shape[1] != 2:
        raise RuntimeError(f"Invalid expert_xz shape: {expert_xz.shape}")
    if ego_xz.ndim != 2 or ego_xz.shape[1] != 2:
        raise RuntimeError(f"Invalid ego_xz shape: {ego_xz.shape}")

    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=140)
    ax.plot(expert_xz[:, 0], expert_xz[:, 1], color="#1f77b4", linewidth=2.0, label="expert")
    ax.plot(ego_xz[:, 0], ego_xz[:, 1], color="#d62728", linewidth=2.0, label="ego")
    ax.scatter(expert_xz[0, 0], expert_xz[0, 1], color="#1f77b4", s=28)
    ax.scatter(ego_xz[0, 0], ego_xz[0, 1], color="#d62728", s=28)
    for step_idx in _step_marker_indices(expert_xz.shape[0], every=5):
        x_val = float(expert_xz[step_idx, 0])
        z_val = float(expert_xz[step_idx, 1])
        ax.scatter([x_val], [z_val], color="#1f77b4", s=34, marker="o")
        ax.annotate(
            f"step {step_idx}",
            (x_val, z_val),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#1f77b4",
            alpha=0.95,
        )
    for step_idx in _step_marker_indices(ego_xz.shape[0], every=5):
        x_val = float(ego_xz[step_idx, 0])
        z_val = float(ego_xz[step_idx, 1])
        ax.scatter([x_val], [z_val], color="#d62728", s=38, marker="^")
        ax.annotate(
            f"step {step_idx}",
            (x_val, z_val),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=7,
            color="#d62728",
            alpha=0.95,
        )
    ax.set_title(f"Scene {scene:03d}: SparseDriveV2 Expert vs Ego (front-frame XZ, markers every 5 steps)")
    ax.set_xlabel("x (right +)")
    ax.set_ylabel("z (forward/north +)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    # fig.savefig(out_path)
    fig.savefig(out_path, format='svg')  
    plt.close(fig)
    return True


def _save_shard_plan_plot(reward_rows: List[Dict[str, Any]], replay_rows: List[Dict[str, Any]], out_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[shard-traj-plot] matplotlib not installed, skip export")
        return False

    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=140)
    cmap = plt.get_cmap("viridis")
    num_steps = max(1, len(replay_rows))
    plotted = 0
    for step, replay in enumerate(replay_rows):
        if not isinstance(replay, dict):
            continue
        try:
            traj = _traj_xyyaw_from_replay(replay)
        except Exception:
            continue
        color = cmap(float(step) / float(max(1, num_steps - 1)))
        ax.plot(traj[:, 1], traj[:, 0], color=color, linewidth=1.4, alpha=0.75)
        ax.scatter([traj[0, 1]], [traj[0, 0]], color=[color], s=12)
        if step % 5 == 0 or step == num_steps - 1:
            ax.annotate(str(step), (float(traj[0, 1]), float(traj[0, 0])), xytext=(3, 3), textcoords="offset points", fontsize=7)
        plotted += 1
    ax.scatter([0.0], [0.0], color="#d62728", s=35, marker="x", label="ego")
    done_steps = [int(row["step"]) for row in reward_rows if bool(row.get("done", False))]
    title = "Shard SparseDriveV2 local BEV plans"
    if done_steps:
        title += f" (done at {','.join(str(v) for v in done_steps)})"
    ax.set_title(title)
    ax.set_xlabel("local y / lateral (m)")
    ax.set_ylabel("local x / forward (m)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, format="svg" if out_path.lower().endswith(".svg") else None)
    plt.close(fig)
    return plotted > 0


def _write_csv_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    fieldnames = sorted({str(k) for row in rows for k in row.keys()}) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _export_shard_replay(args: argparse.Namespace) -> None:
    shard_path = _resolve_repo_path(str(args.from_shard))
    if not os.path.isfile(shard_path):
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    shard = torch.load(shard_path, map_location="cpu")
    if not isinstance(shard, dict):
        raise RuntimeError(f"Expected shard dict, got {type(shard)}")
    obs_all = shard.get("obs", None)
    if not torch.is_tensor(obs_all):
        raise RuntimeError("Shard missing tensor field `obs`")
    if obs_all.ndim != 4 or int(obs_all.shape[1]) != 18:
        raise RuntimeError(f"Expected shard obs shape (T,18,H,W), got {tuple(obs_all.shape)}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    stem = os.path.splitext(os.path.basename(shard_path))[0]
    shard_root = os.path.join(_DEFAULT_OUTPUT_ROOT, f"{stem}-{ts}")
    shard_artifacts_dir = os.path.join(shard_root, "artifacts")
    out_path = args.out or os.path.join(shard_artifacts_dir, f"{stem}_{ts}_shard_replay.mp4")
    traj_csv = args.traj_csv or os.path.join(shard_artifacts_dir, f"{stem}_{ts}_shard_plan.csv")
    traj_plot = args.traj_plot or os.path.join(shard_artifacts_dir, f"{stem}_{ts}_shard_bev.svg")
    _ensure_parent(out_path)
    _ensure_parent(traj_csv)
    _ensure_parent(traj_plot)
    if args.out is None and args.traj_csv is None and args.traj_plot is None:
        _write_run_manifest(
            manifest_path=os.path.join(shard_root, "run_info.md"),
            scene=0,
            config_path="actor_learner_shard",
            ckpt_path=shard_path,
            timestamp=ts,
            extra_lines=[f"from_shard={shard_path}", f"artifacts_dir={shard_artifacts_dir}"],
        )

    reward_rows = _build_shard_reward_rows(shard)
    replay_rows = list(shard.get("replay", [])) if isinstance(shard.get("replay", []), list) else []
    max_steps = min(int(obs_all.shape[0]), len(reward_rows))
    if max_steps <= 0:
        raise RuntimeError("Shard has no steps to render")
    fps = float(args.fps) if args.fps is not None else 2.0

    print("==== generate_video_sparsedrive_v2 shard replay ====")
    print(f"from_shard={shard_path}")
    print(f"steps={max_steps} fps={fps:.3f}")
    print(f"out_video={out_path}")
    print(f"out_traj_csv={traj_csv}")
    print(f"out_traj_plot={traj_plot}")

    writer = imageio.get_writer(
        out_path,
        mode="I",
        fps=float(fps),
        macro_block_size=1,
        codec="libx264",
        ffmpeg_log_level="error",
        input_params=["-framerate", str(float(fps))],
        output_params=["-pix_fmt", "yuv420p"],
    )

    rendered_frames: List[np.ndarray] = []
    traj_rows: List[Dict[str, Any]] = []
    debug_shard: Dict[str, Any] = {"reward": [], "done": [], "replay": replay_rows[:max_steps], "meta": []}
    try:
        for step in range(max_steps):
            row = reward_rows[step]
            replay = replay_rows[step] if step < len(replay_rows) and isinstance(replay_rows[step], dict) else {}
            obs = _obs_tensor_to_camera_observation(obs_all[step])
            try:
                traj_xyyaw = _traj_xyyaw_from_replay(replay)
            except Exception:
                traj_xyyaw = None

            lines = [
                f"shard step={step} scene={row.get('scene_id', -1)} frame={row.get('frame_idx', -1)}",
                f"reward={float(row.get('reward', 0.0)):.5f} cum_reward={float(row.get('cum_reward', 0.0)):.5f} done={bool(row.get('done', False))}",
                f"mode_idx={row.get('mode_idx', -1)} old_logp={float(row.get('old_logp', 0.0)):.5f} sample={str(replay.get('sample_token', ''))[:12]}",
            ]
            frame_out = _resize_frame_min_width(_grid_frame(obs), 960)
            frame_out = _overlay_debug_text(frame_out, lines)
            frame_out = _draw_shard_plan_bev(frame_out, traj_xyyaw, view_m=25.0)
            writer.append_data(frame_out)
            rendered_frames.append(frame_out.copy())

            if traj_xyyaw is not None:
                for pt_idx in range(int(traj_xyyaw.shape[0])):
                    traj_rows.append(
                        {
                            "step": int(step),
                            "scene_id": row.get("scene_id", -1),
                            "frame_idx": row.get("frame_idx", -1),
                            "mode_idx": row.get("mode_idx", -1),
                            "point_idx": int(pt_idx),
                            "local_x": float(traj_xyyaw[pt_idx, 0]),
                            "local_y": float(traj_xyyaw[pt_idx, 1]),
                            "local_yaw": float(traj_xyyaw[pt_idx, 2]),
                        }
                    )

            debug_shard["reward"].append(float(row.get("reward", 0.0)))
            debug_shard["done"].append(bool(row.get("done", False)))
            debug_shard["meta"].append({"step": int(step), "info": dict(row)})
    finally:
        writer.close()

    _write_csv_rows(traj_csv, traj_rows)
    print(f"traj_saved={traj_csv}")

    reward_csv = os.path.splitext(traj_csv)[0] + "_step_reward.csv"
    _write_csv_rows(reward_csv, reward_rows[:max_steps])
    print(f"reward_csv_saved={reward_csv}")

    reward_plot = os.path.splitext(traj_plot)[0] + "_reward_curve.png"
    try:
        xs = [int(r["step"]) for r in reward_rows[:max_steps]]
        ys = [float(r["reward"]) for r in reward_rows[:max_steps]]
        cs = [float(r["cum_reward"]) for r in reward_rows[:max_steps]]
        fig, ax1 = plt.subplots(figsize=(9, 4), dpi=150)
        ax1.plot(xs, ys, color="#d62728", label="step_reward")
        ax1.axhline(0.0, color="#888888", linewidth=1.0)
        ax2 = ax1.twinx()
        ax2.plot(xs, cs, color="#1f77b4", label="cum_reward")
        for row in reward_rows[:max_steps]:
            if bool(row.get("done", False)):
                ax1.axvline(int(row["step"]), color="#444444", linestyle="--", linewidth=1.0, alpha=0.6)
        ax1.set_xlabel("step")
        ax1.set_ylabel("reward")
        ax2.set_ylabel("cum_reward")
        ax1.set_title("Shard Reward Curve")
        fig.tight_layout()
        fig.savefig(reward_plot)
        plt.close(fig)
        print(f"reward_plot_saved={reward_plot}")
    except Exception as e:
        print(f"[reward-plot] failed: {e}")

    if _save_shard_plan_plot(reward_rows[:max_steps], replay_rows[:max_steps], traj_plot):
        print(f"traj_plot_saved={traj_plot}")

    debug_shard_path = os.path.splitext(traj_csv)[0] + "_debug_shard.pt"
    try:
        torch.save(debug_shard, debug_shard_path)
        print(f"debug_shard_saved={debug_shard_path}")
    except Exception as e:
        print(f"[debug-shard] failed: {e}")

    if str(args.reward_detail_format) == "ipynb":
        detail_ipynb = os.path.splitext(traj_csv)[0] + "_reward_detail.ipynb"
        try:
            first_scene = _maybe_int(reward_rows[0].get("scene_id", 0), 0) if reward_rows else 0
            _save_reward_detail_notebook(
                out_path=detail_ipynb,
                scene=first_scene,
                reward_rows=reward_rows[:max_steps],
                debug_shard=debug_shard,
                reward_cfg={"source": "actor_learner_shard", "shard_path": shard_path, "meta": shard.get("meta", {})},
                video_path=out_path,
                reward_csv_path=reward_csv,
                debug_shard_path=debug_shard_path,
            )
            print(f"reward_detail_notebook_saved={detail_ipynb}")
        except Exception as e:
            print(f"[reward-detail-notebook] failed: {e}")

    if bool(args.save_keyframes) and reward_rows and rendered_frames:
        try:
            key_root = os.path.splitext(traj_plot)[0] + "_keyframes"
            os.makedirs(key_root, exist_ok=True)
            order = sorted(range(max_steps), key=lambda i: float(reward_rows[i]["reward"]))
            for idx in order[: max(1, int(args.keyframes_k))]:
                out_png = os.path.join(key_root, f"step{idx:03d}_reward{float(reward_rows[idx]['reward']):+.4f}.png")
                imageio.imwrite(out_png, rendered_frames[idx])
            print(f"keyframes_saved={key_root}")
        except Exception as e:
            print(f"[keyframes] failed: {e}")

    print(f"video_saved={out_path}")
    print("==== shard replay done ====")


def _resolve_hugsim_collect_defaults(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    horizon = getattr(args, "horizon", None)
    if horizon is None:
        horizon = 32
    mode_select = getattr(args, "mode_select", None)
    if mode_select is None:
        mode_select = train_cfg.get("policy_mode_select", train_cfg.get("ddv2_mode_select", "sample"))
    return {
        "horizon": int(horizon),
        "eta": float(train_cfg.get("eta", train_cfg.get("ddv2_eta", 1.0))),
        "mode_idx": int(train_cfg.get("mode_idx", train_cfg.get("ddv2_mode_idx", -1))),
        "mode_select": str(mode_select).strip().lower(),
    }


def _resolve_hugsim_online_output_paths(args: argparse.Namespace, paths: Dict[str, str]) -> Dict[str, str]:
    return {
        "out": args.out or paths["video_path"],
        "traj_csv": args.traj_csv or paths["traj_csv"],
        "traj_plot": args.traj_plot or paths["traj_plot"],
    }


def _build_collected_shard_replay_args(
    args: argparse.Namespace,
    shard_path: str,
    paths: Dict[str, str],
) -> argparse.Namespace:
    return argparse.Namespace(
        from_shard=str(shard_path),
        out=(args.out or paths["video_path"]),
        traj_csv=(args.traj_csv or paths["traj_csv"]),
        traj_plot=(args.traj_plot or paths["traj_plot"]),
        fps=args.fps,
        reward_detail_format=args.reward_detail_format,
        save_keyframes=bool(args.save_keyframes),
        keyframes_k=int(args.keyframes_k),
    )


def _first_hugsim_label(cfg: Dict[str, Any]) -> str:
    env_cfg = cfg.get("env", {}) if isinstance(cfg, dict) else {}
    hugsim_cfg = env_cfg.get("hugsim", {}) if isinstance(env_cfg, dict) else {}
    scenes = hugsim_cfg.get("scenes", None) if isinstance(hugsim_cfg, dict) else None
    if isinstance(scenes, list) and scenes:
        return str(scenes[0])
    return "hugsim_ori"


def _render_hugsim_online_bev(
    frame: np.ndarray,
    *,
    traj_xyyaw: np.ndarray | None,
    world_pose: np.ndarray | None,
    snap: Dict[str, Any] | None,
    draw_collision_bev: bool,
    hugsim_ego_box: Any | None = None,
    hugsim_obj_boxes: Any | None = None,
    hugsim_collision: bool = False,
    aligned_ego_poly: Any | None = None,
    hugsim_recon_objects: Any | None = None,
    recon_cache_objects: Any | None = None,
) -> tuple[np.ndarray, List[str]]:
    if bool(draw_collision_bev) and aligned_ego_poly is not None:
        return _draw_aligned_recon_global_bev(
            frame,
            ego_poly=aligned_ego_poly,
            hugsim_objects=[] if hugsim_recon_objects is None else hugsim_recon_objects,
            recon_objects=[] if recon_cache_objects is None else recon_cache_objects,
            view_m=25.0,
        )
    if bool(draw_collision_bev) and hugsim_ego_box is not None:
        return _draw_hugsim_box_bev(
            frame,
            ego_box=hugsim_ego_box,
            obj_boxes=[] if hugsim_obj_boxes is None else hugsim_obj_boxes,
            collision=bool(hugsim_collision),
            view_m=25.0,
        )
    if bool(draw_collision_bev) and world_pose is not None:
        snap_payload = snap if isinstance(snap, dict) else {}
        return _draw_collision_bev(frame, world_pose=np.asarray(world_pose, dtype=np.float64), snap=snap_payload, view_m=25.0)
    return _draw_shard_plan_bev(frame, traj_xyyaw, view_m=25.0), []


def _apply_cli_ckpt_override(cfg: Dict[str, Any], ckpt_path: str | None) -> None:
    if ckpt_path is None:
        return
    agent_cfg = cfg.setdefault("agent", {})
    if not isinstance(agent_cfg, dict):
        raise RuntimeError("Config field `agent` must be a mapping when overriding --ckpt")
    agent_cfg["ckpt"] = str(ckpt_path)


def _format_hugsim_scene_name(scene: Any) -> str:
    text = str(scene).strip()
    if not text:
        raise ValueError("Empty HUGSIM scene name")
    if text.startswith("scene-"):
        return text
    try:
        return f"scene-{int(text):04d}"
    except ValueError:
        return text


def _apply_hugsim_scene_override(cfg: Dict[str, Any], args: argparse.Namespace) -> str | None:
    scene_arg = getattr(args, "hugsim_scene", None)
    if scene_arg is None:
        scene_arg = getattr(args, "scene", None)
    if scene_arg is None:
        return None
    scene_name = _format_hugsim_scene_name(scene_arg)
    env_cfg = cfg.setdefault("env", {})
    if not isinstance(env_cfg, dict):
        raise RuntimeError("Config field `env` must be a mapping when overriding HUGSIM scene")
    hugsim_cfg = env_cfg.setdefault("hugsim", {})
    if not isinstance(hugsim_cfg, dict):
        raise RuntimeError("Config field `env.hugsim` must be a mapping when overriding HUGSIM scene")
    hugsim_cfg["scenes"] = [scene_name]
    return scene_name


def _build_hugsim_online_reward_row(
    *,
    step: int,
    replay: Dict[str, Any],
    logp: Any,
    reward: float,
    cum_reward: float,
    done: bool,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        traj_points = int(_traj_xyyaw_from_replay(replay).shape[0])
    except Exception:
        traj_points = 0
    logp_v = float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp)
    return {
        "step": int(step),
        "scene_id": _maybe_int(replay.get("scene_id", info.get("scene_id", info.get("scene", -1)))),
        "frame_idx": _maybe_int(replay.get("frame_idx", -1)),
        "frame_after": _maybe_int(info.get("frame_idx", info.get("now_frame", -1))),
        "sample_token": str(replay.get("sample_token", "")),
        "sample_token_after": str(info.get("sample_token", "")),
        "timestamp_s": _maybe_float(replay.get("timestamp_s", 0.0)),
        "mode_idx": _maybe_int(replay.get("mode_idx", -1)),
        "old_logp": float(logp_v),
        "reward": float(reward),
        "cum_reward": float(cum_reward),
        "done": bool(done),
        "done_reason": str(info.get("done_reason", "")),
        "progress_reward": _maybe_float(info.get("progress_reward", 0.0)),
        "cost_reward": _maybe_float(info.get("cost_reward", 0.0)),
        "traj_points": int(traj_points),
    }


def _apply_hugsim_online_reward_fallback(
    *,
    reward: float,
    info: Dict[str, Any],
    prev_route_completion: float | None,
) -> tuple[float, float | None]:
    if not isinstance(info, dict) or "hugsim_route_completion" not in info:
        return float(reward), prev_route_completion
    try:
        route_completion = float(info["hugsim_route_completion"])
    except Exception:
        return float(reward), prev_route_completion

    next_route_completion = route_completion
    raw_reward = float(reward)
    alignment_valid = info.get("hugsim_recon_alignment_valid", None) is True
    if alignment_valid and math.isfinite(raw_reward):
        return raw_reward, next_route_completion

    reward_mode = str(info.get("reward_mode", "")).strip().lower()
    lateral_error = _maybe_float(info.get("lateral_error_m", 0.0), 0.0)
    mixed_step_path = bool(reward_mode == "step_path" and abs(float(lateral_error)) > 100.0)
    near_zero_reward = abs(raw_reward) < 1.0e-8
    nonfinite_reward = not math.isfinite(raw_reward)
    if not bool(nonfinite_reward or mixed_step_path or (not alignment_valid and near_zero_reward)):
        return raw_reward, next_route_completion

    base_reward = _maybe_float(info.get("hugsim_base_reward", 0.0), 0.0)
    if bool(info.get("collision", False)) and abs(float(base_reward)) > 1.0e-8:
        fallback_reward = float(base_reward)
    else:
        prev = 0.0 if prev_route_completion is None else float(prev_route_completion)
        fallback_reward = round(float(route_completion) - prev, 6)

    info["raw_recondreamer_reward"] = raw_reward
    info["invalid_recon_step_path"] = bool(mixed_step_path)
    info["reward_mode"] = "hugsim_route_delta_fallback"
    info["reward"] = float(fallback_reward)
    info["progress_reward"] = float(fallback_reward)
    info["cost_reward"] = 0.0
    info["hugsim_route_completion_delta"] = float(fallback_reward)
    return float(fallback_reward), next_route_completion


def _collect_hugsim_shard_and_export(args: argparse.Namespace) -> None:
    from framework.runner.agent_factory import build_agent
    from framework.runner.env_factory import build_actor_env

    config_path = _resolve_repo_path(str(args.config))
    cfg = _load_yaml(config_path)
    ckpt_path = _resolve_repo_path(str(args.ckpt)) if getattr(args, "ckpt", None) is not None else None
    _apply_cli_ckpt_override(cfg, ckpt_path)
    selected_hugsim_scene = _apply_hugsim_scene_override(cfg, args)
    env_cfg = cfg.get("env", {}) if isinstance(cfg, dict) else {}
    if str(env_cfg.get("backend", "")).strip().lower() != "hugsim_ori":
        raise RuntimeError("--collect-hugsim-shard requires config env.backend: hugsim_ori")

    device = torch.device(f"cuda:{int(args.cuda)}" if torch.cuda.is_available() and int(args.cuda) >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(args.cuda))

    ts = time.strftime("%Y%m%d-%H%M%S")
    paths = _build_hugsim_shard_run_paths(timestamp=ts, label=_first_hugsim_label(cfg))
    output_paths = _resolve_hugsim_online_output_paths(args, paths)
    shard_path = _resolve_repo_path(str(args.save_shard)) if args.save_shard else paths["shard_path"]
    if args.save_shard:
        _ensure_parent(shard_path)
    for value in output_paths.values():
        _ensure_parent(value)

    defaults = _resolve_hugsim_collect_defaults(args, cfg)
    actor_id = int(args.actor_id)
    total_actors = int(args.total_actors) if args.total_actors is not None else 1
    local_ver = int(args.local_ver)
    shard_idx = int(args.shard_idx)

    print("==== collect HUGSIM-ORI shard ====")
    print(f"config={config_path}")
    print(f"ckpt={ckpt_path or str((cfg.get('agent', {}) or {}).get('ckpt', ''))}")
    if selected_hugsim_scene is not None:
        print(f"hugsim_scene={selected_hugsim_scene}")
    print(f"device={device}")
    print(f"horizon={defaults['horizon']} eta={defaults['eta']} mode_idx={defaults['mode_idx']} mode_select={defaults['mode_select']}")
    print(f"out_video={output_paths['out']}")
    print(f"out_traj_csv={output_paths['traj_csv']}")
    print(f"out_traj_plot={output_paths['traj_plot']}")
    if args.save_shard:
        print(f"save_shard={shard_path}")

    env = build_actor_env(cfg, cuda=max(0, int(args.cuda)), actor_id=actor_id, total_actors=total_actors)
    debug_shard: Dict[str, Any] = {
        "reward": [],
        "done": [],
        "old_logp": [],
        "replay": [],
        "meta": [],
    }
    env_cache_by_scene: Dict[int, Dict[int, Dict[str, Any]]] = {}
    reward_rows: List[Dict[str, Any]] = []
    traj_rows: List[Dict[str, Any]] = []
    rendered_frames: List[np.ndarray] = []
    writer = None
    try:
        agent = build_agent(cfg, device=device)
        obs, reset_info = env.reset()
        fps = float(args.fps) if args.fps is not None else 2.0
        writer = imageio.get_writer(
            output_paths["out"],
            mode="I",
            fps=float(fps),
            macro_block_size=1,
            codec="libx264",
            ffmpeg_log_level="error",
            input_params=["-framerate", str(float(fps))],
            output_params=["-pix_fmt", "yuv420p"],
        )

        reward_sum = 0.0
        prev_route_completion: float | None = None
        done = False
        step = 0
        max_steps = None if int(defaults["horizon"]) <= 0 else int(defaults["horizon"])
        while (max_steps is None or step < max_steps) and not done:
            obs_decision = obs
            action, logp, replay = agent.act(
                obs_decision,
                eta=float(defaults["eta"]),
                mode_idx=int(defaults["mode_idx"]),
                mode_select=str(defaults["mode_select"]),
            )
            try:
                traj_xyyaw = _traj_xyyaw_from_replay(replay)
            except Exception:
                traj_xyyaw = None
            setter = getattr(env, "set_external_plan_local_xyyaw", None)
            if callable(setter) and traj_xyyaw is not None:
                setter(np.asarray(traj_xyyaw, dtype=np.float32))

            obs, reward, terminated, truncated, info_raw = env.step(action)
            info = dict(info_raw or {})
            done = bool(terminated or truncated)
            reward, prev_route_completion = _apply_hugsim_online_reward_fallback(
                reward=float(reward),
                info=info,
                prev_route_completion=prev_route_completion,
            )
            reward_sum += float(reward)
            row = _build_hugsim_online_reward_row(
                step=int(step),
                replay=replay if isinstance(replay, dict) else {},
                logp=logp,
                reward=float(reward),
                cum_reward=float(reward_sum),
                done=bool(done),
                info=info,
            )
            reward_rows.append(row)
            debug_shard["reward"].append(float(reward))
            debug_shard["done"].append(bool(done))
            debug_shard["old_logp"].append(float(row["old_logp"]))
            debug_shard["replay"].append(replay)
            debug_shard["meta"].append({"step": int(step), "reset_info": _json_safe(reset_info), "info": _json_safe(info)})

            if traj_xyyaw is not None:
                for pt_idx in range(int(traj_xyyaw.shape[0])):
                    traj_rows.append(
                        {
                            "step": int(step),
                            "scene_id": row["scene_id"],
                            "frame_idx": row["frame_idx"],
                            "mode_idx": row["mode_idx"],
                            "point_idx": int(pt_idx),
                            "local_x": float(traj_xyyaw[pt_idx, 0]),
                            "local_y": float(traj_xyyaw[pt_idx, 1]),
                            "local_yaw": float(traj_xyyaw[pt_idx, 2]),
                        }
                    )

            lines = [
                f"hugsim step={step} scene={row['scene_id']} frame={row['frame_idx']}->{row['frame_after']}",
                f"reward={float(row['reward']):.5f} cum_reward={float(row['cum_reward']):.5f} done={bool(row['done'])}",
                f"mode_idx={row['mode_idx']} old_logp={float(row['old_logp']):.5f} sample={str(row['sample_token'])[:12]}",
            ]
            frame_out = _resize_frame_min_width(_grid_frame_available_cameras(obs), 960)
            frame_out = _overlay_debug_text(frame_out, lines)
            scene_id = _maybe_int(row.get("scene_id", -1))
            frame_after = _maybe_int(row.get("frame_after", -1))
            if scene_id not in env_cache_by_scene:
                env_cache_by_scene[scene_id] = _load_env_cache(scene=scene_id) if scene_id >= 0 else {}
            snap = env_cache_by_scene.get(scene_id, {}).get(frame_after)
            world_pose = obs.get("ego_pose", None) if isinstance(obs, dict) else None
            frame_out, hit_tokens = _render_hugsim_online_bev(
                frame_out,
                traj_xyyaw=traj_xyyaw,
                world_pose=world_pose,
                snap=snap,
                draw_collision_bev=bool(args.draw_collision_bev),
                hugsim_ego_box=info.get("ego_box"),
                hugsim_obj_boxes=info.get("obj_boxes"),
                hugsim_collision=bool(info.get("collision", False)),
                aligned_ego_poly=info.get("hugsim_ego_box_recon_global_poly"),
                hugsim_recon_objects=info.get("hugsim_obj_boxes_recon_global"),
                recon_cache_objects=info.get("recon_cache_dynamic_objects"),
            )
            if hit_tokens:
                row["collision_tokens"] = "|".join(str(v) for v in hit_tokens)
                frame_out = _overlay_debug_text(frame_out, lines + [f"collision_tokens={','.join(str(v) for v in hit_tokens[:2])}"])
            writer.append_data(frame_out)
            rendered_frames.append(frame_out.copy())
            step += 1

        writer.close()
        writer = None

        _write_csv_rows(output_paths["traj_csv"], traj_rows)
        reward_csv = os.path.splitext(output_paths["traj_csv"])[0] + "_step_reward.csv"
        _write_csv_rows(reward_csv, reward_rows)
        print(f"traj_saved={output_paths['traj_csv']}")
        print(f"reward_csv_saved={reward_csv}")
        if _save_shard_plan_plot(reward_rows, debug_shard["replay"], output_paths["traj_plot"]):
            print(f"traj_plot_saved={output_paths['traj_plot']}")

        reward_plot = os.path.splitext(output_paths["traj_plot"])[0] + "_reward_curve.png"
        try:
            xs = [int(r["step"]) for r in reward_rows]
            ys = [float(r["reward"]) for r in reward_rows]
            cs = [float(r["cum_reward"]) for r in reward_rows]
            if xs:
                fig, ax1 = plt.subplots(figsize=(9, 4), dpi=150)
                ax1.plot(xs, ys, color="#d62728", label="step_reward")
                ax1.axhline(0.0, color="#888888", linewidth=1.0)
                ax2 = ax1.twinx()
                ax2.plot(xs, cs, color="#1f77b4", label="cum_reward")
                ax1.set_xlabel("step")
                ax1.set_ylabel("reward")
                ax2.set_ylabel("cum_reward")
                ax1.set_title("HUGSIM-ORI Online Reward Curve")
                fig.tight_layout()
                fig.savefig(reward_plot)
                plt.close(fig)
                print(f"reward_plot_saved={reward_plot}")
        except Exception as e:
            print(f"[reward-plot] failed: {e}")

        debug_shard_path = shard_path
        if args.save_shard:
            shard = {
                "old_logp": torch.tensor(debug_shard["old_logp"], dtype=torch.float32),
                "reward": torch.tensor(debug_shard["reward"], dtype=torch.float32),
                "done": torch.tensor([1.0 if v else 0.0 for v in debug_shard["done"]], dtype=torch.float32),
                "replay": debug_shard["replay"],
                "meta": {
                    "source": "hugsim_ori_visualize_online",
                    "actor_id": actor_id,
                    "horizon": int(defaults["horizon"]),
                    "weights_version": local_ver,
                    "shard_idx": shard_idx,
                    "config_path": str(config_path),
                    "reset_info": _json_safe(reset_info),
                },
            }
            torch.save(shard, shard_path)
            print(f"shard_saved={shard_path}")
        else:
            debug_shard_path = os.path.splitext(output_paths["traj_csv"])[0] + "_debug_shard.pt"
            torch.save(debug_shard, debug_shard_path)
            print(f"debug_shard_saved={debug_shard_path}")

        if str(args.reward_detail_format) == "ipynb":
            detail_ipynb = os.path.splitext(output_paths["traj_csv"])[0] + "_reward_detail.ipynb"
            try:
                first_scene = _maybe_int(reward_rows[0].get("scene_id", 0), 0) if reward_rows else 0
                _save_reward_detail_notebook(
                    out_path=detail_ipynb,
                    scene=first_scene,
                    reward_rows=reward_rows,
                    debug_shard=debug_shard,
                    reward_cfg={"source": "hugsim_ori_online", "config": config_path, "env_reward": env_cfg.get("reward", {})},
                    video_path=output_paths["out"],
                    reward_csv_path=reward_csv,
                    debug_shard_path=debug_shard_path,
                )
                print(f"reward_detail_notebook_saved={detail_ipynb}")
            except Exception as e:
                print(f"[reward-detail-notebook] failed: {e}")

        if bool(args.save_keyframes) and reward_rows and rendered_frames:
            try:
                key_root = os.path.splitext(output_paths["traj_plot"])[0] + "_keyframes"
                os.makedirs(key_root, exist_ok=True)
                order = sorted(range(len(reward_rows)), key=lambda i: float(reward_rows[i]["reward"]))
                for idx in order[: max(1, int(args.keyframes_k))]:
                    out_png = os.path.join(key_root, f"step{idx:03d}_reward{float(reward_rows[idx]['reward']):+.4f}.png")
                    imageio.imwrite(out_png, rendered_frames[idx])
                print(f"keyframes_saved={key_root}")
            except Exception as e:
                print(f"[keyframes] failed: {e}")
    finally:
        if writer is not None:
            writer.close()
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    _write_run_manifest(
        manifest_path=paths["run_manifest"],
        scene=0,
        config_path=config_path,
        ckpt_path=str((cfg.get("agent", {}) or {}).get("ckpt", "")),
        timestamp=ts,
        extra_lines=[
            "mode=collect_hugsim_shard",
            f"artifacts_dir={paths['artifacts_dir']}",
            f"video={output_paths['out']}",
            f"traj_csv={output_paths['traj_csv']}",
            f"traj_plot={output_paths['traj_plot']}",
            f"horizon={int(defaults['horizon'])}",
            f"hugsim_scene={_first_hugsim_label(cfg)}",
            f"actor_id={actor_id}",
            f"local_ver={local_ver}",
        ],
    )
    print(f"video_saved={output_paths['out']}")
    print("==== HUGSIM-ORI online visualization done ====")


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate SparseDriveV2 rollout video in 3DGS env")
    ap.add_argument(
        "--config",
        type=str,
        default=os.path.join(_REPO_ROOT, "script", "configs", "sparsedrive_v2", "reinforcepp_closed_loop_sparsedrive_v2_no_grpo.yaml"),
        help="Config file providing env.reward and rollout defaults",
    )
    ap.add_argument("--from-shard", type=str, default=None, help="Render an existing actor-learner shard instead of running live policy rollout.")
    ap.add_argument("--collect-hugsim-shard", action="store_true", default=True, help="Run SparseDriveV2 online in HUGSIM-ORI and visualize each collected step. Enabled by default for this HUGSIM helper.")
    ap.add_argument("--no-collect-hugsim-shard", dest="collect_hugsim_shard", action="store_false", help="Use the legacy ReconSimulator online rollout path instead of HUGSIM collection.")
    ap.add_argument("--save-shard", type=str, default=None, help="Optional .pt debug shard path for --collect-hugsim-shard.")
    ap.add_argument("--horizon", type=int, default=32, help="Online visualization step count. Defaults to 32 steps at --fps 2, i.e. about 16s. Use 0 to run until env done.")
    ap.add_argument("--actor-id", type=int, default=0)
    ap.add_argument("--total-actors", type=int, default=None)
    ap.add_argument("--local-ver", type=int, default=0)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--scene", type=int, default=None)
    ap.add_argument("--hugsim-scene", type=str, default=None, help="HUGSIM scene override for --collect-hugsim-shard, e.g. scene-0010 or scene-0038-hard-00.")
    ap.add_argument("--ckpt", type=str, default=_DEFAULT_CKPT)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--traj-csv", type=str, default=None)
    ap.add_argument("--traj-plot", type=str, default=None)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--step-frames", type=int, default=5)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--render-w", type=int, default=None)
    ap.add_argument("--render-h", type=int, default=None)
    ap.add_argument("--mode-select", type=str, default=None, choices=["greedy", "sample"])
    ap.add_argument("--expert-high", dest="expert_high", action="store_true", default=True)
    ap.add_argument("--no-expert-high", dest="expert_high", action="store_false")
    ap.add_argument("--save-keyframes", dest="save_keyframes", action="store_true", default=True, help="Save low-reward keyframes as PNG")
    ap.add_argument("--no-save-keyframes", dest="save_keyframes", action="store_false", help="Disable low-reward keyframe PNG export.")
    ap.add_argument("--keyframes-k", type=int, default=8)
    ap.add_argument("--draw-collision-bev", action="store_true", default=True)
    ap.add_argument(
        "--reward-detail-format",
        type=str,
        default="ipynb",
        choices=["none", "ipynb"],
        help="Export detailed per-step reward report. Default writes an .ipynb next to the reward CSV.",
    )
    return ap


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    return _build_arg_parser().parse_args(argv)


def main() -> None:
    args = _parse_args()

    if args.from_shard:
        _export_shard_replay(args)
        return

    if args.collect_hugsim_shard:
        _collect_hugsim_shard_and_export(args)
        return

    if args.scene is None:
        raise SystemExit("--scene is required unless --from-shard or --collect-hugsim-shard is used")
    scene = int(args.scene)
    RLReconEnv, SparseDriveV2Policy = _lazy_import_runtime()
    ckpt_path = _resolve_repo_path(str(args.ckpt))
    config_path = _resolve_repo_path(str(args.config))
    cfg = _load_yaml(config_path)
    env_cfg = cfg.get("env", {}) if isinstance(cfg, dict) else {}
    reward_cfg = env_cfg.get("reward", {}) if isinstance(env_cfg, dict) else {}

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"SparseDriveV2 ckpt not found: {ckpt_path}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    auto_paths = _build_auto_run_paths(scene=scene, timestamp=ts)
    out_path = args.out or auto_paths["video_path"]
    traj_csv = args.traj_csv or auto_paths["traj_csv"]
    traj_plot = args.traj_plot or auto_paths["traj_plot"]
    _ensure_parent(out_path)
    _ensure_parent(traj_csv)
    _ensure_parent(traj_plot)
    _write_run_manifest(
        manifest_path=auto_paths["run_manifest"],
        scene=scene,
        config_path=config_path,
        ckpt_path=ckpt_path,
        timestamp=ts,
        extra_lines=[
            f"artifacts_dir={auto_paths['artifacts_dir']}",
            f"video={out_path}",
            f"traj_csv={traj_csv}",
            f"traj_plot={traj_plot}",
            f"start_frame={int(args.start_frame)}",
            f"step_frames={int(args.step_frames)}",
            f"cuda={int(args.cuda)}",
            f"mode_select={args.mode_select or 'greedy'}",
            f"reward_detail_format={args.reward_detail_format}",
        ],
    )

    env = RLReconEnv(
        cuda=int(args.cuda),
        scene=scene,
        reward_cfg=reward_cfg,
        debug=bool(args.debug),
        render_w=(int(args.render_w) if args.render_w is not None else None),
        render_h=(int(args.render_h) if args.render_h is not None else None),
    )

    obs, _info = env.reset(scene=scene, start_frame=int(args.start_frame), step_frames=int(args.step_frames))
    sim = getattr(env, "env")
    setattr(sim, "use_expert_height", bool(args.expert_high))

    policy = SparseDriveV2Policy(
        ckpt_path=str(ckpt_path),
        device=(f"cuda:{int(args.cuda)}" if torch.cuda.is_available() else "cpu"),
        execute_mode="first_step",
        rl_lr=1e-5,
    )

    step_dt_s = float(getattr(sim, "step_frames", int(args.step_frames))) * 0.1
    if step_dt_s <= 0:
        raise RuntimeError("Invalid step dt")
    max_steps = None if args.duration_s is None else max(1, int(round(float(args.duration_s) / step_dt_s)))
    fps = float(args.fps) if args.fps is not None else (1.0 / step_dt_s)

    print("==== generate_video_sparsedrive_v2 ====")
    print(f"scene={scene} start_frame={int(args.start_frame)} step_frames={int(args.step_frames)}")
    print(f"ckpt={ckpt_path}")
    live_mode_select = args.mode_select or "greedy"
    print(f"mode_select={live_mode_select}")
    print(f"use_expert_height={bool(args.expert_high)}")
    if args.duration_s is None:
        print(f"duration_s=until_done max_steps=none step_dt_s={step_dt_s:.3f} fps={fps:.3f}")
    else:
        print(f"duration_s={float(args.duration_s):.3f} max_steps={max_steps} step_dt_s={step_dt_s:.3f} fps={fps:.3f}")
    print(f"out_video={out_path}")
    print(f"out_traj_csv={traj_csv}")
    print(f"out_traj_plot={traj_plot}")

    writer = imageio.get_writer(
        out_path,
        mode="I",
        fps=float(fps),
        macro_block_size=1,
        codec="libx264",
        ffmpeg_log_level="error",
        input_params=["-framerate", str(float(fps))],
        output_params=["-pix_fmt", "yuv420p"],
    )

    rows: List[Dict[str, float | int]] = []
    ego_xz: List[List[float]] = []
    expert_xz_online: List[List[float]] = []
    online_summary_rows: List[Dict[str, float | int]] = []
    online_rollout_rows: List[Dict[str, float | int]] = []
    reward_rows: List[Dict[str, Any]] = []
    debug_shard: Dict[str, List[Any]] = {"reward": [], "done": [], "replay": [], "meta": []}
    reward_sum = 0.0
    rendered_frames: List[np.ndarray] = []
    env_cache = _load_env_cache(scene=scene)

    start_pose = np.asarray(getattr(sim, "start_ego"), dtype=np.float64)
    ego_xz.append([float(start_pose[0, 3]), float(start_pose[2, 3])])
    try:
        start_frame_expert_xz = _load_expert_front_xz_for_frame(
            scene=scene,
            start_frame=int(args.start_frame),
            frame_idx=int(args.start_frame),
        )
        _append_online_expert_xz(expert_xz_online, start_frame_expert_xz)
    except Exception as e:
        print(f"[traj-online] failed to load expert start pose: {e}")

    done = False
    steps = 0
    frames = 0
    first = _grid_frame(obs)
    first = _overlay_debug_text(
        first,
        [
            f"scene={scene:03d} step=0 frame={int(getattr(sim, 'now_frame', -1))}",
            "reward=NA cum_reward=0.000",
        ],
    )
    writer.append_data(first)
    rendered_frames.append(first.copy())
    frames += 1

    while (max_steps is None or steps < max_steps) and not done:
        obs_in = _ensure_obs_for_sparsedrive_v2(obs, sim)
        start_ego = np.asarray(getattr(sim, "start_ego"), dtype=np.float64).copy()
        now_frame = int(getattr(sim, "now_frame", -1))

        action, logp, replay = policy.sample_sparsedrivev2_with_replay(
            obs_in,
            mode_idx=-1,
            mode_select=str(live_mode_select),
        )

        traj_xyyaw = _traj_xyyaw_from_replay(replay)
        traj_front = _local_plan_to_front_frame(start_ego, traj_xyyaw)

        logp_v = float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp)
        print(f"[plan-v2] step={steps} frame={now_frame} shape={traj_xyyaw.shape}")
        print(np.array2string(traj_xyyaw, precision=6, suppress_small=False))

        # Prediction based on the plan's first point in front-start frame.
        pred_plan_front_xz = np.asarray([float(traj_front[0, 0]), float(traj_front[0, 2])], dtype=np.float64)

        rows.append(
            {
                "step": int(steps),
                "frame": int(now_frame),
                "plan_idx": 0,
                "cmd_idx": int(replay.get("cmd_idx", -1)),
                "mode_idx": int(replay.get("mode_idx", -1)),
                "logp": float(logp_v),
                "local_x": float(traj_xyyaw[0, 0]),
                "local_y": float(traj_xyyaw[0, 1]),
                "local_yaw": float(traj_xyyaw[0, 2]),
                "front_x": float(traj_front[0, 0]),
                "front_y": float(traj_front[0, 1]),
                "front_z": float(traj_front[0, 2]),
                "front_yaw": float(traj_front[0, 3]),
            }
        )

        # Feed the full planned trajectory to simulator so PDM tracks actual planner output.
        setattr(sim, "_external_plan_local_xyyaw", np.asarray(traj_xyyaw, dtype=np.float64).copy())

        obs, _reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)
        info = dict(_info or {})
        reward_v = float(_reward)
        reward_sum += reward_v

        pose_after = np.asarray(obs.get("ego_pose", getattr(sim, "start_ego")), dtype=np.float64)
        ego_xz.append([float(pose_after[0, 3]), float(pose_after[2, 3])])

        pred_xz = pred_plan_front_xz
        real_xz = np.asarray([float(pose_after[0, 3]), float(pose_after[2, 3])], dtype=np.float64)
        err_xz = float(np.linalg.norm(pred_xz - real_xz, ord=2))

        frame_after = int(getattr(sim, "now_frame", -1))
        tracked_first_local = np.asarray(getattr(sim, "_tracked_first_step_xyyaw", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(3)
        executed_first_local = np.asarray(
            getattr(sim, "_executed_first_step_xyyaw", tracked_first_local),
            dtype=np.float64,
        ).reshape(3)
        tracked_rollout = np.asarray(getattr(sim, "_tracked_rollout_local_xyyaw", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64)
        actual_local = _relative_local_xyyaw(start_ego, pose_after)
        tracked_first_front = _local_plan_to_front_frame(start_ego, tracked_first_local.reshape(1, 3))[0]
        try:
            expert_after_xz = _load_expert_front_xz_for_frame(
                scene=scene,
                start_frame=int(args.start_frame),
                frame_idx=int(frame_after),
            )
            _append_online_expert_xz(expert_xz_online, expert_after_xz)
        except Exception as e:
            print(f"[traj-online] failed to load expert pose for frame={frame_after}: {e}")
            expert_after_xz = np.asarray([np.nan, np.nan], dtype=np.float64)
        cmd_obs, vel_obs, acc_obs = _extract_status_from_obs(obs)
        cmd_ds, vel_ds, acc_ds = _dataset_status_from_sim(sim, frame_after)
        reward_rows.append(
            {
                "step": int(steps),
                "frame_before": int(now_frame),
                "frame_after": int(frame_after),
                "reward": float(reward_v),
                "cum_reward": float(reward_sum),
                "done": bool(done),
                "done_reason": str(info.get("done_reason", "")),
                "dynamic_collision": bool(info.get("dynamic_collision", False)),
                "static_collision": bool(info.get("static_collision", False)),
                "cost_reward": float(info.get("cost_reward", 0.0)),
                "progress_reward": float(info.get("progress_reward", 0.0)),
                "craft_safety_cost": float(info.get("craft_safety_cost", 0.0)),
                "craft_forward_progress": float(info.get("craft_forward_progress", 0.0)),
            }
        )
        debug_shard["reward"].append(float(reward_v))
        debug_shard["done"].append(bool(done))
        debug_shard["replay"].append(replay)
        debug_shard["meta"].append({"step": int(steps), "frame_before": int(now_frame), "frame_after": int(frame_after), "info": info})

        online_summary_rows.append(
            {
                "step": int(steps),
                "frame_before": int(now_frame),
                "frame_after": int(frame_after),
                "plan_tracked_xy_err": float(np.linalg.norm(traj_xyyaw[0, :2] - tracked_first_local[:2])),
                "tracked_executed_xy_err": float(np.linalg.norm(tracked_first_local[:2] - executed_first_local[:2])),
                "plan_actual_front_xz_err": float(np.linalg.norm(pred_xz - real_xz, ord=2)),
                "tracked_actual_front_xz_err": float(np.linalg.norm(tracked_first_front[[0, 2]] - np.asarray([real_xz[0], real_xz[1]], dtype=np.float64), ord=2)),
                "expert_actual_front_xz_err": float(np.linalg.norm(expert_after_xz - real_xz, ord=2)) if np.isfinite(expert_after_xz).all() else float("nan"),
                "tracked_local_x": float(tracked_first_local[0]),
                "tracked_local_y": float(tracked_first_local[1]),
                "tracked_local_yaw": float(tracked_first_local[2]),
                "executed_local_x": float(executed_first_local[0]),
                "executed_local_y": float(executed_first_local[1]),
                "executed_local_yaw": float(executed_first_local[2]),
                "actual_local_x": float(actual_local[0]),
                "actual_local_y": float(actual_local[1]),
                "actual_local_yaw": float(actual_local[2]),
            }
        )
        for pt_idx in range(int(traj_xyyaw.shape[0])):
            tracked_pt = tracked_rollout[pt_idx] if tracked_rollout.ndim == 2 and pt_idx < tracked_rollout.shape[0] else np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
            online_rollout_rows.append(
                {
                    "step": int(steps),
                    "point_idx": int(pt_idx),
                    "plan_local_x": float(traj_xyyaw[pt_idx, 0]),
                    "plan_local_y": float(traj_xyyaw[pt_idx, 1]),
                    "plan_local_yaw": float(traj_xyyaw[pt_idx, 2]),
                    "tracked_local_x": float(tracked_pt[0]),
                    "tracked_local_y": float(tracked_pt[1]),
                    "tracked_local_yaw": float(tracked_pt[2]),
                }
            )
        
        print(
            "[pose-check-v2] "
            f"step={steps} frame={now_frame} "
            f"action(dx,dy,dyaw)=({float(action[0]):.6f},{float(action[1]):.6f},{float(action[2]):.6f}) "
            f"tracked_first_local=({tracked_first_local[0]:.6f},{tracked_first_local[1]:.6f},{tracked_first_local[2]:.6f}) "
            f"executed_first_local=({executed_first_local[0]:.6f},{executed_first_local[1]:.6f},{executed_first_local[2]:.6f}) "
            f"actual_local=({actual_local[0]:.6f},{actual_local[1]:.6f},{actual_local[2]:.6f}) "
            f"pred_src=plan_first_point "
            f"pred_next_xz=({pred_xz[0]:.6f},{pred_xz[1]:.6f}) "
            f"real_next_xz=({real_xz[0]:.6f},{real_xz[1]:.6f}) "
            f"l2_err={err_xz:.9f}"
        )
        print(
            "[status-check-v2] "
            f"step={steps} frame_after={frame_after} "
            f"command_obs={np.array2string(cmd_obs, precision=6, suppress_small=False)} "
            f"vel_obs={np.array2string(vel_obs, precision=6, suppress_small=False)} "
            f"acc_obs={np.array2string(acc_obs, precision=6, suppress_small=False)} "
            # f"command_dataset={np.array2string(cmd_ds, precision=6, suppress_small=False)} "
            f"vel_dataset={np.array2string(vel_ds, precision=6, suppress_small=False)} "
            f"acc_dataset={np.array2string(acc_ds, precision=6, suppress_small=False)}"
        )

        snap = env_cache.get(int(frame_after), None)
        world_pose = _world_pose_from_sim(sim)
        dbg = [
            f"scene={scene:03d} step={int(steps)} frame={int(frame_after)}",
            f"reward={reward_v:.5f} cum_reward={reward_sum:.5f}",
            f"progress={float(info.get('progress_reward', 0.0)):.5f} cost={float(info.get('cost_reward', 0.0)):.5f}",
            f"dyn_col={bool(info.get('dynamic_collision', False))} static_col={bool(info.get('static_collision', False))} done_reason={info.get('done_reason', '')}",
        ]
        frame_out = _overlay_debug_text(_grid_frame(obs), dbg)
        if bool(args.draw_collision_bev):
            frame_out, hit_tokens = _draw_collision_bev(frame_out, world_pose=world_pose, snap=snap, view_m=25.0)
            if len(hit_tokens) > 0:
                dbg2 = [f"collision_tokens={','.join(hit_tokens[:2])}"]
                frame_out = _overlay_debug_text(frame_out, dbg + dbg2)
                if isinstance(reward_rows[-1], dict):
                    reward_rows[-1]["collision_tokens"] = "|".join(hit_tokens)
        writer.append_data(frame_out)
        rendered_frames.append(frame_out.copy())
        frames += 1
        steps += 1

    writer.close()

    fieldnames = [
        "step",
        "frame",
        "plan_idx",
        "cmd_idx",
        "mode_idx",
        "logp",
        "local_x",
        "local_y",
        "local_yaw",
        "front_x",
        "front_y",
        "front_z",
        "front_yaw",
    ]
    with open(traj_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    reward_csv = os.path.splitext(traj_csv)[0] + "_step_reward.csv"
    _ensure_parent(reward_csv)
    if reward_rows:
        with open(reward_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({str(k) for row in reward_rows for k in row.keys()})
            rw = csv.DictWriter(f, fieldnames=fieldnames)
            rw.writeheader()
            rw.writerows(reward_rows)
        print(f"reward_csv_saved={reward_csv}")

    reward_plot = os.path.splitext(traj_plot)[0] + "_reward_curve.png"
    try:
        if reward_rows:
            xs = [int(r["step"]) for r in reward_rows]
            ys = [float(r["reward"]) for r in reward_rows]
            cs = [float(r["cum_reward"]) for r in reward_rows]
            fig, ax1 = plt.subplots(figsize=(9, 4), dpi=150)
            ax1.plot(xs, ys, color="#d62728", label="step_reward")
            ax1.axhline(0.0, color="#888888", linewidth=1.0)
            ax2 = ax1.twinx()
            ax2.plot(xs, cs, color="#1f77b4", label="cum_reward")
            ax1.set_xlabel("step")
            ax1.set_ylabel("reward")
            ax2.set_ylabel("cum_reward")
            ax1.set_title(f"Scene {scene:03d} Reward Curve")
            fig.tight_layout()
            fig.savefig(reward_plot)
            plt.close(fig)
            print(f"reward_plot_saved={reward_plot}")
    except Exception as e:
        print(f"[reward-plot] failed: {e}")

    shard_path = os.path.splitext(traj_csv)[0] + "_debug_shard.pt"
    try:
        torch.save(debug_shard, shard_path)
        print(f"debug_shard_saved={shard_path}")
    except Exception as e:
        print(f"[debug-shard] failed: {e}")
    if str(args.reward_detail_format) == "ipynb":
        detail_ipynb = os.path.splitext(traj_csv)[0] + "_reward_detail.ipynb"
        try:
            _save_reward_detail_notebook(
                out_path=detail_ipynb,
                scene=scene,
                reward_rows=reward_rows,
                debug_shard=debug_shard,
                reward_cfg=reward_cfg,
                video_path=out_path,
                reward_csv_path=reward_csv,
                debug_shard_path=shard_path,
            )
            print(f"reward_detail_notebook_saved={detail_ipynb}")
        except Exception as e:
            print(f"[reward-detail-notebook] failed: {e}")

    if bool(args.save_keyframes) and reward_rows and rendered_frames:
        try:
            key_root = os.path.splitext(traj_plot)[0] + "_keyframes"
            os.makedirs(key_root, exist_ok=True)
            order = sorted(range(len(reward_rows)), key=lambda i: float(reward_rows[i]["reward"]))
            for idx in order[: max(1, int(args.keyframes_k))]:
                step_i = int(reward_rows[idx]["step"])
                img_i = min(step_i + 1, len(rendered_frames) - 1)
                out_png = os.path.join(key_root, f"step{step_i:03d}_reward{float(reward_rows[idx]['reward']):+.4f}.png")
                imageio.imwrite(out_png, rendered_frames[img_i])
            print(f"keyframes_saved={key_root}")
        except Exception as e:
            print(f"[keyframes] failed: {e}")

    ego_xz_np = np.asarray(ego_xz, dtype=np.float64)
    expert_xz_np = np.asarray(expert_xz_online, dtype=np.float64)
    print(f"[traj-v2] ego_xz shape={ego_xz_np.shape}")
    print(np.array2string(ego_xz_np, precision=6, suppress_small=False))
    print(f"[traj-v2-online] expert_xz shape={expert_xz_np.shape}")
    print(np.array2string(expert_xz_np, precision=6, suppress_small=False))

    if expert_xz_np.shape[0] >= 2 and ego_xz_np.shape[0] >= 2:
        saved = _save_traj_plot_xz(scene=scene, expert_xz=expert_xz_np, ego_xz=ego_xz_np, out_path=traj_plot)
        if saved:
            print(f"traj_plot_saved={traj_plot}")
    else:
        print("[traj-plot] skip export due to insufficient online trajectory points")

    try:
        stats_paths = _build_online_step_stats_paths(traj_plot)
        stats_module = _load_scene99_step_summary_module()
        rollout_by_step = stats_module.build_step_rollout_arrays(online_rollout_rows)
        per_step_rows, aggregate = stats_module.summarize_step_tracking(online_summary_rows, rollout_by_step)
        stats_module._save_csv_rows(per_step_rows, stats_paths["per_step_csv"])
        stats_module._save_csv_row(aggregate, stats_paths["aggregate_csv"])
        _save_online_rollout_points_csv(online_rollout_rows, stats_paths["rollout_csv"])
        stats_module._save_overlay_plot(rollout_by_step, stats_paths["overlay_svg"])
        stats_module._save_error_hist_plot(per_step_rows, stats_paths["error_hist_svg"])
        stats_module._save_worst_cases_plot(rollout_by_step, per_step_rows, stats_paths["worst_svg"])
        print(f"[online-step-stats] num_steps={int(aggregate['num_steps'])}")
        print(f"[online-step-stats] mean_first_point_plan_tracked_xy_err_m={float(aggregate['mean_first_point_plan_tracked_xy_err_m']):.9f}")
        print(f"[online-step-stats] mean_rollout_mean_xy_err_m={float(aggregate['mean_rollout_mean_xy_err_m']):.9f}")
        print(f"[online-step-stats] mean_expert_actual_front_xz_err_m={float(aggregate['mean_expert_actual_front_xz_err_m']):.9f}")
        print(f"online_step_summary_saved={stats_paths['per_step_csv']}")
        print(f"online_step_aggregate_saved={stats_paths['aggregate_csv']}")
        print(f"online_rollout_points_saved={stats_paths['rollout_csv']}")
        print(f"online_rollout_overlay_saved={stats_paths['overlay_svg']}")
        print(f"online_error_hist_saved={stats_paths['error_hist_svg']}")
        print(f"online_worst_steps_saved={stats_paths['worst_svg']}")
    except Exception as e:
        print(f"[online-step-stats] failed to export online stats: {e}")

    print(f"done={done} steps={steps} frames={frames}")
    print(f"video_saved={out_path}")
    print(f"traj_saved={traj_csv}")
    print("==== all done ====")


if __name__ == "__main__":
    main()
