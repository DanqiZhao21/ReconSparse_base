
'''
# 注意改动一下scene 以及output 的地址
python /root/clone/ReconDreamer-RL/tools/smalltool/visualize/visualize_sparsedrivev2_grpo_CRAFT_online.py \
  --config script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_craft.yaml \
  --ckpt /root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/20260519_grpo_only_craft_ver01_latest.ckpt \
  --scene 123 \
  --out /root/clone/ReconDreamer-RL/outputs/visualize/CRAFT_online_visualize/online_sdv2_craft_grpo_scene123 \
  --num-candidates 16 \
  --top-k 5 \
  --candidate-select topk \
  --mode-select greedy
'''
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CandidateVisualStyle:
    candidate_index: int
    rank: int
    score: float
    alpha: float
    linewidth: float
    color: tuple[int, int, int]
    is_top_k: bool


def _maybe_prepend_env_path(key: str, values: list[str]) -> None:
    existing = os.environ.get(key, "")
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
        os.environ[key] = os.pathsep.join(parts)


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
    _maybe_prepend_env_path("CPATH", include_dirs)
    _maybe_prepend_env_path("CPLUS_INCLUDE_PATH", include_dirs)
    _maybe_prepend_env_path("LIBRARY_PATH", library_dirs)
    _maybe_prepend_env_path("LD_LIBRARY_PATH", library_dirs)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_REPO_ROOT / ".cache" / "torch_extensions"))
    Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)


_prepare_cuda_extension_env()

from framework.utils.repo_paths import resolve_repo_path

_DEFAULT_CKPT = _REPO_ROOT / "egoADs" / "SparseDriveV2" / "ckpt" / "sparsedrive_navsimv2.ckpt"
_DEFAULT_OUT_DIR = _REPO_ROOT / "outputs" / "visualize" / "sparsedrivev2_grpo_craft_online"


@dataclass(frozen=True)
class OutputPaths:
    video: Path
    frames_dir: Path
    bev_dir: Path
    scores_dir: Path


def _stable_desc_sort_indices(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.argsort(-arr, kind="stable").astype(np.int64, copy=False)


def candidate_visual_styles(
    *,
    scores: Sequence[float],
    top_k: int,
    min_alpha: float = 0.12,
    max_alpha: float = 0.70,
) -> list[dict[str, Any]]:
    ranked_indices = _stable_desc_sort_indices(scores)
    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    total = int(score_arr.shape[0])
    if total <= 0:
        return []

    top_k_count = max(1, min(int(top_k), total))
    out: list[dict[str, Any]] = []
    denom = max(1, total - 1)
    for rank, candidate_index in enumerate(ranked_indices.tolist(), start=1):
        score_rank = rank - 1
        fade = float(score_rank) / float(denom)
        alpha = float(max(min_alpha, max_alpha - 0.50 * fade))
        linewidth = float(max(1.0, 2.8 - 1.25 * fade))
        color = _score_to_rgb(float(score_arr[int(candidate_index)]), min_score=float(score_arr.min()), max_score=float(score_arr.max()))
        out.append(
            {
                "candidate_index": int(candidate_index),
                "rank": int(rank),
                "score": float(score_arr[int(candidate_index)]),
                "alpha": float(alpha),
                "linewidth": float(linewidth),
                "color": tuple(int(v) for v in color),
                "is_top_k": bool(rank <= top_k_count),
            }
        )
    return out


def _score_to_rgb(score: float, *, min_score: float, max_score: float) -> tuple[int, int, int]:
    if not math.isfinite(score):
        return (180, 180, 180)
    span = max(1.0e-6, float(max_score) - float(min_score))
    norm = float(np.clip((float(score) - float(min_score)) / span, 0.0, 1.0))
    low = np.asarray([210, 220, 230], dtype=np.float32)
    high = np.asarray([235, 96, 52], dtype=np.float32)
    rgb = (1.0 - norm) * low + norm * high
    return tuple(int(np.clip(v, 0, 255)) for v in np.rint(rgb))


def build_candidate_score_payload(
    *,
    scene: int,
    step: int,
    frame_idx: int,
    sample_token: str,
    traj_xyyaw: np.ndarray,
    scores: Sequence[float],
    score_logits: Sequence[float] | None,
    mode_indices: Sequence[int] | None,
    top_k: int,
) -> dict[str, Any]:
    traj = np.asarray(traj_xyyaw, dtype=np.float32)
    if traj.ndim != 3 or traj.shape[-1] < 3:
        raise RuntimeError(f"Expected traj_xyyaw shape (candidates,horizon,3), got {tuple(traj.shape)}")

    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if int(score_arr.shape[0]) != int(traj.shape[0]):
        raise RuntimeError(
            "Candidate count mismatch between scores and trajectories: "
            f"scores={int(score_arr.shape[0])} traj={int(traj.shape[0])}"
        )
    logits_arr = None if score_logits is None else np.asarray(score_logits, dtype=np.float32).reshape(-1)
    if logits_arr is not None and int(logits_arr.shape[0]) != int(traj.shape[0]):
        raise RuntimeError("score_logits length mismatch")
    mode_idx_arr = None if mode_indices is None else np.asarray(mode_indices, dtype=np.int64).reshape(-1)
    if mode_idx_arr is not None and int(mode_idx_arr.shape[0]) != int(traj.shape[0]):
        raise RuntimeError("mode_indices length mismatch")

    styles = candidate_visual_styles(scores=score_arr, top_k=top_k)
    style_by_candidate = {int(item["candidate_index"]): dict(item) for item in styles}
    ranked_indices = [int(item["candidate_index"]) for item in styles]
    payload_candidates: list[dict[str, Any]] = []
    for candidate_index in ranked_indices:
        visual = dict(style_by_candidate[int(candidate_index)])
        item = {
            "candidate_index": int(candidate_index),
            "rank": int(visual["rank"]),
            "score": float(visual["score"]),
            "visual": {
                "alpha": float(visual["alpha"]),
                "linewidth": float(visual["linewidth"]),
                "color": list(visual["color"]),
                "is_top_k": bool(visual["is_top_k"]),
            },
            "traj_xyyaw": traj[int(candidate_index)].astype(np.float32).tolist(),
        }
        if logits_arr is not None:
            item["score_logit"] = float(logits_arr[int(candidate_index)])
        if mode_idx_arr is not None:
            item["mode_index"] = int(mode_idx_arr[int(candidate_index)])
        payload_candidates.append(item)

    return {
        "scene": int(scene),
        "step": int(step),
        "frame_idx": int(frame_idx),
        "sample_token": str(sample_token),
        "top_k": int(max(1, int(top_k))),
        "top_k_candidate_indices": [int(item["candidate_index"]) for item in styles[: max(1, int(top_k))]],
        "candidates": payload_candidates,
    }


def overlay_top_right_inset(
    frame: np.ndarray,
    inset: np.ndarray,
    *,
    inset_width: int = 320,
    margin: int = 12,
    border_px: int = 2,
) -> np.ndarray:
    out = np.asarray(frame).copy()
    inset_arr = np.asarray(inset)
    if out.ndim != 3 or inset_arr.ndim != 3:
        raise RuntimeError("overlay_top_right_inset expects HWC RGB arrays")
    if int(out.shape[2]) < 3 or int(inset_arr.shape[2]) < 3:
        raise RuntimeError("overlay_top_right_inset expects 3-channel images")

    try:
        import cv2
    except Exception:
        return out

    h, w = out.shape[:2]
    inset_w = max(16, min(int(inset_width), max(16, w - 2 * int(margin))))
    scale = float(inset_w) / float(max(1, int(inset_arr.shape[1])))
    inset_h = max(16, int(round(float(inset_arr.shape[0]) * scale)))
    max_h = max(16, h - 2 * int(margin))
    if inset_h > max_h:
        inset_h = max_h
        scale = float(inset_h) / float(max(1, int(inset_arr.shape[0])))
        inset_w = max(16, int(round(float(inset_arr.shape[1]) * scale)))

    resized = cv2.resize(inset_arr, (int(inset_w), int(inset_h)), interpolation=cv2.INTER_LINEAR)
    x1 = int(w - margin - inset_w)
    y1 = int(margin)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x1 + int(inset_w))
    y2 = min(h, y1 + int(inset_h))
    patch = resized[: max(0, y2 - y1), : max(0, x2 - x1)]
    if patch.size == 0:
        return out

    roi = out[y1:y2, x1:x2]
    out[y1:y2, x1:x2] = patch
    if int(border_px) > 0:
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (245, 245, 245), int(border_px))
    return out


def build_default_paths(*, out_dir: str | Path, scene: int) -> OutputPaths:
    root = Path(out_dir)
    return OutputPaths(
        video=root / f"scene_{int(scene):03d}.mp4",
        frames_dir=root / "frames",
        bev_dir=root / "bev",
        scores_dir=root / "scores",
    )


def write_score_payload(out_dir: str | Path, payload: dict[str, Any]) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    step = int(payload.get("step", 0))
    path = root / f"step_{step:06d}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _grid_frame(observation: dict[str, np.ndarray]) -> np.ndarray:
    keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
    imgs = [observation[k] for k in keys]
    row1 = np.concatenate(imgs[:3], axis=1)
    row2 = np.concatenate(imgs[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)


def _overlay_debug_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
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


def _lazy_import_runtime() -> tuple[Any, Any]:
    from framework.env_wrapper import RLReconEnv
    from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy

    return RLReconEnv, SparseDriveV2Policy


def _obs_for_policy(obs: dict[str, Any], sim: Any) -> dict[str, Any]:
    out = dict(obs)
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


def _extract_plan_and_scores(policy: Any, replay: dict[str, Any], *, num_candidates: int, candidate_select: str) -> dict[str, Any]:
    candidate_fn = getattr(policy, "sample_counterfactual_trajectories_from_replay_batch", None)
    if not callable(candidate_fn):
        raise RuntimeError("SparseDriveV2Policy missing sample_counterfactual_trajectories_from_replay_batch")
    candidates = candidate_fn([replay], num_candidates=int(num_candidates), candidate_select=str(candidate_select))
    traj_xyyaw = np.asarray(candidates["traj_xyyaw"].detach().cpu().numpy(), dtype=np.float32)[0]
    raw_score_logits = candidates.get("score_logits", np.zeros((1, traj_xyyaw.shape[0]), dtype=np.float32))
    if torch.is_tensor(raw_score_logits):
        score_logits = np.asarray(raw_score_logits.detach().cpu().numpy(), dtype=np.float32)[0]
    else:
        score_logits = np.asarray(raw_score_logits, dtype=np.float32)[0]
    log_probs = np.asarray(candidates["log_probs"].detach().cpu().numpy(), dtype=np.float32)[0]
    return {
        "traj_xyyaw": traj_xyyaw,
        "score_logits": score_logits,
        "log_probs": log_probs,
        "mode_indices": np.asarray(candidates["mode_indices"].detach().cpu().numpy(), dtype=np.int64)[0],
        "candidate_bundle": candidates,
    }


def _build_bev_sample_detail(policy: Any, replay: dict[str, Any], traj_xyyaw: np.ndarray) -> dict[str, Any]:
    scorer = getattr(policy, "_ensure_counterfactual_scorer_backend", None)
    if callable(scorer):
        backend = scorer()
        detail_backends = [
            backend,
            getattr(backend, "_delegate", None),
            getattr(getattr(backend, "_pdm", None), "_delegate", None),
        ]
        for detail_backend in detail_backends:
            score_with_details = getattr(detail_backend, "score_with_details", None)
            if callable(score_with_details):
                _scores, details = score_with_details([replay], torch.as_tensor(traj_xyyaw[None, ...], dtype=torch.float32))
                if len(details) > 0:
                    detail = dict(details[0])
                    detail["render_layers"] = detail.get("render_layers", {}) or {}
                    return detail
    return {
        "sample_token": str(replay.get("sample_token", "")),
        "gt_xy": np.asarray(replay.get("gt_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32),
        "gt_history_xy": np.asarray(replay.get("gt_history_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32),
        "ego_history_xy": np.asarray(replay.get("ego_history_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32),
        "scene_objects": [],
        "render_layers": {},
        "map_patch_radius_m": 20.0,
    }


def _ego_history_xy_in_current_frame(history_poses: list[np.ndarray], current_pose: np.ndarray) -> np.ndarray:
    if len(history_poses) <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    cur = np.asarray(current_pose, dtype=np.float64)
    if cur.shape != (4, 4):
        return np.zeros((0, 2), dtype=np.float32)
    inv_cur = np.linalg.inv(cur)
    rows: list[list[float]] = []
    for pose in history_poses:
        arr = np.asarray(pose, dtype=np.float64)
        if arr.shape != (4, 4):
            continue
        rel = inv_cur @ arr
        rows.append([float(rel[0, 3]), float(rel[1, 3])])
    return np.asarray(rows, dtype=np.float32)


def _bev_points_to_px(
    points_xy: np.ndarray,
    *,
    width: int,
    height: int,
    patch_radius: float,
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    cx = float(width) * 0.5
    cy = float(height) * 0.55
    scale = float(min(width, height) * 0.40 / max(1.0e-6, patch_radius))
    px = np.zeros((pts.shape[0], 2), dtype=np.int32)
    px[:, 0] = np.rint(cx - pts[:, 1] * scale).astype(np.int32)
    px[:, 1] = np.rint(cy - pts[:, 0] * scale).astype(np.int32)
    return px


def _draw_map_layers_on_bev(frame: np.ndarray, render_layers: dict[str, Any], *, patch_radius: float) -> None:
    try:
        import cv2
    except Exception:
        return

    h, w = frame.shape[:2]

    def to_px(points_xy: np.ndarray) -> np.ndarray:
        return _bev_points_to_px(points_xy, width=int(w), height=int(h), patch_radius=patch_radius)

    for polygon in render_layers.get("drivable_polygons", []) or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3:
            cv2.fillPoly(frame, [to_px(pts).reshape(-1, 1, 2)], (228, 226, 220))
    for polygon in render_layers.get("road_surface_polygons", []) or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3:
            cv2.fillPoly(frame, [to_px(pts).reshape(-1, 1, 2)], (220, 217, 211))
    for polygon in render_layers.get("lane_fill_polygons", []) or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3:
            cv2.fillPoly(frame, [to_px(pts).reshape(-1, 1, 2)], (212, 210, 206))
    for polygon in render_layers.get("walkway_polygons", []) or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3:
            cv2.fillPoly(frame, [to_px(pts).reshape(-1, 1, 2)], (199, 221, 194))
    for polygon in render_layers.get("crossing_polygons", []) or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3:
            cv2.polylines(frame, [to_px(pts).reshape(-1, 1, 2)], True, (170, 170, 165), 1, cv2.LINE_AA)
    for line in render_layers.get("road_edge_lines", []) or []:
        pts = np.asarray(line, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            cv2.polylines(frame, [to_px(pts).reshape(-1, 1, 2)], False, (140, 136, 130), 1, cv2.LINE_AA)
    for line in render_layers.get("lane_boundary_lines", []) or []:
        pts = np.asarray(line, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            cv2.polylines(frame, [to_px(pts).reshape(-1, 1, 2)], False, (168, 164, 156), 1, cv2.LINE_AA)
    for line in render_layers.get("lane_marking_lines", []) or []:
        pts = np.asarray(line, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            cv2.polylines(frame, [to_px(pts).reshape(-1, 1, 2)], False, (110, 110, 110), 1, cv2.LINE_AA)
    for line in render_layers.get("lane_centerlines", []) or []:
        pts = np.asarray(line, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            cv2.polylines(frame, [to_px(pts).reshape(-1, 1, 2)], False, (72, 149, 217), 1, cv2.LINE_AA)


def _draw_scene_objects_on_bev(frame: np.ndarray, scene_objects: Sequence[dict[str, Any]], *, patch_radius: float) -> None:
    try:
        import cv2
    except Exception:
        return

    h, w = frame.shape[:2]

    def to_px(points_xy: np.ndarray) -> np.ndarray:
        return _bev_points_to_px(points_xy, width=int(w), height=int(h), patch_radius=patch_radius)

    for obj in scene_objects or []:
        corners = np.asarray(obj.get("corners_xy", []), dtype=np.float32)
        if corners.ndim != 2 or corners.shape[0] < 3:
            continue
        color_name = str(obj.get("category", "car"))
        if "pedestrian" in color_name.lower():
            color = (142, 68, 173)
        elif "bicycle" in color_name.lower() or "motorcycle" in color_name.lower():
            color = (39, 174, 96)
        elif "bus" in color_name.lower() or "truck" in color_name.lower() or "trailer" in color_name.lower():
            color = (211, 84, 0)
        else:
            color = (231, 76, 60)
        cv2.fillPoly(frame, [to_px(corners).reshape(-1, 1, 2)], color)
        cv2.polylines(frame, [to_px(corners).reshape(-1, 1, 2)], True, (40, 40, 40), 1, cv2.LINE_AA)
        center = np.asarray(obj.get("center_xy", np.mean(corners[:, :2], axis=0)), dtype=np.float32).reshape(2)
        cpx = to_px(center[None, :])[0]
        cv2.circle(frame, tuple(cpx), 2, (25, 25, 25), -1, cv2.LINE_AA)
        yaw = float(obj.get("yaw_rad", 0.0))
        heading = np.asarray([[center[0], center[1]], [center[0] + 1.5 * np.cos(yaw), center[1] + 1.5 * np.sin(yaw)]], dtype=np.float32)
        hpx = to_px(heading)
        cv2.arrowedLine(frame, tuple(hpx[0]), tuple(hpx[1]), (30, 30, 30), 1, cv2.LINE_AA, tipLength=0.25)


def render_bev_debug_image(
    *,
    sample_detail: dict[str, Any],
    traj_xyyaw: np.ndarray,
    scores: Sequence[float],
    top_k: int,
    width: int = 420,
    height: int = 420,
) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("render_bev_debug_image requires opencv-python") from exc

    detail = dict(sample_detail)
    traj = np.asarray(traj_xyyaw, dtype=np.float32)
    if traj.ndim != 3 or traj.shape[-1] < 3:
        raise RuntimeError(f"Expected traj_xyyaw shape (candidates,horizon,3), got {tuple(traj.shape)}")

    gt_xy = np.asarray(detail.get("gt_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    gt_history_xy = np.asarray(detail.get("gt_history_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    ego_history_xy = np.asarray(detail.get("ego_history_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    scene_objects = list(detail.get("scene_objects", []) or [])
    render_layers = dict(detail.get("render_layers", {}) or {})
    if not render_layers and isinstance(detail.get("map_layers", None), dict):
        try:
            from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils

            render_layers = NuScenesScorerUtils._build_render_layers(dict(detail.get("map_layers", {}) or {}))
        except Exception:
            render_layers = {}
    patch_radius = float(detail.get("map_patch_radius_m", 20.0))

    out = np.full((int(height), int(width), 3), 248, dtype=np.uint8)
    out[:] = np.asarray([245, 243, 238], dtype=np.uint8)
    _draw_map_layers_on_bev(out, render_layers, patch_radius=patch_radius)
    _draw_scene_objects_on_bev(out, scene_objects, patch_radius=patch_radius)

    cx = float(width) * 0.5
    cy = float(height) * 0.55

    def to_px(points_xy: np.ndarray) -> np.ndarray:
        return _bev_points_to_px(points_xy, width=int(width), height=int(height), patch_radius=patch_radius)

    cv2.circle(out, (int(cx), int(cy)), 6, (35, 35, 35), -1)
    cv2.circle(out, (int(cx), int(cy)), 10, (255, 255, 255), 1)
    cv2.putText(out, "EGO", (int(cx) + 10, int(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 35, 35), 1, cv2.LINE_AA)

    if gt_history_xy.ndim == 2 and gt_history_xy.shape[0] > 0:
        pts = to_px(gt_history_xy)
        if pts.shape[0] >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (135, 135, 135), 2, cv2.LINE_AA)
    if gt_xy.ndim == 2 and gt_xy.shape[0] > 0:
        pts = to_px(gt_xy)
        if pts.shape[0] >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (18, 18, 18), 3, cv2.LINE_AA)
    if ego_history_xy.ndim == 2 and ego_history_xy.shape[0] > 0:
        pts = to_px(ego_history_xy)
        if pts.shape[0] >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (33, 102, 172), 3, cv2.LINE_AA)
        cv2.circle(out, tuple(pts[-1]), 5, (33, 102, 172), -1, cv2.LINE_AA)

    style_rows = candidate_visual_styles(scores=scores, top_k=top_k)
    for row in style_rows:
        idx = int(row["candidate_index"])
        cand = np.asarray(traj[idx], dtype=np.float32)
        cand_pts = to_px(cand[:, :2])
        color = tuple(int(v) for v in row["color"])
        alpha = float(row["alpha"])
        linewidth = max(1, int(round(float(row["linewidth"]))))
        if cand_pts.shape[0] >= 2:
            overlay = out.copy()
            cv2.polylines(overlay, [cand_pts.reshape(-1, 1, 2)], False, color, linewidth, cv2.LINE_AA)
            out[:] = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)
        if cand_pts.shape[0] > 0:
            cv2.circle(out, tuple(cand_pts[0]), 4, color, -1, cv2.LINE_AA)
            cv2.circle(out, tuple(cand_pts[-1]), 4, color, -1, cv2.LINE_AA)
            label = f"{row['rank']}:{float(row['score']):.2f}"
            cv2.putText(out, label, (int(cand_pts[-1, 0]) + 4, int(cand_pts[-1, 1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (width - 1, height - 1), (120, 120, 120), 1)
    cv2.putText(out, str(detail.get("sample_token", ""))[:18], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    return out


def _debug_first_candidate_xy(traj_xyyaw: np.ndarray) -> tuple[float, float] | None:
    arr = np.asarray(traj_xyyaw, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] < 2:
        return None
    return float(arr[0, 0, 0]), float(arr[0, 0, 1])


def _prime_sim_external_plan(sim: Any, traj_xyyaw: np.ndarray) -> None:
    arr = np.asarray(traj_xyyaw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
        raise RuntimeError(f"Expected selected traj_xyyaw shape (horizon,3), got {tuple(arr.shape)}")
    setattr(sim, "_external_plan_local_xyyaw", arr[:, :3].copy())


def run_online_scene(
    *,
    config_path: str | Path,
    ckpt_path: str | Path,
    scene: int,
    out_dir: str | Path,
    cuda: int = 0,
    start_frame: int = 0,
    step_frames: int = 5,
    fps: float | None = None,
    duration_s: float | None = None,
    num_candidates: int = 16,
    top_k: int = 5,
    candidate_select: str = "topk",
    mode_select: str = "greedy",
    debug: bool = False,
) -> dict[str, Any]:
    RLReconEnv, SparseDriveV2Policy = _lazy_import_runtime()
    config = _load_yaml(resolve_repo_path(str(config_path)))
    env_cfg = config.get("env", {}) if isinstance(config, dict) else {}
    reward_cfg = env_cfg.get("reward", {}) if isinstance(env_cfg, dict) else {}
    out_root = _ensure_dir(out_dir)
    paths = build_default_paths(out_dir=out_root, scene=scene)
    paths.video.parent.mkdir(parents=True, exist_ok=True)
    paths.frames_dir.mkdir(parents=True, exist_ok=True)
    paths.bev_dir.mkdir(parents=True, exist_ok=True)
    paths.scores_dir.mkdir(parents=True, exist_ok=True)

    env = RLReconEnv(
        cuda=int(cuda),
        scene=int(scene),
        reward_cfg=reward_cfg,
        debug=bool(debug),
        render_w=int(env_cfg.get("render_w", 1280)) if env_cfg.get("render_w", None) is not None else None,
        render_h=int(env_cfg.get("render_h", 720)) if env_cfg.get("render_h", None) is not None else None,
    )
    obs, _info = env.reset(scene=int(scene), start_frame=int(start_frame), step_frames=int(step_frames))
    sim = getattr(env, "env")
    policy = SparseDriveV2Policy(
        ckpt_path=str(resolve_repo_path(str(ckpt_path))),
        device=(f"cuda:{int(cuda)}" if torch.cuda.is_available() else "cpu"),
        execute_mode="first_step",
        rl_lr=1.0e-5,
    )

    step_dt_s = float(getattr(sim, "step_frames", int(step_frames))) * 0.1
    max_steps = None if duration_s is None else max(1, int(round(float(duration_s) / step_dt_s)))
    video_fps = float(fps) if fps is not None else (1.0 / step_dt_s)
    writer = imageio.get_writer(
        str(paths.video),
        mode="I",
        fps=float(video_fps),
        macro_block_size=1,
        codec="libx264",
        ffmpeg_log_level="error",
        input_params=["-framerate", str(float(video_fps))],
        output_params=["-pix_fmt", "yuv420p"],
    )

    score_paths: list[Path] = []
    rendered_frames: list[np.ndarray] = []
    ego_history_poses: list[np.ndarray] = []
    reward_sum = 0.0
    done = False
    steps = 0
    try:
        while (max_steps is None or steps < max_steps) and not done:
            obs_in = _obs_for_policy(obs, sim)
            action, logp, replay = policy.sample_sparsedrivev2_with_replay(
                obs_in,
                mode_idx=-1,
                mode_select=str(mode_select),
            )
            replay_out = _extract_plan_and_scores(policy, replay, num_candidates=num_candidates, candidate_select=candidate_select)
            traj_xyyaw = np.asarray(replay_out["traj_xyyaw"], dtype=np.float32)
            score_logits = np.asarray(replay_out["score_logits"], dtype=np.float32)
            mode_indices = np.asarray(replay_out["mode_indices"], dtype=np.int64)
            candidate_scores_tensor = policy.pdm_score_counterfactuals_from_replay_batch(
                [replay],
                torch.as_tensor(traj_xyyaw[None, ...], dtype=torch.float32),
            )
            if torch.is_tensor(candidate_scores_tensor):
                candidate_scores = np.asarray(candidate_scores_tensor.detach().cpu().numpy(), dtype=np.float32)[0]
            else:
                candidate_scores = np.asarray(candidate_scores_tensor, dtype=np.float32)[0]

            selected_traj_xyyaw = np.asarray(replay.get("traj_xyyaw", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
            _prime_sim_external_plan(sim, selected_traj_xyyaw)
            current_pose = np.asarray(getattr(sim, "start_ego", np.eye(4)), dtype=np.float64)
            if len(ego_history_poses) <= 0:
                ego_history_poses.append(current_pose.copy())

            sample_detail = _build_bev_sample_detail(policy, replay, traj_xyyaw)
            sample_detail["ego_history_xy"] = _ego_history_xy_in_current_frame(ego_history_poses, current_pose)
            bev_img = render_bev_debug_image(
                sample_detail=sample_detail,
                traj_xyyaw=traj_xyyaw,
                scores=candidate_scores,
                top_k=int(top_k),
                width=420,
                height=420,
            )
            imageio.imwrite(paths.bev_dir / f"step_{steps:06d}_bev.png", bev_img)
            candidate_payload = build_candidate_score_payload(
                scene=int(scene),
                step=int(steps),
                frame_idx=int(getattr(sim, "now_frame", -1)),
                sample_token=str(replay.get("sample_token", "")),
                traj_xyyaw=traj_xyyaw,
                scores=candidate_scores,
                score_logits=score_logits,
                mode_indices=mode_indices,
                top_k=int(top_k),
            )
            score_paths.append(write_score_payload(paths.scores_dir, candidate_payload))

            frame = _grid_frame(obs)
            frame = _overlay_debug_text(
                frame,
                [
                    f"scene={scene:03d} step={steps} frame={int(getattr(sim, 'now_frame', -1))}",
                    f"logp={float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp):.5f}",
                    f"reward={reward_sum:.5f}",
                ],
            )
            cand0 = _debug_first_candidate_xy(traj_xyyaw)
            if cand0 is not None:
                frame = _overlay_debug_text(frame, [f"cand0_xy=({cand0[0]:.2f},{cand0[1]:.2f})"])
            frame = overlay_top_right_inset(frame, bev_img, inset_width=360, margin=12, border_px=2)
            writer.append_data(frame)
            rendered_frames.append(frame.copy())

            obs, reward_v, terminated, truncated, _info = env.step(action)
            reward_sum += float(reward_v)
            done = bool(terminated or truncated)
            pose_after = np.asarray(getattr(sim, "start_ego", np.eye(4)), dtype=np.float64)
            ego_history_poses.append(pose_after.copy())
            steps += 1
    finally:
        writer.close()

    for idx, frame in enumerate(rendered_frames):
        imageio.imwrite(paths.frames_dir / f"step_{idx:06d}.png", frame)

    return {
        "video": str(paths.video),
        "frames_dir": str(paths.frames_dir),
        "bev_dir": str(paths.bev_dir),
        "scores_dir": str(paths.scores_dir),
        "score_json": [str(path) for path in score_paths],
        "steps": int(steps),
        "reward_sum": float(reward_sum),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SparseDriveV2 online GRPO/CRAFT visualizer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--out", type=str, default=str(_DEFAULT_OUT_DIR))
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--step-frames", type=int, default=5)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-select", type=str, default="topk", choices=["topk", "all"])
    parser.add_argument("--mode-select", type=str, default="greedy", choices=["greedy", "sample"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    result = run_online_scene(
        config_path=args.config,
        ckpt_path=args.ckpt,
        scene=int(args.scene),
        out_dir=args.out,
        cuda=int(args.cuda),
        start_frame=int(args.start_frame),
        step_frames=int(args.step_frames),
        fps=args.fps,
        duration_s=args.duration_s,
        num_candidates=int(args.num_candidates),
        top_k=int(args.top_k),
        candidate_select=str(args.candidate_select),
        mode_select=str(args.mode_select),
        debug=bool(args.debug),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
