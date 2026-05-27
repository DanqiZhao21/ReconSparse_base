from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Sim2Transform:
    scale: float
    rotation: np.ndarray
    translation_xy: np.ndarray
    rmse_m: float = 0.0

    @property
    def yaw_rad(self) -> float:
        return float(math.atan2(float(self.rotation[1, 0]), float(self.rotation[0, 0])))

    def transform_points(self, points_xy: Any) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float64)
        return float(self.scale) * (pts @ np.asarray(self.rotation, dtype=np.float64).T) + np.asarray(
            self.translation_xy, dtype=np.float64
        )

    def transform_yaw(self, yaw_rad: float) -> float:
        return float(yaw_rad) + self.yaw_rad


@dataclass(frozen=True)
class HUGSIMReconAlignment:
    official_scene_name: str
    recon_scene_id: int
    transform: Sim2Transform
    valid: bool
    reason: str = ""
    mode: str = "global"


def _identity_transform() -> Sim2Transform:
    return Sim2Transform(
        scale=1.0,
        rotation=np.eye(2, dtype=np.float64),
        translation_xy=np.zeros((2,), dtype=np.float64),
        rmse_m=float("inf"),
    )


def fit_sim2(src_xy: Any, dst_xy: Any) -> Sim2Transform:
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[1] != 2 or src.shape[0] < 2:
        raise ValueError(f"Expected matching Nx2 arrays, got {src.shape} and {dst.shape}")
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    u, _s, vt = np.linalg.svd(src_centered.T @ dst_centered)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    denom = float(np.sum(src_centered * src_centered))
    if denom <= 1.0e-12:
        raise ValueError("Cannot fit Sim(2) from degenerate source points")
    scale = float(np.sum((src_centered @ rot.T) * dst_centered) / denom)
    trans = dst_centroid - scale * (rot @ src_centroid)
    pred = scale * (src @ rot.T) + trans
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return Sim2Transform(
        scale=float(scale),
        rotation=rot.astype(np.float64),
        translation_xy=trans.astype(np.float64),
        rmse_m=rmse,
    )


def _dedupe_polyline(points_xy: Any) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected Nx2 polyline, got {pts.shape}")
    if int(pts.shape[0]) <= 1:
        return pts
    kept = [pts[0]]
    for idx in range(1, int(pts.shape[0])):
        if float(np.linalg.norm(pts[idx] - kept[-1])) > 1.0e-9:
            kept.append(pts[idx])
    return np.asarray(kept, dtype=np.float64)


def _resample_polyline_normalized(points_xy: Any, count: int) -> np.ndarray:
    pts = _dedupe_polyline(points_xy)
    if int(pts.shape[0]) < 2:
        raise ValueError("Cannot resample a degenerate polyline")
    count = max(2, int(count))
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg, dtype=np.float64)], axis=0)
    total = float(s[-1])
    if total <= 1.0e-9:
        raise ValueError("Cannot resample a zero-length polyline")
    target = np.linspace(0.0, total, count, dtype=np.float64)
    out = np.zeros((count, 2), dtype=np.float64)
    out[:, 0] = np.interp(target, s, pts[:, 0])
    out[:, 1] = np.interp(target, s, pts[:, 1])
    return out


def _load_hugsim_ground_path(hugsim_model_base: str | Path, official_scene_name: str) -> np.ndarray:
    path = Path(hugsim_model_base) / str(official_scene_name) / "ground_param.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    cam_poses = np.asarray(payload[0], dtype=np.float64)
    if cam_poses.ndim != 3 or cam_poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected HUGSIM ground poses with shape Nx4x4, got {cam_poses.shape}")
    return np.stack([cam_poses[:, 2, 3], -cam_poses[:, 0, 3]], axis=1).astype(np.float64)


def _load_recon_ego_path(recon_data_root: str | Path, recon_scene_id: int) -> np.ndarray:
    scene_dir = Path(recon_data_root) / f"{int(recon_scene_id):03d}" / "ego_pose"
    pts: list[np.ndarray] = []
    for path in sorted(scene_dir.glob("*.txt")):
        pose = np.asarray(np.loadtxt(path), dtype=np.float64).reshape(4, 4)
        pts.append(np.asarray([pose[0, 3], pose[1, 3]], dtype=np.float64))
    if not pts:
        raise ValueError(f"No Recon ego poses found under {scene_dir}")
    return np.asarray(pts, dtype=np.float64)


def _polyline_tangent(points_xy: Any, idx: int) -> np.ndarray:
    pts = _dedupe_polyline(points_xy)
    if int(pts.shape[0]) < 2:
        raise ValueError("Cannot compute tangent for a degenerate polyline")
    center = int(np.clip(int(idx), 0, int(pts.shape[0]) - 1))
    for radius in (1, 2, 4, 8):
        lo = max(0, center - radius)
        hi = min(int(pts.shape[0]) - 1, center + radius)
        vec = pts[hi] - pts[lo]
        norm = float(np.linalg.norm(vec))
        if norm > 1.0e-9:
            return vec / norm
    raise ValueError("Cannot compute tangent for a zero-length local path")


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = float(math.cos(theta)), float(math.sin(theta))
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def build_local_hugsim_recon_alignment(
    *,
    official_scene_name: str,
    recon_scene_id: int,
    hugsim_model_base: str | Path | None,
    recon_data_root: str | Path,
    hugsim_xy: Any,
    recon_frame_idx: int,
    base_transform: Sim2Transform | None = None,
) -> HUGSIMReconAlignment:
    if hugsim_model_base is None:
        return HUGSIMReconAlignment(
            official_scene_name=str(official_scene_name),
            recon_scene_id=int(recon_scene_id),
            transform=_identity_transform(),
            valid=False,
            reason="missing_hugsim_model_base",
            mode="local_frame",
        )
    try:
        hugsim_path = _load_hugsim_ground_path(hugsim_model_base, str(official_scene_name))
        recon_path = _load_recon_ego_path(recon_data_root, int(recon_scene_id))
        current_xy = np.asarray(hugsim_xy, dtype=np.float64).reshape(-1)[:2]
        if current_xy.shape[0] != 2:
            raise ValueError(f"Expected current HUGSIM xy with 2 values, got {current_xy.shape[0]}")

        hugsim_idx = int(np.argmin(np.linalg.norm(hugsim_path - current_xy[None, :], axis=1)))
        recon_idx = int(np.clip(int(recon_frame_idx), 0, int(recon_path.shape[0]) - 1))
        hugsim_tangent = _polyline_tangent(hugsim_path, hugsim_idx)
        recon_tangent = _polyline_tangent(recon_path, recon_idx)
        hugsim_yaw = float(math.atan2(float(hugsim_tangent[1]), float(hugsim_tangent[0])))
        recon_yaw = float(math.atan2(float(recon_tangent[1]), float(recon_tangent[0])))

        scale = float(base_transform.scale) if base_transform is not None else 1.0
        if not math.isfinite(scale) or abs(scale) <= 1.0e-9:
            scale = 1.0
        rotation = _rotation_matrix(recon_yaw - hugsim_yaw)
        hugsim_anchor = hugsim_path[hugsim_idx]
        recon_anchor = recon_path[recon_idx]
        translation = recon_anchor - scale * (hugsim_anchor @ rotation.T)
        rmse = float(base_transform.rmse_m) if base_transform is not None else 0.0
        return HUGSIMReconAlignment(
            official_scene_name=str(official_scene_name),
            recon_scene_id=int(recon_scene_id),
            transform=Sim2Transform(
                scale=scale,
                rotation=rotation.astype(np.float64),
                translation_xy=np.asarray(translation, dtype=np.float64),
                rmse_m=rmse,
            ),
            valid=True,
            reason=f"local_frame:hugsim_idx={hugsim_idx},recon_idx={recon_idx}",
            mode="local_frame",
        )
    except Exception as exc:
        return HUGSIMReconAlignment(
            official_scene_name=str(official_scene_name),
            recon_scene_id=int(recon_scene_id),
            transform=_identity_transform(),
            valid=False,
            reason=str(exc),
            mode="local_frame",
        )


def build_hugsim_recon_alignment(
    *,
    official_scene_name: str,
    recon_scene_id: int,
    hugsim_model_base: str | Path | None,
    recon_data_root: str | Path,
    max_rmse_m: float = 2.0,
    sample_count: int = 80,
) -> HUGSIMReconAlignment:
    if hugsim_model_base is None:
        return HUGSIMReconAlignment(
            official_scene_name=str(official_scene_name),
            recon_scene_id=int(recon_scene_id),
            transform=_identity_transform(),
            valid=False,
            reason="missing_hugsim_model_base",
        )
    try:
        hugsim_xy = _load_hugsim_ground_path(hugsim_model_base, str(official_scene_name))
        recon_xy = _load_recon_ego_path(recon_data_root, int(recon_scene_id))
        n = max(2, min(int(sample_count), max(int(hugsim_xy.shape[0]), int(recon_xy.shape[0]))))
        src = _resample_polyline_normalized(hugsim_xy, n)
        dst = _resample_polyline_normalized(recon_xy, n)
        transform = fit_sim2(src, dst)
        valid = bool(float(transform.rmse_m) <= float(max_rmse_m))
        reason = "" if valid else f"rmse_m>{float(max_rmse_m):.3f}"
        return HUGSIMReconAlignment(
            official_scene_name=str(official_scene_name),
            recon_scene_id=int(recon_scene_id),
            transform=transform,
            valid=valid,
            reason=reason,
        )
    except Exception as exc:
        return HUGSIMReconAlignment(
            official_scene_name=str(official_scene_name),
            recon_scene_id=int(recon_scene_id),
            transform=_identity_transform(),
            valid=False,
            reason=str(exc),
        )


def hugsim_box_poly_xy(box: Any) -> np.ndarray:
    arr = np.asarray(box, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 7:
        raise ValueError(f"Expected HUGSIM box with 7 values, got {arr.shape[0]}")
    x, y, _z, width, length, _height, yaw = [float(v) for v in arr[:7]]
    c, s = float(math.cos(yaw)), float(math.sin(yaw))
    offsets = np.asarray(
        [
            [length * 0.5, width * 0.5],
            [length * 0.5, -width * 0.5],
            [-length * 0.5, -width * 0.5],
            [-length * 0.5, width * 0.5],
        ],
        dtype=np.float64,
    )
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return (offsets @ rot.T) + np.asarray([[x, y]], dtype=np.float64)


def transform_hugsim_box_to_recon_poly(box: Any, transform: Sim2Transform) -> np.ndarray:
    return transform.transform_points(hugsim_box_poly_xy(box))


def transform_hugsim_boxes_to_recon_objects(obj_boxes: Any, transform: Sim2Transform) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, box in enumerate([] if obj_boxes is None else obj_boxes):
        try:
            poly = transform_hugsim_box_to_recon_poly(box, transform)
        except Exception:
            continue
        out.append(
            {
                "source": "hugsim_inserted",
                "token": f"hugsim_obj_{idx}",
                "category": "vehicle.car",
                "poly": poly.astype(float).tolist(),
            }
        )
    return out


def transform_hugsim_ego_box_to_reward_pose(ego_box: Any, transform: Sim2Transform) -> np.ndarray:
    arr = np.asarray(ego_box, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 7:
        raise ValueError(f"Expected HUGSIM ego box with 7 values, got {arr.shape[0]}")
    xy = transform.transform_points(arr[:2].reshape(1, 2))[0]
    yaw = transform.transform_yaw(float(arr[6]))
    c, s = float(math.cos(yaw)), float(math.sin(yaw))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=np.float64,
    )
    pose[:3, 3] = np.asarray([float(xy[0]), 0.0, float(xy[1])], dtype=np.float64)
    return pose
