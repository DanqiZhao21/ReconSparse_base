from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely import contains as shapely_contains
from shapely import points as shapely_points
from shapely import polygons as shapely_polygons

from framework.algorithms.nuscenes_token_scorer import (
    _DEFAULT_DIRECTION_COMPLIANCE_THRESHOLD_M,
    _DEFAULT_DIRECTION_MIN_CONTINUOUS_REVERSE_S,
    _DEFAULT_DIRECTION_REVERSE_ALIGNMENT_THRESHOLD,
    _DEFAULT_DIRECTION_VIOLATION_THRESHOLD_M,
    _DEFAULT_PATCH_RADIUS_M,
    _DEFAULT_TTC_FUTURE_OFFSETS_S,
    _DEFAULT_TTC_STOPPED_SPEED_THRESHOLD_MPS,
    _linear_decay_score,
    _path_yaw_from_xy,
    _polyline_arclength,
    _project_progress,
    _wrap_angle,
    NuScenesTokenScorer,
)


class NuScenesPDMOccupancyMap:
    """Minimal STRtree-backed occupancy map for the NuScenes PDM backend."""

    def __init__(
        self,
        tokens: Sequence[str],
        geometries: Sequence[Any],
    ) -> None:
        self._tokens = [str(token) for token in tokens]
        self._geometries = np.asarray(list(geometries), dtype=object)
        self._token_to_idx = {token: idx for idx, token in enumerate(self._tokens)}
        self._str_tree = STRtree(self._geometries) if self._tokens else None

    def __len__(self) -> int:
        return len(self._tokens)

    @property
    def tokens(self) -> list[str]:
        return list(self._tokens)

    @property
    def geometries(self) -> np.ndarray:
        return self._geometries

    def query(self, geometry: Any, predicate: str | None = None) -> np.ndarray:
        if self._str_tree is None:
            return np.zeros((0,), dtype=np.int64)
        return np.asarray(self._str_tree.query(geometry, predicate=predicate), dtype=np.int64)

    def intersects(self, geometry: Any) -> list[str]:
        indices = self.query(geometry, predicate="intersects")
        return [self._tokens[int(idx)] for idx in indices.tolist()]


@dataclass
class NuScenesPDMSampleContext:
    sample_token: str
    patch_radius: float
    static_context: dict[str, Any]
    drivable_polygons: list[Any]
    lane_centerlines: list[Any]
    scene_objects: list[dict[str, Any]]
    ea_agent_states: list[dict[str, Any]]
    object_tokens: np.ndarray
    object_polygons: np.ndarray
    object_velocity_xy: np.ndarray
    occupancy_map: NuScenesPDMOccupancyMap
    drivable_map: Any
    centerline_segments_xy: np.ndarray
    centerline_tangents_xy: np.ndarray


class NuScenesPDMDrivableMap:
    """Batch point-in-polygon helper for drivable-area checks."""

    def __init__(self, polygons_xy: Sequence[Any]) -> None:
        self._polygons_xy = [np.asarray(poly, dtype=np.float32) for poly in polygons_xy if np.asarray(poly).ndim == 2]
        self._polygons = (
            np.asarray([Polygon(poly[:, :2]) for poly in self._polygons_xy], dtype=object)
            if self._polygons_xy
            else np.empty((0,), dtype=object)
        )

    def batch_contains_points(self, points_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 3 or pts.shape[-1] != 2:
            raise RuntimeError(f"Expected points_xy shape (candidates,horizon,2), got {tuple(pts.shape)}")
        num_candidates, horizon = pts.shape[:2]
        if not self._polygons_xy:
            return np.ones((num_candidates, horizon), dtype=bool)
        point_geoms = shapely_points(pts[..., 0], pts[..., 1])
        contains_mask = shapely_contains(self._polygons[:, None, None], point_geoms[None, ...])
        return np.any(np.asarray(contains_mask, dtype=bool), axis=0)


class NuScenesPDMScorer:
    """NuScenes-native GRPO scorer backend with a PDM-style intermediate context.

    This stage keeps score computation delegated to the current token scorer while
    introducing a reusable sample-level context object that mirrors the structure
    we'll need for batch occupancy and polygon-query based metrics.
    """

    def __init__(
        self,
        *,
        token2vad_path: str | Path,
        **kwargs: Any,
    ) -> None:
        self.token2vad_path = Path(token2vad_path)
        self.score_mode = str(kwargs.pop("score_mode", "full")).strip().lower()
        if self.score_mode not in {"full", "drivable_area_only"}:
            raise ValueError(
                f"Unsupported NuScenesPDMScorer score_mode={self.score_mode!r}; "
                "expected 'full' or 'drivable_area_only'"
            )
        self._delegate = NuScenesTokenScorer(
            token2vad_path=self.token2vad_path,
            **kwargs,
        )
        self._sample_context_cache: dict[str, NuScenesPDMSampleContext] = {}
        self._ego_length_m = 4.6
        self._ego_width_m = 1.9
        self._derived_context_cache_root = self._delegate.scene_cache_root / "_sample_pdm_context"

    def _derived_context_cache_variant(self) -> str:
        return "pdm-v2"

    def _derived_context_cache_path(self, sample_token: str) -> Path:
        return self._derived_context_cache_root / self._delegate._sample_context_cache_filename(
            sample_token,
            cache_variant=self._derived_context_cache_variant(),
        )

    @staticmethod
    def _serialize_sample_context_payload(
        *,
        sample_token: str,
        patch_radius: float,
        static_context: dict[str, Any],
        drivable_polygons: list[Any],
        lane_centerlines: list[Any],
        scene_objects: list[dict[str, Any]],
        ea_agent_states: list[dict[str, Any]],
        object_tokens: np.ndarray,
        object_velocity_xy: np.ndarray,
        centerline_segments_xy: np.ndarray,
        centerline_tangents_xy: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "sample_token": str(sample_token),
            "patch_radius": float(patch_radius),
            "static_context": static_context,
            "drivable_polygons": drivable_polygons,
            "lane_centerlines": lane_centerlines,
            "scene_objects": scene_objects,
            "ea_agent_states": ea_agent_states,
            "object_tokens": np.asarray(object_tokens, dtype=object),
            "object_velocity_xy": np.asarray(object_velocity_xy, dtype=np.float32),
            "centerline_segments_xy": np.asarray(centerline_segments_xy, dtype=np.float32),
            "centerline_tangents_xy": np.asarray(centerline_tangents_xy, dtype=np.float32),
        }

    def _deserialize_sample_context_payload(self, payload: dict[str, Any]) -> NuScenesPDMSampleContext:
        scene_objects = list(payload.get("scene_objects", []))
        object_tokens, object_polygons, object_velocity_xy, occupancy_map = self._build_object_geometry_arrays(scene_objects)
        drivable_polygons = list(payload.get("drivable_polygons", []))
        return NuScenesPDMSampleContext(
            sample_token=str(payload["sample_token"]),
            patch_radius=float(payload["patch_radius"]),
            static_context=dict(payload["static_context"]),
            drivable_polygons=drivable_polygons,
            lane_centerlines=list(payload.get("lane_centerlines", [])),
            scene_objects=scene_objects,
            ea_agent_states=list(payload.get("ea_agent_states", [])),
            object_tokens=np.asarray(payload.get("object_tokens", object_tokens), dtype=object),
            object_polygons=object_polygons,
            object_velocity_xy=np.asarray(payload.get("object_velocity_xy", object_velocity_xy), dtype=np.float32),
            occupancy_map=occupancy_map,
            drivable_map=NuScenesPDMDrivableMap(drivable_polygons),
            centerline_segments_xy=np.asarray(payload.get("centerline_segments_xy", np.zeros((0, 2, 2), dtype=np.float32)), dtype=np.float32),
            centerline_tangents_xy=np.asarray(payload.get("centerline_tangents_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32),
        )

    def _load_persisted_sample_context(self, sample_token: str) -> NuScenesPDMSampleContext | None:
        path = self._derived_context_cache_path(sample_token)
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return self._deserialize_sample_context_payload(payload)
        except Exception:
            return None

    def _save_persisted_sample_context(self, payload: dict[str, Any]) -> None:
        sample_token = str(payload["sample_token"])
        path = self._derived_context_cache_path(sample_token)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=f"{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(tmp_fd, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path_str, path)
        except Exception:
            try:
                os.unlink(tmp_path_str)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _polygon_from_xy(points_xy: np.ndarray) -> Polygon:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            raise RuntimeError(f"Expected polygon points with shape (N>=3,2), got {tuple(pts.shape)}")
        return Polygon([(float(x), float(y)) for x, y in pts[:, :2]])

    @classmethod
    def _build_object_geometry_arrays(
        cls,
        scene_objects: Sequence[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, NuScenesPDMOccupancyMap]:
        tokens: list[str] = []
        polygons: list[Polygon] = []
        velocities: list[np.ndarray] = []
        for idx, obj in enumerate(scene_objects):
            token = str(obj.get("token", f"obj-{idx}"))
            corners = obj.get("corners_xy", None)
            if corners is None:
                center_xy = np.asarray(obj.get("center_xy", [0.0, 0.0]), dtype=np.float32).reshape(-1)
                yaw = float(obj.get("yaw_rad", 0.0))
                length = float(obj.get("length_m", 1.0))
                width = float(obj.get("width_m", 1.0))
                corners_xy = NuScenesTokenScorer._box_corners_xy(
                    float(center_xy[0]) if center_xy.size > 0 else 0.0,
                    float(center_xy[1]) if center_xy.size > 1 else 0.0,
                    length,
                    width,
                    yaw,
                )
            else:
                corners_xy = np.asarray(corners, dtype=np.float32)
            tokens.append(token)
            polygons.append(cls._polygon_from_xy(corners_xy))
            velocities.append(np.asarray(obj.get("velocity_xy", [0.0, 0.0]), dtype=np.float32).reshape(2))

        object_tokens = np.asarray(tokens, dtype=object)
        object_polygons = np.asarray(polygons, dtype=object)
        object_velocity_xy = (
            np.stack(velocities, axis=0).astype(np.float32, copy=False)
            if velocities
            else np.zeros((0, 2), dtype=np.float32)
        )
        occupancy_map = NuScenesPDMOccupancyMap(tokens=tokens, geometries=polygons)
        return object_tokens, object_polygons, object_velocity_xy, occupancy_map

    def _build_candidate_geometry_batch(
        self,
        traj_xyyaw: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        if traj_xyyaw.ndim != 4 or int(traj_xyyaw.shape[-1]) < 3:
            raise RuntimeError(
                "NuScenesPDMScorer expects traj_xyyaw with shape (batch, candidates, horizon, 3); "
                f"got {tuple(traj_xyyaw.shape)}"
            )

        traj_np = traj_xyyaw.detach().cpu().numpy().astype(np.float32, copy=False)
        centers_xy = traj_np[..., :2].copy()
        yaw_rad = traj_np[..., 2].copy()
        flat_centers = centers_xy.reshape(-1, 2)
        flat_yaw = yaw_rad.reshape(-1)

        corners_list = [
            self._delegate._ego_corners_from_state(np.asarray(xy, dtype=np.float32), float(yaw))
            for xy, yaw in zip(flat_centers, flat_yaw, strict=False)
        ]
        corners_xy = np.stack(corners_list, axis=0).astype(np.float32, copy=False).reshape(*centers_xy.shape[:3], 4, 2)
        polygons = shapely_polygons(corners_xy)

        return {
            "centers_xy": centers_xy,
            "yaw_rad": yaw_rad,
            "corners_xy": corners_xy,
            "polygons": polygons,
        }

    def _build_candidate_geometry_batch_torch(
        self,
        traj_xyyaw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if traj_xyyaw.ndim != 4 or int(traj_xyyaw.shape[-1]) < 3:
            raise RuntimeError(
                "NuScenesPDMScorer expects traj_xyyaw with shape (batch, candidates, horizon, 3); "
                f"got {tuple(traj_xyyaw.shape)}"
            )
        traj = traj_xyyaw.to(device=traj_xyyaw.device, dtype=torch.float32)
        centers_xy = traj[..., :2]
        yaw_rad = traj[..., 2]
        corners_xy = self._ego_corners_from_state_torch(centers_xy, yaw_rad)
        return {
            "centers_xy": centers_xy,
            "yaw_rad": yaw_rad,
            "corners_xy": corners_xy,
        }

    def _ego_corners_from_state_torch(self, centers_xy: torch.Tensor, yaw_rad: torch.Tensor) -> torch.Tensor:
        dx = float(self._ego_length_m) * 0.5
        dy = float(self._ego_width_m) * 0.5
        template = torch.tensor(
            [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]],
            device=centers_xy.device,
            dtype=torch.float32,
        )
        cos_yaw = torch.cos(yaw_rad)
        sin_yaw = torch.sin(yaw_rad)
        rot = torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw], dim=-1),
                torch.stack([sin_yaw, cos_yaw], dim=-1),
            ],
            dim=-2,
        )
        rotated = torch.einsum("...qd,...dc->...qc", template.expand(*centers_xy.shape[:-1], 4, 2), rot)
        return rotated + centers_xy.unsqueeze(-2)

    @staticmethod
    def _tensor_from_numpy(array: np.ndarray, *, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(array, device=device, dtype=torch.float32)

    @staticmethod
    def _build_centerline_segment_cache(centerlines: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        segments: list[np.ndarray] = []
        tangents: list[np.ndarray] = []
        for centerline in centerlines:
            coords = np.asarray(centerline, dtype=np.float32)
            if coords.ndim != 2 or coords.shape[0] < 2:
                continue
            for idx in range(int(coords.shape[0]) - 1):
                p0 = coords[idx, :2]
                p1 = coords[idx + 1, :2]
                seg = p1 - p0
                seg_norm = float(np.linalg.norm(seg))
                if seg_norm <= 1.0e-6:
                    continue
                segments.append(np.stack([p0, p1], axis=0).astype(np.float32, copy=False))
                tangents.append((seg / seg_norm).astype(np.float32, copy=False))
        if not segments:
            return (
                np.zeros((0, 2, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )
        return (
            np.stack(segments, axis=0).astype(np.float32, copy=False),
            np.stack(tangents, axis=0).astype(np.float32, copy=False),
        )

    @staticmethod
    def _candidate_speed_and_yaw_rate_batch(
        centers_xy: np.ndarray,
        yaw_rad: np.ndarray,
        *,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if centers_xy.ndim != 3 or yaw_rad.ndim != 2:
            raise RuntimeError(
                f"Expected centers_xy=(candidates,horizon,2) and yaw_rad=(candidates,horizon), got "
                f"{tuple(centers_xy.shape)} and {tuple(yaw_rad.shape)}"
            )
        prev_xy = np.concatenate([centers_xy[:, :1, :], centers_xy[:, :-1, :]], axis=1)
        step_delta = centers_xy - prev_xy
        speed_mps = np.linalg.norm(step_delta, axis=-1) / max(1.0e-6, float(dt_s))
        prev_yaw = np.concatenate([yaw_rad[:, :1], yaw_rad[:, :-1]], axis=1)
        yaw_rate = np.abs(_wrap_angle(yaw_rad - prev_yaw)) / max(1.0e-6, float(dt_s))
        return speed_mps.astype(np.float32, copy=False), yaw_rate.astype(np.float32, copy=False)

    def _build_ttc_projection_geometry(
        self,
        *,
        centers_xy: np.ndarray,
        yaw_rad: np.ndarray,
        dt_s: float,
    ) -> dict[str, np.ndarray]:
        centers = np.asarray(centers_xy, dtype=np.float32)
        yaw = np.asarray(yaw_rad, dtype=np.float32)
        num_candidates, horizon = centers.shape[:2]
        offsets_s = np.asarray([offset for offset in _DEFAULT_TTC_FUTURE_OFFSETS_S if float(offset) > 0.0], dtype=np.float32)
        num_offsets = int(offsets_s.shape[0])
        speed_mps, _ = self._candidate_speed_and_yaw_rate_batch(centers, yaw, dt_s=float(dt_s))
        heading_vec = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1).astype(np.float32, copy=False)
        proj_centers = (
            centers[:, :, None, :]
            + heading_vec[:, :, None, :] * speed_mps[:, :, None, None] * offsets_s.reshape(1, 1, num_offsets, 1)
        ).astype(np.float32, copy=False)
        flat_centers = proj_centers.reshape(-1, 2)
        flat_yaw = np.repeat(yaw[:, :, None], num_offsets, axis=2).reshape(-1)
        proj_corners = np.stack(
            [
                self._delegate._ego_corners_from_state(np.asarray(xy, dtype=np.float32), float(yaw_val))
                for xy, yaw_val in zip(flat_centers, flat_yaw, strict=False)
            ],
            axis=0,
        ).astype(np.float32, copy=False).reshape(num_candidates, horizon, num_offsets, 4, 2)
        proj_polygons = shapely_polygons(proj_corners)

        return {
            "offsets_s": offsets_s,
            "centers_xy": proj_centers,
            "corners_xy": proj_corners,
            "polygons": proj_polygons,
        }

    def _build_ttc_projection_geometry_torch(
        self,
        *,
        centers_xy: torch.Tensor,
        yaw_rad: torch.Tensor,
        dt_s: float,
    ) -> dict[str, torch.Tensor]:
        offsets_s = torch.as_tensor(
            [offset for offset in _DEFAULT_TTC_FUTURE_OFFSETS_S if float(offset) > 0.0],
            device=centers_xy.device,
            dtype=torch.float32,
        )
        speed_mps, _ = self._candidate_speed_and_yaw_rate_batch_torch(centers_xy, yaw_rad, dt_s=float(dt_s))
        heading_vec = torch.stack([torch.cos(yaw_rad), torch.sin(yaw_rad)], dim=-1)
        proj_centers = centers_xy[:, :, None, :] + heading_vec[:, :, None, :] * speed_mps[:, :, None, None] * offsets_s.view(1, 1, -1, 1)
        proj_corners = self._ego_corners_from_state_torch(proj_centers, yaw_rad[:, :, None].expand_as(proj_centers[..., 0]))
        return {
            "offsets_s": offsets_s,
            "centers_xy": proj_centers,
            "corners_xy": proj_corners,
        }

    @staticmethod
    def _batch_project_progress(
        final_points_xy: np.ndarray,
        path_xy: np.ndarray,
        path_s: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(final_points_xy, dtype=np.float32)
        path_xy = np.asarray(path_xy, dtype=np.float32)
        path_s = np.asarray(path_s, dtype=np.float32)
        if points.ndim != 2 or points.shape[-1] != 2:
            raise RuntimeError(f"Expected final_points_xy shape (candidates,2), got {tuple(points.shape)}")
        num_candidates = int(points.shape[0])
        if num_candidates <= 0 or int(path_xy.shape[0]) <= 1 or int(path_s.shape[0]) <= 1:
            return np.zeros((num_candidates,), dtype=np.float32)

        seg_start = path_xy[:-1]
        seg_end = path_xy[1:]
        seg_vec = (seg_end - seg_start).astype(np.float32, copy=False)
        seg_len_sq = np.sum(seg_vec * seg_vec, axis=-1).astype(np.float32, copy=False)
        valid_seg = seg_len_sq > 1.0e-12
        if not bool(np.any(valid_seg)):
            return np.zeros((num_candidates,), dtype=np.float32)

        safe_seg_len_sq = np.where(valid_seg, seg_len_sq, 1.0).astype(np.float32, copy=False)
        delta = points[:, None, :] - seg_start[None, :, :]
        alpha = np.sum(delta * seg_vec[None, :, :], axis=-1) / safe_seg_len_sq[None, :]
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)
        proj = seg_start[None, :, :] + alpha[..., None] * seg_vec[None, :, :]
        dist = np.linalg.norm(points[:, None, :] - proj, axis=-1).astype(np.float32, copy=False)
        dist = np.where(valid_seg[None, :], dist, np.inf).astype(np.float32, copy=False)

        best_idx = np.argmin(dist, axis=1)
        seg_len = np.sqrt(safe_seg_len_sq).astype(np.float32, copy=False)
        best_s_all = path_s[:-1][None, :] + alpha * seg_len[None, :]
        best_s = np.take_along_axis(best_s_all, best_idx[:, None], axis=1).reshape(-1)
        has_finite = np.isfinite(np.take_along_axis(dist, best_idx[:, None], axis=1).reshape(-1))
        return np.where(has_finite, best_s, 0.0).astype(np.float32, copy=False)

    @staticmethod
    def _batch_project_progress_torch(
        final_points_xy: torch.Tensor,
        path_xy: torch.Tensor,
        path_s: torch.Tensor,
    ) -> torch.Tensor:
        num_candidates = int(final_points_xy.shape[0])
        if num_candidates <= 0 or int(path_xy.shape[0]) <= 1 or int(path_s.shape[0]) <= 1:
            return torch.zeros((num_candidates,), device=final_points_xy.device, dtype=torch.float32)
        seg_start = path_xy[:-1]
        seg_end = path_xy[1:]
        seg_vec = seg_end - seg_start
        seg_len_sq = (seg_vec * seg_vec).sum(dim=-1)
        valid_seg = seg_len_sq > 1.0e-12
        if not bool(valid_seg.any()):
            return torch.zeros((num_candidates,), device=final_points_xy.device, dtype=torch.float32)
        safe_seg_len_sq = torch.where(valid_seg, seg_len_sq, torch.ones_like(seg_len_sq))
        delta = final_points_xy[:, None, :] - seg_start[None, :, :]
        alpha = ((delta * seg_vec[None, :, :]).sum(dim=-1) / safe_seg_len_sq[None, :]).clamp(0.0, 1.0)
        proj = seg_start[None, :, :] + alpha.unsqueeze(-1) * seg_vec[None, :, :]
        dist = torch.linalg.norm(final_points_xy[:, None, :] - proj, dim=-1)
        inf = torch.full_like(dist, float("inf"))
        dist = torch.where(valid_seg.unsqueeze(0), dist, inf)
        best_idx = dist.argmin(dim=1)
        seg_len = torch.sqrt(safe_seg_len_sq)
        best_s_all = path_s[:-1].unsqueeze(0) + alpha * seg_len.unsqueeze(0)
        best_s = torch.gather(best_s_all, dim=1, index=best_idx.unsqueeze(-1)).squeeze(-1)
        best_dist = torch.gather(dist, dim=1, index=best_idx.unsqueeze(-1)).squeeze(-1)
        return torch.where(torch.isfinite(best_dist), best_s, torch.zeros_like(best_s))

    @staticmethod
    def _batch_centerline_stats(
        points_xy: np.ndarray,
        segments_xy: np.ndarray,
        tangents_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 3 or pts.shape[-1] != 2:
            raise RuntimeError(f"Expected points_xy shape (candidates,horizon,2), got {tuple(pts.shape)}")
        num_candidates, horizon = pts.shape[:2]
        if segments_xy.size <= 0:
            return (
                np.zeros((num_candidates, horizon), dtype=np.float32),
                np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (num_candidates, horizon, 1)),
            )

        seg_start = segments_xy[:, 0, :].astype(np.float32, copy=False)
        seg_end = segments_xy[:, 1, :].astype(np.float32, copy=False)
        seg_vec = (seg_end - seg_start).astype(np.float32, copy=False)
        seg_len_sq = np.sum(seg_vec * seg_vec, axis=-1).astype(np.float32, copy=False)
        valid_seg = seg_len_sq > 1.0e-12
        safe_seg_len_sq = np.where(valid_seg, seg_len_sq, 1.0).astype(np.float32, copy=False)

        delta = pts[:, :, None, :] - seg_start[None, None, :, :]
        alpha = np.sum(delta * seg_vec[None, None, :, :], axis=-1) / safe_seg_len_sq[None, None, :]
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)
        proj = seg_start[None, None, :, :] + alpha[..., None] * seg_vec[None, None, :, :]
        dist = np.linalg.norm(pts[:, :, None, :] - proj, axis=-1).astype(np.float32, copy=False)
        dist = np.where(valid_seg[None, None, :], dist, np.inf).astype(np.float32, copy=False)

        best_idx = np.argmin(dist, axis=-1)
        best_dist = np.take_along_axis(dist, best_idx[..., None], axis=-1).squeeze(-1).astype(np.float32, copy=False)
        best_tangent = tangents_xy[np.clip(best_idx, 0, max(0, tangents_xy.shape[0] - 1))].astype(np.float32, copy=False)
        finite_mask = np.isfinite(best_dist)
        best_dist = np.where(finite_mask, best_dist, 0.0).astype(np.float32, copy=False)
        best_tangent = np.where(
            finite_mask[..., None],
            best_tangent,
            np.asarray([1.0, 0.0], dtype=np.float32).reshape(1, 1, 2),
        ).astype(np.float32, copy=False)
        return best_dist, best_tangent

    @staticmethod
    def _batch_centerline_stats_torch(
        points_xy: torch.Tensor,
        segments_xy: torch.Tensor,
        tangents_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_candidates, horizon = points_xy.shape[:2]
        if segments_xy.numel() <= 0:
            return (
                torch.zeros((num_candidates, horizon), device=points_xy.device, dtype=torch.float32),
                torch.tensor([1.0, 0.0], device=points_xy.device, dtype=torch.float32).view(1, 1, 2).expand(num_candidates, horizon, 2),
            )
        seg_start = segments_xy[:, 0, :]
        seg_end = segments_xy[:, 1, :]
        seg_vec = seg_end - seg_start
        seg_len_sq = (seg_vec * seg_vec).sum(dim=-1)
        valid_seg = seg_len_sq > 1.0e-12
        safe_seg_len_sq = torch.where(valid_seg, seg_len_sq, torch.ones_like(seg_len_sq))
        delta = points_xy[:, :, None, :] - seg_start[None, None, :, :]
        alpha = ((delta * seg_vec[None, None, :, :]).sum(dim=-1) / safe_seg_len_sq.view(1, 1, -1)).clamp(0.0, 1.0)
        proj = seg_start.view(1, 1, -1, 2) + alpha.unsqueeze(-1) * seg_vec.view(1, 1, -1, 2)
        dist = torch.linalg.norm(points_xy[:, :, None, :] - proj, dim=-1)
        inf = torch.full_like(dist, float("inf"))
        dist = torch.where(valid_seg.view(1, 1, -1), dist, inf)
        best_idx = dist.argmin(dim=-1)
        best_dist = torch.gather(dist, dim=-1, index=best_idx.unsqueeze(-1)).squeeze(-1)
        best_tangent = tangents_xy[best_idx.clamp(0, max(0, tangents_xy.shape[0] - 1))]
        finite_mask = torch.isfinite(best_dist)
        best_dist = torch.where(finite_mask, best_dist, torch.zeros_like(best_dist))
        fallback = torch.tensor([1.0, 0.0], device=points_xy.device, dtype=torch.float32).view(1, 1, 2)
        best_tangent = torch.where(finite_mask.unsqueeze(-1), best_tangent, fallback)
        return best_dist, best_tangent

    def _batch_map_metrics(
        self,
        *,
        sample_context: NuScenesPDMSampleContext,
        candidate_geometry: dict[str, np.ndarray],
        dt_s: float,
    ) -> dict[str, np.ndarray]:
        centers_xy = np.asarray(candidate_geometry["centers_xy"], dtype=np.float32)
        num_candidates, horizon = centers_xy.shape[:2]
        if num_candidates <= 0 or horizon <= 0:
            return {
                "drivable_area": np.ones((num_candidates,), dtype=np.float32),
                "lane_keeping": np.ones((num_candidates,), dtype=np.float32),
                "driving_direction": np.ones((num_candidates,), dtype=np.float32),
            }

        inside = sample_context.drivable_map.batch_contains_points(centers_xy)
        drivable_area = inside.all(axis=1).astype(np.float32, copy=False)

        lateral_errors, tangents_xy = self._batch_centerline_stats(
            centers_xy,
            sample_context.centerline_segments_xy,
            sample_context.centerline_tangents_xy,
        )
        mean_lateral = lateral_errors.mean(axis=1).astype(np.float32, copy=False)
        if sample_context.centerline_segments_xy.size > 0:
            lane_keeping = np.clip(1.0 - (mean_lateral / 2.0), 0.0, 1.0).astype(np.float32, copy=False)
        else:
            lane_keeping = np.ones((num_candidates,), dtype=np.float32)

        prev_xy = np.concatenate([centers_xy[:, :1, :], centers_xy[:, :-1, :]], axis=1)
        step_delta = centers_xy - prev_xy
        step_norm = np.linalg.norm(step_delta, axis=-1).astype(np.float32, copy=False)
        reverse_alignment = np.maximum(
            0.0,
            -np.sum(
                np.where(step_norm[..., None] > 1.0e-6, step_delta / np.maximum(step_norm[..., None], 1.0e-6), 0.0)
                * tangents_xy,
                axis=-1,
            ),
        ).astype(np.float32, copy=False)

        reverse_mask = reverse_alignment > _DEFAULT_DIRECTION_REVERSE_ALIGNMENT_THRESHOLD
        step_indices = np.arange(horizon, dtype=np.int32).reshape(1, -1)
        last_false_idx = np.maximum.accumulate(np.where(reverse_mask, -1, step_indices), axis=1)
        streak_len = (step_indices - last_false_idx).astype(np.float32, copy=False)
        continuous_reverse_time_s = np.where(
            reverse_mask,
            streak_len * max(1.0e-6, float(dt_s)),
            0.0,
        ).astype(np.float32, copy=False)
        accrue_mask = continuous_reverse_time_s > _DEFAULT_DIRECTION_MIN_CONTINUOUS_REVERSE_S
        oncoming_progress_m = np.sum(
            np.where(accrue_mask, step_norm * reverse_alignment, 0.0),
            axis=1,
        ).astype(np.float32, copy=False)

        good_threshold = float(_DEFAULT_DIRECTION_COMPLIANCE_THRESHOLD_M)
        bad_threshold = float(_DEFAULT_DIRECTION_VIOLATION_THRESHOLD_M)
        span = max(1.0e-6, bad_threshold - good_threshold)
        driving_direction = np.where(
            oncoming_progress_m <= good_threshold,
            1.0,
            np.where(
                oncoming_progress_m >= bad_threshold,
                0.0,
                np.clip((bad_threshold - oncoming_progress_m) / span, 0.0, 1.0),
            ),
        ).astype(np.float32, copy=False)
        return {
            "drivable_area": drivable_area,
            "lane_keeping": lane_keeping,
            "driving_direction": driving_direction,
        }

    @staticmethod
    def _query_hits_per_candidate(
        occupancy_map: Any,
        polygons_at_step: np.ndarray,
        *,
        predicate: str = "intersects",
    ) -> np.ndarray:
        num_candidates = int(polygons_at_step.shape[0])
        hits = np.zeros((num_candidates,), dtype=bool)
        if num_candidates <= 0 or len(occupancy_map) <= 0:
            return hits
        query_result = occupancy_map.query(polygons_at_step, predicate=predicate)
        query_arr = np.asarray(query_result)
        if query_arr.size <= 0:
            return hits
        if query_arr.ndim == 2 and query_arr.shape[0] >= 1:
            cand_indices = np.asarray(query_arr[0], dtype=np.int64).reshape(-1)
        else:
            cand_indices = np.asarray(query_arr, dtype=np.int64).reshape(-1)
        cand_indices = cand_indices[(cand_indices >= 0) & (cand_indices < num_candidates)]
        if cand_indices.size > 0:
            hits[cand_indices] = True
        return hits

    @classmethod
    def _query_hits_per_candidate_grid(
        cls,
        occupancy_map: Any,
        polygons_grid: np.ndarray,
        *,
        predicate: str = "intersects",
    ) -> np.ndarray:
        grid = np.asarray(polygons_grid, dtype=object)
        if grid.ndim != 2:
            raise RuntimeError(f"Expected polygons_grid shape (num_candidates,num_queries), got {tuple(grid.shape)}")
        num_candidates, num_queries = grid.shape
        hits = np.zeros((num_candidates, num_queries), dtype=bool)
        if num_candidates <= 0 or num_queries <= 0 or len(occupancy_map) <= 0:
            return hits
        query_result = occupancy_map.query(grid.reshape(-1), predicate=predicate)
        query_arr = np.asarray(query_result)
        if query_arr.size <= 0:
            return hits
        if query_arr.ndim == 2 and query_arr.shape[0] >= 1:
            flat_indices = np.asarray(query_arr[0], dtype=np.int64).reshape(-1)
        else:
            flat_indices = np.asarray(query_arr, dtype=np.int64).reshape(-1)
        flat_indices = flat_indices[(flat_indices >= 0) & (flat_indices < num_candidates * num_queries)]
        if flat_indices.size > 0:
            hits.reshape(-1)[flat_indices] = True
        return hits

    def _batch_collision_ttc_metrics(
        self,
        *,
        sample_context: NuScenesPDMSampleContext,
        candidate_geometry: dict[str, np.ndarray],
        dt_s: float,
    ) -> dict[str, np.ndarray]:
        polygons = np.asarray(candidate_geometry["polygons"], dtype=object)
        centers_xy = np.asarray(candidate_geometry["centers_xy"], dtype=np.float32)
        yaw_rad = np.asarray(candidate_geometry["yaw_rad"], dtype=np.float32)
        num_candidates, horizon = polygons.shape[:2]

        no_collision = np.ones((num_candidates,), dtype=np.float32)
        ttc = np.ones((num_candidates,), dtype=np.float32)
        if horizon <= 0 or len(sample_context.occupancy_map) <= 0:
            return {"no_collision": no_collision, "ttc": ttc}

        speed_mps, _ = self._candidate_speed_and_yaw_rate_batch(centers_xy, yaw_rad, dt_s=float(dt_s))
        ttc_projection = self._build_ttc_projection_geometry(
            centers_xy=centers_xy,
            yaw_rad=yaw_rad,
            dt_s=float(dt_s),
        )
        collision_hits = self._query_hits_per_candidate_grid(
            sample_context.occupancy_map,
            polygons,
            predicate="intersects",
        )
        collision_mask = np.any(collision_hits, axis=1)
        earliest_ttc_risk_s = np.full((num_candidates,), np.inf, dtype=np.float32)

        moving_mask = speed_mps >= float(_DEFAULT_TTC_STOPPED_SPEED_THRESHOLD_MPS)
        if bool(np.any(moving_mask)):
            proj_polygons = np.asarray(ttc_projection["polygons"], dtype=object)
            proj_hits = self._query_hits_per_candidate_grid(
                sample_context.occupancy_map,
                proj_polygons.reshape(num_candidates, horizon * int(ttc_projection["offsets_s"].shape[0])),
                predicate="intersects",
            ).reshape(num_candidates, horizon, int(ttc_projection["offsets_s"].shape[0]))
            risk_mask = np.logical_and(proj_hits, moving_mask[:, :, None])
            if bool(np.any(risk_mask)):
                risk_offsets = np.where(
                    risk_mask,
                    ttc_projection["offsets_s"].reshape(1, 1, -1),
                    np.inf,
                ).astype(np.float32, copy=False)
                earliest_ttc_risk_s = np.min(risk_offsets.reshape(num_candidates, -1), axis=1).astype(np.float32, copy=False)

        no_collision[collision_mask] = 0.0
        if len(_DEFAULT_TTC_FUTURE_OFFSETS_S) > 0:
            ttc_horizon_s = float(_DEFAULT_TTC_FUTURE_OFFSETS_S[-1])
            finite_mask = np.isfinite(earliest_ttc_risk_s)
            ttc[finite_mask] = np.clip(
                earliest_ttc_risk_s[finite_mask] / max(1.0e-6, ttc_horizon_s),
                0.0,
                1.0,
            ).astype(np.float32, copy=False)
        return {"no_collision": no_collision, "ttc": ttc}

    @staticmethod
    def _candidate_speed_and_yaw_rate_batch_torch(
        centers_xy: torch.Tensor,
        yaw_rad: torch.Tensor,
        *,
        dt_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev_xy = torch.cat([centers_xy[:, :1, :], centers_xy[:, :-1, :]], dim=1)
        step_delta = centers_xy - prev_xy
        speed_mps = torch.linalg.norm(step_delta, dim=-1) / max(1.0e-6, float(dt_s))
        prev_yaw = torch.cat([yaw_rad[:, :1], yaw_rad[:, :-1]], dim=1)
        yaw_delta = torch.atan2(torch.sin(yaw_rad - prev_yaw), torch.cos(yaw_rad - prev_yaw))
        yaw_rate = torch.abs(yaw_delta) / max(1.0e-6, float(dt_s))
        return speed_mps, yaw_rate

    @staticmethod
    def _oriented_box_axes_torch(corners_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge0 = corners_xy[..., 1, :] - corners_xy[..., 0, :]
        edge1 = corners_xy[..., 3, :] - corners_xy[..., 0, :]
        axis0 = edge0 / torch.linalg.norm(edge0, dim=-1, keepdim=True).clamp_min(1.0e-6)
        axis1 = edge1 / torch.linalg.norm(edge1, dim=-1, keepdim=True).clamp_min(1.0e-6)
        return axis0, axis1

    @classmethod
    def _obb_intersects_torch(cls, boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
        center_a = boxes_a.mean(dim=-2)
        center_b = boxes_b.mean(dim=-2)
        axis_a0, axis_a1 = cls._oriented_box_axes_torch(boxes_a)
        axis_b0, axis_b1 = cls._oriented_box_axes_torch(boxes_b)
        axes = torch.stack([axis_a0, axis_a1, axis_b0, axis_b1], dim=-2)
        rel = center_b - center_a
        proj_center = torch.abs((rel.unsqueeze(-2) * axes).sum(dim=-1))
        proj_a = torch.abs((boxes_a.unsqueeze(-3) - center_a.unsqueeze(-2).unsqueeze(-2)) * axes.unsqueeze(-2)).sum(dim=-1)
        proj_b = torch.abs((boxes_b.unsqueeze(-3) - center_b.unsqueeze(-2).unsqueeze(-2)) * axes.unsqueeze(-2)).sum(dim=-1)
        radius_a = proj_a.sum(dim=-1)
        radius_b = proj_b.sum(dim=-1)
        overlap = proj_center <= (radius_a + radius_b + 1.0e-5)
        return overlap.all(dim=-1)

    def _build_object_box_tensor(
        self,
        sample_context: NuScenesPDMSampleContext,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scene_objects = list(sample_context.scene_objects)
        num_objects = len(scene_objects)
        if num_objects <= 0:
            return (
                torch.zeros((0, 2), device=device, dtype=torch.float32),
                torch.zeros((0, 2), device=device, dtype=torch.float32),
                torch.zeros((0,), device=device, dtype=torch.float32),
                torch.zeros((0,), device=device, dtype=torch.float32),
            )
        centers = []
        dims = []
        yaws = []
        velocities = []
        for obj in scene_objects:
            center = np.asarray(obj.get("center_xy", [0.0, 0.0]), dtype=np.float32).reshape(2)
            velocity = np.asarray(obj.get("velocity_xy", [0.0, 0.0]), dtype=np.float32).reshape(2)
            length = float(obj.get("length_m", 1.0))
            width = float(obj.get("width_m", 1.0))
            yaw = float(obj.get("yaw_rad", 0.0))
            centers.append(center)
            dims.append([length, width])
            yaws.append(yaw)
            velocities.append(velocity)
        return (
            torch.as_tensor(np.asarray(centers, dtype=np.float32), device=device),
            torch.as_tensor(np.asarray(dims, dtype=np.float32), device=device),
            torch.as_tensor(np.asarray(yaws, dtype=np.float32), device=device),
            torch.as_tensor(np.asarray(velocities, dtype=np.float32), device=device),
        )

    def _object_corners_torch(
        self,
        centers_xy: torch.Tensor,
        dims_lw: torch.Tensor,
        yaw_rad: torch.Tensor,
    ) -> torch.Tensor:
        dx = dims_lw[..., 0] * 0.5
        dy = dims_lw[..., 1] * 0.5
        template = torch.stack(
            [
                torch.stack([dx, dy], dim=-1),
                torch.stack([dx, -dy], dim=-1),
                torch.stack([-dx, -dy], dim=-1),
                torch.stack([-dx, dy], dim=-1),
            ],
            dim=-2,
        )
        cos_yaw = torch.cos(yaw_rad)
        sin_yaw = torch.sin(yaw_rad)
        rot = torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw], dim=-1),
                torch.stack([sin_yaw, cos_yaw], dim=-1),
            ],
            dim=-2,
        )
        rotated = torch.einsum("...qd,...dc->...qc", template, rot)
        return rotated + centers_xy.unsqueeze(-2)

    def _batch_collision_ttc_metrics_torch(
        self,
        *,
        sample_context: NuScenesPDMSampleContext,
        candidate_geometry: dict[str, torch.Tensor],
        dt_s: float,
    ) -> dict[str, torch.Tensor]:
        centers_xy = candidate_geometry["centers_xy"]
        yaw_rad = candidate_geometry["yaw_rad"]
        ego_corners = candidate_geometry["corners_xy"]
        num_candidates, horizon = centers_xy.shape[:2]
        device = centers_xy.device
        no_collision = torch.ones((num_candidates,), device=device, dtype=torch.float32)
        ttc = torch.ones((num_candidates,), device=device, dtype=torch.float32)
        if num_candidates <= 0 or horizon <= 0 or len(sample_context.scene_objects) <= 0:
            return {"no_collision": no_collision, "ttc": ttc}

        object_centers, object_dims, object_yaws, object_vel = self._build_object_box_tensor(sample_context, device=device)
        if object_centers.shape[0] <= 0:
            return {"no_collision": no_collision, "ttc": ttc}

        times = (torch.arange(horizon, device=device, dtype=torch.float32) + 1.0) * float(dt_s)
        obj_centers_now = object_centers.view(1, 1, -1, 2) + object_vel.view(1, 1, -1, 2) * times.view(1, -1, 1, 1)
        obj_corners_now = self._object_corners_torch(
            obj_centers_now,
            object_dims.view(1, 1, -1, 2).expand(1, horizon, -1, -1),
            object_yaws.view(1, 1, -1).expand(1, horizon, -1),
        ).expand(num_candidates, -1, -1, -1, -1)

        ego_boxes = ego_corners.unsqueeze(2).expand(-1, -1, obj_centers_now.shape[2], -1, -1)
        collision_hits = self._obb_intersects_torch(ego_boxes, obj_corners_now)
        collision_mask = collision_hits.any(dim=(1, 2))
        no_collision = torch.where(collision_mask, torch.zeros_like(no_collision), no_collision)

        speed_mps, _ = self._candidate_speed_and_yaw_rate_batch_torch(centers_xy, yaw_rad, dt_s=float(dt_s))
        moving_mask = speed_mps >= float(_DEFAULT_TTC_STOPPED_SPEED_THRESHOLD_MPS)
        if bool(moving_mask.any()):
            ttc_projection = self._build_ttc_projection_geometry_torch(
                centers_xy=centers_xy,
                yaw_rad=yaw_rad,
                dt_s=float(dt_s),
            )
            proj_corners = ttc_projection["corners_xy"]
            offsets = ttc_projection["offsets_s"]
            future_times = times.view(1, horizon, 1, 1) + offsets.view(1, 1, -1, 1)
            obj_centers_future = object_centers.view(1, 1, 1, -1, 2) + object_vel.view(1, 1, 1, -1, 2) * future_times.unsqueeze(-1)
            obj_corners_future = self._object_corners_torch(
                obj_centers_future,
                object_dims.view(1, 1, 1, -1, 2).expand(1, horizon, offsets.shape[0], -1, -1),
                object_yaws.view(1, 1, 1, -1).expand(1, horizon, offsets.shape[0], -1),
            ).expand(num_candidates, -1, -1, -1, -1, -1)
            ego_future = proj_corners.unsqueeze(3).expand(-1, -1, -1, obj_centers_future.shape[3], -1, -1)
            future_hits = self._obb_intersects_torch(ego_future, obj_corners_future).any(dim=-1)
            risk_mask = future_hits & moving_mask.unsqueeze(-1)
            risk_offsets = torch.where(
                risk_mask,
                offsets.view(1, 1, -1).expand(num_candidates, horizon, -1),
                torch.full((num_candidates, horizon, offsets.shape[0]), float("inf"), device=device, dtype=torch.float32),
            )
            earliest = risk_offsets.reshape(num_candidates, -1).min(dim=1).values
            finite_mask = torch.isfinite(earliest)
            if bool(finite_mask.any()):
                ttc_horizon_s = float(_DEFAULT_TTC_FUTURE_OFFSETS_S[-1]) if _DEFAULT_TTC_FUTURE_OFFSETS_S else 1.0
                ttc = torch.where(
                    finite_mask,
                    (earliest / max(1.0e-6, ttc_horizon_s)).clamp(0.0, 1.0),
                    ttc,
                )
        return {"no_collision": no_collision, "ttc": ttc}

    def _score_candidate_batch_for_sample_torch(
        self,
        *,
        sample_context: NuScenesPDMSampleContext,
        candidate_geometry: dict[str, torch.Tensor],
        gt_xy_cmp: np.ndarray,
        gt_yaw_cmp: np.ndarray,
        gt_xy_full: np.ndarray,
        gt_s_full: np.ndarray,
        gt_total_len: float,
        dt_s: float,
    ) -> torch.Tensor:
        centers_xy = candidate_geometry["centers_xy"]
        yaw_rad = candidate_geometry["yaw_rad"]
        device = centers_xy.device
        num_candidates, horizon = centers_xy.shape[:2]
        if num_candidates <= 0 or horizon <= 0:
            return torch.zeros((num_candidates,), device=device, dtype=torch.float32)

        gt_xy_cmp_t = torch.as_tensor(gt_xy_cmp, device=device, dtype=torch.float32)
        gt_yaw_cmp_t = torch.as_tensor(gt_yaw_cmp, device=device, dtype=torch.float32)
        gt_xy_full_t = torch.as_tensor(gt_xy_full, device=device, dtype=torch.float32)
        gt_s_full_t = torch.as_tensor(gt_s_full, device=device, dtype=torch.float32)

        pos_err = torch.linalg.norm(centers_xy - gt_xy_cmp_t.view(1, horizon, 2), dim=-1) if gt_xy_cmp_t.numel() > 0 else torch.zeros((num_candidates, horizon), device=device)
        mean_err = pos_err.mean(dim=1) if pos_err.numel() > 0 else torch.zeros((num_candidates,), device=device)
        yaw_delta = torch.atan2(torch.sin(yaw_rad - gt_yaw_cmp_t.view(1, horizon)), torch.cos(yaw_rad - gt_yaw_cmp_t.view(1, horizon))) if gt_yaw_cmp_t.numel() > 0 else torch.zeros((num_candidates, horizon), device=device)
        _ = torch.abs(yaw_delta).mean(dim=1) if yaw_delta.numel() > 0 else torch.zeros((num_candidates,), device=device)

        cand_progress = self._batch_project_progress_torch(centers_xy[:, -1], gt_xy_full_t, gt_s_full_t)
        progress_ratio = (cand_progress / max(1.0e-6, float(gt_total_len))).clamp(0.0, 1.0)

        prev_xy = torch.cat([centers_xy[:, :1, :], centers_xy[:, :-1, :]], dim=1)
        cand_vel = centers_xy - prev_xy
        cand_acc = cand_vel[:, 1:, :] - cand_vel[:, :-1, :] if horizon > 1 else torch.zeros((num_candidates, 0, 2), device=device)
        prev_yaw = torch.cat([yaw_rad[:, :1], yaw_rad[:, :-1]], dim=1)
        cand_yaw_rate = torch.abs(torch.atan2(torch.sin(yaw_rad - prev_yaw), torch.cos(yaw_rad - prev_yaw))) / max(1.0e-6, float(dt_s))
        smooth_pen = torch.linalg.norm(cand_acc, dim=-1).mean(dim=1) if cand_acc.numel() > 0 else torch.zeros((num_candidates,), device=device)
        comfort_cost = smooth_pen + 0.25 * cand_yaw_rate.mean(dim=1)
        history_comfort = (1.0 - 0.08 * comfort_cost).clamp(0.0, 1.0)

        drivable_polygons = list(sample_context.drivable_polygons)
        lane_centerlines = list(sample_context.lane_centerlines)
        if drivable_polygons:
            inside = torch.as_tensor(sample_context.drivable_map.batch_contains_points(centers_xy.detach().cpu().numpy()), device=device, dtype=torch.bool)
            drivable_area = inside.all(dim=1).to(dtype=torch.float32)
        else:
            drivable_area = torch.ones((num_candidates,), device=device, dtype=torch.float32)
        if self.score_mode == "drivable_area_only":
            return drivable_area

        if sample_context.centerline_segments_xy.size > 0:
            lateral_errors, tangents_xy = self._batch_centerline_stats_torch(
                centers_xy,
                self._tensor_from_numpy(sample_context.centerline_segments_xy, device=device),
                self._tensor_from_numpy(sample_context.centerline_tangents_xy, device=device),
            )
            mean_lateral = lateral_errors.mean(dim=1)
            lane_keeping = (1.0 - (mean_lateral / 2.0)).clamp(0.0, 1.0)
        else:
            tangents_xy = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32).view(1, 1, 2).expand(num_candidates, horizon, 2)
            lane_keeping = (1.0 - 0.25 * mean_err).clamp(0.0, 1.0) if lane_centerlines else torch.ones((num_candidates,), device=device, dtype=torch.float32)

        prev_xy = torch.cat([centers_xy[:, :1, :], centers_xy[:, :-1, :]], dim=1)
        step_delta = centers_xy - prev_xy
        step_norm = torch.linalg.norm(step_delta, dim=-1)
        move_dir = torch.where(
            step_norm.unsqueeze(-1) > 1.0e-6,
            step_delta / step_norm.unsqueeze(-1).clamp_min(1.0e-6),
            torch.zeros_like(step_delta),
        )
        reverse_alignment = torch.maximum(torch.zeros_like(step_norm), -(move_dir * tangents_xy).sum(dim=-1))
        reverse_mask = reverse_alignment > _DEFAULT_DIRECTION_REVERSE_ALIGNMENT_THRESHOLD
        step_indices = torch.arange(horizon, device=device, dtype=torch.int64).view(1, -1)
        last_false_idx = torch.cummax(torch.where(reverse_mask, torch.full_like(step_indices, -1), step_indices), dim=1).values
        streak_len = (step_indices - last_false_idx).to(dtype=torch.float32)
        continuous_reverse_time_s = torch.where(reverse_mask, streak_len * max(1.0e-6, float(dt_s)), torch.zeros_like(streak_len))
        accrue_mask = continuous_reverse_time_s > _DEFAULT_DIRECTION_MIN_CONTINUOUS_REVERSE_S
        oncoming_progress_m = torch.where(accrue_mask, step_norm * reverse_alignment, torch.zeros_like(step_norm)).sum(dim=1)
        good_threshold = float(_DEFAULT_DIRECTION_COMPLIANCE_THRESHOLD_M)
        bad_threshold = float(_DEFAULT_DIRECTION_VIOLATION_THRESHOLD_M)
        span = max(1.0e-6, bad_threshold - good_threshold)
        driving_direction = torch.where(
            oncoming_progress_m <= good_threshold,
            torch.ones_like(oncoming_progress_m),
            torch.where(
                oncoming_progress_m >= bad_threshold,
                torch.zeros_like(oncoming_progress_m),
                ((bad_threshold - oncoming_progress_m) / span).clamp(0.0, 1.0),
            ),
        )

        collision_ttc = self._batch_collision_ttc_metrics_torch(
            sample_context=sample_context,
            candidate_geometry=candidate_geometry,
            dt_s=float(dt_s),
        )
        no_collision = collision_ttc["no_collision"]
        ttc = collision_ttc["ttc"]

        weighted_score = (
            drivable_area * float(self._delegate.dac_weight)
            + progress_ratio * float(self._delegate.progress_weight)
            + ttc * float(self._delegate.ttc_weight)
            + lane_keeping * float(self._delegate.lane_keeping_weight)
            + history_comfort * float(self._delegate.history_comfort_weight)
        ) / max(
            1.0,
            float(self._delegate.dac_weight)
            + float(self._delegate.progress_weight)
            + float(self._delegate.ttc_weight)
            + float(self._delegate.lane_keeping_weight)
            + float(self._delegate.history_comfort_weight),
        )
        drivable_gate = drivable_area if bool(self._delegate.dac_gate_enabled) else torch.ones_like(drivable_area)
        multiplicative_product = no_collision * drivable_gate * driving_direction
        return weighted_score * multiplicative_product

    def _score_candidate_batch_for_sample(
        self,
        *,
        sample_context: NuScenesPDMSampleContext,
        candidate_geometry: dict[str, np.ndarray],
        gt_xy_cmp: np.ndarray,
        gt_yaw_cmp: np.ndarray,
        gt_xy_full: np.ndarray,
        gt_s_full: np.ndarray,
        gt_total_len: float,
        dt_s: float,
    ) -> np.ndarray:
        centers_xy = np.asarray(candidate_geometry["centers_xy"], dtype=np.float32)
        yaw_rad = np.asarray(candidate_geometry["yaw_rad"], dtype=np.float32)
        num_candidates, horizon = centers_xy.shape[:2]
        if num_candidates <= 0 or horizon <= 0:
            return np.zeros((num_candidates,), dtype=np.float32)

        gt_xy_cmp = np.asarray(gt_xy_cmp, dtype=np.float32)
        gt_yaw_cmp = np.asarray(gt_yaw_cmp, dtype=np.float32)
        lane_centerlines = list(sample_context.lane_centerlines)
        drivable_polygons = list(sample_context.drivable_polygons)

        pos_err = (
            np.linalg.norm(centers_xy - gt_xy_cmp.reshape(1, horizon, 2), axis=-1)
            if gt_xy_cmp.size
            else np.zeros((num_candidates, horizon), dtype=np.float32)
        )
        mean_err = pos_err.mean(axis=1) if pos_err.size else np.zeros((num_candidates,), dtype=np.float32)
        yaw_err = (
            np.abs(_wrap_angle(yaw_rad - gt_yaw_cmp.reshape(1, horizon)))
            if gt_yaw_cmp.size
            else np.zeros((num_candidates, horizon), dtype=np.float32)
        )
        mean_yaw_err = yaw_err.mean(axis=1) if yaw_err.size else np.zeros((num_candidates,), dtype=np.float32)

        cand_progress = self._batch_project_progress(centers_xy[:, -1], gt_xy_full, gt_s_full)
        progress_ratio = np.clip(cand_progress / max(1.0e-6, float(gt_total_len)), 0.0, 1.0).astype(np.float32, copy=False)

        cand_prev = np.concatenate([centers_xy[:, :1, :], centers_xy[:, :-1, :]], axis=1)
        cand_vel = centers_xy - cand_prev
        cand_acc = cand_vel[:, 1:, :] - cand_vel[:, :-1, :] if horizon > 1 else np.zeros((num_candidates, 0, 2), dtype=np.float32)
        cand_yaw_prev = np.concatenate([yaw_rad[:, :1], yaw_rad[:, :-1]], axis=1)
        cand_yaw_rate = np.abs(_wrap_angle(yaw_rad - cand_yaw_prev)) / max(1.0e-6, float(dt_s))
        smooth_pen = (
            np.linalg.norm(cand_acc, axis=-1).mean(axis=1).astype(np.float32, copy=False)
            if cand_acc.size
            else np.zeros((num_candidates,), dtype=np.float32)
        )
        comfort_cost = smooth_pen + 0.25 * cand_yaw_rate.mean(axis=1).astype(np.float32, copy=False)
        history_comfort = np.clip(1.0 - 0.08 * comfort_cost, 0.0, 1.0).astype(np.float32, copy=False)

        map_metrics = self._batch_map_metrics(
            sample_context=sample_context,
            candidate_geometry=candidate_geometry,
            dt_s=float(dt_s),
        )
        lane_keeping = map_metrics["lane_keeping"] if lane_centerlines else np.clip(1.0 - 0.25 * mean_err, 0.0, 1.0).astype(np.float32, copy=False)
        driving_direction = (
            map_metrics["driving_direction"]
            if bool(self._delegate.driving_direction_gate_enabled)
            else np.ones((num_candidates,), dtype=np.float32)
        )
        drivable_area = map_metrics["drivable_area"] if drivable_polygons else np.ones((num_candidates,), dtype=np.float32)
        if self.score_mode == "drivable_area_only":
            return drivable_area.astype(np.float32, copy=False)

        collision_ttc = self._batch_collision_ttc_metrics(
            sample_context=sample_context,
            candidate_geometry=candidate_geometry,
            dt_s=float(dt_s),
        )
        no_collision = collision_ttc["no_collision"]
        ttc = collision_ttc["ttc"]

        weighted_score = (
            drivable_area * float(self._delegate.dac_weight)
            + progress_ratio * float(self._delegate.progress_weight)
            + ttc * float(self._delegate.ttc_weight)
            + lane_keeping * float(self._delegate.lane_keeping_weight)
            + history_comfort * float(self._delegate.history_comfort_weight)
        ) / max(
            1.0,
            float(self._delegate.dac_weight)
            + float(self._delegate.progress_weight)
            + float(self._delegate.ttc_weight)
            + float(self._delegate.lane_keeping_weight)
            + float(self._delegate.history_comfort_weight),
        )
        drivable_gate = drivable_area if bool(self._delegate.dac_gate_enabled) else np.ones_like(drivable_area)
        multiplicative_product = (no_collision * drivable_gate * driving_direction).astype(np.float32, copy=False)
        return (weighted_score.astype(np.float32, copy=False) * multiplicative_product).astype(np.float32, copy=False)

    def _build_sample_context(
        self,
        replay: dict[str, Any],
        *,
        patch_radius: float,
    ) -> NuScenesPDMSampleContext:
        sample_token = replay.get("sample_token", None)
        if sample_token is None:
            raise RuntimeError("NuScenesPDMScorer requires replay['sample_token']")
        sample_token_str = str(sample_token)

        cached = self._sample_context_cache.get(sample_token_str, None)
        if cached is not None:
            return cached

        persisted = self._load_persisted_sample_context(sample_token_str)
        if persisted is not None:
            self._sample_context_cache[sample_token_str] = persisted
            return persisted

        static_ctx = self._delegate._build_static_sample_context(
            replay,
            patch_radius=float(patch_radius),
        )
        map_layers = dict(static_ctx.get("map_context", {}).get("layers", {}))
        scene_objects = list(static_ctx.get("scene_objects", []))
        object_tokens, object_polygons, object_velocity_xy, occupancy_map = self._build_object_geometry_arrays(scene_objects)
        lane_centerlines = list(map_layers.get("lane_centerline", []))
        drivable_polygons = list(map_layers.get("drivable_area", []))
        centerline_segments_xy, centerline_tangents_xy = self._build_centerline_segment_cache(lane_centerlines)
        sample_context = NuScenesPDMSampleContext(
            sample_token=sample_token_str,
            patch_radius=float(static_ctx.get("map_context", {}).get("patch_radius", patch_radius)),
            static_context=static_ctx,
            drivable_polygons=drivable_polygons,
            lane_centerlines=lane_centerlines,
            scene_objects=scene_objects,
            ea_agent_states=list(static_ctx.get("ea_agent_states", [])),
            object_tokens=object_tokens,
            object_polygons=object_polygons,
            object_velocity_xy=object_velocity_xy,
            occupancy_map=occupancy_map,
            drivable_map=NuScenesPDMDrivableMap(drivable_polygons),
            centerline_segments_xy=centerline_segments_xy,
            centerline_tangents_xy=centerline_tangents_xy,
        )
        self._sample_context_cache[sample_token_str] = sample_context
        self._save_persisted_sample_context(
            self._serialize_sample_context_payload(
                sample_token=sample_token_str,
                patch_radius=float(sample_context.patch_radius),
                static_context=static_ctx,
                drivable_polygons=drivable_polygons,
                lane_centerlines=lane_centerlines,
                scene_objects=scene_objects,
                ea_agent_states=list(static_ctx.get("ea_agent_states", [])),
                object_tokens=object_tokens,
                object_velocity_xy=object_velocity_xy,
                centerline_segments_xy=centerline_segments_xy,
                centerline_tangents_xy=centerline_tangents_xy,
            )
        )
        return sample_context

    def score(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> np.ndarray:
        if traj_xyyaw.ndim != 4 or int(traj_xyyaw.shape[-1]) < 3:
            raise RuntimeError(
                "NuScenesPDMScorer expects traj_xyyaw with shape (batch, candidates, horizon, 3); "
                f"got {tuple(traj_xyyaw.shape)}"
            )
        if len(replays) != int(traj_xyyaw.shape[0]):
            raise RuntimeError(f"Replay batch length mismatch: replays={len(replays)} traj_batch={int(traj_xyyaw.shape[0])}")

        use_gpu_path = bool(torch.cuda.is_available()) and bool(traj_xyyaw.is_cuda)
        if use_gpu_path:
            geometry_batch_t = self._build_candidate_geometry_batch_torch(traj_xyyaw)
            scores_t = torch.zeros((int(traj_xyyaw.shape[0]), int(traj_xyyaw.shape[1])), device=traj_xyyaw.device, dtype=torch.float32)
        else:
            geometry_batch = self._build_candidate_geometry_batch(traj_xyyaw)
            scores = np.zeros((int(traj_xyyaw.shape[0]), int(traj_xyyaw.shape[1])), dtype=np.float32)

        for batch_idx, replay in enumerate(replays):
            sample_context = self._build_sample_context(replay, patch_radius=_DEFAULT_PATCH_RADIUS_M)
            static_ctx = dict(sample_context.static_context)
            gt_xy = np.asarray(static_ctx.get("gt_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            gt_yaw = np.asarray(static_ctx.get("gt_yaw", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
            gt_s = np.asarray(static_ctx.get("gt_s", _polyline_arclength(gt_xy)), dtype=np.float32)
            gt_total_len = float(static_ctx.get("gt_total_len", float(gt_s[-1]) if gt_s.size else 0.0))
            horizon = min(int(traj_xyyaw.shape[2]), int(gt_xy.shape[0])) if gt_xy.size else int(traj_xyyaw.shape[2])
            gt_xy_cmp = gt_xy[:horizon]
            gt_yaw_cmp = gt_yaw[:horizon] if gt_yaw.size else _path_yaw_from_xy(gt_xy_cmp)
            if use_gpu_path:
                candidate_geometry_t = {
                    key: value[batch_idx, :, :horizon]
                    for key, value in geometry_batch_t.items()
                }
                sample_scores_t = self._score_candidate_batch_for_sample_torch(
                    sample_context=sample_context,
                    candidate_geometry=candidate_geometry_t,
                    gt_xy_cmp=gt_xy_cmp,
                    gt_yaw_cmp=gt_yaw_cmp,
                    gt_xy_full=gt_xy,
                    gt_s_full=gt_s,
                    gt_total_len=gt_total_len,
                    dt_s=0.5,
                )
                scores_t[batch_idx, : int(sample_scores_t.shape[0])] = sample_scores_t
            else:
                candidate_geometry = {
                    key: value[batch_idx, :, :horizon].copy()
                    for key, value in geometry_batch.items()
                }
                sample_scores = self._score_candidate_batch_for_sample(
                    sample_context=sample_context,
                    candidate_geometry=candidate_geometry,
                    gt_xy_cmp=gt_xy_cmp,
                    gt_yaw_cmp=gt_yaw_cmp,
                    gt_xy_full=gt_xy,
                    gt_s_full=gt_s,
                    gt_total_len=gt_total_len,
                    dt_s=0.5,
                )
                scores[batch_idx, : int(sample_scores.shape[0])] = sample_scores.astype(np.float32, copy=False)
        if use_gpu_path:
            return scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
        return scores

    def dump_debug_artifacts(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
        *,
        out_dir: str,
        step_tag: str,
        top_k: int,
    ) -> None:
        self._delegate.dump_debug_artifacts(
            replays,
            traj_xyyaw,
            out_dir=out_dir,
            step_tag=step_tag,
            top_k=top_k,
        )


__all__ = ["NuScenesPDMScorer", "NuScenesPDMSampleContext", "NuScenesPDMOccupancyMap"]
