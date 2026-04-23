from __future__ import annotations

import math
import os
import pickle
import tempfile
from dataclasses import dataclass
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
    ttc_agent_states: list[dict[str, Any]]
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
        if pts.ndim not in {3, 4} or pts.shape[-1] != 2:
            raise RuntimeError(
                "Expected points_xy shape (candidates,horizon,2) or (candidates,horizon,num_points,2), "
                f"got {tuple(pts.shape)}"
            )
        num_candidates, horizon = pts.shape[:2]
        if not self._polygons_xy:
            if pts.ndim == 3:
                return np.ones((num_candidates, horizon), dtype=bool)
            return np.ones((num_candidates, horizon, pts.shape[2]), dtype=bool)
        point_geoms = shapely_points(pts[..., 0], pts[..., 1])
        if pts.ndim == 3:
            contains_mask = shapely_contains(self._polygons[:, None, None], point_geoms[None, ...])
        else:
            contains_mask = shapely_contains(self._polygons[:, None, None, None], point_geoms[None, ...])
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
        self._delegate = NuScenesTokenScorer(
            token2vad_path=self.token2vad_path,
            **kwargs,
        )
        self._sample_context_cache: dict[str, NuScenesPDMSampleContext] = {}
        self._derived_context_cache_root = self._delegate.scene_cache_root / "_sample_pdm_context"

    def _derived_context_cache_variant(self) -> str:
        return f"pdm-v4-ea{int(bool(self._delegate.ea_gate_enabled))}"

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
        ttc_agent_states: list[dict[str, Any]],
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
            "ttc_agent_states": ttc_agent_states,
            "object_tokens": np.asarray(object_tokens, dtype=object),
            "object_velocity_xy": np.asarray(object_velocity_xy, dtype=np.float32),
            "centerline_segments_xy": np.asarray(centerline_segments_xy, dtype=np.float32),
            "centerline_tangents_xy": np.asarray(centerline_tangents_xy, dtype=np.float32),
        }

    def _deserialize_sample_context_payload(self, payload: dict[str, Any]) -> NuScenesPDMSampleContext:
        scene_objects = list(payload.get("scene_objects", []))
        object_tokens, object_polygons, object_velocity_xy, occupancy_map = self._build_object_geometry_arrays(scene_objects)
        drivable_polygons = list(payload.get("drivable_polygons", []))
        static_context = dict(payload["static_context"])
        patch_radius = float(payload["patch_radius"])
        ea_agent_states = list(payload.get("ea_agent_states", []))
        ttc_agent_states = list(payload.get("ttc_agent_states", []))
        if not ttc_agent_states:
            ttc_agent_states = self._build_ttc_agent_states(
                static_context=static_context,
                scene_objects=scene_objects,
                ea_agent_states=ea_agent_states,
                patch_radius=patch_radius,
            )
        return NuScenesPDMSampleContext(
            sample_token=str(payload["sample_token"]),
            patch_radius=patch_radius,
            static_context=static_context,
            drivable_polygons=drivable_polygons,
            lane_centerlines=list(payload.get("lane_centerlines", [])),
            scene_objects=scene_objects,
            ea_agent_states=ea_agent_states,
            ttc_agent_states=ttc_agent_states,
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

    def _build_ttc_agent_states(
        self,
        *,
        static_context: dict[str, Any],
        scene_objects: Sequence[dict[str, Any]],
        ea_agent_states: Sequence[dict[str, Any]],
        patch_radius: float,
    ) -> list[dict[str, Any]]:
        row = static_context.get("row", None)
        if isinstance(row, dict):
            truth_states = self._delegate._lookup_ea_agent_future_truth(row, patch_radius=float(patch_radius))
            if len(truth_states) > 0:
                return [dict(item) for item in truth_states]
        if len(ea_agent_states) > 0:
            return [dict(item) for item in ea_agent_states]
        return [dict(item) for item in scene_objects]

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

    @staticmethod
    def _candidate_ea_state_dynamics_batch(
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
        prev_xy = np.concatenate([np.zeros_like(centers_xy[:, :1, :]), centers_xy[:, :-1, :]], axis=1)
        step_delta = centers_xy - prev_xy
        speed_mps = np.linalg.norm(step_delta, axis=-1) / max(1.0e-6, float(dt_s))
        prev_yaw = np.concatenate([np.zeros_like(yaw_rad[:, :1]), yaw_rad[:, :-1]], axis=1)
        yaw_rate_rps = _wrap_angle(yaw_rad - prev_yaw) / max(1.0e-6, float(dt_s))
        return speed_mps.astype(np.float32, copy=False), yaw_rate_rps.astype(np.float32, copy=False)

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

    @staticmethod
    def _box_corners_batch(
        centers_xy: np.ndarray,
        *,
        lengths_m: np.ndarray,
        widths_m: np.ndarray,
        yaw_rad: np.ndarray,
    ) -> np.ndarray:
        centers = np.asarray(centers_xy, dtype=np.float32)
        lengths = np.asarray(lengths_m, dtype=np.float32)
        widths = np.asarray(widths_m, dtype=np.float32)
        yaw = np.asarray(yaw_rad, dtype=np.float32)
        dx = lengths * 0.5
        dy = widths * 0.5
        template = np.stack(
            [
                np.stack([dx, dy], axis=-1),
                np.stack([dx, -dy], axis=-1),
                np.stack([-dx, -dy], axis=-1),
                np.stack([-dx, dy], axis=-1),
            ],
            axis=-2,
        ).astype(np.float32, copy=False)
        cos_yaw = np.cos(yaw).astype(np.float32, copy=False)
        sin_yaw = np.sin(yaw).astype(np.float32, copy=False)
        rot = np.stack(
            [
                np.stack([cos_yaw, -sin_yaw], axis=-1),
                np.stack([sin_yaw, cos_yaw], axis=-1),
            ],
            axis=-2,
        ).astype(np.float32, copy=False)
        rotated = np.einsum("...qd,...dc->...qc", template, rot, optimize=True)
        return (rotated + centers[..., None, :]).astype(np.float32, copy=False)

    @staticmethod
    def _oriented_box_axes_np(corners_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edge0 = corners_xy[..., 1, :] - corners_xy[..., 0, :]
        edge1 = corners_xy[..., 3, :] - corners_xy[..., 0, :]
        axis0 = edge0 / np.maximum(np.linalg.norm(edge0, axis=-1, keepdims=True), 1.0e-6)
        axis1 = edge1 / np.maximum(np.linalg.norm(edge1, axis=-1, keepdims=True), 1.0e-6)
        return axis0.astype(np.float32, copy=False), axis1.astype(np.float32, copy=False)

    @classmethod
    def _obb_intersects_np(cls, boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        center_a = boxes_a.mean(axis=-2)
        center_b = boxes_b.mean(axis=-2)
        axis_a0, axis_a1 = cls._oriented_box_axes_np(boxes_a)
        axis_b0, axis_b1 = cls._oriented_box_axes_np(boxes_b)
        target_shape = np.broadcast_shapes(axis_a0.shape, axis_a1.shape, axis_b0.shape, axis_b1.shape)
        axis_a0 = np.broadcast_to(axis_a0, target_shape)
        axis_a1 = np.broadcast_to(axis_a1, target_shape)
        axis_b0 = np.broadcast_to(axis_b0, target_shape)
        axis_b1 = np.broadcast_to(axis_b1, target_shape)
        axes = np.stack([axis_a0, axis_a1, axis_b0, axis_b1], axis=-2).astype(np.float32, copy=False)
        rel = center_b - center_a
        proj_center = np.abs(np.sum(rel[..., None, :] * axes, axis=-1)).astype(np.float32, copy=False)
        centered_a = boxes_a[..., None, :, :] - center_a[..., None, None, :]
        centered_b = boxes_b[..., None, :, :] - center_b[..., None, None, :]
        proj_a = np.abs(np.sum(centered_a * axes[..., :, None, :], axis=-1)).astype(np.float32, copy=False)
        proj_b = np.abs(np.sum(centered_b * axes[..., :, None, :], axis=-1)).astype(np.float32, copy=False)
        radius_a = np.max(proj_a, axis=-1)
        radius_b = np.max(proj_b, axis=-1)
        overlap = proj_center <= (radius_a + radius_b + 1.0e-5)
        return np.all(overlap, axis=-1)

    def _sample_agent_boxes_at_times(
        self,
        agent_states: Sequence[dict[str, Any]],
        query_times_s: np.ndarray,
        *,
        default_dt_s: float,
    ) -> np.ndarray:
        times = np.asarray(query_times_s, dtype=np.float32)
        if len(agent_states) <= 0:
            return np.zeros((*times.shape, 0, 4, 2), dtype=np.float32)

        flat_times = times.reshape(-1).astype(np.float32, copy=False)
        num_times = int(flat_times.shape[0])
        num_agents = len(agent_states)
        centers = np.zeros((num_times, num_agents, 2), dtype=np.float32)
        yaws = np.zeros((num_times, num_agents), dtype=np.float32)
        lengths = np.ones((num_times, num_agents), dtype=np.float32)
        widths = np.ones((num_times, num_agents), dtype=np.float32)

        for agent_idx, agent in enumerate(agent_states):
            current_state = self._agent_current_state(agent)
            future_xy = np.asarray(agent.get("future_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            future_yaw = np.asarray(agent.get("future_yaw", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
            future_dt_s = float(agent.get("future_dt_s", default_dt_s))

            sampled_x = np.full((num_times,), float(current_state["x"]), dtype=np.float32)
            sampled_y = np.full((num_times,), float(current_state["y"]), dtype=np.float32)
            sampled_yaw = np.full((num_times,), float(current_state["yaw_rad"]), dtype=np.float32)

            use_ctrv = np.ones((num_times,), dtype=bool)
            if future_xy.ndim == 2 and int(future_xy.shape[0]) > 0 and int(future_xy.shape[1]) >= 2:
                sample_dt = max(1.0e-6, future_dt_s)
                future_times = np.arange(1, int(future_xy.shape[0]) + 1, dtype=np.float32) * sample_dt
                valid_truth = flat_times <= float(future_times[-1]) + 1.0e-6
                if bool(np.any(valid_truth)):
                    series_times = np.concatenate([np.asarray([0.0], dtype=np.float32), future_times], axis=0)
                    series_x = np.concatenate(
                        [np.asarray([float(current_state["x"])], dtype=np.float32), future_xy[:, 0].astype(np.float32, copy=False)],
                        axis=0,
                    )
                    series_y = np.concatenate(
                        [np.asarray([float(current_state["y"])], dtype=np.float32), future_xy[:, 1].astype(np.float32, copy=False)],
                        axis=0,
                    )
                    sampled_x[valid_truth] = np.interp(flat_times[valid_truth], series_times, series_x).astype(np.float32, copy=False)
                    sampled_y[valid_truth] = np.interp(flat_times[valid_truth], series_times, series_y).astype(np.float32, copy=False)

                    if int(future_yaw.shape[0]) == int(future_xy.shape[0]):
                        yaw_series = np.concatenate(
                            [
                                np.asarray([float(current_state["yaw_rad"])], dtype=np.float32),
                                future_yaw.astype(np.float32, copy=False),
                            ],
                            axis=0,
                        )
                    else:
                        derived_yaw = _path_yaw_from_xy(
                            np.concatenate(
                                [
                                    np.asarray([[float(current_state["x"]), float(current_state["y"])]], dtype=np.float32),
                                    future_xy[:, :2].astype(np.float32, copy=False),
                                ],
                                axis=0,
                            )
                        )
                        yaw_series = np.asarray(derived_yaw, dtype=np.float32)
                        if int(yaw_series.shape[0]) > 0:
                            yaw_series[0] = float(current_state["yaw_rad"])
                    sampled_yaw[valid_truth] = np.interp(
                        flat_times[valid_truth],
                        series_times,
                        np.unwrap(yaw_series.astype(np.float64)),
                    ).astype(np.float32, copy=False)
                    use_ctrv[valid_truth] = False

            if bool(np.any(use_ctrv)):
                ctrv_times = flat_times[use_ctrv].astype(np.float32, copy=False)
                x0 = float(current_state["x"])
                y0 = float(current_state["y"])
                speed = float(current_state["speed_mps"])
                yaw0 = float(current_state["yaw_rad"])
                yaw_rate = float(current_state["yaw_rate_rps"])
                if abs(yaw_rate) <= 1.0e-6:
                    sampled_x[use_ctrv] = x0 + speed * math.cos(yaw0) * ctrv_times
                    sampled_y[use_ctrv] = y0 + speed * math.sin(yaw0) * ctrv_times
                    sampled_yaw[use_ctrv] = yaw0
                else:
                    radius = speed / yaw_rate
                    delta_yaw = yaw_rate * ctrv_times
                    sampled_x[use_ctrv] = x0 + radius * (np.sin(yaw0 + delta_yaw) - math.sin(yaw0))
                    sampled_y[use_ctrv] = y0 - radius * (np.cos(yaw0 + delta_yaw) - math.cos(yaw0))
                    sampled_yaw[use_ctrv] = yaw0 + delta_yaw

            centers[:, agent_idx, 0] = sampled_x
            centers[:, agent_idx, 1] = sampled_y
            yaws[:, agent_idx] = np.arctan2(np.sin(sampled_yaw), np.cos(sampled_yaw)).astype(np.float32, copy=False)
            lengths[:, agent_idx] = float(agent.get("length_m", current_state["length_m"]))
            widths[:, agent_idx] = float(agent.get("width_m", current_state["width_m"]))

        corners = self._box_corners_batch(
            centers,
            lengths_m=lengths,
            widths_m=widths,
            yaw_rad=yaws,
        )
        return corners.reshape(*times.shape, num_agents, 4, 2).astype(np.float32, copy=False)

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

    def _batch_map_metrics(
        self,
        *,
        sample_context: NuScenesPDMSampleContext,
        candidate_geometry: dict[str, np.ndarray],
        dt_s: float,
    ) -> dict[str, np.ndarray]:
        centers_xy = np.asarray(candidate_geometry["centers_xy"], dtype=np.float32)
        corners_xy = np.asarray(candidate_geometry["corners_xy"], dtype=np.float32)
        num_candidates, horizon = centers_xy.shape[:2]
        if num_candidates <= 0 or horizon <= 0:
            return {
                "drivable_area": np.ones((num_candidates,), dtype=np.float32),
                "lane_keeping": np.ones((num_candidates,), dtype=np.float32),
                "driving_direction": np.ones((num_candidates,), dtype=np.float32),
            }

        inside = sample_context.drivable_map.batch_contains_points(corners_xy)
        drivable_area = inside.all(axis=(1, 2)).astype(np.float32, copy=False)

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
            if len(sample_context.ttc_agent_states) <= 0:
                return {"no_collision": no_collision, "ttc": ttc}

        speed_mps, _ = self._candidate_speed_and_yaw_rate_batch(centers_xy, yaw_rad, dt_s=float(dt_s))
        ttc_projection = self._build_ttc_projection_geometry(
            centers_xy=centers_xy,
            yaw_rad=yaw_rad,
            dt_s=float(dt_s),
        )
        step_times_s = (np.arange(horizon, dtype=np.float32) + 1.0) * float(dt_s)
        if len(sample_context.ttc_agent_states) > 0:
            agent_step_corners = self._sample_agent_boxes_at_times(
                sample_context.ttc_agent_states,
                step_times_s,
                default_dt_s=float(dt_s),
            )
            if agent_step_corners.shape[1] > 0:
                ego_now = np.asarray(candidate_geometry["corners_xy"], dtype=np.float32)[:, :, None, :, :]
                agent_now = agent_step_corners[None, :, :, :, :]
                collision_hits = np.any(self._obb_intersects_np(ego_now, agent_now), axis=2)
            else:
                collision_hits = np.zeros((num_candidates, horizon), dtype=bool)
        else:
            collision_hits = self._query_hits_per_candidate_grid(
                sample_context.occupancy_map,
                polygons,
                predicate="intersects",
            )
        collision_mask = np.any(collision_hits, axis=1)
        earliest_ttc_risk_s = np.full((num_candidates,), np.inf, dtype=np.float32)

        moving_mask = speed_mps >= float(_DEFAULT_TTC_STOPPED_SPEED_THRESHOLD_MPS)
        if bool(np.any(moving_mask)):
            offsets_s = np.asarray(ttc_projection["offsets_s"], dtype=np.float32)
            if len(sample_context.ttc_agent_states) > 0 and int(offsets_s.shape[0]) > 0:
                query_times_s = step_times_s[:, None] + offsets_s.reshape(1, -1)
                agent_future_corners = self._sample_agent_boxes_at_times(
                    sample_context.ttc_agent_states,
                    query_times_s,
                    default_dt_s=float(dt_s),
                )
                if agent_future_corners.shape[2] > 0:
                    ego_future = np.asarray(ttc_projection["corners_xy"], dtype=np.float32)[:, :, :, None, :, :]
                    agent_future = agent_future_corners[None, :, :, :, :, :]
                    future_hits = self._obb_intersects_np(ego_future, agent_future)
                    proj_hits = np.any(future_hits, axis=3)
                else:
                    proj_hits = np.zeros((num_candidates, horizon, int(offsets_s.shape[0])), dtype=bool)
            else:
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
                    offsets_s.reshape(1, 1, -1),
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
    def _agent_current_state(agent: dict[str, Any]) -> dict[str, float]:
        agent_velocity = np.asarray(agent.get("velocity_xy", [0.0, 0.0]), dtype=np.float32).reshape(2)
        center_xy = np.asarray(agent.get("center_xy", [0.0, 0.0]), dtype=np.float32).reshape(2)
        return {
            "x": float(center_xy[0]),
            "y": float(center_xy[1]),
            "speed_mps": float(agent.get("speed_mps", np.linalg.norm(agent_velocity))),
            "yaw_rad": float(agent.get("yaw_rad", 0.0)),
            "yaw_rate_rps": float(agent.get("yaw_rate_rps", 0.0)),
            "length_m": float(agent.get("length_m", 1.0)),
            "width_m": float(agent.get("width_m", 1.0)),
        }

    def _sample_agent_state_at_time(
        self,
        agent: dict[str, Any],
        *,
        time_s: float,
        dt_s: float,
    ) -> dict[str, Any]:
        current_state = self._agent_current_state(agent)
        future_xy = np.asarray(agent.get("future_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        future_yaw = np.asarray(agent.get("future_yaw", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        sampled_state = self._delegate._sample_state_at_time(
            current_state=current_state,
            future_xy=future_xy,
            future_yaw=future_yaw,
            time_s=float(time_s),
            dt_s=float(agent.get("future_dt_s", dt_s)),
        )
        if sampled_state is not None:
            return sampled_state
        return self._delegate._propagate_ctrv_state(current_state, time_s=float(time_s))

    def _compute_ea_value_batch_for_pairs(
        self,
        ego_states: Sequence[dict[str, Any]],
        agent_states: Sequence[dict[str, Any]],
    ) -> np.ndarray:
        if len(ego_states) != len(agent_states):
            raise RuntimeError(
                f"EA pair batch length mismatch: ego_states={len(ego_states)} agent_states={len(agent_states)}"
            )
        if len(ego_states) <= 0:
            return np.zeros((0,), dtype=np.float32)
        out = np.zeros((len(ego_states),), dtype=np.float32)
        for idx, (ego_state, agent_state) in enumerate(zip(ego_states, agent_states, strict=False)):
            out[idx] = np.float32(self._delegate._compute_ea_value_for_pair(ego_state, agent_state))
        return out

    def _score_ea_safety_gate_batch(
        self,
        *,
        sample_context: NuScenesPDMSampleContext,
        candidate_geometry: dict[str, np.ndarray],
        dt_s: float,
    ) -> dict[str, np.ndarray]:
        centers_xy = np.asarray(candidate_geometry["centers_xy"], dtype=np.float32)
        yaw_rad = np.asarray(candidate_geometry["yaw_rad"], dtype=np.float32)
        num_candidates, horizon = centers_xy.shape[:2]

        gate = np.ones((num_candidates,), dtype=np.float32)
        max_ea = np.zeros((num_candidates,), dtype=np.float32)
        evaluated_pairs = np.zeros((num_candidates,), dtype=np.float32)
        if not self._delegate.ea_gate_enabled or num_candidates <= 0:
            return {"gate": gate, "max_ea": max_ea, "evaluated_pairs": evaluated_pairs}

        vehicle_agents = [
            dict(item)
            for item in sample_context.ea_agent_states
            if "vehicle" in str(item.get("category", "")).strip().lower()
        ]
        if len(vehicle_agents) <= 0:
            return {"gate": gate, "max_ea": max_ea, "evaluated_pairs": evaluated_pairs}
        if horizon <= 0:
            return {
                "gate": np.zeros((num_candidates,), dtype=np.float32),
                "max_ea": max_ea,
                "evaluated_pairs": evaluated_pairs,
            }

        agent_centers = np.asarray(
            [np.asarray(agent.get("center_xy", [0.0, 0.0]), dtype=np.float32).reshape(2) for agent in vehicle_agents],
            dtype=np.float32,
        )
        path_agent_dist = np.linalg.norm(
            centers_xy[:, :, None, :] - agent_centers[None, None, :, :],
            axis=-1,
        ).min(axis=1)
        max_agents = min(int(self._delegate.ea_gate_max_agents), int(agent_centers.shape[0]))
        if max_agents <= 0:
            return {"gate": gate, "max_ea": max_ea, "evaluated_pairs": evaluated_pairs}
        selected_agent_indices = np.argsort(path_agent_dist, axis=1)[:, :max_agents]

        speed_mps, yaw_rate_rps = self._candidate_ea_state_dynamics_batch(
            centers_xy,
            yaw_rad,
            dt_s=float(dt_s),
        )
        step_indices = sorted({0, max(0, horizon // 2), horizon - 1})

        flat_candidate_indices: list[int] = []
        flat_ego_states: list[dict[str, Any]] = []
        flat_agent_states: list[dict[str, Any]] = []
        for candidate_idx in range(num_candidates):
            for step_idx in step_indices:
                time_offset_s = float(step_idx + 1) * float(dt_s)
                ego_state = {
                    "x": float(centers_xy[candidate_idx, step_idx, 0]),
                    "y": float(centers_xy[candidate_idx, step_idx, 1]),
                    "speed_mps": float(speed_mps[candidate_idx, step_idx]),
                    "yaw_rad": float(yaw_rad[candidate_idx, step_idx]),
                    "yaw_rate_rps": float(yaw_rate_rps[candidate_idx, step_idx]),
                    "length_m": 4.9,
                    "width_m": 2.1,
                }
                for agent_idx in selected_agent_indices[candidate_idx].tolist():
                    flat_candidate_indices.append(candidate_idx)
                    flat_ego_states.append(ego_state)
                    flat_agent_states.append(
                        self._sample_agent_state_at_time(
                            vehicle_agents[int(agent_idx)],
                            time_s=time_offset_s,
                            dt_s=float(dt_s),
                        )
                    )

        if len(flat_candidate_indices) <= 0:
            return {"gate": gate, "max_ea": max_ea, "evaluated_pairs": evaluated_pairs}

        try:
            ea_values = np.asarray(
                self._compute_ea_value_batch_for_pairs(flat_ego_states, flat_agent_states),
                dtype=np.float32,
            ).reshape(-1)
        except Exception:
            return {"gate": gate, "max_ea": max_ea, "evaluated_pairs": evaluated_pairs}

        for pair_idx, candidate_idx in enumerate(flat_candidate_indices):
            ea_value = float(ea_values[pair_idx])
            evaluated_pairs[candidate_idx] += 1.0
            if not math.isfinite(ea_value):
                gate[candidate_idx] = min(gate[candidate_idx], 0.0)
                max_ea[candidate_idx] = np.float32(np.inf)
                continue
            clipped_ea = max(0.0, ea_value)
            max_ea[candidate_idx] = max(max_ea[candidate_idx], np.float32(clipped_ea))
            gate[candidate_idx] = min(
                gate[candidate_idx],
                np.float32(
                    _linear_decay_score(
                        clipped_ea,
                        good_threshold=float(self._delegate.ea_gate_good_threshold),
                        bad_threshold=float(self._delegate.ea_gate_bad_threshold),
                    )
                ),
            )
        return {"gate": gate, "max_ea": max_ea, "evaluated_pairs": evaluated_pairs}

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

        collision_ttc = self._batch_collision_ttc_metrics(
            sample_context=sample_context,
            candidate_geometry=candidate_geometry,
            dt_s=float(dt_s),
        )
        no_collision = collision_ttc["no_collision"]
        ttc = collision_ttc["ttc"]
        ea_gate = self._score_ea_safety_gate_batch(
            sample_context=sample_context,
            candidate_geometry=candidate_geometry,
            dt_s=float(dt_s),
        )

        weighted_score = (
            progress_ratio * float(self._delegate.progress_weight)
            + ttc * float(self._delegate.ttc_weight)
            + lane_keeping * float(self._delegate.lane_keeping_weight)
            + history_comfort * float(self._delegate.history_comfort_weight)
        ) / max(
            1.0,
            float(self._delegate.progress_weight)
            + float(self._delegate.ttc_weight)
            + float(self._delegate.lane_keeping_weight)
            + float(self._delegate.history_comfort_weight),
        )
        multiplicative_product = (no_collision * drivable_area * driving_direction).astype(np.float32, copy=False)
        if self._delegate.ea_gate_enabled:
            multiplicative_product = (
                multiplicative_product * np.asarray(ea_gate["gate"], dtype=np.float32)
            ).astype(np.float32, copy=False)
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
        ea_agent_states = list(static_ctx.get("ea_agent_states", []))
        ttc_agent_states = self._build_ttc_agent_states(
            static_context=static_ctx,
            scene_objects=scene_objects,
            ea_agent_states=ea_agent_states,
            patch_radius=float(static_ctx.get("map_context", {}).get("patch_radius", patch_radius)),
        )
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
            ea_agent_states=ea_agent_states,
            ttc_agent_states=ttc_agent_states,
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
                ea_agent_states=ea_agent_states,
                ttc_agent_states=ttc_agent_states,
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
            candidate_geometry = {
                key: value[batch_idx, :, :horizon].copy()
                for key, value in geometry_batch.items()
            }
            gt_xy_cmp = gt_xy[:horizon]
            gt_yaw_cmp = gt_yaw[:horizon] if gt_yaw.size else _path_yaw_from_xy(gt_xy_cmp)
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
