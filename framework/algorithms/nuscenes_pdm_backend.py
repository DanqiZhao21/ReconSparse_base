from __future__ import annotations

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

    def batch_contains_points(self, points_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 3 or pts.shape[-1] != 2:
            raise RuntimeError(f"Expected points_xy shape (candidates,horizon,2), got {tuple(pts.shape)}")
        num_candidates, horizon = pts.shape[:2]
        if not self._polygons_xy:
            return np.ones((num_candidates, horizon), dtype=bool)
        inside = np.zeros((num_candidates, horizon), dtype=bool)
        for cand_idx in range(num_candidates):
            for step_idx in range(horizon):
                inside[cand_idx, step_idx] = NuScenesTokenScorer._point_in_polygons(pts[cand_idx, step_idx], self._polygons_xy)
        return inside


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
        batch_size, num_candidates, horizon = centers_xy.shape[:3]
        corners_xy = np.zeros((batch_size, num_candidates, horizon, 4, 2), dtype=np.float32)
        polygons = np.empty((batch_size, num_candidates, horizon), dtype=object)

        for batch_idx in range(batch_size):
            for cand_idx in range(num_candidates):
                for step_idx in range(horizon):
                    xy = centers_xy[batch_idx, cand_idx, step_idx]
                    yaw = float(yaw_rad[batch_idx, cand_idx, step_idx])
                    corners = self._delegate._ego_corners_from_state(np.asarray(xy, dtype=np.float32), yaw)
                    corners_xy[batch_idx, cand_idx, step_idx] = corners.astype(np.float32, copy=False)
                    polygons[batch_idx, cand_idx, step_idx] = self._polygon_from_xy(corners)

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
        proj_centers = np.zeros((num_candidates, horizon, num_offsets, 2), dtype=np.float32)
        proj_corners = np.zeros((num_candidates, horizon, num_offsets, 4, 2), dtype=np.float32)
        proj_polygons = np.empty((num_candidates, horizon, num_offsets), dtype=object)

        speed_mps, _ = self._candidate_speed_and_yaw_rate_batch(centers, yaw, dt_s=float(dt_s))
        heading_vec = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1).astype(np.float32, copy=False)

        for offset_idx, future_offset_s in enumerate(offsets_s):
            proj_centers[:, :, offset_idx, :] = (
                centers + heading_vec * speed_mps[..., None] * float(future_offset_s)
            ).astype(np.float32, copy=False)
            for cand_idx in range(num_candidates):
                for step_idx in range(horizon):
                    corners = self._delegate._ego_corners_from_state(
                        proj_centers[cand_idx, step_idx, offset_idx],
                        float(yaw[cand_idx, step_idx]),
                    )
                    proj_corners[cand_idx, step_idx, offset_idx] = corners.astype(np.float32, copy=False)
                    proj_polygons[cand_idx, step_idx, offset_idx] = self._polygon_from_xy(corners)

        return {
            "offsets_s": offsets_s,
            "centers_xy": proj_centers,
            "corners_xy": proj_corners,
            "polygons": proj_polygons,
        }

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

        best_dist = np.full((num_candidates, horizon), np.inf, dtype=np.float32)
        best_tangent = np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (num_candidates, horizon, 1))

        for seg_idx in range(int(segments_xy.shape[0])):
            denom = max(1.0e-6, float(seg_len_sq[seg_idx]))
            p0 = seg_start[seg_idx].reshape(1, 1, 2)
            seg = seg_vec[seg_idx].reshape(1, 1, 2)
            alpha = np.sum((pts - p0) * seg, axis=-1) / denom
            alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)
            proj = p0 + alpha[..., None] * seg
            dist = np.linalg.norm(pts - proj, axis=-1).astype(np.float32, copy=False)
            improve = dist < best_dist
            best_dist = np.where(improve, dist, best_dist)
            best_tangent = np.where(improve[..., None], tangents_xy[seg_idx].reshape(1, 1, 2), best_tangent)
        return best_dist, best_tangent

    def _batch_map_metrics(
        self,
        *,
        sample_context: NuScenesPDMSampleContext,
        candidate_geometry: dict[str, np.ndarray],
        dt_s: float,
    ) -> dict[str, np.ndarray]:
        del dt_s
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

        oncoming_progress_m = np.zeros((num_candidates,), dtype=np.float32)
        continuous_reverse_time_s = np.zeros((num_candidates,), dtype=np.float32)
        for step_idx in range(horizon):
            reverse_mask = reverse_alignment[:, step_idx] > _DEFAULT_DIRECTION_REVERSE_ALIGNMENT_THRESHOLD
            continuous_reverse_time_s = np.where(
                reverse_mask,
                continuous_reverse_time_s + 0.5,
                0.0,
            ).astype(np.float32, copy=False)
            accrue_mask = continuous_reverse_time_s > _DEFAULT_DIRECTION_MIN_CONTINUOUS_REVERSE_S
            oncoming_progress_m = np.where(
                accrue_mask,
                oncoming_progress_m + step_norm[:, step_idx] * reverse_alignment[:, step_idx],
                oncoming_progress_m,
            ).astype(np.float32, copy=False)

        driving_direction = np.asarray(
            [
                _linear_decay_score(
                    float(progress_m),
                    good_threshold=_DEFAULT_DIRECTION_COMPLIANCE_THRESHOLD_M,
                    bad_threshold=_DEFAULT_DIRECTION_VIOLATION_THRESHOLD_M,
                )
                for progress_m in oncoming_progress_m
            ],
            dtype=np.float32,
        )
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
        collision_mask = np.zeros((num_candidates,), dtype=bool)
        earliest_ttc_risk_s = np.full((num_candidates,), np.inf, dtype=np.float32)

        for step_idx in range(horizon):
            hits = self._query_hits_per_candidate(sample_context.occupancy_map, polygons[:, step_idx], predicate="intersects")
            collision_mask |= hits

            moving_mask = speed_mps[:, step_idx] >= float(_DEFAULT_TTC_STOPPED_SPEED_THRESHOLD_MPS)
            if not bool(np.any(moving_mask)):
                continue

            for offset_idx, future_offset_s in enumerate(ttc_projection["offsets_s"]):
                proj_polygons = np.asarray(ttc_projection["polygons"][:, step_idx, offset_idx], dtype=object)
                hits = self._query_hits_per_candidate(sample_context.occupancy_map, proj_polygons, predicate="intersects")
                risk_mask = np.logical_and(hits, moving_mask)
                earliest_ttc_risk_s = np.where(
                    np.logical_and(risk_mask, float(future_offset_s) < earliest_ttc_risk_s),
                    float(future_offset_s),
                    earliest_ttc_risk_s,
                )

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

        cand_progress = np.asarray(
            [_project_progress(centers_xy[idx, -1], gt_xy_full, gt_s_full) for idx in range(num_candidates)],
            dtype=np.float32,
        )
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
        driving_direction = map_metrics["driving_direction"]
        drivable_area = map_metrics["drivable_area"] if drivable_polygons else np.ones((num_candidates,), dtype=np.float32)

        collision_ttc = self._batch_collision_ttc_metrics(
            sample_context=sample_context,
            candidate_geometry=candidate_geometry,
            dt_s=float(dt_s),
        )
        no_collision = collision_ttc["no_collision"]
        ttc = collision_ttc["ttc"]

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
