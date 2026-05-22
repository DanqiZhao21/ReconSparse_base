from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from framework.algorithms.craft_reward import (
    CRAFT_CARL_FORWARD_SIM_DEFAULTS,
    compute_carl_reward_numpy,
)
from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer
from framework.algorithms.nuscenes_scorer_utils import _DEFAULT_PATCH_RADIUS_M, _polyline_arclength, _wrap_angle


class NuScenesCraftScorer:
    """NuScenes counterfactual scorer using CRAFT CaRL forward-sim reward."""

    def __init__(
        self,
        *,
        token2vad_path: str | Path,
        carl: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pdm_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "dac_weight",
                "ttc_weight",
                "history_comfort_weight",
                "lane_keeping_weight",
                "progress_weight",
                "score_mode",
                "center_dev_max_m",
                "heading_dev_max_deg",
                "off_global_route_threshold_m",
            }
        }
        self._pdm = NuScenesPDMScorer(token2vad_path=token2vad_path, **pdm_kwargs)
        self.carl_params = {**CRAFT_CARL_FORWARD_SIM_DEFAULTS, **dict(carl or {})}
        self.center_dev_max_m = float(kwargs.get("center_dev_max_m", 2.0))
        self.heading_dev_max_deg = float(kwargs.get("heading_dev_max_deg", 90.0))
        self.off_global_route_threshold_m = float(kwargs.get("off_global_route_threshold_m", 3.0))

    def _project_route_stats_all(
        self,
        points_xy: np.ndarray,
        yaw_rad: np.ndarray,
        path_xy: np.ndarray,
        path_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pts = np.asarray(points_xy, dtype=np.float32)
        yaw = np.asarray(yaw_rad, dtype=np.float32)
        path = np.asarray(path_xy, dtype=np.float32)
        path_s_arr = np.asarray(path_s, dtype=np.float32)
        if pts.ndim != 3 or pts.shape[-1] != 2:
            raise RuntimeError(f"Expected points_xy shape (candidates,horizon,2), got {tuple(pts.shape)}")
        if yaw.shape != pts.shape[:2]:
            raise RuntimeError(f"Expected yaw_rad shape {tuple(pts.shape[:2])}, got {tuple(yaw.shape)}")
        num_candidates, horizon = pts.shape[:2]
        if int(path.shape[0]) <= 1 or int(path_s_arr.shape[0]) <= 1:
            return (
                np.zeros((num_candidates, horizon), dtype=np.float32),
                np.ones((num_candidates, horizon), dtype=np.float32),
                np.ones((num_candidates, horizon), dtype=np.float32),
                np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (num_candidates, horizon, 1)),
            )

        seg_start = path[:-1]
        seg_end = path[1:]
        seg_vec = (seg_end - seg_start).astype(np.float32, copy=False)
        seg_len_sq = np.sum(seg_vec * seg_vec, axis=-1).astype(np.float32, copy=False)
        valid_seg = seg_len_sq > 1.0e-12
        if not bool(np.any(valid_seg)):
            return (
                np.zeros((num_candidates, horizon), dtype=np.float32),
                np.ones((num_candidates, horizon), dtype=np.float32),
                np.ones((num_candidates, horizon), dtype=np.float32),
                np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (num_candidates, horizon, 1)),
            )

        safe_len_sq = np.where(valid_seg, seg_len_sq, 1.0).astype(np.float32, copy=False)
        delta = pts[:, :, None, :] - seg_start[None, None, :, :]
        alpha = np.sum(delta * seg_vec[None, None, :, :], axis=-1) / safe_len_sq[None, None, :]
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)
        proj = seg_start[None, None, :, :] + alpha[..., None] * seg_vec[None, None, :, :]
        dist = np.linalg.norm(pts[:, :, None, :] - proj, axis=-1).astype(np.float32, copy=False)
        dist = np.where(valid_seg[None, None, :], dist, np.inf).astype(np.float32, copy=False)
        best_idx = np.argmin(dist, axis=-1)
        best_dist = np.take_along_axis(dist, best_idx[..., None], axis=-1).squeeze(-1).astype(np.float32, copy=False)

        seg_len = np.sqrt(np.where(valid_seg, seg_len_sq, 1.0)).astype(np.float32, copy=False)
        progress_all = path_s_arr[:-1][None, None, :] + alpha * seg_len[None, None, :]
        progress_s = np.take_along_axis(progress_all, best_idx[..., None], axis=-1).squeeze(-1)
        tangents = np.where(
            seg_len[:, None] > 1.0e-6,
            seg_vec / np.maximum(seg_len[:, None], 1.0e-6),
            np.asarray([1.0, 0.0], dtype=np.float32).reshape(1, 2),
        ).astype(np.float32, copy=False)
        best_tangent = tangents[np.clip(best_idx, 0, max(0, int(tangents.shape[0]) - 1))]
        finite_mask = np.isfinite(best_dist)
        progress_s = np.where(finite_mask, progress_s, 0.0).astype(np.float32, copy=False)
        route_lateral = np.where(finite_mask, best_dist, 0.0).astype(np.float32, copy=False)
        best_tangent = np.where(
            finite_mask[..., None],
            best_tangent,
            np.asarray([1.0, 0.0], dtype=np.float32).reshape(1, 1, 2),
        ).astype(np.float32, copy=False)
        route_yaw = np.arctan2(best_tangent[..., 1], best_tangent[..., 0]).astype(np.float32, copy=False)
        route_heading_ratio = (
            np.abs(_wrap_angle(yaw - route_yaw)) / max(1.0e-6, np.deg2rad(self.heading_dev_max_deg))
        ).astype(np.float32, copy=False)
        return progress_s, route_lateral, route_heading_ratio, best_tangent

    def _score_candidate_batch_for_sample(
        self,
        *,
        sample_context: Any,
        candidate_geometry: dict[str, np.ndarray],
        gt_xy_full: np.ndarray,
        gt_s_full: np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        centers_xy = np.asarray(candidate_geometry["centers_xy"], dtype=np.float32)
        yaw_rad = np.asarray(candidate_geometry["yaw_rad"], dtype=np.float32)
        corners_xy = np.asarray(candidate_geometry["corners_xy"], dtype=np.float32)
        num_candidates, horizon = centers_xy.shape[:2]
        if num_candidates <= 0 or horizon <= 0:
            return np.zeros((num_candidates,), dtype=np.float32)

        gt_xy_full = np.asarray(gt_xy_full, dtype=np.float32)
        gt_s_full = np.asarray(gt_s_full, dtype=np.float32)
        if gt_xy_full.size <= 0 or int(gt_xy_full.shape[0]) <= 1:
            raise RuntimeError("NuScenesCraftScorer requires gt_xy_full with at least two route points")
        if int(gt_s_full.shape[0]) != int(gt_xy_full.shape[0]):
            gt_s_full = _polyline_arclength(gt_xy_full).astype(np.float32, copy=False)

        progress_s, route_lateral, route_heading_ratio, _ = self._project_route_stats_all(
            centers_xy,
            yaw_rad,
            gt_xy_full,
            gt_s_full,
        )
        prev_progress = np.concatenate([np.zeros((num_candidates, 1), dtype=np.float32), progress_s[:, :-1]], axis=1)
        delta_progress = (progress_s - prev_progress).astype(np.float32, copy=False)
        global_dev_ratio = np.clip(route_lateral / max(1.0e-6, self.center_dev_max_m), 0.0, 1.0).astype(np.float32, copy=False)

        prev_xy = np.concatenate([centers_xy[:, :1, :], centers_xy[:, :-1, :]], axis=1)
        step_delta = centers_xy - prev_xy
        center_lateral, tangents_xy = self._pdm._batch_centerline_stats(
            centers_xy,
            sample_context.centerline_segments_xy,
            sample_context.centerline_tangents_xy,
            direction_xy=step_delta,
        )
        has_centerline = np.asarray(sample_context.centerline_segments_xy).size > 0
        center_dev_ratio = np.clip(center_lateral / max(1.0e-6, self.center_dev_max_m), 0.0, 1.0).astype(np.float32, copy=False)
        tangent_yaw = np.arctan2(tangents_xy[..., 1], tangents_xy[..., 0]).astype(np.float32, copy=False)
        map_heading_ratio = np.abs(_wrap_angle(yaw_rad - tangent_yaw)) / max(1.0e-6, np.deg2rad(self.heading_dev_max_deg))
        heading_source_ratio = np.maximum(route_heading_ratio, map_heading_ratio) if has_centerline else route_heading_ratio
        heading_dev_ratio = np.clip(heading_source_ratio, 0.0, 1.0).astype(np.float32, copy=False)

        delta_global = np.concatenate([np.zeros((num_candidates, 1), dtype=np.float32), global_dev_ratio[:, 1:] - global_dev_ratio[:, :-1]], axis=1)
        delta_center = np.concatenate([np.zeros((num_candidates, 1), dtype=np.float32), center_dev_ratio[:, 1:] - center_dev_ratio[:, :-1]], axis=1)
        delta_heading = np.concatenate([np.zeros((num_candidates, 1), dtype=np.float32), heading_dev_ratio[:, 1:] - heading_dev_ratio[:, :-1]], axis=1)

        inside = sample_context.drivable_map.batch_contains_points(corners_xy)
        off_road = (~inside.all(axis=2)).astype(np.float32, copy=False)
        off_global_route = (route_lateral >= self.off_global_route_threshold_m).astype(np.float32, copy=False)

        step_norm = np.linalg.norm(step_delta, axis=-1).astype(np.float32, copy=False)
        step_dir = np.where(step_norm[..., None] > 1.0e-6, step_delta / np.maximum(step_norm[..., None], 1.0e-6), 0.0)
        reverse_alignment = np.maximum(0.0, -np.sum(step_dir * tangents_xy, axis=-1)).astype(np.float32, copy=False)
        opposite_lane = (reverse_alignment > 0.5).astype(np.float32, copy=False)
        emergency_lane = np.zeros((num_candidates, horizon), dtype=np.float32)

        collision_ttc = self._pdm._batch_collision_ttc_metrics(
            sample_context=sample_context,
            candidate_geometry=candidate_geometry,
            dt_s=float(dt_s),
        )
        if "collision_matrix" in collision_ttc:
            collision = np.asarray(collision_ttc["collision_matrix"], dtype=np.float32)
            if collision.shape != (num_candidates, horizon):
                raise RuntimeError(
                    f"Expected collision_matrix shape {(num_candidates, horizon)}, got {tuple(collision.shape)}"
                )
        else:
            no_collision = np.asarray(collision_ttc.get("no_collision", np.ones((num_candidates,), dtype=np.float32)), dtype=np.float32)
            collision = np.repeat((no_collision <= 0.0).astype(np.float32).reshape(num_candidates, 1), horizon, axis=1)

        route_deviation = np.any(off_global_route > 0.0, axis=1)
        result = compute_carl_reward_numpy(
            params=self.carl_params,
            delta_progress=delta_progress,
            global_dev_ratio=global_dev_ratio,
            center_dev_ratio=center_dev_ratio,
            heading_dev_ratio=heading_dev_ratio,
            delta_global_dev_ratio=delta_global,
            delta_center_dev_ratio=delta_center,
            delta_heading_dev_ratio=delta_heading,
            off_road=off_road,
            opposite_lane=opposite_lane,
            emergency_lane=emergency_lane,
            off_global_route=off_global_route,
            collision=collision,
            red_light_dev=np.zeros((num_candidates,), dtype=bool),
            stop_sign_dev=np.zeros((num_candidates,), dtype=bool),
            route_deviation=route_deviation,
        )
        self._last_terms = result.terms
        return result.candidate_scores.astype(np.float32, copy=False)

    def score(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> np.ndarray:
        if traj_xyyaw.ndim != 4 or int(traj_xyyaw.shape[-1]) < 3:
            raise RuntimeError(
                "NuScenesCraftScorer expects traj_xyyaw with shape (batch,candidates,horizon,3); "
                f"got {tuple(traj_xyyaw.shape)}"
            )
        if len(replays) != int(traj_xyyaw.shape[0]):
            raise RuntimeError(f"Replay batch length mismatch: replays={len(replays)} traj_batch={int(traj_xyyaw.shape[0])}")

        geometry_batch = self._pdm._build_candidate_geometry_batch(traj_xyyaw)
        scores = np.zeros((int(traj_xyyaw.shape[0]), int(traj_xyyaw.shape[1])), dtype=np.float32)
        for batch_idx, replay in enumerate(replays):
            sample_context = self._pdm._build_sample_context(replay, patch_radius=_DEFAULT_PATCH_RADIUS_M)
            static_ctx = dict(sample_context.static_context)
            gt_xy = np.asarray(static_ctx.get("gt_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            gt_s = np.asarray(static_ctx.get("gt_s", _polyline_arclength(gt_xy)), dtype=np.float32)
            horizon = min(int(traj_xyyaw.shape[2]), int(gt_xy.shape[0])) if gt_xy.size else int(traj_xyyaw.shape[2])
            candidate_geometry = {
                key: value[batch_idx, :, :horizon].copy()
                for key, value in geometry_batch.items()
            }
            sample_scores = self._score_candidate_batch_for_sample(
                sample_context=sample_context,
                candidate_geometry=candidate_geometry,
                gt_xy_full=gt_xy,
                gt_s_full=gt_s,
                dt_s=0.5,
            )
            scores[batch_idx, : int(sample_scores.shape[0])] = sample_scores.astype(np.float32, copy=False)
        return scores

    def dump_debug_artifacts(self, *args: Any, **kwargs: Any) -> None:
        return self._pdm.dump_debug_artifacts(*args, **kwargs)


__all__ = ["NuScenesCraftScorer"]
