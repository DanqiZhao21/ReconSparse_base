from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    from reconsimulator.envs import nus_config as nus_cfg
except Exception:
    nus_cfg = None

try:
    from shapely import affinity as shapely_affinity
    from shapely.geometry import Polygon as ShapelyPolygon
except Exception:
    shapely_affinity = None
    ShapelyPolygon = None


_DEFAULT_NUSCENES_VERSION = str(getattr(nus_cfg, "NUSCENES_VERSION", "v1.0-trainval"))
_DEFAULT_NUSCENES_ROOTS = tuple(
    str(path)
    for path in (
        os.environ.get("NUSCENES_DATAROOT", ""),
        getattr(nus_cfg, "NUSCENES_DATA_ROOT", ""),
        "/OpenDataset/nuscenes/nuscenes",
        "/OpenDataset/nuscenes",
        "/OpenDataset/nuscenes/nuscenes/v1.0-trainval",
    )
    if str(path).strip() != ""
)
_MAP_POLYGON_LAYERS = (
    "drivable_area",
    "road_segment",
    "road_block",
    "lane",
    "lane_connector",
    "ped_crossing",
    "walkway",
)
_MAP_LINE_LAYERS = (
    "lane_divider",
    "road_divider",
)
_DEFAULT_PATCH_RADIUS_M = 20.0


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))

#从轨迹点 (x, y) 计算每一帧的朝向 yaw
#TODO: 第一个点prev只是一个占位；
def _path_yaw_from_xy(points_xy: np.ndarray) -> np.ndarray:
    if int(points_xy.shape[0]) <= 0:
        return np.zeros((0,), dtype=np.float32)
    prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), points_xy[:-1]], axis=0)
    delta = points_xy - prev
    yaw = np.arctan2(delta[:, 1], delta[:, 0]).astype(np.float32)
    if int(yaw.shape[0]) > 1:
        yaw[0] = yaw[1]
    return yaw

#计算轨迹的累计路径长度（s）;;把一串轨迹点，变成“沿路径的累计距离（arclength）”
def _polyline_arclength(points_xy: np.ndarray) -> np.ndarray:
    if int(points_xy.shape[0]) <= 0:
        return np.zeros((0,), dtype=np.float32)  #np.zeros(shape, dtype=...)   array([], dtype=float32) ; NumPy 的 shape 必须是“元组（tuple）”，而 (0) 不是元组
    if int(points_xy.shape[0]) == 1:
        return np.zeros((1,), dtype=np.float32)
    seg = np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg, dtype=np.float32)], axis=0)

#_project_progress(cand_xy[-1], gt_xy, gt_s)
def _project_progress(point_xy: np.ndarray, path_xy: np.ndarray, path_s: np.ndarray) -> float:
    if int(path_xy.shape[0]) <= 1:
        return 0.0
    best_dist = float("inf")
    best_s = 0.0
    for idx in range(int(path_xy.shape[0]) - 1):
        p0 = path_xy[idx]
        p1 = path_xy[idx + 1]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq <= 1e-12:
            continue
        alpha = float(np.dot(point_xy - p0, seg) / seg_len_sq)
        alpha = max(0.0, min(1.0, alpha))
        proj = p0 + alpha * seg
        dist = float(np.linalg.norm(point_xy - proj))
        if dist < best_dist:
            best_dist = dist
            best_s = float(path_s[idx] + alpha * math.sqrt(seg_len_sq))
    return best_s


class NuScenesTokenScorer:
    def __init__(
        self,
        *,
        token2vad_path: str | Path,
        nuscenes_dataroot: str | Path | None = None,
        nuscenes_version: str = _DEFAULT_NUSCENES_VERSION,
    ) -> None:
        self.token2vad_path = Path(token2vad_path)
        self._token2vad: dict[str, dict[str, Any]] | None = None
        self.nuscenes_version = str(nuscenes_version)
        self.nuscenes_dataroot = self._resolve_nuscenes_dataroot(nuscenes_dataroot)
        self._nusc: Any | None = None
        self._map_cache: dict[str, Any] = {}

    @staticmethod
    def _resolve_nuscenes_dataroot(explicit_root: str | Path | None) -> Path | None:
        candidates: list[Path] = []
        if explicit_root is not None and str(explicit_root).strip() != "":
            candidates.append(Path(str(explicit_root)))
        for item in _DEFAULT_NUSCENES_ROOTS:
            candidates.append(Path(item))
        for candidate in candidates:
            if not candidate.exists():
                continue
            if (candidate / "maps").exists():
                return candidate
            version_dir = candidate / _DEFAULT_NUSCENES_VERSION
            if version_dir.exists() and (version_dir / "maps").exists():
                return version_dir
        return None

    def _ensure_loaded(self) -> dict[str, dict[str, Any]]:
        if self._token2vad is None:
            with self.token2vad_path.open("rb") as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, dict):
                raise RuntimeError(f"token2vad file must contain a dict, got {type(loaded)!r}")
            self._token2vad = loaded
        return self._token2vad

    @staticmethod
    def _gt_to_env_xy(gt_ego_fut_trajs: np.ndarray) -> np.ndarray:
        gt = np.asarray(gt_ego_fut_trajs, dtype=np.float32)
        if gt.ndim != 2 or gt.shape[1] < 2:
            raise RuntimeError(f"Expected gt_ego_fut_trajs with shape (T, 2+), got {gt.shape}")
        # token2vad stores local future as (lateral, forward); simulator/policy uses (forward, left).
        out = np.zeros((gt.shape[0], 2), dtype=np.float32)
        out[:, 0] = gt[:, 1]
        out[:, 1] = gt[:, 0]
        return out

    def _lookup_row(self, sample_token: str) -> dict[str, Any]:
        token2vad = self._ensure_loaded()
        row = token2vad.get(str(sample_token), None)
        if not isinstance(row, dict):
            raise RuntimeError(f"sample_token={sample_token!r} not found in token2vad index")
        return row

    def _lookup_gt(self, sample_token: str) -> np.ndarray:
        row = self._lookup_row(sample_token)
        gt = row.get("gt_ego_fut_trajs", None)
        if gt is None:
            raise RuntimeError(f"sample_token={sample_token!r} missing gt_ego_fut_trajs")
        return self._gt_to_env_xy(np.asarray(gt, dtype=np.float32))

    @staticmethod
    def _rebase_xy(points_xy: np.ndarray, origin_xy: np.ndarray | None) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float32)
        if origin_xy is None:
            return pts.copy()
        origin = np.asarray(origin_xy, dtype=np.float32).reshape(1, 2)
        return pts - origin

    @staticmethod
    def _quat_to_rotmat_xy(rotation_wxyz: Sequence[float]) -> np.ndarray:
        quat = np.asarray(list(rotation_wxyz), dtype=np.float32).reshape(-1)
        if quat.shape[0] < 4:
            return np.eye(2, dtype=np.float32)
        w, x, y, z = [float(val) for val in quat[:4]]
        rot = np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z)],
            ],
            dtype=np.float32,
        )
        return rot

    def _global_to_local_xy(self, points_xy: np.ndarray, row: dict[str, Any]) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float32)
        translation = np.asarray(row.get("ego2global_translation", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        rot = self._quat_to_rotmat_xy(row.get("ego2global_rotation", [1.0, 0.0, 0.0, 0.0]))
        delta = pts[:, :2] - translation[:2].reshape(1, 2)
        return delta @ rot

    def _lookup_gt_history(self, row: dict[str, Any], *, origin_xy: np.ndarray | None) -> np.ndarray:
        history = row.get("gt_ego_his_trajs", None)
        if history is None:
            return np.zeros((0, 2), dtype=np.float32)
        history_xy = self._gt_to_env_xy(np.asarray(history, dtype=np.float32))
        return self._rebase_xy(history_xy, origin_xy)

    @staticmethod
    def _box_corners_xy(center_x: float, center_y: float, length: float, width: float, yaw: float) -> np.ndarray:
        dx = float(length) * 0.5
        dy = float(width) * 0.5
        corners = np.asarray(
            [
                [dx, dy],
                [dx, -dy],
                [-dx, -dy],
                [-dx, dy],
            ],
            dtype=np.float32,
        )
        c = math.cos(float(yaw))
        s = math.sin(float(yaw))
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        return corners @ rot.T + np.asarray([[float(center_x), float(center_y)]], dtype=np.float32)

    def _extract_scene_objects(self, row: dict[str, Any], *, patch_radius: float) -> list[dict[str, Any]]:
        boxes = np.asarray(row.get("gt_boxes", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[0] <= 0:
            return []
        names = np.asarray(row.get("gt_names", np.asarray([], dtype=object)))
        velocities = np.asarray(row.get("gt_velocity", np.zeros((boxes.shape[0], 2), dtype=np.float32)), dtype=np.float32)
        valid = np.asarray(row.get("valid_flag", np.ones((boxes.shape[0],), dtype=bool)))
        lidar_pts = np.asarray(row.get("num_lidar_pts", np.zeros((boxes.shape[0],), dtype=np.int64)))
        radar_pts = np.asarray(row.get("num_radar_pts", np.zeros((boxes.shape[0],), dtype=np.int64)))
        objects: list[dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            if idx < valid.shape[0] and not bool(valid[idx]):
                continue
            x = float(box[0])
            y = float(box[1])
            if max(abs(x), abs(y)) > float(patch_radius) * 1.2 and math.hypot(x, y) > float(patch_radius) * 1.4:
                continue
            width = float(abs(box[3])) if box.shape[0] > 3 else 1.0
            length = float(abs(box[4])) if box.shape[0] > 4 else 1.0
            yaw = float(box[6]) if box.shape[0] > 6 else 0.0
            speed = float(np.linalg.norm(velocities[idx])) if idx < velocities.shape[0] else 0.0
            category = str(names[idx]) if idx < names.shape[0] else "unknown"
            corners = self._box_corners_xy(x, y, length, width, yaw)
            objects.append(
                {
                    "category": category,
                    "center_xy": [x, y],
                    "length_m": length,
                    "width_m": width,
                    "yaw_rad": yaw,
                    "speed_mps": speed,
                    "num_lidar_pts": int(lidar_pts[idx]) if idx < lidar_pts.shape[0] else 0,
                    "num_radar_pts": int(radar_pts[idx]) if idx < radar_pts.shape[0] else 0,
                    "corners_xy": corners.astype(np.float32).tolist(),
                }
            )
        return objects

    def _ensure_nusc(self) -> Any | None:
        if self._nusc is not None:
            return self._nusc
        if self.nuscenes_dataroot is None:
            return None
        try:
            from nuscenes.nuscenes import NuScenes

            self._nusc = NuScenes(version=self.nuscenes_version, dataroot=str(self.nuscenes_dataroot), verbose=False)
        except Exception:
            self._nusc = None
        return self._nusc

    def _lookup_map_layers(self, row: dict[str, Any], *, patch_radius: float = _DEFAULT_PATCH_RADIUS_M) -> dict[str, Any]:
        nusc = self._ensure_nusc()
        if nusc is None:
            return {"patch_radius": float(patch_radius), "layers": {}}
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
        except Exception:
            return {"patch_radius": float(patch_radius), "layers": {}}

        location = row.get("map_location", None)
        if location is None:
            sample_token = str(row.get("token", ""))
            if sample_token == "":
                return {"patch_radius": float(patch_radius), "layers": {}}
            sample_record = nusc.get("sample", sample_token)
            scene_record = nusc.get("scene", sample_record["scene_token"])
            log_record = nusc.get("log", scene_record["log_token"])
            location = log_record["location"]
        location = str(location)
        if location not in self._map_cache:
            try:
                self._map_cache[location] = NuScenesMap(dataroot=str(self.nuscenes_dataroot), map_name=location)
            except Exception:
                self._map_cache[location] = None
        map_api = self._map_cache.get(location, None)
        if map_api is None:
            return {"patch_radius": float(patch_radius), "layers": {}}

        ego_translation = np.asarray(row.get("ego2global_translation", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        x = float(ego_translation[0]) if ego_translation.shape[0] > 0 else 0.0
        y = float(ego_translation[1]) if ego_translation.shape[0] > 1 else 0.0
        layers: dict[str, list[list[list[float]]]] = {}
        try:
            records = map_api.get_records_in_radius(
                x,
                y,
                float(patch_radius),
                list(_MAP_POLYGON_LAYERS) + list(_MAP_LINE_LAYERS),
                mode="intersect",
            )
        except Exception:
            return {"patch_radius": float(patch_radius), "layers": {}}

        for layer in _MAP_POLYGON_LAYERS:
            geometries: list[list[list[float]]] = []
            for token in records.get(layer, []):
                try:
                    record = map_api.get(layer, token)
                    polygon_tokens = record.get("polygon_tokens", None) if layer == "drivable_area" else [record.get("polygon_token", None)]
                    for polygon_token in polygon_tokens or []:
                        if polygon_token is None:
                            continue
                        polygon = map_api.extract_polygon(polygon_token)
                        if polygon is None:
                            continue
                        coords = np.asarray(polygon.exterior.coords, dtype=np.float32)
                        local = self._global_to_local_xy(coords[:, :2], row)
                        if local.shape[0] >= 3:
                            geometries.append(local.astype(np.float32).tolist())
                except Exception:
                    continue
            if len(geometries) > 0:
                layers[layer] = geometries

        for layer in _MAP_LINE_LAYERS:
            geometries = []
            for token in records.get(layer, []):
                try:
                    record = map_api.get(layer, token)
                    line_token = record.get("line_token", None)
                    if line_token is None:
                        continue
                    line = map_api.extract_line(line_token)
                    if line is None:
                        continue
                    coords = np.asarray(line.coords, dtype=np.float32)
                    local = self._global_to_local_xy(coords[:, :2], row)
                    if local.shape[0] >= 2:
                        geometries.append(local.astype(np.float32).tolist())
                except Exception:
                    continue
            if len(geometries) > 0:
                layers[layer] = geometries

        centerlines: list[list[list[float]]] = []
        lane_tokens = list(records.get("lane", [])) + list(records.get("lane_connector", []))
        if len(lane_tokens) > 0:
            try:
                raw_centerlines = map_api.discretize_lanes(lane_tokens, 1.0)
                if isinstance(raw_centerlines, dict):
                    iter_centerlines = raw_centerlines.values()
                else:
                    iter_centerlines = raw_centerlines
                for centerline in iter_centerlines:
                    coords = np.asarray(centerline, dtype=np.float32)
                    if coords.ndim == 2 and coords.shape[0] >= 2:
                        local = self._global_to_local_xy(coords[:, :2], row)
                        centerlines.append(local.astype(np.float32).tolist())
            except Exception:
                pass
        if len(centerlines) > 0:
            layers["lane_centerline"] = centerlines

        return {"patch_radius": float(patch_radius), "layers": layers}

    @staticmethod
    def _category_color(category: str) -> str:
        key = str(category).strip().lower()
        if "pedestrian" in key:
            return "#8e44ad"
        if "bicycle" in key or "motorcycle" in key:
            return "#27ae60"
        if "bus" in key or "truck" in key or "trailer" in key:
            return "#d35400"
        return "#e74c3c"

    @staticmethod
    def _compute_plot_limits(
        *,
        gt_xy: np.ndarray,
        gt_history_xy: np.ndarray,
        candidates: Sequence[dict[str, Any]],
        scene_objects: Sequence[dict[str, Any]],
        patch_radius: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        del gt_xy, gt_history_xy, candidates, scene_objects
        return (-float(patch_radius), float(patch_radius)), (-float(patch_radius), float(patch_radius))

    @staticmethod
    def _candidate_line_style(*, rank: int, total: int) -> dict[str, float]:
        total_count = max(1, int(total))
        rank_idx = max(0, int(rank))
        if rank_idx <= 0:
            alpha = 0.62
            linewidth = 2.3
            marker_size = 26.0
        elif rank_idx == 1:
            alpha = 0.54
            linewidth = 2.1
            marker_size = 23.0
        elif rank_idx == 2:
            alpha = 0.46
            linewidth = 1.9
            marker_size = 20.0
        else:
            tail_total = max(1, total_count - 4)
            tail_rank = min(tail_total, rank_idx - 3)
            tail_fade = float(tail_rank) / float(tail_total)
            alpha = max(0.14, 0.30 - 0.12 * tail_fade)
            linewidth = max(1.05, 1.55 - 0.35 * tail_fade)
            marker_size = max(9.0, 15.0 - 4.5 * tail_fade)
        return {
            "alpha": float(alpha),
            "linewidth": float(linewidth),
            "marker_size": float(marker_size),
        }

    @staticmethod
    def _normalize_polygons(geometries: Sequence[Any]) -> list[list[list[float]]]:
        out: list[list[list[float]]] = []
        for geom in geometries or []:
            arr = np.asarray(geom, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 3:
                out.append(arr[:, :2].astype(np.float32).tolist())
        return out

    @staticmethod
    def _normalize_lines(geometries: Sequence[Any]) -> list[list[list[float]]]:
        out: list[list[list[float]]] = []
        for geom in geometries or []:
            arr = np.asarray(geom, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                out.append(arr[:, :2].astype(np.float32).tolist())
        return out

    @staticmethod
    def _polygon_to_edges(polygons: Sequence[Any]) -> list[list[list[float]]]:
        edges: list[list[list[float]]] = []
        for polygon in polygons or []:
            arr = np.asarray(polygon, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            pts = arr[:, :2]
            closed = np.concatenate([pts, pts[:1]], axis=0)
            edges.append(closed.astype(np.float32).tolist())
        return edges

    @staticmethod
    def _centerline_arrows(lines: Sequence[Any], *, arrow_len: float = 1.8) -> list[list[list[float]]]:
        arrows: list[list[list[float]]] = []
        for line in lines or []:
            pts = np.asarray(line, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            idx = max(0, (pts.shape[0] // 2) - 1)
            start = pts[idx]
            end = pts[idx + 1]
            vec = end - start
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-6:
                continue
            direction = vec / norm
            center = 0.5 * (start + end)
            arrow_start = center - direction * (float(arrow_len) * 0.5)
            arrow_end = center + direction * (float(arrow_len) * 0.5)
            arrows.append([arrow_start.astype(np.float32).tolist(), arrow_end.astype(np.float32).tolist()])
        return arrows

    @classmethod
    def _crossing_stripes(cls, polygons: Sequence[Any]) -> list[list[list[float]]]:
        if ShapelyPolygon is None or shapely_affinity is None:
            return []
        stripes: list[list[list[float]]] = []
        for polygon in polygons or []:
            arr = np.asarray(polygon, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            try:
                poly = ShapelyPolygon(arr[:, :2])
            except Exception:
                continue
            if poly.is_empty or not poly.is_valid or float(poly.area) <= 1e-3:
                continue
            rect = poly.minimum_rotated_rectangle
            coords = np.asarray(rect.exterior.coords[:-1], dtype=np.float32)
            if coords.shape[0] != 4:
                continue
            edges = []
            for idx in range(4):
                p0 = coords[idx]
                p1 = coords[(idx + 1) % 4]
                vec = p1 - p0
                length = float(np.linalg.norm(vec))
                if length > 1e-4:
                    edges.append((length, vec / length))
            if len(edges) < 2:
                continue
            edges.sort(key=lambda item: item[0], reverse=True)
            long_vec = np.asarray(edges[0][1], dtype=np.float32)
            short_vec = np.asarray([-long_vec[1], long_vec[0]], dtype=np.float32)
            centroid = np.asarray(poly.centroid.coords[0], dtype=np.float32)
            centered = arr[:, :2] - centroid.reshape(1, 2)
            long_proj = centered @ long_vec
            short_proj = centered @ short_vec
            long_min = float(np.min(long_proj))
            long_max = float(np.max(long_proj))
            short_min = float(np.min(short_proj))
            short_max = float(np.max(short_proj))
            short_span = short_max - short_min
            if short_span <= 0.2:
                continue
            stripe_width = min(0.6, max(0.25, (long_max - long_min) / 12.0))
            stripe_gap = stripe_width * 0.75
            usable_half = max(0.1, (short_span * 0.5) - 0.08)
            cursor = long_min + stripe_gap * 0.5
            while cursor < long_max:
                center = centroid + long_vec * cursor
                rect = np.asarray(
                    [
                        center + long_vec * (stripe_width * 0.5) + short_vec * usable_half,
                        center + long_vec * (stripe_width * 0.5) - short_vec * usable_half,
                        center - long_vec * (stripe_width * 0.5) - short_vec * usable_half,
                        center - long_vec * (stripe_width * 0.5) + short_vec * usable_half,
                    ],
                    dtype=np.float32,
                )
                try:
                    stripe_poly = ShapelyPolygon(rect)
                    clipped = poly.intersection(stripe_poly)
                except Exception:
                    cursor += stripe_width + stripe_gap
                    continue
                if clipped.is_empty:
                    cursor += stripe_width + stripe_gap
                    continue
                geoms = [clipped] if clipped.geom_type == "Polygon" else list(getattr(clipped, "geoms", []))
                for geom in geoms:
                    try:
                        pts = np.asarray(geom.exterior.coords[:-1], dtype=np.float32)
                    except Exception:
                        continue
                    if pts.ndim == 2 and pts.shape[0] >= 3:
                        stripes.append(pts[:, :2].astype(np.float32).tolist())
                cursor += stripe_width + stripe_gap
        return stripes

    @classmethod
    def _build_render_layers(cls, map_layers: dict[str, Any]) -> dict[str, list[list[list[float]]]]:
        drivable_polygons = cls._normalize_polygons(map_layers.get("drivable_area", []))
        road_surface_polygons = cls._normalize_polygons(
            list(map_layers.get("road_segment", []))
            + list(map_layers.get("road_block", []))
            + list(map_layers.get("lane", []))
            + list(map_layers.get("lane_connector", []))
        )
        if len(road_surface_polygons) == 0:
            road_surface_polygons = list(drivable_polygons)
        lane_fill_polygons = cls._normalize_polygons(
            list(map_layers.get("lane", [])) + list(map_layers.get("lane_connector", []))
        )
        walkway_polygons = cls._normalize_polygons(map_layers.get("walkway", []))
        crossing_polygons = cls._normalize_polygons(map_layers.get("ped_crossing", []))

        road_edge_lines = cls._polygon_to_edges(road_surface_polygons)
        if len(road_edge_lines) == 0:
            road_edge_lines = cls._polygon_to_edges(drivable_polygons)

        lane_marking_lines = cls._normalize_lines(
            list(map_layers.get("lane_divider", [])) + list(map_layers.get("road_divider", []))
        )
        if len(lane_marking_lines) == 0:
            lane_marking_lines = cls._polygon_to_edges(lane_fill_polygons)

        lane_centerlines = cls._normalize_lines(map_layers.get("lane_centerline", []))
        lane_boundary_lines = cls._polygon_to_edges(lane_fill_polygons)
        crossing_stripe_polygons = cls._crossing_stripes(crossing_polygons)

        return {
            "drivable_polygons": drivable_polygons,
            "road_surface_polygons": road_surface_polygons,
            "lane_fill_polygons": lane_fill_polygons,
            "walkway_polygons": walkway_polygons,
            "crossing_polygons": crossing_polygons,
            "crossing_stripe_polygons": crossing_stripe_polygons,
            "road_edge_lines": road_edge_lines,
            "lane_boundary_lines": lane_boundary_lines,
            "lane_marking_lines": lane_marking_lines,
            "lane_centerlines": lane_centerlines,
            "lane_direction_arrows": cls._centerline_arrows(lane_centerlines),
        }

    def _score_batch(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
        *,
        include_debug_context: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if traj_xyyaw.ndim != 4 or traj_xyyaw.shape[-1] < 2:
            raise RuntimeError(
                "NuScenesTokenScorer expects traj_xyyaw with shape (batch, candidates, horizon, 3); "
                f"got {tuple(traj_xyyaw.shape)}"
            )
        if len(replays) != int(traj_xyyaw.shape[0]):
            raise RuntimeError(f"Replay batch length mismatch: replays={len(replays)} traj_batch={int(traj_xyyaw.shape[0])}")

        traj_np = traj_xyyaw.detach().cpu().numpy().astype(np.float32)
        scores = np.zeros((traj_np.shape[0], traj_np.shape[1]), dtype=np.float32)
        details: list[dict[str, Any]] = []
        for batch_idx, replay in enumerate(replays):
            sample_token = replay.get("sample_token", None)
            if sample_token is None:
                raise RuntimeError("NuScenesTokenScorer requires replay['sample_token']")
            row = self._lookup_row(str(sample_token))
            gt_xy_raw = self._lookup_gt(str(sample_token))
            gt_origin_xy = gt_xy_raw[0].copy() if int(gt_xy_raw.shape[0]) > 0 else np.zeros((2,), dtype=np.float32)
            gt_xy = self._rebase_xy(gt_xy_raw, gt_origin_xy)
            gt_yaw = _path_yaw_from_xy(gt_xy)
            gt_s = _polyline_arclength(gt_xy)#给 GT 轨迹加一个“里程表”
            gt_total_len = float(max(1e-6, gt_s[-1] if int(gt_s.shape[0]) > 0 else 1.0))
            gt_history_xy = self._lookup_gt_history(row, origin_xy=gt_origin_xy) if include_debug_context else np.zeros((0, 2), dtype=np.float32)
            map_context = (
                self._lookup_map_layers(row, patch_radius=_DEFAULT_PATCH_RADIUS_M)
                if include_debug_context
                else {"patch_radius": float(_DEFAULT_PATCH_RADIUS_M), "layers": {}}
            )
            scene_objects = (
                self._extract_scene_objects(row, patch_radius=float(map_context.get("patch_radius", _DEFAULT_PATCH_RADIUS_M)))
                if include_debug_context
                else []
            )

            horizon = min(int(gt_xy.shape[0]), int(traj_np.shape[2]))
            gt_xy_cmp = gt_xy[:horizon]
            gt_yaw_cmp = gt_yaw[:horizon]
            sample_detail: dict[str, Any] = {
                "batch_index": int(batch_idx),
                "sample_token": str(sample_token),
                "gt_origin_shift_xy": gt_origin_xy.astype(np.float32).tolist(),
                "gt_xy": gt_xy_cmp.copy(),
                "gt_yaw": gt_yaw_cmp.copy(),
                "gt_history_xy": gt_history_xy.copy(),
                "scene_objects": scene_objects,
                "map_layers": dict(map_context.get("layers", {})),
                "map_patch_radius_m": float(map_context.get("patch_radius", _DEFAULT_PATCH_RADIUS_M)),
                "candidates": [],
            }
            for cand_idx in range(int(traj_np.shape[1])):
                cand = traj_np[batch_idx, cand_idx, :horizon, :]
                cand_xy = cand[:, :2]
                cand_yaw = cand[:, 2] if cand.shape[1] >= 3 else _path_yaw_from_xy(cand_xy)

                pos_err = np.linalg.norm(cand_xy - gt_xy_cmp, axis=1)
                first_err = float(pos_err[0]) if pos_err.size else 0.0
                final_err = float(pos_err[-1]) if pos_err.size else 0.0
                mean_err = float(pos_err.mean()) if pos_err.size else 0.0
                yaw_err = np.abs(_wrap_angle(cand_yaw - gt_yaw_cmp))
                mean_yaw_err = float(yaw_err.mean()) if yaw_err.size else 0.0

                cand_progress = _project_progress(cand_xy[-1], gt_xy, gt_s) if cand_xy.shape[0] > 0 else 0.0
                progress_ratio = float(np.clip(cand_progress / gt_total_len, 0.0, 1.5))

                cand_prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), cand_xy[:-1]], axis=0)
                cand_vel = cand_xy - cand_prev
                cand_acc = cand_vel[1:] - cand_vel[:-1] if cand_vel.shape[0] > 1 else np.zeros((0, 2), dtype=np.float32)
                smooth_pen = float(np.linalg.norm(cand_acc, axis=1).mean()) if cand_acc.size else 0.0

                score_terms = {
                    "progress_reward": 2.0 * progress_ratio,
                    "mean_error_penalty": -0.35 * mean_err,
                    "final_error_penalty": -0.50 * final_err,
                    "first_error_penalty": -0.35 * first_err,
                    "yaw_error_penalty": -0.20 * mean_yaw_err,
                    "smoothness_penalty": -0.05 * smooth_pen,
                }
                score = float(sum(score_terms.values()))
                scores[batch_idx, cand_idx] = np.float32(score)
                sample_detail["candidates"].append(
                    {
                        "candidate_index": int(cand_idx),
                        "traj_xyyaw": cand.copy(),
                        "score": score,
                        "progress_ratio": progress_ratio,
                        "mean_error_m": mean_err,
                        "final_error_m": final_err,
                        "first_error_m": first_err,
                        "mean_yaw_error_rad": mean_yaw_err,
                        "smoothness_penalty_raw": smooth_pen,
                        "score_terms": score_terms,
                    }
                )
            details.append(sample_detail)
        return scores, details

    @staticmethod
    def _path_yaw_from_xy_torch(points_xy: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if points_xy.ndim != 3:
            raise RuntimeError(f"Expected points_xy with shape (batch, horizon, 2), got {tuple(points_xy.shape)}")
        if valid_mask.shape != points_xy.shape[:2]:
            raise RuntimeError(
                "valid_mask must match points_xy batch/horizon dimensions: "
                f"mask={tuple(valid_mask.shape)} points={tuple(points_xy.shape)}"
            )
        if int(points_xy.shape[1]) <= 0:
            return torch.zeros(points_xy.shape[:2], device=points_xy.device, dtype=torch.float32)

        prev = torch.cat(
            [
                torch.zeros((points_xy.shape[0], 1, points_xy.shape[2]), device=points_xy.device, dtype=points_xy.dtype),
                points_xy[:, :-1, :],
            ],
            dim=1,
        )
        delta = points_xy - prev
        yaw = torch.atan2(delta[..., 1], delta[..., 0]).to(dtype=torch.float32)
        if int(points_xy.shape[1]) > 1:
            multi_step = valid_mask.sum(dim=1) > 1
            yaw[:, 0] = torch.where(multi_step, yaw[:, 1], yaw[:, 0])
        return torch.where(valid_mask, yaw, torch.zeros_like(yaw))

    @staticmethod
    def _polyline_arclength_torch(points_xy: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if points_xy.ndim != 3:
            raise RuntimeError(f"Expected points_xy with shape (batch, horizon, 2), got {tuple(points_xy.shape)}")
        if valid_mask.shape != points_xy.shape[:2]:
            raise RuntimeError(
                "valid_mask must match points_xy batch/horizon dimensions: "
                f"mask={tuple(valid_mask.shape)} points={tuple(points_xy.shape)}"
            )
        if int(points_xy.shape[1]) <= 0:
            return torch.zeros(points_xy.shape[:2], device=points_xy.device, dtype=torch.float32)

        if int(points_xy.shape[1]) == 1:
            return torch.zeros(points_xy.shape[:2], device=points_xy.device, dtype=torch.float32)

        seg = torch.linalg.norm(points_xy[:, 1:, :] - points_xy[:, :-1, :], dim=-1).to(dtype=torch.float32)
        seg_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        seg = torch.where(seg_valid, seg, torch.zeros_like(seg))
        prefix = torch.zeros((points_xy.shape[0], 1), device=points_xy.device, dtype=torch.float32)
        return torch.cat([prefix, torch.cumsum(seg, dim=1)], dim=1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
        mask_f = mask.to(device=values.device, dtype=values.dtype)
        denom = mask_f.sum(dim=dim).clamp_min(1.0)
        mean = (values * mask_f).sum(dim=dim) / denom
        empty = mask_f.sum(dim=dim) <= 0
        return torch.where(empty, torch.zeros_like(mean), mean)

    @staticmethod
    def _gather_last_valid(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise RuntimeError(f"Expected values with shape (batch, candidates, horizon), got {tuple(values.shape)}")
        if lengths.ndim != 1 or lengths.shape[0] != values.shape[0]:
            raise RuntimeError(
                "lengths must have shape (batch,) matching values batch dimension: "
                f"lengths={tuple(lengths.shape)} values={tuple(values.shape)}"
            )

        last_idx = lengths.clamp_min(1) - 1
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, values.shape[1], 1)
        gathered = torch.gather(values, dim=2, index=gather_idx).squeeze(2)
        return torch.where(
            lengths.view(-1, 1) > 0,
            gathered,
            torch.zeros_like(gathered),
        )

    @staticmethod
    def _gather_last_valid_xy(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise RuntimeError(f"Expected values with shape (batch, candidates, horizon, 2), got {tuple(values.shape)}")
        if lengths.ndim != 1 or lengths.shape[0] != values.shape[0]:
            raise RuntimeError(
                "lengths must have shape (batch,) matching values batch dimension: "
                f"lengths={tuple(lengths.shape)} values={tuple(values.shape)}"
            )

        last_idx = lengths.clamp_min(1) - 1
        gather_idx = last_idx.view(-1, 1, 1, 1).expand(-1, values.shape[1], 1, values.shape[3])
        gathered = torch.gather(values, dim=2, index=gather_idx).squeeze(2)
        return torch.where(
            (lengths.view(-1, 1, 1) > 0),
            gathered,
            torch.zeros_like(gathered),
        )

    def _prepare_torch_gt_batch(
        self,
        replays: Sequence[dict[str, Any]],
        *,
        traj_horizon: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        gt_arrays: list[np.ndarray] = []
        for replay in replays:
            sample_token = replay.get("sample_token", None)
            if sample_token is None:
                raise RuntimeError("NuScenesTokenScorer requires replay['sample_token']")
            gt_xy_raw = self._lookup_gt(str(sample_token))
            gt_origin_xy = gt_xy_raw[0].copy() if int(gt_xy_raw.shape[0]) > 0 else np.zeros((2,), dtype=np.float32)
            gt_arrays.append(self._rebase_xy(gt_xy_raw, gt_origin_xy))

        batch_size = len(gt_arrays)
        max_gt_horizon = max([1, *[int(arr.shape[0]) for arr in gt_arrays]])
        gt_full_xy = torch.zeros((batch_size, max_gt_horizon, 2), device=device, dtype=torch.float32)
        gt_full_mask = torch.zeros((batch_size, max_gt_horizon), device=device, dtype=torch.bool)
        gt_cmp_xy = torch.zeros((batch_size, traj_horizon, 2), device=device, dtype=torch.float32)

        cmp_lengths_list: list[int] = []
        full_lengths_list: list[int] = []
        for batch_idx, gt_xy in enumerate(gt_arrays):
            full_len = int(gt_xy.shape[0])
            cmp_len = min(full_len, int(traj_horizon))
            full_lengths_list.append(full_len)
            cmp_lengths_list.append(cmp_len)
            if full_len > 0:
                gt_tensor = torch.as_tensor(gt_xy, device=device, dtype=torch.float32)
                gt_full_xy[batch_idx, :full_len, :] = gt_tensor
                gt_full_mask[batch_idx, :full_len] = True
                if cmp_len > 0:
                    gt_cmp_xy[batch_idx, :cmp_len, :] = gt_tensor[:cmp_len]

        cmp_lengths = torch.as_tensor(cmp_lengths_list, device=device, dtype=torch.long)
        full_lengths = torch.as_tensor(full_lengths_list, device=device, dtype=torch.long)
        cmp_valid = torch.arange(traj_horizon, device=device).unsqueeze(0) < cmp_lengths.unsqueeze(1)
        gt_full_yaw = self._path_yaw_from_xy_torch(gt_full_xy, gt_full_mask)
        gt_full_s = self._polyline_arclength_torch(gt_full_xy, gt_full_mask)
        gt_cmp_yaw = torch.zeros((batch_size, traj_horizon), device=device, dtype=torch.float32)
        if traj_horizon > 0:
            copy_horizon = min(int(traj_horizon), int(gt_full_yaw.shape[1]))
            gt_cmp_yaw[:, :copy_horizon] = gt_full_yaw[:, :copy_horizon]
        last_full_idx = full_lengths.clamp_min(1) - 1
        gt_total_len = torch.gather(gt_full_s, dim=1, index=last_full_idx.view(-1, 1)).squeeze(1)
        gt_total_len = torch.where(full_lengths > 0, gt_total_len, torch.ones_like(gt_total_len)).clamp_min(1.0e-6)
        return {
            "gt_full_xy": gt_full_xy,
            "gt_full_mask": gt_full_mask,
            "gt_full_s": gt_full_s,
            "gt_cmp_xy": gt_cmp_xy,
            "gt_cmp_yaw": gt_cmp_yaw,
            "cmp_valid": cmp_valid,
            "cmp_lengths": cmp_lengths,
            "gt_total_len": gt_total_len,
        }

    @staticmethod
    def _project_progress_torch(
        point_xy: torch.Tensor,
        path_xy: torch.Tensor,
        path_s: torch.Tensor,
        path_mask: torch.Tensor,
    ) -> torch.Tensor:
        if point_xy.ndim != 3:
            raise RuntimeError(f"Expected point_xy with shape (batch, candidates, 2), got {tuple(point_xy.shape)}")
        if path_xy.ndim != 3 or path_s.ndim != 2 or path_mask.ndim != 2:
            raise RuntimeError(
                "Expected path tensors with shapes (batch, horizon, 2), (batch, horizon), (batch, horizon); "
                f"got {tuple(path_xy.shape)}, {tuple(path_s.shape)}, {tuple(path_mask.shape)}"
            )

        if int(path_xy.shape[1]) <= 1:
            return torch.zeros(point_xy.shape[:2], device=point_xy.device, dtype=torch.float32)

        p0 = path_xy[:, :-1, :]
        p1 = path_xy[:, 1:, :]
        seg = p1 - p0
        seg_valid = path_mask[:, :-1] & path_mask[:, 1:]
        seg_len_sq = (seg * seg).sum(dim=-1).to(dtype=torch.float32)
        safe_seg_len_sq = torch.where(seg_valid, seg_len_sq.clamp_min(1.0e-12), torch.ones_like(seg_len_sq))

        delta = point_xy.unsqueeze(2) - p0.unsqueeze(1)
        alpha = (delta * seg.unsqueeze(1)).sum(dim=-1) / safe_seg_len_sq.unsqueeze(1)
        alpha = alpha.clamp(0.0, 1.0)
        proj = p0.unsqueeze(1) + alpha.unsqueeze(-1) * seg.unsqueeze(1)
        dist = torch.linalg.norm(point_xy.unsqueeze(2) - proj, dim=-1).to(dtype=torch.float32)
        inf = torch.full_like(dist, float("inf"))
        dist = torch.where(seg_valid.unsqueeze(1), dist, inf)

        seg_len = torch.sqrt(safe_seg_len_sq)
        best_s_all = path_s[:, :-1].unsqueeze(1) + alpha * seg_len.unsqueeze(1)
        best_s_all = torch.where(seg_valid.unsqueeze(1), best_s_all, torch.zeros_like(best_s_all))
        best_idx = dist.argmin(dim=2)
        best_s = torch.gather(best_s_all, dim=2, index=best_idx.unsqueeze(-1)).squeeze(-1)
        has_valid_seg = seg_valid.any(dim=1, keepdim=True)
        return torch.where(has_valid_seg, best_s, torch.zeros_like(best_s))

    def _score_batch_torch(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> torch.Tensor:
        if traj_xyyaw.ndim != 4 or traj_xyyaw.shape[-1] < 2:
            raise RuntimeError(
                "NuScenesTokenScorer expects traj_xyyaw with shape (batch, candidates, horizon, 3); "
                f"got {tuple(traj_xyyaw.shape)}"
            )
        if len(replays) != int(traj_xyyaw.shape[0]):
            raise RuntimeError(f"Replay batch length mismatch: replays={len(replays)} traj_batch={int(traj_xyyaw.shape[0])}")

        batch_size, num_candidates, horizon, traj_dim = tuple(traj_xyyaw.shape)
        del traj_dim
        if horizon <= 0:
            return torch.zeros((batch_size, num_candidates), device=traj_xyyaw.device, dtype=torch.float32)

        traj = traj_xyyaw.to(device=traj_xyyaw.device, dtype=torch.float32)
        gt_batch = self._prepare_torch_gt_batch(replays, traj_horizon=int(horizon), device=traj.device)
        cmp_valid = gt_batch["cmp_valid"]
        cmp_mask = cmp_valid.unsqueeze(1)
        cand_xy = traj[..., :2]

        if traj.shape[-1] >= 3:
            cand_yaw = traj[..., 2]
        else:
            cand_valid = cmp_mask.expand(-1, num_candidates, -1).reshape(-1, horizon)
            cand_yaw = self._path_yaw_from_xy_torch(
                cand_xy.reshape(-1, horizon, 2),
                cand_valid,
            ).reshape(batch_size, num_candidates, horizon)

        pos_err = torch.linalg.norm(cand_xy - gt_batch["gt_cmp_xy"].unsqueeze(1), dim=-1).to(dtype=torch.float32)
        first_err = torch.where(
            cmp_valid[:, :1],
            pos_err[:, :, 0],
            torch.zeros((batch_size, num_candidates), device=traj.device, dtype=torch.float32),
        )
        final_err = self._gather_last_valid(pos_err, gt_batch["cmp_lengths"])
        mean_err = self._masked_mean(pos_err, cmp_mask, dim=2)

        yaw_err = torch.atan2(
            torch.sin(cand_yaw - gt_batch["gt_cmp_yaw"].unsqueeze(1)),
            torch.cos(cand_yaw - gt_batch["gt_cmp_yaw"].unsqueeze(1)),
        ).abs().to(dtype=torch.float32)
        mean_yaw_err = self._masked_mean(yaw_err, cmp_mask, dim=2)

        cand_last_xy = self._gather_last_valid_xy(cand_xy, gt_batch["cmp_lengths"])
        cand_progress = self._project_progress_torch(
            cand_last_xy,
            gt_batch["gt_full_xy"],
            gt_batch["gt_full_s"],
            gt_batch["gt_full_mask"],
        )
        progress_ratio = (cand_progress / gt_batch["gt_total_len"].unsqueeze(1)).clamp(0.0, 1.5)

        cand_prev = torch.cat(
            [
                torch.zeros((batch_size, num_candidates, 1, 2), device=traj.device, dtype=torch.float32),
                cand_xy[:, :, :-1, :],
            ],
            dim=2,
        )
        cand_vel = cand_xy - cand_prev
        cand_acc = cand_vel[:, :, 1:, :] - cand_vel[:, :, :-1, :]
        acc_valid = (cmp_valid[:, 1:] & cmp_valid[:, :-1]).unsqueeze(1)
        smooth_pen = self._masked_mean(torch.linalg.norm(cand_acc, dim=-1).to(dtype=torch.float32), acc_valid, dim=2)

        return (
            (2.0 * progress_ratio)
            - (0.35 * mean_err)
            - (0.50 * final_err)
            - (0.35 * first_err)
            - (0.20 * mean_yaw_err)
            - (0.05 * smooth_pen)
        ).to(dtype=torch.float32)

    def score_with_details(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        scores, details = self._score_batch(replays, traj_xyyaw, include_debug_context=True)
        return torch.from_numpy(scores), details

    def score(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            return self._score_batch_torch(replays, traj_xyyaw)

    def dump_debug_artifacts(
        self,
        replays: Sequence[dict[str, Any]],
        traj_xyyaw: torch.Tensor,
        *,
        out_dir: str | Path,
        step_tag: str,
        top_k: int = 4,
    ) -> list[dict[str, str]]:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon as MplPolygon

        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        _, details = self.score_with_details(replays, traj_xyyaw)
        artifacts: list[dict[str, str]] = []
        for sample_detail in details:
            sample_token = str(sample_detail["sample_token"])
            token_slug = sample_token.replace("/", "_")[:24]
            prefix = f"{step_tag}_b{int(sample_detail['batch_index']):03d}_{token_slug}"
            json_path = out_root / f"{prefix}.json"
            png_path = out_root / f"{prefix}.png"

            ranked = sorted(sample_detail["candidates"], key=lambda item: float(item["score"]), reverse=True)
            kept = ranked[: max(1, int(top_k))]
            payload = {
                "sample_token": sample_token,
                "step_tag": str(step_tag),
                "batch_index": int(sample_detail["batch_index"]),
                "gt_origin_shift_xy": np.asarray(sample_detail["gt_origin_shift_xy"], dtype=np.float32).tolist(),
                "gt_xy": np.asarray(sample_detail["gt_xy"], dtype=np.float32).tolist(),
                "gt_yaw": np.asarray(sample_detail["gt_yaw"], dtype=np.float32).tolist(),
                "gt_history_xy": np.asarray(sample_detail.get("gt_history_xy", []), dtype=np.float32).tolist(),
                "scene_objects": [
                    {
                        "category": str(item["category"]),
                        "center_xy": np.asarray(item["center_xy"], dtype=np.float32).tolist(),
                        "length_m": float(item["length_m"]),
                        "width_m": float(item["width_m"]),
                        "yaw_rad": float(item["yaw_rad"]),
                        "speed_mps": float(item["speed_mps"]),
                        "num_lidar_pts": int(item["num_lidar_pts"]),
                        "num_radar_pts": int(item["num_radar_pts"]),
                        "corners_xy": np.asarray(item["corners_xy"], dtype=np.float32).tolist(),
                    }
                    for item in sample_detail.get("scene_objects", [])
                ],
                "map_patch_radius_m": float(sample_detail.get("map_patch_radius_m", _DEFAULT_PATCH_RADIUS_M)),
                "map_layers": {
                    str(layer): [
                        np.asarray(geom, dtype=np.float32).tolist()
                        for geom in geometries
                    ]
                    for layer, geometries in (sample_detail.get("map_layers", {}) or {}).items()
                },
                "render_layers": {},
                "candidates": [
                    {
                        "candidate_index": int(item["candidate_index"]),
                        "score": float(item["score"]),
                        "progress_ratio": float(item["progress_ratio"]),
                        "mean_error_m": float(item["mean_error_m"]),
                        "final_error_m": float(item["final_error_m"]),
                        "first_error_m": float(item["first_error_m"]),
                        "mean_yaw_error_rad": float(item["mean_yaw_error_rad"]),
                        "smoothness_penalty_raw": float(item["smoothness_penalty_raw"]),
                        "score_terms": {key: float(val) for key, val in item["score_terms"].items()},
                        "traj_xyyaw": np.asarray(item["traj_xyyaw"], dtype=np.float32).tolist(),
                    }
                    for item in kept
                ],
            }
            payload["render_layers"] = self._build_render_layers(payload["map_layers"])
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=160)
            fig.patch.set_facecolor("#f4f2ed")
            ax.set_facecolor("#fbfbf8")

            render_layers = payload["render_layers"]
            for polygon in render_layers.get("drivable_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(MplPolygon(patch[:, :2], closed=True, facecolor="#e6e4df", edgecolor="none", alpha=0.95, zorder=0))
            for polygon in render_layers.get("road_surface_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(MplPolygon(patch[:, :2], closed=True, facecolor="#d8d5cf", edgecolor="none", alpha=0.95, zorder=0.5))
            for polygon in render_layers.get("lane_fill_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(MplPolygon(patch[:, :2], closed=True, facecolor="#d2d0ca", edgecolor="none", alpha=0.45, zorder=0.7))
            for polygon in render_layers.get("walkway_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(MplPolygon(patch[:, :2], closed=True, facecolor="#b8d8b0", edgecolor="none", alpha=0.55, zorder=0.8))
            for polygon in render_layers.get("crossing_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(
                        MplPolygon(
                            patch[:, :2],
                            closed=True,
                            facecolor="#f7f7f4",
                            edgecolor="#b1b1ab",
                            linewidth=0.9,
                            hatch="///",
                            alpha=0.85,
                            zorder=0.9,
                        )
                    )
            for polygon in render_layers.get("crossing_stripe_polygons", []):
                patch = np.asarray(polygon, dtype=np.float32)
                if patch.shape[0] >= 3:
                    ax.add_patch(
                        MplPolygon(
                            patch[:, :2],
                            closed=True,
                            facecolor="#fefefe",
                            edgecolor="none",
                            alpha=0.95,
                            zorder=1.0,
                        )
                    )

            for line in render_layers.get("road_edge_lines", []):
                pts = np.asarray(line, dtype=np.float32)
                if pts.shape[0] >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], color="#8b8881", linewidth=1.2, alpha=0.85, zorder=1.2)
            for line in render_layers.get("lane_boundary_lines", []):
                pts = np.asarray(line, dtype=np.float32)
                if pts.shape[0] >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], color="#9f9b92", linewidth=0.8, alpha=0.55, zorder=1.25)
            for line in render_layers.get("lane_marking_lines", []):
                pts = np.asarray(line, dtype=np.float32)
                if pts.shape[0] >= 2:
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        color="#6c6c6a",
                        linewidth=1.0,
                        alpha=0.9,
                        linestyle=(0, (4.0, 3.0)),
                        zorder=1.3,
                    )
            for line in render_layers.get("lane_centerlines", []):
                pts = np.asarray(line, dtype=np.float32)
                if pts.shape[0] >= 2:
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        color="#4e97d1",
                        linewidth=1.2,
                        alpha=0.95,
                        linestyle=(0, (8.0, 4.0)),
                        zorder=1.4,
                    )
            for arrow in render_layers.get("lane_direction_arrows", []):
                pts = np.asarray(arrow, dtype=np.float32)
                if pts.shape[0] >= 2:
                    start = pts[0]
                    end = pts[1]
                    delta = end - start
                    ax.arrow(
                        float(start[0]),
                        float(start[1]),
                        float(delta[0]),
                        float(delta[1]),
                        width=0.03,
                        head_width=0.35,
                        head_length=0.45,
                        length_includes_head=True,
                        color="#5f5b54",
                        alpha=0.7,
                        zorder=1.45,
                    )

            gt_xy = np.asarray(sample_detail["gt_xy"], dtype=np.float32)
            gt_history_xy = np.asarray(sample_detail.get("gt_history_xy", []), dtype=np.float32)
            for obj in payload["scene_objects"]:
                corners = np.asarray(obj["corners_xy"], dtype=np.float32)
                if corners.shape[0] >= 3:
                    color = self._category_color(str(obj["category"]))
                    ax.add_patch(MplPolygon(corners[:, :2], closed=True, facecolor=color, edgecolor="#2c3e50", alpha=0.45, linewidth=1.0, zorder=2))
                    center = np.asarray(obj["center_xy"], dtype=np.float32)
                    ax.text(center[0], center[1], str(obj["category"]), fontsize=6.5, color="#2c3e50", ha="center", va="center", zorder=3)

            ego_box = self._box_corners_xy(0.0, 0.0, length=4.6, width=1.9, yaw=0.0)
            ax.add_patch(MplPolygon(ego_box, closed=True, facecolor="#111111", edgecolor="#111111", alpha=0.9, linewidth=1.2, zorder=4))
            ax.scatter([0.0], [0.0], color="#e63946", marker="x", s=70, label="ego_origin", zorder=5)
            if gt_history_xy.ndim == 2 and gt_history_xy.shape[0] > 0:
                ax.plot(gt_history_xy[:, 0], gt_history_xy[:, 1], color="#7f8c8d", linestyle="--", linewidth=1.8, label="ego_history", zorder=5)
            cmap = plt.get_cmap("viridis", max(2, len(kept)))
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linewidth=2.7, label="gt_future", zorder=6)
            for rank, item in enumerate(kept):
                cand_xyyaw = np.asarray(item["traj_xyyaw"], dtype=np.float32)
                cand_xy = cand_xyyaw[:, :2]
                label = f"rank{rank + 1} idx{int(item['candidate_index'])} score={float(item['score']):.3f}"
                candidate_style = self._candidate_line_style(rank=rank, total=len(kept))
                ax.plot(
                    cand_xy[:, 0],
                    cand_xy[:, 1],
                    color=cmap(rank),
                    linewidth=float(candidate_style["linewidth"]),
                    alpha=float(candidate_style["alpha"]),
                    label=label,
                    zorder=7,
                )
                ax.scatter(
                    [cand_xy[-1, 0]],
                    [cand_xy[-1, 1]],
                    color=cmap(rank),
                    s=float(candidate_style["marker_size"]),
                    alpha=float(candidate_style["alpha"]),
                    zorder=8,
                )
            xlim, ylim = self._compute_plot_limits(
                gt_xy=gt_xy,
                gt_history_xy=gt_history_xy,
                candidates=kept,
                scene_objects=payload["scene_objects"],
                patch_radius=float(payload["map_patch_radius_m"]),
            )
            ax.set_title(f"NuScenes GRPO debug\n{sample_token}")
            ax.set_xlabel("forward x (m)")
            ax.set_ylabel("left y (m)")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.15, color="#737373")
            semantic_handles = [
                Line2D([0], [0], color="black", linewidth=2.7, label="gt_future"),
                Line2D([0], [0], color="#7f8c8d", linewidth=1.8, linestyle="--", label="ego_history"),
                Line2D([0], [0], color="#6c6c6a", linewidth=1.0, linestyle=(0, (4.0, 3.0)), label="lane_marking"),
                Line2D([0], [0], color="#4e97d1", linewidth=1.2, linestyle=(0, (8.0, 4.0)), label="lane_centerline"),
                Line2D([0], [0], marker="s", markersize=8, linestyle="None", markerfacecolor="#e74c3c", markeredgecolor="#2c3e50", label="vehicles/agents"),
            ]
            candidate_handles = [
                Line2D(
                    [0],
                    [0],
                    color=cmap(rank),
                    linewidth=float(self._candidate_line_style(rank=rank, total=len(kept))["linewidth"]),
                    alpha=float(self._candidate_line_style(rank=rank, total=len(kept))["alpha"]),
                    label=f"rank{rank + 1} score={float(item['score']):.3f}",
                )
                for rank, item in enumerate(kept)
            ]
            ax.legend(handles=semantic_handles + candidate_handles, loc="upper right", fontsize=8, framealpha=0.95)
            fig.tight_layout()
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)

            artifacts.append(
                {
                    "sample_token": sample_token,
                    "json_path": str(json_path),
                    "png_path": str(png_path),
                }
            )
        return artifacts


__all__ = [
    "NuScenesTokenScorer",
]
