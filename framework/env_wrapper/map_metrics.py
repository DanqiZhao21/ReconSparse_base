from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from shapely.geometry import Point, Polygon


DEFAULT_EGO_LENGTH_M = 4.2
DEFAULT_EGO_WIDTH_M = 1.9


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return float(angle)


def _ego_corners_xy(
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    ego_length_m: float,
    ego_width_m: float,
) -> np.ndarray:
    half_l = 0.5 * float(ego_length_m)
    half_w = 0.5 * float(ego_width_m)
    local = np.asarray(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    c = math.cos(float(ego_yaw))
    s = math.sin(float(ego_yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.asarray([float(ego_x), float(ego_y)], dtype=np.float64)


def _valid_polygon(coords: Any) -> Polygon | None:
    if not isinstance(coords, list) or len(coords) < 3:
        return None
    try:
        poly = Polygon(coords)
    except Exception:
        return None
    if poly.is_empty or (not poly.is_valid):
        return None
    return poly


def _centerline_alignment_stats(
    lanes_centerlines: Any,
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    same_dir_dot_threshold: float,
    same_dir_distance_margin_m: float,
) -> tuple[float | None, np.ndarray | None, float | None, float | None]:
    if not isinstance(lanes_centerlines, list) or len(lanes_centerlines) == 0:
        return None, None, None, None

    q = np.asarray([float(ego_x), float(ego_y)], dtype=np.float64)
    ego_heading = np.asarray([math.cos(float(ego_yaw)), math.sin(float(ego_yaw))], dtype=np.float64)
    nearest_dist = float("inf")
    nearest_tangent: np.ndarray | None = None
    nearest_dot: float | None = None
    same_dir_dist = float("inf")
    same_dir_tangent: np.ndarray | None = None
    same_dir_dot: float | None = None

    for centerline in lanes_centerlines:
        try:
            pts = np.asarray(centerline, dtype=np.float64)
        except Exception:
            continue
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            continue
        pts = pts[:, :2]
        seg_start = pts[:-1]
        seg_end = pts[1:]
        seg_vec = seg_end - seg_start
        seg_len_sq = np.sum(seg_vec * seg_vec, axis=1)
        valid = seg_len_sq > 1.0e-12
        if not np.any(valid):
            continue
        safe_len_sq = np.where(valid, seg_len_sq, 1.0)
        alpha = np.sum((q.reshape(1, 2) - seg_start) * seg_vec, axis=1) / safe_len_sq
        alpha = np.clip(alpha, 0.0, 1.0)
        proj = seg_start + alpha.reshape(-1, 1) * seg_vec
        dist = np.linalg.norm(q.reshape(1, 2) - proj, axis=1)
        dist = np.where(valid, dist, np.inf)
        idx = int(np.argmin(dist))
        tangent = seg_vec[idx]
        norm = float(np.linalg.norm(tangent))
        if norm <= 1.0e-12 or not math.isfinite(float(dist[idx])):
            continue
        tangent_unit = np.asarray(tangent / norm, dtype=np.float64)
        dot = float(np.clip(np.dot(ego_heading, tangent_unit), -1.0, 1.0))
        dist_i = float(dist[idx])
        if dist_i < nearest_dist:
            nearest_dist = dist_i
            nearest_tangent = tangent_unit
            nearest_dot = dot
        if dot >= float(same_dir_dot_threshold) and dist_i < same_dir_dist:
            same_dir_dist = dist_i
            same_dir_tangent = tangent_unit
            same_dir_dot = dot

    if nearest_tangent is None or not math.isfinite(nearest_dist):
        return None, None, None, None

    margin = max(0.0, float(same_dir_distance_margin_m))
    if same_dir_tangent is not None and same_dir_dist <= nearest_dist + margin:
        return float(same_dir_dist), same_dir_tangent, float(nearest_dist), float(nearest_dot)
    return float(nearest_dist), nearest_tangent, float(nearest_dist), float(nearest_dot)


def compute_craft_map_metrics(
    snapshot: Dict[str, Any] | None,
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    ego_length_m: float = DEFAULT_EGO_LENGTH_M,
    ego_width_m: float = DEFAULT_EGO_WIDTH_M,
    center_dev_max_m: float = 2.0,
    heading_dev_max_deg: float = 90.0,
    reverse_dot_threshold: float = -0.5,
    same_dir_dot_threshold: float = 0.2,
    same_dir_distance_margin_m: float = 0.75,
    opposite_min_lateral_m: float = 0.0,
) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {
            "map_has_drivable": False,
            "map_has_lane_centerline": False,
            "off_road": False,
            "opposite_lane": False,
            "driving_direction_violation": False,
        }

    out: Dict[str, Any] = {}

    corners = _ego_corners_xy(
        ego_x=float(ego_x),
        ego_y=float(ego_y),
        ego_yaw=float(ego_yaw),
        ego_length_m=float(ego_length_m),
        ego_width_m=float(ego_width_m),
    )

    drivable_polygons = snapshot.get("drivable_polygons", []) or []
    drivable = [_valid_polygon(poly) for poly in drivable_polygons]
    drivable = [poly for poly in drivable if poly is not None]
    out["map_has_drivable"] = bool(len(drivable) > 0)
    if len(drivable) > 0:
        corner_inside = []
        for corner in corners:
            pt = Point(float(corner[0]), float(corner[1]))
            corner_inside.append(any(poly.covers(pt) for poly in drivable))
        out["off_road"] = not all(corner_inside)
    else:
        out["off_road"] = False

    lateral_error_m, tangent, nearest_lateral_error_m, nearest_dot = _centerline_alignment_stats(
        snapshot.get("lanes_centerlines", snapshot.get("lane_centerlines", [])),
        ego_x=float(ego_x),
        ego_y=float(ego_y),
        ego_yaw=float(ego_yaw),
        same_dir_dot_threshold=float(same_dir_dot_threshold),
        same_dir_distance_margin_m=float(same_dir_distance_margin_m),
    )
    out["map_has_lane_centerline"] = bool(tangent is not None and lateral_error_m is not None)
    if tangent is not None and lateral_error_m is not None:
        center_max = max(1.0e-6, float(center_dev_max_m))
        heading_max_rad = max(1.0e-6, math.radians(float(heading_dev_max_deg)))
        ego_heading = np.asarray([math.cos(float(ego_yaw)), math.sin(float(ego_yaw))], dtype=np.float64)
        dot = float(np.clip(np.dot(ego_heading, tangent), -1.0, 1.0))
        heading_err_rad = abs(_wrap_angle(float(ego_yaw) - math.atan2(float(tangent[1]), float(tangent[0]))))
        opposite_lane = bool(
            dot < float(reverse_dot_threshold)
            and float(lateral_error_m) >= max(0.0, float(opposite_min_lateral_m))
        )
        out.update(
            {
                "centerline_lateral_error_m": float(lateral_error_m),
                "nearest_centerline_lateral_error_m": None
                if nearest_lateral_error_m is None
                else float(nearest_lateral_error_m),
                "center_dev_ratio": float(np.clip(float(lateral_error_m) / center_max, 0.0, 1.0)),
                "map_heading_dev_ratio": float(np.clip(heading_err_rad / heading_max_rad, 0.0, 1.0)),
                "map_heading_error_deg": float(math.degrees(heading_err_rad)),
                "lane_tangent_dot": float(dot),
                "nearest_lane_tangent_dot": None if nearest_dot is None else float(nearest_dot),
                "opposite_lane": opposite_lane,
                "driving_direction_violation": opposite_lane,
            }
        )
    else:
        out.update(
            {
                "opposite_lane": False,
                "driving_direction_violation": False,
            }
        )

    return out
