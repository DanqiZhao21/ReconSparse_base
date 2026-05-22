import json
import os
from typing import Dict, Optional, List, Tuple

import numpy as np
from shapely.geometry import Polygon, box, LineString, Point
from shapely import affinity
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from pyquaternion import Quaternion

# Defaults aligned with tools/smalltool/get_info.py
DEFAULT_DATA_ROOT = "/OpenDataset/nuscenes/nuscenes/v1.0-trainval"
DEFAULT_VERSION = "v1.0-trainval"
PATCH_RADIUS = 20.0
EGO_LENGTH = 4.2
EGO_WIDTH = 1.9

# -------------------------------
# Polygon / Map helpers (adapted from tools/smalltool/get_info.py)
# -------------------------------

def extract_polygon(map_api: NuScenesMap, polygon_token: str) -> Polygon:
    polygon_record = map_api.get("polygon", polygon_token)
    exterior_coords = [(map_api.get("node", t)["x"], map_api.get("node", t)["y"]) for t in polygon_record["exterior_node_tokens"]]
    interiors = []
    for hole in polygon_record["holes"]:
        interior_coords = [(map_api.get("node", t)["x"], map_api.get("node", t)["y"]) for t in hole["node_tokens"]]
        if interior_coords:
            interiors.append(interior_coords)
    return Polygon(exterior_coords, interiors)


def quat_to_yaw(q: list | tuple) -> float:
    w, x, y, z = q
    return Quaternion(w=w, x=x, y=y, z=z).yaw_pitch_roll[0]


def oriented_box(x: float, y: float, length: float, width: float, yaw: float) -> Polygon:
    rect = box(x - length/2, y - width/2, x + length/2, y + width/2)
    return affinity.rotate(rect, np.degrees(yaw), origin=(x, y))


def get_layer_polygons(nusc_map: NuScenesMapExplorer, layer: str, tokens: list[str]) -> list[Polygon]:
    polys: list[Polygon] = []
    for token in tokens:
        record = nusc_map.map_api.get(layer, token)
        if layer == "drivable_area":
            polygons = [extract_polygon(nusc_map.map_api, t) for t in record["polygon_tokens"]]
            polys.extend(polygons)
        else:
            polys.append(extract_polygon(nusc_map.map_api, record["polygon_token"]))
    return polys

#NOTE 判断 ego 是否 off-road 以及 静态碰撞
def compute_off_road_and_static(ego_poly: Polygon, nusc_map: NuScenesMapExplorer, x: float, y: float, radius: float):
    box_coords = (x-radius, y-radius, x+radius, y+radius)
    layers = ["drivable_area", "walkway", "road_block", "carpark_area", "lane"]
    recs = nusc_map.get_records_in_patch(box_coords, layers, mode="intersect")

    drivable_polys = get_layer_polygons(nusc_map, "drivable_area", recs.get("drivable_area", []))
    off_road = True if not any(ego_poly.within(p) or ego_poly.intersects(p) for p in drivable_polys) else False

    collisions = []
    for layer in ["walkway", "road_block", "carpark_area", "lane"]:
        for poly in get_layer_polygons(nusc_map, layer, recs.get(layer, [])):
            if ego_poly.intersects(poly):
                collisions.append({"layer": layer})
    return off_road, collisions


# -------------------------------
# Frame & Ego helpers
# -------------------------------

def get_frame_info(sample_token: str, nusc: NuScenes, nusc_map: NuScenesMapExplorer, patch_radius: float = PATCH_RADIUS) -> dict:
    sample_record = nusc.get("sample", sample_token)
    sd_token = sample_record["data"].get("LIDAR_TOP", list(sample_record["data"].values())[0])
    sd = nusc.get("sample_data", sd_token)
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_translation = pose["translation"]
    ego_rotation = pose["rotation"]
    yaw = quat_to_yaw(ego_rotation)
    ego_poly = oriented_box(ego_translation[0], ego_translation[1], 4.2, 1.9, yaw)

    off_road, static_cols = compute_off_road_and_static(ego_poly, nusc_map, ego_translation[0], ego_translation[1], patch_radius)

    frame_info = {
        "sample_token": sample_token,
        "timestamp": sample_record["timestamp"],
        "ego_translation": ego_translation,
        "ego_rotation": ego_rotation,
        "ego_yaw": float(yaw),
        "off_road": bool(off_road),
        "static_collisions": static_cols,
    }
    return frame_info


def build_scene_ego_path(nusc: NuScenes, sample_token: str) -> LineString:
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    tok = scene["first_sample_token"]
    pts: list[tuple[float,float]] = []
    while tok:
        s = nusc.get("sample", tok)
        sd_token = s["data"].get("LIDAR_TOP", list(s["data"].values())[0])
        sd = nusc.get("sample_data", sd_token)
        x, y, _ = nusc.get("ego_pose", sd["ego_pose_token"])["translation"]
        pts.append((float(x), float(y)))
        tok = s.get("next", None)
    return LineString(pts)


def nearest_point_and_tangent(line: LineString, x: float, y: float):
    proj = float(line.project(Point(x, y)))
    p = line.interpolate(proj)
    dl = max(0.1, min(1.0, line.length * 0.001))
    p_prev = line.interpolate(max(0.0, proj-dl))
    p_next = line.interpolate(min(line.length, proj+dl))
    vec = np.array([float(p_next.x - p_prev.x), float(p_next.y - p_prev.y)], dtype=np.float32)
    nrm = np.linalg.norm(vec)
    return p, vec/nrm if nrm > 1e-6 else np.array([1.0,0.0], dtype=np.float32)


def _closest_tangent_from_centerlines(centerlines: List[List[Tuple[float, float]]], x: float, y: float) -> Optional[np.ndarray]:
    """Find nearest segment among multiple polylines and return its unit tangent vector (dx, dy)."""
    q = np.array([x, y], dtype=np.float32)
    best = None
    best_d2 = float('inf')
    for pl in centerlines:
        if len(pl) < 2:
            continue
        pts = np.asarray(pl, dtype=np.float32)
        segs = pts[1:] - pts[:-1]  # (N-1,2)
        # Project q onto each segment [p0,p1]
        p0 = pts[:-1]
        v = segs
        vv = (v * v).sum(-1) + 1e-9
        t = ((q - p0) * v).sum(-1) / vv
        t = np.clip(t, 0.0, 1.0)
        proj = p0 + (t[:, None] * v)
        d2 = ((proj - q) ** 2).sum(-1)
        idx = int(np.argmin(d2))
        if float(d2[idx]) < best_d2:
            best_d2 = float(d2[idx])
            # tangent is along the segment
            tan = v[idx]
            nrm = float(np.linalg.norm(tan))
            best = tan / nrm if nrm > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
    return best


def compute_metrics_from_snapshot(snapshot: Dict[str, any], ego_state: Dict[str, float]) -> Dict[str, any]:
    """
    Compute drivable compliance, direction compliance, and static/dynamic collisions
    from a pre-built environment snapshot given current ego pose.

    snapshot schema (per keyframe):
    {
      'drivable_polygons': [ [[x,y],...], ... ],
      'lanes_centerlines': [ [[x,y],...], ... ],
      'static_objects': [ {'category':str, 'poly': [[x,y],...]}, ... ],
      'dynamic_objects': [ {'token':str,'category':str,'poly': [[x,y],...]}, ... ]
    }
    ego_state: {'x':float, 'y':float, 'yaw':float, 'length':float=EGO_LENGTH, 'width':float=EGO_WIDTH}
    """
    x = float(ego_state.get('x', 0.0))
    y = float(ego_state.get('y', 0.0))
    yaw = float(ego_state.get('yaw', 0.0))
    length = float(ego_state.get('length', EGO_LENGTH))
    width = float(ego_state.get('width', EGO_WIDTH))

    ego_poly = oriented_box(x, y, length, width, yaw)

    # Drivable compliance
    drivable_compliance = 1.0
    try:
        drivable_polys = [Polygon(p) for p in snapshot.get('drivable_polygons', [])]
        if len(drivable_polys) > 0:
            on_drv = any(ego_poly.within(p) or ego_poly.intersects(p) for p in drivable_polys)
            drivable_compliance = 1.0 if on_drv else 0.0
    except Exception:
        drivable_compliance = 1.0

    # Direction compliance using nearest lane centerline tangent
    driving_direction_compliance = 1.0
    try:
        cls = snapshot.get('lanes_centerlines', [])
        tan = _closest_tangent_from_centerlines(cls, x, y)
        if tan is not None:
            lane_yaw = float(np.arctan2(tan[1], tan[0]))
            yaw_diff_deg = float(np.degrees(np.arctan2(np.sin(yaw - lane_yaw), np.cos(yaw - lane_yaw))))
            ay = abs(yaw_diff_deg)
            driving_direction_compliance = 1.0 if ay < 30 else (0.5 if ay < 60 else 0.0)
    except Exception:
        driving_direction_compliance = 1.0

    # Collisions
    static_collision = False
    dynamic_collision = False
    try:
        for obj in snapshot.get('static_objects', []) or []:
            try:
                poly = Polygon(obj.get('poly', []))
                if ego_poly.intersects(poly):
                    static_collision = True
                    break
            except Exception:
                continue
        if not static_collision:
            for obj in snapshot.get('dynamic_objects', []) or []:
                try:
                    poly = Polygon(obj.get('poly', []))
                    if ego_poly.intersects(poly):
                        dynamic_collision = True
                        break
                except Exception:
                    continue
    except Exception:
        pass

    collision = bool(static_collision or dynamic_collision)
    return {
        'drivable_compliance': float(drivable_compliance),
        'driving_direction_compliance': float(driving_direction_compliance),
        'static_collision': bool(static_collision),
        'dynamic_collision': bool(dynamic_collision),
        'collision': bool(collision),
        'collision_status': 'At-Fault' if collision else 'No-Collision',
    }


def check_dynamic_collision(frame_info: dict, nusc: NuScenes) -> list:
    ego_x, ego_y, _ = frame_info["ego_translation"]
    ego_yaw = frame_info.get("ego_yaw", 0.0)
    ego_poly = oriented_box(ego_x, ego_y, 4.2, 1.9, ego_yaw)
    collisions = []
    sample_rec = nusc.get("sample", frame_info["sample_token"])
    for ann_token in sample_rec["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if not (ann["category_name"].startswith("vehicle") or ann["category_name"].startswith("human")):
            continue
        x, y, _ = ann["translation"]
        # nuScenes sample_annotation.size is [width, length, height].
        w, l, _ = ann["size"]
        yaw = quat_to_yaw(ann["rotation"]) if "rotation" in ann else 0.0
        poly = oriented_box(x, y, l, w, yaw)
        if ego_poly.intersects(poly):
            collisions.append({"token": ann_token, "category": ann["category_name"]})
    return collisions


def compute_ego_metrics(frame_info: dict, nusc_map: NuScenesMapExplorer, centerline: LineString):
    """Minimal metrics: drivable compliance and driving direction compliance."""
    ego_x, ego_y, _ = frame_info["ego_translation"]
    ego_poly = oriented_box(ego_x, ego_y, 4.2, 1.9, frame_info["ego_yaw"]) 
    patch_coords = (ego_x - PATCH_RADIUS, ego_y - PATCH_RADIUS, ego_x + PATCH_RADIUS, ego_y + PATCH_RADIUS)
    layers = ["drivable_area", "lane"]
    recs = nusc_map.get_records_in_patch(patch_coords, layers, mode="intersect")

    # Drivable compliance（是否 off-road）
    drivable_polys = get_layer_polygons(nusc_map, "drivable_area", recs.get("drivable_area", []))
    off_road = not any(ego_poly.within(p) or ego_poly.intersects(p) for p in drivable_polys)
    drivable_compliance = 0.0 if off_road else 1.0

    # Driving direction compliance（与中心线切线夹角）
    _, tan = nearest_point_and_tangent(centerline, ego_x, ego_y)
    lane_yaw = np.arctan2(tan[1], tan[0])
    yaw_diff_deg = np.degrees(np.arctan2(np.sin(frame_info["ego_yaw"] - lane_yaw), np.cos(frame_info["ego_yaw"] - lane_yaw)))
    if abs(yaw_diff_deg) < 30:
        driving_direction_compliance = 1.0
    elif abs(yaw_diff_deg) < 60:
        driving_direction_compliance = 0.5
    else:
        driving_direction_compliance = 0.0

    return {
        "drivable_compliance": drivable_compliance,
        "driving_direction_compliance": driving_direction_compliance,
    }


# -------------------------------
# Token map + top-level per-step
# -------------------------------

_token_cache: Dict[int, Dict[int, str]] = {}
_nusc_cache: Dict[str, NuScenes] = {}


def _load_token_frame_map(scene_id: int) -> Dict[int, str]:
    if scene_id in _token_cache:
        return _token_cache[scene_id]
    scene_dir = f"{scene_id:03d}"
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "nus", "data", scene_dir, "token_frame_map.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"token_frame_map.json not found for scene {scene_id}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        tok_to_frame = json.load(f)
    frame_to_tok = {int(v): str(k) for k, v in tok_to_frame.items()}
    _token_cache[scene_id] = frame_to_tok
    return frame_to_tok


def _get_nuscenes(dataroot: str, version: str) -> NuScenes:
    key = f"{version}|{dataroot}"
    if key in _nusc_cache:
        return _nusc_cache[key]
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    _nusc_cache[key] = nusc
    return nusc


def compute_step_metrics(*, scene_id: int, step_idx: int, dataroot: Optional[str] = None, version: Optional[str] = None) -> Dict[str, object]:
    dataroot = dataroot or DEFAULT_DATA_ROOT
    version = version or DEFAULT_VERSION

    # Only compute on keyframe-aligned steps (every 5 steps)
    if step_idx % 5 != 0:
        return {}

    frame_to_tok = _load_token_frame_map(scene_id)
    token = frame_to_tok.get(int(step_idx))
    if token is None:
        # If exact step not present, skip to avoid mismatched map data
        return {}

    nusc = _get_nuscenes(dataroot, version)
    sample_record = nusc.get("sample", token)
    scene_record = nusc.get("scene", sample_record["scene_token"])
    log_record = nusc.get("log", scene_record["log_token"]) 
    nusc_map = NuScenesMapExplorer(NuScenesMap(dataroot=dataroot, map_name=log_record["location"]))

    frame_info = get_frame_info(token, nusc, nusc_map, patch_radius=PATCH_RADIUS)
    dyn_cols = check_dynamic_collision(frame_info, nusc)
    centerline = build_scene_ego_path(nusc, token)
    ego_metrics = compute_ego_metrics(frame_info, nusc_map, centerline)

    collision = bool(frame_info.get("static_collisions", [])) or bool(dyn_cols)
    collision_status = "At-Fault" if collision else "No-Collision"

    out = dict(ego_metrics)
    out.update({
        # 碰撞总览与静/动拆分
        "collision": collision,
        "collision_status": collision_status,
        "static_collision": bool(frame_info.get("static_collisions", [])),
        "dynamic_collision": bool(dyn_cols),
        "static_collisions": frame_info.get("static_collisions", []),
        "dynamic_collisions": dyn_cols,
        # 便于追踪关键帧
        "sample_token": token,
    })
    return out


# # -------------------------------
# # TTC computation helpers
# # -------------------------------

# def _sample_timestamp(nusc: NuScenes, sample_token: str) -> float:
#     s = nusc.get("sample", sample_token)
#     return float(s["timestamp"]) / 1e6  # seconds


# def _ego_velocity(nusc: NuScenes, cur_sample_token: str) -> np.ndarray:
#     cur_s = nusc.get("sample", cur_sample_token)
#     prev_tok = cur_s.get("prev", None)
#     next_tok = cur_s.get("next", None)
#     # Prefer previous for velocity
#     ref_tok = prev_tok if prev_tok else next_tok
#     if not ref_tok:
#         return np.zeros(2, dtype=np.float32)
#     # positions
#     def _pose_xy(sample_token: str) -> np.ndarray:
#         sd_token = nusc.get("sample", sample_token)["data"].get("LIDAR_TOP", list(nusc.get("sample", sample_token)["data"].values())[0])
#         pose = nusc.get("ego_pose", nusc.get("sample_data", sd_token)["ego_pose_token"]) 
#         x, y, _ = pose["translation"]
#         return np.array([float(x), float(y)], dtype=np.float32)
#     p_cur = _pose_xy(cur_sample_token)
#     p_ref = _pose_xy(ref_tok)
#     t_cur = _sample_timestamp(nusc, cur_sample_token)
#     t_ref = _sample_timestamp(nusc, ref_tok)
#     dt = max(1e-3, abs(t_cur - t_ref))
#     # sign: if using next, reverse to represent backward diff
#     sign = 1.0 if ref_tok == prev_tok else -1.0
#     v = sign * (p_cur - p_ref) / dt
#     return v.astype(np.float32)


# def _ann_velocity(nusc: NuScenes, ann_token: str) -> np.ndarray:
#     ann = nusc.get("sample_annotation", ann_token)
#     prev_tok = ann.get("prev", None)
#     next_tok = ann.get("next", None)
#     ref_tok = prev_tok if prev_tok else next_tok
#     if not ref_tok:
#         return np.zeros(2, dtype=np.float32)
#     def _ann_xy(ann_token: str) -> np.ndarray:
#         a = nusc.get("sample_annotation", ann_token)
#         x, y, _ = a["translation"]
#         return np.array([float(x), float(y)], dtype=np.float32)
#     p_cur = _ann_xy(ann_token)
#     p_ref = _ann_xy(ref_tok)
#     t_cur = _sample_timestamp(nusc, nusc.get("sample_annotation", ann_token)["sample_token"])
#     t_ref = _sample_timestamp(nusc, nusc.get("sample_annotation", ref_tok)["sample_token"])
#     dt = max(1e-3, abs(t_cur - t_ref))
#     sign = 1.0 if ref_tok == prev_tok else -1.0
#     v = sign * (p_cur - p_ref) / dt
#     return v.astype(np.float32)


# def _radius_from_box(length: float, width: float) -> float:
#     return float(np.sqrt((length * 0.5) ** 2 + (width * 0.5) ** 2))


# def _ttc_bounding_circles(p_rel: np.ndarray, v_rel: np.ndarray, r_sum: float) -> float:
#     # Solve ||p + v t||^2 = r^2 for minimal t >= 0 when approaching
#     # Quadratic: (v·v) t^2 + 2 (p·v) t + (p·p - r^2) = 0
#     pv = float(np.dot(p_rel, v_rel))
#     vv = float(np.dot(v_rel, v_rel))
#     pp = float(np.dot(p_rel, p_rel))
#     if vv <= 1e-8:
#         return float("inf")
#     disc = pv * pv - vv * (pp - r_sum * r_sum)
#     if disc < 0:
#         return float("inf")
#     # Only consider future roots and approaching (pv < 0)
#     if pv >= 0:
#         return float("inf")
#     t1 = (-pv - np.sqrt(disc)) / vv
#     t2 = (-pv + np.sqrt(disc)) / vv
#     roots = [t for t in (t1, t2) if t >= 0]
#     if len(roots) == 0:
#         return float("inf")
#     return float(min(roots))


# def _compute_ttc(*, nusc: NuScenes, frame_info: dict) -> float:
#     # Ego
#     ego_x, ego_y, _ = frame_info["ego_translation"]
#     ego_v = _ego_velocity(nusc, frame_info["sample_token"])  # m/s
#     ego_speed = float(np.linalg.norm(ego_v))
#     # Ego radius from box 4.2 x 1.9
#     r_ego = _radius_from_box(4.2, 1.9)

#     # Dynamic obstacles
#     sample_rec = nusc.get("sample", frame_info["sample_token"]) 
#     ttc_min = float("inf")
#     for ann_token in sample_rec["anns"]:
#         ann = nusc.get("sample_annotation", ann_token)
#         # focus on vehicles/humans
#         if not (ann["category_name"].startswith("vehicle") or ann["category_name"].startswith("human")):
#             continue
#         x, y, _ = ann["translation"]
#         w, l, _ = ann["size"]
#         p_rel = np.array([float(x - ego_x), float(y - ego_y)], dtype=np.float32)
#         v_ann = _ann_velocity(nusc, ann_token)
#         v_rel = v_ann - ego_v
#         r_sum = r_ego + _radius_from_box(float(l), float(w))
#         ttc = _ttc_bounding_circles(p_rel, v_rel, r_sum)
#         if ttc < ttc_min:
#             ttc_min = ttc

#     # Static obstacles (map): estimate TTC in forward half-plane using ego speed
#     if np.isfinite(ttc_min):
#         return ttc_min
#     if ego_speed < 1e-3:
#         return float("inf")

#     # Forward direction
#     yaw = frame_info.get("ego_yaw", 0.0)
#     heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
#     # Nearest distance to any static polygon whose centroid is in front
#     # Use map around ego
#     dataroot_key = ""  # not needed; we can reuse available info via compute_off_road_and_static
#     # Retrieve polygons via a small helper
#     # Rebuild minimal NuScenesMapExplorer for this function would be expensive; static TTC will be inf unless collision is imminent.
#     # For efficiency, skip static TTC unless already off-road in front.
#     return float("inf")


# def _is_agent_ahead(ego_xy: np.ndarray, ego_yaw: float, agent_xy: np.ndarray) -> bool:
#     heading = np.array([np.cos(ego_yaw), np.sin(ego_yaw)], dtype=np.float32)
#     return float(np.dot(agent_xy - ego_xy, heading)) > 0.0


# def _is_agent_behind(ego_xy: np.ndarray, ego_yaw: float, agent_xy: np.ndarray) -> bool:
#     heading = np.array([np.cos(ego_yaw), np.sin(ego_yaw)], dtype=np.float32)
#     return float(np.dot(agent_xy - ego_xy, heading)) < 0.0


# def _compute_ttc_nuplan_style(*, nusc: NuScenes, frame_info: dict, sampling_interval: float = 0.1, horizon_s: float = 1.0) -> float:
#     """Approximate nuPlan TTC: sample future times, constant-velocity propagation,
#     construct oriented boxes for ego & agents, check intersections, and return earliest collision time.

#     Differences to nuPlan:
#     - No semantic intersection layer; we use ahead/behind tests and simple ego-area flags.
#     - We estimate velocities from adjacent keyframes.
#     """
#     # future time indices similar to [0,3,6,9] * interval_length (0.1s → 0,0.3,0.6,0.9)
#     future_time_idcs = np.arange(0, int(horizon_s / sampling_interval), 3)
#     times = [float(i) * float(sampling_interval) for i in future_time_idcs]

#     ego_x, ego_y, _ = frame_info["ego_translation"]
#     ego_xy = np.array([float(ego_x), float(ego_y)], dtype=np.float32)
#     ego_yaw = float(frame_info.get("ego_yaw", 0.0))
#     ego_v = _ego_velocity(nusc, frame_info["sample_token"])  # m/s
#     ego_speed = float(np.linalg.norm(ego_v))
#     if ego_speed < float(STOPPED_SPEED_THRESHOLD):
#         return float("inf")

#     r_ego = _radius_from_box(4.2, 1.9)
#     sample_rec = nusc.get("sample", frame_info["sample_token"]) 
#     ttc_min = float("inf")

#     # Preload agent geometry/velocities
#     agents = []
#     for ann_token in sample_rec["anns"]:
#         ann = nusc.get("sample_annotation", ann_token)
#         if not (ann["category_name"].startswith("vehicle") or ann["category_name"].startswith("human")):
#             continue
#         x, y, _ = ann["translation"]
#         w, l, _ = ann["size"]
#         yaw = quat_to_yaw(ann["rotation"]) if "rotation" in ann else 0.0
#         v_ann = _ann_velocity(nusc, ann_token)
#         agents.append({
#             "xy": np.array([float(x), float(y)], dtype=np.float32),
#             "lw": (float(l), float(w)),
#             "yaw": float(yaw),
#             "v": v_ann.astype(np.float32),
#         })

#     # Ego area flags (current time)
#     ego_area_flags = {
#         "MULTIPLE_LANES": False,
#         "NON_DRIVABLE_AREA": bool(frame_info.get("off_road", False)),
#     }
#     # If caller computed richer ego_areas, we could pass them to refine conditions.

#     for t in times:
#         # Ego future polygon
#         ego_future_xy = ego_xy + ego_v * float(t)
#         ego_poly_t = oriented_box(float(ego_future_xy[0]), float(ego_future_xy[1]), 4.2, 1.9, ego_yaw)

#         for ag in agents:
#             ag_future_xy = ag["xy"] + ag["v"] * float(t)
#             ag_poly_t = oriented_box(float(ag_future_xy[0]), float(ag_future_xy[1]), ag["lw"][0], ag["lw"][1], ag["yaw"])

#             if not ego_poly_t.intersects(ag_poly_t):
#                 continue

#             # Conditions similar to nuPlan: skip if ego is stopped; consider ahead or permissive in complex areas.
#             if ego_speed < float(STOPPED_SPEED_THRESHOLD):
#                 continue

#             ahead = _is_agent_ahead(ego_future_xy, ego_yaw, ag_future_xy)
#             behind = _is_agent_behind(ego_future_xy, ego_yaw, ag_future_xy)
#             if ahead or ((ego_area_flags["MULTIPLE_LANES"] or ego_area_flags["NON_DRIVABLE_AREA"]) and not behind):
#                 ttc_min = min(ttc_min, float(t))
#                 # early exit if immediate collision
#                 if ttc_min <= 0.0:
#                     return 0.0

#     return float(ttc_min)
