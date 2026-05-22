#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from shapely.geometry import Polygon, box, LineString, Point, MultiPolygon
from shapely import affinity
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from pyquaternion import Quaternion
import copy

# -------------------------------
# 默认配置
# -------------------------------
DATA_ROOT = "/OpenDataset/nuscenes/nuscenes/v1.0-trainval"
VERSION = "v1.0-trainval"
DEFAULT_SAMPLE_TOKEN = "40e413c922184255a94f08d3c10037e0"
PATCH_RADIUS = 20.0
STOPPED_SPEED_THRESHOLD = 0.1  # m/s for TTC

# -------------------------------
# Map / Polygon Helper
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
# Frame & Ego Helpers
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
    """Build expert path (centerline) from scene ego poses across all samples."""
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

# -------------------------------
# Dynamic Agent Collisions
# -------------------------------
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
        w, l, _ = ann["size"]
        yaw = quat_to_yaw(ann["rotation"]) if "rotation" in ann else 0.0
        poly = oriented_box(x, y, l, w, yaw)
        if ego_poly.intersects(poly):
            collisions.append({"token": ann_token, "category": ann["category_name"]})
    return collisions

# -------------------------------
# Ego Area / Drivable / Progress / Driving Direction / TTC
# -------------------------------
def compute_ego_metrics(frame_info: dict, nusc_map: NuScenesMapExplorer, centerline: LineString, states: np.ndarray, headings: np.ndarray, interval: float = 0.1):
    """
    states: (num_proposals, num_steps, 2) xy positions
    headings: (num_proposals, num_steps) yaw angles in radians
    """
    ego_x, ego_y, _ = frame_info["ego_translation"]
    ego_poly = oriented_box(ego_x, ego_y, 4.2, 1.9, frame_info["ego_yaw"])
    patch_coords = (ego_x - PATCH_RADIUS, ego_y - PATCH_RADIUS, ego_x + PATCH_RADIUS, ego_y + PATCH_RADIUS)
    layers = ["drivable_area", "lane"]
    recs = nusc_map.get_records_in_patch(patch_coords, layers, mode="intersect")

    # ---------------- Drivable Area Compliance ----------------
    drivable_polys = get_layer_polygons(nusc_map, "drivable_area", recs.get("drivable_area", []))
    off_road = not any(ego_poly.within(p) or ego_poly.intersects(p) for p in drivable_polys)
    drivable_compliance = 0.0 if off_road else 1.0

    # ---------------- Ego Area Classification ----------------
    lane_polys = get_layer_polygons(nusc_map, "lane", recs.get("lane", []))
    ego_areas = {
        "MULTIPLE_LANES": len([p for p in lane_polys if ego_poly.intersects(p)]) > 1,
        "NON_DRIVABLE_AREA": off_road,
        "ONCOMING_TRAFFIC": False,  # 可扩展
    }

    # ---------------- Progress ----------------
    p, _ = nearest_point_and_tangent(centerline, ego_x, ego_y)
    progress_m = float(np.linalg.norm(np.array([ego_x - p.x, ego_y - p.y])))

    # ---------------- Driving Direction Compliance ----------------
    # 简单用 ego heading 与 centerline 切线方向比较
    _, tan = nearest_point_and_tangent(centerline, ego_x, ego_y)
    lane_yaw = np.arctan2(tan[1], tan[0])
    yaw_diff_deg = np.degrees(np.arctan2(np.sin(frame_info["ego_yaw"] - lane_yaw), np.cos(frame_info["ego_yaw"] - lane_yaw)))
    if abs(yaw_diff_deg) < 30:
        driving_direction_compliance = 1.0
    elif abs(yaw_diff_deg) < 60:
        driving_direction_compliance = 0.5
    else:
        driving_direction_compliance = 0.0

    # ---------------- TTC (简单示例：距离/速度) ----------------
    # 如果没有速度信息，可用固定假设
    ttc = 1.0  # placeholder, 可使用真实预测轨迹计算最小碰撞时间

    return {
        "ego_areas": ego_areas,
        "drivable_compliance": drivable_compliance,
        "progress_m": progress_m,
        "driving_direction_compliance": driving_direction_compliance,
        "ttc": ttc
    }

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=DEFAULT_SAMPLE_TOKEN)
    parser.add_argument("--root", type=str, default=DATA_ROOT)
    parser.add_argument("--version", type=str, default=VERSION)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.root, verbose=False)
    sample_token = args.token
    sample_record = nusc.get("sample", sample_token)
    scene_record = nusc.get("scene", sample_record["scene_token"])
    log_record = nusc.get("log", scene_record["log_token"])
    nusc_map = NuScenesMapExplorer(NuScenesMap(dataroot=args.root, map_name=log_record["location"]))

    info = get_frame_info(sample_token, nusc, nusc_map, patch_radius=PATCH_RADIUS)
    info["dynamic_collisions"] = check_dynamic_collision(info, nusc)

    # Build centerline
    centerline = build_scene_ego_path(nusc, sample_token)
    
    # 示例 states/headings (可扩展为预测轨迹)
    states = np.array([[[info["ego_translation"][0], info["ego_translation"][1]]]])
    headings = np.array([[info["ego_yaw"]]])

    ego_metrics = compute_ego_metrics(info, nusc_map, centerline, states, headings)
    info.update(ego_metrics)

    print(json.dumps(info, indent=2))
