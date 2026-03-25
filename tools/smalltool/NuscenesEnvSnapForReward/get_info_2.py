#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from shapely.geometry import Polygon, box, MultiPolygon, Point, LineString
from shapely import affinity
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
# -------------------------------
# 默认配置（避免每次输入命令行）
# -------------------------------
DATA_ROOT = "/OpenDataset/nuscenes/nuscenes/v1.0-trainval"
VERSION = "v1.0-trainval"
DEFAULT_SAMPLE_TOKEN = "40e413c922184255a94f08d3c10037e0"
PATCH_RADIUS = 20.0

# -------------------------------
# Map Helper Functions
# -------------------------------
def extract_polygon(map_api: NuScenesMap, polygon_token: str) -> Polygon:
    polygon_record = map_api.get("polygon", polygon_token)
    exterior_coords = [
        (map_api.get("node", token)["x"], map_api.get("node", token)["y"]) for token in polygon_record["exterior_node_tokens"]
    ]
    interiors = []
    for hole in polygon_record["holes"]:
        interior_coords = [
            (map_api.get("node", token)["x"], map_api.get("node", token)["y"]) for token in hole["node_tokens"]
        ]
        if len(interior_coords) > 0:
            interiors.append(interior_coords)
    return Polygon(exterior_coords, interiors)

def quat_to_yaw(q: list | tuple) -> float:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return Quaternion(w=w, x=x, y=y, z=z).yaw_pitch_roll[0]

def oriented_box(x: float, y: float, length: float, width: float, yaw: float) -> Polygon:
    rect = box(x - length / 2.0, y - width / 2.0, x + length / 2.0, y + width / 2.0)
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

def compute_off_road_and_static(ego_poly: Polygon, nusc_map: NuScenesMapExplorer, x: float, y: float, radius: float) -> tuple[bool, list[dict]]:
    box_coords = (x - radius, y - radius, x + radius, y + radius)
    layers = ["drivable_area", "walkway", "road_block", "carpark_area", "lane"]
    recs = nusc_map.get_records_in_patch(box_coords, layers, mode="intersect")
    drivable_polys = get_layer_polygons(nusc_map, "drivable_area", recs.get("drivable_area", []))
    off_road = True
    if len(drivable_polys) > 0:
        off_road = not any(ego_poly.within(p) or ego_poly.intersects(p) for p in drivable_polys)
    collisions = []
    for layer in ["walkway", "road_block", "carpark_area", "lane"]:
        for poly in get_layer_polygons(nusc_map, layer, recs.get(layer, [])):
            if ego_poly.intersects(poly):
                collisions.append({"layer": layer})
    return off_road, collisions

def get_frame_info(sample_token: str, nusc: NuScenes, nusc_map: NuScenesMapExplorer, patch_radius: float = PATCH_RADIUS) -> dict:
    sample_record = nusc.get("sample", sample_token)
    if "LIDAR_TOP" in sample_record["data"]:
        sd_token = sample_record["data"]["LIDAR_TOP"]
    else:
        sd_token = list(sample_record["data"].values())[0]
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
    pts: list[tuple[float, float]] = []
    while tok:
        s = nusc.get("sample", tok)
        if "LIDAR_TOP" in s["data"]:
            sd_token = s["data"]["LIDAR_TOP"]
        else:
            sd_token = list(s["data"].values())[0]
        sd = nusc.get("sample_data", sd_token)
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        x, y, _ = pose["translation"]
        pts.append((float(x), float(y)))
        tok = s.get("next", None)
    return LineString(pts)

def nearest_point_and_tangent(line: LineString, x: float, y: float) -> tuple[Point, np.ndarray]:
    """Return nearest point on line and approximate tangent direction vector at that point."""
    proj = float(line.project(Point(x, y)))
    p = line.interpolate(proj)
    # Approximate tangent by sampling small deltas along the curve length
    dl = max(0.1, min(1.0, line.length * 0.001))
    p_prev = line.interpolate(max(0.0, proj - dl))
    p_next = line.interpolate(min(line.length, proj + dl))
    vec = np.array([float(p_next.x - p_prev.x), float(p_next.y - p_prev.y)], dtype=np.float32)
    nrm = np.linalg.norm(vec)
    if nrm < 1e-6:
        vec = np.array([1.0, 0.0], dtype=np.float32)
    else:
        vec = vec / nrm
    return p, vec

def compute_pos_heading_deviation(info: dict, centerline: LineString, dmax: float = 2.0) -> tuple[float, float]:
    """Compute positional deviation (m) and heading deviation (deg) vs expert centerline."""
    x, y, _ = info["ego_translation"]
    yaw = float(info.get("ego_yaw", 0.0))
    p, tan = nearest_point_and_tangent(centerline, x, y)
    pos_dev = float(np.linalg.norm(np.array([x - float(p.x), y - float(p.y)], dtype=np.float32)))
    lane_yaw = float(np.arctan2(tan[1], tan[0]))
    yaw_diff = float(np.degrees(np.arctan2(np.sin(yaw - lane_yaw), np.cos(yaw - lane_yaw))))
    return pos_dev, abs(yaw_diff)

def aggregate_rewards(info: dict, pos_dev_m: float, yaw_err_deg: float,
                      *, dmax: float = 2.0, psi_max_deg: float = 30.0,
                      w_dynamic: float = 5.0, w_static: float = 5.0,
                      w_pos: float = 2.0, w_heading: float = 1.0) -> dict:
    """Compute reward components and termination trigger flags."""
    has_dyn = len(info.get("dynamic_collisions", [])) > 0
    has_sta = len(info.get("static_collisions", [])) > 0
    off_road = bool(info.get("off_road", False))
    rdc = -w_dynamic if has_dyn else 0.0
    rsc = -w_static if has_sta else 0.0
    rpd = -w_pos * max(0.0, pos_dev_m - dmax)
    rhd = -w_heading * max(0.0, yaw_err_deg - psi_max_deg)
    reward = float(rdc + rsc + rpd + rhd)
    # Immediate termination on any violation
    terminate = bool(has_dyn or has_sta or (pos_dev_m > dmax) or (yaw_err_deg > psi_max_deg) or off_road)
    return {
        "rdc": float(rdc),
        "rsc": float(rsc),
        "rpd": float(rpd),
        "rhd": float(rhd),
        "reward": float(reward),
        "terminate": bool(terminate),
        "pos_dev_m": float(pos_dev_m),
        "yaw_err_deg": float(yaw_err_deg),
    }


# -------------------------------
# 动态 agent 相关函数
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
        l, w, _ = ann["size"]
        yaw = quat_to_yaw(ann["rotation"]) if "rotation" in ann else 0.0
        poly = oriented_box(x, y, l, w, yaw)
        if ego_poly.intersects(poly):
            collisions.append({"token": ann_token, "category": ann["category_name"]})
    return collisions


def render_frame_with_agents(frame_info: dict, nusc_map: NuScenesMapExplorer, nusc: NuScenes, patch_radius: float = 20.0, out_path: str = "collision_patch_agents.png"):
    ego_x, ego_y, _ = frame_info["ego_translation"]
    x_min, y_min = ego_x - patch_radius, ego_y - patch_radius
    x_max, y_max = ego_x + patch_radius, ego_y + patch_radius
    patch_box_geom = box(x_min, y_min, x_max, y_max)
    fig, ax = plt.subplots(figsize=(8, 8))
    layer_colors = {
        "drivable_area": "#a6cee3",
        "lane": "#1f78b4",
        "walkway": "#b2df8a",
        "road_block": "#33a02c",
        "carpark_area": "#fb9a99",
    }
    for layer in ["drivable_area", "lane", "walkway", "road_block", "carpark_area"]:
        try:
            for record in getattr(nusc_map.map_api, layer):
                polys = [extract_polygon(nusc_map.map_api, t) for t in record["polygon_tokens"]] if layer == "drivable_area" else [extract_polygon(nusc_map.map_api, record["polygon_token"])]
                for p in polys:
                    geom = p.intersection(patch_box_geom)
                    if geom.is_empty:
                        continue
                    if isinstance(geom, Polygon):
                        x, y = geom.exterior.xy
                        ax.fill(x, y, color=layer_colors.get(layer, "gray"), alpha=0.35)
                    elif isinstance(geom, MultiPolygon):
                        for sub in geom:
                            if not sub.is_empty:
                                x, y = sub.exterior.xy
                                ax.fill(x, y, color=layer_colors.get(layer, "gray"), alpha=0.35)
        except Exception:
            pass
    ego_yaw = frame_info.get("ego_yaw", 0.0)
    ego_poly = oriented_box(ego_x, ego_y, 4.2, 1.9, ego_yaw)
    ex, ey = ego_poly.exterior.xy
    ax.fill(ex, ey, color="blue", alpha=0.6)
    for col in frame_info.get("static_collisions", []):
        layer = col.get("layer")
        try:
            for record in getattr(nusc_map.map_api, layer):
                polys = [extract_polygon(nusc_map.map_api, t) for t in record["polygon_tokens"]] if layer == "drivable_area" else [extract_polygon(nusc_map.map_api, record["polygon_token"])]
                for p in polys:
                    geom = p.intersection(patch_box_geom)
                    if geom.is_empty:
                        continue
                    if isinstance(geom, Polygon):
                        x, y = geom.exterior.xy
                        ax.fill(x, y, color="red", alpha=0.4)
                    elif isinstance(geom, MultiPolygon):
                        for sub in geom:
                            if not sub.is_empty:
                                x, y = sub.exterior.xy
                                ax.fill(x, y, color="red", alpha=0.4)
        except Exception:
            pass
    dyn_cols = check_dynamic_collision(frame_info, nusc)
    sample_rec = nusc.get("sample", frame_info["sample_token"])
    for ann_token in sample_rec["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        x, y, _ = ann["translation"]
        l, w, _ = ann["size"]
        yaw = quat_to_yaw(ann["rotation"]) if "rotation" in ann else 0.0
        poly = oriented_box(x, y, l, w, yaw)
        if ann["category_name"].startswith("vehicle"):
            color = "orange"
        elif ann["category_name"].startswith("human"):
            color = "purple"
        else:
            continue
        px, py = poly.exterior.xy
        ax.fill(px, py, color=color, alpha=0.6)
        if any(c["token"] == ann_token for c in dyn_cols):
            ax.plot(px, py, color="red", linewidth=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    title = f"sample={frame_info['sample_token']} | off_road={frame_info.get('off_road', False)}"
    ax.set_title(title)
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in layer_colors.items()]
    legend_patches += [
        mpatches.Patch(color="blue", label="ego"),
        mpatches.Patch(color="orange", label="vehicle"),
        mpatches.Patch(color="purple", label="pedestrian"),
        mpatches.Patch(edgecolor="red", facecolor="none", label="collision")
    ]
    ax.legend(handles=legend_patches, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved to {out_path}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=DEFAULT_SAMPLE_TOKEN)
    parser.add_argument("--root", type=str, default=DATA_ROOT)
    parser.add_argument("--version", type=str, default=VERSION)
    parser.add_argument("--radius", type=float, default=PATCH_RADIUS)
    parser.add_argument("--out", type=str, default="collision_patch.png")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.root, verbose=False)
    sample_token = args.token
    sample_record = nusc.get("sample", sample_token)
    scene_record = nusc.get("scene", sample_record["scene_token"])
    log_record = nusc.get("log", scene_record["log_token"])
    log_location = log_record["location"]
    nusc_map = NuScenesMapExplorer(NuScenesMap(dataroot=args.root, map_name=log_location))
    info = get_frame_info(sample_token, nusc, nusc_map, patch_radius=args.radius)
    info["dynamic_collisions"] = check_dynamic_collision(info, nusc)
    # Build expert centerline from scene ego poses
    centerline = build_scene_ego_path(nusc, sample_token)
    pos_dev_m, yaw_err_deg = compute_pos_heading_deviation(info, centerline, dmax=2.0)
    rewards = aggregate_rewards(info, pos_dev_m, yaw_err_deg, dmax=2.0, psi_max_deg=30.0)
    info.update({
        "pos_dev_m": pos_dev_m,
        "yaw_err_deg": yaw_err_deg,
        "reward_components": {k: rewards[k] for k in ["rdc","rsc","rpd","rhd"]},
        "reward_total": rewards["reward"],
        "terminate": rewards["terminate"],
    })
    print(json.dumps(info, indent=2))
    # Optional: overlay centerline
    try:
        render_frame_with_agents(info, nusc_map, nusc, patch_radius=args.radius, out_path=args.out)
    except Exception:
        render_frame_with_agents(info, nusc_map, nusc, patch_radius=args.radius, out_path=args.out)
