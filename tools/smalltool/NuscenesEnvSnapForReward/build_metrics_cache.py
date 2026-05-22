#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute and save per-keyframe ENVIRONMENT SNAPSHOTS for a scene into assets/nus/data/{scene}/env_cache.json.
Snapshots contain: drivable polygons (ROI裁剪), lane/lane_connector中心线折线（带方向）, 静态/动态参与者在地面的OBB多边形。

训练时输入自车(x,y,yaw)，直接用快照计算 off-road / 方向一致性 / 静/动碰撞，无需再加载NuScenes。

Usage:
    python tools/smalltool/NuscenesEnvSnapForReward/build_metrics_cache.py --scene 0 --root /OpenDataset/nuscenes/nuscenes/v1.0-trainval --version v1.0-trainval
    python tools/smalltool/NuscenesEnvSnapForReward/build_metrics_cache.py --scene 1
"""

import argparse
import os
import sys
from typing import Dict, Any
import numpy as np

# Ensure repository root is on PYTHONPATH so `reconsimulator` can be imported
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from reconsimulator.envs.metrics import _load_token_frame_map, quat_to_yaw
from reconsimulator.envs.metrics_cache import save_scene_env_cache
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
try:
    from tqdm import tqdm  # optional progress bar
except Exception:
    tqdm = None


def _resolve_sample_token(nusc: NuScenes, token: str) -> str:
    """Return a valid sample token. Accepts either a sample token or a sample_data token."""
    try:
        _ = nusc.get("sample", token)
        return token
    except Exception:
        try:
            sd = nusc.get("sample_data", token)
            return sd["sample_token"]
        except Exception:
            raise


def _get_ego_pose(nusc: NuScenes, sample_token: str) -> Dict[str, Any]:
    sample_record = nusc.get("sample", sample_token)
    sd_token = sample_record["data"].get("LIDAR_TOP", list(sample_record["data"].values())[0])
    sd = nusc.get("sample_data", sd_token)
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    x, y, _ = pose["translation"]
    yaw = quat_to_yaw(pose["rotation"]) if "rotation" in pose else 0.0
    return {"x": float(x), "y": float(y), "yaw": float(yaw)}


def _polygon_to_coords(poly) -> list:
    """Convert shapely Polygon to list of [x,y] coordinates (exterior only)."""
    return [[float(p[0]), float(p[1])] for p in list(poly.exterior.coords)]


def _ann_to_ground_poly(nusc: NuScenes, ann_token: str) -> tuple[list, str]:
    ann = nusc.get("sample_annotation", ann_token)
    cat = ann["category_name"]
    x, y, _ = ann["translation"]
    # nuScenes sample_annotation.size is [width, length, height].
    w, l, _ = ann["size"]
    yaw = quat_to_yaw(ann["rotation"]) if "rotation" in ann else 0.0
    # Construct oriented rectangle polygon and export coords
    # Avoid importing shapely rotation here; use quick 4-corner method on ground.
    cx, cy = float(x), float(y)
    hl, hw = float(l) * 0.5, float(w) * 0.5
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    corners = [
        ( cx + c*hl - s*hw, cy + s*hl + c*hw ),
        ( cx - c*hl - s*hw, cy - s*hl + c*hw ),
        ( cx - c*hl + s*hw, cy - s*hl - c*hw ),
        ( cx + c*hl + s*hw, cy + s*hl - c*hw ),
        ( cx + c*hl - s*hw, cy + s*hl + c*hw ),
    ]
    return [[float(px), float(py)] for px, py in corners], cat


def build_scene_env_cache(scene_id: int, dataroot: str, version: str, roi_radius: float = 80.0, centerline_res: float = 1.0, verbose: bool = False) -> Dict[int, Dict[str, Any]]:
    # _load_token_frame_map returns mapping: frame_idx (int) -> sample_token (str)
    frame_to_token: Dict[int, str] = _load_token_frame_map(scene_id)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    if len(frame_to_token) == 0:
        return {}

    # Pick first token to determine map location
    first_sample_token = _resolve_sample_token(nusc, next(iter(frame_to_token.values())))
    sample_record = nusc.get("sample", first_sample_token)
    scene_record = nusc.get("scene", sample_record["scene_token"])
    log_record = nusc.get("log", scene_record["log_token"]) 
    nusc_map = NuScenesMapExplorer(NuScenesMap(dataroot=dataroot, map_name=log_record["location"]))

    out: Dict[int, Dict[str, Any]] = {}
    iterator = frame_to_token.items()  # (frame_idx, sample_token)
    if tqdm is not None:
        iterator = tqdm(list(iterator), desc=f"Scene {scene_id} env", unit="frame")
    for frame_idx, sample_token in iterator:
        # Robust per-field construction; don't drop whole frame on partial failures.
        entry: Dict[str, Any] = {}
        try:
            sample_token = _resolve_sample_token(nusc, sample_token)
            entry["sample_token"] = sample_token
        except Exception as e:
            if verbose:
                print(f"[warn] frame {frame_idx}: resolve token failed: {e}")
            out[int(frame_idx)] = entry
            continue

        try:
            pose = _get_ego_pose(nusc, sample_token)
            entry["ego_pose"] = pose
            x, y = pose["x"], pose["y"]
        except Exception as e:
            if verbose:
                print(f"[warn] frame {frame_idx}: ego pose failed: {e}")
            # still proceed but map queries may fail without x,y
            x, y = None, None

        # Map layers within ROI
        try:
            if x is not None and y is not None:
                layers = ["drivable_area", "lane", "lane_connector"]
                nearby = nusc_map.map_api.get_records_in_radius(x, y, roi_radius, layers, mode='intersect')
            else:
                nearby = {"drivable_area": [], "lane": [], "lane_connector": []}
        except Exception as e:
            if verbose:
                print(f"[warn] frame {frame_idx}: get_records_in_radius failed: {e}")
            nearby = {"drivable_area": [], "lane": [], "lane_connector": []}

        # Drivable polygons
        drivable_polygons: list = []
        try:
            for tok in nearby.get("drivable_area", []) or []:
                rec = nusc_map.map_api.get("drivable_area", tok)
                for ptk in rec.get("polygon_tokens", []) or []:
                    poly = nusc_map.map_api.extract_polygon(ptk)
                    drivable_polygons.append(_polygon_to_coords(poly))
        except Exception as e:
            if verbose:
                print(f"[warn] frame {frame_idx}: drivable extraction failed: {e}")
        entry["drivable_polygons"] = drivable_polygons

        # Lane/Lane-connector centerlines
        lanes_centerlines: list = []
        try:
            lane_tokens = (nearby.get("lane", []) or []) + (nearby.get("lane_connector", []) or [])
            lane_dict = nusc_map.map_api.discretize_lanes(lane_tokens, centerline_res)
            for _, pts in lane_dict.items():
                if not pts:
                    continue
                lanes_centerlines.append([[float(px), float(py)] for px, py, _ in pts])
        except Exception as e:
            if verbose:
                print(f"[warn] frame {frame_idx}: centerlines failed: {e}")
        entry["lanes_centerlines"] = lanes_centerlines

        # Sample annotations → split static/dynamic
        static_objects: list = []
        dynamic_objects: list = []
        try:
            srec = nusc.get("sample", sample_token)
            for ann_token in srec.get("anns", []) or []:
                try:
                    poly_coords, cat = _ann_to_ground_poly(nusc, ann_token)
                except Exception:
                    if verbose:
                        print(f"[warn] frame {frame_idx}: ann {ann_token} poly failed")
                    continue
                entry_obj = {"category": cat, "poly": poly_coords}
                if cat.startswith("vehicle") or cat.startswith("human"):
                    entry_obj.update({"token": ann_token})
                    dynamic_objects.append(entry_obj)
                else:
                    static_objects.append(entry_obj)
        except Exception as e:
            if verbose:
                print(f"[warn] frame {frame_idx}: annotations failed: {e}")
        entry["static_objects"] = static_objects
        entry["dynamic_objects"] = dynamic_objects

        out[int(frame_idx)] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, required=True, help="scene id, e.g., 0 or 1")
    ap.add_argument("--root", type=str, default="/OpenDataset/nuscenes/nuscenes/v1.0-trainval")
    ap.add_argument("--version", type=str, default="v1.0-trainval")
    ap.add_argument("--roi", type=float, default=80.0, help="ROI radius in meters for map layers")
    ap.add_argument("--cl_res", type=float, default=1.0, help="Centerline discretization resolution (m)")
    ap.add_argument("--verbose", action="store_true", help="Print warnings for per-frame extraction failures")
    args = ap.parse_args()

    cache = build_scene_env_cache(args.scene, args.root, args.version, roi_radius=args.roi, centerline_res=args.cl_res, verbose=bool(args.verbose))
    if len(cache) == 0:
        print(f"No entries for scene {args.scene}")
        return
    meta = {"version": args.version, "dataroot": args.root, "roi_radius": float(args.roi), "centerline_res": float(args.cl_res)}
    path = save_scene_env_cache(args.scene, cache, meta=meta)
    print(f"Saved env cache for scene {args.scene} to: {path} ({len(cache)} steps)")


if __name__ == "__main__":
    main()
