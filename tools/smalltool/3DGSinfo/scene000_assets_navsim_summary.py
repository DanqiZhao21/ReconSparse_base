#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize and print extractable information for scene 000 from Gaussian env (assets/nus)
vs Navsim stack (DiffusionDriveV2/navsim). Designed for quick, clear comparison.

Usage:
  python tools/smalltool/scene000_assets_navsim_summary.py [--scene 000]

Outputs:
  - Prints structured summaries to stdout
  - Optionally writes a compact JSON summary to outputs/summary_scene000.json
"""
import os
import sys
import json
import re
from collections import Counter, defaultdict

# Optional deps
try:
    import numpy as np
except Exception:
    np = None

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS_ROOT = os.path.join(REPO_ROOT, "assets")
NUS_ROOT = os.path.join(ASSETS_ROOT, "nus")
SCENE_ID_DEFAULT = "000"


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_yaml(path: str):
    if yaml is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _read_npy(path: str):
    if np is None:
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _matrix4x4_from_txt(path: str):
    text = _read_text(path).strip()
    rows = [list(map(float, l.strip().split())) for l in text.splitlines() if l.strip()]
    if len(rows) != 4 or any(len(r) != 4 for r in rows):
        return None
    return rows


def summarize_gaussian_env(scene_id: str) -> dict:
    base = os.path.join(NUS_ROOT, "data", scene_id)
    out = {
        "scene_id": scene_id,
        "paths": {
            "ego_pose": os.path.join(base, "ego_pose"),
            "intrinsics": os.path.join(base, "intrinsics"),
            "extrinsics": os.path.join(base, "extrinsics"),
            "cam2ego": os.path.join(base, "cam2ego"),
            "lidar_pose": os.path.join(base, "lidar_pose"),
            "instances": os.path.join(base, "instances"),
            "frame2token": os.path.join(NUS_ROOT, "information", "frame2token", f"{scene_id}.json"),
            "token_frame_map": os.path.join(base, "token_frame_map.json"),
            "anchors": os.path.join(NUS_ROOT, "anchor"),
            "3dgs_ckpt": os.path.join(base, "3DGS_without_prior", "checkpoint_final.pth"),
            "gaussian_cfg": os.path.join(NUS_ROOT, "others", "config.yaml"),
        },
        "exists": {},
        "counts": {},
        "intrinsics": {},
        "extrinsics": {
            "sample": None,
            "count": 0,
        },
        "ego_pose": {
            "frames": 0,
            "first": None,
        },
        "instances": {
            "num_instances": 0,
            "class_distribution": {},
            "frames_per_instance_stats": {},
        },
        "anchors": {
            "traj_shape": None,
            "yaw_shape": None,
            "mask_shape": None,
        },
        "gaussian_cfg_keys": [],
    }

    # existence checks
    for k, p in out["paths"].items():
        out["exists"][k] = os.path.exists(p)

    # ego pose
    ego_pose_dir = out["paths"]["ego_pose"]
    if os.path.isdir(ego_pose_dir):
        files = sorted([f for f in os.listdir(ego_pose_dir) if f.endswith(".txt")])
        out["counts"]["ego_pose_txt"] = len(files)
        out["ego_pose"]["frames"] = len(files)
        if files:
            m = _matrix4x4_from_txt(os.path.join(ego_pose_dir, files[0]))
            out["ego_pose"]["first"] = m

    # intrinsics
    intr_dir = out["paths"]["intrinsics"]
    if os.path.isdir(intr_dir):
        files = sorted([f for f in os.listdir(intr_dir) if f.endswith(".txt")])
        out["counts"]["intrinsics_txt"] = len(files)
        for f in files:
            arr = _read_text(os.path.join(intr_dir, f)).strip().splitlines()
            vals = [float(x) for x in arr if x.strip()] if arr else []
            out["intrinsics"][f] = {
                "num_values": len(vals),
                "sample": vals[:5],
            }

    # extrinsics
    ext_dir = out["paths"]["extrinsics"]
    if os.path.isdir(ext_dir):
        files = sorted([f for f in os.listdir(ext_dir) if f.endswith(".txt")])
        out["extrinsics"]["count"] = len(files)
        if files:
            out["extrinsics"]["sample"] = _matrix4x4_from_txt(os.path.join(ext_dir, files[0]))

    # instances
    inst_dir = out["paths"]["instances"]
    frame_inst = _read_json(os.path.join(inst_dir, "frame_instances.json")) if os.path.isdir(inst_dir) else None
    inst_info = _read_json(os.path.join(inst_dir, "instances_info.json")) if os.path.isdir(inst_dir) else None
    if isinstance(inst_info, dict):
        out["instances"]["num_instances"] = len(inst_info)
        classes = Counter()
        frames_len = []
        for k, v in inst_info.items():
            cname = v.get("class_name", "unknown")
            classes[cname] += 1
            fa = v.get("frame_annotations", {})
            fidx = fa.get("frame_idx", [])
            frames_len.append(len(fidx) if isinstance(fidx, list) else 0)
        out["instances"]["class_distribution"] = dict(classes.most_common(15))
        if frames_len:
            frames_len.sort()
            out["instances"]["frames_per_instance_stats"] = {
                "min": frames_len[0],
                "median": frames_len[len(frames_len)//2],
                "max": frames_len[-1],
            }

    # anchors
    anc_dir = out["paths"]["anchors"]
    if os.path.isdir(anc_dir):
        traj = _read_npy(os.path.join(anc_dir, "traj_anchor_05s_3721.npy"))
        yaw = _read_npy(os.path.join(anc_dir, "traj_anchor_05s_3721_yaw.npy"))
        mask = _read_npy(os.path.join(anc_dir, "traj_anchor_05s_3721_mask.npy"))
        out["anchors"]["traj_shape"] = None if traj is None else tuple(traj.shape)
        out["anchors"]["yaw_shape"] = None if yaw is None else tuple(yaw.shape)
        out["anchors"]["mask_shape"] = None if mask is None else tuple(mask.shape)

    # gaussian cfg
    cfg = _read_yaml(out["paths"]["gaussian_cfg"]) if out["exists"].get("gaussian_cfg") else None
    if isinstance(cfg, dict):
        keys = []
        for topk in ["gaussian_optim_general_cfg", "gaussian_ctrl_general_cfg", "model", "render", "data"]:
            if topk in cfg:
                keys.append(topk)
        out["gaussian_cfg_keys"] = keys

    return out


def summarize_navsim_stack() -> dict:
    navsim_root = os.path.join(REPO_ROOT, "DiffusionDriveV2", "navsim")
    pdm_path = os.path.join(navsim_root, "evaluate", "pdm_score.py")
    txt = _read_text(pdm_path)

    # Extract metric names used in PDM scoring via simple regex on enums
    weighted = sorted(set(re.findall(r"WeightedMetricIndex\.(\w+)", txt)))
    multi = sorted(set(re.findall(r"MultiMetricIndex\.(\w+)", txt)))

    # Check BEV semantic map head presence in model file
    model_rl_path = os.path.join(navsim_root, "agents", "diffusiondrivev2", "diffusiondrivev2_model_rl.py")
    mtxt = _read_text(model_rl_path)
    has_bev_semantic_head = ("_bev_semantic_head" in mtxt)
    has_cross_bev_attention = ("GridSampleCrossBEVAttention" in mtxt)

    return {
        "paths": {
            "pdm_score": pdm_path,
            "model_rl": model_rl_path,
        },
        "exists": {
            "pdm_score": os.path.exists(pdm_path),
            "model_rl": os.path.exists(model_rl_path),
        },
        "pdm_metrics": {
            "weighted": weighted,
            "multi": multi,
        },
        "bev_semantics": {
            "has_bev_semantic_head": has_bev_semantic_head,
            "has_cross_bev_attention": has_cross_bev_attention,
        },
        "notes": {
            "metric_cache_requires_download": True,
            "config_entry_for_maps": "see DiffusionDriveV2/docs/download/download_maps.sh",
        },
    }


def print_summary(gauss: dict, navsim: dict) -> None:
    print("\n=== Gaussian Env (assets/nus) — Scene {} ===".format(gauss.get("scene_id")))
    print("exists:")
    for k, v in gauss.get("exists", {}).items():
        print(f"  - {k}: {v}")

    print("\nego_pose:")
    print("  frames:", gauss.get("ego_pose", {}).get("frames"))
    first = gauss.get("ego_pose", {}).get("first")
    if first:
        print("  first_pose_4x4:")
        for row in first:
            print("   ", " ".join(f"{x:.6f}" for x in row))

    print("\nintrinsics:")
    intr = gauss.get("intrinsics", {})
    for f, meta in list(intr.items())[:6]:
        print(f"  - {f}: num_values={meta.get('num_values')} sample={meta.get('sample')}")

    print("\nextrinsics:")
    print("  count:", gauss.get("extrinsics", {}).get("count"))
    sample = gauss.get("extrinsics", {}).get("sample")
    if sample:
        print("  sample_4x4:")
        for row in sample:
            print("   ", " ".join(f"{x:.6f}" for x in row))

    print("\ninstances:")
    inst = gauss.get("instances", {})
    print("  num_instances:", inst.get("num_instances"))
    print("  class_distribution(top15):")
    for cname, cnt in inst.get("class_distribution", {}).items():
        print(f"    {cname}: {cnt}")
    print("  frames_per_instance_stats:", inst.get("frames_per_instance_stats"))

    print("\nanchors:")
    anc = gauss.get("anchors", {})
    print("  traj_shape:", anc.get("traj_shape"))
    print("  yaw_shape:", anc.get("yaw_shape"))
    print("  mask_shape:", anc.get("mask_shape"))

    print("\ngaussian_cfg_keys:", gauss.get("gaussian_cfg_keys"))
    print("  3DGS checkpoint exists:", gauss.get("exists", {}).get("3dgs_ckpt"))

    print("\n=== Navsim Stack (DiffusionDriveV2/navsim) ===")
    print("exists:")
    for k, v in navsim.get("exists", {}).items():
        print(f"  - {k}: {v}")
    print("\npdm_metrics:")
    print("  weighted:", ", ".join(navsim.get("pdm_metrics", {}).get("weighted", [])))
    print("  multi:", ", ".join(navsim.get("pdm_metrics", {}).get("multi", [])))
    bev = navsim.get("bev_semantics", {})
    print("\nbev_semantics:")
    print("  has_bev_semantic_head:", bev.get("has_bev_semantic_head"))
    print("  has_cross_bev_attention:", bev.get("has_cross_bev_attention"))
    print("\nnotes:")
    for k, v in navsim.get("notes", {}).items():
        print(f"  - {k}: {v}")

    # Compact JSON summary
    out_dir = os.path.join(REPO_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"summary_scene{gauss.get('scene_id')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"gaussian": gauss, "navsim": navsim}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved compact JSON summary to: {out_path}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Summarize Gaussian env vs Navsim (scene 000)")
    parser.add_argument("--scene", type=str, default=SCENE_ID_DEFAULT, help="Scene id under assets/nus/data")
    args = parser.parse_args(argv)

    gauss = summarize_gaussian_env(args.scene)
    navsim = summarize_navsim_stack()
    print_summary(gauss, navsim)


if __name__ == "__main__":
    main()
