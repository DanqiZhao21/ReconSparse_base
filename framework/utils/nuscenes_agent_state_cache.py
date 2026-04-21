from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


_DEFAULT_PROCESSED_ROOT = Path(__file__).resolve().parents[2] / "assets" / "nuscenes" / "processed_10Hz" / "trainval"
_DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[2] / "assets" / "nus" / "data"


def _scene_dir_from_root(root: str | Path, scene_id: int) -> Path:
    return Path(root) / f"{int(scene_id):03d}"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _yaw_from_obj_to_world(mat_like: Any) -> float:
    mat = np.asarray(mat_like, dtype=np.float64)
    if mat.shape != (4, 4):
        raise RuntimeError(f"Expected obj_to_world matrix with shape (4, 4), got {mat.shape}")
    return float(math.atan2(float(mat[1, 0]), float(mat[0, 0])))


def _center_xy_from_obj_to_world(mat_like: Any) -> np.ndarray:
    mat = np.asarray(mat_like, dtype=np.float64)
    if mat.shape != (4, 4):
        raise RuntimeError(f"Expected obj_to_world matrix with shape (4, 4), got {mat.shape}")
    return np.asarray([float(mat[0, 3]), float(mat[1, 3])], dtype=np.float64)


def _differentiate(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if arr.shape[0] <= 1:
        return np.zeros_like(arr, dtype=np.float64)
    if arr.ndim == 1:
        out = np.gradient(arr, times)
        return np.asarray(out, dtype=np.float64)
    cols = [np.gradient(arr[:, idx], times) for idx in range(int(arr.shape[1]))]
    return np.stack(cols, axis=1).astype(np.float64, copy=False)


def _build_agent_series(instance: dict[str, Any], *, fps: float) -> list[tuple[int, dict[str, Any]]]:
    frame_annotations = dict(instance.get("frame_annotations", {}))
    frame_idx = np.asarray(frame_annotations.get("frame_idx", []), dtype=np.int64).reshape(-1)
    obj_to_world = list(frame_annotations.get("obj_to_world", []))
    box_size = list(frame_annotations.get("box_size", []))
    if int(frame_idx.shape[0]) <= 0 or len(obj_to_world) != int(frame_idx.shape[0]) or len(box_size) != int(frame_idx.shape[0]):
        return []
    order = np.argsort(frame_idx)
    frame_idx = frame_idx[order]
    obj_to_world = [obj_to_world[int(idx)] for idx in order]
    box_size = [box_size[int(idx)] for idx in order]

    times_s = frame_idx.astype(np.float64) / max(1.0e-6, float(fps))
    centers = np.stack([_center_xy_from_obj_to_world(item) for item in obj_to_world], axis=0).astype(np.float64, copy=False)
    yaw = np.unwrap(np.asarray([_yaw_from_obj_to_world(item) for item in obj_to_world], dtype=np.float64))
    velocity_xy = _differentiate(centers, times_s)
    yaw_rate = _differentiate(yaw, times_s)

    category = str(instance.get("class_name", "unknown"))
    instance_token = str(instance.get("id", ""))
    out: list[tuple[int, dict[str, Any]]] = []
    for idx in range(int(frame_idx.shape[0])):
        size_arr = np.asarray(box_size[idx], dtype=np.float64).reshape(-1)
        length_m = float(abs(size_arr[0])) if size_arr.size > 0 else 1.0
        width_m = float(abs(size_arr[1])) if size_arr.size > 1 else 1.0
        vel_xy = np.asarray(velocity_xy[idx], dtype=np.float64).reshape(2)
        out.append(
            (
                int(frame_idx[idx]),
                {
                    "instance_token": instance_token,
                    "category": category,
                    "center_xy": [float(centers[idx, 0]), float(centers[idx, 1])],
                    "yaw_rad": float(np.arctan2(np.sin(yaw[idx]), np.cos(yaw[idx]))),
                    "yaw_rate_rps": float(yaw_rate[idx]),
                    "velocity_xy": [float(vel_xy[0]), float(vel_xy[1])],
                    "speed_mps": float(np.linalg.norm(vel_xy)),
                    "length_m": length_m,
                    "width_m": width_m,
                },
            )
        )
    return out


def build_scene_agent_state_cache_from_processed_scene(
    *,
    scene_id: int,
    processed_scene_dir: str | Path,
    out_root: str | Path | None = None,
    fps: float = 10.0,
) -> dict[int, dict[str, Any]]:
    scene_dir = Path(processed_scene_dir)
    instances_dir = scene_dir / "instances" if (scene_dir / "instances").exists() else scene_dir
    instances_info_path = instances_dir / "instances_info.json"
    if not instances_info_path.exists():
        raise FileNotFoundError(f"instances_info.json not found under {instances_dir}")

    instances_info = dict(_load_json(instances_info_path))
    frame_map: dict[int, dict[str, Any]] = {}
    for instance in instances_info.values():
        for frame_idx, agent_state in _build_agent_series(dict(instance), fps=float(fps)):
            frame_map.setdefault(int(frame_idx), {"agents": []})
            frame_map[int(frame_idx)]["agents"].append(agent_state)

    if out_root is not None:
        out_dir = _scene_dir_from_root(out_root, int(scene_id))
        out_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "meta": {
                "source": "processed_10Hz",
                "fps": float(fps),
                "coordinate_frame": "world",
            }
        }
        for frame_idx in sorted(frame_map.keys()):
            payload[str(int(frame_idx))] = frame_map[frame_idx]
        (out_dir / "agent_state_cache.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return frame_map


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-scene agent state cache for NuScenes EA scoring.")
    parser.add_argument("--scene", type=int, required=True, help="Scene id, e.g. 1 or 146")
    parser.add_argument("--processed-root", type=str, default=str(_DEFAULT_PROCESSED_ROOT), help="processed_10Hz split root")
    parser.add_argument("--out-root", type=str, default=str(_DEFAULT_OUT_ROOT), help="output assets/nus/data root")
    parser.add_argument("--fps", type=float, default=10.0, help="frame rate for processed scene annotations")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache = build_scene_agent_state_cache_from_processed_scene(
        scene_id=int(args.scene),
        processed_scene_dir=_scene_dir_from_root(args.processed_root, int(args.scene)),
        out_root=args.out_root,
        fps=float(args.fps),
    )
    print(f"Built agent state cache for scene {int(args.scene):03d} with {len(cache)} frames")


if __name__ == "__main__":
    main()
