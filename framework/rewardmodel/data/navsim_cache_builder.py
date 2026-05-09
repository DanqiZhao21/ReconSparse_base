from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from framework.rewardmodel.supervision.vocabulary import filter_trajectory_vocabulary
from framework.rewardmodel.supervision.teacher_adapter import map_pdm_metric_names, stack_temporal_metric_targets


DEFAULT_NAVSIM_CAMERA_NAMES = ("cam_l0", "cam_f0", "cam_r0")


def load_candidate_trajectories(path: str | Path, *, key: str = "trajectories") -> np.ndarray:
    candidate_path = Path(path)
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate trajectory file not found: {candidate_path}")
    if candidate_path.suffix == ".npy":
        arr = np.load(candidate_path)
    elif candidate_path.suffix == ".npz":
        data = np.load(candidate_path)
        if key in data:
            arr = data[key]
        elif len(data.files) == 1:
            arr = data[data.files[0]]
        else:
            raise KeyError(f"Candidate npz must contain key {key!r}; available keys={data.files}")
    else:
        raise ValueError(f"Unsupported candidate file extension: {candidate_path.suffix}")
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 3 or int(out.shape[-1]) < 3:
        raise ValueError(f"Expected candidate trajectories [G,T,3+], got {tuple(out.shape)}")
    return out[..., :3]


def build_scene_specific_candidates(
    *,
    scene: Any,
    vocabulary: np.ndarray,
    max_longitudinal_error_m: float = 10.0,
    max_lateral_error_m: float = 5.0,
    max_heading_error_rad: float = np.deg2rad(20.0),
    max_candidates: int = 256,
) -> np.ndarray:
    future = scene.get_future_trajectory()
    gt_trajectory = np.asarray(getattr(future, "poses"), dtype=np.float32)
    if gt_trajectory.ndim != 2 or int(gt_trajectory.shape[-1]) < 3:
        raise ValueError(f"Expected scene future trajectory poses [T,3], got {tuple(gt_trajectory.shape)}")
    filtered = filter_trajectory_vocabulary(
        vocabulary=np.asarray(vocabulary, dtype=np.float32),
        gt_trajectory=gt_trajectory[:, :3],
        max_longitudinal_error_m=float(max_longitudinal_error_m),
        max_lateral_error_m=float(max_lateral_error_m),
        max_heading_error_rad=float(max_heading_error_rad),
        max_samples=int(max_candidates),
    )
    if int(filtered.shape[0]) <= 0:
        raise ValueError("Scene-specific vocabulary filtering returned zero candidates")
    return filtered


def build_vocabulary_from_gt_trajectories(
    gt_trajectories: Sequence[np.ndarray],
    *,
    max_vocabulary_size: int | None = None,
) -> np.ndarray:
    items = [np.asarray(traj, dtype=np.float32) for traj in gt_trajectories]
    if len(items) <= 0:
        raise ValueError("gt_trajectories must not be empty")
    first_shape = tuple(items[0].shape)
    for traj in items:
        if tuple(traj.shape) != first_shape:
            raise ValueError("All GT trajectories must have the same shape to form a dense vocabulary")
    unique_items: list[np.ndarray] = []
    seen_keys: set[bytes] = set()
    for traj in items:
        key = np.asarray(np.round(traj, decimals=4), dtype=np.float32).tobytes()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_items.append(traj)
    stacked = np.stack(unique_items, axis=0)
    if max_vocabulary_size is None or int(stacked.shape[0]) <= int(max_vocabulary_size):
        return stacked
    return stacked[: int(max_vocabulary_size)]


def build_reward_sample_from_raw(
    *,
    image_paths: Sequence[str | Path],
    ego_states: Any,
    candidate_trajectories: Any,
    targets: Any,
    token: str,
    valid_mask: Any | None = None,
) -> dict[str, Any]:
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    if valid_mask is None:
        valid_mask_tensor = torch.ones_like(target_tensor, dtype=torch.bool)
    else:
        valid_mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool)
    return {
        "token": str(token),
        "image_paths": [str(Path(path)) for path in image_paths],
        "ego_states": torch.as_tensor(ego_states, dtype=torch.float32),
        "candidate_trajectories": torch.as_tensor(candidate_trajectories, dtype=torch.float32),
        "targets": target_tensor,
        "valid_mask": valid_mask_tensor,
    }


def image_paths_from_scene(
    scene: Any,
    *,
    camera_names: Sequence[str] = DEFAULT_NAVSIM_CAMERA_NAMES,
    frame_indices: Sequence[int] = (-1,),
) -> list[str]:
    paths: list[str] = []
    for frame_index in frame_indices:
        frame = scene.frames[int(frame_index)]
        cameras = getattr(frame, "cameras")
        for camera_name in camera_names:
            camera = getattr(cameras, str(camera_name))
            image_path = getattr(camera, "image_path", None)
            if image_path is None:
                raise ValueError(f"Scene camera {camera_name} at frame_index={frame_index} has no image path")
            paths.append(str(Path(image_path)))
    return paths


def _camera_image_from_scene(scene: Any, camera_name: str, frame_index: int = -1) -> np.ndarray:
    frame = scene.frames[int(frame_index)]
    cameras = getattr(frame, "cameras")
    camera = getattr(cameras, str(camera_name))
    image = getattr(camera, "image", None)
    if image is None:
        raise ValueError(f"Scene camera {camera_name} at frame_index={frame_index} has no loaded image")
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[-1]) != 3:
        raise ValueError(f"Expected HWC RGB image for camera {camera_name}, got shape {tuple(arr.shape)}")
    max_value = float(np.nanmax(arr))
    if max_value > 32.0:
        arr = arr / 255.0
    return arr


def extract_observation_tensor_from_scene(
    scene: Any,
    *,
    camera_names: Sequence[str] = DEFAULT_NAVSIM_CAMERA_NAMES,
    frame_indices: Sequence[int] = (-1,),
) -> np.ndarray:
    channels: list[np.ndarray] = []
    for frame_index in frame_indices:
        for camera_name in camera_names:
            image = _camera_image_from_scene(scene, camera_name, int(frame_index))
            channels.append(np.transpose(image, (2, 0, 1)))
    return np.concatenate(channels, axis=0).astype(np.float32, copy=False)


def default_ego_state_features_from_scene(scene: Any, *, frame_index: int = -1) -> np.ndarray:
    ego_status = getattr(scene.frames[int(frame_index)], "ego_status")
    parts = [
        np.asarray(getattr(ego_status, "ego_pose"), dtype=np.float32).reshape(-1),
        np.asarray(getattr(ego_status, "ego_velocity"), dtype=np.float32).reshape(-1),
        np.asarray(getattr(ego_status, "ego_acceleration"), dtype=np.float32).reshape(-1),
        np.asarray(getattr(ego_status, "driving_command"), dtype=np.float32).reshape(-1),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def build_reward_sample_from_scene_data(
    *,
    scene: Any,
    candidate_trajectories: Any,
    targets: Any,
    camera_names: Sequence[str] = DEFAULT_NAVSIM_CAMERA_NAMES,
    frame_indices: Sequence[int] = (-1,),
    token: str | None = None,
    valid_mask: Any | None = None,
) -> dict[str, torch.Tensor | str]:
    if token is None:
        metadata = getattr(scene, "scene_metadata", None)
        token = str(getattr(metadata, "initial_token", "unknown"))
    return build_reward_sample_from_raw(
        image_paths=image_paths_from_scene(
            scene,
            camera_names=camera_names,
            frame_indices=frame_indices,
        ),
        ego_states=default_ego_state_features_from_scene(scene, frame_index=int(frame_indices[-1])),
        candidate_trajectories=candidate_trajectories,
        targets=targets,
        token=str(token),
        valid_mask=valid_mask,
    )


def save_reward_sample(path: str | Path, sample: Mapping[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(sample), out_path)


def build_targets_from_temporal_pdm(
    metric_scores_per_horizon: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    mapped = [map_pdm_metric_names(horizon_scores) for horizon_scores in metric_scores_per_horizon]
    return torch.as_tensor(stack_temporal_metric_targets(mapped), dtype=torch.float32)


def candidate_prefix_trajectories_for_horizon(candidate_trajectories: Any, horizon_index: int) -> np.ndarray:
    candidates = np.asarray(candidate_trajectories, dtype=np.float32)
    if candidates.ndim != 3 or int(candidates.shape[-1]) < 3:
        raise ValueError(f"Expected candidate trajectories [G,T,3+], got {tuple(candidates.shape)}")
    horizon = int(horizon_index)
    if horizon < 0 or horizon >= int(candidates.shape[1]):
        raise ValueError(f"horizon_index={horizon} out of range for T={int(candidates.shape[1])}")
    out = candidates.copy()
    out[:, horizon + 1 :, :3] = out[:, horizon : horizon + 1, :3]
    return out[..., :3]


def score_candidates_with_navsim_pdm(
    *,
    metric_cache: Any,
    candidate_trajectories: Any,
    simulator: Any,
    scorer: Any,
    traffic_agents_policy: Any,
    future_sampling: Any,
    pdm_score_fn: Callable[..., Mapping[str, Any]],
) -> np.ndarray:
    candidates = np.asarray(candidate_trajectories, dtype=np.float32)
    metric_scores = pdm_score_fn(
        metric_cache=metric_cache,
        model_trajectory=candidates,
        future_sampling=future_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
    )
    mapped = map_pdm_metric_names(metric_scores)
    return stack_temporal_metric_targets([mapped])


def score_candidates_with_navsim_pdm_dense(
    *,
    metric_cache: Any,
    candidate_trajectories: Any,
    simulator: Any,
    scorer: Any,
    traffic_agents_policy: Any,
    future_sampling: Any,
    pdm_score_fn: Callable[..., Mapping[str, Any]],
    num_horizons: int | None = None,
) -> np.ndarray:
    candidates = np.asarray(candidate_trajectories, dtype=np.float32)
    horizon_count = int(num_horizons or candidates.shape[1])
    metric_scores_per_horizon = []
    for horizon_idx in range(horizon_count):
        prefix_candidates = candidate_prefix_trajectories_for_horizon(candidates, horizon_idx)
        metric_scores_per_horizon.append(
            pdm_score_fn(
                metric_cache=metric_cache,
                model_trajectory=prefix_candidates,
                future_sampling=future_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
        )
    mapped = [map_pdm_metric_names(metric_scores) for metric_scores in metric_scores_per_horizon]
    return stack_temporal_metric_targets(mapped)


def build_sample_with_teacher(
    *,
    image_paths: Sequence[str | Path],
    ego_states: Any,
    candidate_trajectories: Any,
    token: str,
    teacher_fn: Callable[[torch.Tensor], Sequence[Mapping[str, Any]]],
) -> dict[str, torch.Tensor | str]:
    traj_tensor = torch.as_tensor(candidate_trajectories, dtype=torch.float32)
    targets = build_targets_from_temporal_pdm(teacher_fn(traj_tensor))
    return build_reward_sample_from_raw(
        image_paths=image_paths,
        ego_states=ego_states,
        candidate_trajectories=traj_tensor,
        targets=targets,
        token=token,
        valid_mask=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached reward-model samples from NavSim scenes and PDM metric cache")
    parser.add_argument("--navsim-root", default="/OpenDataset/navsim/dataset", help="Root containing navsim_logs and sensor_blobs")
    parser.add_argument("--split", default="trainval", help="NavSim split folder, e.g. trainval/test/mini")
    parser.add_argument("--metric-cache-path", required=False, help="Metric cache root produced by NavSim run_metric_caching.py")
    parser.add_argument("--candidate-path", required=False, help="Candidate trajectory .npy/.npz with shape [G,T,3]")
    parser.add_argument("--candidate-key", default="trajectories", help="NPZ key for candidate trajectories")
    parser.add_argument("--output-root", required=True, help="Directory for .pt reward-model samples")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--camera-names", nargs="+", default=list(DEFAULT_NAVSIM_CAMERA_NAMES))
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--future-frames", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--note",
        default="",
        help=(
            "This CLI provides common save/sample helpers. For full NavSim generation, "
            "call build_reward_sample_from_scene_data from a script that already has SceneLoader, "
            "MetricCacheLoader, candidate trajectories, and a PDM teacher."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[rewardmodel-cache] output_root={output_root}")
    print(f"[rewardmodel-cache] navsim_root={args.navsim_root} split={args.split}")
    if args.candidate_path:
        candidates = load_candidate_trajectories(args.candidate_path, key=args.candidate_key)
        print(f"[rewardmodel-cache] loaded candidates shape={tuple(candidates.shape)}")
    else:
        print("[rewardmodel-cache] no candidate path provided; import this module and pass per-scene candidates explicitly")
    if args.metric_cache_path:
        print(f"[rewardmodel-cache] metric_cache_path={args.metric_cache_path}")
    if args.dry_run:
        print("[rewardmodel-cache] dry run complete")
        return
    print(
        "[rewardmodel-cache] Full NavSim iteration is intentionally exposed via build_reward_sample_from_scene_data(...) "
        "so callers can choose candidate sampling and PDM teacher batching policy."
    )
    if args.note:
        print(f"[rewardmodel-cache] note={args.note}")


if __name__ == "__main__":
    main()
