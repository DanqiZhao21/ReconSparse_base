from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from framework.rewardmodel.data.navsim_cache_builder import (
    DEFAULT_NAVSIM_CAMERA_NAMES,
    build_reward_sample_from_scene_data,
    build_scene_specific_candidates,
    build_vocabulary_from_gt_trajectories,
    load_candidate_trajectories,
    save_reward_sample,
    score_candidates_with_navsim_pdm_dense,
)


def build_sensor_config_for_cameras(SensorConfig: Any, camera_names: tuple[str, ...], include: list[int]) -> Any:
    sensor_config = SensorConfig.build_no_sensors()
    for camera_name in camera_names:
        if not hasattr(sensor_config, str(camera_name)):
            raise ValueError(f"Unknown NavSim camera name: {camera_name}")
        setattr(sensor_config, str(camera_name), list(include))
    return sensor_config


def scene_filter_scope_from_metric_cache(metric_cache_paths: dict[str, Path]) -> tuple[list[str], list[str]]:
    tokens = sorted(str(token) for token in metric_cache_paths.keys())
    log_names = sorted({Path(path).parents[2].name for path in metric_cache_paths.values()})
    return log_names, tokens


def split_tokens_for_workers(tokens: Sequence[str], num_workers: int) -> list[list[str]]:
    worker_count = max(1, int(num_workers))
    return [list(tokens[worker_idx::worker_count]) for worker_idx in range(worker_count)]


def load_token_list(path: str | Path) -> list[str]:
    with Path(path).open("r") as f:
        return [line.strip() for line in f if line.strip()]


def _import_navsim_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    candidates = [
        "/root/clone/ReconDreamer-RL/egoADs/SparseDriveV2",
        os.environ.get("NAVSIM_DEVKIT_ROOT", ""),
    ]
    for root in candidates:
        if root and root not in os.sys.path:
            os.sys.path.insert(0, root)
    navsim_common = importlib.import_module("navsim.common.dataclasses")
    navsim_loader = importlib.import_module("navsim.common.dataloader")
    get_pdm_score_mod = importlib.import_module("navsim.agents.sparsedrive.scorer.get_pdm_score_v2")
    hydra_utils = importlib.import_module("hydra.utils")
    omegaconf = importlib.import_module("omegaconf")
    return (
        navsim_common.SceneFilter,
        navsim_common.SensorConfig,
        navsim_loader.SceneLoader,
        navsim_loader.MetricCacheLoader,
        hydra_utils.instantiate,
        omegaconf.OmegaConf,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rewardmodel cache from NavSim scenes and metric cache")
    parser.add_argument("--navsim-root", default="/OpenDataset/navsim/dataset")
    parser.add_argument("--split", default="trainval")
    parser.add_argument("--metric-cache-path", required=True)
    parser.add_argument("--candidate-path", required=False)
    parser.add_argument("--candidate-key", default="trajectories")
    parser.add_argument("--build-vocabulary-from-gt", action="store_true")
    parser.add_argument("--vocabulary-source-split", default=None)
    parser.add_argument("--max-vocabulary-size", type=int, default=8192)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--camera-names", nargs="+", default=list(DEFAULT_NAVSIM_CAMERA_NAMES))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-future-frames", type=int, default=8)
    parser.add_argument("--scene-filter-has-route", action="store_true", default=True)
    parser.add_argument("--max-candidate-samples", type=int, default=256)
    parser.add_argument("--max-longitudinal-error-m", type=float, default=10.0)
    parser.add_argument("--max-lateral-error-m", type=float, default=5.0)
    parser.add_argument("--max-heading-error-deg", type=float, default=20.0)
    parser.add_argument("--pdm-config", default="/root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/navsim/planning/script/config/pdm_scoring/run_pdm_train.yaml")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--token-list-path", default=None)
    return parser.parse_args()


def _build_vocabulary(args: argparse.Namespace, SceneFilter: Any, SensorConfig: Any, SceneLoader: Any) -> np.ndarray:
    navsim_root = Path(args.navsim_root)
    if bool(args.build_vocabulary_from_gt):
        vocabulary_split = str(args.vocabulary_source_split or args.split)
        vocabulary_loader = SceneLoader(
            data_path=navsim_root / "navsim_logs" / vocabulary_split,
            original_sensor_path=navsim_root / "sensor_blobs" / vocabulary_split,
            synthetic_sensor_path=navsim_root / "sensor_blobs" / vocabulary_split,
            synthetic_scenes_path=navsim_root / "navsim_logs" / vocabulary_split,
            scene_filter=SceneFilter(
                num_history_frames=int(args.num_history_frames),
                num_future_frames=int(args.num_future_frames),
                frame_interval=1,
                has_route=bool(args.scene_filter_has_route),
                max_scenes=int(args.max_vocabulary_size),
            ),
            sensor_config=SensorConfig.build_no_sensors(),
        )
        gt_trajectories = [
            np.asarray(vocabulary_loader.get_scene_from_token(token).get_future_trajectory().poses, dtype=np.float32)
            for token in vocabulary_loader.tokens
        ]
        return build_vocabulary_from_gt_trajectories(
            gt_trajectories,
            max_vocabulary_size=int(args.max_vocabulary_size),
        )
    if not args.candidate_path:
        raise ValueError("Provide --candidate-path or enable --build-vocabulary-from-gt")
    return load_candidate_trajectories(args.candidate_path, key=args.candidate_key)


def _make_scene_loader(
    *,
    args: argparse.Namespace,
    SceneFilter: Any,
    SensorConfig: Any,
    SceneLoader: Any,
    metric_cache_paths: dict[str, Path],
    tokens: Sequence[str],
) -> tuple[Any, tuple[str, ...]]:
    navsim_root = Path(args.navsim_root)
    worker_metric_cache_paths = {str(token): metric_cache_paths[str(token)] for token in tokens}
    log_names, scoped_tokens = scene_filter_scope_from_metric_cache(worker_metric_cache_paths)
    camera_names = tuple(str(name).lower() for name in args.camera_names)
    sensor_config = build_sensor_config_for_cameras(
        SensorConfig,
        camera_names,
        include=[int(args.num_history_frames) - 1],
    )
    scene_filter = SceneFilter(
        num_history_frames=int(args.num_history_frames),
        num_future_frames=int(args.num_future_frames),
        frame_interval=1,
        has_route=bool(args.scene_filter_has_route),
        max_scenes=None,
        log_names=log_names,
        tokens=scoped_tokens,
    )
    return (
        SceneLoader(
            data_path=navsim_root / "navsim_logs" / str(args.split),
            original_sensor_path=navsim_root / "sensor_blobs" / str(args.split),
            synthetic_sensor_path=navsim_root / "sensor_blobs" / str(args.split),
            synthetic_scenes_path=navsim_root / "navsim_logs" / str(args.split),
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        ),
        camera_names,
    )


def _worker_build_cache(
    worker_id: int,
    args: argparse.Namespace,
    tokens: Sequence[str],
    vocabulary: np.ndarray,
    saved_counter: Any,
    skipped_counter: Any,
    stop_event: Any,
    lock: Any,
) -> None:
    SceneFilter, SensorConfig, SceneLoader, MetricCacheLoader, instantiate, OmegaConf = _import_navsim_modules()
    get_pdm_score_mod = importlib.import_module("navsim.agents.sparsedrive.scorer.get_pdm_score_v2")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache_path))
    scene_loader, camera_names = _make_scene_loader(
        args=args,
        SceneFilter=SceneFilter,
        SensorConfig=SensorConfig,
        SceneLoader=SceneLoader,
        metric_cache_paths=metric_cache_loader.metric_cache_paths,
        tokens=tokens,
    )

    pdm_cfg = OmegaConf.load(str(args.pdm_config))
    simulator = instantiate(pdm_cfg.simulator)
    scorer = instantiate(pdm_cfg.scorer)
    scorer.train_mode = True
    traffic_agents_policy = instantiate(pdm_cfg.non_reactive, simulator.proposal_sampling)
    pdm_score_fn = getattr(get_pdm_score_mod, "pdm_score")

    local_saved = 0
    local_skipped = 0
    for token in scene_loader.tokens:
        if stop_event.is_set():
            break
        if token not in metric_cache_loader.metric_cache_paths:
            continue
        out_path = output_root / f"{token}.pt"
        if out_path.is_file():
            continue
        if args.max_samples is not None:
            with lock:
                if int(saved_counter.value) >= int(args.max_samples):
                    stop_event.set()
                    break
        try:
            scene = scene_loader.get_scene_from_token(token)
            metric_cache = metric_cache_loader.get_from_token(token)
            candidates = build_scene_specific_candidates(
                scene=scene,
                vocabulary=vocabulary,
                max_longitudinal_error_m=float(args.max_longitudinal_error_m),
                max_lateral_error_m=float(args.max_lateral_error_m),
                max_heading_error_rad=np.deg2rad(float(args.max_heading_error_deg)),
                max_candidates=int(args.max_candidate_samples),
            )
            targets = score_candidates_with_navsim_pdm_dense(
                metric_cache=metric_cache,
                candidate_trajectories=candidates,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
                future_sampling=simulator.proposal_sampling,
                pdm_score_fn=pdm_score_fn,
                num_horizons=int(args.num_future_frames),
            )
            sample = build_reward_sample_from_scene_data(
                scene=scene,
                candidate_trajectories=candidates,
                targets=targets,
                token=token,
                camera_names=camera_names,
                frame_indices=(int(args.num_history_frames) - 1,),
            )
            save_reward_sample(out_path, sample)
            local_saved += 1
            with lock:
                saved_counter.value += 1
                global_saved = int(saved_counter.value)
                if args.max_samples is not None and global_saved >= int(args.max_samples):
                    stop_event.set()
        except (FileNotFoundError, ValueError, KeyError) as exc:
            local_skipped += 1
            with lock:
                skipped_counter.value += 1
            print(f"[rewardmodel-cache][worker={worker_id}] skipped token={token} reason={type(exc).__name__}: {exc}")
            continue
        except Exception:
            print(f"[rewardmodel-cache][worker={worker_id}] failed token={token}")
            traceback.print_exc()
            raise
        if local_saved > 0 and local_saved % 20 == 0:
            print(
                f"[rewardmodel-cache][worker={worker_id}] "
                f"local_saved={local_saved} local_skipped={local_skipped} global_saved={int(saved_counter.value)}"
            )

    print(
        f"[rewardmodel-cache][worker={worker_id}] completed "
        f"local_saved={local_saved} local_skipped={local_skipped} global_saved={int(saved_counter.value)}"
    )


def main() -> None:
    args = parse_args()
    SceneFilter, SensorConfig, _SceneLoader, MetricCacheLoader, _instantiate, _OmegaConf = _import_navsim_modules()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache_path))
    _log_names, tokens = scene_filter_scope_from_metric_cache(metric_cache_loader.metric_cache_paths)
    if args.token_list_path:
        requested_tokens = load_token_list(args.token_list_path)
        metric_tokens = set(tokens)
        tokens = [token for token in requested_tokens if token in metric_tokens]
        if not tokens:
            raise ValueError(f"No requested tokens from {args.token_list_path} exist in metric cache {args.metric_cache_path}")
    existing_samples = len(list(output_root.glob("*.pt")))
    vocabulary = _build_vocabulary(args, SceneFilter, SensorConfig, _SceneLoader)
    num_processes = max(1, int(args.num_processes))
    if num_processes <= 1:
        ctx = mp.get_context("fork")
        saved_counter = ctx.Value("i", int(existing_samples))
        skipped_counter = ctx.Value("i", 0)
        stop_event = ctx.Event()
        lock = ctx.Lock()
        _worker_build_cache(0, args, tokens, vocabulary, saved_counter, skipped_counter, stop_event, lock)
    else:
        ctx = mp.get_context("fork")
        saved_counter = ctx.Value("i", int(existing_samples))
        skipped_counter = ctx.Value("i", 0)
        stop_event = ctx.Event()
        lock = ctx.Lock()
        token_shards = [shard for shard in split_tokens_for_workers(tokens, num_processes) if len(shard) > 0]
        print(
            f"[rewardmodel-cache] starting num_processes={len(token_shards)} "
            f"tokens={len(tokens)} existing_samples={existing_samples} max_samples={args.max_samples}"
        )
        processes = [
            ctx.Process(
                target=_worker_build_cache,
                args=(worker_id, args, shard, vocabulary, saved_counter, skipped_counter, stop_event, lock),
            )
            for worker_id, shard in enumerate(token_shards)
        ]
        for process in processes:
            process.start()
        try:
            for process in processes:
                process.join()
        except KeyboardInterrupt:
            stop_event.set()
            for process in processes:
                process.terminate()
            raise
        failed = [process for process in processes if process.exitcode not in (0, None)]
        if failed:
            stop_event.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
            failed_codes = [(process.pid, process.exitcode) for process in failed]
            raise RuntimeError(f"Reward cache worker failure(s): {failed_codes}")
    print(
        f"[rewardmodel-cache] completed saved={int(saved_counter.value)} "
        f"skipped={int(skipped_counter.value)} output_root={output_root}"
    )


if __name__ == "__main__":
    main()
