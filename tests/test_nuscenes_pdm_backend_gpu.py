from __future__ import annotations

import importlib.util
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer as CpuPDMScorer


def _write_token2vad(path: Path) -> None:
    payload = {
        "tok-a": {
            "token": "tok-a",
            "gt_ego_fut_trajs": np.asarray(
                [
                    [1.0, 3.0],
                    [1.2, 3.3],
                    [1.4, 3.7],
                    [1.6, 4.1],
                ],
                dtype=np.float32,
            ),
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _load_gpu_backend_module():
    path = Path("/root/clone/ReconDreamer-RL/framework/algorithms/nuscenes_pdm_backend-GPU.py")
    spec = importlib.util.spec_from_file_location("nuscenes_pdm_backend_gpu_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_sample_context(scorer) -> object:
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.drivable_polygons = [
        np.asarray([[-5.0, -5.0], [8.0, -5.0], [8.0, 8.0], [-5.0, 8.0]], dtype=np.float32)
    ]
    sample_context.drivable_map = scorer.__class__.__mro__[0].__dict__.get("NuScenesPDMDrivableMap", None)
    return sample_context


def _attach_manual_context(scorer) -> object:
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.drivable_polygons = [
        np.asarray([[-5.0, -5.0], [8.0, -5.0], [8.0, 8.0], [-5.0, 8.0]], dtype=np.float32)
    ]
    sample_context.drivable_map = type(sample_context.drivable_map)(sample_context.drivable_polygons)
    sample_context.lane_centerlines = [
        np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    ]
    sample_context.centerline_segments_xy = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0]],
        ],
        dtype=np.float32,
    )
    sample_context.centerline_tangents_xy = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    sample_context.scene_objects = [
        {
            "token": "obj-a",
            "center_xy": [1.7, 0.0],
            "velocity_xy": [0.0, 0.0],
            "length_m": 4.0,
            "width_m": 1.8,
            "yaw_rad": 0.0,
            "corners_xy": [[-0.3, -0.9], [3.7, -0.9], [3.7, 0.9], [-0.3, 0.9]],
        }
    ]
    object_tokens, object_polygons, object_velocity_xy, occupancy_map = scorer._build_object_geometry_arrays(
        sample_context.scene_objects
    )
    sample_context.object_tokens = object_tokens
    sample_context.object_polygons = object_polygons
    sample_context.object_velocity_xy = object_velocity_xy
    sample_context.occupancy_map = occupancy_map
    return sample_context


def _build_replay_and_traj() -> tuple[list[dict[str, str]], torch.Tensor]:
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0], [1.2, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.4, 1.0, 0.0], [0.8, 1.0, 0.0], [1.2, 1.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.5, -0.2, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    return [{"sample_token": "tok-a"}], traj_xyyaw


def _build_large_replays_and_traj(
    *,
    batch_size: int = 4,
    num_candidates: int = 32,
    horizon: int = 16,
) -> tuple[list[dict[str, str]], torch.Tensor]:
    traj_xyyaw = torch.zeros((batch_size, num_candidates, horizon, 3), dtype=torch.float32)
    for batch_idx in range(batch_size):
        for cand_idx in range(num_candidates):
            traj_xyyaw[batch_idx, cand_idx, :, 0] = torch.linspace(0.0, 6.0, horizon) + 0.05 * cand_idx
            traj_xyyaw[batch_idx, cand_idx, :, 1] = -1.5 + float(cand_idx % 8) * 0.4
    return [{"sample_token": "tok-a"} for _ in range(batch_size)], traj_xyyaw


def _attach_dense_manual_context(scorer, *, num_objects: int = 64) -> object:
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.drivable_polygons = [
        np.asarray([[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]], dtype=np.float32)
    ]
    sample_context.drivable_map = type(sample_context.drivable_map)(sample_context.drivable_polygons)
    sample_context.lane_centerlines = [
        np.stack([np.linspace(0.0, 20.0, 32), np.zeros((32,), dtype=np.float32)], axis=1).astype(np.float32)
    ]
    sample_context.centerline_segments_xy = np.stack(
        [np.asarray([[float(i), 0.0], [float(i + 1), 0.0]], dtype=np.float32) for i in range(31)],
        axis=0,
    )
    sample_context.centerline_tangents_xy = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (31, 1))
    scene_objects = []
    for obj_idx in range(num_objects):
        center_x = 2.0 + 0.6 * obj_idx
        center_y = (float(obj_idx % 4) - 1.5) * 1.2
        scene_objects.append(
            {
                "token": f"obj-{obj_idx}",
                "center_xy": [center_x, center_y],
                "velocity_xy": [0.2 * float((obj_idx % 3) - 1), 0.0],
                "length_m": 4.2,
                "width_m": 1.9,
                "yaw_rad": 0.0,
                "corners_xy": [
                    [center_x - 2.1, center_y - 0.95],
                    [center_x + 2.1, center_y - 0.95],
                    [center_x + 2.1, center_y + 0.95],
                    [center_x - 2.1, center_y + 0.95],
                ],
            }
        )
    sample_context.scene_objects = scene_objects
    object_tokens, object_polygons, object_velocity_xy, occupancy_map = scorer._build_object_geometry_arrays(scene_objects)
    sample_context.object_tokens = object_tokens
    sample_context.object_polygons = object_polygons
    sample_context.object_velocity_xy = object_velocity_xy
    sample_context.occupancy_map = occupancy_map
    return sample_context


def test_gpu_backend_matches_cpu_backend_scores_on_manual_context(tmp_path: Path, monkeypatch) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    gpu_mod = _load_gpu_backend_module()

    cpu = CpuPDMScorer(token2vad_path=token2vad_path)
    gpu = gpu_mod.NuScenesPDMScorer(token2vad_path=token2vad_path)
    cpu_ctx = _attach_manual_context(cpu)
    gpu_ctx = _attach_manual_context(gpu)

    monkeypatch.setattr(cpu, "_build_sample_context", lambda replay, patch_radius: cpu_ctx)
    monkeypatch.setattr(gpu, "_build_sample_context", lambda replay, patch_radius: gpu_ctx)

    replays, traj_xyyaw = _build_replay_and_traj()

    cpu_scores = cpu.score(replays, traj_xyyaw)
    gpu_scores = gpu.score(replays, traj_xyyaw)

    assert cpu_scores.shape == gpu_scores.shape == (1, 3)
    assert np.allclose(cpu_scores, gpu_scores, atol=2.0e-2, rtol=2.0e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU backend benchmark")
def test_gpu_backend_score_executes_on_cuda_input_and_returns_finite_scores(tmp_path: Path, monkeypatch) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    gpu_mod = _load_gpu_backend_module()

    cpu = CpuPDMScorer(token2vad_path=token2vad_path)
    gpu = gpu_mod.NuScenesPDMScorer(token2vad_path=token2vad_path)
    cpu_ctx = _attach_dense_manual_context(cpu)
    gpu_ctx = _attach_dense_manual_context(gpu)

    monkeypatch.setattr(cpu, "_build_sample_context", lambda replay, patch_radius: cpu_ctx)
    monkeypatch.setattr(gpu, "_build_sample_context", lambda replay, patch_radius: gpu_ctx)

    replays, traj_xyyaw = _build_large_replays_and_traj()
    traj_gpu = traj_xyyaw.cuda()

    cpu_scores = cpu.score(replays, traj_xyyaw)
    gpu_scores = None
    gpu_start = time.perf_counter()
    for _ in range(3):
        gpu_scores = gpu.score(replays, traj_gpu)
    torch.cuda.synchronize()
    gpu_elapsed = time.perf_counter() - gpu_start

    assert cpu_scores.shape == gpu_scores.shape == (4, 32)
    assert np.all(np.isfinite(gpu_scores))
    assert gpu_elapsed > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU backend benchmark")
def test_gpu_backend_hot_collision_ttc_path_is_faster_than_cpu_backend(tmp_path: Path, monkeypatch) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    gpu_mod = _load_gpu_backend_module()

    cpu = CpuPDMScorer(token2vad_path=token2vad_path)
    gpu = gpu_mod.NuScenesPDMScorer(token2vad_path=token2vad_path)
    cpu_ctx = _attach_dense_manual_context(cpu)
    gpu_ctx = _attach_dense_manual_context(gpu)

    monkeypatch.setattr(cpu, "_build_sample_context", lambda replay, patch_radius: cpu_ctx)
    monkeypatch.setattr(gpu, "_build_sample_context", lambda replay, patch_radius: gpu_ctx)

    _, traj_xyyaw = _build_large_replays_and_traj(batch_size=1)
    traj_gpu = traj_xyyaw.cuda()

    cpu_geometry = cpu._build_candidate_geometry_batch(traj_xyyaw)
    gpu_geometry = gpu._build_candidate_geometry_batch_torch(traj_gpu)

    cpu_start = time.perf_counter()
    for _ in range(20):
        cpu._batch_collision_ttc_metrics(
            sample_context=cpu_ctx,
            candidate_geometry={key: value[0] for key, value in cpu_geometry.items()},
            dt_s=0.5,
        )
    cpu_elapsed = time.perf_counter() - cpu_start

    gpu_start = time.perf_counter()
    for _ in range(20):
        gpu._batch_collision_ttc_metrics_torch(
            sample_context=gpu_ctx,
            candidate_geometry={key: value[0] for key, value in gpu_geometry.items()},
            dt_s=0.5,
        )
    torch.cuda.synchronize()
    gpu_elapsed = time.perf_counter() - gpu_start

    assert gpu_elapsed < cpu_elapsed
