from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils


def _write_token2vad(path: Path) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {
        "tok-a": {
            "token": "tok-a",
            "gt_ego_fut_trajs": np.asarray(
                [
                    [1.0, 3.0],
                    [1.2, 3.3],
                    [1.4, 3.7],
                ],
                dtype=np.float32,
            ),
        },
        "tok-b": {
            "token": "tok-b",
            "gt_ego_fut_trajs": np.asarray(
                [
                    [0.5, 2.0],
                    [0.7, 2.2],
                    [0.9, 2.5],
                    [1.1, 2.9],
                ],
                dtype=np.float32,
            ),
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


def test_score_uses_batched_torch_path(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    scorer = NuScenesScorerUtils(token2vad_path=token2vad_path)

    def _raise_if_cpu_detail_path(*args, **kwargs):
        del args, kwargs
        raise AssertionError("score() should not call _score_batch() on the training path")

    monkeypatch.setattr(scorer, "_score_batch", _raise_if_cpu_detail_path)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.7, 0.4, 0.0], [0.9, 0.6, 0.0]],
                [[0.2, -0.1, 0.0], [0.4, 0.0, 0.0], [0.6, 0.1, 0.0], [0.8, 0.2, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.2, 0.2, 0.0], [0.5, 0.4, 0.0], [0.9, 0.6, 0.0]],
                [[0.1, 0.0, 0.0], [0.15, 0.1, 0.0], [0.2, 0.2, 0.0], [0.25, 0.3, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )

    scores = scorer.score(
        [{"sample_token": "tok-a"}, {"sample_token": "tok-b"}],
        traj_xyyaw,
    )

    assert scores.shape == (2, 2)


def test_score_matches_detail_path_for_mixed_horizon_batch(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    scorer = NuScenesScorerUtils(token2vad_path=token2vad_path)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.7, 0.4, 0.0], [0.8, 0.5, 0.0], [1.0, 0.6, 0.0]],
                [[1.0, 1.0, 0.0], [1.2, 1.1, 0.0], [1.4, 1.2, 0.0], [1.6, 1.3, 0.0], [1.8, 1.4, 0.0]],
                [[0.0, -0.2, 0.0], [0.2, 0.0, 0.0], [0.45, 0.2, 0.0], [0.7, 0.35, 0.0], [0.95, 0.5, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.2, 0.2, 0.0], [0.5, 0.4, 0.0], [0.9, 0.6, 0.0], [1.3, 0.9, 0.0]],
                [[0.1, 0.0, 0.0], [0.15, 0.1, 0.0], [0.2, 0.2, 0.0], [0.25, 0.3, 0.0], [0.3, 0.4, 0.0]],
                [[-0.1, 0.1, 0.0], [0.0, 0.2, 0.0], [0.1, 0.4, 0.0], [0.2, 0.7, 0.0], [0.4, 1.0, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )

    scores = scorer.score(
        [{"sample_token": "tok-a"}, {"sample_token": "tok-b"}],
        traj_xyyaw,
    )
    detail_scores_np, details = scorer._score_batch(
        [{"sample_token": "tok-a"}, {"sample_token": "tok-b"}],
        traj_xyyaw,
        include_debug_context=False,
    )
    detail_scores = torch.from_numpy(detail_scores_np)

    assert len(details) == 2
    assert scores.shape == (2, 3)
    assert torch.allclose(scores.cpu(), detail_scores.cpu(), atol=1.0e-5, rtol=1.0e-5)


def test_detail_path_reuses_static_context_per_sample_token(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    scorer = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    calls = {
        "lookup_gt": 0,
        "lookup_map_layers": 0,
        "collect_scene_objects": 0,
        "collect_ea_agent_states": 0,
    }

    orig_lookup_gt = scorer._lookup_gt
    orig_lookup_row = scorer._lookup_row
    orig_collect_scene_objects = scorer._collect_scene_objects

    def wrapped_lookup_gt(sample_token: str):
        calls["lookup_gt"] += 1
        return orig_lookup_gt(sample_token)

    def wrapped_lookup_map_layers(row, *, patch_radius=20.0):
        calls["lookup_map_layers"] += 1
        return {"patch_radius": float(patch_radius), "layers": {"lane_centerline": [], "drivable_area": []}}

    def wrapped_collect_scene_objects(row, *, patch_radius: float):
        del row, patch_radius
        calls["collect_scene_objects"] += 1
        return []

    def wrapped_collect_ea_agent_states(replay, *, patch_radius: float, row=None):
        del replay, patch_radius, row
        calls["collect_ea_agent_states"] += 1
        return []

    monkeypatch.setattr(scorer, "_lookup_gt", wrapped_lookup_gt)
    monkeypatch.setattr(scorer, "_lookup_map_layers", wrapped_lookup_map_layers)
    monkeypatch.setattr(scorer, "_collect_scene_objects", wrapped_collect_scene_objects)
    monkeypatch.setattr(scorer, "_collect_ea_agent_states", wrapped_collect_ea_agent_states)
    monkeypatch.setattr(scorer, "_lookup_cached_map_layers", lambda replay, *, patch_radius=20.0: None)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.7, 0.4, 0.0]],
                [[0.2, -0.1, 0.0], [0.4, 0.0, 0.0], [0.6, 0.1, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    replays = [{"sample_token": "tok-a"}]

    scores_0, _details_0 = scorer._score_batch(replays, traj_xyyaw, include_debug_context=False)
    scores_1, _details_1 = scorer._score_batch(replays, traj_xyyaw, include_debug_context=False)

    assert scores_0.shape == (1, 2)
    assert np.allclose(scores_0, scores_1)
    assert calls["lookup_gt"] == 1
    assert calls["lookup_map_layers"] == 1
    assert calls["collect_scene_objects"] == 1
    assert calls["collect_ea_agent_states"] == 1


def test_detail_path_reuses_static_context_from_disk_across_scorer_instances(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    scene_cache_root = tmp_path / "scene_cache"
    _write_token2vad(token2vad_path)

    scorer_first = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root,
    )

    calls = {
        "lookup_gt": 0,
        "lookup_map_layers": 0,
        "collect_scene_objects": 0,
        "collect_ea_agent_states": 0,
    }

    orig_lookup_gt = scorer_first._lookup_gt

    def wrapped_lookup_gt(sample_token: str):
        calls["lookup_gt"] += 1
        return orig_lookup_gt(sample_token)

    def wrapped_lookup_map_layers(row, *, patch_radius=20.0):
        calls["lookup_map_layers"] += 1
        return {"patch_radius": float(patch_radius), "layers": {"lane_centerline": [], "drivable_area": []}}

    def wrapped_collect_scene_objects(row, *, patch_radius: float):
        del row, patch_radius
        calls["collect_scene_objects"] += 1
        return []

    def wrapped_collect_ea_agent_states(replay, *, patch_radius: float, row=None):
        del replay, patch_radius, row
        calls["collect_ea_agent_states"] += 1
        return []

    monkeypatch.setattr(scorer_first, "_lookup_gt", wrapped_lookup_gt)
    monkeypatch.setattr(scorer_first, "_lookup_map_layers", wrapped_lookup_map_layers)
    monkeypatch.setattr(scorer_first, "_collect_scene_objects", wrapped_collect_scene_objects)
    monkeypatch.setattr(scorer_first, "_collect_ea_agent_states", wrapped_collect_ea_agent_states)
    monkeypatch.setattr(scorer_first, "_lookup_cached_map_layers", lambda replay, *, patch_radius=20.0: None)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.7, 0.4, 0.0]],
                [[0.2, -0.1, 0.0], [0.4, 0.0, 0.0], [0.6, 0.1, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    replays = [{"sample_token": "tok-a"}]

    scores_0, _details_0 = scorer_first._score_batch(replays, traj_xyyaw, include_debug_context=False)

    scorer_second = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root,
    )

    def should_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("expected second scorer to reuse persisted static sample context")

    monkeypatch.setattr(scorer_second, "_lookup_gt", should_not_run)
    monkeypatch.setattr(scorer_second, "_lookup_map_layers", should_not_run)
    monkeypatch.setattr(scorer_second, "_collect_scene_objects", should_not_run)
    monkeypatch.setattr(scorer_second, "_collect_ea_agent_states", should_not_run)
    monkeypatch.setattr(scorer_second, "_lookup_cached_map_layers", lambda replay, *, patch_radius=20.0: None)

    scores_1, _details_1 = scorer_second._score_batch(replays, traj_xyyaw, include_debug_context=False)

    assert scores_0.shape == (1, 2)
    assert np.allclose(scores_0, scores_1)
    assert calls["lookup_gt"] == 1
    assert calls["lookup_map_layers"] == 1
    assert calls["collect_scene_objects"] == 1
    assert calls["collect_ea_agent_states"] == 1
