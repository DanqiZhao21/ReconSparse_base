from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from framework.algorithms.nuscenes_token_scorer import NuScenesTokenScorer


def _write_scene_env_cache(root: Path, *, scene_id: int, frame_idx: int, payload: dict[str, object]) -> None:
    scene_dir = root / f"{int(scene_id):03d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    path = scene_dir / "env_cache.json"
    path.write_text(json.dumps({str(int(frame_idx)): payload}, indent=2), encoding="utf-8")


def _write_token2vad(path: Path, *, token: str, gt_ego_fut_trajs: np.ndarray) -> None:
    payload = {
        token: {
            "token": token,
            "ego2global_translation": [0.0, 0.0, 0.0],
            "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
            "gt_ego_fut_trajs": np.asarray(gt_ego_fut_trajs, dtype=np.float32),
            "gt_boxes": np.zeros((0, 7), dtype=np.float32),
            "gt_velocity": np.zeros((0, 2), dtype=np.float32),
            "gt_names": np.asarray([], dtype=object),
            "valid_flag": np.zeros((0,), dtype=bool),
        }
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_score_increases_progress_preference_when_progress_weight_is_increased(tmp_path: Path) -> None:
    token = "tok-progress-weight"
    token2vad_path = tmp_path / "token2vad.pkl"
    cache_root = tmp_path / "scene_cache"
    _write_token2vad(
        token2vad_path,
        token=token,
        gt_ego_fut_trajs=np.asarray(
            [
                [0.0, 2.0],
                [0.0, 2.0],
                [0.0, 2.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        ),
    )
    _write_scene_env_cache(
        cache_root,
        scene_id=230,
        frame_idx=0,
        payload={
            "sample_token": token,
            "ego_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "drivable_polygons": [
                [[-2.0, -3.0], [12.0, -3.0], [12.0, 3.0], [-2.0, 3.0], [-2.0, -3.0]],
            ],
            "lanes_centerlines": [
                [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]],
            ],
            "static_objects": [],
            "dynamic_objects": [],
        },
    )

    traj_xyyaw = torch.tensor(
        [
            [
                [[1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0], [6.0, 0.0, 0.0]],
                [[1.8, 1.6, 0.0], [3.6, 1.6, 0.0], [5.4, 1.6, 0.0], [7.2, 1.6, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    scorer_default = NuScenesTokenScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=cache_root,
        progress_weight=5.0,
        ttc_weight=5.0,
        lane_keeping_weight=2.0,
        history_comfort_weight=2.0,
    )
    default_scores, default_details = scorer_default.score_with_details(
        [{"sample_token": token, "scene_id": 230, "frame_idx": 0}],
        traj_xyyaw,
    )

    scorer = NuScenesTokenScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=cache_root,
        progress_weight=8.0,
        ttc_weight=5.0,
        lane_keeping_weight=2.0,
        history_comfort_weight=2.0,
    )
    scores, details = scorer.score_with_details(
        [{"sample_token": token, "scene_id": 230, "frame_idx": 0}],
        traj_xyyaw,
    )

    default_lower_progress = default_details[0]["candidates"][0]
    default_higher_progress = default_details[0]["candidates"][1]
    lower_progress = details[0]["candidates"][0]
    higher_progress = details[0]["candidates"][1]
    default_gap = float(default_scores[0, 1]) - float(default_scores[0, 0])
    boosted_gap = float(scores[0, 1]) - float(scores[0, 0])

    assert higher_progress["progress_ratio"] > lower_progress["progress_ratio"]
    assert higher_progress["weighted_metrics"]["lane_keeping"] < lower_progress["weighted_metrics"]["lane_keeping"]
    assert default_higher_progress["progress_ratio"] == pytest.approx(higher_progress["progress_ratio"])
    assert default_lower_progress["progress_ratio"] == pytest.approx(lower_progress["progress_ratio"])
    assert boosted_gap > default_gap
