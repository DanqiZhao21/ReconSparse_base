from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils


def _write_scene_env_cache(root: Path, *, scene_id: int, frame_idx: int, payload: dict[str, object]) -> None:
    scene_dir = root / f"{int(scene_id):03d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    path = scene_dir / "env_cache.json"
    path.write_text(json.dumps({str(int(frame_idx)): payload}, indent=2), encoding="utf-8")


def _write_token2vad(path: Path, *, token: str, gt_ego_fut_trajs: np.ndarray, gt_boxes: np.ndarray | None = None) -> None:
    payload: dict[str, dict[str, object]] = {
        token: {
            "token": token,
            "ego2global_translation": [0.0, 0.0, 0.0],
            "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
            "gt_ego_fut_trajs": np.asarray(gt_ego_fut_trajs, dtype=np.float32),
        }
    }
    if gt_boxes is not None:
        payload[token]["gt_boxes"] = np.asarray(gt_boxes, dtype=np.float32)
        payload[token]["gt_velocity"] = np.zeros((gt_boxes.shape[0], 2), dtype=np.float32)
        payload[token]["gt_names"] = np.asarray(["vehicle.car"] * gt_boxes.shape[0], dtype=object)
        payload[token]["valid_flag"] = np.ones((gt_boxes.shape[0],), dtype=bool)
        payload[token]["num_lidar_pts"] = np.ones((gt_boxes.shape[0],), dtype=np.int64)
        payload[token]["num_radar_pts"] = np.zeros((gt_boxes.shape[0],), dtype=np.int64)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_score_prefers_cached_drivable_candidate_over_offroad_gt_fit(tmp_path: Path) -> None:
    token = "tok-cache"
    token2vad_path = tmp_path / "token2vad.pkl"
    cache_root = tmp_path / "scene_cache"
    _write_token2vad(
        token2vad_path,
        token=token,
        gt_ego_fut_trajs=np.asarray(
            [
                [2.0, 3.0],
                [2.0, 3.0],
                [2.0, 3.0],
            ],
            dtype=np.float32,
        ),
    )
    _write_scene_env_cache(
        cache_root,
        scene_id=146,
        frame_idx=0,
        payload={
            "sample_token": token,
            "ego_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "drivable_polygons": [
                [[-1.0, -1.2], [12.0, -1.2], [12.0, 1.2], [-1.0, 1.2], [-1.0, -1.2]],
            ],
            "lanes_centerlines": [
                [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]],
            ],
            "static_objects": [],
            "dynamic_objects": [],
        },
    )

    scorer = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        scene_cache_root=cache_root,
    )
    traj_xyyaw = torch.tensor(
        [
            [
                [[3.0, 2.0, 0.0], [6.0, 2.0, 0.0], [9.0, 2.0, 0.0]],
                [[3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    scores = scorer.score(
        [{"sample_token": token, "scene_id": 146, "frame_idx": 0}],
        traj_xyyaw,
    )

    assert scores.shape == (1, 2)
    assert float(scores[0, 1]) > float(scores[0, 0])


def test_score_with_details_reports_pdm_like_breakdown_and_collision_gate(tmp_path: Path) -> None:
    token = "tok-pdm"
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
            ],
            dtype=np.float32,
        ),
        gt_boxes=np.asarray(
            [
                [4.0, 0.0, 0.0, 2.0, 4.0, 1.8, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    _write_scene_env_cache(
        cache_root,
        scene_id=200,
        frame_idx=3,
        payload={
            "sample_token": token,
            "ego_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "drivable_polygons": [
                    [[-2.0, -5.0], [12.0, -5.0], [12.0, 5.0], [-2.0, 5.0], [-2.0, -5.0]],
                ],
            "lanes_centerlines": [
                [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]],
            ],
            "static_objects": [],
            "dynamic_objects": [],
        },
    )

    scorer = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        scene_cache_root=cache_root,
    )
    traj_xyyaw = torch.tensor(
        [
            [
                [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
                [[2.0, 3.2, 0.0], [4.0, 3.2, 0.0], [6.0, 3.2, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    scores, details = scorer.score_with_details(
        [{"sample_token": token, "scene_id": 200, "frame_idx": 3}],
        traj_xyyaw,
    )

    assert scores.shape == (1, 2)
    assert len(details) == 1
    colliding = details[0]["candidates"][0]
    safe = details[0]["candidates"][1]

    assert "multiplicative_metrics" in colliding
    assert "weighted_metrics" in colliding
    assert "weighted_score" in colliding
    assert "multiplicative_product" in colliding
    assert "traffic_light_compliance" not in colliding["multiplicative_metrics"]
    assert colliding["multiplicative_metrics"]["no_collision"] == pytest.approx(0.0)
    assert safe["multiplicative_metrics"]["no_collision"] == pytest.approx(1.0)
    assert colliding["score"] == pytest.approx(colliding["weighted_score"] * colliding["multiplicative_product"])
    assert safe["score"] == pytest.approx(safe["weighted_score"] * safe["multiplicative_product"])
    assert float(scores[0, 1]) > float(scores[0, 0])


def test_score_with_details_uses_continuous_ttc_for_projected_future_risk(tmp_path: Path) -> None:
    token = "tok-ttc"
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
            ],
            dtype=np.float32,
        ),
        gt_boxes=np.asarray(
            [
                [10.5, 0.0, 0.0, 2.0, 4.0, 1.8, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    _write_scene_env_cache(
        cache_root,
        scene_id=201,
        frame_idx=1,
        payload={
            "sample_token": token,
            "ego_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "drivable_polygons": [
                [[-2.0, -3.0], [16.0, -3.0], [16.0, 3.0], [-2.0, 3.0], [-2.0, -3.0]],
            ],
            "lanes_centerlines": [
                [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]],
            ],
            "static_objects": [],
            "dynamic_objects": [],
        },
    )

    scorer = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        scene_cache_root=cache_root,
    )
    traj_xyyaw = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    scores, details = scorer.score_with_details(
        [{"sample_token": token, "scene_id": 201, "frame_idx": 1}],
        traj_xyyaw,
    )

    risky = details[0]["candidates"][0]
    safe = details[0]["candidates"][1]

    assert scores.shape == (1, 2)
    assert 0.0 < risky["weighted_metrics"]["ttc"] < 1.0
    assert risky["ttc_earliest_risk_time_s"] == pytest.approx(0.3, abs=1.0e-4)
    assert safe["weighted_metrics"]["ttc"] == pytest.approx(1.0)
    assert safe["ttc_earliest_risk_time_s"] == pytest.approx(math.inf)


def test_score_with_details_uses_continuous_driving_direction_gate(tmp_path: Path) -> None:
    token = "tok-direction"
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
                [0.0, 2.0],
            ],
            dtype=np.float32,
        ),
    )
    _write_scene_env_cache(
        cache_root,
        scene_id=202,
        frame_idx=4,
        payload={
            "sample_token": token,
            "ego_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "drivable_polygons": [
                [[-6.0, -3.0], [8.0, -3.0], [8.0, 3.0], [-6.0, 3.0], [-6.0, -3.0]],
            ],
            "lanes_centerlines": [
                [[-6.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [8.0, 0.0]],
            ],
            "static_objects": [],
            "dynamic_objects": [],
        },
    )

    scorer = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        scene_cache_root=cache_root,
    )
    traj_xyyaw = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                [[-1.0, 0.0, math.pi], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[-1.0, 0.0, math.pi], [-2.0, 0.0, math.pi], [-3.0, 0.0, math.pi], [-4.0, 0.0, math.pi], [-5.0, 0.0, math.pi]],
            ]
        ],
        dtype=torch.float32,
    )

    _, details = scorer.score_with_details(
        [{"sample_token": token, "scene_id": 202, "frame_idx": 4}],
        traj_xyyaw,
    )

    forward = details[0]["candidates"][0]
    brief_reverse = details[0]["candidates"][1]
    sustained_reverse = details[0]["candidates"][2]

    assert forward["multiplicative_metrics"]["driving_direction"] == pytest.approx(1.0)
    assert brief_reverse["multiplicative_metrics"]["driving_direction"] == pytest.approx(1.0)
    assert brief_reverse["driving_direction_oncoming_progress_m"] == pytest.approx(0.0, abs=1.0e-4)
    assert 0.0 < sustained_reverse["multiplicative_metrics"]["driving_direction"] < 1.0
    assert sustained_reverse["driving_direction_oncoming_progress_m"] == pytest.approx(4.0, abs=1.0e-4)


def test_score_with_details_can_disable_driving_direction_gate(tmp_path: Path) -> None:
    token = "tok-driving-direction-disabled"
    token2vad_path = tmp_path / "token2vad.pkl"
    cache_root = tmp_path / "scene_cache"
    _write_token2vad(
        token2vad_path,
        token=token,
        gt_ego_fut_trajs=np.asarray(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    _write_scene_env_cache(
        cache_root,
        scene_id=203,
        frame_idx=0,
        payload={
            "sample_token": token,
            "ego_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "drivable_polygons": [
                [[-8.0, -4.0], [8.0, -4.0], [8.0, 4.0], [-8.0, 4.0], [-8.0, -4.0]],
            ],
            "lanes_centerlines": [
                [[-6.0, 0.0], [-3.0, 0.0], [0.0, 0.0], [3.0, 0.0], [6.0, 0.0]],
            ],
            "static_objects": [],
            "dynamic_objects": [],
        },
    )

    traj_xyyaw = torch.tensor(
        [
            [
                [[-1.0, 0.0, math.pi], [-2.0, 0.0, math.pi], [-3.0, 0.0, math.pi], [-4.0, 0.0, math.pi], [-5.0, 0.0, math.pi]],
            ]
        ],
        dtype=torch.float32,
    )

    scorer_default = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        scene_cache_root=cache_root,
    )
    default_scores, default_details = scorer_default.score_with_details(
        [{"sample_token": token, "scene_id": 203, "frame_idx": 0}],
        traj_xyyaw,
    )

    scorer_disabled = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        scene_cache_root=cache_root,
        driving_direction_gate_enabled=False,
    )
    disabled_scores, disabled_details = scorer_disabled.score_with_details(
        [{"sample_token": token, "scene_id": 203, "frame_idx": 0}],
        traj_xyyaw,
    )

    default_candidate = default_details[0]["candidates"][0]
    disabled_candidate = disabled_details[0]["candidates"][0]

    assert default_candidate["multiplicative_metrics"]["driving_direction"] < 1.0
    assert disabled_candidate["multiplicative_metrics"]["driving_direction"] == pytest.approx(1.0)
    assert float(disabled_scores[0, 0]) > float(default_scores[0, 0])
