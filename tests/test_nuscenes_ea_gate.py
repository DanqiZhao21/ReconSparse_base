from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils
from framework.utils.nuscenes_agent_state_cache import (
    build_scene_agent_state_cache_from_processed_scene,
)


def _write_token2vad(
    path: Path,
    *,
    token: str,
    gt_ego_fut_trajs: np.ndarray,
    gt_boxes: np.ndarray | None = None,
    gt_velocity: np.ndarray | None = None,
    gt_names: np.ndarray | None = None,
    valid_flag: np.ndarray | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    gt_boxes_arr = np.asarray(gt_boxes, dtype=np.float32) if gt_boxes is not None else np.zeros((0, 7), dtype=np.float32)
    num_agents = int(gt_boxes_arr.shape[0])
    payload = {
        token: {
            "token": token,
            "ego2global_translation": [0.0, 0.0, 0.0],
            "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
            "gt_ego_fut_trajs": np.asarray(gt_ego_fut_trajs, dtype=np.float32),
            "gt_boxes": gt_boxes_arr,
            "gt_velocity": (
                np.asarray(gt_velocity, dtype=np.float32)
                if gt_velocity is not None
                else np.zeros((num_agents, 2), dtype=np.float32)
            ),
            "gt_names": (
                np.asarray(gt_names, dtype=object)
                if gt_names is not None
                else np.asarray(["vehicle.car"] * num_agents, dtype=object)
            ),
            "valid_flag": (
                np.asarray(valid_flag, dtype=bool)
                if valid_flag is not None
                else np.ones((num_agents,), dtype=bool)
            ),
        }
    }
    if extra_fields:
        payload[token].update(extra_fields)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _write_agent_state_cache(root: Path, *, scene_id: int, frame_idx: int, agents: list[dict[str, object]]) -> None:
    scene_dir = root / f"{int(scene_id):03d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    path = scene_dir / "agent_state_cache.json"
    path.write_text(json.dumps({str(int(frame_idx)): {"agents": agents}}, indent=2), encoding="utf-8")


def test_score_with_ea_gate_is_optional_and_applies_multiplicative_metric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token = "tok-ea-gate"
    token2vad_path = tmp_path / "token2vad.pkl"
    agent_cache_root = tmp_path / "agent_state_cache"
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
    )
    _write_agent_state_cache(
        agent_cache_root,
        scene_id=8,
        frame_idx=0,
        agents=[
            {
                "instance_token": "veh-1",
                "category": "vehicle.car",
                "center_xy": [4.0, 0.0],
                "yaw_rad": 0.0,
                "yaw_rate_rps": 0.0,
                "velocity_xy": [0.0, 0.0],
                "speed_mps": 0.0,
                "length_m": 4.8,
                "width_m": 2.0,
            }
        ],
    )

    traj_xyyaw = torch.tensor(
        [
            [
                [[2.0, 0.0, 0.0], [3.5, 0.0, 0.0], [4.5, 0.0, 0.0]],
                [[1.0, 2.5, 0.0], [2.0, 2.5, 0.0], [3.0, 2.5, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    scorer_plain = NuScenesScorerUtils(token2vad_path=token2vad_path)
    _, plain_details = scorer_plain.score_with_details([{"sample_token": token, "scene_id": 8, "frame_idx": 0}], traj_xyyaw)
    assert "ea_safety" not in plain_details[0]["candidates"][0]["multiplicative_metrics"]

    scorer = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        agent_state_cache_root=agent_cache_root,
        ea_gate_enabled=True,
        ea_gate_good_threshold=0.0,
        ea_gate_bad_threshold=5.0,
    )

    monkeypatch.setattr(
        scorer,
        "_compute_ea_value_for_pair",
        lambda ego_state, agent_state: 4.0 if float(ego_state["x"]) > 1.5 and abs(float(ego_state["y"])) < 0.5 else 0.0,
        raising=False,
    )

    scores, details = scorer.score_with_details([{"sample_token": token, "scene_id": 8, "frame_idx": 0}], traj_xyyaw)

    risky = details[0]["candidates"][0]
    safe = details[0]["candidates"][1]
    assert risky["multiplicative_metrics"]["ea_safety"] == pytest.approx(0.2)
    assert safe["multiplicative_metrics"]["ea_safety"] == pytest.approx(1.0)
    assert float(scores[0, 1]) > float(scores[0, 0])


def test_build_scene_agent_state_cache_from_processed_scene_computes_yawrate_and_velocity(
    tmp_path: Path,
) -> None:
    processed_scene_dir = tmp_path / "processed_10Hz" / "trainval" / "001" / "instances"
    processed_scene_dir.mkdir(parents=True, exist_ok=True)
    (processed_scene_dir / "frame_instances.json").write_text(
        json.dumps({"0": ["veh-a"], "1": ["veh-a"], "2": ["veh-a"]}, indent=2),
        encoding="utf-8",
    )
    (processed_scene_dir / "instances_info.json").write_text(
        json.dumps(
            {
                "veh-a": {
                    "id": "veh-a",
                    "class_name": "vehicle.car",
                    "frame_annotations": {
                        "frame_idx": [0, 1, 2],
                        "box_size": [[4.8, 2.0, 1.7], [4.8, 2.0, 1.7], [4.8, 2.0, 1.7]],
                        "obj_to_world": [
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                            [
                                [0.9950041653, -0.0998334166, 0.0, 1.0],
                                [0.0998334166, 0.9950041653, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                            [
                                [0.9800665778, -0.1986693308, 0.0, 2.0],
                                [0.1986693308, 0.9800665778, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        ],
                    },
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cache = build_scene_agent_state_cache_from_processed_scene(
        scene_id=1,
        processed_scene_dir=processed_scene_dir.parent,
        out_root=tmp_path / "cache_out",
        fps=10.0,
    )

    assert 1 in cache
    agents = cache[1]["agents"]
    assert len(agents) == 1
    agent = agents[0]
    assert agent["instance_token"] == "veh-a"
    assert agent["speed_mps"] == pytest.approx(10.0, rel=1.0e-3)
    assert agent["yaw_rate_rps"] == pytest.approx(1.0, rel=1.0e-2)


def test_score_with_ea_gate_prefers_agent_future_truth_over_ctrv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token = "tok-ea-future"
    token2vad_path = tmp_path / "token2vad.pkl"
    agent_cache_root = tmp_path / "agent_state_cache"
    _write_token2vad(
        token2vad_path,
        token=token,
        gt_ego_fut_trajs=np.asarray(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        gt_boxes=np.asarray(
            [
                [0.0, 0.0, 0.0, 2.0, 4.0, 1.8, 0.0],
            ],
            dtype=np.float32,
        ),
        gt_velocity=np.asarray([[10.0, 0.0]], dtype=np.float32),
        extra_fields={
            "gt_agent_fut_trajs": np.asarray([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], dtype=np.float32),
            "gt_agent_fut_masks": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
            "gt_agent_fut_yaw": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        },
    )
    _write_agent_state_cache(
        agent_cache_root,
        scene_id=12,
        frame_idx=0,
        agents=[
            {
                "instance_token": "veh-1",
                "category": "vehicle.car",
                "center_xy": [0.0, 0.0],
                "yaw_rad": 0.0,
                "yaw_rate_rps": 0.0,
                "velocity_xy": [10.0, 0.0],
                "speed_mps": 10.0,
                "length_m": 4.8,
                "width_m": 2.0,
            }
        ],
    )

    scorer = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        agent_state_cache_root=agent_cache_root,
        scene_cache_root=tmp_path / "scene_cache",
        ea_gate_enabled=True,
        ea_gate_good_threshold=0.0,
        ea_gate_bad_threshold=20.0,
    )

    monkeypatch.setattr(
        scorer,
        "_compute_ea_value_for_pair",
        lambda ego_state, agent_state: float(agent_state["x"]),
        raising=False,
    )

    traj_xyyaw = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    _, details = scorer.score_with_details([{"sample_token": token, "scene_id": 12, "frame_idx": 0}], traj_xyyaw)

    candidate = details[0]["candidates"][0]
    assert candidate["ea_gate_max_ea"] == pytest.approx(3.0, abs=1.0e-4)
    assert candidate["multiplicative_metrics"]["ea_safety"] == pytest.approx(0.85, abs=1.0e-4)


def test_sample_state_at_time_interpolates_future_series_and_clamps_out_of_range() -> None:
    sampled = NuScenesScorerUtils._sample_state_at_time(
        current_state={
            "x": 0.0,
            "y": 0.0,
            "yaw_rad": 0.0,
            "length_m": 4.9,
            "width_m": 2.1,
        },
        future_xy=np.asarray([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]], dtype=np.float32),
        future_yaw=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        time_s=0.75,
        dt_s=0.5,
    )
    assert sampled is not None
    assert sampled["x"] == pytest.approx(2.0, abs=1.0e-4)
    assert sampled["speed_mps"] == pytest.approx(4.0, abs=1.0e-4)

    assert (
        NuScenesScorerUtils._sample_state_at_time(
            current_state={"x": 0.0, "y": 0.0, "yaw_rad": 0.0},
            future_xy=np.asarray([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]], dtype=np.float32),
            future_yaw=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            time_s=2.1,
            dt_s=0.5,
        )
        is None
    )
