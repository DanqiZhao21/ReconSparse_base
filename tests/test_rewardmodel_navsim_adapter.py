from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from framework.rewardmodel.data.navsim_cache_builder import (
    build_reward_sample_from_raw,
    build_reward_sample_from_scene_data,
    build_scene_specific_candidates,
    build_vocabulary_from_gt_trajectories,
    default_ego_state_features_from_scene,
    extract_observation_tensor_from_scene,
    image_paths_from_scene,
    load_candidate_trajectories,
    candidate_prefix_trajectories_for_horizon,
    score_candidates_with_navsim_pdm,
    score_candidates_with_navsim_pdm_dense,
)
from framework.rewardmodel.data.build_navsim_reward_cache import parse_args as parse_reward_cache_args
from framework.rewardmodel.data.build_navsim_reward_cache import build_sensor_config_for_cameras
from framework.rewardmodel.data.build_navsim_reward_cache import load_token_list
from framework.rewardmodel.data.build_navsim_reward_cache import scene_filter_scope_from_metric_cache
from framework.rewardmodel.data.build_navsim_reward_cache import split_tokens_for_workers
from framework.rewardmodel.supervision.teacher_adapter import (
    map_pdm_metric_names,
    stack_temporal_metric_targets,
)


def test_map_pdm_metric_names_converts_navsim_names_to_internal_schema() -> None:
    pdm_metrics = {
        "no_at_fault_collisions": np.asarray([1.0, 0.0], dtype=np.float32),
        "drivable_area_compliance": np.asarray([0.8, 0.7], dtype=np.float32),
        "driving_direction_compliance": np.asarray([0.9, 0.95], dtype=np.float32),
        "traffic_light_compliance": np.asarray([1.0, 1.0], dtype=np.float32),
        "ego_progress": np.asarray([0.6, 0.9], dtype=np.float32),
        "time_to_collision_within_bound": np.asarray([0.7, 0.8], dtype=np.float32),
        "lane_keeping": np.asarray([0.5, 0.55], dtype=np.float32),
        "history_comfort": np.asarray([0.4, 0.45], dtype=np.float32),
    }

    mapped = map_pdm_metric_names(pdm_metrics)

    assert set(mapped.keys()) == {"rnc", "rdac", "rddc", "rtlc", "rep", "rttc", "rlk", "rhc"}
    assert np.allclose(mapped["rnc"], np.asarray([1.0, 0.0], dtype=np.float32))
    assert np.allclose(mapped["rhc"], np.asarray([0.4, 0.45], dtype=np.float32))


def test_stack_temporal_metric_targets_builds_candidate_horizon_metric_tensor() -> None:
    metric_scores_per_horizon = [
        {
            "rnc": np.asarray([1.0, 0.0], dtype=np.float32),
            "rdac": np.asarray([0.9, 0.8], dtype=np.float32),
            "rddc": np.asarray([0.95, 0.7], dtype=np.float32),
            "rtlc": np.asarray([1.0, 1.0], dtype=np.float32),
            "rep": np.asarray([0.5, 0.6], dtype=np.float32),
            "rttc": np.asarray([0.7, 0.2], dtype=np.float32),
            "rlk": np.asarray([0.8, 0.9], dtype=np.float32),
            "rhc": np.asarray([0.9, 0.95], dtype=np.float32),
        },
        {
            "rnc": np.asarray([1.0, 0.0], dtype=np.float32),
            "rdac": np.asarray([0.85, 0.75], dtype=np.float32),
            "rddc": np.asarray([0.9, 0.65], dtype=np.float32),
            "rtlc": np.asarray([1.0, 0.95], dtype=np.float32),
            "rep": np.asarray([0.55, 0.62], dtype=np.float32),
            "rttc": np.asarray([0.72, 0.25], dtype=np.float32),
            "rlk": np.asarray([0.82, 0.92], dtype=np.float32),
            "rhc": np.asarray([0.88, 0.94], dtype=np.float32),
        },
    ]

    targets = stack_temporal_metric_targets(metric_scores_per_horizon)

    assert targets.shape == (2, 2, 8)
    assert np.allclose(targets[0, 0], np.asarray([1.0, 0.9, 0.95, 1.0, 0.5, 0.7, 0.8, 0.9], dtype=np.float32))


def test_build_reward_sample_from_raw_creates_training_example() -> None:
    sample = build_reward_sample_from_raw(
        image_paths=["/tmp/cam_f0.jpg", "/tmp/cam_l0.jpg"],
        ego_states=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        candidate_trajectories=np.zeros((4, 8, 3), dtype=np.float32),
        targets=np.ones((4, 2, 8), dtype=np.float32),
        token="sample-token",
        valid_mask=None,
    )

    assert sample["token"] == "sample-token"
    assert sample["image_paths"] == ["/tmp/cam_f0.jpg", "/tmp/cam_l0.jpg"]
    assert "observations" not in sample
    assert tuple(sample["candidate_trajectories"].shape) == (4, 8, 3)
    assert tuple(sample["targets"].shape) == (4, 2, 8)
    assert torch.all(sample["valid_mask"])


def test_extract_observation_tensor_from_scene_stacks_requested_cameras() -> None:
    class DummyCam:
        def __init__(self, value: float) -> None:
            self.image = np.full((4, 5, 3), value, dtype=np.float32)

    class DummyCameras:
        cam_f0 = DummyCam(1.0)
        cam_l0 = DummyCam(2.0)

    class DummyFrame:
        cameras = DummyCameras()

    class DummyScene:
        frames = [DummyFrame()]

    obs = extract_observation_tensor_from_scene(DummyScene(), camera_names=("cam_f0", "cam_l0"))

    assert obs.shape == (6, 4, 5)
    assert np.allclose(obs[:3], 1.0)
    assert np.allclose(obs[3:], 2.0)


def test_default_ego_state_features_from_scene_uses_last_history_frame() -> None:
    class DummyEgoStatus:
        def __init__(self, pose, vel, acc, cmd) -> None:
            self.ego_pose = np.asarray(pose, dtype=np.float32)
            self.ego_velocity = np.asarray(vel, dtype=np.float32)
            self.ego_acceleration = np.asarray(acc, dtype=np.float32)
            self.driving_command = np.asarray(cmd, dtype=np.float32)

    class DummyFrame:
        def __init__(self, ego_status) -> None:
            self.ego_status = ego_status

    class DummyScene:
        frames = [
            DummyFrame(DummyEgoStatus([0, 0, 0], [0, 0], [0, 0], [1, 0])),
            DummyFrame(DummyEgoStatus([1, 2, 3], [4, 5], [6, 7], [0, 1])),
        ]

    ego = default_ego_state_features_from_scene(DummyScene())

    assert ego.shape == (9,)
    assert np.allclose(ego, np.asarray([1, 2, 3, 4, 5, 6, 7, 0, 1], dtype=np.float32))


def test_build_reward_sample_from_scene_data_combines_scene_features_and_targets() -> None:
    class DummyCam:
        def __init__(self, value: float, image_path: str) -> None:
            self.image = np.full((2, 2, 3), value, dtype=np.float32)
            self.image_path = image_path

    class DummyCameras:
        cam_f0 = DummyCam(1.0, "/tmp/cam_f0.jpg")

    class DummyEgoStatus:
        ego_pose = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        ego_velocity = np.asarray([4.0, 5.0], dtype=np.float32)
        ego_acceleration = np.asarray([6.0, 7.0], dtype=np.float32)
        driving_command = np.asarray([0.0, 1.0], dtype=np.float32)

    class DummyFrame:
        cameras = DummyCameras()
        ego_status = DummyEgoStatus()

    class DummyScene:
        frames = [DummyFrame()]
        scene_metadata = type("Meta", (), {"initial_token": "scene-token"})()

    sample = build_reward_sample_from_scene_data(
        scene=DummyScene(),
        candidate_trajectories=np.zeros((3, 8, 3), dtype=np.float32),
        targets=np.ones((3, 2, 8), dtype=np.float32),
        camera_names=("cam_f0",),
    )

    assert sample["token"] == "scene-token"
    assert sample["image_paths"] == ["/tmp/cam_f0.jpg"]
    assert "observations" not in sample
    assert tuple(sample["ego_states"].shape) == (9,)


def test_image_paths_from_scene_collects_requested_camera_paths() -> None:
    class DummyCam:
        def __init__(self, image_path: str) -> None:
            self.image_path = image_path

    class DummyCameras:
        cam_f0 = DummyCam("/tmp/f0.jpg")
        cam_l0 = DummyCam("/tmp/l0.jpg")

    class DummyFrame:
        cameras = DummyCameras()

    class DummyScene:
        frames = [DummyFrame()]

    paths = image_paths_from_scene(DummyScene(), camera_names=("cam_f0", "cam_l0"))

    assert paths == ["/tmp/f0.jpg", "/tmp/l0.jpg"]


def test_load_candidate_trajectories_supports_npy_and_npz(tmp_path) -> None:
    candidates = np.zeros((5, 8, 3), dtype=np.float32)
    npy_path = tmp_path / "candidates.npy"
    np.save(npy_path, candidates)
    npz_path = tmp_path / "candidates.npz"
    np.savez(npz_path, trajectories=candidates + 1.0)

    assert np.allclose(load_candidate_trajectories(npy_path), candidates)
    assert np.allclose(load_candidate_trajectories(npz_path), candidates + 1.0)


def test_score_candidates_with_navsim_pdm_maps_teacher_output() -> None:
    candidates = np.zeros((2, 8, 3), dtype=np.float32)

    def fake_pdm_score_fn(**kwargs):
        assert kwargs["model_trajectory"].shape == (2, 8, 3)
        return {
            "no_at_fault_collisions": np.asarray([1.0, 0.0], dtype=np.float32),
            "drivable_area_compliance": np.asarray([0.9, 0.8], dtype=np.float32),
            "driving_direction_compliance": np.asarray([0.7, 0.6], dtype=np.float32),
            "traffic_light_compliance": np.asarray([1.0, 1.0], dtype=np.float32),
            "ego_progress": np.asarray([0.2, 0.3], dtype=np.float32),
            "time_to_collision_within_bound": np.asarray([0.4, 0.5], dtype=np.float32),
            "lane_keeping": np.asarray([0.6, 0.7], dtype=np.float32),
            "history_comfort": np.asarray([0.8, 0.9], dtype=np.float32),
        }

    targets = score_candidates_with_navsim_pdm(
        metric_cache=object(),
        candidate_trajectories=candidates,
        simulator=object(),
        scorer=object(),
        traffic_agents_policy=object(),
        future_sampling=object(),
        pdm_score_fn=fake_pdm_score_fn,
    )

    assert targets.shape == (2, 1, 8)
    assert np.allclose(targets[0, 0], np.asarray([1.0, 0.9, 0.7, 1.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float32))


def test_candidate_prefix_trajectories_for_horizon_holds_future_points_constant() -> None:
    candidates = np.asarray(
        [
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.1],
                [3.0, 0.0, 0.2],
                [4.0, 0.0, 0.3],
            ]
        ],
        dtype=np.float32,
    )

    prefix = candidate_prefix_trajectories_for_horizon(candidates, 1)

    assert np.allclose(prefix[0, 0], candidates[0, 0])
    assert np.allclose(prefix[0, 1], candidates[0, 1])
    assert np.allclose(prefix[0, 2], candidates[0, 1])
    assert np.allclose(prefix[0, 3], candidates[0, 1])


def test_score_candidates_with_navsim_pdm_dense_builds_horizon_targets() -> None:
    candidates = np.zeros((2, 3, 3), dtype=np.float32)
    calls = []

    def fake_pdm_score_fn(**kwargs):
        calls.append(kwargs["model_trajectory"].copy())
        value = float(len(calls))
        return {
            "no_at_fault_collisions": np.full((2,), value, dtype=np.float32),
            "drivable_area_compliance": np.full((2,), value, dtype=np.float32),
            "driving_direction_compliance": np.full((2,), value, dtype=np.float32),
            "traffic_light_compliance": np.full((2,), value, dtype=np.float32),
            "ego_progress": np.full((2,), value, dtype=np.float32),
            "time_to_collision_within_bound": np.full((2,), value, dtype=np.float32),
            "lane_keeping": np.full((2,), value, dtype=np.float32),
            "history_comfort": np.full((2,), value, dtype=np.float32),
        }

    targets = score_candidates_with_navsim_pdm_dense(
        metric_cache=object(),
        candidate_trajectories=candidates,
        simulator=object(),
        scorer=object(),
        traffic_agents_policy=object(),
        future_sampling=object(),
        pdm_score_fn=fake_pdm_score_fn,
        num_horizons=3,
    )

    assert len(calls) == 3
    assert targets.shape == (2, 3, 8)
    assert np.allclose(targets[:, 0, :], 1.0)
    assert np.allclose(targets[:, 1, :], 2.0)
    assert np.allclose(targets[:, 2, :], 3.0)


def test_build_navsim_reward_cache_cli_parses_required_args(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_navsim_reward_cache.py",
            "--metric-cache-path",
            "/tmp/metric_cache",
            "--candidate-path",
            "/tmp/candidates.npy",
            "--output-root",
            "/tmp/reward_cache",
        ],
    )
    args = parse_reward_cache_args()
    assert args.metric_cache_path == "/tmp/metric_cache"
    assert args.candidate_path == "/tmp/candidates.npy"
    assert args.output_root == "/tmp/reward_cache"


def test_build_navsim_reward_cache_cli_accepts_gt_vocabulary_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_navsim_reward_cache.py",
            "--metric-cache-path",
            "/tmp/metric_cache",
            "--output-root",
            "/tmp/reward_cache",
            "--build-vocabulary-from-gt",
        ],
    )
    args = parse_reward_cache_args()
    assert args.build_vocabulary_from_gt is True


def test_build_navsim_reward_cache_cli_accepts_num_processes(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_navsim_reward_cache.py",
            "--metric-cache-path",
            "/tmp/metric_cache",
            "--output-root",
            "/tmp/reward_cache",
            "--build-vocabulary-from-gt",
            "--num-processes",
            "32",
        ],
    )
    args = parse_reward_cache_args()
    assert args.num_processes == 32


def test_build_navsim_reward_cache_cli_accepts_token_list_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_navsim_reward_cache.py",
            "--metric-cache-path",
            "/tmp/metric_cache",
            "--output-root",
            "/tmp/reward_cache",
            "--build-vocabulary-from-gt",
            "--token-list-path",
            "/tmp/tokens.txt",
        ],
    )
    args = parse_reward_cache_args()
    assert args.token_list_path == "/tmp/tokens.txt"


def test_load_token_list_strips_blank_lines(tmp_path: Path) -> None:
    token_path = tmp_path / "tokens.txt"
    token_path.write_text(" token-a\n\n token-b \n")

    assert load_token_list(token_path) == ["token-a", "token-b"]


def test_split_tokens_for_workers_round_robins_tokens() -> None:
    shards = split_tokens_for_workers(["a", "b", "c", "d", "e"], 3)

    assert shards == [["a", "d"], ["b", "e"], ["c"]]


def test_build_sensor_config_for_cameras_loads_only_requested_cameras() -> None:
    class DummySensorConfig:
        def __init__(self) -> None:
            self.cam_f0 = False
            self.cam_l0 = False
            self.cam_r0 = False
            self.lidar_pc = False

        @classmethod
        def build_no_sensors(cls):
            return cls()

    sensor_config = build_sensor_config_for_cameras(DummySensorConfig, ("cam_f0", "cam_l0"), include=[3])

    assert sensor_config.cam_f0 == [3]
    assert sensor_config.cam_l0 == [3]
    assert sensor_config.cam_r0 is False
    assert sensor_config.lidar_pc is False


def test_build_sensor_config_for_cameras_rejects_unknown_camera() -> None:
    class DummySensorConfig:
        @classmethod
        def build_no_sensors(cls):
            return cls()

    try:
        build_sensor_config_for_cameras(DummySensorConfig, ("cam_x0",), include=[3])
    except ValueError as exc:
        assert "Unknown NavSim camera name" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_scene_filter_scope_from_metric_cache_collects_tokens_and_logs() -> None:
    metric_cache_paths = {
        "token-a": Path("/root/cache/log_a/unknown/x/metric_cache.pkl"),
        "token-b": Path("/root/cache/log_b/unknown/y/metric_cache.pkl"),
    }

    log_names, tokens = scene_filter_scope_from_metric_cache(metric_cache_paths)

    assert log_names == ["log_a", "log_b"]
    assert tokens == ["token-a", "token-b"]


def test_build_vocabulary_from_gt_trajectories_stacks_and_caps_samples() -> None:
    gt_trajectories = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.5, 0.1]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0], [3.0, 1.0, 0.2]], dtype=np.float32),
    ]

    vocabulary = build_vocabulary_from_gt_trajectories(gt_trajectories, max_vocabulary_size=2)

    assert vocabulary.shape == (2, 2, 3)
    assert np.allclose(vocabulary[0], gt_trajectories[0])
    assert not np.allclose(vocabulary[1], gt_trajectories[1])


def test_build_scene_specific_candidates_filters_vocabulary_with_gt_future() -> None:
    class DummyFuture:
        poses = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

    class DummyScene:
        def get_future_trajectory(self):
            return DummyFuture()

    vocabulary = np.asarray(
        [
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [25.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 6.0, 0.0]],
        ],
        dtype=np.float32,
    )

    filtered = build_scene_specific_candidates(
        scene=DummyScene(),
        vocabulary=vocabulary,
        max_longitudinal_error_m=10.0,
        max_lateral_error_m=5.0,
        max_heading_error_rad=np.deg2rad(20.0),
        max_candidates=16,
    )

    assert filtered.shape == (1, 2, 3)
    assert np.allclose(filtered[0], vocabulary[0])
