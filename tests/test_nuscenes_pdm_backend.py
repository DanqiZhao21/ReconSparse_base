from __future__ import annotations

from pathlib import Path

import pickle

import numpy as np
import torch

from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy


def _write_token2vad(path: Path) -> None:
    payload = {
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
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_policy_uses_nuscenes_pdm_backend_when_configured(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._nuscenes_scorer_config = {"backend": "nuscenes_pdm"}
    policy._nuscenes_token_scorer = None
    policy._nuscenes_pdm_scorer = None

    class DummyBackend:
        pass

    monkeypatch.setattr(
        "framework.algorithms.nuscenes_pdm_backend.NuScenesPDMScorer",
        lambda **kwargs: DummyBackend(),
    )

    scorer = policy._ensure_counterfactual_scorer_backend()

    assert isinstance(scorer, DummyBackend)
    assert policy._nuscenes_pdm_scorer is scorer


def test_policy_pdm_score_hook_accepts_numpy_backend_scores(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device = torch.device("cpu")

    class DummyBackend:
        def score(self, replays, traj_xyyaw):
            del replays, traj_xyyaw
            return np.asarray([[0.1, 0.2]], dtype=np.float32)

    monkeypatch.setattr(policy, "_ensure_counterfactual_scorer_backend", lambda: DummyBackend())
    monkeypatch.setattr(SparseDriveV2Policy, "device", property(lambda self: self._device))

    scores = policy.pdm_score_counterfactuals_from_replay_batch(
        [{"sample_token": "tok-a"}],
        torch.zeros((1, 2, 3, 3), dtype=torch.float32),
    )

    assert torch.is_tensor(scores)
    assert scores.dtype == torch.float32
    assert tuple(scores.shape) == (1, 2)


def test_nuscenes_pdm_backend_returns_batch_candidate_scores(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.7, 0.4, 0.0]],
                [[0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.4, 0.3, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    scores = scorer.score([{"sample_token": "tok-a"}], traj_xyyaw)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1, 2)


def test_nuscenes_pdm_backend_builds_sample_context_once_per_sample(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    calls = {"build_sample_context": 0}
    orig_build = scorer._build_sample_context
    delegate_calls = {"static_ctx": 0}

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        delegate_calls["static_ctx"] += 1
        return {
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [],
            "ea_agent_states": [],
        }

    def wrapped_build(replay, *, patch_radius: float):
        calls["build_sample_context"] += 1
        return orig_build(replay, patch_radius=patch_radius)

    monkeypatch.setattr(scorer, "_build_sample_context", wrapped_build)
    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)
    monkeypatch.setattr(
        scorer._delegate,
        "score",
        lambda replays, traj_xyyaw: np.zeros((len(replays), int(traj_xyyaw.shape[1])), dtype=np.float32),
    )

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.7, 0.4, 0.0]],
                [[0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.4, 0.3, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    replays = [{"sample_token": "tok-a"}]

    scores_0 = scorer.score(replays, traj_xyyaw)
    scores_1 = scorer.score(replays, traj_xyyaw)

    assert scores_0.shape == (1, 2)
    assert np.allclose(scores_0, scores_1)
    assert calls["build_sample_context"] == 2
    assert delegate_calls["static_ctx"] == 1


def test_nuscenes_pdm_backend_reuses_persisted_derived_context_across_instances(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    scene_cache_root = tmp_path / "scene_cache"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer_first = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root,
    )

    delegate_calls = {"static_ctx": 0}

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        delegate_calls["static_ctx"] += 1
        return {
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {
                    "drivable_area": [
                        [[-1.0, -1.0], [3.0, -1.0], [3.0, 3.0], [-1.0, 3.0]],
                    ],
                    "lane_centerline": [
                        [[0.0, 0.0], [2.0, 0.0]],
                    ],
                },
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [1.0, 2.0],
                    "velocity_xy": [0.2, 0.0],
                    "yaw_rad": 0.1,
                    "length_m": 4.5,
                    "width_m": 1.8,
                },
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer_first._delegate, "_build_static_sample_context", fake_static_context)

    ctx0 = scorer_first._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    scorer_second = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root,
    )

    def should_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("expected second scorer to reuse persisted derived pdm context")

    monkeypatch.setattr(scorer_second._delegate, "_build_static_sample_context", should_not_run)

    ctx1 = scorer_second._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    assert delegate_calls["static_ctx"] == 1
    assert list(ctx0.object_tokens) == list(ctx1.object_tokens)
    assert ctx1.centerline_segments_xy.shape == (1, 2, 2)
    assert ctx1.drivable_map.batch_contains_points(np.zeros((1, 1, 2), dtype=np.float32)).shape == (1, 1)


def test_nuscenes_pdm_backend_builds_sample_occupancy_context_once_per_sample(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    delegate_calls = {"static_ctx": 0}

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        delegate_calls["static_ctx"] += 1
        return {
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [1.0, 2.0],
                    "velocity_xy": [0.2, 0.0],
                    "yaw_rad": 0.1,
                    "length_m": 4.5,
                    "width_m": 1.8,
                },
                {
                    "category": "vehicle.bus",
                    "center_xy": [4.0, -1.5],
                    "velocity_xy": [0.0, -0.1],
                    "yaw_rad": -0.2,
                    "length_m": 10.0,
                    "width_m": 2.5,
                },
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    replay = {"sample_token": "tok-a"}
    ctx0 = scorer._build_sample_context(replay, patch_radius=20.0)
    ctx1 = scorer._build_sample_context(replay, patch_radius=20.0)

    assert ctx0 is ctx1
    assert delegate_calls["static_ctx"] == 1
    assert list(ctx0.object_tokens) == ["obj-0", "obj-1"]
    assert ctx0.object_velocity_xy.shape == (2, 2)
    assert ctx0.object_polygons.shape == (2,)
    assert len(ctx0.occupancy_map) == 2
    hits = ctx0.occupancy_map.intersects(ctx0.object_polygons[0])
    assert "obj-0" in hits


def test_nuscenes_pdm_backend_builds_batched_candidate_polygon_arrays(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [2.0, 0.3, 0.2]],
                [[0.2, 0.0, 0.0], [0.8, 0.4, 0.1], [1.5, 0.9, 0.2]],
            ]
        ],
        dtype=torch.float32,
    )

    batch = scorer._build_candidate_geometry_batch(traj_xyyaw)

    assert batch["corners_xy"].shape == (1, 2, 3, 4, 2)
    assert batch["polygons"].shape == (1, 2, 3)
    assert batch["centers_xy"].shape == (1, 2, 3, 2)
    assert batch["yaw_rad"].shape == (1, 2, 3)
    first_polygon = batch["polygons"][0, 0, 0]
    assert first_polygon is not None
    assert float(first_polygon.area) > 0.0


def test_nuscenes_pdm_backend_collision_ttc_uses_batch_query_shapes(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)

    query_shapes: list[tuple[int, ...]] = []

    class RecordingOccupancyMap:
        def __len__(self) -> int:
            return 1

        def query(self, geometry, predicate=None):
            del predicate
            query_shapes.append(np.asarray(geometry, dtype=object).shape)
            return np.zeros((2, 0), dtype=np.int64)

        def intersects(self, geometry):
            del geometry
            return []

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.occupancy_map = RecordingOccupancyMap()

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (2,)
    assert metrics["ttc"].shape == (2,)
    assert query_shapes
    assert all(len(shape) == 1 for shape in query_shapes)
    assert max(shape[0] for shape in query_shapes) == 2


def test_nuscenes_pdm_backend_prebuilds_ttc_projection_polygon_arrays(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    projections = scorer._build_ttc_projection_geometry(
        centers_xy=geometry["centers_xy"][0],
        yaw_rad=geometry["yaw_rad"][0],
        dt_s=0.5,
    )

    assert projections["centers_xy"].shape == (2, 3, 3, 2)
    assert projections["corners_xy"].shape == (2, 3, 3, 4, 2)
    assert projections["polygons"].shape == (2, 3, 3)
    assert projections["offsets_s"].shape == (3,)


def test_nuscenes_pdm_backend_batch_drivable_and_lane_queries_return_expected_shapes(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
                [[0.0, 3.0, 0.0], [0.8, 3.0, 0.0], [1.6, 3.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)

    class FakeDrivableMap:
        def __init__(self):
            self.calls: list[tuple[int, ...]] = []

        def batch_contains_points(self, points_xy):
            self.calls.append(tuple(points_xy.shape))
            return np.asarray(
                [
                    [True, True, True],
                    [False, False, False],
                ],
                dtype=bool,
            )

    fake_map = FakeDrivableMap()
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.drivable_map = fake_map
    sample_context.centerline_segments_xy = np.asarray(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    sample_context.centerline_tangents_xy = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = scorer._batch_map_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["drivable_area"].shape == (2,)
    assert metrics["lane_keeping"].shape == (2,)
    assert metrics["driving_direction"].shape == (2,)
    assert fake_map.calls == [(2, 3, 2)]
