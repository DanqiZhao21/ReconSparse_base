from __future__ import annotations

from pathlib import Path

import pickle

import numpy as np
import pytest
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


def test_nuscenes_pdm_backend_builds_candidate_geometry_in_candidate_batch(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.1], [0.7, 0.4, 0.2]],
                [[0.1, 0.0, -0.1], [0.2, 0.1, 0.0], [0.4, 0.3, 0.1]],
            ]
        ],
        dtype=torch.float32,
    )

    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)

    assert geometry["centers_xy"].shape == (1, 2, 3, 2)
    assert geometry["yaw_rad"].shape == (1, 2, 3)
    assert geometry["corners_xy"].shape == (1, 2, 3, 4, 2)
    assert geometry["polygons"].shape == (1, 2, 3)
    assert geometry["polygons"].dtype == object


def test_nuscenes_pdm_drivable_map_contains_points_in_candidate_batch() -> None:
    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMDrivableMap

    drivable_map = NuScenesPDMDrivableMap(
        polygons_xy=[
            np.asarray([[-1.0, -1.0], [3.0, -1.0], [3.0, 3.0], [-1.0, 3.0]], dtype=np.float32),
            np.asarray([[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0]], dtype=np.float32),
        ]
    )
    points_xy = np.asarray(
        [
            [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]],
            [[10.5, 10.5], [0.5, 0.5], [20.0, 20.0]],
        ],
        dtype=np.float32,
    )

    inside = drivable_map.batch_contains_points(points_xy)

    assert inside.shape == (2, 3)
    assert inside.dtype == bool
    assert inside.tolist() == [[True, True, False], [True, True, False]]


def test_nuscenes_pdm_backend_builds_ttc_projection_geometry_in_candidate_batch(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    centers_xy = np.asarray(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    yaw_rad = np.zeros((2, 3), dtype=np.float32)

    projection = scorer._build_ttc_projection_geometry(
        centers_xy=centers_xy,
        yaw_rad=yaw_rad,
        dt_s=0.5,
    )

    assert projection["centers_xy"].shape[:3] == (2, 3, len(projection["offsets_s"]))
    assert projection["corners_xy"].shape == (2, 3, len(projection["offsets_s"]), 4, 2)
    assert projection["polygons"].shape == (2, 3, len(projection["offsets_s"]))
    assert projection["polygons"].dtype == object


def test_nuscenes_pdm_backend_query_hits_accepts_batched_polygons(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    scene_objects = [
        {
            "token": "obj-a",
            "corners_xy": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            "velocity_xy": [0.0, 0.0],
        }
    ]
    _, _, _, occupancy_map = scorer._build_object_geometry_arrays(scene_objects)
    polygons_at_step = scorer._build_candidate_geometry_batch(
        torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
                    [[3.0, 3.0, 0.0], [4.0, 4.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        )
    )["polygons"][0]

    hits = scorer._query_hits_per_candidate(occupancy_map, polygons_at_step[:, 0], predicate="intersects")

    assert hits.shape == (2,)
    assert hits.dtype == bool
    assert hits.tolist() == [True, False]


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
    assert sorted(shape[0] for shape in query_shapes) == [6, 18]


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
            if tuple(points_xy.shape) == (2, 3, 4, 2):
                return np.asarray(
                    [
                        [
                            [True, True, True, True],
                            [True, True, True, True],
                            [True, True, True, True],
                        ],
                        [
                            [False, False, False, False],
                            [False, False, False, False],
                            [False, False, False, False],
                        ],
                    ],
                    dtype=bool,
                )
            raise AssertionError(f"unexpected drivable query shape: {tuple(points_xy.shape)}")

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
    assert fake_map.calls == [(2, 3, 4, 2)]


def test_nuscenes_pdm_backend_drivable_area_uses_ego_box_corners(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)

    class FakeDrivableMap:
        def batch_contains_points(self, points_xy):
            if tuple(points_xy.shape) != (1, 3, 4, 2):
                raise AssertionError(f"unexpected drivable query shape: {tuple(points_xy.shape)}")
            # Centerline is fully drivable, but one corner leaves the map at the second step.
            return np.asarray(
                [
                    [
                        [True, True, True, True],
                        [True, False, True, True],
                        [True, True, True, True],
                    ]
                ],
                dtype=bool,
            )

    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.drivable_map = FakeDrivableMap()
    sample_context.centerline_segments_xy = np.asarray(
        [
            [[0.0, 0.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    sample_context.centerline_tangents_xy = np.asarray(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = scorer._batch_map_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["drivable_area"].shape == (1,)
    assert metrics["drivable_area"][0] == pytest.approx(0.0)


def test_nuscenes_pdm_backend_batch_project_progress_matches_scalar_reference(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer
    from framework.algorithms.nuscenes_token_scorer import _project_progress

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    final_points_xy = np.asarray(
        [
            [0.4, 0.1],
            [1.6, -0.2],
            [2.5, 0.4],
        ],
        dtype=np.float32,
    )
    path_xy = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    path_s = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    batch_progress = scorer._batch_project_progress(final_points_xy, path_xy, path_s)
    scalar_progress = np.asarray(
        [_project_progress(point_xy, path_xy, path_s) for point_xy in final_points_xy],
        dtype=np.float32,
    )

    assert batch_progress.shape == (3,)
    assert np.allclose(batch_progress, scalar_progress, atol=1.0e-5)


def test_nuscenes_pdm_backend_ea_gate_is_optional_and_gates_batched_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    scene_cache_root = tmp_path / "scene_cache"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    traj_xyyaw = torch.tensor(
        [
            [
                [[2.0, 0.0, 0.0], [3.5, 0.0, 0.0], [4.5, 0.0, 0.0]],
                [[1.0, 2.5, 0.0], [2.0, 2.5, 0.0], [3.0, 2.5, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "gt_xy": np.asarray([[2.0, 0.0], [3.5, 0.0], [4.5, 0.0]], dtype=np.float32),
            "gt_yaw": np.zeros((3,), dtype=np.float32),
            "gt_s": np.asarray([0.0, 1.5, 2.5], dtype=np.float32),
            "gt_total_len": 2.5,
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [],
            "ea_agent_states": [
                {
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
        }

    scorer_plain = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root / "plain",
    )
    monkeypatch.setattr(scorer_plain._delegate, "_build_static_sample_context", fake_static_context)
    plain_scores = scorer_plain.score([{"sample_token": "tok-a"}], traj_xyyaw)

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root / "ea",
        ea_gate_enabled=True,
        ea_gate_good_threshold=0.0,
        ea_gate_bad_threshold=5.0,
    )
    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)
    monkeypatch.setattr(
        scorer,
        "_compute_ea_value_batch_for_pairs",
        lambda ego_states, agent_states: np.asarray(
            [
                4.0 if float(ego_state["x"]) > 1.5 and abs(float(ego_state["y"])) < 0.5 else 0.0
                for ego_state, _agent_state in zip(ego_states, agent_states, strict=False)
            ],
            dtype=np.float32,
        ),
        raising=False,
    )

    gated_scores = scorer.score([{"sample_token": "tok-a"}], traj_xyyaw)

    assert plain_scores.shape == (1, 2)
    assert gated_scores.shape == (1, 2)
    assert float(plain_scores[0, 0]) == pytest.approx(0.0)
    assert float(gated_scores[0, 0]) == pytest.approx(0.0)
    assert float(gated_scores[0, 1]) == pytest.approx(float(plain_scores[0, 1]), rel=1.0e-6)


def test_nuscenes_pdm_backend_can_disable_driving_direction_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    scene_cache_root = tmp_path / "scene_cache"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    traj_xyyaw = torch.tensor(
        [
            [
                [[-1.0, 0.0, np.pi], [-2.0, 0.0, np.pi], [-3.0, 0.0, np.pi], [-4.0, 0.0, np.pi], [-5.0, 0.0, np.pi]],
            ]
        ],
        dtype=torch.float32,
    )

    def fake_static_context(replay, *, patch_radius: float):
            del replay
            return {
                "gt_xy": np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]], dtype=np.float32),
                "gt_yaw": np.zeros((5,), dtype=np.float32),
                "gt_s": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "gt_total_len": 4.0,
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {
                    "drivable_area": [
                        [[-8.0, -4.0], [8.0, -4.0], [8.0, 4.0], [-8.0, 4.0], [-8.0, -4.0]],
                    ],
                    "lane_centerline": [
                        [[-6.0, 0.0], [-3.0, 0.0], [0.0, 0.0], [3.0, 0.0], [6.0, 0.0]],
                    ],
                },
            },
            "scene_objects": [],
            "ea_agent_states": [],
        }

    scorer_default = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root / "default",
    )
    monkeypatch.setattr(scorer_default._delegate, "_build_static_sample_context", fake_static_context)
    default_scores = scorer_default.score([{"sample_token": "tok-a"}], traj_xyyaw)

    scorer_disabled = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root / "disabled",
        driving_direction_gate_enabled=False,
    )
    monkeypatch.setattr(scorer_disabled._delegate, "_build_static_sample_context", fake_static_context)
    disabled_scores = scorer_disabled.score([{"sample_token": "tok-a"}], traj_xyyaw)

    assert default_scores.shape == (1, 1)
    assert disabled_scores.shape == (1, 1)
    assert float(disabled_scores[0, 0]) > float(default_scores[0, 0])


def test_nuscenes_pdm_backend_ttc_uses_agent_future_truth_boxes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "row": {
                    "gt_boxes": np.asarray(
                        [
                            [4.4, 1.0, 0.0, 1.0, 4.0, 1.8, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                "gt_velocity": np.asarray([[0.0, 0.0]], dtype=np.float32),
                "gt_names": np.asarray(["vehicle.car"], dtype=object),
                "valid_flag": np.asarray([True], dtype=bool),
                "gt_agent_fut_trajs": np.asarray(
                    [
                        [[0.0, -1.0], [0.0, -2.0], [0.0, -3.0]],
                    ],
                    dtype=np.float32,
                ),
                "gt_agent_fut_masks": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
                "gt_agent_fut_yaw": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            },
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [4.4, 1.0],
                    "velocity_xy": [0.0, 0.0],
                    "yaw_rad": 0.0,
                    "length_m": 4.0,
                    "width_m": 1.0,
                }
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (1,)
    assert metrics["ttc"].shape == (1,)
    assert metrics["no_collision"][0] == pytest.approx(0.0)
    assert float(metrics["ttc"][0]) < 1.0


def test_nuscenes_pdm_backend_ttc_falls_back_to_ctrv_when_future_truth_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [4.4, 6.5],
                    "velocity_xy": [0.0, -2.0],
                    "yaw_rad": -np.pi * 0.5,
                    "yaw_rate_rps": 0.0,
                    "length_m": 4.0,
                    "width_m": 1.0,
                }
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (1,)
    assert metrics["ttc"].shape == (1,)
    assert metrics["no_collision"][0] == pytest.approx(1.0)
    assert float(metrics["ttc"][0]) < 1.0


def test_nuscenes_pdm_backend_no_collision_uses_future_agent_boxes_not_static_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "row": {
                "gt_boxes": np.asarray(
                    [
                        [2.0, 0.0, 0.0, 1.0, 4.0, 1.8, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "gt_velocity": np.asarray([[0.0, 0.0]], dtype=np.float32),
                "gt_names": np.asarray(["vehicle.car"], dtype=object),
                "valid_flag": np.asarray([True], dtype=bool),
                "gt_agent_fut_trajs": np.asarray(
                    [
                        [[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]],
                    ],
                    dtype=np.float32,
                ),
                "gt_agent_fut_masks": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
                "gt_agent_fut_yaw": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            },
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [2.0, 0.0],
                    "velocity_xy": [0.0, 0.0],
                    "yaw_rad": 0.0,
                    "length_m": 4.0,
                    "width_m": 1.0,
                }
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    traj_xyyaw = torch.tensor(
        [
            [
                [[7.0, 0.0, 0.0], [9.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (1,)
    assert metrics["no_collision"][0] == pytest.approx(1.0)


def test_nuscenes_pdm_backend_no_collision_detects_future_agent_box_overlap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "row": {
                "gt_boxes": np.asarray(
                    [
                        [20.0, 0.0, 0.0, 1.0, 4.0, 1.8, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "gt_velocity": np.asarray([[0.0, 0.0]], dtype=np.float32),
                "gt_names": np.asarray(["vehicle.car"], dtype=object),
                "valid_flag": np.asarray([True], dtype=bool),
                "gt_agent_fut_trajs": np.asarray(
                    [
                        [[0.0, -16.0], [0.0, -16.0], [0.0, -16.0]],
                    ],
                    dtype=np.float32,
                ),
                "gt_agent_fut_masks": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
                "gt_agent_fut_yaw": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            },
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [20.0, 0.0],
                    "velocity_xy": [0.0, 0.0],
                    "yaw_rad": 0.0,
                    "length_m": 4.0,
                    "width_m": 1.0,
                }
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (1,)
    assert metrics["no_collision"][0] == pytest.approx(0.0)


def test_nuscenes_pdm_backend_no_collision_aligns_future_agents_to_candidate_step_times(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_backend import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "row": {
                "gt_boxes": np.asarray(
                    [
                        [20.0, 0.0, 0.0, 1.0, 4.0, 1.8, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "gt_velocity": np.asarray([[0.0, 0.0]], dtype=np.float32),
                "gt_names": np.asarray(["vehicle.car"], dtype=object),
                "valid_flag": np.asarray([True], dtype=bool),
                # Local future deltas that place the agent at x=6.0 at the first future step.
                "gt_agent_fut_trajs": np.asarray(
                    [
                        [[0.0, -14.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                "gt_agent_fut_masks": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
                "gt_agent_fut_yaw": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            },
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [20.0, 0.0],
                    "velocity_xy": [0.0, 0.0],
                    "yaw_rad": 0.0,
                    "length_m": 4.0,
                    "width_m": 1.0,
                }
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    traj_xyyaw = torch.tensor(
        [
            [
                [[6.0, 0.0, 0.0], [8.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (1,)
    assert metrics["no_collision"][0] == pytest.approx(0.0)
