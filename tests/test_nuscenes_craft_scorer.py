from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from framework.algorithms.craft_reward import CRAFT_CARL_FORWARD_SIM_DEFAULTS


def _write_token2vad(path: Path) -> None:
    payload = {
        "tok-a": {
            "token": "tok-a",
            "gt_ego_fut_trajs": np.asarray(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                ],
                dtype=np.float32,
            ),
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_craft_carl_scorer_orders_progress_deviation_offroad_and_collision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from framework.algorithms.nuscenes_craft_scorer import NuScenesCraftScorer

    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    scorer = NuScenesCraftScorer(token2vad_path=token2vad_path)

    def fake_static_context(replay, *, patch_radius):
        del replay, patch_radius
        return {
            "gt_xy": np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            "gt_yaw": np.zeros((3,), dtype=np.float32),
            "gt_s": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
            "gt_total_len": 2.0,
            "map_context": {
                "patch_radius": 20.0,
                "layers": {
                    "lane_centerline": [[[0.0, 0.0], [5.0, 0.0]]],
                    "drivable_area": [[[-10.0, -5.0], [10.0, -5.0], [10.0, 5.0], [-10.0, 5.0]]],
                },
            },
        }

    monkeypatch.setattr(scorer._pdm._delegate, "_build_static_sample_context", fake_static_context)

    traj_xyyaw = torch.zeros((1, 4, 3, 3), dtype=torch.float32)
    traj_xyyaw[0, 0, :, 0] = torch.tensor([1.0, 2.0, 3.0])  # ideal
    traj_xyyaw[0, 1, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    traj_xyyaw[0, 1, :, 1] = 1.0  # deviated
    traj_xyyaw[0, 2, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    traj_xyyaw[0, 2, :, 1] = 20.0  # off-road
    traj_xyyaw[0, 3, :, 0] = torch.tensor([1.0, 2.0, 3.0])

    sample_context = scorer._pdm._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    geometry = scorer._pdm._build_candidate_geometry_batch(traj_xyyaw)["centers_xy"][0]
    del geometry
    monkeypatch.setattr(
        scorer._pdm,
        "_batch_collision_ttc_metrics",
        lambda **kwargs: {
            "no_collision": np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "ttc": np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        },
    )
    scorer._pdm._sample_context_cache["tok-a"] = sample_context

    scores = scorer.score([{"sample_token": "tok-a"}], traj_xyyaw)

    assert tuple(scores.shape) == (1, 4)
    assert scores[0, 0] > scores[0, 1]
    assert scores[0, 1] > scores[0, 2]
    assert scores[0, 2] > scores[0, 3]


def test_craft_carl_scorer_ignores_legacy_pdm_ttc_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from framework.algorithms.nuscenes_craft_scorer import NuScenesCraftScorer

    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    def make_scorer(**kwargs) -> NuScenesCraftScorer:
        scorer = NuScenesCraftScorer(token2vad_path=token2vad_path, **kwargs)

        def fake_static_context(replay, *, patch_radius):
            del replay, patch_radius
            return {
                "gt_xy": np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
                "gt_yaw": np.zeros((3,), dtype=np.float32),
                "gt_s": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                "gt_total_len": 2.0,
                "map_context": {
                    "patch_radius": 20.0,
                    "layers": {
                        "lane_centerline": [[[0.0, 0.0], [5.0, 0.0]]],
                        "drivable_area": [[[-10.0, -5.0], [10.0, -5.0], [10.0, 5.0], [-10.0, 5.0]]],
                    },
                },
            }

        monkeypatch.setattr(scorer._pdm._delegate, "_build_static_sample_context", fake_static_context)
        return scorer

    base = make_scorer(carl=CRAFT_CARL_FORWARD_SIM_DEFAULTS)
    noisy = make_scorer(carl=CRAFT_CARL_FORWARD_SIM_DEFAULTS, dac_weight=999.0, ttc_weight=999.0)

    traj_xyyaw = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    traj_xyyaw[0, 0, :, 0] = torch.tensor([1.0, 2.0, 3.0])

    assert np.allclose(
        base.score([{"sample_token": "tok-a"}], traj_xyyaw),
        noisy.score([{"sample_token": "tok-a"}], traj_xyyaw),
    )


def test_craft_carl_scorer_uses_step_level_collision_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from framework.algorithms.nuscenes_craft_scorer import NuScenesCraftScorer

    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    scorer = NuScenesCraftScorer(token2vad_path=token2vad_path)

    def fake_static_context(replay, *, patch_radius):
        del replay, patch_radius
        return {
            "gt_xy": np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            "gt_yaw": np.zeros((3,), dtype=np.float32),
            "gt_s": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
            "gt_total_len": 2.0,
            "map_context": {
                "patch_radius": 20.0,
                "layers": {
                    "lane_centerline": [[[0.0, 0.0], [5.0, 0.0]]],
                    "drivable_area": [[[-10.0, -5.0], [10.0, -5.0], [10.0, 5.0], [-10.0, 5.0]]],
                },
            },
        }

    monkeypatch.setattr(scorer._pdm._delegate, "_build_static_sample_context", fake_static_context)
    monkeypatch.setattr(
        scorer._pdm,
        "_batch_collision_ttc_metrics",
        lambda **kwargs: {
            "no_collision": np.asarray([0.0], dtype=np.float32),
            "ttc": np.asarray([1.0], dtype=np.float32),
            "collision_matrix": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        },
    )

    traj_xyyaw = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    traj_xyyaw[0, 0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    scorer.score([{"sample_token": "tok-a"}], traj_xyyaw)

    collision_cost = scorer._last_terms["collision_cost"]
    assert np.allclose(
        collision_cost,
        np.asarray([[0.0, CRAFT_CARL_FORWARD_SIM_DEFAULTS["term_collision"], 0.0]], dtype=np.float32),
    )


def test_craft_carl_route_projection_uses_segments_not_nearest_gt_vertex(tmp_path: Path) -> None:
    from framework.algorithms.nuscenes_craft_scorer import NuScenesCraftScorer

    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    scorer = NuScenesCraftScorer(token2vad_path=token2vad_path, center_dev_max_m=2.0)

    centers_xy = np.asarray([[[5.0, 1.0]]], dtype=np.float32)
    yaw_rad = np.asarray([[0.0]], dtype=np.float32)
    gt_xy = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    gt_s = np.asarray([0.0, 10.0], dtype=np.float32)

    progress_s, route_lateral, route_heading_ratio, _ = scorer._project_route_stats_all(
        centers_xy,
        yaw_rad,
        gt_xy,
        gt_s,
    )

    assert progress_s[0, 0] == pytest.approx(5.0)
    assert route_lateral[0, 0] == pytest.approx(1.0)
    assert route_heading_ratio[0, 0] == pytest.approx(0.0)


def test_craft_carl_scorer_returns_neutral_scores_when_gt_route_is_too_short(tmp_path: Path) -> None:
    from framework.algorithms.nuscenes_craft_scorer import NuScenesCraftScorer

    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    scorer = NuScenesCraftScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    geometry = scorer._pdm._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._pdm._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    scores = scorer._score_candidate_batch_for_sample(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        gt_xy_full=np.zeros((1, 2), dtype=np.float32),
        gt_s_full=np.zeros((1,), dtype=np.float32),
        dt_s=0.25,
    )

    assert np.allclose(scores, np.zeros((1,), dtype=np.float32))
    assert scorer._last_terms["skipped_short_gt_route_count"] == pytest.approx(1.0)


def test_craft_carl_scorer_passes_dt_to_collision_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from framework.algorithms.nuscenes_craft_scorer import NuScenesCraftScorer

    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    scorer = NuScenesCraftScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    geometry = scorer._pdm._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._pdm._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    seen_dt: list[float] = []

    def fake_collision_metrics(**kwargs):
        seen_dt.append(float(kwargs["dt_s"]))
        return {
            "no_collision": np.asarray([1.0], dtype=np.float32),
            "ttc": np.asarray([1.0], dtype=np.float32),
            "collision_matrix": np.zeros((1, 3), dtype=np.float32),
        }

    monkeypatch.setattr(scorer._pdm, "_batch_collision_ttc_metrics", fake_collision_metrics)

    scorer._score_candidate_batch_for_sample(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        gt_xy_full=np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        gt_s_full=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        dt_s=0.25,
    )

    assert seen_dt == pytest.approx([0.25])


def test_craft_carl_scorer_ignores_default_map_heading_when_centerlines_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from framework.algorithms.nuscenes_craft_scorer import NuScenesCraftScorer

    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    scorer = NuScenesCraftScorer(token2vad_path=token2vad_path, heading_dev_max_deg=90.0)
    traj_xyyaw = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    traj_xyyaw[0, 0, :, 1] = torch.tensor([1.0, 2.0, 3.0])
    traj_xyyaw[0, 0, :, 2] = np.pi * 0.5
    geometry = scorer._pdm._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._pdm._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.centerline_segments_xy = np.zeros((0, 2, 2), dtype=np.float32)
    sample_context.centerline_tangents_xy = np.zeros((0, 2), dtype=np.float32)
    monkeypatch.setattr(
        scorer._pdm,
        "_batch_collision_ttc_metrics",
        lambda **kwargs: {
            "no_collision": np.asarray([1.0], dtype=np.float32),
            "ttc": np.asarray([1.0], dtype=np.float32),
            "collision_matrix": np.zeros((1, 3), dtype=np.float32),
        },
    )

    scorer._score_candidate_batch_for_sample(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        gt_xy_full=np.asarray([[0.0, 0.0], [0.0, 4.0]], dtype=np.float32),
        gt_s_full=np.asarray([0.0, 4.0], dtype=np.float32),
        dt_s=0.5,
    )

    assert np.allclose(scorer._last_terms["efficiency"], np.ones((1, 3), dtype=np.float32))
