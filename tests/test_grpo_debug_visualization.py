from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils
from framework.lightning.config import ActorLearnerLightningConfig, LearnerOptimizerConfig
from framework.lightning.trajectory_module import TrajectoryLightningModule


class _DebugAgent:
    def __init__(self) -> None:
        self.trainable_module = torch.nn.Linear(1, 1)
        self.dump_calls: list[tuple[str, int]] = []

    def sample_counterfactual_trajectories_from_replay_batch(
        self,
        replay,
        *,
        num_candidates: int,
        candidate_select: str = "topk",
    ):
        del replay, candidate_select
        return {
            "traj_xyyaw": torch.zeros((1, num_candidates, 4, 3), dtype=torch.float32),
            "log_probs": torch.full((1, num_candidates), -0.25, dtype=torch.float32),
        }

    def dump_counterfactual_debug_from_replay_batch(
        self,
        replays,
        traj_xyyaw,
        candidate_scores,
        *,
        out_dir: str,
        step_tag: str,
        top_k: int,
    ) -> None:
        del replays, traj_xyyaw, candidate_scores
        self.dump_calls.append((step_tag, top_k))
        Path(out_dir, f"{step_tag}.txt").write_text("debug", encoding="utf-8")


def test_grpo_debug_dump_runs_without_aux_loss_and_zero_max_batches_is_unlimited(
    monkeypatch,
    tmp_path: Path,
) -> None:
    agent = _DebugAgent()
    learner_config = ActorLearnerLightningConfig(
        algo_kind="ppo",
        optimizer_config=LearnerOptimizerConfig(policy_lr=1.0e-4),
        eta=1.0,
        clip_eps=0.2,
        grpo_enabled=False,
        grpo_coef=0.0,
        grpo_num_candidates=3,
        grpo_candidate_select="topk",
        grpo_debug_visualize=True,
        grpo_debug_dir=str(tmp_path),
        grpo_debug_max_batches=0,
        grpo_debug_top_k=2,
    )
    module = TrajectoryLightningModule(
        agent=agent,
        learner_config=learner_config,
        value_net=torch.nn.Linear(1, 1),
    )

    monkeypatch.setattr(
        "framework.lightning.trajectory_module.score_counterfactual_trajectories",
        lambda *args, **kwargs: torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32),
    )

    base_loss = torch.tensor(2.0, dtype=torch.float32)
    base_metrics = {"loss_pi": torch.tensor(1.0, dtype=torch.float32)}
    replay = [{"scene_id": 146, "frame_idx": 0, "sample_token": "tok"}]

    out_loss_0, out_metrics_0 = module._maybe_apply_grpo_loss(
        replay=replay,
        device=torch.device("cpu"),
        batch_idx=0,
        loss=base_loss,
        metrics=base_metrics,
    )
    out_loss_1, out_metrics_1 = module._maybe_apply_grpo_loss(
        replay=replay,
        device=torch.device("cpu"),
        batch_idx=1,
        loss=base_loss,
        metrics=base_metrics,
    )

    assert torch.equal(out_loss_0, base_loss)
    assert torch.equal(out_loss_1, base_loss)
    assert out_metrics_0 == base_metrics
    assert out_metrics_1 == base_metrics
    assert len(agent.dump_calls) == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == ["step000000_batch0000.txt", "step000000_batch0001.txt"]


def test_nuscenes_debug_dump_exports_cumulative_gt_and_bev_scene_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token = "tok-sample"
    token2vad_path = tmp_path / "token2vad.pkl"
    with token2vad_path.open("wb") as f:
        pickle.dump(
            {
                token: {
                    "token": token,
                    "map_location": "singapore-queenstown",
                    "ego2global_translation": [100.0, 200.0, 0.0],
                    "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
                    "gt_ego_fut_trajs": np.asarray(
                        [
                            [1.0, 3.0],
                            [1.2, 3.3],
                            [1.4, 3.7],
                        ],
                        dtype=np.float32,
                    ),
                    "gt_ego_his_trajs": np.asarray(
                        [
                            [0.7, 2.4],
                            [0.8, 2.7],
                        ],
                        dtype=np.float32,
                    ),
                    "gt_boxes": np.asarray(
                        [
                            [8.0, -2.0, 0.0, 2.0, 4.5, 1.6, 0.25],
                            [-6.0, 3.5, 0.0, 0.8, 0.8, 1.7, -0.1],
                        ],
                        dtype=np.float32,
                    ),
                    "gt_names": np.asarray(["car", "pedestrian"]),
                    "gt_velocity": np.asarray([[4.0, 0.0], [0.5, 0.0]], dtype=np.float32),
                    "valid_flag": np.asarray([True, True]),
                    "num_lidar_pts": np.asarray([12, 8]),
                    "num_radar_pts": np.asarray([0, 0]),
                }
            },
            f,
        )

    scorer = NuScenesScorerUtils(token2vad_path=token2vad_path)
    monkeypatch.setattr(
        scorer,
        "_lookup_map_layers",
        lambda row, *, patch_radius=30.0: {
            "patch_radius": float(patch_radius),
            "layers": {
                "drivable_area": [
                    [[-10.0, -5.0], [18.0, -5.0], [18.0, 5.0], [-10.0, 5.0]],
                ],
                "lane_centerline": [
                    [[-5.0, 0.0], [0.0, 0.0], [8.0, 0.5]],
                ],
            },
        },
        raising=False,
    )

    traj_xyyaw = torch.tensor(
        [
            [
                [
                    [0.05, 0.00, 0.00],
                    [0.30, 0.05, 0.03],
                    [0.68, 0.12, 0.06],
                ],
                [
                    [0.02, -0.03, -0.02],
                    [0.20, -0.02, -0.04],
                    [0.42, 0.01, 0.01],
                ],
            ]
        ],
        dtype=torch.float32,
    )

    artifacts = scorer.dump_debug_artifacts(
        [{"sample_token": token}],
        traj_xyyaw,
        out_dir=tmp_path,
        step_tag="step000123_batch0001",
        top_k=2,
    )

    assert len(artifacts) == 1
    payload = json.loads(Path(artifacts[0]["json_path"]).read_text())
    assert Path(artifacts[0]["png_path"]).exists()
    assert payload["gt_xy"][0] == pytest.approx([3.0, 1.0])
    assert payload["gt_xy"][-1] == pytest.approx([10.0, 3.6])
    assert payload["gt_history_xy"][-1][0] > 0.0
    assert payload["scene_objects"][0]["category"] == "car"
    assert payload["scene_objects"][1]["category"] == "pedestrian"
    assert payload["map_layers"]["drivable_area"]
    assert payload["map_layers"]["lane_centerline"]


def test_nuscenes_scorer_prefers_candidates_aligned_with_cumulative_gt(
    tmp_path: Path,
) -> None:
    token = "tok-score"
    token2vad_path = tmp_path / "token2vad.pkl"
    with token2vad_path.open("wb") as f:
        pickle.dump(
            {
                token: {
                    "token": token,
                    "gt_ego_fut_trajs": np.asarray(
                        [
                            [1.0, 3.0],
                            [1.2, 3.3],
                            [1.4, 3.7],
                        ],
                        dtype=np.float32,
                    ),
                }
            },
            f,
        )

    scorer = NuScenesScorerUtils(token2vad_path=token2vad_path)
    aligned = np.asarray(
        [
            [3.0, 1.0, 0.0],
            [6.3, 2.2, 0.0],
            [10.0, 3.6, 0.0],
        ],
        dtype=np.float32,
    )
    misaligned = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.2, 0.0],
            [0.7, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    candidate_batch = torch.from_numpy(
        np.stack([aligned, misaligned], axis=0)[None, ...]
    )

    scores = scorer.score(
        [{"sample_token": token}],
        candidate_batch,
    )

    assert float(scores[0, 0]) > float(scores[0, 1])


def test_nuscenes_debug_dump_trims_zero_padded_gt_future(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token = "tok-padded"
    token2vad_path = tmp_path / "token2vad.pkl"
    with token2vad_path.open("wb") as f:
        pickle.dump(
            {
                token: {
                    "token": token,
                    "gt_ego_fut_trajs": np.asarray(
                        [
                            [1.0, 3.0],
                            [1.2, 3.3],
                            [1.4, 3.7],
                            [0.0, 0.0],
                            [0.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                }
            },
            f,
        )

    scorer = NuScenesScorerUtils(token2vad_path=token2vad_path)
    monkeypatch.setattr(
        scorer,
        "_lookup_map_layers",
        lambda row, *, patch_radius=30.0: {
            "patch_radius": float(patch_radius),
            "layers": {},
        },
        raising=False,
    )

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.7, 0.4, 0.0], [1.0, 0.5, 0.0], [1.2, 0.6, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    artifacts = scorer.dump_debug_artifacts(
        [{"sample_token": token}],
        traj_xyyaw,
        out_dir=tmp_path,
        step_tag="step000001_batch0000",
        top_k=1,
    )

    assert len(artifacts) == 1
    payload = json.loads(Path(artifacts[0]["json_path"]).read_text())
    assert len(payload["gt_xy"]) == 3
    assert payload["gt_xy"][-1] == pytest.approx([10.0, 3.6])


def test_nuscenes_render_layers_builds_filled_road_and_lane_guides() -> None:
    render_layers = NuScenesScorerUtils._build_render_layers(
        {
            "drivable_area": [
                [[-8.0, -4.0], [8.0, -4.0], [8.0, 4.0], [-8.0, 4.0]],
            ],
            "road_segment": [
                [[-7.0, -3.0], [7.0, -3.0], [7.0, 3.0], [-7.0, 3.0]],
            ],
            "lane": [
                [[-6.0, -1.6], [6.0, -1.6], [6.0, 1.6], [-6.0, 1.6]],
            ],
            "lane_connector": [
                [[6.0, -1.4], [8.0, -0.8], [8.0, 0.8], [6.0, 1.4]],
            ],
            "walkway": [
                [[-8.0, 4.0], [8.0, 4.0], [8.0, 5.5], [-8.0, 5.5]],
            ],
            "ped_crossing": [
                [[-1.0, -3.0], [1.0, -3.0], [1.0, 3.0], [-1.0, 3.0]],
            ],
            "lane_divider": [
                [[-6.0, 0.0], [6.0, 0.0]],
            ],
            "road_divider": [
                [[0.0, -3.0], [0.0, 3.0]],
            ],
            "lane_centerline": [
                [[-6.0, -0.8], [0.0, -0.5], [6.0, -0.2]],
            ],
        }
    )

    assert render_layers["road_surface_polygons"]
    assert render_layers["road_edge_lines"]
    assert render_layers["lane_marking_lines"]
    assert render_layers["lane_boundary_lines"]
    assert render_layers["lane_centerlines"]
    assert render_layers["walkway_polygons"]
    assert render_layers["crossing_polygons"]
    assert render_layers["crossing_stripe_polygons"]


def test_candidate_rank_style_uses_transparent_lines() -> None:
    rank1 = NuScenesScorerUtils._candidate_line_style(rank=0, total=8)
    rank3 = NuScenesScorerUtils._candidate_line_style(rank=2, total=8)
    rank4 = NuScenesScorerUtils._candidate_line_style(rank=3, total=8)
    rank8 = NuScenesScorerUtils._candidate_line_style(rank=7, total=8)

    assert 0.35 <= float(rank1["alpha"]) <= 0.75
    assert float(rank1["linewidth"]) >= 1.8
    assert float(rank3["alpha"]) >= float(rank4["alpha"])
    assert float(rank3["linewidth"]) >= float(rank4["linewidth"])
    assert float(rank4["alpha"]) > float(rank8["alpha"])
    assert float(rank4["linewidth"]) > float(rank8["linewidth"])
