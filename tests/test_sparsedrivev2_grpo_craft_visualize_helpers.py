from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.smalltool.visualize.visualize_sparsedrivev2_grpo_craft_online import (
    build_candidate_score_payload,
    candidate_visual_styles,
    overlay_top_right_inset,
    render_bev_debug_image,
)
from tools.smalltool.visualize.visualize_sparsedrivev2_grpo_hugsim_online import (
    build_hugsim_grpo_object_context,
    build_hugsim_step_payload,
    build_run_manifest_payload,
    build_timestamped_output_paths,
    extract_reward_terms,
    render_candidate_grid_image,
)


def test_candidate_visual_style_orders_high_score_first_and_fades_low_scores() -> None:
    scores = np.asarray([0.2, 0.95, -0.1, 0.5], dtype=np.float32)

    styles = candidate_visual_styles(scores=scores, top_k=2)

    assert [item["candidate_index"] for item in styles] == [1, 3, 0, 2]
    assert styles[0]["rank"] == 1
    assert styles[1]["rank"] == 2
    assert styles[0]["is_top_k"] is True
    assert styles[2]["is_top_k"] is False
    assert styles[0]["alpha"] > styles[1]["alpha"] > styles[2]["alpha"] > styles[3]["alpha"]
    assert styles[0]["linewidth"] > styles[2]["linewidth"]


def test_build_candidate_score_payload_writes_all_candidates_and_top_k(tmp_path: Path) -> None:
    traj_xyyaw = np.asarray(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.1, 0.0]],
            [[0.5, 0.2, 0.1], [1.0, 0.4, 0.1]],
            [[1.5, -0.2, -0.1], [3.0, -0.3, -0.1]],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.4, 0.1, 0.9], dtype=np.float32)
    logits = np.asarray([3.0, 1.0, 2.0], dtype=np.float32)
    mode_indices = np.asarray([10, 11, 12], dtype=np.int64)

    payload = build_candidate_score_payload(
        scene=7,
        step=12,
        frame_idx=60,
        sample_token="sample-token",
        traj_xyyaw=traj_xyyaw,
        scores=scores,
        score_logits=logits,
        mode_indices=mode_indices,
        top_k=2,
    )

    assert payload["scene"] == 7
    assert payload["step"] == 12
    assert payload["top_k_candidate_indices"] == [2, 0]
    assert [item["candidate_index"] for item in payload["candidates"]] == [2, 0, 1]
    assert payload["candidates"][0]["score"] == pytest.approx(0.9)
    assert payload["candidates"][0]["score_logit"] == pytest.approx(2.0)
    assert payload["candidates"][0]["mode_index"] == 12
    assert payload["candidates"][0]["traj_xyyaw"][-1] == pytest.approx([3.0, -0.3, -0.1])

    out_path = tmp_path / "score.json"
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["candidates"][1]["visual"]["is_top_k"] is True
    assert loaded["candidates"][2]["visual"]["is_top_k"] is False


def test_overlay_top_right_inset_places_resized_inset_without_resizing_frame() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    inset = np.full((20, 30, 3), 255, dtype=np.uint8)

    out = overlay_top_right_inset(frame, inset, inset_width=50, margin=10, border_px=2)

    assert out.shape == frame.shape
    assert np.all(frame == 0)
    assert out[10, 140].sum() > 0
    assert out[95, 5].sum() == 0


def test_render_bev_debug_image_draws_context_and_all_candidates() -> None:
    sample_detail = {
        "sample_token": "tok",
        "gt_xy": [[1.0, 0.0], [2.0, 0.0]],
        "gt_history_xy": [[-1.0, 0.0], [0.0, 0.0]],
        "map_patch_radius_m": 8.0,
        "render_layers": {
            "drivable_polygons": [[[-4.0, -3.0], [8.0, -3.0], [8.0, 3.0], [-4.0, 3.0]]],
            "lane_centerlines": [[[-2.0, 0.0], [6.0, 0.0]]],
            "road_edge_lines": [[[-4.0, -3.0], [8.0, -3.0]]],
        },
        "scene_objects": [
            {
                "category": "car",
                "corners_xy": [[3.0, -0.5], [4.0, -0.5], [4.0, 0.5], [3.0, 0.5]],
                "center_xy": [3.5, 0.0],
            }
        ],
    }
    traj_xyyaw = np.asarray(
        [
            [[1.0, -1.0, 0.0], [2.0, -1.0, 0.0]],
            [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            [[0.5, 2.0, 0.0], [1.0, 2.5, 0.0]],
        ],
        dtype=np.float32,
    )

    img = render_bev_debug_image(
        sample_detail=sample_detail,
        traj_xyyaw=traj_xyyaw,
        scores=np.asarray([0.9, 0.3, -0.1], dtype=np.float32),
        top_k=2,
        width=420,
        height=420,
    )

    assert img.shape == (420, 420, 3)
    assert img.dtype == np.uint8
    assert int(img.max()) > int(img.min())


def test_render_bev_places_positive_y_on_screen_left_for_agents() -> None:
    sample_detail = {
        "sample_token": "tok",
        "map_patch_radius_m": 10.0,
        "scene_objects": [
            {
                "category": "car",
                "corners_xy": [[4.0, 2.0], [4.8, 2.0], [4.8, 3.0], [4.0, 3.0]],
                "center_xy": [4.4, 2.5],
                "yaw_rad": 0.0,
            }
        ],
    }
    traj_xyyaw = np.asarray([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float32)

    img = render_bev_debug_image(
        sample_detail=sample_detail,
        traj_xyyaw=traj_xyyaw,
        scores=np.asarray([1.0], dtype=np.float32),
        top_k=1,
        width=200,
        height=200,
    )

    agent_mask = (img[:, :, 0] > 180) & (img[:, :, 1] < 130) & (img[:, :, 2] < 120)
    ys, xs = np.where(agent_mask)

    assert xs.size > 0
    assert float(xs.mean()) < 100.0


def test_nuscenes_scorer_collect_scene_objects_uses_navsim_box_length_width_order(tmp_path: Path) -> None:
    from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils

    scorer = NuScenesScorerUtils(token2vad_path=tmp_path / "missing.pkl")
    row = {
        "gt_boxes": np.asarray(
            [
                [4.0, 0.0, 0.0, 4.6, 1.8, 1.6, 0.0],
            ],
            dtype=np.float32,
        ),
        "gt_velocity": np.zeros((1, 2), dtype=np.float32),
        "gt_names": np.asarray(["vehicle.car"], dtype=object),
        "valid_flag": np.asarray([True], dtype=bool),
    }

    objects = scorer._collect_scene_objects(row, patch_radius=20.0)

    assert len(objects) == 1
    obj = objects[0]
    assert obj["length_m"] == pytest.approx(4.6)
    assert obj["width_m"] == pytest.approx(1.8)
    corners = np.asarray(obj["corners_xy"], dtype=np.float32)
    assert float(corners[:, 0].max() - corners[:, 0].min()) == pytest.approx(4.6)
    assert float(corners[:, 1].max() - corners[:, 1].min()) == pytest.approx(1.8)


def test_nuscenes_scorer_static_context_uses_merged_hugsim_object_override(tmp_path: Path) -> None:
    from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils

    token2vad = tmp_path / "token2vad.pkl"
    import pickle

    with token2vad.open("wb") as handle:
        pickle.dump(
            {
                "tok": {
                    "token": "tok",
                    "gt_ego_fut_trajs": np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
                    "gt_boxes": np.asarray([[3.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], dtype=np.float32),
                    "gt_velocity": np.zeros((1, 2), dtype=np.float32),
                    "gt_names": np.asarray(["vehicle.car"], dtype=object),
                    "valid_flag": np.asarray([True], dtype=bool),
                },
            },
            handle,
        )

    scorer = NuScenesScorerUtils(token2vad_path=token2vad, scene_cache_root=tmp_path / "scene_cache")
    merged_objects = [
        {
            "source": "hugsim_inserted",
            "token": "hugsim_obj_0",
            "category": "vehicle.car",
            "corners_xy": [[8.0, -1.0], [8.0, 1.0], [10.0, 1.0], [10.0, -1.0]],
        },
        {
            "source": "recon_cache",
            "token": "cache_obj_0",
            "category": "vehicle.car",
            "corners_xy": [[12.0, -1.0], [12.0, 1.0], [14.0, 1.0], [14.0, -1.0]],
        },
    ]

    ctx = scorer._build_static_sample_context(
        {
            "sample_token": "tok",
            "scene_objects_override": merged_objects,
            "ea_agent_states_override": merged_objects,
        },
        patch_radius=20.0,
    )

    assert [obj["source"] for obj in ctx["scene_objects"]] == ["hugsim_inserted", "recon_cache"]
    assert [obj["token"] for obj in ctx["scene_objects"]] == ["hugsim_obj_0", "cache_obj_0"]
    assert len(ctx["ea_agent_states"]) == 2
    assert all(obj["token"] != "obj-0" for obj in ctx["scene_objects"])


def test_hugsim_grpo_object_context_merges_aligned_hugsim_and_recon_cache_objects() -> None:
    info = {
        "hugsim_ego_box_recon_global_poly": [[2.0, 1.0], [2.0, -1.0], [-2.0, -1.0], [-2.0, 1.0]],
        "hugsim_obj_boxes_recon_global": [
            {
                "source": "hugsim_inserted",
                "token": "hugsim_obj_0",
                "category": "vehicle.car",
                "poly": [[8.0, 1.0], [8.0, -1.0], [6.0, -1.0], [6.0, 1.0]],
            }
        ],
        "recon_cache_dynamic_objects": [
            {
                "source": "recon_cache",
                "token": "cache_obj_0",
                "category": "vehicle.car",
                "poly": [[13.0, 1.0], [13.0, -1.0], [11.0, -1.0], [11.0, 1.0]],
            }
        ],
    }

    context = build_hugsim_grpo_object_context(info)

    assert context["available"] is True
    assert context["hugsim_object_count"] == 1
    assert context["recon_cache_object_count"] == 1
    assert [obj["source"] for obj in context["scene_objects"]] == ["hugsim_inserted", "recon_cache"]
    centers = np.asarray([obj["center_xy"] for obj in context["scene_objects"]], dtype=np.float32)
    assert centers[:, 0].tolist() == pytest.approx([7.0, 12.0])
    assert centers[:, 1].tolist() == pytest.approx([0.0, 0.0])


def test_hugsim_grpo_object_context_can_override_to_empty_when_aligned_context_exists() -> None:
    context = build_hugsim_grpo_object_context(
        {
            "hugsim_ego_box_recon_global_poly": [[2.0, 1.0], [2.0, -1.0], [-2.0, -1.0], [-2.0, 1.0]],
            "hugsim_obj_boxes_recon_global": [],
            "recon_cache_dynamic_objects": [],
        }
    )

    assert context["available"] is True
    assert context["scene_objects"] == []
    assert context["hugsim_object_count"] == 0
    assert context["recon_cache_object_count"] == 0


def test_hugsim_grpo_object_context_rotates_recon_global_velocity_to_local() -> None:
    context = build_hugsim_grpo_object_context(
        {
            "hugsim_ego_box_recon_global_poly": [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0], [-1.0, -2.0]],
            "hugsim_obj_boxes_recon_global": [],
            "recon_cache_dynamic_objects": [
                {
                    "source": "recon_cache",
                    "token": "cache_obj_0",
                    "category": "vehicle.car",
                    "poly": [[-1.0, 8.0], [1.0, 8.0], [1.0, 6.0], [-1.0, 6.0]],
                    "velocity_xy": [0.0, 3.0],
                }
            ],
        }
    )

    assert context["available"] is True
    obj = context["scene_objects"][0]
    assert obj["center_xy"] == pytest.approx([7.0, 0.0])
    assert obj["velocity_xy"] == pytest.approx([3.0, 0.0])
    assert obj["speed_mps"] == pytest.approx(3.0)


def test_build_hugsim_visualization_manifest_payload_records_scorer_backend(tmp_path: Path) -> None:
    config = {
        "env": {
            "backend": "hugsim_ori",
            "hugsim": {"scenario_dir": "/tmp/scenarios"},
        },
        "agent": {
            "type": "sparsedrive_v2",
            "nuscenes_scorer": {"backend": "craft_carl", "carl": {"w_prog": 8.0}},
        },
        "train": {"grpo": {"num_candidates": 8, "candidate_select": "topk"}},
    }

    payload = build_run_manifest_payload(
        config=config,
        config_path="/repo/config.yaml",
        ckpt_path="/repo/model.ckpt",
        out_dir=tmp_path,
        scene_index=3,
        official_scene_name="scene-0003",
        scenario_path="/hugsim/scenarios/scene-0003.yaml",
        cuda=2,
        num_candidates=8,
        candidate_select="topk",
        mode_select="sample",
    )

    assert payload["env_backend"] == "hugsim_ori"
    assert payload["scene_index"] == 3
    assert payload["official_scene_name"] == "scene-0003"
    assert payload["scenario_path"] == "/hugsim/scenarios/scene-0003.yaml"
    assert payload["scorer_backend"] == "craft_carl"
    assert payload["grpo"]["num_candidates"] == 8
    assert payload["policy"]["mode_select"] == "sample"


def test_hugsim_visualization_output_paths_use_scene_name_and_timestamp(tmp_path: Path) -> None:
    paths = build_timestamped_output_paths(
        output_root=tmp_path,
        official_scene_name="scene/with spaces:001",
        timestamp="20260528-123456",
    )

    assert paths.root == tmp_path / "scene_with_spaces_001_20260528-123456"
    assert paths.video == paths.root / "scene_with_spaces_001_grpo.mp4"
    assert paths.step_csv == paths.root / "step_summary.csv"
    assert paths.frames_dir.name == "frames"
    assert paths.candidate_grid_dir.name == "candidate_grid"


def test_render_candidate_grid_image_draws_one_panel_per_candidate() -> None:
    sample_detail = {
        "sample_token": "tok",
        "map_patch_radius_m": 8.0,
        "gt_xy": [[1.0, 0.0], [2.0, 0.0]],
    }
    traj_xyyaw = np.asarray(
        [
            [[0.4 + 0.1 * i, -1.0 + i * 0.25, 0.0], [2.0, -1.0 + i * 0.25, 0.0]]
            for i in range(8)
        ],
        dtype=np.float32,
    )
    scores = np.asarray([-8.0, -2.0, -6.0, -5.0, -1.0, -4.0, -3.0, -7.0], dtype=np.float32)

    img = render_candidate_grid_image(
        sample_detail=sample_detail,
        traj_xyyaw=traj_xyyaw,
        scores=scores,
        top_k=3,
        panel_size=180,
        columns=4,
    )

    assert img.shape == (360, 720, 3)
    assert img.dtype == np.uint8
    assert int(img.max()) > int(img.min())


def test_hugsim_step_payload_keeps_grpo_scores_and_reward_terms() -> None:
    traj_xyyaw = np.asarray(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.1, 0.0]],
            [[0.5, 0.2, 0.1], [1.0, 0.4, 0.1]],
        ],
        dtype=np.float32,
    )
    candidate_payload = build_candidate_score_payload(
        scene=4,
        step=2,
        frame_idx=30,
        sample_token="tok",
        traj_xyyaw=traj_xyyaw,
        scores=np.asarray([-12.0, -5.0], dtype=np.float32),
        score_logits=np.asarray([0.1, 0.9], dtype=np.float32),
        mode_indices=np.asarray([5, 9], dtype=np.int64),
        top_k=1,
    )
    reward_info = {
        "reward_mode": "craft_corrective",
        "positive_reward": 0.5,
        "cost_reward": 5.5,
        "front_obstacle_cost": 3.0,
        "done_reason": "hugsim_collision",
        "collision": True,
        "debug_blob": {"ignored": "non-scalar"},
    }

    payload = build_hugsim_step_payload(
        step=2,
        scene_index=4,
        official_scene_name="scene-004",
        frame_idx=30,
        sample_token="tok",
        selected_mode_index=9,
        selected_logp=-0.2,
        closed_loop_reward=-5.0,
        closed_loop_reward_sum=-7.5,
        reward_info=reward_info,
        candidate_payload=candidate_payload,
    )

    assert payload["grpo_score_summary"]["min"] == pytest.approx(-12.0)
    assert payload["grpo_score_summary"]["max"] == pytest.approx(-5.0)
    assert payload["selected"]["mode_index"] == 9
    assert payload["closed_loop_reward"]["step_reward"] == pytest.approx(-5.0)
    assert payload["reward_info"]["done_reason"] == "hugsim_collision"
    assert payload["reward_terms"]["cost_reward"] == pytest.approx(5.5)
    assert payload["reward_terms"]["front_obstacle_cost"] == pytest.approx(3.0)
    assert "debug_blob" not in extract_reward_terms(reward_info)
