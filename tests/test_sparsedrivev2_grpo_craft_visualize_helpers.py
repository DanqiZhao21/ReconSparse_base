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
