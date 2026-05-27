from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_hugsim_visualize_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "smalltool"
        / "visualize"
        / "generate_video_sparsedrive_v2-HUGSIMori.py"
    )
    spec = importlib.util.spec_from_file_location("generate_video_sparsedrive_v2_HUGSIMori", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_hugsim_shard_run_paths_groups_shard_and_replay_outputs():
    module = _load_hugsim_visualize_module()

    paths = module._build_hugsim_shard_run_paths(
        timestamp="20260524-120102",
        output_root="/tmp/recondreamer",
        label="scene-0013",
    )

    assert paths["run_dir"] == "/tmp/recondreamer/scene-0013-20260524-120102"
    assert paths["artifacts_dir"] == "/tmp/recondreamer/scene-0013-20260524-120102/artifacts"
    assert paths["shard_path"].endswith("/artifacts/scene-0013_20260524-120102_actor_learner_shard.pt")
    assert paths["video_path"].endswith("/artifacts/scene-0013_20260524-120102_hugsim_online.mp4")
    assert paths["traj_csv"].endswith("/artifacts/scene-0013_20260524-120102_shard_plan.csv")
    assert paths["traj_plot"].endswith("/artifacts/scene-0013_20260524-120102_shard_bev.svg")


def test_resolve_hugsim_online_output_paths_uses_collection_bundle_defaults():
    module = _load_hugsim_visualize_module()
    args = argparse.Namespace(
        out=None,
        traj_csv=None,
        traj_plot=None,
    )
    paths = {
        "video_path": "/tmp/run/artifacts/video.mp4",
        "traj_csv": "/tmp/run/artifacts/plan.csv",
        "traj_plot": "/tmp/run/artifacts/plan.svg",
    }

    resolved = module._resolve_hugsim_online_output_paths(args, paths)

    assert resolved == {
        "out": "/tmp/run/artifacts/video.mp4",
        "traj_csv": "/tmp/run/artifacts/plan.csv",
        "traj_plot": "/tmp/run/artifacts/plan.svg",
    }


def test_resolve_hugsim_collect_defaults_uses_32_step_online_visualization_default():
    module = _load_hugsim_visualize_module()
    args = argparse.Namespace(horizon=None, mode_select=None)
    cfg = {
        "env": {"max_steps": 30},
        "train": {
            "eta": 0.7,
            "mode_idx": 12,
            "policy_mode_select": "sample",
            "actor_learner": {"actor_horizon": 8},
        },
    }

    defaults = module._resolve_hugsim_collect_defaults(args, cfg)

    assert defaults == {"horizon": 32, "eta": 0.7, "mode_idx": 12, "mode_select": "sample"}


def test_parse_hugsim_short_form_defaults_to_collect_visualization():
    module = _load_hugsim_visualize_module()

    args = module._parse_args(
        [
            "--scene",
            "0051",
            "--config",
            "/tmp/hugsim.yaml",
            "--ckpt",
            "/tmp/model.ckpt",
        ]
    )

    assert args.collect_hugsim_shard is True
    assert args.horizon == 32
    assert args.fps == 2.0
    assert args.reward_detail_format == "ipynb"
    assert args.save_keyframes is True
    assert args.out is None
    assert args.traj_csv is None
    assert args.traj_plot is None


def test_build_hugsim_shard_run_paths_uses_hugsim_output_root_by_default():
    module = _load_hugsim_visualize_module()

    paths = module._build_hugsim_shard_run_paths(timestamp="20260524-120102", label="scene-0051")

    assert "/outputs/RewardCheckandVideo/HUGSIM/" in paths["run_dir"]
    assert paths["video_path"].endswith("/artifacts/scene-0051_20260524-120102_hugsim_online.mp4")


def test_render_hugsim_online_bev_prefers_collision_scene_context(monkeypatch):
    module = _load_hugsim_visualize_module()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    traj = np.zeros((8, 3), dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    calls = []

    def fake_collision(frame_in, *, world_pose, snap, view_m):
        calls.append(("collision", snap["tag"], view_m))
        out = frame_in.copy()
        out[:, :, 0] = 99
        return out, ["veh"]

    def fake_plan(frame_in, traj_xyyaw, *, view_m):
        calls.append(("plan", view_m))
        out = frame_in.copy()
        out[:, :, 1] = 77
        return out

    monkeypatch.setattr(module, "_draw_collision_bev", fake_collision)
    monkeypatch.setattr(module, "_draw_shard_plan_bev", fake_plan)

    out, hit_tokens = module._render_hugsim_online_bev(
        frame,
        traj_xyyaw=traj,
        world_pose=pose,
        snap={"tag": "scene"},
        draw_collision_bev=True,
    )

    assert calls == [("collision", "scene", 25.0)]
    assert hit_tokens == ["veh"]
    assert out[:, :, 0].mean() == 99


def test_render_hugsim_online_bev_keeps_collision_style_when_snapshot_missing(monkeypatch):
    module = _load_hugsim_visualize_module()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)
    calls = []

    def fake_collision(frame_in, *, world_pose, snap, view_m):
        calls.append(("collision", snap, view_m))
        return frame_in, []

    def fake_plan(frame_in, traj_xyyaw, *, view_m):
        calls.append(("plan", view_m))
        return frame_in

    monkeypatch.setattr(module, "_draw_collision_bev", fake_collision)
    monkeypatch.setattr(module, "_draw_shard_plan_bev", fake_plan)

    module._render_hugsim_online_bev(
        frame,
        traj_xyyaw=np.zeros((8, 3), dtype=np.float32),
        world_pose=pose,
        snap=None,
        draw_collision_bev=True,
    )

    assert calls == [("collision", {}, 25.0)]


def test_hugsim_box_poly_xy_matches_hugsim_rectangle_convention():
    module = _load_hugsim_visualize_module()

    poly = module._hugsim_box_poly_xy([0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0])

    np.testing.assert_allclose(
        poly,
        np.asarray([[2.0, 1.0], [2.0, -1.0], [-2.0, -1.0], [-2.0, 1.0]], dtype=np.float64),
    )


def test_render_hugsim_online_bev_prefers_hugsim_native_boxes(monkeypatch):
    module = _load_hugsim_visualize_module()
    frame = np.zeros((160, 200, 3), dtype=np.uint8)
    traj = np.zeros((8, 3), dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    calls = []

    def fake_collision(frame_in, *, world_pose, snap, view_m):
        calls.append(("collision", snap, view_m))
        return frame_in, ["recon-cache-hit"]

    monkeypatch.setattr(module, "_draw_collision_bev", fake_collision)

    out, hit_tokens = module._render_hugsim_online_bev(
        frame,
        traj_xyyaw=traj,
        world_pose=pose,
        snap={"dynamic_objects": [{"poly": [[0, 0], [1, 0], [1, 1]], "token": "wrong-source"}]},
        draw_collision_bev=True,
        hugsim_ego_box=[0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
        hugsim_obj_boxes=[
            [0.25, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [30.0, 30.0, 0.0, 2.0, 4.0, 1.5, 0.0],
        ],
        hugsim_collision=True,
    )

    assert calls == []
    assert hit_tokens == ["obj0"]
    assert out.sum() > frame.sum()


def test_render_hugsim_online_bev_prefers_aligned_recon_global_objects(monkeypatch):
    module = _load_hugsim_visualize_module()
    frame = np.zeros((160, 200, 3), dtype=np.uint8)
    calls = []
    hit_tokens = []

    def fake_collision(frame_in, *, world_pose, snap, view_m):
        calls.append(("collision", snap, view_m))
        return frame_in, ["wrong-source"]

    def fake_aligned(frame_in, *, ego_poly, hugsim_objects, recon_objects, view_m):
        calls.append(("aligned", view_m))
        hit_tokens.extend([hugsim_objects[0]["token"], recon_objects[0]["token"]])
        return frame_in + 1, list(hit_tokens)

    monkeypatch.setattr(module, "_draw_collision_bev", fake_collision)
    monkeypatch.setattr(module, "_draw_aligned_recon_global_bev", fake_aligned)

    out, tokens = module._render_hugsim_online_bev(
        frame,
        traj_xyyaw=np.zeros((8, 3), dtype=np.float32),
        world_pose=np.eye(4, dtype=np.float32),
        snap={"dynamic_objects": [{"poly": [[0, 0], [1, 0], [1, 1]], "token": "wrong-source"}]},
        draw_collision_bev=True,
        hugsim_ego_box=[0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
        hugsim_obj_boxes=[[0.25, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]],
        aligned_ego_poly=[[2.0, 1.0], [2.0, -1.0], [-2.0, -1.0], [-2.0, 1.0]],
        hugsim_recon_objects=[
            {"source": "hugsim_inserted", "token": "hugsim_obj_0", "poly": [[1, 1], [2, 1], [2, 2], [1, 2]]}
        ],
        recon_cache_objects=[{"source": "recon_cache", "token": "recon_veh_a", "poly": [[3, 1], [4, 1], [4, 2], [3, 2]]}],
    )

    assert calls == [("aligned", 25.0)]
    assert tokens == ["hugsim_obj_0", "recon_veh_a"]
    assert out.sum() > frame.sum()


def test_apply_cli_ckpt_override_updates_agent_config_used_by_build_agent():
    module = _load_hugsim_visualize_module()
    cfg = {"agent": {"type": "sparsedrive_v2", "ckpt": "old.ckpt"}}

    module._apply_cli_ckpt_override(cfg, "/abs/new.ckpt")

    assert cfg["agent"]["ckpt"] == "/abs/new.ckpt"


def test_apply_hugsim_scene_override_uses_numeric_scene_argument():
    module = _load_hugsim_visualize_module()
    cfg = {"env": {"hugsim": {"scenes": ["scene-0013"]}}}
    args = argparse.Namespace(scene=10, hugsim_scene=None)

    selected = module._apply_hugsim_scene_override(cfg, args)

    assert selected == "scene-0010"
    assert cfg["env"]["hugsim"]["scenes"] == ["scene-0010"]


def test_apply_hugsim_scene_override_prefers_explicit_hugsim_scene():
    module = _load_hugsim_visualize_module()
    cfg = {"env": {"hugsim": {"scenes": ["scene-0013"]}}}
    args = argparse.Namespace(scene=10, hugsim_scene="scene-0038-hard-00")

    selected = module._apply_hugsim_scene_override(cfg, args)

    assert selected == "scene-0038-hard-00"
    assert cfg["env"]["hugsim"]["scenes"] == ["scene-0038-hard-00"]


def test_grid_frame_available_cameras_handles_hugsim_front_only_observation():
    module = _load_hugsim_visualize_module()
    left = np.full((2, 3, 3), 10, dtype=np.uint8)
    front = np.full((2, 3, 3), 20, dtype=np.uint8)
    right = np.full((2, 3, 3), 30, dtype=np.uint8)

    frame = module._grid_frame_available_cameras({"front_left": left, "front": front, "front_right": right})

    assert frame.shape == (2, 9, 3)
    assert frame[:, :3].mean() == 10
    assert frame[:, 3:6].mean() == 20
    assert frame[:, 6:].mean() == 30


def test_grid_frame_available_cameras_renders_six_camera_grid():
    module = _load_hugsim_visualize_module()
    obs = {
        "front_left": np.full((2, 3, 3), 10, dtype=np.uint8),
        "front": np.full((2, 3, 3), 20, dtype=np.uint8),
        "front_right": np.full((2, 3, 3), 30, dtype=np.uint8),
        "back_left": np.full((2, 3, 3), 40, dtype=np.uint8),
        "back": np.full((2, 3, 3), 50, dtype=np.uint8),
        "back_right": np.full((2, 3, 3), 60, dtype=np.uint8),
    }

    frame = module._grid_frame_available_cameras(obs)

    assert frame.shape == (4, 9, 3)
    assert frame[:2, :3].mean() == 10
    assert frame[:2, 3:6].mean() == 20
    assert frame[:2, 6:].mean() == 30
    assert frame[2:, :3].mean() == 40
    assert frame[2:, 3:6].mean() == 50
    assert frame[2:, 6:].mean() == 60


def test_build_hugsim_online_reward_row_includes_one_step_reward_metadata():
    module = _load_hugsim_visualize_module()
    replay = {
        "scene_id": 13,
        "frame_idx": 25,
        "sample_token": "abc123",
        "timestamp_s": 2.5,
        "mode_idx": 7,
        "traj_xyyaw": torch.zeros(8, 3),
    }
    info = {
        "scene_id": 13,
        "frame_idx": 30,
        "sample_token": "def456",
        "done_reason": "timeout",
        "progress_reward": 1.2,
        "cost_reward": -0.4,
    }

    row = module._build_hugsim_online_reward_row(
        step=2,
        replay=replay,
        logp=torch.tensor(-0.25),
        reward=0.8,
        cum_reward=1.5,
        done=True,
        info=info,
    )

    assert row == {
        "step": 2,
        "scene_id": 13,
        "frame_idx": 25,
        "frame_after": 30,
        "sample_token": "abc123",
        "sample_token_after": "def456",
        "timestamp_s": 2.5,
        "mode_idx": 7,
        "old_logp": -0.25,
        "reward": 0.8,
        "cum_reward": 1.5,
        "done": True,
        "done_reason": "timeout",
        "progress_reward": 1.2,
        "cost_reward": -0.4,
        "traj_points": 8,
    }


def test_apply_hugsim_online_reward_fallback_uses_route_delta_for_mixed_step_path():
    module = _load_hugsim_visualize_module()
    info = {
        "reward_mode": "step_path",
        "reward": 0.0,
        "lateral_error_m": 1225.0,
        "completion_ratio": 1.0,
        "hugsim_route_completion": 0.169,
        "collision": False,
    }

    reward, next_route_completion = module._apply_hugsim_online_reward_fallback(
        reward=0.0,
        info=info,
        prev_route_completion=0.120,
    )

    assert reward == 0.049
    assert next_route_completion == 0.169
    assert info["reward_mode"] == "hugsim_route_delta_fallback"
    assert info["raw_recondreamer_reward"] == 0.0
    assert info["invalid_recon_step_path"] is True
    assert info["progress_reward"] == 0.049
    assert info["cost_reward"] == 0.0


def test_hugsim_online_reward_keeps_actor_learner_reward_when_alignment_valid():
    module = _load_hugsim_visualize_module()
    info = {
        "reward_mode": "step_path",
        "hugsim_recon_alignment_valid": True,
        "lateral_error_m": 0.4,
        "hugsim_route_completion": 0.2,
    }

    reward, prev = module._apply_hugsim_online_reward_fallback(
        reward=1.25,
        info=info,
        prev_route_completion=0.1,
    )

    assert reward == 1.25
    assert prev == 0.2
    assert info["reward_mode"] == "step_path"
    assert "raw_recondreamer_reward" not in info
