import json
import os
from pathlib import Path

import numpy as np
import torch

from tools.smalltool.visualize.generate_video_sparsedrive_v2 import (
    _build_shard_reward_rows,
    _build_auto_run_paths,
    _obs_tensor_to_camera_observation,
    _prepare_cuda_extension_env,
    _save_reward_detail_notebook,
    _write_run_manifest,
)


def test_save_reward_detail_notebook_includes_reward_rows_and_info(tmp_path: Path):
    out = tmp_path / "scene056_reward_detail.ipynb"
    reward_rows = [
        {
            "step": 0,
            "frame_before": 10,
            "frame_after": 15,
            "reward": -1.25,
            "cum_reward": -1.25,
            "progress_reward": 0.5,
            "cost_reward": -1.75,
            "done": False,
            "done_reason": "",
            "dynamic_collision": False,
            "static_collision": False,
            "collision_tokens": "veh_a",
        }
    ]
    debug_shard = {
        "meta": [
            {
                "step": 0,
                "info": {
                    "progress_term": 0.5,
                    "lateral_term": -0.1,
                    "yaw_term": -0.02,
                    "jerk_term": -0.03,
                    "nearest_dyn_dist_m": 3.2,
                    "collision_token": "veh_a",
                },
            }
        ]
    }

    saved = _save_reward_detail_notebook(
        out_path=str(out),
        scene=56,
        reward_rows=reward_rows,
        debug_shard=debug_shard,
        reward_cfg={"w_progress": 1.0, "w_yaw": 0.01},
        video_path="/tmp/scene056 video.mp4",
        reward_csv_path="/tmp/scene056_step_reward.csv",
        debug_shard_path="/tmp/scene056_debug_shard.pt",
    )

    assert saved == str(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    text = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
    assert "Scene 056 Reward Detail" in text
    assert "progress_term" in text
    assert "nearest_dyn_dist_m" in text
    assert "veh_a" in text
    assert "scene056_step_reward.csv" in text

    code = "\n".join("".join(cell.get("source", [])) for cell in data["cells"] if cell.get("cell_type") == "code")
    assert "video_path = '/tmp/scene056 video.mp4'" in code


def test_obs_tensor_to_camera_observation_reverses_rollout_camera_order():
    obs_t = torch.zeros(18, 2, 3, dtype=torch.float32)
    for cam_idx in range(6):
        obs_t[cam_idx * 3 + 0].fill_(cam_idx / 10.0)
        obs_t[cam_idx * 3 + 1].fill_(cam_idx / 10.0 + 0.01)
        obs_t[cam_idx * 3 + 2].fill_(cam_idx / 10.0 + 0.02)

    obs = _obs_tensor_to_camera_observation(obs_t)

    def expected(cam_idx: int) -> list[int]:
        return np.rint(np.asarray([cam_idx / 10.0, cam_idx / 10.0 + 0.01, cam_idx / 10.0 + 0.02]) * 255.0).astype(np.uint8).tolist()

    assert list(obs.keys()) == ["front_left", "front", "front_right", "back_left", "back", "back_right"]
    assert obs["front_left"].shape == (2, 3, 3)
    assert obs["front_left"][0, 0].tolist() == expected(0)
    assert obs["front"][0, 0].tolist() == expected(1)
    assert obs["back_right"][0, 0].tolist() == expected(5)


def test_build_shard_reward_rows_includes_scene_frame_logp_and_done():
    shard = {
        "reward": torch.tensor([1.0, -2.5], dtype=torch.float32),
        "done": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "old_logp": torch.tensor([-0.1, -0.2], dtype=torch.float32),
        "replay": [
            {"scene_id": 502, "frame_idx": 20, "timestamp_s": 2.0, "mode_idx": 32, "traj_xyyaw": torch.zeros(8, 3)},
            {"scene_id": 502, "frame_idx": 25, "timestamp_s": 2.5, "mode_idx": 31, "traj_xyyaw": torch.ones(8, 3)},
        ],
    }

    rows = _build_shard_reward_rows(shard)

    assert rows == [
        {
            "step": 0,
            "scene_id": 502,
            "frame_idx": 20,
            "timestamp_s": 2.0,
            "mode_idx": 32,
            "old_logp": -0.10000000149011612,
            "reward": 1.0,
            "cum_reward": 1.0,
            "done": False,
            "traj_points": 8,
        },
        {
            "step": 1,
            "scene_id": 502,
            "frame_idx": 25,
            "timestamp_s": 2.5,
            "mode_idx": 31,
            "old_logp": -0.20000000298023224,
            "reward": -2.5,
            "cum_reward": -1.5,
            "done": True,
            "traj_points": 8,
        },
    ]


def test_build_auto_run_paths_uses_scene_timestamp_bundle_under_reward_check_dir():
    paths = _build_auto_run_paths(
        scene=123,
        timestamp="20260521-140501",
        output_root="/root/clone/ReconDreamer-RL/outputs/RewardCheckandVideo",
    )

    assert paths["run_dir"] == "/root/clone/ReconDreamer-RL/outputs/RewardCheckandVideo/scene123-20260521-140501"
    assert paths["artifacts_dir"] == "/root/clone/ReconDreamer-RL/outputs/RewardCheckandVideo/scene123-20260521-140501/artifacts"
    assert paths["video_path"].endswith("/artifacts/scene123_20260521-140501_sparsedrivev2_rollout.mp4")
    assert paths["traj_csv"].endswith("/artifacts/scene123_20260521-140501_sparsedrivev2_plan_frontframe.csv")
    assert paths["traj_plot"].endswith("/artifacts/scene123_20260521-140501_sparsedrivev2_expert_vs_ego_traj.svg")
    assert paths["run_manifest"].endswith("/scene123-20260521-140501/run_info.md")


def test_write_run_manifest_records_config_and_ckpt(tmp_path: Path):
    manifest = tmp_path / "scene123-20260521-140501" / "run_info.md"

    saved = _write_run_manifest(
        manifest_path=str(manifest),
        scene=123,
        config_path="/root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/example.yaml",
        ckpt_path="/root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt",
        timestamp="20260521-140501",
        extra_lines=["mode_select=greedy", "cuda=0"],
    )

    assert saved == str(manifest)
    text = manifest.read_text(encoding="utf-8")
    assert "scene123-20260521-140501" in text
    assert "example.yaml" in text
    assert "sparsedrive_navsimv2.ckpt" in text
    assert "mode_select=greedy" in text


def test_prepare_cuda_extension_env_sets_expected_paths(monkeypatch):
    for key in [
        "CUDA_HOME",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "TORCH_EXTENSIONS_DIR",
    ]:
        monkeypatch.delenv(key, raising=False)

    _prepare_cuda_extension_env()

    assert os.environ["CUDA_HOME"] == "/usr/local/cuda"
    assert "/usr/local/cuda/include" in os.environ["CPATH"].split(os.pathsep)
    assert os.environ["TORCH_EXTENSIONS_DIR"].endswith(".cache/torch_extensions")
