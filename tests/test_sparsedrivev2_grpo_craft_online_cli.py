from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.smalltool.visualize.visualize_sparsedrivev2_grpo_craft_online import (
    _REPO_ROOT,
    _prime_sim_external_plan,
    _prepare_cuda_extension_env,
    _ego_history_xy_in_current_frame,
    build_default_paths,
    build_candidate_score_payload,
    candidate_visual_styles,
    overlay_top_right_inset,
    render_bev_debug_image,
    write_score_payload,
)


def test_build_default_paths_are_under_output_directory(tmp_path: Path) -> None:
    paths = build_default_paths(out_dir=tmp_path, scene=42)

    assert paths.video == tmp_path / "scene_042.mp4"
    assert paths.frames_dir == tmp_path / "frames"
    assert paths.bev_dir == tmp_path / "bev"
    assert paths.scores_dir == tmp_path / "scores"


def test_write_score_payload_creates_step_json(tmp_path: Path) -> None:
    payload = build_candidate_score_payload(
        scene=3,
        step=4,
        frame_idx=20,
        sample_token="tok",
        traj_xyyaw=np.zeros((2, 3, 3), dtype=np.float32),
        scores=np.asarray([0.1, 0.9], dtype=np.float32),
        score_logits=None,
        mode_indices=np.asarray([5, 6], dtype=np.int64),
        top_k=1,
    )

    path = write_score_payload(tmp_path, payload)

    assert path == tmp_path / "step_000004.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["top_k_candidate_indices"] == [1]
    assert loaded["candidates"][0]["mode_index"] == 6


def test_build_default_paths_has_bev_directory(tmp_path: Path) -> None:
    paths = build_default_paths(out_dir=tmp_path, scene=1)
    assert paths.bev_dir == tmp_path / "bev"


def test_prepare_cuda_extension_env_sets_jit_paths(monkeypatch) -> None:
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

    assert Path(__import__("os").environ["CUDA_HOME"]) == Path("/usr/local/cuda")
    assert "/usr/local/cuda/include" in __import__("os").environ["CPATH"].split(":")
    assert __import__("os").environ["TORCH_EXTENSIONS_DIR"] == str(_REPO_ROOT / ".cache" / "torch_extensions")


def test_ego_history_projection_uses_current_ego_frame() -> None:
    pose0 = np.eye(4, dtype=np.float64)
    pose1 = np.eye(4, dtype=np.float64)
    pose1[0, 3] = 10.0
    pose1[1, 3] = 2.0
    pose2 = np.eye(4, dtype=np.float64)
    pose2[0, 3] = 13.0
    pose2[1, 3] = 5.0

    history = _ego_history_xy_in_current_frame([pose0, pose1], pose2)

    assert history.shape == (2, 2)
    assert history[0].tolist() == pytest.approx([-13.0, -5.0])
    assert history[1].tolist() == pytest.approx([-3.0, -3.0])


def test_prime_sim_external_plan_requires_selected_horizon_array() -> None:
    class Sim:
        pass

    sim = Sim()
    plan = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    _prime_sim_external_plan(sim, plan)

    assert hasattr(sim, "_external_plan_local_xyyaw")
    assert sim._external_plan_local_xyyaw.shape == (2, 3)
    assert sim._external_plan_local_xyyaw[0].tolist() == pytest.approx([1.0, 2.0, 3.0])
