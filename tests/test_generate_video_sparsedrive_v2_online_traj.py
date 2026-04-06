from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/root/clone/ReconDreamer-RL")
MODULE_PATH = REPO_ROOT / "tools" / "smalltool" / "visualize" / "generate_video_sparsedrive_v2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_video_sparsedrive_v2", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_expert_front_xz_for_frame_uses_start_frame_front_origin(tmp_path: Path) -> None:
    module = _load_module()

    base = tmp_path / "data" / "099"
    ego_pose_dir = base / "ego_pose"
    cam2ego_dir = base / "cam2ego"
    ego_pose_dir.mkdir(parents=True)
    cam2ego_dir.mkdir(parents=True)

    np.savetxt(ego_pose_dir / "000.txt", np.eye(4, dtype=np.float64))
    pose_005 = np.eye(4, dtype=np.float64)
    pose_005[0, 3] = 3.0
    pose_005[2, 3] = 7.0
    np.savetxt(ego_pose_dir / "005.txt", pose_005)

    cam2ego0 = np.eye(4, dtype=np.float64)
    cam2ego0[0, 3] = 1.0
    np.savetxt(cam2ego_dir / "0.txt", cam2ego0)

    xz = module._load_expert_front_xz_for_frame(
        scene=99,
        start_frame=0,
        frame_idx=5,
        base_data_dir=str(tmp_path / "data"),
    )

    np.testing.assert_allclose(xz, np.array([2.0, 7.0], dtype=np.float64))


def test_append_online_expert_xz_accumulates_stepwise_values() -> None:
    module = _load_module()

    seq: list[list[float]] = []
    module._append_online_expert_xz(seq, np.array([1.5, -2.0], dtype=np.float64))
    module._append_online_expert_xz(seq, np.array([3.0, 4.5], dtype=np.float64))

    np.testing.assert_allclose(np.asarray(seq, dtype=np.float64), np.array([[1.5, -2.0], [3.0, 4.5]], dtype=np.float64))


def test_relative_local_xyyaw_returns_local_delta() -> None:
    module = _load_module()

    prev = np.eye(4, dtype=np.float64)
    rel = module._pose_matrix_from_xyyaw(1.0, 0.0, np.pi / 2.0)
    nxt = prev @ rel

    delta = module._relative_local_xyyaw(prev, nxt)

    np.testing.assert_allclose(delta, np.array([1.0, 0.0, np.pi / 2.0], dtype=np.float64), atol=1e-7)


def test_build_online_step_stats_paths_uses_traj_plot_prefix(tmp_path: Path) -> None:
    module = _load_module()

    traj_plot = tmp_path / "scene099_demo_expert_vs_ego_traj.svg"

    paths = module._build_online_step_stats_paths(str(traj_plot))

    assert paths["per_step_csv"].endswith("scene099_demo_online_step_summary.csv")
    assert paths["aggregate_csv"].endswith("scene099_demo_online_step_aggregate.csv")
    assert paths["rollout_csv"].endswith("scene099_demo_online_rollout_points.csv")
    assert paths["overlay_svg"].endswith("scene099_demo_online_rollout_overlay.svg")


def test_save_traj_plot_xz_marks_periodic_step_indices(tmp_path: Path) -> None:
    module = _load_module()

    expert_xz = np.stack((np.arange(11, dtype=np.float64), np.arange(11, dtype=np.float64) * 0.5), axis=1)
    ego_xz = expert_xz + np.array([0.2, -0.1], dtype=np.float64)
    out_path = tmp_path / "scene099_demo_expert_vs_ego_traj.svg"

    saved = module._save_traj_plot_xz(scene=99, expert_xz=expert_xz, ego_xz=ego_xz, out_path=str(out_path))

    assert saved is True
    svg = out_path.read_text(encoding="utf-8")
    assert "step 0" in svg
    assert "step 5" in svg
    assert "every 5 steps" in svg
