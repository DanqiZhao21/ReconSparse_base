import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from track_first_step_csv_tool import (
    _build_local_plan_from_absolute_poses,
    _compute_axis_limits_with_padding,
    _downsample_frame_indices,
    _default_output_paths,
    _resolve_output_paths,
    _uniform_sample_indices,
    _valid_start_indices,
    _infer_scene_id_from_csv_path,
    _pose_from_xyyaw,
)


def test_infer_scene_id_from_csv_path():
    path = "/root/clone/ReconDreamer-RL/outputs/visualize/trajTransition-scene099/expert_ego_local_frame.csv"
    assert _infer_scene_id_from_csv_path(path) == 99


def test_build_local_plan_from_absolute_poses_straight_line():
    poses = np.stack(
        [_pose_from_xyyaw(float(i), 0.0, 0.0) for i in range(6)],
        axis=0,
    )

    plan = _build_local_plan_from_absolute_poses(
        absolute_poses=poses,
        start_index=1,
        step_stride=2,
        horizon_points=3,
    )

    assert plan.shape == (3, 3)
    assert np.allclose(plan[0], np.asarray([2.0, 0.0, 0.0], dtype=np.float64))
    assert np.allclose(plan[1], np.asarray([4.0, 0.0, 0.0], dtype=np.float64))
    assert np.allclose(plan[2], np.asarray([4.0, 0.0, 0.0], dtype=np.float64))


def test_default_output_paths_returns_four_svgs():
    paths = _default_output_paths(
        "/root/clone/ReconDreamer-RL/outputs/visualize/trajTransition-scene099/expert_ego_local_frame.csv"
    )

    assert sorted(paths.keys()) == [
        "absolute_xy",
        "absolute_xy_sampled18",
        "local_first_step",
        "tracking_error",
        "velocity_accel",
    ]
    assert paths["absolute_xy"].endswith("tracked_first_step_exact_absolute_xy.svg")
    assert paths["absolute_xy_sampled18"].endswith("tracked_first_step_exact_absolute_xy_sampled18.svg")
    assert paths["local_first_step"].endswith("tracked_first_step_exact_local_first_step.svg")
    assert paths["tracking_error"].endswith("tracked_first_step_exact_tracking_error.svg")
    assert paths["velocity_accel"].endswith("tracked_first_step_exact_velocity_accel.svg")


def test_compute_axis_limits_with_padding_enforces_min_span():
    values = np.asarray([-0.01, 0.0, 0.02], dtype=np.float64)
    lo, hi = _compute_axis_limits_with_padding(values, min_span=0.3, pad_ratio=0.1)

    assert hi > lo
    assert (hi - lo) >= 0.3
    assert lo < float(values.min())
    assert hi > float(values.max())


def test_uniform_sample_indices_returns_18_evenly_spaced_points():
    idx = _uniform_sample_indices(total_count=194, sample_count=18)

    assert len(idx) == 18
    assert idx[0] == 0
    assert idx[-1] == 193
    assert idx == sorted(idx)
    assert len(set(idx)) == 18


def test_downsample_frame_indices_keeps_every_fifth_row():
    idx = _downsample_frame_indices(total_count=196, stride=5)

    assert idx == list(range(0, 196, 5))
    assert len(idx) == 40
    assert idx[-1] == 195


def test_valid_start_indices_require_full_horizon():
    idx = _valid_start_indices(
        total_count=40,
        step_stride=1,
        horizon_points=8,
        require_full_horizon=True,
    )

    assert idx == list(range(32))


def test_resolve_output_paths_directory_includes_sampled_absolute_xy():
    paths = _resolve_output_paths(
        "/root/clone/ReconDreamer-RL/outputs/visualize/trajTransition-scene099/expert_ego_local_frame.csv",
        "/tmp/recon_track_outputs",
    )

    assert paths["absolute_xy"] == "/tmp/recon_track_outputs/tracked_first_step_exact_absolute_xy.svg"
    assert (
        paths["absolute_xy_sampled18"]
        == "/tmp/recon_track_outputs/tracked_first_step_exact_absolute_xy_sampled18.svg"
    )
