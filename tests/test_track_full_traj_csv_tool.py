from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path("/root/clone/ReconDreamer-RL")
MODULE_PATH = REPO_ROOT / "outputs" / "visualize" / "debug_tracker_scene099" / "track_full_traj_csv_with_tracker.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("track_full_traj_csv_with_tracker", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_plan_from_csv_reads_sorted_xyyaw(tmp_path: Path) -> None:
    module = _load_module()

    csv_path = tmp_path / "traj.csv"
    pd.DataFrame(
        [
            {"frame": 2, "x": 2.0, "y": 0.1, "yaw_xy_rad_signed": 0.2},
            {"frame": 0, "x": 0.0, "y": 0.0, "yaw_xy_rad_signed": 0.0},
            {"frame": 1, "x": 1.0, "y": 0.0, "yaw_xy_rad_signed": 0.1},
        ]
    ).to_csv(csv_path, index=False)

    frames, plan = module.load_plan_from_csv(str(csv_path))

    np.testing.assert_array_equal(frames, np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_allclose(
        plan,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.1],
                [2.0, 0.1, 0.2],
            ],
            dtype=np.float64,
        ),
    )


def test_estimate_initial_kinematic_state_uses_first_two_segments() -> None:
    module = _load_module()

    plan = np.array(
        [
            [1.0, 0.0, 0.05],
            [2.0, 0.0, 0.10],
            [3.0, 0.0, 0.15],
        ],
        dtype=np.float64,
    )

    vel_xy, acc_xy = module.estimate_initial_kinematic_state(plan, dt=0.5)

    np.testing.assert_allclose(vel_xy, np.array([2.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(acc_xy, np.array([0.0, 0.0], dtype=np.float64))


def test_repeat_first_step_rollout_accumulates_in_se2() -> None:
    module = _load_module()

    step = np.array([1.0, 0.0, np.pi / 2.0], dtype=np.float64)

    rollout = module.repeat_first_step_rollout(step, count=3)

    np.testing.assert_allclose(
        rollout,
        np.array(
            [
                [1.0, 0.0, np.pi / 2.0],
                [1.0, 1.0, np.pi],
                [0.0, 1.0, -np.pi / 2.0],
            ],
            dtype=np.float64,
        ),
        atol=1e-7,
    )


def test_perfect_transfer_rollout_is_identity_on_plan() -> None:
    module = _load_module()

    plan = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.3]], dtype=np.float64)

    rollout = module.perfect_transfer_rollout(plan)

    np.testing.assert_allclose(rollout, plan)
    assert rollout is not plan


def test_local_plan_to_front_xz_uses_inverse_cam2ego_transform() -> None:
    module = _load_module()

    cam2ego0 = module._pose_from_local_xyyaw(1.0, 0.0, 0.0)
    plan = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )

    front_xz = module.local_plan_to_front_xz(plan, cam2ego0)

    np.testing.assert_allclose(
        front_xz,
        np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
