from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/root/clone/ReconDreamer-RL")
MODULE_PATH = REPO_ROOT / "framework" / "utils" / "tracker_execution.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("tracker_execution", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeStateIndex:
    X = 0
    Y = 1
    HEADING = 2
    VELOCITY_X = 3
    VELOCITY_Y = 4
    ACCELERATION_X = 5
    ACCELERATION_Y = 6
    STEERING_ANGLE = 7
    STEERING_RATE = 8


def test_local_xyyaw_from_state_row_extracts_pose_fields() -> None:
    module = _load_module()

    state_row = np.array([1.25, -0.5, 0.3, 4.0, -1.0, 0.2, -0.1, 0.05, 0.01], dtype=np.float64)

    local_xyyaw = module.local_xyyaw_from_state_row(state_row, _FakeStateIndex)

    np.testing.assert_allclose(local_xyyaw, np.array([1.25, -0.5, 0.3], dtype=np.float64))


def test_build_execution_result_prefers_propagated_state_for_next_pose() -> None:
    module = _load_module()

    prev_pose = np.eye(4, dtype=np.float64)
    tracked_rollout = np.array([[1.0, 2.0, 0.1], [2.0, 3.0, 0.2]], dtype=np.float64)
    tracked_first = tracked_rollout[0]
    propagated_state = np.array([3.0, 4.0, 0.25, 5.0, -0.5, 0.3, -0.2, 0.12, 0.04], dtype=np.float64)
    command_state = np.array([0.3, 0.04], dtype=np.float64)

    result = module.build_execution_result(
        prev_pose=prev_pose,
        tracked_rollout_local_xyyaw=tracked_rollout,
        tracked_first_local_xyyaw=tracked_first,
        velocity_xy=np.array([5.0, -0.5], dtype=np.float32),
        acceleration_xy=np.array([0.3, -0.2], dtype=np.float32),
        steering_angle=0.12,
        steering_rate=0.04,
        propagated_state=propagated_state,
        command_state=command_state,
        state_index=_FakeStateIndex,
    )

    np.testing.assert_allclose(result.executed_local_xyyaw, np.array([3.0, 4.0, 0.25], dtype=np.float64))
    np.testing.assert_allclose(result.executed_pose[:2, 3], np.array([3.0, 4.0], dtype=np.float64))
    np.testing.assert_allclose(result.tracked_first_local_xyyaw, tracked_first)
    np.testing.assert_allclose(result.command_state, command_state)
