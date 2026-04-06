from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/root/clone/ReconDreamer-RL")
MODULE_PATH = REPO_ROOT / "tools" / "smalltool" / "visualize" / "generate_video_sparsedrive_v2_hugsim_controller.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_video_sparsedrive_v2_hugsim_controller", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_local_xyyaw_to_hugsim_plan_xy_matches_axis_convention() -> None:
    module = _load_module()

    plan_local = np.array(
        [
            [2.0, 1.5, 0.1],
            [4.0, -0.5, 0.2],
        ],
        dtype=np.float64,
    )

    hugsim_xy = module._plan_local_xyyaw_to_hugsim_plan_xy(plan_local)

    np.testing.assert_allclose(
        hugsim_xy,
        np.array(
            [
                [-1.5, 2.0],
                [0.5, 4.0],
            ],
            dtype=np.float64,
        ),
    )


def test_hugsim_one_step_local_xyyaw_uses_updated_speed_and_steer() -> None:
    module = _load_module()

    step = module._hugsim_one_step_local_xyyaw(
        speed=2.0,
        steer=0.0,
        acc=1.0,
        steer_rate=0.1,
        dt=0.5,
        wheelbase=2.7,
    )

    np.testing.assert_allclose(step["speed_next"], 2.5)
    np.testing.assert_allclose(step["steer_next"], 0.05)
    np.testing.assert_allclose(step["local_xyyaw"][0], 1.25)
    np.testing.assert_allclose(step["local_xyyaw"][1], 0.0)
    np.testing.assert_allclose(step["local_xyyaw"][2], 2.5 * np.tan(0.05) / 2.7 * 0.5)


def test_build_hugsim_controller_info_uses_speed_norm_and_sim_steer() -> None:
    module = _load_module()

    class _Sim:
        _status_steering_angle = 0.12

    obs = {
        "ego_velocity": np.array([3.0, 4.0], dtype=np.float32),
    }

    info = module._build_hugsim_controller_info(
        obs=obs,
        sim=_Sim(),
        dt=0.5,
        wheelbase=2.8,
    )

    np.testing.assert_allclose(info["ego_velo"], 5.0)
    np.testing.assert_allclose(info["ego_steer"], 0.12)
    np.testing.assert_allclose(info["dt"], 0.5)
    np.testing.assert_allclose(info["wheelbase"], 2.8)


def test_build_execution_from_hugsim_solution_uses_full_state_trajectory() -> None:
    module = _load_module()

    prev_pose = np.eye(4, dtype=np.float64)
    plan_local = np.array([[1.0, 0.0, 0.0], [2.0, 0.2, 0.1]], dtype=np.float64)
    state_trajectory = np.array(
        [
            [0.0, 0.0, 0.0, 2.0, 0.0],
            [1.1, 0.1, 0.02, 2.2, 0.05],
            [2.4, 0.4, 0.08, 2.4, 0.07],
        ],
        dtype=np.float64,
    )
    input_trajectory = np.array(
        [
            [0.4, 0.1],
            [0.2, 0.04],
        ],
        dtype=np.float64,
    )

    execution = module._build_execution_from_hugsim_solution(
        prev_pose=prev_pose,
        plan_local_xyyaw=plan_local,
        state_trajectory=state_trajectory,
        input_trajectory=input_trajectory,
    )

    np.testing.assert_allclose(execution.tracked_rollout_local_xyyaw, state_trajectory[1:, :3])
    np.testing.assert_allclose(execution.tracked_first_local_xyyaw, np.array([1.1, 0.1, 0.02], dtype=np.float64))
    np.testing.assert_allclose(execution.velocity_xy, np.array([2.2, 0.0], dtype=np.float32))
    np.testing.assert_allclose(execution.acceleration_xy, np.array([0.4, 0.0], dtype=np.float32))
    np.testing.assert_allclose(execution.command_state, np.array([0.4, 0.1], dtype=np.float64))
