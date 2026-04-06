from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/root/clone/ReconDreamer-RL")
MODULE_PATH = REPO_ROOT / "framework" / "utils" / "hugsim_execution.py"


def _load_module():
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("hugsim_execution", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controller_info_from_status_uses_speed_norm_and_preserves_steer() -> None:
    module = _load_module()

    info = module.controller_info_from_status(
        velocity_xy=np.array([3.0, 4.0], dtype=np.float32),
        steering_angle=0.12,
        dt=0.5,
        wheelbase=2.8,
    )

    np.testing.assert_allclose(info["ego_velo"], 5.0)
    np.testing.assert_allclose(info["ego_steer"], 0.12)
    np.testing.assert_allclose(info["dt"], 0.5)
    np.testing.assert_allclose(info["wheelbase"], 2.8)


def test_solve_hugsim_execution_converts_solver_rollout_to_tracker_execution() -> None:
    module = _load_module()

    prev_pose = np.eye(4, dtype=np.float64)
    plan_local = np.array(
        [
            [1.0, 0.2, 0.01],
            [2.2, 0.3, 0.03],
        ],
        dtype=np.float64,
    )
    state_trajectory = np.array(
        [
            [0.0, 0.0, 0.0, 2.0, 0.05],
            [1.1, 0.1, 0.02, 2.3, 0.08],
            [2.5, 0.3, 0.06, 2.4, 0.09],
        ],
        dtype=np.float64,
    )
    input_trajectory = np.array(
        [
            [0.4, 0.03],
            [0.2, 0.01],
        ],
        dtype=np.float64,
    )

    captured: dict[str, object] = {}

    def _fake_solve_sequence(plan_traj, info, *, wheelbase, plan_dt, control_dt, build_solver_fn):
        captured["plan_traj"] = np.asarray(plan_traj, dtype=np.float64)
        captured["info"] = dict(info)
        captured["wheelbase"] = float(wheelbase)
        captured["plan_dt"] = float(plan_dt)
        captured["control_dt"] = float(control_dt)
        captured["build_solver_fn"] = build_solver_fn
        return state_trajectory, input_trajectory

    result, meta = module.solve_hugsim_execution(
        prev_pose=prev_pose,
        plan_local_xyyaw=plan_local,
        velocity_xy=np.array([3.0, 4.0], dtype=np.float32),
        steering_angle=0.12,
        dt=0.5,
        wheelbase=2.8,
        solve_sequence_fn=_fake_solve_sequence,
        build_solver_fn="solver_builder",
    )

    np.testing.assert_allclose(
        captured["plan_traj"],
        np.array(
            [
                [-0.2, 1.0],
                [-0.3, 2.2],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(result.tracked_rollout_local_xyyaw, state_trajectory[1:, :3])
    np.testing.assert_allclose(result.tracked_first_local_xyyaw, np.array([1.1, 0.1, 0.02], dtype=np.float64))
    np.testing.assert_allclose(result.executed_local_xyyaw, np.array([1.1, 0.1, 0.02], dtype=np.float64))
    np.testing.assert_allclose(result.velocity_xy, np.array([2.3, 0.0], dtype=np.float32))
    np.testing.assert_allclose(result.acceleration_xy, np.array([0.4, 0.0], dtype=np.float32))
    np.testing.assert_allclose(result.command_state, np.array([0.4, 0.03], dtype=np.float64))
    np.testing.assert_allclose(meta["speed_next"], 2.3)
    np.testing.assert_allclose(meta["steer_next"], 0.08)
    np.testing.assert_allclose(meta["acc"], 0.4)
    np.testing.assert_allclose(meta["steer_rate"], 0.03)


def test_resolve_wheelbase_falls_back_to_default_when_loader_returns_none() -> None:
    module = _load_module()

    wheelbase = module.resolve_wheelbase(
        sparse_repo_path="/tmp/does-not-matter",
        explicit_wheelbase=None,
        load_wheelbase_fn=lambda _path: None,
    )

    np.testing.assert_allclose(wheelbase, 2.7)
