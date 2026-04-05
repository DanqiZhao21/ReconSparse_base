import math
import pathlib
import sys
import types

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_recon_simulator_class():
    if "gymnasium" not in sys.modules:
        gym_stub = types.ModuleType("gymnasium")

        class _Env:
            pass

        class _Space:
            def __init__(self, *args, **kwargs):
                del args, kwargs

        gym_stub.Env = _Env
        gym_stub.spaces = types.SimpleNamespace(Box=_Space, Dict=_Space, MultiDiscrete=_Space)
        sys.modules["gymnasium"] = gym_stub

    if "framework.env_wrapper" not in sys.modules:
        env_wrapper_stub = types.ModuleType("framework.env_wrapper")
        env_wrapper_stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules["framework.env_wrapper"] = env_wrapper_stub

    if "framework.env_wrapper.tool" not in sys.modules:
        tool_stub = types.ModuleType("framework.env_wrapper.tool")

        def _unsupported(*args, **kwargs):
            raise RuntimeError("tool stub should not be executed in nearest expert info tests")

        tool_stub.get_splat = _unsupported
        tool_stub.get_sky_view = _unsupported
        tool_stub.move_to_device = lambda value, *args, **kwargs: value
        tool_stub.slerp = _unsupported
        sys.modules["framework.env_wrapper.tool"] = tool_stub
        setattr(sys.modules["framework.env_wrapper"], "tool", tool_stub)

    from reconsimulator.envs.nus import ReconSimulator

    return ReconSimulator


def _pose_from_xzyaw(x: float, z: float, yaw_rad: float) -> np.ndarray:
    c = float(math.cos(float(yaw_rad)))
    s = float(math.sin(float(yaw_rad)))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=np.float64,
    )
    pose[0, 3] = float(x)
    pose[2, 3] = float(z)
    return pose


def test_get_info_reports_nearest_expert_tracking_metrics():
    ReconSimulator = _load_recon_simulator_class()
    env = ReconSimulator.__new__(ReconSimulator)

    env.start_ego = _pose_from_xzyaw(10.2, 1.3, math.radians(25.0))
    env.expert_world_all = [
        _pose_from_xzyaw(0.0, 0.0, math.radians(0.0)),
        _pose_from_xzyaw(10.0, 1.0, math.radians(10.0)),
        _pose_from_xzyaw(30.0, -5.0, math.radians(-30.0)),
    ]
    env.expert_pair = [pose[:3, 3][[0, 2]] for pose in env.expert_world_all]

    info = env._get_info()

    assert int(info["nearest_expert_idx"]) == 1
    assert np.allclose(info["exp_pos"], np.asarray([10.0, 0.0, 1.0], dtype=np.float32))
    assert np.allclose(info["act_pos"], np.asarray([10.2, 0.0, 1.3], dtype=np.float32))
    assert math.isclose(float(info["xz_err_m"]), math.sqrt(0.2**2 + 0.3**2), rel_tol=1e-6)
    assert math.isclose(float(info["xy_err_m"]), float(info["xz_err_m"]), rel_tol=1e-6)
    assert math.isclose(float(info["exp_yaw_deg"]), 10.0, rel_tol=1e-6)
    assert math.isclose(float(info["act_yaw_deg"]), 25.0, rel_tol=1e-6)
    assert math.isclose(float(info["yaw_err_deg"]), 15.0, rel_tol=1e-6)
