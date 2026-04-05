import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from framework.rewards import TrackingRewardComputer


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


class _EnvStub:
    def __init__(self) -> None:
        self.start_ego = _pose_from_xzyaw(0.0, 0.0, 0.0)
        self.step_frames = 5
        self.now_frame = 5
        self.all_expert_ego = [
            _pose_from_xzyaw(0.0, 0.0, 0.0),
            _pose_from_xzyaw(100.0, 0.0, 0.0),
        ]
        self.expert_pair = [
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([100.0, 0.0], dtype=np.float32),
        ]


def test_tracking_reward_prefers_info_xz_err_for_position_deviation():
    computer = TrackingRewardComputer(
        {
            "dmax": 2.0,
            "psi_max_deg": 30.0,
            "w_pos": 2.0,
            "w_heading": 1.0,
        }
    )

    result = computer.compute(
        env=_EnvStub(),
        info={"xz_err_m": 3.0, "yaw_err_deg": 0.0},
        step_idx=0,
        done=False,
    )

    assert math.isclose(float(result.info["pos_dev"]), 3.0, rel_tol=1e-6)
    assert math.isclose(float(result.info["rpd"]), 2.0, rel_tol=1e-6)
    assert math.isclose(float(result.reward), -2.0, rel_tol=1e-6)


def test_tracking_reward_caps_dense_position_and_heading_penalties():
    computer = TrackingRewardComputer(
        {
            "dmax": 2.0,
            "psi_max_deg": 30.0,
            "w_pos": 2.0,
            "w_heading": 1.0,
            "rpd_cap": 5.0,
            "rhd_cap": 5.0,
        }
    )

    result = computer.compute(
        env=_EnvStub(),
        info={"xz_err_m": 20.0, "yaw_err_deg": 100.0},
        step_idx=0,
        done=False,
    )

    assert math.isclose(float(result.info["rpd"]), 5.0, rel_tol=1e-6)
    assert math.isclose(float(result.info["rhd"]), 5.0, rel_tol=1e-6)
    assert math.isclose(float(result.reward), -10.0, rel_tol=1e-6)
