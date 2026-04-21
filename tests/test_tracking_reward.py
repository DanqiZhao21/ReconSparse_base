from __future__ import annotations

import math

import numpy as np
import pytest

from framework.rewards.tracking import TrackingRewardComputer


def _pose(x: float, z: float, yaw_rad: float = 0.0) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    out = np.eye(4, dtype=np.float64)
    out[0, 0] = c
    out[0, 2] = -s
    out[2, 0] = s
    out[2, 2] = c
    out[0, 3] = float(x)
    out[2, 3] = float(z)
    return out


class _DummyEnv:
    def __init__(self, *, start_ego: np.ndarray, all_expert_ego: list[np.ndarray]) -> None:
        self.start_ego = np.asarray(start_ego, dtype=np.float64)
        self.all_expert_ego = [np.asarray(item, dtype=np.float64) for item in all_expert_ego]


def test_completion_ratio_bonus_grows_near_route_end() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 0.0,
            "w_lateral": 0.0,
            "w_yaw": 0.0,
            "w_completion_ratio": 2.0,
            "completion_ratio_power": 2.0,
        },
        "collision": {
            "w_static": 0.0,
            "w_dynamic": 0.0,
        },
        "comfort": {
            "w_longitudinal_jerk": 0.0,
            "w_yaw_jerk": 0.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0), _pose(0.0, 10.0)]

    early_env = _DummyEnv(start_ego=_pose(0.0, 2.0), all_expert_ego=reference_path)
    late_env = _DummyEnv(start_ego=_pose(0.0, 8.0), all_expert_ego=reference_path)

    early = computer.compute(env=early_env, info={}, step_idx=0, done=False)
    computer.reset()
    late = computer.compute(env=late_env, info={}, step_idx=0, done=False)

    assert early.info["completion_ratio"] == pytest.approx(0.2, abs=1.0e-4)
    assert late.info["completion_ratio"] == pytest.approx(0.8, abs=1.0e-4)
    assert late.info["completion_ratio_bonus"] > early.info["completion_ratio_bonus"]
    assert late.reward > early.reward


def test_terminal_success_bonus_applies_on_env_done() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 0.0,
            "w_lateral": 0.0,
            "w_yaw": 0.0,
        },
        "collision": {
            "w_static": 0.0,
            "w_dynamic": 0.0,
        },
        "comfort": {
            "w_longitudinal_jerk": 0.0,
            "w_yaw_jerk": 0.0,
        },
        "terminal": {
            "success_bonus": 7.5,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)

    result = computer.apply_terminal_penalty(
        reward=1.25,
        info={"done_reason": "env_done"},
        term_cfg=reward_cfg["terminal"],
        terminal_kind="env_done",
    )

    assert result.reward == pytest.approx(8.75)
    assert result.info["terminal_success_bonus"] == pytest.approx(7.5)
    assert result.info["terminal_success_bonus_applied"] is True

