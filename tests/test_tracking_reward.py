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


def test_collision_defaults_to_constraint_gate_not_dense_penalty() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 1.0,
            "w_lateral": 0.0,
            "w_yaw": 0.0,
        },
        "collision": {
            "w_static": 5.0,
            "w_dynamic": 7.0,
        },
        "comfort": {
            "w_longitudinal_jerk": 0.0,
            "w_yaw_jerk": 0.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(
        start_ego=_pose(0.0, 1.0),
        all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)],
    )

    result = computer.compute(
        env=env,
        info={"static_collision": True, "dynamic_collision": True},
        step_idx=0,
        done=False,
    )

    assert result.reward == pytest.approx(0.0)
    assert result.info["positive_reward"] == pytest.approx(0.0)
    assert result.info["cost_reward"] == pytest.approx(0.0)
    assert result.info["static_collision_penalty"] == pytest.approx(0.0)
    assert result.info["dynamic_collision_penalty"] == pytest.approx(0.0)
    assert result.info["safety_gate_active"] is True
    assert result.info["safety_gate_sources"] == ["collision_constraint"]


def test_collision_gate_only_masks_positive_reward_not_cost() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 0.0,
            "w_completion_ratio": 2.0,
            "completion_ratio_power": 1.0,
            "w_lateral": 2.0,
            "lateral_free_m": 0.0,
            "lateral_huber_delta_m": 1.0,
            "w_yaw": 0.0,
        },
        "collision": {
            "mode": "constraint_gate",
            "gate_scale": 0.0,
        },
        "comfort": {
            "w_longitudinal_jerk": 0.0,
            "w_yaw_jerk": 0.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(
        start_ego=_pose(1.0, 4.0),
        all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)],
    )

    result = computer.compute(
        env=env,
        info={"static_collision": True},
        step_idx=0,
        done=False,
    )

    assert result.info["positive_reward"] > 0.0
    assert result.info["cost_reward"] > 0.0
    assert result.reward == pytest.approx(-result.info["cost_reward"])
    assert result.info["safety_gate_active"] is True
    assert result.info["safety_gate_sources"] == ["collision_constraint"]


def test_severe_tracking_gate_masks_positive_reward_before_terminal() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 0.0,
            "w_completion_ratio": 1.5,
            "completion_ratio_power": 1.0,
            "w_lateral": 0.25,
            "lateral_free_m": 0.0,
            "lateral_huber_delta_m": 1.0,
            "w_yaw": 0.0,
            "severe_lateral_error_m": 1.0,
            "severe_gate_scale": 0.0,
        },
        "collision": {
            "mode": "constraint_gate",
        },
        "comfort": {
            "w_longitudinal_jerk": 0.0,
            "w_yaw_jerk": 0.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(
        start_ego=_pose(2.0, 4.0),
        all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)],
    )

    result = computer.compute(env=env, info={}, step_idx=0, done=False)

    assert result.info["positive_reward"] > 0.0
    assert result.info["cost_reward"] > 0.0
    assert result.reward == pytest.approx(-result.info["cost_reward"])
    assert result.info["safety_gate_active"] is True
    assert result.info["safety_gate_sources"] == ["severe_tracking_lateral"]


def test_collision_gate_does_not_improve_negative_dense_reward() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 0.0,
            "w_lateral": 2.0,
            "lateral_free_m": 0.0,
            "lateral_huber_delta_m": 1.0,
            "w_yaw": 0.0,
        },
        "collision": {
            "mode": "constraint_gate",
            "gate_scale": 0.0,
        },
        "comfort": {
            "w_longitudinal_jerk": 0.0,
            "w_yaw_jerk": 0.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(
        start_ego=_pose(2.0, 1.0),
        all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)],
    )

    no_collision = computer.compute(env=env, info={}, step_idx=0, done=False)
    computer.reset()
    collision = computer.compute(
        env=env,
        info={"static_collision": True},
        step_idx=0,
        done=False,
    )

    assert no_collision.reward < 0.0
    assert collision.reward == pytest.approx(no_collision.reward)
    assert collision.info["safety_gate_active"] is True
    assert collision.info["safety_gate_scale"] == pytest.approx(0.0)


def test_collision_penalty_can_still_be_enabled_for_backward_compatibility() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 1.0,
            "w_lateral": 0.0,
            "w_yaw": 0.0,
        },
        "collision": {
            "mode": "dense_penalty",
            "w_static": 5.0,
            "w_dynamic": 7.0,
        },
        "comfort": {
            "w_longitudinal_jerk": 0.0,
            "w_yaw_jerk": 0.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(
        start_ego=_pose(0.0, 1.0),
        all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)],
    )

    result = computer.compute(
        env=env,
        info={"static_collision": True, "dynamic_collision": True},
        step_idx=0,
        done=False,
    )

    assert result.reward == pytest.approx(-12.0)
    assert result.info["static_collision_penalty"] == pytest.approx(5.0)
    assert result.info["dynamic_collision_penalty"] == pytest.approx(7.0)
    assert result.info["safety_gate_active"] is False
