from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from framework.rewards.tracking import TrackingRewardComputer, TrackingRewardResult, select_reward_mode_cfg


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
    def __init__(
        self,
        *,
        start_ego: np.ndarray,
        all_expert_ego: list[np.ndarray],
        expert_pair: list[np.ndarray] | None = None,
    ) -> None:
        self.start_ego = np.asarray(start_ego, dtype=np.float64)
        self.all_expert_ego = [np.asarray(item, dtype=np.float64) for item in all_expert_ego]
        if expert_pair is not None:
            self.expert_pair = [np.asarray(item, dtype=np.float64) for item in expert_pair]


def _zero_reward_cfg() -> dict[str, object]:
    return {
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
    }


def test_nested_reward_mode_config_selects_active_craft_close_loop() -> None:
    reward_cfg = {
        "mode": "craft_close_loop",
        "craft_close_loop": {
            "dt": 0.5,
            "CRAFT": {
                "enable": True,
                "real_reward_model": "close loop",
                "progress_weight": 0.0,
                "progress_max_m": 1.2,
                "progress_min_m": 0.0,
                "cost_off_road": 0.0,
                "cost_opposite_lane": 0.0,
                "cost_off_global_route": 0.0,
                "cost_emergency_lane": 0.0,
                "cost_red_light": 0.0,
                "cost_stop_sign": 0.0,
                "term_collision": 0.0,
                "term_route_dev": 0.0,
            },
        },
        "step_path": {
            "dt": 0.5,
            "path": {
                "w_progress": 0.0,
                "w_lateral": 0.0,
                "w_yaw": 0.0,
            },
        },
    }

    selected = select_reward_mode_cfg(reward_cfg)

    assert selected["CRAFT"]["real_reward_model"] == "close loop"

    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(
        start_ego=_pose(0.0, 1.0),
        all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)],
    )

    result = computer.compute(env=env, info={}, step_idx=0, done=False)

    assert result.info["reward_mode"] == "craft_closed_loop"


def test_compute_dispatches_nested_reward_modes_as_parallel_branches() -> None:
    class DispatchProbeRewardComputer(TrackingRewardComputer):
        def _compute_step_path_reward(self, **_: object) -> TrackingRewardResult:  # type: ignore[override]
            return TrackingRewardResult(reward=1.0, info={"reward_mode": "step_path_probe"})

        def _compute_craft_corrective_reward(self, **_: object) -> TrackingRewardResult:  # type: ignore[override]
            return TrackingRewardResult(reward=2.0, info={"reward_mode": "craft_corrective_probe"})

        def _compute_craft_closed_loop_reward(self, **_: object) -> TrackingRewardResult:  # type: ignore[override]
            return TrackingRewardResult(reward=3.0, info={"reward_mode": "craft_closed_loop_probe"})

    base_cfg = {
        "craft_close_loop": {
            "dt": 0.5,
            "CRAFT": {
                "enable": True,
                "real_reward_model": "close loop",
            },
        },
        "craft_sparse_loop": {
            "dt": 0.5,
            "CRAFT": {
                "enable": True,
                "real_reward_model": "corrective",
            },
        },
        "step_path": {
            "dt": 0.5,
            "CRAFT": {
                "enable": False,
            },
        },
    }
    env = _DummyEnv(
        start_ego=_pose(0.0, 1.0),
        all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)],
    )

    expected_by_mode = {
        "craft_close_loop": "craft_closed_loop_probe",
        "craft_sparse_loop": "craft_corrective_probe",
        "step_path": "step_path_probe",
    }
    for mode, expected_reward_mode in expected_by_mode.items():
        computer = DispatchProbeRewardComputer({"mode": mode, **base_cfg})

        result = computer.compute(env=env, info={}, step_idx=0, done=False)

        assert result.info["reward_mode"] == expected_reward_mode


def test_craft_corrective_reward_info_omits_unused_dev_ratio_fields() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "corrective",
            "heading_max_deg": 60.0,
            "corrective_progress": {
                "enable": True,
                "weight": 1.0,
                "max_m": 1.0,
                "min_m": 0.0,
                "lateral_safe_m": 0.0,
                "lateral_max_m": 2.0,
                "w_lateral_efficiency": 0.0,
                "w_heading_efficiency": 0.0,
            },
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]

    computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 0.0), all_expert_ego=reference_path),
        info={},
        step_idx=0,
        done=False,
    )
    result = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=reference_path),
        info={},
        step_idx=1,
        done=False,
    )

    assert result.info["reward_mode"] == "craft_corrective"
    assert result.reward == pytest.approx(1.0)
    assert result.info["craft_corrective_progress_lateral_ratio"] == pytest.approx(0.0)
    assert "craft_global_dev_ratio" not in result.info
    assert "craft_center_dev_ratio" not in result.info
    assert "craft_lateral_dev_ratio" not in result.info
    for legacy_key in [
        "positive_reward",
        "gated_positive_reward",
        "craft_progress_reward",
        "craft_effective_progress",
        "craft_efficiency",
        "craft_correction_reward",
        "craft_safety_cost",
        "pos_dev",
        "pos_dev_source",
        "yaw_err_deg",
    ]:
        assert legacy_key not in result.info


def test_reference_path_prefers_all_expert_ego_over_legacy_expert_pair() -> None:
    computer = TrackingRewardComputer(_zero_reward_cfg())
    env = _DummyEnv(
        start_ego=_pose(0.0, 2.0),
        all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 10.0)],
        expert_pair=[np.asarray([10.0, 0.0]), np.asarray([10.0, 10.0])],
    )

    result = computer.compute(env=env, info={}, step_idx=0, done=False)

    assert result.info["reference_path_source"] == "all_expert_ego"
    assert result.info["reference_path_source_legacy"] is False
    assert result.info["lateral_error_m"] == pytest.approx(0.0)


def test_reference_path_uses_expert_pair_only_as_legacy_fallback() -> None:
    computer = TrackingRewardComputer(_zero_reward_cfg())
    env = _DummyEnv(
        start_ego=_pose(10.0, 2.0),
        all_expert_ego=[],
        expert_pair=[np.asarray([10.0, 0.0]), np.asarray([10.0, 10.0])],
    )

    result = computer.compute(env=env, info={}, step_idx=0, done=False)

    assert result.info["reference_path_source"] == "expert_pair"
    assert result.info["reference_path_source_legacy"] is True
    assert result.info["lateral_error_m"] == pytest.approx(0.0)


def test_completion_ratio_is_diagnostic_not_reward_term() -> None:
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
    assert "completion_ratio_bonus" not in early.info
    assert "completion_ratio_term" not in early.info
    assert late.reward == pytest.approx(early.reward)


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


def test_step_path_reward_info_does_not_expose_unused_anchor_terms() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 0.0,
            "w_lateral": 0.0,
            "w_yaw": 0.0,
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
        start_ego=_pose(0.0, 1.0),
        all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)],
    )

    result = computer.compute(env=env, info={}, step_idx=0, done=False)

    assert "anchor_progress_term" not in result.info
    assert "anchor_lateral_term" not in result.info


def test_collision_gate_only_masks_positive_reward_not_cost() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 1.0,
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
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]
    computer.compute(
        env=_DummyEnv(start_ego=_pose(1.0, 2.0), all_expert_ego=reference_path),
        info={},
        step_idx=0,
        done=False,
    )

    result = computer.compute(
        env=_DummyEnv(start_ego=_pose(1.0, 4.0), all_expert_ego=reference_path),
        info={"static_collision": True},
        step_idx=1,
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
            "progress_forward_cap_m": 2.0,
            "w_lateral": 0.25,
            "lateral_free_m": 0.0,
            "lateral_huber_delta_m": 1.0,
            "w_yaw": 0.0,
            "severe_lateral_error_m": 1.0,
            "severe_lateral_cost": 3.0,
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

    assert result.info["positive_reward"] == pytest.approx(0.0)
    assert result.info["cost_reward"] > 0.0
    assert result.info["severe_lateral_cost"] == pytest.approx(3.0)
    assert result.reward == pytest.approx(-result.info["cost_reward"])
    assert result.info["safety_gate_active"] is True
    assert result.info["safety_gate_sources"] == ["severe_tracking_lateral"]


def test_severe_yaw_adds_dense_cost_before_terminal() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 0.0,
            "w_lateral": 0.0,
            "w_yaw": 0.0,
            "severe_yaw_error_deg": 30.0,
            "severe_yaw_cost": 2.0,
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
        start_ego=_pose(0.0, 1.0, yaw_rad=math.radians(45.0)),
        all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)],
    )

    result = computer.compute(env=env, info={}, step_idx=0, done=False)

    assert result.info["severe_yaw_cost"] == pytest.approx(2.0)
    assert result.info["cost_reward"] == pytest.approx(2.0)
    assert result.reward == pytest.approx(-2.0)
    assert result.info["safety_gate_active"] is True
    assert result.info["safety_gate_sources"] == ["severe_tracking_yaw"]


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


def test_step_path_reward_penalizes_front_obstacle_risk_and_gates_progress() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 1.0,
            "w_lateral": 0.0,
            "w_yaw": 0.0,
        },
        "collision": {
            "mode": "constraint_gate",
        },
        "safety": {
            "enable": True,
            "lookahead_m": 20.0,
            "corridor_half_width_m": 2.5,
            "safe_gap_m": 8.0,
            "safe_ttc_s": 3.0,
            "w_clearance": 2.0,
            "w_ttc": 3.0,
            "progress_gate_strength": 1.0,
            "min_progress_gate": 0.0,
        },
        "comfort": {
            "w_longitudinal_jerk": 0.0,
            "w_yaw_jerk": 0.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 10.0)]
    computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 0.0), all_expert_ego=reference_path),
        info={},
        step_idx=0,
        done=False,
    )

    result = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=reference_path),
        info={
            "front_obstacle_gap_m": 4.0,
            "front_obstacle_lateral_m": 0.0,
            "front_obstacle_closing_speed_mps": 4.0,
        },
        step_idx=1,
        done=False,
    )

    assert result.info["front_obstacle_active"] is True
    assert result.info["front_obstacle_clearance_risk"] == pytest.approx(0.5)
    assert result.info["front_obstacle_ttc_s"] == pytest.approx(1.0)
    assert result.info["front_obstacle_ttc_risk"] == pytest.approx(2.0 / 3.0)
    assert result.info["front_obstacle_cost"] == pytest.approx(2.0 * 0.5**2 + 3.0 * (2.0 / 3.0) ** 2)
    assert result.info["safe_progress_gate"] == pytest.approx(1.0 / 3.0)
    assert result.info["positive_reward"] == pytest.approx(1.0)
    assert result.info["gated_positive_reward"] == pytest.approx(1.0 / 3.0)
    assert result.info["cost_reward"] == pytest.approx(result.info["front_obstacle_cost"])
    assert result.reward == pytest.approx((1.0 / 3.0) - result.info["front_obstacle_cost"])


def test_step_path_reward_ignores_legacy_front_obstacle_aliases() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 1.0,
            "w_lateral": 0.0,
            "w_yaw": 0.0,
        },
        "safety": {
            "enable": True,
            "lookahead_m": 20.0,
            "corridor_half_width_m": 2.5,
            "safe_gap_m": 8.0,
            "safe_ttc_s": 3.0,
            "w_clearance": 2.0,
            "w_ttc": 3.0,
        },
        "comfort": {
            "w_longitudinal_jerk": 0.0,
            "w_yaw_jerk": 0.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 10.0)]
    computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 0.0), all_expert_ego=reference_path),
        info={},
        step_idx=0,
        done=False,
    )

    result = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=reference_path),
        info={
            "front_gap_m": 4.0,
            "front_lateral_m": 0.0,
            "front_closing_speed_mps": 4.0,
            "front_ttc_s": 1.0,
        },
        step_idx=1,
        done=False,
    )

    assert result.info["front_obstacle_active"] is False
    assert math.isinf(result.info["front_obstacle_gap_m"])
    assert math.isinf(result.info["front_obstacle_lateral_m"])
    assert result.info["front_obstacle_closing_speed_mps"] == pytest.approx(0.0)
    assert math.isinf(result.info["front_obstacle_ttc_s"])
    assert result.info["front_obstacle_cost"] == pytest.approx(0.0)
    assert result.info["safe_progress_gate"] == pytest.approx(1.0)
    assert result.reward == pytest.approx(1.0)


def test_step_path_reward_ignores_front_obstacle_outside_corridor() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 1.0,
            "w_lateral": 0.0,
            "w_yaw": 0.0,
        },
        "safety": {
                    "enable": True,
            "lookahead_m": 20.0,
            "corridor_half_width_m": 2.5,
            "safe_gap_m": 8.0,
            "safe_ttc_s": 3.0,
            "w_clearance": 2.0,
            "w_ttc": 3.0,
        },
        "comfort": {
            "w_longitudinal_jerk": 0.0,
            "w_yaw_jerk": 0.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 10.0)]
    computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 0.0), all_expert_ego=reference_path),
        info={},
        step_idx=0,
        done=False,
    )

    result = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=reference_path),
        info={
            "front_obstacle_gap_m": 4.0,
            "front_obstacle_lateral_m": 5.0,
            "front_obstacle_closing_speed_mps": 4.0,
        },
        step_idx=1,
        done=False,
    )

    assert result.info["front_obstacle_active"] is False
    assert result.info["front_obstacle_cost"] == pytest.approx(0.0)
    assert result.info["safe_progress_gate"] == pytest.approx(1.0)
    assert result.reward == pytest.approx(1.0)


def test_craft_reward_discounts_progress_when_laterally_deviated() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 5.0,
            "progress_max_m": 1.2,
            "lateral_safe_m": 0.0,
            "lateral_max_m": 3.0,
            "heading_max_deg": 60.0,
            "w_lateral_efficiency": 3.0,
            "w_heading_efficiency": 2.0,
            "efficiency_floor": 0.0,
            "correction_lateral_weight": 0.0,
            "correction_heading_weight": 0.0,
            "collision_cost_static": 0.0,
            "collision_cost_dynamic": 0.0,
        },
    }
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]
    aligned = _DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=reference_path)
    deviated = _DummyEnv(start_ego=_pose(2.0, 1.0), all_expert_ego=reference_path)

    computer = TrackingRewardComputer(reward_cfg)
    computer.compute(env=_DummyEnv(start_ego=_pose(0.0, 0.0), all_expert_ego=reference_path), info={}, step_idx=0, done=False)
    aligned_result = computer.compute(env=aligned, info={}, step_idx=1, done=False)

    computer.reset()
    computer.compute(env=_DummyEnv(start_ego=_pose(0.0, 0.0), all_expert_ego=reference_path), info={}, step_idx=0, done=False)
    deviated_result = computer.compute(env=deviated, info={}, step_idx=1, done=False)

    assert aligned_result.info["reward_mode"] == "craft_closed_loop"
    assert aligned_result.info["craft_efficiency"] > deviated_result.info["craft_efficiency"]
    assert aligned_result.reward > deviated_result.reward


def test_craft_closed_loop_uses_effective_progress_without_raw_progress_bonus() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 5.0,
            "progress_max_m": 1.0,
            "progress_min_m": 0.0,
            "w_g": 0.0,
            "w_c": 0.0,
            "w_h": 0.0,
            "efficiency_floor": 1.0,
            "k_g": 0.0,
            "k_c": 0.0,
            "k_h": 0.0,
            "collision_cost_static": 0.0,
            "collision_cost_dynamic": 0.0,
        },
    }
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]
    computer = TrackingRewardComputer(reward_cfg)

    computer.compute(env=_DummyEnv(start_ego=_pose(0.0, 0.0), all_expert_ego=reference_path), info={}, step_idx=0, done=False)
    result = computer.compute(env=_DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=reference_path), info={}, step_idx=1, done=False)

    assert result.info["craft_progress_reward"] == pytest.approx(5.0)
    assert result.info["craft_effective_progress"] == pytest.approx(5.0)
    assert result.reward == pytest.approx(5.0)


def test_craft_reward_adds_correction_when_deviation_improves() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 0.0,
            "lateral_safe_m": 0.0,
            "lateral_max_m": 3.0,
            "heading_max_deg": 60.0,
            "w_lateral_efficiency": 0.0,
            "w_heading_efficiency": 0.0,
            "efficiency_floor": 1.0,
            "correction_lateral_weight": 0.6,
            "correction_heading_weight": 0.0,
            "correction_clip": 0.5,
            "collision_cost_static": 0.0,
            "collision_cost_dynamic": 0.0,
        },
    }
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]
    computer = TrackingRewardComputer(reward_cfg)

    first = computer.compute(env=_DummyEnv(start_ego=_pose(2.0, 1.0), all_expert_ego=reference_path), info={}, step_idx=0, done=False)
    second = computer.compute(env=_DummyEnv(start_ego=_pose(1.0, 2.0), all_expert_ego=reference_path), info={}, step_idx=1, done=False)

    assert first.info["craft_correction_reward"] == pytest.approx(0.0)
    assert second.info["craft_correction_reward"] > 0.0
    assert second.reward > first.reward


def test_craft_reward_uses_navsim_collision_flags_as_safety_cost() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 0.0,
            "lateral_max_m": 3.0,
            "heading_max_deg": 60.0,
            "efficiency_floor": 1.0,
            "correction_lateral_weight": 0.0,
            "correction_heading_weight": 0.0,
            "collision_cost_static": 11.0,
            "collision_cost_dynamic": 13.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)])

    result = computer.compute(
        env=env,
        info={"static_collision": True, "dynamic_collision": True},
        step_idx=0,
        done=False,
    )

    assert result.info["reward_mode"] == "craft_closed_loop"
    assert result.info["craft_safety_cost"] == pytest.approx(24.0)
    assert result.reward == pytest.approx(-24.0)


def test_craft_reward_applies_term_collision_once_without_per_type_overrides() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 0.0,
            "w_g": 0.0,
            "w_c": 0.0,
            "w_h": 0.0,
            "efficiency_floor": 1.0,
            "k_g": 0.0,
            "k_c": 0.0,
            "k_h": 0.0,
            "term_collision": 30.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)])

    result = computer.compute(
        env=env,
        info={"static_collision": True, "dynamic_collision": True},
        step_idx=0,
        done=False,
    )

    assert result.info["craft_collision_terminal_cost"] == pytest.approx(30.0)
    assert result.info["craft_safety_cost"] == pytest.approx(30.0)
    assert result.reward == pytest.approx(-30.0)


def test_craft_closed_loop_treats_null_collision_overrides_as_absent() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 0.0,
            "w_g": 0.0,
            "w_c": 0.0,
            "w_h": 0.0,
            "efficiency_floor": 1.0,
            "k_g": 0.0,
            "k_c": 0.0,
            "k_h": 0.0,
            "term_collision": 30.0,
            "collision_cost_static": None,
            "collision_cost_dynamic": None,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)])

    result = computer.compute(
        env=env,
        info={"static_collision": True, "dynamic_collision": True},
        step_idx=0,
        done=False,
    )

    assert result.info["craft_collision_terminal_cost"] == pytest.approx(30.0)
    assert result.info["craft_static_collision_cost"] == pytest.approx(0.0)
    assert result.info["craft_dynamic_collision_cost"] == pytest.approx(0.0)
    assert result.reward == pytest.approx(-30.0)


def test_craft_reward_caps_correction_with_ddev_clip() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 0.0,
            "lateral_safe_m": 0.0,
            "lateral_max_m": 3.0,
            "heading_max_deg": 60.0,
            "w_g": 0.0,
            "w_c": 0.0,
            "w_h": 0.0,
            "efficiency_floor": 1.0,
            "ddev_clip": 0.1,
            "k_g": 1.0,
            "k_c": 0.0,
            "k_h": 0.0,
            "correction_apply_thresh_global": 0.0,
            "correction_clip": 1.0,
            "collision_cost_static": 0.0,
            "collision_cost_dynamic": 0.0,
        },
    }
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]
    computer = TrackingRewardComputer(reward_cfg)

    computer.compute(env=_DummyEnv(start_ego=_pose(3.0, 1.0), all_expert_ego=reference_path), info={}, step_idx=0, done=False)
    second = computer.compute(env=_DummyEnv(start_ego=_pose(0.3, 2.0), all_expert_ego=reference_path), info={}, step_idx=1, done=False)

    assert second.info["craft_correction_reward"] == pytest.approx(0.1 * (0.5 + 0.5 * (1.0 / 1.2)))


def test_craft_reward_skips_correction_below_apply_threshold() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 0.0,
            "lateral_safe_m": 0.0,
            "lateral_max_m": 3.0,
            "heading_max_deg": 60.0,
            "w_g": 0.0,
            "w_c": 0.0,
            "w_h": 0.0,
            "efficiency_floor": 1.0,
            "ddev_clip": 0.5,
            "k_g": 1.0,
            "k_c": 0.0,
            "k_h": 0.0,
            "correction_apply_thresh_global": 0.7,
            "correction_clip": 1.0,
            "collision_cost_static": 0.0,
            "collision_cost_dynamic": 0.0,
        },
    }
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]
    computer = TrackingRewardComputer(reward_cfg)

    computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=reference_path),
        info={"global_dev_ratio": 0.5},
        step_idx=0,
        done=False,
    )
    second = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 2.0), all_expert_ego=reference_path),
        info={"global_dev_ratio": 0.1},
        step_idx=1,
        done=False,
    )

    assert second.info["craft_global_dev_ratio"] < 0.7
    assert second.info["craft_correction_reward"] == pytest.approx(0.0)


def test_craft_reward_skips_correction_at_equal_threshold_like_carl() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 0.0,
            "lateral_safe_m": 0.0,
            "lateral_max_m": 3.0,
            "heading_max_deg": 60.0,
            "w_g": 0.0,
            "w_c": 0.0,
            "w_h": 0.0,
            "efficiency_floor": 1.0,
            "ddev_clip": 0.5,
            "k_g": 1.0,
            "k_c": 0.0,
            "k_h": 0.0,
            "correction_apply_thresh_global": 0.1,
            "correction_clip": 1.0,
            "collision_cost_static": 0.0,
            "collision_cost_dynamic": 0.0,
        },
    }
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]
    computer = TrackingRewardComputer(reward_cfg)

    computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=reference_path),
        info={"global_dev_ratio": 0.5},
        step_idx=0,
        done=False,
    )
    second = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 2.0), all_expert_ego=reference_path),
        info={"global_dev_ratio": 0.1},
        step_idx=1,
        done=False,
    )

    assert second.info["craft_global_dev_ratio"] == pytest.approx(0.1)
    assert second.info["craft_correction_reward"] == pytest.approx(0.0)


def test_craft_reward_uses_map_and_rule_flags_as_safety_cost() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 0.0,
            "w_g": 0.0,
            "w_c": 0.0,
            "w_h": 0.0,
            "efficiency_floor": 1.0,
            "k_g": 0.0,
            "k_c": 0.0,
            "k_h": 0.0,
            "cost_off_road": 5.0,
            "cost_opposite_lane": 1.0,
            "cost_red_light": 6.0,
            "cost_stop_sign": 7.0,
            "cost_emergency_lane": 3.0,
            "collision_cost_static": 0.0,
            "collision_cost_dynamic": 0.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)])

    result = computer.compute(
        env=env,
        info={
            "off_road": True,
            "opposite_lane": True,
            "red_light_violation": True,
            "stop_sign_violation": True,
            "emergency_lane": True,
        },
        step_idx=0,
        done=False,
    )

    assert result.info["craft_off_road_cost"] == pytest.approx(5.0)
    assert result.info["craft_opposite_lane_cost"] == pytest.approx(1.0)
    assert result.info["craft_red_light_cost"] == pytest.approx(6.0)
    assert result.info["craft_stop_sign_cost"] == pytest.approx(7.0)
    assert result.info["craft_emergency_lane_cost"] == pytest.approx(3.0)
    assert result.info["craft_safety_cost"] == pytest.approx(22.0)
    assert result.reward == pytest.approx(-22.0)


def test_craft_closed_loop_computes_off_global_route_from_lateral_threshold() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 0.0,
            "w_g": 0.0,
            "w_c": 0.0,
            "w_h": 0.0,
            "efficiency_floor": 1.0,
            "k_g": 0.0,
            "k_c": 0.0,
            "k_h": 0.0,
            "cost_off_global_route": 4.0,
            "collision_cost_static": 0.0,
            "collision_cost_dynamic": 0.0,
            "map": {
                "off_global_route_threshold_m": 1.5,
            },
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(start_ego=_pose(2.0, 1.0), all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)])

    result = computer.compute(env=env, info={}, step_idx=0, done=False)

    assert result.info["off_global_route"] is True
    assert result.info["off_global_route_source"] == "route_lateral_threshold"
    assert result.info["off_global_route_threshold_m"] == pytest.approx(1.5)
    assert result.info["craft_off_global_route_cost"] == pytest.approx(4.0)
    assert result.reward == pytest.approx(-4.0)


def test_craft_reward_applies_route_completion_and_deviation_terms() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "dense_carl",
            "progress_weight": 0.0,
            "w_g": 0.0,
            "w_c": 0.0,
            "w_h": 0.0,
            "efficiency_floor": 1.0,
            "k_g": 0.0,
            "k_c": 0.0,
            "k_h": 0.0,
            "reward_completed": 20.0,
            "term_route_dev": 30.0,
            "collision_cost_static": 0.0,
            "collision_cost_dynamic": 0.0,
        },
    }
    env = _DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)])
    computer = TrackingRewardComputer(reward_cfg)

    completed = computer.compute(env=env, info={"route_completed": True}, step_idx=0, done=True)
    deviated = computer.compute(env=env, info={"route_deviation": True}, step_idx=1, done=True)

    assert completed.info["craft_route_completed_reward"] == pytest.approx(20.0)
    assert completed.reward == pytest.approx(20.0)
    assert deviated.info["craft_route_deviation_cost"] == pytest.approx(30.0)
    assert deviated.reward == pytest.approx(-30.0)


def test_craft_default_sparse_corrective_reward_ignores_dense_progress_and_uses_safety_costs() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "corrective",
            "progress_weight": 5.0,
            "w_g": 10.0,
            "w_h": 10.0,
            "k_g": 1.0,
            "corrective": {
                "cost_off_road": 0.5,
                "cost_emergency_lane": 0.2,
                "cost_off_global_route": 0.5,
                "cost_red_light": 2.0,
                "cost_stop_sign": 2.0,
                "cost_collision": 5.0,
            },
        },
    }
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]
    computer = TrackingRewardComputer(reward_cfg)
    computer.compute(env=_DummyEnv(start_ego=_pose(0.0, 0.0), all_expert_ego=reference_path), info={}, step_idx=0, done=False)

    result = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=reference_path),
        info={
            "off_road": True,
            "red_light_violation": True,
            "stop_sign_violation": True,
            "static_collision": True,
        },
        step_idx=1,
        done=True,
    )

    assert result.info["reward_mode"] == "craft_corrective"
    assert result.info["craft_corrective_cost_off_road"] == pytest.approx(0.5)
    assert result.info["craft_corrective_cost_red_light"] == pytest.approx(2.0)
    assert result.info["craft_corrective_cost_stop_sign"] == pytest.approx(2.0)
    assert result.info["craft_corrective_cost_collision"] == pytest.approx(5.0)
    assert result.info["cost_reward"] == pytest.approx(9.5)
    assert result.reward == pytest.approx(-9.5)


def test_craft_corrective_uses_map_heading_ratio_like_closed_loop() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "corrective",
            "heading_max_deg": 60.0,
            "corrective_progress": {
                "enable": True,
                "weight": 1.2,
                "max_m": 1.2,
                "min_m": 0.0,
                "w_lateral_efficiency": 0.0,
                "w_heading_efficiency": 1.0,
            },
            "corrective": {
                "cost_off_road": 0.0,
                "cost_emergency_lane": 0.0,
                "cost_off_global_route": 0.0,
                "cost_red_light": 0.0,
                "cost_stop_sign": 0.0,
                "cost_collision": 0.0,
            },
        },
    }
    reference_path = [_pose(0.0, 0.0, math.pi / 2.0), _pose(0.0, 5.0, math.pi / 2.0)]
    computer = TrackingRewardComputer(reward_cfg)
    computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 0.0, math.pi / 2.0), all_expert_ego=reference_path),
        info={},
        step_idx=0,
        done=False,
    )

    result = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 1.0, math.pi / 2.0), all_expert_ego=reference_path),
        info={"map_heading_dev_ratio": 0.5},
        step_idx=1,
        done=False,
    )

    assert result.info["heading_dev_ratio"] == pytest.approx(0.5)
    assert result.info["craft_corrective_progress_heading_ratio"] == pytest.approx(0.5)
    assert "craft_heading_dev_ratio" not in result.info


def test_craft_corrective_computes_off_global_route_from_lateral_threshold() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "corrective",
            "map": {
                "off_global_route_threshold_m": 1.5,
            },
            "corrective": {
                "cost_off_road": 0.0,
                "cost_emergency_lane": 0.0,
                "cost_off_global_route": 0.5,
                "cost_red_light": 0.0,
                "cost_stop_sign": 0.0,
                "cost_collision": 0.0,
            },
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    env = _DummyEnv(start_ego=_pose(2.0, 1.0), all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)])

    result = computer.compute(env=env, info={}, step_idx=0, done=False)

    assert result.info["reward_mode"] == "craft_corrective"
    assert result.info["off_global_route"] is True
    assert result.info["off_global_route_source"] == "route_lateral_threshold"
    assert result.info["craft_corrective_cost_off_global_route"] == pytest.approx(0.5)
    assert result.reward == pytest.approx(-0.5)


def test_craft_corrective_uses_defaults_and_ignores_opposite_lane() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "corrective",
        },
    }
    env = _DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)])
    computer = TrackingRewardComputer(reward_cfg)

    result = computer.compute(
        env=env,
        info={"opposite_lane": True, "dynamic_collision": True},
        step_idx=0,
        done=False,
    )

    assert result.info["reward_mode"] == "craft_corrective"
    assert "craft_opposite_lane_cost" not in result.info
    assert result.info["craft_corrective_cost_collision"] == pytest.approx(5.0)
    assert result.reward == pytest.approx(-5.0)


def test_craft_training_config_uses_corrective_collision_cost() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "script/configs/sparsedrive_v2/reinforcepp_closed_loop_sparsedrive_v2_craft.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text())
    craft_cfg = cfg["env"]["reward"]["CRAFT"]

    assert craft_cfg["real_reward_model"] == "corrective"
    assert float(craft_cfg["corrective"]["cost_collision"]) == pytest.approx(5.0)
    assert "term_collision" not in craft_cfg
    assert "collision_cost_static" not in craft_cfg
    assert "collision_cost_dynamic" not in craft_cfg
    assert "closed_loop_reward_mode" not in craft_cfg
    assert "cost_opposite_lane" not in craft_cfg
    assert "w_h" not in craft_cfg
    assert "w_heading_efficiency" not in craft_cfg


def test_step_path_reward_subtracts_closed_loop_ea_cost_and_logs_range() -> None:
    reward_cfg = {
        "dt": 0.5,
        "path": {
            "w_progress": 1.0,
            "w_lateral": 0.0,
            "w_yaw": 0.0,
        },
        "collision": {
            "mode": "constraint_gate",
        },
        "comfort": {
            "w_longitudinal_jerk": 0.0,
            "w_yaw_jerk": 0.0,
        },
        "ea": {
            "enable": True,
            "weight": 2.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]

    computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 0.0), all_expert_ego=reference_path),
        info={"ea_available": True, "ea_max": 4.0, "ea_risk": 0.5, "ea_evaluated_pairs": 1.0},
        step_idx=0,
        done=False,
    )
    result = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=reference_path),
        info={"ea_available": True, "ea_max": 4.0, "ea_risk": 0.5, "ea_evaluated_pairs": 1.0},
        step_idx=1,
        done=False,
    )

    assert result.info["ea_enabled"] is True
    assert result.info["ea_available"] is True
    assert result.info["ea_max"] == pytest.approx(4.0)
    assert result.info["ea_risk"] == pytest.approx(0.5)
    assert result.info["ea_cost"] == pytest.approx(1.0)
    assert result.info["cost_reward"] == pytest.approx(1.0)
    assert result.reward == pytest.approx(0.0)


def test_craft_corrective_reward_subtracts_closed_loop_ea_cost() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "corrective",
        },
        "ea": {
            "enable": True,
            "weight": 2.0,
        },
    }
    computer = TrackingRewardComputer(reward_cfg)

    result = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 1.0), all_expert_ego=[_pose(0.0, 0.0), _pose(0.0, 5.0)]),
        info={"ea_available": True, "ea_max": 4.0, "ea_risk": 0.5, "ea_evaluated_pairs": 1.0},
        step_idx=0,
        done=False,
    )

    assert result.info["reward_mode"] == "craft_corrective"
    assert result.info["ea_cost"] == pytest.approx(1.0)
    assert result.info["cost_reward"] == pytest.approx(1.0)
    assert result.reward == pytest.approx(-1.0)


def test_craft_corrective_progress_bonus_rewards_forward_motion() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "corrective",
            "corrective_progress": {
                "enable": True,
                "weight": 2.0,
                "max_m": 1.0,
                "lateral_safe_m": 0.0,
                "lateral_max_m": 2.0,
                "heading_max_deg": 90.0,
                "w_lateral_efficiency": 0.0,
                "w_heading_efficiency": 0.0,
                "efficiency_floor": 0.0,
            },
            "corrective": {
                "cost_off_road": 0.0,
                "cost_emergency_lane": 0.0,
                "cost_off_global_route": 0.0,
                "cost_red_light": 0.0,
                "cost_stop_sign": 0.0,
                "cost_collision": 0.0,
            },
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]
    computer.compute(env=_DummyEnv(start_ego=_pose(0.0, 0.0), all_expert_ego=reference_path), info={}, step_idx=0, done=False)

    result = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.0, 0.8), all_expert_ego=reference_path),
        info={},
        step_idx=1,
        done=False,
    )

    assert result.info["reward_mode"] == "craft_corrective"
    assert result.info["craft_corrective_progress_enabled"] is True
    assert result.info["progress_reward"] == pytest.approx(0.8)
    assert result.info["craft_corrective_progress_reward"] == pytest.approx(1.6)
    assert result.info["cost_reward"] == pytest.approx(0.0)
    assert result.reward == pytest.approx(1.6)


def test_craft_corrective_progress_bonus_is_discounted_by_deviation() -> None:
    reward_cfg = {
        "dt": 0.5,
        "CRAFT": {
            "enable": True,
            "real_reward_model": "corrective",
            "corrective_progress": {
                "enable": True,
                "weight": 1.0,
                "max_m": 1.0,
                "lateral_safe_m": 0.0,
                "lateral_max_m": 1.0,
                "heading_max_deg": 90.0,
                "w_lateral_efficiency": 2.0,
                "w_heading_efficiency": 1.0,
                "efficiency_floor": 0.0,
            },
            "corrective": {
                "cost_off_road": 0.0,
                "cost_emergency_lane": 0.0,
                "cost_off_global_route": 0.0,
                "cost_red_light": 0.0,
                "cost_stop_sign": 0.0,
                "cost_collision": 0.0,
            },
        },
    }
    computer = TrackingRewardComputer(reward_cfg)
    reference_path = [_pose(0.0, 0.0), _pose(0.0, 5.0)]
    computer.compute(env=_DummyEnv(start_ego=_pose(0.5, 0.0, yaw_rad=math.radians(45.0)), all_expert_ego=reference_path), info={}, step_idx=0, done=False)

    result = computer.compute(
        env=_DummyEnv(start_ego=_pose(0.5, 1.0, yaw_rad=math.radians(45.0)), all_expert_ego=reference_path),
        info={},
        step_idx=1,
        done=False,
    )

    expected_efficiency = math.exp(-2.0 * 0.5) * math.exp(-1.0 * 0.75)
    assert result.info["craft_corrective_progress_lateral_ratio"] == pytest.approx(0.5)
    assert result.info["craft_corrective_progress_heading_ratio"] == pytest.approx(0.75)
    assert result.info["craft_corrective_progress_efficiency"] == pytest.approx(expected_efficiency)
    assert result.info["craft_corrective_progress_reward"] == pytest.approx(expected_efficiency)
    assert result.reward == pytest.approx(expected_efficiency)
