from __future__ import annotations

import math

import numpy as np
import pytest

from framework.algorithms.craft_reward import (
    CRAFT_CARL_FORWARD_SIM_DEFAULTS,
    CRAFT_CORRECTIVE_DEFAULTS,
    compute_carl_reward_numpy,
    compute_corrective_reward_scalar,
)


def test_corrective_reward_matches_craft_defaults() -> None:
    reward, info = compute_corrective_reward_scalar(
        params=CRAFT_CORRECTIVE_DEFAULTS,
        off_road=True,
        emergency_lane=True,
        off_global_route=True,
        run_red_light=1.0,
        run_stop_sign=1.0,
        collision=True,
    )

    assert reward == pytest.approx(-(0.5 + 0.2 + 0.5 + 2.0 + 2.0 + 5.0))
    assert info["craft_corrective_cost_off_road"] == pytest.approx(0.5)
    assert info["craft_corrective_cost_collision"] == pytest.approx(5.0)


def test_carl_reward_progress_with_zero_deviation_matches_w_prog() -> None:
    terms = compute_carl_reward_numpy(
        params=CRAFT_CARL_FORWARD_SIM_DEFAULTS,
        delta_progress=np.asarray([[1.2]], dtype=np.float32),
        global_dev_ratio=np.asarray([[0.0]], dtype=np.float32),
        center_dev_ratio=np.asarray([[0.0]], dtype=np.float32),
        heading_dev_ratio=np.asarray([[0.0]], dtype=np.float32),
        delta_global_dev_ratio=np.asarray([[0.0]], dtype=np.float32),
        delta_center_dev_ratio=np.asarray([[0.0]], dtype=np.float32),
        delta_heading_dev_ratio=np.asarray([[0.0]], dtype=np.float32),
        off_road=np.asarray([[0.0]], dtype=np.float32),
        opposite_lane=np.asarray([[0.0]], dtype=np.float32),
        emergency_lane=np.asarray([[0.0]], dtype=np.float32),
        off_global_route=np.asarray([[0.0]], dtype=np.float32),
        collision=np.asarray([[0.0]], dtype=np.float32),
        red_light_dev=np.asarray([False]),
        stop_sign_dev=np.asarray([False]),
    )

    assert terms.candidate_scores.shape == (1,)
    assert terms.candidate_scores[0] == pytest.approx(CRAFT_CARL_FORWARD_SIM_DEFAULTS["w_prog"])
    assert terms.terms["effective_progress"][0, 0] == pytest.approx(CRAFT_CARL_FORWARD_SIM_DEFAULTS["w_prog"])


def test_carl_reward_applies_exponential_efficiency() -> None:
    g = 0.2
    c = 0.3
    h = 0.4
    params = dict(CRAFT_CARL_FORWARD_SIM_DEFAULTS)
    terms = compute_carl_reward_numpy(
        params=params,
        delta_progress=np.asarray([[1.2]], dtype=np.float32),
        global_dev_ratio=np.asarray([[g]], dtype=np.float32),
        center_dev_ratio=np.asarray([[c]], dtype=np.float32),
        heading_dev_ratio=np.asarray([[h]], dtype=np.float32),
        delta_global_dev_ratio=np.asarray([[0.0]], dtype=np.float32),
        delta_center_dev_ratio=np.asarray([[0.0]], dtype=np.float32),
        delta_heading_dev_ratio=np.asarray([[0.0]], dtype=np.float32),
        off_road=np.asarray([[0.0]], dtype=np.float32),
        opposite_lane=np.asarray([[0.0]], dtype=np.float32),
        emergency_lane=np.asarray([[0.0]], dtype=np.float32),
        off_global_route=np.asarray([[0.0]], dtype=np.float32),
        collision=np.asarray([[0.0]], dtype=np.float32),
        red_light_dev=np.asarray([False]),
        stop_sign_dev=np.asarray([False]),
    )

    expected_eff = math.exp(-params["w_g"] * g) * math.exp(-params["w_c"] * c) * math.exp(-params["w_h"] * h)
    assert terms.terms["efficiency"][0, 0] == pytest.approx(expected_eff)
    assert terms.candidate_scores[0] == pytest.approx(params["w_prog"] * expected_eff)


def test_carl_reward_clips_and_gates_correction_reward() -> None:
    params = dict(CRAFT_CARL_FORWARD_SIM_DEFAULTS)
    params.update({"ddev_clip": 0.1, "corr_clip": 0.5, "k_g": 1.0, "k_c": 1.0, "k_h": 1.0})

    terms = compute_carl_reward_numpy(
        params=params,
        delta_progress=np.asarray([[0.6]], dtype=np.float32),
        global_dev_ratio=np.asarray([[0.5]], dtype=np.float32),
        center_dev_ratio=np.asarray([[0.5]], dtype=np.float32),
        heading_dev_ratio=np.asarray([[0.5]], dtype=np.float32),
        delta_global_dev_ratio=np.asarray([[-1.0]], dtype=np.float32),
        delta_center_dev_ratio=np.asarray([[-1.0]], dtype=np.float32),
        delta_heading_dev_ratio=np.asarray([[-1.0]], dtype=np.float32),
        off_road=np.asarray([[0.0]], dtype=np.float32),
        opposite_lane=np.asarray([[0.0]], dtype=np.float32),
        emergency_lane=np.asarray([[0.0]], dtype=np.float32),
        off_global_route=np.asarray([[0.0]], dtype=np.float32),
        collision=np.asarray([[0.0]], dtype=np.float32),
        red_light_dev=np.asarray([False]),
        stop_sign_dev=np.asarray([False]),
    )

    dp_norm = 0.6 / params["dp_max"]
    unclipped = 0.1 + 0.1 + 0.1
    assert terms.terms["correction_reward"][0, 0] == pytest.approx(unclipped * (0.5 + 0.5 * dp_norm))


def test_carl_reward_applies_safety_and_terminal_costs() -> None:
    terms = compute_carl_reward_numpy(
        params=CRAFT_CARL_FORWARD_SIM_DEFAULTS,
        delta_progress=np.asarray([[0.0, 0.0]], dtype=np.float32),
        global_dev_ratio=np.zeros((1, 2), dtype=np.float32),
        center_dev_ratio=np.zeros((1, 2), dtype=np.float32),
        heading_dev_ratio=np.zeros((1, 2), dtype=np.float32),
        delta_global_dev_ratio=np.zeros((1, 2), dtype=np.float32),
        delta_center_dev_ratio=np.zeros((1, 2), dtype=np.float32),
        delta_heading_dev_ratio=np.zeros((1, 2), dtype=np.float32),
        off_road=np.asarray([[1.0, 0.0]], dtype=np.float32),
        opposite_lane=np.asarray([[0.0, 1.0]], dtype=np.float32),
        emergency_lane=np.zeros((1, 2), dtype=np.float32),
        off_global_route=np.zeros((1, 2), dtype=np.float32),
        collision=np.asarray([[0.0, 1.0]], dtype=np.float32),
        red_light_dev=np.asarray([True]),
        stop_sign_dev=np.asarray([True]),
    )

    expected = -(
        CRAFT_CARL_FORWARD_SIM_DEFAULTS["cost_off_road"]
        + CRAFT_CARL_FORWARD_SIM_DEFAULTS["cost_opposite_lane"]
        + CRAFT_CARL_FORWARD_SIM_DEFAULTS["term_collision"]
        + CRAFT_CARL_FORWARD_SIM_DEFAULTS["term_red_light"]
        + CRAFT_CARL_FORWARD_SIM_DEFAULTS["term_stop_sign"]
    )
    assert terms.candidate_scores[0] == pytest.approx(expected)
