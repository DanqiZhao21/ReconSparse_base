from __future__ import annotations

import pytest

from framework.rollout.collector import _accumulate_reward_summary, _reward_summary_defaults


def test_reward_summary_accumulates_closed_loop_ea_range_fields() -> None:
    summary = _reward_summary_defaults()

    _accumulate_reward_summary(
        summary,
        {
            "ea_available": True,
            "ea_max": 3.0,
            "ea_min": 1.0,
            "ea_mean": 2.0,
            "ea_risk": 0.25,
            "ea_cost": 0.5,
            "ea_evaluated_pairs": 2.0,
        },
        reward=1.0,
    )
    _accumulate_reward_summary(
        summary,
        {
            "ea_available": True,
            "ea_max": 5.0,
            "ea_min": 0.5,
            "ea_mean": 3.0,
            "ea_risk": 0.75,
            "ea_cost": 1.5,
            "ea_evaluated_pairs": 1.0,
        },
        reward=-1.0,
    )

    assert summary["ea_available_count"] == pytest.approx(2.0)
    assert summary["ea_evaluated_pairs_sum"] == pytest.approx(3.0)
    assert summary["ea_cost_sum"] == pytest.approx(2.0)
    assert summary["ea_risk_sum"] == pytest.approx(1.0)
    assert summary["ea_max_value"] == pytest.approx(5.0)
    assert summary["ea_min_value"] == pytest.approx(0.5)
    assert summary["ea_mean_sum"] == pytest.approx(5.0)

