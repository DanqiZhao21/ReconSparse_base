from __future__ import annotations

import pytest
import torch

from framework.rollout.collector import _check_replay_logp_consistency


class _ReplayLogpAgent:
    def __init__(self, values: list[float]) -> None:
        self.values = torch.tensor(values, dtype=torch.float32)

    def logp_from_replay_batch(self, replays, *, eta: float = 1.0):
        del eta
        assert len(replays) == int(self.values.numel())
        return self.values


def test_replay_logp_self_check_passes_when_recomputed_logps_match() -> None:
    messages: list[str] = []
    summary = _check_replay_logp_consistency(
        agent=_ReplayLogpAgent([-1.0, -2.0]),
        logps=[torch.tensor(-1.0), torch.tensor(-2.0)],
        replays=[{"i": 0}, {"i": 1}],
        eta=1.0,
        tolerance=1.0e-5,
        context="unit",
        fail_on_error=True,
        log_fn=messages.append,
    )

    assert summary["pass"] == 1.0
    assert summary["mismatch_count"] == 0
    assert summary["max_abs_error"] == 0.0
    assert any("PASS" in message for message in messages)


def test_replay_logp_self_check_reports_and_can_fail_on_mismatch() -> None:
    messages: list[str] = []
    with pytest.raises(RuntimeError, match="replay logp self-check failed"):
        _check_replay_logp_consistency(
            agent=_ReplayLogpAgent([-1.0, -3.5]),
            logps=[torch.tensor(-1.0), torch.tensor(-2.0)],
            replays=[{"i": 0}, {"i": 1}],
            eta=1.0,
            tolerance=1.0e-5,
            context="unit",
            fail_on_error=True,
            log_fn=messages.append,
        )

    assert any("FAIL" in message for message in messages)
    assert any("max_abs_error=1.5" in message for message in messages)
