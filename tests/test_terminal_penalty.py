import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from framework.rewards import TrackingRewardComputer


def test_terminal_penalty_applies_on_failure():
    computer = TrackingRewardComputer({})
    result = computer.apply_terminal_penalty(
        reward=-1.0,
        info={"terminal_kind": "failure"},
        term_cfg={
            "penalty": -5.0,
            "apply_on_failure": True,
            "apply_on_timeout": False,
            "apply_on_env_done": False,
        },
        terminal_kind="failure",
    )

    assert result.reward == -6.0
    assert result.info["terminal_penalty"] == -5.0
    assert result.info["terminal_penalty_applied"] is True


def test_terminal_penalty_applies_on_timeout():
    computer = TrackingRewardComputer({})
    result = computer.apply_terminal_penalty(
        reward=-1.0,
        info={"terminal_kind": "timeout"},
        term_cfg={
            "penalty": -3.0,
            "apply_on_failure": False,
            "apply_on_timeout": True,
            "apply_on_env_done": False,
        },
        terminal_kind="timeout",
    )

    assert result.reward == -4.0
    assert result.info["terminal_kind"] == "timeout"


def test_terminal_penalty_applies_on_env_done():
    computer = TrackingRewardComputer({})
    result = computer.apply_terminal_penalty(
        reward=2.0,
        info={"terminal_kind": "env_done"},
        term_cfg={
            "penalty": -1.5,
            "apply_on_failure": False,
            "apply_on_timeout": False,
            "apply_on_env_done": True,
        },
        terminal_kind="env_done",
    )

    assert result.reward == 0.5
    assert result.info["terminal_penalty"] == -1.5


def test_terminal_penalty_skips_when_kind_disabled():
    computer = TrackingRewardComputer({})
    result = computer.apply_terminal_penalty(
        reward=-2.0,
        info={"terminal_kind": "failure"},
        term_cfg={
            "penalty": -5.0,
            "apply_on_failure": False,
            "apply_on_timeout": False,
            "apply_on_env_done": False,
        },
        terminal_kind="failure",
    )

    assert result.reward == -2.0
    assert "terminal_penalty" not in result.info