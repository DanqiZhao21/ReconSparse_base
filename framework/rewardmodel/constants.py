from __future__ import annotations

REWARD_METRIC_NAMES = (
    "rnc",
    "rdac",
    "rddc",
    "rtlc",
    "rep",
    "rttc",
    "rlk",
    "rhc",
)

SAFETY_REWARD_METRICS = ("rnc", "rdac", "rddc", "rtlc")
TASK_REWARD_METRICS = ("rep", "rttc", "rlk", "rhc")

REWARD_METRIC_TO_INDEX = {name: idx for idx, name in enumerate(REWARD_METRIC_NAMES)}
SAFETY_METRIC_INDICES = tuple(REWARD_METRIC_TO_INDEX[name] for name in SAFETY_REWARD_METRICS)
TASK_METRIC_INDICES = tuple(REWARD_METRIC_TO_INDEX[name] for name in TASK_REWARD_METRICS)

DEFAULT_HORIZON_WEIGHTS = (1.0,) * 8
DEFAULT_METRIC_WEIGHTS = (1.0,) * 8

