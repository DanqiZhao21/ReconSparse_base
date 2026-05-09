from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from framework.rewardmodel.constants import REWARD_METRIC_NAMES


PDM_TO_INTERNAL_METRIC_NAMES = {
    "no_at_fault_collisions": "rnc",
    "drivable_area_compliance": "rdac",
    "driving_direction_compliance": "rddc",
    "traffic_light_compliance": "rtlc",
    "ego_progress": "rep",
    "time_to_collision_within_bound": "rttc",
    "lane_keeping": "rlk",
    "history_comfort": "rhc",
}


def map_pdm_metric_names(metric_dict: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    mapped: dict[str, np.ndarray] = {}
    for key, value in metric_dict.items():
        internal = PDM_TO_INTERNAL_METRIC_NAMES.get(str(key))
        if internal is None:
            continue
        mapped[internal] = np.asarray(value, dtype=np.float32)
    return mapped


def stack_temporal_metric_targets(metric_scores_per_horizon: Sequence[Mapping[str, np.ndarray]]) -> np.ndarray:
    if len(metric_scores_per_horizon) <= 0:
        raise ValueError("metric_scores_per_horizon must not be empty")
    horizons = []
    for horizon_scores in metric_scores_per_horizon:
        row = []
        for name in REWARD_METRIC_NAMES:
            if name not in horizon_scores:
                raise KeyError(f"Missing reward metric {name} in horizon scores")
            row.append(np.asarray(horizon_scores[name], dtype=np.float32))
        horizons.append(np.stack(row, axis=-1))
    return np.stack(horizons, axis=1)

