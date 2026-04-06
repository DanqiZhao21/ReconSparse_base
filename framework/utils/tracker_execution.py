from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass
class TrackerExecutionResult:
    tracked_rollout_local_xyyaw: np.ndarray
    tracked_first_local_xyyaw: np.ndarray
    executed_local_xyyaw: np.ndarray
    executed_pose: np.ndarray
    velocity_xy: np.ndarray
    acceleration_xy: np.ndarray
    steering_angle: float
    steering_rate: float
    propagated_state: np.ndarray | None = None
    command_state: np.ndarray | None = None


def pose_matrix_from_local_xyyaw(x: float, y: float, yaw: float) -> np.ndarray:
    c = float(math.cos(float(yaw)))
    s = float(math.sin(float(yaw)))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    return T


def local_xyyaw_from_state_row(state_row: np.ndarray, state_index: Any) -> np.ndarray:
    row = np.asarray(state_row, dtype=np.float64).reshape(-1)
    return np.asarray(
        [
            float(row[int(state_index.X)]),
            float(row[int(state_index.Y)]),
            float(row[int(state_index.HEADING)]),
        ],
        dtype=np.float64,
    )


def build_execution_result(
    *,
    prev_pose: np.ndarray,
    tracked_rollout_local_xyyaw: np.ndarray,
    tracked_first_local_xyyaw: np.ndarray,
    velocity_xy: np.ndarray,
    acceleration_xy: np.ndarray,
    steering_angle: float,
    steering_rate: float,
    propagated_state: np.ndarray | None = None,
    command_state: np.ndarray | None = None,
    state_index: Any | None = None,
) -> TrackerExecutionResult:
    if propagated_state is not None and state_index is not None:
        executed_local_xyyaw = local_xyyaw_from_state_row(propagated_state, state_index)
    else:
        executed_local_xyyaw = np.asarray(tracked_first_local_xyyaw, dtype=np.float64).reshape(3)

    executed_pose = np.asarray(prev_pose, dtype=np.float64) @ pose_matrix_from_local_xyyaw(
        float(executed_local_xyyaw[0]),
        float(executed_local_xyyaw[1]),
        float(executed_local_xyyaw[2]),
    )

    return TrackerExecutionResult(
        tracked_rollout_local_xyyaw=np.asarray(tracked_rollout_local_xyyaw, dtype=np.float64),
        tracked_first_local_xyyaw=np.asarray(tracked_first_local_xyyaw, dtype=np.float64).reshape(3),
        executed_local_xyyaw=np.asarray(executed_local_xyyaw, dtype=np.float64).reshape(3),
        executed_pose=np.asarray(executed_pose, dtype=np.float64),
        velocity_xy=np.asarray(velocity_xy, dtype=np.float32).reshape(2),
        acceleration_xy=np.asarray(acceleration_xy, dtype=np.float32).reshape(2),
        steering_angle=float(steering_angle),
        steering_rate=float(steering_rate),
        propagated_state=None if propagated_state is None else np.asarray(propagated_state, dtype=np.float64).reshape(-1),
        command_state=None if command_state is None else np.asarray(command_state, dtype=np.float64).reshape(-1),
    )
