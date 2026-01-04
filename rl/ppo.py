from typing import Any, Dict, Tuple

import numpy as np


class PPOAgent:
    """
    Minimal PPO-like interface placeholder.
    - `act(observation)` returns a (ax, ay, flag) action.
    - `update(batch)` is a stub for algorithm updates.

    This is intentionally lightweight to establish the agent–env interface.
    Replace internals with a proper actor-critic when integrating the full model.
    """

    def __init__(self, x_anchor: int = 61, y_anchor: int = 61):
        self.x_anchor = int(x_anchor)
        self.y_anchor = int(y_anchor)
    #TODO:
    def act(self, observation: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
        # Random action over anchor grid; flag=0 uses planned anchors in env.
        ax = np.random.randint(0, self.x_anchor)
        ay = np.random.randint(0, self.y_anchor)
        flag = 0
        return int(ax), int(ay), int(flag)
    #TODO:
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        # Placeholder: return empty metrics.
        return {"loss_pi": 0.0, "loss_v": 0.0}
