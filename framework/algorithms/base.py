from __future__ import annotations

from typing import Any, Dict, List
import torch

class Algorithm:
    """Training algorithm interface (e.g., PPO, Reinforce++)."""

    def update(self, *, agent: Any, batch: Dict[str, Any], device: torch.device) -> Dict[str, float]:
        """Run one optimization update and return metrics."""
        raise NotImplementedError
