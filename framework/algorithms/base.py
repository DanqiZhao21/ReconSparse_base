from __future__ import annotations

from typing import Any, Dict
import torch

class Algorithm:
    """Algorithm specification consumed by the learner runtime.

    The active actor-learner path owns the Trainer loop in `framework.runner`
    and `framework.lightning`. Concrete algorithm classes in this package are
    lightweight containers for optimizer/config state only.
    """

    def update(self, *, agent: Any, batch: Dict[str, Any], device: torch.device) -> Dict[str, float]:
        raise RuntimeError(
            "Algorithm.update() is no longer owned by framework.algorithms. "
            "Use the actor-learner learner runtime update path instead."
        )
