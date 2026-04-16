from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch


def _as_score_tensor(scores: Any, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(scores):
        return scores.to(device=device, dtype=torch.float32)
    if isinstance(scores, np.ndarray):
        return torch.from_numpy(scores).to(device=device, dtype=torch.float32)
    if isinstance(scores, (list, tuple)):
        return torch.as_tensor(scores, device=device, dtype=torch.float32)
    raise TypeError(f"Unsupported counterfactual score type: {type(scores)!r}")


def score_counterfactual_trajectories(
    agent: Any,
    replays: Sequence[dict[str, Any]],
    traj_xyyaw: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    scorer_fn = getattr(agent, "pdm_score_counterfactuals_from_replay_batch", None)
    if not callable(scorer_fn):
        scorer_fn = getattr(agent, "score_counterfactuals_from_replay_batch", None)
    if not callable(scorer_fn):
        raise RuntimeError(
            "GRPO is enabled but the agent does not expose a counterfactual scorer hook. "
            "Expected `pdm_score_counterfactuals_from_replay_batch(...)`."
        )

    scores = scorer_fn(replays, traj_xyyaw)
    return _as_score_tensor(scores, device=device)


__all__ = [
    "score_counterfactual_trajectories",
]
