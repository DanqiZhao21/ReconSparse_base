from __future__ import annotations

from typing import Any, List, Optional

import torch

from framework.algorithms.trajectory_batch import LoadedShardBatch, build_training_batch as _build_training_batch


def build_training_batch(
    *,
    selected: List[str],
    algo_key: str,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
    value_net: Optional[torch.nn.Module],
    value_optim: Optional[torch.optim.Optimizer],
    ddp_enabled: bool,
    dist_module: Any,
    norm_eps: float = 1e-8,
) -> LoadedShardBatch:
    return _build_training_batch(
        selected=selected,
        algo_key=algo_key,
        device=device,
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        value_net=value_net,
        value_optim=value_optim,
        ddp_enabled=ddp_enabled,
        dist_module=dist_module,
        norm_eps=float(norm_eps),
    )