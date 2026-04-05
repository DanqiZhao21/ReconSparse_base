from __future__ import annotations

import datetime
import os
from typing import Optional

import torch
import torch.distributed as dist


def learner_init_dist(*, timeout_s: Optional[int] = None) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if timeout_s is None:
            timeout_s = 2 * 60 * 60
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(seconds=int(timeout_s)),
        )
    return rank, world_size, local_rank


__all__ = ["learner_init_dist"]
