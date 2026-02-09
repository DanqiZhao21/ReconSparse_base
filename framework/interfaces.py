from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

import torch

"""
Interface definitions for environments and agents in closed-loop reinforcement learning.

This module specifies the minimal contracts between the environment (world) and
the agent (policy) using `typing.Protocol`

- EnvAPI defines how an environment is reset, stepped, and how episode-level
  rewards are finalized.
- AgentAPI defines how a policy samples actions with replay support, computes
  log-probabilities for policy-gradient methods, and manages checkpoints and
  distributed wrapping.

Algorithms (e.g., PPO, REINFORCE) depend only on these interfaces, not on concrete
implementations.

"""



class EnvAPI(Protocol):
    def reset(
        self,
        scene: int | None = None,
        *,
        start_frame: int | None = None,
        step_frames: int | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...

    def step(self, action: Tuple[float, float, float, int] | Tuple[int, int] | Tuple[int, int, int]): ...

    def finalize_episode_reward(self, *, done_reason: str = "timeout") -> Tuple[float, Dict[str, Any]]: ...


class AgentAPI(Protocol):
    def sample_with_replay(
        self,
        observation: Dict[str, Any],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ): ...

    def sample_with_replay_batch(
        self,
        observations: List[Dict[str, Any]],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ): ...

    def logp_from_replay(self, replay: Dict[str, Any], *, eta: float = 1.0) -> torch.Tensor: ...

    def state_dict(self) -> Dict[str, torch.Tensor]: ...

    def save_checkpoint(self, path: str) -> None: ...

    def load_from_checkpoint(self, path: str, *, strict: bool = False) -> None: ...

    @property
    def device(self) -> torch.device: ...

    def wrap_ddp(
        self,
        *,
        device_id: int,
        process_group: Any | None = None,
        find_unused_parameters: bool = True,
        rl_lr: float | None = None,
    ) -> None: ...
