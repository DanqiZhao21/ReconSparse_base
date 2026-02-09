from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch

class Agent:
    """Agent interface.
    Implementations should encapsulate model and checkpoint IO, and expose
    single/batch action sampling plus log-prob recomputation for RL updates.
    """

    def initialize(self) -> None:
        """Optional hook to allocate resources."""
        pass

    def act(self, observation: Dict[str, Any], *, eta: float = 1.0,
            mode_idx: int = -1, mode_select: str = "sample") -> Tuple[Tuple[float, float, float, int], torch.Tensor, Dict[str, Any]]:
        """Sample an action and return (action, logp, replay).
        - action: tuple(x, y, yaw, flag)
        - logp: scalar tensor (summed diffusion log-prob of chosen mode)
        - replay: dict containing any data needed to recompute logp later
        """
        raise NotImplementedError

    def act_batch(self, observations: List[Dict[str, Any]], *, eta: float = 1.0,
                  mode_idx: int = -1, mode_select: str = "sample") -> Tuple[List[Tuple[float, float, float, int]], List[torch.Tensor], List[Dict[str, Any]]]:
        """Batched variant of act()."""
        raise NotImplementedError

    def logp_from_replay(self, replay: Dict[str, Any], *, eta: float = 1.0) -> torch.Tensor:
        """Recompute log-prob under current params for stored replay chain."""
        raise NotImplementedError

    def save_checkpoint(self, path: str) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: str, *, strict: bool = False) -> None:
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def wrap_ddp(self, *, device_id: int, process_group: Any | None = None) -> None:
        """Optional: enable DDP for multi-GPU training."""
        pass
