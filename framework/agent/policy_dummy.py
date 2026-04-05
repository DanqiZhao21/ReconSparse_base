from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from .base import Agent


def _coerce_feature(value: Any) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().view(-1)[0].item())
    try:
        return float(value)
    except Exception:
        return 0.0


class DummyPolicy(Agent):
    """Small test-only policy used by framework smoke coverage."""

    def __init__(
        self,
        *,
        ckpt_path: str | None,
        device: str = "cpu",
        rl_lr: float = 1e-3,
    ) -> None:
        self.device = torch.device(device)
        self._model = nn.Linear(1, 1).to(self.device)
        self._optimizer = None
        self.rl_lr = float(rl_lr)
        if ckpt_path:
            self.load_checkpoint(ckpt_path, strict=False)

    def _feature_tensor(self, feature: Any) -> torch.Tensor:
        return torch.tensor([[float(_coerce_feature(feature))]], device=self.device, dtype=torch.float32)

    def act(
        self,
        observation: Dict[str, Any],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ) -> Tuple[Tuple[float, float, float, int], torch.Tensor, Dict[str, Any]]:
        del eta, mode_idx, mode_select
        feature = _coerce_feature(observation.get("feature", 0.0))
        logp = self._model(self._feature_tensor(feature)).view(())
        action = (float(feature), 0.0, 0.0, 0)
        replay = {"feature": float(feature)}
        return action, logp, replay

    def act_batch(
        self,
        observations: List[Dict[str, Any]],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ) -> Tuple[List[Tuple[float, float, float, int]], List[torch.Tensor], List[Dict[str, Any]]]:
        del eta, mode_idx, mode_select
        actions: List[Tuple[float, float, float, int]] = []
        logps: List[torch.Tensor] = []
        replays: List[Dict[str, Any]] = []
        for observation in observations:
            action, logp, replay = self.act(observation)
            actions.append(action)
            logps.append(logp)
            replays.append(replay)
        return actions, logps, replays

    def logp_from_replay(self, replay: Dict[str, Any], *, eta: float = 1.0) -> torch.Tensor:
        del eta
        return self._model(self._feature_tensor(replay.get("feature", 0.0))).view(())

    def logp_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        eta: float = 1.0,
    ) -> torch.Tensor:
        del eta
        if len(replays) == 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)
        features = torch.tensor(
            [[_coerce_feature(replay.get("feature", 0.0))] for replay in replays],
            device=self.device,
            dtype=torch.float32,
        )
        return self._model(features).view(-1)

    def save_checkpoint(self, path: str) -> None:
        torch.save({"state_dict": self._model.state_dict()}, path)

    def load_checkpoint(self, path: str, *, strict: bool = False) -> None:
        try:
            payload = torch.load(path, map_location=self.device)
        except FileNotFoundError:
            if strict:
                raise
            return

        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            if strict:
                raise RuntimeError(f"Unsupported dummy checkpoint payload at {path}")
            return
        self._model.load_state_dict(state_dict, strict=bool(strict))

    def parameters(self):
        return self._model.parameters()


__all__ = ["DummyPolicy"]
