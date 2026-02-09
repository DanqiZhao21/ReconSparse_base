from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from framework.interfaces import AgentAPI, EnvAPI
from framework.agent.policy_diffusiondrivev2 import DiffusionDriveV2Policy
from framework.env_wrapper import RLReconEnv

"""
Adapter layer that wraps concrete env/agent implementations to match
EnvAPI and AgentAPI, enabling interface-based RL training.
"""

class RLReconEnvAdapter(EnvAPI):
    def __init__(
        self,
        *,
        cuda: int,
        scene: int,
        reward_cfg: Dict[str, Any] | None = None,
        debug: bool = False,
        render_w: int | None = None,
        render_h: int | None = None,
    ) -> None:
        self._env = RLReconEnv(
            cuda=cuda,
            scene=scene,
            reward_cfg=reward_cfg or {},
            debug=debug,
            render_w=render_w,
            render_h=render_h,
        )

    def reset(
        self,
        scene: int | None = None,
        *,
        start_frame: int | None = None,
        step_frames: int | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self._env.reset(scene=scene, start_frame=start_frame, step_frames=step_frames)

    def step(self, action: Tuple[float, float, float, int] | Tuple[int, int] | Tuple[int, int, int]):
        return self._env.step(action)

    def finalize_episode_reward(self, *, done_reason: str = "timeout") -> Tuple[float, Dict[str, Any]]:
        return self._env.finalize_episode_reward(done_reason=done_reason)


class DDV2AgentAdapter(AgentAPI):
    def __init__(
        self,
        *,
        x_anchor: int,
        y_anchor: int,
        ckpt_path: str,
        device: str,
        rl_lr: float,
        reinforce_baseline_beta: float,
    ) -> None:
        self._policy = DiffusionDriveV2Policy(
            x_anchor=x_anchor,
            y_anchor=y_anchor,
            ckpt_path=ckpt_path,
            device=device,
            rl_lr=rl_lr,
            reinforce_baseline_beta=reinforce_baseline_beta,
        )

    def sample_with_replay(
        self,
        observation: Dict[str, Any],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ):
        return self._policy.sample_ddv2rl_with_replay(
            observation,
            eta=eta,
            mode_idx=mode_idx,
            mode_select=mode_select,
        )

    def sample_with_replay_batch(
        self,
        observations: List[Dict[str, Any]],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ):
        return self._policy.sample_ddv2rl_with_replay_batch(
            observations,
            eta=eta,
            mode_idx=mode_idx,
            mode_select=mode_select,
        )

    def logp_from_replay(self, replay: Dict[str, Any], *, eta: float = 1.0) -> torch.Tensor:
        return self._policy.logp_from_replay(replay, eta=eta)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self._policy.state_dict()

    def save_checkpoint(self, path: str) -> None:
        self._policy.save_checkpoint(path)

    def load_from_checkpoint(self, path: str, *, strict: bool = False) -> None:
        self._policy.load_from_checkpoint(path, strict=bool(strict))

    @property
    def device(self) -> torch.device:
        return self._policy.device

    def wrap_ddp(
        self,
        *,
        device_id: int,
        process_group: Any | None = None,
        find_unused_parameters: bool = True,
        rl_lr: float | None = None,
    ) -> None:
        self._policy.wrap_ddp(
            device_id=device_id,
            process_group=process_group,
            find_unused_parameters=find_unused_parameters,
            rl_lr=rl_lr,
        )
