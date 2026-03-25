"""Environment wrappers.

This package is the canonical home for environment wrappers and 3DGS-related
utilities used by the simulator + RL training loops.

Backward-compatible re-exports exist under `reconsimulator.envs.*`.
"""

from .rl_wrapper import RLReconEnv
from .subproc_vec_env import (
    SceneSamplingEnv,
    SceneSamplingSpec,
    make_scene_sampling_env,
    SerialVecEnv,
    SubprocVecEnv,
)
from .tool import (
    clear_splat_cache,
    get_splat,
    get_sky_view,
    get_rays,
    get_state,
    move_to_device,
    slerp,
)

__all__ = [
    "RLReconEnv",
    "SceneSamplingEnv",
    "SceneSamplingSpec",
    "make_scene_sampling_env",
    "SerialVecEnv",
    "SubprocVecEnv",
    "clear_splat_cache",
    "get_splat",
    "get_sky_view",
    "get_rays",
    "get_state",
    "move_to_device",
    "slerp",
]
