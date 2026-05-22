"""Environment wrappers.

This package is the canonical home for environment wrappers and 3DGS-related
utilities used by the simulator + RL training loops.

Backward-compatible re-exports exist under `reconsimulator.envs.*`.
"""

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


def __getattr__(name: str):
    if name == "RLReconEnv":
        from .rl_wrapper import RLReconEnv

        return RLReconEnv
    if name in {"SceneSamplingEnv", "SceneSamplingSpec", "make_scene_sampling_env", "SerialVecEnv", "SubprocVecEnv"}:
        from .subproc_vec_env import (
            SceneSamplingEnv,
            SceneSamplingSpec,
            SerialVecEnv,
            SubprocVecEnv,
            make_scene_sampling_env,
        )

        values = {
            "SceneSamplingEnv": SceneSamplingEnv,
            "SceneSamplingSpec": SceneSamplingSpec,
            "make_scene_sampling_env": make_scene_sampling_env,
            "SerialVecEnv": SerialVecEnv,
            "SubprocVecEnv": SubprocVecEnv,
        }
        return values[name]
    if name in {"clear_splat_cache", "get_splat", "get_sky_view", "get_rays", "get_state", "move_to_device", "slerp"}:
        from .tool import (
            clear_splat_cache,
            get_rays,
            get_sky_view,
            get_splat,
            get_state,
            move_to_device,
            slerp,
        )

        values = {
            "clear_splat_cache": clear_splat_cache,
            "get_splat": get_splat,
            "get_sky_view": get_sky_view,
            "get_rays": get_rays,
            "get_state": get_state,
            "move_to_device": move_to_device,
            "slerp": slerp,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
