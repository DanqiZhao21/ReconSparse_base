"""Algorithm specs and objective cores.

Note:
- Keep this module side-effect free and only import symbols that actually exist.
- Some scripts import submodules (e.g. `framework.algorithms.reinforcepp`); Python executes
  this `__init__` first, so stale imports here can crash the whole entrypoint.
"""

from .base import Algorithm


def __getattr__(name: str):
  if name == "PPO":
    from .ppo import PPO

    return PPO
  if name == "ReinforcePP":
    from .reinforcepp import ReinforcePP

    return ReinforcePP
  if name == "SAC":
    from .sac import SAC

    return SAC
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Algorithm",
    "PPO",
    "ReinforcePP",
    "SAC",
]
