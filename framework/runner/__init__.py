"""Runner helpers for actor-learner orchestration and launch setup."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "actor_main",
    "build_launch_env",
    "learner_main",
    "orchestrator_main",
    "normalize_actor_learner_cfg",
    "resolve_actor_gpu_ids",
]

_LAZY_ATTRS = {
    "actor_main": ("framework.runner.actor_runtime", "actor_main"),
    "learner_main": ("framework.runner.learner_runtime", "learner_main"),
    "orchestrator_main": ("framework.runner.orchestrator", "orchestrator_main"),
    "build_launch_env": ("framework.runner.launch_env", "build_launch_env"),
    "normalize_actor_learner_cfg": ("framework.runner.config_normalization", "normalize_actor_learner_cfg"),
    "resolve_actor_gpu_ids": ("framework.runner.config_normalization", "resolve_actor_gpu_ids"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
