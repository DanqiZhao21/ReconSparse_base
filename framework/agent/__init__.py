"""Agent base class and concrete policy implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Agent",
    "DiffusionDriveV2Policy",
    "DummyPolicy",
    "SparseDrivePolicy",
    "SparseDriveV2Policy",
    "DiffusionDriveV2Agent",
]

_LAZY_ATTRS = {
    "Agent": ("framework.agent.base", "Agent"),
    "DiffusionDriveV2Policy": ("framework.agent.policy_diffusiondrivev2", "DiffusionDriveV2Policy"),
    "DummyPolicy": ("framework.agent.policy_dummy", "DummyPolicy"),
    "SparseDrivePolicy": ("framework.agent.policy_sparsedrive", "SparseDrivePolicy"),
    "SparseDriveV2Policy": ("framework.agent.policy_sparsedrive_v2", "SparseDriveV2Policy"),
}


def __getattr__(name: str) -> Any:
    if name == "DiffusionDriveV2Agent":
        value = getattr(import_module("framework.agent.policy_diffusiondrivev2"), "DiffusionDriveV2Policy")
        globals()[name] = value
        return value

    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
