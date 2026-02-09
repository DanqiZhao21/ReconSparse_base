"""Agent interfaces and concrete adapters."""

from .base import Agent
from .policy_diffusiondrivev2 import DiffusionDriveV2Policy

# Backward-friendly alias
DiffusionDriveV2Agent = DiffusionDriveV2Policy

__all__ = [
	"Agent",
	"DiffusionDriveV2Policy",
	"DiffusionDriveV2Agent",
]
