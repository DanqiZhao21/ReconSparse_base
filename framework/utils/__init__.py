"""Small utilities used across the framework."""

from .gsplat_backend import ensure_gsplat_backend, ensure_gsplat_legacy_backend, should_use_legacy_gsplat
from .gsplat_warmup import build_gsplat_warmup_cmd, warmup_gsplat_cuda
from .torch_extension import patch_cpp_extension_load

__all__ = [
    "build_gsplat_warmup_cmd",
    "ensure_gsplat_backend",
    "ensure_gsplat_legacy_backend",
    "patch_cpp_extension_load",
    "should_use_legacy_gsplat",
    "warmup_gsplat_cuda",
]
