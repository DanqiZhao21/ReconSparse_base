"""Utilities for selecting and loading a gsplat backend compatible with this Torch runtime."""

from __future__ import annotations

import glob
import importlib
import os
import sys
import types
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import _get_build_directory, load

from .torch_extension import patch_cpp_extension_load


def _parse_torch_version(version: str) -> Tuple[int, int]:
    core = str(version).split('+', 1)[0]
    parts = core.split('.')
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def should_use_legacy_gsplat(torch_version: Optional[str] = None) -> bool:
    major, minor = _parse_torch_version(torch_version or torch.__version__)
    return (major, minor) < (2, 0)


def ensure_gsplat_legacy_backend():
    module_name = 'gsplat.cuda_legacy._backend'
    module = sys.modules.get(module_name)
    if module is not None and getattr(module, '_C', None) is not None:
        return module

    patch_cpp_extension_load()
    legacy_wrapper = importlib.import_module('gsplat.cuda_legacy._wrapper')
    modern_wrapper = importlib.import_module('gsplat.cuda._wrapper')

    legacy_dir = os.path.dirname(legacy_wrapper.__file__)
    modern_dir = os.path.dirname(modern_wrapper.__file__)
    legacy_csrc = os.path.join(legacy_dir, 'csrc')
    legacy_glm = os.path.join(legacy_csrc, 'third_party', 'glm')
    modern_glm = os.path.join(modern_dir, 'csrc', 'third_party', 'glm')
    glm_path = legacy_glm if os.path.exists(legacy_glm) else modern_glm

    sources = list(glob.glob(os.path.join(legacy_csrc, '*.cu'))) + list(glob.glob(os.path.join(legacy_csrc, '*.cpp')))
    compiled = load(
        'gsplat_cuda_legacy',
        sources,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3'],
        extra_include_paths=[legacy_csrc, glm_path],
        build_directory=_get_build_directory('gsplat_cuda_legacy', verbose=False),
    )

    backend = types.ModuleType(module_name)
    backend._C = compiled
    backend.__file__ = getattr(compiled, '__file__', None)
    backend.__all__ = ['_C']
    sys.modules[module_name] = backend
    return backend


def ensure_gsplat_backend():
    if should_use_legacy_gsplat():
        return ensure_gsplat_legacy_backend()
    return importlib.import_module('gsplat.cuda._backend')
