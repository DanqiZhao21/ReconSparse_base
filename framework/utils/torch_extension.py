"""Helpers for patching torch cpp extension loading quirks."""

from __future__ import annotations

import os


def patch_cpp_extension_load() -> None:
    try:
        import torch.utils.cpp_extension as cpp_extension
    except Exception:
        return

    load = getattr(cpp_extension, 'load', None)
    if load is None or getattr(load, '_recondreamer_builddir_patch', False):
        return

    def _patched_load(name, sources, *args, **kwargs):
        build_directory = kwargs.get('build_directory')
        if build_directory:
            os.makedirs(build_directory, exist_ok=True)
        return load(name, sources, *args, **kwargs)

    _patched_load._recondreamer_builddir_patch = True
    cpp_extension.load = _patched_load
