"""Runtime patches for third-party extension loading in this repo."""

from __future__ import annotations

import os

try:
    import torch.utils.cpp_extension as _cpp_extension
except Exception:  # pragma: no cover
    _cpp_extension = None


if _cpp_extension is not None and not getattr(_cpp_extension.load, '_recondreamer_builddir_patch', False):
    _orig_load = _cpp_extension.load

    def _patched_load(name, sources, *args, **kwargs):
        build_directory = kwargs.get('build_directory')
        if build_directory:
            os.makedirs(build_directory, exist_ok=True)
        return _orig_load(name, sources, *args, **kwargs)

    _patched_load._recondreamer_builddir_patch = True
    _cpp_extension.load = _patched_load
