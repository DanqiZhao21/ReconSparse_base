"""Helpers for serializing gsplat CUDA JIT warmup before actor fan-out."""

from __future__ import annotations

import subprocess
from typing import List, Mapping, Optional


def build_gsplat_warmup_cmd(python_executable: str) -> List[str]:
    return [
        str(python_executable),
        '-c',
        'from framework.utils.gsplat_backend import ensure_gsplat_legacy_backend; ensure_gsplat_legacy_backend()',
    ]


def warmup_gsplat_cuda(python_executable: str, *, env: Optional[Mapping[str, str]] = None) -> bool:
    try:
        cmd = build_gsplat_warmup_cmd(python_executable)
        if env is None:
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, check=True, env=dict(env))
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[gsplat-warmup] skipped due to warmup failure: {exc}", flush=True)
        return False
