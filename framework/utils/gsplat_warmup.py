"""Helpers for serializing gsplat CUDA JIT warmup before actor fan-out."""

from __future__ import annotations

import subprocess
from typing import List, Mapping, Optional


def build_gsplat_warmup_cmd(python_executable: str) -> List[str]:
    return [
        str(python_executable),
        '-c',
        'from framework.utils.gsplat_backend import ensure_gsplat_backend; ensure_gsplat_backend()',
    ]


def warmup_gsplat_cuda(python_executable: str, *, env: Optional[Mapping[str, str]] = None) -> bool:
    try:
        subprocess.run(
            build_gsplat_warmup_cmd(python_executable),
            check=True,
            env=dict(env) if env is not None else None,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[gsplat-warmup] skipped due to warmup failure: {exc}", flush=True)
        return False
