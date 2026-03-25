"""Launch-environment helpers for actor-learner processes.

This module centralizes runtime environment preparation that used to live in
shell launchers: CUDA include/library discovery, PYTHONPATH bootstrapping, and
Torch extension cache setup.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Dict, List, Optional

from framework.utils.repo_paths import REPO_ROOT, resolve_ego_ads_subdir


def _normalize_agent_type(agent_type: Optional[str]) -> str:
    return str(agent_type or "ddv2").strip().lower().replace("-", "_")


def prepend_env_path(env: Dict[str, str], key: str, values: List[str]) -> None:
    existing = env.get(key, "")
    parts: List[str] = []
    seen = set()
    for value in values + ([existing] if existing else []):
        if not value:
            continue
        for item in str(value).split(os.pathsep):
            item = item.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            parts.append(item)
    if len(parts) > 0:
        env[key] = os.pathsep.join(parts)


def candidate_cuda_roots() -> List[str]:
    roots: List[str] = []
    for item in [
        os.environ.get("CUDA_HOME", ""),
        os.environ.get("CUDA_PATH", ""),
        sys.prefix,
        sys.exec_prefix,
        "/usr/local/cuda",
    ]:
        text = str(item).strip()
        if text and text not in roots:
            roots.append(text)
    return roots


def candidate_cuda_include_dirs() -> List[str]:
    includes: List[str] = []
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for root in candidate_cuda_roots():
        for path in [
            os.path.join(root, "include"),
            os.path.join(root, "targets", "x86_64-linux", "include"),
            os.path.join(root, "lib", py_ver, "site-packages", "nvidia", "cuda_runtime", "include"),
            os.path.join(root, "lib64", py_ver, "site-packages", "nvidia", "cuda_runtime", "include"),
        ]:
            if os.path.exists(os.path.join(path, "cuda_runtime.h")) and path not in includes:
                includes.append(path)
    return includes


def candidate_cuda_library_dirs() -> List[str]:
    libs: List[str] = []
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for root in candidate_cuda_roots():
        for path in [
            os.path.join(root, "lib64"),
            os.path.join(root, "targets", "x86_64-linux", "lib"),
            os.path.join(root, "lib", py_ver, "site-packages", "nvidia", "cuda_runtime", "lib"),
            os.path.join(root, "lib", py_ver, "site-packages", "nvidia", "cuda_runtime", "lib64"),
        ]:
            if os.path.isdir(path) and path not in libs:
                libs.append(path)
    return libs


def build_launch_env(*, agent_type: Optional[str] = None) -> Dict[str, str]:
    env = os.environ.copy()

    ddv2_root = resolve_ego_ads_subdir("DiffusionDriveV2")
    sparse_root = resolve_ego_ads_subdir("SparseDrive")
    sparse_v2_root = resolve_ego_ads_subdir("SparseDriveV2")
    navsim_root = os.path.join(ddv2_root, "navsim")
    torch_ext_dir = env.get("TORCH_EXTENSIONS_DIR", os.path.join(REPO_ROOT, ".cache", "torch_extensions"))
    pathlib.Path(torch_ext_dir).mkdir(parents=True, exist_ok=True)
    env["TORCH_EXTENSIONS_DIR"] = str(torch_ext_dir)

    agent_key = _normalize_agent_type(agent_type)
    pythonpath_entries: List[str] = [REPO_ROOT]
    if agent_key in {"sparsedrive_v2", "sparsedrivev2", "sdv2"}:
        pythonpath_entries.extend([sparse_v2_root, sparse_root, ddv2_root, navsim_root])
    elif agent_key == "sparsedrive":
        pythonpath_entries.extend([sparse_root, sparse_v2_root, ddv2_root, navsim_root])
    else:
        pythonpath_entries.extend([ddv2_root, navsim_root, sparse_root, sparse_v2_root])

    prepend_env_path(env, "PYTHONPATH", pythonpath_entries)

    cuda_home = env.get("CUDA_HOME", "").strip()
    if not cuda_home:
        for root in candidate_cuda_roots():
            include_dir = os.path.join(root, "include")
            target_include_dir = os.path.join(root, "targets", "x86_64-linux", "include")
            if os.path.exists(os.path.join(include_dir, "cuda_runtime.h")) or os.path.exists(os.path.join(target_include_dir, "cuda_runtime.h")):
                cuda_home = root
                break
    if cuda_home:
        env["CUDA_HOME"] = str(cuda_home)

    include_dirs = candidate_cuda_include_dirs()
    library_dirs = candidate_cuda_library_dirs()
    prepend_env_path(env, "CPATH", include_dirs)
    prepend_env_path(env, "CPLUS_INCLUDE_PATH", include_dirs)
    prepend_env_path(env, "LIBRARY_PATH", library_dirs)
    prepend_env_path(env, "LD_LIBRARY_PATH", library_dirs)
    return env


__all__ = [
    "build_launch_env",
    "candidate_cuda_include_dirs",
    "candidate_cuda_library_dirs",
    "candidate_cuda_roots",
    "prepend_env_path",
]