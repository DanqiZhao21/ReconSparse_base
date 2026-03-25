from __future__ import annotations

import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EGOADS_ROOT = os.path.join(REPO_ROOT, "egoADs")


def resolve_ego_ads_subdir(name: str) -> str:
    preferred = os.path.join(EGOADS_ROOT, str(name))
    if os.path.isdir(preferred):
        return preferred
    fallback = os.path.join(REPO_ROOT, str(name))
    return preferred if os.path.exists(preferred) else fallback


def resolve_repo_path(path: str) -> str:
    text = str(path)
    if os.path.isabs(text):
        return text
    direct = os.path.join(REPO_ROOT, text)
    if os.path.exists(direct):
        return direct
    if text.startswith("egoADs/"):
        return direct
    egoads_candidate = os.path.join(EGOADS_ROOT, text)
    if os.path.exists(egoads_candidate):
        return egoads_candidate
    return direct


__all__ = [
    "EGOADS_ROOT",
    "REPO_ROOT",
    "resolve_ego_ads_subdir",
    "resolve_repo_path",
]