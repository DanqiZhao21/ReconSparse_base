from __future__ import annotations

import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EGOADS_ROOT = os.path.join(REPO_ROOT, "egoADs")
DEFAULT_HUGSIM_ROOT = "/root/clone/HUGSIM-ORI"


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


def resolve_hugsim_root() -> str:
    return os.path.abspath(os.environ.get("HUGSIM_ROOT", DEFAULT_HUGSIM_ROOT))


def resolve_hugsim_path(path: str | None, *default_parts: str) -> str | None:
    if path is None:
        if not default_parts:
            return None
        return os.path.join(resolve_hugsim_root(), *default_parts)
    text = os.path.expanduser(str(path))
    if os.path.isabs(text):
        return text
    hugsim_candidate = os.path.join(resolve_hugsim_root(), text)
    if os.path.exists(hugsim_candidate):
        return hugsim_candidate
    return resolve_repo_path(text)


__all__ = [
    "EGOADS_ROOT",
    "DEFAULT_HUGSIM_ROOT",
    "REPO_ROOT",
    "resolve_hugsim_path",
    "resolve_hugsim_root",
    "resolve_ego_ads_subdir",
    "resolve_repo_path",
]
