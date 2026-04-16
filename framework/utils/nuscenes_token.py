from __future__ import annotations

from typing import Any

from reconsimulator.envs.metrics import _load_token_frame_map


def resolve_sample_token(scene_id: Any, frame_idx: Any) -> str | None:
    try:
        sid = int(scene_id)
        fidx = int(frame_idx)
    except Exception:
        return None

    try:
        frame_to_token = _load_token_frame_map(int(sid))
    except Exception:
        return None

    token = frame_to_token.get(int(fidx), None)
    if token is None and int(fidx) >= 0:
        token = frame_to_token.get(int(fidx // 5) * 5, None)
    if token is None:
        return None
    return str(token)


__all__ = ["resolve_sample_token"]
