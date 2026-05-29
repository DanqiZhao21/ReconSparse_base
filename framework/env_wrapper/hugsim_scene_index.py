from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HUGSIMFrameMapping:
    official_scene_name: str
    recon_scene_id: int
    sample_token: str
    frame_idx: int
    sample_index: int
    sample_relative_time_s: float
    hugsim_relative_time_s: float


class HUGSIMSceneIndex:
    """Map HUGSIM official nuScenes scenes onto ReconDreamer scene/frame keys."""

    def __init__(self, *, nuscenes_root: str | Path, frame2token_dir: str | Path) -> None:
        self.nuscenes_root = Path(nuscenes_root)
        self.frame2token_dir = Path(frame2token_dir)
        self._scene_rows = self._load_json_list(self.nuscenes_root / "scene.json")
        self._sample_rows = self._load_json_list(self.nuscenes_root / "sample.json")
        self._scene_by_name = {str(row["name"]): row for row in self._scene_rows if "name" in row}
        self._samples_by_scene_token: dict[str, list[dict[str, Any]]] = {}
        for row in self._sample_rows:
            scene_token = str(row.get("scene_token", ""))
            if scene_token:
                self._samples_by_scene_token.setdefault(scene_token, []).append(row)
        for rows in self._samples_by_scene_token.values():
            rows.sort(key=lambda x: int(x["timestamp"]))
        self._token_to_recon: dict[str, tuple[int, int]] = {}
        self._recon_scene_tokens: dict[int, dict[str, int]] = {}
        self._build_recon_token_index()

    @staticmethod
    def _load_json_list(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Expected list JSON at {path}")
        return [row for row in data if isinstance(row, dict)]

    def _build_recon_token_index(self) -> None:
        for path in sorted(self.frame2token_dir.glob("*.json")):
            try:
                recon_scene_id = int(path.stem)
            except ValueError:
                continue
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                continue
            token_to_frame: dict[str, int] = {}
            for token, frame_idx in payload.items():
                try:
                    frame = int(frame_idx)
                except Exception:
                    continue
                token_str = str(token)
                token_to_frame[token_str] = frame
                self._token_to_recon[token_str] = (recon_scene_id, frame)
            self._recon_scene_tokens[recon_scene_id] = token_to_frame

    def samples_for_official_scene(self, official_scene_name: str) -> list[dict[str, Any]]:
        scene = self._scene_by_name.get(str(official_scene_name))
        if scene is None:
            raise KeyError(f"Unknown nuScenes scene name: {official_scene_name}")
        return list(self._samples_by_scene_token.get(str(scene["token"]), []))

    def recon_scene_id_for_official_scene(self, official_scene_name: str) -> int:
        samples = self.samples_for_official_scene(official_scene_name)
        if not samples:
            raise KeyError(f"No samples for nuScenes scene: {official_scene_name}")
        first_token = str(samples[0]["token"])
        mapped = self._token_to_recon.get(first_token)
        if mapped is None:
            raise KeyError(f"Scene {official_scene_name} first token is not present in Recon frame2token")
        return int(mapped[0])

    def frame_for_sample_token(self, recon_scene_id: int, sample_token: str) -> int:
        token_to_frame = self._recon_scene_tokens.get(int(recon_scene_id), {})
        if str(sample_token) not in token_to_frame:
            raise KeyError(f"sample_token={sample_token!r} not found for recon scene {recon_scene_id:03d}")
        return int(token_to_frame[str(sample_token)])

    def remaining_future_sample_count(self, official_scene_name: str, sample_index: int) -> int:
        samples = self.samples_for_official_scene(str(official_scene_name))
        return max(0, int(len(samples)) - int(sample_index) - 1)

    def map_time(self, official_scene_name: str, relative_time_s: float) -> HUGSIMFrameMapping:
        samples = self.samples_for_official_scene(official_scene_name)
        if not samples:
            raise KeyError(f"No samples for nuScenes scene: {official_scene_name}")
        base_ts = int(samples[0]["timestamp"])
        rel_times = [(int(row["timestamp"]) - base_ts) / 1.0e6 for row in samples]
        target = float(relative_time_s)
        sample_index = min(range(len(rel_times)), key=lambda idx: abs(rel_times[idx] - target))
        sample = samples[sample_index]
        token = str(sample["token"])
        recon = self._token_to_recon.get(token)
        if recon is None:
            raise KeyError(f"sample_token={token!r} is not present in Recon frame2token")
        recon_scene_id, frame_idx = recon
        return HUGSIMFrameMapping(
            official_scene_name=str(official_scene_name),
            recon_scene_id=int(recon_scene_id),
            sample_token=token,
            frame_idx=int(frame_idx),
            sample_index=int(sample_index),
            sample_relative_time_s=float(rel_times[sample_index]),
            hugsim_relative_time_s=float(target),
        )
