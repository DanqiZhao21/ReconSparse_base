from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CachedRewardModelDataset(Dataset):
    """Dataset for offline reward-model samples saved as torch dictionaries.

    Each item is expected to contain:
    - image_paths: list[str], ordered by frame then camera
    - ego_states: [E]
    - candidate_trajectories: [G,T,3]
    - targets: [G,H,8]
    - valid_mask: optional [G,H,8]
    """

    def __init__(self, root: str | Path, *, image_size: tuple[int, int] | None = None) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.paths = sorted([*self.root.glob("*.pt"), *self.root.glob("*.pth")])
        if not self.paths:
            raise FileNotFoundError(f"No cached reward-model samples found in {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = torch.load(self.paths[int(index)], map_location="cpu")
        if not isinstance(sample, dict):
            raise RuntimeError(f"Cached sample must be a dict: {self.paths[int(index)]}")
        if "image_paths" not in sample:
            raise KeyError(f"Cached reward-model sample must contain image_paths: {self.paths[int(index)]}")
        sample["observations"] = load_observation_from_image_paths(sample["image_paths"], image_size=self.image_size)
        if "valid_mask" not in sample and "targets" in sample:
            sample["valid_mask"] = torch.ones_like(torch.as_tensor(sample["targets"]), dtype=torch.bool)
        return sample


def load_observation_from_image_paths(
    image_paths: list[str] | tuple[str, ...],
    *,
    image_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    channels: list[torch.Tensor] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image_size is not None:
                height, width = int(image_size[0]), int(image_size[1])
                image = image.resize((width, height), Image.BILINEAR)
            arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        channels.append(tensor)
    if not channels:
        raise ValueError("image_paths must contain at least one image")
    return torch.cat(channels, dim=0)


def reward_model_collate(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    keys = ["observations", "ego_states", "candidate_trajectories", "targets", "valid_mask"]
    batch: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [torch.as_tensor(sample[key]) for sample in samples if key in sample]
        if values:
            batch[key] = torch.stack(values, dim=0)
    return batch
