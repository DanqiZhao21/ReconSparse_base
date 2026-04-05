from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


CAM_KEYS = ["front_left", "front", "front_right", "back_left", "back", "back_right"]


def obs_to_tensor(
    observation: Dict[str, np.ndarray],
    device: torch.device,
    *,
    image_size: Tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """Convert env dict-of-images (uint8 HWC) to a (1, 18, H, W) float tensor."""
    imgs: list[np.ndarray] = []
    for k in CAM_KEYS:
        img = observation.get(k, None)
        if img is None:
            raise KeyError(f"Missing camera key in observation: {k}")
        imgs.append(img)
    arr = np.stack(imgs, axis=0)  # (6,H,W,3)
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    t = t.permute(0, 3, 1, 2) / 255.0  # (6,3,H,W)
    t = t.reshape(1, 18, t.shape[-2], t.shape[-1])
    if image_size is not None:
        t = F.interpolate(t, size=image_size, mode="bilinear", align_corners=False)
    return t

__all__ = ["CAM_KEYS", "obs_to_tensor"]
