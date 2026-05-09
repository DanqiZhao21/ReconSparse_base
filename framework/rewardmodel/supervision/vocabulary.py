from __future__ import annotations

import math

import numpy as np


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def filter_trajectory_vocabulary(
    vocabulary: np.ndarray,
    gt_trajectory: np.ndarray,
    *,
    max_longitudinal_error_m: float = 10.0,
    max_lateral_error_m: float = 5.0,
    max_heading_error_rad: float = math.radians(20.0),
    max_samples: int = 256,
) -> np.ndarray:
    vocab = np.asarray(vocabulary, dtype=np.float32)
    gt = np.asarray(gt_trajectory, dtype=np.float32)
    if vocab.ndim != 3 or vocab.shape[-1] < 3:
        raise ValueError(f"Expected vocabulary shape [N,T,3], got {tuple(vocab.shape)}")
    if gt.ndim != 2 or gt.shape[-1] < 3:
        raise ValueError(f"Expected gt_trajectory shape [T,3], got {tuple(gt.shape)}")

    gt_end = gt[-1, :3]
    end = vocab[:, -1, :3]
    longitudinal_error = np.abs(end[:, 0] - gt_end[0])
    lateral_error = np.abs(end[:, 1] - gt_end[1])
    heading_error = np.abs(_wrap_angle(end[:, 2] - gt_end[2]))

    keep = (
        (longitudinal_error <= float(max_longitudinal_error_m))
        & (lateral_error <= float(max_lateral_error_m))
        & (heading_error <= float(max_heading_error_rad))
    )
    filtered = vocab[keep]
    if int(filtered.shape[0]) <= int(max_samples):
        return filtered
    order = np.argsort(np.abs(filtered[:, -1, 1] - gt_end[1]))
    selected = np.linspace(0, len(order) - 1, num=int(max_samples), dtype=np.int64)
    return filtered[order[selected]]

