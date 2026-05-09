from __future__ import annotations

import numpy as np
import torch

from framework.rewardmodel.config import RewardSupervisionConfig
from framework.rewardmodel.supervision.pdm_teacher import normalize_teacher_scores
from framework.rewardmodel.supervision.vocabulary import filter_trajectory_vocabulary


def test_normalize_scalar_teacher_scores_to_full_metric_targets() -> None:
    cfg = RewardSupervisionConfig(num_horizons=4)
    scores = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    normalized = normalize_teacher_scores(scores, cfg)

    assert normalized.targets.shape == (1, 2, 4, 8)
    assert normalized.valid_mask.shape == (1, 2, 4, 8)
    assert torch.allclose(normalized.targets[0, 0], torch.full((4, 8), 0.25))
    assert torch.all(normalized.valid_mask)


def test_normalize_metric_dict_teacher_scores() -> None:
    cfg = RewardSupervisionConfig(num_horizons=2)
    scores = {
        "rnc": torch.ones((1, 2, 2), dtype=torch.float32),
        "rep": torch.full((1, 2, 2), 0.5, dtype=torch.float32),
    }

    normalized = normalize_teacher_scores(scores, cfg)

    assert normalized.targets.shape == (1, 2, 2, 8)
    assert torch.allclose(normalized.targets[..., 0], torch.ones((1, 2, 2)))
    assert torch.allclose(normalized.targets[..., 4], torch.full((1, 2, 2), 0.5))
    assert torch.all(normalized.valid_mask[..., 0])
    assert torch.all(normalized.valid_mask[..., 4])
    assert not torch.any(normalized.valid_mask[..., 1])


def test_filter_trajectory_vocabulary_keeps_near_gt_end_states() -> None:
    vocab = np.asarray(
        [
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [16.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 7.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 1.0, np.deg2rad(30.0)]],
        ],
        dtype=np.float32,
    )
    gt = np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)

    filtered = filter_trajectory_vocabulary(
        vocab,
        gt,
        max_longitudinal_error_m=10.0,
        max_lateral_error_m=5.0,
        max_heading_error_rad=np.deg2rad(20.0),
        max_samples=16,
    )

    assert filtered.shape == (1, 2, 3)
    assert np.allclose(filtered[0], vocab[0])

