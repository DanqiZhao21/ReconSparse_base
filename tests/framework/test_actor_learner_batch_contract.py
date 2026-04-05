import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from framework.batch import build_training_batch
from framework.batch.actor_learner import LoadedShardBatch, compute_gae


def test_batch_package_exports_canonical_builder():
    assert callable(build_training_batch)
    assert LoadedShardBatch is not None


def test_compute_gae_returns_expected_shapes():
    adv, ret = compute_gae(
        rewards=torch.tensor([1.0, 2.0], dtype=torch.float32),
        dones=torch.tensor([0.0, 1.0], dtype=torch.float32),
        values=torch.tensor([0.5, 0.25], dtype=torch.float32),
        last_value=torch.tensor(0.0, dtype=torch.float32),
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert adv.shape == torch.Size([2])
    assert ret.shape == torch.Size([2])
