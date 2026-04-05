import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from framework.algorithms.trajectory_policy_core import (
    compute_ppo_metrics,
    compute_ppo_objective,
    compute_reinforce_metrics,
    compute_reinforce_objective,
)


def test_compute_ppo_objective_matches_expected_clipped_terms():
    objective = compute_ppo_objective(
        new_logp=torch.log(torch.tensor([1.1, 0.8], dtype=torch.float32)),
        old_logp=torch.zeros(2, dtype=torch.float32),
        adv=torch.tensor([1.0, -1.0], dtype=torch.float32),
        ret=torch.tensor([0.3, -0.1], dtype=torch.float32),
        value_pred=torch.tensor([0.2, -0.2], dtype=torch.float32),
        old_value=torch.tensor([0.1, -0.1], dtype=torch.float32),
        clip_eps=0.1,
        vf_coef=0.5,
        value_clip_eps=0.05,
        kl_coef=0.0,
        dual_clip=None,
    )

    metrics = compute_ppo_metrics(
        new_logp=torch.log(torch.tensor([1.1, 0.8], dtype=torch.float32)),
        old_logp=torch.zeros(2, dtype=torch.float32),
        adv=torch.tensor([1.0, -1.0], dtype=torch.float32),
        ret=torch.tensor([0.3, -0.1], dtype=torch.float32),
        value_pred=torch.tensor([0.2, -0.2], dtype=torch.float32),
        loss=objective,
    )

    assert objective.loss_pi.item() == pytest.approx(-0.1, abs=1e-5)
    assert objective.loss_v.item() == pytest.approx(0.01625, abs=1e-5)
    assert objective.clip_frac.item() == pytest.approx(1.0, abs=1e-5)
    assert objective.value_clip_frac.item() == pytest.approx(1.0, abs=1e-5)
    assert metrics["loss_pi"].item() == pytest.approx(objective.loss_pi.item(), abs=1e-6)
    assert "explained_variance" in metrics


def test_compute_reinforce_objective_supports_plain_policy_gradient_path():
    objective = compute_reinforce_objective(
        new_logp=torch.tensor([0.2, -0.4], dtype=torch.float32),
        old_logp=None,
        adv=torch.tensor([1.5, -0.5], dtype=torch.float32),
        clip_eps=0.2,
        kl_coef=0.0,
    )

    metrics = compute_reinforce_metrics(
        new_logp=torch.tensor([0.2, -0.4], dtype=torch.float32),
        old_logp=None,
        adv=torch.tensor([1.5, -0.5], dtype=torch.float32),
        loss=objective,
    )

    assert objective.loss_pi.item() == pytest.approx(-0.25, abs=1e-6)
    assert objective.approx_kl.item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["ratio_mean"].item() == pytest.approx(1.0, abs=1e-6)
