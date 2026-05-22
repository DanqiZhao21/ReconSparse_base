from __future__ import annotations

import torch
import pytest

from framework.algorithms.trajectory_policy_core import compute_grpo_objective


def test_grpo_logprob_objective_remains_default() -> None:
    log_probs = torch.log_softmax(torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32), dim=1)
    scores = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)

    loss = compute_grpo_objective(candidate_log_probs=log_probs, candidate_scores=scores)

    advantages = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True, unbiased=False) + 1e-6)
    expected = -(advantages.detach() * log_probs).mean()
    assert torch.allclose(loss.loss, expected)


def test_grpo_expected_prob_objective_uses_candidate_logits() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32)
    log_probs = torch.log_softmax(logits, dim=1)
    scores = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)

    loss = compute_grpo_objective(
        candidate_log_probs=log_probs,
        candidate_scores=scores,
        candidate_score_logits=logits,
        objective="expected_prob",
        temperature=1.0,
    )

    advantages = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True, unbiased=False) + 1e-6)
    expected = -(torch.softmax(logits, dim=1) * advantages.detach()).sum(dim=1).mean()
    assert torch.allclose(loss.loss, expected)


def test_grpo_objective_rejects_legacy_craft_alias() -> None:
    log_probs = torch.log_softmax(torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32), dim=1)
    scores = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="Unsupported GRPO objective"):
        compute_grpo_objective(
            candidate_log_probs=log_probs,
            candidate_scores=scores,
            objective="craft",
        )
