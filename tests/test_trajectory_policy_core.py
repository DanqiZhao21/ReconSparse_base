from __future__ import annotations

import torch
import pytest

from framework.algorithms.trajectory_policy_core import (
    compute_grpo_objective,
    compute_risk_decel_auxiliary_objective,
    compute_sac_objective,
)


def test_grpo_objective_uses_old_new_candidate_logp_ratio() -> None:
    new_log_probs = torch.log_softmax(torch.tensor([[1.2, 0.2, -0.4]], dtype=torch.float32), dim=1)
    old_log_probs = torch.log_softmax(torch.tensor([[0.7, 0.4, -0.2]], dtype=torch.float32), dim=1)
    scores = torch.tensor([[3.0, 1.0, 2.0]], dtype=torch.float32)

    loss = compute_grpo_objective(
        candidate_log_probs=new_log_probs,
        old_candidate_log_probs=old_log_probs,
        candidate_scores=scores,
        objective="grpo",
        clip_eps=0.2,
    )

    advantages = (scores - scores.mean(dim=1, keepdim=True)) / (
        scores.std(dim=1, keepdim=True, unbiased=False) + 1e-6
    )
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 0.8, 1.2) * advantages
    expected = -torch.min(unclipped, clipped).mean()
    approx_kl = ((ratio - 1.0) - log_ratio).mean()

    assert torch.allclose(loss.loss, expected)
    assert torch.allclose(loss.approx_kl, approx_kl)
    assert torch.allclose(loss.ratio_mean, ratio.mean())
    assert torch.allclose(loss.clip_frac, ((ratio - 1.0).abs() > 0.2).float().mean())


def test_grpo_objective_requires_old_candidate_log_probs() -> None:
    log_probs = torch.log_softmax(torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32), dim=1)
    scores = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="old_candidate_log_probs"):
        compute_grpo_objective(
            candidate_log_probs=log_probs,
            candidate_scores=scores,
            objective="grpo",
        )


@pytest.mark.parametrize("objective", ["logprob", "expected_prob", "clipped_ratio", "ppo_ratio", "strict_grpo", "craft"])
def test_grpo_objective_rejects_non_grpo_names(objective: str) -> None:
    log_probs = torch.log_softmax(torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32), dim=1)
    old_log_probs = torch.log_softmax(torch.tensor([[0.5, 0.1, -0.4]], dtype=torch.float32), dim=1)
    scores = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="Unsupported GRPO objective"):
        compute_grpo_objective(
            candidate_log_probs=log_probs,
            old_candidate_log_probs=old_log_probs,
            candidate_scores=scores,
            objective=objective,
        )


def test_risk_decel_auxiliary_loss_pushes_decel_modes_up_and_accel_modes_down() -> None:
    logits = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
    traj = torch.tensor(
        [[[[0.2, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.55, 0.0, 0.0]]]],
        dtype=torch.float32,
    )

    out = compute_risk_decel_auxiliary_objective(
        candidate_score_logits=logits,
        candidate_traj_xyyaw=traj,
        high_risk_mask=torch.tensor([True]),
        ego_speed_mps=torch.tensor([1.0], dtype=torch.float32),
        dt_s=0.5,
        speed_margin_mps=0.2,
    )

    out.loss.backward()

    assert out.active_count.item() == 1.0
    assert out.decel_prob_mean.item() > 0.0
    assert out.accel_prob_mean.item() > 0.0
    assert logits.grad is not None
    assert logits.grad[0, 0].item() < 0.0
    assert logits.grad[0, 1].item() > 0.0


def test_sac_objective_uses_closed_loop_advantage_and_entropy_term() -> None:
    new_logp = torch.tensor([-1.0, -0.5, -2.0], dtype=torch.float32)
    old_logp = torch.tensor([-1.2, -0.7, -1.5], dtype=torch.float32)
    adv = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float32)

    loss = compute_sac_objective(
        new_logp=new_logp,
        old_logp=old_logp,
        adv=adv,
        entropy_coef=0.05,
        kl_coef=0.1,
    )

    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    expected_pg = -(adv.detach() * new_logp).mean()
    expected_entropy = 0.05 * new_logp.mean()
    expected = expected_pg + expected_entropy + 0.1 * approx_kl

    assert torch.allclose(loss.loss, expected)
    assert torch.allclose(loss.loss_pi, expected)
    assert torch.allclose(loss.loss_pg, expected_pg)
    assert torch.allclose(loss.loss_entropy, expected_entropy)
    assert torch.allclose(loss.approx_kl, approx_kl)
    assert torch.allclose(loss.ratio_mean, ratio.mean())
