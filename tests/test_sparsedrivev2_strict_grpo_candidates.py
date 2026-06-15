from __future__ import annotations

import torch

from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy


def test_select_old_policy_sampled_candidates_uses_multinomial_indices(monkeypatch) -> None:
    logits = torch.tensor([[3.0, 1.0, -1.0, 0.0]], dtype=torch.float32)
    trajs = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)
    global_indices = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    sampled_local = torch.tensor([[2, 0]], dtype=torch.long)
    calls: list[torch.Tensor] = []

    def fake_multinomial(probs, num_samples, replacement=False):
        del replacement
        calls.append(probs)
        assert num_samples == 2
        return sampled_local

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    selected = SparseDriveV2Policy._select_strict_grpo_old_policy_candidates(
        score_logits=logits,
        candidate_trajs=trajs,
        candidate_global_indices=global_indices,
        num_candidates=2,
    )

    expected_log_probs = torch.log_softmax(logits, dim=1).gather(1, sampled_local)
    assert calls
    assert torch.allclose(calls[0], torch.softmax(logits, dim=1))
    assert torch.equal(selected["mode_indices"], torch.tensor([[12, 10]], dtype=torch.long))
    assert torch.allclose(selected["old_log_probs"], expected_log_probs)
    assert torch.equal(selected["local_indices"], sampled_local)
    assert selected["traj_xyyaw"].shape == (1, 2, 2, 3)


def test_new_log_probs_for_stored_grpo_candidates_gathers_by_global_index() -> None:
    score_logits = torch.tensor([[1.0, 3.0, 0.0]], dtype=torch.float32)
    candidate_global_indices = torch.tensor([[20, 10, 30]], dtype=torch.long)
    replay = {"grpo_candidate_mode_indices": torch.tensor([10, 30], dtype=torch.long)}

    new_log_probs = SparseDriveV2Policy._new_log_probs_for_stored_grpo_candidates_from_outputs(
        score_logits=score_logits,
        candidate_global_indices=candidate_global_indices,
        replays=[replay],
    )

    expected = torch.log_softmax(score_logits, dim=1)[:, [1, 2]]
    assert torch.allclose(new_log_probs, expected)


def test_score_logits_for_stored_grpo_candidates_gathers_by_global_index() -> None:
    score_logits = torch.tensor([[1.0, 3.0, 0.0]], dtype=torch.float32)
    candidate_global_indices = torch.tensor([[20, 10, 30]], dtype=torch.long)
    replay = {"grpo_candidate_mode_indices": torch.tensor([10, 30], dtype=torch.long)}

    selected_logits = SparseDriveV2Policy._score_logits_for_stored_grpo_candidates_from_outputs(
        score_logits=score_logits,
        candidate_global_indices=candidate_global_indices,
        replays=[replay],
    )

    assert torch.equal(selected_logits, torch.tensor([[3.0, 0.0]], dtype=torch.float32))
