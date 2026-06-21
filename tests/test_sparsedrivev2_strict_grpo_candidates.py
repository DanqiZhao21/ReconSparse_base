from __future__ import annotations

import torch

from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy


class _EvalOnlyModel:
    def eval(self) -> None:
        return None


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


def test_logp_from_replay_batch_uses_single_loop_for_multiple_replays(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device_override = "cpu"
    policy._model = _EvalOnlyModel()

    replays = [
        {
            "camera_feature": {"imgs": torch.zeros((1, 1), dtype=torch.float32)},
            "status_feature": torch.zeros((1, 1), dtype=torch.float32),
            "global_mode_idx": 11,
        },
        {
            "camera_feature": {"imgs": torch.ones((1, 1), dtype=torch.float32)},
            "status_feature": torch.ones((1, 1), dtype=torch.float32),
            "global_mode_idx": 21,
        },
    ]
    forward_batch_sizes: list[int] = []

    def fake_to_device(features, device):
        del device
        return features

    def fake_forward(features):
        status = features["status_feature"].view(-1).to(dtype=torch.long)
        forward_batch_sizes.append(int(status.numel()))
        all_scores = torch.tensor([[0.0, 3.0, 1.0], [2.0, -1.0, 4.0]], dtype=torch.float32)
        all_indices = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long)
        return {
            "candidate_scores": all_scores.index_select(0, status),
            "candidate_global_indices": all_indices.index_select(0, status),
        }

    monkeypatch.setattr(policy, "replay_is_compatible", lambda replay: True)
    monkeypatch.setattr(policy, "_to_device_features", fake_to_device)
    monkeypatch.setattr(policy, "_forward_policy", fake_forward)
    monkeypatch.setattr(
        policy,
        "_candidate_identity_from_outputs",
        lambda out: {"candidate_global_indices": out["candidate_global_indices"]},
    )

    logp = policy.logp_from_replay_batch(replays)

    expected = torch.log_softmax(
        torch.tensor([[0.0, 3.0, 1.0], [2.0, -1.0, 4.0]], dtype=torch.float32),
        dim=1,
    )[torch.arange(2), torch.tensor([1, 1])]
    assert forward_batch_sizes == [1, 1]
    assert torch.allclose(logp, expected)


def test_logp_from_replay_batch_falls_back_single_replays(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device_override = "cpu"
    policy._model = _EvalOnlyModel()

    replays = [
        {
            "camera_feature": {"imgs": torch.zeros((1, 1), dtype=torch.float32)},
            "status_feature": torch.zeros((1, 1), dtype=torch.float32),
            "global_mode_idx": 11,
        },
        {
            "camera_feature": {"imgs": torch.ones((1, 1), dtype=torch.float32)},
            "status_feature": torch.ones((1, 1), dtype=torch.float32),
            "global_mode_idx": 21,
        },
    ]
    forward_batch_sizes: list[int] = []

    def fake_to_device(features, device):
        del device
        return features

    def fake_forward(features):
        status = features["status_feature"].view(-1).to(dtype=torch.long)
        forward_batch_sizes.append(int(status.numel()))
        assert int(status.numel()) == 1
        if int(status.item()) == 0:
            return {
                "candidate_scores": torch.tensor([[0.0, 3.0, 1.0]], dtype=torch.float32),
                "candidate_global_indices": torch.tensor([[10, 11, 12]], dtype=torch.long),
            }
        return {
            "candidate_scores": torch.tensor([[2.0, -1.0, 4.0]], dtype=torch.float32),
            "candidate_global_indices": torch.tensor([[20, 21, 22]], dtype=torch.long),
        }

    monkeypatch.setattr(policy, "replay_is_compatible", lambda replay: True)
    monkeypatch.setattr(policy, "_to_device_features", fake_to_device)
    monkeypatch.setattr(policy, "_forward_policy", fake_forward)
    monkeypatch.setattr(
        policy,
        "_candidate_identity_from_outputs",
        lambda out: {"candidate_global_indices": out["candidate_global_indices"]},
    )

    logp = policy.logp_from_replay_batch(replays)

    expected = torch.log_softmax(
        torch.tensor([[0.0, 3.0, 1.0], [2.0, -1.0, 4.0]], dtype=torch.float32),
        dim=1,
    )[torch.arange(2), torch.tensor([1, 1])]
    assert forward_batch_sizes == [1, 1]
    assert torch.allclose(logp, expected)


def test_replay_policy_outputs_strict_grpo_uses_single_loop(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device_override = "cpu"
    policy._model = _EvalOnlyModel()

    replays = [
        {
            "camera_feature": {"imgs": torch.zeros((1, 1), dtype=torch.float32)},
            "status_feature": torch.zeros((1, 1), dtype=torch.float32),
            "global_mode_idx": 11,
            "grpo_candidate_mode_indices": torch.tensor([11, 12], dtype=torch.long),
            "grpo_candidate_old_log_probs": torch.tensor([-0.1, -0.2], dtype=torch.float32),
            "grpo_candidate_traj_xyyaw": torch.zeros((2, 3, 3), dtype=torch.float32),
        },
        {
            "camera_feature": {"imgs": torch.ones((1, 1), dtype=torch.float32)},
            "status_feature": torch.ones((1, 1), dtype=torch.float32),
            "global_mode_idx": 21,
            "grpo_candidate_mode_indices": torch.tensor([21, 22], dtype=torch.long),
            "grpo_candidate_old_log_probs": torch.tensor([-0.3, -0.4], dtype=torch.float32),
            "grpo_candidate_traj_xyyaw": torch.ones((2, 3, 3), dtype=torch.float32),
        },
    ]
    forward_batch_sizes: list[int] = []

    def fake_batched_features(batch_replays):
        return {
            "camera_feature": {"imgs": torch.cat([rep["camera_feature"]["imgs"] for rep in batch_replays], dim=0)},
            "status_feature": torch.cat([rep["status_feature"] for rep in batch_replays], dim=0),
        }

    def fake_forward_on_model(model, features, targets=None):
        del model, targets
        status = features["status_feature"].view(-1).to(dtype=torch.long)
        forward_batch_sizes.append(int(status.numel()))
        all_scores = torch.tensor([[0.0, 3.0, 1.0], [2.0, -1.0, 4.0]], dtype=torch.float32)
        all_indices = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long)
        return {
            "candidate_scores": all_scores.index_select(0, status),
            "candidate_global_indices": all_indices.index_select(0, status),
            "candidate_trajectories": torch.zeros((int(status.numel()), 3, 2, 3), dtype=torch.float32),
        }

    monkeypatch.setattr(policy, "_batched_replay_features", fake_batched_features)
    monkeypatch.setattr(policy, "_forward_policy_on_model", fake_forward_on_model)
    monkeypatch.setattr(
        policy,
        "_candidate_identity_from_outputs",
        lambda out: {"candidate_global_indices": out["candidate_global_indices"]},
    )

    outputs = policy.replay_policy_outputs_from_replay_batch(
        replays,
        num_candidates=2,
        candidate_select="topk",
    )

    expected_logp_all = torch.log_softmax(
        torch.tensor([[0.0, 3.0, 1.0], [2.0, -1.0, 4.0]], dtype=torch.float32),
        dim=1,
    )
    assert forward_batch_sizes == [1, 1]
    assert torch.allclose(outputs["new_logp"], expected_logp_all[torch.arange(2), torch.tensor([1, 1])])
    assert torch.allclose(outputs["counterfactual"]["log_probs"], torch.stack([expected_logp_all[0, [1, 2]], expected_logp_all[1, [1, 2]]]))
    assert torch.equal(outputs["counterfactual"]["score_logits"], torch.tensor([[3.0, 1.0], [-1.0, 4.0]]))
