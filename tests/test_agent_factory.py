from __future__ import annotations

import torch

from framework.runner.agent_factory import build_agent


def test_build_sparsedrive_v2_agent_forwards_frozen_prefixes(monkeypatch) -> None:
    captured = {}

    class FakeSparseDriveV2Policy:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "framework.agent.policy_sparsedrive_v2.SparseDriveV2Policy",
        FakeSparseDriveV2Policy,
    )

    build_agent(
        {
            "train": {"policy_lr": 1.0e-5},
            "agent": {
                "type": "sparsedrive_v2",
                "ckpt": "missing-but-not-loaded.ckpt",
                "trainable_prefixes": [],
                "frozen_prefixes": ["_backbone"],
            },
        },
        device=torch.device("cpu"),
    )

    assert captured["trainable_prefixes"] == []
    assert captured["frozen_prefixes"] == ["_backbone"]


def test_sparsedrive_v2_policy_uses_craft_carl_backend(monkeypatch) -> None:
    from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy

    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._nuscenes_scorer_config = {"backend": "craft_carl", "carl": {"w_prog": 8.0}}
    policy._nuscenes_pdm_scorer = None
    policy._nuscenes_craft_scorer = None

    class DummyCraftBackend:
        pass

    monkeypatch.setattr(
        "framework.algorithms.nuscenes_craft_scorer.NuScenesCraftScorer",
        lambda **kwargs: DummyCraftBackend(),
    )

    scorer = policy._ensure_counterfactual_scorer_backend()

    assert isinstance(scorer, DummyCraftBackend)
    assert policy._nuscenes_craft_scorer is scorer
