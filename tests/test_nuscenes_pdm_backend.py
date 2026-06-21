from __future__ import annotations

from pathlib import Path

import pickle
import types

import numpy as np
import pytest
import torch

from framework.agent.policy_sparsedrive_v2 import SparseDriveV2Policy, _apply_trainable_prefixes


def _write_token2vad(path: Path) -> None:
    payload = {
        "tok-a": {
            "token": "tok-a",
            "gt_ego_fut_trajs": np.asarray(
                [
                    [1.0, 3.0],
                    [1.2, 3.3],
                    [1.4, 3.7],
                ],
                dtype=np.float32,
            ),
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _fake_policy_model_with_traj_vocab(traj_vocab: torch.Tensor) -> torch.nn.Module:
    model = torch.nn.Module()
    model._trajectory_head = types.SimpleNamespace(traj_vocab=torch.nn.Parameter(traj_vocab, requires_grad=False))
    return model


def _structured_replay(
    *,
    idx: int = 0,
    global_mode_idx: int,
    selected_path_idx: int = 0,
    selected_vel_idx: int = 0,
) -> dict:
    return {
        "schema_version": 2,
        "policy": {
            "backend": "sparsedrive_v2",
            "schema_version": 1,
            "model_inputs": {
                "camera_feature": {
                    "imgs": torch.full((1, 2, 3, 4, 5), float(idx), dtype=torch.float32),
                    "lidar2img": torch.full((1, 2, 4, 4), float(idx), dtype=torch.float32),
                },
                "status_feature": torch.full((1, 8), float(idx), dtype=torch.float32),
            },
            "action_id": {"global_mode_idx": int(global_mode_idx)},
        },
        "debug": {
            "selected_path_idx": int(selected_path_idx),
            "selected_vel_idx": int(selected_vel_idx),
        },
    }


def test_apply_trainable_prefixes_can_freeze_backbone_while_training_other_modules() -> None:
    module = torch.nn.Module()
    module._backbone = torch.nn.Sequential(torch.nn.Linear(2, 2))
    module._status_encoding = torch.nn.Linear(2, 2)
    module._trajectory_head = torch.nn.Sequential(torch.nn.Linear(2, 2))

    total, trainable = _apply_trainable_prefixes(
        module,
        prefixes=[],
        frozen_prefixes=["_backbone"],
    )

    assert total == 6
    assert trainable == 4
    assert all(not param.requires_grad for param in module._backbone.parameters())
    assert all(param.requires_grad for param in module._status_encoding.parameters())
    assert all(param.requires_grad for param in module._trajectory_head.parameters())


def test_policy_uses_nuscenes_pdm_backend_when_configured(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._nuscenes_scorer_config = {"backend": "nuscenes_pdm"}
    policy._nuscenes_pdm_scorer = None

    class DummyBackend:
        pass

    monkeypatch.setattr(
        "framework.algorithms.nuscenes_pdm_scorer.NuScenesPDMScorer",
        lambda **kwargs: DummyBackend(),
    )

    scorer = policy._ensure_counterfactual_scorer_backend()

    assert isinstance(scorer, DummyBackend)
    assert policy._nuscenes_pdm_scorer is scorer


def test_policy_defaults_to_nuscenes_pdm_backend(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._nuscenes_scorer_config = {}
    policy._nuscenes_pdm_scorer = None

    class DummyBackend:
        pass

    monkeypatch.setattr(
        "framework.algorithms.nuscenes_pdm_scorer.NuScenesPDMScorer",
        lambda **kwargs: DummyBackend(),
    )

    scorer = policy._ensure_counterfactual_scorer_backend()

    assert isinstance(scorer, DummyBackend)
    assert policy._nuscenes_pdm_scorer is scorer


def test_policy_pdm_score_hook_accepts_numpy_backend_scores(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device = torch.device("cpu")

    class DummyBackend:
        def score(self, replays, traj_xyyaw):
            del replays, traj_xyyaw
            return np.asarray([[0.1, 0.2]], dtype=np.float32)

    monkeypatch.setattr(policy, "_ensure_counterfactual_scorer_backend", lambda: DummyBackend())
    monkeypatch.setattr(SparseDriveV2Policy, "device", property(lambda self: self._device))

    scores = policy.pdm_score_counterfactuals_from_replay_batch(
        [{"schema_version": 2, "grpo": {"scorer": {"sample_token": "tok-a"}}}],
        torch.zeros((1, 2, 3, 3), dtype=torch.float32),
    )

    assert torch.is_tensor(scores)
    assert scores.dtype == torch.float32
    assert tuple(scores.shape) == (1, 2)


def test_nuscenes_pdm_backend_accepts_shared_visualization_scorer_kwargs(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
        ea_gate_enabled=False,
        driving_direction_gate_enabled=False,
        center_dev_max_m=2.0,
        heading_dev_max_deg=90.0,
        off_global_route_threshold_m=3.0,
        carl={"dp_min": 0.0},
    )

    assert scorer._delegate.scene_cache_root == tmp_path / "scene_cache"


def test_sparsedrivev2_replay_sampling_forwards_observations_as_one_batch(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device = torch.device("cpu")
    policy._execute_mode = "first_step"
    traj_vocab = torch.arange(8 * 10 * 3 * 3, dtype=torch.float32).reshape(8, 10, 3, 3)
    policy._model = _fake_policy_model_with_traj_vocab(traj_vocab)
    forward_batch_sizes = []

    monkeypatch.setattr(SparseDriveV2Policy, "device", property(lambda self: self._device))

    def fake_build_features(observation):
        idx = int(observation["idx"])
        return {
            "camera_feature": {
                "imgs": torch.full((1, 2, 3, 4, 5), float(idx), dtype=torch.float32),
                "lidar2img": torch.full((1, 2, 4, 4), float(idx), dtype=torch.float32),
            },
            "status_feature": torch.full((1, 8), float(idx), dtype=torch.float32),
            "feature_missing_fields": [],
        }

    def fake_forward_policy(features_dev):
        forward_batch_sizes.append(int(features_dev["status_feature"].shape[0]))
        batch_size = int(features_dev["status_feature"].shape[0])
        scores = torch.arange(batch_size * 2, dtype=torch.float32).reshape(batch_size, 2)
        trajs = torch.stack([traj_vocab[4, 2], traj_vocab[7, 5]], dim=0)[None].expand(batch_size, -1, -1, -1).clone()
        return {
            "candidate_scores": scores,
            "candidate_trajectories": trajs,
        }

    monkeypatch.setattr(policy, "_build_features", fake_build_features)
    monkeypatch.setattr(policy, "_forward_policy", fake_forward_policy)

    actions, logps, replays = policy.sample_sparsedrivev2_with_replay_batch(
        [{"idx": 0}, {"idx": 1}, {"idx": 2}],
        mode_select="greedy",
    )

    assert forward_batch_sizes == [3]
    assert len(actions) == 3
    assert len(logps) == 3
    assert len(replays) == 3
    assert all("mode_idx" not in rep for rep in replays)
    assert all("global_mode_idx" not in rep for rep in replays)
    assert [int(rep["debug"]["selected_path_idx"]) for rep in replays] == [7, 7, 7]
    assert [int(rep["debug"]["selected_vel_idx"]) for rep in replays] == [5, 5, 5]
    assert [int(rep["policy"]["action_id"]["global_mode_idx"]) for rep in replays] == [75, 75, 75]
    assert [tuple(rep["policy"]["model_inputs"]["status_feature"].shape) for rep in replays] == [(1, 8), (1, 8), (1, 8)]
    assert [tuple(rep["env"]["plan_xyyaw"].shape) for rep in replays] == [(3, 3), (3, 3), (3, 3)]


def test_sparsedrivev2_logp_from_replay_uses_global_mode_identity(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device = torch.device("cpu")
    traj_vocab = torch.arange(4 * 10 * 4 * 3, dtype=torch.float32).reshape(4, 10, 4, 3)
    policy._model = _fake_policy_model_with_traj_vocab(traj_vocab)

    monkeypatch.setattr(SparseDriveV2Policy, "device", property(lambda self: self._device))

    replays = [
        _structured_replay(idx=idx, selected_path_idx=idx + 1, selected_vel_idx=idx + 2, global_mode_idx=[20, 11][idx])
        for idx in range(2)
    ]

    def fake_forward_policy_on_model(model, features_dev, targets=None):
        del model, targets
        sample_indices = features_dev["status_feature"][:, 0].to(dtype=torch.long)
        all_scores = torch.tensor(
            [[0.1, 0.4, 0.2], [0.9, 0.3, 0.7]],
            dtype=torch.float32,
        )
        all_trajs = torch.stack(
            [
                torch.stack([traj_vocab[9 // 10, 9 % 10], traj_vocab[20 // 10, 20 % 10], traj_vocab[31 // 10, 31 % 10]], dim=0),
                torch.stack([traj_vocab[11 // 10, 11 % 10], traj_vocab[22 // 10, 22 % 10], traj_vocab[33 // 10, 33 % 10]], dim=0),
            ],
            dim=0,
        )
        return {
            "candidate_scores": all_scores[sample_indices],
            "candidate_trajectories": all_trajs[sample_indices],
        }

    monkeypatch.setattr(policy, "_forward_policy_on_model", fake_forward_policy_on_model)

    expected = torch.stack(
        [
            torch.log_softmax(torch.tensor([0.1, 0.4, 0.2]), dim=0)[1],
            torch.log_softmax(torch.tensor([0.9, 0.3, 0.7]), dim=0)[0],
        ]
    )
    assert torch.allclose(policy.logp_from_replay_batch(replays), expected)


def test_sparsedrivev2_logp_from_replay_requires_selected_global_mode_in_candidates(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device = torch.device("cpu")
    traj_vocab = torch.arange(3 * 10 * 4 * 3, dtype=torch.float32).reshape(3, 10, 4, 3)
    policy._model = _fake_policy_model_with_traj_vocab(traj_vocab)

    monkeypatch.setattr(SparseDriveV2Policy, "device", property(lambda self: self._device))

    replays = [
        _structured_replay(idx=0, selected_path_idx=2, selected_vel_idx=3, global_mode_idx=23)
    ]

    def fake_forward_policy_on_model(model, features_dev, targets=None):
        del model, features_dev, targets
        return {
            "candidate_scores": torch.tensor([[0.1, 0.6]], dtype=torch.float32),
            "candidate_trajectories": torch.stack([traj_vocab[1, 0], traj_vocab[1, 1]], dim=0).unsqueeze(0),
        }

    monkeypatch.setattr(policy, "_forward_policy_on_model", fake_forward_policy_on_model)

    with pytest.raises(RuntimeError, match="global_mode_idx was not present"):
        policy.logp_from_replay_batch(replays)


def test_sparsedrivev2_logp_from_replay_forces_selected_global_mode_into_candidates(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device = torch.device("cpu")
    traj_vocab = torch.arange(3 * 10 * 4 * 3, dtype=torch.float32).reshape(3, 10, 4, 3)
    policy._model = _fake_policy_model_with_traj_vocab(traj_vocab)
    seen_forced: list[list[int]] = []

    monkeypatch.setattr(SparseDriveV2Policy, "device", property(lambda self: self._device))

    replays = [
        _structured_replay(idx=0, selected_path_idx=2, selected_vel_idx=3, global_mode_idx=23)
    ]

    def fake_forward_policy_on_model(model, features_dev, targets=None):
        del model, features_dev
        if targets is None:
            return {
                "candidate_scores": torch.tensor([[0.1, 0.6]], dtype=torch.float32),
                "candidate_trajectories": torch.stack([traj_vocab[1, 0], traj_vocab[1, 1]], dim=0).unsqueeze(0),
            }
        forced = targets["forced_global_indices"].detach().cpu().tolist()
        seen_forced.append([int(v) for v in forced])
        return {
            "candidate_scores": torch.tensor([[0.1, 0.6, 0.2]], dtype=torch.float32),
            "candidate_trajectories": torch.stack(
                [traj_vocab[1, 0], traj_vocab[1, 1], traj_vocab[2, 3]],
                dim=0,
            ).unsqueeze(0),
        }

    monkeypatch.setattr(policy, "_forward_policy_on_model", fake_forward_policy_on_model)

    expected = torch.log_softmax(torch.tensor([0.1, 0.6, 0.2]), dim=0)[2]

    assert torch.allclose(policy.logp_from_replay_batch(replays), expected.view(1))
    assert seen_forced == [[23]]


def test_sparsedrivev2_logp_from_replay_recomputes_each_sample_independently(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device = torch.device("cpu")
    traj_vocab = torch.arange(4 * 10 * 4 * 3, dtype=torch.float32).reshape(4, 10, 4, 3)
    policy._model = _fake_policy_model_with_traj_vocab(traj_vocab)
    forward_batch_sizes: list[int] = []

    monkeypatch.setattr(SparseDriveV2Policy, "device", property(lambda self: self._device))

    replays = [
        _structured_replay(idx=idx, selected_path_idx=idx + 1, selected_vel_idx=idx + 2, global_mode_idx=[20, 11][idx])
        for idx in range(2)
    ]

    def fake_forward_policy_on_model(model, features_dev, targets=None):
        del model, targets
        batch_size = int(features_dev["status_feature"].shape[0])
        forward_batch_sizes.append(batch_size)
        if batch_size == 2:
            return {
                "candidate_scores": torch.tensor([[0.1, 0.4], [0.9, 0.3]], dtype=torch.float32),
                "candidate_trajectories": torch.stack(
                    [
                        torch.stack([traj_vocab[0, 9], traj_vocab[3, 1]], dim=0),
                        torch.stack([traj_vocab[1, 1], traj_vocab[2, 2]], dim=0),
                    ],
                    dim=0,
                ),
            }

        sample_idx = int(features_dev["status_feature"][0, 0].item())
        if sample_idx == 0:
            scores = torch.tensor([[0.1, 0.4]], dtype=torch.float32)
            trajs = torch.stack([traj_vocab[0, 9], traj_vocab[2, 0]], dim=0).unsqueeze(0)
        else:
            scores = torch.tensor([[0.9, 0.3]], dtype=torch.float32)
            trajs = torch.stack([traj_vocab[1, 1], traj_vocab[2, 2]], dim=0).unsqueeze(0)
        return {"candidate_scores": scores, "candidate_trajectories": trajs}

    monkeypatch.setattr(policy, "_forward_policy_on_model", fake_forward_policy_on_model)

    expected = torch.stack(
        [
            torch.log_softmax(torch.tensor([0.1, 0.4]), dim=0)[1],
            torch.log_softmax(torch.tensor([0.9, 0.3]), dim=0)[0],
        ]
    )

    assert torch.allclose(policy.logp_from_replay_batch(replays), expected)
    assert forward_batch_sizes == [1, 1]


def test_sparsedrivev2_fused_replay_policy_outputs_match_existing_hooks(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device = torch.device("cpu")
    traj_vocab = torch.arange(4 * 10 * 4 * 3, dtype=torch.float32).reshape(4, 10, 4, 3)
    policy._model = _fake_policy_model_with_traj_vocab(traj_vocab)
    forward_batch_sizes: list[int] = []

    monkeypatch.setattr(SparseDriveV2Policy, "device", property(lambda self: self._device))

    replays = [
        _structured_replay(idx=idx, selected_path_idx=idx + 1, selected_vel_idx=idx + 2, global_mode_idx=[20, 11][idx])
        for idx in range(2)
    ]

    def fake_forward_policy_on_model(model, features_dev, targets=None):
        del model, targets
        forward_batch_sizes.append(int(features_dev["status_feature"].shape[0]))
        sample_indices = features_dev["status_feature"][:, 0].to(dtype=torch.long)
        all_scores = torch.tensor(
            [[0.1, 0.4, 0.2], [0.9, 0.3, 0.7]],
            dtype=torch.float32,
        )
        all_trajs = torch.stack(
            [
                torch.stack([traj_vocab[9 // 10, 9 % 10], traj_vocab[20 // 10, 20 % 10], traj_vocab[31 // 10, 31 % 10]], dim=0),
                torch.stack([traj_vocab[11 // 10, 11 % 10], traj_vocab[22 // 10, 22 % 10], traj_vocab[33 // 10, 33 % 10]], dim=0),
            ],
            dim=0,
        )
        return {
            "candidate_scores": all_scores[sample_indices],
            "candidate_trajectories": all_trajs[sample_indices],
        }

    monkeypatch.setattr(policy, "_forward_policy_on_model", fake_forward_policy_on_model)

    fused = policy.replay_policy_outputs_from_replay_batch(
        replays,
        num_candidates=2,
        candidate_select="topk",
    )

    assert forward_batch_sizes == [1, 1]

    forward_batch_sizes.clear()
    expected_logp = policy.logp_from_replay_batch(replays)
    expected_candidates = policy.sample_counterfactual_trajectories_from_replay_batch(
        replays,
        num_candidates=2,
        candidate_select="topk",
    )

    assert forward_batch_sizes == [1, 1, 1, 1]
    assert torch.allclose(fused["new_logp"], expected_logp)
    assert torch.allclose(fused["counterfactual"]["log_probs"], expected_candidates["log_probs"])
    assert torch.equal(fused["counterfactual"]["mode_indices"], expected_candidates["mode_indices"])
    assert torch.allclose(fused["counterfactual"]["traj_xyyaw"], expected_candidates["traj_xyyaw"])


def test_sparsedrivev2_counterfactual_candidates_recompute_each_sample_independently(monkeypatch) -> None:
    policy = SparseDriveV2Policy.__new__(SparseDriveV2Policy)
    policy._device = torch.device("cpu")
    traj_vocab = torch.arange(4 * 10 * 4 * 3, dtype=torch.float32).reshape(4, 10, 4, 3)
    policy._model = _fake_policy_model_with_traj_vocab(traj_vocab)
    forward_batch_sizes: list[int] = []

    monkeypatch.setattr(SparseDriveV2Policy, "device", property(lambda self: self._device))

    replays = [
        _structured_replay(idx=idx, selected_path_idx=idx + 1, selected_vel_idx=idx + 2, global_mode_idx=[20, 11][idx])
        for idx in range(2)
    ]

    def fake_forward_policy_on_model(model, features_dev, targets=None):
        del model, targets
        forward_batch_sizes.append(int(features_dev["status_feature"].shape[0]))
        sample_idx = int(features_dev["status_feature"][0, 0].item())
        if sample_idx == 0:
            scores = torch.tensor([[0.1, 0.4, 0.2]], dtype=torch.float32)
            trajs = torch.stack([traj_vocab[0, 9], traj_vocab[2, 0], traj_vocab[3, 1]], dim=0).unsqueeze(0)
        else:
            scores = torch.tensor([[0.9, 0.3, 0.7]], dtype=torch.float32)
            trajs = torch.stack([traj_vocab[1, 1], traj_vocab[2, 2], traj_vocab[3, 3]], dim=0).unsqueeze(0)
        return {"candidate_scores": scores, "candidate_trajectories": trajs}

    monkeypatch.setattr(policy, "_forward_policy_on_model", fake_forward_policy_on_model)

    candidates = policy.sample_counterfactual_trajectories_from_replay_batch(
        replays,
        num_candidates=2,
        candidate_select="topk",
    )

    assert forward_batch_sizes == [1, 1]
    assert candidates["log_probs"].shape == (2, 2)
    assert torch.equal(candidates["mode_indices"], torch.tensor([[1, 2], [0, 2]], dtype=torch.long))


def test_nuscenes_pdm_backend_returns_batch_candidate_scores(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.7, 0.4, 0.0]],
                [[0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.4, 0.3, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    scores = scorer.score([{"sample_token": "tok-a"}], traj_xyyaw)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1, 2)


def test_nuscenes_pdm_backend_drivable_area_only_score_ignores_other_terms(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path, score_mode="drivable_area_only")

    def fake_map_metrics(**kwargs):
        del kwargs
        return {
            "drivable_area": np.asarray([1.0, 0.0], dtype=np.float32),
            "lane_keeping": np.asarray([0.0, 1.0], dtype=np.float32),
            "driving_direction": np.asarray([0.0, 1.0], dtype=np.float32),
        }

    monkeypatch.setattr(scorer, "_batch_map_metrics", fake_map_metrics)
    monkeypatch.setattr(
        scorer,
        "_batch_collision_ttc_metrics",
        lambda **kwargs: {
            "no_collision": np.asarray([0.0, 1.0], dtype=np.float32),
            "ttc": np.asarray([0.0, 1.0], dtype=np.float32),
        },
    )

    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.drivable_polygons = [
        np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    ]
    static_ctx = dict(sample_context.static_context)
    candidate_geometry = scorer._build_candidate_geometry_batch(
        torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    )
    sample_scores = scorer._score_candidate_batch_for_sample(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in candidate_geometry.items()},
        gt_xy_cmp=np.asarray(static_ctx["gt_xy"], dtype=np.float32),
        gt_yaw_cmp=np.asarray(static_ctx["gt_yaw"], dtype=np.float32),
        gt_xy_full=np.asarray(static_ctx["gt_xy"], dtype=np.float32),
        gt_s_full=np.asarray(static_ctx["gt_s"], dtype=np.float32),
        gt_total_len=float(static_ctx["gt_total_len"]),
        dt_s=0.5,
    )

    assert np.allclose(sample_scores, np.asarray([1.0, 0.0], dtype=np.float32))


def test_nuscenes_pdm_backend_dac_weight_scores_drivable_area_as_weighted_term(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        progress_weight=0.0,
        ttc_weight=5.0,
        lane_keeping_weight=0.0,
        history_comfort_weight=0.0,
        dac_weight=5.0,
        dac_gate_enabled=False,
        driving_direction_gate_enabled=False,
    )

    monkeypatch.setattr(
        scorer,
        "_batch_map_metrics",
        lambda **kwargs: {
            "drivable_area": np.asarray([1.0, 0.0], dtype=np.float32),
            "lane_keeping": np.asarray([1.0, 1.0], dtype=np.float32),
            "driving_direction": np.asarray([1.0, 1.0], dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        scorer,
        "_batch_collision_ttc_metrics",
        lambda **kwargs: {
            "no_collision": np.asarray([1.0, 1.0], dtype=np.float32),
            "ttc": np.asarray([1.0, 1.0], dtype=np.float32),
        },
    )

    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.drivable_polygons = [
        np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    ]
    static_ctx = dict(sample_context.static_context)
    candidate_geometry = scorer._build_candidate_geometry_batch(
        torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    )
    sample_scores = scorer._score_candidate_batch_for_sample(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in candidate_geometry.items()},
        gt_xy_cmp=np.asarray(static_ctx["gt_xy"], dtype=np.float32),
        gt_yaw_cmp=np.asarray(static_ctx["gt_yaw"], dtype=np.float32),
        gt_xy_full=np.asarray(static_ctx["gt_xy"], dtype=np.float32),
        gt_s_full=np.asarray(static_ctx["gt_s"], dtype=np.float32),
        gt_total_len=float(static_ctx["gt_total_len"]),
        dt_s=0.5,
    )

    assert np.allclose(sample_scores, np.asarray([1.0, 0.5], dtype=np.float32))


def test_nuscenes_pdm_backend_builds_candidate_geometry_in_candidate_batch(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.1], [0.7, 0.4, 0.2]],
                [[0.1, 0.0, -0.1], [0.2, 0.1, 0.0], [0.4, 0.3, 0.1]],
            ]
        ],
        dtype=torch.float32,
    )

    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)

    assert geometry["centers_xy"].shape == (1, 2, 3, 2)
    assert geometry["yaw_rad"].shape == (1, 2, 3)
    assert geometry["corners_xy"].shape == (1, 2, 3, 4, 2)
    assert geometry["polygons"].shape == (1, 2, 3)
    assert geometry["polygons"].dtype == object


def test_nuscenes_pdm_drivable_map_contains_points_in_candidate_batch() -> None:
    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMDrivableMap

    drivable_map = NuScenesPDMDrivableMap(
        polygons_xy=[
            np.asarray([[-1.0, -1.0], [3.0, -1.0], [3.0, 3.0], [-1.0, 3.0]], dtype=np.float32),
            np.asarray([[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0]], dtype=np.float32),
        ]
    )
    points_xy = np.asarray(
        [
            [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]],
            [[10.5, 10.5], [0.5, 0.5], [20.0, 20.0]],
        ],
        dtype=np.float32,
    )

    inside = drivable_map.batch_contains_points(points_xy)

    assert inside.shape == (2, 3)
    assert inside.dtype == bool
    assert inside.tolist() == [[True, True, False], [True, True, False]]


def test_nuscenes_pdm_backend_builds_ttc_projection_geometry_in_candidate_batch(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    centers_xy = np.asarray(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    yaw_rad = np.zeros((2, 3), dtype=np.float32)

    projection = scorer._build_ttc_projection_geometry(
        centers_xy=centers_xy,
        yaw_rad=yaw_rad,
        dt_s=0.5,
    )

    assert projection["centers_xy"].shape[:3] == (2, 3, len(projection["offsets_s"]))
    assert projection["corners_xy"].shape == (2, 3, len(projection["offsets_s"]), 4, 2)
    assert projection["polygons"].shape == (2, 3, len(projection["offsets_s"]))
    assert projection["polygons"].dtype == object


def test_nuscenes_pdm_backend_query_hits_accepts_batched_polygons(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    scene_objects = [
        {
            "token": "obj-a",
            "corners_xy": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            "velocity_xy": [0.0, 0.0],
        }
    ]
    _, _, _, occupancy_map = scorer._build_object_geometry_arrays(scene_objects)
    polygons_at_step = scorer._build_candidate_geometry_batch(
        torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
                    [[3.0, 3.0, 0.0], [4.0, 4.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        )
    )["polygons"][0]

    hits = scorer._query_hits_per_candidate(occupancy_map, polygons_at_step[:, 0], predicate="intersects")

    assert hits.shape == (2,)
    assert hits.dtype == bool
    assert hits.tolist() == [True, False]


def test_nuscenes_pdm_backend_builds_sample_context_once_per_sample(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    calls = {"build_sample_context": 0}
    orig_build = scorer._build_sample_context
    delegate_calls = {"static_ctx": 0}

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        delegate_calls["static_ctx"] += 1
        return {
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [],
            "ea_agent_states": [],
        }

    def wrapped_build(replay, *, patch_radius: float):
        calls["build_sample_context"] += 1
        return orig_build(replay, patch_radius=patch_radius)

    monkeypatch.setattr(scorer, "_build_sample_context", wrapped_build)
    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)
    monkeypatch.setattr(
        scorer._delegate,
        "score",
        lambda replays, traj_xyyaw: np.zeros((len(replays), int(traj_xyyaw.shape[1])), dtype=np.float32),
    )

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.7, 0.4, 0.0]],
                [[0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.4, 0.3, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    replays = [{"sample_token": "tok-a"}]

    scores_0 = scorer.score(replays, traj_xyyaw)
    scores_1 = scorer.score(replays, traj_xyyaw)

    assert scores_0.shape == (1, 2)
    assert np.allclose(scores_0, scores_1)
    assert calls["build_sample_context"] == 2
    assert delegate_calls["static_ctx"] == 1


def test_nuscenes_pdm_backend_reuses_persisted_derived_context_across_instances(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    scene_cache_root = tmp_path / "scene_cache"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer_first = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root,
    )

    delegate_calls = {"static_ctx": 0}

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        delegate_calls["static_ctx"] += 1
        return {
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {
                    "drivable_area": [
                        [[-1.0, -1.0], [3.0, -1.0], [3.0, 3.0], [-1.0, 3.0]],
                    ],
                    "lane_centerline": [
                        [[0.0, 0.0], [2.0, 0.0]],
                    ],
                },
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [1.0, 2.0],
                    "velocity_xy": [0.2, 0.0],
                    "yaw_rad": 0.1,
                    "length_m": 4.5,
                    "width_m": 1.8,
                },
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer_first._delegate, "_build_static_sample_context", fake_static_context)

    ctx0 = scorer_first._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    scorer_second = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root,
    )

    def should_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("expected second scorer to reuse persisted derived pdm context")

    monkeypatch.setattr(scorer_second._delegate, "_build_static_sample_context", should_not_run)

    ctx1 = scorer_second._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    assert delegate_calls["static_ctx"] == 1
    assert list(ctx0.object_tokens) == list(ctx1.object_tokens)
    assert ctx1.centerline_segments_xy.shape == (1, 2, 2)
    assert ctx1.drivable_map.batch_contains_points(np.zeros((1, 1, 2), dtype=np.float32)).shape == (1, 1)


def test_nuscenes_pdm_backend_builds_sample_occupancy_context_once_per_sample(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    delegate_calls = {"static_ctx": 0}

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        delegate_calls["static_ctx"] += 1
        return {
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [1.0, 2.0],
                    "velocity_xy": [0.2, 0.0],
                    "yaw_rad": 0.1,
                    "length_m": 4.5,
                    "width_m": 1.8,
                },
                {
                    "category": "vehicle.bus",
                    "center_xy": [4.0, -1.5],
                    "velocity_xy": [0.0, -0.1],
                    "yaw_rad": -0.2,
                    "length_m": 10.0,
                    "width_m": 2.5,
                },
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    replay = {"sample_token": "tok-a"}
    ctx0 = scorer._build_sample_context(replay, patch_radius=20.0)
    ctx1 = scorer._build_sample_context(replay, patch_radius=20.0)

    assert ctx0 is ctx1
    assert delegate_calls["static_ctx"] == 1
    assert list(ctx0.object_tokens) == ["obj-0", "obj-1"]
    assert ctx0.object_velocity_xy.shape == (2, 2)
    assert ctx0.object_polygons.shape == (2,)
    assert len(ctx0.occupancy_map) == 2
    hits = ctx0.occupancy_map.intersects(ctx0.object_polygons[0])
    assert "obj-0" in hits


def test_nuscenes_pdm_backend_builds_future_tracks_from_scene_cache_tokens(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    scene_dir = tmp_path / "scene_cache" / "007"
    scene_dir.mkdir(parents=True)
    env_cache = {
        "meta": {},
        "0": {
            "ego_pose": {"x": 10.0, "y": 20.0, "yaw": 0.0},
            "dynamic_objects": [
                {
                    "token": "veh-a",
                    "category": "vehicle.car",
                    "poly": [[11.0, 20.5], [11.0, 19.5], [9.0, 19.5], [9.0, 20.5]],
                }
            ],
        },
        "5": {
            "ego_pose": {"x": 10.0, "y": 20.0, "yaw": 0.0},
            "dynamic_objects": [
                {
                    "token": "veh-a",
                    "category": "vehicle.car",
                    "poly": [[12.0, 20.5], [12.0, 19.5], [10.0, 19.5], [10.0, 20.5]],
                }
            ],
        },
        "10": {
            "ego_pose": {"x": 10.0, "y": 20.0, "yaw": 0.0},
            "dynamic_objects": [
                {
                    "token": "veh-a",
                    "category": "vehicle.car",
                    "poly": [[13.0, 20.5], [13.0, 19.5], [11.0, 19.5], [11.0, 20.5]],
                }
            ],
        },
    }
    import json

    (scene_dir / "env_cache.json").write_text(json.dumps(env_cache), encoding="utf-8")

    from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils

    scorer = NuScenesScorerUtils(token2vad_path=token2vad_path, scene_cache_root=tmp_path / "scene_cache")
    snapshot = scorer._lookup_scene_snapshot({"scene_id": 7, "frame_idx": 0})
    assert isinstance(snapshot, dict)

    objects = scorer._collect_scene_cache_dynamic_objects_with_future(
        replay={"scene_id": 7, "frame_idx": 0},
        snapshot=snapshot,
        patch_radius=20.0,
        future_horizon=3,
        future_step_frames=5,
        future_dt_s=0.5,
    )

    assert len(objects) == 1
    assert objects[0]["token"] == "veh-a"
    assert np.asarray(objects[0]["center_xy"], dtype=np.float32).tolist() == pytest.approx([0.0, 0.0])
    assert np.allclose(
        np.asarray(objects[0]["future_xy"], dtype=np.float32),
        np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )
    assert np.asarray(objects[0]["future_mask"], dtype=np.float32).tolist() == pytest.approx([1.0, 1.0])
    assert objects[0]["future_dt_s"] == pytest.approx(0.5)


def test_nuscenes_pdm_backend_builds_future_tracks_from_scene_cache_nearest_when_tokens_change(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    scene_dir = tmp_path / "scene_cache" / "007"
    scene_dir.mkdir(parents=True)
    env_cache = {
        "meta": {},
        "0": {
            "ego_pose": {"x": 10.0, "y": 20.0, "yaw": 0.0},
            "dynamic_objects": [
                {
                    "token": "veh-a-0",
                    "category": "vehicle.car",
                    "poly": [[11.0, 20.5], [11.0, 19.5], [9.0, 19.5], [9.0, 20.5]],
                }
            ],
        },
        "5": {
            "ego_pose": {"x": 10.0, "y": 20.0, "yaw": 0.0},
            "dynamic_objects": [
                {
                    "token": "veh-a-1",
                    "category": "vehicle.car",
                    "poly": [[12.0, 20.5], [12.0, 19.5], [10.0, 19.5], [10.0, 20.5]],
                },
                {
                    "token": "far-car",
                    "category": "vehicle.car",
                    "poly": [[30.0, 20.5], [30.0, 19.5], [28.0, 19.5], [28.0, 20.5]],
                },
            ],
        },
        "10": {
            "ego_pose": {"x": 10.0, "y": 20.0, "yaw": 0.0},
            "dynamic_objects": [
                {
                    "token": "veh-a-2",
                    "category": "vehicle.car",
                    "poly": [[13.0, 20.5], [13.0, 19.5], [11.0, 19.5], [11.0, 20.5]],
                }
            ],
        },
    }
    import json

    (scene_dir / "env_cache.json").write_text(json.dumps(env_cache), encoding="utf-8")

    from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils

    scorer = NuScenesScorerUtils(token2vad_path=token2vad_path, scene_cache_root=tmp_path / "scene_cache")
    snapshot = scorer._lookup_scene_snapshot({"scene_id": 7, "frame_idx": 0})
    assert isinstance(snapshot, dict)

    objects = scorer._collect_scene_cache_dynamic_objects_with_future(
        replay={"scene_id": 7, "frame_idx": 0},
        snapshot=snapshot,
        patch_radius=20.0,
        future_horizon=3,
        future_step_frames=5,
        future_dt_s=0.5,
    )

    assert len(objects) == 1
    assert np.allclose(
        np.asarray(objects[0]["future_xy"], dtype=np.float32),
        np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )


def test_nuscenes_pdm_backend_static_context_uses_scene_cache_future_tracks(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    scene_dir = tmp_path / "scene_cache" / "007"
    scene_dir.mkdir(parents=True)
    env_cache = {
        "meta": {},
        "0": {
            "ego_pose": {"x": 10.0, "y": 20.0, "yaw": 0.0},
            "dynamic_objects": [
                {
                    "token": "veh-a",
                    "category": "vehicle.car",
                    "poly": [[11.0, 20.5], [11.0, 19.5], [9.0, 19.5], [9.0, 20.5]],
                }
            ],
        },
        "5": {
            "ego_pose": {"x": 10.0, "y": 20.0, "yaw": 0.0},
            "dynamic_objects": [
                {
                    "token": "veh-a",
                    "category": "vehicle.car",
                    "poly": [[12.0, 20.5], [12.0, 19.5], [10.0, 19.5], [10.0, 20.5]],
                }
            ],
        },
    }
    import json

    (scene_dir / "env_cache.json").write_text(json.dumps(env_cache), encoding="utf-8")

    from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils

    scorer = NuScenesScorerUtils(token2vad_path=token2vad_path, scene_cache_root=tmp_path / "scene_cache")
    context = scorer._build_static_sample_context(
        {"sample_token": "tok-a", "scene_id": 7, "frame_idx": 0},
        patch_radius=20.0,
    )

    assert [obj["token"] for obj in context["scene_objects"]] == ["veh-a"]
    assert np.allclose(
        np.asarray(context["scene_objects"][0]["future_xy"], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    assert np.allclose(
        np.asarray(context["ttc_agent_states"][0]["future_xy"], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )


def test_nuscenes_pdm_backend_ea_gate_falls_back_to_scene_cache_future_tracks(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    scene_dir = tmp_path / "scene_cache" / "007"
    scene_dir.mkdir(parents=True)
    env_cache = {
        "meta": {},
        "0": {
            "ego_pose": {"x": 10.0, "y": 20.0, "yaw": 0.0},
            "dynamic_objects": [
                {
                    "token": "veh-a",
                    "category": "vehicle.car",
                    "poly": [[11.0, 20.5], [11.0, 19.5], [9.0, 19.5], [9.0, 20.5]],
                }
            ],
        },
        "5": {
            "ego_pose": {"x": 10.0, "y": 20.0, "yaw": 0.0},
            "dynamic_objects": [
                {
                    "token": "veh-a",
                    "category": "vehicle.car",
                    "poly": [[12.0, 20.5], [12.0, 19.5], [10.0, 19.5], [10.0, 20.5]],
                }
            ],
        },
    }
    import json

    (scene_dir / "env_cache.json").write_text(json.dumps(env_cache), encoding="utf-8")

    from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils

    scorer = NuScenesScorerUtils(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
        ea_gate_enabled=True,
    )
    context = scorer._build_static_sample_context(
        {"sample_token": "tok-a", "scene_id": 7, "frame_idx": 0},
        patch_radius=20.0,
    )

    assert [obj["token"] for obj in context["ea_agent_states"]] == ["veh-a"]
    assert np.allclose(
        np.asarray(context["ea_agent_states"][0]["future_xy"], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )


def test_nuscenes_pdm_backend_builds_batched_candidate_polygon_arrays(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [2.0, 0.3, 0.2]],
                [[0.2, 0.0, 0.0], [0.8, 0.4, 0.1], [1.5, 0.9, 0.2]],
            ]
        ],
        dtype=torch.float32,
    )

    batch = scorer._build_candidate_geometry_batch(traj_xyyaw)

    assert batch["corners_xy"].shape == (1, 2, 3, 4, 2)
    assert batch["polygons"].shape == (1, 2, 3)
    assert batch["centers_xy"].shape == (1, 2, 3, 2)
    assert batch["yaw_rad"].shape == (1, 2, 3)
    first_polygon = batch["polygons"][0, 0, 0]
    assert first_polygon is not None
    assert float(first_polygon.area) > 0.0


def test_nuscenes_pdm_backend_collision_ttc_uses_batch_query_shapes(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)

    query_shapes: list[tuple[int, ...]] = []

    class RecordingOccupancyMap:
        def __len__(self) -> int:
            return 1

        def query(self, geometry, predicate=None):
            del predicate
            query_shapes.append(np.asarray(geometry, dtype=object).shape)
            return np.zeros((2, 0), dtype=np.int64)

        def intersects(self, geometry):
            del geometry
            return []

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.occupancy_map = RecordingOccupancyMap()

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (2,)
    assert metrics["ttc"].shape == (2,)
    assert query_shapes
    assert all(len(shape) == 1 for shape in query_shapes)
    assert sorted(shape[0] for shape in query_shapes) == [6, 18]


def test_nuscenes_pdm_backend_returns_step_level_collision_matrix(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.ttc_agent_states = []

    class NonEmptyOccupancyMap:
        def __len__(self) -> int:
            return 1

        def query(self, geometry, predicate=None):
            del predicate
            return np.zeros((2, 0), dtype=np.int64)

    sample_context.occupancy_map = NonEmptyOccupancyMap()

    step_hits = np.asarray(
        [
            [False, True, False],
            [False, False, False],
        ],
        dtype=bool,
    )
    monkeypatch.setattr(
        scorer,
        "_query_hits_per_candidate_grid",
        lambda occupancy_map, polygons_grid, *, predicate="intersects": step_hits.copy()
        if np.asarray(polygons_grid, dtype=object).shape == (2, 3)
        else np.zeros((2, 9), dtype=bool),
    )

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].tolist() == pytest.approx([0.0, 1.0])
    assert metrics["collision_matrix"].shape == (2, 3)
    assert np.array_equal(metrics["collision_matrix"], step_hits)


def test_nuscenes_pdm_backend_prebuilds_ttc_projection_polygon_arrays(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    projections = scorer._build_ttc_projection_geometry(
        centers_xy=geometry["centers_xy"][0],
        yaw_rad=geometry["yaw_rad"][0],
        dt_s=0.5,
    )

    assert projections["centers_xy"].shape == (2, 3, 3, 2)
    assert projections["corners_xy"].shape == (2, 3, 3, 4, 2)
    assert projections["polygons"].shape == (2, 3, 3)
    assert projections["offsets_s"].shape == (3,)


def test_nuscenes_pdm_backend_batch_drivable_and_lane_queries_return_expected_shapes(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
                [[0.0, 3.0, 0.0], [0.8, 3.0, 0.0], [1.6, 3.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)

    class FakeDrivableMap:
        def __init__(self):
            self.calls: list[tuple[int, ...]] = []

        def batch_contains_points(self, points_xy):
            self.calls.append(tuple(points_xy.shape))
            if tuple(points_xy.shape) == (2, 3, 4, 2):
                return np.asarray(
                    [
                        [
                            [True, True, True, True],
                            [True, True, True, True],
                            [True, True, True, True],
                        ],
                        [
                            [False, False, False, False],
                            [False, False, False, False],
                            [False, False, False, False],
                        ],
                    ],
                    dtype=bool,
                )
            raise AssertionError(f"unexpected drivable query shape: {tuple(points_xy.shape)}")

    fake_map = FakeDrivableMap()
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.drivable_map = fake_map
    sample_context.centerline_segments_xy = np.asarray(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    sample_context.centerline_tangents_xy = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = scorer._batch_map_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["drivable_area"].shape == (2,)
    assert metrics["lane_keeping"].shape == (2,)
    assert metrics["driving_direction"].shape == (2,)
    assert fake_map.calls == [(2, 3, 4, 2)]


def test_nuscenes_pdm_backend_drivable_area_uses_ego_box_corners(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)

    class FakeDrivableMap:
        def batch_contains_points(self, points_xy):
            if tuple(points_xy.shape) != (1, 3, 4, 2):
                raise AssertionError(f"unexpected drivable query shape: {tuple(points_xy.shape)}")
            # Centerline is fully drivable, but one corner leaves the map at the second step.
            return np.asarray(
                [
                    [
                        [True, True, True, True],
                        [True, False, True, True],
                        [True, True, True, True],
                    ]
                ],
                dtype=bool,
            )

    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.drivable_map = FakeDrivableMap()
    sample_context.centerline_segments_xy = np.asarray(
        [
            [[0.0, 0.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    sample_context.centerline_tangents_xy = np.asarray(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = scorer._batch_map_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["drivable_area"].shape == (1,)
    assert metrics["drivable_area"][0] == pytest.approx(0.0)


def test_nuscenes_pdm_backend_driving_direction_prefers_plausible_same_direction_lane(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    horizon = 8
    traj_xyyaw = torch.zeros((1, 1, horizon, 3), dtype=torch.float32)
    traj_xyyaw[0, 0, :, 0] = torch.arange(horizon, dtype=torch.float32)
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)

    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)
    sample_context.centerline_segments_xy = np.asarray(
        [
            [[10.0, 2.24], [0.0, 2.24]],
            [[0.0, -2.50], [10.0, -2.50]],
        ],
        dtype=np.float32,
    )
    sample_context.centerline_tangents_xy = np.asarray(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = scorer._batch_map_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["driving_direction"].shape == (1,)
    assert metrics["driving_direction"][0] == pytest.approx(1.0)


def test_nuscenes_pdm_backend_batch_project_progress_matches_scalar_reference(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer
    from framework.algorithms.nuscenes_scorer_utils import _project_progress

    scorer = NuScenesPDMScorer(token2vad_path=token2vad_path)
    final_points_xy = np.asarray(
        [
            [0.4, 0.1],
            [1.6, -0.2],
            [2.5, 0.4],
        ],
        dtype=np.float32,
    )
    path_xy = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    path_s = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    batch_progress = scorer._batch_project_progress(final_points_xy, path_xy, path_s)
    scalar_progress = np.asarray(
        [_project_progress(point_xy, path_xy, path_s) for point_xy in final_points_xy],
        dtype=np.float32,
    )

    assert batch_progress.shape == (3,)
    assert np.allclose(batch_progress, scalar_progress, atol=1.0e-5)


def test_nuscenes_pdm_backend_ea_gate_is_optional_and_gates_batched_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    scene_cache_root = tmp_path / "scene_cache"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    traj_xyyaw = torch.tensor(
        [
            [
                [[2.0, 0.0, 0.0], [3.5, 0.0, 0.0], [4.5, 0.0, 0.0]],
                [[1.0, 2.5, 0.0], [2.0, 2.5, 0.0], [3.0, 2.5, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "gt_xy": np.asarray([[2.0, 0.0], [3.5, 0.0], [4.5, 0.0]], dtype=np.float32),
            "gt_yaw": np.zeros((3,), dtype=np.float32),
            "gt_s": np.asarray([0.0, 1.5, 2.5], dtype=np.float32),
            "gt_total_len": 2.5,
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [],
            "ea_agent_states": [
                {
                    "category": "vehicle.car",
                    "center_xy": [4.0, 0.0],
                    "yaw_rad": 0.0,
                    "yaw_rate_rps": 0.0,
                    "velocity_xy": [0.0, 0.0],
                    "speed_mps": 0.0,
                    "length_m": 4.8,
                    "width_m": 2.0,
                }
            ],
        }

    scorer_plain = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root / "plain",
    )
    monkeypatch.setattr(scorer_plain._delegate, "_build_static_sample_context", fake_static_context)
    plain_scores = scorer_plain.score([{"sample_token": "tok-a"}], traj_xyyaw)

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root / "ea",
        ea_gate_enabled=True,
        ea_gate_good_threshold=0.0,
        ea_gate_bad_threshold=5.0,
    )
    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)
    monkeypatch.setattr(
        scorer,
        "_compute_ea_value_batch_for_pairs",
        lambda ego_states, agent_states: np.asarray(
            [
                4.0 if float(ego_state["x"]) > 1.5 and abs(float(ego_state["y"])) < 0.5 else 0.0
                for ego_state, _agent_state in zip(ego_states, agent_states, strict=False)
            ],
            dtype=np.float32,
        ),
        raising=False,
    )

    gated_scores = scorer.score([{"sample_token": "tok-a"}], traj_xyyaw)

    assert plain_scores.shape == (1, 2)
    assert gated_scores.shape == (1, 2)
    assert float(plain_scores[0, 0]) == pytest.approx(0.0)
    assert float(gated_scores[0, 0]) == pytest.approx(0.0)
    assert float(gated_scores[0, 1]) == pytest.approx(float(plain_scores[0, 1]), rel=1.0e-6)


def test_nuscenes_pdm_backend_can_disable_driving_direction_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    scene_cache_root = tmp_path / "scene_cache"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    traj_xyyaw = torch.tensor(
        [
            [
                [[-1.0, 0.0, np.pi], [-2.0, 0.0, np.pi], [-3.0, 0.0, np.pi], [-4.0, 0.0, np.pi], [-5.0, 0.0, np.pi]],
            ]
        ],
        dtype=torch.float32,
    )

    def fake_static_context(replay, *, patch_radius: float):
            del replay
            return {
                "gt_xy": np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]], dtype=np.float32),
                "gt_yaw": np.zeros((5,), dtype=np.float32),
                "gt_s": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "gt_total_len": 4.0,
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {
                    "drivable_area": [
                        [[-8.0, -4.0], [8.0, -4.0], [8.0, 4.0], [-8.0, 4.0], [-8.0, -4.0]],
                    ],
                    "lane_centerline": [
                        [[-6.0, 0.0], [-3.0, 0.0], [0.0, 0.0], [3.0, 0.0], [6.0, 0.0]],
                    ],
                },
            },
            "scene_objects": [],
            "ea_agent_states": [],
        }

    scorer_default = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root / "default",
    )
    monkeypatch.setattr(scorer_default._delegate, "_build_static_sample_context", fake_static_context)
    default_scores = scorer_default.score([{"sample_token": "tok-a"}], traj_xyyaw)

    scorer_disabled = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=scene_cache_root / "disabled",
        driving_direction_gate_enabled=False,
    )
    monkeypatch.setattr(scorer_disabled._delegate, "_build_static_sample_context", fake_static_context)
    disabled_scores = scorer_disabled.score([{"sample_token": "tok-a"}], traj_xyyaw)

    assert default_scores.shape == (1, 1)
    assert disabled_scores.shape == (1, 1)
    assert float(disabled_scores[0, 0]) > float(default_scores[0, 0])


def test_nuscenes_pdm_backend_ttc_uses_agent_future_truth_boxes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "row": {
                    "gt_boxes": np.asarray(
                        [
                            [4.4, 1.0, 0.0, 1.0, 4.0, 1.8, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                "gt_velocity": np.asarray([[0.0, 0.0]], dtype=np.float32),
                "gt_names": np.asarray(["vehicle.car"], dtype=object),
                "valid_flag": np.asarray([True], dtype=bool),
                "gt_agent_fut_trajs": np.asarray(
                    [
                        [[0.0, -1.0], [0.0, -2.0], [0.0, -3.0]],
                    ],
                    dtype=np.float32,
                ),
                "gt_agent_fut_masks": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
                "gt_agent_fut_yaw": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            },
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [4.4, 1.0],
                    "velocity_xy": [0.0, 0.0],
                    "yaw_rad": 0.0,
                    "length_m": 4.0,
                    "width_m": 1.0,
                }
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (1,)
    assert metrics["ttc"].shape == (1,)
    assert metrics["no_collision"][0] == pytest.approx(0.0)
    assert float(metrics["ttc"][0]) < 1.0


def test_nuscenes_pdm_backend_ttc_falls_back_to_ctrv_when_future_truth_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [4.4, 6.5],
                    "velocity_xy": [0.0, -2.0],
                    "yaw_rad": -np.pi * 0.5,
                    "yaw_rate_rps": 0.0,
                    "length_m": 4.0,
                    "width_m": 1.0,
                }
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (1,)
    assert metrics["ttc"].shape == (1,)
    assert metrics["no_collision"][0] == pytest.approx(1.0)
    assert float(metrics["ttc"][0]) < 1.0


def test_nuscenes_pdm_backend_no_collision_uses_future_agent_boxes_not_static_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "row": {
                "gt_boxes": np.asarray(
                    [
                        [2.0, 0.0, 0.0, 1.0, 4.0, 1.8, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "gt_velocity": np.asarray([[0.0, 0.0]], dtype=np.float32),
                "gt_names": np.asarray(["vehicle.car"], dtype=object),
                "valid_flag": np.asarray([True], dtype=bool),
                "gt_agent_fut_trajs": np.asarray(
                    [
                        [[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]],
                    ],
                    dtype=np.float32,
                ),
                "gt_agent_fut_masks": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
                "gt_agent_fut_yaw": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            },
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [2.0, 0.0],
                    "velocity_xy": [0.0, 0.0],
                    "yaw_rad": 0.0,
                    "length_m": 4.0,
                    "width_m": 1.0,
                }
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    traj_xyyaw = torch.tensor(
        [
            [
                [[7.0, 0.0, 0.0], [9.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (1,)
    assert metrics["no_collision"][0] == pytest.approx(1.0)


def test_nuscenes_pdm_backend_no_collision_detects_future_agent_box_overlap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "row": {
                "gt_boxes": np.asarray(
                    [
                        [20.0, 0.0, 0.0, 4.0, 1.0, 1.8, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "gt_velocity": np.asarray([[0.0, 0.0]], dtype=np.float32),
                "gt_names": np.asarray(["vehicle.car"], dtype=object),
                "valid_flag": np.asarray([True], dtype=bool),
                "gt_agent_fut_trajs": np.asarray(
                    [
                        [[0.0, -16.0], [0.0, -16.0], [0.0, -16.0]],
                    ],
                    dtype=np.float32,
                ),
                "gt_agent_fut_masks": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
                "gt_agent_fut_yaw": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            },
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [20.0, 0.0],
                    "velocity_xy": [0.0, 0.0],
                    "yaw_rad": 0.0,
                    "length_m": 4.0,
                    "width_m": 1.0,
                }
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (1,)
    assert metrics["no_collision"][0] == pytest.approx(0.0)


def test_nuscenes_pdm_backend_no_collision_aligns_future_agents_to_candidate_step_times(
    monkeypatch,
    tmp_path: Path,
) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )

    def fake_static_context(replay, *, patch_radius: float):
        del replay
        return {
            "row": {
                "gt_boxes": np.asarray(
                    [
                        [20.0, 0.0, 0.0, 1.0, 4.0, 1.8, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "gt_velocity": np.asarray([[0.0, 0.0]], dtype=np.float32),
                "gt_names": np.asarray(["vehicle.car"], dtype=object),
                "valid_flag": np.asarray([True], dtype=bool),
                # Local future deltas that place the agent at x=6.0 at the first future step.
                "gt_agent_fut_trajs": np.asarray(
                    [
                        [[0.0, -14.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                "gt_agent_fut_masks": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
                "gt_agent_fut_yaw": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            },
            "map_context": {
                "patch_radius": float(patch_radius),
                "layers": {"drivable_area": [], "lane_centerline": []},
            },
            "scene_objects": [
                {
                    "category": "vehicle.car",
                    "center_xy": [20.0, 0.0],
                    "velocity_xy": [0.0, 0.0],
                    "yaw_rad": 0.0,
                    "length_m": 4.0,
                    "width_m": 1.0,
                }
            ],
            "ea_agent_states": [],
        }

    monkeypatch.setattr(scorer._delegate, "_build_static_sample_context", fake_static_context)

    traj_xyyaw = torch.tensor(
        [
            [
                [[6.0, 0.0, 0.0], [8.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context({"sample_token": "tok-a"}, patch_radius=20.0)

    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert metrics["no_collision"].shape == (1,)
    assert metrics["no_collision"][0] == pytest.approx(0.0)


def test_nuscenes_pdm_backend_replay_dynamic_objects_use_future_xy_for_collision(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)

    from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

    scorer = NuScenesPDMScorer(
        token2vad_path=token2vad_path,
        scene_cache_root=tmp_path / "scene_cache",
    )
    traj_xyyaw = torch.tensor(
        [
            [
                [[6.0, 0.0, 0.0], [8.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    replay = {
        "sample_token": "tok-a",
        "scene_objects_override": [
            {
                "token": "dyn-a",
                "category": "vehicle.car",
                "center_xy": [20.0, 0.0],
                "yaw_rad": 0.0,
                "velocity_xy": [0.0, 0.0],
                "length_m": 4.0,
                "width_m": 1.0,
                "future_xy": [[6.0, 0.0], [20.0, 0.0], [20.0, 0.0]],
                "future_yaw": [0.0, 0.0, 0.0],
                "future_dt_s": 0.5,
            }
        ],
    }

    geometry = scorer._build_candidate_geometry_batch(traj_xyyaw)
    sample_context = scorer._build_sample_context(replay, patch_radius=20.0)
    metrics = scorer._batch_collision_ttc_metrics(
        sample_context=sample_context,
        candidate_geometry={key: value[0] for key, value in geometry.items()},
        dt_s=0.5,
    )

    assert "future_xy" in sample_context.ttc_agent_states[0]
    assert metrics["no_collision"].shape == (1,)
    assert metrics["no_collision"][0] == pytest.approx(0.0)
