from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from framework.algorithms.nuscenes_token_scorer import NuScenesTokenScorer


def _write_token2vad(path: Path) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {
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
        "tok-b": {
            "token": "tok-b",
            "gt_ego_fut_trajs": np.asarray(
                [
                    [0.5, 2.0],
                    [0.7, 2.2],
                    [0.9, 2.5],
                    [1.1, 2.9],
                ],
                dtype=np.float32,
            ),
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


def test_score_uses_batched_torch_path(monkeypatch, tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    scorer = NuScenesTokenScorer(token2vad_path=token2vad_path)

    def _raise_if_cpu_detail_path(*args, **kwargs):
        del args, kwargs
        raise AssertionError("score() should not call _score_batch() on the training path")

    monkeypatch.setattr(scorer, "_score_batch", _raise_if_cpu_detail_path)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.7, 0.4, 0.0], [0.9, 0.6, 0.0]],
                [[0.2, -0.1, 0.0], [0.4, 0.0, 0.0], [0.6, 0.1, 0.0], [0.8, 0.2, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.2, 0.2, 0.0], [0.5, 0.4, 0.0], [0.9, 0.6, 0.0]],
                [[0.1, 0.0, 0.0], [0.15, 0.1, 0.0], [0.2, 0.2, 0.0], [0.25, 0.3, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )

    scores = scorer.score(
        [{"sample_token": "tok-a"}, {"sample_token": "tok-b"}],
        traj_xyyaw,
    )

    assert scores.shape == (2, 2)


def test_score_matches_detail_path_for_mixed_horizon_batch(tmp_path: Path) -> None:
    token2vad_path = tmp_path / "token2vad.pkl"
    _write_token2vad(token2vad_path)
    scorer = NuScenesTokenScorer(token2vad_path=token2vad_path)

    traj_xyyaw = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.7, 0.4, 0.0], [0.8, 0.5, 0.0], [1.0, 0.6, 0.0]],
                [[1.0, 1.0, 0.0], [1.2, 1.1, 0.0], [1.4, 1.2, 0.0], [1.6, 1.3, 0.0], [1.8, 1.4, 0.0]],
                [[0.0, -0.2, 0.0], [0.2, 0.0, 0.0], [0.45, 0.2, 0.0], [0.7, 0.35, 0.0], [0.95, 0.5, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.2, 0.2, 0.0], [0.5, 0.4, 0.0], [0.9, 0.6, 0.0], [1.3, 0.9, 0.0]],
                [[0.1, 0.0, 0.0], [0.15, 0.1, 0.0], [0.2, 0.2, 0.0], [0.25, 0.3, 0.0], [0.3, 0.4, 0.0]],
                [[-0.1, 0.1, 0.0], [0.0, 0.2, 0.0], [0.1, 0.4, 0.0], [0.2, 0.7, 0.0], [0.4, 1.0, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )

    scores = scorer.score(
        [{"sample_token": "tok-a"}, {"sample_token": "tok-b"}],
        traj_xyyaw,
    )
    detail_scores_np, details = scorer._score_batch(
        [{"sample_token": "tok-a"}, {"sample_token": "tok-b"}],
        traj_xyyaw,
        include_debug_context=False,
    )
    detail_scores = torch.from_numpy(detail_scores_np)

    assert len(details) == 2
    assert scores.shape == (2, 3)
    assert torch.allclose(scores.cpu(), detail_scores.cpu(), atol=1.0e-5, rtol=1.0e-5)
