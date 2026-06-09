from __future__ import annotations

import pytest
import torch

from tools.smalltool.debbbug.compare_sparsedrive_v2_modes import compare_policy_outputs


def test_compare_policy_outputs_reports_mode_and_trajectory_changes() -> None:
    base_scores = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    trained_scores = torch.tensor(
        [
            [2.5, 3.0, 0.0],
            [0.0, 2.5, 0.5],
        ],
        dtype=torch.float32,
    )
    base_trajs = torch.zeros((2, 3, 2, 3), dtype=torch.float32)
    trained_trajs = base_trajs.clone()
    base_trajs[0, 0, 0, 0] = 1.0
    trained_trajs[0, 1, 0, 0] = 4.0
    base_trajs[1, 1, 0, 1] = 2.0
    trained_trajs[1, 1, 0, 1] = 5.0

    summary = compare_policy_outputs(
        base_scores=base_scores,
        trained_scores=trained_scores,
        base_trajs=base_trajs,
        trained_trajs=trained_trajs,
        replay_mode_indices=torch.tensor([0, 1], dtype=torch.long),
    )

    assert summary["num_samples"] == 2
    assert summary["top1_changed_rate"] == pytest.approx(0.5)
    assert summary["top1_base_mode_hist"] == {"0": 1, "1": 1}
    assert summary["top1_trained_mode_hist"] == {"1": 2}
    assert summary["top1_logit_margin_base_mean"] == pytest.approx(1.5)
    assert summary["top1_logit_margin_trained_mean"] == pytest.approx(1.25)
    assert summary["top1_traj_l2_mean"] == pytest.approx(3.0)
    assert summary["first_step_l2_mean"] == pytest.approx(3.0)
    assert summary["selected_replay_mode_logp_delta_mean"] < 0.0
