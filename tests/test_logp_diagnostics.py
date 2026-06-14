from __future__ import annotations

import math

import torch

from framework.debug.logp_diagnostics import (
    collect_shard_paths,
    compute_recompute_error_summary,
    compute_logp_diagnostics,
    summarize_rows_by_key,
    summarize_replay_entry,
)


def test_compute_logp_diagnostics_reports_kl_clip_and_top_anomalies() -> None:
    old_logp = torch.tensor([-2.0, -1.0, -3.0], dtype=torch.float32)
    new_logp = torch.tensor([-2.0, -0.5, -5.0], dtype=torch.float32)

    diagnostics = compute_logp_diagnostics(
        old_logp=old_logp,
        new_logp=new_logp,
        clip_eps=0.2,
        top_k=2,
    )

    delta = new_logp - old_logp
    ratio = torch.exp(delta)
    approx_kl_terms = (ratio - 1.0) - delta

    assert diagnostics.count == 3
    assert math.isclose(diagnostics.summary["approx_kl"], float(approx_kl_terms.mean()), rel_tol=1.0e-6)
    assert math.isclose(diagnostics.summary["clip_frac"], 2.0 / 3.0, rel_tol=1.0e-6)
    assert diagnostics.top_anomalies[0]["index"] == 2
    assert diagnostics.top_anomalies[1]["index"] == 1
    assert math.isclose(diagnostics.rows[2]["delta_logp"], -2.0, rel_tol=1.0e-6)


def test_summarize_replay_entry_reports_nested_tensor_stats() -> None:
    replay = {
        "global_mode_idx": 75,
        "selected_path_idx": 7,
        "selected_vel_idx": 5,
        "camera_feature": {
            "front": torch.tensor([[1.0, 2.0], [float("nan"), float("inf")]], dtype=torch.float32),
        },
        "status_feature": torch.tensor([[0.5, -0.5]], dtype=torch.float32),
        "traj_xyyaw": torch.zeros((6, 3), dtype=torch.float32),
    }

    summary = summarize_replay_entry(replay)

    assert summary["global_mode_idx"] == 75
    assert summary["selected_path_idx"] == 7
    assert summary["selected_vel_idx"] == 5
    assert summary["camera_feature.front.shape"] == [2, 2]
    assert summary["camera_feature.front.nan_count"] == 1
    assert summary["camera_feature.front.inf_count"] == 1
    assert summary["camera_feature.front.finite_count"] == 2
    assert summary["status_feature.shape"] == [1, 2]
    assert summary["traj_xyyaw.shape"] == [6, 3]


def test_collect_shard_paths_reads_live_and_consumed_buffers(tmp_path) -> None:
    live_dir = tmp_path / "buffer" / "shards"
    consumed_dir = tmp_path / "buffer" / "consumed"
    live_dir.mkdir(parents=True)
    consumed_dir.mkdir(parents=True)
    live_path = live_dir / "actor0_shard000.pt"
    consumed_path = consumed_dir / "actor1_shard000.pt"
    live_path.write_bytes(b"live")
    consumed_path.write_bytes(b"consumed")

    only_live = collect_shard_paths(buffer_dir=tmp_path, include_consumed=False)
    with_consumed = collect_shard_paths(buffer_dir=tmp_path, include_consumed=True)
    explicit = collect_shard_paths(shards=[consumed_path, live_path], limit_shards=1)

    assert only_live == [live_path]
    assert with_consumed == [consumed_path, live_path]
    assert explicit == [consumed_path]


def test_summarize_rows_by_key_reports_per_shard_kl_and_clip() -> None:
    rows = [
        {"shard_path": "a.pt", "approx_kl_term": 0.1, "is_clipped": 0, "delta_logp": 0.1},
        {"shard_path": "a.pt", "approx_kl_term": 0.3, "is_clipped": 1, "delta_logp": 0.6},
        {"shard_path": "b.pt", "approx_kl_term": 2.0, "is_clipped": 1, "delta_logp": -2.0},
    ]

    summary = summarize_rows_by_key(rows, key="shard_path")

    assert summary[0]["shard_path"] == "b.pt"
    assert summary[0]["count"] == 1
    assert summary[0]["approx_kl"] == 2.0
    assert summary[0]["clip_frac"] == 1.0
    assert summary[1]["shard_path"] == "a.pt"
    assert summary[1]["count"] == 2
    assert math.isclose(summary[1]["approx_kl"], 0.2, rel_tol=1.0e-6)
    assert math.isclose(summary[1]["clip_frac"], 0.5, rel_tol=1.0e-6)


def test_compute_recompute_error_summary_flags_old_logp_mismatches() -> None:
    stored = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    recomputed = torch.tensor([1.0, 2.000001, 3.1], dtype=torch.float32)

    summary = compute_recompute_error_summary(stored, recomputed, tolerance=1.0e-5)

    assert summary["count"] == 3
    assert summary["mismatch_count"] == 1
    assert summary["pass"] == 0.0
    assert math.isclose(summary["max_abs_error"], 0.1, rel_tol=1.0e-5)
