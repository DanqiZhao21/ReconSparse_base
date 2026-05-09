from __future__ import annotations

import os
from pathlib import Path

from tools.monitor_training import build_snapshot, find_latest_train_log, infer_buffer_dir, parse_log_text


def test_parse_log_text_extracts_progress_and_anomalies() -> None:
    parsed = parse_log_text(
        """
[00:13:43] [learner] start algo=reinforcepp device=cuda:0 weights_version=1 max_updates=32
[00:15:48] [actor1] wrote shard 1 horizon=32 ver=1 collect=55.92s step_avg=1.00s act_avg=0.21s env_avg=0.79s render_avg=0.69s backpressure=0.00s save=0.35s reset=2/10.67s done=2
[00:16:17] [learner] stage2 train: selected_shards=24 inner_epochs=1
[00:16:26] [learner] slow_step update=0 batch_idx=0 part=grpo_debug took=6.69s
[00:19:01] [learner] update=0 shards=24 samples=768 ver=2 metrics={'loss_pi': 0.12, 'approx_kl': 0.03}
[00:19:01] [learner] reward_summary update=0 summary={'positive_reward_mean': 0.4, 'cost_reward_mean': 0.2, 'safety_gate_rate': 0.1}
[00:19:01] [learner] step_timing update=0 parts={'training_step_total_s': 120.0, 'grpo_debug_s': 80.0}
[00:19:01] [learner] timing update=0 collect=90.50s load=1.20s prepare=0.50s train=120.00s save=0.80s update=210.50s time_per_shard=5.00s
"""
    )

    assert parsed.max_updates == 32
    assert parsed.latest_update == 0
    assert parsed.weight_version == 2
    assert parsed.num_completed_updates == 1
    assert len(parsed.actor_shards) == 1
    assert len(parsed.slow_steps) == 1
    assert parsed.update_timings[0].update_time_s == 210.5
    assert parsed.update_metrics[0]["loss_pi"] == 0.12
    assert parsed.reward_summary[0]["positive_reward_mean"] == 0.4
    assert any("grpo_debug" in item for item in parsed.anomalies)


def test_find_latest_train_log_prefers_newest_file(tmp_path: Path) -> None:
    old_run = tmp_path / "old_run"
    new_run = tmp_path / "new_run"
    old_run.mkdir()
    new_run.mkdir()
    old_log = old_run / "train.log"
    new_log = new_run / "train.log"
    old_log.write_text("old", encoding="utf-8")
    new_log.write_text("new", encoding="utf-8")

    os.utime(old_log, (1, 1))
    os.utime(new_log, (2, 2))

    latest = find_latest_train_log(tmp_path)

    assert latest == new_log


def test_build_snapshot_reports_buffer_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "ops_logs" / "demo_run"
    run_dir.mkdir(parents=True)
    log_path = run_dir / "train.log"
    log_path.write_text(
        """
[00:13:43] [learner] start algo=reinforcepp device=cuda:0 weights_version=1 max_updates=8
[00:19:01] [learner] timing update=0 collect=90.50s load=1.20s prepare=0.50s train=120.00s save=0.80s update=210.50s time_per_shard=5.00s
""",
        encoding="utf-8",
    )

    buffer_dir = tmp_path / "buffer_root"
    shards_dir = buffer_dir / "buffer" / "shards"
    consumed_dir = buffer_dir / "buffer" / "consumed"
    weights_dir = buffer_dir / "weights"
    shards_dir.mkdir(parents=True)
    consumed_dir.mkdir(parents=True)
    weights_dir.mkdir(parents=True)
    (shards_dir / "actor0.pt").write_text("", encoding="utf-8")
    (consumed_dir / "actor0.pt").write_text("", encoding="utf-8")
    (weights_dir / "version.txt").write_text("3\n", encoding="utf-8")
    (buffer_dir / "TRAINING_LOCK").write_text("training\n", encoding="utf-8")

    snapshot = build_snapshot(log_path=log_path, buffer_dir=buffer_dir)

    assert snapshot["buffer"]["pending_shards"] == 1
    assert snapshot["buffer"]["consumed_shards"] == 1
    assert snapshot["buffer"]["training_lock_present"] is True
    assert snapshot["progress"]["completed_updates"] == 1
    assert snapshot["progress"]["weights_version"] == 3


def test_infer_buffer_dir_resolves_relative_shard_dir_from_repo_root(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    log_path = repo_root / "outputs" / "ops_logs" / "demo_run" / "train.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("", encoding="utf-8")

    parsed = parse_log_text(
        """
[00:13:47] [learner] stage1 collect: have_shards=0/24 (dir=outputs/actor_learner_reinfroce_batched/buffer/shards)
"""
    )

    monkeypatch.chdir(repo_root)

    inferred = infer_buffer_dir(log_path, parsed)

    assert inferred == repo_root / "outputs" / "actor_learner_reinfroce_batched"
