from __future__ import annotations

import json
import sys
from pathlib import Path

from script.debug_sparsedrive_replay_forward_report import (
    build_child_command,
    choose_target_index,
    find_config,
    find_latest_manifest,
    load_stage_summary,
    prepare_candidate_shard_dir,
    resolve_ckpt_history_dir,
)


def test_find_config_selects_single_yaml_in_run_dir(tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text("agent: {}\n", encoding="utf-8")

    assert find_config(tmp_path, None) == config


def test_find_latest_manifest_uses_highest_update_index(tmp_path: Path) -> None:
    older = tmp_path / "debug_retention" / "update_000000_from_v000001_to_v000002"
    newer = tmp_path / "debug_retention" / "update_000003_from_v000004_to_v000005"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "manifest.json").write_text(json.dumps({"update_index": 0, "shards": []}), encoding="utf-8")
    (newer / "manifest.json").write_text(json.dumps({"update_index": 3, "shards": []}), encoding="utf-8")

    assert find_latest_manifest(tmp_path, None) == newer / "manifest.json"


def test_resolve_ckpt_history_dir_defaults_to_run_weights_history(tmp_path: Path) -> None:
    history = tmp_path / "weights" / "history"
    history.mkdir(parents=True)

    assert resolve_ckpt_history_dir(tmp_path, None) == history


def test_choose_target_index_prefers_single_recovered_candidate(tmp_path: Path) -> None:
    rows = tmp_path / "rows.csv"
    rows.write_text(
        "index,label,batch_present,single_present_when_missing\n"
        "0,a,1,\n"
        "1,b,0,1\n"
        "2,c,0,0\n",
        encoding="utf-8",
    )
    first_failures = tmp_path / "first_failures.csv"
    first_failures.write_text("sample_index,stage\n2,decoder0/path_scores\n", encoding="utf-8")

    assert choose_target_index(rows, first_failures, explicit=None) == 1


def test_choose_target_index_falls_back_to_forward_first_failure(tmp_path: Path) -> None:
    rows = tmp_path / "rows.csv"
    rows.write_text(
        "index,label,batch_present,single_present_when_missing\n"
        "0,a,1,\n"
        "1,b,0,0\n",
        encoding="utf-8",
    )
    first_failures = tmp_path / "first_failures.csv"
    first_failures.write_text("sample_index,stage\n3,decoder0/path_scores\n", encoding="utf-8")

    assert choose_target_index(rows, first_failures, explicit=None) == 3


def test_load_stage_summary_returns_first_failing_stage(tmp_path: Path) -> None:
    stage_summary = tmp_path / "stage_summary.csv"
    stage_summary.write_text(
        "stage,count,failed,max_abs_error,mismatch_count\n"
        "input/status_feature,4,0,0.0,0\n"
        "decoder0/p_deform/key_points,4,2,1.5,10\n",
        encoding="utf-8",
    )

    summary = load_stage_summary(stage_summary)

    assert summary["first_failed_stage"] == "decoder0/p_deform/key_points"
    assert summary["failed_stage_count"] == 1


def test_build_child_command_includes_common_determinism_flags(tmp_path: Path) -> None:
    command = build_child_command(
        "script/debug_sparsedrive_batch_invariance.py",
        config=tmp_path / "config.yaml",
        manifest=tmp_path / "manifest.json",
        ckpt_history_dir=tmp_path / "weights" / "history",
        version0_ckpt=None,
        weights_version=3,
        batch_size=8,
        target_index=None,
        out_dir=tmp_path / "out",
        device="cuda:0",
        tolerance=1.0e-5,
        disable_tf32=True,
        deterministic=True,
        extra_args=["--limit-samples", "8"],
    )

    assert command[:2] == [sys.executable, "script/debug_sparsedrive_batch_invariance.py"]
    assert "--weights-version" in command
    assert "3" in command
    assert "--disable-tf32" in command
    assert "--deterministic" in command
    assert "--limit-samples" in command


def test_prepare_candidate_shard_dir_filters_selected_version_in_manifest_order(tmp_path: Path) -> None:
    shard_a = tmp_path / "a.pt"
    shard_b = tmp_path / "b.pt"
    shard_c = tmp_path / "c.pt"
    shard_a.write_bytes(b"a")
    shard_b.write_bytes(b"b")
    shard_c.write_bytes(b"c")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "shards": [
                    {"archive_path": str(shard_b), "weights_version": 3},
                    {"archive_path": str(shard_a), "weights_version": 2},
                    {"archive_path": str(shard_c), "weights_version": 3},
                ]
            }
        ),
        encoding="utf-8",
    )

    out = prepare_candidate_shard_dir(manifest, weights_version=3, target_dir=tmp_path / "filtered")

    assert [path.name for path in sorted(out.glob("*.pt"))] == ["000000_b.pt", "000001_c.pt"]
