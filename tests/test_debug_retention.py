from __future__ import annotations

import json
from pathlib import Path

import torch

from framework.io.buffer import BufferPaths, ensure_buffer_layout
from framework.io.debug_retention import (
    archive_selected_shards_for_debug,
    history_checkpoint_path,
)


def test_history_checkpoint_path_uses_zero_padded_version(tmp_path: Path) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer"))

    assert Path(history_checkpoint_path(paths, version=5)).as_posix().endswith(
        "weights/history/version_000005.ckpt"
    )


def test_archive_selected_shards_for_debug_copies_shards_and_writes_manifest(tmp_path: Path) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer"))
    ensure_buffer_layout(paths)
    shard_path = Path(paths.shards_dir) / "actor0_v3_shard000.pt"
    torch.save(
        {
            "old_logp": torch.zeros(2),
            "reward": torch.ones(2),
            "replay": [{"step": 0}, {"step": 1}],
            "meta": {"weights_version": 3, "actor_id": 0, "num_steps": 2},
        },
        shard_path,
    )

    manifest_path = archive_selected_shards_for_debug(
        paths,
        selected=[str(shard_path)],
        update_index=2,
        cur_version=3,
        new_version=4,
    )

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    archived_shard = Path(manifest["shards"][0]["archive_path"])
    assert archived_shard.exists()
    assert archived_shard.name == shard_path.name
    assert manifest["update_index"] == 2
    assert manifest["cur_version"] == 3
    assert manifest["new_version"] == 4
    assert manifest["shards"][0]["weights_version"] == 3
    assert manifest["shards"][0]["actor_id"] == 0
