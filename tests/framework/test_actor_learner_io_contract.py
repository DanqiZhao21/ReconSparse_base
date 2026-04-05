import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from framework.io.buffer import BufferPaths, ensure_buffer_layout, list_shards, read_int, wait_for_version, write_int


def test_buffer_paths_create_expected_layout(tmp_path):
    paths = BufferPaths(root=str(tmp_path / "actor_learner"))
    ensure_buffer_layout(paths)

    assert pathlib.Path(paths.shards_dir).is_dir()
    assert pathlib.Path(paths.consumed_dir).is_dir()
    assert pathlib.Path(paths.weights_dir).is_dir()


def test_version_round_trip_and_shard_listing(tmp_path):
    paths = BufferPaths(root=str(tmp_path / "actor_learner"))
    ensure_buffer_layout(paths)
    write_int(paths.version_file, 7)
    torch.save({"ok": True}, pathlib.Path(paths.shards_dir) / "actor0_e0_v7_t0_deadbeef.pt")

    assert read_int(paths.version_file, default=0) == 7
    assert len(list_shards(paths)) == 1
    assert wait_for_version(paths, min_version=7, poll_s=0.001, timeout_s=0.01) == 7
