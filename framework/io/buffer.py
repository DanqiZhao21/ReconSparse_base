from __future__ import annotations

# Thin wrappers re-exporting buffer IO helpers.

from framework.io.actor_learner_io import (
    BufferPaths,
    atomic_torch_save,
    ensure_buffer_layout,
    list_shards,
    move_to_consumed,
    prune_consumed,
    read_int,
    write_int,
    wait_for_version,
    count_inflight,
    stop_requested,
)
