from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from framework.io.buffer import BufferPaths, ensure_buffer_layout, write_int
from framework.runner.learner_runtime import (
    _retain_initial_checkpoint_if_configured,
    _restore_learner_checkpoint_if_available,
    _restore_rank_checkpoint_if_available,
)


class _RecordingAgent:
    def __init__(self) -> None:
        self.loaded: list[tuple[str, bool]] = []

    def load_checkpoint(self, path: str, *, strict: bool = False) -> None:
        self.loaded.append((str(path), bool(strict)))


def test_restore_learner_checkpoint_loads_latest_published_weights(tmp_path: Path) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    write_int(paths.version_file, 26)
    Path(paths.latest_ckpt).write_text("stub checkpoint", encoding="utf-8")
    agent = _RecordingAgent()

    restored_version = _restore_learner_checkpoint_if_available(
        agent=agent,
        paths=paths,
        stage_fn=lambda *_args, **_kwargs: None,
    )

    assert restored_version == 26
    assert agent.loaded == [(paths.latest_ckpt, False)]


def test_restore_rank_checkpoint_loads_latest_published_weights_on_nonzero_rank(tmp_path: Path) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    write_int(paths.version_file, 7)
    Path(paths.latest_ckpt).write_text("stub checkpoint", encoding="utf-8")
    agent = _RecordingAgent()

    restored_version = _restore_rank_checkpoint_if_available(
        agent=agent,
        paths=paths,
        rank=1,
        stage_fn=lambda *_args, **_kwargs: None,
    )

    assert restored_version == 7
    assert agent.loaded == [(paths.latest_ckpt, False)]


def test_retain_initial_checkpoint_copies_version_one_when_enabled(tmp_path: Path) -> None:
    paths = BufferPaths(root=str(tmp_path / "buffer_root"))
    ensure_buffer_layout(paths)
    Path(paths.latest_ckpt).write_text("stub checkpoint", encoding="utf-8")
    learner_config = SimpleNamespace(debug_retain_versions=5, debug_retain_ckpts=True)

    _retain_initial_checkpoint_if_configured(
        paths=paths,
        learner_config=learner_config,
        stage_fn=lambda *_args, **_kwargs: None,
    )

    assert (Path(paths.weights_dir) / "history" / "version_000001.ckpt").read_text(encoding="utf-8") == "stub checkpoint"
