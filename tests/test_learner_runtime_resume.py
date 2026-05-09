from __future__ import annotations

from pathlib import Path

from framework.io.buffer import BufferPaths, ensure_buffer_layout, write_int
from framework.runner.learner_runtime import _restore_learner_checkpoint_if_available


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
