from __future__ import annotations

import threading
import time
from typing import Any, Callable

from framework.io.buffer import write_actor_heartbeat


class PeriodicActorHeartbeat:
    def __init__(
        self,
        *,
        paths: Any,
        actor_id: int,
        interval_s: float,
        writer: Callable[..., Any] = write_actor_heartbeat,
        stage_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self.paths = paths
        self.actor_id = int(actor_id)
        self.interval_s = float(interval_s)
        self._writer = writer
        self._stage_fn = stage_fn
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = "init"
        self._step: int | None = None
        self._last_write_t = 0.0

    def start(self) -> None:
        if self.interval_s <= 0.0:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"actor{self.actor_id}-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, min(float(self.interval_s), 5.0)))
        self._thread = None

    def beat(self, phase: str | None = None, step: int | None = None, *, force: bool = False) -> None:
        if self.interval_s <= 0.0:
            return
        if phase is not None:
            with self._lock:
                self._phase = str(phase)
                self._step = None if step is None else int(step)
        if (not bool(force)) and (time.time() - float(self._last_write_t)) < float(self.interval_s):
            return
        self._write_current()

    def _run(self) -> None:
        while not self._stop.wait(float(self.interval_s)):
            self._write_current()

    def _write_current(self) -> None:
        with self._lock:
            phase = str(self._phase)
            step = self._step
        message = phase if step is None else f"{phase} step={int(step)}"
        try:
            self._writer(self.paths, int(self.actor_id), message=message)
            self._last_write_t = time.time()
        except Exception as exc:
            if callable(self._stage_fn):
                self._stage_fn(f"[actor{self.actor_id}] heartbeat write failed: {exc}")


__all__ = ["PeriodicActorHeartbeat"]
