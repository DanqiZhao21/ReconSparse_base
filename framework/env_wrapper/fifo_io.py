from __future__ import annotations

import errno
import os
import pickle
import select
import struct
import time
from pathlib import Path
from typing import Any


class FifoCommunicationError(RuntimeError):
    pass


def write_fifo_payload(
    path: str | Path,
    payload: Any,
    *,
    process: Any = None,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.2,
) -> None:
    fifo_path = str(path)
    deadline = time.monotonic() + float(timeout_s)
    data = pickle.dumps(payload)
    packet = struct.pack("!Q", int(len(data))) + data

    fd = None
    while fd is None:
        _ensure_process_alive(process)
        remaining = _remaining_time(deadline, fifo_path, "open for writing")
        try:
            fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno not in {errno.ENXIO, errno.ENOENT}:
                raise
            time.sleep(min(float(poll_interval_s), remaining))

    try:
        view = memoryview(packet)
        while view:
            _ensure_process_alive(process)
            remaining = _remaining_time(deadline, fifo_path, "write")
            _, writable, _ = select.select([], [fd], [], min(float(poll_interval_s), remaining))
            if not writable:
                continue
            try:
                written = os.write(fd, view)
            except BlockingIOError:
                continue
            except BrokenPipeError as exc:
                raise FifoCommunicationError(f"FIFO reader disappeared while writing to {fifo_path}") from exc
            view = view[written:]
    finally:
        os.close(fd)


def read_fifo_payload(
    path: str | Path,
    *,
    process: Any = None,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.2,
) -> Any:
    fifo_path = str(path)
    deadline = time.monotonic() + float(timeout_s)
    fd = None
    while fd is None:
        _ensure_process_alive(process)
        remaining = _remaining_time(deadline, fifo_path, "open for reading")
        try:
            fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise
            time.sleep(min(float(poll_interval_s), remaining))

    try:
        chunks = bytearray()
        expected_size: int | None = None
        while True:
            _ensure_process_alive(process)
            remaining = _remaining_time(deadline, fifo_path, "read")
            readable, _, _ = select.select([fd], [], [], min(float(poll_interval_s), remaining))
            if not readable:
                continue
            try:
                chunk = os.read(fd, 1024 * 1024)
            except BlockingIOError:
                continue

            if chunk:
                chunks.extend(chunk)
                if expected_size is None and len(chunks) >= 8:
                    expected_size = int(struct.unpack("!Q", bytes(chunks[:8]))[0])
                if expected_size is not None and len(chunks) >= 8 + int(expected_size):
                    payload_bytes = bytes(chunks[8 : 8 + int(expected_size)])
                    return pickle.loads(payload_bytes)
                continue

            if len(chunks) > 0:
                raise FifoCommunicationError(
                    f"FIFO writer closed before sending a complete framed payload: {fifo_path}"
                )

            time.sleep(min(float(poll_interval_s), remaining))
    finally:
        os.close(fd)


def _ensure_process_alive(process: Any) -> None:
    if process is None:
        return
    return_code = process.poll()
    if return_code is None:
        return
    raise FifoCommunicationError(f"FIFO peer process exited with return code {return_code}.")


def _remaining_time(deadline: float, fifo_path: str, action: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise FifoCommunicationError(f"Timed out waiting to {action} FIFO: {fifo_path}")
    return remaining
