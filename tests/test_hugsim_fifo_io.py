import os
import time
import threading
from pathlib import Path

from framework.env_wrapper.fifo_io import read_fifo_payload, write_fifo_payload


def test_fifo_payload_round_trip(tmp_path: Path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    received = {}

    def reader():
        received["payload"] = read_fifo_payload(fifo, timeout_s=2.0, poll_interval_s=0.01)

    thread = threading.Thread(target=reader)
    thread.start()
    write_fifo_payload(fifo, {"obs": 1}, timeout_s=2.0, poll_interval_s=0.01)
    thread.join(timeout=2.0)

    assert received["payload"] == {"obs": 1}


def test_fifo_read_waits_for_pipe_creation(tmp_path: Path):
    fifo = tmp_path / "late_pipe"
    received = {}

    def reader():
        received["payload"] = read_fifo_payload(fifo, timeout_s=2.0, poll_interval_s=0.01)

    thread = threading.Thread(target=reader)
    thread.start()
    time.sleep(0.05)
    os.mkfifo(fifo)
    write_fifo_payload(fifo, {"late": True}, timeout_s=2.0, poll_interval_s=0.01)
    thread.join(timeout=2.0)

    assert received["payload"] == {"late": True}
