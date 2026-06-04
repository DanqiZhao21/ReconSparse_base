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


def test_fifo_read_returns_with_framed_payload_even_if_writer_fd_stays_open(tmp_path: Path):
    fifo = tmp_path / "framed_pipe"
    os.mkfifo(fifo)
    received = {}
    writer_ready = threading.Event()
    close_writer = threading.Event()

    def reader():
        received["payload"] = read_fifo_payload(fifo, timeout_s=2.0, poll_interval_s=0.01)

    def writer():
        fd = os.open(fifo, os.O_WRONLY)
        writer_ready.set()
        write_fifo_payload(fifo, {"framed": True}, timeout_s=2.0, poll_interval_s=0.01)
        close_writer.wait(timeout=2.0)
        os.close(fd)

    reader_thread = threading.Thread(target=reader)
    writer_thread = threading.Thread(target=writer)
    reader_thread.start()
    writer_thread.start()
    writer_ready.wait(timeout=2.0)
    reader_thread.join(timeout=2.0)
    close_writer.set()
    writer_thread.join(timeout=2.0)

    assert received["payload"] == {"framed": True}
