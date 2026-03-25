import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from framework.utils.gsplat_warmup import build_gsplat_warmup_cmd, warmup_gsplat_cuda


def test_build_gsplat_warmup_cmd_uses_target_python():
    cmd = build_gsplat_warmup_cmd('/tmp/custom-python')

    assert cmd[0] == '/tmp/custom-python'
    assert cmd[1] == '-c'
    assert 'ensure_gsplat_backend' in cmd[2]


def test_warmup_gsplat_cuda_invokes_python_subprocess(monkeypatch):
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, 'run', fake_run)

    warmup_gsplat_cuda('/tmp/custom-python')

    assert len(calls) == 1
    assert calls[0][0] == build_gsplat_warmup_cmd('/tmp/custom-python')
    assert calls[0][1] is True
