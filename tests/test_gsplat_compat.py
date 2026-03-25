import importlib.util
import os
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.utils.gsplat_backend import should_use_legacy_gsplat
from framework.utils.torch_extension import patch_cpp_extension_load


def _load_gsplat_compat_module():
    module_path = ROOT / 'reconsimulator' / 'render' / 'models' / 'gaussians' / 'gsplat_compat.py'
    spec = importlib.util.spec_from_file_location('gsplat_compat_test', module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_patch_cpp_extension_load_creates_build_directory(monkeypatch):
    calls = []

    fake_cpp_extension = types.ModuleType('torch.utils.cpp_extension')

    def fake_load(name, sources, *args, **kwargs):
        calls.append(kwargs.get('build_directory'))
        return 'ok'

    fake_cpp_extension.load = fake_load
    fake_utils = types.ModuleType('torch.utils')
    fake_utils.cpp_extension = fake_cpp_extension
    fake_torch = types.ModuleType('torch')
    fake_torch.utils = fake_utils

    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'torch.utils', fake_utils)
    monkeypatch.setitem(sys.modules, 'torch.utils.cpp_extension', fake_cpp_extension)

    patch_cpp_extension_load()

    build_dir = ROOT / 'tmp_build_dir'
    if build_dir.exists():
        os.rmdir(build_dir)
    fake_cpp_extension.load('x', [], build_directory=str(build_dir))

    assert build_dir.is_dir()
    assert calls == [str(build_dir)]
    os.rmdir(build_dir)


def test_should_use_legacy_gsplat_for_torch_1_13():
    assert should_use_legacy_gsplat('1.13.0+cu116') is True


def test_should_use_legacy_gsplat_for_torch_2_x():
    assert should_use_legacy_gsplat('2.1.0+cu118') is False


def test_gsplat_compat_reexports_legacy_selector():
    mod = _load_gsplat_compat_module()
    assert mod.should_use_legacy_gsplat('1.13.0+cu116') is True
