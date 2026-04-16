from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_reconsimulator_class(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))

    tool_module = types.ModuleType("framework.env_wrapper.tool")
    tool_module.build_sky_view_template = lambda *args, **kwargs: None
    tool_module.get_splat = lambda *args, **kwargs: (None, 0)
    tool_module.get_sky_view = lambda *args, **kwargs: None
    tool_module.get_sky_view_from_template = lambda *args, **kwargs: None
    tool_module.move_to_device = lambda *args, **kwargs: None
    tool_module.slerp = lambda *args, **kwargs: None

    env_wrapper_pkg = types.ModuleType("framework.env_wrapper")
    env_wrapper_pkg.__path__ = []  # type: ignore[attr-defined]

    hugsim_module = types.ModuleType("framework.utils.hugsim_execution")
    hugsim_module.DEFAULT_HUGSIM_REPO = ""
    hugsim_module.load_hugsim_runtime = lambda *args, **kwargs: None
    hugsim_module.resolve_wheelbase = lambda *args, **kwargs: 2.7
    hugsim_module.solve_hugsim_execution = lambda *args, **kwargs: None

    tracker_module = types.ModuleType("framework.utils.tracker_execution")

    class _TrackerExecutionResult:
        pass

    tracker_module.TrackerExecutionResult = _TrackerExecutionResult
    tracker_module.build_execution_result = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "framework.env_wrapper", env_wrapper_pkg)
    monkeypatch.setitem(sys.modules, "framework.env_wrapper.tool", tool_module)
    monkeypatch.setitem(sys.modules, "framework.utils.hugsim_execution", hugsim_module)
    monkeypatch.setitem(sys.modules, "framework.utils.tracker_execution", tracker_module)

    module_path = repo_root / "reconsimulator" / "envs" / "nus.py"
    spec = importlib.util.spec_from_file_location("test_reconsimulator_env_nus", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReconSimulator


def test_tracker_debug_plot_disabled_by_default(tmp_path, monkeypatch) -> None:
    recon_simulator = _load_reconsimulator_class(monkeypatch)
    env = recon_simulator.__new__(recon_simulator)
    env.save_tracker_debug = False
    env.scene = 3
    env._tracker_debug_pending = True
    env._tracker_debug_cleaned = False
    env._tracker_debug_image_index = 0
    env._tracker_debug_last_path = None
    env._tracker_debug_output_dir = lambda: str(tmp_path / "trajTransition-scene003" / "trackerdebug")

    pyplot_module = types.ModuleType("matplotlib.pyplot")

    class _FakeFigure:
        def tight_layout(self) -> None:
            return None

        def savefig(self, path: str, **kwargs) -> None:
            Path(path).write_text("unexpected-debug-plot")

    class _FakeAxes:
        def scatter(self, *args, **kwargs) -> None:
            return None

        def plot(self, *args, **kwargs) -> None:
            return None

        def text(self, *args, **kwargs) -> None:
            return None

        def set_title(self, *args, **kwargs) -> None:
            return None

        def set_xlabel(self, *args, **kwargs) -> None:
            return None

        def set_ylabel(self, *args, **kwargs) -> None:
            return None

        def grid(self, *args, **kwargs) -> None:
            return None

        def axis(self, *args, **kwargs) -> None:
            return None

        def legend(self, *args, **kwargs) -> None:
            return None

    pyplot_module.subplots = lambda *args, **kwargs: (_FakeFigure(), _FakeAxes())
    pyplot_module.close = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "matplotlib", types.ModuleType("matplotlib"))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_module)

    out_dir = Path(env._tracker_debug_output_dir())
    env._save_tracker_debug_plot(
        frame_idx=0,
        plan_local_xyyaw=np.zeros((2, 3), dtype=np.float64),
        tracked_rollout_local_xyyaw=np.zeros((2, 3), dtype=np.float64),
        gt_local_xyyaw=None,
    )

    assert not out_dir.exists()
    assert env._tracker_debug_pending is False
