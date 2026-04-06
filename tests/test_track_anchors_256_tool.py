from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/root/clone/ReconDreamer-RL")
MODULE_PATH = (
    REPO_ROOT
    / "outputs"
    / "visualize"
    / "debug_tracker_scene099"
    / "track_trajectory_anchors_256_with_tracker.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("track_trajectory_anchors_256_with_tracker", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_anchor_plans_from_npy_keeps_batch_shape(tmp_path: Path) -> None:
    module = _load_module()

    npy_path = tmp_path / "anchors.npy"
    anchors = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    np.save(npy_path, anchors)

    loaded = module.load_anchor_plans_from_npy(str(npy_path))

    assert loaded.shape == (2, 3, 3)
    assert loaded.dtype == np.float64
    np.testing.assert_allclose(loaded, anchors.astype(np.float64))


def test_summarize_anchor_tracking_reports_expected_errors() -> None:
    module = _load_module()

    plans = np.array(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.1]],
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.2]],
        ],
        dtype=np.float64,
    )
    tracked = np.array(
        [
            [[1.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.3]],
        ],
        dtype=np.float64,
    )

    per_anchor_rows, aggregate = module.summarize_anchor_tracking(plans, tracked)

    assert len(per_anchor_rows) == 2
    np.testing.assert_allclose(per_anchor_rows[0]["mean_xy_err_m"], 0.5)
    np.testing.assert_allclose(per_anchor_rows[1]["max_xy_err_m"], 1.0)
    np.testing.assert_allclose(per_anchor_rows[0]["final_yaw_err_deg"], np.degrees(0.1))
    np.testing.assert_allclose(aggregate["mean_anchor_mean_xy_err_m"], 0.75)
    np.testing.assert_allclose(aggregate["global_mean_xy_err_m"], 0.75)
