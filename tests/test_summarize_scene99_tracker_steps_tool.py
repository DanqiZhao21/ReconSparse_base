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
    / "summarize_scene99_tracker_steps.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("summarize_scene99_tracker_steps", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_step_rollout_arrays_groups_rows_by_step() -> None:
    module = _load_module()

    rollout_rows = [
        {"step": 0, "point_idx": 0, "plan_local_x": 1.0, "plan_local_y": 0.0, "plan_local_yaw": 0.1, "tracked_local_x": 1.2, "tracked_local_y": 0.0, "tracked_local_yaw": 0.2},
        {"step": 0, "point_idx": 1, "plan_local_x": 2.0, "plan_local_y": 0.0, "plan_local_yaw": 0.2, "tracked_local_x": 2.4, "tracked_local_y": 0.0, "tracked_local_yaw": 0.3},
        {"step": 1, "point_idx": 0, "plan_local_x": 0.0, "plan_local_y": 1.0, "plan_local_yaw": 0.0, "tracked_local_x": 0.0, "tracked_local_y": 0.5, "tracked_local_yaw": 0.0},
    ]

    grouped = module.build_step_rollout_arrays(rollout_rows)

    assert sorted(grouped.keys()) == [0, 1]
    np.testing.assert_allclose(grouped[0]["plan"], np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.2]], dtype=np.float64))
    np.testing.assert_allclose(grouped[1]["tracked"], np.array([[0.0, 0.5, 0.0]], dtype=np.float64))


def test_summarize_step_tracking_combines_first_point_and_rollout_errors() -> None:
    module = _load_module()

    summary_rows = [
        {
            "step": 0,
            "frame_before": 0,
            "frame_after": 5,
            "plan_tracked_xy_err": 0.2,
            "plan_actual_front_xz_err": 0.4,
            "tracked_actual_front_xz_err": 0.0,
            "expert_actual_front_xz_err": 1.0,
        },
        {
            "step": 1,
            "frame_before": 5,
            "frame_after": 10,
            "plan_tracked_xy_err": 0.6,
            "plan_actual_front_xz_err": 0.8,
            "tracked_actual_front_xz_err": 0.0,
            "expert_actual_front_xz_err": 2.0,
        },
    ]
    rollout_by_step = {
        0: {
            "plan": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
            "tracked": np.array([[1.5, 0.0, 0.0], [2.5, 0.0, 0.1]], dtype=np.float64),
        },
        1: {
            "plan": np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
            "tracked": np.array([[0.0, 2.0, 0.2]], dtype=np.float64),
        },
    }

    per_step, aggregate = module.summarize_step_tracking(summary_rows, rollout_by_step)

    assert len(per_step) == 2
    np.testing.assert_allclose(per_step[0]["rollout_mean_xy_err_m"], 0.5)
    np.testing.assert_allclose(per_step[1]["rollout_max_xy_err_m"], 1.0)
    np.testing.assert_allclose(per_step[0]["rollout_mean_abs_yaw_err_deg"], np.degrees(0.05))
    np.testing.assert_allclose(aggregate["num_steps"], 2)
    np.testing.assert_allclose(aggregate["mean_first_point_plan_tracked_xy_err_m"], 0.4)
    np.testing.assert_allclose(aggregate["mean_rollout_mean_xy_err_m"], 0.75)


def test_summarize_step_tracking_keeps_optional_executed_fields() -> None:
    module = _load_module()

    summary_rows = [
        {
            "step": 0,
            "frame_before": 0,
            "frame_after": 5,
            "plan_tracked_xy_err": 0.2,
            "tracked_executed_xy_err": 0.05,
            "plan_actual_front_xz_err": 0.4,
            "tracked_actual_front_xz_err": 0.0,
            "expert_actual_front_xz_err": 1.0,
            "executed_local_x": 1.1,
            "executed_local_y": -0.2,
            "executed_local_yaw": 0.03,
            "actual_local_x": 1.1,
            "actual_local_y": -0.2,
            "actual_local_yaw": 0.03,
        }
    ]
    rollout_by_step = {
        0: {
            "plan": np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            "tracked": np.array([[1.2, 0.0, 0.1]], dtype=np.float64),
        }
    }

    per_step, aggregate = module.summarize_step_tracking(summary_rows, rollout_by_step)

    assert per_step[0]["first_point_tracked_executed_xy_err_m"] == 0.05
    assert per_step[0]["executed_local_x"] == 1.1
    assert per_step[0]["actual_local_yaw"] == 0.03
    assert aggregate["mean_first_point_tracked_executed_xy_err_m"] == 0.05


def test_save_worst_cases_plot_marks_shared_origin_and_point_order(tmp_path: Path) -> None:
    module = _load_module()

    rollout_by_step = {
        3: {
            "plan": np.array([[1.0, 0.0, 0.0], [2.0, 0.1, 0.0], [3.0, 0.2, 0.0]], dtype=np.float64),
            "tracked": np.array([[1.2, 0.0, 0.0], [2.3, 0.2, 0.0], [3.1, 0.1, 0.0]], dtype=np.float64),
        }
    }
    per_step_rows = [
        {
            "step": 3,
            "rollout_mean_xy_err_m": 0.5,
        }
    ]
    out_path = tmp_path / "worst.svg"

    module._save_worst_cases_plot(rollout_by_step, per_step_rows, str(out_path), top_k=1)

    svg = out_path.read_text(encoding="utf-8")
    assert "p0" in svg
    assert "t0" in svg
    assert "shared start" in svg
