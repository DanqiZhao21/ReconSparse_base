from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path("/root/clone/ReconDreamer-RL")
MODULE_PATH = REPO_ROOT / "framework" / "rollout" / "timing.py"


def _load_module():
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("rollout_timing", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_rollout_timing_summarizes_step_and_reset_metrics() -> None:
    module = _load_module()

    timing = module.build_rollout_timing(
        horizon=32,
        step_records=[
            {
                "obs_tensor_s": 0.02,
                "act_s": 0.20,
                "env_step_s": 0.80,
                "render_s": 0.50,
                "get_sky_view_s": 0.10,
                "camera_setup_s": 0.15,
            },
            {
                "obs_tensor_s": 0.03,
                "act_s": 0.40,
                "env_step_s": 1.00,
                "render_s": 0.60,
                "get_sky_view_s": 0.12,
                "camera_setup_s": 0.18,
                "env_reset_s": 1.20,
            },
        ],
        collect_shard_s=11.0,
        next_value_feature_s=0.30,
        counters={"done_count": 1, "reset_count": 1},
    )

    assert timing["horizon"] == 32
    assert timing["num_steps"] == 2
    assert timing["done_count"] == 1
    assert timing["reset_count"] == 1
    assert timing["collect_shard_s"] == 11.0
    assert timing["next_value_feature_s"] == 0.30
    assert abs(timing["act_avg_s"] - 0.30) < 1e-9
    assert abs(timing["act_total_s"] - 0.60) < 1e-9
    assert abs(timing["env_step_avg_s"] - 0.90) < 1e-9
    assert abs(timing["render_avg_s"] - 0.55) < 1e-9
    assert abs(timing["camera_setup_total_s"] - 0.33) < 1e-9
    assert abs(timing["env_reset_total_s"] - 1.20) < 1e-9


def test_format_rollout_timing_summary_highlights_core_fields() -> None:
    module = _load_module()

    summary = module.format_rollout_timing_summary(
        {
            "collect_shard_s": 12.5,
            "closed_loop_step_avg_s": 0.91,
            "act_avg_s": 0.18,
            "env_step_avg_s": 0.73,
            "render_avg_s": 0.41,
            "env_reset_total_s": 3.0,
            "reset_count": 2,
        }
    )

    assert "collect=12.50s" in summary
    assert "step_avg=0.91s" in summary
    assert "act_avg=0.18s" in summary
    assert "env_avg=0.73s" in summary
    assert "render_avg=0.41s" in summary
    assert "reset=2/3.00s" in summary
