from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path("/root/clone/ReconDreamer-RL")
MODULE_PATH = REPO_ROOT / "tools" / "analyze_actor_timing.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_actor_timing", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_shards_aggregates_timing_meta(tmp_path: Path) -> None:
    module = _load_module()

    shard_a = tmp_path / "actor0_e0_v1_t1_a.pt"
    shard_b = tmp_path / "actor1_e0_v1_t2_b.pt"
    torch.save(
        {
            "meta": {
                "actor_id": 0,
                "timing": {
                    "collect_shard_s": 32.0,
                    "closed_loop_step_avg_s": 1.0,
                    "act_avg_s": 0.2,
                    "env_step_avg_s": 0.8,
                    "render_avg_s": 0.5,
                },
            }
        },
        shard_a,
    )
    torch.save(
        {
            "meta": {
                "actor_id": 1,
                "timing": {
                    "collect_shard_s": 48.0,
                    "closed_loop_step_avg_s": 1.5,
                    "act_avg_s": 0.3,
                    "env_step_avg_s": 1.2,
                    "render_avg_s": 0.7,
                },
            }
        },
        shard_b,
    )

    rows, summary = module.summarize_shards([str(shard_a), str(shard_b)])

    assert len(rows) == 2
    assert summary["num_shards"] == 2
    assert abs(summary["collect_shard_avg_s"] - 40.0) < 1e-9
    assert abs(summary["closed_loop_step_avg_s"] - 1.25) < 1e-9
    assert abs(summary["act_avg_s"] - 0.25) < 1e-9
    assert abs(summary["render_avg_s"] - 0.60) < 1e-9


def test_format_summary_renders_readable_line() -> None:
    module = _load_module()

    text = module.format_summary(
        {
            "num_shards": 3,
            "collect_shard_avg_s": 40.0,
            "closed_loop_step_avg_s": 1.25,
            "act_avg_s": 0.25,
            "env_step_avg_s": 1.0,
            "render_avg_s": 0.60,
        }
    )

    assert "shards=3" in text
    assert "collect_avg=40.00s" in text
    assert "step_avg=1.25s" in text
    assert "act_avg=0.25s" in text
    assert "env_avg=1.00s" in text
    assert "render_avg=0.60s" in text
